# DSpark profile：数值未复现与退出失败分开记录

后续 po6aUaSP 运行确认外层正常记录返回、无 timeout 且 loop 关闭，但 worker 仍被强制
清理。阶段审计与默认关闭的退出观测见 [PROFILE_WORKER_SHUTDOWN.md](PROFILE_WORKER_SHUTDOWN.md)。

本轮是 **profile 退出计时与状态记录修复**。没有修改模型计算或新增数值观测。
原 Target NaN 根因仍为 **UNKNOWN**；这次 layer 1 诊断中数值故障**未复现**，
不能称为 NaN 修复成功。本次清理改动的真实服务器复验为 **PENDING**。
Core/custom op、confidence、采样、混合长度和 graph 执行语义保持不变。
所有诊断产物仍为 `performance_eligible=false`，不能编译成可用成本表。

## 独立归档审计

实际读取 `dspark-large-batch.vZQfCb3q-evidence.tar.gz`，SHA256 为
`971aab92e76ec030bfb6d6770b1e61e0099cbfd891e67cc7d2403b92794795ef`，与预期一致。
57 个成员，展开 49542480 bytes；检查成员路径及类型后安全展开，逐文件比对 tar 原始字节。
没有执行归档中的命令。完整逐 rank、逐 point 检查结果及日志行号保存在
[PROFILE_vZQfCb3q_AUDIT.json](PROFILE_vZQfCb3q_AUDIT.json)。

`runs/b64/plan.json` 的 Plugin 是 `f25ac779f5e1d4fbe5f4cccd31bf3519faf1416c`，
Core 是 `897306c43bf800e2480cb5c0f3e2da408d85a2fd`。
开始修改前本地工作区干净，重新 fetch 的 `origin/feat/dspark` 与该 Plugin 一致；
本地 Core 也为该 SHA 且干净。运行是 profile / target-boundaries / target layer 1，
不是 B64 吞吐测试。归档 focused 为 **997 passed, 14 warnings**，属于旧版本服务器测试。

- `retained.json` 包含十点，十个原始 point JSON 的 SHA256 全部匹配。
  所有请求都完成 512 tokens、finish reason 为 length，无 streaming/request error。
  第十点的四个请求均完成，不能套用前次 NaN 发生时的三请求布局。
- 各 point 的八份 rank observation 均启用 numeric，无记录错误或已记录数值异常，
  合并 D2H 次数均完成。第十点每 rank 有 203 次 `both_finite`，Target 覆盖为
  200 次 FULL、3 次明确标记的 NON_FULL_UNOBSERVED；不能把后者称为完整内部数值覆盖。
- 八份 `rank-N-latest.json` 是本次唯一的 worker 数值文件，没有首 NaN 文件。
  最后三轮 execution/proposal 分别为 **1908/1895、1909/1896、1910/1897**，
  各轮有效行与 capacity 均为 6。九个 Target 切点、raw/persistent/consumed 三组
  auxiliary 40/41/42、head hidden/logits 全部有限，无 Inf 或转存差异。
  九个 Target 回执、三份 raw 回执与三份 consume 回执均属于各自 execution；
  `raw_replay_verified=true`，mapping 一致，无 missing boundary/recording error。
  最新文件哈希也与第十点保存的 local evidence 引用相符。

`cleanup.json` 仅记录内层/外层共享的 timeout 8 秒、`shutdown_completed=false`、
`timed_out=true`、`error=null`。`lifecycle.json` 和 `diagnostic.json` 因 cleanup 失败，
supervisor 的 raw return code 与 `status.txt` 的 MAIN_RC 均为 1，signals_sent 为空。
本次没有 Markov NaN → 后续 owner 的事件链；失败发生于十点完成之后。

`runs/b64.log` 的关键行（时间只有秒精度）：

| 行 | 时间 | 原始事件 |
| --- | --- | --- |
| 767–773 | 12:49:56 | MPClient shutdown timeout=8；EngineCore 收到 SIGTERM；executor 等待八个 worker |
| 775 | 12:50:01 | worker 首段宽限耗尽，executor 发送 SIGTERM |
| 776 | 12:50:04 | process manager 强制杀死剩余 EngineCore |
| 777–779 | 12:50:04 | manager stopped、background resource cleanup、MPClient complete |
| 780 | 随后 | Profile engine cleanup did not complete |

该日志证明发生过强制清理，不证明正常退出。旧 JSON 没有 thread return/done 时间，
无法独立确认超时判断和线程返回的微观顺序。用户报告的最终无残留与 supervisor
`signals_sent=[]` 相容，但归档没有独立的残留进程枚举；无残留不能抵消强制退出事实。

## 源码问题与修复

源码锚点：[ProfileFailureGuard](profile_failure.py)、[StreamingEngine](performance_stream.py)、
[collect/run](startup_cost_profile.py)、[supervisor](profile_process_guard.py)、
[profile executor](../../vllm_ascend/diagnostics/dspark_profile_executor.py)。

冻结 Core `vllm/v1/utils.py::shutdown` 先 terminate、等待传入 timeout，随后可以执行
kill_process_tree；`MPClient.shutdown` 接着清理 BackgroundResources。
`AsyncLLM.shutdown` 最后调度 output-handler 取消，资源清理使用
`loop.call_soon_threadsafe`。故函数返回本来就可能晚于传给 manager 的 timeout。
旧外层也只等 8 秒，没有余量；先复制 state、后读取 done 又可能组成不一致结果。
`StreamingEngine.shutdown` 不论清理线程是否结束都关闭 loop，会与迟到的资源回调竞争。

冻结 `MultiprocExecutor._ensure_worker_termination` 默认先等 **5 秒**，SIGTERM 后再等
**4 秒**，仍存活则 SIGKILL。旧内层 8 秒会在这条 5+4 秒路径完成之前截断父 EngineCore；
这与本次日志一致，但没有证明 worker 为何未在最初 5 秒内退出。

现在使用独立、有界预算：

| 范围 | 预算 | 到期行为 |
| --- | --- | --- |
| Core manager 内层 shutdown | 12 秒 | 保留 Core 原有强制清理逻辑；实际升级记录为失败 |
| 外层清理线程 | 12+4=16 秒 | 超时锁定 outer_timeout；不等待无界 destructor |
| event loop task cancellation | 1 秒 | 未结束则失败，保留 loop 交给 supervisor |
| supervisor 首错后宽限 | 24 秒 | 覆盖操作取消 1 + 外层 16 + loop 1 + 收尾余量 6 |
| supervisor SIGTERM / reap | 各 5 秒 | 保留所属 process group 的 SIGKILL/reap 保障 |
| 本次诊断总运行期限 | 3600 秒 | 原有受控失败流程；不继续后续测试 |

预算不是硬实时承诺：OS 调度和文件系统仍可能阻塞，由独立进程 supervisor 提供最终边界。
没有改变 Core worker 的 5+4 秒等待，也没有以增加预算发布 PASS。
即使 12 秒足够让 shutdown 返回，发生 worker escalation 仍为失败。

- 外层以实际线程终止加锁内快照取得一致状态；开始即保存 running receipt。
  记录 UTC 开始/调用/返回/观察时刻与 monotonic 耗时，明确 engine_returned、
  thread_completed、outer timeout、异常、force 和 success 的区别。
- `cleanup-thread.json` 单独记录调用实际返回或异常。迟到回执不会重写已经锁定的
  timeout 或首错。只有线程结束且 loop tasks 收尾完成才关闭事件循环；迟到的
  Core 资源回调有机会先执行。真正卡住时保留 loop，不新增永久等待线程。
- profile executor 包裹其原有 `super().shutdown()`，不替换 termination 算法。
  `worker-cleanup.json` 从持有真实 worker process handles 的父进程记录退出码，
  在发生升级时立即写盘，完成时再写回执。原 worker-exit 首错回执保持不变。
- 共享的 [ShutdownForceObserver](../../vllm_ascend/diagnostics/dspark_cleanup.py)
  只监听冻结 Core 的确切 WARNING 模板、当前 cleanup 线程，每处最多四条。
  不改变 logger 级别或原日志。它是默认 profile 路径已有 executor 的 host-only 记录，
  不向模型增加 hook。WARNING 被禁用/过滤、worker 回执缺失或退出码不可读时，
  按 unavailable/incomplete 处理，不能依据“没看到警告”或无残留宣称成功。
- `shutdown_completed` 只表示库调用/线程完成；`success` 还要求未超时、未强制清理、
  所有 worker 正常退出且回执和 loop 收尾成功。manager 初始 SIGTERM 是其常规停止请求，
  worker 宽限后的 SIGTERM/SIGKILL 和 manager 强杀才是 escalation。
- `point-completion.json` 在 cleanup 前保存已接受点、原始 SHA、生成 token 数和
  已有 CPU numeric 计数的结果；不增加 D2H/RPC 或重写原始证据。
  cleanup-only 失败不会将这十点改为生成失败。
  `cleanup-failure.json`、lifecycle 和 diagnostic 同时保存 cleanup 失败，整体仍非零。
  已有生成首错优先于清理异常，新增清理证据写盘失败也不能覆盖它。

开销仅发生在主机：每个完成点一个小 JSON；清理的开始/返回/最终状态及最多四条
升级事件写盘，临时 logging handler 和一个已有的 cleanup daemon thread。
无新 NPU 运算、同步、tensor 拷贝或层内诊断。非 profile StreamingEngine 退出路径不变。
这不能解决未知的 worker 慢退出原因；下一次回执可能继续报告强制清理，必须如实保留。

## 验证与服务器复验

CPU 回归使用真实 `ProfileFailureGuard`、`StreamingEngine.shutdown`、collect 及
profile executor 包装；提取冻结 Core 的 shutdown/worker termination 函数体，只 mock
进程、时钟或资源叶子。覆盖临界返回、真实线程卡住、迟到回调、loop 取消超时、
SIGTERM/SIGKILL（包括收到 SIGTERM 仍 exitcode=0）、回执缺失、重复清理、清理异常、
清理写盘失败与原始生成首错优先、十点完成与 cleanup 失败分离。
这些不是安装态 Ascend/NPU 退出测试。本地 18 个相关测试文件实测 **507 passed、3 skipped**，
三项跳过均要求安装态 vLLM/Ascend。修改文件的全部 manual hooks 通过。
隔离 worktree 执行完整 `bash format.sh ci` 返回 1：Ruff check/format、codespell、typos、
markdownlint、workflow lint、shell lint、forbidden imports 共八个既有失败项，与此前
layer 1 交付的基线 hook 结果完全相同；78 个自动格式化文件集合也完全相同，本轮文件
均未被全仓检查改写。没有提交这些无关格式化，也不把全仓 CI 报告为通过。
服务器 focused 将运行本轮新增测试；本轮没有连接服务器。

下一次只执行一遍同配置 `target-boundaries 1`。沿用现有 control 入口：同一引擎，
前十点原顺序，TP8+EP、MRV2、K5、target FULL_DECODE_ONLY、draft eager、512 输出，
原模型与合成输入。模型配置/index/confidence head 与 manifest 的既有冻结 gate 保留；
没有完整权重逐字节校验，也不宣称两次运行的历史 KV/随机状态完全相同。

最终交付消息提供已推送的完整 Plugin SHA。将它赋给下面子 Bash 的 `plugin_sha` 后执行；
Core 必须为文首冻结 SHA。严格 shell 选项仅在子 Bash 生效，父 shell 接收退出码：

```bash
if bash <<'BASH'
set -euo pipefail
plugin_sha=REPLACE_WITH_DELIVERED_COMMIT_SHA
plugin=/workspace/vllm-ascend-hust
core=/workspace/vllm-hust
manifest=/workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json
test -z "$(git -C "$plugin" status --porcelain)"
test -z "$(git -C "$core" status --porcelain)"
test "$(git -C "$core" rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd
test "$(git -C "$plugin" branch --show-current)" = feat/dspark
git -C "$plugin" fetch origin feat/dspark
test "$(git -C "$plugin" rev-parse origin/feat/dspark)" = "$plugin_sha"
git -C "$plugin" pull --ff-only origin feat/dspark
test "$(git -C "$plugin" rev-parse HEAD)" = "$plugin_sha"
bash "$plugin/tools/dspark/run_dspark_profile_control.sh" "$plugin_sha" "$manifest" target-boundaries 1
BASH
then
    printf 'DSpark diagnostic completed; inspect separate numeric and cleanup receipts.\n'
else
    DSpark_RC=$?
    printf 'DSpark failed: rc=%s; retain the first error and exported evidence.\n' "$DSpark_RC"
fi
```

入口打印新的 `SERVER_RESULT_DIR`，不覆盖旧归档。另一终端把 `DSpark_RUN` 设置成这次
打印的目录后，可查看 `runs/b64/point-completion.json`、`runs/b64/cleanup.json`、
`runs/b64/worker-cleanup.json`、`runs/b64-supervisor.json` 和 `generation.log`；
文件尚未出现表示阶段尚未到达。受控停止只用 `touch "$DSpark_RUN/STOP"`，等待既有
supervisor 清理所属 group，勿按进程名杀其他作业。

入口自动生成 `$DSpark_RUN-evidence.tar.gz` 及 `.sha256`，并保存 `status.txt`、
各阶段 PIPESTATUS；任一步失败停止后续阶段。导出失败保留原始 MAIN_RC，不能覆盖首错。
若导出本身失败，仅重新打包已有目录，不重跑模型：

```bash
if bash -c 'set -euo pipefail; tar -czf "$1-evidence.retry.tar.gz" -C "$(dirname "$1")" "$(basename "$1")"; sha256sum "$1-evidence.retry.tar.gz" > "$1-evidence.retry.sha256"' bash "$DSpark_RUN"; then
    printf 'Evidence export completed; original status.txt remains authoritative.\n'
else
    DSpark_EXPORT_RC=$?
    printf 'Evidence export failed: rc=%s; retain original run directory and MAIN_RC.\n' "$DSpark_EXPORT_RC"
fi
```

最少回传归档及对应 SHA256 文件。包内需保留十点原始 JSON/retained、八 rank 数值文件、
point-completion、cleanup、cleanup-thread（若线程确实返回）、worker-cleanup、
cleanup-failure（若失败）、lifecycle、diagnostic、supervisor、日志和 status/PIPESTATUS。

验收分别判断：十点原始哈希和请求完成情况；数值边界及当前回执是否仍有效；cleanup 是否
`success=true`、未超时/升级、线程已结束、loop 已关闭、worker 全部正常退出；supervisor
无补救信号且 MAIN_RC=0。任何不可用字段都不能判正常清理。数值未复现仍仅记未复现；
强制清理或超时即使无残留也保留非零。无论结果如何均不启动 B128/B256、成本表或性能比较。
