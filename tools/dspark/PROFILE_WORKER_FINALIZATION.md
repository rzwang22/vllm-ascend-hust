# 完整模型数值通过后的 worker 退出修复验收

## iwviRLAg 原始证据

实际读取 `dspark-large-batch.iwviRLAg-evidence.tar.gz`，SHA256：
`199da908dcd93d22540ac3f2c6cd905a26d2695b94ed0731e1005de9dd984f44`。
运行 Plugin `08ade2ad1f13e0bd2add275dbc369fae198f6102`，
Core `71d2c1c436eba894a8e9eeb2c5af17e05cb42970`。
[独立审计记录](PROFILE_iwviRLAg_AUDIT.json) 保存十份 raw 哈希、全部 rank 的退出时间线、
原始退出码、PIPESTATUS 和 cleanup/supervisor 回执；原归档没有修改。

重新执行正式 `swa_acceptance.model_report(runs/b64, 1)`，输出与归档报告完全一致：
十点共26请求，每请求512 tokens；已启用数值边界未见 NaN，请求归属与 FULL samples
通过，首点八 rank 连续三轮真实回执通过，`PASSED_THIS_RUN`。这是一个 B64 引擎中
实际并发1/2/4的十点结果，不覆盖真实64并发或全部长上下文，也不产生性能/成本结论。

`generation.pipestatus` 和 `acceptance-report.pipestatus` 都为 `1 0`。
cleanup 是 `forced_cleanup`，整体失败；`generation.log` 最后确有
`RESIDUAL_PROCESSES []`，它只证明强制清理后的残留检查为空。

下表均为 UTC（原文件时间）：

| 时间 | 已确认事件 | 不能推出的结论 |
| --- | --- | --- |
| 16:13:25.278 | 前端开始清理，原预算12秒、收尾4秒 | 不是生成首错 |
| 16:13:28.466～28.624 | 八 rank 的181条 steps 无 error，WorkerProc.shutdown 返回 | 函数返回不等于进程结束 |
| 16:13:28.913～28.994 | multiprocessing exit_function、threading shutdown 返回，atexit marker到达 | marker不代表所有 atexit/native finalizer 已执行完 |
| 16:13:29.992～30.194 | rank0/1/2/4 的最终化 GC start（is_finalizing=true） | 在 SIGTERM 前，但没有原生调用栈 |
| 16:13:30.288 | 父进程准备向8 worker发送 SIGTERM | 日志在实际发送之前；不是精确送达时刻 |
| 16:13:31.128～31.734 | rank3/5/6/7 的 GC stop 记录 | 在 SIGTERM 后，不能视为未受信号干扰的耗时对照 |
| 16:13:32.282 | 第二轮 checkpoint；仍活的4 worker Python栈为 Garbage-collecting/no Python frame | 不是 SIGTERM 前的原生挂起证据 |
| 16:13:34.298 | 升级 SIGKILL 四 worker；随后直接保存 worker 回执 | 发送信号不能替代 wait/join 的退出码 |
| 16:13:37.289 | 前端强制清理 EngineCore，随后 EngineDeadError | 不能替代此前的数值通过结论 |

rank0/1/2/4 的原退出码是 -15；rank3/5/6/7 是 null。supervisor 没有额外发信号，
其子进程实际返回1。前端清理线程返回、事件循环关闭，但未自然退出。
早期 destroy_process_group 栈后有 returned 记录，**没有证明通信组销毁死锁**。

## 有依据的最小修改与限制

本轮是 **profile 诊断引用清理和父进程回收修复**，不是再次修改 SWA 或模型数值算法。
worker 原生析构阻塞的具体函数仍为 **UNKNOWN**，自然退出 NPU 验收 **PENDING**。

[IsolatedCostProfiler](../../vllm_ascend/diagnostics/dspark_cost_profile.py) 持有绑定的 draft 方法；
[ProfileObservation](../../vllm_ascend/diagnostics/dspark_profile_observation.py) 的 hooks 持有原方法及模型；
[FullReplayObserver](../../vllm_ascend/diagnostics/dspark_benchmark_worker.py) 是最外层包装器。
原路径从未卸载这些包装器。真实包装函数和 Core shutdown 函数体的 CPU 回归证明：
Core 将 runner.speculator 置空并执行原 GC 后，包装器仍保活 speculator/模拟 draft 资源。
按安装逆序卸载后，同一回归中的资源在原有 shutdown/GC 阶段释放。
**此证明针对引用生命周期，不证明某个 NPU 析构函数导致本次超时。**

新增 close 恢复原实例/类方法，不覆盖之后由其他持有者替换的方法，支持重复关闭。
专用 ProfileNPUWorker 在 dispatch loop 已停止之后先卸载 FULL，再关闭 profiler/observation，
随后调用原 Ascend shutdown。记录两个 close 的开始/完成或异常。
即使 close 抛错仍执行原清理；若原清理也失败，保留其异常及附注，不让清理错误消失。
未启用 profile worker 的执行路径不调用本次 teardown。

没有清空图、cache、捕获 bank，未移动通信组销毁，也没有增加 GC、NPU D2H 或同步。
计时 Event 仍由 runner 上已关闭的 profiler 持有，不在 host 卸载时提前释放。
runner↔graph manager、graph/native storage 和 Core async output daemon 等剩余引用关系
仍按原生命周期处理。现有弱引用记录会显示关闭后的 observation/speculator 存活变化；
**弱引用清除不证明原生析构完成**。
下一次有区别的验证是：消除已证明的诊断保活后，原5秒宽限内是否能自然退出。
若仍失败，不继续凭这个保活关系推断根因；缺少的是未受强制信号影响的原生析构栈及耗时。
本次不添加整层扫描或泛化退出日志，不声称所有晚期 GC 都由诊断导致。

Core `MultiprocExecutor._ensure_worker_termination` 在最后的 kill 后没有等待。
profile executor 现在在返回后对父进程实际拥有的 handles 做 **总计最多1秒** join：
共享同一 deadline，后续 handles 即使预算耗尽也 join(0)，不逐 rank 额外等待1秒。
`worker-cleanup.json.reap` 保存 join 前后真实码、状态、耗时和不可用原因。
回收/记录异常不会覆盖 Core 原始异常；不可用结果仍阻止自然退出验收。
强制退出即使成功回收也仍失败，不将 -9 或0补造给 null。

原预算不变：worker grace5秒、TERM4秒；新增回收最多1秒在前端12秒预算内，
前端收尾4秒、supervisor24秒、模型3600秒仍不变。
剩余约2秒只是预算余量，不保证 EngineCore 自然退出；未退出仍严格失败。

## 本地验证

[回归](../../tests/ut/test_dspark_profile_teardown.py) 覆盖旧引用保活/卸载释放、完整包装顺序、
原实例方法恢复、重复关闭、外部替换、close异常、原清理异常、真实 spawn 子进程自然/kill后回收、
总 deadline 和 unavailable。现有真实 Core death-pipe/MQ/WorkerProc/Ascend shutdown 函数体回归
也覆盖新入口；NPU资源叶子是 mock。
本轮未重复已通过的22/3/23项局部生命周期测试，未连接服务器或运行 NPU。
具体结果见 [本地检查记录](PROFILE_WORKER_FINALIZATION_VALIDATION.json)。

## 唯一服务器运行

使用本次交付 SHA 替换 PLUGIN_SHA。Core 保持精确71d2…，显式选择已有 rzwang remote，
记录来源/实际 HEAD，不永久改写 origin。继续使用原 manifest、单 B64 引擎、原十点、
512 tokens、capture sizes、数值/owner/FULL门槛；operator-capture/write-timeline关闭。
同一入口先做相关 host 退出回归，失败不加载模型，不重跑22/3/23。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
mkdir -p /workspace/dspark-logs
outer_log=$(mktemp /workspace/dspark-logs/dspark-worker-exit.XXXXXXXX.log)
set +e
bash -s -- "$1" <<'RUN' 2>&1 | tee "$outer_log"
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
test "$(git rev-parse HEAD)" = "$1"
bash tools/dspark/run_dspark_swa_acceptance.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json rzwang
RUN
codes=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${codes[*]}" > "$outer_log.pipestatus"
printf 'OUTER_LOG=%s\nOUTER_PIPESTATUS=%s\n' "$outer_log" "${codes[*]}"
rc=${codes[0]}
if test "$rc" -eq 0; then rc=${codes[1]}; fi
exit "$rc"
BASH
then echo '生成、数值/FULL、自然退出整体通过'
else rc=$?; echo "退出码=$rc；保留独立数值与清理结论，整体失败"
fi
```

入口新建 `SERVER_RESULT_DIR`，保留内层真实 PIPESTATUS、JUnit、独立
`model-acceptance.json`、原始错误、cleanup、worker-exit、supervisor 和完整归档/SHA256。
外层日志及其 PIPESTATUS 是单独的 sidecar（不宣称已收入内层归档）。导出失败不覆盖首个非零码。
从另一终端 `tail -f NEW_DIR/generation.log` 查看状态，受控停止用 `touch NEW_DIR/STOP`。
不要修改旧证据目录。最少回传 **新 evidence.tar.gz + SHA256 + 外层 log/pipestatus**。

通过标准必须同时满足：十点26请求各512 tokens；数值与FULL为 PASSED_THIS_RUN且无owner错误；
八 worker 真实退出码全0、回收无 unavailable、无TERM/KILL或超时；EngineCore/前端清理成功、
残留为空、原退出码0、overall_pass=true。生成通过而cleanup失败仍保留生成通过并判整体失败。
自然退出未通过时，先看 close和原 shutdown是否返回、回收结果及现有弱生命周期/栈记录，
不以无残留或一次未复现宣称全部问题已修复。
