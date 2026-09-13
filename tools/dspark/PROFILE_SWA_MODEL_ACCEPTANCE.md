# SWA 修复：局部验证通过后的完整模型验收

后续 iwviRLAg 完整模型数值/FULL 已通过，cleanup 失败。当前退出修复与命令见
[worker 最终化验收](PROFILE_WORKER_FINALIZATION.md)，下面保留此前入口交付时的历史状态。

## 已读取的归档

`dspark-swa-lifecycle.sGR4YXDq-evidence.tar.gz` 的实际 SHA256 为
`45e7dbe6679aa5192b644d542d0ac16bac67c7c735cd70502cb8cc6800023408`。
核对了顶层 status、十份 PIPESTATUS、JUnit 和三份受限加载的实际 lifetime.pt：

- Core 22 项、存储生命周期 3 项、capsule 23 项全部通过，零失败、零跳过。
- CPU eager、真实 NPU eager 和 NPU ACLGraph 的有效快照计数均为 1→2→3→4，
  BF16 历史槽位均保持 3，实际写入页集合不含该历史页。
- ACLGraph capture 单独记录 not_executed；没有把 capture buffer 当成已执行结果。
- Plugin `116c26ccaca5acbf4ea9c4d6f0d4be462c0a5513`，
  Core `71d2c1c436eba894a8e9eeb2c5af17e05cb42970`。
- 服务器 Python 3.12.13、Torch 2.10.0+cpu、torch_npu 2.10.0.post2。

[审计结果](PROFILE_sGR4YXDq_AUDIT.json) 保留快照哈希、版本与 OPP 身份。
[swa_acceptance.py](swa_acceptance.py) 的 audit-local 使用 weights_only=True、map_location=cpu，
不执行归档文件、不跟随归档内链接。capsule 测试夹具包含故意失败的子实验文件，
其中非零 PIPESTATUS 不属于此次运行的顶层阶段；也不能将夹具的重放记录当成额外真实 NPU 实验。

**局部生命周期验证 PASSED；完整模型 NaN 验收 PENDING；完整模型 worker 自然退出 PENDING。**
本轮仅修改 host 运行入口、精确版本要求和验收报告，没有再次修改 Core 或模型计算。

## 入口与版本

`run_dspark_large_batch.sh` 和 Python 共用的 `run_performance_suite.CORE_SHA`
均要求精确的修复版 `71d2c1c436eba894a8e9eeb2c5af17e05cb42970`。
Python source gate、plan、结果比较与 profile 元数据一致，不会再记录旧 Core 897306…。
旧版本生成的成本产物不能因这次更新获得兼容性豁免。

[一次性完整模型入口](run_dspark_swa_acceptance.sh) 显式接收 Core remote，服务器应传 `rzwang`。
它执行 fetch 所选 remote 的 feat/dspark，确认修复 SHA 可从该分支到达，只允许普通快进到精确 SHA。
不改 origin URL，不 reset、不回退、更不 force push；错误 remote、脏工作区、无法快进都立即失败。
`core-source.json` 保存所选 remote/URL、fetch HEAD、变更前及实际 HEAD、原 origin URL 和失败原因。

归档里的旧 core-fetch.log 显示当时 origin 对应 rzwang22 URL；这不证明现在仍然如此。
当前用户明确指定 origin=vLLM-HUST/vllm-hust、rzwang=rzwang22/vllm-hust，入口按运行时查询记录。

## 本次唯一模型运行

新入口只读并复用已通过的局部归档，不重跑 22/3/23 项；仅运行本次新增 host 入口测试。
保留原 checkpoint/manifest 字节检查、空闲 NPU 门槛与精确 source gate。
随后只创建一个 B64 引擎，按原顺序运行前十点，止于 ctx128-n4-t12-skewed：

- 原 manifest、合成 prompt 方法、指定 lengths、每请求512 tokens，warmup=2、samples=5。
- TP8+EP、K5、target FULL_DECODE_ONLY、draft eager；capture sizes=6/12/24/48/96/192/384。
- max_model_len=8192、max_num_batched_tokens=8192、memory utilization=0.9。
- 保留 target-boundaries 1、现有 attention 数值/回执和请求归属检查，首点八 rank 连续三轮提前验收。
- operator-capture、write-timeline、scheduler page timeline 均关闭，不增加新切点。
- 保留 worker-exit 的既有退出证据；不改变 stream 依赖、模型/算法、任何同步或退出预算。
- 模型运行上限仍3600秒；RPC120秒；内层清理12秒、外层收尾4秒，supervisor 原预算不变。

所有产物 performance_eligible=false；仍走 diagnostic 返回分支，不生成可用成本表。
不串联 B128/B256、完整成本 profile、性能比较或第二次模型初始化。

本地相关回归为241 passed、5项安装态/NPU测试跳过；新增入口16项均通过。
变更文件检查通过，全仓格式检查仍有范围外既有失败。
详见 [验证记录](PROFILE_SWA_MODEL_VALIDATION.json)。本地未启动完整模型。

## 服务器命令

使用交付的完整 Plugin SHA 替换下面 PLUGIN_SHA。严格选项只在子 Bash，父 shell 接收退出码：

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
bash tools/dspark/run_dspark_swa_acceptance.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  rzwang
BASH
then echo '本次完整模型数值、回执及自然退出验收通过'
else rc=$?; echo "退出码=$rc；分别查看数值和清理结果，不判整体通过"
fi
```

必要条件是原模型/CANN/custom OPP 环境、原 manifest、上述局部归档和 `rzwang` remote 均仍可访问。
任何缺项都在加载模型前失败；不改写 remote、不补造输入、不追加诊断。

入口输出新的 SERVER_RESULT_DIR；可在另一终端查看该路径：

```bash
# 将路径替换为本次打印的新目录。
tail -f /workspace/dspark-results/dspark-large-batch.NEW/generation.log
cat /workspace/dspark-results/dspark-large-batch.NEW/runs/b64/point-completion.json
# 受控停止，仅针对本次运行；supervisor 保留原失败并按原预算收尾。
touch /workspace/dspark-results/dspark-large-batch.NEW/STOP
```

## 独立验收口径与回传

运行结束（包括非零退出）后自动写 `model-acceptance.json`：

| 字段 | 含义 |
| --- | --- |
| ten_points_generation_complete | 精确原十点布局，每个实际请求512 tokens、无生成错误 |
| numerical_and_FULL_acceptance | 各点 raw 哈希、八 rank FULL samples、数值观测、请求归属与首点真实回执均通过；只表示本次通过 |
| markov_NaN_in_log / owner_error_in_log | 两类错误分别搜索原始模型日志，缺日志不报“已排除” |
| cleanup / cleanup_failure | 保留超时、强制清理、原始错误和 prior_error |
| workers / worker_force_events | 八 rank 的原始退出码与信号升级；null 保留 unavailable，绝不补写 -9 或0 |
| worker_natural_exit | 清理成功、无超时／强制升级且八 rank 实际退出码全为0 |
| overall_pass | 数值与 FULL 验收、自然退出及运行退出码同时通过 |

生成/数值通过而 cleanup 失败时，数值字段保留 PASSED_THIS_RUN，整体仍失败。
原始阶段非零退出码优先于报告或导出失败；MAIN_RC、generation.pipestatus 和 supervisor 原始退出码都保留。
没有 post-mortem RPC，不以最终无残留替代自然退出证据。

只需回传一个新的 `dspark-large-batch.*-evidence.tar.gz` 和旁边的 SHA256。
该归档包含 source/local-validation、JUnit、PIPESTATUS、原始十点/首错、数值回执及 cleanup；
无需再次回传旧局部验证归档。
