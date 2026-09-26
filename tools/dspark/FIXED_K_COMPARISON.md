# B256 固定 K5 / K8 对照

本入口只执行 B256 两个组合，不重跑成本采集、confidence 或 B64/B128。
K 是实际 draft 候选数，target 每请求最大 query 长度为 K+1。
实现基于 Plugin `53dbc261db9db69294f647220cb1ae479574cdc2`；
Core 固定 `71d2c1c436eba894a8e9eeb2c5af17e05cb42970`，不修改权重。

## 支持边界与源码审计

原实现不能直接运行 K8：`model_loader._validate_w8a8_runtime_contract`
和 `AscendDSparkSpeculator._validate_markov_inputs` 显式要求 K5。
本补丁只允许显式 `additional_config.dspark_fixed_k8_experiment=true`
且没有 confidence runtime 时使用 K8；默认仍为 K5。其他 K 继续拒绝。
checkpoint `dspark_block_size=5` 检查保留，不把配置改成8冒充训练支持。

- `vllm_ascend/models/deepseek_v4_dspark.py` 的 embedding、三个 MTP 层、
  main projection、LM/Markov/HC head 均没有 K 参数轴。
  `self.block_size=int(config.dspark_block_size)` 没有在 forward 中使用。
  `forward` 按实际 flattened tokens 处理，参数本身不限制行数5。
- `speculator.prepare_proposal_inputs` 使用实际有效 query end 减去拒绝数，
  为每请求构造 K 个位置、anchor 加 K-1 个 noise 输入。
  `_build_query_slot_mappings` 按真实 KV group、页表、物理块长及位置生成 K 个槽位；
  `_build_draft_forward_metadata` 以实际 K 建立 starts、seq_lens、非因果 eager attention。
  没有重复旧候选或从5个候选填充成8个。
- `_execute_sequential_markov_sampling` 按 K 次执行真实 Markov embedding/bias，
  每步使用对应 backbone/logit 行及上一步选出的 token。
  `_build_core_proposal` 继续校验所有步骤、shape、epoch、owner 后原样发布。
  固定模式 `confidence_verification=None`，不调用 confidence head，
  不执行其 D2H、长度分配、查表或专用 TP 通信。head 权重仍随 checkpoint 加载。
- MRV2 `model_runner` 的 `decode_query_len=num_speculative_steps+1`、RoPE 输入、
  展开长度、Core rejection sampler 的 query/candidate 维度均由运行时 K 决定。
  现有 buffer token 上限8192可容纳2304；共享 block tables/slot mapping 的
  **实际分配维度**由容量 RPC 检查。SWA 的安全回收修复保持不变。
- `csrc/attention/sparse_attn_sharedkv_metadata` 从实际 cu_seqlens_q / max_seqlen_q
  建立任务；所审 host tiling 没有 query 长度必须6的限制。加载的 OPP 身份仍须现场保存。
- 不修改 attention/custom op。K8 的实际 kernel、target FULL graph、显存是否足够，
  必须由本次 NPU 运行回答，不能以这些静态检查或 capture 成功代替。

结论：结构与张量路径允许增加候选长度，两个运行时 guard 已作有界扩展。
**K8 是同权重长度外推实验；训练是否覆盖 K8 UNKNOWN，NPU 正确执行 PENDING。**
本地没有重新取得完整模型权重；服务器 prepare 会核对全部权重文件哈希，
并与已冻结 B256 发布记录逐项比较。原始成本表只读保留，
仅引用其权重/环境来源，不加载到 fixed runtime，不声称 K8 成本兼容。

## 精确实验清单

| 顺序 | 最大请求/客户端在途 | 实际 K | draft 满档行 | target 满档行 | capture sizes |
| --- | --- | --- | --- | --- | --- |
| 1 | 256 / 256 | 5 | 1280 | 1536 | 6,12,24,48,96,192,384,768,1536 |
| 2 | 256 / 256 | 8 | 2048 | 2304 | 9,18,36,72,144,288,576,1152,2304 |

每组合独立引擎，1轮同规模预热 + 3轮测量，同引擎内仅重置统计，
不重置 proposal epoch、KV 所有权或 RNG；两组各自从相同 seed=0 初始化。
同64道 GSM8K、同256实例及顺序，保留冻结 token、temperature=0、top_p=1、top_k=-1，
自然 EOS，最多256输出，不延长输出维持并发。每轮使用相同的 round 前缀实例ID。
每个组合每轮均须取得八rank一致的 `256*(K+1)` FULL 实际执行记录，
否则标记容量覆盖不足并停止，不能用 max_num_seqs 代替并发证明。
常规 prefill 不要求 FULL。真实并发自然下降时保留实际 query/capacity 分布。

最多2次模型初始化、8轮、2048请求实例、524288输出tokens。
加载前打印上述清单及预算：host/安装态检查600秒，完整权重身份预检1800秒，
每引擎运行3600秒，监督收尾48秒、最后保护15秒；各阶段和上限9726秒。
整组10000秒，外层终止余量65秒。失败停止，不追加实验或重试。
`dspark-profile-25s-v1` 不变：worker25秒、TERM4秒、共享回收1秒、
EngineCore/前端内层36秒、外层40秒、supervisor48秒。
没有 gdb、SIGUSR1 抓栈、额外退出观察、operator 或逐层探针。

## 指标与门槛

- 生成计时仍是 AsyncLLM 请求提交至全部完成，包括正常 stream、host、TP 和输入准备开销。
  加载/capture、权重/环境预检、phase RPC、文件保存和 shutdown 单列。
  吞吐为实际输出tokens除以生成秒数；保留三轮全量结果，不选择最快轮。
  speedup=三轮 K8 吞吐中位数 / 三轮 K5 吞吐中位数。
- TTFT 为提交到首个非空 DELTA；逐请求 TPOT 为 `(完成-首输出)/(N-1)`，N>1。
  多token chunk 不生成虚构的逐token时间戳。保留每请求输出token、文本、finish_reason、长度。
- Core 已有 SchedulerStats 提供真实验证候选数、接受数和位置向量。
  验证批数按 `num_drafts>0` 的 Core stats delivery 计数，不混用 proposal 发布步数。
  `accepted+request_verifications` 是 EOS 截断前 sampler 推进量；
  同时保存每批 frontend 输出数（可含同批 prefill），缺字段保留 null。
  因而不把 sampler 推进量等同最终交付长度。
- 八rank的轻量 publication 包装器仅读取已成功发布 tensor 的维度、epoch，
  统计实际生成候选数，不读 tensor 数值、不拷贝、不同步。
  FULL 与 publication 各轮有界65536次，溢出失败；关闭时恢复原方法并断开引用。
  原始 NaN/owner 校验保留，不安装重型数值探针。
- 显存 RPC 在各轮开始重置 allocator 峰值、结束读取八rank计数，
  不插入全局同步。保存 allocated/reserved/peak、真实 KV 分配和外层 npu-smi。
  allocator 峰值不包含所有 CANN/HCCL 内存，**整卡运行期峰值仍 UNAVAILABLE**。
  单算子耗时亦 UNAVAILABLE；不能从总时间反推 target kernel 时间。
- 满档候选行增加60%、target行增加50%，不是耗时/显存必然增加相同比例。
  仅全词表 logits 多出的 `768*129280` 元素，若BF16约189.375MiB、若FP32约378.75MiB/rank；
  此为条件性张量尺寸估算，不是实测峰值或运行成功保证。
  不通过降低并发、调低显存配置、缩短输出或回退 eager 获得通过。
- 严格日志、请求完整性、跨rank FULL/publication 证据、八worker真实退出码0，
  无超时、强制升级、取消失败及残留，全部满足才将该组合标为有效。
  输出有差异会单独报告，quality_equivalence 保持 NOT_EVALUATED。

## 一次服务器任务

在已保留 CANN/custom OPP 的原 shell 中执行。用交付完整SHA替换 `PLUGIN_SHA`。
Core remote 显式 `rzwang`，保存实际 URL/HEAD；不会永久重写 origin，
若 URL 为本地 file://，不会声称已从 GitHub 验证外部来源。

```bash
PLUGIN_SHA=<交付的完整SHA>
if bash -c '
  set -euo pipefail
  cd /workspace/vllm-ascend-hust
  test -z "$(git status --porcelain)"
  git fetch origin feat/dspark
  git merge --ff-only "$1"
  test "$(git rev-parse HEAD)" = "$1"
  bash tools/dspark/run_dspark_fixed_k_comparison.sh "$1" \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json rzwang
' _ "$PLUGIN_SHA"; then
  echo "B256 K5/K8 task completed; inspect performance-summary.json"
else
  rc=$?
  echo "Task failed rc=$rc; preserve evidence; do not retry automatically"
fi
```

预检包括无权重的安装态真实 speculator proposal/Markov 路径及 W8A8 loader guard 回归；
未通过不加载模型。两组模型各自启动前检查八卡空闲，保留占用证据。
唯一需回传 `dspark-fixed-k-comparison.*-evidence.tar.gz` 及 SHA256；
其中内嵌模型归档保存 plan、完整权重/OPP身份、原成本表只读副本、冻结输入、
各轮 stream/metrics/worker回执、allocator计数、容量、退出码、日志及 PIPESTATUS。
K8是否正确执行、每步推进量是否增加、吞吐是否提高、显存峰值差异均待此次证据回答。
本地 CPU 测试不能关闭这些 NPU PENDING 项。

## 本地交付验证

Python 3.12.13 / Torch 2.10.0，macOS CPU：以下相关回归143 passed、零失败/零跳过。
覆盖真实源码方法的 CPU Torch 八步递推、最后一步 NaN、epoch 重用拒绝、跨页 slot，
以及配置、实际 FULL/候选回执、包装器释放、统计口径、容量不足、失败即停和退出门槛。
其中小型 head 与 runner 是测试替身，不是完整权重或 NPU kernel 验证。

```bash
python -m pytest --noconftest -q \
  tests/ut/test_dspark_fixed_k.py tests/ut/test_dspark_performance_comparison.py \
  tests/ut/test_dspark_performance_delivery.py tests/ut/test_dspark_graph_replay.py \
  tests/ut/test_dspark_shutdown_policy.py tests/ut/test_dspark_passive_exit.py
```

改动文件的 manual pre-commit 检查通过。隔离工作树中执行了 `bash format.sh ci`；
全仓检查仍失败，78个其他文件会被自动格式化，本次文件无自动改动。
未将这些无关修改带入提交，也不把全仓 CI 报告成通过。
本地未安装 vLLM/torch_npu，三个安装态检查与真实 NPU 两组测量均 PENDING；
它们已列入同一个服务器入口，在权重加载前或正式组合中执行，不另加前置模型运行。
