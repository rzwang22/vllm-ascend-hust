# SWA 历史页提前回收：09bpL1Zp 审计与修正

本轮包含 **Core 生产修复** 和独立的 **Plugin 采集边界修正**。
真实 NPU／完整模型修复验收仍为 **PENDING**，worker 自然退出未解决。
这份归档四请求均生成 512 tokens，没有记录到 NaN；最终失败来自诊断 coverage，随后强制清理。
不能将本次生成完成作为原 NaN 已修复的证明。

## 可复核的原始证据

归档 `dspark-large-batch.09bpL1Zp-evidence.tar.gz`：
SHA256 `535d1555ee61da4ad0a82859daf2db67a5a760959c044a319cfe2795e7f5054f`。
实际读取 217 个成员，展开 225053033 bytes；不执行归档脚本、不跟随其中的临时路径链接。
运行计划记载 Plugin `890fc350627eba18e8ca1296ad1b4b310283c2ff`、
Core `897306c43bf800e2480cb5c0f3e2da408d85a2fd`，performance_eligible=false。
`focused.log` 为 1161 passed；九个 retained 原始结果哈希吻合。
第十点 last_stream 的四份 output_token_ids 均长 512，生成 error 均为 null。

仓库保留 [独立核验脚本](verify_09bpl1zp_evidence.py) 和
[全部 rank 的证据索引及字节哈希](PROFILE_09bpL1Zp_AUDIT.json)。复核命令：

```bash
python -m tools.dspark.verify_09bpl1zp_evidence \
  /path/to/dspark-large-batch.09bpL1Zp-evidence.tar.gz --output audit.json
```

脚本用 map_location=cpu、weights_only=True 读取 32 份 writes 和 24 份末尾 capsule；
检查 unsafe globals、逐字节差异、实际存储区间和回执。下列 execution、页号、UUID
只用于指向旧证据，未写入生产修复或新运行选择条件。

| 证据锚点 | 独立核验结果 |
| --- | --- |
| scheduler-page-timeline sequence 11788 | schedule 1801，原请求 batch10-3-8dd4972f，group 3 回收参数 total_computed_tokens=223 |
| sequence 11790/11791 | block 35 ref_cnt 1→0，原 group 3 块表 index 2 置 null |
| sequence 11948/11949 | schedule 1803 分配给 batch10-1-b8283095，group 5 块表 index 64=35 |
| rank 0–7 writes 1803/1804/1805 | 当前图内回执有效，原请求仍绑定 logical95/block35/offset31；前两轮该槽位未改变 |
| rank 0–7 writes 1805 | layer 3 compressor 前后实际 981 bytes 不同，receipt=[1805,1805]，BF16 absmax 14.125→1.502027635236955e38 |
| writes-index 1806 | changed_sites=[] 是最新轮，first_change=1805 文件仍在；1806 绑定已移到 logical127 |

1805 的实际三请求顺序为 batch10-3-8dd4972f、batch10-2-9af2b0ef、batch10-1-b8283095；
query starts=[0,1,5,11]，有效 11、capacity 12；受影响历史所属请求的 query position=222。
pool rows 存在 rank 间差异，报告逐 rank 保存，不与 target/candidate 行混用。

## 已证明的责任位置

Core `vllm/v1/core/sched/async_scheduler.py::_update_after_schedule` 将目标计算数与
输出 placeholders 预记账；`scheduler.py::update_from_output` 根据 reject 回退两者，
随后 `_update_request_with_output` 扣除实际采样结果。回收时的 computed 不是已经不可回退的前沿。
实际 schedule 1801 前 computed=223、placeholders=6、num_tokens=218；
再次 schedule 后为 229/12；处理上一轮结果后再回退。记录中的实际参数已核对，未用 worker position 替代。

`vllm/v1/core/kv_cache_manager.py::allocate_slots` 原来把该上界直接交给
`coordinator.remove_skipped_blocks`。SWA 窗口 128、block 32 时：

- 原回收依据 223：floor((223−128+1)/32)=3，释放包含 logical95 的整页。
- 可保守使用的前沿 223−6=217：floor((217−128+1)/32)=2，该页必须保留。

这破坏了“页面重新分配之前，所有在途／reject 后仍可能读取的历史都已越过整页”的生命周期约束。
worker MRV2 `vllm/v1/worker/gpu/model_runner.py::update_requests` 按协议追加新 block IDs，
不会因 scheduler 移除窗口外前缀而重写旧前缀。该协议依赖回收安全；这里原页实际上仍在有效窗口。
清空 worker 映射会丢失仍需要的历史，不能修正此问题。

生产修改只改变 remove_skipped_blocks 的输入：
`max(0, total_computed_tokens - request.num_output_placeholders)`。
此保守前沿与 AsyncScheduler 已用于 cache_blocks 的前沿一致；
分配容量、lookahead、采样和 confidence 逻辑仍使用原上界。
没有关闭回收；确认输出后跨过整页边界仍正常释放并可跨 group/request 复用。
代价是暂时保留在途输出可能需要的历史页；内存紧张时原有容量检查可拒绝本轮分配。
这修正的是所有权安全，尚无性能结论。

Core 提交：`71d2c1c436eba894a8e9eeb2c5af17e05cb42970`，父提交为冻结 Core。
仅 `kv_cache_manager.py` 和针对性 CPU 回归改变；没有 Core custom op 或 ABI 修改、无需重建 OPP。

## 物理写入与 stream 证据的强弱

SWA BF16 [16231,32,1,512] 和 compressor state FP32 [16231,8,1,1024]
共享同一实际 storage，每个物理页 32768 bytes，总存储 531857408 bytes，format=2。
这是 hybrid KV group 共用全局页池的设计；不同 dtype 不是缺陷本身。
每个 rank 的实际地址、布局保存在审计 JSON；rank 0 目标绝对区间为
[20739783457792,20739783458816)。

目标页内偏移 31×1024 = 7×4096 + 768×4，因此同一 1024 bytes 对应
compressor FP32 state 的 row7、features768:1024。
真实 operand 为 state_cache.squeeze(-2)，stride=[8192,1024,1]、offset=0，
覆盖的 storage 与目标完全一致。此次不再仅凭 FP32 位视图猜测 writer：
已同时取得页重新归属、明确写 operand、调用前后字节改变和当轮执行回执。

Plugin `vllm_ascend/attention/dsa_v1.py` 的 decode compressor 调用接收
compressor state block table；`_mla_prolog_multistream` 的 SWA scatter 在 aux stream，
尾部 main_stream.wait_stream(aux_stream) 后才做 Q RoPE 和 attention。
layer 3 compressor 在主路径，其前后快照在该调用 stream 内执行。
`dspark_write_timeline.py::before/after` 保存独立快照；当前 receipt 证明 replay 已执行，
不是 capture 时的 host 读数。catalog stream ID 是 capture 描述，不能冒充新 replay 的 host stream ID。

现有依赖和六处别名候选写入观测把变化定位到 layer 3 compressor 调用区间；
没有原生指令 trace 或完整二进制构建 provenance，不能宣称已排除任意隐藏的并行写入或 kernel 内越界。
本修复无需依赖“compressor kernel 有 bug”的假设：它取得了 scheduler 合法重新分配、
但实际尚未安全回收的页。错误在回收依据，非不同 group 共用存储的设计。

**本轮不构成 NaN 复现**：1805 的 layer 1 先读取有限历史，layer 3 随后改写；
1806 position=223，语义窗口下界96，logical95 已在窗口外。
此前单槽位反事实证明的数值因果性仍保留，但不能将此次 writer 证据无条件回填给所有旧 NaN 运行。

## 独立诊断错误与清理状态

末三份 capsule：1908 的最大 seq_len=635、coverage 有效；1909=641、1910=647，
均超过旧采集上限640。三轮 mapping=true、回执当前。此处有具体 coverage 失败依据，
不是 mapping 错误；旧全局 recording_error 没有冻结最早异常，最早实际失败轮次仍 UNKNOWN，
只能称 1909 为最早**保留**的无效轮。

`capture_runtime_options` 将成本域 ceiling 与运行域区分：640 +
max_concurrent_batches(2) × (K5+1) = 652，且不超过 max_model_len。
两个在途批次可能各包含完整待验证 query，停止输出512并不撤销已调度的执行。
新界限保存到 capsule；成本域仍为640。本配置下每个 plan 多至一个完整页列和相应已有批量拷贝字节，
无新增层内切点、等待次数或全局同步，原总内存上限仍强制执行，默认关闭。
超过新界限依然失败；首次 coverage/mapping 错误及前两轮冻结保存，原 recording_error 不清空。

cleanup.json：forced_cleanup，engine/thread 返回、timed_out=false，但 success=false；
SIGTERM 八个 worker、SIGKILL 三个，null 退出码继续 unavailable。
这是独立退出失败，本轮未修改退出顺序、预算或状态门槛。

## 下一步最小验收

先只运行 [无权重的有界验证脚本](run_dspark_swa_lifecycle.sh)，不再重复单槽位反事实、
不加载模型、不执行十点 profile。脚本将 Core 普通快进至上面新 SHA，保留日志、JUnit、
PIPESTATUS、版本/OPP 身份和 CPU 快照于单个新目录；任一失败立即停止并归档。
每个测试阶段 timeout=300s、TERM 后最多15s KILL；不改变 profile 清理预算。

使用交付的完整 Plugin SHA，在服务器父 shell 运行下面结构；严格选项仅在子 Bash：

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
bash tools/dspark/run_dspark_swa_lifecycle.sh "$1"
BASH
then echo '局部验证完成；原模型 NaN 与自然退出仍待验收'
else rc=$?; echo "局部验证失败，保留日志和归档，退出码=$rc"
fi
```

验收要求：Core 22项、lifetime 3项、capsule 23项全部执行、零失败、零跳过。
lifetime 包括 CPU eager、真实 NPU eager、真实 ACLGraph 三次 replay；
capture 标为未执行，不保存虚假有效快照。
此测试由真实 Core 调度器和分配器产生页号，用合成 FP32 写入检验持久 BF16 历史所有权；
不是生产 compressor 的算子复现或完整 MRV2 模型验证。
CPU/mock 验证范围和 NPU 待验证项见 [验证记录](PROFILE_SWA_LIFECYCLE_VALIDATION.json)。

只需回传新 `dspark-swa-lifecycle.*-evidence.tar.gz` 及 SHA256。
通过这一步后再安排关闭额外写入采集、沿原单引擎十点顺序的完整场景验收；
此文不自动串联该运行，也不生成成本表、吞吐量结论或 B128/B256 比较。
