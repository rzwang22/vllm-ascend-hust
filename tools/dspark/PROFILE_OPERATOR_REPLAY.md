# DSpark Fppa5L9e：SWA 算子因果取证

本轮是**诊断补充，不是生产修复**。根因 UNKNOWN；新增采集/原生算子重放 NPU 验证 PENDING。
Core/custom op、权重、confidence、原 NaN/owner 检查、stream 顺序及退出预算均未修改。
所有产物 performance_eligible=false，不能生成成本表或性能结论。

## 归档证据

已直接读取 `dspark-large-batch.Fppa5L9e-evidence.tar.gz`：168成员、235782231展开bytes，
SHA256=`384013d562464d278a23ee4c584960110593f1ef05cf6a85cae697f1d9d22b43`。
[可重算脚本](verify_fppa5l9e_evidence.py)、[全部八rank证据锚点](PROFILE_Fppa5L9e_AUDIT.json) 入库。
归档内容按数据读取，没有执行归档指令。

| 项目 | 已独立核验 | 证据边界 |
| --- | --- | --- |
| 版本/验证 | Plugin 7ac2c4ae9d671a07e868478adaef9be51633a255；Core 897306c43bf800e2480cb5c0f3e2da408d85a2fd；focused1098 passed | 本地从此干净基线继续；不是本轮新代码的NPU结果 |
| 点完成 | 前9个原始文件SHA与retained一致；第10点输出512/397/122/95 | 第10点未完成 |
| 回执 | 首点八rank execution2/3/4通过；1802/1788、1803/1789、1804/1790的21 target、4 KV、raw/consume全部当前，FULL，无recording_error | 原生kernel内部中间值未观测 |
| 真实布局 | 三请求，starts=[0,1,5,11]，11有效/capacity12；row0=batch10-3-90e8cc44，position222，窗口[95,223)，block119/offset30 | 不能替换成初始四请求或pool行 |
| 局部输入 | Q/KV norm、RoPE、source、写后目标、语义窗口有限；全部有效写后目标等于源，无重复slot/非法窗口索引 | 尚不能证明kernel实际读取范围、SAS调度metadata、所有状态或二进制正确 |
| 首异常 | rank0–5 raw_attention与本地投影row0 NaN；rank6–7这些局部边界有限，但最终attn_output异常 | 不应称为八个独立producer；不知道是哪一个头/原生内部阶段 |
| 写前旧值 | 所有rank前三轮row0写前均NaN，写后均有限；前两轮attention正常 | 已排除“看到写前NaN就判scatter失败”的推论；没有证明旧值被kernel读到 |
| 错误顺序 | 1804 Markov NaN阻止proposal1790发布；1805才缺少owner | 保留首错；没有独立owner缺陷的新证据 |
| 退出 | worker_cleanup_incomplete | 未证明自然退出；与数值首错分别报告 |

本次归档没有可重建调用的真实 Q、KV、sinks 或完整 SAS 数值。
shape、地址、布尔flags和page_runs不能冒充这些值。上一轮窗口异常与本轮窗口有限均有效，
不能通过混合两轮状态推断同一个物理slot或写入者。

## 实际调用链与尚未覆盖的范围

1. [dsa_v1.py](../../vllm_ascend/attention/dsa_v1.py) 的 decode SWA 分支：aux stream 完成
   KV projection/norm/RoPE/scatter；已有 main wait_stream(aux) 后进行窗口探针、原生attention。
   本层compress_ratio<=1，无compressor/indexer；没有发现源码层面可证明的缺失等待。
2. `DeviceOperator.get_dsa_sparse_attn_op()` 返回 `_C_ascend.npu_sparse_attn_sharedkv`。
   实参是q、同一swa_kv_cache、block_table、actual_seq_lengths_query/key、attn_sink、sas_metadata，
   scale、cmp_ratio1、ori_mask4、left127/right0、TND/PA_ND；其他可选输入缺省。
3. [torch_binding.cpp](../../csrc/torch_binding.cpp) 的 `npu_sparse_attn_sharedkv_npu`，
   dispatcher PrivateUse1 → `EXEC_NPU_CMD(aclnnSparseAttnSharedkv, ...)`；stride来自cache.stride(0)，
   返回新建attentionOut/softmaxLse。注册schema与C++函数均已审阅。
4. [host tiling](../../csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp)：
   `GetSASTemplateMode` 在没有cmp KV/indices时选SWA；`SplitBalanced` 设置mBaseSize=gSize，
   一个query的head组单独处理。TND batch来自cu_seqlens_q维度，**包含padding descriptor请求**，
   不能拿实际3请求覆盖12行block table/13项starts/12项seq_lens。
5. [metadata AICPU](../../csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.cpp)
   按序列、窗口和负载生成1024个int32的SAS核分工；kernel从metadata取bN2/gS1/s2起止。
   本次归档没有其逐轮完整值，无法独立验证分工与实际三请求一致。
6. [kernel入口](../../csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp)
   → `SparseAttnSharedkvSwa::ProcessBalance/GetActualSeqLenQ/KV`：TND用starts差分，PA用seq_lens；
   逻辑闭区间为seq-query_len+query_index-left/right。没有证据把整个capacity作为有效query长度。
7. [cube](../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_cube.h)
   的ComputeMm1/2、[DataCopyPA](../../csrc/attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_common.h)
   按logical/block_size查真实页，逐页限制copyRowCnt，尾块按16对齐片上矩阵。
   对本请求语义长度128，页段为1+32+32+32+31；源码并未直接证明从GM多读了位置223。
   片上padding/流水复用或错误tiling仍需原生证据；不能仅凭“0×NaN可能为NaN”判定本故障。
8. [vector](../../csrc/attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_vector.h)
   的ElewiseCompute → SoftmaxFlashV2Compute → BMM2/RowDivs使用float中间值；sink参与分母。
   flags有限不代表幅值/累加不会溢出，也未观测softmax、cube workspace或片上残留。
9. 本地wo_b由[linear.py](../../vllm_ascend/ops/linear.py)的quant method观测；冻结Core
   `vllm/model_executor/layers/linear.py::RowParallelLinear.forward`在局部结果之后调用
   `tensor_model_parallel_all_reduce`。rank6–7先有限、汇总后NaN符合TP传播，非独立producer证明。

**二进制缺口**：source.log仅有torch2.10.0+cpu、torch_npu2.10.0.post2和
`.../_cann_ops_custom/vendors/custom_transformer`路径，没有loaded library/device object哈希或build manifest。
不能断言上述源码就是服务器执行的二进制。新增运行身份证据保存实际loaded路径、扩展/opapi/tiling/
可发现的sharedkv对象及源码哈希（最多256项，截断明示）。哈希标识文件，仍需可重现构建清单才能证明源码映射。
不构建、不替换OPP；若原生小复现证明custom op缺陷，再单独交付kernel补丁与新二进制部署/回归方案。

## 新增算子采集

显式 `--operator-capture`，要求已有 `target-boundaries 1 --attention`。
采集point、max_tokens、max_seq_len从选中诊断前缀最后一个point的id/capacity/context_ceiling取得，
没有硬编码execution、proposal、请求UUID或物理block。本命令对应上限12/640。

- 原生op前复制真实Q、所有Tensor参数（含完整padded starts/seq/table、1024项sas_metadata、sinks），
  保存全部scalar kwargs；op返回后复制原始输出。仍在原attention stream、opaque DSA调用及真实graph内。
- KV复制所选层block table每一行的前`ceil(max_seq_len/block_size)+2`列对应的**完整物理页**。
  本命令22列，覆盖完整前缀、当前页未写尾部和两列额外保护页；也保存padding descriptor行。
  不只截取有效窗口，不清零尾部，不扫描16231页完整cache。guard无有效页号时明确unmapped，
  安全诊断读取不代表该非法地址的原始内容；必须有效的前缀页非法则UNAVAILABLE。
- 保存原dtype/shape/stride/storage_offset/storage_nbytes/base_dtype/NPU format、别名storage标识。
  图内layout_id绑定实际调用描述，前/后各一个epoch回执。reset=-1；全部原有回执门槛保留，
  首点八rank连续三轮另要求恰好一组当前算子回执。零/陈旧flags不得放行。
- 持久buffer在warmup分配，capture期间缺少预分配shape即失败；最多8个shape计划、128个layout/计划，
  持久内存硬上限256MiB。调用前后不保留原模型tensor引用，只有owned副本和描述。
- replay返回后立即owned-pack，在head检查前保存独立文件；每rank保留最新三轮，head非有限时冻结。
  rank6–7也保留，因此可对照异常rank和有限rank。写盘错误记录为诊断错误，原Markov错误仍抛出。
  当前机制覆盖正常返回的attention；原生调用抛异常/图未返回时回执不可用，不能宣称已取得完整胶囊。

新增开销：capacity12、block32、BF16/512时页副本约8.25MiB/调用，加Q/输出/metadata。
cap6/12的常见两计划合计约12.7MiB持久存储/rank（实际字节由bank统计）；计划变体按硬上限约束。
所有<=12的图增加页gather及owned copies，但只在选中point逐轮新增一次bulk D2H及写盘；原一次紧凑
D2H保持不变。packet、临时gather、CPU unpack和3文件会增加内存/IO，可能改变复现时序。
运行期不新增global synchronize、跨stream等待；一份文件内记录D2H bytes与host耗时。
只在首次落盘收集运行二进制身份证据。默认关闭没有这些拷贝/扫描/IO。

## 无模型重放与判读

[operator_replay.py](operator_replay.py)不初始化TP/EP、不加载权重。读取受限tensor数据包，校验回执、
行映射、覆盖上限及重复物理页的逐字节一致性；重复页不一致时报告并发写入/采集不稳定，拒绝作为有效重放。
重建原页号和原table，保留shape、stride、storage offset、底层dtype/大小及可重建别名；不压缩地址。
恢复内存上限2GiB，不支持的重叠view/NPU format会失败。绝对地址、整个target图、旧stream历史与kernel
workspace初值无法重建，不能声称与原模型完全同状态。

未采集页用NaN哨兵标为**UNKNOWN**，不是原值；不以零填充获得PASS。只有源码契约读区间完全在已保存页内，
结果才可称为覆盖输入的局部重放。若metadata/二进制实际越界读未采集页，重放异常不能证明原调用的同一原因。
实际host tiling私有buffer字节和kernel workspace未导出；保存了原始SAS metadata及全部公开形状/属性，
对应host tiling仍需验证加载二进制/必要的CANN调试证据。

默认基线为原saved metadata、局部ACLGraph三次真实replay，复制每轮输出。随后**单独**选择eager，或
独立重建SAS metadata对照，不能静默fallback。reference使用CPU float64，保留SWA/causal、scale、
per-head sink分母；padding行无语义reference。报告有效行NaN位置和误差，不自动称为根因或修复。

```bash
# 无权重、单卡；先从异常rank胶囊开始，再比较有限rank6/7。输出目录必须全新。
timeout --signal=TERM --kill-after=15s 300s python tools/dspark/operator_replay.py \
  /path/to/rank-0-operator-EXECUTION.pt --output /new/operator-saved --mode aclgraph --metadata saved
# 后续有判别力的单项对照：--mode eager；或 --metadata regenerated；CPU只算reference则 --mode reference。
```

新增加的4项原生微型测试先比较eager/ACLGraph及窗口外有限/NaN，使用合成Q/KV和相同3请求/12 descriptor
几何形状，保留完整尾部。若触发同类错误，先停止模型加载并回传原生算子证据；**合成不是原失败调用**。
若未触发，仍需要一次原十点前序执行来取得真实Q/KV与SAS状态；当前归档无法离线补造这些数值。

## 唯一下一次服务器入口

先更新开发分支至交付SHA，然后执行下面父shell命令；严格选项/退出trap仅在子Bash中。
[new runbook script](run_dspark_operator_capture.sh)先做4项无权重原生算子对照，再做原dispatcher/AOT
四参数预检（现在也覆盖胶囊的真实图内复制）。全部通过且零skip才运行完整focused及原单引擎十点，
增加operator-capture，一次收齐八rank首错和前两轮。任一预检失败停止，不加载完整模型。

```bash
# plugin_sha 使用本次交付完整SHA；Core固定897306c43bf800e2480cb5c0f3e2da408d85a2fd。
if bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_operator_capture.sh "$plugin_sha"; then
  printf '运行返回0；分别验收数值/回执/自然退出，不能据此宣称修复。\n'
else
  rc=$?; printf '退出码%s；停止后续实验，保留原始错误和证据。\n' "$rc"
fi
```

命令冻结原manifest/model，创建新compile-check/profile目录，保留PIPESTATUS、JUnit、微型用例输入输出
及runtime身份；原timeout/supervisor/STOP/失败归档保持不变。微型用例各阶段上限300秒。
profile清理失败不会被后续重放覆盖；不在worker可能仍存活时自动启动新NPU任务。
最少回传新compile-check和large-batch的evidence.tar.gz与.sha256；若预检失败仅回传前者。
large-batch应含八份operator-index/runtime、每rank三份.pt、首错JSON、retained及独立cleanup状态。
没有B128/B256、成本表、性能比较；未复现仍标未复现。

## 本地验证

本地Python3.12.13、Torch2.10.0/macOS CPU；归档服务器为Linux Torch2.10.0+cpu、torch_npu2.10.0.post2。
27个host/source focused文件：**626 passed、9 skipped**；最终相关/ABI回归**47 passed、2 skipped**。
CPU覆盖真实dispatcher/AOT、逐轮capsule copy/pack/D2H/落盘、有效行与padding、页尾NaN、sink/reference、
物理页号/stride/offset/别名恢复、陈旧/缺失回执、重复页不一致、首错冻结和写盘异常保留Markov首错。
本地跳过4项真实native SWA eager/ACLGraph、2项真实NPU dispatcher/npugraph_ex/ACLGraph、3项安装态vLLM/Ascend。
这些都必须由服务器执行，不能用CPU替代。

变更文件hooks全通过。隔离工作树执行 `bash format.sh ci` 返回1，8项失败、78个非本轮文件被formatter修改，
与此前基线一致；本轮文件无自动修改，未带入基线格式变更。[机器可读验证结果](PROFILE_OPERATOR_VALIDATION.json)。
