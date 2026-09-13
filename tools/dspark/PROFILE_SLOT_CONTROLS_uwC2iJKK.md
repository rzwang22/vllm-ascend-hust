# uwC2iJKK：历史槽位的原生重放与双向反事实

本轮是**诊断补充**，没有生产修复。原生单卡重放已在服务器归档中复现；
四组单槽位 NPU 因果对照 PENDING；实际写入者 UNKNOWN。完整模型的 worker 自然退出
没有在这个无权重实验中验证。Core/custom op、模型、confidence、模型内观测、退出预算均不变。

## 实际读取的证据

实际读取 `dspark-saved-operator.uwC2iJKK-evidence.tar.gz`，SHA256
`48bd6fb4e1048c9cf3f637eaae6090668698d7a6c9fdd5147493b2b1dc78f580`，
59 个成员、展开 22214066 bytes。Plugin 为 `3b638d07062014c61a19af28f886d365f32ca4f4`，
Core 为 `897306c43bf800e2480cb5c0f3e2da408d85a2fd`。归档作为数据读取。

[可重算审计](verify_uwc2ijkk_evidence.py) 与 [机器记录](PROFILE_uwC2iJKK_AUDIT.json)
核验实际 `replay.pt`、`reference.pt`、`result.json`、两个输入哈希及全部 14 份 PIPESTATUS
（六次重放、六次输入完整性检查、输入和 runtime 检查），均为 `0 0`。
受限加载 `weights_only=True, map_location="cpu"` 成功，unsafe globals 为空。

| 实际运行 | 1802，三次有效输出 | 1803，三次有效输出 |
| --- | --- | --- |
| saved metadata + ACLGraph | 有限，与原 capsule 逐字节一致 | row0 heads1/3/7 NaN，与原 capsule 逐字节一致 |
| saved metadata + eager | 同上 | 同上 |
| regenerated metadata + ACLGraph | 同上 | 同上 |

实际 regenerated metadata 字节与 saved 不同；saved 字节不变。六组 runtime 记录的 27 个共同
extension/OPP artifact 哈希与采集时相同，包括扩展和 `libcust_opapi.so`。采集时另有两个
内置 aicpu engine SO 未出现在重放进程中（完整路径保存在 JSON），不能表述成所有加载库完全相同。
返回 0 表示实验完成，不表示数值通过。已捕获输入足以重现这个输出异常；没有证明全部 graph、
metadata、越界或完整模型异步状态问题已排除。未采集的物理区域仍为 UNKNOWN。

## 单槽位证据与源码审计

沿用 [上一轮全24份 capsule 审计](PROFILE_SAVED_OPERATOR_ZvqiDthD.md)：
1802→1803 重叠有效历史中 position95 对应 block123/offset31 的512个 BF16分量全部改变；
当前 position222 的 scatter 目标是 **block194/offset30**。保持 target/candidate/pool 行区别。
所选槽位是1024 bytes；相对 cache storage 的半开字节范围 `[4062208,4063232)`；
保存的 pages payload 范围 `[97280,98304)`。实际改变977个字节，精确范围记录在 intervention 中。

把异常槽位的完整1024字节与原始24份 capsule 的所有已保存 tensor payload 作精确匹配，
只匹配到1803各rank的该槽位，以及部分下游输出 head；没有匹配 Q、sinks、metadata 或此前轮次。
输出 head 是消费者结果，不能当写入者。完整匹配位置保存在 JSON。
FP32位视图最大幅值约2.9663只是线索；这些文件没有其他层/压缩器/workspace 的实际源内容。

本次检查的真实源码路径（均为上述冻结基线，无 Core 修改）：

- `vllm_ascend/models/deepseek_v4.py` 的 `AscendDeepseekV4SWACache.get_kv_cache_spec`
  使用 BF16、window128、storage block32；`AscendCompressorStateCache` 使用 FP32。
  dtype不同本身不能证明共享地址或错误写入。
- `vllm_ascend/core/kv_cache_interface.py` 将该 SWA spec 注册到 Core `SlidingWindowManager`。
  Core `vllm/v1/core/kv_cache_manager.py:allocate_slots` 在分配前调用 `remove_skipped_blocks`；
  `single_type_kv_cache_manager.py:SlidingWindowManager.get_num_skipped_tokens` 按
  `num_computed_tokens - sliding_window + 1` 计算完整可丢弃块。
  若实际 computed=222，则95个跳过token仅释放前2个完整块，**公式本身不会释放包含position95的第2号逻辑块**。
  Core scheduler 有提前记账和 reject 回退；归档缺少当时实际调度参数、free/allocate事件及页拥有者，
  不能把 worker position 直接当 Core 的实际 computed，也不能据此修复 Core 边界。
- `worker/v2/attn_utils.py:_adjust_dsv4_kv_layout/_view_dsv4_cache/_validate_dsv4_packed_layout`
  按元素大小换算 stride/offset，检查页内范围和 packed 区间不重叠；`_allocate_kv_cache`
  区分显式共享、packed backing 与独立 byte allocation。静态 layer1 descriptor 与已保存地址中
  未发现另一已记录对象覆盖所选地址；其他 group/workspace 的完整运行时区间未被采集，不能排除重叠。
- `attention/dsa_v1.py` 的 `compress_ratio <= 1` 分支只走 layer1 SWA，无该层自己的
  compressor/indexer。`device/device_op.py:dsa_kv_compress_scatter` 的非A5路径调用
  `_C_ascend.npu_scatter_nd_update_v2`；`csrc/torch_binding.cpp` 传真实 cache strides。
  `csrc/moe/scatter_nd_update_v2/op_host/op_api/aclnn_scatter_nd_update_v2.cpp`
  要求 updates 与 varRef dtype 一致，不能凭FP32位视图认定该 scatter直接写FP32。
  `op_kernel/scatter_nd_update_linear_index.h` 以 index×stride 计算线性位置，nonsort kernel
  按分区范围写 `tileLength * sizeof(T)`；此静态契约审计不替代实际地址/长度/全部写入回执。
- `diagnostics/dspark_profile_operator.py:OperatorCapture.before` 把选定物理页复制到独立持久
  snapshot，原 attention 仍读取原cache；capsule只有调用时快照，缺少1802采集到1803采集间的写入事件。
  稳定指针、图对象和回执并不证明所有资源生命周期正确。

因此尚未证明具体 writer、释放复用错误或 dtype 写入错误，不修改生产代码。

## 四组对照与局部写观察

`operator_replay.py` 新增可选 `--slot-source/--slot-block/--slot-offset`，通过
[单槽位工具](operator_slot_controls.py) 生成独立 `counterfactual.pt`，原文件只读。
所有同一物理页的重复副本一起更新；先后 validate 均检查重复页字节一致性、原回执和覆盖。
保留所有其他 tensor、标量、metadata、格式、布局和输出对照。原 epoch 是原采集身份，
不会伪造为反事实的模型观测；原 output 仍明确标为原采集输出。

| 对照 | 只改变的内容 | 本地 CPU FP32 NaN heads | 原生 NPU |
| --- | --- | --- | --- |
| original-1802 | 无 | 无 | 新四组运行 PENDING；旧归档已有限 |
| original-1803 | 无 | row0:1,3,7 | 新四组运行 PENDING；旧归档已异常 |
| 1803-from-1802 | 所选1024字节恢复1802值 | 无 | PENDING |
| 1802-from-1803 | 所选1024字节换成1803值 | row0:1,3,4,6 | PENDING |

反向对照的 Q 等输入仍属于1802，不能预设必须产生与1803相同的head集合。
CPU FP32是算术反事实，float64 sink-aware reference另行保留；两者均不冒充原生kernel验证。

后续 qe7Lb9dE 预检暴露了旧版本把capture当执行的问题，见
[快照时序修复](PROFILE_WATCH_CAPTURE.md)。以下为修正后的验收约定：
仅在无权重 replay 入口增加默认关闭的 `--watch-slot`：warmup、三次显式replay在
实际 native attention 调用前后复制该1024字节槽位，保存四对拥有独立CPU存储的快照及字节差异。
capture阶段只记 `captured_not_replayed`，`snapshot_valid=false`，不读取尚未执行的clone。
图内前后clone每轮replay执行，CPU数据在下一轮buffer复用前取得所有权。
每个完成阶段一次2KB D2H，共4次/8KB每组；图内两份1KB缓冲，栈叠临时2KB，保存CPU数据8KB。
新增clone/局部读取和D2H等待可能影响局部时序，四组使用同样观察设置，
并与已归档无watch基线比较。保持已有stream顺序，无新增全局同步。

该观察只回答**所选槽位是否在单算子的warmup/显式replay调用中改变**。
不能由“未改变”判定完整模型中没有其他writer；若改变，也需复核前后读顺序和实际算子调用后再归因。
先用真实NPU小型合成用例逐阶段更新独立guard槽位，证明图内快照确实随每轮变化且不被复用覆盖；
预检零跳过后才运行四组真实数据。它不加载权重，也不把合成输入当原始故障复现。

## 本地验证和下一步判定

本地 Python3.12/Torch2.10，无torch_npu。相关测试56 passed、7 skipped：4个原生SWA、
2个dispatcher NPU参数及1个新watch NPU用例缺少环境；两个真实CPU dispatcher/AOT参数通过。
测试覆盖重复页同步替换、精确字节范围、原件不变、双向恢复、受限加载/正式CLI、坏布局/回执拒绝、
CPU快照所有权、driver失败即停/退出码/归档；新NPU用例必须实际ACLGraph多次replay，服务器待验。

[本地验证记录](PROFILE_SLOT_VALIDATION.json)：修改文件的manual hooks全部通过。按规范在隔离worktree
执行全仓 `bash format.sh ci`，返回1；8个失败hook及78个被自动格式化的无关文件与前轮基线一致，
没有自动修改本次文件；没有把这些无关改动带入工作区，不能报告全仓CI通过。

验收顺序：预检1 passed零跳过；四组各3份输出和4对watch快照齐全，实际输入哈希及原格式/metadata
保留，逐行/head NaN/Inf、有限幅值和误差写入result，OPP身份一致。
预期用于检验“单槽位变化足以触发/消除异常”与“还需其他状态”，以及“该算子自身改变该槽位”
与“完整模型中其他调用/资源更新改变该槽位”。不把预期写成强制PASS条件，也不吞掉数值异常。

若NPU双向因果成立且watch未改变，下一项唯一优先证据是原模型1802采集后到1803采集前的
**该物理槽位写入时间线**：同一实际storage区间、各KV group/压缩状态/workspace的运行时交集，
围绕可能写入的调用前后保存1024字节、源dtype/stride/offset、execution/stream，以及实际页free/allocate
与request owner。只要多个stream存在并行写，边界夹出变化也不直接证明具体kernel，仍需依赖证据。
现有capsule没有这些源内容和生命周期记录，无法离线恢复；本轮先完成单卡对照，不要求重载完整模型。

## 服务器：一次无权重四组运行

先更新到交付的完整SHA（`PLUGIN_SHA`），沿用已有激活环境与OPP；不要构建OPP。
严格选项只在子Bash，父shell接收退出码。以下只启动单卡算子进程，不启动模型或成本表：

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
exec bash tools/dspark/run_dspark_saved_operator.sh "$1" --slot-controls
BASH
then
  echo 'Single-slot experiment completed; inspect numerical results separately.'
else
  rc=$?
  echo "Single-slot experiment failed: rc=$rc; preserve the printed evidence directory."
fi
```

driver冻结Core SHA，校验原两份capsule哈希和OPP身份，创建全新
`/workspace/dspark-results/dspark-slot-controls.XXXXXXXX`。runtime最多120秒；watch预检、每组
各180秒，TERM后15秒KILL；任一命令失败停止后续组。保留日志、JUnit、PIPESTATUS、真实退出码，
EXIT归档失败不覆盖原退出码。返回新 `dspark-slot-controls.*-evidence.tar.gz` 和 `.sha256` 即可。
原始归档/模型结果目录不改。需要受控停止时只停止这个driver对应的timeout/子进程，保留打印目录；
不要对所有Python进程执行pkill。完整模型worker自然退出和原生产修复继续独立保持未关闭。

本地重算审计（无需NPU）：

```bash
python -m tools.dspark.verify_uwc2ijkk_evidence \
  /path/to/dspark-saved-operator.uwC2iJKK-evidence.tar.gz \
  --capture-archive /path/to/dspark-large-batch.ZvqiDthD-evidence.tar.gz \
  --output /new/path/audit.json
```
