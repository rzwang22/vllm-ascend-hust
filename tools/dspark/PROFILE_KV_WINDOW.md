# DSpark layer 1 KV 窗口：7wU85bLE 审计与局部诊断

本轮为**局部诊断补充，不是生产修复**。已证明所观测的 KV 语义窗口含 NaN，尚未证明
异常逻辑/物理 slot、最后写入者、错误页归属或具体 kernel。Core/custom op、模型数值、
confidence、采样、NaN/owner 检查、stream 等待和退出预算保持不变；root cause=UNKNOWN。
新代码 NPU 验证 PENDING，performance_eligible=false，禁止编译成本表。

## 独立归档核验

[可重算脚本](verify_7wu85ble_evidence.py) 和[全部八 rank 审计锚点](PROFILE_7wU85bLE_AUDIT.json)
均入库。直接读取两个 tar，核对原始 SHA 和内容；没有运行归档代码。
预检中一个 pytest current 绝对符号链接被记录但未跟随/解压，不能借它读取服务器文件。

```bash
python tools/dspark/verify_7wu85ble_evidence.py \
  /path/to/dspark-compile-check.7NnzsCBp-evidence.tar.gz \
  /path/to/dspark-large-batch.7wU85bLE-evidence.tar.gz --output /tmp/kv-audit.json
```

- compile-check SHA=`9742d99889b7ea0b07649aade3d610572590957f2e84a0d4c22c2d1c107a99b8`，
  27 成员、328210 bytes；JUnit 原四参数 **4 passed、零跳过**，PIPESTATUS=0 0。
  这是上一版本真实 CPU/NPU dispatcher、npugraph_ex 和 ACLGraph 微型回归的有效结果。
- large-batch SHA=`e6eef44f121983b79c5ae9a5e9c4b8fcb64fd690495aacb81eafa8918d856e6a`，
  168 成员、229614085 bytes；focused **1089 passed**，PIPESTATUS=0 0。
- versions.log：Python3.12.13/GCC12.3.1、Linux aarch64，Torch2.10.0+cpu、torch_npu2.10.0.post2。
  plan 的 Plugin=`9d4c3c6381cad4e6ba878fd63513a75b0a6e26ce`，Core=
  `897306c43bf800e2480cb5c0f3e2da408d85a2fd`。本地开发从相同 Plugin 干净工作区继续。
- 首点八 rank 的 execution **2/3/4** 连续回执有效：21 项 target 和 raw/consume 全为当前轮。
  前九个 retained 原始 JSON 的 SHA256 全匹配，所有请求输出512、无生成错误。
- 第十点失败。八 rank 的 **1800/1788、1801/1789** 已记录21项有限；
  **1802/1790** 的 `recording_error=null`、FULL、raw_replay_verified=true，21项及
  raw/consume 回执有效。第0行 attn_input、Q/KV normalization、Q/KV RoPE 有限；
  kv_window、raw_attention 和后续输出 NaN。head hidden/logits 对应 candidate0–4 NaN，无 Inf。
- 失败实际顺序为 batch10-3-b5ceb0ef、batch10-2-96aa4ea9、batch10-1-b67892c5，
  starts=[0,1,5,11]，query lengths=[1,4,6]，3请求、11有效行、capacity12；不是初始四请求布局。
  第0行 position220/221/222，当前 slot 为 block105、offset28/29/30；失败语义窗口[95,223)。
- **1802** Markov NaN 阻止新 proposal1790 发布；**1803** 才出现缺少 owner。
  两轮 published epoch=null、owners为空。原数值首错保留；没有独立 owner 生命周期故障的新证据。
- cleanup=worker_cleanup_incomplete、success=false；engine/thread returned 且未超时，
  但归档中的 descendant worker-cleanup 仍是 running。不能以此前返回、空 force_events 或
  某个时刻的残留列表证明自然退出；cleanup-failure.prior_error 保留前端 EngineDead，
  worker 首错文件保留 Markov NaN。退出问题本轮不修改。

## 真实读写链与证据界限

[ops/dsa.py](../../vllm_ascend/ops/dsa.py) 的 `_build_kv_cache` 从本层
`swa_cache_layer.kv_cache` 取实际 cache，经过原 unfold/unpack 传给 AscendDSAImpl。
Core `vllm/v1/worker/gpu/attn_utils.py::build_attn_metadata` 按真实 KV group 配对
block_tables[i]/slot_mappings[i]；plugin metadata builder 再提供 decode 的对应切片。
本轮非 A5、compress_ratio=0 的 SWA 分支只消费 swa metadata；不执行 compressor/indexer。
这条源码链不能证明运行期分配/页复用正确，也不能以 request-state pool 索引替代 token/KV 页索引。

[实际多 stream prolog](../../vllm_ascend/attention/dsa_v1.py) 中，辅助 stream 执行
KV projection → norm → RoPE → scatter；主 stream 沿用已有 `wait_stream(aux_stream)` 后
执行 Q 尾部、窗口探针和 attention。scatter 在非 A5 的
[DeviceOperator](../../vllm_ascend/device/device_op.py) 中直接调用
`_C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, kv)`。
attention 读取同一个 swa_kv_cache/block_table，cu_seqlens_q 和 seqused_kv 来自 SWA metadata，
PA_ND、TND、ori_mask_mode=4、left127/right0。现有源码未显示可证明的缺失等待或错误 slot 算式。

[AttentionProbe.window](../../vllm_ascend/diagnostics/dspark_profile_attention.py) 按 device
starts 查找 request row，position=seq_lens-(query_end-token_row)，对每个有效行读取因果128窗口：
physical page=block_table[request, logical_position//block_size]，offset=position%block_size。
padding 行由 device starts 屏蔽。归约/窗口 state 在图中执行，replay 后立即 owned-copy，
通过已有合并 D2H 保存；不是事后读取复用 cache。它观察的是接口语义读区间，未证明 native
kernel 内部最终读取地址完全相同。invalid_window_indices=0/slot_mismatch=0 分别只证明
本探针检查的范围、当前末端槽位对应关系，不能证明历史页属于该请求或 contents 正确。

旧 `routes[capacity]` 为 Python capture/eager 描述，会被后续同容量调用覆盖。
例如归档 route 的 block_table_shape=[2,256]，而失败真实请求数为3；这**不能直接判成错误**。
当前 state/flags 有 replay 回执，但该静态 shape 没有逐轮设备关联；不可替代失败轮布局。
前两轮窗口有限也不能推出只有新位置222异常：窗口内容或 block table 可能在其间改变。

## 新增的最小证据

继续使用默认关闭的 `target-boundaries 1 --attention`，没有全层/全 cache 扫描。
新增 [KVProbe](../../vllm_ascend/diagnostics/dspark_profile_kv.py)，与原21项 bank 都不共享 storage。
它记录四个真实图内回执：binding、before_scatter、after_scatter、window。
每轮 reset=-1，缺失/陈旧将导致 unavailable 和首点提前门槛失败，不能按零值判有限。

1. **实际绑定**：缓存、block table、slots、starts、seq_lens 的 pointer/storage/shape/stride
   与 layer_name 形成最多32项不可变 catalog；图内写 binding token 和当前 epoch。
   后续 eager 描述无法重标记已重放图。catalog 不持有原 cache/model tensor 引用。
   同时保存匹配的 CPU KV group id/spec，缺少配置时为空并明确不可用；它不构成设备页所有权证明。
2. **写入前后**：在原 scatter 的同一 stream 中分别读取实际目标 slot，对实际 source KV
   记录逐行 NaN/Inf、目标 NaN/Inf、与 source 是否相等、slot合法性和本轮重复slot。
   比较采用数值精确相等、两侧 NaN 视为相等，同时另存 NaN 标志；不是 bitwise identity。
   未完成或被跳过的写入若刚好与旧值相同，仅凭相等仍无法证明发生过物理 store。
   padding/非法 slot 使用诊断安全读取并标 invalid，不修改原算子的参数或行为。
3. **历史窗口**：复用原来已 gather 的窗口 values，增加每个逻辑位置 NaN/Inf 和实际物理 page。
   保存连续 page runs、全部异常 slot、按逻辑顺序的 first_nonfinite 及 offset；不导出 KV 向量。
   异常 slot 与本轮实际写入行交叉关联 request ID/position，空 current_writers 表示没有本轮
   对应写入，**不是已证明不存在历史 writer**。此前两轮对应包也保留，可对照页映射及内容变化。

原 point/rank/execution/proposal、请求重排映射、有效行/padding、graph object 关联及首错保留
沿用原 packet。新增诊断首先回答：

| 新证据 | 可缩小的范围；仍不越界认定 |
| --- | --- |
| source有限，after目标异常/不同 | 当前 scatter、重复/并发写入或目标映射；尚非 kernel 根因证明 |
| after目标匹配有限源，但窗口历史位置异常 | 历史 slot、页归属/映射变化或之后覆盖；需对照前两轮 |
| after目标有限，窗口同一物理slot随后异常 | 两次观测间状态变化或读依赖问题；不能靠同步跑通认定根因 |
| window有限而raw_attention异常 | 继续核对 native读取范围/其他输入；不能归咎 cache |
| 非法索引、缺失回执或非当前binding | 该部分内容不可判定；保留失败证据 |

没有跨所有历史轮次维护 cache 写入日志，也没有记录其他层/请求在更早轮次的完整写入。
如果异常页的最后 writer 不在保留三轮内，其归属继续 UNKNOWN；下一步再依据具体 slot 缩小。

## 开销、测试与服务器入口

新增持久 buffer 为 **1234984 bytes/rank**（最大384行、128窗口），原 attention bank
总量从47024变为1282008 bytes/rank；binding catalog最多32项。典型失败容量12的新增包为
**38632 bytes/rank/replay**，最大容量384为1234984 bytes。四个 tensor 在 replay 后 owned
D2D copy，复用原一次合并 D2H，不新增 host wait、全局同步或跨 stream 等待。
写入前后各新增至多384×512个 KV 元素的读取及统计；窗口不新增 KV gather，增加逐 slot
NaN/Inf归约。BF16/512宽时当前-slot读取临时约0.375MiB；每个窗口布尔临时上限24MiB，
加上 compact integer pack、比较等临时分配，可能增加图内存及改变复现时序。
没有将这些开销当作生产修复。默认关闭没有新统计、拷贝、catalog或同步。

CPU 回归覆盖当前目标异常被正确覆盖、跳过写入、历史 NaN/Inf、页重映射、request ID变化、
重复slot、padding、独立owned包、catalog不被后续eager覆盖、缺失回执提前失败，以及实际
DSA 两种 stream 分支的三轮首错保存。原 dispatcher/AOT 四参数微型测试继续保留，并增加
新KV图内回执断言；NPU两项仍需真实 npugraph_ex/ACLGraph，CPU不可替代。
本地 Torch2.10 CPU/source focused 为609 passed、5 skipped；最终相关回归93 passed、
2个NPU skipped。变更文件 hooks 全通过。隔离工作树执行 `bash format.sh ci` 返回1，
8项失败、78个非本轮文件被 formatter 修改，与此前基线相同；本轮文件没有自动修改，
没有带入这些全仓格式变更。具体执行结果见 [validation](PROFILE_KV_WINDOW_VALIDATION.json)。

唯一必要的下一次运行沿用[两阶段完整 runbook](PROFILE_DYNAMO_COMPATIBILITY.md#服务器分阶段命令)，
将 plugin_sha 设置成本次交付完整 SHA。它先执行原四参数微型测试（现也覆盖新 KV receipts）；
4/4通过且零skip，才运行完整focused及下面这一组，同一引擎保留原前九点和失败点：

```bash
bash tools/dspark/run_dspark_profile_control.sh "$plugin_sha" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  target-boundaries 1 --attention --worker-exit
```

不要直接跳过四项门槛。完整 runbook 已将严格选项放在子 Bash，父 shell接收退出码，保存
PIPESTATUS、JUnit/FX证据；创建新目录，任阶段失败停止；profile supervisor/STOP/退出预算
和自动失败归档保持不变。没有 OPP构建、B128/B256、成本表或性能比较。
最少回传新 compile-check 和 large-batch 各自的 **evidence.tar.gz + .sha256**，包括新增
kv.* packet、首错及前两轮、八份attention-validity、原始point/retained和独立cleanup记录。
若未复现仅报告未复现；自然退出仍须单独证明。
