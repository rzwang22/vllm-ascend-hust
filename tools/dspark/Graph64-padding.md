# DSpark MRV2 Graph64 请求 padding 修复

状态：**SERVER_NOT_REVALIDATED**。本地复现并修复了维度错误，尚未验证高并发 NPU graph 或新提交的性能。

## 证据核验与根因

双仓初始 HEAD 分别为 plugin `7e23a859defa5a12b3e583fcf4ce57a52da94c71`、
core `897306c43bf800e2480cb5c0f3e2da408d85a2fd`，分支 feat/dspark，工作树干净。
归档安全检查拒绝绝对路径、`..`、链接和特殊文件，解包在仓库外；完整读取 Graph64 失败日志。

- 归档 SHA256：`7774d0e534a88c265185d227201141f235a0d1a65700dc18d4d868cf3d5efb7a`。
- 400 条实际输入 SHA256：`3889b4bda22442e69062cd5c3888090515b67a303006ef59afea8d72424258b2`，与 plan 一致。
- plan 与 command 的模型、revision、采样、EOS、warmup、显存比例和 Graph64 参数一致。
  s32 两份结果也记录相同 plugin/core SHA。
- 附带 reference.json 的 SHA 为 `f3f361f2922d59a704f39f8c1a61faef008e4e10d308ba33e398dfcb6bfc1576`，
  与 plan 所记 `455ed73ff5305a8df49fa5b33b361af65d41c1b18e2f89fb3397a5a4c4782f65` 不符。
  不宣称旧 reference 字节核验通过；复测只使用已独立核验的实际 input-400.jsonl。
- scheduler dump：63 个请求，各 6 tokens，共 378；无新请求。
  首错是 metadata 赋值 `64 <- 63`，不是 OOM，EngineDeadError 为后续级联。
- 既有 s32 结果为 eager 219.7294 tok/s、graph 283.5494 tok/s、1.29045 倍；
  正式 FULL replay 281 次、最大 shape 192。这些是旧提交结果，不是本次 Graph64 验收。

调用链：core `GPUModelRunner.execute_model` → plugin `AscendModelState.prepare_attn` →
`worker/v2/attn_utils.build_attn_metadata` → `AscendDSAMetadataBuilder.build` → `build_decode_metadata`。

core FULL descriptor 将 378 tokens 匹配到 384-token 档位，即 64 个 6-token 请求行。
旧 runner 的 GPU seq_lens/query_start_loc views 仍是实际长度 63/64，CPU query offsets 却是 padded 长度 65。
model state 使用 padded num_reqs=64；DSA 从 CPU offsets 得出 num_decodes=64，
从短 GPU views 算出的 start_pos_decode 只有 63 行，于是赋值失败。
此外，旧 `_pad_query_start_loc_for_fia` 丢弃 descriptor.num_reqs，
例如 49→64 时会把 15 个 dummy query 合成一个 90-token query，错误分到 prefill。

## 字段与消费契约

| 字段 | 本次 63→64 | 语义及更新 |
|---|---|---|
| input_batch.num_reqs / DSA num_reqs_actual | 63 | 本轮真实请求数；原来错误传成 padded 数 |
| num_reqs_after_padding / common.num_reqs / num_decodes | 64 | FULL 图布局，不改为 63 |
| num_tokens / common.num_actual_tokens / num_decode_tokens | 378 | 本轮真实 verification tokens，供 preflight 等使用 |
| num_tokens_after_padding / num_input_tokens | 384 | 图输入长度 |
| 原 input_batch.seq_lens / query_start_loc | 63 / 64 行 | 采样、proposal、slot 计算继续用实际 views |
| attention seq_lens / query_start_loc | 64 / 65 行 | 新增持久输入 buffer 的 padded views，不复制数据 |
| query lengths | 64 个 6 | 真实请求值不变，dummy 行按 descriptor 的 uniform layout 补齐 |
| CPU/GPU seq_lens 尾行 | 0 | GPU 原 kernel 已清零；CPU 修正为从 actual 行开始清零 |
| start_pos_decode | 真实 seq_len−6；dummy=0 | DSA 原 in-place 更新和 dummy 清零逻辑，地址不变 |
| block table | 64 行；dummy=0 | core gather 与 DSA 原清零逻辑；各组保留真实 block size |
| SWA slot 持久 buffer | 384 行；末 6 行 `[-1,31]` | 原 raw PAD_SLOT_ID=-1 与 block=32 格式化契约 |

本轮返回的 decode slot view 为实际 378 行；captured slot view 为 384 行，底层 384 行均原位更新。
SAS/QLI 使用 padded query/seq vectors 与 64 请求布局；其持久输出 buffer 原位更新。
同一轮的各 KV group 仍共享三组 batch-local dictionaries；不同轮次重新创建，不复用旧请求长度。

实际 graph consumer 捕获的是持久 tensor 地址：SWA 的 `seqused_kv`、query offsets、
slot、block table、start_pos 都从本轮更新后的 storage 读取。
SWA scatter 的精确 -1 sentinel 不写 KV；SWA kernel 的零 seq_len 不读取有效请求窗口。
压缩算子的 scalar 请求数在 graph 内仍是 capture 档位数，不能假称它在 replay 变成了 63。
它可能处理 dummy query，但 start_pos=0、该请求所有 cache/state block table=0，
仅访问 core BlockPool 构造时移出 free queue 的保留 null block 0；不会访问上一轮真实请求页。
沿用已有 null/padding 契约，不把任意非法负值 clamp 到 block 0，也不改变 custom op。

生产修改仅涉及 InputBatch 的两个可选 views、runner padding、model state 的 attention view 选择、
attn_utils 的实际请求数转发。R9C make_dummy 浅层字段传递保留。
没有新增逐步 synchronize、D2H、device tensor 分配或复制；沿用已有 host metadata、设备清零和 SAS 更新。
既有 `_update_seq_lens_cpu` 同步及 R5 preflight 未增加频率。

## 本地验证

- 以基线函数执行真实 runner/model state/DSA build/decode 函数体，在 CPU Torch 复现相同 `64 <- 63` 报错。
- 新回归执行上述真实函数体，替换 NPU leaf：64→63→49→64，K=5，ratio=1/4/128，
  检查真实值、dummy 清零、metadata 共享、SAS 更新和持久地址；actual=padded 与 eager 也覆盖。
- 使用原 C++ scatter 地址选择器和 CPU 写入检查 dummy 不改写 KV；保留真实 validator 的非法索引测试。
  压缩路径复用现有 CPU reference 检查 captured 请求数下的 dummy 仅指向 null/sentinel。
- 脚本回归覆盖输入/完整命令核验、每 rank measured replay、warmup 不能代替 measured、
  缺 rank、失败执行、缺输出、NaN、错误配置、自然 EOS 差异与 PIPESTATUS/失败产物保留。
- 本地：128 个 metadata/脚本/source/CPU 回归通过，202 个 benchmark/R9 回归通过。
  97 个完整 runtime 变体未选入该 CPU 命令；新增 4 个 import 变体另行尝试，因缺 vllm/Ascend 跳过。
  Torch/NPU 全栈、custom kernel 执行和服务器性能未执行。

## 服务器同步与复测

在已经设置好的 CANN/custom OPP 环境下执行，脚本不重新 source 环境、不 build、不 reset、不 kill：

```bash
cd /workspace/vllm-ascend-hust &&
git fetch origin feat/dspark &&
git merge --ff-only <本轮完整_PLUGIN_SHA> &&
bash tools/dspark/run_dspark_graph64.sh <本轮完整_PLUGIN_SHA> \
  /workspace/dspark-results/p08-r9c-400request-sweep._fb6g2_4
```

脚本检查精确 plugin/core SHA、分支、干净状态与 import 路径，锁定归档实际输入和原 command 全部参数。
TP8+EP、K=5、max_num_seqs=64、max_model_len/max_num_batched_tokens=8192、block=32、显存 0.9，
400 measured/1 warmup、最多 256 tokens、自然 EOS、seed=0、temperature=0、top_p=1、top_k=-1；
capture=[6,12,18,24,30,36,42,48,96,192,288,384]，target FULL_DECODE_ONLY/draft eager，安全 RPC 序列化=0。

流程为 focused → Graph64 → 完成/错误/残留/replay 验收 → eager64 → 比较。
新目录保留每阶段日志、退出码、PIPESTATUS、完整 result/command/input 和 evidence 归档；失败不自动继续。
验收全部 400 输出记录完整、每 rank 正式 FULL replay>0，TP8 不相加；保存实际 padded/unpadded shape。
不要求每轮 64 并发，不声称 100% graph coverage；fallback 不可观测则保留 unavailable。
对同 SHA 同配置的实际 tok/s、elapsed、output tokens、接受长度和 speedup 作汇总；
跨 mode token hash/自然 EOS 长度差异仅报告。只运行这一对，不扩展到更高并发，诊断保持关闭。
最终通过状态以 gate.txt 和脚本退出码为准；summary.json 是已取得的计时数据，不替代最终残留检查。
