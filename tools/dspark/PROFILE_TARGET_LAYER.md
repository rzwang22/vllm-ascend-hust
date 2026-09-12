# DSpark：从 layer 0 → layer 1 区间继续定位

本轮交付是**诊断补充**。已有 NPU 故障复现和定位区间已核验；具体根因仍为
**UNKNOWN**，新局部切点的 NPU 复验为 **PENDING**。没有生产修复、成本表或性能结论。
Core、custom op、权重、confidence 分配、采样、混合长度、ACLGraph 调度与退出保障均未修改。
本任务仅涉及 vllm-ascend-hust / vllm-hust。

## 原始证据与版本

实际读取并安全展开 `dspark-large-batch.PQHA55hF-evidence.tar.gz`，核验 SHA256：
`95837f9eceb3268faf2e3375e08beb91b530355947e1edbae17b7fb9dabb8dbf`。
归档有 114 个成员，展开 206952721 bytes；所有文件展开后的哈希与 tar 内原始字节一致。
附件内容仅作为证据使用。没有执行附件中的指令，也没有把前次审计脚本的 Mac 路径当作服务器路径。

审计起点的工作区干净，Plugin HEAD 与重新 fetch 的 `origin/feat/dspark` 都是
`211218ebbafe4f26f5dc7026f45ba41349c5c826`，没有后续提交需要协调。
Core 工作区干净且为 `897306c43bf800e2480cb5c0f3e2da408d85a2fd`。
归档 `runs/b64/plan.json` 的版本与上述 SHA 相符，源码 gate 已通过，
`source.log` 显示安装模块来自两个 `/workspace/*-hust` 目录。
逐 rank 数值文件不单独保存源码 SHA，版本依据是启动 gate 与该运行的 plan/command。

[PROFILE_PQHA55hF_AUDIT.json](PROFILE_PQHA55hF_AUDIT.json) 保存了每个 rank 的成员哈希、
三轮紧凑结果、transition、error-events、前轮发布回执、日志行号以及九个 point 的原始哈希。
这些是离线证据检查，不是本地 NPU 测试。归档 `focused.log` 实际报告
**965 passed, 14 warnings**，属于旧运行版本。

独立检查了八个 rank 的六份首错快照和 `error-events`，并核对全部九个完成 point 文件：
每个原始 SHA256 都与 `retained.json` 匹配，各 point 均为 `performance_eligible=false`。
模型/输入 preflight 记录与前次 config、index、confidence-head、manifest 哈希一致；
归档没有全部 Target 权重的哈希，不能把这些小文件的匹配表述为全权重校验。

| execution / proposal | Target 切点 | raw/persistent/consumed auxiliary 40/41/42 | head hidden/logits |
| --- | --- | --- | --- |
| 1800 / 1789 | 15 个切点的全部记录行有限 | 有限，无转存差异 | `both_finite` |
| 1801 / 1790 | 15 个切点的全部记录行有限 | 有限，无转存差异 | `both_finite` |
| 1802 / 1791 | embedding、layer 0 output 有限；从 layer 1 output 起，已观测切点有效行 0 为 NaN | 行 0 NaN，无转存差异 | 候选行 0–4 NaN |

全部八个 rank 的上述结果一致，无 Inf；有效行 1–10 和 padding 行 11 有限。
`recording_error=null`、`missing_boundaries=[]`、`raw_replay_verified=true`，
15 个切点和三组 raw/consume 回执都等于各自 execution。98 次合并 D2H 全部完成；
numeric 计数为 97 次 `both_finite`、1 次 `hidden_nonfinite`。
该 point 共有 95 次 FULL 和 3 次明确标注的非 FULL 观测，保留的最后三轮都是 FULL。

这证明当前已观测异常区间是 **`layer.0.output → layer.1.output`**。
这里的 layer 1 是零起始编号的第二个 decoder；尚未观察它的实际输入、attention 或 FFN
子边界。主 hidden 有限不能证明历史 KV、HC mixing 参数、RoPE 或其他状态有限。
layer 40 auxiliary 是传播后的异常边界，不是最早异常层的证据。

## 实际失败布局、回执和两个错误

配置的第十个 point 为四请求、ell `[5,3,0,0]`、12 tokens。
失败轮只有三个请求，不能沿用配置的四请求布局：

| request row | request ID | ell / query length | Target rows | positions / seq_len | Draft candidate rows |
| --- | --- | --- | --- | --- | --- |
| 0 | `batch10-3-92df055f` | 0 / 1 | [0,1) | 222 / 223 | [0,5) |
| 1 | `batch10-2-8cbf0b0e` | 3 / 4 | [1,5) | 245–248 / 249 | [5,10) |
| 2 | `batch10-1-80c39dc2` | 5 / 6 | [5,11) | 524–529 / 530 | [10,15) |

query starts 是 `[0,1,5,11]`，11 个有效 Target tokens、graph capacity 12、15 个候选。
本次实际 seq_lens 为 **`[223,249,530]`**，不能套用上一归档的第二请求长度。
padding 的旧 position 为 648、token ID 为 90，不属于任何请求。
受影响有效行 token ID 为 58；依据有效 query end 和零拒绝，下一轮起草位置为 223–227。
三轮都记录 `num_rejected=[0,0,0]`、`num_sampled=[1,4,6]`。

各 rank 的本地 pool rows 分别为：0/2/4 `[63,61,60]`、1/6 `[63,62,61]`、
3 `[63,61,62]`、5 `[63,62,60]`、7 `[63,60,61]`。这些不是 packed Target rows，
也不是候选行。所有 rank 都保留了 execution **1794→1795** 的四请求到三请求转换；
NaN 在 1802，晚七次 execution，不能归因于退出当轮。

按八份 `rank-N-error-events.json` 与首错 ring 排序：

1. execution 1800/1801 分别成功发布 proposal 1789/1790。
2. execution 1802 消费旧 1790 后尝试 1791。`markov.error` 时
   `_markov_attempt_step_epoch=1791`，`_markov_step_epoch=null`，
   `_published_proposal_step_epoch=null`，owner map 为空；新 proposal 尚未发布。
3. execution **1803** 的下次 `execute_model` 在 `ConfidenceVerification.select`
   检查 scheduled candidates 时抛出缺少 owner，proposal epoch 仍为 1791，发布仍为空。
   它没有消费一个成功返回的 1791 proposal，也没有执行下一次 Target replay。

源码中 `_build_core_proposal` 和 owner 发布位于 Markov 成功之后。这条实际事件链支持
owner 错误是首次数值失败后的连带异常；没有独立的、发生在 NaN 之前的 owner 生命周期
缺陷证据。保留原始 Markov 错误和所有 ownership 检查。RPC/EngineDead 为后续错误；
cleanup 完成且未超时，supervisor `signals_sent=[]`。

## layer 1 的真实调用链审计

主要源码锚点：
[DeepseekV2DecoderLayer / DeepseekV4Model](../../vllm_ascend/models/deepseek_v4.py)、
[DSA wrapper](../../vllm_ascend/ops/dsa.py)、
[DSA metadata 与实际 forward](../../vllm_ascend/attention/dsa_v1.py)、
[slot mapping kernel](../../vllm_ascend/worker/v2/block_table.py)、
[MRV2 prepare_inputs](../../vllm_ascend/worker/v2/model_runner.py)、
[DSpark proposal/context 生命周期](../../vllm_ascend/worker/v2/spec_decode/dspark/speculator.py)。

- `DeepseekV4Model.forward` 把 layer 0 返回的 `hidden_states` 直接传入 layer 1，
  两者之间没有 request/pool gather。`DeepseekV2DecoderLayer.forward` 一进入就用
  `hidden_states.clone()` 替换传入的 residual；它不消费上一层返回的旧 residual。
  当前 layer 0/1 间没有被源码证明的别名写坏，但设备/编译器生命周期不能仅由 Python
  语句推定正确。新输入切点在 clone 与任何 HC 操作之前。
- Attention 前执行 `npu_hc_pre_v2`，产生主 hidden 和 `post/comb`，再 RMSNorm。
  `self_attn` 实际通过 `torch.ops.vllm.dsa_forward` 的 PrivateUse1 实现运行；
  Python fake 不是数值生产者。输出被写入新建 `output`，返回包含 attention 输出投影
  与通信。Attention 后 `npu_hc_post` 消费该输出、cloned residual 和 `post/comb`。
  随后再 clone 为 FFN 的 residual。
- FFN 前另一次 HC pre + RMSNorm，然后 `DeepseekV4MoE.forward`，最后 HC post。
  MoE 包括 router/专家/共享专家及归约；hash routing 是否启用由 `num_hash_layers`
  和层号决定。当前 archive 没有单独证明 layer 1 实际路由值、量化 scale 或专家输出有限。
  返回边界将这些内部步骤作为一个区间，不提前指定某个 kernel 为根因。
- 每个 rank 的当前 layer 1 metadata 前缀只含 `self_attn.swa_cache`；结合
  `filter_metadata`、`_build_kv_cache`、`_forward_decode`，对应 SWA 分支，而非本层
  c4 indexer 或 c128 compressor。准备时 block table 地址与 `input_block_tables[3]`
  一致，block size 32；这只验证 binding，**没有验证实际 KV 内容**。
- SWA prolog 为 Q/KV 投影、norm、Q RMS 与原地 partial RoPE，然后向本层 SWA cache
  scatter。W8A8 dynamic 与 multistream 的具体分支由已加载 quant method/config 决定，
  archive 没有保存所有这些开关，不能把某条 prolog 分支称为已测 producer。
  Attention 实际接收 SWA KV、group block table、query starts、seq_lens、sink、
  SAS metadata；非 A5 选择 `npu_sparse_attn_sharedkv`。之后还有 inverse RoPE、
  `wo_a/wo_b` 和配置相应的 TP/OTP 通信，均在 `attn_output` 之前。

实际非 A5 slot 是 `[physical_block, offset]`，不是 flat pool index。
无 CP 时位置 222 的逻辑 block 为 6、offset 为 30，但 physical block 值未被数值采集。
源码的 slot kernel 先使用当前 request→pool 映射，再按真实 group 的 block size 与位置
查询该请求的 block table；末尾分支填 PAD_ID。Target 与 Draft 具有各自 metadata builder，
同次 build 的共享 query/ratio 字典每批新建；持久 slot/start/SAS buffers 在准备时更新。
Draft context slots 在复用 query slot buffer 前 clone。已有 occupied-byte 隔离检查保留，
不能据此证明历史写入、KV cache 有效内容或跨 stream 依赖正确。

[shared-KV 算子契约](../../csrc/attention/sparse_attn_sharedkv/README.md) 的 TND 前缀允许相等
相邻值；PA_ND 不使用 `cu_seqlens_ori_kv` 来表示 token mask。
SWA 调用指定 `ori_mask_mode=4`、left=`window_size-1`、right=0。
需验证的是算子真实读取的 page/window，而非按字段名称或 0/1 模式推断。
本轮没有可证明的长度、索引或状态接口违反，因而不提交猜测性生产修复。

| metadata / 状态 | 本归档核验范围 | 未覆盖 |
| --- | --- | --- |
| request/pool/query end | 八 rank 三轮 CPU/device 一致，query ends 扣除实际 rejected | 更早的所有 pool 复用和历史写入 |
| token IDs / positions | captured 输入前 11 行与 Target/proposal 相符 | padding 原值不证明有无跨行传播 |
| captured query / seq / start | 实际 capture 对象的 starts 为 `[0,1,5,11,11,…]`，seq 为 `[223,249,530,0,…]`，start 为 `[222,245,524,0,…]`；input_positions 前缀相符 | 其他 attention/KV metadata 不能由此推断正确 |
| group / slots / blocks | layer 1 绑定 group 3；32-token blocks；当前 metadata 的 slots shape `[11,2]`，block table `[12,256]` | graph 绑定的 slot、physical page 值，历史 KV 每次读写、SAS 有效范围均未导出 |
| RoPE / sink / HC / MoE 状态 | 代码消费位置和调用次序已核对 | 实际数值与隐藏状态间因果关系 |
| padding | 零 query tails，零 seq/start tails，数值快照的 padding 行有限 | capture 时 scalar capacity 与 kernel 实际读范围不能仅用实时 metadata descriptor 验证 |

## 参数化局部观测

新增可选 `--profile-target-layer <zero-based index>`，只允许与隔离 B64 profile 的
`target-boundaries` 一起使用；模型构造时再次验证层号范围、PP1、capture 上限。
未指定层号时保留原 15-cut 计划。`target-boundaries 1` 的 control 入口选择九个切点：

1. `embedding`。
2. `layer.0.output`，前置正常边界。
3. `layer.1.input`，实际层输入，形状 `[capacity,hc_mult,hidden_size]`。
4. `layer.1.attn_input`，HC pre + norm 后、实际 attention 调用前。
5. `layer.1.attn_output`，包括输出投影和通信的 attention 返回。
6. `layer.1.residual`，attention 后 HC/residual 更新。
7. `layer.1.ffn_input`，第二次 HC pre + norm 后。
8. `layer.1.ffn_output`，MoE/FFN 返回。
9. `layer.1.output`，最终 HC/residual 更新。

层号并未写死为 1；选 0 时前置边界为 embedding，其他层使用紧邻前层 output。
这次移除远处的内部切点，但继续保留所有原 raw/persistent/consumed auxiliary 与 head
对照，没有再打开整个 drafter 诊断，没有新增全 KV/全层 tensor dump。

复用 `TargetBoundaryFlags` 和现有六个 decoder 切点，只有实际层输入是新增调用位置。
每个选中切点在真实图执行中按行归约 NaN/Inf，写本次 execution receipt；未选切点编译为
无设备工作。保留 `torch.sym_min`，兼容冻结 Core 绕过 Dynamo guards 后从 8192 profile
切到小 capture shape 的执行。R9 旧快照没有 `observe_layer_input`，不会引入新调用。

Replay 前清 receipt、arm execution；图返回后立即以 `copy=True` 保存紧凑 flags/receipt，
避免后轮复用覆盖。raw aux copy 仍在模型返回后、Core 持久转存前的 capture closure 内；
其 epoch copy 随每次 replay 执行，capture 数据不能伪装成当前 execution。
消费边界观察的是本次实际拼接的有效 auxiliary，随后统一回传并检查所有回执。

结果沿用 `auxiliary.rounds[].target_internal`，新增顶层 `target_internal.target_layer`
便于确认所选计划。按本轮 request IDs、有效 query prefix、execution/proposal 映射，
保留首异常和前两轮。有效 Target 行与 padding 分开；head 仍用 candidate 映射。
缺失/过期/未返回的 replay 为 `INVALID_RECEIPT`，非 FULL 为 `NON_FULL_UNOBSERVED`，
均为 **unavailable**，不给有限性结论。原始 NaN 抛错前仍保存并 fsync 首异常证据。
`target-first-nonfinite` 文件的 numeric 计数可能早于 head latch，但同轮 `head_flags`
已经在合并 packet 中，不能误读为异常 head 未采集。

局部计划在 C=384 时持久 bank 为 **6992 bytes/rank**（旧计划 11648）：
9×384×2 bool，9 个 int64 receipt 和一个 epoch。每轮有 18 个按行归约、9 次 flag 写入、
9 次 receipt 写入；与旧 broad 模式相比减少六个切点，但增加了实际 layer input 扫描。
单个 `[384,4,4096]` 临时 bool mask 为 6 MiB，C=12 时 192 KiB；
编译器融合/复用及 NPU 峰值 workspace 尚未测量。

后 replay owned flags/receipts 在 C=12 时贡献 **1800 bytes**，C=384 时 55368 bytes，
与原有 bounded integer、auxiliary/head flags 一起拼成 int64 packet。
仍为每个完成 proposal/rank **一次 blocking D2H**，每个切点零 host 等待，没有新增全局同步。
异常或没有 proposal 的部分 Target 仍按既有路径 drain 一次。旧 raw aux banks 的
约 17.86 MiB/rank 及其拷贝未改变。正常有限轮不增加写盘；首异常与 error-events 的
有界写盘/等待会改变复现时序。默认关闭时无这些分配、归约、拷贝或同步。
所有诊断身份仍为 `performance_eligible=false`，不能编译成本表。

## 服务器唯一下一次运行

使用最终交付 SHA 替换下方 `DSpark_SHA`，在既有 CANN/custom OPP shell 执行一次。
Core SHA 已填入。严格 shell 选项仅在子 Bash 内，父 shell 接收退出码。

```bash
DSpark_SHA='<交付的 signed-off commit SHA>'
if bash -s -- "$DSpark_SHA" <<'BASH'
set -euo pipefail
sha=$1
plugin=/workspace/vllm-ascend-hust
core=/workspace/vllm-hust
manifest=/workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json
test -z "$(git -C "$plugin" status --porcelain)"
test -z "$(git -C "$core" status --porcelain)"
test "$(git -C "$core" rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd
test "$(git -C "$plugin" branch --show-current)" = feat/dspark
git -C "$plugin" fetch origin feat/dspark
test "$(git -C "$plugin" rev-parse origin/feat/dspark)" = "$sha"
git -C "$plugin" pull --ff-only origin feat/dspark
test "$(git -C "$plugin" rev-parse HEAD)" = "$sha"
bash "$plugin/tools/dspark/run_dspark_profile_control.sh" "$sha" "$manifest" target-boundaries 1
BASH
then
  printf 'Diagnostic completed; inspect coverage and numerical evidence.\n'
else
  DSpark_RC=$?
  printf 'Diagnostic/preflight exit=%s; stop and retain first error.\n' "$DSpark_RC"
fi
```

入口打印新的 `SERVER_RESULT_DIR`，保留所有旧目录；依次验证源码、小文件 provenance、
安装态 focused tests，任一失败就不进入后续阶段。沿用同一单引擎前十 point 和合成生成器；
模型 `/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8`、revision
`9e8679a9db7eec11efed9925f7efb96549077545`、greedy seed 0、TP8+EP、MRV2、K5、
BF16/Ascend 量化、target FULL_DECODE_ONLY、draft eager、captures `[6,12,24,48,96,192,384]`、
8192 长度/token 上限、block size 32、memory 0.9、contexts `[128,2048]`、output 512、
warmup 2、samples 5 都不变。任何 point 失败即停止；没有 B128/B256、全成本表或性能比较。
本轮没有 graph/eager 对照；新启动不能宣称 request、KV/state、epoch 与随机状态完全相同。

保持既有诊断限时：supervisor 从加载前计 3600 秒，失败 grace 20 秒、TERM grace 5 秒，
RPC 120 秒、cleanup 8 秒。受控停止只针对本次拥有的 child session。
在第二个 shell 将 `DSpark_RUN` 设为本次精确打印的目录，可查询状态：

```bash
DSpark_RUN='/workspace/dspark-results/dspark-large-batch.<本次打印的后缀>'
cat "$DSpark_RUN/runs/b64-supervisor.json"
cat "$DSpark_RUN/runs/b64/frontend-operation.json"
tail -n 60 "$DSpark_RUN/generation.log"
```

需要受控停止时，单独执行 `touch "$DSpark_RUN/STOP"` 并等待退出/归档；
不要 killall。preflight 期间创建 STOP 会阻止 NPU 阶段。
原入口自动导出 `$DSpark_RUN-evidence.tar.gz` 和 `.sha256`。
诊断导出错误不能覆盖原始非零退出码、`status.txt`、phase `*.pipestatus` 或首错文件。
若导出失败，先确认 supervisor 已结束，再独立重试导出到新文件：

```bash
if bash -s -- "$DSpark_RUN" <<'BASH'
set -euo pipefail
run=$1
test -f "$run/status.txt"
retry=$(mktemp --suffix=.tar.gz "$run-evidence.retry.XXXXXXXX")
tar -czf "$retry" -C "$(dirname "$run")" "$(basename "$run")"
sha256sum "$retry" > "$retry.sha256"
printf 'Retry evidence: %s\n' "$retry"
BASH
then
  printf 'Export complete; original run status unchanged.\n'
else
  DSpark_EXPORT_RC=$?
  printf 'Export failed (%s); retain result directory.\n' "$DSpark_EXPORT_RC"
fi
```

最少回传**新 evidence.tar.gz 及其 .sha256**。归档需要八个 rank 的 target/auxiliary/head
首错与前两轮、first-failure、error-events、latest（存在的文件），以及原始 point 文件、
retained/plan/command/provenance、focused/generation 日志和 cleanup/supervisor/退出状态。
没有异常文件时仍回传完整归档；不需要全 hidden/logits/KV tensor。

## 验收与下一步

先检查新 SHA、层号 1、原 point 前序、全部 rank 的 recording_error、D2H 完成数，
以及异常轮和前两轮 **9 个切点 + raw/consume 的同轮回执**。缺少切点或过期回执为
unavailable，不能当作有限。确认实际 request/Target/candidate/padding 映射后判读：

- layer 0 output 有限而 layer 1 input 异常：收窄到层间传递/生命周期。
- layer 1 input 有限而 attn_input 异常：HC pre 或 RMSNorm 区间。
- attn_input 有限而 attn_output 异常：Q/KV prolog、历史 SWA KV/metadata、attention、
  inverse RoPE、输出投影/通信；届时才增加针对实际访问 group/slots 的局部证据。
- attn_output 有限而 residual 异常：HC post 及其 residual/post/comb 输入。
- residual 有限而 ffn_input 异常：FFN HC pre/norm。
- ffn_input 有限而 ffn_output 异常：MoE/router/专家/通信区间。
- ffn_output 有限而 output 异常：最终 HC post 及其额外输入。
- 所选切点有限但 raw auxiliary 异常：剩余未观测 Target 区间，不能称 layer 1 已修复。
- 未复现：仅报告新局部观测下未复现，根因与生产修复仍未证明。

以上是定位区间，不是对某个算子或最初异常 rank 的归因。

## 本地验证

Python 3.12 / CPU Torch 的相关回归 **492 passed, 3 skipped**；
capture/变长与 padding metadata 的 source/ABI 子集 **98 passed, 110 deselected**。
新测试使用实际 decoder forward 语句和实际 capture/replay wrapper，叶子 NPU 算子为 mock；
验证 layer 1 六个内部阶段、真实输入边界、NaN/Inf 映射、epoch、前两轮保留、重排/结束/
复用 pool、快照不被覆盖、缺失回执、原始异常、默认关闭、旧 R9 兼容、入口传递，以及
8192-profile FX 图在绕过 guards 后按多种小形状逐轮执行。相关退出/owner 回归继续通过。

三项 skip 需要安装态 vLLM/Ascend；110 个未选 runtime 变体和真实 DSA/HC/MoE 需要服务器环境。
这些本地结果不是 NPU 修复验证；服务器下一轮局部诊断复验 **PENDING**。
本轮修改文件的全部适用 manual pre-commit hooks、Ruff、shell syntax 和 `git diff --check`
通过。按规范在隔离临时 worktree 中对暂存补丁运行 `bash format.sh ci`，返回 1：
与上次全仓检查完全相同的八个失败 hooks、相同的 78 个自动修改文件，均不属于本轮变更。
失败项为 Ruff check/format、codespell、typos、Markdown、workflow/shell lint 和 forbidden imports。
未把这些既有失败报告为 CI 通过；无关自动格式修改仅发生在已移除的临时 worktree。
