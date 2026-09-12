# DSpark：layer 1 attention 内部诊断与退出记录容量

本轮是 **NaN 诊断补充 + 退出记录覆盖修正**，没有生产数值或进程退出修复。
原始 NaN 本次再次复现，最早已定位区间为 `layer.1.attn_input → layer.1.attn_output`；
具体 producer/错误状态仍为 **UNKNOWN**。新代码的 NPU 复验为 **PENDING**。
Core/custom op、模型权重、confidence、采样、NaN/owner 检查、资源释放次序和超时均不变。

## 原始归档审计

实际读取 `dspark-large-batch.Ftmn8d8I-evidence.tar.gz`，SHA256
`d74077045654f3a6488dc24462e7b193b7a12f9da330d840734724a64f4ea69f`，
152 个成员，展开 207083872 bytes。路径/类型与展开原始字节已核对；没有执行归档内代码。
运行版本 Plugin `abd554d8d0ed6ff05e54ce0040db10887c9cb942`，Core
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`。开发起点、fetch 后 origin/feat/dspark
与 Plugin 相同，两个工作区原本干净；本改动叠加在该版本之上，无 reset/force push。
source.log 的安装路径、source.pipestatus、plan 和 supervisor 命令相互匹配。

[逐 rank 审计数据](PROFILE_Ftmn8d8I_AUDIT.json) 保存原始哈希、三轮数值/metadata、
转存回执、错误事件和退出截断锚点。可用任意本地归档路径独立复算，无 Mac/服务器路径依赖：

```bash
python tools/dspark/verify_ftmn8d8i_evidence.py \
  /path/to/dspark-large-batch.Ftmn8d8I-evidence.tar.gz --output /tmp/ft-audit.json
```

这些是离线断言，不是新 NPU 测试。核心证据路径为 `runs/b64/retained.json`、各 point
原始 JSON、`worker-first-failure/rank-{0..7}-first-nan.json`、`*-first-failure.json`、
`*-target-first-nonfinite.json`、`*-auxiliary-first-nan.json`、`*-error-events.json`。

- 前九 point 原始 SHA 全部匹配 retained；全部请求完成 512 tokens、无生成错误。
  第十点 `ctx128-n4-t12-skewed` 四个初始请求累计输出 **512、397、124、95** tokens，
  后三者记录 EngineDeadError。没有十点成功或可用成本表。
- 八 rank 数值计数均为 97 次 both_finite、1 次 hidden_nonfinite，98 次紧凑回传全部完成。
  `recording_error=null`。execution/proposal **1802/1790、1803/1791** 全部已观测边界有限。
- **1804/1792**：embedding、layer.0.output、layer.1.input、layer.1.attn_input 有限；
  layer.1.attn_output 首次在**有效 target 行 0** 出现 NaN。后续 residual/FFN/output 同行异常。
  九个 Target 回执、三份 raw/consume 回执均是 1804，FULL、raw_replay_verified=true。
  图内归约和 receipt copy 随 replay 执行，replay 后立即保存 owned compact copy；
  不是 capture 数值或下一轮覆盖后的读取。
- raw/persistent auxiliary 40/41/42 的行 **0 和 11** 含 NaN，consumed 只有有效行 0。
  转存没有记录差异；head hidden/logits candidate 行 **0–4** 异常，无 Inf。
  layer 1 已观测切点的 padding 行仍有限，因此不能从 raw padding NaN 推导 padding 污染起因。

失败轮实际顺序为 `batch10-3-b9c74bbc`、`batch10-2-bfad06ee`、`batch10-1-8467a66b`；
query starts `[0,1,5,11]`，query lengths `[1,4,6]`，有效 target tokens=11、图容量=12、
draft candidates=15。异常请求 target position=222，target seq_lens=`[223,255,530]`。
第十点初始四请求的指定长度不是失败轮布局。pool 索引在各 rank 为不同的稳定排列，见审计 JSON；
它们不是 target/candidate 行，也不能据此认定 TP 映射错误。

| execution | proposal | 原始事件 | 发布状态 |
| --- | --- | --- | --- |
| 1804 | 1792 | Markov base logits contain NaN | 新 proposal 尚未发布，markov/published epoch 为 null、owners 为空 |
| 1805 | 1792 | Scheduled candidates lack current proposal owners | 更晚 execute 尝试消费未成功发布的结果 |

各 rank 的首错独占文件均保留数值错误；RPC/EngineDead 是后续观察到的失败。
这条链支持 owner 错误为连带异常；没有独立 owner 生命周期缺陷的复现证据。

## 真实调用链与当前缺口

[decoder](../../vllm_ascend/models/deepseek_v4.py) 的 `attn_input` 在 HC pre/RMSNorm 之后；
`attn_output` 在整个 self_attn 返回之后，包含输出投影和 TP。
实际链路为 DeepseekV4Attention →
[AscendDeepseekSparseAttention / dsa_forward](../../vllm_ascend/ops/dsa.py) →
[AscendDSAImpl.forward / _forward_decode](../../vllm_ascend/attention/dsa_v1.py)。
`_build_kv_cache` 从该层实际 swa_cache_layer 的 KV 取值；decode 对 metadata 按前缀排序，
SWA 分支仅有一个 swa metadata，使用 block table 和 block/offset slot mapping。
该路径按请求位置读 KV，不使用本地 request-state pool row 直接索引 KV。

本次日志为 910B2、flashcomm1=false、flashcomm2 size=0；layer 1 的 captured metadata
只有 SWA。结合冻结源代码，支持**非 A5 的 SWA decode**，compress_ratio<=1；本层不执行
compressor/indexer。不能把别层的 compressor/state metadata 推断成本层输入。
[DeviceOperator](../../vllm_ascend/device/device_op.py) 选中
`_C_ascend.npu_sparse_attn_sharedkv`，PA_ND KV、TND Q、ori_mask_mode=4、
left=window_size-1、right=0，传入真实 SWA block table、query starts、seq_lens、sink 与 SAS metadata。
[custom op 接口](../../csrc/attention/sparse_attn_sharedkv/README.md) 使用绝对逻辑 token/block
位置与因果窗口；相等 query starts 允许空请求。没有据 mask 的 0/1 外观或 padding 数量判错。

`multistream_dsv4_dsa_overlap` 在当前配置默认 true；归档没有每个投影实际 quant method 的
回执，因此这是源码/配置推导，不能仅由 checkpoint 的 w8a8 名称确定所有投影实现。
新诊断保存真实 capture 分支、quant method 类型和投影路径，并以逐轮 receipt 验证其执行。

多流 prolog 的 Q 路径为 hidden quant → wq_a → qr norm/quant → wq_b → Q RMS → RoPE；
KV 路径为 hidden quant → wkv → KV norm → RoPE → scatter。原有 main/aux stream
事件与等待将 scatter 放在 attention 之前；没有静态证据证明缺少某个依赖，也没有新增等待。
raw attention 之后执行 inverse RoPE、wo_a batch matmul、wo_b。
标准 wo_b 的插件 quant/unquantized `apply` 返回 local projection，冻结 Core 的
`RowParallelLinear.forward` 随后执行 TP all-reduce，再回到 layer attn_output。
flashcomm/oproj/olora/A5 的其它融合分支不由本次 probe 支持，安装时明确拒绝，避免误报覆盖。

已核验的 device/CPU 字段为实际 query starts、seq_lens、positions、input_ids、pool 映射和
captured layer 1 start_pos/seq_lens；这些记录来自每轮 owned copy，仍不等于完整 metadata 正确。
**尚未证明**的包括 Q/KV 中间值、历史 KV 内容与写入来源、权重/量化 scale、RoPE 值正确性、
真实 block/slot 映射、SAS packed metadata/native mask/实际 kernel 读取范围、TP 原始异常 rank。
层输入有限只约束 hidden；全 rank 最终同样 NaN 不能确定最初 rank 或算子。
没有找到足以构造真实失败机制回归的索引/长度/生命周期违约，因此没有做推测性生产修复。

## 新增的有限观测

默认关闭的 `--profile-target-attention` 要求 `target-boundaries` 和明确的 target layer。
快捷入口为 `target-boundaries 1 --attention`。保留已有九切点、raw/persistent/consumed auxiliary
与 head 对照，在**这一层**增加 12 个逐 target 行 NaN/Inf 切点，总计 21 个 Target 切点：

| 名称（前缀 layer.1.attention） | 所处位置及用途 |
| --- | --- |
| rope_cos、rope_sin、sink | 实际 decode 参数的有限性；有限不证明值/位置正确 |
| q_normalized | wq_a/qr norm/quant/wq_b/Q RMS 后，首次 Q RoPE 前 |
| q_rope | 实际旋转后的 Q，原地更新之前已保存上一个切点统计 |
| kv_normalized、kv_rope | wkv/KV norm 后，以及旋转后、scatter 前 |
| kv_window | scatter 后、attention 前，按真实 metadata 推导的因果 SWA 窗口 |
| raw_attention | sparse attention 原始返回，inverse RoPE 前 |
| inverse_rope、wo_a | inverse RoPE 后，以及 wo_a 后、wo_b 前 |
| wo_b_local | 标准量化/非量化 wo_b 的 local 输出，Core TP all-reduce 前 |

[AttentionProbe](../../vllm_ascend/diagnostics/dspark_profile_attention.py) 只读当前 layer 的窗口，
窗口上限 128、token 容量上限 384。用实际 device query starts/seq_lens 推导每行 request、
position、窗口起止，从真实 SWA block table 解析物理页；记录当前写 slot 与期望 slot、
窗口越界计数、slot mismatch、position_matches_target，padding 明确无 request。
越界索引仅在诊断读取中 clamp，原 kernel 的参数完全不动；若 window_range_valid=false，
该窗口的有限性不能充当“有效 KV 全部有限”的证据。未读取的历史 cache 不影响统计。
这里只验证接口定义的窗口，不能证明 kernel 没有读越界、读错 mask 或使用错误 SAS 内容。
SAS 二进制调度结构、完整 block table、权重、全部 KV 和其它层 tensor 不导出。

Q 与 KV 是并行分支，不伪造一个全序“首次 producer”。保留 outer brackets，内部统计单列在
`auxiliary.rounds[].target_internal.attention`；旧 `boundaries` 同时包含全部切点。
所有记录带 point/rank/execution/proposal、真实 target 行/request 映射；head 保持独立 candidate
映射。missing/stale replay/缺失任一切点产生 INVALID_RECEIPT、空 rows、recording_error，
不得当作有限。首次有效行非有限和此前两轮独占保存，即使该异常尚未传到 head；重复错误不覆盖。
owned flags/state 在 replay 返回后、buffer 仍属本轮时保存；D2H 仍在原 head 统计合并处完成，
既有 Markov 检查不变，抛错前有保存机会。

额外开销（诊断开启时）：12 组逐行 isnan/isinf 归约、图内 receipt copy、窗口索引/gather/掩码，
以及 10 列 int64 行状态；不增加每边界 host wait、D2H 次数或 stream/global synchronize。
沿用每轮一次合并紧凑 D2H。在容量 12 时新增回传约 **3360 bytes/轮**；最大容量 384 时，
总 Target bank（含行状态）**47024 bytes/rank**，比九切点多 40032 bytes。
窗口临时量规模为 T×W×KV_head_dim：以 W128、D512、bf16 为例，每个 gather/masked tensor
在 T12 时 1.5 MiB、T384 时 48 MiB，另有 bool 归约临时量；实际图内峰值受 allocator/fusion 影响，
尚无 NPU 实测，不把 bank 大小当总峰值。各 capture shape 也可能保留自己的图内临时存储。
统计会改变多流负载和复现时序。默认关闭时没有这些归约、拷贝、分配、同步，仅现有调用点的
空 probe 分支。`performance_eligible=false` 与成本表拒绝门保持。

下一次结果用于区分：Q/KV prolog 已坏；RoPE 后才坏；当前 KV 写入有限但读取窗口坏；
Q/旋转参数/sink/窗口有限而 raw attention 坏；inverse RoPE/wo_a/local wo_b 后才坏；
或 local wo_b 有限而 attn_output 在 TP 后异常。这些仍是区间证据，不是最终 kernel/root-rank 证明。
未复现时记录“本轮未复现”，不发布修复 PASS，也不由相同 prompt 宣称同状态 graph/eager 对照。

## 独立小项：退出记录容量

本归档全部八份 `worker-exit/*-lifetimes.jsonl` 均在第 128 条 event_limit_reached。
后续 exit_function/threading/atexit/GC 覆盖不足；最后一条 close_fds/存活列表不能证明挂起或泄漏。
cleanup.json 是 engine_returned/thread_completed=true、timed_out=false、event_loop=closed，
status=worker_cleanup_incomplete、success=false；内嵌及独立 worker-cleanup 仍为 running。
cleanup-failure.prior_error 保留已有 EngineDead；supervisor raw_returncode=1、signals_sent=[]。
这些是不同时间/范围的记录，不据此前端快速返回或无新信号发布正常 worker 退出。

[PostShutdownTrace](../../vllm_ascend/diagnostics/dspark_post_shutdown.py) 修正的是**日志覆盖**：

- append 总上限仍 128：ordinary 64、critical 48、error 16 独立预算，各有自己的截断标记。
  finalizer/GC 普通开始/返回不再消耗 append；关键 exit_function/threading/atexit 等保留容量。
- 新 `*-lifetimes-state.json` 原子替换：重复事件计数、最近 16 条、最多 16 个当前未完成调用、
  前 8 个异常、最多 16 种计数键及 other 桶；history 覆盖、pending/异常/键溢出和重入丢弃均显式记录。
  嵌套调用通过 call_id 配对；GC 按 generation/thread 配对。异常对象与原返回行为不变。
- 每份 state 最大 64 KiB，append 单条仍最多 16 KiB；不会无限扩大日志。
  暂存写入成功才 replace，写失败保留上一有效 snapshot，后续记录携带 recording_error。
  不等待自身重入锁，不保活 model/graph/tensor；记录 weakref 清除仍不证明原生析构完成。

这是退出时的 CPU/文件 I/O，每事件更新有界 snapshot，可能影响退出时序；没有运行期 D2H、
新切点、GC/释放/信号或超时改动。pending、overflow 或最后 alive 都不能替代父进程 wait 结果。
CPython 最终 GC 可能绕过 gc.callbacks，覆盖不足仍必须保留 UNKNOWN。
无需为此日志策略单独启动十点 NPU 实验。

## 验证与唯一下一次服务器实验

[本地验证记录](PROFILE_Ftmn8d8I_VALIDATION.json)：21 个相关文件实际 **563 passed、3 skipped**
（35.42 秒）；三项跳过需要安装态 vLLM/Ascend。新增 attention 文件 30 项、退出容量 3 项，
以及控制脚本参数覆盖均通过。本次修改文件全部 manual hooks 通过。
隔离 worktree 实际执行完整 `bash format.sh ci` 返回 1：八个失败 hook 和 78 个自动修改路径
与前一基线完全一致，本次文件没有被自动改写。没有提交无关格式化，没有报告全仓 CI 通过。
最终 Plugin SHA 见本次交付；归档中 **1032 passed、14 warnings** 是旧服务器运行的
focused 结果，不是新代码验证。新增 CPU 测试执行真实 DSA/projection/Core row-linear 函数体、
ATen 逐轮 replay，只有 NPU kernel、通信、流为 mock；覆盖各阶段注入、NaN/Inf/重排/padding、
窗口索引、首错和两轮历史、回执失效与 owned copy、默认关闭及 CLI 传播。
退出测试覆盖 finalizer 洪泛、未完成嵌套调用、异常、I/O 失败和真实 spawn 的正常/阻塞收尾。
安装态 vLLM/Ascend 的检查和 NPU kernel/ACLGraph 支持必须由服务器验证，不能以 CPU 通过代替。

仍须保留原单引擎十点前序：没有证明独立 point 或只重启相同 prompt 能复现相同 KV/随机状态。
**只运行一次**下列新 attention 诊断，兼带已有退出记录；不串联其它 batch、成本或性能任务。
最终消息提供填好新 SHA 的完整命令，严格选项只在子 Bash，父 if 接收退出码：

```bash
# plugin_sha 使用本次交付的完整 SHA；本块结构也见最终消息。
if bash -s -- "$plugin_sha" <<'BASH'
set -euo pipefail
plugin_sha=$1
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = feat/dspark
test -z "$(git -C /workspace/vllm-hust status --porcelain)"
test "$(git -C /workspace/vllm-hust rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd
git fetch origin feat/dspark
git merge --ff-only "$plugin_sha"
test "$(git rev-parse HEAD)" = "$plugin_sha"
bash tools/dspark/run_dspark_profile_control.sh "$plugin_sha" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  target-boundaries 1 --attention --worker-exit
BASH
then
  printf '诊断命令退出码：0（仍须检查数值/退出证据）\n'
else
  rc=$?
  printf '诊断命令退出码：%s；停止后续测试，保留首错与归档。\n' "$rc"
fi
```

入口保留 CANN/custom OPP 环境，核验模型 checkpoint 与冻结 manifest，自动创建新
`/workspace/dspark-results/dspark-large-batch.XXXXXXXX`，不覆盖旧归档；TP8+EP、MRV2、K5、
target FULL_DECODE_ONLY、draft eager、capture sizes 6/12/24/48/96/192/384、ctx128/2048、
每请求 512 tokens、warmup2/samples5、max model/batched tokens8192/memory0.9 均沿用。
仅跑到 ctx128-n4-t12-skewed，任一点失败停止。supervisor max_runtime=3600 秒，worker5+4、
engine12、outer16、failure grace24 秒保持有界；没有调大预算。

另一终端将实际输出的 SERVER_RESULT_DIR 填入 `result_dir` 后可查询和受控停止：

```bash
cat "$result_dir/status.txt" "$result_dir/runs/b64-supervisor.json"
tail -n 60 "$result_dir/runs/b64.log"
# 仅需主动停止本次运行时执行：
touch "$result_dir/STOP"
```

STOP 由原 supervisor 处理，不向无关作业发信号。结束后自动生成同名 `-evidence.tar.gz`、
`-evidence.sha256`，导出失败不覆盖原始 MAIN_RC/首错。若仅导出失败，可在进程停止后独立补导出：

```bash
if bash -s -- "$result_dir" <<'BASH'
set -euo pipefail
result_dir=$1
test -f "$result_dir/status.txt"
tar -czf "$result_dir-reexport.tar.gz" -C "$(dirname "$result_dir")" "$(basename "$result_dir")"
sha256sum "$result_dir-reexport.tar.gz" > "$result_dir-reexport.sha256"
BASH
then printf '补导出成功；原退出码仍以 status.txt 为准。\n'
else printf '补导出失败；不覆盖原始状态。\n'
fi
```

最少回传 **新 evidence.tar.gz + .sha256**；其中须保留 source/focused/checkpoint、plan/
retained/所有 point 原始 JSON、worker-first-failure 全目录（首次异常+前两轮+latest/error-events）、
worker-exit 全目录（包括新 state 文件）、point-completion/cleanup/worker-cleanup/supervisor/status。

验收分开报告：生成完成数；数值是否复现；21 切点+raw/consume 是否当前 FULL 回执及真实分支；
最早异常区间和 KV 范围状态；显式清理是否返回；各 worker 是否未经强制升级自然 exit=0。
null 退出码保持 unavailable。没有有效回执的边界不能判有限；一次不复现不关闭原故障；
进程强制/状态不全不能发布正常退出。诊断结果始终禁止编译可用成本表。
