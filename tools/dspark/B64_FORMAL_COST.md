# B64 正式成本采集与真实 confidence 闭环计划

## 合成功能验收已完成

已实际读取 pJap5lfX 外层归档和 v8vohAeE 模型归档，核验SHA256、内嵌归档、原始逐点文件哈希、
全部请求实际输入输出、逐rank FULL样本、预算、自然退出回执及各级PIPESTATUS。
正式 `model_report` 重建结果与原报告完全一致。

| 项目 | 核验结果 |
| --- | --- |
| Plugin | fe6b29be454ea6eec4e37f4c2989a9c5440f951f |
| Core | 71d2c1c436eba894a8e9eeb2c5af17e05cb42970 |
| 外层SHA256 | 2298ebd238982adede7fcfab1c68394aae2f02c93e68f53c3c42ec60e89a3fbf |
| 模型SHA256 | 3ddf1943036b083a994e97ef106f6d1e8d57ac7d34171d82bc083bcd74c7b848 |
| 完成情况 | 7点288请求，各实际2048输入、512输出；数值/owner/FULL通过 |
| 逐rank样本 | 每点每rank，target与draft各5个匹配样本，另有2个warmup |
| 退出 | 八worker真实退出码0、均回收；无强制升级、超时、残留、日志错误 |
| 耗时 | worker 12.942815秒；前端16.336888秒 |
| 整体 | overall_pass=true；synthetic_functional_exit_criterion=PASSED_THIS_RUN |
| 边界 | specified_lengths；real_text_validation=NOT_RUN；performance_eligible=false |

[证据摘录](PROFILE_v8vohAeE_ACCEPTED.json) 保留原报告、原始哈希和退出/来源回执。
原十点、第一阶段、第二阶段均冻结为通过。**合成功能扩展结束，不建立第三阶段。**
原5秒预算未验收、原生析构尾部未定位仍按原边界记录，不阻塞25秒命名策略下的后续工作。

## 当前缺项与源码契约

可访问的仓库及本次归档中没有与本轮运行身份匹配的可用正式表；本次identity明确包含
`diagnostic_only=true`，不能移除该字段或改标签编译成正式成本。没有读取服务器上未提供的其他表。

- `startup_cost_profile.grid/collect/point_samples/compile_startup` 已支持单引擎真实请求、NPU events、
  逐rank/布局中位数与单调包络；本次复用，不另建计时算法。
- `CostTable.load_startup/cost`（`vllm_ascend/spec_decode/dspark_verification.py`）使用schema 2，
  按target token量向上选图容量，再向上选请求/上下文桶。每个可达桶必须有真实数据，缺项/越界失败。
- `ConfidenceVerification.select` 使用当前proposal epoch的confidence，枚举全部候选预算。
  请求结束后查表请求数会下降；因此只测16/32/64或只测最终选中容量不足。
- 现有正式profile路径尚未接入25秒退出回执，并在父进程日志扫描前直接写可加载表。
  本次只为明确的 `--formal-cost-plan b64-confidence-cost-v1` 接入已验证的退出与发布门槛。
- 当前真实文本旧Shell入口仍固定旧Core；不能照旧直接运行。Python入口有真实confidence模式，
  但尚需下一步接入同一退出策略、正式表发布证明与逐请求决策回执。本轮不提供该运行命令。

`performance_eligible=false` 在成本表中表示校准不是性能结果，不表示表不可查。
是否可用由 `cost-publication.json.cost_table_usable=true` 与正式 `cost-profile.json` 共同证明。

## 下一轮工作负载决定的采样范围

已实际校验现有manifest及400条记录的原始哈希。前64条是64个独立prompt：
`openai/gsm8k`，revision `cc7b047b6e5bb11b4f1af84efc572db110a51b3c`，`main/test`。
实际输入29–116 tokens。其余336条为重复实例，不当作新增质量样本。
现有manifest SHA256为 `298f9560364fee5a30507aa645ad449abde24e917d2dc309f51083b36cce8cfa`。
此为归档内来源回执和内容哈希核验，不声称本地重新从外部数据源下载核验。

[真实文本计划](B64_REAL_TEXT_PLAN.json) 冻结这64个请求ID、来源case/revision、token/record哈希、
模型tokenizer revision及文件哈希。下一轮拟使用B64引擎、client outstanding=64、自然EOS、
最多256输出tokens、temperature=0、top_p=1、top_k=-1、seed=0。
不能重新渲染/截断prompt或补足到512输出来制造覆盖；实际调度并发由回执确认。

因此本轮仅采 **128-token输入锚点，上下文桶[0,640]**。不为未安排的长文本新增2560桶。
预计真实文本常规上界为116+256；成本上下文实际使用scheduler预查询computed上界，包含异步预记账，
与纠正后的物理KV长度不同。仍以运行期 `CostTable.cost` 的640硬边界为准；超过就失败，不假定
某个固定异步偏移一定成立，不静默继续。以后若要长文本，必须单独说明新工作负载和缺失成本桶。

请求锚点为1/6/12/24/48/64，来自原Graph容量边界与引擎上限，覆盖实际请求数1–64。
每个锚点仅采原合法矩阵中可达容量，balanced/skewed均保留；端点布局相同也按原schema独立采样。

| 请求锚点 | 采样Graph容量 | 对应有效target tokens | 两布局point数 |
| --- | --- | --- | --- |
| 1 | 6 | 6 | 2 |
| 6 | 6 / 12 / 24 / 48 | 6 / 12 / 24 / 36 | 8 |
| 12 | 12 / 24 / 48 / 96 | 12 / 24 / 48 / 72 | 8 |
| 24 | 24 / 48 / 96 / 192 | 24 / 48 / 96 / 144 | 8 |
| 48 | 48 / 96 / 192 / 384 | 48 / 96 / 192 / 288 | 8 |
| 64 | 96 / 192 / 384 | 96 / 192 / 384 | 6 |

[完整40点清单](B64_FORMAL_COST_PLAN.json) 保存精确ID、逐请求ell、有效token、图容量和顺序。
**单引擎，40点1106请求，每请求512输出，共566272输出tokens。**
每点/每rank/每kind丢弃2个匹配warmup，保留随后5个匹配样本，共3200个保留NPU计时样本。
不因耗时大小挑样本。CPU回归枚举1–64请求的全部10464个合法候选token布局，验证查表闭合。
这40点是成本表20个cell×两布局，不是补齐剩余合成功能矩阵。

上下文640是预先定义的估计桶上界，**不是声称在640长度测量过**。保留各样本实际context范围；
同桶上下文、未直接采样的请求数以及其他混合布局用向上桶与max包络估计。
max(rank/layout median)再取单调包络是保守处理，但不是任意布局/周期性压缩状态的严格最坏耗时界。
这项误差边界明确保留；640之外、64请求之外、缺少任一必需cell或样本均报错，不插值补造数据。

## 计时与发布门槛

target测量 `cudagraph_manager.run_fullgraph` 的真实FULL执行；draft测量紧邻该target的
`speculator._execute_draft`，实际draft为eager。draft记录中的full_decode是target邻接条件，
不能写成draft使用ACLGraph。实际capture回执必须覆盖全部7档且八rank为target FULL_DECODE_ONLY、draft NONE。
NPU event跨度包含这些调用内的设备工作/发射间隙，不等于完整decode步骤或端到端时间。

保留原profile每点begin/snapshot的同步边界和event计时，没有新增模型期全局同步。
关闭numeric/target/attention/operator/write-timeline观测，不重新扫描hidden/KV。
保留原Markov NaN、owner、confidence有限性与生成失败检查；清洁计时不声称所有内部张量均被检查。
退出步骤记录只在退出阶段生效，保留observer引用解绑与output-handler取消/等待修复。

另测20次CPU `allocate_prefixes`（64请求、context0、统一0.9概率）并保存全部perf_counter样本和中位数。
这是分配算法开销估计，不包含confidence head/D2H或TP broadcast；不假装它们已包含在成本估计内。
后续端到端固定K/confidence性能对照必须完整计入这些开销，不能从计时中扣掉。

发布过程：

1. 模型加载前打印清单、请求量和预算，离线重建第二阶段通过报告，并核验冻结真实文本契约。
2. 记录所有top-level safetensors分片与模型JSON的SHA256，以及实际confidence字节、config/index。
   过去只有部分checkpoint哈希，本次不把它们描述为全权重身份。发布前重新哈希，变更即失败。
3. 新引擎保留精确Plugin/Core、runtime identity（模型/revision/config、硬件、TP8+EP、BF16/量化、
   K5、target/draft模式、capture、Torch/torch_npu、memory/block/token预算），每point原始文件及哈希。
4. 原请求映射、输出完整性、成功执行与FULL样本门槛通过才接受point；失败停止并保留部分数据。
5. 先写 `cost-profile.pending.json`，source为unpublished，正式 `CostTable` 明确拒绝加载。
6. 子进程退出后检查严格日志、supervisor、Core/前端、残留及25秒命名预算；八worker必须真实0且回收。
7. 重建所有逐rank样本、采样窗口、中位数与包络，核对候选表、实际加载head及跨rank身份。
   重验全部查表布局后原子发布 `cost-profile.json` 和独立发布报告；无持久报告不保留可用表。

表保存生产者代码SHA；现有 `CostTable` runtime identity不包含代码SHA和全权重分片hash，
所以下一步真实文本入口还须显式核验已发布表、生产者代码兼容性和完整权重证明。
后续代码变化不能改写旧表SHA冒充同版，需要审核是否影响执行/计时，必要时重新采集。
CANN/custom OPP环境沿用，来源日志保留实际路径；未提供可复现二进制构建证明，不宣称源码与OPP一一对应。

## 唯一服务器任务及上限

模型运行上限7200秒（含初始化/capture/采样/退出）；40点输出量约为第二阶段的3.84倍，
关闭重型诊断且输入更短也不保证按比例缩短。本轮选择明确两小时上限，避免无界采集或自动重试。
到达上限按原监督器停止，worker25秒、TERM4秒、共享回收1秒、前端36/40秒、supervisor48秒及15秒最终收尾。
无gdb、额外观察等待0。采集前审计/全权重哈希、发布前复核各有1800秒上限；归档压缩时间另计。
CPU预检只运行本次host回归，不重复22/3/23局部生命周期或已通过完整模型验证。

填入交付完整SHA后，在原CANN/custom OPP环境执行；Core精确71d2…，remote=rzwang，记录实际URL/HEAD，
不永久改写远端。保留原v8vohAeE归档和manifest。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
test "$(git rev-parse HEAD)" = "$1"
bash tools/dspark/run_dspark_formal_cost.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  rzwang
BASH
then echo '成本采集调用完成；核对 cost-publication.json，不启动后续实验'
else rc=$?; echo "成本采集失败 rc=$rc；保留证据，不自动重试"
fi
```

外层新目录 `dspark-formal-cost.*` 自动保存日志、PIPESTATUS、内嵌模型归档及SHA256；
内层保留预检/JUnit、完整计划、来源/权重哈希、逐rank原始样本、retained、CPU开销、pending表、
发布表/报告（若通过）、清理/信号/退出码/残留。回传一个外层归档及SHA256。
必要时在日志 `SERVER_RESULT_DIR` 对应内层目录创建 `STOP`，保留受控失败；不追加点或第二次模型运行。

## 成本通过后的真实文本验收（本轮不执行）

先审核正式表及兼容性，再为冻结64题准备单独入口，仍为B64同一配置、25秒退出策略。
使用 `mode=confidence`、`profile=false` 和真实加载的 `mtp.2.confidence_head` 权重回执，
不通过 `specified_lengths` 或人为改分数制造长度变化。没有独立校准证据时报告uncalibrated。

复用runtime已有的current-epoch probabilities、last_selection、proposal ownership和执行布局记录，
需要有界保存逐请求ID/producer epoch/选择长度/预算/所选capacity与实际query starts/lengths/执行回执。
源码路径为 `record → select → allocate_prefixes → trim_scheduler_output → metadata/target执行 → accepted`；
trim修改本轮实际验证token/query，不能只检查输出长度或直方图来证明闭环。
当前性能入口默认不保留完整逐请求decision历史，下一步应做局部回执适配，不重新安装整层数值探针。

验收要同时证明：真实head调用、confidence批次、费用表加载来源、TP一致的当前epoch长度、实际FULL消费、
数值/owner与自然退出。若策略一直选择K5，应如实记录，不能改概率以制造自适应。
只覆盖冻结64题及≤640成本上下文，不声称真实文本质量、长输入、校准或吞吐收益已验证。
该闭环通过后才单独安排同输入/同配置固定K5与confidence端到端性能对照。

本轮正式NPU成本采集 **PENDING**，真实confidence闭环 **NOT_RUN**，性能比较 **NOT_SCHEDULED**。
