# B64 功能覆盖第一阶段（原十点验收已关闭）

## 冻结通过基线

已实际读取两份归档，独立核验SHA256、外层内嵌模型归档一致性，重新运行正式 `model_report`，
重算十点原始文件哈希、数值/归属/FULL及命名退出预算证据，结果与保存报告完全一致。

| 项目 | 已核验结果 |
| --- | --- |
| Plugin | b2810899840141f1a71324d90e263008eecc4f06 |
| Core | 71d2c1c436eba894a8e9eeb2c5af17e05cb42970 |
| 退出策略 | dspark-profile-25s-v1；实际Core25秒，额外等待0，无gdb |
| 外层 IqsNuZU6 SHA256 | 23ac9472ec9d04a035769e199ff791c22826e39b56f3b1bd9fef23d22a0b3302 |
| 模型 Ck6iA7rN SHA256 | 7b91b1b039cd2de7cb020d802862583f0c34b4decdecede0e29477bb3e1c2f8e |
| 生成及正确性 | 十点26请求各512tokens；数值、归属、FULL通过 |
| 命名预算及整体 | PASSED_THIS_RUN；overall_pass=true |
| worker | 八个真实退出码全0，全部回收；12.436236秒 |
| 前端 | 15.536660秒；无超时、强制升级、残留或日志异常 |
| host测试/返回码 | 25 passed、零跳过；各阶段及外层均0 |

[完整审计摘录](PROFILE_Ck6iA7rN_ACCEPTED.json) 保留报告、源码来源、哈希和PIPESTATUS。
**关闭：该命名配置下原十点完整模型验收阻塞。** 原5秒预算仍NOT_EVALUATED，具体原生尾部
析构耗时仍UNKNOWN；两者不再作为本阶段阻塞，不追加同场景退出诊断。
新场景失败不会修改这份通过记录或旧归档。

Core历史回执的 selected_remote=rzwang，但 remote_url=`file:///workspace/vllm-hust/`。
actual/fetched HEAD均为指定71d2…，origin仍为 `https://github.com/vLLM-HUST/vllm-hust.git`。
这是**本地所选来源和精确HEAD验证**，不是从GitHub个人远端取回的证明。
下一次仍显式选择rzwang、记录实际URL和HEAD，新增transport/scope说明；不永久改写远端。
本轮不要求额外外部来源核验，已接受的精确SHA保持不变。

## 覆盖边界与阶段选择

原 `startup_cost_profile.grid(64, [6,12,24,48,96,192,384], [128,2048], 512)`
共136个point、2948请求、1509376个输出token；请求数网格为1/2/4/6/8/12/16/24/32/48/64。
已通过的前十点实际仅覆盖输入128、请求1/2/4，不能称为实际64并发或2048输入验收。

新 `b64-functional-1` 从该矩阵按原顺序选择12点，**328请求、167936输出token**，
一次引擎初始化，模型子进程运行上限3600秒（含初始化和capture）。不删除stop后跑全矩阵。
输入继续为原合成方法：tokenizer.encode("x")的首个token重复到指定输入长度，
每请求输出512；manifest只保留既有来源门槛，不能把其中400条来源记录混作本轮328条合成请求。

本轮优先覆盖8/16/32/64请求的中间验证长度balanced/skewed；包含ell=0与ell=5混合。
额外选64请求完整K5/容量384，并用8请求的低、中、高验证量覆盖2048输入代表场景。
原B64 capture列表、TP8+EP、target FULL_DECODE_ONLY、draft eager、模型/权重和采样保持。
不重跑前十点，不声称这个新点序列等同原故障复现序列。

### 精确执行清单

`ell` 是每请求指定的draft验证长度，范围0–5。表中“验证量”为Σell；
“target query tokens”为Σ(ell+1)，包含每请求基础query；capacity是图token容量，
不是请求数。每请求仍会起草K5，不能把candidate行当作target有效行。

| 顺序 / point | 输入tokens | 提交请求数 | Σell | target query tokens | graph capacity |
| --- | --- | --- | --- | --- | --- |
| 1 ctx128-n8-t24-balanced | 128 | 8 | 16 | 24 | 24 |
| 2 ctx128-n8-t24-skewed | 128 | 8 | 16 | 24 | 24 |
| 3 ctx128-n16-t48-balanced | 128 | 16 | 32 | 48 | 48 |
| 4 ctx128-n16-t48-skewed | 128 | 16 | 32 | 48 | 48 |
| 5 ctx128-n32-t96-balanced | 128 | 32 | 64 | 96 | 96 |
| 6 ctx128-n32-t96-skewed | 128 | 32 | 64 | 96 | 96 |
| 7 ctx128-n64-t192-balanced | 128 | 64 | 128 | 192 | 192 |
| 8 ctx128-n64-t192-skewed | 128 | 64 | 128 | 192 | 192 |
| 9 ctx128-n64-t384-balanced | 128 | 64 | 320 | 384 | 384 |
| 10 ctx2048-n8-t12-balanced | 2048 | 8 | 4 | 12 | 12 |
| 11 ctx2048-n8-t24-skewed | 2048 | 8 | 16 | 24 | 24 |
| 12 ctx2048-n8-t48-balanced | 2048 | 8 | 40 | 48 | 48 |

[机器可读固定清单](B64_FUNCTIONAL_PHASE1.json) 包含完整逐请求lengths、统计和未覆盖point ID。
运行前再次由同一个合法矩阵生成并核对，不手写新形状。未知阶段、修改capture/输入/输出预算、
缺点或改变顺序均失败。入口不提供任意点拼接或无界继续选项。

表中请求数是提交数和FULL样本目标，不预先宣称已实现的调度并发。
每rank要求2个warmup加5个对应FULL样本，必须同时匹配实际调度请求数、query lengths、
target有效tokens和graph capacity。若提交64请求但只出现32请求的FULL，仍失败。
请求结束后并发降低、query长度变化和padding可以发生，保留原始事件和行映射，不将全程描述成64并发。
实际报告单独给出已保存target事件的请求数集合、query token数集合、FULL capacity集合、
逐rank匹配样本数及实测输入长度。事件历史不是每次scheduler执行的完备日志。

2048输入通过还必须在每个请求的 `observed_prompt_token_ids` 中确认长度为2048，
不能只看point名字或配置。这里只覆盖8请求的2048输入，不能外推到64请求长输入。

## 运行边界与分阶段安排

原完整矩阵比本轮多约9倍请求，耗时不能按请求数线性预测。先给本轮60分钟的硬运行预算，
不是承诺60分钟必定完成。沿用25/4/1秒worker宽限、TERM及共享回收，前端36/40秒，supervisor48秒。
达到3600秒会保留首错并受控退出；之后最多约48+15秒收尾（加轮询/OS调度误差），
预检、报告和压缩时间另计。预算耗尽、强杀或退出码缺失仍失败，不自动换点或追加运行。

通过前十点及本轮后，仍有114个原矩阵point未覆盖；下一阶段优先考虑2048输入的16/32/64请求、
128输入的其余低/高验证形状，再评估6/12/24/48请求网格。不为这些剩余点交付执行命令，
待第一阶段的实际覆盖、耗时和结果再选择有界子集。完整矩阵和性能比较均未授权自动启动。

## 最小实现及验收

仅改host入口、清单选择、报告和测试；无Core/custom op、模型、confidence分配或退出实现改动。
新CLI只允许冻结B64/输入/capture/采样次数、原数值/owner/FULL与命名退出预算组合；
operator-capture、write-timeline、额外退出观察/gdb均拒绝。现有数值切点没有增加。

- 预检重新受限展开已通过模型归档，重算原正式报告；不执行归档脚本，不重跑22/3/23局部测试。
- 打印并保存coverage-plan和来源回执，运行相关CPU入口回归，零失败/跳过后才初始化模型。
- 每点结束立即检查数值信息可用且有限、实际输入/输出长度、请求归属及FULL样本；失败即停止后续点。
- 原有retained/progress/first-error保存机制继续保留此前通过point、失败点及原始数据。
- 新报告使用 `planned_points_generation_complete`，不把12点标成“十点通过”；独立保留已关闭的prior_baseline。
- 命名预算、原始退出码、严格日志扫描、残留和整体检查不放宽；生成通过而cleanup失败仍分别报告。
- performance_eligible=false；不写可用cost-profile，也不报告吞吐收益。

## 唯一服务器命令

将交付完整SHA填入PLUGIN_SHA。保留当前CANN/custom OPP环境、输入manifest和已通过模型归档。
Core继续精确71d2…且remote=rzwang。新目录保存；严格选项仅在子Bash，父shell接收退出码。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
test "$(git rev-parse HEAD)" = "$1"
bash tools/dspark/run_dspark_functional_coverage.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  rzwang
BASH
then echo '第一阶段调用完成；核对覆盖、FULL、退出及整体报告'
else rc=$?; echo "第一阶段失败 rc=$rc；保留已有通过结果，不追加运行"
fi
```

外层为 `dspark-functional-coverage.*`，内含原样模型归档；自动保存日志、真实PIPESTATUS、
JUnit、精确点清单、来源/基线、原始样本、实际覆盖、关闭耗时/退出码、报告及SHA256。
如需受控停止，在本次日志显示的内层 `SERVER_RESULT_DIR` 下创建 `STOP` 文件；由原supervisor处理并保留失败。
回传一个外层 `dspark-functional-coverage.*-evidence.tar.gz` 及SHA256。
本地只执行CPU/mock及脚本验证；新12点真实NPU功能覆盖PENDING。原十点已通过状态不受影响。
