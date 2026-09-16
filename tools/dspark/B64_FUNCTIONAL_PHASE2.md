# B64 功能覆盖第二阶段：2048 输入与较大并发

第二阶段已通过并独立审计冻结，见 [正式成本与confidence计划](B64_FORMAL_COST.md)。
不重复以下7点，也不新增第三阶段合成功能矩阵；以下保留第二阶段实施与验收依据。

## 第一阶段已通过并冻结

已实际读取 hBoDUBTA 外层和 XM02ngWZ 模型归档，验证两份SHA256与内嵌归档一致性。
重新执行正式受限展开及 `model_report`，结果与原保存报告完全一致；独立核验每点原始文件哈希、
实际输入/输出token、八rank的FULL样本、退出回执、监督进程及各级PIPESTATUS。

| 项目 | 独立核验结果 |
| --- | --- |
| Plugin / Core | c271239bc3bfc104309b61740476df6f2d07c9bc / 71d2c1c436eba894a8e9eeb2c5af17e05cb42970 |
| 外层SHA256 | 6fae3c072c96159efe6cb029191bf607d18c4686cd72d57c553ccd3e05b346f8 |
| 模型SHA256 | ed690d6971ec777af5a43359ae79f9a24c9f082fb7105485d256022c32cb1fed |
| 完成情况 | 12点、328请求各512输出tokens；实际输入128或2048与point匹配 |
| 数值/归属/FULL | 通过；每点每rank各5个匹配target及draft FULL样本 |
| 命名预算/整体 | dspark-profile-25s-v1；PASSED_THIS_RUN；overall_pass=true |
| worker / 前端 | 八worker真实退出码0且全部回收；12.838708 / 16.439346秒 |
| 失败/日志 | 无超时、强制升级、残留或日志异常；各级PIPESTATUS均0 0 |
| 服务器host测试 | 26 passed，零失败、零跳过 |
| 监督进程总时间 | 1286.277346秒；含初始化、capture、退出，不能作为推理性能 |

[审计证据摘录](PROFILE_XM02ngWZ_ACCEPTED.json) 保留逐点哈希、逐rank计数、预算与退出回执。
原十点和第一阶段均保持通过；新阶段失败不追改历史。
原5秒预算未验收、具体析构尾部开销未定位，仍记录为未解决但不阻塞当前命名策略下的功能扩展。

最新 `core-source.json` 记录 rzwang URL 为 `https://github.com/rzwang22/vllm-hust.git`，
source_transport=network_remote，actual/fetched HEAD均为71d2…；origin仍为HUST组织远端。
这是本次归档中的来源回执；此前Ck6iA7rN的 `file://` 回执仍是本地来源，不追改为网络核验。
本地审计不连接服务器；下次继续显式选择rzwang、核验精确SHA并记录实际URL，不永久改写远端。

## 本轮唯一合成扩展清单

7点均由原 `grid(64, [6,12,24,48,96,192,384], [128,2048], 512)` 合法矩阵选出，
保持原顺序和逐请求lengths，无替代点。配置中的128上下文用于保持原矩阵定义，本阶段实际仅执行2048输入。
完整清单见 [B64_FUNCTIONAL_PHASE2.json](B64_FUNCTIONAL_PHASE2.json)。

| 顺序 / point | 每请求输入 | 请求数 / FULL目标并发 | Σell | target query tokens | graph capacity |
| --- | --- | --- | --- | --- | --- |
| 1 ctx2048-n16-t48-balanced | 2048 | 16 | 32 | 48 | 48 |
| 2 ctx2048-n16-t48-skewed | 2048 | 16 | 32 | 48 | 48 |
| 3 ctx2048-n32-t96-balanced | 2048 | 32 | 64 | 96 | 96 |
| 4 ctx2048-n32-t96-skewed | 2048 | 32 | 64 | 96 | 96 |
| 5 ctx2048-n64-t192-balanced | 2048 | 64 | 128 | 192 | 192 |
| 6 ctx2048-n64-t192-skewed | 2048 | 64 | 128 | 192 | 192 |
| 7 ctx2048-n64-t384-balanced | 2048 | 64 | 320 | 384 | 384 |

总计 **288请求、147456输出tokens**，每请求512输出tokens。
`ell` 为指定验证长度0–5，target query量为Σ(ell+1)，capacity为图token容量，均不等于请求数。
实际调度并发须由样本证明；不能将提交64请求或B64引擎容量直接当作全程64并发。
不改变每请求起草K5、TP8+EP、target FULL_DECODE_ONLY、draft eager、原capture列表及采样设置。
输入仍为原tokenizer的 `encode("x")` 首个token重复2048次，不是实际文本质量验证。

## 边界与验收

- 一个B64引擎，模型监督运行上限3600秒（含初始化与capture）。第一阶段约21.44分钟，
  为本轮有界尝试提供参考；更长输入和更高并发不能按请求数线性推算，不承诺一小时必然完成。
- 沿用命名退出策略：worker25秒、TERM4秒、共享回收1秒、前端36/40秒、supervisor收尾48秒；
  额外观察等待0、gdb关闭。超限后原受控清理最多约48+15秒，另计轮询/OS调度、预检和归档时间。
- 每请求从实际 `observed_prompt_token_ids` 验证2048输入、实际输出512；请求归属和原NaN检查保留。
- 每点每rank经过2个warmup后保留5个匹配FULL样本：实际请求数、query lengths、有效token数、
  graph capacity及当前回执均须通过原检查。target/draft分别核验，首点连续回执提前门槛不变。
- 保留原始事件的动态并发、有效token及capacity集合；匹配样本仅证明采样覆盖，不声称每轮并发固定。
- 任一点失败立即停止后续点，保留此前retained、首错、失败点、原始输出与归档。没有自动追加实验。
- 全部worker真实退出码必须0且回收，无超时/强制升级/残留/日志异常，所有阶段退出码通过。
  生成通过但cleanup失败仍分开记录，整体失败；缺失退出码不能补造。
- operator-capture/write-timeline及额外退出观察拒绝启用。无新数值切点、全局同步或模型修改。
  Core/SWA与输出消费者修复均不变；不重跑22/3/23局部测试或已通过完整模型场景。

第二阶段整体通过后，报告 `synthetic_functional_exit_criterion=PASSED_THIS_RUN`，
`real_text_validation_readiness=READY_FOR_SEPARATE_REAL_TEXT_VALIDATION`；
真实文本验证仍为 `NOT_RUN`，需另行定义文本、正确性对照和验收。
**合成功能扩展到此结束，不将剩余107矩阵点作为待完成清单，不自动生成第三阶段。**
本轮不覆盖任意长上下文、全部验证形状、真实文本质量或吞吐收益；performance_eligible=false，
不生成可用成本表，不启动B128/B256或性能比较。

## 唯一服务器命令

将交付的完整提交SHA填入PLUGIN_SHA。保留当前CANN/custom OPP环境和原manifest，
并保留 `/workspace/dspark-results/dspark-large-batch.XM02ngWZ-evidence.tar.gz`，
入口会先核验该归档哈希并重建第一阶段报告，再做本阶段host回归；通过才初始化模型。
这只是离线审计，不重跑第一阶段。Core固定71d2…、remote=rzwang；新目录不覆盖历史。

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
  rzwang b64-functional-2
BASH
then echo '第二阶段调用完成；核对整体报告及真实文本验证准入状态'
else rc=$?; echo "第二阶段失败 rc=$rc；保留第一阶段通过记录，不追加运行"
fi
```

自动保存外层日志、真实PIPESTATUS、JUnit、7点计划、基线/源码来源、原始输入输出、逐rank样本、
数值/归属/FULL、关闭耗时、真实退出码、残留和整体验收报告，导出内嵌模型归档与SHA256。
如需受控停止，在日志 `SERVER_RESULT_DIR` 对应目录创建 `STOP`，按既有supervisor机制保留失败证据。
最少回传一个新 `dspark-functional-coverage.*-evidence.tar.gz` 及其SHA256。
第二阶段NPU验收已由 pJap5lfX / v8vohAeE 归档确认通过；真实confidence调度仍未执行。
