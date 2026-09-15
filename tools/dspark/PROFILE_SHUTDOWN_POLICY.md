# DSpark 正式退出预算：dspark-profile-25s-v1

## 剩余问题与工程决策

剩余的工程问题是：正式入口仍使用 Core 默认5秒 worker 宽限，早于这组模型的自然退出。
本轮没有证据证明可消除的重复释放、诊断对象保活或多余等待，不猜测修改析构顺序。
引入显式、有界、独立命名的 `dspark-profile-25s-v1`，直接配置 Core 已有等待，
不依赖 exit-observation 的额外20秒等待。原默认方案命名为 `profile-default-5s`；默认值和历史失败报告保持不变。

下一轮通过可关闭**该命名配置、原B64单引擎前十点**的数值/归属/FULL及前端、worker自然退出验收。
它不验收原5秒预算，不证明具体原生析构函数、所有运行均无阻塞、真实64并发或长上下文覆盖，
不生成可用成本表或性能结果。

## 最新归档独立审计

已读取两份附件，核验SHA256及外层内嵌模型归档逐字节哈希一致：

| 归档 | SHA256 |
| --- | --- |
| dspark-exit-observation.cY8yC0P3 | 695445a070b106f971765681d72c9945a80417a930547577f1228a892aac807c |
| dspark-large-batch.Lxqwijf8 | 4041bbcc74936f6594e820e270f8d4735f916a5dbf2e5ed01cfc69f76a7aa7ac |

Plugin `a0b1048f2ce23626d7df26b89a04025763527af3`，Core
`71d2c1c436eba894a8e9eeb2c5af17e05cb42970`。在本地调用正式 `model_report`，
重算十点原始文件哈希、26请求各512tokens、数值/归属/FULL，与归档报告完全一致。
26项host测试零失败/跳过。重算独立退出报告为 `NATURAL_EXIT_NO_DEBUGGER`，
debugger_enabled=false、attachment_count=0、no_debugger_control_valid=true。
[审计摘录与原始文件锚点](PROFILE_Lxqwijf8_AUDIT.json) 保存真实退出码、回收、阶段时间及PIPESTATUS。

| 分项 | 本次已有结果 |
| --- | --- |
| 原场景数值、归属、FULL | PASSED_THIS_RUN；实际并发1/2/4请求 |
| 前端输出任务关闭 | 成功；无未完成请求、既有错误或取消异常；EngineDeadError未复现 |
| 无gdb延长预算自然退出 | 八worker真实退出码均0且回收；无TERM/KILL、超时、强制清理 |
| 前端/残留/日志 | cleanup.success=true；末尾RESIDUAL_PROCESSES=[]，ERROR_COUNT=0 |
| 返回码 | 原子进程、扫描、generation、报告、外层均0 |
| 原预算、新命名正式预算 | 都未由此历史运行验收；历史overall_pass=false不改写 |

worker清理12.762401秒，前端完整清理15.629096秒。各rank shutdown返回后至首次观察退出
上界8.4485–9.4588秒。该运行未持续阻塞至强杀；这些是父进程轮询上界，不是某个原生函数耗时。
具体退出尾部开销仍UNKNOWN。模型数值与退出分别保留已有通过证据。

## 5秒的来源及实际调用链

冻结Core源码：

- `vllm/envs.py` 注册 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`，默认字符串`5`，
  注释为engine/worker process shutdown timeout；是可配置保护值。所审项目规范未要求5秒退出SLO。
- `vllm/v1/executor/multiproc_executor.py:MultiprocExecutor._ensure_worker_termination`
  在关闭death-pipe后，以这个配置给所有worker共享一次等待；超时才TERM、共享4秒、KILL。
- `vllm/v1/engine/async_llm.py:AsyncLLM.shutdown` 将显式timeout传给MPClient；
  `core_client.py:MPClient.shutdown` 将其传给manager。其后的`BackgroundResources.__call__`
  也使用上述Core环境默认值；正常显式路径此时manager已经停止。没有修改这个调用链或Core。
- 原profile前端`ProfileFailureGuard`使用12秒内层、4秒收尾、16秒外层；supervisor失败宽限24秒。
  旧exit-observation检查Core必须为5，是为保留诊断对照，不是模型正确性约束。

## 正式配置和边界

| 阶段 | 原默认 | dspark-profile-25s-v1 |
| --- | --- | --- |
| worker自然退出共享宽限 | Core默认5秒 | Core显式25秒 |
| worker TERM / 回收 | 4秒 / 共享1秒 | 不变 |
| EngineCore/前端内层 | 12秒 | 36秒 |
| 前端外层（含输出任务关闭） | 16秒 | 40秒 |
| supervisor首错后宽限 | 24秒 | 48秒 |
| 模型运行上限 | 3600秒 | 不变 |

依据：无gdb实测worker12.762秒，25秒保留约12.24秒余量；前端15.629秒，36秒保留约20.37秒余量。
这是有实测依据的保守配置，未据单样本推断高分位耗时或所有场景保证。
36秒内层涵盖25+4+1以及EngineCore余量；外层再留4秒；supervisor48秒从首错开始计时，
不在内层观察结束前强杀。supervisor另保留原有最多5秒退出组观察、5秒TERM、5秒回收；
从运行上限触发算起，控制流程约不超过3600+48+15秒（另有轮询/OS调度误差）。
报告和归档耗时另计。它不是硬实时承诺；任何预算耗尽、强制信号或未知退出码均失败。

`shutdown_policy.py`集中定义该策略。`run_large_batch.py`仅为新模型子进程设置现有Core变量，
不修改调用shell或其他任务环境；在Core env缓存/模型初始化前设置。
`profile_engine_kwargs`和EngineCore里的ProfileMultiprocExecutor核验实际Core accessor为25；
不匹配立即失败。没有新增环境变量、模型操作、设备同步、Core/custom op修改。

前端、worker、supervisor保存命名预算；worker额外保存实际Core值。旧退出步骤/引用清理/共享回收保持。
新策略不进入observe_workers、不运行gdb预检、不附加ptrace，额外观察等待为0。
既有Python退出步骤记录仍有host开销。逐rank回收UTC给出退出上界，不声称内核退出瞬间。

## 验收门槛与产物

`model-acceptance.json`分开输出：生成完成、数值/归属/FULL、八rank真实退出码、cleanup、
命名预算证据有效性、`named_budget_acceptance`、`overall_pass`。`original_budget_acceptance=NOT_EVALUATED`。
历史报告不含该策略的实际证据，不能重算成正式通过。

新策略除原数值/owner/FULL和严格日志扫描外，要求：

- 前端36/40、worker实际25、supervisor48及命名预算相符，无额外观察/native回执。
- output-handler关闭成功；前端loop已关闭，无清理/记录错误。
- 八rank各有真实0退出码与共享回收记录；无强制退出或超时。
- supervisor原始返回0、无信号/首错、owned process group为空；日志扫描与child返回均0。
- 模型子进程退出后，再保存资源空闲检查和 `b64-residual.json`；失败不覆盖此前错误。

保留现有严格扫描，不依据日志时间或cleanup阶段过滤EngineDeadError。
生成通过而cleanup失败时数值通过记录仍在，命名预算和整体失败。

## 一次服务器验收

使用交付的完整SHA作为 `PLUGIN_SHA`。命令不加载第二个引擎，不重跑已通过22/3/23项局部测试；
仅先执行相关host预算回归，再启动原前十点。Core显式remote=rzwang、精确71d2…，
不永久改写origin；保留当前CANN/custom OPP。不得使用旧版本脚本解释新参数。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
test "$(git rev-parse HEAD)" = "$1"
bash tools/dspark/run_dspark_model_acceptance.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  rzwang
BASH
then echo '调用完成；核对命名预算、数值与整体报告均通过'
else rc=$?; echo "验收失败 rc=$rc；保留证据，不追加实验"
fi
```

新 `dspark-model-acceptance.*` 外层目录只复用现有脚本的日志/归档传输，**不启用退出观察运行时**。
它包含外层日志、PIPESTATUS、状态、原样内层模型证据；内层包括精确source/Core remote、原始十点、
关闭阶段时间、回收退出码、supervisor、post-run资源门槛、报告与每阶段PIPESTATUS。
自动归档并生成SHA256；原始阶段失败码优先于导出错误。回传一个外层归档及其SHA256即可。

只安排这一轮。NPU正式命名预算验收PENDING；若通过，可结束当前配置的完整模型验收阻塞。
原5秒预算不自动变为通过，原生尾部具体函数仍UNKNOWN，不开始B128/B256、成本表或性能比较。
