# 正常关闭输出任务与无 gdb 退出对照

## 本轮已有证据（不追改历史结果）

已实际读取两份归档并核验 SHA256：

- `dspark-exit-observation.A9vFCFDv-evidence.tar.gz`：
  `6dbf6c48e3a0e07e983598d6abea7b2eb9cf571d2634d3de3c556d1a0385ec36`。
- `dspark-large-batch.be36p2mQ-evidence.tar.gz`：
  `61cbfc3cdaab3d769994541daea3cb8a0739a1f1dc382bdc0374f5a5b8f7c3bf`。
  外层内嵌的模型归档与此文件哈希完全相同。

Plugin `7c33932af172ecd7618234b095e3fc4edf8d7036`，Core
`71d2c1c436eba894a8e9eeb2c5af17e05cb42970`。独立调用正式 `model_report`
重算十点 raw 哈希、请求完成和数值/FULL门槛，与原 model-acceptance.json 完全一致。
[审计摘录及原始事件锚点](PROFILE_be36p2mQ_AUDIT.json) 保留如下分项结果：

| 项目 | 原始结果 |
| --- | --- |
| host 测试 / 真实 gdb 预检 | 29 passed、零跳过；预检 passed |
| 生成 / 数值 / 归属 / FULL | 十点26请求各512 tokens；PASSED_THIS_RUN |
| worker | 八 rank 退出码均0且已回收，无worker TERM/KILL |
| cleanup / 残留 | success=true，无超时/强制升级，最终 RESIDUAL_PROCESSES=[] |
| 耗时 | worker 20.718441秒，前端23.552212秒，包含调试器影响 |
| 原生栈 | 两次附加rank0后超时且脱离；无实际回溯帧；UNAVAILABLE |
| 独立退出报告 | NATURAL_EXIT_DEBUGGER_AFFECTED |
| 模型/监督进程 | runs/b64.pipestatus=0 0，supervisor raw_returncode=0，无监督信号 |
| 日志扫描 / generation | 原stages.rc=0，但扫描抛AssertionError；generation.pipestatus=1 0 |
| 报告 / 外层 | acceptance-report=1 0；独立报告=0 0；外层driver=1 0 |

两次SIGKILL的对象是本次gdb子进程，不能当作worker被杀。原生析构具体根因仍UNKNOWN。
带暂停的耗时不能直接用于生产预算。原预算失败与这次扫描失败均保留，不将旧结果改为整体成功。
服务器实际Python3.12.13、Torch2.10.0+cpu、torch_npu2.10.0.post2；原CANN/OPP身份及路径保留。

## 源码责任与最小修正

冻结Core真实调用链：

1. `vllm/v1/engine/async_llm.py:AsyncLLM.shutdown` 先执行 `engine_core.shutdown`，最后才
   `cancel_task_threadsafe(output_handler)`。
2. `vllm/v1/engine/core_client.py:MPClient.shutdown` 先标记engine_dead、关闭manager，再清理资源。
   `AsyncMPClient._ensure_output_queue_task` 的socket任务取消分支会向输出queue放入EngineDeadError；
   其他IPC异常也会送入该queue。
3. `AsyncMPClient.get_output_async` 取出异常并抛出；`AsyncLLM._run_output_handler` 捕获、打印
   traceback并调用output_processor.propagate_error。即使请求已完成，该后台consumer仍继续等待。
4. `performance_stream.StreamingEngine.shutdown` 原来在独立线程中运行Core shutdown，所属loop继续
   调度输出consumer，形成producer先关闭、consumer尚未取消的窗口。归档中异常出现在
   MPClient complete附近，回溯到上述get_output_async/consumer；所有请求与进程回执均正常。

CPU回归实际执行冻结Core的socket任务、get_output_async、异常转换、consumer和shutdown方法体，
用受控CPU传输/manager复现旧顺序的EngineDeadError日志；同一路径提前排空consumer后不再产生此日志。
这是前端关闭调用顺序缺陷的可复现证据。归档没有区分当时queue异常究竟来自取消分支还是其他关闭IPC，
不据此声称已识别原生析构函数。无需修改Core或SWA修复。

只对诊断前端的profile关闭路径，新增 `_stop_profile_output`：

- 在所属event loop检查真实output_processor未完成请求数、最后batch完成记录、guard pending/首错、
  consumer状态及socket producer状态。缺失或矛盾失败；不把待完成请求当作正常排空并丢弃。
- 正常空闲时，先cancel consumer并有界等待、取得实际任务结果，再启动原Core shutdown线程。
  Core producer在排空期间失败也会被保留，不能以消费者已取消掩盖尚未读取的IPC错误。
- 非正常状态保留首错，交由原异常清理；任务取消错误、超时、后续loop任务错误均导致cleanup失败。
  不把已返回的consumer（Core内部可能已吞获并传播异常）当作健康状态。
- `output-handler-shutdown.json` 保存开始/完成、未完成数、最后batch状态、原首错、取消结果及异常。
  cleanup.json嵌入该回执。最多1秒，计入原前端40秒截止时间，使用既有4秒收尾余量，不另加退出窗口。
- 原日志扫描保持不变。新增stages.log_scan_rc/log_scan_error，把子进程rc和扫描结果显式分开。
  前面阶段失败导致未扫描时，log_scan_rc=null。原日志原样保留。

非profile前端路径、模型、confidence、Core/custom op、通信/stream、退出默认预算均未改变。
此项是诊断前端生命周期修正；worker原预算自然退出并未修复或验收。

## 相同预算、无 gdb 模式

显式 `--profile-exit-observation --profile-exit-no-debugger`（要求既有worker-exit/target-boundaries）。
外层入口对应 `--no-debugger`。无新环境变量；不永久改写Core origin。

- worker额外20秒父进程轮询 + 原5秒宽限，共25秒；TERM4秒，共享回收最多1秒。
- EngineCore/前端内层36秒，前端外层40秒，supervisor收尾48秒，均与上轮相同。
- 不启动gdb、不查询gdb是否安装、不ptrace、不执行原生附加预检。本次只跑相关host回归，零失败/跳过后
  记录 `native-preflight-disabled.json`，校验原Core宽限仍5秒，随后才启动一个原配置模型引擎。
- observation.json明确 `debugger_enabled=false`、`attachment_count=0`、
  `native_sampling=disabled_by_configuration`、native_samples=[]。
  报告native_coverage=DISABLED_BY_CONFIGURATION，绝不标为采样成功。
- 原有worker-exit步骤、弱引用/父进程回收及Python栈记录保持（非ptrace）；无新增原生栈。
  这仍有原host日志与轮询开销，不是零诊断性能测试。
- 每rank保存shutdown返回、最后存活、首次观察退出及join时间。前20秒是轮询上界；之后由原Core返回后的
  join/状态读取给出更粗上界，均不是内核退出瞬间。未取得退出码仍为null。

模型运行上限仍3600秒，supervisor另有48秒收尾及原5秒TERM/5秒回收；证据压缩时间另计。
达到预算失败就归档，本轮不自动追加实验。全部0退出只证明**无gdb延长预算对照**完成，
original_budget_acceptance/正式验收继续NOT_EVALUATED，overall_pass不改为true。

## 唯一服务器命令

使用交付完整SHA替换PLUGIN_SHA；保留当前CANN/custom OPP环境。
新目录保存，旧归档只读；Core精确71d2…且显式remote=rzwang。
严格选项仅在子Bash内，父shell接收退出码。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
bash tools/dspark/run_dspark_exit_observation.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  rzwang --no-debugger
BASH
then echo '无gdb延长预算对照调用完成；不判原预算验收通过'
else rc=$?; echo "对照失败 rc=$rc；保留首错及全部证据，不追加实验"
fi
```

仍为单B64引擎、原前十point/指定长度、每请求512 tokens、原capture和数值/owner/FULL门槛。
不启用operator-capture/write-timeline，不跑B128/B256或成本表/性能比较。

回传一个外层 `dspark-exit-observation.*-evidence.tar.gz` 及SHA256。它包含原样内层模型归档、外层
日志/PIPESTATUS；内层含生成数据、output-handler回执、无附加配置回执、worker轮询/reap/真实退出码、
supervisor、model-acceptance及独立exit-observation-report。报告分别保存数值、worker、前端清理、
原首错、子进程和扫描状态；报告/导出自身退出码见外层对应PIPESTATUS（报告生成后才可取得）。
有需要时通过新模型目录的STOP文件受控停止，不按名称批量kill。

验收需同时检查：26请求完成、数值/归属/FULL通过、八worker全0、无超时/强制/残留、
无新运行期或关闭异常、子进程/扫描/报告/导出均0、debugger=false/附加0且采样关闭。
这仍不是原预算验收。任一项缺失/矛盾则失败或UNAVAILABLE；不按日志时间豁免错误。

本地实际测试与检查见 [验证记录](PROFILE_EXIT_NO_DEBUGGER_VALIDATION.json)。
真实无gdb模型对照 **NPU PENDING**；未取得新服务器结果前不报告正常析构生产预算或原生阻塞根因。
