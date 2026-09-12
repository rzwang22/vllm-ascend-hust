# DSpark：po6aUaSP worker 退出阶段审计与观测

本轮交付是**退出阶段诊断补充**，没有已证明的 worker 根因或生产退出修复。
原 Target NaN 本次未复现，根因仍为 **UNKNOWN**。新诊断的 NPU 复验 **PENDING**。
没有连接服务器、改 Core/custom op、改超时、调整退出顺序或增加模型数值观测。

## 原始归档核验

独立读取 `dspark-large-batch.po6aUaSP-evidence.tar.gz`，核验 SHA256：
`0a36fc1e0de998bd7abe200a6563e9868bfcff691d8998af1d4afc408b82ca89`。
61 个成员，展开 49557417 bytes；检查成员路径和类型后安全展开，逐文件比对 tar 原始字节。
没有执行归档中的指令。逐 point 哈希、八 rank 三轮结果和原始退出回执存入
[PROFILE_po6aUaSP_AUDIT.json](PROFILE_po6aUaSP_AUDIT.json)。

归档 plan 的 Plugin 为 `c44d97b13f3348645b10770baf1a08502a34df58`，
Core 为 `897306c43bf800e2480cb5c0f3e2da408d85a2fd`，与审计起点的干净本地工作区一致。
服务器旧版本 focused 实际报告 **1012 passed, 14 warnings**，不是本轮新增实现的验证。

- 十个 retained point 的原始 SHA256 全部一致，所有请求均生成 512 tokens，
  finish reason=length，request/streaming error 均为空。第十点四个请求全部完成。
  新 point-completion 回执的十点、哈希、token 数及数值摘要也与原始文件一致。
- 每个 point 的八 rank numeric/auxiliary/target 计数均无已记录异常，D2H 次数全部完成，
  recording_error 为空。第十点每 rank 有 203 次 both_finite、200 次 FULL 和
  3 次 NON_FULL_UNOBSERVED；后者不是完整内部数值覆盖。
- 八份 latest 三轮为 execution/proposal **1907/1893、1908/1894、1909/1895**。
  九个 Target 切点、raw/persistent/consumed auxiliary 40/41/42 与 head 均有限；
  三组 raw/consume 及九个内部回执均属于各自 execution，无 missing/转存差异。
  latest 文件 SHA 与第十点的 local evidence 引用匹配，没有首 NaN 文件。

## 本次失败的真实时序

| UTC | 事件与证据 |
| --- | --- |
| 14:00:58.604400 | frontend cleanup 开始；十点此前已完成 |
| 14:00:58.607394 | executor cleanup 开始，等待 worker |
| 14:00:58 | 日志显示 rank 0 的 DeathPipeMonitor 收到 EOF 并准备终止队列 |
| 14:01:03.613242 | 5 秒 worker 宽限耗尽，executor 对八个存活进程发 SIGTERM |
| 14:01:07.623354 | 再等 4 秒后，对四个剩余进程发 SIGKILL |
| 14:01:07.638089 | executor shutdown 返回并保存 worker 回执，耗时 9.0307 秒 |
| 14:01:10.616630 | manager 强制清理尚未退出的 EngineCore |
| 14:01:10.622342 | AsyncLLM shutdown 返回，内层调用耗时 12.0173 秒 |
| 14:01:10.634462 | 外层观察到线程完成，累计 12.0301 秒 |
| 14:01:10 | output_handler 因 Core 已退出报告 EngineDeadError；整体因 forced_cleanup 失败 |

八个 worker 的父进程回执如下。它是**发出 kill 后立即取得的状态**，不是所有进程已 reap
的证明；冻结 `_ensure_worker_termination` 在最后 kill 后没有额外 join。

| rank | PID | raw_exitcode |
| --- | --- | --- |
| 0 | 106510 | null / unavailable |
| 1 | 106560 | null / unavailable |
| 2 | 106637 | -15 |
| 3 | 106712 | -15 |
| 4 | 106795 | -15 |
| 5 | 106868 | null / unavailable |
| 6 | 106945 | -15 |
| 7 | 107022 | null / unavailable |

不能把 null 补造为 -9，也不能判正常退出。`cleanup.engine_returned` 和
`thread_completed` 为 true，`timed_out=false`、`event_loop=closed`，但
`status=forced_cleanup`、`success=false`、`graceful_worker_exit=false`。
`cleanup-failure.prior_error=null`、points_status=completed；engine-failure.phase=cleanup，
supervisor raw_returncode=1、signals_sent=[]，MAIN_RC=1。
这证明前次外层计时/loop 状态修复本次正常记录了返回，**没有证明 worker 正常退出**。
EngineDead 是退出阶段的后果，不是此次生成首错。包内没有独立残留进程枚举。

## 冻结源码真实调用链与缺口

Core 均指上述 897306c 版本：`vllm/v1/executor/multiproc_executor.py` 中：

1. `MultiprocExecutor.shutdown` 在 EngineCore 原线程关闭各 worker death_writer，
   再调用 `_ensure_worker_termination`；它没有先发送退出 RPC。
2. `WorkerProc.monitor_death_pipe` 的 DeathPipeMonitor 线程阻塞在 recv。
   EOF 后先设置 shutdown_requested，再依次调用 input/response MQ.shutdown。
   rank 0 的日志证明进入了该分支，不能证明八 rank 均完成了队列取消。
3. `MessageQueue.shutdown` 置 shutting_down，再调用 SpinCondition.cancel。
   对本地 reader，这通过取消 socket 唤醒 poll；acquire_read 随后抛 cancelled。
   busy_loop 的 dequeue 在 try 块外，所以取消异常向 worker_main 传播并进入 finally。
   remote/overflow socket recv 的路径不同；新 setup 回执保存真实 reader/writer 分支。
   现有日志不能证明某 rank 的 poll/recv 已返回，也不能证明跨线程 cancel 没有阻塞。
4. `WorkerProc.shutdown` 先对两个 MQ 再次调用 shutdown，然后
   `WorkerWrapperBase.shutdown → NPUWorker.shutdown`，清空 MQ 引用，销毁 model parallel
   和 distributed environment。新诊断不消除第二次 cancel、不改变线程或顺序。
5. [Ascend worker](../../vllm_ascend/worker/worker.py) 依次 flush attention path probe、
   ensure_kv_transfer_shutdown、可选 profiler、weight transfer engine、model runner shutdown。
   它不调用另一套 worker 退出实现。
6. MRV2 [NPUModelRunner](../../vllm_ascend/worker/v2/model_runner.py) 继承冻结 Core
   `vllm/v1/worker/gpu/model_runner.py::shutdown`：原有 accelerator.synchronize，清空
   KV/attention groups/config，free_before_shutdown，释放 model_state/speculator/model，
   gc.collect 和 accelerator.empty_cache。这些原有设备操作可能等待，但本次没有其
   输入状态或线程栈，不能认定 synchronize 或析构为根因。
7. group 实际类型受 [GroupCoordinatorPatch](../../vllm_ascend/patch/worker/patch_distributed.py)
   影响：释放 communicator、按 registry 引用计数释放 HCCL groups、销毁独有 group，
   最后 CPU group。registry.release 的 process-group 销毁在 registry lock 外；
   clear 只清 registry metadata。destroy_distributed_environment 的实际 alias 是否
   包含 registry-clear wrapper 取决于导入 binding，新观测围绕真实消费 alias 执行。
   源码存在 Ascend 额外 group，但仅凭函数名或未调用某 helper 不能证明此次设备释放卡死。

冻结 worker_main 的 Python SIGTERM handler 在 shutdown_requested 已置位后不会再次
raise SystemExit；实际 -15 本身不能反推出 handler 当时的状态、执行线程或终止原因。
新 setup/EOF 保存 Python handler 描述，procfs 保留原始 SigCgt/SigIgn 等状态；Python
getsignal 不能完整反映原生库重置的处理器。不会替换 SIGTERM/SIGINT handler 来强行退出。

原 profile executor 只在 EngineCore 包装 super().shutdown，未替换 death-pipe、MQ、
worker class、worker main 或 worker shutdown，也未在模型运行期增加清理线程。
frontend 原有 daemon cleanup thread 调用 AsyncLLM.shutdown；worker 的清理仍在其
原 main thread，死亡通知仍在 DeathPipeMonitor。现有包没有证明包装改变 worker 退出
顺序或造成挂起。EngineCore 在 executor 返回后的 9–12 秒具体停点也仍未覆盖。

## 默认关闭的局部诊断

新参数 `--profile-worker-exit` 只允许隔离 B64 target-boundaries profile。
control 入口的第五个参数 `--worker-exit` 将它传到底层。默认不选诊断 worker、不安装
这些包装、不注册诊断信号、不创建 watcher、不采集 procfs 或栈。

[ProfileNPUWorker](../../vllm_ascend/diagnostics/dspark_profile_worker.py) 继承原 NPUWorker。
它在独立 worker 进程内安装有限的退出包装，原函数、参数、调用次序、返回值与异常均保留。
Core 文件、worker_busy_loop 内容、队列取消协议及信号终止逻辑不变；busy_loop 只有一次
外层包装，不逐 RPC/token 执行观测。设备与模型初始化完成后、READY 前注册 faulthandler。

- 每 rank 最多 256 条未缓冲 JSONL，包含 PID/rank/worker instance、cleanup request ID、
  point、UTC/monotonic 时间、线程名/ident/native ID 和 begin/returned/error。
  setup queue binding 没有 cleanup request；真正退出开始后读取同目录 request.json 关联。
  覆盖 EOF、两个 MQ.shutdown/cancel、busy_loop 退出、WorkerProc/Wrapper/Ascend worker、
  profiler/transfer、model runner、**已有**同步/缓存释放/GC、实际 group/registry/PG 销毁。
- 队列包装通过 weak method 避免增加自引用环；group 通过其类型包装，映射只保存对象 ID，
  不额外保活已清理 group。模型 runner 包装只在原 cleanup 期间存在，结束后恢复绑定。
- [ExitWatch](../../vllm_ascend/diagnostics/dspark_worker_exit.py) 仅在 executor shutdown
  开始时启动，沿原 Core 5 秒宽限在 2.5 秒、沿 SIGTERM 后原 4 秒窗口在 7 秒发起采集。
  不修改这些期限。每次只给父进程持有的、尚无退出码且 PID/rank 注册匹配的 worker
  发 SIGUSR1，由 faulthandler 输出最多 100 threads × 100 frames 的 Python 栈。
  若已有 Python handler、注册失败、PID/rank 不符或 worker 已退出，则不发送。
- 两次采集同时保存 procfs status/stat，以及最多 128 个线程的 status/wchan/kernel stack，
  每字段最多 4096 bytes；权限拒绝/进程消失如实记录 unavailable。Mac CPU 测试的 /proc
  缺失也如此处理。原生 C/CANN/HCCL 用户栈并未导出，Python 栈只能显示进入它的调用位置。
- 在 Core 原有 WARNING、实际 SIGTERM/SIGKILL 循环**之前**，保存轻量进程状态和栈文件
  当前字节数，不再请求栈、不等待回复。信号 sent 是请求记录；只有升级前已存在的栈
  字节才证明该时刻已有内容。相同 rank 的两份 native dump 顺序在同一文件中，不能把
  请求时间当作精确采集完成时间。无栈或进程不可读不等于正常退出。

开销限定为 setup 小文件、退出步骤 JSONL、两次 host-only watcher 采集和最多两次
SIGUSR1/rank；原终止信号不变。日志/文件系统调度会改变退出时序，升级前轻量 I/O
也可能延迟发信号；原 12/16/24 秒与 supervisor 的有界退出仍保留。
SIGUSR1 也可能中断原有阻塞 syscall，因此有观测时正常结束不能单独证明无观测时的退出缺陷已修复。
没有 NPU D2H、额外同步、模型层切点或退出后 RPC。步骤文件完整不证明进程最终 exit=0；
若停在 Python 解释器 finalizer、原生调用或观察范围外，仍需结合可用栈继续缩小范围。

## 测试与下一次唯一运行

CPU 测试运行实际 Core death monitor/busy loop/MQ shutdown/WorkerProc/Wrapper shutdown，
实际 Ascend worker 和继承的 MRV2 shutdown 函数体，mock 设备与资源叶子；验证清理顺序、
异常传播、队列弱引用、不增加设备调用、默认关闭、参数传递及退出前栈采集。
真实独立子进程测试 SIGUSR1 栈在强制升级前可用，并保留升级前 null、实际 SIGTERM 后 -15。
这些是 CPU/source/mock 测试，不是安装态 Ascend 或 NPU 验证。

本地 19 个相关文件实际 **520 passed、3 skipped**，三项跳过要求安装态 vLLM/Ascend；
本轮新增 13 项退出测试全部通过。修改文件 manual hooks（含 shell/Markdown）全部通过。
隔离 worktree 的完整 `bash format.sh ci` 返回 1，八个失败 hook 和 78 个自动格式化文件
与此前基线完全一致，本轮文件未被自动改写；未提交无关格式化，也不报告全仓 CI 通过。

下一次保留原单引擎、十点顺序、模型/输入 gate 与 target-boundaries 1；只多启用退出诊断。
最终交付消息提供完整 plugin SHA，Core 继续冻结 897306c。使用该消息中的完整子 Bash
命令，control 调用仅为：

```bash
bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_profile_control.sh \
  "$plugin_sha" /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  target-boundaries 1 --worker-exit
```

入口创建新结果目录，不覆盖旧证据。状态查看、3600 秒总运行限制、STOP 文件受控停止和
导出失败保留 MAIN_RC 的流程沿用 [PROFILE_CLEANUP.md](PROFILE_CLEANUP.md)。不自动运行
B128/B256、成本表或性能测试。本次所有产物仍为 performance_eligible=false。

最少回传新 evidence.tar.gz 和 .sha256；其中必须包括原 point/numeric/cleanup 文件及
新增 `runs/b64/worker-exit/` 全目录：request、各 rank ready/steps/stacks、parent checkpoints
和 before-escalation 文件。若正常快速退出，watcher 尚未到期便停止，没有栈是预期情况；
若强制退出却没有可用栈/步骤，按覆盖不足报告，不能据此定位 producer。

验收分开：生成完成与原始 SHA；数值有限性及覆盖（未复现不等于修复）；正常退出或原始
强制信号/退出码；最后一个 returned 与第一个未完成 begin、对应线程和升级前可用栈。
若所有 WorkerProc.shutdown 都返回但进程仍不退出，区间移到解释器/原生 finalizer；
若某个步骤 begin 无返回并与栈一致，才据该具体阶段提出下一步最小修复。
