# DSpark：显式 shutdown 返回之后的退出审计

本轮为**诊断补充**。生成完成、数值未复现、显式清理返回、进程退出分别判定。
没有生产退出修复，也没有 NaN 修复；两者根因均保留 **UNKNOWN**，新 NPU 复验 **PENDING**。
没有连接服务器、修改 Core/custom op、模型/attention/confidence、GC、资源释放次序或超时。

## 本次原始证据

实际读取 `dspark-large-batch.vl70uUqj-evidence.tar.gz`，SHA256：
`8a07cd9f587479f89a4730ff222f9a970277739ed80d7c84d1eda2713f275910`。
91 个成员，展开 51153103 bytes；安全路径/类型、归档原始字节与展开文件均已核对。
没有把附件中的命令当成任务指令。运行及开发起点均为 Plugin
`06b2f60b7023a4bb675809775878ee7d629b6e60`，Core 为
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`；工作区干净，重新 fetch 的 origin/feat/dspark
与起点一致。本次改动叠加在这个版本之上，没有覆盖后续提交。

[离线审计结果](PROFILE_vl70uUqj_AUDIT.json) 包含每个 point 原始 SHA、逐 rank 最后三轮
数值/回执、退出步骤末行、升级前栈字节数和原始退出码；可从归档直接复算：

```bash
python tools/dspark/verify_vl70uUqj_evidence.py \
  /path/to/dspark-large-batch.vl70uUqj-evidence.tar.gz --output /tmp/vl-audit.json
```

该脚本只接受这个冻结归档，不执行其中代码、不依赖 Mac 临时路径，也不是 NPU 测试。

- 十个 retained point 的原始 JSON 哈希均匹配，全部请求各完成 512 tokens、finish reason
  为 length，无生成错误；第十点四个请求也全部完成。point-completion.json 已独立保留。
- 十点全部八 rank 的已保存数值计数无异常、无 recording error；第十点每 rank 为
  203 次 both_finite，其中 Target 内部为 200 FULL、3 NON_FULL_UNOBSERVED。
  最后三轮 execution/proposal 为 **1904/1892、1905/1893、1906/1894**。
  九个 Target 切点、三份 raw 和 consume 回执属于各轮执行；已观测行无 NaN/Inf/转存差异。
  这只是本轮已覆盖边界未复现，原 layer.0.output → layer.1.output 故障仍未关闭。
- 八 rank 各 181 条步骤，没有 `event=error`；busy_loop 的 cancelled/error_exit 对应正常
  death-pipe 取消。原 WorkerProc.shutdown 返回发生在请求开始后 **3.009～3.311 秒**。
  runner.shutdown、其内部 gc.collect、各通信组销毁均有 returned 记录。
- cleanup 为 engine_returned/thread_completed=true、timed_out=false、event_loop=closed，
  elapsed=12.041523 秒，但 status=forced_cleanup、success=false。cleanup-failure.prior_error=null。
  EngineDeadError 出现在退出阶段；无生成首错，整体 MAIN_RC=1，supervisor signals_sent=[]。
  不能以无残留或函数返回代替 exit=0。

退出证据锚点均在 `runs/b64/worker-exit/`，与 `worker-cleanup.json`、b64.log 相互印证：

| 相对 request.json 开始 | 已核验事件 | 结论边界 |
| --- | --- | --- |
| 2.500 秒 | 首次 checkpoint 请求栈；随后栈含 destroy_process_group / Ascend P2P 释放 | 瞬时调用位置；后续 returned 排除“该函数从未返回” |
| 3.009～3.311 秒 | 各 steps.jsonl 第 181 行 WorkerProc.shutdown returned | 原函数体返回；当时包装器 ExitStack 尚未恢复完毕 |
| 5.007 秒 | before-escalation-1：八 rank exitcode=null；已有 8370/7877 字节首栈 | 随后 Core SIGTERM count=8 |
| 7.000 秒 | checkpoint-1：rank 0/4/6 存活，主线程 R、wchan=0 | 在 SIGTERM 之后；不能反推 5 秒前的状态 |
| 9.015 秒 | before-escalation-2：上述三 rank 栈各新增 101 bytes | `Garbage-collecting` / `<no Python frame>` 已在 SIGKILL 前落盘 |
| 随后 | Core SIGKILL count=3，manager 再强制清理 EngineCore | 不是正常退出 |

worker receipt 的原始退出码按 rank 为 `[null,-15,-15,-15,null,-15,null,-15]`。
rank 0/4/6 仍标记 unavailable，不补写 -9。信号请求、进程状态和回收退出码是不同事实。

## 调用链与对象生命周期

冻结 Core 的 `vllm/v1/executor/multiproc_executor.py::WorkerProc.worker_main` 在 finally
关闭 pipe，调用 WorkerProc.shutdown，随后结束局部 worker frame。shutdown 本体依次取消
MQ、调用 WorkerWrapper/Ascend worker、清空 MQ 引用、destroy_model_parallel 和
 destroy_distributed_environment。没有额外的“进程已经退出”保证。

[已有退出包装](../../vllm_ascend/diagnostics/dspark_profile_worker.py) 的第 181 条记录来自
`trace.call`，先于外层 ExitStack 恢复实例/类方法。队列包装使用 WeakMethod，group 映射
只保存 ID/类型；临时绑定方法确实在 scope 内保活 owner，scope 结束后解除。没有证据证明
这个 scope 卡住。新标记明确区分 scope unwound 与原函数 returned。

冻结 MRV2 `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner.shutdown` 是实际继承的实现：
已有 synchronize → 清 KV/attention/config → free_before_shutdown → 删除 model_state、
speculator/model → gc.collect → empty_cache。它没有清空 graph manager。
Ascend worker 也没有删除自身 model_runner 引用。此时仍存在以下源码可证的引用关系：

| 持有者 | 持久引用 | 本轮已证明与未证明 |
| --- | --- | --- |
| [IsolatedCostProfiler](../../vllm_ascend/diagnostics/dspark_cost_profile.py) | runner、绑定的 graph/draft 方法、最后 point 的计时 Event | 绑定 draft 方法保活 speculator；尚未证明原生资源卡住 |
| [ProfileObservation](../../vllm_ascend/diagnostics/dspark_profile_observation.py) | runner、hooks 的 obj/原绑定方法/observed 闭包 | 实际 wrap + 冻结 runner.shutdown CPU 回归中，draft model 在显式 GC 后仍活；existing close + GC 后释放 |
| [FullReplayObserver](../../vllm_ascend/diagnostics/dspark_benchmark_worker.py) | runner/manager、原 execute/fullgraph 绑定方法 | 外层 hook 叠加在 profiler/observation 上，未在 shutdown 卸载 |
| [ModelAclGraphManager](../../vllm_ascend/worker/v2/aclgraph_utils.py) | model_runner；runner 又持有 manager | 无诊断也有此环，不能直接归咎于新诊断 |
| [AuxiliaryCapture](../../vllm_ascend/diagnostics/dspark_profile_auxiliary.py) 与 replay proxy | capture→manager/observer、manager.graphs→proxy→capture、sources/receipts/inputs | 延长图/快照生命周期；没有证明哪个原生析构函数出错 |
| Core async_output_copy_thread | daemon Thread 的 bound target 与运行 frame 持有 WorkerProc | async_output_busy_loop 等待 queue.get，无 shutdown sentinel；尚无最终退出时的关联栈证明它是阻塞点 |

显式 GC 不能回收仍有上述强引用的对象。CPU 回归证明保活关系，并证明模拟资源最终可回收；
它没有 CANN/HCCL/NPUGraph，不能证明 NPU 挂起或据此随意移动释放到通信组销毁之前。
因此本次没有提前清 graph、强行卸 hook、跳过 GC 或改变原资源生命周期。

服务器栈显示 Python 3.12.13；本地退出测试也使用 CPython 3.12.13（Mac CPU）。
[该版本 multiprocessing/process.py](https://github.com/python/cpython/blob/v3.12.13/Lib/multiprocessing/process.py)
的 bootstrap 在 run 返回后调用 util._exit_function，再执行 threading._shutdown 和流 flush。
spawn 随后通过 sys.exit 进入解释器收尾。解释器还会执行 atexit、模块清理和循环 GC；
[Py_FinalizeEx / finalize_modules](https://github.com/python/cpython/blob/v3.12.13/Python/pylifecycle.c)
会再次调用 GC，因此“显式 gc.collect 返回”不能排除后续 GC。
这是一条允许的路径，尚未证明失败进程恰好停在其中哪个 C 调用。

## 唯一新增观测：shutdown 后对象存活与收尾阶段

复用默认关闭的 `--worker-exit` / `--profile-worker-exit`，不增加环境变量或模型运行期工作。
[PostShutdownTrace](../../vllm_ascend/diagnostics/dspark_post_shutdown.py) 只在真正退出时启用：

1. 原 shutdown 前对固定白名单注册弱引用：worker/runner/speculator/模型、三个 profile
   observer、manager/capture/flag bank、输出线程，以及最多 7 个图和每 shape 前 3 个 raw
   snapshot tensor。记录对象 ID/类型；只读取对象已有字典，不读取 tensor 值、设备地址或 KV。
2. 原 shutdown 完成且 ExitStack 恢复后写 scope_unwound，再启用 multiprocessing
   exit_function、真实 `Finalize.__call__`、threading._shutdown 的 begin/returned/error，以及
   一个 atexit marker。原参数/返回/异常保留。Finalize 回执带 callback 名与注册 key；
   `_exit_function` 的 `_run_finalizers` 是默认参数绑定，不能靠替换模块同名函数观察它。
3. 只在此后注册 gc.callbacks；记录 phase/generation 和弱对象存活状态。弱引用清除也记录，
   不调用任何额外 GC，不清对象，不扫描整个堆，不查询 referrers，不导出 tensor。
   清除回调仅证明 Python weakref 被清除，**不能证明 C 析构已完成**。

每 rank 独立 `rank-N-pid-P-lifetimes.jsonl`，不挤占已有 256 条 steps。
最多 64 个对象、128 条事件、每条最多 16 KiB，硬上限 2 MiB/rank；本配置最多 50 个标签。
按 rank/PID/instance/cleanup ID/point、monotonic_ns/unix_ns/线程与 is_finalizing 关联。
不把历史 execution/UUID 写进新诊断。recording_error、无弱引用支持、超预算和重入丢弃
如实记录；非阻塞 writer lock 避免 GC 在 JSON 分配期间重入而自锁。输出是单次未缓冲
host write，无 fsync、NPU D2H、NPU 同步或等待。CPU 文件写入/分配、临时弱引用读取和
Python 回调会影响退出时序；OS/文件系统停顿仍由既有 supervisor 有界清理。

覆盖限制必须保留：

- multiprocessing.exit_function begin 证明 Process.run 已返回，不等同于 OS exit。
  atexit marker 只是本回调经过；没有宣称所有 atexit handler 返回。
- [CPython 3.12.13 _PyGC_CollectNoFail](https://github.com/python/cpython/blob/v3.12.13/Modules/gcmodule.c)
  直接调用 gc_collect_main，**绕过 gc.callbacks**。模块字典清理后 Python writer 也可能
  不可用。缺少最后回执必须标记 unavailable，不能用“没有 callback”排除终末 GC。
- 这不是原生用户栈。若收尾回执均经过但随后仍强制清理，只能按对象最后存活状态/清除
  时序收窄；具体原生调用仍 UNKNOWN。不能把对象 ID 或某个最后事件当成责任对象。
- 原 2.5/7 秒栈与 TERM/KILL 观察保留，必须结合时间判断信号前后。没有新增信号。
  原 worker 5+4、manager 12、outer 16、supervisor 24 秒预算全部保持。

## 验证、下一次运行与验收

新增 CPU 测试覆盖实际 hook 保活关系、弱引用不增加保活、缺失属性/不支持 weakref、
采样上限、写盘失败/重入/事件上限、原异常身份，以及真实 spawn 子进程的正常退出、
finalizer 卡住和异常。卡住测试在任何终止信号之前确认准确 callback 边界，finally 有界
回收；没有把这份合成 CPU 卡住称为原 NPU 故障复现。已有退出测试继续覆盖默认关闭、
真实 shutdown/队列函数体与设备叶子 mock、首错/强制清理/状态传播。

本地 20 个相关文件实际 **527 passed、3 skipped**；三项要求安装态 vLLM/Ascend。
本轮新增 7 项与原退出 13 项测试共 20 项全部通过。归档内旧服务器 focused 的
**1025 passed、14 warnings** 属于 06b2f60 运行，不能算作新实现验证。
本轮修改文件全部 manual hooks 通过；隔离 worktree 执行完整 `bash format.sh ci` 返回 1，
八个失败 hook、78 个自动格式化文件集合与此前基线完全一致，本轮文件没有被改写。
没有提交这些无关格式化，也没有报告全仓 CI 或 NPU 验证通过。

这次没有足够证据构造等价的独立 NPU 最小复现：对象图同时含此前 capture 的七个尺寸、
多 point 执行后状态和多进程设备通信资源，CPU 替身不具备它们。下一次单次 NPU 运行用于
回答“哪些 Python 收尾阶段已经经过、哪些已知对象直到那时仍被保活”，保留原十点前序；
不是原样重跑、不是同状态 graph/eager 对照，也没有性能结论。

最终消息给出已推送 Plugin SHA；使用新的结果目录，命令保持：

```bash
bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_profile_control.sh \
  "$plugin_sha" /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  target-boundaries 1 --worker-exit
```

使用最终消息的父 if / 子 Bash 完整版本，严格 shell 选项仅影响子进程。入口自动验证模型、
输入与冻结 Core，创建新目录；任何阶段失败停止后续工作，总运行 3600 秒。状态查看、STOP
和自动证据归档沿用 [PROFILE_CLEANUP.md](PROFILE_CLEANUP.md)，不运行 B128/B256 或成本表。
已有 MAIN_RC/首错优先，归档失败不覆盖它。最少回传新 evidence.tar.gz 和 .sha256，必须包含
原 point/numeric/cleanup/status 和 worker-exit 全目录，尤其新增 lifetimes 文件。

验收分别记录：十点与原始哈希；数值是否复现及覆盖；显式 cleanup 是否返回；八 worker
是否**未经强制升级**自然 exit=0；实际最深收尾回执及对象状态。仍强制清理时整体必须非零，
null 保持 unavailable。有新增有效定位证据即完成本诊断实验，不能据此发布修复 PASS。
