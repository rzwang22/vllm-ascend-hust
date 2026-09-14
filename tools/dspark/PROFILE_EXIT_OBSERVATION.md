# Worker 自然退出：有界原生观察对照

## l4CYeLJv 已核验结果

实际读取 `dspark-large-batch.l4CYeLJv-evidence.tar.gz`，SHA256：
`21054737b97607bc783039e02edb7d96d5e635362b981b1de8a5497d375aafea`。
Plugin `200ba86a76531017d56e5c27692fda39598eedac`，
Core `71d2c1c436eba894a8e9eeb2c5af17e05cb42970`。
[原始数据审计摘录](PROFILE_l4CYeLJv_AUDIT.json) 保存独立重算的 model report、各 rank
关闭/弱引用清除事件、真实回收结果和 Core 来源。重算报告与归档报告完全一致。

- 入口 JUnit 85 passed，无 failure/error/skipped；十点26请求各512 tokens。
- 各点 raw 哈希、数值/请求归属/FULL 样本和首点真实回执通过，继续为 PASSED_THIS_RUN。
- 八 rank 的两项 close 和外层清理包装均返回；profile_observation、speculator、
  被跟踪 target_model 的弱引用在显式 shutdown 阶段清除。
- worker 清理开始 UTC 17:14:07.948984；WorkerProc.shutdown 在
  17:14:11.052984～11.277690 返回。原5秒预算只余约1.67～1.90秒。
- UTC 17:14:12.954844 宣布 SIGTERM 八进程，16.972898 宣布 SIGKILL 两进程。
  最后前端强制清理 EngineCore。早期通信组销毁随后有 returned；晚期 GC 栈在 TERM 后。
- 共享1秒 join 实际耗时0.686312秒，全部 reaped；rank0–5 为 -15，rank6–7 为 -9，
  没有 null。强制清理、overall_pass=false；最终 RESIDUAL_PROCESSES=[]。

引用清理与回收修复已有实际效果。本轮排除了“这些 close 未返回”和“仍因未回收而缺退出码”，
没有排除其他持久资源。弱引用清除不代表原生析构完成，reaped 不代表正常退出。
**正常但较慢的析构与真正阻塞仍 UNKNOWN；原预算失败保留。**
数值/FULL 结论只覆盖原 B64 引擎的实际1/2/4并发十点，不扩展为64并发或全部长上下文。

## 默认关闭的选项和唯一判别目标

新增 `--profile-exit-observation`，只允许配合既有 `--profile-worker-exit` 和
`target-boundaries` 的 B64 profile。未启用时不附加调试器、不延长预算。
保留所有引用清理、真实回收、SWA修复、数值/owner/回执门槛及原输入顺序。
不新增模型切点、NPU张量统计/拷贝、同步或资源清零，不修改 Core/custom op。

[实现](../../vllm_ascend/diagnostics/dspark_exit_observation.py) 在 Core 关闭 worker death
writers 后、原 `_ensure_worker_termination` 前增加一次共享20秒观察；随后仍委托原 Core
的5秒宽限、TERM4秒和KILL。没有修改或冒充 Core 环境变量的值，预检和构造时要求原值恰为5。
不是把固定预算永久改大，也不是新的生产退出策略。

| 阶段 | 原预算 | 本诊断预算与作用 |
| --- | --- | --- |
| worker 正常观察宽限 | 5秒 | 额外20秒 + 原5秒，共25秒；八 rank 共享 |
| 原生采样 | 无 | 第8/14秒；每轮最多1个仍活 rank，单次4秒 + 脱离/回收0.5秒，计入前20秒 |
| TERM等待 | 4秒 | 原4秒，约第25秒后才开始 |
| KILL后父进程回收 | 共1秒 | 原共享1秒，最坏约第30秒结束 |
| EngineCore/前端内层shutdown | 12秒 | 36秒，为worker最坏30秒留6秒 |
| 前端外层 | 16秒 | 36+4=40秒 |
| supervisor失败收尾 | 24秒 | 48秒，覆盖取消最多2秒 + 外层40秒 + 余量6秒 |

正常故障传播路径中外层不会在上述内层观察结束前升级。超出各预算仍严格失败并执行原清理。
模型运行上限仍3600秒；supervisor触发后最多48秒收尾、5秒TERM和5秒回收，
受管运行最坏约3658秒（另有轮询/进程调度误差及证据压缩时间）。
模型加载前原生工具预检单独限30秒，超时再给2秒终止余量。

父进程每约50ms观察所有退出码；记录实际最后存活与首次观察到退出的时间。
采样期间不轮询，时间区间会扩大。20秒后退出的进程用原Core返回后的join/状态读取时间
作更粗的上界，明确区别于内核退出瞬间。join前后UTC和状态读取UTC也保存。
`exit-observation-report.json.rank_timings` 对齐每rank shutdown返回、最后存活、退出观测及join。
原 weak lifetimes-state、steps和Python栈继续保留，不用Python栈冒充新增原生栈。

## 原生工具门槛与观察影响

服务器在加载模型权重前运行新增host回归，要求零失败、零跳过；不重跑85项原入口测试，
也不重跑22/3/23项局部生命周期测试，只重新读取其已有审计归档。
然后 [原生预检](exit_native_preflight.py) 创建一个临时CPU进程，导入真实torch/torch_npu，
记录实际Python/Torch/torch_npu版本；不创建模型、NPU tensor、通信组或graph。
对该进程验证与worker相同祖先关系下的gdb附加/回溯/脱离权限。
再让已确认附加的调试器停留至超时，终止调试器，检查TracerPid、必要的SIGCONT恢复，
并确认临时进程heartbeat继续推进。预检子进程最后受控销毁，不是模型worker验收。

需要Linux `/proc`、gdb及实际ptrace权限。禁用gdb初始化脚本、自动加载脚本和debuginfod网络访问；
保存gdb版本/命令、映射文件、原生栈、工具退出码和暂停上界。
没有工具/权限、库导入失败、不能安全脱离或子进程不恢复时，在加载模型前失败并归档。
本地Mac没有验证Linux ptrace；服务器权限 **PENDING**。

具体缺项见 `native-preflight/preflight.json`、normal/timeout.stack.txt 和 native-preflight.log。
若是权限拒绝，替代路径是管理员提供同一环境下具备ptrace权限的执行账号/容器，再运行此预检；
若无法开放权限，需另行提供可用的CANN原生回溯或core-dump采集设施。当前入口不修改
ptrace_scope、不自行安装工具、不假设替代设施可用，也不继续加载模型或降级为Python栈。
预检不证明所有完整worker库/析构栈都能在4秒内展开；运行期超时仍明确为UNAVAILABLE。

正式观察最多2次附加，不选固定历史rank；每次取仍活的最小rank，若进程已退出就跳过。
不附加已被跟踪或已停止的进程；校验PID start ticks，超时只终止自己的gdb。
若原本运行的同一进程留下T/t状态且已无tracer，才发SIGCONT并核对恢复。
gdb继承本次受管session，外层紧急清理能覆盖它，不留下另一个独立调试会话。
调试器/写盘异常不阻止原Core清理，错误写入worker-cleanup并阻止错误发布成功。

每次最多16帧/线程；栈和maps各最多2MiB，最多两次。输出截断、超时、无法脱离均标为unavailable，
保留部分内容。图、模型tensor与KV均不导出。gdb暂停会改变选中worker及其通信对端的时间，
**不能简单减去暂停耗时得到无干扰析构耗时**；也不能把两张相同栈直接当成永久死锁证明。

## 独立结果解释

`model-acceptance.json` 保留数值/FULL，增加 exit_observation=true 和
original_budget_acceptance=NOT_EVALUATED_BY_THIS_DIAGNOSTIC；overall_pass始终为false。
若延长窗口下全部正常清理且数值通过，运行命令可以返回0，表示诊断调用完成，**不是正式验收通过**。
强制退出和原生成错误仍返回原非零码，报告/导出不替换首错。

| 独立报告status | 能证明什么 | 下一步 |
| --- | --- | --- |
| NATURAL_EXIT_NO_DEBUGGER | worker在调试器介入前实际全0退出，且无强制信号 | 用阶段耗时判断是否超过原5秒，并提出生产预算依据；不自动采用本次25秒 |
| NATURAL_EXIT_DEBUGGER_AFFECTED | worker全0退出，但已有调试暂停 | 支持可完成析构；分析原生调用和干扰，不能视为纯耗时基准 |
| NATURAL_EXIT_OBSERVATION_UNAVAILABLE | 退出码支持自然退出，但观测过程不完整 | 保留unavailable，不能宣称未受调试器影响 |
| FORCED_OR_INCOMPLETE | 延长窗口仍未取得全部正常退出 | 结合强制信号前原生帧、maps和两轮变化定位操作；慢与阻塞仍需源码/调用证据 |

worker_natural_exit_observed 与 frontend_cleanup_success 分开。
八worker全0但EngineCore/前端失败时保留worker结果，整体仍失败。
任何结果都不产生性能/成本结论；本轮是诊断交付，生产退出修复与NPU复验仍PENDING。

## 服务器单次入口

将 PLUGIN_SHA 替换为交付的完整SHA。只需以下一组命令；严格选项限于子Bash。
原 CANN/custom OPP 环境保持，Core精确71d2…且remote显式rzwang，不永久修改origin。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
bash tools/dspark/run_dspark_exit_observation.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json rzwang
BASH
then echo '诊断调用完成；查看独立退出报告，不判原预算验收通过'
else rc=$?; echo "诊断退出码=$rc；原错误、工具预检与退出证据均需保留"
fi
```

[外层入口](run_dspark_exit_observation.sh) 新建 dspark-exit-observation.*，保存fetch、checkout、
完整driver日志、真实PIPESTATUS和status。模型阶段仍由原入口新建dspark-large-batch.*；
单B64引擎、原十点与指定长度、26请求各512 tokens、原capture sizes和数值/FULL/owner检查，
operator-capture/write-timeline关闭。新模式不自动运行其他batch、成本表或性能比较。

外层把本次内层证据归档原样收入 `model-evidence.tar.gz`，与外层日志一同生成一个主要
`dspark-exit-observation.*-evidence.tar.gz` 及SHA256。旧归档不覆盖。
内层包含原生预检、JUnit、worker-exit/native、各rank生命周期、join/真实退出码、
model-acceptance和独立exit-observation-report、supervisor、原始生成数据。
主回传物：**外层 evidence.tar.gz、对应SHA256**。

另一终端可tail外层driver.log；实际模型目录写入model-source.txt（结束后）且启动时打印
SERVER_RESULT_DIR。受控停止用该模型目录的 `STOP` 文件；不按进程名称批量kill。

本地测试和项目检查见 [验证记录](PROFILE_EXIT_OBSERVATION_VALIDATION.json)。
