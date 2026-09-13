# 6D9EWxZk：单槽位因果成立，追踪完整模型写入区间

本轮是**诊断补充**，实际写入者 UNKNOWN，生产修复未完成。Core 固定
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`；所审 Plugin 基线
`d90993191fb4df6c1f2761c575acaab77d85b226`。不修改 Core/custom op、模型计算、
confidence、退出顺序或超时。没有在本机执行 NPU 或连接服务器。

## 已独立核验的证据

实际读取 `dspark-slot-controls.6D9EWxZk-evidence.tar.gz`，SHA256
`83c47d3330329ba0a7fe83b5a36bbb513ad9e2a5f4357b313322788174d1a742`，
67 成员、展开39002064 bytes。归档内 pytest-current 绝对符号链接没有被跟随。
[可重算审计](verify_6d9ewxzk_evidence.py) 与 [实际结果](PROFILE_6D9EWxZk_AUDIT.json)
使用正式 validate/reference、`weights_only=True, map_location="cpu"`；unsafe globals 为空。
核验输入哈希、实际 counterfactual/replay/watch 张量、intervention 字节范围、metadata、
12份 PIPESTATUS 均为 `0 0`、原生预检 JUnit 1 passed 零跳过及实际 guard 快照。

| 原生 saved metadata + ACLGraph，各三次 | 实际结果 |
| --- | --- |
| original1802 | 有限 |
| original1803 | row0/head1、3、7 NaN |
| 1803 仅恢复历史槽位为1802值 | 无NaN，最大绝对值12.625 |
| 1802 仅注入1803异常槽位 | row0/head1、3、4、6 NaN |

两个反事实只改变同一1024字节槽位内977字节，其他输入、布局和metadata相同。
旧运行位置95，block123/offset31，cache相对范围 `[4062208,4063232)`；
保存pages范围 `[97280,98304)`。16对实际warmup/replay前后快照均逐字节不变。
capture counter=0、无有效clone快照；三次显式replay逐次更新。服务器实际版本为
Python3.12.13、Torch2.10.0+cpu、torch_npu2.10.0.post2；共同加载的extension/OPP文件哈希
与采集身份一致，不同加载集合在JSON分别保留，不能据此证明二进制与源码完全对应。

这证明所选历史槽位变化足以触发/消除该调用异常，并且被观察的单算子调用没有改写该槽位。
它没有确定完整模型writer，也没有排除未采集区域、其他操作、页映射/生命周期或异步依赖问题。
本无权重实验未验证完整模型worker自然退出。原数值故障与cleanup分别保持未关闭。

## 源码收敛与缺少的决定性证据

- `worker/v2/attn_utils.py:_allocate_kv_cache/_view_dsv4_cache` 按 `KVCacheTensor.shared_by`
  复用真实backing，packed视图另有byte offset/block stride。相同storage不等于同时拥有同一物理页。
- `ops/dsa.py:dsa_forward` 是已有 opaque PrivateUse1 调度边界。内部调用
  `attention/dsa_v1.py`，候选写操作包括声明可写cache的scatter以及compressor的state_cache。
  `csrc/torch_binding.cpp`、scatter host API要求实际dtype匹配，并传stride；
  FP32位视图约2.97不能证明实际FP32 writer。
- 冻结Core `kv_cache_manager.py:allocate_slots` 在分配前remove_skipped_blocks；
  `single_type_kv_cache_manager.py:SlidingWindowManager` 按完整页释放。
  window128/block32下，**若**真实computed=222，跳过95token，仅移除前2页；
  **若**真实computed=228，跳过101token，可移除前3页。这个条件差异不是实际提前释放证明。
  `sched/async_scheduler.py:_update_after_schedule` 有投机预记账，scheduler输出处理有reject回退；
  worker的position222不能代替传给manager的实际computed。
- capsule只有attention调用时输入，没有上一次采集后至下一次采集前的free/allocate、
  实际源buffer或写入回执。继续重复四组对照无法补齐这些数据。

下一次只增加两条关联时间线：真实scheduler accounting/页拥有者变化，以及与目标cache存储区间
相交的可写操作前后1024字节。`schedule_id`随现有SchedulerOutput pickle及worker浅复制保留，
与worker execution/proposal/request建立显式关联。缺失关联或回执报告 UNAVAILABLE。

## 新局部诊断及边界

显式 `--write-timeline` 必须与 `target-boundaries 1 --attention --operator-capture` 一起使用，
默认关闭。CLI只把最后一个指定point的请求标记为诊断请求；采样参数数值不变。
plugin的 `WriteTimelineScheduler` 继承冻结AsyncScheduler，实例包装原有方法，转发原参数和返回值。
记录实际computed/lookahead、placeholder、num_tokens_with_spec、reject前后状态、group、
request blocks、free前后ref_cnt、pool分配/释放顺序。free迭代器随Core消费，未提前耗尽。
共享group由实际allocation/shared_by选择；catalog保留所有group、packedoffset和blockstride。
此包装是诊断CPU开销，不是Core生产修复。

worker从本轮query length=1请求中选CPU seq最短的一行；CPU字段仅用于选请求。
实际设备query start/position及所选层block table决定绑定：取当前SWA窗口中首个完整页的末位置。
只在该位置已经成为有效历史时采集；同时校验request/position/当前epoch。
例如旧position220/221/222都会选择逻辑95，但新运行不写死95、页123、请求UUID或execution。
窗口滑动后绑定按实际页表更新，页重绑定单独记状态变化。

`SlotWriteTimeline` 在原 `dsa_forward` opaque边界内部使用TorchDispatchMode：
仅对schema声明可写、真实tensor字节范围与目标cache allocation相交的调用取前后字节。
其他目标层仅可能贡献共享backing的writer候选，没有新增整层数值扫描。
保存算子schema名、层标签、源/目标dtype/shape/stride/storage offset/格式、storage基址、
目标span及与所选1024字节范围交集、capture时真实stream。
strided span包含holes，不能把bounding span当成kernel精确写集合；
索引等小tensor输入保留至多两个、每个16KiB逻辑连续前缀，完整descriptor及截断范围保留。
捕获时的layout ID由实际图内操作写入，避免把warmup地址描述当作replay绑定。

图内前后copy及epoch copy在真实replay执行；capture不读取为有效数据。
每轮reset，target结束即把所有site包复制到独立存储；target/draft/context间另存该单槽位快照。
head之前统一回传并落盘。无当前回执、mapping错误、预算耗尽均不能发布完整覆盖；
缺失site原始包在诊断拒绝前保存。数值首错和后续owner错误保持原有行为。

**覆盖不足仍须解释**：仅观察opaque DSA内schema暴露的可写tensor参数，
不覆盖错误schema隐瞒的写入、private kernel workspace、opaque DSA外的FFN或其他写操作。
这些操作只可能被target/draft间槽位变化区间包围。若多stream并发，前后变化只给候选区间，
不能凭host/capture注册顺序认定具体kernel；没有加入全局同步或关闭多stream。
原有metadata更新与replay依赖保持不变，新copy在相应原stream排队。

## 有界开销与取证验收

- 最多256个site、独立持久buffer总量8MiB；每site两个1KiB值、binding/receipt/layout ID、
  至多两个16KiB源前缀。source原tensor超过64KiB不导出内容。最多128个layout变体/site。
- 所选point每轮额外16字节控制H2D、一个合并writer D2H；实际 `d2h_bytes` 随产物保存。
  这是在既有auxiliary/operator回传之外的一次局部等待。没有逐site host等待。
  对选定容量的图，诊断copy会随所有replay执行（包括前序point），仅最后point启用CPU留存。
  图内gather/copy、临时packet、JSONL写盘和D2H会改变时序，不能生成成本或性能结论。
- 正常仅保留末3轮；首次字节变化/页重绑定/数值异常或覆盖不足时冻结此前2轮与本轮，
  再保留下一轮（若原失败链允许执行）。每rank最多4份pt；catalog独立留存。
- scheduler最多30000事件，worker catalog最多512项；耗尽保存截断/未完成调用并失败，
  不无限增加日志。原异常已发生时，诊断异常附注，不替代原异常。
- `writes-*.pt` 保存当前和不可用site包、源前缀、host快照、身份、schedule_id和graph对象；
  `writes-index.json` 保存首变化/首不可用轮次、文件及当前文件SHA。
  只有完整当前回执、真实页归属与具体操作前后差异相互支持，才能进一步指认writer。
  未发生变化或原NaN未复现均不等于已修复。

## 本地验证与一次服务器入口

见 [验证记录](PROFILE_WRITER_VALIDATION.json)：相关测试198 passed、4 skipped；本次文件manual hooks全通过，
全仓 `bash format.sh ci` 返回1，8个既有hook失败，78个自动修改路径与前轮基线相同，
没有自动修改本次文件，未把无关格式改动带入工作区。
本地CPU检查包含真实CPU dispatcher、
动态query/page绑定、无scalar设备提取、可写alias筛选、独立存储与冻结历史、
缺失/陈旧回执及mapping失败仍落盘、free惰性迭代、原异常保留。
真实NPU opaque dispatch + ACLGraph三轮和安装态Core接口检查在本地未执行。
旧Torch2.10真实CPU dispatcher/AOT测试继续通过；NPU不以CPU通过替代。

必须再进入一次完整模型，是因为现有capsule没有实际writer/free/accounting事件。
这次先执行无权重NPU writer小图及安装态Core实际remove路径预检，2 passed且零跳过，
失败立即停止并归档；通过后完整focused，再同一引擎原前十point（不重复原单卡对照）。
保留已有前三轮全rank回执验收、原输入/模型/OPP、NaN/owner校验、失败即停及退出保障。
以下填交付SHA，在已有服务器环境执行；严格选项只在子Bash，不关闭父交互终端。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
exec bash tools/dspark/run_dspark_profile_control.sh "$1" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  target-boundaries 1 --attention --worker-exit --operator-capture --write-timeline
BASH
then
  echo 'Diagnostic completed; inspect writer evidence and numerical outcome separately.'
else
  rc=$?
  echo "Diagnostic failed: rc=$rc; retain the printed result directory and archive."
fi
```

预检300秒、TERM后15秒KILL；原target诊断supervisor总时限3600秒及原清理预算不变。
用脚本打印的新 `SERVER_RESULT_DIR` 查看 `writer-preflight.log/xml`、`generation.log`、
`status.txt`；受控停止可 `touch "$SERVER_RESULT_DIR/STOP"`，由既有supervisor处理，勿删除旧目录。
主回传产物只有新 `dspark-large-batch.*-evidence.tar.gz` 及其SHA256；内含日志、PIPESTATUS、
预检JUnit/实际pt、scheduler-page-timeline、各rank writes/capsules和原首错/cleanup。
归档失败不覆盖最初退出码。若预检失败，不加载完整模型；若模型再次NaN，优先用对应schedule_id
检查页是否仍被请求持有，再查同一物理区间最早发生字节变化的候选调用及源前缀。
