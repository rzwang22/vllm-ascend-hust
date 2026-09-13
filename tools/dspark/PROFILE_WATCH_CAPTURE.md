# qe7Lb9dE：修正 capture 未执行时的 watch 取证

这是 replay/watch 工具修复，不是原 layer1 NaN 的生产修复。
基线 `beaf9459c2aab5b23242dd571d0ff178b5c03b83`，Core 保持
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`。没有模型、Core、custom op、
原始capsule、stream等待策略或退出预算修改。

## 证据等级与源码依据

本地未找到 `dspark-slot-controls.qe7Lb9dE-evidence.tar.gz` 或对应日志文件，
服务器路径在本机不可访问；**未核验该归档哈希和内容**。
用户报告输入哈希通过、watch warmup通过、capture快照值异常、四组真实对照未启动。
这些是服务器报告，不是本地重新执行结果；capture分配区域的大数不能归因于生产attention写坏KV。

实际下载并读取了 PyPI `torch_npu-2.10.0.post2-cp312-cp312-manylinux_2_28_x86_64.whl`：
SHA256 `4e970b4ab06f46ee7a5ae50d07007222e1fb42ddc691a9e9bbc74d037b6a0dd8`。
其 `version.py` 记录 git_version `8751b36d5d6959e499e6bf6530c1928060ced030`；
`torch_npu/npu/graphs.py` SHA256
`5ed98b6b5bbbaa162513801e877a24671b58ec2b8eaba3c8d446357efe0e5dc1`。
这证明所审发行包身份，不证明服务器安装文件与此wheel完全相同；新预检保存实际wrapper文件哈希和版本。

- wheel的 `graph.__exit__` 只执行 `capture_end`、退出dispatch mode与stream上下文，不调用replay。
- 对应提交的 [NPUGraph.cpp](https://github.com/Ascend/pytorch/blob/8751b36d5d6959e499e6bf6530c1928060ced030/torch_npu/csrc/core/npu/NPUGraph.cpp)
  `capture_end` 调用 `AclmdlRICaptureEnd` 并标记图可执行；`replay` 才调用
  `AclmdlRIExecuteAsync`，使用当前stream。读取的cpp SHA256为
  `7880f4fa1d2a02d0498435ba7fd6488d6154e51a74783682adfd3911572238a5`。
- 旧工具退出capture后直接 `observe("capture")`，没有执行凭据便读取clone存储；
  测试又按阶段序号机械期待guard=1。这是已确定的工具时序缺陷。
  源码与报告支持“clone尚未执行”，但本地无NPU，不能宣称完成服务器级因果验证。

不能用缺少stream wait、生命周期破坏或allocator大数直接解释此失败。新测试将首先隔离这些方向：
原生counter在capture内递增，capture结束后**测试专用**caller等待capture stream，再读取已有counter。
若仍为0且首次replay后为1，即使建立了局部依赖capture仍未执行；不需要读取未初始化clone证明这一点。
如果这一步不满足，就保存实际值并失败，四组对照不启动。

## 修正后的执行与产物契约

`operator_replay.execute_replays` 复用原有warmup/图捕获/三次replay顺序，没有额外attention调用。
`capture` 只生成状态 `captured_not_replayed`、`snapshot_valid=false`，不读取clone；
首次有效图快照明确标为 `replay-0`。一次成功ACLGraph实验为：

| 项目 | 次数/状态 |
| --- | --- |
| Python invoke（warmup + capture录图） | 2；不等于设备已执行次数 |
| warmup有效调用 | 1 |
| capture录图 | 1；无有效快照，不计完成调用 |
| 显式replay提交/完成 | 3/3 |
| 有效before/after快照 | 4对：warmup、replay-0/1/2 |

每次局部watch D2H仍为2KB，现在4次共8KB。两份1KB图内clone保持独立存储，
D2H后CPU快照拥有自己的stack存储，下一次replay不能覆盖历史数据。没有新增全局同步。
原torch_npu图上下文本身的同步行为保持不变；测试中的局部stream依赖只用于验证capture语义，
没有以猜测性依赖改变真实重放路径。

`slot-watch.json` schema=2包含阶段状态、有效性、active_phase、Python调用次数、replay提交/完成计数、
完成快照数以及before/after哈希与字节差异；`slot-watch.pt`只保存真实已完成快照。
计数是工具执行记录，不冒充模型epoch或设备回执。`result.json.slot_watch`现在是上述对象，
不再只是旧records列表。CLI异常时保存已获得的部分证据及原错误，导出异常不覆盖原异常。
原始文件、物理页号、重复页一致性、格式、覆盖、NaN及owner检查均保留。

## 无权重预检与失败证据

原NPU预检用例名称不变，内部顺序如下：

1. **原生add/clone小图**：先在独立counter副本上预热并保存实际值；主counter从0开始且不因预热修改。capture内before clone、counter加1、after clone。
   capture结束只读取有定义的counter，clone记录为不可用；三次replay各检查before/after/counter。
   replay完成后再次核对全部CPU历史。未初始化clone不被当作有效数据，不用填充期望值伪造结果。
2. **真实SWA/ACLGraph watch**：原合成guard初值不变；仅在已完成D2H记录后提交一次guard递增。
   每条快照同时记录当时独立维护的期望值和完成调用编号；capture没有guard更新。
   检查4对快照、3次replay、逐轮变化、before/after逐字节相等及复用后的CPU历史。

`capture-semantics.pt`保存各阶段counter/有效快照、期望值、指针、运行库身份、提交/完成计数、
活动阶段及异常；`native-watch-evidence.pt`保存SWA快照、guard期望值、更新计数和输出。
两处都在finally落盘，assert失败也有证据；独立 `slot-watch.json/pt` 保存部分watch。
预检成功仍必须检查JUnit为1 passed、零失败零跳过，且实际取证文件和执行计数齐全，才能进入四组对照。

CPU回归模拟“capture分配但不执行”，通过真实工具调度入口证明不会读取capture快照；
注入第二次replay失败，核对提交数与完成数不同、已有历史落盘；正式CLI异常路径也验证原错误和部分证据。
CPU测试不验证Ascend capture实现。真实NPU两个步骤都 PENDING，本机不能替代服务器执行。

[本地验证记录](PROFILE_WATCH_VALIDATION.json)：Python3.12.13/Torch2.10.0，59 passed、7 skipped。
两个真实CPU dispatcher/AOT参数仍通过；7项跳过均因缺少torch_npu/NPU。
本次文件manual hooks通过。隔离worktree运行全仓 `bash format.sh ci` 返回1，8个失败hook及
78个自动格式化的无关文件与上轮基线一致，没有自动修改本次文件；未报告全仓CI通过。

## 服务器一次命令

沿用已有激活环境和原OPP，填入交付完整SHA。该命令先只执行修复后的watch预检，
通过后才执行原四组无权重单卡对照。新结果目录、输入哈希/只读、OPP身份、
每组180秒及TERM后15秒KILL、PIPESTATUS、失败即停和EXIT归档机制不变。

```bash
if bash -s -- PLUGIN_SHA <<'BASH'
set -euo pipefail
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
git fetch origin feat/dspark
git merge --ff-only "$1"
exec bash tools/dspark/run_dspark_saved_operator.sh "$1" --slot-controls
BASH
then
  echo 'Watch and slot experiments completed; inspect numerical outcomes separately.'
else
  rc=$?
  echo "Failed: rc=$rc; preserve the printed evidence directory."
fi
```

回传新 `dspark-slot-controls.*-evidence.tar.gz` 和 `.sha256`。预检失败不进入真实四组，
成功也只证明工具可取证；单槽位NPU因果验证、实际写入者、生产修复、worker自然退出分别保持未关闭。
不重载完整模型，不生成成本表或性能结论。
