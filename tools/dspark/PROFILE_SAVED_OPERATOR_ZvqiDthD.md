# ZvqiDthD：真实 capsule CPU 审计与单卡重放

本轮是离线审计和重放报告补充，没有生产修复。原数值问题已在归档复现；CPU 算术对照取得了
新的因果证据，但 Ascend 原生算子重放 PENDING，历史 KV 的实际写入者 UNKNOWN。
Core/custom op、模型、confidence、模型内观测、stream 等待和退出预算没有修改。

## 独立核验与可重算产物

实际读取归档 `dspark-large-batch.ZvqiDthD-evidence.tar.gz`，SHA256：
`95ce4baf4c95a6313193f5921fd9437c7280c8cdd11f183037e98abd4d74f281`。
208 个成员、展开 448443829 bytes；归档内容作为数据读取，没有执行其中的指令。
Plugin `7847c94dc8c8895efb0ae22f8f30636990bd12ee`、
Core `897306c43bf800e2480cb5c0f3e2da408d85a2fd` 与 plan 一致。
focused 为 **1141 passed**、PIPESTATUS `0 0`；前九点原始文件与 retained 哈希一致，
请求均完成 512 tokens。第十点输出 **512/397/124/95**，随后失败。

[审计脚本](verify_zvqidthd_evidence.py) 调用正式 replay 的受限加载、validate、restore 和 reference：

```bash
python -m tools.dspark.verify_zvqidthd_evidence \
  /path/to/dspark-large-batch.ZvqiDthD-evidence.tar.gz --output /new/audit-directory
```

[完整机器记录](PROFILE_ZvqiDthD_AUDIT.json) 包含逐 rank/轮次文件哈希、行/head、布局、窗口物理页、
非有限位置、有限误差、历史差异、CPU 对照、原 recording_error、错误事件和 runtime 身份。
脚本另生成 24 份 float64 `reference.pt`。这些是离线证据断言，不是 NPU 测试。

全部 24 份 capsule（rank0–7 × execution1801/1802/1803）通过：

- `torch.load(map_location="cpu", weights_only=True)`；unsafe globals 为空，全部格式字段是普通 int 2。
- 原格式编号、dtype、shape、stride、storage offset、alias 描述保留；CPU restore 核对视图及输入字节。
  已保存完整物理页逐字节匹配，重复页内容一致；实际 NPU 布局恢复仍待验。
  layout/pointer 是图捕获时描述，device layout_id 将它与图内复制关联；当前回执证明本轮复制执行，
  不能把静态地址本身当作当前页所有权或整个设备状态的证明。
- capsule before/after 回执、21 target、4 KV、raw/consume 回执都是当前 execution。
  首点八 rank execution2/3/4 提前验收通过。
- FULL、PREFIX_AND_FULL_GUARD_PAGES、请求映射成立；全部文件哈希还与对应数值记录的 last_sha256 相等。
- 实际三请求顺序为 `batch10-3-92dac912`、`batch10-2-92657419`、`batch10-1-90aab81e`。
  starts `[0,1,5,11]`，query lengths `[1,4,6]`；13 项 padded starts 的其余项都是11，
  12 项 seq_lens 的其余9项是0。1803 的有效 seq_lens `[223,255,530]`。
  11 有效 target 行、capacity12、每 rank Q 为 `[12,8,512]`。静态 descriptor 不是12个实际请求。

## 比“窗口有限”更具体的发现

所有 rank 的 Q、sinks、有效 SWA 窗口 KV 都没有 NaN/Inf；float64 sink-aware reference 有限。
**但这不表示输入数值正常**：1803 的有效窗口起点 position95（block123/offset31）变为极大有限值。
同一请求 1801/1802/1803 的 position 分别为220/221/222，窗口分别为 `[93,221)`、`[94,222)`、
`[95,223)`。历史 position95 在这三轮均被实际语义窗口使用，不是第12行 padding 或窗口外数据。
当前 position222 的写入槽位是 block194/offset30，不是发生历史变化的 block123/offset31。

比较同一请求映射、相同物理页、当前 query 开始之前的重叠历史，1802→1803 **仅 position95 改变**。
另外两个请求的这部分历史没有变化；完整 block table 未变。全部八 rank：

| 项目 | 1802 | 1803 |
| --- | --- | --- |
| 该槽位最大绝对值 | 14.125 | 3.1502703500102506e38 |
| 512 个 BF16 分量 | 正常幅值 | 全部512个分量的位模式改变，仍有限 |
| 1024 bytes SHA256 | f45ea3b1d6209d0a2286c302702a536b5a2794e5803c8be78ff6fc1e3457080a | ba0c0858b77bce653920c041c2706e8639514457189405f372321492d42c949b |

后者按 256 个 FP32 **位视图**解读时最大绝对值为 2.966325283050537，偶数 BF16 分量呈极端幅值，
这是后续检查 FP32 写入/存储复用的具体线索；位视图不是原始 dtype 证明，不能据此指认 compressor、
scatter、某个 kernel 或释放函数。归档未保存那个写入者的实际调用、其他 KV group 的同轮内容或物理别名关系。

原始输出第0行的以下本地 head 各有512个 NaN 分量，无 Inf；其他有效行有限：

| rank | 1801 | 1802 | 1803 NaN heads | 原 capsule 的 CPU FP32 算术 NaN heads |
| --- | --- | --- | --- | --- |
| 0 | 无 | 无 | 1,3,7 | 1,3,7 |
| 1 | 无 | 无 | 2,3 | 2,3 |
| 2 | 无 | 无 | 1,4,7 | 1,4,7 |
| 3 | 无 | 无 | 1,7 | 1,7 |
| 4 | 无 | 无 | 0,6 | 0,6 |
| 5 | 无 | 无 | 3,6 | 3,6 |
| 6 | 无 | 无 | 3 | 3 |
| 7 | 无 | 无 | 0,3,5 | 0,3,5 |

CPU 对照直接用保存的真实 Q/KV/sinks/scale，执行 FP32 QK、sink-aware softmax、WV；
每个 rank 的 NaN head 集合均与归档一致。只在独立内存副本中把上述历史槽位换回1802的值，
其余输入不变，八 rank 的 CPU NaN 全部消失。未改原文件，未将此对照当生产修复，也未在服务器自动运行修改输入的对照。
这证明历史内容变化在该 CPU 算术路径中足以解释 NaN，强烈支持输入幅值导致 FP32 溢出的方向；
还没有证明 Ascend 的内部累加次序、精确溢出指令和原始写入者。

float64 reference 没有溢出，但其部分 head 输出同样达到3.15e38；原始有限 head 中也有这些巨大值，
不能将“有限输出与 reference 误差小”理解为模型正确。
24 个样本有限交集的最大绝对误差约0.02836～0.04093，记录同时给出排除的 NaN 元素数。
保存窗口以外的物理槽位也有 NaN/Inf：例如 rank0 的1803共有407个窗外槽位含NaN、12个含Inf；
1801/1802 也有这类值。它们与有效窗口的极大有限值是不同证据，不证明 kernel 读了窗外值。

## coverage/mapping 全局错误的追溯边界

八 rank 保留原错误：`Operator capsule coverage/mapping unavailable; preserve original failure`。
[save](../../vllm_ascend/diagnostics/dspark_profile_operator.py) 在写文件/索引之后检查以下条件：
seq_lens 非负且不超过采集上限640；所需前缀页有效；device starts 的请求前缀与当前 record 相同。
任一不满足就抛错；[guard](../../vllm_ascend/diagnostics/dspark_profile_auxiliary.py) 只把文本写入
全局 recording_error，没有保存触发 epoch 或是哪一项失败。后续成功保存不会清空该字段。

上一份 clean latest 是第九点 execution1704；现在保留的1801/1802/1803均逐项通过原条件，
所以这些末三轮不是该错误的触发轮。早期 capsule 被三文件 ring 覆盖，文本没有更精确时间戳。
**确切轮次及 coverage 还是 mapping 分支仍 UNKNOWN**；不能补写1793或推断必定是请求退出。
第十点四到三请求转换实际为1794；可能的 seq>640 等条件只是待查项，不足以认定模型 metadata 错误。
本轮未清空该错误、未放宽 validate；使用末三轮是依据独立逐文件校验，不是忽略历史错误。

## 调用路径、二进制与剩余状态缺口

[dsa_v1.py](../../vllm_ascend/attention/dsa_v1.py) `compress_ratio<=1` 分支，在原等待与 scatter 后调用
`_C_ascend.npu_sparse_attn_sharedkv`，参数为cmp_ratio1、mask4、left127/right0、TND/PA_ND。
采集发生在该调用前，raw output 在调用返回后保存。这里仍没有原生kernel内部写入/读取轨迹。
冻结 Core 的 `SlidingWindowManager.get_num_skipped_tokens` 以“下一待计算位置”为语义，
position222、window128对应跳过95个token、仅2个完整32-token块；公式本身不能证明block123被提前释放。
缺少实际释放调用参数、页池所有权变更和跨group写入证据，不修改Core或据此猜测 off-by-one。

八rank runtime 的29个二进制/配置指纹一致，无截断；所列custom op源文件哈希与本地基线一致。
服务器实际 Python3.12.13 / Torch2.10.0+cpu / torch_npu2.10.0.post2，CANN9.0.1。
已保存扩展、opapi、tiling、device对象等指纹，但没有可重现构建清单，源码与二进制的构建对应仍有缺口。
原绝对地址、整个模型图、其他stream历史与workspace初值不在capsule内。
未采集页在现有restore中标为NaN哨兵 UNKNOWN，不是原值；若算子读取它们，不能称精确重放。

## 单卡、无权重服务器命令

[受控脚本](run_dspark_saved_operator.sh) 只调用已有 [replay](operator_replay.py)：
先 rank0/1803 异常样本和1802正常对照的 **saved metadata + ACLGraph**，
再分别做 **saved + eager**、**regenerated + ACLGraph**。每个组合三轮输出，共6个进程；
每个执行180s、TERM后15s升级，运行身份预检120s。无权重、无TP/EP初始化、无模型十点或性能实验。

```bash
if bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_saved_operator.sh <最终工具提交SHA>; then
  printf 'operator experiments completed; inspect numerical results\n'
else
  rc=$?
  printf 'operator experiments stopped: rc=%s\n' "$rc"
fi
```

第一次取得此新增脚本，需要先在干净 feat/dspark 上 `git fetch origin feat/dspark` 并
`git merge --ff-only <最终工具提交SHA>`。脚本复核Core SHA，创建全新目录，复制并校验两份输入哈希，
设置只读；每个case之后再次检查哈希。记录原始runtime与本次加载扩展/OPP身份，扩展/opapi必须匹配，
已发现的共同artifact若不同立即停止。其他延迟加载对象仍需结合每次result.runtime比较。
严格选项与trap只在子Bash；日志、PIPESTATUS、原始退出码、输入副本、reference.pt、replay.pt、
result.json全部归档，导出错误不覆盖原退出码。运行错误/超时立即停止后续case。
**捕获到NaN是实验结果，进程返回0只表示工具运行完毕，不是数值PASS。**

判读：若 saved/ACLGraph复现原head NaN且正常对照有限，再看 eager 是否相同；
regenerated对照只改变SAS metadata。两者都异常会强化“已捕获输入足以触发”的判断，
只有graph或saved异常才进一步区分图状态/metadata方向，不能凭一次对照直接归因kernel。
未复现仍写未复现。最少回传新 `dspark-saved-operator.*-evidence.tar.gz` 和 `.sha256`。

## 两类错误与本地检查

八rank 1803 Markov NaN阻止proposal1790发布；1804才缺少owner。原首错和检查均保留。
cleanup 独立为 `worker_cleanup_incomplete`、success=false；显式shutdown返回不代表worker自然退出。
本轮没有退出实现修改。

本地Python3.12.13/Torch2.10.0 macOS CPU：24份真实数据审计通过；相关回归44 passed、6 skipped，
跳过的是4项原生NPU SWA和2项NPU dispatcher/ACLGraph，本机未安装torch_npu。
新增测试覆盖逐head NaN/Inf、padding排除、部分有限误差及全非有限输入；
另以模拟子进程实际执行shell driver，验证六组顺序、首个失败后停止、PIPESTATUS与失败归档。
改动文件检查通过；全仓 `bash format.sh ci` 仍返回1，8个失败hook及78个自动修改路径与基线一致，
没有自动修改本轮文件。该检查在隔离工作树执行，未把基线格式修改带入提交。
本轮replay仅新增报告字段，执行方式、恢复布局、metadata重建方式都未改变。
