# Operator capsule 格式编号序列化修复

本轮仅修复诊断文件的保存/读取兼容性，不修改模型数值、Core/custom op、confidence、stream、
退出预算或采集范围。默认关闭、performance_eligible=false 和全部回执/数值验收保持原样。
原 layer 1 attention NaN 与 worker 自然退出问题均未关闭，NPU 复验 PENDING。

## 依据与版本

开发起点 Plugin `8b1fc3fa4df12c90af9913b024e98d148204dba0`；
Core `897306c43bf800e2480cb5c0f3e2da408d85a2fd` 保持不变。
用户提供服务器结果目录 `dspark-compile-check.ShmgeaMM` 和 driver `operator-capture.jj9cL023.log`：
原生 SWA 4 passed、dispatcher CPU 两项通过、NPU 两项在受限加载时拒绝
`torch_npu.npu._format.Format`；完整 focused/模型十点未启动。
**本轮没有实际取得这些日志或归档，以上是用户报告，未独立核验。**

本地 Python 3.12.13 / Torch 2.10.0（macOS CPU，没有 torch_npu）独立复现：
把 IntEnum 放入 `npu_format` 后，`torch.save` 序列化枚举类引用，
`get_unsafe_globals_in_checkpoint` 能列出该类，`torch.load(..., weights_only=True)` 拒绝读取。
这是可复现的序列化类型缺陷；本地没有运行服务器的 Format 实现。
服务器报告环境是 Python 3.12.13 / Torch 2.10.0+cpu / torch_npu 2.10.0.post2（Linux）。

## 进入文件的路径与最小修复

[descriptor](../../vllm_ascend/diagnostics/dspark_profile_operator.py) 原先把
`get_npu_format(tensor)` 返回对象直接存入 layout。IntEnum 满足 `isinstance(value, int)`，
但不是普通 int。`before` 中的 `json.dumps(layout)` 仅生成索引键，没有替换
`layouts_by_id` 保留的对象；`save` 选中 layout 后将其写入 capsule，最终 pickle 仍携带枚举类。
真实 NPU 采集与 dispatcher 测试均走这条路径，故修复放在采集实现中。

- `descriptor` 立即将非 None 格式值转换为普通 `int`，None 保留；不按格式名称推断或替换编号。
- 审阅 capsule 的 scalars、layouts、identity、options、kv_binding 及统计字段：
  已知字段预期是基础类型，实际 tensor 参数在 `values` 中。未发现另一项已证明的后端类污染。
  在最终保存边界递归规范化非 tensor metadata 的 Enum 值，保留基础类型、list/tuple/dict；
  无明确编号/值语义的后端对象明确报错，不字符串化、不注册 safe globals。
  该检查仅在已启用采集的 CPU 保存阶段执行，不增加设备操作或等待。
- [replay](operator_replay.py) 将恢复后后端报告的格式也按 int 编号比较。
  None 表示没有格式信息；其他编号完整保留。不支持/无法保持原格式仍明确失败，绝不默认为 ND。
  正式入口继续使用 `map_location="cpu", weights_only=True`。
- dtype、shape、stride、storage offset、alias、物理页号、tensor 内容、当前 epoch 回执及数值检查均保留。
  旧的不安全文件不会被自动放行；本次目标是新采集文件安全保存和读取。

## 本地验证

[新增回归](../../tests/ut/test_dspark_operator_serialization.py) 覆盖 None、普通 int、IntEnum、
未知编号及负编号，保存前后断言 `type(value) is int`；旧枚举文件仍复现受限加载失败。
完整 `OperatorCapture.before/own/save` 生成的 capsule 分别覆盖 None/int/IntEnum：
在独立 `python -I` CPU 进程通过正式 replay CLI 的 reference 模式读取并验证，
没有导入 torch_npu 或添加 allowlist，扫描文件的 unsupported globals 为空。
CPU descriptor 测试模拟后端返回值；保存、受限加载及子进程入口均真实执行。

原 [dispatcher 四参数回归](../../tests/ut/test_dspark_attention_receipts.py) 未改参数、后端或跳过条件，
每轮额外检查真实采集文件的精确格式字段类型及 unsupported globals。
共享存储覆盖负例、独立存储多轮回执、NaN 注入、窗口更新、地址检查和静态/符号形状编译均保留。
相关七文件 CPU/source/mock 测试实际结果见 [验证记录](PROFILE_OPERATOR_SERIALIZATION_VALIDATION.json)。
本地实际为 **97 passed、6 skipped**；最终序列化文件单独复跑 **22 passed**。
本地不能验证真实 Format、npugraph_ex 或 ACLGraph；CPU 通过不替代这些 NPU 项目。
改动文件的 pre-commit 检查通过；全仓 `bash format.sh ci` 在隔离工作树执行，返回 1：
8 个失败 hook 和 78 个自动改动路径与基线一致，没有自动改动本轮文件。未将全仓检查报告为通过。

## 唯一服务器复验入口

把最终交付 SHA 作为参数传给已有 [分阶段脚本](run_dspark_operator_capture.sh)，父 shell 用 if 接收退出码：

```bash
if bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_operator_capture.sh <最终完整Plugin-SHA>; then
  printf 'operator capture workflow completed\n'
else
  rc=$?
  printf 'operator capture workflow failed: rc=%s\n' "$rc"
fi
```

脚本在子 Bash 内使用严格选项，核对冻结 Core、ff-only 更新 Plugin，使用全新结果目录。
先运行原生 SWA 四项，再运行 dispatcher 四项（每阶段 300s，TERM 后 15s 升级）；
两份 JUnit 均要求恰好四项、零失败、零跳过。dispatcher 每轮实际 capsule 必须受限加载成功、
无额外类依赖，真实观测回执/独立存储断言必须通过。任一步失败即停止，保留日志、JUnit、
PIPESTATUS、原退出码及预检归档，导出失败不覆盖原退出码。
通过后才执行完整 focused（已加入序列化回归）及同一引擎原前十点：
`target-boundaries 1 --attention --worker-exit --operator-capture`，原模型/输入 manifest 不变。
未增加模型初始化、诊断切点或实验组；首点八 rank 连续三轮提前验收保留。

最少回传新的 `dspark-compile-check.*-evidence.tar.gz` 和 SHA256；若进入 profile，
同时回传新的 `dspark-large-batch.*-evidence.tar.gz` 和 SHA256，内含 operator capsule、索引、
回执/首错及独立 cleanup 状态。仅序列化预检通过不等于原 NaN 修复或自然退出成功。
