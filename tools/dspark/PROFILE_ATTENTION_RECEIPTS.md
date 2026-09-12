# DSpark attention 回执修复：6DoBME2M

后续 Torch2.10 静态尺寸兼容性修复及两阶段预检见 [YYOqidKO 说明](PROFILE_DYNAMO_COMPATIBILITY.md)。

本轮是**诊断回执修复**。原始 NaN 未进入复现场景，worker 自然退出也未修复。
CPU 实际 dispatcher/AOT 回归证明共享存储覆盖缺陷；服务器最终编译图未随归档保存，
对该次运行具体覆盖指令的确认、完整 Ascend 编译、ACLGraph、模型复验为
**PENDING**。Core 保持 `897306c43bf800e2480cb5c0f3e2da408d85a2fd`。
开发起点与 fetch 后 origin/feat/dspark 均为 `9af2b22b54d74029647e5e7ad040cf03d740bd96`，
工作区原本干净；本轮叠加修改，无回退、无模型/custom op/schema/算法修改。

## 独立审计

[可重算脚本](verify_6dobme2m_evidence.py) 实际读取归档，不解压执行其内容：

```bash
python tools/dspark/verify_6dobme2m_evidence.py \
  /path/to/dspark-large-batch.6DoBME2M-evidence.tar.gz --output /tmp/receipt-audit.json
```

SHA256 `e7e6021068ce6048c0b6856a962f0fce1f3ebf88fab132896f1c8f064912197e`，
98 成员、28416185 展开 bytes。版本来自 plan/source；[逐 rank 审计](PROFILE_6DoBME2M_AUDIT.json)
保留每份 latest 原始 SHA、窗口状态、回执计数以及原始退出记录。

- 首点 `ctx128-n1-t6-balanced` 请求输出 512 tokens，无生成错误；snapshot/证据验收失败，
  completed_points=0。没有 retained 的完成点，也没有第十点记录。
- 全八 rank 的 execution 86/87/88：九个外部 target receipts 和三份 raw/consume receipts
  均为当前 execution；十二个 attention receipts 全为 -1。raw_replay_verified=true。
  attention_state 三轮不同，窗口位置实际更新。其 flags 没有有效回执，必须判 unavailable。
- cleanup 返回 forced_cleanup，success=false。worker SIGTERM count=8，SIGKILL count=6；
  rank 3/4 的 raw_exitcode=-15，其余 null 保持 unavailable；manager 也强制清理。
  cleanup-failure.prior_error 保留证据验收错误。没有以最终无残留发布正常退出。

这些为离线断言，不是 NPU 测试。原始 `layer.1.attn_input → layer.1.attn_output` NaN
仍是此前有效定位边界；本次不能缩小数值区间或宣称已经修复。

## 已证明的诊断缺陷与修复

实际链路：AttentionProbe.write → TargetBoundaryFlags.write，以
`layer.1.attention.<stage>` 查找同一 names 中的后十二项。名称/索引本身正确。
[DSA](../../vllm_ascend/ops/dsa.py) 的 `dsa_forward` 是 opaque 自定义算子，schema
只声明 `output` 可变。probe 经 context 找到的 bank 不属于已声明输出。
归档 b64.log 631–652 行确认八 rank 使用 `enable_npugraph_ex=True`、static_kernel=false，
保存了编译图缓存；675–706 行确认 FULL capture。实际入口为
[npugraph_ex_compile](../../vllm_ascend/compilation/compiler_interface.py)，配置
force_eager=true、inplace_pass=false、clone_input/output=false，然后进入 nge backend。
不能把另一个分支的 plugin compile_fx 当成本次服务器入口。
上游 [npugraph_ex 源码](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/npugraph_ex/npugraph_ex/npu_fx_compiler.py#L856)
的 `_compile_graph_with_aot` 调用 aot_module_simplified，保留输入 mutation；这是接口机制依据，
不是归档内安装包版本的独立验证。归档没有该安装包源码或最终 FX 指令，仍保留此覆盖缺口。

外部 decoder 同时修改原先整块 21 项 bank，在真实 CPU AOT 回归中，
AOT 将这些外部写入 functionalize，返回 mutation 结果，由 AOT runtime wrapper 整块 `copy_` 回原 bank。
算子内部未声明的十二项副作用被这个回写覆盖；仅在算子内写的独立 attention_state 不受影响。

[test_dspark_attention_receipts.py](../../tests/ut/test_dspark_attention_receipts.py)
使用实际 dsa_forward/filter_metadata/_build_kv_cache/fake 函数体、相同 mutation schema 的
真实 torch.library 分派、实际 plugin compile_fx 和 AOT。加速器数值叶子替换为小张量运算。
旧共享存储在三轮中精确重现 9 当前、12=-1、state 更新；独立存储在相同路径三轮通过，
最后一轮 NaN 行映射正确。测试保存 functionalized-graph.txt（返回 mutation 的 FX 图）和 copyback-evidence.json；
CPU profiler 实际记录 wrapper 对整块 flags/receipts 的 `aten::copy_` shape。
这比直接 bank.write 或仅 ATen 重放多覆盖了导致缺陷的 functionalization/custom-op 边界。

最小修改在 [TargetBoundaryFlags](../../vllm_ascend/diagnostics/dspark_profile_target.py)：
外部 flags/receipts 与算子内 attention_flags/attention_receipts 使用**不同 storage**，
不是大 bank 的两个 view。共享 epoch_input 在 replay 前更新，图内只读；两组 receipts 每轮
reset=-1，每个真实观测仍在图内 copy 当前 epoch。图返回后才合并为原有 21 项协议并立即
生成 owned compact packet。原有失效回执拒绝逻辑、首次异常及前两轮保留均不变。
不在 host 补写 receipt，不增加同步，不改变图内算子数值或 schema。

NPU 参数化测试会实际 capture 上述 AOT 图并多次 replay，检查无 Python dispatcher 重入、
storage 地址稳定、每轮 reset/receipt/NaN 和窗口更新。它需要 torch_npu 与可用 Ascend；
本地跳过，服务器 focused 阶段会执行。NPU 用例走实际 plugin npugraph_ex_compile（相同配置），保存实际 backend 生成代码；
CPU 用例走 plugin compile_fx 来隔离共享 AOT 机制，两个入口明确区分。
完整安装态 vLLM 和真实模型仍需
下述同引擎运行确认，不能仅凭 CPU 或这个微型图测试宣称 NPU 修复成立。

## 首点提前验收和开销

仍使用原 `target-boundaries 1 --attention` 默认关闭选项，不新增环境变量或数值切点。
worker 从**已经合并 D2H 的 FULL 包**发布 `rank-N-attention-validity.json`，包含真实
21 项和 raw/consume receipts、execution/proposal、point/rank、完整快照与五个 buffer 描述。
每 rank 最多三次原子写盘：首个无效包立即失败冻结；连续三轮完整包通过后冻结。
缺失、陈旧、非连续记录不能通过；若诊断写盘失败，前端也不能据此通过。

前端 [AttentionValidity](profile_attention_validity.py) 在生成期间复用 guard 的 0.1 秒
轮询，无 RPC；收到无效记录立即有界取消。另按实际已输出 token 总数检查：到 64 tokens
仍未收齐全八 rank 三轮回执则失败（DELTA 事件合并可能超出一批输出），不等整点 512。
自然提前完成但回执不全也失败；已有生成首错优先，不用门控错误替换。
全部通过后关闭读取，**同一 generator/engine** 完成首点及原先其余九点，没有第二次模型初始化。
门控只证明观测有效，不证明数值有限或原 NaN 已修复。

持久 bank 总字节不变，本配置仍 47024 bytes/rank。图内归约数量不增加，reset 多一个小
receipt bank；replay 后多两次紧凑 device cat，容量384时合计临时 16296 bytes/rank，之后
沿用已有 owned integer packet 和一次合并 D2H，没有额外 host wait/全局同步。
最多三次快照写盘及首次全 rank 文件轮询会影响时序，须记录此观测影响。默认关闭不做这些工作。
performance_eligible=false，不能生成成本表。退出步骤、信号、预算和状态不变。

## 唯一服务器复验

将 `plugin_sha` 设置为本次交付的完整 commit SHA；仅运行下面一次。
严格 shell 选项留在子 Bash，父 shell 接收退出码。原 manifest、模型检查、前十点顺序、TP8+EP、
K5、FULL_DECODE_ONLY、draft eager、512 tokens 及所有预算沿用原入口。

```bash
if bash -s -- "$plugin_sha" <<'BASH'
set -euo pipefail
plugin_sha=$1
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = feat/dspark
test -z "$(git -C /workspace/vllm-hust status --porcelain)"
test "$(git -C /workspace/vllm-hust rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd
git fetch origin feat/dspark
git merge --ff-only "$plugin_sha"
test "$(git rev-parse HEAD)" = "$plugin_sha"
bash tools/dspark/run_dspark_profile_control.sh "$plugin_sha" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  target-boundaries 1 --attention --worker-exit
BASH
then printf '退出码 0；仍须分别检查诊断、数值和自然退出。\n'
else rc=$?; printf '退出码 %s；停止后续测试，保留首错与归档。\n' "$rc"
fi
```

入口创建新 result_dir，失败不继续其他点；supervisor 3600 秒上限、worker 5+4 秒、engine12、
outer16、failure grace24 秒均不变。状态查询：`cat "$result_dir/status.txt"`、
`cat "$result_dir/runs/b64-supervisor.json"`；受控停止：`touch "$result_dir/STOP"`。
自动导出失败不覆盖原始 MAIN_RC；独立补导出方式见[既有 runbook](PROFILE_TARGET_ATTENTION.md)。

最少回传新 **evidence.tar.gz + .sha256**，保留新增八份 attention-validity、全部
worker-first-failure/worker-exit、原始 point JSON、source/focused/plan/cleanup/supervisor/status。
验收分别回答：全部十二项是否连续由当前 replay 写入；若达到第十点，原 NaN 是否复现及首异常
位置；worker 是否自然退出。任何 INVALID_RECEIPT、强制清理或 unavailable 都不可发布 PASS。

## 本地验证结果

[验证记录与实际 copy-back shape](PROFILE_ATTENTION_RECEIPTS_VALIDATION.json)：
24 个 CPU/source focused 文件 **596 passed、5 skipped**；额外 profiler 回归 **2 passed、2 skipped**。
五个 focused 跳过项为两个 Ascend npugraph_ex/ACLGraph 用例及三个安装态 vLLM/Ascend 用例。
服务端 focused 列表中其余 worker/attention/spec_decode 安装态文件未在本地执行。
门控测试覆盖全八 rank、stale/missing/非连续、64-token 提前停止、无输出时轮询失败、
同引擎继续、已有生成首错优先；worker 测试覆盖原子首次失败及三轮后冻结、后续 buffer 复用。
默认关闭路径、原有清理失败与首错测试保持通过。

改动文件 hooks 通过。按项目要求在隔离 worktree 运行 `bash format.sh ci`，返回 1：
八类失败及 78 个无关自动格式修改路径与基线一致，本轮文件没有被修改；未把全仓检查报为通过。
没有远程 CI 或 NPU 结果；服务器复验仍为 PENDING。
