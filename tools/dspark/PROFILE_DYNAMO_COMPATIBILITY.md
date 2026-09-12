# DSpark attention 诊断：Torch 2.10 静态尺寸兼容性

本轮是**诊断编译兼容性修复**。独立 bank 修复继续保留；原 layer 1 attention NaN 和
worker 自然退出没有新的模型运行证据，仍未关闭。Core/custom op、模型计算、confidence、
退出顺序和超时均不变。没有 NPU 或模型运行，Ascend 验证为 **PENDING**。

## 实际读取的证据与环境差异

已读取 `dspark-large-batch.YYOqidKO-evidence.tar.gz`，11 个成员、展开 50071 bytes。
独立计算 SHA256 为 `86043901898181d7ddff116ce0501ef2d9989de65aa4555daaa7aea051e49a7f`；
本轮用户没有提供预期 SHA，因此这是所收附件的内容指纹，不宣称与未提供的服务器校验值匹配。
[审计记录](PROFILE_YYOqidKO_AUDIT.json) 保存各文件哈希和四份 traceback 行锚点。
没有执行归档中的代码。

`source.log` 明确记录 torch=2.10.0+cpu、torch_npu=2.10.0.post2。
traceback 路径为 `/usr/local/python3.12.13/lib/python3.12`；归档未打印完整 sys.version，
下一阶段会独立记录解释器版本。source.pipestatus=0 0；focused.pipestatus=1 0、MAIN_RC=1。
归档没有 git SHA 输出，运行基线来自用户提供的 Plugin
`d8dd65848fc46d4e5fc2143f7a1baa8478519274` / Core
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`，且既有入口在 source 前强制核对二者。
本地开发起点及 fetch 后 origin/feat/dspark 与 Plugin 相同，两个仓库原本干净。

focused=**4 failed、1081 passed**。cpu/npu × shared=True/False 均在首次 run(x) 的 Dynamo
tracing 失败：`TargetBoundaryFlags.write → torch.sym_min` 返回普通 int，Unsupported。
没有完成 AOT/capture/replay，没有 runs/或十点结果，不能报告独立存储修复失败、NaN 复现或
worker 正常/异常退出的新结论。

此前本地环境实际为 Python3.12.13、Torch **2.14.0**、macOS ARM64，无 torch_npu；
并非服务器的 Torch2.10。现增加独立 Python3.12.13 / Torch **2.10.0** 环境。
它与服务器 Torch 主版本一致，但 wheel 为 macOS CPU，服务器为 Linux `+cpu` 并加载 Ascend
扩展，不能声称二进制/后端完全一致。两套环境的独立探针保存在[验证记录](PROFILE_YYOqidKO_VALIDATION.json)。

## 根因及最小改动

[独立复现脚本](repro_sym_min_compile.py) 只导入 Torch，不加载项目、pytest fixture 或 NPU。
同样的 fullgraph=True 编译得到：

| 表达式 | Torch2.10 静态 | Torch2.10 符号 | Torch2.14 静态/符号 |
| --- | --- | --- | --- |
| 直接 torch.sym_min(shape[0], cap) | Unsupported：example_value=int | 通过 | 通过 |
| 内置 min(shape[0], cap) | 通过 | 通过，FX 中仍为 torch.sym_min | 通过 |

实际绑定为 torch 模块的原始 sym_min 函数，普通 tensor.shape[0] 类型为 int。
2.10 `torch/_dynamo/trace_rules.py` 将 torch.sym_min 交给 TorchInGraphFunctionVariable；
`variables/builder.py::wrap_fx_proxy_cls` 的静态 int 特例不包含 sym_min，最终报 non-Tensor。
2.14 的对应特例已包含 sym_min。这解释了此前本地通过、服务器失败的版本差异。
2.10 `variables/builtin.py::_call_min_max_binary` 对 ConstantVariable 直接折叠，对
SymNodeVariable 显式生成 torch.sym_min 及 SymNodeVariable，保留运行时符号上限。

最小改动只有两处表达式：
[TargetBoundaryFlags.write](../../vllm_ascend/diagnostics/dspark_profile_target.py) 和同类
[TargetLayerSnapshots.write](../../vllm_ascend/diagnostics/dspark_replay.py) 使用内置 min。
整个 diagnostics 目录已扫描相同写法，无第三处直接 sym_min/sym_max 调用。
没有把 SymInt 转 int，没有从已截断 tensor 的 shape 推导常量上限，没有强制某种 shape。
默认关闭路径、所有 21 项图内真实 epoch/reset、独立 storage、raw/consume、首错和历史保留、
首点八 rank 连续三轮提前验收均未修改。

干净进程中的同版本最小复现排除了 fixture/顺序污染作为**此次错误的必要原因**。
原四参数文件在无 conftest 的干净2.10环境也得到同样的两个 CPU 编译失败；NPU因无环境跳过。
源码检查没有 fixture 替换 sym_min；回归增加 module.torch、sys.modules['torch'] 和 sym_min
绑定恒等断言。完整 host focused 顺序在修复后验证；尚未在本地复现安装态 NPU conftest 环境，
不把所有未来 NPU 后端错误都排除为污染。

## 验证范围

原真实 dispatcher 回归保持 fullgraph=True、实际 dsa_forward 函数体及 mutation schema、AOT，
旧共享 bank 必须按断言重现覆盖，新独立 bank 必须连续三轮正确；继续核对 NaN 行、窗口更新、
持久地址和 wrapper 整块 copy-back。新增静态/动态测试经过实际两种 writer 的 Dynamo+AOT，
覆盖输入行数29、5、24、30（容量24），校验截断、未写区域、epoch、NaN/Inf及 snapshot bucket。
动态 FX 必须包含符号 min；既有大 profile8192 → 小 capture 的无 guard FX 测试继续运行。

独立 CPU 探针命令（不加载模型，不需 NPU）：

```bash
python tools/dspark/repro_sym_min_compile.py --output /tmp/dynamo-compatibility.json
```

它记录 legacy 的预期失败，不做 fallback；修复表达式任何失败会返回非零。
NPU 四参数中的两个用例仍须真实进入 plugin npugraph_ex_compile 并 ACLGraph replay，
不能用 CPU AOT 或跳过代替。实际结果：Torch2.10 的24文件 host focused 为 **600 passed、5 skipped**；
两种 Torch 版本的相关三文件分别为 **108 passed、2 skipped**。五个 focused 跳过项为
两个真实 NPU replay 用例和三个安装态 vLLM/Ascend 用例。改动文件 hooks、runbook 的 bash -n
通过；隔离 worktree 的 `bash format.sh ci` 返回1，仍是基线相同的八类失败和78个无关格式路径，
本轮文件没有被其修改。没有远程 CI/NPU PASS。完整记录见验证 JSON。

## 服务器分阶段命令

将 `plugin_sha` 设置为本次交付完整 SHA，然后一次执行下面的子 Bash。
第一阶段**只运行原四参数节点**，不加载模型权重；300秒微型测试上限不改变模型退出预算。
保存版本、日志、PIPESTATUS、JUnit、FX/copy-back 临时证据；任何失败、跳过或用例数不足均停止。
全部四项通过后才调用原入口，它再次执行完整 focused（含新增 shape 用例）和原单引擎十点。

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
test -n "${ASCEND_CUSTOM_OPP_PATH:-}"
export PYTHONPATH="/workspace/vllm-ascend-hust:/workspace/vllm-hust:${PYTHONPATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=0 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
mkdir -p /workspace/dspark-results
check_dir=$(mktemp -d /workspace/dspark-results/dspark-compile-check.XXXXXXXX)
printf 'COMPILE_CHECK_DIR=%s\n' "$check_dir"
finish() {
  local rc=$?
  trap - EXIT
  printf 'MAIN_RC=%s\nPLUGIN=%s\n' "$rc" "$plugin_sha" > "$check_dir/status.txt"
  if tar -czf "$check_dir-evidence.tar.gz" -C "$(dirname "$check_dir")" "$(basename "$check_dir")"; then
    sha256sum "$check_dir-evidence.tar.gz" > "$check_dir-evidence.sha256" || true
  else printf '预检证据导出失败；原退出码=%s\n' "$rc"; fi
  exit "$rc"
}
trap finish EXIT
logged() {
  local name=$1
  shift
  local codes
  if "$@" 2>&1 | tee "$check_dir/$name.log"; then
    codes=("${PIPESTATUS[@]}")
  else
    codes=("${PIPESTATUS[@]}")
  fi
  printf '%s\n' "${codes[*]}" > "$check_dir/$name.pipestatus"
  if test "${codes[0]}" -ne 0; then return "${codes[0]}"; fi
  return "${codes[1]}"
}
logged versions python -c 'import sys, platform, torch, torch_npu; print(sys.version, sys.executable, platform.platform()); print("torch", torch.__version__, "torch_npu", torch_npu.__version__); print("sym_min", torch.sym_min, torch.sym_min.__module__)'
logged source python tools/dspark/p08_r8_checks.py source /workspace/vllm-ascend-hust /workspace/vllm-hust
logged four timeout --signal=TERM --kill-after=15s 300s python -m pytest -q -ra \
  tests/ut/test_dspark_attention_receipts.py::test_real_dispatch_aot_copyback_and_repeated_replay \
  --basetemp "$check_dir/pytest" --junitxml "$check_dir/four.xml"
logged acceptance python - "$check_dir/four.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
expected = {f'test_real_dispatch_aot_copyback_and_repeated_replay[{device}-{shared}]'
            for device in ('cpu', 'npu') for shared in ('True', 'False')}
assert len(cases) == 4 and {c.get('name') for c in cases} == expected
assert not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print('4/4 passed; CPU AOT and NPU npugraph_ex/ACLGraph checks completed')
PY
logged profile bash tools/dspark/run_dspark_profile_control.sh "$plugin_sha" \
  /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
  target-boundaries 1 --attention --worker-exit
BASH
then printf '退出码0；分别验收回执、原NaN和自然退出。\n'
else rc=$?; printf '退出码%s；停止后续运行，保留本阶段证据。\n' "$rc"
fi
```

预检失败只回传新 compile-check evidence.tar.gz + .sha256。若进入第二阶段，再回传新
large-batch evidence.tar.gz + .sha256。旧目录不覆盖；完整 profile 的状态查询/STOP/独立导出
继续沿用[回执修复 runbook](PROFILE_ATTENTION_RECEIPTS.md)。预检导出失败不替换最初退出码。
没有 OPP 构建、额外模型初始化、B128/B256 或性能比较。所有产物属于诊断，不能编译成本表。
