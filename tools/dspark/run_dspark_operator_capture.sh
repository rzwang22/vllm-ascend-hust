#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Invoke from a parent if/then/else; strict options remain in this child Bash.
set -euo pipefail
test "$#" -eq 1
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
  printf 'MAIN_RC=%s\nPLUGIN=%s\n' "$rc" "$plugin_sha" > "$check_dir/status.txt" || true
  if tar -czf "$check_dir-evidence.tar.gz" -C "$(dirname "$check_dir")" "$(basename "$check_dir")"; then
    sha256sum "$check_dir-evidence.tar.gz" > "$check_dir-evidence.sha256" || true
  else printf '预检证据导出失败；原退出码=%s\n' "$rc" || true; fi
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
logged operator timeout --signal=TERM --kill-after=15s 300s python -m pytest -q -ra \
  tests/ut/test_dspark_operator_npu.py --basetemp "$check_dir/operator" --junitxml "$check_dir/operator.xml"
logged operator-acceptance python - "$check_dir/operator.xml" <<'PYTEST'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert len(cases) == 4 and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print('Synthetic native SWA controls passed; this is not the archived call')
PYTEST
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
  target-boundaries 1 --attention --worker-exit --operator-capture
