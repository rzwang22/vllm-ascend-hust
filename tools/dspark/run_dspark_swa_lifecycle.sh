#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run from a parent if/then/else. No model weights, engine, or custom-op build.
set -euo pipefail
test "$#" -eq 1
plugin_sha=$1
core_sha=71d2c1c436eba894a8e9eeb2c5af17e05cb42970
cd /workspace/vllm-ascend-hust
mkdir -p /workspace/dspark-results
result_dir=$(mktemp -d /workspace/dspark-results/dspark-swa-lifecycle.XXXXXXXX)
printf 'RESULT_DIR=%s\n' "$result_dir"
finish() {
  local rc=$?
  trap - EXIT
  printf 'MAIN_RC=%s\nPLUGIN=%s\nCORE=%s\nPERFORMANCE_ELIGIBLE=false\n' "$rc" "$plugin_sha" "$core_sha" > "$result_dir/status.txt" || true
  if tar -czf "$result_dir-evidence.tar.gz" -C "$(dirname "$result_dir")" "$(basename "$result_dir")"; then
    sha256sum "$result_dir-evidence.tar.gz" > "$result_dir-evidence.sha256" || true
  else
    printf 'Evidence export failed; original MAIN_RC=%s\n' "$rc" >&2
    if test "$rc" -eq 0; then rc=1; fi
  fi
  exit "$rc"
}
trap finish EXIT
logged() {
  local name=$1
  shift
  local codes
  if "$@" 2>&1 | tee "$result_dir/$name.log"; then codes=("${PIPESTATUS[@]}");
  else codes=("${PIPESTATUS[@]}"); fi
  printf '%s\n' "${codes[*]}" > "$result_dir/$name.pipestatus"
  if test "${codes[0]}" -ne 0; then return "${codes[0]}"; fi
  return "${codes[1]}"
}
accept() {
  python - "$result_dir/$1.xml" "$2" <<'PY'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert len(cases) == int(sys.argv[2]), (len(cases), sys.argv[2])
assert not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print('All expected cases executed; zero failures/errors/skips. Not a model NaN/exit acceptance.')
PY
}
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$plugin_sha"
test -z "$(git -C /workspace/vllm-hust status --porcelain)"
test "$(git -C /workspace/vllm-hust branch --show-current)" = feat/dspark
logged core-fetch git -C /workspace/vllm-hust fetch origin feat/dspark
logged core-update git -C /workspace/vllm-hust merge --ff-only "$core_sha"
test "$(git -C /workspace/vllm-hust rev-parse HEAD)" = "$core_sha"
export PYTHONPATH="/workspace/vllm-ascend-hust:/workspace/vllm-hust:${PYTHONPATH:-}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=0 VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
logged source python tools/dspark/p08_r8_checks.py source /workspace/vllm-ascend-hust /workspace/vllm-hust
logged identity timeout --signal=TERM --kill-after=15s 180s python - "$result_dir/runtime.json" <<'PY'
import json
import sys
from pathlib import Path
import torch
import torch_npu
from tools.dspark.operator_replay import runtime_identity
assert sys.version_info[:3] == (3, 12, 13), sys.version
assert str(torch.__version__) == '2.10.0+cpu', torch.__version__
assert str(torch_npu.__version__) == '2.10.0.post2', torch_npu.__version__
Path(sys.argv[1]).write_text(json.dumps(runtime_identity(), indent=2) + '\n')
PY
# These fixtures invoke actual installed Core classes and never download weights.
# --noconftest prevents plugin global mocks; --confcutdir omits unrelated Core model fixtures.
logged core timeout --signal=TERM --kill-after=15s 300s python -m pytest \
  --confcutdir=/workspace/vllm-hust/tests/v1/core -q -ra \
  /workspace/vllm-hust/tests/v1/core/test_async_swa_reclamation.py \
  /workspace/vllm-hust/tests/v1/core/test_single_type_kv_cache_manager.py \
  --basetemp "$result_dir/core" --junitxml "$result_dir/core.xml"
logged core-accept accept core 22
logged lifetime timeout --signal=TERM --kill-after=15s 300s python -m pytest --noconftest -q -ra \
  tests/ut/test_dspark_swa_lifecycle.py --basetemp "$result_dir/lifetime" --junitxml "$result_dir/lifetime.xml"
logged lifetime-accept accept lifetime 3
logged capsule timeout --signal=TERM --kill-after=15s 300s python -m pytest --noconftest -q -ra \
  tests/ut/test_dspark_operator_capture.py --basetemp "$result_dir/capsule" --junitxml "$result_dir/capsule.xml"
logged capsule-accept accept capsule 23
printf 'Bounded lifecycle/coverage checks completed. Full model NaN and worker exit remain PENDING.\n'
