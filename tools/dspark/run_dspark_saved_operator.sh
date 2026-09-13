#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Bounded, weight-free ZvqiDthD controls. Invoke from a parent if/then/else.
set -euo pipefail
test "$#" -ge 1 && test "$#" -le 2
plugin_sha=$1
slot_controls=${2:-}
test -z "$slot_controls" || test "$slot_controls" = --slot-controls
cd /workspace/vllm-ascend-hust
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = feat/dspark
git fetch origin feat/dspark
git merge --ff-only "$plugin_sha"
test "$(git rev-parse HEAD)" = "$plugin_sha"
test "$(git -C /workspace/vllm-hust rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd
test -z "$(git -C /workspace/vllm-hust status --porcelain)"
test -n "${ASCEND_CUSTOM_OPP_PATH:-}"
export PYTHONPATH="/workspace/vllm-ascend-hust:/workspace/vllm-hust:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0 ASCEND_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
prefix=dspark-saved-operator
if test -n "$slot_controls"; then prefix=dspark-slot-controls; fi
out=$(mktemp -d "/workspace/dspark-results/$prefix.XXXXXXXX")
printf 'REPLAY_DIR=%s\n' "$out"
finish() {
  local rc=$?
  trap - EXIT
  printf 'MAIN_RC=%s\nPLUGIN=%s\n' "$rc" "$plugin_sha" > "$out/status.txt" || true
  if tar -czf "$out-evidence.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"; then
    sha256sum "$out-evidence.tar.gz" > "$out-evidence.sha256" || true
  else printf 'Archive failed; original rc=%s\n' "$rc" || true; fi
  exit "$rc"
}
trap finish EXIT
logged() {
  local name=$1
  shift
  local codes
  if "$@" 2>&1 | tee "$out/$name.log"; then codes=("${PIPESTATUS[@]}"); else codes=("${PIPESTATUS[@]}"); fi
  printf '%s\n' "${codes[*]}" > "$out/$name.pipestatus"
  if test "${codes[0]}" -ne 0; then return "${codes[0]}"; fi
  return "${codes[1]}"
}
mkdir "$out/inputs"
source_dir=/workspace/dspark-results/dspark-large-batch.ZvqiDthD/runs/b64/worker-first-failure
cp "$source_dir/rank-0-operator-1803.pt" "$source_dir/rank-0-operator-1802.pt" "$out/inputs/"
printf '%s  %s\n' \
  f065f77610d99505069c2077125a9bb8db1a2211af1200c05f8a16cd1d6001d8 "$out/inputs/rank-0-operator-1803.pt" \
  ba628505667035fd9c146d43a12f0709a50863ee24af59bffa634b48716673fd "$out/inputs/rank-0-operator-1802.pt" \
  > "$out/inputs.sha256"
logged inputs sha256sum -c "$out/inputs.sha256"
chmod a-w "$out/inputs/"*.pt
cp "$source_dir/rank-0-operator-runtime.json" "$out/capture-runtime.json"
# Record the loaded extension/OPP before invoking any captured attention call.
logged runtime timeout --signal=TERM --kill-after=15s 120s python - "$out/capture-runtime.json" <<'PY'
import json
import sys
from pathlib import Path
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env
bootstrap_custom_op_env(include_vendor_lib=True)
from vllm_ascend import vllm_ascend_C
from tools.dspark.operator_replay import runtime_identity
current = runtime_identity()
print(json.dumps(current, indent=2))
previous = json.loads(Path(sys.argv[1]).read_text())
expected = {a["path"]: a["sha256"] for a in previous["artifacts"]}
actual = {a["path"]: a["sha256"] for a in current["artifacts"]}
required = [p for p in expected if "vllm_ascend_C." in p or p.endswith("/libcust_opapi.so")]
assert len(required) == 2 and all(actual.get(p) == expected[p] for p in required), "Loaded extension/opapi differs from capture"
assert all(actual[p] == expected[p] for p in actual.keys() & expected.keys()), "OPP artifact differs from capture"
PY
if test -n "$slot_controls"; then
  logged watch-preflight timeout --signal=TERM --kill-after=15s 180s python -m pytest -q -ra \
    tests/ut/test_dspark_operator_slot_controls.py::test_npu_watch_observes_each_graph_replay \
    --basetemp "$out/watch-test" --junitxml "$out/watch.xml"
  logged watch-acceptance python - "$out/watch.xml" "$out/watch-test" <<'CHECK'
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert len(cases) == 1 and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
root = Path(sys.argv[2])
for name in ('capture-semantics.pt', 'native-watch-evidence.pt'):
    files = list(root.rglob(name))
    assert len(files) == 1, (name, files)
    evidence = torch.load(files[0], map_location='cpu', weights_only=True)
    if name == 'capture-semantics.pt':
        assert 'error' not in evidence and evidence['replays_completed'] == 3
        assert not evidence['stages'][0]['snapshot_valid']
        assert evidence['stages'][0]['counter'].item() == 0
    else:
        watch = evidence['watch']
        assert 'error' not in evidence['expectations']
        assert len(watch['snapshots']) == 4 and watch['replays_submitted'] == 3
        assert [s['phase'] for s in watch['snapshots']] == ['warmup', 'replay-0', 'replay-1', 'replay-2']
        assert watch['stages'][1]['status'] == 'captured_not_replayed'
print('WATCH_ACCEPTED: capture unavailable; warmup + 3 executed replay snapshots')
CHECK
  for control in original-1802 original-1803 1803-from-1802 1802-from-1803; do
    extra=(--mode aclgraph --metadata saved)
    case "$control" in
      original-*) execution=${control#original-} ;;
      *-from-*) execution=${control%%-from-*}; donor=${control##*-from-}
        extra+=(--slot-source "$out/inputs/rank-0-operator-$donor.pt") ;;
    esac
    logged "$control" timeout --signal=TERM --kill-after=15s 180s \
      python tools/dspark/operator_replay.py "$out/inputs/rank-0-operator-$execution.pt" \
      --output "$out/$control" \
      --watch-slot --slot-block 123 --slot-offset 31 "${extra[@]}"
    logged "$control-input-integrity" sha256sum -c "$out/inputs.sha256"
  done
else
for control in aclgraph-saved eager-saved aclgraph-regenerated; do
  mode=${control%-*}
  metadata=${control#*-}
  for execution in 1803 1802; do
    logged "$control-$execution" timeout --signal=TERM --kill-after=15s 180s \
      python tools/dspark/operator_replay.py "$out/inputs/rank-0-operator-$execution.pt" \
      --mode "$mode" --metadata "$metadata" --output "$out/$control-$execution"
    logged "$control-$execution-input-integrity" sha256sum -c "$out/inputs.sha256"
  done
done
# A zero driver exit means the experiment completed, never numerical correctness.

fi
