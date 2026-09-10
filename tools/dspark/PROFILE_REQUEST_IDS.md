# Startup profile request identity repair

## Scope and source audit

This repair addresses an attribution failure, not a demonstrated KV/proposal
leak. Frozen Core `897306c43bf800e2480cb5c0f3e2da408d85a2fd` follows:

1. `AsyncLLM.generate()` awaits this instance's `add_request()`.
2. `InputProcessor.assign_request_id()` saves the external ID and adds a random
   suffix to the internal ID. Randomization remains enabled.
3. `AsyncLLM.add_request()` constructs `RequestOutputCollector` with the internal
   ID, awaits `_add_request()` (output processor registration and EngineCore
   enqueue), then returns that collector for `n=1`.
4. Public `RequestOutput` uses the external ID. Output processor active mappings
   are removed on completion, so reading public output or the final active map
   cannot recover all internal IDs.

`profile_request_ids.RequestIdObserver` observes the successful return in step 3
on one engine instance, only during one profile point. Signature binding uses
that instance's real method signature. It accepts only the existing token-prompt,
`n=1` profile protocol; it does not support child fan-out or streaming input.
No ID string parsing, environment override, global patch or worker ID rewriting
is used. The prior instance method is restored in normal, error and cancellation
paths. Live observer mappings and point references are cleared; a detached JSON
receipt remains in the artifact. Ordinary performance generation installs no
observer and retains its timing/output semantics.

`startup_cost_profile.collect()` validates every worker event against the exact
internal IDs for its point, including request-instance indexes. A local history
of validated internal IDs distinguishes unknown IDs from known previous-point
IDs; external IDs can recur in later points with new internal IDs. Original
worker event IDs remain unchanged. The point loop still uses one engine per
B/capture configuration, with the existing warmup exclusion, five valid samples,
TP checks and cost coverage checks. This change introduces only profile-side
CPU receipt bookkeeping, no device synchronization or data transfer.

## Evidence

Each point JSON contains:

- `streaming.request_id_mapping`: schema 1 receipt, source of mapping, point,
  expected external IDs, per-instance external/internal IDs, hook restoration
  and any admission conflict.
- `ranks[].cost_profile.measurements`: original internal IDs, point and event
  kind, along with the existing real timing/layout evidence.
- `request_identity_validation`: successfully validated internal IDs.
- On failure, `request_identity_failure`: mapping failure reason, expected IDs,
  observed events (rank, kind, point, actual/unknown IDs), and, when provable,
  the point that previously owned an unknown ID. The same evidence is retained
  in `profile-failure.json`. A failed generation also attempts a boundary
  snapshot; a snapshot error is recorded without replacing the original error.

`mapping_missing`, `mapping_conflict`, `invalid_mapping_provenance` and
`previous_point_event` are distinct conditions. None produces a cost-table PASS.
A successful receipt is not evidence of NPU correctness by itself.

The supplied server archive was not available on the local Mac. This repair
uses the provided raw `batch1-0` / `batch1-0-bea9330c` output and the frozen source;
it does not claim to have independently reread that archive.

## Server: focused tests and B64 only

Use the delivered signed-off commit as `SHA`. Preserve CANN/custom OPP and do not
build OPP. The wrapper checks exact Plugin/Core SHA, clean worktrees, environment,
checkpoint, focused tests and resource availability. Every invocation creates a
new result directory and retains child return codes, PIPESTATUS and evidence.
The supplied manifest is reused without import, tokenization or reordering:
400 request instances / 64 unique prompts, with prefix caching still disabled.
Profiling uses separate disposable synthetic requests as before; the manifest
is retained for the subsequent, separately authorized comparisons.

```bash
cd /workspace/vllm-ascend-hust
set -o pipefail
SHA='<delivered-40-character-commit>'
git fetch origin feat/dspark && git merge --ff-only "$SHA"
```

After that succeeds, run the focused gate and B64 cost profile:

```bash
MANIFEST='/workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json'
mkdir -p /workspace/dspark-results
PROFILE_LOG=$(mktemp /workspace/dspark-results/request-id-b64-driver.XXXXXXXX)
bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
  --stage profile --batches 64 --num-prompts 400 \
  --model /workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --profile-contexts 128 2048 --profile-output-tokens 512 \
  --profile-warmup 2 --profile-samples 5 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 2>&1 | tee "$PROFILE_LOG"
PROFILE_CODES=("${PIPESTATUS[@]}")
printf '%s\n' "${PROFILE_CODES[*]}" > "$PROFILE_LOG.pipestatus"
```

Check `MAIN_RC=0`, the driver/child PIPESTATUS values, and
`runs/b64/cost-profile.json`. The former failing point
`runs/b64/ctx128-n1-t6-balanced.json` must show actual internal-ID receipts and
successful attribution. `runs/b64/lifecycle.json` must report one completed
model initialization and shutdown, with all grid points retained. Target graph,
draft eager, TP8/EP and the generated capture grid are recorded in the existing
plan, capture and cost identity artifacts. Do not treat passing only the first
point as completion of the B64 table.

## After B64 cost-table completion: B128 and B256 only

Run this separately after reviewing successful B64 evidence. B256 runs only if
B128 completes. No validate/repeat performance stage is launched by these commands.

```bash
PROFILE_LOG=$(mktemp /workspace/dspark-results/request-id-b128-b256-driver.XXXXXXXX)
bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
  --stage profile --batches 128 256 --num-prompts 400 \
  --model /workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --profile-contexts 128 2048 --profile-output-tokens 512 \
  --profile-warmup 2 --profile-samples 5 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 2>&1 | tee "$PROFILE_LOG"
PROFILE_CODES=("${PIPESTATUS[@]}")
printf '%s\n' "${PROFILE_CODES[*]}" > "$PROFILE_LOG.pipestatus"
```

## Local validation boundaries

The focused tests execute the frozen Core bodies for ID assignment,
`AsyncLLM.add_request/_add_request`, collector construction and public output
construction. Model initialization and process transport are CPU substitutes;
this is not an installed AsyncLLM/NPU integration run. They exercise suffix
mapping, two points in one engine, old-ID injection, similar prefixes, external
ID reuse, duplicate prompt instances, invalid mappings, failure artifacts,
method restoration and cancellation. Existing performance, repeated-input,
cost coverage and graph telemetry tests remain part of the regression run.

Actual server model initialization count for this revision is **not measured**.
CPU lifecycle tests verify one initialization per configuration. Installed
Torch-NPU/vLLM/model execution, real FULL replay and complete B64/128/256 cost
curves remain **SERVER_NOT_REVALIDATED**.

Local regression result: **230 passed, 3 skipped** across request identity,
startup costs, repeated inputs, streaming/performance delivery, acceptance,
confidence verification and graph RPC/replay tests. The three skips require
installed vLLM/Ascend. All changed-file pre-commit checks pass. The required
`format.sh ci` run in an isolated worktree fails on repository-wide baseline
issues; it auto-modifies 78 unrelated files and zero files in this change. Those
unrelated edits are not included.
