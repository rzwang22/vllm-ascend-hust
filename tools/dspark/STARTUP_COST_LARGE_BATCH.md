# DSpark startup costs and large-batch performance

This is the DSpark confidence-versus-fixed-K performance experiment. It is not
V3.8 A1–A4 acceptance or a formal B0/B1 comparison. `SERVER_NOT_REVALIDATED`:
local CPU/reference tests do not validate Ascend execution or performance.

## Reference and scope

Pinned upstream PR [#47808](https://github.com/vllm-project/vllm/pull/47808)
head: `e2e335334669d1c94c7351937474c0104dcbfdfb`; merge:
`7f7a32cfec0f1bc5b73c37200b86631523a1ea8f`.
The [official description](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification)
is background; the implementation reference is the pinned source:

- `vllm/v1/worker/gpu/model_runner.py`: after graph capture, collect step timings
  over `AdaptiveVerificationManager.batches_to_profile()` within the loaded engine.
- `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py`:
  `_PROFILE_REPLAYS=5`, `set_initial_cost_curves()` takes medians; draft timings
  come only from FULL target steps. `build_cost_tables_from_curves()` applies
  a monotone envelope and graph-padding lookup; upstream also models eager tails.

Ascend now reuses a single isolated `StreamingEngine` across profile points.
Frozen core `897306c43bf800e2480cb5c0f3e2da408d85a2fd` has no upstream step-timing
collector. Rather than copy its runner or fabricate dummy KV, the plugin uses
normal synthetic requests, real admission, real FULL replay, and normal terminal
cleanup. This measures event spans around target replay and eager draft, including
host launch gaps inside those spans. It does not claim identical upstream timings.
There is no core, custom-op, attention or proposal ownership change.

Confidence remains current-producer-epoch host scheduling, with the existing
batched confidence D2H and TP CPU broadcast. Upstream uses stale CPU confidences
for budget selection and current GPU confidences for allocation. This change
aligns cost collection, not the whole decision implementation. Scoring, greedy
prefix allocation and uncalibrated probabilities are unchanged.

## Costs and lifecycle

Each B/capture configuration has one profile process and one engine initialization
(TP8 has eight worker replicas, not eight independent lifecycles). B64/128/256 therefore require **three**
profile engine initializations, not one initialization per point. Actual attempts,
completed initializations and shutdown are written to `b*/lifecycle.json` and the
profile index. No local NPU model was initialized while developing this change.

The request grid includes powers of two, graph capacities clipped to B, and B.
Every reachable graph tier is sampled for these request counts. Each cell measures
balanced and skewed contiguous prefixes, with actual query lengths in [1,6].
For example, request count 4 and capacity 24 do not imply all other points have
query length 6. Request capacity remains the actual graph descriptor's capacity;
it is never reconstructed from `tokens // 6`.

Defaults are prompt contexts 128 and 2048, synthetic output budget 512,
two discarded eligible warmup executions and five retained eligible samples
per rank, kind and point. Generation uses natural allocator-owned KV and unique
`batchN-request` IDs. Synthetic profile requests alone use ignore-EOS to obtain
samples; formal performance retains natural EOS. Every call is drained before
the next named point RPC. The RPC changes only profile events and the specified
prefix pattern; the scheduler alone frees KV and retires terminal proposals.
Profile events record request IDs, actual requests/tokens, padded token/request
capacities, query lengths and host context. Failed, prefill, nonmatching or
non-FULL-adjacent draft samples cannot price a cell. Insufficient samples fail
with raw evidence retained; increasing the profile output budget is explicit.
All events are integrity-checked before domain selection. Legal context/layout
exclusions remain annotated in raw evidence. See [context semantics and domain
selection](PROFILE_CONTEXT.md) for the physical-length checks and B64-first retry.

Schema 2 records source SHAs, checkpoint preflight (including actual index hash),
loaded confidence fingerprint, model revision/config, hardware, TP/EP, dtype,
quantization, Torch/torch-npu versions, Ascend compilation options, K, capture
sizes, model/token/memory budgets, block size, seconds,
raw sample hashes, medians, context/request grids and processing method.
Schema 1 is rejected for policy loading. Schema 2 now also requires explicit
`context_semantics` and `identity.cost_context_semantics` markers; older tables
must be regenerated without rewriting metadata.

Lookup first rounds actual target tokens up to graph capacity, then uses ceiling
request/context buckets. Raw per-layout/rank curves remain in the artifacts;
processed costs use max of rank medians and layout medians, then a monotone upper
envelope. Draft curves use request count and context, across FULL-adjacent target
capacities. Target retains capacity, request bucket and context dimensions.
The default context buckets end at 640 and 2560 scheduler pre-query computed
upper-bound tokens, not rejection-corrected KV lengths. Intermediate
contexts and request counts use a **fitted ceiling estimate**, not an assertion
that they were all measured. Balanced/skewed envelopes are not a proven bound
for every possible mixture or context distribution. This limitation is explicit;
no unvalidated one-dimensional target curve or out-of-range extrapolation is used.
Inputs needing longer contexts require explicit additional profile anchors within
`max_model_len`. All capacities and reachable request buckets must be covered.

Lookup entries are prepared once when loading the table. No per-candidate scan
of profile cells occurs. CPU policy timing is separately measured on synthetic
confidence rows with the actual table; this estimates allocator overhead, not
D2H/broadcast time. Runtime D2H, CPU allocation, TP broadcast and aggregate host
counter updates remain inside end-to-end measurement. NPU event recording and
phase synchronization are profile-only; detailed per-step decision history is
no longer enabled by the benchmark. Existing aggregated replay layouts remain.

## Frozen input preparation

Reuse an existing compatible performance manifest directly. The driver verifies
its hashes and at least 400 request instances. A 64-record manifest still fails;
there is no automatic loop expansion. To preserve a historical file that already
contains 400 instances / 64 unique prompts, import with explicit
`--allow-repeated-prompts`; see [historical input commands](REPEATED_INPUTS.md).
Never pair code results with old GSM8K runs.

If the old frozen file has final token IDs but no performance manifest, import
those IDs verbatim. Set the actual field name if it differs. No chat template,
re-tokenization, sorting or truncation is applied. Display text is decoded and
labelled as such; raw input tasks, original file hash, token hashes and order
are retained. Tokenizer provenance must match the earlier frozen run.

```bash
cd /workspace/vllm-ascend-hust
FROZEN_INPUT=/path/to/original/frozen-400.jsonl
FROZEN_SHA=$(sha256sum "$FROZEN_INPUT" | awk '{print $1}')
DATA_PARENT=$(mktemp -d /workspace/dspark-results/dspark-frozen-import.XXXXXXXX)
python tools/dspark/prepare_performance_data.py \
  --input-jsonl "$FROZEN_INPUT" --expected-source-sha256 "$FROZEN_SHA" \
  --frozen-token-field prompt_token_ids --source-name frozen-400 \
  --source-revision "$FROZEN_SHA" --kind general --num-samples 400 \
  --max-input-tokens 2048 \
  --tokenizer /workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
  --tokenizer-revision 9e8679a9db7eec11efed9925f7efb96549077545 \
  --output-dir "$DATA_PARENT/input"
MANIFEST="$DATA_PARENT/input/manifest.json"
```

## Server commands

Set `SHA` to the delivered full commit and `MANIFEST` to the verified input.
Sync normally (`git fetch origin feat/dspark`, then `git merge --ff-only "$SHA"`).
Keep the existing CANN/custom OPP environment. The wrapper checks exact source
SHAs and clean worktrees, sets TP8/EP MRV2 and secure serialization, checks loaded
confidence provenance, runs focused tests, and stops if another task owns an NPU.
It never builds OPP, installs packages, invokes Docker, kills a process or resets
a device. It creates a new result directory and archives success/failure evidence.

1. Generate costs for the three configurations. In the same Bash session retain
   the printed path for subsequent commands. This runs profile workloads only.

   ```bash
   set -o pipefail
   PROFILE_DRIVER=$(mktemp /workspace/dspark-results/startup-cost-driver.XXXXXXXX)
   bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
     --stage profile --batches 64 128 256 --num-prompts 400 \
     --profile-contexts 128 2048 --profile-output-tokens 512 \
     --profile-warmup 2 --profile-samples 5 2>&1 | tee "$PROFILE_DRIVER"
   PROFILE_CODES=("${PIPESTATUS[@]}")
   printf '%s\n' "${PROFILE_CODES[*]}" > "$PROFILE_DRIVER.pipestatus"
   PROFILE_ROOT=$(sed -n 's/^SERVER_RESULT_DIR=//p' "$PROFILE_DRIVER" | tail -n 1)
   COST_DIR="$PROFILE_ROOT/runs"
   test "${PROFILE_CODES[0]}" -eq 0 && test "${PROFILE_CODES[1]}" -eq 0
   ```

2. Only after profile succeeds, run each mode once at B64, then B128, then B256.
   An execution/resource/artifact failure stops the sequence, retaining evidence.

   ```bash
   bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
     --stage validate --cost-dir "$COST_DIR" --batches 64 128 256 \
     --num-prompts 400 --output-len 256 --warmup-prompts 4 --repeats 1
   ```

3. Only after the once-per-mode runs are valid, collect independent repeats:

   ```bash
   bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
     --stage repeat --cost-dir "$COST_DIR" --batches 64 128 256 \
     --num-prompts 400 --output-len 256 --warmup-prompts 4 --repeats 3
   ```

Default modes are `dspark_graph dspark_confidence_graph`. Add `target_graph`
through `--modes` for a separately labelled third mode. All requests are submitted
without an outstanding cap by default; explicit `--client-outstanding` must be
at least the largest tested B. Captures default to 6 times powers of two ending
at 384/768/1536. Target-only uses corresponding q=1 capacities. Explicit
`--capture-sizes` requires one B and the same configuration must be profiled.
Paths, contexts, total requests, concurrent requests, output budget, memory/token
budget and repeat count are independent CLI arguments. No B400 or larger sweep
runs automatically. Each formal repeat is a new process/load/warmup/measure/shutdown;
mode order reverses on even repeats. Neither profile nor warmup is formally timed.

## Reading evidence

- Top `stages.json`, command JSON, `.pipestatus` and logs identify failures and
  unrun stages. `*-evidence.tar.gz` includes all child evidence and cost assets.
- Profile `plan.json`, `capture.json`, point JSON and `retained.json` contain
  actual shapes, request identities, excluded warmup, samples and medians.
  `cost-profile.json` contains the processed curves and bounded lookup contract.
- Performance `b*/summary.json`, `.csv`, `.md` retain per-run values and independent
  median/mean/min/max/sample CV. `n<2` gives CV=null. The primary pair is
  `dspark_confidence_graph / dspark_graph` output tok/s, by B and repeat; the
  report also gives median-based ratios and raw per-pair speedups.
- `result.json` and `requests.json` retain outputs and frontend monotonic DELTA
  event times. TTFT is submit-to-first nonempty event including engine queueing;
  completion is submit-to-finish; TPOT=(finish-first)/(output tokens-1), null for
  <=1 token. Multi-token events are not fabricated into per-token timestamps.
- `confidence_verification.per_rank` contains verified/generated/accepted counts,
  actual position denominators, length histogram, aggregate confidence histogram
  (ten [0,1] bins), estimated seconds/progress sums, logical tokens before/after,
  selected capacities and the actual profile identity/coverage. These estimates
  do not establish measured throughput improvement. All-full decisions remain
  valid and report unchanged confidence scores and cost curves.
- `graph_execution.measured_runtime` and phase snapshots retain each rank's
  successful FULL execution and shapes. TP8 is counted once logically. Mixed
  learned-policy replay is reported when observed, never required or fabricated.
  Fallback coverage can be unavailable; no claim of 100% graph coverage is made.
- NaN/corrupted output, errors or absent measured replay cannot publish a valid
  performance result. Natural EOS length/text differences are reported, not
  exact-token gates. Quality stays unavailable in this performance-only entry.

Local validation uses real CPU Torch and source-body mocks of frozen interfaces.
Installed vLLM/torch_npu/model and actual profile/large-batch runs remain server
work. The initialization count measured by CPU lifecycle tests is one injected
engine across multiple points; this is not a hardware load measurement.

## Local checks for this delivery

The related CPU/source regression command completed with 303 passed, 1 skipped
and 103 runtime-name cases deselected. A separate first attempt including the
installed speculator test modules failed collection in 11 files because `vllm`
is not installed; those are not test passes. The server focused command retains
these modules. Real CPU Torch is 2.10.0; no torch_npu, model, CANN execution or
NPU initialization was performed locally. Model initialization behavior was
checked with an injected engine factory and actual profile controller.

Scoped manual pre-commit hooks passed. Full `bash format.sh ci` was run in a
disposable worktree and failed existing repository lint/format rules; it touched
78 unrelated files there and zero delivery files. Those baseline changes were
not included. Shell syntax and scoped shellcheck pass.
