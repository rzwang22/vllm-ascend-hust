# Startup cost context and sample domains

## Source audit and scope

The supplied `pk2MnJJT` archive is not available on this Mac. The audit uses the
provided raw point summary and frozen Core
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`; it does not independently verify the
archive or infer physical KV lengths that were absent from those records.

The previous `context=643` is `InputBatch.num_computed_tokens_np.max()`, captured
in the worker around a successful target/draft execution. It is a snapshot of
scheduler-side progress **before the current query**, not effective device KV.
The array uses the current request index mapping. Core
`Scheduler._update_after_schedule()` advances scheduler counts by scheduled
tokens; `AsyncScheduler._update_after_schedule()` tracks output placeholders.
`Scheduler.update_from_output()` later removes rejected tokens. The asynchronous
budget guard explicitly accounts for unresolved output placeholders and does
not shorten the last speculative query merely to match the final output cap.
These source facts explain why an upper bound and an output budget need not
match; the numerical difference of three is not proof of a legal physical KV
access in this particular execution.

| Field | Meaning and availability |
| --- | --- |
| `context` | Maximum scheduler pre-query computed upper bound over actual request rows. May include unresolved speculation. |
| `scheduler_computed_upper_bounds` | Per-row scheduler values used to derive `context`; no padded rows. |
| `effective_kv_before_query` | Existing corrected CPU attention length minus the current query length, in batch request order. |
| `attention_seq_lens` | Corrected prior KV plus current scheduled query, before this target verification's acceptance result. |
| `context_ceiling` | Synthetic prompt + requested final output budget; sampling/query bucket boundary, not a physical KV limit. |
| `max_model_len` | Configured attention length limit used in execution-integrity checks; not replaced by the output budget. |

Ascend `update_requests()` copies rejection-corrected device computed lengths to
CPU through its existing copy stream. `_update_seq_lens_cpu()` waits on that
existing event and builds `seq_lens_cpu` in current request order.
`prepare_inputs()` passes the NumPy alias as `AscendInputBatch.seq_lens_np`.
The profile snapshots these already-available CPU arrays before each timed
call. Values are detached into lists, restricted to `num_reqs`, and remain
stable across updates to persistent input buffers. No new D2H or device wait is
introduced. The adjacent draft timing is indexed by its target's context,
not represented as an independently measured post-acceptance draft KV length.

Runtime selection happens before `runner.update_requests()/prepare_inputs()`.
`current_host_contexts()` overlays this `SchedulerOutput`'s cached-request counts
on host state; the policy queries the maximum over the scheduled active rows.
That is the same scheduler pre-query upper-bound convention used by profiling.
Switching only profiling to effective KV would mix two different cost axes.
The policy, admission, scoring, allocation, attention and graph execution are
unchanged by this repair.

## Selection and compatibility

Every raw event is checked for finite positive timing, nonnegative lengths,
query/row/capacity consistency and physical attention length within configured
`max_model_len`, including events that will not be sampled. Worker execution
failure and request identity gates still apply to the whole point.

Only after integrity checks are events classified by the predetermined FULL
layout and scheduler-context domain `[0, context_ceiling]`. Legal events outside
either domain remain in the raw artifact as `out_of_domain`, with a reason and
raw index. The first two eligible events are warmup and the next five are
retained with default arguments. Later eligible events are `extra`. Selection
never depends on speed. Insufficient eligible samples fails the point, and all
raw values and exclusion evidence remain available. No upper-bound tolerance,
clamping, shortened sample count or per-point engine restart is used.

For the synthetic regression of 171 matching events with a physically legal
last event at scheduler context 643 and ceiling 640: indices 0–1 are warmup,
2–6 retained, 7–169 extra, and 170 is outside the context domain. If that event
has an illegal timing, physical length, layout or request owner, the point
fails instead. The actual server event's corrected lengths remain unverified.

Cost schema 2 now requires the identity field `cost_context_semantics` and the
table field `context_semantics`, both
`scheduler_pre_query_computed_upper_bound_v1`. Runtime rejects missing or
incompatible markers; previous tables must be re-profiled, not relabeled.
The repeated-input manifest's schema 2 is independent and remains unchanged.

Bucket ceilings and the existing ceil lookup remain unchanged. Costs are
layout/rank/context envelope **estimates**, not measurements taken at each
bucket ceiling or guarantees for all lengths/layouts in the bucket. Cells now
retain each layout/rank/kind's actual selected context range. No coverage is
extended to accommodate 643; lookup beyond the final bucket still fails.

## Server commands: profiles only

Set `SHA` to the delivered signed-off commit. Keep the existing CANN/custom OPP
environment; no OPP build or device reset. The wrapper preserves TP8+EP, MRV2,
K=5, target FULL_DECODE_ONLY, draft eager, generated capture sizes and prefix
cache disabled. It checks exact dual-repo SHAs and resource availability, runs
focused tests, and creates a fresh result directory with logs, PIPESTATUS,
per-point evidence, lifecycle counts and a tar archive.

```bash
cd /workspace/vllm-ascend-hust
set -o pipefail
SHA='<delivered-40-character-commit>'
git fetch origin feat/dspark && git merge --ff-only "$SHA"
```

After source synchronization succeeds:

```bash
MANIFEST='/workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json'
mkdir -p /workspace/dspark-results
PROFILE_LOG=$(mktemp /workspace/dspark-results/context-b64-driver.XXXXXXXX)
bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
  --stage profile --batches 64 --num-prompts 400 \
  --profile-contexts 128 2048 --profile-output-tokens 512 \
  --profile-warmup 2 --profile-samples 5 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 2>&1 | tee "$PROFILE_LOG"
PROFILE_CODES=("${PIPESTATUS[@]}")
printf '%s\n' "${PROFILE_CODES[*]}" > "$PROFILE_LOG.pipestatus"
```

Require successful `ctx128-n2-t6-balanced.json`, complete
`runs/b64/cost-profile.json`, `MAIN_RC=0` and lifecycle initialization count 1.
Inspect each raw event's `sample_selection`, and `retained.json`'s
`selected_raw_indices`, `selected_context_range`, `excluded_samples` and counts.
Cost cells retain `selected_context_ranges`; these must not be read as timings
at `context_ceiling`. Failure artifacts retain the original events and errors.

Only after the entire B64 table passes, separately run:

```bash
bash tools/dspark/run_dspark_large_batch.sh "$SHA" "$MANIFEST" \
  --stage profile --batches 128 256 --num-prompts 400 \
  --profile-contexts 128 2048 --profile-output-tokens 512 \
  --profile-warmup 2 --profile-samples 5 \
  --max-model-len 8192 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9
```

B256 starts only after B128 succeeds; each configuration has its own single
engine lifetime. No validate/repeat stage is launched. The frozen manifest
remains 400 request instances / 64 unique prompts in its original order;
profile requests remain the existing separate disposable synthetic workload.

Local CPU/source/mock tests do not validate NPU execution or complete cost-table
generation. **SERVER_NOT_REVALIDATED** until these server runs complete.

Local result: **248 passed, 3 skipped** across context, request identity,
startup cost, repeated-input, performance delivery, acceptance, confidence and
graph RPC/replay regressions. The skipped tests require installed vLLM/Ascend.
Changed-file pre-commit checks pass. The required repository-wide `format.sh ci`
run in an isolated worktree fails on existing baseline issues; 78 unrelated
files are auto-modified and zero task files are changed. Those unrelated edits
are excluded from this commit.
