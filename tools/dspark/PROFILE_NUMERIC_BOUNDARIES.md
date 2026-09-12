# DSpark profile: hidden and base-logits boundary diagnosis

This delivery implements diagnosis. The archived failure reproduced NaN; the new
instrumentation has **not** been run on NPU. Root cause is unproven and no
production repair has been validated. A successful diagnostic run would mean
only that NaN did not reproduce with this observation overhead.

## Reused and independently checked evidence

The supplied `audit.md`, `audit-data.json` and `verify_evidence.py` were read
before implementation. The original archive was available and independently
read, rather than relying only on the audit summary:

- Archive: `dspark-large-batch.QKmLESDh-evidence.tar.gz`.
- SHA256: `08e122fd7f6904a2fce4096d1ae816d04e939263c2429b5dd3339affb6b92a61`.
- Failed plugin: `3db4db06f27df18f056fc1026e075d10058788af`.
- Frozen Core: `897306c43bf800e2480cb5c0f3e2da408d85a2fd`.
- Both local source repositories were clean at those SHAs before implementation;
  `origin/feat/dspark` matched the plugin after fetch.

The 64 archive members were checked for relative paths, traversal, links and
special files before fresh extraction. The audit script's extraction-root and
output-marker paths were adapted in a separate local copy. Its original Mac
repository/archive paths were valid locally; **that script is not a server
entry point**. All 156 offline assertions passed again, including all extracted
file bytes, eight rank histories, raw point hashes, lifecycle/cleanup and the
real manifest reader. The manifest contains 400 instances / 64 unique prompts;
profile itself uses the original synthetic constant-token prompts.

One B64 engine ran the first ten points in order. The first nine completed;
`ctx128-n4-t12-skewed` failed. Every rank's first error was execution 1804,
proposal attempt epoch 1794, `markov.error`:
`Ascend DSpark Markov base logits contain NaN.`

| Actual row | Internal request ID | ell | Target query length | Target valid span | Draft candidate span |
| --- | --- | --- | --- | --- | --- |
| 0 | batch10-3-af04b61d | 0 | 1 | [0,1) | [0,5) |
| 1 | batch10-2-ac8cf6fd | 3 | 4 | [1,5) | [5,10) |
| 2 | batch10-1-801223de | 5 | 6 | [5,11) | [10,15) |

`query_start_loc=[0,1,5,11]`: three actual requests, 11 valid target tokens,
12 graph tokens, 15 draft candidates. Target output padding is not a draft
candidate row. Executions 1802/1803 published epochs 1792/1793; 1804 consumed
1793 and attempted 1794, which was never published. Later ownership errors are
downstream of this failure and do not identify its producer.

The visible complete executions 1798–1804 already have this same three-request
layout. The real four-to-three transition was evicted from the ordinary ring;
NaN cannot be assigned to the first exit/reordering from that evidence.
Rank-local request-state indices differ stably (0/4/5: `[63,61,62]`; other ranks:
`[63,62,61]`). Core's local free-list allocation can differ; those pool rows are
not batch rows or candidate publication rows. This does not prove a TP mapping
error. CPU scheduling upper bounds also differ semantically from rejection-
corrected attention lengths; neither addresses nor CPU descriptors prove device
indices, KV values or asynchronous buffer lifetimes correct.

The source/evidence agree on draft hidden `[15,4096]` entering
`compute_draft_logits` and returned full-vocabulary base logits `[15,129280]`.
Metadata-only captured no finiteness at either boundary. A `.return` event is
neither a device fence nor a numerical pass. Eight bad final vocabulary tensors
do not identify the first bad TP rank, operator or communication stage.

Source anchors: `AscendDSparkSpeculator._execute_sequential_markov_sampling`,
`_build_draft_forward_metadata`, `prepare_proposal_inputs`, `_release_consumed_proposal`
in [speculator.py](../../vllm_ascend/worker/v2/spec_decode/dspark/speculator.py);
`DSparkDeepseekV4ForCausalLM.compute_draft_logits` and
`DeepseekV4DSparkModel.compute_logits` in
[deepseek_v4_dspark.py](../../vllm_ascend/models/deepseek_v4_dspark.py).
The latter applies norm before the loaded LM head/logits processor. No edits to
these production methods, Core, custom ops, confidence allocation, model math,
ACLGraph execution or existing Markov checks are part of this delivery.

## Explicit opt-in and observation lifetime

Use `--profile-experiment numeric-boundaries`. Its default is absent. Existing
baseline, metadata-only and context-kv-sync controls retain their device
behavior; metadata-only gains only the CPU records described below. The new
mode uses metadata hooks and the existing process/engine exit protection, but
does not enable context-kv-sync or the broad `--profile-nan-diagnostic` observer.
CLI guards restrict experiments to an isolated B64 profile, mutually exclusive
with broad diagnostics. Installation occurs after capture on these instances.

The model hook reduces exactly the actual `compute_draft_logits` argument,
**before its internal norm**, and the actual return value, **before the existing
Markov check**. Each boundary calculates two boolean vectors using row-wise
`isnan().any(dim=1)` and `isinf().any(dim=1)`. Both Inf signs are recorded; an Inf
flag alone does not alter or fail the model's checks (negative infinity logits
may be intentional).

Reductions enqueue on the same caller stream as the eager draft and head, before
subsequent reuse can modify the hidden/input buffer. Only fresh reduced flags
survive the boundary. After head return, they are concatenated into an
`[candidate_rows,4]` boolean packet and copied to CPU with a blocking `.to()`.
There is **one additional compact D2H/host wait per completed head per rank**,
zero waits at the hidden boundary and zero new global synchronizations. The
packet is 4 bytes per candidate (60 bytes for the historical 15 rows); full
hidden/logits are never exported. Device reductions still scan each complete
boundary tensor and allocate temporary boolean masks, so small D2H volume does
not mean zero compute/memory overhead.

If the head raises before returning, a single hidden-only packet is attempted,
with logits flags `null`, while preserving the original exception. A device or
filesystem failure can prevent evidence collection: check `recording_error` and
missing/truncated files. The observer does not turn missing evidence into a
finite result, and point completion fails if observation reported an error.

Flags bind to the actual Markov `proposal_inputs`: point, rank, execution,
proposal epoch, current epoch fields, request ID, request row, and zero-based
candidate position. Row `i` maps using that proposal's `num_speculative_tokens`,
not target `query_start_loc`, target valid token count or graph padding. No
request count, shape, execution 1804 or epoch 1794 is hardcoded as a capture
window. No tensor is read after it has been returned for later-round reuse.

## Retention, files and interpretation

Worker schema 3 keeps an independent three-proposal numeric ring. It contains
the current boundary pair and the preceding two observed proposals in the same
point, or fewer if the point just began. Each record owns CPU values. Ordinary
metadata ring eviction cannot remove these numeric records.

Files in `runs/b64/worker-first-failure/` are rank-local:

| File suffix | Meaning |
| --- | --- |
| `rank-N-first-nan.json` | First NaN observed at either boundary plus up to two previous rounds; saved before returning to Markov |
| `rank-N-first-nonfinite.json` | First NaN or Inf, with its preceding rounds; may predate first-nan |
| `rank-N-first-failure.json` | Original caught exception and current histories; later ownership/cleanup errors cannot overwrite it |
| `rank-N-latest.json` | Last completed point snapshot; replaced at the next successful point boundary, never presumed to be post-error state |

First-NaN and first-nonfinite files are written and file-flushed/fsynced before
the model hook returns; the original NaN guard then runs unchanged. Thus an
uncatchable later device assertion still has an opportunity to leave evidence.
If first NaN and first nonfinite coincide, these are two file writes/fsyncs on
that rank; the caught first-failure adds one more. No normal finite-step disk
writes were added. These diagnostic waits and first-error writes can change
reproduction timing; they are not a production fix.

`numeric.rounds[].rows` contains `hidden_nan`, `hidden_inf`, `logits_nan`,
`logits_inf` for every actual candidate row. The round's `classification` means:

- `hidden_nonfinite`: start the next investigation upstream of this hidden input.
- `hidden_finite_logits_nonfinite`: narrow the next investigation to internal
  norm, projection/logits processing or intervening communication. Inf-only
  results require interpreting the model's allowed masks first.
- `both_finite`: these two tensors are finite for this local round only.
- `logits_unavailable`: the head did not return; do not infer its output values.

Classification narrows a boundary interval, never proves a specific rank or
operator originated the NaN. `numeric.classification_counts` and `nan_rounds`
cover the whole point, including rounds evicted from the three-round ring.
`compact_host_transfers` counts attempts; `compact_host_transfers_completed`
counts completed host packets. The existing `sync.calls` field counts only the
context-kv-sync experiment and remains zero here; it does **not** include D2H
waits. Compact point RPC receipts contain these counts and a hash/path to the
worker-local file, without transferring numeric histories through RPC.

Two CPU additions are available in metadata-only and numeric-boundaries:

- `draft_metadata.return.draft_decode_seq_lens` explicitly copies each layer's
  existing `decode.seq_lens_list` outside the generic descriptor depth budget.
  It carries the proposal epoch and is also copied into the numeric round.
  Null means that layer has no decode metadata. This adds no D2H; the existing
  builder's CPU length construction remains unchanged.
- `transitions` retains up to 16 request-ID **set changes per point**, independently
  of the 128-stage ring. `before` is the last observed prior layout; `after` is
  the first observed new layout, with scheduler receipt, actual row IDs, local
  state indices, target query spans/lengths, selected lengths/producer epochs,
  owner publication rows/epochs and proposal epoch fields. Preparation and
  publication snapshots are attached when reached in that execution. Later
  same-set reorders remain in ordinary metadata records. `transitions_seen`
  and `transitions_dropped` expose bounded eviction; no claim is made about an
  unobserved transition. Point reset clears both histories, while first-error
  files remain exclusive for the engine lifetime.

Every diagnostic snapshot/receipt stays `performance_eligible=false` or carries
`identity.diagnostic_only=true`. The existing profile collector returns before
cost compilation, and the compiler rejects diagnostic identity. Timings are
not usable costs or performance comparisons, even if all ten points finish.

## One server run

In the existing CANN/custom OPP environment, after receiving the new signed-off
SHA, use this single run. Keep the original manifest and frozen Core; no package
reinstall, operator rebuild, B128/B256 sweep or performance comparison is needed.

```bash
DSpark_SHA='<40-character SHA from this delivery>'
cd /workspace/vllm-ascend-hust &&
  git fetch origin feat/dspark &&
  git merge --ff-only "$DSpark_SHA" &&
  bash tools/dspark/run_dspark_profile_control.sh "$DSpark_SHA" \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
    numeric-boundaries
```

This reuses one B64 engine for exactly the original first ten profile points
through `ctx128-n4-t12-skewed`. It retains TP8+EP, seed 0, greedy sampling,
synthetic prompts, contexts 128/2048 (the stop point precedes context 2048),
512 output tokens, warmup 2, samples 5, captures `[6,12,24,48,96,192,384]`,
8192 model/batched-token limits, memory utilization 0.9, target FULL_DECODE_ONLY,
eager draft and block size 32. Source SHA/clean-tree, checkpoint, installed-state
focused tests, idle-resource checks, bounded cleanup, supervisor exit receipts
and failure archiving run through the existing wrapper.

Return the generated `dspark-large-batch.*-evidence.tar.gz` and its `.sha256`.
Those two files suffice; there is no need to copy tensor dumps separately. If
sending selected contents instead, retain all eight ranks' worker files, all
ten raw point JSONs that exist (including streaming request-ID mapping),
`diagnostic.json`, `plan.json`, `retained.json`, capture/lifecycle/cleanup and
failure/worker-exit receipts, `runs/b64.log`, `runs/b64-supervisor.json`, source
and focused-test logs, and top-level `status.txt`. Missing files must remain
explicitly missing. In a no-reproduction run, keep each completed point's
numeric counters and final worker latest files; do not report NaN repaired.

## Local validation scope

The focused tests use CPU Torch, mock engine/device streams and selected frozen
source method bodies. They cover hidden NaN, head-stage NaN, positive/negative
Inf row mapping, arbitrary candidate counts, epoch binding, in-place hidden
reuse, one compact host transfer, first-NaN evidence before the production
Markov guard, head exceptions, original-error preservation, independent history,
CPU transitions/lengths, compact receipts and disabled-mode behavior.

On a host without installed vLLM/Ascend, use `python -m pytest --noconftest` for
these standalone `tests/ut/test_dspark_profile_*.py` CPU/mock tests. The normal
repository conftest imports installed vLLM/Ascend and cannot run here. Three
existing confidence tests also require installed vLLM/Ascend and skip locally.
The server wrapper deliberately uses the ordinary installed-state pytest entry,
including `tests/ut/spec_decode/test_dspark_v2_*.py` and NPU-specific components.
Local tests do not validate NPU operator support, stream behavior, TP collectives,
ACLGraph replay or NaN reproduction. Server execution remains user-owned.

Delivery checks on Mac (Python 3.12, CPU Torch 2.14.0): **337 passed, 3 skipped**
in the related regression suite; skips require installed vLLM/Ascend. The server
shell entry was also executed against an argv recorder to verify its single B64
prefix invocation without starting inference. Changed-file pre-commit/manual
hooks and shell syntax passed. Required `bash format.sh ci` ran in a throwaway
worktree and failed on repository-wide existing lint (including unrelated Ruff,
archived text spelling, shell and forbidden-import issues). It auto-modified
78 unrelated files and zero delivery files; those changes were discarded.
This is not a claim that the full repository checks or installed NPU suite pass.
