# DSpark profile: target auxiliary output transfers

The subsequent server run reproduced NaN at the raw Target return. See
[the target boundary audit and next diagnostic](PROFILE_TARGET_BOUNDARIES.md)
for the verified `f9kqffNA` evidence, remaining gaps and current runbook.

This delivery is **diagnostic supplementation, not a production repair**.
The supplied upstream-boundaries run reproduced NaN at the actual auxiliary
consumer. The original target return and its transfer were unobserved. This
change brackets that gap without adding target-layer or drafter-layer probes.
The new NPU experiment remains user-owned and has not yet been executed.

## Independent archive audit

The archive was read, safely extracted, and every extracted regular file was
compared byte-for-byte by hash with its tar member. No instructions from archive
contents were used to direct execution.

- Archive: `dspark-large-batch.rlmP9WCW-evidence.tar.gz`.
- SHA256: `fa36f8bd0d91a854c9ebf66e62f14d33673ba4ab4c7cf0bf32e7c080a972e58e`.
- 96 members, 166437044 expanded bytes.
- Plugin: `acd74c75836ff37f84fe5a910a2af4591428f825`.
- Frozen Core: `897306c43bf800e2480cb5c0f3e2da408d85a2fd`.
- Both actual local checkouts were clean and matched these commits at start.
- Server `focused.log`: **848 passed**, 14 warnings. Nine completed point JSON
  SHA256 values match `runs/b64/retained.json`.
- Cleanup completed without error or timeout; supervisor `signals_sent=[]`.

[PROFILE_rlmP9WCW_AUDIT.json](PROFILE_rlmP9WCW_AUDIT.json) retains the independent
per-rank audit, five first-error member hashes per rank, the three rounds of
boundary summaries/device integers, transitions and completed-point hashes.
These are offline evidence assertions, not additional NPU tests.

All eight ranks have `recording_error=null`, 98 completed compact transfers,
`numeric.classification_counts={both_finite:97, hidden_nonfinite:1}` and
`upstream.counts={rounds:98, nan_rounds:1, inf_rounds:0}`. Their upstream histories
match across `first-failure`, `first-nan`, `first-nonfinite`,
`upstream-first-nan` and `upstream-first-nonfinite` files:

| Execution / proposal epoch | Auxiliary and context target rows | Draft candidate rows |
| --- | --- | --- |
| 1801 / 1789 | All observed values finite | All observed values finite |
| 1802 / 1790 | All observed values finite | All observed values finite |
| 1803 / 1791 | Aux 40/41/42, projected context and all three context KV write inputs: row 0 NaN | Initial hidden finite; all three MTP outputs, HC input/output and head hidden/logits: rows 0–4 NaN |

All other requests' observed rows are finite, with no Inf anywhere. Actual IDs:
`[batch10-3-85e30da4,batch10-2-93ae7b91,batch10-1-84f9ac01]`. Target query lengths
are `[1,4,6]`, ell `[0,3,5]`, CPU and device query starts `[0,1,5,11]`, 11 valid
target rows, graph capacity 12 and 15 draft candidates. Every round has
`target_mapping_matches_device=true` and `missing_boundaries=[]`.

For the affected request, target positions are 220, 221, 222 across the three
rounds; draft positions are 221–225, 222–226, 223–227. The actual sampled counts
are `[1,4,6]` and rejected counts `[0,0,0]` in all three rounds. Under the inspected
proposal construction, valid query ends therefore remain `[1,5,11]`. This does
not validate every attention index, historical KV value or earlier iteration.

All ranks retain transitions 1706→1707 (one to two requests), 1707→1708 (two to
four) and 1795→1796 (four to three). The NaN occurs seven executions after the
exit conversion. Pool rows remain rank-local: rank 1 `[62,61,63]`, ranks 4/6
`[63,62,61]`, other ranks `[63,61,62]`. They are not target or candidate rows.

Evidence anchors within the archive:

- `runs/b64/worker-first-failure/rank-N-upstream-first-nan.json` →
  `upstream.rounds`, `device_integers`, `boundaries`.
- `rank-N-first-failure.json` → `failure`, `records`, `transitions`, `numeric`.
  The first exception is `markov`: base logits contain NaN.
- `runs/b64/retained.json`, nine point JSONs, `cleanup.json`,
  `runs/b64-supervisor.json`, and `focused.log` establish completion/exit facts.
  The later snapshot RPC, ownership and EngineDead errors are consequences.

## Source audit and remaining gap

The real path at the frozen SHAs is:

1. [DeepseekV4Model.forward](../../vllm_ascend/models/deepseek_v4.py) appends
   `hidden_states.mean(dim=1)` after decoder layers 40, 41 and 42. Core maps
   zero-based layer IDs to output boundaries 41/42/43. These are HC-mean token
   tensors, distinct from the model's final normalized hidden and persistent
   flattened pre-HC buffer. At the Python/ATen level, `mean` is a new reduction
   result, not a view of a later in-place residual. Actual compiler/graph storage
   reuse is not proved safe solely by that source-level observation.
2. Frozen Core `vllm/v1/worker/gpu/cudagraph_utils.py`,
   `ModelCudaGraphManager.capture.create_forward_fn.forward_fn`, obtains
   `model_output`, then copies each raw aux into
   `self.aux_hidden_states[i][:num_tokens]`. Persistent storage is allocated at
   the first/largest captured shape; every smaller FULL graph writes its prefix.
   The copy is itself captured. There is no request-pool indexing in this copy.
3. `CudaGraphManager.run_fullgraph` first calls
   `get_offloader().sync_prev_onload()`, then exactly one
   `self.graphs[desc].replay()`. `ModelCudaGraphManager.run_fullgraph` returns
   prefixes of the persistent outputs. The plugin
   [ModelAclGraphManager](../../vllm_ascend/worker/v2/aclgraph_utils.py) delegates
   that ABI unchanged; the DSA task-update method is a no-op. Existing metadata
   copies and stream dependencies, including offloader ordering, remain intact.
4. Frozen Core `model_runner.py` retains those views in `ExecuteModelState`.
   `sample_tokens` takes local references and clears the state container, then
   passes the auxiliary list through the DSpark proposal path.
   [prepare_proposal_inputs](../../vllm_ascend/worker/v2/spec_decode/dspark/speculator.py)
   validates layer IDs, dtype and padded shape and retains the tensors in a tuple;
   it does not numerically validate or clone them. Query/pool mappings govern
   preparation, while these aux tensors remain packed in target token order.
5. `_combine_and_precompute_draft_context` slices each aux to
   `num_target_tokens` and concatenates features. The preceding run observed
   segments of **this actual concatenation** before `combine_hidden_states`.
   Thus graph padding is excluded at consumption. Scheduled verification rows
   include the queries evaluated by target; they are not candidate rows.

The current source shows same-caller-stream ordering through the graph output
copies and the subsequent consumer. No specific broken event, stale epoch,
wrong copy extent, cross-request gather or buffer alias was established by the
archive. Request exit/reorder changes packed row assignments and metadata, but
cannot by itself establish that a full token-prefix copy is incorrect. Ell=0
still provides one target query and drafts K5; that semantics is preserved.

**Proved:** auxiliary is nonfinite at consumption. **Unproved:** raw target
output before persistent transfer was already nonfinite. Aux 40 being bad does
not identify decoder layer 40 as the first producer. Matching CPU/device query
starts cannot validate all attention, compression, block table or KV metadata.
Target output addresses and successful replay returns are not numeric evidence.

The last execution's descriptors were checked on every rank: returned auxiliary
views are `[12,4096]` BF16 prefixes of the corresponding `[384,4096]` persistent
buffers, with equal data/storage pointers and zero offsets. The three declared
persistent auxiliary byte intervals are disjoint within each rank. This checks
the reported prefix/storage layout only; raw graph output aliases, intervening
writes and numerical values at target return remain unobserved.

The existing [P08-R9](P08-R9.md) implementation already has the correct local
raw-output snapshot location and an instance graph replay proxy. Its full
`ReplaySnapshots` also requires target-layer banks, a historical detailed window
and consecutive uniform-query sizes. Enabling it wholesale would enlarge this
experiment and does not match the variable-length buckets. This delivery reuses
`ModelWithContext`'s snapshot protocol and `DiagnosticReplayGraph` only.

## Default-off auxiliary-transfers experiment

`--profile-experiment auxiliary-transfers` is a separate option. Existing modes
retain their meanings. The new capture collector is created only for an
isolated specified-length profile with auxiliary outputs, FlashComm1 off, DP1,
no LoRA and no full/R9 diagnostic. It installs before FULL capture. The profile
observer attaches after capture; capture/warmup data has no real execution owner.
There is no target-model patch, new compiled model layer hook or drafter layer
instrumentation. Core, custom ops, confidence allocation, mixed-length
attention/ACLGraph semantics and existing failure checks are unchanged.

For each actual captured size, the raw collector owns three separate auxiliary
buffers and per-output device receipts. Concrete capture-closure sizes select
these banks; this is outside the compiled model and does not use a symbolic
bucket-selection expression. Nonconsecutive sizes `[6,12,24,48,96,192,384]` work.

| Boundary | Timing and data |
| --- | --- |
| `raw.<layer>` | Independent D2D snapshot at `ModelWithContext` return, before Core's persistent copy; stats reduced immediately after actual replay |
| `persistent.<layer>` | Actual persistent prefix immediately after replay, before runner sampling/proposal work; NaN/Inf flags and per-row inequality against the owned raw snapshot |
| `consumed.<layer>` | Feature segment of the actual valid-token concatenation, before context projection; NaN/Inf and per-row inequality against that same raw snapshot |

Raw and persistent records include the graph padding rows, explicitly marked
`valid_target_row=false`, with null request identity. Only valid target rows
latch the first anomaly. Consumption records have only valid rows. No tensor is
modified to compute comparisons; paired NaNs compare equal for transfer checking
but still independently set the NaN flag. Comparisons are numeric equality,
not bitwise equality (e.g. signed zeros compare equal).

Before each actual graph replay, receipts are cleared and armed with the real
observer execution ID. Graph-internal per-output `copy_` publishes that ID after
the raw snapshot. After replay, receipts and raw/destination row statistics are
copied/reduced while this execution owns them. Consumption takes another owned
receipt snapshot. A missing, old or changed receipt, failed replay or missing
boundary is explicit and cannot pass as finite evidence. These receipts prove
the snapshot publication path ran; they do not prove every target kernel or KV
read was correct. A no-op replay cannot reuse capture-time values as evidence.

Records bind point, rank, actual execution, new proposal epoch, CPU query spans,
request IDs and distinct pool rows. Owned integer snapshots include pre-replay
input IDs/positions/query starts/sequence lengths/pool rows/padding, the actual
captured input references, and captured attention query/sequence/position fields.
Identical metadata/tensors are referenced once. Proposal-time counterparts and
sampled/rejected counts allow offline comparison across the transfer interval.
Full KV contents, block tables, full RoPE tables and unrelated target internals
are outside this observation. `target_mapping_matches_device` covers only the
named query starts; inspect captured attention fields separately.

For eager/prefill calls the raw transfer is **unobserved**, explicitly
`NON_FULL_RAW_UNOBSERVED`; actual consumption may still be recorded. A target
that produces no next proposal is drained as a partial record, with null proposal
epoch and missing consumption. It is not silently attached to the next request
layout. This preserves visibility across exits without imposing a new model
check or requiring an extra draft. Histories keep the current target execution
and preceding two records within the point, including explicit partial records.

## Cost and evidence limitations

Each captured target execution adds three auxiliary D2D copies plus three receipt
copies. For BF16, hidden width 4096 and the seven original sizes, raw banks use
18726912 bytes; receipts use another 224 bytes, approximately **17.86 MiB/rank**.
They are allocated during capture warmup, after KV memory profiling, and can alter
memory layout/headroom. For capacity 12, the additional raw payload copied per
execution is 294912 bytes per rank. No full floating tensor leaves the device.

Outside the graph, each of the nine auxiliary boundaries adds NaN/Inf row
reductions; persistent and consumed boundaries also compare against raw. Masks
and comparison temporaries scale with that boundary's token/feature size and
are discarded after reduction. Only fresh row flags and owned compact integer
snapshots wait for serialization. Layout descriptors contain no device values.

All auxiliary flags/receipts/integers and the existing head hidden/logits flags
join **one int64 packet, one blocking D2H/host wait per completed proposal/rank**.
No per-boundary host wait or global synchronize is introduced. A target without
a head drains one partial packet on failure, before the next execution, or at
point end. Therefore the transfer counter counts packets, including partial
target records. `auxiliary.packet_bytes` reports actual accumulated bytes;
`auxiliary.capture_bytes` reports allocated banks/receipts. Metadata sharing and
actual capacities determine packet size; no historical epoch/window is hardcoded.

Normal finite/equal rounds add no files. First valid-row NaN, first nonfinite
and first source inequality each get an independent exclusive, flushed/fsynced
file before the original Markov guard can raise. With simultaneous auxiliary
and head NaN this is four anomaly writes (five if there is also a first transfer
inequality), then the original first-failure file. Device/storage failures can
prevent complete evidence; `recording_error` and coverage must be checked.

These copies, reductions, allocations, waits and exceptional disk writes can
change reproduction. `performance_eligible=false` and cost-compilation rejection
remain enforced. Completion without NaN means **not reproduced under this
observation**, never a repaired inference or usable cost table.

## Artifacts and next decision

Worker files live in the existing `worker-first-failure` directory:

- `rank-N-auxiliary-first-nan.json`, `auxiliary-first-nonfinite.json`, and when
  applicable `auxiliary-first-difference.json`: first occurrence and prior rounds.
- Existing `rank-N-first-nan.json`, `first-nonfinite.json`, `first-failure.json`
  also include `auxiliary.rounds` in this mode. Auxiliary-first files contain the
  current combined `head_flags`; their base numeric history may end one round
  earlier because the head packet is decoded after the auxiliary save.
- `rank-N-latest.json`: last completed point. RPCs carry compact counts and the
  hashed worker-file receipt, not full histories.

First require `recording_error=null`, current execution/point/proposal mapping,
`raw_replay_verified=true`, `coverage=FULL`, and no missing boundaries for a
complete proposal. Then compare the same layer and actual target row:

- Raw NaN/Inf already present: bracket lies at or before the raw target return.
  Only then investigate target internal last-finite/first-bad boundaries.
- Raw finite, persistent nonfinite: investigate captured output transfer,
  destination aliasing or writes up to the replay-return observation.
- Both finite, consumed nonfinite: investigate post-replay lifetime, proposal
  views and concatenation. The minimal probe does not separately snapshot the
  pre-concatenation view, so it cannot distinguish that final pair yet.
- All finite but a transfer comparison differs: retain the evidence and analyze
  the concrete changed rows; inequality alone is not proof of the NaN producer.
- Missing/invalid replay receipt or non-FULL coverage: do not infer raw finiteness.

Agreement of all ranks still cannot identify the first failing rank/operator.
Original Markov/ownership checks, point order, supervisor and cleanup behavior
are retained. Nothing skips the failing point or substitutes a smaller layout.

## One next server run

Use the delivered signed-off SHA in the unchanged CANN/custom OPP environment.
Core remains frozen. This runs only the original single-engine B64 first-ten
synthetic profile points; no B128/B256, full cost table or performance comparison.

```bash
DSpark_SHA='<40-character signed-off SHA from this delivery>'
cd /workspace/vllm-ascend-hust &&
  git fetch origin feat/dspark &&
  git merge --ff-only "$DSpark_SHA" &&
  bash tools/dspark/run_dspark_profile_control.sh "$DSpark_SHA" \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
    auxiliary-transfers
```

The original script keeps TP8+EP, K5, greedy/seed 0, MRV2, target FULL_DECODE_ONLY,
eager draft, captures `[6,12,24,48,96,192,384]`, contexts 128/2048, output 512,
warmup 2, samples 5, 8192 limits, block size 32 and memory utilization 0.9. It
checks source/config and installed-state focused tests before the new experiment.
Return only the generated `dspark-large-batch.*-evidence.tar.gz` and
`dspark-large-batch.*-evidence.sha256`; the archive includes all rank evidence,
point metadata, commands, logs and shutdown/failure receipts.

## Local validation

On macOS with Python 3.12 and CPU Torch 2.14.0:

- **415 passed, 3 skipped** in the related standalone suite below. The skipped
  confidence-verification cases require installed vLLM/Ascend; source/CPU paths
  still ran. This includes the existing R9 replay/receipt regression suite.
- **26 passed, 27 deselected** for the capture ABI/source selection. Installed
  runtime variants were not selected locally; the server focused suite runs
  the whole capture file with its normal fixtures.
- Changed-file manual pre-commit hooks (including Markdown/shell), shell syntax
  and `git diff --check` passed.
- Required `bash format.sh ci` ran with the staged patch in an isolated worktree
  and returned 1. Failing hooks and 78 autoformatted files exactly match the
  preceding upstream-boundaries delivery; none belongs to this change. Existing
  failures include Ruff, spelling, Markdown, workflow/shell and forbidden-import
  checks. Those unrelated automatic changes were discarded.

```bash
python -m pytest --noconftest -q -ra \
  tests/ut/test_dspark_profile_auxiliary.py \
  tests/ut/test_dspark_replay_diagnostics.py \
  tests/ut/test_dspark_profile_upstream.py \
  tests/ut/test_dspark_profile_numerics.py \
  tests/ut/test_dspark_profile_observation.py \
  tests/ut/test_dspark_profile_nan.py \
  tests/ut/test_dspark_profile_failure.py \
  tests/ut/test_dspark_profile_context.py \
  tests/ut/test_dspark_profile_request_ids.py \
  tests/ut/test_dspark_startup_cost_profile.py \
  tests/ut/test_dspark_repeated_inputs.py \
  tests/ut/test_dspark_nan_diagnostics.py \
  tests/ut/test_dspark_confidence_verification.py \
  tests/ut/test_dspark_graph_rpc.py \
  tests/ut/test_dspark_graph_replay.py \
  tests/ut/test_dspark_acceptance_benchmark.py \
  tests/ut/test_dspark_performance_delivery.py
python -m pytest --noconftest -q tests/ut/worker/test_aclgraph_capture.py -k 'source or abi'
```

New CPU tests execute the actual `ModelWithContext`, frozen Core's output-copy
statements and FULL replay dispatch with an ATen recorder. Replay runs recorded
operations without another Python target forward. Fault injection separates
raw-source corruption, persistent destination corruption and later consumption
corruption. Coverage includes NaN/Inf, valid versus padding rows, finite-value
transfer inequality, alternate shapes/two requests, reordered rows, proposal
binding, history, partial/no-proposal records, missing hooks, stale/missing
receipts, no-op/failed replay, storage errors, default-off capture and the real
profile factory. Tests do not establish NPU graph-copy support, TP/EP timing,
true attention/KV values or a production repair. New hardware validation remains
pending; no performance measurements from this diagnostic are admissible.
