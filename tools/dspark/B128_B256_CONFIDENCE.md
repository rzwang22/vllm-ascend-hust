# B128 → B256 costs and real confidence acceptance

B64 is frozen at Plugin `082c8ea67b32ed1c71a408de93c53abacefb5af3`, Core
`71d2c1c436eba894a8e9eeb2c5af17e05cb42970`, policy `dspark-profile-25s-v1`.
`B64_CONFIDENCE_ACCEPTED.json` records the independent archive reconstruction:
outer SHA `a76a927b2d5f354f9f8e7f6e223fe15722091f35733c213aa2616d6f6e27e118`,
embedded model SHA `08581ac6689389879c3dce293794914136694eb9ec9983cc2d5fe5ae3ae9c573`.
The auditor checks the frozen manifest/source/token hashes, copied input hash,
cost publication and weight receipts, all eight ranks' decisions and actual
execution/acceptance, reconstructed counters, shutdown, PIPESTATUS and logs.
64 requests completed, 68 confidence FULL executions, 11 mixed-length executions,
eight natural zero exits. Weight shards are not in the archive; their recorded
identities agree, and server preflight rehashes the actual shards.

The original ten points and two synthetic functional phases remain passed.
The original five-second exit budget remains NOT_EVALUATED; the native destructor
tail remains unlocalized. Neither reopens the completed B64 named-policy gate.
The B64 cost file and its SHA are unchanged. This archive records an actual
`https://github.com/rzwang22/vllm-hust.git` fetch through `rzwang`; earlier
`file://` receipts retain their narrower scope. No remote URL is rewritten.

## Capacity audit and limits

| Parameter | B128 | B256 |
| --- | --- | --- |
| `max_num_seqs` / submitted instances | 128 | 256 |
| Distinct GSM8K questions | 64 | 64 |
| Instances per question | 2 | 4 |
| Target query tokens at full K5 | 768 | 1536 |
| Draft candidate rows at K5 | 640 | 1280 |
| Capture sizes | 6,12,24,48,96,192,384,768 | 6,12,24,48,96,192,384,768,1536 |
| Cost points / sampling requests | 48 / 2258 | 56 / 4562 |
| Synthetic calibration output tokens | 1156096 | 2335744 |
| Real-text output upper bound | 32768 | 65536 |
| Retained target+draft event samples, eight ranks | 3840 | 4480 |
| Candidate cost lookups checked at publication | 41408 | 164736 |

Both keep the same weights, TP8+EP, K5, target FULL_DECODE_ONLY, draft eager,
max model length 8192, max batched tokens 8192, memory utilization 0.9,
prefix caching disabled, CANN/custom OPP environment and sampling parameters.
128/256 are request capacities, not token capacities. The report separately
records actual scheduled requests, valid target tokens and graph capacity.

Source audit found no 64-request constant in the relevant production allocations:

- Ascend `worker/v2/model_runner.py` obtains request/input-state capacity from
  `max_num_reqs`; its query-start host array has `max_num_reqs + 2` entries.
- Ascend DSpark inherits Core `BaseSpeculator`, **not** `DraftModelSpeculator`.
  It has no `max_num_reqs` member. The previous reference to Core's separate
  drafter allocation path did not establish Ascend's interface and is corrected.
  Ascend `set_attn()` binds the runner's actual `BlockTables` and `KVCacheConfig`;
  proposal tensors are created for the current execution.
- Frozen Core `v1/worker/gpu/block_table.py` allocates per-group tables with
  `max_num_reqs` rows. Ascend DSpark validates shared cache views, group mappings,
  and slot-buffer lengths in `speculator.py`; the SWA lifecycle fix is retained.
- CostTable startup validation already iterates all request counts up to the
  identity's max. The new entry derives the ceiling request grid from capture
  buckets and the tier maximum. Completion tails are covered down to one request.

This is a source/capacity-contract audit, not proof of NPU memory fit or kernel
correctness at the larger sizes. Actual per-rank KV bytes, blocks, groups,
page sizes, tensor sharing, runner capacity and captured sizes are saved by one
host-descriptor RPC after initialization and checked before sampling/generation.
Draft request capacity is the minimum allocated row dimension of its active KV
groups' stored and input block tables; token capacity is the allocated slot-map
width. The receipt records each shape and checks the shared target binding.
No expected tier, configuration fallback or invented drafter attribute is used.
No tensor reduction, D2H or synchronization is added by this RPC. `npu-smi`
receipts preserve actual device usage. Weights, persistent tensors, graph pools,
KV and workspaces must fit the configured 90% memory allowance; no unsupported
GiB estimate is used as an acceptance result. OOM, allocation failure, or failure
to execute all 128/256 requests together fails that tier. No capacity reduction
or eager fallback is introduced.

## Fresh costs and input identity

`B128_B256_EXPANSION_PLAN.json` is the exact pre-load point/instance manifest.
It is also printed before host preflight or model loading. Each cost engine
visits all its points without reinitializing. Request grids are
`1,6,12,24,48,96,128` and `1,6,12,24,48,96,192,256`. For each legal reachable
Graph cell the existing balanced/skewed layouts are measured, using 128 input
and 512 synthetic output tokens, two warmups and five retained samples per rank.
These are calibration points required by the lookup contract, not another
synthetic functional matrix. Target FULL and eager draft timings remain separate.

Each table is rebuilt from raw NPU event records, with max-rank/layout medians
and a monotone envelope; this is a conservative estimate, not a proven latency
upper bound. The context ceiling remains 640. Missing cells, incompatible
identity, or queries outside request/context bounds fail. Twenty host allocator
samples are independently retained. Confidence-head/D2H/TP/end-to-end overhead
is not claimed to be fully represented by these isolated timings.

B128/B256 require fresh cost files produced by the **same exact commit** as their
consumer, full weight-shard/config identity, real hardware/runtime identity,
matching capture sizes and publication proof. There is no relabeled B64 data or
cross-tier reuse. The change affects host tooling, capacity receipts and opt-in
receipt bounds; it changes no model, Core, custom op or allocation algorithm.
The old B64 compatibility allowlist remains frozen and is not broadened to
approve this new commit against old timing data.

For real text, the original 64 records and tokens are reused verbatim in
`input.jsonl`; `preflight.json` maps each source record/hash to the unique external
instance ID `ORIGINAL:b128:instance0/1` or `ORIGINAL:b256:instance0/1/2/3`.
The original manifest/assets remain intact. Actual generation uses these unique
IDs and records the mapping to internal IDs. Natural EOS, at most 256 output
tokens, temperature 0, top-p 1, top-k -1 and seed 0 are unchanged.
Calibration remains **uncalibrated**. All-K5 selection is valid if it occurs
naturally; scores or outputs are not altered to force length diversity.

## Gates, overhead and bounded execution

Each rank must have an actual current-epoch confidence decision consumed by a
FULL target execution with exactly 128/256 requests. Corresponding valid query
counts, graph capacity, costs, owner epochs and accepted counts must agree across
all eight ranks. Prefill may run normally and cannot satisfy this concurrency
witness. Submitting all clients or observing all requests over separate smaller
executions cannot pass. The existing numerical/owner checks and strict log scan
remain active; this is not an all-layer finite-value claim.

Receipts remain opt-in: 2048 executions, 512 × tier request rows and 1 MiB × tier
JSONL bytes per rank. Overflow fails explicitly. Accepted-count clones use the
existing consuming stream and are transferred together at the final snapshot;
no layer probes, operator capture, write timeline, gdb or new global sync.
Exit policy remains worker 25 s, TERM 4 s, shared reap 1 s, engine 36 s,
frontend 40 s, supervisor 48 s; no added observation wait. Every worker must be
reaped with actual exit code 0; missing codes, cancellation errors, escalation,
timeout or residual processes fail.

| Stage | Limit |
| --- | --- |
| Host regressions, no weights | 600 s |
| B64 archive audit, no model | 1800 s |
| Per-tier preflight / publication / confidence preflight | 1800 s each |
| Per-tier cost engine including initialization | 7200 s + supervised cleanup |
| Per-tier cost wrapper hard deadline | 7400 s |
| Per-tier confidence engine including initialization | 3600 s + supervised cleanup |
| Per-tier confidence wrapper hard deadline | 3800 s |
| Individual wrapper TERM→KILL margin | 15 s |
| Sum of all stage limits/margins | 35780 s |
| Whole task hard runtime / cleanup margin | 36000 s / 65 s |

At most four model initializations: B128 cost, B128 confidence, B256 cost, B256
confidence. These are ceilings, not duration predictions. Two full tier stages
can take substantially longer than B64. Source fetch/environment checks precede
the task deadline; evidence export follows it and is not a model-running budget.
Every stage saves its actual return code, log PIPESTATUS and elapsed time.
Failure stops later stages and preserves any published cost or completed tier.
No retry, no performance run, no automatic extra experiment.

## Server task

Use the existing activated Torch/NPU/CANN/custom OPP environment. The manifest
and frozen B64 archive must remain available at their original paths. Replace
`DELIVERY_SHA` with the exact signed-off commit returned with this delivery:

```bash
if bash -c '
  set -euo pipefail
  cd /workspace/vllm-ascend-hust
  test -z "$(git status --porcelain)"
  git fetch origin feat/dspark
  git merge --ff-only "$1"
  test "$(git rev-parse HEAD)" = "$1"
  bash tools/dspark/run_dspark_batch_expansion.sh "$1" \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json rzwang
' _ DELIVERY_SHA; then
  echo "B128 and B256 task passed; inspect both actual-concurrency reports"
else
  rc=$?
  echo "Task failed (rc=$rc); retain first error and earlier passed stages"
fi
```

Return the single `dspark-batch-expansion.*-evidence.tar.gz` plus its SHA256.
It embeds the newly created `dspark-large-batch.*` bundle containing both tier
costs, original inputs, outputs, decision/execution receipts, capacity reports,
all exit receipts, logs, JUnit and PIPESTATUS. `expansion-report.json` separates
tier results; `cost-publication.json` and `confidence-acceptance.json` retain
stage-specific facts. Nonzero export status also fails the outer task.

Local CPU/mock checks validate the publication/lookup/CLI/receipt/stop contracts.
They do not establish NPU throughput, memory fit, real larger-concurrency FULL
execution or natural exit. All B128/B256 NPU stages remain **PENDING** until this
single server task returns. Only after both tiers pass will a separate task
compare fixed K5 and confidence for B64/B128/B256 on identical real inputs,
including confidence head, D2H, host scheduling and TP communication costs.

## Local validation at delivery

Python 3.12.13 / Torch 2.10.0 CPU: **160 passed, 3 skipped** across the expansion,
confidence acceptance, formal costs, startup profile, shutdown policy and
confidence verification regressions. The skipped installed-vLLM/Ascend cases
cover the real builder, confidence head and Graph path; they were not substituted
with NPU success claims. The new expansion file has 17 passing CPU/mock cases.
Tests exercise real CostTable construction and every candidate lookup,
publication from mock raw event records, both parent/child CLI paths, actual
engine-argument construction, insufficient real concurrency, stale epochs,
cross-rank disagreement, bounded receipts and first-failure stage stopping.

All changed-file pre-commit/manual hooks pass. Required `bash format.sh ci`
was run in a disposable checkout: exit 1 from pre-existing repository-wide lint,
format, spelling, workflow, shell and forbidden-import issues; 78 unrelated
files would be auto-modified, zero task files. Those unrelated edits were not
imported. This is not a claim that the full repository CI passed.

## Capacity RPC failure and correction (DKPzjeew)

`B128_CAPACITY_FAILURE_AUDIT.json` preserves independently read evidence anchors
and hashes. Outer SHA is
`367034e86d5b1d678f1cce950c23aaa1f1b717ecb1f93bb84ea61e81121f2adb`;
embedded SHA is
`05d43efefeadfa71823a6fd5679fcf075cebf8a6d64bf106d41634e4176b9d72`.
Plugin `008884b24320c75ddb8b52205335e3f171fb0690` and Core
`71d2c1c436eba894a8e9eeb2c5af17e05cb42970` match the archived plan/source receipt.
All eight ranks captured 6 through 768 tokens. The capacity RPC then failed on
`runner.speculator.max_num_reqs`. No sampling point started, retained samples
or new cost tables exist; B128 confidence and both B256 stages did not run.
The lifecycle counter increments only after the capacity gate, so its zero
completed initializations does not contradict the recorded model/capture work.

Root cause is an interface error in the benchmark extension. Ascend's direct
base class has no request-capacity member; only Core's different
`DraftModelSpeculator` constructor creates that member. The old host fixture
invented `speculator=NS(max_num_reqs=256)` and therefore masked this mistake.
The corrected RPC measures `draft.block_tables.block_tables[g].gpu.shape[0]`,
`input_block_tables[g].shape[0]` and `slot_mappings.shape[1]` after `set_attn()`.
The runner/config bindings must be identical, groups valid and buffers nonempty.
The publication and confidence gates still reject undersized allocations.

Other RPC fields were checked against the fixed sources: Core GPUModelRunner
initializes `max_num_reqs/max_num_tokens`; `KVCacheConfig` defines
`num_blocks/kv_cache_tensors/kv_cache_groups`; `KVCacheTensor` defines
`size/shared_by/offset/block_stride`; `KVCacheGroupSpec` defines
`layer_names/kv_cache_spec/is_eagle_group`. The allocator already consumes the
spec's `block_size/page_size_bytes` contract. The graph-runtime RPC itself
completed for all eight real ranks in this archive. These descriptors are
capacity evidence, not proof of kernel correctness or actual concurrency.

Before loading weights, the existing host stage now first runs
`tools.dspark.capacity_preflight` in a fresh process. It imports and constructs
the installed Ascend class, records its MRO and source hashes, and calls the
**complete** capacity RPC using actual Core schema/container types with small
synthetic CPU bindings. It covers 128/256 rows and an undersized input table.
No class attribute is supplied to imitate `max_num_reqs`. Hardware/UVA allocation
is not exercised by that interface test, and its result is explicitly not NPU
fit evidence. Real allocation/capture/concurrency checks still run afterward.
Import/constructor/RPC failures stop before pytest or weights and are saved to
`capacity-interface.json`; pytest runs afterward in a separate process to avoid
mock fixture pollution. Both share the original 600-second host-stage deadline.

### Independent cleanup finding

At 04:21:19.435518 UTC the capacity RPC was recorded as the first error.
The output handler was cancelled/drained in 0.000147 s with zero unfinished
requests and no cancellation exception; its `success=false` preserves that
prior RPC error and must not be interpreted as another observed cancellation
failure. All eight workers exited with actual code 0 and were reaped by
04:21:32.159 (worker cleanup 12.720 s, no worker force events).
At 04:21:55.472545 the frontend process manager force-killed one remaining
managed process; frontend cleanup took 36.071 s. `timed_out=false` describes
the outer cleanup thread, while `forced_cleanup=true/success=false` correctly
records the inner process-manager failure. The supervisor sent no signals.
The generation, batch-expansion and outer driver PIPESTATUS are all `1 0`.

The frozen Core utility RPC catches the worker exception and returns a failed
utility response. The frontend records it and invokes shutdown. The executor's
complete shutdown method returned, including worker and queue cleanup. Core's
remaining path includes scheduler shutdown, distributed/memory cleanup and
interpreter finalization. This archive does not identify which later operation
kept EngineCore alive. No post-executor native stack proves a specific resource,
reference cycle, queue or GC cause. The one forced process is not evidence that
any worker failed to exit. Cleanup remains an independent **unresolved failure**;
this patch changes neither cleanup order nor budget and does not declare it
fixed. The next existing expansion task must still pass natural-exit checks.

This correction changes telemetry, preload interface checking, tests and these
documents only. Core/SWA, model computation, confidence policy, capture lists,
inputs, cost sampling plans, timeouts and frozen B64 results are unchanged.
There is no usable new table to resume: the next task starts B128 cost sampling,
then B128 confidence, B256 costs and B256 confidence, with failure stopping it.
Use the single server command above with the new delivery SHA. Return the new
outer `dspark-batch-expansion.*-evidence.tar.gz` and its SHA256; it includes the
installed interface report, JUnit, capacity shapes, costs, real execution
receipts, first error, separate cleanup receipts and all PIPESTATUS files.

### Validation of this correction

Local Python 3.12.13 / Torch 2.10.0 CPU: **169 passed, zero skipped** across
batch expansion (33), confidence acceptance, formal costs, profile failure,
shutdown policy and startup cost regressions. This includes executing the actual
Ascend constructor body in an isolated CPU fixture (the old member access fails),
real allocated tensor dimensions, undersized/malformed bindings, frozen Core
KV dataclass bodies, JSON-safe complete RPC results, and preload failure stopping
all later stages. Cleanup regressions retain the original error and reject
forced or incomplete cleanup. No new NPU execution was performed.

The installed-class preload check requires the server's vLLM/Ascend environment
and remains PENDING locally; source-body/CPU tests do not replace that check.
B128 sampling/confidence, B256 sampling/confidence, actual concurrency, memory
fit and frontend natural exit all remain PENDING. No B64 model is rerun.

Changed-file lint/manual hooks pass after formatting. Required repository-wide
`bash format.sh ci` was executed in a disposable checkout and returned 1 for
existing lint/format/spelling/workflow/shell/forbidden-import issues. Unrelated
auto-format edits were discarded. This does not claim full repository CI success.

## Subsequent SIGBUS transport correction

The next run passed the installed capacity gate and completed 35 cost points
before a snapshot-phase SIGBUS. See `B128_SIGBUS_TRANSPORT.md` and the frozen
`B128_SIGBUS_AUDIT.json`; no new complete cost table exists. The same server
command now includes a no-weight eight-worker communication precheck using
`/workspace/dspark-results/dspark-batch-expansion.qsQcxCRh-evidence.tar.gz`
(which must remain available), plus read-only system evidence. Expanded formal
profiles use verified rank files and small RPC receipts. Core, model execution,
point plans, deadlines, B64 status and natural-exit gates are unchanged.

## IF4hUVja: passive exit observation

The subsequent run completed all 48 B128 cost points but rank 5 exited on the
observer's SIGUSR1. See `B128_PASSIVE_EXIT.md` and `B128_IF4hUVja_AUDIT.json` for
independent sample/transfer verification and the stale-registration timeline.
No table was published. The new entry runs real passive-exit subprocess checks
before weights and explicitly disables both stack signals and gdb. It retains
all natural-exit/publication gates and the same B128-to-B256 order and budgets.
