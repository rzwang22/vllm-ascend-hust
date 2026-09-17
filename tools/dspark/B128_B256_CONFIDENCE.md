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
- Frozen Core `v1/worker/gpu/spec_decode/speculator.py` derives request and token
  buffers from scheduler configuration; its DSpark subclass allocates persistent
  anchors and logits using those dimensions.
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
