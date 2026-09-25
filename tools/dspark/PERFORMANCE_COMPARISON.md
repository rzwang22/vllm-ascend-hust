# B64 / B128 / B256 DSpark performance comparison

## Frozen acceptance

`PERFORMANCE_BASELINE_AUDIT.json` freezes the independent CPU audit of
`dspark-batch-expansion.XUJieSKP` and `dspark-large-batch.6tf75BWi`.
Both uploaded SHA256 values and the embedded model archive match. The audit
rebuilds all retained samples and published tables from raw point files, checks
frozen inputs, eight-rank confidence FULL receipts, four passive natural exits
and every archived PIPESTATUS. Reproduce without weights:

```bash
python -m tools.dspark.audit_performance_baseline OUTER.tar.gz MODEL.tar.gz audit.json
```

| Tier | Cost points / requests / retained samples | Confidence instances / distinct questions | Confidence FULL calls |
| --- | --- | --- | --- |
| B128 | 48 / 2258 / 3840 | 128 / 64 | 78 |
| B256 | 56 / 4562 / 4480 | 256 / 64 | 74 |

Actual full-tier FULL coverage and all four sets of eight zero worker exit codes
passed. B64's published cost and confidence baseline remain frozen. These are
functional/calibration results, not performance results. Historical SIGBUS cause
remains UNKNOWN; original five-second shutdown policy remains NOT_EVALUATED.
The new archive's Core receipt uses `rzwang` at
`https://github.com/rzwang22/vllm-hust.git`, exact HEAD
`71d2c1c436eba894a8e9eeb2c5af17e05cb42970`; earlier file-URL receipts retain their
original, narrower provenance.

## One bounded task

Order: B64 fixed, B64 confidence, B128 fixed, B128 confidence, B256 fixed,
B256 confidence. Each case uses one fresh engine for one full-population warmup
and three measured rounds. Six model initializations; 3584 request instances
including warmup, at most 917504 output tokens. This is 64 distinct frozen GSM8K
questions, repeated two/four times for B128/B256, not new questions.

| Maximum requests and client in-flight cap | Target capture capacities | Instances per round |
| --- | --- | --- |
| 64 | 6, 12, 24, 48, 96, 192, 384 | 64 |
| 128 | 6, 12, 24, 48, 96, 192, 384, 768 | 128 |
| 256 | 6, 12, 24, 48, 96, 192, 384, 768, 1536 | 256 |

Natural EOS, maximum 256 new tokens, temperature 0, top-p 1, top-k -1, seed 0;
original input order/token bytes, TP8+EP, BF16/Ascend quantization, block size 32,
max model/batched tokens 8192, memory utilization .9, prefix caching disabled.
Target is FULL_DECODE_ONLY, draft eager. Ordinary prefill is not required to be
FULL. Actual scheduled request counts, query tokens and padding/capacity are
reported independently; falling concurrency is not artificially sustained.

Before any weights load the entry prints the complete plan. Host preflight has
600 seconds, full-weight/input/table preparation 1800 seconds. Each model child
has 3600 seconds including initialization, capture, warmup and all measured
rounds. Named shutdown remains worker 25 / TERM 4 / shared reap 1 / frontend
inner 36 / outer 40 / supervisor 48 seconds. No added observation wait, gdb or
stack signals. The group deadline is 25000 seconds plus a 65-second emergency
termination margin. Budget expiry is a failure. No retries or later cases after
failure. Archive/export time is outside model/performance timing.

## Immutable cost compatibility

`performance_comparison.COSTS` pins the original B64/B128/B256 bytes, producer
commits and server paths. No table is rewritten or recompiled. Publication
proof, workload, complete weight hashes and actual worker runtime identity are
checked, including hardware, Torch/torch_npu, quantization, TP/EP, K, capture and
KV configuration. Missing/out-of-range costs retain the existing hard error.

`PERFORMANCE_CODE_COMPATIBILITY.json` pins the exact set and hashes of changed
runtime files against each cost producer. No model, attention, speculator,
allocator, cost lookup or custom-op source has changed. The admitted differences
are benchmark RPC/export/observation and passive teardown gates; old detailed
confidence/file-transport observers are not installed in performance mode.
Future differences or mismatched file hashes fail before loading weights.
This is an explicit consumer compatibility record, not a claim of same commit.

Current OPP/extension hashes are saved before initialization and from every
worker after capture, checked for cross-rank/cross-case consistency. The old
cost format does not contain a complete CANN/OPP binary build manifest: binary
correspondence to historical source remains UNKNOWN. Preserve the accepted
CANN/custom OPP environment; these hashes do not retroactively prove an old
binary's build provenance. The runtime identity contract remains exact.

## Measurement and instrumentation

The fixed branch does not instantiate `ConfidenceVerification`; it does not
compute the confidence head, transfer its scores, consult costs, broadcast
adaptive choices or trim verification lengths. The same loaded draft model
contains confidence-head weights in both cases; their mere presence is not
head execution (`speculator.py` gates `record()` on the adaptive instance).
No model computation is changed for this comparison.

Each generation timer starts before client tasks are scheduled and stops after
all request iterators complete. It includes actual confidence head execution,
D2H, host allocation, TP broadcasts, graph input preparation, prefill, draft,
target and output delivery. Initialization/capture, phase-boundary RPCs,
summary D2H, writing artifacts and shutdown are timed/reported separately.
No extra device events, synchronization or per-token disk writes are added.

The existing FULL wrapper records bounded CPU-only histograms of actual
completed InputBatch request count, query count, capacity and query lengths.
It is used identically in both modes, capped at 65536 FULL calls per phase;
missing, replaced, overflowed or inconsistent eight-rank evidence invalidates
the case. There are no per-step request/score JSON records or tensor probes.
This lightweight proof does not repeat the full functional acceptance trace.
Existing numerical and owner checks remain active. Confidence cumulative
counters and head identity are checked at phase boundaries; phase deltas
exclude warmup. Accepted-token statistics use the existing frontend logger.

Each round checks no unfinished frontend requests and no engine error before
and after generation. Request IDs are `warmup:` / `round-N:` plus the frozen
instance ID, identical across modes but unique within an engine. Only telemetry
histograms reset. Proposal epochs, RNG, scheduler and model state are not
rewound; existing request retirement/owner checks remain authoritative.

Saved raw monotonic submit/first-nonempty-DELTA/completion timestamps and chunk
token counts allow reconstruction. TTFT = first minus submit. Per-request TPOT
= (completion minus first) / (output tokens minus one), undefined for <=1 token.
It is a request mean, not a measured per-token arrival sequence. Multi-token
chunks never create fictitious token timestamps. P95 uses nearest rank,
`ceil(.95*N)` after sorting valid request means; median uses ordinary median.
Throughput = actual output tokens / whole-round generation duration.

Three rounds are all retained, with median, range and sample standard deviation.
Speedup = median confidence throughput / median fixed throughput. Output tokens,
lengths and content differences are explicit; differing output sequences flag
comparability. Quality equivalence is NOT_EVALUATED. Fixed-first order is
reported as a possible time/order bias, not hidden by selecting fastest rounds.
Existing confidence transfer/policy host clocks are saved by rank without
additional instrumentation; they include overlapping execution/communication
and cannot isolate kernel savings. Detailed operator timing is NOT_MEASURED.
E2E speedup answers net benefit, including scheduling overhead.

## Server execution

Use the exact delivered commit in place of `PLUGIN_SHA` below, in a child Bash
so failure does not close the interactive parent shell. Existing CANN/custom
OPP settings are inherited. No OPP build/install or permanent remote rewrite.

```bash
if bash -c '
  set -euo pipefail
  cd /workspace/vllm-ascend-hust
  git fetch origin feat/dspark
  git merge --ff-only PLUGIN_SHA
  bash tools/dspark/run_dspark_performance_comparison.sh \
    PLUGIN_SHA \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
    rzwang
'; then
  echo "Six performance cases completed; inspect performance-summary.json"
else
  rc=$?
  echo "Stopped on first failure: $rc; return the saved evidence archive"
fi
```

Preflight requires exactly eight visible NPUs and checks `npu-smi`/process
occupancy before and after every engine. A busy device is an error, not grounds
to kill another job. Post-run gates require strict logs, passive exit evidence,
eight actual zero exits/reaps, successful output-consumer/frontend/EngineCore
cleanup, no timeout/escalation/residual and valid measurements.

Return only the new outer `dspark-performance-comparison.*-evidence.tar.gz` and
its SHA256. It embeds the model archive with inputs/source/weight/cost/binary
identities, all warmup/round streams and outputs, counter receipts, capture and
capacity evidence, original exit codes, logs/PIPESTATUS and summary. An invalid
later case preserves earlier valid cases and never alters frozen acceptances.
Installed/NPU performance runs remain PENDING until this task is executed.

## Local validation

Python 3.12.13 / Torch 2.10.0 CPU: 149 related tests passed, covering the new
comparison, existing replay/streaming/acceptance contracts, named shutdown and
owned real child-process exit regressions. No installed Ascend or NPU performance
execution was performed. Changed-file manual pre-commit checks (including
shellcheck) passed. Full `bash format.sh ci` ran in a disposable checkout and
still fails on existing repository-wide format/spelling/workflow/import issues;
it auto-modified 78 unrelated files and none of this task's files. Those unrelated
changes were discarded with the disposable checkout, not applied to this tree.
