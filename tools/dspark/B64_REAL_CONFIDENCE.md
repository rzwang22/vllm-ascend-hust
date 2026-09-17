# B64 frozen GSM8K confidence acceptance

The formal table produced by `89cfb54d53b1443cb7df0b4c800196435b21e5ad`
with Core `71d2c1c436eba894a8e9eeb2c5af17e05cb42970` is frozen. Its SHA256 is
`576188ba88fd839a00308179b91a951b3d30467b951b159954e57549f869029a`.
The original ten points and both synthetic functional phases remain passed.
This task does not reopen those model tests, the original five-second budget,
or the unlocalized native destructor tail.

## Independent cost audit

`FORMAL_COST_Ta3Z6V3Z_ACCEPTED.json` records the read-only audit. Both actual
archive hashes match the uploaded identities. The auditor reconstructs all
40 points / 1106 requests (128 input / 512 output), validates internal/external
request mappings, hashes every raw point file, and reconstructs the 3200
retained NPU event samples using the existing selector. Recompilation matches
both the pending and published cost tables. The actual CostTable accepts all
10464 candidate layouts for 1–64 requests, context ceiling 640. Twenty raw
host allocator samples reproduce the published median. These host samples
are separate from target FULL and eager draft NPU events.

The publication, natural exit, supervisor, log-scan, outer/inner PIPESTATUS
and 27-test JUnit receipts pass. Eight workers exited 0 and were reaped,
without escalation or residual processes. Full weight-shard hashes agree
between archived preflight and publication; loaded confidence-head fingerprints
agree across every rank and point. **Model weight bytes are absent from the
local archive:** this is receipt verification, not a local rehash of weights.
The new server preflight hashes every loader-visible shard and model metadata
again and requires equality with the immutable published provenance.

This run's `core-source.json` records remote `rzwang`, URL
`https://github.com/rzwang22/vllm-hust.git`, exact fetched/current Core SHA.
Earlier runs whose URL was `file:///workspace/vllm-hust/` retain their narrower
local-source verification scope. No remote URL is rewritten.

Reproduce the CPU audit without vLLM/Ascend installed:

```bash
python -m tools.dspark.audit_formal_cost \
  /path/to/dspark-formal-cost.Ta3Z6V3Z-evidence.tar.gz /tmp/cost-audit.json
```

## Single workload and acceptance

`B64_REAL_TEXT_PLAN.json` remains the immutable input contract: first 64 distinct
GSM8K questions, original request IDs, frozen token arrays and source hashes.
The full original manifest/assets are copied as evidence; exactly its first
64 records are submitted once. Prompts are not rendered or tokenized again.
There is no separate request warmup or model reload. Sampling is greedy,
temperature 0, top-p 1, top-k -1, seed 0, natural EOS, maximum 256 new tokens.
Short answers are preserved; length-limited completions are reported separately.
Input lengths are 29–116. Total output is at most 16384 tokens, not a target.
This is scheduling correctness acceptance, not GSM8K accuracy certification.

The engine remains B64 / TP8+EP / K5, target FULL_DECODE_ONLY, draft eager,
BF16/Ascend quantization, 8192 model/batched-token limit, memory utilization .9,
capture capacities `[6,12,24,48,96,192,384]`. Initial outstanding requests are
64; actual scheduled concurrency and selected lengths are reported as observed.
Ordinary prefill and mixed admission do not query the decode cost table and
are not required to be FULL. Real pure-decode confidence decisions must consume
current proposal owners through actual FULL target executions.

Costs retain the producer's ceiling-request buckets `[1,6,12,24,48,64]`, context
ceiling 640 and measured capacity cells. Runtime rejects missing/out-of-domain
cells; there is no extrapolation. Max-rank/layout medians and monotone estimates
are not a mathematical worst-case latency bound. `mode=confidence`, `profile=false`
and a checkpoint-loaded DSparkConfidenceHead are mandatory. No calibration is
supplied, so confidence remains **uncalibrated**. All-K5 is a valid observed
outcome; the entry never modifies scores to manufacture length variation.

`confidence_acceptance.py` joins every actual target row to its selected
request/epoch, query length, capacity, cost cell and token budget. It requires
all 64 requests to have target execution evidence, all eight ranks to agree
on decisions, execution layouts and accepted counts, and receipt totals to
match the existing runtime counters and FULL observer. Accepted draft counts
are `num_sampled - 1`; rejected counts are `verified + 1 - num_sampled`, following
the existing device assertion that these partition the verification query.
The reported execution ordinal counts real nonempty calls after capture;
proposal epochs are the actual runtime owner epochs, not invented host epochs.

Built-in Markov NaN, confidence finite checks, owner/token contracts, actual
Graph capture/FULL checks and strict merged-log scanning stay enabled. This
does not claim every model tensor was inspected. Numerical probes, operator
capture, write timelines and debugger observation remain disabled.

## Code compatibility and observation cost

`CONFIDENCE_CODE_COMPATIBILITY.json` pins the exact changed runtime-adjacent
files against the producer. Preflight rejects extra runtime changes or a
changed audited file. Core, custom operators, model/attention, SWA reclamation,
confidence computation, allocator, trimming and CostTable lookup remain byte
identical to the producer. The original cost file is copied byte for byte;
its producer SHA and table SHA are never rewritten.

Changes are opt-in benchmark wrappers plus exit-worker admission for this exact
confidence configuration. The selector is delegated once, returning its actual
trimmed scheduler output. Target evidence delegates the existing real FULL
observer; it does not install graph tensor nodes or manufacture device receipts.
The accepted-count vector is cloned on the consuming stream before reuse,
then transferred in **one combined D2H at the final quiescent RPC** in addition
to the pre-existing aggregate snapshot. There is no per-decision host wait,
new global synchronize, or change to existing stream dependencies. Up to 32768
request rows of counts are retained (128 KiB at int32, plus final concatenation
and allocator overhead); the artifact records the actual byte count.

Host JSON history records selection, target return and verification return,
with limits of 2048 nonempty target calls, 32768 request rows and 64 MiB per
rank's append log. Exceeding a bound fails; records are never silently dropped.
The final RPC adds detached CPU accepted counts to `after.json`; append logs
retain the earlier stages even if a later execution/RPC fails. Pending counts
on failure remain explicitly unavailable. Wrappers and count snapshots are
released by the existing orderly observer close before worker resource teardown.

CPU logging/clones may affect end-to-end scheduling and timing. Existing costs
remain the published computation estimates; this instrumented run is always
`performance_eligible=false`. It cannot establish throughput gains or head/D2H/
host/TP overhead equivalence. Performance comparison is a later separate task.

## Bounds and server entry

One model run has a 3600-second limit including initialization/capture/generation/
exit. Preflight full-weight hashing has its own 1800-second timeout (+15-second
kill guard); its full plan prints before loading. The named shutdown budget is
worker 25, TERM 4, shared reap 1, EngineCore/frontend inner 36, frontend outer 40,
supervisor grace 48 (+existing final group TERM/KILL guard). No gdb or extra
observation wait. Defaults and historical five-second results are unchanged.
Failure stops; there is no retry, next synthetic phase or performance task.

Run the following once after substituting the delivered full plugin SHA. Keep
CANN/custom OPP environment unchanged. The script records Core remote `rzwang`
and actual URL/HEAD without permanently changing origin. Strict shell options
stay inside child Bash; failure cannot close the interactive parent shell.

```bash
if bash -c '
  set -euo pipefail
  cd /workspace/vllm-ascend-hust
  test -z "$(git status --porcelain)"
  git fetch origin feat/dspark
  git merge --ff-only "$1"
  test "$(git rev-parse HEAD)" = "$1"
  exec bash tools/dspark/run_dspark_real_confidence.sh "$1" \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json rzwang
' _ PLUGIN_SHA; then
  echo "Confidence acceptance command completed; inspect overall_pass"
else
  rc=$?
  echo "Confidence acceptance failed: rc=$rc; retain the new evidence archive"
fi
```

Inputs required on the server: unchanged original manifest/assets; published
`dspark-large-batch.8CR50Czp/runs/b64/{cost-profile,cost-publication}.json`;
original complete model files; installed producer-compatible Torch/NPU/OPP
runtime. No full synthetic or cost collection is repeated. The entry runs only
its host regression file first and rejects failed/skipped preflight tests.

Success requires all inputs/natural completions valid, real confidence FULL
consumption on eight ranks, strict logs clean, output consumer closed, all eight
worker codes actually 0 and reaped, EngineCore/frontend successful, no timeout,
force event or residual. Generation and cleanup outcomes remain separate, with
overall failure if either fails. Original-budget acceptance remains NOT_EVALUATED.

The outer script prints a new `dspark-confidence-acceptance.*` directory and
exports one `*-evidence.tar.gz` with SHA256. Return that archive and its SHA256.
It embeds the new model archive containing frozen inputs/costs/proof, full-weight
preflight, source identity, before/after snapshots, per-rank decision/execution
logs, stream outputs, capture/exit receipts, strict logs, JUnit, PIPESTATUS and
`confidence-acceptance.json`. `touch MODEL_RESULT_DIR/STOP` requests controlled
stop; do not kill workers manually. Export failure cannot replace the original
failed-phase return code. No old archive is overwritten.

Local CPU checks are recorded in `REAL_CONFIDENCE_VALIDATION.json`. Real NPU
confidence closed-loop acceptance is **PENDING** until this task returns. A pass
closes this frozen real-text scheduling workload only; fixed-K5 comparison and
all performance claims remain NOT_RUN.

The producer's runtime identity covers hardware name, Torch/torch_npu versions,
compile switches and capture/TP/EP/model configuration. Its source log records
the custom OPP path but contains no complete binary build fingerprint. Binary
source correspondence therefore remains UNKNOWN; this handoff preserves the
existing OPP environment and does not rebuild/install it. Code compatibility
here is the audited Python/Core/custom-op source scope plus the exact recorded
runtime identity, not a claim of independently proven OPP build provenance.
