# DSpark additional performance benchmark

This is a DSpark performance study, not V3.8 A1–A4 acceptance or a formal B0/B1 comparison.
The supplied task specifies the code-workload limits; no V3.8 document was present in this checkout.
Production, frozen core and custom operators are unchanged. Do not run `build_aclnn.sh`.

## Measurement and result contracts

The default comparison is `target_graph` versus `dspark_graph`. The former constructs
`speculative_config=None`; the latter uses K=5 with target `FULL_DECODE_ONLY` and eager draft/speculator.
Optional `target_eager` and `dspark_eager` use the same input and measurement path.

`benchmark_dspark_acceptance.py --measurement-protocol async_stream` uses the frozen core's
`AsyncLLM.generate(..., request_id)` with `RequestOutputKind.DELTA`. The benchmark facade owns one
event loop and reuses the existing resolved-config, KV-group, acceptance and named worker-RPC checks.
It does not copy or replace core execution/sample implementations.

- `num_prompts`: total requests in each measured run, independent of scheduler `max_num_seqs`.
- `client_outstanding`: optional number admitted to the frontend iterator concurrently. Omitted means
  all requests are offered in source order, without an artificial client concurrency cap. This is an
  in-process streaming protocol, not an HTTP arrival-rate benchmark.
- Scheduler observations record actual `num_running_reqs`/waiting requests and KV usage. They are not
  assumed equal to the configured cap. Per-rank FULL records retain padded and unpadded token counts.
- The frontend uses `time.monotonic()`. `client_ready_monotonic` precedes admission;
  `submitted_monotonic` is immediately before starting the async generator. Submission-to-first
  nonempty token event is TTFT, including engine/frontend queuing but excluding client admission wait.
  Submission-to-final finished event is completion latency. Raw clock values are process-local;
  do not subtract clocks between independent runs or hosts.
- A coalesced/speculative event carries one time and its actual newly delivered token count.
  Request mean TPOT is `(completion - first_output) / (output_tokens - 1)` when output length exceeds
  one. Otherwise it is null and excluded from distributions. A multi-token completion in one event
  can legitimately have zero observed mean TPOT; this is not a measurement of individual token times.
  Event intervals have their own distribution and are not relabeled TPOT.
- Throughput uses actual output tokens and the interval from before admission to completion of all
  streams. Model loading, rendering/tokenization, graph capture, warmup, phase-boundary RPC,
  string joins/result writing and code evaluation are outside it. Lightweight CPU event lists and
  scheduler counters are collected during execution; no per-step RPC, device synchronization,
  D2H copy or forensic snapshot is added.
- Scheduler spec metrics count per-request verifications, candidate tokens and accepted tokens.
  Frozen MRV2 sets `num_forwards=int(proposal_generated)`: this is reported as
  `proposal_publication_steps`, with `committed_tokens_on_proposal_publication_steps` separately.
  It is not an all-target-forward counter; `verification_batch_count` is unavailable.
  `1 + accepted / drafts` remains the existing effective-advancement definition; it is not substituted
  for actual output length or forward count. Preemptions come from `IterationStats`.
  Peak device bytes and recomputed tokens are null because this path has no reliable producer.
- FULL replay evidence comes from the existing worker observer, read before warmup, before measured
  generation and after measured generation. Every TP rank must agree on real successful executions;
  use one rank's count, never their sum. Measured replay must exceed zero. Capture/warmup alone cannot
  pass. Eager fallback coverage is unavailable; no claim of 100% graph coverage is made.
- Log errors/NaN, corrupt or incomplete output, abnormal termination, residual inference processes,
  changed input/configuration and invalid replay evidence invalidate performance. Internal tensors
  are not inspected in these performance runs. Reported core corruption counters and merged logs are
  retained; absence of a logged error is not a tensor-level NaN audit. Forensic diagnostics stay off.
- Every repeat is a new benchmark subprocess: load, warmup, one measured batch, shutdown, process
  termination. Mode order reverses on even repeats. All successful, failed and not-run entries remain.
  Three model loads per mode are three repeats; three generates after one load are not.
- Per-request mean/p50/p95/p99 use linear interpolation. Fewer than 100 observations carry a sparse
  p99 warning. Independent-run statistics retain raw values, mean, median, min/max, sample standard
  deviation (`ddof=1`) and sample CV. With n<2 CV is null. High CV never discards a completed run.
  The primary speedup is the ratio of median actual tok/s; all ratios paired by repeat are also saved.
  Incomplete pairs do not publish the aggregate speedup. This throughput is not SLO goodput.

Existing offline-batch JSON remains readable. New results retain schema 1 base fields and add
`performance_schema_version=1` and `measurement_protocol=async_llm_delta_stream_v1`.
The old summarizer refuses to mix offline and streaming protocols, or apply its offline comparison
to new streaming files. Missing historical per-request times remain unavailable, never inferred.
Natural EOS, output lengths, stop reasons and output token hashes are reported per run/pair, without
an exact-token performance gate or borrowed Qwen quality threshold.

## Freeze code inputs and executable tests

The existing `build_m2_5a_dataset_assets.py` LiveCodeBench importer preserves optional `test` metadata,
but does not define or validate a runnable test schema/reference. Prompt/token assets alone cannot
certify code correctness. This delivery therefore supplies an explicit import flow rather than
inventing tests. It has not created a real 64-case DSV4 asset on the Mac: that requires the server's
checkpoint tokenizer and actual source data.

Supported imports:

1. HumanEval-shaped JSONL: `task_id`, `prompt`, `canonical_solution`, `test`, `entry_point`.
   The reference is original prompt plus canonical completion; the provided `check(candidate)` harness
   is retained verbatim. See the [official dataset](https://huggingface.co/datasets/openai/openai_humaneval)
   and [HumanEval repository](https://github.com/openai/human-eval).
2. Normalized code JSONL: `task_id`, `prompt`, and optional `tests` object with
   `kind="python_function_v1"`, `entry_point`, `test` (defines `check(candidate)`), and full `reference`.
   Import actual source tests/references, including their revision. Other languages, stdin/stdout
   judges, compressed/private LCB tests and external dependency environments need an explicit importer;
   they are not silently treated as Python function tests.
3. General real-request JSONL or pinned HF dataset: configurable prompt/ID columns. Use `--id-field @index`
   if the fixed source has no ID column. These source-index IDs do not manufacture additional samples.

The builder sorts by SHA256(case ID) after checking uniqueness and full rendered length. It never
truncates tasks, cycles requests or samples with replacement. Duplicate IDs/prompts and insufficient
legal samples fail. Source provenance, raw task/hash, full DSV4 chat prompt/hash, token IDs/hash, actual
input count, tests/hash and tokenizer file hashes are saved. Omitted test data stays null.
The builder requires a 40/64-hex immutable source revision and preserves a full source snapshot.
Replay/duplication is deliberately unsupported. Repeats reuse the same frozen set; they are independent
model lifecycles, not new independent dataset samples.

On a data-preparation host with network access, obtain and record the immutable dataset SHA once.
The following uses the actual HF dataset revision, not a moving name during benchmark runs:

```bash
cd /workspace/vllm-ascend-hust
PERF_MODEL=/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8
PERF_MODEL_REV=9e8679a9db7eec11efed9925f7efb96549077545
PERF_CODE_REV=$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("openai/openai_humaneval").sha)')
PERF_CODE_DIR=/workspace/dspark-data/code64-$(date +%Y%m%d-%H%M%S)
python tools/dspark/prepare_performance_data.py \
  --hf-repo openai/openai_humaneval --source-name openai/openai_humaneval \
  --source-revision "$PERF_CODE_REV" --kind humaneval --num-samples 64 \
  --max-input-tokens 2048 --tokenizer "$PERF_MODEL" --tokenizer-revision "$PERF_MODEL_REV" \
  --output-dir "$PERF_CODE_DIR"
```

Use the frozen core/plugin import environment for its DSV4 tokenizer. If HF access is unavailable,
export the same immutable source to JSONL elsewhere, transfer it, and replace `--hf-repo` with
`--input-jsonl /path/source.jsonl`. Keep the same source revision/name. Never execute its contents on
the host. Do not change the server CANN/custom OPP setup to prepare data.

Code outputs are limited to 1024 tokens with natural EOS; no task is shortened to fit the 2048-token
input cap. The fixed `single_python_fence_v1` extractor accepts exactly one Python/blank-language
fence and saves both generated text and extracted solution. It requests a full solution, not a
HumanEval completion suffix. This is a frozen code-service workload, not the original completion-only
HumanEval pass@k protocol.

## Isolated code evaluation

Provision a locally available Python container image and pin its digest before the benchmark.
For example, on a host allowed to provision Docker images:

```bash
docker pull python:3.12-slim
PERF_JUDGE_IMAGE=$(docker image inspect python:3.12-slim --format '{{index .RepoDigests 0}}')
printf '%s\n' "$PERF_JUDGE_IMAGE"
```

Pass that `name@sha256:...` to `--sandbox-image`. Execution uses `--pull=never`, no network, read-only
root and task mount, unprivileged UID, no added capabilities, no-new-privileges, PID/memory/CPU/file
limits and a bounded tmpfs. No host credentials, Docker socket, NPU device or other writable host
directory is mounted. The benchmark never executes generated code with the host Python interpreter.
Docker itself must already be provisioned in a suitably isolated evaluation environment.

`python_function_v1` compiles syntax, then runs the real test harness against the extracted entry point.
The child has a 10-second test deadline, container launch/judge a 30-second deadline. An outer timeout
is infrastructure failure, with cleanup restricted to this judge's UUID-named container. Child logs
are bounded. The fixed judge first runs every supplied canonical reference outside inference timing;
unavailable/failing references make certified quality unavailable. Syntax, unit-test failure, timeout,
extraction failure, generation failure and infrastructure failure remain separate counts.

Syntax rate uses actual syntax-check attempts as denominator; extraction failures are reported separately.
Unit-test task denominator includes evaluable tasks failing extraction/syntax/tests, excludes missing
tests/infrastructure/generation failures, and is explicit. `all_task_pass_fraction` is only available
when the entire task set has usable tests/runtime/reference validation. No Docker/tests means quality
unavailable, never passed; throughput implementation/execution remains independently usable.
Quality rates/differences are reports rather than additional hard performance thresholds.

## Server A/B: source, focused, then small smoke

Replace the SHA with the delivered commit. Preserve all existing CANN/custom OPP paths. These commands
use fast-forward synchronization only. The script verifies exact plugin/core HEAD and clean trees.

```bash
cd /workspace/vllm-ascend-hust
git fetch origin feat/dspark && git merge --ff-only origin/feat/dspark
PERF_SHA=$(git rev-parse HEAD)
PERF_MANIFEST="$PERF_CODE_DIR/manifest.json"
bash tools/dspark/run_dspark_performance.sh "$PERF_SHA" "$PERF_MANIFEST" smoke \
  --sandbox-image "$PERF_JUDGE_IMAGE"
```

Smoke defaults: four frozen code requests, scheduler cap two, one independent process per graph mode,
one warmup request, 128 output tokens. Target capture `[1,2]`, DSpark `[6,12]`. Check both receipts,
saved texts/code, stream times and `quality/quality.json`; require usable reference/judge infrastructure
before calling the code-quality chain exercised. A low model test pass rate is reported, not reclassified
as infrastructure failure. This script does not automatically continue to the formal workload.

The script runs focused tests in the actual installed environment (no `--noconftest` there), checks
idle inference processes and `npu-smi` before/after each run, and stops on busy/unknown resource status.
It never kills another task, resets devices, reduces configuration or runs a custom-op build.
Every tee immediately saves both PIPESTATUS entries. Failures retain logs/results; the outer script
still archives evidence. No `set -e` or shell `exit` is used.

## Server C: 64 code cases, independent repeats

Only after reviewing the smoke result, explicitly run:

```bash
bash tools/dspark/run_dspark_performance.sh "$PERF_SHA" "$PERF_MANIFEST" code64 \
  --sandbox-image "$PERF_JUDGE_IMAGE"
```

Defaults are 64 identical ordered code inputs, cap four, output 1024, warmup one, three independent
lifecycles per mode, target capture `[1,2,4]` and DSpark `[6,12,24]`. TP8+EP, MRV2, bf16/ascend,
seed 0, temperature 0, top-p 1, top-k -1, natural EOS, no prefix caching, model length/token budget 8192,
memory fraction 0.9. All are recorded in command/result JSON. Use explicit options for a chosen study:

```bash
bash tools/dspark/run_dspark_performance.sh "$PERF_SHA" "$PERF_MANIFEST" code64 \
  --max-num-seqs 8 --capture-target 1 2 4 8 --capture-dspark 6 12 24 48 \
  --client-outstanding 16 --gpu-memory-utilization 0.9 \
  --modes target_graph dspark_graph target_eager dspark_eager \
  --sandbox-image "$PERF_JUDGE_IMAGE"
```

All selected modes receive the same client cap; omission means all-at-once in all modes. Eager modes
receive no capture sizes. Explicit graph sizes must be ordered, unique multiples of q=1/q=6 ending at
`q*max_num_seqs`, within token budget. For multiple scheduler caps, omit explicit overrides to derive
the documented sparse `[1,min(2,cap),min(4,cap),cap]*q` lists independently, or run separate invocations
with explicit sizes. Requested, resolved and rank-observed capture sizes are all retained.

## Server D: optional larger real inputs and concurrency exploration

Build a separate larger frozen dataset before exploration. This example deterministically chooses
2048 unique real GSM8K train questions from the already-used pinned repository revision:

```bash
PERF_LARGE_DIR=/workspace/dspark-data/real2048-$(date +%Y%m%d-%H%M%S)
python tools/dspark/prepare_performance_data.py \
  --hf-repo openai/gsm8k --hf-config main --split train --source-name openai/gsm8k \
  --source-revision cc7b047b6e5bb11b4f1af84efc572db110a51b3c --kind general \
  --prompt-field question --id-field @index --num-samples 2048 --max-input-tokens 2048 \
  --tokenizer "$PERF_MODEL" --tokenizer-revision "$PERF_MODEL_REV" --output-dir "$PERF_LARGE_DIR"
```

An offline JSONL export with real IDs is equally supported. Insufficient/duplicate data fails rather
than repeating questions. General workload code quality is not applicable. Do not confuse this
larger request set with the 64-case code study.

First explicitly execute only cap 400, using all 2048 requests:

```bash
bash tools/dspark/run_dspark_performance.sh "$PERF_SHA" "$PERF_LARGE_DIR/manifest.json" explore \
  --num-prompts 2048 --max-num-seqs 400 --repeats 3 --output-len 256
```

After reviewing its results/resource evidence, a separate invocation may use `--max-num-seqs 512`,
then another `--max-num-seqs 768`. The script never starts those automatically. A CPU-only plan can be
created by invoking `run_performance_suite.py` with the same options and without `--execute`.
Neither reaching the concurrency cap on every step nor 100% graph coverage is required or asserted.

## Server E: reading, resummarizing and evidence

Every script invocation prints a fresh `/workspace/dspark-results/dspark-performance.XXXXXX` directory.

- `source.log`, `focused.log`, `*.pipestatus`, `status.txt`: environment, tests and exit evidence.
- `runs/plan.json`, `input/manifest.json`, `input/requests.jsonl`, `requests.jsonl`: frozen input,
  sample provenance, independent process order and exact per-mode commands.
- `runs/<case>/command.json`, `generation.log`, `generation.pipestatus`, `npu-before/after.log`:
  launch, merged runtime errors and resource state. Failed or not-run cases are not erased.
- `runs/<case>/result.json`: legacy-compatible base, requested/effective config, stream data,
  acceptance metrics, phase snapshots and every rank's actual replay shapes.
- `runs/<case>/requests.json`: request/case ID, prompt/output lengths, text/token IDs, termination,
  three observation boundaries and raw stream events. It is written after inference.
- `partial-stream.json`: available stream data retained if generation raises, including completed,
  failed/cancelled and not-submitted requests. An engine-start failure has no manufactured outputs.
- `reference-quality/quality.json` and `<case>/quality/`: reference receipts, original generations,
  extracted solutions and per-task syntax/tests/timeout/infrastructure statuses with denominators.
- `<case>/receipt.json`: performance validity and separate quality status, artifact/log hashes.
  `valid` does not mean code quality passed or hidden tensors were proven finite.
- `summary.csv`: each run's throughput, elapsed, request quantiles, acceptance/replay and quality status.
  `summary.json` additionally retains all per-run evidence, raw independent-run values/CV and pairs.
  `summary.md` gives a compact throughput table; quality detail remains explicit in JSON.

Resummarize without loading a model:

```bash
PERF_RESULTS=/workspace/dspark-results/dspark-performance.REPLACE/runs
python tools/dspark/summarize_dspark_acceptance_benchmark.py \
  --performance-suite "$PERF_RESULTS" --output-json "$PERF_RESULTS/summary.json" \
  --output-csv "$PERF_RESULTS/summary.csv" --output-markdown "$PERF_RESULTS/summary.md"
```

The server wrapper packages the fresh directory as `*-evidence.tar.gz`, with SHA256 and hash-pipeline
status beside it, even after failure. It does not package an old successful run as the new result.
Large successful historical runs were not recalculated locally; no historical archive was required
to implement these interfaces. Local CPU/mock validation is not installed-core, Docker, model or
Ascend validation: **SERVER_NOT_REVALIDATED** until the server stages are actually run.

## Local delivery checks

On 2026-09-08, Python 3.12.13 with CPU Torch 2.10.0 executed the new performance suite and related
benchmark, RPC, replay, draft-config and padded-request tests: **157 passed, 4 skipped**.
The four skipped tests require installed vLLM/Ascend classes. Local invocation used `--noconftest`
because the repository-wide UT conftest imports the unavailable installed vLLM package; the server
script retains the normal installed-environment invocation. The new test file contributes 36 tests,
including real local subprocess lifecycle launches and frozen-core AST interface checks.

Changed-file manual pre-commit checks, shell syntax and CLI help checks were run. The full
`bash format.sh ci` was also exercised in an isolated worktree; it found existing repository-wide
lint/spelling/format issues and unrelated auto-format changes. Those changes were discarded, and
task-file issues were corrected and checked separately. This is not a claim that full-repository
lint passed. No actual model inference, NPU graph capture/replay, Docker judge execution, real DSV4
dataset rendering or historical server-result recomputation was performed locally.
