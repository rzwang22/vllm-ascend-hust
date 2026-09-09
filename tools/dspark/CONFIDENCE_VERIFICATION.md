# DSpark confidence-scheduled verification on Ascend MRV2

Status: implementation submitted for server validation, **SERVER_NOT_REVALIDATED**.
CPU/reference execution does not certify ACLGraph, NPU numerics, quality, or speedup.
This is an opt-in DSpark feature, not a V3.8 acceptance milestone. Core stays at
`897306c43bf800e2480cb5c0f3e2da408d85a2fd`; custom ops and OPP are unchanged.

## Audited references and strategy

Read the [DSpark paper v1](https://arxiv.org/html/2607.05147v1), the
[official adaptive verification article](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification),
and all changed source files of [PR #47808](https://github.com/vllm-project/vllm/pull/47808).
The reference implementation is PR head `e2e335334669d1c94c7351937474c0104dcbfdfb`
(merge `7f7a32cfec0f1bc5b73c37200b86631523a1ea8f`). It was inspected, not merged.
The article's reported performance commit is a different revision (`73b8394`);
those NVIDIA results are not Ascend measurements.

Drafting still generates five candidates. For candidate position `j`, use the
actual normalized draft hidden state and the Markov embedding of its predecessor
in the greedy recurrence. The checkpoint confidence projection emits a logit;
`sigmoid(scale * logit + bias)` gives conditional acceptance. A cumulative product
gives prefix survival. NaN/Inf logits fail before publication.

The host policy enumerates budgets over the heap of next-prefix marginal survival
scores, maximizing `(sampling_requests + sum(survival)) / (target + draft + overhead)`.
It selects only contiguous prefixes. Ties use request ID then position; equal
utility keeps the smaller budget. This handles exact probability ties without
relying on unspecified device `topk` tie ordering.

Upstream uses older host scores to choose a budget and current device scores to
allocate it. This implementation uses **the current candidate producer epoch for
both decisions**, because Ascend metadata also consumes host query offsets. One
batch of `B*5` float logits is copied to host after draft execution, before those
candidates are published. This is an intentional synchronization cost; it is not
hidden behind the name “async”. Budget computation and the CPU TP broadcast remain
inside the end-to-end inference interval. No per-request D2H is added.

Pure decode uses the cost policy. Mixed prefill/admission retains existing widths
and reports `fixed_admission_batches`: incomplete prefill does not earn a sampling
benefit and a decode profile is not extrapolated to prefill. The captured target
path is limited to pure decode, greedy sampling, DP1/PP1, no LoRA, eager drafter,
FlashComm1 off, and the audited A2/910B arch32 DSV4 DSA builder for every KV group. The
backend's global graph capability is not promoted to `ALWAYS`.

## Data flow and actual graph execution

1. `deepseek_v4_dspark.py::load_weights` verifies all required parameters were
   loaded, then fingerprints the actual loaded confidence projection bytes.
   `ConfidenceVerification.bind_model` requires that receipt and the real module.
   A config entry or allocated module without loaded parameters is insufficient.
2. `speculator.py::_execute_sequential_markov_sampling` collects the exact five
   predecessor embeddings. Confidence rows carry request ID and producer epoch.
   Delayed owners retain their scores; consumed suffixes and terminal/preempted
   owners lose them. An `ell=0` installed owner is consumed, then a new proposal
   is produced. It is not treated as a delayed proposal.
3. `NPUModelRunner.execute_model` reads current scheduler host counts, selects
   prefixes, and copies only worker-local scheduling dictionaries. It does not
   mutate the scheduler's original output or shared target configuration. The
   frozen scheduler subtracts `original K - accepted`; worker post-update adds
   `1 + ell - actual_rejections`. Both advance by `1 + accepted`.
4. Existing `prepare_inputs` computes compact input IDs, positions, query offsets,
   expanded logits rows, and slot mappings from the selected lengths. The real
   target receives those prefixes. This is not output-only suffix truncation.
5. `ModelAclGraphManager` captures descriptors with request capacity
   `min(token_capacity, max_num_seqs)`, not `token_capacity // 6`. Pure decode
   dispatch uses actual request and token counts. A missing graph is an error.
   Different combinations share the same descriptor. The current core's
   q-aligned capture-size normalization is retained; capacities are explicit
   multiples of six, but **actual query lengths are not fixed at six**.
6. Dummy capture requests distribute the remainder evenly, keeping query length
   at most six. Real replay writes cumulative offsets for actual rows and repeats
   the terminal offset for dummy rows. The input tensor remains graph-sized.
   Unused slots retain the exact `-1` sentinel (formatted `[-1,31]` for block 32),
   and unused sequence/start-position/block-table rows are cleared. No request is
   duplicated to fill capacity and persistent tensors are not replaced.

For `[5,2,0,4]`, actual query lengths are `[6,3,1,5]`, 15 tokens and 4 requests.
The runner may reorder them, preserving request/state/logits mapping. Capacity 24
still performs graph-sized matrix work; saving nine effective tokens is not a
claim of proportional latency reduction.

The relevant existing consumers were inspected:

- `AscendModelState.prepare_attn` distinguishes actual tokens from input capacity,
  passes padded tensor views, and performs current-input shared-KV preflight.
- `AscendDSAMetadataBuilder.build_decode_metadata` computes each start position as
  `seq_len - diff(query_start_loc)`. Batch-local dictionaries are shared across KV
  groups; each group's actual block size and persistent SAS/QLI buffers remain.
- `sparse_attn_sharedkv_metadata` AICPU `GetS1SeqSize` differences TND offsets and
  skips zero-query/zero-KV rows. The actual sparse attention consumes this updated
  metadata. DSA `update_graph_params` is a no-op; there is no hidden host retiling
  that substitutes a different query layout at replay.
- `vllm_quant_lightning_indexer_metadata` consumes cumulative query endpoints and
  per-request KV lengths; its README and AICPU implementation permit zero rows.
- arch32 compressor metadata `BuildCompressedPrefix` reads current device offsets
  and start positions, calculates per-request compression crossings, and fills
  unused output rows with padding. The existing scatter selector skips the exact
  padding address. None of these custom operators was changed or rebuilt.

These are source and CPU reference findings. B1/B4 NPU replay remains the required
runtime check for the full kernel combination and its padding behavior.

## Weights, calibration and cost provenance

No model checkpoint or Ascend device is available on this Mac. **Confidence weight
availability, calibration quality, and measured NPU cost tables are not claimed.**

`verification_tools.py checkpoint` checks the actual safetensors index/header,
projection shape, FLOAT quantization declaration and payload hash. The runtime
receipt separately hashes loaded parameter bytes and records the module and
checkpoint namespace. Missing weights fail explicitly. Specified-length tests
never count as learned-policy validation.

No calibration is invented. Without an explicit calibration file, results say
`uncalibrated` and use scale 1/bias 0. For a separate, held-out calibration corpus,
prepare paired `conditional_logits` and binary `conditional_accepted` arrays from
fixed-K greedy verification. Include only positions whose preceding candidates
were actually accepted; positions beyond first rejection have no conditional
label. Input JSON must declare `split: calibration` and the exact runtime
`weights_sha256`. Keep its request/epoch/position provenance alongside it. Do not
fit on the evaluation input set. Fit offline, outside inference timing:

```bash
python tools/dspark/verification_tools.py calibrate \
  --inputs /workspace/dspark-calibration/conditional-labels.json \
  --weights-sha256 LOADED_WEIGHT_HASH \
  --output /workspace/dspark-calibration/platt.json
```

The profile command below launches **separate disposable engines and requests**.
It never profiles against a live benchmark engine or carries KV/proposals/epochs
into a later run. Profile-only NPU events wrap actual target FULL replay and eager
draft execution, including the confidence head/host transfer. Events synchronize
only at profile boundaries. The cache records raw result hashes, actual contexts,
model/config/loaded-confidence hash, hardware name, TP/EP, dtype, graph/draft modes,
request/token limits and capture sizes. It uses conservative observed event costs
across ranks, not summed TP times. CPU heap scheduling overhead is also measured;
the CPU broadcast overhead is included in reported end-to-end/policy time but is
not separately fitted into the cost table. No interpolation beyond observed
context limits or missing draft batch sizes is allowed.

At B4, the exhaustive profile entry runs 16 independent small trials (outstanding
1–4, specified ell 0/1/3/5). It is **explicit**, not part of default first-round
validation or a performance result. Each trial has independent logs and receipts.
A profile missing a graph tier fails compilation, rather than fabricating a cost.
For larger configurations, collect compatible isolated measurements; do not reuse
a B4 cache for B64. `verification_tools.py profile --inputs ... --output ...` can
compile selected compatible raw profile result files.

## Server commands

Use Bash with the existing CANN/custom OPP environment already active. No OPP build
is needed. The manifest is one produced by `prepare_performance_data.py`; all modes
use its same ordered token IDs. Do not duplicate inputs to meet a requested count.
Replace `NEW_PLUGIN_SHA` below with the delivered commit and paths with your real
manifest/profile. The wrapper sets MRV2, TP8+EP, K5, BF16/W8A8, natural EOS,
temperature 0/top-p 1/top-k -1/seed 0, pickle fallback off, and devices 0–7. It checks
both exact repository SHAs/clean trees and resource ownership before each engine.

Obtain a compatible measured profile first if one is not already available:

```bash
git -C /workspace/vllm-ascend-hust pull --ff-only origin feat/dspark && \
bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_confidence.sh \
  NEW_PLUGIN_SHA /workspace/dspark-data/performance/manifest.json \
  --stage profile --batch 4 --num-prompts 4 --output-len 256 \
  --capture 6 12 18 24
```

The profile is printed under the new `dspark-confidence.XXXXXXXX/runs/cost-profile.json`.
Then the requested first round (only specified B1, specified mixed B4, learned B4):

```bash
bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_confidence.sh \
  NEW_PLUGIN_SHA /workspace/dspark-data/performance/manifest.json \
  --cost-profile /workspace/dspark-results/PROFILE_RUN/runs/cost-profile.json \
  --num-prompts 4 --output-len 128 --warmup-prompts 1
```

The specified B4 gate requires measured mixed-length FULL replay. Learned B4 must
have real head/budget decisions and measured FULL replay. If learned scores select
only K5 or no mixed batch is observed, the receipt says so; scores are not altered
to manufacture adaptive mixing. Request scheduling need not reach configured
maximum concurrency. All launches, logs, `PIPESTATUS`, return codes, results and
receipts are retained in a new directory and packaged as evidence, even on failure.
No test failure triggers larger runs or kills another process.

After the first round passes, B16/B64 are separate explicit commands, each requiring
a matching measured profile and token budget. For example:

```bash
bash tools/dspark/run_dspark_confidence.sh NEW_PLUGIN_SHA MANIFEST \
  --stage extend --batch 16 --num-prompts 16 --capture 6 12 24 48 96 \
  --cost-profile PROFILE_B16 --output-len 256
```

Use batch 64 and captures ending at 384 for B64. For the existing 400-request
workload, use `--num-prompts 400 --batch 128` (then independently 256 or 400), capture
sizes ending at 768/1536/2400, and compatible profiles. None is run automatically.
Client outstanding is a separate `--client-outstanding` option; absence means
all-at-once source-order admission. Actual batch shapes come from replay records.

For performance, save `confidence.json` with mode `confidence`, a compatible
`cost_profile` path, and optional `calibration` path. After smoke passes, the existing
performance wrapper accepts the third mode and independent repeats. For example,
the explicitly created configuration file contains:

```json
{
  "mode": "confidence",
  "cost_profile": "/workspace/dspark-results/PROFILE_RUN/runs/cost-profile.json"
}
```

Use it for the comparison:

```bash
bash tools/dspark/run_dspark_performance.sh code64 NEW_PLUGIN_SHA MANIFEST \
  --modes target_graph dspark_graph dspark_confidence_graph --repeats 3 \
  --num-prompts 64 --max-num-seqs 4 --output-len 128 \
  --capture-target 1 2 3 4 --capture-dspark 6 12 18 24 \
  --confidence-verification /workspace/dspark-profiles/confidence.json
```

Use the same source/model/input/sampling/memory/delivery configuration and a profile
covering the full chosen context range. The three modes are target-only graph
(`speculative_config=None`), fixed-K hybrid, and confidence-scheduled hybrid.
Run order reverses on alternating repeats; every repeat launches a fresh process,
loads, captures, warms up, measures and shuts down. Natural EOS/output differences
are reported, not an exact-token performance gate. Docker-unavailable code quality
stays unavailable and is not a performance blocker.

## Reading the results and remaining checks

- `result.json`: actual output text/tokens, throughput, existing monotonic TTFT and
  request TPOT, graph snapshots and `confidence_verification` schema 1. Events that
  deliver multiple tokens remain one event; TPOT is unavailable for <=1 token.
- `confidence_verification.per_rank`: measured generated/scheduled/verified/accepted
  counts, correct per-position denominators, verification-length histogram,
  effective/capacity token shapes and policy/transfer time. Rank agreement is
  checked, and one logical execution is counted once, not eight times.
- `graph_execution.boundary_snapshots`: raw baseline/warmup/measured snapshots,
  including actual successful FULL calls and request/producer/confidence epoch decision records. Capture/dummy/failed calls cannot count
  as measured success. Eager fallback evidence remains unavailable, not a fake 0.
- `receipt.json`: individual outcome, with no PASS on error, NaN log report,
  incomplete output or absent replay evidence. Earlier completed outputs/timing
  survive telemetry failures. Specified-length/profile runs are ineligible for
  learned-policy performance comparison.
- Performance suite `summary.json`, `.csv`, `.md`: original runs, mean/median,
  sample standard deviation/CV (`null` for n<2), per-repeat paired speedups and
  ratio of medians. Request quantiles remain separate from across-run statistics.

The local test suite executes policy and real production function bodies with CPU
Torch, including real DSA shared-KV validation and the compiled padding-scatter
selector. Installed-class variants are also provided. It does not run CANN,
ACLGraph replay, TP collectives, the real checkpoint, or the three-mode performance
comparison. Server focused and staged graph tests must pass before expanding.

Local verification uses Python 3.12.13 / Torch 2.10 CPU. The scoped source/CPU
regression command passed 278 tests, skipped one missing-vLLM installed test and
deselected 103 runtime variants. The dedicated feature file additionally runs its
installed variants when dependencies exist; here they skip because neither vLLM
nor torch_npu is installed. No model weights, calibration corpus, measured NPU
cost profile, Docker code evaluation or server benchmark was executed locally.
The raw profile and policy/calibration JSON assets are frozen and hashed before
engine launch; phase snapshots retain the decision-to-producer epoch mapping.

Changed-file manual pre-commit checks pass. The required full `bash format.sh ci`
was also run in an isolated worktree: it fails on the repository-wide baseline
(lint/spelling/import/shell/workflow issues; 78 unrelated files autoformatted).
Those unrelated changes are not included. Full-repository lint is not reported
as passing. The feature plus benchmark regression run passed 169 tests with
three dependency skips; this is separate from server/NPU validation.

The dedicated feature tests passed 36 tests with three installed-dependency skips,
including a real CPU Torch logistic calibration fit on explicitly synthetic test
labels (not a model calibration result). Confidence head calls are counted after
finite logits are observed. Both head calls and current-epoch budget decisions
are required for measured learned-policy evidence.

Checkpoint preflight accepts `model.safetensors.index.json` and
`quant_model_weights.safetensors.index.json`. The standard index has precedence,
matching frozen `DefaultModelLoader._prepare_weights` and
`filter_duplicate_safetensors_files`. Without it, the loader scans top-level
`*.safetensors`; the quantized index is used as an audit manifest and its confidence
shard must belong to that selection. If both indices exist, their complete
`weight_map` dictionaries must agree; conflicting maps fail before shard access.
The receipt records `index_file`, `index_sha256`, `index_selection` and hashes of
all available supported indices. No model files are renamed or synthesized.
Cost compilation and compatibility checks do not depend on an index filename;
they retain loaded-confidence/config/hardware/capture identity checks unchanged.
The compatibility regression run passed 41 tests with three dependency skips.
