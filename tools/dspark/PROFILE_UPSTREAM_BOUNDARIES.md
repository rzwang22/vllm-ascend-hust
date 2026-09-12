# DSpark profile: local draft upstream boundaries

The `numeric-boundaries` server run reproduced NaN **before** the internal
norm/LM head. This delivery adds the next local observation interval; it does not
claim a proven producer or a validated production repair. New NPU execution is
still required and remains user-owned.

## Independently verified server evidence

Read `PROFILE_NUMERIC_BOUNDARIES.md` first for the preceding implementation.
The new archive was actually read and safely extracted, with all extracted
regular-file bytes verified against the tar members:

- `dspark-large-batch.4T8j76eT-evidence.tar.gz`, 80 members, 116463351 expanded bytes.
- SHA256 `9eedf838b7dd3a6347f4d11815ead523a02d3a9a3a11057ca6b5fae8cb27939f`.
- Plugin `7e29173585cd07238cc2b18c3e70fb485b365b69`.
- Frozen Core `897306c43bf800e2480cb5c0f3e2da408d85a2fd`.
- The local repositories were clean and matched these SHAs before development;
  the fetched `origin/feat/dspark` also matched. No reset was performed.
- Server `focused.log`: 826 passed. All nine completed point JSON hashes match
  `runs/b64/retained.json`. One B64 engine, TP8+EP, MRV2, K5, target
  FULL_DECODE_ONLY, eager draft, original synthetic point prefix.

[PROFILE_4T8j76eT_AUDIT.json](PROFILE_4T8j76eT_AUDIT.json) saves the independent
rank-by-rank extraction: member paths/hashes, row identities, transition
selections/owner epochs, three numeric rounds and raw completed-point hashes.
It is an offline witness, not another NPU test.

For **all eight ranks**, `recording_error=null`, `numeric.enabled=true`, all
98 compact D2H attempts completed, and classification counts are
`both_finite=97`, `hidden_nonfinite=1`. `first-nan` and `first-nonfinite` contain
the same numeric records as `first-failure`:

| Execution | Proposal epoch | Hidden and base logits | Target CPU seq_lens | Draft CPU seq_lens |
| --- | --- | --- | --- | --- |
| 1803 | 1790 | All rows finite | [221,245,518] | [226,250,523] |
| 1804 | 1791 | All rows finite | [222,249,524] | [227,254,529] |
| 1805 | 1792 | NaN only at candidate rows 0–4; no Inf | [223,253,530] | [228,258,535] |

The affected request is `batch10-3-a775c70b`, request row 0, candidate positions
0–4. Actual order is `[batch10-3-a775c70b, batch10-2-875306bf, batch10-1-a3f7a933]`;
query lengths `[1,4,6]`, ell `[0,3,5]`, target `query_start_loc=[0,1,5,11]`.
There are three requests, 11 valid target rows and 12 graph rows, while the
candidate tensors have 15 rows: hidden `[15,4096]`, logits `[15,129280]`.

Every rank retained the transitions 1708→1709 (one to two requests), 1709→1710
(two to four), and 1795→1796 (four to three), with zero transition eviction.
The last transition verified old epoch 1782 and published 1783 for survivors.
NaN occurs nine executions later; it is not the exit iteration. Rank-local pool
rows differ: ranks 0/1/3/4/7 use `[62,61,63]`, ranks 2/5/6 use `[62,63,61]`.
These remain separate from request row and candidate coordinates.

Original Markov NaN checking raised normally. Ownership and EngineDead errors
followed. Cleanup completed without timeout; supervisor `signals_sent=[]` and
return code 1. The exit protection worked; this change does not revisit it.

Evidence anchors within the archive are
`runs/b64/worker-first-failure/rank-N-first-nan.json` → `numeric.rounds`,
`rank-N-first-failure.json` → `records` and `transitions`, the nine raw point
JSONs plus `retained.json`, `cleanup.json`, and `runs/b64-supervisor.json`.
Final all-rank NaN agreement cannot locate the first bad rank or operator.

## Source investigation and limits

The source path was followed through
[proposal preparation/context/backbone](../../vllm_ascend/worker/v2/spec_decode/dspark/speculator.py),
[draft model](../../vllm_ascend/models/deepseek_v4_dspark.py),
[decoder HC/attention/FFN](../../vllm_ascend/models/deepseek_v4.py), and
[DSA metadata/scatter/attention](../../vllm_ascend/attention/dsa_v1.py).

`prepare_proposal_inputs` computes valid query ends from device query starts
minus this round's rejected counts, indexes the last valid target position,
and constructs K positions beginning one token later. Ell controls target
verification width; all live requests, including ell=0, still draft K5.
Context preparation concatenates only `num_target_tokens` auxiliary rows and
applies `main_proj` then `main_norm`. Each layer projects shared KV, applies KV
norm/RoPE and scatters it to the cloned context slots. Draft queries use their
own persistent slot rows. The real groups are mtp.0/mtp.2 → 3 and mtp.1 → 2,
all block size 32, with separate layer KV tensors despite shared group IDs.

The draft starts with embedding expanded across HC streams. Each MTP layer
performs attention HC pre/norm/attention/HC post, then FFN HC pre/norm/MoE/HC
post; its hidden output feeds the next layer. The residual argument is replaced
by a clone of that layer's incoming hidden. Final `hc_head` aggregates the HC
streams to `[candidate_rows,4096]`, which is the observed head input. The
head-before-norm NaN rules out an exclusively later-head explanation for the
first detected anomaly, but none of these upstream numerical values were saved
in the supplied run.

A block-boundary hypothesis was checked without turning it into a diagnosis.
For request row 0, the saved draft CPU lengths imply query positions
221–225, 222–226, 223–227 under the unchanged construction. **Both preceding
finite rounds already cross the same 32-token block boundary.** Frozen Core
`scheduler.py` sets DSpark `num_lookahead_tokens=K` and passes it to
`KVCacheManager.allocate_slots`; allocation includes those lookahead tokens.
Thus ell=0 does not by itself prove missing K5 space. Nor does this prove actual
worker block entries/allocation lifetimes correct: the archive contains no
device index/slot values or cache contents. Bounds near `max_model_len` and
block-table column clamps are not shown to activate in this failure.

Target CPU lengths and draft CPU lengths describe different sequences. Their
difference is not evidence of contamination. Stable storage addresses, matching
host layouts and same-epoch method returns cannot establish finite auxiliary
values, correct device gather indices, cache contents, or a correct consumer
stream dependency. No definite off-by-one, stale epoch, wrong pool mapping or
allocation defect was proved, so no speculative production fix is included.

## Independent opt-in experiment

Use `--profile-experiment upstream-boundaries`, absent by default. It retains
`numeric-boundaries` for comparison and includes those same final two checks.
The profile factory installs a separate observer after capture. Draft loading
already enforces `CompilationMode.NONE` and `CUDAGraphMode.NONE` in
[model_loader.py](../../vllm_ascend/worker/v2/spec_decode/dspark/model_loader.py),
so the draft instance hooks run in Python. No target layer or global class is
patched; no production model, Core, custom op, confidence allocator, graph
execution or existing NaN/ownership check is changed.

The observer reduces these **13 upstream boundaries** for this three-layer
model, deriving layer counts/names and tensor shapes at runtime:

| Boundary | Actual tensor and row domain |
| --- | --- |
| `target_aux.<layer_id>` (three) | Feature segments of the actual valid-token concatenation entering `combine_hidden_states`, before main projection/norm; target rows |
| `context.projected` | Returned projected/normalized context; target rows |
| `mtp.i.context_kv` (three) | Actual shared KV input immediately before that layer's context scatter, after KV projection/norm/RoPE; target rows |
| `draft.initial_hidden` | Actual first MTP layer input after embedding expansion; candidate rows, all HC/features reduced within a row |
| `mtp.i.output` (three) | Each layer's actual returned hidden; candidate rows |
| `draft.hc_input`, `draft.hc_output` | Actual final HC aggregation argument and return; candidate rows |

The last layer output and HC input normally describe the same values at adjacent
call boundaries. They are recorded explicitly to avoid assuming that aliasing
proves equality. Context KV statistics describe **current write inputs**, not
cache readback or historical cache finiteness. No whole KV cache, target model,
or internal attention/FFN tensor sweep is exported.

Every row has NaN/Inf flags, point, rank, execution, proposal epoch, request ID,
request row and position within that request. Target rows use the copied actual
CPU query spans; candidate rows use proposal K and never use target padding.
`target_mapping_matches_device` explicitly compares the captured device query
starts with the CPU mapping. If false, row labels remain the intended CPU batch
assignment and do not establish which request the malformed device spans read.

A small owned device integer snapshot is taken at draft entry: request-state
indices, target query starts/positions/lengths, sampled/rejected counts, draft
query starts/positions/lengths and input token IDs. Context slot IDs are copied
at each actual scatter input and carry the cache-layer/group name. These expose
whether `query_start_loc[i+1]-num_rejected[i]` remains inside request i's query
span, and allow safe offline reconstruction of its last valid position. They
do not dump block tables, allocator state or claim slot ownership is proved.
No potentially invalid diagnostic device gather is introduced.

All reductions run on the corresponding eager caller stream before subsequent
buffer reuse. Only fresh `[rows,2]` flags and owned integer snapshots remain
pending. On head return they join the original hidden/logits flags in **one
int64 packet, one blocking D2H and one host wait per proposal/rank**. There is no
wait per upstream boundary and no additional global synchronization. For the
historical T=11 target rows, Q=15 candidate rows and N=3 requests, this packet
is 3928 bytes (rather than numeric-boundaries' 60 bytes). The actual total is
reported as `upstream.packet_bytes` per point. Existing model CPU waits remain.

Each boundary adds two full tensor scans and temporary boolean masks; HC tensors
reduce all trailing dimensions. The masks are transient; only row flags remain
until the combined transfer. Packet assembly widens flags to int64, and the
small integer fields are copied before reuse. No full floating tensor is cloned
or transferred by this observer. These allocations, launches, CPU bookkeeping
and waits can change reproduction even though they leave model values untouched.

## Reading the new artifacts

`upstream.rounds` is an independent three-proposal ring with the current and
preceding two rounds in the same point (fewer at point start). It includes
boundary shapes/dtypes, row flags, integer snapshots, expected-but-missing
boundaries and current `head_flags` in candidate order. Aggregate counts cover
all observed rounds in the point. Point RPCs carry only compact counts plus the
hashed worker-file receipt; histories stay worker-local.

- `rank-N-upstream-first-nan.json` saves the first upstream NaN and prior rounds.
- `rank-N-upstream-first-nonfinite.json` separately saves the first NaN/Inf, even
  if a later stage hides it and Markov sees finite values.
- Existing `rank-N-first-nan.json`, `first-nonfinite.json` and `first-failure.json`
  now also contain the upstream history in this mode. Original first exceptions
  are never replaced by later ownership or cleanup errors.
- `rank-N-latest.json` still means the last completed point, not post-error state.

First-anomaly files are written/flushed/fsynced before returning to the original
Markov check. If upstream and head first NaN coincide, there are four anomaly
file writes/fsyncs per rank, plus a fifth for the caught first-failure. Normal
finite rounds add no disk writes. An exception before the head drains one
partial packet, marks unreached boundaries missing and preserves the original
error. Missing hooks/observation errors cannot become a finite pass: inspect
`recording_error`; successful point completion rejects an observation error.
Storage/device failures can still prevent a complete file and must stay explicit.

Interpret the **first observed bad boundary for the same request**:

- Bad auxiliary segments: investigate their target producer/consumption path.
- Finite auxiliary but bad projected context or context KV: narrow to the
  corresponding projection/norm/RoPE interval, without assuming cache contents.
- Bad initial hidden: investigate embedding/input IDs and initial construction.
- Finite observed inputs but first bad `mtp.i.output`: narrow to that layer,
  including its attention/KV reads, HC and MoE/communication. This is not yet an
  operator attribution; only then add the necessary cache/index or sublayer read.
- Finite HC input but bad HC output: narrow to final HC aggregation.

Same bad flags on all ranks do not identify the first bad rank. If all ten points
finish, record **not reproduced under upstream observation**, not repaired.
`performance_eligible=false`, diagnostic identity, cost compilation rejection,
original point order, bounded teardown and failure archiving remain enforced.

## One server run

Use the delivered SHA in the existing CANN/custom OPP environment, retaining the
frozen Core and original manifest. This starts exactly one B64 profile engine
through the same tenth point; no B128/B256, full cost profile or comparison runs:

```bash
DSpark_SHA='<40-character signed-off SHA from this delivery>'
cd /workspace/vllm-ascend-hust &&
  git fetch origin feat/dspark &&
  git merge --ff-only "$DSpark_SHA" &&
  bash tools/dspark/run_dspark_profile_control.sh "$DSpark_SHA" \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
    upstream-boundaries
```

The existing wrapper keeps TP8+EP, K5, greedy/seed 0, synthetic contexts 128/2048,
512 output tokens, warmup 2, samples 5, captures `[6,12,24,48,96,192,384]`,
8192 model/batched-token limits, memory utilization 0.9 and block size 32. It
checks source/config, runs installed-state focused tests, preserves process
supervision and archives failures. Return only the generated
`dspark-large-batch.*-evidence.tar.gz` and `.sha256`; those two artifacts include
all eight ranks, per-point summaries, request mappings, logs and exit receipts.

Local CPU/mock tests exercise the real draft forward/HC source bodies, each
upstream injection, target/candidate mapping with padding, Inf and reused
buffers, single-packet transfer, histories, early errors, disabled hooks,
missing boundaries, profile factory and the exact shell argv. They do not test
NPU execution, TP collectives, custom kernels or repaired inference. The three
existing installed-vLLM/Ascend confidence tests remain unavailable locally.

## Local validation for this delivery

On macOS, Python 3.12 and CPU Torch 2.14.0, the focused standalone suite below
completed with **359 passed, 3 skipped**. The new upstream file contributes
17 tests, including a head exception with a combined hidden-only packet and a
stale proposal epoch that cannot produce valid evidence. `--noconftest` avoids
the repository's installed-vLLM/Ascend fixtures; the server command above runs
the installed-state suite normally before its single NPU experiment.

```bash
python -m pytest --noconftest -q -ra \
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
```

All changed-file manual pre-commit hooks, shell syntax and `git diff --check`
passed. Required `bash format.sh ci` was also run against the staged patch in a
disposable worktree and returned 1. Its failing hooks and 78 automatic changes
match the preceding numeric-boundaries delivery; none of those autochanged
files belongs to this change. Existing failures include Ruff in
`examples/offline_inference_npu.py`, forbidden imports in
`vllm_ascend/diagnostics/w8a8_nz.py`, and repository-wide spelling/Markdown,
workflow and shell checks. The disposable changes were discarded. A full
repository lint pass or NPU validation is not claimed.
