# Profile NaN: on/off evidence and controlled reproduction

Status: **ROOT_CAUSE_NOT_YET_PROVEN / SERVER_NOT_REVALIDATED**.
This change adds controlled reproduction and host metadata evidence. It does not
repair or identify a numerical producer. Core, attention, custom ops, the
confidence policy, sampling and proposal computation are unchanged.

## Verified evidence

Both local tar archives were SHA256 checked, and every member checked for safe
relative paths, regular files/directories and absence of links before extraction.
The audit read both complete worker logs, plans, captures, all ten point files,
streaming records and all eight A `latest` files including `previous_executions`.

| Run | Archive SHA256 | Result |
| --- | --- | --- |
| A: `dspark-large-batch.kHH7EWE3` | `b74eb11aa379ad9527062a69af57d4695dc4e19b35cfc3e5b9b4bf6d3a4e97e3` | 10 points; no observed failure |
| B: `dspark-large-batch.goVkaa1v` | `d25714576aa3e71a9ff36cabe7e9c962969257ac5cb35ecadb2e3d5b2012a3a6` | 9 points; point 10 failed |

The separately provided `profile-nan-off.V56IeQ10.log` was also read: SHA256
`af28e298cc1aa310c7c4d281241fe7f36d714fb9a983786c5c8cfa7c4f9ed4ad`.
Both source gates identify plugin `52cc5900efa889fb21c1b2e13d88613ca90eab1b`
and core `897306c43bf800e2480cb5c0f3e2da408d85a2fd`. Both lifecycles report one
initialization and shutdown. All eight ranks report seven captures at
`[6,12,24,48,96,192,384]`, target FULL_DECODE_ONLY, actual draft mode NONE,
npugraph_ex enabled and static_kernel disabled.

Both runs use contexts `[128,2048]`, output budget 512, warmup 2, samples 5,
model/token budgets 8192, memory utilization 0.9, TP8+EP and B64 engine capacity.
The source input hash is
`3889b4bda22442e69062cd5c3888090515b67a303006ef59afea8d72424258b2`;
imported requests hash is
`323363ee93f04f0064eb544acf12dcad63aee6352f0fc26dffde222de2fd5448` in both.
The manifest remains 400 instances / 64 unique prompts with prefix caching off.
Profile points actually submit synthetic prompts, not the 400-request performance
load: `collect()` repeats the first token from `tokenizer.encode("x")` to
`prompt_tokens` for each point request.
Observed 128-token prompt sequences are identical in the compared point files.
Synthetic profile uses ignore-EOS with 512 output tokens; this does not change
natural EOS in performance runs. Random internal ID suffixes differ as expected.

The only intended command difference before point 10 was A's full diagnostic
flag and stop-point flag. B had the complete grid; A ended at point 10.
This is not an execution-identical trace: admission, completion and output events
already differ before the failing point. For each point below, successful FULL
calls are per rank, never summed over TP8. Each available point's eight ranks
agree on its target/draft shape counts.

| Point (all `ctx128-`) | A FULL | B FULL |
| --- | ---: | ---: |
| n1-t6-balanced | 87 | 87 |
| n1-t6-skewed | 87 | 87 |
| n2-t6-balanced | 172 | 172 |
| n2-t6-skewed | 186 | 186 |
| n2-t12-balanced | 87 | 87 |
| n2-t12-skewed | 86 | 87 |
| n4-t6-balanced | 384 | 384 |
| n4-t6-skewed | 411 | 411 |
| n4-t12-balanced | 172 | 172 |
| n4-t12-skewed | 199 | unavailable |

For n4-t12-balanced, A has 171 FULL `[3,3,3,3]` and one `[3,3,3]`, capacity 12.
B has 169 FULL `[3,3,3,3]`, two `[3,3,3]` at capacity 12 and one `[3]` at
capacity 6. Both have three additional mixed-admission draft calls. Times were
76.4468905 s versus 21.2219873 s. These are diagnostic/profile times, not
publishable throughput results.

For n4-t12-skewed, A has 85 FULL `[1,1,4,6]`, 29 `[1,4,6]`, 53 `[4,6]` at
capacity 12, then 32 `[6]` at capacity 6, plus three mixed-admission draft calls.
Requests complete in external-ID order 0,1,2,3, each with 512 output tokens.
All eight latest files end at target execution epoch 1908, empty scheduler
requests, `no_target_forward`, proposal epoch 1893 and no owners. Their retained
preceding executions contain no recorded numerical failure. They only describe
A's final rounds, not B's failure round.

B fails at **ctx128-n4-t12-skewed**, but its failed point has no rank snapshot.
`batch10-0` completes 512 tokens at monotonic 4325606.325238012. Surviving
requests 1/2/3 have 397/122/95 frontend output tokens when EngineDead arrives.
Their last nine output events deliver 6/4/1 tokens respectively. This proves
progress after an exit; event sizes do not prove target query lengths or the
producer epoch. B's final point RPC failed: `request_identity_failure=null`
is not a successful worker ID validation.

Every B rank logs base-logits NaN before ownership failure at 11:04:43.
The later scheduler dump contains survivors 1/2/3, six scheduled tokens each,
placeholder proposals, computed `[524,251,227]`, output counts `[397,124,100]`.
Those differ from frontend delivered counts and are **not** the failed producer
input. No failure epoch/actual query/rejected counts can be recovered from B.
The ownership error is consistent with the subsequent consumer lacking a new
publication after NaN; it is not evidence for an independent root cause.

A's outer `stages.json`, PIPESTATUS and status.txt report success (0); its old
`b64-command.json` retained the initial `rc=null/status=failed` placeholder.
The driver now finalizes that file too. This reporting defect did not cause NaN.
Neither archive contains a cost table; A correctly cannot publish one.

## Source audit and uncertainty

* `verification_runtime.select()` applies specified lengths to sorted eligible
  candidate IDs each round, broadcasts lengths/epochs and trims SchedulerOutput.
  After an exit, lengths are reassigned among survivors, not permanently attached
  to their initial row. Mixed admissions retain their existing query widths.
* `model_runner.prepare_inputs()` sorts actual rows by prefill status/query
  length and rebuilds idx_mapping, cumulative query offsets, logits mapping and
  padding. It calls `_update_seq_lens_cpu()` before reading corrected CPU lengths.
  That routine already waits for `num_computed_tokens_event`; the preceding D2H
  stream waits for the default stream. CPU scheduler upper bounds and effective
  KV lengths are separate fields, as retained by `profile_context()`.
* Core `buffer_utils.async_copy_to_gpu()` pins host input then copies with
  non_blocking=True. Query/idx/logits host arrays are newly prepared; persistent
  input buffers and the corrected-length CPU staging area are reused. There is
  no B device-content or in-flight allocator evidence proving premature reuse.
* `AscendInputBatch.make_dummy()` retains shallow field aliases. Model state and
  DSA builders update persistent metadata; `ModelAclGraphManager.run_fullgraph()`
  consumes the per-update shared-KV preflight, replays and returns core's views
  of persistent hidden/aux buffers. DSA `update_graph_params()` is a no-op;
  creation of `update_stream` alone does not prove missing DSA stream ordering.
* `prepare_proposal_inputs()` binds request/state rows, epoch, sampled/rejected
  counts, target query ends and aux views. Last valid position is selected at
  `query_start_loc[1:] - num_rejected - 1`; ell=0 still has one target query.
  Draft input remains five tokens/request. Context slot mappings are cloned by
  KV group, while target hidden/aux and several input tensors remain aliases.
* `_combine_and_precompute_draft_context()` concatenates target aux, projects
  context, then `precompute_and_store_context_kv()` projects/normalizes/RoPEs KV
  and invokes `DeviceOperator.dsa_kv_compress_scatter()`. `_run_draft_model_forward()`
  subsequently reads draft KV. The outer context projection/scatter path has no
  explicit stream switch. Attention has internal auxiliary-stream event/wait
  pairs; CPU source inspection cannot establish their device completion here.
* Markov sampling computes draft logits from draft hidden, checks NaN before
  returning a result, and publishes only after success. B proves the first
  **detected** boundary is base logits. It does not distinguish bad target/aux,
  proposal indexing, KV contents, draft forward or LM-head output as producer.

## Full diagnostic interference

`dspark_nan.py` introduces the following in addition to ordinary profile NPU
Events and point-boundary synchronization. These operations are not present in
the new metadata-only path.

| Boundary | Added device work / allocations | Added host waits / files |
| --- | --- | --- |
| begin execution / every write | Owner-row tensor detach/copy | Owner D2H, Python copies, JSON, latest-file replace |
| target output | NaN/Inf row reductions, stacks/flags, possible reshape allocation | Flags, positions, seq_lens, logits rows, query spans, slot/block IDs D2H; JSON |
| proposal preparation | Hidden/aux row checks | Rejected/sampled/state indices, draft positions/spans/slots and block prefixes D2H; JSON |
| combined context | NaN/Inf reductions | Flags D2H and JSON before KV write |
| context KV | Valid-slot mask, integer division/remainder, indexed KV gathers, row reductions | Gather/check results and slots D2H; JSON before draft forward |
| draft hidden / base logits | NaN/Inf reductions and temporary flags | Flags D2H; JSON before original Markov check/publication |

`_stats()` copies reduction results to CPU; `_integer_record()` copies integer
tensors. The host waits can delay next metadata updates/replay and allow pending
stream work to finish. Gather/temporary allocations can change allocator reuse;
JSON/file writes change host admission/completion timing. These simultaneous
changes prevent attributing non-reproduction to one dependency. No missing event
or ownership fix is justified by these archives alone.

## New experiments

`--profile-experiment` is independent of full diagnostics; it reuses the original
grid prefix through `--profile-stop-after-point` (default n4-t12-skewed).
No new flag means the unchanged full startup profile and cost generation.
Full diagnostics and experiment flags are mutually exclusive. All experiments
are B64 profile only, one engine, performance_eligible=false, no cost publication.

| Mode | Added work |
| --- | --- |
| baseline | Prefix stop/receipt only; no new worker observer or synchronization |
| metadata-only | Instance-local wrappers, bounded 128-record deque, CPU array-to-list copies and tensor shape/stride/object/storage address descriptions |
| context-kv-sync | Exactly one wrapper after `model.precompute_and_store_context_kv` returns; `torch.npu.current_stream().synchronize()` once per completed context write |

Metadata recording adds no device numerical checks, tensor transfers, waits,
per-step RPC or disk writes. It never retains tensor/numpy views in history.
It records actual request rows, host query offsets/lengths, scheduler/effective
lengths, epochs/owners/selection, tensor descriptors for target/proposal/sampling,
KV groups, persistent inputs and captured outputs. Tensor **values** including
sampled/rejected counts, slots and KV contents remain explicitly unavailable.
Entry scheduler metadata is labeled separately from the prepared InputBatch.
Return records mean the call returned, not that tensors were checked finite.

On an original exception, the innermost installed boundary saves a worker-local
first-failure file before rethrowing that same exception. Cascaded catches cannot
replace it. A disk error cannot replace the original exception either (the
absence of a file then means missing evidence). At the existing point RPC, latest
history is saved; observation errors fail evidence collection. Wrappers are
instance-local; `close()` restores previous methods. No global or core patch is
installed. Rank process teardown ends the observer's lifetime.

The sync control installs **no metadata-only wrappers** or full numeric checks.
Its snapshot records point-local attempted/completed waits, stream handle and
boundary. It waits only the caller stream, not the whole device or all TP ranks.
Compare it to baseline first: improvement is correlation around context write /
subsequent draft work, not proof of a race or of which producer is faulty.
Metadata overhead can also change reproduction; completion alone is never a fix.

## Server commands

Set `PROFILE_SHA` to the exact delivered commit. Keep existing CANN/custom OPP;
no OPP build, package reinstall, device reset, process killing or later sweep.
Each invocation checks both SHAs and idle NPUs, runs focused tests and creates
`/workspace/dspark-results/dspark-large-batch.XXXXXXXX`. Failures retain logs,
PIPESTATUS, lifecycle, raw points, worker evidence and a tar archive.

```bash
cd /workspace/vllm-ascend-hust
set -o pipefail
PROFILE_SHA=<delivered-40-character-SHA>
git fetch origin feat/dspark && git merge --ff-only "$PROFILE_SHA"
PROFILE_MANIFEST=/workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json
bash tools/dspark/run_dspark_profile_control.sh "$PROFILE_SHA" "$PROFILE_MANIFEST" baseline
```

Run the next experiment manually, in a new invocation. These do not start B128,
B256, validate or repeat. Do not use a shell loop or chain success into large runs.

```bash
bash tools/dspark/run_dspark_profile_control.sh "$PROFILE_SHA" "$PROFILE_MANIFEST" metadata-only
```

Then, when ready to compare a single wait boundary against baseline:

```bash
bash tools/dspark/run_dspark_profile_control.sh "$PROFILE_SHA" "$PROFILE_MANIFEST" context-kv-sync
```

Read `runs/b64/diagnostic.json` for mode/prefix/outcome, `lifecycle.json` for one
initialization, `runs/stages.json` and `*.pipestatus` for execution results.
In metadata mode read **all** `worker-first-failure/rank-*-first-failure.json`:
point, execution, proposal epochs, ordered request/query rows, ownership and the
last 128 stage records. `rank-*-latest.json` is from the last reachable boundary;
it must not override first-failure. Raw point JSON retains frontend internal-ID
mappings and point-boundary rank snapshots. Baseline intentionally has no new
worker history. Sync mode has counters, not numeric producer evidence.

A complete run is `completed_without_observed_failure`, never “NaN fixed” or a
performance PASS. Still required: a failure trace without full numeric diagnosis,
then evidence tying the first bad values to a producer and its buffer lifetime.

## Local validation

Related CPU/source/mock suite: **341 passed, 3 skipped**. The three skipped
installed-vLLM/Ascend confidence tests require dependencies absent on this Mac.
Tests exercise immutable CPU snapshots across exits/reordering, original CPU NaN
rejection, bounded first-error retention, failed writes, cleanup, actual profiler
and replay-observer composition, exact sync counts, point isolation, prefix/CLI
routing and refusal to compile costs for every experiment. The original graph,
request-ID, context, repeated-input and benchmark tests remain covered.

Changed-file pre-commit checks and shell syntax pass. NPU execution, real model
outputs, asynchronous device ordering and installation-state integration have
**not** run locally. CPU mocks are not evidence of a successful server profile.

After final metadata/timer-boundary adjustments, the affected focused subset
also passes: **61 passed**. Full `bash format.sh ci` was run in a disposable
worktree with this patch: it remains failing on repository-wide lint/format
issues (78 other files automatically modified; zero task files modified).
Those unrelated changes were not included. Changed-file checks pass independently.

## Targeted numeric follow-up

The QKmLESDh archive audit and the opt-in `numeric-boundaries` experiment are
documented in [PROFILE_NUMERIC_BOUNDARIES.md](PROFILE_NUMERIC_BOUNDARIES.md).
It observes only the actual hidden input and returned base logits around
`compute_draft_logits`, with one compact D2H wait per returned head. It retains
first-NaN evidence, prior rounds and CPU request-set transitions. Existing
controls retain their device behavior; this is diagnosis, not a validated fix.
