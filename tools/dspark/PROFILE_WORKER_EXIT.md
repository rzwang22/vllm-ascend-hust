# Metadata-only worker exit and bounded failure propagation

Status: **SERVER_NOT_REVALIDATED / ROOT_CAUSE_NOT_YET_PROVEN**.
This repair preserves exit evidence, bounds failed profile teardown and removes
large metadata histories from RPC responses. It does not establish why worker 5
died, or reproduce/repair the earlier point-10 base-logits NaN.

## Archive audit

The local archive `metadata-only-4tANjjLZ.rdMokEoz.tar.gz` was read in full.
Computed SHA256:
`84625e13e2c27b28d2eb97ff4a617ef42e47294027a0d6438f3326dbae94fbfb`.
No expected hash was supplied for this new archive, so this is its independently
computed identity, not a comparison with a supplied checksum. All 37 members
were checked before extraction: safe relative paths, regular files/directories,
no links; total expanded size 251,839,365 bytes.

Source receipts match plugin `cd4d8fc6bf51326b6aaee4afd963a061a74d7713` and
core `897306c43bf800e2480cb5c0f3e2da408d85a2fd`. Plan: B64 engine limit, TP8+EP,
MRV2, K5, target FULL_DECODE_ONLY/draft eager, captures
`[6,12,24,48,96,192,384]`, contexts `[128,2048]`, synthetic output budget 512,
warmup 2, retained samples 5, model/token budgets 8192, memory utilization 0.9.
The performance input remains 400 instances / 64 unique prompts, prefix cache
false. Profile generation uses the existing synthetic token prompts and its
explicit ignore-EOS budget; it is not performance inference on those 400 inputs.

The complete logs, plan/capture/lifecycle, first-point raw and retained data,
streaming records and all eight worker `latest` histories were inspected.

| Time (server log) | What is evidenced |
| --- | --- |
| 14:59:58 | First point `ctx128-n1-t6-balanced` completed; frontend saved its raw and retained records |
| Before the exit | All eight `latest` files are already for second point `ctx128-n1-t6-skewed`; each has 128 retained records, 87 successful FULL call returns, 90 target-execute returns, and terminal empty executions |
| 15:00:24 | Existing parent monitor reports `VllmWorker-5` unexpected death (PID 28580, EngineCore PID 28104) and starts cleanup |
| 15:00:30 | Blocking collective RPC response wait raises `RuntimeError: cancelled`; EngineCore reports executor failure; AsyncLLM output handler reports EngineDead |
| 15:25:45 | User interruption finally unwinds the outer operation; old diagnostic receipt records KeyboardInterrupt |

The second point's final worker record is sequence 3206, execution 181, proposal
epoch 178. Each worker has no recorder error. `CostProfiler.snapshot()` performs
its existing point-boundary synchronization, then `finish_point()` writes local
`latest` **before returning** metadata/timing data through the worker response
queue and frontend utility serializer. `collect()` saves the point JSON only
after this RPC returns. Thus the second point ran and reached local snapshot
writing, but its frontend snapshot transaction did not finish. This does not
identify the exact instruction at which worker 5 died.

Each latest file is approximately 15.97 MB. The first-point JSON is 121,766,183
bytes; retaining everything except its duplicated observation histories yields
998,452 bytes when reserialized locally. The old ring was limited to 128 records,
but every record could recursively expand shared attention metadata, and the
whole ring was copied through both RPC boundaries. Per-point events/rings are
cleared on begin_point; no evidence proves unbounded cross-point accumulation.
Large host allocation/serialization is a concrete load, not proof of OOM or a
particular signal. Raw exitcode, OS signal and system/kernel logs are absent.
A missing worker Python first-failure file cannot distinguish abrupt process
termination from failure outside an installed wrapper.

## Failure path and changes

Frozen `AsyncMPClient._call_utility_async()` awaits a future in `utility_results`.
The output socket task forwards its exception to `outputs_queue`, but does not
complete outstanding utility futures. AsyncLLM's output handler then stops and
`errored` becomes true. The previous synchronous facade still waited indefinitely
for the independent utility future. Cancellation gathers and synchronous engine
shutdown also had no outer bound. A postmortem snapshot RPC could enter the same
wait again. The late KeyboardInterrupt was consequently the exception ultimately
saved by `startup_cost_profile.run()`.

Only metadata-only profile config enables the following supervision:

* `ProfileMultiprocExecutor`, selected through Core's supported importable
  `distributed_executor_backend`, inherits the existing multiprocess executor.
  The existing parent monitor calls its `shutdown()` after `is_failed=True` and
  before cleanup signals. It saves worker handles' PID, authoritative rank, raw
  exitcode, signal, UTC observation time, point and current blocking RPC.
  Running/unreaped handles have unavailable exit status. A signal is not a cause
  diagnosis. No extra worker monitor or global patch is installed.
* The frontend reads existing host `AsyncLLM.errored` state every 100 ms while
  waiting. It adds no health RPC, tensor copy, NPU operation or synchronization.
  Utility RPCs also have a 120-second timeout; generation has no duration cap.
  Pending operations are written at operation boundaries, never per decode step.
* On the first failure, original worker/engine evidence wins. Requests are
  cancelled with a one-second wait; no postmortem RPC is retried after known
  death. Partial streaming output and request-ID receipts remain available.
  Cleanup calls `AsyncLLM.shutdown(timeout=8)` in a bounded daemon thread, with
  another bounded cancellation drain. Failed cleanup cannot publish completion.
* An outer profile subprocess supervisor starts an owned process session. After
  a failure receipt it allows 20 seconds to save artifacts and clean up, then
  terminates **only that session**; after five more seconds it can send SIGKILL.
  Child return codes, sent signals and unreaped status are retained separately
  from the first failure. It also closes stranded descendant output pipes so
  `tee` can finish and the existing shell can save PIPESTATUS and create evidence.
  Unrelated NPU jobs/processes are never selected or signalled.

Worker metadata history stays local. RPC now returns a schema-2 compact receipt:
counts, PID, time, filename, bytes and SHA256. `latest` is overwritten at the next
point boundary; the receipt explicitly says so. Exception first-failure remains
exclusive. Descriptor expansion has an overall 512-node budget per described
object and marks truncation/alias references. Actual batch CPU arrays remain
independent copies; device fields are descriptors only. Normal metadata recording
still has no D2H, numerical checks, device waits, per-step RPC or disk writes.
These changes can affect reproduction through host overhead; a successful
metadata-only run is still not evidence that NaN or worker death has been fixed.

Core, custom ops, attention, ACLGraph execution, confidence allocation and
proposal/KV computation are unchanged. Normal profile/performance runs do not
enable the new failure observer/executor. The single-engine prefix, ID mapping,
context/sample validation and diagnostic cost-table prohibition are retained.

## Only rerun metadata-only

After synchronizing the signed-off commit printed in the delivery, run in the
existing CANN/custom OPP environment (no OPP build or package reinstall):

```bash
set -o pipefail
DSpark_SHA='<new 40-character commit from delivery>'
cd /workspace/vllm-ascend-hust
git fetch origin feat/dspark
git merge --ff-only "$DSpark_SHA" &&
  bash tools/dspark/run_dspark_profile_control.sh "$DSpark_SHA" \
    /workspace/dspark-results/dspark-repeated-input.WSJur2Ev/input/manifest.json \
    metadata-only
DSpark_RC=$?
printf 'METADATA_ONLY_DRIVER_RC=%s\n' "$DSpark_RC"
```

The existing wrapper enforces exact plugin/core SHAs, clean worktrees, idle NPU
resources, focused tests, fresh `/workspace/dspark-results/dspark-large-batch.*`
output, secure serialization, TP8+EP, the above captures/contexts/budgets and the
original first ten points in one engine. It retains logs/PIPESTATUS and archives
on failure without `set -e`, shell `exit`, device reset or unrelated process kills.
It does not run B128/B256, validate/repeat or generate a usable cost table.

Read `runs/b64/worker-exit.json` first if present, then `engine-failure.json`,
`frontend-operation.json`, `profile-failure.json`, `cleanup.json`, `lifecycle.json`
and `runs/b64-supervisor.json`. The last one is outside the engine directory and
survives a hung/aborted child. Worker-local `worker-first-failure/rank-*-latest.json`
contains the bounded CPU history; rank first-failure is only expected when a
wrapped Python boundary catches an exception. Missing raw exit information stays
unavailable. `diagnostic.json` must remain performance_eligible=false; failure
receipts prohibit a successful outer status even if the child returns zero.

Local checks exercise real CPU subprocess exits/signals, frozen Core monitor and
AsyncLLM state-property bodies, hung utility futures, resistant cancellation,
bounded cleanup, first-error preservation, compact msgpack receipts/hash and
metadata alias/row history. They do not exercise installed Ascend executor
initialization, NPU capture/replay, the server's abrupt exit, or NaN reproduction.

Validation on Mac (CPU Torch 2.10.0): related suite **355 passed, 3 skipped**;
the three existing confidence installation tests require unavailable vLLM/Ascend.
The focused exit/failure tests are included in that suite. Changed-file pre-commit
hooks and shell syntax pass. Required `bash format.sh ci` was also run in a
throwaway worktree with this patch: it fails on repository-wide existing lint
issues, and auto-formats 78 unrelated files, **zero files from this patch**.
Those unrelated changes were discarded with the throwaway worktree. No installed
vLLM/torch_npu/model or server/NPU tests ran locally.
