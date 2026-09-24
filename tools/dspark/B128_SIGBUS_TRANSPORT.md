# B128 SIGBUS: audited partial run and bounded snapshot transport

This is a profile-evidence transport correction. The original worker SIGBUS
cause is **UNKNOWN**; full NPU acceptance is **PENDING**. No model, Core, custom
op, confidence algorithm, SWA policy, capture shape or exit budget is changed.
B64 remains frozen and is not rerun.

## Independently checked evidence

Producer Plugin `3d242eb3a80f942febb64acb52c40051e13bb492`, Core
`71d2c1c436eba894a8e9eeb2c5af17e05cb42970`:

- Outer `dspark-batch-expansion.qsQcxCRh-evidence.tar.gz` SHA256:
  `ffe56dd4d67c6d1f45b1e3f6a7f2256bd37f078d9b42cdeb1f95e8043df9110f`.
- Embedded `dspark-large-batch.3eW01ldi` archive SHA256:
  `b0bdc8737e9a01d4d05a300e2e6ad47d2a0f34aaecf3da46fd1911018c8b6b8a`.
- Installed capacity-interface precheck passed; host 92 passed, no skipped cases.
  All eight capacity receipts report 128 allocated request rows. The previous
  nonexistent `max_num_reqs` access did not recur.
- All 35 retained point hashes match both raw JSON and point-completion entries.
  The auditor rebuilds retained event samples from raw rank records. Every
  cost event belongs to its current point. These remain **partial evidence**.
- Point 36, `ctx128-n96-t96-skewed`: 96 requests each produced 512 tokens,
  streaming error is null. Its complete eight-rank snapshot is unavailable.
  The pending RPC started at `2026-09-24T07:06:48.404057+00:00`.
- The monitor first observed worker 6 dead at display-log time 07:06:49.
  This does not identify the originating rank, thread, native instruction or
  backing mapping. Final raw exit codes are `[0,-7,-7,-7,-7,-7,-7,-7]`.
  `cancelled`, EngineDeadError and compiler-parent-disappeared messages follow.
- Cleanup is `worker_cleanup_incomplete`, without force events or timeout.
  “All workers exited gracefully” only means Core's wait returned, not that
  their exit codes were zero. Overall remains failed. No new cost table was
  produced; B128 confidence and both B256 stages did not start.

`B128_SIGBUS_AUDIT.json` stores evidence hashes, all 35 point/sample checks,
capacity summaries, first error, PIPESTATUS and separate cleanup status.
Reproduce the read-only archive audit without weights:

```bash
python -m tools.dspark.audit_sigbus_transport ARCHIVE.tar.gz audit.json
```

## Actual transport and memory lifetime

The worker extension builds its replay/verification counters and
`IsolatedCostProfiler.snapshot()` measurements. Frozen Core
`WorkerProc.handle_output()` hands the result to the async output thread;
`enqueue_output()` wraps `(ResponseStatus.SUCCESS, result)` and calls
`MessageQueue.enqueue()` in `distributed/device_communicators/shm_broadcast.py`.
It uses pickle protocol 5 (highest in Python 3.12), with its real out-of-band
buffer callback. The plain JSON-compatible snapshots have no OOB buffers.

A local response queue defaults to **24 MiB × 10 slots per rank**, not a small
4 MiB response limit. Serialization below its threshold is copied into SHM;
only overflow uses ZMQ. Consuming marks a slot reusable but does not clear or
decommit its pages. Eight response rings reserve about 1920 MiB of address
space, plus the command ring. Reservation is not proof of committed tmpfs use.
Successive large replies can touch previously untouched slots even when Python
objects are released. The Core send-side output thread also retains its most
recent local value until its next iteration. Neither fact proves an unbounded
Python-object leak.

`MultiprocExecutor.collective_rpc()` consumes all response queues and clears
its pending future. Core then wraps the list in `UtilityResult` and sends it
through `MsgpackEncoder.encode_into()` and tracked ZMQ buffers to the frontend.
Completed socket buffers are reclaimed/reused. Thus this second leg is Msgpack,
not the worker SHM pickle representation. The new installed precheck also
measures/reconstructs this utility envelope with insecure serialization off.

`IsolatedCostProfiler.begin_point()` synchronizes at the existing point boundary
and clears timing events. The 35 raw snapshots confirm no cross-point cost-event
accumulation. Replay layout histograms are cumulative by design and remain
unchanged. Frontend snapshots/raw objects and the worker's last response can
briefly overlap; the evidence does not measure crash-time heap or tmpfs peaks.

### Measured no-weight communication

The preceding balanced point is 36,813,028 bytes as saved JSON. A reconstructed
rank snapshot contains 1026 events. The **actual Core enqueue serializer**,
including its SUCCESS wrapper, produces 3,448,341 pickle bytes per rank and
3,448,348 inline framing bytes. This is below the production 24 MiB threshold.
These are measured retransmission sizes of JSON-reconstructed objects; original
Python aliases and the failed round's wire bytes were not captured. They are
not presented as crash-time measurements.

The local eight-spawn-process test uses the frozen SHM/ZMQ queue,
`WorkerProc.enqueue_output`, FutureWrapper and `collective_rpc` bodies. Only
unavailable platform imports are adapted on this Mac; queue/serializer bodies
are unchanged. It repeats the archived snapshots three times, tests the exact
threshold minus one/equal/plus one, then reconstructs full evidence from file
receipts. Fifteen collectives return 120 rank replies. All values compare equal,
eight exit codes are zero and no owned SHM names remain. Consumed slots retaining
bytes is directly checked. New receipts are **325 pickle bytes / 332 framed
bytes per rank** for this saved point; original data are retained in files.

The safe test rings are **1 MiB × 2 slots**, with less than 17 MiB total reserved
SHM. Consequently the saved bulk snapshot takes the test overflow path, while
it would take the default production inline path. Both paths and repeated slot
reuse are exercised; production-size rings are not allocated. On Linux the
precheck requires twice its reservation plus 16 MiB of free `/dev/shm` headroom
before allocating anything. Insufficient headroom fails before weights. No
public tmpfs exhaustion/truncation experiment is performed. This test did **not**
reproduce SIGBUS. `B128_SIGBUS_LOCAL_TRANSPORT.json` records its measured scope.

Local Python 3.12.13 / Torch 2.10.0 CPU regressions passed **190 tests**, zero
skips, with `pytest --noconftest` (the same host-test isolation used by the server
entry). They cover profile point lifecycle, lossless files, stale/missing/corrupt
receipts, bounded writes, preservation of first errors, real eight-process
transport, publication gates and unchanged shutdown policy. The final added
preflight-error persistence assertion also passed its targeted test. No installed
Ascend, server utility Msgpack or NPU acceptance is claimed locally. The installed
transport precheck and the complete B128/B256 route remain **PENDING**.

Required `bash format.sh ci` ran in a disposable checkout and failed on existing
unrelated repository findings (including Ruff errors, markdown and forbidden
imports) and unavailable local shellcheck. Its automatic formatting touched no
delivery file; unrelated changes were discarded. Delivery-file manual hooks
are checked separately.

Missing decisive evidence: faulting native frame/address, its mapping/backing
object, fault-time tmpfs/cgroup state, and kernel/CANN crash records. Resource
pressure, truncation and device/native faults remain hypotheses. Snapshot size
alone cannot prove `/dev/shm` exhaustion or exclude a device-side cause.

## Minimal profile correction

Only expanded formal-cost collection uses the new
`dspark_benchmark_profile_snapshot_file` RPC. Existing non-profile and B64
paths retain their defaults. The worker still constructs the same snapshot
at the same quiescent boundary, with the existing profile synchronization.
It streams every original field to a rank-specific file, flushes/fsyncs and
atomically publishes it, then returns a small receipt. No full event history
enters the worker response queue or the EngineCore utility response.

Each invocation has a fresh transfer ID, expected point and rank set. Workers
check the active profiler point and fixed configured output directory. Frontend
checks the transfer, point, all ranks, exact filename, byte count and SHA256,
then reconstructs the unchanged raw point format before existing request,
sampling and publication checks. Missing, duplicate, stale, truncated, corrupt,
symlinked or oversized data fail; no fallback to bulk RPC or partial acceptance.
Files are limited to 64 MiB per rank and successful receipts to 4096 JSON bytes.
Worker building/committed/failed and frontend requested/validated/failed states
remain available on failure; incomplete `.partial` files are never accepted.

This adds disk I/O, hashing and fsync **after generation, outside timed target
and draft calls**. It preserves all raw events instead of selecting fewer rows.
The parent still reconstructs all eight rank objects, so host heap cost is not
eliminated. Raw point files and rank transport files coexist, increasing disk
usage. Inter-point I/O can alter wall-clock timing/cache conditions; this is not
an end-to-end performance claim. No new NPU D2H, layer probe or global device
synchronization is added. Timing events and their reset/lifetimes are unchanged.
The per-rank file ceiling and existing finite point plans bound evidence size.

## Server task and cheap gates

Use the same single command in `B128_B256_CONFIDENCE.md` with the delivery SHA.
It preserves the activated CANN/custom OPP environment, exact Core SHA and
`rzwang` source selection. Before loading weights the existing 600-second host
stage now performs:

1. Installed Ascend capacity-interface precheck.
2. Bounded read-only host capture in `transport-preflight/system-evidence.json`:
   `/dev/shm` mount/current capacity/names, cgroup memory files, process summary,
   kernel journal/dmesg and available coredump listings for the UTC incident
   window; limited existing CANN log tails. Timezone, errors, permissions and
   truncation are retained. Current free space cannot establish past exhaustion.
3. Exact outer/embedded archive SHA checks and the eight-worker **installed**
   Core communication/reconstruction test on the saved balanced snapshot.
   The queue test has a 120-second total deadline and bounded shared child
   reap. It only signals its own test children on failure. No model or NPU kernel
   is loaded/executed by this check; installed NPU libraries may be imported.
4. Host regressions; any failure or skip stops before weights.

If only read-only system evidence is wanted, this command performs no model
loading, queue allocation, ptrace, signal, deletion or restart:

```bash
python -m tools.dspark.sigbus_system_evidence /workspace/dspark-results/sigbus-system-evidence.json
```

The main entry already includes it; do not run it a second time for this task.
Per-command host reads have four-second timeouts; log traversal is bounded.
Unavailable historical logs remain unavailable, never an implicit success.

After cheap gates pass, continue unchanged: **B128 costs → B128 confidence →
B256 costs → B256 confidence**. Fresh engines/costs are necessary because the
old engine died and its 35-point partial table cannot be published or relabeled
with a new producer SHA. Keep the original 48/56 points, 128/256 actual-concurrency
FULL witnesses, numerical/owner checks, strict logs and `dspark-profile-25s-v1`.
At most four model initializations; whole-task limit 36000 s plus the existing
65 s cleanup margin. Cost/consumer timeouts and exit budgets are not enlarged.
No retry, B64 rerun, performance comparison or new matrix is introduced.

Return the new `dspark-batch-expansion.*-evidence.tar.gz` and SHA256. It embeds
`transport-preflight/` input hashes, actual wire sizes, file receipts, host
resource/log evidence, process exits and test JUnit; each cost run additionally
contains `snapshot-transfers/`, original reconstructed point JSON, partial or
complete retained records, first failure, real worker exit codes, capacity and
cleanup receipts, and all stage logs/PIPESTATUS. Publication remains blocked
unless the full plan and cleanup gates pass. SIGBUS root cause stays UNKNOWN
until direct evidence resolves it; a successful later run is reported as such.
