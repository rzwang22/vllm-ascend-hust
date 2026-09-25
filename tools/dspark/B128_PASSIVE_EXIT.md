# B128 completed collection: passive exit correction

The `IF4hUVja` run completed collection but failed exit acceptance. This change
fixes active diagnostic interference; B128/B256 NPU acceptance remains PENDING.
The historical run is not relabeled or published. Core remains
`71d2c1c436eba894a8e9eeb2c5af17e05cb42970`.

## Evidence audited independently

- Producer Plugin: `ebdf9006e55cffdc6d8dc545102dc2f3d67d8d05`.
- Outer SHA256: `f4603a2a0254188cfd0e9172f6920237816b5c2e7946cea2089806dd7d96e817`.
- Embedded SHA256: `00f594d3cacd5c687dca25dccb2f8745e123f3f75072d8127de5462e808c6307`.
- Host 112 passed, no failures/skips; installed real communication precheck passed.
- 48 points, 2258 requests, each output 512 tokens. All raw hashes match retained
  and completion records. All 3840 retained timing samples rebuild from raw data.
- 49 validated transfers including initialization, 392 rank files. Receipt
  nonce/point/rank/size/SHA checks pass. Transfer files precede frontend
  `sample_selection` annotation; rerunning the original selection implementation
  reproduces both retained samples and annotated raw rank payloads exactly.
- Real worker exits `[0,0,0,0,0,-10,0,0]`; all reaped. No TERM/KILL escalation,
  timeout or residuals. Worker cleanup 13.035 s; frontend 16.536 s.
- `worker_cleanup_incomplete`, overall false. No published cost table, no B128
  confidence or B256 stage. SIGBUS did not recur; its original cause remains UNKNOWN.

`B128_IF4hUVja_AUDIT.json` preserves per-point hashes, transfers, sample counts,
signal anchors, stage return codes and PIPESTATUS. Recheck the original bundle
without weights with:

```bash
python -m tools.dspark.audit_passive_exit ARCHIVE.tar.gz audit.json
```

## Proven unsafe call path and remaining limitation

`ProfileNPUWorker` installs `WorkerExitTrace`. Its `monitor_death_pipe` wrapper
called `arm_stacks()` after initialization, publishing a one-time ready record.
`ProfileMultiprocExecutor._shutdown_with_receipt()` enabled `ExitWatch` whenever
worker exit receipts were requested. The background checkpoints called
`snapshot(request_stacks=True)` at half the configured worker grace and during
the TERM wait. `snapshot` trusted the old registration plus a still-live process
handle and called `os.kill(SIGUSR1)`. Disabling gdb never disabled this path.
The escalation callback also calls snapshot; it does not need a stack signal.
Exception cleanup follows the same executor path. No postmortem RPC is involved.

Rank 5 anchors, all UTC on 2026-09-24:

- `WorkerProc.shutdown` returned at 11:27:39.539 (full precision in audit JSON).
- Checkpoint 0 at about 12.5 s after parent cleanup began found PID 659146 alive.
- Its procfs `SigCgt=0000000108000002`, `SigIgn=0000000001001000` lacked bit 9,
  the Linux SIGUSR1 bit. The original ready file still said registered.
- Parent recorded signal 10 sent at 11:27:48.677986. Stack file stayed empty;
  parent subsequently reaped status -10. This supports diagnostic signal death,
  not a new model numerical error or a cleanup timeout.

Plugin/Core contain no explicit faulthandler unregister in this exit wrapper.
CPython 3.12.13 does have an interpreter-finalization path that invokes
`_PyFaulthandler_Fini`, which unregisters user signals and restores prior handlers.
See the versioned primary sources:
[faulthandler.c](https://github.com/python/cpython/blob/v3.12.13/Modules/faulthandler.c#L1279)
and [pylifecycle.c](https://github.com/python/cpython/blob/v3.12.13/Python/pylifecycle.c#L1827).
This is a possible removal path, not a captured rank 5 native call stack.
The exact handler-removal caller and remaining native tail cost stay UNKNOWN.
A second SigCgt/liveness check cannot eliminate the check-to-send race.

## Minimal correction and acceptance

- `WorkerExitTrace` and `ExitWatch` now default to passive operation. Passive
  worker startup writes ready evidence without registering a user signal or
  opening a stack output file. Steps, lifetimes and existing release order stay.
- `ExitWatch.snapshot` gates at the actual sender, including direct callers,
  background checkpoints and escalation callbacks. A passive instance cannot
  be turned active by `request_stacks=True`.
- Formal cost and confidence builders explicitly set stack signals and debugger
  off. Additional config `dspark_profile_stack_signals=true` is accepted only
  with explicit exit observation and worker tracing, without named formal policy
  or confidence acceptance. It is independent of the gdb flag, default false,
  and not enabled by the delivered server task. Active mode can perturb or kill
  a finalizing worker and is not formal acceptance evidence.
- Ready/request/checkpoint and worker-cleanup records carry
  `stack_signals_enabled`; cleanup separately records `debugger_enabled` and
  `diagnostic_signals_sent`. Existing force events remain separate. Expanded
  publication and confidence acceptance require consistent new passive receipts,
  all eight identities, no sent signals and no registered diagnostic handler.
  Old B64 acceptance readers are unchanged.
- Core/model/SWA/capture/timing/file transport are unchanged. No new device sync,
  resource clearing, delays or larger timeout. Core grace 25 s, TERM 4 s, shared
  reap 1 s, engine 36 s, outer frontend 40 s and supervisor 48 s remain intact.

## Validation and next task

The no-weight regression uses owned Python subprocesses with real faulthandler
registration, then explicit revocation in `atexit` while the process remains
alive. The active control exercises the same sender and reproduces signal death.
Passive direct/checkpoint/escalation paths preserve exit 0 or 7. Actual frozen
Core escalation code sends TERM/KILL to deliberately stuck owned children;
nonzero exit, forced cleanup and outer timeout still fail, and late completion
cannot rewrite the timeout. No sender is mocked. Platform imports/model setup
are isolated; this is CPU lifecycle validation, not NPU validation or a claim
that explicit test revocation is the server's actual revocation caller.

The existing host preflight now includes this regression before any weights.
Use the single command in `B128_B256_CONFIDENCE.md` with the delivery SHA, retaining
the activated CANN/custom OPP environment, Core remote `rzwang`, frozen input
manifest and earlier evidence. No additional model diagnostic matrix is added.

Old complete timing samples remain available for offline audit, but
`formal_cost.publish` requires raw return code 0 and natural exit from that same
run. The failed process cannot be resumed after exit, and a different process's
exit cannot validate these samples. Thus the minimal permitted NPU continuation
is a fresh original B128 cost run, then B128 confidence, B256 cost and B256
confidence, each stage only after the previous passes. No B64 rerun or performance
comparison. The existing 48/56-point plans, per-stage budgets, at most four model
initializations and 36000 s total runtime plus existing 65 s cleanup margin remain.

Return one new `dspark-batch-expansion.*-evidence.tar.gz` and SHA256. It includes
preload JUnit and communication checks, raw samples and file receipts, passive
signal settings/actual sends, force events, all worker exit codes/reap, frontend
cleanup, first error, stage logs/PIPESTATUS and publication/acceptance reports.
NPU acceptance requires all eight zero exits, no escalation/timeout/residual,
strict logs, complete samples/coverage and actual confidence FULL concurrency.
Local tests cannot close that pending acceptance.

Local delivery validation: Python 3.12.13, **218 CPU/mock tests passed**, zero
skips, including the real subprocess controls and existing file-transfer,
publication, confidence, budget and first-error regressions. No NPU run was
performed. Modified-file manual hooks pass. Required `bash format.sh ci` was
executed in an isolated checkout; whole-repository checks remain blocked by
pre-existing unrelated lint/format issues and missing local shellcheck. No
unrelated formatter edits are included.
