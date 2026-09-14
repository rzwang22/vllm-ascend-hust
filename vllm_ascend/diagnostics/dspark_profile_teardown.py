# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile-only host teardown; no device operations or replacement exit policy."""

import time
from datetime import datetime, timezone

WORKER_REAP_SECONDS = 1.0


def shutdown_profile_observers(runner, shutdown):
    """Detach in reverse installation order, then always run original shutdown.

    Keep closed observers (and timing events) on the runner; release only their
    host wrapper references. This does not clear graph/cache storage early.
    A teardown error is an error even if Core cleanup subsequently succeeds.
    """
    errors = []
    for name in ("_dspark_benchmark_replay_observer", "_dspark_cost_profiler"):
        observer = getattr(runner, name, None)
        if observer is not None:
            try:
                observer.close()
            except BaseException as error:
                errors.append(error)
    try:
        result = shutdown()
    except BaseException as error:
        for earlier in errors:
            error.add_note(f"Profile observer teardown also failed: {type(earlier).__name__}: {earlier}")
        raise
    if errors:
        for later in errors[1:]:
            errors[0].add_note(f"Additional observer teardown error: {type(later).__name__}: {later}")
        raise errors[0]
    return result


def reap_workers(workers, *, timeout=WORKER_REAP_SECONDS):
    """Parent-owned joins under one total deadline, including after Core kill.

    No signals and no inferred exit codes. join(0) still polls later handles
    after the budget is exhausted. Forced termination remains a failed run.
    """
    started = time.monotonic()
    deadline = started + timeout
    rows = []
    for handle in workers:
        proc = handle.proc
        row = {"rank": handle.rank, "pid": proc.pid, "join_started_utc": datetime.now(timezone.utc).isoformat()}
        try:
            row["exitcode_before_join"] = proc.exitcode
            proc.join(timeout=max(0.0, deadline - time.monotonic()))
            row["raw_exitcode"] = proc.exitcode
            row["status"] = "reaped" if row["raw_exitcode"] is not None else "unavailable"
            if row["status"] == "unavailable":
                row["reason"] = "No exit status after bounded parent join"
        except Exception as error:
            row.update(status="unavailable", reason=f"{type(error).__name__}: {error}")
        row["join_finished_utc"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
    return {"budget_seconds": timeout, "elapsed_seconds": time.monotonic() - started, "workers": rows}
