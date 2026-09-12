# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound failed metadata experiments, including stuck native cleanup/destructors."""

import argparse
import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

from tools.dspark.profile_failure import write_json

FAILURE_GRACE_SECONDS = 20
TERM_GRACE_SECONDS = 5
POLL_SECONDS = 0.1


def read_failure(directory):
    for name in ("worker-exit.json", "engine-failure.json", "profile-failure.json"):
        try:
            return {"source": name, "receipt": json.loads((directory / name).read_text())}
        except (OSError, ValueError):
            continue
    return None


def group_exists(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_owned_group(pid, sig):
    # Popen below creates this dedicated session. Never enumerate/kill jobs
    # by executable name, device allocation, or another caller's PID.
    with suppress(ProcessLookupError):
        os.killpg(pid, sig)


def supervise(
    command,
    directory,
    receipt_path,
    *,
    grace=FAILURE_GRACE_SECONDS,
    term_grace=TERM_GRACE_SECONDS,
    max_runtime=None,
    stop_file=None,
):
    if max_runtime is not None and max_runtime <= 0:
        raise ValueError("Profile runtime bound must be positive")
    child = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    receipt = {
        "schema_version": 1,
        "performance_eligible": False,
        "child_pid": child.pid,
        "owned_process_group": child.pid,
        "command": command,
        "first_failure": None,
        "signals_sent": [],
        "raw_returncode": None,
        "failure_grace_seconds": grace,
        "term_grace_seconds": term_grace,
        "max_runtime_seconds": max_runtime,
        "stop_file": str(stop_file) if stop_file is not None else None,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    failure_at = None
    try:
        write_json(receipt_path, receipt)
        while True:
            rc = child.poll()
            failure = read_failure(directory)
            if failure is None and rc is None:
                if stop_file is not None and stop_file.exists():
                    failure = {"source": "controlled stop file", "path": str(stop_file)}
                elif max_runtime is not None and time.monotonic() - started >= max_runtime:
                    failure = {"source": "diagnostic runtime bound", "seconds": max_runtime}
            if failure is not None and receipt["first_failure"] is None:
                receipt["first_failure"] = failure
                receipt["observed_utc"] = datetime.now(timezone.utc).isoformat()
                failure_at = time.monotonic()
                write_json(receipt_path, receipt)
            if rc is not None:
                if rc and receipt["first_failure"] is None:
                    receipt["first_failure"] = {
                        "source": "owned child exit",
                        "raw_returncode": rc,
                        "worker_exit": "unavailable; no parent receipt",
                    }
                break
            if failure_at is not None and time.monotonic() - failure_at >= grace:
                break
            time.sleep(POLL_SECONDS)
    except BaseException as error:
        if receipt["first_failure"] is None:
            receipt["first_failure"] = read_failure(directory) or {
                "source": "supervisor",
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            receipt["secondary_error"] = f"{type(error).__name__}: {error}"
    finally:
        # Even a frontend that has exited can leave descendants holding tee's
        # output pipe. Reap/terminate only this experiment's owned session.
        if child.poll() is not None and receipt["first_failure"] is None:
            deadline = time.monotonic() + term_grace
            while group_exists(child.pid) and time.monotonic() < deadline:
                time.sleep(POLL_SECONDS)
        if group_exists(child.pid):
            if receipt["first_failure"] is None:
                receipt["first_failure"] = {"source": "owned descendants survived frontend exit"}
            receipt["signals_sent"].append("SIGTERM")
            signal_owned_group(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + term_grace
            while group_exists(child.pid) and time.monotonic() < deadline:
                child.poll()
                time.sleep(POLL_SECONDS)
            if group_exists(child.pid):
                receipt["signals_sent"].append("SIGKILL")
                signal_owned_group(child.pid, signal.SIGKILL)
        try:
            receipt["raw_returncode"] = child.wait(timeout=term_grace)
        except subprocess.TimeoutExpired:
            receipt["reap_status"] = "unavailable: owned child did not reap within bound"
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(receipt_path, receipt)
    return 1 if receipt["first_failure"] or receipt["raw_returncode"] != 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Child command is required")
    return supervise(
        command, args.directory, args.receipt, max_runtime=args.max_runtime_seconds, stop_file=args.stop_file
    )


if __name__ == "__main__":
    raise SystemExit(main())
