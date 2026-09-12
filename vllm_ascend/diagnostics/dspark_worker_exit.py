# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host-only worker teardown receipts; no model/RPC/device operations."""

import faulthandler
import json
import os
import signal
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

MAX_EVENTS = 256
MAX_THREADS = 128
MAX_PROC_BYTES = 4096
STACK_SIGNAL = signal.SIGUSR1
WORKER_TERM_WAIT_SECONDS = 4  # frozen Core's second worker wait; not an override


def utc():
    return datetime.now(timezone.utc).isoformat()


def termination_handlers():
    result = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        handler = signal.getsignal(sig)
        code = getattr(handler, "__code__", None)
        result[sig.name] = {
            "python_handler": f"{getattr(handler, '__module__', '')}.{getattr(handler, '__qualname__', str(handler))}",
            "file": code.co_filename if code else None,
            "line": code.co_firstlineno if code else None,
        }
    return result  # Python's view; procfs SigCgt/SigIgn additionally cover native state


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False))
    tmp.replace(path)


class WorkerExitTrace:
    """One per worker. Wrappers only run at exit (busy-loop wrapper runs once)."""

    def __init__(self, directory, rank, *, arm=True):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.pid = os.getpid()
        self.instance = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.count = 0
        self.request = None
        self.recording_error = None
        self.prefix = f"rank-{rank}-pid-{self.pid}"
        self.fd = os.open(self.directory / f"{self.prefix}-steps.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.stack_file = None
        self.armed = False
        if arm:
            self.arm_stacks()

    def arm_stacks(self):
        if self.armed:
            return
        self.armed = True
        registered = False
        error = None
        try:
            if signal.getsignal(STACK_SIGNAL) != signal.SIG_DFL:
                raise RuntimeError("Stack signal already has a handler; no signal will be sent")
            self.stack_file = (self.directory / f"{self.prefix}-stacks.txt").open("xb", buffering=0)
            faulthandler.register(STACK_SIGNAL, file=self.stack_file, all_threads=True, chain=False)
            registered = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        save(
            self.directory / f"rank-{self.rank}-ready.json",
            {
                "schema_version": 1,
                "performance_eligible": False,
                "rank": self.rank,
                "pid": self.pid,
                "instance": self.instance,
                "signal_registered": registered,
                "signal": int(STACK_SIGNAL),
                "stack_file": f"{self.prefix}-stacks.txt",
                "error": error,
                "started_utc": utc(),
                "termination_handlers": termination_handlers(),
            },
        )

    def record(self, stage, event, **fields):
        # Single unbuffered append per event; no fsync/device wait. An I/O error
        # must not prevent the real shutdown or mask its original exception.
        try:
            with self.lock:
                if self.count >= MAX_EVENTS:
                    return
                if self.request is None:
                    path = self.directory / "request.json"
                    if path.exists():
                        self.request = json.loads(path.read_text())
                self.count += 1
                data = {
                    "seq": self.count,
                    "rank": self.rank,
                    "pid": self.pid,
                    "instance": self.instance,
                    "performance_eligible": False,
                    "request": (self.request or {}).get("id"),
                    "point": (self.request or {}).get("point"),
                    "monotonic_ns": time.monotonic_ns(),
                    "utc": utc(),
                    "thread": threading.current_thread().name,
                    "thread_id": threading.get_ident(),
                    "native_id": threading.get_native_id(),
                    "stage": stage,
                    "event": event,
                    **fields,
                }
                if self.count == MAX_EVENTS:
                    data = {**data, "stage": "coverage", "event": "event_limit_reached"}
                os.write(self.fd, (json.dumps(data, allow_nan=False) + "\n").encode())
        except Exception as exc:
            self.recording_error = f"{type(exc).__name__}: {exc}"
            print(f"PROFILE_WORKER_EXIT_RECORDING_ERROR: {self.recording_error}", flush=True)

    def call(self, stage, function, *args, **kwargs):
        self.record(stage, "begin")
        try:
            result = function(*args, **kwargs)
        except BaseException as exc:
            self.record(stage, "error", error=f"{type(exc).__name__}: {exc}"[:MAX_PROC_BYTES])
            raise
        self.record(stage, "returned")
        return result

    @contextmanager
    def wrapping(self, owner, name, stage):
        """Temporarily observe an actual cleanup callable, preserving binding."""
        original = getattr(owner, name, None)
        if not callable(original):
            self.record(stage, "unavailable")
            yield
            return
        own = name in vars(owner)
        thread = threading.get_ident()

        @wraps(original)
        def observed(*args, **kwargs):
            if threading.get_ident() != thread:
                return original(*args, **kwargs)
            label = stage(*args, **kwargs) if callable(stage) else stage
            return self.call(label, original, *args, **kwargs)

        setattr(owner, name, observed)
        try:
            yield
        finally:
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


class ObservedDeathPipe:
    def __init__(self, pipe, trace):
        self.pipe = pipe
        self.trace = trace

    def recv(self):
        try:
            result = self.pipe.recv()
        except EOFError:
            self.trace.record("death_pipe.recv", "eof", termination_handlers=termination_handlers())
            raise
        except BaseException as exc:
            self.trace.record("death_pipe.recv", "error", error=f"{type(exc).__name__}: {exc}")
            raise
        self.trace.record("death_pipe.recv", "returned_without_eof")
        return result


def process_snapshot(pid, proc_root=Path("/proc"), *, detailed=True):
    """Only procfs for owned PIDs; null/error stays unavailable, never an exit code."""
    root = proc_root / str(pid)

    def read(path):
        try:
            with path.open("rb") as f:
                return {"text": f.read(MAX_PROC_BYTES).decode(errors="replace")}
        except OSError as exc:
            return {"unavailable": f"{type(exc).__name__}: {exc}"}

    result = {"pid": pid, "status": read(root / "status"), "stat": read(root / "stat")}
    if not detailed:
        return result
    try:
        tids = sorted((root / "task").iterdir(), key=lambda p: int(p.name))
        result["threads_truncated"] = len(tids) > MAX_THREADS
        result["threads"] = [
            {
                "tid": int(p.name),
                "status": read(p / "status"),
                "wchan": read(p / "wchan"),
                "kernel_stack": read(p / "stack"),
            }
            for p in tids[:MAX_THREADS]
        ]
    except OSError as exc:
        result["threads_unavailable"] = str(exc)
    return result


class ExitWatch:
    """Two pre-escalation stack requests, only while the original shutdown runs."""

    def __init__(self, directory, workers, point, grace):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workers = tuple(workers)
        self.request = uuid.uuid4().hex
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.checkpoints = (grace / 2, grace + WORKER_TERM_WAIT_SECONDS / 2)
        save(
            self.directory / "request.json",
            {
                "id": self.request,
                "point": point,
                "parent_pid": os.getpid(),
                "started_utc": utc(),
                "performance_eligible": False,
                "checkpoints_seconds": self.checkpoints,
            },
        )
        self.thread = threading.Thread(target=self.run, name="ProfileExitWatch", daemon=True)

    def snapshot(self, label, *, request_stacks=False):
        data = {
            "request": self.request,
            "label": label,
            "utc": utc(),
            "performance_eligible": False,
            "elapsed_seconds": time.monotonic() - self.started,
            "workers": [],
        }
        for handle in self.workers:
            proc = handle.proc
            # The parent owns this process handle. A registration from another
            # PID/rank is not authority to signal, even if its path looks right.
            row = {"rank": handle.rank, "pid": proc.pid, "raw_exitcode": proc.exitcode}
            try:
                ready = json.loads((self.directory / f"rank-{handle.rank}-ready.json").read_text())
                if ready["pid"] != proc.pid or ready["rank"] != handle.rank:
                    raise ValueError("worker registration PID/rank mismatch")
                stack = self.directory / ready["stack_file"]
                row["stack_bytes_available_now"] = stack.stat().st_size
            except (OSError, ValueError, KeyError) as exc:
                row["stack_unavailable"] = str(exc)
                ready = None
            if row["raw_exitcode"] is None:
                row["procfs"] = process_snapshot(proc.pid, detailed=request_stacks)
                if request_stacks:
                    try:
                        if ready is None or not ready["signal_registered"]:
                            raise ValueError("matching live worker stack registration unavailable")
                        os.kill(proc.pid, STACK_SIGNAL)
                        row["stack_request"] = {
                            "signal_sent": int(STACK_SIGNAL),
                            "instance": ready["instance"],
                            "file": ready["stack_file"],
                            "utc": utc(),
                        }
                    except (OSError, ValueError, KeyError) as exc:
                        row["stack_unavailable"] = str(exc)
            data["workers"].append(row)
        save(self.directory / f"parent-{label}.json", data)

    def run(self):
        for index, at in enumerate(self.checkpoints):
            if self.stop.wait(max(0, self.started + at - time.monotonic())):
                return
            try:
                self.snapshot(f"checkpoint-{index}", request_stacks=True)
            except Exception as exc:
                print(f"PROFILE_EXIT_WATCH_UNAVAILABLE: {type(exc).__name__}: {exc}", flush=True)

    def close(self):
        self.stop.set()  # no join or extension of Core's termination budget
