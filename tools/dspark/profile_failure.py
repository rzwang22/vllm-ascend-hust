# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only failure supervision for the isolated profiling frontend."""

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

POLL_SECONDS = 0.1
RPC_TIMEOUT_SECONDS = 120
CANCEL_TIMEOUT_SECONDS = 1
CLEANUP_TIMEOUT_SECONDS = 8


class ProfileEngineFailed(RuntimeError):
    pass


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False))
    temporary.replace(path)


class ProfileFailureGuard:
    def __init__(self, engine, directory):
        self.engine = engine
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.point = None
        self.first = None
        self.pending = None

    def remember(self, error):
        if self.first is None:
            self.first = {
                "schema_version": 1,
                "performance_eligible": False,
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "point": self.point,
                "pending_operation": self.pending,
                "error": f"{type(error).__name__}: {error}",
                "observed_frontend_error": f"{type(error).__name__}: {error}",
            }
            path = self.directory / "worker-exit.json"
            if path.exists():
                try:
                    self.first["worker_exit"] = json.loads(path.read_text())
                    self.first["error"] = "Worker process exited; see original parent exit receipt"
                except (OSError, ValueError):
                    self.first["worker_exit"] = "unavailable: incomplete parent receipt"
            try:
                write_json(self.directory / "engine-failure.json", self.first)
            except OSError as write_error:
                self.first["evidence_write_error"] = str(write_error)
                print(f"PROFILE_FAILURE_EVIDENCE_UNAVAILABLE: {self.first}", flush=True)
        return self.first

    async def run(self, operation, awaitable, timeout=None):
        self.pending = {"name": operation, "point": self.point, "started_utc": datetime.now(timezone.utc).isoformat()}
        write_json(self.directory / "frontend-operation.json", {**self.pending, "state": "pending"})
        task = asyncio.ensure_future(awaitable)
        started = time.monotonic()
        try:
            while True:
                # The frozen output socket task forwards errors to generation
                # but does not complete pending utility futures. Observe its
                # existing engine state; never send a health RPC per step.
                if self.first is not None or self.engine.errored:
                    raise ProfileEngineFailed("AsyncLLM/EngineCore is dead while awaiting " + operation)
                done, _ = await asyncio.wait({task}, timeout=POLL_SECONDS)
                if done:
                    return task.result()
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Profile {operation} exceeded {timeout}s")
        except BaseException as error:
            self.remember(error)
            task.cancel()
            await asyncio.wait({task}, timeout=CANCEL_TIMEOUT_SECONDS)
            if task.done() and not task.cancelled():
                task.exception()  # retrieve a late exception without replacing the first
            raise ProfileEngineFailed(self.first["error"]) from error
        finally:
            try:
                write_json(
                    self.directory / "frontend-operation.json",
                    {**self.pending, "state": "failed" if self.first else "completed"},
                )
            except OSError:
                if self.first is None:
                    raise
            finally:
                self.pending = None

    async def shutdown(self):
        state = {
            "performance_eligible": False,
            "timeout_seconds": CLEANUP_TIMEOUT_SECONDS,
            "shutdown_completed": False,
            "error": None,
        }
        done = threading.Event()

        def close():
            try:
                self.engine.shutdown(timeout=CLEANUP_TIMEOUT_SECONDS)
                state["shutdown_completed"] = True
            except BaseException as error:
                state["error"] = f"{type(error).__name__}: {error}"
            finally:
                done.set()

        # A stuck library destructor must not hold this event loop forever.
        # The outer owned-process supervisor enforces the final process bound.
        thread = threading.Thread(target=close, daemon=True, name="ProfileEngineCleanup")
        thread.start()
        deadline = time.monotonic() + CLEANUP_TIMEOUT_SECONDS
        while not done.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(POLL_SECONDS)
        result = dict(state)
        result["timed_out"] = not done.is_set()
        if result["timed_out"] or result["error"]:
            self.remember(RuntimeError("Profile cleanup incomplete"))
        write_json(self.directory / "cleanup.json", result)
        return result
