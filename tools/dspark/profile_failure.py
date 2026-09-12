# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only failure supervision for the isolated profiling frontend."""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

POLL_SECONDS = 0.1
RPC_TIMEOUT_SECONDS = 120
CANCEL_TIMEOUT_SECONDS = 1
CLEANUP_TIMEOUT_SECONDS = 12
CLEANUP_FINALIZE_SECONDS = 4
SUPERVISOR_MARGIN_SECONDS = 6
# Includes cancellation of a failed operation, cleanup, loop task cancellation
# and receipt/dispatch margin. Core's worker escalation alone allows 5+4s.
FAILURE_GRACE_SECONDS = (
    2 * CANCEL_TIMEOUT_SECONDS + CLEANUP_TIMEOUT_SECONDS + CLEANUP_FINALIZE_SECONDS + SUPERVISOR_MARGIN_SECONDS
)


class ProfileEngineFailed(RuntimeError):
    pass


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False))
    temporary.replace(path)


class ProfileFailureGuard:
    def __init__(self, engine, directory, *, require_worker_receipt=False):
        self.engine = engine
        self.require_worker_receipt = require_worker_receipt
        self.phase = "runtime"
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.point = None
        self.first = None
        self.pending = None
        self.cleanup_result = None

    def remember(self, error):
        if self.first is None:
            self.first = {
                "schema_version": 1,
                "phase": self.phase,
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

    def save_cleanup(self, result):
        self.cleanup_result = result
        try:
            write_json(self.directory / "cleanup.json", result)
        except OSError as error:
            result["recording_error"] = f"{type(error).__name__}: {error}"
            result["success"] = False
            self.remember(error)

    async def shutdown(self):
        # Keep process supervision/data tools importable without installed
        # vLLM. Only the engine-owning profile frontend needs the plugin here.
        from vllm_ascend.diagnostics.dspark_cleanup import PROCESS_FORCE_MESSAGES, ShutdownForceObserver

        if self.cleanup_result is not None:
            return self.cleanup_result  # do not start a second destructor thread
        self.phase = "cleanup"
        started = time.monotonic()
        outer_budget = CLEANUP_TIMEOUT_SECONDS + CLEANUP_FINALIZE_SECONDS
        deadline = started + outer_budget
        state = {
            "schema_version": 2,
            "performance_eligible": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": CLEANUP_TIMEOUT_SECONDS,
            "finalize_timeout_seconds": CLEANUP_FINALIZE_SECONDS,
            "outer_timeout_seconds": outer_budget,
            "supervisor_failure_grace_seconds": FAILURE_GRACE_SECONDS,
            "status": "running",
            "shutdown_completed": False,
            "engine_returned": False,
            "thread_completed": False,
            "timed_out": False,
            "success": False,
            "force_events": [],
            "forced_cleanup": None,
            "force_observation_scope": "frontend Core process-manager warning on cleanup thread; not descendant logs",
            "graceful_worker_exit": "unverified",
            "event_loop": "running",
            "error": None,
            "recording_error": None,
            "returned_utc": None,
            "engine_started_utc": None,
            "engine_finished_utc": None,
            "engine_elapsed_seconds": None,
        }
        self.save_cleanup(dict(state))  # visible start receipt, even if the library never returns
        state["recording_error"] = self.cleanup_result["recording_error"]
        lock = threading.Lock()

        def publish_force(events):
            with lock:
                state["force_events"] = events
                state["forced_cleanup"] = True

        def close():
            logger = logging.getLogger("vllm.v1.utils")
            observer = ShutdownForceObserver(PROCESS_FORCE_MESSAGES, publish_force)
            available = logger.isEnabledFor(logging.WARNING) and not logger.filters
            logger.addHandler(observer)
            call_started = time.monotonic()
            try:
                with lock:
                    state["engine_started_utc"] = datetime.now(timezone.utc).isoformat()
                    state["forced_cleanup"] = False if available else None
                self.engine.shutdown(timeout=CLEANUP_TIMEOUT_SECONDS)
                with lock:
                    state["engine_returned"] = True
                    state["returned_utc"] = datetime.now(timezone.utc).isoformat()
            except BaseException as error:
                with lock:
                    state["error"] = f"{type(error).__name__}: {error}"
            finally:
                logger.removeHandler(observer)
                finished = time.monotonic()
                with lock:
                    state.update(
                        engine_elapsed_seconds=finished - call_started,
                        engine_finished_utc=datetime.now(timezone.utc).isoformat(),
                        returned_after_outer_deadline=finished > deadline,
                        force_events=list(observer.events),
                        forced_cleanup=True if observer.events else False if available else None,
                    )
                    terminal = {
                        key: state[key]
                        for key in (
                            "schema_version",
                            "performance_eligible",
                            "engine_started_utc",
                            "engine_returned",
                            "returned_utc",
                            "engine_finished_utc",
                            "engine_elapsed_seconds",
                            "error",
                            "returned_after_outer_deadline",
                            "force_events",
                            "forced_cleanup",
                            "force_observation_scope",
                        )
                    }
                # A late return has its own receipt; it never rewrites a latched
                # outer timeout or the first generation/cleanup failure.
                try:
                    write_json(self.directory / "cleanup-thread.json", terminal)
                except OSError as error:
                    with lock:
                        state["recording_error"] = f"{type(error).__name__}: {error}"

        # A stuck library destructor must not hold this event loop forever.
        # The outer owned-process supervisor enforces the final process bound.
        thread = threading.Thread(target=close, daemon=True, name="ProfileEngineCleanup")
        thread.start()
        while thread.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(min(POLL_SECONDS, max(0, deadline - time.monotonic())))
        # One coherent snapshot. When the thread has ended, all state and its
        # terminal receipt are published. Otherwise it is unsafe to close the
        # event loop, even if engine.shutdown is concurrently returning.
        with lock:
            completed = not thread.is_alive()
            result = dict(state)
        result["thread_completed"] = completed
        result["elapsed_seconds"] = time.monotonic() - started
        result["observed_utc"] = datetime.now(timezone.utc).isoformat()
        result["timed_out"] = not completed or result.get("returned_after_outer_deadline", False)
        result["shutdown_completed"] = completed and result["engine_returned"] and not result["error"]
        workers_ok = not self.require_worker_receipt
        if self.require_worker_receipt:
            try:
                worker = json.loads((self.directory / "worker-cleanup.json").read_text())
            except (OSError, ValueError) as error:
                worker = {"status": "unavailable", "error": str(error)}
            result["worker_cleanup"] = worker
            if worker.get("forced_cleanup"):
                result["forced_cleanup"] = True
            workers_ok = (
                worker.get("status") == "returned"
                and worker.get("forced_cleanup") is False
                and not worker.get("error")
                and not worker.get("recording_error")
                and bool(worker.get("workers"))
                and all(w.get("raw_exitcode") == 0 for w in worker["workers"])
            )
            result["graceful_worker_exit"] = workers_ok
        result["status"] = (
            "outer_timeout"
            if result["timed_out"]
            else "error"
            if result["error"]
            else "forced_cleanup"
            if result["forced_cleanup"]
            else "force_observation_unavailable"
            if result["forced_cleanup"] is None
            else "worker_cleanup_incomplete"
            if not workers_ok
            else "returned"
        )
        result["success"] = (
            result["shutdown_completed"]
            and not result["timed_out"]
            and result["forced_cleanup"] is False
            and result["recording_error"] is None
            and workers_ok
        )
        if not result["success"]:
            self.remember(
                RuntimeError(f"Profile cleanup {result['status']}: {result['error'] or 'see cleanup receipt'}")
            )
        self.save_cleanup(result)
        return result
