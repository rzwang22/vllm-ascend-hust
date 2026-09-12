# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile-only exit receipts from the existing MultiprocWorkerMonitor.

No alternate worker monitor, scheduler or execution implementation. The parent
calls our shutdown override after setting is_failed, before sending cleanup
signals. Only that parent's multiprocessing handles provide raw exit codes.
"""

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from vllm_ascend.diagnostics.dspark_cleanup import EXECUTOR_FORCE_MESSAGES, ShutdownForceObserver


def process_status(handle):
    proc = handle.proc
    code = proc.exitcode
    sig = -code if code is not None and code < 0 else None
    try:
        name = signal.Signals(sig).name if sig is not None else None
    except ValueError:
        name = "unknown_signal_number"
    return {
        "pid": proc.pid,
        "rank": handle.rank,
        "name": proc.name,
        "raw_exitcode": code,
        "signal": sig,
        "signal_name": name,
        "exit_status": "available" if code is not None else "unavailable: not reaped/exited",
    }


class ProfileMultiprocExecutor(MultiprocExecutor):
    def __init__(self, vllm_config, monitor_workers=True):
        options = vllm_config.additional_config.get("dspark_confidence_verification", {})
        if not options.get("profile") or options.get("mode") != "specified_lengths":
            raise ValueError("Exit receipts require an isolated specified-length profile")
        self._profile_directory = Path(vllm_config.additional_config["dspark_profile_failure_dir"])
        self._profile_directory.mkdir(parents=True, exist_ok=True)
        self._profile_point = None
        self._profile_operation = None
        self._profile_cleanup_observed = False
        self._profile_worker_exit = vllm_config.additional_config.get("dspark_profile_worker_exit", False)
        super().__init__(vllm_config, monitor_workers=monitor_workers)

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        unique_reply_rank=None,
        kv_output_aggregator=None,
    ):
        if method == "dspark_benchmark_profile_point":
            self._profile_point = (kwargs or {}).get("point")
        operation = {
            "method": method if isinstance(method, str) else "callable",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "point": self._profile_point,
            "non_block": non_block,
        }
        self._profile_operation = operation
        try:
            return super().collective_rpc(
                method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                non_block=non_block,
                unique_reply_rank=unique_reply_rank,
                kv_output_aggregator=kv_output_aggregator,
            )
        finally:
            # For async execute/sample dispatch this is dispatch completion,
            # not device completion. Profile point utilities are blocking.
            if self._profile_operation is operation:
                self._profile_operation = None

    def shutdown(self):
        if getattr(self, "is_failed", False) and not getattr(self, "shutting_down", False):
            try:
                data = {
                    "schema_version": 1,
                    "performance_eligible": False,
                    "source": "MultiprocWorkerMonitor -> shutdown; before cleanup signals",
                    "observed_utc": datetime.now(timezone.utc).isoformat(),
                    "parent_pid": os.getpid(),
                    "point": self._profile_point,
                    "pending_operation": self._profile_operation,
                    "workers": [process_status(h) for h in getattr(self, "workers", [])],
                    "cause": "unavailable; exit code/signal does not establish why the worker exited",
                }
                with (self._profile_directory / "worker-exit.json").open("x") as stream:
                    json.dump(data, stream)
            except FileExistsError:
                pass  # First pre-cleanup receipt wins, including concurrent teardown.
            except Exception as error:
                # Never prevent the original executor's failure callback/cleanup.
                print(f"PROFILE_EXIT_RECEIPT_UNAVAILABLE: {type(error).__name__}: {error}", flush=True)
        if self._profile_cleanup_observed:
            return super().shutdown()
        self._profile_cleanup_observed = True
        return self._shutdown_with_receipt()

    def _shutdown_with_receipt(self):
        # Observe the original executor shutdown; do not duplicate its wait,
        # termination, queue cleanup or worker-monitor implementation.
        started = time.monotonic()
        logger = logging.getLogger("vllm.v1.executor.multiproc_executor")
        workers = tuple(getattr(self, "workers", ()))
        watch = None
        if self._profile_worker_exit:
            # Isolation-process import; default profile modes do not load this
            # observer or create a watcher thread / send diagnostic signals.
            from vllm import envs

            from vllm_ascend.diagnostics.dspark_worker_exit import ExitWatch

            try:
                watch = ExitWatch(
                    self._profile_directory / "worker-exit",
                    workers,
                    self._profile_point,
                    envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS,
                )
                watch.thread.start()
            except Exception as error:
                print(f"PROFILE_EXIT_WATCH_UNAVAILABLE: {type(error).__name__}: {error}", flush=True)
        state = {
            "performance_eligible": False,
            "parent_pid": os.getpid(),
            "point": self._profile_point,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "force_events": [],
            "forced_cleanup": False if logger.isEnabledFor(logging.WARNING) and not logger.filters else None,
            "error": None,
            "recording_error": None,
        }

        def save():
            try:
                path = self._profile_directory / "worker-cleanup.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(state))
                temporary.replace(path)
            except Exception as error:
                state["recording_error"] = f"{type(error).__name__}: {error}"
                print(f"PROFILE_CLEANUP_RECEIPT_UNAVAILABLE: {state['recording_error']}", flush=True)

        def publish(events):
            state.update(force_events=events, forced_cleanup=True)
            save()  # retain escalation even if EngineCore is killed before return
            if watch is not None:
                try:
                    watch.snapshot(f"before-escalation-{len(events)}")
                except Exception as error:
                    state["exit_watch_error"] = f"{type(error).__name__}: {error}"

        observer = ShutdownForceObserver(EXECUTOR_FORCE_MESSAGES, publish)
        save()
        logger.addHandler(observer)
        try:
            result = super().shutdown()
            state["status"] = "returned"
            return result
        except BaseException as error:
            state.update(status="error", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if watch is not None:
                watch.close()
            logger.removeHandler(observer)
            state.update(
                elapsed_seconds=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat()
            )
            try:
                state["workers"] = [process_status(h) for h in workers]
            except Exception as error:
                state["worker_status_error"] = str(error)
            save()
