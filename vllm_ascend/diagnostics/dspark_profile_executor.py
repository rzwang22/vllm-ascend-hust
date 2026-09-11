# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile-only exit receipts from the existing MultiprocWorkerMonitor.

No alternate worker monitor, scheduler or execution implementation. The parent
calls our shutdown override after setting is_failed, before sending cleanup
signals. Only that parent's multiprocessing handles provide raw exit codes.
"""

import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path

from vllm.v1.executor.multiproc_executor import MultiprocExecutor


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
        return super().shutdown()
