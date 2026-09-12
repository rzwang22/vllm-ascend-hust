# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit profile worker class: exit-only wrappers of the real shutdown path."""

import gc
import inspect
import weakref
from contextlib import ExitStack
from functools import wraps
from pathlib import Path

import torch
from vllm.distributed import parallel_state
from vllm.v1.executor import multiproc_executor

from vllm_ascend.diagnostics.dspark_worker_exit import ObservedDeathPipe, WorkerExitTrace
from vllm_ascend.worker.worker import NPUWorker

CORE_GROUPS = ("_TP", "_DCP", "_PCP", "_PP", "_DP", "_EP", "_EPLB", "_WORLD")


def observe_queue_method(trace, queue, method, stage):
    original = getattr(queue, method)
    # Holding a bound method in queue.shutdown would create a new self-cycle
    # and defer ZMQ/resource destruction. Keep only a weak method reference.
    ref = weakref.WeakMethod(original) if inspect.ismethod(original) else lambda: original

    def observed(*args, **kwargs):
        return trace.call(stage, ref(), *args, **kwargs)

    setattr(queue, method, observed)


def install_proc_exit_trace(trace, worker_proc):
    # Installed once in this dedicated worker process, before worker_main calls
    # monitor_death_pipe/busy_loop. No new busy loop or signal/queue semantics.
    original_monitor = worker_proc.monitor_death_pipe
    original_busy = worker_proc.worker_busy_loop
    original_shutdown = worker_proc.shutdown

    def monitor(proc, pipe, requested):
        trace.arm_stacks()  # after real device/model initialization, before READY
        for role in ("rpc_broadcast_mq", "worker_response_mq"):
            mq = getattr(proc, role, None)
            if mq is None:
                trace.record(f"queue.{role}", "absent")
                continue
            observe_queue_method(trace, mq, "shutdown", f"queue.{role}.shutdown")
            spin = getattr(mq, "_spin_condition", None)
            if spin is not None:
                observe_queue_method(trace, spin, "cancel", f"queue.{role}.cancel")
            trace.record(
                f"queue.{role}",
                "binding",
                local_reader=getattr(mq, "_is_local_reader", None),
                remote_reader=getattr(mq, "_is_remote_reader", None),
                writer=getattr(mq, "_is_writer", None),
            )
        return original_monitor(proc, ObservedDeathPipe(pipe, trace) if pipe is not None else None, requested)

    @wraps(original_busy)
    def busy(proc):
        # One wrapper around the whole loop, not one per RPC/token/iteration.
        try:
            result = original_busy(proc)
        except BaseException as exc:
            trace.record("worker_busy_loop", "error_exit", error=f"{type(exc).__name__}: {exc}")
            raise
        trace.record("worker_busy_loop", "returned")
        return result

    @wraps(original_shutdown)
    def shutdown(proc):
        from vllm_ascend.patch.worker import patch_distributed

        with ExitStack() as scope:
            scope.enter_context(trace.wrapping(proc.worker, "shutdown", "WorkerWrapperBase.shutdown"))
            for name in ("destroy_model_parallel", "destroy_distributed_environment"):
                scope.enter_context(trace.wrapping(multiproc_executor, name, f"distributed.{name}"))
            scope.enter_context(
                trace.wrapping(torch.distributed, "destroy_process_group", "distributed.destroy_process_group")
            )
            for method in ("release", "clear"):
                scope.enter_context(
                    trace.wrapping(patch_distributed._HCCL_PG_REGISTRY, method, f"hccl.registry.{method}")
                )
            # Observe the actual patched coordinator instances and group order;
            # no communicator or group is destroyed/queried by the observer.
            groups = {}
            classes = set()
            for name in CORE_GROUPS:
                group = getattr(parallel_state, name, None)
                if group is not None and id(group) not in groups:
                    groups[id(group)] = name
                    classes.add(type(group))
                    trace.record(
                        f"group.{name}",
                        "binding",
                        unique_name=getattr(group, "unique_name", None),
                        backend=str(getattr(group, "backend", "unknown")),
                        implementation=f"{type(group).__module__}.{type(group).__qualname__}",
                    )
            del group  # do not keep a group alive after Core clears its global
            for cls in classes:
                for method in ("destroy", "_release_hccl_resources"):
                    if hasattr(cls, method):

                        def label(group, method=method):
                            return f"group.{groups.get(id(group), 'unmapped')}.{method}"

                        scope.enter_context(trace.wrapping(cls, method, label))
            return trace.call("WorkerProc.shutdown", original_shutdown, proc)

    worker_proc.monitor_death_pipe = monitor
    worker_proc.worker_busy_loop = busy
    worker_proc.shutdown = shutdown


class ProfileNPUWorker(NPUWorker):
    def __init__(self, *args, **kwargs):
        config = kwargs["vllm_config"]
        options = config.additional_config
        profile = options.get("dspark_confidence_verification", {})
        if (
            not options.get("dspark_profile_worker_exit")
            or not profile.get("profile")
            or profile.get("mode") != "specified_lengths"
        ):
            raise ValueError("Worker exit tracing requires explicit isolated profile configuration")
        self._exit_trace = WorkerExitTrace(
            Path(options["dspark_profile_failure_dir"]) / "worker-exit", kwargs["rank"], arm=False
        )
        install_proc_exit_trace(self._exit_trace, multiproc_executor.WorkerProc)
        super().__init__(*args, **kwargs)

    def shutdown(self):
        # Lazy imports occur only during teardown, after the ordinary worker has
        # loaded its real profiler/model runner and installed distributed patches.
        from vllm_ascend.attention import path_probe
        from vllm_ascend.worker import worker as worker_module

        trace = self._exit_trace
        with ExitStack() as scope:
            scope.enter_context(trace.wrapping(path_probe, "shutdown_attention_path_probe", "ascend.attention_probe"))
            scope.enter_context(trace.wrapping(worker_module, "ensure_kv_transfer_shutdown", "ascend.kv_transfer"))
            for name in ("profiler", "weight_transfer_engine"):
                obj = getattr(self, name, None)
                if obj is not None:
                    scope.enter_context(trace.wrapping(obj, "shutdown", f"ascend.{name}.shutdown"))
                else:
                    trace.record(f"ascend.{name}", "absent")
            runner = getattr(self, "model_runner", None)
            if runner is not None and callable(getattr(runner, "shutdown", None)):
                module = inspect.getmodule(runner.shutdown)
                scope.enter_context(trace.wrapping(runner, "shutdown", "model_runner.shutdown"))
                scope.enter_context(trace.wrapping(torch.accelerator, "synchronize", "device.existing_synchronize"))
                scope.enter_context(trace.wrapping(torch.accelerator, "empty_cache", "device.empty_cache"))
                scope.enter_context(trace.wrapping(gc, "collect", "python.gc_collect"))
                if module is not None and hasattr(module, "free_before_shutdown"):
                    scope.enter_context(
                        trace.wrapping(module, "free_before_shutdown", "model_runner.free_before_shutdown")
                    )
            return trace.call("ascend.worker.shutdown", super().shutdown)
