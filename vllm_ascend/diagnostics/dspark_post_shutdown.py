# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exit-only weak lifetimes and Python finalization receipts, never cleanup."""

import atexit
import gc
import json
import multiprocessing.util
import os
import sys
import threading
import time
import weakref
from collections import deque
from itertools import islice
from types import MappingProxyType

MAX_OBJECTS = 64
MAX_GRAPHS = 7
MAX_EVENTS = 128
EVENT_BUDGETS = MappingProxyType({"ordinary": 64, "critical": 48, "error": 16})
CRITICAL_STAGES = (
    "WorkerProc.shutdown",
    "shutdown_wrapper",
    "lifetimes",
    "multiprocessing.exit_function",
    "threading.shutdown",
    "atexit",
)
MAX_HISTORY = 16
MAX_PENDING = 16
MAX_ERRORS = 8
MAX_COUNTERS = 16
MAX_STATE_BYTES = 65536
MAX_EVENT_BYTES = 16384


def stored(obj, name):
    # No properties, tensor operations or proxy __getattr__ during teardown.
    return vars(obj).get(name) if hasattr(type(obj), "__dict__") and hasattr(obj, "__dict__") else None


class PostShutdownTrace:
    """Weak references cannot keep runner/model/graph resources alive.

    A cleared weakref does NOT prove a native destructor completed. CPython's
    final _PyGC_CollectNoFail bypasses gc.callbacks; absence of GC events does
    not prove absence of GC. Process exit remains a parent-side observation.
    """

    def __init__(self, trace):
        self.fd = os.open(
            trace.directory / f"{trace.prefix}-lifetimes.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        request_path = trace.directory / "request.json"
        request = json.loads(request_path.read_text()) if request_path.exists() else {}
        self.identity = {
            "rank": trace.rank,
            "pid": trace.pid,
            "instance": trace.instance,
            "request": request.get("id"),
            "point": request.get("point"),
            "performance_eligible": False,
        }
        self.refs = {}
        self.objects = {}
        self.count = self.dropped = 0
        self.state_path = trace.directory / f"{trace.prefix}-lifetimes-state.json"
        self.temp_path = str(self.state_path) + ".tmp"
        self.budgets = dict(EVENT_BUDGETS)
        self.written = dict.fromkeys(EVENT_BUDGETS, 0)
        self.truncated = dict.fromkeys(EVENT_BUDGETS, 0)
        self.history = deque(maxlen=MAX_HISTORY)
        self.pending = {}
        self.errors = []
        self.counters = {}
        self.overflow = {"pending": 0, "errors": 0, "counter_keys": 0, "unpaired_returns": 0}
        self.max_event_bytes = MAX_EVENT_BYTES
        self.lock = threading.Lock()
        self.patches = []
        self.armed = False
        self.recording_error = None
        # Keep only host functions usable while module dictionaries are cleared.
        self.write = os.write
        self.dumps = json.dumps
        self.clock = time.monotonic_ns
        self.wall_clock = time.time_ns
        self.finalizing = sys.is_finalizing
        self.ident = threading.get_ident
        self.open_fd, self.close_fd, self.replace = os.open, os.close, os.replace

    def record(self, stage, event, *, call_id=None, **fields):
        # GC/weak callbacks may interrupt this writer on the SAME thread. Never
        # wait for our lock or recursively log an observer failure.
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return None
        try:
            self.count += 1
            data = {
                **self.identity,
                "seq": self.count,
                "call_id": self.count if event in ("begin", "start") else call_id,
                "monotonic_ns": self.clock(),
                "unix_ns": self.wall_clock(),
                "thread_id": self.ident(),
                "is_finalizing": self.finalizing(),
                "stage": stage,
                "event": event,
                "alive": {label: ref() is not None for label, ref in self.refs.items()},
                "dropped_reentrant_events": self.dropped,
                "recording_error": self.recording_error,
                **fields,
            }
            self.update_state(data, call_id)
            repeated = stage == "multiprocessing.finalizer" or stage.startswith("gc.callback.")
            category = "error" if event == "error" else "critical" if stage in CRITICAL_STAGES else "ordinary"
            if not repeated or event == "error":
                used, limit = self.written[category], self.budgets[category]
                if used < limit:
                    self.written[category] += 1
                    line = data
                    if used == limit - 1:
                        line = data | {"stage": "coverage", "event": "event_limit_reached", "category": category}
                        self.truncated[category] += 1
                    self.emit(self.fd, line, self.max_event_bytes)
                else:
                    self.truncated[category] += 1
            # Repeated finalizers/GC never spend critical append capacity. The
            # atomic fixed-size snapshot holds their active calls and history.
            self.save_state(data)
            return data["seq"]
        except Exception as exc:
            self.recording_error = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            self.lock.release()

    def update_state(self, data, call_id):
        compact = {k: v for k, v in data.items() if k not in (*self.identity, "alive")}
        self.history.append(compact)
        key = data["stage"] + ":" + str(data.get("callback", ""))[:256]
        if key not in self.counters and len(self.counters) >= MAX_COUNTERS:
            self.overflow["counter_keys"] += 1
            key = "other"
        counts = self.counters.setdefault(key, {"begin": 0, "returned": 0, "error": 0, "other": 0})
        event = data["event"]
        counts[event if event in counts else "other"] += 1
        if event in ("begin", "start"):
            if len(self.pending) < MAX_PENDING:
                self.pending[data["seq"]] = compact
            else:
                self.overflow["pending"] += 1
        elif event in ("returned", "stop", "error"):
            if event == "stop":
                # GC callbacks carry generation/thread identity, no bound resource.
                call_id = next(
                    (
                        k
                        for k, v in self.pending.items()
                        if v["stage"] == data["stage"] and v["thread_id"] == data["thread_id"]
                    ),
                    None,
                )
            if self.pending.pop(call_id, None) is None:
                self.overflow["unpaired_returns"] += 1
        if event == "error":
            if len(self.errors) < MAX_ERRORS:
                self.errors.append(compact)
            else:
                self.overflow["errors"] += 1

    def emit(self, fd, data, limit):
        payload = (self.dumps(data, allow_nan=False) + "\n").encode()
        if len(payload) > limit:
            raise ValueError("Lifetime receipt exceeds byte budget")
        if self.write(fd, payload) != len(payload):
            raise OSError("Partial lifetime receipt write")

    def save_state(self, latest):
        state = {
            **self.identity,
            "last_seq": self.count,
            "latest": latest,
            "pending_calls": list(self.pending.values()),
            "history": list(self.history),
            "first_errors": self.errors,
            "counts": self.counters,
            "overflow": self.overflow,
            "append_written": self.written,
            "append_truncated": self.truncated,
            "history_overwritten": max(0, self.count - MAX_HISTORY),
            "dropped_reentrant_events": self.dropped,
            "recording_error": self.recording_error,
            "process_exit": "UNAVAILABLE: this is a worker receipt, not a parent wait result",
        }
        fd = self.open_fd(self.temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            self.emit(fd, state, MAX_STATE_BYTES)
        finally:
            self.close_fd(fd)
        self.replace(self.temp_path, self.state_path)

    def watch(self, label, obj):
        if len(self.objects) >= MAX_OBJECTS:
            return
        info = {"id": id(obj) if obj is not None else None, "type": type(obj).__qualname__}
        if obj is None:
            info["unavailable"] = "attribute absent or None"
        else:
            try:
                # Callback captures label + host writer, never obj/bound methods.
                self.refs[label] = weakref.ref(obj, lambda ref: self.record("weakref", "cleared", label=label))
            except TypeError:
                info["unavailable"] = "object does not support weak references"
        self.objects[label] = info

    def capture(self, proc):
        wrapper = stored(proc, "worker")
        worker = stored(wrapper, "worker")
        runner = stored(worker, "model_runner")
        spec = stored(runner, "speculator")
        manager = stored(runner, "cudagraph_manager")
        profiler = stored(runner, "_dspark_cost_profiler")
        capture = stored(manager, "_dspark_auxiliary_capture")
        for label, obj in (
            ("worker_proc", proc),
            ("worker_wrapper", wrapper),
            ("worker", worker),
            ("runner", runner),
            ("speculator", spec),
            ("target_model", stored(runner, "model")),
            ("draft_model", stored(spec, "model")),
            ("graph_manager", manager),
            ("cost_profiler", profiler),
            ("profile_observation", stored(profiler, "observation")),
            ("full_replay_observer", stored(runner, "_dspark_benchmark_replay_observer")),
            ("auxiliary_capture", capture),
            ("target_flags", stored(capture, "bank")),
            ("persistent_hidden", stored(manager, "hidden_states")),
            ("async_output_thread", stored(proc, "async_output_copy_thread")),
        ):
            self.watch(label, obj)
        graphs = stored(manager, "graphs") or {}
        for i, graph in enumerate(islice(graphs.values(), MAX_GRAPHS)):
            self.watch(f"graph.{i}.proxy", graph)
            self.watch(f"graph.{i}.native", stored(graph, "graph"))
        shapes = stored(capture, "shapes") or {}
        for i, shape in enumerate(islice(shapes.values(), MAX_GRAPHS)):
            for j, source in enumerate(islice(shape.get("sources", ()), 3)):
                self.watch(f"capture.{i}.source.{j}", source)
        self.record(
            "lifetimes",
            "before_shutdown",
            objects=self.objects,
            graph_count=len(graphs),
            capture_shape_count=len(shapes),
            sampled_graph_limit=MAX_GRAPHS,
        )

    def wrap(self, owner, name, stage):
        original = getattr(owner, name)

        def observed(*args, **kwargs):
            detail = {}
            if stage == "multiprocessing.finalizer":
                callback = stored(args[0], "_callback")
                detail = {
                    "callback": f"{getattr(callback, '__module__', '')}.{getattr(callback, '__qualname__', '')}"[:256],
                    "key": stored(args[0], "_key"),
                }
                del callback  # do not extend callback/resource lifetime
            call_id = self.record(stage, "begin", **detail)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self.record(stage, "error", call_id=call_id, error=f"{type(exc).__name__}: {exc}"[:1024], **detail)
                raise
            self.record(stage, "returned", call_id=call_id, **detail)
            return result

        setattr(owner, name, observed)
        self.patches.append((owner, name, original, observed))

    def gc_callback(self, phase, info):
        self.record(f"gc.callback.{info.get('generation', 'unknown')}", phase, gc_info=dict(info))

    def at_exit(self):
        self.record("atexit", "marker")  # one callback, not completion of all atexit handlers

    def arm(self):
        if self.armed:
            return
        self.armed = True
        self.record("shutdown_wrapper", "scope_unwound")
        for owner, name, stage in (
            (multiprocessing.util, "_exit_function", "multiprocessing.exit_function"),
            # _exit_function binds _run_finalizers in a default argument. A
            # module alias wrapper would miss the real call. Observe Finalize.
            (multiprocessing.util.Finalize, "__call__", "multiprocessing.finalizer"),
            (threading, "_shutdown", "threading.shutdown"),
        ):
            self.wrap(owner, name, stage)
        gc.callbacks.append(self.gc_callback)
        atexit.register(self.at_exit)
        self.record("lifetimes", "armed_after_shutdown")

    def close(self):
        """For isolated CPU tests only; workers retain receipts until process exit."""
        for owner, name, original, observed in reversed(self.patches):
            if getattr(owner, name) is observed:
                setattr(owner, name, original)
        self.patches.clear()
        if self.gc_callback in gc.callbacks:
            gc.callbacks.remove(self.gc_callback)
        atexit.unregister(self.at_exit)
        self.refs.clear()
        os.close(self.fd)
