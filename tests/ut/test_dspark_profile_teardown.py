# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real host wrappers/Core shutdown body and child processes; no NPU proof."""

import ast
import gc
import logging
import multiprocessing
import signal
import weakref
from collections import deque
from functools import wraps
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_failure import methods
from tests.ut.test_dspark_worker_exit import load_source

ROOT = Path(__file__).parents[2]
TEARDOWN = load_source("profile_teardown_test", ROOT / "vllm_ascend/diagnostics/dspark_profile_teardown.py")


def source_class(file, name, namespace, selected=None):
    path = ROOT / "vllm_ascend/diagnostics" / file
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    if selected is not None:
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class Resource:
    def _execute_draft(self, value):
        return value

    def compute_draft_logits(self, value):
        return value

    def execute_model(self, *args, **kwargs):
        return args

    def run_fullgraph(self, desc):
        return desc


@pytest.mark.parametrize("detach", [False, True])
def test_real_wrapper_stack_retention_and_original_core_release(detach):
    runner_class = methods(
        "v1/worker/gpu/model_runner.py",
        "GPUModelRunner",
        ["shutdown"],
        {
            "torch": NS(accelerator=NS(synchronize=lambda: None, empty_cache=lambda: None)),
            "free_before_shutdown": lambda config: None,
            "gc": gc,
            "logger": logging.getLogger(__name__),
        },
    )
    runner_class.execute_model = Resource.execute_model
    runner = runner_class()
    runner.vllm_config = NS(additional_config={})
    runner.cudagraph_manager = Resource()
    runner.speculator = Resource()
    runner.speculator.model = Resource()
    spec_ref, model_ref = weakref.ref(runner.speculator), weakref.ref(runner.speculator.model)
    profiler_class = source_class("dspark_cost_profile.py", "IsolatedCostProfiler", {})
    profiler = profiler_class(runner)
    runner._dspark_cost_profiler = profiler
    observation_class = source_class(
        "dspark_profile_observation.py", "ProfileObservation", {"wraps": wraps}, {"wrap", "close"}
    )
    obs = observation_class.__new__(observation_class)
    obs.runner, obs.hooks = runner, []
    obs.records, obs.numeric_records, obs.transitions = deque(), deque(), deque()
    obs.wrap(runner, "execute_model", "target_execute")
    obs.wrap(runner.cudagraph_manager, "run_fullgraph", "target_full")
    obs.wrap(runner.speculator.model, "compute_draft_logits", "base_logits")
    profiler.observation = obs
    full_module = load_source("full_replay_teardown_test", ROOT / "vllm_ascend/diagnostics/dspark_benchmark_worker.py")
    full = full_module._FullReplayObserver(runner)
    runner._dspark_benchmark_replay_observer = full
    if detach:
        TEARDOWN.shutdown_profile_observers(runner, runner.shutdown)
        assert spec_ref() is None and model_ref() is None
        assert "execute_model" not in vars(runner)
        assert "run_fullgraph" not in vars(runner.cudagraph_manager)
        assert full.runner is profiler.runner is None
        full.close()
        profiler.close()  # idempotent after Core removed speculator
        assert runner.execute_model(12) == (12,)
        assert runner.cudagraph_manager.run_fullgraph(7) == 7
    else:
        runner.shutdown()
        assert spec_ref() is not None and model_ref() is not None
        # The old path has deleted its ownership but hooks still retain it.
        obs.close()
    del obs, full, profiler, runner
    gc.collect()
    assert spec_ref() is None and model_ref() is None


@pytest.mark.parametrize("prior_local", [False, True])
def test_full_close_preserves_preexisting_methods_and_later_replacement(prior_local):
    module = load_source("full_close_test", ROOT / "vllm_ascend/diagnostics/dspark_benchmark_worker.py")
    runner = Resource()
    runner.cudagraph_manager = Resource()
    if prior_local:
        runner.execute_model = lambda *a: "prior"
        runner.cudagraph_manager.run_fullgraph = lambda *a: "prior-graph"
    execute, graph = runner.execute_model, runner.cudagraph_manager.run_fullgraph
    full = module._FullReplayObserver(runner)
    full.close()
    assert runner.execute_model == execute and runner.cudagraph_manager.run_fullgraph == graph
    assert ("execute_model" in vars(runner)) == prior_local
    full = module._FullReplayObserver(runner)
    replacement = lambda *a: "new-owner"
    runner.execute_model = replacement
    full.close()
    assert runner.execute_model is replacement


@pytest.mark.parametrize("observer_error,core_error", [(False, False), (True, False), (False, True), (True, True)])
def test_cleanup_errors_never_skip_core_or_replace_its_error(observer_error, core_error):
    calls = []
    observer_failure, core_failure = RuntimeError("detach"), RuntimeError("original Core failure")

    def close():
        calls.append("outer")
        if observer_error:
            raise observer_failure

    def shutdown():
        calls.append("core")
        if core_error:
            raise core_failure
        return "returned"

    runner = NS(
        _dspark_benchmark_replay_observer=NS(close=close), _dspark_cost_profiler=NS(close=lambda: calls.append("inner"))
    )
    if observer_error or core_error:
        with pytest.raises(RuntimeError) as caught:
            TEARDOWN.shutdown_profile_observers(runner, shutdown)
        assert caught.value is (core_failure if core_error else observer_failure)
    else:
        assert TEARDOWN.shutdown_profile_observers(runner, shutdown) == "returned"
    assert calls == ["outer", "inner", "core"]
    assert TEARDOWN.shutdown_profile_observers(None, lambda: 17) == 17


def child_wait(connection):
    connection.send("ready")
    connection.recv()


@pytest.mark.parametrize("kill", [False, True])
def test_real_parent_reap_records_observed_status(kill):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    proc = context.Process(target=child_wait, args=(child,))
    proc.start()
    child.close()
    try:
        assert parent.poll(10) and parent.recv() == "ready"
        if kill:
            proc.kill()
        else:
            parent.send("finish")
        result = TEARDOWN.reap_workers([NS(proc=proc, rank=3)], timeout=2)
        row = result["workers"][0]
        assert row["status"] == "reaped"
        assert row["raw_exitcode"] == (-signal.SIGKILL if kill else 0)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(2)
        proc.close()
        parent.close()


def test_join_deadline_shared_and_unavailable_not_inferred(monkeypatch):
    now, waits = [0.0], []
    monkeypatch.setattr(TEARDOWN.time, "monotonic", lambda: now[0])

    class Process:
        pid, exitcode = 123, None

        def join(self, timeout):
            waits.append(timeout)
            now[0] += timeout

    result = TEARDOWN.reap_workers([NS(proc=Process(), rank=i) for i in range(8)])
    assert waits == [1.0] + [0.0] * 7
    assert all(w["raw_exitcode"] is None and w["status"] == "unavailable" for w in result["workers"])
    assert result["elapsed_seconds"] == 1


def test_join_error_remains_unavailable():
    proc = NS(pid=1, exitcode=None)
    result = TEARDOWN.reap_workers([NS(proc=proc, rank=0)], timeout=0)
    assert result["workers"][0]["status"] == "unavailable"
    assert "AttributeError" in result["workers"][0]["reason"]
