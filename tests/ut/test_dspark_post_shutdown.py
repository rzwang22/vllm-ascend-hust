# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifetime proofs + real spawn finalization; no CANN/NPU exit proof."""

import ast
import gc
import json
import logging
import subprocess
import sys
import weakref
from collections import deque
from functools import wraps
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_failure import methods
from tests.ut.test_dspark_worker_exit import load_source

ROOT = Path(__file__).parents[2]
MODULE = ROOT / "vllm_ascend/diagnostics/dspark_post_shutdown.py"


class Resource:
    def compute_draft_logits(self):
        pass


@pytest.fixture
def module():
    return load_source("post_shutdown_lifetimes", MODULE)


@pytest.fixture
def trace(tmp_path, module):
    (tmp_path / "request.json").write_text(json.dumps({"id": "this-cleanup", "point": "last-point"}))
    trace = module.PostShutdownTrace(NS(directory=tmp_path, prefix="test", rank=4, pid=123, instance="worker-A"))
    yield trace
    trace.close()


def records(tmp_path):
    return [json.loads(line) for line in (tmp_path / "test-lifetimes.jsonl").read_text().splitlines()]


def test_real_profile_hook_retains_draft_after_core_shutdown_until_unwrapped():
    # Actual profile hook implementation and frozen MRV2 shutdown. Only device
    # leaves are mocks. Retention is proven; a native destructor hang is not.
    path = ROOT / "vllm_ascend/diagnostics/dspark_profile_observation.py"
    cls = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "ProfileObservation"
    )
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("wrap", "close")]
    namespace = {"wraps": wraps}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    runner = methods(
        "v1/worker/gpu/model_runner.py",
        "GPUModelRunner",
        ["shutdown"],
        {
            "torch": NS(accelerator=NS(synchronize=lambda: None, empty_cache=lambda: None)),
            "free_before_shutdown": lambda config: None,
            "gc": gc,
            "logger": logging.getLogger("post-shutdown-test"),
        },
    )()
    runner.vllm_config = NS()
    runner.speculator = NS(model=Resource())
    ref = weakref.ref(runner.speculator.model)
    obs = namespace["ProfileObservation"].__new__(namespace["ProfileObservation"])
    obs.runner = runner
    obs.hooks = []
    obs.records, obs.numeric_records, obs.transitions = deque(), deque(), deque()
    obs.wrap(runner.speculator.model, "compute_draft_logits", "base_logits")
    runner.shutdown()
    assert runner.speculator is None and ref() is not None
    obs.close()
    gc.collect()
    assert ref() is None


def test_census_is_weak_bounded_and_unavailable_is_explicit(trace, tmp_path, module):
    runner, manager, native, draft = Resource(), Resource(), Resource(), Resource()
    runner.cudagraph_manager = manager
    manager.model_runner = runner
    manager.graphs = {i: NS(graph=native) for i in range(module.MAX_GRAPHS + 2)}
    runner.speculator = NS(model=draft)
    proc = Resource()
    proc.worker = NS(worker=NS(model_runner=runner))
    ref = weakref.ref(runner)
    trace.capture(proc)
    before = records(tmp_path)[0]
    assert before["alive"]["runner"] and before["request"] == "this-cleanup" and before["rank"] == 4
    assert before["graph_count"] == 9 and "graph.7.native" not in before["objects"]
    assert before["objects"]["worker_wrapper"]["unavailable"] == "object does not support weak references"
    assert before["objects"]["target_model"]["unavailable"] == "attribute absent or None"
    del runner, manager, proc, draft, native
    gc.collect()
    assert ref() is None  # observer does not keep the runner/graph cycle alive
    assert any(r["stage"] == "weakref" and r["label"] == "runner" for r in records(tmp_path))
    assert all(not r["performance_eligible"] for r in records(tmp_path))


def test_writer_reentrancy_failure_and_bound_do_not_block_cleanup(trace, tmp_path, module):
    with trace.lock:
        trace.record("inside-allocation", "begin")
    assert trace.dropped == 1
    trace.record("next", "returned")
    assert records(tmp_path)[0]["dropped_reentrant_events"] == 1
    original = trace.write
    trace.write = lambda *args: (_ for _ in ()).throw(OSError("disk unavailable"))
    trace.record("disk", "failed")
    trace.write = original
    trace.record("later", "returned")
    assert "disk unavailable" in records(tmp_path)[-1]["recording_error"]
    for _ in range(module.MAX_EVENTS + 1):
        trace.record("repeat", "returned")
    rows = records(tmp_path)
    assert len(rows) <= module.MAX_EVENTS and rows[-1]["event"] == "event_limit_reached"
    assert (
        max(len(line) for line in (tmp_path / "test-lifetimes.jsonl").read_bytes().splitlines())
        < module.MAX_EVENT_BYTES
    )


def test_phase_wrapper_preserves_first_exception_and_binding(trace, tmp_path):
    original_error = RuntimeError("original shutdown error")

    def original():
        raise original_error

    owner = NS(exit=original)
    trace.wrap(owner, "exit", "fake.exit")
    with pytest.raises(RuntimeError) as caught:
        owner.exit()
    assert caught.value is original_error
    assert [(r["stage"], r["event"]) for r in records(tmp_path)] == [("fake.exit", "begin"), ("fake.exit", "error")]


@pytest.mark.parametrize("mode", ["natural", "blocked", "error"])
def test_real_spawn_finalization_and_bounded_parent_observation(tmp_path, mode):
    # Uses real spawn.Process -> run -> util._exit_function -> threading/atexit.
    # The blocked case stops inside a multiprocessing finalizer BEFORE signals.
    script = tmp_path / "spawn_exit.py"
    script.write_text("""
import importlib.util,json,multiprocessing,multiprocessing.util,sys,threading,time
from pathlib import Path
from types import SimpleNamespace as NS
def child(module_path,directory,mode):
    s=importlib.util.spec_from_file_location('post',module_path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    t=m.PostShutdownTrace(NS(directory=Path(directory),prefix='test',rank=0,pid=__import__('os').getpid(),instance='spawn'))
    class Resource: pass
    obj=Resource();obj.cycle=obj;t.watch('cycle',obj)
    t.record('WorkerProc.shutdown','returned')
    t.arm();t.arm()
    def finish():
        if mode=='blocked': threading.Event().wait(60)
        if mode=='error': raise RuntimeError('fixture finalizer error')
    multiprocessing.util.Finalize(None,finish,exitpriority=1)
if __name__=='__main__':
    p=multiprocessing.get_context('spawn').Process(target=child,args=tuple(sys.argv[1:]));p.start()
    try:
        p.join(timeout=2)
        if p.is_alive():
            rows=[json.loads(x) for x in (Path(sys.argv[2])/'test-lifetimes.jsonl').read_text().splitlines()]
            assert any(r['stage']=='multiprocessing.finalizer' and r['event']=='begin' for r in rows)
            assert not any(r['stage']=='multiprocessing.finalizer' and r['event']=='returned' for r in rows)
            (Path(sys.argv[2])/'before-signal.json').write_text(json.dumps({'raw_exitcode':p.exitcode,'rows':len(rows)}))
    finally:
        if p.is_alive(): p.terminate();p.join(timeout=2)
        if p.is_alive(): p.kill();p.join(timeout=2)
    print('CHILD_EXITCODE='+str(p.exitcode))
""")
    result = subprocess.run(
        [sys.executable, str(script), str(MODULE), str(tmp_path), mode], capture_output=True, text=True, timeout=12
    )
    assert result.returncode == 0, result.stderr
    rows = records(tmp_path)
    phases = [(r["stage"], r["event"]) for r in rows]
    assert phases.count(("lifetimes", "armed_after_shutdown")) == 1
    assert phases.index(("shutdown_wrapper", "scope_unwound")) < phases.index(
        ("multiprocessing.exit_function", "begin")
    )
    if mode == "blocked":
        assert json.loads((tmp_path / "before-signal.json").read_text())["raw_exitcode"] is None
        assert ("multiprocessing.exit_function", "returned") not in phases
    else:
        assert "CHILD_EXITCODE=0" in result.stdout
        assert ("multiprocessing.exit_function", "returned") in phases
        assert ("threading.shutdown", "returned") in phases and ("atexit", "marker") in phases
        assert any(r["stage"] == "weakref" and r["label"] == "cycle" for r in rows)
        if mode == "error":
            # CPython's existing finalizer policy prints exceptions and proceeds;
            # our wrapper preserves it, it does not invent a new process code.
            assert "fixture finalizer error" in result.stderr
            assert any(r["stage"] == "multiprocessing.finalizer" and r["event"] == "error" for r in rows)
