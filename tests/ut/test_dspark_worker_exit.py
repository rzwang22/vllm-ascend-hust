# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real host threads/signals and frozen cleanup bodies; no NPU validation."""

import ast
import gc
import importlib.util
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.ut.test_dspark_profile_failure import methods
from tools.dspark import run_confidence_verification as driver
from tools.dspark import run_large_batch as large
from tools.dspark import startup_cost_profile as profile

ROOT = Path(__file__).parents[2]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trace_module():
    return load_source("exit_trace_test", ROOT / "vllm_ascend/diagnostics/dspark_worker_exit.py")


@pytest.fixture
def trace(tmp_path, monkeypatch, trace_module):
    # Real faulthandler signal registration is separately exercised in an owned
    # subprocess. Never change pytest's process-wide signal handlers.
    monkeypatch.setattr(trace_module.faulthandler, "register", lambda *a, **kw: None)
    monkeypatch.setattr(trace_module.signal, "getsignal", lambda *a: signal.SIG_DFL)
    t = trace_module.WorkerExitTrace(tmp_path, 3)
    yield t
    os.close(t.fd)
    if t.stack_file:
        t.stack_file.close()


def records(trace):
    return [json.loads(line) for line in (trace.directory / f"{trace.prefix}-steps.jsonl").read_text().splitlines()]


def test_stage_records_preserve_binding_errors_threads_and_bounds(trace, trace_module):
    calls = []

    class Resource:
        def shutdown(self, value):
            calls.append(value)
            if value == "bad":
                raise OSError("original teardown error")
            return value

    resource = Resource()
    with trace.wrapping(resource, "shutdown", "resource.shutdown"):
        assert resource.shutdown("ok") == "ok"
        with pytest.raises(OSError, match="original teardown error"):
            resource.shutdown("bad")
    assert "shutdown" not in vars(resource)  # restore inherited method, not an extra bound alias
    assert calls == ["ok", "bad"]
    assert [(r["stage"], r["event"]) for r in records(trace)] == [
        ("resource.shutdown", e) for e in ("begin", "returned", "begin", "error")
    ]
    assert all(r["rank"] == 3 and r["pid"] == os.getpid() and not r["performance_eligible"] for r in records(trace))
    for _ in range(trace_module.MAX_EVENTS + 5):
        trace.record("repeated", "begin")
    assert len(records(trace)) == trace_module.MAX_EVENTS
    assert records(trace)[-1]["event"] == "event_limit_reached"


def test_record_write_error_does_not_change_cleanup_error(trace, monkeypatch):
    monkeypatch.setattr(os, "write", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(ValueError, match="original"):
        trace.call("cleanup", lambda: (_ for _ in ()).throw(ValueError("original")))
    assert "disk" in trace.recording_error


def test_queue_wrapper_does_not_retain_queue_or_add_self_cycle(monkeypatch, trace, trace_module):
    impl, *_ = load_integration(monkeypatch, trace_module)

    class Queue:
        def shutdown(self):
            return 7

    queue = Queue()
    reference = weakref.ref(queue)
    impl.observe_queue_method(trace, queue, "shutdown", "queue.shutdown")
    assert queue.shutdown() == 7
    # With cyclic GC disabled, the queue must still be released immediately,
    # as it would be when original WorkerProc clears its queue attributes.
    enabled = gc.isenabled()
    gc.disable()
    try:
        del queue
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


def load_integration(monkeypatch, trace_module):
    """Load the real plugin wrappers and actual Core/Ascend shutdown methods."""

    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    calls = []
    torch = module(
        "torch",
        accelerator=SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"), empty_cache=lambda: calls.append("empty_cache")
        ),
        distributed=SimpleNamespace(destroy_process_group=lambda *a: calls.append("pg.destroy")),
    )
    path_probe = module("vllm_ascend.attention.path_probe", shutdown_attention_path_probe=lambda: calls.append("probe"))
    module("vllm_ascend.attention", path_probe=path_probe)
    registry = SimpleNamespace(release=lambda *a: None, clear=lambda: calls.append("registry.clear"))
    patch = module("vllm_ascend.patch.worker.patch_distributed", _HCCL_PG_REGISTRY=registry)
    module("vllm_ascend.patch.worker", patch_distributed=patch)
    # Ascend's actual shutdown body, without importing torch_npu/model code.
    tree = ast.parse((ROOT / "vllm_ascend/worker/worker.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUWorker")
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown"]
    worker_module = module("vllm_ascend.worker.worker", ensure_kv_transfer_shutdown=lambda: calls.append("kv_transfer"))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "ascend-worker-shutdown", "exec"), worker_module.__dict__)
    module("vllm_ascend.worker", worker=worker_module)
    core = module("vllm.v1.executor.multiproc_executor")
    parallel = module("vllm.distributed.parallel_state")
    core.destroy_model_parallel = lambda: calls.append("model_parallel")
    core.destroy_distributed_environment = lambda: calls.append("distributed_environment")
    # The real monitor starts DeathPipeMonitor; the real busy loop blocks on MQ.
    proc = methods(
        "v1/executor/multiproc_executor.py",
        "WorkerProc",
        ["monitor_death_pipe", "worker_busy_loop", "shutdown"],
        {
            "Thread": threading.Thread,
            "threading": threading,
            "logger": SimpleNamespace(info_once=lambda *a: None, warning=lambda *a: None),
            "destroy_model_parallel": core.destroy_model_parallel,
            "destroy_distributed_environment": core.destroy_distributed_environment,
        },
    )
    # Source function globals must see the same aliases that production wraps.
    for method in ("shutdown",):
        fn = getattr(proc, method)
        import types

        setattr(proc, method, types.FunctionType(fn.__code__, core.__dict__, fn.__name__))
    core.WorkerProc = proc
    module("vllm.v1.executor", multiproc_executor=core)
    module("vllm.distributed", parallel_state=parallel)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_worker_exit", trace_module)
    implementation = load_source("profile_worker_test", ROOT / "vllm_ascend/diagnostics/dspark_profile_worker.py")
    return implementation, proc, worker_module, torch, calls


@pytest.mark.parametrize("failure_stage", [None, "synchronize", "model_parallel"])
def test_actual_death_queue_loop_and_shutdown_path(monkeypatch, trace, trace_module, failure_stage):
    impl, proc_cls, worker_module, torch, calls = load_integration(monkeypatch, trace_module)
    runner_module = ModuleType("runner_for_exit_test")
    runner_module.__dict__.update(
        torch=torch,
        gc=SimpleNamespace(collect=lambda: calls.append("gc")),
        free_before_shutdown=lambda _: calls.append("free_before_shutdown"),
        logger=logging.getLogger("exit-test"),
    )
    runner = methods("v1/worker/gpu/model_runner.py", "GPUModelRunner", ["shutdown"], runner_module.__dict__)()
    # inspect.getmodule must resolve the actual inherited method's global alias.
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    runner.__class__.__module__ = runner_module.__name__
    runner.shutdown.__func__.__module__ = runner_module.__name__
    runner.vllm_config = SimpleNamespace()
    runner.kv_caches = [1]
    runner.attn_groups = [2]
    runner.speculator = object()
    runner.model = object()
    worker = impl.ProfileNPUWorker.__new__(impl.ProfileNPUWorker)
    worker._exit_trace = trace
    worker.profiler = None
    worker.model_runner = runner
    if failure_stage:

        def fail():
            raise RuntimeError(f"blocked stage {failure_stage}")

        if failure_stage == "synchronize":
            torch.accelerator.synchronize = fail
        else:
            impl.multiproc_executor.destroy_model_parallel = fail
    impl.install_proc_exit_trace(trace, proc_cls)
    requested = threading.Event()
    wake = threading.Event()
    mq_cls = methods("distributed/device_communicators/shm_broadcast.py", "MessageQueue", ["shutdown"], {})
    mq = mq_cls()
    mq.shutting_down = False
    mq._spin_condition = SimpleNamespace(cancel=wake.set)
    mq._is_local_reader = True
    mq._is_writer = mq._is_remote_reader = False

    def dequeue(**kwargs):
        assert kwargs == {"indefinite": True}
        assert wake.wait(2)
        assert mq.shutting_down
        raise RuntimeError("cancelled")

    mq.dequeue = dequeue
    proc = proc_cls()
    proc.rpc_broadcast_mq = mq
    proc.worker_response_mq = None
    wrapper_cls = methods("v1/worker/worker_base.py", "WorkerWrapperBase", ["shutdown"], {})
    proc.worker = wrapper_cls()
    proc.worker.worker = worker
    read, write = multiprocessing.Pipe(duplex=False)
    trace_module.save(trace.directory / "request.json", {"id": "cleanup-this-engine", "point": "last-point"})
    proc.monitor_death_pipe(read, requested)
    write.close()
    with pytest.raises(RuntimeError, match="cancelled"):
        proc.worker_busy_loop()
    assert requested.is_set()
    if failure_stage:
        with pytest.raises(RuntimeError, match=failure_stage):
            proc.shutdown()
    else:
        proc.shutdown()
        assert calls == [
            "probe",
            "kv_transfer",
            "synchronize",
            "free_before_shutdown",
            "gc",
            "empty_cache",
            "model_parallel",
            "distributed_environment",
        ]
        assert runner.kv_caches == [] and runner.attn_groups == [] and not hasattr(runner, "model")
    read.close()
    rows = records(trace)
    eof = next(r for r in rows if r["stage"] == "death_pipe.recv")
    assert eof["event"] == "eof" and eof["thread"] == "DeathPipeMonitor"
    assert eof["request"] == "cleanup-this-engine"
    assert any(r["stage"] == "worker_busy_loop" and r["event"] == "error_exit" for r in rows)
    stages = [(r["stage"], r["event"]) for r in rows]
    assert ("device.existing_synchronize", "begin") in stages
    assert ("WorkerProc.shutdown", "returned" if failure_stage is None else "error") in stages
    if failure_stage == "synchronize":
        assert ("device.existing_synchronize", "error") in stages
        assert ("model_runner.free_before_shutdown", "begin") not in stages


def test_real_pre_escalation_stack_request_and_unknown_exitcode(tmp_path, trace_module, monkeypatch):
    module_path = ROOT / "vllm_ascend/diagnostics/dspark_worker_exit.py"
    code = """
import importlib.util,sys,time,threading
from pathlib import Path
s=importlib.util.spec_from_file_location('trace',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
t=m.WorkerExitTrace(sys.argv[2],0)
while not (Path(sys.argv[2])/'request.json').exists(): time.sleep(.005)
def blocked_device_release(): threading.Event().wait(20)
t.call('simulated.device_release',blocked_device_release)
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(module_path), str(tmp_path)])

    class Process:
        pid = child.pid

        @property
        def exitcode(self):
            return child.poll()

    watch = None
    try:
        end = time.monotonic() + 3
        while not (tmp_path / "rank-0-ready.json").exists() and time.monotonic() < end:
            time.sleep(0.01)
        ready = json.loads((tmp_path / "rank-0-ready.json").read_text())
        assert ready["signal_registered"]
        watch = trace_module.ExitWatch(tmp_path, [SimpleNamespace(proc=Process(), rank=0)], "last", grace=0.3)
        watch.thread.start()
        end = time.monotonic() + 3
        stacks = tmp_path / ready["stack_file"]
        while stacks.stat().st_size == 0 and time.monotonic() < end:
            time.sleep(0.01)
        assert "blocked_device_release" in stacks.read_text()
        watch.snapshot("before-escalation-1")
        pre = json.loads((tmp_path / "parent-before-escalation-1.json").read_text())["workers"][0]
        assert pre["raw_exitcode"] is None and pre["stack_bytes_available_now"] > 0
        child.terminate()
        assert child.wait(timeout=3) == -signal.SIGTERM
        assert pre["raw_exitcode"] is None  # no retroactive -9 fabrication
    finally:
        if watch:
            watch.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)


def test_watch_refuses_mismatched_registration_and_bounds_procfs(tmp_path, trace_module, monkeypatch):
    calls = []
    monkeypatch.setattr(os, "kill", lambda *a: calls.append(a))
    trace_module.save(tmp_path / "rank-0-ready.json", {"pid": 999, "rank": 0, "signal_registered": True})
    watch = trace_module.ExitWatch(
        tmp_path, [SimpleNamespace(proc=SimpleNamespace(pid=123, exitcode=None), rank=0)], "p", 5
    )
    watch.snapshot("test", request_stacks=True)
    assert calls == []
    assert json.loads((tmp_path / "parent-test.json").read_text())["workers"][0]["raw_exitcode"] is None
    root = tmp_path / "123"
    root.mkdir()
    (root / "status").write_text("x" * (trace_module.MAX_PROC_BYTES + 10))
    snapshot = trace_module.process_snapshot(123, tmp_path)
    assert len(snapshot["status"]["text"]) == trace_module.MAX_PROC_BYTES
    assert "unavailable" in snapshot["stat"] and "threads_unavailable" in snapshot


@pytest.mark.parametrize("enabled", [False, True])
def test_executor_watch_is_exit_only_and_keeps_original_error(tmp_path, trace_module, monkeypatch, enabled):
    from tests.ut.test_dspark_profile_failure import load_executor

    cleanup = load_source("cleanup_for_exit_test", ROOT / "vllm_ascend/diagnostics/dspark_cleanup.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_cleanup", cleanup)
    calls = []

    class Watch:
        def __init__(self, *args):
            calls.append("watch.init")
            self.thread = SimpleNamespace(start=lambda: calls.append("watch.start"))

        def snapshot(self, label):
            assert label == "before-escalation-1"
            calls.append("snapshot")
            assert json.loads((tmp_path / "worker-cleanup.json").read_text())["forced_cleanup"]

        def close(self):
            calls.append("watch.close")

    monkeypatch.setattr(trace_module, "ExitWatch", Watch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_worker_exit", trace_module)
    # load_executor inspects frozen sources before the lightweight env module is installed.
    module = load_executor(monkeypatch)
    vllm = ModuleType("vllm")
    vllm.envs = SimpleNamespace(VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=5)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    logger = logging.getLogger("vllm.v1.executor.multiproc_executor")

    def original(self):
        logger.warning(cleanup.EXECUTOR_FORCE_MESSAGES[0], 8)
        calls.append("original.shutdown")
        raise OSError("original shutdown error")

    monkeypatch.setattr(module.ProfileMultiprocExecutor.__mro__[1], "shutdown", original)
    executor = module.ProfileMultiprocExecutor(
        SimpleNamespace(
            additional_config={
                "dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"},
                "dspark_profile_failure_dir": str(tmp_path),
                "dspark_profile_worker_exit": enabled,
            }
        )
    )
    assert calls == []  # no watcher during generation / construction
    with pytest.raises(OSError, match="original shutdown error"):
        executor.shutdown()
    assert calls == (
        ["watch.init", "watch.start", "snapshot", "original.shutdown", "watch.close"]
        if enabled
        else ["original.shutdown"]
    )


def test_shell_control_opt_in_keeps_original_single_run(tmp_path):
    bash = tmp_path / "bash"
    bash.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    bash.chmod(0o755)
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "tools/dspark/run_dspark_profile_control.sh"),
            "sha",
            "manifest",
            "target-boundaries",
            "1",
            "--worker-exit",
        ],
        env=dict(os.environ, PATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    cmd = result.stdout.splitlines()
    assert cmd.count("--profile-worker-exit") == 1
    for flag, expected in (
        ("--batches", "64"),
        ("--profile-target-layer", "1"),
        ("--profile-stop-after-point", "ctx128-n4-t12-skewed"),
    ):
        assert cmd[cmd.index(flag) + 1] == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_default_off_and_profile_worker_selection(tmp_path, monkeypatch, enabled):
    seen = []
    monkeypatch.setattr(large, "run", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(driver, "run", lambda args: seen.append(args) or 0)
    argv = [
        "--plugin-sha",
        "sha",
        "--manifest",
        str(tmp_path),
        "--output-dir",
        str(tmp_path),
        "--stage",
        "profile",
        "--batches",
        "64",
        "--profile-experiment",
        "target-boundaries",
        "--profile-target-layer",
        "1",
    ]
    assert large.main(argv + (["--profile-worker-exit"] if enabled else [])) == 0
    cmd = large.command(seen[-1], 64, tmp_path)
    assert ("--profile-worker-exit" in cmd) == enabled
    assert driver.main(cmd[2:]) == 0 and seen[-1].profile_worker_exit == enabled
    monkeypatch.setattr(
        profile.benchmark,
        "build_engine_kwargs",
        lambda _: {
            "additional_config": {"dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"}}
        },
    )
    kw = profile.profile_engine_kwargs(None, tmp_path, False, "target-boundaries", 1, enabled)
    assert ("worker_cls" in kw) == enabled
    assert kw["additional_config"].get("dspark_profile_worker_exit", False) == enabled
    with pytest.raises(ValueError, match="target-boundaries"):
        profile.profile_engine_kwargs(None, tmp_path, False, "baseline", worker_exit=True)
