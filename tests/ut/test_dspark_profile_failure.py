# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU subprocesses and frozen Core lifecycle bodies, not NPU validation."""

import ast
import asyncio
import importlib.util
import inspect
import json
import logging
import multiprocessing
import multiprocessing.connection
import signal
import subprocess
import sys
import threading
import time
import uuid
import weakref
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tools.dspark import performance_stream as stream
from tools.dspark import profile_failure as failure
from tools.dspark import profile_process_guard as process_guard
from tools.dspark import startup_cost_profile as profile

ROOT = Path(__file__).parents[2]


@pytest.fixture(autouse=True)
def cleanup_observer_source(monkeypatch):
    name = "vllm_ascend.diagnostics.dspark_cleanup"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_ascend/diagnostics/dspark_cleanup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)


def core_path(relative):
    spec = importlib.util.find_spec("vllm")
    root = Path(next(iter(spec.submodule_search_locations))) if spec else ROOT.parent / "vllm-hust/vllm"
    path = root / relative
    if not path.is_file():
        pytest.skip(f"Frozen Core source unavailable: {path}")
    return path


def methods(relative, name, names, namespace):
    path = core_path(relative)
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in cls.body} == set(names)
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, cls], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def fast_bounds(monkeypatch):
    monkeypatch.setattr(failure, "POLL_SECONDS", 0.005)
    monkeypatch.setattr(failure, "CANCEL_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(failure, "CLEANUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(failure, "CLEANUP_FINALIZE_SECONDS", 0.10)
    monkeypatch.setattr(stream, "CANCEL_TIMEOUT_SECONDS", 0.02)


def frozen_process_shutdown(namespace):
    path = core_path("v1/utils.py")
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "shutdown")
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["shutdown"]


@pytest.mark.parametrize("forced", [False, True])
def test_frozen_shutdown_near_inner_deadline_has_finalization_margin(tmp_path, fast_bounds, caplog, forced):
    logger = logging.getLogger("vllm.v1.utils")
    caplog.set_level(logging.WARNING, logger=logger.name)
    killed = []

    class Process:
        pid = 123
        alive = True

        def terminate(self):
            pass  # Core's normal initial termination request

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            time.sleep(timeout + 0.015)  # old outer=inner budget loses this return
            self.alive = forced

    proc = Process()
    core_shutdown = frozen_process_shutdown(
        {
            "time": time,
            "logger": NS(debug=logger.debug, warning=logger.warning, debug_once=logger.debug),
            "kill_process_tree": lambda pid: killed.append(pid),
        }
    )
    guard = failure.ProfileFailureGuard(NS(shutdown=lambda timeout: core_shutdown([proc], timeout)), tmp_path)
    result = asyncio.run(guard.shutdown())
    assert result["thread_completed"] and result["engine_returned"] and result["shutdown_completed"]
    assert not result["timed_out"]
    assert result["engine_elapsed_seconds"] > failure.CLEANUP_TIMEOUT_SECONDS
    assert result["elapsed_seconds"] < result["outer_timeout_seconds"]
    assert result["forced_cleanup"] == forced and result["success"] == (not forced)
    assert result["status"] == ("forced_cleanup" if forced else "returned")
    assert killed == ([123] if forced else [])
    assert bool(result["force_events"]) == forced
    assert bool(guard.first) == forced  # a force-kill return is not a successful cleanup
    terminal = json.loads((tmp_path / "cleanup-thread.json").read_text())
    assert terminal["engine_returned"] and terminal["returned_utc"]
    assert not terminal["returned_after_outer_deadline"]


def streaming_facade(engine, directory):
    facade = stream.StreamingEngine.__new__(stream.StreamingEngine)
    facade.loop = asyncio.new_event_loop()
    facade.engine = engine
    facade.profile_guard = failure.ProfileFailureGuard(engine, directory)
    facade.cleanup_result = None
    return facade


def test_cleanup_exception_retains_original_generation_failure(tmp_path, fast_bounds):
    def close(timeout):
        raise OSError("actual cleanup exception")

    guard = failure.ProfileFailureGuard(NS(shutdown=close), tmp_path)
    guard.remember(ValueError("original generation NaN"))
    first = (tmp_path / "engine-failure.json").read_bytes()
    result = asyncio.run(guard.shutdown())
    assert result["status"] == "error" and not result["timed_out"] and not result["engine_returned"]
    assert result["thread_completed"] and result["error"] == "OSError: actual cleanup exception"
    assert (tmp_path / "engine-failure.json").read_bytes() == first


def test_stuck_cleanup_retains_loop_and_late_return_has_separate_receipt(tmp_path, fast_bounds):
    release = threading.Event()
    callbacks = []

    def close(timeout):
        release.wait(2)
        facade.loop.call_soon_threadsafe(callbacks.append, "late socket cleanup")

    facade = streaming_facade(NS(shutdown=close), tmp_path)
    try:
        facade.shutdown()
        result = facade.cleanup_result
        assert result["timed_out"] and not result["thread_completed"] and not result["success"]
        assert not facade.loop.is_closed() and result["event_loop"] == "retained_for_supervisor"
        first = (tmp_path / "cleanup.json").read_bytes()
        release.set()
        deadline = time.monotonic() + 1
        while not (tmp_path / "cleanup-thread.json").exists() and time.monotonic() < deadline:
            facade.loop.run_until_complete(asyncio.sleep(0.01))
        assert callbacks == ["late socket cleanup"]
        terminal = json.loads((tmp_path / "cleanup-thread.json").read_text())
        assert terminal["engine_returned"] and terminal["returned_after_outer_deadline"]
        assert (tmp_path / "cleanup.json").read_bytes() == first
    finally:
        release.set()
        facade.loop.run_until_complete(asyncio.sleep(0.02))
        facade.loop.close()


def test_actual_async_llm_shutdown_callbacks_drain_before_loop_close(tmp_path, fast_bounds):
    callbacks = []
    cls = methods(
        "v1/engine/async_llm.py",
        "AsyncLLM",
        ["shutdown"],
        {
            "shutdown_prometheus": None,
            "cancel_task_threadsafe": lambda task: task.get_loop().call_soon_threadsafe(task.cancel),
        },
    )
    engine = cls()
    facade = streaming_facade(engine, tmp_path)

    def core_shutdown(timeout):
        time.sleep(timeout + 0.015)
        facade.loop.call_soon_threadsafe(callbacks.append, "socket cleanup")

    engine.engine_core = NS(shutdown=core_shutdown)

    async def handler():
        try:
            await asyncio.Future()
        finally:
            callbacks.append("output handler cancelled")

    engine.output_handler = facade.loop.create_task(handler())
    facade.loop.run_until_complete(asyncio.sleep(0))
    facade.shutdown()
    assert facade.loop.is_closed() and facade.cleanup_result["success"]
    assert callbacks == ["socket cleanup", "output handler cancelled"]
    assert json.loads((tmp_path / "cleanup.json").read_text())["event_loop"] == "closed"


def test_loop_cancellation_timeout_is_not_success(tmp_path, fast_bounds):
    facade = streaming_facade(NS(shutdown=lambda timeout: None), tmp_path)
    release = asyncio.Event()

    async def resistant():
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()

    task = facade.loop.create_task(resistant())
    facade.loop.run_until_complete(asyncio.sleep(0))
    try:
        facade.shutdown()
        assert not facade.loop.is_closed() and not facade.cleanup_result["success"]
        assert facade.cleanup_result["pending_loop_tasks"] == 1
        assert "cancellation" in facade.profile_guard.first["error"]
    finally:
        release.set()
        facade.loop.run_until_complete(task)
        facade.loop.close()


def test_default_cleanup_budgets_fit_supervisor():
    assert failure.CLEANUP_TIMEOUT_SECONDS == 12
    assert failure.CLEANUP_FINALIZE_SECONDS > 0
    assert process_guard.FAILURE_GRACE_SECONDS == failure.FAILURE_GRACE_SECONDS == 24
    assert (
        2 * failure.CANCEL_TIMEOUT_SECONDS + failure.CLEANUP_TIMEOUT_SECONDS + failure.CLEANUP_FINALIZE_SECONDS
        < failure.FAILURE_GRACE_SECONDS
    )


@pytest.mark.parametrize("state", ["output_handler_done", "engine_dead"])
def test_frozen_engine_state_releases_stranded_utility_and_preserves_first(tmp_path, fast_bounds, state):
    # Real frozen AsyncLLM properties: the public stream is already dead while
    # its independent utility future can still be pending.
    cls = methods("v1/engine/async_llm.py", "AsyncLLM", ["errored", "is_running"], {})

    async def run():
        engine = cls()
        engine.engine_core = NS(resources=NS(engine_dead=False))
        engine.output_handler = asyncio.Future()
        guard = failure.ProfileFailureGuard(engine, tmp_path)
        guard.point = "ctx128-n1-t6-skewed"
        utility = asyncio.Future()
        task = asyncio.create_task(guard.run("collective_rpc:dspark_benchmark_replay_snapshot", utility))
        await asyncio.sleep(0.01)
        if state == "output_handler_done":
            engine.output_handler.set_result(None)
        else:
            engine.engine_core.resources.engine_dead = True
        with pytest.raises(failure.ProfileEngineFailed, match="dead"):
            await asyncio.wait_for(task, timeout=0.5)
        saved = (tmp_path / "engine-failure.json").read_bytes()
        guard.remember(KeyboardInterrupt())
        guard.remember(RuntimeError("cleanup failed"))
        assert (tmp_path / "engine-failure.json").read_bytes() == saved
        assert guard.first["point"] == "ctx128-n1-t6-skewed"
        assert "replay_snapshot" in guard.first["pending_operation"]["name"]
        assert utility.cancelled()
        assert json.loads((tmp_path / "frontend-operation.json").read_text())["state"] == "failed"

    asyncio.run(run())


def test_live_engine_rpc_timeout_and_cancellation_resistance_are_bounded(tmp_path, fast_bounds):
    async def run():
        release = asyncio.Event()
        cancelled = asyncio.Event()

        async def stubborn():
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()

        child = asyncio.create_task(stubborn())
        guard = failure.ProfileFailureGuard(NS(errored=False), tmp_path)
        with pytest.raises(failure.ProfileEngineFailed, match="exceeded"):
            await asyncio.wait_for(guard.run("rpc", child, timeout=0.01), timeout=0.5)
        assert cancelled.is_set() and not child.done()
        release.set()
        await child

    asyncio.run(run())


def test_success_two_points_and_bounded_cleanup(tmp_path, fast_bounds):
    release = threading.Event()
    calls = []

    def shutdown(timeout):
        calls.append(timeout)
        release.wait(2)

    async def run():
        guard = failure.ProfileFailureGuard(NS(errored=False, shutdown=shutdown), tmp_path)
        for point in ("one", "two"):
            guard.point = point
            assert await guard.run("point", asyncio.sleep(0, result=point)) == point
            assert guard.pending is None
        result = await asyncio.wait_for(guard.shutdown(), timeout=0.5)
        assert result["timed_out"] and not result["shutdown_completed"]
        assert calls == [failure.CLEANUP_TIMEOUT_SECONDS]
        assert guard.first["point"] == "two"
        release.set()
        await asyncio.sleep(0.01)
        assert result["timed_out"] and not result["shutdown_completed"]  # detached receipt

    try:
        asyncio.run(run())
    finally:
        release.set()


def load_executor(monkeypatch):
    class Base:
        def __init__(self, vllm_config, monitor_workers=True):
            self.is_failed = self.shutting_down = False
            self.workers = []

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
            self.delegated = (method, timeout, args, kwargs, non_block, unique_reply_rank, kv_output_aggregator)
            return self.delegated

        def shutdown(self):
            self.shutting_down = True
            for handle in self.workers:
                if handle.proc.is_alive():
                    handle.proc.terminate()
                handle.proc.join(timeout=1)

    # Check the actual frozen optional parameter order; a **kwargs fake would
    # conceal interface drift. Execute the real parent monitor below as well.
    real = methods("v1/executor/multiproc_executor.py", "MultiprocExecutor", ["collective_rpc"], {})

    assert list(inspect.signature(real.collective_rpc).parameters) == list(
        inspect.signature(Base.collective_rpc).parameters
    )
    module = ModuleType("vllm.v1.executor.multiproc_executor")
    module.MultiprocExecutor = Base
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location(
        "profile_executor_test", ROOT / "vllm_ascend/diagnostics/dspark_profile_executor.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("exit_mode", ["graceful", "term", "kill"])
def test_frozen_worker_escalation_receipt_controls_frontend_result(
    tmp_path, monkeypatch, fast_bounds, caplog, exit_mode
):
    module = load_executor(monkeypatch)
    logger = logging.getLogger("vllm.v1.executor.multiproc_executor")
    caplog.set_level(logging.WARNING, logger=logger.name)
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    termination = methods(
        "v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        ["_ensure_worker_termination"],
        {
            "time": NS(time=lambda: now[0], sleep=sleep),
            "envs": NS(VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=0.1),
            "logger": NS(info=logger.info, info_once=logger.info, warning=logger.warning),
        },
    )

    class Process:
        pid = 123
        name = "worker"
        exitcode = 0 if exit_mode == "graceful" else None

        def is_alive(self):
            return self.exitcode is None

        def terminate(self):
            if exit_mode == "term":
                self.exitcode = 0  # even a zero exit after forced SIGTERM is an escalation

        def kill(self):
            self.exitcode = -signal.SIGKILL

    def original_shutdown(self):
        self.shutting_down = True
        termination._ensure_worker_termination([h.proc for h in self.workers])

    monkeypatch.setattr(module.ProfileMultiprocExecutor.__mro__[1], "shutdown", original_shutdown)
    executor = module.ProfileMultiprocExecutor(
        NS(
            additional_config={
                "dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"},
                "dspark_profile_failure_dir": str(tmp_path),
            }
        )
    )
    executor.workers = [NS(proc=Process(), rank=0)]
    guard = failure.ProfileFailureGuard(
        NS(shutdown=lambda timeout: executor.shutdown()), tmp_path, require_worker_receipt=True
    )
    result = asyncio.run(guard.shutdown())
    assert result["shutdown_completed"] and result["thread_completed"] and not result["timed_out"]
    assert result["success"] == (exit_mode == "graceful")
    assert result["forced_cleanup"] == (exit_mode != "graceful")
    worker = result["worker_cleanup"]
    assert worker["status"] == "returned"
    assert len(worker["force_events"]) == {"graceful": 0, "term": 1, "kill": 2}[exit_mode]
    assert worker["workers"][0]["raw_exitcode"] == (-9 if exit_mode == "kill" else 0)
    saved = (tmp_path / "worker-cleanup.json").read_bytes()
    executor.shutdown()
    assert (tmp_path / "worker-cleanup.json").read_bytes() == saved


def test_missing_worker_receipt_is_unavailable_not_normal(tmp_path, fast_bounds):
    guard = failure.ProfileFailureGuard(NS(shutdown=lambda timeout: None), tmp_path, require_worker_receipt=True)
    result = asyncio.run(guard.shutdown())
    assert result["shutdown_completed"] and not result["success"]
    assert result["status"] == "worker_cleanup_incomplete"
    assert result["worker_cleanup"]["status"] == "unavailable"


def abrupt_worker():
    signal.raise_signal(signal.SIGTERM)


def test_parent_monitor_captures_actual_exit_before_cleanup(tmp_path, monkeypatch):
    module = load_executor(monkeypatch)
    monitor = methods(
        "v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        ["start_worker_monitor"],
        {
            "multiprocessing": multiprocessing,
            "weakref": weakref,
            "Thread": threading.Thread,
            "logger": logging.getLogger(__name__),
        },
    )
    executor = module.ProfileMultiprocExecutor(
        NS(
            additional_config={
                "dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"},
                "dspark_profile_failure_dir": str(tmp_path),
            }
        )
    )
    child = multiprocessing.get_context("spawn").Process(target=abrupt_worker, name="VllmWorker-5")
    child.start()
    executor.workers = [NS(proc=child, rank=5)]
    executor._profile_point = "ctx128-n1-t6-skewed"
    executor._profile_operation = {"method": "dspark_benchmark_replay_snapshot"}
    callbacks = []
    executor.failure_callback = lambda: callbacks.append("failed")
    try:
        monitor.start_worker_monitor(executor, inline=True)
        receipt = json.loads((tmp_path / "worker-exit.json").read_text())
        worker = receipt["workers"][0]
        assert worker["raw_exitcode"] == -signal.SIGTERM
        assert worker["signal_name"] == "SIGTERM" and worker["rank"] == 5
        assert worker["pid"] == child.pid
        assert receipt["pending_operation"]["method"] == "dspark_benchmark_replay_snapshot"
        assert receipt["point"] == "ctx128-n1-t6-skewed" and callbacks == ["failed"]
        saved = (tmp_path / "worker-exit.json").read_bytes()
        executor.shutdown()
        assert (tmp_path / "worker-exit.json").read_bytes() == saved
        guard = failure.ProfileFailureGuard(NS(), tmp_path)
        guard.remember(RuntimeError("EngineDead"))
        assert guard.first["worker_exit"] == receipt
    finally:
        if child.is_alive():
            child.kill()
        child.join(timeout=2)
        child.close()


def test_executor_normal_shutdown_unavailable_status_and_interface(tmp_path, monkeypatch):
    module = load_executor(monkeypatch)
    executor = module.ProfileMultiprocExecutor(
        NS(
            additional_config={
                "dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"},
                "dspark_profile_failure_dir": str(tmp_path),
            }
        )
    )
    response = executor.collective_rpc(
        "dspark_benchmark_profile_point",
        timeout=7,
        args=(1,),
        kwargs={"point": "p2"},
        unique_reply_rank=3,
        kv_output_aggregator="agg",
    )
    assert response == ("dspark_benchmark_profile_point", 7, (1,), {"point": "p2"}, False, 3, "agg")
    assert executor._profile_point == "p2" and executor._profile_operation is None
    executor.shutdown()
    assert not (tmp_path / "worker-exit.json").exists()
    status = module.process_status(NS(rank=0, proc=NS(pid=12, name="worker", exitcode=None)))
    assert status["raw_exitcode"] is None and status["signal"] is None
    assert status["exit_status"].startswith("unavailable")


def test_process_guard_bounds_stuck_child_preserves_receipt_and_other_process(tmp_path, monkeypatch):
    monkeypatch.setattr(process_guard, "POLL_SECONDS", 0.01)
    directory = tmp_path / "child"
    directory.mkdir()
    first = {"point": "second", "workers": [{"pid": 123, "rank": 5, "raw_exitcode": -9}]}
    (directory / "worker-exit.json").write_text(json.dumps(first))
    outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
    start = time.monotonic()
    try:
        rc = process_guard.supervise(
            [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(20)"],
            directory,
            tmp_path / "supervisor.json",
            grace=0.1,
            term_grace=0.1,
        )
        assert rc == 1 and time.monotonic() - start < 3
        assert outsider.poll() is None
        receipt = json.loads((tmp_path / "supervisor.json").read_text())
        assert receipt["first_failure"]["receipt"] == first
        assert receipt["signals_sent"] == ["SIGTERM", "SIGKILL"]
        assert receipt["raw_returncode"] == -signal.SIGKILL
    finally:
        outsider.terminate()
        outsider.wait(timeout=2)


@pytest.mark.parametrize("rc", [0, 7])
def test_process_guard_records_raw_exit_and_no_false_pass(tmp_path, rc):
    result = process_guard.supervise(
        [sys.executable, "-c", f"raise SystemExit({rc})"], tmp_path, tmp_path / "supervisor.json"
    )
    data = json.loads((tmp_path / "supervisor.json").read_text())
    assert data["raw_returncode"] == rc
    assert result == (1 if rc else 0)
    assert data["signals_sent"] == []


def test_collect_saves_failure_without_postmortem_rpc_or_keyboard_override(tmp_path):
    class Engine:
        def __init__(self):
            self.profile_guard = failure.ProfileFailureGuard(NS(errored=True), tmp_path)
            self.last_batch = {"requests": [{"request_id": "batch2-0", "output_token_ids": [7]}]}
            self.calls = []

        def collective_rpc(self, method, kwargs=None):
            self.calls.append(method)
            if method == "dspark_benchmark_profile_point":
                self.profile_guard.point = kwargs["point"]

        def get_tokenizer(self):
            return NS(encode=lambda *a, **k: [3])

        def generate(self, *args, **kwargs):
            self.profile_guard.remember(RuntimeError("EngineCore is dead"))
            raise failure.ProfileEngineFailed("EngineCore is dead")

        def shutdown(self):
            raise KeyboardInterrupt()

    engine = Engine()
    point = {"id": "second", "lengths": [5], "prompt_tokens": 128, "requests": 1}
    with pytest.raises(failure.ProfileEngineFailed, match="dead"):
        profile.collect(lambda: engine, [point], None, tmp_path, warmup=2, samples=5)
    saved = json.loads((tmp_path / "profile-failure.json").read_text())
    assert "dead" in saved["error"] and "KeyboardInterrupt" not in saved["error"]
    assert engine.calls.count("dspark_benchmark_replay_snapshot") == 1
    assert saved["last_stream"]["requests"][0]["output_token_ids"] == [7]
    assert "KeyboardInterrupt" in json.loads((tmp_path / "lifecycle.json").read_text())["cleanup_error"]


@pytest.mark.parametrize("generation_fails", [False, True])
@pytest.mark.parametrize("cleanup_write_fails", [False, True])
def test_completed_point_evidence_survives_cleanup_failure(
    tmp_path, monkeypatch, generation_fails, cleanup_write_fails
):
    points = profile.diagnostic_points(
        profile.grid(64, [6, 12, 24, 48, 96, 192, 384], [128], 512)[1], "ctx128-n4-t12-skewed"
    )
    monkeypatch.setattr(profile, "point_samples", lambda *a: [])
    monkeypatch.setattr(profile, "validate_point_request_ids", lambda *a: {})

    class Engine:
        def __init__(self):
            self.profile_guard = failure.ProfileFailureGuard(NS(), tmp_path)
            self.last_batch = None

        def collective_rpc(self, method, kwargs=None):
            if kwargs:
                self.point = kwargs["point"]
            return [
                {
                    "cost_profile": {
                        "identity": {},
                        "observation": {
                            "recording_error": None,
                            "numeric": {
                                "enabled": True,
                                "nan_rounds": 0,
                                "compact_host_transfers": 1,
                                "compact_host_transfers_completed": 1,
                            },
                        },
                    }
                }
                for _ in range(8)
            ]

        def get_tokenizer(self):
            return NS(encode=lambda *a, **kw: [1])

        def generate(self, prompts, *a, **kw):
            self.last_batch = {
                "error": None,
                "scheduler": {},
                "requests": [{"output_token_ids": [1] * 512} for _ in prompts],
            }
            if generation_fails:
                self.profile_guard.remember(RuntimeError("original generation NaN"))
                raise RuntimeError("original generation NaN")
            return [None] * len(prompts)

        def shutdown(self):
            self.cleanup_result = {"shutdown_completed": True, "success": False, "status": "forced_cleanup"}
            if cleanup_write_fails:
                write = profile.benchmark._atomic_write_json

                def fail_cleanup_write(path, data):
                    if path.name in ("cleanup-failure.json", "lifecycle.json"):
                        raise OSError("cleanup receipt disk failure")
                    return write(path, data)

                monkeypatch.setattr(profile.benchmark, "_atomic_write_json", fail_cleanup_write)

    engine = Engine()
    with pytest.raises(RuntimeError, match="original generation NaN" if generation_fails else "forced_cleanup"):
        profile.collect(lambda: engine, points, None, tmp_path, warmup=2, samples=5)
    progress = json.loads((tmp_path / "point-completion.json").read_text())
    assert progress["status"] == ("failed" if generation_fails else "completed")
    assert len(progress["completed_points"]) == (0 if generation_fails else 10)
    if not cleanup_write_fails:
        cleanup = json.loads((tmp_path / "cleanup-failure.json").read_text())
        assert "forced_cleanup" in cleanup["error"]
        lifecycle = json.loads((tmp_path / "lifecycle.json").read_text())
        assert not lifecycle["shutdown"] and lifecycle["points_status"] == progress["status"]
        assert ("original generation NaN" in (cleanup["prior_error"] or "")) == generation_fails
    if generation_fails:
        assert "original generation NaN" in json.loads((tmp_path / "profile-failure.json").read_text())["error"]
    else:
        assert progress["completed_points"][-1]["generated_tokens"] == [512] * 4
        for item in progress["completed_points"]:
            assert item["numeric_result"] == "no_nan_observed_at_enabled_boundaries"
            assert profile.benchmark._sha256_file(tmp_path / f"{item['point']}.json") == item["raw_sha256"]
        assert not (tmp_path / "profile-failure.json").exists()
    assert not (tmp_path / "cost-profile.json").exists()


def test_frozen_output_socket_failure_leaves_utility_pending_guard_releases_it(tmp_path, fast_bounds):
    client_cls = methods(
        "v1/engine/core_client.py",
        "AsyncMPClient",
        [
            "_call_utility_async",
            "_ensure_output_queue_task",
            "get_output_async",
        ],
        {
            "asyncio": asyncio,
            "weakref": weakref,
            "uuid": uuid,
            "EngineCoreRequestType": NS(UTILITY=NS(value=b"utility")),
            "EngineDeadError": RuntimeError,
        },
    )
    engine_cls = methods("v1/engine/async_llm.py", "AsyncLLM", ["errored", "is_running"], {})
    parent = methods("v1/engine/core_client.py", "MPClient", ["_format_exception"], {})
    client_cls._format_exception = parent._format_exception

    async def run():
        async def socket_failure(copy):
            raise RuntimeError("EngineDead original output-socket failure")

        async def send(*args):
            pass

        client = client_cls()
        client.resources = NS(
            output_queue_task=None, output_socket=NS(recv_multipart=socket_failure), engine_dead=False
        )
        client.client_index = 0
        client.encoder = NS(encode=lambda x: [b"encoded"])
        client.decoder = None
        client.utility_results = {}
        client.outputs_queue = asyncio.Queue()
        client._send_input_message = send
        utility = asyncio.create_task(client._call_utility_async("collective_rpc", engine=b"core"))

        async def consume_error():
            with pytest.raises(RuntimeError, match="EngineDead original"):
                await client.get_output_async()

        engine = engine_cls()
        engine.engine_core = client
        engine.output_handler = asyncio.create_task(consume_error())
        await engine.output_handler
        assert engine.errored
        assert len(client.utility_results) == 1
        assert not utility.done() and not next(iter(client.utility_results.values())).done()
        guard = failure.ProfileFailureGuard(engine, tmp_path)
        with pytest.raises(failure.ProfileEngineFailed, match="dead"):
            await asyncio.wait_for(guard.run("collective_rpc", utility), timeout=0.5)
        assert utility.cancelled()

    asyncio.run(run())


def test_guarded_child_failure_finishes_tee_and_keeps_pipestatus(tmp_path):
    from tools.dspark.run_performance_suite import logged

    log = tmp_path / "child.log"
    command = [
        sys.executable,
        "-m",
        "tools.dspark.profile_process_guard",
        "--directory",
        str(tmp_path / "engine"),
        "--receipt",
        str(tmp_path / "supervisor.json"),
        "--",
        sys.executable,
        "-c",
        "print('worker failure retained'); raise SystemExit(7)",
    ]
    assert logged(command, log) != 0
    assert "worker failure retained" in log.read_text()
    assert log.with_suffix(".pipestatus").read_text().strip() == "1 0"
    assert json.loads((tmp_path / "supervisor.json").read_text())["raw_returncode"] == 7


@pytest.mark.parametrize("trigger", ["deadline", "stop", "first_error_then_stop"])
def test_optional_runtime_bound_and_controlled_stop_keep_first_error(tmp_path, monkeypatch, trigger):
    monkeypatch.setattr(process_guard, "POLL_SECONDS", 0.005)
    stop = tmp_path / "STOP"
    if trigger != "deadline":
        stop.touch()
    if trigger == "first_error_then_stop":
        (tmp_path / "engine-failure.json").write_text(json.dumps({"error": "original NaN"}))
    rc = process_guard.supervise(
        [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"],
        tmp_path,
        tmp_path / "supervisor.json",
        grace=0.15,
        term_grace=0.1,
        max_runtime=0.03,
        stop_file=stop,
    )
    receipt = json.loads((tmp_path / "supervisor.json").read_text())
    assert rc == 1 and receipt["raw_returncode"] is not None
    expected = {
        "deadline": "diagnostic runtime bound",
        "stop": "controlled stop file",
        "first_error_then_stop": "engine-failure.json",
    }
    assert receipt["first_failure"]["source"] == expected[trigger]
    assert receipt["signals_sent"][0] == "SIGTERM"
    if trigger == "first_error_then_stop":
        assert receipt["first_failure"]["receipt"]["error"] == "original NaN"


@pytest.mark.parametrize("main_rc,archive_rc,expected", [(7, 9, 7), (7, 0, 7), (0, 9, 9), (0, 0, 0)])
def test_real_shell_export_footer_preserves_first_exit(tmp_path, main_rc, archive_rc, expected):
    # Execute the actual footer with local shell stubs, never /workspace setup.
    import subprocess

    source = (ROOT / "tools/dspark/run_dspark_large_batch.sh").read_text()
    footer = source[source.index('main "$@"\n') :]
    script = tmp_path / "footer.sh"
    script.write_text(
        "set -o pipefail\nCONF_OUT=$1\n"
        f"main() {{ return {main_rc}; }}\n"
        f"tar() {{ return {archive_rc}; }}\n"
        'sha256sum() { printf "mock hash\\n"; }\n' + footer
    )
    result = subprocess.run(["bash", str(script), str(tmp_path)], capture_output=True, text=True, timeout=5)
    assert result.returncode == expected
    assert f"MAIN_RC={main_rc}" in (tmp_path / "status.txt").read_text()
