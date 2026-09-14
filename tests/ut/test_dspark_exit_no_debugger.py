# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU event-loop/frozen Core method regressions; no debugger or NPU invocation."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_failure import (
    cleanup_observer_source,  # noqa: F401
    fast_bounds,  # noqa: F401
    load_executor,
    methods,
    streaming_facade,
)
from tests.ut.test_dspark_repeated_inputs import import_inputs
from tests.ut.test_dspark_worker_exit import load_source
from tools.dspark import exit_observation_report, startup_cost_profile
from tools.dspark import run_large_batch as large

ROOT = Path(__file__).parents[2]


class EngineDeadError(RuntimeError):
    def __init__(self, **kwargs):
        super().__init__("EngineCore dead")


@pytest.mark.usefixtures("fast_bounds")
@pytest.mark.parametrize("fixed", [False, True])
def test_frozen_output_socket_cancel_and_async_llm_shutdown_order(tmp_path, caplog, fixed):
    # Execute the frozen socket task + consumer + shutdown bodies, not just a
    # bank/helper mock. Only transport/manager are substituted with CPU waiters.
    client_cls = methods(
        "v1/engine/core_client.py",
        "AsyncMPClient",
        ["_ensure_output_queue_task", "get_output_async"],
        {"asyncio": asyncio, "EngineDeadError": EngineDeadError},
    )
    parent_cls = methods(
        "v1/engine/core_client.py", "MPClient", ["_format_exception"], {"EngineDeadError": EngineDeadError}
    )
    client_cls._format_exception = parent_cls._format_exception
    engine_cls = methods(
        "v1/engine/async_llm.py",
        "AsyncLLM",
        ["_run_output_handler", "shutdown"],
        {
            "asyncio": asyncio,
            "envs": NS(VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=16),
            "logger": logging.getLogger("frozen-output-test"),
            "shutdown_prometheus": None,
            "cancel_task_threadsafe": lambda t: t.get_loop().call_soon_threadsafe(t.cancel),
        },
    )
    engine, client = engine_cls(), client_cls()
    facade = streaming_facade(engine, tmp_path)
    propagated, seen = [], []
    engine.log_stats, engine.logger_manager, engine.renderer = False, None, None
    engine.output_processor.propagate_error = propagated.append
    engine.engine_core = client
    client.decoder, client.utility_results = None, {}
    client.outputs_queue = asyncio.Queue()

    async def recv(**kwargs):
        await asyncio.Future()

    client.resources = NS(output_queue_task=None, output_socket=NS(recv_multipart=recv), engine_dead=False)

    def core_shutdown(timeout):
        seen.append(engine.output_handler.done())
        client.resources.engine_dead = True
        facade.loop.call_soon_threadsafe(client.resources.output_queue_task.cancel)
        # Force the socket cancellation to run before AsyncLLM.shutdown reaches
        # its final consumer cancellation. This is the original race window.
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.015), facade.loop).result(timeout=0.3)

    client.shutdown = core_shutdown

    async def start():
        engine._run_output_handler()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    facade.loop.run_until_complete(start())
    if fixed:
        facade.shutdown()
        assert facade.cleanup_result["success"]
        receipt = json.loads((tmp_path / "output-handler-shutdown.json").read_text())
        assert receipt["status"] == "drained" and receipt["cancelled"]
    else:
        facade.loop.run_until_complete(facade.profile_guard.shutdown())
        facade.loop.run_until_complete(asyncio.sleep(0))
        facade.loop.close()
    assert seen == [fixed]
    assert bool(propagated) is (not fixed)
    assert ("output_handler failed" in caplog.text) is (not fixed)
    if not fixed:
        assert isinstance(propagated[0], EngineDeadError)


@pytest.mark.usefixtures("fast_bounds")
@pytest.mark.parametrize(
    "problem", ["pending", "partial", "dead", "task_error", "returned", "cancel_error", "cancel_timeout", "missing"]
)
def test_output_drain_failures_are_not_normal_completion(tmp_path, problem):
    engine = NS(shutdown=lambda timeout: None)
    facade = streaming_facade(engine, tmp_path)
    release = asyncio.Event()

    async def consumer():
        if problem == "task_error":
            raise EngineDeadError()
        if problem == "returned":
            return
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            if problem == "cancel_error":
                raise OSError("cancel failed")
            if problem == "cancel_timeout":
                await release.wait()
            else:
                raise

    task = facade.loop.create_task(consumer())
    engine.output_handler = task
    facade.loop.run_until_complete(asyncio.sleep(0))
    if problem == "pending":
        engine.output_processor.get_num_unfinished_requests = lambda: 1
    if problem == "partial":
        facade.last_batch = {"requests": [{"completed_monotonic": None}], "error": None}
    if problem == "dead":
        engine.errored = True
    if problem == "missing":
        del engine.output_processor
    try:
        result = facade.loop.run_until_complete(facade._stop_profile_output())
        assert not result["success"] and facade.profile_guard.first
        if problem in ("pending", "partial", "missing"):
            assert not task.done() and task.cancelling() == 0  # no premature normal drain
        if problem == "task_error":
            assert "EngineDeadError" in result["error"]
        if problem == "cancel_error":
            assert "cancel failed" in result["error"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
        facade.loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
        facade.loop.close()


@pytest.mark.usefixtures("fast_bounds")
def test_prior_generation_error_and_cleanup_error_both_survive(tmp_path):
    def shutdown(timeout):
        raise OSError("cleanup exception")

    facade = streaming_facade(NS(shutdown=shutdown), tmp_path)
    facade.profile_guard.remember(ValueError("original generation error"))
    original = (tmp_path / "engine-failure.json").read_bytes()
    facade.shutdown()
    assert not facade.cleanup_result["success"]
    assert "cleanup exception" in facade.cleanup_result["error"]
    assert (tmp_path / "engine-failure.json").read_bytes() == original
    assert "original generation error" in facade.cleanup_result["output_handler_shutdown"]["prior_error"]["error"]


@pytest.mark.parametrize("exit_at", [3, 19, 30])
def test_poll_only_never_calls_debugger_or_tool_lookup(tmp_path, monkeypatch, exit_at):
    module = load_source("poll_only", ROOT / "vllm_ascend/diagnostics/dspark_exit_observation.py")
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def forbidden(*args, **kwargs):
        raise AssertionError("No debugger/preflight/procfs access allowed")

    monkeypatch.setattr(module, "native_stack", forbidden)
    monkeypatch.setattr(module, "preflight", forbidden)
    monkeypatch.setattr(module, "proc_state", forbidden)
    monkeypatch.setattr(module.shutil, "which", forbidden)

    class Proc:
        pid = 123

        @property
        def exitcode(self):
            return 0 if clock[0] >= exit_at else None

    result = module.observe_workers([NS(rank=r, proc=Proc()) for r in range(8)], tmp_path, debugger_enabled=False)
    assert result["debugger_enabled"] is False and result["attachment_count"] == 0
    assert result["native_sampling"] == "disabled_by_configuration" and result["native_samples"] == []
    assert result["worker_grace_seconds"] == 25 and result["elapsed_seconds"] <= 20.1
    assert len(result["exits"]) == (8 if exit_at < 20 else 0)


def test_no_debugger_cli_and_engine_config(tmp_path, monkeypatch):
    monkeypatch.setattr(large, "run", lambda args: args)
    monkeypatch.setattr(
        startup_cost_profile.benchmark,
        "build_engine_kwargs",
        lambda parsed: {
            "additional_config": {"dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"}}
        },
    )
    args = [
        "--plugin-sha",
        "a" * 40,
        "--manifest",
        "manifest",
        "--output-dir",
        str(tmp_path),
        "--stage",
        "profile",
        "--batches",
        "64",
        "--profile-experiment",
        "target-boundaries",
        "--profile-worker-exit",
        "--profile-exit-no-debugger",
    ]
    with pytest.raises(SystemExit):
        large.main(args)
    config = large.main(args + ["--profile-exit-observation"])
    assert "--profile-exit-no-debugger" in large.command(config, 64, tmp_path)
    kwargs = startup_cost_profile.profile_engine_kwargs(
        None,
        tmp_path,
        False,
        experiment="target-boundaries",
        worker_exit=True,
        exit_observation=True,
        exit_no_debugger=True,
    )
    assert kwargs["additional_config"]["dspark_profile_exit_debugger"] is False


def test_disabled_native_coverage_is_not_sampling_success(tmp_path):
    root = tmp_path / "runs/b64"
    native = root / "worker-exit/native"
    native.mkdir(parents=True)
    (root / "worker-cleanup.json").write_text(
        json.dumps({"workers": [{"rank": r, "raw_exitcode": 0} for r in range(8)], "forced_cleanup": False})
    )
    (native / "observation.json").write_text(
        json.dumps({"status": "returned", "debugger_enabled": False, "attachment_count": 0, "native_samples": []})
    )
    report = exit_observation_report.report(tmp_path)
    assert report["worker_natural_exit_observed"]
    assert report["native_coverage"] == "DISABLED_BY_CONFIGURATION"
    assert report["formal_acceptance"] == "NOT_EVALUATED"


def test_shell_no_debugger_mode_keeps_original_model_plan(tmp_path):
    shim = tmp_path / "bash"
    shim.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_OUT"\n')
    shim.chmod(0o755)
    output = tmp_path / "arguments"
    subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "tools/dspark/run_dspark_swa_acceptance.sh"),
            "a" * 40,
            "manifest",
            "rzwang",
            "--exit-observation-no-debugger",
        ],
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "ARGS_OUT": str(output)},
        check=True,
    )
    args = output.read_text().splitlines()
    assert "--profile-exit-no-debugger" in args and "--profile-exit-observation" in args
    assert args[args.index("--profile-output-tokens") + 1] == "512"
    assert args[args.index("--core-remote") + 1] == "rzwang"


@pytest.mark.usefixtures("fast_bounds")
@pytest.mark.parametrize("during_cancel", [False, True])
def test_socket_failure_not_yet_consumed_cannot_be_hidden(tmp_path, during_cancel):
    facade = streaming_facade(NS(shutdown=lambda timeout: None), tmp_path)

    async def failed_socket():
        return  # Frozen socket catches/queues an IPC exception then returns normally.

    async def consumer():
        try:
            await asyncio.Future()
        finally:
            if during_cancel:
                facade.engine.engine_core.resources.engine_dead = True

    producer = facade.loop.create_task(failed_socket())
    task = facade.loop.create_task(consumer())
    facade.engine.output_handler = task
    facade.engine.engine_core.resources.output_queue_task = None if during_cancel else producer
    facade.loop.run_until_complete(asyncio.sleep(0))
    result = facade.loop.run_until_complete(facade._stop_profile_output())
    assert not result["success"] and facade.profile_guard.first
    if not during_cancel:
        assert task.cancelling() == 0
    task.cancel()
    facade.loop.run_until_complete(asyncio.gather(task, producer, return_exceptions=True))
    facade.loop.close()


@pytest.mark.parametrize("child_rc,log_error", [(0, False), (0, True), (7, True)])
def test_child_rc_and_strict_log_scan_remain_separate(tmp_path, monkeypatch, child_rc, log_error):
    path, _, _ = import_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(large.suite, "source_gate", lambda _: None)
    monkeypatch.setattr(large.suite, "resources_idle", lambda _: None)

    def child(command, log):
        log.write_text("EngineDeadError: original runtime or cleanup error\n" if log_error else "completed\n")
        return child_rc

    monkeypatch.setattr(large.suite, "logged", child)
    out = tmp_path / "run"
    rc = large.main(
        [
            "--stage",
            "profile",
            "--plugin-sha",
            "a" * 40,
            "--manifest",
            str(path),
            "--output-dir",
            str(out),
            "--batches",
            "64",
            "--profile-experiment",
            "target-boundaries",
            "--profile-worker-exit",
            "--profile-exit-observation",
            "--profile-exit-no-debugger",
        ]
    )
    row = json.loads((out / "stages.json").read_text())[0]
    assert row["rc"] == child_rc
    assert row["log_scan_rc"] == (None if child_rc else int(log_error))
    assert rc == int(bool(child_rc or log_error))
    assert (out / "b64.log").read_text().startswith("EngineDeadError" if log_error else "completed")


def test_executor_forwards_poll_only_without_replacing_core_escalation(tmp_path, monkeypatch):
    implementation = load_executor(monkeypatch)
    calls = []
    native = ModuleType("vllm_ascend.diagnostics.dspark_exit_observation")
    native.observe_workers = lambda *args, **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    parent = implementation.ProfileMultiprocExecutor.__mro__[1]
    monkeypatch.setattr(
        parent, "_ensure_worker_termination", staticmethod(lambda procs: calls.append("core")), raising=False
    )
    executor = implementation.ProfileMultiprocExecutor.__new__(implementation.ProfileMultiprocExecutor)
    executor._profile_exit_observation, executor._profile_exit_debugger = True, False
    executor.workers, executor._profile_directory = [], tmp_path
    executor._ensure_worker_termination([])
    assert calls == [{"debugger_enabled": False}, "core"]


@pytest.mark.usefixtures("fast_bounds")
def test_output_drain_time_is_inside_existing_frontend_deadline(tmp_path):
    from tools.dspark import profile_failure

    guard = profile_failure.ProfileFailureGuard(NS(shutdown=lambda timeout: None), tmp_path)
    result = asyncio.run(
        guard.shutdown(frontend_started=(profile_failure.time.monotonic() - 0.02, "2026-09-14T00:00:00+00:00"))
    )
    assert result["pre_shutdown_elapsed_seconds"] >= 0.02
    assert result["elapsed_seconds"] >= 0.02
    assert result["outer_timeout_seconds"] == pytest.approx(0.15)


@pytest.mark.parametrize("contradiction", [False, True])
def test_requested_no_debugger_control_requires_consistent_receipts(tmp_path, contradiction):
    root = tmp_path / "runs/b64"
    native = root / "worker-exit/native"
    native.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps({"exit_no_debugger": True}))
    (native / "observation.json").write_text(
        json.dumps(
            {
                "status": "returned",
                "debugger_enabled": False,
                "attachment_count": 1 if contradiction else 0,
                "native_samples": [],
                "native_sampling": "disabled_by_configuration",
            }
        )
    )
    (tmp_path / "native-preflight-disabled.json").write_text(
        json.dumps({"debugger_enabled": False, "attachment_count": 0, "attach_preflight": "not_run_by_configuration"})
    )
    assert exit_observation_report.report(tmp_path)["no_debugger_control_valid"] is (not contradiction)
