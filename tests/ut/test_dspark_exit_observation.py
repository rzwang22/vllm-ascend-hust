# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host budgets/subprocess protocol tests; real Linux ptrace remains a server preflight."""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_failure import cleanup_observer_source, load_executor  # noqa: F401
from tests.ut.test_dspark_swa_acceptance import model_fixture
from tests.ut.test_dspark_worker_exit import load_source
from tools.dspark import exit_observation_report, profile_failure, swa_acceptance
from tools.dspark import run_large_batch as large

ROOT = Path(__file__).parents[2]


@pytest.fixture
def module():
    return load_source("native_exit_test", ROOT / "vllm_ascend/diagnostics/dspark_exit_observation.py")


@pytest.fixture
def fake_native(module, tmp_path, monkeypatch):
    # Real debugger-controller subprocess, mocked procfs only. It does not ptrace.
    proc = tmp_path / "proc" / "123"
    proc.mkdir(parents=True)
    (proc / "maps").write_text("native library mapping\n")
    executable = tmp_path / "gdb"
    monkeypatch.setattr(module.shutil, "which", lambda _: str(executable))
    original_path = Path
    monkeypatch.setattr(module, "Path", lambda p: tmp_path / "proc" if str(p) == "/proc" else original_path(p))
    monkeypatch.setattr(
        module, "proc_state", lambda pid: {"pid": pid, "start_ticks": "17", "state": "R", "tracer_pid": 0}
    )
    signals = []
    monkeypatch.setattr(module, "os", NS(read=os.read, kill=lambda pid, sig: signals.append((pid, sig))))

    def program(body, *, python=True):
        executable.write_text(f"#!{sys.executable if python else '/bin/sh'}\n" + body)
        executable.chmod(0o700)

    return program, signals


def test_native_controller_captures_bounded_file_and_detaches(module, fake_native, tmp_path):
    program, signals = fake_native
    program('print("#0 fake_native_frame")\n')
    result = module.native_stack(123, tmp_path / "result", "one")
    assert result["status"] == "captured" and result["detached"]
    assert result["debugger_returncode"] == 0 and not signals
    assert "#0 fake_native_frame" in (tmp_path / "result/one.stack.txt").read_text()
    assert result["pause_upper_bound_seconds"] >= 0
    assert result == json.loads((tmp_path / "result/one.json").read_text())


def test_timeout_keeps_partial_stack_kills_only_debugger_and_records_resume(module, fake_native, tmp_path, monkeypatch):
    program, signals = fake_native
    # Avoid measuring Python interpreter startup in this sub-second timeout test.
    # exec keeps sleep in the owned debugger PID (no orphan child).
    program("printf '#0 slow_frame\\n'\nexec sleep 10\n", python=False)
    n = [0]

    def state(pid):
        n[0] += 1
        return {"pid": pid, "start_ticks": "17", "state": "T" if n[0] > 1 and not signals else "R", "tracer_pid": 0}

    monkeypatch.setattr(module, "proc_state", state)
    start = time.monotonic()
    result = module.native_stack(123, tmp_path / "result", "timeout", timeout=0.8)
    assert result["timed_out"] and result["status"] == "unavailable"
    assert result["detached"] and result["debugger_returncode"] == -signal.SIGKILL
    assert signals == [(123, signal.SIGCONT)]
    assert time.monotonic() - start < 2
    assert (tmp_path / "result/timeout.stack.txt").read_text().startswith("#0")
    assert (tmp_path / "result/timeout.json").exists()


@pytest.mark.parametrize("state,tracer", [("T", 0), ("R", 500)])
def test_never_attach_or_resume_preexisting_stop_or_tracer(module, fake_native, tmp_path, monkeypatch, state, tracer):
    _, signals = fake_native
    monkeypatch.setattr(
        module, "proc_state", lambda pid: {"pid": pid, "start_ticks": "17", "state": state, "tracer_pid": tracer}
    )
    result = module.native_stack(123, tmp_path, "refused")
    assert result["status"] == "unavailable" and "debugger_pid" not in result
    assert not signals


def test_missing_tool_and_procfs_errors_are_saved(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "proc_state", lambda pid: {"exited": True})
    result = module.native_stack(123, tmp_path, "gone")
    assert result["status"] == "unavailable" and result["detached"]
    monkeypatch.setattr(module.sys, "platform", "not-linux")
    result = module.preflight(tmp_path / "preflight")
    assert result["status"] == "failed" and "Linux" in result["error"]
    assert (tmp_path / "preflight/preflight.json").exists()


def test_output_limit_no_unbounded_stack(module, fake_native, tmp_path, monkeypatch):
    program, _ = fake_native
    program('print("x" * 10000)\n')
    monkeypatch.setattr(module, "MAX_STACK_BYTES", 1024)
    # Child exits on its own; do not fake a long-running tracer in this case.
    result = module.native_stack(123, tmp_path / "result", "large")
    assert result.get("output_truncated")
    assert (tmp_path / "result/large.stack.txt").stat().st_size == 1024
    assert result["status"] == "unavailable"


@pytest.mark.parametrize("exits_at", [3, 10, 30])
def test_shared_observation_clock_sample_limit_and_real_exit_codes(module, tmp_path, monkeypatch, exits_at):
    now, samples = [0.0], []
    monkeypatch.setattr(module, "time", NS(monotonic=lambda: now[0], sleep=lambda t: now.__setitem__(0, now[0] + t)))

    class Proc:
        pid = 123

        @property
        def exitcode(self):
            return 0 if now[0] >= exits_at else None

    handles = [NS(rank=r, proc=Proc()) for r in range(8)]

    def capture(pid, directory, label, **kw):
        samples.append((now[0], label))
        now[0] += 0.1
        return {"status": "captured", "detached": True}

    monkeypatch.setattr(module, "native_stack", capture)
    result = module.observe_workers(handles, tmp_path)
    assert now[0] <= module.OBSERVE_SECONDS + 0.1
    assert len(samples) <= 4
    assert all(when < 20 for when, _ in samples)
    if exits_at < 20:
        assert len(result["exits"]) == 8 and all(x["raw_exitcode"] == 0 for x in result["exits"].values())
    else:
        assert not result["exits"]
    assert bool(samples) == (exits_at >= 8)


@pytest.mark.parametrize("enabled", [False, True])
def test_executor_delegates_core_termination_after_only_opt_in_observation(monkeypatch, tmp_path, enabled):
    implementation = load_executor(monkeypatch)
    calls = []
    parent = implementation.ProfileMultiprocExecutor.__mro__[1]
    monkeypatch.setattr(
        parent, "_ensure_worker_termination", staticmethod(lambda procs: calls.append("core")), raising=False
    )
    fake = ModuleType("vllm_ascend.diagnostics.dspark_exit_observation")
    fake.observe_workers = lambda handles, directory: calls.append("observe")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    vllm = ModuleType("vllm")
    vllm.envs = NS(VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=5)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    executor = implementation.ProfileMultiprocExecutor.__new__(implementation.ProfileMultiprocExecutor)
    executor._profile_exit_observation, executor.workers, executor._profile_directory = enabled, [], tmp_path
    executor._ensure_worker_termination([])
    assert calls == (["observe", "core"] if enabled else ["core"])


@pytest.mark.parametrize("enabled", [False, True])
def test_frontend_budget_and_outer_margin(enabled, tmp_path, monkeypatch):
    passed = []
    engine = NS(shutdown=lambda timeout: passed.append(timeout))
    guard = profile_failure.ProfileFailureGuard(engine, tmp_path, exit_observation=enabled)
    result = asyncio.run(guard.shutdown())
    assert passed == [36 if enabled else 12]
    assert result["outer_timeout_seconds"] == (40 if enabled else 16)
    assert result["supervisor_failure_grace_seconds"] == (48 if enabled else 24)


def test_all_budget_layers_compatible(module):
    assert module.OBSERVE_SECONDS + 5 + 4 + 1 < profile_failure.EXIT_OBSERVATION_ENGINE_SECONDS
    assert (
        profile_failure.EXIT_OBSERVATION_ENGINE_SECONDS + 4 + 2 + 6
        == profile_failure.EXIT_OBSERVATION_SUPERVISOR_SECONDS
    )
    assert module.STACK_AT_SECONDS[-1] + module.MAX_RANKS * (module.STACK_TIMEOUT_SECONDS + module.DETACH_SECONDS) < 20


@pytest.mark.parametrize("enabled", [False, True])
def test_extended_window_never_counts_as_original_acceptance(tmp_path, enabled):
    root = tmp_path / "b64"
    model_fixture(root)
    p = root / "plan.json"
    plan = json.loads(p.read_text())
    plan["exit_observation"] = enabled
    p.write_text(json.dumps(plan))
    result = swa_acceptance.model_report(root, 0)
    assert result["numerical_and_FULL_acceptance"] == "PASSED_THIS_RUN"
    assert result["worker_natural_exit"]
    assert result["overall_pass"] is (not enabled)
    if enabled:
        assert result["original_budget_acceptance"] == "NOT_EVALUATED_BY_THIS_DIAGNOSTIC"


@pytest.mark.parametrize(
    "natural,samples,status",
    [
        (True, [], "NATURAL_EXIT_NO_DEBUGGER"),
        (True, [{"status": "captured", "detached": True}], "NATURAL_EXIT_DEBUGGER_AFFECTED"),
        (False, [], "FORCED_OR_INCOMPLETE"),
    ],
)
def test_independent_report_labels_limits(tmp_path, natural, samples, status):
    (tmp_path / "model-acceptance.json").write_text(json.dumps({"natural_exit_with_extended_budget": natural}))
    p = tmp_path / "runs/b64/worker-exit/native"
    p.mkdir(parents=True)
    (p / "observation.json").write_text(json.dumps({"native_samples": samples, "status": "returned"}))
    (tmp_path / "runs/b64/worker-cleanup.json").write_text(
        json.dumps(
            {
                "workers": [{"rank": r, "raw_exitcode": 0 if natural else -15} for r in range(8)],
                "forced_cleanup": not natural,
            }
        )
    )
    result = exit_observation_report.report(tmp_path)
    assert result["status"] == status and result["formal_acceptance"] == "NOT_EVALUATED"


def test_exit_option_requires_profile_worker_and_does_not_enable_heavy_capture():
    source = (ROOT / "tools/dspark/run_dspark_swa_acceptance.sh").read_text()
    assert "--profile-operator-capture" not in source and "--profile-write-timeline" not in source
    with pytest.raises(SystemExit):
        large.main(["--profile-exit-observation"])
    # Both child flags are forwarded; exact runnable argv is covered by existing entry tests.
    source = (ROOT / "tools/dspark/run_large_batch.py").read_text()
    assert '*(["--exit-observation"]' in source
    assert '*(["--profile-exit-observation"]' in source


def test_extended_frontend_hard_bound_preserves_prior_failure(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(profile_failure, "EXIT_OBSERVATION_ENGINE_SECONDS", 0.02)
    monkeypatch.setattr(profile_failure, "CLEANUP_FINALIZE_SECONDS", 0.02)
    monkeypatch.setattr(profile_failure, "POLL_SECONDS", 0.002)
    guard = profile_failure.ProfileFailureGuard(
        NS(shutdown=lambda timeout: release.wait(2)), tmp_path, exit_observation=True
    )
    guard.remember(RuntimeError("original generation failure"))
    prior = dict(guard.first)
    started = time.monotonic()
    try:
        result = asyncio.run(guard.shutdown())
        assert result["timed_out"] and not result["success"]
        assert result["outer_timeout_seconds"] == 0.04
        assert guard.first == prior and time.monotonic() - started < 1
    finally:
        release.set()


def test_preflight_rejects_timeout_before_actual_attach(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(module.shutil, "which", lambda _: "/fake/gdb")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "GDB fake\n")
    monkeypatch.setattr(
        module,
        "native_stack",
        lambda *a, **kw: {"status": "captured", "detached": True, "timed_out": True, "attached_observed": False},
    )
    result = module.preflight(tmp_path)
    assert result["status"] == "failed"
    assert "Forced debugger-timeout detach not verified" in result["error"]


def test_real_cli_forwards_only_explicit_observation(monkeypatch, tmp_path):
    monkeypatch.setattr(large, "run", lambda args: args)
    args = [
        "--stage",
        "profile",
        "--plugin-sha",
        "a" * 40,
        "--manifest",
        "manifest",
        "--output-dir",
        str(tmp_path),
        "--profile-experiment",
        "target-boundaries",
        "--batches",
        "64",
    ]
    with pytest.raises(SystemExit):
        large.main(args + ["--profile-exit-observation"])
    config = large.main(args + ["--profile-worker-exit", "--profile-exit-observation"])
    assert config.profile_exit_observation
    assert "--profile-exit-observation" in large.command(config, 64, tmp_path)
    config = large.main(args + ["--profile-worker-exit"])
    assert "--profile-exit-observation" not in large.command(config, 64, tmp_path)


def test_actual_wrapper_explicit_observation_keeps_original_inputs(tmp_path):
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
            "--exit-observation",
        ],
        check=True,
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "ARGS_OUT": str(output)},
    )
    args = output.read_text().splitlines()
    assert "--profile-exit-observation" in args
    assert args[args.index("--profile-output-tokens") + 1] == "512"
    assert args[args.index("--core-remote") + 1] == "rzwang"


def test_diagnostic_recording_error_still_runs_original_termination(monkeypatch, tmp_path):
    implementation = load_executor(monkeypatch)
    called = []
    parent = implementation.ProfileMultiprocExecutor.__mro__[1]
    monkeypatch.setattr(
        parent, "_ensure_worker_termination", staticmethod(lambda procs: called.append("core")), raising=False
    )
    fake = ModuleType("vllm_ascend.diagnostics.dspark_exit_observation")

    def fail(*args):
        raise OSError("disk full")

    fake.observe_workers = fail
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    executor = implementation.ProfileMultiprocExecutor.__new__(implementation.ProfileMultiprocExecutor)
    executor._profile_exit_observation, executor.workers, executor._profile_directory = True, [], tmp_path
    executor._ensure_worker_termination([])
    assert called == ["core"] and executor._profile_native_error == "OSError: disk full"


def test_natural_codes_without_observation_do_not_claim_no_debugger(tmp_path):
    path = tmp_path / "runs/b64"
    path.mkdir(parents=True)
    (path / "worker-cleanup.json").write_text(
        json.dumps(
            {
                "workers": [{"rank": r, "raw_exitcode": 0} for r in range(8)],
                "forced_cleanup": False,
                "exit_observation_error": "write unavailable",
            }
        )
    )
    result = exit_observation_report.report(tmp_path)
    assert result["status"] == "NATURAL_EXIT_OBSERVATION_UNAVAILABLE"
    assert result["worker_natural_exit_observed"] and result["native_coverage"] == "UNAVAILABLE"


@pytest.mark.parametrize("driver_rc,have_model_archive", [(0, True), (7, True), (7, False)])
def test_outer_entry_archives_logs_pipestatus_and_preserves_driver_error(tmp_path, driver_rc, have_model_archive):
    workspace = tmp_path / "workspace"
    plugin = workspace / "vllm-ascend-hust"
    scripts = plugin / "tools/dspark"
    scripts.mkdir(parents=True)
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "git").write_text(
        '#!/bin/sh\ncase "$*" in "rev-parse HEAD") echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;; esac\n'
    )
    (binary / "sha256sum").write_text('#!/bin/sh\nshasum -a 256 "$@"\n')
    for path in binary.iterdir():
        path.chmod(0o755)
    model = workspace / "dspark-results/dspark-large-batch.TEST"
    model.mkdir(parents=True)
    original = b"immutable model archive bytes"
    if have_model_archive:
        Path(str(model) + "-evidence.tar.gz").write_bytes(original)
        Path(str(model) + "-evidence.sha256").write_text(hashlib.sha256(original).hexdigest() + "\n")
    (scripts / "run_dspark_swa_acceptance.sh").write_text(
        f"#!/bin/sh\nprintf 'SERVER_RESULT_DIR=%s\\n' '{model}'\nexit {driver_rc}\n"
    )
    entry = tmp_path / "entry.sh"
    entry.write_text(
        (ROOT / "tools/dspark/run_dspark_exit_observation.sh").read_text().replace("/workspace", str(workspace))
    )
    run = subprocess.run(
        ["/bin/bash", str(entry), "a" * 40, "manifest", "rzwang"],
        env={**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"]},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert run.returncode == driver_rc, run.stdout + run.stderr
    archive = next((workspace / "dspark-results").glob("dspark-exit-observation.*-evidence.tar.gz"))
    with tarfile.open(archive) as saved:

        def content(suffix):
            return saved.extractfile(next(m for m in saved.getmembers() if m.name.endswith(suffix))).read()

        assert content("/driver.pipestatus").strip() == f"{driver_rc} 0".encode()
        assert f"MAIN_RC={driver_rc}".encode() in content("/status.txt")
        assert b"SERVER_RESULT_DIR=" in content("/driver.log")
        if have_model_archive:
            assert content("/model-evidence.tar.gz") == original
        else:
            assert b"unavailable" in content("/export-error.txt")
    checksum = Path(str(archive).replace(".tar.gz", ".sha256")).read_text().split()[0]
    assert checksum == hashlib.sha256(archive.read_bytes()).hexdigest()
