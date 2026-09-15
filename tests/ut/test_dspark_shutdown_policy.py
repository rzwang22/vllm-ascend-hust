# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host regressions of the real entry/guard/Core wait; no model or NPU claims."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_failure import cleanup_observer_source, load_executor, methods  # noqa: F401
from tests.ut.test_dspark_swa_acceptance import model_fixture
from tools.dspark import profile_failure, profile_process_guard, swa_acceptance
from tools.dspark import shutdown_policy as policy

ROOT = Path(__file__).parents[2]


def test_child_environment_is_scoped_and_defaults_unchanged():
    cmd = [sys.executable, "-c", f"import os; print(os.environ.get({policy.CORE_WORKER_ENV!r}))"]
    before = os.environ.get(policy.CORE_WORKER_ENV)
    assert subprocess.check_output(policy.child_command(policy.POLICY_NAME, cmd), text=True).strip() == "25"
    assert os.environ.get(policy.CORE_WORKER_ENV) == before
    assert policy.child_command(None, cmd) is cmd
    assert policy.budget(None) is None
    assert profile_failure.CLEANUP_TIMEOUT_SECONDS == 12
    assert profile_failure.FAILURE_GRACE_SECONDS == 24


@pytest.mark.parametrize(
    "name,observation,worker",
    [("unknown", False, True), (policy.POLICY_NAME, True, True), (policy.POLICY_NAME, False, False)],
)
def test_invalid_policy_rejected(name, observation, worker):
    with pytest.raises(ValueError):
        policy.budget(name, exit_observation=observation, worker_exit=worker)


@pytest.mark.parametrize("actual", [5, 25])
def test_effective_core_accessor_required(monkeypatch, actual):
    core = ModuleType("vllm")
    core.envs = NS(**{policy.CORE_WORKER_ENV: actual})
    monkeypatch.setitem(sys.modules, "vllm", core)
    if actual != 25:
        with pytest.raises(ValueError, match="Effective Core"):
            policy.installed_budget(policy.POLICY_NAME)
    else:
        assert policy.installed_budget(policy.POLICY_NAME)["effective_core_worker_seconds"] == 25


@pytest.mark.parametrize("grace,exit_at", [(5, 12.8), (25, 12.8), (25, None)])
def test_actual_core_shared_wait_and_force_deadlines(grace, exit_at):
    now = [0.0]
    events = []

    class Process:
        pid = 123

        def is_alive(self):
            return exit_at is None or now[0] < exit_at

        def terminate(self):
            events.append(("TERM", now[0]))

        def kill(self):
            events.append(("KILL", now[0]))

    clock = NS(time=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    cls = methods(
        "v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        ["_ensure_worker_termination"],
        {
            "time": clock,
            "envs": NS(**{policy.CORE_WORKER_ENV: grace}),
            "logger": NS(info=lambda *a: None, warning=lambda *a: None, info_once=lambda *a: None),
        },
    )
    cls._ensure_worker_termination([Process() for _ in range(8)])
    if grace == 25 and exit_at:
        assert not events and 12.8 <= now[0] < 13
    else:
        assert len(events) == 16
        assert all(grace <= t < grace + 0.2 for kind, t in events if kind == "TERM")
        assert all(grace + 4 <= t < grace + 4.3 for kind, t in events if kind == "KILL")


def test_executor_named_policy_has_no_observation_prewait(monkeypatch):
    module = load_executor(monkeypatch)
    cls = module.ProfileMultiprocExecutor
    parent = cls.__mro__[1]
    called = []
    monkeypatch.setattr(parent, "_ensure_worker_termination", staticmethod(lambda p: called.append(p)), raising=False)
    executor = cls.__new__(cls)
    executor._profile_exit_observation = False
    executor._profile_shutdown_budget = policy.budget(policy.POLICY_NAME)
    fake = ModuleType("vllm_ascend.diagnostics.dspark_exit_observation")
    fake.observe_workers = lambda *a, **k: pytest.fail("No observation or native attach allowed")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    procs = [object()]
    executor._ensure_worker_termination(procs)
    assert called == [procs]


def test_frontend_actual_shutdown_uses_inner_outer_budgets(tmp_path):
    calls = []
    engine = NS(shutdown=lambda timeout: calls.append(timeout))
    (tmp_path / "worker-cleanup.json").write_text(
        json.dumps(
            {
                "forced_cleanup": False,
                "recording_error": None,
                "workers": [{"rank": r, "raw_exitcode": 0} for r in range(8)],
            }
        )
    )
    guard = profile_failure.ProfileFailureGuard(
        engine, tmp_path, require_worker_receipt=True, shutdown_policy=policy.POLICY_NAME
    )
    result = asyncio.run(guard.shutdown())
    assert calls == [36]
    assert result["timeout_seconds"] == 36 and result["outer_timeout_seconds"] == 40
    assert result["supervisor_failure_grace_seconds"] == 48
    assert result["exit_observation"] is False


def test_supervisor_real_child_and_policy_receipt(tmp_path):
    output = tmp_path / "supervisor.json"
    assert (
        profile_process_guard.supervise(
            [sys.executable, "-c", "pass"], tmp_path, output, shutdown_policy=policy.POLICY_NAME
        )
        == 0
    )
    data = json.loads(output.read_text())
    assert data["failure_grace_seconds"] == 48 and data["raw_returncode"] == 0
    assert data["signals_sent"] == [] and data["owned_group_remaining"] is False


def configured_model(root):
    model_fixture(root)

    def update(path, **fields):
        data = json.loads(path.read_text()) if path.exists() else {}
        data.update(fields)
        path.write_text(json.dumps(data))

    expected = policy.budget(policy.POLICY_NAME)
    update(root / "plan.json", shutdown_policy=policy.POLICY_NAME, exit_observation=False)
    update(
        root / "cleanup.json",
        shutdown_budget=expected,
        timeout_seconds=36,
        outer_timeout_seconds=40,
        event_loop="closed",
        error=None,
        recording_error=None,
    )
    update(
        root / "worker-cleanup.json",
        shutdown_budget={**expected, "effective_core_worker_seconds": 25},
        exit_observation=False,
        reap={"budget_seconds": 1, "workers": [{"rank": r, "raw_exitcode": 0, "status": "reaped"} for r in range(8)]},
    )
    update(root / "output-handler-shutdown.json", success=True, error=None)
    update(
        root.parent / "b64-supervisor.json",
        shutdown_budget=expected,
        failure_grace_seconds=48,
        raw_returncode=0,
        first_failure=None,
        signals_sent=[],
        owned_group_remaining=False,
    )
    update(root.parent / "b64-command.json", rc=0, log_scan_rc=0)
    update(root.parent / "b64-residual.json", success=True, error=None)
    return update


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "budget",
        "missing",
        "null",
        "timeout",
        "forced",
        "residual",
        "scan",
        "cancel",
        "unreaped",
        "prior_error",
        "observation",
    ],
)
def test_policy_acceptance_requires_real_evidence_and_preserves_numeric(tmp_path, problem):
    root = tmp_path / "b64"
    update = configured_model(root)
    if problem == "budget":
        update(root / "worker-cleanup.json", shutdown_budget={})
    elif problem == "missing":
        (root.parent / "b64-supervisor.json").unlink()
    elif problem == "null":
        update(
            root / "worker-cleanup.json", workers=[{"rank": r, "raw_exitcode": None if r == 0 else 0} for r in range(8)]
        )
    elif problem == "timeout":
        update(root / "cleanup.json", timed_out=True)
    elif problem == "forced":
        update(root / "cleanup.json", forced_cleanup=True)
    elif problem == "residual":
        update(root.parent / "b64-residual.json", success=False)
    elif problem == "scan":
        update(root.parent / "b64-command.json", log_scan_rc=1)
    elif problem == "cancel":
        update(root / "output-handler-shutdown.json", success=False, error="cancel failed")
    elif problem == "unreaped":
        update(root / "worker-cleanup.json", reap={})
    elif problem == "prior_error":
        update(root.parent / "b64-supervisor.json", first_failure={"error": "EngineDeadError"})
    elif problem == "observation":
        update(root / "plan.json", exit_observation=True)
    result = swa_acceptance.model_report(root, 0)
    assert result["numerical_and_FULL_acceptance"] == "PASSED_THIS_RUN"
    assert result["overall_pass"] is (problem is None)
    if problem != "observation":
        assert result["original_budget_acceptance"] == "NOT_EVALUATED"


def test_real_shell_policy_forwarding_keeps_model_inputs(tmp_path):
    shim = tmp_path / "bash"
    shim.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_OUT"\n')
    shim.chmod(0o755)
    output = tmp_path / "args"
    subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "tools/dspark/run_dspark_swa_acceptance.sh"),
            "a" * 40,
            "manifest",
            "rzwang",
            "--shutdown-policy=" + policy.POLICY_NAME,
        ],
        check=True,
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "ARGS_OUT": str(output)},
    )
    args = output.read_text().splitlines()
    for flag, value in [
        ("--profile-shutdown-policy", policy.POLICY_NAME),
        ("--core-remote", "rzwang"),
        ("--profile-output-tokens", "512"),
        ("--batches", "64"),
        ("--profile-stop-after-point", "ctx128-n4-t12-skewed"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--profile-exit-observation" not in args
    assert "--profile-operator-capture" not in args and "--profile-write-timeline" not in args
