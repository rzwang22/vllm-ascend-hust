# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned real subprocesses and actual exit observer/Core escalation; no weights."""

import asyncio
import json
import logging
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_failure import load_executor, methods
from tests.ut.test_dspark_worker_exit import load_source
from tools.dspark import profile_failure as failure
from tools.dspark.shutdown_policy import stack_signals_enabled

ROOT = Path(__file__).parents[2]
TRACE = ROOT / "vllm_ascend/diagnostics/dspark_worker_exit.py"
CHILD = """
import importlib.util,sys,time,faulthandler,signal,atexit
from pathlib import Path
s=importlib.util.spec_from_file_location('trace',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
t=m.WorkerExitTrace(sys.argv[2],0,stack_signals=True)
mode=sys.argv[3]
if mode=='kill': signal.signal(signal.SIGTERM,signal.SIG_IGN)
def exit_tail():
    faulthandler.unregister(m.STACK_SIGNAL)
    assert signal.getsignal(m.STACK_SIGNAL)==signal.SIG_DFL
    Path(sys.argv[2],'revoked').write_text('handler revoked during atexit; original ready unchanged')
    while not Path(sys.argv[2],'finish').exists(): time.sleep(.005)
    time.sleep(.55 if mode in ('0','7') else 30)
atexit.register(exit_tail)
sys.exit(int(mode) if mode in ('0','7') else 0)
"""


class ChildHandle:
    """multiprocessing-compatible view of an owned Popen; no synthetic exit codes."""

    def __init__(self, child):
        self.child, self.pid, self.name = child, child.pid, "passive-exit-regression"

    @property
    def exitcode(self):
        return self.child.poll()

    def is_alive(self):
        return self.child.poll() is None

    def terminate(self):
        self.child.terminate()

    def kill(self):
        self.child.kill()

    def join(self, timeout):
        with suppress(subprocess.TimeoutExpired):
            self.child.wait(timeout=timeout)


def spawn(tmp_path, mode):
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(TRACE), str(tmp_path), mode])
    deadline = time.monotonic() + 5
    while not (tmp_path / "revoked").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not (tmp_path / "revoked").exists():
        child.kill()
        child.wait(timeout=3)
        pytest.fail("Child did not reach revoked-handler state")
    assert json.loads((tmp_path / "rank-0-ready.json").read_text())["signal_registered"] is True
    assert child.poll() is None
    return child


def test_stale_registration_active_control_reproduces_signal_death(tmp_path):
    trace = load_source("active_control", TRACE)
    child = spawn(tmp_path, "0")
    try:
        watch = trace.ExitWatch(tmp_path, [NS(rank=0, proc=ChildHandle(child))], "p", 1, stack_signals=True)
        # The old call path is deliberately opt-in in this isolated control.
        watch.snapshot("control", request_stacks=True)
        assert child.wait(timeout=3) == -signal.SIGUSR1
        assert len(watch.signal_events) == 1
        assert next(tmp_path.glob("*-stacks.txt")).stat().st_size == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)


@pytest.mark.parametrize(
    "mode,code,forces",
    [
        ("0", 0, 0),
        ("7", 7, 0),
        ("term", -signal.SIGTERM, 1),
        ("kill", -signal.SIGKILL, 2),
        ("timeout", -signal.SIGTERM, 1),
    ],
)
def test_real_passive_executor_preserves_exit_and_escalation(tmp_path, monkeypatch, caplog, mode, code, forces):
    trace = load_source("passive_trace", TRACE)
    cleanup = load_source("passive_cleanup", ROOT / "vllm_ascend/diagnostics/dspark_cleanup.py")
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_cleanup", cleanup)
    module = load_executor(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_worker_exit", trace)
    vllm = ModuleType("vllm")
    vllm.envs = NS(VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=0.7)
    logger = logging.getLogger("vllm.v1.executor.multiproc_executor")
    caplog.set_level(logging.WARNING, logger=logger.name)
    termination = methods(
        "v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        ["_ensure_worker_termination"],
        {
            "time": time,
            "envs": vllm.envs,
            "logger": NS(info=logger.info, info_once=logger.info, warning=logger.warning),
        },
    )
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    def original(self):
        self.shutting_down = True
        (tmp_path / "worker-exit/finish").write_text("shutdown returned; delayed process tail")
        termination._ensure_worker_termination([h.proc for h in self.workers])

    monkeypatch.setattr(module.ProfileMultiprocExecutor.__mro__[1], "shutdown", original)
    directory = tmp_path / "worker-exit"
    directory.mkdir()
    child = spawn(directory, mode)
    try:
        executor = module.ProfileMultiprocExecutor(
            NS(
                additional_config={
                    "dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"},
                    "dspark_profile_failure_dir": str(tmp_path),
                    "dspark_profile_worker_exit": True,
                    "dspark_profile_stack_signals": False,
                    "dspark_profile_exit_debugger": False,
                }
            )
        )
        executor.workers = [NS(rank=0, proc=ChildHandle(child))]
        passive = trace.ExitWatch(directory, executor.workers, "p", 1)
        passive.snapshot("direct-live", request_stacks=True)
        assert child.poll() is None and not passive.signal_events
        guard = failure.ProfileFailureGuard(
            NS(shutdown=lambda timeout: executor.shutdown()), tmp_path, require_worker_receipt=True
        )
        if mode == "timeout":
            monkeypatch.setattr(failure, "CLEANUP_TIMEOUT_SECONDS", 0.1)
            monkeypatch.setattr(failure, "CLEANUP_FINALIZE_SECONDS", 0.1)
            monkeypatch.setattr(failure, "POLL_SECONDS", 0.005)
        result = asyncio.run(guard.shutdown())
        assert child.wait(timeout=3) == code
        assert result["success"] is (code == 0)
        if mode == "timeout":
            assert result["timed_out"] and not result["thread_completed"]
            deadline = time.monotonic() + 3
            while not (tmp_path / "cleanup-thread.json").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (tmp_path / "cleanup-thread.json").exists()
            assert json.loads((tmp_path / "cleanup.json").read_text())["timed_out"]
        else:
            assert result["forced_cleanup"] is bool(forces)
        workers = json.loads((tmp_path / "worker-cleanup.json").read_text())
        assert workers["workers"][0]["raw_exitcode"] == code
        assert workers["reap"]["workers"][0]["status"] == "reaped"
        assert len(workers["force_events"]) == forces
        assert workers["stack_signals_enabled"] is False and workers["diagnostic_signals_sent"] == []
        assert workers["debugger_enabled"] is False
        checkpoint = json.loads((directory / "parent-checkpoint-0.json").read_text())
        assert checkpoint["workers"][0]["raw_exitcode"] is None
        assert checkpoint["stack_request_enabled"] is False and checkpoint["diagnostic_signals_sent"] == []
        # Even an explicit snapshot request cannot override this watch's policy.
        other = tmp_path / "direct"
        other.mkdir()
        watch = trace.ExitWatch(other, executor.workers, "p", 1)
        watch.snapshot("forced-call", request_stacks=True)
        assert not watch.signal_events
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3)


def test_formal_policy_rejects_active_signals():
    assert not stack_signals_enabled({})
    assert stack_signals_enabled(
        dict(dspark_profile_worker_exit=True, dspark_profile_exit_observation=True, dspark_profile_stack_signals=True)
    )
    for extra in (
        {},
        {"dspark_profile_shutdown_policy": "dspark-profile-25s-v1"},
        {"dspark_confidence_acceptance": True},
    ):
        with pytest.raises(ValueError):
            stack_signals_enabled(dict(dspark_profile_stack_signals=True, **extra))


def test_passive_worker_does_not_register_signal(tmp_path, monkeypatch):
    trace = load_source("passive_registration", TRACE)
    monkeypatch.setattr(
        trace.faulthandler, "register", lambda *a, **k: pytest.fail("passive worker registered a signal")
    )
    t = trace.WorkerExitTrace(tmp_path, 0)
    ready = json.loads((tmp_path / "rank-0-ready.json").read_text())
    assert ready["signal_registered"] is False and ready["stack_signals_enabled"] is False
    assert t.stack_file is None
    trace.os.close(t.fd)


def passive_receipts(root):
    """Synthetic file-integrity fixture, separate from real subprocess tests above."""
    path = root / "worker-cleanup.json"
    workers = json.loads(path.read_text()) if path.exists() else {"workers": [{"rank": r} for r in range(8)]}
    workers.update(stack_signals_enabled=False, diagnostic_signals_sent=[], debugger_enabled=False)
    folder = root / "worker-exit"
    folder.mkdir(exist_ok=True)
    for worker in workers["workers"]:
        rank = worker["rank"]
        worker["pid"] = rank + 100
        (folder / f"rank-{rank}-ready.json").write_text(
            json.dumps(
                dict(rank=rank, pid=rank + 100, stack_signals_enabled=False, signal_registered=False, error=None)
            )
        )
    path.write_text(json.dumps(workers))


@pytest.mark.parametrize("fault", [None, "missing", "enabled", "sent", "registered", "pid", "checkpoint"])
def test_passive_acceptance_requires_new_consistent_receipts(tmp_path, fault):
    from tools.dspark.shutdown_acceptance import require_passive

    passive_receipts(tmp_path)
    path = tmp_path / "worker-cleanup.json"
    worker = json.loads(path.read_text())
    if fault == "missing":
        del worker["stack_signals_enabled"]
    if fault == "enabled":
        worker["stack_signals_enabled"] = True
    if fault == "sent":
        worker["diagnostic_signals_sent"] = [{"signal_sent": int(signal.SIGUSR1)}]
    path.write_text(json.dumps(worker))
    ready = tmp_path / "worker-exit/rank-0-ready.json"
    data = json.loads(ready.read_text())
    if fault == "registered":
        data["signal_registered"] = True
    if fault == "pid":
        data["pid"] = 999
    ready.write_text(json.dumps(data))
    if fault == "checkpoint":
        (tmp_path / "worker-exit/parent-checkpoint-0.json").write_text(
            json.dumps(dict(stack_signals_enabled=False, diagnostic_signals_sent=[1]))
        )
    if fault is None:
        require_passive(tmp_path)
    else:
        with pytest.raises(ValueError):
            require_passive(tmp_path)
