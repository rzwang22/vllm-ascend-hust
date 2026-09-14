# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded pre-signal native exit observation. No device operations."""

import json
import os
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OBSERVE_SECONDS = 20.0
STACK_AT_SECONDS = (8.0, 14.0)
MAX_RANKS = 1
STACK_TIMEOUT_SECONDS = 4.0
DETACH_SECONDS = 0.5
MAX_STACK_BYTES = 2 * 1024 * 1024
POLL_SECONDS = 0.05


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def proc_state(pid):
    root = Path("/proc") / str(pid)
    try:
        status = (root / "status").read_text()
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        return {
            "pid": pid,
            "state": fields["State"].strip(),
            "tracer_pid": int(fields["TracerPid"]),
            "start_ticks": (root / "stat").read_text().rsplit(")", 1)[1].split()[19],
        }
    except FileNotFoundError:
        return {"pid": pid, "exited": True}


def native_stack(pid, directory, label, *, timeout=STACK_TIMEOUT_SECONDS, probe_delay=False):
    """Attach only to a caller-owned PID. Always detach/release before returning.

    SIGKILL is sent only to our debugger on timeout. A matching previously
    running tracee left in T/t is resumed, never a preexisting stopped process.
    Output and waits are bounded; ptrace pause is an observation intervention.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {
        "pid": pid,
        "label": label,
        "started_utc": utc(),
        "status": "unavailable",
        "timeout_seconds": timeout,
        "performance_eligible": False,
        "debugger_may_pause_tracee": True,
    }
    started = time.monotonic()
    debugger = None
    output = bytearray()
    try:
        before = receipt["before"] = proc_state(pid)
        if before.get("exited") or before["tracer_pid"] or before["state"][0] in "Tt":
            raise RuntimeError("Tracee exited, already traced, or already stopped")
        with (Path("/proc") / str(pid) / "maps").open("rb") as stream:
            maps = stream.read(MAX_STACK_BYTES + 1)
        receipt["maps_truncated"] = len(maps) > MAX_STACK_BYTES
        (directory / f"{label}.maps").write_bytes(maps[:MAX_STACK_BYTES])
        gdb = shutil.which("gdb")
        if gdb is None:
            raise RuntimeError("gdb executable unavailable")
        commands = [
            "set pagination off",
            "set confirm off",
            "set auto-load off",
            "set debuginfod enabled off",
            f"attach {pid}",
        ]
        if probe_delay:
            commands.append("python import time; time.sleep(10)")  # preflight timeout only
        commands += ["thread apply all bt 16", "detach", "quit"]
        argv = [gdb, "-nx", "-nh", "-batch"]
        for command in commands:
            argv += ["-ex", command]
        receipt["command"] = argv
        debugger = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        receipt["debugger_pid"] = debugger.pid
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(debugger.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    receipt["timed_out"] = True
                    break
                if proc_state(pid).get("tracer_pid") == debugger.pid:
                    receipt["attached_observed"] = True
                for key, _ in selector.select(min(remaining, POLL_SECONDS)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        break
                    output.extend(chunk[: max(0, MAX_STACK_BYTES - len(output))])
                    if len(output) == MAX_STACK_BYTES:
                        receipt["output_truncated"] = True
                        break
                if receipt.get("output_truncated"):
                    break
        if not receipt.get("timed_out") and not receipt.get("output_truncated"):
            try:
                debugger.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                receipt["timed_out"] = True
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        detach_deadline = time.monotonic() + DETACH_SECONDS
        if debugger is not None:
            if debugger.poll() is None:
                receipt["debugger_kill"] = {"signal": "SIGKILL", "utc": utc(), "pid": debugger.pid}
                debugger.kill()
                try:
                    debugger.wait(timeout=max(0, detach_deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    receipt["debugger_reap_unavailable"] = True
            receipt["debugger_returncode"] = debugger.returncode
            receipt["status"] = (
                "captured"
                if debugger.returncode == 0
                and b"#0 " in output
                and not receipt.get("error")
                and not receipt.get("timed_out")
                and not receipt.get("output_truncated")
                else "unavailable"
            )
            debugger.stdout.close()
        try:
            after = proc_state(pid)
            same = after.get("start_ticks") == receipt.get("before", {}).get("start_ticks")
            if (
                debugger is not None
                and same
                and before.get("tracer_pid") == 0
                and before.get("state", "")[:1] not in ("T", "t")
                and after.get("tracer_pid") == 0
                and after.get("state", "")[:1] in ("T", "t")
            ):
                os.kill(pid, signal.SIGCONT)
                receipt["resume_signal"] = {"signal": "SIGCONT", "utc": utc()}
                after = proc_state(pid)
                while after.get("state", "")[:1] in ("T", "t") and time.monotonic() < detach_deadline:
                    time.sleep(0.01)
                    after = proc_state(pid)
            receipt["after"] = after
            receipt["detached"] = after.get("exited", False) or (
                same and after.get("tracer_pid") == 0 and after.get("state", "")[:1] not in ("T", "t")
            )
        except Exception as error:
            receipt["detach_error"] = str(error)
            receipt["detached"] = False
        receipt["finished_utc"] = utc()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["pause_upper_bound_seconds"] = receipt["elapsed_seconds"]
        if not receipt["detached"]:
            receipt["status"] = "unavailable"
        (directory / f"{label}.stack.txt").write_bytes(output)
        save(directory / f"{label}.json", receipt)
    return receipt


def preflight(directory, *, import_runtime=False):
    """No model/device tensors. Test actual sibling ptrace and timeout detach."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {"started_utc": utc(), "platform": platform.platform(), "status": "failed", "performance_eligible": False}
    child = None
    try:
        if sys.platform != "linux" or shutil.which("gdb") is None:
            raise RuntimeError("Linux /proc and gdb required; no Python-stack fallback")
        result["gdb_version"] = subprocess.check_output(["gdb", "--version"], text=True, timeout=2).splitlines()[0]
        scope = Path("/proc/sys/kernel/yama/ptrace_scope")
        result["ptrace_scope"] = scope.read_text().strip() if scope.exists() else None
        heartbeat = directory / "heartbeat"
        ready_path = directory / "runtime.json"
        if ready_path.exists():
            raise RuntimeError("Native preflight requires a fresh directory")
        imports = (
            "import torch,torch_npu\nruntime.update(torch=torch.__version__,torch_npu=torch_npu.__version__)\n"
            if import_runtime
            else ""
        )
        # Same ancestry as gdb attaching to an EngineCore-owned worker: siblings.
        child = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "import time,sys,json\nfrom pathlib import Path\np=Path(sys.argv[1])\ni=0\n"
                "runtime={'python':sys.version}\n"
                + imports
                + "r=Path(sys.argv[2]);t=r.with_suffix('.tmp');t.write_text(json.dumps(runtime));t.replace(r)\n"
                "while True:\n i+=1\n p.write_text(str(i))\n time.sleep(.01)",
                str(heartbeat),
                str(ready_path),
            ],
        )
        ready_deadline = time.monotonic() + 10
        while not ready_path.exists() and child.poll() is None and time.monotonic() < ready_deadline:
            time.sleep(0.02)
        if not ready_path.exists():
            raise RuntimeError("Disposable runtime imports did not become ready within 10 seconds; see preflight log")
        result["child_runtime"] = json.loads(ready_path.read_text())
        result["runtime_imports_only"] = import_runtime
        normal = native_stack(child.pid, directory, "normal")
        delayed = native_stack(child.pid, directory, "timeout", probe_delay=True)
        result["normal"], result["timeout"] = normal, delayed
        if normal["status"] != "captured" or not normal["detached"]:
            raise RuntimeError(
                "Native attach/backtrace/detach unavailable; inspect normal.stack.txt and ptrace permissions"
            )
        if not delayed.get("timed_out") or not delayed["detached"] or not delayed.get("attached_observed"):
            raise RuntimeError("Forced debugger-timeout detach not verified")
        previous = heartbeat.read_text()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and heartbeat.read_text() == previous:
            time.sleep(0.02)
        if child.poll() is not None or heartbeat.read_text() == previous:
            raise RuntimeError("Disposable tracee did not resume/progress after debugger timeout")
        result["status"] = "passed"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if child is not None:
            try:
                child.kill()
                child.wait(timeout=1)
            except Exception as error:
                result["status"] = "failed"
                result["tracee_cleanup_error"] = f"{type(error).__name__}: {error}"
        result["finished_utc"] = utc()
        save(directory / "preflight.json", result)
    return result


def observe_workers(handles, directory, *, debugger_enabled=True):
    """Shared 20s pre-wait, then delegate unchanged Core 5s/4s escalation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = {
        "performance_eligible": False,
        "diagnostic_only": True,
        "status": "observing",
        "started_utc": utc(),
        "baseline_worker_grace_seconds": 5,
        "additional_observe_seconds": OBSERVE_SECONDS,
        "worker_grace_seconds": OBSERVE_SECONDS + 5,
        "term_seconds": 4,
        "reap_seconds": 1,
        "engine_seconds": 36,
        "frontend_seconds": 40,
        "supervisor_seconds": 48,
        "exits": {},
        "last_alive": {},
        "native_samples": [],
        "debugger_enabled": debugger_enabled,
        "attachment_count": 0,
        "native_sampling": "enabled" if debugger_enabled else "disabled_by_configuration",
        "stack_plan": {
            "at_seconds": STACK_AT_SECONDS if debugger_enabled else (),
            "max_ranks_per_round": MAX_RANKS,
            "timeout_seconds": STACK_TIMEOUT_SECONDS,
            "detach_seconds": DETACH_SECONDS,
        },
        "poll_resolution_seconds": POLL_SECONDS,
    }
    try:
        next_sample = 0
        while time.monotonic() - started < OBSERVE_SECONDS:
            alive = []
            for handle in handles:
                code = handle.proc.exitcode
                if code is None:
                    alive.append(handle)
                    data["last_alive"][str(handle.rank)] = {"utc": utc(), "elapsed_seconds": time.monotonic() - started}
                elif str(handle.rank) not in data["exits"]:
                    data["exits"][str(handle.rank)] = {
                        "pid": handle.proc.pid,
                        "raw_exitcode": code,
                        "observed_utc": utc(),
                        "elapsed_seconds": time.monotonic() - started,
                        "meaning": "parent poll/reap timestamp, not kernel exit instant",
                    }
                    save(directory / "observation.json", data)
            if not alive:
                break
            elapsed = time.monotonic() - started
            if debugger_enabled and next_sample < len(STACK_AT_SECONDS) and elapsed >= STACK_AT_SECONDS[next_sample]:
                for handle in sorted(alive, key=lambda h: h.rank)[:MAX_RANKS]:
                    if handle.proc.exitcode is not None:
                        continue
                    label = f"round-{next_sample}-rank-{handle.rank}"
                    remaining = OBSERVE_SECONDS - (time.monotonic() - started) - DETACH_SECONDS
                    if remaining <= 0:
                        break
                    sample = native_stack(
                        handle.proc.pid, directory, label, timeout=min(STACK_TIMEOUT_SECONDS, remaining)
                    )
                    sample["rank"] = handle.rank
                    data["native_samples"].append(sample)
                    data["attachment_count"] += int(
                        sample.get("attached_observed", False) or sample["status"] == "captured"
                    )
                    save(directory / "observation.json", data)
                next_sample += 1
            time.sleep(POLL_SECONDS)
        data["status"] = "returned"
    except Exception as error:
        data["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        data["finished_utc"] = utc()
        data["elapsed_seconds"] = time.monotonic() - started
        data["interpretation"] = (
            "Debugger pauses are included; compare stages, not pure shutdown performance"
            if debugger_enabled
            else "No debugger/ptrace; parent polling/reap timestamps are upper bounds, not kernel exit instants"
        )
        save(directory / "observation.json", data)
    return data
