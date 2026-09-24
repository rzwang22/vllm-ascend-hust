# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded read-only host evidence. Current free space is NOT crash-time proof."""

import argparse
import datetime
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

LIMIT = 65536


def capture(output):
    started = time.monotonic()
    report = dict(
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observed_local=datetime.datetime.now().astimezone().isoformat(),
        incident_window_utc=["2026-09-24T07:06:48+00:00", "2026-09-24T07:07:03+00:00"],
        incident_time_source="engine-failure.json UTC; display-log timezone must be checked separately",
        scope="Read-only, no attach/signal/cleanup; current capacity does not establish past exhaustion",
        files={},
        commands=[],
        cann_logs=[],
    )
    paths = [
        "/proc/self/mountinfo",
        "/proc/self/cgroup",
        "/proc/meminfo",
        "/proc/sys/kernel/core_pattern",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory.events",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.failcnt",
    ]
    for name in paths:
        try:
            with open(name, "rb") as stream:
                data = stream.read(LIMIT + 1)
            report["files"][name] = dict(text=data[:LIMIT].decode(errors="replace"), truncated=len(data) > LIMIT)
        except OSError as exc:
            report["files"][name] = dict(unavailable=str(exc))
    try:
        if Path("/dev/shm").exists():
            stat = os.statvfs("/dev/shm")
            report["shm_current"] = dict(total=stat.f_blocks * stat.f_frsize, available=stat.f_bavail * stat.f_frsize)
            entries = []
            for i, path in enumerate(Path("/dev/shm").iterdir()):
                if i == 512:
                    break
                try:
                    st = path.stat()
                    entries.append(
                        dict(name=path.name, size=st.st_size, allocated_bytes=st.st_blocks * 512, uid=st.st_uid)
                    )
                except OSError:
                    continue
            report["shm_entries_up_to_512"] = entries
    except OSError as exc:
        report["shm_unavailable"] = str(exc)
    start, end = "2026-09-24 07:06:40 UTC", "2026-09-24 07:07:10 UTC"
    commands = [
        ["date", "--iso-8601=seconds"],
        ["findmnt", "/dev/shm"],
        ["df", "-B1", "/dev/shm"],
        ["ps", "-eo", "pid,ppid,stat,comm"],
        ["ipcs", "-m"],
        ["journalctl", "-k", "--since", start, "--until", end, "--no-pager", "-n", "200"],
        ["dmesg", "--time-format", "iso", "--since", start, "--until", end],
        ["coredumpctl", "list", "--since", start, "--until", end, "--no-pager"],
    ]
    for cmd in commands:
        record = dict(command=cmd)
        with tempfile.TemporaryFile() as stream:
            try:
                done = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=4, check=False)
                record["rc"] = done.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                record["unavailable"] = str(exc)
            stream.seek(0)
            data = stream.read(LIMIT + 1)
            record.update(text=data[:LIMIT].decode(errors="replace"), truncated=len(data) > LIMIT)
        report["commands"].append(record)
    # No dump bodies or NPU memory read. Only limited existing CANN text logs.
    deadline = started + 40
    for directory in (Path.home() / "ascend/log", Path("/var/log/npu"), Path("/var/log/ascend")):
        seen = 0
        for base, _, names in os.walk(directory):
            if time.monotonic() > deadline or seen >= 512 or len(report["cann_logs"]) >= 16:
                break
            for name in names:
                seen += 1
                if seen > 512 or len(report["cann_logs"]) >= 16:
                    break
                path = Path(base) / name
                if path.suffix != ".log" or path.is_symlink():
                    continue
                try:
                    st = path.stat()
                    with path.open("rb") as stream:
                        stream.seek(max(0, st.st_size - 16384))
                        text = stream.read(16384).decode(errors="replace")
                    if "2026-09-24" in text or "SIGBUS" in text or "main process disappeared" in text:
                        report["cann_logs"].append(dict(path=str(path), mtime=st.st_mtime, size=st.st_size, tail=text))
                except OSError as exc:
                    report["cann_logs"].append(dict(path=str(path), unavailable=str(exc)))
    report["cann_coverage"] = "Bounded existing log tails only; empty/missing evidence does not exclude a fault"
    report["elapsed_seconds"] = time.monotonic() - started
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    capture(parser.parse_args().output)
