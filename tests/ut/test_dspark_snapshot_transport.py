# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File integrity + real frozen Core IPC in eight CPU spawn processes; not NPU."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_confidence_acceptance import observer  # noqa: F401
from tools.dspark import snapshot_transport_check as check

files = check.files
ROOT = Path(__file__).parents[2]


def payload(rank=0, point="point"):
    return dict(rank=rank, cost_profile=dict(measurements=[dict(point=point, seconds=0.01, request_ids=["a", "b"])]))


def roundtrip(tmp_path):
    transfer = files.prepare(tmp_path, "point", 2)
    receipts = [files.persist(tmp_path, transfer, "point", r, lambda r=r: payload(r)) for r in range(2)]
    return transfer, receipts


def test_lossless_files_and_small_receipts(tmp_path):
    transfer, receipts = roundtrip(tmp_path)
    assert files.restore(tmp_path, transfer, "point", receipts, 2) == [payload(0), payload(1)]
    assert all(len(json.dumps(r).encode()) < files.MAX_RECEIPT_BYTES for r in receipts)
    assert len(list((tmp_path / "snapshot-transfers" / transfer).glob("rank-*-state.json"))) == 2
    later, _ = roundtrip(tmp_path)
    assert later != transfer
    assert files.restore(tmp_path, transfer, "point", receipts, 2) == [payload(0), payload(1)]


@pytest.mark.parametrize(
    "problem",
    [
        "missing_rank",
        "duplicate",
        "point",
        "transfer",
        "path",
        "size",
        "hash",
        "missing_file",
        "truncate",
        "corrupt",
        "symlink",
    ],
)
def test_missing_corrupt_stale_evidence_fails(tmp_path, problem):
    transfer, receipts = roundtrip(tmp_path)
    path = files.transfer_dir(tmp_path, transfer) / receipts[0]["file"]
    if problem == "missing_rank":
        receipts.pop()
    elif problem == "duplicate":
        receipts[1] = receipts[0]
    elif problem == "point":
        receipts[0]["point"] = "wrong"
    elif problem == "transfer":
        receipts[0]["transfer"] = "0" * 32
    elif problem == "path":
        receipts[0]["file"] = "../rank-0.json"
    elif problem == "size":
        receipts[0]["bytes"] = files.MAX_RANK_BYTES + 1
    elif problem == "hash":
        receipts[0]["sha256"] = "0" * 64
    elif problem == "missing_file":
        path.unlink()
    elif problem == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    elif problem == "corrupt":
        path.write_bytes(b"X" + path.read_bytes()[1:])
    elif problem == "symlink":
        other = path.with_suffix(".saved")
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises((ValueError, FileNotFoundError)):
        files.restore(tmp_path, transfer, "point", receipts, 2)


def test_partial_limit_and_build_error_are_not_published(tmp_path):
    with pytest.raises(ValueError, match="exceeds"):
        files.atomic_json(tmp_path / "large.json", {"data": "x" * 200}, limit=50)
    assert not (tmp_path / "large.json").exists() and (tmp_path / "large.json.partial").exists()
    transfer = files.prepare(tmp_path, "point", 1)

    def fail():
        raise RuntimeError("original build failure")

    with pytest.raises(RuntimeError, match="original build failure"):
        files.persist(tmp_path, transfer, "point", 0, fail)
    folder = files.transfer_dir(tmp_path, transfer)
    assert json.loads((folder / "rank-0-state.json").read_text())["status"] == "failed"
    assert not (folder / "rank-0.json").exists()


def test_point_history_cannot_be_relabelled(tmp_path):
    transfer = files.prepare(tmp_path, "point", 1)
    with pytest.raises(ValueError, match="another point"):
        files.persist(tmp_path, transfer, "point", 0, lambda: payload(point="previous"))


def test_original_rpc_error_survives_secondary_write_failure(tmp_path, monkeypatch):
    old = files.atomic_json

    def write(path, value, **kwargs):
        if Path(path).name == "frontend.json":
            raise OSError("secondary disk failure")
        return old(path, value, **kwargs)

    monkeypatch.setattr(files, "atomic_json", write)

    def rpc(*args, **kwargs):
        raise RuntimeError("original worker SIGBUS")

    with pytest.raises(RuntimeError, match="original worker SIGBUS"):
        files.fetch(NS(collective_rpc=rpc), tmp_path, "point", 1)


def test_actual_extension_file_rpc_checks_profiler_identity(tmp_path, observer, monkeypatch):  # noqa: F811
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_snapshot_transport", files)
    extension = sys.modules["vllm_ascend.diagnostics.dspark_benchmark_worker"].DSparkBenchmarkWorkerExtension
    worker = NS(
        rank=0,
        model_runner=NS(
            speculator=NS(confidence_verification=NS(options=dict(profile=True))),
            vllm_config=NS(additional_config=dict(dspark_profile_failure_dir=str(tmp_path))),
            _dspark_cost_profiler=NS(point="point"),
        ),
        dspark_benchmark_replay_snapshot=lambda: payload(),
    )
    transfer = files.prepare(tmp_path, "point", 1)
    receipt = extension.dspark_benchmark_profile_snapshot_file(worker, transfer, "point")
    assert files.restore(tmp_path, transfer, "point", [receipt], 1) == [payload()]
    with pytest.raises(ValueError, match="active profiler"):
        extension.dspark_benchmark_profile_snapshot_file(worker, transfer, "wrong")
    worker.model_runner.speculator.confidence_verification.options["profile"] = False
    with pytest.raises(ValueError, match="isolated cost"):
        extension.dspark_benchmark_profile_snapshot_file(worker, transfer, "point")


def test_real_eight_process_ipc_boundary_wrap_and_file_reconstruction(tmp_path):
    # Synthetic large text forces Core's real overflow path; the server preload
    # additionally runs the archived 35th-point snapshots with installed Core.
    ranks = [dict(payload(r), large="P" * (check.CHUNK + 1024)) for r in range(8)]
    source = tmp_path / "snapshots.json"
    source.write_text(json.dumps(dict(ranks=ranks)))
    output = tmp_path / "transport"
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.dspark.snapshot_transport_check",
            "--core",
            str(ROOT.parent / "vllm-hust"),
            "--snapshot",
            str(source),
            "--output",
            str(output),
            "--source-adapter",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=150,
        check=False,
    )
    assert done.returncode == 0, done.stdout
    report = json.loads((output / "transport-report.json").read_text())
    assert report["status"] == "PASSED_TRANSPORT_ONLY"
    assert report["exitcodes"] == [0] * 8 and report["owned_shm_remaining"] == []
    assert len(report["phases"]) == 15 and report["consumed_slots_retain_data"]
    for rank in report["worker_metrics"]:
        assert len(rank) == 15
        assert all(r["framed_bytes"] < 4096 for r in rank if r["method"] == "file")
        boundary = [r["threshold_bytes"] for r in rank if r["method"] == "boundary"]
        assert boundary == [check.CHUNK - 1, check.CHUNK, check.CHUNK + 1] * 3


def test_safe_shm_headroom_failure_allocates_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(check.Path, "exists", lambda p: True)
    monkeypatch.setattr(check.os, "statvfs", lambda _: NS(f_bavail=1, f_frsize=4096))
    monkeypatch.setattr(check, "core_api", lambda *args: pytest.fail("must not allocate a queue"))
    with pytest.raises(ValueError, match="headroom"):
        check.run(ROOT.parent / "vllm-hust", tmp_path, [payload(r) for r in range(8)])


def test_transport_precheck_failure_prevents_weights_or_pytest(tmp_path, monkeypatch):
    from tools.dspark import capacity_preflight

    monkeypatch.setattr(capacity_preflight, "installed_check", lambda: dict(status="PASSED_INTERFACE_ONLY"))

    def fail(*args):
        raise ValueError("transport verification failed")

    monkeypatch.setattr(check, "preflight", fail)
    monkeypatch.setattr(capacity_preflight.subprocess, "run", lambda *a, **k: pytest.fail("later stage started"))
    with pytest.raises(ValueError, match="transport verification failed"):
        capacity_preflight.main(
            [
                "--output",
                str(tmp_path / "interface.json"),
                "--core",
                str(ROOT.parent / "vllm-hust"),
                "--transport-evidence",
                str(tmp_path / "evidence.tar.gz"),
                "--",
                "-q",
            ]
        )
    error = json.loads((tmp_path / "transport-preflight-error.json").read_text())
    assert error == {"status": "FAILED", "error": "ValueError: transport verification failed"}


def test_read_only_collector_keeps_permissions_and_time_basis(tmp_path, monkeypatch):
    from tools.dspark import sigbus_system_evidence as system

    def denied(cmd, **kwargs):
        raise PermissionError("access denied")

    monkeypatch.setattr(system.subprocess, "run", denied)
    monkeypatch.setattr(system.os, "walk", lambda *a: iter(()))
    result = system.capture(tmp_path / "system.json")
    assert all("access denied" in r["unavailable"] for r in result["commands"])
    assert result["incident_window_utc"][0] == "2026-09-24T07:06:48+00:00"
    assert "current capacity" in result["scope"]
