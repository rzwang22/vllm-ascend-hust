# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded profile evidence files and small RPC receipts, outside timed calls."""

import hashlib
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from uuid import UUID, uuid4

MAX_RANK_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 4096
SCHEMA = "dspark-profile-file-v1"


def transfer_dir(root, transfer):
    if not isinstance(transfer, str) or UUID(transfer).hex != transfer:
        raise ValueError("Invalid snapshot transfer ID")
    return Path(root) / "snapshot-transfers" / transfer


def atomic_json(path, value, limit=MAX_RANK_BYTES):
    """Stream encoding bounds temporary host memory and refuses truncated publication."""
    path = Path(path)
    part = path.with_name(path.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    with part.open("xb") as stream:
        for text in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
            data = text.encode("utf-8")
            size += len(data)
            if size > limit:
                raise ValueError(f"Snapshot exceeds {limit} bytes; incomplete file retained")
            stream.write(data)
            digest.update(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)
    return size, digest.hexdigest()


def prepare(root, point, ranks):
    if point is not None and (not isinstance(point, str) or not point or len(point) > 128):
        raise ValueError("Invalid snapshot point")
    if type(ranks) is not int or not 1 <= ranks <= 8:
        raise ValueError("Invalid snapshot rank count")
    transfer = uuid4().hex
    folder = transfer_dir(root, transfer)
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / "request.json", dict(schema=SCHEMA, transfer=transfer, point=point, ranks=ranks))
    return transfer


def request(root, transfer, point, rank):
    folder = transfer_dir(root, transfer)
    spec = json.loads((folder / "request.json").read_text())
    if (spec["schema"], spec["transfer"], spec["point"]) != (SCHEMA, transfer, point):
        raise ValueError("Snapshot request identity mismatch")
    if type(rank) is not int or not 0 <= rank < spec["ranks"]:
        raise ValueError("Snapshot rank mismatch")
    return folder


def check_payload(payload, point, rank):
    if payload["rank"] != rank or "cost_profile" not in payload:
        raise ValueError("Snapshot payload rank/profile unavailable")
    if any(row["point"] != point for row in payload["cost_profile"]["measurements"]):
        raise ValueError("Snapshot contains measurements from another point")


def persist(root, transfer, point, rank, build):
    folder = request(root, transfer, point, rank)
    started = time.monotonic()
    status = dict(schema=SCHEMA, transfer=transfer, point=point, rank=rank, status="building")
    state_path = folder / f"rank-{rank}-state.json"
    atomic_json(state_path, status)
    try:
        payload = build()
        check_payload(payload, point, rank)
        name = f"rank-{rank}.json"
        size, digest = atomic_json(folder / name, payload)
        receipt = dict(schema=SCHEMA, transfer=transfer, point=point, rank=rank, file=name, bytes=size, sha256=digest)
        if len(json.dumps(receipt).encode()) > MAX_RECEIPT_BYTES:
            raise ValueError("Snapshot RPC receipt exceeded bound")
        status.update(status="committed", receipt=receipt, elapsed_seconds=time.monotonic() - started)
        atomic_json(state_path, status)
        return receipt
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started)
        # Preserve the original build/write failure; partial files remain.
        with suppress(Exception):
            atomic_json(state_path, status)
        raise


def restore(root, transfer, point, receipts, ranks):
    folder = transfer_dir(root, transfer)
    if len(receipts) != ranks or sorted(r["rank"] for r in receipts) != list(range(ranks)):
        raise ValueError("Missing/duplicate snapshot rank receipts")
    result = []
    for receipt in sorted(receipts, key=lambda r: r["rank"]):
        if len(json.dumps(receipt).encode()) > MAX_RECEIPT_BYTES:
            raise ValueError("Oversized snapshot receipt")
        rank = receipt["rank"]
        request(root, transfer, point, rank)
        if (receipt["schema"], receipt["transfer"], receipt["point"], receipt["file"]) != (
            SCHEMA,
            transfer,
            point,
            f"rank-{rank}.json",
        ):
            raise ValueError("Snapshot receipt identity mismatch")
        size = receipt["bytes"]
        if type(size) is not int or not 0 < size <= MAX_RANK_BYTES:
            raise ValueError("Invalid snapshot file length")
        path = folder / receipt["file"]
        if path.is_symlink() or path.stat().st_size != size:
            raise ValueError("Missing/truncated snapshot file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != receipt["sha256"]:
            raise ValueError("Corrupt snapshot file")
        with path.open() as stream:
            payload = json.load(stream)
        check_payload(payload, point, rank)
        result.append(payload)
    return result


def fetch(engine, root, point, ranks):
    transfer = prepare(root, point, ranks)
    report = dict(schema=SCHEMA, transfer=transfer, point=point, status="requested")
    folder = transfer_dir(root, transfer)
    try:
        receipts = engine.collective_rpc(
            "dspark_benchmark_profile_snapshot_file", kwargs=dict(transfer=transfer, point=point)
        )
        report.update(status="receipts_returned", receipts=receipts)
        atomic_json(folder / "frontend.json", report)
        result = restore(root, transfer, point, receipts, ranks)
        report["status"] = "validated"
        return result
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # A write error must not replace the original RPC/read error.
        try:
            atomic_json(folder / "frontend.json", report)
        except Exception:
            if report["status"] != "failed":
                raise
