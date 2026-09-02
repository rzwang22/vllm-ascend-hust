# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Rank-local, test-only observations for M2.5A repeat boundaries."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_REQUIRED_EVENT_ORDER = (
    "request_begin",
    "request_start_barrier_before",
    "request_start_barrier_after",
    "request_start_npu_sync_before",
    "request_start_npu_sync_after",
    "inference_timer_start",
    "scheduler_admission_after",
    "first_commit",
    "last_commit",
    "request_end_npu_sync_before",
    "request_end_npu_sync_after",
    "inference_timer_end",
    "scheduler_cleanup_before",
    "zero_token_worker_cleanup_after",
    "logical_runner_kv_proposal_cleanup_after",
    "request_final_barrier_before",
    "request_final_barrier_after",
    "request_complete",
)
_REPEATABLE_EVENTS = frozenset({"slow_host_step"})
_REQUIRED_SINGLETON_EVENTS = frozenset((*_REQUIRED_EVENT_ORDER, "finished_event_delivered"))
_KNOWN_EVENTS = _REQUIRED_SINGLETON_EVENTS | _REPEATABLE_EVENTS


class PerformanceBoundaryWriter:
    """Write an exclusive rank-local event stream without merged stdout."""

    def __init__(self, root: Path, owner: dict[str, Any]) -> None:
        self.owner = {
            "mode": owner["mode"],
            "rank": owner["rank"],
            "case_id": owner["case_id"],
        }
        self.path = root / "performance-boundary" / owner["mode"] / f"rank-{owner['rank']}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x", encoding="utf-8")
        self.event_count = 0
        self.complete = False
        self._write("header", {"boundary_semantics": "observe_existing_no_barrier"})

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        line = json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                **self.owner,
                "monotonic_seconds": time.monotonic(),
                "payload": payload,
            },
            allow_nan=False,
            sort_keys=True,
        )
        self.handle.write(line + "\n")
        self.handle.flush()

    def write_event(
        self,
        event: str,
        *,
        request_id: str,
        request_sequence_index: int,
        repeat_kind: str,
        repeat_index: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self.complete:
            raise RuntimeError("Performance-boundary trace is already complete.")
        self._write(
            "event",
            {
                "event": event,
                "request_id": request_id,
                "request_sequence_index": request_sequence_index,
                "repeat_kind": repeat_kind,
                "repeat_index": repeat_index,
                **(payload or {}),
            },
        )
        self.event_count += 1

    def finish(self, request_count: int) -> None:
        if self.complete:
            raise RuntimeError("Performance-boundary trace completed more than once.")
        self._write(
            "complete",
            {
                "event_count": self.event_count,
                "request_count": request_count,
            },
        )
        self.complete = True

    def close(self) -> None:
        self.handle.close()


@contextmanager
def performance_boundary_writer(
    root: Path | None,
    owner: dict[str, Any],
    *,
    enabled: bool,
):
    if not enabled:
        yield None
        return
    if root is None:
        raise ValueError("Performance-boundary diagnostics require a result directory.")
    writer = PerformanceBoundaryWriter(root, owner)
    primary_error = None
    try:
        yield writer
        if not writer.complete:
            raise RuntimeError("Performance-boundary diagnostics did not complete.")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            writer.close()
        except BaseException:
            if primary_error is None:
                raise


def load_performance_boundary_traces(
    root: Path,
    *,
    expected_ranks: int,
    expected_requests: int,
) -> dict[int, list[dict[str, Any]]]:
    """Strictly validate completed rank-local boundary evidence."""
    traces: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((root / "performance-boundary").rglob("rank-*.jsonl")):
        text = path.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            raise ValueError(f"Truncated performance-boundary trace: {path}.")
        rows = [json.loads(line) for line in text.splitlines()]
        if len(rows) < 3 or rows[0]["kind"] != "header" or rows[-1]["kind"] != "complete":
            raise ValueError(f"Incomplete performance-boundary trace: {path}.")
        rank = rows[0]["rank"]
        if rank in traces or rank not in range(expected_ranks):
            raise ValueError(f"Duplicate or unexpected boundary-trace rank {rank!r}.")
        owner = {key: rows[0][key] for key in ("mode", "rank", "case_id")}
        timestamps = []
        request_events: dict[str, list[dict[str, Any]]] = {}
        request_owners: dict[str, tuple[int, str, int]] = {}
        for row in rows:
            if row["schema_version"] != SCHEMA_VERSION or any(row[key] != value for key, value in owner.items()):
                raise ValueError(f"Mixed performance-boundary ownership: {path}.")
            timestamps.append(row["monotonic_seconds"])
            if row["kind"] == "event":
                payload = row["payload"]
                event = payload.get("event")
                request_id = payload.get("request_id")
                if event not in _KNOWN_EVENTS or not isinstance(request_id, str):
                    raise ValueError(f"Unknown performance-boundary event: {path}.")
                request_owner = (
                    payload.get("request_sequence_index"),
                    payload.get("repeat_kind"),
                    payload.get("repeat_index"),
                )
                previous_owner = request_owners.setdefault(request_id, request_owner)
                if request_owner != previous_owner:
                    raise ValueError(f"Mixed performance-boundary request ownership: {path}.")
                request_events.setdefault(request_id, []).append(row)
        if timestamps != sorted(timestamps):
            raise ValueError(f"Out-of-order performance-boundary timestamps: {path}.")
        complete = rows[-1]["payload"]
        if complete["event_count"] != len(rows) - 2 or complete["request_count"] != expected_requests:
            raise ValueError(f"Invalid performance-boundary completion: {path}.")
        if len(request_events) != expected_requests:
            raise ValueError(f"Performance-boundary request count mismatch: {path}.")
        sequence_indices = sorted(owner[0] for owner in request_owners.values())
        if sequence_indices != list(range(expected_requests)):
            raise ValueError(f"Performance-boundary request sequence mismatch: {path}.")
        for request_id, events in request_events.items():
            event_names = [row["payload"]["event"] for row in events]
            counts = {event: event_names.count(event) for event in _REQUIRED_SINGLETON_EVENTS}
            if any(count != 1 for count in counts.values()):
                raise ValueError(f"Missing or duplicate performance-boundary event for {request_id}: {path}.")
            ordered_names = [event for event in event_names if event in _REQUIRED_EVENT_ORDER]
            if ordered_names != list(_REQUIRED_EVENT_ORDER):
                raise ValueError(f"Out-of-order performance-boundary events for {request_id}: {path}.")
        traces[rank] = rows
    if set(traces) != set(range(expected_ranks)):
        raise ValueError(f"Incomplete performance-boundary ranks: {sorted(traces)}.")
    return traces
