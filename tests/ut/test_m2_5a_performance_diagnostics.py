# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
from pathlib import Path

import pytest

from tests.e2e.nightly.single_node.spec_decode.m2_5a_performance_diagnostics import (
    PerformanceBoundaryWriter,
    load_performance_boundary_traces,
    performance_boundary_writer,
)

EVENTS = (
    "request_begin",
    "request_start_barrier_before",
    "request_start_barrier_after",
    "request_start_npu_sync_before",
    "request_start_npu_sync_after",
    "inference_timer_start",
    "scheduler_admission_after",
    "first_commit",
    "finished_event_delivered",
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
OWNER = {"mode": "dspark", "rank": 0, "case_id": "case-a"}


def _write_valid_trace(root: Path) -> Path:
    writer = PerformanceBoundaryWriter(root, OWNER)
    for event in EVENTS:
        writer.write_event(
            event,
            request_id="request-a",
            request_sequence_index=0,
            repeat_kind="measured",
            repeat_index=0,
            payload=({"public_pending_future_count": None} if event == "request_complete" else None),
        )
    writer.finish(1)
    writer.close()
    return writer.path


def test_performance_boundary_trace_is_rank_local_ordered_and_complete(
    tmp_path: Path,
) -> None:
    path = _write_valid_trace(tmp_path)

    traces = load_performance_boundary_traces(
        tmp_path,
        expected_ranks=1,
        expected_requests=1,
    )

    assert [row["kind"] for row in traces[0]] == [
        "header",
        *("event" for _ in EVENTS),
        "complete",
    ]
    assert traces[0][-2]["payload"]["public_pending_future_count"] is None
    assert "output_token_ids" not in path.read_text(encoding="utf-8")


def test_disabled_performance_boundary_writer_has_no_files(tmp_path: Path) -> None:
    with performance_boundary_writer(tmp_path, OWNER, enabled=False) as writer:
        assert writer is None
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("failure", ("truncated", "corrupt"))
def test_performance_boundary_trace_rejects_truncated_or_corrupt_json(
    tmp_path: Path,
    failure: str,
) -> None:
    path = _write_valid_trace(tmp_path)
    text = path.read_text(encoding="utf-8")
    if failure == "truncated":
        path.write_text(text.rstrip("\n"), encoding="utf-8")
        match = "Truncated"
    else:
        lines = text.splitlines()
        lines[1] = "{corrupt-json"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        match = "Expecting property name"
    with pytest.raises(ValueError if failure == "truncated" else json.JSONDecodeError, match=match):
        load_performance_boundary_traces(
            tmp_path,
            expected_ranks=1,
            expected_requests=1,
        )


@pytest.mark.parametrize("failure", ("missing", "duplicate"))
def test_performance_boundary_trace_rejects_missing_or_duplicate_event(
    tmp_path: Path,
    failure: str,
) -> None:
    path = _write_valid_trace(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    event_index = next(
        index
        for index, row in enumerate(rows)
        if row["kind"] == "event" and row["payload"]["event"] == "scheduler_admission_after"
    )
    if failure == "missing":
        rows.pop(event_index)
    else:
        rows.insert(event_index, rows[event_index])
    rows[-1]["payload"]["event_count"] = len(rows) - 2
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing or duplicate"):
        load_performance_boundary_traces(
            tmp_path,
            expected_ranks=1,
            expected_requests=1,
        )
