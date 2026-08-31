# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Test-only, exclusive rank-local evidence files; never a merged-stdout parser."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
OWNER_FIELDS = ("mode", "rank", "case_id", "lifecycle_repeat", "request_sequence_index", "request_id")


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value in trace: {value}.")


class RankTraceWriter:
    def __init__(self, root: Path, owner: dict[str, Any], early_tokens: int) -> None:
        self.owner = {key: owner[key] for key in OWNER_FIELDS}
        self.path = (
            root / "traces" / owner["mode"] / f"repeat-{owner['lifecycle_repeat']}" / f"rank-{owner['rank']}.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A rerun must use a new result directory, including after a failure.
        self.handle = self.path.open("x", encoding="utf-8")
        self.count = 0
        self.complete = False
        try:
            self._write("header", {"early_output_token_limit": early_tokens})
        except BaseException:
            self.handle.close()
            raise
        print(f"DSPARK_M2_5A_RANK_TRACE_FILE={self.path}", flush=True)

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        # Serialize before writing: unsupported values and NaN must fail, not
        # silently become strings or a truncated record accepted by a reader.
        line = json.dumps(
            {"schema_version": SCHEMA_VERSION, "kind": kind, **self.owner, "payload": payload},
            allow_nan=False,
            sort_keys=True,
        )
        self.handle.write(line + "\n")
        self.handle.flush()

    def write_step(self, record: dict[str, Any]) -> None:
        if self.complete or any(record[key] != self.owner[key] for key in OWNER_FIELDS):
            raise ValueError("Rank-local trace has stale/foreign ownership or is already complete.")
        self._write("step", record)
        self.count += 1

    def finish(self, result: dict[str, Any], completion: dict[str, Any] | None = None) -> None:
        if self.complete or any(result[key] != self.owner[key] for key in OWNER_FIELDS):
            raise ValueError("Rank-local trace completion has stale/foreign ownership.")
        self._write("complete", {"step_count": self.count, "result": result, "completion": completion})
        self.complete = True

    def close(self) -> None:
        self.handle.close()


@contextmanager
def rank_trace_writer(root: Path | None, owner: dict[str, Any], early_tokens: int):
    if not early_tokens:
        yield None
        return
    if root is None:
        raise ValueError("Early-range trace requires the run's result directory.")
    writer = RankTraceWriter(root, owner, early_tokens)
    primary_error = None
    try:
        yield writer
        if not writer.complete:
            raise RuntimeError("Rank-local trace did not complete its request cleanup lifecycle.")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            writer.close()
        except BaseException as exc:
            # Preserve the generation/cleanup exception. The missing complete
            # record still makes the evidence unusable for semantic PASS.
            if primary_error is None:
                raise
            print(f"DSPARK_M2_5A_TRACE_CLOSE_ERROR={type(exc).__name__}", flush=True)


def load_rank_traces(
    root: Path, records: dict[str, dict[str, Any]], expected_ranks: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Strictly read one completed selected-case execution per rank.

    Both a final newline and a completion record are required: interruption at
    a JSON boundary is truncation too. Historical stdout is not an input here.
    """
    ranks, summaries, steps = set(), {}, []
    for path in sorted(root.rglob("rank-*.jsonl")):
        text = path.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            raise ValueError(f"Truncated rank-local JSONL (no final newline): {path}.")
        rows = [json.loads(line, parse_constant=_invalid_json_constant) for line in text.splitlines()]
        if len(rows) < 3 or rows[0]["kind"] != "header" or rows[-1]["kind"] != "complete":
            raise ValueError(f"Incomplete rank-local trace: {path}.")
        owner = {key: rows[0][key] for key in OWNER_FIELDS}
        rank = owner["rank"]
        if type(rank) is not int or rank not in range(expected_ranks) or str(rank) not in records:
            raise ValueError(f"Unexpected rank {rank!r}: {path}.")
        if rank in ranks:
            raise ValueError(f"Duplicate rank {rank}: {path}.")
        ranks.add(rank)
        expected = records[str(rank)]
        if any(owner[key] != expected[key] for key in OWNER_FIELDS):
            raise ValueError(f"Trace/artifact ownership mismatch: {path}.")
        if path.name != f"rank-{rank}.jsonl" or path.parent.name != f"repeat-{owner['lifecycle_repeat']}":
            raise ValueError(f"Rank/repeat path does not match trace owner: {path}.")
        if path.parent.parent.name != owner["mode"]:
            raise ValueError(f"Mode path does not match trace owner: {path}.")
        seen_steps = set()
        for row in rows:
            if row["schema_version"] != SCHEMA_VERSION or any(row[key] != owner[key] for key in OWNER_FIELDS):
                raise ValueError(f"Mixed trace schema/ownership: {path}.")
        for row in rows[1:-1]:
            payload = row["payload"]
            if row["kind"] != "step" or any(payload[key] != owner[key] for key in OWNER_FIELDS):
                raise ValueError(f"Invalid trace step/ownership: {path}.")
            step = payload["target_step_index"]
            if step in seen_steps or (seen_steps and step <= max(seen_steps)):
                raise ValueError(f"Duplicate/out-of-order trace step: {path}.")
            seen_steps.add(step)
            steps.append(payload)
        complete = rows[-1]["payload"]
        if complete["step_count"] != len(rows) - 2 or complete["result"] != expected:
            raise ValueError(f"Trace completion/result artifact mismatch: {path}.")
        summaries[str(rank)] = {
            "path": str(path),
            "header": rows[0]["payload"],
            "step_count": len(rows) - 2,
            "completion": complete["completion"],
        }
    if ranks != set(range(expected_ranks)):
        raise ValueError(f"Incomplete rank-local traces: expected 0..{expected_ranks - 1}, got {sorted(ranks)}.")
    return steps, summaries
