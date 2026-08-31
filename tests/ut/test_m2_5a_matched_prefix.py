# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from tests.e2e.nightly.single_node.spec_decode import m2_5a_determinism_report as report
from tests.e2e.nightly.single_node.spec_decode.m2_5a_matched_prefix import matched_prefix_report, semantic_checks
from tests.e2e.nightly.single_node.spec_decode.m2_5a_trace_io import load_rank_traces, rank_trace_writer
from tests.e2e.nightly.single_node.spec_decode.test_dspark_single_request_realdata import _committed_prefix_trace
from tools.dspark.m2_5a_common import token_ids_sha256, write_jsonl


def _artifact(mode: str, rank: int) -> dict:
    return {
        "mode": mode,
        "rank": rank,
        "case_id": "synthetic:1024:0",
        "lifecycle_repeat": 0,
        "request_sequence_index": 7,
        "request_id": f"request-{mode}",
        "prompt_token_count": 1,
        "prompt_token_sha256": token_ids_sha256([80]),
        "manifest_sha256": "frozen-manifest",
        "output_cap": 3,
        "ignore_eos": True,
        "output_token_ids": [19, 28, 223],
        "output_token_count": 3,
        "output_token_sha256": token_ids_sha256([19, 28, 223]),
        "cleanup_complete": True,
        "state_isolation_verified": True,
        "proposal_consumed_count": int(mode == "dspark"),
        "verification_count": int(mode == "dspark"),
        "terminal_discarded_proposal_count": int(mode == "dspark"),
        "proposal_epoch_end": 2 if mode == "dspark" else None,
    }


def _lifecycle(artifact: dict, *, consumed: bool = False, terminal: bool = False, size: int = 5) -> dict:
    return {
        "proposal_epoch": 2 if terminal else 1,
        "owner_epoch": 2 if terminal else 1,
        "consumer_epoch": 2 if consumed else None,
        "request_ids": [artifact["request_id"]],
        "generated": True,
        "returned_to_core": True,
        "installed": not terminal,
        "consumed": consumed,
        "discarded_terminal": terminal,
        "scheduled_lengths": [] if terminal else [size],
        "disposition": "DROPPED" if terminal else "INSTALLED" if size == 5 else "TRUNCATED",
        "token_prefix_match": True if consumed else None,
        "truncated": size < 5,
        "dropped": terminal,
        "drop_reason": "terminal" if terminal else None,
    }


def _steps(artifact: dict, *, bonus: bool = False) -> list[dict]:
    """Construct host-only checker inputs, not fake production/NPU execution."""
    from types import SimpleNamespace

    rows = []
    starts = [0, 1] if artifact["mode"] == "dspark" else [0, 1, 2]
    for step, start in enumerate(starts):
        verification = artifact["mode"] == "dspark" and start == 1
        end = 3 if verification else start + 1
        commit = artifact["output_token_ids"][start:end]
        size = (1 if bonus else 5) if verification else 0
        candidates = [28, 44, 45, 46, 47] if verification else None
        top1 = [28, 223] + ([] if bonus else [999] * 4) if verification else commit
        row = {
            **{
                key: artifact[key]
                for key in ("rank", "mode", "case_id", "request_id", "lifecycle_repeat", "request_sequence_index")
            },
            "target_step_index": step,
            "step_kind": "verification" if verification else "target_decode",
            "output_length_before": start,
            "scheduler_committed_tokens": commit,
            "artifact_appended_tokens": commit[:],
            "raw_sampled_tokens": commit[:],
            "consumed_candidate_tokens": candidates[:size] if candidates else None,
            "published_candidate_tokens": candidates,
            "scheduled_candidate_tokens": [-1] * size if verification else None,
            "scheduled_proposal_length": size,
            "early_output_token_limit": 3,
            "target_top1_token_ids": top1,
            "target_top2_token_ids": [777] * len(top1),
            "target_top1_logits": [8.0] * len(top1),
            "target_top2_logits": [7.0] * len(top1),
            "target_top1_top2_margins": [1.0] * len(top1),
            "logits_observation": "recomputed_from_target_hidden_states_before_sampling",
            "logits_positions": list(range(start, start + len(top1))),
            "logits_input_ids": [
                80 if start == 0 else artifact["output_token_ids"][start - 1],
                *(candidates[:size] if candidates else []),
            ],
            "query_lengths": [size + 1 if verification else 1],
            "sampling_state": {"temperature": [0], "seeds": [0]},
            "model_seed": 0,
            "execution_path": {"dtype": "bfloat16", "kernel_tiling_observed": False},
            "next_model_input_available": end != 3,
            "next_model_input_terminal_reason": "request_finished" if end == 3 else None,
            "next_runner_prefix_matches_request": True,
            "next_runner_prefix_sha256": token_ids_sha256([80, *artifact["output_token_ids"][:end]]),
            "next_prefix_token_sha256": token_ids_sha256([80, *artifact["output_token_ids"][:end]]),
            "next_request_logical_length": 1 + end,
        }
        row.update(
            _committed_prefix_trace(
                SimpleNamespace(prompt_token_ids=[80], output_token_ids=artifact["output_token_ids"]), row
            )
        )
        if verification:
            row.update(
                lifecycle_before_sample=_lifecycle(artifact, size=size),
                consumed_lifecycle=_lifecycle(artifact, consumed=True, size=size),
                proposal_epoch=1,
                consumer_epoch=2,
                published_owner_request_ids=[artifact["request_id"]],
                verification_request_ids=[artifact["request_id"]],
                published_owner_state_indices=[0],
                verification_state_indices=[0],
                counters_before_sample={"consumed": 0},
                counters_after_sample={"consumed": 1},
                expected_greedy_tokens=commit[:],
                accepted_prefix_length=1,
                bonus_used=bonus,
                replacement_used=not bonus,
                bonus_token=223 if bonus else None,
                replacement_token=None if bonus else 223,
                num_sampled=[2],
                num_rejected=[size - 1],
            )
        rows.append(row)
    return rows


def _completion(artifact: dict) -> dict:
    return {
        "finished_event_observed_count": 1,
        "finished_event_worker_delivery_count": 1,
        "released_scheduler_runner_kv_proposal_state": True,
        "terminal_lifecycle": _lifecycle(artifact, terminal=True) if artifact["mode"] == "dspark" else None,
        "consumer_epochs": [2] if artifact["mode"] == "dspark" else [],
    }


def _write_launch(root: Path, mode: str, ranks: int = 8) -> None:
    def write_rank(rank: int) -> None:
        artifact = _artifact(mode, rank)
        with rank_trace_writer(root, artifact, 3) as writer:
            for row in _steps(artifact):
                writer.write_step(row)
            writer.finish(artifact, _completion(artifact))
        write_jsonl(root / f"rank-{rank}.jsonl", [artifact])

    with ThreadPoolExecutor(max_workers=ranks) as pool:
        list(pool.map(write_rank, range(ranks)))


def test_eight_concurrent_rank_files_and_strict_semantic_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    streams, traces = {}, {}
    for name, mode in (("target", "target_only"), ("draft", "dspark")):
        root = tmp_path / name
        _write_launch(root, mode)
        streams[name], traces[name] = root, root / "traces"
    result = report.build_report(streams, traces, expected_ranks=8)
    assert result["hard_semantic_gates_pass"] is True
    assert all(s["trace_rank_identity"] for s in result["streams"].values())
    assert result["matched_prefix"]["eligible_cross_path_pair_count"] == 2
    assert result["comparisons"][0]["cross_launch_exact_match"] is True
    assert result["whole_generation_correctness_proven"] is False
    assert result["exact_gate_changed"] is False
    stdout = capsys.readouterr().out
    assert "scheduler_committed_tokens" not in stdout
    assert stdout.count("DSPARK_M2_5A_RANK_TRACE_FILE=") == 16
    assert json.loads(json.dumps(result))["matched_prefix"]["tolerance"] is None


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "malformed", "truncated", "no_footer", "nan", "wrong_owner"]
)
def test_corrupt_or_incomplete_rank_evidence_fails_closed(tmp_path: Path, damage: str) -> None:
    _write_launch(tmp_path, "dspark")
    root = tmp_path / "traces"
    path = root / "dspark/repeat-0/rank-7.jsonl"
    if damage == "missing":
        path.unlink()
    elif damage == "duplicate":
        duplicate = root / "copy/dspark/repeat-0/rank-7.jsonl"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(path.read_bytes())
    elif damage == "truncated":
        path.write_text(path.read_text()[:-5])
    elif damage == "malformed":
        path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": {bad', 1))
    elif damage == "no_footer":
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    elif damage == "nan":
        path.write_text(path.read_text().replace('"early_output_token_limit": 3', '"early_output_token_limit": NaN', 1))
    else:
        path.write_text(path.read_text().replace('"request_id": "request-dspark"', '"request_id": "foreign"', 1))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_rank_traces(root, {str(rank): _artifact("dspark", rank) for rank in range(8)}, 8)


def test_disabled_writer_creates_nothing_and_failure_closes_without_completion(tmp_path: Path) -> None:
    artifact = _artifact("dspark", 0)
    root = tmp_path / "off"
    with rank_trace_writer(root, artifact, 0) as writer:
        assert writer is None
    assert not root.exists()
    with (
        pytest.raises(RuntimeError, match="primary generation"),
        rank_trace_writer(tmp_path / "failed", artifact, 16) as writer,
    ):
        writer.write_step(_steps(artifact)[0])
        raise RuntimeError("primary generation")
    assert writer.handle.closed
    assert '"kind": "complete"' not in writer.path.read_text()
    with pytest.raises(FileExistsError), rank_trace_writer(tmp_path / "failed", artifact, 16):
        pass


@pytest.mark.parametrize("bonus", [False, True])
def test_replacement_and_bonus_semantics_are_exact(bonus: bool) -> None:
    artifact = _artifact("dspark", 0)
    stream = {
        "rank_identity": True,
        "trace_rank_identity": True,
        "records": {"0": artifact},
        "traces": _steps(artifact, bonus=bonus),
        "rank_trace_summaries": {"0": {"header": {"early_output_token_limit": 3}, "completion": _completion(artifact)}},
    }
    assert semantic_checks(stream)["status"] == "OBSERVED_GATES_PASS"
    damaged = deepcopy(stream)
    damaged["traces"][1]["bonus_used"] = not bonus
    assert semantic_checks(damaged)["status"] == "FAIL"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("consumed_candidate_tokens", [99] * 5),
        ("consumer_epoch", 99),
        ("verification_request_ids", ["foreign"]),
        ("verification_state_indices", [7]),
        ("raw_sampled_tokens", [28, 99]),
        ("artifact_appended_tokens", [28, 99]),
        ("num_rejected", [0]),
        ("accepted_prefix_length", 0),
        ("counters_after_sample", {"consumed": 2}),
    ],
)
def test_hard_semantic_violations_never_become_pass(field: str, value) -> None:
    artifact = _artifact("dspark", 0)
    rows = _steps(artifact)
    rows[1][field] = value
    stream = {
        "rank_identity": True,
        "trace_rank_identity": True,
        "records": {"0": artifact},
        "traces": rows,
        "rank_trace_summaries": {"0": {"header": {"early_output_token_limit": 3}, "completion": _completion(artifact)}},
    }
    assert semantic_checks(stream)["status"] == "FAIL"


def test_matched_prefix_uses_identity_not_index_and_no_arbitrary_tolerance() -> None:
    streams = {}
    for name, mode, delta in (("t1", "target_only", 0.0), ("t2", "target_only", 0.25), ("d1", "dspark", 0.125)):
        artifact = _artifact(mode, 0)
        rows = _steps(artifact)
        for row in rows:
            row["target_top1_logits"] = [value + delta for value in row["target_top1_logits"]]
            row["target_top1_top2_margins"] = [value + delta for value in row["target_top1_top2_margins"]]
        streams[name] = {"records": {"0": artifact}, "traces": rows}
    result = matched_prefix_report(streams)
    assert result["eligible_cross_path_pair_count"] == 4
    group = next(g for g in result["groups"] if g["identity"]["position"] == 1)
    target = next(e for e in group["same_path_empirical_envelopes"] if e["path"][0] == "target_only")
    assert target["tokens"]["28"]["observed_spread"] == 0.25
    assert len(target["tokens"]["28"]["raw_samples"]) == 2
    pair = group["cross_path_comparisons"][0]
    assert pair["common_token_logit_deltas_right_minus_left"]["28"] == 0.125
    assert result["tolerance"] is None and result["production_correctness_proven"] is False
    for row in streams["d1"]["traces"]:
        row["logit_row_output_prefix_sha256"] = ["different"] * len(row["scheduler_committed_tokens"])
    assert matched_prefix_report(streams)["eligible_cross_path_pair_count"] == 0


def test_missing_rank_local_evidence_is_not_a_semantic_pass() -> None:
    assert semantic_checks({})["status"] == "UNAVAILABLE"


@pytest.mark.parametrize(
    "field",
    [
        "case_id",
        "lifecycle_repeat",
        "prompt_token_sha256",
        "logits_input_ids",
        "logits_positions",
        "logit_row_request_logical_lengths",
    ],
)
def test_every_matched_prefix_identity_component_is_required(field: str) -> None:
    streams = {}
    for mode in ("target_only", "dspark"):
        artifact = _artifact(mode, 0)
        streams[mode] = {"records": {"0": artifact}, "traces": _steps(artifact)}
    for row in streams["dspark"]["traces"]:
        value = row[field]
        row[field] = [99999] * len(value) if isinstance(value, list) else 99 if isinstance(value, int) else "other"
    assert matched_prefix_report(streams)["eligible_cross_path_pair_count"] == 0


def test_valid_legacy_rows_are_exploratory_not_a_fabricated_error_envelope() -> None:
    streams = {}
    for label, mode in (("t1", "target_only"), ("t2", "target_only"), ("d1", "dspark")):
        artifact = _artifact(mode, 0)
        rows = _steps(artifact)
        for row in rows:
            for field in ("logit_row_output_prefix_sha256", "logit_row_request_logical_lengths", "execution_path"):
                del row[field]
        streams[label] = {"records": {"0": artifact}, "traces": rows}
    result = matched_prefix_report(streams)
    assert result["eligible_cross_path_pair_count"] == 4
    assert all(sample["legacy_derived_identity"] for group in result["groups"] for sample in group["samples"])
    assert all(
        token["observed_spread"] is None
        for group in result["groups"]
        for env in group["same_path_empirical_envelopes"]
        for token in env["tokens"].values()
    )


def test_report_rejects_the_same_launch_renamed_as_two_repeats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not independent launches"):
        report.build_report({"repeat1": tmp_path, "repeat2": tmp_path}, {})


def test_cli_keeps_report_and_fails_on_semantic_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    roots = {name: tmp_path / name for name in ("t", "d")}
    for name, mode in (("t", "target_only"), ("d", "dspark")):
        _write_launch(roots[name], mode, ranks=1)
    path = roots["d"] / "traces/dspark/repeat-0/rank-0.jsonl"
    rows = list(map(json.loads, path.read_text().splitlines()))
    rows[2]["payload"]["consumer_epoch"] = 999
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--stream",
            f"t={roots['t']}",
            "--stream",
            f"d={roots['d']}",
            "--trace",
            f"t={roots['t'] / 'traces'}",
            "--trace",
            f"d={roots['d'] / 'traces'}",
            "--expected-ranks",
            "1",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Observed semantic violation"):
        report.main()
    result = json.loads(output.read_text())
    assert result["hard_semantic_gates_pass"] is False
    assert result["streams"]["d"]["hard_semantic_gates"]["status"] == "FAIL"
