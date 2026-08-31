# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.nightly.single_node.spec_decode import m2_5a_determinism_report as report
from tests.e2e.nightly.single_node.spec_decode import test_dspark_single_request_realdata as harness
from tests.e2e.nightly.single_node.spec_decode.m2_5a_trace_io import rank_trace_writer
from tools.dspark.m2_5a_common import token_ids_sha256, write_jsonl


@pytest.mark.parametrize("limit", [1, 8, 16])
def test_early_trace_requires_exact_case_and_keeps_original_case_identity(limit: int) -> None:
    plan = [{"case_id": "other"}, {"case_id": "synthetic:1024:0"}]
    env = {"DSPARK_M25A_TRACE_EARLY_TOKENS": str(limit)}
    with pytest.raises(ValueError, match="requires an exact"):
        harness._select_forensic_cases(plan, env)
    env["DSPARK_M25A_CASE_ID"] = "synthetic:1024:0"
    selected, config = harness._select_forensic_cases(plan, env)
    assert selected == [plan[1]] and selected[0] is plan[1]
    assert config.early_tokens == limit
    assert not config.first_round and config.output_index is None


@pytest.mark.parametrize("value", ["-1", "17", "1.5", "true", ""])
def test_invalid_early_limit_fails_closed(value: str) -> None:
    with pytest.raises(ValueError, match="must be"):
        harness._select_forensic_cases([], {"DSPARK_M25A_TRACE_EARLY_TOKENS": value})


@pytest.mark.parametrize("old_trace", ["DSPARK_M25A_TRACE_FIRST_ROUND", "DSPARK_M25A_TRACE_OUTPUT_INDEX"])
def test_early_trace_does_not_mix_observer_modes(old_trace: str) -> None:
    with pytest.raises(ValueError, match="must not be combined"):
        harness._select_forensic_cases([], {"DSPARK_M25A_TRACE_EARLY_TOKENS": "16", old_trace: "1"})


@pytest.mark.parametrize("env", [{}, {"DSPARK_M25A_TRACE_EARLY_TOKENS": "0"}])
def test_default_trace_is_off_and_plan_is_not_copied(env: dict[str, str]) -> None:
    plan = [{"case_id": "synthetic:1024:0"}]
    selected, config = harness._select_forensic_cases(plan, env)
    assert selected is plan
    assert config.early_tokens == 0


@pytest.mark.parametrize(
    ("start", "committed", "candidates", "valid"),
    [
        (0, [19], None, [True]),
        (4, [28, 223, 5769, 22, 28036, 35977], [28, 223, 5769, 22, 28036], [True] * 6),
        (15, [28, 88338], [28, 4, 5, 6, 7], [True, True]),
        (4, [28, 999, 8], [28, 4, 5, 6, 7], [True, True, False]),
    ],
)
def test_each_committed_index_has_explicit_logit_and_actual_prefix(
    start: int, committed: list[int], candidates: list[int] | None, valid: list[bool]
) -> None:
    request = SimpleNamespace(prompt_token_ids=[80, 81], output_token_ids=[19] * start + committed)
    record = {
        "output_length_before": start,
        "scheduler_committed_tokens": committed,
        "consumed_candidate_tokens": candidates,
    }
    original = deepcopy((request.__dict__, record))
    trace = harness._committed_prefix_trace(request, record)
    assert trace["commit_start_output_index"] == start
    assert trace["commit_end_output_index_exclusive"] == start + len(committed)
    assert trace["logit_row_matches_committed_prefix"] == valid
    for row in range(len(committed)):
        assert trace["committed_prefix_sha256"][row] == token_ids_sha256(
            [80, 81, *request.output_token_ids[: start + row]]
        )
    assert (request.__dict__, record) == original
    assert json.loads(json.dumps(trace)) == trace


@pytest.mark.parametrize("mode", ["target_only", "dspark"])
def test_early_trace_on_off_preserves_real_harness_commit_and_cleanup_order(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CPU lifecycle doubles exercise the real _run_case, not any NPU kernel."""
    outputs = (
        [[i] for i in range(20)]
        if mode == "target_only"
        else [[0], [1, 2], list(range(3, 9)), list(range(9, 15)), list(range(15, 20))]
    )

    def run(limit: int):
        events, markers, observations = [], [], []
        request = SimpleNamespace(
            prompt_token_ids=[80],
            output_token_ids=[],
            num_computed_tokens=1,
            stop_reason=None,
            is_finished=lambda: len(request.output_token_ids) == 20,
            get_finished_reason=lambda: "length",
        )
        speculator = (
            SimpleNamespace(
                _proposal_step_epoch=0,
                _proposal_consumer_step_epoch=None,
                _proposal_generated_count=0,
                _proposal_installed_count=0,
                _proposal_consumption_count=0,
                _terminal_proposal_discard_count=0,
                _published_candidate_tokens=None,
                _current_proposal_lifecycle=None,
                _last_consumed_proposal_lifecycle=None,
                _terminal_proposal_lifecycle=None,
            )
            if mode == "dspark"
            else None
        )
        index = -1
        finished_scheduled = False
        request_id = f"m25a-{mode}-7-0-synthetic:1024:0"

        def lifecycle(epoch, consumed=False):
            return SimpleNamespace(
                proposal_epoch=epoch,
                owner_epoch=epoch,
                consumer_epoch=epoch + 1 if consumed else None,
                request_ids=(request_id,),
                generated=True,
                returned_to_core=True,
                installed=True,
                consumed=consumed,
                discarded_terminal=False,
                scheduled_lengths=(5,),
                disposition="INSTALLED",
                token_prefix_match=consumed,
                truncated=False,
                dropped=False,
                drop_reason=None,
            )

        def schedule():
            nonlocal index, finished_scheduled
            index += 1
            events.append(("schedule", index))
            if index >= len(outputs):
                ids = {request_id} if not finished_scheduled else set()
                finished_scheduled = True
                return SimpleNamespace(total_num_scheduled_tokens=0, finished_req_ids=ids)
            candidates = list(range(outputs[index][0], outputs[index][0] + 5)) if speculator and index else []
            if candidates and len(outputs[index]) < 6:
                candidates[len(outputs[index]) - 1] = 9999  # Reject, then commit the target correction.
            return SimpleNamespace(
                total_num_scheduled_tokens=len(candidates) + 1,
                finished_req_ids=set(),
                scheduled_spec_decode_tokens={request_id: candidates} if candidates else {},
            )

        def execute(output):
            events.append(("execute", index, sorted(output.finished_req_ids)))
            if speculator and index < len(outputs):
                candidates = output.scheduled_spec_decode_tokens.get(request_id)
                speculator._published_candidate_tokens = [candidates] if candidates else None
                speculator._current_proposal_lifecycle = lifecycle(index) if candidates else None
                speculator._published_proposal_request_ids = (request_id,)
                speculator._published_proposal_request_state_indices = [0]
                runtime.runner.execute_model_state.input_batch = SimpleNamespace(
                    num_reqs=1,
                    req_ids=[request_id],
                    idx_mapping=[0],
                    num_tokens=6,
                    query_start_loc=[0, 6],
                    input_ids=[outputs[index][0] - 1, *candidates] if candidates else [80],
                )
            return None

        def sample(_):
            events.append(("sample", index))
            if speculator:
                speculator._proposal_step_epoch += 1
                speculator._proposal_generated_count += 1
                if index:
                    speculator._proposal_consumer_step_epoch = index + 1
                    speculator._last_consumed_proposal_lifecycle = lifecycle(index, consumed=True)
                    speculator._proposal_consumption_count += 1
                    speculator._proposal_installed_count += 1
            return SimpleNamespace(sampled_token_ids=[outputs[index]])

        def update(_, result):
            events.append(("commit", index, tuple(result.sampled_token_ids[0])))
            request.output_token_ids.extend(result.sampled_token_ids[0])

        scheduler = SimpleNamespace(
            add_request=lambda _: events.append(("add",)),
            schedule=schedule,
            has_requests=lambda: not finished_scheduled,
            update_from_output=update,
            update_draft_token_ids=lambda _: events.append(("update_draft", index)),
            kv_cache_manager=SimpleNamespace(get_block_ids=lambda _: ([10], [20])),
        )
        runtime = SimpleNamespace(
            launch=SimpleNamespace(rank=0),
            speculator=speculator,
            worker=SimpleNamespace(execute_model=execute, sample_tokens=sample, take_draft_token_ids=lambda: object()),
            runner=SimpleNamespace(execute_model_state=SimpleNamespace(input_batch=None)),
            torch=SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)),
        )

        def top2(_):
            observations.append(("top2", index))
            assert limit  # The trace-off path must never recompute logits.
            tokens = outputs[index]
            top1 = [*tokens, *([999] * (6 - len(tokens)))] if speculator and index else tokens
            return {"target_top1_token_ids": top1}

        def next_input(*_, **kwargs):
            observations.append(("next", index, kwargs["commit_end"]))
            assert limit
            return {"next_model_input_available": True}

        monkeypatch.setattr(harness, "_request", lambda *_: request)
        monkeypatch.setattr(harness, "_marker", lambda name, record: markers.append((name, deepcopy(record))))
        monkeypatch.setattr(harness, "_target_top2_trace", top2)
        monkeypatch.setattr(harness, "_next_model_input_trace", next_input)
        monkeypatch.setattr(harness, "_materialize_model_output", lambda output: output)
        monkeypatch.setattr(harness, "_assert_scheduler_proposal_disposition", lambda *_: None)
        monkeypatch.setattr(harness, "_assert_released_request_state", lambda *_: events.append(("released",)))
        case = {
            "case_id": "synthetic:1024:0",
            "dataset": "synthetic",
            "request_sequence_index": 7,
            "lifecycle_repeat": 0,
            "profile_case_index": 7,
            "prompt_token_count": 1,
            "output_cap": 20,
            "ignore_eos": True,
            "ordered_prompt_token_sha256": token_ids_sha256([80]),
        }
        owner = {**case, "rank": 0, "mode": mode, "request_id": request_id}
        root = tmp_path / f"{mode}-{limit}"
        with rank_trace_writer(root, owner, limit) as writer:
            result = harness._run_case(
                runtime,
                scheduler,
                case,
                mode=mode,
                profile="smoke",
                manifest_sha256="manifest",
                trace_early_tokens=limit,
                early_trace_writer=writer,
            )
        early = (
            [row["payload"] for row in map(json.loads, writer.path.read_text().splitlines()) if row["kind"] == "step"]
            if writer is not None
            else []
        )
        assert not any(name == harness.EARLY_RANGE_TRACE for name, _ in markers)
        if not limit:
            assert observations == [] and early == []
            assert not root.exists()
        else:
            traced_indices = [
                i
                for row in early
                for i in range(row["commit_start_output_index"], row["commit_end_output_index_exclusive"])
            ]
            assert traced_indices[:16] == list(range(16))
            for row in early:
                assert row["raw_sampled_tokens"] == row["scheduler_committed_tokens"] == row["artifact_appended_tokens"]
                assert row["logit_row_matches_committed_prefix"] == [True] * len(row["scheduler_committed_tokens"])
                if row["step_kind"] == "verification":
                    assert row["raw_matches_expected_greedy"] is True
                    assert row["bonus_contract_valid"] is True
        return result, events

    runs = [run(0), run(16)]
    assert runs[0] == runs[1]
    assert runs[0][0]["output_token_ids"] == list(range(20))
    assert runs[0][1][-1] == ("released",)


def _stream(tmp_path: Path, name: str, tokens: list[int], *, ranks: int = 2) -> Path:
    root = tmp_path / name
    root.mkdir()
    for rank in range(ranks):
        row = {field: 0 for field in report.PAIR_FIELDS}
        row.update(
            rank=rank,
            mode=name,
            request_id="request",
            output_token_ids=tokens,
            output_token_count=len(tokens),
            output_token_sha256=token_ids_sha256(tokens),
        )
        write_jsonl(root / f"rank-{rank}.jsonl", [row])
    return root


def test_full_alignment_reports_all_edits_and_rank_coverage(tmp_path: Path) -> None:
    left = [1, 2, 3, 4, 5, 6]
    right = [1, 9045, 2, 3, 44, 5, 6, 7]
    streams = {"left": _stream(tmp_path, "left", left), "right": _stream(tmp_path, "right", right)}
    result = report.build_report(streams, {}, expected_ranks=2)
    pair = result["comparisons"][0]
    assert pair["first_different_index"] == 1
    assert [edit["tag"] for edit in pair["edits"]] == ["insert", "replace", "insert"]
    assert pair["same_logit_prefix"] is None
    assert result["streams"]["right"]["token_9045_positions"] == [1]
    assert result["streams"]["left"]["records"]["0"]["output_token_ids"] == left
    assert result["streams"]["left"]["rank_identity"] is True
    with pytest.raises(ValueError, match="Incomplete rank"):
        report.build_report(streams, {}, expected_ranks=8)
    assert json.loads(json.dumps(result))["exact_gate_changed"] is False


def test_report_does_not_hide_rank_drift_or_mispairing(tmp_path: Path) -> None:
    root = _stream(tmp_path, "left", [1, 2])
    path = root / "rank-1.jsonl"
    row = json.loads(path.read_text())
    row.update(output_token_ids=[1, 3], output_token_sha256=token_ids_sha256([1, 3]))
    write_jsonl(path, [row])
    other = _stream(tmp_path, "right", [1, 2])
    row = json.loads((other / "rank-0.jsonl").read_text())
    row["prompt_token_sha256"] = "different"
    write_jsonl(other / "rank-0.jsonl", [row])
    result = report.build_report({"left": root, "right": other}, {})
    assert result["streams"]["left"]["rank_identity"] is False
    assert result["comparisons"][0]["pairing_mismatches"]["prompt_token_sha256"] == [0, "different"]


def test_trace_comparison_requires_coverage_and_matching_logical_prefix() -> None:
    record = {
        "rank": 0,
        "commit_start_output_index": 4,
        "commit_end_output_index_exclusive": 6,
        "target_top1_token_ids": [28, 223],
        "target_top2_token_ids": [13226, 9045],
        "target_top1_top2_margins": [0.125, 1],
        "logits_positions": [1027, 1028],
        "logits_input_ids": [5664, 28],
        "logit_row_prefix_sha256": ["a", "b"],
        "logit_row_matches_committed_prefix": [True, False],
    }
    assert report.trace_at_index([record], 3) is None
    assert report.trace_at_index([record], 6) is None
    trace = report.trace_at_index([record], 5)
    assert trace["logit_row"] == 1
    assert trace["logit_prefix_matches_committed"] is False
    assert trace["top1"] == 223


def test_report_pairs_full_trace_with_artifact_and_reports_rank_identity(tmp_path: Path) -> None:
    streams = {"left": _stream(tmp_path, "left", [19, 28]), "right": _stream(tmp_path, "right", [19, 582])}
    trace_paths = {}
    for name, token in (("left", 28), ("right", 582)):
        records = []
        for rank in range(2):
            record = {field: 0 for field in report.PAIR_FIELDS}
            record.update(
                mode=name,
                request_id="request",
                rank=rank,
                target_step_index=1,
                commit_start_output_index=1,
                commit_end_output_index_exclusive=2,
                target_top1_token_ids=[token],
                target_top2_token_ids=[13226],
                target_top1_top2_margins=[0.125],
                logits_positions=[1024],
                logits_input_ids=[19],
                logit_row_prefix_sha256=["same-prefix"],
                logit_row_matches_committed_prefix=[True],
            )
            records.append(record)
        path = tmp_path / f"{name}.log"
        path.write_text("\n".join(report.TRACE_MARKERS[0] + json.dumps(row) for row in records))
        trace_paths[name] = path
    result = report.build_report(streams, trace_paths, expected_ranks=2)
    assert result["comparisons"][0]["same_logit_prefix"] is True
    assert result["comparisons"][0]["first_difference_traces"][0]["top1"] == 28
    assert all(stream["trace_rank_identity"] is True for stream in result["streams"].values())
    path = trace_paths["right"]
    path.write_text(path.read_text().replace('"request_id": "request"', '"request_id": "other"'))
    with pytest.raises(ValueError, match="ownership mismatch"):
        report.build_report(streams, trace_paths)


def test_report_rejects_corrupt_token_hash(tmp_path: Path) -> None:
    root = _stream(tmp_path, "left", [1, 2])
    path = root / "rank-0.jsonl"
    row = json.loads(path.read_text())
    row["output_token_ids"][1] = 999
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="Invalid token"):
        report.load_stream(root)


def test_alignment_does_not_claim_a_single_insertion_explains_later_edits() -> None:
    result = report.align_tokens([19, 16, 223, 7, 8, 9], [19, 16, 9045, 223, 7, 88])
    assert result["first_different_index"] == 2
    assert len(result["edits"]) == 2
