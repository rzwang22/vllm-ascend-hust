# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.dspark import summarize_m2_5a_performance as summary


def _record(
    mode: str,
    *,
    repeat: int,
    decode_seconds: float,
    verification_count: int,
    output_hash: str = "same",
    accepted_per_verification: int = 2,
) -> dict:
    dspark = mode == "dspark"
    accepted = accepted_per_verification * verification_count
    replacement = verification_count
    verification_committed = accepted + replacement
    generated = verification_count + (1 if dspark else 0)
    performance = {field: 0.0 for field in summary.PERFORMANCE_FIELDS}
    performance.update(
        prefill_latency_seconds=0.1,
        decode_latency_seconds=decode_seconds,
        inference_latency_seconds=decode_seconds + 0.1,
        decode_milliseconds_per_output_token=1000.0 * decode_seconds / 99,
        decode_output_tokens_per_second=99 / decode_seconds,
        model_execute_host_seconds=decode_seconds * 0.4,
        sample_materialize_seconds=decode_seconds * 0.3,
        spec_decode_proposer_latency_seconds=(decode_seconds * 0.2 if dspark else 0.0),
        spec_decode_verification_latency_seconds=(decode_seconds * 0.1 if dspark else 0.0),
        prefill_output_token_count=1,
        decode_output_token_count=99,
        scheduled_draft_token_count=(accepted + verification_count if dspark else 0),
        accepted_candidate_metrics_source=summary.ACCEPTED_METRICS_SOURCE,
        accepted_candidate_tokens_total=accepted,
        average_accepted_candidate_tokens_per_verification=(
            accepted / verification_count if verification_count else None
        ),
        replacement_tokens_total=replacement,
        bonus_tokens_total=0,
        committed_tokens_total=100,
        verification_committed_tokens_total=verification_committed,
        terminal_truncated_candidate_tokens=(0 if dspark else None),
        effective_committed_tokens_per_verification=(
            verification_committed / verification_count if verification_count else None
        ),
        accepted_draft_token_count=accepted,
        average_accepted_tokens_per_verification=(accepted / verification_count if verification_count else 0.0),
        draft_token_acceptance_rate=0.5 if dspark else 0.0,
        npu_memory={
            "allocated_before": 1,
            "reserved_before": 1,
            "allocated_after": 1,
            "reserved_after": 1,
            "peak_allocated": 1,
            "peak_reserved": 1,
            "peak_allocated_increment": 0,
            "peak_reserved_increment": 0,
        },
    )
    rank_performance = [
        {"rank": rank, **{field: float(performance[field]) for field in summary.PERFORMANCE_FIELDS}}
        for rank in range(2)
    ]
    return {
        "mode": mode,
        "performance": performance,
        "performance_repeat_index": repeat,
        "request_id": f"{mode}-{repeat}",
        "output_token_count": 100,
        "output_token_sha256": output_hash,
        "target_forward_count": verification_count if dspark else 100,
        "verification_count": verification_count,
        "proposal_generated_count": generated if dspark else 0,
        "proposal_installed_count": verification_count if dspark else 0,
        "proposal_consumed_count": verification_count if dspark else 0,
        "terminal_discarded_proposal_count": 1 if dspark else 0,
        "terminal_partial_commit": False,
        "post_finish_target_forward_count": 0,
        "post_finish_verification_count": 0,
        "cleanup_complete": True,
        "state_isolation_verified": True,
        "historical_error_count": 0,
        "cross_rank_output_consistent": True,
        "_rank_performance": rank_performance,
    }


def test_variable_work_can_have_unstable_raw_and_stable_normalized_decode() -> None:
    records = [
        _record(
            "dspark",
            repeat=index,
            decode_seconds=verification_count / 10,
            verification_count=verification_count,
        )
        for index, verification_count in enumerate((10, 20, 30))
    ]

    result = summary._steady_state_mode_case(records, [], "dspark")

    assert result["raw_decode_stability"] == "unstable"
    assert result["work_normalized_stability"] == "stable"
    assert result["decode_seconds_per_verification_cv"] == pytest.approx(0.0)
    assert result["formal_steady_state_pass"] is True


def test_constant_work_with_variable_per_verification_cost_is_unstable() -> None:
    records = [
        _record(
            "dspark",
            repeat=index,
            decode_seconds=decode,
            verification_count=10,
        )
        for index, decode in enumerate((1.0, 2.0, 4.0))
    ]

    result = summary._steady_state_mode_case(records, [], "dspark")

    assert result["work_normalized_stability"] == "unstable"
    assert result["formal_steady_state_pass"] is False


def test_zero_verification_has_null_normalized_metrics() -> None:
    record = _record("dspark", repeat=0, decode_seconds=1.0, verification_count=0)

    normalized = summary._normalized_repeat_metrics(record, "dspark")

    assert all(normalized[field] is None for field in summary.DSPARK_NORMALIZED_FIELDS)


def _gate_case() -> dict:
    stable_mode = {
        "formal_steady_state_pass": True,
        "raw_decode_stability": "stable",
        "accepted_candidate_tokens_per_verification_cv": 0.0,
        "cross_rank_output_consistent": True,
        "lifecycle_accounting_invariants": True,
    }
    return {
        "decode_speedup": 2.0,
        "tp8_consistent": True,
        "target_only": deepcopy(stable_mode),
        "dspark": deepcopy(stable_mode),
    }


def _slow_events(kind: str | None) -> list[dict]:
    return [
        {
            "slow_host_events": {
                "available": True,
                "warmup_slow_host_event_count": int(kind == "warmup"),
                "measured_slow_host_event_count": int(kind == "measured"),
                "warmup_slow_host_seconds": 53.71 if kind == "warmup" else 0.0,
                "measured_slow_host_seconds": 9.0 if kind == "measured" else 0.0,
            }
        }
        for _ in summary.MODES
    ]


def test_warmup_slow_event_is_retained_but_does_not_fail_formal_gate() -> None:
    gate = summary._work_normalized_gate([{"cases": [_gate_case()]}], _slow_events("warmup"))

    assert gate["warmup_slow_host_event_count"] == 2
    assert gate["measured_slow_host_event_count"] == 0
    assert gate["formal_steady_state_pass"] is True


def test_measured_slow_event_fails_formal_gate() -> None:
    gate = summary._work_normalized_gate([{"cases": [_gate_case()]}], _slow_events("measured"))

    assert gate["measured_slow_host_event_count"] == 2
    assert gate["formal_steady_state_pass"] is False


def test_cross_repeat_output_nondeterminism_is_reported_not_blocking() -> None:
    records = [
        _record(
            "dspark",
            repeat=index,
            decode_seconds=1.0,
            verification_count=10,
            output_hash=output_hash,
        )
        for index, output_hash in enumerate(("a", "b", "a"))
    ]

    result = summary._steady_state_mode_case(records, [], "dspark")

    assert result["cross_repeat_output_deterministic"] is False
    assert result["unique_output_hash_count"] == 2
    assert result["formal_steady_state_pass"] is True


def test_terminal_partial_commit_accounting_is_exact() -> None:
    record = _record("dspark", repeat=0, decode_seconds=1.0, verification_count=1)
    record.update(
        performance_validated=True,
        performance_provisional=True,
        bit_exact_validated=False,
        performance_protocol=None,
        output_token_count=100,
    )
    performance = record["performance"]
    performance.update(
        accepted_candidate_tokens_total=4,
        replacement_tokens_total=1,
        bonus_tokens_total=0,
        scheduled_draft_token_count=5,
        verification_committed_tokens_total=3,
        terminal_truncated_candidate_tokens=2,
        average_accepted_candidate_tokens_per_verification=4.0,
        effective_committed_tokens_per_verification=3.0,
    )

    summary._validate_performance_record(record, "dspark")

    performance["terminal_truncated_candidate_tokens"] = 1
    with pytest.raises(ValueError, match="truncation accounting"):
        summary._validate_performance_record(record, "dspark")


def test_legacy_terminal_truncation_is_not_fabricated() -> None:
    records = [_record("dspark", repeat=index, decode_seconds=1.0, verification_count=10) for index in range(3)]
    for record in records:
        record["performance"].pop("terminal_truncated_candidate_tokens")

    result = summary._steady_state_mode_case(records, [], "dspark")

    assert all(repeat["terminal_truncated_candidate_tokens"] is None for repeat in result["measured_repeats"])
    assert result["lifecycle_accounting_invariants"] is False
    assert result["formal_steady_state_pass"] is False


def test_sample_cv_requires_three_measured_repeats() -> None:
    assert summary._sample_coefficient_of_variation([1.0, 1.0]) is None
    assert summary._sample_stability(None) == "not_formal"
    assert summary._sample_coefficient_of_variation([1.0, 2.0, 3.0]) == pytest.approx(0.5)


@pytest.mark.parametrize(("require_formal", "expected_return"), ((False, 0), (True, 2)))
def test_formal_gate_only_changes_exit_status_when_explicitly_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    require_formal: bool,
    expected_return: int,
) -> None:
    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        run=["target_only=/target", "dspark=/dspark"],
        expected_ranks=8,
        min_runs_per_mode=1,
        error_log_root=[],
        output_json=tmp_path / "summary.json",
        output_csv=tmp_path / "runs.csv",
        output_case_csv=None,
        output_case_markdown=None,
        output_steady_state_csv=None,
        output_steady_state_markdown=None,
        output_repeats_csv=None,
        require_formal_steady_state=require_formal,
    )
    result = {
        "runs": [],
        "speedup": {},
        "matched_case_performance": [],
        "steady_state_case_performance": [],
        "performance_stability_gate": {
            "applicable": True,
            "formal_steady_state_pass": False,
        },
        "telemetry_conclusions": {
            "M2_5A_P04_1_TELEMETRY_PASS": True,
            "TARGET_STEADY_STATE_PASS": False,
            "DSPARK_RAW_DECODE_STABILITY_PASS": False,
            "DSPARK_NORMALIZED_STEADY_STATE_PASS": False,
            "ACCEPTANCE_VARIABILITY_DETECTED": False,
            "MEASURED_SLOW_HOST_EVENT_COUNT": 1,
            "FORMAL_STEADY_STATE_PASS": False,
        },
    }
    monkeypatch.setattr(summary, "_parse_args", lambda: args)
    monkeypatch.setattr(summary, "summarize_performance", lambda *args, **kwargs: result)
    monkeypatch.setattr(summary, "atomic_write_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(summary, "_write_csv", lambda *args, **kwargs: None)

    assert summary.main() == expected_return
