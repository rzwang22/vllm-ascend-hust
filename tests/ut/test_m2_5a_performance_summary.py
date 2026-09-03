# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pytest

from tools.dspark import summarize_m2_5a_performance as summary


def _record(
    *,
    mode: str,
    repeat_index: int,
    repeat_kind: str = "measured",
    decode_seconds: float,
    target_forwards: int,
    verifications: int,
    accepted: int,
    verification_committed: int,
) -> dict[str, Any]:
    output_tokens = 1024
    decode_tokens = output_tokens - 1
    performance = {field: 0.0 for field in summary.PERFORMANCE_FIELDS}
    performance.update(
        prefill_latency_seconds=1.0,
        decode_latency_seconds=decode_seconds,
        inference_latency_seconds=decode_seconds + 1.0,
        milliseconds_per_output_token=1000.0 * (decode_seconds + 1.0) / output_tokens,
        output_tokens_per_second=output_tokens / (decode_seconds + 1.0),
        decode_milliseconds_per_output_token=1000.0 * decode_seconds / decode_tokens,
        decode_output_tokens_per_second=decode_tokens / decode_seconds,
        model_execute_host_seconds=0.4 * target_forwards,
        spec_decode_proposer_latency_seconds=0.1 * verifications,
        spec_decode_verification_latency_seconds=0.2 * verifications,
        prefill_output_token_count=1,
        decode_output_token_count=decode_tokens,
        scheduled_draft_token_count=5 * verifications,
        accepted_draft_token_count=accepted,
        accepted_candidate_metrics_source=summary.ACCEPTED_METRICS_SOURCE,
        accepted_candidate_tokens_total=accepted,
        average_accepted_candidate_tokens_per_verification=(accepted / verifications if verifications else None),
        replacement_tokens_total=verifications,
        bonus_tokens_total=0,
        committed_tokens_total=output_tokens,
        verification_committed_tokens_total=verification_committed,
        effective_committed_tokens_per_verification=(verification_committed / verifications if verifications else None),
        npu_memory={"peak_allocated": 1, "peak_reserved": 2},
    )
    record = {field: None for field in summary.FUNCTIONAL_FIELDS}
    record.update(
        case_id="synthetic:4096:0",
        request_id=f"{mode}-{repeat_kind}-{repeat_index}",
        output_token_count=output_tokens,
        target_forward_count=target_forwards,
        verification_count=verifications,
        proposal_generated_count=verifications,
        proposal_installed_count=verifications,
        proposal_consumed_count=verifications,
        terminal_discarded_proposal_count=0,
        performance_repeat_kind=repeat_kind,
        performance_repeat_index=repeat_index,
        performance=performance,
    )
    return record


def test_population_and_sample_statistics_are_distinct_and_fail_closed() -> None:
    values = [1.0, 2.0, 3.0]

    stats = summary._statistics(values)

    assert stats["population_standard_deviation"] == pytest.approx(math.sqrt(2 / 3))
    assert stats["sample_standard_deviation"] == pytest.approx(1.0)
    assert stats["population_coefficient_of_variation"] == pytest.approx(math.sqrt(2 / 3) / 2)
    assert stats["sample_coefficient_of_variation"] == pytest.approx(0.5)
    assert summary._statistics([1.0])["sample_coefficient_of_variation"] is None
    assert summary._safe_ratio(1.0, 0) is None
    assert summary._pearson_correlation([1.0], [1.0]) is None
    assert summary._pearson_correlation([1.0, 2.0], [1.0, 1.0]) is None


def test_work_normalization_keeps_slow_repeat_and_separates_work_variance() -> None:
    measured = [
        _record(
            mode="dspark",
            repeat_index=index,
            decode_seconds=decode,
            target_forwards=verifications + 1,
            verifications=verifications,
            accepted=accepted,
            verification_committed=committed,
        )
        for index, decode, verifications, accepted, committed in (
            (0, 50.0, 100, 400, 500),
            (1, 100.0, 200, 400, 600),
            (2, 200.0, 400, 400, 800),
        )
    ]
    warmup = [
        _record(
            mode="dspark",
            repeat_index=0,
            repeat_kind="warmup",
            decode_seconds=500.0,
            target_forwards=501,
            verifications=500,
            accepted=400,
            verification_committed=900,
        )
    ]

    case = summary._steady_state_mode_case(measured, warmup, "dspark")

    assert [repeat["decode_latency_seconds"] for repeat in case["measured_repeats"]] == [50.0, 100.0, 200.0]
    assert case["cold_first_use"]["decode_latency_seconds"] == 500.0
    assert case["statistics"]["decode_latency_seconds"]["max"] == 200.0
    assert case["work_count_statistics"]["verification_count"]["sample_coefficient_of_variation"] > 0
    assert (
        case["work_normalized_statistics"]["decode_seconds_per_verification"]["sample_coefficient_of_variation"] == 0.0
    )
    assert case["correlations"]["decode_latency_vs_verification_count"] == pytest.approx(1.0)
    assert case["correlations"]["decode_latency_vs_accepted_candidate_tokens_per_verification"] < 0
    assert case["work_normalized_execution_stability"]["changes_formal_gate"] is False


def test_critical_path_normalization_uses_max_tp_rank_timer() -> None:
    rank_zero = _record(
        mode="dspark",
        repeat_index=0,
        decode_seconds=1.0,
        target_forwards=2,
        verifications=1,
        accepted=4,
        verification_committed=5,
    )
    rank_one = _record(
        mode="dspark",
        repeat_index=0,
        decode_seconds=9.5,
        target_forwards=2,
        verifications=1,
        accepted=4,
        verification_committed=5,
    )

    critical = summary._critical_case_records({0: [rank_zero], 1: [rank_one]})[0]
    metrics = summary._repeat_work_metrics(critical, "dspark")

    assert critical["performance"]["decode_latency_seconds"] == 9.5
    assert metrics["decode_seconds_per_verification"] == 9.5


def test_repeat_csv_reports_warmup_without_counting_it_as_measured(tmp_path: Path) -> None:
    target = summary._steady_state_mode_case(
        [
            _record(
                mode="target_only",
                repeat_index=0,
                decode_seconds=2.0,
                target_forwards=4,
                verifications=0,
                accepted=0,
                verification_committed=0,
            )
        ],
        [
            _record(
                mode="target_only",
                repeat_index=0,
                repeat_kind="warmup",
                decode_seconds=8.0,
                target_forwards=4,
                verifications=0,
                accepted=0,
                verification_committed=0,
            )
        ],
        "target_only",
    )
    dspark = summary._steady_state_mode_case(
        [
            _record(
                mode="dspark",
                repeat_index=0,
                decode_seconds=1.0,
                target_forwards=2,
                verifications=1,
                accepted=4,
                verification_committed=5,
            )
        ],
        [],
        "dspark",
    )
    pairs = [
        {
            "pair_index": 0,
            "cases": [
                {
                    "case_id": "synthetic:4096:0",
                    "profile_case_index": 9,
                    "prompt_token_count": 4096,
                    "output_token_count": 1024,
                    "target_only": target,
                    "dspark": dspark,
                    "tp8_consistent": True,
                }
            ],
        }
    ]
    output = tmp_path / "repeats.csv"

    summary._write_repeat_csv(output, pairs)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))

    assert len(rows) == 3
    assert rows[0]["repeat_kind"] == "warmup"
    assert rows[0]["included_in_measured_statistics"] == "False"
    assert rows[1]["included_in_measured_statistics"] == "True"
    assert rows[2]["decode_seconds_per_verification"] == "1.0"


def test_timer_relationships_remain_non_additive_and_formal_gate_is_raw() -> None:
    assert summary.TIMER_RELATIONSHIPS["inference"]["relation"] == "inclusive_root"
    assert summary.TIMER_RELATIONSHIPS["spec_decode_proposer"]["relation"] == "nested_in_sample_materialize"
    assert "model_execute_host" in summary.TIMER_RELATIONSHIPS["spec_decode_verification"]["overlaps"]
    value = {
        "performance_stability_gate": {
            "applicable": True,
            "passed": False,
            "metric": "max(decode_latency_cv, inference_latency_cv)",
        }
    }
    assert summary._formal_performance_gate_status(value) == "FAIL"


def test_legacy_acceptance_remains_unavailable_and_legacy_csv_prefix_is_stable() -> None:
    legacy = _record(
        mode="dspark",
        repeat_index=0,
        decode_seconds=1.0,
        target_forwards=2,
        verifications=1,
        accepted=4,
        verification_committed=5,
    )
    for field in (
        "accepted_candidate_metrics_source",
        "accepted_candidate_tokens_total",
        "average_accepted_candidate_tokens_per_verification",
        "replacement_tokens_total",
        "bonus_tokens_total",
        "committed_tokens_total",
        "verification_committed_tokens_total",
        "effective_committed_tokens_per_verification",
    ):
        legacy["performance"].pop(field)

    acceptance = summary._case_acceptance_metrics(legacy)

    assert acceptance["available"] is False
    assert acceptance["accepted"] is None
    legacy_csv_fields = (
        "pair_index",
        "case_id",
        "profile_case_index",
        "prompt_token_count",
        "output_token_count",
        "measured_repeat_count",
        "target_cold_decode_seconds",
        "dspark_cold_decode_seconds",
        "target_decode_median_seconds",
        "target_decode_mean_seconds",
        "target_decode_min_seconds",
        "target_decode_max_seconds",
        "target_decode_standard_deviation",
        "target_decode_cv",
        "target_decode_p50_seconds",
        "target_decode_p90_seconds",
        "dspark_decode_median_seconds",
        "dspark_decode_mean_seconds",
        "dspark_decode_min_seconds",
        "dspark_decode_max_seconds",
        "dspark_decode_standard_deviation",
        "dspark_decode_cv",
        "dspark_decode_p50_seconds",
        "dspark_decode_p90_seconds",
        "decode_speedup",
        "inference_speedup",
        "accepted_candidate_tokens_per_verification",
        "effective_committed_tokens_per_verification",
        "proposer_median_seconds",
        "verification_median_seconds",
        "target_stability",
        "dspark_stability",
        "tp8_consistent",
    )
    assert summary.STEADY_STATE_CSV_FIELDS[: len(legacy_csv_fields)] == legacy_csv_fields
