#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dspark.m2_5a_common import atomic_write_json, sha256_file, verify_asset_bundle
from tools.dspark.validate_m2_5a_results import HISTORICAL_ERROR_MARKERS, load_results

MODES = ("target_only", "dspark")
PERFORMANCE_FIELDS = (
    "prefill_latency_seconds",
    "decode_latency_seconds",
    "inference_latency_seconds",
    "milliseconds_per_output_token",
    "output_tokens_per_second",
    "decode_milliseconds_per_output_token",
    "decode_output_tokens_per_second",
    "scheduler_seconds",
    "scheduler_update_seconds",
    "model_execute_host_seconds",
    "sample_materialize_seconds",
    "draft_install_seconds",
    "spec_decode_proposer_latency_seconds",
    "spec_decode_verification_latency_seconds",
)
ACCEPTED_METRICS_SOURCE = "greedy_rejection_output_length_minus_one"
ACCEPTED_COUNT_FIELDS = (
    "accepted_candidate_tokens_total",
    "replacement_tokens_total",
    "bonus_tokens_total",
    "committed_tokens_total",
    "verification_committed_tokens_total",
)
TIMER_RELATIONSHIPS = {
    "inference": {
        "relation": "inclusive_root",
        "boundary": "request admission through final scheduler commit and final NPU synchronization",
    },
    "prefill": {
        "relation": "nested_prefix_of_inference",
        "boundary": "inference start through first scheduler commit",
    },
    "decode": {
        "relation": "derived_disjoint_remainder",
        "boundary": "inference minus prefill",
    },
    "scheduler": {"relation": "nested_host_interval_in_inference"},
    "scheduler_update": {"relation": "nested_host_interval_in_inference"},
    "model_execute_host": {
        "relation": "nested_host_enqueue_interval_in_inference",
        "warning": "does not exclusively measure asynchronous NPU model execution",
    },
    "sample_materialize": {
        "relation": "nested_in_inference",
        "contains": ["rejection sampling", "output materialization", "spec_decode_proposer"],
        "overlaps": ["spec_decode_verification"],
    },
    "draft_install": {"relation": "nested_host_interval_in_inference"},
    "spec_decode_proposer": {
        "relation": "nested_in_sample_materialize",
        "disjoint_from": ["spec_decode_verification"],
    },
    "spec_decode_verification": {
        "relation": "nested_in_inference",
        "boundary": "before target forward through target sampling",
        "overlaps": ["model_execute_host", "sample_materialize"],
        "warning": "current core starts this timer for every speculator target forward, including producer steps",
    },
    "model_load": {"relation": "disjoint_before_inference"},
    "kv_cache_initialization": {"relation": "disjoint_before_inference"},
}
COUNT_FIELDS = (
    "target_forward_count",
    "verification_count",
    "proposal_generated_count",
    "proposal_installed_count",
    "proposal_consumed_count",
    "terminal_discarded_proposal_count",
)
FUNCTIONAL_FIELDS = (
    "case_id",
    "dataset",
    "profile",
    "lifecycle_repeat",
    "request_sequence_index",
    "prompt_token_count",
    "prompt_token_sha256",
    "output_cap",
    "ignore_eos",
    "output_token_count",
    "output_token_ids",
    "output_token_sha256",
    "stop_reason",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "completed_rounds",
    *COUNT_FIELDS,
    "terminal_partial_commit",
    "cleanup_complete",
    "state_isolation_verified",
    "historical_error_count",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize provisional M2.5A target-only/DSpark performance artifacts."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="MODE=RESULT_DIR",
        help="Repeat for every independent target_only or dspark run.",
    )
    parser.add_argument(
        "--error-log-root",
        action="append",
        type=Path,
        default=[],
        help="Optional log tree to scan for historical fatal signatures.",
    )
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--min-runs-per-mode", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-case-csv", type=Path)
    parser.add_argument("--output-case-markdown", type=Path)
    return parser.parse_args()


def _parse_run(value: str) -> tuple[str, Path]:
    mode, separator, root = value.partition("=")
    if not separator or mode not in MODES or not root:
        raise ValueError(f"--run must be target_only=PATH or dspark=PATH, got {value!r}.")
    return mode, Path(root).expanduser().resolve()


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Performance field {field} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Performance field {field} must be finite and non-negative, got {value!r}.")
    return result


def _accepted_metrics_available(performance: Mapping[str, Any]) -> bool:
    source = performance.get("accepted_candidate_metrics_source")
    if source is None:
        return False
    if source != ACCEPTED_METRICS_SOURCE:
        raise ValueError(f"Unknown accepted-candidate telemetry source: {source!r}.")
    return True


def _validate_performance_record(record: Mapping[str, Any], mode: str) -> None:
    if record.get("performance_validated") is not True or record.get("performance_provisional") is not True:
        raise ValueError("Result was not produced by provisional M2.5A performance mode.")
    if record.get("bit_exact_validated") is not False:
        raise ValueError("Performance artifacts must not claim bit-exact validation.")
    performance = record.get("performance")
    if not isinstance(performance, Mapping):
        raise ValueError("Performance result has no structured performance payload.")
    for field in PERFORMANCE_FIELDS:
        _finite_nonnegative(performance.get(field), field)
    for field in (
        "prefill_output_token_count",
        "decode_output_token_count",
        "scheduled_draft_token_count",
        "accepted_draft_token_count",
    ):
        value = performance.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Performance count {field} must be a non-negative integer.")
    if (
        performance["prefill_output_token_count"] + performance["decode_output_token_count"]
        != record["output_token_count"]
    ):
        raise ValueError("Prefill/decode token accounting differs from the generated output count.")
    if _accepted_metrics_available(performance):
        for field in ACCEPTED_COUNT_FIELDS:
            value = performance.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Accepted-candidate count {field} must be a non-negative integer.")
        verification_count = record["verification_count"]
        accepted = performance["accepted_candidate_tokens_total"]
        scheduled = performance["scheduled_draft_token_count"]
        replacements = performance["replacement_tokens_total"]
        bonuses = performance["bonus_tokens_total"]
        committed = performance["verification_committed_tokens_total"]
        if accepted > scheduled:
            raise ValueError("Accepted candidate count exceeds scheduled candidates.")
        if replacements + bonuses != verification_count:
            raise ValueError("Every verification must emit exactly one replacement or bonus token.")
        if performance["committed_tokens_total"] != record["output_token_count"]:
            raise ValueError("Committed-token telemetry differs from the generated output count.")
        if committed > record["output_token_count"]:
            raise ValueError("Verification commits exceed the generated output count.")
        average = performance.get("average_accepted_candidate_tokens_per_verification")
        effective = performance.get("effective_committed_tokens_per_verification")
        if verification_count:
            if not math.isclose(float(average), accepted / verification_count):
                raise ValueError("Average accepted-candidate telemetry is inconsistent.")
            if not math.isclose(float(effective), committed / verification_count):
                raise ValueError("Effective verification progress telemetry is inconsistent.")
        elif average is not None or effective is not None:
            raise ValueError("Non-speculative results must use null verification averages.")
    memory = performance.get("npu_memory")
    if not isinstance(memory, Mapping):
        raise ValueError("Performance result has no NPU memory payload.")
    for field in (
        "allocated_before",
        "reserved_before",
        "allocated_after",
        "reserved_after",
        "peak_allocated",
        "peak_reserved",
        "peak_allocated_increment",
        "peak_reserved_increment",
    ):
        value = memory.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"NPU memory field {field} must be a non-negative integer.")
    if record["cleanup_complete"] is not True or record["state_isolation_verified"] is not True:
        raise ValueError("Performance result did not prove request cleanup/state isolation.")
    if record["historical_error_count"] != 0:
        raise ValueError("Performance result recorded a historical runtime error.")
    if mode == "target_only":
        if any(record[field] != 0 for field in COUNT_FIELDS[1:]):
            raise ValueError("Target-only performance result contains speculative activity.")
    else:
        if record["proposal_installed_count"] != record["proposal_consumed_count"]:
            raise ValueError("DSpark installed proposals were not consumed exactly once.")
        if record["verification_count"] != record["proposal_consumed_count"]:
            raise ValueError("DSpark verification count differs from consumed proposal count.")
        if record["proposal_generated_count"] < record["proposal_installed_count"]:
            raise ValueError("DSpark installed more proposals than it generated.")


def _rank_summary(root: Path, rank: int) -> dict[str, Any]:
    path = root / f"rank-{rank}.summary.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read performance rank summary: {path}.") from exc
    if not isinstance(value, dict) or value.get("rank") != rank:
        raise ValueError(f"Invalid performance rank summary ownership: {path}.")
    if value.get("performance_validated") is not True or value.get("performance_provisional") is not True:
        raise ValueError(f"Rank summary was not emitted in performance mode: {path}.")
    phase_timings = value.get("phase_timings")
    if not isinstance(phase_timings, Mapping):
        raise ValueError(f"Rank summary has no phase timings: {path}.")
    for field in ("init_device_seconds", "model_load_seconds", "kv_cache_init_seconds"):
        _finite_nonnegative(phase_timings.get(field), field)
    return value


def _critical_case_records(
    rank_records: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    baseline = rank_records[0]
    critical: list[dict[str, Any]] = []
    for index, expected in enumerate(baseline):
        per_rank = [records[index] for records in rank_records.values()]
        for rank, actual in zip(rank_records, per_rank):
            for field in FUNCTIONAL_FIELDS:
                if actual.get(field) != expected.get(field):
                    raise ValueError(f"Rank {rank} disagrees with rank 0 for {expected['case_id']} field {field}.")
            for field in (
                "prefill_output_token_count",
                "decode_output_token_count",
                "scheduled_draft_token_count",
                "accepted_draft_token_count",
            ):
                if actual["performance"][field] != expected["performance"][field]:
                    raise ValueError(
                        f"Rank {rank} disagrees with rank 0 for {expected['case_id']} performance count {field}."
                    )
            expected_source = expected["performance"].get("accepted_candidate_metrics_source")
            actual_source = actual["performance"].get("accepted_candidate_metrics_source")
            if actual_source != expected_source:
                raise ValueError(
                    f"Rank {rank} disagrees with rank 0 for {expected['case_id']} accepted-metrics source."
                )
            if expected_source is not None:
                for field in ACCEPTED_COUNT_FIELDS:
                    if actual["performance"].get(field) != expected["performance"].get(field):
                        raise ValueError(
                            f"Rank {rank} disagrees with rank 0 for {expected['case_id']} "
                            f"accepted-candidate count {field}."
                        )
        record = dict(expected)
        performance = dict(expected["performance"])
        for field in PERFORMANCE_FIELDS:
            performance[field] = max(float(actual["performance"][field]) for actual in per_rank)
        output_tokens = int(expected["output_token_count"])
        decode_tokens = int(performance["decode_output_token_count"])
        inference_seconds = float(performance["inference_latency_seconds"])
        decode_seconds = float(performance["decode_latency_seconds"])
        performance["milliseconds_per_output_token"] = 1000.0 * inference_seconds / output_tokens
        performance["output_tokens_per_second"] = output_tokens / inference_seconds
        performance["decode_milliseconds_per_output_token"] = (
            1000.0 * decode_seconds / decode_tokens if decode_tokens else 0.0
        )
        performance["decode_output_tokens_per_second"] = (
            decode_tokens / decode_seconds if decode_tokens and decode_seconds else 0.0
        )
        memory = dict(expected["performance"]["npu_memory"])
        for field in memory:
            memory[field] = max(int(actual["performance"]["npu_memory"][field]) for actual in per_rank)
        performance["npu_memory"] = memory
        record["performance"] = performance
        critical.append(record)
    return critical


def _case_acceptance_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    performance = record["performance"]
    verification_count = int(record["verification_count"])
    if _accepted_metrics_available(performance):
        accepted = int(performance["accepted_candidate_tokens_total"])
        return {
            "available": True,
            "source": performance["accepted_candidate_metrics_source"],
            "accepted": accepted,
            "accepted_lower_bound": accepted,
            "accepted_upper_bound": accepted,
            "replacement": int(performance["replacement_tokens_total"]),
            "bonus": int(performance["bonus_tokens_total"]),
            "verification_committed": int(performance["verification_committed_tokens_total"]),
        }

    if verification_count == 0:
        return {
            "available": False,
            "source": "not_applicable_without_verification",
            "accepted": None,
            "accepted_lower_bound": 0,
            "accepted_upper_bound": 0,
            "replacement": None,
            "bonus": None,
            "verification_committed": 0,
        }

    # Legacy P0 artifacts compared the synchronous scheduler's -1 proposal
    # placeholders with sampled IDs. That zero is not acceptance telemetry.
    nonverification_forwards = int(record["target_forward_count"]) - verification_count
    verification_committed = int(record["output_token_count"]) - nonverification_forwards
    if verification_committed < 0:
        raise ValueError("Unable to derive legacy verification commit progress.")
    accepted_lower = max(0, verification_committed - verification_count)
    accepted_upper = accepted_lower
    if record["terminal_partial_commit"] and verification_count:
        # A terminal partial commit omitted at least one and at most K raw
        # verification tokens. The old artifact did not serialize that count.
        scheduled = int(performance["scheduled_draft_token_count"])
        maximum_scheduled_length = scheduled // verification_count
        accepted_lower += 1
        accepted_upper += maximum_scheduled_length
    return {
        "available": False,
        "source": "unavailable_legacy_scheduler_placeholder_comparison",
        "accepted": None,
        "accepted_lower_bound": accepted_lower,
        "accepted_upper_bound": accepted_upper,
        "replacement": None,
        "bonus": None,
        "verification_committed": verification_committed,
    }


def _aggregate_case_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output_tokens = sum(int(record["output_token_count"]) for record in records)
    decode_tokens = sum(int(record["performance"]["decode_output_token_count"]) for record in records)
    prefill_seconds = sum(float(record["performance"]["prefill_latency_seconds"]) for record in records)
    decode_seconds = sum(float(record["performance"]["decode_latency_seconds"]) for record in records)
    inference_seconds = sum(float(record["performance"]["inference_latency_seconds"]) for record in records)
    verification_count = sum(int(record["verification_count"]) for record in records)
    scheduled_tokens = sum(int(record["performance"]["scheduled_draft_token_count"]) for record in records)
    acceptance = [_case_acceptance_metrics(record) for record in records]
    acceptance_available = all(item["available"] for item in acceptance)
    accepted_tokens = sum(int(item["accepted"]) for item in acceptance) if acceptance_available else None
    replacement_tokens = sum(int(item["replacement"]) for item in acceptance) if acceptance_available else None
    bonus_tokens = sum(int(item["bonus"]) for item in acceptance) if acceptance_available else None
    verification_committed = sum(int(item["verification_committed"]) for item in acceptance)
    accepted_lower = sum(int(item["accepted_lower_bound"]) for item in acceptance)
    accepted_upper = sum(int(item["accepted_upper_bound"]) for item in acceptance)
    return {
        "case_count": len(records),
        "output_token_count": output_tokens,
        "decode_output_token_count": decode_tokens,
        "prefill_latency_seconds": prefill_seconds,
        "decode_latency_seconds": decode_seconds,
        "inference_latency_seconds": inference_seconds,
        "milliseconds_per_output_token": 1000.0 * inference_seconds / output_tokens,
        "output_tokens_per_second": output_tokens / inference_seconds,
        "decode_milliseconds_per_output_token": 1000.0 * decode_seconds / decode_tokens if decode_tokens else 0.0,
        "decode_output_tokens_per_second": decode_tokens / decode_seconds if decode_tokens and decode_seconds else 0.0,
        "target_forward_count": sum(int(record["target_forward_count"]) for record in records),
        "verification_count": verification_count,
        "proposal_generated_count": sum(int(record["proposal_generated_count"]) for record in records),
        "proposal_installed_count": sum(int(record["proposal_installed_count"]) for record in records),
        "proposal_consumed_count": sum(int(record["proposal_consumed_count"]) for record in records),
        "scheduled_draft_token_count": scheduled_tokens,
        "accepted_candidate_metrics_available": acceptance_available,
        "accepted_candidate_metrics_source": (
            ACCEPTED_METRICS_SOURCE
            if acceptance_available
            else (
                "not_applicable_without_verification"
                if verification_count == 0
                else "unavailable_legacy_scheduler_placeholder_comparison"
            )
        ),
        "accepted_candidate_tokens_total": accepted_tokens,
        "accepted_candidate_tokens_lower_bound": accepted_lower,
        "accepted_candidate_tokens_upper_bound": accepted_upper,
        "average_accepted_candidate_tokens_per_verification": (
            accepted_tokens / verification_count if accepted_tokens is not None and verification_count else None
        ),
        "replacement_tokens_total": replacement_tokens,
        "bonus_tokens_total": bonus_tokens,
        "committed_tokens_total": output_tokens,
        "verification_committed_tokens_total": verification_committed,
        "effective_committed_tokens_per_verification": (
            verification_committed / verification_count if verification_count else None
        ),
        # Deprecated aliases remain null when the source artifact is not
        # trustworthy rather than preserving a misleading numeric zero.
        "accepted_draft_token_count": accepted_tokens,
        "average_accepted_tokens_per_verification": (
            accepted_tokens / verification_count if accepted_tokens is not None and verification_count else None
        ),
        "draft_token_acceptance_rate": (
            accepted_tokens / scheduled_tokens if accepted_tokens is not None and scheduled_tokens else None
        ),
        "scheduler_seconds": sum(float(record["performance"]["scheduler_seconds"]) for record in records),
        "scheduler_update_seconds": sum(float(record["performance"]["scheduler_update_seconds"]) for record in records),
        "model_execute_host_seconds": sum(
            float(record["performance"]["model_execute_host_seconds"]) for record in records
        ),
        "sample_materialize_seconds": sum(
            float(record["performance"]["sample_materialize_seconds"]) for record in records
        ),
        "draft_install_seconds": sum(float(record["performance"]["draft_install_seconds"]) for record in records),
        "spec_decode_proposer_latency_seconds": sum(
            float(record["performance"]["spec_decode_proposer_latency_seconds"]) for record in records
        ),
        "spec_decode_verification_latency_seconds": sum(
            float(record["performance"]["spec_decode_verification_latency_seconds"]) for record in records
        ),
    }


def _load_run(
    mode: str,
    root: Path,
    manifest_hash: str,
    expected_ranks: int,
) -> dict[str, Any]:
    rank_records = load_results(
        root,
        mode=mode,
        manifest_hash=manifest_hash,
        expected_ranks=expected_ranks,
    )
    for records in rank_records.values():
        for record in records:
            _validate_performance_record(record, mode)
    critical = _critical_case_records(rank_records)
    rank_summaries = [_rank_summary(root, rank) for rank in range(expected_ranks)]
    if any(summary["mode"] != mode for summary in rank_summaries):
        raise ValueError(f"Rank summary mode differs from --run mode for {root}.")

    aggregate = _aggregate_case_records(critical)
    warmup_excluded = _aggregate_case_records(critical[1:])
    first_case_seconds = float(critical[0]["performance"]["inference_latency_seconds"])
    later_case_seconds = [float(record["performance"]["inference_latency_seconds"]) for record in critical[1:]]
    later_median = statistics.median(later_case_seconds) if later_case_seconds else first_case_seconds

    return {
        "mode": mode,
        "root": str(root),
        "profile": critical[0]["profile"],
        **aggregate,
        "model_load_seconds": max(float(summary["phase_timings"]["model_load_seconds"]) for summary in rank_summaries),
        "kv_cache_init_seconds": max(
            float(summary["phase_timings"]["kv_cache_init_seconds"]) for summary in rank_summaries
        ),
        "peak_npu_allocated_bytes": max(
            int(record["performance"]["npu_memory"]["peak_allocated"]) for record in critical
        ),
        "peak_npu_reserved_bytes": max(
            int(record["performance"]["npu_memory"]["peak_reserved"]) for record in critical
        ),
        "first_case_inference_seconds": first_case_seconds,
        "later_case_median_inference_seconds": later_median,
        "first_case_warmup_ratio": first_case_seconds / later_median if later_median else 0.0,
        "first_case_warmup_ratio_workload_matched": False,
        "warmup_case_id": critical[0]["case_id"],
        "warmup_excluded": warmup_excluded,
        "records": critical,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile over no samples.")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _statistics(values: Sequence[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    standard_deviation = statistics.pstdev(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "median": statistics.median(values),
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / mean if mean else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
    }


def _nullable_statistics(values: Sequence[float | None]) -> dict[str, float] | None:
    if any(value is None for value in values):
        return None
    return _statistics([float(value) for value in values if value is not None])


def _first_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _cross_mode_diagnostics(target: Mapping[str, Any], dspark: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    target_records = target["records"]
    dspark_records = dspark["records"]
    if len(target_records) != len(dspark_records):
        raise ValueError("Target-only and DSpark runs contain different case counts.")
    for target_record, dspark_record in zip(target_records, dspark_records):
        for field in (
            "case_id",
            "dataset",
            "profile",
            "request_sequence_index",
            "prompt_token_count",
            "prompt_token_sha256",
            "output_cap",
            "ignore_eos",
            "output_token_count",
            "stop_reason",
        ):
            if target_record[field] != dspark_record[field]:
                raise ValueError(f"Performance comparability mismatch for {target_record['case_id']} field {field}.")
        target_ids = target_record["output_token_ids"]
        dspark_ids = dspark_record["output_token_ids"]
        diagnostics.append(
            {
                "case_id": target_record["case_id"],
                "exact_token_match": target_ids == dspark_ids,
                "first_different_token_index": _first_difference(target_ids, dspark_ids),
                "target_output_token_sha256": target_record["output_token_sha256"],
                "dspark_output_token_sha256": dspark_record["output_token_sha256"],
            }
        )
    return diagnostics


def _matched_case_performance(target: Mapping[str, Any], dspark: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_record, dspark_record in zip(target["records"], dspark["records"]):
        target_performance = target_record["performance"]
        dspark_performance = dspark_record["performance"]
        acceptance = _case_acceptance_metrics(dspark_record)
        verification_count = int(dspark_record["verification_count"])
        rows.append(
            {
                "case_id": target_record["case_id"],
                "request_sequence_index": target_record["request_sequence_index"],
                "prompt_token_count": target_record["prompt_token_count"],
                "output_token_count": target_record["output_token_count"],
                "warmup": target_record["request_sequence_index"] == 0,
                "target_prefill_seconds": target_performance["prefill_latency_seconds"],
                "target_decode_seconds": target_performance["decode_latency_seconds"],
                "target_inference_seconds": target_performance["inference_latency_seconds"],
                "dspark_prefill_seconds": dspark_performance["prefill_latency_seconds"],
                "dspark_decode_seconds": dspark_performance["decode_latency_seconds"],
                "dspark_inference_seconds": dspark_performance["inference_latency_seconds"],
                "target_decode_milliseconds_per_output_token": target_performance[
                    "decode_milliseconds_per_output_token"
                ],
                "dspark_decode_milliseconds_per_output_token": dspark_performance[
                    "decode_milliseconds_per_output_token"
                ],
                "target_decode_output_tokens_per_second": target_performance["decode_output_tokens_per_second"],
                "dspark_decode_output_tokens_per_second": dspark_performance["decode_output_tokens_per_second"],
                "decode_speedup": (
                    target_performance["decode_latency_seconds"] / dspark_performance["decode_latency_seconds"]
                ),
                "target_target_forward_count": target_record["target_forward_count"],
                "dspark_target_forward_count": dspark_record["target_forward_count"],
                "verification_count": verification_count,
                "proposal_generated_count": dspark_record["proposal_generated_count"],
                "proposal_installed_count": dspark_record["proposal_installed_count"],
                "proposal_consumed_count": dspark_record["proposal_consumed_count"],
                "accepted_candidate_tokens_total": acceptance["accepted"],
                "accepted_candidate_tokens_lower_bound": acceptance["accepted_lower_bound"],
                "accepted_candidate_tokens_upper_bound": acceptance["accepted_upper_bound"],
                "average_accepted_candidate_tokens_per_verification": (
                    acceptance["accepted"] / verification_count
                    if acceptance["accepted"] is not None and verification_count
                    else None
                ),
                "effective_committed_tokens_per_verification": (
                    acceptance["verification_committed"] / verification_count if verification_count else None
                ),
                "stop_reason": target_record["stop_reason"],
                "tp8_consistent": True,
            }
        )
    return rows


def _scan_error_logs(roots: Sequence[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Error-log root does not exist: {root}.")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise ValueError(f"Unable to scan log file: {path}.") from exc
            matches = [marker for marker in HISTORICAL_ERROR_MARKERS if marker in text]
            if matches:
                found[str(path)] = matches
    return found


def summarize_performance(
    manifest_path: Path,
    runs: Sequence[tuple[str, Path]],
    *,
    expected_ranks: int,
    min_runs_per_mode: int,
    error_log_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = verify_asset_bundle(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    if len({root for _, root in runs}) != len(runs):
        raise ValueError("Every performance run must use an independent result directory.")
    loaded = [_load_run(mode, root, manifest_hash, expected_ranks) for mode, root in runs]
    for run in loaded:
        expected_case_ids = manifest["profiles"][run["profile"]]
        actual_case_ids = [record["case_id"] for record in run["records"]]
        if actual_case_ids != expected_case_ids:
            raise ValueError(f"Performance run {run['root']} does not match the frozen profile order.")
    by_mode = {mode: [run for run in loaded if run["mode"] == mode] for mode in MODES}
    for mode, mode_runs in by_mode.items():
        if len(mode_runs) < min_runs_per_mode:
            raise ValueError(f"Expected at least {min_runs_per_mode} independent {mode} runs, got {len(mode_runs)}.")
        expected_case_contract = [
            (
                record["case_id"],
                record["prompt_token_sha256"],
                record["output_token_count"],
                record["stop_reason"],
            )
            for record in mode_runs[0]["records"]
        ]
        for run in mode_runs[1:]:
            actual = [
                (
                    record["case_id"],
                    record["prompt_token_sha256"],
                    record["output_token_count"],
                    record["stop_reason"],
                )
                for record in run["records"]
            ]
            if actual != expected_case_contract:
                raise ValueError(f"Independent {mode} runs changed case/prompt/count/stop contracts.")

    log_errors = _scan_error_logs(error_log_roots)
    if log_errors:
        raise ValueError(f"Historical runtime error signatures were found: {log_errors}.")
    if len(by_mode["target_only"]) != len(by_mode["dspark"]):
        raise ValueError("Target-only and DSpark performance run counts must match.")

    metric_names = (
        "prefill_latency_seconds",
        "decode_latency_seconds",
        "inference_latency_seconds",
        "milliseconds_per_output_token",
        "output_tokens_per_second",
        "decode_milliseconds_per_output_token",
        "decode_output_tokens_per_second",
        "target_forward_count",
        "verification_count",
        "scheduler_seconds",
        "scheduler_update_seconds",
        "model_execute_host_seconds",
        "sample_materialize_seconds",
        "draft_install_seconds",
        "spec_decode_proposer_latency_seconds",
        "spec_decode_verification_latency_seconds",
        "model_load_seconds",
        "kv_cache_init_seconds",
        "peak_npu_allocated_bytes",
        "peak_npu_reserved_bytes",
        "first_case_warmup_ratio",
    )
    aggregate = {
        mode: {metric: _statistics([float(run[metric]) for run in mode_runs]) for metric in metric_names}
        for mode, mode_runs in by_mode.items()
    }
    nullable_metric_names = (
        "average_accepted_candidate_tokens_per_verification",
        "effective_committed_tokens_per_verification",
    )
    for mode, mode_runs in by_mode.items():
        aggregate[mode].update(
            {metric: _nullable_statistics([run[metric] for run in mode_runs]) for metric in nullable_metric_names}
        )
    paired_diagnostics = [
        {
            "pair_index": index,
            "target_root": target["root"],
            "dspark_root": dspark["root"],
            "cases": _cross_mode_diagnostics(target, dspark),
        }
        for index, (target, dspark) in enumerate(zip(by_mode["target_only"], by_mode["dspark"]))
    ]
    matched_case_performance = [
        {
            "pair_index": index,
            "target_root": target["root"],
            "dspark_root": dspark["root"],
            "cases": _matched_case_performance(target, dspark),
        }
        for index, (target, dspark) in enumerate(zip(by_mode["target_only"], by_mode["dspark"]))
    ]
    target_decode = aggregate["target_only"]["decode_latency_seconds"]["median"]
    dspark_decode = aggregate["dspark"]["decode_latency_seconds"]["median"]
    target_inference = aggregate["target_only"]["inference_latency_seconds"]["median"]
    dspark_inference = aggregate["dspark"]["inference_latency_seconds"]["median"]
    target_warmup_excluded_decode = statistics.median(
        run["warmup_excluded"]["decode_latency_seconds"] for run in by_mode["target_only"]
    )
    dspark_warmup_excluded_decode = statistics.median(
        run["warmup_excluded"]["decode_latency_seconds"] for run in by_mode["dspark"]
    )
    target_warmup_excluded_inference = statistics.median(
        run["warmup_excluded"]["inference_latency_seconds"] for run in by_mode["target_only"]
    )
    dspark_warmup_excluded_inference = statistics.median(
        run["warmup_excluded"]["inference_latency_seconds"] for run in by_mode["dspark"]
    )
    if (
        dspark_decode <= 0
        or dspark_inference <= 0
        or dspark_warmup_excluded_decode <= 0
        or dspark_warmup_excluded_inference <= 0
    ):
        raise ValueError("DSpark performance timings must be positive before calculating speedup.")
    return {
        "manifest_sha256": manifest_hash,
        "expected_ranks": expected_ranks,
        "performance_provisional": True,
        "exact_token_cross_mode_blocking": False,
        "run_aggregation": (
            "single-run aggregate" if len(by_mode["target_only"]) == 1 else "median of independent run aggregates"
        ),
        "runs": [{key: value for key, value in run.items() if key != "records"} for run in loaded],
        "statistics": aggregate,
        "speedup": {
            "all_case_decode": target_decode / dspark_decode,
            "all_case_inference": target_inference / dspark_inference,
            "primary_warmup_excluded_decode": target_warmup_excluded_decode / dspark_warmup_excluded_decode,
            "warmup_excluded_inference": target_warmup_excluded_inference / dspark_warmup_excluded_inference,
        },
        "cross_mode_exact_token_diagnostics": paired_diagnostics,
        "matched_case_performance": matched_case_performance,
        "timer_relationships": TIMER_RELATIONSHIPS,
        "historical_error_scan": {"roots": [str(path.resolve()) for path in error_log_roots], "matches": {}},
    }


def _write_csv(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "mode",
        "root",
        "profile",
        "case_count",
        "output_token_count",
        "prefill_latency_seconds",
        "decode_latency_seconds",
        "inference_latency_seconds",
        "milliseconds_per_output_token",
        "output_tokens_per_second",
        "decode_milliseconds_per_output_token",
        "decode_output_tokens_per_second",
        "target_forward_count",
        "verification_count",
        "accepted_candidate_metrics_available",
        "accepted_candidate_tokens_total",
        "accepted_candidate_tokens_lower_bound",
        "accepted_candidate_tokens_upper_bound",
        "average_accepted_candidate_tokens_per_verification",
        "replacement_tokens_total",
        "bonus_tokens_total",
        "effective_committed_tokens_per_verification",
        "proposal_generated_count",
        "proposal_installed_count",
        "proposal_consumed_count",
        "scheduler_seconds",
        "scheduler_update_seconds",
        "model_execute_host_seconds",
        "sample_materialize_seconds",
        "draft_install_seconds",
        "spec_decode_proposer_latency_seconds",
        "spec_decode_verification_latency_seconds",
        "model_load_seconds",
        "kv_cache_init_seconds",
        "peak_npu_allocated_bytes",
        "peak_npu_reserved_bytes",
        "first_case_warmup_ratio",
        "warmup_excluded_case_count",
        "warmup_excluded_output_token_count",
        "warmup_excluded_prefill_latency_seconds",
        "warmup_excluded_decode_latency_seconds",
        "warmup_excluded_inference_latency_seconds",
        "warmup_excluded_decode_output_tokens_per_second",
        "warmup_excluded_decode_milliseconds_per_output_token",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = dict(run)
            row.update({f"warmup_excluded_{key}": value for key, value in run["warmup_excluded"].items()})
            writer.writerow({field: row[field] for field in fields})


CASE_CSV_FIELDS = (
    "pair_index",
    "case_id",
    "request_sequence_index",
    "prompt_token_count",
    "output_token_count",
    "warmup",
    "target_prefill_seconds",
    "target_decode_seconds",
    "target_inference_seconds",
    "dspark_prefill_seconds",
    "dspark_decode_seconds",
    "dspark_inference_seconds",
    "target_decode_milliseconds_per_output_token",
    "dspark_decode_milliseconds_per_output_token",
    "target_decode_output_tokens_per_second",
    "dspark_decode_output_tokens_per_second",
    "decode_speedup",
    "target_target_forward_count",
    "dspark_target_forward_count",
    "verification_count",
    "proposal_generated_count",
    "proposal_installed_count",
    "proposal_consumed_count",
    "accepted_candidate_tokens_total",
    "accepted_candidate_tokens_lower_bound",
    "accepted_candidate_tokens_upper_bound",
    "average_accepted_candidate_tokens_per_verification",
    "effective_committed_tokens_per_verification",
    "stop_reason",
    "tp8_consistent",
)


def _flatten_matched_cases(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"pair_index": pair["pair_index"], **case} for pair in pairs for case in pair["cases"]]


def _write_case_csv(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CASE_CSV_FIELDS)
        writer.writeheader()
        for row in _flatten_matched_cases(pairs):
            serialized = dict(row)
            serialized["stop_reason"] = json.dumps(row["stop_reason"], sort_keys=True)
            writer.writerow({field: serialized[field] for field in CASE_CSV_FIELDS})


def _write_case_markdown(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        ("case_id", "case"),
        ("prompt_token_count", "prompt"),
        ("output_token_count", "output"),
        ("warmup", "warmup"),
        ("target_decode_seconds", "target decode s"),
        ("dspark_decode_seconds", "DSpark decode s"),
        ("decode_speedup", "speedup"),
        ("verification_count", "verification"),
        ("average_accepted_candidate_tokens_per_verification", "accepted/ver"),
        ("effective_committed_tokens_per_verification", "committed/ver"),
        ("tp8_consistent", "TP8"),
    )
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in _flatten_matched_cases(pairs):
        values = []
        for field, _label in columns:
            value = row[field]
            if isinstance(value, float):
                value = f"{value:.6f}"
            elif value is None:
                value = "UNAVAILABLE"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    runs = [_parse_run(value) for value in args.run]
    summary = summarize_performance(
        args.manifest,
        runs,
        expected_ranks=args.expected_ranks,
        min_runs_per_mode=args.min_runs_per_mode,
        error_log_roots=args.error_log_root,
    )
    output_json = args.output_json.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_json, summary)
    _write_csv(output_csv, summary["runs"])
    if args.output_case_csv is not None:
        _write_case_csv(args.output_case_csv.expanduser().resolve(), summary["matched_case_performance"])
    if args.output_case_markdown is not None:
        _write_case_markdown(
            args.output_case_markdown.expanduser().resolve(),
            summary["matched_case_performance"],
        )
    marker = {
        "runs": len(summary["runs"]),
        "speedup": summary["speedup"],
        "output_json": str(args.output_json.expanduser().resolve()),
        "output_csv": str(args.output_csv.expanduser().resolve()),
        "output_case_csv": (str(args.output_case_csv.expanduser().resolve()) if args.output_case_csv else None),
        "output_case_markdown": (
            str(args.output_case_markdown.expanduser().resolve()) if args.output_case_markdown else None
        ),
        "performance_provisional": True,
        "exact_token_cross_mode_blocking": False,
    }
    print("M2_5A_PERFORMANCE_SUMMARY_PASS=" + json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
