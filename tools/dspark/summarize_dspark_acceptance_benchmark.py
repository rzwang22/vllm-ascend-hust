#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Summarize independent PR-style target-only and DSpark benchmark runs."""

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

from tools.dspark import benchmark_dspark_acceptance as benchmark


def _read_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read benchmark result {path}.") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Benchmark result {path} must be a JSON object.")
    value["_source_path"] = str(path)
    return value


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive, got {value!r}.")
    return result


def _validate_result(result: Mapping[str, Any], expected_mode: str) -> None:
    if result.get("schema_version") != benchmark.SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark result schema.")
    if result.get("benchmark") != "dspark_pr_style_batch_throughput":
        raise ValueError("JSON is not a DSpark PR-style throughput result.")
    if result.get("runner") != benchmark.RUNNER:
        raise ValueError("Only MRV2 benchmark results are accepted.")
    if result.get("mode") != expected_mode:
        raise ValueError(f"Expected mode {expected_mode!r}, got {result.get('mode')!r}.")
    if result.get("cleanup", {}).get("engine_shutdown_complete") is not True:
        raise ValueError("Benchmark result does not prove engine cleanup.")
    timing = result.get("timing")
    if not isinstance(timing, Mapping) or any(
        timing.get(field) is not False
        for field in (
            "model_load_included",
            "dataset_load_included",
            "prompt_tokenization_included",
            "warmup_included",
        )
    ):
        raise ValueError("Measured interval includes setup or warmup work.")
    elapsed = _finite_positive(timing.get("elapsed_seconds"), "elapsed_seconds")
    throughput = result.get("throughput")
    if not isinstance(throughput, Mapping):
        raise ValueError("Benchmark result has no throughput payload.")
    total_output = throughput.get("total_output_tokens")
    measured_requests = result.get("measured_request_count")
    if isinstance(total_output, bool) or not isinstance(total_output, int) or total_output <= 0:
        raise ValueError("total_output_tokens must be a positive integer.")
    if isinstance(measured_requests, bool) or not isinstance(measured_requests, int) or measured_requests <= 0:
        raise ValueError("measured_request_count must be a positive integer.")
    if not math.isclose(
        _finite_positive(throughput.get("output_tokens_per_second"), "output_tokens_per_second"),
        total_output / elapsed,
    ):
        raise ValueError("Output throughput is inconsistent with token count and elapsed time.")
    if not math.isclose(
        _finite_positive(throughput.get("requests_per_second"), "requests_per_second"),
        measured_requests / elapsed,
    ):
        raise ValueError("Request throughput is inconsistent with count and elapsed time.")
    outputs = result.get("outputs")
    prompts = result.get("prompt_identities")
    if not isinstance(outputs, list) or len(outputs) != measured_requests:
        raise ValueError("Measured output records are missing or incomplete.")
    if not isinstance(prompts, list) or len(prompts) != measured_requests:
        raise ValueError("Measured prompt identities are missing or incomplete.")
    prompt_set_sha256 = benchmark._sha256_bytes(
        benchmark._canonical_json_bytes(
            [
                {
                    "request_index": item.get("request_index"),
                    "prompt_token_sha256": item.get("prompt_token_sha256"),
                }
                for item in prompts
            ]
        )
    )
    if result.get("prompt_set_sha256") != prompt_set_sha256:
        raise ValueError("Measured prompt-set fingerprint is invalid.")
    if sum(item.get("output_token_count", -1) for item in outputs) != total_output:
        raise ValueError("Per-request output counts differ from the throughput total.")
    if [item.get("request_index") for item in outputs] != list(range(measured_requests)):
        raise ValueError("Measured output request indexes are not complete and ordered.")
    comparison = result.get("comparison_config")
    if not isinstance(comparison, Mapping):
        raise ValueError("Benchmark result has no comparison configuration.")
    fingerprint = benchmark._sha256_bytes(benchmark._canonical_json_bytes(comparison))
    if result.get("comparison_config_fingerprint") != fingerprint:
        raise ValueError("Comparison configuration fingerprint is invalid.")
    model = result.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Benchmark result has no model descriptor.")
    model_without_fingerprint = dict(model)
    saved_model_fingerprint = model_without_fingerprint.pop("fingerprint_sha256", None)
    expected_model_fingerprint = benchmark._sha256_bytes(benchmark._canonical_json_bytes(model_without_fingerprint))
    if saved_model_fingerprint != expected_model_fingerprint:
        raise ValueError("Model descriptor fingerprint is invalid.")

    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Benchmark result has no metric snapshots.")
    if metrics.get("delta_boundary") != "post_warmup_snapshot_to_post_measured_snapshot":
        raise ValueError("Speculative metrics do not exclude warmup by snapshot delta.")
    acceptance = result.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("Benchmark result has no acceptance payload.")
    if expected_mode == "dspark":
        k = result["effective_config"]["speculative_config"]["num_speculative_tokens"]
        recomputed_delta = benchmark.metric_snapshot_delta(
            metrics.get("warmup_snapshot", {}),
            metrics.get("measured_end_snapshot", {}),
            k,
        )
        if metrics.get("measured_delta") != recomputed_delta:
            raise ValueError("Saved measured metric delta differs from its snapshots.")
        recomputed_acceptance = benchmark.acceptance_from_delta(recomputed_delta, k)
        if acceptance != recomputed_acceptance:
            raise ValueError("Saved acceptance values differ from the measured metric delta.")
    else:
        if metrics.get("measured_delta") is not None:
            raise ValueError("target_only must not publish speculative metric deltas.")
        if any(
            acceptance.get(field) is not None
            for field in (
                "num_drafts",
                "num_draft_tokens",
                "num_accepted_candidate_tokens",
                "accepted_candidate_tokens_per_verification",
                "effective_acceptance_length",
            )
        ):
            raise ValueError("target_only speculative metrics must be not applicable.")


def _statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty run set.")
    mean = statistics.fmean(values)
    population_std = statistics.pstdev(values)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": mean,
        "minimum": min(values),
        "maximum": max(values),
        "population_standard_deviation": population_std,
        "population_coefficient_of_variation": population_std / mean if mean else 0.0,
    }


def _mode_statistics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    throughput = [float(result["throughput"]["output_tokens_per_second"]) for result in results]
    request_throughput = [float(result["throughput"]["requests_per_second"]) for result in results]
    elapsed = [float(result["timing"]["elapsed_seconds"]) for result in results]
    return {
        "output_tokens_per_second": _statistics(throughput),
        "requests_per_second": _statistics(request_throughput),
        "elapsed_seconds": _statistics(elapsed),
    }


def _functional_signature(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_index": output["request_index"],
            "prompt_token_sha256": output["prompt_token_sha256"],
            "output_token_count": output["output_token_count"],
            "finish_reason": output["finish_reason"],
            "stop_reason": output["stop_reason"],
        }
        for output in result["outputs"]
    ]


def _exact_token_diagnostic(
    target_results: Sequence[Mapping[str, Any]],
    dspark_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request_pairs = 0
    identical_pairs = 0
    mismatches: list[dict[str, Any]] = []
    for target_index, target in enumerate(target_results):
        for dspark_index, dspark in enumerate(dspark_results):
            for output_index, (target_output, dspark_output) in enumerate(zip(target["outputs"], dspark["outputs"])):
                request_pairs += 1
                if target_output["output_token_sha256"] == dspark_output["output_token_sha256"]:
                    identical_pairs += 1
                else:
                    mismatches.append(
                        {
                            "target_run_index": target_index,
                            "dspark_run_index": dspark_index,
                            "request_index": output_index,
                            "target_output_token_sha256": target_output["output_token_sha256"],
                            "dspark_output_token_sha256": dspark_output["output_token_sha256"],
                        }
                    )
    return {
        "blocking": False,
        "request_pair_count": request_pairs,
        "identical_request_pair_count": identical_pairs,
        "mismatched_request_pair_count": len(mismatches),
        "mismatches": mismatches,
    }


def summarize_results(
    target_results: Sequence[dict[str, Any]],
    dspark_results: Sequence[dict[str, Any]],
    *,
    min_runs_per_mode: int = 3,
) -> dict[str, Any]:
    if min_runs_per_mode <= 0:
        raise ValueError("min_runs_per_mode must be positive.")
    if len(target_results) < min_runs_per_mode or len(dspark_results) < min_runs_per_mode:
        raise ValueError(f"At least {min_runs_per_mode} independent results per mode are required.")
    for result in target_results:
        _validate_result(result, "target_only")
    for result in dspark_results:
        _validate_result(result, "dspark")
    all_results = [*target_results, *dspark_results]
    run_ids = [result.get("run_id") for result in all_results]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        raise ValueError("Every result requires a non-empty run_id.")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate benchmark run_id detected.")
    fingerprints = {result["comparison_config_fingerprint"] for result in all_results}
    if len(fingerprints) != 1:
        raise ValueError("Model, engine, sampling, dataset, or prompt configuration differs across runs.")
    if len({result.get("plugin_sha") for result in all_results}) != 1:
        raise ValueError("Plugin source identity differs across benchmark runs.")
    if len({result.get("core_sha") for result in all_results}) != 1:
        raise ValueError("Core source identity differs across benchmark runs.")
    signatures = [_functional_signature(result) for result in all_results]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Prompt identity, output count, or stop reason differs across benchmark runs.")

    target_stats = _mode_statistics(target_results)
    dspark_stats = _mode_statistics(dspark_results)
    target_median = target_stats["output_tokens_per_second"]["median"]
    dspark_median = dspark_stats["output_tokens_per_second"]["median"]
    k = len(dspark_results[0]["acceptance"]["acceptance_per_position"])
    acceptance_at_position = [
        statistics.median([result["acceptance"]["acceptance_per_position"][position] for result in dspark_results])
        for position in range(k)
    ]
    peak_values = [
        result.get("peak_npu_memory")
        for result in all_results
        if isinstance(result.get("peak_npu_memory"), (int, float))
    ]
    summary = {
        "schema_version": 1,
        "benchmark": "dspark_pr_style_batch_throughput_summary",
        "runner": benchmark.RUNNER,
        "comparison_config_fingerprint": fingerprints.pop(),
        "independent_process_runs": {
            "target_only": len(target_results),
            "dspark": len(dspark_results),
        },
        "target_only": target_stats,
        "dspark": dspark_stats,
        "dspark_over_target_output_throughput_speedup": dspark_median / target_median,
        "dspark_acceptance": {
            "acceptance_per_position_median": acceptance_at_position,
            "accepted_candidate_tokens_per_verification_median": statistics.median(
                [result["acceptance"]["accepted_candidate_tokens_per_verification"] for result in dspark_results]
            ),
            "effective_acceptance_length_median": statistics.median(
                [result["acceptance"]["effective_acceptance_length"] for result in dspark_results]
            ),
            "num_drafts_median": statistics.median([result["acceptance"]["num_drafts"] for result in dspark_results]),
            "num_draft_tokens_median": statistics.median(
                [result["acceptance"]["num_draft_tokens"] for result in dspark_results]
            ),
        },
        "peak_npu_memory_median": statistics.median(peak_values) if peak_values else None,
        "exact_token_comparison": _exact_token_diagnostic(target_results, dspark_results),
        "exact_token_gate": "nonblocking_for_performance",
        "formal_statistic": "median_of_independent_process_runs",
    }
    return summary


def _write_run_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "mode",
        "run_id",
        "source_path",
        "measured_requests",
        "total_output_tokens",
        "elapsed_seconds",
        "requests_per_second",
        "output_tokens_per_second",
        "num_drafts",
        "num_draft_tokens",
        "num_accepted_candidate_tokens",
        "accepted_candidate_tokens_per_verification",
        "effective_acceptance_length",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "mode": result["mode"],
                    "run_id": result["run_id"],
                    "source_path": result.get("_source_path"),
                    "measured_requests": result["measured_request_count"],
                    "total_output_tokens": result["throughput"]["total_output_tokens"],
                    "elapsed_seconds": result["timing"]["elapsed_seconds"],
                    "requests_per_second": result["throughput"]["requests_per_second"],
                    "output_tokens_per_second": result["throughput"]["output_tokens_per_second"],
                    "num_drafts": result["acceptance"]["num_drafts"],
                    "num_draft_tokens": result["acceptance"]["num_draft_tokens"],
                    "num_accepted_candidate_tokens": result["acceptance"]["num_accepted_candidate_tokens"],
                    "accepted_candidate_tokens_per_verification": result["acceptance"][
                        "accepted_candidate_tokens_per_verification"
                    ],
                    "effective_acceptance_length": result["acceptance"]["effective_acceptance_length"],
                }
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize independent DSpark PR-style throughput runs.")
    parser.add_argument("--target-result", action="append", type=Path, required=True)
    parser.add_argument("--dspark-result", action="append", type=Path, required=True)
    parser.add_argument("--min-runs-per-mode", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_results = [_read_result(path.expanduser().resolve()) for path in args.target_result]
    dspark_results = [_read_result(path.expanduser().resolve()) for path in args.dspark_result]
    summary = summarize_results(
        target_results,
        dspark_results,
        min_runs_per_mode=args.min_runs_per_mode,
    )
    benchmark._atomic_write_json(args.output_json.expanduser().resolve(), summary)
    _write_run_csv(args.output_csv.expanduser().resolve(), [*target_results, *dspark_results])
    print("-" * 60)
    print(f"runner: {summary['runner']}")
    print(
        "target-only output throughput median: "
        f"{summary['target_only']['output_tokens_per_second']['median']:.6f} tok/s"
    )
    print(f"DSpark output throughput median: {summary['dspark']['output_tokens_per_second']['median']:.6f} tok/s")
    print(f"DSpark/target-only speedup: {summary['dspark_over_target_output_throughput_speedup']:.6f}x")
    print(
        "mean accepted candidate length median: "
        f"{summary['dspark_acceptance']['accepted_candidate_tokens_per_verification_median']:.6f}"
    )
    print(
        f"effective acceptance length median: {summary['dspark_acceptance']['effective_acceptance_length_median']:.6f}"
    )
    print("-" * 60)
    print(f"DSPARK_PR_STYLE_SUMMARY_PASS={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
