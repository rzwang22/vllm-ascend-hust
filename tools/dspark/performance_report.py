# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request distributions and independent-process statistics are separate populations."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark.performance_stream import request_latency


def run_statistics(values):
    values = list(values)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite statistic")
    mean = statistics.fmean(values) if values else None
    std = statistics.stdev(values) if len(values) >= 2 else None
    return {
        "n": len(values),
        "raw": values,
        "mean": mean,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "sample_standard_deviation": std,
        "sample_cv": std / mean if std is not None and mean else None,
    }


def request_distribution(values):
    values = sorted(value for value in values if value is not None)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Invalid request latency")

    def percentile(fraction):
        if not values:
            return None
        index = (len(values) - 1) * fraction
        low = math.floor(index)
        return values[low] + (values[math.ceil(index)] - values[low]) * (index - low)

    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "quantile_method": "linear interpolation at (n-1)*p",
        "p99_sample_note": "fewer than 100 observations; sparse tail estimate" if len(values) < 100 else None,
    }


def latency_summary(result):
    stream = result.get("streaming")
    if not stream:
        return {"status": "unavailable", "reason": "legacy result has no observed stream timestamps"}
    requests = stream["requests"]
    return {
        "status": "available",
        **{
            name: request_distribution(row[name] for row in requests)
            for name in ("ttft_seconds", "completion_seconds", "mean_tpot_seconds")
        },
        "event_intervals_seconds": request_distribution(
            value for row in requests for value in row["event_intervals_seconds"]
        ),
    }


def validate_stream_result(result, records, *, specified_verification_test=False):
    from tools.dspark.summarize_dspark_acceptance_benchmark import _validate_result

    _validate_result(result, result["mode"], specified_verification_test=specified_verification_test)
    if (
        result.get("measurement_protocol") != "async_llm_delta_stream_v1"
        or result.get("performance_schema_version") != 1
    ):
        raise ValueError("Performance suite requires the same streaming protocol")
    stream = result["streaming"]
    if stream["error"] or len(stream["requests"]) != len(records) or len(result["outputs"]) != len(records):
        raise ValueError("Incomplete/failed stream")
    elapsed = stream["finished_monotonic"] - stream["started_monotonic"]
    if elapsed <= 0 or not math.isclose(elapsed, result["timing"]["elapsed_seconds"]):
        raise ValueError("Inconsistent measured interval")
    request_ids = set()
    for i, (request, output, task) in enumerate(zip(stream["requests"], result["outputs"], records)):
        if request is None or request["error"] or request["request_id"] in request_ids:
            raise ValueError("Missing/duplicate/failed request")
        request_ids.add(request["request_id"])
        if request["request_index"] != i or request["finish_reason"] not in ("stop", "length"):
            raise ValueError("Invalid completion identity/reason")
        if (
            not stream["started_monotonic"]
            <= request["submitted_monotonic"]
            <= request["completed_monotonic"]
            <= stream["finished_monotonic"]
        ):
            raise ValueError("Request outside measurement interval")
        previous = request["submitted_monotonic"]
        for event in request["events"]:
            if (
                not previous <= event["monotonic"] <= request["completed_monotonic"]
                or type(event["new_tokens"]) is not int
                or event["new_tokens"] <= 0
            ):
                raise ValueError("Invalid stream event")
            previous = event["monotonic"]
        first = request["events"][0]["monotonic"] if request["events"] else None
        if first != request["first_output_monotonic"]:
            raise ValueError("First output timestamp was not observed")
        for key, value in request_latency(request).items():
            if request[key] != value:
                raise ValueError("Fabricated/inconsistent request latency")
        tokens = request["output_token_ids"]
        if (
            any(type(token) is not int or token < 0 for token in tokens)
            or len(tokens) != output["output_token_count"]
            or len(tokens) > result["sampling"]["output_len"]
            or benchmark._sha256_bytes(benchmark._canonical_json_bytes(tokens)) != output["output_token_sha256"]
            or output["prompt_token_sha256"] != task["prompt_token_sha256"]
            or not isinstance(request["text"], str)
        ):
            raise ValueError("Output artifact corrupt or wrong input")
    if result["target_execution_mode_effective"] == "full_decode_only":
        args = SimpleNamespace(tensor_parallel_size=result["effective_config"]["tensor_parallel_size"])
        snapshots = result["graph_execution"]["boundary_snapshots"]
        recomputed = benchmark._replay_interval(args, snapshots[1], snapshots[2])
        if recomputed != result["graph_execution"]["measured_runtime"] or recomputed["graph_replay_count"] <= 0:
            raise ValueError("Measured replay does not match per-rank execution evidence")
    return latency_summary(result)


def summarize_suite(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    if not plan["runs"]:
        raise ValueError("No independent runs in plan")
    runs, groups, pairs = [], defaultdict(list), []
    run_ids = set()
    for case in plan["runs"]:
        directory = root / case["directory"]
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {"status": "not_run"}
        row = {**case, "status": receipt["status"], "receipt": receipt}
        if receipt["status"] == "valid":
            result = json.loads((directory / "result.json").read_text())
            if not result.get("run_id") or result["run_id"] in run_ids:
                raise ValueError("Duplicate independent run identity")
            run_ids.add(result["run_id"])
            # Revalidate saved bytes as well as the original run's gates.
            if benchmark._sha256_file(directory / "result.json") != receipt["result_sha256"]:
                raise ValueError("Result changed after validation")
            if benchmark._sha256_file(directory / "generation.log") != receipt["log_sha256"]:
                raise ValueError("Log changed after validation")
            for relative, expected in receipt.get("artifact_sha256", {}).items():
                path = (directory / relative).resolve()
                if directory.resolve() not in path.parents or benchmark._sha256_file(path) != expected:
                    raise ValueError("Output/quality artifact changed after validation")
            row.update(
                {
                    "throughput": result["throughput"],
                    "elapsed_seconds": result["timing"]["elapsed_seconds"],
                    "latency": latency_summary(result),
                    "acceptance": result["acceptance"],
                    "scheduler": result["streaming"]["scheduler"],
                    "measured_graph_replay_count": result["measured_graph_replay_count"],
                    "graph": result["graph_execution"],
                    "confidence_verification": result.get("confidence_verification"),
                    "quality": receipt["quality"],
                    "output_lengths": [output["output_token_count"] for output in result["outputs"]],
                    "finish_reasons": [output["finish_reason"] for output in result["outputs"]],
                    "output_hashes": [output["output_token_sha256"] for output in result["outputs"]],
                }
            )
            groups[(case["max_num_seqs"], case["mode"])].append(row)
        runs.append(row)
    aggregated = []
    for (concurrency, mode), rows in sorted(groups.items()):
        aggregated.append(
            {
                "max_num_seqs": concurrency,
                "mode": mode,
                **{
                    key: run_statistics(row["throughput"][key] for row in rows)
                    for key in ("output_tokens_per_second", "requests_per_second", "total_output_tokens")
                },
                "elapsed_seconds": run_statistics(row["elapsed_seconds"] for row in rows),
                "request_distribution_statistics": {
                    metric: {
                        stat: run_statistics(
                            row["latency"][metric][stat] for row in rows if row["latency"][metric][stat] is not None
                        )
                        for stat in ("mean", "p50", "p95", "p99")
                    }
                    for metric in ("ttft_seconds", "completion_seconds", "mean_tpot_seconds")
                },
                "acceptance_statistics": {
                    metric: run_statistics(
                        row["acceptance"][metric] for row in rows if row["acceptance"][metric] is not None
                    )
                    for metric in (
                        "accepted_candidate_tokens_per_verification",
                        "effective_acceptance_length",
                        "num_drafts",
                    )
                },
            }
        )
    for concurrency in plan["max_num_seqs"]:
        for baseline_mode, candidate in (
            ("target_graph", "dspark_graph"),
            ("target_graph", "dspark_confidence_graph"),
            ("dspark_graph", "dspark_confidence_graph"),
            ("target_eager", "dspark_eager"),
            ("target_eager", "target_graph"),
            ("dspark_eager", "dspark_graph"),
        ):
            if candidate not in plan["modes"] or baseline_mode not in plan["modes"]:
                continue
            baseline_rows = groups.get((concurrency, baseline_mode), [])
            candidate_rows = groups.get((concurrency, candidate), [])
            by_repeat = {row["repeat"]: row for row in baseline_rows}
            paired = []
            for row in candidate_rows:
                baseline = by_repeat.get(row["repeat"])
                if baseline is not None:
                    bq, cq = baseline["quality"], row["quality"]
                    paired.append(
                        {
                            "repeat": row["repeat"],
                            "baseline_directory": baseline["directory"],
                            "candidate_directory": row["directory"],
                            "speedup": row["throughput"]["output_tokens_per_second"]
                            / baseline["throughput"]["output_tokens_per_second"],
                            "output_token_counts_equal": row["output_lengths"] == baseline["output_lengths"],
                            "output_tokens_equal": row["output_hashes"] == baseline["output_hashes"],
                            "quality_pass_fraction_difference": cq["all_task_pass_fraction"]
                            - bq["all_task_pass_fraction"]
                            if cq.get("all_task_pass_fraction") is not None
                            and bq.get("all_task_pass_fraction") is not None
                            else None,
                        }
                    )
            complete = len(paired) == plan["repeats"]
            pairs.append(
                {
                    "max_num_seqs": concurrency,
                    "baseline": baseline_mode,
                    "candidate": candidate,
                    "pairs": paired,
                    "status": "complete" if complete else "incomplete",
                    "ratio_of_median_tok_s": statistics.median(
                        row["throughput"]["output_tokens_per_second"] for row in candidate_rows
                    )
                    / statistics.median(row["throughput"]["output_tokens_per_second"] for row in baseline_rows)
                    if complete
                    else None,
                }
            )
    return {
        "schema_version": 1,
        "benchmark": "dspark_additional_performance",
        "status": "valid" if all(row["status"] == "valid" for row in runs) else "incomplete",
        "protocol": "async_llm_delta_stream_v1",
        "runs": runs,
        "independent_run_statistics": aggregated,
        "comparisons": pairs,
        "speedup_definition": (
            "candidate median actual output tok/s / baseline median actual output tok/s; also paired by repeat. "
            "Primary comparison: dspark_graph / target_graph"
        ),
        "quality_note": "unavailable is not a quality pass; no borrowed quality thresholds",
        "latency_note": (
            "TTFT includes engine queue, excludes client admission wait; "
            "mean TPOT is a request average, not individual token times"
        ),
        "coverage_note": "FULL-only observer; eager fallback unavailable; TP replay counts are not summed",
        "plan": plan,
    }


def write_reports(summary, json_path, csv_path, markdown_path):
    benchmark._atomic_write_json(Path(json_path), summary)
    with Path(csv_path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "directory",
                "mode",
                "repeat",
                "max_num_seqs",
                "status",
                "output_tokens",
                "elapsed",
                "tok_s",
                "requests_s",
                "ttft_mean",
                "ttft_p50",
                "ttft_p95",
                "ttft_p99",
                "completion_mean",
                "completion_p50",
                "completion_p95",
                "completion_p99",
                "tpot_mean",
                "tpot_p50",
                "tpot_p95",
                "tpot_p99",
                "full_replays",
                "accepted_candidate_length",
                "effective_advancement",
                "quality_status",
            ]
        )
        for row in summary["runs"]:
            throughput = row.get("throughput", {})
            latency = row.get("latency", {})
            writer.writerow(
                [
                    row["directory"],
                    row["mode"],
                    row["repeat"],
                    row["max_num_seqs"],
                    row["status"],
                    throughput.get("total_output_tokens"),
                    row.get("elapsed_seconds"),
                    throughput.get("output_tokens_per_second"),
                    throughput.get("requests_per_second"),
                    *[
                        latency.get(metric, {}).get(stat)
                        for metric in ("ttft_seconds", "completion_seconds", "mean_tpot_seconds")
                        for stat in ("mean", "p50", "p95", "p99")
                    ],
                    row.get("measured_graph_replay_count"),
                    row.get("acceptance", {}).get("accepted_candidate_tokens_per_verification"),
                    row.get("acceptance", {}).get("effective_acceptance_length"),
                    row.get("quality", {}).get("status"),
                ]
            )
    lines = [
        "# DSpark additional performance",
        "",
        f"Status: {summary['status']}. Protocol: {summary['protocol']}.",
        "",
        "| Mode | Concurrency cap | n | Median tok/s | Sample CV |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in summary["independent_run_statistics"]:
        stats = group["output_tokens_per_second"]
        lines.append(
            f"| {group['mode']} | {group['max_num_seqs']} | {stats['n']} | {stats['median']} | {stats['sample_cv']} |"
        )
    lines += [
        "",
        summary["speedup_definition"],
        "",
        summary["latency_note"],
        "",
        summary["coverage_note"],
        "",
        "Quality and failures are retained per run in JSON; unavailable is not a pass. "
        "Diagnostics are excluded from performance.",
        "",
    ]
    Path(markdown_path).write_text("\n".join(lines))
