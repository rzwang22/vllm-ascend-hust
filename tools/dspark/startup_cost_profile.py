# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One isolated engine per configuration, real disposable requests per profile point.

Unlike upstream dummy_run, normal scheduler admission owns all KV and terminal
cleanup. No model/process is created inside the point loop. This is a startup
calibration workload, never a performance or quality result.
"""

import argparse
import json
import math
import statistics

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import run_performance_suite as suite
from tools.dspark.profile_request_ids import validate_point_request_ids
from tools.dspark.verification_tools import checkpoint_preflight, measured_scheduler_overhead

UPSTREAM_COMMIT = "e2e335334669d1c94c7351937474c0104dcbfdfb"


def grid(maximum, captures, contexts, output_tokens):
    if maximum < 1 or not contexts or contexts != sorted(set(contexts)) or contexts[0] < 1 or output_tokens < 1:
        raise ValueError("Invalid profile request/context grid")
    suite.capture_sizes("dspark_graph", maximum, maximum * 6, captures)
    counts = {1, maximum, *(min(c, maximum) for c in captures)}
    n = 1
    while n < maximum:
        counts.add(n)
        n *= 2
    points = []
    for context in contexts:
        for n in sorted(counts):
            for previous, capacity in zip([0] + captures[:-1], captures):
                if n > capacity or 6 * n <= previous:
                    continue
                tokens = min(capacity, 6 * n)
                extra = tokens - n
                balanced = [extra // n + (i < extra % n) for i in range(n)]
                skewed = [max(0, min(5, extra - 5 * i)) for i in range(n)]
                for name, lengths in {"balanced": balanced, "skewed": skewed}.items():
                    points.append(
                        {
                            "id": f"ctx{context}-n{n}-t{capacity}-{name}",
                            "requests": n,
                            "actual_tokens": tokens,
                            "capacity": capacity,
                            "lengths": lengths,
                            "layout": name,
                            "prompt_tokens": context,
                            "context_ceiling": context + output_tokens,
                        }
                    )
    return sorted(counts), points


def _validate_event(row, point, identity):
    """Check execution integrity before any sampling-domain/window filtering."""
    n = row["requests"]
    q = row["query_lengths"]
    upper = row["scheduler_computed_upper_bounds"]
    kv = row["effective_kv_before_query"]
    attention = row["attention_seq_lens"]
    integers = [
        n,
        row["request_capacity"],
        row["capacity"],
        row["actual_tokens"],
        row["size"],
        row["context"],
        row["max_model_len"],
        *q,
        *upper,
        *kv,
        *attention,
    ]
    if (
        any(type(value) is not int for value in integers)
        or row["point"] != point["id"]
        or row["kind"] not in ("target", "draft")
        or row["context_semantics"] != identity["cost_context_semantics"]
        or not math.isfinite(row["seconds"])
        or row["seconds"] <= 0
        or not 0 < n <= row["request_capacity"] <= row["capacity"]
        or not 0 < row["actual_tokens"] <= row["capacity"]
        or any(len(values) != n for values in (q, upper, kv, attention))
        or any(value < 0 for value in (*upper, *kv))
        or any(value <= 0 or (row["full_decode"] and value > 6) for value in q)
        or sum(q) != row["actual_tokens"]
        or row["context"] != max(upper)
        or row["max_model_len"] <= 0
        or row["max_model_len"] != identity["max_model_len"]
        or any(
            length != previous + query or length > row["max_model_len"]
            for previous, query, length in zip(kv, q, attention)
        )
        or (row["kind"] == "target" and row["size"] != row["capacity"])
        or (row["kind"] == "draft" and not 0 < row["size"] <= n)
        or row["capacity"] > identity["max_num_batched_tokens"]
        or (row["full_decode"] and row["capacity"] not in identity["capture_sizes"])
    ):
        raise ValueError("Invalid NPU timing/context/layout sample")


def point_samples(point, snapshots, warmup, samples, ranks):
    if warmup < 1 or samples < 5:
        raise ValueError("Each point requires warmup and at least five valid samples")
    by_rank = {r["rank"]: r for r in snapshots}
    if len(snapshots) != ranks or set(by_rank) != set(range(ranks)):
        raise ValueError("Missing/duplicate profile rank")
    retained = []
    for rank, snapshot in sorted(by_rank.items()):
        if (
            snapshot.get("error")
            or snapshot["failed_execution_count"]
            or snapshot["cost_profile"].get("source") != "isolated_npu_event_profile"
        ):
            raise ValueError("Failed profile execution or missing event producer")
        events = snapshot["cost_profile"]["measurements"]
        identity = snapshot["cost_profile"]["identity"]
        # All events, including nonmatching layouts and extra records, must be
        # structurally valid. No timing threshold is used to choose samples.
        for index, row in enumerate(events):
            row["sample_selection"] = {"raw_index": index, "classification": "invalid"}
            try:
                _validate_event(row, point, identity)
            except (ValueError, KeyError, TypeError) as error:
                row["sample_selection"]["reason"] = str(error)
                raise ValueError(f"Invalid profile event: rank={rank}, raw_index={index}: {error}") from error
            row["sample_selection"]["classification"] = "validated"
        for kind in ("target", "draft"):
            eligible, excluded = [], []
            for index, row in enumerate(events):
                if row["kind"] != kind:
                    continue
                reason = None
                if not (
                    row["full_decode"]
                    and row["requests"] == point["requests"]
                    and row["size"] == (point["capacity"] if kind == "target" else point["requests"])
                    and row["capacity"] == point["capacity"]
                    and row["actual_tokens"] == point["actual_tokens"]
                    and sorted(row["query_lengths"]) == sorted(x + 1 for x in point["lengths"])
                ):
                    reason = "outside_layout_domain"
                elif row["context"] > point["context_ceiling"]:
                    reason = "outside_context_domain"
                if reason:
                    row["sample_selection"].update(classification="out_of_domain", reason=reason)
                    excluded.append({"raw_index": index, "reason": reason, "context": row["context"]})
                else:
                    ordinal = len(eligible)
                    window = "warmup" if ordinal < warmup else "retained" if ordinal < warmup + samples else "extra"
                    row["sample_selection"].update(classification=window, eligible_index=ordinal)
                    eligible.append(row)
            if len(eligible) < warmup + samples:
                raise ValueError(
                    f"Insufficient real FULL samples for {point['id']}, rank {rank}, {kind}: {len(eligible)}"
                )
            measured = eligible[warmup : warmup + samples]
            retained.append(
                {
                    "rank": rank,
                    "kind": kind,
                    "context_semantics": identity["cost_context_semantics"],
                    "sampling_context_domain": [0, point["context_ceiling"]],
                    "selected_context_range": [
                        min(r["context"] for r in measured),
                        max(r["context"] for r in measured),
                    ],
                    "selected_raw_indices": [r["sample_selection"]["raw_index"] for r in measured],
                    "warmup": eligible[:warmup],
                    "samples": measured,
                    "median_seconds": statistics.median(r["seconds"] for r in measured),
                    "extra_sample_count": len(eligible) - warmup - samples,
                    "excluded_sample_count": len(excluded),
                    "excluded_counts_by_reason": {
                        reason: sum(item["reason"] == reason for item in excluded)
                        for reason in ("outside_context_domain", "outside_layout_domain")
                    },
                    "excluded_samples": excluded,
                }
            )
    return retained


def compile_startup(records, identity, request_grid, *, checkpoint, plugin_sha, raw_hashes, overhead):
    if identity.get("diagnostic_only"):
        raise ValueError("Diagnostic profile samples cannot produce a cost table")
    cells = {}
    for record in records:
        p = record["point"]
        key = (p["requests"], p["capacity"], p["context_ceiling"])
        cell = cells.setdefault(
            key, {"requests": key[0], "capacity": key[1], "context_ceiling": key[2], "raw_layout_medians": {}}
        )
        cell.setdefault("selected_context_ranges", {})[p["layout"]] = [
            {"rank": r["rank"], "kind": r["kind"], "range": r["selected_context_range"]} for r in record["retained"]
        ]
        cell["raw_layout_medians"][p["layout"]] = {
            kind: max(r["median_seconds"] for r in record["retained"] if r["kind"] == kind)
            for kind in ("target", "draft")
        }
    for cell in cells.values():
        if set(cell["raw_layout_medians"]) != {"balanced", "skewed"}:
            raise ValueError("Both sampled layouts are required")
        for kind in ("target", "draft"):
            cell[f"raw_{kind}_seconds"] = max(row[kind] for row in cell["raw_layout_medians"].values())
    for cell in cells.values():
        # Draft is eager: request count is its primary size index; only timings
        # adjacent to actual FULL target replay are eligible (upstream rule).
        draft = [
            c
            for c in cells.values()
            if c["requests"] <= cell["requests"] and c["context_ceiling"] <= cell["context_ceiling"]
        ]
        target = [c for c in draft if c["capacity"] <= cell["capacity"]]
        cell["target_seconds"] = max(c["raw_target_seconds"] for c in target)
        cell["draft_seconds"] = max(c["raw_draft_seconds"] for c in draft)
    return {
        "schema_version": 2,
        "source": "startup_npu_event_profile",
        "unit": "seconds",
        "performance_eligible": False,
        "plugin_sha": plugin_sha,
        "core_sha": suite.CORE_SHA,
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": checkpoint,
        "identity": identity,
        "model_initializations": 1,
        "worker_replicas": identity["tp"],
        "raw_measurements_sha256": raw_hashes,
        "request_grid": request_grid,
        "context_semantics": identity["cost_context_semantics"],
        "context_ceilings": sorted({c["context_ceiling"] for c in cells.values()}),
        "cells": list(cells.values()),
        "scheduler_seconds": overhead,
        "method": {
            "timing": "NPU events around successful FULL target and adjacent eager draft",
            "processing": "per-rank median; max across ranks/layouts; monotone upper envelope",
            "lookup": "ceil graph token capacity, then ceil context and request grid; no extrapolation",
            "layout_scope": "balanced/skewed prefixes; envelope estimate, not an arbitrary-layout bound",
            "context_scope": "scheduler pre-query upper bound; ceil prompt+output bucket, not corrected KV",
            "context_estimate": "bucket envelope from selected ranges; bucket ceiling is not a measured length",
            "cleanup": "normal scheduler terminal cleanup, unique request IDs; separate engine",
            "synchronization": "profile-only phase boundaries; not performance eligible",
        },
    }


def point_numeric_status(snapshots):
    observations = [(s.get("cost_profile") or {}).get("observation") for s in snapshots]
    if not observations or any(
        not o
        or o.get("recording_error")
        or not (o.get("numeric") or {}).get("enabled")
        or o["numeric"].get("nan_rounds") is None
        or not o["numeric"].get("compact_host_transfers")
        or o["numeric"].get("compact_host_transfers_completed") != o["numeric"].get("compact_host_transfers")
        for o in observations
    ):
        return "unavailable"
    if any(
        o["numeric"].get("nan_rounds", 0) or (o.get("auxiliary") or {}).get("counts", {}).get("nan_rounds", 0)
        for o in observations
    ):
        return "nan_observed"
    if any((o.get("target_internal") or {}).get("counts", {}).get("nonfinite_rounds", 0) for o in observations):
        return "nonfinite_observed"
    return "no_nan_observed_at_enabled_boundaries"


def collect(engine_factory, points, sampling, directory, *, warmup, samples, ranks=8):
    """Injected factory for CPU lifecycle tests; one construction, unconditional shutdown."""
    lifecycle = {"engine_initialization_attempts": 1, "engine_initializations": 0, "shutdown": False}
    benchmark._atomic_write_json(directory / "lifecycle.json", lifecycle)
    engine = engine_factory()
    lifecycle["engine_initializations"] = 1
    benchmark._atomic_write_json(directory / "lifecycle.json", lifecycle)
    records = []
    identity = None
    point = None
    request_history = {}
    raw = None
    primary_error = None
    progress = {
        "performance_eligible": False,
        "status": "running",
        "planned_points": len(points),
        "completed_points": [],
    }
    try:
        benchmark._atomic_write_json(directory / "point-completion.json", progress)
        engine.collective_rpc("dspark_benchmark_replay_snapshot")  # install after capture
        token = engine.get_tokenizer().encode("x", add_special_tokens=False)[0]
        for point in points:
            raw = None
            engine.collective_rpc(
                "dspark_benchmark_profile_point", kwargs={"point": point["id"], "lengths": point["lengths"]}
            )
            prompts = [{"prompt_token_ids": [token] * point["prompt_tokens"]} for _ in range(point["requests"])]
            # generate drains all requests. StreamingEngine gives each call a
            # fresh batch namespace; scheduler retires proposals on next admission.
            outputs = engine.generate(prompts, sampling, use_tqdm=False, profile_point=point["id"])
            snapshots = engine.collective_rpc("dspark_benchmark_replay_snapshot")
            raw = {"point": point, "ranks": snapshots, "streaming": engine.last_batch, "performance_eligible": False}
            path = directory / f"{point['id']}.json"
            benchmark._atomic_write_json(path, raw)  # keep raw evidence before acceptance
            if len(outputs) != point["requests"] or engine.last_batch["scheduler"].get("corrupted_requests", 0):
                raise ValueError("Incomplete/corrupted synthetic requests")
            raw["request_identity_validation"] = validate_point_request_ids(
                point["id"], engine.last_batch, snapshots, request_history
            )
            benchmark._atomic_write_json(path, raw)
            for snapshot in snapshots:
                current = snapshot["cost_profile"]["identity"]
                if identity is not None and current != identity:
                    raise ValueError("Profile configuration changed within one engine")
                identity = current
            selected = point_samples(point, snapshots, warmup, samples, ranks)
            benchmark._atomic_write_json(path, raw)  # include classifications, preserve original values
            records.append(
                {
                    "point": point,
                    "retained": selected,
                    "raw_sha256": benchmark._sha256_file(path),
                }
            )
            benchmark._atomic_write_json(directory / "retained.json", records)
            progress["completed_points"].append(
                {
                    "point": point["id"],
                    "raw_sha256": records[-1]["raw_sha256"],
                    "numeric_result": point_numeric_status(snapshots),
                    "stream_error": engine.last_batch.get("error"),
                    "generated_tokens": [
                        len(r["output_token_ids"]) if "output_token_ids" in r else None
                        for r in engine.last_batch.get("requests", [])
                    ],
                }
            )
            progress["status"] = "completed" if len(records) == len(points) else "running"
            benchmark._atomic_write_json(directory / "point-completion.json", progress)
        return records, identity
    except BaseException as error:
        primary_error = error
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            benchmark._atomic_write_json(directory / "point-completion.json", progress)
        except OSError as evidence_error:
            lifecycle["progress_evidence_error"] = str(evidence_error)
        if point is not None:
            if raw is None:
                raw = {"point": point, "streaming": engine.last_batch, "performance_eligible": False}
                try:
                    if getattr(getattr(engine, "profile_guard", None), "first", None) is not None:
                        raise RuntimeError("Engine failed; post-mortem RPC not retried")
                    raw["ranks"] = engine.collective_rpc("dspark_benchmark_replay_snapshot")
                except Exception as snapshot_error:
                    raw["snapshot_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
            raw["request_identity_failure"] = getattr(error, "evidence", None)
            if raw.get("ranks") is None:
                raw["request_identity_status"] = "unavailable: worker snapshots were not returned"
            if (
                raw["request_identity_failure"] is None
                and raw.get("ranks") is not None
                and "request_identity_validation" not in raw
            ):
                try:
                    validate_point_request_ids(point["id"], engine.last_batch, raw["ranks"], request_history)
                except ValueError as identity_error:
                    raw["request_identity_failure"] = getattr(identity_error, "evidence", None)
            benchmark._atomic_write_json(directory / f"{point['id']}.json", raw)
        first = getattr(getattr(engine, "profile_guard", None), "first", None)
        benchmark._atomic_write_json(
            directory / "profile-failure.json",
            {
                "point": point,
                "error": first["error"] if first else f"{type(error).__name__}: {error}",
                "first_failure": first,
                "propagated_error": f"{type(error).__name__}: {error}",
                "last_stream": engine.last_batch,
                "request_identity_failure": (raw or {}).get("request_identity_failure"),
                "performance_eligible": False,
            },
        )
        raise
    finally:
        try:
            engine.shutdown()
            cleanup = getattr(engine, "cleanup_result", None)
            if cleanup is not None and not cleanup.get("success", cleanup["shutdown_completed"]):
                raise RuntimeError(f"Profile engine cleanup failed: {cleanup.get('status', 'incomplete')}")
        except BaseException as cleanup_error:
            lifecycle["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            try:
                benchmark._atomic_write_json(
                    directory / "cleanup-failure.json",
                    {
                        "performance_eligible": False,
                        "phase": "cleanup",
                        "error": lifecycle["cleanup_error"],
                        "prior_error": f"{type(primary_error).__name__}: {primary_error}" if primary_error else None,
                        "points_status": progress["status"],
                        "completed_points": len(records),
                    },
                )
            except OSError as evidence_error:
                lifecycle["cleanup_evidence_error"] = str(evidence_error)
            if primary_error is None:
                raise
        finally:
            cleanup = getattr(engine, "cleanup_result", None)
            lifecycle["shutdown"] = (
                cleanup.get("success", cleanup.get("shutdown_completed", False))
                if cleanup
                else "cleanup_error" not in lifecycle
            )
            if cleanup is not None:
                lifecycle["cleanup"] = cleanup
            lifecycle["points_status"] = progress["status"]
            lifecycle["completed_points"] = len(records)
            try:
                benchmark._atomic_write_json(directory / "lifecycle.json", lifecycle)
            except OSError as evidence_error:
                if primary_error is None and "cleanup_error" not in lifecycle:
                    raise
                print(f"PROFILE_LIFECYCLE_EVIDENCE_UNAVAILABLE: {evidence_error}", flush=True)


def diagnostic_points(points, stop):
    indices = [i for i, point in enumerate(points) if point["id"] == stop]
    if len(indices) != 1:
        raise ValueError("Diagnostic stop point must exist exactly once in the configured grid")
    return points[: indices[0] + 1]  # keep every predecessor in the same engine


def profile_engine_kwargs(parsed, directory, diagnostic, experiment=None, target_layer=None, worker_exit=False):
    if worker_exit and experiment != "target-boundaries":
        raise ValueError("Worker exit tracing requires target-boundaries")
    if target_layer is not None and (
        experiment != "target-boundaries" or type(target_layer) is not int or target_layer < 0
    ):
        raise ValueError("A nonnegative target detail layer requires target-boundaries")
    kwargs = benchmark.build_engine_kwargs(parsed)
    if diagnostic:
        options = kwargs.get("additional_config", {}).get("dspark_confidence_verification", {})
        if not options.get("profile") or options.get("mode") != "specified_lengths":
            raise ValueError("NaN observer requires isolated specified-length profiling")
        kwargs["additional_config"] = {
            **kwargs["additional_config"],
            "dspark_profile_nan_diagnostic_dir": str(directory.resolve()),
        }
    if experiment in (
        "metadata-only",
        "context-kv-sync",
        "numeric-boundaries",
        "upstream-boundaries",
        "auxiliary-transfers",
        "target-boundaries",
    ):
        if diagnostic:
            raise ValueError("Full diagnostics and low-interference experiments are mutually exclusive")
        options = kwargs.get("additional_config", {}).get("dspark_confidence_verification", {})
        if not options.get("profile") or options.get("mode") != "specified_lengths":
            raise ValueError("Profile observation requires isolated specified-length profiling")
        kwargs["additional_config"] = {
            **kwargs["additional_config"],
            "dspark_profile_observation": {"mode": experiment, "directory": str(directory.resolve())},
        }
        if target_layer is not None:
            kwargs["additional_config"]["dspark_profile_observation"]["target_layer"] = target_layer
    if experiment in (
        "metadata-only",
        "numeric-boundaries",
        "upstream-boundaries",
        "auxiliary-transfers",
        "target-boundaries",
    ):
        kwargs["distributed_executor_backend"] = (
            "vllm_ascend.diagnostics.dspark_profile_executor.ProfileMultiprocExecutor"
        )
        kwargs["additional_config"]["dspark_profile_failure_dir"] = str(directory.parent.resolve())
    if worker_exit:
        kwargs["worker_cls"] = "vllm_ascend.diagnostics.dspark_profile_worker.ProfileNPUWorker"
        kwargs["additional_config"]["dspark_profile_worker_exit"] = True
    return kwargs


def run(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    suite.source_gate(args)
    checkpoint = checkpoint_preflight(args.model)
    benchmark._atomic_write_json(root / "checkpoint.json", checkpoint)
    counts, points = grid(args.batch, args.capture, args.profile_contexts, args.profile_output_tokens)
    diagnostic = getattr(args, "profile_nan_diagnostic", False)
    experiment = getattr(args, "profile_experiment", None)
    target_layer = getattr(args, "profile_target_layer", None)
    if diagnostic and experiment:
        raise ValueError("Full diagnostics and low-interference experiments are mutually exclusive")
    isolated = diagnostic or experiment is not None
    if isolated:
        points = diagnostic_points(points, args.profile_stop_after_point)
        benchmark._atomic_write_json(
            root / "diagnostic.json",
            {
                "performance_eligible": False,
                "experiment": experiment or "full-diagnostic",
                "target_layer": target_layer,
                "worker_exit_trace": getattr(args, "profile_worker_exit", False),
                "status": "running",
                "root_cause": "ROOT_CAUSE_NOT_YET_PROVEN",
                "points": [point["id"] for point in points],
                "worker_directory": str((root / "worker-first-failure").resolve()),
                "observation_effect": (
                    "Any observation/wait may change reproduction; completion is not a repair; no cost table"
                ),
            },
        )
    if max(args.profile_contexts) + args.profile_output_tokens > args.max_model_len:
        raise ValueError("Synthetic context/output budget exceeds max_model_len; no truncation")
    if args.profile_output_tokens < 6 * (args.profile_warmup + args.profile_samples + 2):
        raise ValueError("Synthetic output budget cannot supply the required FULL samples")
    options = root / "verification.json"
    benchmark._atomic_write_json(options, {"mode": "specified_lengths", "lengths": [5], "profile": True})
    local = argparse.Namespace(**vars(args))
    local.max_num_seqs = [args.batch]
    local.repeats = 1
    local.modes = ["dspark_confidence_graph"]
    local.capture_dspark = args.capture
    local.capture_target = None
    local.confidence_verification = options
    local.num_prompts = args.batch
    local.warmup_prompts = 0
    local.client_outstanding = None
    local.output_len = args.profile_output_tokens
    plan = suite.create_plan(local, root / "synthetic.jsonl", root)
    benchmark._atomic_write_json(root / "plan.json", {**plan, "points": points, "performance_eligible": False})
    argv = plan["runs"][0]["command"][2:]
    argv[argv.index("--no-ignore-eos")] = "--ignore-eos"  # synthetic profile ONLY
    parsed = benchmark.parse_args(argv)
    if isolated:
        receipt = json.loads((root / "diagnostic.json").read_text())
        receipt["effective_benchmark_argv"] = argv
        benchmark._atomic_write_json(root / "diagnostic.json", receipt)
    suite.resources_idle(root / "npu-before.log")
    from tools.dspark.performance_stream import StreamingEngine

    sampling = benchmark._sampling_params(parsed)

    def initialize():
        engine = StreamingEngine(
            profile_engine_kwargs(
                parsed,
                root / "worker-first-failure",
                diagnostic,
                experiment,
                target_layer,
                getattr(args, "profile_worker_exit", False),
            ),
            parsed,
        )
        try:
            runtime = benchmark._collect_worker_graph_runtime(engine, parsed)
            benchmark._atomic_write_json(root / "capture.json", runtime)
        except BaseException as error:
            guard = getattr(engine, "profile_guard", None)
            if guard is not None:
                guard.remember(error)
            try:
                engine.shutdown()
            except BaseException as cleanup_error:
                benchmark._atomic_write_json(
                    root / "initialization-cleanup-error.json",
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "cleanup_error": f"{type(cleanup_error).__name__}: {cleanup_error}",
                        "performance_eligible": False,
                    },
                )
            raise
        return engine

    try:
        records, identity = collect(
            initialize, points, sampling, root, warmup=args.profile_warmup, samples=args.profile_samples
        )
    except BaseException as error:
        if isolated:
            receipt = json.loads((root / "diagnostic.json").read_text())
            failure_path = root / "engine-failure.json"
            first = json.loads(failure_path.read_text()) if failure_path.exists() else None
            receipt.update(
                status="failed",
                error=first["error"] if first else f"{type(error).__name__}: {error}",
                first_failure=first,
                propagated_error=f"{type(error).__name__}: {error}",
            )
            for name in ("point-completion", "cleanup"):
                path = root / f"{name}.json"
                if path.exists():
                    receipt[name] = json.loads(path.read_text())
            benchmark._atomic_write_json(root / "diagnostic.json", receipt)
        raise
    if isolated:
        receipt = json.loads((root / "diagnostic.json").read_text())
        receipt["status"] = "completed_without_observed_failure"
        for name in ("point-completion", "cleanup"):
            path = root / f"{name}.json"
            if path.exists():
                receipt[name] = json.loads(path.read_text())
        benchmark._atomic_write_json(root / "diagnostic.json", receipt)
        return 0  # diagnostic timings must never become a usable cost table
    suite.resources_idle(root / "npu-after.log")
    table = compile_startup(
        records,
        identity,
        counts,
        checkpoint=checkpoint,
        plugin_sha=args.plugin_sha,
        raw_hashes=[r["raw_sha256"] for r in records],
        overhead=0,
    )
    from vllm_ascend.spec_decode.dspark_verification import CostTable

    table["scheduler_seconds"] = measured_scheduler_overhead(identity, CostTable.load_startup(table, identity))
    CostTable.load_startup(table, identity)
    benchmark._atomic_write_json(root / "cost-profile.json", table)
    return 0
