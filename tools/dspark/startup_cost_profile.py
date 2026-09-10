# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One isolated engine per configuration, real disposable requests per profile point.

Unlike upstream dummy_run, normal scheduler admission owns all KV and terminal
cleanup. No model/process is created inside the point loop. This is a startup
calibration workload, never a performance or quality result.
"""

import argparse
import math
import statistics

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import run_performance_suite as suite
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
        for kind in ("target", "draft"):
            rows = [
                r
                for r in snapshot["cost_profile"]["measurements"]
                if r["point"] == point["id"]
                and r["kind"] == kind
                and r["full_decode"]
                and r["requests"] == point["requests"]
                and r["size"] == (point["capacity"] if kind == "target" else point["requests"])
                and r["capacity"] == point["capacity"]
                and r["actual_tokens"] == point["actual_tokens"]
                and sorted(r["query_lengths"]) == sorted(x + 1 for x in point["lengths"])
            ]
            if len(rows) < warmup + samples:
                raise ValueError(f"Insufficient real FULL samples for {point['id']}, rank {rank}, {kind}: {len(rows)}")
            if any(
                not math.isfinite(r["seconds"])
                or r["seconds"] <= 0
                or not 0 <= r["context"] <= point["context_ceiling"]
                or not r["requests"] <= r["request_capacity"] <= r["capacity"]
                or len(r["query_lengths"]) != r["requests"]
                or sum(r["query_lengths"]) != r["actual_tokens"]
                for r in rows
            ):
                raise ValueError("Invalid NPU timing/context sample")
            measured = rows[warmup : warmup + samples]
            retained.append(
                {
                    "rank": rank,
                    "kind": kind,
                    "warmup": rows[:warmup],
                    "samples": measured,
                    "median_seconds": statistics.median(r["seconds"] for r in measured),
                    "extra_sample_count": len(rows) - warmup - samples,
                }
            )
    return retained


def compile_startup(records, identity, request_grid, *, checkpoint, plugin_sha, raw_hashes, overhead):
    cells = {}
    for record in records:
        p = record["point"]
        key = (p["requests"], p["capacity"], p["context_ceiling"])
        cell = cells.setdefault(
            key, {"requests": key[0], "capacity": key[1], "context_ceiling": key[2], "raw_layout_medians": {}}
        )
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
        "context_ceilings": sorted({c["context_ceiling"] for c in cells.values()}),
        "cells": list(cells.values()),
        "scheduler_seconds": overhead,
        "method": {
            "timing": "NPU events around successful FULL target and adjacent eager draft",
            "processing": "per-rank median; max across ranks/layouts; monotone upper envelope",
            "lookup": "ceil graph token capacity, then ceil context and request grid; no extrapolation",
            "layout_scope": "balanced/skewed prefixes; envelope estimate, not an arbitrary-layout bound",
            "context_scope": "max host computed length; ceil sampled prompt+output bucket",
            "cleanup": "normal scheduler terminal cleanup, unique request IDs; separate engine",
            "synchronization": "profile-only phase boundaries; not performance eligible",
        },
    }


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
    try:
        engine.collective_rpc("dspark_benchmark_replay_snapshot")  # install after capture
        token = engine.get_tokenizer().encode("x", add_special_tokens=False)[0]
        for point in points:
            engine.collective_rpc(
                "dspark_benchmark_profile_point", kwargs={"point": point["id"], "lengths": point["lengths"]}
            )
            prompts = [{"prompt_token_ids": [token] * point["prompt_tokens"]} for _ in range(point["requests"])]
            # generate drains all requests. StreamingEngine gives each call a
            # fresh batch namespace; scheduler retires proposals on next admission.
            outputs = engine.generate(prompts, sampling, use_tqdm=False)
            snapshots = engine.collective_rpc("dspark_benchmark_replay_snapshot")
            raw = {"point": point, "ranks": snapshots, "streaming": engine.last_batch, "performance_eligible": False}
            path = directory / f"{point['id']}.json"
            benchmark._atomic_write_json(path, raw)  # keep raw evidence before acceptance
            if len(outputs) != point["requests"] or engine.last_batch["scheduler"].get("corrupted_requests", 0):
                raise ValueError("Incomplete/corrupted synthetic requests")
            request_ids = {r["request_id"] for r in engine.last_batch["requests"]}
            for snapshot in snapshots:
                for measurement in snapshot["cost_profile"]["measurements"]:
                    if set(measurement["request_ids"]) - request_ids:
                        raise ValueError("Previous point request metadata leaked into current profile events")
                current = snapshot["cost_profile"]["identity"]
                if identity is not None and current != identity:
                    raise ValueError("Profile configuration changed within one engine")
                identity = current
            records.append(
                {
                    "point": point,
                    "retained": point_samples(point, snapshots, warmup, samples, ranks),
                    "raw_sha256": benchmark._sha256_file(path),
                }
            )
            benchmark._atomic_write_json(directory / "retained.json", records)
        return records, identity
    except BaseException as error:
        benchmark._atomic_write_json(
            directory / "profile-failure.json",
            {
                "point": point,
                "error": f"{type(error).__name__}: {error}",
                "last_stream": engine.last_batch,
                "performance_eligible": False,
            },
        )
        raise
    finally:
        engine.shutdown()
        lifecycle["shutdown"] = True
        benchmark._atomic_write_json(directory / "lifecycle.json", lifecycle)


def run(args):
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    suite.source_gate(args)
    checkpoint = checkpoint_preflight(args.model)
    benchmark._atomic_write_json(root / "checkpoint.json", checkpoint)
    counts, points = grid(args.batch, args.capture, args.profile_contexts, args.profile_output_tokens)
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
    suite.resources_idle(root / "npu-before.log")
    from tools.dspark.performance_stream import StreamingEngine

    sampling = benchmark._sampling_params(parsed)

    def initialize():
        engine = StreamingEngine(benchmark.build_engine_kwargs(parsed), parsed)
        try:
            runtime = benchmark._collect_worker_graph_runtime(engine, parsed)
            benchmark._atomic_write_json(root / "capture.json", runtime)
        except BaseException:
            engine.shutdown()
            raise
        return engine

    records, identity = collect(
        initialize, points, sampling, root, warmup=args.profile_warmup, samples=args.profile_samples
    )
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
