# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Six frozen GSM8K comparisons; immutable published costs, one engine per case."""

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.dspark import confidence_acceptance as acceptance
from tools.dspark import fixed_k_comparison as fixed_k
from tools.dspark import formal_cost as formal
from tools.dspark import shutdown_acceptance, shutdown_policy
from tools.dspark.audit_formal_cost import require
from tools.dspark.batch_expansion import capacity_check
from tools.dspark.performance_stream import request_latency

benchmark = acceptance.benchmark
suite = acceptance.suite
write = acceptance.write
read = formal.read
PRODUCER = "94328d6020d1988a45afe64f8894bb6324bf5bcd"
EXPANSION = Path("/workspace/dspark-results/dspark-large-batch.6tf75BWi")
COSTS = {
    64: (acceptance.PRODUCER, acceptance.TABLE_SHA, acceptance.COST_DIRECTORY),
    128: (
        PRODUCER,
        "d60fb91e3dd89548566f717b1144e11578d1138d80fba6adcc7e857a16c7b50b",
        EXPANSION / "b128/cost/runs/b128",
    ),
    256: (
        PRODUCER,
        "94cb6f920a1ab387b5fc1b7173ad3089c5c79d9f7469046174db4ccfa3ee161b",
        EXPANSION / "b256/cost/runs/b256",
    ),
}
ROUNDS = 3
MODEL_SECONDS = 3600
PREPARE_SECONDS = 1800
GROUP_SECONDS = 25000
COMPATIBILITY = Path(__file__).with_name("PERFORMANCE_CODE_COMPATIBILITY.json")


def plan():
    return {
        "name": "dspark-six-case-performance-v1",
        "cases": [
            {
                "batch": b,
                "mode": m,
                "name": f"b{b}-{m}",
                "client_outstanding": b,
                "distinct_questions": 64,
                "instances": b,
                "capture_sizes": formal.captures(b),
                "warmup_rounds": 1,
                "measured_rounds": ROUNDS,
                "max_new_tokens": 256,
                "max_requests_total": b * (ROUNDS + 1),
                "max_output_tokens_total": b * (ROUNDS + 1) * 256,
                "model_runtime_limit_seconds": MODEL_SECONDS,
            }
            for b in (64, 128, 256)
            for m in ("fixed", "confidence")
        ],
        "model_initializations": 6,
        "host_preflight_seconds": 600,
        "prepare_seconds": PREPARE_SECONDS,
        "total_limit_seconds": GROUP_SECONDS,
        "outer_kill_margin_seconds": 65,
        "shutdown_budget": shutdown_policy.budget(shutdown_policy.POLICY_NAME),
        "stage_sum_upper_seconds": 600 + PREPARE_SECONDS + 6 * (MODEL_SECONDS + 48 + 15),
        "sampling": {"temperature": 0, "top_p": 1, "top_k": -1, "seed": 0, "ignore_eos": False},
        "order": "B64 fixed/confidence, B128 fixed/confidence, B256 fixed/confidence; no retries",
        "scope": "steady generation; actual outputs may differ; no quality equivalence claim",
        "original_5s_budget": "NOT_EVALUATED",
    }


def experiment_plan(args):
    return fixed_k.plan(plan()) if getattr(args, "fixed_k_comparison", False) else plan()


def case_name(args):
    return (
        f"b{args.batch}-{args.mode}-k{args.draft_k}"
        if getattr(args, "fixed_k_comparison", False)
        else f"b{args.batch}-{args.mode}"
    )


def verify_code(plugin):
    contract = read(COMPATIBILITY)
    for producer in (acceptance.PRODUCER, PRODUCER):
        changes = subprocess.check_output(
            ["git", "-C", str(plugin), "diff", "--name-only", producer, "HEAD", "--", "vllm_ascend", "csrc"], text=True
        ).splitlines()
        expected = contract["producers"][producer]
        require(sorted(changes) == sorted(expected), "Unaudited execution changes since cost producer")
        for path, digest in expected.items():
            require(formal.sha(plugin / path) == digest, f"Changed audited runtime file: {path}")
    require(contract["core"] == suite.CORE_SHA, "Compatibility Core changed")
    return contract


def table_at(path, batch):
    producer, digest, _ = COSTS[batch]
    require(formal.sha(path / "cost-profile.json") == digest, "Frozen cost bytes changed")
    return acceptance.publication(path, batch, producer)


def prepare(args):
    suite.source_gate(args)
    k_experiment = getattr(args, "fixed_k_comparison", False)
    compatibility = (
        {"cost_runtime_compatibility": "NOT_CLAIMED; fixed K experiment"} if k_experiment else verify_code(args.plugin)
    )
    require(formal.real_text_contract(args.manifest) == formal.workload_contract(), "Frozen GSM8K identity changed")
    weights = formal.weight_identity(args.model)  # complete bytes, once before any model
    acceptance.copy_manifest_assets(args.manifest, args.output_dir / "input")
    _, inputs, _ = acceptance.read_manifest(args.manifest, 64)
    assets = args.output_dir / "assets"
    for batch, (_, _, source) in COSTS.items():
        if k_experiment and batch != 256:
            continue
        table, proof = table_at(source, batch)
        require(table["weight_provenance"] == weights, "Weights differ from cost producer")
        destination = assets / f"b{batch}"
        destination.mkdir(parents=True)
        for name in ("cost-profile.json", "cost-publication.json"):
            shutil.copyfile(source / name, destination / name)
        write(
            destination / "identity.json",
            {
                "producer": table["plugin_sha"],
                "table_sha256": COSTS[batch][1],
                "publication": proof,
                "runtime": table["identity"],
            },
        )
        (assets / f"b{batch}.jsonl").write_bytes(
            b"".join(benchmark._canonical_json_bytes(r) for r in inputs * (batch // 64))
        )
    import torch
    import torch_npu  # noqa: F401 -- register actual device backend

    from tools.dspark.operator_replay import runtime_identity

    require(torch.npu.device_count() == 8, "Exactly eight visible NPUs are required")
    environment = runtime_identity()
    require(not environment["artifact_truncated"], "OPP identity truncated")
    write(args.output_dir / "runtime-environment.json", environment)
    write(
        args.output_dir / "preflight.json",
        {
            "plugin_sha": args.plugin_sha,
            "core_sha": suite.CORE_SHA,
            "weights": weights,
            "code_compatibility": compatibility,
            "input_contract": formal.workload_contract(),
            "inputs_sha256": {str(b): formal.sha(assets / f"b{b}.jsonl") for b in ([256] if k_experiment else COSTS)},
            "plan": experiment_plan(args),
        },
    )


def case_config(args, root):
    config = root / "verification.json"
    if args.mode == "confidence":
        write(
            config,
            {
                "mode": "confidence",
                "profile": False,
                "cost_profile": str((args.output_dir / f"assets/b{args.batch}/cost-profile.json").resolve()),
            },
        )
    local = argparse.Namespace(
        **{
            **vars(args),
            "max_num_seqs": [args.batch],
            "repeats": 1,
            "modes": ["dspark_graph" if args.mode == "fixed" else "dspark_confidence_graph"],
            "capture_dspark": fixed_k.captures(args.draft_k)
            if getattr(args, "fixed_k_comparison", False)
            else formal.captures(args.batch),
            "capture_target": None,
            "confidence_verification": config if args.mode == "confidence" else None,
            "num_prompts": args.batch,
            "warmup_prompts": 0,
            "client_outstanding": args.batch,
            "output_len": 256,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.9,
        }
    )
    execution = suite.create_plan(local, args.output_dir / f"assets/b{args.batch}.jsonl", root)
    parsed = benchmark.parse_args(execution["runs"][0]["command"][2:])
    kwargs = benchmark.build_engine_kwargs(parsed)
    additional = kwargs.setdefault("additional_config", {})
    additional.update(
        dspark_performance_comparison=True,
        dspark_profile_failure_dir=str(root.resolve()),
        dspark_profile_worker_exit=True,
        dspark_profile_shutdown_policy=shutdown_policy.POLICY_NAME,
        dspark_profile_stack_signals=False,
        dspark_profile_exit_debugger=False,
    )
    if getattr(args, "fixed_k_comparison", False):
        require(args.batch == 256 and args.mode == "fixed" and args.draft_k in (5, 8), "Invalid fixed K experiment")
        additional["dspark_fixed_k8_experiment"] = args.draft_k == 8
        additional["dspark_fixed_k_comparison"] = True
    require(shutdown_policy.performance_enabled(additional), "Invalid performance configuration")
    require(
        ("dspark_confidence_verification" in additional) == (args.mode == "confidence"), "Baseline has adaptive policy"
    )
    kwargs["worker_cls"] = "vllm_ascend.diagnostics.dspark_profile_worker.ProfileNPUWorker"
    kwargs["distributed_executor_backend"] = "vllm_ascend.diagnostics.dspark_profile_executor.ProfileMultiprocExecutor"
    write(
        root / "plan.json",
        {
            "execution": execution,
            "exit_observation": False,
            "shutdown_policy": shutdown_policy.POLICY_NAME,
            "contract": experiment_plan(args),
        },
    )
    write(root / "engine-config.json", kwargs)
    return parsed, kwargs


def percentile(values, p):
    """Nearest-rank quantile, not interpolation; exclude undefined one-token TPOT."""
    return sorted(values)[max(0, math.ceil(p * len(values)) - 1)] if values else None


def distribution(values):
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p95_nearest_rank": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def round_metrics(stream, records, batch, name):
    expected = [f"{name}:{r['request_id']}" for r in formal.workload_contract(batch)["records"]]
    require([r["request_id"] for r in stream["requests"]] == expected, "Round request identities differ")
    normalized = {
        **stream,
        "requests": [{**r, "request_id": r["request_id"].removeprefix(name + ":")} for r in stream["requests"]],
    }
    acceptance.validate_stream(normalized, records, batch)
    require(not stream["scheduler"]["corrupted_requests"], "Corrupted requests reported")
    duration = stream["finished_monotonic"] - stream["started_monotonic"]
    require(math.isfinite(duration) and duration > 0, "Invalid generation duration")
    latencies = [request_latency(r) for r in stream["requests"]]
    tokens = sum(len(r["output_token_ids"]) for r in stream["requests"])
    require(tokens > 0, "No generated output")
    return {
        "completed_requests": batch,
        "actual_output_tokens": tokens,
        "generation_seconds": duration,
        "output_tokens_per_second": tokens / duration,
        "ttft_seconds": distribution([r["ttft_seconds"] for r in latencies if r["ttft_seconds"] is not None]),
        "request_tpot_seconds": distribution(
            [r["mean_tpot_seconds"] for r in latencies if r["mean_tpot_seconds"] is not None]
        ),
        "formula": (
            "TTFT=first nonempty DELTA observation-submit; request TPOT=(completion-first)/(N-1), N>1; "
            "chunk times are not per-token arrival times"
        ),
        "output_lengths": [len(r["output_token_ids"]) for r in stream["requests"]],
        "scheduler": stream["scheduler"],
    }


def validate_replays(before, after, table, mode):
    require(
        sorted(r["rank"] for r in after) == list(range(8)) and sorted(r["rank"] for r in before) == list(range(8)),
        "Missing ranks",
    )
    before = sorted(before, key=lambda r: r["rank"])
    after = sorted(after, key=lambda r: r["rank"])
    counters = []
    for initial, final in zip(before, after):
        require(
            final["observer_id"] == initial["observer_id"]
            and not final["error"]
            and not final["failed_execution_count"],
            "Unavailable FULL evidence",
        )
        require(initial["performance"]["calls"] == 0 and final["performance"]["calls"] > 0, "Missing/reset FULL calls")
        require(final["performance"] == after[0]["performance"], "Cross-rank FULL distributions differ")
        require(
            not final.get("cost_profile") and not final.get("confidence_execution_receipts"),
            "Heavy instrumentation installed",
        )
        if mode == "fixed":
            require("confidence_verification" not in final, "Adaptive policy active in baseline")
            continue
        a, b = initial["confidence_verification"], final["confidence_verification"]
        require(
            b["cost_profile"]["identity"] == table["identity"] and b["weights"] == table["loaded_confidence_weights"],
            "Actual runtime/weights mismatch",
        )
        require(
            b["mode"] == "confidence" and b["decisions"] is None and b["specified_batches"] == 0,
            "Wrong or heavy policy",
        )
        delta = {
            key: b[key] - a[key]
            for key in ("confidence_head_calls", "confidence_batches", "verified", "accepted", "scheduled")
        }
        delta["length_histogram"] = [y - x for x, y in zip(a["length_histogram"], b["length_histogram"])]
        require(delta["confidence_batches"] > 0 and delta["confidence_head_calls"] > 0, "No real confidence decisions")
        counters.append(delta)
    require(not counters or all(c == counters[0] for c in counters), "TP policy counters differ")
    full = after[0]["performance"]
    return {
        "FULL": full,
        "confidence": counters[0] if counters else None,
        "max_actual_scheduled_requests": max(r["requests"] for r in full["layouts"]),
        "actual_query_tokens": sum(r["query_tokens"] * r["count"] for r in full["layouts"]),
        "graph_padding_tokens": sum((r["capacity"] - r["query_tokens"]) * r["count"] for r in full["layouts"]),
        "calibration": "uncalibrated" if mode == "confidence" else None,
        "existing_runtime_wall_seconds": [
            {
                "rank": b["rank"],
                **{
                    key: b["confidence_verification"][key] - a["confidence_verification"][key]
                    for key in ("policy_seconds", "confidence_transfer_seconds")
                },
            }
            for a, b in zip(before, after)
        ]
        if mode == "confidence"
        else [],
        "timing_note": (
            "Existing runtime host clocks only, included in E2E; "
            "head/D2H and policy/TP overlap are not isolated kernels"
        ),
        "operator_breakdown": "NOT_MEASURED: would require extra instrumentation/synchronization",
    }


def acceptance_metrics(delta, mode, k=5):
    totals = delta["totals"]
    if mode == "confidence" and totals["vllm:spec_decode_num_draft_tokens"] == 0:
        require(
            totals["vllm:spec_decode_num_accepted_tokens"] == 0
            and all(v == 0 for v in totals[benchmark.VECTOR_METRIC_NAME]),
            "Accepted tokens without verified candidates",
        )
        return {"accepted_per_verified": None, "reason": "No verified candidates in this interval", "totals": totals}
    return benchmark.acceptance_from_delta(delta, k)


def run_rounds(engine, parsed, args, records, root, table):
    rounds = []
    k = getattr(args, "draft_k", 5)
    k_experiment = getattr(args, "fixed_k_comparison", False)
    for number in range(ROUNDS + 1):
        name = "warmup" if number == 0 else f"round-{number}"
        require(
            engine.engine.output_processor.get_num_unfinished_requests() == 0 and not engine.engine.errored,
            "Round began with pending/error state",
        )
        before = engine.collective_rpc("dspark_benchmark_performance_reset")
        metrics_before = benchmark.capture_spec_metrics(engine.get_metrics())
        ids = [f"{name}:{r['request_id']}" for r in formal.workload_contract(args.batch)["records"]]
        write(
            root / f"{name}-inputs.json",
            {
                "request_ids": ids,
                "frozen_instance_ids": [r["request_id"] for r in formal.workload_contract(args.batch)["records"]],
            },
        )
        if k_experiment:
            memory_before = engine.collective_rpc("dspark_benchmark_performance_memory", kwargs={"reset_peak": True})
        engine.generate(
            [{"prompt_token_ids": r["prompt_token_ids"]} for r in records],
            benchmark._sampling_params(parsed),
            profile_point=name,
            request_ids=ids,
        )
        # All RPC/D2H summaries and file writes are outside stream_batch's timer.
        stream = engine.last_batch
        write(root / f"{name}-stream.json", stream)
        after = engine.collective_rpc("dspark_benchmark_replay_snapshot")
        write(root / f"{name}-workers.json", {"before": before, "after": after})
        require(engine.engine.output_processor.get_num_unfinished_requests() == 0, "Unfinished requests after round")
        metrics = {
            **round_metrics(stream, records, args.batch, name),
            "execution": validate_replays(before, after, table, args.mode),
            "name": name,
        }
        metrics_after = benchmark.capture_spec_metrics(engine.get_metrics())
        metric_delta = benchmark.metric_snapshot_delta(metrics_before, metrics_after, k)
        metrics["acceptance"] = acceptance_metrics(metric_delta, args.mode, k)
        if k_experiment:
            metrics["draft_k"] = k
            fixed_k.validate_full(metrics["execution"], k)
            metrics["progress"] = fixed_k.progress(stream["scheduler"], metric_delta, metrics["execution"])
            memory_after = engine.collective_rpc("dspark_benchmark_performance_memory")
            require(sorted(r["rank"] for r in memory_after) == list(range(8)), "Missing memory ranks")
            metrics["memory"] = {"before_reset": memory_before, "after": memory_after}
        write(
            root / f"{name}-spec-metrics.json",
            {"before": metrics_before, "after": metrics_after, "delta": metric_delta},
        )
        write(root / f"{name}-metrics.json", metrics)
        if number:
            rounds.append(metrics)
    return rounds


def validate_binaries(ranks, preflight):
    def fingerprint(row):
        require(not row["artifact_truncated"] and row["artifacts"], "Missing/truncated binary identities")
        return {r["path"]: r["sha256"] for r in row["artifacts"]}

    expected = fingerprint(preflight)
    actual = [fingerprint(r["binaries"]) for r in ranks]
    require(actual and all(a == actual[0] for a in actual), "Rank OPP/extension identities differ")
    require(all(actual[0].get(k) == v for k, v in expected.items()), "OPP changed after preflight")
    return actual[0]


def model_run(args):
    suite.source_gate(args)
    root = args.output_dir / "runs" / case_name(args)
    root.mkdir(parents=True, exist_ok=False)
    preflight = read(args.output_dir / "preflight.json")
    path = args.output_dir / f"assets/b{args.batch}.jsonl"
    require(
        preflight["plugin_sha"] == args.plugin_sha
        and preflight["plan"] == experiment_plan(args)
        and formal.sha(path) == preflight["inputs_sha256"][str(args.batch)],
        "Preflight changed",
    )
    table, _ = table_at(args.output_dir / f"assets/b{args.batch}", args.batch)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    parsed, kwargs = case_config(args, root)
    from tools.dspark.performance_stream import StreamingEngine

    result = {"status": "FAILED", "error": None, "cleanup_error": None, "rounds": []}
    engine = None
    try:
        started = time.monotonic()
        engine = StreamingEngine(kwargs, parsed)
        result["initialization_and_capture_seconds"] = time.monotonic() - started
        environment = engine.collective_rpc("dspark_benchmark_performance_identity")
        write(root / "runtime-environment.json", environment)
        require(sorted(r["rank"] for r in environment) == list(range(8)), "Missing runtime identity ranks")
        for row in environment:
            expected = (
                fixed_k.runtime_expected(table["identity"], args.draft_k)
                if getattr(args, "fixed_k_comparison", False)
                else table["identity"]
            )
            require(row["identity"] == expected, "Actual model runtime differs from frozen experiment identity")
            if args.mode == "fixed":
                require(row["confidence_head_used"] is False, "Confidence runtime active in fixed mode")
        binary = validate_binaries(environment, read(args.output_dir / "runtime-environment.json"))
        reference = args.output_dir / "measurement-binaries.json"
        if reference.exists():
            require(read(reference) == binary, "Loaded OPP/extension identity changed across performance cases")
        else:
            write(reference, binary)
        write(root / "capture.json", benchmark._collect_worker_graph_runtime(engine, parsed))
        capacity = engine.collective_rpc("dspark_benchmark_capacity")
        write(root / "capacity.json", capacity)
        if getattr(args, "fixed_k_comparison", False):
            fixed_k.capacity_check(capacity, args.draft_k)
        else:
            capacity_check(capacity, args.batch)
        result["rounds"] = run_rounds(engine, parsed, args, records, root, table)
        result["status"] = "MEASURED"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if engine is not None and engine.profile_guard is not None:
            engine.profile_guard.remember(error)
    finally:
        if engine is not None:
            try:
                if engine.last_batch is not None:
                    write(root / "last-stream.json", engine.last_batch)
            except BaseException as error:
                result["evidence_error"] = f"{type(error).__name__}: {error}"
                result["error"] = result["error"] or result["evidence_error"]
                if engine.profile_guard is not None:
                    engine.profile_guard.remember(error)
            try:
                started = time.monotonic()
                engine.shutdown()
                result["shutdown_seconds"] = time.monotonic() - started
                require(engine.cleanup_result["success"], "Cleanup failed")
            except BaseException as error:
                result["cleanup_error"] = f"{type(error).__name__}: {error}"
        write(root / "generation-result.json", result)
    return int(result["status"] != "MEASURED" or result["error"] is not None or result["cleanup_error"] is not None)


def supervise(args, case):
    root = args.output_dir / "runs" / case["name"]
    root.parent.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tools.dspark.performance_comparison",
        "model",
        "--plugin-sha",
        args.plugin_sha,
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--plugin",
        str(args.plugin),
        "--core",
        str(args.core),
        "--model",
        str(args.model),
        "--batch",
        str(case["batch"]),
        "--mode",
        case["mode"],
    ]
    if getattr(args, "fixed_k_comparison", False):
        command += ["--fixed-k-comparison", "--draft-k", str(case["draft_k"])]
    guarded = [
        sys.executable,
        "-m",
        "tools.dspark.profile_process_guard",
        "--directory",
        str(root),
        "--receipt",
        str(root.parent / f"{root.name}-supervisor.json"),
        "--max-runtime-seconds",
        str(MODEL_SECONDS),
        "--stop-file",
        str(args.output_dir / "STOP"),
        "--shutdown-policy",
        shutdown_policy.POLICY_NAME,
        "--",
        *shutdown_policy.child_command(shutdown_policy.POLICY_NAME, command),
    ]
    row = {"command": guarded, "rc": None, "log_scan_rc": None}
    report = {"case": case, "valid": False, "performance_eligible": False, "error": None}
    residual = {"success": False, "error": None}
    try:
        suite.resources_idle(root.parent / f"{root.name}-npu-before.log")
        row["rc"] = suite.logged(guarded, root.parent / f"{root.name}.log")
        acceptance.scan(root.parent / f"{root.name}.log")
        row["log_scan_rc"] = 0
        require(row["rc"] == 0, "Model/supervisor failed")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if row["log_scan_rc"] is None:
            row["log_scan_rc"] = 1
    finally:
        try:
            suite.resources_idle(root.parent / f"{root.name}-npu-after.log")
            residual["success"] = True
        except Exception as error:
            residual["error"] = str(error)
        write(root.parent / f"{root.name}-command.json", row)
        write(root.parent / f"{root.name}-residual.json", residual)
    report["shutdown"] = shutdown_acceptance.check(root, shutdown_policy.POLICY_NAME)
    try:
        shutdown_acceptance.require_passive(root)
        result = read(root / "generation-result.json")
        require(
            result["status"] == "MEASURED"
            and len(result["rounds"]) == ROUNDS
            and not result["error"]
            and not result["cleanup_error"],
            "Incomplete measurements",
        )
        require(
            report["error"] is None and report["shutdown"]["shutdown_policy_evidence_valid"] and residual["success"],
            "Exit/log/resource checks failed",
        )
        report.update(valid=True, performance_eligible=True, generation=result)
    except Exception as error:
        report["error"] = report["error"] or str(error)
    write(root.parent / f"{root.name}-report.json", report)
    return report


def summarize(root, cases):
    comparisons = []
    for batch in (64, 128, 256):
        pair = [r for r in cases if r["case"]["batch"] == batch and r["valid"]]
        if len(pair) != 2:
            continue
        fixed, confidence = pair
        modes = {}
        for report in pair:
            rows = report["generation"]["rounds"]
            modes[report["case"]["mode"]] = {
                "throughput": distribution([r["output_tokens_per_second"] for r in rows]),
                "rounds": rows,
            }
        differences = []
        for i in range(1, ROUNDS + 1):
            a = read(root / f"runs/b{batch}-fixed/round-{i}-stream.json")["requests"]
            b = read(root / f"runs/b{batch}-confidence/round-{i}-stream.json")["requests"]
            differences.append(
                {
                    "round": i,
                    "different_output_sequences": sum(
                        x["output_token_ids"] != y["output_token_ids"] for x, y in zip(a, b)
                    ),
                    "fixed_tokens": sum(len(x["output_token_ids"]) for x in a),
                    "confidence_tokens": sum(len(x["output_token_ids"]) for x in b),
                }
            )
        comparisons.append(
            {
                "batch": batch,
                "modes": modes,
                "output_differences": differences,
                "speedup": modes["confidence"]["throughput"]["median"] / modes["fixed"]["throughput"]["median"],
                "speedup_formula": (
                    "median(confidence actual-output tokens/s) / median(fixed K5 actual-output tokens/s)"
                ),
                "comparability_warning": "OUTPUTS_DIFFER: speedup is not quality-equivalent"
                if any(r["different_output_sequences"] for r in differences)
                else None,
                "quality_equivalence": "NOT_EVALUATED",
                "order_bias": "fixed then confidence in separate engines; order is not randomized",
            }
        )
    return {
        "all_six_valid": len(cases) == 6 and all(r["valid"] for r in cases),
        "cases": cases,
        "comparisons": comparisons,
    }


def run(args):
    contract = experiment_plan(args)
    print(json.dumps(contract, indent=2), flush=True)
    write(args.output_dir / "performance-plan.json", contract)
    stages = []
    reports = []
    error = None
    try:
        for label, seconds, cmd in (
            (
                "host",
                600,
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--noconftest",
                    "-q",
                    "-ra",
                    "tests/ut/test_dspark_fixed_k.py"
                    if getattr(args, "fixed_k_comparison", False)
                    else "tests/ut/test_dspark_performance_comparison.py",
                    *(
                        [
                            "tests/ut/spec_decode/test_dspark_v2_markov_sampling.py::test_fixed_k8_installed_proposal_and_recurrence",
                            "tests/ut/spec_decode/test_dspark_v2_model_loading.py::test_modelslim_loader_fixed_k8_opt_in_preserves_checkpoint_contract",
                            "tests/ut/spec_decode/test_dspark_v2_model_loading.py::test_fixed_k8_real_vllm_config_keeps_runtime_and_checkpoint_distinct",
                        ]
                        if getattr(args, "fixed_k_comparison", False)
                        else []
                    ),
                    "--junitxml",
                    str(args.output_dir / "host.xml"),
                ],
            ),
            (
                "prepare",
                PREPARE_SECONDS,
                [sys.executable, "-m", "tools.dspark.performance_comparison", "prepare", *sys.argv[2:]],
            ),
        ):
            started = time.monotonic()
            rc = suite.logged(
                ["timeout", "--signal=TERM", "--kill-after=15s", str(seconds), *cmd], args.output_dir / f"{label}.log"
            )
            stages.append({"name": label, "rc": rc, "elapsed_seconds": time.monotonic() - started})
            require(rc == 0, f"{label} failed")
            if label == "host":
                tests = ET.parse(args.output_dir / "host.xml").getroot().findall(".//testcase")
                require(
                    tests and not any(t.find(k) is not None for t in tests for k in ("failure", "error", "skipped")),
                    "Host preflight requires zero skips",
                )
        for case in contract["cases"]:
            report = supervise(args, case)
            reports.append(report)
            require(report["valid"], f"{case['name']} failed: {report['error']}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            summary = (
                fixed_k.summarize(args.output_dir, reports, distribution)
                if getattr(args, "fixed_k_comparison", False)
                else summarize(args.output_dir, reports)
            )
        except Exception as exc:
            summary = {"all_six_valid": False, "cases": reports, "summary_error": str(exc)}
            error = error or f"Summary failed: {exc}"
        result = {**summary, "stages": stages, "error": error}
        write(args.output_dir / "performance-summary.json", result)
    return int(
        error is not None
        or not result.get("all_two_valid" if getattr(args, "fixed_k_comparison", False) else "all_six_valid", False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "model"))
    parser.add_argument("--plugin-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, default=Path("/workspace/vllm-ascend-hust"))
    parser.add_argument("--core", type=Path, default=Path("/workspace/vllm-hust"))
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument("--batch", type=int, choices=(64, 128, 256))
    parser.add_argument("--mode", choices=("fixed", "confidence"))
    parser.add_argument("--fixed-k-comparison", action="store_true")
    parser.add_argument("--draft-k", type=int, choices=(5, 8), default=5)
    args = parser.parse_args()
    if args.draft_k != 5 and not args.fixed_k_comparison:
        parser.error("K8 is only available in the explicit fixed-K experiment")
    if args.action == "prepare":
        prepare(args)
        return 0
    return model_run(args) if args.action == "model" else run(args)


if __name__ == "__main__":
    sys.exit(main())
