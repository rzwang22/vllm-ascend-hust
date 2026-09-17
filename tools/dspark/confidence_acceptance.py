# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One frozen GSM8K confidence run, publication gate and post-exit acceptance."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import formal_cost as formal
from tools.dspark import run_performance_suite as suite
from tools.dspark import shutdown_acceptance, shutdown_policy
from tools.dspark.graph64_checks import scan
from tools.dspark.prepare_performance_data import copy_manifest_assets, read_manifest
from tools.dspark.verification_tools import summarize_verification

PRODUCER = "89cfb54d53b1443cb7df0b4c800196435b21e5ad"
TABLE_SHA = "576188ba88fd839a00308179b91a951b3d30467b951b159954e57549f869029a"
COST_DIRECTORY = Path("/workspace/dspark-results/dspark-large-batch.8CR50Czp/runs/b64")
NAME = "b64-gsm8k-confidence-v1"
RUNTIME_SECONDS = 3600
# Only these pre-existing execution-adjacent files may differ from producer.
# The allowlist is supplemented by a frozen diff hash, not a blanket exemption.
COMPATIBILITY = Path(__file__).with_name("CONFIDENCE_CODE_COMPATIBILITY.json")


def write(path, value):
    benchmark._atomic_write_json(path, value)


def read(path):
    return formal.read(path)


def plan():
    return {
        "name": NAME,
        "mode": "confidence",
        "profile": False,
        "performance_eligible": False,
        "inputs": read(Path(__file__).with_name("B64_REAL_TEXT_PLAN.json")),
        "request_count": 64,
        "model_initializations": 1,
        "max_output_tokens": 256,
        "max_total_output_tokens": 16384,
        "natural_eos": True,
        "context_range": [0, 640],
        "actual_decode_requests": [1, 64],
        "capture_sizes": list(formal.coverage.CAPTURES),
        "max_runtime_seconds": RUNTIME_SECONDS,
        "shutdown_budget": shutdown_policy.budget(shutdown_policy.POLICY_NAME),
        "prefill": "ordinary prefill; no cost lookup or FULL requirement",
        "decode": "real confidence selection, current owner epochs, actual FULL consumption",
        "calibration": "uncalibrated",
        "cost_sha256": TABLE_SHA,
        "original_budget_acceptance": "NOT_EVALUATED",
    }


def verify_code(plugin):
    contract = read(COMPATIBILITY)
    if (
        contract.get("producer_plugin") != PRODUCER
        or contract.get("core") != suite.CORE_SHA
        or contract.get("cost_sha256") != TABLE_SHA
    ):
        raise ValueError("Code compatibility contract refers to another calibration")
    changes = subprocess.check_output(
        ["git", "-C", str(plugin), "diff", "--name-only", PRODUCER, "HEAD", "--", "vllm_ascend", "csrc"], text=True
    ).splitlines()
    if sorted(changes) != sorted(contract["changed_runtime_files"]):
        raise ValueError("Unaudited runtime changes relative to cost producer")
    for name, digest in contract["changed_runtime_files"].items():
        if formal.sha(plugin / name) != digest:
            raise ValueError(f"Audited runtime file changed: {name}")
    return contract


def publication(directory):
    table = read(directory / "cost-profile.json")
    proof = read(directory / "cost-publication.json")
    if (
        formal.sha(directory / "cost-profile.json") != TABLE_SHA
        or proof.get("table_sha256") != TABLE_SHA
        or proof.get("status") != "PASSED"
        or proof.get("cost_table_usable") is not True
        or table.get("publication", {}).get("status") != "PASSED"
        or table.get("plugin_sha") != PRODUCER
        or table.get("core_sha") != suite.CORE_SHA
        or table.get("future_workload") != plan()["inputs"]
    ):
        raise ValueError("Frozen cost publication/producer/workload mismatch")
    formal.validate_identity(table["identity"])
    return table, proof


def prepare(args):
    print(json.dumps(plan(), indent=2), flush=True)  # before weights or model load
    suite.source_gate(args)
    compatibility = verify_code(args.plugin)
    table, proof = publication(COST_DIRECTORY)
    if formal.real_text_contract(args.manifest) != plan()["inputs"]:
        raise ValueError("Frozen GSM8K manifest/IDs/tokens/source changed")
    weights = formal.weight_identity(args.model)
    if weights != table["weight_provenance"]:
        raise ValueError("Full model weights/config differ from cost producer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=False)
    for name in ("cost-profile.json", "cost-publication.json"):
        shutil.copyfile(COST_DIRECTORY / name, assets / name)
    copy_manifest_assets(args.manifest, args.output_dir / "input")
    _, records, _ = read_manifest(args.manifest, 64)
    (args.output_dir / "input.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write(
        args.output_dir / "preflight.json",
        {
            "plan": plan(),
            "plugin_sha": args.plugin_sha,
            "core_sha": suite.CORE_SHA,
            "weights": weights,
            "publication": proof,
            "code_compatibility": compatibility,
            "input_sha256": formal.sha(args.output_dir / "input.jsonl"),
        },
    )


def validate_receipts(ranks, stream, table):
    """Fail closed on every recorded selection/consumption, not histograms alone."""
    mapping = stream.get("request_id_mapping") or {}
    pairs = mapping.get("mappings", [])
    expected = [r["request_id"] for r in plan()["inputs"]["records"]]
    if (
        mapping.get("errors") != []
        or mapping.get("hook_restored") is not True
        or sorted(p["external_id"] for p in pairs) != sorted(expected)
        or len({p["internal_id"] for p in pairs}) != 64
        or sorted(r["rank"] for r in ranks) != list(range(8))
    ):
        raise ValueError("Request/rank mapping unavailable or inconsistent")
    internal = {p["internal_id"] for p in pairs}
    reference = None
    confidence_calls = 0
    for rank in ranks:
        evidence = rank.get("confidence_execution_receipts", {})
        adaptive = rank["confidence_verification"]
        if (
            evidence.get("truncated") is not False
            or not evidence.get("records")
            or rank.get("error") is not None
            or rank.get("failed_execution_count") != 0
            or adaptive.get("mode") != "confidence"
            or adaptive.get("specified_batches") != 0
            or adaptive.get("confidence_head_calls", 0) <= 0
            or adaptive.get("weights") != table["loaded_confidence_weights"]
            or adaptive.get("cost_profile", {}).get("identity") != table["identity"]
        ):
            raise ValueError("Missing genuine confidence/runtime/complete execution evidence")
        comparable = []
        observed_requests = set()
        for i, row in enumerate(evidence["records"], 1):
            target = row.get("target") or {}
            ids, queries = target.get("request_ids", []), target.get("query_lengths", [])
            if (
                row["execution"] != i
                or not 1 <= len(ids) <= 64
                or len(set(ids)) != len(ids)
                or not set(ids) <= internal
                or len(ids) != len(queries)
                or dict(zip(ids, queries)) != row["scheduled_queries"]
                or sum(queries) != target.get("valid_tokens")
                or target.get("capacity", 0) < sum(queries)
                or any(v < 0 or v > 640 for v in row["context_upper"].values())
            ):
                raise ValueError("Target execution does not match scheduled request/query layout")
            observed_requests.update(ids)
            selection = row["selection"]
            if selection:
                lengths, epochs = selection["lengths"], selection["producer_epochs"]
                accepted = row.get("accepted") or {}
                if set(lengths) != set(epochs) or not set(lengths) <= set(ids):
                    raise ValueError("Selection owner mapping mismatch")
                for key, length in lengths.items():
                    if type(length) is not int or not 0 <= length <= 5 or row["scheduled_queries"][key] != length + 1:
                        raise ValueError("Decision was not consumed as actual verification query")
                aid = accepted.get("request_ids", [])
                counts = accepted.get("num_sampled") or []
                if (
                    accepted.get("status") != "available"
                    or len(aid) != len(lengths)
                    or set(aid) != set(lengths)
                    or accepted.get("producer_epochs") != [epochs[k] for k in aid]
                    or accepted.get("verified") != [lengths[k] for k in aid]
                    or len(counts) != len(aid)
                    or any(type(v) is not int or not 1 <= v <= lengths[k] + 1 for k, v in zip(aid, counts))
                ):
                    raise ValueError("Verification/acceptance epoch or counts unavailable")
                if selection["policy"] == "current_epoch_survival_cost":
                    confidence_calls += 1
                    lookup = row.get("lookup", {})
                    cap = next((c for c in table["identity"]["capture_sizes"] if c >= sum(queries)), None)
                    cell = min(
                        (c for c in table["cells"] if c["requests"] >= len(ids) and c["capacity"] == cap),
                        key=lambda c: c["requests"],
                    )
                    seconds = cell["target_seconds"] + cell["draft_seconds"] + table["scheduler_seconds"]
                    if (
                        not target.get("full_replay")
                        or row.get("graph") != {"mode": "FULL", "capacity": cap}
                        or target["capacity"] != cap
                        or selection["selected_graph_capacity"] != cap
                        or selection["actual_tokens"] != sum(queries)
                        or lookup.get("token_budget") != sum(queries)
                        or lookup.get("requests") != len(ids)
                        or lookup.get("sampled_requests") != cell["requests"]
                        or lookup.get("context_ceiling") != 640
                        or lookup.get("estimated_seconds") != seconds
                        or selection["estimated_seconds"] != seconds
                        or selection["confidence_epochs"] != epochs
                    ):
                        raise ValueError("FULL/cost budget/epoch receipt mismatch")
                    for key in lengths:
                        score = row["confidence"].get(key, {})
                        if (
                            score.get("producer_epoch") != epochs[key]
                            or len(score.get("conditional", [])) != 5
                            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in score.get("conditional", []))
                        ):
                            raise ValueError("Stale/nonfinite confidence source")
                elif selection["policy"] != "fixed_mixed_admission":
                    raise ValueError("Unexpected specified/fallback policy")
            elif row.get("accepted"):
                raise ValueError("Acceptance without decision")
            comparable.append(
                {
                    k: row.get(k)
                    for k in (
                        "execution",
                        "selection",
                        "scheduled_queries",
                        "target",
                        "graph",
                        "accepted",
                        "lookup",
                        "context_upper",
                    )
                }
            )
        if observed_requests != internal:
            raise ValueError("Some submitted requests have no actual target execution evidence")
        if reference is not None and comparable != reference:
            raise ValueError("TP ranks disagree on decision/actual execution/acceptance")
        reference = comparable
    if not confidence_calls:
        raise ValueError("No actual confidence FULL consumption")
    per_request = []
    for pair in pairs:
        key = pair["internal_id"]
        decisions = [r for r in reference if r["selection"] and key in r["selection"]["lengths"]]
        learned = [r for r in decisions if r["selection"]["policy"] == "current_epoch_survival_cost"]
        per_request.append(
            {
                **pair,
                "target_calls": sum(key in r["target"]["request_ids"] for r in reference),
                "confidence_decisions": len(learned),
                "confidence_length_histogram": dict(Counter(r["selection"]["lengths"][key] for r in learned)),
                "verified_candidates": sum(r["selection"]["lengths"][key] for r in decisions),
                "accepted_candidates": sum(
                    r["accepted"]["num_sampled"][r["accepted"]["request_ids"].index(key)] - 1 for r in decisions
                ),
            }
        )
    return {
        "status": "PASSED_THIS_RUN",
        "per_request": per_request,
        "confidence_length_histogram": dict(
            Counter(
                v
                for r in reference
                if r["selection"] and r["selection"]["policy"] == "current_epoch_survival_cost"
                for v in r["selection"]["lengths"].values()
            )
        ),
        "logical_confidence_calls": confidence_calls // 8,
        "length_histogram": dict(
            Counter(v for r in reference if r["selection"] for v in r["selection"]["lengths"].values())
        ),
        "actual_request_count_histogram": dict(Counter(len(r["target"]["request_ids"]) for r in reference)),
        "prefill_or_admission_calls": sum(
            not r["selection"] or r["selection"]["policy"] == "fixed_mixed_admission" for r in reference
        ),
        "calibration": "uncalibrated",
        "numerical": "built-in Markov/confidence/owner contracts and strict log scan; no all-layer finiteness claim",
    }


def validate_stream(stream, records):
    rows = stream.get("requests", [])
    frozen = plan()["inputs"]["records"]
    if (
        len(rows) != 64
        or len(records) != 64
        or stream.get("error")
        or stream.get("scheduler", {}).get("corrupted_requests")
    ):
        raise ValueError("Incomplete/corrupted real text generation")
    for actual, source, contract in zip(rows, records, frozen):
        if (
            actual is None
            or actual.get("request_id") != contract["request_id"]
            or actual.get("observed_prompt_token_ids") != source["prompt_token_ids"]
            or actual.get("finish_reason") not in ("stop", "length")
            or actual.get("error") is not None
            or actual.get("completed_monotonic") is None
            or not 0 <= len(actual["output_token_ids"]) <= 256
            or (actual["finish_reason"] == "length" and len(actual["output_token_ids"]) != 256)
        ):
            raise ValueError("Frozen request/input or natural completion mismatch")


def engine_config(args, root):
    config = root / "verification.json"
    write(
        config,
        {
            "mode": "confidence",
            "profile": False,
            "cost_profile": str((args.output_dir / "assets/cost-profile.json").resolve()),
        },
    )
    local = argparse.Namespace(
        **vars(args),
        max_num_seqs=[64],
        repeats=1,
        modes=["dspark_confidence_graph"],
        capture_dspark=list(formal.coverage.CAPTURES),
        capture_target=None,
        confidence_verification=config,
        num_prompts=64,
        warmup_prompts=0,
        client_outstanding=64,
        output_len=256,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
    )
    execution_plan = suite.create_plan(local, args.output_dir / "input.jsonl", root)
    write(
        root / "plan.json",
        {
            **execution_plan,
            "contract": plan(),
            "performance_eligible": False,
            "exit_observation": False,
            "shutdown_policy": shutdown_policy.POLICY_NAME,
        },
    )
    parsed = benchmark.parse_args(execution_plan["runs"][0]["command"][2:])
    kwargs = benchmark.build_engine_kwargs(parsed)
    kwargs["additional_config"].update(
        dspark_confidence_acceptance=True,
        dspark_profile_failure_dir=str(root.resolve()),
        dspark_profile_worker_exit=True,
        dspark_profile_shutdown_policy=shutdown_policy.POLICY_NAME,
    )
    kwargs["worker_cls"] = "vllm_ascend.diagnostics.dspark_profile_worker.ProfileNPUWorker"
    kwargs["distributed_executor_backend"] = "vllm_ascend.diagnostics.dspark_profile_executor.ProfileMultiprocExecutor"
    return parsed, kwargs


def model_run(args):
    suite.source_gate(args)
    root = args.output_dir / "runs/b64"
    root.mkdir(parents=True, exist_ok=False)
    preflight = read(args.output_dir / "preflight.json")
    if (
        preflight["plan"] != plan()
        or preflight["plugin_sha"] != args.plugin_sha
        or formal.sha(args.output_dir / "input.jsonl") != preflight["input_sha256"]
    ):
        raise ValueError("Preflight/inputs changed before model initialization")
    table, _ = publication(args.output_dir / "assets")
    records = [json.loads(line) for line in (args.output_dir / "input.jsonl").read_text().splitlines()]
    parsed, kwargs = engine_config(args, root)
    from tools.dspark.performance_stream import StreamingEngine

    engine = None
    result = {"performance_eligible": False, "status": "FAILED", "error": None, "cleanup_error": None}
    try:
        engine = StreamingEngine(kwargs, parsed)
        result["engine_initializations"] = 1
        write(root / "capture.json", benchmark._collect_worker_graph_runtime(engine, parsed))
        before = engine.collective_rpc("dspark_benchmark_replay_snapshot")
        write(root / "before.json", before)
        # Runtime CostTable.load above checks the full actual runtime identity.
        for rank in before:
            if rank["confidence_verification"]["cost_profile"]["identity"] != table["identity"]:
                raise ValueError("Worker runtime identity differs from frozen costs")
        engine.generate(
            [{"prompt_token_ids": r["prompt_token_ids"]} for r in records],
            benchmark._sampling_params(parsed),
            profile_point=NAME,
            request_ids=[r["request_id"] for r in plan()["inputs"]["records"]],
        )
        after = engine.collective_rpc("dspark_benchmark_replay_snapshot")
        write(root / "after.json", after)
        write(root / "stream.json", engine.last_batch)
        validate_stream(engine.last_batch, records)
        result["execution_acceptance"] = validate_receipts(after, engine.last_batch, table)
        result["verification"] = summarize_verification(before, after, 8)
        for initial, final in zip(sorted(before, key=lambda r: r["rank"]), sorted(after, key=lambda r: r["rank"])):
            records_for_rank = final["confidence_execution_receipts"]["records"]
            verified = sum(sum(r["accepted"]["verified"]) for r in records_for_rank if r["accepted"])
            accepted = sum(sum(v - 1 for v in r["accepted"]["num_sampled"]) for r in records_for_rank if r["accepted"])
            for key, count in (("verified", verified), ("accepted", accepted)):
                if final["confidence_verification"][key] - initial["confidence_verification"][key] != count:
                    raise ValueError("Per-request receipts differ from runtime verification counters")
            full_count = sum(r["target"]["full_replay"] for r in records_for_rank)
            if sum(r["count"] for r in final["records"]) - sum(r["count"] for r in initial["records"]) != full_count:
                raise ValueError("Execution receipts omit actual FULL calls")
        result["status"] = "PASSED_THIS_RUN"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        if engine is not None and engine.profile_guard is not None:
            engine.profile_guard.remember(error)
    finally:
        if engine is not None:
            try:
                if engine.last_batch is not None:
                    write(root / "stream.json", engine.last_batch)
            except BaseException as error:
                result["evidence_error"] = f"{type(error).__name__}: {error}"
                result["error"] = result["error"] or result["evidence_error"]
                if engine.profile_guard is not None:
                    engine.profile_guard.remember(error)
            try:
                engine.shutdown()
                if not (engine.cleanup_result or {}).get("success"):
                    raise ValueError("Named cleanup failed")
            except BaseException as error:
                result["cleanup_error"] = f"{type(error).__name__}: {error}"
        write(root / "generation-result.json", result)
    return int(result["error"] is not None or result["cleanup_error"] is not None)


def supervise(args):
    root = args.output_dir / "runs/b64"
    root.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "tools.dspark.confidence_acceptance", "model", *sys.argv[2:]]
    guarded = [
        sys.executable,
        "-m",
        "tools.dspark.profile_process_guard",
        "--directory",
        str(root),
        "--receipt",
        str(root.parent / "b64-supervisor.json"),
        "--max-runtime-seconds",
        str(RUNTIME_SECONDS),
        "--stop-file",
        str(args.output_dir / "STOP"),
        "--shutdown-policy",
        shutdown_policy.POLICY_NAME,
        "--",
        *shutdown_policy.child_command(shutdown_policy.POLICY_NAME, cmd),
    ]
    row = {"command": cmd, "supervised_command": guarded, "rc": None, "log_scan_rc": None}
    report = {"overall_pass": False, "performance_eligible": False, "error": None}
    residual = {"success": False, "error": None}
    try:
        suite.resources_idle(root.parent / "b64-npu-before.log")
        row["rc"] = suite.logged(guarded, root.parent / "b64.log")
        try:
            scan(root.parent / "b64.log")
            row["log_scan_rc"] = 0
        except Exception as error:
            row.update(log_scan_rc=1, log_scan_error=str(error))
        if row["rc"] or row["log_scan_rc"]:
            raise ValueError("Model/supervisor or strict log scan failed")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            suite.resources_idle(root.parent / "b64-npu-after.log")
            residual["success"] = True
        except Exception as error:
            residual["error"] = str(error)
        write(root.parent / "b64-command.json", row)
        write(root.parent / "b64-residual.json", residual)
    report["shutdown"] = shutdown_acceptance.check(root, shutdown_policy.POLICY_NAME)
    try:
        result = read(root / "generation-result.json")
        report["generation"] = result
        workers, cleanup = read(root / "worker-cleanup.json"), read(root / "cleanup.json")
        if (
            workers.get("forced_cleanup") is not False
            or workers.get("force_events") != []
            or cleanup.get("forced_cleanup") is not False
            or cleanup.get("timed_out") is not False
        ):
            raise ValueError("Forced or timed-out cleanup cannot pass")
        if (
            result["status"] != "PASSED_THIS_RUN"
            or result["error"]
            or result["cleanup_error"]
            or not report["shutdown"]["shutdown_policy_evidence_valid"]
            or report["error"]
        ):
            raise ValueError("Generation/closed-loop/natural exit acceptance failed")
        report["overall_pass"] = True
    except Exception as error:
        report["error"] = report["error"] or str(error)
    write(args.output_dir / "confidence-acceptance.json", report)
    return int(not report["overall_pass"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "model", "run"))
    parser.add_argument("--plugin", type=Path, default=Path("/workspace/vllm-ascend-hust"))
    parser.add_argument("--core", type=Path, default=Path("/workspace/vllm-hust"))
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument("--plugin-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
        return 0
    return model_run(args) if args.action == "model" else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
