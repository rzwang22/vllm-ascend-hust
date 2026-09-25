# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded B64 calibration and post-exit publication; never performance evidence."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
import tempfile
from pathlib import Path

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import functional_coverage as coverage
from tools.dspark import shutdown_acceptance, swa_acceptance
from tools.dspark import startup_cost_profile as profile
from tools.dspark.prepare_performance_data import read_manifest
from tools.dspark.verification_tools import checkpoint_preflight

NAME = "b64-confidence-cost-v1"
REQUEST_GRID = (1, 6, 12, 24, 48, 64)
RUNTIME_SECONDS = 7200
ACCEPTED_PLUGIN = "fe6b29be454ea6eec4e37f4c2989a9c5440f951f"
ACCEPTED_SHA = "3ddf1943036b083a994e97ef106f6d1e8d57ac7d34171d82bc083bcd74c7b848"


def captures(batch=64):
    if batch not in (64, 128, 256):
        raise ValueError("Unsupported bounded engine tier")
    return [6 * 2**i for i in range(batch.bit_length())]


def request_grid(batch=64):
    return sorted({1, batch, *(c for c in captures(batch) if c <= batch)})


def plan(batch=64):
    _, matrix = profile.grid(batch, captures(batch), [128], 512)
    points = [p for p in matrix if p["requests"] in request_grid(batch)]
    return {
        "name": f"b{batch}-confidence-cost-v1",
        "purpose": "NPU cost calibration only; not synthetic functional phase3 or performance",
        "performance_eligible": False,
        "model_initializations": 1,
        "request_grid": request_grid(batch),
        "context_ceilings": [640],
        "capture_sizes": captures(batch),
        "points": points,
        "point_count": len(points),
        "total_requests": sum(p["requests"] for p in points),
        "total_output_tokens": sum(p["requests"] * 512 for p in points),
        "warmup": 2,
        "samples": 5,
        "max_runtime_seconds": RUNTIME_SECONDS,
        "future_workload": {
            "engine_max_num_seqs": batch,
            "actual_decode_requests": [1, batch],
            "prompt_tokens_max": 116,
            "max_new_tokens": 256,
            "scheduler_pre_query_upper_bound_max": 640,
            "mode": "confidence",
            "calibration": "uncalibrated unless separately verified",
            "real_text_validation": "NOT_RUN",
        },
        "estimate": (
            "ceil requests/context; max rank/layout medians and monotone envelope, not a proven worst-case bound"
        ),
        "out_of_range": "fail; no extrapolation, padding fake samples or eager fallback",
    }


def validate_args(args):
    if not getattr(args, "formal_cost_plan", None):
        return
    batch = getattr(args, "batch", getattr(args, "batches", [64])[0])
    if (
        args.formal_cost_plan != plan(batch)["name"]
        or args.stage != "profile"
        or getattr(args, "batch", batch) != batch
        or getattr(args, "batches", [batch]) != [batch]
        or args.profile_contexts != [128]
        or args.profile_output_tokens != 512
        or args.profile_warmup != 2
        or args.profile_samples != 5
        or (getattr(args, "capture_sizes", None) or getattr(args, "capture", None)) != captures(batch)
        or args.max_model_len != 8192
        or args.max_num_batched_tokens != 8192
        or args.gpu_memory_utilization != 0.9
        or str(args.model) != "/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"
        or args.profile_shutdown_policy != "dspark-profile-25s-v1"
        or not args.profile_worker_exit
        or any(
            getattr(args, key, None)
            for key in (
                "profile_experiment",
                "profile_coverage_phase",
                "profile_nan_diagnostic",
                "profile_target_layer",
                "profile_target_attention",
                "profile_operator_capture",
                "profile_write_timeline",
                "profile_exit_observation",
                "profile_exit_no_debugger",
            )
        )
    ):
        raise ValueError(
            "Formal costs require the frozen tier plan and named exit receipts, without numerical diagnostics"
        )


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def audit_baseline(path):
    if sha(path) != ACCEPTED_SHA:
        raise ValueError("Phase2 baseline archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="dspark-phase2-accepted-") as directory:
        root = Path(directory).resolve()
        with tarfile.open(path) as archive:
            for member in archive:
                target = root / member.name
                if not target.resolve().is_relative_to(root):
                    raise ValueError("Unsafe archive member")
                if member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        model = root / "dspark-large-batch.v8vohAeE"
        result = swa_acceptance.model_report(model / "runs/b64", 0)
        if (
            result != read(model / "model-acceptance.json")
            or not result["overall_pass"]
            or read(model / "runs/b64/plan.json")["plugin_sha"] != ACCEPTED_PLUGIN
        ):
            raise ValueError("Accepted phase2 report could not be reconstructed")
        return {
            "archive_sha256": ACCEPTED_SHA,
            "plugin": ACCEPTED_PLUGIN,
            "status": "SYNTHETIC_FUNCTIONAL_ACCEPTANCE_FROZEN",
            "confidence_validation": "NOT_RUN",
            "checkpoint": read(model / "runs/b64/checkpoint.json"),
        }


def weight_identity(model):
    """Hash all loader-visible shards outside model execution, not just the head."""
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError("No model weight shards")
    files = []
    for path in shards:
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Weights changed while hashing")
        files.append({"name": path.name, "bytes": after.st_size, "sha256": digest})
    return {
        "model": str(model.resolve()),
        "shards": files,
        "metadata_sha256": {p.name: sha(p) for p in sorted(model.glob("*.json"))},
        "checkpoint": checkpoint_preflight(model),
        "scope": "all top-level safetensors, index/config and confidence bytes",
    }


def real_text_contract(manifest):
    data, rows, _ = read_manifest(manifest, 64)
    if len({r["prompt_token_sha256"] for r in rows}) != 64:
        raise ValueError("Future real-text pilot requires 64 distinct frozen prompts")
    return {
        "status": "PLANNED_NOT_RUN",
        "manifest_sha256": sha(manifest),
        "source_snapshot_sha256": data["source_snapshot_sha256"],
        "tokenizer_revision": data["tokenizer_revision"],
        "tokenizer_files_sha256": data["tokenizer_files_sha256"],
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 0, "max_tokens": 256, "ignore_eos": False},
        "engine_max_num_seqs": 64,
        "client_outstanding": 64,
        "records": [
            {
                "request_id": r["request_instance_id"],
                "source_case_id": r["raw_task"]["case_id"],
                "record_sha256": r["record_sha256"],
                "prompt_token_sha256": r["prompt_token_sha256"],
                "prompt_token_count": r["prompt_token_count"],
                "source_repo": r["raw_task"]["source_repo"],
                "source_revision": r["raw_task"]["source_revision"],
                "source_split": r["raw_task"]["source_split"],
            }
            for r in rows
        ],
    }


CONFIDENCE_BASELINE_SHA = "a76a927b2d5f354f9f8e7f6e223fe15722091f35733c213aa2616d6f6e27e118"


def workload_contract(batch=64):
    base = read(Path(__file__).with_name("B64_REAL_TEXT_PLAN.json"))
    if batch == 64:
        return base
    captures(batch)  # reject unsupported tiers
    return {
        **base,
        "engine_max_num_seqs": batch,
        "client_outstanding": batch,
        "distinct_questions": 64,
        "instances_per_question": batch // 64,
        "records": [
            {
                **row,
                "original_request_id": row["request_id"],
                "request_id": f"{row['request_id']}:b{batch}:instance{replica}",
            }
            for replica in range(batch // 64)
            for row in base["records"]
        ],
    }


def prepare(model, archive, output, plugin_sha, manifest):
    print(json.dumps(plan(), indent=2), flush=True)  # Before hashing or loading any model.
    workload = real_text_contract(manifest)
    if workload != read(Path(__file__).with_name("B64_REAL_TEXT_PLAN.json")):
        raise ValueError("Future workload differs from the frozen 64-request source contract")
    baseline = audit_baseline(archive)
    weights = weight_identity(model)
    if workload["tokenizer_revision"] != profile.suite.MODEL_REVISION or any(
        weights["metadata_sha256"].get(name) != digest for name, digest in workload["tokenizer_files_sha256"].items()
    ):
        raise ValueError("Frozen real-text tokenizer differs from model files")
    for key in ("config_sha256", "index_sha256", "checkpoint_weight_sha256"):
        if weights["checkpoint"][key] != baseline["checkpoint"][key]:
            raise ValueError(f"Model provenance differs from accepted phase2: {key}")
    result = {
        "plan": plan(),
        "baseline": baseline,
        "plugin_sha": plugin_sha,
        "core_sha": profile.suite.CORE_SHA,
        "weights": weights,
        "future_workload": workload,
    }
    benchmark._atomic_write_json(output, result)
    return result


def validate_identity(identity, batch=64):
    expected = {
        "max_num_seqs": batch,
        "max_num_batched_tokens": 8192,
        "max_model_len": 8192,
        "capture_sizes": captures(batch),
        "tp": 8,
        "ep": True,
        "K": 5,
        "target_mode": "FULL_DECODE_ONLY",
        "draft_mode": "eager",
        "gpu_memory_utilization": 0.9,
        "revision": profile.suite.MODEL_REVISION,
        "dtype": "torch.bfloat16",
        "quantization": "ascend",
        "block_size": 32,
    }
    if identity.get("diagnostic_only") or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Incompatible or diagnostic timing identity")
    for key in (
        "hardware",
        "torch_version",
        "torch_npu_version",
        "confidence_weights_sha256",
        "model",
        "revision",
        "hf_config",
        "dtype",
        "quantization",
        "cost_context_semantics",
    ):
        if not identity.get(key):
            raise ValueError(f"Missing runtime identity: {key}")


def publish(root, raw_rc, plugin_sha, batch=64):
    """Rebuild from raw records only after parent log/resource/exit gates complete."""
    target = root / "cost-profile.json"
    if target.exists():
        raise ValueError("Refuse to overwrite a published cost table")
    report = {"status": "FAILED", "cost_table_usable": False, "performance_eligible": False, "raw_rc": raw_rc}
    try:
        from vllm_ascend.spec_decode.dspark_verification import CostTable

        if raw_rc != 0:
            raise ValueError("Generation/log scan failed")
        status = shutdown_acceptance.check(root, "dspark-profile-25s-v1")
        if not status["shutdown_policy_evidence_valid"]:
            raise ValueError(f"Shutdown acceptance failed: {status['shutdown_policy_errors']}")
        cleanup, workers = read(root / "cleanup.json"), read(root / "worker-cleanup.json")
        if (
            cleanup.get("timed_out") is not False
            or cleanup.get("forced_cleanup") is not False
            or workers.get("forced_cleanup") is not False
            or workers.get("force_events") != []
            or len(workers.get("workers", [])) != 8
            or sorted((w["rank"], w["raw_exitcode"]) for w in workers["workers"]) != [(r, 0) for r in range(8)]
        ):
            raise ValueError("Workers did not all exit naturally")
        if batch > 64:
            from tools.dspark.batch_expansion import capacity_check

            shutdown_acceptance.require_passive(root)
            capacity_check(read(root / "capacity.json"), batch)
        saved_plan = read(root / "plan.json")
        if (
            saved_plan.get("formal_cost") != plan(batch)
            or saved_plan.get("points") != plan(batch)["points"]
            or saved_plan.get("plugin_sha") != plugin_sha
            or saved_plan.get("core_sha") != profile.suite.CORE_SHA
        ):
            raise ValueError("Formal plan/source mismatch")
        lifecycle = read(root / "lifecycle.json")
        if lifecycle.get("engine_initializations") != 1 or lifecycle.get("shutdown") is not True:
            raise ValueError("Incomplete model lifecycle")
        provenance = read(root.parent.parent / "formal-cost-preflight.json")
        if (
            provenance["plan"] != plan(batch)
            or provenance["plugin_sha"] != plugin_sha
            or provenance["core_sha"] != profile.suite.CORE_SHA
            or provenance["baseline"]["archive_sha256"] != (ACCEPTED_SHA if batch == 64 else CONFIDENCE_BASELINE_SHA)
        ):
            raise ValueError("Preflight provenance mismatch")
        if provenance.get("future_workload") != workload_contract(batch):
            raise ValueError("Future workload contract changed")
        if weight_identity(Path(provenance["weights"]["model"])) != provenance["weights"]:
            raise ValueError("Checkpoint bytes changed during collection")
        if read(root / "checkpoint.json") != provenance["weights"]["checkpoint"]:
            raise ValueError("Checkpoint preflight changed")
        capture = read(root / "capture.json")
        if (
            capture.get("configured_capture_sizes") != captures(batch)
            or capture.get("observed_capture_sizes") != captures(batch)
            or capture.get("npugraph_ex_enabled") is not True
            or len(capture.get("workers", [])) != 8
            or {w["rank"] for w in capture["workers"]} != set(range(8))
            or any(
                w.get("target_cudagraph_mode") != "FULL_DECODE_ONLY"
                or w.get("dspark_cudagraph_mode") != "NONE"
                or w.get("observed_capture_sizes") != captures(batch)
                for w in capture["workers"]
            )
        ):
            raise ValueError("Actual target/draft Graph capture configuration mismatch")
        saved = read(root / "retained.json")
        if [r["point"] for r in saved] != plan(batch)["points"]:
            raise ValueError("Missing/duplicate calibration points")
        rebuilt, identity, loaded_weights = [], None, None
        for record in saved:
            point = record["point"]
            path = root / (point["id"] + ".json")
            if sha(path) != record["raw_sha256"]:
                raise ValueError("Raw sample hash mismatch")
            raw = read(path)
            if not raw.get("request_identity_validation") or not coverage.requests_complete(point, raw["streaming"]):
                raise ValueError("Incomplete requests or mapping")
            selected = profile.point_samples(point, raw["ranks"], 2, 5, 8)
            if selected != record["retained"]:
                raise ValueError("Retained samples differ from raw reconstruction")
            for rank in raw["ranks"]:
                current = rank["cost_profile"]["identity"]
                validate_identity(current, batch)
                if current["model"] != provenance["weights"]["model"]:
                    raise ValueError("Runtime model path differs from hashed weights")
                weights = rank["confidence_verification"]["weights"]
                if (
                    rank["cost_profile"].get("observation") is not None
                    or not weights.get("loaded_parameters")
                    or weights.get("weights_sha256") != current["confidence_weights_sha256"]
                ):
                    raise ValueError("Diagnostic observation or unloaded confidence head")
                if identity is not None and (current != identity or weights != loaded_weights):
                    raise ValueError("Cross-point/rank identity changed")
                identity, loaded_weights = current, weights
            rebuilt.append({**record, "retained": selected})
        candidate = read(root / "cost-profile.pending.json")
        host = read(root / "scheduler-overhead.json")
        if (
            host.get("source") != "host_allocate_prefixes_perf_counter"
            or host.get("unit") != "seconds"
            or len(host.get("samples", [])) != 20
            or any(not math.isfinite(v) or v <= 0 for v in host["samples"])
            or statistics.median(host["samples"]) != host.get("median_seconds")
            or host["median_seconds"] != candidate["scheduler_seconds"]
        ):
            raise ValueError("CPU allocation overhead receipt mismatch")
        table = profile.compile_startup(
            rebuilt,
            identity,
            request_grid(batch),
            checkpoint=read(root / "checkpoint.json"),
            plugin_sha=plugin_sha,
            raw_hashes=[r["raw_sha256"] for r in rebuilt],
            overhead=candidate["scheduler_seconds"],
        )
        if {**table, "source": "unpublished_startup_npu_event_profile"} != candidate:
            raise ValueError("Candidate differs from recomputed NPU samples")
        costs = CostTable.load_startup(table, identity)
        lookups = 0
        for n in range(1, batch + 1):
            for tokens in range(n, 6 * n + 1):
                costs.cost(n, tokens, 640)
                lookups += 1
        table.update(
            publication={
                "status": "PASSED",
                "plan": plan(batch)["name"],
                "shutdown": status,
                "validated_candidate_lookups": lookups,
            },
            weight_provenance=provenance["weights"],
            loaded_confidence_weights=loaded_weights,
            scheduler_measurements_sha256=sha(root / "scheduler-overhead.json"),
            future_workload=provenance["future_workload"],
        )
        report.update(
            status="PASSED",
            cost_table_usable=True,
            table_sha256=hashlib.sha256(benchmark._canonical_json_bytes(table)).hexdigest(),
            identity=identity,
            points=len(rebuilt),
            validated_candidate_lookups=lookups,
        )
        # Publish the proof first and the usable filename last. An interruption
        # before the final atomic rename leaves no loadable official cost file.
        benchmark._atomic_write_json(root / "cost-publication.json", report)
        benchmark._atomic_write_json(target, table)
    except Exception as error:
        report.update(status="FAILED", cost_table_usable=False, error=f"{type(error).__name__}: {error}")
        if target.exists():
            target.unlink()
        benchmark._atomic_write_json(root / "cost-publication.json", report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("model", type=Path)
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("plugin_sha")
    p.add_argument("manifest", type=Path)
    p = sub.add_parser("publish")
    p.add_argument("root", type=Path)
    p.add_argument("raw_rc", type=int)
    p.add_argument("plugin_sha")
    p.add_argument("--batch", type=int, choices=(64, 128, 256), default=64)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.model, args.archive, args.output, args.plugin_sha, args.manifest)
    else:
        print(json.dumps(publish(args.root, args.raw_rc, args.plugin_sha, args.batch), indent=2))


if __name__ == "__main__":
    main()
