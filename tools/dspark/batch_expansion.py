# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B128 then B256, fresh measured costs then genuine confidence, stop on first failure."""

import argparse
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.dspark import confidence_acceptance as confidence
from tools.dspark import formal_cost as formal
from tools.dspark.audit_formal_cost import extract, require

BASELINE_PLUGIN = "082c8ea67b32ed1c71a408de93c53abacefb5af3"
BASELINE_INNER_SHA = "08581ac6689389879c3dce293794914136694eb9ec9983cc2d5fe5ae3ae9c573"
BASELINE = Path("/workspace/dspark-results/dspark-confidence-acceptance.bEz6qbG5-evidence.tar.gz")
PREPARE_SECONDS = 1800
HOST_SECONDS = 600
COST_SECONDS = 7400  # inner engine 7200 + 48 supervised cleanup, wrapper margin
CONFIDENCE_SECONDS = 3800  # inner engine 3600 + 48 supervised cleanup
KILL_MARGIN_SECONDS = 15
GROUP_SECONDS = 36000
GROUP_KILL_SECONDS = 65


def plan():
    tiers = []
    for batch in (128, 256):
        tiers.append(
            {
                "batch": batch,
                "cost": formal.plan(batch),
                "confidence": confidence.plan(batch),
                "distinct_questions": 64,
                "instances": batch,
                "model_initializations": 2,
                "stage_deadlines_seconds": {
                    "cost_prepare": PREPARE_SECONDS,
                    "cost": COST_SECONDS,
                    "cost_publish": PREPARE_SECONDS,
                    "confidence_prepare": PREPARE_SECONDS,
                    "confidence": CONFIDENCE_SECONDS,
                },
                "tier_upper_seconds": 3 * PREPARE_SECONDS + COST_SECONDS + CONFIDENCE_SECONDS + 5 * KILL_MARGIN_SECONDS,
                "capacity_constraints": {
                    "max_num_seqs": batch,
                    "max_target_query_tokens": 6 * batch,
                    "draft_candidate_rows": 5 * batch,
                    "max_num_batched_tokens": 8192,
                    "max_model_len": 8192,
                    "gpu_memory_utilization": 0.9,
                    "kv_context_ceiling": 640,
                    "prefix_caching": False,
                    "memory": (
                        "same weights; weights + graph pools + KV + workspace must fit "
                        "90% device memory; no capacity fallback"
                    ),
                    "coverage": "every rank must consume real confidence FULL with exactly the tier request count",
                },
            }
        )
    return {
        "name": "b128-b256-confidence-expansion-v1",
        "performance_eligible": False,
        "order": [128, 256],
        "model_initializations": 4,
        "tiers": tiers,
        "host_seconds": HOST_SECONDS,
        "baseline_audit_seconds": PREPARE_SECONDS,
        "group_runtime_limit_seconds": GROUP_SECONDS,
        "group_kill_margin_seconds": GROUP_KILL_SECONDS,
        "stage_budget_sum_seconds": HOST_SECONDS
        + PREPARE_SECONDS
        + 2 * KILL_MARGIN_SECONDS
        + sum(t["tier_upper_seconds"] for t in tiers),
        "stop": "first error; no retry, no later tier; preserve completed evidence",
    }


def audit_baseline(archive):
    require(formal.sha(archive) == formal.CONFIDENCE_BASELINE_SHA, "B64 outer archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="dspark-confidence-frozen-") as tmp:
        base = Path(tmp).resolve()
        extract(archive, base)
        outer = base / "dspark-confidence-acceptance.bEz6qbG5"
        require(formal.sha(outer / "model-evidence.tar.gz") == BASELINE_INNER_SHA, "B64 inner hash mismatch")
        extract(outer / "model-evidence.tar.gz", base)
        model = base / "dspark-large-batch.JcClwldH"
        root = model / "runs/b64"
        preflight = formal.read(model / "preflight.json")
        require(
            preflight["plugin_sha"] == BASELINE_PLUGIN and preflight["core_sha"] == confidence.suite.CORE_SHA,
            "B64 source mismatch",
        )
        require(preflight["plan"] == confidence.plan(), "B64 frozen plan changed")
        require(formal.sha(model / "input.jsonl") == preflight["input_sha256"], "B64 inputs hash mismatch")
        require(
            formal.real_text_contract(model / "input/manifest.json") == formal.workload_contract(),
            "B64 original source/token contract changed",
        )
        table, proof = confidence.publication(model / "assets")
        require(
            proof == preflight["publication"] and table["weight_provenance"] == preflight["weights"],
            "B64 weights/publication changed",
        )
        inputs = [json.loads(line) for line in (model / "input.jsonl").read_text().splitlines()]
        _, original_inputs, _ = formal.read_manifest(model / "input/manifest.json", 64)
        require(inputs == original_inputs, "B64 actual inputs differ from original frozen records")
        stream, after = formal.read(root / "stream.json"), formal.read(root / "after.json")
        confidence.validate_stream(stream, inputs)
        rebuilt = confidence.validate_receipts(after, stream, table)
        report = formal.read(model / "confidence-acceptance.json")
        verification = confidence.summarize_verification(formal.read(root / "before.json"), after, 8)
        require(
            json.loads(json.dumps(verification)) == report["generation"]["verification"],
            "B64 verification counters differ",
        )
        require(
            json.loads(json.dumps(rebuilt)) == report["generation"]["execution_acceptance"],
            "B64 per-rank execution differs",
        )
        require(
            confidence.shutdown_acceptance.check(root, confidence.shutdown_policy.POLICY_NAME) == report["shutdown"],
            "B64 shutdown differs",
        )
        require(report["overall_pass"] and report["shutdown"]["shutdown_policy_evidence_valid"], "B64 failed")
        for directory in (outer, model):
            for path in directory.rglob("*.pipestatus"):
                require(
                    path.read_text().split() and all(int(v) == 0 for v in path.read_text().split()),
                    f"Nonzero PIPESTATUS: {path.name}",
                )
        workers = formal.read(root / "worker-cleanup.json")
        cleanup = formal.read(root / "cleanup.json")
        exits = sorted((w["rank"], w["raw_exitcode"]) for w in workers["workers"])
        require(
            exits == [(r, 0) for r in range(8)]
            and workers["force_events"] == []
            and workers["forced_cleanup"] is False
            and cleanup["forced_cleanup"] is False
            and cleanup["timed_out"] is False,
            "B64 natural-exit evidence failed",
        )
        confidence.scan(model / "runs/b64.log")
        return {
            "archive_sha256": formal.CONFIDENCE_BASELINE_SHA,
            "inner_sha256": BASELINE_INNER_SHA,
            "plugin": BASELINE_PLUGIN,
            "core": confidence.suite.CORE_SHA,
            "status": "B64_CONFIDENCE_FROZEN",
            "requests": 64,
            "worker_exitcodes": exits,
            "confidence_FULL": rebuilt["logical_confidence_calls"],
            "mixed_length_FULL": report["generation"]["verification"]["mixed_length_full_replays"],
            "weights": preflight["weights"],
            "cost_sha256": confidence.TABLE_SHA,
            "source": formal.read(model / "core-source.json"),
            "original_5s_budget": "NOT_EVALUATED",
            "native_destructor_cause": "UNKNOWN",
        }


def capacity_check(rows, batch):
    require(sorted(r["rank"] for r in rows) == list(range(8)), "Missing capacity ranks")
    for r in rows:
        require(r["max_requests"] == batch and r["max_tokens"] >= 6 * batch, "Runner buffer capacity mismatch")
        require(r["draft_max_requests"] == batch, "Draft buffer capacity mismatch")
        require(r["kv_num_blocks"] > 0 and r["kv_bytes"] > 0 and r["groups"], "Missing allocated KV evidence")
        require(r["capture_sizes"] == formal.captures(batch), "Actual captured capacity mismatch")
    return {
        "status": "ALLOCATED_NOT_CONCURRENCY_PROOF",
        "ranks": rows,
        "memory_scope": (
            "KV descriptor bytes; graph/workspace fit checked by real initialization; "
            "device totals retained in npu-smi logs"
        ),
    }


def cost_prepare(args):
    confidence.suite.source_gate(args)
    baseline = formal.read(args.output_dir.parent.parent / "baseline.json")
    require(baseline["archive_sha256"] == formal.CONFIDENCE_BASELINE_SHA, "Missing frozen B64 proof")
    require(formal.real_text_contract(args.manifest) == formal.workload_contract(), "Frozen source changed")
    weights = formal.weight_identity(args.model)
    require(weights == baseline["weights"], "Weights differ from accepted B64")
    confidence.write(
        args.output_dir / "formal-cost-preflight.json",
        {
            "plan": formal.plan(args.batch),
            "baseline": baseline,
            "weights": weights,
            "plugin_sha": args.plugin_sha,
            "core_sha": confidence.suite.CORE_SHA,
            "future_workload": formal.workload_contract(args.batch),
        },
    )


def cost_command(args, batch, directory):
    return [
        sys.executable,
        "-m",
        "tools.dspark.run_large_batch",
        "--stage",
        "profile",
        "--batches",
        str(batch),
        "--plugin",
        str(args.plugin),
        "--core",
        str(args.core),
        "--model",
        str(args.model),
        "--plugin-sha",
        args.plugin_sha,
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(directory / "runs"),
        "--num-prompts",
        "64",
        "--capture-sizes",
        *map(str, formal.captures(batch)),
        "--formal-cost-plan",
        formal.plan(batch)["name"],
        "--profile-contexts",
        "128",
        "--profile-output-tokens",
        "512",
        "--profile-warmup",
        "2",
        "--profile-samples",
        "5",
        "--profile-worker-exit",
        "--profile-shutdown-policy",
        confidence.shutdown_policy.POLICY_NAME,
    ]


def run(args):
    """Run bounded child transactions; only publish/advance after their actual success."""
    contract = plan()
    print(json.dumps(contract, indent=2), flush=True)
    confidence.write(args.output_dir / "expansion-plan.json", contract)
    report = {"performance_eligible": False, "tiers": {}, "overall_pass": False, "first_error": None, "stages": []}

    def stage(name, command, seconds):
        guarded = [
            "timeout",
            "--signal=TERM",
            f"--kill-after={KILL_MARGIN_SECONDS}s",
            f"{seconds}s",
            *map(str, command),
        ]
        started = time.monotonic()
        rc = confidence.suite.logged(guarded, args.output_dir / f"{name}.log")
        report["stages"].append(
            {"name": name, "command": guarded, "rc": rc, "elapsed_seconds": time.monotonic() - started}
        )
        confidence.write(args.output_dir / "expansion-report.json", report)
        if rc:
            raise RuntimeError(f"{name} failed with exit {rc}; subsequent stages not started")

    common = [
        "--plugin",
        str(args.plugin),
        "--core",
        str(args.core),
        "--model",
        str(args.model),
        "--plugin-sha",
        args.plugin_sha,
        "--manifest",
        str(args.manifest),
    ]
    try:
        confidence.suite.source_gate(args)
        junit = args.output_dir / "host.xml"
        stage(
            "host",
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                "-q",
                "-ra",
                "tests/ut/test_dspark_batch_expansion.py",
                "tests/ut/test_dspark_confidence_acceptance.py",
                "tests/ut/test_dspark_formal_cost.py",
                "--junitxml",
                str(junit),
            ],
            HOST_SECONDS,
        )
        cases = ET.parse(junit).getroot().findall(".//testcase")
        require(
            cases and not any(c.find(k) is not None for c in cases for k in ("failure", "error", "skipped")),
            "Host preflight failed/skipped",
        )
        stage(
            "baseline",
            [
                sys.executable,
                "-m",
                "tools.dspark.batch_expansion",
                "audit",
                *common,
                "--output-dir",
                str(args.output_dir),
                "--baseline",
                str(args.baseline),
            ],
            PREPARE_SECONDS,
        )
        for batch in (128, 256):
            tier = args.output_dir / f"b{batch}"
            cost, text = tier / "cost", tier / "confidence"
            cost.mkdir(parents=True, exist_ok=False)
            report["tiers"][str(batch)] = {
                "status": "RUNNING",
                "cost_status": "PENDING",
                "confidence_status": "PENDING",
            }
            tier_args = [*common, "--batch", str(batch)]
            stage(
                f"b{batch}-cost-prepare",
                [
                    sys.executable,
                    "-m",
                    "tools.dspark.batch_expansion",
                    "prepare",
                    *tier_args,
                    "--output-dir",
                    str(cost),
                ],
                PREPARE_SECONDS,
            )
            stage(f"b{batch}-cost", cost_command(args, batch, cost), COST_SECONDS)
            stage(
                f"b{batch}-publish",
                [
                    sys.executable,
                    "-m",
                    "tools.dspark.formal_cost",
                    "publish",
                    str(cost / f"runs/b{batch}"),
                    "0",
                    args.plugin_sha,
                    "--batch",
                    str(batch),
                ],
                PREPARE_SECONDS,
            )
            report["tiers"][str(batch)]["cost_status"] = "PASSED_THIS_RUN"
            report["tiers"][str(batch)]["cost"] = formal.read(cost / f"runs/b{batch}/cost-publication.json")
            confidence.write(args.output_dir / "expansion-report.json", report)
            stage(
                f"b{batch}-confidence-prepare",
                [
                    sys.executable,
                    "-m",
                    "tools.dspark.confidence_acceptance",
                    "prepare",
                    *tier_args,
                    "--output-dir",
                    str(text),
                    "--cost-directory",
                    str(cost / f"runs/b{batch}"),
                ],
                PREPARE_SECONDS,
            )
            stage(
                f"b{batch}-confidence",
                [
                    sys.executable,
                    "-m",
                    "tools.dspark.confidence_acceptance",
                    "run",
                    *tier_args,
                    "--output-dir",
                    str(text),
                ],
                CONFIDENCE_SECONDS,
            )
            result = formal.read(text / "confidence-acceptance.json")
            require(result["overall_pass"], "Confidence report failed")
            report["tiers"][str(batch)] = {
                "status": "PASSED_THIS_RUN",
                "cost_status": "PASSED_THIS_RUN",
                "confidence_status": "PASSED_THIS_RUN",
                "confidence": result,
                "cost": formal.read(cost / f"runs/b{batch}/cost-publication.json"),
            }
            confidence.write(args.output_dir / "expansion-report.json", report)
        report["overall_pass"] = True
    except Exception as error:
        report["first_error"] = f"{type(error).__name__}: {error}"
        for tier in report["tiers"].values():
            if tier["status"] == "RUNNING":
                tier["status"] = "FAILED"
        for batch in (128, 256):
            report["tiers"].setdefault(str(batch), {"status": "NOT_RUN"})
    finally:
        confidence.write(args.output_dir / "expansion-report.json", report)
    return int(not report["overall_pass"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "audit", "prepare", "plan"))
    p.add_argument("--plugin", type=Path, default=Path("/workspace/vllm-ascend-hust"))
    p.add_argument("--core", type=Path, default=Path("/workspace/vllm-hust"))
    p.add_argument("--model", type=Path, default=Path("/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"))
    p.add_argument("--plugin-sha", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--batch", type=int, choices=(128, 256))
    args = p.parse_args()
    if args.action == "run":
        return run(args)
    if args.action == "audit":
        confidence.write(args.output_dir / "baseline.json", audit_baseline(args.baseline))
    elif args.action == "prepare":
        cost_prepare(args)
    else:
        print(json.dumps(plan(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
