# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit B64/128/256 profile or independent performance runs; stop on failure."""

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import run_performance_suite as suite
from tools.dspark.graph64_checks import scan
from tools.dspark.prepare_performance_data import copy_manifest_assets, input_population, read_manifest

TARGET_DIAGNOSTIC_RUNTIME_SECONDS = 3600


def captures(batch, explicit=None):
    sizes = []
    n = 1
    while n < batch:
        sizes.append(6 * n)
        n *= 2
    return suite.capture_sizes("dspark_graph", batch, 6 * batch, explicit or [*sizes, 6 * batch])


def common_arguments(args):
    return [
        "--plugin",
        str(args.plugin),
        "--core",
        str(args.core),
        "--plugin-sha",
        args.plugin_sha,
        "--model",
        str(args.model),
        "--manifest",
        str(args.manifest),
        "--num-prompts",
        str(args.num_prompts),
        "--output-len",
        str(args.output_len),
        "--warmup-prompts",
        str(args.warmup_prompts),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]


def command(args, batch, root):
    sizes = captures(batch, args.capture_sizes)
    if max(sizes) > args.max_num_batched_tokens:
        raise ValueError("Capture token capacity exceeds token budget")
    if args.client_outstanding is not None and args.client_outstanding < batch:
        raise ValueError("Client outstanding must be at least the tested scheduling concurrency")
    common = common_arguments(args)
    if args.stage == "profile":
        return [
            sys.executable,
            str(args.plugin / "tools/dspark/run_confidence_verification.py"),
            *common,
            "--stage",
            "profile",
            "--batch",
            str(batch),
            "--capture",
            *map(str, sizes),
            "--output-dir",
            str(root),
            *(
                ["--profile-nan-diagnostic", "--profile-stop-after-point", args.profile_stop_after_point]
                if getattr(args, "profile_nan_diagnostic", False)
                else [
                    "--profile-experiment",
                    args.profile_experiment,
                    "--profile-stop-after-point",
                    args.profile_stop_after_point,
                ]
                if getattr(args, "profile_experiment", None)
                else []
            ),
            "--profile-contexts",
            *map(str, args.profile_contexts),
            "--profile-output-tokens",
            str(args.profile_output_tokens),
            "--profile-warmup",
            str(args.profile_warmup),
            "--profile-samples",
            str(args.profile_samples),
        ]
    if args.cost_dir is None:
        raise ValueError("Performance requires --cost-dir from a successful new startup profile")
    profile = args.cost_dir / f"b{batch}" / "cost-profile.json"
    data = json.loads(profile.read_text())
    # Reject schema1 explicitly, without rewriting provenance into a new table.
    if (
        data.get("schema_version") != 2
        or data.get("plugin_sha") != args.plugin_sha
        or data.get("core_sha") != suite.CORE_SHA
        or data["identity"]["max_num_seqs"] != batch
        or data["identity"]["capture_sizes"] != sizes
    ):
        raise ValueError("Cost table source SHA/schema/capacity mismatch; profile this configuration")
    root.parent.mkdir(parents=True, exist_ok=True)
    options = root.parent / f"b{batch}-verification.json"
    benchmark._atomic_write_json(options, {"mode": "confidence", "cost_profile": str(profile.resolve())})
    result = [
        sys.executable,
        str(args.plugin / "tools/dspark/run_performance_suite.py"),
        *common,
        "--max-num-seqs",
        str(batch),
        "--modes",
        *args.modes,
        "--repeats",
        str(args.repeats),
        "--capture-dspark",
        *map(str, sizes),
        "--capture-target",
        *map(str, [n // 6 for n in sizes]),
        "--confidence-verification",
        str(options),
        "--output-dir",
        str(root),
        "--skip-code-evaluation",
        "--execute",
    ]
    if args.client_outstanding is not None:
        result += ["--client-outstanding", str(args.client_outstanding)]
    return result


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    suite.source_gate(args)
    manifest, rows, _ = read_manifest(args.manifest, args.num_prompts)
    if len(rows) != args.num_prompts:
        raise ValueError(
            "Insufficient request instances; import existing repetitions explicitly instead of manufacturing requests"
        )
    benchmark._atomic_write_json(
        args.output_dir / "input-identity.json",
        {
            "manifest_sha256": benchmark._sha256_file(args.manifest),
            "manifest": manifest,
            **input_population(manifest, rows),
            "enable_prefix_caching": False,
        },
    )
    copy_manifest_assets(args.manifest, args.output_dir / "input")
    statuses = []
    rc = 0
    for batch in args.batches:
        row = {"batch": batch, "status": "failed", "rc": None}
        try:
            cmd = command(args, batch, args.output_dir / f"b{batch}")
            row["command"] = cmd
            if args.stage == "profile" and getattr(args, "profile_experiment", None) in (
                "metadata-only",
                "numeric-boundaries",
                "upstream-boundaries",
                "auxiliary-transfers",
                "target-boundaries",
            ):
                controls = []
                if args.profile_experiment == "target-boundaries":
                    controls = [
                        "--max-runtime-seconds",
                        str(TARGET_DIAGNOSTIC_RUNTIME_SECONDS),
                        "--stop-file",
                        str(args.output_dir.parent / "STOP"),
                    ]
                cmd = [
                    sys.executable,
                    "-m",
                    "tools.dspark.profile_process_guard",
                    "--directory",
                    str(args.output_dir / f"b{batch}"),
                    "--receipt",
                    str(args.output_dir / f"b{batch}-supervisor.json"),
                    *controls,
                    "--",
                    *cmd,
                ]
                row["supervised_command"] = cmd
            benchmark._atomic_write_json(args.output_dir / f"b{batch}-command.json", row)
            suite.resources_idle(args.output_dir / f"b{batch}-npu.log")
            row["rc"] = suite.logged(cmd, args.output_dir / f"b{batch}.log")
            if row["rc"]:
                raise RuntimeError("Child failed; stopping subsequent concurrency stages")
            scan(args.output_dir / f"b{batch}.log")
            row["status"] = (
                "diagnostic_completed"
                if (getattr(args, "profile_nan_diagnostic", False) or getattr(args, "profile_experiment", None))
                else "valid"
            )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            rc = 1
        benchmark._atomic_write_json(args.output_dir / f"b{batch}-command.json", row)
        statuses.append(row)
        benchmark._atomic_write_json(args.output_dir / "stages.json", statuses)
        if rc:
            break
    # Per-B suite reports preserve all repeats and paired ratios. The index does
    # not combine different concurrency/input populations into one speedup.
    benchmark._atomic_write_json(
        args.output_dir / "summary.json",
        {
            "status": "failed"
            if rc
            else "diagnostic_completed"
            if (getattr(args, "profile_nan_diagnostic", False) or getattr(args, "profile_experiment", None))
            else "valid",
            "performance_eligible": False
            if (getattr(args, "profile_nan_diagnostic", False) or getattr(args, "profile_experiment", None))
            else None,
            "stage": args.stage,
            "stages": statuses,
            "primary_result": "ROOT_CAUSE_NOT_YET_PROVEN; isolated diagnostic, not performance"
            if (getattr(args, "profile_nan_diagnostic", False) or getattr(args, "profile_experiment", None))
            else "confidence / fixed K end-to-end output tok/s, paired by B and repeat",
            "reports": [f"b{row['batch']}/summary.json" for row in statuses] if args.stage != "profile" else [],
            "model_initializations": sum(
                json.loads((args.output_dir / f"b{r['batch']}" / "lifecycle.json").read_text())[
                    "engine_initializations"
                ]
                for r in statuses
                if (args.output_dir / f"b{r['batch']}" / "lifecycle.json").is_file()
            )
            if args.stage == "profile"
            else None,
        },
    )
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "validate", "repeat"), required=True)
    parser.add_argument("--plugin", type=Path, default=Path("/workspace/vllm-ascend-hust"))
    parser.add_argument("--core", type=Path, default=Path("/workspace/vllm-hust"))
    parser.add_argument("--plugin-sha", required=True)
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-dir", type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--num-prompts", type=int, default=400)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--warmup-prompts", type=int, default=4)
    parser.add_argument("--client-outstanding", type=int, help="Omitted: submit all requests, no B4 client cap")
    parser.add_argument("--repeats", type=int)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("dspark_graph", "dspark_confidence_graph", "target_graph"),
        default=["dspark_graph", "dspark_confidence_graph"],
    )
    parser.add_argument("--capture-sizes", nargs="+", type=int, help="Explicit sizes require one --batches value")
    parser.add_argument("--profile-contexts", nargs="+", type=int, default=[128, 2048])
    parser.add_argument("--profile-output-tokens", type=int, default=512)
    parser.add_argument("--profile-warmup", type=int, default=2)
    parser.add_argument("--profile-samples", type=int, default=5)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    observer = parser.add_mutually_exclusive_group()
    observer.add_argument("--profile-nan-diagnostic", action="store_true")
    observer.add_argument(
        "--profile-experiment",
        choices=(
            "baseline",
            "metadata-only",
            "context-kv-sync",
            "numeric-boundaries",
            "upstream-boundaries",
            "auxiliary-transfers",
            "target-boundaries",
        ),
        help="Replay the original point prefix; never publish costs or performance",
    )
    parser.add_argument("--profile-stop-after-point", default="ctx128-n4-t12-skewed")
    args = parser.parse_args(argv)
    if (args.profile_nan_diagnostic or args.profile_experiment) and (args.stage != "profile" or args.batches != [64]):
        parser.error("NaN diagnostics require the isolated B64 profile stage")
    args.repeats = args.repeats if args.repeats is not None else (3 if args.stage == "repeat" else 1)
    if (
        not args.batches
        or args.batches != sorted(set(args.batches))
        or args.batches[0] < 1
        or args.num_prompts < 1
        or args.repeats < 1
        or (args.capture_sizes and len(args.batches) != 1)
    ):
        parser.error("Require positive sorted batches, requests/repeats, and one B for explicit captures")
    try:
        return run(args)
    except Exception as error:
        print(f"LARGE_BATCH_FAILED={type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
