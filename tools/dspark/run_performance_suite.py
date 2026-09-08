# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Independent process driver. Planning is CPU-only; execution is explicit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import regex as re

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark.performance_code_eval import evaluate
from tools.dspark.performance_report import summarize_suite, validate_stream_result, write_reports
from tools.dspark.prepare_performance_data import read_manifest

CORE_SHA = "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
MODEL_REVISION = "9e8679a9db7eec11efed9925f7efb96549077545"
MODES = ("target_graph", "dspark_graph", "target_eager", "dspark_eager")


def capture_sizes(mode, concurrency, budget, explicit=None):
    if mode not in MODES or concurrency < 1 or budget < 1:
        raise ValueError("Invalid mode, concurrency or token budget")
    q = 6 if mode.startswith("dspark") else 1
    if concurrency * q > budget:
        raise ValueError("Decode token budget cannot hold max_num_seqs * query length")
    if mode.endswith("eager"):
        return []
    sizes = (
        explicit
        if explicit is not None
        else [q * n for n in sorted({1, min(2, concurrency), min(4, concurrency), concurrency})]
    )
    if (
        not sizes
        or sizes != sorted(set(sizes))
        or any(type(n) is not int or n <= 0 or n % q for n in sizes)
        or max(sizes) != concurrency * q
        or max(sizes) > budget
    ):
        raise ValueError(f"Capture sizes must be sorted unique q={q} multiples ending at q*max_num_seqs within budget")
    return list(sizes)


def create_plan(args, records_file, root):
    """Build one process command per repeat/mode, keeping request and graph budgets distinct."""
    if args.num_prompts < 1 or args.repeats < 1 or not 0 <= args.warmup_prompts <= args.num_prompts:
        raise ValueError("Invalid request/repeat/warmup count")
    if args.client_outstanding is not None and args.client_outstanding < 1:
        raise ValueError("Invalid client outstanding cap")
    if len(set(args.modes)) != len(args.modes) or len(set(args.max_num_seqs)) != len(args.max_num_seqs):
        raise ValueError("Duplicate mode or concurrency")
    runs = []
    for concurrency in args.max_num_seqs:
        for repeat in range(args.repeats):
            order = args.modes if repeat % 2 == 0 else list(reversed(args.modes))
            for mode in order:
                dspark = mode.startswith("dspark")
                sizes = capture_sizes(
                    mode,
                    concurrency,
                    args.max_num_batched_tokens,
                    args.capture_dspark if dspark else args.capture_target,
                )
                directory = f"s{concurrency}-{mode}-r{repeat + 1}"
                command = [
                    sys.executable,
                    str(args.plugin / "tools/dspark/benchmark_dspark_acceptance.py"),
                    "--model-dir",
                    str(args.model),
                    "--revision",
                    MODEL_REVISION,
                    "--mode",
                    "dspark" if dspark else "target_only",
                    "--num-spec-tokens",
                    "5",
                    "--measurement-protocol",
                    "async_stream",
                    "--dataset-name",
                    "jsonl",
                    "--dataset-path",
                    str(records_file),
                    "--prompt-field",
                    "prompt_token_ids",
                    "--num-prompts",
                    str(args.num_prompts),
                    "--warmup-prompts",
                    str(args.warmup_prompts),
                    "--output-len",
                    str(args.output_len),
                    "--no-ignore-eos",
                    "--temperature",
                    "0",
                    "--top-p",
                    "1",
                    "--top-k",
                    "-1",
                    "--seed",
                    "0",
                    "--tensor-parallel-size",
                    "8",
                    "--enable-expert-parallel",
                    "--async-scheduling",
                    "--dtype",
                    "bfloat16",
                    "--quantization",
                    "ascend",
                    "--max-num-seqs",
                    str(concurrency),
                    "--max-num-batched-tokens",
                    str(args.max_num_batched_tokens),
                    "--max-model-len",
                    str(args.max_model_len),
                    "--gpu-memory-utilization",
                    str(args.gpu_memory_utilization),
                    "--block-size",
                    "32",
                    "--target-execution-mode",
                    "full_decode_only" if sizes else "eager",
                    "--result-json",
                    str(root / directory / "result.json"),
                ]
                if sizes:
                    command += ["--cudagraph-capture-sizes", *map(str, sizes)]
                if args.client_outstanding is not None:
                    command += ["--client-outstanding", str(args.client_outstanding)]
                runs.append(
                    {
                        "directory": directory,
                        "mode": mode,
                        "repeat": repeat + 1,
                        "max_num_seqs": concurrency,
                        "query_length": 6 if dspark else 1,
                        "capture_sizes": sizes,
                        "command": command,
                    }
                )
    return {
        "schema_version": 1,
        "protocol": "async_llm_delta_stream_v1",
        "plugin_sha": args.plugin_sha,
        "core_sha": CORE_SHA,
        "num_prompts": args.num_prompts,
        "max_num_seqs": args.max_num_seqs,
        "modes": args.modes,
        "repeats": args.repeats,
        "client_outstanding": args.client_outstanding,
        "independence": "one new benchmark process/load/warmup/measure/shutdown per run",
        "mode_order": "forward on odd repeats, reversed on even repeats",
        "runs": runs,
    }


def logged(command, path):
    # Shell positional parameters keep user paths out of shell source. Preserve
    # PIPESTATUS immediately after tee, including when the generator fails.
    script = """log=$1; codes_file=$2; shift 2
"$@" 2>&1 | tee "$log"
codes=("${PIPESTATUS[@]}")
printf '%s\\n' "${codes[*]}" > "$codes_file"
test "${codes[0]}" -eq 0 && test "${codes[1]}" -eq 0
"""
    return subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            script,
            "dspark-log",
            str(path),
            str(path.with_suffix(".pipestatus")),
            *command,
        ]
    ).returncode


def npu_report_is_idle(report):
    if "No running processes found" not in report or "Process id" not in report:
        return False
    for line in report.split("Process id", 1)[1].splitlines():
        fields = line.replace("|", " ").split()
        if len(fields) >= 3 and all(field.isdigit() for field in fields[:3]):
            return False
    return True


def resources_idle(log_path):
    from tools.dspark.p08_r8_checks import idle

    idle()
    result = subprocess.run(["npu-smi", "info"], text=True, capture_output=True)
    log_path.write_text(result.stdout + result.stderr)
    if result.returncode or not npu_report_is_idle(result.stdout):
        raise RuntimeError("NPU resource gate requires the explicit no-running-processes report; inspect npu-smi log")


def source_gate(args):
    for root, expected in ((args.plugin, args.plugin_sha), (args.core, CORE_SHA)):
        if not re.fullmatch(r"[0-9a-f]{40}", expected) or benchmark._git_head(root) != expected:
            raise ValueError(f"Source SHA mismatch: {root}")
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
            raise ValueError(f"Dirty source tree: {root}")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "0":
        raise ValueError("Require VLLM_ALLOW_INSECURE_SERIALIZATION=0")
    if not os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        raise ValueError("Preserve the existing CANN/custom OPP environment")
    from tools.dspark.p08_r8_checks import source

    source(args.plugin, args.core)


def check_case(result, case, plan, records):
    validate_stream_result(result, records)
    if (
        result["plugin_sha"] != plan["plugin_sha"]
        or result["core_sha"] != CORE_SHA
        or result["measured_request_count"] != plan["num_prompts"]
        or result["effective_config"]["max_num_seqs"] != case["max_num_seqs"]
        or result["delivery"]["client_outstanding"] != plan["client_outstanding"]
    ):
        raise ValueError("Run differs from the frozen source/input/delivery plan")
    if case["capture_sizes"] and result["configured_capture_sizes"] != case["capture_sizes"]:
        raise ValueError("Effective capture sizes differ from the plan")
    scheduler = result["streaming"]["scheduler"]
    if scheduler.get("corrupted_requests", 0):
        raise ValueError("Engine reported corrupted/NaN requests")


def run_suite(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest, records, frozen_data = read_manifest(args.manifest, args.num_prompts)
    if manifest["kind"] != "general" and (args.output_len > 1024 or manifest["max_input_tokens"] > 2048):
        raise ValueError("Code workload requires input <=2048 and output <=1024 tokens")
    if max(row["prompt_token_count"] for row in records) + args.output_len > args.max_model_len:
        raise ValueError("Prompt plus output budget exceeds max_model_len; no truncation permitted")
    if manifest["tokenizer_revision"] != MODEL_REVISION:
        raise ValueError("Frozen tokenizer revision differs from benchmark model revision")
    if args.execute:
        source_gate(args)
        for relative, expected in manifest["tokenizer_files_sha256"].items():
            if benchmark._sha256_file(args.model / relative) != expected:
                raise ValueError("Model tokenizer bytes differ from frozen inputs")
    data = root / "requests.jsonl"
    data.write_bytes(b"".join(benchmark._canonical_json_bytes(row) for row in records))
    frozen = root / "input"
    frozen.mkdir()
    shutil.copyfile(args.manifest, frozen / "manifest.json")
    shutil.copyfile(frozen_data, frozen / manifest["records_file"])
    plan = create_plan(args, data, root)
    plan["input_sha256"] = benchmark._sha256_file(data)
    plan["input_manifest_sha256"] = benchmark._sha256_file(args.manifest)
    benchmark._atomic_write_json(root / "plan.json", plan)
    if not args.execute:
        print(f"PLAN_ONLY={root}; no model process launched")
        return 0
    resources_idle(root / "npu-initial.log")
    reference_quality = evaluate(records, [], root / "reference-quality", args.sandbox_image, reference=True)
    baseline_comparison = {}
    for case in plan["runs"]:
        directory = root / case["directory"]
        directory.mkdir()
        benchmark._atomic_write_json(directory / "command.json", case)
        receipt = {"status": "failed", "generation_rc": None, "quality": {"status": "unavailable"}}
        try:
            resources_idle(directory / "npu-before.log")
            receipt["generation_rc"] = logged(case["command"], directory / "generation.log")
            resources_idle(directory / "npu-after.log")
            if receipt["generation_rc"] != 0:
                raise RuntimeError("Benchmark process failed; partial artifacts retained")
            from tools.dspark.graph64_checks import read, scan

            scan(directory / "generation.log")
            result = read(directory / "result.json")
            check_case(result, case, plan, records)
            common = {
                "input": result["prompt_set_sha256"],
                "sampling": result["sampling"],
                "model": result["model"],
                "delivery": result["delivery"],
                "warmup": result["warmup_request_count"],
                "effective": {
                    key: value
                    for key, value in result["effective_config"].items()
                    if key
                    not in (
                        "speculative_config",
                        "enforce_eager",
                        "target_execution_mode",
                        "cudagraph_mode",
                        "cudagraph_metrics",
                        "frontend_configured_capture_sizes",
                        "npugraph_ex_enabled",
                        "static_kernel_enabled",
                    )
                },
            }
            previous = baseline_comparison.setdefault(case["max_num_seqs"], common)
            if previous != common:
                raise ValueError("Modes used different inputs/configuration/delivery")
            requests = result["streaming"]["requests"]
            for request, task in zip(requests, records):
                request["case_id"] = task["case_id"]
                request["input_length"] = task["prompt_token_count"]
                request["output_length"] = len(request["output_token_ids"])
            benchmark._atomic_write_json(directory / "requests.json", requests)
            quality = evaluate(records, requests, directory / "quality", args.sandbox_image)
            if (
                reference_quality["summary"]["status"] != "available"
                or reference_quality["summary"]["all_task_pass_fraction"] != 1
            ):
                quality["summary"]["status"] = "unavailable"
                quality["summary"]["all_task_pass_fraction"] = None
                quality["summary"]["reference_validation"] = "unavailable_or_failed; cannot certify unit-test quality"
                benchmark._atomic_write_json(directory / "quality" / "quality.json", quality)
            receipt.update(
                status="valid",
                quality=quality["summary"],
                result_sha256=benchmark._sha256_file(directory / "result.json"),
                log_sha256=benchmark._sha256_file(directory / "generation.log"),
                nan_evidence="No reported NaN in merged log; internal tensors not inspected",
                diagnostics_enabled=False,
                artifact_sha256={
                    str(path.relative_to(directory)): benchmark._sha256_file(path)
                    for path in [directory / "requests.json", *(directory / "quality").iterdir()]
                    if path.is_file()
                },
            )
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            partial = directory / "partial-stream.json"
            if partial.is_file():
                saved = json.loads(partial.read_text())
                partial_requests = saved["requests"]
                receipt["generation_diagnostics"] = {
                    "completed": sum(
                        row is not None and row.get("completed_monotonic") is not None for row in partial_requests
                    ),
                    "not_submitted": sum(row is None for row in partial_requests),
                    "failed_or_cancelled": sum(
                        row is not None and row.get("completed_monotonic") is None for row in partial_requests
                    ),
                }
        finally:
            benchmark._atomic_write_json(directory / "receipt.json", receipt)
        if receipt["status"] != "valid":
            break  # No escalation after an execution, resource or evidence failure.
    summary = summarize_suite(root)
    write_reports(summary, root / "summary.json", root / "summary.csv", root / "summary.md")
    print(f"PERFORMANCE_STATUS={summary['status']}; QUALITY_STATUS is reported separately per run; {root}")
    return 0 if summary["status"] == "valid" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, default=Path("/workspace/vllm-ascend-hust"))
    parser.add_argument("--core", type=Path, default=Path("/workspace/vllm-hust"))
    parser.add_argument("--plugin-sha", required=True)
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-num-seqs", nargs="+", type=int, default=[4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["target_graph", "dspark_graph"])
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--client-outstanding", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--capture-target", nargs="+", type=int)
    parser.add_argument("--capture-dspark", nargs="+", type=int)
    parser.add_argument(
        "--sandbox-image", help="Locally installed digest-pinned Python image; omitted => quality unavailable"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Explicitly launch independent model processes; default plans only"
    )
    args = parser.parse_args(argv)
    if args.output_len <= 0 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Require positive output length and 0 < memory fraction < 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        print("Output directory already exists; refusing to modify an earlier run", file=sys.stderr)
        return 1
    try:
        return run_suite(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # No overwrite of prior results even on setup failure.
        path = args.output_dir / "setup-failure.json"
        if not path.exists():
            benchmark._atomic_write_json(path, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"PERFORMANCE_FAILED={type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
