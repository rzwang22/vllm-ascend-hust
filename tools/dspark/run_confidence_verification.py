# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit first-round or isolated profile runs; never launches a concurrency sweep."""

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import run_performance_suite as suite
from tools.dspark.graph64_checks import scan
from tools.dspark.prepare_performance_data import read_manifest
from tools.dspark.verification_tools import checkpoint_preflight, compile_profile, freeze_verification_config


def cases(stage, batch):
    if stage == "first":
        return [
            ("specified-b1", 1, [0], False),
            ("specified-b4", 4, [5, 2, 0, 4], False),
            ("confidence-b4", 4, None, False),
        ]
    if stage == "extend":
        return [(f"confidence-b{batch}", batch, None, False)]
    # Same configured max_num_seqs and captures across all profile processes;
    # client outstanding changes the real request count. No production engine
    # survives a profile trial, and these records cannot be performance PASS.
    return [(f"profile-n{n}-ell{ell}", n, [ell], True) for n in range(1, batch + 1) for ell in (0, 1, 3, 5)]


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    suite.source_gate(args)
    benchmark._atomic_write_json(args.output_dir / "checkpoint.json", checkpoint_preflight(args.model))
    manifest, records, _ = read_manifest(args.manifest, args.num_prompts)
    if len(records) < args.num_prompts:
        raise ValueError("Manifest has fewer real requests than --num-prompts; no replay/duplication.")
    if manifest["tokenizer_revision"] != suite.MODEL_REVISION:
        raise ValueError("Frozen tokenizer revision differs from the model revision.")
    for relative, expected in manifest["tokenizer_files_sha256"].items():
        if benchmark._sha256_file(args.model / relative) != expected:
            raise ValueError("Tokenizer bytes differ from frozen inputs.")
    if max(row["prompt_token_count"] for row in records) + args.output_len > args.max_model_len:
        raise ValueError("Prompt/output budget exceeds max_model_len; no truncation.")
    frozen = args.output_dir / "input.jsonl"
    frozen.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    shutil.copyfile(args.manifest, args.output_dir / "manifest.json")
    benchmark._atomic_write_json(
        args.output_dir / "plan.json",
        {
            "schema_version": 1,
            "plugin_sha": args.plugin_sha,
            "core_sha": suite.CORE_SHA,
            "stage": args.stage,
            "num_prompts": args.num_prompts,
            "input_sha256": benchmark._sha256_file(frozen),
            "manifest": manifest,
            "cases": cases(args.stage, args.batch),
            "server_status": "NOT_YET_VALIDATED",
        },
    )
    profile_results = []
    status = 0
    for name, concurrent, lengths, profiling in cases(args.stage, args.batch):
        directory = args.output_dir / name
        directory.mkdir()
        receipt = {"status": "failed", "generation_rc": None, "performance_eligible": not profiling and lengths is None}
        try:
            suite.resources_idle(directory / "npu-before.log")
            maximum = args.batch if profiling else concurrent
            if lengths is None:
                if args.cost_profile is None or not args.cost_profile.is_file():
                    raise ValueError(
                        "A measured compatible --cost-profile is required; run the separate profile command first."
                    )
                options = {"mode": "confidence", "cost_profile": str(args.cost_profile.resolve())}
                if args.calibration:
                    options["calibration"] = str(args.calibration.resolve())
            else:
                options = {"mode": "specified_lengths", "lengths": lengths, "profile": profiling}
            config = directory / "verification.json"
            benchmark._atomic_write_json(config, options)
            config = freeze_verification_config(config, directory / "assets")
            local = argparse.Namespace(**vars(args))
            local.max_num_seqs = [maximum]
            local.repeats = 1
            local.modes = ["dspark_graph"]
            local.capture_dspark = [6] if args.stage == "first" and maximum == 1 else args.capture
            local.capture_target = None
            local.client_outstanding = concurrent if profiling else args.client_outstanding
            local.confidence_verification = config
            plan = suite.create_plan(local, frozen, args.output_dir)
            command = plan["runs"][0]["command"]
            # Reuse the exact streaming benchmark and sampling defaults.
            index = command.index("--result-json")
            command[index + 1] = str(directory / "result.json")
            command += ["--confidence-verification", str(config)]
            benchmark._atomic_write_json(directory / "command.json", {"command": command, "options": options})
            receipt["generation_rc"] = suite.logged(command, directory / "generation.log")
            suite.resources_idle(directory / "npu-after.log")
            if receipt["generation_rc"]:
                raise RuntimeError("Generation failed; no subsequent case will run.")
            scan(directory / "generation.log")
            result = json.loads((directory / "result.json").read_text())
            if len(result["outputs"]) != args.num_prompts or result["measured_request_count"] != args.num_prompts:
                raise ValueError("Incomplete output artifacts.")
            suite.validate_stream_result(result, records, specified_verification_test=lengths is not None)
            evidence = result["confidence_verification"]
            if evidence["status"] != "available" or evidence["logical_full_replays"] <= 0:
                raise ValueError("No successful measured FULL verification evidence.")
            if lengths is not None and len(lengths) > 1 and evidence["mixed_length_full_replays"] <= 0:
                raise ValueError("Specified-length B4 did not produce measured mixed-length FULL replay.")
            if lengths is None and evidence["mode"] != "confidence":
                raise ValueError("Specified lengths are not confidence-scheduled verification.")
            if lengths is None and evidence["mixed_length_full_replays"] == 0:
                receipt["adaptive_mixing"] = (
                    "not_observed; policy scores unchanged, no claim of mixed learned-policy success"
                )
            receipt.update(
                status="valid", verification=evidence, result_sha256=benchmark._sha256_file(directory / "result.json")
            )
            if profiling:
                profile_results.append(directory / "result.json")
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            status = 1
        finally:
            benchmark._atomic_write_json(directory / "receipt.json", receipt)
        if status:
            break
    if not status and args.stage == "profile":
        benchmark._atomic_write_json(args.output_dir / "cost-profile.json", compile_profile(profile_results))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("first", "profile", "extend"), default="first")
    parser.add_argument("--plugin", type=Path, default=Path("/workspace/vllm-ascend-hust"))
    parser.add_argument("--core", type=Path, default=Path("/workspace/vllm-hust"))
    parser.add_argument("--plugin-sha", required=True)
    parser.add_argument("--model", type=Path, default=Path("/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cost-profile", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--client-outstanding", type=int)
    parser.add_argument("--capture", nargs="+", type=int, default=[6, 12, 18, 24])
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.num_prompts < args.batch or args.output_len < 1:
        parser.error("Require positive lengths and num_prompts >= batch.")
    try:
        return run(args)
    except Exception as error:
        # Never touch an earlier directory. run() creates this run exactly once.
        print(f"VERIFICATION_FAILED={type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
