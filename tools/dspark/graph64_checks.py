# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Fixed 400-request Graph64/eager64 reproduction, without forensic diagnostics."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import regex as re

from tools.dspark.p08_r8_checks import scan as scan_runtime_errors

BASELINE = "7e23a859defa5a12b3e583fcf4ce57a52da94c71"
CORE_SHA = "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
INPUT_SHA = "3889b4bda22442e69062cd5c3888090515b67a303006ef59afea8d72424258b2"
MODEL_DIR = "/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"
REVISION = "9e8679a9db7eec11efed9925f7efb96549077545"
MODEL_FINGERPRINT = "8a795361bf830c594b4c6e21e0682da4df2a63a3d78c27099e92ed3629ee6f66"
CAPTURE = (6, 12, 18, 24, 30, 36, 42, 48, 96, 192, 288, 384)


def sampling():
    return dict(ignore_eos=False, output_len=256, seed=0, temperature=0.0, top_k=-1, top_p=1.0)


def read(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def scan(path):
    scan_runtime_errors(path)
    hits = [line for line in path.read_text(errors="replace").splitlines() if re.search(r"\bnan\b", line, re.I)]
    assert not hits, f"NaN reported in generation log: {hits}"


def command(plugin, output, mode):
    assert mode in ("graph", "eager")
    args = [
        sys.executable,
        str(plugin / "tools/dspark/benchmark_dspark_acceptance.py"),
        "--model-dir",
        MODEL_DIR,
        "--revision",
        REVISION,
        "--mode",
        "dspark",
        "--num-spec-tokens",
        "5",
        "--dataset-name",
        "jsonl",
        "--dataset-path",
        str(output / "input-400.jsonl"),
        "--prompt-field",
        "prompt_token_ids",
        "--num-prompts",
        "400",
        "--warmup-prompts",
        "1",
        "--output-len",
        "256",
        "--no-ignore-eos",
        "--temperature",
        "0.0",
        "--top-p",
        "1.0",
        "--top-k",
        "-1",
        "--seed",
        "0",
        "--tensor-parallel-size",
        "8",
        "--enable-expert-parallel",
        "--async-scheduling",
        "--max-num-seqs",
        "64",
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "8192",
        "--block-size",
        "32",
        "--gpu-memory-utilization",
        "0.9",
        "--dtype",
        "bfloat16",
        "--quantization",
        "ascend",
        "--tokenizer-mode",
        "deepseek_v4",
        "--result-json",
        str(output / mode / "result.json"),
        "--target-execution-mode",
        "full_decode_only" if mode == "graph" else "eager",
    ]
    if mode == "graph":
        args += ["--cudagraph-capture-sizes", *map(str, CAPTURE)]
    return args


def prepare(evidence, output, plugin):
    plan = read(evidence / "plan.json")
    assert (plan["plugin_sha"], plan["core_sha"]) == (BASELINE, CORE_SHA)
    assert plan["sampling"] == sampling() and plan["diagnostics"] is False
    assert plan["measured_requests"] == 400 and plan["warmup_requests"] == 1
    assert plan["gpu_memory_utilization"] == 0.9 and plan["capture_sizes"]["64"] == list(CAPTURE)
    data = (evidence / "input-400.jsonl").read_bytes()
    assert digest(data) == plan["input_sha256"] == INPUT_SHA
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == 400
    assert all(
        row["prompt_token_ids"] and all(type(token) is int and token >= 0 for token in row["prompt_token_ids"])
        for row in rows
    )
    original = read(evidence / "s64-graph-r1/command.json")
    expected = command(plugin, output, "graph")
    # Validate every original argument; only interpreter/source/output paths move.
    normalized = list(original)
    normalized[:2] = expected[:2]
    for flag in ("--dataset-path", "--result-json"):
        normalized[normalized.index(flag) + 1] = expected[expected.index(flag) + 1]
    assert normalized == expected, "Original command differs from the audited Graph64 configuration"
    (output / "input-400.jsonl").write_bytes(data)
    write(output / "original-plan.json", plan)
    write(output / "original-command.json", original)
    # The archive reference bytes differ from plan.reference_sha256. Record that
    # fact; the actual benchmark input is independently pinned above.
    write(
        output / "input-audit.json",
        dict(
            input_sha256=INPUT_SHA,
            requests=400,
            reference_sha256_declared=plan["reference_sha256"],
            reference_sha256_observed=digest((evidence / "reference.json").read_bytes()),
            reference_used=False,
        ),
    )
    print("INPUT_AND_COMMAND_VERIFIED", INPUT_SHA)


def result(path, sha, mode):
    r = read(path)
    assert (r["plugin_sha"], r["core_sha"]) == (sha, CORE_SHA)
    assert r["performance_eligible"] is True and r["nan_diagnostic"]["enabled"] is False
    assert r["runner"] == "mrv2" and r["mode"] == "dspark"
    assert r["model"]["fingerprint_sha256"] == MODEL_FINGERPRINT
    assert r["dataset"]["file_sha256"] == INPUT_SHA
    assert r["measured_request_count"] == 400 and r["warmup_request_count"] == 1
    assert r["sampling"] == sampling() and r["cleanup"]["engine_shutdown_complete"] is True
    c = r["effective_engine_config"]
    assert c["tensor_parallel_size"] == 8 and c["enable_expert_parallel"] is True
    assert c["max_num_seqs"] == 64 and c["max_model_len"] == c["max_num_batched_tokens"] == 8192
    assert c["async_scheduling"] is True and c["enable_prefix_caching"] is False
    assert c["speculative_config"] == dict(enforce_eager=True, method="dspark", num_speculative_tokens=5)
    assert r["dspark_enforce_eager"] is True
    assert r["target_enforce_eager"] is (mode == "eager")
    assert r["cudagraph_mode_effective"] == ("FULL_DECODE_ONLY" if mode == "graph" else "NONE")
    outputs = r["outputs"]
    assert len(outputs) == len(r["prompt_identities"]) == 400
    assert [row["request_index"] for row in outputs] == list(range(400))
    for row, identity in zip(outputs, r["prompt_identities"]):
        assert row["finish_reason"] in ("stop", "length")
        assert type(row["output_token_count"]) is int and 0 < row["output_token_count"] <= 256
        assert re.fullmatch("[0-9a-f]{64}", row["output_token_sha256"])
        assert row["prompt_token_sha256"] == identity["prompt_token_sha256"]
        assert row["source_record_sha256"] == identity["source_record_sha256"]
    tokens = sum(row["output_token_count"] for row in outputs)
    elapsed = r["timing"]["elapsed_seconds"]
    throughput = r["throughput"]["output_tokens_per_second"]
    assert elapsed > 0 and math.isfinite(elapsed) and math.isfinite(throughput)
    assert r["timing"]["warmup_included"] is False
    assert r["timing"]["graph_telemetry_rpc_included"] is False
    assert tokens == r["throughput"]["total_output_tokens"]
    assert math.isclose(throughput, tokens / elapsed, rel_tol=1e-9)
    acceptance = r["acceptance"]
    accepted = acceptance["accepted_candidate_tokens_per_verification"]
    effective = acceptance["effective_committed_tokens_per_verification"]
    assert math.isfinite(accepted) and 0 <= accepted <= 5
    assert math.isfinite(effective) and 1 <= effective <= 6
    records = []
    count = 0
    if mode == "graph":
        assert r["configured_capture_sizes"] == r["observed_capture_sizes"] == list(CAPTURE)
        graph = r["graph_execution"]
        assert graph["replay_evidence_status"] == "available"
        assert graph["source"] == "mrv2_successful_execute_model_full_replay"
        assert len(graph["boundary_snapshots"]) == 3
        interval = graph["measured_runtime"]
        workers = interval["workers"]
        assert sorted(w["rank"] for w in workers) == list(range(8))
        count = interval["graph_replay_count"]
        records = interval["records"]
        assert count == r["measured_graph_replay_count"] > 0
        assert sum(row["count"] for row in records) == count
        for row in records:
            assert row["runtime_mode"] == "FULL" and row["count"] > 0
            assert row["num_padded_tokens"] in CAPTURE
            assert 0 < row["num_unpadded_tokens"] <= row["num_padded_tokens"]
            assert row["num_paddings"] == row["num_padded_tokens"] - row["num_unpadded_tokens"]
        for worker in workers:
            assert worker["graph_replay_count"] == count and worker["records"] == records
            assert worker["failed_execution_count"] == 0
        assert r["measured_eager_fallback_count"] is None
    summary = dict(
        mode=mode,
        requests=400,
        output_tokens=tokens,
        elapsed_seconds=elapsed,
        tok_s=throughput,
        accepted_length=accepted,
        effective_length=effective,
        measured_full_replays=count,
        measured_full_shapes=records,
        eager_fallback=r["measured_eager_fallback_count"],
        coverage="unavailable",
    )
    write(Path(path).with_name("validated.json"), summary)
    return r, summary


def compare(output, sha):
    graph, g = result(output / "graph/result.json", sha, "graph")
    eager, e = result(output / "eager/result.json", sha, "eager")
    assert graph["prompt_set_sha256"] == eager["prompt_set_sha256"]
    assert graph["model"] == eager["model"] and graph["sampling"] == eager["sampling"]
    # Execution/telemetry options differ by design; all other resolved settings
    # must agree, including the model's real hybrid KV block mapping.
    execution_fields = {
        "cudagraph_metrics",
        "cudagraph_mode",
        "enforce_eager",
        "frontend_configured_capture_sizes",
        "target_execution_mode",
        "npugraph_ex_enabled",
    }

    def comparable(r):
        return {k: v for k, v in r["effective_engine_config"].items() if k not in execution_fields}

    assert comparable(graph) == comparable(eager)
    pairs = list(zip(graph["outputs"], eager["outputs"]))
    summary = dict(
        plugin_sha=sha,
        core_sha=CORE_SHA,
        status="MEASUREMENTS_VALIDATED",
        performance_provisional=True,
        runs=[g, e],
        graph_over_eager_speedup=g["tok_s"] / e["tok_s"],
        different_output_hashes=sum(a["output_token_sha256"] != b["output_token_sha256"] for a, b in pairs),
        different_output_lengths=sum(a["output_token_count"] != b["output_token_count"] for a, b in pairs),
        exact_token_cross_mode_blocking=False,
        full_occupancy_required=False,
        replay_rank_aggregation="require_identical_then_use_one_rank",
    )
    write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "result", "compare", "scan"))
    parser.add_argument("values", nargs="+")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(*map(Path, args.values))
    elif args.action == "launch":
        plugin, output = map(Path, args.values[:2])
        mode = args.values[2]
        (output / mode).mkdir()
        argv = command(plugin, output, mode)
        write(output / mode / "command.json", argv)
        os.chdir("/workspace")
        os.execv(argv[0], argv)
    elif args.action == "result":
        _, summary = result(Path(args.values[0]), *args.values[1:])
        print(json.dumps(summary, indent=2, allow_nan=False))
    elif args.action == "scan":
        scan(Path(args.values[0]))
    else:
        compare(Path(args.values[0]), args.values[1])


if __name__ == "__main__":
    main()
