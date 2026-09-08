# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
import json
import subprocess
from pathlib import Path

import pytest

from tools.dspark import graph64_checks as checks

SHA = "a" * 40
ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "tools/dspark/run_dspark_graph64.sh"


def _result(mode):
    records = [dict(count=11, num_padded_tokens=384, num_unpadded_tokens=378, num_paddings=6, runtime_mode="FULL")]
    worker = dict(graph_replay_count=11, records=records, failed_execution_count=0)
    config = dict(
        tensor_parallel_size=8,
        enable_expert_parallel=True,
        max_num_seqs=64,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        async_scheduling=True,
        enable_prefix_caching=False,
        speculative_config=dict(enforce_eager=True, method="dspark", num_speculative_tokens=5),
        npugraph_ex_enabled=mode == "graph",
    )
    identity = dict(prompt_token_sha256="b" * 64, source_record_sha256="c" * 64)
    outputs = [
        dict(request_index=i, finish_reason="stop", output_token_count=2, output_token_sha256="d" * 64, **identity)
        for i in range(400)
    ]
    return dict(
        plugin_sha=SHA,
        core_sha=checks.CORE_SHA,
        performance_eligible=True,
        nan_diagnostic=dict(enabled=False),
        runner="mrv2",
        mode="dspark",
        model=dict(fingerprint_sha256=checks.MODEL_FINGERPRINT),
        dataset=dict(file_sha256=checks.INPUT_SHA),
        measured_request_count=400,
        warmup_request_count=1,
        sampling=dict(checks.sampling()),
        cleanup=dict(engine_shutdown_complete=True),
        effective_engine_config=config,
        dspark_enforce_eager=True,
        target_enforce_eager=mode == "eager",
        cudagraph_mode_effective="FULL_DECODE_ONLY" if mode == "graph" else "NONE",
        outputs=outputs,
        prompt_identities=[identity] * 400,
        prompt_set_sha256="e" * 64,
        throughput=dict(output_tokens_per_second=80, total_output_tokens=800),
        timing=dict(elapsed_seconds=10, warmup_included=False, graph_telemetry_rpc_included=False),
        acceptance=dict(
            accepted_candidate_tokens_per_verification=3.1, effective_committed_tokens_per_verification=4.1
        ),
        configured_capture_sizes=list(checks.CAPTURE),
        observed_capture_sizes=list(checks.CAPTURE),
        measured_graph_replay_count=11,
        measured_eager_fallback_count=None if mode == "graph" else 0,
        graph_execution=dict(
            replay_evidence_status="available",
            source="mrv2_successful_execute_model_full_replay",
            boundary_snapshots=[[], [], []],
            warmup_runtime=dict(graph_replay_count=999),
            measured_runtime=dict(
                graph_replay_count=11,
                records=records,
                workers=[dict(rank=i, **copy.deepcopy(worker)) for i in range(8)],
            ),
        ),
    )


def test_measured_replay_is_counted_once_and_eos_differences_do_not_block(tmp_path):
    for mode in ("graph", "eager"):
        (tmp_path / mode).mkdir()
        value = _result(mode)
        if mode == "eager":
            value["outputs"][0]["output_token_sha256"] = "f" * 64
            value["outputs"][0]["output_token_count"] = 1
            value["throughput"]["total_output_tokens"] = 799
            value["throughput"]["output_tokens_per_second"] = 79.9
        checks.write(tmp_path / mode / "result.json", value)
    checks.compare(tmp_path, SHA)
    summary = checks.read(tmp_path / "summary.json")
    assert summary["runs"][0]["measured_full_replays"] == 11
    assert summary["different_output_hashes"] == summary["different_output_lengths"] == 1
    assert summary["graph_over_eager_speedup"] == pytest.approx(80 / 79.9)
    assert summary["runs"][0]["coverage"] == "unavailable"


@pytest.mark.parametrize(
    "failure",
    [
        "warmup_only",
        "missing_rank",
        "failed_execution",
        "missing_output",
        "bad_output",
        "wrong_sha",
        "diagnostic",
        "bad_shape",
        "bad_tokens",
        "bad_timing",
    ],
)
def test_result_gate_rejects_incomplete_or_false_success(tmp_path, failure):
    value = _result("graph")
    interval = value["graph_execution"]["measured_runtime"]
    if failure == "warmup_only":
        interval["graph_replay_count"] = value["measured_graph_replay_count"] = 0
    elif failure == "missing_rank":
        interval["workers"].pop()
    elif failure == "failed_execution":
        interval["workers"][0]["failed_execution_count"] = 1
    elif failure == "missing_output":
        value["outputs"].pop()
    elif failure == "bad_output":
        value["outputs"][0]["output_token_sha256"] = ""
    elif failure == "wrong_sha":
        value["plugin_sha"] = "b" * 40
    elif failure == "diagnostic":
        value["nan_diagnostic"]["enabled"] = True
    elif failure == "bad_shape":
        interval["records"][0]["num_padded_tokens"] = 400
    elif failure == "bad_tokens":
        value["throughput"]["total_output_tokens"] = 999
    else:
        value["timing"]["elapsed_seconds"] = float("nan")
    (tmp_path / "result.json").write_text(json.dumps(value))
    with pytest.raises((AssertionError, ValueError)):
        checks.result(tmp_path / "result.json", SHA, "graph")
    assert not (tmp_path / "validated.json").exists()


def test_original_command_validation_and_exact_input_hash(tmp_path, monkeypatch):
    original, output = tmp_path / "original", tmp_path / "new"
    original.mkdir()
    output.mkdir()
    (original / "s64-graph-r1").mkdir()
    data = (json.dumps(dict(prompt_token_ids=[1, 2])) + "\n").encode() * 400
    monkeypatch.setattr(checks, "INPUT_SHA", checks.digest(data))
    (original / "input-400.jsonl").write_bytes(data)
    (original / "reference.json").write_text("{}\n")
    checks.write(
        original / "plan.json",
        dict(
            plugin_sha=checks.BASELINE,
            core_sha=checks.CORE_SHA,
            sampling=checks.sampling(),
            diagnostics=False,
            measured_requests=400,
            warmup_requests=1,
            gpu_memory_utilization=0.9,
            capture_sizes={"64": list(checks.CAPTURE)},
            input_sha256=checks.INPUT_SHA,
            reference_sha256="unverified",
        ),
    )
    command = checks.command(ROOT, original, "graph")
    checks.write(original / "s64-graph-r1/command.json", command)
    checks.prepare(original, output, ROOT)
    assert (output / "input-400.jsonl").read_bytes() == data
    assert checks.read(output / "input-audit.json")["reference_used"] is False
    command[command.index("--max-num-seqs") + 1] = "128"
    checks.write(original / "s64-graph-r1/command.json", command)
    with pytest.raises(AssertionError, match="Original command"):
        checks.prepare(original, output, ROOT)


def test_run_order_failure_stops_eager_but_keeps_evidence(tmp_path):
    # Execute the real run_case/main orchestration with external commands replaced.
    # Preserve shell control flow and PIPESTATUS rather than accepting a fake RPC.
    script = SCRIPT.read_text()
    assert "set -e" not in script and "\nexit " not in script and "build_aclnn" not in script
    assert 'local codes=("${PIPESTATUS[@]}")' in script
    assert "run_case graph" in script and script.index("run_case graph") < script.index("run_case eager")
    assert "--ignore-eos" not in checks.command(ROOT, tmp_path, "graph")
    assert not any("diagnostic" in item for item in checks.command(ROOT, tmp_path, "graph"))
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    # Call the original run_case and logged functions. A failing launcher must
    # still invoke error/idle checks and record all pipeline statuses.
    functions = script[script.index("logged()") : script.index("main()")]
    functions = functions[: functions.index("source_gate()")] + functions[functions.index("run_case()") :]
    harness = f"""set -o pipefail
G64_OUT='{tmp_path}'
G64_PLUGIN='{ROOT}'
G64_CHECKS=checks
G64_COMMON=common
G64_SHA={SHA}
{functions}
python() {{
    printf '%s\\n' "$*" >> "$G64_OUT/calls.txt"
    if test "$2" = launch; then return 7; fi
    return 0
}}
npu-smi() {{ return 0; }}
run_case graph
case_rc=$?
printf '%s' "$case_rc" > "$G64_OUT/case_rc"
"""
    subprocess.run(["bash", "-c", harness], check=True, capture_output=True, text=True)
    assert (tmp_path / "case_rc").read_text() == "1"
    assert (tmp_path / "graph-run-pipestatus.txt").read_text().strip() == "7 0"
    assert "LAUNCH_RC=7" in (tmp_path / "graph-gate.txt").read_text()
    calls = (tmp_path / "calls.txt").read_text()
    assert "checks scan" in calls and calls.count("common idle") == 2
    assert "checks result" not in calls


@pytest.mark.parametrize(
    "message",
    ["NaN at target output", "RuntimeError: shape mismatch", "EngineDeadError"],
    ids=["nan", "shape_error", "engine_dead"],
)
def test_errors_and_nan_fail_without_enabling_forensics(tmp_path, message):
    log = tmp_path / "run.log"
    log.write_text(message + "\n")
    with pytest.raises(AssertionError):
        checks.scan(log)
