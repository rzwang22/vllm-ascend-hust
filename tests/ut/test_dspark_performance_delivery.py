# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import ast
import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.ut.test_dspark_acceptance_benchmark import _args, _run, _run_graph
from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import performance_code_eval as code_eval
from tools.dspark import performance_report as report
from tools.dspark import performance_stream as stream
from tools.dspark import prepare_performance_data as data
from tools.dspark import run_performance_suite as driver


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text.encode())}


def raw_tasks(n):
    return [
        {
            "task_id": f"task-{i}",
            "prompt": f"Task {i}: def f(x):\n    # implement x + {i}",
            "test": f"def check(candidate):\n    assert candidate(2) == {i + 2}",
            "entry_point": "f",
            "canonical_solution": f"\n    return x + {i}\n",
        }
        for i in range(n)
    ]


def frozen(tmp_path, n=6, kind="humaneval"):
    rows, _ = data.build_records(
        raw_tasks(n), Tokenizer(), count=n, max_input_tokens=2048, source="test-fixture", revision="a" * 40, kind=kind
    )
    root = tmp_path / "frozen"
    root.mkdir()
    file = root / "requests.jsonl"
    file.write_bytes(b"".join(benchmark._canonical_json_bytes(row) for row in rows))
    manifest = {
        "schema_version": 1,
        "records_file": file.name,
        "records_sha256": benchmark._sha256_file(file),
        "num_unique_samples": n,
        "max_input_tokens": 2048,
        "kind": kind,
        "tokenizer_revision": driver.MODEL_REVISION,
        "tokenizer_files_sha256": {},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, rows


@pytest.mark.parametrize("num_prompts,cap", [(1, 64), (63, 64), (400, 32), (2048, 768)])
def test_request_count_is_independent_from_scheduler_cap(tmp_path, num_prompts, cap):
    args = driver.parse_args(
        [
            "--plugin-sha",
            "a" * 40,
            "--manifest",
            "unused",
            "--output-dir",
            str(tmp_path),
            "--num-prompts",
            str(num_prompts),
            "--max-num-seqs",
            str(cap),
        ]
    )
    plan = driver.create_plan(args, tmp_path / "input.jsonl", tmp_path)
    assert plan["num_prompts"] == num_prompts and plan["max_num_seqs"] == [cap]
    assert len(plan["runs"]) == 6
    assert [case["mode"] for case in plan["runs"]] == [
        "target_graph",
        "dspark_graph",
        "dspark_graph",
        "target_graph",
        "target_graph",
        "dspark_graph",
    ]
    assert len({case["directory"] for case in plan["runs"]}) == 6
    for case in plan["runs"]:
        command = case["command"]
        parsed = benchmark.parse_args(command[2:])
        assert parsed.measurement_protocol == "async_stream" and parsed.ignore_eos is False
        assert command[command.index("--num-prompts") + 1] == str(num_prompts)
        assert command[command.index("--max-num-seqs") + 1] == str(cap)
        assert command[command.index("--measurement-protocol") + 1] == "async_stream"
        assert max(case["capture_sizes"]) == cap * case["query_length"]


@pytest.mark.parametrize(
    "mode,explicit",
    [
        ("dspark_graph", [1, 2, 4]),
        ("dspark_graph", [6, 12, 25]),
        ("target_graph", [6, 12, 24]),
        ("target_graph", [1, 1, 4]),
    ],
)
def test_capture_sizes_cannot_confuse_q1_q6(mode, explicit):
    with pytest.raises(ValueError, match="Capture sizes"):
        driver.capture_sizes(mode, 4, 8192, explicit)
    with pytest.raises(ValueError, match="token budget"):
        driver.capture_sizes("dspark_graph", 64, 378)


@pytest.mark.parametrize("mode,sizes", [("target_only", [6, 12]), ("dspark", [1, 2])])
def test_direct_stream_cli_rejects_wrong_query_shapes(mode, sizes):
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--model-dir",
                "unused",
                "--mode",
                mode,
                "--dataset-name",
                "jsonl",
                "--dataset-path",
                "unused",
                "--result-json",
                "unused.json",
                "--measurement-protocol",
                "async_stream",
                "--target-execution-mode",
                "full_decode_only",
                "--max-num-seqs",
                "2",
                "--cudagraph-capture-sizes",
                *map(str, sizes),
            ]
        )


def test_target_only_really_has_no_speculative_config(tmp_path):
    args = _args(tmp_path, mode="target_only")
    assert benchmark.build_engine_kwargs(args)["speculative_config"] is None


def test_frozen_samples_hashes_and_no_truncation(tmp_path):
    path, rows = frozen(tmp_path)
    manifest, selected, _ = data.read_manifest(path, 4)
    assert selected == rows[:4] and manifest["num_unique_samples"] == 6
    assert all(row["tests"]["test"] == row["raw_task"]["test"] for row in rows)
    assert all(row["replay_of"] is None for row in rows)
    with pytest.raises(ValueError, match="Insufficient"):
        data.read_manifest(path, 7)
    source = raw_tasks(3)
    source[0]["prompt"] += "x" * 3000
    filtered, dispositions = data.build_records(
        source, Tokenizer(), count=2, max_input_tokens=2048, source="fixture", revision="a" * 40, kind="humaneval"
    )
    assert len(filtered) == 2 and dispositions[0]["reason"] == "rendered_input_too_long"
    assert len(source[0]["prompt"]) > 3000
    with pytest.raises(ValueError, match="no truncation/repetition"):
        data.build_records(
            source, Tokenizer(), count=3, max_input_tokens=2048, source="fixture", revision="a" * 40, kind="humaneval"
        )
    file = path.parent / "requests.jsonl"
    file.write_text(file.read_text().replace("Task 1", "Task 9"))
    with pytest.raises(ValueError, match="hash mismatch"):
        data.read_manifest(path)


@pytest.mark.parametrize("duplicate", ["id", "prompt"])
def test_duplicate_source_is_rejected(duplicate):
    raw = raw_tasks(2)
    raw[1]["task_id" if duplicate == "id" else "prompt"] = raw[0]["task_id" if duplicate == "id" else "prompt"]
    with pytest.raises(ValueError, match="Duplicate"):
        data.build_records(
            raw, Tokenizer(), count=1, max_input_tokens=2048, source="fixture", revision="a" * 40, kind="humaneval"
        )


def test_prompt_only_code_is_not_executable_quality():
    raw = [{"task_id": "x", "prompt": "real task with no provided tests"}]
    records, _ = data.build_records(
        raw, Tokenizer(), count=1, max_input_tokens=2048, source="fixture", revision="a" * 40, kind="code"
    )
    assert records[0]["tests"] is None
    with pytest.raises(ValueError, match="missing real HumanEval"):
        data.build_records(
            raw, Tokenizer(), count=1, max_input_tokens=2048, source="fixture", revision="a" * 40, kind="humaneval"
        )


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.1
        return self.value


class DeltaEngine:
    def __init__(self, chunks=(3, 2), fail=False):
        self.chunks = chunks
        self.fail = fail
        self.active = 0
        self.peak = 0
        self.ids = []

    async def generate(self, prompt, sampling_params, request_id):
        self.ids.append(request_id)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            for i, count in enumerate(self.chunks):
                await asyncio.sleep(0)
                if self.fail and i == 1:
                    raise RuntimeError("execution failed")
                yield SimpleNamespace(
                    finished=i == len(self.chunks) - 1,
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            token_ids=[i + 1] * count,
                            text="x" * count,
                            finish_reason="stop" if i == len(self.chunks) - 1 else None,
                            stop_reason=None,
                        )
                    ],
                )
        finally:
            self.active -= 1


@pytest.mark.parametrize("chunks", [(3, 2), (1,), (6,), (0, 2, 0)])
def test_real_stream_events_not_fabricated_token_timestamps(chunks):
    engine = DeltaEngine(chunks)
    batch = asyncio.run(stream.stream_batch(engine, [{"prompt_token_ids": [1]}], object(), None, "test", clock=Clock()))
    row = batch["requests"][0]
    assert batch["error"] is None
    assert [event["new_tokens"] for event in row["events"]] == [n for n in chunks if n]
    assert len(row["output_token_ids"]) == sum(chunks)
    assert row["ttft_seconds"] == row["first_output_monotonic"] - row["submitted_monotonic"]
    assert row["completion_seconds"] == row["completed_monotonic"] - row["submitted_monotonic"]
    if sum(chunks) > 1:
        assert row["mean_tpot_seconds"] == (row["completed_monotonic"] - row["first_output_monotonic"]) / (
            sum(chunks) - 1
        )
    else:
        assert row["mean_tpot_seconds"] is None


def test_outstanding_cap_and_two_generation_intervals():
    engine = DeltaEngine()
    prompts = [{"prompt_token_ids": [1]}] * 5
    first = asyncio.run(stream.stream_batch(engine, prompts, object(), 2, "warmup", clock=Clock()))
    second = asyncio.run(stream.stream_batch(engine, prompts, object(), 2, "measured", clock=Clock()))
    assert engine.peak == 2 and engine.active == 0
    assert first["requests"] is not second["requests"]
    assert all(row["request_id"].startswith("measured-") for row in second["requests"])
    assert first["requests"][0]["events"] is not second["requests"][0]["events"]


def test_stream_exception_preserves_partial_output_and_cancels_peers():
    engine = DeltaEngine(fail=True)
    batch = asyncio.run(
        stream.stream_batch(engine, [{"prompt_token_ids": [1]}] * 3, object(), 2, "test", clock=Clock())
    )
    assert "execution failed" in batch["error"]
    assert batch["requests"][0]["output_token_ids"] == [1, 1, 1]
    assert batch["requests"][0]["completed_monotonic"] is None
    assert engine.active == 0


def test_request_quantiles_and_sample_cv_are_different_populations():
    assert report.run_statistics([1])["sample_cv"] is None
    assert report.run_statistics([])["sample_cv"] is None
    stats = report.run_statistics([1, 3, 5])
    assert stats["sample_standard_deviation"] == 2 and stats["sample_cv"] == pytest.approx(2 / 3)
    assert report.request_distribution([None, 1, 3])["n"] == 2
    assert report.request_distribution([None])["mean"] is None
    assert report.request_distribution([1])["p99_sample_note"] is not None
    assert report.latency_summary({})["status"] == "unavailable"


def test_scheduler_collector_uses_real_cpu_stats_and_vector():
    collector = stream.SchedulerCollector(5)
    spec = SimpleNamespace(
        num_spec_tokens=5,
        num_drafts=2,
        num_draft_tokens=10,
        num_accepted_tokens=3,
        num_accepted_tokens_per_pos=[2, 1, 0, 0, 0],
        num_forwards=1,
        num_committed_tokens=5,
    )
    scheduler = SimpleNamespace(num_running_reqs=63, num_waiting_reqs=7, kv_cache_usage=0.2, spec_decoding_stats=spec)
    before = benchmark.capture_spec_metrics(collector.metrics())
    collector.record(scheduler, SimpleNamespace(num_preempted_reqs=1, num_corrupted_reqs=0))
    delta = benchmark.metric_snapshot_delta(before, benchmark.capture_spec_metrics(collector.metrics()), 5)
    acceptance = benchmark.acceptance_from_delta(delta, 5)
    assert acceptance["effective_acceptance_length"] == 2.5
    assert collector.rows[0]["num_running_reqs"] == 63 and collector.preemptions == 1
    assert collector.forwards == 1 and collector.committed == 5
    with pytest.raises(ValueError, match="single DP"):
        collector.record(scheduler, None, engine_idx=1)


def test_frozen_core_interfaces_are_not_variadic_mocks():
    core = Path(__file__).resolve().parents[3] / "vllm-hust"
    if not core.is_dir():
        pytest.skip("Frozen core checkout unavailable")
    source = ast.parse((core / "vllm/v1/engine/async_llm.py").read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AsyncLLM")
    methods = {node.name: node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert [arg.arg for arg in methods["generate"].args.args][:4] == ["self", "prompt", "sampling_params", "request_id"]
    assert isinstance(methods["collective_rpc"], ast.AsyncFunctionDef)
    assert [arg.arg for arg in methods["collective_rpc"].args.args] == ["self", "method", "timeout", "args", "kwargs"]
    assert isinstance(methods["shutdown"], ast.FunctionDef)
    assert "stat_loggers" in [arg.arg for arg in methods["from_engine_args"].args.args]
    logger_ast = ast.parse((core / "vllm/v1/metrics/loggers.py").read_text())
    base = next(node for node in logger_ast.body if isinstance(node, ast.ClassDef) and node.name == "StatLoggerBase")
    record = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == "record")
    assert [arg.arg for arg in record.args.args] == [
        "self",
        "scheduler_stats",
        "iteration_stats",
        "mm_cache_stats",
        "engine_idx",
    ]


def test_fixed_extractor_and_no_network_sandbox(tmp_path):
    assert code_eval.extract_code("Explanation\n```python\ndef f(x): return x\n```") == "def f(x): return x\n"
    with pytest.raises(ValueError):
        code_eval.extract_code("```python\nx=1\n```\n```python\nx=2\n```")
    command = code_eval.sandbox_command("python@sha256:" + "a" * 64, tmp_path, "own-container")
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--user=65534:65534", "--pull=never"):
        assert flag in command
    assert "--privileged" not in command
    with pytest.raises(ValueError, match="pinned"):
        code_eval.sandbox_command("python:latest", tmp_path, "own-container")
    assert code_eval.judge("pass", None, None)["status"] == "infrastructure_unavailable"


def test_judge_boundary_preserves_failure_and_timeout_cleanup():
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, 30)
        return SimpleNamespace(returncode=0)

    result = code_eval.judge("pass", None, "python@sha256:" + "a" * 64, run=run)
    assert result["status"] == "infrastructure_failure"
    assert calls[1] == ["docker", "rm", "-f", calls[0][calls[0].index("--name") + 1]]


def test_quality_denominators_do_not_hide_missing_tests_or_infrastructure():
    rows = [
        dict(status="passed", has_tests=True, syntax_pass=True),
        dict(status="timeout", has_tests=True, syntax_pass=True),
        dict(status="extraction_failure", has_tests=True, syntax_pass=None),
        dict(status="infrastructure_failure", has_tests=True, syntax_pass=None),
        dict(status="tests_unavailable", has_tests=False, syntax_pass=True),
    ]
    summary = code_eval.quality_summary(rows, 5)
    assert summary["status"] == "unavailable"
    assert summary["unit_test_task_denominator"] == 3 and summary["unit_test_pass_rate"] == 1 / 3
    assert summary["syntax_denominator"] == 3
    assert summary["all_task_pass_fraction"] is None


def test_pipe_status_failure_is_retained(tmp_path):
    log = tmp_path / "generation.log"
    rc = driver.logged(["bash", "-c", "printf 'partial output\\n'; false"], log)
    assert rc != 0
    assert log.read_text() == "partial output\n"
    assert log.with_suffix(".pipestatus").read_text().strip() == "1 0"


def test_resource_gate_cannot_ignore_busy_other_device():
    assert driver.npu_report_is_idle("| Process id | Process name |\nNo running processes found")
    assert not driver.npu_report_is_idle(
        "| Process id | Process name |\n| 0 0 | 1234 | python | 300 |\nNo running processes found"
    )
    assert not driver.npu_report_is_idle("unknown output format")


def test_plan_only_never_starts_an_engine_and_refuses_overwrite(tmp_path, monkeypatch):
    manifest, _ = frozen(tmp_path)
    out = tmp_path / "plan"
    argv = ["--plugin-sha", "a" * 40, "--manifest", str(manifest), "--output-dir", str(out), "--num-prompts", "4"]
    monkeypatch.setattr(driver, "source_gate", lambda _: pytest.fail("plan-only contacted runtime"))
    assert driver.main(argv) == 0
    assert not list(out.glob("*/result.json"))
    before = (out / "plan.json").read_bytes()
    assert driver.main(argv) == 1
    assert (out / "plan.json").read_bytes() == before
    data.read_manifest(out / "input/manifest.json", 4)
    summary = report.summarize_suite(out)
    assert summary["status"] == "incomplete" and all(row["status"] == "not_run" for row in summary["runs"])


def as_stream_result(result):
    result = copy.deepcopy(result)
    result.update(
        benchmark="dspark_additional_performance",
        measurement_protocol="async_llm_delta_stream_v1",
        performance_schema_version=1,
        delivery={"client_outstanding": None, "policy": "source_order_all_at_once"},
    )
    requests, records = [], []
    for i, output in enumerate(result["outputs"]):
        # Existing fake benchmark emits these real token IDs, not inferred timing.
        tokens = [1, 2, 3, 4]
        output["output_token_sha256"] = benchmark._sha256_bytes(benchmark._canonical_json_bytes(tokens))
        row = dict(
            request_id=f"r-{i}",
            request_index=i,
            error=None,
            submitted_monotonic=10.0,
            first_output_monotonic=11.0,
            completed_monotonic=12.0,
            finish_reason="stop",
            stop_reason=None,
            events=[{"monotonic": 11.0, "new_tokens": 3}, {"monotonic": 12.0, "new_tokens": 1}],
            output_token_ids=tokens,
            text="text",
        )
        row.update(stream.request_latency(row))
        requests.append(row)
        records.append({"prompt_token_sha256": output["prompt_token_sha256"]})
    result["streaming"] = {
        "requests": requests,
        "error": None,
        "started_monotonic": 10.0,
        "finished_monotonic": 12.0,
        "elapsed_seconds": 2.0,
        "scheduler": {},
    }
    return result, records


def test_stream_validation_reuses_real_graph_rank_gate(tmp_path, monkeypatch):
    result, _ = _run_graph(tmp_path, monkeypatch, "dspark")
    result, records = as_stream_result(result)
    assert report.validate_stream_result(result, records)["status"] == "available"
    broken = copy.deepcopy(result)
    broken["graph_execution"]["boundary_snapshots"][2].pop()
    with pytest.raises(RuntimeError):
        report.validate_stream_result(broken, records)
    broken = copy.deepcopy(result)
    broken["graph_execution"]["boundary_snapshots"][2] = broken["graph_execution"]["boundary_snapshots"][1]
    with pytest.raises(ValueError, match="Measured replay"):
        report.validate_stream_result(broken, records)


def test_old_eager_compatibility_and_false_timing_rejected(tmp_path, monkeypatch):
    result, _ = _run(tmp_path, monkeypatch, "target_only")
    assert report.latency_summary(result)["status"] == "unavailable"
    result, records = as_stream_result(result)
    assert report.validate_stream_result(result, records)["status"] == "available"
    result["streaming"]["requests"][0]["ttft_seconds"] = 0
    with pytest.raises(ValueError, match="latency"):
        report.validate_stream_result(result, records)


def test_async_facade_named_dispatch_delta_and_shutdown(monkeypatch):
    engine = DeltaEngine()
    engine.vllm_config = SimpleNamespace(marker="real-config-object")
    calls = []

    async def utility(method, *args):
        calls.append(("utility", method))
        return [{"group_idx": 0, "block_size": 32}]

    async def collective_rpc(method, timeout=None, args=(), kwargs=None):
        assert isinstance(method, str)
        calls.append(("rpc", method))
        return [{"rank": 0}]

    engine.engine_core = SimpleNamespace(call_utility_async=utility)
    engine.collective_rpc = collective_rpc
    engine.get_tokenizer = lambda: Tokenizer()
    engine.shutdown = lambda: calls.append(("shutdown",))

    class EngineArgs:
        def __init__(self, *, speculative_config):
            assert speculative_config is None

    class AsyncLLM:
        @classmethod
        def from_engine_args(cls, engine_args, *, stat_loggers):
            assert asyncio.get_running_loop().is_running()
            collector = stat_loggers[0](vllm_config=engine.vllm_config, engine_index=0)
            collector.log_engine_initialized()
            return engine

    modules = {
        "vllm.engine.arg_utils": {"AsyncEngineArgs": EngineArgs},
        "vllm.sampling_params": {"RequestOutputKind": SimpleNamespace(DELTA="delta")},
        "vllm.v1.engine.async_llm": {"AsyncLLM": AsyncLLM},
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    facade = stream.StreamingEngine(
        {"speculative_config": None}, SimpleNamespace(num_spec_tokens=5, client_outstanding=1)
    )
    assert facade.llm_engine.vllm_config is engine.vllm_config
    assert facade.call_utility("get_kv_cache_group_metadata")[0]["block_size"] == 32
    assert facade.collective_rpc("dspark_benchmark_replay_snapshot") == [{"rank": 0}]
    with pytest.raises(TypeError, match="named"):
        facade.collective_rpc(lambda: None)
    sampling = SimpleNamespace(output_kind="original")
    for _ in range(2):
        outputs = facade.generate([{"prompt_token_ids": [1]}], sampling)
        assert list(outputs[0].outputs[0].token_ids) == [1, 1, 1, 2, 2]
        assert facade.last_batch["error"] is None
    assert sampling.output_kind == "original"
    assert engine.ids == ["batch1-0", "batch2-0"]
    facade.shutdown()
    assert facade.loop.is_closed() and calls[-1] == ("shutdown",)


def test_summary_retains_all_lifecycles_pairs_and_failed_run(tmp_path):
    out = tmp_path / "suite"
    out.mkdir()
    runs = []
    for repeat in range(1, 4):
        for mode in ("target_graph", "dspark_graph"):
            name = f"{mode}-{repeat}"
            directory = out / name
            directory.mkdir()
            runs.append({"directory": name, "mode": mode, "repeat": repeat, "max_num_seqs": 4})
            latency = dict(ttft_seconds=1, completion_seconds=2, mean_tpot_seconds=0.5, event_intervals_seconds=[0.5])
            result = {
                "run_id": name,
                "streaming": {"requests": [latency], "scheduler": {}},
                "throughput": {
                    "output_tokens_per_second": repeat * (2 if mode == "dspark_graph" else 1),
                    "requests_per_second": 0.5,
                    "total_output_tokens": 4,
                },
                "timing": {"elapsed_seconds": 2},
                "measured_graph_replay_count": 3,
                "graph_execution": {},
                "acceptance": {
                    "num_drafts": 2,
                    "accepted_candidate_tokens_per_verification": 1,
                    "effective_acceptance_length": 2,
                },
                "outputs": [{"output_token_count": 4, "finish_reason": "stop", "output_token_sha256": name}],
            }
            benchmark._atomic_write_json(directory / "result.json", result)
            (directory / "generation.log").write_text("completed\n")
            benchmark._atomic_write_json(
                directory / "receipt.json",
                {
                    "status": "valid",
                    "quality": {"status": "unavailable"},
                    "result_sha256": benchmark._sha256_file(directory / "result.json"),
                    "log_sha256": benchmark._sha256_file(directory / "generation.log"),
                },
            )
    plan = {"runs": runs, "max_num_seqs": [4], "modes": ["target_graph", "dspark_graph"], "repeats": 3}
    benchmark._atomic_write_json(out / "plan.json", plan)
    summary = report.summarize_suite(out)
    assert summary["status"] == "valid" and len(summary["runs"]) == 6
    assert summary["comparisons"][0]["ratio_of_median_tok_s"] == 2
    assert [pair["speedup"] for pair in summary["comparisons"][0]["pairs"]] == [2, 2, 2]
    assert all(pair["output_tokens_equal"] is False for pair in summary["comparisons"][0]["pairs"])
    report.write_reports(summary, out / "summary.json", out / "summary.csv", out / "summary.md")
    benchmark._atomic_write_json(out / "dspark_graph-3/receipt.json", {"status": "failed", "error": "test failure"})
    summary = report.summarize_suite(out)
    assert summary["status"] == "incomplete" and len(summary["runs"]) == 6
    assert summary["comparisons"][0]["ratio_of_median_tok_s"] is None
    assert (out / "dspark_graph-3/result.json").is_file()


def test_logged_runs_are_new_process_lifecycles(tmp_path):
    pids = []
    for repeat in range(3):
        log = tmp_path / f"{repeat}.log"
        assert driver.logged([sys.executable, "-c", "import os; print(os.getpid())"], log) == 0
        pids.append(int(log.read_text()))
    assert len(set(pids)) == 3


def test_driver_execution_failure_keeps_partial_requests_and_stops(tmp_path, monkeypatch):
    manifest, _ = frozen(tmp_path)
    out = tmp_path / "failed-suite"
    monkeypatch.setattr(driver, "source_gate", lambda _: None)
    monkeypatch.setattr(driver, "resources_idle", lambda path: path.write_text("test resources idle"))
    calls = []

    def failed_generation(command, log):
        calls.append(command)
        log.write_text("Execution failed after partial output\n")
        log.with_suffix(".pipestatus").write_text("1 0\n")
        benchmark._atomic_write_json(
            log.parent / "partial-stream.json",
            {"requests": [{"completed_monotonic": 2.0}, {"completed_monotonic": None}, None, None]},
        )
        return 1

    monkeypatch.setattr(driver, "logged", failed_generation)
    assert (
        driver.main(
            [
                "--plugin-sha",
                "a" * 40,
                "--manifest",
                str(manifest),
                "--output-dir",
                str(out),
                "--num-prompts",
                "4",
                "--execute",
            ]
        )
        == 1
    )
    assert len(calls) == 1
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "incomplete" and len(summary["runs"]) == 6
    failed = summary["runs"][0]
    assert failed["status"] == "failed"
    assert failed["receipt"]["generation_diagnostics"] == {"completed": 1, "failed_or_cancelled": 1, "not_submitted": 2}
    assert (out / failed["directory"] / "partial-stream.json").is_file()
    assert all(row["status"] == "not_run" for row in summary["runs"][1:])
