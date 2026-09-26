# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts, actual streaming coroutine, observer ABI; no NPU claim."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_graph_replay import _EXTENSION
from tools.dspark import performance_comparison as task
from tools.dspark import performance_stream, shutdown_policy


@pytest.fixture
def observation(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_benchmark_worker", _EXTENSION)
    path = Path(__file__).parents[2] / "vllm_ascend/diagnostics/dspark_performance.py"
    spec = importlib.util.spec_from_file_location("performance_observer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(tmp_path, batch=64, mode="fixed"):
    return NS(
        output_dir=tmp_path,
        batch=batch,
        mode=mode,
        plugin_sha="a" * 40,
        model=Path("/model"),
        plugin=Path("."),
        core=Path("../vllm-hust"),
        manifest=tmp_path / "manifest.json",
    )


@pytest.mark.parametrize("batch", [64, 128, 256])
@pytest.mark.parametrize("mode", ["fixed", "confidence"])
def test_real_config_builder_keeps_comparability(tmp_path, monkeypatch, batch, mode):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0")
    monkeypatch.setenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", "0")
    parsed, kw = task.case_config(args(tmp_path, batch, mode), tmp_path)
    assert kw["max_num_seqs"] == batch and kw["enable_prefix_caching"] is False
    assert kw["compilation_config"]["cudagraph_capture_sizes"] == task.formal.captures(batch)
    assert kw["speculative_config"] == dict(method="dspark", num_speculative_tokens=5, enforce_eager=True)
    assert parsed.seed == 0 and parsed.output_len == 256 and not parsed.ignore_eos
    assert shutdown_policy.performance_enabled(kw["additional_config"])
    assert ("dspark_confidence_verification" in kw["additional_config"]) == (mode == "confidence")
    assert not kw["additional_config"].get("dspark_confidence_acceptance")


def test_plan_is_six_engines_and_bounded():
    p = task.plan()
    assert [(c["batch"], c["mode"]) for c in p["cases"]] == [
        (b, m) for b in (64, 128, 256) for m in ("fixed", "confidence")
    ]
    assert p["model_initializations"] == 6
    assert p["stage_sum_upper_seconds"] < p["total_limit_seconds"]
    assert sum(c["max_requests_total"] for c in p["cases"]) == 3584
    assert all(c["warmup_rounds"] == 1 and c["measured_rounds"] == 3 for c in p["cases"])


@pytest.mark.parametrize(
    "bad",
    [
        "dspark_profile_stack_signals",
        "dspark_profile_exit_debugger",
        "dspark_confidence_acceptance",
        "dspark_profile_exit_observation",
    ],
)
def test_passive_gate_rejects_active_diagnostics(bad):
    config = dict(
        dspark_performance_comparison=True,
        dspark_profile_worker_exit=True,
        dspark_profile_shutdown_policy=shutdown_policy.POLICY_NAME,
        dspark_profile_stack_signals=False,
        dspark_profile_exit_debugger=False,
    )
    assert shutdown_policy.performance_enabled(config)
    config[bad] = True
    assert not shutdown_policy.performance_enabled(config)


def runner_fixture():
    class Runner:
        def __init__(self):
            self.speculator = NS(confidence_verification=None, num_speculative_steps=5)
            self.vllm_config = NS(additional_config={})
            self.cudagraph_manager = NS(run_fullgraph=lambda desc: desc)
            self.epoch = 71

        def execute_model(self, layout, **kwargs):
            lengths, padded = layout
            self.epoch += 1
            self.cudagraph_manager.run_fullgraph(NS(cg_mode="FULL", num_tokens=padded))
            self.execute_model_state = NS(
                input_batch=NS(num_tokens=sum(lengths), num_tokens_after_padding=padded, num_scheduled_tokens=lengths)
            )
            return self.epoch

    return Runner()


def test_observer_records_actual_layout_and_resets_only_statistics(observation):
    r = runner_fixture()
    o = observation.PerformanceReplayObserver(r)
    for _ in range(4):
        assert r.execute_model(([1, 4, 6], 12)) == r.epoch
        result = o.snapshot()
        assert result["performance"]["layouts"] == [dict(requests=3, query_tokens=11, capacity=12, count=1)]
        assert result["query_layouts"] == []
        o.reset()
        assert o.snapshot()["performance"]["calls"] == 0
    assert r.epoch == 75
    assert result["performance"]["calls"] == 1  # detached host snapshot
    o.close()
    assert "execute_model" not in vars(r)


def test_observer_missing_layout_and_overflow_are_errors(observation, monkeypatch):
    r = runner_fixture()
    o = observation.PerformanceReplayObserver(r)
    r.execute_model(([7], 12))
    assert o.snapshot()["error"]
    with pytest.raises(ValueError):
        o.reset()
    o.close()
    r = runner_fixture()
    o = observation.PerformanceReplayObserver(r)
    monkeypatch.setattr(observation, "MAX_FULL_CALLS", 1)
    r.execute_model(([6], 6))
    r.execute_model(([6], 6))
    assert "capacity exceeded" in o.snapshot()["error"]


def test_real_delta_stream_multitoken_round_metrics(monkeypatch):
    ticks = iter(range(100, 1000))

    class Engine:
        async def generate(self, prompt, sampling, request_id):
            for tokens, finished in (([8, 9], False), ([10], True)):
                yield NS(
                    prompt_token_ids=prompt["prompt_token_ids"],
                    finished=finished,
                    outputs=[
                        NS(
                            index=0,
                            token_ids=tokens,
                            text="x",
                            finish_reason="stop" if finished else None,
                            stop_reason=None,
                        )
                    ],
                )

    stream = asyncio.run(
        performance_stream.stream_batch(
            Engine(),
            [{"prompt_token_ids": [1, 2]}],
            NS(),
            1,
            "test",
            clock=lambda: next(ticks),
            request_ids=["round-1:r"],
        )
    )
    stream["scheduler"] = {"corrupted_requests": 0}
    monkeypatch.setattr(task.formal, "workload_contract", lambda batch: {"records": [{"request_id": "r"}]})
    monkeypatch.setattr(task.acceptance, "plan", lambda batch: {"inputs": {"records": [{"request_id": "r"}]}})
    metrics = task.round_metrics(stream, [{"prompt_token_ids": [1, 2]}], 1, "round-1")
    row = stream["requests"][0]
    assert len(row["events"]) == 2 and metrics["actual_output_tokens"] == 3
    assert metrics["request_tpot_seconds"]["median"] == (row["completed_monotonic"] - row["first_output_monotonic"]) / 2
    assert metrics["output_tokens_per_second"] == 3 / stream["elapsed_seconds"]
    with pytest.raises(ValueError, match="identities"):
        task.round_metrics(stream, [{"prompt_token_ids": [1, 2]}], 1, "round-2")


def replay_rows(observation):
    rows = []
    for rank in range(8):
        r = runner_fixture()
        o = observation.PerformanceReplayObserver(r)
        initial = {"rank": rank, **o.snapshot()}
        r.execute_model(([6, 6], 12))
        rows.append((initial, {"rank": rank, **o.snapshot()}))
    return [r[0] for r in rows], [r[1] for r in rows]


def test_cross_rank_light_proof_and_wrong_baseline_fail(observation):
    before, after = replay_rows(observation)
    result = task.validate_replays(before, after, {}, "fixed")
    assert result["actual_query_tokens"] == 12 and result["FULL"]["calls"] == 1
    after[7]["performance"]["calls"] = 2
    with pytest.raises(ValueError, match="Cross-rank"):
        task.validate_replays(before, after, {}, "fixed")
    before, after = replay_rows(observation)
    after[0]["confidence_verification"] = {}
    with pytest.raises(ValueError, match="Adaptive"):
        task.validate_replays(before, after, {}, "fixed")


def test_cost_bytes_cannot_be_relabelled(tmp_path):
    (tmp_path / "cost-profile.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        task.table_at(tmp_path, 128)


def test_compatibility_rejects_new_execution_change(tmp_path, monkeypatch):
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"core": task.suite.CORE_SHA, "producers": {task.acceptance.PRODUCER: {}, task.PRODUCER: {}}})
    )
    monkeypatch.setattr(task, "COMPATIBILITY", contract)
    monkeypatch.setattr(task.subprocess, "check_output", lambda *a, **k: "vllm_ascend/attention/changed.py\n")
    with pytest.raises(ValueError, match="Unaudited"):
        task.verify_code(tmp_path)


def test_rounds_reuse_engine_and_stop_on_first_error(tmp_path, monkeypatch):
    class Engine:
        def __init__(self):
            self.engine = NS(output_processor=NS(get_num_unfinished_requests=lambda: 0), errored=False)
            self.last_batch = {}
            self.calls = []

        def collective_rpc(self, method):
            self.calls.append(method)
            return []

        def get_metrics(self):
            return []

        def generate(self, *a, **k):
            self.calls.append(k["request_ids"])
            if k["profile_point"] == "round-2":
                raise RuntimeError("first failure")

    engine = Engine()
    monkeypatch.setattr(task.formal, "workload_contract", lambda b: {"records": [{"request_id": "r"}]})
    monkeypatch.setattr(task.benchmark, "_sampling_params", lambda a: NS())
    monkeypatch.setattr(task.benchmark, "metric_snapshot_delta", lambda *a: {})
    monkeypatch.setattr(task, "acceptance_metrics", lambda *a: {})
    monkeypatch.setattr(task, "round_metrics", lambda *a: {})
    monkeypatch.setattr(task, "validate_replays", lambda *a: {})
    with pytest.raises(RuntimeError, match="first failure"):
        task.run_rounds(engine, NS(), args(tmp_path), [{"prompt_token_ids": [1]}], tmp_path, {})
    assert [x for x in engine.calls if isinstance(x, list)] == [["warmup:r"], ["round-1:r"], ["round-2:r"]]
    assert (tmp_path / "round-1-metrics.json").exists() and not (tmp_path / "round-3-metrics.json").exists()


def test_quantiles_and_singleton():
    assert task.distribution([1, 2, 3, 4])["p95_nearest_rank"] == 4
    assert task.distribution([])["median"] is None
    assert task.distribution([1])["stdev"] is None


def test_binary_identity_cannot_change_between_preflight_and_execution():
    baseline = dict(artifact_truncated=False, artifacts=[dict(path="/opp/operator.so", sha256="a")])
    ranks = [dict(binaries=baseline) for _ in range(8)]
    assert task.validate_binaries(ranks, baseline) == {"/opp/operator.so": "a"}
    bad = dict(artifact_truncated=False, artifacts=[dict(path="/opp/operator.so", sha256="b")])
    with pytest.raises(ValueError, match="OPP changed"):
        task.validate_binaries([dict(binaries=bad) for _ in range(8)], baseline)
    with pytest.raises(ValueError, match="Rank OPP"):
        task.validate_binaries([*ranks[:7], dict(binaries=bad)], baseline)


def test_partial_evidence_error_still_shuts_engine_down(tmp_path, monkeypatch):
    a = args(tmp_path)
    inputs = tmp_path / "assets/b64.jsonl"
    inputs.parent.mkdir()
    inputs.write_text('{"prompt_token_ids": [1]}\n')
    task.write(
        tmp_path / "preflight.json",
        dict(plugin_sha=a.plugin_sha, plan=task.plan(), inputs_sha256={"64": task.formal.sha(inputs)}),
    )
    monkeypatch.setattr(task.suite, "source_gate", lambda _: None)
    monkeypatch.setattr(task, "table_at", lambda *a: ({"identity": {}}, {}))
    monkeypatch.setattr(task, "case_config", lambda *a: (NS(), {}))
    events = []

    class Engine:
        def __init__(self, *args):
            self.last_batch = {"error": "original"}
            self.profile_guard = NS(remember=lambda e: events.append(str(e)))
            self.cleanup_result = {"success": True}

        def collective_rpc(self, *a):
            raise RuntimeError("original RPC failure")

        def shutdown(self):
            events.append("shutdown")

    monkeypatch.setattr(performance_stream, "StreamingEngine", Engine)
    real_write = task.write

    def write(path, value):
        if path.name == "last-stream.json":
            raise OSError("disk write failure")
        real_write(path, value)

    monkeypatch.setattr(task, "write", write)
    assert task.model_run(a) == 1
    saved = task.read(tmp_path / "runs/b64-fixed/generation-result.json")
    assert saved["error"] == "RuntimeError: original RPC failure"
    assert saved["evidence_error"] == "OSError: disk write failure"
    assert "shutdown" in events


def test_group_stops_after_invalid_case_and_preserves_previous(tmp_path, monkeypatch):
    a = args(tmp_path)
    monkeypatch.setattr(task.suite, "logged", lambda *a: 0)
    (tmp_path / "host.xml").write_text('<testsuite><testcase name="host"/></testsuite>')
    calls = []

    def supervise(args, case):
        calls.append(case["name"])
        return dict(case=case, valid=len(calls) == 1, error=None if len(calls) == 1 else "failed")

    monkeypatch.setattr(task, "supervise", supervise)
    monkeypatch.setattr(task, "summarize", lambda root, cases: dict(all_six_valid=False, cases=cases))
    assert task.run(a) == 1
    assert calls == ["b64-fixed", "b64-confidence"]
    saved = task.read(tmp_path / "performance-summary.json")
    assert saved["cases"][0]["valid"] and not saved["cases"][1]["valid"]


def test_named_rpc_installs_light_observer_once_and_resets_between_rounds(observation, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_performance", observation)
    r = runner_fixture()
    r.vllm_config.additional_config = dict(
        dspark_performance_comparison=True,
        dspark_profile_worker_exit=True,
        dspark_profile_shutdown_policy=shutdown_policy.POLICY_NAME,
        dspark_profile_stack_signals=False,
        dspark_profile_exit_debugger=False,
    )
    worker = _EXTENSION.DSparkBenchmarkWorkerExtension()
    worker.model_runner = r
    worker.rank = 3
    first = worker.dspark_benchmark_performance_reset()
    for _ in range(4):
        r.execute_model(([1, 6], 12))
        final = worker.dspark_benchmark_replay_snapshot()
        assert final["observer_id"] == first["observer_id"] and final["performance"]["calls"] == 1
        assert "cost_profile" not in final and "confidence_execution_receipts" not in final
        assert worker.dspark_benchmark_performance_reset()["performance"]["calls"] == 0
    assert r.epoch == 75


def test_zero_selected_candidates_do_not_fabricate_acceptance_rate():
    totals = {
        "vllm:spec_decode_num_draft_tokens": 0,
        "vllm:spec_decode_num_accepted_tokens": 0,
        task.benchmark.VECTOR_METRIC_NAME: [0] * 5,
    }
    assert task.acceptance_metrics({"totals": totals}, "confidence")["accepted_per_verified"] is None
    totals["vllm:spec_decode_num_accepted_tokens"] = 1
    with pytest.raises(ValueError, match="without verified"):
        task.acceptance_metrics({"totals": totals}, "confidence")
