# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Host observer tests. No tensor, NPU graph or permissive callable RPC fake."""

import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.dspark import benchmark_dspark_acceptance as benchmark

ROOT = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "p08_replay_extension", ROOT / "vllm_ascend/diagnostics/dspark_benchmark_worker.py"
)
_EXTENSION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXTENSION)


class ReplayGraph:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def replay(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("replay failed")


class ReplayManager:
    def __init__(self):
        self.graph = ReplayGraph()
        self.output = object()

    def run_fullgraph(self, desc):
        assert desc.cg_mode == "FULL"
        self.graph.replay()
        return self.output


class ReplayRunner:
    """CPU execution stand-in with the frozen core's explicit callable ABI."""

    def __init__(self):
        self.cudagraph_manager = ReplayManager()
        self.execute_model_state = None
        self.fail_after_replay = False
        self.output = object()
        self.forwarded = None

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
    ):
        self.forwarded = (scheduler_output, intermediate_tensors, dummy_run, skip_attn_for_dummy_run, is_profile)
        if scheduler_output.cg_mode == "FULL":
            self.cudagraph_manager.run_fullgraph(scheduler_output)
        if self.fail_after_replay:
            raise RuntimeError("execution failed after replay")
        self.execute_model_state = SimpleNamespace(
            input_batch=SimpleNamespace(
                num_tokens=scheduler_output.unpadded, num_tokens_after_padding=scheduler_output.num_tokens
            )
        )
        return self.output


def replay_worker(rank=0):
    worker = _EXTENSION.DSparkBenchmarkWorkerExtension()
    worker.rank, worker.model_runner = rank, ReplayRunner()
    return worker


def batch(unpadded=6, padded=6, mode="FULL"):
    return SimpleNamespace(cg_mode=mode, num_tokens=padded, unpadded=unpadded)


def snapshot(workers):
    return [worker.dspark_benchmark_replay_snapshot() for worker in workers]


def interval(start, end):
    return benchmark._replay_interval(SimpleNamespace(tensor_parallel_size=8), start, end)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_actual_replay_success_phase_isolation_and_tp_deduplication(size):
    workers = [replay_worker(rank) for rank in range(8)]
    start = snapshot(workers)
    for worker in workers:
        # Capture is already complete at installation; direct/dummy calls are
        # excluded even if they actually call replay after installation.
        runner = worker.model_runner
        runner.cudagraph_manager.run_fullgraph(batch())
        runner.execute_model(batch(), None, True, False, False)
        runner.execute_model(batch())  # Benchmark warmup, real request.
    warmup_end = snapshot(workers)
    for worker in workers:
        worker.model_runner.execute_model(batch(size * 6 - 1, size * 6))
        worker.model_runner.execute_model(batch(size * 6, size * 6))
    measured_end = snapshot(workers)
    assert interval(start, warmup_end)["graph_replay_count"] == 1
    measured = interval(warmup_end, measured_end)
    assert measured["graph_replay_count"] == 2  # Not 16 for TP8.
    assert len(measured["workers"]) == 8
    assert measured["eager_fallback_count"] is None
    assert [r["num_unpadded_tokens"] for r in measured["records"]] == [size * 6 - 1, size * 6]
    assert measured["records"][0]["num_paddings"] == 1
    assert interval(measured_end, snapshot(workers))["graph_replay_count"] == 0
    assert warmup_end[0]["excluded_dummy_replay_count"] == 1
    assert warmup_end[0]["excluded_unscoped_replay_count"] == 1


@pytest.mark.parametrize("fault", ["replay", "execute"])
def test_failed_execution_is_never_committed_and_next_call_is_fresh(fault):
    worker = replay_worker()
    start = worker.dspark_benchmark_replay_snapshot()
    runner = worker.model_runner
    if fault == "replay":
        runner.cudagraph_manager.graph.fail = True
    else:
        runner.fail_after_replay = True
    with pytest.raises(RuntimeError, match="failed"):
        runner.execute_model(batch())
    failed = worker.dspark_benchmark_replay_snapshot()
    assert failed["records"] == start["records"] == []
    assert failed["failed_execution_count"] == 1
    runner.cudagraph_manager.graph.fail = runner.fail_after_replay = False
    assert runner.execute_model(batch()) is runner.output
    assert worker.dspark_benchmark_replay_snapshot()["records"][0]["count"] == 1


def test_repeat_installation_preserves_wrappers_counts_return_and_arguments():
    worker = replay_worker()
    first = worker.dspark_benchmark_replay_snapshot()
    runner = worker.model_runner
    execute, full = runner.execute_model, runner.cudagraph_manager.run_fullgraph
    second = worker.dspark_benchmark_replay_snapshot()
    assert first == second and runner.execute_model == execute
    assert runner.cudagraph_manager.run_fullgraph == full
    descriptor, intermediate = batch(), object()
    assert runner.execute_model(descriptor, intermediate) is runner.output
    assert runner.forwarded == (descriptor, intermediate, False, False, False)
    assert worker.dspark_benchmark_replay_snapshot()["records"][0]["count"] == 1
    assert runner.cudagraph_manager.graph.calls == 1


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "duplicate_rank",
        "reset",
        "decreased",
        "rank_mismatch",
        "bad_shape",
        "failed",
        "observer_replaced",
    ],
)
def test_interval_rejects_incomplete_or_corrupt_evidence(fault):
    workers = [replay_worker(rank) for rank in range(8)]
    start = snapshot(workers)
    for worker in workers:
        worker.model_runner.execute_model(batch())
    end = snapshot(workers)
    if fault == "missing_rank":
        end.pop()
    elif fault == "duplicate_rank":
        end[-1]["rank"] = 0
    elif fault == "reset":
        end[-1]["observer_id"] = "reset"
    elif fault == "decreased":
        start, end = end, start
    elif fault == "rank_mismatch":
        end[-1]["records"] = []
    elif fault == "bad_shape":
        end[-1]["records"][0]["num_paddings"] = 1
    elif fault == "failed":
        end[-1]["failed_execution_count"] = 1
    else:
        workers[-1].model_runner.execute_model = lambda *args: None
        end = snapshot(workers)
    with pytest.raises(RuntimeError):
        interval(start, end)


def test_non_full_and_profile_calls_are_excluded_without_changing_results():
    worker = replay_worker()
    worker.dspark_benchmark_replay_snapshot()
    runner = worker.model_runner
    assert runner.execute_model(batch(mode="NONE")) is runner.output
    assert runner.execute_model(batch(), is_profile=True) is runner.output
    state = worker.dspark_benchmark_replay_snapshot()
    assert state["records"] == [] and state["excluded_dummy_replay_count"] == 1


def test_observer_signature_matches_frozen_core_and_never_reads_tensors():
    core_spec = importlib.util.find_spec("vllm")
    core = Path(next(iter(core_spec.submodule_search_locations))) if core_spec else ROOT.parent / "vllm-hust/vllm"
    source = ast.parse((core / "v1/worker/gpu/model_runner.py").read_text())
    method = next(
        node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    )
    signature = inspect.signature(_EXTENSION._FullReplayObserver.execute_model)
    assert list(signature.parameters) == [a.arg for a in method.args.args]
    assert [p.default for p in signature.parameters.values() if p.default is not inspect.Parameter.empty] == [
        ast.literal_eval(value) for value in method.args.defaults
    ]

    class NoHostCopy:
        def __int__(self):
            raise AssertionError("Must not convert device tensor to host")

    worker = replay_worker()
    worker.dspark_benchmark_replay_snapshot()
    assert worker.model_runner.execute_model(batch(NoHostCopy())) is worker.model_runner.output
    assert "host integers" in worker.dspark_benchmark_replay_snapshot()["error"]


def test_capture_and_dummy_only_cannot_pass_measured_gate():
    workers = [replay_worker(rank) for rank in range(8)]
    start = snapshot(workers)
    for worker in workers:
        worker.model_runner.cudagraph_manager.run_fullgraph(batch())
        worker.model_runner.execute_model(batch(), dummy_run=True)
    phase = interval(start, snapshot(workers))
    assert phase["graph_replay_count"] == 0
    with pytest.raises(RuntimeError, match="measured decode performed no target graph replay"):
        benchmark._graph_execution_result(
            SimpleNamespace(target_execution_mode="full_decode_only"),
            {"npugraph_ex_enabled": True, "static_kernel_enabled": False},
            {"npugraph_ex_enabled": True, "static_kernel_enabled": False, "graph_capture_count": 1},
            phase,
            phase,
        )
