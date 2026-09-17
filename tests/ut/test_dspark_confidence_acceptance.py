# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU real policy + host wrapper lifecycle; no Ascend/model execution claim."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tests.ut.test_dspark_confidence_verification import POLICY, load_module
from tests.ut.test_dspark_formal_cost import records_table
from tools.dspark import confidence_acceptance as task
from tools.dspark import performance_stream

ROOT = Path(__file__).parents[2]


@pytest.fixture
def observer(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(**POLICY))
    load_module(
        ROOT / "vllm_ascend/diagnostics/dspark_benchmark_worker.py",
        "vllm_ascend.diagnostics.dspark_benchmark_worker",
        monkeypatch,
    )
    module = load_module(
        ROOT / "vllm_ascend/diagnostics/dspark_confidence_receipts.py", "confidence_receipts_test", monkeypatch
    )
    runtime = load_module(
        ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/verification_runtime.py",
        "confidence_runtime_test",
        monkeypatch,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        NS(get_tp_group=lambda: NS(rank_in_group=0, broadcast_object=lambda packet, src: packet)),
    )
    return module, runtime


def runner_fixture(observer, tmp_path):
    module, runtime = observer
    table = records_table()
    state = runtime.ConfidenceVerification({"mode": "confidence", "profile": False}, None, "cpu")
    state.costs = POLICY["CostTable"].load_startup(table, table["identity"])
    state.rows = {"r": POLICY["ConfidenceRow"]("r", 7, (0.9,) * 5)}

    class Runner:
        def __init__(self):
            self.vllm_config = NS(
                additional_config={
                    "dspark_confidence_acceptance": True,
                    "dspark_profile_worker_exit": True,
                    "dspark_profile_shutdown_policy": task.shutdown_policy.POLICY_NAME,
                    "dspark_confidence_verification": state.options,
                    "dspark_profile_failure_dir": str(tmp_path),
                }
            )
            self.speculator = NS(
                confidence_verification=state, rank=0, _published_proposal_owners={"r": NS(producer_epoch=7)}
            )
            self.req_states = NS(
                req_id_to_index={"r": 0}, num_computed_tokens_np=np.array([32]), prefill_len=NS(np=np.array([16]))
            )
            self.cudagraph_manager = NS(run_fullgraph=lambda desc: "actual graph return")
            self.execute_model_state = None

        def execute_model(self, output, **kwargs):
            selected = state.select(self, output)
            if state.last_selection:
                cap = state.last_selection["selected_graph_capacity"]
                self.cudagraph_manager.run_fullgraph(NS(cg_mode="FULL", num_tokens=cap))
            else:
                cap = selected.total_num_scheduled_tokens
            self.execute_model_state = NS(
                input_batch=NS(
                    req_ids=list(selected.num_scheduled_tokens),
                    num_scheduled_tokens=list(selected.num_scheduled_tokens.values()),
                    num_tokens=selected.total_num_scheduled_tokens,
                    num_tokens_after_padding=cap,
                )
            )
            return selected

    runner = Runner()
    obs = module.ConfidenceReceipts(runner)
    return runner, state, obs


def test_real_policy_actual_dispatch_acceptance_reorder_snapshot_and_close(observer, tmp_path):
    runner, state, obs = runner_fixture(observer, tmp_path)
    original = NS(
        num_scheduled_tokens={"r": 6}, total_num_scheduled_tokens=6, scheduled_spec_decode_tokens={"r": [1, 2, 3, 4, 5]}
    )
    for epoch in (7, 8, 9):
        state.rows = {"r": POLICY["ConfidenceRow"]("r", epoch, (0.9,) * 5)}
        runner.speculator._published_proposal_owners["r"].producer_epoch = epoch
        selected = runner.execute_model(original)
        length = selected.num_scheduled_tokens["r"] - 1
        counts = torch.tensor([length + 1], dtype=torch.int32)
        state.accepted(("r",), (length,), counts, (epoch,))
        counts.fill_(999)  # device-owned source may be reused immediately
    saved = obs.snapshot()["confidence_execution_receipts"]["records"]
    assert [r["execution"] for r in saved] == [1, 2, 3]
    assert all(r["target"]["full_replay"] for r in saved)
    assert [r["accepted"]["producer_epochs"] for r in saved] == [[7], [8], [9]]
    assert all(r["accepted"]["num_sampled"] == [r["selection"]["lengths"]["r"] + 1] for r in saved)
    assert original.num_scheduled_tokens == {"r": 6}
    assert len((tmp_path / "confidence-rank-0.jsonl").read_text().splitlines()) == 9
    obs.close()
    assert "select" not in vars(state) and "accepted" not in vars(state)
    assert "execute_model" not in vars(runner) and not obs.sampled
    assert saved[0]["accepted"]["producer_epochs"] == [7]


def test_missing_epoch_and_capacity_fail_before_execution(observer, tmp_path):
    runner, state, obs = runner_fixture(observer, tmp_path)
    output = NS(
        num_scheduled_tokens={"r": 6}, total_num_scheduled_tokens=6, scheduled_spec_decode_tokens={"r": [1] * 5}
    )
    runner.req_states.num_computed_tokens_np[0] = 641
    with pytest.raises(ValueError, match="coverage"):
        runner.execute_model(output)
    runner.req_states.num_computed_tokens_np[0] = 32
    state.rows["r"] = POLICY["ConfidenceRow"]("r", 1, (0.9,) * 5)
    with pytest.raises(ValueError, match="Stale"):
        runner.execute_model(output)
    assert not obs.records
    obs.close()


def test_prefill_not_required_full_and_bound_fail_closed(observer, tmp_path, monkeypatch):
    module, _ = observer
    runner, state, obs = runner_fixture(observer, tmp_path)
    state.rows.clear()
    runner.req_states.num_computed_tokens_np[0] = 0
    runner.execute_model(
        NS(num_scheduled_tokens={"r": 16}, total_num_scheduled_tokens=16, scheduled_spec_decode_tokens={})
    )
    assert obs.records[0]["selection"] is None
    assert obs.records[0]["target"]["full_replay"] is False
    monkeypatch.setattr(module, "MAX_EXECUTIONS", 1)
    with pytest.raises(ValueError, match="bound"):
        runner.execute_model(
            NS(num_scheduled_tokens={"r": 16}, total_num_scheduled_tokens=16, scheduled_spec_decode_tokens={})
        )
    obs.close()


def receipts():
    table = records_table()
    table["loaded_confidence_weights"] = {"loaded_parameters": ["real.weight"], "weights_sha256": "hash"}
    pairs = [
        {"external_id": r["request_id"], "internal_id": f"internal-{i}"}
        for i, r in enumerate(task.plan()["inputs"]["records"])
    ]
    key = pairs[0]["internal_id"]
    cell = next(c for c in table["cells"] if c["requests"] == 1 and c["capacity"] == 6)
    seconds = cell["target_seconds"] + cell["draft_seconds"] + table["scheduler_seconds"]
    row = dict(
        execution=2,
        context_upper={key: 32},
        scheduled_queries={key: 6},
        selection=dict(
            mode="confidence",
            lengths={key: 5},
            producer_epochs={key: 7},
            confidence_epochs={key: 7},
            policy="current_epoch_survival_cost",
            actual_tokens=6,
            selected_graph_capacity=6,
            estimated_seconds=seconds,
        ),
        target=dict(request_ids=[key], query_lengths=[6], valid_tokens=6, capacity=6, full_replay=True),
        graph=dict(mode="FULL", capacity=6),
        accepted=dict(request_ids=[key], verified=[5], producer_epochs=[7], num_sampled=[3], status="available"),
        lookup=dict(
            requests=1, sampled_requests=1, capacity=6, context_ceiling=640, estimated_seconds=seconds, token_budget=6
        ),
        confidence={key: dict(producer_epoch=7, conditional=[0.9] * 5)},
    )
    all_ids = [p["internal_id"] for p in pairs]
    prefill = dict(
        execution=1,
        context_upper=dict.fromkeys(all_ids, 0),
        scheduled_queries=dict.fromkeys(all_ids, 1),
        selection=None,
        accepted=None,
        target=dict(request_ids=all_ids, query_lengths=[1] * 64, valid_tokens=64, capacity=64, full_replay=False),
    )
    ranks = [
        dict(
            rank=i,
            error=None,
            failed_execution_count=0,
            confidence_verification=dict(
                mode="confidence",
                specified_batches=0,
                confidence_head_calls=1,
                weights=table["loaded_confidence_weights"],
                cost_profile=dict(identity=table["identity"]),
            ),
            confidence_execution_receipts=dict(records=[copy.deepcopy(prefill), copy.deepcopy(row)], truncated=False),
        )
        for i in range(8)
    ]
    return ranks, dict(request_id_mapping=dict(mappings=pairs, errors=[], hook_restored=True)), table


def test_all_k5_valid_without_manufacturing_adaptation():
    ranks, stream, table = receipts()
    result = task.validate_receipts(ranks, stream, table)
    assert result["length_histogram"] == {5: 1}
    assert result["calibration"] == "uncalibrated"


@pytest.mark.parametrize(
    "failure",
    ["epoch", "query", "accepted", "graph", "bucket", "score", "rank", "mapping", "truncated", "mode", "identity"],
)
def test_invalid_or_stale_receipts_rejected(failure):
    ranks, stream, table = receipts()
    row = ranks[0]["confidence_execution_receipts"]["records"][-1]
    if failure == "epoch":
        row["accepted"]["producer_epochs"] = [6]
    if failure == "query":
        row["target"]["query_lengths"] = [1]
    if failure == "accepted":
        row["accepted"]["num_sampled"] = None
    if failure == "graph":
        row["target"]["full_replay"] = False
    if failure == "bucket":
        row["lookup"]["sampled_requests"] = 64
    if failure == "score":
        row["confidence"]["internal-0"]["conditional"][0] = float("nan")
    if failure == "rank":
        ranks.pop()
    if failure == "mapping":
        stream["request_id_mapping"]["errors"] = ["unknown"]
    if failure == "truncated":
        ranks[0]["confidence_execution_receipts"]["truncated"] = True
    if failure == "mode":
        ranks[0]["confidence_verification"]["mode"] = "specified_lengths"
    if failure == "identity":
        ranks[0]["confidence_verification"]["cost_profile"]["identity"] = {}
    with pytest.raises(ValueError):
        task.validate_receipts(ranks, stream, table)


def test_frozen_plan_and_no_performance_automatic_continuation():
    plan = task.plan()
    assert plan["request_count"] == 64 and plan["max_total_output_tokens"] == 16384
    assert plan["max_runtime_seconds"] == 3600 and not plan["profile"] and not plan["performance_eligible"]
    assert plan["inputs"]["sampling"]["ignore_eos"] is False
    assert plan["shutdown_budget"]["worker_seconds"] == 25
    for script in (
        "run_dspark_real_confidence.sh",
        "run_dspark_large_batch.sh",
        "run_dspark_exit_observation.sh",
        "run_dspark_swa_acceptance.sh",
    ):
        subprocess.run(["bash", "-n", str(ROOT / "tools/dspark" / script)], check=True)


@pytest.mark.parametrize("failure", [None, "hash", "proof", "producer"])
def test_publication_immutable_safe_load(tmp_path, monkeypatch, failure):
    table = records_table()
    table.update(
        plugin_sha=task.PRODUCER,
        core_sha=task.suite.CORE_SHA,
        publication={"status": "PASSED"},
        future_workload=task.plan()["inputs"],
    )
    if failure == "producer":
        table["plugin_sha"] = "bad"
    path = tmp_path / "cost-profile.json"
    path.write_text(json.dumps(table))
    digest = task.formal.sha(path)
    monkeypatch.setattr(task, "TABLE_SHA", digest)
    (tmp_path / "cost-publication.json").write_text(
        json.dumps(
            dict(table_sha256=digest, status="FAILED" if failure == "proof" else "PASSED", cost_table_usable=True)
        )
    )
    if failure == "hash":
        path.write_text(path.read_text() + " ")
    if failure:
        with pytest.raises(ValueError):
            task.publication(tmp_path)
    else:
        assert task.publication(tmp_path)[0]["plugin_sha"] == task.PRODUCER


def test_explicit_ids_and_natural_eos_preserved():
    import asyncio

    class Engine:
        async def generate(self, prompt, sampling, request_id):
            assert request_id == "frozen-id"
            yield NS(
                prompt_token_ids=[1, 2],
                finished=True,
                outputs=[NS(index=0, token_ids=[9], text="answer", finish_reason="stop", stop_reason=9)],
            )

    result = asyncio.run(
        performance_stream.stream_batch(
            Engine(), [{"prompt_token_ids": [1, 2]}], NS(), 1, "ignored", request_ids=["frozen-id"]
        )
    )
    assert result["requests"][0]["output_token_ids"] == [9]
    assert result["requests"][0]["finish_reason"] == "stop"
    with pytest.raises(ValueError):
        asyncio.run(performance_stream.stream_batch(Engine(), [{}], NS(), 1, "ignored", request_ids=[]))


def test_actual_benchmark_configuration_no_profile_no_heavy_probes(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "0")
    root = tmp_path / "runs/b64"
    root.mkdir(parents=True)
    args = NS(
        output_dir=tmp_path,
        plugin=ROOT,
        plugin_sha=task.PRODUCER,
        core=ROOT.parent / "vllm-hust",
        model=tmp_path / "model",
        manifest=tmp_path / "manifest",
    )
    parsed, kwargs = task.engine_config(args, root)
    options = kwargs["additional_config"]
    assert parsed.ignore_eos is False and parsed.output_len == 256 and parsed.num_prompts == 64
    assert kwargs["max_num_seqs"] == 64 and kwargs["speculative_config"]["enforce_eager"] is True
    assert kwargs["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert options["dspark_confidence_verification"]["mode"] == "confidence"
    assert options["dspark_confidence_verification"]["profile"] is False
    assert task.shutdown_policy.confidence_acceptance_enabled(options)
    assert set(options) == {
        "dspark_confidence_verification",
        "dspark_confidence_acceptance",
        "dspark_profile_failure_dir",
        "dspark_profile_worker_exit",
        "dspark_profile_shutdown_policy",
    }


@pytest.mark.parametrize("mutation", ["profile", "mode", "worker", "policy", "observation", "flag"])
def test_explicit_opt_in_cannot_bypass_exit_or_confidence_gates(mutation):
    config = dict(
        dspark_confidence_acceptance=True,
        dspark_profile_worker_exit=True,
        dspark_profile_shutdown_policy=task.shutdown_policy.POLICY_NAME,
        dspark_confidence_verification=dict(mode="confidence", profile=False),
    )
    if mutation == "profile":
        config["dspark_confidence_verification"]["profile"] = True
    if mutation == "mode":
        config["dspark_confidence_verification"]["mode"] = "specified_lengths"
    if mutation == "worker":
        config["dspark_profile_worker_exit"] = False
    if mutation == "policy":
        config["dspark_profile_shutdown_policy"] = None
    if mutation == "observation":
        config["dspark_profile_exit_observation"] = True
    if mutation == "flag":
        config["dspark_confidence_acceptance"] = False
    assert not task.shutdown_policy.confidence_acceptance_enabled(config)


@pytest.mark.parametrize("fault", ["child", "log", "cleanup"])
def test_parent_preserves_failures_and_does_not_retry(tmp_path, monkeypatch, fault):
    from tests.ut.test_dspark_formal_cost import setup_publication

    model, _ = setup_publication(tmp_path, monkeypatch)
    # Real named budget receipt validator, separate from generation result.
    task.write(model / "generation-result.json", dict(status="PASSED_THIS_RUN", error=None, cleanup_error=None))
    calls = []
    monkeypatch.setattr(task.suite, "resources_idle", lambda path: None)
    monkeypatch.setattr(task.suite, "logged", lambda cmd, path: calls.append(cmd) or int(fault == "child"))

    def scan(path):
        if fault == "log":
            raise ValueError("EngineDeadError")

    monkeypatch.setattr(task, "scan", scan)
    if fault == "cleanup":
        data = task.read(model / "cleanup.json")
        data["success"] = False
        task.write(model / "cleanup.json", data)
    monkeypatch.setattr(sys, "argv", ["tool", "run", "--plugin-sha", task.PRODUCER])
    assert task.supervise(NS(output_dir=tmp_path)) == 1
    report = task.read(tmp_path / "confidence-acceptance.json")
    assert not report["overall_pass"] and len(calls) == 1
    assert "--max-runtime-seconds" in calls[0] and "3600" in calls[0]
    assert report["generation"]["status"] == "PASSED_THIS_RUN"


def test_model_error_preserved_and_shutdown_runs_even_when_partial_save_fails(tmp_path, monkeypatch):
    args = NS(
        output_dir=tmp_path,
        plugin=ROOT,
        plugin_sha=task.PRODUCER,
        core=ROOT.parent / "vllm-hust",
        model=tmp_path / "model",
        manifest=tmp_path / "manifest",
    )
    (tmp_path / "input.jsonl").write_text(json.dumps({"prompt_token_ids": [1]}) + "\n")
    task.write(
        tmp_path / "preflight.json",
        dict(
            plan=task.plan(),
            publication={},
            plugin_sha=task.PRODUCER,
            input_sha256=task.formal.sha(tmp_path / "input.jsonl"),
        ),
    )
    monkeypatch.setattr(task.suite, "source_gate", lambda args: None)
    monkeypatch.setattr(task, "publication", lambda *args: ({}, {}))

    def fail_capture(*args):
        raise RuntimeError("original capture failure")

    monkeypatch.setattr(task.benchmark, "_collect_worker_graph_runtime", fail_capture)
    original_write = task.write

    def write(path, value):
        if path.name == "stream.json":
            raise OSError("disk unavailable")
        original_write(path, value)

    monkeypatch.setattr(task, "write", write)
    calls = []

    class Engine:
        def __init__(self, kwargs, parsed):
            self.profile_guard = NS(remember=lambda error: calls.append(str(error)))
            self.last_batch = {"requests": []}
            self.cleanup_result = {"success": True}

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(performance_stream, "StreamingEngine", Engine)
    assert task.model_run(args) == 1
    result = task.read(tmp_path / "runs/b64/generation-result.json")
    assert result["error"] == "RuntimeError: original capture failure"
    assert result["evidence_error"] == "OSError: disk unavailable"
    assert calls[-1] == "shutdown"
