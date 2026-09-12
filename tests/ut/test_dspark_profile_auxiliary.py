# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ATen replay of real output-copy statements, never an NPU claim."""

import ast
import json
import runpy
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.ut.test_dspark_nan_diagnostics import _NAN
from tests.ut.test_dspark_profile_observation import OBS, ROOT, Runner, torch
from tests.ut.test_dspark_replay_diagnostics import CPURecordedGraph, _production_method, replay


def load_auxiliary(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_nan", _NAN)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_observation", OBS)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_replay", replay)
    return NS(**runpy.run_path(str(ROOT / "vllm_ascend/diagnostics/dspark_profile_auxiliary.py")))


def output_copy_body():
    path = ROOT.parent / "vllm-hust/vllm/v1/worker/gpu/cudagraph_utils.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelCudaGraphManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
    closure = next(n for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n.name == "forward_fn")
    copy = next(
        n
        for n in closure.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr == "is_last_pp_rank"
    )
    function = ast.parse("def transfer(self, num_tokens, model_output): pass").body[0]
    function.body = [copy]
    ns = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), ns)
    return ns["transfer"]


def fixture(tmp_path, monkeypatch, *, capacity=12, mode="auxiliary-transfers", target=None):
    module = load_auxiliary(monkeypatch)
    if not hasattr(torch, "npu"):
        monkeypatch.setattr(torch, "npu", NS(is_current_stream_capturing=lambda: False), raising=False)
    else:
        monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    runner = Runner()
    spec = runner.speculator
    spec.target_layer_ids = (40, 41, 42)
    ids = torch.zeros(capacity, dtype=torch.int32)
    positions = torch.arange(capacity)
    query = torch.zeros(capacity + 1, dtype=torch.int32)
    seq = torch.zeros(capacity, dtype=torch.int32)
    padding = torch.zeros(capacity, dtype=torch.bool)
    runner.input_buffers = NS(input_ids=ids, positions=positions, is_padding=padding)
    raw_values = [torch.ones(capacity, 4) * i for i in (1, 2, 3)]
    corruption = torch.zeros(capacity, 4)
    manager = NS(
        model_runner=runner,
        hidden_states=None,
        aux_hidden_states=[],
        use_aux_hidden_state_outputs=True,
        is_last_pp_rank=True,
    )
    capture = module.AuxiliaryCapture(manager) if target is None else target.module.TargetCapture(manager, target.bank)
    manager._dspark_auxiliary_capture = capture
    runner.cudagraph_manager = manager
    tree = ast.parse((ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelWithContext")
    ns = {"torch": torch, "nn": torch.nn, "_EXTRA_CTX": NS(capturing=False)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "real_model_wrapper", "exec"), ns)
    calls = []

    def model(**kwargs):
        calls.append("target Python forward")
        if target is not None:
            return target.forward(raw_values, **kwargs)
        return torch.ones(capacity, 4), [x + 1 for x in raw_values]

    wrapper = ns["ModelWithContext"](model, replay_diagnostics=capture)
    transfer = output_copy_body()

    def closure():
        output = wrapper(input_ids=ids, positions=positions)
        transfer(manager, capacity, output)
        manager.aux_hidden_states[1].add_(corruption)  # injected transfer/write corruption, normally zero

    closure()  # allocate outside capture, as real Core warmup does
    graph = CPURecordedGraph()
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    with graph:
        closure()
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    desc = type("Descriptor", (), {"num_tokens": capacity, "cg_mode": NS(name="FULL")})()
    manager.graphs = {desc: graph}
    metadata = NS(
        query_start_loc=query,
        seq_lens=seq,
        decode=NS(query_start_loc=query, seq_lens=seq, positions=positions, input_positions=positions, start_pos=seq),
    )
    capture.finish_capture({desc: NS(captured=NS(attn_metadata={"target.attn": metadata}))})
    actual_replay = _production_method(
        ROOT.parent / "vllm-hust/vllm/v1/worker/gpu/cudagraph_utils.py",
        "CudaGraphManager",
        "run_fullgraph",
        {"CUDAGraphMode": NS(FULL=desc.cg_mode), "get_offloader": lambda: NS(sync_prev_onload=lambda: None)},
    )
    manager.run_fullgraph = lambda d: actual_replay(manager, d)

    def execute(scheduler, **kwargs):
        if target is not None and getattr(target, "before_execute", None) is not None:
            target.before_execute(scheduler)
        names, lengths = list(scheduler.num_scheduled_tokens), list(scheduler.num_scheduled_tokens.values())
        n, t = len(names), sum(lengths)
        offsets = [0, *np.cumsum(lengths).tolist()]
        query.fill_(t)
        query[: n + 1] = torch.tensor(offsets)
        seq.zero_()
        seq[:n] = torch.arange(n) + 223
        padding.copy_(torch.arange(capacity) >= t)
        runner.input_batch = NS(
            req_ids=names,
            num_reqs=n,
            num_tokens=t,
            num_tokens_after_padding=capacity,
            num_reqs_after_padding=capacity,
            query_start_loc_np=np.array(offsets),
            query_start_loc=query,
            seq_lens_np=seq[:n].numpy().copy(),
            seq_lens=seq,
            num_scheduled_tokens=np.array(lengths),
            is_prefilling_np=np.zeros(n, dtype=bool),
            idx_mapping_np=np.arange(n)[::-1].copy(),
            idx_mapping=torch.arange(n).flip(0),
            num_computed_tokens_np=np.ones(n, dtype=int) * 220,
            input_ids=ids,
            positions=positions,
        )
        manager.run_fullgraph(desc)
        runner.execute_model_state = NS(
            input_batch=runner.input_batch,
            hidden_states=manager.hidden_states,
            aux_hidden_states=manager.aux_hidden_states,
        )

    runner.execute_model = execute
    spec.prepare_proposal_inputs = lambda p: p
    spec.model = NS(
        combine_hidden_states=lambda x: x.reshape(-1, 3, 4).sum(1),
        precompute_and_store_context_kv=lambda *a: None,
        compute_draft_logits=lambda h: h.clone(),
    )

    def markov(proposal_inputs, hidden_states):
        spec._markov_attempt_step_epoch = proposal_inputs.step_epoch
        logits = spec.model.compute_draft_logits(hidden_states)
        if torch.isnan(logits).any():
            assert (tmp_path / "rank-0-first-nan.json").exists()
            raise RuntimeError("original Markov NaN")
        return logits

    spec._execute_sequential_markov_sampling = markov
    observer_cls = module.AuxiliaryProfileObservation if mode == "auxiliary-transfers" else OBS.ProfileObservation
    if target is not None:
        observer_cls = target.module.TargetProfileObservation
    observer = observer_cls(runner, {"mode": mode, "directory": str(tmp_path)})
    observer.begin_point("test-point")

    def run(epoch, lengths=(1, 4, 6), names=("third", "second", "first"), mutate=None, proposal=True):
        runner.execute_model(NS(num_scheduled_tokens=dict(zip(names, lengths)), finished_req_ids=[]))
        if mutate:
            mutate(manager.aux_hidden_states)
        if not proposal:
            return
        batch = runner.input_batch
        spec._proposal_step_epoch = epoch
        p = NS(
            step_epoch=epoch,
            rank=0,
            request_ids=names,
            num_reqs=len(names),
            num_target_tokens=sum(lengths),
            num_query_tokens=len(names) * 5,
            num_speculative_tokens=5,
            auxiliary_hidden_states=tuple(manager.aux_hidden_states),
            target_query_start_loc=query,
            target_positions=positions,
            target_sequence_lengths=seq,
            request_state_indices=batch.idx_mapping,
            num_sampled=torch.tensor(lengths),
            num_rejected=torch.zeros(len(names), dtype=torch.int32),
        )
        spec.prepare_proposal_inputs(p)
        context = spec.model.combine_hidden_states(
            torch.cat([x[: sum(lengths)] for x in p.auxiliary_hidden_states], dim=-1)
        )
        hidden = context[query[: len(names)]].repeat_interleave(5, dim=0)
        spec._execute_sequential_markov_sampling(p, hidden)

    return NS(**locals())


@pytest.mark.parametrize("where", ["raw", "persistent", "consumed"])
def test_actual_replay_distinguishes_raw_transfer_and_later_corruption(tmp_path, monkeypatch, where):
    f = fixture(tmp_path, monkeypatch)
    for epoch in (80, 81, 82):
        f.run(epoch)
    mutate = None
    if where == "raw":
        f.raw_values[1][0].fill_(torch.nan)
    elif where == "persistent":
        f.corruption[0].fill_(torch.nan)
    else:
        mutate = lambda aux: aux[1][0].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="original Markov NaN"):
        f.run(83, mutate=mutate)
    data = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    rounds = data["auxiliary"]["rounds"]
    assert [x["execution"] for x in rounds] == [2, 3, 4]
    assert [x["proposal_epoch"] for x in rounds] == [81, 82, 83]
    r = rounds[-1]
    assert r["raw_replay_verified"] and r["coverage"] == "FULL" and r["missing_boundaries"] == []
    assert r["target_mapping_matches_device"] is True and data["recording_error"] is None
    bad = [b["boundary"] for b in r["boundaries"] if any(x["nan"] for x in b["rows"])]
    assert bad == (
        ["raw", "persistent", "consumed"]
        if where == "raw"
        else ["persistent", "consumed"]
        if where == "persistent"
        else ["consumed"]
    )
    first = next(b for b in r["boundaries"] if b["boundary"] == where and b["layer"] == 41)["rows"][0]
    assert first["request_id"] == "third" and first["request_row"] == 0 and first["position_in_request"] == 0
    assert r["device_integers"]["raw_receipts"] == r["device_integers"]["consume_receipts"] == [4] * 3
    assert len(f.calls) == 2 and f.graph.calls == 4  # no Python forward on replay
    assert data["numeric"]["compact_host_transfers_completed"] == 4 and not data["performance_eligible"]
    f.observer.close()


def test_one_packet_padding_inf_and_owned_statistics(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.raw_values[0][-1].fill_(torch.nan)  # graph padding does not latch a valid-row anomaly
    transfers = []
    original = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            transfers.append(tuple(tensor.shape))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    f.run(12)
    assert len(transfers) == 1 and not (tmp_path / "rank-0-auxiliary-first-nan.json").exists()
    record = f.observer.auxiliary_records[-1]
    assert record["boundaries"][0]["rows"][-1]["request_id"] is None
    assert record["boundaries"][0]["rows"][-1]["nan"]
    f.raw_values[0][1].fill_(-torch.inf)
    f.run(13)
    data = json.loads((tmp_path / "rank-0-auxiliary-first-nonfinite.json").read_text())
    assert data["auxiliary"]["rounds"][-1]["boundaries"][0]["rows"][1]["request_id"] == "second"
    frozen = json.dumps(f.observer.auxiliary_records[-1])
    f.raw_values[0].zero_()
    f.manager.aux_hidden_states[0].zero_()
    f.query.zero_()
    assert json.dumps(f.observer.auxiliary_records[-1]) == frozen
    f.observer.close()


@pytest.mark.parametrize("fault", ["noop", "missing_copy", "throw", "later_receipt"])
def test_capture_values_cannot_pass_as_current_replay_evidence(tmp_path, monkeypatch, fault):
    f = fixture(tmp_path, monkeypatch)
    if fault == "noop":
        monkeypatch.setattr(f.graph, "replay", lambda: None)
    elif fault == "missing_copy":
        f.graph.operations = [
            op
            for op in f.graph.operations
            if not ("copy_" in str(op[0]) and op[1][0].numel() == 1 and op[1][0].dtype == torch.int64)
        ]
    elif fault == "throw":

        def bad():
            raise RuntimeError("original replay failure")

        monkeypatch.setattr(f.graph, "replay", bad)
    mutate = (lambda _: f.capture.shapes[12]["receipts"].fill_(999)) if fault == "later_receipt" else None
    if fault == "throw":
        with pytest.raises(RuntimeError, match="original replay failure"):
            f.run(3)
        r = json.loads((tmp_path / "rank-0-first-failure.json").read_text())["auxiliary"]["rounds"][-1]
    else:
        f.run(3, mutate=mutate)
        r = f.observer.auxiliary_records[-1]
    assert r["coverage"] == "INVALID_RECEIPT" and f.observer.recording_error
    with pytest.raises(RuntimeError, match="evidence unavailable"):
        f.observer.finish_point()
    f.observer.close()


def test_partial_no_proposal_history_reorder_and_point_reset(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.run(8, proposal=False)
    f.run(9, lengths=(4, 1, 6), names=("second", "third", "first"))
    previous, current = f.observer.auxiliary_records
    assert not previous["consumption_reached"] and previous["proposal_epoch"] is None
    assert current["request_ids"] == ["second", "third", "first"]
    assert current["boundaries"][0]["rows"][1]["request_id"] == "second"
    assert f.observer.recording_error is None
    receipt = f.observer.finish_point()
    assert "rounds" not in receipt["auxiliary"]
    f.observer.begin_point("next")
    assert not f.observer.auxiliary_records and f.observer.pending is None
    f.observer.close()
    assert f.capture.observer is None


def test_missing_consume_boundary_does_not_report_finite_success(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.runner.speculator.model.combine_hidden_states = lambda x: x.reshape(-1, 3, 4).sum(1)
    f.run(1)
    assert "Missing auxiliary boundaries" in f.observer.recording_error
    assert "consumed.40" in f.observer.auxiliary_records[-1]["missing_boundaries"]
    f.observer.close()


def test_finite_value_difference_is_retained_without_claiming_nan(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.corruption[0].fill_(0.5)
    f.run(10)
    data = json.loads((tmp_path / "rank-0-auxiliary-first-difference.json").read_text())
    assert not (tmp_path / "rank-0-auxiliary-first-nan.json").exists()
    assert data["auxiliary"]["counts"]["nan_rounds"] == 0
    r = data["auxiliary"]["rounds"][-1]
    b = next(b for b in r["boundaries"] if b["boundary"] == "persistent" and b["layer"] == 41)
    assert b["rows"][0]["differs_from_raw"] and not b["rows"][0]["nan"]
    f.observer.close()


def test_two_request_layout_and_shape_banks_are_separate(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch, capacity=6)
    f.run(5, lengths=(1, 4), names=("left", "right"))
    r = f.observer.auxiliary_records[-1]
    assert r["target_rows"] == 5 and r["graph_capacity"] == 6
    assert r["boundaries"][-1]["shape"][0] == 5
    old = f.capture.shapes[6]["sources"][0].clone()
    f.capture.model_inputs({"input_ids": torch.zeros(12, dtype=torch.int32), "positions": torch.arange(12)})
    f.capture.model_outputs(12, (torch.zeros(12, 4), [torch.ones(12, 4) * 99 for _ in range(3)]))
    assert torch.equal(old, f.capture.shapes[6]["sources"][0])
    assert f.capture.shapes[6]["sources"][0].data_ptr() != f.capture.shapes[12]["sources"][0].data_ptr()
    assert len(f.capture.shapes) == 2
    f.observer.close()


def test_proposal_identity_mismatch_and_storage_error_preserve_model_error(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    original = f.observer.bind_proposal

    def stale(p):
        p.step_epoch += 1
        return original(p)

    monkeypatch.setattr(f.observer, "bind_proposal", stale)
    f.run(6)
    assert f.observer.recording_error
    with pytest.raises(RuntimeError, match="evidence unavailable"):
        f.observer.finish_point()
    assert f.observer.auxiliary_records[-1]["proposal_epoch"] is None
    f.observer.close()
    other = fixture(tmp_path / "other", monkeypatch)

    def failed(*a, **kw):
        raise OSError("disk unavailable")

    monkeypatch.setattr(other.observer, "write", failed)

    def graph_failed():
        raise RuntimeError("original replay failure")

    monkeypatch.setattr(other.graph, "replay", graph_failed)
    with pytest.raises(RuntimeError, match="original replay failure"):
        other.run(7)
    assert other.observer.pending is None
    other.observer.close()


@pytest.mark.parametrize("mode", ["auxiliary-transfers", "target-boundaries"])
def test_profiler_factory_uses_capture_collector_and_no_draft_layer_hooks(tmp_path, monkeypatch, mode):
    if mode == "target-boundaries":
        from tests.ut.test_dspark_profile_target import target_fixture

        f = target_fixture(tmp_path, monkeypatch)
        monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_target", f.target_module)
    else:
        f = fixture(tmp_path, monkeypatch)
    f.observer.close()
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_auxiliary", f.module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(COST_CONTEXT_SEMANTICS="test"))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.worker.v2.spec_decode.dspark.verification_runtime", NS(runtime_identity=lambda *a: {})
    )
    monkeypatch.setattr(
        torch.npu, "Event", lambda **kw: NS(record=lambda: None, elapsed_time=lambda e: 1), raising=False
    )
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(torch.npu, "get_device_name", lambda d: "mock", raising=False)
    f.runner.vllm_config.additional_config["dspark_profile_observation"] = {
        "mode": mode,
        "directory": str(tmp_path),
    }
    f.runner.vllm_config.model_config = NS(max_model_len=8192)
    f.spec.confidence_verification.receipt = {"weights_sha256": "test"}
    f.spec._execute_draft = lambda p: p
    f.runner.device = "cpu"
    cls = runpy.run_path(str(ROOT / "vllm_ascend/diagnostics/dspark_cost_profile.py"))["IsolatedCostProfiler"]
    profiler = cls(f.runner)
    profiler.observation.begin_point("factory")
    f.run(5)
    result = profiler.snapshot()
    assert result["identity"]["diagnostic_only"] is True
    assert result["observation"]["auxiliary"]["counts"]["FULL"] == 1
    assert "upstream" not in result["observation"]
    assert f.capture.observer is profiler.observation
    profiler.observation.close()


def test_non_full_consumption_is_explicitly_unobserved_at_raw_boundary(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    # An eager target returns auxiliaries without calling the FULL proxy.
    monkeypatch.setattr(f.manager, "run_fullgraph", lambda d: None)
    f.run(20)
    r = f.observer.auxiliary_records[-1]
    assert r["coverage"] == "NON_FULL_RAW_UNOBSERVED" and not r["raw_replay_verified"]
    assert {b["boundary"] for b in r["boundaries"]} == {"consumed"}
    assert f.observer.recording_error is None
    f.observer.close()


def test_point_change_flushes_first_anomaly_without_a_following_proposal(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.raw_values[1][0].fill_(torch.nan)
    f.run(8, proposal=False)
    assert f.observer.pending is not None
    f.observer.begin_point("next")
    data = json.loads((tmp_path / "rank-0-auxiliary-first-nan.json").read_text())
    record = data["auxiliary"]["rounds"][-1]
    assert record["point"] == "test-point" and record["proposal_epoch"] is None
    assert record["raw_replay_verified"] and not record["consumption_reached"]
    assert not f.observer.auxiliary_records and f.observer.pending is None
    f.observer.close()
