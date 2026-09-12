# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tensors and the real draft forward/HC bodies; no NPU claim."""

import ast
import json
import runpy
import sys
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_observation import OBS, ROOT, Runner, torch


def source_method(name):
    tree = ast.parse((ROOT / "vllm_ascend/models/deepseek_v4_dspark.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4DSparkModel")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), "draft_source_body", "exec"), namespace)
    return namespace[name]


class Layer(torch.nn.Module):
    def __init__(self, index, inject):
        super().__init__()
        self.inject = inject
        self.self_attn = NS(dsa_attn=NS(swa_cache_layer=NS(prefix=f"mtp.{index}.swa_cache")))
        self.index = index

    def forward(self, positions, hidden_states, residual, **kwargs):
        return self.inject(f"mtp.{self.index}.output", hidden_states + 1), hidden_states


class Backbone(torch.nn.Module):
    forward = source_method("forward")
    real_hc = source_method("hc_head")

    def __init__(self, inject):
        super().__init__()
        self.inject = inject
        self.hc_mult, self.norm_eps, self.hc_eps = 2, 1e-6, 1e-6
        self.hc_head_fn, self.hc_head_scale, self.hc_head_base = torch.ones(2, 8), torch.ones(1), torch.zeros(2)
        self.layers = torch.nn.ModuleDict({str(44 + i): Layer(i, inject) for i in range(3)})

    def embed_tokens(self, ids):
        return self.inject("draft.initial_hidden", torch.ones(len(ids), 4))

    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        return self.inject("draft.hc_output", self.real_hc(x, hc_fn, hc_scale, hc_base))

    def _store_standard_swa_kv(self, shared_kv, slot_mapping, attn):
        # Consumer may reuse the buffer; flags must already own its state.
        shared_kv.fill_(0)


class Draft(torch.nn.Module):
    def __init__(self, inject):
        super().__init__()
        self.model = Backbone(inject)
        self.inject = inject

    def combine_hidden_states(self, aux_hidden_states):
        return self.inject("context.projected", aux_hidden_states.reshape(-1, 3, 4).sum(1))

    def precompute_and_store_context_kv(self, context, positions, slots):
        for index, layer in enumerate(self.model.layers.values()):
            kv = self.inject(f"mtp.{index}.context_kv", context.unsqueeze(1).clone())
            self.model._store_standard_swa_kv(kv, slots[index], layer.self_attn)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_draft_logits(self, hidden_states):
        return self.inject("head", hidden_states.clone())


def installed(tmp_path, monkeypatch, mode="upstream-boundaries"):
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_observation", OBS)
    cls = runpy.run_path(str(ROOT / "vllm_ascend/diagnostics/dspark_profile_upstream.py"))["UpstreamProfileObservation"]
    runner, fault = Runner(), {"boundary": None, "value": float("nan"), "exception": None}

    def inject(name, tensor):
        if fault["exception"] == name:
            raise RuntimeError("original draft failure")
        if fault["boundary"] == name:
            tensor[0] = fault["value"]
        return tensor

    spec = runner.speculator
    spec.model = Draft(inject)
    spec.target_layer_ids = (3, 8, 15)
    spec.draft_layer_group_ids = {"mtp.0.swa_cache": 3, "mtp.1.swa_cache": 2, "mtp.2.swa_cache": 3}
    spec._run_draft_model_forward = lambda p, metadata: spec.model(p.draft_input_ids, p.draft_positions)

    def checked_markov(proposal_inputs, hidden_states):
        spec._markov_attempt_step_epoch = proposal_inputs.step_epoch
        logits = spec.model.compute_draft_logits(hidden_states)
        if torch.isnan(logits).any():
            assert (tmp_path / "rank-0-first-nan.json").exists()
            raise RuntimeError("Ascend DSpark Markov base logits contain NaN.")
        return logits

    spec._execute_sequential_markov_sampling = checked_markov

    def draft(p):
        aux = torch.cat([h[: p.num_target_tokens] for h in p.auxiliary_hidden_states], dim=-1)
        context = spec.model.combine_hidden_states(aux)
        slots = [torch.arange(p.num_target_tokens, dtype=torch.int32) for _ in range(3)]
        spec.model.precompute_and_store_context_kv(context, p.target_positions, slots)
        hidden = spec._run_draft_model_forward(p, {})
        return spec._execute_sequential_markov_sampling(p, hidden)

    spec._execute_draft = draft
    observer = (
        cls(runner, {"mode": mode, "directory": str(tmp_path)})
        if mode == "upstream-boundaries"
        else OBS.ProfileObservation(runner, {"mode": mode, "directory": str(tmp_path)})
    )
    observer.begin_point("test-point")
    return runner, observer, fault


def proposal(runner, epoch, ids=("actual-0", "actual-2", "actual-1"), k=5):
    runner.execute_model(NS(num_scheduled_tokens=dict(zip(ids, (1, 4, 6))), finished_req_ids=[]))
    n, t = len(ids), runner.input_batch.num_tokens
    runner.speculator._proposal_step_epoch = epoch
    return NS(
        step_epoch=epoch,
        rank=0,
        request_ids=ids,
        num_reqs=n,
        num_target_tokens=t,
        num_query_tokens=n * k,
        num_speculative_tokens=k,
        target_layer_ids=(3, 8, 15),
        auxiliary_hidden_states=[torch.ones(t + 1, 4) for _ in range(3)],
        request_state_indices=torch.tensor([7, 3, 4][:n]),
        target_query_start_loc=torch.tensor(runner.input_batch.query_start_loc_np[: n + 1]),
        target_positions=torch.arange(t),
        target_sequence_lengths=torch.arange(n) + 230,
        num_sampled=torch.ones(n, dtype=torch.int32),
        num_rejected=torch.zeros(n, dtype=torch.int32),
        draft_positions=torch.arange(n * k),
        draft_query_start_loc=torch.arange(n + 1) * k,
        draft_sequence_lengths=torch.arange(n) + 235,
        draft_input_ids=torch.ones(n * k, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "target_aux.8",
        "context.projected",
        "mtp.1.context_kv",
        "draft.initial_hidden",
        "mtp.0.output",
        "mtp.1.output",
        "mtp.2.output",
        "draft.hc_output",
    ],
)
def test_first_bad_upstream_boundary_and_previous_two_rounds(tmp_path, monkeypatch, boundary):
    runner, observer, fault = installed(tmp_path, monkeypatch)
    for epoch in (61, 62, 63):
        runner.speculator._execute_draft(proposal(runner, epoch))
    p = proposal(runner, 64)
    if boundary.startswith("target_aux"):
        p.auxiliary_hidden_states[1][0] = float("nan")
    else:
        fault["boundary"] = boundary
    if boundary in ("target_aux.8", "context.projected", "mtp.1.context_kv"):
        runner.speculator._execute_draft(p)  # mocked KV consumer does not propagate
    else:
        with pytest.raises(RuntimeError, match="base logits contain NaN"):
            runner.speculator._execute_draft(p)
    data = json.loads((tmp_path / "rank-0-upstream-first-nan.json").read_text())
    rounds = data["upstream"]["rounds"]
    assert [r["proposal_epoch"] for r in rounds] == [62, 63, 64]
    assert [r["execution"] for r in rounds] == [2, 3, 4]
    record = rounds[-1]
    assert record["point"] == "test-point" and record["rank"] == 0
    bad = [b for b in record["boundaries"] if any(r["nan"] for r in b["rows"])]
    assert bad[0]["name"] == boundary
    row = bad[0]["rows"][0]
    assert row["request_id"] == "actual-0" and row["request_row"] == row["position_in_request"] == 0
    assert all(not r["inf"] for b in record["boundaries"] for r in b["rows"])
    assert record["device_integers"]["request_state_indices"] == [7, 3, 4]
    assert record["missing_boundaries"] == []
    assert data["recording_error"] is None and data["performance_eligible"] is False
    assert data["numeric"]["compact_host_transfers_completed"] == 4
    assert data["upstream"]["packet_bytes"] == 4 * 3928
    assert record["target_mapping_matches_device"] is True
    assert observer.pending is None
    observer.close()


def test_one_packet_padding_domains_and_consumption_time_copies(tmp_path, monkeypatch):
    runner, observer, fault = installed(tmp_path, monkeypatch)
    p = proposal(runner, 12, ids=("a", "b"), k=3)
    for aux in p.auxiliary_hidden_states:
        aux[-1] = float("nan")  # excluded graph padding
    fault.update(boundary="mtp.1.context_kv", value=float("-inf"))
    transfers = []
    original = torch.Tensor.to

    def transfer(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu":
            transfers.append((tuple(tensor.shape), tensor.dtype, kwargs))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    runner.speculator._execute_draft(p)
    assert len(transfers) == 1 and transfers[0][1] == torch.int64
    record = observer.upstream_records[-1]
    assert record["target_rows"] == 5 and record["candidate_rows"] == 6
    boundaries = {b["name"]: b for b in record["boundaries"]}
    assert not any(row["nan"] for b in record["boundaries"] for row in b["rows"])
    assert boundaries["mtp.1.context_kv"]["rows"][0]["inf"]  # store zeroed its input afterwards
    assert boundaries["target_aux.3"]["rows"][1]["request_id"] == "b"
    assert boundaries["draft.initial_hidden"]["rows"][1]["request_id"] == "a"
    frozen = json.dumps(record)
    p.num_rejected.fill_(5)
    p.target_positions.fill_(999)
    assert json.dumps(record) == frozen and record["device_integers"]["num_rejected"] == [0, 0]
    receipt = observer.finish_point()
    assert "rounds" not in receipt["upstream"] and receipt["upstream"]["counts"]["inf_rounds"] == 1
    observer.close()


def test_early_layer_error_drains_partial_evidence_before_original_exception(tmp_path, monkeypatch):
    runner, observer, fault = installed(tmp_path, monkeypatch)
    fault["exception"] = "mtp.1.output"
    with pytest.raises(RuntimeError, match="original draft failure"):
        runner.speculator._execute_draft(proposal(runner, 8))
    data = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    record = data["upstream"]["rounds"][-1]
    assert record["proposal_epoch"] == 8 and not record["head_reached"]
    assert "mtp.1.output" in record["missing_boundaries"] and "mtp.0.output" not in record["missing_boundaries"]
    assert data["numeric"]["compact_host_transfers_completed"] == 1
    assert observer.pending is None
    observer.close()


def test_upstream_off_does_not_install_layer_hooks(tmp_path, monkeypatch):
    runner, observer, _ = installed(tmp_path, monkeypatch, mode="numeric-boundaries")
    assert all("forward" not in vars(layer) for layer in runner.speculator.model.model.layers.values())
    runner.speculator._execute_draft(proposal(runner, 7))
    assert "upstream" not in observer.snapshot() and observer.numeric_transfers == 1
    observer.close()


def test_head_exception_keeps_upstream_and_hidden_in_one_partial_packet(tmp_path, monkeypatch):
    runner, observer, fault = installed(tmp_path, monkeypatch)
    fault["exception"] = "head"
    with pytest.raises(RuntimeError, match="original draft failure"):
        runner.speculator._execute_draft(proposal(runner, 19))
    data = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    record = data["upstream"]["rounds"][-1]
    assert record["proposal_epoch"] == 19 and record["head_reached"]
    assert record["missing_boundaries"] == []
    assert record["head_flag_columns"] == ["hidden_nan", "hidden_inf"]
    assert data["numeric"]["rounds"][-1]["classification"] == "logits_unavailable"
    assert data["numeric"]["compact_host_transfers_completed"] == 1
    assert data["recording_error"] is None and observer.pending is None
    observer.close()


def test_stale_upstream_epoch_cannot_produce_a_valid_record(tmp_path, monkeypatch):
    runner, observer, _ = installed(tmp_path, monkeypatch)
    p = proposal(runner, 17)
    runner.speculator._proposal_step_epoch = 18
    runner.speculator._execute_draft(p)
    assert observer.recording_error and not observer.upstream_records
    assert observer.numeric_transfers == 0
    with pytest.raises(RuntimeError, match="evidence unavailable"):
        observer.finish_point()
    observer.close()


def test_mismatched_device_spans_are_exposed_without_relabeling_cpu_rows(tmp_path, monkeypatch):
    runner, observer, _ = installed(tmp_path, monkeypatch)
    p = proposal(runner, 16)
    p.target_query_start_loc[1] = 2
    runner.speculator._execute_draft(p)
    record = observer.upstream_records[-1]
    assert record["target_mapping_matches_device"] is False
    assert record["target_query_start_loc_cpu"] == [0, 1, 5, 11]
    assert record["device_integers"]["target_query_start_loc"] == [0, 2, 5, 11]
    observer.close()


def test_recording_failure_preserves_original_draft_exception(tmp_path, monkeypatch):
    runner, observer, fault = installed(tmp_path, monkeypatch)
    fault["exception"] = "mtp.0.output"
    monkeypatch.setattr(observer, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("storage failed")))
    with pytest.raises(RuntimeError, match="original draft failure"):
        runner.speculator._execute_draft(proposal(runner, 2))
    assert observer.pending is None
    observer.close()


def test_point_reset_close_and_missing_boundaries_are_explicit(tmp_path, monkeypatch):
    runner, observer, fault = installed(tmp_path, monkeypatch)
    # Bypass an installed HC hook to model a changed execution path.
    runner.speculator.model.model.hc_head = lambda x, *a: x.mean(1)
    runner.speculator._execute_draft(proposal(runner, 1))
    assert "Missing upstream boundaries" in observer.recording_error
    assert "draft.hc_output" in observer.upstream_records[-1]["missing_boundaries"]
    with pytest.raises(RuntimeError, match="evidence unavailable"):
        observer.finish_point()
    observer.begin_point("next-point")
    assert not observer.upstream_records and observer.pending is None
    observer.close()
    assert all("forward" not in vars(layer) for layer in runner.speculator.model.model.layers.values())


def test_profiler_factory_installs_upstream_scope_around_real_timing_wrapper(tmp_path, monkeypatch):
    runner, old, _ = installed(tmp_path, monkeypatch)
    upstream_class = type(old)
    old.close()
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.diagnostics.dspark_profile_upstream", NS(UpstreamProfileObservation=upstream_class)
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(COST_CONTEXT_SEMANTICS="test"))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.worker.v2.spec_decode.dspark.verification_runtime", NS(runtime_identity=lambda *a: {})
    )
    event = lambda **kwargs: NS(record=lambda: None, elapsed_time=lambda end: 1)
    monkeypatch.setattr(
        torch, "npu", NS(Event=event, synchronize=lambda: None, get_device_name=lambda device: "mock"), raising=False
    )
    runner.vllm_config.additional_config["dspark_profile_observation"] = {
        "mode": "upstream-boundaries",
        "directory": str(tmp_path),
    }
    runner.vllm_config.model_config = NS(max_model_len=8192)
    runner.speculator.confidence_verification.receipt = {"weights_sha256": "test"}
    runner.device = "cpu"
    profiler_class = runpy.run_path(str(ROOT / "vllm_ascend/diagnostics/dspark_cost_profile.py"))[
        "IsolatedCostProfiler"
    ]
    profiler = profiler_class(runner)
    assert isinstance(profiler.observation, upstream_class)
    profiler.observation.begin_point("factory")
    runner.speculator._execute_draft(proposal(runner, 3))
    data = profiler.snapshot()
    assert data["identity"]["diagnostic_only"] is True
    assert data["observation"]["upstream"]["counts"]["rounds"] == 1
    assert len(data["measurements"]) == 2
    assert profiler.observation.recording_error is None
    profiler.observation.close()
