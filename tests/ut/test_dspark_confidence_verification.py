# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU reference execution and installed-class regression for adaptive verification."""

import copy
import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.ut.worker import test_dsa_padded_requests as padded_requests
from tests.ut.worker.test_dsa_capture_metadata import _load_functions
from tools.dspark.verification_tools import summarize_verification

ROOT = Path(__file__).parents[2]
POLICY = runpy.run_path(str(ROOT / "vllm_ascend/spec_decode/dspark_verification.py"))
ConfidenceRow = POLICY["ConfidenceRow"]
CostTable = POLICY["CostTable"]
allocate_prefixes = POLICY["allocate_prefixes"]
fill_varlen_query_padding = POLICY["fill_varlen_query_padding"]
trim_scheduler_output = POLICY["trim_scheduler_output"]
validate_length = POLICY["validate_length"]

torch = pytest.importorskip("torch")


def load_module(path, name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(**POLICY))
    module = load_module(
        ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/verification_runtime.py", "verification_cpu", monkeypatch
    )
    return module.ConfidenceVerification


def costs(target=None):
    return CostTable(target or {6: 1.0, 12: 1.1, 18: 1.2, 24: 1.3}, {4: 0.5, 1: 0.5}, 0.001, (0, 8192), {})


def decide(rows, table=None, capacities=None):
    return allocate_prefixes(
        rows,
        capacities or {r.request_id: 5 for r in rows},
        base_tokens=len(rows),
        sampling_requests=len(rows),
        draft_requests=len(rows),
        context=32,
        costs=table or costs(),
    )


def test_survival_and_cost_aware_allocation():
    row = ConfidenceRow("r", 7, (0.9, 0.8, 0.7, 0.6, 0.5))
    assert row.survival() == pytest.approx([0.9, 0.72, 0.504, 0.3024, 0.1512])
    rows = [ConfidenceRow(str(i), 7, (0.9,) * 5) for i in range(4)]
    cheap = decide(rows)
    expensive = decide(rows, costs({6: 1, 12: 100, 18: 200, 24: 300}))
    assert sum(cheap.lengths.values()) > sum(expensive.lengths.values())
    assert expensive.actual_tokens == 6
    assert expensive.lengths == {"0": 1, "1": 1, "2": 0, "3": 0}
    assert decide(list(reversed(rows)), costs({6: 1, 12: 100, 18: 200, 24: 300})).lengths == expensive.lengths


def test_zero_confidence_smaller_budget_and_prefix_ties():
    rows = [ConfidenceRow(str(i), 1, (0.0,) * 5) for i in range(4)]
    assert list(decide(rows).lengths.values()) == [0] * 4
    rows = [ConfidenceRow(str(i), 1, (1.0,) * 5) for i in range(4)]
    choice = decide(rows, costs({6: 1, 12: 100, 18: 100, 24: 100}))
    assert choice.lengths == {"0": 2, "1": 0, "2": 0, "3": 0}


@pytest.mark.parametrize("length", [-1, 6, 1.0, True, None])
def test_invalid_length(length):
    with pytest.raises(ValueError):
        validate_length(length)


@pytest.mark.parametrize("p", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_confidence(p):
    with pytest.raises(ValueError):
        ConfidenceRow("r", 1, (p,) * 5)


def test_profile_fails_closed(tmp_path):
    path = tmp_path / "cost.json"
    path.write_text(json.dumps({"schema_version": 1, "source": "synthetic"}))
    with pytest.raises(ValueError, match="measured"):
        CostTable.load(str(path), {})
    with pytest.raises(ValueError, match="Context"):
        costs().cost(4, 15, 9999)
    with pytest.raises(ValueError, match="draft cost"):
        costs().cost(3, 15, 0)
    with pytest.raises(ValueError, match="capacity"):
        costs().cost(4, 25, 0)


@pytest.mark.parametrize("lengths", [[5, 2, 0, 4], [0] * 4, [5] * 4])
def test_real_scheduler_copy_and_accounting(lengths):
    original = NS(
        num_scheduled_tokens={str(i): 6 for i in range(4)},
        total_num_scheduled_tokens=24,
        scheduled_spec_decode_tokens={str(i): list(range(5)) for i in range(4)},
        unrelated=object(),
    )
    shortened = trim_scheduler_output(original, dict(zip(original.num_scheduled_tokens, lengths)))
    assert original.total_num_scheduled_tokens == 24
    assert list(original.num_scheduled_tokens.values()) == [6] * 4
    assert shortened.total_num_scheduled_tokens == sum(lengths) + 4
    assert list(shortened.num_scheduled_tokens.values()) == [ell + 1 for ell in lengths]
    assert shortened.unrelated is original.unrelated
    for i, ell in enumerate(lengths):
        assert shortened.scheduled_spec_decode_tokens[str(i)] == list(range(ell))
        for accepted in range(ell + 1):
            # Frozen scheduler uses the original five; worker uses actual ell.
            assert 6 - (5 - accepted) == 1 + ell - (ell - accepted)


def test_same_graph_buffer_varlen_reorder_and_next_round():
    buffer = np.full(9, -99, dtype=np.int32)
    identity = buffer.ctypes.data
    for lengths in ([5, 2, 0, 4], [4, 0, 2, 5], [0] * 4, [5] * 4, [2, 0]):
        n = len(lengths)
        buffer[0] = 0
        np.cumsum(np.array(lengths) + 1, out=buffer[1 : n + 1])
        result, padded = fill_varlen_query_padding(buffer, n, 8, 48)
        assert result is buffer and padded == 8 and buffer.ctypes.data == identity
        assert np.array_equal(np.diff(buffer[: n + 1]), np.array(lengths) + 1)
        assert np.all(buffer[n + 1 :] == sum(lengths) + n)
    with pytest.raises(ValueError):
        fill_varlen_query_padding(buffer, 9, 8, 48)
    with pytest.raises(ValueError):
        fill_varlen_query_padding(buffer, 2, 8, 1)


def test_real_confidence_head_inputs_and_provenance(runtime):
    head = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    seen = []

    def logits(hidden, markov):
        seen.append((hidden.clone(), markov.clone()))
        return head(torch.cat([hidden, markov], dim=-1)).squeeze(-1)

    model = NS(
        model=NS(confidence_head=head),
        confidence_logits=logits,
        confidence_weight_receipt={"loaded_parameters": ["proj.weight"], "weights_sha256": "checked"},
    )
    state = runtime({"mode": "confidence"}, None, "cpu")
    state.bind_model(model)
    hidden = torch.arange(20, dtype=torch.float32).reshape(10, 2) / 100
    embeddings = [torch.tensor([[float(i)], [float(10 + i)]]) for i in range(5)]
    state.record(("b", "a"), 8, hidden, embeddings, model)
    assert seen[0][1].flatten().tolist() == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    expected = torch.sigmoid(logits(hidden, seen[0][1])).reshape(2, 5)
    assert state.rows["b"].conditional == pytest.approx(expected[0].tolist())
    assert state.rows["a"].producer_epoch == 8
    assert state.snapshot()["confidence_head_calls"] == 1
    assert state.calibration["status"] == "uncalibrated"
    model.confidence_weight_receipt = None
    with pytest.raises(ValueError, match="checkpoint-loaded"):
        state.bind_model(model)


def test_real_head_nan_rejected_before_publication(runtime):
    model = NS(confidence_logits=lambda hidden, markov: torch.full((5,), float("inf")))
    state = runtime({"mode": "confidence"}, None, "cpu")
    state.receipt = {"loaded_parameters": ["weight"]}
    with pytest.raises(ValueError, match="Non-finite"):
        state.record(("r",), 1, torch.zeros(5, 2), [torch.zeros(1, 1)] * 5, model)
    assert not state.rows


def test_selection_stale_epoch_delayed_and_terminal(runtime, monkeypatch):
    state = runtime({"mode": "confidence"}, None, "cpu")
    state.costs = costs()
    state.record_decisions = True
    state.rows = {key: ConfidenceRow(key, 2, (0.9,) * 5) for key in ("r", "delayed", "terminal")}
    owners = {key: NS(producer_epoch=2) for key in ("r", "delayed")}
    states = NS(req_id_to_index={"r": 0}, num_computed_tokens_np=np.array([32]), prefill_len=NS(np=np.array([16])))
    runner = NS(req_states=states, speculator=NS(_published_proposal_owners=owners))
    output = NS(
        num_scheduled_tokens={"r": 6}, total_num_scheduled_tokens=6, scheduled_spec_decode_tokens={"r": [1, 2, 3, 4, 5]}
    )
    group = NS(rank_in_group=0, broadcast_object=lambda packet, src: packet)
    monkeypatch.setitem(sys.modules, "vllm.distributed", NS(get_tp_group=lambda: group))
    selected = state.select(runner, output)
    assert selected.num_scheduled_tokens["r"] == 6
    assert state.snapshot()["decisions"][0]["producer_epochs"] == {"r": 2}
    assert state.snapshot()["decisions"][0]["confidence_epochs"] == {"r": 2}
    assert "delayed" in state.rows and "terminal" not in state.rows
    state.accepted(("r",), (5,), torch.tensor([3]), (2,))
    assert "r" not in state.rows and "delayed" in state.rows
    state.rows["r"] = ConfidenceRow("r", 1, (0.9,) * 5)
    with pytest.raises(ValueError, match="Stale"):
        state.select(runner, output)


def test_zero_length_consumes_real_proposal_and_produces_next(monkeypatch):
    namespace = {"torch": torch}
    _load_functions(
        ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
        namespace,
        ["propose"],
        "AscendDSparkSpeculator",
    )
    calls = []
    owner = NS(
        confidence_verification=object(),
        _next_proposal_skipped=False,
        _published_candidate_tokens=torch.ones(1, 5),
        _active_proposal_reconciled=True,
        _active_published_proposal_owner_ids=("r",),
        continue_after_verification=True,
        _consume_published_proposal_after_verification=lambda *args: calls.append("consume"),
        _release_consumed_proposal=lambda: calls.append("retire_suffix"),
        prepare_proposal_inputs=lambda **kwargs: "fresh_inputs",
        _execute_draft=lambda inputs: calls.append(inputs),
    )
    namespace["propose"](owner, NS(num_draft_tokens=0), None, None, None, None, None, None, None, None, None, None)
    assert calls == ["consume", "retire_suffix", "fresh_inputs"]


def snapshots(replays=1):
    adaptive = {
        "mode": "confidence",
        "confidence_head_calls": 1,
        "confidence_batches": 1,
        "specified_batches": 0,
        "fixed_admission_batches": 0,
        "weights": {"loaded_parameters": ["confidence_head.proj.weight"], "weights_sha256": "test-receipt"},
        "calibration": {},
        "generated": 20,
        "scheduled": 11,
        "verified": 11,
        "accepted": 4,
        "policy_seconds": 0.01,
        "confidence_transfer_seconds": 0.01,
        "length_histogram": [1, 0, 1, 0, 1, 1],
        "verified_by_position": [3, 3, 2, 2, 1],
        "generated_by_position": [4] * 5,
        "accepted_by_position": [3, 1, 0, 0, 0],
    }
    before, after = [], []
    for rank in range(8):
        b = {
            "rank": rank,
            "failed_execution_count": 0,
            "error": None,
            "query_layouts": [],
            "confidence_verification": copy.deepcopy(adaptive),
        }
        a = copy.deepcopy(b)
        a["query_layouts"] = [{"capacity": 24, "query_lengths": [6, 3, 1, 5], "count": replays}]
        for key, value in adaptive.items():
            if type(value) in (int, float):
                b["confidence_verification"][key] = 0
            elif isinstance(value, list):
                b["confidence_verification"][key] = [0] * len(value)
        before.append(b)
        after.append(a)
    return before, after


def test_measured_real_replay_rank_dedup_and_intervals():
    before, after = snapshots()
    result = summarize_verification(before, after, 8)
    assert result["logical_full_replays"] == 1 and result["mixed_length_full_replays"] == 1
    assert result["effective_target_tokens"] == 15 and result["padded_target_tokens"] == 24
    assert result["accepted_per_verified_position"] == [1, 1 / 3, 0, 0, 0]
    with pytest.raises(ValueError, match="No successful measured"):
        summarize_verification(after, after, 8)
    with pytest.raises(ValueError, match="Missing rank"):
        summarize_verification(before, after[:-1], 8)
    after[0]["failed_execution_count"] = 1
    with pytest.raises(ValueError, match="Failed execution"):
        summarize_verification(before, after, 8)


metadata_api = padded_requests.api
scatter_selector = padded_requests.scatter_selector


def test_real_builder_varlen_shared_groups_padding_and_aliases(metadata_api, monkeypatch):
    api = metadata_api
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(**POLICY))
    runner = padded_requests._runner(api)
    runner._dspark_varlen_decode = True
    runs, _ = padded_requests._groups(api, monkeypatch)
    captured_offsets = []

    def checked_metadata(**kwargs):
        query = kwargs["cu_seqlens_q"]
        seq = kwargs["seqused_kv"]
        assert query.shape[0] == seq.shape[0] + 1
        assert torch.all(query.diff() >= 0)
        assert torch.all(query.diff() <= 6)
        captured_offsets.append(query.clone())
        return torch.full((1024,), len(captured_offsets), dtype=torch.int32)

    monkeypatch.setattr(
        api.dsa_globals["DeviceOperator"], "get_dsa_sparse_attn_metadata_op", staticmethod(lambda: checked_metadata)
    )

    def checked_qli(**kwargs):
        ends = kwargs["actual_seq_lengths_query"]
        lengths = torch.diff(ends, prepend=torch.zeros(1, dtype=ends.dtype))
        assert kwargs["layout_key"] == "PA_BSND"
        assert kwargs["max_seqlen_q"] == int(lengths.max())
        assert torch.all((lengths >= 0) & (lengths <= 6))
        return torch.full((1024,), 3, dtype=torch.int32)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_vllm_quant_lightning_indexer_metadata", checked_qli)
    groups = [[run.group] for run in runs]
    config = NS(kv_cache_groups=[object() for _ in runs])
    blocks = tuple(torch.zeros(24, 8, dtype=torch.int32) for _ in runs)
    slots = torch.full((3, 24), -1, dtype=torch.int32)
    aliases = None
    for step, lengths in enumerate(([5, 2, 0, 4], [4, 0, 2, 5], [0] * 4, [5] * 4, [2, 0])):
        count = len(lengths)
        scheduler = NS(
            num_scheduled_tokens={str(i): ell + 1 for i, ell in enumerate(lengths)},
            total_num_scheduled_tokens=sum(lengths) + count,
            scheduled_spec_decode_tokens={str(i): [1] * ell for i, ell in enumerate(lengths)},
            scheduled_cached_reqs=NS(req_ids=[]),
            has_structured_output_requests=False,
        )
        batch = runner.prepare_inputs(scheduler, NS(num_tokens=24, num_reqs=24, cg_mode=api.mode.FULL))
        ordered = sorted(ell + 1 for ell in lengths)
        assert batch.query_start_loc.diff().tolist() == ordered
        slots.fill_(-1)
        valid = batch.num_tokens
        slots[:, :valid] = 32 + batch.positions[:valid] % 32
        for table in blocks:
            table.fill_(1)
        metadata = runs[0].state.prepare_attn(batch, api.mode.FULL, blocks, slots, groups, config)
        current = []
        for i, run in enumerate(runs):
            leaf = metadata[f"group{i}"].decode
            assert metadata[f"group{i}"] is metadata[f"group{i}.shared"]
            assert leaf.query_start_loc[: count + 1].diff().tolist() == ordered
            assert torch.all(leaf.query_start_loc[count + 1 :] == valid)
            assert torch.all(leaf.start_pos[count:] == 0)
            assert torch.all(leaf.seq_lens[count:] == 0)
            torch.testing.assert_close(
                leaf.start_pos[:count], batch.seq_lens - torch.tensor(ordered, dtype=torch.int32)
            )
            assert run.builder.decode_ratio_to_sas_metadata is runs[0].builder.decode_ratio_to_sas_metadata
            current.append(
                tuple(
                    getattr(leaf, key).data_ptr()
                    for key in ("query_start_loc", "seq_lens", "start_pos", "sas_metadata")
                )
            )
        if aliases:
            assert current == aliases
        aliases = current
        leaf = metadata["group0"].decode
        cache = torch.full((8, 32, 1, 512), 7, dtype=torch.bfloat16)
        updates = torch.full((24, 1, 512), 9, dtype=torch.bfloat16)
        updates[valid:] = 99
        formatted = runs[0].builder.slot_mapping[:24]
        api.validate(
            layer_name="swa",
            kv_cache=cache,
            block_table=leaf.block_table,
            slot_mapping=leaf.slot_mapping,
            query_start_loc=leaf.query_start_loc,
            seqused_kv=leaf.seq_lens,
            block_size=32,
            num_query_tokens=valid,
            num_reqs_actual=count,
            sas_metadata=leaf.sas_metadata,
            sinks=torch.zeros(8, dtype=torch.float32),
        )
        api.device.dsa_kv_compress_scatter(cache, updates, formatted)
        assert not torch.any(cache == 99)
        assert torch.all(cache[0] == 7)


def test_installed_confidence_head_returns_actual_projection():
    pytest.importorskip("vllm")
    pytest.importorskip("torch_npu")
    from vllm_ascend.models.deepseek_v4_dspark import DSparkConfidenceHead

    head = DSparkConfidenceHead.__new__(DSparkConfidenceHead)
    torch.nn.Module.__init__(head)
    head.proj = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        head.proj.weight.fill_(1)
    result = head(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]]))
    assert result.tolist() == [6.0]


@pytest.fixture(params=["source", "runtime"])
def graph_api(request):
    if request.param == "runtime":
        pytest.importorskip("vllm")
        pytest.importorskip("torch_npu")
        from vllm.config.compilation import CUDAGraphMode

        from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

        return ModelAclGraphManager, CUDAGraphMode
    import ast
    from collections import defaultdict
    from dataclasses import dataclass
    from enum import Enum
    from itertools import product

    class Mode(Enum):
        NONE = 0
        FULL = 1
        PIECEWISE = 2
        FULL_DECODE_ONLY = 3

        def decode_mode(self):
            return Mode.FULL

        def mixed_mode(self):
            return None

        def separate_routine(self):
            return True

    namespace = dict(CUDAGraphMode=Mode, dataclass=dataclass, defaultdict=defaultdict, product=product)
    source = ROOT.parent / "vllm-hust/vllm/v1/worker/gpu/cudagraph_utils.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "BatchExecutionDescriptor"
    )
    exec(
        compile(
            ast.Module(body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[]),
            str(source),
            "exec",
        ),
        namespace,
    )
    _load_functions(source, namespace, ["_is_compatible"])
    _load_functions(source, namespace, ["dispatch", "_init_candidates", "_resolve_effective_loras"], "CudaGraphManager")
    parent = type(
        "ModelCudaGraphManager",
        (),
        {name: namespace[name] for name in ("dispatch", "_init_candidates", "_resolve_effective_loras")},
    )
    namespace["ModelCudaGraphManager"] = parent
    path = ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "ModelAclGraphManager"
    )
    exec(
        compile(
            ast.Module(body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["ModelAclGraphManager"], Mode


def test_real_graph_dispatch_same_capacity_cross_tier_and_off(graph_api):
    cls, mode = graph_api
    manager = cls.__new__(cls)
    manager.compilation_config = NS(cudagraph_capture_sizes=[6, 12, 24])
    manager.max_num_reqs = 4
    manager.decode_query_len = 6
    manager.cudagraph_mode = mode.FULL_DECODE_ONLY
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {0: 0}
    manager._max_lora_case = 0
    manager._graphs_captured = True
    manager.model_runner = NS(_dspark_varlen_decode=True)
    for enabled in (True, False):
        manager.verification_options = {"mode": "confidence"} if enabled else None
        manager._candidates = {}
        manager._capture_descs = {}
        manager._init_candidates()
        mixed = manager.dispatch(4, 15, None, 0)
        if enabled:
            assert mixed.cg_mode == mode.FULL and mixed.num_reqs == 4 and mixed.num_tokens == 24
            assert mixed == manager.dispatch(4, 16, None, 0)
            assert manager.dispatch(4, 4, 1, 0).num_tokens == 6
            assert manager.dispatch(4, 24, 6, 0) == mixed
            with pytest.raises(ValueError, match="fallback"):
                manager.dispatch(4, 25, None, 0)
        else:
            assert mixed.cg_mode == mode.NONE
            assert manager.dispatch(4, 24, 6, 0).cg_mode == mode.FULL


@pytest.fixture
def proposal_class(monkeypatch):
    import ast
    import logging
    from dataclasses import dataclass, replace
    from types import MappingProxyType

    namespace = dict(
        torch=torch,
        np=np,
        dataclass=dataclass,
        replace=replace,
        MappingProxyType=MappingProxyType,
        BaseSpeculator=object,
        logging=logging,
        logger=logging.getLogger("verification_test"),
        _DSPARK_MARKOV_FIXED_K=5,
    )
    inputs = runpy.run_path(str(ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/proposal_inputs.py"))
    namespace.update({key: value for key, value in inputs.items() if key.startswith("AscendDSpark")})
    source = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
    nodes = [
        node
        for node in __import__("ast").parse(source.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("AscendDSparkSpeculator", "_PublishedProposalOwner", "_assert_markov_tensor_contract")
    ]
    # Dataclass registration is isolated to this fixture, never production globals.
    namespace["__name__"] = "verification_proposal_cpu"
    monkeypatch.setitem(sys.modules, "verification_proposal_cpu", NS(__dict__=namespace))
    exec(
        compile(
            ast.Module(body=[*ast.parse("from __future__ import annotations").body, *nodes], type_ignores=[]),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("lengths", [(0, 0), (0, 3), (5, 2)])
def test_real_reconcile_consume_reorder_delayed_terminal(proposal_class, runtime, lengths):
    api = proposal_class
    cls = api["AscendDSparkSpeculator"]
    spec = cls.__new__(cls)
    spec.device = torch.device("cpu")
    spec.rank = 0
    spec.num_speculative_steps = 5
    spec._proposal_step_epoch = 7
    spec._proposal_installed_count = 0
    spec._proposal_consumption_count = 0
    spec._proposal_dropped_count = 0
    spec._terminal_proposal_discard_count = 0
    spec._clear_active_published_proposal()
    state = runtime({"mode": "specified_lengths", "lengths": list(lengths)}, None, "cpu")
    state.selected_epochs = {"a": 7, "b": 7}
    spec.confidence_verification = state
    candidates = torch.tensor([[11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]])
    indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    owners = {}
    for row, key in enumerate(("a", "b", "delayed")):
        lifecycle = api["AscendDSparkProposalLifecycle"](7, 7, None, (key,), True, True, False, False, False)
        owners[key] = api["_PublishedProposalOwner"](key, 7, indices, candidates, row, 5, lifecycle)
    spec._published_proposal_owners = owners
    disposition = spec.reconcile_scheduler_proposal(
        scheduled_spec_decode_tokens={
            "a": candidates[0, : lengths[0]].tolist(),
            "b": candidates[1, : lengths[1]].tolist(),
        },
        scheduled_request_ids={"a", "b", "new"},
        finished_request_ids=set(),
        preempted_request_ids=set(),
        known_request_ids={"a", "b", "delayed", "new"},
    )
    assert disposition == "TRUNCATED"
    reordered_lengths = (lengths[1], lengths[0], 0)
    offsets = torch.tensor([0, 1 + lengths[1], 2 + sum(lengths), 5 + sum(lengths)], dtype=torch.int32)
    tokens = torch.tensor(
        [99, *candidates[1, : lengths[1]].tolist(), 88, *candidates[0, : lengths[0]].tolist(), 7, 8, 9],
        dtype=torch.int32,
    )
    batch = NS(
        req_ids=["b", "a", "new"],
        num_reqs=3,
        num_tokens=tokens.numel(),
        input_ids=tokens,
        idx_mapping=torch.tensor([1, 0, 3], dtype=torch.int32),
        query_start_loc=offsets,
        num_draft_tokens_per_req=reordered_lengths,
        num_draft_tokens=sum(lengths),
    )
    sampled = torch.tensor([1 + min(lengths[1], 1), 1 + min(lengths[0], 2), 0], dtype=torch.int32)
    rejected = torch.tensor([lengths[1] - min(lengths[1], 1), lengths[0] - min(lengths[0], 2), 0], dtype=torch.int32)
    spec._consume_published_proposal_after_verification(batch, sampled, rejected, torch.zeros(4))
    assert spec._published_proposal_consumed
    assert all(owners[key].lifecycle.consumed for key in ("a", "b"))
    spec._release_consumed_proposal()
    assert set(owners) == {"delayed"}  # The truncated suffix must not survive.
    spec.discard_terminal_proposal({"delayed"})
    assert not owners
    assert state.verified == sum(lengths)


@pytest.fixture
def capture_batches(request, monkeypatch):
    from tests.ut.worker.test_capture_input_aliases import batches

    return batches.__wrapped__(monkeypatch)


def test_varlen_capture_balance_and_live_offsets(capture_batches):
    api = capture_batches
    api.buffers.dspark_varlen_capture = True
    captured = {}
    for capacity in (6, 12, 18, 24):
        warmup = api.cls.make_dummy(4, capacity, api.buffers)
        graph = api.cls.make_dummy(4, capacity, api.buffers)
        lengths = graph.query_start_loc.diff()
        assert lengths.sum() == capacity and lengths.max() <= 6
        assert lengths.max() - lengths.min() <= 1
        assert graph.num_scheduled_tokens.tolist() == lengths.tolist()
        assert graph.seq_lens_cpu_upper_bound.tolist() == lengths.tolist()
        assert graph.logits_indices.tolist() == (graph.query_start_loc[1:] - 1).tolist()
        for field in ("seq_lens", "query_start_loc", "input_ids", "positions"):
            assert getattr(warmup, field).data_ptr() == getattr(graph, field).data_ptr()
            assert getattr(graph, field).data_ptr() == getattr(api.buffers, field).data_ptr()
        captured[capacity] = graph
    for query_lengths in ([6, 3, 1, 5], [1, 1, 1, 1], [6, 6, 6, 6]):
        offsets = torch.tensor([0, *np.cumsum(query_lengths)], dtype=torch.int32)
        api.buffers.query_start_loc[:5].copy_(offsets)
        for graph in captured.values():
            assert graph.query_start_loc.tolist() == offsets.tolist()


def test_profile_and_configuration_assets_are_immutable(tmp_path):
    from tools.dspark.verification_tools import freeze_verification_config

    costs_path = tmp_path / "cost.json"
    costs_path.write_text('{"actual_measurement": 1}')
    original = tmp_path / "options.json"
    original.write_text(json.dumps({"mode": "confidence", "cost_profile": "cost.json"}))
    frozen = freeze_verification_config(original, tmp_path / "frozen")
    options = json.loads(frozen.read_text())
    costs_path.write_text("changed after planning")
    assert json.loads(Path(options["cost_profile"]).read_text()) == {"actual_measurement": 1}
    assert json.loads((frozen.parent / "hashes.json").read_text())["cost_profile"]
    with pytest.raises(FileExistsError):
        freeze_verification_config(original, frozen.parent)


@pytest.mark.parametrize("index_kind", ["standard", "quantized", "both"])
def test_checkpoint_checks_real_safetensor_payload_and_missing_weight(tmp_path, index_kind):
    import struct

    from tools.dspark.verification_tools import checkpoint_preflight

    name = "mtp.2.confidence_head.proj.weight"
    config = {"hidden_size": 2, "dspark_markov_rank": 1, "n_mtp_layers": 3}
    (tmp_path / "config.json").write_text(json.dumps(config))
    index = {"weight_map": {name: "model.safetensors"}}
    names = {"standard": "model.safetensors.index.json", "quantized": "quant_model_weights.safetensors.index.json"}
    selected = [names[index_kind]] if index_kind != "both" else list(names.values())
    for filename in selected:
        (tmp_path / filename).write_text(json.dumps(index))
    (tmp_path / "quant_model_description.json").write_text(json.dumps({name: "FLOAT"}))
    header = json.dumps({name: {"shape": [1, 3], "dtype": "F32", "data_offsets": [0, 12]}}).encode()
    weight = torch.tensor([1.0, 2.0, 3.0]).numpy().tobytes()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + weight)
    receipt = checkpoint_preflight(tmp_path)
    assert receipt["status"] == "present_not_runtime_loaded"
    assert receipt["shape"] == [1, 3] and receipt["checkpoint_weight_sha256"]
    assert receipt["index_file"] == selected[0]
    import hashlib

    assert receipt["index_sha256"] == hashlib.sha256((tmp_path / selected[0]).read_bytes()).hexdigest()
    assert set(receipt["available_indices_sha256"]) == set(selected)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="Missing confidence checkpoint shard"):
        checkpoint_preflight(tmp_path)
    for filename in selected:
        (tmp_path / filename).write_text(json.dumps({"weight_map": {"unrelated.weight": "model.safetensors"}}))
    with pytest.raises(ValueError, match="lacks real confidence"):
        checkpoint_preflight(tmp_path)


def test_confidence_mode_requires_weights_and_current_policy_decisions():
    before, after = snapshots()
    for row in after:
        row["confidence_verification"]["confidence_batches"] = 0
    with pytest.raises(ValueError, match="No measured confidence"):
        summarize_verification(before, after, 8)
    before, after = snapshots()
    for rows in (before, after):
        rows[0]["confidence_verification"]["weights"] = {}
    with pytest.raises(ValueError, match="weight provenance"):
        summarize_verification(before, after, 8)


def test_explicit_profile_compiles_rank_maxima_and_rejects_failed_source(tmp_path, monkeypatch):
    from tools.dspark import verification_tools as helpers

    identity = {"capture_sizes": [6, 24], "max_num_seqs": 4}
    rows = [
        {
            "rank": rank,
            "failed_execution_count": 0,
            "cost_profile": {
                "source": "isolated_npu_event_profile",
                "identity": identity,
                "measurements": [
                    {"kind": "target", "size": 6, "context": 32, "seconds": 0.001 + rank * 0.001},
                    {"kind": "target", "size": 24, "context": 128, "seconds": 0.008},
                    {"kind": "draft", "size": 4, "context": 64, "seconds": 0.002},
                ],
            },
        }
        for rank in range(2)
    ]
    data = {
        "performance_eligible": False,
        "effective_config": {"tensor_parallel_size": 2},
        "cleanup": {"engine_shutdown_complete": True},
        "graph_execution": {"boundary_snapshots": [rows]},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    # Synthetic timings exercise the compiler only, never emitted as real NPU evidence.
    monkeypatch.setattr(helpers, "measured_scheduler_overhead", lambda identity: 0.0001)
    table = helpers.compile_profile([path])
    assert table["target_seconds"] == {6: 0.002, 24: 0.008}
    assert table["context_range"] == [32, 128]
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="legacy schema 1"):
        CostTable.load(str(cache), identity)
    rows.pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing profile rank"):
        helpers.compile_profile([path])
    data["performance_eligible"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="relabel"):
        helpers.compile_profile([path])


def test_first_driver_executes_real_manifest_and_preserves_failure(tmp_path, monkeypatch):
    from tests.ut.test_dspark_performance_delivery import frozen
    from tools.dspark import run_confidence_verification as driver

    manifest, records = frozen(tmp_path)
    monkeypatch.setattr(driver.suite, "source_gate", lambda args: None)
    monkeypatch.setattr(driver, "checkpoint_preflight", lambda model: {"test_fixture": True})
    monkeypatch.setattr(driver.suite, "resources_idle", lambda path: None)
    commands = []

    def failed(command, log):
        commands.append(command)
        log.write_text("fixture engine startup failed")
        return 9

    monkeypatch.setattr(driver.suite, "logged", failed)
    directory = tmp_path / "first"
    result = driver.main(["--plugin-sha", "a" * 40, "--manifest", str(manifest), "--output-dir", str(directory)])
    assert result == 1 and len(commands) == 1
    assert len((directory / "input.jsonl").read_text().splitlines()) == 4
    receipt = json.loads((directory / "specified-b1/receipt.json").read_text())
    assert receipt["generation_rc"] == 9 and receipt["status"] == "failed"
    assert not (directory / "specified-b4").exists()
    assert driver.main(["--plugin-sha", "a" * 40, "--manifest", str(manifest), "--output-dir", str(directory)]) == 1
    assert len(commands) == 1  # No overwrite and no second model lifecycle.
    parsed = driver.benchmark.parse_args(commands[0][2:])
    assert parsed.max_num_seqs == 1 and parsed.num_prompts == 4
    assert parsed.cudagraph_capture_sizes == [6] and parsed.temperature == 0
    assert json.loads(parsed.confidence_verification.read_text())["mode"] == "specified_lengths"


def test_three_mode_plan_preserves_protocol_and_default_off(tmp_path):
    from tools.dspark import benchmark_dspark_acceptance as benchmark
    from tools.dspark import run_performance_suite as suite

    config = tmp_path / "confidence.json"
    config.write_text(json.dumps({"mode": "confidence", "cost_profile": "measured.json"}))
    args = suite.parse_args(
        [
            "--plugin-sha",
            "a" * 40,
            "--manifest",
            "unused",
            "--output-dir",
            str(tmp_path),
            "--max-num-seqs",
            "4",
            "--modes",
            "target_graph",
            "dspark_graph",
            "dspark_confidence_graph",
            "--confidence-verification",
            str(config),
            "--repeats",
            "3",
        ]
    )
    plan = suite.create_plan(args, tmp_path / "input.jsonl", tmp_path)
    assert len(plan["runs"]) == 9 and len({r["directory"] for r in plan["runs"]}) == 9
    for case in plan["runs"]:
        parsed = benchmark.parse_args(case["command"][2:])
        kwargs = benchmark.build_engine_kwargs(parsed)
        assert parsed.measurement_protocol == "async_stream" and not parsed.ignore_eos
        if case["mode"] == "target_graph":
            assert kwargs["speculative_config"] is None
        elif case["mode"] == "dspark_confidence_graph":
            assert kwargs["additional_config"]["dspark_confidence_verification"]["mode"] == "confidence"
        else:
            assert "dspark_confidence_verification" not in kwargs.get("additional_config", {})


def test_opt_in_rejects_unsupported_config_and_keeps_default_off():
    config = NS(additional_config={}, speculative_config=None)
    assert POLICY["verification_options"](config) is None
    config.additional_config = {"dspark_confidence_verification": {"mode": "specified_lengths", "lengths": [0, 5]}}
    config.speculative_config = NS(method="dspark", num_speculative_tokens=5, enforce_eager=True)
    config.parallel_config = NS(data_parallel_size=1, pipeline_parallel_size=1)
    config.model_config = NS(enforce_eager=False)
    config.compilation_config = NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY"))
    assert POLICY["verification_options"](config)["lengths"] == [0, 5]
    for target, field, bad in (
        (config.parallel_config, "data_parallel_size", 2),
        (config.speculative_config, "enforce_eager", False),
        (config.model_config, "enforce_eager", True),
    ):
        previous = getattr(target, field)
        setattr(target, field, bad)
        with pytest.raises(ValueError, match="requires K=5"):
            POLICY["verification_options"](config)
        setattr(target, field, previous)


def test_specified_validation_does_not_enable_performance_or_nan_comparison():
    from tools.dspark.summarize_dspark_acceptance_benchmark import _validate_result

    result = {
        "performance_eligible": False,
        "confidence_verification": {"mode": "specified_lengths", "status": "available"},
    }
    with pytest.raises(ValueError, match="not eligible"):
        _validate_result(result, "dspark")
    # Explicit test validation still executes all artifact checks.
    with pytest.raises(ValueError, match="schema"):
        _validate_result(result, "dspark", specified_verification_test=True)
    result["nan_diagnostic"] = {"enabled": True}
    with pytest.raises(ValueError, match="not eligible"):
        _validate_result(result, "dspark", specified_verification_test=True)


def test_offline_calibration_fit_requires_separate_labeled_split(tmp_path):
    from tools.dspark.verification_tools import calibrate

    path = tmp_path / "calibration.json"
    samples = {
        "split": "calibration",
        "weights_sha256": "fixture-weights",
        "conditional_logits": [-2, -1, 0, 0, 1, 2],
        "conditional_accepted": [0, 1, 0, 1, 0, 1],
    }
    path.write_text(json.dumps(samples))
    result = calibrate(path, "fixture-weights")
    assert result["scale"] > 0 and np.isfinite(result["bias"])
    assert result["n"] == 6 and result["dataset_sha256"]
    samples["split"] = "evaluation"
    path.write_text(json.dumps(samples))
    with pytest.raises(ValueError, match="separate calibration"):
        calibrate(path, "fixture-weights")


@pytest.mark.parametrize("conflict", ["shard", "missing_confidence", "other_weight"])
def test_checkpoint_index_conflicts_fail_before_shard_selection(tmp_path, conflict):
    from tools.dspark.verification_tools import checkpoint_preflight

    name = "mtp.2.confidence_head.proj.weight"
    mapping = {name: "model.safetensors", "other.weight": "other.safetensors"}
    quantized = dict(mapping)
    if conflict == "shard":
        quantized[name] = "quant.safetensors"
    elif conflict == "missing_confidence":
        del quantized[name]
    else:
        quantized["other.weight"] = "different.safetensors"
    (tmp_path / "config.json").write_text("{}")
    for filename, weights in (
        ("model.safetensors.index.json", mapping),
        ("quant_model_weights.safetensors.index.json", quantized),
    ):
        (tmp_path / filename).write_text(json.dumps({"weight_map": weights}))
    with pytest.raises(ValueError, match="Conflicting checkpoint index"):
        checkpoint_preflight(tmp_path)
