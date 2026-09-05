# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Real CPU tensor boundary tests; no claim about NPU kernels/stream ordering.

Only import dependencies are isolated when loading speculator method bodies.
The new observer, reductions, row selection, file writes and guards execute.
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from tests.ut.test_dspark_graph_replay import _EXTENSION, ReplayManager
from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import summarize_dspark_acceptance_benchmark as summary

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[2]
SPECULATOR = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
_SPEC = importlib.util.spec_from_file_location("p08_nan", ROOT / "vllm_ascend/diagnostics/dspark_nan.py")
_NAN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NAN)


@pytest.fixture(autouse=True)
def cpu_capture_state(monkeypatch):
    # These are CPU tests even on an NPU host. Exercise the guard explicitly in
    # the capture test; do not initialize a real NPU just to inspect CPU data.
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)


def _methods(*names):
    tree = ast.parse(SPECULATOR.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendDSparkSpeculator")
    nodes = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {
        "torch": torch,
        "np": np,
        "MappingProxyType": MappingProxyType,
        "AscendDSparkDraftExecution": SimpleNamespace,
        "AscendDSparkProposalInputs": SimpleNamespace,
        "InputBatch": SimpleNamespace,
    }
    top = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_assert_markov_tensor_contract"
    ]
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *top, *nodes], type_ignores=[])
    exec(compile(module, str(SPECULATOR), "exec"), namespace)
    return type("ProductionMethods", (), {name: namespace[name] for name in names})


def _scheduler(size=4):
    return SimpleNamespace(num_scheduled_tokens={f"r{i}": 6 for i in range(size)}, total_num_scheduled_tokens=6 * size)


def _diagnostic(tmp_path, rank=0):
    diagnostic = _NAN.DSparkNaNDiagnostics(str(tmp_path), rank)
    diagnostic.phase = "warmup"
    diagnostic.begin_execution(_scheduler())
    return diagnostic


def _report(tmp_path, kind="latest", rank=0):
    return json.loads((tmp_path / f"rank-{rank}-{kind}.json").read_text())


def _proposal(size=4, epoch=1):
    tokens = size * 6
    slots = torch.arange(tokens, dtype=torch.int32)
    slots[-1] = -1
    aux = tuple(torch.ones(tokens, 8) * i for i in (1, 2, 3))
    return SimpleNamespace(
        step_epoch=epoch,
        request_ids=tuple(f"r{i}" for i in range(size)),
        num_target_tokens=tokens,
        num_query_tokens=size * 5,
        draft_query_start_loc=torch.arange(size + 1) * 5,
        draft_positions=torch.arange(size * 5),
        draft_sequence_lengths=torch.full((size,), 100),
        draft_layer_group_ids={"draft": 0},
        last_hidden_states=torch.ones(tokens, 8),
        auxiliary_hidden_states=aux,
        draft_context_slot_mappings={"draft": slots},
        draft_input_ids=torch.zeros(size * 5, dtype=torch.int32),
        target_positions=torch.arange(tokens),
    )


@pytest.mark.parametrize(
    "boundary", ["target_outputs", "proposal_inputs", "combined_context", "draft_hidden", "base_logits"]
)
def test_first_nonfinite_boundary_is_preserved_and_no_values_are_dumped(tmp_path, boundary):
    diagnostic = _diagnostic(tmp_path)
    value = torch.ones(24, 8)
    diagnostic.check("previous_finite", {"hidden": value}, 24)
    value[7, 2] = torch.nan
    with pytest.raises(RuntimeError, match=boundary) as error:
        diagnostic.check(boundary, {"hidden": value}, 24)
    first = (tmp_path / "rank-0-first-failure.json").read_bytes()
    diagnostic.failed_execution("outer", error.value)
    assert (tmp_path / "rank-0-first-failure.json").read_bytes() == first
    record = _report(tmp_path, "first-failure")
    assert record["current"]["stage"] == boundary
    stats = record["current"]["checks"][-1]["tensors"]["hidden"]
    assert stats["nan_rows_first"] == [7] and stats["nan_row_count"] == 1
    assert stats["data_ptr"] == value.data_ptr()
    assert "values" not in stats and record["performance_eligible"] is False


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_live_target_shapes_alias_updates_reorder_and_phases(tmp_path, size):
    diagnostic = _diagnostic(tmp_path)
    persistent = torch.ones(24, 8)
    batch = SimpleNamespace(
        req_ids=[f"r{i}" for i in range(size)],
        num_reqs=size,
        num_tokens=size * 6 - 1,
        num_tokens_after_padding=size * 6,
        query_start_loc_np=np.arange(size + 1) * 6,
        seq_lens=torch.full((size,), 100),
        positions=torch.arange(size * 6),
    )
    batch.query_start_loc_np[-1] -= 1
    state = SimpleNamespace(
        input_batch=batch, hidden_states=persistent[: size * 6], aux_hidden_states=[persistent[: size * 6]] * 3
    )
    runner = SimpleNamespace(execute_model_state=state)
    persistent[batch.num_tokens :] = torch.nan  # Not valid rows; must not trip diagnostics.
    diagnostic.target_completed(runner, [SimpleNamespace(num_tokens=size * 6)])
    first_ptr = diagnostic.current["checks"][0]["tensors"]["target_aux_0"]["data_ptr"]
    diagnostic.phase = "measured"
    diagnostic.begin_execution(_scheduler(size))
    batch.req_ids.reverse()
    batch.positions.add_(32)
    persistent[: batch.num_tokens].fill_(2)
    diagnostic.target_completed(runner, [SimpleNamespace(num_tokens=size * 6)])
    record = _report(tmp_path)
    assert record["current"]["actual_full_shapes"] == [size * 6]
    assert record["current"]["positions"][0] == 32
    assert record["current"]["request_ids"] == batch.req_ids
    assert record["current"]["phase"] == "measured"
    assert record["previous_executions"][0]["phase"] == "warmup"
    assert record["current"]["checks"][0]["tensors"]["target_aux_0"]["data_ptr"] == first_ptr
    for _ in range(4):
        diagnostic.begin_execution(_scheduler(size))
    assert len(_report(tmp_path)["previous_executions"]) == 2


@pytest.mark.parametrize("size", [1, 2, 4])
def test_exact_padding_current_context_only_and_kv_identity(tmp_path, size):
    diagnostic, proposal = _diagnostic(tmp_path), _proposal(size)
    cache = torch.full((2, 32, 1, 8), torch.nan)
    slots = proposal.draft_context_slot_mappings["draft"]
    valid = slots[slots != -1].long()
    cache[valid // 32, valid % 32] = 3  # CPU reference write: exact -1 never writes.
    before = cache.clone()
    diagnostic.proposal_inputs(proposal)
    diagnostic.context_kv(proposal, {"draft": [cache]}, [32])
    torch.testing.assert_close(cache, before, equal_nan=True)
    assert diagnostic.current["context_kv"]["draft"]["valid_target_rows"] == list(range(size * 6 - 1))
    assert diagnostic.current["context_kv"]["draft"]["cache_data_ptr"] == cache.data_ptr()
    # Same persistent buffer, new real block and request state on the next step.
    diagnostic.begin_execution(_scheduler(size))
    proposal.step_epoch += 1
    slots[slots != -1] += 32
    cache[1, 0 : size * 6 - 1] = 5
    diagnostic.proposal_inputs(proposal)
    diagnostic.context_kv(proposal, {"draft": cache}, [32])
    assert diagnostic.current["context_kv"]["draft"]["context_slots"][0] == 32
    cache[1, 0] = torch.nan
    with pytest.raises(RuntimeError, match="context_kv_written:draft"):
        diagnostic.context_kv(proposal, {"draft": cache}, [32])


@pytest.mark.parametrize("slot", [-2, 64])
def test_invalid_real_slots_are_not_clamped(tmp_path, slot):
    diagnostic, proposal = _diagnostic(tmp_path), _proposal(1)
    proposal.draft_context_slot_mappings["draft"][0] = slot
    with pytest.raises(ValueError, match="Invalid diagnostic context slot"):
        diagnostic.context_kv(proposal, {"draft": torch.zeros(2, 32, 1, 8)}, [32])


def test_all_dummy_no_kv_mutation_wrong_block_size_and_capture_guard(tmp_path, monkeypatch):
    diagnostic, proposal = _diagnostic(tmp_path), _proposal(1)
    proposal.draft_context_slot_mappings["draft"].fill_(-1)
    cache = torch.full((2, 32, 1, 8), torch.nan)
    diagnostic.context_kv(proposal, {"draft": cache}, [32])
    assert torch.isnan(cache).all()
    with pytest.raises(ValueError, match="block size"):
        diagnostic.context_kv(proposal, {"draft": cache}, [128])
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    monkeypatch.setattr(diagnostic, "_stats", lambda *args: pytest.fail("device check inside capture"))
    with pytest.raises(RuntimeError, match="outside ACLGraph capture"):
        diagnostic.check("capture", {"hidden": cache}, 1)


def test_masked_logits_negative_infinity_allowed_but_other_nonfinite_rejected(tmp_path):
    diagnostic = _diagnostic(tmp_path)
    value = torch.tensor([[1.0, -torch.inf]])
    diagnostic.check("base_logits", {"base": value}, 1, allow_negative_infinity=True)
    value[0, 0] = torch.inf
    with pytest.raises(RuntimeError, match="base_logits"):
        diagnostic.check("base_logits", {"base": value}, 1, allow_negative_infinity=True)


def test_real_execute_draft_guard_and_disabled_path(tmp_path):
    cls = _methods("_execute_draft")
    speculator = cls()
    proposal = _proposal()
    calls = []
    speculator.validate_prepared_inputs_current = lambda p: calls.append("validate")
    speculator._execute_draft_backbone = lambda p: torch.ones(20, 8)
    speculator._execute_sequential_markov_sampling = lambda p, h: calls.append("markov") or h
    speculator._build_core_proposal = lambda p, r: calls.append("publish") or r
    speculator._execute_draft(proposal)
    assert calls == ["validate", "markov", "publish"]
    speculator._nan_diagnostic = _diagnostic(tmp_path)
    speculator._execute_draft_backbone = lambda p: torch.full((20, 8), torch.nan)
    calls.clear()
    with pytest.raises(RuntimeError, match="draft_hidden"):
        speculator._execute_draft(proposal)
    assert calls == ["validate"]  # No Markov or publication after nonfinite hidden.
    assert _report(tmp_path, "first-failure")["current"]["stage"] == "draft_hidden"


def test_real_markov_body_diagnostic_precedes_original_nan_assert(tmp_path):
    cls = _methods("_execute_sequential_markov_sampling")
    speculator = cls()
    speculator._markov_attempt_step_epoch = None
    speculator._proposal_step_epoch = 1
    speculator._validate_markov_inputs = lambda p, h: (4, 5, 20)
    speculator._require_greedy_markov_sampling = lambda p: None
    speculator._markov_module_contract = {"lm_head_id": 1, "markov_head_id": 2, "confidence_head_id": 3}
    speculator._inspect_markov_modules = lambda model: speculator._markov_module_contract
    speculator.model = SimpleNamespace(compute_draft_logits=lambda h: torch.full((20, 16), torch.nan))
    speculator._validate_step_tensor = lambda name, tensor, *, ndim: tensor
    speculator.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=16))
    proposal = _proposal()
    for enabled in (False, True):
        speculator._markov_attempt_step_epoch = None
        if enabled:
            speculator._nan_diagnostic = _diagnostic(tmp_path)
        with pytest.raises(
            RuntimeError if enabled else ValueError, match="base_logits" if enabled else "base logits contain NaN"
        ):
            speculator._execute_sequential_markov_sampling(proposal, torch.ones(20, 8))
        assert speculator._markov_result is None and speculator._markov_step_epoch is None
        with pytest.raises(RuntimeError, match="already attempted"):
            speculator._execute_sequential_markov_sampling(proposal, torch.ones(20, 8))


class _TensorRunner:
    def __init__(self, directory):
        self.vllm_config = SimpleNamespace(additional_config={"dspark_nan_diagnostic_dir": str(directory)})
        self.speculator = SimpleNamespace(rank=0)
        self.cudagraph_manager = ReplayManager()
        self.value = torch.ones(24, 8)

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
    ):
        descriptor = SimpleNamespace(cg_mode="FULL", num_tokens=24)
        self.cudagraph_manager.run_fullgraph(descriptor)
        self.execute_model_state = SimpleNamespace(
            hidden_states=self.value,
            aux_hidden_states=[self.value] * 3,
            input_batch=SimpleNamespace(
                req_ids=list(scheduler_output.num_scheduled_tokens),
                num_reqs=4,
                num_tokens=24,
                num_tokens_after_padding=24,
                query_start_loc_np=np.arange(5) * 6,
                seq_lens=torch.full((4,), 100),
                positions=torch.arange(24),
            ),
        )
        return self.value


def test_extension_install_phase_and_failed_target_never_counts(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_nan", _NAN)
    worker = _EXTENSION.DSparkBenchmarkWorkerExtension()
    worker.rank, worker.model_runner = 0, _TensorRunner(tmp_path)
    first = worker.dspark_benchmark_replay_snapshot(diagnostic_phase="warmup")
    diagnostic = worker.model_runner.speculator._nan_diagnostic
    worker.model_runner.execute_model(_scheduler(), dummy_run=True)
    assert not list(tmp_path.glob("*.json"))  # No capture/dummy diagnostics.
    worker.model_runner.execute_model(_scheduler())
    second = worker.dspark_benchmark_replay_snapshot(diagnostic_phase="measured")
    assert first["observer_id"] == second["observer_id"]
    assert worker.model_runner.speculator._nan_diagnostic is diagnostic
    assert second["records"][0]["count"] == 1
    worker.model_runner.value[1] = torch.nan
    with pytest.raises(RuntimeError, match="target_outputs"):
        worker.model_runner.execute_model(_scheduler())
    failed = worker.dspark_benchmark_replay_snapshot(diagnostic_phase="complete")
    assert failed["records"] == second["records"] and failed["failed_execution_count"] == 1
    assert _report(tmp_path, "first-failure")["current"]["phase"] == "measured"
    # The safe RPC payload remains primitive even when diagnostics are active.
    msgspec = pytest.importorskip("msgspec")
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(failed)) == failed


def test_rank_files_independent_and_performance_gate(tmp_path):
    for rank in range(8):
        diagnostic = _diagnostic(tmp_path, rank)
        diagnostic.check("finite", {"hidden": torch.ones(6, 8)}, 6)
        assert _report(tmp_path, rank=rank)["rank"] == rank
    with pytest.raises(ValueError, match="not eligible"):
        summary._validate_result({"performance_eligible": False}, "dspark")


def test_cli_optin_and_no_performance_publication(tmp_path, monkeypatch, capsys):
    args = [
        "--model-dir",
        str(tmp_path),
        "--mode",
        "dspark",
        "--dataset-name",
        "jsonl",
        "--dataset-path",
        str(tmp_path / "input.jsonl"),
        "--result-json",
        str(tmp_path / "result.json"),
    ]
    assert benchmark.parse_args(args).dspark_nan_diagnostic_dir is None
    with pytest.raises(SystemExit):
        benchmark.parse_args([*args, "--dspark-nan-diagnostic-dir", str(tmp_path)])
    args += [
        "--target-execution-mode",
        "full_decode_only",
        "--cudagraph-capture-sizes",
        "6",
        "12",
        "18",
        "24",
        "--dspark-nan-diagnostic-dir",
        str(tmp_path),
    ]
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    kwargs = benchmark.build_engine_kwargs(benchmark.parse_args(args))
    assert kwargs["additional_config"] == {"dspark_nan_diagnostic_dir": str(tmp_path)}
    assert kwargs["speculative_config"]["enforce_eager"] is True
    monkeypatch.setattr(benchmark, "run_benchmark", lambda args: {"performance_eligible": False, "graph_execution": {}})
    monkeypatch.setattr(benchmark, "_print_summary", lambda result: pytest.fail("published diagnostic throughput"))
    assert benchmark.main(args) == 0
    assert "DSPARK_PR_STYLE_BENCHMARK_PASS" not in capsys.readouterr().out


def _real_preparation(size, diagnostic=None):
    cls = _methods(
        "prepare_proposal_inputs",
        "_validate_step_tensor",
        "_validate_aux_hidden_states",
        "_build_query_slot_mappings",
        "validate_prepared_inputs_current",
    )
    speculator = cls()
    speculator.rank, speculator.device = 0, torch.device("cpu")
    speculator._model, speculator.kv_cache_config = object(), object()
    speculator._loaded_target_model = SimpleNamespace(model=SimpleNamespace(aux_hidden_state_layers=(41, 42, 43)))
    speculator.target_layer_ids, speculator.num_speculative_steps = (40, 41, 42), 5
    speculator.parallel_drafting_token_id = 127
    speculator._proposal_step_epoch, speculator._prepared_step_epoch = 0, None
    speculator._published_candidate_tokens = None
    speculator.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))
    speculator.draft_attn_layer_order = ("draft0", "draft1", "draft2")
    speculator.draft_layer_group_ids = {"draft0": 0, "draft1": 1, "draft2": 1}
    speculator.block_tables = SimpleNamespace(
        slot_mappings=torch.full((2, 24), -1, dtype=torch.int32),
        kernel_block_sizes=[32, 16],
        input_block_tables=[torch.arange(size * 8, dtype=torch.int32).reshape(size, 8) + i * 100 for i in (0, 1)],
    )
    if diagnostic is not None:
        speculator._nan_diagnostic = diagnostic
    slots0, slots1 = speculator.block_tables.slot_mappings
    slots0[: size * 6] = torch.arange(size * 6)
    slots1[: size * 6] = torch.arange(size * 6) + 100
    tokens = size * 6
    batch = SimpleNamespace(
        num_reqs=size,
        num_tokens=tokens,
        num_tokens_after_padding=tokens,
        req_ids=[f"r{i}" for i in range(size)],
        idx_mapping_np=np.arange(size),
        idx_mapping=torch.arange(size, dtype=torch.int32),
        input_ids=torch.arange(tokens, dtype=torch.int32),
        positions=torch.arange(tokens),
        query_start_loc=torch.arange(size + 1, dtype=torch.int32) * 6,
        seq_lens=torch.arange(1, size + 1, dtype=torch.int32) * 6,
        is_prefilling_np=np.zeros(size, dtype=bool),
    )
    kwargs = dict(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={"draft0": slots0, "draft1": slots1, "draft2": slots1},
        last_hidden_states=torch.ones(tokens, 8),
        aux_hidden_states=[torch.ones(tokens, 8) * i for i in (1, 2, 3)],
        num_sampled=torch.ones(size, dtype=torch.int32),
        num_rejected=torch.zeros(size, dtype=torch.int32),
        last_sampled=torch.arange(size),
        next_prefill_tokens=torch.zeros(size, dtype=torch.int32),
        temperature=torch.zeros(size),
        seeds=torch.arange(size),
    )
    return speculator, kwargs


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_preparation_rows_group_identity_epoch_and_ownership(tmp_path, size, enabled):
    diagnostic = _diagnostic(tmp_path) if enabled else None
    speculator, kwargs = _real_preparation(size, diagnostic)
    context_before = kwargs["slot_mappings"]["draft0"][: size * 6].clone()
    aux = kwargs["aux_hidden_states"]
    proposal = speculator.prepare_proposal_inputs(**kwargs)
    assert proposal.num_query_tokens == size * 5 and proposal.num_target_tokens == size * 6
    assert proposal.auxiliary_hidden_states[1] is aux[1]
    assert proposal.draft_context_slot_mappings["draft1"] is proposal.draft_context_slot_mappings["draft2"]
    torch.testing.assert_close(proposal.draft_context_slot_mappings["draft0"], context_before)
    ptr = speculator.block_tables.slot_mappings.data_ptr()
    speculator.validate_prepared_inputs_current(proposal)
    speculator._published_candidate_tokens = object()
    with pytest.raises(RuntimeError, match="consume the active published proposal"):
        speculator.prepare_proposal_inputs(**kwargs)
    speculator._published_candidate_tokens = None
    kwargs["input_batch"].req_ids.reverse()
    kwargs["input_batch"].positions.add_(32)
    if diagnostic:
        diagnostic.begin_execution(_scheduler(size))
    next_proposal = speculator.prepare_proposal_inputs(**kwargs)
    assert next_proposal.step_epoch == proposal.step_epoch + 1
    with pytest.raises(RuntimeError, match="stale"):
        speculator.validate_prepared_inputs_current(proposal)
    assert speculator.block_tables.slot_mappings.data_ptr() == ptr
    if diagnostic:
        assert diagnostic.current["proposal_epoch"] == next_proposal.step_epoch
        assert diagnostic.current["proposal_request_ids"] == kwargs["input_batch"].req_ids
        assert diagnostic.current["checks"][-1]["boundary"] == "proposal_inputs"
    # Actual prepare hook catches a bad auxiliary view on the next call.
    if diagnostic:
        diagnostic.begin_execution(_scheduler(size))
        aux[2][0, 0] = torch.nan
        with pytest.raises(RuntimeError, match="proposal_inputs"):
            speculator.prepare_proposal_inputs(**kwargs)


def test_real_context_precompute_consumes_before_write_and_diagnoses_written_kv(tmp_path):
    cls = _methods("_combine_and_precompute_draft_context", "_validate_step_tensor")
    speculator, proposal = cls(), _proposal(4)
    speculator.device = torch.device("cpu")
    speculator._context_kv_step_epoch = None
    speculator._prepared_step_epoch = 1
    speculator._nan_diagnostic = _diagnostic(tmp_path)
    speculator._validate_draft_backbone_inputs = lambda p: None  # Model loading is outside this CPU fixture.
    speculator.audit_target_draft_cache_isolation = lambda: {
        "target_cache_object_alias_count": 0,
        "target_cache_byte_range_overlap_count": 0,
    }
    speculator.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=8))
    speculator.draft_attn_layer_order = ("draft",)
    cache = torch.zeros(2, 32, 1, 8)
    speculator.draft_kv_caches = {"draft": cache}
    speculator.block_tables = SimpleNamespace(kernel_block_sizes=[32])

    def write_context(context_states, positions, slot_mappings):
        assert speculator._context_kv_step_epoch == proposal.step_epoch
        assert speculator._prepared_step_epoch is None
        slots = slot_mappings[0]
        valid = slots != -1
        cache[slots[valid].long() // 32, slots[valid].long() % 32, 0] = context_states[valid]
        cache[0, 7] = torch.nan  # Fault at write boundary, after finite combined context.

    speculator.model = SimpleNamespace(
        combine_hidden_states=lambda aux: aux[:, :8], precompute_and_store_context_kv=write_context
    )
    with pytest.raises(RuntimeError, match="context_kv_written"):
        speculator._combine_and_precompute_draft_context(proposal)
    assert speculator._prepared_step_epoch is None
    first = _report(tmp_path, "first-failure")["current"]
    assert first["checks"][0]["boundary"] == "combined_context"
    assert first["checks"][0]["tensors"]["context_hidden"]["invalid"] is False
    assert first["checks"][1]["tensors"]["written_kv_rows"]["nan_rows_first"] == [7]
    with pytest.raises(RuntimeError, match="already consumed"):
        speculator._combine_and_precompute_draft_context(proposal)


@pytest.mark.parametrize("insecure", [None, "0"])
def test_phase_keywords_cross_frozen_core_safe_rpc_boundary(tmp_path, monkeypatch, insecure):
    # Frozen frontend request encoding, EngineCore utility, extension injection
    # and WorkerProc string dispatch. Transport queues and NPU model are CPU fixtures.
    from tests.ut import test_dspark_graph_rpc as rpc

    if insecure is None:
        monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    else:
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", insecure)
    api = rpc._boundary(monkeypatch)
    workers = rpc._workers(api, monkeypatch, benchmark._GRAPH_WORKER_EXTENSION)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_nan", _NAN)
    for rank, worker in enumerate(workers):
        worker.worker.model_runner = _TensorRunner(tmp_path)
        worker.worker.model_runner.speculator.rank = rank
    executor = rpc._executor(api, workers)

    def serve(payload, extension=None):
        engine = api.EngineCore()
        engine.model_executor = executor
        _, call_id, method_name, args = api.MsgpackDecoder().decode(payload)
        method = getattr(engine, method_name)
        output = api.UtilityOutput(call_id)
        api.EngineCoreProc._invoke_utility_method(
            method_name, lambda: method(*api.EngineCoreProc._convert_msgspec_args(method, args)), output, lambda _: None
        )
        return api.MsgpackEncoder().encode(output)[0]

    monkeypatch.setattr(rpc, "_serve", serve)
    frontend, sent = rpc._frontend(api)
    for phase in ("warmup", "measured", "complete"):
        records = frontend.collective_rpc(benchmark._REPLAY_SNAPSHOT_METHOD, kwargs={"diagnostic_phase": phase})
        rpc._assert_basic(records)
        assert all(worker.model_runner.speculator._nan_diagnostic.phase == phase for worker in workers)
        if phase != "complete":
            for worker in workers:
                worker.model_runner.execute_model(_scheduler())
    assert len(sent) == 3
    assert all(record["records"][0]["count"] == 2 for record in records)


def test_existing_rank_directory_refused_instead_of_mixing_runs(tmp_path):
    _diagnostic(tmp_path)
    with pytest.raises(ValueError, match="fresh rank output directory"):
        _NAN.DSparkNaNDiagnostics(str(tmp_path), 0)


@pytest.mark.parametrize("fail_final_rpc", [False, True])
def test_benchmark_diagnostic_phases_and_raw_timing_marked_ineligible(tmp_path, monkeypatch, fail_final_rpc):
    from tests.ut.test_dspark_acceptance_benchmark import _FakeEngine, _graph_args

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    args = _graph_args(tmp_path)
    args.dspark_nan_diagnostic_dir = tmp_path / "diagnostics"
    engine = _FakeEngine(args)
    original = engine.collective_rpc
    phases = []

    def collective_rpc(method, *, kwargs=None):
        if kwargs is not None:
            assert method == benchmark._REPLAY_SNAPSHOT_METHOD
            assert set(kwargs) == {"diagnostic_phase"}
            phases.append(kwargs["diagnostic_phase"])
            if fail_final_rpc and kwargs["diagnostic_phase"] == "complete":
                raise RuntimeError("Final diagnostic RPC failed after outputs returned")
        return original(method)

    engine.collective_rpc = collective_rpc
    monkeypatch.setattr(benchmark, "_git_head", lambda path: "a" * 40)
    result = benchmark.run_benchmark(
        args,
        engine_factory=lambda kw: engine,
        sampling_factory=lambda args: object(),
        clock=iter((10.0, 12.0)).__next__,
        plugin_root=tmp_path,
        core_root=tmp_path,
    )
    assert phases == ["warmup", "measured", "complete"]
    assert result["performance_eligible"] is False and result["nan_diagnostic"]["enabled"] is True
    assert result["timing"]["explicit_device_synchronization"] is True
    assert result["timing"]["graph_telemetry_rpc_included"] is False
    assert result["outputs"] and result["timing"]["elapsed_seconds"] == 2.0
    if fail_final_rpc:
        assert result["graph_execution"]["replay_evidence_status"] == "unavailable"
    with pytest.raises(ValueError, match="not eligible"):
        summary._validate_result(result, "dspark")
