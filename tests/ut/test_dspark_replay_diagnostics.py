# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Executable CPU tensors/ATen replay, not a claim about NPU graph support.

The recorder replays captured ATen operations without calling model.forward.
Target decoder method and MRV2 replay dispatch are loaded from actual source;
only hardware kernels/transports are CPU fixtures.
"""

import ast
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

from tests.ut.test_dspark_nan_diagnostics import _NAN, _scheduler
from tools.dspark import benchmark_dspark_acceptance as benchmark

torch = pytest.importorskip("torch")
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_map  # noqa: E402

ROOT = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("p08_replay", ROOT / "vllm_ascend/diagnostics/dspark_replay.py")
replay = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = replay  # Dynamo resolves the real method's global torch import.
_previous = sys.modules.get("vllm_ascend.diagnostics.dspark_nan")
sys.modules["vllm_ascend.diagnostics.dspark_nan"] = _NAN
try:
    _spec.loader.exec_module(replay)
finally:
    if _previous is None:
        del sys.modules["vllm_ascend.diagnostics.dspark_nan"]
    else:
        sys.modules["vllm_ascend.diagnostics.dspark_nan"] = _previous


class CPURecordedGraph(TorchDispatchMode):
    def __init__(self):
        self.operations = []
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        self.operations.append((func, args, kwargs, output))
        return output

    def replay(self):
        self.calls += 1
        replacements = {}

        def resolve(value):
            return replacements.get(id(value), value) if isinstance(value, torch.Tensor) else value

        for func, args, kwargs, original in self.operations:
            result = func(*tree_map(resolve, args), **tree_map(resolve, kwargs))
            if isinstance(original, torch.Tensor):
                replacements[id(original)] = result
            elif isinstance(original, (list, tuple)):
                for old, new in zip(original, result):
                    if isinstance(old, torch.Tensor):
                        replacements[id(old)] = new
        return None


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    backend = getattr(torch, "npu", None)
    if backend is None:
        monkeypatch.setattr(torch, "npu", NS(is_current_stream_capturing=lambda: False), raising=False)
    else:
        # Dynamo queries accelerator availability through the installed module.
        # Keep that module and its interfaces; only the capture query is mocked.
        monkeypatch.setattr(backend, "is_current_stream_capturing", lambda: False)


@pytest.mark.parametrize("initial_state", ["existing", "module", "missing", "none"])
@pytest.mark.parametrize("body_raises", [False, True])
def test_cpu_only_preserves_backend_and_restores_between_scopes(monkeypatch, initial_state, body_raises):
    # On an NPU host, "existing" uses the installed backend itself. The module
    # case also exercises its interface/identity contract on a pure CPU host.
    backend = torch.npu
    if initial_state == "module":
        backend = ModuleType("torch.npu")
        backend.is_available = lambda: True
        backend.current_stream = lambda: "backend stream"
        backend.is_current_stream_capturing = lambda: True
        monkeypatch.setattr(torch, "npu", backend)
    elif initial_state == "missing":
        monkeypatch.delattr(torch, "npu")
        backend = None
    elif initial_state == "none":
        monkeypatch.setattr(torch, "npu", None)
        backend = None
    original_attributes = dict(vars(backend)) if backend is not None else {}
    accelerator_query = torch.accelerator.is_available

    # Repeat the complete setup/teardown, including exceptional test bodies,
    # to detect state leaking into the next test in the same process.
    for _ in range(2):
        expectation = pytest.raises(RuntimeError, match="fixture body failed") if body_raises else nullcontext()
        with expectation, pytest.MonkeyPatch.context() as scoped:
            cpu_only.__wrapped__(scoped)
            assert torch.npu.is_current_stream_capturing() is False
            assert torch.accelerator.is_available is accelerator_query
            if backend is not None:
                assert torch.npu is backend
                assert vars(backend).keys() == original_attributes.keys()
                for name, value in original_attributes.items():
                    if name != "is_current_stream_capturing":
                        assert vars(backend)[name] is value
                if initial_state == "module":
                    assert torch.npu.is_available() is True
                    assert torch.npu.current_stream() == "backend stream"
            else:
                assert set(vars(torch.npu)) == {"is_current_stream_capturing"}
            if body_raises:
                raise RuntimeError("fixture body failed")
        assert torch.accelerator.is_available is accelerator_query
        if initial_state == "missing":
            assert not hasattr(torch, "npu")
        else:
            assert torch.npu is backend
        if backend is not None:
            assert vars(backend).keys() == original_attributes.keys()
            assert all(vars(backend)[name] is value for name, value in original_attributes.items())


def bank(rank=0, layers=(0, 1)):
    return replay.TargetLayerSnapshots(
        sizes=[6, 12, 18, 24],
        query_len=6,
        hidden_size=8,
        hc_mult=4,
        layers=layers,
        dtype=torch.float32,
        device="cpu",
        rank=rank,
    )


def _production_method(path, cls_name, name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, method], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def decoder(index, snapshots, fault):
    forward = _production_method(
        ROOT / "vllm_ascend/models/deepseek_v4.py", "DeepseekV2DecoderLayer", "forward", {"torch": torch}
    )
    cls = type("ActualDecoderBody", (torch.nn.Module,), {"forward": forward})
    layer = cls()
    layer.layer_idx, layer._dspark_layer_snapshots = index, snapshots
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        setattr(layer, name, None)
    layer.hc_pre = lambda x, *params: (x.mean(1), None, None)
    layer.hc_post = lambda x, residual, post, comb: residual + x[:, None, :]
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.self_attn = lambda *, hidden_states, **kwargs: hidden_states + fault
    layer.mlp = lambda x, **kwargs: x * 0.5
    return layer


def runner_fixture(size=24, requests=4):
    n = requests * 6
    positions = torch.cat([torch.arange(32, 38) + i for i in range(requests)])
    padded_positions = torch.cat((positions, torch.zeros(size - n, dtype=torch.int64)))
    offsets = torch.arange(requests + 1, dtype=torch.int32) * 6
    input_ids = torch.arange(size, dtype=torch.int32) + 7
    seq_lens = torch.tensor([38 + i for i in range(requests)], dtype=torch.int32)
    inputs = NS(
        input_ids=input_ids,
        positions=padded_positions,
        query_start_loc=offsets,
        seq_lens=seq_lens,
        is_padding=torch.arange(size) >= n,
    )
    batch = NS(
        **vars(inputs),
        req_ids=[f"fresh-request-{i}" for i in range(requests)],
        num_tokens=n,
        num_tokens_after_padding=size,
        num_reqs=requests,
        idx_mapping_np=np.arange(requests)[::-1].copy(),
        idx_mapping=torch.arange(requests - 1, -1, -1),
        query_start_loc_np=offsets.numpy(),
        seq_lens_cpu_upper_bound=seq_lens + 5,
    )
    table = torch.arange(requests * 3, dtype=torch.int32).reshape(requests, 3) + 1
    source_table = table.flip(0).clone()
    raw = torch.tensor(
        [
            int(table[i, int(pos) // 32]) * 32 + int(pos) % 32
            for i in range(requests)
            for pos in positions[i * 6 : (i + 1) * 6]
        ]
        + [-1] * (size - n),
        dtype=torch.int32,
    )
    metadata = NS(
        decode=NS(
            query_start_loc=offsets,
            seq_lens=seq_lens,
            input_positions=padded_positions,
            start_pos=seq_lens - 6,
            slot_mapping=torch.stack((raw // 32, raw % 32), 1),
            sas_metadata=torch.arange(32, dtype=torch.int32),
            qli_metadata=torch.arange(16, dtype=torch.int32),
            sin=torch.ones(size, 1, 8),
            cos=torch.ones(size, 1, 8),
            block_table=table,
            block_size=32,
            num_reqs_actual=requests,
        ),
        prefill=None,
    )
    name = "model.layers.0.self_attn.attn"
    tables = NS(
        num_blocks=NS(np=np.full((1, requests), 3)),
        input_block_tables=[table],
        block_tables=[NS(gpu=source_table)],
        slot_mappings=raw[None, :],
        block_sizes=[32],
        kernel_block_sizes=[32],
        cp_size=1,
    )
    runner = NS(
        input_batch=batch,
        input_buffers=inputs,
        block_tables=tables,
        model_state=NS(attn_metadata={name: metadata}),
        attn_groups=[[NS(layer_names=[name], kv_cache_spec=NS(block_size=32))]],
        model=NS(model=NS(_mtp_hidden_buffer=torch.zeros(size, 32))),
    )
    captured = NS(attn_metadata={name: metadata}, slot_mappings={name: raw})
    return runner, captured


def fixture(tmp_path, *, size=24, requests=4, window=(1, 10), rank=0, transfer_fault=False):
    snapshots = bank(rank)
    diagnostic = _NAN.DSparkNaNDiagnostics(str(tmp_path), rank)
    diagnostic.phase = "warmup"
    runner, captured = runner_fixture(size, requests)
    manager = NS(
        model_runner=runner,
        hidden_states=torch.zeros(size, 8),
        aux_hidden_states=[torch.zeros(size, 8) for _ in range(3)],
    )
    detail = replay.ReplaySnapshots(diagnostic, manager, snapshots, window)
    inputs = {"input_ids": runner.input_buffers.input_ids, "positions": runner.input_buffers.positions}
    value = torch.ones(size, 8)
    fault = torch.zeros(size, 8)
    layers = [decoder(0, snapshots, torch.zeros_like(fault)), decoder(1, snapshots, fault)]
    desc = type("Descriptor", (), {"num_tokens": size, "cg_mode": NS(name="FULL")})()
    python_calls = []

    def target_forward(**kwargs):
        python_calls.append("forward")
        snapshots.write("embedding", value)
        hidden = value[:, None, :].repeat(1, 4, 1)
        residual = None
        for layer in layers:
            hidden, residual = layer(inputs["positions"], hidden, residual)
        snapshots.write("pre_hc", hidden)
        runner.model.model._mtp_hidden_buffer.copy_(hidden.flatten(1))
        hidden = hidden.mean(1)
        snapshots.write("post_hc", hidden)
        snapshots.write("post_norm", hidden)
        output = (hidden, [hidden + i for i in range(3)])
        return output

    wrapper_tree = ast.parse((ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py").read_text())
    wrapper_class = next(
        node for node in wrapper_tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelWithContext"
    )
    ns = {"torch": torch, "nn": torch.nn, "_EXTRA_CTX": NS(capturing=False)}
    exec(compile(ast.Module(body=[wrapper_class], type_ignores=[]), "actual_model_with_context", "exec"), ns)
    wrapper = ns["ModelWithContext"](target_forward, replay_diagnostics=detail)

    def forward():
        output = wrapper(**inputs)
        hidden = output[0]
        manager.hidden_states.copy_(hidden)
        for destination, source in zip(manager.aux_hidden_states, output[1]):
            destination.copy_(source)
        if transfer_fault:
            manager.hidden_states[0].fill_(torch.nan)

    forward()  # Real warmup allocates output banks outside capture.
    graph = CPURecordedGraph()
    with patch.object(torch.npu, "is_current_stream_capturing", return_value=True), graph:
        forward()
    manager.graphs = {desc: graph}
    detail.finish_capture({desc: NS(captured=captured)})
    actual_run = _production_method(
        ROOT.parent / "vllm-hust/vllm/v1/worker/gpu/cudagraph_utils.py",
        "CudaGraphManager",
        "run_fullgraph",
        {"CUDAGraphMode": NS(FULL=desc.cg_mode), "get_offloader": lambda: NS(sync_prev_onload=lambda: None)},
    )

    def run():
        diagnostic.begin_execution(
            NS(
                num_scheduled_tokens={request: 6 for request in runner.input_batch.req_ids},
                total_num_scheduled_tokens=requests * 6,
            )
        )
        actual_run(manager, desc)
        runner.execute_model_state = NS(
            input_batch=runner.input_batch,
            hidden_states=manager.hidden_states,
            aux_hidden_states=manager.aux_hidden_states,
        )
        diagnostic.target_completed(runner, [desc])

    return NS(
        bank=snapshots,
        diagnostic=diagnostic,
        runner=runner,
        manager=manager,
        detail=detail,
        graph=graph,
        desc=desc,
        forward_calls=python_calls,
        fault=fault,
        value=value,
        run=run,
    )


def test_actual_aten_replay_refreshes_snapshots_without_python_forward(tmp_path):
    f = fixture(tmp_path)
    assert len(f.forward_calls) == 2 and not f.diagnostic.current
    f.value.fill_(3)
    f.run()
    detail = f.diagnostic.current["replay_detail"]
    assert detail["status"] == "ACTUAL_FULL_REPLAY_SNAPSHOTS"
    assert detail["replay_epoch_receipt"] == [1] and detail["missing_replay_boundaries"] == []
    assert len(f.forward_calls) == 2 and f.graph.calls == 1
    assert torch.all(f.bank.views(24)["embedding"] == 3)
    assert detail["pre_replay"]["mismatches"] == []
    assert detail["localization"]["first_invalid_boundary"] is None
    assert all(not transfer["different_rows"] for transfer in detail["output_transfers"].values())


def test_compiled_dynamic_bank_copies_and_receipts():
    snapshots = bank(layers=[])

    def write(value):
        snapshots.write("embedding", value)
        return value * 2

    compiled = torch.compile(write, backend="eager", fullgraph=True, dynamic=True)
    # Profile and arbitrary eager sizes remain bounded; actual capture shapes
    # select disjoint offsets in the same dynamically compiled function.
    for size in (50, 1, 6, 12, 18, 24):
        snapshots.epoch_input.fill_(size)
        value = torch.full((size, 8), float(size))
        torch.testing.assert_close(compiled(value), value * 2)
        if size in snapshots.sizes:
            assert torch.all(snapshots.views(size)["embedding"] == size)
            assert snapshots.receipts["embedding"][snapshots.sizes.index(size)] == size


def test_large_profile_graph_reused_without_dynamo_guards_or_recompile():
    snapshots = bank(layers=[])
    graphs = []

    def backend(graph, examples):
        graphs.append((graph, examples))
        return graph.forward

    def write(value):
        snapshots.write("embedding", value)
        return value * 2

    profile_tokens = 8192
    torch.compile(write, backend=backend, fullgraph=True, dynamic=True)(torch.ones(profile_tokens, 8))
    assert len(graphs) == 1
    graph, examples = graphs[0]
    # Call the same FX graph directly as vLLM's no-guards wrapper does; calling
    # torch.compile's guarded wrapper here would hide a specialization bug.
    for size in (6, 12, 18, 24, 1):
        value = torch.full((size, 8), float(size))
        arguments = []
        for example in examples:
            if isinstance(example, torch.SymInt):
                arguments.append(size if int(example) == profile_tokens else int(example))
            elif isinstance(example, torch.Tensor) and tuple(example.shape) == (profile_tokens, 8):
                arguments.append(value)
            else:
                arguments.append(example)
        snapshots.epoch_input.fill_(size)
        graph(*arguments)
        if size in snapshots.sizes:
            torch.testing.assert_close(snapshots.views(size)["embedding"], value)
            assert snapshots.receipts["embedding"][snapshots.sizes.index(size)] == size
    assert len(graphs) == 1


def test_rank_shape_layer_banks_do_not_overlap_and_snapshots_survive_inplace():
    a, b = bank(0), bank(1)
    pointers = []
    for snapshots in (a, b):
        for size in snapshots.sizes:
            for value in snapshots.views(size).values():
                pointers.append((value.data_ptr(), value.data_ptr() + value.numel() * value.element_size()))
    assert all(
        end <= other_start or other_end <= start
        for i, (start, end) in enumerate(pointers)
        for other_start, other_end in pointers[i + 1 :]
    )
    value = torch.ones(6, 8)
    a.write("embedding", value)
    value.fill_(torch.nan)
    assert torch.isfinite(a.views(6)["embedding"]).all()


@pytest.mark.parametrize("requests", [1, 2, 4])
def test_inputs_request_identity_and_padding(tmp_path, requests):
    runner, captured = runner_fixture(24, requests)
    inputs = {"input_ids": runner.input_buffers.input_ids, "positions": runner.input_buffers.positions}
    result = replay.inspect_replay_inputs(runner, captured, inputs, 24)
    assert result["valid_query_range"] == [0, requests * 6]
    assert result["padding_query_range"] == [requests * 6, 24]
    assert not result["mismatches"]
    runner.block_tables.slot_mappings[0, 0] = -2
    result = replay.inspect_replay_inputs(runner, captured, inputs, 24)
    assert any("raw_slots" in mismatch["field"] for mismatch in result["mismatches"])
    runner.input_buffers.query_start_loc[1] += 1
    # CPU expected offset array is detached for the next mismatch check.
    runner.input_batch.query_start_loc_np = np.arange(requests + 1) * 6
    result = replay.inspect_replay_inputs(
        runner, captured, {**inputs, "input_ids": inputs["input_ids"].clone() + 1}, 24
    )
    assert any("query_start_loc" in item["field"] for item in result["mismatches"])
    assert any("captured_inputs.input_ids" in item["field"] for item in result["mismatches"])


@pytest.mark.parametrize("fault", ["internal", "transfer"])
def test_first_invalid_layer_vs_finite_source_transfer(tmp_path, fault):
    f = fixture(tmp_path, transfer_fault=fault == "transfer")
    if fault == "internal":
        f.fault[:6].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="target_outputs"):
        f.run()
    current = json.loads((tmp_path / "rank-0-first-failure.json").read_text())["current"]
    detail = current["replay_detail"]
    if fault == "internal":
        assert detail["localization"]["last_finite_boundary"] == "layer.1.attn_input"
        assert detail["localization"]["first_invalid_boundary"] == "layer.1.attn_output"
        assert detail["boundaries"]["layer.1.attn_output"]["nan_rows"] == list(range(6))
        assert "hidden" in detail["localization"]["raw_output_nonfinite"]
        assert not detail["localization"]["finite_source_nonfinite_destination"]
    else:
        assert detail["localization"]["first_invalid_boundary"] is None
        assert detail["localization"]["finite_source_nonfinite_destination"] == ["hidden"]
        assert detail["output_transfers"]["hidden"]["different_rows"] == [0]


def test_first_failure_previous_two_epochs_and_later_error_immutable(tmp_path):
    f = fixture(tmp_path)
    for _ in range(3):
        f.run()
    f.fault[2].fill_(torch.nan)
    with pytest.raises(RuntimeError):
        f.run()
    path = tmp_path / "rank-0-first-failure.json"
    original = path.read_bytes()
    report = json.loads(original)
    assert [e["target_execution_epoch"] for e in report["previous_executions"]] == [2, 3]
    assert report["current"]["target_execution_epoch"] == 4
    assert all(e["replay_detail"]["epoch"] == e["target_execution_epoch"] for e in report["previous_executions"])
    f.diagnostic.begin_execution(_scheduler())
    f.diagnostic.failed_execution("later ownership", RuntimeError("secondary"))
    assert path.read_bytes() == original


def test_last_detailed_window_survives_later_execution(tmp_path):
    f = fixture(tmp_path, window=(1, 3))
    for _ in range(4):
        f.run()
    assert f.diagnostic.current["replay_detail"]["status"] == "unavailable"
    window = json.loads((tmp_path / "rank-0-window.json").read_text())
    assert window["current"]["target_execution_epoch"] == 3
    assert [item["target_execution_epoch"] for item in window["previous_executions"]] == [1, 2]
    assert window["current"]["replay_detail"]["status"] == "ACTUAL_FULL_REPLAY_SNAPSHOTS"


@pytest.mark.parametrize("kind", ["outside_window", "no_op", "layer_missing", "exception"])
def test_capture_or_failed_execution_cannot_claim_replay_data(tmp_path, kind):
    f = fixture(tmp_path, window=(5, 9) if kind == "outside_window" else (1, 9))
    if kind == "no_op":
        f.graph.operations.clear()
    elif kind == "layer_missing":
        # Keep outer output receipt, omit one model-compiled receipt operation.
        original = f.graph.replay

        def no_layer_receipt():
            original()
            f.bank.receipts["embedding"].fill_(-1)

        f.graph.replay = no_layer_receipt
    elif kind == "exception":
        f.graph.replay = lambda: (_ for _ in ()).throw(RuntimeError("graph failed"))
    if kind == "exception":
        with pytest.raises(RuntimeError, match="graph failed"):
            f.run()
        assert f.diagnostic.current["replay_detail"]["status"] == "REPLAY_NOT_COMPLETED"
    else:
        f.run()
        assert f.diagnostic.current["replay_detail"]["status"] == "unavailable"
    assert f.detail.completed == 0


def test_duplicate_proxy_installation_rejected(tmp_path):
    f = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="Duplicate"):
        f.detail.finish_capture({f.desc: NS(captured=f.detail.captured[f.desc])})


def test_one_good_round_does_not_hide_later_missing_receipt(tmp_path):
    f = fixture(tmp_path)
    f.run()
    f.graph.operations.clear()
    f.run()
    configuration = f.diagnostic.replay_configuration
    assert configuration["completed_detailed_replays"] == 1
    assert configuration["attempted_detailed_replays"] == 2
    assert configuration["unavailable_detailed_replays"] == 1
    assert configuration["first_unavailable_detail"]["epoch"] == 2


def test_stale_captured_metadata_and_request_state_row_are_reported():
    runner, captured = runner_fixture()
    name = "model.layers.0.self_attn.attn"
    stale = NS(**vars(captured.attn_metadata[name].decode))
    stale.start_pos = stale.start_pos.clone() - 1
    captured.attn_metadata = {name: NS(decode=stale, prefill=None)}
    runner.req_states = NS(req_id_to_index={request: 99 for request in runner.input_batch.req_ids})
    result = replay.inspect_replay_inputs(runner, captured, {"input_ids": runner.input_buffers.input_ids}, 24)
    fields = [mismatch["field"] for mismatch in result["mismatches"]]
    assert f"{name}.decode.start_pos.captured_values" in fields
    assert "request identity vs req_states.req_id_to_index" in fields


def test_unavailable_device_input_is_not_filled_from_host_expected():
    runner, captured = runner_fixture()
    runner.input_buffers.positions = None
    result = replay.inspect_replay_inputs(runner, captured, {}, 24)
    assert result["match_status"] == "unavailable"
    assert "values" not in result["device_inputs"]["positions"]


@pytest.mark.parametrize("value,kind", [(torch.nan, "nan"), (torch.inf, "positive_inf"), (-torch.inf, "negative_inf")])
def test_row_flags_include_only_valid_rows_and_distinguish_nonfinite_kinds(value, kind):
    tensor = torch.ones(24, 8)
    tensor[3, 1] = value
    tensor[18:] = torch.nan
    stats = replay.finite_summaries({"target": tensor}, 18)["target"]
    assert stats[f"{kind}_rows"] == [3]
    assert stats["valid_row_range"] == [0, 18]


def test_r6_extension_reuses_pre_capture_diagnostic(tmp_path, monkeypatch):
    from tests.ut.test_dspark_nan_diagnostics import _EXTENSION, _TensorRunner

    diagnostic = _NAN.DSparkNaNDiagnostics(str(tmp_path), 0)
    diagnostic.begin_execution(_scheduler())  # A duplicate constructor would refuse these files.
    runner = _TensorRunner(tmp_path)
    runner.cudagraph_manager._dspark_nan_diagnostic = diagnostic
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_nan", _NAN)
    observer = _EXTENSION._FullReplayObserver(runner)
    assert observer.nan_diagnostic is diagnostic
    assert runner.speculator._nan_diagnostic is diagnostic


def test_disabled_real_decoder_has_no_snapshot_side_effects():
    fault = torch.zeros(6, 8)
    layer = decoder(0, None, fault)
    assert not layer._forward_hooks and not layer._forward_pre_hooks
    value = torch.ones(6, 4, 8)
    hidden, residual = layer(torch.arange(6), value, None)
    assert hidden.shape == residual.shape == value.shape
    assert layer._dspark_layer_snapshots is None


def test_cli_window_requires_opt_in_and_preserves_draft_eager(tmp_path, monkeypatch):
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
        str(tmp_path / "out.json"),
    ]
    assert benchmark.parse_args(args).dspark_nan_replay_window is None
    with pytest.raises(SystemExit):
        benchmark.parse_args([*args, "--dspark-nan-replay-window", "60", "80"])
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
        "--dspark-nan-replay-window",
        "60",
        "80",
    ]
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    kwargs = benchmark.build_engine_kwargs(benchmark.parse_args(args))
    assert kwargs["additional_config"]["dspark_nan_replay_window"] == [60, 80]
    assert kwargs["speculative_config"]["enforce_eager"] is True
