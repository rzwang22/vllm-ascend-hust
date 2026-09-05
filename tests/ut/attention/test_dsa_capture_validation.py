# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU execution of capture-external validation and the real replay boundary."""

import ast
from contextlib import nullcontext
from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.ut.attention import test_dsa_padding_contract as padding
from tests.ut.attention.test_dsa_padding_contract import (
    ROOT,
    _cache,
    _contract,
    _metadata_run,
    torch,
)
from tests.ut.worker.test_dsa_capture_metadata import _load_functions

api = padding.api
scatter_selector = padding.scatter_selector


def _setup(api, monkeypatch, batch=1):
    run = _metadata_run(api, monkeypatch, batch)
    prepare = run.prepare
    run.prepare = lambda capture: prepare(capture, api.mode.FULL)
    spec_cls = api.dsa_globals.get("AscendSlidingWindowMLASpec")
    if spec_cls is None:
        spec_cls = type("AscendSlidingWindowMLASpec", (SimpleNamespace,), {})
        monkeypatch.setitem(api.dsa_globals, "AscendSlidingWindowMLASpec", spec_cls)
    run.group.kv_cache_spec = spec_cls(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        model_version="deepseek_v4",
    )
    cache = _cache(11)
    run.group.backend = object()
    context = {"swa": SimpleNamespace(kv_cache=[cache])}
    run.state.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        speculative_config=SimpleNamespace(method="dspark"),
        compilation_config=SimpleNamespace(static_forward_context=context),
    )
    state_globals = api.state_cls.prepare_attn.__globals__
    monkeypatch.setitem(
        state_globals, "preflight_dsa_sharedkv_indices", api.dsa_globals["preflight_dsa_sharedkv_indices"]
    )
    monkeypatch.setitem(state_globals, "_should_validate_dspark_sharedkv_contract", lambda config: True)
    run.context = context
    run.cache = cache
    run.capturing = False
    run.transfers = []
    original_to = torch.Tensor.to
    original_assert = api.dsa_globals["_assert_dsa_tensor_range"]

    def tensor_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and tensor.dtype == torch.bool:
            assert not run.capturing, "status transfer entered the captured stream"
            run.transfers.append(tensor.numel())
        return original_to(tensor, *args, **kwargs)

    def assert_range(predicate, message):
        assert not run.capturing, "legacy scalar assertion entered graph capture"
        return original_assert(predicate, message)

    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    monkeypatch.setitem(api.dsa_globals, "_assert_dsa_tensor_range", assert_range)
    return run


def _validate(api, run, leaf):
    inputs = _contract(leaf.slot_mapping, run.cache, leaf.seq_lens.shape[0])
    inputs.update(
        block_table=leaf.block_table,
        query_start_loc=leaf.query_start_loc,
        seqused_kv=leaf.seq_lens,
        sas_metadata=leaf.sas_metadata,
        index_validation=leaf.sharedkv_index_validation,
    )
    api.validate(**inputs)


def _graph_manager(api, monkeypatch, run, replay):
    """Use the plugin's actual run_fullgraph body; replace only hardware replay."""
    if api.builder.__module__.startswith("vllm_ascend"):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

        from vllm_ascend.worker.v2 import aclgraph_utils

        cls = aclgraph_utils.ModelAclGraphManager
        namespace = vars(aclgraph_utils)
        monkeypatch.setattr(ModelCudaGraphManager, "run_fullgraph", lambda self, desc: replay())
    else:

        class Parent:
            def run_fullgraph(self, desc):
                return replay()

        source = ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py"
        cls_node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "ModelAclGraphManager"
        )
        cls_node.body = [n for n in cls_node.body if isinstance(n, ast.FunctionDef) and n.name == "run_fullgraph"]
        namespace = {"ModelCudaGraphManager": Parent}
        module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, cls_node], type_ignores=[])
        exec(compile(module, str(source), "exec"), namespace)
        cls = namespace["ModelAclGraphManager"]
    for name, value in {
        "torch": torch,
        "logger": SimpleNamespace(info_once=lambda *args: None),
        "set_forward_context": lambda *args, **kwargs: nullcontext(),
        "get_forward_context": lambda: object(),
        "update_full_graph_params": lambda *args: None,  # DSA's update_graph_params is a no-op.
    }.items():
        monkeypatch.setitem(namespace, name, value)
    manager = object.__new__(cls)
    manager.model_runner = SimpleNamespace(
        model_state=run.state,
        input_buffers=SimpleNamespace(positions=torch.zeros(run.builder.slot_mapping.shape[0])),
        dp_size=1,
        update_stream=object(),
        attn_groups=[[run.group]],
        speculative_config=run.state.vllm_config.speculative_config,
    )
    manager.vllm_config = run.state.vllm_config
    manager.device = torch.device("cpu")
    return manager


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_capture_and_every_replay_update_are_validated(api, monkeypatch, batch):
    run = _setup(api, monkeypatch, batch)
    warmup = run.prepare(True)  # Frozen core passes NONE + for_capture=True.
    _validate(api, run, warmup)
    captured = run.prepare(True)
    assert warmup is not captured
    assert warmup.sharedkv_index_validation is not captured.sharedkv_index_validation
    pointer = captured.slot_mapping.data_ptr()
    assert pointer == warmup.slot_mapping.data_ptr() == run.builder.slot_mapping.data_ptr()
    run.capturing = True
    _validate(api, run, captured)  # All static checks still run; no scalar assertion/copy.
    run.capturing = False
    assert run.transfers == [8, 8]

    updates = torch.full((batch * 6, 1, 512), 9, dtype=run.cache.dtype)
    replays = []

    def replay():
        # Real C++-derived scatter selector and real CPU KV writes, through the
        # captured buffer address. Replaying does not execute Python forward.
        replays.append(captured.slot_mapping.data_ptr())
        api.device.dsa_kv_compress_scatter(run.cache, updates, captured.slot_mapping)
        return updates

    manager = _graph_manager(api, monkeypatch, run, replay)
    desc = SimpleNamespace(num_tokens=batch * 6, cg_mode=api.mode.FULL)
    with pytest.raises(RuntimeError, match="newly prepared"):
        manager.run_fullgraph(desc)  # Capture's dummy receipt cannot authorize real replay.
    run.tables.slot_mappings[0, : batch * 6 : 2] = torch.arange(0, batch * 6, 2, dtype=torch.int32) + 32
    live = run.prepare(False)
    assert live is not captured and live.sharedkv_index_validation is not captured.sharedkv_index_validation
    assert live.slot_mapping.data_ptr() == pointer
    manager.run_fullgraph(desc)
    expected = torch.full_like(run.cache, 7)
    expected[1, : batch * 6 : 2] = 9
    assert torch.equal(run.cache, expected)
    assert replays == [pointer] and run.transfers == [8, 8, 8]
    with pytest.raises(RuntimeError, match="newly prepared"):
        manager.run_fullgraph(desc)  # Requires another preparation, not a buffer-valid cache.

    run.tables.slot_mappings[0, 0] = -2
    with pytest.raises(ValueError, match="physical block outside ori_kv"):
        run.prepare(False)
    with pytest.raises(RuntimeError, match="newly prepared"):
        manager.run_fullgraph(desc)
    assert replays == [pointer] and torch.equal(run.cache, expected)
    assert run.transfers == [8, 8, 8, 8]
    run.prepare(True)  # Dummy preparation resets the entire source to actual -1.
    assert torch.all(captured.slot_mapping[:, 0] == -1) and torch.all(captured.slot_mapping[:, 1] == 31)
    assert captured.slot_mapping.data_ptr() == pointer


@pytest.mark.parametrize(
    ("field", "index", "value", "message"),
    [
        ("query_start_loc", 0, 1, "must start at zero"),
        ("query_start_loc", 1, 0, "strictly increasing"),
        ("query_start_loc", 1, 7, "terminal value"),
        ("seq_lens", 0, 5, "include every current query token"),
        ("seq_lens", 0, 129, "physical cache capacity"),
        ("block_table", (0, 0), 4, "ori_block_table"),
        ("slot_mapping", (0, 0), 4, "physical block outside ori_kv"),
        ("slot_mapping", 0, [0, 32], "offset outside the cache block"),
        ("slot_mapping", 0, [0, -1], "offset outside the cache block"),
        ("slot_mapping", (0, 1), 30, "physical block outside ori_kv"),
    ],
)
def test_preflight_rejects_all_dynamic_failures_from_current_device_values(
    api, monkeypatch, field, index, value, message
):
    run = _setup(api, monkeypatch)
    leaf = run.prepare(True)
    getattr(leaf, field)[index] = torch.tensor(value)  # CPU mirrors remain valid; check actual tensor values.
    with pytest.raises(ValueError, match=message):
        api.dsa_globals["preflight_dsa_sharedkv_indices"](run.state.attn_metadata, [[run.group]], run.context)
    assert leaf.sharedkv_index_validation is None and not leaf.sharedkv_contract_validated
    assert run.transfers == [8, 8]
    assert torch.all(run.cache == 7)


def test_receipt_preserves_static_checks_and_rejects_other_buffers(api, monkeypatch):
    run = _setup(api, monkeypatch)
    leaf = run.prepare(True)
    original = leaf.slot_mapping
    leaf.slot_mapping = original.clone()
    with pytest.raises(ValueError, match="current input buffers"):
        _validate(api, run, leaf)
    leaf.slot_mapping = original
    run.cache = torch.empty((8, 32, 1, 512), dtype=torch.bfloat16)[::2]
    with pytest.raises(ValueError, match="Interleaved packed KV pages"):
        _validate(api, run, leaf)
    assert run.transfers == [8]


def test_shared_layers_and_groups_use_one_status_transfer(api, monkeypatch):
    run = _setup(api, monkeypatch)
    leaf = run.prepare(True)
    run.group.layer_names = [f"swa{i}" for i in range(48)]
    metadata = run.state.attn_metadata["swa"]
    run.state.attn_metadata = {name: metadata for name in run.group.layer_names}
    run.context = {name: SimpleNamespace(kv_cache=[run.cache]) for name in run.group.layer_names}
    preflight = api.dsa_globals["preflight_dsa_sharedkv_indices"]
    preflight(run.state.attn_metadata, [[run.group], [run.group]], run.context)
    assert leaf.sharedkv_index_validation is not None and run.transfers == [8, 8]
    non_dsa = SimpleNamespace(get_metadata_builder=lambda index: object(), kv_cache_spec=object())
    preflight({}, [[non_dsa]], {})
    assert run.transfers == [8, 8]


def test_eager_path_still_checks_actual_indices(api, monkeypatch):
    run = _setup(api, monkeypatch)
    run.state.vllm_config.model_config.enforce_eager = True
    leaf = run.prepare(True)
    assert leaf.sharedkv_index_validation is None and run.transfers == []
    _validate(api, run, leaf)
    leaf.slot_mapping[0] = torch.tensor([-1, 30])
    with pytest.raises(ValueError, match="physical block outside ori_kv"):
        _validate(api, run, leaf)


@pytest.mark.parametrize("mismatch", ["metadata", "tokens"])
def test_replay_permission_is_bound_to_prepared_metadata_and_shape(api, monkeypatch, mismatch):
    run = _setup(api, monkeypatch)
    run.prepare(False)
    replays = []
    manager = _graph_manager(api, monkeypatch, run, lambda: replays.append(True))
    desc = SimpleNamespace(num_tokens=6, cg_mode=api.mode.FULL)
    if mismatch == "metadata":
        run.state.attn_metadata = dict(run.state.attn_metadata)
    else:
        desc.num_tokens = 12
    with pytest.raises(RuntimeError, match="newly prepared"):
        manager.run_fullgraph(desc)
    assert not replays and run.state._sharedkv_replay_input is None


def test_only_ori_kv_groups_are_preflighted(api, monkeypatch):
    run = _setup(api, monkeypatch)
    run.prepare(True)
    spec = run.group.kv_cache_spec
    # Actual compressor state specs have model_version=None and ratio=1 too.
    if isinstance(spec, SimpleNamespace):
        state_spec = copy(spec)
        state_spec.model_version = None
    else:
        state_spec = replace(spec, model_version=None)
    state_group = SimpleNamespace(
        get_metadata_builder=lambda index: run.builder,
        kv_cache_spec=state_spec,
        layer_names=["compressor.state_cache"],
    )
    compressed_builder = copy(run.builder)
    compressed_builder.compressor_ratio = 4
    compressed_group = SimpleNamespace(
        get_metadata_builder=lambda index: compressed_builder,
        kv_cache_spec=spec,
        layer_names=["compressor"],
    )
    api.dsa_globals["preflight_dsa_sharedkv_indices"]({}, [[state_group, compressed_group]], {})
    assert run.transfers == [8]


def test_prefill_decode_statuses_are_combined_and_published_atomically(api, monkeypatch):
    run = _setup(api, monkeypatch)
    decode = run.prepare(True)
    prefill = copy(decode)
    prefill.slot_mapping = decode.slot_mapping.clone()
    prefill.query_start_loc = torch.tensor([0, 7], dtype=torch.int32)
    prefill.seq_lens = torch.tensor([7], dtype=torch.int32)
    prefill.slot_mapping = torch.cat((prefill.slot_mapping, prefill.slot_mapping[:1]), dim=0)
    metadata = run.state.attn_metadata["swa"]
    metadata.prefill = prefill
    metadata.num_actual_tokens = 13  # 6 decode + 7 prefill, not the decode token count.
    preflight = api.dsa_globals["preflight_dsa_sharedkv_indices"]
    preflight(run.state.attn_metadata, [[run.group]], run.context)
    assert run.transfers == [8, 16]
    assert prefill.sharedkv_index_validation is not None and decode.sharedkv_index_validation is not None
    # Break the last leaf: no successful receipt is published for either leaf.
    decode.block_table[0, 0] = 4
    with pytest.raises(ValueError, match="ori_block_table"):
        preflight(run.state.attn_metadata, [[run.group]], run.context)
    assert prefill.sharedkv_index_validation is None and decode.sharedkv_index_validation is None
    assert run.transfers == [8, 16, 16]


@pytest.mark.parametrize("stage", ["prefill", "decode"])
@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_real_forward_entry_uses_receipt_with_common_group_indices(api, monkeypatch, stage, ratio):
    run = _setup(api, monkeypatch)
    leaf = run.prepare(True)
    namespace = api.dsa_globals
    if api.builder.__module__.startswith("vllm_ascend"):
        from vllm_ascend.attention.dsa_v1 import AscendDSAImpl

        forward = getattr(AscendDSAImpl, f"_forward_{stage}")
    else:
        _load_functions(ROOT / "vllm_ascend/attention/dsa_v1.py", namespace, [f"_require_{stage}_metadata"])
        _load_functions(ROOT / "vllm_ascend/attention/dsa_v1.py", namespace, [f"_forward_{stage}"], "AscendDSAImpl")
        _load_functions(
            ROOT / "vllm_ascend/device/device_op.py",
            namespace,
            ["unpack_dsa_forward_kv_cache"],
            "BaseDeviceAdaptor",
        )
        monkeypatch.setattr(
            namespace["DeviceOperator"],
            "unpack_dsa_forward_kv_cache",
            staticmethod(namespace["unpack_dsa_forward_kv_cache"]),
            raising=False,
        )
        forward = namespace[f"_forward_{stage}"]
    leaf.cos = leaf.sin = {"c1": torch.zeros(6)}
    common = copy(leaf)  # Common query/sequence indices share storage across KV groups (R3).
    swa = SimpleNamespace(**{stage: leaf})
    compressed = SimpleNamespace(**{stage: common})
    metadata = [swa] if ratio == 1 else [compressed, object(), swa]
    if ratio == 4:
        metadata = [compressed, object(), object(), object(), swa]

    class ReachedProlog(Exception):
        pass

    def prolog(hidden_states, cos, sin, cache, slots, *, is_prefill):
        assert cache is run.cache and slots is leaf.slot_mapping
        assert is_prefill == (stage == "prefill")
        assert leaf.sharedkv_contract_validated
        # Stop at the first NPU computation after executing the real validator.
        raise ReachedProlog

    impl = SimpleNamespace(
        compress_ratio=ratio,
        validate_dspark_sharedkv_contract=True,
        attn_sink=torch.zeros(2),
        multistream_dsv4_dsa_overlap=True,
        _mla_prolog_multistream=prolog,
    )
    run.capturing = True
    with pytest.raises(ReachedProlog):
        forward(impl, "c1", torch.zeros((6, 512)), (None, run.cache, None, None, None, None), metadata)
    assert run.transfers == [8]
    leaf.sharedkv_contract_validated = False
    common.query_start_loc = common.query_start_loc.clone()
    if ratio == 1:
        leaf.query_start_loc = common.query_start_loc
    with pytest.raises(ValueError, match="current input buffers"):
        forward(impl, "c1", torch.zeros((6, 512)), (None, run.cache, None, None, None, None), metadata)


def test_unprepared_npu_assertion_fails_before_dispatch_in_capture(monkeypatch):
    namespace = {}
    _load_functions(ROOT / "vllm_ascend/attention/dsa_v1.py", namespace, ["_assert_dsa_tensor_range"])
    calls = []
    fake_torch = SimpleNamespace(
        npu=SimpleNamespace(is_current_stream_capturing=lambda: True),
        _assert_async=lambda *args: calls.append(args),
    )
    monkeypatch.setitem(namespace, "torch", fake_torch)
    predicate = SimpleNamespace(device=SimpleNamespace(type="npu"))
    with pytest.raises(RuntimeError, match="capture-external preflight"):
        namespace["_assert_dsa_tensor_range"](predicate, "bad range")
    assert calls == []  # Merely naming the assertion async cannot authorize it.
