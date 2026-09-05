# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Real CPU Torch validator/metadata plus a source-derived scatter reference.

Run with --noconftest on Mac. Hardware-free source execution is paired with
normal imported-class CPU tests on the server. Neither is NPU validation.
"""

import ast
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from tests.ut.attention.dsa_scatter_host import build_scatter_selector
from tests.ut.worker.test_dsa_capture_metadata import ROOT, _core_graph_source, _load_functions

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def scatter_selector(tmp_path_factory):
    return build_scatter_selector(ROOT, tmp_path_factory.mktemp("dsa-scatter-host"))


@pytest.fixture(params=["source", "runtime"])
def api(request, monkeypatch, scatter_selector):
    core = _core_graph_source().parents[2]
    tree = ast.parse((core / "attention/backends/utils.py").read_text())
    pad_id = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PAD_SLOT_ID" for t in n.targets)
    )
    assert pad_id == -1
    if request.param == "source":
        from tests.ut.worker.test_dsa_capture_metadata import _Mode

        namespace = {
            "torch": torch,
            "np": np,
            "PAD_SLOT_ID": pad_id,
            "CUDAGraphMode": _Mode,
            "BUILD_METADATA_STEP_PREFILL": 0,
            "BUILD_METADATA_STEP_DECODE": 1,
            "is_pd_decode_recompute_scheduler_enabled": lambda: False,
            "AscendCommonAttentionMetadata": lambda **values: SimpleNamespace(_seq_lens_cpu=None, **values),
            "AscendDSADecodeMetadata": lambda **values: SimpleNamespace(sharedkv_contract_validated=False, **values),
        }
        _load_functions(ROOT / "vllm_ascend/attention/utils.py", namespace, ["split_decodes_and_prefills"])
        names = [
            "build",
            "build_for_cudagraph_capture",
            "build_decode_metadata",
            "set_num_actual_tokens",
            "get_block_table_size",
        ]
        _load_functions(ROOT / "vllm_ascend/attention/dsa_v1.py", namespace, names, "AscendDSAMetadataBuilder")
        builder = type("AscendDSAMetadataBuilder", (), {name: namespace[name] for name in names})
        builder.hadamard = None
        namespace["AscendDSAMetadataBuilder"] = builder
        _load_functions(
            ROOT / "vllm_ascend/attention/dsa_v1.py",
            namespace,
            ["_assert_dsa_tensor_range", "validate_dsa_sharedkv_page_contract"],
        )
        methods = [
            "format_dsa_slot_mapping",
            "dsa_kv_compress_scatter",
            "pad_dsa_decode_slot_mapping",
            "get_dsa_decode_cu_seqlens_ori_kv",
            "get_dsa_decode_cu_seqlens_cmp_kv",
        ]
        _load_functions(ROOT / "vllm_ascend/device/device_op.py", namespace, methods, "BaseDeviceAdaptor")
        device = type("BaseDeviceAdaptor", (), {name: staticmethod(namespace[name]) for name in methods})
        _load_functions(core / "worker/gpu/block_table.py", namespace, ["get_dummy_slot_mappings"], "BlockTables")
        block_cls = type("BlockTables", (), {"get_dummy_slot_mappings": namespace["get_dummy_slot_mappings"]})
        _load_functions(ROOT / "vllm_ascend/worker/v2/attn_utils.py", namespace, ["build_attn_metadata"])
        _load_functions(
            ROOT / "vllm_ascend/worker/v2/model_states/default.py", namespace, ["prepare_attn"], "AscendModelState"
        )
        state_cls = type("AscendModelState", (), {"prepare_attn": namespace["prepare_attn"]})
        _load_functions(_core_graph_source(), namespace, ["prepare_inputs_to_capture"])
        result = SimpleNamespace(
            builder=builder,
            block_cls=block_cls,
            state_cls=state_cls,
            device=device,
            validate=namespace["validate_dsa_sharedkv_page_contract"],
            prepare=namespace["prepare_inputs_to_capture"],
            mode=_Mode,
            dsa_globals=namespace,
            graph_globals=namespace,
        )
    else:
        if importlib.util.find_spec("vllm") is None:
            pytest.skip("Imported-class CPU tests require installed vllm/Ascend; CPU Torch source tests still run")
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu import cudagraph_utils

        from vllm_ascend.attention import dsa_v1
        from vllm_ascend.device.device_op import BaseDeviceAdaptor
        from vllm_ascend.worker.v2.block_table import AscendBlockTables
        from vllm_ascend.worker.v2.model_states.default import AscendModelState

        result = SimpleNamespace(
            builder=dsa_v1.AscendDSAMetadataBuilder,
            block_cls=AscendBlockTables,
            state_cls=AscendModelState,
            device=BaseDeviceAdaptor,
            validate=dsa_v1.validate_dsa_sharedkv_page_contract,
            prepare=cudagraph_utils.prepare_inputs_to_capture,
            mode=CUDAGraphMode,
            dsa_globals=vars(dsa_v1),
            graph_globals=vars(cudagraph_utils),
        )

    def sas_metadata(**values):
        assert values["layout_kv"] == "PA_ND" and values["max_seqlen_q"] == 6
        assert values["cmp_ratio"] == 1 and values["batch_size"] in (1, 2, 4)
        return torch.zeros(1024, dtype=torch.int32)

    def qli_metadata(**values):
        assert values["layout_key"] == "PA_BSND" and values["max_seqlen_q"] == 6
        return torch.zeros(1024, dtype=torch.int32)

    class Device(result.device):
        get_dsa_sparse_attn_metadata_op = staticmethod(lambda: sas_metadata)
        get_dsa_sparse_attn_metadata_kwargs = staticmethod(lambda device: {})

    monkeypatch.setitem(result.dsa_globals, "DeviceOperator", Device)
    monkeypatch.setitem(result.dsa_globals, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setitem(
        result.dsa_globals, "get_cos_and_sin_dsa", lambda positions, use_cache=False: (positions, positions)
    )
    monkeypatch.setattr(torch.ops._C_ascend, "npu_vllm_quant_lightning_indexer_metadata", qli_metadata, raising=False)
    result.scatter_calls = []

    def scatter(cache, slots, updates):
        # Execute original C++ address computation and the actual key 10/11
        # bounds logic. CPU copies model the writes; this is not a no-op mock.
        assert slots.dtype == torch.int32 and slots.shape == (len(updates), 2)
        key, rows, addresses = scatter_selector(slots.numpy(), cache.stride(0), cache.stride(1), cache.numel())
        flat = cache.view(-1)
        for row, address in zip(rows, addresses):
            flat[int(address) : int(address) + updates[row].numel()].copy_(updates[row].reshape(-1))
        result.scatter_calls.append((key, rows.copy()))

    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_v2", scatter, raising=False)
    monkeypatch.setitem(result.graph_globals, "build_slot_mappings_by_layer", lambda slots, config: slots)
    monkeypatch.setitem(result.graph_globals, "AttentionState", lambda metadata, slots: (metadata, slots))
    return result


def _cache(tiling_key):
    # PA_ND int32 uses key 11 below 2**24 elements, otherwise key 10.
    num_blocks = 4 if tiling_key == 11 else 1025
    return torch.full((num_blocks, 32, 1, 512), 7, dtype=torch.bfloat16)


def _contract(slots, cache, batch=1):
    return dict(
        layer_name="model.layers.0.self_attn.attn",
        kv_cache=cache,
        block_table=torch.zeros((batch, 1), dtype=torch.int32),
        slot_mapping=slots,
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32) * 6,
        seqused_kv=torch.full((batch,), 6, dtype=torch.int32),
        sas_metadata=torch.zeros(1024, dtype=torch.int32),
        sinks=torch.zeros(2),
        block_size=32,
        num_query_tokens=batch * 6,
        num_reqs_actual=batch,
    )


def _tables(api, batch):
    tables = object.__new__(api.block_cls)
    # The Ascend subclass allocates int32; the inherited core method fills the
    # whole persistent buffer, including entries beyond the requested slice.
    tables.slot_mappings = torch.zeros((1, batch * 6 + 8), dtype=torch.int32)
    return tables


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("tiling_key", [10, 11])
def test_real_dummy_and_mixed_slots_only_write_valid_kv(api, batch, tiling_key):
    cache = _cache(tiling_key)
    tables = _tables(api, batch)
    address = tables.slot_mappings.data_ptr()
    flat = tables.get_dummy_slot_mappings(batch * 6)[0]
    assert flat.data_ptr() == address and torch.all(tables.slot_mappings == -1)
    slots = api.device.format_dsa_slot_mapping(flat, 32)
    assert torch.all(slots[:, 0] == -1) and torch.all(slots[:, 1] == 31)
    updates = torch.full((batch * 6, 1, 512), 9, dtype=cache.dtype)
    api.validate(**_contract(slots, cache, batch))
    api.device.dsa_kv_compress_scatter(cache, updates, slots)
    assert torch.all(cache == 7)  # All dummy tokens wrote zero cache elements.
    assert api.scatter_calls[-1][0] == tiling_key
    assert len(api.scatter_calls[-1][1]) == 0

    flat[::2] = torch.arange(0, batch * 6, 2, dtype=torch.int32) + 32
    slots.copy_(api.device.format_dsa_slot_mapping(flat, 32))
    api.validate(**_contract(slots, cache, batch))
    api.device.dsa_kv_compress_scatter(cache, updates, slots)
    expected = torch.full_like(cache, 7)
    expected[1, : batch * 6 : 2] = 9
    assert torch.equal(cache, expected)
    assert len(api.scatter_calls[-1][1]) == batch * 3
    assert tables.slot_mappings.data_ptr() == address


@pytest.mark.parametrize("invalid", [[-1, 30], [-1, -1], [-1, 32], [-1, 64], [-2, 31], [4, 0], [0, -1], [0, 32]])
def test_invalid_block_or_offset_is_rejected_before_scatter(api, invalid):
    cache = _cache(11)
    slots = api.device.format_dsa_slot_mapping(torch.full((6,), -1, dtype=torch.int32), 32)
    slots[0] = torch.tensor(invalid, dtype=torch.int32)
    with pytest.raises(ValueError, match="slot_mapping contains"):
        api.validate(**_contract(slots, cache))
        api.device.dsa_kv_compress_scatter(cache, torch.zeros((6, 1, 512), dtype=cache.dtype), slots)
    assert api.scatter_calls == [] and torch.all(cache == 7)


def test_padding_does_not_relax_block_table_or_page_layout_validation(api):
    cache = _cache(11)
    slots = api.device.format_dsa_slot_mapping(torch.full((6,), -1, dtype=torch.int32), 32)
    inputs = _contract(slots, cache)
    inputs["block_table"][0, 0] = -1
    with pytest.raises(ValueError, match="ori_block_table"):
        api.validate(**inputs)
    inputs["block_table"].zero_()
    backing = torch.zeros((8, 32, 1, 512), dtype=cache.dtype)
    inputs["kv_cache"] = backing[::2]
    with pytest.raises(ValueError, match="Interleaved packed KV pages"):
        api.validate(**inputs)


def _metadata_run(api, monkeypatch, batch):
    builder = object.__new__(api.builder)
    builder.decode_threshold = 6
    builder.compressor_ratio = 1
    builder.model_config = SimpleNamespace(
        get_head_size=lambda: 512,
        hf_config=SimpleNamespace(
            num_attention_heads=16, index_topk=512, sliding_window=128, index_n_heads=64, index_head_dim=128
        ),
    )
    builder.metadata_cls = SimpleNamespace
    builder.slot_mapping = torch.zeros((batch * 6, 2), dtype=torch.int32)
    builder.start_pos_decode = torch.zeros(batch, dtype=torch.int32)
    builder.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32)
    builder.seqused_q = torch.empty(0, dtype=torch.int32)
    builder.cu_seqlens_ori_kv = torch.empty(0, dtype=torch.int32)
    builder.cu_seqlens_cmp_kv = torch.empty(0, dtype=torch.int32)
    builder._zero_i32 = torch.zeros(1, dtype=torch.int32)
    tables = _tables(api, batch)
    tables.cp_size = 1
    block_table = torch.zeros((batch, 1), dtype=torch.int32)
    tables.get_dummy_block_tables = lambda count: (block_table,)
    state = object.__new__(api.state_cls)
    state.max_model_len = 8192
    group = SimpleNamespace(
        layer_names=["swa"], kv_cache_spec=SimpleNamespace(block_size=32), get_metadata_builder=lambda index: builder
    )
    config = SimpleNamespace(kv_cache_groups=[object()])
    batch_input = SimpleNamespace(
        num_reqs=batch,
        num_reqs_after_padding=batch,
        num_tokens=batch * 6,
        num_tokens_after_padding=batch * 6,
        query_start_loc_np=np.arange(batch + 1, dtype=np.int32) * 6,
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32) * 6,
        num_scheduled_tokens=np.full(batch, 6, dtype=np.int32),
        seq_lens=torch.full((batch,), 6, dtype=torch.int32),
        seq_lens_np=np.full(batch, 6, dtype=np.int32),
        positions=torch.zeros(batch * 6, dtype=torch.int64),
        is_prefilling_np=np.zeros(batch, dtype=np.bool_),
        attn_state=object(),
        dcp_local_seq_lens=None,
    )

    def make_dummy(num_reqs, num_tokens, buffers):
        assert (num_reqs, num_tokens) == (batch, batch * 6) and buffers is batch_input
        return buffers

    monkeypatch.setitem(api.graph_globals, "InputBatch", SimpleNamespace(make_dummy=make_dummy))

    def prepare(capture):
        if capture:
            metadata, _ = api.prepare(batch, batch * 6, state, batch_input, tables, [[group]], config)
        else:
            metadata = state.prepare_attn(
                batch_input, api.mode.NONE, (block_table,), tables.slot_mappings[:, : batch * 6], [[group]], config
            )
        return metadata["swa"].decode

    return SimpleNamespace(builder=builder, tables=tables, prepare=prepare)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_warmup_capture_and_replay_updates_keep_persistent_addresses(api, monkeypatch, batch):
    run = _metadata_run(api, monkeypatch, batch)
    cache = _cache(11)
    warmup = run.prepare(True)
    captured = run.prepare(True)
    assert warmup is not captured
    assert warmup.slot_mapping.data_ptr() == captured.slot_mapping.data_ptr() == run.builder.slot_mapping.data_ptr()
    assert not warmup.sharedkv_contract_validated and not captured.sharedkv_contract_validated
    updates = torch.full((batch * 6, 1, 512), 9, dtype=cache.dtype)

    def replay_reference(metadata):
        inputs = _contract(metadata.slot_mapping, cache, batch)
        inputs.update(
            block_table=metadata.block_table,
            query_start_loc=metadata.query_start_loc,
            seqused_kv=metadata.seq_lens,
            sas_metadata=metadata.sas_metadata,
        )
        api.validate(**inputs)
        api.device.dsa_kv_compress_scatter(cache, updates, metadata.slot_mapping)

    replay_reference(warmup)
    replay_reference(captured)
    assert torch.all(cache == 7)
    pointer = captured.slot_mapping.data_ptr()
    run.tables.slot_mappings[0, : batch * 6 : 2] = torch.arange(0, batch * 6, 2, dtype=torch.int32) + 32
    current = run.prepare(False)  # Real build rewrites the captured buffer in place.
    assert current is not captured and current.slot_mapping.data_ptr() == pointer
    replay_reference(captured)  # Reads current values through the saved address.
    expected = torch.full_like(cache, 7)
    expected[1, : batch * 6 : 2] = 9
    assert torch.equal(cache, expected)
    run.prepare(True)  # Re-enter dummy preparation: stale valid slots must disappear.
    replay_reference(captured)
    assert torch.equal(cache, expected) and captured.slot_mapping.data_ptr() == pointer
    run.tables.slot_mappings[0, 0] = -2
    run.prepare(False)
    with pytest.raises(ValueError, match="slot_mapping contains"):
        replay_reference(captured)
