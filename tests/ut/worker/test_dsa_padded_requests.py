# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU Torch execution of runner preparation and DSA build/decode bodies.

NPU leaves are replaced with checked CPU implementations. Captured references
are retained across updates; this does not emulate ACLGraph kernel execution.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.ut.attention import test_dsa_padding_contract as padding_contract
from tests.ut.worker.test_dsa_capture_metadata import ROOT, _load_functions

torch = pytest.importorskip("torch")


api = padding_contract.api
scatter_selector = padding_contract.scatter_selector
_metadata_run = padding_contract._metadata_run


def _runner(api):
    def copy(value, out=None, device=None):
        value = torch.as_tensor(value)
        if out is not None:
            out.copy_(value)
            return out
        return value.clone()

    def prepare_positions(indices, query, computed, positions, seq):
        # Core prepare_pos_seq_lens includes an extra program clearing every
        # unused sequence row. Model that real kernel contract, not stale zeros.
        seq[indices.numel() :].zero_()
        for row, index in enumerate(indices.tolist()):
            start, end = query[row : row + 2].tolist()
            seq[row] = computed[index] + end - start
            positions[start:end] = torch.arange(int(computed[index]), int(seq[row]))

    namespace = dict(
        torch=torch,
        np=np,
        CUDAGraphMode=api.mode,
        envs=NS(VLLM_MOE_SKIP_PADDING=True),
        AscendInputBatch=NS,
        async_copy_to_gpu=copy,
        prepare_pos_seq_lens=prepare_positions,
        build_attn_state=lambda *args: None,
        update_cos_sin=lambda positions: None,
        combine_sampled_and_draft_tokens=lambda *args: torch.arange(args[-1]),
        expand_idx_mapping=lambda indices, count, offsets, maximum: (
            indices.repeat_interleave(6),
            torch.arange(6).repeat(len(indices)),
        ),
    )
    names = ["prepare_inputs", "_pad_query_start_loc_for_fia", "_update_seq_lens_cpu"]
    _load_functions(ROOT / "vllm_ascend/worker/v2/model_runner.py", namespace, names, "NPUModelRunner")
    cls = type("Runner", (), {name: namespace[name] for name in names})
    runner = cls()
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 64
    runner.decode_query_len = 6
    runner.num_speculative_steps = 5
    runner.model_config = NS(rswa_window=None)
    runner.vllm_config = NS()
    runner.num_computed_tokens_event = NS(synchronize=lambda: None)
    cpu_seq = torch.zeros(64, dtype=torch.int32)
    runner.input_buffers = NS(
        input_ids=torch.zeros(384, dtype=torch.int32),
        positions=torch.zeros(384, dtype=torch.int64),
        is_padding=torch.zeros(384, dtype=torch.bool),
        query_start_loc=torch.zeros(66, dtype=torch.int32),
        seq_lens=torch.zeros(64, dtype=torch.int32),
        seq_lens_cpu=cpu_seq,
        seq_lens_np=cpu_seq.numpy(),
    )
    computed = torch.arange(64, dtype=torch.int32) + 70
    runner.req_states = NS(
        req_id_to_index={str(i): 63 - i for i in range(64)},
        num_computed_tokens_cpu=computed,
        num_computed_tokens_np=computed.numpy(),
        num_computed_tokens=NS(gpu=computed),
        prefill_len=NS(np=np.ones(64, dtype=np.int32), gpu=torch.ones(64, dtype=torch.int32)),
        num_computed_prefill_tokens=np.ones(64, dtype=np.int32),
        last_sampled_tokens=None,
        draft_tokens=None,
    )
    return runner


def _batch(runner, actual, padded, mode):
    scheduled = {str(i): 6 for i in range(actual)}
    scheduler = NS(
        total_num_scheduled_tokens=actual * 6,
        num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens={key: [1] * 5 for key in scheduled},
        scheduled_cached_reqs=NS(req_ids=[]),
        has_structured_output_requests=False,
    )
    return runner.prepare_inputs(scheduler, NS(num_tokens=padded * 6, num_reqs=padded, cg_mode=mode))


def _groups(api, monkeypatch):
    runs = [_metadata_run(api, monkeypatch, 64) for _ in range(3)]
    calls = []

    def sas(**kw):
        count = kw["batch_size"]
        assert kw["seqused_kv"].shape == (count,)
        assert kw["cu_seqlens_q"].shape == (count + 1,)
        assert torch.all(kw["cu_seqlens_q"].diff() == 6)
        assert kw["max_seqlen_q"] == 6
        calls.append(kw)
        # Persistent SAS buffers must also get this round's contents.
        return torch.full((1024,), len(calls), dtype=torch.int32)

    monkeypatch.setattr(api.dsa_globals["DeviceOperator"], "get_dsa_sparse_attn_metadata_op", staticmethod(lambda: sas))
    monkeypatch.setitem(api.dsa_globals, "F", torch.nn.functional)
    monkeypatch.setitem(api.dsa_globals, "get_full_cos_and_sin_dsa", lambda layer: (None, None))
    method_globals = dict(api.dsa_globals)
    _load_functions(
        ROOT / "vllm_ascend/attention/dsa_v1.py",
        method_globals,
        ["_num_compressor_metadata_rows"],
        "AscendDSAMetadataBuilder",
    )
    monkeypatch.setattr(
        api.builder, "_num_compressor_metadata_rows", method_globals["_num_compressor_metadata_rows"], raising=False
    )
    for index, (run, ratio) in enumerate(zip(runs, (1, 4, 128))):
        run.builder.compressor_ratio = ratio
        run.group.layer_names = [f"group{index}", f"group{index}.shared"]
    return runs, calls


def test_actual_and_padded_rows_update_all_shared_groups(api, monkeypatch):
    runner = _runner(api)
    runs, calls = _groups(api, monkeypatch)
    groups = [[run.group] for run in runs]
    config = NS(kv_cache_groups=[object() for _ in runs])
    blocks = tuple(torch.zeros(64, 8, dtype=torch.int32) for _ in runs)
    raw_slots = torch.full((3, 384), -1, dtype=torch.int32)
    snapshots = []
    # The smaller real batch still uses the same sparse 64-request graph tier.
    for step, actual in enumerate((64, 63, 49, 64)):
        runner.req_states.num_computed_tokens_cpu.add_(1)
        batch = _batch(runner, actual, 64, api.mode.FULL)
        assert batch.num_reqs == actual and batch.num_reqs_after_padding == 64
        assert batch.query_start_loc.shape == (actual + 1,) and batch.seq_lens.shape == (actual,)
        assert batch.query_start_loc_padded.shape == (65,) and batch.seq_lens_padded.shape == (64,)
        assert batch.query_start_loc_np.tolist() == list(range(0, 385, 6))
        assert np.all(batch.seq_lens_np[actual:] == 0)
        blocks[0].fill_(3)
        blocks[1].fill_(2)
        blocks[2].fill_(1)
        raw_slots.fill_(-1)
        for index in range(3):
            raw_slots[index, : actual * 6] = (3 - index) * 32 + batch.positions[: actual * 6] % 32
        metadata = runs[0].state.prepare_attn(batch, api.mode.FULL, blocks, raw_slots, groups, config)
        leaves = [metadata[f"group{i}"].decode for i in range(3)]
        for i, (run, leaf) in enumerate(zip(runs, leaves)):
            assert metadata[f"group{i}"] is metadata[f"group{i}.shared"]
            assert metadata[f"group{i}"].num_decodes == 64 and metadata[f"group{i}"].num_prefills == 0
            assert metadata[f"group{i}"].num_decode_tokens == actual * 6
            assert metadata[f"group{i}"].num_input_tokens == 384
            assert leaf.num_reqs_actual == actual
            torch.testing.assert_close(leaf.seq_lens[:actual], batch.seq_lens)
            torch.testing.assert_close(leaf.start_pos[:actual], batch.seq_lens - 6)
            assert torch.all(leaf.seq_lens[actual:] == 0) and torch.all(leaf.start_pos[actual:] == 0)
            assert torch.all(leaf.block_table[actual:] == 0)
            assert leaf.query_start_loc.tolist() == list(range(0, 385, 6))
            assert leaf.seq_lens.data_ptr() == runner.input_buffers.seq_lens.data_ptr()
            assert leaf.query_start_loc.data_ptr() == runner.input_buffers.query_start_loc.data_ptr()
            assert torch.all(leaf.sas_metadata == step * 3 + i + 1)
            assert run.builder.common_ratio_to_sas_metadata is runs[0].builder.common_ratio_to_sas_metadata
            assert run.builder.decode_ratio_to_sas_metadata is runs[0].builder.decode_ratio_to_sas_metadata
            if snapshots:
                old = snapshots[i]
                for name in ("seq_lens", "query_start_loc", "start_pos", "block_table", "sas_metadata"):
                    assert getattr(old, name).data_ptr() == getattr(leaf, name).data_ptr()
                    torch.testing.assert_close(getattr(old, name), getattr(leaf, name))
        slots = runs[0].builder.slot_mapping
        assert torch.all(slots[actual * 6 :, 0] == -1) and torch.all(slots[actual * 6 :, 1] == 31)
        # The real scatter selector skips exact -1 padding, even when the
        # captured leaf still has all 384 rows. Padding updates cannot touch KV.
        cache = torch.full((8, 32, 1, 512), 7, dtype=torch.bfloat16)
        updates = torch.full((384, 1, 512), 9, dtype=torch.bfloat16)
        updates[actual * 6 :].fill_(99)
        api.device.dsa_kv_compress_scatter(cache, updates, slots)
        assert torch.all(cache[0] == 7)
        assert not torch.any(cache == 99)
        assert torch.any(cache[3] == 9)
        assert len(api.scatter_calls[-1][1]) == actual * 6
        leaf = leaves[0]
        api.validate(
            layer_name="swa",
            kv_cache=cache,
            block_table=leaf.block_table,
            slot_mapping=leaf.slot_mapping,
            query_start_loc=leaf.query_start_loc,
            seqused_kv=leaf.seq_lens,
            sas_metadata=leaf.sas_metadata,
            sinks=torch.zeros(2),
            block_size=32,
            num_query_tokens=actual * 6,
            num_reqs_actual=actual,
        )
        # In the graph the compressor's scalar request count stays at capture
        # size. Its dummy work must only address core's reserved null block 0,
        # never a previous request page. Exercise the existing CPU op reference.
        reference = dict(torch=torch, KV_BLOCK_SIZE=32, SLOT_MAPPING_BLOCK_OFFSET=2)
        _load_functions(
            ROOT / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_metadata.py",
            reference,
            ["_reference_outputs"],
        )
        rope = torch.ones(256, 2)
        for i in (1, 2):
            held = snapshots[i] if snapshots else leaves[i]
            ratio = runs[i].builder.compressor_ratio
            case = NS(
                query_start_loc=held.query_start_loc.tolist(),
                start_pos=held.start_pos.tolist(),
                block_table=held.block_table.tolist(),
                expected_slot_mapping=None,
                num_rows=held.num_compressed_tokens,
                compress_ratio=ratio,
                slot_mapping_format=2,
            )
            _, _, cmp_slots = reference["_reference_outputs"](case, rope, rope)
            valid_rows = sum(
                (int(held.start_pos[row]) + 6) // ratio - int(held.start_pos[row]) // ratio for row in range(actual)
            )
            assert torch.all((cmp_slots[valid_rows:, 0] == 0) | (cmp_slots[valid_rows:, 0] == -1))
            assert not torch.any(cmp_slots[valid_rows:, 0] == 3 - i)
        if not snapshots:
            snapshots = leaves
        assert len(calls) == (step + 1) * 3


@pytest.mark.parametrize("mode_name", ["FULL", "NONE"])
def test_equal_actual_layout_and_invalid_short_input(api, monkeypatch, mode_name):
    runner = _runner(api)
    runs, _ = _groups(api, monkeypatch)
    mode = getattr(api.mode, mode_name)
    batch = _batch(runner, 4, 4, mode)
    result = runs[0].state.prepare_attn(
        batch,
        mode,
        (torch.zeros(4, 8, dtype=torch.int32),),
        torch.full((1, 24), -1, dtype=torch.int32),
        [[runs[0].group]],
        NS(kv_cache_groups=[object()]),
    )
    leaf = result["group0"].decode
    assert leaf.num_reqs_actual == 4 and leaf.seq_lens.shape == (4,)
    torch.testing.assert_close(leaf.start_pos, batch.seq_lens - 6)
    batch.seq_lens = batch.seq_lens[:3]
    batch.seq_lens_padded = batch.seq_lens
    with pytest.raises(ValueError, match="full request layout"):
        runs[0].state.prepare_attn(batch, mode, (), (), [], NS(kv_cache_groups=[]))


def test_invalid_actual_count_is_not_truncated(api):
    namespace = dict(api.dsa_globals)
    _load_functions(ROOT / "vllm_ascend/worker/v2/attn_utils.py", namespace, ["build_attn_metadata"])
    with pytest.raises(ValueError, match="Actual attention requests"):
        namespace["build_attn_metadata"](
            attn_groups=[],
            num_reqs=2,
            num_reqs_actual=3,
            num_tokens=12,
            query_start_loc_gpu=torch.arange(3) * 6,
            query_start_loc_cpu=torch.arange(3) * 6,
            max_query_len=6,
            seq_lens=torch.ones(2),
            max_seq_len=16,
            block_tables=(),
            slot_mappings=(),
            kv_cache_config=NS(kv_cache_groups=[]),
        )
