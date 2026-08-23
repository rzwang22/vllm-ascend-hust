# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
from dataclasses import dataclass
from math import sqrt

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.worker.v2.attn_utils import _adjust_dsv4_kv_layout  # noqa: E402

BLOCK_SIZE = 128
HEAD_DIM = 512
NUM_Q_HEADS = 64
NUM_KV_HEADS = 1
SOFTMAX_SCALE = 1.0 / sqrt(HEAD_DIM)
DEFAULT_WIN_LEFT = 127
DEFAULT_WIN_RIGHT = 0


@dataclass(frozen=True)
class SharedKVAddressDiagnosticCase:
    label: str
    block_size: int
    physical_block: int
    seqused_kv: int
    dtype: torch.dtype


SHAREDKV_ADDRESS_CASES = (
    pytest.param(
        SharedKVAddressDiagnosticCase("A", 32, 0, 1, torch.bfloat16),
        id="A-bs32-pb0-s2-1-bf16",
    ),
    pytest.param(
        SharedKVAddressDiagnosticCase("B", 32, 1, 16, torch.bfloat16),
        id="B-bs32-pb1-s2-16-bf16",
    ),
    pytest.param(
        SharedKVAddressDiagnosticCase("C", 32, 1, 32, torch.bfloat16),
        id="C-bs32-pb1-s2-32-bf16",
    ),
    pytest.param(
        SharedKVAddressDiagnosticCase("D", 128, 1, 1, torch.bfloat16),
        id="D-bs128-pb1-s2-1-bf16",
    ),
    pytest.param(
        SharedKVAddressDiagnosticCase("E", 32, 1, 1, torch.float16),
        id="E-bs32-pb1-s2-1-fp16",
    ),
)

SHAREDKV_S2_BOUNDARY_CASES = tuple(
    pytest.param(
        SharedKVAddressDiagnosticCase(f"S2-{seqused_kv}", 32, 1, seqused_kv, torch.bfloat16),
        id=f"s2-{seqused_kv}",
    )
    for seqused_kv in (1, 8, 15, 16, 17, 31, 32, 33)
)


@dataclass
class SparseAttnCase:
    q: torch.Tensor
    ori_kv: torch.Tensor
    ori_block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor | None
    seqused_kv: torch.Tensor
    sinks: torch.Tensor
    metadata: torch.Tensor
    q_cpu: torch.Tensor
    ori_kv_cpu: torch.Tensor
    block_table_cpu: torch.Tensor
    layout_q: str
    kv_tokens: int
    win_left: int


def _make_case(
    *,
    q_tokens: int = 1,
    kv_tokens: int = 512,
    layout_q: str = "TND",
    win_left: int = DEFAULT_WIN_LEFT,
) -> SparseAttnCase:
    assert kv_tokens % BLOCK_SIZE == 0
    num_blocks = kv_tokens // BLOCK_SIZE
    generator = torch.Generator().manual_seed(20260820 + q_tokens + kv_tokens)
    q_shape = (q_tokens, NUM_Q_HEADS, HEAD_DIM)
    if layout_q == "BSND":
        q_shape = (1, q_tokens, NUM_Q_HEADS, HEAD_DIM)
    q_cpu = torch.randn(q_shape, dtype=torch.float32, generator=generator).mul_(0.1).to(torch.float16)
    ori_kv_cpu = (
        torch.randn(
            (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.1)
        .to(torch.float16)
    )

    block_ids = torch.arange(num_blocks, dtype=torch.int32)
    if num_blocks > 1:
        block_ids = torch.roll(block_ids, shifts=1)
    block_table_cpu = block_ids.unsqueeze(0)

    q = q_cpu.npu()
    ori_kv = ori_kv_cpu.npu()
    ori_block_table = block_table_cpu.npu()
    cu_seqlens_q = None
    if layout_q == "TND":
        cu_seqlens_q = torch.tensor([0, q_tokens], dtype=torch.int32).npu()
    seqused_kv = torch.tensor([kv_tokens], dtype=torch.int32).npu()
    sinks = torch.full((NUM_Q_HEADS,), -10000.0, dtype=torch.float32).npu()

    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=NUM_Q_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=1,
        max_seqlen_q=q_tokens,
        max_seqlen_kv=kv_tokens,
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=win_left,
        ori_win_right=DEFAULT_WIN_RIGHT,
        layout_q=layout_q,
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device="npu:0",
    )
    return SparseAttnCase(
        q=q,
        ori_kv=ori_kv,
        ori_block_table=ori_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        q_cpu=q_cpu,
        ori_kv_cpu=ori_kv_cpu,
        block_table_cpu=block_table_cpu,
        layout_q=layout_q,
        kv_tokens=kv_tokens,
        win_left=win_left,
    )


def _run(case: SparseAttnCase, ori_sparse_indices: torch.Tensor | None) -> torch.Tensor:
    out, _ = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        case.q,
        ori_kv=case.ori_kv,
        cmp_kv=None,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=None,
        ori_block_table=case.ori_block_table,
        cmp_block_table=None,
        cu_seqlens_q=case.cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=case.seqused_kv,
        sinks=case.sinks,
        metadata=case.metadata,
        softmax_scale=SOFTMAX_SCALE,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=case.win_left,
        ori_win_right=DEFAULT_WIN_RIGHT,
        layout_q=case.layout_q,
        layout_kv="PA_ND",
        return_softmax_lse=False,
    )
    return out


def _physical_slots_for_logical_positions(case: SparseAttnCase, logical_positions: torch.Tensor) -> torch.Tensor:
    logical_blocks = torch.div(logical_positions, BLOCK_SIZE, rounding_mode="floor")
    block_offsets = logical_positions.remainder(BLOCK_SIZE)
    physical_blocks = case.block_table_cpu[0, logical_blocks].to(torch.int64)
    return physical_blocks * BLOCK_SIZE + block_offsets


def _reference_attention(case: SparseAttnCase, physical_slots: torch.Tensor) -> torch.Tensor:
    q = case.q_cpu.reshape(-1, NUM_Q_HEADS, HEAD_DIM)[0].float()
    flat_kv = case.ori_kv_cpu.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    selected_kv = flat_kv[physical_slots.to(torch.int64), 0].float()
    scores = torch.matmul(q, selected_kv.transpose(0, 1)) * SOFTMAX_SCALE
    return torch.matmul(torch.softmax(scores, dim=-1), selected_kv)


def _assert_matches_reference(out: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(out.reshape(-1, NUM_Q_HEADS, HEAD_DIM)[0].float().cpu(), expected, atol=3e-2, rtol=3e-2)


def test_none_path_matches_contiguous_window_reference() -> None:
    case = _make_case(kv_tokens=256)
    logical_positions = torch.arange(case.kv_tokens - BLOCK_SIZE, case.kv_tokens, dtype=torch.int64)
    physical_slots = _physical_slots_for_logical_positions(case, logical_positions)

    out = _run(case, ori_sparse_indices=None)

    _assert_matches_reference(out, _reference_attention(case, physical_slots))


def test_contiguous_physical_indices_match_none_path() -> None:
    case = _make_case(kv_tokens=256)
    logical_positions = torch.arange(case.kv_tokens - BLOCK_SIZE, case.kv_tokens, dtype=torch.int64)
    physical_slots = _physical_slots_for_logical_positions(case, logical_positions)
    ori_sparse_indices = physical_slots.to(torch.int32).reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    none_out = _run(case, ori_sparse_indices=None)
    indexed_out = _run(case, ori_sparse_indices=ori_sparse_indices)

    torch.testing.assert_close(indexed_out.float().cpu(), none_out.float().cpu(), atol=3e-2, rtol=3e-2)
    _assert_matches_reference(indexed_out, _reference_attention(case, physical_slots))


def test_discrete_physical_indices_match_pytorch_reference() -> None:
    index_width = 256
    case = _make_case(kv_tokens=512, win_left=index_width - 1)
    physical_slots = (torch.arange(index_width, dtype=torch.int64) * 193 + 17).remainder(case.kv_tokens)
    ori_sparse_indices = physical_slots.to(torch.int32).reshape(1, NUM_KV_HEADS, index_width).npu()

    out = _run(case, ori_sparse_indices=ori_sparse_indices)

    _assert_matches_reference(out, _reference_attention(case, physical_slots))


def test_minus_one_terminates_sparse_slot_reads() -> None:
    case = _make_case(kv_tokens=512)
    valid_slots = (torch.arange(37, dtype=torch.int64) * 11 + 3).remainder(case.kv_tokens)
    first_indices = torch.full((BLOCK_SIZE,), -1, dtype=torch.int32)
    second_indices = torch.full((BLOCK_SIZE,), -1, dtype=torch.int32)
    first_indices[: valid_slots.numel()] = valid_slots.to(torch.int32)
    second_indices[: valid_slots.numel()] = valid_slots.to(torch.int32)
    first_indices[valid_slots.numel() + 1 :] = torch.arange(
        BLOCK_SIZE - valid_slots.numel() - 1, dtype=torch.int32
    ).remainder(case.kv_tokens)
    second_indices[valid_slots.numel() + 1 :] = (
        torch.arange(BLOCK_SIZE - valid_slots.numel() - 1, dtype=torch.int32).mul(7).add(101).remainder(case.kv_tokens)
    )

    first_out = _run(case, first_indices.reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu())
    second_out = _run(case, second_indices.reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu())

    torch.testing.assert_close(first_out.float().cpu(), second_out.float().cpu(), atol=3e-2, rtol=3e-2)
    _assert_matches_reference(first_out, _reference_attention(case, valid_slots))


def _build_diagnostic_block_table(
    case: SharedKVAddressDiagnosticCase,
) -> tuple[int, torch.Tensor]:
    num_logical_blocks = (case.seqused_kv + case.block_size - 1) // case.block_size
    num_physical_blocks = max(case.physical_block + 1, num_logical_blocks)
    physical_blocks = [case.physical_block]
    physical_blocks.extend(block_id for block_id in range(num_physical_blocks) if block_id != case.physical_block)
    return num_physical_blocks, torch.tensor([physical_blocks[:num_logical_blocks]], dtype=torch.int32)


def _build_diagnostic_slot_mapping(
    case: SharedKVAddressDiagnosticCase,
    block_table_cpu: torch.Tensor,
) -> torch.Tensor:
    logical_slots = torch.arange(case.seqused_kv, dtype=torch.int64)
    logical_blocks = torch.div(logical_slots, case.block_size, rounding_mode="floor")
    block_offsets = logical_slots.remainder(case.block_size)
    physical_blocks = block_table_cpu[0, logical_blocks].to(torch.int64)
    return torch.stack((physical_blocks, block_offsets), dim=1).to(torch.int32)


def _diagnostic_reference_attention(
    q_cpu: torch.Tensor,
    kv_cpu: torch.Tensor,
) -> torch.Tensor:
    q = q_cpu[0].float()
    kv = kv_cpu[:, 0].float()
    scores = torch.matmul(q, kv.transpose(0, 1)) * SOFTMAX_SCALE
    return torch.matmul(torch.softmax(scores, dim=-1), kv).unsqueeze(0)


def _run_sharedkv_address_diagnostic(
    case: SharedKVAddressDiagnosticCase,
) -> None:
    """Run one synchronized PA_ND/TND scatter plus shared-KV read."""
    if get_ascend_device_type() is not AscendDeviceType.A2:
        pytest.skip("The shared-KV address regression targets Ascend 910B2/arch32.")

    num_physical_blocks, block_table_cpu = _build_diagnostic_block_table(case)
    slot_mapping_cpu = _build_diagnostic_slot_mapping(case, block_table_cpu)
    element_size = torch.tensor([], dtype=case.dtype).element_size()
    page_elements = case.block_size * NUM_KV_HEADS * HEAD_DIM
    page_stride_bytes = page_elements * element_size
    page_offset_bytes = 0
    backing_bytes = num_physical_blocks * page_stride_bytes
    layer_name = "model.layers.0.self_attn.swa_cache"

    tensors: list[torch.Tensor] = []
    try:
        backing = torch.zeros(backing_bytes, dtype=torch.uint8, device="npu:0")
        (swa_kv_cache,) = _adjust_dsv4_kv_layout(
            backing,
            [
                (
                    num_physical_blocks,
                    case.block_size,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                )
            ],
            [case.dtype],
            page_stride_bytes,
            page_offset_bytes,
        )
        generator = torch.Generator().manual_seed(
            20260823 + case.block_size + case.physical_block + case.seqused_kv + element_size
        )
        q_cpu = (
            torch.randn(
                (1, NUM_Q_HEADS, HEAD_DIM),
                dtype=torch.float32,
                generator=generator,
            )
            .mul_(0.1)
            .to(case.dtype)
        )
        kv_cpu = (
            torch.randn(
                (case.seqused_kv, NUM_KV_HEADS, HEAD_DIM),
                dtype=torch.float32,
                generator=generator,
            )
            .mul_(0.1)
            .to(case.dtype)
        )
        q = q_cpu.npu()
        kv = kv_cpu.npu()
        ori_block_table = block_table_cpu.npu()
        slot_mapping = slot_mapping_cpu.npu()
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="npu:0")
        seqused_kv = torch.tensor([case.seqused_kv], dtype=torch.int32, device="npu:0")
        sinks = torch.full((NUM_Q_HEADS,), -10000.0, dtype=torch.float32, device="npu:0")
        metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=NUM_Q_HEADS,
            num_heads_kv=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            batch_size=1,
            max_seqlen_q=1,
            max_seqlen_kv=case.seqused_kv,
            ori_topk=0,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=DEFAULT_WIN_LEFT,
            ori_win_right=DEFAULT_WIN_RIGHT,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
            device="npu:0",
        )
        tensors.extend(
            (
                backing,
                swa_kv_cache,
                q,
                kv,
                ori_block_table,
                slot_mapping,
                cu_seqlens_q,
                seqused_kv,
                sinks,
                metadata,
            )
        )

        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)
        torch.npu.synchronize()

        cache_storage_offset_bytes = swa_kv_cache.storage_offset() * element_size
        block_stride_bytes = swa_kv_cache.stride(0) * element_size
        token_stride_bytes = swa_kv_cache.stride(1) * element_size
        accessed_starts = [
            cache_storage_offset_bytes
            + int(physical_block) * block_stride_bytes
            + int(block_offset) * token_stride_bytes
            for physical_block, block_offset in slot_mapping_cpu.tolist()
        ]
        access_min_bytes = min(accessed_starts)
        access_max_bytes_exclusive = max(accessed_starts) + HEAD_DIM * element_size
        assert swa_kv_cache.data_ptr() == backing.data_ptr() + page_offset_bytes
        assert block_stride_bytes == page_stride_bytes
        assert access_min_bytes >= 0
        assert access_max_bytes_exclusive <= backing.untyped_storage().nbytes()
        diagnostic = {
            "case": case.label,
            "rank": int(os.getenv("RANK", "0")),
            "layer_name": layer_name,
            "compress_ratio": 1,
            "dtype": str(case.dtype),
            "block_size": case.block_size,
            "physical_block": case.physical_block,
            "seqused_kv": case.seqused_kv,
            "q": {
                "shape": list(q.shape),
                "dtype": str(q.dtype),
                "stride": list(q.stride()),
                "data_ptr": q.data_ptr(),
            },
            "kv": {
                "shape": list(swa_kv_cache.shape),
                "dtype": str(swa_kv_cache.dtype),
                "stride": list(swa_kv_cache.stride()),
                "storage_offset": swa_kv_cache.storage_offset(),
                "data_ptr": swa_kv_cache.data_ptr(),
            },
            "backing": {
                "data_ptr": backing.data_ptr(),
                "storage_bytes": backing.untyped_storage().nbytes(),
            },
            "ori_block_table": block_table_cpu.tolist(),
            "slot_mapping": slot_mapping_cpu.tolist(),
            "cu_seqlens_q": [0, 1],
            "cu_seqlens_ori_kv": None,
            "sas_metadata": {"shape": list(metadata.shape), "dtype": str(metadata.dtype)},
            "sinks": {"shape": list(sinks.shape), "dtype": str(sinks.dtype)},
            "theoretical_accessed_byte_range": {
                "min": access_min_bytes,
                "max_exclusive": access_max_bytes_exclusive,
            },
        }
        print(
            "SHAREDKV_ADDRESS_DIAGNOSTIC " + json.dumps(diagnostic, sort_keys=True),
            flush=True,
        )

        out, _ = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=swa_kv_cache,
            cmp_kv=None,
            ori_sparse_indices=None,
            cmp_sparse_indices=None,
            ori_block_table=ori_block_table,
            cmp_block_table=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=SOFTMAX_SCALE,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=DEFAULT_WIN_LEFT,
            ori_win_right=DEFAULT_WIN_RIGHT,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=False,
        )
        tensors.append(out)
        torch.npu.synchronize()

        expected = _diagnostic_reference_attention(q_cpu, kv_cpu)
        assert out.dtype == case.dtype
        assert out.shape == expected.shape
        assert torch.isfinite(out).all().cpu().item()
        torch.testing.assert_close(out.float().cpu(), expected.float(), atol=3e-2, rtol=3e-2)
        print(f"SHAREDKV_ADDRESS_PASS case={case.label}", flush=True)
    finally:
        tensors.clear()
        gc.collect()
        torch.npu.empty_cache()


@pytest.mark.parametrize("case", SHAREDKV_ADDRESS_CASES)
def test_first_token_prefill_reads_kernel_compatible_swa_page(
    case: SharedKVAddressDiagnosticCase,
) -> None:
    _run_sharedkv_address_diagnostic(case)


@pytest.mark.parametrize("case", SHAREDKV_S2_BOUNDARY_CASES)
def test_first_token_prefill_sharedkv_s2_boundaries(
    case: SharedKVAddressDiagnosticCase,
) -> None:
    _run_sharedkv_address_diagnostic(case)


def test_rejects_wrong_ori_sparse_indices_dtype() -> None:
    case = _make_case()
    indices = torch.arange(BLOCK_SIZE, dtype=torch.int64).reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()


def test_rejects_non_tnd_ori_sparse_indices_layout() -> None:
    case = _make_case(layout_q="BSND")
    indices = torch.arange(BLOCK_SIZE, dtype=torch.int32).reshape(1, 1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()


def test_rejects_query_and_index_token_shape_mismatch() -> None:
    case = _make_case(q_tokens=2)
    indices = torch.arange(BLOCK_SIZE, dtype=torch.int32).reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()


@pytest.mark.parametrize("index_width", [0, 64, 129, 2176])
def test_rejects_invalid_or_unaligned_index_width(index_width: int) -> None:
    case = _make_case()
    indices = torch.zeros((1, NUM_KV_HEADS, index_width), dtype=torch.int32).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()
