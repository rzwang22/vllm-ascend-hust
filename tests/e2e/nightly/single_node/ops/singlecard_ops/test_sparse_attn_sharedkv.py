# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from math import sqrt

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

BLOCK_SIZE = 128
HEAD_DIM = 512
NUM_Q_HEADS = 64
NUM_KV_HEADS = 1
SOFTMAX_SCALE = 1.0 / sqrt(HEAD_DIM)
DEFAULT_WIN_LEFT = 127
DEFAULT_WIN_RIGHT = 0


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
    ori_kv_cpu = torch.randn(
        (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.1).to(torch.float16)

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
    second_indices[valid_slots.numel() + 1 :] = torch.arange(
        BLOCK_SIZE - valid_slots.numel() - 1, dtype=torch.int32
    ).mul(7).add(101).remainder(case.kv_tokens)

    first_out = _run(case, first_indices.reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu())
    second_out = _run(case, second_indices.reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu())

    torch.testing.assert_close(first_out.float().cpu(), second_out.float().cpu(), atol=3e-2, rtol=3e-2)
    _assert_matches_reference(first_out, _reference_attention(case, valid_slots))


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
