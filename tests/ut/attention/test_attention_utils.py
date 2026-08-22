import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    AscendPrefillContextParallelMetadata,
    split_decodes_and_prefills,
)


def build_common_attention_metadata(
    query_lens: list[int],
    seq_lens: list[int] | None = None,
    pcp_full_query_lens: list[int] | None = None,
    is_prefilling: list[bool] | None = None,
) -> AscendCommonAttentionMetadata:
    if seq_lens is None:
        seq_lens = [max(query_len, 1) + 8 for query_len in query_lens]
    if is_prefilling is None:
        is_prefilling = [False] * len(query_lens)

    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens, dtype=torch.int32).cumsum(dim=0).tolist()],
        dtype=torch.int32,
    )
    pcp_metadata = None
    if pcp_full_query_lens is not None:
        pcp_metadata = AscendPrefillContextParallelMetadata(
            query_lens_pcp_full_cpu=torch.tensor(pcp_full_query_lens, dtype=torch.int32),
            max_query_len_pcp_full=max(pcp_full_query_lens, default=0),
        )

    return AscendCommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens, default=0),
        decode_token_per_req=torch.ones(len(query_lens), dtype=torch.int32),
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.arange(max(sum(query_lens), 1), dtype=torch.int32),
        actual_seq_lengths_q=torch.arange(max(sum(query_lens), 1), dtype=torch.int32),
        positions=torch.arange(max(sum(query_lens), 1), dtype=torch.int32),
        attn_state=AscendAttentionState.PrefillNoCache,
        is_prefilling=torch.tensor(is_prefilling, dtype=torch.bool),
        num_computed_tokens_cpu=None,
        seq_lens=None,
        max_seq_len=max(seq_lens, default=0),
        prefill_context_parallel_metadata=pcp_metadata,
    )


def test_split_decodes_and_prefills_mixed_batch():
    common_attn_metadata = build_common_attention_metadata(
        query_lens=[2, 1, 5, 6],
    )

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=4,
    )

    assert num_decodes == 2
    assert num_prefills == 2
    assert num_decode_tokens == 3
    assert num_prefill_tokens == 11


def test_split_decodes_and_prefills_all_prefills():
    common_attn_metadata = build_common_attention_metadata(
        query_lens=[5, 6, 7],
    )

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=4,
    )

    assert num_decodes == 0
    assert num_prefills == 3
    assert num_decode_tokens == 0
    assert num_prefill_tokens == 18


def test_split_decodes_and_prefills_uses_pcp_full_query_lens():
    common_attn_metadata = build_common_attention_metadata(
        query_lens=[1, 1, 1],
        pcp_full_query_lens=[2, 5, 6],
    )

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=4,
    )

    assert num_decodes == 1
    assert num_prefills == 2
    assert num_decode_tokens == 1
    assert num_prefill_tokens == 2


def test_split_decodes_and_prefills_keeps_fresh_short_request_in_prefill():
    common_attn_metadata = build_common_attention_metadata(
        query_lens=[1],
        seq_lens=[1],
        is_prefilling=[True],
    )

    result = split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=6,
        treat_short_extends_as_decodes=False,
    )

    assert result == (0, 1, 0, 1)


def test_split_decodes_and_prefills_keeps_true_decode_at_k_plus_one():
    common_attn_metadata = build_common_attention_metadata(
        query_lens=[6],
        seq_lens=[7],
        is_prefilling=[False],
    )

    result = split_decodes_and_prefills(
        common_attn_metadata,
        decode_threshold=6,
        treat_short_extends_as_decodes=False,
    )

    assert result == (1, 0, 6, 0)
