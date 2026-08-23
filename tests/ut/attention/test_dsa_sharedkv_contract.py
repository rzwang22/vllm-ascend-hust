from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import (
    AscendDSADecodeMetadata,
    AscendDSAPrefillMetadata,
    _should_validate_dspark_sharedkv_contract,
    validate_dsa_sharedkv_page_contract,
    validate_dsa_sharedkv_query_contract,
)
from vllm_ascend.utils import AscendDeviceType

BLOCK_SIZE = 32
HEAD_DIM = 64
NUM_BLOCKS = 4
PAGE_ELEMENTS = BLOCK_SIZE * HEAD_DIM
PACKED_PAGE_COUNT = 4
DEFAULT_PAGE_SLOT = 1


def _page_local_cache(page_slot: int = DEFAULT_PAGE_SLOT) -> torch.Tensor:
    page_offset = page_slot * PAGE_ELEMENTS
    required_elements = page_offset + NUM_BLOCKS * PAGE_ELEMENTS
    backing = torch.empty(required_elements, dtype=torch.bfloat16)
    return torch.as_strided(
        backing,
        size=(NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        stride=(PAGE_ELEMENTS, HEAD_DIM, HEAD_DIM, 1),
        storage_offset=page_offset,
    )


def _packed_cache(page_slot: int) -> torch.Tensor:
    page_offset = page_slot * PAGE_ELEMENTS
    packed_stride = PAGE_ELEMENTS * PACKED_PAGE_COUNT
    required_elements = page_offset + (NUM_BLOCKS - 1) * packed_stride + PAGE_ELEMENTS
    backing = torch.empty(required_elements, dtype=torch.bfloat16)
    return torch.as_strided(
        backing,
        size=(NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        stride=(packed_stride, HEAD_DIM, HEAD_DIM, 1),
        storage_offset=page_offset,
    )


def _contract_inputs() -> dict[str, object]:
    return {
        "layer_name": "model.layers.0.self_attn.attn",
        "kv_cache": _page_local_cache(),
        "block_table": torch.tensor([[0, 1]], dtype=torch.int32),
        "slot_mapping": torch.tensor([[0, 0]], dtype=torch.int32),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "seqused_kv": torch.tensor([1], dtype=torch.int32),
        "sas_metadata": torch.zeros(1024, dtype=torch.int32),
        "sinks": torch.zeros(2, dtype=torch.float32),
        "block_size": BLOCK_SIZE,
        "num_query_tokens": 1,
        "num_reqs_actual": 1,
    }


def test_sharedkv_contract_accepts_nonzero_offset_page_local_view() -> None:
    inputs = _contract_inputs()
    cache = inputs["kv_cache"]
    assert isinstance(cache, torch.Tensor)
    assert cache.storage_offset() == DEFAULT_PAGE_SLOT * PAGE_ELEMENTS
    assert cache.stride() == (PAGE_ELEMENTS, HEAD_DIM, HEAD_DIM, 1)

    validate_dsa_sharedkv_page_contract(**inputs)  # type: ignore[arg-type]

    last_element = cache.storage_offset() + sum(
        (size - 1) * stride for size, stride in zip(cache.shape, cache.stride())
    )
    assert last_element < cache.untyped_storage().nbytes() // cache.element_size()


@pytest.mark.parametrize(
    ("compress_ratio", "page_slot", "layer_name"),
    [
        (1, 1, "model.layers.0.self_attn.swa_cache"),
        (4, 2, "model.layers.2.self_attn.attn"),
        (128, 3, "model.layers.3.self_attn.attn"),
    ],
)
def test_sharedkv_contract_rejects_interleaved_packed_block_stride(
    compress_ratio: int,
    page_slot: int,
    layer_name: str,
) -> None:
    cache = _packed_cache(page_slot)
    physical_block = NUM_BLOCKS - 1
    block_offset = BLOCK_SIZE - 1
    element_size = cache.element_size()
    page_offset_bytes = page_slot * PAGE_ELEMENTS * element_size
    packed_stride_bytes = PAGE_ELEMENTS * PACKED_PAGE_COUNT * element_size

    # Physical block IDs are local to the KV group. The cache view contributes
    # its page offset once through data_ptr(), while the custom op applies the
    # packed stride once for the selected physical block.
    expected_block_base = cache.untyped_storage().data_ptr() + page_offset_bytes + physical_block * packed_stride_bytes
    assert cache.data_ptr() == cache.untyped_storage().data_ptr() + page_offset_bytes
    assert cache[physical_block].data_ptr() == expected_block_base

    last_element_offset = (
        page_offset_bytes
        + physical_block * packed_stride_bytes
        + (block_offset * HEAD_DIM + HEAD_DIM - 1) * element_size
    )
    assert last_element_offset < cache.untyped_storage().nbytes()

    inputs = _contract_inputs()
    inputs.update(
        layer_name=f"{layer_name}[ratio={compress_ratio}]",
        kv_cache=cache,
        block_table=torch.tensor([[physical_block]], dtype=torch.int32),
        slot_mapping=torch.tensor([[physical_block, block_offset]], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="Interleaved packed KV pages are unsupported"):
        validate_dsa_sharedkv_page_contract(**inputs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        (
            "block_table",
            torch.tensor([[NUM_BLOCKS]], dtype=torch.int32),
            "physical block outside ori_kv",
        ),
        (
            "slot_mapping",
            torch.tensor([[NUM_BLOCKS, 0]], dtype=torch.int32),
            "physical block outside ori_kv",
        ),
        (
            "slot_mapping",
            torch.tensor([[0, BLOCK_SIZE]], dtype=torch.int32),
            "offset outside the cache block",
        ),
        (
            "query_start_loc",
            torch.tensor([1, 2], dtype=torch.int32),
            "must start at zero",
        ),
        (
            "query_start_loc",
            torch.tensor([0, 0], dtype=torch.int32),
            "strictly increasing",
        ),
        (
            "query_start_loc",
            torch.tensor([0, 2], dtype=torch.int32),
            "terminal value",
        ),
        (
            "seqused_kv",
            torch.tensor([0], dtype=torch.int32),
            "include every current query token",
        ),
        (
            "seqused_kv",
            torch.tensor([NUM_BLOCKS * BLOCK_SIZE + 1], dtype=torch.int32),
            "physical cache capacity",
        ),
    ],
)
def test_sharedkv_contract_rejects_invalid_indices_before_kernel(
    name: str,
    value: torch.Tensor,
    match: str,
) -> None:
    inputs = _contract_inputs()
    inputs[name] = value

    with pytest.raises(ValueError, match=match):
        validate_dsa_sharedkv_page_contract(**inputs)  # type: ignore[arg-type]


def test_sharedkv_contract_rejects_non_dense_inner_cache_layout() -> None:
    inputs = _contract_inputs()
    backing = torch.empty(NUM_BLOCKS * PAGE_ELEMENTS * 2, dtype=torch.bfloat16)
    inputs["kv_cache"] = torch.as_strided(
        backing,
        size=(NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        stride=(PAGE_ELEMENTS * 2, HEAD_DIM * 2, HEAD_DIM, 1),
    )

    with pytest.raises(ValueError, match="inner PA_ND layout must be dense"):
        validate_dsa_sharedkv_page_contract(**inputs)  # type: ignore[arg-type]


def test_sharedkv_query_contract_matches_cache_and_sink_shape() -> None:
    cache = _page_local_cache()
    query = torch.empty((1, 2, HEAD_DIM), dtype=cache.dtype)
    sinks = torch.empty(2, dtype=torch.float32)

    validate_dsa_sharedkv_query_contract(
        layer_name="model.layers.0.self_attn.attn",
        query=query,
        kv_cache=cache,
        sinks=sinks,
        num_query_tokens=1,
    )


@pytest.mark.parametrize(
    ("device_type", "method", "expected"),
    [
        (AscendDeviceType.A2, "dspark", True),
        (AscendDeviceType.A2, "mtp", False),
        (AscendDeviceType.A2, None, False),
        (AscendDeviceType.A5, "dspark", False),
    ],
)
def test_sharedkv_contract_validation_is_scoped_to_arch32_dspark(
    monkeypatch: pytest.MonkeyPatch,
    device_type: AscendDeviceType,
    method: str | None,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        "vllm_ascend.attention.dsa_v1.get_ascend_device_type",
        lambda: device_type,
    )
    speculative_config = None if method is None else type("SpecConfig", (), {"method": method})()
    vllm_config = type("Config", (), {"speculative_config": speculative_config})()

    assert _should_validate_dspark_sharedkv_contract(vllm_config) is expected  # type: ignore[arg-type]


def test_sharedkv_metadata_validation_marker_is_per_path_instance() -> None:
    common = dict(
        input_positions=torch.zeros(1),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        max_seq_lens=1,
        slot_mapping=torch.zeros((1, 2), dtype=torch.int32),
        block_size=BLOCK_SIZE,
        num_reqs_actual=1,
        sas_metadata=torch.zeros(1024, dtype=torch.int32),
    )
    decode = AscendDSADecodeMetadata(
        max_seqlen_kv=1,
        max_seqlen_q=1,
        seq_lens_list=[1],
        **common,
    )
    prefill = AscendDSAPrefillMetadata(
        attn_mask=torch.empty(0),
        query_lens=torch.ones(1, dtype=torch.int32),
        context_lens=torch.ones(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        **common,
    )

    decode.sharedkv_contract_validated = True
    assert decode.sharedkv_contract_validated
    assert replace(decode).sharedkv_contract_validated
    assert not prefill.sharedkv_contract_validated
