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
PAGE_STRIDE = PAGE_ELEMENTS * 2


def _page_strided_cache() -> torch.Tensor:
    required_elements = (NUM_BLOCKS - 1) * PAGE_STRIDE + PAGE_ELEMENTS
    backing = torch.empty(required_elements, dtype=torch.bfloat16)
    return torch.as_strided(
        backing,
        size=(NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        stride=(PAGE_STRIDE, HEAD_DIM, HEAD_DIM, 1),
    )


def _contract_inputs() -> dict[str, object]:
    return {
        "layer_name": "model.layers.0.self_attn.attn",
        "kv_cache": _page_strided_cache(),
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


def test_sharedkv_contract_accepts_page_strided_backing_view() -> None:
    inputs = _contract_inputs()
    cache = inputs["kv_cache"]
    assert isinstance(cache, torch.Tensor)
    assert not cache.is_contiguous()
    assert cache.stride() == (PAGE_STRIDE, HEAD_DIM, HEAD_DIM, 1)

    validate_dsa_sharedkv_page_contract(**inputs)  # type: ignore[arg-type]

    last_element = cache.storage_offset() + sum(
        (size - 1) * stride for size, stride in zip(cache.shape, cache.stride())
    )
    assert last_element < cache.untyped_storage().nbytes() // cache.element_size()


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
    backing = torch.empty((NUM_BLOCKS - 1) * PAGE_STRIDE + PAGE_ELEMENTS * 2, dtype=torch.bfloat16)
    inputs["kv_cache"] = torch.as_strided(
        backing,
        size=(NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        stride=(PAGE_STRIDE, HEAD_DIM * 2, HEAD_DIM, 1),
    )

    with pytest.raises(ValueError, match="inner PA_ND layout must be dense"):
        validate_dsa_sharedkv_page_contract(**inputs)  # type: ignore[arg-type]


def test_sharedkv_query_contract_matches_cache_and_sink_shape() -> None:
    cache = _page_strided_cache()
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
