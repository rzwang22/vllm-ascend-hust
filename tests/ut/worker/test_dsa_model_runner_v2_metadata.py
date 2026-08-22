# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import (
    AscendDSABackend,
    AscendDSAMetadataBuilder,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class _RecordingDSAMetadataBuilder(AscendDSAMetadataBuilder):
    def __init__(
        self,
        calls: list[dict[str, Any]],
        compress_ratio: int,
        decode_threshold: int,
    ) -> None:
        self.calls = calls
        self.compressor_ratio = compress_ratio
        self.decode_threshold = decode_threshold

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
        **kwargs,
    ):
        del common_prefix_len, fast_build
        self.prefill_ratio_to_sas_metadata = kwargs["prefill_ratio_to_sas_metadata"]
        self.decode_ratio_to_sas_metadata = kwargs["decode_ratio_to_sas_metadata"]
        self.common_ratio_to_sas_metadata = kwargs["common_ratio_to_sas_metadata"]

        query_lens = (common_attn_metadata.query_start_loc_cpu[1:] - common_attn_metadata.query_start_loc_cpu[:-1])[
            : common_attn_metadata.num_reqs
        ]
        num_decodes = int((query_lens <= self.decode_threshold).sum())
        num_decode_tokens = int(query_lens[:num_decodes].sum())
        num_prefills = common_attn_metadata.num_reqs - num_decodes
        num_prefill_tokens = common_attn_metadata.num_actual_tokens - num_decode_tokens
        self.common_ratio_to_sas_metadata.setdefault("num_decodes", num_decodes)
        self.common_ratio_to_sas_metadata.setdefault("num_prefills", num_prefills)
        self.common_ratio_to_sas_metadata.setdefault(
            "num_decode_tokens",
            num_decode_tokens,
        )
        self.common_ratio_to_sas_metadata.setdefault(
            "num_prefill_tokens",
            num_prefill_tokens,
        )
        self.common_ratio_to_sas_metadata.setdefault(
            "positions",
            common_attn_metadata.positions[: common_attn_metadata.num_input_tokens],
        )
        self.common_ratio_to_sas_metadata.setdefault(
            "seq_lens",
            common_attn_metadata.seq_lens,
        )
        ratio_key = f"c{self.compressor_ratio}"
        if num_prefills:
            self.prefill_ratio_to_sas_metadata.setdefault(
                ratio_key,
                (
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                ),
            )
        if num_decodes:
            self.decode_ratio_to_sas_metadata.setdefault(
                ratio_key,
                (
                    common_attn_metadata.block_table_tensor,
                    common_attn_metadata.slot_mapping,
                ),
            )

        call = {
            "common_attn_metadata": common_attn_metadata,
            "block_size": kwargs["block_size"],
            "num_reqs_actual": kwargs["num_reqs_actual"],
            "prefill": self.prefill_ratio_to_sas_metadata,
            "decode": self.decode_ratio_to_sas_metadata,
            "common": self.common_ratio_to_sas_metadata,
            "ratio_key": ratio_key,
        }
        self.calls.append(call)
        return SimpleNamespace(common_attn_metadata=common_attn_metadata)


class _FailingDSAMetadataBuilder(_RecordingDSAMetadataBuilder):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        raise RuntimeError("DSA metadata construction failed")


class _GenericMetadataBuilder:
    def __init__(self) -> None:
        self.kwargs = None

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        del common_prefix_len
        self.kwargs = kwargs
        return SimpleNamespace(common_attn_metadata=common_attn_metadata)


class _GenericBackend:
    pass


def _metadata_groups(
    *,
    decode_threshold: int = 1,
    failing_second_group: bool = False,
):
    layer_names = (
        "model.layers.0.self_attn.swa_cache",
        "model.layers.0.self_attn.compressor.state_cache",
    )
    specs = tuple(
        AscendMLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=compress_ratio,
        )
        for block_size, compress_ratio in ((32, 1), (64, 4))
    )
    calls: list[dict[str, Any]] = []
    builders: list[AscendDSAMetadataBuilder] = [
        _RecordingDSAMetadataBuilder(calls, 1, decode_threshold),
        (
            _FailingDSAMetadataBuilder(calls, 4, decode_threshold)
            if failing_second_group
            else _RecordingDSAMetadataBuilder(calls, 4, decode_threshold)
        ),
    ]
    attn_groups = [
        [
            AttentionGroup(
                backend=AscendDSABackend,
                layer_names=[layer_name],
                kv_cache_spec=spec,
                kv_cache_group_id=group_id,
                metadata_builders=[builder],
            )
        ]
        for group_id, (layer_name, spec, builder) in enumerate(zip(layer_names, specs, builders))
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
            for layer_name, spec in zip(layer_names, specs)
        ],
    )
    return layer_names, specs, calls, attn_groups, kv_cache_config


def _build_metadata(
    query_lens: list[int],
    seq_lens: list[int],
    *,
    attn_state: AscendAttentionState,
    num_input_tokens: int | None = None,
    decode_threshold: int = 1,
    failing_second_group: bool = False,
):
    layer_names, specs, calls, attn_groups, kv_cache_config = _metadata_groups(
        decode_threshold=decode_threshold,
        failing_second_group=failing_second_group,
    )
    num_reqs = len(query_lens)
    num_actual_tokens = sum(query_lens)
    if num_input_tokens is None:
        num_input_tokens = num_actual_tokens
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    positions = torch.arange(num_input_tokens, dtype=torch.int64)
    block_tables = tuple(
        torch.full((num_reqs, 2), group_id + 10, dtype=torch.int32) for group_id in range(len(attn_groups))
    )
    slot_mappings = torch.stack(
        [torch.arange(num_input_tokens, dtype=torch.int32) + group_id * 100 for group_id in range(len(attn_groups))]
    )
    metadata = attn_utils.build_attn_metadata(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_input_tokens,
        num_actual_tokens=num_actual_tokens,
        num_input_tokens=num_input_tokens,
        query_start_loc_gpu=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        max_query_len=max(query_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        max_seq_len=max(seq_lens),
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        seq_lens_np=np.array(seq_lens, dtype=np.int32),
        positions=positions,
        attn_state=attn_state,
    )
    return SimpleNamespace(
        metadata=metadata,
        layer_names=layer_names,
        specs=specs,
        calls=calls,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        positions=positions,
    )


@pytest.mark.parametrize(
    ("speculative_method", "decode_threshold"),
    [(None, 1), ("dspark", 6)],
)
def test_dsa_single_request_prefill_metadata_is_complete(
    speculative_method: str | None,
    decode_threshold: int,
) -> None:
    result = _build_metadata(
        [7],
        [7],
        attn_state=AscendAttentionState.PrefillNoCache,
        decode_threshold=decode_threshold,
    )

    assert (decode_threshold > 1) is (speculative_method == "dspark")
    assert set(result.metadata) == set(result.layer_names)
    assert all(call["prefill"] for call in result.calls)
    assert result.calls[0]["decode"] == {}
    assert result.calls[0]["common"]["num_prefills"] == 1
    assert result.calls[0]["common"]["num_prefill_tokens"] == 7
    assert result.calls[0]["common"]["positions"].data_ptr() == (result.positions.data_ptr())
    assert result.calls[0]["common_attn_metadata"].attn_state is (AscendAttentionState.PrefillNoCache)


def test_dsa_single_request_decode_metadata_is_complete() -> None:
    result = _build_metadata(
        [1],
        [9],
        attn_state=AscendAttentionState.DecodeOnly,
    )

    assert all(call["decode"] for call in result.calls)
    assert result.calls[0]["prefill"] == {}
    assert result.calls[0]["common"]["num_decodes"] == 1
    assert result.calls[0]["common"]["num_decode_tokens"] == 1
    assert result.calls[0]["common_attn_metadata"].seq_lens.tolist() == [9]


def test_dsa_mixed_prefill_decode_preserves_request_and_tensor_contract() -> None:
    result = _build_metadata(
        [1, 3],
        [8, 3],
        attn_state=AscendAttentionState.ChunkedPrefill,
    )

    common = result.calls[0]["common"]
    assert common["num_decodes"] == 1
    assert common["num_prefills"] == 1
    assert common["num_decode_tokens"] == 1
    assert common["num_prefill_tokens"] == 3
    assert result.calls[0]["common_attn_metadata"].query_start_loc.tolist() == [
        0,
        1,
        4,
    ]
    assert result.calls[0]["common_attn_metadata"].positions is result.positions


@pytest.mark.parametrize(
    "attn_state",
    [
        AscendAttentionState.PrefillNoCache,
        AscendAttentionState.PrefillCacheHit,
    ],
)
def test_dsa_prefill_attention_state_is_forwarded(
    attn_state: AscendAttentionState,
) -> None:
    result = _build_metadata([2], [7], attn_state=attn_state)

    assert result.calls[0]["common_attn_metadata"].attn_state is attn_state
    assert result.calls[0]["prefill"]


def test_dsa_ratio_groups_share_metadata_but_keep_cache_layout_identity() -> None:
    result = _build_metadata(
        [1, 3],
        [8, 3],
        attn_state=AscendAttentionState.ChunkedPrefill,
    )

    assert [call["block_size"] for call in result.calls] == [32, 64]
    assert [call["ratio_key"] for call in result.calls] == ["c1", "c4"]
    for cache_name in ("prefill", "decode", "common"):
        assert result.calls[0][cache_name] is result.calls[1][cache_name]
    for group_id, call in enumerate(result.calls):
        common_metadata = call["common_attn_metadata"]
        assert common_metadata.block_table_tensor is result.block_tables[group_id]
        assert common_metadata.slot_mapping.data_ptr() == (result.slot_mappings[group_id].data_ptr())


def test_dsa_metadata_is_fresh_for_every_scheduler_step() -> None:
    first = _build_metadata(
        [3],
        [3],
        attn_state=AscendAttentionState.PrefillNoCache,
    )
    second = _build_metadata(
        [1],
        [4],
        attn_state=AscendAttentionState.DecodeOnly,
    )

    for cache_name in ("prefill", "decode", "common"):
        assert first.calls[0][cache_name] is not second.calls[0][cache_name]
    assert "c1" not in second.calls[0]["prefill"]
    assert second.calls[0]["common"]["num_decodes"] == 1


def test_dsa_builder_fails_closed_without_ratio_metadata_provider() -> None:
    builder = AscendDSAMetadataBuilder.__new__(AscendDSAMetadataBuilder)
    common_metadata = SimpleNamespace(
        num_reqs=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )

    with pytest.raises(AssertionError):
        builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_metadata,
        )


def test_non_dsa_attention_builder_receives_no_dsa_specific_kwargs() -> None:
    builder = _GenericMetadataBuilder()
    layer_name = "model.layers.0.self_attn.attn"
    spec = AscendMLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    attn_group = AttentionGroup(
        backend=_GenericBackend,
        layer_names=[layer_name],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        metadata_builders=[builder],
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=spec,
            )
        ],
    )

    metadata = attn_utils.build_attn_metadata(
        attn_groups=[[attn_group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        max_query_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_seq_len=1,
        block_tables=(torch.zeros((1, 1), dtype=torch.int32),),
        slot_mappings=torch.zeros((1, 1), dtype=torch.int32),
        kv_cache_config=kv_cache_config,
        seq_lens_np=np.array([1], dtype=np.int32),
        positions=torch.zeros(1, dtype=torch.int64),
    )

    assert set(metadata) == {layer_name}
    assert builder.kwargs == {}


def test_model_state_distinguishes_actual_and_graph_padded_tokens() -> None:
    layer_names, specs, calls, attn_groups, kv_cache_config = _metadata_groups()
    model_state = AscendModelState.__new__(AscendModelState)
    model_state.max_model_len = 16
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        num_tokens=4,
        num_tokens_after_padding=8,
        query_start_loc_np=np.array([0, 1, 4, 8], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 4, 8], dtype=torch.int32),
        num_scheduled_tokens=np.array([1, 3], dtype=np.int32),
        seq_lens=torch.tensor([8, 3, 0], dtype=torch.int32),
        seq_lens_np=np.array([8, 3, 0], dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.arange(8, dtype=torch.int64),
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    block_tables = tuple(torch.zeros((3, 2), dtype=torch.int32) for _ in attn_groups)
    slot_mappings = torch.zeros((len(attn_groups), 8), dtype=torch.int32)

    metadata = model_state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.FULL,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
    )

    assert set(metadata) == set(layer_names)
    assert [call["block_size"] for call in calls] == [spec.block_size for spec in specs]
    assert all(call["common_attn_metadata"].num_actual_tokens == 4 for call in calls)
    assert all(call["common_attn_metadata"].num_input_tokens == 8 for call in calls)


def test_model_state_preserves_previous_metadata_on_builder_failure() -> None:
    _, _, _, attn_groups, kv_cache_config = _metadata_groups(
        failing_second_group=True,
    )
    model_state = AscendModelState.__new__(AscendModelState)
    model_state.max_model_len = 8
    previous_metadata = object()
    model_state.attn_metadata = previous_metadata
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=2,
        num_tokens_after_padding=2,
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens_np=np.array([2], dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.arange(2, dtype=torch.int64),
        attn_state=AscendAttentionState.PrefillNoCache,
    )

    with pytest.raises(RuntimeError, match="DSA metadata construction failed"):
        model_state.prepare_attn(
            input_batch=input_batch,
            cudagraph_mode=CUDAGraphMode.NONE,
            block_tables=tuple(torch.zeros((1, 2), dtype=torch.int32) for _ in attn_groups),
            slot_mappings=torch.zeros((len(attn_groups), 2), dtype=torch.int32),
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
        )

    assert model_state.attn_metadata is previous_metadata
