# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
import vllm.forward_context as core_forward_context
from vllm.forward_context import ForwardContext

from vllm_ascend import ascend_forward_context
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import experts_selector


def _context(input_ids: torch.Tensor) -> ForwardContext:
    prepare_finalize = SimpleNamespace(
        all_gather_input_id_with_dp_group=lambda values: values,
    )
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={
            "input_ids": input_ids,
            "moe_comm_type": MoECommType.ALLGATHER,
            "moe_comm_method": SimpleNamespace(prepare_finalize=prepare_finalize),
            "flash_comm_v1_enabled": False,
        },
    )


def test_hash_router_receives_real_context_input_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = torch.tensor([31, -1], dtype=torch.int32)
    router_logits = torch.randn(2, 4)
    observed = {}

    def fake_hash(**kwargs):
        observed.update(kwargs)
        return (
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            torch.empty(0),
        )

    monkeypatch.setattr(ascend_forward_context.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "moe_gating_top_k_hash",
        fake_hash,
        raising=False,
    )

    with core_forward_context.override_forward_context(_context(input_ids)):
        experts_selector._select_experts_with_fusion_ops(
            hidden_states=torch.randn(2, 8),
            router_logits=router_logits,
            top_k=1,
            use_grouped_topk=True,
            renormalize=False,
            e_score_correction_bias=None,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sqrtsoftplus",
            tid2eid=torch.ones(4, 1, dtype=torch.int32),
            input_ids=ascend_forward_context._EXTRA_CTX.input_ids,
        )

    assert observed["input_ids"].dtype is torch.int64
    assert observed["input_ids"].tolist() == [31, 0]


@pytest.mark.parametrize(
    "input_ids",
    [
        None,
        torch.tensor([[1, 2]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    ],
)
def test_hash_router_fails_closed_for_missing_or_misaligned_ids(input_ids) -> None:
    with pytest.raises(ValueError, match="requires input_ids|must match"):
        experts_selector._select_experts_with_fusion_ops(
            hidden_states=torch.randn(2, 8),
            router_logits=torch.randn(2, 4),
            top_k=1,
            use_grouped_topk=True,
            renormalize=False,
            e_score_correction_bias=None,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sqrtsoftplus",
            tid2eid=torch.ones(4, 1, dtype=torch.int32),
            input_ids=input_ids,
        )
