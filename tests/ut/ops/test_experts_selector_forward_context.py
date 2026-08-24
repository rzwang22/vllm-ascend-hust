# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import experts_selector


def test_hash_router_receives_explicit_model_input_ids(
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

    monkeypatch.setattr(
        torch.ops._C_ascend,
        "moe_gating_top_k_hash",
        fake_hash,
        raising=False,
    )
    monkeypatch.setattr(
        experts_selector,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_type=MoECommType.ALLGATHER,
            moe_comm_method=SimpleNamespace(
                prepare_finalize=SimpleNamespace(
                    all_gather_input_id_with_dp_group=lambda values: values,
                )
            ),
            flash_comm_v1_enabled=False,
        ),
    )

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
        input_ids=input_ids,
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


def test_non_hash_routing_does_not_require_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def fake_topk(*args, **kwargs):
        observed.update(kwargs)
        return (
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            torch.empty(0),
        )

    monkeypatch.setattr(
        experts_selector.DeviceOperator,
        "moe_gating_top_k",
        fake_topk,
    )

    weights, ids = experts_selector._select_experts_with_fusion_ops(
        hidden_states=torch.randn(2, 8),
        router_logits=torch.randn(2, 4),
        top_k=1,
        use_grouped_topk=True,
        renormalize=False,
        e_score_correction_bias=None,
        topk_group=1,
        num_expert_group=1,
        scoring_func="softmax",
        tid2eid=None,
        input_ids=None,
    )

    assert weights.shape == (2, 1)
    assert ids.shape == (2, 1)
    assert "input_ids" not in observed
