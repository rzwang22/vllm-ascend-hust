from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend import ascend_forward_context
from vllm_ascend.models import deepseek_v4 as deepseek_v4_module
from vllm_ascend.models.deepseek_v4 import (
    DeepseekV2DecoderLayer,
    DeepseekV4Model,
    DeepseekV4MoE,
)
from vllm_ascend.models.deepseek_v4_dspark import DeepseekV4DSparkModel
from vllm_ascend.models.deepseek_v4_mtp import DeepSeekMultiTokenPredictorLayer


def _empty_module(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


@pytest.mark.parametrize("internal_router", [False, True])
def test_deepseek_moe_passes_same_token_tensor_to_internal_and_external_router(
    internal_router: bool,
) -> None:
    moe = _empty_module(DeepseekV4MoE)
    input_ids = torch.tensor([11, 22], dtype=torch.int32)
    hidden_states = torch.randn(2, 4)
    experts = MagicMock(return_value=hidden_states.clone())
    experts.is_internal_router = internal_router
    moe.experts = experts
    moe.gate = SimpleNamespace(
        tid2eid=torch.ones(32, 1, dtype=torch.int32),
        weight=torch.randn(8, 4),
    )
    moe.is_sequence_parallel = False
    moe.shared_experts = None
    moe.routed_scaling_factor = 1.0
    moe.tp_size = 1

    output = DeepseekV4MoE.forward(moe, hidden_states, input_ids=input_ids)

    assert output.shape == hidden_states.shape
    assert experts.call_args.kwargs["input_ids"] is input_ids


def test_deepseek_hash_moe_fails_before_routing_without_explicit_ids() -> None:
    moe = _empty_module(DeepseekV4MoE)
    moe.gate = SimpleNamespace(tid2eid=torch.ones(32, 1, dtype=torch.int32))

    with pytest.raises(ValueError, match="hash MoE routing requires input_ids"):
        DeepseekV4MoE.forward(moe, torch.randn(2, 4), input_ids=None)


def test_decoder_passes_same_token_tensor_to_mlp() -> None:
    layer = _empty_module(DeepseekV2DecoderLayer)
    hidden_states = torch.randn(2, 1, 4)
    input_ids = torch.tensor([7, 9], dtype=torch.int32)
    layer.hc_pre = MagicMock(side_effect=lambda value, *_: (value, "post", "comb"))
    layer.hc_post = MagicMock(side_effect=lambda value, *_: value)
    layer.input_layernorm = MagicMock(side_effect=lambda value: value)
    layer.post_attention_layernorm = MagicMock(side_effect=lambda value: value)
    layer.self_attn = MagicMock(side_effect=lambda **kwargs: kwargs["hidden_states"])
    layer.mlp = MagicMock(side_effect=lambda value, **_: value)

    DeepseekV2DecoderLayer.forward(
        layer,
        positions=torch.arange(2),
        hidden_states=hidden_states,
        residual=None,
        input_ids=input_ids,
    )

    assert layer.mlp.call_args.kwargs["input_ids"] is input_ids


def test_target_model_passes_same_token_tensor_to_every_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _empty_module(DeepseekV4Model)
    input_ids = torch.tensor([101, 202], dtype=torch.int32)
    embedded = torch.randn(2, 4)
    decoder = MagicMock(side_effect=lambda *args, **_: (args[1], args[2]))
    decoder.layer_idx = 0
    model.embed_input_ids = MagicMock(return_value=embedded)
    model.hc_mult = 1
    model.layers = [decoder]
    model.start_layer = 0
    model.end_layer = 1
    model.aux_hidden_state_layers = ()
    model._mtp_hidden_buffer = torch.empty(2, 4)
    model.hc_head = MagicMock(side_effect=lambda value, *_: value.squeeze(1))
    model.hc_head_fn = None
    model.hc_head_scale = None
    model.hc_head_base = None
    model.norm = MagicMock(side_effect=lambda value: value)
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(deepseek_v4_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        ascend_forward_context,
        "_EXTRA_CTX",
        SimpleNamespace(flash_comm_v1_enabled=False),
    )

    DeepseekV4Model.forward(
        model,
        input_ids=input_ids,
        positions=torch.arange(2),
        intermediate_tensors=None,
    )

    assert decoder.call_args.kwargs["input_ids"] is input_ids


def test_mtp_block_receives_same_real_token_tensor() -> None:
    layer = _empty_module(DeepSeekMultiTokenPredictorLayer)
    input_ids = torch.tensor([3, 4], dtype=torch.int32)
    inputs_embeds = torch.randn(2, 4)
    previous_hidden_states = torch.randn(2, 4)
    layer.config = SimpleNamespace(hidden_size=4)
    layer.hc_mult = 1
    layer.enorm = MagicMock(side_effect=lambda value: value)
    layer.hnorm = MagicMock(side_effect=lambda value: value)
    layer.e_proj = MagicMock(side_effect=lambda value: value)
    layer.h_proj = MagicMock(side_effect=lambda value: value)
    layer.mtp_block = MagicMock(side_effect=lambda **kwargs: (kwargs["hidden_states"], None))

    DeepSeekMultiTokenPredictorLayer.forward(
        layer,
        input_ids=input_ids,
        positions=torch.arange(2),
        previous_hidden_states=previous_hidden_states,
        inputs_embeds=inputs_embeds,
    )

    assert layer.mtp_block.call_args.kwargs["input_ids"] is input_ids


def test_dspark_draft_passes_same_real_token_tensor_to_every_decoder() -> None:
    model = _empty_module(DeepseekV4DSparkModel)
    input_ids = torch.tensor([5, 6], dtype=torch.int32)
    embedded = torch.randn(2, 4)
    decoder = MagicMock(side_effect=lambda *args, **_: (args[1], args[2]))
    model.embed_tokens = MagicMock(return_value=embedded)
    model.hc_mult = 1
    model.layers = SimpleNamespace(values=lambda: [decoder])
    model.hc_head = MagicMock(side_effect=lambda value, *_: value.squeeze(1))
    model.hc_head_fn = None
    model.hc_head_scale = None
    model.hc_head_base = None

    DeepseekV4DSparkModel.forward(
        model,
        input_ids=input_ids,
        positions=torch.arange(2),
    )

    assert decoder.call_args.kwargs["input_ids"] is input_ids
