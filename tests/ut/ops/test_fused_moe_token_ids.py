from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEPrepareOutput
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods import w8a8_dynamic as w8a8_module
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.skipif(
    vllm_version_is("0.23.0"),
    reason="Fix8B targets the current vLLM 0.19.1 MoERunner ABI.",
)


def _runner() -> AscendMoERunner:
    runner = AscendMoERunner.__new__(AscendMoERunner)
    torch.nn.Module.__init__(runner)
    runner._sequence_parallel_context = lambda: nullcontext()
    return runner


@pytest.mark.parametrize("has_shared_experts", [False, True])
def test_forward_impl_preserves_token_identity_for_shared_and_no_shared_paths(
    has_shared_experts: bool,
) -> None:
    runner = _runner()
    input_ids = torch.tensor([17, 19], dtype=torch.int32)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 8)
    runner._shared_experts = MagicMock() if has_shared_experts else None
    runner.no_shared_forward_impl = MagicMock(return_value="routed")
    runner.shared_forward_impl = MagicMock(return_value=("shared", "routed"))

    result = AscendMoERunner._forward_impl(
        runner,
        hidden_states,
        router_logits,
        shared_experts_input=None,
        input_ids=input_ids,
    )

    if has_shared_experts:
        assert result == ("shared", "routed")
        assert runner.shared_forward_impl.call_args.kwargs["input_ids"] is input_ids
        runner.no_shared_forward_impl.assert_not_called()
    else:
        assert result == "routed"
        assert runner.no_shared_forward_impl.call_args.kwargs["input_ids"] is input_ids
        runner.shared_forward_impl.assert_not_called()


def test_shared_path_passes_same_ids_to_routed_experts() -> None:
    runner = _runner()
    input_ids = torch.tensor([23, 29], dtype=torch.int32)
    hidden_states = torch.randn(2, 4)
    router_logits = torch.randn(2, 8)
    runner._shared_experts = MagicMock()
    runner.shared_multistream_overlap_gate = False
    runner.gate = None
    runner.no_shared_forward_impl = MagicMock(
        return_value=FusedExpertsResult(routed_out=torch.ones_like(hidden_states))
    )
    runner._forward_shared_experts = MagicMock(return_value=torch.full_like(hidden_states, 2))

    shared, routed = AscendMoERunner.shared_forward_impl(
        runner,
        hidden_states,
        router_logits,
        input_ids=input_ids,
    )

    assert shared.shape == hidden_states.shape
    assert routed.shape == hidden_states.shape
    assert runner.no_shared_forward_impl.call_args.kwargs["input_ids"] is input_ids


def _configure_no_shared_runner(runner: AscendMoERunner, quant_method: MagicMock) -> MagicMock:
    runner.enable_npugraph_ex_static_kernel = False
    runner.multistream_overlap_gate = False
    runner.enable_shared_expert_dp = False
    runner.quant_type = QuantType.W8A8
    runner.top_k = 1
    runner.renormalize = False
    runner.use_grouped_topk = True
    runner.moe_config = SimpleNamespace(num_experts=8)
    runner._expert_map = None
    runner.topk_group = 1
    runner.num_expert_group = 1
    runner.custom_routing_function = None
    runner.scoring_func = "sqrtsoftplus"
    runner._original_routed_scaling_factor = 1.0
    runner.e_score_correction_bias = None
    runner.activation = "silu"
    runner.apply_router_weight_on_input = False
    runner.global_redundant_expert_num = 0
    runner.dynamic_eplb = False
    runner.routed_experts = SimpleNamespace(quant_method=quant_method)
    comm = MagicMock()
    comm.prepare.return_value = MoEPrepareOutput(
        hidden_states=torch.randn(2, 4),
        router_logits=torch.randn(2, 8),
        mc2_mask=None,
        padded_hidden_states_shape=None,
        pertoken_scale=None,
    )
    comm.finalize.side_effect = lambda hidden_states, **_: hidden_states
    return comm


def test_no_shared_path_passes_same_ids_to_quant_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    quant_method = MagicMock()
    quant_method.apply.return_value = FusedExpertsResult(routed_out=torch.randn(2, 4))
    comm = _configure_no_shared_runner(runner, quant_method)
    input_ids = torch.tensor([31, 37], dtype=torch.int32)
    monkeypatch.setattr(fused_moe_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            in_profile_run=False,
            moe_comm_method=comm,
            flash_comm_v1_enabled=False,
            eplb_heat_collection_status=False,
        ),
    )

    AscendMoERunner.no_shared_forward_impl(
        runner,
        torch.randn(2, 4),
        torch.randn(2, 8),
        input_ids=input_ids,
    )

    assert quant_method.apply.call_args.kwargs["input_ids"] is input_ids


def test_multistream_gate_uses_explicit_ids_in_both_routing_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    quant_method = MagicMock()
    quant_method.apply.return_value = FusedExpertsResult(routed_out=torch.randn(2, 4))
    comm = _configure_no_shared_runner(runner, quant_method)
    runner.multistream_overlap_gate = True
    input_ids = torch.tensor([41, 43], dtype=torch.int32)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    selector = MagicMock(return_value=(topk_weights, topk_ids))
    flash_context = SimpleNamespace(shared_experts=lambda value: value)
    monkeypatch.setattr(AscendMoERunner, "gate_stream", MagicMock())
    monkeypatch.setattr(fused_moe_module, "npu_stream_switch", lambda *_, **__: nullcontext())
    monkeypatch.setattr(fused_moe_module, "get_flash_common3_context", lambda: flash_context)
    monkeypatch.setattr(fused_moe_module, "set_flash_common3_context", MagicMock())
    monkeypatch.setattr(fused_moe_module, "shared_expert_dp_enabled", lambda: True)
    monkeypatch.setattr(fused_moe_module, "select_experts", selector)
    monkeypatch.setattr(fused_moe_module, "get_forward_context", lambda: SimpleNamespace())
    monkeypatch.setattr(
        fused_moe_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            in_profile_run=False,
            moe_comm_method=comm,
            moe_comm_type=MoECommType.MC2,
            flash_comm_v1_enabled=False,
            eplb_heat_collection_status=False,
        ),
    )

    AscendMoERunner.no_shared_forward_impl(
        runner,
        torch.randn(2, 4),
        torch.randn(2, 8),
        input_ids=input_ids,
    )

    assert selector.call_args.kwargs["input_ids"] is input_ids
    assert quant_method.apply.call_args.kwargs["input_ids"] is input_ids


@pytest.mark.parametrize("hash_routing", [False, True])
def test_modelslim_adapter_only_extends_hash_scheme_abi(hash_routing: bool) -> None:
    method = AscendFusedMoEMethod.__new__(AscendFusedMoEMethod)
    scheme = MagicMock()
    scheme.apply.return_value = "output"
    method.quant_method = scheme
    method.tid2eid = object() if hash_routing else None
    input_ids = torch.tensor([47, 53], dtype=torch.int32)

    result = AscendFusedMoEMethod.apply(
        method,
        layer=MagicMock(),
        x=torch.randn(2, 4),
        router_logits=torch.randn(2, 8),
        top_k=1,
        renormalize=False,
        input_ids=input_ids,
    )

    assert result == "output"
    if hash_routing:
        assert scheme.apply.call_args.kwargs["input_ids"] is input_ids
    else:
        assert "input_ids" not in scheme.apply.call_args.kwargs


def test_w8a8_scheme_passes_same_ids_to_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = AscendW8A8DynamicFusedMoEMethod.__new__(AscendW8A8DynamicFusedMoEMethod)
    method.multistream_overlap_gate = False
    method.dynamic_eplb = False
    method.in_dtype = torch.float32
    input_ids = torch.tensor([59, 61], dtype=torch.int32)
    hidden_states = torch.randn(2, 4)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    selector = MagicMock(return_value=(topk_weights, topk_ids))
    comm = MagicMock()
    comm.fused_experts.return_value = hidden_states.clone()
    layer = SimpleNamespace(
        w13_weight=torch.empty(8, 8, 4, dtype=torch.int8),
        w13_weight_scale_fp32=torch.ones(8, 8),
        w2_weight=torch.empty(8, 4, 4, dtype=torch.int8),
        w2_weight_scale=torch.ones(8, 4),
        n_shared_experts=0,
        mix_placement=False,
        zero_expert_num=0,
        zero_expert_type=None,
        swiglu_limit=0.0,
    )
    monkeypatch.setattr(w8a8_module, "select_experts", selector)
    monkeypatch.setattr(
        w8a8_module,
        "get_moe_num_logical_experts",
        lambda *_, **__: 8,
    )
    monkeypatch.setattr(
        w8a8_module,
        "_EXTRA_CTX",
        SimpleNamespace(
            moe_comm_method=comm,
            moe_comm_type=MoECommType.ALLGATHER,
        ),
    )

    AscendW8A8DynamicFusedMoEMethod.apply(
        method,
        layer=layer,
        x=hidden_states,
        router_logits=torch.randn(2, 8),
        top_k=1,
        renormalize=False,
        num_experts=8,
        tid2eid=torch.ones(64, 1, dtype=torch.int32),
        input_ids=input_ids,
    )

    assert selector.call_args.kwargs["input_ids"] is input_ids
