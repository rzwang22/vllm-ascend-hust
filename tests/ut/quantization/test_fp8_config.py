from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    MoERunner,
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.linear import LinearBase

from vllm_ascend.quantization import fp8_config
from vllm_ascend.quantization.method_adapters import (
    AscendFusedMoEMethod,
    AscendLinearMethod,
)
from vllm_ascend.quantization.methods.base import (
    AscendLinearScheme,
    AscendMoEScheme,
)
from vllm_ascend.quantization.methods.fp8 import (
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
)
from vllm_ascend.utils import FP8_METHOD

DEEPSEEK_V4_FP4_CONFIG = {
    "format": "float-quantized",
    "quant_method": "fp8",
    "expert_dtype": "fp4",
    "scale_fmt": "ue8m0",
    "group_size": 32,
    "weight_block_size": [128, 128],
}


def _fp8_config() -> fp8_config.AscendFp8Config:
    return fp8_config.AscendFp8Config.from_config(DEEPSEEK_V4_FP4_CONFIG)


def _moe_layer(layer_type):
    if layer_type is RoutedExperts:
        layer = object.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
    else:
        layer = MagicMock(spec=layer_type)
    layer.moe_config = MagicMock(spec=FusedMoEConfig)
    return layer


def test_routed_experts_is_recognized_as_fused_moe_layer() -> None:
    assert fp8_config._is_fused_moe_layer(_moe_layer(RoutedExperts))


def test_routed_experts_selects_ascend_w4a8_moe_method(monkeypatch) -> None:
    config = _fp8_config()
    layer = _moe_layer(RoutedExperts)
    scheme = MagicMock(spec=AscendMoEScheme)
    tid2eid = object()
    create_calls = []

    def create_scheme(quant_description, prefix, layer_type, packed_modules_mapping):
        create_calls.append(
            (
                quant_description,
                prefix,
                layer_type,
                packed_modules_mapping,
            )
        )
        return scheme

    monkeypatch.setattr(fp8_config, "create_scheme_for_layer", create_scheme)

    method = config.get_quant_method(
        layer,
        "model.layers.3.mlp.experts",
        tid2eid=tid2eid,
    )

    assert isinstance(method, AscendFusedMoEMethod)
    assert isinstance(method, FusedMoEMethodBase)
    assert not isinstance(method, UnquantizedFusedMoEMethod)
    assert method.quant_method is scheme
    assert method.tid2eid is tid2eid
    assert layer.ascend_quant_method == FP8_METHOD
    assert create_calls == [
        (
            config.quant_description,
            "model.layers.3.mlp.experts",
            "w4a8_moe",
            config.packed_modules_mapping,
        ),
    ]


def test_routed_experts_w4a8_method_allocates_packed_fp4_weights(
    monkeypatch,
) -> None:
    config = _fp8_config()
    routed_experts = _moe_layer(RoutedExperts)
    monkeypatch.setattr(
        "vllm_ascend.quantization.methods.w4a8_mxfp4.get_ep_group",
        lambda: object(),
    )
    monkeypatch.setattr(
        "vllm_ascend.quantization.methods.w4a8_mxfp4.get_current_vllm_config",
        lambda: SimpleNamespace(
            quant_config=SimpleNamespace(
                quant_description=DEEPSEEK_V4_FP4_CONFIG,
            ),
            compilation_config=SimpleNamespace(mode=None),
            model_config=SimpleNamespace(enforce_eager=True),
        ),
    )
    monkeypatch.setattr(
        "vllm_ascend.quantization.methods.w4a8_mxfp4.get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False),
        ),
    )
    method = config.get_quant_method(
        routed_experts,
        "model.layers.3.mlp.experts",
    )
    assert isinstance(method, AscendFusedMoEMethod)
    assert isinstance(
        method.quant_method,
        AscendW4A8MXFPDSDynamicFusedMoEMethod,
    )

    layer = torch.nn.Module()
    method.create_weights(
        layer,
        num_experts=1,
        hidden_size=2048,
        intermediate_size_per_partition=2048,
        params_dtype=torch.bfloat16,
    )

    assert layer.w13_weight.shape == (1, 4096, 1024)
    assert layer.w2_weight.shape == (1, 2048, 1024)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w2_weight.dtype == torch.uint8
    assert layer.w13_weight_scale.shape == (1, 4096, 64)
    assert layer.w2_weight_scale.shape == (1, 2048, 64)
    assert layer.w13_weight_scale.dtype == torch.float8_e8m0fnu
    assert layer.w2_weight_scale.dtype == torch.float8_e8m0fnu


def test_moe_detection_preserves_runner_and_legacy_fused_moe(
    monkeypatch,
) -> None:
    config = _fp8_config()
    scheme = MagicMock(spec=AscendMoEScheme)
    monkeypatch.setattr(
        fp8_config,
        "create_scheme_for_layer",
        lambda *_args, **_kwargs: scheme,
    )

    runner = _moe_layer(MoERunner)
    assert fp8_config._is_fused_moe_layer(runner)
    assert isinstance(
        config.get_quant_method(runner, "model.layers.1.mlp.experts"),
        AscendFusedMoEMethod,
    )

    class LegacyFusedMoE(torch.nn.Module):
        pass

    monkeypatch.setattr(fp8_config, "vllm_version_is", lambda _version: True)
    monkeypatch.setattr(fp8_config, "FusedMoE", LegacyFusedMoE, raising=False)

    legacy_layer = LegacyFusedMoE()
    legacy_layer.moe_config = MagicMock(spec=FusedMoEConfig)
    assert fp8_config._is_fused_moe_layer(legacy_layer)
    assert isinstance(
        config.get_quant_method(legacy_layer, "model.layers.2.mlp.experts"),
        AscendFusedMoEMethod,
    )


def test_linear_fp8_selection_is_unchanged(monkeypatch) -> None:
    config = _fp8_config()
    layer = MagicMock(spec=LinearBase)
    scheme = MagicMock(spec=AscendLinearScheme)
    create_calls = []

    def create_scheme(quant_description, prefix, layer_type, packed_modules_mapping):
        create_calls.append(
            (
                quant_description,
                prefix,
                layer_type,
                packed_modules_mapping,
            )
        )
        return scheme

    monkeypatch.setattr(fp8_config, "create_scheme_for_layer", create_scheme)
    monkeypatch.setattr(
        "vllm_ascend.quantization.method_adapters.enable_dsa_cp_with_layer_shard",
        lambda: False,
    )

    method = config.get_quant_method(layer, "model.layers.0.self_attn.q_proj")

    assert isinstance(method, AscendLinearMethod)
    assert method.quant_method is scheme
    assert layer.ascend_quant_method == FP8_METHOD
    assert create_calls == [
        (
            config.quant_description,
            "model.layers.0.self_attn.q_proj",
            "ds_linear",
            config.packed_modules_mapping,
        ),
    ]


def test_non_moe_non_linear_layer_remains_unquantized(monkeypatch) -> None:
    create_scheme = MagicMock()
    monkeypatch.setattr(fp8_config, "create_scheme_for_layer", create_scheme)

    assert _fp8_config().get_quant_method(torch.nn.Identity(), "model.norm") is None
    create_scheme.assert_not_called()


def test_deepseek_v4_fp8_fp4_contract_is_preserved_by_parser() -> None:
    config = _fp8_config()

    assert config.get_name() == FP8_METHOD
    assert config.quant_format == "float-quantized"
    assert config.quant_description == DEEPSEEK_V4_FP4_CONFIG
    assert config.quant_description["quant_method"] == "fp8"
    assert config.quant_description["expert_dtype"] == "fp4"
    assert config.quant_description["scale_fmt"] == "ue8m0"
