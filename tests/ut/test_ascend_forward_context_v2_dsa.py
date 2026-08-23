# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.forward_context as core_forward_context
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext

from vllm_ascend import ascend_forward_context
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.platform import NPUPlatform

CORE_FORWARD_CONTEXT_FIELDS = {
    "no_compile_layers",
    "attn_metadata",
    "slot_mapping",
    "dp_metadata",
    "cudagraph_runtime_mode",
    "batch_descriptor",
    "ubatch_slices",
    "is_padding",
    "skip_compiled",
    "all_moe_layers",
    "moe_layer_index",
    "additional_kwargs",
}
V2_PLATFORM_EXTRA_FIELDS = {
    "capturing",
    "flash_comm_v1_enabled",
    "flashcomm_v2_enabled",
    "in_profile_run",
    "input_ids",
    "is_draft_model",
    "is_draft_model_prefill",
    "max_tokens_across_dp",
    "mc2_mask",
    "mmrs_fusion",
    "model_instance",
    "moe_comm_method",
    "moe_comm_type",
    "num_tokens",
    "pad_size",
    "padded_length",
    "padded_num_tokens",
    "sinks",
}


def _vllm_config(
    *,
    speculative_method: str | None = "dspark",
    model_type: str = "deepseek_v4",
):
    speculative_config = None if speculative_method is None else SimpleNamespace(method=speculative_method)
    return SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_forward_context={},
            static_all_moe_layers=[],
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            is_moe_model=True,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type),
        ),
        speculative_config=speculative_config,
    )


def _model_with_layer_range() -> torch.nn.Module:
    model = torch.nn.Module()
    model.model = SimpleNamespace(start_layer=0)
    return model


@contextmanager
def _npu_platform_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    flashcomm1: bool | list[bool] = False,
    flashcomm2: bool | list[bool] = False,
    is_moe: bool = True,
    tp_size: int = 8,
    mc2_masks: torch.Tensor | list[torch.Tensor] | None = None,
):
    monkeypatch.setattr(
        ascend_forward_context.envs_vllm,
        "VLLM_USE_V2_MODEL_RUNNER",
        True,
    )
    monkeypatch.setattr(
        core_forward_context,
        "current_platform",
        NPUPlatform,
    )

    def mock_options(value):
        if isinstance(value, list):
            return {"side_effect": value}
        return {"return_value": value}

    comm_method = object()
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "vllm_ascend.ascend_forward_context.is_moe_model",
                return_value=is_moe,
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.ascend_forward_context.enable_sp",
                **mock_options(flashcomm1),
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.ascend_forward_context.flashcomm2_enable",
                **mock_options(flashcomm2),
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.ascend_forward_context.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.ascend_forward_context.get_dp_group",
                return_value=SimpleNamespace(world_size=1),
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.ascend_forward_context.select_moe_comm_method",
                return_value=(MoECommType.ALLGATHER if is_moe else None),
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method",
                return_value=comm_method,
            )
        )
        stack.enter_context(
            patch(
                "vllm_ascend.ascend_forward_context.get_mc2_mask",
                **mock_options(mc2_masks),
            )
        )
        yield comm_method


def _set_forward_context(
    vllm_config,
    *,
    attn_metadata=None,
    num_tokens: int = 5,
    input_ids: torch.Tensor | None = None,
    model_instance: torch.nn.Module | None = None,
):
    if attn_metadata is None:
        attn_metadata = {"model.layers.0.self_attn": object()}
    return core_forward_context.set_forward_context(
        attn_metadata,
        vllm_config,
        num_tokens=num_tokens,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping={"model.layers.0.self_attn": torch.arange(num_tokens)},
        input_ids=input_ids,
        model_instance=model_instance,
    )


def test_current_core_forward_context_uses_additional_kwargs_for_platform_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert {field.name for field in fields(ForwardContext)} == (CORE_FORWARD_CONTEXT_FIELDS)
    assert not hasattr(ForwardContext, "__slots__")

    with _npu_platform_runtime(monkeypatch), _set_forward_context(_vllm_config()):
        context = core_forward_context.get_forward_context()
        assert not hasattr(context, "flash_comm_v1_enabled")
        assert context.additional_kwargs.keys() >= V2_PLATFORM_EXTRA_FIELDS
        assert _EXTRA_CTX.flash_comm_v1_enabled is False
        assert _EXTRA_CTX.flashcomm_v2_enabled is False


def test_v2_context_bridges_current_model_input_identity_and_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = torch.tensor([101, 202, -1], dtype=torch.int32)
    model = _model_with_layer_range()

    with (
        _npu_platform_runtime(monkeypatch),
        _set_forward_context(
            _vllm_config(),
            num_tokens=3,
            input_ids=input_ids,
            model_instance=model,
        ),
    ):
        context = core_forward_context.get_forward_context()
        assert context.additional_kwargs["input_ids"] is input_ids
        assert context.additional_kwargs["model_instance"] is model
        assert _EXTRA_CTX.input_ids is input_ids
        assert _EXTRA_CTX.model_instance is model

    with pytest.raises(AssertionError, match="Forward context is not set"):
        core_forward_context.get_forward_context()


def test_v2_context_rejects_input_ids_that_exclude_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _npu_platform_runtime(monkeypatch),
        pytest.raises(ValueError, match="include every padded model-input token"),
        _set_forward_context(
            _vllm_config(),
            num_tokens=4,
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
        ),
    ):
        pass


def test_v1_shared_builder_preserves_direct_context_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_ids = torch.tensor([7, 8], dtype=torch.int32)
    model = _model_with_layer_range()
    vllm_config = _vllm_config()
    vllm_config.use_v2_model_runner = False
    with _npu_platform_runtime(monkeypatch):
        monkeypatch.setattr(ascend_forward_context.envs_vllm, "VLLM_USE_V2_MODEL_RUNNER", False)
        with ascend_forward_context.set_ascend_forward_context(
            None,
            vllm_config,
            num_tokens=2,
            input_ids=input_ids,
            model_instance=model,
        ):
            context = core_forward_context.get_forward_context()
            assert context.input_ids is input_ids
            assert context.model_instance is model
            assert context.moe_comm_type is MoECommType.ALLGATHER


@pytest.mark.parametrize(
    ("flashcomm1", "flashcomm2", "expected_pad_size"),
    [
        (False, False, 0),
        (True, False, 3),
        (False, True, 3),
        (True, True, 3),
    ],
)
def test_platform_normalizes_flashcomm_runtime_flags(
    monkeypatch: pytest.MonkeyPatch,
    flashcomm1: bool,
    flashcomm2: bool,
    expected_pad_size: int,
) -> None:
    with (
        _npu_platform_runtime(
            monkeypatch,
            flashcomm1=flashcomm1,
            flashcomm2=flashcomm2,
        ),
        _set_forward_context(_vllm_config(), num_tokens=5),
    ):
        context = core_forward_context.get_forward_context()
        assert context.additional_kwargs["flash_comm_v1_enabled"] is (flashcomm1)
        assert context.additional_kwargs["flashcomm_v2_enabled"] is (flashcomm2)
        assert context.additional_kwargs["pad_size"] == expected_pad_size
        assert _EXTRA_CTX.flash_comm_v1_enabled is flashcomm1
        assert _EXTRA_CTX.flashcomm_v2_enabled is flashcomm2


def test_dsa_wrapper_reads_flashcomm1_from_platform_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.ops import dsa

    observed_flags: list[bool] = []

    def fake_dsa_forward(hidden_states, need_gather_q_kv, output, layer_name):
        del layer_name
        observed_flags.append(need_gather_q_kv)
        output.copy_(hidden_states)

    monkeypatch.setattr(
        torch.ops.vllm,
        "dsa_forward",
        fake_dsa_forward,
    )
    wrapper = SimpleNamespace(prefix="model.layers.0.self_attn")
    hidden_states = torch.randn(2, 4)

    with _npu_platform_runtime(
        monkeypatch,
        flashcomm1=[False, True],
    ):
        for _ in range(2):
            with _set_forward_context(_vllm_config(), num_tokens=2):
                output = dsa.AscendDeepseekSparseAttention.forward(
                    wrapper,
                    positions=torch.arange(2),
                    hidden_states=hidden_states,
                )
                assert torch.equal(output, hidden_states)

    assert observed_flags == [False, True]


def test_dsa_impl_reads_num_tokens_from_platform_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.attention import dsa_v1

    impl = SimpleNamespace(n_local_heads=1, head_dim=4)
    hidden_states = torch.ones(2, 4)
    output = torch.ones_like(hidden_states)
    monkeypatch.setattr(dsa_v1, "oproj_tp_enable", lambda: False)

    with _npu_platform_runtime(monkeypatch), _set_forward_context(_vllm_config(), num_tokens=2):
        result = dsa_v1.AscendDSAImpl.forward(
            impl,
            "model.layers.0.self_attn",
            hidden_states,
            None,
            None,
            output=output,
        )

    assert result is output
    assert torch.count_nonzero(output) == 0


def test_dsa_custom_op_resolves_real_metadata_and_bound_cache_from_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.ops import dsa

    layer_name = "model.layers.0.self_attn.wrapper"
    prefix = "model.layers.0.self_attn"
    metadata = object()
    kv_cache = object()
    calls = []

    class FakeImpl:
        def forward(self, *args):
            calls.append(args)

    layer = SimpleNamespace(
        prefix=prefix,
        dsa_attn=SimpleNamespace(
            layer_name=f"{prefix}.dsa",
            impl=FakeImpl(),
        ),
    )
    context = ForwardContext(
        no_compile_layers={layer_name: layer},
        attn_metadata={f"{prefix}.c1": metadata},
        slot_mapping={},
        additional_kwargs={"flash_comm_v1_enabled": False},
    )
    monkeypatch.setattr(dsa, "_build_kv_cache", lambda *_args: kv_cache)
    hidden_states = torch.randn(2, 4)
    output = torch.empty_like(hidden_states)

    with core_forward_context.override_forward_context(context):
        dsa.dsa_forward(
            hidden_states,
            False,
            output,
            layer_name,
        )

    assert len(calls) == 1
    assert calls[0][2] is kv_cache
    assert calls[0][3] == [metadata]


def test_nested_and_exceptional_forward_contexts_restore_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        _npu_platform_runtime(
            monkeypatch,
            flashcomm1=[False, True],
            flashcomm2=[False, True],
        ),
        _set_forward_context(_vllm_config(), num_tokens=4),
    ):
        outer = core_forward_context.get_forward_context()
        assert _EXTRA_CTX.flash_comm_v1_enabled is False
        with (
            pytest.raises(RuntimeError, match="nested failure"),
            _set_forward_context(_vllm_config(), num_tokens=8),
        ):
            inner = core_forward_context.get_forward_context()
            assert inner is not outer
            assert _EXTRA_CTX.flash_comm_v1_enabled is True
            assert _EXTRA_CTX.flashcomm_v2_enabled is True
            raise RuntimeError("nested failure")
        assert core_forward_context.get_forward_context() is outer
        assert _EXTRA_CTX.flash_comm_v1_enabled is False

    with pytest.raises(AssertionError, match="Forward context is not set"):
        core_forward_context.get_forward_context()


def test_consecutive_steps_do_not_reuse_flashcomm_or_mc2_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_mask = torch.zeros(8, dtype=torch.bool)
    second_mask = torch.zeros(8, dtype=torch.bool)
    observed = []

    with _npu_platform_runtime(
        monkeypatch,
        flashcomm1=[True, False],
        mc2_masks=[first_mask, second_mask],
    ):
        for num_tokens in (3, 5):
            with _set_forward_context(_vllm_config(), num_tokens=num_tokens):
                context = core_forward_context.get_forward_context()
                observed.append(
                    (
                        context.additional_kwargs,
                        _EXTRA_CTX.flash_comm_v1_enabled,
                        _EXTRA_CTX.mc2_mask,
                    )
                )

    assert observed[0][0] is not observed[1][0]
    assert [step[1] for step in observed] == [True, False]
    assert observed[0][2].data_ptr() != observed[1][2].data_ptr()
    assert observed[0][2].tolist() == [True, True, True, False, False, False, False, False]
    assert observed[1][2].tolist() == [True, True, True, True, True, False, False, False]


@pytest.mark.parametrize(
    ("speculative_method", "model_type"),
    [
        ("dspark", "deepseek_v4"),
        (None, "deepseek_v4"),
        (None, "qwen2"),
    ],
)
def test_forward_context_provider_is_model_and_speculator_neutral(
    monkeypatch: pytest.MonkeyPatch,
    speculative_method: str | None,
    model_type: str,
) -> None:
    config = _vllm_config(
        speculative_method=speculative_method,
        model_type=model_type,
    )
    metadata = {"attention": object()}
    with (
        _npu_platform_runtime(monkeypatch, flashcomm1=False),
        _set_forward_context(
            config,
            attn_metadata=metadata,
            num_tokens=2,
        ),
    ):
        context = core_forward_context.get_forward_context()
        assert context.attn_metadata is metadata
        assert _EXTRA_CTX.flash_comm_v1_enabled is False
        assert config.model_config.hf_text_config.model_type == model_type
        assert getattr(config.speculative_config, "method", None) == speculative_method


def test_all_dsa_runtime_extras_have_platform_and_proxy_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(ascend_forward_context._ExtraForwardContextProxy.extra_attrs) >= V2_PLATFORM_EXTRA_FIELDS
    with _npu_platform_runtime(monkeypatch), _set_forward_context(_vllm_config()):
        assert (core_forward_context.get_forward_context().additional_kwargs.keys()) >= V2_PLATFORM_EXTRA_FIELDS

    repository_root = Path(__file__).parents[2]
    consumers = {
        repository_root / "vllm_ascend/ops/dsa.py": {
            "flash_comm_v1_enabled",
        },
        repository_root / "vllm_ascend/attention/dsa_v1.py": {
            "num_tokens",
        },
        repository_root / "vllm_ascend/models/deepseek_v4.py": {
            "flash_comm_v1_enabled",
            "pad_size",
        },
    }
    for path, extra_fields in consumers.items():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in extra_fields:
                continue
            assert isinstance(node.value, ast.Name) and node.value.id == "_EXTRA_CTX", (
                f"{path}:{node.lineno} bypasses _EXTRA_CTX for {node.attr}"
            )
