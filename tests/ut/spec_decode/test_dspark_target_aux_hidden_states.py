# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.models.interfaces import supports_eagle, supports_eagle3
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)

CHECKPOINT_LAYER_IDS = [40, 41, 42]
OUTPUT_BOUNDARIES = (41, 42, 43)
NUM_TARGET_LAYERS = 61


def _deepseek_v4_module():
    return importlib.import_module("vllm_ascend.models.deepseek_v4")


def _target_wrapper():
    model_module = _deepseek_v4_module()
    target = object.__new__(model_module.AscendDeepseekV4ForCausalLM)
    nn.Module.__init__(target)
    target.config = SimpleNamespace(num_hidden_layers=NUM_TARGET_LAYERS)

    backbone = object.__new__(model_module.DeepseekV4Model)
    nn.Module.__init__(backbone)
    backbone.start_layer = 0
    backbone.end_layer = NUM_TARGET_LAYERS
    target.model = backbone
    return target


def _dspark_spec_config():
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                dspark_target_layer_ids=list(CHECKPOINT_LAYER_IDS),
            ),
        ),
    )


class _BoundaryLayer(nn.Module):
    def __init__(self, layer_idx: int, increment: float) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.increment = increment

    def forward(
        self,
        _positions: torch.Tensor,
        hidden_states: torch.Tensor,
        _residual: torch.Tensor | None,
        _llama_4_scaling: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = hidden_states + self.increment
        return output, hidden_states


def _forward_backbone(*, capture_boundaries: tuple[int, ...]):
    model_module = _deepseek_v4_module()
    backbone = object.__new__(model_module.DeepseekV4Model)
    nn.Module.__init__(backbone)
    backbone.hc_mult = 2
    backbone.start_layer = 0
    backbone.end_layer = NUM_TARGET_LAYERS
    backbone.layers = nn.ModuleList(
        [
            _BoundaryLayer(40, 1.0),
            _BoundaryLayer(41, 2.0),
            _BoundaryLayer(42, 3.0),
        ]
    )
    backbone._set_aux_hidden_state_layers(capture_boundaries)
    backbone._mtp_hidden_buffer = torch.empty(2, 8)
    backbone.hc_head_fn = None
    backbone.hc_head_scale = None
    backbone.hc_head_base = None
    backbone.hc_head = lambda hidden_states, *_args: hidden_states.mean(dim=1)
    backbone.norm = nn.Identity()
    return backbone


def _target_forward_context() -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        additional_kwargs={
            "flash_comm_v1_enabled": False,
            "pad_size": 0,
        },
    )


def test_ascend_deepseek_v4_satisfies_real_eagle3_interface() -> None:
    from vllm.model_executor.models.interfaces import EagleModelMixin

    model_module = _deepseek_v4_module()
    target = _target_wrapper()

    assert issubclass(model_module.DeepseekV4Model, EagleModelMixin)
    assert supports_eagle(target)
    assert supports_eagle3(target)
    assert callable(target.set_aux_hidden_state_layers)
    assert callable(target.get_eagle3_default_aux_hidden_state_layers)
    assert target.model.aux_hidden_state_layers == ()


def test_core_dspark_layer_ids_configure_decoder_output_boundaries() -> None:
    target = _target_wrapper()

    set_eagle3_aux_hidden_state_layers(target, _dspark_spec_config())

    assert CHECKPOINT_LAYER_IDS == [40, 41, 42]
    assert target.model.aux_hidden_state_layers == OUTPUT_BOUNDARIES


@pytest.mark.parametrize(
    ("layers", "error_type", "error_match"),
    [
        ((), ValueError, "At least one"),
        ((41, 41), ValueError, "unique"),
        ((42, 41), ValueError, "strictly increasing"),
        ((0,), ValueError, "one-based"),
        ((NUM_TARGET_LAYERS + 1,), ValueError, "one-based"),
        ((True,), TypeError, "integers"),
        ([41], TypeError, "tuple"),
    ],
)
def test_aux_hidden_state_layer_validation(
    layers,
    error_type: type[Exception],
    error_match: str,
) -> None:
    target = _target_wrapper()

    with pytest.raises(error_type, match=error_match):
        target.set_aux_hidden_state_layers(layers)
    assert target.model.aux_hidden_state_layers == ()


def test_aux_hidden_state_layers_require_pipeline_parallel_size_one() -> None:
    target = _target_wrapper()
    target.model.start_layer = 30
    target.model.end_layer = NUM_TARGET_LAYERS

    with pytest.raises(NotImplementedError, match="pipeline parallel size 1"):
        target.set_aux_hidden_state_layers(OUTPUT_BOUNDARIES)


def test_target_forward_returns_requested_real_aux_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = _deepseek_v4_module()
    forward_context = importlib.import_module(
        "vllm_ascend.ascend_forward_context",
    )
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        forward_context.envs_vllm,
        "VLLM_USE_V2_MODEL_RUNNER",
        True,
    )
    backbone = _forward_backbone(capture_boundaries=OUTPUT_BOUNDARIES)
    inputs_embeds = torch.zeros(2, 4)
    positions = torch.arange(2)

    with override_forward_context(_target_forward_context()):
        output = backbone.forward(
            input_ids=torch.zeros(2, dtype=torch.long),
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )

    assert isinstance(output, tuple)
    final_hidden_states, aux_hidden_states = output
    assert final_hidden_states.shape == (2, 4)
    assert len(aux_hidden_states) == 3
    assert all(aux.shape == (2, 4) for aux in aux_hidden_states)
    assert torch.equal(aux_hidden_states[0], torch.full((2, 4), 1.0))
    assert torch.equal(aux_hidden_states[1], torch.full((2, 4), 3.0))
    assert torch.equal(aux_hidden_states[2], torch.full((2, 4), 6.0))
    assert torch.equal(final_hidden_states, aux_hidden_states[-1])


def test_non_dspark_forward_shape_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = _deepseek_v4_module()
    forward_context = importlib.import_module(
        "vllm_ascend.ascend_forward_context",
    )
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        forward_context.envs_vllm,
        "VLLM_USE_V2_MODEL_RUNNER",
        True,
    )
    backbone = _forward_backbone(capture_boundaries=())

    with override_forward_context(_target_forward_context()):
        output = backbone.forward(
            input_ids=torch.zeros(2, dtype=torch.long),
            positions=torch.arange(2),
            intermediate_tensors=None,
            inputs_embeds=torch.zeros(2, 4),
        )

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 4)


def test_target_aux_interface_does_not_import_nvidia_runtime() -> None:
    conftest_path = Path(__file__).parents[1] / "conftest.py"
    child_script = textwrap.dedent(
        f"""
        import json
        import runpy
        import sys
        import types

        build_info = types.ModuleType("vllm_ascend._build_info")
        build_info.__device_type__ = "A2"
        build_info.__soc_version__ = "ASCEND910B2"
        sys.modules["vllm_ascend._build_info"] = build_info
        runpy.run_path({str(conftest_path)!r})

        before = set(sys.modules)
        assert "vllm_ascend.models.deepseek_v4" not in before

        from vllm.model_executor.models.interfaces import supports_eagle3
        from vllm_ascend.models.deepseek_v4 import (
            AscendDeepseekV4ForCausalLM,
        )

        new_modules = set(sys.modules) - before
        forbidden_prefixes = (
            "vllm.models.deepseek_v4.nvidia",
            "vllm.v1.worker.gpu.spec_decode.dspark",
            "vllm_ascend.ops.triton.spec_decode",
        )

        def referenced_module(value):
            if isinstance(value, types.ModuleType):
                return value.__name__
            return getattr(value, "__module__", "")

        model_module = sys.modules[AscendDeepseekV4ForCausalLM.__module__]
        payload = {{
            "supports_eagle3": supports_eagle3(AscendDeepseekV4ForCausalLM),
            "new_forbidden": sorted(
                name for name in new_modules if name.startswith(forbidden_prefixes)
            ),
            "mro_modules": [base.__module__ for base in AscendDeepseekV4ForCausalLM.__mro__],
            "nvidia_globals": sorted(
                name
                for name, value in vars(model_module).items()
                if referenced_module(value).startswith("vllm.models.deepseek_v4.nvidia")
            ),
        }}
        print("DSPARK_TARGET_AUX_IMPORT_AUDIT=" + json.dumps(payload, sort_keys=True))
        """
    )
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(entry for entry in sys.path if entry)
    result = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=Path(__file__).parents[3],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    audit_line = next(line for line in result.stdout.splitlines() if line.startswith("DSPARK_TARGET_AUX_IMPORT_AUDIT="))
    audit = json.loads(
        audit_line.removeprefix("DSPARK_TARGET_AUX_IMPORT_AUDIT="),
    )
    assert audit["supports_eagle3"]
    assert audit["new_forbidden"] == []
    assert all(not module.startswith("vllm.models.deepseek_v4.nvidia") for module in audit["mro_modules"])
    assert audit["nvidia_globals"] == []
