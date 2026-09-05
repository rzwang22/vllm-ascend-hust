# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Execute loader config logic without Torch; real VllmConfig UTs live in spec_decode.

Run with --noconftest on hosts without Torch. Only imports/runtime dependencies
are stubbed; the builder, validator and loader bodies come from production AST.
"""

import ast
import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parents[2]
LOADER = REPO_ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/model_loader.py"


class _LiveLayer:
    def __deepcopy__(self, memo):
        raise AssertionError("Live target layers must not be copied")


class _QuantConfig:
    pass


def _config(target_eager=True, draft_override=None):
    target = SimpleNamespace(enforce_eager=target_eager, hf_overrides={})
    return SimpleNamespace(
        model_config=target,
        speculative_config=SimpleNamespace(
            enforce_eager=draft_override,
            target_model_config=target,
            draft_model_config=SimpleNamespace(
                enforce_eager=target_eager,
                hf_overrides=lambda config: config,
                hf_config=SimpleNamespace(dspark_block_size=5, dspark_target_layer_ids=[40, 41, 42]),
            ),
            attention_backend="ASCEND",
            num_speculative_tokens=5,
        ),
        compilation_config=SimpleNamespace(
            mode="NONE" if target_eager else "COMPILE",
            cudagraph_mode="NONE" if target_eager else "FULL_DECODE_ONLY",
            cudagraph_capture_sizes=[] if target_eager else [6],
            max_cudagraph_capture_size=0 if target_eager else 6,
            pass_config=SimpleNamespace(enable_sp=False),
            static_forward_context={"target": _LiveLayer()},
            static_all_moe_layers=["target.moe"],
        ),
        attention_config=SimpleNamespace(use_non_causal=False, backend=None),
        parallel_config=SimpleNamespace(tensor_parallel_size=8, enable_expert_parallel=True, pipeline_parallel_size=1),
        cache_config=SimpleNamespace(block_size=32),
    )


def _namespace_replace(value, **changes):
    result = copy.copy(value)
    vars(result).update(changes)
    return result


@pytest.fixture
def loader(monkeypatch):
    names = {"_build_draft_vllm_config", "_validate_w8a8_runtime_contract", "load_dspark_model"}
    tree = ast.parse(LOADER.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *selected], type_ignores=[])
    namespace = {
        "copy": copy,
        "replace": _namespace_replace,
        "CompilationMode": SimpleNamespace(NONE="NONE"),
        "CUDAGraphMode": SimpleNamespace(NONE="NONE"),
        "AttentionBackendEnum": SimpleNamespace(FLASH_ATTN="FLASH_ATTN"),
        "_DSPARK_BLOCK_SIZE": 5,
        "_DSPARK_TARGET_LAYER_IDS": (40, 41, 42),
        "_build_draft_quant_config": lambda *_args: _QuantConfig(),
        "get_pp_group": lambda: SimpleNamespace(world_size=1),
        "_should_share": lambda *_args: False,
    }
    exec(compile(module, str(LOADER), "exec"), namespace)
    quant_module = ModuleType("vllm_ascend.quantization.modelslim_config")
    quant_module.AscendModelSlimConfig = _QuantConfig
    monkeypatch.setitem(sys.modules, quant_module.__name__, quant_module)
    backends = ModuleType("vllm.compilation.backends")
    backends.set_model_tag = lambda _tag: nullcontext()
    monkeypatch.setitem(sys.modules, backends.__name__, backends)
    return namespace


@pytest.mark.parametrize("target_eager,draft_override", [(True, None), (True, True), (False, True)])
def test_loader_constructs_eager_draft_without_mutating_target(loader, target_eager, draft_override):
    config = _config(target_eager, draft_override)
    context = config.compilation_config.static_forward_context
    moe_layers = config.compilation_config.static_all_moe_layers
    calls = []

    def get_model(*, vllm_config, model_config):
        calls.append(vllm_config)
        assert model_config is vllm_config.speculative_config.draft_model_config
        assert model_config is not config.speculative_config.draft_model_config
        assert model_config.enforce_eager and vllm_config.model_config.enforce_eager
        assert callable(model_config.hf_overrides)
        assert vllm_config.model_config.hf_overrides == {}
        assert vllm_config.compilation_config.mode == "NONE"
        assert vllm_config.compilation_config.cudagraph_mode == "NONE"
        assert vllm_config.compilation_config.cudagraph_capture_sizes == []
        assert vllm_config.compilation_config.max_cudagraph_capture_size == 0
        assert vllm_config.attention_config.use_non_causal
        assert vllm_config.compilation_config.static_forward_context is context
        assert vllm_config.compilation_config.static_all_moe_layers is moe_layers
        context["draft"] = _LiveLayer()
        moe_layers.append("draft.moe")
        return SimpleNamespace(model=SimpleNamespace(), vllm_config=vllm_config)

    loader["get_model"] = get_model
    model = loader["load_dspark_model"](SimpleNamespace(model=SimpleNamespace()), config)
    assert len(calls) == 1
    assert model.vllm_config is calls[0]
    assert config.model_config.enforce_eager is target_eager
    assert config.speculative_config.draft_model_config.enforce_eager is target_eager
    assert config.speculative_config.enforce_eager is draft_override
    assert config.compilation_config.cudagraph_mode == ("NONE" if target_eager else "FULL_DECODE_ONLY")
    assert config.compilation_config.cudagraph_capture_sizes == ([] if target_eager else [6])
    assert config.compilation_config.max_cudagraph_capture_size == (0 if target_eager else 6)
    assert not config.attention_config.use_non_causal
    assert "draft" in context and "draft.moe" in moe_layers


@pytest.mark.parametrize("target_eager,draft_override", [(True, False), (False, False), (False, None)])
def test_unsupported_draft_graph_fails_before_construction(loader, target_eager, draft_override):
    config = _config(target_eager, draft_override)
    loader["get_model"] = lambda **_kwargs: pytest.fail("must reject before get_model")
    with pytest.raises(ValueError, match="draft graph execution is unsupported.*speculative_config.enforce_eager=True"):
        loader["load_dspark_model"](SimpleNamespace(), config)
    assert config.model_config.enforce_eager is target_eager
    assert config.speculative_config.enforce_eager is draft_override


def test_construction_failure_cannot_pollute_target(loader):
    config = _config(False, True)

    def failing_get_model(*, vllm_config, model_config):
        vllm_config.compilation_config.pass_config.enable_sp = True
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.cache_config.block_size = 1
        vllm_config.model_config.hf_overrides["mutated"] = True
        raise ValueError("construction failure")

    loader["get_model"] = failing_get_model
    with pytest.raises(ValueError, match="construction failure"):
        loader["load_dspark_model"](SimpleNamespace(), config)
    assert not config.model_config.enforce_eager
    assert config.model_config.hf_overrides == {}
    assert not config.speculative_config.draft_model_config.enforce_eager
    assert config.compilation_config.cudagraph_mode == "FULL_DECODE_ONLY"
    assert config.compilation_config.cudagraph_capture_sizes == [6]
    assert not config.compilation_config.pass_config.enable_sp
    assert config.parallel_config.tensor_parallel_size == 8
    assert config.cache_config.block_size == 32


@pytest.mark.parametrize("backend,expected", [(None, None), ("CUSTOM", None), ("FLASH_ATTN", "FLASH_ATTN")])
def test_builder_normalizes_backend_without_revalidating_vllm_config(loader, backend, expected):
    config = _config(False, True)
    config.speculative_config.attention_backend = backend

    def replace_attention_only(value, **changes):
        assert value is not config.attention_config
        assert not hasattr(value, "model_config"), "must not reinitialize process-wide Ascend config"
        return _namespace_replace(value, **changes)

    loader["replace"] = replace_attention_only
    draft = loader["_build_draft_vllm_config"](config, _QuantConfig())
    assert draft.attention_config.backend == expected
    assert draft.attention_config.use_non_causal
    assert not config.attention_config.use_non_causal


@pytest.mark.parametrize(
    "section,field,value,error",
    [
        ("model_config", "enforce_eager", False, "eager draft runtime"),
        ("compilation_config", "mode", "COMPILE", "compilation and graphs"),
        ("compilation_config", "cudagraph_mode", "FULL", "compilation and graphs"),
        ("parallel_config", "tensor_parallel_size", 1, "tensor parallel size 8"),
        ("parallel_config", "enable_expert_parallel", False, "expert parallelism"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline parallelism"),
        ("speculative_config", "num_speculative_tokens", 4, "num_speculative_tokens=5"),
    ],
)
def test_w8a8_runtime_contract_still_fails_closed(loader, section, field, value, error):
    config = _config()
    setattr(getattr(config, section), field, value)
    with pytest.raises((ValueError, NotImplementedError), match=error):
        loader["_validate_w8a8_runtime_contract"](config, config.speculative_config.draft_model_config, _QuantConfig())


def test_w8a8_rejects_non_eager_registry_model_config(loader):
    config = _config()
    config.speculative_config.draft_model_config.enforce_eager = False
    with pytest.raises(ValueError, match="eager draft runtime"):
        loader["_validate_w8a8_runtime_contract"](config, config.speculative_config.draft_model_config, _QuantConfig())
