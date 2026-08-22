import builtins
import importlib
import json
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm import ModelRegistry
from vllm.config.compilation import CompilationMode
from vllm.v1.worker.gpu import model_runner as vllm_model_runner

from vllm_ascend.models import (
    DSPARK_MODEL_ARCHITECTURE,
    register_dspark_model,
)
from vllm_ascend.quantization.methods import get_scheme_class
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
)
from vllm_ascend.quantization.modelslim_config import (
    AscendModelSlimConfig,
    get_packed_modules_mapping,
    get_quant_type_for_layer,
)
from vllm_ascend.spec_decode import (
    DSparkRuntimeNotWiredError,
)
from vllm_ascend.worker.v2.spec_decode.dspark import (
    create_dspark_speculator,
)
from vllm_ascend.worker.v2.spec_decode.runner_init import (
    include_ascend_dspark_in_core_load_lifecycle,
)


def _dspark_config(*, method: str = "dspark"):
    draft_hf_config = SimpleNamespace(dspark_noise_token_id=128799)
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        num_speculative_tokens=3,
        use_dspark=lambda: method == "dspark",
    )
    return SimpleNamespace(speculative_config=speculative_config)


def _new_ascend_draft_model() -> nn.Module:
    model_type = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark").DSparkDeepseekV4ForCausalLM
    model = object.__new__(model_type)
    nn.Module.__init__(model)
    return model


def _modelslim_dspark_descriptor() -> dict[str, str]:
    descriptor = {
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        # The target-level entry is part of the real DeepSeek V4 descriptor
        # and activates its HF-to-vLLM aliases in AscendModelSlimConfig.
        "hc_head_fn": "FLOAT",
        "mtp.0.embed.weight": "FLOAT",
        "mtp.0.main_proj.weight": "W8A8_DYNAMIC",
        "mtp.0.main_norm.weight": "FLOAT",
        "mtp.2.norm.weight": "FLOAT",
        "mtp.2.head.weight": "FLOAT",
        "mtp.2.markov_head.markov_w1.weight": "FLOAT",
        "mtp.2.markov_head.markov_w2.weight": "FLOAT",
        "mtp.2.confidence_head.proj.weight": "FLOAT",
        "mtp.2.hc_head_fn": "FLOAT",
        "mtp.2.hc_head_base": "FLOAT",
        "mtp.2.hc_head_scale": "FLOAT",
    }
    for stage in range(3):
        stage_prefix = f"mtp.{stage}"
        descriptor.update(
            {
                f"{stage_prefix}.attn.wq_a.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.attn.wq_b.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.attn.wkv.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.attn.wo_a.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.attn.wo_b.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.ffn.experts.0.w1.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.ffn.experts.0.w2.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.ffn.experts.0.w3.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.ffn.shared_experts.w1.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.ffn.shared_experts.w2.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.ffn.shared_experts.w3.weight": "W8A8_DYNAMIC",
                f"{stage_prefix}.attn_norm.weight": "FLOAT",
                f"{stage_prefix}.ffn_norm.weight": "FLOAT",
                f"{stage_prefix}.hc_attn_fn": "FLOAT",
                f"{stage_prefix}.hc_attn_base": "FLOAT",
                f"{stage_prefix}.hc_attn_scale": "FLOAT",
                f"{stage_prefix}.hc_ffn_fn": "FLOAT",
                f"{stage_prefix}.hc_ffn_base": "FLOAT",
                f"{stage_prefix}.hc_ffn_scale": "FLOAT",
            }
        )
    return descriptor


def _modelslim_loader_config(
    *,
    target_quant_config: AscendModelSlimConfig | None = None,
):
    checkpoint = "Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"
    draft_hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        expert_dtype="fp4",
        n_mtp_layers=3,
        dspark_block_size=5,
        dspark_target_layer_ids=[40, 41, 42],
    )
    draft_model_config = SimpleNamespace(
        model=checkpoint,
        hf_config=draft_hf_config,
    )
    target_model_config = SimpleNamespace(
        model=checkpoint,
        enforce_eager=True,
    )
    speculative_config = SimpleNamespace(
        method="dspark",
        draft_model_config=draft_model_config,
        num_speculative_tokens=5,
        attention_backend="ASCEND",
    )
    return SimpleNamespace(
        model_config=target_model_config,
        quant_config=(
            target_quant_config
            if target_quant_config is not None
            else AscendModelSlimConfig(_modelslim_dspark_descriptor())
        ),
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
        ),
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
        ),
    )


def test_dspark_registry_resolves_to_ascend_model_without_touching_other_architectures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_models = dict(ModelRegistry.models)
    monkeypatch.setattr(ModelRegistry, "models", isolated_models)
    non_dspark_architecture = "DeepseekV4ForCausalLM"
    non_dspark_registration = isolated_models[non_dspark_architecture]

    register_dspark_model()
    first_registration = isolated_models[DSPARK_MODEL_ARCHITECTURE]
    register_dspark_model()

    resolved_type = ModelRegistry._try_load_model_cls(DSPARK_MODEL_ARCHITECTURE)

    assert resolved_type is not None
    assert resolved_type.__module__ == "vllm_ascend.models.deepseek_v4_dspark"
    assert resolved_type.__name__ == "DSparkDeepseekV4ForCausalLM"
    assert "nvidia" not in resolved_type.__module__
    assert isolated_models[DSPARK_MODEL_ARCHITECTURE] is first_registration
    assert isolated_models[non_dspark_architecture] is non_dspark_registration


def test_dspark_registry_resolution_isolated_from_nvidia_runtime() -> None:
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

        import vllm_ascend
        from vllm import ModelRegistry

        vllm_ascend.register_model()
        nvidia_prefix = "vllm.models.deepseek_v4.nvidia"
        forbidden_prefixes = (
            nvidia_prefix,
            "vllm.v1.worker.gpu.spec_decode.dspark",
            "vllm_ascend.ops.triton.spec_decode",
        )
        before = set(sys.modules)
        preexisting_nvidia = sorted(
            module for module in before if module.startswith(nvidia_prefix)
        )
        model_cls = ModelRegistry._try_load_model_cls("DSparkDraftModel")
        if model_cls is None:
            raise RuntimeError("DSparkDraftModel did not resolve")
        from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

        if not callable(model_loader.load_dspark_model):
            raise RuntimeError("Ascend DSpark model loader did not resolve")
        new_modules = set(sys.modules) - before

        def referenced_module(value):
            if isinstance(value, types.ModuleType):
                return value.__name__
            return getattr(value, "__module__", "")

        model_global_modules = (
            model_cls.__module__,
            "vllm_ascend.models.deepseek_v4",
        )
        nvidia_globals = sorted(
            f"{{module_name}}.{{name}}"
            for module_name in model_global_modules
            for name, value in vars(sys.modules[module_name]).items()
            if referenced_module(value).startswith(nvidia_prefix)
        )
        payload = {{
            "class_module": model_cls.__module__,
            "class_name": model_cls.__name__,
            "preexisting_nvidia": preexisting_nvidia,
            "new_forbidden": sorted(
                module
                for module in new_modules
                if module.startswith(forbidden_prefixes)
            ),
            "mro_modules": [base.__module__ for base in model_cls.__mro__],
            "nvidia_globals": nvidia_globals,
        }}
        print("DSPARK_IMPORT_AUDIT=" + json.dumps(payload, sort_keys=True))
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
    audit_line = next(line for line in result.stdout.splitlines() if line.startswith("DSPARK_IMPORT_AUDIT="))
    audit = json.loads(audit_line.removeprefix("DSPARK_IMPORT_AUDIT="))

    assert audit["class_module"] == "vllm_ascend.models.deepseek_v4_dspark"
    assert audit["class_name"] == "DSparkDeepseekV4ForCausalLM"
    assert audit["preexisting_nvidia"] == []
    assert audit["new_forbidden"] == []
    assert all(not module.startswith("vllm.models.deepseek_v4.nvidia") for module in audit["mro_modules"])
    assert audit["nvidia_globals"] == []


def test_speculator_loads_draft_model_once_and_publishes_loader_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))
    target_model = nn.Module()
    draft_model = _new_ascend_draft_model()
    loader_calls = 0

    def fake_loader(loaded_target, _config):
        nonlocal loader_calls
        loader_calls += 1
        assert loaded_target is target_model
        return draft_model

    monkeypatch.setattr(model_loader, "load_dspark_model", fake_loader)

    speculator.load_model(target_model)
    speculator.load_model(target_model)

    assert loader_calls == 1
    assert speculator.model is draft_model


def test_speculator_rejects_reloading_for_different_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))
    first_target = nn.Module()
    draft_model = _new_ascend_draft_model()
    loader_calls = 0

    def fake_loader(_target, _config):
        nonlocal loader_calls
        loader_calls += 1
        return draft_model

    monkeypatch.setattr(model_loader, "load_dspark_model", fake_loader)
    speculator.load_model(first_target)

    with pytest.raises(RuntimeError, match="different target"):
        speculator.load_model(nn.Module())

    assert loader_calls == 1
    assert speculator.model is draft_model


def test_failed_draft_load_leaves_speculator_uninitialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))

    def failing_loader(_target, _config):
        raise ValueError("checkpoint failure")

    monkeypatch.setattr(model_loader, "load_dspark_model", failing_loader)

    with pytest.raises(ValueError, match="checkpoint failure"):
        speculator.load_model(nn.Module())
    with pytest.raises(DSparkRuntimeNotWiredError, match="draft-model loading"):
        _ = speculator.model


def test_non_ascend_loader_result_is_rejected_without_partial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))
    monkeypatch.setattr(
        model_loader,
        "load_dspark_model",
        lambda _target, _config: nn.Module(),
    )

    with pytest.raises(RuntimeError, match="non-Ascend implementation"):
        speculator.load_model(nn.Module())
    with pytest.raises(DSparkRuntimeNotWiredError, match="draft-model loading"):
        _ = speculator.model


def test_model_loading_does_not_import_core_dspark_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    forbidden_prefixes = (
        "vllm.models.deepseek_v4.nvidia",
        "vllm.v1.worker.gpu.spec_decode.dspark",
        "vllm_ascend.ops.triton.spec_decode",
    )
    import_attempts: list[str] = []
    original_import = builtins.__import__
    draft_model = _new_ascend_draft_model()

    def import_spy(name, *args, **kwargs):
        if name.startswith(forbidden_prefixes):
            import_attempts.append(name)
            raise AssertionError(f"unexpected CUDA/Triton DSpark import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(
        model_loader,
        "load_dspark_model",
        lambda _target, _config: draft_model,
    )
    monkeypatch.setattr(builtins, "__import__", import_spy)

    before = set(sys.modules)
    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))
    speculator.load_model(nn.Module())
    new_modules = set(sys.modules) - before

    assert speculator.model is draft_model
    assert import_attempts == []
    assert not any(module.startswith(forbidden_prefixes) for module in new_modules)


def _run_hardware_agnostic_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    draft_model,
    target_model,
):
    from vllm.compilation import backends as compilation_backends

    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    def fake_replace(value, **changes):
        fields = vars(value).copy()
        fields.update(changes)
        return SimpleNamespace(**fields)

    draft_model_config = object()
    config = SimpleNamespace(
        model_config=SimpleNamespace(),
        quant_config=None,
        speculative_config=SimpleNamespace(
            draft_model_config=draft_model_config,
            attention_backend="ASCEND",
        ),
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
        ),
    )
    monkeypatch.setattr(model_loader, "replace", fake_replace)
    monkeypatch.setattr(
        model_loader,
        "get_model",
        lambda *, vllm_config, model_config: (
            draft_model
            if model_config is draft_model_config and vllm_config.attention_config.use_non_causal
            else pytest.fail("DSpark loader did not build the expected config")
        ),
    )
    monkeypatch.setattr(
        model_loader,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        compilation_backends,
        "set_model_tag",
        lambda _tag: nullcontext(),
    )
    return model_loader.load_dspark_model(target_model, config)


def _weighted_module(value: float) -> nn.Module:
    module = nn.Module()
    module.register_parameter(
        "weight",
        nn.Parameter(torch.tensor([value])),
    )
    return module


def test_modelslim_loader_uses_independent_draft_quant_config_and_own_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.compilation import backends as compilation_backends

    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    def fake_replace(value, **changes):
        fields = vars(value).copy()
        fields.update(changes)
        return SimpleNamespace(**fields)

    target_quant_config = AscendModelSlimConfig(_modelslim_dspark_descriptor())
    target_quant_description = dict(target_quant_config.quant_description)
    config = _modelslim_loader_config(target_quant_config=target_quant_config)
    draft_model_config = config.speculative_config.draft_model_config
    draft_embed = _weighted_module(3.0)
    draft_lm_head = _weighted_module(4.0)
    draft_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed),
        lm_head=draft_lm_head,
        has_own_embed_tokens=True,
        has_own_lm_head=True,
    )
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=_weighted_module(1.0)),
        lm_head=_weighted_module(2.0),
    )
    captured_config = None

    def fake_get_model(*, vllm_config, model_config):
        nonlocal captured_config
        captured_config = vllm_config
        assert model_config is draft_model_config
        return draft_model

    monkeypatch.setattr(model_loader, "replace", fake_replace)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        model_loader,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        compilation_backends,
        "set_model_tag",
        lambda _tag: nullcontext(),
    )

    loaded = model_loader.load_dspark_model(target_model, config)

    assert loaded is draft_model
    assert captured_config is not None
    assert captured_config.model_config is draft_model_config
    assert captured_config.quant_config is not target_quant_config
    assert target_quant_config.quant_description == target_quant_description
    assert target_quant_description.items() <= captured_config.quant_config.quant_description.items()
    assert captured_config.quant_config.quant_description is not target_quant_config.quant_description
    assert captured_config.quant_config.quant_description["mtp.0.self_attn.wq_a.weight"] == "W8A8_DYNAMIC"
    assert draft_model.model.embed_tokens is draft_embed
    assert draft_model.lm_head is draft_lm_head


def test_modelslim_descriptor_selects_w8a8_despite_fp4_hf_hint() -> None:
    config = _modelslim_loader_config()
    assert config.speculative_config.draft_model_config.hf_config.expert_dtype == "fp4"
    quant_config = config.quant_config
    packed_mapping = get_packed_modules_mapping("deepseek_v4")

    attention_prefix = quant_config.quant_prefix_mapper(
        "deepseek_v4",
        "mtp.0.self_attn.wq_a",
    )
    expert_prefix = quant_config.quant_prefix_mapper(
        "deepseek_v4",
        "mtp.0.mlp.experts",
    )

    assert (
        get_quant_type_for_layer(
            quant_config.quant_description,
            attention_prefix,
            "linear",
            packed_mapping,
        )
        == "W8A8_DYNAMIC"
    )
    assert (
        get_quant_type_for_layer(
            quant_config.quant_description,
            expert_prefix,
            "moe",
            packed_mapping,
        )
        == "W8A8_DYNAMIC"
    )
    assert get_scheme_class("W8A8_DYNAMIC", "moe") is AscendW8A8DynamicFusedMoEMethod
    assert all(
        "MXFP" not in quant_type and "W4A8" not in quant_type
        for quant_type in quant_config.quant_description.values()
        if isinstance(quant_type, str)
    )


@pytest.mark.parametrize(
    ("mutate_descriptor", "error_match"),
    [
        (
            lambda descriptor: descriptor.__setitem__(
                "mtp.0.ffn.experts.0.w1.weight",
                "W4A8_MXFP",
            ),
            "forbids MXFP/FP4",
        ),
        (
            lambda descriptor: descriptor.pop("mtp.2.head.weight"),
            "mtp.2.head.weight.*must be FLOAT",
        ),
        (
            lambda descriptor: descriptor.__setitem__(
                "mtp.2.confidence_head.proj.weight",
                "W8A8_DYNAMIC",
            ),
            "confidence-head parameters must remain FLOAT",
        ),
    ],
)
def test_modelslim_descriptor_rejects_mxfp_and_nonfloat_heads(
    mutate_descriptor: Callable[[dict[str, str]], object],
    error_match: str,
) -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark.model_loader import (
        _validate_w8a8_descriptor,
    )

    descriptor = _modelslim_dspark_descriptor()
    mutate_descriptor(descriptor)
    draft_hf_config = SimpleNamespace(n_mtp_layers=3)

    with pytest.raises(ValueError, match=error_match):
        _validate_w8a8_descriptor(descriptor, draft_hf_config)


def test_modelslim_loader_enforces_initial_tp8_ep_eager_contract() -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark.model_loader import (
        _build_draft_quant_config,
        _validate_w8a8_runtime_contract,
    )

    config = _modelslim_loader_config()
    draft_model_config = config.speculative_config.draft_model_config
    draft_quant_config = _build_draft_quant_config(config, draft_model_config)

    assert draft_quant_config is not config.quant_config
    _validate_w8a8_runtime_contract(
        config,
        draft_model_config,
        draft_quant_config,
    )

    config.parallel_config.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="tensor parallel size 8"):
        _validate_w8a8_runtime_contract(
            config,
            draft_model_config,
            draft_quant_config,
        )


def test_modelslim_loader_requires_same_target_and_draft_checkpoint() -> None:
    from vllm_ascend.worker.v2.spec_decode.dspark.model_loader import (
        _build_draft_quant_config,
    )

    config = _modelslim_loader_config()
    draft_model_config = config.speculative_config.draft_model_config
    draft_model_config.model = "Eco-Tech/a-different-dspark-checkpoint"

    with pytest.raises(ValueError, match="target and draft to use the same checkpoint"):
        _build_draft_quant_config(config, draft_model_config)


def test_dspark_model_uses_checkpoint_quant_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark")
    calls: list[tuple[str, str, object]] = []

    class FakeEmbedding(nn.Module):
        def __init__(self, *_args, quant_config=None, prefix="", **_kwargs):
            super().__init__()
            calls.append(("embedding", prefix, quant_config))

    class FakeLMHead(nn.Module):
        def __init__(self, *_args, quant_config=None, prefix="", **_kwargs):
            super().__init__()
            calls.append(("lm_head", prefix, quant_config))

    class FakeLinear(nn.Module):
        def __init__(self, *_args, quant_config=None, prefix="", **_kwargs):
            super().__init__()
            calls.append(("linear", prefix, quant_config))

    class FakeDecoder(nn.Module):
        def __init__(self, _vllm_config, prefix, **_kwargs):
            super().__init__()
            calls.append(("decoder", prefix, _vllm_config.quant_config))

    class FakeNorm(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    quant_config = AscendModelSlimConfig(_modelslim_dspark_descriptor())
    hf_config = SimpleNamespace(
        hc_mult=2,
        hidden_size=8,
        dspark_block_size=5,
        dspark_target_layer_ids=[40, 41, 42],
        n_mtp_layers=3,
        num_hidden_layers=61,
        vocab_size=16,
        dspark_markov_rank=4,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
    )
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config),
        ),
    )
    monkeypatch.setattr(model_module, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(model_module, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(model_module, "ReplicatedLinear", FakeLinear)
    monkeypatch.setattr(model_module, "DeepseekV2DecoderLayer", FakeDecoder)
    monkeypatch.setattr(model_module, "RMSNorm", FakeNorm)

    model = model_module.DeepseekV4DSparkModel(vllm_config=vllm_config)

    assert model.confidence_head is not None
    assert ("embedding", "mtp.0.embed", quant_config) in calls
    assert ("linear", "mtp.0.main_proj", quant_config) in calls
    assert ("embedding", "mtp.2.markov_head.markov_w1", quant_config) in calls
    assert ("lm_head", "mtp.2.markov_head.markov_w2", quant_config) in calls
    assert ("linear", "mtp.2.confidence_head.proj", quant_config) in calls
    assert [prefix for kind, prefix, _ in calls if kind == "decoder"] == [
        "mtp.0",
        "mtp.1",
        "mtp.2",
    ]


def test_dspark_wrapper_owns_descriptor_embedding_and_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark")
    head_calls: list[tuple[str, object]] = []

    class FakeBackbone(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.layers = nn.ModuleDict()

    class FakeLMHead(nn.Module):
        def __init__(self, *_args, quant_config=None, prefix="", **_kwargs):
            super().__init__()
            head_calls.append((prefix, quant_config))

    quant_config = AscendModelSlimConfig(_modelslim_dspark_descriptor())
    hf_config = SimpleNamespace(
        n_mtp_layers=3,
        vocab_size=16,
        hidden_size=8,
    )
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf_config),
        ),
    )
    monkeypatch.setattr(model_module, "DeepseekV4DSparkModel", FakeBackbone)
    monkeypatch.setattr(model_module, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(
        model_module.DSparkDeepseekV4ForCausalLM,
        "set_moe_parameters",
        lambda _self: None,
    )

    model = model_module.DSparkDeepseekV4ForCausalLM(vllm_config=vllm_config)

    assert model.has_own_embed_tokens
    assert model.has_own_lm_head
    assert head_calls == [("mtp.2.head", quant_config)]


def test_same_checkpoint_contract_shares_target_embedding_and_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_embed = _weighted_module(1.0)
    target_lm_head = _weighted_module(2.0)
    draft_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=_weighted_module(3.0)),
        lm_head=_weighted_module(4.0),
        has_own_embed_tokens=False,
        has_own_lm_head=False,
    )
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=target_lm_head,
    )

    loaded = _run_hardware_agnostic_loader(
        monkeypatch,
        draft_model=draft_model,
        target_model=target_model,
    )

    assert loaded is draft_model
    assert draft_model.model.embed_tokens is target_embed
    assert draft_model.lm_head is target_lm_head


def test_distinct_checkpoint_embedding_and_lm_head_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft_embed = _weighted_module(3.0)
    draft_lm_head = _weighted_module(4.0)
    draft_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=draft_embed),
        lm_head=draft_lm_head,
        has_own_embed_tokens=True,
        has_own_lm_head=True,
    )
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=_weighted_module(1.0)),
        lm_head=_weighted_module(2.0),
    )

    loaded = _run_hardware_agnostic_loader(
        monkeypatch,
        draft_model=draft_model,
        target_model=target_model,
    )

    assert loaded is draft_model
    assert draft_model.model.embed_tokens is draft_embed
    assert draft_model.lm_head is draft_lm_head


def test_npu_runner_uses_core_post_target_load_point_for_dspark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner
    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    config = _dspark_config()
    speculator = create_dspark_speculator(config, torch.device("cpu"))
    target_model = nn.Module()
    draft_model = _new_ascend_draft_model()
    original_type = vllm_model_runner.DraftModelSpeculator
    core_load_calls = 0

    monkeypatch.setattr(
        model_loader,
        "load_dspark_model",
        lambda loaded_target, _config: (
            draft_model if loaded_target is target_model else pytest.fail("wrong target model")
        ),
    )

    def fake_core_load(runner, _load_dummy_weights=False, *_args, **_kwargs):
        nonlocal core_load_calls
        core_load_calls += 1
        assert isinstance(runner.speculator, vllm_model_runner.DraftModelSpeculator)
        runner.speculator.load_model(target_model)

    monkeypatch.setattr(vllm_model_runner.GPUModelRunner, "load_model", fake_core_load)
    runner = object.__new__(ascend_model_runner.NPUModelRunner)
    runner.vllm_config = config
    runner.speculator = speculator

    ascend_model_runner.NPUModelRunner.load_model(runner)

    assert core_load_calls == 1
    assert speculator.model is draft_model
    assert vllm_model_runner.DraftModelSpeculator is original_type


@pytest.mark.parametrize("method", ["mtp", "eagle", "dflash"])
def test_non_dspark_load_lifecycle_gate_is_unchanged(method: str) -> None:
    original_type = vllm_model_runner.DraftModelSpeculator

    with include_ascend_dspark_in_core_load_lifecycle(_dspark_config(method=method)):
        assert vllm_model_runner.DraftModelSpeculator is original_type


def _minimal_weight_model(*, has_own_embedding_and_head: bool = False):
    model_type = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark").DSparkDeepseekV4ForCausalLM
    draft_model = object.__new__(model_type)
    nn.Module.__init__(draft_model)
    draft_model.config = SimpleNamespace(
        num_hidden_layers=10,
        num_attention_heads=1,
        expert_dtype="fp4",
    )
    draft_model.has_own_embed_tokens = has_own_embedding_and_head
    draft_model.has_own_lm_head = has_own_embedding_and_head

    backbone = nn.Module()
    backbone.num_dspark_layers = 1
    backbone.confidence_head = None
    backbone.get_expert_mapping = lambda: []
    backbone.layers = nn.ModuleDict({"10": nn.Module()})
    backbone.layers["10"].main_proj = nn.Module()
    backbone.layers["10"].main_proj.register_parameter(
        "weight",
        nn.Parameter(torch.zeros(1)),
    )
    backbone.embed_tokens = _weighted_module(2.0)
    draft_model.model = backbone
    draft_model.lm_head = _weighted_module(3.0)
    return draft_model


def test_checkpoint_mapping_loads_expected_parameter_strictly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark")
    draft_model = _minimal_weight_model()
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    loaded = draft_model.load_weights([("mtp.0.main_proj.weight", torch.ones(1))])

    assert loaded == {"model.layers.10.main_proj.weight"}
    assert torch.equal(
        draft_model.model.layers["10"].main_proj.weight,
        torch.ones(1),
    )


def test_checkpoint_mapping_requires_only_nonshared_embedding_and_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark")
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    weights = [("mtp.0.main_proj.weight", torch.ones(1))]

    shared_model = _minimal_weight_model()
    assert shared_model.load_weights(weights) == {"model.layers.10.main_proj.weight"}

    own_model = _minimal_weight_model(has_own_embedding_and_head=True)
    with pytest.raises(ValueError, match="model.embed_tokens.weight"):
        own_model.load_weights(weights)

    loaded = own_model.load_weights(
        [
            *weights,
            ("mtp.0.embed.weight", torch.ones(1)),
            ("mtp.0.head.weight", torch.ones(1)),
        ]
    )
    assert loaded == {
        "model.layers.10.main_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    }


def test_checkpoint_mapping_rejects_missing_and_unexpected_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark")
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    with pytest.raises(ValueError, match="did not initialize required"):
        _minimal_weight_model().load_weights([])
    with pytest.raises(ValueError, match="Unexpected DSpark checkpoint weight"):
        _minimal_weight_model().load_weights([("mtp.0.unknown.weight", torch.ones(1))])


def test_checkpoint_mapping_loads_confidence_head_strictly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark")
    draft_model = _minimal_weight_model()
    draft_model.model.num_dspark_layers = 3
    confidence_head = nn.Module()
    confidence_head.proj = _weighted_module(0.0)
    draft_model.model.confidence_head = confidence_head
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    loaded = draft_model.load_weights(
        [
            ("mtp.0.main_proj.weight", torch.ones(1)),
            ("mtp.2.confidence_head.proj.weight", torch.ones(1)),
        ]
    )

    assert loaded == {
        "model.layers.10.main_proj.weight",
        "model.confidence_head.proj.weight",
    }
    assert torch.equal(draft_model.model.confidence_head.proj.weight, torch.ones(1))


def test_checkpoint_mapping_uses_parameter_scheme_for_w8a8_expert_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_module = importlib.import_module("vllm_ascend.models.deepseek_v4_dspark")
    model_type = model_module.DSparkDeepseekV4ForCausalLM
    draft_model = object.__new__(model_type)
    nn.Module.__init__(draft_model)
    draft_model.config = SimpleNamespace(
        num_hidden_layers=10,
        num_attention_heads=1,
        expert_dtype="fp4",
    )
    draft_model.has_own_embed_tokens = False
    draft_model.has_own_lm_head = False

    routed_experts = nn.Module()
    expert_scale = nn.Parameter(torch.zeros(1))
    routed_experts.register_parameter("w13_weight_scale", expert_scale)
    expert_loader_calls: list[int] = []

    def expert_weight_loader(
        param,
        loaded_weight,
        _name,
        *,
        shard_id,
        expert_id,
        return_success,
    ):
        assert shard_id == "w1"
        assert return_success
        expert_loader_calls.append(expert_id)
        if expert_id == 0:
            return False
        param.data.copy_(loaded_weight)
        return True

    expert_scale.weight_loader = expert_weight_loader
    experts = nn.Module()
    experts.routed_experts = routed_experts
    mlp = nn.Module()
    mlp.experts = experts
    layer = nn.Module()
    layer.mlp = mlp
    backbone = nn.Module()
    backbone.num_dspark_layers = 1
    backbone.confidence_head = None
    backbone.layers = nn.ModuleDict({"10": layer})
    backbone.get_expert_mapping = lambda: [
        (
            "experts.routed_experts.w13_",
            "experts.0.gate_proj.",
            0,
            "w1",
        ),
        (
            "experts.routed_experts.w13_",
            "experts.1.gate_proj.",
            1,
            "w1",
        ),
    ]
    backbone.embed_tokens = _weighted_module(2.0)
    draft_model.model = backbone
    draft_model.lm_head = _weighted_module(3.0)
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        model_module,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )

    loaded = draft_model.load_weights(
        [
            ("mtp.0.ffn.experts.0.w1.scale", torch.zeros(1)),
            ("mtp.0.ffn.experts.1.w1.scale", torch.ones(1)),
        ]
    )

    assert loaded == {"model.layers.10.mlp.experts.routed_experts.w13_weight_scale"}
    assert expert_loader_calls == [0, 1]
    assert torch.equal(expert_scale, torch.ones(1))
