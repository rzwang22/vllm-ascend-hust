import builtins
import importlib
import json
import os
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm import ModelRegistry
from vllm.v1.worker.gpu import model_runner as vllm_model_runner

from vllm_ascend.models import (
    DSPARK_MODEL_ARCHITECTURE,
    register_dspark_model,
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
