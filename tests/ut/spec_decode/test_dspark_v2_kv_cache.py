# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import subprocess
import sys
import textwrap
from contextlib import nullcontext
from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode.dspark import (
    AscendDSparkSpeculator,
    create_dspark_speculator,
)

TARGET_LAYER = "model.layers.0.self_attn.swa_cache"
DRAFT_LAYERS = (
    "mtp.0.self_attn.swa_cache",
    "mtp.1.self_attn.swa_cache",
    "mtp.2.self_attn.swa_cache",
)


class _FakeBackend:
    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        return True


class _FakeAttentionLayer(nn.Module, AttentionLayerBase):
    def __init__(self, spec: FullAttentionSpec) -> None:
        super().__init__()
        self.spec = spec
        self.kv_cache = torch.tensor([])

    def get_attn_backend(self):
        return _FakeBackend

    def get_kv_cache_spec(self, _vllm_config):
        return self.spec


def _spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )


def _config():
    speculative_config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(dspark_noise_token_id=127),
        ),
        num_speculative_tokens=5,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[1]),
        ),
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
        ),
    )


def _loaded_speculator(monkeypatch: pytest.MonkeyPatch):
    config = _config()
    context = config.compilation_config.static_forward_context
    context[TARGET_LAYER] = _FakeAttentionLayer(_spec())
    speculator = create_dspark_speculator(config, torch.device("cpu"))

    from vllm_ascend.models.deepseek_v4_dspark import (
        DSparkDeepseekV4ForCausalLM,
    )
    from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

    draft_model = object.__new__(DSparkDeepseekV4ForCausalLM)
    nn.Module.__init__(draft_model)
    draft_model.get_draft_kv_cache_layer_names = lambda: list(DRAFT_LAYERS)

    def load_draft(_target_model, _vllm_config):
        for name in DRAFT_LAYERS:
            context[name] = _FakeAttentionLayer(_spec())
        return draft_model

    monkeypatch.setattr(model_loader, "load_dspark_model", load_draft)
    target_model = nn.Module()
    speculator.load_model(target_model)
    return config, speculator, target_model


def _kv_cache_config() -> KVCacheConfig:
    spec = _spec()
    groups = [
        KVCacheGroupSpec(
            layer_names=[TARGET_LAYER, DRAFT_LAYERS[0]],
            kv_cache_spec=spec,
        ),
        KVCacheGroupSpec(
            layer_names=list(DRAFT_LAYERS[1:]),
            kv_cache_spec=spec,
        ),
    ]
    tensors = [
        KVCacheTensor(
            size=spec.page_size_bytes,
            shared_by=list(group.layer_names),
        )
        for group in groups
    ]
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


def test_custom_dsv4_cache_layer_participates_in_kv_spec_discovery() -> None:
    config = _config()
    layer = _FakeAttentionLayer(_spec())
    config.compilation_config.static_forward_context[DRAFT_LAYERS[0]] = layer

    specs = attn_utils.get_kv_cache_spec(config)

    assert specs.keys() == {DRAFT_LAYERS[0]}
    assert specs[DRAFT_LAYERS[0]].indexes_kv_by_block_stride is True


@pytest.mark.parametrize("method", [None, "mtp", "eagle", "dflash"])
def test_custom_dsv4_kv_discovery_does_not_change_non_dspark_paths(
    method: str | None,
) -> None:
    config = _config()
    config.speculative_config = None if method is None else SimpleNamespace(method=method)
    config.compilation_config.static_forward_context[DRAFT_LAYERS[0]] = _FakeAttentionLayer(_spec())

    assert attn_utils.get_kv_cache_spec(config) == {}


def test_dsv4_allocator_keeps_single_shared_kv_page() -> None:
    config = _config()
    config.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(compress_ratios=[1]),
    )
    config.kv_transfer_config = None
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=32,
        model_version="deepseek_v4",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * 2,
                shared_by=[DRAFT_LAYERS[0]],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[DRAFT_LAYERS[0]],
                kv_cache_spec=spec,
            ),
        ],
    )

    class _DSABackend:
        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            *_args,
        ):
            return num_blocks, block_size, num_kv_heads, head_size

    class _Group:
        kv_cache_group_id = 0
        kv_cache_spec = spec
        backend = _DSABackend
        layer_names = [DRAFT_LAYERS[0]]

    with set_current_vllm_config(config):
        raw_caches = attn_utils._allocate_kv_cache(
            kv_cache_config,
            {},
            torch.device("cpu"),
        )
        caches = attn_utils._reshape_kv_cache_v2(
            [_Group()],
            raw_caches,
            "auto",
            [16],
            {},
            kv_cache_config,
        )

    raw_cache = raw_caches[DRAFT_LAYERS[0]]
    assert isinstance(raw_cache, torch.Tensor)
    assert isinstance(caches[DRAFT_LAYERS[0]], list)
    assert len(caches[DRAFT_LAYERS[0]]) == 1
    assert caches[DRAFT_LAYERS[0]][0].shape == (2, 16, 1, 64)
    assert caches[DRAFT_LAYERS[0]][0].data_ptr() == raw_cache.data_ptr()


def test_dsv4_allocator_honors_packed_offset_and_block_stride() -> None:
    config = _config()
    config.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(compress_ratios=[1]),
    )
    config.kv_transfer_config = None
    spec = AscendSlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=32,
        model_version="deepseek_v4",
    )
    page_size = spec.page_size_bytes
    block_stride = 2 * page_size
    allocation_size = 2 * block_stride
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=allocation_size,
                shared_by=[DRAFT_LAYERS[0]],
                offset=0,
                block_stride=block_stride,
            ),
            KVCacheTensor(
                size=allocation_size,
                shared_by=[DRAFT_LAYERS[1]],
                offset=page_size,
                block_stride=block_stride,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=list(DRAFT_LAYERS[:2]),
                kv_cache_spec=spec,
            ),
        ],
    )

    class _DSABackend:
        @staticmethod
        def get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            *_args,
        ):
            return num_blocks, block_size, num_kv_heads, head_size

    class _Group:
        kv_cache_group_id = 0
        kv_cache_spec = spec
        backend = _DSABackend
        layer_names = list(DRAFT_LAYERS[:2])

    with set_current_vllm_config(config):
        raw_caches = attn_utils._allocate_kv_cache(
            kv_cache_config,
            {},
            torch.device("cpu"),
        )
        caches = attn_utils._reshape_kv_cache_v2(
            [_Group()],
            raw_caches,
            "auto",
            [16],
            {},
            kv_cache_config,
        )

    assert raw_caches[DRAFT_LAYERS[0]] is raw_caches[DRAFT_LAYERS[1]]
    first_cache = caches[DRAFT_LAYERS[0]][0]
    second_cache = caches[DRAFT_LAYERS[1]][0]
    assert first_cache.shape == second_cache.shape == (2, 16, 1, 64)
    assert second_cache.data_ptr() - first_cache.data_ptr() == page_size
    assert first_cache.stride(0) * first_cache.element_size() == block_stride
    assert second_cache.stride(0) * second_cache.element_size() == block_stride


def test_draft_attention_layer_discovery_is_explicit_and_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config_value, speculator, target_model = _loaded_speculator(monkeypatch)

    assert speculator.model is speculator._model
    assert speculator._loaded_target_model is target_model
    assert speculator.target_attn_layer_names == frozenset({TARGET_LAYER})
    assert speculator.draft_attn_layer_names == frozenset(DRAFT_LAYERS)
    assert speculator.target_attn_layer_names.isdisjoint(speculator.draft_attn_layer_names)


def test_kv_specs_preserve_target_identity_and_publish_draft_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config_value, speculator, _target_model = _loaded_speculator(monkeypatch)
    target_spec = _spec()
    draft_specs = {name: _spec() for name in DRAFT_LAYERS}
    specs = {TARGET_LAYER: target_spec, **draft_specs}

    speculator.validate_kv_cache_specs(specs)

    assert specs[TARGET_LAYER] is target_spec
    assert speculator.draft_kv_cache_specs is not None
    assert set(speculator.draft_kv_cache_specs) == set(DRAFT_LAYERS)
    assert all(speculator.draft_kv_cache_specs[name] is draft_specs[name] for name in DRAFT_LAYERS)


def test_set_attn_installs_draft_only_groups_and_real_cache_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, speculator, _target_model = _loaded_speculator(monkeypatch)
    kv_cache_config = _kv_cache_config()
    context = config.compilation_config.static_forward_context
    installed_caches = {name: (torch.zeros(1), torch.zeros(1)) for name in DRAFT_LAYERS}
    for name, cache in installed_caches.items():
        context[name].kv_cache = cache

    class _Group:
        def __init__(self, layer_names):
            self.layer_names = list(layer_names)

    draft_groups = [
        [_Group([DRAFT_LAYERS[0]])],
        [_Group(DRAFT_LAYERS[1:])],
    ]
    init_calls = 0

    def init_attn_backend(*_args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        assert kwargs["active_layer_names"] == set(DRAFT_LAYERS)
        return draft_groups, object(), [16, 16]

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.init_attn_backend",
        init_attn_backend,
    )
    model_state = object()
    block_tables = object()

    speculator.set_attn(model_state, kv_cache_config, block_tables)
    speculator.set_attn(model_state, kv_cache_config, block_tables)

    assert init_calls == 1
    assert speculator.model_state is model_state
    assert speculator.block_tables is block_tables
    assert speculator.kv_cache_config is kv_cache_config
    assert speculator.attn_groups is draft_groups
    assert speculator.draft_kv_cache_group_ids == (0, 1)
    assert speculator.draft_kv_caches is not None
    assert all(speculator.draft_kv_caches[name] is context[name].kv_cache for name in DRAFT_LAYERS)
    assert TARGET_LAYER not in {
        name for group_list in speculator.attn_groups for group in group_list for name in group.layer_names
    }


def test_set_attn_failure_does_not_publish_partial_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, speculator, _target_model = _loaded_speculator(monkeypatch)
    kv_cache_config = _kv_cache_config()
    for name in DRAFT_LAYERS[:-1]:
        config.compilation_config.static_forward_context[name].kv_cache = (
            torch.zeros(1),
            torch.zeros(1),
        )

    class _Group:
        def __init__(self, layer_names):
            self.layer_names = list(layer_names)

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.init_attn_backend",
        lambda *_args, **_kwargs: (
            [[_Group([DRAFT_LAYERS[0]])], [_Group(DRAFT_LAYERS[1:])]],
            object(),
            [16, 16],
        ),
    )

    with pytest.raises(RuntimeError, match="were not installed"):
        speculator.set_attn(object(), kv_cache_config, object())

    assert speculator.model_state is None
    assert speculator.kv_cache_config is None
    assert speculator.block_tables is None
    assert speculator.attn_groups is None
    assert speculator.attn_backends is None
    assert speculator.draft_kv_caches is None
    assert speculator._kv_cache_signature is None


def test_runner_restores_kv_state_when_dspark_installation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, speculator, _target_model = _loaded_speculator(monkeypatch)
    kv_cache_config = _kv_cache_config()
    context = config.compilation_config.static_forward_context
    previous_caches = {name: layer.kv_cache for name, layer in context.items()}

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    runner = object.__new__(ascend_model_runner.NPUModelRunner)
    runner.speculator = speculator
    runner.compilation_config = config.compilation_config
    runner.model_state = object()

    def allocate_then_fail(_runner, cache_config):
        _runner.kv_cache_config = cache_config
        _runner.block_tables = object()
        _runner.kv_caches = [torch.zeros(1)]
        for layer in context.values():
            layer.kv_cache = (torch.zeros(1), torch.zeros(1))

    monkeypatch.setattr(
        GPUModelRunner,
        "initialize_kv_cache",
        allocate_then_fail,
    )
    monkeypatch.setattr(
        ascend_model_runner,
        "graph_manager_wrapper",
        lambda _runner: nullcontext(),
    )
    monkeypatch.setattr(
        speculator,
        "set_attn",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("install failed")),
    )

    with pytest.raises(RuntimeError, match="install failed"):
        ascend_model_runner.NPUModelRunner.initialize_kv_cache(
            runner,
            kv_cache_config,
        )

    assert not hasattr(runner, "kv_cache_config")
    assert not hasattr(runner, "block_tables")
    assert not hasattr(runner, "kv_caches")
    assert all(context[name].kv_cache is previous_caches[name] for name in previous_caches)
    assert speculator.kv_cache_config is None
    assert speculator.draft_kv_caches is None


def test_runner_repeated_dspark_kv_initialization_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config_value, speculator, _target_model = _loaded_speculator(monkeypatch)
    kv_cache_config = _kv_cache_config()
    speculator.kv_cache_config = kv_cache_config
    speculator.draft_kv_caches = MappingProxyType({name: torch.zeros(1) for name in DRAFT_LAYERS})
    speculator._kv_cache_signature = speculator._cache_config_signature(kv_cache_config)

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    runner = object.__new__(ascend_model_runner.NPUModelRunner)
    runner.speculator = speculator
    monkeypatch.setattr(
        GPUModelRunner,
        "initialize_kv_cache",
        lambda *_args: pytest.fail("idempotent DSpark init reallocated KV cache"),
    )

    ascend_model_runner.NPUModelRunner.initialize_kv_cache(
        runner,
        deepcopy(kv_cache_config),
    )


def test_runner_non_dspark_kv_initialization_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    kv_cache_config = _kv_cache_config()
    runner = object.__new__(ascend_model_runner.NPUModelRunner)
    runner.speculator = None
    calls: list[KVCacheConfig] = []
    monkeypatch.setattr(
        GPUModelRunner,
        "initialize_kv_cache",
        lambda _runner, config: calls.append(config),
    )
    monkeypatch.setattr(
        ascend_model_runner,
        "graph_manager_wrapper",
        lambda _runner: nullcontext(),
    )

    ascend_model_runner.NPUModelRunner.initialize_kv_cache(
        runner,
        kv_cache_config,
    )

    assert calls == [kv_cache_config]


@pytest.mark.parametrize("invalid_count", [0, 2])
def test_kv_group_membership_must_be_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    invalid_count: int,
) -> None:
    _config_value, speculator, _target_model = _loaded_speculator(monkeypatch)
    kv_cache_config = _kv_cache_config()
    name = DRAFT_LAYERS[0]
    for group in kv_cache_config.kv_cache_groups:
        group.layer_names = [layer for layer in group.layer_names if layer != name]
    if invalid_count == 2:
        for group in kv_cache_config.kv_cache_groups:
            group.layer_names.append(name)

    with pytest.raises(RuntimeError, match="exactly one"):
        speculator.validate_kv_cache_config(kv_cache_config)


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("tensor_parallel_size", 4, "tensor parallel size 8"),
        ("enable_expert_parallel", False, "expert parallelism"),
        ("pipeline_parallel_size", 2, "pipeline parallel size 1"),
    ],
)
def test_kv_lifecycle_enforces_tp8_ep_pp1_ownership(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value,
    error: str,
) -> None:
    config, speculator, _target_model = _loaded_speculator(monkeypatch)
    setattr(config.parallel_config, attribute, value)

    with pytest.raises((ValueError, NotImplementedError), match=error):
        speculator.validate_kv_cache_config(_kv_cache_config())


def test_dspark_kv_lifecycle_imports_no_forbidden_runtime() -> None:
    module = __import__(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator",
        fromlist=["AscendDSparkSpeculator"],
    )
    forbidden_prefixes = (
        "vllm.models.deepseek_v4.nvidia",
        "vllm.v1.worker.gpu.spec_decode.dspark",
        "vllm_ascend.ops.triton.spec_decode",
    )

    assert module.AscendDSparkSpeculator is AscendDSparkSpeculator
    assert all(not getattr(value, "__module__", "").startswith(forbidden_prefixes) for value in vars(module).values())


def test_dspark_kv_lifecycle_has_no_forbidden_import_delta() -> None:
    script = textwrap.dedent(
        """
        import sys
        import types

        build_info = types.ModuleType("vllm_ascend._build_info")
        build_info.__device_type__ = "A2"
        build_info.__soc_version__ = "ASCEND910B2"
        sys.modules[build_info.__name__] = build_info
        before = set(sys.modules)
        from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
            AscendDSparkSpeculator,
        )
        after = set(sys.modules)
        forbidden = (
            "vllm.models.deepseek_v4.nvidia",
            "vllm.v1.worker.gpu.spec_decode.dspark",
            "vllm_ascend.ops.triton.spec_decode",
        )
        delta = sorted(
            name for name in after - before if name.startswith(forbidden)
        )
        assert delta == [], delta
        assert AscendDSparkSpeculator.__module__ == (
            "vllm_ascend.worker.v2.spec_decode.dspark.speculator"
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
