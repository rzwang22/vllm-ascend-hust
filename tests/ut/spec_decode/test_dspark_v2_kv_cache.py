# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import subprocess
import sys
import textwrap
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from inspect import signature
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

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
TARGET_LAYER_2_CACHES = (
    "model.layers.2.self_attn.attn",
    "model.layers.2.self_attn.compressor.state_cache",
    "model.layers.2.self_attn.indexer.compressor.state_cache",
    "model.layers.2.self_attn.indexer.k_cache",
    "model.layers.2.self_attn.swa_cache",
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


class _FailingAttentionLayer(_FakeAttentionLayer):
    def __init__(self, spec: FullAttentionSpec) -> None:
        self.rejected_cache = None
        super().__init__(spec)

    def __setattr__(self, name, value) -> None:
        if name == "kv_cache" and value is getattr(self, "rejected_cache", None):
            raise RuntimeError("cache installation failed")
        super().__setattr__(name, value)


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


def _official_target_cache_names() -> list[str]:
    compress_ratios = [1, 1, *(4 if layer % 2 == 0 else 128 for layer in range(2, 43))]
    layer_names: list[str] = []
    for layer, compress_ratio in enumerate(compress_ratios):
        prefix = f"model.layers.{layer}.self_attn"
        layer_names.append(f"{prefix}.swa_cache")
        if compress_ratio > 1:
            layer_names.extend(
                (
                    f"{prefix}.attn",
                    f"{prefix}.compressor.state_cache",
                )
            )
        if compress_ratio == 4:
            layer_names.extend(
                (
                    f"{prefix}.indexer.k_cache",
                    f"{prefix}.indexer.compressor.state_cache",
                )
            )
    return layer_names


def test_official_0731_cache_inventory_and_numeric_collisions() -> None:
    target_names = _official_target_cache_names()
    all_names = [*target_names, *DRAFT_LAYERS]
    collision_groups: dict[int, list[str]] = defaultdict(list)
    for layer_name in all_names:
        collision_groups[extract_layer_index(layer_name, 1)].append(layer_name)

    assert len(target_names) == 167
    assert len(all_names) == 170
    assert sum(name.endswith(".swa_cache") for name in target_names) == 43
    assert sum(name.endswith(".self_attn.attn") for name in target_names) == 41
    assert sum(name.endswith(".self_attn.compressor.state_cache") for name in target_names) == 41
    assert sum(name.endswith(".indexer.k_cache") for name in target_names) == 21
    assert sum(name.endswith(".indexer.compressor.state_cache") for name in target_names) == 21
    assert len(collision_groups) == 43
    assert collision_groups[0] == [
        "model.layers.0.self_attn.swa_cache",
        DRAFT_LAYERS[0],
    ]
    assert set(collision_groups[2]) == {*TARGET_LAYER_2_CACHES, DRAFT_LAYERS[2]}
    assert len(collision_groups[2]) == 6
    assert all(len(collision_groups[layer]) == 5 for layer in range(4, 43, 2))
    assert all(len(collision_groups[layer]) == 3 for layer in range(3, 42, 2))


def test_bind_kv_cache_uses_full_names_and_target_before_draft() -> None:
    config = _config()
    names = (*reversed(TARGET_LAYER_2_CACHES), DRAFT_LAYERS[2])
    context = config.compilation_config.static_forward_context
    caches = {name: torch.tensor([index]) for index, name in enumerate(names)}
    for name in names:
        context[name] = _FakeAttentionLayer(_spec())
    runner_caches = []

    with set_current_vllm_config(config):
        attn_utils.bind_kv_cache(caches, context, runner_caches)

    expected_order = [*TARGET_LAYER_2_CACHES, DRAFT_LAYERS[2]]
    assert len(runner_caches) == len(expected_order)
    assert all(runner_cache is caches[layer_name] for runner_cache, layer_name in zip(runner_caches, expected_order))
    assert all(context[name].kv_cache is caches[name] for name in names)
    assert len({id(context[name].kv_cache) for name in names}) == len(names)


@pytest.mark.parametrize("missing_count", [1, 2])
def test_bind_kv_cache_rejects_incomplete_authoritative_registry(
    missing_count: int,
) -> None:
    config = _config()
    names = TARGET_LAYER_2_CACHES[:2]
    context = config.compilation_config.static_forward_context
    for name in names[missing_count:]:
        context[name] = _FakeAttentionLayer(_spec())
    caches = {name: torch.zeros(1) for name in names}
    runner_caches = []

    with (
        set_current_vllm_config(config),
        pytest.raises(RuntimeError, match="absent from static_forward_context"),
    ):
        attn_utils.bind_kv_cache(caches, context, runner_caches)

    assert runner_caches == []


def test_bind_kv_cache_rejects_non_authoritative_registry_object() -> None:
    config = _config()
    name = TARGET_LAYER_2_CACHES[0]
    config.compilation_config.static_forward_context[name] = _FakeAttentionLayer(_spec())

    with (
        set_current_vllm_config(config),
        pytest.raises(RuntimeError, match="authoritative"),
    ):
        attn_utils.bind_kv_cache(
            {name: torch.zeros(1)},
            dict(config.compilation_config.static_forward_context),
            [],
        )


def test_bind_kv_cache_failure_restores_layers_and_runner() -> None:
    config = _config()
    names = TARGET_LAYER_2_CACHES[:2]
    context = config.compilation_config.static_forward_context
    first_layer = _FakeAttentionLayer(_spec())
    failing_layer = _FailingAttentionLayer(_spec())
    context[names[0]] = first_layer
    context[names[1]] = failing_layer
    previous_caches = {name: context[name].kv_cache for name in names}
    caches = {name: torch.tensor([index]) for index, name in enumerate(names)}
    failing_layer.rejected_cache = caches[names[1]]
    runner_caches = []

    with (
        set_current_vllm_config(config),
        pytest.raises(RuntimeError, match="cache installation failed"),
    ):
        attn_utils.bind_kv_cache(caches, context, runner_caches)

    assert runner_caches == []
    assert all(context[name].kv_cache is previous_caches[name] for name in names)


def test_bind_kv_cache_preserves_non_dspark_numeric_order() -> None:
    config = _config()
    config.speculative_config = None
    names = (
        "model.layers.1.self_attn",
        "model.layers.0.self_attn",
    )
    context = config.compilation_config.static_forward_context
    caches = {name: torch.tensor([index]) for index, name in enumerate(names)}
    for name in names:
        context[name] = _FakeAttentionLayer(_spec())
    runner_caches = []

    with set_current_vllm_config(config):
        attn_utils.bind_kv_cache(caches, context, runner_caches)

    assert runner_caches[0] is caches[names[1]]
    assert runner_caches[1] is caches[names[0]]


def test_v2_patch_installs_full_name_binder_at_core_call_site() -> None:
    from vllm.v1.worker.gpu import attn_utils as core_attn_utils

    import vllm_ascend.patch.worker.patch_v2.patch_attn_utils  # noqa: F401

    assert core_attn_utils.bind_kv_cache is attn_utils.bind_kv_cache


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
    assert raw_caches[DRAFT_LAYERS[0]].untyped_storage().nbytes() == allocation_size
    first_page_bytes = first_cache[0].numel() * first_cache.element_size()
    second_page_bytes = second_cache[0].numel() * second_cache.element_size()
    assert first_page_bytes == second_page_bytes == page_size
    for block_index in range(kv_cache_config.num_blocks):
        first_range = (
            block_index * block_stride,
            block_index * block_stride + page_size,
        )
        second_range = (
            block_index * block_stride + page_size,
            block_index * block_stride + 2 * page_size,
        )
        assert first_range[1] <= second_range[0]
        assert second_range[1] <= allocation_size


def test_dsv4_allocator_rejects_overlapping_packed_pages() -> None:
    config = _config()
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
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=2 * block_stride,
                shared_by=[DRAFT_LAYERS[0]],
                offset=0,
                block_stride=block_stride,
            ),
            KVCacheTensor(
                size=2 * block_stride,
                shared_by=[DRAFT_LAYERS[1]],
                offset=page_size // 2,
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

    with (
        set_current_vllm_config(config),
        pytest.raises(ValueError, match="pages overlap"),
    ):
        attn_utils._allocate_kv_cache(
            kv_cache_config,
            {},
            torch.device("cpu"),
        )


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


def test_acl_graph_manager_forwards_lora_capture_cases_to_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

    core_parameters = signature(ModelCudaGraphManager).parameters
    assert core_parameters["lora_capture_cases"].default is None

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
    )
    runner = object()
    device = torch.device("cpu")
    lora_capture_cases = [0, 2]
    init_call = None

    def core_init(
        manager,
        vllm_config,
        graph_device,
        cudagraph_mode,
        decode_query_len,
        lora_capture_cases=None,
    ):
        nonlocal init_call
        init_call = (
            vllm_config,
            graph_device,
            cudagraph_mode,
            decode_query_len,
            lora_capture_cases,
        )
        manager.compilation_config = vllm_config.compilation_config

    monkeypatch.setattr(ModelCudaGraphManager, "__init__", core_init)
    monkeypatch.setattr(ModelCudaGraphManager, "needs_capture", lambda _self: False)

    manager = ModelAclGraphManager(
        config,
        device,
        CUDAGraphMode.FULL,
        6,
        runner,
        lora_capture_cases=lora_capture_cases,
    )

    assert init_call == (
        config,
        device,
        CUDAGraphMode.FULL,
        6,
        lora_capture_cases,
    )
    assert init_call[-1] is lora_capture_cases
    assert manager.model_runner is runner


@pytest.mark.parametrize(
    ("cudagraph_mode", "use_current_core_abi", "lora_capture_cases"),
    [
        pytest.param(CUDAGraphMode.NONE, False, None, id="eager-legacy"),
        pytest.param(CUDAGraphMode.FULL, True, [0, 2], id="graph-current-lora"),
    ],
)
def test_graph_manager_wrapper_matches_core_constructor_abi(
    monkeypatch: pytest.MonkeyPatch,
    cudagraph_mode: CUDAGraphMode,
    use_current_core_abi: bool,
    lora_capture_cases: list[int] | None,
) -> None:
    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    runner = object()
    config = object()
    device = torch.device("cpu")
    manager = object()
    calls = []

    def create_manager(
        vllm_config,
        graph_device,
        graph_mode,
        decode_query_len,
        model_runner,
        lora_capture_cases=None,
    ):
        calls.append(
            (
                vllm_config,
                graph_device,
                graph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases,
            )
        )
        return manager

    monkeypatch.setattr(ascend_model_runner, "ModelAclGraphManager", create_manager)
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    with ascend_model_runner.graph_manager_wrapper(runner):
        factory = vllm_model_runner.ModelCudaGraphManager
        if use_current_core_abi:
            actual = factory(
                config,
                device,
                cudagraph_mode,
                decode_query_len=6,
                lora_capture_cases=lora_capture_cases,
            )
        else:
            actual = factory(
                config,
                device,
                cudagraph_mode,
                decode_query_len=6,
            )

        assert actual is manager

    assert vllm_model_runner.ModelCudaGraphManager is original_graph_manager
    assert calls == [
        (
            config,
            device,
            cudagraph_mode,
            6,
            runner,
            lora_capture_cases,
        )
    ]


def test_graph_manager_wrapper_restores_constructor_after_exception() -> None:
    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    with (
        pytest.raises(RuntimeError, match="graph initialization failed"),
        ascend_model_runner.graph_manager_wrapper(object()),
    ):
        assert vllm_model_runner.ModelCudaGraphManager is not original_graph_manager
        raise RuntimeError("graph initialization failed")

    assert vllm_model_runner.ModelCudaGraphManager is original_graph_manager


@pytest.mark.parametrize("use_dspark", [False, True], ids=["non-dspark", "dspark"])
@pytest.mark.parametrize(
    "lora_capture_cases",
    [pytest.param([0], id="lora-disabled"), pytest.param([0, 2], id="lora-enabled")],
)
def test_runner_kv_initialization_forwards_lora_capture_cases(
    monkeypatch: pytest.MonkeyPatch,
    use_dspark: bool,
    lora_capture_cases: list[int],
) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    config, speculator, _target_model = _loaded_speculator(monkeypatch)
    kv_cache_config = _kv_cache_config()
    runner = object.__new__(ascend_model_runner.NPUModelRunner)
    runner.speculator = speculator if use_dspark else None
    runner.vllm_config = config
    runner.compilation_config = config.compilation_config
    runner.model_state = object()
    runner.device = torch.device("cpu")
    runner.decode_query_len = 6
    runner.lora_capture_cases = lora_capture_cases
    manager = object()
    manager_calls = []
    set_attn_calls = []

    def create_manager(
        vllm_config,
        device,
        cudagraph_mode,
        decode_query_len,
        model_runner,
        lora_capture_cases=None,
    ):
        manager_calls.append(
            (
                vllm_config,
                device,
                cudagraph_mode,
                decode_query_len,
                model_runner,
                lora_capture_cases,
            )
        )
        return manager

    def core_initialize(_runner, cache_config):
        _runner.kv_cache_config = cache_config
        _runner.block_tables = object()
        _runner.cudagraph_manager = vllm_model_runner.ModelCudaGraphManager(
            _runner.vllm_config,
            _runner.device,
            CUDAGraphMode.NONE,
            decode_query_len=_runner.decode_query_len,
            lora_capture_cases=_runner.lora_capture_cases,
        )

    monkeypatch.setattr(ascend_model_runner, "ModelAclGraphManager", create_manager)
    monkeypatch.setattr(GPUModelRunner, "initialize_kv_cache", core_initialize)
    if use_dspark:
        monkeypatch.setattr(
            speculator,
            "set_attn",
            lambda *args: set_attn_calls.append(args),
        )
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    ascend_model_runner.NPUModelRunner.initialize_kv_cache(
        runner,
        kv_cache_config,
    )

    assert vllm_model_runner.ModelCudaGraphManager is original_graph_manager
    assert runner.cudagraph_manager is manager
    assert manager_calls == [
        (
            config,
            runner.device,
            CUDAGraphMode.NONE,
            6,
            runner,
            lora_capture_cases,
        )
    ]
    assert manager_calls[0][-1] is lora_capture_cases
    assert len(set_attn_calls) == int(use_dspark)


def test_runner_restores_kv_state_when_graph_manager_factory_fails(
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
    runner.vllm_config = config
    runner.compilation_config = config.compilation_config
    runner.model_state = object()
    runner.device = torch.device("cpu")
    runner.decode_query_len = 6
    runner.lora_capture_cases = [0]

    def allocate_then_create_manager(_runner, cache_config):
        _runner.kv_cache_config = cache_config
        _runner.block_tables = object()
        _runner.kv_caches = [torch.zeros(1)]
        for layer in context.values():
            layer.kv_cache = (torch.zeros(1), torch.zeros(1))
        _runner.cudagraph_manager = vllm_model_runner.ModelCudaGraphManager(
            _runner.vllm_config,
            _runner.device,
            CUDAGraphMode.NONE,
            decode_query_len=_runner.decode_query_len,
            lora_capture_cases=_runner.lora_capture_cases,
        )

    def fail_manager_creation(*_args, **_kwargs):
        raise RuntimeError("graph manager factory failed")

    monkeypatch.setattr(
        GPUModelRunner,
        "initialize_kv_cache",
        allocate_then_create_manager,
    )
    monkeypatch.setattr(
        ascend_model_runner,
        "ModelAclGraphManager",
        fail_manager_creation,
    )
    original_graph_manager = vllm_model_runner.ModelCudaGraphManager

    with pytest.raises(RuntimeError, match="graph manager factory failed"):
        ascend_model_runner.NPUModelRunner.initialize_kv_cache(
            runner,
            kv_cache_config,
        )

    assert vllm_model_runner.ModelCudaGraphManager is original_graph_manager
    assert not hasattr(runner, "kv_cache_config")
    assert not hasattr(runner, "block_tables")
    assert not hasattr(runner, "cudagraph_manager")
    assert not hasattr(runner, "kv_caches")
    assert all(context[name].kv_cache is previous_caches[name] for name in previous_caches)
    assert speculator.kv_cache_config is None
    assert speculator.draft_kv_caches is None


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
