# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
DSA_PATH = REPO_ROOT / "vllm_ascend/ops/dsa.py"
V1_RUNNER_PATH = REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py"
V2_ATTN_UTILS_PATH = REPO_ROOT / "vllm_ascend/worker/v2/attn_utils.py"


class _DeviceTypes:
    A5 = object()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing {name} in {path}")


def _load_dependency_free_function(path: Path, name: str):
    function = _function(path, name)
    namespace: dict[str, object] = {}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _load_build_kv_cache(device_type):
    functions = [
        _function(DSA_PATH, "_build_kv_cache"),
        _function(DSA_PATH, "unfold_kvcache"),
    ]
    namespace = {
        "AscendDeviceType": _DeviceTypes,
        "get_ascend_device_type": lambda: device_type,
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(DSA_PATH), "exec"), namespace)
    return namespace["_build_kv_cache"]


def _owner(indexer_cache):
    caches = {name: object() for name in ("compress", "swa", "state", "indexer_state")}
    owner = SimpleNamespace(
        compress_ratio=4,
        dsa_attn=SimpleNamespace(kv_cache=[caches["compress"]]),
        swa_cache_layer=SimpleNamespace(kv_cache=[caches["swa"]]),
        compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=[caches["state"]])),
        indexer=SimpleNamespace(
            compressor=SimpleNamespace(state_cache=SimpleNamespace(kv_cache=[caches["indexer_state"]])),
            k_cache=SimpleNamespace(kv_cache=indexer_cache),
        ),
    )
    return owner, caches


def test_unfold_kvcache_supports_v2_direct_and_v1_singleton_containers() -> None:
    unfold_kvcache = _load_dependency_free_function(DSA_PATH, "unfold_kvcache")
    k_cache = object()
    scale_cache = object()
    full_cache = object()

    direct = [k_cache, scale_cache]
    nested = [[k_cache, scale_cache]]
    direct_a5 = [k_cache, scale_cache, full_cache]
    nested_a5 = [[k_cache, scale_cache, full_cache]]

    assert unfold_kvcache(direct) is direct
    assert unfold_kvcache(nested) is nested[0]
    assert unfold_kvcache(direct_a5) is direct_a5
    assert unfold_kvcache(nested_a5) is nested_a5[0]


def test_build_kv_cache_preserves_direct_and_nested_component_identity() -> None:
    non_a5 = object()
    build_kv_cache = _load_build_kv_cache(non_a5)

    for nested in (False, True):
        k_cache = object()
        scale_cache = object()
        components = [k_cache, scale_cache]
        owner, caches = _owner([components] if nested else components)

        result = build_kv_cache(owner, SimpleNamespace())

        assert result == (
            caches["compress"],
            caches["swa"],
            caches["state"],
            caches["indexer_state"],
            k_cache,
            scale_cache,
        )


def test_build_kv_cache_preserves_a5_three_component_identity() -> None:
    build_kv_cache = _load_build_kv_cache(_DeviceTypes.A5)

    for nested in (False, True):
        k_cache = object()
        scale_cache = object()
        full_cache = object()
        components = [k_cache, scale_cache, full_cache]
        owner, caches = _owner([components] if nested else components)

        result = build_kv_cache(owner, SimpleNamespace())

        assert result == (
            caches["compress"],
            caches["swa"],
            caches["state"],
            caches["indexer_state"],
            k_cache,
            scale_cache,
            full_cache,
        )


def test_build_kv_cache_unfolds_indexer_container_before_component_unpacking() -> None:
    function = _function(DSA_PATH, "_build_kv_cache")
    indexer_container = "self.indexer.k_cache.kv_cache"
    direct_unfold_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "unfold_kvcache"
        and len(node.args) == 1
        and ast.unparse(node.args[0]) == indexer_container
    ]
    indexer_subscripts = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Subscript) and ast.unparse(node).startswith(indexer_container)
    ]

    assert len(direct_unfold_calls) == 2
    assert indexer_subscripts == []


def test_v2_binder_is_direct_while_v1_keeps_the_virtual_engine_outer_list() -> None:
    v2_source = V2_ATTN_UTILS_PATH.read_text(encoding="utf-8")
    v1_source = V1_RUNNER_PATH.read_text(encoding="utf-8")

    assert "forward_context[layer_name].kv_cache = kv_caches[layer_name]" in v2_source
    assert "].kv_cache = [kv_cache]" in v1_source


def test_v2_indexer_view_keeps_separate_key_and_scale_components() -> None:
    function = _function(V2_ATTN_UTILS_PATH, "_view_dsv4_cache")
    source = ast.unparse(function)

    assert "cache_shapes.append(scale_shape)" in source
    assert "cache_dtypes.append(scale_dtype)" in source
    assert "return _adjust_dsv4_kv_layout" in source
