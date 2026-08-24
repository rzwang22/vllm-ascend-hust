# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
ASCEND_KV_SPEC = REPO_ROOT / "vllm_ascend/core/kv_cache_interface.py"
COMPRESS_MANAGER = REPO_ROOT / "vllm_ascend/core/single_type_kv_cache_manager.py"
DSA_LAYER = REPO_ROOT / "vllm_ascend/models/layer/attention/layer.py"
V2_ATTN_UTILS = REPO_ROOT / "vllm_ascend/worker/v2/attn_utils.py"
SHARED_KV_TILING = REPO_ROOT / "csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(path: Path, name: str) -> ast.ClassDef:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name} in {path}")


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in _class(path, class_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing {name} in {path}")


def _load_method(path: Path, class_name: str, method_name: str):
    method = _method(path, class_name, method_name)
    method.decorator_list = []
    namespace: dict[str, object] = {}
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def test_ascend_mla_spec_restores_physical_storage_block_size() -> None:
    storage_property = _method(
        ASCEND_KV_SPEC,
        "AscendMLAAttentionSpec",
        "storage_block_size",
    )
    assert any(
        isinstance(decorator, ast.Name) and decorator.id == "property" for decorator in storage_property.decorator_list
    )
    source = ast.unparse(storage_property)
    assert "return self.block_size" in source
    assert "compress_ratio" not in source

    page_size_source = ast.unparse(
        _method(
            ASCEND_KV_SPEC,
            "AscendMLAAttentionSpec",
            "page_size_bytes",
        )
    )
    assert "self.block_size" in page_size_source
    assert "self.storage_block_size" not in page_size_source


def test_old_compress_manager_owns_logical_token_conversion() -> None:
    registration = ast.unparse(_function(ASCEND_KV_SPEC, "register_ascend_kv_cache_specs"))
    assert "kvcache_spec_cls=AscendMLAAttentionSpec" in registration
    assert "manager_class=CompressAttentionManager" in registration

    manager = _class(COMPRESS_MANAGER, "CompressAttentionManager")
    manager_source = ast.unparse(manager)
    assert "num_tokens //= self.compress_ratio" in manager_source
    assert "num_full_blocks = num_tokens // (self.block_size * self.compress_ratio)" in manager_source
    assert "logical_block_size = block_size * kv_cache_spec.compress_ratio" in manager_source


def test_dsa_spec_publishes_physical_block_without_ratio_multiplication() -> None:
    get_spec = ast.unparse(_method(DSA_LAYER, "DSAAttention", "get_kv_cache_spec"))
    assert "block_size=DSV4_BLOCK_SIZES[vllm_config.cache_config.block_size][0][0]" in get_spec
    assert "block_size * self.compress_ratio" not in get_spec
    assert "block_size=self.compress_ratio" not in get_spec


def test_physical_page_and_logical_coverage_contract() -> None:
    physical_block_sizes = (32, 64, 128)
    storage_block_size = _load_method(
        ASCEND_KV_SPEC,
        "AscendMLAAttentionSpec",
        "storage_block_size",
    )
    assert all(block_size % 16 == 0 for block_size in physical_block_sizes)
    assert all(
        storage_block_size(
            SimpleNamespace(block_size=block_size, compress_ratio=128),
        )
        == block_size
        for block_size in physical_block_sizes
    )
    assert 128 * 4 == 512
    assert 128 * 128 == 16384

    view_source = ast.unparse(_function(V2_ATTN_UTILS, "_view_dsv4_cache"))
    assert "kv_cache_spec.storage_block_size" in view_source
    assert ".clone(" not in view_source
    assert ".copy_(" not in view_source


def test_shared_kv_alignment_gate_remains_strict() -> None:
    tiling_source = SHARED_KV_TILING.read_text(encoding="utf-8")
    assert "CheckSingleParaCmpBlockTable" in tiling_source
    assert "static_cast<uint64_t>(cmpBlockSize_) % 16 != 0UL" in tiling_source
    assert "cmp_block_size should be in [1, 1024], and be aligned to 16" in tiling_source
