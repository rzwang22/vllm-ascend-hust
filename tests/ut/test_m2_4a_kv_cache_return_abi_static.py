# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANAGER_PATH = ROOT / "vllm_ascend/core/single_type_kv_cache_manager.py"
COORDINATOR_PATH = ROOT / "vllm_ascend/patch/platform/patch_kv_cache_coordinator.py"
REGRESSION_PATH = ROOT / "tests/ut/test_compressed_prefix_cache.py"


def _class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def test_compressed_manager_has_explicit_integer_return_contract() -> None:
    manager = _class_node(MANAGER_PATH, "CompressAttentionManager")
    cache_blocks = next(
        node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "cache_blocks"
    )

    assert isinstance(cache_blocks.returns, ast.Name)
    assert cache_blocks.returns.id == "int"
    return_values = [node.value for node in ast.walk(cache_blocks) if isinstance(node, ast.Return)]
    assert any(isinstance(value, ast.Constant) and value.value == 0 for value in return_values)
    assert any(isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub) for value in return_values)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(cache_blocks))
    assert not any(isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) for node in ast.walk(cache_blocks))


def test_ascend_hybrid_coordinator_uses_core_return_abi_without_override() -> None:
    coordinator = _class_node(
        COORDINATOR_PATH,
        "AscendHybridKVCacheCoordinator",
    )

    assert any(isinstance(base, ast.Name) and base.id == "HybridKVCacheCoordinator" for base in coordinator.bases)
    assert not any(isinstance(node, ast.FunctionDef) and node.name == "cache_blocks" for node in coordinator.body)


def test_regression_covers_zero_positive_compressed_and_scheduler_paths() -> None:
    source = REGRESSION_PATH.read_text()

    assert "test_hybrid_cache_blocks_zero_aligned_calls_every_manager" in source
    assert "test_hybrid_cache_blocks_positive_aligned_returns_max_new_blocks" in source
    assert "test_compressed_cache_blocks_returns_new_physical_block_count" in source
    assert "test_scheduler_allocation_boundary_handles_zero_aligned_hybrid_cache" in source
    assert "(1, 4, 128)" in source
    assert '@pytest.mark.parametrize("compress_ratio", [4, 128])' in source
