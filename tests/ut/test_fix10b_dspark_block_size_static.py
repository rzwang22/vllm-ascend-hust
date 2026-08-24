# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT.parent / "vllm-hust"
CORE_KV_SPEC = CORE_ROOT / "vllm/v1/kv_cache_interface.py"
ASCEND_KV_SPEC = REPO_ROOT / "vllm_ascend/core/kv_cache_interface.py"
DSA_MODEL = REPO_ROOT / "vllm_ascend/models/deepseek_v4.py"
DSA_LAYER = REPO_ROOT / "vllm_ascend/models/layer/attention/layer.py"
ASCEND_PLATFORM = REPO_ROOT / "vllm_ascend/platform.py"
ASCEND_UTILS = REPO_ROOT / "vllm_ascend/utils.py"
V2_ATTN_UTILS = REPO_ROOT / "vllm_ascend/worker/v2/attn_utils.py"
LIGHTNING_INDEXER_TILING = (
    REPO_ROOT / "csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_tiling.cpp"
)
PREPARE_ONLY_HARNESS = REPO_ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_proposal_inputs_prepare.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing {name} in {path}")


def _assignment_value(function: ast.FunctionDef, name: str):
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment to {name}")


def test_deepseek_v4_c4_uses_explicit_ascend_physical_page_geometry() -> None:
    refresh_block_size = ast.unparse(_function(ASCEND_UTILS, "refresh_block_size"))
    assert "cache_config.block_size not in [32, 64, 128]" in refresh_block_size
    assert "cache_config.block_size = 32" in refresh_block_size

    update_block_size = ast.unparse(
        _class_method(
            ASCEND_PLATFORM,
            "NPUPlatform",
            "update_block_size_for_backend",
        )
    )
    assert "if cache_config.user_specified_block_size:\n        return" in update_block_size

    block_sizes_function = _function(DSA_LAYER, "get_dsv4_block_sizes")
    block_sizes = _assignment_value(block_sizes_function, "_DSV4_BLOCK_SIZES")
    assert block_sizes[32][0][0] == 32
    assert block_sizes[128][0][0] == 128

    core_storage_block_size = _class_method(CORE_KV_SPEC, "MLAAttentionSpec", "storage_block_size")
    assert "return self.block_size // self.compress_ratio" in ast.unparse(core_storage_block_size)
    ascend_storage_block_size = _class_method(
        ASCEND_KV_SPEC,
        "AscendMLAAttentionSpec",
        "storage_block_size",
    )
    assert "return self.block_size" in ast.unparse(ascend_storage_block_size)
    assert block_sizes[32][0][0] == 32
    assert block_sizes[128][0][0] == 128

    indexer_spec = ast.unparse(
        _class_method(
            DSA_MODEL,
            "AscendDeepseekV4IndexerCache",
            "get_kv_cache_spec",
        )
    )
    assert "_dsv4_block_sizes()[vllm_config.cache_config.block_size][0][0]" in indexer_spec
    assert "compress_ratio=self.compress_ratio" in indexer_spec


def test_v2_indexer_view_uses_storage_block_size_as_pa_bsnd_dim_one() -> None:
    view_source = ast.unparse(_function(V2_ATTN_UTILS, "_view_dsv4_cache"))
    assert "attn_backend.get_kv_cache_shape(num_blocks, kv_cache_spec.storage_block_size" in view_source

    tiling_source = LIGHTNING_INDEXER_TILING.read_text(encoding="utf-8")
    assert "GetStorageShape().GetDim(1)" in tiling_source
    assert "blockSize_ % BLOCK_SIZE_FACTOR" in tiling_source
    assert "constexpr uint32_t BLOCK_SIZE_FACTOR = 16" in (
        LIGHTNING_INDEXER_TILING.with_suffix(".h").read_text(encoding="utf-8")
    )


def test_prepare_only_is_the_only_harness_that_requests_128_cache_blocks() -> None:
    harness_tree = _tree(PREPARE_ONLY_HARNESS)
    constants = {
        target.id: node.value.value
        for node in harness_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["PREPARE_ONLY_CACHE_BLOCK_SIZE"] == 128

    test_function = _function(
        PREPARE_ONLY_HARNESS,
        "test_dspark_proposal_inputs_prepare_only_npu",
    )
    engine_args_calls = [
        node
        for node in ast.walk(test_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "EngineArgs"
    ]
    assert len(engine_args_calls) == 1
    block_size_keywords = [keyword.value for keyword in engine_args_calls[0].keywords if keyword.arg == "block_size"]
    assert len(block_size_keywords) == 1
    assert isinstance(block_size_keywords[0], ast.Name)
    assert block_size_keywords[0].id == "PREPARE_ONLY_CACHE_BLOCK_SIZE"

    other_harnesses = (
        REPO_ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_model_loading.py",
        REPO_ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_kv_cache_init.py",
    )
    assert all(
        "block_size=PREPARE_ONLY_CACHE_BLOCK_SIZE" not in path.read_text(encoding="utf-8") for path in other_harnesses
    )


def test_prepare_only_asserts_config_spec_runtime_shape_dtype_and_identity() -> None:
    source = PREPARE_ONLY_HARNESS.read_text(encoding="utf-8")
    required_contracts = (
        "vllm_config.cache_config.block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE",
        "spec.block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE",
        "spec.storage_block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE",
        "primary_cache.shape[1] == spec.storage_block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE",
        "indexer_scale_cache.shape[1] == spec.storage_block_size",
        "indexer_k_cache.dtype == torch.int8",
        "indexer_scale_cache.dtype == torch.float16",
        "forward_kv_cache[4] is primary_cache",
        "forward_kv_cache[5] is indexer_scale_cache",
        "forward_kv_cache[0] is primary_cache",
        '"compress_ratio": spec.compress_ratio',
        "DSPARK_INDEXER_BLOCK_SIZE_CONTRACT=",
        "DSPARK_COMPRESSED_KV_CONTRACT=",
    )
    assert all(contract in source for contract in required_contracts)
