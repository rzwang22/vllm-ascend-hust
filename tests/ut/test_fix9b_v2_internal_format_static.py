# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V1_RUNNER = REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py"
V2_RUNNER = REPO_ROOT / "vllm_ascend/worker/v2/model_runner.py"
W8A8_METHOD = REPO_ROOT / "vllm_ascend/quantization/methods/w8a8_dynamic.py"
W8A8_NPU_TEST = REPO_ROOT / "tests/e2e/nightly/single_node/ops/singlecard_ops/test_w8a8_moe_910b2.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _is_allow_internal_format_target(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "allow_internal_format"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "npu"
        and isinstance(node.value.value.value, ast.Name)
        and node.value.value.value.id == "torch"
    )


def _module_level_internal_format_assignments(path: Path) -> list[ast.Assign]:
    tree = ast.parse(_source(path))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and _is_allow_internal_format_target(node.targets[0])
    ]


def test_v1_and_v2_enable_internal_formats_before_runner_construction() -> None:
    for path in (V1_RUNNER, V2_RUNNER):
        assignments = _module_level_internal_format_assignments(path)
        assert len(assignments) == 1
        assignment = assignments[0]
        assert isinstance(assignment.value, ast.Constant)
        assert assignment.value.value is True

        tree = ast.parse(_source(path))
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
        assert assignment.lineno < runner.lineno


def test_w8a8_post_load_persists_the_batched_nz_cast_results() -> None:
    source = _source(W8A8_METHOD)
    assert "layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)" in source
    assert "layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)" in source
    assert "layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()" in source
    assert "layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()" in source


def test_npu_probe_requires_real_five_dimensional_nz_storage() -> None:
    source = _source(W8A8_NPU_TEST)
    assert "torch.ops._C_ascend.get_npu_storage_shape(layer.w13_weight)" in source
    assert "torch.ops._C_ascend.get_npu_storage_shape(layer.w2_weight)" in source
    assert "assert w13_storage_shape == (" in source
    assert "assert w2_storage_shape == (" in source
    assert "(2 * INTERMEDIATE_SIZE) // 32" in source
    assert "HIDDEN_SIZE // 16" in source
    assert "HIDDEN_SIZE // 32" in source
    assert "INTERMEDIATE_SIZE // 16" in source
