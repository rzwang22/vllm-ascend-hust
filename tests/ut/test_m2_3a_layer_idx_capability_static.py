# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
UTILS = REPO_ROOT / "vllm_ascend/utils.py"
FORWARD_CONTEXT = REPO_ROOT / "vllm_ascend/ascend_forward_context.py"
TARGET_MODEL = REPO_ROOT / "vllm_ascend/models/deepseek_v4.py"
DRAFT_MODEL = REPO_ROOT / "vllm_ascend/models/deepseek_v4_dspark.py"


def _function(path: Path, name: str) -> tuple[str, ast.FunctionDef]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return source, function


def test_layer_idx_capability_has_no_process_global_cache() -> None:
    source, function = _function(UTILS, "has_layer_idx")
    function_source = ast.get_source_segment(source, function)

    assert "_HAS_LAYER_IDX" not in source
    assert not any(isinstance(node, ast.Global) for node in ast.walk(function))
    assert 'getattr(model_instance, "model", None)' in function_source
    assert 'hasattr(model, "start_layer")' in function_source


def test_forward_context_reads_start_layer_only_after_instance_check() -> None:
    source, function = _function(
        FORWARD_CONTEXT,
        "build_ascend_forward_context",
    )
    function_source = ast.get_source_segment(source, function)

    capability_check = "if has_layer_idx(model_instance):"
    start_layer_read = "layer_idx = model_instance.model.start_layer"
    assert function_source.index(capability_check) < function_source.index(start_layer_read)


def test_target_and_draft_keep_distinct_layer_index_namespaces() -> None:
    target_source = TARGET_MODEL.read_text(encoding="utf-8")
    draft_source = DRAFT_MODEL.read_text(encoding="utf-8")

    assert "self.start_layer, self.end_layer, self.layers = make_layers(" in target_source
    assert "self.mtp_start_layer_idx = config.num_hidden_layers" in draft_source
    assert "self.start_layer" not in draft_source
