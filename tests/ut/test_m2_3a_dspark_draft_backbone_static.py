# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SPECULATOR = REPO_ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
PREPARE_HARNESS = REPO_ROOT / "tests/e2e/nightly/single_node/spec_decode" / "test_dspark_proposal_inputs_prepare.py"
DRAFT_FORWARD_HARNESS = (
    REPO_ROOT / "tests/e2e/nightly/single_node/spec_decode" / "test_dspark_draft_backbone_forward.py"
)


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _calls(node: ast.AST) -> set[str]:
    names = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


def test_execute_draft_runs_backbone_before_markov_fail_closed() -> None:
    tree = ast.parse(SPECULATOR.read_text(encoding="utf-8"))
    method = _class_method(tree, "AscendDSparkSpeculator", "_execute_draft")
    calls = _calls(method)
    source = ast.get_source_segment(SPECULATOR.read_text(encoding="utf-8"), method)

    assert "_execute_draft_backbone" in calls
    assert "dspark_runtime_not_wired" in calls
    assert '"V2 DSpark Markov sampling"' in source
    assert '"V2 draft execution"' not in source
    assert not any(isinstance(node, ast.Return) for node in ast.walk(method))


def test_backbone_helper_runs_real_context_metadata_and_wrapper_forward() -> None:
    source_text = SPECULATOR.read_text(encoding="utf-8")
    tree = ast.parse(source_text)
    orchestrator = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_execute_draft_backbone",
    )
    precompute = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_combine_and_precompute_draft_context",
    )
    metadata = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_build_draft_forward_metadata",
    )
    forward = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_run_draft_model_forward",
    )

    assert {
        "_combine_and_precompute_draft_context",
        "_build_draft_forward_metadata",
        "_run_draft_model_forward",
    }.issubset(_calls(orchestrator))
    assert {"cat", "combine_hidden_states", "precompute_and_store_context_kv"}.issubset(_calls(precompute))
    assert "build_attn_metadata" in _calls(metadata)
    assert {"set_forward_context", "build_ascend_forward_context", "model"}.issubset(_calls(forward))
    assert "synchronize" not in _calls(precompute)
    assert "synchronize" not in _calls(metadata)
    assert "synchronize" not in _calls(forward)


def test_production_does_not_wire_markov_or_proposal_tokens() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    assert "_sample_sequential" not in source
    assert "markov_bias(" not in source
    assert "markov_embed(" not in source
    assert 'dspark_runtime_not_wired("V2 DSpark Markov sampling")' in source
    assert "vllm.v1.worker.gpu.spec_decode.dspark" not in source
    assert "vllm_ascend.ops.triton.spec_decode" not in source


def test_prepare_only_harness_keeps_original_entry_and_acceptance_markers() -> None:
    source = PREPARE_HARNESS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_dspark_proposal_inputs_prepare_only_npu"
    )
    function_source = ast.get_source_segment(source, function)

    assert "PREPARE_ONLY_PASS" in function_source
    assert "PER_STEP_KV_METADATA_UPDATED" in function_source
    assert "speculator.prepare_proposal_inputs" in function_source
    assert "block_size=PREPARE_ONLY_CACHE_BLOCK_SIZE" in function_source


def test_draft_forward_harness_uses_production_phases_and_syncs_only_in_test() -> None:
    source = DRAFT_FORWARD_HARNESS.read_text(encoding="utf-8")
    assert "test_dspark_draft_backbone_forward_only_npu" in source
    assert "speculator._combine_and_precompute_draft_context" in source
    assert "speculator._build_draft_forward_metadata" in source
    assert "speculator._run_draft_model_forward" in source
    assert source.count("torch.npu.synchronize()") == 2
    assert "DRAFT_CONTEXT_KV_PRECOMPUTED" in source
    assert "DRAFT_FORWARD_COMPLETED" in source
    assert "DRAFT_FORWARD_ONLY_PASS" in source
    assert "DSPARK_DRAFT_FORWARD_CONTRACT=" in source
