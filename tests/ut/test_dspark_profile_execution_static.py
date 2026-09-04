# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SPECULATOR = REPO_ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
MODEL = REPO_ROOT / "vllm_ascend/models/deepseek_v4_dspark.py"
WORKER = REPO_ROOT / "vllm_ascend/worker/worker.py"


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _calls(node: ast.AST) -> list[str]:
    calls = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            calls.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            calls.append(child.func.attr)
    return calls


def test_profile_runs_real_draft_compute_without_proposal_publication() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    profile = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_profile_draft_execution",
    )
    markov = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_profile_markov_heads",
    )
    profile_calls = _calls(profile)
    markov_calls = _calls(markov)

    assert "combine_hidden_states" in profile_calls
    assert "precompute_and_store_context_kv" in profile_calls
    assert "set_forward_context" in profile_calls
    assert "build_ascend_forward_context" in profile_calls
    assert "model" in profile_calls
    assert "compute_draft_logits" in markov_calls
    assert "markov_embed" in markov_calls
    assert "markov_bias" in markov_calls
    assert "argmax" in markov_calls
    assert "_build_core_proposal" not in profile_calls + markov_calls
    assert "prepare_proposal_inputs" not in profile_calls + markov_calls
    assert "synchronize" not in profile_calls + markov_calls


def test_profile_context_projection_runs_without_publishing_kv_slots() -> None:
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = _class_method(
        tree,
        "DeepseekV4DSparkModel",
        "precompute_and_store_context_kv",
    )
    method_source = ast.get_source_segment(source, method)

    assert "_project_shared_kv" in _calls(method)
    assert "_store_standard_swa_kv" in _calls(method)
    assert "context_slot_mapping is None" in method_source
    assert "context_states.numel() == 0 or context_slot_mapping is None" not in method_source


def test_profile_restores_every_proposal_lifecycle_field() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    profile = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_run_profile_without_proposal_state",
    )
    profile_source = ast.get_source_segment(source, profile)

    assert "_DSPARK_PROFILE_PRESERVED_STATE" in profile_source
    assert "finally:" in profile_source
    assert "setattr(self, name, value)" in profile_source
    assert "_profile_draft_execution" in _calls(profile)


def test_fixed_kv_worker_path_still_profiles_the_loaded_models() -> None:
    source = WORKER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = _class_method(tree, "NPUWorker", "determine_available_memory")
    method_source = ast.get_source_segment(source, method)
    fixed_branch = method_source[
        method_source.index("if kv_cache_memory_bytes") : method_source.index("return kv_cache_memory_bytes")
    ]

    assert "self.model_runner.profile_run()" in fixed_branch
    assert method_source.index("self.model_runner.profile_run()") < method_source.index("return kv_cache_memory_bytes")


def test_profile_contract_requires_the_exact_core_dummy_flags() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    propose = _class_method(tree, "AscendDSparkSpeculator", "propose")
    propose_source = ast.get_source_segment(source, propose)

    assert "dummy_run and is_profile and skip_attn_for_dummy_run" in propose_source
    assert "V2 draft execution (dummy/profile)" not in propose_source
    assert "return None" in propose_source
