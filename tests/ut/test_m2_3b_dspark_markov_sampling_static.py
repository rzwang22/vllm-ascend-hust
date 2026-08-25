# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
SPECULATOR = REPO_ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
PROPOSAL_INPUTS = REPO_ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/proposal_inputs.py"
MODEL = REPO_ROOT / "vllm_ascend/models/deepseek_v4_dspark.py"
HARNESS = REPO_ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_markov_sampling.py"


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _calls(node: ast.AST) -> list[str]:
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def test_production_runs_real_sequential_heads_and_global_argmax() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_execute_sequential_markov_sampling",
    )
    calls = _calls(method)
    method_source = ast.get_source_segment(source, method)

    assert calls.count("compute_draft_logits") == 1
    assert "markov_embed" in calls
    assert "markov_bias" in calls
    assert "argmax" in calls
    assert "map_draft_to_target" in calls
    assert "range(num_speculative_tokens)" in method_source
    assert "predecessor = selected" in method_source
    assert "gumbel" not in method_source.lower()
    assert "seeds" not in method_source
    assert ".item(" not in method_source
    assert ".tolist(" not in method_source
    assert "synchronize" not in method_source


def test_markov_result_is_epoch_owned_and_published_only_after_all_steps() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = _class_method(
        tree,
        "AscendDSparkSpeculator",
        "_execute_sequential_markov_sampling",
    )
    assignments = [node for node in method.body if isinstance(node, (ast.Assign, ast.AnnAssign))]
    source_segments = [ast.get_source_segment(source, node) for node in assignments]

    assert "class AscendDSparkMarkovResult" in PROPOSAL_INPUTS.read_text(encoding="utf-8")
    assert any("self._markov_result = None" in segment for segment in source_segments)
    assert source.rfind("self._markov_result = result") < source.rfind("return result")
    assert "_build_core_proposal" in source


def test_model_adapters_delegate_loaded_full_vocab_modules() -> None:
    source = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    draft_logits = _class_method(
        tree,
        "DSparkDeepseekV4ForCausalLM",
        "compute_draft_logits",
    )
    vocab_map = _class_method(
        tree,
        "DSparkDeepseekV4ForCausalLM",
        "map_draft_to_target",
    )

    assert "compute_logits" in _calls(draft_logits)
    assert "get_vocab_size" in _calls(vocab_map)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"clone", "reshape", "view"}
        for node in ast.walk(vocab_map)
    )


def test_harness_has_five_step_records_and_stops_before_publication() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert "speculator._execute_sequential_markov_sampling" in source
    assert "torch.distributed.all_gather" in source
    assert "DSPARK_MARKOV_STEP=" in source
    assert "DSPARK_MARKOV_CONTRACT=" in source
    assert "MARKOV_SAMPLING_ONLY_PASS" in source
    assert '"proposal_publication": False' in source
    assert '"verification": False' in source
    assert '"generation": False' in source
    assert "speculator.propose(" not in source


def test_no_forbidden_runtime_or_scheduler_publication_imports() -> None:
    production = SPECULATOR.read_text(encoding="utf-8")

    assert "vllm.v1.worker.gpu.spec_decode.dspark" not in production
    assert "vllm_ascend.ops.triton.spec_decode" not in production
    assert "scheduler" not in production.lower()
    assert "rejection" not in production.lower()
