# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

import pytest

from tests.e2e.nightly.single_node.spec_decode.test_dspark_single_round_verification import (
    _expected_greedy_verification,
)

PLUGIN_ROOT = Path(__file__).parents[2]
CORE_ROOT = PLUGIN_ROOT.parent / "vllm-hust"
SPECULATOR = PLUGIN_ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
CORE_SPECULATOR = CORE_ROOT / "vllm/v1/worker/gpu/spec_decode/speculator.py"
CORE_RUNNER = CORE_ROOT / "vllm/v1/worker/gpu/model_runner.py"
CORE_HANDLER = CORE_ROOT / "vllm/v1/worker/gpu/spec_decode/utils.py"
NPU_HARNESS = PLUGIN_ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_single_round_verification.py"


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def test_core_optional_proposal_abi_and_empty_publication_are_present() -> None:
    propose = _method(CORE_SPECULATOR, "BaseSpeculator", "propose")
    publish = ast.unparse(_method(CORE_RUNNER, "GPUModelRunner", "_publish_draft_tokens"))
    handler = CORE_HANDLER.read_text(encoding="utf-8")

    assert propose.returns is not None
    assert ast.unparse(propose.returns) == "torch.Tensor | None"
    assert "[:input_batch.num_reqs, :0]" in publish
    assert "if draft_tokens is None" in publish
    assert "[[] for _ in self.req_ids]" in handler


def test_dspark_publication_uses_exact_markov_candidate_tensor() -> None:
    method = ast.unparse(_method(SPECULATOR, "AscendDSparkSpeculator", "_build_core_proposal"))

    assert "result is not self._markov_result" in method
    assert "candidate_tokens = self._validate_step_tensor" in method
    assert "self._published_candidate_tokens = candidate_tokens" in method
    assert "return candidate_tokens" in method
    assert ".cpu(" not in method
    assert ".tolist(" not in method
    assert ".item(" not in method


def test_consumer_validates_scheduled_candidates_then_returns_none() -> None:
    method = ast.unparse(
        _method(
            SPECULATOR,
            "AscendDSparkSpeculator",
            "_skip_next_proposal_after_verification",
        )
    )

    assert "num_draft_tokens_per_req" in method
    assert "consumed_candidates" in method
    assert "candidate_tokens" in method
    assert "self._published_proposal_consumed = True" in method
    assert "self._next_proposal_skipped = True" in method
    assert method.rstrip().endswith("return None")
    assert "_execute_draft_backbone" not in method
    assert "_execute_sequential_markov_sampling" not in method


def test_third_proposal_stops_at_m2_4b_without_plugin_verifier() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")

    assert 'dspark_runtime_not_wired("M2.4B multi-round DSpark lifecycle")' in source
    assert "RejectionSampler" not in source
    assert "rejection_sample(" not in source
    assert "scheduler.update_from_output" not in source


def test_npu_harness_drives_real_scheduler_worker_and_core_sampler() -> None:
    source = NPU_HARNESS.read_text(encoding="utf-8")

    assert "Scheduler(" in source
    assert "scheduler.schedule()" in source
    assert source.count("worker.execute_model(") == 2
    assert source.count("worker.sample_tokens(None)") == 2
    assert source.count("scheduler.update_from_output(") == 2
    assert "scheduler.update_draft_token_ids(" in source
    assert "rejection_sample(" not in source
    assert "RejectionSampler" not in source
    assert "speculator._execute_draft(" not in source


@pytest.mark.parametrize(
    ("target_selected", "expected", "accepted", "replacement", "bonus"),
    [
        ([90, 11, 12, 13, 14, 15], [90], 0, True, False),
        ([10, 11, 92, 13, 14, 15], [10, 11, 92], 2, True, False),
        ([10, 11, 12, 13, 14, 95], [10, 11, 12, 13, 14, 95], 5, False, True),
    ],
)
def test_greedy_verification_contract_covers_rejection_and_bonus(
    target_selected: list[int],
    expected: list[int],
    accepted: int,
    replacement: bool,
    bonus: bool,
) -> None:
    result = _expected_greedy_verification(
        [10, 11, 12, 13, 14],
        target_selected,
    )

    assert result == (expected, accepted, replacement, bonus)
