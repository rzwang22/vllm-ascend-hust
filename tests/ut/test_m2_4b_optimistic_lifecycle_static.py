# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPECULATOR = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
PROPOSAL_INPUTS = ROOT / ("vllm_ascend/worker/v2/spec_decode/dspark/proposal_inputs.py")
MODEL_RUNNER = ROOT / "vllm_ascend/worker/v2/model_runner.py"
PREPARE_HARNESS = ROOT / ("tests/e2e/nightly/single_node/spec_decode/test_dspark_proposal_inputs_prepare.py")
MULTI_ROUND_HARNESS = ROOT / ("tests/e2e/nightly/single_node/spec_decode/test_dspark_multi_round_generation.py")


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_active_scheduler_commit_must_equal_raw_verified_tokens() -> None:
    validate = _load_function(
        MULTI_ROUND_HARNESS,
        "_validate_scheduler_commit",
    )

    assert validate([11, 12], [11, 12], request_finished=False)
    with pytest.raises(RuntimeError, match="active request cannot diverge"):
        validate([11, 12], [11], request_finished=False)


def test_finished_scheduler_commit_must_be_a_raw_prefix() -> None:
    validate = _load_function(
        MULTI_ROUND_HARNESS,
        "_validate_scheduler_commit",
    )

    assert validate([11, 12], [11], request_finished=True) is False
    with pytest.raises(RuntimeError, match="outside the raw verified prefix"):
        validate([11, 12], [12], request_finished=True)


def test_proposal_lifecycle_fields_and_terminal_cleanup_are_explicit() -> None:
    lifecycle_source = PROPOSAL_INPUTS.read_text(encoding="utf-8")
    speculator_source = SPECULATOR.read_text(encoding="utf-8")
    runner_source = MODEL_RUNNER.read_text(encoding="utf-8")

    for field in (
        "proposal_epoch",
        "owner_epoch",
        "consumer_epoch",
        "generated",
        "returned_to_core",
        "installed",
        "consumed",
        "discarded_terminal",
    ):
        assert field in lifecycle_source
    assert "def discard_terminal_proposal(" in speculator_source
    assert "_terminal_proposal_discard_count += 1" in speculator_source
    assert "speculator.discard_terminal_proposal(" in runner_source
    assert "finally:\n            super().finish_requests" in runner_source


def test_single_round_and_multi_round_harnesses_select_distinct_modes() -> None:
    prepare_source = PREPARE_HARNESS.read_text(encoding="utf-8")
    multi_source = MULTI_ROUND_HARNESS.read_text(encoding="utf-8")

    assert "_CONTINUE_AFTER_VERIFICATION = False" in prepare_source
    assert '"dspark_continue_after_verification"' in prepare_source
    assert "prepare_harness._CONTINUE_AFTER_VERIFICATION = True" in multi_source
    assert "prepare_harness._CONTINUE_AFTER_VERIFICATION = previous_mode" in (multi_source)
