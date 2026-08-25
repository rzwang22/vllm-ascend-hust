# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPECULATOR = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
PROPOSAL_INPUTS = ROOT / ("vllm_ascend/worker/v2/spec_decode/dspark/proposal_inputs.py")
MODEL_RUNNER = ROOT / "vllm_ascend/worker/v2/model_runner.py"
PREPARE_HARNESS = ROOT / ("tests/e2e/nightly/single_node/spec_decode/test_dspark_proposal_inputs_prepare.py")
MULTI_ROUND_HARNESS = ROOT / ("tests/e2e/nightly/single_node/spec_decode/test_dspark_multi_round_generation.py")
KV_COORDINATOR = ROOT / ("vllm_ascend/patch/platform/patch_kv_cache_coordinator.py")


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"Any": object}
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


def test_target_only_topology_reports_single_unitary_group() -> None:
    topology = _load_function(
        MULTI_ROUND_HARNESS,
        "_target_only_kv_topology",
    )
    spec = SimpleNamespace(
        block_size=128,
        storage_block_size=128,
        compress_ratio=1,
    )
    group = SimpleNamespace(
        kv_cache_spec=spec,
        layer_names=["model.layers.0.self_attn.swa_cache"],
        is_eagle_group=False,
    )

    class _UnitaryCoordinator:
        single_type_managers = (SimpleNamespace(),)

    runtime = SimpleNamespace(
        launch=SimpleNamespace(rank=0),
        target_kv_cache_layer_names=tuple(group.layer_names),
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
            coordinator=_UnitaryCoordinator(),
        )
    )

    result = topology(runtime, scheduler)

    assert result["raw_group_count"] == 1
    assert result["unique_attention_group_count"] == 1
    assert result["spec_equality_groups"] == [[0]]
    assert result["block_sizes"] == [128]
    assert result["storage_block_sizes"] == [128]
    assert result["compress_ratios"] == [1]
    assert result["coordinator_class"] == "_UnitaryCoordinator"
    assert result["contains_draft_groups"] is False


def test_target_only_topology_groups_equivalent_raw_specs() -> None:
    topology = _load_function(
        MULTI_ROUND_HARNESS,
        "_target_only_kv_topology",
    )
    specs = [
        SimpleNamespace(block_size=128, storage_block_size=128, compress_ratio=1),
        SimpleNamespace(block_size=128, storage_block_size=128, compress_ratio=1),
    ]
    groups = [
        SimpleNamespace(
            kv_cache_spec=spec,
            layer_names=[f"model.layers.{index}.self_attn.swa_cache"],
            is_eagle_group=False,
        )
        for index, spec in enumerate(specs)
    ]
    coordinator = SimpleNamespace(
        single_type_managers=(SimpleNamespace(), SimpleNamespace()),
    )
    runtime = SimpleNamespace(
        launch=SimpleNamespace(rank=0),
        target_kv_cache_layer_names=tuple(name for group in groups for name in group.layer_names),
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
            coordinator=coordinator,
        )
    )

    result = topology(runtime, scheduler)

    assert result["raw_group_count"] == 2
    assert result["unique_attention_group_count"] == 1
    assert result["spec_equality_groups"] == [[0, 1]]


def test_target_only_topology_rejects_draft_cache_owners() -> None:
    topology = _load_function(
        MULTI_ROUND_HARNESS,
        "_target_only_kv_topology",
    )
    group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            block_size=128,
            storage_block_size=128,
            compress_ratio=1,
        ),
        layer_names=["mtp.0.self_attn.swa_cache"],
        is_eagle_group=False,
    )
    runtime = SimpleNamespace(
        launch=SimpleNamespace(rank=0),
        target_kv_cache_layer_names=tuple(group.layer_names),
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
            coordinator=SimpleNamespace(single_type_managers=(SimpleNamespace(),)),
        )
    )

    with pytest.raises(RuntimeError, match="draft/MTP KV cache owners"):
        topology(runtime, scheduler)


def test_deepseek_detection_follows_core_topology_guards() -> None:
    source = KV_COORDINATOR.read_text(encoding="utf-8")
    factory = source[source.index("def get_kv_cache_coordinator(") :]

    topology_guard = "if len(kv_cache_config.kv_cache_groups) == 1 or not enable_caching:"
    deepseek_dispatch = "if _is_deepseek_v4_kv_cache_config(kv_cache_config):"
    assert factory.index(topology_guard) < factory.index(deepseek_dispatch)
