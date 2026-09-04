# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

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
ATTN_UTILS = ROOT / "vllm_ascend/worker/v2/attn_utils.py"


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    required_names = {name}
    if name == "_target_only_kv_topology":
        required_names.add("_group_specs_by_equality")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in required_names]
    assert {function.name for function in functions} == required_names
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"Any": object, "VllmConfig": object}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _load_class_method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


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
    assert "speculator.reconcile_scheduler_proposal(" in runner_source
    assert "finished_request_ids=scheduler_output.finished_req_ids" in runner_source
    assert "finally:\n            super().finish_requests" in runner_source


def test_batched_proposal_ownership_uses_explicit_row_bijection() -> None:
    speculator_source = SPECULATOR.read_text(encoding="utf-8")
    verification_rows = _load_class_method(
        SPECULATOR,
        "AscendDSparkSpeculator",
        "_verification_to_published_rows",
    )

    assert "def _verification_to_published_rows(" in speculator_source
    assert "published_rows[request_id] = row" in speculator_source
    assert "verification_rows.append(published_row)" in speculator_source
    assert "verification_candidate_tokens = candidate_tokens.index_select(" in (speculator_source)
    assert "expected_request_state_indices = request_state_indices.index_select(" in (speculator_source)
    assert "tuple(input_batch.req_ids) != request_ids" not in speculator_source
    assert "with -1 placeholders" in speculator_source
    assert verification_rows(
        ("request-1", "request-2", "request-3", "request-4"),
        ("request-4", "request-3", "request-1", "request-2"),
    ) == (3, 2, 0, 1)
    with pytest.raises(RuntimeError, match="duplicate request ownership"):
        verification_rows(
            ("request-1", "request-2"),
            ("request-1", "request-1"),
        )
    with pytest.raises(RuntimeError, match="does not match"):
        verification_rows(
            ("request-1", "request-2"),
            ("request-1", "request-extra"),
        )


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


def test_target_only_topology_exposes_invalid_equivalent_raw_specs() -> None:
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


def test_target_only_topology_preserves_distinct_c4_c128_specs() -> None:
    topology = _load_function(
        MULTI_ROUND_HARNESS,
        "_target_only_kv_topology",
    )
    groups = [
        SimpleNamespace(
            kv_cache_spec=SimpleNamespace(
                block_size=128,
                storage_block_size=128,
                compress_ratio=compress_ratio,
            ),
            layer_names=[f"model.layers.{index}.self_attn.attn"],
            is_eagle_group=False,
        )
        for index, compress_ratio in enumerate((4, 128))
    ]

    class _AscendHybridKVCacheCoordinator:
        single_type_managers = (SimpleNamespace(), SimpleNamespace())

    runtime = SimpleNamespace(
        launch=SimpleNamespace(rank=0),
        target_kv_cache_layer_names=tuple(name for group in groups for name in group.layer_names),
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
            coordinator=_AscendHybridKVCacheCoordinator(),
        )
    )

    result = topology(runtime, scheduler)

    assert result["raw_group_count"] == 2
    assert result["unique_attention_group_count"] == 2
    assert result["spec_equality_groups"] == [[0], [1]]
    assert result["compress_ratios"] == [4, 128]
    assert result["coordinator_class"] == "_AscendHybridKVCacheCoordinator"


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


@pytest.mark.parametrize("method", [None, "dspark", "mtp", "eagle", "dflash"])
def test_dsv4_kv_lifecycle_detection_is_model_specific(
    method: str | None,
) -> None:
    uses_dsv4 = _load_function(
        ATTN_UTILS,
        "_uses_ascend_dsv4_kv_lifecycle",
    )
    speculative_config = None if method is None else SimpleNamespace(method=method)
    dsv4_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[1, 4, 128]),
        ),
        speculative_config=speculative_config,
    )
    ordinary_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        speculative_config=speculative_config,
    )

    assert uses_dsv4(dsv4_config) is True
    assert uses_dsv4(ordinary_config) is False


def test_target_only_diagnostic_runs_before_scheduler_dispatch() -> None:
    source = MULTI_ROUND_HARNESS.read_text(encoding="utf-8")
    run_start = source.index("def _run_target_only_generation(")
    run_end = source.index("\ndef ", run_start + 1)
    run_source = source[run_start:run_end]
    runtime_start = source.index("def _target_only_runtime(")
    runtime_end = source.index("\ndef ", runtime_start + 1)
    runtime_source = source[runtime_start:runtime_end]

    assert runtime_source.index("_target_only_kv_topology_pre_dispatch(") < runtime_source.index(
        "worker.initialize_from_config(kv_cache_config)"
    )
    assert "DSPARK_M2_4B_TARGET_KV_TOPOLOGY_PRE_DISPATCH=" in runtime_source
    assert run_source.index("runtime.target_only_kv_topology_pre_dispatch") < run_source.index(
        "_build_scheduler(runtime)"
    )
    for field in (
        "raw_group_count_before",
        "normalized_group_count",
        "unique_group_count_after",
        "selected_coordinator",
        "layer_set_preserved",
        "layer_multiplicity_preserved",
    ):
        assert field in source


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"draft_layer_count": 1}, "draft/MTP KV cache owners"),
        ({"layer_set_preserved": False}, "target KV owner set"),
        ({"layer_multiplicity_preserved": False}, "owner multiplicity"),
        (
            {"dispatch_route": "invalid-equivalent-multi-group"},
            "multiple equivalent KV groups",
        ),
    ],
)
def test_target_only_pre_dispatch_validation_rejects_invalid_topology(
    mutation: dict[str, object],
    message: str,
) -> None:
    validate = _load_function(
        MULTI_ROUND_HARNESS,
        "_validate_target_only_kv_topology_pre_dispatch",
    )
    topology = {
        "draft_layer_count": 0,
        "layer_set_preserved": True,
        "layer_multiplicity_preserved": True,
        "dispatch_route": "ascend-hybrid",
    }

    validate(topology)
    topology.update(mutation)
    with pytest.raises(RuntimeError, match=message):
        validate(topology)
