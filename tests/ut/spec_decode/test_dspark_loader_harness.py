# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json

import pytest

from tests.e2e.nightly.single_node.spec_decode.dspark_loader_harness import (
    CLEANUP_COMPLETE,
    CONFIG_READY,
    LOAD_STAGES,
    HarnessNotConfigured,
    StageTracker,
    enforce_offline_mode,
    forbidden_import_delta,
    parse_launch_context,
    parse_loader_settings,
    run_cleanup_steps,
)


def _checkpoint(tmp_path, name: str, **config):
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "dspark_block_size": 5,
                **config,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loader_settings_skip_when_checkpoints_are_not_configured() -> None:
    with pytest.raises(
        HarnessNotConfigured,
        match="HARNESS_READY; REAL_CHECKPOINT_NOT_VALIDATED",
    ):
        parse_loader_settings({})


def test_loader_settings_parse_only_local_checkpoints(tmp_path) -> None:
    target = _checkpoint(tmp_path, "target", torch_dtype="bfloat16")
    draft = _checkpoint(tmp_path, "draft")

    settings = parse_loader_settings(
        {
            "DSPARK_TARGET_MODEL": str(target),
            "DSPARK_DRAFT_MODEL": str(draft),
            "DSPARK_TP_SIZE": "8",
            "DSPARK_MAX_MODEL_LEN": "4096",
            "DSPARK_DTYPE": "BFLOAT16",
        }
    )

    assert settings.target_model == target.resolve()
    assert settings.draft_model == draft.resolve()
    assert settings.tp_size == 8
    assert settings.max_model_len == 4096
    assert settings.dtype == "bfloat16"
    assert settings.num_speculative_tokens == 5


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"DSPARK_DRAFT_MODEL": None}, "DSPARK_DRAFT_MODEL is required"),
        ({"DSPARK_TP_SIZE": "0"}, "positive integer"),
        ({"DSPARK_MAX_MODEL_LEN": "many"}, "positive integer"),
        ({"DSPARK_DTYPE": "int4"}, "DSPARK_DTYPE must be one of"),
    ],
)
def test_loader_settings_reject_invalid_environment(
    tmp_path,
    updates,
    error,
) -> None:
    target = _checkpoint(tmp_path, "target")
    draft = _checkpoint(tmp_path, "draft")
    environ = {
        "DSPARK_TARGET_MODEL": str(target),
        "DSPARK_DRAFT_MODEL": str(draft),
        "DSPARK_TP_SIZE": "8",
        "DSPARK_MAX_MODEL_LEN": "4096",
    }
    for name, value in updates.items():
        if value is None:
            environ.pop(name)
        else:
            environ[name] = value

    with pytest.raises(ValueError, match=error):
        parse_loader_settings(environ)


def test_loader_settings_reject_remote_id_without_downloading() -> None:
    with pytest.raises(ValueError, match="downloads are forbidden"):
        parse_loader_settings(
            {
                "DSPARK_TARGET_MODEL": "deepseek-ai/not-a-local-path",
                "DSPARK_DRAFT_MODEL": "deepseek-ai/not-a-local-draft",
                "DSPARK_TP_SIZE": "8",
                "DSPARK_MAX_MODEL_LEN": "4096",
            }
        )


def test_loader_settings_reject_mismatched_dspark_block_sizes(tmp_path) -> None:
    target = _checkpoint(tmp_path, "target", dspark_block_size=5)
    draft = _checkpoint(tmp_path, "draft", dspark_block_size=7)

    with pytest.raises(ValueError, match="must match"):
        parse_loader_settings(
            {
                "DSPARK_TARGET_MODEL": str(target),
                "DSPARK_DRAFT_MODEL": str(draft),
                "DSPARK_TP_SIZE": "8",
                "DSPARK_MAX_MODEL_LEN": "4096",
            }
        )


def test_offline_mode_overrides_download_capable_environment() -> None:
    environ = {
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
    }

    enforce_offline_mode(environ)

    assert environ["HF_HUB_OFFLINE"] == "1"
    assert environ["HF_DATASETS_OFFLINE"] == "1"
    assert environ["TRANSFORMERS_OFFLINE"] == "1"
    assert environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_launch_context_requires_matching_torchrun_world() -> None:
    context = parse_launch_context(
        {"RANK": "3", "LOCAL_RANK": "3", "WORLD_SIZE": "8"},
        tp_size=8,
    )

    assert context.rank == 3
    assert context.local_rank == 3
    assert context.world_size == 8

    with pytest.raises(ValueError, match="must run under torchrun"):
        parse_launch_context({}, tp_size=8)
    with pytest.raises(ValueError, match="does not match"):
        parse_launch_context(
            {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"},
            tp_size=8,
        )


def test_stage_tracker_records_order_and_failure_point() -> None:
    messages: list[str] = []
    tracker = StageTracker(rank=2, emit=messages.append)

    tracker.mark(CONFIG_READY, detail="ready")
    tracker.failed(RuntimeError("boom"))
    tracker.mark(CLEANUP_COMPLETE, cleanup_errors=[])

    assert tracker.last_load_stage == CONFIG_READY
    assert tracker.stages == [CONFIG_READY, CLEANUP_COMPLETE]
    assert '"failed_after": "CONFIG_READY"' in messages[1]

    with pytest.raises(RuntimeError, match="already recorded"):
        tracker.mark(CLEANUP_COMPLETE)


def test_stage_tracker_accepts_the_complete_loader_sequence() -> None:
    tracker = StageTracker(rank=0, emit=lambda _message: None)

    for stage in LOAD_STAGES:
        tracker.mark(stage)
    tracker.mark(CLEANUP_COMPLETE)

    assert tracker.stages == [*LOAD_STAGES, CLEANUP_COMPLETE]


def test_cleanup_runs_every_step_and_reports_errors() -> None:
    called: list[str] = []
    tracker = StageTracker(rank=0, emit=lambda _message: None)

    def first() -> None:
        called.append("first")
        raise RuntimeError("first failed")

    def second() -> None:
        called.append("second")

    errors = run_cleanup_steps(
        (("first", first), ("second", second)),
        tracker,
    )

    assert called == ["first", "second"]
    assert errors == ["first: RuntimeError: first failed"]
    assert tracker.stages == [CLEANUP_COMPLETE]


def test_forbidden_import_delta_checks_only_new_modules() -> None:
    preexisting = {"vllm.models.deepseek_v4.nvidia.model"}
    after = preexisting | {
        "vllm.v1.worker.gpu.spec_decode.dspark.utils",
        "vllm_ascend.models.deepseek_v4_dspark",
    }

    assert forbidden_import_delta(preexisting, after) == ["vllm.v1.worker.gpu.spec_decode.dspark.utils"]
