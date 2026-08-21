# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_READY = "CONFIG_READY"
REGISTRY_RESOLVED = "REGISTRY_RESOLVED"
DISTRIBUTED_READY = "DISTRIBUTED_READY"
TARGET_LOADED = "TARGET_LOADED"
DRAFT_LOADED = "DRAFT_LOADED"
EMBEDDING_CONTRACT_VERIFIED = "EMBEDDING_CONTRACT_VERIFIED"
CHECKPOINT_MAPPING_VERIFIED = "CHECKPOINT_MAPPING_VERIFIED"
LOADER_ONLY_PASS = "LOADER_ONLY_PASS"
CLEANUP_COMPLETE = "CLEANUP_COMPLETE"

LOAD_STAGES = (
    CONFIG_READY,
    REGISTRY_RESOLVED,
    DISTRIBUTED_READY,
    TARGET_LOADED,
    DRAFT_LOADED,
    EMBEDDING_CONTRACT_VERIFIED,
    CHECKPOINT_MAPPING_VERIFIED,
    LOADER_ONLY_PASS,
)

FORBIDDEN_IMPORT_PREFIXES = (
    "vllm.models.deepseek_v4.nvidia",
    "vllm.v1.worker.gpu.spec_decode.dspark",
    "vllm_ascend.ops.triton.spec_decode",
)

_SUPPORTED_DTYPES = {
    "auto",
    "bfloat16",
    "float",
    "float16",
    "float32",
    "half",
}


class HarnessNotConfigured(RuntimeError):
    """Raised when neither local checkpoint path was configured."""


@dataclass(frozen=True)
class LoaderOnlySettings:
    target_model: Path
    draft_model: Path
    tp_size: int
    max_model_len: int
    dtype: str
    num_speculative_tokens: int
    target_config: dict[str, Any]
    draft_config: dict[str, Any]


@dataclass(frozen=True)
class LaunchContext:
    rank: int
    local_rank: int
    world_size: int


def enforce_offline_mode(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Disable remote model/config resolution for this validation process."""
    target = os.environ if environ is None else environ
    for name in (
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    ):
        target[name] = "1"


def _positive_int(value: str | None, name: str) -> int:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be set to a positive integer.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return parsed


def _local_checkpoint(
    value: str | None,
    name: str,
) -> tuple[Path, dict[str, Any]]:
    if value is None or not value.strip():
        raise ValueError(f"{name} must point to a local checkpoint directory.")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(
            f"{name} must be an existing local checkpoint directory; "
            f"remote model IDs and downloads are forbidden: {value!r}."
        )
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{name} has no local config.json: {config_path}.")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {name} config: {config_path}.") from exc
    if not isinstance(config, dict):
        raise ValueError(f"{name} config.json must contain a JSON object.")
    return path, config


def parse_loader_settings(
    environ: Mapping[str, str],
) -> LoaderOnlySettings:
    target_value = environ.get("DSPARK_TARGET_MODEL")
    draft_value = environ.get("DSPARK_DRAFT_MODEL")
    if not target_value and not draft_value:
        raise HarnessNotConfigured(
            "HARNESS_READY; REAL_CHECKPOINT_NOT_VALIDATED: set "
            "DSPARK_TARGET_MODEL and "
            "DSPARK_DRAFT_MODEL to run the loader-only NPU harness."
        )
    if not target_value or not draft_value:
        missing = "DSPARK_TARGET_MODEL" if not target_value else "DSPARK_DRAFT_MODEL"
        raise ValueError(f"{missing} is required when either checkpoint path is set.")

    target_model, target_config = _local_checkpoint(
        target_value,
        "DSPARK_TARGET_MODEL",
    )
    draft_model, draft_config = _local_checkpoint(
        draft_value,
        "DSPARK_DRAFT_MODEL",
    )
    tp_size = _positive_int(environ.get("DSPARK_TP_SIZE"), "DSPARK_TP_SIZE")
    max_model_len = _positive_int(
        environ.get("DSPARK_MAX_MODEL_LEN"),
        "DSPARK_MAX_MODEL_LEN",
    )
    dtype = environ.get("DSPARK_DTYPE", "auto").strip().lower() or "auto"
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"DSPARK_DTYPE must be one of {sorted(_SUPPORTED_DTYPES)}, got {dtype!r}.")

    target_block_size = target_config.get("dspark_block_size")
    draft_block_size = draft_config.get("dspark_block_size")
    configured_block_sizes = [value for value in (target_block_size, draft_block_size) if value is not None]
    if not configured_block_sizes or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in configured_block_sizes
    ):
        raise ValueError("A positive dspark_block_size is required in the local target or draft config.json.")
    if len(set(configured_block_sizes)) != 1:
        raise ValueError(
            f"Target and draft dspark_block_size values must match, got {target_block_size!r} and {draft_block_size!r}."
        )
    block_size = configured_block_sizes[0]

    return LoaderOnlySettings(
        target_model=target_model,
        draft_model=draft_model,
        tp_size=tp_size,
        max_model_len=max_model_len,
        dtype=dtype,
        num_speculative_tokens=block_size,
        target_config=target_config,
        draft_config=draft_config,
    )


def parse_launch_context(
    environ: Mapping[str, str],
    tp_size: int,
) -> LaunchContext:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in environ]
    if missing:
        raise ValueError("The loader-only harness must run under torchrun; missing " + ", ".join(missing) + ".")
    try:
        context = LaunchContext(
            rank=int(environ["RANK"]),
            local_rank=int(environ["LOCAL_RANK"]),
            world_size=int(environ["WORLD_SIZE"]),
        )
    except ValueError as exc:
        raise ValueError("RANK, LOCAL_RANK, and WORLD_SIZE must be integers.") from exc
    if context.world_size != tp_size:
        raise ValueError(f"torchrun WORLD_SIZE={context.world_size} does not match DSPARK_TP_SIZE={tp_size}.")
    if not 0 <= context.rank < context.world_size:
        raise ValueError(f"RANK={context.rank} is outside WORLD_SIZE={context.world_size}.")
    if not 0 <= context.local_rank < context.world_size:
        raise ValueError(f"LOCAL_RANK={context.local_rank} is outside WORLD_SIZE={context.world_size}.")
    return context


class StageTracker:
    def __init__(
        self,
        rank: int,
        emit: Callable[[str], None] = print,
    ) -> None:
        self.rank = rank
        self.stages: list[str] = []
        self._emit = emit

    @property
    def last_load_stage(self) -> str | None:
        return next(
            (stage for stage in reversed(self.stages) if stage != CLEANUP_COMPLETE),
            None,
        )

    def mark(self, stage: str, **details: Any) -> None:
        if stage == CLEANUP_COMPLETE:
            if CLEANUP_COMPLETE in self.stages:
                raise RuntimeError("CLEANUP_COMPLETE was already recorded.")
        else:
            completed = sum(item != CLEANUP_COMPLETE for item in self.stages)
            expected = LOAD_STAGES[completed] if completed < len(LOAD_STAGES) else None
            if stage != expected:
                raise RuntimeError(f"Invalid loader stage transition: expected {expected!r}, got {stage!r}.")
        self.stages.append(stage)
        payload = {"rank": self.rank, "stage": stage, **details}
        self._emit("DSPARK_LOADER_STAGE=" + json.dumps(payload, default=str, sort_keys=True))

    def failed(self, exc: BaseException) -> None:
        payload = {
            "rank": self.rank,
            "failed_after": self.last_load_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        self._emit("DSPARK_LOADER_FAILURE=" + json.dumps(payload, sort_keys=True))


def forbidden_import_delta(
    before: set[str],
    after: set[str],
) -> list[str]:
    return sorted(module for module in after - before if module.startswith(FORBIDDEN_IMPORT_PREFIXES))


def run_cleanup_steps(
    steps: Sequence[tuple[str, Callable[[], None]]],
    tracker: StageTracker,
) -> list[str]:
    errors: list[str] = []
    for name, cleanup in steps:
        try:
            cleanup()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    tracker.mark(CLEANUP_COMPLETE, cleanup_errors=errors)
    return errors
