# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import gc
import json
import os
import sys
from contextlib import ExitStack
from typing import Any

import pytest

from tests.e2e.nightly.single_node.spec_decode.dspark_loader_harness import (
    CLEANUP_COMPLETE,
    HarnessNotConfigured,
    dspark_loader_config_context,
    enforce_offline_mode,
    forbidden_import_delta,
    forbidden_instance_attribute_types,
    forbidden_module_tree_types,
    parse_launch_context,
    parse_loader_settings,
    run_cleanup_steps,
)

TARGET_LOADED = "TARGET_LOADED"
DRAFT_LOADED = "DRAFT_LOADED"
DRAFT_KV_SPEC_READY = "DRAFT_KV_SPEC_READY"
DRAFT_KV_ALLOCATED = "DRAFT_KV_ALLOCATED"
DRAFT_ATTN_INSTALLED = "DRAFT_ATTN_INSTALLED"
KV_INIT_ONLY_PASS = "KV_INIT_ONLY_PASS"
KV_INIT_STAGES = (
    TARGET_LOADED,
    DRAFT_LOADED,
    DRAFT_KV_SPEC_READY,
    DRAFT_KV_ALLOCATED,
    DRAFT_ATTN_INSTALLED,
    KV_INIT_ONLY_PASS,
)
DEFAULT_KV_CACHE_BYTES = 512 * 1024 * 1024

enforce_offline_mode()


class _KVInitStageTracker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.stages: list[str] = []

    def mark(self, stage: str, **details: Any) -> None:
        if stage == CLEANUP_COMPLETE:
            if stage in self.stages:
                raise RuntimeError("CLEANUP_COMPLETE was already recorded.")
        else:
            completed = sum(item != CLEANUP_COMPLETE for item in self.stages)
            expected = KV_INIT_STAGES[completed]
            if stage != expected:
                raise RuntimeError(f"Invalid DSpark KV-init stage transition: expected {expected!r}, got {stage!r}.")
        self.stages.append(stage)
        print(
            "DSPARK_KV_INIT_STAGE="
            + json.dumps(
                {"rank": self.rank, "stage": stage, **details},
                default=str,
                sort_keys=True,
            ),
            flush=True,
        )

    def failed(self, exc: BaseException) -> None:
        print(
            "DSPARK_KV_INIT_FAILURE="
            + json.dumps(
                {
                    "rank": self.rank,
                    "failed_after": self.stages[-1] if self.stages else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _memory_snapshot(torch, device) -> dict[str, int]:
    return {
        "allocated": int(torch.npu.memory_allocated(device)),
        "reserved": int(torch.npu.memory_reserved(device)),
        "max_allocated": int(torch.npu.max_memory_allocated(device)),
        "max_reserved": int(torch.npu.max_memory_reserved(device)),
    }


def _kv_cache_budget(environ: dict[str, str]) -> int:
    raw_value = environ.get("DSPARK_KV_CACHE_BYTES")
    if raw_value is None:
        return DEFAULT_KV_CACHE_BYTES
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("DSPARK_KV_CACHE_BYTES must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("DSPARK_KV_CACHE_BYTES must be a positive integer.")
    return value


def test_dspark_kv_cache_init_only_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    tracker = _KVInitStageTracker(launch.rank)
    config_context = ExitStack()
    state: dict[str, Any] = {"worker": None}
    primary_error: BaseException | None = None

    try:
        import torch
        from vllm.distributed import get_ep_group, get_pp_group, get_tp_group
        from vllm.engine.arg_utils import EngineArgs
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        import vllm_ascend
        from vllm_ascend.worker.v2.spec_decode.dspark import (
            AscendDSparkSpeculator,
        )
        from vllm_ascend.worker.worker import NPUWorker

        engine_args = EngineArgs(
            model=str(settings.target_model),
            tokenizer=str(settings.target_model),
            skip_tokenizer_init=True,
            dtype=settings.dtype,
            max_model_len=settings.max_model_len,
            tensor_parallel_size=settings.tp_size,
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            distributed_executor_backend="external_launcher",
            enforce_eager=True,
            max_num_seqs=1,
            speculative_config={
                "method": "dspark",
                "model": str(settings.draft_model),
                "num_speculative_tokens": settings.num_speculative_tokens,
            },
        )
        vllm_config = engine_args.create_engine_config()
        config_context.enter_context(dspark_loader_config_context(vllm_config))
        vllm_ascend.register_model()

        worker = NPUWorker(
            vllm_config=vllm_config,
            local_rank=launch.local_rank,
            rank=launch.rank,
            distributed_init_method="env://",
            is_driver_worker=launch.rank == 0,
        )
        state["worker"] = worker
        worker.init_device()
        worker.load_model()
        runner = worker.model_runner
        assert runner is not None
        speculator = runner.speculator
        assert type(speculator) is AscendDSparkSpeculator
        target_model = runner.model
        draft_model = speculator.model
        assert type(target_model).__module__ == "vllm_ascend.models.deepseek_v4"
        assert type(target_model).__name__ == "AscendDeepseekV4ForCausalLM"
        assert type(draft_model).__module__ == "vllm_ascend.models.deepseek_v4_dspark"
        assert type(draft_model).__name__ == "DSparkDeepseekV4ForCausalLM"
        assert forbidden_module_tree_types(target_model) == []
        assert forbidden_module_tree_types(draft_model) == []
        assert forbidden_instance_attribute_types(speculator) == []
        assert get_tp_group().world_size == 8
        assert get_ep_group().world_size == 8
        assert get_pp_group().world_size == 1

        target_memory = _memory_snapshot(torch, worker.device)
        tracker.mark(TARGET_LOADED, memory=target_memory)
        tracker.mark(
            DRAFT_LOADED,
            memory=target_memory,
            draft_layers=sorted(speculator.draft_attn_layer_names or ()),
        )
        expected_draft_layers = {f"mtp.{stage}.self_attn.swa_cache" for stage in range(3)}
        assert set(speculator.draft_attn_layer_names or ()) == expected_draft_layers
        kv_init_import_baseline = set(sys.modules)

        kv_cache_specs = worker.get_kv_cache_spec()
        draft_specs = speculator.draft_kv_cache_specs
        assert draft_specs is not None
        assert set(draft_specs) == set(speculator.draft_attn_layer_names or ())
        assert all(kv_cache_specs[name] is draft_specs[name] for name in draft_specs)
        forward_context = vllm_config.compilation_config.static_forward_context
        assert runner.compilation_config is vllm_config.compilation_config
        assert runner.compilation_config.static_forward_context is forward_context
        missing_cache_owners = set(kv_cache_specs).difference(forward_context)
        assert not missing_cache_owners
        tracker.mark(
            DRAFT_KV_SPEC_READY,
            total_spec_count=len(kv_cache_specs),
            draft_spec_count=len(draft_specs),
            draft_layers=sorted(draft_specs),
            forward_context_count=len(forward_context),
        )

        available_memory = _kv_cache_budget(os.environ)
        kv_cache_config = get_kv_cache_configs(
            vllm_config,
            [kv_cache_specs],
            [available_memory],
        )[0]
        before_kv_memory = _memory_snapshot(torch, worker.device)
        worker.initialize_from_config(kv_cache_config)
        after_kv_memory = _memory_snapshot(torch, worker.device)
        assert speculator.draft_kv_caches is not None
        assert all(
            speculator.draft_kv_caches[name] is vllm_config.compilation_config.static_forward_context[name].kv_cache
            for name in speculator.draft_kv_caches
        )
        tracker.mark(
            DRAFT_KV_ALLOCATED,
            configured_bytes=available_memory,
            memory_before=before_kv_memory,
            memory_after=after_kv_memory,
            allocated_delta=(after_kv_memory["allocated"] - before_kv_memory["allocated"]),
            reserved_delta=(after_kv_memory["reserved"] - before_kv_memory["reserved"]),
        )

        assert speculator.attn_groups is not None
        assert speculator.attn_backends is not None
        assert speculator.block_tables is runner.block_tables
        assert speculator.kv_cache_config is runner.kv_cache_config
        assert speculator.draft_kv_cache_group_ids
        slot_mappings = runner.block_tables.slot_mappings
        assert slot_mappings.device.type == "npu"
        assert slot_mappings.dtype == torch.int32
        assert slot_mappings.shape[0] == len(runner.kv_cache_config.kv_cache_groups)
        draft_group_layers = {
            name
            for group_id in speculator.draft_kv_cache_group_ids
            for group in speculator.attn_groups[group_id]
            for name in group.layer_names
        }
        assert draft_group_layers == set(speculator.draft_attn_layer_names or ())
        assert not draft_group_layers.intersection(speculator.target_attn_layer_names or ())
        kv_init_forbidden_imports = forbidden_import_delta(
            kv_init_import_baseline,
            set(sys.modules),
        )
        assert kv_init_forbidden_imports == []
        tracker.mark(
            DRAFT_ATTN_INSTALLED,
            draft_group_ids=speculator.draft_kv_cache_group_ids,
            draft_group_layers=sorted(draft_group_layers),
            backend_count=len(speculator.attn_backends),
            block_table_identity=True,
            slot_mapping_dtype=str(slot_mappings.dtype),
            slot_mapping_shape=tuple(slot_mappings.shape),
            forbidden_import_delta=kv_init_forbidden_imports,
        )

        torch.distributed.barrier()
        tracker.mark(
            KV_INIT_ONLY_PASS,
            profile_run=False,
            dummy_run=False,
            proposal=False,
            generation=False,
        )
        del (
            target_model,
            draft_model,
            runner,
            speculator,
            kv_cache_specs,
            draft_specs,
            kv_cache_config,
            slot_mappings,
        )
    except BaseException as exc:
        primary_error = exc
        tracker.failed(exc)
        target_model = draft_model = runner = speculator = None
        kv_cache_specs = draft_specs = kv_cache_config = slot_mappings = None
        del (
            target_model,
            draft_model,
            runner,
            speculator,
            kv_cache_specs,
            draft_specs,
            kv_cache_config,
            slot_mappings,
        )
        raise
    finally:
        worker = state.get("worker")

        def shutdown_worker() -> None:
            if worker is not None:
                worker.shutdown()

        def release_state() -> None:
            state.clear()
            gc.collect()

        def destroy_ascend_groups() -> None:
            from vllm_ascend.distributed.parallel_state import (
                destroy_ascend_model_parallel,
            )

            destroy_ascend_model_parallel()

        def destroy_vllm_groups() -> None:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()

        def clear_npu_cache() -> None:
            import torch

            if hasattr(torch, "npu") and torch.npu.is_initialized():
                torch.npu.synchronize()
                torch.npu.empty_cache()

        cleanup_errors = run_cleanup_steps(
            (
                ("worker.shutdown", shutdown_worker),
                ("release KV/model state", release_state),
                ("destroy_ascend_model_parallel", destroy_ascend_groups),
                ("destroy_distributed_environment", destroy_vllm_groups),
                ("torch.npu.empty_cache", clear_npu_cache),
                ("vllm config context", config_context.close),
            ),
            tracker,
        )
        if cleanup_errors and primary_error is None:
            raise RuntimeError("DSpark KV-init-only cleanup failed: " + "; ".join(cleanup_errors))
