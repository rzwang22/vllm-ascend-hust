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
    CHECKPOINT_MAPPING_VERIFIED,
    CONFIG_CREATED,
    CONFIG_READY,
    DISTRIBUTED_READY,
    DRAFT_LOADED,
    DRAFT_MODEL_LOADED,
    DRAFT_VLLM_CONFIG_BUILT,
    EMBEDDING_CONTRACT_VERIFIED,
    FORBIDDEN_IMPORT_PREFIXES,
    IMPORTS_COMPLETED,
    LOADER_ONLY_PASS,
    REGISTRY_RESOLVED,
    TARGET_LOADED,
    TARGET_MODEL_LOADED,
    WORKER_INIT_DEVICE_COMPLETED,
    HarnessNotConfigured,
    ImportStageTracker,
    StageTracker,
    dspark_loader_config_context,
    enforce_offline_mode,
    forbidden_class_references,
    forbidden_import_delta,
    forbidden_instance_attribute_types,
    forbidden_module_tree_types,
    parse_launch_context,
    parse_loader_settings,
    run_cleanup_steps,
)

enforce_offline_mode()


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "architectures": config.get("architectures"),
        "dtype": config.get("torch_dtype", config.get("dtype")),
        "quantization": config.get(
            "quantization_config",
            config.get("quantization"),
        ),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "n_routed_experts": config.get("n_routed_experts"),
        "dspark_block_size": config.get("dspark_block_size"),
        "dspark_num_mtp_layers": config.get(
            "dspark_num_mtp_layers",
            config.get("n_mtp_layers"),
        ),
    }


def _memory_snapshot(torch, device) -> dict[str, int]:
    return {
        "allocated": int(torch.npu.memory_allocated(device)),
        "reserved": int(torch.npu.memory_reserved(device)),
        "max_allocated": int(torch.npu.max_memory_allocated(device)),
        "max_reserved": int(torch.npu.max_memory_reserved(device)),
    }


def _language_model_inner(model):
    language_model = model.get_language_model() if hasattr(model, "get_language_model") else model
    return language_model.model


def _assert_materialized_on_npu(model) -> int:
    parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += 1
        assert not parameter.is_meta, f"DSpark parameter remains meta: {name}"
        assert parameter.device.type == "npu", (
            f"DSpark parameter is not materialized on NPU: {name} is on {parameter.device}"
        )
    assert parameter_count > 0, "DSpark model has no registered parameters."
    return parameter_count


def _assert_embedding_contract(target_model, draft_model, quant_config) -> str:
    target_inner = _language_model_inner(target_model)
    target_embed = getattr(target_inner, "embed_tokens", None)
    target_lm_head = getattr(target_model, "lm_head", None)
    draft_embed = getattr(draft_model.model, "embed_tokens", None)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    assert target_embed is not None and target_lm_head is not None
    assert draft_embed is not None and draft_lm_head is not None

    if quant_config is None:
        assert draft_embed is target_embed
        assert draft_lm_head is target_lm_head
        return "shared"

    assert draft_embed is not target_embed
    assert draft_lm_head is not target_lm_head
    _assert_materialized_on_npu(draft_embed)
    _assert_materialized_on_npu(draft_lm_head)
    return "independent_quantized"


def test_dspark_loader_only_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    tracker = StageTracker(launch.rank)
    lifecycle_before_modules = set(sys.modules)
    import_tracker = ImportStageTracker(
        launch.rank,
        lifecycle_before_modules,
    )
    state: dict[str, Any] = {
        "worker": None,
        "target_model": None,
        "draft_model": None,
        "loader_result": None,
        "checkpoint_load_calls": 0,
        "loaded_parameter_names": None,
    }
    cleanup_errors: list[str] = []
    primary_error: BaseException | None = None
    config_context = ExitStack()

    try:
        import torch
        from vllm import ModelRegistry
        from vllm.distributed import get_ep_group, get_pp_group, get_tp_group
        from vllm.engine.arg_utils import EngineArgs

        import vllm_ascend

        import_tracker.mark(IMPORTS_COMPLETED)
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
        import_tracker.mark(CONFIG_CREATED)
        assert vllm_config.use_v2_model_runner
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        config_context.enter_context(
            dspark_loader_config_context(vllm_config),
        )
        tracker.mark(
            CONFIG_READY,
            target=str(settings.target_model),
            draft=str(settings.draft_model),
            tp_size=settings.tp_size,
            expert_parallel=True,
            max_model_len=settings.max_model_len,
            dtype=settings.dtype,
            target_config=_config_summary(settings.target_config),
            draft_config=_config_summary(settings.draft_config),
        )

        vllm_ascend.register_model()
        model_cls = ModelRegistry._try_load_model_cls("DSparkDraftModel")
        assert model_cls is not None
        assert model_cls.__module__ == "vllm_ascend.models.deepseek_v4_dspark"
        assert model_cls.__name__ == "DSparkDeepseekV4ForCausalLM"
        assert all(not base.__module__.startswith("vllm.models.deepseek_v4.nvidia") for base in model_cls.__mro__)
        import_tracker.mark(REGISTRY_RESOLVED)
        tracker.mark(
            REGISTRY_RESOLVED,
            model_class=f"{model_cls.__module__}.{model_cls.__name__}",
        )

        from vllm.config import get_current_vllm_config

        from vllm_ascend.worker.worker import NPUWorker

        assert get_current_vllm_config() is vllm_config
        worker = NPUWorker(
            vllm_config=vllm_config,
            local_rank=launch.local_rank,
            rank=launch.rank,
            distributed_init_method="env://",
            is_driver_worker=launch.rank == 0,
        )
        state["worker"] = worker
        assert get_current_vllm_config() is vllm_config
        worker.init_device()
        assert get_current_vllm_config() is vllm_config
        runner = worker.model_runner
        assert runner is not None
        from vllm_ascend.worker.v2.spec_decode.dspark import (
            AscendDSparkSpeculator,
        )

        assert type(runner.speculator) is AscendDSparkSpeculator
        assert get_pp_group().world_size == 1
        assert get_tp_group().world_size == settings.tp_size
        assert get_tp_group().rank_in_group == launch.rank
        assert get_ep_group().world_size == settings.tp_size
        import_tracker.mark(WORKER_INIT_DEVICE_COMPLETED)
        baseline_memory = _memory_snapshot(torch, worker.device)
        tracker.mark(
            DISTRIBUTED_READY,
            local_rank=launch.local_rank,
            world_size=launch.world_size,
            tp_rank=get_tp_group().rank_in_group,
            tp_world_size=get_tp_group().world_size,
            ep_rank=get_ep_group().rank_in_group,
            ep_world_size=get_ep_group().world_size,
            pp_world_size=get_pp_group().world_size,
            device=str(worker.device),
            memory=baseline_memory,
        )

        from vllm_ascend.worker.v2.spec_decode.dspark import model_loader

        original_loader = model_loader.load_dspark_model
        original_build_draft_vllm_config = model_loader._build_draft_vllm_config
        original_load_weights = model_cls.load_weights

        def observed_loader(target_model, config):
            state["target_model"] = target_model
            state["draft_import_baseline"] = set(sys.modules)
            import_tracker.mark(TARGET_MODEL_LOADED)
            target_memory = _memory_snapshot(torch, worker.device)
            tracker.mark(
                TARGET_LOADED,
                memory=target_memory,
                allocated_delta=target_memory["allocated"] - baseline_memory["allocated"],
                reserved_delta=target_memory["reserved"] - baseline_memory["reserved"],
            )
            draft_model = original_loader(target_model, config)
            state["loader_result"] = draft_model
            state["draft_model"] = draft_model
            import_tracker.mark(DRAFT_MODEL_LOADED)
            draft_memory = _memory_snapshot(torch, worker.device)
            tracker.mark(
                DRAFT_LOADED,
                memory=draft_memory,
                allocated_delta=draft_memory["allocated"] - target_memory["allocated"],
                reserved_delta=draft_memory["reserved"] - target_memory["reserved"],
            )
            return draft_model

        def observed_build_draft_vllm_config(config, draft_quant_config):
            draft_vllm_config = original_build_draft_vllm_config(
                config,
                draft_quant_config,
            )
            state["draft_vllm_config"] = draft_vllm_config
            import_tracker.mark(DRAFT_VLLM_CONFIG_BUILT)
            return draft_vllm_config

        def observed_load_weights(draft_model, weights):
            loaded_parameter_names = original_load_weights(draft_model, weights)
            state["checkpoint_load_calls"] += 1
            state["loaded_parameter_names"] = set(loaded_parameter_names)
            return loaded_parameter_names

        model_loader.load_dspark_model = observed_loader
        model_loader._build_draft_vllm_config = observed_build_draft_vllm_config
        model_cls.load_weights = observed_load_weights
        execution_calls: list[str] = []

        def forbidden_execution(name):
            def fail_if_called(*_args, **_kwargs):
                execution_calls.append(name)
                raise AssertionError(f"loader-only harness reached forbidden boundary: {name}")

            return fail_if_called

        missing_attribute = object()
        guarded_attributes = []
        for owner, name in (
            (runner, "profile_run"),
            (runner, "_dummy_run"),
            (runner.speculator, "capture"),
            (runner.speculator, "propose"),
        ):
            guarded_attributes.append((owner, name, owner.__dict__.get(name, missing_attribute)))
            setattr(owner, name, forbidden_execution(name))
        try:
            worker.load_model()
        finally:
            model_loader.load_dspark_model = original_loader
            model_loader._build_draft_vllm_config = original_build_draft_vllm_config
            model_cls.load_weights = original_load_weights
            for owner, name, original in reversed(guarded_attributes):
                if original is missing_attribute:
                    delattr(owner, name)
                else:
                    setattr(owner, name, original)
            guarded_attributes.clear()
        assert get_current_vllm_config() is vllm_config

        draft_model = runner.speculator.model
        target_model = runner.model
        assert draft_model is state["loader_result"]
        assert draft_model is state["draft_model"]
        assert target_model is state["target_model"]
        assert type(draft_model) is model_cls
        target_cls = type(target_model)
        assert target_cls.__module__ == "vllm_ascend.models.deepseek_v4"
        assert target_cls.__name__ == "AscendDeepseekV4ForCausalLM"
        assert type(runner.speculator) is AscendDSparkSpeculator
        for selected_cls in (target_cls, model_cls, AscendDSparkSpeculator):
            assert forbidden_class_references(selected_cls) == []
        assert forbidden_module_tree_types(target_model) == []
        assert forbidden_module_tree_types(draft_model) == []
        assert forbidden_instance_attribute_types(runner.speculator) == []
        sharing = _assert_embedding_contract(
            target_model,
            draft_model,
            vllm_config.quant_config,
        )
        import_tracker.mark(EMBEDDING_CONTRACT_VERIFIED)
        tracker.mark(EMBEDDING_CONTRACT_VERIFIED, contract=sharing)

        parameter_count = _assert_materialized_on_npu(draft_model)
        loaded_parameter_names = state["loaded_parameter_names"]
        assert state["checkpoint_load_calls"] == 1
        assert isinstance(loaded_parameter_names, set)
        assert loaded_parameter_names
        lifecycle_forbidden_modules = forbidden_import_delta(
            lifecycle_before_modules,
            set(sys.modules),
        )
        forbidden_core_dspark_or_triton = [
            module for module in lifecycle_forbidden_modules if module.startswith(FORBIDDEN_IMPORT_PREFIXES[1:])
        ]
        assert forbidden_core_dspark_or_triton == [], (
            "The V2 loader lifecycle imported a core DSpark or Ascend Triton "
            "spec-decode runtime: " + json.dumps(forbidden_core_dspark_or_triton)
        )
        pre_draft_stages = {
            IMPORTS_COMPLETED,
            CONFIG_CREATED,
            REGISTRY_RESOLVED,
            WORKER_INIT_DEVICE_COMPLETED,
            TARGET_MODEL_LOADED,
        }
        nvidia_first_seen = {
            module: stage
            for module, stage in import_tracker.first_seen.items()
            if module.startswith(FORBIDDEN_IMPORT_PREFIXES[0])
        }
        assert all(stage in pre_draft_stages for stage in nvidia_first_seen.values()), (
            "NVIDIA DeepSeek V4 modules first appeared during the isolated "
            "draft lifecycle: " + json.dumps(nvidia_first_seen, sort_keys=True)
        )
        draft_forbidden_modules = forbidden_import_delta(
            state["draft_import_baseline"],
            set(sys.modules),
        )
        assert draft_forbidden_modules == [], (
            "The target-to-draft loader boundary imported forbidden "
            "NVIDIA/Core-DSpark/Triton modules: " + json.dumps(draft_forbidden_modules)
        )
        assert execution_calls == []
        tracker.mark(
            CHECKPOINT_MAPPING_VERIFIED,
            materialized_parameters=parameter_count,
            checkpoint_mapped_parameters=len(loaded_parameter_names),
            checkpoint_load_calls=state["checkpoint_load_calls"],
            missing_weights=0,
            unexpected_weights=0,
            evidence="DSparkDeepseekV4ForCausalLM.load_weights strict success",
            lifecycle_forbidden_import_delta=lifecycle_forbidden_modules,
            forbidden_import_first_seen=import_tracker.first_seen,
            draft_forbidden_import_delta=draft_forbidden_modules,
        )
        torch.distributed.barrier()
        tracker.mark(LOADER_ONLY_PASS, execution_calls=execution_calls)
        del draft_model, target_model, model_cls
    except BaseException as exc:
        primary_error = exc
        tracker.failed(exc)
        raise
    finally:
        worker = state.get("worker")

        def shutdown_worker() -> None:
            if worker is not None:
                worker.shutdown()

        def release_models() -> None:
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
                ("release_models", release_models),
                ("destroy_ascend_model_parallel", destroy_ascend_groups),
                ("destroy_distributed_environment", destroy_vllm_groups),
                ("torch.npu.empty_cache", clear_npu_cache),
                ("vllm config context", config_context.close),
            ),
            tracker,
        )
        if cleanup_errors and primary_error is None:
            raise RuntimeError("DSpark loader-only cleanup failed: " + "; ".join(cleanup_errors))
