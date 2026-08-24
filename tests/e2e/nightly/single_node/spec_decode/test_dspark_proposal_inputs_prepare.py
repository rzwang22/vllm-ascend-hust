# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import gc
import json
import os
import sys
from contextlib import ExitStack
from types import SimpleNamespace
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
from tests.e2e.nightly.single_node.spec_decode.test_dspark_kv_cache_init import (
    _kv_cache_budget,
    _memory_snapshot,
)

TARGET_FORWARD_COMPLETED = "TARGET_FORWARD_COMPLETED"
AUX_HIDDEN_STATES_CAPTURED = "AUX_HIDDEN_STATES_CAPTURED"
PROPOSAL_INPUTS_PREPARED = "PROPOSAL_INPUTS_PREPARED"
PER_STEP_KV_METADATA_UPDATED = "PER_STEP_KV_METADATA_UPDATED"
PREPARE_ONLY_PASS = "PREPARE_ONLY_PASS"
PREPARE_STAGES = (
    TARGET_FORWARD_COMPLETED,
    AUX_HIDDEN_STATES_CAPTURED,
    PROPOSAL_INPUTS_PREPARED,
    PER_STEP_KV_METADATA_UPDATED,
    PREPARE_ONLY_PASS,
)
EXPECTED_TARGET_LAYER_IDS = (40, 41, 42)
INDEXER_KV_CACHE_SUFFIX = ".indexer.k_cache"
PREPARE_ONLY_CACHE_BLOCK_SIZE = 128

enforce_offline_mode()


class _PrepareStageTracker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.stages: list[str] = []

    def mark(self, stage: str, **details: Any) -> None:
        if stage == CLEANUP_COMPLETE:
            if stage in self.stages:
                raise RuntimeError("CLEANUP_COMPLETE was already recorded.")
        else:
            completed = sum(item != CLEANUP_COMPLETE for item in self.stages)
            expected = PREPARE_STAGES[completed]
            if stage != expected:
                raise RuntimeError(
                    f"Invalid DSpark prepare-only stage transition: expected {expected!r}, got {stage!r}."
                )
        self.stages.append(stage)
        print(
            "DSPARK_PREPARE_STAGE="
            + json.dumps(
                {"rank": self.rank, "stage": stage, **details},
                default=str,
                sort_keys=True,
            ),
            flush=True,
        )

    def failed(self, exc: BaseException) -> None:
        print(
            "DSPARK_PREPARE_FAILURE="
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


def _prompt_token_id(config: dict[str, Any]) -> int:
    for name in ("bos_token_id", "eos_token_id"):
        value = config.get(name)
        if type(value) is int and value >= 0:
            return value
    raise ValueError(
        "The local target config must provide a non-negative integer "
        "bos_token_id or eos_token_id for the prepare-only target step."
    )


def test_dspark_proposal_inputs_prepare_only_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    tracker = _PrepareStageTracker(launch.rank)
    config_context = ExitStack()
    state: dict[str, Any] = {"worker": None}
    primary_error: BaseException | None = None

    try:
        import torch
        from vllm.distributed import get_ep_group, get_pp_group, get_tp_group
        from vllm.engine.arg_utils import EngineArgs
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            SchedulerOutput,
        )

        import vllm_ascend
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
        from vllm_ascend.ops.dsa import _build_kv_cache
        from vllm_ascend.worker.v2.spec_decode.dspark import (
            AscendDSparkProposalInputs,
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
            block_size=PREPARE_ONLY_CACHE_BLOCK_SIZE,
            max_num_seqs=1,
            speculative_config={
                "method": "dspark",
                "model": str(settings.draft_model),
                "num_speculative_tokens": settings.num_speculative_tokens,
            },
        )
        vllm_config = engine_args.create_engine_config()
        assert vllm_config.cache_config.block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE
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
        assert type(draft_model).__module__ == ("vllm_ascend.models.deepseek_v4_dspark")
        assert type(draft_model).__name__ == "DSparkDeepseekV4ForCausalLM"
        assert forbidden_module_tree_types(target_model) == []
        assert forbidden_module_tree_types(draft_model) == []
        assert forbidden_instance_attribute_types(speculator) == []
        assert get_tp_group().world_size == 8
        assert get_ep_group().world_size == 8
        assert get_pp_group().world_size == 1
        assert speculator.target_layer_ids == EXPECTED_TARGET_LAYER_IDS

        kv_cache_specs = worker.get_kv_cache_spec()
        indexer_kv_specs = {
            name: spec for name, spec in kv_cache_specs.items() if name.endswith(INDEXER_KV_CACHE_SUFFIX)
        }
        compressed_kv_specs = {
            name: spec
            for name, spec in kv_cache_specs.items()
            if isinstance(spec, AscendMLAAttentionSpec) and spec.compress_ratio > 1
        }
        assert indexer_kv_specs, "DeepSeek V4 prepare-only must discover at least one indexer KV cache."
        assert compressed_kv_specs, "DeepSeek V4 prepare-only must discover compressed KV caches."
        assert {spec.compress_ratio for spec in compressed_kv_specs.values()} == {4, 128}
        assert all(spec.block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE for spec in indexer_kv_specs.values())
        assert all(spec.block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE for spec in compressed_kv_specs.values())
        assert all(spec.storage_block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE for spec in compressed_kv_specs.values())
        available_memory = _kv_cache_budget(os.environ)
        kv_cache_config = get_kv_cache_configs(
            vllm_config,
            [kv_cache_specs],
            [available_memory],
        )[0]
        worker.initialize_from_config(kv_cache_config)
        assert speculator.block_tables is runner.block_tables
        assert speculator.kv_cache_config is runner.kv_cache_config

        forward_context = vllm_config.compilation_config.static_forward_context
        indexer_kv_diagnostics = []
        compressed_kv_diagnostics = []
        for layer_name, spec in sorted(compressed_kv_specs.items()):
            cache_owner = forward_context[layer_name]
            runtime_container = cache_owner.kv_cache
            assert isinstance(runtime_container, list)
            assert runtime_container
            primary_cache = runtime_container[0]
            assert primary_cache.dtype == spec.dtype
            assert primary_cache.ndim == 4
            assert primary_cache.shape[1] == spec.storage_block_size == PREPARE_ONLY_CACHE_BLOCK_SIZE

            is_indexer = layer_name.endswith(INDEXER_KV_CACHE_SUFFIX)
            cache_role = "indexer" if is_indexer else "compressed_kv"
            if is_indexer:
                assert len(runtime_container) == 2
                indexer_k_cache, indexer_scale_cache = runtime_container
                assert indexer_k_cache.dtype == torch.int8
                assert indexer_scale_cache.dtype == torch.float16
                assert indexer_scale_cache.ndim == 4
                assert indexer_scale_cache.shape[1] == spec.storage_block_size
                assert indexer_k_cache.shape[:3] == indexer_scale_cache.shape[:3]
                dsa_layer_name = layer_name.removesuffix(INDEXER_KV_CACHE_SUFFIX)
            else:
                assert layer_name.endswith(".attn")
                assert len(runtime_container) == 1
                indexer_scale_cache = None
                dsa_layer_name = layer_name.removesuffix(".attn")

            dsa_layer = forward_context[dsa_layer_name]
            forward_kv_cache = _build_kv_cache(
                dsa_layer,
                SimpleNamespace(virtual_engine=None),
            )
            if is_indexer:
                assert forward_kv_cache[4] is primary_cache
                assert forward_kv_cache[5] is indexer_scale_cache
            else:
                assert forward_kv_cache[0] is primary_cache
            assert sum(cache is runtime_container for cache in runner.kv_caches) == 1
            diagnostic = {
                "rank": launch.rank,
                "layer_name": layer_name,
                "cache_role": cache_role,
                "compress_ratio": spec.compress_ratio,
                "spec_block_size": spec.block_size,
                "storage_block_size": spec.storage_block_size,
                "runtime_shape": tuple(primary_cache.shape),
                "dtype": str(primary_cache.dtype),
                "binder_identity": True,
                "forward_identity": True,
            }
            if indexer_scale_cache is not None:
                diagnostic.update(
                    scale_shape=tuple(indexer_scale_cache.shape),
                    scale_dtype=str(indexer_scale_cache.dtype),
                )
                indexer_kv_diagnostics.append(
                    {
                        **diagnostic,
                        "key_shape": tuple(primary_cache.shape),
                        "key_dtype": str(primary_cache.dtype),
                    }
                )
            compressed_kv_diagnostics.append(diagnostic)
            print(
                "DSPARK_COMPRESSED_KV_CONTRACT=" + json.dumps(diagnostic, sort_keys=True),
                flush=True,
            )
        print(
            "DSPARK_INDEXER_BLOCK_SIZE_CONTRACT="
            + json.dumps(
                {
                    "rank": launch.rank,
                    "cache_config_block_size": vllm_config.cache_config.block_size,
                    "indexer_cache_count": len(indexer_kv_diagnostics),
                    "caches": indexer_kv_diagnostics,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        prepare_import_baseline = set(sys.modules)
        req_id = "dspark-prepare-only"
        token_id = _prompt_token_id(settings.target_config)
        num_kv_groups = len(kv_cache_config.kv_cache_groups)
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id=req_id,
                    prompt_token_ids=[token_id],
                    prefill_token_ids=[token_id],
                    mm_features=[],
                    sampling_params=SamplingParams(
                        max_tokens=1,
                        temperature=0.0,
                        seed=0,
                    ),
                    pooling_params=None,
                    block_ids=tuple([0] for _ in range(num_kv_groups)),
                    num_computed_tokens=0,
                    lora_request=None,
                )
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={req_id: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * num_kv_groups,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=[],
        )

        memory_before_forward = _memory_snapshot(torch, worker.device)
        output = worker.execute_model(scheduler_output)
        assert output is None
        execute_state = runner.execute_model_state
        assert execute_state is not None
        input_batch = execute_state.input_batch
        hidden_states = execute_state.hidden_states
        aux_hidden_states = execute_state.aux_hidden_states
        assert hidden_states is not None
        assert input_batch.req_ids == [req_id]
        tracker.mark(
            TARGET_FORWARD_COMPLETED,
            num_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            padded_tokens=input_batch.num_tokens_after_padding,
            memory_before=memory_before_forward,
            memory_after=_memory_snapshot(torch, worker.device),
        )

        assert aux_hidden_states is not None
        assert len(aux_hidden_states) == len(EXPECTED_TARGET_LAYER_IDS)
        expected_boundaries = tuple(layer_id + 1 for layer_id in EXPECTED_TARGET_LAYER_IDS)
        assert target_model.model.aux_hidden_state_layers == expected_boundaries
        assert all(
            tensor.shape[0] == input_batch.num_tokens_after_padding
            and tensor.dtype == hidden_states.dtype
            and tensor.device == hidden_states.device
            for tensor in aux_hidden_states
        )
        assert len({id(tensor) for tensor in aux_hidden_states}) == 3
        tracker.mark(
            AUX_HIDDEN_STATES_CAPTURED,
            target_layer_ids=EXPECTED_TARGET_LAYER_IDS,
            output_boundaries=expected_boundaries,
            shapes=[tuple(tensor.shape) for tensor in aux_hidden_states],
            dtype=str(aux_hidden_states[0].dtype),
            device=str(aux_hidden_states[0].device),
        )

        sampler_output, num_sampled, num_rejected = runner.sample(
            hidden_states,
            input_batch,
            None,
        )
        runner.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )
        pre_hc_hidden_states = target_model.get_mtp_target_hidden_states()
        assert pre_hc_hidden_states is not None
        spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]
        assert runner.sampler is not None
        assert execute_state.attn_metadata is not None
        assert execute_state.slot_mappings_by_layer is not None
        prepared = speculator.prepare_proposal_inputs(
            input_batch=input_batch,
            attn_metadata=execute_state.attn_metadata,
            slot_mappings=execute_state.slot_mappings_by_layer,
            last_hidden_states=spec_hidden_states,
            aux_hidden_states=aux_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=runner.req_states.last_sampled_tokens,
            next_prefill_tokens=runner.req_states.next_prefill_tokens,
            temperature=runner.sampler.sampling_states.temperature.gpu,
            seeds=runner.sampler.sampling_states.seeds.gpu,
        )
        assert type(prepared) is AscendDSparkProposalInputs
        assert prepared.rank == launch.rank
        assert prepared.request_ids == (req_id,)
        assert prepared.target_layer_ids == EXPECTED_TARGET_LAYER_IDS
        assert prepared.last_hidden_states is spec_hidden_states
        assert all(
            actual is expected
            for actual, expected in zip(
                prepared.auxiliary_hidden_states,
                aux_hidden_states,
            )
        )
        assert prepared.num_query_tokens == settings.num_speculative_tokens
        assert prepared.draft_input_ids.dtype == torch.int32
        assert prepared.draft_positions.dtype == torch.int64
        assert prepared.draft_query_start_loc.dtype == torch.int32
        assert prepared.draft_sequence_lengths.dtype == torch.int32
        speculator.validate_prepared_inputs_current(prepared)
        tracker.mark(
            PROPOSAL_INPUTS_PREPARED,
            request_ids=prepared.request_ids,
            target_token_shape=tuple(prepared.target_input_ids.shape),
            draft_token_shape=tuple(prepared.draft_input_ids.shape),
            draft_position_shape=tuple(prepared.draft_positions.shape),
            draft_sequence_lengths=tuple(prepared.draft_sequence_lengths.shape),
        )

        for layer_name, group_id in prepared.draft_layer_group_ids.items():
            assert prepared.draft_block_tables[group_id] is (runner.block_tables.input_block_tables[group_id])
            assert prepared.draft_query_slot_mappings[layer_name].data_ptr() == (
                runner.block_tables.slot_mappings[group_id].data_ptr()
            )
            assert prepared.draft_query_slot_mappings[layer_name].dtype == (torch.int32)
            assert prepared.draft_context_slot_mappings[layer_name].dtype == (torch.int32)
        assert (
            forbidden_import_delta(
                prepare_import_baseline,
                set(sys.modules),
            )
            == []
        )
        tracker.mark(
            PER_STEP_KV_METADATA_UPDATED,
            layer_group_ids=dict(prepared.draft_layer_group_ids),
            query_start_loc_shape=tuple(prepared.draft_query_start_loc.shape),
            slot_mapping_shapes={
                name: tuple(mapping.shape) for name, mapping in prepared.draft_query_slot_mappings.items()
            },
            block_table_identity=True,
        )

        torch.distributed.barrier()
        tracker.mark(
            PREPARE_ONLY_PASS,
            draft_forward=False,
            markov_sampling=False,
            proposal=False,
            generation=False,
        )
        runner.execute_model_state = None
        del (
            target_model,
            draft_model,
            runner,
            speculator,
            kv_cache_specs,
            indexer_kv_specs,
            indexer_kv_diagnostics,
            compressed_kv_specs,
            compressed_kv_diagnostics,
            kv_cache_config,
            scheduler_output,
            execute_state,
            input_batch,
            hidden_states,
            aux_hidden_states,
            sampler_output,
            spec_hidden_states,
            prepared,
        )
    except BaseException as exc:
        primary_error = exc
        tracker.failed(exc)
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
                ("release target/draft/KV state", release_state),
                ("destroy_ascend_model_parallel", destroy_ascend_groups),
                ("destroy_distributed_environment", destroy_vllm_groups),
                ("torch.npu.empty_cache", clear_npu_cache),
                ("vllm config context", config_context.close),
            ),
            tracker,
        )
        if cleanup_errors and primary_error is None:
            raise RuntimeError("DSpark prepare-only cleanup failed: " + "; ".join(cleanup_errors))
