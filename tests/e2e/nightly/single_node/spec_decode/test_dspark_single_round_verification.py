# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import json
import os
from typing import Any

import pytest

import tests.e2e.nightly.single_node.spec_decode.test_dspark_proposal_inputs_prepare as prepare_harness
from tests.e2e.nightly.single_node.spec_decode.dspark_loader_harness import (
    CLEANUP_COMPLETE,
    HarnessNotConfigured,
    parse_launch_context,
    parse_loader_settings,
)

PROPOSAL_PUBLICATION_COMPLETED = "PROPOSAL_PUBLICATION_COMPLETED"
TARGET_VERIFICATION_COMPLETED = "TARGET_VERIFICATION_COMPLETED"
REJECTION_SAMPLING_COMPLETED = "REJECTION_SAMPLING_COMPLETED"
NEXT_PROPOSAL_SKIPPED = "NEXT_PROPOSAL_SKIPPED"
SCHEDULER_VERIFICATION_COMMITTED = "SCHEDULER_VERIFICATION_COMMITTED"
SINGLE_ROUND_VERIFICATION_PASS = "SINGLE_ROUND_VERIFICATION_PASS"
DEVICE_STAGES = (
    PROPOSAL_PUBLICATION_COMPLETED,
    TARGET_VERIFICATION_COMPLETED,
    REJECTION_SAMPLING_COMPLETED,
    NEXT_PROPOSAL_SKIPPED,
    SINGLE_ROUND_VERIFICATION_PASS,
)


class _SingleRoundStageTracker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.stages: list[str] = []

    def mark(self, stage: str, **details: Any) -> None:
        if stage == CLEANUP_COMPLETE:
            if stage in self.stages:
                raise RuntimeError("CLEANUP_COMPLETE was already recorded.")
        elif stage == SCHEDULER_VERIFICATION_COMMITTED:
            if self.rank != 0:
                raise RuntimeError("Only the scheduler owner may record its commit.")
        else:
            device_stages = [item for item in self.stages if item in DEVICE_STAGES]
            expected = DEVICE_STAGES[len(device_stages)]
            if stage != expected:
                raise RuntimeError(
                    f"Invalid DSpark single-round stage transition: expected {expected!r}, got {stage!r}."
                )
        self.stages.append(stage)
        print(
            "DSPARK_M24A_STAGE="
            + json.dumps(
                {"rank": self.rank, "stage": stage, **details},
                default=str,
                sort_keys=True,
            ),
            flush=True,
        )

    def failed(self, exc: BaseException) -> None:
        print(
            "DSPARK_M24A_FAILURE="
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


def _materialize_model_output(output: Any) -> Any:
    return output.get_output() if hasattr(output, "get_output") else output


def _assert_tp_consistent(torch: Any, tensor: Any) -> bool:
    gathered = [torch.empty_like(tensor) for _rank in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    torch.npu.synchronize()
    consistent = all(bool((rank_tensor == gathered[0]).all()) for rank_tensor in gathered[1:])
    if not consistent:
        raise RuntimeError("DSpark single-round tensors differ across TP ranks.")
    return True


def _expected_greedy_verification(
    candidates: list[int],
    target_selected: list[int],
) -> tuple[list[int], int, bool, bool]:
    if len(target_selected) != len(candidates) + 1:
        raise RuntimeError("Target verification must produce K+1 causal choices.")
    mismatch_index = next(
        (
            index
            for index, (draft_token, target_token) in enumerate(zip(candidates, target_selected))
            if draft_token != target_token
        ),
        None,
    )
    if mismatch_index is None:
        return candidates + [target_selected[-1]], len(candidates), False, True
    return (
        candidates[:mismatch_index] + [target_selected[mismatch_index]],
        mismatch_index,
        True,
        False,
    )


def _build_scheduler(runtime: Any) -> Any:
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        resolve_kv_cache_block_sizes,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.structured_output import StructuredOutputManager

    vllm_config = runtime.vllm_config
    if vllm_config.kv_cache_compression_config is not None:
        raise RuntimeError("The DSpark single-round harness does not bypass the core KV compression runtime contract.")
    scheduler_kv_config = generate_scheduler_kv_cache_config([runtime.kv_cache_config])
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_config.num_blocks
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        scheduler_kv_config,
        vllm_config,
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=scheduler_kv_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )


def _target_greedy_choices(runtime: Any) -> Any:
    runner = runtime.runner
    state = runner.execute_model_state
    if state is None or state.hidden_states is None:
        raise RuntimeError("The target verification forward produced no hidden state.")
    input_batch = state.input_batch
    sample_hidden_states = state.hidden_states[input_batch.logits_indices]
    logits = runner.model.compute_logits(sample_hidden_states)
    if runner.sampler is None:
        raise RuntimeError("The target sampler is unavailable.")
    processed_logits = runner.sampler.apply_sampling_params(
        logits,
        input_batch.expanded_idx_mapping,
        input_batch.idx_mapping_np,
        input_batch.positions[input_batch.logits_indices],
        input_batch.input_ids[input_batch.logits_indices],
        input_batch.expanded_local_pos,
    )
    target_selected = processed_logits.argmax(dim=-1).view(
        input_batch.num_reqs,
        -1,
    )
    del logits, processed_logits, sample_hidden_states
    return target_selected


def _run_single_round(runtime: Any, tracker: _SingleRoundStageTracker) -> bool:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    torch = runtime.torch
    worker = runtime.worker
    runner = runtime.runner
    speculator = runtime.speculator
    settings = runtime.settings
    scheduler = _build_scheduler(runtime)
    request_id = "dspark-m24a-single-round"
    token_id = prepare_harness._prompt_token_id(settings.target_config)
    request = Request(
        request_id=request_id,
        prompt_token_ids=[token_id],
        sampling_params=SamplingParams(
            max_tokens=settings.num_speculative_tokens + 8,
            temperature=0.0,
            seed=0,
        ),
        pooling_params=None,
    )
    scheduler.add_request(request)

    target_forward_count = 0
    draft_forward_count = 0
    markov_step_count = 0
    verification_count = 0

    producer_output = scheduler.schedule()
    if producer_output.scheduled_spec_decode_tokens:
        raise RuntimeError("The producer step unexpectedly scheduled draft tokens.")
    if worker.execute_model(producer_output) is not None:
        raise RuntimeError("PP=1 target execution must defer output to sampling.")
    target_forward_count += 1
    producer_async_output = worker.sample_tokens(None)
    if producer_async_output is None:
        raise RuntimeError("The producer step returned no target output.")
    torch.npu.synchronize()

    candidate_tensor = speculator._published_candidate_tokens
    proposal_epoch = speculator._published_proposal_step_epoch
    if candidate_tensor is None or proposal_epoch is None:
        raise RuntimeError("The first DSpark proposer did not publish candidates.")
    expected_shape = (1, settings.num_speculative_tokens)
    if tuple(candidate_tensor.shape) != expected_shape:
        raise RuntimeError(f"DSpark proposal shape must be {expected_shape}, got {tuple(candidate_tensor.shape)}.")
    proposal_rank_consistent = _assert_tp_consistent(torch, candidate_tensor)
    candidates = candidate_tensor.cpu().view(-1).tolist()
    draft_forward_count += 1
    markov_step_count += settings.num_speculative_tokens

    producer_model_output = _materialize_model_output(producer_async_output)
    scheduler.update_from_output(producer_output, producer_model_output)
    first_draft_ids = worker.take_draft_token_ids()
    if first_draft_ids is None:
        raise RuntimeError("The first DSpark proposal did not reach the handler.")
    scheduler.update_draft_token_ids(first_draft_ids)
    if len(request.spec_token_ids) != settings.num_speculative_tokens:
        raise RuntimeError("The scheduler did not record the full proposal length.")

    proposal_contract = {
        "rank": runtime.launch.rank,
        "step_epoch": proposal_epoch,
        "request_id": request_id,
        "request_count": 1,
        "K": settings.num_speculative_tokens,
        "physical_shape": tuple(candidate_tensor.shape),
        "logical_shape": tuple(candidate_tensor.shape),
        "dtype": str(candidate_tensor.dtype),
        "device": str(candidate_tensor.device),
        "candidate_tokens": candidates,
        "candidate_count": len(candidates),
        "all_tokens_in_vocab": all(
            0 <= token < runtime.vllm_config.model_config.get_vocab_size() for token in candidates
        ),
        "markov_steps_completed": markov_step_count,
        "proposal_published": True,
        "draft_buffer_shape": tuple(runner.req_states.draft_tokens.shape),
        "scheduled_proposal_length": len(request.spec_token_ids),
        "cross_rank_consistent": proposal_rank_consistent,
    }
    print(
        "DSPARK_M24A_PROPOSAL_CONTRACT=" + json.dumps(proposal_contract, sort_keys=True),
        flush=True,
    )
    tracker.mark(
        PROPOSAL_PUBLICATION_COMPLETED,
        proposal_epoch=proposal_epoch,
        request_id=request_id,
        K=settings.num_speculative_tokens,
    )

    consumer_output = scheduler.schedule()
    scheduled_placeholders = consumer_output.scheduled_spec_decode_tokens.get(request_id)
    if scheduled_placeholders is None or len(scheduled_placeholders) != (settings.num_speculative_tokens):
        raise RuntimeError("The consumer step did not schedule the published K tokens.")
    if worker.execute_model(consumer_output) is not None:
        raise RuntimeError("PP=1 target execution must defer output to sampling.")
    target_forward_count += 1
    consumer_input = runner.execute_model_state.input_batch
    target_input_count = consumer_input.num_scheduled_tokens.tolist()
    if target_input_count != [settings.num_speculative_tokens + 1]:
        raise RuntimeError("Target verification must consume anchor + K tokens.")
    consumed_candidates_tensor = consumer_input.input_ids[1 : settings.num_speculative_tokens + 1].to(torch.int64)
    if not bool((consumed_candidates_tensor == candidate_tensor[0]).all()):
        raise RuntimeError("The target did not consume the exact published candidates.")
    target_selected_tensor = _target_greedy_choices(runtime)
    target_rank_consistent = _assert_tp_consistent(torch, target_selected_tensor)
    tracker.mark(
        TARGET_VERIFICATION_COMPLETED,
        proposal_epoch=proposal_epoch,
        request_id=request_id,
        target_input_token_count=consumer_input.num_tokens,
        verification_logits_shape=(
            target_selected_tensor.shape[0],
            target_selected_tensor.shape[1],
            runtime.vllm_config.model_config.get_vocab_size(),
        ),
    )

    consumer_async_output = worker.sample_tokens(None)
    if consumer_async_output is None:
        raise RuntimeError("The verification step returned no target output.")
    torch.npu.synchronize()
    verification_count += 1
    sampled_device = consumer_async_output.sampler_output.sampled_token_ids
    num_sampled_device = consumer_async_output.num_sampled_tokens
    count_rank_consistent = _assert_tp_consistent(torch, num_sampled_device)
    valid_sampled_count = int(num_sampled_device[0].cpu())
    sampled_rank_consistent = _assert_tp_consistent(
        torch,
        sampled_device[:, :valid_sampled_count],
    )
    consumer_model_output = _materialize_model_output(consumer_async_output)
    verified_tokens = consumer_model_output.sampled_token_ids[0]
    if len(verified_tokens) != valid_sampled_count:
        raise RuntimeError("Async output truncated the verified token count incorrectly.")
    target_selected = target_selected_tensor.cpu().view(-1).tolist()
    expected_tokens, accepted_prefix_length, replacement_used, bonus_used = _expected_greedy_verification(
        candidates, target_selected
    )
    if verified_tokens != expected_tokens:
        raise RuntimeError(
            "Core rejection sampling output does not match the independently derived greedy target contract."
        )
    tracker.mark(
        REJECTION_SAMPLING_COMPLETED,
        proposal_epoch=proposal_epoch,
        request_id=request_id,
        verified_output_count=len(verified_tokens),
    )

    if not speculator._next_proposal_skipped:
        raise RuntimeError("The verification proposer did not skip the next proposal.")
    if speculator._proposal_consumption_count != 1:
        raise RuntimeError("The proposal was not consumed exactly once.")
    if speculator._next_proposal_skip_count != 1:
        raise RuntimeError("The next proposal was not skipped exactly once.")
    tracker.mark(
        NEXT_PROPOSAL_SKIPPED,
        proposal_epoch=proposal_epoch,
        consumer_epoch=speculator._proposal_consumer_step_epoch,
        next_proposal_length=0,
    )

    tokens_before_commit = list(request.output_token_ids)
    scheduler.update_from_output(consumer_output, consumer_model_output)
    next_draft_ids = worker.take_draft_token_ids()
    if next_draft_ids is None:
        raise RuntimeError("The canonical empty proposal did not reach the handler.")
    if next_draft_ids.draft_token_ids != [[]]:
        raise RuntimeError("The next proposal must be canonical and zero length.")
    scheduler.update_draft_token_ids(next_draft_ids)
    if request.spec_token_ids:
        raise RuntimeError("The scheduler retained stale draft tokens after the skip.")
    committed_tokens = list(request.output_token_ids)[len(tokens_before_commit) :]
    if committed_tokens != verified_tokens:
        raise RuntimeError("The scheduler did not commit the verified output exactly.")

    consumer_epoch = speculator._proposal_consumer_step_epoch
    verification_contract = {
        "rank": runtime.launch.rank,
        "proposal_epoch": proposal_epoch,
        "consumer_epoch": consumer_epoch,
        "request_id": request_id,
        "K": settings.num_speculative_tokens,
        "scheduled_candidate_tokens": consumed_candidates_tensor.cpu().tolist(),
        "target_input_token_count": consumer_input.num_tokens,
        "verification_logits_shape": (
            target_selected_tensor.shape[0],
            target_selected_tensor.shape[1],
            runtime.vllm_config.model_config.get_vocab_size(),
        ),
        "accepted_prefix_length": accepted_prefix_length,
        "replacement_used": replacement_used,
        "bonus_used": bonus_used,
        "verified_output_tokens": verified_tokens,
        "verified_output_count": len(verified_tokens),
        "output_dtype": str(sampled_device.dtype),
        "output_device": str(sampled_device.device),
        "cross_rank_consistent": (target_rank_consistent and sampled_rank_consistent and count_rank_consistent),
        "target_verification": True,
        "rejection_sampling": True,
        "next_proposal_length": 0,
    }
    print(
        "DSPARK_M24A_VERIFICATION_CONTRACT=" + json.dumps(verification_contract, sort_keys=True),
        flush=True,
    )
    if runtime.launch.rank == 0:
        scheduler_contract = {
            "owner_rank": 0,
            "request_id": request_id,
            "verification_output_committed": True,
            "committed_tokens": committed_tokens,
            "committed_token_count": len(committed_tokens),
            "next_scheduled_proposal_length": len(request.spec_token_ids),
            "single_round_complete": True,
        }
        print(
            "DSPARK_M24A_SCHEDULER_COMMIT=" + json.dumps(scheduler_contract, sort_keys=True),
            flush=True,
        )
        tracker.mark(
            SCHEDULER_VERIFICATION_COMMITTED,
            request_id=request_id,
            committed_token_count=len(committed_tokens),
        )

    counters = {
        "rank": runtime.launch.rank,
        "target_forward_count": target_forward_count,
        "draft_forward_count": draft_forward_count,
        "markov_step_count": markov_step_count,
        "proposal_publication_count": speculator._proposal_publication_count,
        "verification_count": verification_count,
        "next_proposal_skip_count": speculator._next_proposal_skip_count,
        "third_target_forward_count": 0,
        "second_draft_forward_count": 0,
        "second_markov_sequence_count": 0,
        "partial_proposal_publication_count": 0,
    }
    print(
        "DSPARK_M24A_COUNTERS=" + json.dumps(counters, sort_keys=True),
        flush=True,
    )
    torch.distributed.barrier()
    tracker.mark(
        SINGLE_ROUND_VERIFICATION_PASS,
        proposal_epoch=proposal_epoch,
        consumer_epoch=consumer_epoch,
        scheduler_committed=True,
        multi_round=False,
        generation=False,
    )
    return True


def test_dspark_single_round_proposal_and_verification_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    tracker = _SingleRoundStageTracker(launch.rank)

    def callback(runtime: Any) -> bool:
        return _run_single_round(runtime, tracker)

    if prepare_harness._INITIALIZED_WORKER_CALLBACK is not None:
        raise RuntimeError("The DSpark initialized-worker callback is already installed.")
    prepare_harness._INITIALIZED_WORKER_CALLBACK = callback
    primary_error: BaseException | None = None
    try:
        prepare_harness.test_dspark_proposal_inputs_prepare_only_npu()
    except BaseException as exc:
        primary_error = exc
        tracker.failed(exc)
        raise
    finally:
        prepare_harness._INITIALIZED_WORKER_CALLBACK = None
        tracker.mark(
            CLEANUP_COMPLETE,
            cleanup_after_error=primary_error is not None,
        )
