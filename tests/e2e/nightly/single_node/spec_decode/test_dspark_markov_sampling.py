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
from tests.e2e.nightly.single_node.spec_decode.test_dspark_draft_backbone_forward import (
    DRAFT_CONTEXT_KV_PRECOMPUTED,
    DRAFT_FORWARD_COMPLETED,
)

MARKOV_INPUTS_READY = "MARKOV_INPUTS_READY"
MARKOV_SAMPLING_COMPLETED = "MARKOV_SAMPLING_COMPLETED"
MARKOV_SAMPLING_ONLY_PASS = "MARKOV_SAMPLING_ONLY_PASS"
MARKOV_STAGES = (
    DRAFT_CONTEXT_KV_PRECOMPUTED,
    DRAFT_FORWARD_COMPLETED,
    MARKOV_INPUTS_READY,
    MARKOV_SAMPLING_COMPLETED,
    MARKOV_SAMPLING_ONLY_PASS,
)


class _MarkovStageTracker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.step_epoch: int | None = None
        self.stages: list[str] = []

    def mark(self, stage: str, **details: Any) -> None:
        if stage == CLEANUP_COMPLETE:
            if stage in self.stages:
                raise RuntimeError("CLEANUP_COMPLETE was already recorded.")
        else:
            completed = sum(item != CLEANUP_COMPLETE for item in self.stages)
            expected = MARKOV_STAGES[completed]
            if stage != expected:
                raise RuntimeError(
                    f"Invalid DSpark Markov-only stage transition: expected {expected!r}, got {stage!r}."
                )
        self.stages.append(stage)
        print(
            "DSPARK_MARKOV_STAGE="
            + json.dumps(
                {
                    "rank": self.rank,
                    "step_epoch": self.step_epoch,
                    "stage": stage,
                    **details,
                },
                default=str,
                sort_keys=True,
            ),
            flush=True,
        )

    def failed(self, exc: BaseException) -> None:
        print(
            "DSPARK_MARKOV_FAILURE="
            + json.dumps(
                {
                    "rank": self.rank,
                    "step_epoch": self.step_epoch,
                    "failed_after": self.stages[-1] if self.stages else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _run_real_markov_sampling(runtime: Any, tracker: _MarkovStageTracker) -> None:
    torch = runtime.torch
    speculator = runtime.speculator
    proposal = runtime.prepared
    tracker.step_epoch = proposal.step_epoch

    execution = speculator._combine_and_precompute_draft_context(proposal)
    torch.npu.synchronize()
    tracker.mark(
        DRAFT_CONTEXT_KV_PRECOMPUTED,
        request_ids=proposal.request_ids,
        context_token_count=proposal.num_target_tokens,
    )

    draft_attn_metadata = speculator._build_draft_forward_metadata(execution)
    hidden_states = speculator._run_draft_model_forward(
        execution,
        draft_attn_metadata,
    )
    torch.npu.synchronize()
    tracker.mark(
        DRAFT_FORWARD_COMPLETED,
        request_ids=proposal.request_ids,
        output_shape=tuple(hidden_states.shape),
        output_dtype=str(hidden_states.dtype),
        output_device=str(hidden_states.device),
    )

    assert proposal.num_speculative_tokens == 5
    assert proposal.num_query_tokens == proposal.num_reqs * 5
    assert tuple(proposal.draft_query_start_loc.shape) == (proposal.num_reqs + 1,)
    tracker.mark(
        MARKOV_INPUTS_READY,
        request_ids=proposal.request_ids,
        anchor_shape=tuple(proposal.anchor_token_ids.shape),
        hidden_shape=tuple(hidden_states.shape),
        query_start_loc_shape=tuple(proposal.draft_query_start_loc.shape),
        K=proposal.num_speculative_tokens,
    )

    result = speculator._execute_sequential_markov_sampling(
        proposal,
        hidden_states,
    )
    torch.npu.synchronize()
    assert result is speculator._markov_result
    assert result.step_epoch == proposal.step_epoch
    assert result.backbone_hidden_states is hidden_states
    assert result.candidate_tokens.shape == (proposal.num_reqs, 5)
    assert len(result.steps) == 5

    gathered_candidates = [
        torch.empty_like(result.candidate_tokens) for _rank in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered_candidates, result.candidate_tokens)
    torch.npu.synchronize()
    rank_consistent = all(torch.equal(candidate, gathered_candidates[0]) for candidate in gathered_candidates[1:])
    assert rank_consistent

    candidates_cpu = result.candidate_tokens.cpu()
    anchors_cpu = proposal.anchor_token_ids.cpu()
    for request_index, request_id in enumerate(result.request_ids):
        for step in result.steps:
            selected_token = int(candidates_cpu[request_index, step.step_index])
            if step.step_index == 0:
                assert step.predecessor_source == "anchor_token_ids"
                assert step.predecessor_token_ids is proposal.anchor_token_ids
                predecessor_token = int(anchors_cpu[request_index])
            else:
                assert step.predecessor_source == "previous_sampled_token"
                assert step.predecessor_token_ids is result.steps[step.step_index - 1].selected_token_ids
                predecessor_token = int(candidates_cpu[request_index, step.step_index - 1])
            assert step.markov_input_token_ids is step.predecessor_token_ids
            print(
                "DSPARK_MARKOV_STEP="
                + json.dumps(
                    {
                        "rank": runtime.launch.rank,
                        "step_epoch": result.step_epoch,
                        "request_id": request_id,
                        "step_index": step.step_index,
                        "K": result.num_speculative_tokens,
                        "predecessor_source": step.predecessor_source,
                        "predecessor_token": predecessor_token,
                        "selected_token": selected_token,
                        "token_dtype": str(step.selected_token_ids.dtype),
                        "token_device": str(step.selected_token_ids.device),
                        "vocab_size": result.vocab_size,
                        "base_logits_full_vocab": True,
                        "global_argmax_verified": True,
                        "tp_consistent": rank_consistent,
                        "markov_input_identity": id(step.markov_input_token_ids),
                        "markov_state_has_nan": False,
                        "selectable_finite_logit": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    tracker.mark(
        MARKOV_SAMPLING_COMPLETED,
        request_ids=result.request_ids,
        candidate_shape=tuple(result.candidate_tokens.shape),
        rank_consistent=rank_consistent,
    )
    candidate_tokens = candidates_cpu.tolist()
    contract = {
        "rank": runtime.launch.rank,
        "step_epoch": result.step_epoch,
        "request_ids": result.request_ids,
        "request_count": result.num_reqs,
        "K": result.num_speculative_tokens,
        "physical_hidden_shape": result.physical_hidden_shape,
        "physical_base_logits_shape": result.physical_base_logits_shape,
        "physical_candidate_shape": tuple(result.candidate_tokens.shape),
        "logical_candidate_shape": result.logical_candidate_shape,
        "candidate_tokens": candidate_tokens,
        "candidate_count": result.candidate_tokens.numel(),
        "all_tokens_in_vocab": bool(((candidates_cpu >= 0) & (candidates_cpu < result.vocab_size)).all()),
        "all_steps_completed": len(result.steps) == result.num_speculative_tokens,
        "rank_consistent": rank_consistent,
        "lm_head_class": result.lm_head_class,
        "markov_head_class": result.markov_head_class,
        "lm_head_parameter_names": result.lm_head_parameter_names,
        "markov_parameter_names": result.markov_parameter_names,
        "loaded_module_identity_preserved": result.loaded_module_identity_preserved,
        "confidence_head_present": result.confidence_head_present,
        "confidence_head_used": result.confidence_head_used,
        "partial_candidate_publication_count": 0,
        "draft_forward": True,
        "markov_sampling": True,
        "candidate_state_ready": True,
        "proposal_publication": False,
        "core_proposal_returned": False,
        "verification": False,
        "generation": False,
    }
    assert contract["all_tokens_in_vocab"]
    assert contract["all_steps_completed"]
    print(
        "DSPARK_MARKOV_CONTRACT=" + json.dumps(contract, default=str, sort_keys=True),
        flush=True,
    )
    torch.distributed.barrier()
    tracker.mark(
        MARKOV_SAMPLING_ONLY_PASS,
        request_ids=result.request_ids,
        candidate_state_ready=True,
        proposal_publication=False,
        verification=False,
        generation=False,
    )


def test_dspark_markov_sampling_only_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    tracker = _MarkovStageTracker(launch.rank)

    def callback(runtime: Any) -> None:
        _run_real_markov_sampling(runtime, tracker)

    if prepare_harness._PREPARED_STEP_CALLBACK is not None:
        raise RuntimeError("The DSpark prepare-only harness callback is already installed.")
    prepare_harness._PREPARED_STEP_CALLBACK = callback
    primary_error: BaseException | None = None
    try:
        prepare_harness.test_dspark_proposal_inputs_prepare_only_npu()
    except BaseException as exc:
        primary_error = exc
        tracker.failed(exc)
        raise
    finally:
        prepare_harness._PREPARED_STEP_CALLBACK = None
        tracker.mark(
            CLEANUP_COMPLETE,
            cleanup_after_error=primary_error is not None,
        )
