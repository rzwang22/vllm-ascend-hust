# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.ut.spec_decode.test_dspark_v2_markov_sampling import (
    _ready_markov_step,
)
from tests.ut.spec_decode.test_dspark_v2_proposal_inputs import _step_kwargs
from vllm_ascend.spec_decode import DSparkRuntimeNotWiredError


def _publish_proposal(*, continue_after_verification: bool = False):
    speculator, proposal_inputs, _model, hidden_states = _ready_markov_step(
        continue_after_verification=continue_after_verification,
    )
    result = speculator._execute_sequential_markov_sampling(
        proposal_inputs,
        hidden_states,
    )
    proposal = speculator._build_core_proposal(proposal_inputs, result)
    return speculator, proposal_inputs, result, proposal


def _consumer_batch(proposal_inputs, proposal):
    batch = _step_kwargs()[0]["input_batch"]
    num_reqs, num_speculative_tokens = proposal.shape
    query_length = num_speculative_tokens + 1
    anchors = torch.tensor([17, 29], dtype=torch.int32)[:num_reqs]
    input_ids = torch.stack(
        [torch.cat((anchors[index : index + 1], proposal[index].to(torch.int32))) for index in range(num_reqs)]
    ).reshape(-1)
    query_start_loc = (
        torch.arange(
            num_reqs + 1,
            dtype=torch.int32,
        )
        * query_length
    )
    batch.num_scheduled_tokens = np.full(num_reqs, query_length, dtype=np.int32)
    batch.num_tokens = num_reqs * query_length
    batch.num_tokens_after_padding = batch.num_tokens
    batch.num_draft_tokens = num_reqs * num_speculative_tokens
    batch.num_draft_tokens_per_req = np.full(
        num_reqs,
        num_speculative_tokens,
        dtype=np.int32,
    )
    batch.query_start_loc = query_start_loc
    batch.query_start_loc_np = query_start_loc.numpy()
    batch.input_ids = input_ids
    batch.positions = torch.arange(batch.num_tokens, dtype=torch.int64)
    batch.is_padding = torch.zeros(batch.num_tokens, dtype=torch.bool)
    return batch


def test_markov_candidate_is_zero_copy_core_proposal() -> None:
    speculator, proposal_inputs, result, proposal = _publish_proposal()

    assert proposal is result.candidate_tokens
    assert proposal.data_ptr() == result.candidate_tokens.data_ptr()
    assert proposal.untyped_storage().data_ptr() == (result.candidate_tokens.untyped_storage().data_ptr())
    assert proposal.shape == (proposal_inputs.num_reqs, 5)
    assert proposal.dtype is torch.int64
    assert proposal.is_contiguous()
    assert speculator._published_proposal_step_epoch == proposal_inputs.step_epoch
    assert speculator._published_proposal_request_ids == proposal_inputs.request_ids
    assert speculator._proposal_publication_count == 1
    assert speculator.draft_logits is None


def test_candidate_cannot_be_published_twice() -> None:
    speculator, proposal_inputs, result, proposal = _publish_proposal()

    with pytest.raises(RuntimeError, match="already published"):
        speculator._build_core_proposal(proposal_inputs, result)

    assert speculator._published_candidate_tokens is proposal
    assert speculator._proposal_publication_count == 1


def test_consumer_rejects_stale_published_epoch_without_state_change() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal()
    consumer = _consumer_batch(proposal_inputs, proposal)
    speculator._proposal_step_epoch += 1

    with pytest.raises(RuntimeError, match="stale proposal epoch"):
        speculator._skip_next_proposal_after_verification(
            consumer,
            num_sampled=torch.tensor([1, 1], dtype=torch.int32),
            num_rejected=torch.tensor([5, 5], dtype=torch.int32),
            temperature=torch.zeros(2, dtype=torch.float32),
        )

    assert speculator._published_proposal_consumed is False
    assert speculator._next_proposal_skipped is False
    assert speculator._proposal_consumption_count == 0


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (
            lambda result: replace(result, step_epoch=result.step_epoch + 1),
            RuntimeError,
            "stale target step",
        ),
        (
            lambda result: replace(result, rank=result.rank + 1),
            RuntimeError,
            "different NPU rank",
        ),
        (
            lambda result: replace(result, request_ids=("wrong", "request")),
            RuntimeError,
            "request ownership",
        ),
        (
            lambda result: replace(result, candidate_tokens=result.candidate_tokens.to(torch.int32)),
            TypeError,
            "candidate tokens",
        ),
        (
            lambda result: replace(
                result,
                steps=result.steps[:-1],
            ),
            RuntimeError,
            "every Markov step",
        ),
        (
            lambda result: replace(
                result,
                candidate_tokens=result.candidate_tokens.to("meta"),
            ),
            RuntimeError,
            "current rank device",
        ),
    ],
)
def test_invalid_candidate_never_publishes_partial_state(mutation, error, match) -> None:
    speculator, proposal_inputs, _model, hidden_states = _ready_markov_step()
    result = speculator._execute_sequential_markov_sampling(
        proposal_inputs,
        hidden_states,
    )

    invalid_result = mutation(result)
    speculator._markov_result = invalid_result
    with pytest.raises(error, match=match):
        speculator._build_core_proposal(proposal_inputs, invalid_result)

    assert speculator._published_candidate_tokens is None
    assert speculator._published_proposal_step_epoch is None
    assert speculator._proposal_publication_count == 0


def test_verification_consumes_same_candidates_and_returns_none() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal()
    consumer = _consumer_batch(proposal_inputs, proposal)

    skipped = speculator._skip_next_proposal_after_verification(
        consumer,
        num_sampled=torch.tensor([1, 4], dtype=torch.int32),
        num_rejected=torch.tensor([5, 2], dtype=torch.int32),
        temperature=torch.zeros(2, dtype=torch.float32),
    )

    assert skipped is None
    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_consumed is True
    assert speculator._next_proposal_skipped is True
    assert speculator._proposal_consumer_step_epoch == proposal_inputs.step_epoch + 1
    assert speculator._proposal_consumption_count == 1
    assert speculator._next_proposal_skip_count == 1
    assert speculator._markov_result is None


def test_public_propose_skips_without_running_second_draft(monkeypatch) -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal()
    consumer = _consumer_batch(proposal_inputs, proposal)
    kwargs, _auxiliary_states = _step_kwargs()
    kwargs["input_batch"] = consumer
    kwargs["num_sampled"] = torch.tensor([6, 1], dtype=torch.int32)
    kwargs["num_rejected"] = torch.tensor([0, 5], dtype=torch.int32)

    monkeypatch.setattr(
        speculator,
        "_execute_draft_backbone",
        pytest.fail,
    )
    skipped = speculator.propose(**kwargs)

    assert skipped is None
    assert speculator._proposal_publication_count == 1
    assert speculator._proposal_consumption_count == 1
    assert speculator._next_proposal_skip_count == 1


def test_third_proposer_fails_closed_at_multi_round_boundary() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal()
    consumer = _consumer_batch(proposal_inputs, proposal)
    kwargs, _auxiliary_states = _step_kwargs()
    kwargs["input_batch"] = consumer
    kwargs["num_sampled"] = torch.tensor([1, 1], dtype=torch.int32)
    kwargs["num_rejected"] = torch.tensor([5, 5], dtype=torch.int32)
    assert speculator.propose(**kwargs) is None

    with pytest.raises(
        DSparkRuntimeNotWiredError,
        match="M2.4B multi-round DSpark lifecycle",
    ):
        speculator.propose(**kwargs)


@pytest.mark.parametrize(
    ("mutate", "expected_exception", "match"),
    [
        (
            lambda batch: setattr(
                batch,
                "num_draft_tokens_per_req",
                np.array([5, 4], dtype=np.int32),
            ),
            RuntimeError,
            "exactly the configured",
        ),
        (
            lambda batch: batch.input_ids.__setitem__(1, 255),
            ValueError,
            "published candidate set",
        ),
        (
            lambda batch: setattr(batch, "req_ids", ["wrong", "request"]),
            RuntimeError,
            "request ownership",
        ),
    ],
)
def test_consumer_mismatch_does_not_consume_published_proposal(
    mutate,
    expected_exception,
    match,
    monkeypatch,
) -> None:
    speculator, proposal_inputs, markov_result, proposal = _publish_proposal()
    consumer = _consumer_batch(proposal_inputs, proposal)
    mutate(consumer)

    request_state_indices = speculator._published_proposal_request_state_indices
    draft_forward_epoch = speculator._draft_forward_step_epoch
    markov_attempt_epoch = speculator._markov_attempt_step_epoch
    markov_epoch = speculator._markov_step_epoch

    def fail_unexpected_execution(*_args, **_kwargs):
        pytest.fail("verification mismatch started another draft or Markov step")

    monkeypatch.setattr(speculator, "_execute_draft_backbone", fail_unexpected_execution)
    monkeypatch.setattr(speculator, "_execute_sequential_markov_sampling", fail_unexpected_execution)

    return_sentinel = object()
    return_value = return_sentinel
    with pytest.raises(expected_exception, match=match):
        return_value = speculator._skip_next_proposal_after_verification(
            consumer,
            num_sampled=torch.tensor([1, 1], dtype=torch.int32),
            num_rejected=torch.tensor([5, 5], dtype=torch.int32),
            temperature=torch.zeros(2, dtype=torch.float32),
        )

    assert return_value is return_sentinel
    assert speculator._proposal_step_epoch == proposal_inputs.step_epoch
    assert speculator._published_proposal_step_epoch == proposal_inputs.step_epoch
    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_request_ids == proposal_inputs.request_ids
    assert speculator._published_proposal_request_state_indices is request_state_indices
    assert speculator._published_proposal_consumed is False
    assert speculator._next_proposal_skipped is False
    assert speculator._proposal_consumer_step_epoch is None
    assert speculator._proposal_consumption_count == 0
    assert speculator._next_proposal_skip_count == 0
    assert speculator._draft_forward_step_epoch == draft_forward_epoch
    assert speculator._markov_attempt_step_epoch == markov_attempt_epoch
    assert speculator._markov_step_epoch == markov_epoch
    assert speculator._markov_result is markov_result


def test_stochastic_consumer_fails_before_skip_publication() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal()
    consumer = _consumer_batch(proposal_inputs, proposal)

    with pytest.raises(ValueError, match="deterministic greedy"):
        speculator._skip_next_proposal_after_verification(
            consumer,
            num_sampled=torch.tensor([1, 1], dtype=torch.int32),
            num_rejected=torch.tensor([5, 5], dtype=torch.int32),
            temperature=torch.tensor([0.0, 0.5], dtype=torch.float32),
        )

    assert speculator._published_proposal_consumed is False
    assert speculator._next_proposal_skipped is False


def test_incomplete_verification_counts_fail_before_consumption() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal()
    consumer = _consumer_batch(proposal_inputs, proposal)

    with pytest.raises(ValueError, match="full proposal window"):
        speculator._skip_next_proposal_after_verification(
            consumer,
            num_sampled=torch.tensor([1, 1], dtype=torch.int32),
            num_rejected=torch.tensor([4, 5], dtype=torch.int32),
            temperature=torch.zeros(2, dtype=torch.float32),
        )

    assert speculator._published_proposal_consumed is False
    assert speculator._next_proposal_skipped is False
    assert speculator._proposal_consumption_count == 0
