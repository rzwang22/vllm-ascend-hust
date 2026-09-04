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

REQUEST_IDS = ("request-1", "request-2", "request-3", "request-4")
REQUEST_STATE_INDICES = torch.tensor([3, 0, 2, 1], dtype=torch.int32)
VERIFICATION_ORDER = ("request-4", "request-3", "request-1", "request-2")


def _publish_batched_proposal(
    request_ids: tuple[str, ...] = REQUEST_IDS,
):
    speculator, proposal_inputs, _model, hidden_states = _ready_markov_step(
        continue_after_verification=True,
    )
    result = speculator._execute_sequential_markov_sampling(
        proposal_inputs,
        hidden_states,
    )
    num_reqs = len(request_ids)
    request_state_indices = REQUEST_STATE_INDICES[:num_reqs].clone()
    candidates = (
        torch.arange(num_reqs, dtype=torch.int64)[:, None] * 20 + torch.arange(1, 6, dtype=torch.int64)[None, :] + 10
    )
    proposal_inputs = replace(
        proposal_inputs,
        request_ids=request_ids,
        request_state_indices=request_state_indices,
        num_reqs=num_reqs,
    )
    result = replace(
        result,
        request_ids=request_ids,
        request_state_indices=request_state_indices,
        num_reqs=num_reqs,
        candidate_tokens=candidates,
        logical_candidate_shape=tuple(candidates.shape),
    )
    speculator._markov_result = result
    proposal = speculator._build_core_proposal(proposal_inputs, result)
    return speculator, proposal_inputs, proposal


def _candidate_rows_by_request(
    proposal_inputs,
    proposal: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {request_id: proposal[row] for row, request_id in enumerate(proposal_inputs.request_ids)}


def _state_index_by_request(proposal_inputs) -> dict[str, int]:
    return {
        request_id: int(proposal_inputs.request_state_indices[row])
        for row, request_id in enumerate(proposal_inputs.request_ids)
    }


def _consumer_batch(
    proposal_inputs,
    proposal: torch.Tensor,
    request_order: tuple[str, ...],
    lengths_by_request: dict[str, int],
):
    batch = _step_kwargs()[0]["input_batch"]
    candidates_by_request = _candidate_rows_by_request(
        proposal_inputs,
        proposal,
    )
    state_indices_by_request = _state_index_by_request(proposal_inputs)
    rows = []
    query_lengths = []
    for verification_row, request_id in enumerate(request_order):
        scheduled_length = lengths_by_request[request_id]
        anchor = torch.tensor([100 + verification_row], dtype=torch.int32)
        rows.append(
            torch.cat(
                (
                    anchor,
                    candidates_by_request[request_id][:scheduled_length].to(torch.int32),
                )
            )
        )
        query_lengths.append(scheduled_length + 1)

    query_start_loc = torch.tensor(
        [0, *np.cumsum(query_lengths).tolist()],
        dtype=torch.int32,
    )
    state_indices = torch.tensor(
        [state_indices_by_request[request_id] for request_id in request_order],
        dtype=torch.int32,
    )
    scheduled_lengths = np.asarray(
        [lengths_by_request[request_id] for request_id in request_order],
        dtype=np.int32,
    )
    batch.req_ids = list(request_order)
    batch.num_reqs = len(request_order)
    batch.num_reqs_after_padding = len(request_order)
    batch.idx_mapping = state_indices
    batch.idx_mapping_np = state_indices.numpy()
    batch.num_scheduled_tokens = np.asarray(query_lengths, dtype=np.int32)
    batch.num_tokens = sum(query_lengths)
    batch.num_tokens_after_padding = batch.num_tokens
    batch.num_draft_tokens = int(scheduled_lengths.sum())
    batch.num_draft_tokens_per_req = scheduled_lengths
    batch.query_start_loc = query_start_loc
    batch.query_start_loc_np = query_start_loc.numpy()
    batch.input_ids = torch.cat(rows)
    batch.positions = torch.arange(batch.num_tokens, dtype=torch.int64)
    batch.is_padding = torch.zeros(batch.num_tokens, dtype=torch.bool)
    return batch


def _reconcile(
    speculator,
    request_order: tuple[str, ...],
    lengths_by_request: dict[str, int],
    *,
    finished_request_ids: set[str] | None = None,
    preempted_request_ids: set[str] | None = None,
) -> str:
    scheduled_tokens = {request_id: [-1] * lengths_by_request[request_id] for request_id in request_order}
    result = speculator.reconcile_scheduler_proposal(
        scheduled_spec_decode_tokens=scheduled_tokens,
        scheduled_request_ids=set(request_order),
        finished_request_ids=finished_request_ids or set(),
        preempted_request_ids=preempted_request_ids or set(),
        known_request_ids=set(REQUEST_IDS),
    )
    assert result is not None
    return result


def _consume(
    speculator,
    batch,
) -> None:
    scheduled_lengths = torch.from_numpy(
        batch.num_draft_tokens_per_req.copy(),
    )
    speculator._consume_published_proposal_after_verification(
        batch,
        num_sampled=torch.ones(batch.num_reqs, dtype=torch.int32),
        num_rejected=scheduled_lengths,
        temperature=torch.zeros(4, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "request_order",
    [
        ("request-1",),
        REQUEST_IDS,
        VERIFICATION_ORDER,
    ],
)
def test_verification_consumes_request_owned_candidate_rows(
    request_order: tuple[str, ...],
) -> None:
    publication_order = REQUEST_IDS[: len(request_order)]
    if len(request_order) == 1:
        request_order = publication_order
    speculator, proposal_inputs, proposal = _publish_batched_proposal(
        publication_order,
    )
    lengths_by_request = {request_id: row + 1 for row, request_id in enumerate(publication_order)}
    _reconcile(
        speculator,
        request_order,
        lengths_by_request,
    )
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        request_order,
        lengths_by_request,
    )

    _consume(speculator, batch)

    lifecycle = speculator._last_consumed_proposal_lifecycle
    assert lifecycle is not None
    assert lifecycle.request_ids == publication_order
    assert lifecycle.scheduled_lengths == tuple(lengths_by_request[request_id] for request_id in publication_order)
    assert lifecycle.token_prefix_match is True
    assert lifecycle.consumed is True
    assert speculator._proposal_consumption_count == 1


def test_permutation_rejects_candidate_row_from_another_request() -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    lengths_by_request = dict.fromkeys(REQUEST_IDS, 5)
    _reconcile(
        speculator,
        VERIFICATION_ORDER,
        lengths_by_request,
    )
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        VERIFICATION_ORDER,
        lengths_by_request,
    )
    first_candidate_index = int(batch.query_start_loc[0]) + 1
    wrong_row = REQUEST_IDS.index("request-1")
    batch.input_ids[first_candidate_index] = proposal[wrong_row, 0]

    with pytest.raises(ValueError, match="published candidate set prefix"):
        _consume(speculator, batch)

    assert speculator._published_proposal_consumed is False
    assert speculator._proposal_consumption_count == 0


@pytest.mark.parametrize(
    "verification_request_ids",
    [
        ("request-1", "request-2", "request-3"),
        ("request-1", "request-2", "request-3", "request-3"),
        ("request-1", "request-2", "request-3", "request-4", "request-5"),
    ],
)
def test_missing_duplicate_and_extra_verification_ownership_fail_closed(
    verification_request_ids: tuple[str, ...],
) -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        REQUEST_IDS,
        dict.fromkeys(REQUEST_IDS, 5),
    )
    batch.req_ids = list(verification_request_ids)
    batch.num_reqs = len(verification_request_ids)

    with pytest.raises(RuntimeError, match="request ownership"):
        _consume(speculator, batch)

    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_consumed is False
    assert speculator._proposal_consumption_count == 0


def test_permuted_verification_epoch_mismatch_fails_closed() -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    lengths_by_request = dict.fromkeys(REQUEST_IDS, 5)
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        VERIFICATION_ORDER,
        lengths_by_request,
    )
    speculator._proposal_step_epoch += 1

    with pytest.raises(RuntimeError, match="stale proposal epoch"):
        _consume(speculator, batch)

    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_consumed is False


def test_scheduler_extra_and_unexplained_missing_owners_fail_closed() -> None:
    speculator, _proposal_inputs, proposal = _publish_batched_proposal()
    scheduled_tokens = {request_id: [-1] * 5 for request_id in REQUEST_IDS}
    scheduled_tokens["request-extra"] = [-1] * 5

    with pytest.raises(RuntimeError, match="outside published ownership"):
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens=scheduled_tokens,
            scheduled_request_ids=set(scheduled_tokens),
            finished_request_ids=set(),
            preempted_request_ids=set(),
            known_request_ids=set(scheduled_tokens),
        )
    with pytest.raises(RuntimeError, match="only partially present"):
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens={},
            scheduled_request_ids=set(),
            finished_request_ids=set(),
            preempted_request_ids=set(),
            known_request_ids=set(REQUEST_IDS[1:]),
        )

    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_request_ids == REQUEST_IDS
    assert speculator._published_proposal_consumed is False


@pytest.mark.parametrize(
    ("retired_field", "retired_request_id"),
    [
        ("finished_request_ids", "request-2"),
        ("preempted_request_ids", "request-3"),
    ],
)
def test_explicitly_retired_owner_does_not_block_remaining_permutation(
    retired_field: str,
    retired_request_id: str,
) -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    remaining = tuple(request_id for request_id in VERIFICATION_ORDER if request_id != retired_request_id)
    lengths_by_request = dict.fromkeys(remaining, 5)
    reconcile_kwargs = {retired_field: {retired_request_id}}

    assert (
        _reconcile(
            speculator,
            remaining,
            lengths_by_request,
            **reconcile_kwargs,
        )
        == "INSTALLED"
    )
    expected_publication_order = tuple(request_id for request_id in REQUEST_IDS if request_id != retired_request_id)
    expected_rows = torch.tensor(
        [REQUEST_IDS.index(request_id) for request_id in expected_publication_order],
        dtype=torch.int64,
    )
    assert speculator._published_proposal_request_ids == expected_publication_order
    assert torch.equal(
        speculator._published_candidate_tokens,
        proposal.index_select(0, expected_rows),
    )
    batch = _consumer_batch(
        replace(
            proposal_inputs,
            request_ids=expected_publication_order,
            request_state_indices=speculator._published_proposal_request_state_indices,
            num_reqs=len(expected_publication_order),
        ),
        speculator._published_candidate_tokens,
        remaining,
        lengths_by_request,
    )

    _consume(speculator, batch)

    assert speculator._proposal_consumption_count == 1
    assert speculator._last_consumed_proposal_lifecycle.request_ids == (expected_publication_order)
    dropped = speculator._dropped_proposal_lifecycle
    assert dropped is not None
    assert dropped.request_ids == (retired_request_id,)


def test_permuted_proposal_is_consumed_exactly_once() -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    lengths_by_request = dict.fromkeys(REQUEST_IDS, 5)
    _reconcile(
        speculator,
        VERIFICATION_ORDER,
        lengths_by_request,
    )
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        VERIFICATION_ORDER,
        lengths_by_request,
    )
    _consume(speculator, batch)

    with pytest.raises(RuntimeError, match="already consumed"):
        _consume(speculator, batch)

    assert speculator._proposal_consumption_count == 1
