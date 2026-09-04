# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from tests.ut.spec_decode.test_dspark_v2_markov_sampling import (
    _ready_markov_step,
)
from tests.ut.spec_decode.test_dspark_v2_proposal_inputs import _step_kwargs
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

REQUEST_IDS = ("request-1", "request-2", "request-3", "request-4")
REQUEST_STATE_INDICES = torch.tensor([3, 0, 2, 1], dtype=torch.int32)
VERIFICATION_ORDER = ("request-4", "request-3", "request-1", "request-2")
NEW_PREFILL_REQUEST_ID = "request-5-prefill"
SYNTHETIC_TOKEN_FLOOR = 10


def _bounded_candidate_tokens(
    num_reqs: int,
    num_speculative_tokens: int,
    shared_vocab_size: int,
) -> torch.Tensor:
    usable_vocab_size = shared_vocab_size - SYNTHETIC_TOKEN_FLOOR
    assert usable_vocab_size > 0
    candidates = (
        torch.arange(num_reqs, dtype=torch.int64)[:, None] * num_speculative_tokens
        + torch.arange(
            1,
            num_speculative_tokens + 1,
            dtype=torch.int64,
        )[None, :]
    ) % usable_vocab_size + SYNTHETIC_TOKEN_FLOOR
    assert candidates.shape == (num_reqs, num_speculative_tokens)
    assert bool((candidates >= 0).all())
    assert bool((candidates < shared_vocab_size).all())
    assert torch.unique(candidates, dim=0).shape[0] == num_reqs
    assert torch.unique(candidates[:, 0]).shape[0] == num_reqs
    return candidates


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
    request_state_indices = (
        REQUEST_STATE_INDICES[:num_reqs].clone()
        if num_reqs <= len(REQUEST_STATE_INDICES)
        else torch.arange(num_reqs, dtype=torch.int32)
    )
    candidates = _bounded_candidate_tokens(
        num_reqs,
        result.num_speculative_tokens,
        result.vocab_size,
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


@pytest.mark.parametrize("num_reqs", [55, 64])
def test_large_batch_proposal_fixture_stays_within_shared_vocabulary(
    num_reqs: int,
) -> None:
    num_speculative_tokens = 5
    shared_vocab_size = 256

    candidates = _bounded_candidate_tokens(
        num_reqs,
        num_speculative_tokens,
        shared_vocab_size,
    )

    assert candidates.shape == (num_reqs, num_speculative_tokens)
    assert int(candidates.min()) >= 0
    assert int(candidates.max()) < shared_vocab_size
    assert torch.unique(candidates, dim=0).shape[0] == num_reqs
    assert torch.unique(candidates[:, 0]).shape[0] == num_reqs


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
    *,
    prefill_lengths_by_request: dict[str, int] | None = None,
):
    batch = _step_kwargs()[0]["input_batch"]
    candidates_by_request = _candidate_rows_by_request(
        proposal_inputs,
        proposal,
    )
    state_indices_by_request = _state_index_by_request(proposal_inputs)
    prefill_lengths_by_request = prefill_lengths_by_request or {}
    next_state_index = max(state_indices_by_request.values(), default=-1) + 1
    rows = []
    query_lengths = []
    scheduled_lengths = []
    for batch_row, request_id in enumerate(request_order):
        if request_id in candidates_by_request:
            scheduled_length = lengths_by_request[request_id]
            anchor = torch.tensor([100 + batch_row], dtype=torch.int32)
            rows.append(
                torch.cat(
                    (
                        anchor,
                        candidates_by_request[request_id][:scheduled_length].to(torch.int32),
                    )
                )
            )
            query_lengths.append(scheduled_length + 1)
            scheduled_lengths.append(scheduled_length)
        else:
            prefill_length = prefill_lengths_by_request[request_id]
            rows.append(
                torch.arange(
                    1000 + batch_row * prefill_length,
                    1000 + (batch_row + 1) * prefill_length,
                    dtype=torch.int32,
                )
            )
            query_lengths.append(prefill_length)
            scheduled_lengths.append(0)
            state_indices_by_request[request_id] = next_state_index
            next_state_index += 1

    query_start_loc = torch.tensor(
        [0, *np.cumsum(query_lengths).tolist()],
        dtype=torch.int32,
    )
    state_indices = torch.tensor(
        [state_indices_by_request[request_id] for request_id in request_order],
        dtype=torch.int32,
    )
    scheduled_lengths = np.asarray(scheduled_lengths, dtype=np.int32)
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
    batch.is_prefilling_np = np.asarray(
        [request_id in prefill_lengths_by_request for request_id in request_order],
        dtype=np.bool_,
    )
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
        temperature=torch.zeros(
            int(batch.idx_mapping_np.max()) + 1,
            dtype=torch.float32,
        ),
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


def test_mixed_prefill_is_not_treated_as_installed_proposal_owner() -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    lengths_by_request = dict.fromkeys(REQUEST_IDS, 5)
    mixed_request_order = (*VERIFICATION_ORDER, NEW_PREFILL_REQUEST_ID)
    scheduled_tokens = {request_id: [-1] * lengths_by_request[request_id] for request_id in VERIFICATION_ORDER}

    assert (
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens=scheduled_tokens,
            scheduled_request_ids=set(mixed_request_order),
            finished_request_ids=set(),
            preempted_request_ids=set(),
            known_request_ids=set(mixed_request_order),
        )
        == "INSTALLED"
    )
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        mixed_request_order,
        lengths_by_request,
        prefill_lengths_by_request={NEW_PREFILL_REQUEST_ID: 61},
    )
    new_request_row = mixed_request_order.index(NEW_PREFILL_REQUEST_ID)
    new_request_start = int(batch.query_start_loc[new_request_row])
    new_request_tokens = batch.input_ids[new_request_start : int(batch.query_start_loc[new_request_row + 1])].clone()

    _consume(speculator, batch)

    lifecycle = speculator._last_consumed_proposal_lifecycle
    assert lifecycle is not None
    assert lifecycle.request_ids == REQUEST_IDS
    assert lifecycle.scheduled_lengths == (5, 5, 5, 5)
    assert lifecycle.consumed is True
    assert NEW_PREFILL_REQUEST_ID not in lifecycle.request_ids
    assert torch.equal(
        batch.input_ids[new_request_start : int(batch.query_start_loc[new_request_row + 1])],
        new_request_tokens,
    )
    assert speculator._proposal_consumption_count == 1


def test_mixed_terminal_verification_permutation_and_new_prefill() -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    terminal_request_id = "request-2"
    verification_order = tuple(request_id for request_id in VERIFICATION_ORDER if request_id != terminal_request_id)
    mixed_request_order = (*verification_order, NEW_PREFILL_REQUEST_ID)
    lengths_by_request = dict.fromkeys(verification_order, 5)

    assert (
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens=dict.fromkeys(
                verification_order,
                [-1] * 5,
            ),
            scheduled_request_ids=set(mixed_request_order),
            finished_request_ids={terminal_request_id},
            preempted_request_ids=set(),
            known_request_ids={*REQUEST_IDS, NEW_PREFILL_REQUEST_ID},
        )
        == "INSTALLED"
    )
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        mixed_request_order,
        lengths_by_request,
        prefill_lengths_by_request={NEW_PREFILL_REQUEST_ID: 61},
    )

    _consume(speculator, batch)

    consumed = speculator._last_consumed_proposal_lifecycle
    assert consumed is not None
    assert consumed.request_ids == tuple(request_id for request_id in REQUEST_IDS if request_id != terminal_request_id)
    assert terminal_request_id not in consumed.request_ids
    assert NEW_PREFILL_REQUEST_ID not in consumed.request_ids
    assert speculator._terminal_proposal_lifecycle is not None
    assert speculator._terminal_proposal_lifecycle.request_ids == (terminal_request_id,)
    assert speculator._proposal_consumption_count == 1


def test_runner_reconciles_mixed_admission_from_spec_token_keys_only(
    monkeypatch,
) -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    runner = object.__new__(NPUModelRunner)
    runner.speculator = speculator
    runner.req_states = SimpleNamespace(
        req_id_to_index={request_id: row for row, request_id in enumerate(proposal_inputs.request_ids)}
    )
    base_calls = []
    monkeypatch.setattr(
        GPUModelRunner,
        "finish_requests",
        lambda self, scheduler_output: base_calls.append((self, scheduler_output)),
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_new_reqs = [SimpleNamespace(req_id=NEW_PREFILL_REQUEST_ID)]
    scheduler_output.num_scheduled_tokens = {
        **dict.fromkeys(REQUEST_IDS, 6),
        NEW_PREFILL_REQUEST_ID: 61,
    }
    scheduler_output.scheduled_spec_decode_tokens = dict.fromkeys(
        REQUEST_IDS,
        [-1] * 5,
    )

    runner.finish_requests(scheduler_output)

    assert base_calls == [(runner, scheduler_output)]
    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_request_ids == REQUEST_IDS
    lifecycle = speculator._current_proposal_lifecycle
    assert lifecycle is not None
    assert lifecycle.request_ids == REQUEST_IDS
    assert lifecycle.scheduled_lengths == (5, 5, 5, 5)
    assert lifecycle.installed is True
    assert NEW_PREFILL_REQUEST_ID not in lifecycle.request_ids


def test_terminal_subset_of_fifty_five_owner_publication_is_reconciled() -> None:
    published_request_ids = tuple(f"request-{index}" for index in range(1, 56))
    terminal_request_ids = {"request-23", "request-42"}
    scheduled_request_ids = tuple(
        request_id for request_id in reversed(published_request_ids) if request_id not in terminal_request_ids
    )
    speculator, proposal_inputs, proposal = _publish_batched_proposal(
        published_request_ids,
    )
    lengths_by_request = dict.fromkeys(scheduled_request_ids, 5)

    assert (
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens=dict.fromkeys(
                scheduled_request_ids,
                [-1] * 5,
            ),
            scheduled_request_ids=set(scheduled_request_ids),
            finished_request_ids=terminal_request_ids,
            preempted_request_ids=set(),
            known_request_ids=set(published_request_ids),
        )
        == "INSTALLED"
    )

    expected_installed_ids = tuple(
        request_id for request_id in published_request_ids if request_id not in terminal_request_ids
    )
    assert speculator._published_proposal_request_ids == expected_installed_ids
    assert speculator._deferred_published_proposal is None
    assert speculator._terminal_proposal_lifecycle is not None
    assert speculator._terminal_proposal_lifecycle.request_ids == (
        "request-23",
        "request-42",
    )
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        scheduled_request_ids,
        lengths_by_request,
    )

    _consume(speculator, batch)

    consumed = speculator._last_consumed_proposal_lifecycle
    assert consumed is not None
    assert consumed.request_ids == expected_installed_ids
    assert consumed.consumed is True
    assert speculator._proposal_consumption_count == 1
    assert speculator._markov_result is None


def test_terminal_scheduled_and_delayed_subsets_preserve_row_ownership() -> None:
    published_request_ids = tuple(f"request-{index}" for index in range(1, 65))
    terminal_request_ids = {"request-23", "request-42"}
    delayed_request_ids = {
        "request-7",
        "request-8",
        "request-16",
        "request-19",
        "request-36",
        "request-39",
        "request-48",
        "request-55",
        "request-64",
    }
    scheduled_request_ids = tuple(
        request_id
        for request_id in reversed(published_request_ids)
        if request_id not in terminal_request_ids | delayed_request_ids
    )
    speculator, proposal_inputs, proposal = _publish_batched_proposal(
        published_request_ids,
    )
    lengths_by_request = dict.fromkeys(scheduled_request_ids, 5)

    assert (
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens=dict.fromkeys(
                scheduled_request_ids,
                [-1] * 5,
            ),
            scheduled_request_ids=set(scheduled_request_ids),
            finished_request_ids=terminal_request_ids,
            preempted_request_ids=set(),
            known_request_ids=set(published_request_ids),
        )
        == "INSTALLED"
    )
    deferred = speculator._deferred_published_proposal
    assert deferred is not None
    assert deferred.request_ids == tuple(
        request_id for request_id in published_request_ids if request_id in delayed_request_ids
    )
    expected_deferred_rows = torch.tensor(
        [published_request_ids.index(request_id) for request_id in deferred.request_ids],
        dtype=torch.int64,
    )
    assert torch.equal(
        deferred.candidate_tokens,
        proposal.index_select(0, expected_deferred_rows),
    )
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        scheduled_request_ids,
        lengths_by_request,
    )

    result = speculator.propose(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.empty(0),
        aux_hidden_states=None,
        num_sampled=torch.ones(batch.num_reqs, dtype=torch.int32),
        num_rejected=torch.full(
            (batch.num_reqs,),
            5,
            dtype=torch.int32,
        ),
        last_sampled=torch.empty(0),
        next_prefill_tokens=torch.empty(0),
        temperature=torch.zeros(len(published_request_ids), dtype=torch.float32),
        seeds=torch.empty(0),
    )

    assert result is None
    assert speculator._proposal_consumption_count == 1
    assert speculator._published_proposal_request_ids == deferred.request_ids
    assert torch.equal(
        speculator._published_candidate_tokens,
        deferred.candidate_tokens,
    )
    assert speculator._deferred_published_proposal is None

    delayed_order = tuple(reversed(deferred.request_ids))
    delayed_lengths = dict.fromkeys(delayed_order, 5)
    assert (
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens=dict.fromkeys(
                delayed_order,
                [-1] * 5,
            ),
            scheduled_request_ids=set(delayed_order),
            finished_request_ids=set(),
            preempted_request_ids=set(),
            known_request_ids=set(delayed_request_ids),
        )
        == "INSTALLED"
    )
    delayed_batch = _consumer_batch(
        proposal_inputs,
        proposal,
        delayed_order,
        delayed_lengths,
    )

    _consume(speculator, delayed_batch)

    assert speculator._last_consumed_proposal_lifecycle is not None
    assert speculator._last_consumed_proposal_lifecycle.request_ids == deferred.request_ids
    assert speculator._proposal_generated_count == 1
    assert speculator._proposal_installed_count == 2
    assert speculator._proposal_consumption_count == 2
    speculator._release_consumed_proposal()
    assert speculator._published_candidate_tokens is None
    assert speculator._published_proposal_request_ids is None
    assert speculator._current_proposal_lifecycle is None
    assert speculator._markov_result is None


def test_terminal_active_subset_restores_existing_delayed_owners() -> None:
    speculator, _proposal_inputs, _proposal = _publish_batched_proposal()
    installed_request_id = REQUEST_IDS[0]
    delayed_request_ids = set(REQUEST_IDS[1:])

    assert (
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens={installed_request_id: [-1] * 5},
            scheduled_request_ids={installed_request_id},
            finished_request_ids=set(),
            preempted_request_ids=set(),
            known_request_ids=set(REQUEST_IDS),
        )
        == "INSTALLED"
    )
    assert speculator._deferred_published_proposal is not None

    assert speculator.discard_terminal_proposal({installed_request_id})

    assert speculator._deferred_published_proposal is None
    assert speculator._published_proposal_request_ids == REQUEST_IDS[1:]
    assert set(speculator._published_proposal_request_ids) == delayed_request_ids
    assert speculator._current_proposal_lifecycle is not None
    assert speculator._current_proposal_lifecycle.disposition == "GENERATED"


def test_mixed_prefill_can_join_only_the_next_proposal_epoch(
    monkeypatch,
) -> None:
    speculator, proposal_inputs, proposal = _publish_batched_proposal()
    lengths_by_request = dict.fromkeys(REQUEST_IDS, 5)
    mixed_request_order = (*VERIFICATION_ORDER, NEW_PREFILL_REQUEST_ID)
    batch = _consumer_batch(
        proposal_inputs,
        proposal,
        mixed_request_order,
        lengths_by_request,
        prefill_lengths_by_request={NEW_PREFILL_REQUEST_ID: 61},
    )
    speculator.reconcile_scheduler_proposal(
        scheduled_spec_decode_tokens={
            request_id: [-1] * lengths_by_request[request_id] for request_id in VERIFICATION_ORDER
        },
        scheduled_request_ids=set(mixed_request_order),
        finished_request_ids=set(),
        preempted_request_ids=set(),
        known_request_ids=set(mixed_request_order),
    )
    prepared_inputs = object()
    next_candidates = torch.full(
        (len(mixed_request_order), 5),
        77,
        dtype=torch.int64,
    )
    calls = []

    def prepare_next(**kwargs):
        assert kwargs["input_batch"] is batch
        assert speculator._published_candidate_tokens is None
        calls.append(("prepare", tuple(kwargs["input_batch"].req_ids)))
        return prepared_inputs

    def execute_next(inputs):
        assert inputs is prepared_inputs
        calls.append(("execute", inputs))
        return next_candidates

    monkeypatch.setattr(speculator, "prepare_proposal_inputs", prepare_next)
    monkeypatch.setattr(speculator, "_execute_draft", execute_next)

    result = speculator.propose(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.empty(0),
        aux_hidden_states=None,
        num_sampled=torch.ones(batch.num_reqs, dtype=torch.int32),
        num_rejected=torch.from_numpy(batch.num_draft_tokens_per_req.copy()),
        last_sampled=torch.empty(0),
        next_prefill_tokens=torch.empty(0),
        temperature=torch.zeros(5, dtype=torch.float32),
        seeds=torch.empty(0),
    )

    assert result is next_candidates
    assert calls == [
        ("prepare", mixed_request_order),
        ("execute", prepared_inputs),
    ]
    assert speculator._last_consumed_proposal_lifecycle is not None
    assert speculator._last_consumed_proposal_lifecycle.request_ids == REQUEST_IDS
    assert speculator._last_consumed_proposal_lifecycle.consumed is True
    assert speculator._proposal_consumption_count == 1


def test_unpublished_prefill_with_scheduled_proposal_tokens_fails_closed() -> None:
    speculator, _proposal_inputs, proposal = _publish_batched_proposal()
    scheduled_tokens = {request_id: [-1] * 5 for request_id in REQUEST_IDS}
    scheduled_tokens[NEW_PREFILL_REQUEST_ID] = [-1] * 5

    with pytest.raises(RuntimeError, match="outside published ownership"):
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens=scheduled_tokens,
            scheduled_request_ids=set(scheduled_tokens),
            finished_request_ids=set(),
            preempted_request_ids=set(),
            known_request_ids=set(scheduled_tokens),
        )

    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_consumed is False


@pytest.mark.parametrize("retired_field", ["finished_request_ids", "preempted_request_ids"])
def test_scheduled_and_retired_owner_conflict_fails_closed(
    retired_field: str,
) -> None:
    speculator, _proposal_inputs, proposal = _publish_batched_proposal()
    conflicted_request_id = REQUEST_IDS[0]
    kwargs = {
        "finished_request_ids": set(),
        "preempted_request_ids": set(),
    }
    kwargs[retired_field] = {conflicted_request_id}

    with pytest.raises(RuntimeError, match="conflicting proposal owner dispositions"):
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens={conflicted_request_id: [-1] * 5},
            scheduled_request_ids={conflicted_request_id},
            known_request_ids=set(REQUEST_IDS),
            **kwargs,
        )

    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_request_ids == REQUEST_IDS
    assert speculator._proposal_consumption_count == 0


def test_spec_tokens_for_unscheduled_owner_fail_closed() -> None:
    speculator, _proposal_inputs, proposal = _publish_batched_proposal()
    request_id = REQUEST_IDS[0]

    with pytest.raises(RuntimeError, match="conflicting proposal owner dispositions"):
        speculator.reconcile_scheduler_proposal(
            scheduled_spec_decode_tokens={request_id: [-1] * 5},
            scheduled_request_ids=set(),
            finished_request_ids=set(),
            preempted_request_ids=set(),
            known_request_ids=set(REQUEST_IDS),
        )

    assert speculator._published_candidate_tokens is proposal
    assert speculator._published_proposal_request_ids == REQUEST_IDS
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
    with pytest.raises(RuntimeError, match="no scheduled, finished, preempted, or delayed disposition"):
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
