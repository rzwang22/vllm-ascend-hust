# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from tests.ut.spec_decode.test_dspark_v2_proposal_inputs import _step_kwargs
from tests.ut.spec_decode.test_dspark_v2_proposal_publication import (
    _publish_proposal,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _scheduler_tokens(request_ids: tuple[str, ...], lengths: tuple[int, ...]):
    return {request_id: [-1] * length for request_id, length in zip(request_ids, lengths)}


def _reconcile(
    speculator,
    request_ids: tuple[str, ...],
    *,
    lengths: tuple[int, ...] | None,
    scheduled_request_ids: set[str] | None = None,
    finished_request_ids: set[str] | None = None,
    preempted_request_ids: set[str] | None = None,
    known_request_ids: set[str] | None = None,
):
    scheduled_request_ids = set(request_ids) if scheduled_request_ids is None else scheduled_request_ids
    return speculator.reconcile_scheduler_proposal(
        scheduled_spec_decode_tokens=({} if lengths is None else _scheduler_tokens(request_ids, lengths)),
        scheduled_request_ids=scheduled_request_ids,
        finished_request_ids=finished_request_ids or set(),
        preempted_request_ids=preempted_request_ids or set(),
        known_request_ids=(set(request_ids) if known_request_ids is None else known_request_ids),
    )


def _consumer_batch(proposal_inputs, proposal, lengths: tuple[int, ...]):
    batch = _step_kwargs()[0]["input_batch"]
    anchors = torch.tensor([17, 29], dtype=torch.int32)[: len(lengths)]
    rows = [
        torch.cat(
            (
                anchors[index : index + 1],
                proposal[index, :length].to(torch.int32),
            )
        )
        for index, length in enumerate(lengths)
    ]
    query_lengths = tuple(length + 1 for length in lengths)
    query_start_loc = torch.tensor(
        [0, *np.cumsum(query_lengths).tolist()],
        dtype=torch.int32,
    )
    batch.num_scheduled_tokens = np.asarray(query_lengths, dtype=np.int32)
    batch.num_tokens = sum(query_lengths)
    batch.num_tokens_after_padding = batch.num_tokens
    batch.num_draft_tokens = sum(lengths)
    batch.num_draft_tokens_per_req = np.asarray(lengths, dtype=np.int32)
    batch.query_start_loc = query_start_loc
    batch.query_start_loc_np = query_start_loc.numpy()
    batch.input_ids = torch.cat(rows)
    batch.positions = torch.arange(batch.num_tokens, dtype=torch.int64)
    batch.is_padding = torch.zeros(batch.num_tokens, dtype=torch.bool)
    batch.req_ids = list(proposal_inputs.request_ids)
    return batch


@pytest.mark.parametrize("scheduled_length", range(1, 6))
def test_scheduler_prefix_length_is_authoritative_and_consumed_once(
    scheduled_length: int,
) -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )
    lengths = (scheduled_length,) * proposal_inputs.num_reqs
    expected_disposition = "INSTALLED" if scheduled_length == 5 else "TRUNCATED"

    assert _reconcile(speculator, proposal_inputs.request_ids, lengths=lengths) == (expected_disposition)
    lifecycle = speculator._current_proposal_lifecycle
    assert lifecycle is not None
    assert lifecycle.disposition == expected_disposition
    assert lifecycle.scheduled_lengths == lengths
    assert lifecycle.truncated is (scheduled_length < 5)
    assert lifecycle.installed is True
    assert speculator._proposal_installed_count == 1

    consumer = _consumer_batch(proposal_inputs, proposal, lengths)
    speculator._consume_published_proposal_after_verification(
        consumer,
        num_sampled=torch.ones(proposal_inputs.num_reqs, dtype=torch.int32),
        num_rejected=torch.full(
            (proposal_inputs.num_reqs,),
            scheduled_length,
            dtype=torch.int32,
        ),
        temperature=torch.zeros(proposal_inputs.num_reqs, dtype=torch.float32),
    )

    consumed = speculator._last_consumed_proposal_lifecycle
    assert consumed is not None
    assert consumed.scheduled_lengths == lengths
    assert consumed.token_prefix_match is True
    assert consumed.installed is True
    assert consumed.consumed is True
    assert speculator._proposal_installed_count == 1
    assert speculator._proposal_consumption_count == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        speculator._consume_published_proposal_after_verification(
            consumer,
            num_sampled=torch.ones(proposal_inputs.num_reqs, dtype=torch.int32),
            num_rejected=torch.full(
                (proposal_inputs.num_reqs,),
                scheduled_length,
                dtype=torch.int32,
            ),
            temperature=torch.zeros(
                proposal_inputs.num_reqs,
                dtype=torch.float32,
            ),
        )


def test_chunked_prefill_drops_uninstalled_proposal_before_next_target() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )

    assert _reconcile(speculator, proposal_inputs.request_ids, lengths=None) == ("DROPPED")
    dropped = speculator._dropped_proposal_lifecycle
    assert dropped is not None
    assert dropped.drop_reason == "scheduled_without_proposal"
    assert dropped.dropped is True
    assert dropped.installed is False
    assert dropped.consumed is False
    assert speculator._published_candidate_tokens is None
    assert speculator._current_proposal_lifecycle is None
    assert speculator._proposal_dropped_count == 1
    with pytest.raises(RuntimeError, match="no published proposal"):
        speculator._consume_published_proposal_after_verification(
            _consumer_batch(proposal_inputs, proposal, (5, 5)),
            num_sampled=torch.ones(proposal_inputs.num_reqs, dtype=torch.int32),
            num_rejected=torch.full(
                (proposal_inputs.num_reqs,),
                5,
                dtype=torch.int32,
            ),
            temperature=torch.zeros(
                proposal_inputs.num_reqs,
                dtype=torch.float32,
            ),
        )


@pytest.mark.parametrize(
    ("kind", "expected_reason", "terminal"),
    [
        ("terminal", "terminal", True),
        ("preempted", "preempted", False),
        ("missing", "request_missing", False),
    ],
)
def test_terminal_preemption_and_missing_request_retire_ownership(
    kind: str,
    expected_reason: str,
    terminal: bool,
) -> None:
    speculator, proposal_inputs, _result, _proposal = _publish_proposal(
        continue_after_verification=True,
    )
    owners = set(proposal_inputs.request_ids)

    assert (
        _reconcile(
            speculator,
            proposal_inputs.request_ids,
            lengths=None,
            scheduled_request_ids=set(),
            finished_request_ids=owners if kind == "terminal" else set(),
            preempted_request_ids=owners if kind == "preempted" else set(),
            known_request_ids=set() if kind == "missing" else owners,
        )
        == "DROPPED"
    )

    dropped = speculator._dropped_proposal_lifecycle
    assert dropped is not None
    assert dropped.drop_reason == expected_reason
    assert dropped.discarded_terminal is terminal
    assert speculator._published_candidate_tokens is None


def test_delayed_owner_is_not_inferred_as_dropped() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )

    assert (
        _reconcile(
            speculator,
            proposal_inputs.request_ids,
            lengths=None,
            scheduled_request_ids=set(),
        )
        == "DELAYED"
    )
    assert speculator._published_candidate_tokens is proposal
    assert speculator._current_proposal_lifecycle is not None
    assert speculator._proposal_dropped_count == 0


def test_non_prefix_verification_fails_without_consuming() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )
    lengths = (3,) * proposal_inputs.num_reqs
    _reconcile(speculator, proposal_inputs.request_ids, lengths=lengths)
    consumer = _consumer_batch(proposal_inputs, proposal, lengths)
    consumer.input_ids[1] = (consumer.input_ids[1] + 1) % 256

    with pytest.raises(ValueError, match="published candidate set prefix"):
        speculator._consume_published_proposal_after_verification(
            consumer,
            num_sampled=torch.ones(proposal_inputs.num_reqs, dtype=torch.int32),
            num_rejected=torch.full(
                (proposal_inputs.num_reqs,),
                3,
                dtype=torch.int32,
            ),
            temperature=torch.zeros(
                proposal_inputs.num_reqs,
                dtype=torch.float32,
            ),
        )

    assert speculator._published_proposal_consumed is False
    assert speculator._proposal_consumption_count == 0


def test_repeated_disposition_is_idempotent_but_cannot_change_length() -> None:
    speculator, proposal_inputs, _result, _proposal = _publish_proposal(
        continue_after_verification=True,
    )
    lengths = (3,) * proposal_inputs.num_reqs

    assert _reconcile(speculator, proposal_inputs.request_ids, lengths=lengths) == ("TRUNCATED")
    lifecycle = speculator._current_proposal_lifecycle
    assert _reconcile(speculator, proposal_inputs.request_ids, lengths=lengths) == ("TRUNCATED")
    assert speculator._current_proposal_lifecycle is lifecycle
    with pytest.raises(RuntimeError, match="changed an already reconciled"):
        _reconcile(
            speculator,
            proposal_inputs.request_ids,
            lengths=(2,) * proposal_inputs.num_reqs,
        )


def test_partial_owner_batch_fails_closed() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )

    with pytest.raises(RuntimeError, match="partially execute"):
        _reconcile(
            speculator,
            proposal_inputs.request_ids,
            lengths=None,
            scheduled_request_ids={proposal_inputs.request_ids[0]},
        )
    assert speculator._published_candidate_tokens is proposal


def test_ten_dropped_cases_leave_no_active_proposal_state() -> None:
    for _case_index in range(10):
        speculator, proposal_inputs, _result, _proposal = _publish_proposal(
            continue_after_verification=True,
        )

        assert (
            _reconcile(
                speculator,
                proposal_inputs.request_ids,
                lengths=None,
            )
            == "DROPPED"
        )
        assert speculator._published_candidate_tokens is None
        assert speculator._published_proposal_step_epoch is None
        assert speculator._published_proposal_request_ids is None
        assert speculator._published_proposal_request_state_indices is None
        assert speculator._current_proposal_lifecycle is None
        assert speculator._prepared_step_epoch is None
        assert speculator._context_kv_step_epoch is None
        assert speculator._draft_forward_step_epoch is None
        assert speculator._markov_result is None


def test_runner_reconciles_before_base_request_cleanup(monkeypatch) -> None:
    speculator, proposal_inputs, _result, _proposal = _publish_proposal(
        continue_after_verification=True,
    )
    runner = object.__new__(NPUModelRunner)
    runner.speculator = speculator
    runner.req_states = SimpleNamespace(
        req_id_to_index={request_id: index for index, request_id in enumerate(proposal_inputs.request_ids)}
    )
    calls = []
    monkeypatch.setattr(
        GPUModelRunner,
        "finish_requests",
        lambda self, output: calls.append((self, output, speculator._current_proposal_lifecycle)),
    )
    lengths = (2,) * proposal_inputs.num_reqs
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {request_id: 3 for request_id in proposal_inputs.request_ids}
    scheduler_output.scheduled_spec_decode_tokens = _scheduler_tokens(
        proposal_inputs.request_ids,
        lengths,
    )

    runner.finish_requests(scheduler_output)

    assert len(calls) == 1
    reconciled = calls[0][2]
    assert reconciled is not None
    assert reconciled.disposition == "TRUNCATED"
    assert reconciled.scheduled_lengths == lengths
