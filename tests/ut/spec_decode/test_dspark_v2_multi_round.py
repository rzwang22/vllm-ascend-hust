# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput

from tests.ut.spec_decode.test_dspark_v2_proposal_inputs import (
    DRAFT_LAYERS,
    _config,
    _step_kwargs,
)
from tests.ut.spec_decode.test_dspark_v2_proposal_publication import (
    _consumer_batch,
    _publish_proposal,
)
from vllm_ascend.worker.v2.spec_decode.dspark import (
    AscendDSparkProposalInputs,
    create_dspark_speculator,
)


def _verification_kwargs(
    proposal_inputs: AscendDSparkProposalInputs,
    proposal: torch.Tensor,
) -> dict:
    consumer = _consumer_batch(proposal_inputs, proposal)
    kwargs, _auxiliary_states = _step_kwargs()
    num_tokens = consumer.num_tokens_after_padding
    group_zero_slots = torch.arange(num_tokens, dtype=torch.int32)
    group_one_slots = torch.arange(num_tokens, dtype=torch.int32) + num_tokens
    kwargs.update(
        input_batch=consumer,
        slot_mappings={
            DRAFT_LAYERS[0]: group_zero_slots,
            DRAFT_LAYERS[1]: group_one_slots,
            DRAFT_LAYERS[2]: group_one_slots,
        },
        last_hidden_states=torch.ones(
            num_tokens,
            16,
            dtype=torch.bfloat16,
        ),
        aux_hidden_states=[
            torch.full(
                (num_tokens, 8),
                float(layer_id),
                dtype=torch.bfloat16,
            )
            for layer_id in proposal_inputs.target_layer_ids
        ],
        num_sampled=torch.tensor([1, 4], dtype=torch.int32),
        num_rejected=torch.tensor([5, 2], dtype=torch.int32),
    )
    return kwargs


def _install_backbone(monkeypatch, speculator, calls: list) -> None:
    def execute_backbone(proposal_inputs):
        calls.append(proposal_inputs)
        speculator._prepared_step_epoch = None
        speculator._context_kv_step_epoch = proposal_inputs.step_epoch
        speculator._draft_forward_step_epoch = proposal_inputs.step_epoch
        return torch.ones(
            proposal_inputs.num_query_tokens,
            8,
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(
        speculator,
        "_execute_draft_backbone",
        execute_backbone,
    )


def test_production_defaults_to_optimistic_multi_round() -> None:
    config = _config()
    del config.additional_config

    speculator = create_dspark_speculator(config, torch.device("cpu"))

    assert speculator.continue_after_verification is True


def test_multi_round_config_rejects_non_boolean_value() -> None:
    config = _config()
    config.additional_config["dspark_continue_after_verification"] = 1

    with pytest.raises(TypeError, match="must be a bool"):
        create_dspark_speculator(config, torch.device("cpu"))


@pytest.mark.parametrize(
    ("finish_kind", "raw_tokens", "expected_committed"),
    [
        ("eos", [11, 99, 12], [11, 99]),
        ("stop_token", [11, 88, 12], [11, 88]),
        ("max_tokens", [11, 12, 13], [11, 12]),
    ],
)
def test_core_truncation_always_finishes_request(
    finish_kind,
    raw_tokens,
    expected_committed,
) -> None:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    sampling_params = SamplingParams(
        max_tokens=2 if finish_kind == "max_tokens" else 16,
        temperature=0.0,
        stop_token_ids=[88] if finish_kind == "stop_token" else None,
    )
    if finish_kind == "eos":
        sampling_params.update_from_generation_config({}, eos_token_id=99)
    request = Request(
        request_id=f"finish-{finish_kind}",
        prompt_token_ids=[1],
        sampling_params=sampling_params,
        pooling_params=None,
    )
    scheduler = SimpleNamespace(max_model_len=64)

    committed, stopped = Scheduler._update_request_with_output(
        scheduler,
        request,
        raw_tokens.copy(),
    )

    assert stopped is True
    assert request.is_finished()
    assert committed == expected_committed
    assert committed == raw_tokens[: len(committed)]


def test_core_active_request_commits_all_raw_verified_tokens() -> None:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    request = Request(
        request_id="active",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
        pooling_params=None,
    )
    scheduler = SimpleNamespace(max_model_len=64)
    raw_tokens = [11, 12, 13]

    committed, stopped = Scheduler._update_request_with_output(
        scheduler,
        request,
        raw_tokens.copy(),
    )

    assert stopped is False
    assert request.is_finished() is False
    assert committed == raw_tokens


def test_three_optimistic_rounds_advance_owned_epochs_once(monkeypatch) -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )
    backbone_calls: list[AscendDSparkProposalInputs] = []
    _install_backbone(monkeypatch, speculator, backbone_calls)
    proposal_epochs = [proposal_inputs.step_epoch]
    consumed_candidates = []

    for _round_index in range(3):
        previous_proposal = proposal
        previous_epoch = proposal_epochs[-1]
        proposal = speculator.propose(
            **_verification_kwargs(proposal_inputs, previous_proposal),
        )
        assert proposal is not None
        assert speculator._last_consumed_proposal_lifecycle is not None
        consumed = speculator._last_consumed_proposal_lifecycle
        assert consumed.proposal_epoch == previous_epoch
        assert consumed.owner_epoch == previous_epoch
        assert consumed.consumer_epoch == previous_epoch + 1
        assert consumed.generated is True
        assert consumed.returned_to_core is True
        assert consumed.installed is True
        assert consumed.consumed is True
        assert consumed.discarded_terminal is False
        consumed_candidates.append(previous_proposal)

        current = speculator._current_proposal_lifecycle
        assert current is not None
        assert current.proposal_epoch == previous_epoch + 1
        assert current.owner_epoch == current.proposal_epoch
        assert current.consumer_epoch is None
        assert current.installed is False
        assert current.consumed is False
        assert current.discarded_terminal is False
        proposal_inputs = backbone_calls[-1]
        proposal_epochs.append(current.proposal_epoch)

    assert proposal_epochs == [1, 2, 3, 4]
    assert len({id(candidate) for candidate in consumed_candidates}) == 3
    assert len(backbone_calls) == 3
    assert speculator._proposal_generated_count == 4
    assert speculator._proposal_returned_count == 4
    assert speculator._proposal_installed_count == 3
    assert speculator._proposal_consumption_count == 3
    assert speculator._terminal_proposal_discard_count == 0
    assert speculator._next_proposal_skip_count == 0


@pytest.mark.parametrize("stop_reason", ["eos", "max_tokens", "stop_token"])
def test_terminal_scheduler_outcome_discards_one_uninstalled_proposal(
    stop_reason,
) -> None:
    del stop_reason
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )
    consumer = _consumer_batch(proposal_inputs, proposal)
    lifecycle = speculator._current_proposal_lifecycle

    assert speculator.discard_terminal_proposal(set(proposal_inputs.request_ids))
    terminal = speculator._terminal_proposal_lifecycle
    assert terminal is not None
    assert terminal is not lifecycle
    assert terminal.generated is True
    assert terminal.returned_to_core is True
    assert terminal.installed is False
    assert terminal.consumed is False
    assert terminal.discarded_terminal is True
    assert speculator._published_candidate_tokens is None
    assert speculator._current_proposal_lifecycle is None
    assert speculator._terminal_proposal_discard_count == 1
    assert (
        speculator.discard_terminal_proposal(
            set(proposal_inputs.request_ids),
        )
        is False
    )
    assert speculator._terminal_proposal_discard_count == 1
    with pytest.raises(RuntimeError, match="no published proposal"):
        speculator._consume_published_proposal_after_verification(
            consumer,
            num_sampled=torch.tensor([1, 1], dtype=torch.int32),
            num_rejected=torch.tensor([5, 5], dtype=torch.int32),
            temperature=torch.zeros(2, dtype=torch.float32),
        )
    assert proposal.shape == (proposal_inputs.num_reqs, 5)


def test_partial_terminal_batch_retires_only_finished_owner() -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )
    lifecycle = speculator._current_proposal_lifecycle

    assert speculator.discard_terminal_proposal(
        {proposal_inputs.request_ids[0]},
    )

    assert speculator._current_proposal_lifecycle is not lifecycle
    assert speculator._published_proposal_request_ids == (proposal_inputs.request_ids[1],)
    assert torch.equal(speculator._published_candidate_tokens, proposal[1:])
    assert torch.equal(
        speculator._published_proposal_request_state_indices,
        proposal_inputs.request_state_indices[1:],
    )
    assert speculator._terminal_proposal_lifecycle is not None
    assert speculator._terminal_proposal_lifecycle.request_ids == (proposal_inputs.request_ids[0],)
    assert speculator._terminal_proposal_discard_count == 1


def test_next_round_failure_does_not_republish_consumed_proposal(
    monkeypatch,
) -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal(
        continue_after_verification=True,
    )
    kwargs = _verification_kwargs(proposal_inputs, proposal)
    kwargs["aux_hidden_states"] = None
    monkeypatch.setattr(
        speculator,
        "_execute_draft_backbone",
        pytest.fail,
    )

    with pytest.raises(RuntimeError, match="auxiliary hidden states are missing"):
        speculator.propose(**kwargs)

    assert speculator._last_consumed_proposal_lifecycle is not None
    assert speculator._last_consumed_proposal_lifecycle.consumed is True
    assert speculator._published_candidate_tokens is None
    assert speculator._current_proposal_lifecycle is None
    assert speculator._proposal_consumption_count == 1
    assert speculator._proposal_publication_count == 1


def test_m2_4a_mode_still_skips_after_first_verification(monkeypatch) -> None:
    speculator, proposal_inputs, _result, proposal = _publish_proposal()
    monkeypatch.setattr(
        speculator,
        "_execute_draft_backbone",
        pytest.fail,
    )

    assert (
        speculator.propose(
            **_verification_kwargs(proposal_inputs, proposal),
        )
        is None
    )
    assert speculator.continue_after_verification is False
    assert speculator._proposal_generated_count == 1
    assert speculator._proposal_installed_count == 1
    assert speculator._proposal_consumption_count == 1
    assert speculator._next_proposal_skip_count == 1


def test_non_dspark_finish_path_keeps_base_runner_behavior(monkeypatch) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    runner = object.__new__(NPUModelRunner)
    runner.speculator = SimpleNamespace()
    calls = []
    monkeypatch.setattr(
        GPUModelRunner,
        "finish_requests",
        lambda self, output: calls.append((self, output)),
    )
    scheduler_output = SimpleNamespace(finished_req_ids={"finished"})

    runner.finish_requests(scheduler_output)

    assert calls == [(runner, scheduler_output)]


def test_runner_finish_discards_terminal_proposal_before_base_cleanup(
    monkeypatch,
) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    speculator, proposal_inputs, _result, _proposal = _publish_proposal(
        continue_after_verification=True,
    )
    runner = object.__new__(NPUModelRunner)
    runner.speculator = speculator
    runner.req_states = SimpleNamespace(
        req_id_to_index={request_id: index for index, request_id in enumerate(proposal_inputs.request_ids)}
    )
    calls = []

    def finish_base(self, output):
        calls.append(
            (
                self,
                output,
                speculator._current_proposal_lifecycle,
                speculator._terminal_proposal_lifecycle,
            )
        )

    monkeypatch.setattr(GPUModelRunner, "finish_requests", finish_base)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.finished_req_ids = set(proposal_inputs.request_ids)

    runner.finish_requests(scheduler_output)

    assert len(calls) == 1
    assert calls[0][:2] == (runner, scheduler_output)
    assert calls[0][2] is None
    assert calls[0][3] is speculator._terminal_proposal_lifecycle
    assert speculator._terminal_proposal_lifecycle is not None
    assert speculator._terminal_proposal_discard_count == 1


def test_runner_finish_keeps_base_cleanup_on_partial_terminal_discard(
    monkeypatch,
) -> None:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

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
        lambda self, output: calls.append((self, output)),
    )
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.finished_req_ids = {proposal_inputs.request_ids[0]}

    runner.finish_requests(scheduler_output)

    assert calls == [(runner, scheduler_output)]
    assert speculator._current_proposal_lifecycle is not None
    assert speculator._published_proposal_request_ids == (proposal_inputs.request_ids[1],)
    assert speculator._terminal_proposal_lifecycle is not None
    assert speculator._terminal_proposal_lifecycle.request_ids == (proposal_inputs.request_ids[0],)
