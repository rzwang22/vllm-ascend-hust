# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from tests.ut.spec_decode.test_dspark_v2_proposal_inputs import (
    _ready_speculator,
    _step_kwargs,
)
from vllm_ascend.spec_decode import DSparkRuntimeNotWiredError
from vllm_ascend.worker.v2.spec_decode.dspark import AscendDSparkMarkovResult


class _LoadedHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _LoadedMarkovHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.markov_w1 = _LoadedHead()
        self.markov_w2 = _LoadedHead()


class _MarkovBackbone(nn.Module):
    def __init__(self, *, confidence: bool = True) -> None:
        super().__init__()
        self.num_dspark_layers = 3
        self.markov_head = _LoadedMarkovHead()
        self.confidence_head = _LoadedHead() if confidence else None


class _MarkovDraftModel(nn.Module):
    def __init__(self, *, vocab_size: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.lm_head = _LoadedHead()
        self.model = _MarkovBackbone()
        self.markov_inputs: list[torch.Tensor] = []
        self.bias_calls = 0
        self.fail_bias_step: int | None = None
        self.return_local_vocab = False
        self.base_logits_nan = False
        self.step_logits_all_negative_inf = False
        self.confidence_calls = 0

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        vocab_size = self.vocab_size // 2 if self.return_local_vocab else self.vocab_size
        logits = torch.zeros(
            hidden_states.shape[0],
            vocab_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        if self.base_logits_nan:
            logits[0, 0] = float("nan")
        if self.step_logits_all_negative_inf:
            logits.fill_(float("-inf"))
        return logits

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.markov_inputs.append(token_ids)
        return token_ids.to(torch.float32).unsqueeze(-1)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        if self.fail_bias_step == self.bias_calls:
            raise RuntimeError("markov-step-failure")
        self.bias_calls += 1
        predecessor = markov_embed[:, 0].to(torch.int64)
        selected = (predecessor + 1) % self.vocab_size
        bias = torch.zeros(
            predecessor.shape[0],
            self.vocab_size,
            dtype=torch.float32,
            device=markov_embed.device,
        )
        if self.step_logits_all_negative_inf:
            return bias.fill_(float("-inf"))
        return bias.scatter(1, selected[:, None], 10.0)

    @staticmethod
    def map_draft_to_target(draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def confidence_logits(
        self,
        hidden_states: torch.Tensor,
        markov_embed: torch.Tensor,
    ) -> torch.Tensor:
        self.confidence_calls += 1
        return hidden_states[:, :1] + markov_embed[:, :1]


def _ready_markov_step(
    *,
    seeds: torch.Tensor | None = None,
    continue_after_verification: bool = False,
) -> tuple[object, object, _MarkovDraftModel, torch.Tensor]:
    speculator = _ready_speculator(
        continue_after_verification=continue_after_verification,
    )
    speculator.draft_model_config.hf_config.vocab_size = 256
    model = _MarkovDraftModel()
    speculator._model = model
    speculator._markov_module_contract = None
    kwargs, _auxiliary_states = _step_kwargs()
    if seeds is not None:
        kwargs["seeds"] = seeds
    proposal = speculator.prepare_proposal_inputs(**kwargs)
    speculator._prepared_step_epoch = None
    speculator._context_kv_step_epoch = proposal.step_epoch
    speculator._draft_forward_step_epoch = proposal.step_epoch
    hidden_states = (
        torch.arange(
            proposal.num_query_tokens * 8,
            dtype=torch.float32,
        )
        .to(torch.bfloat16)
        .view(proposal.num_query_tokens, 8)
    )
    return speculator, proposal, model, hidden_states


def test_fixed_k_recurrence_uses_anchor_then_previous_selected_identity() -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()

    result = speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert type(result) is AscendDSparkMarkovResult
    assert result.step_epoch == proposal.step_epoch
    assert result.backbone_hidden_states is hidden_states
    assert torch.equal(
        result.request_state_indices,
        proposal.request_state_indices[: proposal.num_reqs],
    )
    assert result.request_state_indices.data_ptr() != proposal.request_state_indices.data_ptr()
    assert result.candidate_tokens.shape == (2, 5)
    assert torch.equal(
        result.candidate_tokens,
        torch.tensor(
            [
                [203, 204, 205, 206, 207],
                [102, 103, 104, 105, 106],
            ]
        ),
    )
    assert len(result.steps) == 5
    assert len(model.markov_inputs) == 5
    assert result.steps[0].predecessor_source == "anchor_token_ids"
    assert result.steps[0].predecessor_token_ids is proposal.anchor_token_ids
    assert result.steps[0].markov_input_token_ids is proposal.anchor_token_ids
    for step_index in range(1, 5):
        previous = result.steps[step_index - 1].selected_token_ids
        current = result.steps[step_index]
        assert current.predecessor_source == "previous_sampled_token"
        assert current.predecessor_token_ids is previous
        assert current.markov_input_token_ids is previous
        assert model.markov_inputs[step_index] is previous
        assert current.predecessor_token_ids is not proposal.anchor_token_ids


def test_base_logits_restore_request_major_b_k_vocab_layout() -> None:
    speculator, proposal, _model, hidden_states = _ready_markov_step()

    result = speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert result.physical_hidden_shape == (10, 8)
    assert result.physical_base_logits_shape == (10, 256)
    assert result.logical_base_logits_shape == (2, 5, 256)
    assert result.logical_candidate_shape == (2, 5)
    assert result.vocab_size == 256


def test_loaded_modules_are_used_and_confidence_is_not_consumed() -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()

    result = speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert result.loaded_module_identity_preserved is True
    assert result.lm_head_class.endswith("._LoadedHead")
    assert result.markov_head_class.endswith("._LoadedMarkovHead")
    assert result.lm_head_parameter_names == ("mtp.2.head.weight",)
    assert result.markov_parameter_names == (
        "mtp.2.markov_head.markov_w1.weight",
        "mtp.2.markov_head.markov_w2.weight",
    )
    assert result.confidence_head_present is True
    assert result.confidence_head_used is False
    assert model.confidence_calls == 0


def test_seeds_do_not_change_greedy_candidates() -> None:
    first = _ready_markov_step(seeds=torch.tensor([7, 9], dtype=torch.int64))
    second = _ready_markov_step(seeds=torch.tensor([900, -41], dtype=torch.int64))

    first_result = first[0]._execute_sequential_markov_sampling(first[1], first[3])
    second_result = second[0]._execute_sequential_markov_sampling(second[1], second[3])

    assert torch.equal(first_result.candidate_tokens, second_result.candidate_tokens)


def test_stochastic_temperature_fails_closed_before_head_execution() -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()
    proposal = replace(
        proposal,
        temperature=torch.tensor([0.0, 0.7], dtype=torch.float32),
    )

    with pytest.raises(
        DSparkRuntimeNotWiredError,
        match="V2 DSpark stochastic Markov sampling",
    ):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert model.markov_inputs == []
    assert speculator._markov_result is None
    assert speculator._markov_step_epoch is None


def test_local_vocab_logits_fail_before_argmax() -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()
    model.return_local_vocab = True

    with pytest.raises(RuntimeError, match="local vocabulary shard"):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert model.markov_inputs == []
    assert speculator._markov_result is None


def test_anchor_outside_shared_vocabulary_fails_before_markov_embedding() -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()
    proposal = replace(
        proposal,
        anchor_token_ids=torch.tensor([256, 1], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="anchor token.*outside"):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert model.markov_inputs == []
    assert speculator._markov_result is None


@pytest.mark.parametrize(
    ("attribute", "match"),
    [
        ("base_logits_nan", "base logits contain NaN"),
        ("step_logits_all_negative_inf", "no selectable finite token"),
    ],
)
def test_invalid_logits_fail_before_candidate_publication(attribute, match) -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()
    setattr(model, attribute, True)

    with pytest.raises(ValueError, match=match):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert speculator._markov_result is None
    assert speculator._markov_step_epoch is None


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (
            lambda proposal: replace(
                proposal,
                draft_query_start_loc=torch.tensor([0, 4, 10], dtype=torch.int32),
            ),
            ValueError,
            "request-major",
        ),
        (
            lambda proposal: replace(proposal, rank=proposal.rank + 1),
            RuntimeError,
            "different NPU rank",
        ),
        (
            lambda proposal: replace(
                proposal,
                request_ids=(proposal.request_ids[0], proposal.request_ids[0]),
            ),
            RuntimeError,
            "request ownership",
        ),
    ],
)
def test_markov_ownership_and_query_layout_fail_closed(mutation, error, match) -> None:
    speculator, proposal, _model, hidden_states = _ready_markov_step()

    with pytest.raises(error, match=match):
        speculator._execute_sequential_markov_sampling(mutation(proposal), hidden_states)


@pytest.mark.parametrize(
    ("hidden_states", "error", "match"),
    [
        (torch.ones(9, 8, dtype=torch.bfloat16), ValueError, "request-major"),
        (torch.ones(10, 7, dtype=torch.bfloat16), ValueError, "request-major"),
        (torch.ones(10, 8, dtype=torch.int32), TypeError, "floating point"),
        (torch.ones(10, 8, device="meta"), RuntimeError, "current rank device"),
    ],
)
def test_invalid_hidden_contract_fails_closed(hidden_states, error, match) -> None:
    speculator, proposal, _model, _valid_hidden = _ready_markov_step()

    with pytest.raises(error, match=match):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)


def test_markov_failure_never_publishes_partial_candidate() -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()
    model.fail_bias_step = 2

    with pytest.raises(RuntimeError, match="markov-step-failure"):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert len(model.markov_inputs) == 3
    assert speculator._markov_result is None
    assert speculator._markov_step_epoch is None
    assert speculator._context_kv_step_epoch == proposal.step_epoch
    assert speculator._prepared_step_epoch is None
    with pytest.raises(RuntimeError, match="already attempted"):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)


def test_new_target_step_immediately_invalidates_old_markov_result() -> None:
    speculator, proposal, _model, hidden_states = _ready_markov_step()
    first_result = speculator._execute_sequential_markov_sampling(proposal, hidden_states)
    kwargs, _auxiliary_states = _step_kwargs()

    speculator.prepare_proposal_inputs(**kwargs)

    assert first_result.candidate_tokens.shape == (2, 5)
    assert speculator._markov_result is None
    assert speculator._markov_step_epoch is None
    assert speculator._markov_attempt_step_epoch is None
    with pytest.raises(RuntimeError, match="stale target step"):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)


def test_loaded_module_replacement_fails_identity_check() -> None:
    speculator, proposal, model, hidden_states = _ready_markov_step()
    speculator._markov_module_contract = speculator._inspect_markov_modules(model)
    model.lm_head = _LoadedHead()

    with pytest.raises(RuntimeError, match="module identity changed"):
        speculator._execute_sequential_markov_sampling(proposal, hidden_states)

    assert speculator._markov_result is None


def test_execute_draft_publishes_completed_markov_candidates(monkeypatch) -> None:
    speculator, proposal, _model, hidden_states = _ready_markov_step()
    speculator._prepared_step_epoch = proposal.step_epoch

    def execute_backbone(actual):
        speculator._prepared_step_epoch = None
        speculator._context_kv_step_epoch = actual.step_epoch
        speculator._draft_forward_step_epoch = actual.step_epoch
        return hidden_states

    monkeypatch.setattr(speculator, "_execute_draft_backbone", execute_backbone)

    published = speculator._execute_draft(proposal)

    assert speculator._markov_result is not None
    assert published is speculator._markov_result.candidate_tokens
    assert published.shape == (2, 5)
    assert speculator._markov_step_epoch == proposal.step_epoch
    assert speculator._published_candidate_tokens is published
    assert speculator._proposal_publication_count == 1
