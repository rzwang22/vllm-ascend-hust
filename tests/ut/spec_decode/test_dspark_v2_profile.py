# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from tests.ut.spec_decode.test_dspark_v2_proposal_inputs import (
    _ready_speculator,
    _step_kwargs,
)
from vllm_ascend.models.deepseek_v4_dspark import DeepseekV4DSparkModel
from vllm_ascend.worker.v2.spec_decode.dspark import speculator as speculator_module


class _ProfileDraftModel(nn.Module):
    def __init__(self, *, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.combined_aux: torch.Tensor | None = None
        self.precomputed: tuple[torch.Tensor, torch.Tensor, None] | None = None
        self.forward_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.markov_inputs: list[torch.Tensor] = []
        self.fail_markov_step: int | None = None

    def combine_hidden_states(self, auxiliary_states: torch.Tensor) -> torch.Tensor:
        self.combined_aux = auxiliary_states
        return auxiliary_states[:, :8]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: None,
    ) -> None:
        self.precomputed = (context_states, positions, slot_mappings)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        self.forward_inputs = (input_ids, positions)
        return torch.ones(input_ids.shape[0], 8, dtype=torch.bfloat16)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(hidden_states.shape[0], self.vocab_size, dtype=torch.float32)

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.markov_inputs.append(token_ids)
        return token_ids.to(torch.float32).unsqueeze(-1)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        step = len(self.markov_inputs) - 1
        if step == self.fail_markov_step:
            raise RuntimeError("profile-markov-failure")
        selected = (markov_embed[:, 0].to(torch.int64) + 1) % self.vocab_size
        return torch.zeros(markov_embed.shape[0], self.vocab_size).scatter(
            1,
            selected[:, None],
            1.0,
        )

    @staticmethod
    def map_draft_to_target(draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids


def _profile_kwargs() -> dict:
    kwargs, _auxiliary_states = _step_kwargs()
    kwargs["input_batch"].logits_indices = kwargs["input_batch"].logits_indices.to(torch.int32)
    kwargs.update(
        attn_metadata=None,
        slot_mappings=None,
        dummy_run=True,
        skip_attn_for_dummy_run=True,
        is_profile=True,
    )
    return kwargs


def _proposal_state(speculator) -> dict[str, object]:
    return {name: getattr(speculator, name) for name in speculator_module._DSPARK_PROFILE_PRESERVED_STATE}


def _install_forward_context_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict], list[SimpleNamespace]]:
    set_calls: list[dict] = []
    contexts: list[SimpleNamespace] = []

    @contextmanager
    def fake_set_forward_context(*_args, **kwargs):
        set_calls.append(kwargs)
        context = SimpleNamespace(
            dp_metadata=object(),
            additional_kwargs={},
        )
        contexts.append(context)
        try:
            yield
        finally:
            context.exited = True

    monkeypatch.setattr(
        speculator_module,
        "set_forward_context",
        fake_set_forward_context,
    )
    monkeypatch.setattr(
        speculator_module,
        "get_forward_context",
        lambda: contexts[-1],
    )
    monkeypatch.setattr(
        speculator_module,
        "build_ascend_forward_context",
        lambda **kwargs: {"profile_contract": kwargs},
    )
    return set_calls, contexts


def test_profile_executes_context_backbone_and_all_markov_heads_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator = _ready_speculator()
    model = _ProfileDraftModel()
    speculator._model = model
    before = _proposal_state(speculator)
    set_calls, contexts = _install_forward_context_stubs(monkeypatch)

    result = speculator.propose(**_profile_kwargs())

    assert result is None
    assert model.combined_aux is not None
    assert model.combined_aux.shape == (5, 24)
    assert model.precomputed is not None
    assert model.precomputed[0].shape == (5, 8)
    assert model.precomputed[2] is None
    assert model.forward_inputs is not None
    draft_input_ids, draft_positions = model.forward_inputs
    assert torch.equal(
        draft_input_ids.view(2, 5),
        torch.tensor(
            [
                [2, 127, 127, 127, 127],
                [5, 127, 127, 127, 127],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        draft_positions.view(2, 5),
        torch.tensor(
            [
                [7, 8, 9, 10, 11],
                [13, 14, 15, 16, 17],
            ],
            dtype=torch.int64,
        ),
    )
    assert len(model.markov_inputs) == 5
    assert len(set_calls) == 1
    assert set_calls[0]["slot_mapping"] is None
    assert contexts[0].additional_kwargs["is_draft_model_prefill"] is True
    assert contexts[0].additional_kwargs["profile_contract"]["in_profile_run"] is True
    assert contexts[0].additional_kwargs["profile_contract"]["is_draft_model"] is True
    assert contexts[0].exited is True
    after = _proposal_state(speculator)
    assert all(after[name] is value for name, value in before.items())
    assert speculator._published_candidate_tokens is None
    assert speculator._proposal_generated_count == 0


def test_profile_failure_restores_context_and_all_proposal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speculator = _ready_speculator()
    model = _ProfileDraftModel()
    model.fail_markov_step = 2
    speculator._model = model
    sentinel = torch.tensor([[9]], dtype=torch.int64)
    speculator._published_candidate_tokens = sentinel
    speculator._proposal_step_epoch = 17
    speculator._proposal_generated_count = 4
    before = _proposal_state(speculator)
    _set_calls, contexts = _install_forward_context_stubs(monkeypatch)

    with pytest.raises(RuntimeError, match="profile-markov-failure"):
        speculator.propose(**_profile_kwargs())

    after = _proposal_state(speculator)
    assert all(after[name] is value for name, value in before.items())
    assert speculator._published_candidate_tokens is sentinel
    assert speculator._proposal_step_epoch == 17
    assert speculator._proposal_generated_count == 4
    assert contexts[0].exited is True


def test_profile_shape_contract_is_rank_local_and_tp_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_forward_context_stubs(monkeypatch)
    rank_inputs = []
    for rank in (0, 7):
        speculator = _ready_speculator()
        speculator.rank = rank
        model = _ProfileDraftModel()
        speculator._model = model

        speculator.propose(**_profile_kwargs())

        assert model.forward_inputs is not None
        rank_inputs.append(tuple(tensor.clone() for tensor in model.forward_inputs))

    assert torch.equal(rank_inputs[0][0], rank_inputs[1][0])
    assert torch.equal(rank_inputs[0][1], rank_inputs[1][1])
    assert rank_inputs[0][0].shape == (2 * 5,)


def test_profile_context_precompute_executes_real_projections_without_kv_writes() -> None:
    projected_layers: list[tuple[object, torch.Tensor, torch.Tensor]] = []
    stored_layers: list[tuple[object, torch.Tensor, None]] = []
    layers = {
        str(layer_index): SimpleNamespace(self_attn=SimpleNamespace(name=f"mtp.{layer_index}.self_attn"))
        for layer_index in range(3)
    }

    def project(context_states, context_positions, attention):
        projected_layers.append((attention, context_states, context_positions))
        return context_states[:, :4].unsqueeze(1)

    def store(shared_kv, slot_mapping, attention):
        stored_layers.append((attention, shared_kv, slot_mapping))

    owner = SimpleNamespace(
        layers=layers,
        _project_shared_kv=project,
        _store_standard_swa_kv=store,
    )
    context_states = torch.ones(5, 8, dtype=torch.bfloat16)
    context_positions = torch.arange(5, dtype=torch.int64)

    DeepseekV4DSparkModel.precompute_and_store_context_kv(
        owner,
        context_states,
        context_positions,
        None,
    )

    assert [entry[0] for entry in projected_layers] == [layer.self_attn for layer in layers.values()]
    assert all(entry[1] is context_states for entry in projected_layers)
    assert all(entry[2] is context_positions for entry in projected_layers)
    assert [entry[0] for entry in stored_layers] == [layer.self_attn for layer in layers.values()]
    assert all(entry[2] is None for entry in stored_layers)


def test_fixed_kv_memory_configuration_still_profiles_target_and_draft() -> None:
    from vllm_ascend.worker.worker import NPUWorker

    fixed_kv_bytes = 2 * 1024**3
    worker = object.__new__(NPUWorker)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=fixed_kv_bytes)
    worker.model_runner = SimpleNamespace(profile_run=Mock())
    worker.init_snapshot = SimpleNamespace(free_memory=8 * 1024**3)

    available = NPUWorker.determine_available_memory(worker)

    assert available == fixed_kv_bytes
    worker.model_runner.profile_run.assert_called_once_with()


@pytest.mark.parametrize(
    "flags",
    [
        {"dummy_run": True},
        {"is_profile": True},
        {"skip_attn_for_dummy_run": True},
        {"dummy_run": True, "is_profile": True},
    ],
)
def test_incomplete_profile_protocol_fails_closed(flags: dict[str, bool]) -> None:
    speculator = _ready_speculator()
    kwargs, _auxiliary_states = _step_kwargs()
    kwargs.update(flags)

    with pytest.raises(ValueError, match="requires dummy_run=True"):
        speculator.propose(**kwargs)
