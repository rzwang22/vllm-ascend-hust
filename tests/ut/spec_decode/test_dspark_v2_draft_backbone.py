# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_ascend.worker.v2.spec_decode.dspark.speculator as speculator_module
from tests.ut.spec_decode.test_dspark_v2_proposal_inputs import (
    DRAFT_LAYERS,
    _ready_speculator,
    _step_kwargs,
)
from vllm_ascend.spec_decode import DSparkRuntimeNotWiredError


def _prepare():
    speculator = _ready_speculator()
    kwargs, auxiliary_states = _step_kwargs()
    proposal = speculator.prepare_proposal_inputs(**kwargs)
    return speculator, proposal, auxiliary_states


def test_context_precompute_consumes_real_auxiliary_states_in_layer_order() -> None:
    speculator, proposal, auxiliary_states = _prepare()

    execution = speculator._combine_and_precompute_draft_context(proposal)

    assert all(
        actual is expected
        for actual, expected in zip(
            proposal.auxiliary_hidden_states,
            auxiliary_states,
        )
    )
    draft_model = speculator.model
    assert torch.equal(
        draft_model.combined_aux,
        torch.cat(auxiliary_states, dim=-1),
    )
    assert draft_model.precomputed is not None
    context_states, positions, context_slots = draft_model.precomputed
    assert context_states.data_ptr() == draft_model.combined_aux.data_ptr()
    assert torch.equal(positions, proposal.target_positions[: proposal.num_target_tokens])
    assert context_slots == [
        proposal.draft_context_slot_mappings[layer_name] for layer_name in speculator.draft_attn_layer_order
    ]
    assert execution.execution_token_count == proposal.num_reqs * proposal.num_speculative_tokens
    assert speculator._prepared_step_epoch is None


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda proposal: replace(proposal, rank=proposal.rank + 1), RuntimeError, "different NPU rank"),
        (lambda proposal: replace(proposal, request_ids=()), ValueError, "request ID"),
        (
            lambda proposal: replace(
                proposal,
                request_ids=("", proposal.request_ids[1]),
            ),
            ValueError,
            "request ID",
        ),
        (
            lambda proposal: replace(proposal, target_layer_ids=(40, 42, 41)),
            RuntimeError,
            "target-layer order",
        ),
        (
            lambda proposal: replace(proposal, auxiliary_hidden_states=proposal.auxiliary_hidden_states[:2]),
            ValueError,
            "exactly three",
        ),
        (
            lambda proposal: replace(
                proposal,
                auxiliary_hidden_states=(
                    proposal.auxiliary_hidden_states[0],
                    torch.ones(5, 7, dtype=torch.bfloat16),
                    proposal.auxiliary_hidden_states[2],
                ),
            ),
            ValueError,
            "draft hidden size",
        ),
        (
            lambda proposal: replace(
                proposal,
                auxiliary_hidden_states=(
                    proposal.auxiliary_hidden_states[0],
                    torch.ones(5, 8, dtype=torch.int32),
                    proposal.auxiliary_hidden_states[2],
                ),
            ),
            TypeError,
            "floating",
        ),
        (
            lambda proposal: replace(
                proposal,
                auxiliary_hidden_states=(
                    proposal.auxiliary_hidden_states[0],
                    torch.ones(5, 8, dtype=torch.bfloat16, device="meta"),
                    proposal.auxiliary_hidden_states[2],
                ),
            ),
            RuntimeError,
            "current rank device",
        ),
    ],
)
def test_backbone_input_contract_fails_closed(mutation, error, match) -> None:
    speculator, proposal, _auxiliary_states = _prepare()

    with pytest.raises(error, match=match):
        speculator._combine_and_precompute_draft_context(mutation(proposal))


def test_consumed_or_stale_proposal_cannot_be_reused() -> None:
    speculator, proposal, _auxiliary_states = _prepare()
    speculator._combine_and_precompute_draft_context(proposal)

    with pytest.raises(RuntimeError, match="stale|already consumed"):
        speculator._combine_and_precompute_draft_context(proposal)

    kwargs, _ = _step_kwargs()
    next_proposal = speculator.prepare_proposal_inputs(**kwargs)
    speculator._combine_and_precompute_draft_context(next_proposal)
    with pytest.raises(RuntimeError, match="stale"):
        speculator._validate_draft_execution_current(
            SimpleNamespace(
                proposal_inputs=proposal,
                execution_token_count=proposal.num_query_tokens,
            )
        )


def _install_target_cache(monkeypatch, speculator, target_cache: torch.Tensor) -> None:
    target_layer = SimpleNamespace(kv_cache=target_cache)
    speculator.target_attn_layer_names = frozenset({"model.layers.0.self_attn.swa_cache"})
    monkeypatch.setattr(
        speculator_module,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {"model.layers.0.self_attn.swa_cache": target_layer},
    )


def test_cache_audit_rejects_object_alias(monkeypatch) -> None:
    speculator, proposal, _auxiliary_states = _prepare()
    shared = next(iter(speculator.draft_kv_caches.values()))
    _install_target_cache(monkeypatch, speculator, shared)

    audit = speculator.audit_target_draft_cache_isolation()
    assert audit["target_cache_object_alias_count"] == 1
    assert audit["target_cache_byte_range_overlap_count"] == 1
    with pytest.raises(RuntimeError, match="same tensor object"):
        speculator._combine_and_precompute_draft_context(proposal)


def test_cache_audit_rejects_byte_overlap_without_object_alias(monkeypatch) -> None:
    speculator, proposal, _auxiliary_states = _prepare()
    backing = torch.zeros(64, dtype=torch.uint8)
    _install_target_cache(monkeypatch, speculator, backing[:32])
    draft_caches = dict(speculator.draft_kv_caches)
    draft_caches[DRAFT_LAYERS[0]] = backing[16:48]
    speculator.draft_kv_caches = MappingProxyType(draft_caches)

    audit = speculator.audit_target_draft_cache_isolation()
    assert audit["target_cache_object_alias_count"] == 0
    assert audit["target_cache_byte_range_overlap_count"] == 1
    assert audit["shared_backing_base_count"] == 1
    with pytest.raises(RuntimeError, match="occupied byte ranges"):
        speculator._combine_and_precompute_draft_context(proposal)


def test_cache_audit_accepts_shared_backing_with_disjoint_byte_ranges(monkeypatch) -> None:
    speculator, _proposal, _auxiliary_states = _prepare()
    backing = torch.zeros(128, dtype=torch.uint8)
    _install_target_cache(monkeypatch, speculator, backing[:32])
    speculator.draft_kv_caches = MappingProxyType(
        {
            DRAFT_LAYERS[0]: backing[32:48],
            DRAFT_LAYERS[1]: backing[48:64],
            DRAFT_LAYERS[2]: backing[64:80],
        }
    )

    assert speculator.audit_target_draft_cache_isolation() == {
        "target_cache_object_alias_count": 0,
        "target_cache_byte_range_overlap_count": 0,
        "shared_backing_base_count": 1,
    }


def test_page_strided_cache_audit_uses_occupied_intervals(monkeypatch) -> None:
    speculator, _proposal, _auxiliary_states = _prepare()
    backing = torch.zeros(64, dtype=torch.uint8)
    target = torch.as_strided(backing, (2, 8), (32, 1), storage_offset=0)
    draft = torch.as_strided(backing, (2, 8), (32, 1), storage_offset=8)
    _install_target_cache(monkeypatch, speculator, target)
    speculator.draft_kv_caches = MappingProxyType(
        {
            DRAFT_LAYERS[0]: draft,
            DRAFT_LAYERS[1]: torch.zeros(8),
            DRAFT_LAYERS[2]: torch.zeros(8),
        }
    )

    audit = speculator.audit_target_draft_cache_isolation()
    assert audit["shared_backing_base_count"] == 1
    assert audit["target_cache_byte_range_overlap_count"] == 0


def test_metadata_uses_proposal_query_group_and_slot_state(monkeypatch) -> None:
    speculator, proposal, _auxiliary_states = _prepare()
    speculator.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    speculator.attn_groups = [[], []]
    execution = speculator._combine_and_precompute_draft_context(proposal)
    recorded = {}

    def fake_build_attn_metadata(**kwargs):
        recorded.update(kwargs)
        return {layer_name: object() for layer_name in reversed(DRAFT_LAYERS)}

    monkeypatch.setattr(speculator_module, "build_attn_metadata", fake_build_attn_metadata)
    metadata = speculator._build_draft_forward_metadata(execution)

    assert tuple(metadata) == DRAFT_LAYERS
    assert recorded["query_start_loc_gpu"] is proposal.draft_query_start_loc
    assert recorded["seq_lens"] is proposal.draft_sequence_lengths
    assert recorded["positions"] is proposal.draft_positions
    assert recorded["is_prefilling"] is proposal.draft_is_prefilling
    assert recorded["num_tokens"] == proposal.num_query_tokens
    assert recorded["num_actual_tokens"] == proposal.num_query_tokens
    assert recorded["attn_state"].name == "DecodeOnly"
    assert recorded["causal"] is False
    for group_id, block_table in proposal.draft_block_tables.items():
        assert recorded["block_tables"][group_id] is block_table
    for layer_name, group_id in proposal.draft_layer_group_ids.items():
        assert recorded["slot_mappings"][group_id].data_ptr() == (
            proposal.draft_query_slot_mappings[layer_name].data_ptr()
        )


def test_metadata_preserves_current_target_prefill_ownership(monkeypatch) -> None:
    speculator = _ready_speculator()
    kwargs, _auxiliary_states = _step_kwargs()
    kwargs["input_batch"].is_prefilling_np = np.array(
        [True, False],
        dtype=bool,
    )
    proposal = speculator.prepare_proposal_inputs(**kwargs)
    speculator.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    speculator.attn_groups = [[], []]
    execution = speculator._combine_and_precompute_draft_context(proposal)
    recorded = {}

    def fake_build_attn_metadata(**metadata_kwargs):
        recorded.update(metadata_kwargs)
        return {layer_name: object() for layer_name in DRAFT_LAYERS}

    monkeypatch.setattr(speculator_module, "build_attn_metadata", fake_build_attn_metadata)
    speculator._build_draft_forward_metadata(execution)

    assert torch.equal(
        proposal.draft_is_prefilling,
        torch.tensor([True, False]),
    )
    assert recorded["is_prefilling"] is proposal.draft_is_prefilling
    assert recorded["attn_state"].name == "ChunkedPrefill"


def _execution_and_metadata(monkeypatch):
    speculator, proposal, _auxiliary_states = _prepare()
    execution = speculator._combine_and_precompute_draft_context(proposal)
    metadata = {layer_name: object() for layer_name in DRAFT_LAYERS}
    previous_context = object()
    context_state = {"current": previous_context}
    calls = {}

    @contextmanager
    def fake_set_forward_context(*args, **kwargs):
        calls["set_args"] = args
        calls["set_kwargs"] = kwargs
        old_context = context_state["current"]
        context_state["current"] = SimpleNamespace(
            dp_metadata=None,
            additional_kwargs={"target_only": True},
        )
        try:
            yield
        finally:
            context_state["current"] = old_context

    def fake_build_ascend_forward_context(**kwargs):
        calls["ascend_kwargs"] = kwargs
        return {"is_draft_model": True}

    monkeypatch.setattr(speculator_module, "set_forward_context", fake_set_forward_context)
    monkeypatch.setattr(
        speculator_module,
        "get_forward_context",
        lambda: context_state["current"],
    )
    monkeypatch.setattr(
        speculator_module,
        "build_ascend_forward_context",
        fake_build_ascend_forward_context,
    )
    return speculator, proposal, execution, metadata, previous_context, context_state, calls


def test_eager_forward_context_receives_real_model_inputs_and_restores(monkeypatch) -> None:
    speculator, proposal, execution, metadata, previous, context_state, calls = _execution_and_metadata(monkeypatch)

    output = speculator._run_draft_model_forward(execution, metadata)

    assert output is speculator.model.forward_output
    assert output.shape == (proposal.num_query_tokens, 8)
    assert calls["set_kwargs"]["cudagraph_runtime_mode"].name == "NONE"
    assert calls["set_kwargs"]["input_ids"] is proposal.draft_input_ids
    assert calls["set_kwargs"]["model_instance"] is speculator.model
    assert all(
        calls["set_kwargs"]["slot_mapping"][layer_name] is slots
        for layer_name, slots in proposal.draft_query_slot_mappings.items()
    )
    assert calls["ascend_kwargs"]["is_draft_model"] is True
    assert calls["ascend_kwargs"]["model_instance"] is speculator.model
    assert context_state["current"] is previous


def test_forward_exception_restores_context_and_invalidates_execution(monkeypatch) -> None:
    speculator, _proposal, execution, metadata, previous, context_state, _calls = _execution_and_metadata(monkeypatch)

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("draft-forward-failure")

    monkeypatch.setattr(speculator.model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="draft-forward-failure"):
        speculator._run_draft_model_forward(execution, metadata)
    assert context_state["current"] is previous
    with pytest.raises(RuntimeError, match="already consumed"):
        speculator._run_draft_model_forward(execution, metadata)


def test_forward_rejects_output_shape_from_real_model_abi(monkeypatch) -> None:
    speculator, _proposal, execution, metadata, _previous, _context_state, _calls = _execution_and_metadata(monkeypatch)
    monkeypatch.setattr(
        speculator.model,
        "forward",
        lambda *_args, **_kwargs: torch.ones(execution.execution_token_count, 7),
    )

    with pytest.raises(RuntimeError, match="HC-head ABI"):
        speculator._run_draft_model_forward(execution, metadata)


def test_execute_draft_runs_backbone_then_fails_at_markov_boundary(monkeypatch) -> None:
    speculator, proposal, _auxiliary_states = _prepare()
    output = torch.ones(proposal.num_query_tokens, 8)
    calls = []

    def execute_backbone(actual):
        calls.append(actual)
        return output

    monkeypatch.setattr(speculator, "_execute_draft_backbone", execute_backbone)
    with pytest.raises(DSparkRuntimeNotWiredError, match="V2 DSpark Markov sampling"):
        speculator._execute_draft(proposal)
    assert calls == [proposal]
