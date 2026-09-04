# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.worker.v2.spec_decode.dspark import (
    AscendDSparkProposalInputs,
    create_dspark_speculator,
)

TARGET_LAYER_IDS = (40, 41, 42)
TARGET_OUTPUT_BOUNDARIES = (41, 42, 43)
DRAFT_LAYERS = (
    "mtp.0.self_attn.swa_cache",
    "mtp.1.self_attn.swa_cache",
    "mtp.2.self_attn.swa_cache",
)


class _TargetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(
            aux_hidden_state_layers=TARGET_OUTPUT_BOUNDARIES,
        )


class _DraftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.combined_aux: torch.Tensor | None = None
        self.precomputed: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | None = None
        self.forward_output: torch.Tensor | None = None

    def combine_hidden_states(self, auxiliary_states: torch.Tensor) -> torch.Tensor:
        self.combined_aux = auxiliary_states
        return auxiliary_states[:, :8]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        positions: torch.Tensor,
        slot_mappings: list[torch.Tensor],
    ) -> None:
        self.precomputed = (context_states, positions, slot_mappings)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        self.forward_output = torch.ones(input_ids.shape[0], 8, dtype=torch.bfloat16)
        return self.forward_output


class _BlockTables:
    def __init__(self) -> None:
        self.kernel_block_sizes = [4, 4]
        self.slot_mappings = torch.full((2, 32), -1, dtype=torch.int32)
        self.input_block_tables = [
            torch.tensor(
                [
                    [10, 11, 12, 13, 14, 15, 16, 17],
                    [20, 21, 22, 23, 24, 25, 26, 27],
                    [50, 51, 52, 53, 54, 55, 56, 57],
                ],
                dtype=torch.int32,
            ),
            torch.tensor(
                [
                    [30, 31, 32, 33, 34, 35, 36, 37],
                    [40, 41, 42, 43, 44, 45, 46, 47],
                    [60, 61, 62, 63, 64, 65, 66, 67],
                ],
                dtype=torch.int32,
            ),
        ]


def _config(*, continue_after_verification: bool = False):
    return SimpleNamespace(
        additional_config={
            "dspark_continue_after_verification": continue_after_verification,
        },
        model_config=SimpleNamespace(max_model_len=64),
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            is_moe_model=False,
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    dspark_noise_token_id=127,
                    dspark_target_layer_ids=list(TARGET_LAYER_IDS),
                    hidden_size=8,
                    vocab_size=16,
                ),
            ),
            num_speculative_tokens=5,
        ),
    )


def _ready_speculator(*, continue_after_verification: bool = False):
    speculator = create_dspark_speculator(
        _config(continue_after_verification=continue_after_verification),
        torch.device("cpu"),
    )
    speculator._model = _DraftModel()
    speculator._loaded_target_model = _TargetModel()
    speculator.target_attn_layer_names = frozenset()
    speculator.draft_attn_layer_names = frozenset(DRAFT_LAYERS)
    speculator.draft_attn_layer_order = DRAFT_LAYERS
    speculator.draft_layer_group_ids = MappingProxyType(
        {
            DRAFT_LAYERS[0]: 0,
            DRAFT_LAYERS[1]: 1,
            DRAFT_LAYERS[2]: 1,
        },
    )
    speculator.draft_kv_cache_group_ids = (0, 1)
    speculator.draft_kv_caches = MappingProxyType(
        {layer_name: torch.zeros(8, dtype=torch.bfloat16) for layer_name in DRAFT_LAYERS},
    )
    speculator.block_tables = _BlockTables()
    speculator.kv_cache_config = object()
    return speculator


def _input_batch() -> InputBatch:
    num_scheduled_tokens = np.array([2, 3], dtype=np.int32)
    query_start_loc_np = np.array([0, 2, 5], dtype=np.int32)
    return InputBatch(
        req_ids=["request-b", "request-a"],
        num_reqs=2,
        num_reqs_after_padding=2,
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        idx_mapping_np=np.array([1, 0], dtype=np.int32),
        expanded_idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        expanded_local_pos=torch.zeros(2, dtype=torch.int32),
        num_scheduled_tokens=num_scheduled_tokens,
        num_tokens=5,
        num_tokens_after_padding=5,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=torch.from_numpy(query_start_loc_np.copy()),
        query_start_loc_np=query_start_loc_np,
        seq_lens=torch.tensor([7, 13], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([7, 13], dtype=torch.int32),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.array([5, 10], dtype=np.int32),
        prefill_len_np=np.array([2, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([2, 3], dtype=np.int32),
        is_prefilling_np=np.array([False, False]),
        max_seq_len_np=None,
        input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        positions=torch.tensor([5, 6, 10, 11, 12], dtype=torch.int64),
        is_padding=torch.zeros(5, dtype=torch.bool),
        logits_indices=torch.tensor([1, 4], dtype=torch.int64),
        cu_num_logits=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 1, 2], dtype=np.int32),
        has_structured_output_reqs=False,
        prompt_lens=None,
    )


def _step_kwargs() -> tuple[dict, list[torch.Tensor]]:
    aux_hidden_states = [torch.full((5, 8), float(layer), dtype=torch.bfloat16) for layer in TARGET_LAYER_IDS]
    group_zero_slots = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32)
    group_one_slots = torch.tensor([200, 201, 202, 203, 204], dtype=torch.int32)
    kwargs = {
        "input_batch": _input_batch(),
        "attn_metadata": {"target": object()},
        "slot_mappings": {
            DRAFT_LAYERS[0]: group_zero_slots,
            DRAFT_LAYERS[1]: group_one_slots,
            DRAFT_LAYERS[2]: group_one_slots,
        },
        "last_hidden_states": torch.ones(5, 16, dtype=torch.bfloat16),
        "aux_hidden_states": aux_hidden_states,
        "num_sampled": torch.tensor([1, 1], dtype=torch.int32),
        "num_rejected": torch.tensor([0, 1], dtype=torch.int32),
        "last_sampled": torch.tensor([[101], [202]], dtype=torch.int64),
        "next_prefill_tokens": torch.tensor([11, 22], dtype=torch.int32),
        "temperature": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "seeds": torch.tensor([7, 9], dtype=torch.int64),
    }
    return kwargs, aux_hidden_states


def test_prepare_proposal_inputs_preserves_aux_order_identity_and_shapes() -> None:
    speculator = _ready_speculator()
    kwargs, aux_hidden_states = _step_kwargs()

    prepared = speculator.prepare_proposal_inputs(**kwargs)

    assert type(prepared) is AscendDSparkProposalInputs
    assert prepared.request_ids == ("request-b", "request-a")
    assert prepared.target_layer_ids == TARGET_LAYER_IDS
    assert all(
        actual is expected
        for actual, expected in zip(
            prepared.auxiliary_hidden_states,
            aux_hidden_states,
        )
    )
    assert prepared.num_reqs == 2
    assert prepared.num_target_tokens == 5
    assert prepared.num_query_tokens == 10
    assert prepared.request_state_indices is kwargs["input_batch"].idx_mapping
    assert prepared.last_hidden_states is kwargs["last_hidden_states"]
    assert torch.equal(prepared.anchor_token_ids, torch.tensor([202, 101], dtype=torch.int32))
    assert torch.equal(
        prepared.draft_input_ids.view(2, 5),
        torch.tensor(
            [
                [202, 127, 127, 127, 127],
                [101, 127, 127, 127, 127],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        prepared.draft_positions.view(2, 5),
        torch.tensor(
            [
                [7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16],
            ],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        prepared.draft_query_start_loc,
        torch.tensor([0, 5, 10], dtype=torch.int32),
    )
    assert torch.equal(
        prepared.draft_sequence_lengths,
        torch.tensor([12, 17], dtype=torch.int32),
    )
    assert torch.equal(prepared.draft_is_prefilling, torch.tensor([False, False]))


def test_prepare_proposal_inputs_updates_block_and_slot_metadata_by_group() -> None:
    speculator = _ready_speculator()
    kwargs, _aux_hidden_states = _step_kwargs()

    prepared = speculator.prepare_proposal_inputs(**kwargs)

    assert prepared.draft_layer_group_ids == {
        DRAFT_LAYERS[0]: 0,
        DRAFT_LAYERS[1]: 1,
        DRAFT_LAYERS[2]: 1,
    }
    assert prepared.draft_block_tables[0] is speculator.block_tables.input_block_tables[0]
    assert prepared.draft_block_tables[1] is speculator.block_tables.input_block_tables[1]
    assert (
        prepared.draft_query_slot_mappings[DRAFT_LAYERS[1]].data_ptr()
        == prepared.draft_query_slot_mappings[DRAFT_LAYERS[2]].data_ptr()
    )
    assert (
        prepared.draft_context_slot_mappings[DRAFT_LAYERS[1]] is prepared.draft_context_slot_mappings[DRAFT_LAYERS[2]]
    )
    assert torch.equal(
        prepared.draft_context_slot_mappings[DRAFT_LAYERS[0]],
        kwargs["slot_mappings"][DRAFT_LAYERS[0]],
    )
    assert prepared.draft_context_slot_mappings[DRAFT_LAYERS[0]] is not kwargs["slot_mappings"][DRAFT_LAYERS[0]]
    assert (
        prepared.draft_query_slot_mappings[DRAFT_LAYERS[0]].data_ptr()
        == speculator.block_tables.slot_mappings[0].data_ptr()
    )
    assert (
        prepared.draft_query_slot_mappings[DRAFT_LAYERS[1]].data_ptr()
        == speculator.block_tables.slot_mappings[1].data_ptr()
    )
    assert torch.equal(
        prepared.draft_query_slot_mappings[DRAFT_LAYERS[0]].view(2, 5),
        torch.tensor(
            [
                [47, 48, 49, 50, 51],
                [92, 93, 94, 95, 96],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        prepared.draft_query_slot_mappings[DRAFT_LAYERS[1]].view(2, 5),
        torch.tensor(
            [
                [127, 128, 129, 130, 131],
                [172, 173, 174, 175, 176],
            ],
            dtype=torch.int32,
        ),
    )


@pytest.mark.parametrize(
    ("num_scheduled_tokens", "positions", "seq_lens"),
    [
        ([1], [20], [21]),
        ([2, 1, 3], [1, 2, 8, 15, 16, 17], [3, 9, 18]),
    ],
)
def test_prepare_proposal_inputs_supports_different_batch_and_decode_shapes(
    num_scheduled_tokens: list[int],
    positions: list[int],
    seq_lens: list[int],
) -> None:
    speculator = _ready_speculator()
    num_reqs = len(num_scheduled_tokens)
    num_tokens = len(positions)
    query_start = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, out=query_start[1:])
    batch = _input_batch()
    batch.req_ids = [f"request-{index}" for index in range(num_reqs)]
    batch.num_reqs = num_reqs
    batch.num_reqs_after_padding = num_reqs
    batch.idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
    batch.idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    batch.num_scheduled_tokens = np.asarray(num_scheduled_tokens, dtype=np.int32)
    batch.num_tokens = num_tokens
    batch.num_tokens_after_padding = num_tokens
    batch.is_prefilling_np = np.zeros(num_reqs, dtype=np.bool_)
    batch.query_start_loc = torch.from_numpy(query_start.copy())
    batch.query_start_loc_np = query_start
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    batch.input_ids = torch.arange(num_tokens, dtype=torch.int32)
    batch.positions = torch.tensor(positions, dtype=torch.int64)
    batch.is_padding = torch.zeros(num_tokens, dtype=torch.bool)
    kwargs, _aux = _step_kwargs()
    kwargs.update(
        input_batch=batch,
        slot_mappings={
            DRAFT_LAYERS[0]: torch.arange(num_tokens, dtype=torch.int32),
            DRAFT_LAYERS[1]: torch.arange(num_tokens, dtype=torch.int32),
        },
        last_hidden_states=torch.ones(num_tokens, 16, dtype=torch.bfloat16),
        aux_hidden_states=[torch.ones(num_tokens, 8, dtype=torch.bfloat16) for _ in TARGET_LAYER_IDS],
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
        last_sampled=torch.arange(num_reqs, dtype=torch.int64).view(-1, 1),
        next_prefill_tokens=torch.arange(num_reqs, dtype=torch.int32),
        temperature=torch.zeros(num_reqs),
        seeds=torch.arange(num_reqs, dtype=torch.int64),
    )
    kwargs["slot_mappings"][DRAFT_LAYERS[2]] = kwargs["slot_mappings"][DRAFT_LAYERS[1]]

    prepared = speculator.prepare_proposal_inputs(**kwargs)

    assert prepared.draft_input_ids.shape == (num_reqs * 5,)
    assert prepared.draft_positions.shape == (num_reqs * 5,)
    assert prepared.draft_query_start_loc.shape == (num_reqs + 1,)
    assert prepared.draft_sequence_lengths.shape == (num_reqs,)


@pytest.mark.parametrize(
    ("mutate", "error_type", "error_match"),
    [
        (lambda kwargs: kwargs.update(aux_hidden_states=None), RuntimeError, "missing"),
        (lambda kwargs: kwargs.update(aux_hidden_states=kwargs["aux_hidden_states"][:2]), ValueError, "one auxiliary"),
        (
            lambda kwargs: kwargs["aux_hidden_states"].__setitem__(1, torch.ones(4, 8, dtype=torch.bfloat16)),
            ValueError,
            "token dimension",
        ),
        (
            lambda kwargs: kwargs["aux_hidden_states"].__setitem__(
                1,
                torch.ones(5, 7, dtype=torch.bfloat16),
            ),
            ValueError,
            "identical shape",
        ),
        (
            lambda kwargs: kwargs["aux_hidden_states"].__setitem__(1, torch.ones(5, 8, dtype=torch.int32)),
            TypeError,
            "floating point",
        ),
        (
            lambda kwargs: kwargs.update(
                last_hidden_states=torch.ones(5, 16, dtype=torch.float32),
            ),
            TypeError,
            "same dtype",
        ),
        (
            lambda kwargs: kwargs["slot_mappings"].pop(DRAFT_LAYERS[2]),
            RuntimeError,
            "did not provide a slot mapping",
        ),
        (
            lambda kwargs: kwargs.update(last_hidden_states=torch.ones(5, 16, device="meta")),
            RuntimeError,
            "current rank device",
        ),
        (
            lambda kwargs: kwargs["input_batch"].req_ids.__setitem__(
                1,
                "request-b",
            ),
            ValueError,
            "request IDs",
        ),
        (
            lambda kwargs: kwargs["input_batch"].idx_mapping_np.__setitem__(
                1,
                -1,
            ),
            ValueError,
            "request-state ownership",
        ),
    ],
)
def test_prepare_proposal_inputs_rejects_invalid_or_cross_rank_state(
    mutate,
    error_type: type[Exception],
    error_match: str,
) -> None:
    speculator = _ready_speculator()
    kwargs, _aux = _step_kwargs()
    mutate(kwargs)

    with pytest.raises(error_type, match=error_match):
        speculator.prepare_proposal_inputs(**kwargs)


def test_prepare_failure_is_atomic_and_next_step_invalidates_old_inputs() -> None:
    speculator = _ready_speculator()
    kwargs, first_aux = _step_kwargs()
    first = speculator.prepare_proposal_inputs(**kwargs)

    invalid_kwargs, _invalid_aux = _step_kwargs()
    invalid_kwargs["aux_hidden_states"] = invalid_kwargs["aux_hidden_states"][:2]
    with pytest.raises(ValueError, match="one auxiliary"):
        speculator.prepare_proposal_inputs(**invalid_kwargs)
    with pytest.raises(RuntimeError, match="stale"):
        speculator.validate_prepared_inputs_current(first)

    second_kwargs, second_aux = _step_kwargs()
    second_kwargs["input_batch"].req_ids = ["request-d", "request-c"]
    second = speculator.prepare_proposal_inputs(**second_kwargs)

    assert second.step_epoch == first.step_epoch + 2
    assert second.request_ids == ("request-d", "request-c")
    assert second.auxiliary_hidden_states[0] is second_aux[0]
    assert second.auxiliary_hidden_states[0] is not first_aux[0]
    with pytest.raises(RuntimeError, match="stale"):
        speculator.validate_prepared_inputs_current(first)
    speculator.validate_prepared_inputs_current(second)


def test_prepare_rejects_inconsistent_group_slots_without_partial_update() -> None:
    speculator = _ready_speculator()
    kwargs, _aux = _step_kwargs()
    original_slots = speculator.block_tables.slot_mappings.clone()
    kwargs["slot_mappings"][DRAFT_LAYERS[2]] = kwargs["slot_mappings"][DRAFT_LAYERS[1]].clone()

    with pytest.raises(RuntimeError, match="different target slot-mapping"):
        speculator.prepare_proposal_inputs(**kwargs)

    assert speculator._proposal_step_epoch == 1
    assert speculator._prepared_step_epoch is None
    assert torch.equal(speculator.block_tables.slot_mappings, original_slots)


def test_repeated_prepare_keeps_shared_slot_buffer_identity() -> None:
    speculator = _ready_speculator()
    kwargs, _aux = _step_kwargs()
    shared_ptrs = tuple(row.data_ptr() for row in speculator.block_tables.slot_mappings)

    first = speculator.prepare_proposal_inputs(**kwargs)
    first_slots = {name: slots.clone() for name, slots in first.draft_query_slot_mappings.items()}
    second = speculator.prepare_proposal_inputs(**kwargs)

    assert second.step_epoch == 2
    assert tuple(row.data_ptr() for row in speculator.block_tables.slot_mappings) == shared_ptrs
    assert all(torch.equal(second.draft_query_slot_mappings[name], expected) for name, expected in first_slots.items())


def test_prepared_inputs_reject_different_rank_ownership() -> None:
    speculator = _ready_speculator()
    kwargs, _aux = _step_kwargs()
    prepared = speculator.prepare_proposal_inputs(**kwargs)

    with pytest.raises(RuntimeError, match="different NPU rank"):
        speculator.validate_prepared_inputs_current(
            replace(prepared, rank=prepared.rank + 1),
        )


def test_propose_runs_backbone_and_markov_then_publishes_candidates(monkeypatch) -> None:
    # Imported here to avoid a circular dependency: the shared Markov test
    # fixture imports this module's proposal-input helpers.
    from tests.ut.spec_decode.test_dspark_v2_markov_sampling import (
        _MarkovDraftModel,
    )

    speculator = _ready_speculator()
    speculator.draft_model_config.hf_config.vocab_size = 256
    speculator._model = _MarkovDraftModel()
    speculator._markov_module_contract = None
    kwargs, _aux = _step_kwargs()
    backbone_calls: list[AscendDSparkProposalInputs] = []
    hidden_state_outputs: list[torch.Tensor] = []
    markov_calls: list[tuple[AscendDSparkProposalInputs, torch.Tensor]] = []
    execute_markov_impl = speculator._execute_sequential_markov_sampling

    def execute_backbone(proposal_inputs):
        backbone_calls.append(proposal_inputs)
        speculator._prepared_step_epoch = None
        speculator._context_kv_step_epoch = proposal_inputs.step_epoch
        speculator._draft_forward_step_epoch = proposal_inputs.step_epoch
        hidden_states = torch.ones(
            proposal_inputs.num_query_tokens,
            8,
            dtype=torch.bfloat16,
        )
        hidden_state_outputs.append(hidden_states)
        return hidden_states

    def execute_markov(proposal_inputs, hidden_states):
        markov_calls.append((proposal_inputs, hidden_states))
        return execute_markov_impl(proposal_inputs, hidden_states)

    monkeypatch.setattr(speculator, "_execute_draft_backbone", execute_backbone)
    monkeypatch.setattr(speculator, "_execute_sequential_markov_sampling", execute_markov)

    published = speculator.propose(**kwargs)
    owned_result = speculator._markov_result

    assert speculator._proposal_step_epoch == 1
    assert len(backbone_calls) == 1
    assert backbone_calls[0].step_epoch == 1
    assert len(hidden_state_outputs) == 1
    assert len(markov_calls) == 1
    assert markov_calls[0][0] is backbone_calls[0]
    assert markov_calls[0][1] is hidden_state_outputs[0]
    assert owned_result is not None
    assert owned_result.backbone_hidden_states is hidden_state_outputs[0]
    assert published is owned_result.candidate_tokens
    assert speculator._published_candidate_tokens is published
    assert speculator._proposal_publication_count == 1
    assert published.shape == (
        backbone_calls[0].num_reqs,
        backbone_calls[0].num_speculative_tokens,
    )
    assert published.dtype is torch.int64


@pytest.mark.parametrize("flag", ["dummy_run", "is_profile", "skip_attn_for_dummy_run"])
def test_incomplete_dummy_profile_protocol_fails_closed(flag: str) -> None:
    speculator = _ready_speculator()
    kwargs = {
        "input_batch": None,
        "attn_metadata": {},
        "slot_mappings": {},
        "last_hidden_states": None,
        "aux_hidden_states": None,
        "num_sampled": None,
        "num_rejected": None,
        "last_sampled": None,
        "next_prefill_tokens": None,
        "temperature": None,
        "seeds": None,
        flag: True,
    }

    with pytest.raises(ValueError, match="profile execution requires dummy_run=True"):
        speculator.propose(**kwargs)


def test_proposal_input_module_does_not_import_forbidden_runtime() -> None:
    conftest_path = Path(__file__).parents[1] / "conftest.py"
    child_script = textwrap.dedent(
        f"""
        import json
        import runpy
        import sys
        import types

        build_info = types.ModuleType("vllm_ascend._build_info")
        build_info.__device_type__ = "A2"
        build_info.__soc_version__ = "ASCEND910B2"
        sys.modules["vllm_ascend._build_info"] = build_info
        runpy.run_path({str(conftest_path)!r})

        before = set(sys.modules)
        module = __import__(
            "vllm_ascend.worker.v2.spec_decode.dspark.proposal_inputs",
            fromlist=["AscendDSparkProposalInputs"],
        )
        new_modules = set(sys.modules) - before
        forbidden = (
            "vllm.models.deepseek_v4.nvidia",
            "vllm.v1.worker.gpu.spec_decode.dspark",
            "vllm_ascend.ops.triton.spec_decode",
        )
        print(json.dumps({{
            "class_module": module.AscendDSparkProposalInputs.__module__,
            "forbidden": sorted(
                name for name in new_modules if name.startswith(forbidden)
            ),
        }}, sort_keys=True))
        """
    )
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(entry for entry in sys.path if entry)
    result = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=Path(__file__).parents[3],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    audit = json.loads(result.stdout.splitlines()[-1])
    assert audit == {
        "class_module": "vllm_ascend.worker.v2.spec_decode.dspark.proposal_inputs",
        "forbidden": [],
    }
