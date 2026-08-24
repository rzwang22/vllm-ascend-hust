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

DRAFT_CONTEXT_KV_PRECOMPUTED = "DRAFT_CONTEXT_KV_PRECOMPUTED"
DRAFT_FORWARD_COMPLETED = "DRAFT_FORWARD_COMPLETED"
DRAFT_FORWARD_ONLY_PASS = "DRAFT_FORWARD_ONLY_PASS"
DRAFT_FORWARD_STAGES = (
    DRAFT_CONTEXT_KV_PRECOMPUTED,
    DRAFT_FORWARD_COMPLETED,
    DRAFT_FORWARD_ONLY_PASS,
)


class _DraftForwardStageTracker:
    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.step_epoch: int | None = None
        self.stages: list[str] = []

    def mark(self, stage: str, **details: Any) -> None:
        if stage == CLEANUP_COMPLETE:
            if stage in self.stages:
                raise RuntimeError("CLEANUP_COMPLETE was already recorded.")
        else:
            expected = DRAFT_FORWARD_STAGES[len(self.stages)]
            if stage != expected:
                raise RuntimeError(
                    f"Invalid DSpark draft-forward-only stage transition: expected {expected!r}, got {stage!r}."
                )
        self.stages.append(stage)
        print(
            "DSPARK_DRAFT_FORWARD_STAGE="
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
            "DSPARK_DRAFT_FORWARD_FAILURE="
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


def _run_real_draft_backbone(runtime: Any, tracker: _DraftForwardStageTracker) -> None:
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
    assert tuple(draft_attn_metadata) == speculator.draft_attn_layer_order
    output = speculator._run_draft_model_forward(execution, draft_attn_metadata)
    torch.npu.synchronize()

    hidden_size = int(speculator.draft_model_config.hf_config.hidden_size)
    logical_shape = (proposal.num_query_tokens, hidden_size)
    assert output.shape == logical_shape
    assert output.dtype.is_floating_point
    assert output.device == proposal.draft_input_ids.device
    assert output.numel() > 0
    assert output.device.type != "meta"
    output_finite = bool(torch.isfinite(output).all().item())
    assert output_finite

    cache_audit = speculator.audit_target_draft_cache_isolation()
    assert cache_audit["target_cache_object_alias_count"] == 0
    assert cache_audit["target_cache_byte_range_overlap_count"] == 0
    assert len(speculator.draft_attn_layer_order) == 3
    assert all(layer_name.startswith("mtp.") for layer_name in speculator.draft_attn_layer_order)
    assert all(
        actual is expected
        for actual, expected in zip(
            proposal.auxiliary_hidden_states,
            runtime.aux_hidden_states,
        )
    )
    for layer_name in speculator.draft_attn_layer_order:
        group_id = proposal.draft_layer_group_ids[layer_name]
        assert speculator.draft_layer_group_ids[layer_name] == group_id
        assert proposal.draft_block_tables[group_id] is runtime.runner.block_tables.input_block_tables[group_id]
        assert proposal.draft_query_slot_mappings[layer_name].data_ptr() == (
            runtime.runner.block_tables.slot_mappings[group_id].data_ptr()
        )
        assert layer_name in draft_attn_metadata

    tracker.mark(
        DRAFT_FORWARD_COMPLETED,
        request_ids=proposal.request_ids,
        output_shape=tuple(output.shape),
    )
    contract = {
        "rank": runtime.launch.rank,
        "step_epoch": proposal.step_epoch,
        "request_ids": proposal.request_ids,
        "request_count": proposal.num_reqs,
        "K": proposal.num_speculative_tokens,
        "logical_draft_token_count": proposal.num_query_tokens,
        "execution_draft_token_count": execution.execution_token_count,
        "aux_count": len(proposal.auxiliary_hidden_states),
        "aux_layer_order": proposal.target_layer_ids,
        "aux_source": "ExecuteModelState.aux_hidden_states",
        "aux_identity_preserved": True,
        "context_token_count": proposal.num_target_tokens,
        "context_kv_precomputed": True,
        "draft_model_class": (f"{type(runtime.draft_model).__module__}.{type(runtime.draft_model).__name__}"),
        "draft_cache_full_names": speculator.draft_attn_layer_order,
        "draft_layer_group_ids": dict(speculator.draft_layer_group_ids),
        **cache_audit,
        "output_structure": "torch.Tensor",
        "output_shape": tuple(output.shape),
        "logical_output_shape": logical_shape,
        "output_hidden_size": hidden_size,
        "output_dtype": str(output.dtype),
        "output_device": str(output.device),
        "output_finite": output_finite,
        "output_is_empty": output.numel() == 0,
        "output_is_meta": output.device.type == "meta",
        "draft_forward": True,
        "markov_sampling": False,
        "proposal": False,
        "verification": False,
        "generation": False,
    }
    print(
        "DSPARK_DRAFT_FORWARD_CONTRACT=" + json.dumps(contract, default=str, sort_keys=True),
        flush=True,
    )
    torch.distributed.barrier()
    tracker.mark(
        DRAFT_FORWARD_ONLY_PASS,
        request_ids=proposal.request_ids,
        draft_forward=True,
        markov_sampling=False,
        proposal=False,
        verification=False,
        generation=False,
    )


def test_dspark_draft_backbone_forward_only_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    tracker = _DraftForwardStageTracker(launch.rank)

    def callback(runtime: Any) -> None:
        _run_real_draft_backbone(runtime, tracker)

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
