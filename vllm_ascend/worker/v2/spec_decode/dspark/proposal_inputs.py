# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class AscendDSparkProposalInputs:
    """One target step's validated inputs for the Ascend DSpark draft.

    The object is valid only while ``step_epoch`` is current for the
    speculator that created it. Persistent block tables and query slot rows
    remain references to runner-owned buffers; context slots are snapshots,
    and the epoch guard prevents a later target step from reusing either kind
    of metadata as though it still described the current request batch.
    """

    step_epoch: int
    rank: int
    request_ids: tuple[str, ...]
    target_layer_ids: tuple[int, ...]
    num_reqs: int
    num_target_tokens: int
    num_query_tokens: int
    num_speculative_tokens: int

    request_state_indices: torch.Tensor
    target_input_ids: torch.Tensor
    target_positions: torch.Tensor
    target_query_start_loc: torch.Tensor
    target_sequence_lengths: torch.Tensor
    last_hidden_states: torch.Tensor
    auxiliary_hidden_states: tuple[torch.Tensor, ...]
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor

    anchor_token_ids: torch.Tensor
    draft_input_ids: torch.Tensor
    draft_positions: torch.Tensor
    draft_query_start_loc: torch.Tensor
    draft_sequence_lengths: torch.Tensor
    draft_is_prefilling: torch.Tensor

    draft_layer_group_ids: Mapping[str, int]
    draft_block_tables: Mapping[int, torch.Tensor]
    draft_context_slot_mappings: Mapping[str, torch.Tensor]
    draft_query_slot_mappings: Mapping[str, torch.Tensor]
    target_attn_metadata: Mapping[str, Any]

    temperature: torch.Tensor
    seeds: torch.Tensor
    num_tokens_across_dp: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class AscendDSparkDraftExecution:
    """A consumed proposal step whose context KV has been precomputed.

    Construction is intentionally restricted to the speculator's production
    phase helper. Once published, the associated proposal epoch cannot be
    prepared or executed again because context KV writes are not reversible
    without copying the complete cache backing.
    """

    proposal_inputs: AscendDSparkProposalInputs
    execution_token_count: int


__all__ = ["AscendDSparkDraftExecution", "AscendDSparkProposalInputs"]
