# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.gpu.attn_utils import init_attn_backend
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import (
    get_parallel_drafting_token_id,
)

from vllm_ascend.ascend_forward_context import build_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.spec_decode import dspark_runtime_not_wired
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.dspark.proposal_inputs import (
    AscendDSparkDraftExecution,
    AscendDSparkMarkovResult,
    AscendDSparkMarkovStep,
    AscendDSparkProposalInputs,
    AscendDSparkProposalLifecycle,
)

_DSPARK_MARKOV_FIXED_K = 5
_DSPARK_CONTINUE_AFTER_VERIFICATION = "dspark_continue_after_verification"
_DSPARK_PROFILE_PRESERVED_STATE = (
    "_proposal_step_epoch",
    "_prepared_step_epoch",
    "_context_kv_step_epoch",
    "_draft_forward_step_epoch",
    "_markov_attempt_step_epoch",
    "_markov_step_epoch",
    "_markov_result",
    "_published_proposal_step_epoch",
    "_published_proposal_request_ids",
    "_published_proposal_request_state_indices",
    "_published_proposal_owner_epochs",
    "_published_candidate_tokens",
    "_published_proposal_owners",
    "_active_published_proposal_owner_ids",
    "_active_proposal_reconciled",
    "_proposal_consumer_step_epoch",
    "_published_proposal_consumed",
    "_next_proposal_skipped",
    "_proposal_publication_count",
    "_proposal_consumption_count",
    "_next_proposal_skip_count",
    "_current_proposal_lifecycle",
    "_last_consumed_proposal_lifecycle",
    "_terminal_proposal_lifecycle",
    "_dropped_proposal_lifecycle",
    "_proposal_generated_count",
    "_proposal_returned_count",
    "_proposal_installed_count",
    "_proposal_dropped_count",
    "_terminal_proposal_discard_count",
)


@dataclass(frozen=True, slots=True)
class _PublishedProposalOwner:
    """One request-owned row retained until scheduler disposition."""

    request_id: str
    producer_epoch: int
    request_state_indices: torch.Tensor
    candidate_tokens: torch.Tensor
    publication_row: int
    published_length: int
    lifecycle: AscendDSparkProposalLifecycle


def _assert_markov_tensor_contract(predicate: torch.Tensor, message: str) -> None:
    """Validate device state without synchronizing the NPU hot path."""
    if predicate.device.type == "cpu":
        if not bool(predicate):
            raise ValueError(message)
        return
    torch._assert_async(predicate, message)


def _iter_cache_tensors(cache: Any):
    if isinstance(cache, torch.Tensor):
        yield cache
    elif isinstance(cache, dict):
        for value in cache.values():
            yield from _iter_cache_tensors(value)
    elif isinstance(cache, (list, tuple)):
        for value in cache:
            yield from _iter_cache_tensors(value)


def _tensor_byte_intervals(tensor: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Return exact occupied byte intervals for page-strided cache views."""
    if tensor.numel() == 0:
        return ()
    if any(stride < 0 for stride in tensor.stride()):
        raise RuntimeError("Ascend DSpark KV cache views must not use negative strides.")

    shape = tuple(tensor.shape)
    strides = tuple(tensor.stride())
    contiguous_span = 1
    suffix_start = len(shape)
    for dimension in range(len(shape) - 1, -1, -1):
        if strides[dimension] != contiguous_span:
            break
        contiguous_span *= shape[dimension]
        suffix_start = dimension

    prefix_shape = shape[:suffix_start]
    prefix_strides = strides[:suffix_start]
    interval_count = int(np.prod(prefix_shape, dtype=np.int64)) if prefix_shape else 1
    if interval_count > 1_000_000:
        raise RuntimeError(
            "Ascend DSpark cannot audit a KV cache view with more than one million discontiguous byte intervals."
        )

    element_size = tensor.element_size()
    storage_base = tensor.untyped_storage().data_ptr()
    storage_offset = tensor.storage_offset()
    intervals: list[tuple[int, int]] = []
    prefix_indices = product(*(range(size) for size in prefix_shape)) if prefix_shape else ((),)
    for indices in prefix_indices:
        element_offset = storage_offset + sum(index * stride for index, stride in zip(indices, prefix_strides))
        start = storage_base + element_offset * element_size
        intervals.append((start, start + contiguous_span * element_size))

    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _byte_intervals_overlap(
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
) -> bool:
    first_index = second_index = 0
    while first_index < len(first) and second_index < len(second):
        first_start, first_end = first[first_index]
        second_start, second_end = second[second_index]
        if max(first_start, second_start) < min(first_end, second_end):
            return True
        if first_end <= second_end:
            first_index += 1
        else:
            second_index += 1
    return False


class AscendDSparkSpeculator(BaseSpeculator):
    """Ascend V2 DSpark construction and runtime-boundary contract.

    The target model and the registered Ascend draft model are loaded before
    execution. Sparse-index metadata generation and proposal execution remain
    explicit fail-closed boundaries until their tensor plumbing is connected.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config is None:
            raise ValueError("Ascend DSpark V2 requires vllm_config.speculative_config.")
        method = getattr(speculative_config, "method", None)
        if method != "dspark":
            raise ValueError(f"AscendDSparkSpeculator requires method='dspark', got {method!r}.")

        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        if draft_model_config is None:
            raise ValueError("Ascend DSpark V2 requires speculative_config.draft_model_config.")
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        if draft_hf_config is None:
            raise ValueError("Ascend DSpark V2 requires draft_model_config.hf_config.")

        num_speculative_steps = getattr(speculative_config, "num_speculative_tokens", None)
        if (
            not isinstance(num_speculative_steps, int)
            or isinstance(num_speculative_steps, bool)
            or num_speculative_steps <= 0
        ):
            raise ValueError(
                "Ascend DSpark V2 requires num_speculative_tokens to be a "
                f"positive integer, got {num_speculative_steps!r}."
            )

        try:
            parallel_drafting_token_id = get_parallel_drafting_token_id(draft_hf_config)
        except ValueError as exc:
            raise ValueError("Ascend DSpark V2 requires a configured parallel-drafting token id.") from exc

        target_layer_ids = getattr(draft_hf_config, "dspark_target_layer_ids", None)
        if target_layer_ids is None:
            target_layer_ids = ()
        elif not isinstance(target_layer_ids, (list, tuple)) or len(target_layer_ids) != 3:
            raise ValueError("Ascend DeepSeek V4 DSpark requires exactly three dspark_target_layer_ids.")
        if any(type(layer_id) is not int for layer_id in target_layer_ids):
            raise TypeError("Ascend DSpark target layer IDs must be integers.")
        if tuple(sorted(set(target_layer_ids))) != tuple(target_layer_ids):
            raise ValueError(
                f"Ascend DSpark target layer IDs must be unique and strictly increasing, got {target_layer_ids}."
            )
        if target_layer_ids and target_layer_ids[0] < 0:
            raise ValueError("Ascend DSpark target layer IDs must be non-negative.")

        self.vllm_config = vllm_config
        self.device = device
        device_index = device.index if device is not None else None
        self.rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else (device_index or 0)
        )
        self.speculative_config = speculative_config
        self.draft_model_config = draft_model_config
        self.num_speculative_steps = num_speculative_steps
        self.parallel_drafting_token_id = parallel_drafting_token_id
        self.target_layer_ids = tuple(target_layer_ids)
        additional_config = getattr(vllm_config, "additional_config", None)
        continue_after_verification = True
        if isinstance(additional_config, Mapping):
            continue_after_verification = additional_config.get(
                _DSPARK_CONTINUE_AFTER_VERIFICATION,
                True,
            )
        if type(continue_after_verification) is not bool:
            raise TypeError(
                "additional_config.dspark_continue_after_verification must "
                f"be a bool, got {continue_after_verification!r}."
            )
        self.continue_after_verification = continue_after_verification
        self._model: torch.nn.Module | None = None
        self._loaded_target_model: torch.nn.Module | None = None
        self.target_attn_layer_names: frozenset[str] | None = None
        self.draft_attn_layer_names: frozenset[str] | None = None
        self.draft_attn_layer_order: tuple[str, ...] = ()
        self.draft_kv_cache_specs: MappingProxyType[str, KVCacheSpec] | None = None
        self.draft_kv_cache_group_ids: tuple[int, ...] = ()
        self.draft_kv_caches: MappingProxyType[str, Any] | None = None
        self.model_state: ModelState | None = None
        self.kv_cache_config: KVCacheConfig | None = None
        self.block_tables: BlockTables | None = None
        self.attn_groups: list[list[Any]] | None = None
        self.attn_backends: MappingProxyType[str, type[AttentionBackend]] | None = None
        self.draft_layer_group_ids: MappingProxyType[str, int] | None = None
        self._kv_cache_signature: tuple[Any, ...] | None = None
        self._proposal_step_epoch = 0
        self._prepared_step_epoch: int | None = None
        self._context_kv_step_epoch: int | None = None
        self._draft_forward_step_epoch: int | None = None
        self._markov_attempt_step_epoch: int | None = None
        self._markov_step_epoch: int | None = None
        self._markov_result: AscendDSparkMarkovResult | None = None
        self._markov_module_contract: MappingProxyType[str, Any] | None = None
        self._draft_cache_isolation_audit: MappingProxyType[str, int] | None = None
        self._published_proposal_step_epoch: int | None = None
        self._published_proposal_request_ids: tuple[str, ...] | None = None
        self._published_proposal_request_state_indices: torch.Tensor | None = None
        self._published_proposal_owner_epochs: tuple[int, ...] = ()
        self._published_candidate_tokens: torch.Tensor | None = None
        self._published_proposal_owners: dict[str, _PublishedProposalOwner] = {}
        self._active_published_proposal_owner_ids: tuple[str, ...] = ()
        self._active_proposal_reconciled = False
        self._proposal_consumer_step_epoch: int | None = None
        self._published_proposal_consumed = False
        self._next_proposal_skipped = False
        self._proposal_publication_count = 0
        self._proposal_consumption_count = 0
        self._next_proposal_skip_count = 0
        self._current_proposal_lifecycle: AscendDSparkProposalLifecycle | None = None
        self._last_consumed_proposal_lifecycle: AscendDSparkProposalLifecycle | None = None
        self._terminal_proposal_lifecycle: AscendDSparkProposalLifecycle | None = None
        self._dropped_proposal_lifecycle: AscendDSparkProposalLifecycle | None = None
        self._proposal_generated_count = 0
        self._proposal_returned_count = 0
        self._proposal_installed_count = 0
        self._proposal_dropped_count = 0
        self._terminal_proposal_discard_count = 0
        self.eplb_state: Any | None = None

        # These fields are consumed directly by GPUModelRunner's generic V2
        # speculator path before proposal execution.
        self.supports_mm_inputs = False
        self.draft_logits: torch.Tensor | None = None

        # The reference Ascend DSpark path is eager-only. Keep graph lifecycle
        # valid without claiming that proposal capture is implemented.
        self.requested_cudagraph_mode = CUDAGraphMode.NONE
        self.cudagraph_mode = CUDAGraphMode.NONE

    @property
    def model(self) -> torch.nn.Module:
        """Return the loaded Ascend draft model or fail before loading."""
        if self._model is None:
            dspark_runtime_not_wired("V2 draft-model loading")
        return self._model

    @property
    def draft_vllm_config(self) -> VllmConfig:
        """Use the loader's isolated eager config for draft attention/forward."""
        return self.model.vllm_config

    def load_model(self, target_model: torch.nn.Module) -> None:
        """Load the Ascend DSpark model once, then publish it atomically."""
        if self._model is not None:
            if target_model is self._loaded_target_model:
                return
            raise RuntimeError("Ascend DSpark draft model is already loaded for a different target model.")

        layer_type = cast(type[Any], AttentionLayerBase)
        target_attn_layer_names = frozenset(
            get_layers_from_vllm_config(
                self.vllm_config,
                layer_type,
            )
        )

        # This plugin-local helper performs config replacement, ModelRegistry
        # loading, PP validation, and target embedding/lm-head sharing without
        # importing the core CUDA/Triton DSpark runtime package.
        from vllm_ascend.worker.v2.spec_decode.dspark.model_loader import (
            load_dspark_model,
        )

        draft_model = load_dspark_model(target_model, self.vllm_config)

        from vllm_ascend.models.deepseek_v4_dspark import (
            DSparkDeepseekV4ForCausalLM,
        )

        if not isinstance(draft_model, DSparkDeepseekV4ForCausalLM):
            raise RuntimeError(
                "DSparkDraftModel resolved to a non-Ascend implementation: "
                f"{type(draft_model).__module__}.{type(draft_model).__name__}."
            )

        draft_model_config = draft_model.vllm_config.speculative_config.draft_model_config
        markov_module_contract = self._inspect_markov_modules(draft_model)

        get_draft_layer_names = getattr(
            draft_model,
            "get_draft_kv_cache_layer_names",
            None,
        )
        if not callable(get_draft_layer_names):
            raise RuntimeError("Ascend DSpark draft model must expose get_draft_kv_cache_layer_names().")
        draft_layer_names_list = list(get_draft_layer_names())
        if not draft_layer_names_list:
            raise RuntimeError("Ascend DSpark draft model registered no KV cache layers.")
        if len(draft_layer_names_list) != len(set(draft_layer_names_list)):
            raise RuntimeError("Ascend DSpark draft model returned duplicate KV cache layer names.")
        draft_attn_layer_names = frozenset(draft_layer_names_list)
        all_attn_layer_names = frozenset(
            get_layers_from_vllm_config(
                self.vllm_config,
                layer_type,
            )
        )
        missing_layers = draft_attn_layer_names - all_attn_layer_names
        if missing_layers:
            raise RuntimeError(
                "Ascend DSpark draft KV cache layers are absent from the V2 "
                f"static forward context: {sorted(missing_layers)}."
            )
        target_overlap = draft_attn_layer_names & target_attn_layer_names
        if target_overlap:
            raise RuntimeError(f"Ascend DSpark target and draft KV cache layers overlap: {sorted(target_overlap)}.")

        # Commit state only after construction, checkpoint loading, sharing,
        # and implementation validation all succeed.
        self._loaded_target_model = target_model
        self._model = draft_model
        self.draft_model_config = draft_model_config
        self._markov_module_contract = markov_module_contract
        self.target_attn_layer_names = target_attn_layer_names
        self.draft_attn_layer_names = draft_attn_layer_names
        self.draft_attn_layer_order = tuple(draft_layer_names_list)

    def validate_kv_cache_specs(
        self,
        kv_cache_specs: dict[str, KVCacheSpec],
    ) -> None:
        """Validate and publish the draft-only view of the global KV specs."""
        if self.draft_attn_layer_names is None:
            raise RuntimeError("Ascend DSpark draft model must be loaded before KV cache spec discovery.")
        missing = self.draft_attn_layer_names - kv_cache_specs.keys()
        if missing:
            raise RuntimeError(f"Ascend DSpark draft KV cache specs are missing for layers: {sorted(missing)}.")
        draft_specs = {layer_name: kv_cache_specs[layer_name] for layer_name in self.draft_attn_layer_names}
        self.draft_kv_cache_specs = MappingProxyType(draft_specs)

    def validate_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Check that every draft cache layer belongs to exactly one group."""
        if self.draft_attn_layer_names is None:
            raise RuntimeError("Ascend DSpark draft model must be loaded before KV cache initialization.")
        parallel_config = self.vllm_config.parallel_config
        if parallel_config.tensor_parallel_size != 8:
            raise ValueError("The initial Ascend DSpark KV cache lifecycle requires tensor parallel size 8.")
        if not parallel_config.enable_expert_parallel:
            raise ValueError("The initial Ascend DSpark KV cache lifecycle requires expert parallelism.")
        if parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("The initial Ascend DSpark KV cache lifecycle requires pipeline parallel size 1.")
        occurrences = {name: 0 for name in self.draft_attn_layer_names}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in self.draft_attn_layer_names.intersection(group.layer_names):
                occurrences[layer_name] += 1
        invalid = {name: count for name, count in occurrences.items() if count != 1}
        if invalid:
            raise RuntimeError(
                f"Each Ascend DSpark draft KV cache layer must occur in exactly one V2 KV cache group, got {invalid}."
            )

    @staticmethod
    def _cache_config_signature(
        kv_cache_config: KVCacheConfig,
    ) -> tuple[Any, ...]:
        return (
            kv_cache_config.num_blocks,
            tuple(
                (
                    tensor.size,
                    tuple(tensor.shared_by),
                    tensor.offset,
                    tensor.block_stride,
                )
                for tensor in kv_cache_config.kv_cache_tensors
            ),
            tuple((tuple(group.layer_names), repr(group.kv_cache_spec)) for group in kv_cache_config.kv_cache_groups),
        )

    def is_kv_cache_initialized_for(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> bool:
        """Return whether the same cache configuration is already installed."""
        return (
            self.kv_cache_config is not None
            and self.draft_kv_caches is not None
            and self._kv_cache_signature == self._cache_config_signature(kv_cache_config)
        )

    @staticmethod
    def _is_materialized_kv_cache(kv_cache: Any) -> bool:
        caches = kv_cache if isinstance(kv_cache, (list, tuple)) else (kv_cache,)
        return bool(caches) and all(isinstance(cache, torch.Tensor) and cache.numel() > 0 for cache in caches)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        """Install the real draft-only attention state after cache allocation.

        Core binds the target and draft tensors into static_forward_context
        before this hook. All validation and backend construction happen in
        locals so a failure never publishes a partially initialized DSpark
        lifecycle.
        """
        if self.draft_attn_layer_names is None:
            raise RuntimeError("Ascend DSpark draft model must be loaded before set_attn().")
        if self.is_kv_cache_initialized_for(kv_cache_config):
            return
        if self.kv_cache_config is not None:
            raise RuntimeError("Ascend DSpark KV cache is already initialized with a different configuration.")

        self.validate_kv_cache_config(kv_cache_config)
        attn_groups, _attn_cg_support, _kernel_block_sizes = init_attn_backend(
            kv_cache_config,
            self.draft_vllm_config,
            self.device,
            active_layer_names=set(self.draft_attn_layer_names),
        )
        active_group_ids = tuple(group_id for group_id, groups in enumerate(attn_groups) if groups)
        if not active_group_ids:
            raise RuntimeError("Ascend DSpark has no active draft KV cache groups.")

        layer_type = cast(type[Any], AttentionLayerBase)
        draft_layers = get_layers_from_vllm_config(
            self.vllm_config,
            layer_type,
            self.draft_attn_layer_names,
        )
        if set(draft_layers) != set(self.draft_attn_layer_names):
            raise RuntimeError("Ascend DSpark draft attention layers changed during KV cache initialization.")
        draft_kv_caches = {name: draft_layers[name].kv_cache for name in self.draft_attn_layer_names}
        unmaterialized = [
            name for name, kv_cache in draft_kv_caches.items() if not self._is_materialized_kv_cache(kv_cache)
        ]
        if unmaterialized:
            raise RuntimeError(
                f"Ascend DSpark draft KV cache tensors were not installed for layers: {sorted(unmaterialized)}."
            )

        attn_backends = {name: draft_layers[name].get_attn_backend() for name in self.draft_attn_layer_names}
        grouped_layer_names = {
            name for group_id in active_group_ids for group in attn_groups[group_id] for name in group.layer_names
        }
        if grouped_layer_names != set(self.draft_attn_layer_names):
            raise RuntimeError(
                "Ascend DSpark draft attention groups do not match the model's "
                f"draft KV cache layers: {sorted(grouped_layer_names)}."
            )

        layer_group_ids: dict[str, int] = {}
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in self.draft_attn_layer_names.intersection(group.layer_names):
                if layer_name in layer_group_ids:
                    raise RuntimeError(f"Ascend DSpark draft layer belongs to multiple KV cache groups: {layer_name}.")
                layer_group_ids[layer_name] = group_id
        if set(layer_group_ids) != set(self.draft_attn_layer_names):
            raise RuntimeError("Ascend DSpark draft KV cache group mapping is incomplete.")

        # Publish only after every layer, tensor, group and backend is valid.
        draft_kv_cache_view = MappingProxyType(draft_kv_caches)
        attn_backend_view = MappingProxyType(attn_backends)
        cache_signature = self._cache_config_signature(kv_cache_config)
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = attn_groups
        self.attn_backends = attn_backend_view
        self.draft_layer_group_ids = MappingProxyType(layer_group_ids)
        self.draft_kv_cache_group_ids = active_group_ids
        self.draft_kv_caches = draft_kv_cache_view
        self._kv_cache_signature = cache_signature
        self._draft_cache_isolation_audit = None

    def _validate_step_tensor(
        self,
        name: str,
        tensor: Any,
        *,
        ndim: int | None = None,
        dtypes: tuple[torch.dtype, ...] | None = None,
        min_size: int | None = None,
    ) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Ascend DSpark {name} must be a torch.Tensor.")
        if tensor.device.type != self.device.type or (
            self.device.index is not None and tensor.device.index != self.device.index
        ):
            raise RuntimeError(
                f"Ascend DSpark {name} belongs to {tensor.device}, expected the current rank device {self.device}."
            )
        if ndim is not None and tensor.ndim != ndim:
            raise ValueError(f"Ascend DSpark {name} must have {ndim} dimensions, got shape {tuple(tensor.shape)}.")
        if dtypes is not None and tensor.dtype not in dtypes:
            raise TypeError(f"Ascend DSpark {name} must use one of {dtypes}, got {tensor.dtype}.")
        if min_size is not None and tensor.shape[0] < min_size:
            raise ValueError(f"Ascend DSpark {name} has {tensor.shape[0]} rows, expected at least {min_size}.")
        return tensor

    def _validate_aux_hidden_states(
        self,
        aux_hidden_states: list[torch.Tensor] | None,
        last_hidden_states: torch.Tensor,
        num_tokens_after_padding: int,
    ) -> tuple[torch.Tensor, ...]:
        if len(self.target_layer_ids) != 3:
            raise RuntimeError("Ascend DSpark proposal preparation requires exactly three configured target layer IDs.")
        if aux_hidden_states is None:
            raise RuntimeError("Ascend DSpark target auxiliary hidden states are missing.")
        if len(aux_hidden_states) != len(self.target_layer_ids):
            raise ValueError(
                "Ascend DSpark requires one auxiliary hidden state for each "
                f"target layer {self.target_layer_ids}, got {len(aux_hidden_states)}."
            )

        target_backbone = getattr(self._loaded_target_model, "model", None)
        configured_boundaries = getattr(target_backbone, "aux_hidden_state_layers", None)
        expected_boundaries = tuple(layer_id + 1 for layer_id in self.target_layer_ids)
        if configured_boundaries != expected_boundaries:
            raise RuntimeError(
                "Ascend DSpark target auxiliary layer ownership changed: "
                f"expected decoder output boundaries {expected_boundaries}, got "
                f"{configured_boundaries}."
            )

        expected_shape: tuple[int, ...] | None = None
        expected_dtype: torch.dtype | None = None
        validated: list[torch.Tensor] = []
        for layer_id, hidden_states in zip(self.target_layer_ids, aux_hidden_states):
            hidden_states = self._validate_step_tensor(
                f"auxiliary hidden state for target layer {layer_id}",
                hidden_states,
                ndim=2,
            )
            if hidden_states.shape[0] != num_tokens_after_padding:
                raise ValueError(
                    "Ascend DSpark auxiliary hidden-state token dimension must "
                    f"equal {num_tokens_after_padding}, got "
                    f"{hidden_states.shape[0]} for layer {layer_id}."
                )
            if not hidden_states.dtype.is_floating_point:
                raise TypeError(
                    "Ascend DSpark auxiliary hidden states must be floating "
                    f"point, got {hidden_states.dtype} for layer {layer_id}."
                )
            if expected_shape is None:
                expected_shape = tuple(hidden_states.shape)
                expected_dtype = hidden_states.dtype
            elif tuple(hidden_states.shape) != expected_shape or hidden_states.dtype != expected_dtype:
                raise ValueError(
                    "Ascend DSpark auxiliary hidden states must have identical shape and dtype in target-layer order."
                )
            validated.append(hidden_states)

        if last_hidden_states.shape[0] != num_tokens_after_padding:
            raise ValueError(
                "Ascend DSpark last_hidden_states token dimension must equal "
                f"{num_tokens_after_padding}, got {last_hidden_states.shape[0]}."
            )
        if expected_dtype is not None and last_hidden_states.dtype != expected_dtype:
            raise TypeError(
                "Ascend DSpark target and auxiliary hidden states must use the "
                f"same dtype, got {last_hidden_states.dtype} and {expected_dtype}."
            )
        return tuple(validated)

    def _build_query_slot_mappings(
        self,
        draft_positions: torch.Tensor,
        num_reqs: int,
    ) -> tuple[
        MappingProxyType[str, int],
        MappingProxyType[int, torch.Tensor],
        MappingProxyType[str, torch.Tensor],
    ]:
        if self.block_tables is None or self.draft_layer_group_ids is None:
            raise RuntimeError("Ascend DSpark KV cache must be initialized before proposal input preparation.")
        if not self.draft_attn_layer_order:
            raise RuntimeError("Ascend DSpark draft attention layer order is unavailable.")

        num_query_tokens = draft_positions.shape[0]
        shared_slot_mappings = self._validate_step_tensor(
            "shared KV slot-mapping buffer",
            self.block_tables.slot_mappings,
            ndim=2,
            dtypes=(torch.int32,),
        )
        query_positions_2d = draft_positions.view(num_reqs, self.num_speculative_steps)
        computed_slots_by_group: dict[int, torch.Tensor] = {}
        block_tables_by_group: dict[int, torch.Tensor] = {}
        for layer_name in self.draft_attn_layer_order:
            group_id = self.draft_layer_group_ids[layer_name]
            if group_id >= shared_slot_mappings.shape[0]:
                raise RuntimeError(f"Ascend DSpark KV group {group_id} has no shared slot-mapping row.")
            if shared_slot_mappings.shape[1] < num_query_tokens:
                raise RuntimeError(
                    f"Ascend DSpark shared slot-mapping buffer is too small for {num_query_tokens} draft query tokens."
                )
            if group_id not in computed_slots_by_group:
                block_table = self.block_tables.input_block_tables[group_id]
                self._validate_step_tensor(
                    f"block table for KV group {group_id}",
                    block_table,
                    ndim=2,
                    dtypes=(torch.int32,),
                    min_size=num_reqs,
                )
                block_size = self.block_tables.kernel_block_sizes[group_id]
                if block_size <= 0 or block_table.shape[1] == 0:
                    raise RuntimeError(f"Ascend DSpark KV group {group_id} has an invalid block-table layout.")
                logical_blocks = torch.div(
                    query_positions_2d,
                    block_size,
                    rounding_mode="floor",
                ).clamp(max=block_table.shape[1] - 1)
                physical_blocks = torch.gather(
                    block_table[:num_reqs],
                    1,
                    logical_blocks.to(torch.int64),
                )
                query_slots = (
                    physical_blocks.to(torch.int64) * block_size + torch.remainder(query_positions_2d, block_size)
                ).to(torch.int32)
                computed_slots_by_group[group_id] = query_slots.reshape(num_query_tokens)
                block_tables_by_group[group_id] = block_table

        # Attention metadata reads these stable rows by address. Compute every
        # group first, then publish with rollback so a failed copy cannot leave
        # a partially updated DSpark step.
        original_slots = {
            group_id: shared_slot_mappings[group_id, :num_query_tokens].clone() for group_id in computed_slots_by_group
        }
        published_groups: list[int] = []
        try:
            for group_id, query_slots in computed_slots_by_group.items():
                shared_slot_mappings[group_id, :num_query_tokens].copy_(query_slots)
                published_groups.append(group_id)
        except BaseException:
            for group_id in published_groups:
                shared_slot_mappings[group_id, :num_query_tokens].copy_(original_slots[group_id])
            raise

        slot_mappings_by_layer = {
            layer_name: shared_slot_mappings[self.draft_layer_group_ids[layer_name], :num_query_tokens]
            for layer_name in self.draft_attn_layer_order
        }

        return (
            MappingProxyType(dict(self.draft_layer_group_ids)),
            MappingProxyType(block_tables_by_group),
            MappingProxyType(slot_mappings_by_layer),
        )

    def prepare_proposal_inputs(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
    ) -> AscendDSparkProposalInputs:
        """Build one real target step's DSpark inputs without running the draft."""
        if self._model is None:
            raise RuntimeError("Ascend DSpark draft model must be loaded before proposal input preparation.")
        if self.block_tables is None or self.kv_cache_config is None:
            raise RuntimeError("Ascend DSpark KV cache must be initialized before proposal input preparation.")
        if not isinstance(input_batch, InputBatch):
            raise TypeError("Ascend DSpark proposal preparation requires a V2 InputBatch.")
        if not isinstance(attn_metadata, dict):
            raise TypeError("Ascend DSpark target attention metadata must be a dictionary.")
        if not isinstance(slot_mappings, dict):
            raise TypeError("Ascend DSpark target slot mappings must be a dictionary.")
        if self._published_candidate_tokens is not None:
            raise RuntimeError(
                "Ascend DSpark must consume the active published proposal through "
                "the core verification lifecycle before preparing another target step."
            )

        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_tokens_after_padding = input_batch.num_tokens_after_padding
        if num_reqs <= 0 or num_target_tokens <= 0:
            raise ValueError("Ascend DSpark proposal preparation requires a non-empty target step.")
        if num_target_tokens > num_tokens_after_padding:
            raise ValueError("Ascend DSpark target token count exceeds the padded target shape.")
        # A real target step invalidates all previously prepared inputs before
        # validating this step. If validation fails, no older request/hidden
        # state can be reused and no new step is published.
        step_epoch = self._proposal_step_epoch + 1
        self._proposal_step_epoch = step_epoch
        self._prepared_step_epoch = None
        self._markov_attempt_step_epoch = None
        self._markov_step_epoch = None
        self._markov_result = None

        if len(input_batch.req_ids) != num_reqs or len(set(input_batch.req_ids)) != num_reqs:
            raise ValueError("Ascend DSpark request IDs must be unique and match the active request count.")
        if input_batch.idx_mapping_np.shape != (num_reqs,) or (input_batch.idx_mapping_np < 0).any():
            raise ValueError("Ascend DSpark request-state ownership must contain one non-negative index per request.")

        last_hidden_states = self._validate_step_tensor(
            "last_hidden_states",
            last_hidden_states,
            ndim=2,
        )
        auxiliary_hidden_states = self._validate_aux_hidden_states(
            aux_hidden_states,
            last_hidden_states,
            num_tokens_after_padding,
        )
        request_state_indices = self._validate_step_tensor(
            "request state indices",
            input_batch.idx_mapping,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_reqs,
        )
        target_input_ids = self._validate_step_tensor(
            "target input IDs",
            input_batch.input_ids,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_tokens_after_padding,
        )
        target_positions = self._validate_step_tensor(
            "target positions",
            input_batch.positions,
            ndim=1,
            dtypes=(torch.int64,),
            min_size=num_tokens_after_padding,
        )
        target_query_start_loc = self._validate_step_tensor(
            "target query-start locations",
            input_batch.query_start_loc,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_reqs + 1,
        )
        target_sequence_lengths = self._validate_step_tensor(
            "target sequence lengths",
            input_batch.seq_lens,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_reqs,
        )
        num_sampled = self._validate_step_tensor(
            "sampled-token counts",
            num_sampled,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_reqs,
        )
        num_rejected = self._validate_step_tensor(
            "rejected-token counts",
            num_rejected,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_reqs,
        )
        max_request_index = int(input_batch.idx_mapping_np.max()) + 1
        last_sampled = self._validate_step_tensor(
            "last sampled token IDs",
            last_sampled,
            dtypes=(torch.int64,),
            min_size=max_request_index,
        )
        next_prefill_tokens = self._validate_step_tensor(
            "next prefill token IDs",
            next_prefill_tokens,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=max_request_index,
        )
        temperature = self._validate_step_tensor(
            "sampling temperatures",
            temperature,
            ndim=1,
            min_size=max_request_index,
        )
        seeds = self._validate_step_tensor(
            "sampling seeds",
            seeds,
            ndim=1,
            dtypes=(torch.int64,),
            min_size=max_request_index,
        )
        if num_tokens_across_dp is not None:
            self._validate_step_tensor("DP token counts", num_tokens_across_dp)

        req_indices = request_state_indices[:num_reqs].to(torch.int64)
        last_sampled_ids = last_sampled[req_indices]
        if last_sampled_ids.ndim == 2:
            if last_sampled_ids.shape[1] != 1:
                raise ValueError("Ascend DSpark last sampled token IDs must have one token per request.")
            last_sampled_ids = last_sampled_ids[:, 0]
        elif last_sampled_ids.ndim != 1:
            raise ValueError("Ascend DSpark last sampled token IDs have an invalid shape.")
        next_prefill_ids = next_prefill_tokens[req_indices]
        anchor_token_ids = torch.where(
            num_sampled[:num_reqs] > 0,
            last_sampled_ids.to(torch.int32),
            next_prefill_ids,
        )

        valid_query_ends = target_query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        last_valid_positions = target_positions[valid_query_ends.to(torch.int64) - 1]
        query_offsets = torch.arange(
            self.num_speculative_steps,
            dtype=target_positions.dtype,
            device=self.device,
        )
        raw_draft_positions = last_valid_positions[:, None] + 1 + query_offsets
        max_model_len = int(self.vllm_config.model_config.max_model_len)
        draft_positions = raw_draft_positions.clamp(max=max_model_len - 1).reshape(-1)
        num_query_tokens = num_reqs * self.num_speculative_steps
        draft_input_ids = torch.full(
            (num_reqs, self.num_speculative_steps),
            self.parallel_drafting_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        draft_input_ids[:, 0] = anchor_token_ids
        draft_input_ids = draft_input_ids.reshape(num_query_tokens)
        draft_query_start_loc = (
            torch.arange(
                num_reqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            * self.num_speculative_steps
        )
        draft_sequence_lengths = (last_valid_positions + 1 + self.num_speculative_steps).to(torch.int32)

        context_slots_by_group: dict[int, torch.Tensor] = {}
        context_slots_by_layer: dict[str, torch.Tensor] = {}
        context_sources_by_group: dict[int, torch.Tensor] = {}
        for layer_name in self.draft_attn_layer_order:
            if layer_name not in slot_mappings:
                raise RuntimeError(
                    f"Ascend DSpark target step did not provide a slot mapping for draft layer {layer_name}."
                )
            if self.draft_layer_group_ids is None:
                raise RuntimeError("Ascend DSpark draft KV group mapping is unavailable.")
            group_id = self.draft_layer_group_ids[layer_name]
            source_slots = self._validate_step_tensor(
                f"target context slot mapping for {layer_name}",
                slot_mappings[layer_name],
                ndim=1,
                dtypes=(torch.int32,),
                min_size=num_target_tokens,
            )
            if group_id not in context_slots_by_group:
                context_slots_by_group[group_id] = source_slots[:num_target_tokens].clone()
                context_sources_by_group[group_id] = source_slots
            else:
                first_source = context_sources_by_group[group_id]
                if (
                    source_slots.data_ptr() != first_source.data_ptr()
                    or source_slots.shape != first_source.shape
                    or source_slots.stride() != first_source.stride()
                ):
                    raise RuntimeError(
                        "Ascend DSpark layers in the same KV group received "
                        f"different target slot-mapping buffers for group {group_id}."
                    )
            context_slots_by_layer[layer_name] = context_slots_by_group[group_id]

        layer_group_ids, draft_block_tables, draft_query_slots = self._build_query_slot_mappings(
            raw_draft_positions.reshape(-1),
            num_reqs,
        )

        proposal_inputs = AscendDSparkProposalInputs(
            step_epoch=step_epoch,
            rank=self.rank,
            request_ids=tuple(input_batch.req_ids),
            target_layer_ids=self.target_layer_ids,
            num_reqs=num_reqs,
            num_target_tokens=num_target_tokens,
            num_query_tokens=num_query_tokens,
            num_speculative_tokens=self.num_speculative_steps,
            request_state_indices=request_state_indices,
            target_input_ids=target_input_ids,
            target_positions=target_positions,
            target_query_start_loc=target_query_start_loc,
            target_sequence_lengths=target_sequence_lengths,
            last_hidden_states=last_hidden_states,
            auxiliary_hidden_states=auxiliary_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            anchor_token_ids=anchor_token_ids,
            draft_input_ids=draft_input_ids,
            draft_positions=draft_positions,
            draft_query_start_loc=draft_query_start_loc,
            draft_sequence_lengths=draft_sequence_lengths,
            draft_is_prefilling=torch.from_numpy(input_batch.is_prefilling_np.copy()),
            draft_layer_group_ids=layer_group_ids,
            draft_block_tables=draft_block_tables,
            draft_context_slot_mappings=MappingProxyType(context_slots_by_layer),
            draft_query_slot_mappings=draft_query_slots,
            target_attn_metadata=MappingProxyType(attn_metadata),
            temperature=temperature,
            seeds=seeds,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        self._prepared_step_epoch = step_epoch
        return proposal_inputs

    def validate_prepared_inputs_current(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> None:
        if (
            proposal_inputs.step_epoch != self._proposal_step_epoch
            or proposal_inputs.step_epoch != self._prepared_step_epoch
        ):
            raise RuntimeError(
                "Ascend DSpark proposal inputs are stale; prepare inputs again for the current target step."
            )
        if proposal_inputs.rank != self.rank:
            raise RuntimeError("Ascend DSpark proposal inputs belong to a different NPU rank.")

    def _validate_draft_backbone_inputs(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> None:
        self.validate_prepared_inputs_current(proposal_inputs)
        if (
            not proposal_inputs.request_ids
            or len(proposal_inputs.request_ids) != proposal_inputs.num_reqs
            or any(not request_id for request_id in proposal_inputs.request_ids)
        ):
            raise ValueError("Ascend DSpark draft execution requires one non-empty request ID per request.")
        if len(set(proposal_inputs.request_ids)) != proposal_inputs.num_reqs:
            raise ValueError("Ascend DSpark draft execution request IDs must remain unique.")
        if proposal_inputs.target_layer_ids != self.target_layer_ids:
            raise RuntimeError("Ascend DSpark auxiliary target-layer order changed after proposal preparation.")
        if proposal_inputs.num_speculative_tokens != self.num_speculative_steps:
            raise RuntimeError("Ascend DSpark speculative-token count changed after proposal preparation.")
        expected_query_tokens = proposal_inputs.num_reqs * proposal_inputs.num_speculative_tokens
        if proposal_inputs.num_query_tokens != expected_query_tokens:
            raise ValueError(
                "Ascend DSpark logical draft-token count must equal request count times speculative-token count."
            )
        if len(proposal_inputs.auxiliary_hidden_states) != len(self.target_layer_ids):
            raise ValueError("Ascend DSpark draft execution requires exactly three auxiliary hidden states.")

        draft_hidden_size = int(self.draft_model_config.hf_config.hidden_size)
        auxiliary_dtype: torch.dtype | None = None
        for layer_id, hidden_states in zip(
            proposal_inputs.target_layer_ids,
            proposal_inputs.auxiliary_hidden_states,
        ):
            self._validate_step_tensor(
                f"draft auxiliary hidden state for target layer {layer_id}",
                hidden_states,
                ndim=2,
            )
            if not hidden_states.dtype.is_floating_point:
                raise TypeError(
                    f"Ascend DSpark draft auxiliary hidden states must be floating point, got {hidden_states.dtype}."
                )
            if auxiliary_dtype is None:
                auxiliary_dtype = hidden_states.dtype
            elif hidden_states.dtype != auxiliary_dtype:
                raise TypeError("Ascend DSpark draft auxiliary hidden states must have one common dtype.")
            if (
                hidden_states.shape[0] < proposal_inputs.num_target_tokens
                or hidden_states.shape[1] != draft_hidden_size
            ):
                raise ValueError(
                    "Ascend DSpark auxiliary hidden-state shape must match the "
                    "valid target context prefix and draft hidden size, got "
                    f"{tuple(hidden_states.shape)} for at least "
                    f"{proposal_inputs.num_target_tokens} rows and hidden size {draft_hidden_size}."
                )

        tensor_contracts = (
            ("target positions", proposal_inputs.target_positions, 1, (torch.int64,)),
            ("draft input IDs", proposal_inputs.draft_input_ids, 1, (torch.int32,)),
            ("draft positions", proposal_inputs.draft_positions, 1, (torch.int64,)),
            ("draft query-start locations", proposal_inputs.draft_query_start_loc, 1, (torch.int32,)),
            ("draft sequence lengths", proposal_inputs.draft_sequence_lengths, 1, (torch.int32,)),
        )
        for name, tensor, ndim, dtypes in tensor_contracts:
            self._validate_step_tensor(name, tensor, ndim=ndim, dtypes=dtypes)
        if proposal_inputs.draft_input_ids.shape[0] != expected_query_tokens:
            raise ValueError("Ascend DSpark draft input IDs do not match the logical draft-token count.")
        if proposal_inputs.draft_positions.shape != proposal_inputs.draft_input_ids.shape:
            raise ValueError("Ascend DSpark draft positions must align one-to-one with draft input IDs.")
        if proposal_inputs.draft_query_start_loc.shape[0] != proposal_inputs.num_reqs + 1:
            raise ValueError("Ascend DSpark draft query-start locations must contain B+1 entries.")
        if proposal_inputs.draft_sequence_lengths.shape[0] != proposal_inputs.num_reqs:
            raise ValueError("Ascend DSpark draft sequence lengths must contain one entry per request.")
        if (
            not isinstance(proposal_inputs.draft_is_prefilling, torch.Tensor)
            or proposal_inputs.draft_is_prefilling.device.type != "cpu"
            or proposal_inputs.draft_is_prefilling.dtype != torch.bool
            or proposal_inputs.draft_is_prefilling.shape != (proposal_inputs.num_reqs,)
        ):
            raise ValueError(
                "Ascend DSpark draft prefill ownership must be a CPU bool tensor with one row per request."
            )
        if proposal_inputs.target_positions.shape[0] < proposal_inputs.num_target_tokens:
            raise ValueError("Ascend DSpark target positions do not cover the valid context-token prefix.")

        if self.draft_kv_caches is None or self.draft_layer_group_ids is None:
            raise RuntimeError("Ascend DSpark draft KV cache must be installed before draft execution.")
        if set(proposal_inputs.draft_layer_group_ids) != set(self.draft_attn_layer_order):
            raise RuntimeError("Ascend DSpark draft layer/group mapping changed after proposal preparation.")
        for mapping_name, mapping in (
            ("context slots", proposal_inputs.draft_context_slot_mappings),
            ("query slots", proposal_inputs.draft_query_slot_mappings),
        ):
            if set(mapping) != set(self.draft_attn_layer_order):
                raise RuntimeError(f"Ascend DSpark draft {mapping_name} mapping is incomplete.")
        if set(self.draft_kv_caches) != set(self.draft_attn_layer_order):
            raise RuntimeError("Ascend DSpark draft cache registry is incomplete.")
        for layer_name in self.draft_attn_layer_order:
            if not layer_name.startswith("mtp."):
                raise RuntimeError(f"Ascend DSpark draft cache escaped the MTP namespace: {layer_name}.")
            if proposal_inputs.draft_layer_group_ids[layer_name] != self.draft_layer_group_ids[layer_name]:
                raise RuntimeError(f"Ascend DSpark KV group changed for draft layer {layer_name}.")
            if not self._is_materialized_kv_cache(self.draft_kv_caches[layer_name]):
                raise RuntimeError(f"Ascend DSpark draft KV cache is not materialized for {layer_name}.")
            group_id = self.draft_layer_group_ids[layer_name]
            if group_id not in proposal_inputs.draft_block_tables:
                raise RuntimeError(f"Ascend DSpark draft block table is missing for KV group {group_id}.")
            self._validate_step_tensor(
                f"draft block table for {layer_name}",
                proposal_inputs.draft_block_tables[group_id],
                ndim=2,
                dtypes=(torch.int32,),
                min_size=proposal_inputs.num_reqs,
            )
            self._validate_step_tensor(
                f"draft context slots for {layer_name}",
                proposal_inputs.draft_context_slot_mappings[layer_name],
                ndim=1,
                dtypes=(torch.int32,),
                min_size=proposal_inputs.num_target_tokens,
            )
            query_slots = self._validate_step_tensor(
                f"draft query slots for {layer_name}",
                proposal_inputs.draft_query_slot_mappings[layer_name],
                ndim=1,
                dtypes=(torch.int32,),
                min_size=proposal_inputs.num_query_tokens,
            )
            if query_slots.shape[0] != proposal_inputs.num_query_tokens:
                raise ValueError(f"Ascend DSpark query-slot count changed for draft layer {layer_name}.")

    def audit_target_draft_cache_isolation(self) -> dict[str, int]:
        """Compare stable target/draft cache objects and their occupied bytes."""
        if self._draft_cache_isolation_audit is not None:
            return dict(self._draft_cache_isolation_audit)
        if self.target_attn_layer_names is None or self.draft_kv_caches is None:
            raise RuntimeError("Ascend DSpark target and draft caches must be installed before alias auditing.")
        layer_type = cast(type[Any], AttentionLayerBase)
        target_layers = get_layers_from_vllm_config(
            self.vllm_config,
            layer_type,
            self.target_attn_layer_names,
        )
        target_tensors = {
            id(tensor): tensor for layer in target_layers.values() for tensor in _iter_cache_tensors(layer.kv_cache)
        }
        draft_tensors = {
            id(tensor): tensor for cache in self.draft_kv_caches.values() for tensor in _iter_cache_tensors(cache)
        }
        object_alias_count = len(target_tensors.keys() & draft_tensors.keys())
        target_storage_bases = {tensor.untyped_storage().data_ptr() for tensor in target_tensors.values()}
        draft_storage_bases = {tensor.untyped_storage().data_ptr() for tensor in draft_tensors.values()}
        shared_backing_base_count = len(target_storage_bases & draft_storage_bases)

        target_intervals = {tensor_id: _tensor_byte_intervals(tensor) for tensor_id, tensor in target_tensors.items()}
        draft_intervals = {tensor_id: _tensor_byte_intervals(tensor) for tensor_id, tensor in draft_tensors.items()}
        byte_range_overlap_count = sum(
            _byte_intervals_overlap(target_range, draft_range)
            for target_range in target_intervals.values()
            for draft_range in draft_intervals.values()
        )
        audit = {
            "target_cache_object_alias_count": object_alias_count,
            "target_cache_byte_range_overlap_count": byte_range_overlap_count,
            "shared_backing_base_count": shared_backing_base_count,
        }
        self._draft_cache_isolation_audit = MappingProxyType(audit)
        return dict(audit)

    @torch.inference_mode()
    def _combine_and_precompute_draft_context(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> AscendDSparkDraftExecution:
        """Consume one proposal epoch and write its real context into draft KV."""
        self._validate_draft_backbone_inputs(proposal_inputs)
        if self._context_kv_step_epoch == proposal_inputs.step_epoch:
            raise RuntimeError("Ascend DSpark proposal inputs were already consumed by context-KV precompute.")

        cache_audit = self.audit_target_draft_cache_isolation()
        if cache_audit["target_cache_object_alias_count"]:
            raise RuntimeError("Ascend DSpark target and draft KV caches alias the same tensor object.")
        if cache_audit["target_cache_byte_range_overlap_count"]:
            raise RuntimeError("Ascend DSpark target and draft KV caches overlap in occupied byte ranges.")

        # Mark the proposal consumed before the first irreversible KV write.
        # A failure keeps the epoch invalid; the next target step must prepare
        # fresh metadata instead of replaying a partially written cache step.
        self._context_kv_step_epoch = proposal_inputs.step_epoch
        self._prepared_step_epoch = None

        auxiliary_states = tuple(
            hidden_states[: proposal_inputs.num_target_tokens]
            for hidden_states in proposal_inputs.auxiliary_hidden_states
        )
        concatenated_aux = torch.cat(auxiliary_states, dim=-1)
        context_states = self.model.combine_hidden_states(concatenated_aux)
        self._validate_step_tensor("combined draft context states", context_states, ndim=2)
        if context_states.dtype != auxiliary_states[0].dtype:
            raise RuntimeError(
                "Ascend DSpark combined context states must preserve the "
                f"target auxiliary dtype {auxiliary_states[0].dtype}, got {context_states.dtype}."
            )
        if context_states.shape != (
            proposal_inputs.num_target_tokens,
            int(self.draft_model_config.hf_config.hidden_size),
        ):
            raise RuntimeError(
                "Ascend DSpark combined context-state shape does not match the "
                f"draft model ABI: {tuple(context_states.shape)}."
            )

        context_slot_mappings = [
            proposal_inputs.draft_context_slot_mappings[layer_name] for layer_name in self.draft_attn_layer_order
        ]
        self.model.precompute_and_store_context_kv(
            context_states,
            proposal_inputs.target_positions[: proposal_inputs.num_target_tokens],
            context_slot_mappings,
        )
        return AscendDSparkDraftExecution(
            proposal_inputs=proposal_inputs,
            execution_token_count=proposal_inputs.draft_input_ids.shape[0],
        )

    def _validate_draft_execution_current(
        self,
        execution: AscendDSparkDraftExecution,
    ) -> AscendDSparkProposalInputs:
        proposal_inputs = execution.proposal_inputs
        if proposal_inputs.step_epoch != self._proposal_step_epoch:
            raise RuntimeError("Ascend DSpark draft execution belongs to a stale target step.")
        if proposal_inputs.rank != self.rank:
            raise RuntimeError("Ascend DSpark draft execution belongs to a different NPU rank.")
        if self._context_kv_step_epoch != proposal_inputs.step_epoch:
            raise RuntimeError("Ascend DSpark draft context KV was not precomputed for this execution.")
        if execution.execution_token_count != proposal_inputs.num_query_tokens:
            raise RuntimeError("Ascend DSpark eager execution token count differs from its logical token count.")
        return proposal_inputs

    def _build_draft_forward_metadata(
        self,
        execution: AscendDSparkDraftExecution,
    ) -> dict[str, Any]:
        proposal_inputs = self._validate_draft_execution_current(execution)
        if self.attn_groups is None or self.block_tables is None or self.kv_cache_config is None:
            raise RuntimeError("Ascend DSpark attention state is unavailable for draft metadata construction.")

        block_tables: list[torch.Tensor] = []
        for group_id in range(len(self.kv_cache_config.kv_cache_groups)):
            block_table = self.block_tables.input_block_tables[group_id]
            if group_id in proposal_inputs.draft_block_tables:
                if proposal_inputs.draft_block_tables[group_id] is not block_table:
                    raise RuntimeError(f"Ascend DSpark draft block table identity changed for KV group {group_id}.")
                block_table = proposal_inputs.draft_block_tables[group_id]
            block_tables.append(block_table)

        query_slots = self.block_tables.slot_mappings[:, : execution.execution_token_count]
        for layer_name in self.draft_attn_layer_order:
            group_id = proposal_inputs.draft_layer_group_ids[layer_name]
            layer_slots = proposal_inputs.draft_query_slot_mappings[layer_name]
            group_slots = query_slots[group_id]
            if (
                layer_slots.data_ptr() != group_slots.data_ptr()
                or layer_slots.shape != group_slots.shape
                or layer_slots.stride() != group_slots.stride()
            ):
                raise RuntimeError(f"Ascend DSpark draft query slots changed for layer {layer_name}.")

        query_start_loc_cpu = (
            torch.arange(proposal_inputs.num_reqs + 1, dtype=torch.int32) * proposal_inputs.num_speculative_tokens
        )
        sequence_lengths_np = proposal_inputs.draft_sequence_lengths.detach().cpu().numpy()
        draft_attn_state = (
            AscendAttentionState.ChunkedPrefill
            if bool(proposal_inputs.draft_is_prefilling.any())
            else AscendAttentionState.DecodeOnly
        )
        draft_attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=proposal_inputs.num_reqs,
            num_tokens=execution.execution_token_count,
            num_actual_tokens=proposal_inputs.num_query_tokens,
            num_input_tokens=execution.execution_token_count,
            query_start_loc_gpu=proposal_inputs.draft_query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=proposal_inputs.num_speculative_tokens,
            seq_lens=proposal_inputs.draft_sequence_lengths,
            max_seq_len=int(sequence_lengths_np.max()),
            block_tables=block_tables,
            slot_mappings=query_slots,
            kv_cache_config=self.kv_cache_config,
            seq_lens_np=sequence_lengths_np,
            positions=proposal_inputs.draft_positions,
            is_prefilling=proposal_inputs.draft_is_prefilling,
            attn_state=draft_attn_state,
            causal=False,
        )
        if set(draft_attn_metadata) != set(self.draft_attn_layer_order):
            raise RuntimeError(
                "Ascend DSpark draft attention metadata does not match the "
                f"runtime layers: {tuple(draft_attn_metadata)}."
            )
        return {layer_name: draft_attn_metadata[layer_name] for layer_name in self.draft_attn_layer_order}

    @torch.inference_mode()
    def _run_draft_model_forward(
        self,
        execution: AscendDSparkDraftExecution,
        draft_attn_metadata: dict[str, Any],
    ) -> torch.Tensor:
        proposal_inputs = self._validate_draft_execution_current(execution)
        if self._draft_forward_step_epoch == proposal_inputs.step_epoch:
            raise RuntimeError("Ascend DSpark proposal inputs were already consumed by draft forward.")
        if tuple(draft_attn_metadata) != self.draft_attn_layer_order:
            raise RuntimeError("Ascend DSpark draft forward received incomplete attention metadata.")
        self._draft_forward_step_epoch = proposal_inputs.step_epoch

        batch_descriptor = BatchDescriptor(
            num_tokens=execution.execution_token_count,
            num_reqs=proposal_inputs.num_reqs,
            uniform=True,
        )
        slot_mappings = dict(proposal_inputs.draft_query_slot_mappings)
        with set_forward_context(
            draft_attn_metadata,
            self.draft_vllm_config,
            num_tokens=execution.execution_token_count,
            num_tokens_across_dp=proposal_inputs.num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=batch_descriptor,
            slot_mapping=slot_mappings,
            input_ids=proposal_inputs.draft_input_ids,
            model_instance=self.model,
        ):
            forward_context = get_forward_context()
            draft_context = build_ascend_forward_context(
                attn_metadata=draft_attn_metadata,
                vllm_config=self.draft_vllm_config,
                num_tokens=execution.execution_token_count,
                num_tokens_across_dp=proposal_inputs.num_tokens_across_dp,
                dp_metadata=forward_context.dp_metadata,
                num_actual_tokens=proposal_inputs.num_query_tokens,
                model_instance=self.model,
                is_draft_model=True,
                draft_attn_metadatas=draft_attn_metadata,
                input_ids=proposal_inputs.draft_input_ids,
            )
            draft_context["is_draft_model_prefill"] = True
            forward_context.additional_kwargs.update(draft_context)
            output = self.model(
                input_ids=proposal_inputs.draft_input_ids,
                positions=proposal_inputs.draft_positions,
            )

        if not isinstance(output, torch.Tensor):
            raise TypeError("Ascend DSpark draft backbone must return one rank-local tensor.")
        expected_shape = (
            execution.execution_token_count,
            int(self.draft_model_config.hf_config.hidden_size),
        )
        if output.shape != expected_shape:
            raise RuntimeError(
                "Ascend DSpark draft backbone output shape does not match its "
                f"HC-head ABI: {tuple(output.shape)} instead of {expected_shape}."
            )
        self._validate_step_tensor("draft backbone output", output, ndim=2)
        if not output.dtype.is_floating_point or output.numel() == 0 or output.device.type == "meta":
            raise RuntimeError("Ascend DSpark draft backbone returned an invalid floating-point tensor.")
        return output

    def _execute_draft_backbone(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> torch.Tensor:
        """Run context precompute, real metadata, and the eager draft backbone."""
        execution = self._combine_and_precompute_draft_context(proposal_inputs)
        draft_attn_metadata = self._build_draft_forward_metadata(execution)
        return self._run_draft_model_forward(execution, draft_attn_metadata)

    @staticmethod
    def _qualified_class_name(instance: Any) -> str:
        instance_type = type(instance)
        return f"{instance_type.__module__}.{instance_type.__name__}"

    def _inspect_markov_modules(
        self,
        model: torch.nn.Module,
    ) -> MappingProxyType[str, Any]:
        lm_head = getattr(model, "lm_head", None)
        backbone = getattr(model, "model", None)
        markov_head = getattr(backbone, "markov_head", None)
        confidence_head = getattr(backbone, "confidence_head", None)
        if not isinstance(lm_head, torch.nn.Module):
            raise RuntimeError("Ascend DSpark draft model has no loaded LM head module.")
        if not isinstance(markov_head, torch.nn.Module):
            raise RuntimeError("Ascend DSpark draft model has no loaded Markov head module.")

        last_stage = int(getattr(backbone, "num_dspark_layers", 3)) - 1
        if last_stage < 0:
            raise RuntimeError("Ascend DSpark draft model has no MTP stages.")
        lm_head_parameter_names = tuple(
            f"mtp.{last_stage}.head.{name}" for name, _parameter in lm_head.named_parameters()
        )
        markov_parameter_names = tuple(
            f"mtp.{last_stage}.markov_head.{name}" for name, _parameter in markov_head.named_parameters()
        )
        if not lm_head_parameter_names or not markov_parameter_names:
            raise RuntimeError("Ascend DSpark Markov sampling requires loaded LM and Markov parameters.")
        return MappingProxyType(
            {
                "lm_head": lm_head,
                "markov_head": markov_head,
                "confidence_head": confidence_head,
                "lm_head_id": id(lm_head),
                "markov_head_id": id(markov_head),
                "confidence_head_id": id(confidence_head) if confidence_head is not None else None,
                "lm_head_class": self._qualified_class_name(lm_head),
                "markov_head_class": self._qualified_class_name(markov_head),
                "lm_head_parameter_names": lm_head_parameter_names,
                "markov_parameter_names": markov_parameter_names,
            }
        )

    def _validate_markov_inputs(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
        hidden_states: torch.Tensor,
    ) -> tuple[int, int, int]:
        if proposal_inputs.step_epoch != self._proposal_step_epoch:
            raise RuntimeError("Ascend DSpark Markov inputs belong to a stale target step.")
        if proposal_inputs.rank != self.rank:
            raise RuntimeError("Ascend DSpark Markov inputs belong to a different NPU rank.")
        if self._context_kv_step_epoch != proposal_inputs.step_epoch:
            raise RuntimeError("Ascend DSpark Markov inputs have no matching context-KV write.")
        if self._draft_forward_step_epoch != proposal_inputs.step_epoch:
            raise RuntimeError("Ascend DSpark Markov inputs have no matching draft backbone output.")
        if (
            len(proposal_inputs.request_ids) != proposal_inputs.num_reqs
            or len(set(proposal_inputs.request_ids)) != proposal_inputs.num_reqs
        ):
            raise RuntimeError("Ascend DSpark Markov request ownership changed after draft forward.")

        num_reqs = proposal_inputs.num_reqs
        num_speculative_tokens = proposal_inputs.num_speculative_tokens
        if num_speculative_tokens != _DSPARK_MARKOV_FIXED_K:
            raise NotImplementedError(
                f"Ascend DSpark Markov sampling currently requires fixed K=5, got K={num_speculative_tokens}."
            )
        expected_tokens = num_reqs * num_speculative_tokens
        if proposal_inputs.num_query_tokens != expected_tokens:
            raise ValueError("Ascend DSpark Markov sampling requires B*K draft tokens.")

        hidden_states = self._validate_step_tensor(
            "Markov backbone hidden states",
            hidden_states,
            ndim=2,
        )
        hidden_size = int(self.draft_model_config.hf_config.hidden_size)
        if hidden_states.shape != (expected_tokens, hidden_size):
            raise ValueError(
                "Ascend DSpark Markov hidden states must use request-major "
                f"[B*K,H] layout, got {tuple(hidden_states.shape)}."
            )
        if not hidden_states.dtype.is_floating_point:
            raise TypeError("Ascend DSpark Markov hidden states must be floating point.")

        query_start_loc = self._validate_step_tensor(
            "Markov draft query-start locations",
            proposal_inputs.draft_query_start_loc,
            ndim=1,
            dtypes=(torch.int32,),
        )
        if query_start_loc.shape != (num_reqs + 1,):
            raise ValueError("Ascend DSpark Markov query layout must contain B+1 offsets.")
        expected_query_start_loc = (
            torch.arange(num_reqs + 1, dtype=torch.int32, device=self.device) * num_speculative_tokens
        )
        _assert_markov_tensor_contract(
            torch.all(query_start_loc == expected_query_start_loc),
            "Ascend DSpark Markov query layout must be request-major with fixed K.",
        )

        anchor_token_ids = self._validate_step_tensor(
            "Markov anchor token IDs",
            proposal_inputs.anchor_token_ids,
            ndim=1,
            dtypes=(torch.int32, torch.int64),
        )
        if anchor_token_ids.shape != (num_reqs,):
            raise ValueError("Ascend DSpark Markov sampling requires one anchor per request.")
        self._validate_step_tensor(
            "Markov request-state indices",
            proposal_inputs.request_state_indices,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_reqs,
        )
        return num_reqs, num_speculative_tokens, expected_tokens

    def _require_greedy_markov_sampling(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> None:
        temperatures = self._validate_step_tensor(
            "Markov sampling temperatures",
            proposal_inputs.temperature,
            ndim=1,
            dtypes=(torch.float32,),
        )
        request_indices = proposal_inputs.request_state_indices[: proposal_inputs.num_reqs].to(torch.int64)
        _assert_markov_tensor_contract(
            ((request_indices >= 0) & (request_indices < temperatures.shape[0])).all(),
            "Ascend DSpark Markov request-state mapping is outside the temperature buffer.",
        )
        active_temperatures = temperatures[request_indices]
        greedy = torch.all(active_temperatures == 0)
        if active_temperatures.device.type == "cpu":
            if not bool(greedy):
                dspark_runtime_not_wired("V2 DSpark stochastic Markov sampling")
            return
        torch._assert_async(
            greedy,
            "Ascend DSpark V2 DSpark stochastic Markov sampling is not yet wired.",
        )

    @torch.inference_mode()
    def _execute_sequential_markov_sampling(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
        hidden_states: torch.Tensor,
    ) -> AscendDSparkMarkovResult:
        """Run fixed-K greedy Markov recurrence without publishing a proposal."""
        if proposal_inputs.step_epoch != self._proposal_step_epoch:
            raise RuntimeError("Ascend DSpark Markov inputs belong to a stale target step.")
        if self._markov_attempt_step_epoch == proposal_inputs.step_epoch:
            raise RuntimeError("Ascend DSpark Markov sampling already attempted this consumed proposal epoch.")

        # Clear visibility before any head execution. A failed step remains
        # consumed because the context KV write and draft forward are not
        # reversible, while no partial candidate state becomes observable.
        self._markov_attempt_step_epoch = proposal_inputs.step_epoch
        self._markov_step_epoch = None
        self._markov_result = None
        num_reqs, num_speculative_tokens, expected_tokens = self._validate_markov_inputs(
            proposal_inputs,
            hidden_states,
        )
        self._require_greedy_markov_sampling(proposal_inputs)

        module_contract = self._markov_module_contract
        if module_contract is None:
            module_contract = self._inspect_markov_modules(self.model)
            self._markov_module_contract = module_contract
        current_contract = self._inspect_markov_modules(self.model)
        if (
            current_contract["lm_head_id"] != module_contract["lm_head_id"]
            or current_contract["markov_head_id"] != module_contract["markov_head_id"]
            or current_contract["confidence_head_id"] != module_contract["confidence_head_id"]
        ):
            raise RuntimeError("Ascend DSpark loaded LM/Markov module identity changed before sampling.")

        base_logits = self.model.compute_draft_logits(hidden_states)
        if not isinstance(base_logits, torch.Tensor):
            raise TypeError("Ascend DSpark draft LM head must return full-vocabulary logits on every rank.")
        self._validate_step_tensor("Markov base logits", base_logits, ndim=2)
        vocab_size = int(self.draft_model_config.hf_config.vocab_size)
        if base_logits.shape != (expected_tokens, vocab_size):
            raise RuntimeError(
                "Ascend DSpark cannot sample a local vocabulary shard: "
                f"expected {(expected_tokens, vocab_size)}, got {tuple(base_logits.shape)}."
            )
        if not base_logits.dtype.is_floating_point:
            raise TypeError("Ascend DSpark Markov base logits must be floating point.")
        _assert_markov_tensor_contract(
            ~torch.isnan(base_logits).any(),
            "Ascend DSpark Markov base logits contain NaN.",
        )
        logical_base_logits = base_logits.view(
            num_reqs,
            num_speculative_tokens,
            vocab_size,
        )

        predecessor = proposal_inputs.anchor_token_ids
        _assert_markov_tensor_contract(
            ((predecessor >= 0) & (predecessor < vocab_size)).all(),
            "Ascend DSpark Markov anchor token is outside the shared target/draft vocabulary.",
        )
        steps: list[AscendDSparkMarkovStep] = []
        selected_steps: list[torch.Tensor] = []
        for step_index in range(num_speculative_tokens):
            markov_input = predecessor
            markov_embed = self.model.markov_embed(markov_input)
            if not isinstance(markov_embed, torch.Tensor):
                raise TypeError("Ascend DSpark Markov embedding must be a tensor.")
            self._validate_step_tensor("Markov embedding", markov_embed, ndim=2)
            if markov_embed.shape[0] != num_reqs or not markov_embed.dtype.is_floating_point:
                raise RuntimeError("Ascend DSpark Markov embedding must contain one floating-point row per request.")
            _assert_markov_tensor_contract(
                ~torch.isnan(markov_embed).any(),
                "Ascend DSpark Markov embedding contains NaN.",
            )
            markov_bias = self.model.markov_bias(markov_embed)
            if not isinstance(markov_bias, torch.Tensor):
                raise TypeError("Ascend DSpark Markov head must return a tensor bias.")
            self._validate_step_tensor("Markov vocabulary bias", markov_bias, ndim=2)
            if markov_bias.shape != (num_reqs, vocab_size):
                raise RuntimeError(
                    f"Ascend DSpark Markov bias must cover the full vocabulary, got {tuple(markov_bias.shape)}."
                )
            step_logits = logical_base_logits[:, step_index, :] + markov_bias
            _assert_markov_tensor_contract(
                ~torch.isnan(step_logits).any(),
                "Ascend DSpark Markov logits contain NaN.",
            )
            _assert_markov_tensor_contract(
                torch.isfinite(step_logits).any(dim=-1).all(),
                "Ascend DSpark Markov logits contain a row with no selectable finite token.",
            )
            draft_selected = torch.argmax(step_logits, dim=-1)
            selected = self.model.map_draft_to_target(draft_selected)
            if selected is not draft_selected:
                raise RuntimeError(
                    "Ascend DSpark 0731 full-vocabulary checkpoint requires identity draft-to-target mapping."
                )
            self._validate_step_tensor(
                "Markov selected token IDs",
                selected,
                ndim=1,
                dtypes=(torch.int64,),
            )
            if selected.shape != (num_reqs,):
                raise RuntimeError("Ascend DSpark Markov sampling returned an invalid request dimension.")
            _assert_markov_tensor_contract(
                ((selected >= 0) & (selected < vocab_size)).all(),
                "Ascend DSpark Markov sampling returned a token outside the full vocabulary.",
            )
            steps.append(
                AscendDSparkMarkovStep(
                    step_index=step_index,
                    predecessor_source=("anchor_token_ids" if step_index == 0 else "previous_sampled_token"),
                    predecessor_token_ids=predecessor,
                    markov_input_token_ids=markov_input,
                    selected_token_ids=selected,
                )
            )
            selected_steps.append(selected)
            predecessor = selected

        candidate_tokens = torch.stack(selected_steps, dim=1)
        result = AscendDSparkMarkovResult(
            step_epoch=proposal_inputs.step_epoch,
            rank=proposal_inputs.rank,
            request_ids=proposal_inputs.request_ids,
            request_state_indices=proposal_inputs.request_state_indices[:num_reqs].clone(),
            num_reqs=num_reqs,
            num_speculative_tokens=num_speculative_tokens,
            backbone_hidden_states=hidden_states,
            candidate_tokens=candidate_tokens,
            steps=tuple(steps),
            physical_hidden_shape=tuple(hidden_states.shape),
            physical_base_logits_shape=tuple(base_logits.shape),
            logical_base_logits_shape=tuple(logical_base_logits.shape),
            logical_candidate_shape=tuple(candidate_tokens.shape),
            vocab_size=vocab_size,
            lm_head_class=module_contract["lm_head_class"],
            markov_head_class=module_contract["markov_head_class"],
            lm_head_parameter_names=module_contract["lm_head_parameter_names"],
            markov_parameter_names=module_contract["markov_parameter_names"],
            loaded_module_identity_preserved=True,
            confidence_head_present=module_contract["confidence_head"] is not None,
            confidence_head_used=False,
        )
        self._markov_result = result
        self._markov_step_epoch = proposal_inputs.step_epoch
        return result

    def _build_core_proposal(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
        result: AscendDSparkMarkovResult,
    ) -> torch.Tensor:
        """Atomically publish one complete Markov result through core's ABI."""
        if self._published_candidate_tokens is not None:
            raise RuntimeError("Ascend DSpark candidate state was already published.")
        if result is not self._markov_result:
            raise RuntimeError("Ascend DSpark proposal must use the current owned Markov result.")
        if result.step_epoch != proposal_inputs.step_epoch or result.step_epoch != self._proposal_step_epoch:
            raise RuntimeError("Ascend DSpark proposal belongs to a stale target step.")
        if result.rank != self.rank or result.rank != proposal_inputs.rank:
            raise RuntimeError("Ascend DSpark proposal belongs to a different NPU rank.")
        if result.request_ids != proposal_inputs.request_ids:
            raise RuntimeError("Ascend DSpark proposal request ownership does not match its target step.")
        if result.num_reqs != proposal_inputs.num_reqs:
            raise RuntimeError("Ascend DSpark proposal request count does not match its target step.")
        if len(set(result.request_ids)) != result.num_reqs:
            raise RuntimeError("Ascend DSpark proposal request ownership contains duplicate request IDs.")
        if (
            result.num_speculative_tokens != self.num_speculative_steps
            or result.num_speculative_tokens != proposal_inputs.num_speculative_tokens
        ):
            raise RuntimeError("Ascend DSpark proposal length does not match the configured speculative length.")
        if len(result.steps) != result.num_speculative_tokens or tuple(
            step.step_index for step in result.steps
        ) != tuple(range(result.num_speculative_tokens)):
            raise RuntimeError("Ascend DSpark proposal requires every Markov step in order.")

        request_state_indices = self._validate_step_tensor(
            "proposal request-state indices",
            result.request_state_indices,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=result.num_reqs,
        )
        expected_request_state_indices = proposal_inputs.request_state_indices[: result.num_reqs]
        _assert_markov_tensor_contract(
            (request_state_indices == expected_request_state_indices).all(),
            "Ascend DSpark proposal request-state ownership changed before publication.",
        )

        candidate_tokens = self._validate_step_tensor(
            "proposal candidate tokens",
            result.candidate_tokens,
            ndim=2,
            dtypes=(torch.int64,),
        )
        expected_shape = (result.num_reqs, result.num_speculative_tokens)
        if candidate_tokens.shape != expected_shape or result.logical_candidate_shape != expected_shape:
            raise RuntimeError(
                "Ascend DSpark proposal candidates must use request-major "
                f"layout {expected_shape}, got {tuple(candidate_tokens.shape)}."
            )
        if not candidate_tokens.is_contiguous():
            raise RuntimeError("Ascend DSpark proposal candidates must be contiguous in request-major order.")
        _assert_markov_tensor_contract(
            ((candidate_tokens >= 0) & (candidate_tokens < result.vocab_size)).all(),
            "Ascend DSpark proposal contains a token outside the shared vocabulary.",
        )

        conflicting_request_ids = set(result.request_ids).intersection(
            self._published_proposal_owners,
        )
        if conflicting_request_ids:
            raise RuntimeError(
                "Ascend DSpark proposal publication conflicts with outstanding "
                "request ownership for requests "
                f"{sorted(conflicting_request_ids)!r}."
            )

        # Publish only after the complete provenance, ownership and tensor
        # contract has been validated. Core receives this exact tensor.
        self._next_proposal_skipped = False
        self._proposal_publication_count += 1
        self._proposal_generated_count += 1
        self._proposal_returned_count += 1
        lifecycle = AscendDSparkProposalLifecycle(
            proposal_epoch=result.step_epoch,
            owner_epoch=result.step_epoch,
            consumer_epoch=None,
            request_ids=result.request_ids,
            generated=True,
            returned_to_core=True,
            installed=False,
            consumed=False,
            discarded_terminal=False,
        )
        new_owners = {
            request_id: _PublishedProposalOwner(
                request_id=request_id,
                producer_epoch=result.step_epoch,
                request_state_indices=request_state_indices,
                candidate_tokens=candidate_tokens,
                publication_row=row,
                published_length=result.num_speculative_tokens,
                lifecycle=replace(lifecycle, request_ids=(request_id,)),
            )
            for row, request_id in enumerate(result.request_ids)
        }
        # Commit the entire cohort only after all ownership and tensor checks
        # succeed. Outstanding delayed owners from older cohorts remain in the
        # registry and cannot be overwritten by this publication.
        self._published_proposal_owners.update(new_owners)
        self._set_active_published_proposal(
            result.request_ids,
            reconciled=False,
        )
        return candidate_tokens

    def _log_proposal_disposition(
        self,
        lifecycle: AscendDSparkProposalLifecycle,
    ) -> None:
        """Emit one content-free record for a disposition transition."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        published_length = (
            int(self._published_candidate_tokens.shape[1])
            if self._published_candidate_tokens is not None
            else self.num_speculative_steps
        )
        for request_index, request_id in enumerate(lifecycle.request_ids):
            scheduled_length = (
                lifecycle.scheduled_lengths[request_index] if request_index < len(lifecycle.scheduled_lengths) else 0
            )
            logger.debug(
                "DSPARK_PROPOSAL_DISPOSITION=%s",
                json.dumps(
                    {
                        "rank": self.rank,
                        "request_id": request_id,
                        "producer_epoch": lifecycle.proposal_epoch,
                        "consumer_epoch": lifecycle.consumer_epoch,
                        "published_length": published_length,
                        "scheduled_length": scheduled_length,
                        "disposition": lifecycle.disposition,
                        "token_prefix_match": lifecycle.token_prefix_match,
                        "installed": lifecycle.installed,
                        "truncated": lifecycle.truncated,
                        "dropped": lifecycle.dropped,
                        "drop_reason": lifecycle.drop_reason,
                        "terminal": lifecycle.discarded_terminal,
                        "consumed": lifecycle.consumed,
                    },
                    sort_keys=True,
                ),
            )

    def _clear_published_proposal_state(self) -> None:
        self._published_proposal_owners.clear()
        self._clear_active_published_proposal()
        self._clear_proposal_execution_state()

    def _clear_proposal_execution_state(self) -> None:
        """Clear transient producer state after its last owner retires."""
        self._prepared_step_epoch = None
        self._context_kv_step_epoch = None
        self._draft_forward_step_epoch = None
        self._markov_attempt_step_epoch = None
        self._markov_step_epoch = None
        self._markov_result = None

    def _clear_active_published_proposal(self) -> None:
        """Clear only the cohort selected for the current verification."""
        self._published_proposal_step_epoch = None
        self._published_proposal_request_ids = None
        self._published_proposal_request_state_indices = None
        self._published_proposal_owner_epochs = ()
        self._published_candidate_tokens = None
        self._active_published_proposal_owner_ids = ()
        self._active_proposal_reconciled = False
        self._published_proposal_consumed = False
        self._current_proposal_lifecycle = None

    @staticmethod
    def _owner_row_tensors(
        owners: tuple[_PublishedProposalOwner, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize request-major rows without changing row ownership."""
        if not owners:
            raise RuntimeError("Ascend DSpark cannot assemble an empty proposal owner set.")
        candidate_source = owners[0].candidate_tokens
        state_source = owners[0].request_state_indices
        same_source = all(
            owner.candidate_tokens is candidate_source and owner.request_state_indices is state_source
            for owner in owners
        )
        rows = tuple(owner.publication_row for owner in owners)
        if same_source and rows == tuple(range(candidate_source.shape[0])):
            return candidate_source, state_source

        if same_source:
            row_indices = torch.tensor(
                rows,
                dtype=torch.int64,
                device=candidate_source.device,
            )
            candidate_rows = candidate_source.index_select(0, row_indices)
            state_rows = state_source.index_select(
                0,
                row_indices.to(state_source.device),
            )
            return candidate_rows, state_rows

        candidate_rows = torch.cat(
            tuple(owner.candidate_tokens[owner.publication_row : owner.publication_row + 1] for owner in owners),
            dim=0,
        )
        state_rows = torch.cat(
            tuple(owner.request_state_indices[owner.publication_row : owner.publication_row + 1] for owner in owners),
            dim=0,
        )
        return candidate_rows, state_rows

    @staticmethod
    def _batch_lifecycle(
        owners: tuple[_PublishedProposalOwner, ...],
    ) -> AscendDSparkProposalLifecycle:
        """Build a diagnostic batch view from request-owned lifecycle rows."""
        if not owners:
            raise RuntimeError("Ascend DSpark cannot summarize an empty proposal owner set.")
        lifecycles = tuple(owner.lifecycle for owner in owners)
        dispositions = {lifecycle.disposition for lifecycle in lifecycles}
        disposition = (
            next(iter(dispositions))
            if len(dispositions) == 1
            else ("TRUNCATED" if any(lifecycle.truncated for lifecycle in lifecycles) else "INSTALLED")
        )
        consumer_epochs = {lifecycle.consumer_epoch for lifecycle in lifecycles if lifecycle.consumer_epoch is not None}
        consumer_epoch = next(iter(consumer_epochs)) if len(consumer_epochs) == 1 else None
        base = lifecycles[0]
        has_scheduled_lengths = any(lifecycle.scheduled_lengths for lifecycle in lifecycles)
        return replace(
            base,
            proposal_epoch=max(owner.producer_epoch for owner in owners),
            owner_epoch=max(owner.producer_epoch for owner in owners),
            consumer_epoch=consumer_epoch,
            request_ids=tuple(owner.request_id for owner in owners),
            scheduled_lengths=(
                tuple(lifecycle.scheduled_lengths[0] if lifecycle.scheduled_lengths else 0 for lifecycle in lifecycles)
                if has_scheduled_lengths
                else ()
            ),
            disposition=disposition,
            installed=all(lifecycle.installed for lifecycle in lifecycles),
            consumed=all(lifecycle.consumed for lifecycle in lifecycles),
            discarded_terminal=all(lifecycle.discarded_terminal for lifecycle in lifecycles),
            token_prefix_match=(
                True if all(lifecycle.token_prefix_match is True for lifecycle in lifecycles) else None
            ),
            truncated=any(lifecycle.truncated for lifecycle in lifecycles),
            dropped=all(lifecycle.dropped for lifecycle in lifecycles),
            drop_reason=(
                lifecycles[0].drop_reason if len({lifecycle.drop_reason for lifecycle in lifecycles}) == 1 else None
            ),
        )

    def _set_active_published_proposal(
        self,
        request_ids: tuple[str, ...],
        *,
        reconciled: bool,
    ) -> None:
        owners = tuple(self._published_proposal_owners[request_id] for request_id in request_ids)
        candidate_tokens, request_state_indices = self._owner_row_tensors(owners)
        owner_epochs = tuple(owner.producer_epoch for owner in owners)
        self._published_proposal_step_epoch = max(owner_epochs)
        self._published_proposal_request_ids = request_ids
        self._published_proposal_request_state_indices = request_state_indices
        self._published_proposal_owner_epochs = owner_epochs
        self._published_candidate_tokens = candidate_tokens
        self._active_published_proposal_owner_ids = request_ids
        self._active_proposal_reconciled = reconciled
        self._published_proposal_consumed = False
        self._current_proposal_lifecycle = self._batch_lifecycle(owners)

    def _retire_published_proposal_rows(
        self,
        request_ids_to_drop: set[str],
        *,
        reason: str,
        terminal: bool,
    ) -> bool:
        """Atomically retire authoritative request-owned registry entries."""
        dropped_owners = tuple(
            owner for request_id, owner in self._published_proposal_owners.items() if request_id in request_ids_to_drop
        )
        if not dropped_owners:
            return False
        dropped_lifecycles = tuple(
            replace(
                owner.lifecycle,
                disposition="DROPPED",
                token_prefix_match=None,
                installed=False,
                truncated=False,
                dropped=True,
                drop_reason=reason,
                discarded_terminal=terminal,
            )
            for owner in dropped_owners
        )
        dropped = self._batch_lifecycle(
            tuple(replace(owner, lifecycle=lifecycle) for owner, lifecycle in zip(dropped_owners, dropped_lifecycles))
        )

        for owner, lifecycle in zip(dropped_owners, dropped_lifecycles):
            self._log_proposal_disposition(lifecycle)
            current = self._published_proposal_owners.get(owner.request_id)
            if current is not owner:
                raise RuntimeError("Ascend DSpark proposal ownership changed during retirement.")
        for owner in dropped_owners:
            del self._published_proposal_owners[owner.request_id]

        self._dropped_proposal_lifecycle = dropped
        self._proposal_dropped_count += 1
        if terminal:
            self._terminal_proposal_lifecycle = dropped
            self._terminal_proposal_discard_count += 1

        active_survivors = tuple(
            request_id
            for request_id in self._active_published_proposal_owner_ids
            if request_id in self._published_proposal_owners
        )
        if active_survivors:
            self._set_active_published_proposal(
                active_survivors,
                reconciled=self._active_proposal_reconciled,
            )
        else:
            self._clear_active_published_proposal()
        if self._published_proposal_owners:
            self._markov_step_epoch = None
            self._markov_result = None
        else:
            self._clear_proposal_execution_state()
        return True

    @staticmethod
    def _verification_to_published_rows(
        published_request_ids: tuple[str, ...],
        verification_request_ids: tuple[str, ...],
    ) -> tuple[int, ...]:
        """Build the exact verification-row to publication-row bijection."""
        if len(published_request_ids) != len(verification_request_ids):
            raise RuntimeError("Ascend DSpark verification request ownership does not match the published proposal.")
        published_rows: dict[str, int] = {}
        for row, request_id in enumerate(published_request_ids):
            if request_id in published_rows:
                raise RuntimeError("Ascend DSpark published proposal contains duplicate request ownership.")
            published_rows[request_id] = row

        verification_rows: list[int] = []
        seen_verification_ids: set[str] = set()
        for request_id in verification_request_ids:
            if request_id in seen_verification_ids:
                raise RuntimeError("Ascend DSpark verification contains duplicate request ownership.")
            seen_verification_ids.add(request_id)
            published_row = published_rows.get(request_id)
            if published_row is None:
                raise RuntimeError(
                    "Ascend DSpark verification request ownership does not match the published proposal."
                )
            verification_rows.append(published_row)
        if len(verification_rows) != len(published_rows):
            raise RuntimeError("Ascend DSpark verification request ownership does not match the published proposal.")
        return tuple(verification_rows)

    def reconcile_scheduler_proposal(
        self,
        *,
        scheduled_spec_decode_tokens: Mapping[str, list[int]],
        scheduled_request_ids: set[str],
        finished_request_ids: set[str],
        preempted_request_ids: set[str],
        known_request_ids: set[str],
    ) -> str | None:
        """Reconcile scheduler truth before executing its target batch.

        The registry may contain rows from multiple producer epochs because the
        async core can schedule the next batch before the prior batch's output
        is materialized. A known owner omitted from this batch remains delayed;
        only ``scheduled_spec_decode_tokens`` installs a row for verification.
        """
        registry = self._published_proposal_owners
        if not registry:
            unexpected = set(scheduled_spec_decode_tokens)
            if unexpected:
                raise RuntimeError(
                    "Ascend DSpark scheduler installed proposal tokens without "
                    f"published ownership for requests {sorted(unexpected)!r}."
                )
            return None

        owners = set(registry)
        scheduled_spec_owners = set(scheduled_spec_decode_tokens)
        unexpected_scheduled_owners = scheduled_spec_owners.difference(owners)
        if unexpected_scheduled_owners:
            raise RuntimeError(
                "Ascend DSpark scheduler installed proposal tokens for requests "
                f"outside published ownership {sorted(unexpected_scheduled_owners)!r}."
            )
        retired_owners = owners.intersection(
            finished_request_ids | preempted_request_ids,
        )
        duplicate_retirement = owners.intersection(finished_request_ids).intersection(
            preempted_request_ids,
        )
        scheduled_retired_owners = retired_owners.intersection(
            scheduled_request_ids | scheduled_spec_owners,
        )
        unscheduled_spec_owners = scheduled_spec_owners.difference(scheduled_request_ids)
        if duplicate_retirement or scheduled_retired_owners or unscheduled_spec_owners:
            raise RuntimeError("Ascend DSpark scheduler returned conflicting proposal owner dispositions.")
        unresolved_owners = owners.difference(known_request_ids | retired_owners)
        if unresolved_owners:
            raise RuntimeError(
                "Ascend DSpark published proposal owners have no scheduled, "
                "finished, preempted, or delayed disposition: "
                f"{sorted(unresolved_owners)!r}."
            )

        scheduled_lengths_by_request = {
            request_id: len(tokens) for request_id, tokens in scheduled_spec_decode_tokens.items()
        }
        invalid_lengths = {
            request_id: length
            for request_id, length in scheduled_lengths_by_request.items()
            if length <= 0 or length > registry[request_id].published_length
        }
        if invalid_lengths:
            raise RuntimeError("Ascend DSpark scheduler returned an invalid installed proposal length.")

        # Validate every installed transition before retiring any other owner.
        # A conflicting repeated disposition must leave the complete registry
        # unchanged, including terminal and preempted rows from this output.
        installed_request_ids = tuple(request_id for request_id in registry if request_id in scheduled_spec_owners)
        scheduled_lengths = tuple(scheduled_lengths_by_request[request_id] for request_id in installed_request_ids)
        updated_owners: dict[str, _PublishedProposalOwner] = {}
        newly_installed = False
        for request_id, scheduled_length in zip(
            installed_request_ids,
            scheduled_lengths,
        ):
            owner = registry[request_id]
            published_length = owner.published_length
            disposition = "INSTALLED" if scheduled_length == published_length else "TRUNCATED"
            lifecycle = owner.lifecycle
            if lifecycle.disposition in {"INSTALLED", "TRUNCATED"}:
                if lifecycle.disposition != disposition or lifecycle.scheduled_lengths != (scheduled_length,):
                    raise RuntimeError("Ascend DSpark scheduler changed an already reconciled proposal disposition.")
                updated_owners[request_id] = owner
                continue
            if lifecycle.disposition != "GENERATED":
                raise RuntimeError("Ascend DSpark proposal received an invalid repeated scheduler disposition.")
            updated_lifecycle = replace(
                lifecycle,
                scheduled_lengths=(scheduled_length,),
                disposition=disposition,
                installed=True,
                truncated=disposition == "TRUNCATED",
            )
            updated_owners[request_id] = replace(
                owner,
                lifecycle=updated_lifecycle,
            )
            newly_installed = True

        registry_mutated = False
        terminal_owners = owners.intersection(finished_request_ids)
        if terminal_owners:
            self._retire_published_proposal_rows(
                terminal_owners,
                reason="terminal",
                terminal=True,
            )
            registry_mutated = True

        owners = set(registry)
        preempted_owners = owners.intersection(preempted_request_ids)
        if preempted_owners:
            self._retire_published_proposal_rows(
                preempted_owners,
                reason="preempted",
                terminal=False,
            )
            registry_mutated = True

        owners = set(registry)
        active_owners = owners.intersection(scheduled_request_ids)
        scheduled_owners = owners.intersection(scheduled_spec_owners)
        scheduled_without_proposal = active_owners.difference(scheduled_owners)
        if scheduled_without_proposal:
            self._retire_published_proposal_rows(
                scheduled_without_proposal,
                reason="scheduled_without_proposal",
                terminal=False,
            )
            registry_mutated = True

        owners = set(registry)
        scheduled_owners = owners.intersection(scheduled_spec_owners)
        if not scheduled_owners:
            self._clear_active_published_proposal()
            if not registry:
                return "DROPPED"
            return "DELAYED"

        # Preserve publication order within every outstanding cohort. The
        # target InputBatch may use a different row order and is mapped back to
        # these owner rows at consumption time.
        if set(installed_request_ids) != scheduled_owners:
            raise RuntimeError("Ascend DSpark installed proposal ownership changed during scheduler reconciliation.")

        if (
            not newly_installed
            and not registry_mutated
            and self._active_proposal_reconciled
            and self._active_published_proposal_owner_ids == installed_request_ids
        ):
            lifecycle = self._current_proposal_lifecycle
            if lifecycle is None:
                raise RuntimeError("Ascend DSpark lost idempotent proposal lifecycle state.")
            return lifecycle.disposition

        # In the padded async core path these lists may initially be filled
        # with -1 placeholders and are updated request-by-request. The
        # SchedulerOutput remains authoritative for presence and length. The
        # exact device-token prefix is validated below from the InputBatch
        # built by the core runner without adding a host sync.
        registry.update(updated_owners)
        self._set_active_published_proposal(
            installed_request_ids,
            reconciled=True,
        )
        lifecycle = self._current_proposal_lifecycle
        if lifecycle is None:
            raise RuntimeError("Ascend DSpark lost installed ownership during reconciliation.")
        disposition = lifecycle.disposition
        if newly_installed:
            self._proposal_installed_count += 1
            for owner in updated_owners.values():
                self._log_proposal_disposition(owner.lifecycle)
        return disposition

    def _consume_published_proposal_after_verification(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        """Validate and consume the proposal installed for this target step."""
        candidate_tokens = self._published_candidate_tokens
        producer_epoch = self._published_proposal_step_epoch
        request_ids = self._published_proposal_request_ids
        request_state_indices = self._published_proposal_request_state_indices
        if candidate_tokens is None or producer_epoch is None or request_ids is None or request_state_indices is None:
            raise RuntimeError("Ascend DSpark has no published proposal to verify.")
        if self._published_proposal_consumed:
            raise RuntimeError("Ascend DSpark proposal was already consumed.")
        owner_epochs = self._published_proposal_owner_epochs
        if len(owner_epochs) != len(request_ids):
            raise RuntimeError("Ascend DSpark verification proposal epoch ownership is incomplete.")
        active_owners = tuple(self._published_proposal_owners.get(request_id) for request_id in request_ids)
        if any(owner is None for owner in active_owners):
            raise RuntimeError("Ascend DSpark verification proposal ownership is no longer published.")
        if any(
            owner.producer_epoch != owner_epoch
            for owner, owner_epoch in zip(active_owners, owner_epochs)
            if owner is not None
        ):
            raise RuntimeError("Ascend DSpark verification belongs to a stale proposal epoch.")
        if not self._active_proposal_reconciled and any(
            owner_epoch != self._proposal_step_epoch for owner_epoch in owner_epochs
        ):
            raise RuntimeError("Ascend DSpark verification belongs to a stale proposal epoch.")
        input_request_ids = tuple(input_batch.req_ids)
        if input_batch.num_reqs != len(input_request_ids):
            raise RuntimeError("Ascend DSpark verification InputBatch request count is inconsistent.")
        if len(set(input_request_ids)) != len(input_request_ids):
            raise RuntimeError("Ascend DSpark target InputBatch contains duplicate request ownership.")
        if input_batch.num_draft_tokens_per_req is None:
            raise RuntimeError("Ascend DSpark verification is missing scheduled proposal lengths.")
        input_scheduled_lengths = tuple(int(length) for length in input_batch.num_draft_tokens_per_req)
        published_length = candidate_tokens.shape[1]
        if len(input_scheduled_lengths) != input_batch.num_reqs:
            raise RuntimeError(
                "Ascend DSpark verification request ownership does not match its scheduled proposal lengths."
            )
        if any(length < 0 or length > published_length for length in input_scheduled_lengths):
            raise RuntimeError("Ascend DSpark verification received an invalid scheduled proposal length.")

        # A target batch may mix proposal verification rows with newly admitted
        # prefill rows. Core's ``num_draft_tokens_per_req`` is derived solely
        # from SchedulerOutput.scheduled_spec_decode_tokens, so a positive
        # length is the authoritative verification-row identity. A zero-length
        # prefill row must neither index nor consume the previous publication.
        verification_batch_rows = tuple(row for row, length in enumerate(input_scheduled_lengths) if length > 0)
        verification_request_ids = tuple(input_request_ids[row] for row in verification_batch_rows)
        verification_scheduled_lengths = tuple(input_scheduled_lengths[row] for row in verification_batch_rows)
        # Core may permute equal-length requests between proposal production
        # and verification. Resolve every verification tensor row back to its
        # published owner instead of treating either dictionary order as ABI.
        verification_to_published = self._verification_to_published_rows(
            request_ids,
            verification_request_ids,
        )
        published_scheduled_lengths = [0] * len(request_ids)
        for verification_row, published_row in enumerate(verification_to_published):
            published_scheduled_lengths[published_row] = verification_scheduled_lengths[verification_row]
        scheduled_lengths = tuple(published_scheduled_lengths)
        lifecycle = self._current_proposal_lifecycle
        if lifecycle is None or producer_epoch != max(owner_epochs):
            raise RuntimeError("Ascend DSpark proposal lifecycle ownership is missing or stale.")
        disposition = "INSTALLED" if all(length == published_length for length in scheduled_lengths) else "TRUNCATED"
        if lifecycle.disposition == "GENERATED":
            # InputBatch is constructed directly from SchedulerOutput by the
            # core runner. This preserves the direct speculator test boundary
            # while production normally reconciles earlier in finish_requests.
            lifecycle = replace(
                lifecycle,
                scheduled_lengths=scheduled_lengths,
                disposition=disposition,
                installed=True,
                truncated=disposition == "TRUNCATED",
            )
            self._current_proposal_lifecycle = lifecycle
            for request_id, scheduled_length in zip(
                request_ids,
                scheduled_lengths,
            ):
                owner = self._published_proposal_owners[request_id]
                owner_disposition = "INSTALLED" if scheduled_length == published_length else "TRUNCATED"
                self._published_proposal_owners[request_id] = replace(
                    owner,
                    lifecycle=replace(
                        owner.lifecycle,
                        scheduled_lengths=(scheduled_length,),
                        disposition=owner_disposition,
                        installed=True,
                        truncated=owner_disposition == "TRUNCATED",
                    ),
                )
            self._active_proposal_reconciled = True
            self._proposal_installed_count += 1
            self._log_proposal_disposition(lifecycle)
        elif lifecycle.disposition != disposition or lifecycle.scheduled_lengths != scheduled_lengths:
            raise RuntimeError("Ascend DSpark verification lengths differ from the reconciled scheduler disposition.")
        if input_batch.num_draft_tokens != sum(verification_scheduled_lengths):
            raise RuntimeError("Ascend DSpark verification draft-token count is inconsistent.")

        identity_rows = tuple(range(len(request_ids)))
        if verification_to_published == identity_rows:
            verification_candidate_tokens = candidate_tokens
            expected_request_state_indices = request_state_indices
        else:
            candidate_row_indices = torch.tensor(
                verification_to_published,
                dtype=torch.int64,
                device=candidate_tokens.device,
            )
            verification_candidate_tokens = candidate_tokens.index_select(
                0,
                candidate_row_indices,
            )
            state_row_indices = candidate_row_indices.to(request_state_indices.device)
            expected_request_state_indices = request_state_indices.index_select(
                0,
                state_row_indices,
            )

        input_request_indices = self._validate_step_tensor(
            "verification request-state indices",
            input_batch.idx_mapping,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=input_batch.num_reqs,
        )[: input_batch.num_reqs]
        verification_batch_row_indices = torch.tensor(
            verification_batch_rows,
            dtype=torch.int64,
            device=input_request_indices.device,
        )
        active_request_indices = input_request_indices.index_select(
            0,
            verification_batch_row_indices,
        )
        _assert_markov_tensor_contract(
            (active_request_indices == expected_request_state_indices).all(),
            "Ascend DSpark verification request-state mapping changed after publication.",
        )
        query_start_loc = self._validate_step_tensor(
            "verification query-start locations",
            input_batch.query_start_loc,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=input_batch.num_reqs + 1,
        )[: input_batch.num_reqs + 1]
        query_row_indices = verification_batch_row_indices.to(query_start_loc.device)
        verification_query_starts = query_start_loc.index_select(
            0,
            query_row_indices,
        )
        verification_query_ends = query_start_loc.index_select(
            0,
            query_row_indices + 1,
        )
        scheduled_lengths_tensor = torch.tensor(
            verification_scheduled_lengths,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        expected_query_lengths = scheduled_lengths_tensor + 1
        _assert_markov_tensor_contract(
            (verification_query_ends - verification_query_starts == expected_query_lengths).all(),
            "Ascend DSpark verification input must contain one anchor and every scheduled candidate per request.",
        )
        target_input_ids = self._validate_step_tensor(
            "verification target input IDs",
            input_batch.input_ids,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=input_batch.num_tokens,
        )
        max_scheduled_length = max(scheduled_lengths)
        candidate_offsets = torch.arange(
            1,
            max_scheduled_length + 1,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        scheduled_mask = candidate_offsets[None, :] <= scheduled_lengths_tensor[:, None]
        candidate_indices = (verification_query_starts[:, None] + candidate_offsets[None, :])[scheduled_mask]
        consumed_candidates = target_input_ids[candidate_indices.to(torch.int64)]
        expected_candidates = verification_candidate_tokens[:, :max_scheduled_length][scheduled_mask]
        _assert_markov_tensor_contract(
            (consumed_candidates.to(torch.int64) == expected_candidates).all(),
            "Ascend DSpark verification inputs do not contain the published candidate set prefix scheduled by core.",
        )

        input_num_sampled = self._validate_step_tensor(
            "verification sampled-token counts",
            num_sampled,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=input_batch.num_reqs,
        )[: input_batch.num_reqs]
        input_num_rejected = self._validate_step_tensor(
            "verification rejected-token counts",
            num_rejected,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=input_batch.num_reqs,
        )[: input_batch.num_reqs]
        sampling_row_indices = verification_batch_row_indices.to(input_num_sampled.device)
        num_sampled = input_num_sampled.index_select(0, sampling_row_indices)
        num_rejected = input_num_rejected.index_select(
            0,
            sampling_row_indices.to(input_num_rejected.device),
        )
        _assert_markov_tensor_contract(
            ((num_sampled >= 1) & (num_sampled <= expected_query_lengths)).all(),
            "Ascend DSpark verification returned an invalid sampled-token count.",
        )
        _assert_markov_tensor_contract(
            ((num_rejected >= 0) & (num_rejected <= scheduled_lengths_tensor)).all(),
            "Ascend DSpark verification returned an invalid rejected-token count.",
        )
        _assert_markov_tensor_contract(
            (num_sampled + num_rejected == expected_query_lengths).all(),
            "Ascend DSpark verification sampled/rejected counts do not cover the full proposal window.",
        )
        temperatures = self._validate_step_tensor(
            "verification temperatures",
            temperature,
            ndim=1,
            dtypes=(torch.float32,),
        )
        _assert_markov_tensor_contract(
            (temperatures[active_request_indices.to(torch.int64)] == 0).all(),
            "Ascend DSpark single-round verification supports deterministic greedy sampling only.",
        )

        consumed_owners: list[_PublishedProposalOwner] = []
        for request_id, owner_epoch, scheduled_length in zip(
            request_ids,
            owner_epochs,
            scheduled_lengths,
        ):
            owner = self._published_proposal_owners[request_id]
            owner_disposition = "INSTALLED" if scheduled_length == owner.published_length else "TRUNCATED"
            consumed_owner = replace(
                owner,
                lifecycle=replace(
                    owner.lifecycle,
                    consumer_epoch=owner_epoch + 1,
                    disposition=owner_disposition,
                    scheduled_lengths=(scheduled_length,),
                    token_prefix_match=True,
                    installed=True,
                    consumed=True,
                ),
            )
            consumed_owners.append(consumed_owner)
        for owner in consumed_owners:
            self._published_proposal_owners[owner.request_id] = owner

        consumer_epoch = max(owner_epochs) + 1
        self._proposal_consumer_step_epoch = consumer_epoch
        self._published_proposal_consumed = True
        self._proposal_consumption_count += 1
        lifecycle = self._batch_lifecycle(tuple(consumed_owners))
        if lifecycle.consumer_epoch is None:
            lifecycle = replace(lifecycle, consumer_epoch=consumer_epoch)
        self._current_proposal_lifecycle = lifecycle
        self._last_consumed_proposal_lifecycle = lifecycle
        for owner in consumed_owners:
            self._log_proposal_disposition(owner.lifecycle)
        self._prepared_step_epoch = None
        self._context_kv_step_epoch = None
        self._draft_forward_step_epoch = None
        self._markov_attempt_step_epoch = None
        self._markov_step_epoch = None
        self._markov_result = None

    def _skip_next_proposal_after_verification(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        """Consume one proposal and retain M2.4A's explicit stop mode."""
        self._consume_published_proposal_after_verification(
            input_batch,
            num_sampled,
            num_rejected,
            temperature,
        )
        self._next_proposal_skipped = True
        self._next_proposal_skip_count += 1
        return None

    def _release_consumed_proposal(self) -> None:
        lifecycle = self._current_proposal_lifecycle
        if lifecycle is None or not lifecycle.consumed:
            raise RuntimeError("Ascend DSpark cannot release an unconsumed proposal.")
        active_request_ids = self._active_published_proposal_owner_ids
        if not active_request_ids:
            raise RuntimeError("Ascend DSpark consumed proposal has no request-owned registry rows.")
        for request_id, owner_epoch in zip(
            active_request_ids,
            self._published_proposal_owner_epochs,
        ):
            owner = self._published_proposal_owners.get(request_id)
            if owner is None or owner.producer_epoch != owner_epoch or not owner.lifecycle.consumed:
                raise RuntimeError("Ascend DSpark consumed proposal ownership changed before release.")
        for request_id in active_request_ids:
            del self._published_proposal_owners[request_id]
        self._clear_active_published_proposal()

    def discard_terminal_proposal(self, finished_request_ids: set[str]) -> bool:
        """Invalidate an optimistic proposal rejected by request completion.

        The runner invokes this from the normal ``finished_req_ids`` cleanup
        step. This notification arrives before another model forward and is the
        first plugin-visible proof that core did not install the proposal.
        """
        if not finished_request_ids:
            return False
        overlap = set(self._published_proposal_owners).intersection(
            finished_request_ids,
        )
        if not overlap:
            return False
        if self._published_proposal_consumed and overlap.intersection(
            self._active_published_proposal_owner_ids,
        ):
            self._release_consumed_proposal()
            return False
        return self._retire_published_proposal_rows(
            overlap,
            reason="terminal",
            terminal=True,
        )

    def _execute_draft(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> torch.Tensor:
        self.validate_prepared_inputs_current(proposal_inputs)
        hidden_states = self._execute_draft_backbone(proposal_inputs)
        result = self._execute_sequential_markov_sampling(proposal_inputs, hidden_states)
        return self._build_core_proposal(proposal_inputs, result)

    def set_eplb_state(self, eplb_state: Any) -> None:
        """Accept the runner's EPLB state after a successful model load."""
        self.eplb_state = eplb_state

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Record the target runner's mode; the draft remains eager."""
        self.requested_cudagraph_mode = cudagraph_mode
        self.cudagraph_mode = CUDAGraphMode.NONE

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        """Skip proposal graph capture because Ascend DSpark is eager-only."""
        return None

    def _profile_markov_heads(
        self,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Execute the real LM/Markov heads without publishing candidates."""
        expected_tokens = num_reqs * self.num_speculative_steps
        self._validate_step_tensor(
            "profile draft hidden states",
            hidden_states,
            ndim=2,
        )
        expected_hidden_shape = (
            expected_tokens,
            int(self.draft_model_config.hf_config.hidden_size),
        )
        if hidden_states.shape != expected_hidden_shape:
            raise RuntimeError(
                "Ascend DSpark profile draft backbone returned shape "
                f"{tuple(hidden_states.shape)} instead of {expected_hidden_shape}."
            )
        if not hidden_states.dtype.is_floating_point:
            raise TypeError("Ascend DSpark profile draft hidden states must be floating point.")

        base_logits = self.model.compute_draft_logits(hidden_states)
        if not isinstance(base_logits, torch.Tensor):
            raise TypeError("Ascend DSpark profile LM head must return a tensor.")
        self._validate_step_tensor("profile LM logits", base_logits, ndim=2)
        vocab_size = int(self.draft_model_config.hf_config.vocab_size)
        expected_logits_shape = (expected_tokens, vocab_size)
        if base_logits.shape != expected_logits_shape:
            raise RuntimeError(
                "Ascend DSpark profile LM head must cover the full vocabulary: "
                f"expected {expected_logits_shape}, got {tuple(base_logits.shape)}."
            )
        if not base_logits.dtype.is_floating_point:
            raise TypeError("Ascend DSpark profile LM logits must be floating point.")
        logical_logits = base_logits.view(
            num_reqs,
            self.num_speculative_steps,
            vocab_size,
        )

        predecessor = anchor_token_ids
        self._validate_step_tensor(
            "profile Markov anchor token IDs",
            predecessor,
            ndim=1,
            dtypes=(torch.int32, torch.int64),
        )
        for step_index in range(self.num_speculative_steps):
            markov_embed = self.model.markov_embed(predecessor)
            if not isinstance(markov_embed, torch.Tensor):
                raise TypeError("Ascend DSpark profile Markov embedding must be a tensor.")
            self._validate_step_tensor(
                "profile Markov embedding",
                markov_embed,
                ndim=2,
            )
            if markov_embed.shape[0] != num_reqs or not markov_embed.dtype.is_floating_point:
                raise RuntimeError(
                    "Ascend DSpark profile Markov embedding must contain one floating-point row per request."
                )
            markov_bias = self.model.markov_bias(markov_embed)
            if not isinstance(markov_bias, torch.Tensor):
                raise TypeError("Ascend DSpark profile Markov head must return a tensor bias.")
            self._validate_step_tensor(
                "profile Markov vocabulary bias",
                markov_bias,
                ndim=2,
            )
            if markov_bias.shape != (num_reqs, vocab_size):
                raise RuntimeError(
                    f"Ascend DSpark profile Markov bias must cover the full vocabulary, got {tuple(markov_bias.shape)}."
                )
            selected = torch.argmax(
                logical_logits[:, step_index, :] + markov_bias,
                dim=-1,
            )
            mapped = self.model.map_draft_to_target(selected)
            if mapped is not selected:
                raise RuntimeError(
                    "Ascend DSpark profile requires the 0731 checkpoint's identity draft-to-target vocabulary mapping."
                )
            self._validate_step_tensor(
                "profile Markov selected token IDs",
                mapped,
                ndim=1,
                dtypes=(torch.int64,),
            )
            predecessor = mapped

    @torch.inference_mode()
    def _profile_draft_execution(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        """Profile the loaded draft without touching proposal lifecycle state.

        Initial memory profiling runs before KV allocation, so the paired core
        runner deliberately supplies no attention metadata or block tables.
        The DSA profile branch consumes ``attn_metadata=None`` while this path
        still executes the real context projection, draft backbone, LM head,
        and every sequential Markov head at the configured batch and K.
        """
        if self._model is None or self._loaded_target_model is None:
            raise RuntimeError("Ascend DSpark models must be loaded before profile execution.")
        if not isinstance(input_batch, InputBatch):
            raise TypeError("Ascend DSpark profile execution requires a V2 InputBatch.")
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_tokens_after_padding = input_batch.num_tokens_after_padding
        if num_reqs <= 0 or num_target_tokens <= 0:
            raise ValueError("Ascend DSpark profile execution requires a non-empty target batch.")
        if num_target_tokens != num_tokens_after_padding:
            raise RuntimeError(
                "Ascend DSpark profile execution requires the unpadded eager "
                "shape produced by the core memory profiler."
            )

        last_hidden_states = self._validate_step_tensor(
            "profile last_hidden_states",
            last_hidden_states,
            ndim=2,
        )
        auxiliary_states = self._validate_aux_hidden_states(
            aux_hidden_states,
            last_hidden_states,
            num_tokens_after_padding,
        )
        target_positions = self._validate_step_tensor(
            "profile target positions",
            input_batch.positions,
            ndim=1,
            dtypes=(torch.int64,),
            min_size=num_target_tokens,
        )
        target_input_ids = self._validate_step_tensor(
            "profile target input IDs",
            input_batch.input_ids,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_target_tokens,
        )
        logits_indices = self._validate_step_tensor(
            "profile logits indices",
            input_batch.logits_indices,
            ndim=1,
            dtypes=(torch.int32, torch.int64),
            min_size=num_reqs,
        )
        seq_lens = self._validate_step_tensor(
            "profile sequence lengths",
            input_batch.seq_lens,
            ndim=1,
            dtypes=(torch.int32,),
            min_size=num_reqs,
        )

        concatenated_aux = torch.cat(
            tuple(state[:num_target_tokens] for state in auxiliary_states),
            dim=-1,
        )
        context_states = self.model.combine_hidden_states(concatenated_aux)
        expected_context_shape = (
            num_target_tokens,
            int(self.draft_model_config.hf_config.hidden_size),
        )
        if not isinstance(context_states, torch.Tensor) or context_states.shape != expected_context_shape:
            actual_shape = tuple(context_states.shape) if isinstance(context_states, torch.Tensor) else None
            raise RuntimeError(
                "Ascend DSpark profile context projection returned shape "
                f"{actual_shape} instead of {expected_context_shape}."
            )
        self._validate_step_tensor(
            "profile combined context states",
            context_states,
            ndim=2,
        )
        if context_states.dtype != auxiliary_states[0].dtype:
            raise RuntimeError(
                "Ascend DSpark profile context states must preserve the "
                f"target auxiliary dtype {auxiliary_states[0].dtype}, got "
                f"{context_states.dtype}."
            )
        self.model.precompute_and_store_context_kv(
            context_states,
            target_positions[:num_target_tokens],
            None,
        )

        active_logits_indices = logits_indices[:num_reqs].to(torch.int64)
        _assert_markov_tensor_contract(
            ((active_logits_indices >= 0) & (active_logits_indices < num_target_tokens)).all(),
            "Ascend DSpark profile logits indices are outside the active target batch.",
        )
        anchor_token_ids = target_input_ids[active_logits_indices]
        num_query_tokens = num_reqs * self.num_speculative_steps
        draft_input_ids = torch.full(
            (num_reqs, self.num_speculative_steps),
            self.parallel_drafting_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        draft_input_ids[:, 0] = anchor_token_ids
        draft_input_ids = draft_input_ids.reshape(num_query_tokens)
        query_offsets = torch.arange(
            self.num_speculative_steps,
            dtype=torch.int64,
            device=self.device,
        )
        draft_positions = (seq_lens[:num_reqs].to(torch.int64)[:, None] + query_offsets).clamp(
            max=int(self.vllm_config.model_config.max_model_len) - 1
        )
        draft_positions = draft_positions.reshape(num_query_tokens)

        batch_descriptor = BatchDescriptor(
            num_tokens=num_query_tokens,
            num_reqs=num_reqs,
            uniform=True,
        )
        with set_forward_context(
            None,
            self.draft_vllm_config,
            num_tokens=num_query_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            batch_descriptor=batch_descriptor,
            slot_mapping=None,
            input_ids=draft_input_ids,
            model_instance=self.model,
        ):
            forward_context = get_forward_context()
            draft_context = build_ascend_forward_context(
                attn_metadata=None,
                vllm_config=self.draft_vllm_config,
                num_tokens=num_query_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                dp_metadata=forward_context.dp_metadata,
                in_profile_run=True,
                num_actual_tokens=num_query_tokens,
                model_instance=self.model,
                is_draft_model=True,
                draft_attn_metadatas=None,
                input_ids=draft_input_ids,
            )
            draft_context["is_draft_model_prefill"] = True
            forward_context.additional_kwargs.update(draft_context)
            hidden_states = self.model(
                input_ids=draft_input_ids,
                positions=draft_positions,
            )

        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("Ascend DSpark profile draft backbone must return a tensor.")
        self._profile_markov_heads(
            hidden_states,
            anchor_token_ids,
            num_reqs,
        )

    def _run_profile_without_proposal_state(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        state = {name: getattr(self, name) for name in _DSPARK_PROFILE_PRESERVED_STATE}
        try:
            self._profile_draft_execution(
                input_batch,
                last_hidden_states,
                aux_hidden_states,
                num_tokens_across_dp,
            )
        finally:
            for name, value in state.items():
                setattr(self, name, value)

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor | None:
        """Publish DSpark proposals using core's optimistic V2 lifecycle."""
        if mm_inputs is not None:
            raise ValueError("Ascend DeepSeek V4 DSpark does not accept multimodal proposal inputs.")
        if dummy_run or is_profile or skip_attn_for_dummy_run:
            if not (dummy_run and is_profile and skip_attn_for_dummy_run):
                raise ValueError(
                    "Ascend DSpark profile execution requires dummy_run=True, "
                    "is_profile=True, and skip_attn_for_dummy_run=True."
                )
            self._run_profile_without_proposal_state(
                input_batch,
                last_hidden_states,
                aux_hidden_states,
                num_tokens_across_dp,
            )
            return None
        if self._next_proposal_skipped:
            dspark_runtime_not_wired("M2.4B multi-round DSpark lifecycle")
        if self._published_candidate_tokens is not None:
            if input_batch.num_draft_tokens == 0:
                # A previous publication may contain only scheduler-proven
                # delayed owners. Do not consume it from an unrelated target
                # batch or overwrite its request-owned candidate rows.
                return None
            if not self.continue_after_verification:
                return self._skip_next_proposal_after_verification(
                    input_batch,
                    num_sampled,
                    num_rejected,
                    temperature,
                )
            self._consume_published_proposal_after_verification(
                input_batch,
                num_sampled,
                num_rejected,
                temperature,
            )
            self._release_consumed_proposal()
        proposal_inputs = self.prepare_proposal_inputs(
            input_batch=input_batch,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings,
            last_hidden_states=last_hidden_states,
            aux_hidden_states=aux_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        return self._execute_draft(proposal_inputs)
