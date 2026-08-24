# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

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
    AscendDSparkProposalInputs,
)


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
        self._draft_cache_isolation_audit: MappingProxyType[str, int] | None = None
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
            self.vllm_config,
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
            self.vllm_config,
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
                vllm_config=self.vllm_config,
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

    def _execute_draft(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> torch.Tensor:
        self.validate_prepared_inputs_current(proposal_inputs)
        self._execute_draft_backbone(proposal_inputs)
        dspark_runtime_not_wired("V2 DSpark Markov sampling")

    def set_eplb_state(self, eplb_state: Any) -> None:
        """Accept the runner's EPLB state after a successful model load."""
        self.eplb_state = eplb_state

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Record the requested mode while keeping DSpark execution eager."""
        self.requested_cudagraph_mode = cudagraph_mode
        self.cudagraph_mode = CUDAGraphMode.NONE

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        """Skip proposal graph capture because Ascend DSpark is eager-only."""
        return None

    def propose(
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
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        """Run the real draft backbone, then fail before Markov sampling."""
        if dummy_run or is_profile:
            dspark_runtime_not_wired("V2 draft execution (dummy/profile)")
        if skip_attn_for_dummy_run:
            raise ValueError("skip_attn_for_dummy_run is only valid for a DSpark dummy run.")
        if mm_inputs is not None:
            raise ValueError("Ascend DeepSeek V4 DSpark does not accept multimodal proposal inputs.")
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
