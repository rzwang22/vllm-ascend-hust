# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import MappingProxyType
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
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

from vllm_ascend.spec_decode import dspark_runtime_not_wired
from vllm_ascend.worker.v2.spec_decode.dspark.proposal_inputs import (
    AscendDSparkProposalInputs,
)


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

    def _execute_draft(
        self,
        proposal_inputs: AscendDSparkProposalInputs,
    ) -> torch.Tensor:
        self.validate_prepared_inputs_current(proposal_inputs)
        dspark_runtime_not_wired("V2 draft execution")

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
        """Prepare real per-step inputs, then fail before draft execution."""
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
