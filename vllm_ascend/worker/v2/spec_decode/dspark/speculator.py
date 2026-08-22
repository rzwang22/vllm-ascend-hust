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

        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = speculative_config
        self.draft_model_config = draft_model_config
        self.num_speculative_steps = num_speculative_steps
        self.parallel_drafting_token_id = parallel_drafting_token_id
        self._model: torch.nn.Module | None = None
        self._loaded_target_model: torch.nn.Module | None = None
        self.target_attn_layer_names: frozenset[str] | None = None
        self.draft_attn_layer_names: frozenset[str] | None = None
        self.draft_kv_cache_specs: MappingProxyType[str, KVCacheSpec] | None = None
        self.draft_kv_cache_group_ids: tuple[int, ...] = ()
        self.draft_kv_caches: MappingProxyType[str, Any] | None = None
        self.model_state: ModelState | None = None
        self.kv_cache_config: KVCacheConfig | None = None
        self.block_tables: BlockTables | None = None
        self.attn_groups: list[list[Any]] | None = None
        self.attn_backends: MappingProxyType[str, type[AttentionBackend]] | None = None
        self._kv_cache_signature: tuple[Any, ...] | None = None
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

        # Publish only after every layer, tensor, group and backend is valid.
        draft_kv_cache_view = MappingProxyType(draft_kv_caches)
        attn_backend_view = MappingProxyType(attn_backends)
        cache_signature = self._cache_config_signature(kv_cache_config)
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = attn_groups
        self.attn_backends = attn_backend_view
        self.draft_kv_cache_group_ids = active_group_ids
        self.draft_kv_caches = draft_kv_cache_view
        self._kv_cache_signature = cache_signature

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
        """Fail closed where DSpark metadata and draft forward are required."""
        boundary = "V2 dummy-run metadata/proposal" if dummy_run else "V2 metadata/proposal"
        dspark_runtime_not_wired(boundary)
