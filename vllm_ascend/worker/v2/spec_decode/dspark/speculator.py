# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch
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

        # This helper only performs config replacement, ModelRegistry loading,
        # PP validation, and target embedding/lm-head sharing. It does not
        # import or invoke the core CUDA/Triton DSpark speculator runtime.
        from vllm.v1.worker.gpu.spec_decode.dspark.utils import (
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

        # Commit state only after construction, checkpoint loading, sharing,
        # and implementation validation all succeed.
        self._loaded_target_model = target_model
        self._model = draft_model

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
        boundary = "V2 dummy-run proposal" if dummy_run else "V2 proposal"
        dspark_runtime_not_wired(boundary)
