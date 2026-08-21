# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from vllm.config import VllmConfig
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

SpeculatorFactory = Callable[[VllmConfig, torch.device], BaseSpeculator]


def _uses_dspark(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    return speculative_config is not None and speculative_config.method == "dspark"


@contextmanager
def override_core_dspark_speculator_factory(
    vllm_config: VllmConfig,
    ascend_factory: SpeculatorFactory,
) -> Iterator[None]:
    """Let GPUModelRunner construct the Ascend DSpark speculator directly."""
    if not _uses_dspark(vllm_config):
        yield
        return

    core_factory = vllm_model_runner.init_speculator
    try:
        vllm_model_runner.init_speculator = ascend_factory
        yield
    finally:
        vllm_model_runner.init_speculator = core_factory


@contextmanager
def include_ascend_dspark_in_core_load_lifecycle(
    vllm_config: VllmConfig,
) -> Iterator[None]:
    """Let core's existing post-target load point call the Ascend loader."""
    if not _uses_dspark(vllm_config):
        yield
        return

    from vllm_ascend.worker.v2.spec_decode.dspark import (
        AscendDSparkSpeculator,
    )

    core_draft_speculator_type = vllm_model_runner.DraftModelSpeculator
    try:
        vllm_model_runner.DraftModelSpeculator = (
            core_draft_speculator_type,
            AscendDSparkSpeculator,
        )
        yield
    finally:
        vllm_model_runner.DraftModelSpeculator = core_draft_speculator_type


def initialize_ascend_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
    core_initialized_speculator: BaseSpeculator | None,
    ascend_factory: SpeculatorFactory,
) -> BaseSpeculator | None:
    """Reuse core initialization for DSpark and preserve other Ascend paths."""
    if vllm_config.speculative_config is None:
        return None

    if _uses_dspark(vllm_config):
        from vllm_ascend.worker.v2.spec_decode.dspark import (
            AscendDSparkSpeculator,
        )

        if not isinstance(core_initialized_speculator, AscendDSparkSpeculator):
            raise RuntimeError(
                "GPUModelRunner did not initialize the required AscendDSparkSpeculator for method='dspark'."
            )
        return core_initialized_speculator

    return ascend_factory(vllm_config, device)
