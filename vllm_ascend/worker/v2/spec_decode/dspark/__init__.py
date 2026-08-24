# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.config import VllmConfig

from vllm_ascend.worker.v2.spec_decode.dspark.proposal_inputs import (
    AscendDSparkDraftExecution,
    AscendDSparkProposalInputs,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)


def create_dspark_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
) -> AscendDSparkSpeculator:
    """Construct the Ascend V2 DSpark speculator."""
    return AscendDSparkSpeculator(vllm_config, device)


__all__ = [
    "AscendDSparkDraftExecution",
    "AscendDSparkProposalInputs",
    "AscendDSparkSpeculator",
    "create_dspark_speculator",
]
