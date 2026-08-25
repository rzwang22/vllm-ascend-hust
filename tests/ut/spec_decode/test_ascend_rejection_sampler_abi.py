# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as core_rejection_sampler
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as core_rejection_utils

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.worker.v2.spec_decode import rejection_sampler_utils as npu_rejection_utils

BLOCK_UNSUPPORTED = "Ascend V2 rejection sampler does not support block verification"


class _KernelBoundary:
    def __init__(self, name: str, calls: list[tuple[str, object, dict[str, object]]]) -> None:
        self.name = name
        self.calls = calls

    def __getitem__(self, grid):
        def launch(*args, **kwargs) -> None:
            self.calls.append((self.name, grid, kwargs))
            if self.name == "probabilistic":
                args[0].fill_(7)
                args[2].fill_(1)

        return launch


def _guard_arguments() -> tuple[None, None, None, None, None, None, None, None, None, None, int]:
    return (None, None, None, None, None, None, None, None, None, None, 1)


def test_patch_installs_one_npu_function_with_core_compatible_signature() -> None:
    patch_module = importlib.import_module("vllm_ascend.patch.worker.patch_v2.patch_triton")

    importlib.reload(core_rejection_utils)
    importlib.reload(core_rejection_sampler)
    core_function_before_patch = core_rejection_utils.rejection_sample
    caller_function_before_patch = core_rejection_sampler.rejection_sample

    assert core_function_before_patch is caller_function_before_patch
    assert core_function_before_patch is not npu_rejection_utils.rejection_sample

    importlib.reload(patch_module)

    assert core_rejection_utils.rejection_sample is npu_rejection_utils.rejection_sample
    assert core_rejection_sampler.rejection_sample is npu_rejection_utils.rejection_sample
    core_parameters = inspect.signature(core_function_before_patch).parameters
    npu_parameters = inspect.signature(npu_rejection_utils.rejection_sample).parameters
    assert set(core_parameters) <= set(npu_parameters)
    assert npu_parameters["use_block_verification"].default is False


def test_standard_greedy_one_hot_draft_reaches_existing_kernel_boundary(monkeypatch) -> None:
    calls: list[tuple[str, object, dict[str, object]]] = []
    monkeypatch.setattr(
        npu_rejection_utils,
        "triton",
        SimpleNamespace(
            cdiv=lambda dividend, divisor: (dividend + divisor - 1) // divisor,
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        ),
    )
    monkeypatch.setattr(
        npu_rejection_utils,
        "_compute_block_stats_kernel",
        _KernelBoundary("stats", calls),
    )
    monkeypatch.setattr(
        npu_rejection_utils,
        "_probabilistic_rejection_kernel",
        _KernelBoundary("probabilistic", calls),
    )
    monkeypatch.setattr(npu_rejection_utils, "_resample_kernel", _KernelBoundary("resample", calls))
    monkeypatch.setattr(
        npu_rejection_utils,
        "_insert_resampled_kernel",
        _KernelBoundary("insert", calls),
    )

    sampled, num_sampled = npu_rejection_utils.rejection_sample(
        target_logits=torch.zeros((2, 16), dtype=torch.float32),
        draft_logits=None,
        draft_sampled=torch.tensor([3, 4], dtype=torch.int64),
        cu_num_logits=torch.tensor([0, 2], dtype=torch.int32),
        pos=torch.tensor([0, 1], dtype=torch.int64),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([0, 0], dtype=torch.int32),
        expanded_local_pos=torch.tensor([0, 1], dtype=torch.int32),
        temperature=torch.tensor([0.0], dtype=torch.float32),
        seed=torch.tensor([1], dtype=torch.int64),
        num_speculative_steps=1,
        use_block_verification=False,
    )

    assert [call[0] for call in calls] == ["stats", "probabilistic", "resample", "insert"]
    assert all(call[2].get("HAS_DRAFT_LOGITS") is False for call in calls[:3])
    assert sampled.shape == (1, 2)
    assert sampled.dtype is torch.int64
    assert num_sampled.shape == (1,)
    assert num_sampled.dtype is torch.int32


def test_direct_block_verification_guard_precedes_kernel_launch(monkeypatch) -> None:
    def fail_if_launched(*args, **kwargs):
        raise AssertionError("NPU kernel boundary must not be reached")

    monkeypatch.setattr(npu_rejection_utils, "_compute_block_stats_kernel", fail_if_launched)

    with pytest.raises(NotImplementedError, match=BLOCK_UNSUPPORTED):
        npu_rejection_utils.rejection_sample(*_guard_arguments(), use_block_verification=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_fp64": True}, "FP64 rejection sampling is not supported on NPU"),
        (
            {"synthetic_conditional_rates": object()},
            "Synthetic rejection sampling is not supported on NPU yet",
        ),
    ],
)
def test_existing_unsupported_modes_remain_fail_closed(kwargs, message) -> None:
    with pytest.raises(NotImplementedError, match=message):
        npu_rejection_utils.rejection_sample(*_guard_arguments(), **kwargs)


def test_platform_rejects_block_and_accepts_standard_or_no_speculation() -> None:
    block_config = SimpleNamespace(
        device_config=SimpleNamespace(device_type="npu"),
        speculative_config=SimpleNamespace(rejection_sample_method="block"),
        model_config=None,
    )
    with pytest.raises(NotImplementedError, match=BLOCK_UNSUPPORTED):
        NPUPlatform.check_and_update_config(block_config)

    for speculative_config in (
        None,
        SimpleNamespace(rejection_sample_method="standard"),
        SimpleNamespace(rejection_sample_method="synthetic"),
    ):
        config = SimpleNamespace(
            device_config=SimpleNamespace(device_type="npu"),
            speculative_config=speculative_config,
            model_config=None,
        )
        NPUPlatform.check_and_update_config(config)
