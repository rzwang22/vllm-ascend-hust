# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import dsa
from vllm_ascend.utils import AscendDeviceType


def _tensor(dtype: torch.dtype) -> torch.Tensor:
    # A non-singleton leading shape makes the old cache[0][i] implementation
    # return tensor slices instead of the registered cache components.
    return torch.empty((2, 3), dtype=dtype)


def _dsa_owner(indexer_cache) -> tuple[SimpleNamespace, dict[str, torch.Tensor]]:
    caches = {
        "compress": _tensor(torch.bfloat16),
        "swa": _tensor(torch.bfloat16),
        "state": _tensor(torch.float32),
        "indexer_state": _tensor(torch.float32),
    }
    owner = SimpleNamespace(
        compress_ratio=4,
        dsa_attn=SimpleNamespace(kv_cache=[caches["compress"]]),
        swa_cache_layer=SimpleNamespace(kv_cache=[caches["swa"]]),
        compressor=SimpleNamespace(
            state_cache=SimpleNamespace(kv_cache=[caches["state"]]),
        ),
        indexer=SimpleNamespace(
            compressor=SimpleNamespace(
                state_cache=SimpleNamespace(kv_cache=[caches["indexer_state"]]),
            ),
            k_cache=SimpleNamespace(kv_cache=indexer_cache),
        ),
    )
    return owner, caches


@pytest.mark.parametrize("v1_outer_container", [False, True])
def test_non_a5_indexer_cache_components_preserve_binder_identity(
    monkeypatch: pytest.MonkeyPatch,
    v1_outer_container: bool,
) -> None:
    k_cache = _tensor(torch.int8)
    scale_cache = _tensor(torch.float16)
    direct_container = [k_cache, scale_cache]
    container = [direct_container] if v1_outer_container else direct_container
    owner, other_caches = _dsa_owner(container)
    monkeypatch.setattr(dsa, "get_ascend_device_type", lambda: AscendDeviceType.A2)

    result = dsa._build_kv_cache(owner, SimpleNamespace())

    assert result[0] is other_caches["compress"]
    assert result[1] is other_caches["swa"]
    assert result[2] is other_caches["state"]
    assert result[3] is other_caches["indexer_state"]
    assert result[4] is k_cache
    assert result[5] is scale_cache
    assert result[4].dtype is torch.int8
    assert result[5].dtype is torch.float16


@pytest.mark.parametrize("v1_outer_container", [False, True])
def test_a5_indexer_cache_components_preserve_binder_identity(
    monkeypatch: pytest.MonkeyPatch,
    v1_outer_container: bool,
) -> None:
    k_cache = _tensor(torch.int8)
    scale_cache = _tensor(torch.float32)
    full_cache = _tensor(torch.int8)
    direct_container = [k_cache, scale_cache, full_cache]
    container = [direct_container] if v1_outer_container else direct_container
    owner, other_caches = _dsa_owner(container)
    monkeypatch.setattr(dsa, "get_ascend_device_type", lambda: AscendDeviceType.A5)

    result = dsa._build_kv_cache(owner, SimpleNamespace())

    assert result[0] is other_caches["compress"]
    assert result[1] is other_caches["swa"]
    assert result[2] is other_caches["state"]
    assert result[3] is other_caches["indexer_state"]
    assert result[4] is k_cache
    assert result[5] is scale_cache
    assert result[6] is full_cache
