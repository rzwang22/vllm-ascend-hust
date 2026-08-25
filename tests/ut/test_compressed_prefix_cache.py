# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHashListWithBlockSize,
    get_block_hash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.core.single_type_kv_cache_manager import CompressAttentionManager
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash_seed():
    init_none_hash(sha256)


def _make_request(request_id: str, token_ids: list[int], hash_block_size: int) -> Request:
    sampling_params = SamplingParams(max_tokens=1)
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def _make_compress_manager(
    block_size: int = 128,
    compress_ratio: int = 4,
) -> tuple[AscendMLAAttentionSpec, BlockPool, CompressAttentionManager]:
    spec = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )
    block_pool = BlockPool(
        num_gpu_blocks=8,
        enable_caching=True,
        hash_block_size=block_size,
    )
    manager = CompressAttentionManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )
    return spec, block_pool, manager


def _make_hybrid_config(
    compress_ratios: tuple[int, ...],
    *,
    block_size: int = 128,
    num_blocks: int = 32,
) -> KVCacheConfig:
    groups = [
        KVCacheGroupSpec(
            [f"ratio-{compress_ratio}"],
            AscendMLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                compress_ratio=compress_ratio,
                model_version="deepseek_v4",
            ),
        )
        for compress_ratio in compress_ratios
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def _make_hybrid_coordinator(
    compress_ratios: tuple[int, ...],
    *,
    block_size: int = 128,
    num_blocks: int = 32,
    use_eagle: bool = False,
) -> AscendHybridKVCacheCoordinator:
    return AscendHybridKVCacheCoordinator(
        kv_cache_config=_make_hybrid_config(
            compress_ratios,
            block_size=block_size,
            num_blocks=num_blocks,
        ),
        max_model_len=block_size * max(compress_ratios),
        use_eagle=use_eagle,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=block_size,
        max_num_batched_tokens=block_size * max(compress_ratios),
        scheduler_block_size=block_size,
    )


def test_compressed_prefix_cache_uses_logical_block_hash() -> None:
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio
    spec, block_pool, manager = _make_compress_manager(block_size, compress_ratio)

    request_a_tokens = list(range(logical_block_size))
    request_b_tokens = request_a_tokens.copy()
    request_b_tokens[block_size + 7] = 999_999

    request_a = _make_request("a", request_a_tokens, block_size)
    request_b = _make_request("b", request_b_tokens, block_size)

    manager.allocate_new_blocks(
        request_a.request_id,
        num_tokens=logical_block_size,
        num_tokens_main_model=logical_block_size,
    )
    manager.cache_blocks(request_a, num_tokens=logical_block_size)

    cached_hash = get_block_hash(manager.req_to_blocks[request_a.request_id][0].block_hash)
    expected_hash = BlockHashListWithBlockSize(
        request_a.block_hashes,
        block_size,
        logical_block_size,
    )[0]
    assert cached_hash == expected_hash

    hit_blocks = CompressAttentionManager.find_longest_cache_hit(
        block_hashes=request_b.block_hashes,
        max_length=logical_block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=logical_block_size,
    )[0]

    assert hit_blocks == []


def test_compressed_prefix_cache_hits_identical_logical_block() -> None:
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio
    spec, block_pool, manager = _make_compress_manager(block_size, compress_ratio)

    request = _make_request("a", list(range(logical_block_size)), block_size)
    manager.allocate_new_blocks(
        request.request_id,
        num_tokens=logical_block_size,
        num_tokens_main_model=logical_block_size,
    )
    manager.cache_blocks(request, num_tokens=logical_block_size)

    hit_blocks = CompressAttentionManager.find_longest_cache_hit(
        block_hashes=request.block_hashes,
        max_length=logical_block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        drop_eagle_block=False,
        alignment_tokens=logical_block_size,
    )[0]

    assert hit_blocks == manager.req_to_blocks[request.request_id]


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_compressed_cache_blocks_returns_new_physical_block_count(compress_ratio: int) -> None:
    block_size = 128
    logical_block_size = block_size * compress_ratio
    _spec, _block_pool, manager = _make_compress_manager(
        block_size,
        compress_ratio,
    )
    request = _make_request(
        f"ratio-{compress_ratio}",
        list(range(logical_block_size)),
        block_size,
    )
    manager.allocate_new_blocks(
        request.request_id,
        num_tokens=logical_block_size,
        num_tokens_main_model=logical_block_size,
    )

    assert (
        manager.cache_blocks(
            request,
            num_tokens=logical_block_size,
            retention_interval=logical_block_size,
        )
        == 1
    )
    assert manager.num_cached_block[request.request_id] == 1
    assert manager.req_to_blocks[request.request_id][0].block_hash is not None
    assert manager.cache_blocks(request, num_tokens=logical_block_size) == 0


def test_hybrid_cache_blocks_zero_aligned_calls_every_manager(monkeypatch) -> None:
    block_size = 128
    coordinator = _make_hybrid_coordinator((1, 4, 128))
    request = _make_request("zero-aligned", [1], block_size)
    calls = []

    assert len(coordinator.single_type_managers) == 3
    assert all(isinstance(manager, CompressAttentionManager) for manager in coordinator.single_type_managers)
    for manager in coordinator.single_type_managers:
        cache_blocks_impl = manager.cache_blocks

        def record_cache_blocks(
            actual_request,
            num_tokens,
            retention_interval=None,
            *,
            alignment_tokens=None,
            _manager=manager,
            _cache_blocks_impl=cache_blocks_impl,
        ):
            calls.append(
                (
                    _manager,
                    actual_request,
                    num_tokens,
                    retention_interval,
                    alignment_tokens,
                )
            )
            return _cache_blocks_impl(
                actual_request,
                num_tokens,
                retention_interval=retention_interval,
                alignment_tokens=alignment_tokens,
            )

        monkeypatch.setattr(manager, "cache_blocks", record_cache_blocks)

    assert coordinator.cache_blocks(request, num_computed_tokens=1) == 0
    assert len(calls) == 3
    assert all(call[1] is request for call in calls)
    assert all(call[2:] == (0, None, None) for call in calls)
    assert all(manager.num_cached_block.get(request.request_id, 0) == 0 for manager in coordinator.single_type_managers)


def test_hybrid_cache_blocks_positive_aligned_returns_max_new_blocks() -> None:
    block_size = 128
    num_tokens = block_size * 4
    coordinator = _make_hybrid_coordinator((1, 4), num_blocks=16)
    request = _make_request(
        "positive-aligned",
        list(range(num_tokens)),
        block_size,
    )
    for manager in coordinator.single_type_managers:
        manager.allocate_new_blocks(
            request.request_id,
            num_tokens=num_tokens,
            num_tokens_main_model=num_tokens,
        )

    assert coordinator.cache_blocks(request, num_tokens) == 4
    assert [manager.num_cached_block[request.request_id] for manager in coordinator.single_type_managers] == [4, 1]
    assert all(
        block.block_hash is not None
        for manager in coordinator.single_type_managers
        for block in manager.req_to_blocks[request.request_id]
    )


def test_hybrid_cache_blocks_preserves_eagle_and_retention_arguments() -> None:
    class RecordingManager:
        def __init__(self, *, use_eagle: bool, result: int) -> None:
            self.use_eagle = use_eagle
            self.block_size = 128
            self.result = result
            self.calls = []

        def cache_blocks(self, request, num_tokens, retention_interval=None):
            self.calls.append((request, num_tokens, retention_interval))
            return self.result

    request = _make_request("eagle-retention", list(range(129)), 128)
    eagle_manager = RecordingManager(use_eagle=True, result=1)
    regular_manager = RecordingManager(use_eagle=False, result=0)
    coordinator = AscendHybridKVCacheCoordinator.__new__(AscendHybridKVCacheCoordinator)
    coordinator.scheduler_block_size = 128
    coordinator.retention_interval = 256
    coordinator.single_type_managers = (eagle_manager, regular_manager)

    assert coordinator.cache_blocks(request, 129) == 1
    assert eagle_manager.calls == [(request, 129, 256)]
    assert regular_manager.calls == [(request, 128, 256)]


def test_scheduler_allocation_boundary_handles_zero_aligned_hybrid_cache() -> None:
    block_size = 128
    manager = KVCacheManager(
        kv_cache_config=_make_hybrid_config((1, 4), num_blocks=16),
        max_model_len=block_size * 4,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        max_num_batched_tokens=block_size * 4,
        enable_caching=True,
        watermark=0,
    )
    request = _make_request("scheduler-boundary", [11, 12], block_size)
    request.num_computed_tokens = 1

    allocated = manager.allocate_slots(
        request,
        num_new_tokens=1,
        has_scheduled_reqs=False,
    )

    assert isinstance(manager.coordinator, AscendHybridKVCacheCoordinator)
    assert allocated is not None
    assert all(
        cache_manager.num_cached_block.get(request.request_id, 0) == 0
        for cache_manager in manager.coordinator.single_type_managers
    )


def test_hybrid_coordinator_rejects_partial_compressed_prefix_hit() -> None:
    block_size = 128
    compress_ratio = 4
    logical_block_size = block_size * compress_ratio
    request_a_tokens = list(range(logical_block_size))
    request_b_tokens = request_a_tokens.copy()
    request_b_tokens[block_size + 7] = 999_999

    request_a = _make_request("a", request_a_tokens, block_size)
    request_b = _make_request("b", request_b_tokens, block_size)
    compressed_spec = AscendMLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=compress_ratio,
        model_version="deepseek_v4",
    )
    full_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    coordinator = AscendHybridKVCacheCoordinator(
        kv_cache_config=KVCacheConfig(
            num_blocks=16,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["compressed"], compressed_spec),
                KVCacheGroupSpec(["full"], full_spec),
            ],
        ),
        max_model_len=logical_block_size,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=block_size,
        max_num_batched_tokens=logical_block_size,
        scheduler_block_size=block_size,
    )

    for manager in coordinator.single_type_managers:
        manager.allocate_new_blocks(
            request_a.request_id,
            num_tokens=logical_block_size,
            num_tokens_main_model=logical_block_size,
        )
        manager.cache_blocks(request_a, num_tokens=logical_block_size)

    hit_blocks, hit_length = coordinator.find_longest_cache_hit(
        request_b.block_hashes,
        max_cache_hit_length=logical_block_size,
    )

    assert hit_length == 0
    assert hit_blocks == ([], [])
