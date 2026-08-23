# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type

bootstrap_custom_op_env(include_vendor_lib=True)
from vllm_ascend import vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.worker.v2.attn_utils import _adjust_dsv4_kv_layout  # noqa: E402

BLOCK_SIZE = 128
HEAD_DIM = 512
NUM_Q_HEADS = 64
NUM_KV_HEADS = 1
SOFTMAX_SCALE = 1.0 / sqrt(HEAD_DIM)
DEFAULT_WIN_LEFT = 127
DEFAULT_WIN_RIGHT = 0


@dataclass(frozen=True)
class SharedKVAddressDiagnosticCase:
    label: str
    block_size: int
    physical_block: int
    seqused_kv: int
    dtype: torch.dtype
    use_typed_backing: bool = True
    use_scatter: bool = True
    provide_cu_seqlens_ori_kv: bool = False
    provide_ori_sparse_indices: bool = False


SHAREDKV_ABI_A0 = SharedKVAddressDiagnosticCase("A0", 32, 0, 1, torch.bfloat16)
SHAREDKV_ABI_A1 = SharedKVAddressDiagnosticCase("A1", 32, 0, 1, torch.bfloat16, provide_cu_seqlens_ori_kv=True)
SHAREDKV_ABI_A2 = SharedKVAddressDiagnosticCase(
    "A2",
    32,
    0,
    1,
    torch.bfloat16,
    use_typed_backing=False,
    provide_cu_seqlens_ori_kv=True,
)
SHAREDKV_ABI_A3 = SharedKVAddressDiagnosticCase(
    "A3",
    32,
    0,
    1,
    torch.bfloat16,
    use_typed_backing=False,
    use_scatter=False,
    provide_cu_seqlens_ori_kv=True,
)
SHAREDKV_ABI_A5 = SharedKVAddressDiagnosticCase(
    "A5",
    32,
    0,
    1,
    torch.bfloat16,
    provide_cu_seqlens_ori_kv=True,
    provide_ori_sparse_indices=True,
)
SHAREDKV_ADDRESS_B = SharedKVAddressDiagnosticCase("B", 32, 1, 16, torch.bfloat16)
SHAREDKV_ADDRESS_C = SharedKVAddressDiagnosticCase("C", 32, 1, 32, torch.bfloat16)
SHAREDKV_ADDRESS_D = SharedKVAddressDiagnosticCase("D", 128, 1, 1, torch.bfloat16)
SHAREDKV_ADDRESS_E = SharedKVAddressDiagnosticCase("E", 32, 1, 1, torch.float16)

SHAREDKV_S2_BOUNDARY_CASES = tuple(
    pytest.param(
        SharedKVAddressDiagnosticCase(f"S2-{seqused_kv}", 32, 1, seqused_kv, torch.bfloat16),
        id=f"s2-{seqused_kv}",
    )
    for seqused_kv in (1, 8, 15, 16, 17, 31, 32, 33)
)


class _SparseAttnSharedkvSwaParams(ctypes.Structure):
    _fields_ = [
        ("batchSize", ctypes.c_uint32),
        ("qSeqSize", ctypes.c_uint32),
        ("kvSeqSize", ctypes.c_uint32),
        ("paBlockSize", ctypes.c_int64),
        ("oriBlockSize", ctypes.c_int64),
        ("cmpBlockSize", ctypes.c_int64),
        ("oriMaxBlockNumPerBatch", ctypes.c_uint32),
        ("nNumOfQInOneGroup", ctypes.c_uint32),
        ("actualLenDimsQ", ctypes.c_uint32),
        ("actualLenDimsKV", ctypes.c_uint32),
        ("softmaxScale", ctypes.c_float),
        ("outputLayout", ctypes.c_uint32),
        ("oriMaskMode", ctypes.c_uint64),
        ("oriKvStride", ctypes.c_int64),
        ("oriWinLeft", ctypes.c_int64),
        ("oriWinRight", ctypes.c_int64),
        ("sparseBlockSize", ctypes.c_int64),
        ("hasOriSparseIndices", ctypes.c_bool),
        ("oriSparseIndexWidth", ctypes.c_uint32),
        ("usedCoreNum", ctypes.c_uint32),
        ("mmResUbSize", ctypes.c_uint32),
        ("bmm2ResUbSize", ctypes.c_uint32),
        ("mBaseSize", ctypes.c_uint32),
        ("s2BaseSize", ctypes.c_uint32),
        ("returnSoftmaxLse", ctypes.c_bool),
    ]


class _SparseAttnSharedkvCmpParams(ctypes.Structure):
    _fields_ = [
        ("cmpMaxBlockNumPerBatch", ctypes.c_uint32),
        ("sparseBlockCount", ctypes.c_uint32),
        ("cmpRatio", ctypes.c_int64),
        ("cmpMaskMode", ctypes.c_uint64),
        ("cmpKvStride", ctypes.c_int64),
    ]


class _SparseAttnSharedkvTilingData(ctypes.Structure):
    _fields_ = [
        ("baseParams", _SparseAttnSharedkvSwaParams),
        ("cmpParams", _SparseAttnSharedkvCmpParams),
    ]


@dataclass
class SparseAttnCase:
    q: torch.Tensor
    ori_kv: torch.Tensor
    ori_block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor | None
    seqused_kv: torch.Tensor
    sinks: torch.Tensor
    metadata: torch.Tensor
    q_cpu: torch.Tensor
    ori_kv_cpu: torch.Tensor
    block_table_cpu: torch.Tensor
    layout_q: str
    kv_tokens: int
    win_left: int


def _make_case(
    *,
    q_tokens: int = 1,
    kv_tokens: int = 512,
    layout_q: str = "TND",
    win_left: int = DEFAULT_WIN_LEFT,
) -> SparseAttnCase:
    assert kv_tokens % BLOCK_SIZE == 0
    num_blocks = kv_tokens // BLOCK_SIZE
    generator = torch.Generator().manual_seed(20260820 + q_tokens + kv_tokens)
    q_shape = (q_tokens, NUM_Q_HEADS, HEAD_DIM)
    if layout_q == "BSND":
        q_shape = (1, q_tokens, NUM_Q_HEADS, HEAD_DIM)
    q_cpu = torch.randn(q_shape, dtype=torch.float32, generator=generator).mul_(0.1).to(torch.float16)
    ori_kv_cpu = (
        torch.randn(
            (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM),
            dtype=torch.float32,
            generator=generator,
        )
        .mul_(0.1)
        .to(torch.float16)
    )

    block_ids = torch.arange(num_blocks, dtype=torch.int32)
    if num_blocks > 1:
        block_ids = torch.roll(block_ids, shifts=1)
    block_table_cpu = block_ids.unsqueeze(0)

    q = q_cpu.npu()
    ori_kv = ori_kv_cpu.npu()
    ori_block_table = block_table_cpu.npu()
    cu_seqlens_q = None
    if layout_q == "TND":
        cu_seqlens_q = torch.tensor([0, q_tokens], dtype=torch.int32).npu()
    seqused_kv = torch.tensor([kv_tokens], dtype=torch.int32).npu()
    sinks = torch.full((NUM_Q_HEADS,), -10000.0, dtype=torch.float32).npu()

    metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=NUM_Q_HEADS,
        num_heads_kv=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=1,
        max_seqlen_q=q_tokens,
        max_seqlen_kv=kv_tokens,
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=win_left,
        ori_win_right=DEFAULT_WIN_RIGHT,
        layout_q=layout_q,
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device="npu:0",
    )
    return SparseAttnCase(
        q=q,
        ori_kv=ori_kv,
        ori_block_table=ori_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        q_cpu=q_cpu,
        ori_kv_cpu=ori_kv_cpu,
        block_table_cpu=block_table_cpu,
        layout_q=layout_q,
        kv_tokens=kv_tokens,
        win_left=win_left,
    )


def _run(case: SparseAttnCase, ori_sparse_indices: torch.Tensor | None) -> torch.Tensor:
    out, _ = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
        case.q,
        ori_kv=case.ori_kv,
        cmp_kv=None,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=None,
        ori_block_table=case.ori_block_table,
        cmp_block_table=None,
        cu_seqlens_q=case.cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=case.seqused_kv,
        sinks=case.sinks,
        metadata=case.metadata,
        softmax_scale=SOFTMAX_SCALE,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=case.win_left,
        ori_win_right=DEFAULT_WIN_RIGHT,
        layout_q=case.layout_q,
        layout_kv="PA_ND",
        return_softmax_lse=False,
    )
    return out


def _physical_slots_for_logical_positions(case: SparseAttnCase, logical_positions: torch.Tensor) -> torch.Tensor:
    logical_blocks = torch.div(logical_positions, BLOCK_SIZE, rounding_mode="floor")
    block_offsets = logical_positions.remainder(BLOCK_SIZE)
    physical_blocks = case.block_table_cpu[0, logical_blocks].to(torch.int64)
    return physical_blocks * BLOCK_SIZE + block_offsets


def _reference_attention(case: SparseAttnCase, physical_slots: torch.Tensor) -> torch.Tensor:
    q = case.q_cpu.reshape(-1, NUM_Q_HEADS, HEAD_DIM)[0].float()
    flat_kv = case.ori_kv_cpu.reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    selected_kv = flat_kv[physical_slots.to(torch.int64), 0].float()
    scores = torch.matmul(q, selected_kv.transpose(0, 1)) * SOFTMAX_SCALE
    return torch.matmul(torch.softmax(scores, dim=-1), selected_kv)


def _assert_matches_reference(out: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(out.reshape(-1, NUM_Q_HEADS, HEAD_DIM)[0].float().cpu(), expected, atol=3e-2, rtol=3e-2)


def test_none_path_matches_contiguous_window_reference() -> None:
    case = _make_case(kv_tokens=256)
    logical_positions = torch.arange(case.kv_tokens - BLOCK_SIZE, case.kv_tokens, dtype=torch.int64)
    physical_slots = _physical_slots_for_logical_positions(case, logical_positions)

    out = _run(case, ori_sparse_indices=None)

    _assert_matches_reference(out, _reference_attention(case, physical_slots))


def test_contiguous_physical_indices_match_none_path() -> None:
    case = _make_case(kv_tokens=256)
    logical_positions = torch.arange(case.kv_tokens - BLOCK_SIZE, case.kv_tokens, dtype=torch.int64)
    physical_slots = _physical_slots_for_logical_positions(case, logical_positions)
    ori_sparse_indices = physical_slots.to(torch.int32).reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    none_out = _run(case, ori_sparse_indices=None)
    indexed_out = _run(case, ori_sparse_indices=ori_sparse_indices)

    torch.testing.assert_close(indexed_out.float().cpu(), none_out.float().cpu(), atol=3e-2, rtol=3e-2)
    _assert_matches_reference(indexed_out, _reference_attention(case, physical_slots))


def test_discrete_physical_indices_match_pytorch_reference() -> None:
    index_width = 256
    case = _make_case(kv_tokens=512, win_left=index_width - 1)
    physical_slots = (torch.arange(index_width, dtype=torch.int64) * 193 + 17).remainder(case.kv_tokens)
    ori_sparse_indices = physical_slots.to(torch.int32).reshape(1, NUM_KV_HEADS, index_width).npu()

    out = _run(case, ori_sparse_indices=ori_sparse_indices)

    _assert_matches_reference(out, _reference_attention(case, physical_slots))


def test_minus_one_terminates_sparse_slot_reads() -> None:
    case = _make_case(kv_tokens=512)
    valid_slots = (torch.arange(37, dtype=torch.int64) * 11 + 3).remainder(case.kv_tokens)
    first_indices = torch.full((BLOCK_SIZE,), -1, dtype=torch.int32)
    second_indices = torch.full((BLOCK_SIZE,), -1, dtype=torch.int32)
    first_indices[: valid_slots.numel()] = valid_slots.to(torch.int32)
    second_indices[: valid_slots.numel()] = valid_slots.to(torch.int32)
    first_indices[valid_slots.numel() + 1 :] = torch.arange(
        BLOCK_SIZE - valid_slots.numel() - 1, dtype=torch.int32
    ).remainder(case.kv_tokens)
    second_indices[valid_slots.numel() + 1 :] = (
        torch.arange(BLOCK_SIZE - valid_slots.numel() - 1, dtype=torch.int32).mul(7).add(101).remainder(case.kv_tokens)
    )

    first_out = _run(case, first_indices.reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu())
    second_out = _run(case, second_indices.reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu())

    torch.testing.assert_close(first_out.float().cpu(), second_out.float().cpu(), atol=3e-2, rtol=3e-2)
    _assert_matches_reference(first_out, _reference_attention(case, valid_slots))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    result: dict[str, object] = {
        "path": str(path),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": _sha256(path),
    }
    if path.name in {"version.info", "version.json"}:
        result["version"] = path.read_text(errors="replace")[:4096]
    return result


def _loaded_custom_op_libraries() -> list[str]:
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return []
    markers = ("libcust_opapi", "optiling", "vllm_ascend_C")
    libraries = {
        line.rsplit(maxsplit=1)[-1]
        for line in maps_path.read_text(errors="replace").splitlines()
        if "/" in line and any(marker in line for marker in markers)
    }
    return sorted(libraries)


def _custom_op_artifacts() -> list[dict[str, object]]:
    paths: set[Path] = set()
    for entry in os.getenv("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if not entry:
            continue
        opp_path = Path(entry).resolve()
        for version_name in ("version.info", "version.json"):
            version_path = opp_path / version_name
            if version_path.is_file():
                paths.add(version_path)
        op_api_library = opp_path / "op_api" / "lib" / "libcust_opapi.so"
        if op_api_library.is_file():
            paths.add(op_api_library)
        if opp_path.is_dir():
            paths.update(path for path in opp_path.rglob("*sparse_attn_sharedkv*") if path.is_file())
            paths.update(path for path in opp_path.rglob("*SparseAttnSharedkv*") if path.is_file())
    return [_file_fingerprint(path) for path in sorted(paths)[:64]]


def _print_sharedkv_runtime_fingerprint() -> None:
    repo_root = Path(__file__).resolve().parents[6]
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    base_fields = _SparseAttnSharedkvSwaParams._fields_
    cmp_fields = _SparseAttnSharedkvCmpParams._fields_
    scalar_payload_bytes = sum(ctypes.sizeof(field_type) for _, field_type in (*base_fields, *cmp_fields))
    operator_root = repo_root / "csrc" / "attention" / "sparse_attn_sharedkv"
    swa_kernel_path = operator_root / "op_kernel" / "arch32" / "sparse_attn_sharedkv_swa_kernel.h"
    scfa_kernel_path = operator_root / "op_kernel" / "arch32" / "sparse_attn_sharedkv_scfa_kernel.h"
    source_paths = (
        repo_root / "csrc" / "torch_binding.cpp",
        operator_root / "op_host" / "sparse_attn_sharedkv_tiling.cpp",
        operator_root / "op_host" / "sparse_attn_sharedkv_tiling.h",
        operator_root / "op_kernel" / "sparse_attn_sharedkv_common.h",
        swa_kernel_path,
        scfa_kernel_path,
    )
    initialized_runinfo = "RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE] = {};"
    fingerprint = {
        "plugin_head": git_result.stdout.strip() if git_result.returncode == 0 else None,
        "plugin_head_error": git_result.stderr.strip() if git_result.returncode != 0 else None,
        "ascend_custom_opp_path": os.getenv("ASCEND_CUSTOM_OPP_PATH"),
        "extension": _file_fingerprint(Path(vllm_ascend_C.__file__).resolve()),
        "loaded_custom_op_libraries": _loaded_custom_op_libraries(),
        "custom_op_artifacts": _custom_op_artifacts(),
        "operator_sources": [_file_fingerprint(path) for path in source_paths],
        "runinfo_zero_initialized": {
            "swa": initialized_runinfo in swa_kernel_path.read_text(),
            "scfa": initialized_runinfo in scfa_kernel_path.read_text(),
        },
        "tiling": {
            "scalar_payload_bytes_without_padding": scalar_payload_bytes,
            "expected_native_layout_bytes": ctypes.sizeof(_SparseAttnSharedkvTilingData),
            "base_params_bytes": ctypes.sizeof(_SparseAttnSharedkvSwaParams),
            "cmp_params_bytes": ctypes.sizeof(_SparseAttnSharedkvCmpParams),
            "base_field_offsets": {name: getattr(_SparseAttnSharedkvSwaParams, name).offset for name, _ in base_fields},
        },
        "schemas": {
            "attention": str(torch.ops._C_ascend.npu_sparse_attn_sharedkv.default._schema),
            "metadata": str(torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata.default._schema),
        },
    }
    print("SHAREDKV_RUNTIME_FINGERPRINT " + json.dumps(fingerprint, sort_keys=True), flush=True)


def _cleanup_diagnostic_tensors(tensors: list[torch.Tensor]) -> list[str]:
    errors: list[str] = []
    try:
        tensors.clear()
    except Exception as exc:  # pragma: no cover - only observable after a device fault
        errors.append(f"release references: {exc!r}")
    try:
        gc.collect()
    except Exception as exc:  # pragma: no cover - only observable after a device fault
        errors.append(f"gc.collect: {exc!r}")
    try:
        torch.npu.empty_cache()
    except Exception as exc:  # pragma: no cover - only observable after a device fault
        errors.append(f"torch.npu.empty_cache: {exc!r}")
    for error in errors:
        print(f"SHAREDKV_CLEANUP_ERROR {error}", flush=True)
    return errors


def _build_diagnostic_block_table(
    case: SharedKVAddressDiagnosticCase,
) -> tuple[int, torch.Tensor]:
    num_logical_blocks = (case.seqused_kv + case.block_size - 1) // case.block_size
    num_physical_blocks = max(case.physical_block + 1, num_logical_blocks)
    physical_blocks = [case.physical_block]
    physical_blocks.extend(block_id for block_id in range(num_physical_blocks) if block_id != case.physical_block)
    return num_physical_blocks, torch.tensor([physical_blocks[:num_logical_blocks]], dtype=torch.int32)


def _build_diagnostic_slot_mapping(
    case: SharedKVAddressDiagnosticCase,
    block_table_cpu: torch.Tensor,
) -> torch.Tensor:
    logical_slots = torch.arange(case.seqused_kv, dtype=torch.int64)
    logical_blocks = torch.div(logical_slots, case.block_size, rounding_mode="floor")
    block_offsets = logical_slots.remainder(case.block_size)
    physical_blocks = block_table_cpu[0, logical_blocks].to(torch.int64)
    return torch.stack((physical_blocks, block_offsets), dim=1).to(torch.int32)


def _diagnostic_reference_attention(
    q_cpu: torch.Tensor,
    kv_cpu: torch.Tensor,
) -> torch.Tensor:
    q = q_cpu[0].float()
    kv = kv_cpu[:, 0].float()
    scores = torch.matmul(q, kv.transpose(0, 1)) * SOFTMAX_SCALE
    return torch.matmul(torch.softmax(scores, dim=-1), kv).unsqueeze(0)


def _run_sharedkv_address_diagnostic(
    case: SharedKVAddressDiagnosticCase,
) -> None:
    """Run one synchronized PA_ND/TND scatter plus shared-KV read."""
    if get_ascend_device_type() is not AscendDeviceType.A2:
        pytest.skip("The shared-KV address regression targets Ascend 910B2/arch32.")

    num_physical_blocks, block_table_cpu = _build_diagnostic_block_table(case)
    slot_mapping_cpu = _build_diagnostic_slot_mapping(case, block_table_cpu)
    element_size = torch.tensor([], dtype=case.dtype).element_size()
    page_elements = case.block_size * NUM_KV_HEADS * HEAD_DIM
    page_stride_bytes = page_elements * element_size
    page_offset_bytes = 0
    backing_bytes = num_physical_blocks * page_stride_bytes
    layer_name = "model.layers.0.self_attn.swa_cache"

    tensors: list[torch.Tensor] = []
    primary_failure: BaseException | None = None
    try:
        _print_sharedkv_runtime_fingerprint()
        if case.use_typed_backing:
            backing = torch.zeros(backing_bytes, dtype=torch.uint8, device="npu:0")
            (swa_kv_cache,) = _adjust_dsv4_kv_layout(
                backing,
                [
                    (
                        num_physical_blocks,
                        case.block_size,
                        NUM_KV_HEADS,
                        HEAD_DIM,
                    )
                ],
                [case.dtype],
                page_stride_bytes,
                page_offset_bytes,
            )
        else:
            swa_kv_cache = torch.zeros(
                (
                    num_physical_blocks,
                    case.block_size,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                ),
                dtype=case.dtype,
                device="npu:0",
            )
            backing = swa_kv_cache
        generator = torch.Generator().manual_seed(
            20260823 + case.block_size + case.physical_block + case.seqused_kv + element_size
        )
        q_cpu = (
            torch.randn(
                (1, NUM_Q_HEADS, HEAD_DIM),
                dtype=torch.float32,
                generator=generator,
            )
            .mul_(0.1)
            .to(case.dtype)
        )
        kv_cpu = (
            torch.randn(
                (case.seqused_kv, NUM_KV_HEADS, HEAD_DIM),
                dtype=torch.float32,
                generator=generator,
            )
            .mul_(0.1)
            .to(case.dtype)
        )
        q = q_cpu.npu()
        kv = kv_cpu.npu()
        ori_block_table = block_table_cpu.npu()
        slot_mapping = slot_mapping_cpu.npu()
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="npu:0")
        cu_seqlens_ori_kv = None
        if case.provide_cu_seqlens_ori_kv:
            cu_seqlens_ori_kv = torch.tensor([0, case.seqused_kv], dtype=torch.int32, device="npu:0")
        seqused_kv = torch.tensor([case.seqused_kv], dtype=torch.int32, device="npu:0")
        sinks = torch.full((NUM_Q_HEADS,), -10000.0, dtype=torch.float32, device="npu:0")
        ori_sparse_indices = None
        sparse_indices_cpu = None
        if case.provide_ori_sparse_indices:
            sparse_indices_cpu = torch.full((1, NUM_KV_HEADS, BLOCK_SIZE), -1, dtype=torch.int32)
            physical_slots = slot_mapping_cpu[:, 0].to(torch.int64) * case.block_size + slot_mapping_cpu[:, 1].to(
                torch.int64
            )
            sparse_indices_cpu[0, 0, : case.seqused_kv] = physical_slots.to(torch.int32)
            ori_sparse_indices = sparse_indices_cpu.npu()
        metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=NUM_Q_HEADS,
            num_heads_kv=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            batch_size=1,
            max_seqlen_q=1,
            max_seqlen_kv=case.seqused_kv,
            ori_topk=0,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=DEFAULT_WIN_LEFT,
            ori_win_right=DEFAULT_WIN_RIGHT,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
            device="npu:0",
        )
        tensors.extend(
            (
                backing,
                swa_kv_cache,
                q,
                kv,
                ori_block_table,
                slot_mapping,
                cu_seqlens_q,
                seqused_kv,
                sinks,
                metadata,
            )
        )
        if cu_seqlens_ori_kv is not None:
            tensors.append(cu_seqlens_ori_kv)
        if ori_sparse_indices is not None:
            tensors.append(ori_sparse_indices)

        print(
            f"SHAREDKV_POPULATE_BEGIN case={case.label} method={'scatter' if case.use_scatter else 'direct-copy'}",
            flush=True,
        )
        if case.use_scatter:
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)
        else:
            for logical_token, (physical_block, block_offset) in enumerate(slot_mapping_cpu.tolist()):
                swa_kv_cache[physical_block, block_offset].copy_(kv[logical_token])
        print(f"SHAREDKV_POPULATE_SYNC_BEGIN case={case.label}", flush=True)
        torch.npu.synchronize()
        print(f"SHAREDKV_POPULATE_SYNC_COMPLETE case={case.label}", flush=True)

        cache_base_offset_bytes = swa_kv_cache.data_ptr() - backing.data_ptr()
        block_stride_bytes = swa_kv_cache.stride(0) * element_size
        token_stride_bytes = swa_kv_cache.stride(1) * element_size
        accessed_starts = [
            cache_base_offset_bytes + int(physical_block) * block_stride_bytes + int(block_offset) * token_stride_bytes
            for physical_block, block_offset in slot_mapping_cpu.tolist()
        ]
        access_min_bytes = min(accessed_starts)
        access_max_bytes_exclusive = max(accessed_starts) + HEAD_DIM * element_size
        assert swa_kv_cache.data_ptr() == backing.data_ptr() + page_offset_bytes
        assert block_stride_bytes == page_stride_bytes
        assert access_min_bytes >= 0
        assert access_max_bytes_exclusive <= backing.untyped_storage().nbytes()
        diagnostic = {
            "case": case.label,
            "rank": int(os.getenv("RANK", "0")),
            "layer_name": layer_name,
            "compress_ratio": 1,
            "dtype": str(case.dtype),
            "block_size": case.block_size,
            "physical_block": case.physical_block,
            "seqused_kv": case.seqused_kv,
            "q": {
                "shape": list(q.shape),
                "dtype": str(q.dtype),
                "stride": list(q.stride()),
                "data_ptr": q.data_ptr(),
            },
            "kv": {
                "shape": list(swa_kv_cache.shape),
                "dtype": str(swa_kv_cache.dtype),
                "stride": list(swa_kv_cache.stride()),
                "storage_offset": swa_kv_cache.storage_offset(),
                "data_ptr": swa_kv_cache.data_ptr(),
            },
            "backing": {
                "data_ptr": backing.data_ptr(),
                "storage_bytes": backing.untyped_storage().nbytes(),
            },
            "ori_block_table": block_table_cpu.tolist(),
            "slot_mapping": slot_mapping_cpu.tolist(),
            "cu_seqlens_q": [0, 1],
            "cu_seqlens_ori_kv": ([0, case.seqused_kv] if cu_seqlens_ori_kv is not None else None),
            "metadata_parameters": {
                "layout_q": "TND",
                "layout_kv": "PA_ND",
                "has_ori_kv": True,
                "has_cmp_kv": False,
                "max_seqlen_q": 1,
                "max_seqlen_kv": case.seqused_kv,
                "ori_topk": 0,
                "cmp_topk": 0,
            },
            "attention_parameters": {
                "ori_kv": True,
                "cmp_kv": False,
                "ori_sparse_indices": ori_sparse_indices is not None,
                "cmp_sparse_indices": False,
                "ori_block_table": True,
                "cmp_block_table": False,
                "cu_seqlens_q": True,
                "cu_seqlens_ori_kv": cu_seqlens_ori_kv is not None,
                "cu_seqlens_cmp_kv": False,
                "seqused_q": False,
                "seqused_kv": True,
                "ori_kv_stride_elements": swa_kv_cache.stride(0),
            },
            "tiling_expectation": {
                "template": "SWA",
                "hasOriSparseIndices": ori_sparse_indices is not None,
                "oriSparseIndexWidth": (int(ori_sparse_indices.shape[-1]) if ori_sparse_indices is not None else 0),
                "host_serialized_actualLenDimsQ": 2,
                "kernel_cu_seqlens_q_entries_read": 2,
            },
            "ori_sparse_indices": (
                {
                    "shape": list(sparse_indices_cpu.shape),
                    "dtype": str(sparse_indices_cpu.dtype),
                    "valid_physical_slots": sparse_indices_cpu[0, 0, : case.seqused_kv].tolist(),
                    "terminator_index": case.seqused_kv,
                }
                if sparse_indices_cpu is not None
                else None
            ),
            "sas_metadata": {"shape": list(metadata.shape), "dtype": str(metadata.dtype)},
            "sinks": {"shape": list(sinks.shape), "dtype": str(sinks.dtype)},
            "theoretical_accessed_byte_range": {
                "min": access_min_bytes,
                "max_exclusive": access_max_bytes_exclusive,
            },
        }
        print(
            "SHAREDKV_ADDRESS_DIAGNOSTIC " + json.dumps(diagnostic, sort_keys=True),
            flush=True,
        )

        print(f"SHAREDKV_ATTENTION_CALL_BEGIN case={case.label}", flush=True)
        out, _ = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
            q,
            ori_kv=swa_kv_cache,
            cmp_kv=None,
            ori_sparse_indices=ori_sparse_indices,
            cmp_sparse_indices=None,
            ori_block_table=ori_block_table,
            cmp_block_table=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=SOFTMAX_SCALE,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=DEFAULT_WIN_LEFT,
            ori_win_right=DEFAULT_WIN_RIGHT,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=False,
        )
        tensors.append(out)
        print(f"SHAREDKV_ATTENTION_SYNC_BEGIN case={case.label}", flush=True)
        torch.npu.synchronize()
        print(f"SHAREDKV_ATTENTION_SYNC_COMPLETE case={case.label}", flush=True)

        expected = _diagnostic_reference_attention(q_cpu, kv_cpu)
        assert out.dtype == case.dtype
        assert out.shape == expected.shape
        assert torch.isfinite(out).all().cpu().item()
        torch.testing.assert_close(out.float().cpu(), expected.float(), atol=3e-2, rtol=3e-2)
        print(f"SHAREDKV_ADDRESS_PASS case={case.label}", flush=True)
    except BaseException as exc:
        primary_failure = exc
        print(f"SHAREDKV_ADDRESS_FAILURE case={case.label} error={exc!r}", flush=True)
        raise
    finally:
        cleanup_errors = _cleanup_diagnostic_tensors(tensors)
        if cleanup_errors and primary_failure is None:
            raise RuntimeError("shared-KV diagnostic cleanup failed: " + "; ".join(cleanup_errors))


def test_first_token_prefill_reads_kernel_compatible_swa_page() -> None:
    """A0: typed page-local cache, scatter, and absent optional inputs."""
    _run_sharedkv_address_diagnostic(SHAREDKV_ABI_A0)


def test_sharedkv_abi_a1_explicit_cu_seqlens_ori_kv() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ABI_A1)


def test_sharedkv_abi_a2_direct_contiguous_kv() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ABI_A2)


def test_sharedkv_abi_a3_direct_contiguous_kv_without_scatter() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ABI_A3)


def test_sharedkv_abi_a4_known_passing_operator_baseline() -> None:
    if get_ascend_device_type() is not AscendDeviceType.A2:
        pytest.skip("The shared-KV ABI regression targets Ascend 910B2/arch32.")
    tensors: list[torch.Tensor] = []
    primary_failure: BaseException | None = None
    try:
        _print_sharedkv_runtime_fingerprint()
        case = _make_case(q_tokens=1, kv_tokens=256)
        tensors.extend(
            (
                case.q,
                case.ori_kv,
                case.ori_block_table,
                case.seqused_kv,
                case.sinks,
                case.metadata,
            )
        )
        assert case.cu_seqlens_q is not None
        tensors.append(case.cu_seqlens_q)
        print("SHAREDKV_POPULATE_SYNC_BEGIN case=A4", flush=True)
        torch.npu.synchronize()
        print("SHAREDKV_POPULATE_SYNC_COMPLETE case=A4", flush=True)
        block_stride_bytes = case.ori_kv.stride(0) * case.ori_kv.element_size()
        selected_physical_block = int(case.block_table_cpu[0, -1])
        access_min = selected_physical_block * block_stride_bytes
        access_max = access_min + BLOCK_SIZE * case.ori_kv.stride(1) * case.ori_kv.element_size()
        diagnostic = {
            "case": "A4",
            "dtype": str(case.ori_kv.dtype),
            "block_size": BLOCK_SIZE,
            "physical_block": selected_physical_block,
            "seqused_kv": case.kv_tokens,
            "q": {
                "shape": list(case.q.shape),
                "dtype": str(case.q.dtype),
                "stride": list(case.q.stride()),
                "data_ptr": case.q.data_ptr(),
            },
            "kv": {
                "shape": list(case.ori_kv.shape),
                "dtype": str(case.ori_kv.dtype),
                "stride": list(case.ori_kv.stride()),
                "storage_offset": case.ori_kv.storage_offset(),
                "data_ptr": case.ori_kv.data_ptr(),
            },
            "backing": {
                "data_ptr": case.ori_kv.untyped_storage().data_ptr(),
                "storage_bytes": case.ori_kv.untyped_storage().nbytes(),
            },
            "ori_block_table": case.block_table_cpu.tolist(),
            "cu_seqlens_q": [0, 1],
            "cu_seqlens_ori_kv": None,
            "seqused_kv_tensor": [case.kv_tokens],
            "sas_metadata": {
                "shape": list(case.metadata.shape),
                "dtype": str(case.metadata.dtype),
            },
            "sinks": {"shape": list(case.sinks.shape), "dtype": str(case.sinks.dtype)},
            "theoretical_accessed_byte_range": {
                "min": access_min,
                "max_exclusive": access_max,
            },
            "metadata_parameters": {
                "layout_q": case.layout_q,
                "layout_kv": "PA_ND",
                "max_seqlen_q": 1,
                "max_seqlen_kv": case.kv_tokens,
            },
            "attention_parameters": {
                "ori_kv": True,
                "cmp_kv": False,
                "ori_block_table": True,
                "cmp_block_table": False,
                "cu_seqlens_q": True,
                "cu_seqlens_ori_kv": False,
                "cu_seqlens_cmp_kv": False,
                "seqused_q": False,
                "seqused_kv": True,
                "ori_sparse_indices": False,
                "cmp_sparse_indices": False,
                "ori_kv_stride_elements": case.ori_kv.stride(0),
            },
            "tiling_expectation": {
                "template": "SWA",
                "hasOriSparseIndices": False,
                "oriSparseIndexWidth": 0,
                "host_serialized_actualLenDimsQ": 2,
                "kernel_cu_seqlens_q_entries_read": 2,
            },
        }
        print(
            "SHAREDKV_ADDRESS_DIAGNOSTIC " + json.dumps(diagnostic, sort_keys=True),
            flush=True,
        )
        print("SHAREDKV_ATTENTION_CALL_BEGIN case=A4", flush=True)
        out = _run(case, ori_sparse_indices=None)
        tensors.append(out)
        print("SHAREDKV_ATTENTION_SYNC_BEGIN case=A4", flush=True)
        torch.npu.synchronize()
        print("SHAREDKV_ATTENTION_SYNC_COMPLETE case=A4", flush=True)
        logical_positions = torch.arange(case.kv_tokens - BLOCK_SIZE, case.kv_tokens, dtype=torch.int64)
        physical_slots = _physical_slots_for_logical_positions(case, logical_positions)
        _assert_matches_reference(out, _reference_attention(case, physical_slots))
        print("SHAREDKV_ADDRESS_PASS case=A4", flush=True)
    except BaseException as exc:
        primary_failure = exc
        print(f"SHAREDKV_ADDRESS_FAILURE case=A4 error={exc!r}", flush=True)
        raise
    finally:
        cleanup_errors = _cleanup_diagnostic_tensors(tensors)
        if cleanup_errors and primary_failure is None:
            raise RuntimeError("shared-KV diagnostic cleanup failed: " + "; ".join(cleanup_errors))


def test_sharedkv_abi_a5_explicit_physical_sparse_indices() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ABI_A5)


def test_sharedkv_address_b_block_32_physical_1_s2_16_bf16() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ADDRESS_B)


def test_sharedkv_address_c_block_32_physical_1_s2_32_bf16() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ADDRESS_C)


def test_sharedkv_address_d_block_128_physical_1_s2_1_bf16() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ADDRESS_D)


def test_sharedkv_address_e_block_32_physical_1_s2_1_fp16() -> None:
    _run_sharedkv_address_diagnostic(SHAREDKV_ADDRESS_E)


@pytest.mark.parametrize("case", SHAREDKV_S2_BOUNDARY_CASES)
def test_first_token_prefill_sharedkv_s2_boundaries(
    case: SharedKVAddressDiagnosticCase,
) -> None:
    _run_sharedkv_address_diagnostic(case)


def test_rejects_wrong_ori_sparse_indices_dtype() -> None:
    case = _make_case()
    indices = torch.arange(BLOCK_SIZE, dtype=torch.int64).reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()


def test_rejects_non_tnd_ori_sparse_indices_layout() -> None:
    case = _make_case(layout_q="BSND")
    indices = torch.arange(BLOCK_SIZE, dtype=torch.int32).reshape(1, 1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()


def test_rejects_query_and_index_token_shape_mismatch() -> None:
    case = _make_case(q_tokens=2)
    indices = torch.arange(BLOCK_SIZE, dtype=torch.int32).reshape(1, NUM_KV_HEADS, BLOCK_SIZE).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()


@pytest.mark.parametrize("index_width", [0, 64, 129, 2176])
def test_rejects_invalid_or_unaligned_index_width(index_width: int) -> None:
    case = _make_case()
    indices = torch.zeros((1, NUM_KV_HEADS, index_width), dtype=torch.int32).npu()

    with pytest.raises(RuntimeError):
        _run(case, indices)
        torch.npu.synchronize()
