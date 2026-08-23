from __future__ import annotations

import ctypes
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = REPO_ROOT / "csrc" / "attention" / "sparse_attn_sharedkv"
TILING_HEADER = OP_ROOT / "op_host" / "sparse_attn_sharedkv_tiling.h"
TILING_SOURCE = OP_ROOT / "op_host" / "sparse_attn_sharedkv_tiling.cpp"
KERNEL_ENTRY = OP_ROOT / "op_kernel" / "sparse_attn_sharedkv.cpp"
KERNEL_COMMON = OP_ROOT / "op_kernel" / "sparse_attn_sharedkv_common.h"
SWA_KERNEL = OP_ROOT / "op_kernel" / "arch32" / "sparse_attn_sharedkv_swa_kernel.h"
SCFA_KERNEL = OP_ROOT / "op_kernel" / "arch32" / "sparse_attn_sharedkv_scfa_kernel.h"
TORCH_BINDING = REPO_ROOT / "csrc" / "torch_binding.cpp"


class SparseAttnSharedkvSwaParams(ctypes.Structure):
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


class SparseAttnSharedkvCmpParams(ctypes.Structure):
    _fields_ = [
        ("cmpMaxBlockNumPerBatch", ctypes.c_uint32),
        ("sparseBlockCount", ctypes.c_uint32),
        ("cmpRatio", ctypes.c_int64),
        ("cmpMaskMode", ctypes.c_uint64),
        ("cmpKvStride", ctypes.c_int64),
    ]


class SparseAttnSharedkvTilingData(ctypes.Structure):
    _fields_ = [
        ("baseParams", SparseAttnSharedkvSwaParams),
        ("cmpParams", SparseAttnSharedkvCmpParams),
    ]


def _read(path: Path) -> str:
    return path.read_text()


def _assert_in_order(source: str, names: tuple[str, ...]) -> None:
    offset = 0
    for name in names:
        next_offset = source.find(name, offset)
        assert next_offset >= 0, f"missing ABI field {name}"
        offset = next_offset + len(name)


def _expected_q_extent(values: list[int]) -> int:
    return len(values)


def test_single_batch_cu_seqlens_q_extent_includes_both_offsets() -> None:
    assert _expected_q_extent([0, 1]) == 2


def test_eight_batch_cu_seqlens_q_extent_includes_terminal_offset() -> None:
    assert _expected_q_extent(list(range(9))) == 9


def test_eight_batch_seq_used_q_extent_is_eight() -> None:
    assert _expected_q_extent([1] * 8) == 8


def test_tnd_query_extent_uses_full_tensor_element_count_at_both_assignments() -> None:
    host = _read(TILING_SOURCE)
    cu_seqlens_assignment = "actualLenDimsQ_ = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize();"
    seq_used_assignment = "actualLenDimsQ_ = opParamInfo_.seqUsedQ.tensor->GetShapeSize();"
    assert host.count(cu_seqlens_assignment) == 2
    assert host.count(seq_used_assignment) == 2
    assert "cuSeqLensQ.tensor->GetShapeSize() - 1" not in host

    for kernel_path in (SWA_KERNEL, SCFA_KERNEL):
        kernel = _read(kernel_path)
        assert "actualSeqLengthsQGm.GetValue(bIdx)" in kernel
        assert "actualSeqLengthsQGm.GetValue(bIdx + 1)" in kernel
        assert (
            "actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, constInfo.actualLenDimsQ)"
        ) in kernel


def test_preload_run_info_pipeline_state_is_aggregate_zero_initialized() -> None:
    common = _read(KERNEL_COMMON)
    run_info = common[common.index("struct RunInfo {") : common.index("struct ConstInfo {")]
    declarations = [
        line.split("//", maxsplit=1)[0].strip()
        for line in run_info.splitlines()[1:]
        if line.split("//", maxsplit=1)[0].strip().endswith(";")
    ]
    instance_fields = [line for line in declarations if line != "};" and not line.startswith("static constexpr")]
    assert instance_fields
    assert all("=" in line for line in instance_fields)

    initialized_declaration = "RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE] = {};"
    bare_declaration = "RunInfo extraInfo[SAS_PRELOAD_TASK_CACHE_SIZE];"
    for kernel_path in (SWA_KERNEL, SCFA_KERNEL):
        kernel = _read(kernel_path)
        assert kernel.count(initialized_declaration) == 1
        assert bare_declaration not in kernel


def test_preload_pipeline_guards_each_cache_slot_with_is_valid() -> None:
    for kernel_path in (SWA_KERNEL, SCFA_KERNEL):
        kernel = _read(kernel_path)
        pipeline = kernel[kernel.index("::PreloadPipeline(") :]
        _assert_in_order(
            pipeline,
            (
                "CalcParams(loop, cmpLoop, s2Start, s2LoopIdx, extraInfo0);",
                "if (extraInfo0.isValid)",
                "if (extraInfo2.isValid)",
                "if (extraInfo1.isValid)",
                "extraInfo1.isValid = false;",
            ),
        )


def test_optional_sparse_indices_are_guarded_by_serialized_tiling_flags() -> None:
    host = _read(TILING_SOURCE)
    kernel = _read(SWA_KERNEL)

    _assert_in_order(
        host,
        (
            "set_hasOriSparseIndices(tilingInfo->hasOriSparseIndices)",
            "set_oriSparseIndexWidth(tilingInfo->oriSparseIndexWidth)",
            "set_returnSoftmaxLse(tilingInfo->returnSoftmaxLse)",
            "set_usedCoreNum(usedCoreNum_)",
        ),
    )
    _assert_in_order(
        kernel,
        (
            "constInfo.hasOriSparseIndices = tilingData->baseParams.hasOriSparseIndices",
            "constInfo.oriSparseIndexWidth = tilingData->baseParams.oriSparseIndexWidth",
            "if (constInfo.hasOriSparseIndices)",
            "oriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)oriSparseIndices)",
        ),
    )
    assert "bool hasOriSparseIndices = false;" in _read(TILING_HEADER)
    assert "uint32_t oriSparseIndexWidth = 0;" in _read(TILING_HEADER)


def test_python_binding_host_and_kernel_input_slot_order_match() -> None:
    expected_inputs = (
        "q",
        "ori_kv",
        "cmp_kv",
        "ori_sparse_indices",
        "cmp_sparse_indices",
        "ori_block_table",
        "cmp_block_table",
        "cu_seqlens_q",
        "cu_seqlens_ori_kv",
        "cu_seqlens_cmp_kv",
        "seqused_q",
        "seqused_kv",
        "sinks",
        "metadata",
    )
    binding = _read(TORCH_BINDING)
    schema = binding[binding.index('"npu_sparse_attn_sharedkv("') :]
    schema = schema[: schema.index('ops.impl("npu_sparse_attn_sharedkv"')]
    _assert_in_order(schema, expected_inputs)

    host_indices = _read(TILING_HEADER)
    expected_constants = (
        "Q_INDEX = 0",
        "ORI_KV_INDEX = 1",
        "CMP_KV_INDEX = 2",
        "ORI_SPARSE_INDICES_INDEX = 3",
        "CMP_SPARSE_INDICES_INDEX = 4",
        "ORI_BLOCK_TABLE_INDEX = 5",
        "CMP_BLOCK_TABLE_INDEX = 6",
        "CU_SEQLENS_Q_INDEX = 7",
        "CU_SEQLENS_KV_INDEX = 8",
        "CU_SEQLENS_CMP_KV_INDEX = 9",
        "SEQUSED_Q_INDEX = 10",
        "SEQUSED_KV_INDEX = 11",
        "SINKS_INDEX = 12",
        "METADATA_INDEX = 13",
    )
    _assert_in_order(host_indices, expected_constants)

    kernel = _read(KERNEL_ENTRY)
    kernel_signature = kernel[kernel.index("sparse_attn_sharedkv(__gm__") :]
    _assert_in_order(
        kernel_signature,
        (
            "query",
            "oriKV",
            "cmpKV",
            "oriSparseIndices",
            "cmpSparseIndices",
            "oriBlockTable",
            "cmpBlockTable",
            "cuSeqlensQ",
            "cuSeqlensOriKv",
            "cuSeqlensCmpKv",
            "seqUsedQ",
            "seqUsedKV",
            "sinks",
            "metadata",
        ),
    )


def test_tiling_field_order_and_expected_native_size_are_stable() -> None:
    header = _read(TILING_HEADER)
    base_fields = tuple(name for name, _ in SparseAttnSharedkvSwaParams._fields_)
    cmp_fields = tuple(name for name, _ in SparseAttnSharedkvCmpParams._fields_)
    _assert_in_order(header, base_fields + cmp_fields)

    assert ctypes.sizeof(SparseAttnSharedkvSwaParams) == 136
    assert ctypes.sizeof(SparseAttnSharedkvCmpParams) == 32
    assert ctypes.sizeof(SparseAttnSharedkvTilingData) == 168
    assert SparseAttnSharedkvSwaParams.hasOriSparseIndices.offset == 104
    assert SparseAttnSharedkvSwaParams.oriSparseIndexWidth.offset == 108
    assert SparseAttnSharedkvSwaParams.usedCoreNum.offset == 112


def test_pa_stride_is_produced_and_consumed_in_elements() -> None:
    binding = _read(TORCH_BINDING)
    assert "ori_kv_stride = tmp_kv.stride(0);" in binding
    assert (
        "EXEC_NPU_CMD(aclnnSparseAttnSharedkv, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices"
    ) in binding

    common = _read(KERNEL_COMMON)
    assert "uint64_t offset = idInBlockTable * shape.kvStride" in common
    assert "GlobalTensor<T> tmpSrcTensor = srcTensor[offset]" in common
    assert "uint64_t offset = blockId * shape.kvStride" in common


def test_pa_nd_uses_seqused_kv_and_does_not_consume_cu_seqlens_ori_kv() -> None:
    for kernel_path in (SWA_KERNEL, SCFA_KERNEL):
        kernel = _read(kernel_path)
        pa_branch = (
            "(KV_LAYOUT_T == SAS_LAYOUT::PA_ND || KV_LAYOUT_T == SAS_LAYOUT::BSND) && LAYOUT_T == SAS_LAYOUT::TND"
        )
        assert pa_branch in kernel
        assert "InitActualSeqLen(cuSeqlensQ, seqUsedKV);" in kernel


def test_cmp_optional_addresses_are_only_bound_for_cmp_templates() -> None:
    cmp_bind = "cmpKvGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV)"
    swa = _read(SWA_KERNEL)
    assert "if (constInfo.templateMode == CFA_TEMPLATE)" in swa
    assert cmp_bind in swa

    # The SCFA class is instantiated only for TEMPLATE_MODE == SCFA_TEMPLATE,
    # whose host contract requires cmp_kv and cmp_sparse_indices together.
    scfa = _read(SCFA_KERNEL)
    assert cmp_bind in scfa
    entry = _read(KERNEL_ENTRY)
    assert "if constexpr (TEMPLATE_MODE == SCFA_TEMPLATE)" in entry
