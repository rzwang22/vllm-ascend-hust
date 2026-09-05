# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Host execution of the frozen scatter kernel's address/range arithmetic.

This compiles original function bodies with CPU vector primitives. It models
the union of all cores' write ranges, and replaces DMA with CPU tensor copies
in the caller. It does not execute Ascend instructions or validate NPU codegen.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np


def _body(source, signature):
    start = source.index("{", source.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def build_scatter_selector(root, directory):
    kernel = root / "csrc/moe/scatter_nd_update_v2/op_kernel"
    linear = (kernel / "scatter_nd_update_linear_index.h").read_text()
    sorted_kernel = (kernel / "scatter_nd_update_v2.h").read_text()
    no_sort = (kernel / "scatter_nd_update_no_sort.h").read_text()
    tiling = (kernel.parent / "op_host/scatter_nd_update_v2_tiling.cpp").read_text()
    dispatch = (kernel / "scatter_nd_update_v2.cpp").read_text()
    assert "LinearIndexKernel<false, int>" in _body(dispatch, "TILING_KEY_IS(10)")
    assert "ScatterNdUpdateV2KernelNoSort<DTYPE_VAR>" in _body(dispatch, "TILING_KEY_IS(10)")
    assert "LinearIndexKernel<true, int>" in _body(dispatch, "TILING_KEY_IS(11)")
    assert "ScatterNdUpdateV2Kernel<DTYPE_VAR>" in _body(dispatch, "TILING_KEY_IS(11)")
    # Keep this probe coupled to the active int32 selector and stride-based
    # address calculation, rather than assuming the large-int64 branch applies.
    assert "tilingKey_ = indexType * 10 + sortFlag" in tiling
    assert "isSort_ = IsSort(totalPhysicalRange, indexRow)" in tiling
    assert "indicesMask_[i] = static_cast<uint64_t>(stridesPtr->GetData()[i])" in tiling
    assert "Sort<float, true>" in linear
    condition = no_sort.split("if (linearIndex", 1)[1].split(" {", 1)[0]
    condition = "linearIndex" + condition[:-1]  # Remove the if statement's final ')'.
    constant = next(
        line for line in tiling.splitlines() if line.startswith("constexpr uint64_t MAX_FLOAT_EXPRESS_INT32")
    )
    source = r"""
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

template <typename T> struct LocalTensor {
    std::vector<int32_t>* values;
    T GetValue(int64_t index) const { return static_cast<T>(values->at(index)); }
    template <typename U> LocalTensor<U> ReinterpretCast() { return {values}; }
};
template <typename T> void Duplicate(LocalTensor<T> out, int value, uint64_t n) {
    for (uint64_t i=0; i<n; ++i) out.values->at(i)=value;
}
void CreateVecIndex(LocalTensor<int> out, int start, uint64_t n) {
    for (uint64_t i=0; i<n; ++i) out.values->at(i)=start+i;
}
void Muls(LocalTensor<int> out, LocalTensor<int> in, int value, uint64_t n) {
    for (uint64_t i=0; i<n; ++i) out.values->at(i)=in.values->at(i)*value;
}
void Adds(LocalTensor<int> out, LocalTensor<int> in, int value, uint64_t n) {
    for (uint64_t i=0; i<n; ++i) out.values->at(i)=in.values->at(i)+value;
}
void Add(LocalTensor<int> out, LocalTensor<int> lhs, LocalTensor<int> rhs, uint64_t n) {
    for (uint64_t i=0; i<n; ++i) out.values->at(i)=lhs.values->at(i)+rhs.values->at(i);
}
void Gather(LocalTensor<int> out, LocalTensor<int> in, LocalTensor<uint32_t> offsets, uint32_t base, uint32_t n) {
    for (uint64_t i=0; i<n; ++i) out.values->at(i)=in.values->at((base+offsets.GetValue(i))/sizeof(int));
}
constexpr int PIPE_V=0;
template <int> void PipeBarrier() {}
void PipeVToMte3() {}

struct Linear {
    static constexpr bool isSort=false;
    uint64_t blockLength_, blockRemainLength_, indexDim_=2;
    uint64_t indicesMask_[2];
    std::vector<int32_t> out, origin, tmp, range;
    LocalTensor<int> indicesLocal{&out}, indicesOriginLocal{&origin}, addTmpLocal{&tmp}, rangeLocal{&range};
    Linear(const int32_t* slots, int n, int64_t stride0, int64_t stride1)
        : blockLength_(n), blockRemainLength_(n), indicesMask_{uint64_t(stride0), uint64_t(stride1)},
          out(n), origin(slots, slots+2*n), tmp(n), range(n) {}
    void Compute4LinearIndex(uint64_t process, bool isTail) LINEAR_BODY
};

struct Sorted {
    uint64_t start_, end_, blockLength_, blockRemainLength_;
    int64_t leftBound_, rightBound_;
    bool isValidBound_;
    int64_t findFirstLt(LocalTensor<int>& indiceLocal, int64_t target, bool isTail) FIRST_BODY
    int64_t findLastGe(LocalTensor<int>& indiceLocal, int64_t target, bool isTail) LAST_BODY
    void UpdateSearchParam(LocalTensor<int>& indiceLocal, bool isTail) BOUNDS_BODY
};
SORT_CONSTANT
bool IsSort(uint64_t totalLength, uint64_t indexRow) SORT_BODY

extern "C" int select_rows(const int32_t* slots, int n, int64_t stride0, int64_t stride1,
                           int64_t physical_range, int32_t* selected, int32_t* addresses, int* key) {
    Linear linear(slots, n, stride0, stride1);
    linear.Compute4LinearIndex(0, false);
    int count=0;
    if (IsSort(physical_range, n)) {
        *key=11;
        std::vector<int32_t> order(n), values(n);
        std::iota(order.begin(), order.end(), 0);
        // Every address is exactly representable in float in this tiling branch.
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {return linear.out[a]>linear.out[b];});
        for (int i=0; i<n; ++i) values[i]=linear.out[order[i]];
        LocalTensor<int> tensor{&values};
        Sorted range{0, uint64_t(physical_range), uint64_t(n), uint64_t(n)};
        range.UpdateSearchParam(tensor, false);
        if (range.isValidBound_) {
            for (int64_t i=range.rightBound_; i>=range.leftBound_; --i) {
                selected[count]=order[i]; addresses[count++]=values[i];
            }
        }
    } else {
        *key=10;
        uint64_t start_=0, end_=physical_range;
        for (int i=0; i<n; ++i) {
            int64_t linearIndex=linear.out[i];
            if (NO_SORT_CONDITION) { selected[count]=i; addresses[count++]=linearIndex; }
        }
    }
    return count;
}
"""
    replacements = {
        "LINEAR_BODY": _body(linear, "inline void Compute4LinearIndex("),
        "FIRST_BODY": _body(sorted_kernel, "inline int64_t findFirstLt("),
        "LAST_BODY": _body(sorted_kernel, "inline int64_t findLastGe("),
        "BOUNDS_BODY": _body(sorted_kernel, "inline void UpdateSearchParam("),
        "SORT_CONSTANT": constant,
        "SORT_BODY": _body(tiling, "inline bool ScatterNdUpdateV2Tiling::IsSort("),
        "NO_SORT_CONDITION": condition,
    }
    for key, body in replacements.items():
        source = source.replace(key, body)
    directory = Path(directory)
    cpp, library = directory / "scatter_host.cpp", directory / "scatter_host.so"
    cpp.write_text(source)
    compiler = shutil.which("c++")
    assert compiler, "CPU scatter contract tests require a C++ compiler"
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", str(cpp), "-o", str(library)], check=True)
    dll = ctypes.CDLL(str(library))
    ptr = ctypes.POINTER(ctypes.c_int32)
    dll.select_rows.argtypes = [
        ptr,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ptr,
        ptr,
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.select_rows.restype = ctypes.c_int

    def select(slots, stride0, stride1, physical_range):
        slots = np.ascontiguousarray(slots, dtype=np.int32)
        selected, addresses = np.empty(len(slots), dtype=np.int32), np.empty(len(slots), dtype=np.int32)
        key = ctypes.c_int()
        count = dll.select_rows(
            slots.ctypes.data_as(ptr),
            len(slots),
            stride0,
            stride1,
            physical_range,
            selected.ctypes.data_as(ptr),
            addresses.ctypes.data_as(ptr),
            ctypes.byref(key),
        )
        return key.value, selected[:count], addresses[:count]

    return select
