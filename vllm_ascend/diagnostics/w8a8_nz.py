# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One-shot ModelSlim W8A8 expert NZ descriptor diagnostics.

This module intentionally imports torch and torch_npu only inside NPU-only
metadata helpers. It never reads tensor values or creates tensor copies.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

POST_LOAD_STAGE = "A_CHECKPOINT_POST_LOAD"
POST_KV_INIT_STAGE = "B_KV_CACHE_INITIALIZED"
PRE_GMM_STAGE = "C_GROUPED_MATMUL_PRE_CALL"
DIAGNOSTIC_EVENT = "DSPARK_W8A8_NZ_DIAGNOSTIC"


class _OnceGate:
    def __init__(self) -> None:
        self._claimed = False
        self._lock = threading.Lock()

    def claim(self) -> bool:
        with self._lock:
            if self._claimed:
                return False
            self._claimed = True
            return True


@dataclass
class _WeightRecord:
    parameter_name: str
    tensor_ref: Any
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _DiagnosticState:
    records: dict[int, _WeightRecord] = field(default_factory=dict)
    kv_backings: list[dict[str, Any]] = field(default_factory=list)
    kv_object_ids: set[int] = field(default_factory=set)
    once: _OnceGate = field(default_factory=_OnceGate)
    lock: threading.Lock = field(default_factory=threading.Lock)


_STATE = _DiagnosticState()


def pointer_ranges_overlap(
    first_pointer: int,
    first_nbytes: int,
    second_pointer: int,
    second_nbytes: int,
) -> bool:
    """Return whether two non-empty byte ranges overlap."""
    if first_pointer <= 0 or second_pointer <= 0:
        return False
    if first_nbytes <= 0 or second_nbytes <= 0:
        return False
    return first_pointer < second_pointer + second_nbytes and second_pointer < first_pointer + first_nbytes


def expected_int8_nz_shape(logical_shape: list[int] | tuple[int, ...]) -> list[int] | None:
    """Derive [E, N/32, K/16, 16, 32] for logical [E, K, N]."""
    if len(logical_shape) != 3:
        return None
    experts, k_size, n_size = (int(value) for value in logical_shape)
    if experts <= 0 or k_size <= 0 or n_size <= 0:
        return None
    if k_size % 16 != 0 or n_size % 32 != 0:
        return None
    return [experts, n_size // 32, k_size // 16, 16, 32]


def compare_descriptor_to_backings(
    weight: dict[str, Any],
    kv_backings: list[dict[str, Any]],
    kv_object_ids: set[int] | None = None,
) -> dict[str, bool]:
    """Compare tensor identity and storage byte ranges without dereferencing data."""
    object_ids = kv_object_ids or set()
    object_alias = int(weight.get("object_id", -1)) in object_ids
    weight_pointer = int(weight.get("storage_pointer", 0) or 0)
    weight_nbytes = int(weight.get("storage_bytes", 0) or 0)
    storage_alias = False
    pointer_overlap = False
    for backing in kv_backings:
        backing_pointer = int(backing.get("storage_pointer", 0) or 0)
        backing_nbytes = int(backing.get("storage_bytes", 0) or 0)
        storage_alias = storage_alias or (
            weight_pointer > 0 and backing_pointer > 0 and weight_pointer == backing_pointer
        )
        pointer_overlap = pointer_overlap or pointer_ranges_overlap(
            weight_pointer,
            weight_nbytes,
            backing_pointer,
            backing_nbytes,
        )
    return {
        "WEIGHT_KV_OBJECT_ALIAS": object_alias,
        "WEIGHT_KV_STORAGE_ALIAS": storage_alias,
        "WEIGHT_KV_POINTER_OVERLAP": pointer_overlap,
    }


def _diagnostics_enabled_on_rank_zero() -> bool:
    from vllm_ascend import envs

    if not envs.DSPARK_DIAG_W8A8_NZ:
        return False
    try:
        import torch

        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except (AttributeError, ImportError, RuntimeError):
        return False
    return _safe_int(os.getenv("RANK", "0"), default=-1) == 0


def _current_rank() -> int:
    try:
        import torch

        if torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except (AttributeError, ImportError, RuntimeError):
        pass
    return _safe_int(os.getenv("RANK", "0"), default=0)


def _weak_tensor_ref(tensor: Any) -> Any:
    try:
        return weakref.ref(tensor)
    except TypeError:
        return lambda: tensor


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _storage_metadata(tensor: Any) -> tuple[int, int]:
    try:
        storage = tensor.untyped_storage()
        pointer = _safe_int(storage.data_ptr())
        nbytes = _safe_int(storage.nbytes())
        return pointer, nbytes
    except (AttributeError, RuntimeError, TypeError):
        return 0, 0


def _root_tensor(tensor: Any) -> Any:
    current = tensor
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        base = getattr(current, "_base", None)
        if base is None:
            break
        current = base
    return current


def _shape(tensor: Any) -> list[int] | None:
    try:
        return [_safe_int(value) for value in tensor.shape]
    except (AttributeError, TypeError):
        return None


def _npu_descriptor(tensor: Any) -> tuple[int | None, list[int] | None, str | None]:
    device = str(getattr(tensor, "device", "unknown"))
    if not device.startswith("npu"):
        return None, None, None
    errors: list[str] = []
    format_code: int | None = None
    storage_shape: list[int] | None = None
    try:
        import torch_npu

        format_code = _safe_int(torch_npu.get_npu_format(tensor), default=-1)
    except (AttributeError, RuntimeError, TypeError) as exc:
        errors.append(f"format:{type(exc).__name__}:{exc}")
    try:
        import torch

        storage_shape = [_safe_int(value) for value in torch.ops._C_ascend.get_npu_storage_shape(tensor)]
    except (AttributeError, RuntimeError, TypeError) as exc:
        errors.append(f"storage_shape:{type(exc).__name__}:{exc}")
    return format_code, storage_shape, "; ".join(errors) or None


def tensor_descriptor(tensor: Any) -> dict[str, Any]:
    """Collect metadata only; never materialize or inspect tensor values."""
    storage_pointer, storage_bytes = _storage_metadata(tensor)
    root = _root_tensor(tensor)
    root_storage_pointer, root_storage_bytes = _storage_metadata(root)
    logical_shape = _shape(tensor)
    expected_nz_shape = None if logical_shape is None else expected_int8_nz_shape(logical_shape)
    format_code, storage_shape, descriptor_error = _npu_descriptor(tensor)
    try:
        stride = [_safe_int(value) for value in tensor.stride()]
    except (AttributeError, RuntimeError, TypeError):
        stride = None
    try:
        is_contiguous = bool(tensor.is_contiguous())
    except (AttributeError, RuntimeError, TypeError):
        is_contiguous = None
    descriptor = {
        "object_id": id(tensor),
        "shape": logical_shape,
        "stride": stride,
        "dtype": str(getattr(tensor, "dtype", "unknown")),
        "device": str(getattr(tensor, "device", "unknown")),
        "numel": _safe_int(tensor.numel()) if hasattr(tensor, "numel") else None,
        "element_size": _safe_int(tensor.element_size()) if hasattr(tensor, "element_size") else None,
        "storage_offset": _safe_int(tensor.storage_offset()) if hasattr(tensor, "storage_offset") else None,
        "is_contiguous": is_contiguous,
        "data_ptr": _safe_int(tensor.data_ptr()) if hasattr(tensor, "data_ptr") else 0,
        "storage_pointer": storage_pointer,
        "storage_bytes": storage_bytes,
        "base_object_id": None if root is tensor else id(root),
        "base_shape": None if root is tensor else _shape(root),
        "base_data_ptr": (None if root is tensor or not hasattr(root, "data_ptr") else _safe_int(root.data_ptr())),
        "base_storage_pointer": root_storage_pointer,
        "base_storage_bytes": root_storage_bytes,
        "npu_format_code": format_code,
        "npu_storage_shape": storage_shape,
        "npu_descriptor_error": descriptor_error,
        "expected_int8_nz_shape": expected_nz_shape,
        "expected_nz_storage_bytes": (
            None
            if expected_nz_shape is None or not hasattr(tensor, "element_size")
            else math.prod(expected_nz_shape) * _safe_int(tensor.element_size())
        ),
        "npu_storage_shape_matches_expected": (
            None if storage_shape is None or expected_nz_shape is None else storage_shape == expected_nz_shape
        ),
    }
    descriptor["storage_pointer_range"] = [
        storage_pointer,
        storage_pointer + storage_bytes,
    ]
    descriptor["base_storage_pointer_range"] = [
        root_storage_pointer,
        root_storage_pointer + root_storage_bytes,
    ]
    return descriptor


def _layer_name(layer: Any) -> str:
    for attribute in ("layer_name", "prefix"):
        value = getattr(layer, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(layer).__name__


def _parameter_name(layer_name: str, short_name: str) -> str:
    routed_experts_prefix = layer_name if layer_name.endswith(".routed_experts") else f"{layer_name}.routed_experts"
    return f"{routed_experts_prefix}.{short_name}"


def _layer_index(parameter_name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", parameter_name)
    return None if match is None else int(match.group(1))


def _emit_diagnostic_error(stage: str, exc: Exception) -> None:
    print(
        "DSPARK_W8A8_NZ_DIAGNOSTIC_ERROR="
        + json.dumps(
            {
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def record_post_load(layer: Any) -> None:
    """Record W8A8 expert parameters after checkpoint post-processing."""
    if not _diagnostics_enabled_on_rank_zero():
        return
    try:
        layer_name = _layer_name(layer)
        with _STATE.lock:
            for short_name in ("w13_weight", "w2_weight"):
                tensor = getattr(layer, short_name, None)
                if tensor is None:
                    continue
                record = _WeightRecord(
                    parameter_name=_parameter_name(layer_name, short_name),
                    tensor_ref=_weak_tensor_ref(tensor),
                )
                record.stages[POST_LOAD_STAGE] = tensor_descriptor(tensor)
                _STATE.records[id(tensor)] = record
    except Exception as exc:
        _emit_diagnostic_error(POST_LOAD_STAGE, exc)


def _iter_cache_tensors(value: Any, name: str):
    if hasattr(value, "data_ptr") and hasattr(value, "untyped_storage"):
        yield name, value
        return
    if isinstance(value, dict):
        for child_name, child in value.items():
            yield from _iter_cache_tensors(child, f"{name}.{child_name}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_cache_tensors(child, f"{name}[{index}]")


def _collect_kv_backings(kv_caches: dict[str, Any]) -> tuple[list[dict[str, Any]], set[int]]:
    grouped: dict[tuple[int, int, int], dict[str, Any]] = {}
    object_ids: set[int] = set()
    for cache_name, tensor in _iter_cache_tensors(kv_caches, "kv_caches"):
        object_ids.add(id(tensor))
        root = _root_tensor(tensor)
        object_ids.add(id(root))
        descriptor = tensor_descriptor(root)
        key = (
            _safe_int(descriptor.get("storage_pointer")),
            _safe_int(descriptor.get("storage_bytes")),
            id(root) if not descriptor.get("storage_pointer") else 0,
        )
        if key not in grouped:
            grouped[key] = {
                **descriptor,
                "cache_names": [],
                "cache_view_count": 0,
            }
        grouped[key]["cache_names"].append(cache_name)
        grouped[key]["cache_view_count"] += 1
    return list(grouped.values()), object_ids


def record_kv_cache_initialized(kv_caches: dict[str, Any]) -> None:
    """Record the same weights and physical KV backings after KV binding."""
    if not _diagnostics_enabled_on_rank_zero():
        return
    try:
        backings, object_ids = _collect_kv_backings(kv_caches)
        with _STATE.lock:
            _STATE.kv_backings = backings
            _STATE.kv_object_ids = object_ids
            for record in _STATE.records.values():
                tensor = record.tensor_ref()
                if tensor is not None:
                    record.stages[POST_KV_INIT_STAGE] = tensor_descriptor(tensor)
    except Exception as exc:
        _emit_diagnostic_error(POST_KV_INIT_STAGE, exc)


def _find_weight_record(weight: Any) -> tuple[_WeightRecord | None, str]:
    record = _STATE.records.get(id(weight))
    if record is not None:
        return record, "object_identity"
    weight_pointer, _ = _storage_metadata(weight)
    if weight_pointer <= 0:
        return None, "unresolved"
    for candidate in _STATE.records.values():
        tensor = candidate.tensor_ref()
        if tensor is None:
            continue
        candidate_pointer, _ = _storage_metadata(tensor)
        if candidate_pointer == weight_pointer:
            return candidate, "storage_pointer"
    return None, "unresolved"


def record_pre_gmm(weight: Any, *, operator_variant: str) -> None:
    """Print one rank-zero record immediately before the first expert GMM."""
    if not _diagnostics_enabled_on_rank_zero() or not _STATE.once.claim():
        return
    try:
        with _STATE.lock:
            record, resolution_basis = _find_weight_record(weight)
            current = tensor_descriptor(weight)
            if record is None:
                parameter_name = "UNRESOLVED_WEIGHT"
                stages = {PRE_GMM_STAGE: current}
            else:
                parameter_name = record.parameter_name
                record.stages[PRE_GMM_STAGE] = current
                stages = dict(record.stages)
            alias_report = compare_descriptor_to_backings(
                current,
                _STATE.kv_backings,
                _STATE.kv_object_ids,
            )
            object_ids = [stage["object_id"] for stage in stages.values() if stage.get("object_id") is not None]
            storage_pointers = [stage["storage_pointer"] for stage in stages.values() if stage.get("storage_pointer")]
            payload = {
                "event": DIAGNOSTIC_EVENT,
                "rank": _current_rank(),
                "layer_index": _layer_index(parameter_name),
                "layer_name": parameter_name.rsplit(".", 1)[0],
                "parameter_name": parameter_name,
                "parameter_resolution": resolution_basis,
                "runtime_weight_is_registered_parameter": (resolution_basis == "object_identity"),
                "operator_variant": operator_variant,
                "operator_second_input": "weight",
                "stages": stages,
                "same_weight_object_across_stages": len(set(object_ids)) <= 1,
                "same_weight_storage_across_stages": (len(set(storage_pointers)) <= 1),
                "captured_at_all_stages": set(stages) == {POST_LOAD_STAGE, POST_KV_INIT_STAGE, PRE_GMM_STAGE},
                "kv_backings": _STATE.kv_backings,
                **alias_report,
            }
        print(DIAGNOSTIC_EVENT + "=" + json.dumps(payload, sort_keys=True), flush=True)
    except Exception as exc:
        _emit_diagnostic_error(PRE_GMM_STAGE, exc)
