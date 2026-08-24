# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_PATH = REPO_ROOT / "vllm_ascend/diagnostics/w8a8_nz.py"


def _load_diagnostics():
    spec = importlib.util.spec_from_file_location("fix9a_w8a8_nz", DIAGNOSTIC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_diagnostic_is_centralized_and_disabled_by_default() -> None:
    envs_source = _source("vllm_ascend/envs.py")
    assert '"DSPARK_DIAG_W8A8_NZ": lambda: bool(' in envs_source
    assert 'os.getenv("DSPARK_DIAG_W8A8_NZ", "0")' in envs_source


def test_rank_zero_diagnostic_hooks_cover_all_three_lifecycle_stages() -> None:
    post_load = _source("vllm_ascend/quantization/methods/w8a8_dynamic.py")
    kv_binding = _source("vllm_ascend/worker/v2/attn_utils.py")
    device_op = _source("vllm_ascend/device/device_op.py")
    assert "record_post_load(layer)" in post_load
    assert "record_kv_cache_initialized(kv_caches)" in kv_binding
    assert "record_pre_gmm(" in device_op
    assert "if ascend_envs.DSPARK_DIAG_W8A8_NZ and self.quant_type == QuantType.W8A8:" in post_load
    assert "if ascend_envs.DSPARK_DIAG_W8A8_NZ:" in kv_binding
    assert "if ascend_envs.DSPARK_DIAG_W8A8_NZ:" in device_op
    assert "single_tensor::grouped_matmul_swiglu_quant_weight_nz" in device_op
    assert "return torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(" in device_op


def test_single_weight_operator_contract_requires_one_five_dimensional_nz_storage() -> None:
    binding = " ".join(_source("csrc/torch_binding.cpp").split())
    adapter = _source(
        "csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/grouped_matmul_swiglu_quant_torch_adpt.h"
    )
    host = _source("csrc/gmm/grouped_matmul_swiglu_quant/op_host/op_api/aclnn_grouped_matmul_swiglu_quant.cpp")
    mlp = _source("vllm_ascend/ops/fused_moe/moe_mlp.py")
    diagnostics = DIAGNOSTIC_PATH.read_text(encoding="utf-8")

    assert "grouped_matmul_swiglu_quant_weight_nz(Tensor x, Tensor weight," in binding
    assert "grouped_matmul_swiglu_quant_weight_nz_tensor_list(Tensor x, Tensor[] weight," in binding
    assert "aclnnGroupedMatmulSwigluQuantWeightNZ" in adapter
    assert "must be 5." in host
    assert '_require_single_tensor_for_swiglu_quant(w1, name="w1")' in mlp
    assert "torch_npu.get_npu_format(tensor)" in diagnostics
    assert "torch.ops._C_ascend.get_npu_storage_shape(tensor)" in diagnostics


def test_once_gate_allows_only_the_first_expert_gmm() -> None:
    diagnostics = _load_diagnostics()
    gate = diagnostics._OnceGate()
    assert gate.claim() is True
    assert gate.claim() is False


def test_enabled_diagnostic_emits_once_without_mutating_weight(capsys) -> None:
    diagnostics = _load_diagnostics()
    diagnostics._diagnostics_enabled_on_rank_zero = lambda: True
    weight = SimpleNamespace(device="cpu")
    layer = SimpleNamespace(layer_name="model.layers.0.mlp.experts", w13_weight=weight)

    diagnostics.record_post_load(layer)
    assert diagnostics.record_pre_gmm(weight, operator_variant="single_tensor") is None
    assert diagnostics.record_pre_gmm(weight, operator_variant="single_tensor") is None

    output = capsys.readouterr().out
    assert output.count(diagnostics.DIAGNOSTIC_EVENT + "=") == 1
    assert layer.w13_weight is weight


def test_int8_nz_shape_is_derived_from_logical_expert_dimensions() -> None:
    diagnostics = _load_diagnostics()
    assert diagnostics.expected_int8_nz_shape([32, 7168, 4096]) == [32, 128, 448, 16, 32]
    assert diagnostics.expected_int8_nz_shape([32, 2048, 7168]) == [32, 224, 128, 16, 32]
    assert diagnostics.expected_int8_nz_shape([1, 15, 32]) is None
    assert diagnostics.expected_int8_nz_shape([1, 16, 31]) is None


def test_layer_index_is_derived_from_the_real_parameter_prefix() -> None:
    diagnostics = _load_diagnostics()
    parameter_name = "model.layers.12.mlp.experts.routed_experts.w13_weight"
    assert diagnostics._parameter_name("model.layers.12.mlp.experts", "w13_weight") == parameter_name
    assert diagnostics._layer_index(parameter_name) == 12
    assert diagnostics._layer_index("mtp.0.mlp.experts.w13_weight") is None


def test_pointer_alias_and_overlap_are_distinguished() -> None:
    diagnostics = _load_diagnostics()
    backing = {
        "object_id": 20,
        "storage_pointer": 1000,
        "storage_bytes": 512,
    }
    object_alias = diagnostics.compare_descriptor_to_backings(
        {"object_id": 20, "storage_pointer": 4000, "storage_bytes": 64},
        [backing],
        {20},
    )
    assert object_alias == {
        "WEIGHT_KV_OBJECT_ALIAS": True,
        "WEIGHT_KV_STORAGE_ALIAS": False,
        "WEIGHT_KV_POINTER_OVERLAP": False,
    }

    storage_alias = diagnostics.compare_descriptor_to_backings(
        {"object_id": 21, "storage_pointer": 1000, "storage_bytes": 64},
        [backing],
        {20},
    )
    assert storage_alias == {
        "WEIGHT_KV_OBJECT_ALIAS": False,
        "WEIGHT_KV_STORAGE_ALIAS": True,
        "WEIGHT_KV_POINTER_OVERLAP": True,
    }

    overlap_only = diagnostics.compare_descriptor_to_backings(
        {"object_id": 21, "storage_pointer": 1400, "storage_bytes": 200},
        [backing],
        {20},
    )
    assert overlap_only == {
        "WEIGHT_KV_OBJECT_ALIAS": False,
        "WEIGHT_KV_STORAGE_ALIAS": False,
        "WEIGHT_KV_POINTER_OVERLAP": True,
    }


def test_diagnostic_never_reads_values_or_fakes_a_five_dimensional_view() -> None:
    tree = ast.parse(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    forbidden_calls = {"cpu", "numpy", "reshape", "tolist", "view"}
    called_attributes = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_attributes.isdisjoint(forbidden_calls)
    assert "npu_format_cast" not in DIAGNOSTIC_PATH.read_text(encoding="utf-8")
