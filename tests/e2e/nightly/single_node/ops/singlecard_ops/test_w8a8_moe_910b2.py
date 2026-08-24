# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Single-card production-path W8A8 MoE probe for Ascend 910B2."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

REQUIRED_MODULES = ("torch", "torch_npu", "vllm", "vllm_ascend")
MISSING_MODULES = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
if MISSING_MODULES:
    pytest.skip(
        "W8A8 MoE NPU probe dependencies are unavailable: " + ", ".join(MISSING_MODULES),
        allow_module_level=True,
    )

NUM_EXPERTS = 2
NUM_TOKENS = 4
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256
TOP_K = 1
EXPERT_GROUP_COUNT = 1
TOPK_GROUP_COUNT = 1
MOE_GATING_GROUP_SELECT_MODE = 1
MIN_EXPERTS_PER_GROUP_FOR_TOP2_SUM = 2
NPU_DEVICE = 0
PEAK_ALLOCATED_LIMIT_BYTES = 64 * 1024 * 1024
LAUNCH_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE")


def _emit(stage: str, **details: Any) -> None:
    suffix = f" {json.dumps(details, default=str, sort_keys=True)}" if details else ""
    print(f"{stage}{suffix}", flush=True)


def _validate_expert_group_fixture() -> None:
    assert EXPERT_GROUP_COUNT > 0
    assert NUM_EXPERTS % EXPERT_GROUP_COUNT == 0
    experts_per_group = NUM_EXPERTS // EXPERT_GROUP_COUNT
    assert MOE_GATING_GROUP_SELECT_MODE == 1
    assert experts_per_group >= MIN_EXPERTS_PER_GROUP_FOR_TOP2_SUM, (
        "MoeGatingTopK group_select_mode=1 ranks groups by their top-2 "
        f"expert scores, but this fixture has only {experts_per_group} expert(s) per group"
    )
    assert 1 <= TOPK_GROUP_COUNT <= EXPERT_GROUP_COUNT


def _require_910b2_npu() -> tuple[Any, Any]:
    # Delayed imports keep collection safe on hosts without torch-npu.
    try:
        import torch
        import torch_npu
    except Exception as exc:
        pytest.skip(f"torch-npu is not usable: {exc}")

    try:
        npu_available = torch.npu.is_available() and torch.npu.device_count() > 0
    except Exception as exc:
        pytest.skip(f"Unable to query Ascend NPU devices: {exc}")
    if not npu_available:
        pytest.skip("An Ascend NPU is required")

    torch.npu.set_device(NPU_DEVICE)
    device_name = torch.npu.get_device_name(NPU_DEVICE)
    if "910B2" not in device_name.upper():
        pytest.skip(f"This probe requires Ascend 910B2, got {device_name!r}")
    if not torch.distributed.is_backend_available("hccl"):
        pytest.fail("torch.distributed HCCL backend is unavailable")

    torch_npu.npu.config.allow_internal_format = True
    return torch, torch_npu


def _configure_single_rank_launch() -> dict[str, str | None]:
    from vllm.utils.network_utils import get_open_port

    previous = {name: os.environ.get(name) for name in LAUNCH_ENV_KEYS}
    expected = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }
    for name, expected_value in expected.items():
        actual = os.environ.get(name)
        if actual not in (None, expected_value):
            pytest.fail(f"{name} must be {expected_value}, got {actual}")

    for name, expected_value in expected.items():
        os.environ[name] = expected_value

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(get_open_port())
    return previous


def _restore_launch_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _build_vllm_config(torch: Any) -> Any:
    from vllm.config import ModelConfig, VllmConfig

    tests_root = Path(__file__).resolve().parents[5]
    config_fixture = tests_root / "ut" / "_fake_weight"
    assert (config_fixture / "config.json").is_file(), f"Missing config-only fixture: {config_fixture}"

    # The local fixture supplies only Hugging Face configuration. This probe
    # never constructs a model loader or loads checkpoint tensors.
    model_config = ModelConfig(
        model=str(config_fixture),
        dtype=torch.bfloat16,
        enforce_eager=True,
        skip_tokenizer_init=True,
        max_model_len=128,
        hf_overrides={
            "n_routed_experts": NUM_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "n_group": EXPERT_GROUP_COUNT,
            "topk_group": TOPK_GROUP_COUNT,
            "moe_quantize": "w8a8_dynamic",
        },
    )
    assert model_config.is_moe
    assert model_config.get_num_experts() == NUM_EXPERTS

    vllm_config = VllmConfig(
        model_config=model_config,
        additional_config={
            "enable_flashcomm1": False,
            "enable_fused_mc2": 0,
            "multistream_overlap_gate": False,
            "eplb_config": {
                "dynamic_eplb": False,
                "num_redundant_experts": 0,
            },
            "ascend_fusion_config": {
                "fusion_ops_gmmswigluquant": True,
            },
            "ascend_compilation_config": {
                "enable_npugraph_ex": False,
                "enable_static_kernel": False,
            },
        },
    )
    parallel_config = vllm_config.parallel_config
    parallel_config.enable_expert_parallel = False
    parallel_config.enable_eplb = False
    assert parallel_config.world_size == 1
    assert parallel_config.tensor_parallel_size == 1
    assert parallel_config.pipeline_parallel_size == 1
    assert parallel_config.data_parallel_size == 1
    assert parallel_config.prefill_context_parallel_size == 1
    assert parallel_config.decode_context_parallel_size == 1
    return vllm_config


def _initialize_plugin_runtime(torch: Any, vllm_config: Any) -> None:
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import (
        AscendDeviceType,
        adapt_patch,
        check_ascend_device_type,
        enable_custom_op,
        get_ascend_device_type,
        register_ascend_customop,
    )

    adapt_patch()
    register_ascend_customop(vllm_config)
    init_ascend_config(vllm_config)
    check_ascend_device_type()
    assert get_ascend_device_type() is AscendDeviceType.A2
    assert enable_custom_op(), "vllm-ascend custom ops are unavailable"

    _emit(
        "ENV_READY",
        device=torch.npu.get_device_name(NPU_DEVICE),
        hccl_available=True,
        custom_ops=True,
    )


def _initialize_distributed(torch: Any, vllm_config: Any, state: dict[str, Any]) -> None:
    # Import after adapt_patch() so all coordinators use the production NPU patch.
    from vllm.distributed import get_dp_group, get_ep_group, get_pp_group, get_tp_group, get_world_group
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment

    from vllm_ascend.distributed.parallel_state import get_mc2_group, init_ascend_model_parallel

    assert not torch.distributed.is_initialized(), "The focused probe requires a fresh distributed process"
    state["distributed_attempted"] = True
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method="env://",
        local_rank=0,
        backend="hccl",
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        prefill_context_model_parallel_size=1,
        decode_context_model_parallel_size=1,
    )
    torch.distributed.barrier()

    groups = {
        "world": get_world_group(),
        "tp": get_tp_group(),
        "pp": get_pp_group(),
        "dp": get_dp_group(),
        "ep": get_ep_group(),
    }
    for name, group in groups.items():
        assert group.world_size == 1, f"{name} world size is {group.world_size}"
        assert group.rank_in_group == 0, f"{name} rank is {group.rank_in_group}"
    world_backend = str(torch.distributed.get_backend(get_world_group().device_group)).lower()
    assert "hccl" in world_backend
    _emit(
        "DISTRIBUTED_READY",
        backend=world_backend,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )

    state["ascend_groups_attempted"] = True
    init_ascend_model_parallel(vllm_config.parallel_config)
    mc2_group = get_mc2_group()
    mc2_backend = str(torch.distributed.get_backend(mc2_group.device_group)).lower()
    assert mc2_group.world_size == 1
    assert mc2_group.rank_in_group == 0
    assert "hccl" in mc2_backend
    state["payload"]["mc2_group"] = mc2_group
    _emit(
        "MC2_READY",
        backend=mc2_backend,
        rank=mc2_group.rank_in_group,
        world_size=mc2_group.world_size,
        unique_name=mc2_group.unique_name,
    )


def _construct_w8a8_routed_experts(torch: Any, torch_npu: Any, vllm_config: Any) -> Any:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

    from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
    from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod
    from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig
    from vllm_ascend.quantization.quant_type import QuantType
    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

    layer_prefix = "model.layers.0.mlp.experts"
    quant_config = AscendModelSlimConfig(
        {
            "version": "1.0.0",
            "model_quant_type": "W8A8_DYNAMIC",
            f"{layer_prefix}.weight": "W8A8_DYNAMIC",
        }
    )
    vllm_config.quant_config = quant_config
    parallel_config = FusedMoEParallelConfig.make_no_parallel()
    moe_config = FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        device=torch.device("npu", NPU_DEVICE),
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        max_num_tokens=NUM_TOKENS,
        swiglu_limit=0.0,
        max_capture_size=NUM_TOKENS,
    )
    expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=NUM_TOKENS,
        top_k=TOP_K,
        global_num_experts=NUM_EXPERTS,
        num_redundant_experts=0,
        num_expert_group=EXPERT_GROUP_COUNT,
        moe_parallel_config=parallel_config,
        placement_strategy="linear",
        enable_eplb=False,
    )
    assert expert_map_manager.global_num_experts == NUM_EXPERTS
    assert expert_map_manager.local_num_experts == NUM_EXPERTS
    assert expert_map_manager.expert_map is None
    assert expert_map_manager.get_local_expert_ids() == list(range(NUM_EXPERTS))

    # H=I=256 makes every INT8/NZ matrix dimension a multiple of 256,
    # satisfying the production FRACTAL_NZ and grouped-matmul alignment.
    with torch.device(f"npu:{NPU_DEVICE}"):
        layer = RoutedExperts(
            layer_name=layer_prefix,
            params_dtype=torch.bfloat16,
            moe_config=moe_config,
            quant_config=quant_config,
            expert_map_manager=expert_map_manager,
            renormalize=True,
            use_grouped_topk=False,
            num_expert_group=EXPERT_GROUP_COUNT,
            topk_group=TOPK_GROUP_COUNT,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            swiglu_limit=0.0,
        )

    method = layer.quant_method
    scheme = method.quant_method
    assert layer.local_num_experts == NUM_EXPERTS
    assert layer.expert_map is None
    assert type(method) is AscendFusedMoEMethod
    assert isinstance(method, FusedMoEMethodBase)
    assert not isinstance(method, UnquantizedFusedMoEMethod)
    assert type(scheme) is AscendW8A8DynamicFusedMoEMethod
    assert scheme.quant_type is QuantType.W8A8
    assert scheme.quant_type not in {QuantType.MXFP4, QuantType.MXFP8, QuantType.W4A8MXFP}
    assert scheme.moe_all_to_all_group_name, "The W8A8 scheme did not resolve real MC2 HCCL metadata"
    _emit(
        "SCHEME_SELECTED",
        config=type(quant_config).__name__,
        method=type(method).__name__,
        scheme=type(scheme).__name__,
        quant_type=scheme.quant_type.name,
        num_experts=NUM_EXPERTS,
        expert_group_count=EXPERT_GROUP_COUNT,
        experts_per_group=NUM_EXPERTS // EXPERT_GROUP_COUNT,
        topk_group_count=TOPK_GROUP_COUNT,
        group_select_mode=MOE_GATING_GROUP_SELECT_MODE,
        unquantized_fallback=False,
        mxfp=False,
    )

    expected_parameters = {
        "w13_weight": (torch.int8, (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)),
        "w2_weight": (torch.int8, (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)),
        "w13_weight_scale": (torch.bfloat16, (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, 1)),
        "w13_weight_offset": (torch.bfloat16, (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, 1)),
        "w2_weight_scale": (torch.bfloat16, (NUM_EXPERTS, HIDDEN_SIZE, 1)),
        "w2_weight_offset": (torch.bfloat16, (NUM_EXPERTS, HIDDEN_SIZE, 1)),
    }
    for name, (dtype, shape) in expected_parameters.items():
        parameter = getattr(layer, name)
        assert parameter.dtype == dtype
        assert tuple(parameter.shape) == shape
        assert parameter.device.type == "npu"

    with torch.no_grad():
        for name in ("w13_weight", "w2_weight"):
            parameter = getattr(layer, name)
            values = torch.randint(-4, 5, tuple(parameter.shape), dtype=torch.int8)
            parameter.copy_(values.to(parameter.device))
        layer.w13_weight_scale.fill_(0.015625)
        layer.w13_weight_offset.zero_()
        layer.w2_weight_scale.fill_(0.015625)
        layer.w2_weight_offset.zero_()
    torch.npu.synchronize()
    _emit(
        "WEIGHTS_CREATED",
        w13_dtype=str(layer.w13_weight.dtype),
        w13_shape=list(layer.w13_weight.shape),
        w2_dtype=str(layer.w2_weight.dtype),
        w2_shape=list(layer.w2_weight.shape),
    )

    method.process_weights_after_loading(layer)
    torch.npu.synchronize()
    assert tuple(layer.w13_weight.shape) == (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE)
    assert tuple(layer.w2_weight.shape) == (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE)
    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert tuple(layer.w13_weight_scale.shape) == (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE)
    assert tuple(layer.w13_weight_offset.shape) == (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE)
    assert tuple(layer.w2_weight_scale.shape) == (NUM_EXPERTS, HIDDEN_SIZE)
    assert tuple(layer.w2_weight_offset.shape) == (NUM_EXPERTS, HIDDEN_SIZE)
    assert tuple(layer.w13_weight_scale_fp32.shape) == (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE)
    assert layer.w13_weight_scale_fp32.dtype == torch.float32
    assert int(torch_npu.get_npu_format(layer.w13_weight)) == ACL_FORMAT_FRACTAL_NZ
    assert int(torch_npu.get_npu_format(layer.w2_weight)) == ACL_FORMAT_FRACTAL_NZ
    w13_storage_shape = tuple(torch.ops._C_ascend.get_npu_storage_shape(layer.w13_weight))
    w2_storage_shape = tuple(torch.ops._C_ascend.get_npu_storage_shape(layer.w2_weight))
    assert w13_storage_shape == (
        NUM_EXPERTS,
        (2 * INTERMEDIATE_SIZE) // 32,
        HIDDEN_SIZE // 16,
        16,
        32,
    )
    assert w2_storage_shape == (
        NUM_EXPERTS,
        HIDDEN_SIZE // 32,
        INTERMEDIATE_SIZE // 16,
        16,
        32,
    )
    _emit(
        "WEIGHTS_POSTPROCESSED",
        nz_format=ACL_FORMAT_FRACTAL_NZ,
        w13_shape=list(layer.w13_weight.shape),
        w13_storage_shape=list(w13_storage_shape),
        w2_shape=list(layer.w2_weight.shape),
        w2_storage_shape=list(w2_storage_shape),
        w13_scale_dtype=str(layer.w13_weight_scale_fp32.dtype),
    )
    return layer


def _execute_production_w8a8_moe(torch: Any, layer: Any, vllm_config: Any) -> Any:
    from vllm.distributed import get_dp_group, get_tp_group
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType, set_ascend_forward_context
    from vllm_ascend.ops.fused_moe.moe_comm_method import (
        AllGatherCommImpl,
        FusedExpertsResult,
        setup_moe_comm_method,
    )
    from vllm_ascend.quantization.quant_type import QuantType

    layer.moe_config.tp_group = get_tp_group()
    layer.moe_config.dp_group = get_dp_group()
    setup_moe_comm_method(layer.moe_config)

    hidden_states = (
        torch.randn(
            (NUM_TOKENS, HIDDEN_SIZE),
            device=f"npu:{NPU_DEVICE}",
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    router_logits = torch.zeros(
        (NUM_TOKENS, NUM_EXPERTS),
        device=f"npu:{NPU_DEVICE}",
        dtype=torch.bfloat16,
    )

    with (
        torch.no_grad(),
        set_ascend_forward_context(
            None,
            vllm_config,
            num_tokens=NUM_TOKENS,
            num_actual_tokens=NUM_TOKENS,
            in_profile_run=False,
            is_draft_model=False,
        ),
    ):
        # EP world size one intentionally selects the production all-gather
        # dispatcher. The MC2 coordinator was still created and consumed by
        # the W8A8 scheme constructor above.
        assert _EXTRA_CTX.moe_comm_type is MoECommType.ALLGATHER
        comm_method = _EXTRA_CTX.moe_comm_method
        assert isinstance(comm_method, AllGatherCommImpl)
        prepared = comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            enable_shared_expert_dp=False,
            replace_allreduce=False,
            quant_type=QuantType.W8A8,
        )
        # With world-size=1 and SP disabled, token dispatch receives BF16 and
        # invokes production npu_moe_init_routing_custom with quant_mode=1.
        assert prepared.pertoken_scale is None
        result = layer.quant_method.apply(
            layer=layer,
            x=prepared.hidden_states,
            router_logits=prepared.router_logits,
            top_k=TOP_K,
            renormalize=True,
            use_grouped_topk=False,
            num_experts=NUM_EXPERTS,
            expert_map=layer.expert_map,
            topk_group=TOPK_GROUP_COUNT,
            num_expert_group=EXPERT_GROUP_COUNT,
            custom_routing_function=None,
            scoring_func="softmax",
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            is_prefill=True,
            enable_force_load_balance=False,
            log2phy=None,
            global_redundant_expert_num=0,
            pertoken_scale=prepared.pertoken_scale,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            mc2_mask=prepared.mc2_mask,
        )
        assert isinstance(result, FusedExpertsResult)
        assert result.before_gmm2_evt is not None
        output = comm_method.finalize(
            result.routed_out,
            reduce_results=True,
            padded_hidden_states_shape=prepared.padded_hidden_states_shape,
        )

    torch.npu.synchronize()
    _emit("GMM1_COMPLETE", dynamic_int8_activation_quant=True, before_gmm2_event=True)
    _emit("SWIGLU_COMPLETE", activation=MoEActivation.SILU.value, fused_with_gmm1=True)
    _emit("GMM2_COMPLETE", dtype=str(output.dtype), shape=list(output.shape))
    assert output.dtype == torch.bfloat16
    assert tuple(output.shape) == (NUM_TOKENS, HIDDEN_SIZE)
    # The single host sync is intentional at the final E2E assertion boundary.
    assert bool(torch.isfinite(output).all().item())
    _emit("FINITE_OUTPUT", dtype=str(output.dtype), shape=list(output.shape), finite=True)
    return hidden_states, router_logits, result, output


def _memory_snapshot(torch: Any, label: str) -> dict[str, int]:
    torch.npu.synchronize()
    snapshot = {
        "allocated": int(torch.npu.memory_allocated()),
        "reserved": int(torch.npu.memory_reserved()),
        "max_allocated": int(torch.npu.max_memory_allocated()),
        "max_reserved": int(torch.npu.max_memory_reserved()),
    }
    _emit("NPU_MEMORY", label=label, **snapshot)
    return snapshot


def _run_probe(torch: Any, torch_npu: Any, vllm_config: Any, state: dict[str, Any]) -> None:
    state["ascend_config_attempted"] = True
    _initialize_plugin_runtime(torch, vllm_config)
    _initialize_distributed(torch, vllm_config, state)

    torch.npu.synchronize()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    baseline = _memory_snapshot(torch, "post_hccl_baseline")

    layer = _construct_w8a8_routed_experts(torch, torch_npu, vllm_config)
    state["payload"]["layer"] = layer
    hidden_states, router_logits, result, output = _execute_production_w8a8_moe(torch, layer, vllm_config)
    state["payload"].update(
        {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
            "result": result,
            "output": output,
        }
    )

    final_memory = _memory_snapshot(torch, "after_w8a8_moe")
    peak_allocated_delta = final_memory["max_allocated"] - baseline["allocated"]
    assert peak_allocated_delta < PEAK_ALLOCATED_LIMIT_BYTES, (
        f"Probe allocator peak {peak_allocated_delta} exceeds {PEAK_ALLOCATED_LIMIT_BYTES} bytes"
    )
    _emit("W8A8_PROBE_PASS", peak_allocated_delta=peak_allocated_delta)


def _run_cleanup_steps(steps: Sequence[tuple[str, Callable[[], None]]]) -> list[str]:
    errors: list[str] = []
    for name, cleanup in steps:
        try:
            cleanup()
        except Exception:
            details = traceback.format_exc()
            errors.append(f"{name}:\n{details}")
            _emit("CLEANUP_ERROR", cleanup_stage=name)
            print(details, flush=True)
    _emit("CLEANUP_COMPLETE", cleanup_errors=errors)
    return errors


def _cleanup_probe(
    torch: Any,
    state: dict[str, Any],
    config_context: Any,
    previous_vllm_config: Any,
    previous_launch_env: dict[str, str | None],
) -> list[str]:
    def synchronize_npu() -> None:
        if torch.npu.is_initialized():
            torch.npu.synchronize()

    def release_payload() -> None:
        state["payload"].clear()

    def collect_garbage() -> None:
        gc.collect()

    def destroy_ascend_groups() -> None:
        if state["ascend_groups_attempted"]:
            from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

            destroy_ascend_model_parallel()

    def destroy_vllm_model_parallel() -> None:
        if state["distributed_attempted"]:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()

    def destroy_vllm_world() -> None:
        if state["distributed_attempted"]:
            from vllm.distributed.parallel_state import destroy_distributed_environment

            destroy_distributed_environment()

    def destroy_torch_process_group() -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def empty_npu_cache() -> None:
        if torch.npu.is_initialized():
            torch.npu.empty_cache()

    def clear_plugin_config() -> None:
        if state["ascend_config_attempted"]:
            from vllm_ascend.ascend_config import clear_ascend_config

            clear_ascend_config()

    def restore_current_config() -> None:
        config_context.__exit__(None, None, None)
        from vllm.config import get_current_vllm_config_or_none

        assert get_current_vllm_config_or_none() is previous_vllm_config

    return _run_cleanup_steps(
        (
            ("torch.npu.synchronize", synchronize_npu),
            ("release_layer_and_tensors", release_payload),
            ("gc.collect", collect_garbage),
            ("destroy_ascend_model_parallel", destroy_ascend_groups),
            ("destroy_model_parallel", destroy_vllm_model_parallel),
            ("destroy_distributed_environment", destroy_vllm_world),
            ("torch.distributed.destroy_process_group", destroy_torch_process_group),
            ("torch.npu.empty_cache", empty_npu_cache),
            ("clear_ascend_config", clear_plugin_config),
            ("restore_current_vllm_config", restore_current_config),
            ("restore_launch_environment", lambda: _restore_launch_environment(previous_launch_env)),
        )
    )


def test_w8a8_moe_world_size_one_hccl_mc2() -> None:
    _validate_expert_group_fixture()
    torch, torch_npu = _require_910b2_npu()
    vllm_config = _build_vllm_config(torch)
    previous_launch_env = _configure_single_rank_launch()

    from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config

    previous_vllm_config = get_current_vllm_config_or_none()
    config_context = set_current_vllm_config(vllm_config)
    try:
        config_context.__enter__()
    except BaseException:
        _restore_launch_environment(previous_launch_env)
        raise
    state: dict[str, Any] = {
        "payload": {},
        "distributed_attempted": False,
        "ascend_groups_attempted": False,
        "ascend_config_attempted": False,
    }
    primary_error = False
    try:
        _run_probe(torch, torch_npu, vllm_config, state)
    except BaseException:
        primary_error = True
        raise
    finally:
        cleanup_errors = _cleanup_probe(
            torch,
            state,
            config_context,
            previous_vllm_config,
            previous_launch_env,
        )
        if cleanup_errors and not primary_error:
            pytest.fail("W8A8 MoE probe cleanup failed:\n" + "\n".join(cleanup_errors))
