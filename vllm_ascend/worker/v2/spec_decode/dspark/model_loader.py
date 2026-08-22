# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
import os
from collections.abc import Callable, Mapping
from typing import Any

import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

_DSPARK_BLOCK_SIZE = 5
_DSPARK_NUM_MTP_LAYERS = 3
_DSPARK_TARGET_LAYER_IDS = (40, 41, 42)
_MODELSLIM_FLOAT = "FLOAT"
_MODELSLIM_W8A8_DYNAMIC = "W8A8_DYNAMIC"
_DSPARK_ATTENTION_NAMESPACES = ("attn", "self_attn")
_DSPARK_W8A8_ATTENTION_COMPONENTS = ("wq_a", "wq_b", "wkv")
_DSPARK_FLOAT_ATTENTION_COMPONENTS = (
    "q_norm",
    "kv_norm",
    "wo_a",
    "wo_b",
)
_DSPARK_QUANT_PREFIX_REPLACEMENTS = (
    (".attn.", ".self_attn."),
    (".ffn_norm.", ".post_attention_layernorm."),
    (".attn_norm.", ".input_layernorm."),
    (".ffn.", ".mlp."),
    (".w1.", ".gate_proj."),
    (".w2.", ".down_proj."),
    (".w3.", ".up_proj."),
)


def _same_checkpoint(target_model_config: Any, draft_model_config: Any) -> bool:
    target_name = os.fspath(target_model_config.model)
    draft_name = os.fspath(draft_model_config.model)
    if os.path.exists(target_name) and os.path.exists(draft_name):
        return os.path.realpath(target_name) == os.path.realpath(draft_name)
    return target_name == draft_name


def _add_dspark_quant_prefix_aliases(
    quant_description: dict[str, Any],
) -> None:
    aliases: dict[str, Any] = {}
    for checkpoint_name, quant_type in quant_description.items():
        if not checkpoint_name.startswith("mtp."):
            continue
        module_name = checkpoint_name
        for source, destination in _DSPARK_QUANT_PREFIX_REPLACEMENTS:
            module_name = module_name.replace(source, destination)
        if module_name == checkpoint_name:
            continue
        existing = quant_description.get(module_name)
        if existing is not None and existing != quant_type:
            raise ValueError(
                f"Conflicting DSpark ModelSlim quantization entries for {checkpoint_name!r} and {module_name!r}."
            )
        aliases[module_name] = quant_type
    quant_description.update(aliases)


def _require_float_entries(
    quant_description: Mapping[str, Any],
    *,
    description: str,
    predicate: Callable[[str], bool],
) -> None:
    matching = {
        name: quant_type
        for name, quant_type in quant_description.items()
        if name.startswith("mtp.") and predicate(name)
    }
    if not matching:
        raise ValueError(f"The DSpark ModelSlim descriptor is missing {description}.")
    non_float = {name: quant_type for name, quant_type in matching.items() if quant_type != _MODELSLIM_FLOAT}
    if non_float:
        raise ValueError(f"DSpark {description} must remain FLOAT in the ModelSlim descriptor, got {non_float}.")


def _attention_component_entries(
    quant_description: Mapping[str, Any],
    *,
    stage: int,
    component: str,
) -> dict[str, Any]:
    return {
        name: quant_description[name]
        for namespace in _DSPARK_ATTENTION_NAMESPACES
        if (name := f"mtp.{stage}.{namespace}.{component}.weight") in quant_description
    }


def _require_attention_component(
    quant_description: Mapping[str, Any],
    *,
    stage: int,
    component: str,
    quant_type: str,
) -> None:
    entries = _attention_component_entries(
        quant_description,
        stage=stage,
        component=component,
    )
    if not entries:
        raise ValueError(
            f"The DSpark ModelSlim descriptor is missing MTP stage {stage} attention component {component!r}."
        )
    invalid = {name: value for name, value in entries.items() if value != quant_type}
    if invalid:
        raise ValueError(
            f"DSpark attention component {component!r} must use {quant_type} for MTP stage {stage}, got {invalid}."
        )


def _validate_attention_entries(
    quant_description: Mapping[str, Any],
    *,
    stage: int,
) -> None:
    for component in _DSPARK_W8A8_ATTENTION_COMPONENTS:
        _require_attention_component(
            quant_description,
            stage=stage,
            component=component,
            quant_type=_MODELSLIM_W8A8_DYNAMIC,
        )
    for component in _DSPARK_FLOAT_ATTENTION_COMPONENTS:
        _require_attention_component(
            quant_description,
            stage=stage,
            component=component,
            quant_type=_MODELSLIM_FLOAT,
        )

    float_entry_names = {
        f"mtp.{stage}.{namespace}.{component}.weight"
        for namespace in _DSPARK_ATTENTION_NAMESPACES
        for component in _DSPARK_FLOAT_ATTENTION_COMPONENTS
    }
    other_attention_weights = {
        name: quant_type
        for name, quant_type in quant_description.items()
        if any(name.startswith(f"mtp.{stage}.{namespace}.") for namespace in _DSPARK_ATTENTION_NAMESPACES)
        and name.endswith(".weight")
        and name not in float_entry_names
    }
    invalid = {
        name: quant_type
        for name, quant_type in other_attention_weights.items()
        if quant_type != _MODELSLIM_W8A8_DYNAMIC
    }
    if invalid:
        raise ValueError(
            "DSpark attention projections outside the official FLOAT "
            f"q_norm/kv_norm/wo_a/wo_b set must use W8A8_DYNAMIC, got {invalid}."
        )


def _validate_w8a8_descriptor(
    quant_description: Mapping[str, Any],
    draft_hf_config: Any,
) -> None:
    if quant_description.get("model_quant_type") != _MODELSLIM_W8A8_DYNAMIC:
        raise ValueError("Ascend DSpark loader requires a ModelSlim W8A8_DYNAMIC descriptor for the 0731 checkpoint.")

    forbidden = {
        name: quant_type
        for name, quant_type in quant_description.items()
        if isinstance(quant_type, str) and any(token in quant_type.upper() for token in ("MXFP", "FP4", "W4A8"))
    }
    if forbidden:
        raise ValueError(f"Ascend DSpark 910B2 loading forbids MXFP/FP4 draft schemes; found {forbidden}.")

    unknown_quant_types = {
        name: quant_type
        for name, quant_type in quant_description.items()
        if name.startswith("mtp.")
        and isinstance(quant_type, str)
        and quant_type not in {_MODELSLIM_FLOAT, _MODELSLIM_W8A8_DYNAMIC}
    }
    if unknown_quant_types:
        raise ValueError(f"The DSpark ModelSlim descriptor contains unsupported quant types: {unknown_quant_types}.")

    num_mtp_layers = int(
        getattr(draft_hf_config, "n_mtp_layers", None)
        or getattr(draft_hf_config, "dspark_num_mtp_layers", _DSPARK_NUM_MTP_LAYERS)
    )
    if num_mtp_layers != _DSPARK_NUM_MTP_LAYERS:
        raise ValueError(f"The 0731 DSpark W8A8 loader supports exactly three MTP layers, got {num_mtp_layers}.")

    for stage in range(num_mtp_layers):
        stage_prefix = f"mtp.{stage}."
        stage_entries = {
            name: quant_type for name, quant_type in quant_description.items() if name.startswith(stage_prefix)
        }
        if not stage_entries:
            raise ValueError(f"The DSpark ModelSlim descriptor is missing {stage_prefix} entries.")

        expert_entries = {
            name: quant_type
            for name, quant_type in stage_entries.items()
            if (".ffn.experts." in name or ".mlp.experts." in name) and name.endswith(".weight")
        }
        shared_expert_entries = {
            name: quant_type
            for name, quant_type in stage_entries.items()
            if (".ffn.shared_experts." in name or ".mlp.shared_experts." in name) and name.endswith(".weight")
        }
        for description, entries in (
            ("routed experts", expert_entries),
            ("shared experts", shared_expert_entries),
        ):
            if not entries:
                raise ValueError(f"The DSpark ModelSlim descriptor is missing {description} for MTP stage {stage}.")
            invalid = {
                name: quant_type for name, quant_type in entries.items() if quant_type != _MODELSLIM_W8A8_DYNAMIC
            }
            if invalid:
                raise ValueError(f"DSpark {description} must use W8A8_DYNAMIC, got {invalid}.")

        _validate_attention_entries(
            quant_description,
            stage=stage,
        )

    last_stage = num_mtp_layers - 1
    required_float_weights = (
        "mtp.0.embed.weight",
        f"mtp.{last_stage}.head.weight",
    )
    for name in required_float_weights:
        if quant_description.get(name) != _MODELSLIM_FLOAT:
            raise ValueError(f"DSpark checkpoint-owned parameter {name!r} must be FLOAT.")

    _require_float_entries(
        quant_description,
        description="normalization parameters",
        predicate=lambda name: ".norm." in name or "_norm." in name,
    )
    _require_float_entries(
        quant_description,
        description="Markov-head parameters",
        predicate=lambda name: ".markov_head." in name,
    )
    _require_float_entries(
        quant_description,
        description="HC parameters",
        predicate=lambda name: ".hc_" in name,
    )
    _require_float_entries(
        quant_description,
        description="confidence-head parameters",
        predicate=lambda name: ".confidence_head." in name,
    )


def _build_draft_quant_config(
    vllm_config: VllmConfig,
    draft_model_config: Any,
) -> Any:
    target_quant_config = vllm_config.quant_config
    if target_quant_config is None:
        return None

    from vllm_ascend.quantization.modelslim_config import (
        AscendModelSlimConfig,
    )

    if not isinstance(target_quant_config, AscendModelSlimConfig):
        return target_quant_config
    if not _same_checkpoint(vllm_config.model_config, draft_model_config):
        raise ValueError("The 0731 ModelSlim DSpark loader requires target and draft to use the same checkpoint.")

    draft_quant_config = copy.deepcopy(target_quant_config)
    draft_quant_config.hf_to_vllm_mapper = None
    draft_quant_config._mapper_applied = False
    _add_dspark_quant_prefix_aliases(draft_quant_config.quant_description)
    _validate_w8a8_descriptor(
        draft_quant_config.quant_description,
        draft_model_config.hf_config,
    )
    return draft_quant_config


def _validate_w8a8_runtime_contract(
    vllm_config: VllmConfig,
    draft_model_config: Any,
    draft_quant_config: Any,
) -> None:
    from vllm_ascend.quantization.modelslim_config import (
        AscendModelSlimConfig,
    )

    if not isinstance(draft_quant_config, AscendModelSlimConfig):
        return

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_hf_config = draft_model_config.hf_config
    if speculative_config.num_speculative_tokens != _DSPARK_BLOCK_SIZE:
        raise ValueError(
            f"The initial Ascend DSpark W8A8 contract requires num_speculative_tokens={_DSPARK_BLOCK_SIZE}."
        )
    if int(draft_hf_config.dspark_block_size) != _DSPARK_BLOCK_SIZE:
        raise ValueError(f"The initial Ascend DSpark W8A8 contract requires dspark_block_size={_DSPARK_BLOCK_SIZE}.")
    if tuple(draft_hf_config.dspark_target_layer_ids) != _DSPARK_TARGET_LAYER_IDS:
        raise ValueError(
            f"The 0731 Ascend DSpark W8A8 contract requires dspark_target_layer_ids={list(_DSPARK_TARGET_LAYER_IDS)}."
        )

    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    if not model_config.enforce_eager:
        raise ValueError("The initial Ascend DSpark W8A8 loader requires enforce_eager=True.")
    if parallel_config.tensor_parallel_size != 8:
        raise ValueError("The initial Ascend DSpark W8A8 loader requires tensor parallel size 8.")
    if not parallel_config.enable_expert_parallel:
        raise ValueError("The initial Ascend DSpark W8A8 loader requires expert parallelism.")
    if parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")


def load_dspark_model(
    target_model: nn.Module,
    vllm_config: VllmConfig,
) -> nn.Module:
    """Load the Ascend DSpark draft model and apply its sharing contract."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    draft_quant_config = _build_draft_quant_config(
        vllm_config,
        draft_model_config,
    )
    _validate_w8a8_runtime_contract(
        vllm_config,
        draft_model_config,
        draft_quant_config,
    )

    from vllm.compilation.backends import set_model_tag

    draft_vllm_config = replace(
        vllm_config,
        model_config=draft_model_config,
        quant_config=draft_quant_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=True,
            backend=speculative_config.attention_backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config,
            model_config=draft_model_config,
        )

    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")

    target_language_model = (
        target_model.get_language_model() if hasattr(target_model, "get_language_model") else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if target_embed is not None and _should_share(
        draft_model,
        "has_own_embed_tokens",
        draft_embed,
        target_embed,
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = getattr(target_model, "lm_head", None)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model,
        "has_own_lm_head",
        draft_lm_head,
        target_lm_head,
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
