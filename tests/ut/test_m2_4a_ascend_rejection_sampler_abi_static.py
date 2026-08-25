# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path

PLUGIN_ROOT = Path(__file__).parents[2]
CORE_ROOT = PLUGIN_ROOT.parent / "vllm-hust"
PLUGIN_REJECTION = PLUGIN_ROOT / "vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py"
CORE_REJECTION = CORE_ROOT / "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
CORE_CALLER = CORE_ROOT / "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
PATCH_TRITON = PLUGIN_ROOT / "vllm_ascend/patch/worker/patch_v2/patch_triton.py"
PLATFORM = PLUGIN_ROOT / "vllm_ascend/platform.py"


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _parameter_defaults(function: ast.FunctionDef) -> dict[str, ast.expr]:
    parameters = [*function.args.posonlyargs, *function.args.args]
    default_parameters = parameters[len(parameters) - len(function.args.defaults) :]
    defaults = dict(zip((parameter.arg for parameter in default_parameters), function.args.defaults))
    defaults.update(
        (parameter.arg, default)
        for parameter, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
        if default is not None
    )
    return defaults


def test_plugin_rejection_sampler_explicitly_matches_core_caller_keywords() -> None:
    plugin = _function(PLUGIN_REJECTION, "rejection_sample")
    core = _function(CORE_REJECTION, "rejection_sample")
    caller = _class_method(CORE_CALLER, "RejectionSampler", "__call__")

    plugin_arguments = [*plugin.args.posonlyargs, *plugin.args.args]
    core_arguments = [*core.args.posonlyargs, *core.args.args]
    plugin_parameters = {parameter.arg for parameter in plugin_arguments}
    core_parameters = {parameter.arg for parameter in core_arguments}
    rejection_call = next(
        node
        for node in ast.walk(caller)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "rejection_sample"
    )
    caller_keywords = {keyword.arg for keyword in rejection_call.keywords}

    assert caller_keywords == {"use_fp64", "use_block_verification"}
    assert caller_keywords <= core_parameters
    assert caller_keywords <= plugin_parameters
    assert plugin.args.vararg is None
    assert plugin.args.kwarg is None
    assert [parameter.arg for parameter in plugin_arguments] == [parameter.arg for parameter in core_arguments]

    defaults = _parameter_defaults(plugin)
    assert isinstance(defaults["use_block_verification"], ast.Constant)
    assert defaults["use_block_verification"].value is False
    use_block_verification = next(
        parameter for parameter in plugin_arguments if parameter.arg == "use_block_verification"
    )
    assert use_block_verification.annotation is not None
    assert ast.unparse(use_block_verification.annotation) == "bool"


def test_block_guard_is_immediate_and_does_not_retry_type_errors() -> None:
    function = _function(PLUGIN_REJECTION, "rejection_sample")
    first_statement = function.body[0]

    assert isinstance(first_statement, ast.If)
    assert ast.unparse(first_statement.test) == "use_block_verification"
    assert any(isinstance(node, ast.Raise) for node in ast.walk(first_statement))
    assert "Ascend V2 rejection sampler does not support block verification" in ast.unparse(first_statement)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(function))
    assert "inspect.signature" not in ast.unparse(function)


def test_patch_replaces_both_core_symbols_with_the_same_npu_function() -> None:
    tree = ast.parse(PATCH_TRITON.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "npu_rejection_sample"
    ]

    targets = {ast.unparse(node.targets[0]) for node in assignments}
    assert targets == {
        "rejection_sampler.rejection_sample",
        "rejection_sampler_utils.rejection_sample",
    }


def test_platform_rejects_block_verification_before_model_initialization() -> None:
    validator = _class_method(PLATFORM, "NPUPlatform", "_validate_rejection_sampling_config")
    updater = _class_method(PLATFORM, "NPUPlatform", "check_and_update_config")
    validator_source = ast.unparse(validator)
    updater_source = ast.unparse(updater)

    assert "rejection_sample_method" in validator_source
    assert "== 'block'" in validator_source
    assert "Ascend V2 rejection sampler does not support block verification" in validator_source
    assert "raise NotImplementedError" in validator_source
    assert "cls._validate_rejection_sampling_config(vllm_config)" in updater_source
    assert updater_source.index("cls._validate_rejection_sampling_config(vllm_config)") < updater_source.index(
        "if vllm_config.model_config is None"
    )
