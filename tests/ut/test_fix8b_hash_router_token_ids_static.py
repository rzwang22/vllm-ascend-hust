import ast
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PLUGIN_ROOT.parent / "vllm-hust"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"missing {class_name}.{method_name} in {path}")


def _calls(method: ast.FunctionDef, callee: str) -> list[ast.Call]:
    result = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == callee
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == callee
        ):
            result.append(node)
    return result


def _passes_name(call: ast.Call, keyword: str, name: str) -> bool:
    return any(
        item.arg == keyword and isinstance(item.value, ast.Name) and item.value.id == name for item in call.keywords
    )


def test_target_model_passes_one_explicit_token_identity_through_every_layer() -> None:
    path = PLUGIN_ROOT / "vllm_ascend/models/deepseek_v4.py"

    model_forward = _class_method(path, "DeepseekV4Model", "forward")
    layer_calls = _calls(model_forward, "layer")
    assert len(layer_calls) == 1
    assert _passes_name(layer_calls[0], "input_ids", "input_ids")

    decoder_forward = _class_method(path, "DeepseekV2DecoderLayer", "forward")
    mlp_calls = _calls(decoder_forward, "mlp")
    assert len(mlp_calls) == 1
    assert _passes_name(mlp_calls[0], "input_ids", "input_ids")

    moe_forward = _class_method(path, "DeepseekV4MoE", "forward")
    expert_calls = _calls(moe_forward, "experts")
    assert len(expert_calls) == 2
    assert all(_passes_name(call, "input_ids", "input_ids") for call in expert_calls)


def test_mtp_and_dspark_draft_callsites_pass_their_real_token_ids() -> None:
    mtp_path = PLUGIN_ROOT / "vllm_ascend/models/deepseek_v4_mtp.py"
    mtp_forward = _class_method(mtp_path, "DeepSeekMultiTokenPredictorLayer", "forward")
    mtp_calls = _calls(mtp_forward, "mtp_block")
    assert len(mtp_calls) == 1
    assert _passes_name(mtp_calls[0], "input_ids", "input_ids")

    dspark_path = PLUGIN_ROOT / "vllm_ascend/models/deepseek_v4_dspark.py"
    dspark_forward = _class_method(dspark_path, "DeepseekV4DSparkModel", "forward")
    layer_calls = _calls(dspark_forward, "layer")
    assert len(layer_calls) == 1
    assert _passes_name(layer_calls[0], "input_ids", "input_ids")


def test_ascend_moe_keeps_token_identity_explicit_across_all_routing_consumers() -> None:
    path = PLUGIN_ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py"
    source = path.read_text()
    assert "input_ids = _EXTRA_CTX.input_ids" not in source

    runner_forward = _class_method(path, "AscendMoERunner", "_forward_impl")
    no_shared_calls = _calls(runner_forward, "no_shared_forward_impl")
    shared_calls = _calls(runner_forward, "shared_forward_impl")
    assert len(no_shared_calls) == 1
    assert len(shared_calls) == 1
    assert _passes_name(no_shared_calls[0], "input_ids", "input_ids")
    assert _passes_name(shared_calls[0], "input_ids", "input_ids")

    no_shared_forward = _class_method(path, "AscendMoERunner", "no_shared_forward_impl")
    apply_calls = _calls(no_shared_forward, "apply")
    selector_calls = _calls(no_shared_forward, "select_experts")
    assert len(apply_calls) == 1
    assert len(selector_calls) == 1
    assert _passes_name(apply_calls[0], "input_ids", "input_ids")
    assert _passes_name(selector_calls[0], "input_ids", "input_ids")

    shared_forward = _class_method(path, "AscendMoERunner", "shared_forward_impl")
    delegated_calls = _calls(shared_forward, "no_shared_forward_impl")
    assert len(delegated_calls) == 1
    assert _passes_name(delegated_calls[0], "input_ids", "input_ids")


def test_modelslim_w8a8_adapter_forwards_ids_only_for_hash_routing() -> None:
    adapter_path = PLUGIN_ROOT / "vllm_ascend/quantization/method_adapters.py"
    adapter = _class_method(adapter_path, "AscendFusedMoEMethod", "apply")
    assignments = [
        node
        for node in ast.walk(adapter)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "input_ids"
    ]
    assert any(
        isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "apply_kwargs"
        for assignment in assignments
        for target in assignment.targets
    )
    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and ast.unparse(node.test) == "self.tid2eid is not None"
        for node in ast.walk(adapter)
    )

    scheme_path = PLUGIN_ROOT / "vllm_ascend/quantization/methods/w8a8_dynamic.py"
    scheme = _class_method(scheme_path, "AscendW8A8DynamicFusedMoEMethod", "apply")
    selector_calls = _calls(scheme, "select_experts")
    assert len(selector_calls) == 1
    assert _passes_name(selector_calls[0], "input_ids", "input_ids")


def test_current_core_custom_op_abi_already_carries_optional_input_ids() -> None:
    runner_path = CORE_ROOT / "vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
    runner_forward = _class_method(runner_path, "MoERunner", "forward")
    forward_impl = _class_method(runner_path, "MoERunner", "_forward_impl")
    assert any(arg.arg == "input_ids" for arg in runner_forward.args.args)
    assert any(arg.arg == "input_ids" for arg in forward_impl.args.args)

    custom_op_calls = _calls(runner_forward, "_forward_entry")
    assert len(custom_op_calls) == 1
    assert any(isinstance(arg, ast.Name) and arg.id == "input_ids" for arg in custom_op_calls[0].args)


def test_selector_still_owns_hash_input_validation_and_padding_normalization() -> None:
    source = (PLUGIN_ROOT / "vllm_ascend/ops/fused_moe/experts_selector.py").read_text()
    assert "if input_ids is None:" in source
    assert "if input_ids.ndim != 1 or input_ids.shape[0] != router_logits.shape[0]:" in source
    assert "input_ids = input_ids.to(torch.int64)" in source
    assert "input_ids = torch.where(input_ids == -1, 0, input_ids)" in source
