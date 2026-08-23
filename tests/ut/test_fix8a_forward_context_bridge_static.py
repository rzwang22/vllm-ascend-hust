from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PLUGIN_ROOT.parent / "vllm-hust"


def _read(path: Path) -> str:
    return path.read_text()


def test_core_runner_passes_real_model_inputs_to_platform_context() -> None:
    source = _read(CORE_ROOT / "vllm/v1/worker/gpu/model_runner.py")
    context_call = source[source.index("with set_forward_context(") :]
    assert 'input_ids=model_inputs["input_ids"]' in context_call
    assert "model_instance=self.model" in context_call


def test_core_forward_context_forwards_optional_model_inputs_to_platform() -> None:
    source = _read(CORE_ROOT / "vllm/forward_context.py")
    assert "input_ids: torch.Tensor | None = None" in source
    assert "model_instance: torch.nn.Module | None = None" in source
    hook = source[source.index("current_platform.set_additional_forward_context(") :]
    assert "input_ids=input_ids" in hook
    assert "model_instance=model_instance" in hook


def test_plugin_uses_one_builder_and_proxy_for_v1_v2_moe_context() -> None:
    context = _read(PLUGIN_ROOT / "vllm_ascend/ascend_forward_context.py")
    platform = _read(PLUGIN_ROOT / "vllm_ascend/platform.py")
    selector = _read(PLUGIN_ROOT / "vllm_ascend/ops/fused_moe/experts_selector.py")
    fused_moe = _read(PLUGIN_ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py")
    assert "def build_ascend_forward_context(" in context
    assert "extra_context = build_ascend_forward_context(" in context
    assert "return build_ascend_forward_context(" in platform
    assert '"input_ids",' in context
    assert "forward_context.input_ids" not in selector
    assert "input_ids = _EXTRA_CTX.input_ids" in fused_moe
    assert "_EXTRA_CTX.moe_comm_type" in selector
    assert "DeepSeek V4 hash MoE routing requires input_ids" in selector


def test_prepare_only_harness_still_executes_real_worker_forward() -> None:
    harness = _read(PLUGIN_ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_proposal_inputs_prepare.py")
    assert "output = worker.execute_model(scheduler_output)" in harness
    assert "forward_context.input_ids" not in harness
