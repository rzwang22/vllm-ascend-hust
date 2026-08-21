from types import SimpleNamespace

import pytest
from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.config.speculative_capability import resolve_speculative_capability
from vllm.platforms import Platform, current_platform

from vllm_ascend.platform import NPUPlatform
from vllm_ascend.spec_decode import (
    DSPARK_PROPOSER_IDENTITY,
    DSparkRuntimeNotWiredError,
    get_spec_decode_method,
)
from vllm_ascend.worker.v2.spec_decode import (
    init_speculator as init_v2_speculator,
)
from vllm_ascend.worker.v2.spec_decode.dspark import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.eagle import (
    init_speculator as init_active_v2_speculator,
)


def test_npu_platform_registers_dspark_additively() -> None:
    portable_capabilities = Platform.get_speculative_proposer_capabilities()

    capabilities = NPUPlatform.get_speculative_proposer_capabilities()

    assert capabilities["dspark"] == DSPARK_PROPOSER_IDENTITY
    assert {method: capabilities[method] for method in portable_capabilities} == portable_capabilities


def test_dspark_capability_resolves_to_ascend_selector() -> None:
    capability = resolve_speculative_capability(
        requested_method="dspark",
        hf_config={
            "model_type": "deepseek_v4",
            "dspark_block_size": 5,
        },
        platform="npu",
        registered_proposers=NPUPlatform.get_speculative_proposer_capabilities(),
    )

    assert capability.status == "enabled"
    assert capability.requested_method == "dspark"
    assert capability.detected_checkpoint_method == "dspark"
    assert capability.resolved_method == "dspark"
    assert capability.proposer == DSPARK_PROPOSER_IDENTITY


def test_speculative_config_parses_dspark_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_model_config = SimpleNamespace(
        model="deepseek-ai/DeepSeek-V4",
        tokenizer="target-tokenizer",
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path="",
        allowed_media_domains=None,
        dtype="auto",
        seed=0,
        tokenizer_revision=None,
        max_model_len=128,
        quantization=None,
        enforce_eager=True,
        max_logprobs=20,
        config_format="hf",
        hf_text_config=SimpleNamespace(
            model_type="deepseek_v4",
            dspark_block_size=5,
        ),
    )
    draft_hf_config = SimpleNamespace(
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
    )
    draft_model_config = SimpleNamespace(
        model=target_model_config.model,
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
        quantization=None,
        verify_with_parallel_config=lambda _parallel_config: None,
    )

    monkeypatch.setattr(
        "vllm.config.speculative.ModelConfig",
        lambda **_kwargs: draft_model_config,
    )
    monkeypatch.setattr(SpeculativeConfig, "update_arch_", lambda _self: None)
    monkeypatch.setattr(current_platform, "device_name", "npu")
    monkeypatch.setattr(
        current_platform,
        "get_speculative_proposer_capabilities",
        NPUPlatform.get_speculative_proposer_capabilities,
    )

    speculative_config = SpeculativeConfig(
        method="dspark",
        model=target_model_config.model,
        target_model_config=target_model_config,
        target_parallel_config=ParallelConfig(),
        num_speculative_tokens=1,
    )

    assert speculative_config.method == "dspark"
    assert speculative_config.capability.status == "enabled"
    assert speculative_config.capability.proposer == DSPARK_PROPOSER_IDENTITY


def test_v1_dspark_selection_fails_without_fallback() -> None:
    with pytest.raises(DSparkRuntimeNotWiredError, match="not yet wired"):
        get_spec_decode_method("dspark", None, None, None)


@pytest.mark.parametrize(
    "init_speculator",
    [init_v2_speculator, init_active_v2_speculator],
)
def test_v2_dspark_selection_constructs_dspark_speculator(init_speculator) -> None:
    speculative_config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(dspark_noise_token_id=128799)),
        num_speculative_tokens=3,
        use_dspark=lambda: True,
        use_eagle=lambda: True,
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    speculator = init_speculator(vllm_config, None)

    assert type(speculator) is AscendDSparkSpeculator
