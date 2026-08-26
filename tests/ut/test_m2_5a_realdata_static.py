# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools/dspark/build_m2_5a_dataset_assets.py"
VALIDATOR = ROOT / "tools/dspark/validate_m2_5a_results.py"
HARNESS = ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_single_request_realdata.py"
M2_4A = ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_single_round_verification.py"
M2_4B = ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_multi_round_generation.py"
PREPARE = ROOT / "tests/e2e/nightly/single_node/spec_decode/test_dspark_proposal_inputs_prepare.py"


def test_asset_builder_is_offline_and_uses_exact_template_contract() -> None:
    source = BUILDER.read_text(encoding="utf-8")

    assert "from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer" in source
    assert "DeepseekV4Tokenizer.from_pretrained" in source
    assert "local_files_only=True" in source
    assert "AutoTokenizer" not in source
    assert "AutoConfig" not in source
    assert "trust_remote_code" not in source
    assert "apply_chat_template" in source
    assert "add_generation_prompt=True" in source
    assert "add_special_tokens=False" in source
    assert "truncate" not in source.lower()
    assert "requests.get" not in source
    assert "hf_hub_download" not in source


def test_m2_5a_engine_args_disable_prefix_cache_before_config_creation() -> None:
    harness = HARNESS.read_text(encoding="utf-8")
    target_builder = M2_4B.read_text(encoding="utf-8")
    dspark_builder = PREPARE.read_text(encoding="utf-8")

    assert harness.count("enable_prefix_caching=False") == 2
    assert "enable_prefix_caching=enable_prefix_caching" in target_builder
    assert "enable_prefix_caching=launch_config.enable_prefix_caching" in dspark_builder
    assert target_builder.index("enable_prefix_caching=enable_prefix_caching") < target_builder.index(
        "vllm_config = engine_args.create_engine_config()"
    )
    assert dspark_builder.index("enable_prefix_caching=launch_config.enable_prefix_caching") < dspark_builder.index(
        "vllm_config = engine_args.create_engine_config()"
    )
    assert "cache.enable_prefix_caching =" not in harness
    assert "cache_config.enable_prefix_caching =" not in harness


def test_m2_5a_budget_is_preflighted_and_forwarded_without_changing_other_harness_defaults() -> None:
    harness = HARNESS.read_text(encoding="utf-8")
    target_builder = M2_4B.read_text(encoding="utf-8")
    dspark_builder = PREPARE.read_text(encoding="utf-8")

    assert 'if "DSPARK_KV_CACHE_BYTES" not in environ:' in harness
    assert "MINIMUM_KV_CACHE_BYTES = 2 * 1024 * 1024 * 1024" in harness
    assert harness.count("kv_cache_bytes=kv_cache_bytes") == 2
    assert "enable_prefix_caching: bool | None = None" in target_builder
    assert "kv_cache_bytes: int | None = None" in target_builder
    assert "class _PrepareOnlyLaunchConfig:" in dspark_builder
    assert "enable_prefix_caching: bool | None = None" in dspark_builder
    assert "kv_cache_bytes: int | None = None" in dspark_builder
    assert "_PREPARE_ONLY_LAUNCH_CONFIG.reset(token)" in dspark_builder
    assert "def test_dspark_proposal_inputs_prepare_only_npu() -> None:" in dspark_builder
    assert "prepare_only_launch_config(" in harness


def test_m2_5a_runtime_marker_reads_final_config_fields() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert 'RUNTIME_CONTRACT = "DSPARK_M2_5A_RUNTIME_CONTRACT"' in source
    for expression in (
        '"max_model_len": config.model_config.max_model_len',
        '"block_size": cache.block_size',
        '"prefix_caching_enabled": cache.enable_prefix_caching',
        '"enforce_eager": config.model_config.enforce_eager',
        '"tp_size": parallel.tensor_parallel_size',
        '"pp_size": parallel.pipeline_parallel_size',
        '"expert_parallel": parallel.enable_expert_parallel',
    ):
        assert expression in source


def test_harness_uses_real_scheduler_and_strict_greedy_sampling() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert "scheduler.add_request(request)" in source
    assert "runtime.worker.execute_model(scheduler_output)" in source
    assert "scheduler.update_from_output(scheduler_output, model_output)" in source
    assert "temperature=0.0" in source
    assert "top_p=1.0" in source
    assert "top_k=-1" in source
    assert "seed=0" in source
    assert "DSPARK_M25A_MODE" in source
    assert "target_only" in source and "dspark" in source


def test_per_request_cleanup_and_state_isolation_are_explicit() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert "class _FinishedRequestLifecycle:" in source
    assert "finished_lifecycle.assert_delivered_once()" in source
    assert "cleanup_output.finished_req_ids" in source
    assert "must not repeat or" in source
    assert "retain finished request events" in source
    assert "kv_cache_manager.get_block_ids(request_id)" in source
    assert "coordinator.single_type_managers" in source
    assert '"scheduler running"' in source
    assert '"scheduler waiting"' in source
    assert '"scheduler skipped waiting"' in source
    assert "_assert_released_request_state(runtime, scheduler, request_id)" in source
    for field in (
        "_published_candidate_tokens",
        "_published_proposal_step_epoch",
        "_published_proposal_request_ids",
        "_published_proposal_request_state_indices",
        "_current_proposal_lifecycle",
        "_prepared_step_epoch",
        "_context_kv_step_epoch",
        "_draft_forward_step_epoch",
        "_markov_attempt_step_epoch",
        "_markov_step_epoch",
        "_markov_result",
    ):
        assert field in source


def test_generation_error_remains_primary_over_process_cleanup_errors() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert "target_primary_error = exc" in source
    assert "if cleanup_errors and target_primary_error is None:" in source
    assert source.index("scheduler.update_from_output") < source.index(
        "_flush_finished_request(runtime, scheduler, finished_lifecycle)"
    )


def test_stochastic_sampling_and_invalid_runtime_shape_fail_closed() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert "EXPECTED_MAX_MODEL_LEN = 8192" in source
    assert "EXPECTED_TP_SIZE = 8" in source
    assert "EXPECTED_K = 5" in source
    assert "EXPECTED_BLOCK_SIZE = 128" in source
    assert "SamplingParams(" in source
    assert "temperature=0.0" in source
    assert "os.environ" not in source[source.index("def _request(") : source.index("def _counter_snapshot(")]


def test_result_validator_enforces_exact_tokens_usage_and_rank_consistency() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")

    for field in (
        "output_token_ids",
        "output_token_count",
        "stop_reason",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "post_finish_target_forward_count",
        "post_finish_verification_count",
        "cleanup_complete",
        "state_isolation_verified",
    ):
        assert field in source
    assert "Expected rank artifacts 0.." in source
    assert "M2_5A_EXACT_TOKEN_GATE_PASS=" in source


def test_existing_m2_4a_and_m2_4b_acceptance_markers_are_unchanged() -> None:
    m2_4a = M2_4A.read_text(encoding="utf-8")
    m2_4b = M2_4B.read_text(encoding="utf-8")

    assert "DSPARK_M24A_VERIFICATION_CONTRACT=" in m2_4a
    assert "DSPARK_M24A_FAILURE=" in m2_4a
    assert 'SINGLE_ROUND_VERIFICATION_PASS = "SINGLE_ROUND_VERIFICATION_PASS"' in m2_4a
    assert "DSPARK_M2_4B_GENERATION=" in m2_4b
    assert "DSPARK_M2_4B_TARGET_ONLY=" in m2_4b
    assert "DSPARK_M2_4B_TARGET_KV_TOPOLOGY=" in m2_4b
