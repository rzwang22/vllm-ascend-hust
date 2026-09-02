# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools/dspark/build_m2_5a_dataset_assets.py"
VALIDATOR = ROOT / "tools/dspark/validate_m2_5a_results.py"
PERFORMANCE_SUMMARIZER = ROOT / "tools/dspark/summarize_m2_5a_performance.py"
PERFORMANCE_DIAGNOSTICS = ROOT / "tests/e2e/nightly/single_node/spec_decode/m2_5a_performance_diagnostics.py"
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


def test_zero_token_runner_output_is_validated_without_sampling_or_forward_count() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    zero_token_branch = source[
        source.index("if scheduler_output.total_num_scheduled_tokens == 0:") : source.index(
            "is_verification = bool(scheduler_output.scheduled_spec_decode_tokens)"
        )
    ]

    assert "_assert_canonical_zero_token_runner_output(runner_output)" in zero_token_branch
    assert "sample_tokens" not in zero_token_branch
    assert "target_forward_count += 1" not in zero_token_branch
    assert "verification_count += 1" not in zero_token_branch
    assert "continue" in zero_token_branch


def test_generation_error_remains_primary_over_process_cleanup_errors() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    run_case = source[source.index("def _run_case(") : source.index("def _run_plan(")]

    assert "target_primary_error = exc" in source
    assert "if cleanup_errors and target_primary_error is None:" in source
    assert run_case.index("scheduler.update_from_output") < run_case.index("_flush_finished_request(")


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


def test_r6b_forensic_trace_is_targeted_and_disabled_by_default() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert 'FIRST_ROUND_TRACE = "DSPARK_M2_5A_FIRST_ROUND_TRACE"' in source
    assert '"DSPARK_M25A_CASE_ID"' in source
    assert '"DSPARK_M25A_TRACE_FIRST_ROUND"' in source
    assert 'environ.get(_FIRST_ROUND_TRACE_ENV, "0")' in source
    assert "if (first_round or output_index is not None or early_tokens) and case_id is None:" in source
    assert 'case["case_id"] == case_id' in source
    assert "traced_step_count < 2" in source
    assert "raw_tokens or is_verification" in source


def test_r6b_trace_observes_target_proposal_raw_and_commit_without_changing_them() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    host_source = source[source.index("def _host_json_value(") : source.index("def _target_top2_trace(")]
    trace_source = source[source.index("def _target_top2_trace(") : source.index("def _m2_5a_kv_cache_budget(")]
    run_case_source = source[source.index("def _run_case(") : source.index("def _run_plan(")]

    for field in (
        '"target_top1_token_ids"',
        '"target_top2_token_ids"',
        '"target_top1_logits"',
        '"target_top2_logits"',
        '"target_top1_top2_margins"',
        '"published_candidate_tokens"',
        '"consumed_candidate_tokens"',
        '"num_sampled"',
        '"num_rejected"',
        '"raw_sampled_tokens"',
        '"scheduler_committed_tokens"',
        '"expected_greedy_tokens"',
        '"accepted_prefix_length"',
        '"replacement_used"',
        '"bonus_used"',
        '"published_candidates_match_consumed"',
        '"raw_matches_target_top1"',
    ):
        assert field in source
    assert "runner.model.compute_logits(sample_hidden_states)" in trace_source
    assert "runner.sampler.apply_sampling_params(" in trace_source
    assert "runtime.torch.topk(" in trace_source
    assert "type(value).__module__.partition" in host_source
    assert host_source.index('== "numpy"') < host_source.index('detach = getattr(value, "detach", None)')
    assert "input_batch.num_scheduled_tokens" in trace_source
    assert ".detach().cpu().tolist()" not in trace_source
    assert "scheduler.update_from_output(scheduler_output, model_output)" in run_case_source
    assert run_case_source.index("scheduler.update_from_output") < run_case_source.index(
        '"scheduler_committed_tokens":'
    )
    assert "output_token_ids =" not in trace_source
    assert ".reshape(" not in trace_source
    assert "exact_token" not in run_case_source.lower()


def test_r6c_output_index_trace_is_exact_targeted_and_observational() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    selector_source = source[source.index("def _select_forensic_cases(") : source.index("def _host_json_value(")]
    run_case_source = source[source.index("def _run_case(") : source.index("def _run_plan(")]
    next_input_source = source[
        source.index("def _next_model_input_trace(") : source.index("def _m2_5a_kv_cache_budget(")
    ]

    assert 'OUTPUT_INDEX_TRACE = "DSPARK_M2_5A_OUTPUT_INDEX_TRACE"' in source
    assert '_OUTPUT_INDEX_TRACE_ENV = "DSPARK_M25A_TRACE_OUTPUT_INDEX"' in source
    assert "output_index: int | None" in source
    assert 'output_index_value = environ.get(_OUTPUT_INDEX_TRACE_ENV, "").strip()' in selector_source
    assert "requires an exact" in selector_source
    assert "trace_output_index: int | None = None" in run_case_source
    assert "_commit_output_index_trace(" in run_case_source
    assert "_next_model_input_trace(" in run_case_source
    assert "_marker(OUTPUT_INDEX_TRACE" in run_case_source
    for field in (
        '"commit_start_output_index"',
        '"commit_end_output_index_exclusive"',
        '"traced_committed_token"',
        '"scheduled_candidate_tokens"',
        '"replacement_token"',
        '"bonus_token"',
        '"artifact_appended_tokens"',
        '"artifact_append_matches_scheduler_commit"',
        '"request_output_token_count"',
        '"next_model_input_ids"',
        '"next_model_positions"',
        '"next_prefix_token_sha256"',
        '"next_runner_num_computed_tokens"',
        '"next_model_input_contains_traced_token"',
        '"next_runner_contains_traced_token_at_output_index"',
    ):
        assert field in source
    assert "runner.req_states.all_token_ids.gpu" in next_input_source
    assert "scheduler.update_from_output(scheduler_output, model_output)" in run_case_source
    assert run_case_source.index("scheduler.update_from_output") < run_case_source.index('"artifact_appended_tokens"')
    assert '"artifact_append_matches_scheduler_commit"' in run_case_source
    assert "request.output_token_ids =" not in run_case_source
    assert "scheduler.update_draft_token_ids" in run_case_source


def test_existing_m2_4a_and_m2_4b_acceptance_markers_are_unchanged() -> None:
    m2_4a = M2_4A.read_text(encoding="utf-8")
    m2_4b = M2_4B.read_text(encoding="utf-8")

    assert "DSPARK_M24A_VERIFICATION_CONTRACT=" in m2_4a
    assert "DSPARK_M24A_FAILURE=" in m2_4a
    assert 'SINGLE_ROUND_VERIFICATION_PASS = "SINGLE_ROUND_VERIFICATION_PASS"' in m2_4a
    assert "DSPARK_M2_4B_GENERATION=" in m2_4b
    assert "DSPARK_M2_4B_TARGET_ONLY=" in m2_4b
    assert "DSPARK_M2_4B_TARGET_KV_TOPOLOGY=" in m2_4b


def test_r6d_early_trace_is_bounded_and_explicit_about_observer_effect() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    run_case = source[source.index("def _run_case(") : source.index("def _run_plan(")]

    assert 'environ.get(_EARLY_TOKENS_TRACE_ENV, "0")' in source
    assert "MAX_EARLY_TRACE_TOKENS = 16" in source
    assert "Early-range trace must not be combined" in source
    assert "trace_early_tokens: int = 0" in run_case
    assert "trace_early_step = output_length_before < trace_early_tokens" in run_case
    assert "if trace_step:" in run_case
    assert "if pending_early_trace is not None:" in run_case
    assert "NOT the sampler's saved logits" in source
    assert '"logit_row_matches_committed_prefix"' in source
    assert '"next_scheduler_kv_block_ids"' in run_case
    assert "torch.manual_seed" not in source
    assert "request.output_token_ids =" not in source
    assert "setattr(" not in source


def test_r6e_early_trace_uses_rank_local_writer_without_sampler_interposition() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    run_case = source[source.index("def _run_case(") : source.index("def _run_plan(")]
    assert "_marker(EARLY_RANGE_TRACE," not in source
    assert "early_trace_writer.write_step(pending_early_trace)" in run_case
    assert run_case.index("_flush_finished_request(") < run_case.index("early_trace_writer.finish(")
    assert "with rank_trace_writer(trace_result_dir, owner, trace_early_tokens) as writer:" in source
    assert "runner.sampler =" not in source
    assert "runner.sample =" not in source
    assert "register_forward_hook" not in source


def test_performance_mode_removes_only_per_step_synchronize_and_keeps_boundary_sync() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    run_case = source[source.index("def _run_case(") : source.index("def _run_plan(")]

    assert '_PERFORMANCE_ENV = "DSPARK_M25A_PERFORMANCE"' in source
    assert "M2.5A performance mode cannot be combined with a case filter or forensic trace" in source
    assert "M2.5A performance mode requires ASCEND_LAUNCH_BLOCKING=0" in source
    assert "if not performance:\n            runtime.torch.npu.synchronize()" in run_case
    assert run_case.count("runtime.torch.npu.synchronize()") == 3
    assert "runtime.torch.npu.reset_peak_memory_stats(runtime.worker.device)" in run_case
    assert '"timing_boundary"' in run_case
    assert '"prefill_latency_seconds"' in run_case
    assert '"decode_latency_seconds"' in run_case
    assert '"inference_latency_seconds"' in run_case
    assert '"performance_provisional": performance' in run_case
    assert '"bit_exact_validated": False' in run_case
    assert "_verification_token_telemetry(" in run_case
    assert '"accepted_candidate_metrics_source"' in run_case
    assert "_accepted_draft_prefix(" not in run_case


def test_performance_phase_timing_is_outside_generation_interval() -> None:
    harness = HARNESS.read_text(encoding="utf-8")
    target = M2_4B.read_text(encoding="utf-8")
    dspark = PREPARE.read_text(encoding="utf-8")

    for source in (target, dspark):
        assert 'phase_timings["model_load_seconds"]' in source
        assert 'phase_timings["kv_cache_init_seconds"]' in source
    assert "phase_timings = dict(runtime.phase_timings)" in harness
    assert "phase_timings.update(runtime.phase_timings)" in harness
    assert '"phase_timings": phase_timings' in harness


def test_per_case_steady_state_protocol_is_test_only_and_preserves_cleanup_boundary() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    run_case = source[source.index("def _run_case(") : source.index("def _run_plan(")]
    run_plan = source[source.index("def _run_plan(") : source.index("def _write_results(")]

    assert '"DSPARK_M25A_PERFORMANCE_WARMUP_REPEATS"' in source
    assert '"DSPARK_M25A_PERFORMANCE_MEASURED_REPEATS"' in source
    assert '"DSPARK_M25A_PERFORMANCE_CASE_IDS"' in source
    assert 'performance_protocol="per_case_steady_state_v1"' in source
    assert '("warmup", config.warmup_repeats)' in source
    assert '("measured", config.measured_repeats)' in source
    assert "performance_repeat_kind=repeat_kind" in source
    assert "performance_repeat_index=repeat_index" in source
    assert "request_sequence_index=sequence_index" in source
    assert "scheduler = _build_scheduler(runtime)" in run_plan
    assert "_flush_finished_request(" in run_case
    assert '"cleanup_complete": True' in run_case
    assert '"state_isolation_verified": True' in run_case
    assert "runtime.torch.npu.synchronize()" not in run_plan
    assert "model_load" not in run_plan


def test_p04_boundary_diagnostics_are_default_off_rank_local_and_do_not_add_step_sync() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    diagnostics = PERFORMANCE_DIAGNOSTICS.read_text(encoding="utf-8")
    run_case = source[source.index("def _run_case(") : source.index("def _run_plan(")]
    flush_finished = source[source.index("def _flush_finished_request(") : source.index("def _run_case(")]

    assert '"DSPARK_M25A_PERFORMANCE_BOUNDARY_DIAGNOSTICS"' in source
    assert 'environ.get(_PERFORMANCE_BOUNDARY_DIAGNOSTICS_ENV, "0")' in source
    assert "requires exactly one" in source
    assert '"slow_host_step"' in run_case
    assert '"request_start_npu_sync_before"' in run_case
    assert '"request_end_npu_sync_after"' in run_case
    assert '"scheduler_cleanup_before"' in flush_finished
    assert '"logical_runner_kv_proposal_cleanup_after"' in flush_finished
    assert run_case.count("runtime.torch.npu.synchronize()") == 3
    assert '"boundary_semantics": "observe_existing_no_barrier"' in diagnostics
    assert 'self.path.open("x"' in diagnostics
    assert 'root / "performance-boundary"' in diagnostics
    assert "output_token_ids" not in diagnostics


def test_performance_summarizer_preserves_exact_validator_and_reports_provisional_speedups() -> None:
    summarizer = PERFORMANCE_SUMMARIZER.read_text(encoding="utf-8")
    validator = VALIDATOR.read_text(encoding="utf-8")

    assert '"primary_warmup_excluded_decode"' in summarizer
    assert '"warmup_excluded_inference"' in summarizer
    assert '"single-run aggregate"' in summarizer
    assert '"accepted_candidate_metrics_available"' in summarizer
    assert '"matched_case_performance"' in summarizer
    assert "M2_5A_PERFORMANCE_REPORT_GENERATION_PASS" in summarizer
    assert "M2_5A_FORMAL_PERFORMANCE_GATE_" in summarizer
    assert '"formal_performance_gate"' in summarizer
    assert '"steady_state_case_performance"' in summarizer
    assert '"performance_stability_gate"' in summarizer
    assert '"per-case steady-state measured repeats"' in summarizer
    assert "CV <= 0.05" in summarizer
    assert "TIMER_RELATIONSHIPS" in summarizer
    assert '"exact_token_cross_mode_blocking": False' in summarizer
    assert '"cross_mode_exact_token_diagnostics"' in summarizer
    assert '"proposal_installed_count"] != record["proposal_consumed_count"' in summarizer
    assert "HISTORICAL_ERROR_MARKERS" in summarizer
    assert "M2_5A_EXACT_TOKEN_GATE_PASS=" in validator
    assert '"output_token_ids",' in validator
