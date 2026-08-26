# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pytest

import tests.e2e.nightly.single_node.spec_decode.test_dspark_proposal_inputs_prepare as prepare_harness
from tests.e2e.nightly.single_node.spec_decode.dspark_loader_harness import (
    HarnessNotConfigured,
    parse_launch_context,
    parse_loader_settings,
)
from tests.e2e.nightly.single_node.spec_decode.test_dspark_multi_round_generation import (
    _build_scheduler,
    _cleanup_target_only,
    _materialize_model_output,
    _target_only_runtime,
)
from tools.dspark.m2_5a_common import (
    atomic_write_json,
    atomic_write_jsonl,
    build_execution_plan,
    load_profile_cases,
    sha256_file,
    token_ids_sha256,
)

REAL_DATA_CASE_STARTED = "REAL_DATA_CASE_STARTED"
REAL_DATA_CASE_COMPLETED = "REAL_DATA_CASE_COMPLETED"
REAL_DATA_LIFECYCLE_COMPLETED = "REAL_DATA_LIFECYCLE_COMPLETED"
REAL_DATA_SINGLE_REQUEST_PASS = "REAL_DATA_SINGLE_REQUEST_PASS"
CLEANUP_COMPLETE = "CLEANUP_COMPLETE"
EXPECTED_MAX_MODEL_LEN = 8192
EXPECTED_TP_SIZE = 8
EXPECTED_K = 5
EXPECTED_BLOCK_SIZE = 128


def _marker(name: str, record: dict[str, Any]) -> None:
    print(name + "=" + json.dumps(record, default=str, sort_keys=True), flush=True)


def _request(case: dict[str, Any], request_id: str) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=list(case["prompt_token_ids"]),
        sampling_params=SamplingParams(
            max_tokens=case["output_cap"],
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            seed=0,
            ignore_eos=case["ignore_eos"],
        ),
        pooling_params=None,
    )


def _validate_runtime_contract(runtime: Any, mode: str) -> None:
    config = runtime.vllm_config
    parallel = config.parallel_config
    cache = config.cache_config
    if (
        parallel.tensor_parallel_size != EXPECTED_TP_SIZE
        or parallel.pipeline_parallel_size != 1
        or not parallel.enable_expert_parallel
    ):
        raise RuntimeError("M2.5A requires TP=8, EP enabled, and PP=1.")
    if not config.model_config.enforce_eager:
        raise RuntimeError("M2.5A requires eager execution.")
    if cache.enable_prefix_caching or cache.block_size != EXPECTED_BLOCK_SIZE:
        raise RuntimeError("M2.5A requires prefix caching disabled and block_size=128.")
    speculative = config.speculative_config
    if mode == "target_only" and speculative is not None:
        raise RuntimeError("The target-only M2.5A process must not construct a speculator.")
    if mode == "dspark" and (
        speculative is None or speculative.method != "dspark" or speculative.num_speculative_tokens != EXPECTED_K
    ):
        raise RuntimeError("The DSpark M2.5A process must use method=dspark and K=5.")


def _counter_snapshot(speculator: Any | None) -> dict[str, int | None]:
    if speculator is None:
        return {
            "step_epoch": None,
            "consumer_epoch": None,
            "generated": 0,
            "installed": 0,
            "consumed": 0,
            "terminal_discarded": 0,
        }
    return {
        "step_epoch": speculator._proposal_step_epoch,
        "consumer_epoch": speculator._proposal_consumer_step_epoch,
        "generated": speculator._proposal_generated_count,
        "installed": speculator._proposal_installed_count,
        "consumed": speculator._proposal_consumption_count,
        "terminal_discarded": speculator._terminal_proposal_discard_count,
    }


def _assert_released_request_state(runtime: Any, scheduler: Any, request_id: str) -> None:
    runner = runtime.runner
    registries = {
        "runner request tensors": getattr(runner.req_states, "req_id_to_index", {}),
        "runner input batch": getattr(runner.input_batch, "req_id_to_index", {}),
        "runner cached requests": getattr(runner, "requests", {}),
        "scheduler requests": getattr(scheduler, "requests", {}),
    }
    stale_registries = sorted(name for name, registry in registries.items() if request_id in registry)
    if stale_registries:
        raise RuntimeError(f"Finished request {request_id!r} remains in: {stale_registries}.")
    speculator = getattr(runtime, "speculator", None)
    if speculator is None:
        return
    stale_fields = {
        "published_candidate_tokens": speculator._published_candidate_tokens,
        "published_step_epoch": speculator._published_proposal_step_epoch,
        "published_request_ids": speculator._published_proposal_request_ids,
        "published_request_state_indices": speculator._published_proposal_request_state_indices,
        "current_lifecycle": speculator._current_proposal_lifecycle,
        "prepared_step_epoch": speculator._prepared_step_epoch,
        "context_kv_step_epoch": speculator._context_kv_step_epoch,
        "draft_forward_step_epoch": speculator._draft_forward_step_epoch,
        "markov_attempt_step_epoch": speculator._markov_attempt_step_epoch,
        "markov_step_epoch": speculator._markov_step_epoch,
        "markov_result": speculator._markov_result,
    }
    stale = sorted(name for name, value in stale_fields.items() if value is not None)
    if stale:
        raise RuntimeError(f"Finished request retained DSpark logical state: {stale}.")


def _flush_finished_request(runtime: Any, scheduler: Any, request_id: str) -> None:
    cleanup_output = scheduler.schedule()
    if cleanup_output.total_num_scheduled_tokens != 0:
        raise RuntimeError("Per-request cleanup unexpectedly scheduled another model forward.")
    if cleanup_output.finished_req_ids != {request_id}:
        raise RuntimeError(
            f"Per-request cleanup expected finished_req_ids={{{request_id!r}}}, "
            f"got {cleanup_output.finished_req_ids!r}."
        )
    if runtime.worker.execute_model(cleanup_output) is not None:
        raise RuntimeError("Per-request cleanup unexpectedly returned model output.")
    _assert_released_request_state(runtime, scheduler, request_id)


def _run_case(
    runtime: Any,
    scheduler: Any,
    case: dict[str, Any],
    *,
    mode: str,
    profile: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    request_id = f"m25a-{mode}-{case['request_sequence_index']}-{case['lifecycle_repeat']}-{case['case_id']}"
    request = _request(case, request_id)
    speculator = runtime.speculator if mode == "dspark" else None
    before = _counter_snapshot(speculator)
    target_forward_count = 0
    verification_count = 0
    completed_rounds = 0
    terminal_partial_commit = False
    consumer_epochs: list[int] = []
    _marker(
        REAL_DATA_CASE_STARTED,
        {
            "rank": runtime.launch.rank,
            "mode": mode,
            "case_id": case["case_id"],
            "request_id": request_id,
            "prompt_token_count": case["prompt_token_count"],
            "output_cap": case["output_cap"],
            "ignore_eos": case["ignore_eos"],
        },
    )
    scheduler.add_request(request)
    while scheduler.has_requests():
        scheduler_output = scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            runtime.worker.execute_model(scheduler_output)
            continue
        is_verification = bool(scheduler_output.scheduled_spec_decode_tokens)
        output_length_before = len(request.output_token_ids)
        if request.is_finished():
            raise RuntimeError("A finished request reached another target forward.")
        if runtime.worker.execute_model(scheduler_output) is not None:
            raise RuntimeError("PP=1 execution must defer model output to sampling.")
        target_forward_count += 1
        async_output = runtime.worker.sample_tokens(None)
        if async_output is None:
            raise RuntimeError("Single-request generation returned no sampled output.")
        runtime.torch.npu.synchronize()
        model_output = _materialize_model_output(async_output)
        raw_tokens = list(model_output.sampled_token_ids[0])
        if is_verification:
            verification_count += 1
            completed_rounds += 1
            consumer_epoch = speculator._proposal_consumer_step_epoch
            if consumer_epoch is None:
                raise RuntimeError("DSpark verification did not publish a consumer epoch.")
            consumer_epochs.append(consumer_epoch)
        scheduler.update_from_output(scheduler_output, model_output)
        if mode == "dspark":
            next_draft_ids = runtime.worker.take_draft_token_ids()
            if next_draft_ids is None:
                raise RuntimeError("DSpark did not return its canonical draft-token result.")
            scheduler.update_draft_token_ids(next_draft_ids)
        committed_tokens = list(request.output_token_ids)[output_length_before:]
        if request.is_finished():
            if committed_tokens != raw_tokens[: len(committed_tokens)]:
                raise RuntimeError("Terminal scheduler commit is outside the raw verified prefix.")
            terminal_partial_commit |= len(committed_tokens) < len(raw_tokens)
        elif committed_tokens != raw_tokens:
            raise RuntimeError("Active scheduler commit differs from raw sampled/verified tokens.")

    _flush_finished_request(runtime, scheduler, request_id)
    after = _counter_snapshot(speculator)
    if case["ignore_eos"] and len(request.output_token_ids) != case["output_cap"]:
        raise RuntimeError("Synthetic ignore_eos request stopped before its fixed output cap.")
    output_ids = list(request.output_token_ids)
    finish_reason = request.get_finished_reason()
    record = {
        "rank": runtime.launch.rank,
        "mode": mode,
        "profile": profile,
        "dataset": case["dataset"],
        "case_id": case["case_id"],
        "lifecycle_repeat": case["lifecycle_repeat"],
        "profile_case_index": case["profile_case_index"],
        "request_sequence_index": case["request_sequence_index"],
        "request_id": request_id,
        "proposal_epoch_start": (
            before["step_epoch"] + 1 if mode == "dspark" and after["step_epoch"] > before["step_epoch"] else None
        ),
        "proposal_epoch_end": after["step_epoch"] if mode == "dspark" else None,
        "consumer_epoch_start": consumer_epochs[0] if consumer_epochs else None,
        "consumer_epoch_end": consumer_epochs[-1] if consumer_epochs else None,
        "prompt_token_count": len(request.prompt_token_ids),
        "prompt_token_sha256": token_ids_sha256(request.prompt_token_ids),
        "output_cap": case["output_cap"],
        "ignore_eos": case["ignore_eos"],
        "output_token_count": len(output_ids),
        "output_token_ids": output_ids,
        "output_token_sha256": token_ids_sha256(output_ids),
        "stop_reason": {
            "finish_reason": str(finish_reason),
            "stop_reason": request.stop_reason,
        },
        "usage_prompt_tokens": len(request.prompt_token_ids),
        "usage_completion_tokens": len(output_ids),
        "completed_rounds": completed_rounds,
        "proposal_generated_count": after["generated"] - before["generated"],
        "proposal_installed_count": after["installed"] - before["installed"],
        "proposal_consumed_count": after["consumed"] - before["consumed"],
        "terminal_discarded_proposal_count": after["terminal_discarded"] - before["terminal_discarded"],
        "target_forward_count": target_forward_count,
        "verification_count": verification_count,
        "terminal_partial_commit": terminal_partial_commit,
        "post_finish_target_forward_count": 0,
        "post_finish_verification_count": 0,
        "cleanup_complete": True,
        "state_isolation_verified": True,
        "historical_error_count": 0,
        "manifest_sha256": manifest_sha256,
        "diagnostic_only": True,
        "performance_validated": False,
    }
    if record["prompt_token_sha256"] != case["ordered_prompt_token_sha256"]:
        raise RuntimeError("Runtime prompt token IDs differ from the frozen manifest.")
    _marker(
        "DSPARK_M2_5A_CASE",
        {key: value for key, value in record.items() if key != "output_token_ids"},
    )
    _marker(
        REAL_DATA_CASE_COMPLETED,
        {
            "rank": runtime.launch.rank,
            "mode": mode,
            "case_id": case["case_id"],
            "request_id": request_id,
            "output_token_count": len(output_ids),
            "output_token_sha256": record["output_token_sha256"],
            "stop_reason": record["stop_reason"],
        },
    )
    return record


def _run_plan(
    runtime: Any, plan: list[dict[str, Any]], *, mode: str, profile: str, manifest_hash: str
) -> list[dict[str, Any]]:
    scheduler = _build_scheduler(runtime)
    records = []
    current_repeat = None
    for case in plan:
        if current_repeat is not None and case["lifecycle_repeat"] != current_repeat:
            _marker(
                "DSPARK_M2_5A_LIFECYCLE",
                {"rank": runtime.launch.rank, "mode": mode, "lifecycle_repeat": current_repeat},
            )
            _marker(
                REAL_DATA_LIFECYCLE_COMPLETED,
                {"rank": runtime.launch.rank, "mode": mode, "lifecycle_repeat": current_repeat},
            )
        current_repeat = case["lifecycle_repeat"]
        records.append(_run_case(runtime, scheduler, case, mode=mode, profile=profile, manifest_sha256=manifest_hash))
    if current_repeat is not None:
        _marker(
            "DSPARK_M2_5A_LIFECYCLE",
            {"rank": runtime.launch.rank, "mode": mode, "lifecycle_repeat": current_repeat},
        )
        _marker(
            REAL_DATA_LIFECYCLE_COMPLETED,
            {"rank": runtime.launch.rank, "mode": mode, "lifecycle_repeat": current_repeat},
        )
    return records


def _write_results(result_dir: Path, rank: int, records: list[dict[str, Any]]) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / f"rank-{rank}.jsonl"
    atomic_write_jsonl(path, records)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")
    return path


def _settings_and_matrix() -> tuple[Any, Any, str, str, Path, str, list[dict[str, Any]]]:
    manifest_value = os.environ.get("DSPARK_M25A_MANIFEST")
    if not manifest_value:
        pytest.skip("Set DSPARK_M25A_MANIFEST to the offline frozen manifest.")
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    mode = os.environ.get("DSPARK_M25A_MODE", "")
    profile = os.environ.get("DSPARK_M25A_PROFILE", "")
    result_value = os.environ.get("DSPARK_M25A_RESULT_DIR")
    if mode not in {"target_only", "dspark"}:
        raise ValueError("DSPARK_M25A_MODE must be target_only or dspark.")
    if not result_value:
        raise ValueError("DSPARK_M25A_RESULT_DIR must be set.")
    if settings.tp_size != EXPECTED_TP_SIZE or settings.max_model_len != EXPECTED_MAX_MODEL_LEN:
        raise ValueError("M2.5A requires TP=8 and max_model_len=8192.")
    if settings.num_speculative_tokens != EXPECTED_K:
        raise ValueError("M2.5A requires DSpark K=5.")
    manifest_path = Path(manifest_value).expanduser().resolve()
    _, cases = load_profile_cases(manifest_path, profile)
    lifecycle_repeat = int(os.environ.get("DSPARK_M25A_LIFECYCLE_REPEAT", "1"))
    plan = build_execution_plan(cases, lifecycle_repeat)
    return settings, launch, mode, profile, Path(result_value).expanduser().resolve(), sha256_file(manifest_path), plan


def test_dspark_single_request_realdata_npu() -> None:
    settings, launch, mode, profile, result_dir, manifest_hash, plan = _settings_and_matrix()
    records: list[dict[str, Any]] = []
    try:
        if mode == "target_only":
            contexts = ExitStack()
            state: dict[str, Any] = {}
            target_primary_error: BaseException | None = None
            try:
                runtime = _target_only_runtime(settings, launch, contexts, state)
                _validate_runtime_contract(runtime, mode)
                records = _run_plan(
                    runtime,
                    plan,
                    mode=mode,
                    profile=profile,
                    manifest_hash=manifest_hash,
                )
            except BaseException as exc:
                target_primary_error = exc
                raise
            finally:
                cleanup_errors = _cleanup_target_only(contexts, state)
                _marker(CLEANUP_COMPLETE, {"rank": launch.rank, "mode": mode, "errors": cleanup_errors})
                if cleanup_errors and target_primary_error is None:
                    raise RuntimeError("Target-only cleanup failed: " + "; ".join(cleanup_errors))
        else:
            captured: dict[str, list[dict[str, Any]]] = {}

            def callback(runtime: Any) -> bool:
                _validate_runtime_contract(runtime, mode)
                captured["records"] = _run_plan(
                    runtime,
                    plan,
                    mode=mode,
                    profile=profile,
                    manifest_hash=manifest_hash,
                )
                return True

            if prepare_harness._INITIALIZED_WORKER_CALLBACK is not None:
                raise RuntimeError("The initialized-worker callback is already installed.")
            previous_continue = prepare_harness._CONTINUE_AFTER_VERIFICATION
            prepare_harness._CONTINUE_AFTER_VERIFICATION = True
            prepare_harness._INITIALIZED_WORKER_CALLBACK = callback
            try:
                prepare_harness.test_dspark_proposal_inputs_prepare_only_npu()
            finally:
                prepare_harness._INITIALIZED_WORKER_CALLBACK = None
                prepare_harness._CONTINUE_AFTER_VERIFICATION = previous_continue
            records = captured["records"]
            _marker(CLEANUP_COMPLETE, {"rank": launch.rank, "mode": mode, "errors": []})

        artifact_path = _write_results(result_dir, launch.rank, records)
        summary = {
            "rank": launch.rank,
            "mode": mode,
            "profile": profile,
            "case_executions": len(records),
            "unique_cases": len({record["case_id"] for record in records}),
            "manifest_sha256": manifest_hash,
            "artifact": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "performance_validated": False,
        }
        atomic_write_json(result_dir / f"rank-{launch.rank}.summary.json", summary)
        _marker("DSPARK_M2_5A_SUMMARY", summary)
        _marker(REAL_DATA_SINGLE_REQUEST_PASS, summary)
    except BaseException as exc:
        failure = {
            "rank": launch.rank,
            "mode": mode,
            "profile": profile,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        result_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(result_dir / f"rank-{launch.rank}.failure.json", failure)
        _marker("DSPARK_M2_5A_FAILURE", failure)
        raise
