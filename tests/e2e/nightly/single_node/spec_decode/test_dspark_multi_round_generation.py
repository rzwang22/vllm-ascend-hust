# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import gc
import json
import os
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any

import pytest

import tests.e2e.nightly.single_node.spec_decode.test_dspark_proposal_inputs_prepare as prepare_harness
from tests.e2e.nightly.single_node.spec_decode.dspark_loader_harness import (
    HarnessNotConfigured,
    dspark_loader_config_context,
    parse_launch_context,
    parse_loader_settings,
)
from tests.e2e.nightly.single_node.spec_decode.test_dspark_kv_cache_init import (
    _kv_cache_budget,
)
from tests.e2e.nightly.single_node.spec_decode.test_dspark_single_round_verification import (
    _build_scheduler,
    _expected_greedy_verification,
    _materialize_model_output,
    _target_greedy_choices,
)

MAX_NEW_TOKENS = 16
MIN_COMPLETED_ROUNDS = 3
REQUEST_ID = "dspark-m24b-multi-round"


def _validate_scheduler_commit(
    raw_verified_tokens: list[int],
    committed_tokens: list[int],
    *,
    request_finished: bool,
) -> bool:
    """Enforce the optimistic runner/scheduler two-case safety contract."""
    if request_finished:
        if committed_tokens != raw_verified_tokens[: len(committed_tokens)]:
            raise RuntimeError("A finished request committed tokens outside the raw verified prefix.")
        return committed_tokens == raw_verified_tokens
    if committed_tokens != raw_verified_tokens:
        raise RuntimeError("An active request cannot diverge from the optimistic raw verified token sequence.")
    return True


def _generation_request(runtime: Any, request_id: str) -> Any:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    token_id = prepare_harness._prompt_token_id(runtime.settings.target_config)
    return Request(
        request_id=request_id,
        prompt_token_ids=[token_id],
        sampling_params=SamplingParams(
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            seed=0,
        ),
        pooling_params=None,
    )


def _run_target_only_generation(runtime: Any) -> dict[str, Any]:
    scheduler = _build_scheduler(runtime)
    request = _generation_request(runtime, "dspark-m24b-target-only")
    scheduler.add_request(request)
    target_forward_count = 0

    while scheduler.has_requests():
        scheduler_output = scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            runtime.worker.execute_model(scheduler_output)
            continue
        if runtime.worker.execute_model(scheduler_output) is not None:
            raise RuntimeError("PP=1 target-only execution must defer output to sampling.")
        target_forward_count += 1
        async_output = runtime.worker.sample_tokens(None)
        if async_output is None:
            raise RuntimeError("Target-only generation returned no sampled output.")
        runtime.torch.npu.synchronize()
        model_output = _materialize_model_output(async_output)
        scheduler.update_from_output(scheduler_output, model_output)

    if len(request.output_token_ids) != MAX_NEW_TOKENS:
        raise RuntimeError(
            "The fixed target-only prompt produced an early stop; choose a "
            "prompt whose first 16 greedy tokens contain no EOS/stop token."
        )
    output = {
        "rank": runtime.launch.rank,
        "request_id": request.request_id,
        "prompt_token_ids": list(request.prompt_token_ids),
        "output_token_ids": list(request.output_token_ids),
        "output_count": len(request.output_token_ids),
        "stop_reason": str(request.stop_reason),
        "target_forward_count": target_forward_count,
    }
    print(
        "DSPARK_M2_4B_TARGET_ONLY=" + json.dumps(output, sort_keys=True),
        flush=True,
    )
    return output


def _run_dspark_multi_round(runtime: Any) -> dict[str, Any]:
    torch = runtime.torch
    worker = runtime.worker
    speculator = runtime.speculator
    scheduler = _build_scheduler(runtime)
    request = _generation_request(runtime, REQUEST_ID)
    scheduler.add_request(request)

    target_forward_count = 0
    verification_count = 0
    completed_rounds = 0
    post_finish_target_forward_count = 0
    post_finish_verification_count = 0
    previous_proposal_epoch: int | None = None
    round_records: list[dict[str, Any]] = []

    while scheduler.has_requests():
        scheduler_output = scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            worker.execute_model(scheduler_output)
            continue

        scheduled_candidates = speculator._published_candidate_tokens
        scheduled_lifecycle = speculator._current_proposal_lifecycle
        is_verification = bool(scheduler_output.scheduled_spec_decode_tokens)
        if is_verification:
            if scheduled_candidates is None or scheduled_lifecycle is None:
                raise RuntimeError("A verification step has no owned DSpark candidate tensor.")
            proposal_tokens = scheduled_candidates[0].cpu().tolist()
            proposal_epoch = scheduled_lifecycle.proposal_epoch
            output_length_before = len(request.output_token_ids)
            remaining_budget_before = MAX_NEW_TOKENS - output_length_before
        else:
            proposal_tokens = []
            proposal_epoch = None
            output_length_before = len(request.output_token_ids)
            remaining_budget_before = MAX_NEW_TOKENS - output_length_before

        if request.is_finished():
            post_finish_target_forward_count += 1
            raise RuntimeError("A finished DSpark request reached another target forward.")
        if worker.execute_model(scheduler_output) is not None:
            raise RuntimeError("PP=1 DSpark execution must defer output to sampling.")
        target_forward_count += 1
        target_selected = _target_greedy_choices(runtime) if is_verification else None
        async_output = worker.sample_tokens(None)
        if async_output is None:
            raise RuntimeError("DSpark generation returned no sampled output.")
        torch.npu.synchronize()
        model_output = _materialize_model_output(async_output)
        raw_verified_tokens = list(model_output.sampled_token_ids[0])

        if is_verification:
            verification_count += 1
            consumed = speculator._last_consumed_proposal_lifecycle
            if consumed is None or consumed.proposal_epoch != proposal_epoch:
                raise RuntimeError("The verification step consumed the wrong proposal epoch.")
            if not consumed.installed or not consumed.consumed:
                raise RuntimeError("The verified proposal was not installed and consumed exactly once.")
            if previous_proposal_epoch is not None and proposal_epoch != previous_proposal_epoch:
                raise RuntimeError("DSpark proposal epochs are not strictly consecutive.")

        scheduler.update_from_output(scheduler_output, model_output)
        next_draft_ids = worker.take_draft_token_ids()
        if next_draft_ids is None:
            raise RuntimeError("DSpark did not publish a canonical draft-token result.")
        scheduler.update_draft_token_ids(next_draft_ids)
        committed_tokens = list(request.output_token_ids)[output_length_before:]
        request_finished = request.is_finished()
        raw_equals_committed = _validate_scheduler_commit(
            raw_verified_tokens,
            committed_tokens,
            request_finished=request_finished,
        )

        if not is_verification:
            initial = speculator._current_proposal_lifecycle
            if initial is None or len(request.spec_token_ids) != runtime.settings.num_speculative_tokens:
                raise RuntimeError("The producer step did not install the first K-token proposal.")
            previous_proposal_epoch = initial.proposal_epoch
            continue

        assert target_selected is not None
        expected_tokens, accepted_prefix_length, replacement_used, bonus_used = _expected_greedy_verification(
            proposal_tokens,
            target_selected.cpu().view(-1).tolist(),
        )
        if raw_verified_tokens != expected_tokens:
            raise RuntimeError(
                "The raw rejection-sampler output violates the independently derived greedy verification contract."
            )
        completed_rounds += 1
        next_lifecycle = speculator._current_proposal_lifecycle
        next_proposal_epoch = next_lifecycle.proposal_epoch if next_lifecycle is not None else None
        next_installed = not request_finished and len(request.spec_token_ids) == runtime.settings.num_speculative_tokens
        if not request_finished and not next_installed:
            raise RuntimeError("An active request did not install its next proposal.")
        if request_finished and request.spec_token_ids:
            raise RuntimeError("A finished request retained an optimistic proposal.")

        record = {
            "rank": runtime.launch.rank,
            "round_index": completed_rounds - 1,
            "request_id": request.request_id,
            "proposal_epoch": proposal_epoch,
            "consumer_epoch": speculator._proposal_consumer_step_epoch,
            "proposal_tokens": proposal_tokens,
            "proposal_length": len(proposal_tokens),
            "proposal_generated": True,
            "proposal_returned_to_core": True,
            "proposal_installed": True,
            "proposal_consumed": True,
            "raw_verified_tokens": raw_verified_tokens,
            "raw_verified_count": len(raw_verified_tokens),
            "scheduler_committed_tokens": committed_tokens,
            "scheduler_committed_count": len(committed_tokens),
            "raw_equals_committed": raw_equals_committed,
            "accepted_prefix_length": accepted_prefix_length,
            "replacement_used": replacement_used,
            "bonus_used": bonus_used,
            "output_length_before": output_length_before,
            "output_length_after": len(request.output_token_ids),
            "remaining_budget_before": remaining_budget_before,
            "remaining_budget_after": MAX_NEW_TOKENS - len(request.output_token_ids),
            "request_finished": request_finished,
            "stop_reason": str(request.stop_reason),
            "next_proposal_epoch": next_proposal_epoch,
            "next_proposal_installed": next_installed,
        }
        round_records.append(record)
        print(
            "DSPARK_M2_4B_ROUND=" + json.dumps(record, sort_keys=True),
            flush=True,
        )
        previous_proposal_epoch = next_proposal_epoch

    terminal = speculator._terminal_proposal_lifecycle
    terminal_record = {
        "rank": runtime.launch.rank,
        "proposal_epoch": terminal.proposal_epoch if terminal is not None else None,
        "generated": terminal.generated if terminal is not None else False,
        "returned_to_core": (terminal.returned_to_core if terminal is not None else False),
        "installed": terminal.installed if terminal is not None else False,
        "consumed": terminal.consumed if terminal is not None else False,
        "discarded_terminal": (terminal.discarded_terminal if terminal is not None else False),
        "post_finish_target_forward_count": post_finish_target_forward_count,
        "post_finish_verification_count": post_finish_verification_count,
    }
    print(
        "DSPARK_M2_4B_TERMINAL_PROPOSAL=" + json.dumps(terminal_record, sort_keys=True),
        flush=True,
    )
    if completed_rounds < MIN_COMPLETED_ROUNDS:
        raise RuntimeError(
            f"DSpark completed only {completed_rounds} verification rounds; "
            f"at least {MIN_COMPLETED_ROUNDS} are required."
        )
    if len(request.output_token_ids) != MAX_NEW_TOKENS:
        raise RuntimeError(
            "The fixed DSpark prompt produced an early stop; choose a prompt "
            "whose first 16 greedy tokens contain no EOS/stop token."
        )
    if speculator._proposal_generated_count not in (
        completed_rounds,
        completed_rounds + 1,
    ):
        raise RuntimeError("DSpark generated more than one terminal extra proposal.")
    if speculator._terminal_proposal_discard_count > 1:
        raise RuntimeError("DSpark terminal-discarded more than one proposal.")

    output = {
        "rank": runtime.launch.rank,
        "request_id": request.request_id,
        "prompt_token_ids": list(request.prompt_token_ids),
        "output_token_ids": list(request.output_token_ids),
        "output_count": len(request.output_token_ids),
        "stop_reason": str(request.stop_reason),
        "completed_rounds": completed_rounds,
        "target_forward_count": target_forward_count,
        "verification_count": verification_count,
        "proposal_generated_count": speculator._proposal_generated_count,
        "proposal_returned_count": speculator._proposal_returned_count,
        "proposal_installed_count": speculator._proposal_installed_count,
        "proposal_consumed_count": speculator._proposal_consumption_count,
        "terminal_discarded_proposal_count": (speculator._terminal_proposal_discard_count),
        "round_indices": [record["round_index"] for record in round_records],
    }
    print(
        "DSPARK_M2_4B_GENERATION=" + json.dumps(output, sort_keys=True),
        flush=True,
    )
    return output


def _target_only_runtime(
    settings: Any,
    launch: Any,
    contexts: ExitStack,
    state: dict[str, Any],
) -> Any:
    import torch
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    import vllm_ascend
    from vllm_ascend.worker.worker import NPUWorker

    engine_args = EngineArgs(
        model=str(settings.target_model),
        tokenizer=str(settings.target_model),
        skip_tokenizer_init=True,
        dtype=settings.dtype,
        max_model_len=settings.max_model_len,
        tensor_parallel_size=settings.tp_size,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        distributed_executor_backend="external_launcher",
        enforce_eager=True,
        block_size=prepare_harness.PREPARE_ONLY_CACHE_BLOCK_SIZE,
        max_num_seqs=1,
    )
    vllm_config = engine_args.create_engine_config()
    contexts.enter_context(dspark_loader_config_context(vllm_config))
    vllm_ascend.register_model()
    worker = NPUWorker(
        vllm_config=vllm_config,
        local_rank=launch.local_rank,
        rank=launch.rank,
        distributed_init_method="env://",
        is_driver_worker=launch.rank == 0,
    )
    state["worker"] = worker
    worker.init_device()
    worker.load_model()
    kv_cache_specs = worker.get_kv_cache_spec()
    kv_cache_config = get_kv_cache_configs(
        vllm_config,
        [kv_cache_specs],
        [_kv_cache_budget(os.environ)],
    )[0]
    worker.initialize_from_config(kv_cache_config)
    runtime = SimpleNamespace(
        torch=torch,
        launch=launch,
        settings=settings,
        vllm_config=vllm_config,
        worker=worker,
        runner=worker.model_runner,
        kv_cache_config=kv_cache_config,
    )
    return runtime


def _cleanup_target_only(contexts: ExitStack, state: dict) -> list[str]:
    errors: list[str] = []

    def cleanup(name: str, function) -> None:
        try:
            function()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    worker = state.get("worker")
    cleanup("worker.shutdown", lambda: worker.shutdown() if worker else None)

    def release_state() -> None:
        state.clear()
        gc.collect()

    cleanup("release target/KV state", release_state)

    def destroy_ascend() -> None:
        from vllm_ascend.distributed.parallel_state import (
            destroy_ascend_model_parallel,
        )

        destroy_ascend_model_parallel()

    def destroy_vllm() -> None:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()

    cleanup("destroy_ascend_model_parallel", destroy_ascend)
    cleanup("destroy_distributed_environment", destroy_vllm)

    def empty_cache() -> None:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_initialized():
            torch.npu.synchronize()
            torch.npu.empty_cache()

    cleanup("torch.npu.empty_cache", empty_cache)
    cleanup("vllm config context", contexts.close)
    return errors


def test_target_only_greedy_16_tokens_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    launch = parse_launch_context(os.environ, settings.tp_size)
    contexts = ExitStack()
    state: dict[str, Any] = {}
    primary_error: BaseException | None = None
    try:
        runtime = _target_only_runtime(
            settings,
            launch,
            contexts,
            state,
        )
        _run_target_only_generation(runtime)
    except BaseException as exc:
        primary_error = exc
        print(
            "DSPARK_M2_4B_TARGET_ONLY_FAILURE="
            + json.dumps(
                {
                    "rank": launch.rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    finally:
        cleanup_errors = _cleanup_target_only(contexts, state)
        print(
            "DSPARK_M2_4B_TARGET_ONLY_CLEANUP="
            + json.dumps(
                {"rank": launch.rank, "errors": cleanup_errors},
                sort_keys=True,
            ),
            flush=True,
        )
        if cleanup_errors and primary_error is None:
            raise RuntimeError("Target-only cleanup failed: " + "; ".join(cleanup_errors))


def test_dspark_optimistic_multi_round_generation_npu() -> None:
    try:
        settings = parse_loader_settings(os.environ)
    except HarnessNotConfigured as exc:
        pytest.skip(str(exc))
    parse_launch_context(os.environ, settings.tp_size)

    def callback(runtime: Any) -> bool:
        _run_dspark_multi_round(runtime)
        return True

    if prepare_harness._INITIALIZED_WORKER_CALLBACK is not None:
        raise RuntimeError("The DSpark initialized-worker callback is already installed.")
    previous_mode = prepare_harness._CONTINUE_AFTER_VERIFICATION
    prepare_harness._CONTINUE_AFTER_VERIFICATION = True
    prepare_harness._INITIALIZED_WORKER_CALLBACK = callback
    try:
        prepare_harness.test_dspark_proposal_inputs_prepare_only_npu()
    finally:
        prepare_harness._INITIALIZED_WORKER_CALLBACK = None
        prepare_harness._CONTINUE_AFTER_VERIFICATION = previous_mode
