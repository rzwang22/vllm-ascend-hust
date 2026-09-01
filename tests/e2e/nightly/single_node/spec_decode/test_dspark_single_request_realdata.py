# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import tests.e2e.nightly.single_node.spec_decode.test_dspark_proposal_inputs_prepare as prepare_harness
from tests.e2e.nightly.single_node.spec_decode.dspark_loader_harness import (
    HarnessNotConfigured,
    parse_launch_context,
    parse_loader_settings,
)
from tests.e2e.nightly.single_node.spec_decode.m2_5a_trace_io import RankTraceWriter, rank_trace_writer
from tests.e2e.nightly.single_node.spec_decode.test_dspark_kv_cache_init import (
    _kv_cache_budget,
)
from tests.e2e.nightly.single_node.spec_decode.test_dspark_multi_round_generation import (
    _build_scheduler,
    _cleanup_target_only,
    _materialize_model_output,
    _target_only_runtime,
)
from tests.e2e.nightly.single_node.spec_decode.test_dspark_single_round_verification import (
    _expected_greedy_verification,
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
MINIMUM_KV_CACHE_BYTES = 2 * 1024 * 1024 * 1024
RUNTIME_CONTRACT = "DSPARK_M2_5A_RUNTIME_CONTRACT"
FIRST_ROUND_TRACE = "DSPARK_M2_5A_FIRST_ROUND_TRACE"
OUTPUT_INDEX_TRACE = "DSPARK_M2_5A_OUTPUT_INDEX_TRACE"
EARLY_RANGE_TRACE = "DSPARK_M2_5A_EARLY_RANGE_TRACE"
TRACE_CONFIG = "DSPARK_M2_5A_TRACE_CONFIG"
PERFORMANCE_CONFIG = "DSPARK_M2_5A_PERFORMANCE_CONFIG"
_CASE_FILTER_ENV = "DSPARK_M25A_CASE_ID"
_FIRST_ROUND_TRACE_ENV = "DSPARK_M25A_TRACE_FIRST_ROUND"
_OUTPUT_INDEX_TRACE_ENV = "DSPARK_M25A_TRACE_OUTPUT_INDEX"
# Test-only, non-sensitive: 0 disables; 1..16 observes every commit starting
# before this output length. Requires one exact case; not a production env var.
_EARLY_TOKENS_TRACE_ENV = "DSPARK_M25A_TRACE_EARLY_TOKENS"
# Test-only, non-sensitive: enables timing and memory observations in this
# repository-native benchmark harness. It is not a production runtime flag.
_PERFORMANCE_ENV = "DSPARK_M25A_PERFORMANCE"
MAX_EARLY_TRACE_TOKENS = 16


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


@dataclass(frozen=True)
class _ForensicTraceConfig:
    case_id: str | None
    first_round: bool
    output_index: int | None
    early_tokens: int = 0


@dataclass(frozen=True)
class _PerformanceConfig:
    enabled: bool


def _performance_config(
    environ: Mapping[str, str],
    trace_config: _ForensicTraceConfig,
    lifecycle_repeat: int,
) -> _PerformanceConfig:
    value = environ.get(_PERFORMANCE_ENV, "0").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{_PERFORMANCE_ENV} must be 0 or 1, got {value!r}.")
    enabled = value == "1"
    if not enabled:
        return _PerformanceConfig(enabled=False)

    if (
        trace_config.case_id is not None
        or trace_config.first_round
        or trace_config.output_index is not None
        or trace_config.early_tokens
    ):
        raise ValueError("M2.5A performance mode cannot be combined with a case filter or forensic trace.")
    launch_blocking = environ.get("ASCEND_LAUNCH_BLOCKING", "0").strip().lower()
    if launch_blocking not in {"", "0", "false", "off"}:
        raise ValueError("M2.5A performance mode requires ASCEND_LAUNCH_BLOCKING=0.")
    if lifecycle_repeat != 1:
        raise ValueError("M2.5A performance repetitions must use independent launches, not lifecycle repeats.")
    return _PerformanceConfig(enabled=True)


def _select_forensic_cases(
    plan: list[dict[str, Any]],
    environ: Mapping[str, str],
) -> tuple[list[dict[str, Any]], _ForensicTraceConfig]:
    case_id = environ.get(_CASE_FILTER_ENV, "").strip() or None
    trace_value = environ.get(_FIRST_ROUND_TRACE_ENV, "0").strip()
    if trace_value not in {"0", "1"}:
        raise ValueError(f"{_FIRST_ROUND_TRACE_ENV} must be 0 or 1, got {trace_value!r}.")
    first_round = trace_value == "1"
    output_index_value = environ.get(_OUTPUT_INDEX_TRACE_ENV, "").strip()
    output_index = None
    if output_index_value:
        try:
            output_index = int(output_index_value)
        except ValueError as exc:
            raise ValueError(
                f"{_OUTPUT_INDEX_TRACE_ENV} must be a non-negative integer, got {output_index_value!r}."
            ) from exc
        if output_index < 0:
            raise ValueError(f"{_OUTPUT_INDEX_TRACE_ENV} must be non-negative, got {output_index}.")
    early_value = environ.get(_EARLY_TOKENS_TRACE_ENV, "0").strip()
    try:
        early_tokens = int(early_value)
    except ValueError as exc:
        raise ValueError(f"{_EARLY_TOKENS_TRACE_ENV} must be an integer in 0..{MAX_EARLY_TRACE_TOKENS}.") from exc
    if not 0 <= early_tokens <= MAX_EARLY_TRACE_TOKENS:
        raise ValueError(f"{_EARLY_TOKENS_TRACE_ENV} must be in 0..{MAX_EARLY_TRACE_TOKENS}.")
    if early_tokens and (first_round or output_index is not None):
        raise ValueError("Early-range trace must not be combined with first-round/output-index trace.")
    if (first_round or output_index is not None or early_tokens) and case_id is None:
        enabled = (
            f"{_FIRST_ROUND_TRACE_ENV}=1"
            if first_round
            else _EARLY_TOKENS_TRACE_ENV
            if early_tokens
            else _OUTPUT_INDEX_TRACE_ENV
        )
        raise ValueError(f"{enabled} requires an exact {_CASE_FILTER_ENV}.")
    if case_id is None:
        return plan, _ForensicTraceConfig(
            case_id=None,
            first_round=False,
            output_index=None,
        )
    selected = [case for case in plan if case["case_id"] == case_id]
    if not selected:
        available = sorted({str(case["case_id"]) for case in plan})
        raise ValueError(f"{_CASE_FILTER_ENV}={case_id!r} is not in the selected profile: {available!r}.")
    return selected, _ForensicTraceConfig(
        case_id=case_id,
        first_round=first_round,
        output_index=output_index,
        early_tokens=early_tokens,
    )


def _commit_output_index_trace(
    output_length_before: int,
    committed_tokens: list[int],
    output_index: int,
) -> dict[str, Any]:
    if output_length_before < 0 or output_index < 0:
        raise ValueError("Output lengths and trace indices must be non-negative.")
    output_length_after = output_length_before + len(committed_tokens)
    covers = output_length_before <= output_index < output_length_after
    offset = output_index - output_length_before if covers else None
    return {
        "commit_start_output_index": output_length_before,
        "commit_end_output_index_exclusive": output_length_after,
        "commit_covers_traced_output_index": covers,
        "traced_commit_offset": offset,
        "traced_committed_token": committed_tokens[offset] if offset is not None else None,
    }


def _host_json_value(
    value: Any,
    *,
    synchronize_tensor: Callable[[], None] | None = None,
) -> Any:
    """Convert known trace metadata to JSON-compatible host values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _host_json_value(
                item,
                synchronize_tensor=synchronize_tensor,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _host_json_value(
                item,
                synchronize_tensor=synchronize_tensor,
            )
            for item in value
        ]

    # NumPy arrays/scalars are already host-resident. Check them before the
    # Tensor-like protocol so this boundary never calls detach/cpu on NumPy.
    if type(value).__module__.partition(".")[0] == "numpy":
        return value.tolist()

    detach = getattr(value, "detach", None)
    if callable(detach):
        if synchronize_tensor is not None:
            synchronize_tensor()
        detached = detach()
        cpu = getattr(detached, "cpu", None)
        if not callable(cpu):
            raise TypeError(f"Trace Tensor-like value has no cpu() method: {type(value)!r}.")
        host_value = cpu()
        tolist = getattr(host_value, "tolist", None)
        if not callable(tolist):
            raise TypeError(f"Trace Tensor-like host value has no tolist() method: {type(host_value)!r}.")
        return tolist()

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Unsupported M2.5A trace metadata type: {type(value)!r}.")


def _target_top2_trace(runtime: Any) -> dict[str, Any]:
    """Recompute diagnostic logits; these are NOT the sampler's saved logits.

    The extra LM head, allocations, top-k and D2H synchronization can change
    timing. Compare independent trace-off runs before interpreting a mismatch.
    """
    runner = runtime.runner
    state = runner.execute_model_state
    if state is None or state.hidden_states is None:
        raise RuntimeError("The traced target forward produced no hidden states.")
    input_batch = state.input_batch
    if runner.sampler is None:
        raise RuntimeError("The traced target forward has no sampler.")

    sample_hidden_states = state.hidden_states[input_batch.logits_indices]
    logits = runner.model.compute_logits(sample_hidden_states)
    processed_logits = runner.sampler.apply_sampling_params(
        logits,
        input_batch.expanded_idx_mapping,
        input_batch.idx_mapping_np,
        input_batch.positions[input_batch.logits_indices],
        input_batch.input_ids[input_batch.logits_indices],
        input_batch.expanded_local_pos,
    )
    top2_values, top2_ids = runtime.torch.topk(
        processed_logits.to(runtime.torch.float32),
        k=2,
        dim=-1,
    )
    query_start_loc = input_batch.query_start_loc[: input_batch.num_reqs + 1]
    sampled_positions = input_batch.positions[input_batch.logits_indices]
    sampled_input_ids = input_batch.input_ids[input_batch.logits_indices]
    runtime.torch.npu.synchronize()

    top2_values_host = _host_json_value(top2_values)
    sampling_states = runner.sampler.sampling_states
    trace = {
        "logits_observation": "recomputed_from_target_hidden_states_before_sampling",
        "sampling_state": {
            name: _host_json_value(getattr(sampling_states, name).np[input_batch.idx_mapping_np])
            for name in ("temperature", "seeds", "top_k", "top_p", "min_p")
        },
        "model_seed": runtime.vllm_config.model_config.seed,
        "execution_path": {
            "model_class": f"{type(runner.model).__module__}.{type(runner.model).__qualname__}",
            "model": runtime.vllm_config.model_config.model,
            "revision": runtime.vllm_config.model_config.revision,
            "quantization": runtime.vllm_config.model_config.quantization,
            "runner_class": f"{type(runner).__module__}.{type(runner).__qualname__}",
            "dtype": str(runtime.vllm_config.model_config.dtype),
            "block_size": runtime.vllm_config.cache_config.block_size,
            "attention_backend": str(runtime.vllm_config.attention_config.backend),
            "enforce_eager": runtime.vllm_config.model_config.enforce_eager,
            "tp_size": runtime.vllm_config.parallel_config.tensor_parallel_size,
            "kernel_tiling_observed": False,
        },
        "torch_deterministic_algorithms_enabled": runtime.torch.are_deterministic_algorithms_enabled(),
        "target_logits_shape": list(processed_logits.shape),
        "query_start_loc": _host_json_value(query_start_loc),
        "query_lengths": _host_json_value(query_start_loc[1:] - query_start_loc[:-1]),
        "num_scheduled_tokens": _host_json_value(input_batch.num_scheduled_tokens[: input_batch.num_reqs]),
        "num_computed_tokens_before": _host_json_value(input_batch.num_computed_tokens_np[: input_batch.num_reqs]),
        "num_draft_tokens": int(input_batch.num_draft_tokens),
        "logits_positions": _host_json_value(sampled_positions),
        "logits_input_ids": _host_json_value(sampled_input_ids),
        "target_top1_token_ids": _host_json_value(top2_ids[:, 0]),
        "target_top2_token_ids": _host_json_value(top2_ids[:, 1]),
        "target_top1_logits": [row[0] for row in top2_values_host],
        "target_top2_logits": [row[1] for row in top2_values_host],
        "target_top1_top2_margins": [row[0] - row[1] for row in top2_values_host],
    }
    del logits, processed_logits, sample_hidden_states, top2_ids, top2_values
    return trace


def _trace_host_value(runtime: Any, value: Any) -> Any:
    return _host_json_value(
        value,
        synchronize_tensor=runtime.torch.npu.synchronize,
    )


def _next_model_input_trace(
    runtime: Any,
    request: Any,
    request_id: str,
    output_index: int,
    traced_token: int,
    *,
    commit_end: int | None = None,
) -> dict[str, Any]:
    """Observe the next real model input and the runner's logical prefix."""
    runner = runtime.runner
    state = runner.execute_model_state
    if state is None:
        raise RuntimeError("The step after the traced commit produced no model state.")
    input_batch = state.input_batch
    if request_id not in runner.req_states.req_id_to_index:
        raise RuntimeError("The traced request disappeared before its next model input.")
    req_state_index = runner.req_states.req_id_to_index[request_id]
    prompt_length = len(request.prompt_token_ids)
    absolute_token_index = prompt_length + output_index
    window_start = max(prompt_length, absolute_token_index - 4)
    window_end = min(
        prompt_length + len(request.output_token_ids),
        max(absolute_token_index + 5, prompt_length + (commit_end or 0)),
    )
    request_tokens = [*request.prompt_token_ids, *request.output_token_ids]
    runner_tokens = _trace_host_value(
        runtime,
        runner.req_states.all_token_ids.gpu[
            req_state_index,
            window_start:window_end,
        ],
    )
    request_window = request_tokens[window_start:window_end]
    query_start_loc = _trace_host_value(
        runtime,
        input_batch.query_start_loc[: input_batch.num_reqs + 1],
    )
    next_input_ids = _trace_host_value(
        runtime,
        input_batch.input_ids[: input_batch.num_tokens],
    )
    next_positions = _trace_host_value(
        runtime,
        input_batch.positions[: input_batch.num_tokens],
    )
    runner_num_computed = int(runner.req_states.num_computed_tokens_np[req_state_index])
    runner_prefix = _trace_host_value(
        runtime, runner.req_states.all_token_ids.gpu[req_state_index, : len(request_tokens)]
    )
    return {
        "next_model_input_available": True,
        "next_prefix_token_sha256": token_ids_sha256(request_tokens),
        "next_model_query_start_loc": query_start_loc,
        "next_model_query_lengths": [end - start for start, end in zip(query_start_loc[:-1], query_start_loc[1:])],
        "next_model_input_ids": next_input_ids,
        "next_model_positions": next_positions,
        "next_model_num_draft_tokens": int(input_batch.num_draft_tokens),
        "next_input_num_computed_tokens": _host_json_value(input_batch.num_computed_tokens_np[: input_batch.num_reqs]),
        "next_runner_num_computed_tokens": runner_num_computed,
        "next_runner_prefix_sha256": token_ids_sha256(runner_prefix),
        "next_request_logical_length": len(request_tokens),
        "next_runner_prefix_matches_request": runner_prefix == request_tokens,
        "next_model_input_contains_traced_token": traced_token in next_input_ids,
        "next_request_output_window": request_window,
        "next_runner_output_window": runner_tokens,
        "next_runner_window_matches_request": runner_tokens == request_window,
        "next_output_window_start": window_start - prompt_length,
        "next_output_window_end": window_end - prompt_length,
        "next_runner_traced_token": (
            runner_tokens[absolute_token_index - window_start]
            if window_start <= absolute_token_index < window_end
            else None
        ),
        "next_runner_contains_traced_token_at_output_index": (
            window_start <= absolute_token_index < window_end
            and runner_tokens[absolute_token_index - window_start] == traced_token
        ),
    }


def _committed_prefix_trace(request: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Match each diagnostic logit row to the *actual* committed history.

    Hashes are observations, not a replay or a substitute for physical KV
    contents. A rejected candidate prefix must never be labelled comparable.
    """
    start = record["output_length_before"]
    committed = record["scheduler_committed_tokens"]
    base = [*request.prompt_token_ids, *request.output_token_ids[:start]]
    candidates = record["consumed_candidate_tokens"] or []
    row_prefixes = [token_ids_sha256([*base, *candidates[:i]]) for i in range(len(committed))]
    actual_prefixes = [token_ids_sha256([*base, *committed[:i]]) for i in range(len(committed))]
    return {
        "commit_start_output_index": start,
        "commit_end_output_index_exclusive": start + len(committed),
        "logit_row_prefix_sha256": row_prefixes,
        "committed_prefix_sha256": actual_prefixes,
        "logit_row_matches_committed_prefix": [left == right for left, right in zip(row_prefixes, actual_prefixes)],
        "prompt_token_sha256": token_ids_sha256(request.prompt_token_ids),
        "prompt_token_count": len(request.prompt_token_ids),
        "logit_row_output_prefix_sha256": [
            token_ids_sha256([*request.output_token_ids[:start], *candidates[:i]]) for i in range(len(committed))
        ],
        "logit_row_request_logical_lengths": [len(base) + i for i in range(len(committed))],
        "logit_row_request_input_token_ids": [base[-1], *committed[:-1]] if committed else [],
    }


def _proposal_lifecycle_trace(lifecycle: Any) -> dict[str, Any] | None:
    if lifecycle is None:
        return None
    return {
        name: _host_json_value(getattr(lifecycle, name))
        for name in (
            "proposal_epoch",
            "owner_epoch",
            "consumer_epoch",
            "request_ids",
            "generated",
            "returned_to_core",
            "installed",
            "consumed",
            "discarded_terminal",
            "scheduled_lengths",
            "disposition",
            "token_prefix_match",
            "truncated",
            "dropped",
            "drop_reason",
        )
    }


def _m2_5a_kv_cache_budget(environ: Mapping[str, str]) -> int:
    if "DSPARK_KV_CACHE_BYTES" not in environ:
        raise ValueError("M2.5A requires DSPARK_KV_CACHE_BYTES to be explicitly set before model loading.")
    kv_cache_bytes = _kv_cache_budget(environ)
    if kv_cache_bytes < MINIMUM_KV_CACHE_BYTES:
        raise ValueError(
            "M2.5A KV budget preflight failed before model loading: "
            f"got {kv_cache_bytes} bytes, require at least {MINIMUM_KV_CACHE_BYTES} bytes "
            f"for max_model_len={EXPECTED_MAX_MODEL_LEN}."
        )
    return kv_cache_bytes


def _validate_runtime_contract(
    runtime: Any,
    mode: str,
    profile: str,
    kv_cache_bytes: int,
) -> dict[str, Any]:
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
    if config.model_config.max_model_len != EXPECTED_MAX_MODEL_LEN:
        raise RuntimeError(f"M2.5A requires max_model_len={EXPECTED_MAX_MODEL_LEN}.")
    if cache.enable_prefix_caching is not False:
        raise RuntimeError(f"M2.5A requires prefix caching disabled, got {cache.enable_prefix_caching!r}.")
    if cache.block_size != EXPECTED_BLOCK_SIZE:
        raise RuntimeError(f"M2.5A requires block_size={EXPECTED_BLOCK_SIZE}, got {cache.block_size}.")
    speculative = config.speculative_config
    if mode == "target_only" and speculative is not None:
        raise RuntimeError("The target-only M2.5A process must not construct a speculator.")
    if mode == "dspark" and (
        speculative is None or speculative.method != "dspark" or speculative.num_speculative_tokens != EXPECTED_K
    ):
        raise RuntimeError("The DSpark M2.5A process must use method=dspark and K=5.")
    contract = {
        "rank": runtime.launch.rank,
        "mode": mode,
        "profile": profile,
        "max_model_len": config.model_config.max_model_len,
        "kv_cache_bytes": kv_cache_bytes,
        "block_size": cache.block_size,
        "prefix_caching_enabled": cache.enable_prefix_caching,
        "enforce_eager": config.model_config.enforce_eager,
        "tp_size": parallel.tensor_parallel_size,
        "pp_size": parallel.pipeline_parallel_size,
        "expert_parallel": parallel.enable_expert_parallel,
    }
    _marker(RUNTIME_CONTRACT, contract)
    return contract


def _performance_memory_snapshot(runtime: Any) -> dict[str, int]:
    device = runtime.worker.device
    npu = runtime.torch.npu
    return {
        "allocated": int(npu.memory_allocated(device)),
        "reserved": int(npu.memory_reserved(device)),
        "peak_allocated": int(npu.max_memory_allocated(device)),
        "peak_reserved": int(npu.max_memory_reserved(device)),
    }


def _verification_token_telemetry(
    scheduled_candidate_count: int,
    raw_tokens: list[int],
) -> tuple[int, int, int]:
    """Decode greedy rejection output without inspecting placeholder IDs.

    The synchronous core scheduler carries ``-1`` proposal placeholders: it is
    authoritative for the installed proposal length, not candidate identity.
    Greedy rejection output always contains the accepted candidate prefix plus
    exactly one replacement or bonus token, so its valid length is sufficient
    for low-overhead performance telemetry.
    """
    if scheduled_candidate_count <= 0:
        raise ValueError("Verification telemetry requires a positive scheduled proposal length.")
    if not raw_tokens or len(raw_tokens) > scheduled_candidate_count + 1:
        raise ValueError(
            "Greedy verification output must contain an accepted prefix plus one replacement or bonus token."
        )
    accepted_candidate_count = len(raw_tokens) - 1
    all_candidates_accepted = accepted_candidate_count == scheduled_candidate_count
    return (
        accepted_candidate_count,
        int(not all_candidates_accepted),
        int(all_candidates_accepted),
    )


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


@dataclass
class _FinishedRequestLifecycle:
    """Track the one-shot scheduler event through worker cleanup."""

    request_id: str
    observed_count: int = 0
    worker_delivery_count: int = 0

    def observe(self, scheduler_output: Any) -> bool:
        finished_req_ids = scheduler_output.finished_req_ids
        unexpected = finished_req_ids.difference({self.request_id})
        if unexpected:
            raise RuntimeError(f"Per-request cleanup observed stale finished request events: {sorted(unexpected)!r}.")
        if self.request_id not in finished_req_ids:
            return False
        if self.observed_count:
            raise RuntimeError(f"Finished request {self.request_id!r} was published more than once.")
        self.observed_count += 1
        return True

    def record_worker_delivery(self, included_finished_event: bool) -> None:
        if included_finished_event:
            self.worker_delivery_count += 1

    def assert_delivered_once(self) -> None:
        if self.observed_count != 1 or self.worker_delivery_count != 1:
            raise RuntimeError(
                f"Finished request {self.request_id!r} must be published and "
                "delivered to the worker exactly once; "
                f"observed={self.observed_count}, "
                f"delivered={self.worker_delivery_count}."
            )


def _execute_scheduler_output(
    runtime: Any,
    scheduler_output: Any,
    finished_lifecycle: _FinishedRequestLifecycle,
) -> Any:
    included_finished_event = finished_lifecycle.observe(scheduler_output)
    output = runtime.worker.execute_model(scheduler_output)
    finished_lifecycle.record_worker_delivery(included_finished_event)
    return output


def _assert_canonical_zero_token_runner_output(output: Any) -> None:
    """Accept only the current core's semantically empty runner envelope."""
    if output is None:
        return

    empty_sequence_fields = ("req_ids", "sampled_token_ids")
    empty_mapping_fields = ("req_id_to_index", "prompt_logprobs_dict")
    for field_name in empty_sequence_fields:
        value = getattr(output, field_name, None)
        if not isinstance(value, (list, tuple)) or value:
            raise RuntimeError(
                f"A zero-token scheduler output returned request/token payload in {field_name}: {value!r}."
            )
    for field_name in empty_mapping_fields:
        value = getattr(output, field_name, None)
        if not isinstance(value, dict) or value:
            raise RuntimeError(f"A zero-token scheduler output returned request payload in {field_name}: {value!r}.")
    for field_name in ("logprobs", "routed_experts", "cudagraph_stats"):
        value = getattr(output, field_name, None)
        if value is not None:
            raise RuntimeError(f"A zero-token scheduler output returned model-forward payload in {field_name}.")
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None and (not isinstance(pooler_output, (list, tuple)) or pooler_output):
        raise RuntimeError("A zero-token scheduler output returned pooler payload.")
    for field_name in ("kv_cache_compression_plans", "num_nans_in_logits"):
        value = getattr(output, field_name, None)
        if value not in (None, [], {}):
            raise RuntimeError(f"A zero-token scheduler output returned execution payload in {field_name}: {value!r}.")
    if getattr(output, "spec_decode_num_forwards", 0) != 0:
        raise RuntimeError("A zero-token scheduler output reported speculative model forwards.")
    for field_name in (
        "spec_decode_proposer_latency_seconds",
        "spec_decode_verification_latency_seconds",
    ):
        if getattr(output, field_name, 0.0) != 0.0:
            raise RuntimeError(f"A zero-token scheduler output reported speculative execution latency in {field_name}.")


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
    queues = {
        "scheduler running": getattr(scheduler, "running", ()),
        "scheduler waiting": getattr(scheduler, "waiting", ()),
        "scheduler skipped waiting": getattr(scheduler, "skipped_waiting", ()),
    }
    stale_queues = sorted(
        name
        for name, queue in queues.items()
        if any(getattr(request, "request_id", None) == request_id for request in queue)
    )
    if stale_queues:
        raise RuntimeError(f"Finished request {request_id!r} remains in: {stale_queues}.")
    if request_id in getattr(scheduler, "finished_req_ids", set()):
        raise RuntimeError(f"Finished request {request_id!r} remains pending in the scheduler event set.")

    kv_cache_manager = scheduler.kv_cache_manager
    block_ids = kv_cache_manager.get_block_ids(request_id)
    owned_block_ids = [block_id for group in block_ids for block_id in group]
    if owned_block_ids:
        raise RuntimeError(f"Finished request {request_id!r} retains KV block ownership: {owned_block_ids!r}.")
    coordinator = kv_cache_manager.coordinator
    stale_kv_managers = [
        index for index, manager in enumerate(coordinator.single_type_managers) if request_id in manager.req_to_blocks
    ]
    if stale_kv_managers:
        raise RuntimeError(f"Finished request {request_id!r} remains in KV manager groups: {stale_kv_managers!r}.")
    for state_name in (
        "_compressed_request_physical_tokens",
        "_compression_destination_reservations",
    ):
        state = getattr(kv_cache_manager, state_name, {})
        if request_id in state:
            raise RuntimeError(f"Finished request {request_id!r} remains in KV state {state_name}.")
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


def _assert_scheduler_proposal_disposition(
    speculator: Any,
    scheduler_output: Any,
) -> None:
    lifecycle = speculator._current_proposal_lifecycle
    if lifecycle is None:
        raise RuntimeError("A scheduler-installed DSpark proposal has no active lifecycle.")
    scheduled_lengths = tuple(
        len(scheduler_output.scheduled_spec_decode_tokens[request_id]) for request_id in lifecycle.request_ids
    )
    published_length = speculator._published_candidate_tokens.shape[1]
    expected_disposition = (
        "INSTALLED" if all(length == published_length for length in scheduled_lengths) else "TRUNCATED"
    )
    if (
        lifecycle.disposition != expected_disposition
        or lifecycle.scheduled_lengths != scheduled_lengths
        or not lifecycle.installed
        or lifecycle.consumed
    ):
        raise RuntimeError(
            "DSpark proposal disposition was not reconciled from the current "
            "SchedulerOutput before target verification."
        )


def _flush_finished_request(
    runtime: Any,
    scheduler: Any,
    finished_lifecycle: _FinishedRequestLifecycle,
) -> None:
    request_id = finished_lifecycle.request_id
    finished_lifecycle.assert_delivered_once()
    cleanup_output = scheduler.schedule()
    if cleanup_output.total_num_scheduled_tokens != 0:
        raise RuntimeError("Per-request cleanup unexpectedly scheduled another model forward.")
    if cleanup_output.finished_req_ids:
        raise RuntimeError(
            "The schedule after one-shot request cleanup must not repeat or "
            f"retain finished request events, got {cleanup_output.finished_req_ids!r}."
        )
    runner_output = _execute_scheduler_output(runtime, cleanup_output, finished_lifecycle)
    _assert_canonical_zero_token_runner_output(runner_output)
    finished_lifecycle.assert_delivered_once()
    _assert_released_request_state(runtime, scheduler, request_id)


def _run_case(
    runtime: Any,
    scheduler: Any,
    case: dict[str, Any],
    *,
    mode: str,
    profile: str,
    manifest_sha256: str,
    trace_first_round: bool = False,
    trace_output_index: int | None = None,
    trace_early_tokens: int = 0,
    early_trace_writer: RankTraceWriter | None = None,
    performance: bool = False,
) -> dict[str, Any]:
    if trace_early_tokens and early_trace_writer is None:
        raise ValueError("Early-range trace must have an exclusive rank-local writer.")
    request_id = f"m25a-{mode}-{case['request_sequence_index']}-{case['lifecycle_repeat']}-{case['case_id']}"
    request = _request(case, request_id)
    speculator = runtime.speculator if mode == "dspark" else None
    before = _counter_snapshot(speculator)
    target_forward_count = 0
    traced_step_count = 0
    output_index_trace_emitted = False
    pending_output_index_trace: dict[str, Any] | None = None
    pending_early_trace: dict[str, Any] | None = None
    verification_count = 0
    completed_rounds = 0
    accepted_candidate_tokens_total = 0
    replacement_tokens_total = 0
    bonus_tokens_total = 0
    verification_committed_tokens_total = 0
    scheduled_draft_token_count = 0
    scheduler_seconds = 0.0
    scheduler_update_seconds = 0.0
    model_execute_host_seconds = 0.0
    sample_materialize_seconds = 0.0
    draft_install_seconds = 0.0
    proposer_latency_seconds = 0.0
    verification_latency_seconds = 0.0
    first_commit_seconds: float | None = None
    first_commit_token_count = 0
    performance_started_at: float | None = None
    memory_before: dict[str, int] | None = None
    memory_after: dict[str, int] | None = None
    terminal_partial_commit = False
    consumer_epochs: list[int] = []
    finished_lifecycle = _FinishedRequestLifecycle(request_id)
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
    if performance:
        runtime.torch.npu.synchronize()
        runtime.torch.npu.reset_peak_memory_stats(runtime.worker.device)
        memory_before = _performance_memory_snapshot(runtime)
        performance_started_at = time.perf_counter()
    scheduler.add_request(request)
    while scheduler.has_requests():
        schedule_started_at = time.perf_counter() if performance else None
        scheduler_output = scheduler.schedule()
        if schedule_started_at is not None:
            scheduler_seconds += time.perf_counter() - schedule_started_at
        if scheduler_output.total_num_scheduled_tokens == 0:
            execute_started_at = time.perf_counter() if performance else None
            runner_output = _execute_scheduler_output(runtime, scheduler_output, finished_lifecycle)
            if execute_started_at is not None:
                model_execute_host_seconds += time.perf_counter() - execute_started_at
            _assert_canonical_zero_token_runner_output(runner_output)
            continue
        is_verification = bool(scheduler_output.scheduled_spec_decode_tokens)
        scheduled_length = len(scheduler_output.scheduled_spec_decode_tokens[request_id]) if is_verification else 0
        proposal_pending_before_execute = bool(mode == "dspark" and speculator._published_candidate_tokens is not None)
        output_length_before = len(request.output_token_ids)
        trace_first_round_step = trace_first_round and traced_step_count < 2
        maximum_commit_length = scheduled_length + 1 if is_verification else 1
        trace_output_index_step = bool(
            trace_output_index is not None
            and not output_index_trace_emitted
            and pending_output_index_trace is None
            and output_length_before <= trace_output_index < output_length_before + maximum_commit_length
        )
        trace_early_step = output_length_before < trace_early_tokens
        trace_step = trace_first_round_step or trace_output_index_step or trace_early_step
        trace_record: dict[str, Any] | None = None
        if request.is_finished():
            raise RuntimeError("A finished request reached another target forward.")
        execute_started_at = time.perf_counter() if performance else None
        execute_output = _execute_scheduler_output(runtime, scheduler_output, finished_lifecycle)
        if execute_started_at is not None:
            model_execute_host_seconds += time.perf_counter() - execute_started_at
        if execute_output is not None:
            raise RuntimeError("PP=1 execution must defer model output to sampling.")
        if mode == "dspark" and is_verification:
            _assert_scheduler_proposal_disposition(
                speculator,
                scheduler_output,
            )
        elif mode == "dspark" and proposal_pending_before_execute:
            dropped = speculator._dropped_proposal_lifecycle
            if (
                speculator._current_proposal_lifecycle is not None
                or speculator._published_candidate_tokens is not None
                or dropped is None
                or dropped.disposition != "DROPPED"
                or dropped.request_ids != (request_id,)
            ):
                raise RuntimeError(
                    "An uninstalled DSpark proposal was not atomically retired before the next target step."
                )
        if pending_output_index_trace is not None:
            traced_token = pending_output_index_trace["traced_committed_token"]
            if not isinstance(traced_token, int) or trace_output_index is None:
                raise RuntimeError("The pending output-index trace has no committed token identity.")
            pending_output_index_trace.update(
                _next_model_input_trace(
                    runtime,
                    request,
                    request_id,
                    trace_output_index,
                    traced_token,
                )
            )
            _marker(OUTPUT_INDEX_TRACE, pending_output_index_trace)
            pending_output_index_trace = None
            output_index_trace_emitted = True
        if pending_early_trace is not None:
            pending_early_trace.update(
                _next_model_input_trace(
                    runtime,
                    request,
                    request_id,
                    pending_early_trace["commit_start_output_index"],
                    pending_early_trace["scheduler_committed_tokens"][0],
                    commit_end=pending_early_trace["commit_end_output_index_exclusive"],
                )
            )
            pending_early_trace["next_scheduler_kv_block_ids"] = _host_json_value(
                scheduler.kv_cache_manager.get_block_ids(request_id)
            )
            early_trace_writer.write_step(pending_early_trace)
            pending_early_trace = None
        if trace_step:
            lifecycle = speculator._current_proposal_lifecycle if speculator is not None else None
            published_candidates = (
                _trace_host_value(runtime, speculator._published_candidate_tokens)[0] if is_verification else None
            )
            scheduled_candidates = (
                list(scheduler_output.scheduled_spec_decode_tokens[request_id]) if is_verification else None
            )
            consumed_candidates = None
            if is_verification:
                input_batch = runtime.runner.execute_model_state.input_batch
                query_start_loc = _trace_host_value(
                    runtime,
                    input_batch.query_start_loc[: input_batch.num_reqs + 1],
                )
                input_ids = _trace_host_value(
                    runtime,
                    input_batch.input_ids[: input_batch.num_tokens],
                )
                consumed_candidates = input_ids[query_start_loc[0] + 1 : query_start_loc[0] + 1 + scheduled_length]
            trace_record = {
                "rank": runtime.launch.rank,
                "mode": mode,
                "case_id": case["case_id"],
                "request_id": request_id,
                "lifecycle_repeat": case["lifecycle_repeat"],
                "request_sequence_index": case["request_sequence_index"],
                "target_step_index": target_forward_count,
                "step_kind": (
                    "verification" if is_verification else "producer" if output_length_before == 0 else "target_decode"
                ),
                "prefix_token_sha256": token_ids_sha256([*request.prompt_token_ids, *request.output_token_ids]),
                "output_length_before": output_length_before,
                "request_num_computed_tokens_before": request.num_computed_tokens,
                "scheduled_proposal_length": scheduled_length,
                "scheduled_candidate_tokens": scheduled_candidates,
                "proposal_epoch": (lifecycle.proposal_epoch if lifecycle is not None else None),
                "proposal_disposition": (lifecycle.disposition if lifecycle is not None else None),
                "published_candidate_tokens": published_candidates,
                "consumed_candidate_tokens": consumed_candidates,
                "published_candidates_match_consumed": (
                    published_candidates[:scheduled_length] == consumed_candidates if is_verification else None
                ),
                **_target_top2_trace(runtime),
            }
            if trace_early_step:
                trace_record["manifest_sha256"] = manifest_sha256
                trace_record["counters_before_sample"] = _counter_snapshot(speculator)
                trace_record["lifecycle_before_sample"] = _proposal_lifecycle_trace(lifecycle)
                if is_verification:
                    trace_record["published_owner_request_ids"] = list(speculator._published_proposal_request_ids)
                    trace_record["published_owner_state_indices"] = _trace_host_value(
                        runtime, speculator._published_proposal_request_state_indices
                    )
                    trace_record["verification_request_ids"] = list(input_batch.req_ids)
                    trace_record["verification_state_indices"] = _trace_host_value(
                        runtime, input_batch.idx_mapping[: input_batch.num_reqs]
                    )
        target_forward_count += 1
        sample_started_at = time.perf_counter() if performance else None
        async_output = runtime.worker.sample_tokens(None)
        if async_output is None:
            raise RuntimeError("Single-request generation returned no sampled output.")
        if not performance:
            runtime.torch.npu.synchronize()
        sampler_output = getattr(async_output, "sampler_output", None)
        traced_num_sampled = (
            _trace_host_value(runtime, getattr(sampler_output, "num_sampled", None))
            if trace_record is not None
            else None
        )
        traced_num_rejected = (
            _trace_host_value(runtime, getattr(sampler_output, "num_rejected", None))
            if trace_record is not None
            else None
        )
        model_output = _materialize_model_output(async_output)
        if sample_started_at is not None:
            sample_materialize_seconds += time.perf_counter() - sample_started_at
        proposer_latency_seconds += float(getattr(model_output, "spec_decode_proposer_latency_seconds", 0.0))
        verification_latency_seconds += float(getattr(model_output, "spec_decode_verification_latency_seconds", 0.0))
        raw_tokens = list(model_output.sampled_token_ids[0])
        if is_verification:
            verification_count += 1
            completed_rounds += 1
            scheduled_draft_token_count += scheduled_length
            accepted_count, replacement_count, bonus_count = _verification_token_telemetry(
                scheduled_length,
                raw_tokens,
            )
            accepted_candidate_tokens_total += accepted_count
            replacement_tokens_total += replacement_count
            bonus_tokens_total += bonus_count
            consumer_epoch = speculator._proposal_consumer_step_epoch
            if consumer_epoch is None:
                raise RuntimeError("DSpark verification did not publish a consumer epoch.")
            consumer_epochs.append(consumer_epoch)
        update_started_at = time.perf_counter() if performance else None
        scheduler.update_from_output(scheduler_output, model_output)
        if update_started_at is not None:
            scheduler_update_seconds += time.perf_counter() - update_started_at
        if mode == "dspark":
            draft_install_started_at = time.perf_counter() if performance else None
            next_draft_ids = runtime.worker.take_draft_token_ids()
            if next_draft_ids is None:
                raise RuntimeError("DSpark did not return its canonical draft-token result.")
            scheduler.update_draft_token_ids(next_draft_ids)
            if draft_install_started_at is not None:
                draft_install_seconds += time.perf_counter() - draft_install_started_at
        committed_tokens = list(request.output_token_ids)[output_length_before:]
        if is_verification:
            verification_committed_tokens_total += len(committed_tokens)
        if performance and first_commit_seconds is None and committed_tokens:
            assert performance_started_at is not None
            first_commit_seconds = time.perf_counter() - performance_started_at
            first_commit_token_count = len(committed_tokens)
        if trace_record is not None and (raw_tokens or is_verification):
            expected_greedy_tokens = None
            accepted_prefix_length = None
            replacement_used = None
            bonus_used = None
            if is_verification:
                scheduled_candidates = trace_record["published_candidate_tokens"][
                    : trace_record["scheduled_proposal_length"]
                ]
                (
                    expected_greedy_tokens,
                    accepted_prefix_length,
                    replacement_used,
                    bonus_used,
                ) = _expected_greedy_verification(
                    scheduled_candidates,
                    trace_record["target_top1_token_ids"],
                )
            replacement_token = expected_greedy_tokens[-1] if replacement_used and expected_greedy_tokens else None
            bonus_token = expected_greedy_tokens[-1] if bonus_used and expected_greedy_tokens else None
            commit_trace = (
                _commit_output_index_trace(
                    output_length_before,
                    committed_tokens,
                    trace_output_index,
                )
                if trace_output_index is not None
                else {}
            )
            trace_record.update(
                {
                    "consumer_epoch": (speculator._proposal_consumer_step_epoch if is_verification else None),
                    "num_sampled": traced_num_sampled,
                    "num_rejected": traced_num_rejected,
                    "raw_sampled_tokens": raw_tokens,
                    "scheduler_committed_tokens": committed_tokens,
                    "artifact_appended_tokens": list(request.output_token_ids)[output_length_before:],
                    "output_length_after": len(request.output_token_ids),
                    "request_output_token_count": len(request.output_token_ids),
                    "request_num_computed_tokens_after": request.num_computed_tokens,
                    "expected_greedy_tokens": expected_greedy_tokens,
                    "accepted_prefix_length": accepted_prefix_length,
                    "replacement_used": replacement_used,
                    "replacement_token": replacement_token,
                    "bonus_used": bonus_used,
                    "bonus_token": bonus_token,
                    "all_candidates_accepted": (
                        accepted_prefix_length == trace_record["scheduled_proposal_length"] if is_verification else None
                    ),
                    "bonus_contract_valid": (
                        not bonus_used or accepted_prefix_length == trace_record["scheduled_proposal_length"]
                        if is_verification
                        else None
                    ),
                    "raw_matches_expected_greedy": (
                        raw_tokens == expected_greedy_tokens if expected_greedy_tokens is not None else None
                    ),
                    "raw_matches_target_top1": (
                        raw_tokens == trace_record["target_top1_token_ids"][: len(raw_tokens)]
                        if not is_verification
                        else None
                    ),
                    "raw_matches_scheduler_commit": raw_tokens == committed_tokens,
                    "artifact_append_matches_scheduler_commit": (
                        list(request.output_token_ids)[output_length_before:] == committed_tokens
                    ),
                    **commit_trace,
                }
            )
            if trace_early_step:
                trace_record["counters_after_sample"] = _counter_snapshot(speculator)
                trace_record["consumed_lifecycle"] = _proposal_lifecycle_trace(
                    speculator._last_consumed_proposal_lifecycle if is_verification else None
                )
            if trace_first_round_step:
                _marker(FIRST_ROUND_TRACE, trace_record)
                traced_step_count += 1
            if trace_early_step and committed_tokens:
                trace_record.update(_committed_prefix_trace(request, trace_record))
                trace_record["early_output_token_limit"] = trace_early_tokens
                if request.is_finished():
                    trace_record.update(
                        next_model_input_available=False,
                        next_model_input_terminal_reason="request_finished",
                    )
                    early_trace_writer.write_step(trace_record)
                else:
                    pending_early_trace = dict(trace_record)
            if trace_output_index_step and trace_record["commit_covers_traced_output_index"]:
                if request.is_finished():
                    trace_record.update(
                        {
                            "next_model_input_available": False,
                            "next_model_input_terminal_reason": "request_finished",
                        }
                    )
                    _marker(OUTPUT_INDEX_TRACE, trace_record)
                    output_index_trace_emitted = True
                else:
                    pending_output_index_trace = dict(trace_record)
        if request.is_finished():
            if committed_tokens != raw_tokens[: len(committed_tokens)]:
                raise RuntimeError("Terminal scheduler commit is outside the raw verified prefix.")
            terminal_partial_commit |= len(committed_tokens) < len(raw_tokens)
        elif committed_tokens != raw_tokens:
            raise RuntimeError("Active scheduler commit differs from raw sampled/verified tokens.")

    if performance:
        assert performance_started_at is not None
        runtime.torch.npu.synchronize()
        inference_seconds = time.perf_counter() - performance_started_at
        memory_after = _performance_memory_snapshot(runtime)
    else:
        inference_seconds = 0.0
    if trace_output_index is not None and not output_index_trace_emitted:
        raise RuntimeError(f"Output-index trace {trace_output_index} was not reached by the selected request.")
    _flush_finished_request(runtime, scheduler, finished_lifecycle)
    after = _counter_snapshot(speculator)
    if case["ignore_eos"] and len(request.output_token_ids) != case["output_cap"]:
        raise RuntimeError("Synthetic ignore_eos request stopped before its fixed output cap.")
    output_ids = list(request.output_token_ids)
    finish_reason = request.get_finished_reason()
    performance_metrics: dict[str, Any] | None = None
    if performance:
        if first_commit_seconds is None or memory_before is None or memory_after is None:
            raise RuntimeError("Performance mode completed without a timed first commit or NPU memory snapshot.")
        decode_seconds = max(0.0, inference_seconds - first_commit_seconds)
        decode_token_count = max(0, len(output_ids) - first_commit_token_count)
        performance_metrics = {
            "timing_boundary": (
                "after request construction/before scheduler admission through final "
                "scheduler commit and NPU synchronization; excludes model loading and cleanup"
            ),
            "prefill_definition": "time_to_first_scheduler_commit",
            "prefill_latency_seconds": first_commit_seconds,
            "prefill_output_token_count": first_commit_token_count,
            "decode_latency_seconds": decode_seconds,
            "decode_output_token_count": decode_token_count,
            "inference_latency_seconds": inference_seconds,
            "milliseconds_per_output_token": 1000.0 * inference_seconds / len(output_ids),
            "output_tokens_per_second": len(output_ids) / inference_seconds,
            "decode_milliseconds_per_output_token": (
                1000.0 * decode_seconds / decode_token_count if decode_token_count else 0.0
            ),
            "decode_output_tokens_per_second": (
                decode_token_count / decode_seconds if decode_token_count and decode_seconds else 0.0
            ),
            "scheduler_seconds": scheduler_seconds,
            "scheduler_update_seconds": scheduler_update_seconds,
            "model_execute_host_seconds": model_execute_host_seconds,
            "sample_materialize_seconds": sample_materialize_seconds,
            "draft_install_seconds": draft_install_seconds,
            "spec_decode_proposer_latency_seconds": proposer_latency_seconds,
            "spec_decode_verification_latency_seconds": verification_latency_seconds,
            "scheduled_draft_token_count": scheduled_draft_token_count,
            "accepted_candidate_metrics_source": "greedy_rejection_output_length_minus_one",
            "accepted_candidate_tokens_total": accepted_candidate_tokens_total,
            "average_accepted_candidate_tokens_per_verification": (
                accepted_candidate_tokens_total / verification_count if verification_count else None
            ),
            "replacement_tokens_total": replacement_tokens_total,
            "bonus_tokens_total": bonus_tokens_total,
            "committed_tokens_total": len(output_ids),
            "verification_committed_tokens_total": verification_committed_tokens_total,
            "effective_committed_tokens_per_verification": (
                verification_committed_tokens_total / verification_count if verification_count else None
            ),
            # Backward-compatible aliases. Unlike the previous implementation,
            # these values no longer compare scheduler -1 placeholders with
            # real sampled token IDs.
            "accepted_draft_token_count": accepted_candidate_tokens_total,
            "average_accepted_tokens_per_verification": (
                accepted_candidate_tokens_total / verification_count if verification_count else 0.0
            ),
            "draft_token_acceptance_rate": (
                accepted_candidate_tokens_total / scheduled_draft_token_count if scheduled_draft_token_count else 0.0
            ),
            "npu_memory": {
                "allocated_before": memory_before["allocated"],
                "reserved_before": memory_before["reserved"],
                "allocated_after": memory_after["allocated"],
                "reserved_after": memory_after["reserved"],
                "peak_allocated": memory_after["peak_allocated"],
                "peak_reserved": memory_after["peak_reserved"],
                "peak_allocated_increment": max(
                    0,
                    memory_after["peak_allocated"] - memory_before["allocated"],
                ),
                "peak_reserved_increment": max(
                    0,
                    memory_after["peak_reserved"] - memory_before["reserved"],
                ),
            },
        }
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
        "diagnostic_only": not performance,
        "performance_validated": performance,
        "performance_provisional": performance,
        "bit_exact_validated": False,
        "performance": performance_metrics,
    }
    if record["prompt_token_sha256"] != case["ordered_prompt_token_sha256"]:
        raise RuntimeError("Runtime prompt token IDs differ from the frozen manifest.")
    if early_trace_writer is not None:
        # This is reached only after the existing one-shot worker delivery,
        # scheduler/runner/KV/speculator release assertions have succeeded.
        early_trace_writer.finish(
            record,
            {
                "finished_event_observed_count": finished_lifecycle.observed_count,
                "finished_event_worker_delivery_count": finished_lifecycle.worker_delivery_count,
                "released_scheduler_runner_kv_proposal_state": True,
                "terminal_lifecycle": _proposal_lifecycle_trace(
                    speculator._terminal_proposal_lifecycle if speculator is not None else None
                ),
                "consumer_epochs": consumer_epochs,
            },
        )
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
    runtime: Any,
    plan: list[dict[str, Any]],
    *,
    mode: str,
    profile: str,
    manifest_hash: str,
    trace_first_round: bool = False,
    trace_output_index: int | None = None,
    trace_early_tokens: int = 0,
    trace_result_dir: Path | None = None,
    performance: bool = False,
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
        owner = {
            "rank": runtime.launch.rank,
            "mode": mode,
            "case_id": case["case_id"],
            "lifecycle_repeat": case["lifecycle_repeat"],
            "request_sequence_index": case["request_sequence_index"],
            "request_id": f"m25a-{mode}-{case['request_sequence_index']}-{case['lifecycle_repeat']}-{case['case_id']}",
        }
        with rank_trace_writer(trace_result_dir, owner, trace_early_tokens) as writer:
            record = _run_case(
                runtime,
                scheduler,
                case,
                mode=mode,
                profile=profile,
                manifest_sha256=manifest_hash,
                trace_first_round=trace_first_round,
                trace_output_index=trace_output_index,
                trace_early_tokens=trace_early_tokens,
                early_trace_writer=writer,
                performance=performance,
            )
        records.append(record)
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


def _settings_and_matrix() -> tuple[
    Any,
    Any,
    str,
    str,
    Path,
    str,
    list[dict[str, Any]],
    int,
    _ForensicTraceConfig,
    _PerformanceConfig,
]:
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
    kv_cache_bytes = _m2_5a_kv_cache_budget(os.environ)
    manifest_path = Path(manifest_value).expanduser().resolve()
    _, cases = load_profile_cases(manifest_path, profile)
    lifecycle_repeat = int(os.environ.get("DSPARK_M25A_LIFECYCLE_REPEAT", "1"))
    plan = build_execution_plan(cases, lifecycle_repeat)
    plan, trace_config = _select_forensic_cases(plan, os.environ)
    performance_config = _performance_config(
        os.environ,
        trace_config,
        lifecycle_repeat,
    )
    return (
        settings,
        launch,
        mode,
        profile,
        Path(result_value).expanduser().resolve(),
        sha256_file(manifest_path),
        plan,
        kv_cache_bytes,
        trace_config,
        performance_config,
    )


def test_dspark_single_request_realdata_npu() -> None:
    (
        settings,
        launch,
        mode,
        profile,
        result_dir,
        manifest_hash,
        plan,
        kv_cache_bytes,
        trace_config,
        performance_config,
    ) = _settings_and_matrix()
    _marker(
        TRACE_CONFIG,
        {
            "rank": launch.rank,
            "mode": mode,
            "case_id": trace_config.case_id,
            "first_round": trace_config.first_round,
            "output_index": trace_config.output_index,
            "early_tokens": trace_config.early_tokens,
            "selected_case_executions": len(plan),
        },
    )
    _marker(
        PERFORMANCE_CONFIG,
        {
            "rank": launch.rank,
            "mode": mode,
            "enabled": performance_config.enabled,
            "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING", "0"),
            "per_step_synchronize": not performance_config.enabled,
            "exact_token_cross_mode_blocking": not performance_config.enabled,
            "provisional": performance_config.enabled,
        },
    )
    records: list[dict[str, Any]] = []
    phase_timings: dict[str, float] = {}
    try:
        if mode == "target_only":
            contexts = ExitStack()
            state: dict[str, Any] = {}
            target_primary_error: BaseException | None = None
            try:
                runtime = _target_only_runtime(
                    settings,
                    launch,
                    contexts,
                    state,
                    enable_prefix_caching=False,
                    kv_cache_bytes=kv_cache_bytes,
                )
                phase_timings = dict(runtime.phase_timings)
                _validate_runtime_contract(runtime, mode, profile, kv_cache_bytes)
                records = _run_plan(
                    runtime,
                    plan,
                    mode=mode,
                    profile=profile,
                    manifest_hash=manifest_hash,
                    trace_first_round=trace_config.first_round,
                    trace_output_index=trace_config.output_index,
                    trace_early_tokens=trace_config.early_tokens,
                    trace_result_dir=result_dir,
                    performance=performance_config.enabled,
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
                phase_timings.update(runtime.phase_timings)
                _validate_runtime_contract(runtime, mode, profile, kv_cache_bytes)
                captured["records"] = _run_plan(
                    runtime,
                    plan,
                    mode=mode,
                    profile=profile,
                    manifest_hash=manifest_hash,
                    trace_first_round=trace_config.first_round,
                    trace_output_index=trace_config.output_index,
                    trace_early_tokens=trace_config.early_tokens,
                    trace_result_dir=result_dir,
                    performance=performance_config.enabled,
                )
                return True

            if prepare_harness._INITIALIZED_WORKER_CALLBACK is not None:
                raise RuntimeError("The initialized-worker callback is already installed.")
            previous_continue = prepare_harness._CONTINUE_AFTER_VERIFICATION
            prepare_harness._CONTINUE_AFTER_VERIFICATION = True
            prepare_harness._INITIALIZED_WORKER_CALLBACK = callback
            try:
                with prepare_harness.prepare_only_launch_config(
                    enable_prefix_caching=False,
                    kv_cache_bytes=kv_cache_bytes,
                ):
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
            "phase_timings": phase_timings,
            "performance_validated": performance_config.enabled,
            "performance_provisional": performance_config.enabled,
            "exact_token_cross_mode_blocking": not performance_config.enabled,
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
