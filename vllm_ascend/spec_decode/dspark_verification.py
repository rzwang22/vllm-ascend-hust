# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host policy for greedy DSpark verification. No device or global state."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_DRAFTS = 5
CONFIG_KEY = "dspark_confidence_verification"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def verification_options(config: Any) -> dict | None:
    options = (getattr(config, "additional_config", None) or {}).get(CONFIG_KEY)
    if options is None:
        return None
    if not isinstance(options, dict) or options.get("mode") not in {"confidence", "specified_lengths"}:
        raise ValueError("DSpark verification mode must be confidence or specified_lengths.")
    unknown = options.keys() - {"mode", "lengths", "cost_profile", "calibration", "profile"}
    if unknown:
        raise ValueError(f"Unknown DSpark verification options: {sorted(unknown)}")
    spec = config.speculative_config
    parallel = config.parallel_config
    if (
        spec is None
        or spec.method != "dspark"
        or spec.num_speculative_tokens != MAX_DRAFTS
        or not spec.enforce_eager
        or config.model_config.enforce_eager
        or config.compilation_config.cudagraph_mode.name != "FULL_DECODE_ONLY"
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or getattr(config, "lora_config", None) is not None
    ):
        raise ValueError("DSpark variable verification requires K=5, FULL_DECODE_ONLY, eager draft, DP1/PP1, no LoRA.")
    if type(options.get("profile", False)) is not bool:
        raise ValueError("profile must be an explicit boolean.")
    if options.get("profile") and options["mode"] != "specified_lengths":
        raise ValueError("Isolated cost profiling uses specified_lengths, not learned-policy performance runs.")
    if options["mode"] == "specified_lengths":
        lengths = options.get("lengths")
        if not isinstance(lengths, list) or not lengths:
            raise ValueError("Test-only specified_lengths requires a nonempty lengths list.")
        for length in lengths:
            validate_length(length)
    elif not options.get("cost_profile"):
        raise ValueError("Confidence verification requires a measured cost_profile.")
    return dict(options)


def validate_length(length: int) -> None:
    if type(length) is not int or not 0 <= length <= MAX_DRAFTS:
        raise ValueError(f"Verification length must be an integer in [0,5], got {length!r}.")


@dataclass(frozen=True)
class ConfidenceRow:
    request_id: str
    producer_epoch: int
    conditional: tuple[float, ...]

    def __post_init__(self):
        if not self.request_id or self.producer_epoch < 1 or len(self.conditional) != MAX_DRAFTS:
            raise ValueError("Invalid confidence ownership or width.")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.conditional):
            raise ValueError("Confidence probabilities must be finite and in [0,1].")

    def survival(self) -> tuple[float, ...]:
        product = 1.0
        values = []
        for probability in self.conditional:
            product *= probability
            values.append(product)
        return tuple(values)


@dataclass(frozen=True)
class CostTable:
    """Measured target graph costs (by capacity), draft costs (by request count)."""

    target_seconds: dict[int, float]
    draft_seconds: dict[int, float]
    scheduler_seconds: float
    context_range: tuple[int, int]
    identity: dict
    cells: tuple = ()
    startup_lookup: dict | None = None
    context_ceilings: tuple = ()

    @classmethod
    def load(cls, path: str, identity: dict) -> CostTable:
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") == 2:
            return cls.load_startup(data, identity)
        raise ValueError("Cost profile requires measured startup schema 2; legacy schema 1 must be re-profiled.")

    @classmethod
    def load_startup(cls, data: dict, identity: dict) -> CostTable:
        if (
            data.get("source") != "startup_npu_event_profile"
            or data.get("identity") != identity
            or not data.get("raw_measurements_sha256")
            or data.get("unit") != "seconds"
            or data.get("performance_eligible") is not False
            or data.get("model_initializations") != 1
        ):
            raise ValueError("Incompatible startup cost profile identity, units or lifecycle.")
        cells = data.get("cells", [])
        contexts = data.get("context_ceilings", [])
        requests = data.get("request_grid", [])
        captures = identity["capture_sizes"]
        if (
            not cells
            or not contexts
            or not requests
            or contexts != sorted(set(contexts))
            or requests != sorted(set(requests))
            or contexts[0] < 1
            or requests[0] != 1
            or requests[-1] != identity["max_num_seqs"]
        ):
            raise ValueError("Missing startup profile coverage.")
        expected = {
            (n, cap, ctx)
            for ctx in contexts
            for n in requests
            for previous, cap in zip([0] + captures[:-1], captures)
            if n <= cap and 6 * n > previous
        }
        actual = {(c["requests"], c["capacity"], c["context_ceiling"]) for c in cells}
        if actual != expected or len(actual) != len(cells):
            raise ValueError("Incomplete or duplicate startup layout/context coverage.")
        for c in cells:
            if any(not math.isfinite(c[key]) or c[key] <= 0 for key in ("target_seconds", "draft_seconds")):
                raise ValueError("Invalid measured startup cost.")
        overhead = data["scheduler_seconds"]
        if not math.isfinite(overhead) or overhead < 0:
            raise ValueError("Invalid scheduling overhead.")
        lookup = {}
        # Resolve request bucketing once at startup, not once per candidate
        # budget. Entries retain measured layout/context costs, no extrapolation.
        for n in range(1, identity["max_num_seqs"] + 1):
            for previous, cap in zip([0] + captures[:-1], captures):
                if n <= cap and 6 * n > previous:
                    for context in contexts:
                        candidates = [
                            c
                            for c in cells
                            if c["capacity"] == cap and c["requests"] >= n and c["context_ceiling"] == context
                        ]
                        if not candidates:
                            raise ValueError("Missing bounded startup layout/context cost coverage.")
                        cell = min(candidates, key=lambda c: c["requests"])
                        lookup[n, cap, context] = cell["target_seconds"] + cell["draft_seconds"] + overhead
        return cls(
            {cap: max(c["target_seconds"] for c in cells if c["capacity"] == cap) for cap in captures},
            {},
            overhead,
            (0, contexts[-1]),
            identity,
            tuple(cells),
            lookup,
            tuple(contexts),
        )

    def cost(self, requests: int, tokens: int, context: int) -> tuple[int, float]:
        if not self.context_range[0] <= context <= self.context_range[1]:
            raise ValueError("Context outside measured cost profile; run an isolated profile for this context.")
        if self.cells:
            if requests <= 0 or not requests <= tokens <= requests * MAX_DRAFTS + requests:
                raise ValueError("Invalid pure-decode request/token layout.")
            capacity = next((cap for cap in sorted(self.target_seconds) if cap >= tokens), None)
            bucket = next((ctx for ctx in self.context_ceilings if ctx >= context), None)
            value = self.startup_lookup.get((requests, capacity, bucket))
            if value is None:
                raise ValueError("Missing bounded startup layout/context cost coverage.")
            return capacity, value
        if requests not in self.draft_seconds:
            raise ValueError(f"Missing measured eager draft cost for {requests} requests.")
        capacities = sorted(capacity for capacity in self.target_seconds if capacity >= tokens)
        if not capacities:
            raise ValueError("Actual target tokens exceed profiled graph capacity.")
        capacity = capacities[0]
        return capacity, self.target_seconds[capacity] + self.draft_seconds[requests] + self.scheduler_seconds


@dataclass(frozen=True)
class Decision:
    lengths: dict[str, int]
    epochs: dict[str, int]
    actual_tokens: int
    capacity: int
    expected_progress: float
    estimated_seconds: float


def allocate_prefixes(
    rows: list[ConfidenceRow],
    capacities: dict[str, int],
    *,
    base_tokens: int,
    sampling_requests: int,
    draft_requests: int,
    context: int,
    costs: CostTable,
) -> Decision:
    """Enumerate marginal-survival budgets; ties use request ID then position.

    Confidence and budget both use the current candidate's producer epoch.
    Unlike upstream's stale host budget, accurate host lengths are available
    before Ascend builds attention metadata. No post-verification feedback is
    used to select the current prefix.
    """
    if len({row.request_id for row in rows}) != len(rows) or set(capacities) != {row.request_id for row in rows}:
        raise ValueError("Duplicate or missing confidence request ownership.")
    if base_tokens < sampling_requests or sampling_requests < len(rows) or draft_requests <= 0:
        raise ValueError("Invalid sampling/admission counts.")
    lengths = {row.request_id: 0 for row in rows}
    epochs = {row.request_id: row.producer_epoch for row in rows}
    survival = {row.request_id: row.survival() for row in rows}
    heap = []
    for row in rows:
        validate_length(capacities[row.request_id])
        if capacities[row.request_id]:
            heapq.heappush(heap, (-survival[row.request_id][0], row.request_id, 0))
    capacity, seconds = costs.cost(draft_requests, base_tokens, context)
    best = Decision(dict(lengths), epochs, base_tokens, capacity, float(sampling_requests), seconds)
    progress = float(sampling_requests)
    tokens = base_tokens
    while heap:
        negative_probability, request_id, position = heapq.heappop(heap)
        lengths[request_id] += 1
        progress -= negative_probability
        tokens += 1
        capacity, seconds = costs.cost(draft_requests, tokens, context)
        # Equal utility keeps the smaller budget, avoiding useless padding work.
        if progress / seconds > best.expected_progress / best.estimated_seconds:
            best = Decision(dict(lengths), epochs, tokens, capacity, progress, seconds)
        next_position = position + 1
        if next_position < capacities[request_id]:
            heapq.heappush(heap, (-survival[request_id][next_position], request_id, next_position))
    return best


def trim_scheduler_output(output: Any, lengths: dict[str, int]) -> Any:
    """Copy worker-local scheduling fields; preserve scheduler-side accounting.

    Core subtracts original draft count minus accepted count on completion.
    Worker post_update instead sees the compact query and actual rejections;
    both advance by one plus accepted tokens, including ell=0.
    """
    import copy

    result = copy.copy(output)
    result.num_scheduled_tokens = dict(output.num_scheduled_tokens)
    result.scheduled_spec_decode_tokens = {
        key: list(value) for key, value in output.scheduled_spec_decode_tokens.items()
    }
    for request_id, length in lengths.items():
        validate_length(length)
        old = result.scheduled_spec_decode_tokens.get(request_id)
        if old is None or length > len(old) or result.num_scheduled_tokens[request_id] != len(old) + 1:
            raise ValueError("Selected prefix must belong to a scheduled pure decode proposal.")
        result.num_scheduled_tokens[request_id] -= len(old) - length
        result.scheduled_spec_decode_tokens[request_id] = old[:length]
    result.total_num_scheduled_tokens = sum(result.num_scheduled_tokens.values())
    return result


def fill_varlen_query_padding(query_start_loc: Any, actual_requests: int, capacity_requests: int, capacity_tokens: int):
    """Zero-query dummy rows; trailing token storage is PAD_SLOT_ID, not a request."""
    if not 0 < actual_requests <= capacity_requests or len(query_start_loc) < capacity_requests + 1:
        raise ValueError("Variable graph request capacity exceeded.")
    actual = int(query_start_loc[actual_requests])
    if not 0 < actual <= capacity_tokens or query_start_loc[0] != 0:
        raise ValueError("Variable graph token capacity exceeded.")
    for index in range(actual_requests):
        if not 1 <= int(query_start_loc[index + 1] - query_start_loc[index]) <= MAX_DRAFTS + 1:
            raise ValueError("Variable decode query lengths must be in [1,6].")
    query_start_loc[actual_requests + 1 :] = actual
    return query_start_loc, capacity_requests


def current_host_contexts(states: Any, output: Any) -> dict[str, int]:
    """Current scheduler upper bounds, before runner.update_requests executes."""
    contexts = {key: int(states.num_computed_tokens_np[index]) for key, index in states.req_id_to_index.items()}
    cached = getattr(output, "scheduled_cached_reqs", None)
    if cached is not None:
        contexts.update(zip(cached.req_ids, getattr(cached, "num_computed_tokens", ())))
    return contexts


def is_pure_decode(states: Any, output: Any) -> bool:
    contexts = current_host_contexts(states, output)
    return bool(output.num_scheduled_tokens) and all(
        key in states.req_id_to_index
        and contexts[key] >= int(states.prefill_len.np[states.req_id_to_index[key]])
        and 1 <= tokens <= MAX_DRAFTS + 1
        for key, tokens in output.num_scheduled_tokens.items()
    )
