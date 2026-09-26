# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host-only FULL counters; no per-step I/O, tensor reads or synchronization."""

from collections import Counter

from vllm_ascend.diagnostics.dspark_benchmark_worker import _FullReplayObserver, _host_count

MAX_FULL_CALLS = 65536  # four workloads cannot silently overflow telemetry


class PerformanceReplayObserver(_FullReplayObserver):
    def __init__(self, runner):
        super().__init__(runner)
        self.max_query_length = runner.speculator.num_speculative_steps + 1
        self.original_publish = None
        if runner.vllm_config.additional_config.get("dspark_fixed_k_comparison") is True:
            self.publish_had_local = "_build_core_proposal" in vars(runner.speculator)
            self.original_publish = runner.speculator._build_core_proposal
            runner.speculator._build_core_proposal = self.publish
        self.reset()

    def publish(self, proposal, result):
        # Runs after real Markov generation and original owner/epoch validation.
        # Tensor dimensions are host metadata: no device value reads or copies.
        tokens = self.original_publish(proposal, result)
        if tuple(tokens.shape) != (result.num_reqs, self.max_query_length - 1):
            raise ValueError("Published candidate dimensions disagree with fixed K")
        self.proposals["calls"] += 1
        if self.proposals["calls"] > MAX_FULL_CALLS:
            raise ValueError("Proposal telemetry capacity exceeded")
        self.proposals["requests"] += result.num_reqs
        self.proposals["candidates"] += tokens.numel()
        if self.proposals["first_epoch"] is None:
            self.proposals["first_epoch"] = result.step_epoch
        self.proposals["last_epoch"] = result.step_epoch
        return tokens

    def close(self):
        if self.original_publish is not None:
            speculator = self.runner.speculator
            if speculator._build_core_proposal == self.publish:
                if self.publish_had_local:
                    speculator._build_core_proposal = self.original_publish
                else:
                    del speculator._build_core_proposal
            self.original_publish = None
        super().close()

    def reset(self):
        if self.pending is not None or self.error or self.failed_execution_count:
            raise ValueError("Cannot reset in-flight/failed performance observations")
        # Only telemetry resets; never rewind proposal epochs, RNG or request state.
        self.shapes.clear()
        self.query_layouts.clear()
        self.layouts = Counter()
        self.lengths = Counter()
        self.calls = 0
        self.proposals = dict(calls=0, requests=0, candidates=0, first_epoch=None, last_epoch=None)

    def record_layout(self, batch, padded):
        lengths = tuple(_host_count(v) for v in batch.num_scheduled_tokens)
        if (
            not lengths
            or sum(lengths) != _host_count(batch.num_tokens)
            or any(not 1 <= v <= self.max_query_length for v in lengths)
        ):
            raise ValueError("FULL decode query layout unavailable or invalid")
        self.calls += 1
        if self.calls > MAX_FULL_CALLS:
            raise ValueError("Performance receipt capacity exceeded")
        self.layouts[(len(lengths), sum(lengths), padded)] += 1
        self.lengths.update(lengths)

    def snapshot(self):
        row = super().snapshot()
        row["performance"] = {
            "source": "completed MRV2 FULL InputBatch CPU dimensions",
            "calls": self.calls,
            "max_calls": MAX_FULL_CALLS,
            "layouts": [
                {"requests": n, "query_tokens": q, "capacity": c, "count": count}
                for (n, q, c), count in sorted(self.layouts.items())
            ],
            "query_length_histogram": dict(sorted(self.lengths.items())),
            "added_device_reads": 0,
            "step_file_writes": 0,
            "scope": "FULL decode only; ordinary prefill excluded",
        }
        if self.original_publish is not None:
            row["performance"]["published_proposals"] = dict(self.proposals)
        return row
