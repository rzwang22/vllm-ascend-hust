# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded host-only FULL counters; no per-step I/O, tensor reads or synchronization."""

from collections import Counter

from vllm_ascend.diagnostics.dspark_benchmark_worker import _FullReplayObserver, _host_count

MAX_FULL_CALLS = 65536  # four workloads cannot silently overflow telemetry


class PerformanceReplayObserver(_FullReplayObserver):
    def __init__(self, runner):
        super().__init__(runner)
        self.reset()

    def reset(self):
        if self.pending is not None or self.error or self.failed_execution_count:
            raise ValueError("Cannot reset in-flight/failed performance observations")
        # Only telemetry resets; never rewind proposal epochs, RNG or request state.
        self.shapes.clear()
        self.query_layouts.clear()
        self.layouts = Counter()
        self.lengths = Counter()
        self.calls = 0

    def record_layout(self, batch, padded):
        lengths = tuple(_host_count(v) for v in batch.num_scheduled_tokens)
        if not lengths or sum(lengths) != _host_count(batch.num_tokens) or any(not 1 <= v <= 6 for v in lengths):
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
        return row
