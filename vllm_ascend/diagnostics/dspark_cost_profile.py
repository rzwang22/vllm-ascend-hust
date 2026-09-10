# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit isolated-profile instrumentation; never enabled by performance defaults."""

import torch

from vllm_ascend.worker.v2.spec_decode.dspark.verification_runtime import runtime_identity


class IsolatedCostProfiler:
    def __init__(self, runner):
        self.runner = runner
        self.events = []
        self.point = None
        self.last_full_batch = None
        self.graph = runner.cudagraph_manager.run_fullgraph
        self.draft = runner.speculator._execute_draft
        runner.cudagraph_manager.run_fullgraph = self.target
        runner.speculator._execute_draft = self.propose

    def timed(self, kind, size, function, argument):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = function(argument)
        end.record()
        # Failed launches are not recorded. Synchronization/errors are checked
        # at the boundary before any profile cache can be written.
        batch = self.runner.input_batch
        context = int(batch.num_computed_tokens_np.max())
        metadata = {
            "request_ids": list(batch.req_ids),
            "kind": kind,
            "size": size,
            "context": context,
            "requests": int(batch.num_reqs),
            "actual_tokens": int(batch.num_tokens),
            "capacity": int(batch.num_tokens_after_padding),
            "request_capacity": int(batch.num_reqs_after_padding),
            "query_lengths": [int(n) for n in batch.num_scheduled_tokens[: batch.num_reqs]],
            "full_decode": (kind == "target" or self.last_full_batch is batch)
            and not bool(batch.is_prefilling_np[: batch.num_reqs].any()),
            "point": self.point,
        }
        self.events.append((metadata, start, end))
        return result

    def target(self, descriptor):
        result = self.timed("target", descriptor.num_tokens, self.graph, descriptor)
        self.last_full_batch = self.runner.input_batch
        return result

    def propose(self, inputs):
        try:
            return self.timed("draft", inputs.num_reqs, self.draft, inputs)
        finally:
            self.last_full_batch = None

    def snapshot(self):
        torch.npu.synchronize()  # Profile-only phase boundary, not a performance step.
        return {
            "source": "isolated_npu_event_profile",
            "identity": runtime_identity(
                self.runner.vllm_config,
                torch.npu.get_device_name(self.runner.device),
                self.runner.speculator.confidence_verification.receipt["weights_sha256"],
            ),
            "measurements": [
                {**metadata, "seconds": start.elapsed_time(end) / 1000} for metadata, start, end in self.events
            ],
        }

    def begin_point(self, point, lengths):
        """Called only after the frontend drained all requests from the last point.

        Scheduler still owns terminal cleanup/freeing KV. New globally unique
        request IDs enter via normal admission; no cache/state is transplanted.
        Only diagnostic events and the test-only prefix pattern change here.
        """
        from vllm_ascend.spec_decode.dspark_verification import validate_length

        adaptive = self.runner.speculator.confidence_verification
        if not adaptive.options.get("profile") or not isinstance(point, str) or not lengths:
            raise ValueError("Profile point RPC requires an isolated specified-length profile engine.")
        for length in lengths:
            validate_length(length)
        torch.npu.synchronize()
        self.events.clear()
        self.last_full_batch = None
        self.point = point
        adaptive.options["lengths"] = list(lengths)
        return {"point": point, "lengths": list(lengths), "cleanup": "scheduler_owned_unique_request_ids"}
