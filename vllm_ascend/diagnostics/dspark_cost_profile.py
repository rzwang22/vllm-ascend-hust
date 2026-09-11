# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit isolated-profile instrumentation; never enabled by performance defaults."""

import torch

from vllm_ascend.spec_decode.dspark_verification import COST_CONTEXT_SEMANTICS
from vllm_ascend.worker.v2.spec_decode.dspark.verification_runtime import runtime_identity


def profile_context(batch, max_model_len):
    """Snapshot existing host metadata, only actual rows, before the timed call.

    seq_lens_np was prepared by Ascend's existing _update_seq_lens_cpu wait and
    rejection-corrected copy. No tensor transfer/synchronization is added here.
    Both target and adjacent draft are indexed by the target's pre-query
    scheduler upper bound, matching the pre-admission policy lookup.
    """
    n = batch.num_reqs
    scheduled = [int(v) for v in batch.num_scheduled_tokens[:n]]
    upper = [int(v) for v in batch.num_computed_tokens_np[:n]]
    attention = [int(v) for v in batch.seq_lens_np[:n]]
    if n <= 0 or len(scheduled) != n or len(upper) != n or len(attention) != n:
        raise ValueError("Incomplete actual-request profile context")
    return {
        "context": max(upper),
        "context_semantics": COST_CONTEXT_SEMANTICS,
        "scheduler_computed_upper_bounds": upper,
        "effective_kv_before_query": [length - q for length, q in zip(attention, scheduled)],
        "attention_seq_lens": attention,
        "max_model_len": int(max_model_len),
        "effective_length_source": "AscendInputBatch.seq_lens_np minus current query; existing corrected CPU copy",
    }


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
        batch = self.runner.input_batch
        context = profile_context(batch, self.runner.vllm_config.model_config.max_model_len)
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        result = function(argument)
        end.record()
        # Failed launches are not recorded. Synchronization/errors are checked
        # at the boundary before any profile cache can be written.
        metadata = {
            "request_ids": list(batch.req_ids),
            "kind": kind,
            "size": size,
            **context,
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
