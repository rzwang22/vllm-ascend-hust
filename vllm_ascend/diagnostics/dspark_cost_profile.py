# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit isolated-profile instrumentation; never enabled by performance defaults."""

import torch

from vllm_ascend.worker.v2.spec_decode.dspark.verification_runtime import runtime_identity


class IsolatedCostProfiler:
    def __init__(self, runner):
        self.runner = runner
        self.events = []
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
        self.events.append((kind, size, context, start, end))
        return result

    def target(self, descriptor):
        return self.timed("target", descriptor.num_tokens, self.graph, descriptor)

    def propose(self, inputs):
        return self.timed("draft", inputs.num_reqs, self.draft, inputs)

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
                {"kind": kind, "size": size, "context": context, "seconds": start.elapsed_time(end) / 1000}
                for kind, size, context, start, end in self.events
            ],
        }
