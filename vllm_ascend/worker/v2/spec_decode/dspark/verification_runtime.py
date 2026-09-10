# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-owned confidence state. Instantiated only for explicit opt-in."""

import importlib.metadata
import json
import math
import time
from pathlib import Path

import torch

from vllm_ascend.spec_decode.dspark_verification import (
    ConfidenceRow,
    CostTable,
    allocate_prefixes,
    current_host_contexts,
    fingerprint,
    is_pure_decode,
    trim_scheduler_output,
)


def runtime_identity(config, hardware, confidence_sha256=None):
    model = config.model_config
    parallel = config.parallel_config
    return {
        "confidence_weights_sha256": confidence_sha256,
        "model": model.model,
        "revision": model.revision,
        "hf_config": fingerprint(model.hf_config.to_dict()),
        "hardware": hardware,
        "torch_version": str(torch.__version__),
        "torch_npu_version": importlib.metadata.version("torch-npu"),
        "ascend_compilation_config": (config.additional_config or {}).get("ascend_compilation_config", {}),
        "tp": parallel.tensor_parallel_size,
        "ep": parallel.enable_expert_parallel,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
        "target_mode": config.compilation_config.cudagraph_mode.name,
        "draft_mode": "eager",
        "capture_sizes": list(config.compilation_config.cudagraph_capture_sizes),
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "K": 5,
        "max_model_len": model.max_model_len,
        "block_size": config.cache_config.block_size,
        "gpu_memory_utilization": config.cache_config.gpu_memory_utilization,
    }


class ConfidenceVerification:
    def __init__(self, options, config, device):
        self.options = options
        self.config = config
        self.device = device
        self.rows = {}
        self.costs = None
        self.receipt = None
        self.scale = 1.0
        self.bias = 0.0
        self.calibration = {"status": "uncalibrated", "scale": 1.0, "bias": 0.0}
        self.confidence_head_calls = 0
        self.confidence_batches = 0
        self.specified_batches = 0
        self.fixed_admission_batches = 0
        self.generated = 0
        self.scheduled = 0
        self.verified = 0
        self.lengths = [0] * 6
        self.verified_by_position = [0] * 5
        self.generated_by_position = [0] * 5
        self.accepted_by_position = torch.zeros(5, dtype=torch.int64, device=device)
        self._candidate_positions = torch.arange(5, device=device)
        self.confidence_transfer_seconds = 0.0
        self.policy_seconds = 0.0
        self.selected_epochs = {}
        self.last_selection = None
        self.confidence_histogram = [0] * 10
        self.record_decisions = False
        self.decisions = []
        self.aggregate = {
            "logical_tokens_before": 0,
            "logical_tokens_after": 0,
            "selected_capacity_sum": 0,
            "estimated_seconds_sum": 0.0,
            "expected_progress_sum": 0.0,
            "confidence_sum": 0.0,
            "confidence_count": 0,
            "all_full_decisions": 0,
        }

    def bind_model(self, model):
        if self.options["mode"] != "confidence" and not self.options.get("profile"):
            self.receipt = {"status": "test_only_specified_lengths", "confidence_head_used": False}
            return
        receipt = getattr(model, "confidence_weight_receipt", None)
        if model.model.confidence_head is None or not receipt or not receipt["loaded_parameters"]:
            raise ValueError("Confidence verification requires checkpoint-loaded confidence head weights.")
        self.receipt = dict(receipt)
        path = self.options.get("calibration")
        if path:
            calibration = json.loads(Path(path).read_text())
            if (
                calibration.get("schema_version") != 1
                or calibration.get("weights_sha256") != receipt["weights_sha256"]
                or calibration.get("split") != "calibration"
                or not calibration.get("dataset_sha256")
            ):
                raise ValueError(
                    "Calibration must identify separate calibration data and the loaded confidence weights."
                )
            self.scale, self.bias = float(calibration["scale"]), float(calibration["bias"])
            if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.bias):
                raise ValueError("Invalid confidence calibration parameters.")
            self.calibration = calibration

    def record(self, request_ids, epoch, hidden_states, markov_embeds, model):
        self.generated += len(request_ids) * 5
        for position in range(5):
            self.generated_by_position[position] += len(request_ids)
        if self.options["mode"] != "confidence" and not self.options.get("profile"):
            return
        if self.receipt is None:
            raise RuntimeError("Confidence head was not bound to checkpoint provenance.")
        started = time.perf_counter()
        logits = model.confidence_logits(hidden_states, torch.stack(markov_embeds, dim=1).flatten(0, 1))
        logits = logits.float().reshape(len(request_ids), 5)
        # One batch transfer, after draft completion, never one transfer/request.
        # Check logits too: sigmoid(inf) must not silently turn a bad head into 1.
        host_logits = logits.to(device="cpu")
        if not torch.isfinite(host_logits).all():
            raise ValueError("Non-finite DSpark confidence logits.")
        probabilities = torch.sigmoid(host_logits * self.scale + self.bias).tolist()
        self.confidence_head_calls += 1
        self.confidence_transfer_seconds += time.perf_counter() - started
        for request_id, values in zip(request_ids, probabilities):
            self.rows[request_id] = ConfidenceRow(request_id, epoch, tuple(values))

    def load_costs(self):
        """Benchmark calls this at the startup RPC, before warmup/measurement."""
        if self.costs is None:
            identity = runtime_identity(
                self.config, torch.npu.get_device_name(self.device), self.receipt["weights_sha256"]
            )
            self.costs = CostTable.load(self.options["cost_profile"], identity)

    def select(self, runner, output):
        started = time.perf_counter()
        owners = runner.speculator._published_proposal_owners
        self.rows = {key: row for key, row in self.rows.items() if key in owners}
        candidates = {
            key: len(tokens)
            for key, tokens in output.scheduled_spec_decode_tokens.items()
            if output.num_scheduled_tokens[key] == len(tokens) + 1
        }
        self.last_selection = None
        self.selected_epochs = {}
        if not candidates:
            return output
        if set(candidates) - owners.keys():
            raise ValueError("Scheduled candidates lack current proposal owners.")
        epochs = {key: owners[key].producer_epoch for key in candidates}
        pure_decode = is_pure_decode(runner.req_states, output)
        decision = None
        if not pure_decode:
            # Mixed admission retains the existing verification widths. No
            # benefit is assigned to incomplete prefill and no decode cost is
            # extrapolated to an unprofiled prefill execution.
            self.fixed_admission_batches += 1
            lengths = dict(candidates)
        elif self.options["mode"] == "specified_lengths":
            self.specified_batches += 1
            specified = self.options["lengths"]
            lengths = {
                key: min(candidates[key], specified[index % len(specified)])
                for index, key in enumerate(sorted(candidates))
            }
        else:
            if set(candidates) - self.rows.keys():
                raise ValueError("Missing current confidence for scheduled candidate owners.")
            rows = [self.rows[key] for key in sorted(candidates)]
            if any(row.producer_epoch != epochs[row.request_id] for row in rows):
                raise ValueError("Stale confidence producer epoch; refusing to schedule another candidate's scores.")
            self.load_costs()
            base_tokens = output.total_num_scheduled_tokens - sum(candidates.values())
            # Only completed-prefill requests earn a sampling benefit.
            states = runner.req_states
            sampling = 0
            current_contexts = current_host_contexts(states, output)
            contexts = []
            for key, tokens in output.num_scheduled_tokens.items():
                index = states.req_id_to_index.get(key)
                if index is None:
                    continue
                non_draft = tokens - candidates.get(key, 0)
                computed = current_contexts[key]
                contexts.append(computed)
                sampling += computed + non_draft >= int(states.prefill_len.np[index])
            decision = allocate_prefixes(
                rows,
                candidates,
                base_tokens=base_tokens,
                sampling_requests=sampling,
                draft_requests=len(output.num_scheduled_tokens),
                context=max(contexts, default=0),
                costs=self.costs,
            )
            self.confidence_batches += 1
            lengths = decision.lengths
        # CPU collective, once per target batch. Rank zero owns the policy;
        # every rank validates its local owner epochs against that decision.
        from vllm.distributed import get_tp_group

        group = get_tp_group()
        decision_packet = group.broadcast_object((lengths, epochs) if group.rank_in_group == 0 else None, src=0)
        lengths, broadcast_epochs = decision_packet
        if broadcast_epochs != epochs or set(lengths) != set(candidates):
            raise ValueError("TP ranks disagree on request/producer-epoch ownership.")
        result = trim_scheduler_output(output, lengths)
        self.selected_epochs = epochs
        self.last_selection = {
            "mode": self.options["mode"],
            "lengths": lengths,
            "producer_epochs": epochs,
            "confidence_epochs": epochs if self.options["mode"] == "confidence" and pure_decode else None,
            "policy": "current_epoch_survival_cost"
            if decision is not None
            else ("specified_lengths" if pure_decode else "fixed_mixed_admission"),
            "selected_graph_capacity": decision.capacity if decision is not None else None,
            "estimated_seconds": decision.estimated_seconds if decision is not None else None,
            "expected_progress": decision.expected_progress if decision is not None else None,
            "actual_tokens": result.total_num_scheduled_tokens,
        }
        if decision is not None:
            self.aggregate["logical_tokens_before"] += output.total_num_scheduled_tokens
            self.aggregate["logical_tokens_after"] += result.total_num_scheduled_tokens
            self.aggregate["selected_capacity_sum"] += decision.capacity
            self.aggregate["estimated_seconds_sum"] += decision.estimated_seconds
            self.aggregate["expected_progress_sum"] += decision.expected_progress
            self.aggregate["all_full_decisions"] += lengths == candidates
            self.aggregate["confidence_sum"] += sum(sum(row.conditional) for row in rows)
            self.aggregate["confidence_count"] += len(rows) * 5
            for row in rows:
                for probability in row.conditional:
                    self.confidence_histogram[min(int(probability * 10), 9)] += 1
        if self.record_decisions:
            self.decisions.append(self.last_selection)
        self.scheduled += sum(lengths.values())
        self.policy_seconds += time.perf_counter() - started
        return result

    def accepted(self, request_ids, lengths, num_sampled, producer_epochs):
        if len(request_ids) != len(producer_epochs) or any(
            self.selected_epochs.get(key) != epoch for key, epoch in zip(request_ids, producer_epochs)
        ):
            raise ValueError("Verification consumption lacks a current request/epoch decision.")
        self.verified += sum(lengths)
        for length in lengths:
            self.lengths[length] += 1
            for position in range(length):
                self.verified_by_position[position] += 1
        positions = self._candidate_positions
        self.accepted_by_position.add_((positions[None, :] < (num_sampled - 1)[:, None]).sum(0))
        for key in request_ids:
            self.rows.pop(key, None)

    def snapshot(self):
        accepted = self.accepted_by_position.cpu().tolist()  # phase-boundary RPC only
        return {
            "schema_version": 1,
            "aggregate": dict(self.aggregate),
            "confidence_histogram": list(self.confidence_histogram),
            "cost_profile": {
                "path": self.options.get("cost_profile"),
                "identity": self.costs.identity,
                "context_range": list(self.costs.context_range),
                "capacities": sorted(self.costs.target_seconds),
                "layout_cells": len(self.costs.cells),
            }
            if self.costs is not None
            else None,
            "mode": self.options["mode"],
            "weights": self.receipt,
            "calibration": self.calibration,
            "confidence_head_calls": self.confidence_head_calls,
            "confidence_batches": self.confidence_batches,
            "specified_batches": self.specified_batches,
            "fixed_admission_batches": self.fixed_admission_batches,
            "generated": self.generated,
            "scheduled": self.scheduled,
            "verified": self.verified,
            "accepted": sum(accepted),
            "length_histogram": list(self.lengths),
            "verified_by_position": list(self.verified_by_position),
            "generated_by_position": list(self.generated_by_position),
            "accepted_by_position": accepted,
            "confidence_transfer_seconds": self.confidence_transfer_seconds,
            "policy_seconds": self.policy_seconds,
            "last_selection": self.last_selection,
            "decisions": list(self.decisions) if self.record_decisions else None,
        }
