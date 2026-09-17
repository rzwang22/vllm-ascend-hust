# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host decision/real execution links; no model or graph tensor probes."""

import copy
import json
from pathlib import Path

from tools.dspark.shutdown_policy import confidence_acceptance_enabled
from vllm_ascend.diagnostics.dspark_benchmark_worker import _cudagraph_mode_name, _FullReplayObserver

MAX_EXECUTIONS = 2048
MAX_REQUEST_ROWS = 32768
MAX_LOG_BYTES = 64 * 1024 * 1024


class ConfidenceReceipts(_FullReplayObserver):
    def __init__(self, runner):
        if not confidence_acceptance_enabled(runner.vllm_config.additional_config):
            raise ValueError("Confidence receipts require explicit confidence/profile=false acceptance")
        self.max_requests = getattr(getattr(runner.vllm_config, "scheduler_config", None), "max_num_seqs", 64)
        if self.max_requests not in (64, 128, 256):
            raise ValueError("Unsupported confidence acceptance tier")
        self.max_request_rows = MAX_REQUEST_ROWS * (self.max_requests // 64)
        self.max_log_bytes = MAX_LOG_BYTES * (self.max_requests // 64)
        super().__init__(runner)
        self.adaptive = runner.speculator.confidence_verification
        self.original_select = self.adaptive.select
        self.original_accepted = self.adaptive.accepted
        self.select_local = "select" in vars(self.adaptive)
        self.accepted_local = "accepted" in vars(self.adaptive)
        self.adaptive.select = self.select
        self.adaptive.accepted = self.accepted
        self.records = []
        self.sampled = []
        self.request_rows = 0
        self.log_bytes = 0
        self.active = None
        self.directory = Path(runner.vllm_config.additional_config["dspark_profile_failure_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"confidence-rank-{runner.speculator.rank}.jsonl"

    def persist(self, record, phase):
        payload = json.dumps({"phase": phase, **record}, allow_nan=False) + "\n"
        self.log_bytes += len(payload.encode())
        if self.log_bytes > self.max_log_bytes:
            raise ValueError("Confidence receipt log byte bound exceeded")
        with self.path.open("a") as out:
            out.write(payload)

    def select(self, runner, output):
        from vllm_ascend.spec_decode.dspark_verification import current_host_contexts

        if len(self.records) >= MAX_EXECUTIONS:
            raise ValueError("Confidence execution receipt bound exceeded")
        if not output.total_num_scheduled_tokens:
            return self.original_select(runner, output)
        contexts = current_host_contexts(runner.req_states, output)
        if len(output.num_scheduled_tokens) > self.max_requests or max(contexts.values(), default=0) > 640:
            raise ValueError("Confidence acceptance exceeds frozen request/context coverage")
        result = self.original_select(runner, output)
        selection = copy.deepcopy(self.adaptive.last_selection)
        self.request_rows += len(result.num_scheduled_tokens)
        if self.request_rows > self.max_request_rows:
            raise ValueError("Confidence request receipt bound exceeded")
        record = {
            "execution": len(self.records) + 1,
            "selection": selection,
            "context_upper": contexts,
            "scheduled_queries": dict(result.num_scheduled_tokens),
            "confidence": {
                key: {"producer_epoch": row.producer_epoch, "conditional": list(row.conditional)}
                for key, row in self.adaptive.rows.items()
                if selection and key in selection["lengths"]
            },
            "confidence_source": "loaded DSparkConfidenceHead conditional probabilities; uncalibrated",
            "target": None,
            "accepted": None,
        }
        if selection and selection["policy"] == "current_epoch_survival_cost":
            costs = self.adaptive.costs
            n = len(result.num_scheduled_tokens)
            cap, seconds = costs.cost(n, result.total_num_scheduled_tokens, max(contexts.values(), default=0))
            cell = min(
                (c for c in costs.cells if c["requests"] >= n and c["capacity"] == cap),
                key=lambda c: (c["context_ceiling"], c["requests"]),
            )
            record["lookup"] = {
                "requests": n,
                "sampled_requests": cell["requests"],
                "capacity": cap,
                "context_ceiling": cell["context_ceiling"],
                "estimated_seconds": seconds,
                "token_budget": selection["actual_tokens"],
                "policy": "maximize expected progress / measured seconds over legal token budgets",
            }
        self.active = record
        self.records.append(record)
        self.persist(record, "selected_before_target")
        return result

    def execute_model(self, *args, **kwargs):
        self.active = None
        result = super().execute_model(*args, **kwargs)
        record = self.active
        if record is not None:
            state = self.runner.execute_model_state
            if state is None:
                raise ValueError("Selected target has no completed execution state")
            batch = state.input_batch
            record["target"] = {
                "request_ids": list(batch.req_ids),
                "query_lengths": [int(v) for v in batch.num_scheduled_tokens],
                "valid_tokens": int(batch.num_tokens),
                "capacity": int(batch.num_tokens_after_padding),
                "full_replay": False,
            }
            # run_fullgraph marked this exact record only after the real call returned.
            if record.get("graph"):
                record["target"]["full_replay"] = True
            self.persist(record, "target_returned")
        return result

    def run_fullgraph(self, desc):
        result = super().run_fullgraph(desc)
        if self.active is not None:
            self.active["graph"] = {"mode": _cudagraph_mode_name(desc.cg_mode), "capacity": int(desc.num_tokens)}
        return result

    def accepted(self, request_ids, lengths, num_sampled, producer_epochs):
        result = self.original_accepted(request_ids, lengths, num_sampled, producer_epochs)
        record = self.active
        if record is None or record["accepted"] is not None:
            raise ValueError("Acceptance without a unique current execution")
        # Same consuming stream, before reusable counts can change. No host wait.
        counts = num_sampled.detach().clone()
        self.sampled.append((record["execution"], counts))
        record["accepted"] = {
            "request_ids": list(request_ids),
            "verified": list(lengths),
            "producer_epochs": list(producer_epochs),
            "num_sampled": None,
            "status": "device_snapshot_pending_phase_boundary",
        }
        self.persist(record, "verification_returned")
        return result

    def snapshot(self):
        import torch

        result = super().snapshot()
        # One compact counts D2H at the quiescent final RPC, in addition to the
        # existing aggregate snapshot. No tensor is read after its buffer reuse.
        if self.sampled:
            values = torch.cat([v for _, v in self.sampled]).cpu().tolist()
            offset = 0
            for execution, tensor in self.sampled:
                row = self.records[execution - 1]["accepted"]
                size = tensor.numel()
                row.update(num_sampled=values[offset : offset + size], status="available")
                offset += size
        result["confidence_execution_receipts"] = {
            "records": copy.deepcopy(self.records),
            "truncated": False,
            "max_executions": MAX_EXECUTIONS,
            "max_request_rows": self.max_request_rows,
            "max_log_bytes": self.max_log_bytes,
            "max_requests": self.max_requests,
            "sampled_counts": sum(t.numel() for _, t in self.sampled),
            "device_snapshot_bytes": sum(t.numel() * t.element_size() for _, t in self.sampled),
            "log_bytes": self.log_bytes,
            "execution_basis": (
                "real nonempty target calls after observer installation; excludes capture and no-op retirement"
            ),
            "coverage": "all observed calls; phase-boundary counts; built-in numerical checks only",
        }
        return result

    def close(self):
        if self.runner is None:
            return
        for name, wrapped, original, local in (
            ("select", self.select, self.original_select, self.select_local),
            ("accepted", self.accepted, self.original_accepted, self.accepted_local),
        ):
            if getattr(self.adaptive, name) == wrapped:
                if local:
                    setattr(self.adaptive, name, original)
                else:
                    delattr(self.adaptive, name)
        self.sampled.clear()
        self.adaptive = self.original_select = self.original_accepted = self.active = None
        super().close()
