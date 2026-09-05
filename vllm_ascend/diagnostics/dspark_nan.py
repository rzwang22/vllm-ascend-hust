# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Opt-in, capture-external first-failure diagnostics for DSpark Graph B4.

Never enabled by default. Reductions and small D2H copies below intentionally
synchronize diagnostic runs; those runs must not publish performance results.
No logits, hidden values, weights or KV contents are written to disk.
"""

import json
from collections import deque
from pathlib import Path
from typing import Any

import torch

_MAX_BAD_ROWS = 8


class DSparkNaNDiagnostics:
    def __init__(self, directory: str, rank: int) -> None:
        if not isinstance(directory, str) or not directory or not Path(directory).is_absolute():
            raise ValueError("dspark_nan_diagnostic_dir must be a non-empty absolute path.")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        if any(self.directory.glob(f"rank-{self.rank}-*.json")):
            raise ValueError("DSpark NaN diagnostics require a fresh rank output directory.")
        self.execution_epoch = 0
        self.current: dict[str, Any] = {}
        self.history: deque[dict[str, Any]] = deque(maxlen=2)
        self.failed = False
        self.phase = "not_started"

    @staticmethod
    def _outside_capture() -> None:
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_current_stream_capturing():
            raise RuntimeError("DSpark NaN diagnostics must run outside ACLGraph capture.")

    def _write(self, *, first_failure: bool = False) -> None:
        report = {
            "status": "ROOT_CAUSE_NOT_YET_PROVEN",
            "performance_eligible": False,
            "rank": self.rank,
            "previous_executions": list(self.history),
            "current": self.current,
        }
        text = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        path = self.directory / f"rank-{self.rank}-latest.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text)
        temporary.replace(path)
        if first_failure and not self.failed:
            path = self.directory / f"rank-{self.rank}-first-failure.json"
            with path.open("x") as output:
                output.write(text)
            self.failed = True

    def begin_execution(self, scheduler_output: Any) -> None:
        self._outside_capture()
        if self.current:
            self.history.append(self.current)
        self.execution_epoch += 1
        self.current = {
            "target_execution_epoch": self.execution_epoch,
            "phase": self.phase,
            "scheduler_request_ids": list(scheduler_output.num_scheduled_tokens),
            "scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
            "scheduled_total_tokens": int(scheduler_output.total_num_scheduled_tokens),
            "stage": "target_execute_started",
            "checks": [],
        }
        self._write()

    def failed_execution(self, boundary: str, error: BaseException) -> None:
        # Keep the earliest boundary's file even when an outer wrapper catches
        # the same error. The latest file can include propagation information.
        self.current["exception_boundary"] = boundary
        self.current["exception"] = f"{type(error).__name__}: {error}"
        self._write(first_failure=True)

    @staticmethod
    def _stats(tensor: torch.Tensor, valid_rows: int, allow_negative_infinity: bool) -> dict[str, Any]:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or not 0 <= valid_rows <= tensor.shape[0]:
            raise ValueError("Diagnostic tensor does not cover the current valid row range.")
        value = tensor[:valid_rows].detach()
        result = {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "data_ptr": tensor.data_ptr(),
            "valid_row_range": [0, valid_rows],
            "allow_negative_infinity": allow_negative_infinity,
        }
        if not value.dtype.is_floating_point:
            raise TypeError("NaN boundary diagnostics require a floating-point tensor.")
        if valid_rows and value.numel():
            flat = value.reshape(valid_rows, -1)
            # Copy only three flags per row, never tensor contents.
            flags = (
                torch.stack((torch.isnan(flat).any(1), torch.isposinf(flat).any(1), torch.isneginf(flat).any(1)), dim=1)
                .cpu()
                .tolist()
            )
        else:
            flags = []
        for column, name in enumerate(("nan", "positive_inf", "negative_inf")):
            rows = [index for index, flags_for_row in enumerate(flags) if flags_for_row[column]]
            result[f"{name}_row_count"] = len(rows)
            result[f"{name}_rows_first"] = rows[:_MAX_BAD_ROWS]
        result["invalid"] = bool(
            result["nan_row_count"]
            or result["positive_inf_row_count"]
            or (result["negative_inf_row_count"] and not allow_negative_infinity)
        )
        return result

    def check(
        self, boundary: str, tensors: dict[str, torch.Tensor], valid_rows: int, *, allow_negative_infinity: bool = False
    ) -> None:
        self._outside_capture()
        self.current["stage"] = boundary
        checks = {name: self._stats(value, valid_rows, allow_negative_infinity) for name, value in tensors.items()}
        self.current["checks"].append({"boundary": boundary, "tensors": checks})
        invalid = [name for name, stats in checks.items() if stats["invalid"]]
        self._write(first_failure=bool(invalid))
        if invalid:
            raise RuntimeError(f"DSpark NaN diagnostic detected non-finite values at {boundary}: {invalid}.")

    def target_completed(self, runner: Any, descriptors: list[Any]) -> None:
        self._outside_capture()
        state = runner.execute_model_state
        if state is None:  # A zero-token execute is not a target forward.
            self.current["stage"] = "no_target_forward"
            self._write()
            return
        batch = state.input_batch
        self.current.update(
            request_ids=list(batch.req_ids),
            target_unpadded_tokens=int(batch.num_tokens),
            target_padded_tokens=int(batch.num_tokens_after_padding),
            actual_full_shapes=[int(desc.num_tokens) for desc in descriptors],
            target_runtime="FULL" if descriptors else "non_FULL",
            query_offsets=[int(value) for value in batch.query_start_loc_np[: batch.num_reqs + 1]],
            # These are small integer diagnostics. The host upper bound alone
            # cannot identify the actual async request state.
            sequence_lengths=batch.seq_lens[: batch.num_reqs].detach().cpu().tolist(),
            positions=batch.positions[: batch.num_tokens].detach().cpu().tolist(),
        )
        tensors = {"target_hidden": state.hidden_states}
        for index, tensor in enumerate(state.aux_hidden_states or []):
            tensors[f"target_aux_{index}"] = tensor
        self.check("target_outputs", tensors, batch.num_tokens)

    def proposal_inputs(self, proposal: Any) -> None:
        self._outside_capture()
        self.current.update(
            proposal_epoch=int(proposal.step_epoch),
            proposal_request_ids=list(proposal.request_ids),
            draft_query_tokens=int(proposal.num_query_tokens),
            draft_query_offsets=proposal.draft_query_start_loc.detach().cpu().tolist(),
            draft_positions=proposal.draft_positions.detach().cpu().tolist(),
            draft_sequence_lengths=proposal.draft_sequence_lengths.detach().cpu().tolist(),
            draft_layer_group_ids=dict(proposal.draft_layer_group_ids),
        )
        tensors = {"proposal_last_hidden": proposal.last_hidden_states}
        tensors.update({f"proposal_aux_{i}": tensor for i, tensor in enumerate(proposal.auxiliary_hidden_states)})
        self.check("proposal_inputs", tensors, proposal.num_target_tokens)

    def context_kv(self, proposal: Any, caches: Any, block_sizes: list[int]) -> None:
        self._outside_capture()
        # Inspect precisely the context rows just written, including every valid
        # token, using the actual draft layer's physical block size.
        for layer_name, cache in caches.items():
            while isinstance(cache, (list, tuple)) and len(cache) == 1:
                cache = cache[0]
            if not isinstance(cache, torch.Tensor) or cache.ndim != 4:
                raise ValueError(f"Unsupported diagnostic KV layout for {layer_name}; expected paged rank 4.")
            group = proposal.draft_layer_group_ids[layer_name]
            block_size = block_sizes[group]
            if cache.shape[1] != block_size:
                raise ValueError(f"Diagnostic KV block size disagrees for {layer_name}.")
            slots = proposal.draft_context_slot_mappings[layer_name][: proposal.num_target_tokens]
            host_slots = slots.detach().cpu().tolist()
            if any(slot < -1 or slot >= cache.shape[0] * block_size for slot in host_slots):
                raise ValueError(f"Invalid diagnostic context slot for {layer_name}.")
            valid = slots != -1  # Exact R4 sentinel, not arbitrary negative clamping.
            selected = slots[valid].long()
            rows = cache[selected // block_size, selected % block_size]
            self.current.setdefault("context_kv", {})[layer_name] = {
                "block_size": block_size,
                "cache_shape": list(cache.shape),
                "cache_stride": list(cache.stride()),
                "cache_data_ptr": cache.data_ptr(),
                "context_slots": host_slots,
                "valid_target_rows": [i for i, slot in enumerate(host_slots) if slot != -1],
            }
            self.check(f"context_kv_written:{layer_name}", {"written_kv_rows": rows}, rows.shape[0])
