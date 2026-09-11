# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-capture, instance-local observations for isolated profile experiments.

Metadata mode never reads tensor values, launches device work or writes per step.
Host arrays are copied to Python values, never retained as views. Device fields
are descriptors ONLY: they cannot establish numerical finiteness or KV contents.
"""

import hashlib
import json
import os
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from functools import wraps
from itertools import islice
from pathlib import Path

import numpy as np
import torch

RING_RECORDS = 128
MAX_FIELDS = 128
MAX_DESCRIPTOR_NODES = 512
EPOCH_FIELDS = (
    "_proposal_step_epoch",
    "_prepared_step_epoch",
    "_context_kv_step_epoch",
    "_draft_forward_step_epoch",
    "_markov_attempt_step_epoch",
    "_markov_step_epoch",
    "_published_proposal_step_epoch",
    "_proposal_consumer_step_epoch",
)
BATCH_HOST_FIELDS = (
    "idx_mapping_np",
    "query_start_loc_np",
    "num_scheduled_tokens",
    "num_computed_tokens_np",
    "seq_lens_np",
    "is_prefilling_np",
    "num_draft_tokens_per_req",
    "cu_num_logits_np",
)
BATCH_TENSOR_FIELDS = (
    "idx_mapping",
    "query_start_loc",
    "seq_lens",
    "positions",
    "input_ids",
    "logits_indices",
    "cu_num_logits",
    "expanded_idx_mapping",
    "expanded_local_pos",
)


def describe(value, depth=0, budget=None, seen=None):
    """Host descriptors with a total node bound, not just a per-container limit."""
    if budget is None:
        budget, seen = [MAX_DESCRIPTOR_NODES], set()
    if budget[0] <= 0:
        return {"fields": "node_limit"}
    budget[0] -= 1
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "object_id": id(value),
            "data_ptr": value.data_ptr(),
            "storage_data_ptr": value.untyped_storage().data_ptr(),
            "storage_offset": value.storage_offset(),
            "values": "unavailable",
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.tolist()
    if depth >= 4:
        return {"type": type(value).__name__, "fields": "depth_limit"}
    if id(value) in seen:
        return {"object_id": id(value), "fields": "already_described"}
    seen.add(id(value))
    if isinstance(value, Mapping):
        count, items = len(value), value.items()
    elif is_dataclass(value) and not isinstance(value, type):
        names = fields(value)
        count, items = len(names), ((field.name, getattr(value, field.name)) for field in names)
    elif isinstance(value, (list, tuple)):
        result = []
        for entry in value[:MAX_FIELDS]:
            if budget[0] <= 0:
                break
            result.append(describe(entry, depth + 1, budget, seen))
        if len(result) < len(value):
            result.append({"truncated_fields": len(value) - len(result)})
        return result
    else:
        return {"type": type(value).__name__}
    result = {"object_id": id(value)}
    used = 0
    for key, entry in islice(items, MAX_FIELDS):
        if budget[0] <= 0:
            break
        result[str(key)] = describe(entry, depth + 1, budget, seen)
        used += 1
    if used < count:
        result["truncated_fields"] = count - used
    return result


class ProfileObservation:
    def __init__(self, runner, options):
        adaptive = runner.speculator.confidence_verification
        additional = runner.vllm_config.additional_config
        if (
            not adaptive.options.get("profile")
            or adaptive.options.get("mode") != "specified_lengths"
            or additional.get("dspark_nan_diagnostic_dir")
            or additional.get("dspark_profile_nan_diagnostic_dir")
        ):
            raise ValueError(
                "Profile observation requires an isolated specified-length engine without full diagnostics"
            )
        self.mode = options["mode"]
        if self.mode not in ("metadata-only", "context-kv-sync"):
            raise ValueError("Unsupported profile observation mode")
        self.runner = runner
        self.directory = Path(options["directory"])
        self.directory.mkdir(parents=True, exist_ok=True)  # installation only
        if any(self.directory.glob(f"rank-{runner.speculator.rank}-*.json")):
            raise ValueError("Profile observation requires a fresh worker output directory")
        self.rank = int(runner.speculator.rank)
        self.records = deque(maxlen=RING_RECORDS)
        self.point = None
        self.execution = 0
        self.batch_current = False
        self.sequence = 0
        self.counts = Counter()
        self.sync_calls = 0
        self.sync_completed = 0
        self.sync_stream = None
        self.first_failure = False
        self.hooks = []
        self.recording_error = None
        spec = runner.speculator
        try:
            if self.mode == "metadata-only":
                self.wrap(runner, "execute_model", "target_execute")
                self.wrap(runner.cudagraph_manager, "run_fullgraph", "target_full")
                self.wrap(spec, "prepare_proposal_inputs", "proposal_prepare")
                self.wrap(spec, "propose", "proposal_publish")
                self.wrap(spec, "_run_draft_model_forward", "draft_forward")
                self.wrap(spec, "_execute_sequential_markov_sampling", "markov")
                self.wrap(spec.model, "combine_hidden_states", "combined_context")
                self.wrap(spec.model, "precompute_and_store_context_kv", "context_kv")
                self.wrap(spec.model, "compute_draft_logits", "base_logits")
            else:
                # One changed device boundary, no metadata-only hooks/checks.
                self.wrap(spec.model, "precompute_and_store_context_kv", "context_kv")
        except BaseException:
            self.close()
            raise

    def wrap(self, obj, name, stage):
        original = getattr(obj, name)
        had_local = name in vars(obj)
        local = vars(obj).get(name)

        @wraps(original)
        def observed(*args, **kwargs):
            if stage == "target_execute":
                self.execution += 1
                self.batch_current = False
            elif stage == "target_full":
                self.batch_current = True
            self.record(stage + ".enter", args=args, kwargs=kwargs)
            try:
                result = original(*args, **kwargs)
                if stage == "target_execute":
                    self.batch_current = getattr(self.runner, "execute_model_state", None) is not None
                if self.mode == "context-kv-sync":
                    # Only diagnostic. The producer issues context projection,
                    # RoPE and scatter on the caller stream; draft reads follow.
                    stream = torch.npu.current_stream()
                    self.sync_stream = str(stream.npu_stream)
                    self.sync_calls += 1
                    stream.synchronize()
                    self.sync_completed += 1
                self.record(stage + ".return", result=result)
                return result
            except BaseException as error:
                self.failed(stage, error)
                raise

        setattr(obj, name, observed)
        self.hooks.append((obj, name, original, observed, had_local, local))

    def batch(self):
        batch = getattr(self.runner, "input_batch", None)
        if batch is None or not self.batch_current:
            return None
        n = int(batch.num_reqs)
        result = {
            "request_ids": list(batch.req_ids),
            "num_reqs": n,
            "num_tokens": int(batch.num_tokens),
            "token_capacity": int(batch.num_tokens_after_padding),
            "request_capacity": int(batch.num_reqs_after_padding),
        }
        result["host_buffers"] = {}
        for field in BATCH_HOST_FIELDS:
            array = getattr(batch, field, None)
            if array is None:
                continue
            if not isinstance(array, np.ndarray):
                raise TypeError(f"Expected existing CPU numpy metadata for {field}")
            end = n + 1 if field in ("query_start_loc_np", "cu_num_logits_np") else n
            result[field] = array[:end].tolist()  # independent Python scalars
            result["host_buffers"][field] = {
                "shape": list(array.shape),
                "strides_bytes": list(array.strides),
                "data_ptr": int(array.__array_interface__["data"][0]),
            }
        result["buffers"] = {name: describe(getattr(batch, name, None)) for name in BATCH_TENSOR_FIELDS}
        return result

    def record(self, stage, **payload):
        if self.mode != "metadata-only":
            return
        try:
            spec = self.runner.speculator
            self.sequence += 1
            self.counts[stage] += 1
            record = {
                "sequence": self.sequence,
                "execution": self.execution,
                "point": self.point,
                "stage": stage,
                "epochs": {f: getattr(spec, f, None) for f in EPOCH_FIELDS},
                "payload": describe(payload),
            }
            # execute entry precedes preparation: old InputBatch is NOT this step.
            if stage != "target_execute.enter":
                record["batch"] = self.batch()
                record["batch_current"] = self.batch_current
            else:
                scheduled = payload["args"][0] if payload.get("args") else payload["kwargs"]["scheduler_output"]
                record["scheduler"] = {
                    "scheduled": dict(scheduled.num_scheduled_tokens),
                    "finished": sorted(scheduled.finished_req_ids),
                    "preempted": sorted(getattr(scheduled, "preempted_req_ids", None) or []),
                }
            record["owners"] = describe(getattr(spec, "_published_proposal_owners", {}))
            record["selection"] = describe(getattr(spec.confidence_verification, "last_selection", None))
            if stage in ("target_full.enter", "proposal_prepare.return", "context_kv.return"):
                model_state = getattr(self.runner, "model_state", None)
                record["attention"] = describe(getattr(model_state, "attn_metadata", None))
                manager = self.runner.cudagraph_manager
                record["captured_outputs"] = {
                    name: describe(getattr(manager, name, None)) for name in ("hidden_states", "aux_hidden_states")
                }
                record["draft_kv"] = describe(getattr(spec, "draft_kv_caches", None))
                tables = getattr(self.runner, "block_tables", None)
                record["block_tables"] = describe(getattr(tables, "input_block_tables", None))
                record["kv_group_block_sizes"] = describe(getattr(tables, "kernel_block_sizes", None))
                inputs = getattr(self.runner, "input_buffers", None)
                record["persistent_inputs"] = {
                    name: describe(getattr(inputs, name, None)) for name in BATCH_TENSOR_FIELDS
                }
                record["cpu_staging_buffers"] = {
                    # Descriptors ONLY; the D2H destination may still be in flight.
                    name: describe(getattr(self.runner, name, None))
                    for name in ("num_computed_tokens_cpu",)
                }
            state = getattr(self.runner, "execute_model_state", None)
            if stage == "target_execute.return" and state is not None:
                record["target_outputs"] = {
                    name: describe(getattr(state, name, None))
                    for name in ("hidden_states", "aux_hidden_states", "slot_mappings_by_layer")
                }
            # Device values (including sampled/rejected counts and KV pages)
            # stay unavailable. Descriptors expose alias/row/span relationships.
            self.records.append(record)
        except Exception as error:
            # Observer failure never substitutes for a numerical/runtime error.
            self.recording_error = f"{type(error).__name__}: {error}"

    def snapshot(self):
        return {
            "schema_version": 2,
            "pid": os.getpid(),
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "descriptor_node_limit": MAX_DESCRIPTOR_NODES,
            "performance_eligible": False,
            "mode": self.mode,
            "rank": self.rank,
            "point": self.point,
            "root_cause": "ROOT_CAUSE_NOT_YET_PROVEN",
            "recording_error": self.recording_error,
            "ring_capacity": RING_RECORDS,
            "stage_counts": dict(self.counts),
            "records": list(self.records),
            "device_values": "unavailable; no numeric checks or device copies",
            "sync": {
                "boundary": "precompute_and_store_context_kv return / before draft forward",
                "object": "torch.npu.current_stream()",
                "stream_handle": self.sync_stream,
                "calls": self.sync_calls,
                "completed": self.sync_completed,
            },
        }

    def failed(self, stage, error):
        if self.first_failure:
            return
        self.first_failure = True
        try:
            self.record(stage + ".error")
            data = self.snapshot()
            data.update(failure={"stage": stage, "type": type(error).__name__, "message": str(error)})
            self.write("first-failure", data, exclusive=True)
        except Exception:
            # Preserve the original exception even if storage is unavailable.
            pass

    def write(self, suffix, data, exclusive=False):
        path = self.directory / f"rank-{self.rank}-{suffix}.json"
        encoded = json.dumps(data, allow_nan=False).encode("utf-8")
        if exclusive:
            with path.open("xb") as stream:
                stream.write(encoded)
        else:
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            temporary.replace(path)
        return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)}

    def finish_point(self):
        # Full history is worker-local. Do not fan it out through two RPC
        # serializers (8 ranks previously returned ~122 MB per point).
        data = self.snapshot()
        receipt = self.write("latest", data)
        if self.recording_error is not None:
            raise RuntimeError(f"Profile metadata evidence unavailable: {self.recording_error}")
        return {key: value for key, value in data.items() if key != "records"} | {
            "records_count": len(data["records"]),
            "local_evidence": receipt,
            "records": "worker-local; latest file is replaced at the next point boundary",
        }

    def begin_point(self, point):
        self.point = point
        self.batch_current = False
        self.records.clear()
        self.counts.clear()
        self.sync_calls = self.sync_completed = 0
        self.sync_stream = None

    def close(self):
        for obj, name, original, observed, had_local, local in reversed(self.hooks):
            if getattr(obj, name) is observed:
                if had_local:
                    setattr(obj, name, local)
                else:
                    delattr(obj, name)
        self.hooks.clear()
        self.records.clear()
        self.point = None
