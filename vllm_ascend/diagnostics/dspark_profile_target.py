# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded target graph reductions; installed before compile, default off."""

from collections import Counter

import torch

from vllm_ascend.diagnostics.dspark_profile_auxiliary import AuxiliaryCapture, AuxiliaryProfileObservation
from vllm_ascend.diagnostics.dspark_profile_observation import EPOCH_FIELDS

MAX_TARGET_TOKENS = 384
MAX_TARGET_BOUNDARIES = 24
EARLY_LAYER_COUNT = 4
LAYER_CHECKPOINT_INTERVAL = 10
MAX_ERROR_EVENTS = 4
MAX_METADATA_TENSORS = 32
DETAILED_BOUNDARIES = ("attn_input", "attn_output", "residual", "ffn_input", "ffn_output", "output")


class TargetBoundaryFlags:
    """One reusable bank of row flags, never full hidden-state snapshots.

    Every selected write/reduction and receipt copy is compiled and captured.
    The replay observer owns copies of the compact results before the next
    replay can overwrite this bank. Profile calls have no execution receipt.
    """

    def __init__(self, *, sizes, auxiliary_layers, start_layer, end_layer, hidden_size, hc_mult, device):
        if (
            not sizes
            or min(sizes) <= 0
            or max(sizes) > MAX_TARGET_TOKENS
            or start_layer != 0
            or not auxiliary_layers
            or any(type(i) is not int or not start_layer <= i < end_layer for i in auxiliary_layers)
        ):
            raise ValueError("Target boundaries require PP1, valid auxiliary layers and capture sizes within 384")
        anchor = min(auxiliary_layers)
        checkpoints = set(range(min(EARLY_LAYER_COUNT, anchor)))
        checkpoints.update(range(LAYER_CHECKPOINT_INTERVAL - 1, anchor, LAYER_CHECKPOINT_INTERVAL))
        if anchor:
            checkpoints.add(anchor - 1)
        self.layers = tuple(sorted(checkpoints | {anchor}))
        self.names = (
            ("embedding",)
            + tuple(f"layer.{i}.output" for i in sorted(checkpoints))
            + tuple(f"layer.{anchor}.{stage}" for stage in DETAILED_BOUNDARIES)
        )
        if len(self.names) > MAX_TARGET_BOUNDARIES:
            raise ValueError("Target diagnostic boundary budget exceeded")
        self.max_tokens = max(sizes)
        self.tails = {
            name: [hc_mult, hidden_size] if name.endswith((".residual", ".output")) else [hidden_size]
            for name in self.names
        }
        self.flags = torch.empty((len(self.names), self.max_tokens, 2), dtype=torch.bool, device=device)
        self.receipts = torch.zeros((len(self.names), 1), dtype=torch.int64, device=device)
        self.epoch_input = torch.zeros(1, dtype=torch.int64, device=device)

    def write(self, name, value):
        if name not in self.names:
            return  # compile-time constant: unselected stages add no device work
        index = self.names.index(name)
        # Frozen Core bypasses Dynamo guards. Keep the symbolic minimum even
        # when the first profile/compile uses 8192 rows and capture uses <=384.
        n = torch.sym_min(value.shape[0], self.max_tokens)
        flat = value[:n].flatten(1)
        self.flags[index, :n].copy_(torch.stack((torch.isnan(flat).any(1), torch.isinf(flat).any(1)), dim=1))
        self.receipts[index].copy_(self.epoch_input)

    @property
    def allocated_bytes(self):
        return sum(x.numel() * x.element_size() for x in (self.flags, self.receipts, self.epoch_input))


def install_target_boundaries(model, config):
    """Called at target construction, before its first compile/profile."""
    additional = config.additional_config or {}
    if (additional.get("dspark_profile_observation") or {}).get("mode") != "target-boundaries":
        return
    verification = additional.get("dspark_confidence_verification") or {}
    if (
        not verification.get("profile")
        or verification.get("mode") != "specified_lengths"
        or additional.get("dspark_nan_replay_window") is not None
        or config.parallel_config.pipeline_parallel_size != 1
    ):
        raise ValueError("Target boundaries require isolated specified-length profiling and PP1")
    draft = config.speculative_config.draft_model_config.hf_config
    bank = TargetBoundaryFlags(
        sizes=sorted(config.compilation_config.cudagraph_capture_sizes),
        auxiliary_layers=getattr(draft, "dspark_target_layer_ids", ()),
        start_layer=model.start_layer,
        end_layer=model.end_layer,
        hidden_size=model.config.hidden_size,
        hc_mult=model.hc_mult,
        device=model.device,
    )
    model._dspark_layer_snapshots = bank
    for index in bank.layers:
        model.layers[index]._dspark_layer_snapshots = bank


class TargetCapture(AuxiliaryCapture):
    def __init__(self, manager, bank):
        super().__init__(manager)
        if not isinstance(bank, TargetBoundaryFlags):
            raise ValueError("Target flags must be installed before target compilation")
        self.bank = bank


class TargetProfileObservation(AuxiliaryProfileObservation):
    def __init__(self, runner, options):
        self.target_counts = Counter()
        self.target_latches = set()
        self.error_events = []
        super().__init__(runner, options)
        self.bank = self.capture.bank

    def before_replay(self, desc, graph):
        pending = super().before_replay(desc, graph)
        self.bank.receipts.fill_(-1)
        self.bank.epoch_input.fill_(self.execution)
        # These are the actual captured DSA fields; the previous experiment
        # observed `positions`, while DSA decode names it `input_positions`.
        seen = {}
        fields = {}
        for layer, metadata in self.capture.captured[desc].attn_metadata.items():
            for name in ("input_positions", "start_pos"):
                value = getattr(getattr(metadata, "decode", None), name, None)
                if not isinstance(value, torch.Tensor):
                    continue
                if id(value) not in seen:
                    if len(seen) >= MAX_METADATA_TENSORS or value.numel() > self.bank.max_tokens:
                        raise ValueError("Target position/start metadata budget exceeded")
                    key = f"target_internal.{layer}.{name}"
                    self.integer(key, value)
                    seen[id(value)] = key
                fields[f"{layer}.{name}"] = seen[id(value)]
        pending["target_metadata_fields"] = fields
        return pending

    def after_replay(self, desc):
        super().after_replay(desc)
        # copy=True owns flags/receipts now; later draft/capture/profile writes
        # cannot change this execution's packet. No host wait at this boundary.
        self.integer("target_internal.flags", self.bank.flags[:, : desc.num_tokens])
        self.integer("target_internal.receipts", self.bank.receipts)

    def complete_record(self, record, pending):
        device = record["device_integers"]
        capacity = record["graph_capacity"]
        flags = device.get("target_internal.flags", [])
        receipts = device.get("target_internal.receipts", [])
        fresh = (
            record["raw_replay_verified"]
            and receipts == [record["execution"]] * len(self.bank.names)
            and len(flags) == len(self.bank.names) * (capacity or 0) * 2
        )
        if capacity is not None and not fresh:
            self.recording_error = "Target internal flags missing/stale; no complete replay receipt"
        boundaries = []
        if fresh:
            for index, name in enumerate(self.bank.names):
                values = flags[index * capacity * 2 : (index + 1) * capacity * 2]
                boundaries.append(
                    {
                        "name": name,
                        "shape": [capacity, *self.bank.tails[name]],
                        "rows": self.map_rows(record, values, 2),
                    }
                )
        brackets = []
        for row in range(record["target_rows"] if fresh else 0):
            last = None
            for boundary in boundaries:
                value = boundary["rows"][row]
                if value["nan"] or value["inf"]:
                    brackets.append(
                        value | {"last_observed_finite": last, "first_observed_nonfinite": boundary["name"]}
                    )
                    break
                last = boundary["name"]
        record["target_internal"] = {
            "coverage": "FULL" if fresh else "INVALID_RECEIPT" if capacity is not None else "NON_FULL_UNOBSERVED",
            "boundaries": boundaries,
            "valid_row_brackets": brackets,
            "metadata_fields": pending.get("target_metadata_fields"),
            "root_cause": "UNKNOWN",
        }
        self.target_counts[record["target_internal"]["coverage"]] += 1
        self.target_counts["nonfinite_rounds"] += bool(brackets)
        if brackets and "nonfinite" not in self.target_latches:
            self.write("target-first-nonfinite", self.snapshot(), exclusive=True)
            self.target_latches.add("nonfinite")

    def failed(self, stage, error):
        super().failed(stage, error)  # preserve the original, exclusively saved numerical failure
        self.guard(self.save_error_event, stage, error)

    def save_error_event(self, stage, error):
        key = (self.execution, type(error).__name__, str(error))
        if len(self.error_events) >= MAX_ERROR_EVENTS or any(tuple(x["key"]) == key for x in self.error_events):
            return
        spec = self.runner.speculator
        self.error_events.append(
            {
                "key": key,
                "point": self.point,
                "rank": self.rank,
                "stage": stage,
                "execution": self.execution,
                "epochs": {name: getattr(spec, name, None) for name in EPOCH_FIELDS},
                "owner_epochs": {key: value.producer_epoch for key, value in spec._published_proposal_owners.items()},
                "scheduler": self.last_scheduler,
            }
        )
        self.guard(self.write, "error-events", {"performance_eligible": False, "events": self.error_events})

    def snapshot(self):
        data = super().snapshot()
        return data | {
            "target_internal": {
                "enabled": True,
                "boundary_order": self.capture.bank.names,
                "bank_bytes": self.capture.bank.allocated_bytes,
                "counts": dict(self.target_counts),
                "root_cause": "UNKNOWN",
                "scope": (
                    "selected target graph boundaries; unobserved layers and attention/KV internals remain grouped"
                ),
            }
        }

    def begin_point(self, point):
        super().begin_point(point)
        self.target_counts.clear()
