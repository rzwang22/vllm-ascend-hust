# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded target graph reductions; installed before compile, default off."""

from collections import Counter

import torch

from vllm_ascend.diagnostics.dspark_profile_attention import ATTENTION_STAGES, STATE_COLUMNS, install_attention_probe
from vllm_ascend.diagnostics.dspark_profile_auxiliary import AuxiliaryCapture, AuxiliaryProfileObservation
from vllm_ascend.diagnostics.dspark_profile_observation import EPOCH_FIELDS, describe

MAX_TARGET_TOKENS = 384
MAX_TARGET_BOUNDARIES = 24
EARLY_LAYER_COUNT = 4
LAYER_CHECKPOINT_INTERVAL = 10
MAX_ERROR_EVENTS = 4
MAX_METADATA_TENSORS = 32
VALIDITY_ROUNDS = 3
DETAILED_BOUNDARIES = ("attn_input", "attn_output", "residual", "ffn_input", "ffn_output", "output")


class TargetBoundaryFlags:
    """One reusable bank of row flags, never full hidden-state snapshots.

    Every selected write/reduction and receipt copy is compiled and captured.
    The replay observer owns copies of the compact results before the next
    replay can overwrite this bank. Profile calls have no execution receipt.
    """

    def __init__(
        self,
        *,
        sizes,
        auxiliary_layers,
        start_layer,
        end_layer,
        hidden_size,
        hc_mult,
        device,
        target_layer=None,
        attention=False,
    ):
        if (
            not sizes
            or min(sizes) <= 0
            or max(sizes) > MAX_TARGET_TOKENS
            or start_layer != 0
            or not auxiliary_layers
            or any(type(i) is not int or not start_layer <= i < end_layer for i in auxiliary_layers)
        ):
            raise ValueError("Target boundaries require PP1, valid auxiliary layers and capture sizes within 384")
        if target_layer is not None and (type(target_layer) is not int or not start_layer <= target_layer < end_layer):
            raise ValueError("Target detail layer must be a zero-based decoder index within the target model")
        self.target_layer = target_layer
        if attention and target_layer is None:
            raise ValueError("Attention detail requires an explicit target layer")
        self.attention_probe = None
        # Keep the original broad plan when no detail layer is requested.
        # A local plan replaces its distant cuts, while AuxiliaryCapture still
        # observes all configured raw/persistent/consumed auxiliary outputs.
        anchor = min(auxiliary_layers) if target_layer is None else target_layer
        checkpoints = {anchor - 1} if anchor else set()
        if target_layer is None:
            checkpoints.update(range(min(EARLY_LAYER_COUNT, anchor)))
            checkpoints.update(range(LAYER_CHECKPOINT_INTERVAL - 1, anchor, LAYER_CHECKPOINT_INTERVAL))
        self.observe_layer_input = target_layer is not None
        self.layers = tuple(sorted(checkpoints | {anchor}))
        self.names = (
            ("embedding",)
            + tuple(f"layer.{i}.output" for i in sorted(checkpoints))
            + ((f"layer.{anchor}.input",) if self.observe_layer_input else ())
            + tuple(f"layer.{anchor}.{stage}" for stage in DETAILED_BOUNDARIES)
        )
        self.outer_names = self.names
        if attention:
            self.names += tuple(f"layer.{anchor}.attention.{stage}" for stage in ATTENTION_STAGES)
        if len(self.names) > MAX_TARGET_BOUNDARIES:
            raise ValueError("Target diagnostic boundary budget exceeded")
        self.max_tokens = max(sizes)
        self.tails = {
            name: [hc_mult, hidden_size] if name.endswith((".input", ".residual", ".output")) else [hidden_size]
            for name in self.names
        }
        self.flags = torch.empty((len(self.outer_names), self.max_tokens, 2), dtype=torch.bool, device=device)
        self.receipts = torch.zeros((len(self.outer_names), 1), dtype=torch.int64, device=device)
        # Opaque DSA writes must not share storage with AOT-functionalized
        # outer writes: their copy-back would overwrite undeclared side effects.
        count = len(self.names) - len(self.outer_names)
        self.attention_flags = (
            torch.empty((count, self.max_tokens, 2), dtype=torch.bool, device=device) if count else None
        )
        self.attention_receipts = torch.zeros((count, 1), dtype=torch.int64, device=device) if count else None
        self.epoch_input = torch.zeros(1, dtype=torch.int64, device=device)

    def write(self, name, value):
        if name not in self.names:
            return  # compile-time constant: unselected stages add no device work
        index = self.names.index(name)
        flags, receipts = self.flags, self.receipts
        if index >= len(self.outer_names):
            index -= len(self.outer_names)
            flags, receipts = self.attention_flags, self.attention_receipts
        # Frozen Core bypasses Dynamo guards. Keep the symbolic minimum even
        # when the first profile/compile uses 8192 rows and capture uses <=384.
        # Dynamo 2.10 folds builtin min for static ints and emits sym_min
        # for SymInt, preserving the runtime bound without an int conversion.
        n = min(value.shape[0], self.max_tokens)
        flat = value[:n].flatten(1)
        flags[index, :n].copy_(torch.stack((torch.isnan(flat).any(1), torch.isinf(flat).any(1)), dim=1))
        receipts[index].copy_(self.epoch_input)

    @property
    def allocated_bytes(self):
        values = (self.flags, self.receipts, self.epoch_input)
        if self.attention_flags is not None:
            values += (self.attention_flags, self.attention_receipts)
        if self.attention_probe is not None:
            values += (self.attention_probe.state, *self.attention_probe.kv.tensors)
        return sum(x.numel() * x.element_size() for x in values)


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
        target_layer=additional["dspark_profile_observation"].get("target_layer"),
        attention=additional["dspark_profile_observation"].get("attention", False),
    )
    model._dspark_layer_snapshots = bank
    for index in bank.layers:
        model.layers[index]._dspark_layer_snapshots = bank
    if additional["dspark_profile_observation"].get("attention", False):
        install_attention_probe(bank, model, bank.target_layer)


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
        self.attention_validity_rounds = []
        self.attention_validity_done = False
        super().__init__(runner, options)
        self.bank = self.capture.bank

    def before_replay(self, desc, graph):
        pending = super().before_replay(desc, graph)
        self.bank.receipts.fill_(-1)
        if self.bank.attention_receipts is not None:
            self.bank.attention_receipts.fill_(-1)
        if self.bank.attention_probe is not None:
            self.bank.attention_probe.kv.receipts.fill_(-1)
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
        flags, receipts = self.bank.flags, self.bank.receipts
        if self.bank.attention_flags is not None:
            # Merge only after replay, outside the model's compiled function.
            flags = torch.cat((flags, self.bank.attention_flags))
            receipts = torch.cat((receipts, self.bank.attention_receipts))
        self.integer("target_internal.flags", flags[:, : desc.num_tokens])
        self.integer("target_internal.receipts", receipts)
        if self.bank.attention_probe is not None:
            self.integer("target_internal.attention_state", self.bank.attention_probe.state[: desc.num_tokens])
            self.bank.attention_probe.kv.packet(self, desc.num_tokens)

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
        if self.bank.attention_probe is not None:
            fresh = fresh and (
                len(device.get("target_internal.attention_state", [])) == (capacity or 0) * len(STATE_COLUMNS)
                and capacity in self.bank.attention_probe.routes
                and self.bank.attention_probe.kv.fresh(device, capacity or 0, record["execution"])
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
            for boundary in (b for b in boundaries if b["name"] in self.bank.outer_names):
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
        if self.bank.attention_probe is not None:
            probe = self.bank.attention_probe
            state = device.get("target_internal.attention_state", [])
            attention_rows = []
            positions = device.get("target.positions", [])
            if fresh:
                for row, i in enumerate(range(0, len(state), len(STATE_COLUMNS))):
                    entry = dict(zip(STATE_COLUMNS, state[i : i + len(STATE_COLUMNS)]))
                    req = entry["request_row"]
                    entry["row"] = row
                    entry["request_id"] = record["request_ids"][req] if 0 <= req < len(record["request_ids"]) else None
                    entry["valid_target_row"] = row < record["target_rows"]
                    entry["position_matches_target"] = (
                        entry["position"] == positions[row] if row < len(positions) else None
                    )
                    entry["window_range_valid"] = (
                        entry["invalid_window_indices"] == 0 if entry["valid_target_row"] else None
                    )
                    attention_rows.append(entry)
            kv_detail = probe.kv.decode(device, attention_rows, capacity) if fresh else None
            if kv_detail is not None:
                name = kv_detail["binding"]["layer_name"].rsplit(".", 1)[0] + ".swa_cache"
                config = getattr(self.runner, "kv_cache_config", None)
                kv_detail["cache_group_catalog"] = [
                    {"group_id": index, "layer_name": name, "spec_type": type(group.kv_cache_spec).__qualname__}
                    for index, group in enumerate(getattr(config, "kv_cache_groups", ()))
                    if name in group.layer_names
                ]
                kv_detail["cache_group_status"] = "available" if kv_detail["cache_group_catalog"] else "unavailable"
                kv_detail["group_scope"] = "CPU config catalog; not a device page ownership proof"
            record["target_internal"]["attention"] = {
                "route": probe.routes.get(capacity),
                "route_scope": "capture/eager description keyed by capacity; not live replay layout",
                "kv": kv_detail,
                "state_columns": STATE_COLUMNS,
                "rows": attention_rows,
                "boundaries": [b for b in boundaries if b["name"] not in self.bank.outer_names],
                "coverage": record["target_internal"]["coverage"],
                "scope": "parallel Q/KV inputs and a sequential output path; not one causal boundary chain",
            }
        self.save_attention_validity(record, fresh)
        self.target_counts[record["target_internal"]["coverage"]] += 1
        nonfinite = any(v["nan"] or v["inf"] for b in boundaries for v in b["rows"] if v["row"] < record["target_rows"])
        self.target_counts["nonfinite_rounds"] += nonfinite
        if nonfinite and "nonfinite" not in self.target_latches:
            self.write("target-first-nonfinite", self.snapshot(), exclusive=True)
            self.target_latches.add("nonfinite")

    def save_attention_validity(self, record, fresh):
        if self.bank.attention_probe is None or self.attention_validity_done or record["graph_capacity"] is None:
            return
        # Consume the already transferred packet; no extra device read or RPC.
        good = fresh and record["coverage"] == "FULL" and not self.recording_error
        if good and not (record["consumption_reached"] and record["head_reached"]):
            return
        previous = self.attention_validity_rounds
        good = bool(good and (not previous or record["execution"] == previous[-1]["execution"] + 1))
        previous.append(
            {
                "execution": record["execution"],
                "proposal_epoch": record["proposal_epoch"],
                "valid": good,
                "target_receipts": record["device_integers"].get("target_internal.receipts"),
                "raw_receipts": record["device_integers"].get("raw_receipts"),
                "consume_receipts": record["device_integers"].get("consume_receipts"),
                "kv_receipts": record["device_integers"].get("kv.receipts"),
            }
        )
        status = "failed" if not good else "passed" if len(previous) == VALIDITY_ROUNDS else "pending"
        data = {
            "performance_eligible": False,
            "point": record["point"],
            "rank": self.rank,
            "status": status,
            "required_boundaries": list(self.bank.names),
            "kv_required": True,
            "rounds": list(previous),
            "error": (self.recording_error or "Incomplete/nonconsecutive FULL receipts") if not good else None,
            "snapshot": self.snapshot(),
            "buffers": {
                name: describe(getattr(self.bank, name))
                for name in ("flags", "receipts", "attention_flags", "attention_receipts", "epoch_input")
            },
        }
        # Atomic publication includes first failure evidence before frontend abort.
        self.write("attention-validity", data)
        self.attention_validity_done = status != "pending"

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
                "target_layer": self.capture.bank.target_layer,
                "attention_enabled": self.capture.bank.attention_probe is not None,
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
