# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager draft observations; no target hooks or KV cache readback."""

from collections import Counter, deque
from functools import partial, wraps

import torch

from vllm_ascend.diagnostics.dspark_profile_observation import NUMERIC_ROUNDS, ProfileObservation


class UpstreamProfileObservation(ProfileObservation):
    def __init__(self, runner, options):
        self.pending = None
        self.upstream_records = deque(maxlen=NUMERIC_ROUNDS)
        self.upstream_counts = Counter()
        self.upstream_first_nan = False
        self.upstream_first_nonfinite = False
        self.packet_bytes = 0
        super().__init__(runner, options)
        try:
            spec = runner.speculator
            backbone = spec.model.model
            layers = list(backbone.layers.values())
            self.context_layers = {id(layer.self_attn): f"mtp.{index}" for index, layer in enumerate(layers)}
            self.expected_boundaries = [
                *(f"target_aux.{layer}" for layer in spec.target_layer_ids),
                "context.projected",
                *(f"mtp.{index}.context_kv" for index in range(len(layers))),
                "draft.initial_hidden",
                *(f"mtp.{index}.output" for index in range(len(layers))),
                "draft.hc_input",
                "draft.hc_output",
            ]
            self.install(spec, "_execute_draft", before=self.begin_draft, scope=True)
            self.install(spec.model, "combine_hidden_states", before=self.auxiliary, after=self.context)
            self.install(backbone, "_store_standard_swa_kv", before=self.context_kv)
            for index, layer in enumerate(layers):
                self.install(
                    layer,
                    "forward",
                    before=self.initial if index == 0 else None,
                    after=partial(self.layer_output, f"mtp.{index}.output"),
                )
            self.install(backbone, "hc_head", before=self.hc_input, after=self.hc_output)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def argument(args, kwargs, index, name):
        return args[index] if len(args) > index else kwargs[name]

    def observe(self, callback, *args):
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as error:
            self.recording_error = f"upstream: {type(error).__name__}: {error}"

    def install(self, obj, name, before=None, after=None, scope=False):
        original = getattr(obj, name)
        had_local, local = name in vars(obj), vars(obj).get(name)

        @wraps(original)
        def observed(*args, **kwargs):
            self.observe(before, args, kwargs)
            try:
                result = original(*args, **kwargs)
                self.observe(after, result, args, kwargs)
                return result
            except BaseException as error:
                self.failed(name, error)
                raise
            finally:
                if scope:
                    # Normally drained inside compute_draft_logits; if failure
                    # occurred earlier, failed() drained the partial packet.
                    if self.pending is not None:
                        self.recording_error = "Draft returned without completing head observations"
                        self.observe(self.drain)
                    self.pending = None

        setattr(obj, name, observed)
        self.hooks.append((obj, name, original, observed, had_local, local))

    def begin_draft(self, args, kwargs):
        self.pending = None
        inputs = self.argument(args, kwargs, 0, "proposal_inputs")
        batch = self.runner.input_batch
        ids = list(inputs.request_ids)
        starts = batch.query_start_loc_np[: inputs.num_reqs + 1].tolist()
        if (
            inputs.step_epoch != self.runner.speculator._proposal_step_epoch
            or inputs.rank != self.rank
            or ids != list(batch.req_ids)
            or len(ids) != inputs.num_reqs
            or len(set(ids)) != len(ids)
            or starts[0] != 0
            or starts[-1] != inputs.num_target_tokens
            or any(a >= b for a, b in zip(starts, starts[1:]))
            or inputs.num_query_tokens != inputs.num_reqs * inputs.num_speculative_tokens
        ):
            raise ValueError("Upstream observation lacks a current proposal/row mapping")
        self.pending = {
            "identity": {
                "point": self.point,
                "rank": self.rank,
                "execution": self.execution,
                "proposal_epoch": int(inputs.step_epoch),
                "request_ids": ids,
                "target_query_start_loc_cpu": starts,
                "candidate_k": inputs.num_speculative_tokens,
                "target_rows": inputs.num_target_tokens,
                "candidate_rows": inputs.num_query_tokens,
            },
            "target_layer_ids": list(inputs.target_layer_ids),
            "aux_widths": [tensor.shape[-1] for tensor in inputs.auxiliary_hidden_states],
            "boundaries": [],
            "integers": {},
        }
        # Copy only compact device metadata while this proposal owns it. Never
        # retain original views until the later D2H. Derived ends are decoded on
        # CPU from this packet, avoiding an unsafe diagnostic device gather.
        n, t, q = inputs.num_reqs, inputs.num_target_tokens, inputs.num_query_tokens
        fields = {
            "request_state_indices": inputs.request_state_indices[:n],
            "target_query_start_loc": inputs.target_query_start_loc[: n + 1],
            "target_positions": inputs.target_positions[:t],
            "target_sequence_lengths": inputs.target_sequence_lengths[:n],
            "num_sampled": inputs.num_sampled[:n],
            "num_rejected": inputs.num_rejected[:n],
            "draft_positions": inputs.draft_positions[:q],
            "draft_query_start_loc": inputs.draft_query_start_loc[: n + 1],
            "draft_sequence_lengths": inputs.draft_sequence_lengths[:n],
            "draft_input_ids": inputs.draft_input_ids[:q],
        }
        for name, tensor in fields.items():
            self.pending["integers"][name] = tensor.to(dtype=torch.int64, copy=True).reshape(-1)

    def capture(self, name, tensor, domain, **details):
        if self.pending is None:
            raise ValueError(f"Boundary {name} has no live upstream proposal")
        rows = self.pending["identity"][f"{domain}_rows"]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2 or tensor.shape[0] != rows:
            raise ValueError(f"Boundary {name} does not match {domain} rows")
        if name in [b["name"] for b in self.pending["boundaries"]]:
            raise ValueError(f"Duplicate upstream boundary {name}")
        # Reduce now on the consumer's caller stream; all trailing HC/features
        # belong to the same token row. Retain only fresh [rows, 2] flags.
        flags = torch.stack((torch.isnan(tensor).flatten(1).any(1), torch.isinf(tensor).flatten(1).any(1)), dim=1)
        self.pending["boundaries"].append(
            {
                "name": name,
                "domain": domain,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "elements_scanned": tensor.numel(),
                "flags": flags,
                **details,
            }
        )

    def auxiliary(self, args, kwargs):
        tensor = self.argument(args, kwargs, 0, "aux_hidden_states")
        if self.pending is None:
            raise ValueError("Auxiliary input has no live proposal")
        # This is the actual concatenated, valid-token argument consumed by
        # main_proj/main_norm. Split by original feature widths without copying.
        for layer, part in zip(self.pending["target_layer_ids"], tensor.split(self.pending["aux_widths"], dim=-1)):
            self.capture(f"target_aux.{layer}", part, "target", target_layer_id=layer)

    def context(self, result, args, kwargs):
        self.capture("context.projected", result, "target")

    def context_kv(self, args, kwargs):
        attn = self.argument(args, kwargs, 2, "attn")
        name = self.context_layers[id(attn)]
        layer_name = attn.dsa_attn.swa_cache_layer.prefix
        self.capture(
            f"{name}.context_kv",
            self.argument(args, kwargs, 0, "shared_kv"),
            "target",
            cache_layer=layer_name,
            group=self.runner.speculator.draft_layer_group_ids[layer_name],
        )
        slots = self.argument(args, kwargs, 1, "slot_mapping")
        self.pending["integers"][f"{name}.context_slots"] = slots.to(dtype=torch.int64, copy=True).reshape(-1)

    def initial(self, args, kwargs):
        self.capture("draft.initial_hidden", self.argument(args, kwargs, 1, "hidden_states"), "candidate")

    def layer_output(self, name, result, args, kwargs):
        self.capture(name, result[0], "candidate")

    def hc_input(self, args, kwargs):
        self.capture("draft.hc_input", self.argument(args, kwargs, 0, "x"), "candidate")

    def hc_output(self, result, args, kwargs):
        self.capture("draft.hc_output", result, "candidate")

    def transfer_numeric(self, flags):
        if self.pending is None:
            raise ValueError("Head boundary lacks its upstream observation packet")
        return self.drain(flags)

    def drain(self, head_flags=None):
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        entries = pending["boundaries"]
        head_count = head_flags.numel() if head_flags is not None else 0
        packets = [head_flags.reshape(-1).to(torch.int64)] if head_flags is not None else []
        packets += [entry["flags"].reshape(-1).to(torch.int64) for entry in entries]
        packets += list(pending["integers"].values())
        if not packets:
            raise ValueError("Empty upstream packet")
        packet = torch.cat(packets)
        self.numeric_transfers += 1
        host = packet.to(device="cpu", non_blocking=False).tolist()
        self.numeric_transfers_completed += 1
        self.packet_bytes += packet.numel() * packet.element_size()
        cursor = head_count
        boundaries = []
        identity = pending["identity"]
        for entry in entries:
            count = entry["flags"].shape[0]
            values = host[cursor : cursor + count * 2]
            cursor += count * 2
            mapped = self.map_rows(identity, entry["domain"], values)
            boundaries.append({k: v for k, v in entry.items() if k != "flags"} | {"rows": mapped})
        integers = {}
        for name, tensor in pending["integers"].items():
            integers[name] = host[cursor : cursor + tensor.numel()]
            cursor += tensor.numel()
        missing = [name for name in self.expected_boundaries if name not in [b["name"] for b in boundaries]]
        if head_flags is not None and missing:
            self.recording_error = f"Missing upstream boundaries: {missing}"
        record = identity | {
            "boundaries": boundaries,
            "device_integers": integers,
            "target_mapping_matches_device": integers.get("target_query_start_loc")
            == identity["target_query_start_loc_cpu"],
            "missing_boundaries": missing,
            "head_reached": head_flags is not None,
            "head_flag_columns": ["hidden_nan", "hidden_inf", "logits_nan", "logits_inf"][: head_flags.shape[1]]
            if head_flags is not None
            else [],
            "head_flags": [
                [bool(v) for v in host[i : i + head_flags.shape[1]]] for i in range(0, head_count, head_flags.shape[1])
            ]
            if head_flags is not None
            else None,
        }
        self.upstream_records.append(record)
        self.upstream_counts["rounds"] += 1
        has_nan = any(row["nan"] for b in boundaries for row in b["rows"])
        has_inf = any(row["inf"] for b in boundaries for row in b["rows"])
        self.upstream_counts["nan_rounds"] += int(has_nan)
        self.upstream_counts["inf_rounds"] += int(has_inf)
        # Upstream NaN/Inf may be sanitized downstream. Keep an independent
        # first file even when the existing head/Markov guard sees finite values.
        if has_nan and not self.upstream_first_nan:
            self.write("upstream-first-nan", self.snapshot(), exclusive=True)
            self.upstream_first_nan = True
        if (has_nan or has_inf) and not self.upstream_first_nonfinite:
            self.write("upstream-first-nonfinite", self.snapshot(), exclusive=True)
            self.upstream_first_nonfinite = True
        if head_flags is not None:
            columns = head_flags.shape[1]
            return [[bool(v) for v in host[i : i + columns]] for i in range(0, head_count, columns)]
        return None

    @staticmethod
    def map_rows(identity, domain, values):
        rows = []
        starts = identity["target_query_start_loc_cpu"]
        request_row = 0
        for index in range(len(values) // 2):
            if domain == "candidate":
                request_row, position = divmod(index, identity["candidate_k"])
            else:
                while index >= starts[request_row + 1]:
                    request_row += 1
                position = index - starts[request_row]
            rows.append(
                {
                    "row": index,
                    "request_id": identity["request_ids"][request_row],
                    "request_row": request_row,
                    "position_in_request": position,
                    "nan": bool(values[2 * index]),
                    "inf": bool(values[2 * index + 1]),
                }
            )
        return rows

    def failed(self, stage, error):
        if self.pending is not None:
            self.observe(self.drain)
        super().failed(stage, error)

    def snapshot(self):
        return super().snapshot() | {
            "device_values": "per-row upstream/hidden/head NaN/Inf flags and compact owned proposal integers",
            "upstream": {
                "enabled": True,
                "history_capacity": NUMERIC_ROUNDS,
                "rounds": list(self.upstream_records),
                "counts": dict(self.upstream_counts),
                "packet_bytes": self.packet_bytes,
                "wait": "one combined int64 D2H after head return, or partial packet on earlier exception",
                "kv_values": "current context write inputs only; cache contents not read",
            },
        }

    def finish_point(self):
        result = super().finish_point()
        result["upstream"].pop("rounds")  # keep RPC compact; full histories stay worker-local
        return result

    def begin_point(self, point):
        super().begin_point(point)
        self.pending = None
        self.upstream_records.clear()
        self.upstream_counts.clear()
        self.packet_bytes = 0

    def close(self):
        super().close()
        self.pending = None
        self.upstream_records.clear()
