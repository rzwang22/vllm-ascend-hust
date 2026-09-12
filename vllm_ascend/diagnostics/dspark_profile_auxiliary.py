# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in auxiliary output transfers, using actual FULL replay receipts."""

from collections import Counter, deque

import torch

from vllm_ascend.diagnostics.dspark_nan import DSparkNaNDiagnostics
from vllm_ascend.diagnostics.dspark_profile_observation import NUMERIC_ROUNDS, ProfileObservation, describe
from vllm_ascend.diagnostics.dspark_replay import DiagnosticReplayGraph


class AuxiliaryCapture:
    """ModelWithContext protocol; only raw auxiliary copies enter the graph."""

    def __init__(self, manager):
        self.manager = manager
        self.layer_ids = tuple(manager.model_runner.speculator.target_layer_ids)
        self.shapes = {}
        self.captured = {}
        self.observer = None

    def model_inputs(self, kwargs):
        size = kwargs["input_ids"].shape[0]
        if size not in self.shapes:
            DSparkNaNDiagnostics._outside_capture()
            self.shapes[size] = {
                "sources": [],
                "layouts": [],
                "epoch_input": torch.zeros(1, dtype=torch.int64, device=kwargs["input_ids"].device),
                "receipts": torch.zeros(len(self.layer_ids), dtype=torch.int64, device=kwargs["input_ids"].device),
            }
        self.shapes[size]["inputs"] = {k: kwargs[k] for k in ("input_ids", "positions")}
        return size

    def model_outputs(self, size, output):
        if not isinstance(output, tuple) or len(output[1]) != len(self.layer_ids):
            raise ValueError("Auxiliary capture requires the actual target auxiliary outputs")
        state = self.shapes[size]
        aux = output[1]
        if not state["sources"]:
            DSparkNaNDiagnostics._outside_capture()
            state["sources"] = [torch.empty_like(value) for value in aux]
        for i, (source, value) in enumerate(zip(state["sources"], aux)):
            # Executed in ModelWithContext before the unchanged Core closure's
            # persistent output copy. Capture records these ATen copies, while
            # the original compiled model and its output ownership stay intact.
            source.copy_(value)
            state["receipts"][i : i + 1].copy_(state["epoch_input"])
        state["layouts"] = [describe(value) for value in aux]  # layouts only, never live raw views

    def finish_capture(self, states):
        for desc, pair in states.items():
            if desc.cg_mode.name != "FULL":
                continue
            if desc.num_tokens not in self.shapes:
                raise RuntimeError("Auxiliary capture missed a FULL shape")
            graph = self.manager.graphs[desc]
            if isinstance(graph, DiagnosticReplayGraph):
                raise RuntimeError("Auxiliary capture cannot replace another replay observer")
            self.captured[desc] = pair.captured
            self.manager.graphs[desc] = DiagnosticReplayGraph(graph, self, desc)

    def before(self, desc, graph):
        if self.observer is None or self.observer.point is None:
            return None  # capture/profile replay has no actual execution identity
        return self.observer.guard(self.observer.before_replay, desc, graph)

    def after(self, desc, detail):
        if detail is not None:
            self.observer.guard(self.observer.after_replay, desc)

    def unavailable(self, detail, reason, *, failed=False):
        # DiagnosticReplayGraph rethrows the original replay exception. Defer
        # file saving to the enclosing target hook; never mask that exception.
        detail["replay_returned"] = False
        detail["replay_error"] = reason

    @property
    def allocated_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for state in self.shapes.values()
            for t in (*state["sources"], state["epoch_input"], state["receipts"])
        )


class AuxiliaryProfileObservation(ProfileObservation):
    def __init__(self, runner, options):
        self.capture = getattr(runner.cudagraph_manager, "_dspark_auxiliary_capture", None)
        if self.capture is None or not self.capture.captured:
            raise ValueError("Auxiliary transfers require snapshots installed during FULL capture")
        self.pending = None
        self.auxiliary_records = deque(maxlen=NUMERIC_ROUNDS)
        self.auxiliary_counts = Counter()
        self.auxiliary_latches = set()
        self.packet_bytes = 0
        super().__init__(runner, options)
        self.capture.observer = self

    def guard(self, callback, *args):
        try:
            return callback(*args)
        except Exception as error:
            self.recording_error = f"auxiliary: {type(error).__name__}: {error}"
            return None

    def record(self, stage, **payload):
        super().record(stage, **payload)
        if stage == "target_execute.enter":
            self.guard(self.drain)  # a prior target may legitimately have no next proposal
        elif stage == "proposal_prepare.return":
            self.guard(self.bind_proposal, payload["result"])
        elif stage == "combined_context.enter":
            args, kwargs = payload["args"], payload["kwargs"]
            self.guard(self.consume, args[0] if args else kwargs["aux_hidden_states"])

    def start_record(self, capacity=None):
        batch = self.runner.input_batch
        n, t = batch.num_reqs, batch.num_tokens
        ids = list(batch.req_ids)
        starts = batch.query_start_loc_np[: n + 1].tolist()
        if (
            len(ids) != n
            or len(set(ids)) != n
            or len(starts) != n + 1
            or starts[0] != 0
            or starts[-1] != t
            or (capacity is not None and t > capacity)
            or any(a >= b for a, b in zip(starts, starts[1:]))
        ):
            raise ValueError("Auxiliary observation lacks actual target row mapping")
        self.pending = {
            "identity": {
                "point": self.point,
                "rank": self.rank,
                "execution": self.execution,
                "request_ids": ids,
                "query_start_loc_cpu": starts,
                "pool_rows_cpu": batch.idx_mapping_np[:n].tolist(),
                "target_rows": t,
                "graph_capacity": capacity,
                "proposal_epoch": None,
            },
            "flags": [],
            "integers": {},
            "layouts": {},
            "replay_returned": False,
        }
        return self.pending

    def integer(self, name, value):
        # These must be owned before buffers are reused, including graph receipts.
        self.pending["integers"][name] = value.to(dtype=torch.int64, copy=True).reshape(-1)

    def before_replay(self, desc, graph):
        if self.pending is not None:
            raise ValueError("Multiple FULL replays for one auxiliary execution")
        pending = self.start_record(desc.num_tokens)
        state = self.capture.shapes[desc.num_tokens]
        pending["state"] = state
        pending["graph_object_id"] = id(graph)
        pending["layouts"]["raw_at_capture"] = state["layouts"]
        batch = self.runner.input_batch
        n, t = batch.num_reqs, batch.num_tokens
        for name, value in {
            "target.query_start_loc": batch.query_start_loc[: n + 1],
            "target.pool_rows": batch.idx_mapping[:n],
            "target.seq_lens": batch.seq_lens[:n],
            "target.positions": batch.positions[:t],
            "target.input_ids": batch.input_ids[:t],
            "captured.positions": state["inputs"]["positions"][: desc.num_tokens],
            "captured.input_ids": state["inputs"]["input_ids"][: desc.num_tokens],
            "target.is_padding": self.runner.input_buffers.is_padding[: desc.num_tokens],
        }.items():
            self.integer(name, value)
        # Save only query/position/sequence fields actually bound to captured
        # attention objects, deduplicated by identity. No block table/KV sweep.
        seen, tensor_sources = {}, {}
        metadata_sources = {}
        for layer, metadata in self.capture.captured[desc].attn_metadata.items():
            if id(metadata) in seen:
                metadata_sources[layer] = {"same_object_as": seen[id(metadata)]}
                continue
            seen[id(metadata)] = layer
            fields = {}
            for part, obj in (("root", metadata), ("decode", getattr(metadata, "decode", None))):
                for field in ("query_start_loc", "seq_lens", "positions"):
                    value = getattr(obj, field, None)
                    if isinstance(value, torch.Tensor):
                        key = f"captured_attn.{layer}.{part}.{field}"
                        if id(value) not in tensor_sources:
                            self.integer(key, value)
                            tensor_sources[id(value)] = key
                        fields[f"{part}.{field}"] = tensor_sources[id(value)]
            metadata_sources[layer] = fields
        pending["captured_attention_fields"] = metadata_sources
        state["receipts"].fill_(-1)
        state["epoch_input"].fill_(self.execution)
        return pending

    def flags(self, boundary, layer, value, reference=None):
        rows = (
            self.pending["identity"]["target_rows"]
            if boundary == "consumed"
            else self.pending["identity"]["graph_capacity"]
        )
        if value.ndim != 2 or value.shape[0] != rows:
            raise ValueError("Auxiliary boundary must be a token/feature tensor")
        if any(e["boundary"] == boundary and e["layer"] == layer for e in self.pending["flags"]):
            raise ValueError("Duplicate auxiliary boundary")
        flags = [torch.isnan(value).any(1), torch.isinf(value).any(1)]
        if reference is not None:
            flags.append(~((value == reference) | (torch.isnan(value) & torch.isnan(reference))).all(1))
        self.pending["flags"].append(
            {"boundary": boundary, "layer": layer, "shape": list(value.shape), "values": torch.stack(flags, dim=1)}
        )

    def after_replay(self, desc):
        pending = self.pending
        state = pending["state"]
        pending["replay_returned"] = True
        self.integer("raw_receipts", state["receipts"])
        dest = self.runner.cudagraph_manager.aux_hidden_states
        if len(dest) != len(self.capture.layer_ids):
            raise ValueError("Auxiliary persistent destination count changed")
        pending["layouts"]["persistent_after_replay"] = [describe(x) for x in dest]
        for layer, raw, value in zip(self.capture.layer_ids, state["sources"], dest):
            self.flags("raw", layer, raw)
            self.flags("persistent", layer, value[: desc.num_tokens], raw)

    def bind_proposal(self, inputs):
        if self.pending is None:
            self.start_record()  # eager/prefill: consume stats, raw transfer explicitly unavailable
        identity = self.pending["identity"]
        if (
            identity["execution"] != self.execution
            or inputs.step_epoch != self.runner.speculator._proposal_step_epoch
            or inputs.rank != self.rank
            or list(inputs.request_ids) != identity["request_ids"]
            or inputs.num_target_tokens != identity["target_rows"]
        ):
            raise ValueError("Auxiliary producer and proposal identities differ")
        identity["proposal_epoch"] = int(inputs.step_epoch)
        self.pending["widths"] = [x.shape[1] for x in inputs.auxiliary_hidden_states]
        self.pending["layouts"]["proposal_views"] = [describe(x) for x in inputs.auxiliary_hidden_states]
        n, t = inputs.num_reqs, inputs.num_target_tokens
        for name, value in {
            "proposal.query_start_loc": inputs.target_query_start_loc[: n + 1],
            "proposal.positions": inputs.target_positions[:t],
            "proposal.seq_lens": inputs.target_sequence_lengths[:n],
            "proposal.pool_rows": inputs.request_state_indices[:n],
            "proposal.num_rejected": inputs.num_rejected[:n],
            "proposal.num_sampled": inputs.num_sampled[:n],
        }.items():
            self.integer(name, value)

    def consume(self, tensor):
        pending = self.pending
        if pending is None or pending["identity"]["proposal_epoch"] != self.runner.speculator._proposal_step_epoch:
            raise ValueError("Auxiliary consumption has no live producer/proposal identity")
        t = pending["identity"]["target_rows"]
        if tensor.shape[0] != t:
            raise ValueError("Auxiliary consumption must exclude graph padding")
        pending["layouts"]["consumed_concat"] = describe(tensor)
        state = pending.get("state")
        if state is not None:
            self.integer("consume_receipts", state["receipts"])
        for i, (layer, value) in enumerate(zip(self.capture.layer_ids, tensor.split(pending["widths"], dim=-1))):
            reference = state["sources"][i][:t] if state is not None else None
            self.flags("consumed", layer, value, reference)

    def transfer_numeric(self, flags):
        if self.pending is None or self.pending["identity"]["proposal_epoch"] != self.numeric_context.step_epoch:
            raise ValueError("Head lacks the matching auxiliary proposal record")
        return self.drain(flags)

    def drain(self, head_flags=None):
        pending, self.pending = self.pending, None
        if pending is None:
            return None
        entries, integers = pending["flags"], pending["integers"]
        head_count = head_flags.numel() if head_flags is not None else 0
        packets = [head_flags.flatten().to(torch.int64)] if head_flags is not None else []
        packets += [e["values"].flatten().to(torch.int64) for e in entries]
        packets += list(integers.values())
        if not packets:
            raise ValueError("No auxiliary observation packet was produced")
        packet = torch.cat(packets)
        self.numeric_transfers += 1
        host = packet.to(device="cpu", non_blocking=False).tolist()
        self.numeric_transfers_completed += 1
        self.packet_bytes += packet.numel() * packet.element_size()
        identity, cursor = pending["identity"], head_count
        boundaries = []
        for entry in entries:
            count, columns = entry["values"].shape
            values = host[cursor : cursor + count * columns]
            cursor += count * columns
            rows = self.map_rows(identity, values, columns)
            boundaries.append({k: v for k, v in entry.items() if k != "values"} | {"rows": rows})
        device = {}
        for name, value in integers.items():
            device[name] = host[cursor : cursor + value.numel()]
            cursor += value.numel()
        full = identity["graph_capacity"] is not None
        fresh = pending["replay_returned"] and device.get("raw_receipts") == [identity["execution"]] * len(
            self.capture.layer_ids
        )
        consumed = any(e["boundary"] == "consumed" for e in entries)
        receipt_valid = fresh and (not consumed or device.get("consume_receipts") == device.get("raw_receipts"))
        if full and not receipt_valid:
            self.recording_error = "Auxiliary snapshot receipt missing/stale or replay did not return"
        expected = [
            (boundary, layer)
            for boundary in (("raw", "persistent", "consumed") if full else ("consumed",))
            for layer in self.capture.layer_ids
        ]
        missing = [
            f"{boundary}.{layer}"
            for boundary, layer in expected
            if not any(e["boundary"] == boundary and e["layer"] == layer for e in entries)
        ]
        if head_flags is not None and missing:
            self.recording_error = f"Missing auxiliary boundaries: {missing}"
        head_rows = (
            [[bool(x) for x in host[i : i + head_flags.shape[1]]] for i in range(0, head_count, head_flags.shape[1])]
            if head_flags is not None
            else None
        )
        record = identity | {
            "boundaries": boundaries,
            "device_integers": device,
            "raw_replay_verified": bool(full and fresh),
            "coverage": "FULL" if full and receipt_valid else "INVALID_RECEIPT" if full else "NON_FULL_RAW_UNOBSERVED",
            "missing_boundaries": missing,
            "consumption_reached": consumed,
            "head_reached": head_flags is not None,
            "head_flags": head_rows,
            "head_flag_columns": ["hidden_nan", "hidden_inf", "logits_nan", "logits_inf"][: head_flags.shape[1]]
            if head_flags is not None
            else [],
            "layouts": pending["layouts"],
            "graph_object_id": pending.get("graph_object_id"),
            "captured_attention_fields": pending.get("captured_attention_fields"),
            "replay_error": pending.get("replay_error"),
            "target_mapping_matches_device": device.get("target.query_start_loc") == identity["query_start_loc_cpu"]
            if full
            else None,
        }
        self.auxiliary_records.append(record)
        self.auxiliary_counts[record["coverage"]] += 1
        valid = [r for b in boundaries for r in b["rows"] if r["valid_target_row"]]
        for suffix, occurs in (
            ("nan", any(r["nan"] for r in valid)),
            ("nonfinite", any(r["nan"] or r["inf"] for r in valid)),
            ("difference", any(r.get("differs_from_raw", False) for r in valid)),
        ):
            self.auxiliary_counts[suffix + "_rounds"] += int(occurs)
            if occurs and suffix not in self.auxiliary_latches:
                self.write(f"auxiliary-first-{suffix}", self.snapshot(), exclusive=True)
                self.auxiliary_latches.add(suffix)
        if head_flags is not None:
            return head_rows
        return None

    @staticmethod
    def map_rows(identity, values, columns):
        rows, req = [], 0
        starts = identity["query_start_loc_cpu"]
        for row in range(len(values) // columns):
            valid = row < identity["target_rows"]
            if valid:
                while row >= starts[req + 1]:
                    req += 1
            item = {
                "row": row,
                "valid_target_row": valid,
                "request_id": identity["request_ids"][req] if valid else None,
                "request_row": req if valid else None,
                "position_in_request": row - starts[req] if valid else None,
                "nan": bool(values[row * columns]),
                "inf": bool(values[row * columns + 1]),
            }
            if columns == 3:
                item["differs_from_raw"] = bool(values[row * columns + 2])
            rows.append(item)
        return rows

    def failed(self, stage, error):
        self.guard(self.drain)
        super().failed(stage, error)

    def snapshot(self):
        data = super().snapshot()
        data["numeric"]["wait"] = "one auxiliary/head packet; partial targets may drain without a head"
        return data | {
            "device_values": "auxiliary raw/persistent/consumed row flags, comparisons and owned integer metadata",
            "auxiliary": {
                "enabled": True,
                "rounds": list(self.auxiliary_records),
                "history_capacity": NUMERIC_ROUNDS,
                "counts": dict(self.auxiliary_counts),
                "packet_bytes": self.packet_bytes,
                "capture_bytes": self.capture.allocated_bytes,
                "wait": "one combined D2H at head, or partial record on error/next execution/point end",
                "scope": "FULL raw model-return auxiliaries only; no target layer or KV contents",
            },
        }

    def finish_point(self):
        self.guard(self.drain)
        result = super().finish_point()
        result["auxiliary"].pop("rounds")
        return result

    def begin_point(self, point):
        self.guard(self.drain)  # preserve an unconsumed target before changing its point identity
        super().begin_point(point)
        self.pending = None
        self.auxiliary_records.clear()
        self.auxiliary_counts.clear()
        self.packet_bytes = 0

    def close(self):
        super().close()
        if self.capture.observer is self:
            self.capture.observer = None
        self.pending = None
        self.auxiliary_records.clear()
