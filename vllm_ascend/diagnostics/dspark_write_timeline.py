# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in single-history-slot byte timeline inside opaque DSA calls, not layer scans."""

import hashlib
import json
import weakref
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_ascend.diagnostics.dspark_profile_operator import byte_pack, descriptor, unpack

MAX_SITES = 256
MAX_DEVICE_BYTES = 8 * 1024 * 1024
MAX_SOURCES = 2
SOURCE_BYTES = 16 * 1024
MAX_CATALOG = 512


def span(tensor):
    d = descriptor(tensor)
    start = d["data_ptr"]
    end = start + (1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))) * tensor.element_size()
    return {**d, "byte_span": [start, end], "scope": "strided bounding span; holes not proven touched"}


def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensors(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]
    return []


class WriteMode(TorchDispatchMode):
    def __init__(self, owner, label, binding):
        super().__init__()
        self.owner, self.label, self.binding = owner, label, binding
        self.ordinal = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        values = {a.name: v for a, v in zip(func._schema.arguments, args)}
        values.update(kwargs)
        candidates = []
        target = self.owner.cache()
        target_span = span(target)["byte_span"]
        for name, value in values.items():
            for t in tensors(value):
                if t.device == target.device and t.numel() and overlap(span(t)["byte_span"], target_span):
                    candidates.append((name, t))
        writes = [a.name for a in func._schema.arguments if a.alias_info and a.alias_info.is_write]
        if not any(name in writes for name, _ in candidates):
            return func(*args, **kwargs)
        ordinal = self.ordinal
        self.ordinal += 1
        key = (self.label, ordinal, str(func), self.owner.capacity)
        sources = [
            (k, t)
            for k, v in values.items()
            for t in tensors(v)
            if k not in {n for n, _ in candidates}
            and t.device == target.device
            and 0 < t.numel() * t.element_size() <= 65536
        ][:MAX_SOURCES]
        plan = self.owner.before(key, self.binding, candidates, sources, writes)
        result = func(*args, **kwargs)
        self.owner.after(plan, self.binding)
        return result


class SlotWriteTimeline:
    def __init__(self, bank, target, options):
        self.target = weakref.ref(target)
        self.epoch = bank.epoch_input
        self.options = options
        self.control = torch.zeros(2, dtype=torch.int64, device=self.epoch.device)
        self.plans = {}
        self.capacity = 0
        self.pending = None
        self.history = []
        self.first_change = None
        self.first_unavailable = None
        self.catalog = []
        self.catalog_written = False
        self.previous_tail = None
        self.following_saved = False

    @property
    def tensors(self):
        return (self.control, *(t for p in self.plans.values() for t in p["buffers"].values()))

    def cache(self):
        cache = self.target().swa_cache_layer.kv_cache
        if isinstance(cache, list) and len(cache) == 1:
            cache = cache[0]
        if not isinstance(cache, torch.Tensor) or cache.dtype != torch.bfloat16 or cache.shape[2:] != (1, 512):
            raise ValueError("Write timeline requires audited BF16 PA_ND 1024-byte slots")
        return cache

    def binding(self, metadata):
        """Select the first page-end in the current SWA window, using real device metadata."""
        cache = self.cache()
        block_size = cache.shape[1]
        md = metadata[self.target().swa_cache_layer.prefix].decode
        row = self.control[:1].clamp(0, md.seq_lens.shape[0] - 1)
        starts = md.query_start_loc
        index = starts[row].long().clamp(0, md.input_positions.numel() - 1)
        position = md.input_positions[index].long()
        left = (position - self.target().window_size + 1).clamp_min(0)
        logical = ((left // block_size) + 1) * block_size - 1
        column = (logical // block_size).clamp(0, md.block_table.shape[1] - 1)
        page = md.block_table[row, column].long()
        valid = (
            (self.control[1:2] > 0) & (starts[row + 1] - starts[row] == 1) & (logical < position) & (left <= logical)
        )
        valid = valid & (page >= 0) & (page < cache.shape[0])
        return torch.stack(
            (row, position, logical, page, logical.remainder(block_size), valid.long(), self.epoch[:1])
        ).reshape(-1)

    def observe(self, label, context, capacity):
        if capacity > self.options["max_tokens"] or context.attn_metadata is None:
            return nullcontext()
        md = context.attn_metadata.get(self.target().swa_cache_layer.prefix)
        if md is None or getattr(md, "decode", None) is None:
            return nullcontext()
        self.capacity = capacity
        return WriteMode(self, label, self.binding(context.attn_metadata))

    def read(self, binding):
        cache = self.cache()
        return cache[binding[3:4].clamp(0, cache.shape[0] - 1), binding[4:5]].contiguous().view(torch.uint8).reshape(-1)

    def before(self, key, binding, candidates, sources, writes):
        if key not in self.plans:
            if torch.npu.is_current_stream_capturing():
                raise ValueError("Write timeline site was not warmed before capture")
            if len(self.plans) >= MAX_SITES:
                raise ValueError("Write timeline site budget exhausted")
            buffers = {
                "layout_id": torch.zeros(1, dtype=torch.int64, device=self.epoch.device),
                "before": torch.empty(1024, dtype=torch.uint8, device=self.epoch.device),
                "after": torch.empty(1024, dtype=torch.uint8, device=self.epoch.device),
                "binding": torch.empty(7, dtype=torch.int64, device=self.epoch.device),
                "receipt": torch.full((2,), -1, dtype=torch.int64, device=self.epoch.device),
            }
            for i, (_, t) in enumerate(sources):
                buffers[f"source{i}"] = torch.empty(
                    min(t.numel() * t.element_size(), SOURCE_BYTES), dtype=torch.uint8, device=t.device
                )
            self.plans[key] = {
                "buffers": buffers,
                "layouts": {},
                "site": list(key),
                "targets": [(n, span(t)) for n, t in candidates],
                "sources": [(n, span(t)) for n, t in sources],
                "declared_writes": writes,
                "stream_at_capture": str(torch.npu.current_stream().npu_stream),
                "source_scope": "logical contiguous prefix, at most 16384 bytes per source; not full source",
            }
            if (
                sum(t.numel() * t.element_size() for p in self.plans.values() for t in p["buffers"].values())
                > MAX_DEVICE_BYTES
            ):
                raise ValueError("Write timeline memory budget exhausted")
        plan = self.plans[key]
        b = plan["buffers"]
        layout = {
            "targets": [(n, span(t)) for n, t in candidates],
            "sources": [(n, span(t)) for n, t in sources],
            "stream_at_capture": str(torch.npu.current_stream().npu_stream),
        }
        encoded = json.dumps(layout, sort_keys=True)
        if encoded not in plan["layouts"]:
            if len(plan["layouts"]) >= 128:
                raise ValueError("Write layout catalog exhausted")
            plan["layouts"][encoded] = (len(plan["layouts"]) + 1, layout)
        b["layout_id"].fill_(plan["layouts"][encoded][0])
        b["before"].copy_(self.read(binding))
        b["binding"].copy_(binding)
        b["receipt"][:1].copy_(self.epoch)
        for i, (_, t) in enumerate(sources):
            b[f"source{i}"].copy_(t.contiguous().reshape(-1).view(torch.uint8)[: b[f"source{i}"].numel()])
        return plan

    def after(self, plan, binding):
        plan["buffers"]["after"].copy_(self.read(binding))
        plan["buffers"]["receipt"][1:].copy_(self.epoch)

    def begin(self, pending, runner, schedule_id, metadata):
        ids = pending["identity"]["request_ids"]
        starts = pending["identity"]["query_start_loc_cpu"]
        candidates = [i for i in range(len(ids)) if starts[i + 1] - starts[i] == 1]
        active = (
            pending["identity"]["point"] == self.options["point"]
            and bool(candidates)
            and ((self.first_change is None and self.first_unavailable is None) or not self.following_saved)
        )
        # Row selection is host metadata only; position and physical mapping are read on device.
        row = min(candidates, key=lambda i: int(runner.input_batch.seq_lens_np[i])) if candidates else 0
        self.control.copy_(torch.tensor([row, int(active)], dtype=torch.int64, device="cpu"), non_blocking=True)
        for p in self.plans.values():
            p["buffers"]["receipt"].fill_(-1)
        self.pending = pending if active else None
        if active:
            pending["write_schedule_id"] = schedule_id
            pending["write_request_id"] = ids[row]
            pending["write_host"] = []
            self.bound_device = self.binding(metadata)
            self.host("target.before")
            if not self.catalog_written:
                context = runner.compilation_config.static_forward_context
                target_span = span(self.cache())["byte_span"]
                self.catalog = []
                for name, layer in context.items():
                    for t in tensors(getattr(layer, "kv_cache", None)):
                        d = span(t)
                        self.catalog.append(
                            {"name": name, **d, "intersects_target_allocation": overlap(d["byte_span"], target_span)}
                        )
                        if len(self.catalog) > MAX_CATALOG:
                            raise ValueError("KV catalog budget exhausted")
                pending["write_catalog"] = {
                    "caches": self.catalog,
                    "target": span(self.cache()),
                    "groups": [
                        {"group": i, "layers": g.layer_names, "spec": str(g.kv_cache_spec)}
                        for i, g in enumerate(runner.kv_cache_config.kv_cache_groups)
                    ],
                    "workspace_scope": (
                        "only tensor arguments exposed to observed operators; private kernel workspace UNKNOWN"
                    ),
                }
                self.catalog_written = True

    def host(self, stage):
        if self.pending is None:
            return
        if len(self.pending["write_host"]) >= 16:
            raise ValueError("Write host stage budget exceeded")
        self.pending["write_host"].append(
            {
                "phase": stage,
                "binding": self.bound_device.clone(),
                "values": self.read(self.bound_device).clone(),
                "epoch": self.epoch.clone(),
                "stream": str(torch.npu.current_stream().npu_stream),
            }
        )

    def own(self, pending):
        if self.pending is not pending:
            return
        self.host("target.after")
        pending["write_packets"] = []
        for p in self.plans.values():
            if p["site"][-1] != pending["identity"]["graph_capacity"]:
                continue
            specs = {
                k: {"shape": list(v.shape), "dtype": str(v.dtype), "bytes": v.numel() * v.element_size()}
                for k, v in p["buffers"].items()
            }
            pending["write_packets"].append(
                {
                    "specs": specs,
                    "metadata": {k: v for k, v in p.items() if k != "buffers"},
                    "packet": byte_pack(list(p["buffers"].values())),
                }
            )

    @staticmethod
    def mapping_matches(bind, pending, record):
        identity = pending["identity"]
        row = bind[0]
        starts = identity["query_start_loc_cpu"]
        positions = record.get("device_integers", {}).get("target.positions", [])
        return (
            bind[5] == 1
            and bind[6] == record["execution"]
            and 0 <= row < len(identity["request_ids"])
            and identity["request_ids"][row] == pending["write_request_id"]
            and starts[row + 1] - starts[row] == 1
            and starts[row] < len(positions)
            and positions[starts[row]] == bind[1]
        )

    def save(self, record, pending, directory):
        """Own one CPU packet, validate receipts/mapping, then freeze bounded change or unavailable evidence."""
        if "write_packets" not in pending:
            return
        packets = pending["write_packets"]
        results, unavailable = [], []
        if not packets:
            unavailable.append({"reason": "No writer candidates captured"})
        if pending["write_schedule_id"] is None:
            unavailable.append({"reason": "Missing scheduler association"})
        host = pending["write_host"]
        host_packet = byte_pack([t for h in host for t in (h["binding"], h["epoch"], h["values"])])
        cpu = torch.cat([p["packet"] for p in packets] + [host_packet]).cpu()
        cursor = 0
        for p in packets:
            size = p["packet"].numel()
            v = unpack(cpu[cursor : cursor + size], p["specs"])
            cursor += size
            bind = v["binding"].tolist()
            reason = None
            if v["receipt"].tolist() != [record["execution"]] * 2 or bind[6] != record["execution"]:
                reason = "missing/stale replay receipt"
            elif not self.mapping_matches(bind, pending, record):
                reason = "request/position/binding unavailable"
            if reason:
                unavailable.append({"site": p["metadata"]["site"], "reason": reason, "values": v})
                continue
            start = (
                span(self.cache())["storage_ptr"]
                + (self.cache().storage_offset() + bind[3] * self.cache().stride(0) + bind[4] * self.cache().stride(1))
                * 2
            )
            layout = next(item[1] for item in p["metadata"]["layouts"].values() if item[0] == int(v["layout_id"][0]))
            targets = [
                {"name": name, **d, "intersects_selected_slot": overlap(d["byte_span"], [start, start + 1024])}
                for name, d in layout["targets"]
            ]
            changed = not torch.equal(v["before"], v["after"])
            results.append(
                {
                    **{k: value for k, value in p["metadata"].items() if k != "layouts"},
                    **layout,
                    "targets": targets,
                    "values": v,
                    "slot_byte_range": [start, start + 1024],
                    "changed": changed,
                }
            )
        host_records = []
        for h in host:
            binding = cpu[cursor : cursor + 56].clone().view(torch.int64)
            cursor += 56
            epoch = cpu[cursor : cursor + 8].clone().view(torch.int64)
            cursor += 8
            value = cpu[cursor : cursor + 1024].clone()
            cursor += 1024
            if int(epoch[0]) != record["execution"] or not self.mapping_matches(binding.tolist(), pending, record):
                unavailable.append({"phase": h["phase"], "reason": "Stale/mismatched host write receipt"})
            host_records.append({"phase": h["phase"], "stream": h["stream"], "binding": binding, "values": value})
        transitions = []
        ordered = (
            host_records[:1]
            + [{"phase": r["site"], "binding": r["values"]["binding"], "values": r["values"]["after"]} for r in results]
            + host_records[1:]
        )
        previous = self.previous_tail
        for entry in ordered:
            key = (pending["write_request_id"], int(entry["binding"][2]))
            rebound = previous is not None and not torch.equal(previous["binding"][3:5], entry["binding"][3:5])
            if (
                previous
                and previous["key"] == key
                and (rebound or not torch.equal(previous["values"], entry["values"]))
            ):
                transitions.append(
                    {
                        "page_rebinding": rebound,
                        "from": previous["phase"],
                        "to": entry["phase"],
                        "scope": "host/capture order brackets only; concurrent streams UNKNOWN",
                    }
                )
            previous = {"key": key, **entry}
        self.previous_tail = previous
        report = {
            "identity": pending["identity"],
            "schedule_id": pending["write_schedule_id"],
            "request_id": pending["write_request_id"],
            "sites": results,
            "unavailable": unavailable,
            "coverage": "UNAVAILABLE" if unavailable else "CURRENT_OBSERVED_SITES",
            "graph_object_id": pending.get("graph_object_id"),
            "host": host_records,
            "transitions": transitions,
            "catalog": pending.get("write_catalog"),
            "d2h_bytes": cpu.numel(),
            "performance_eligible": False,
            "interpretation": "changes bracket operations; overlapping streams prevent automatic writer attribution",
        }
        directory = Path(directory)
        if pending.get("write_catalog") is not None:
            (directory / f"rank-{record['rank']}-writes-catalog.json").write_text(
                json.dumps(pending["write_catalog"], indent=2) + "\n"
            )
        path = directory / f"rank-{record['rank']}-writes-{record['execution']}.pt"
        torch.save(report, path)
        self.history.append(path)
        changed = [r["site"] for r in results if r["changed"]]
        if self.first_change is not None or self.first_unavailable is not None:
            self.following_saved = True
        if unavailable and self.first_unavailable is None:
            self.first_unavailable = record["execution"]
        if (
            changed or transitions or any(any(row) for row in record.get("head_flags") or ())
        ) and self.first_change is None:
            self.first_change = record["execution"]
        self.pending = None
        if record["execution"] in (self.first_change, self.first_unavailable):
            while len(self.history) > 3:
                self.history.pop(0).unlink()
        while len(self.history) > 3 and self.first_change is None and self.first_unavailable is None:
            self.history.pop(0).unlink()
        (directory / f"rank-{record['rank']}-writes-index.json").write_text(
            json.dumps(
                {
                    "paths": [p.name for p in self.history],
                    "first_change": self.first_change,
                    "changed_sites": changed,
                    "execution": record["execution"],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "first_unavailable": self.first_unavailable,
                    "recording_error": "writer coverage unavailable" if unavailable else None,
                    "performance_eligible": False,
                },
                indent=2,
            )
            + "\n"
        )

        if unavailable:
            raise ValueError("Writer coverage unavailable; raw packets saved before rejection")
