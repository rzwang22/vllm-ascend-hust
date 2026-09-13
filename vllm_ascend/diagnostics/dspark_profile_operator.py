# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in SWA call capsules. No original input is changed or retained by reference."""

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

MAX_PLANS = 8
MAX_LAYOUTS = 128
MAX_BYTES = 256 * 1024 * 1024
GUARD_PAGES = 2


def descriptor(tensor):
    backend = sys.modules.get("torch_npu")
    base = tensor._base
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "data_ptr": tensor.data_ptr(),
        "storage_ptr": tensor.untyped_storage().data_ptr(),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
        "base_dtype": str(base.dtype) if base is not None else str(tensor.dtype),
        "npu_format": backend.get_npu_format(tensor) if tensor.device.type == "npu" and backend else None,
    }


def byte_pack(tensors):
    return torch.cat([t.contiguous().reshape(-1).view(torch.uint8) for t in tensors])


def unpack(packet, specs):
    result, cursor = {}, 0
    for name, spec in specs.items():
        count = spec["bytes"]
        dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
        result[name] = packet[cursor : cursor + count].clone().view(dtype).reshape(spec["shape"])
        cursor += count
    if cursor != packet.numel():
        raise ValueError("Operator packet byte count mismatch")
    return result


class OperatorCapture:
    def __init__(self, bank, options):
        self.epoch_input, self.options = bank.epoch_input, options
        self.plans = {}
        self.frozen = False
        self.history = []
        self.identity_written = False

    @property
    def tensors(self):
        return tuple(t for p in self.plans.values() for t in p["buffers"].values())

    @property
    def allocated_bytes(self):
        return sum(sum(t.numel() * t.element_size() for t in p["buffers"].values()) for p in self.plans.values())

    def before(self, query, kwargs):
        """Copies occur on the actual attention stream, inside the opaque op/graph."""
        if query.shape[0] > self.options["max_tokens"]:
            return None
        tensors = {"q": query, **{k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}}
        scalars = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
        cache = tensors["ori_kv"]
        if cache.ndim != 4 or cache.shape[2] != 1 or not cache.is_contiguous():
            raise ValueError("Operator capsule supports audited contiguous PA_ND only")
        columns = min(
            (self.options["max_seq_len"] + cache.shape[1] - 1) // cache.shape[1] + GUARD_PAGES,
            tensors["ori_block_table"].shape[1],
        )
        # Full prefix pages, including both ends and two trailing page columns.
        # Padding descriptor rows are preserved too; no semantic-window zeroing.
        pages = tensors["ori_block_table"][:, :columns].to(torch.int64)
        values = {k: v for k, v in tensors.items() if k != "ori_kv"}
        values["page_ids"] = pages
        values["pages"] = cache[pages.clamp(0, cache.shape[0] - 1)].flatten(0, 1)
        layout = {k: descriptor(v) for k, v in tensors.items()}
        # Q addresses can change on eager calls; layouts in a capsule describe
        # the capture call, while epoch receipts certify device copy execution.
        key = (
            query.shape[0],
            tuple((k, tuple(v.shape), tuple(v.stride()), v.storage_offset()) for k, v in tensors.items()),
            tuple(scalars.items()),
        )
        if key not in self.plans:
            backend = getattr(torch, "npu", None)
            if backend is not None and backend.is_current_stream_capturing():
                raise ValueError("Operator capsule requires shape warmup before ACLGraph capture")
            if len(self.plans) >= MAX_PLANS:
                raise ValueError("Operator capsule plan limit exceeded")
            buffers = {k: torch.empty_like(v, memory_format=torch.contiguous_format) for k, v in values.items()}
            buffers["output"] = torch.empty_like(query, memory_format=torch.contiguous_format)
            buffers["layout_id"] = torch.zeros(1, dtype=torch.int64, device=query.device)
            buffers["receipts"] = torch.full((2,), -1, dtype=torch.int64, device=query.device)
            self.plans[key] = {
                "buffers": buffers,
                "layouts_by_id": {},
                "layout_keys": {},
                "scalars": scalars,
                "capacity": query.shape[0],
                "columns": columns,
            }
            if self.allocated_bytes > MAX_BYTES:
                raise ValueError("Operator capsule device budget exceeded")
        plan = self.plans[key]
        layout_key = json.dumps(layout, sort_keys=True)
        if layout_key not in plan["layout_keys"]:
            if len(plan["layout_keys"]) >= MAX_LAYOUTS:
                raise ValueError("Operator capsule layout catalog exhausted")
            token = len(plan["layout_keys"]) + 1
            plan["layout_keys"][layout_key] = token
            plan["layouts_by_id"][token] = layout
        plan["buffers"]["layout_id"].fill_(plan["layout_keys"][layout_key])
        for name, value in values.items():
            plan["buffers"][name].copy_(value)
        plan["buffers"]["receipts"][:1].copy_(self.epoch_input)
        return plan

    def wrap(self, op):
        def observed(query, **kwargs):
            plan = self.before(query, kwargs)
            result = op(query, **kwargs)
            self.after(plan, result[0])
            return result

        return observed

    def after(self, plan, output):
        if plan is not None:
            plan["buffers"]["output"].copy_(output)
            plan["buffers"]["receipts"][1:].copy_(self.epoch_input)

    def reset(self):
        for plan in self.plans.values():
            plan["buffers"]["receipts"].fill_(-1)

    def own(self, pending):
        if pending["identity"]["point"] != self.options["point"] or self.frozen:
            return
        # Own before another target can reuse graph buffers. No D2H here.
        pending["operator_packets"] = [
            {
                "plan": {k: v for k, v in plan.items() if k not in ("buffers", "layout_keys")},
                "specs": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype), "bytes": v.numel() * v.element_size()}
                    for k, v in plan["buffers"].items()
                },
                "packet": byte_pack(list(plan["buffers"].values())),
            }
            for plan in self.plans.values()
            if plan["capacity"] == pending["identity"]["graph_capacity"]
        ]

    def save(self, record, pending, directory):
        """Persist owned current-call data before Markov checks, retaining a bounded ring."""
        if "operator_packets" not in pending:
            return
        started = time.monotonic()
        packets = pending["operator_packets"]
        if not packets:
            raise ValueError("Operator capsule missing captured shape")
        sizes = [p["packet"].numel() for p in packets]
        # One additional bounded bulk D2H after the existing numeric wait.
        cpu = torch.cat([p["packet"] for p in packets]).to("cpu", non_blocking=False)
        matching, cursor = [], 0
        for packet, size in zip(packets, sizes):
            values = unpack(cpu[cursor : cursor + size], packet["specs"])
            cursor += size
            if values["receipts"].tolist() == [record["execution"]] * 2:
                matching.append((packet, values))
        if len(matching) != 1:
            raise ValueError("Operator capsule unavailable: missing/ambiguous current call receipts")
        packet, values = matching[0]
        plan = packet["plan"]
        layouts = plan["layouts_by_id"][int(values["layout_id"][0])]
        seq, starts = values["seqused_kv"], values["cu_seqlens_q"]
        page_ids = values["page_ids"]
        required = torch.arange(page_ids.shape[1])[None, :] * layouts["ori_kv"]["shape"][1] < seq[:, None]
        valid_pages = (page_ids >= 0) & (page_ids < layouts["ori_kv"]["shape"][0])
        covered = bool(
            (seq >= 0).all() and (seq <= self.options["max_seq_len"]).all() and (valid_pages | ~required).all()
        )
        capsule = {
            "schema": 1,
            "performance_eligible": False,
            "values": values,
            **{k: v for k, v in plan.items() if k != "layouts_by_id"},
            "layouts": layouts,
            "identity": {
                k: record[k]
                for k in (
                    "point",
                    "rank",
                    "execution",
                    "proposal_epoch",
                    "request_ids",
                    "query_start_loc_cpu",
                    "graph_capacity",
                    "graph_object_id",
                )
            },
            "coverage": "PREFIX_AND_FULL_GUARD_PAGES" if covered else "UNAVAILABLE",
            "options": self.options,
            "unmapped_guard_entries": int((~valid_pages & ~required).sum()),
            "d2h_bytes": cpu.numel(),
            "capture_layout_scope": (
                "device-written layout_id binds actual capture call; copied tensor values belong to this replay"
            ),
            "kv_binding": record["target_internal"]["attention"]["kv"]["binding"],
            "query_mapping_matches": starts[: len(record["query_start_loc_cpu"])].tolist()
            == record["query_start_loc_cpu"],
            "host_transfer_seconds": time.monotonic() - started,
        }
        path = Path(directory) / f"rank-{record['rank']}-operator-{record['execution']}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(capsule, temporary)
        temporary.replace(path)
        self.history.append(path)
        while len(self.history) > 3:
            self.history.pop(0).unlink()
        self.frozen = bool(record["head_flags"] and any(any(row) for row in record["head_flags"]))
        summary = {
            "paths": [p.name for p in self.history],
            "first_error_frozen": self.frozen,
            "execution": record["execution"],
            "coverage": capsule["coverage"],
            "d2h_bytes": cpu.numel(),
            "save_seconds": time.monotonic() - started,
            "last_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        record["operator_capture"] = summary
        (Path(directory) / f"rank-{record['rank']}-operator-index.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        fingerprint_started = time.monotonic()
        if not self.identity_written:
            # Host-only and opt-in. Never query the worker after it has failed.
            from tools.dspark.operator_replay import runtime_identity

            identity = runtime_identity()
            (Path(directory) / f"rank-{record['rank']}-operator-runtime.json").write_text(
                json.dumps(identity, indent=2) + "\n"
            )
            self.identity_written = True
        summary["runtime_identity_seconds"] = time.monotonic() - fingerprint_started
        summary["total_host_seconds"] = time.monotonic() - started
        (Path(directory) / f"rank-{record['rank']}-operator-index.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        if not covered or not capsule["query_mapping_matches"]:
            raise ValueError("Operator capsule coverage/mapping unavailable; preserve original failure")
