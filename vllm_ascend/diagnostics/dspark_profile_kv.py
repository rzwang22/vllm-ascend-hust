# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded selected-layer KV provenance; all writes stay inside the opaque DSA op."""

import torch

MAX_BINDINGS = 32
WINDOW_COLUMNS = ("physical_block", "nan", "inf")
WRITE_COLUMNS = (
    "slot_block",
    "slot_offset",
    "slot_valid",
    "duplicate_slot",
    "source_nan",
    "source_inf",
    "target_nan",
    "target_inf",
    "differs_source",
)
RECEIPT_STAGES = ("binding", "before_scatter", "after_scatter", "window")


def layout(value):
    return {
        "data_ptr": value.data_ptr(),
        "storage_data_ptr": value.untyped_storage().data_ptr(),
        "storage_offset": value.storage_offset(),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
    }


class KVProbe:
    def __init__(self, bank, window):
        self.bank = bank
        self.bindings = {}
        self.keys = {}
        self.writes = torch.zeros((2, bank.max_tokens, len(WRITE_COLUMNS)), dtype=torch.int64, device=bank.flags.device)
        self.window = torch.zeros(
            (bank.max_tokens, window, len(WINDOW_COLUMNS)), dtype=torch.int64, device=bank.flags.device
        )
        self.receipts = torch.full((len(RECEIPT_STAGES), 1), -1, dtype=torch.int64, device=bank.flags.device)
        self.binding = torch.zeros(1, dtype=torch.int64, device=bank.flags.device)

    @property
    def tensors(self):
        return self.writes, self.window, self.receipts, self.binding

    def bind(self, layer_name, cache, metadata, window):
        # This catalog is capture provenance. The token and receipt below are
        # actual device writes, so a later eager call cannot relabel a replay.
        values = {
            "cache": cache,
            "block_table": metadata.block_table,
            "slots": metadata.slot_mapping,
            "query_starts": metadata.query_start_loc,
            "seq_lens": metadata.seq_lens,
        }
        descriptions = {k: layout(v) for k, v in values.items()}
        key = (
            layer_name,
            window,
            tuple((k, v.data_ptr(), tuple(v.shape), tuple(v.stride())) for k, v in values.items()),
        )
        if key not in self.keys:
            if len(self.keys) >= MAX_BINDINGS:
                raise ValueError("KV diagnostic binding catalog exhausted")
            token = len(self.keys) + 1
            self.keys[key] = token
            self.bindings[token] = {"layer_name": layer_name, "window": window, "tensors": descriptions}
        self.binding.fill_(self.keys[key])
        self.receipts[0].copy_(self.bank.epoch_input)

    def scatter(self, cache, source, slots, phase):
        n = min(source.shape[0], self.bank.max_tokens)
        indices = slots[:n].to(torch.int64)
        valid = (
            (indices[:, 0] >= 0)
            & (indices[:, 0] < cache.shape[0])
            & (indices[:, 1] >= 0)
            & (indices[:, 1] < cache.shape[1])
        )
        target = cache[indices[:, 0].clamp(0, cache.shape[0] - 1), indices[:, 1].clamp(0, cache.shape[1] - 1)]
        src = source[:n].flatten(1)
        dst = target.flatten(1)
        same = (src == dst) | (torch.isnan(src) & torch.isnan(dst))
        duplicate = ((indices[:, None, :] == indices[None, :, :]).all(-1) & valid[None, :]).sum(1) > 1
        stats = torch.stack(
            (
                indices[:, 0],
                indices[:, 1],
                valid,
                duplicate,
                torch.isnan(src).any(1),
                torch.isinf(src).any(1),
                torch.isnan(dst).any(1),
                torch.isinf(dst).any(1),
                ~same.all(1),
            ),
            1,
        )
        self.writes[phase, :n].copy_(stats)
        self.receipts[phase + 1].copy_(self.bank.epoch_input)

    def observe_window(self, values, pages, needed):
        n, width = pages.shape
        flags = torch.stack(
            (
                torch.where(needed, pages, -1),
                torch.isnan(values).flatten(2).any(2),
                torch.isinf(values).flatten(2).any(2),
            ),
            2,
        )
        self.window[:n].fill_(-1)
        self.window[:n, :width].copy_(flags)
        self.receipts[3].copy_(self.bank.epoch_input)

    def packet(self, observer, tokens):
        observer.integer("kv.writes", self.writes[:, :tokens])
        observer.integer("kv.window", self.window[:tokens])
        observer.integer("kv.receipts", self.receipts)
        observer.integer("kv.binding", self.binding)

    def fresh(self, device, tokens, execution):
        return (
            device.get("kv.receipts") == [execution] * len(RECEIPT_STAGES)
            and len(device.get("kv.binding", [])) == 1
            and device["kv.binding"][0] in self.bindings
            and len(device.get("kv.writes", [])) == 2 * tokens * len(WRITE_COLUMNS)
            and len(device.get("kv.window", [])) == tokens * self.window.shape[1] * len(WINDOW_COLUMNS)
        )

    def decode(self, device, rows, tokens):
        result = {
            "receipts": device["kv.receipts"],
            "receipt_stages": RECEIPT_STAGES,
            "binding": self.bindings[device["kv.binding"][0]],
            "rows": [],
        }
        width = self.window.shape[1]
        for row in rows:
            if not row["valid_target_row"]:
                continue
            index = row["row"]
            item = {k: row[k] for k in ("row", "request_id", "request_row", "position")}
            item["invalid_window_indices"] = row.get("invalid_window_indices")
            for phase, name in enumerate(("before_scatter", "after_scatter")):
                start = (phase * tokens + index) * len(WRITE_COLUMNS)
                item[name] = dict(zip(WRITE_COLUMNS, device["kv.writes"][start : start + len(WRITE_COLUMNS)]))
            page_runs, bad = [], []
            observed_width = result["binding"]["window"]
            for j in range(observed_width):
                start = (index * width + j) * len(WINDOW_COLUMNS)
                block, nan, inf = device["kv.window"][start : start + len(WINDOW_COLUMNS)]
                if block == -1:
                    continue
                position = row["position"] - (observed_width - 1) + j
                if not page_runs or page_runs[-1]["physical_block"] != block:
                    page_runs.append({"logical_start": position, "logical_end": position + 1, "physical_block": block})
                else:
                    page_runs[-1]["logical_end"] = position + 1
                if nan or inf:
                    bad.append(
                        {
                            "logical_position": position,
                            "physical_block": block,
                            "offset": position % result["binding"]["tensors"]["cache"]["shape"][1],
                            "nan": bool(nan),
                            "inf": bool(inf),
                        }
                    )
            item.update(page_runs=page_runs, nonfinite_slots=bad, first_nonfinite=bad[0] if bad else None)
            result["rows"].append(item)
        for item in result["rows"]:
            for slot in item["nonfinite_slots"]:
                slot["current_writers"] = [
                    {k: other[k] for k in ("row", "request_id", "request_row", "position")}
                    for other in result["rows"]
                    if other["after_scatter"]["slot_valid"]
                    and other["after_scatter"]["slot_block"] == slot["physical_block"]
                    and other["after_scatter"]["slot_offset"] == slot["offset"]
                ]
        return result
