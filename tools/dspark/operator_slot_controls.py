# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit counterfactual copies of one physical KV slot; never a production fix."""

import hashlib

import torch


def raw_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def slot_hash(tensor):
    return hashlib.sha256(raw_bytes(tensor)).hexdigest()


def changed_ranges(before, after):
    """Exact half-open byte ranges relative to the selected physical slot."""
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    ranges = []
    for i in changed:
        if ranges and ranges[-1][1] == i:
            ranges[-1][1] += 1
        else:
            ranges.append([i, i + 1])
    return ranges


def replace_slot(base, donor, block, offset):
    from tools.dspark.operator_replay import validate

    validate(base)
    validate(donor)
    spec, other = base["layouts"]["ori_kv"], donor["layouts"]["ori_kv"]
    for key in ("shape", "stride", "dtype", "storage_offset", "npu_format", "base_dtype"):
        if spec[key] != other[key]:
            raise ValueError(f"Slot donor cache layout mismatch: {key}")
    if not 0 <= block < spec["shape"][0] or not 0 <= offset < spec["shape"][1]:
        raise ValueError("Physical slot outside cache")
    if spec["stride"][-1] != 1 or spec["stride"][-2] != spec["shape"][-1]:
        raise ValueError("Slot byte range requires contiguous head dimensions")
    indices = [i for i, p in enumerate(base["values"]["page_ids"].flatten().tolist()) if p == block]
    source_indices = [i for i, p in enumerate(donor["values"]["page_ids"].flatten().tolist()) if p == block]
    if not indices or not source_indices:
        raise ValueError("Selected physical page was not captured in both files")
    before = base["values"]["pages"][indices[0], offset]
    after = donor["values"]["pages"][source_indices[0], offset]
    if (
        before.dtype != after.dtype
        or str(before.dtype) != spec["dtype"]
        or list(before.shape) != spec["shape"][2:]
        or after.shape != before.shape
    ):
        raise ValueError("Slot payload dtype/shape differs from layout")
    if not base["values"]["pages"].is_contiguous() or not donor["values"]["pages"].is_contiguous():
        raise ValueError("Slot payload must be contiguous")
    result = {**base, "values": {**base["values"], "pages": base["values"]["pages"].clone()}}
    for i in indices:
        result["values"]["pages"][i, offset].copy_(after)
    validate(result)  # includes every duplicate physical page, including NaN bytes
    before_bytes, after_bytes = raw_bytes(before), raw_bytes(after)
    row_bytes = len(before_bytes)
    info = {
        "kind": "single-physical-slot counterfactual; not observed model input or production fix",
        "performance_eligible": False,
        "base_identity": base["identity"],
        "donor_identity": donor["identity"],
        "block": block,
        "offset": offset,
        "dtype": spec["dtype"],
        "shape": list(before.shape),
        "slot_bytes": row_bytes,
        "before_sha256": slot_hash(before),
        "after_sha256": slot_hash(after),
        "cache_storage_byte_range": [
            (spec["storage_offset"] + block * spec["stride"][0] + offset * spec["stride"][1]) * before.element_size(),
            (spec["storage_offset"] + block * spec["stride"][0] + offset * spec["stride"][1]) * before.element_size()
            + row_bytes,
        ],
        "captured_page_indices_updated": indices,
        "pages_payload_byte_ranges": [
            [(i * spec["shape"][1] + offset) * row_bytes, (i * spec["shape"][1] + offset + 1) * row_bytes]
            for i in indices
        ],
        "changed_byte_ranges_within_slot": changed_ranges(before_bytes, after_bytes),
        "changed_bytes_per_copy": sum(a != b for a, b in zip(before_bytes, after_bytes)),
        "preserved": (
            "all other tensors/scalars/layouts/formats/metadata/receipts; original output remains the original capture"
        ),
    }
    result["counterfactual"] = info
    return result, info


def save_watch(watch, phase, before, after):
    """One 2-slot D2H per completed standalone invocation; no global sync."""
    packet = torch.stack((before, after)).cpu()
    watch["snapshots"].append({"phase": phase, "values": packet})
    watch["records"].append(
        {
            "phase": phase,
            "before_sha256": slot_hash(packet[0]),
            "after_sha256": slot_hash(packet[1]),
            "changed_byte_ranges": changed_ranges(raw_bytes(packet[0]), raw_bytes(packet[1])),
            "d2h_bytes": packet.numel() * packet.element_size(),
        }
    )
