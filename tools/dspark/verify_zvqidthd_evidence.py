# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU audit of all 24 ZvqiDthD capsules using the formal restricted replay path."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from tools.dspark import operator_replay as replay
from tools.dspark.verify_fppa5l9e_evidence import read_archive

ARCHIVE_SHA = "95ce4baf4c95a6313193f5921fd9437c7280c8cdd11f183037e98abd4d74f281"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tensor_digest(value):
    return digest(value.contiguous().view(torch.uint8).numpy().tobytes())


def windows(capsule):
    """Audit semantic reads, without claiming they are every physical kernel read."""
    values, params = capsule["values"], capsule["scalars"]
    lookup = {page: values["pages"][i] for i, page in enumerate(values["page_ids"].flatten().tolist())}
    starts = values["cu_seqlens_q"].tolist()
    block_size = capsule["layouts"]["ori_kv"]["shape"][1]
    result, used = [], set()
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        seq = int(values["seqused_kv"][request])
        for row in range(start, end):
            position = seq - (end - row)
            left, right = max(0, position - params["ori_win_left"]), min(seq, position + params["ori_win_right"] + 1)
            bad, blocks, abs_max = [], [], 0.0
            for pos in range(left, right):
                block = int(values["ori_block_table"][request, pos // block_size])
                offset = pos % block_size
                value = lookup[block][offset]
                used.add((block, offset))
                if not blocks or blocks[-1] != block:
                    blocks.append(block)
                if not value.isfinite().all():
                    bad.append(
                        {
                            "position": pos,
                            "block": block,
                            "offset": offset,
                            "nan_indices": value.isnan().nonzero().tolist(),
                            "inf_indices": value.isinf().nonzero().tolist(),
                        }
                    )
                else:
                    abs_max = max(abs_max, float(value.abs().max()))
            result.append(
                {
                    "row": row,
                    "request_id": capsule["identity"]["request_ids"][request],
                    "position": position,
                    "window": [left, right],
                    "physical_blocks": blocks,
                    "nonfinite": bad,
                    "finite_abs_max": abs_max,
                }
            )
    others = {"slots": 0, "nan_slots": 0, "inf_slots": 0, "nan_elements": 0, "inf_elements": 0}
    valid_pages = {p for p in lookup if 0 <= p < capsule["layouts"]["ori_kv"]["shape"][0]}
    for page in valid_pages:
        for offset, value in enumerate(lookup[page]):
            if (page, offset) in used:
                continue
            others["slots"] += 1
            others["nan_slots"] += int(value.isnan().any())
            others["inf_slots"] += int(value.isinf().any())
            others["nan_elements"] += int(value.isnan().sum())
            others["inf_elements"] += int(value.isinf().sum())
    return result, {
        "unique_pages": len(valid_pages),
        "window_union_physical_slots": len(used),
        "outside_all_valid_windows": others,
        "kernel_actual_read_scope": "UNKNOWN",
    }


def float32_control(capsule):
    """CPU arithmetic experiment on saved values, NOT the Ascend kernel."""
    v, p = capsule["values"], capsule["scalars"]
    q, sink = v["q"].float(), v["sinks"].float()
    out = torch.full_like(q, float("nan"))
    lookup = {page: v["pages"][i] for i, page in enumerate(v["page_ids"].flatten().tolist())}
    starts = v["cu_seqlens_q"].tolist()
    block_size = capsule["layouts"]["ori_kv"]["shape"][1]
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        seq = int(v["seqused_kv"][request])
        for row in range(start, end):
            position = seq - end + row
            kv = torch.stack(
                [
                    lookup[int(v["ori_block_table"][request, pos // block_size])][pos % block_size, 0]
                    for pos in range(max(0, position - p["ori_win_left"]), min(seq, position + 1))
                ]
            ).float()
            scores = q[row] @ kv.T * p["softmax_scale"]
            out[row] = torch.softmax(torch.cat((scores, sink[:, None]), 1), 1)[:, :-1] @ kv
    return out


def history_control(previous, current):
    """Compare retained history before current query starts, not rewritten query tokens."""
    old, new = previous["values"], current["values"]
    lookup = {p: old["pages"][i] for i, p in enumerate(old["page_ids"].flatten().tolist())}
    next_lookup = {p: new["pages"][i] for i, p in enumerate(new["page_ids"].flatten().tolist())}
    patched = {**current, "values": {**new, "pages": new["pages"].clone()}}
    changes = []
    block_size = current["layouts"]["ori_kv"]["shape"][1]
    for row, req in enumerate(current["identity"]["request_ids"]):
        old_row = previous["identity"]["request_ids"].index(req)
        old_start = int(old["seqused_kv"][old_row] - old["cu_seqlens_q"].diff()[old_row])
        new_start = int(new["seqused_kv"][row] - new["cu_seqlens_q"].diff()[row])
        left = max(0, max(old_start, new_start) - current["scalars"]["ori_win_left"])
        for pos in range(left, min(new_start, int(old["seqused_kv"][old_row]))):
            before_page = int(old["ori_block_table"][old_row, pos // block_size])
            after_page = int(new["ori_block_table"][row, pos // block_size])
            a, b = lookup[before_page][pos % block_size], next_lookup[after_page][pos % block_size]
            if before_page == after_page and tensor_digest(a) == tensor_digest(b):
                continue
            changes.append(
                {
                    "request_id": req,
                    "position": pos,
                    "old_block": before_page,
                    "new_block": after_page,
                    "offset": pos % block_size,
                    "old_sha256": tensor_digest(a),
                    "new_sha256": tensor_digest(b),
                    "changed_components": int((a.view(torch.int16) != b.view(torch.int16)).sum()),
                    "old_abs_max": float(a.abs().max()),
                    "new_abs_max": float(b.abs().max()),
                    "new_fp32_bit_view_abs_max": float(b.view(torch.float32).abs().max()),
                }
            )
            for index, page in enumerate(new["page_ids"].flatten().tolist()):
                if page == after_page:
                    patched["values"]["pages"][index, pos % block_size].copy_(a)
    replay.validate(patched)
    valid = int(new["cu_seqlens_q"][-1])
    original = float32_control(current)
    repaired_history = float32_control(patched)
    return {
        "previous_execution": previous["identity"]["execution"],
        "current_execution": current["identity"]["execution"],
        "historical_changes": changes,
        "original_cpu_fp32_nan_row_heads": original[:valid].isnan().any(-1).nonzero().tolist(),
        "counterfactual_cpu_fp32_nan_row_heads": repaired_history[:valid].isnan().any(-1).nonzero().tolist(),
        "counterfactual_scope": (
            "CPU-only copy with reported historical slots restored; original capsule unchanged; not a fix or NPU replay"
        ),
    }


def audit_capsule(blob, output):
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(blob))
    assert unsafe == []
    capsule = replay.validate(torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True))
    values, layouts = capsule["values"], capsule["layouts"]
    assert all(type(spec["npu_format"]) is int and spec["npu_format"] == 2 for spec in layouts.values())
    assert all(
        list(values[k].shape) == v["shape"] and str(values[k].dtype) == v["dtype"]
        for k, v in layouts.items()
        if k != "ori_kv"
    )
    assert capsule["scalars"] == dict(
        softmax_scale=512**-0.5,
        cmp_ratio=1,
        ori_mask_mode=4,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    # Real CPU restoration verifies original views, byte equality and aliases.
    restored = replay.restore(capsule, "cpu")
    for name, value in restored.items():
        assert list(value.stride()) == layouts[name]["stride"]
        assert value.storage_offset() == layouts[name]["storage_offset"]
    for index, page in enumerate(values["page_ids"].flatten().tolist()):
        if 0 <= page < restored["ori_kv"].shape[0]:
            assert tensor_digest(restored["ori_kv"][page]) == tensor_digest(values["pages"][index])
    del restored
    ref = replay.reference(capsule)
    torch.save(ref, output / "reference.pt")
    valid = capsule["identity"]["query_start_loc_cpu"][-1]
    assert values["cu_seqlens_q"].tolist() == capsule["identity"]["query_start_loc_cpu"] + [valid] * 9
    assert values["seqused_kv"][3:].tolist() == [0] * 9
    assert ref[:valid].isfinite().all()
    rows, regions = windows(capsule)
    result = {
        "identity": capsule["identity"],
        "sha256": digest(blob),
        "bytes": len(blob),
        "unsafe_globals": unsafe,
        "coverage": capsule["coverage"],
        "query_mapping_matches": capsule["query_mapping_matches"],
        "receipts": values["receipts"].tolist(),
        "layouts": layouts,
        "scalars": capsule["scalars"],
        "padded_starts": values["cu_seqlens_q"].tolist(),
        "padded_seq_lens": values["seqused_kv"].tolist(),
        "q": {
            k: v
            for k, v in replay.compare_output(values["q"], torch.zeros_like(values["q"]), valid).items()
            if k.startswith(("nan_", "inf_"))
        },
        "q_all_rows_nonfinite": (~values["q"].isfinite()).any(-1).nonzero().tolist(),
        "q_abs_max": float(values["q"][:valid].abs().max()),
        "sinks": values["sinks"].tolist(),
        "sinks_finite": bool(values["sinks"].isfinite().all()),
        "windows": rows,
        "saved_regions": regions,
        "captured_comparison": replay.compare_output(values["output"], ref, valid),
        "padding_output_nonfinite": (~values["output"][valid:].isfinite()).any(-1).nonzero().tolist(),
        "reference_finite": True,
        "reference_sha256": tensor_digest(ref),
        "tensor_sha256": {k: tensor_digest(v) for k, v in values.items()},
        "restore": "CPU views/offsets/aliases/input bytes/full saved pages passed; NPU formats pending",
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify(archive, output):
    """Archive assertions are offline evidence checks, not NPU tests."""
    files, links = read_archive(archive, ARCHIVE_SHA, 208, 448443829)
    assert b"1141 passed" in files["focused.log"] and files["focused.pipestatus"].strip() == b"0 0"

    def read(name):
        return json.loads(files["runs/b64/" + name])

    plan = read("plan.json")
    assert plan["plugin_sha"] == "7847c94dc8c8895efb0ae22f8f30636990bd12ee"
    assert plan["core_sha"] == "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
    retained = read("retained.json")
    assert len(retained) == 9
    for item in retained:
        name = item["point"]["id"] + ".json"
        assert digest(files["runs/b64/" + name]) == item["raw_sha256"]
        raw = read(name)
        assert raw["streaming"]["error"] is None and not raw["performance_eligible"]
        assert all(len(r["output_token_ids"]) == 512 and r["error"] is None for r in raw["streaming"]["requests"])
    failed = read("ctx128-n4-t12-skewed.json")
    tokens = [len(r["output_token_ids"]) for r in failed["streaming"]["requests"]]
    assert tokens == [512, 397, 124, 95]
    capsules, ranks = [], []
    assert len([k for k in files if k.endswith(".pt")]) == 24
    for rank in range(8):
        previous = None
        base = f"worker-first-failure/rank-{rank}-"
        first, latest = read(base + "first-nan.json"), read(base + "latest.json")
        gate = read(base + "attention-validity.json")
        assert gate["status"] == "passed" and [r["execution"] for r in gate["rounds"]] == [2, 3, 4]
        assert latest["recording_error"] is None
        assert (
            first["recording_error"]
            == "auxiliary: ValueError: Operator capsule coverage/mapping unavailable; preserve original failure"
        )
        events = read(base + "error-events.json")["events"]
        assert [e["execution"] for e in events] == [1803, 1804]
        assert (
            "base logits contain NaN" in events[0]["key"][2] and "lack current proposal owners" in events[1]["key"][2]
        )
        assert all(e["epochs"]["_published_proposal_step_epoch"] is None for e in events)
        assert [r["execution"] for r in first["auxiliary"]["rounds"]] == [1801, 1802, 1803]
        for record in first["auxiliary"]["rounds"]:
            epoch = record["execution"]
            ints = record["device_integers"]
            assert ints["target_internal.receipts"] == [epoch] * 21
            assert ints["kv.receipts"] == [epoch] * 4
            assert ints["raw_receipts"] == ints["consume_receipts"] == [epoch] * 3
            assert record["raw_replay_verified"] and record["target_internal"]["coverage"] == "FULL"
            name = base + f"operator-{epoch}.pt"
            target = output / f"rank-{rank}-execution-{epoch}"
            target.mkdir()
            result = audit_capsule(files["runs/b64/" + name], target)
            assert result["sha256"] == record["operator_capture"]["last_sha256"]
            assert result["identity"]["proposal_epoch"] == epoch - 13
            assert result["identity"]["query_start_loc_cpu"] == [0, 1, 5, 11]
            assert result["q"]["nan_row_heads"] == result["q"]["inf_row_heads"] == []
            assert result["sinks_finite"] and all(not r["nonfinite"] for r in result["windows"])
            assert result["captured_comparison"]["inf_row_heads"] == []
            assert bool(result["captured_comparison"]["nan_row_heads"]) == (epoch == 1803)
            current = torch.load(io.BytesIO(files["runs/b64/" + name]), map_location="cpu", weights_only=True)
            if previous is not None:
                result["history_control"] = history_control(previous, current)
                assert (
                    result["history_control"]["original_cpu_fp32_nan_row_heads"]
                    == result["captured_comparison"]["nan_row_heads"]
                )
                assert result["history_control"]["counterfactual_cpu_fp32_nan_row_heads"] == []
            previous = current
            capsules.append(result)
        ranks.append(
            {
                "rank": rank,
                "recording_error": first["recording_error"],
                "events": events,
                "last_clean_saved_execution": latest["auxiliary"]["rounds"][-1]["execution"],
                "transitions": [
                    {
                        "before": t["before"]["execution"],
                        "after": t["after"]["execution"],
                        "before_requests": t["before"]["request_ids"],
                        "after_requests": t["after"]["request_ids"],
                    }
                    for t in first["transitions"]
                ],
                "operator_runtime": read(base + "operator-runtime.json"),
            }
        )
    cleanup = read("cleanup.json")
    assert cleanup["status"] == "worker_cleanup_incomplete" and not cleanup["success"]
    result = {
        "archive_sha256": ARCHIVE_SHA,
        "members": 208,
        "expanded_bytes": 448443829,
        "links": links,
        "plugin": plan["plugin_sha"],
        "core": plan["core_sha"],
        "focused_passed": 1141,
        "retained": [{"point": r["point"]["id"], "sha256": r["raw_sha256"]} for r in retained],
        "failed_point_output_tokens": tokens,
        "capsules": capsules,
        "ranks": ranks,
        "cleanup": cleanup,
        "root_cause": "UNKNOWN",
        "NPU_replay": "PENDING",
        "coverage_error_execution": "UNKNOWN: global error has no epoch; earlier capsules overwritten",
        "performance_eligible": False,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = verify(args.archive, args.output)
    print(json.dumps({"capsules_validated": len(data["capsules"]), "NPU_replay": "PENDING"}))
