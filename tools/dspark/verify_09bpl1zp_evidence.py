# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit actual page lifecycle events and owned write snapshots, never execute archive code."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from tools.dspark.verify_fppa5l9e_evidence import read_archive


def verify(path):
    """Check all ranks and byte/layout evidence, keeping completion and failures separate."""
    files, links = read_archive(
        path, "535d1555ee61da4ad0a82859daf2db67a5a760959c044a319cfe2795e7f5054f", 217, 225053033
    )
    prefix = "runs/b64/worker-first-failure/"

    def js(name):
        return json.loads(files[name])

    def load(name):
        assert not torch.serialization.get_unsafe_globals_in_checkpoint(io.BytesIO(files[name]))
        return torch.load(io.BytesIO(files[name]), weights_only=True, map_location="cpu")

    events = [json.loads(line) for line in files[prefix + "scheduler-page-timeline.jsonl"].splitlines()]
    by_seq = {e["sequence"]: e for e in events}
    free, returned, allocate = [by_seq[s] for s in (11788, 11790, 11948)]
    owner = "batch10-3-8dd4972f"
    assert free["schedule_id"] == 1801 and free["group"] == 3 and free["arguments"]["request_id"] == owner
    assert free["arguments"]["total_computed_tokens"] == 223 and free["request_blocks"][2]["block_id"] == 35
    assert returned["freed_before"][0]["ref_cnt"] == 1 and returned["freed_blocks"][0]["ref_cnt"] == 0
    assert returned["freed_blocks"][0]["block_id"] == 35
    assert allocate["schedule_id"] == 1803 and allocate["result_blocks"][0]["block_id"] == 35
    assert allocate["context"][1] == {"call": "allocate_new_blocks", "group": 5, "request_id": "batch10-1-b8283095"}
    assert by_seq[11949]["request_blocks"][64]["block_id"] == 35
    accounting = []
    for e in events:
        if 1799 <= e["schedule_id"] <= 1806 and e["event"] in (
            "schedule.before",
            "schedule.after",
            "output.before",
            "output.after",
        ):
            accounting.append(
                {k: e[k] for k in ("sequence", "schedule_id", "event")}
                | {
                    "output_schedule_id": e.get("output_schedule_id"),
                    "request": next((v for v in e["requests"] if v["request_id"] == owner), None),
                    "sampled_lengths": e.get("sampled_lengths"),
                }
            )
    at_free = next(e for e in accounting if e["schedule_id"] == 1801 and e["event"] == "schedule.before")["request"]
    assert (at_free["num_computed_tokens"], at_free["num_output_placeholders"], at_free["num_tokens"]) == (223, 6, 218)
    ranks = []
    for rank in range(8):
        latest = js(prefix + f"rank-{rank}-latest.json")
        assert latest["target_internal"]["counts"]["nonfinite_rounds"] == 0
        assert "Operator capsule coverage/mapping unavailable" in latest["recording_error"]
        rounds = []
        catalog = js(prefix + f"rank-{rank}-writes-catalog.json")
        swa = next(c for c in catalog["caches"] if c["name"] == "model.layers.1.self_attn.swa_cache")
        state = next(c for c in catalog["caches"] if c["name"] == "model.layers.3.self_attn.compressor.state_cache")
        assert swa["storage_ptr"] == state["storage_ptr"] and swa["storage_nbytes"] == state["storage_nbytes"]
        assert swa["dtype"] == "torch.bfloat16" and state["dtype"] == "torch.float32"
        assert swa["shape"] == [16231, 32, 1, 512] and state["shape"] == [16231, 8, 1, 1024]
        assert swa["storage_nbytes"] == 531857408
        assert swa["storage_offset"] == state["storage_offset"] == 0
        for execution in (1803, 1804, 1805, 1806):
            filename = prefix + f"rank-{rank}-writes-{execution}.pt"
            w = load(filename)
            assert w["schedule_id"] == execution and w["coverage"] == "CURRENT_OBSERVED_SITES" and not w["unavailable"]
            changes = []
            for s in w["sites"]:
                v = s["values"]
                assert v["receipt"].tolist() == [execution, execution]
                assert v["binding"][6].item() == execution
                if execution <= 1805:
                    assert v["binding"][2:5].tolist() == [95, 35, 31]
                before, after = v["before"], v["after"]
                n = int((before != after).sum())
                if n:
                    assert execution == 1805 and n == 981
                    assert s["site"][0] == "model.layers.3.self_attn" and s["site"][2] == "_C_ascend.compressor.default"
                    start = swa["storage_ptr"] + (35 * swa["stride"][0] + 31 * swa["stride"][1]) * 2
                    assert s["slot_byte_range"] == [start, start + 1024]
                    target = next(t for t in s["targets"] if t["name"] == "state_cache")
                    assert target["storage_ptr"] == swa["storage_ptr"] and target["stride"] == [8192, 1024, 1]
                    assert start - state["storage_ptr"] == 35 * 32768 + 7 * 4096 + 768 * 4
                    assert float(before.view(torch.bfloat16).abs().max()) == 14.125
                    assert float(after.view(torch.bfloat16).abs().max()) > 1e38
                    changes.append(
                        {
                            "site": s["site"],
                            "changed_bytes": n,
                            "slot_byte_range": s["slot_byte_range"],
                            "before_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest(),
                            "after_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
                            "before_bf16_abs_max": float(before.view(torch.bfloat16).abs().max()),
                            "after_bf16_abs_max": float(after.view(torch.bfloat16).abs().max()),
                            "capture_stream": s["stream_at_capture"],
                        }
                    )
            assert len(changes) == int(execution == 1805)
            if execution == 1805:
                assert (
                    w["identity"]["query_start_loc_cpu"] == [0, 1, 5, 11]
                    and w["identity"]["target_rows"] == 11
                    and w["identity"]["graph_capacity"] == 12
                )
                assert w["host"][0]["binding"].tolist() == [0, 222, 95, 35, 31, 1, 1805]
            rounds.append(
                {
                    "execution": execution,
                    "identity": w["identity"],
                    "binding": w["host"][0]["binding"].tolist(),
                    "changes": changes,
                    "sha256": hashlib.sha256(files[filename]).hexdigest(),
                }
            )
        index = js(prefix + f"rank-{rank}-writes-index.json")
        assert index["first_change"] == 1805 and index["execution"] == 1806 and index["changed_sites"] == []
        tail = []
        for ex in (1908, 1909, 1910):
            c = load(prefix + f"rank-{rank}-operator-{ex}.pt")
            assert c["query_mapping_matches"] and c["values"]["receipts"].tolist() == [ex, ex]
            seq = int(c["values"]["seqused_kv"].max())
            assert seq == {1908: 635, 1909: 641, 1910: 647}[ex]
            assert c["options"]["max_seq_len"] == 640
            assert c["coverage"] == ("PREFIX_AND_FULL_GUARD_PAGES" if ex == 1908 else "UNAVAILABLE")
            tail.append({"execution": ex, "max_seq": seq, "coverage": c["coverage"], "mapping": True})
        ranks.append({"rank": rank, "shared_storage_ptr": swa["storage_ptr"], "rounds": rounds, "capsule_tail": tail})
    plan = js("runs/b64/plan.json")
    assert plan["plugin_sha"] == "890fc350627eba18e8ca1296ad1b4b310283c2ff"
    assert plan["core_sha"] == "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
    assert plan["performance_eligible"] is False
    retained = js("runs/b64/retained.json")
    assert len(retained) == 9
    for row in retained:
        assert hashlib.sha256(files["runs/b64/" + row["point"]["id"] + ".json"]).hexdigest() == row["raw_sha256"]
    failure = js("runs/b64/profile-failure.json")
    stream = failure["last_stream"]
    assert stream["error"] is None and all(r["error"] is None for r in stream["requests"])
    counts = [len(r["output_token_ids"]) for r in stream["requests"]]
    assert counts == [512] * 4
    assert "Operator capsule coverage/mapping unavailable" in failure["error"]
    cleanup = js("runs/b64/cleanup.json")
    assert cleanup["forced_cleanup"] and not cleanup["success"]
    assert "1161 passed" in files["focused.log"].decode()
    return {
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plugin_sha": plan["plugin_sha"],
        "core_sha": plan["core_sha"],
        "links_not_followed": links,
        "page_events": [free, returned, allocate],
        "accounting": accounting,
        "ranks": ranks,
        "retained_hashes_verified": 9,
        "tenth_generated_tokens": counts,
        "focused_passed": 1161,
        "numerical_nan": "NOT_REPRODUCED",
        "lifecycle": (
            "page freed at speculative upper bound, reused across groups while worker historical mapping still live"
        ),
        "diagnostic_error": failure["error"],
        "cleanup_status": cleanup["status"],
        "worker_natural_exit": False,
        "performance_eligible": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(verify(a.archive), indent=2) + "\n")
