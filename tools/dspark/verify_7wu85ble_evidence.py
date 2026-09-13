# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify preflight and full KV-window evidence directly from immutable archives."""

import argparse
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath


def read_archive(path, expected, count, size):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    files, links = {}, []
    with tarfile.open(path) as tar:
        members = tar.getmembers()
        assert len(members) == count and sum(m.size for m in members) == size
        for member in members:
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and ".." not in name.parts
            key = str(PurePosixPath(*name.parts[1:]))
            if member.isfile():
                assert key not in files
                files[key] = tar.extractfile(member).read()
            elif member.issym():
                links.append({"path": key, "target": member.linkname, "followed": False})
            else:
                assert member.isdir()
    return files, links


def verify(preflight, profile):
    """Verify archive bytes, completed points, all-rank receipts and error order."""
    pre, links = read_archive(preflight, "9742d99889b7ea0b07649aade3d610572590957f2e84a0d4c22c2d1c107a99b8", 27, 328210)
    files, _ = read_archive(profile, "e6eef44f121983b79c5ae9a5e9c4b8fcb64fd690495aacb81eafa8918d856e6a", 168, 229614085)
    cases = ET.fromstring(pre["four.xml"]).findall(".//testcase")
    assert len(cases) == 4 and all(not list(c) for c in cases)
    assert "1089 passed" in files["focused.log"].decode()
    assert pre["four.pipestatus"].strip() == files["focused.pipestatus"].strip() == b"0 0"

    def read(name):
        return json.loads(files["runs/b64/" + name])

    plan = read("plan.json")
    assert plan["plugin_sha"] == "9d4c3c6381cad4e6ba878fd63513a75b0a6e26ce"
    assert plan["core_sha"] == "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
    retained = read("retained.json")
    assert len(retained) == 9
    points = []
    for item in retained:
        name = item["point"]["id"] + ".json"
        digest = hashlib.sha256(files["runs/b64/" + name]).hexdigest()
        assert item["raw_sha256"] == digest
        raw = read(name)
        assert raw["streaming"]["error"] is None and not raw["performance_eligible"]
        assert all(len(r["output_token_ids"]) == 512 and r["error"] is None for r in raw["streaming"]["requests"])
        points.append({"point": item["point"]["id"], "raw_sha256": digest})
    ranks = []
    for rank in range(8):
        base = f"worker-first-failure/rank-{rank}-"
        gate = read(base + "attention-validity.json")
        assert gate["status"] == "passed" and [r["execution"] for r in gate["rounds"]] == [2, 3, 4]
        for row in gate["rounds"]:
            e = row["execution"]
            assert row["target_receipts"] == [e] * 21
            assert row["raw_receipts"] == row["consume_receipts"] == [e] * 3
        d = read(base + "first-nan.json")
        assert d["recording_error"] is None
        for index, numeric in enumerate(d["numeric"]["rounds"]):
            assert numeric["execution"] == 1800 + index and numeric["proposal_epoch"] == 1788 + index
            for column in ("hidden_nan", "logits_nan"):
                assert [v["candidate_row"] for v in numeric["rows"] if v[column]] == (
                    list(range(5)) if index == 2 else []
                )
            assert not any(v["hidden_inf"] or v["logits_inf"] for v in numeric["rows"])
        rounds = d["auxiliary"]["rounds"]
        assert [r["execution"] for r in rounds] == [1800, 1801, 1802]
        evidence = []
        for i, row in enumerate(rounds):
            e = row["execution"]
            ints = row["device_integers"]
            assert ints["target_internal.receipts"] == [e] * 21
            assert ints["raw_receipts"] == ints["consume_receipts"] == [e] * 3
            assert row["coverage"] == row["target_internal"]["coverage"] == "FULL" and row["raw_replay_verified"]
            cuts = row["target_internal"]["boundaries"]
            if i < 2:
                assert not any(r["nan"] or r["inf"] for cut in cuts for r in cut["rows"])
            else:
                by_name = {c["name"]: c["rows"][0] for c in cuts}
                for name in (
                    "attn_input",
                    "attention.q_normalized",
                    "attention.q_rope",
                    "attention.kv_normalized",
                    "attention.kv_rope",
                ):
                    assert not by_name["layer.1." + name]["nan"] and not by_name["layer.1." + name]["inf"]
                assert (
                    by_name["layer.1.attention.kv_window"]["nan"] and by_name["layer.1.attention.raw_attention"]["nan"]
                )
            state = row["target_internal"]["attention"]["rows"][0]
            assert state["position"] == 220 + i and state["slot_block"] == 105 and state["slot_offset"] == 28 + i
            assert state["invalid_window_indices"] == state["slot_mismatch"] == 0
            evidence.append(
                {
                    "execution": e,
                    "proposal_epoch": row["proposal_epoch"],
                    "state_row0": state,
                    "nan_rows": {c["name"]: [r["row"] for r in c["rows"] if r["nan"]] for c in cuts},
                }
            )
        assert row["proposal_epoch"] == 1790 and row["query_start_loc_cpu"] == [0, 1, 5, 11]
        assert row["request_ids"] == ["batch10-3-b5ceb0ef", "batch10-2-96aa4ea9", "batch10-1-b67892c5"]
        assert row["target_rows"] == 11 and row["graph_capacity"] == 12
        events = read(base + "error-events.json")["events"]
        assert [v["execution"] for v in events] == [1802, 1803]
        assert "Markov base logits contain NaN" in events[0]["key"][2]
        assert "lack current proposal owners" in events[1]["key"][2]
        assert all(v["epochs"]["_published_proposal_step_epoch"] is None and not v["owner_epochs"] for v in events)
        ranks.append(
            {
                "rank": rank,
                "first_nan_sha256": hashlib.sha256(files["runs/b64/" + base + "first-nan.json"]).hexdigest(),
                "rounds": evidence,
                "events": events,
                "route_scope": "capacity catalog, not failure layout",
                "route": row["target_internal"]["attention"]["route"],
            }
        )
    cleanup = read("cleanup.json")
    assert cleanup["status"] == "worker_cleanup_incomplete" and cleanup["success"] is False
    return {
        "kind": "independent offline archive verification, not a new NPU run",
        "preflight_passed": 4,
        "focused_passed": 1089,
        "ignored_links": links,
        "points": points,
        "ranks": ranks,
        "cleanup": cleanup,
        "cleanup_failure": read("cleanup-failure.json"),
        "root_cause": "UNKNOWN: window contents nonfinite; producer and logical/physical slot not yet identified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preflight", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.preflight, args.profile), indent=2) + "\n")
    print("Verified both archive hashes, 4/4 preflight, 1089 focused, nine raw hashes and all eight rank histories")


if __name__ == "__main__":
    main()
