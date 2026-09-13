# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the Fppa5L9e operator-boundary evidence directly from its archive."""

import argparse
import hashlib
import json
import tarfile
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


def verify(profile):
    """Verify archive bytes, completed points, all-rank receipts and error order."""
    files, links = read_archive(
        profile, "384013d562464d278a23ee4c584960110593f1ef05cf6a85cae697f1d9d22b43", 168, 235782231
    )
    assert "1098 passed" in files["focused.log"].decode()
    assert files["focused.pipestatus"].strip() == b"0 0"

    def read(name):
        return json.loads(files["runs/b64/" + name])

    plan = read("plan.json")
    assert plan["plugin_sha"] == "7ac2c4ae9d671a07e868478adaef9be51633a255"
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
            assert row["kv_receipts"] == [e] * 4
            assert row["raw_receipts"] == row["consume_receipts"] == [e] * 3
        d = read(base + "first-nan.json")
        assert d["recording_error"] is None
        for index, numeric in enumerate(d["numeric"]["rounds"]):
            assert numeric["execution"] == 1802 + index and numeric["proposal_epoch"] == 1788 + index
            for column in ("hidden_nan", "logits_nan"):
                assert [v["candidate_row"] for v in numeric["rows"] if v[column]] == (
                    list(range(5)) if index == 2 else []
                )
            assert not any(v["hidden_inf"] or v["logits_inf"] for v in numeric["rows"])
        rounds = d["auxiliary"]["rounds"]
        assert [r["execution"] for r in rounds] == [1802, 1803, 1804]
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
                assert not by_name["layer.1.attention.kv_window"]["nan"]
                assert by_name["layer.1.attention.raw_attention"]["nan"] == (rank < 6)
                assert by_name["layer.1.attention.wo_b_local"]["nan"] == (rank < 6)
                assert by_name["layer.1.attn_output"]["nan"]
            kv = row["target_internal"]["attention"]["kv"]
            assert kv["rows"][0]["before_scatter"]["target_nan"] == 1
            assert kv["receipts"] == [e] * 4
            for item in kv["rows"]:
                assert not item["nonfinite_slots"] and item["invalid_window_indices"] == 0
                after = item["after_scatter"]
                assert after["slot_valid"] == 1
                assert not any(
                    after[k]
                    for k in (
                        "duplicate_slot",
                        "source_nan",
                        "source_inf",
                        "target_nan",
                        "target_inf",
                        "differs_source",
                    )
                )

            state = row["target_internal"]["attention"]["rows"][0]
            assert state["position"] == 220 + i and state["slot_block"] == 119 and state["slot_offset"] == 28 + i
            assert state["invalid_window_indices"] == state["slot_mismatch"] == 0
            evidence.append(
                {
                    "execution": e,
                    "proposal_epoch": row["proposal_epoch"],
                    "state_row0": state,
                    "kv_row0": kv["rows"][0],
                    "nan_rows": {c["name"]: [r["row"] for r in c["rows"] if r["nan"]] for c in cuts},
                }
            )
        assert row["proposal_epoch"] == 1790 and row["query_start_loc_cpu"] == [0, 1, 5, 11]
        assert row["request_ids"][0] == "batch10-3-90e8cc44"
        assert row["target_rows"] == 11 and row["graph_capacity"] == 12
        events = read(base + "error-events.json")["events"]
        assert [v["execution"] for v in events] == [1804, 1805]
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
    failed = read("ctx128-n4-t12-skewed.json")
    outputs = [len(r["output_token_ids"]) for r in failed["streaming"]["requests"]]
    assert outputs == [512, 397, 122, 95]
    cleanup = read("cleanup.json")
    assert cleanup["status"] == "worker_cleanup_incomplete" and cleanup["success"] is False
    return {
        "kind": "independent offline archive verification, not a new NPU run",
        "focused_passed": 1098,
        "ignored_links": links,
        "points": points,
        "failed_point_outputs": outputs,
        "source_log": files["source.log"].decode(),
        "binary_to_source": "UNKNOWN: OPP path only, no loaded binary hashes/build manifest",
        "ranks": ranks,
        "cleanup": cleanup,
        "cleanup_failure": read("cleanup-failure.json"),
        "root_cause": (
            "UNKNOWN: finite semantic window; raw attention NaN on ranks 0-5, local finite on 6-7; no real Q/KV payload"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.profile), indent=2) + "\n")
    print("Verified archive hash, 1098 focused, nine raw hashes and all eight rank histories")


if __name__ == "__main__":
    main()
