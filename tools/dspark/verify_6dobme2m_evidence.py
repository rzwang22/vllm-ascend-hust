# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify original receipt-failure archive without extraction or execution."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

EXPECTED_SHA = "e7e6021068ce6048c0b6856a962f0fce1f3ebf88fab132896f1c8f064912197e"


def verify(archive):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA
    files = {}
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == 98 and sum(m.size for m in members) == 28416185
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert member.isfile() or member.isdir()
            if member.isfile():
                name = str(PurePosixPath(*path.parts[1:]))
                assert name not in files
                files[name] = tar.extractfile(member).read()

    def read(name):
        return json.loads(files[name])

    plan = read("runs/b64/plan.json")
    assert plan["plugin_sha"] == "9af2b22b54d74029647e5e7ad040cf03d740bd96"
    assert plan["core_sha"] == "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
    point_path = "runs/b64/ctx128-n1-t6-balanced.json"
    point = read(point_path)
    assert point["performance_eligible"] is False and point["snapshot_error"]
    assert point["streaming"]["error"] is None
    assert [len(x["output_token_ids"]) for x in point["streaming"]["requests"]] == [512]
    assert all(x["error"] is None for x in point["streaming"]["requests"])
    assert "runs/b64/ctx128-n4-t12-skewed.json" not in files
    ranks = []
    for rank in range(8):
        name = f"runs/b64/worker-first-failure/rank-{rank}-latest.json"
        data = read(name)
        rounds = data["auxiliary"]["rounds"]
        assert [r["execution"] for r in rounds] == [86, 87, 88]
        states = []
        for row in rounds:
            e = row["execution"]
            ints = row["device_integers"]
            assert ints["target_internal.receipts"] == [e] * 9 + [-1] * 12
            assert ints["raw_receipts"] == ints["consume_receipts"] == [e] * 3
            assert row["raw_replay_verified"] and row["coverage"] == "FULL"
            state = ints["target_internal.attention_state"]
            states.append(state)
        assert states[0] != states[1] != states[2]
        ranks.append(
            {
                "rank": rank,
                "file": name,
                "sha256": hashlib.sha256(files[name]).hexdigest(),
                "executions": [86, 87, 88],
                "outer_current": 9,
                "attention_invalid": 12,
                "attention_state_first_rows": [s[:10] for s in states],
                "recording_error": data["recording_error"],
            }
        )
    cleanup = read("runs/b64/cleanup.json")
    assert cleanup["status"] == "forced_cleanup" and not cleanup["success"]
    workers = read("runs/b64/worker-cleanup.json")
    assert [w["raw_exitcode"] for w in workers["workers"]] == [None, None, None, -15, -15, None, None, None]
    supervisor = read("runs/b64-supervisor.json")
    assert supervisor["raw_returncode"] == 1
    assert files["source.pipestatus"].strip() == b"0 0"
    compiler_anchors = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(files["runs/b64.log"].decode().splitlines())
        if "[compiler_interface.py:" in line
    ]
    assert sum("enable_npugraph_ex': True" in x["text"] for x in compiler_anchors) == 8
    return {
        "compiler_anchors": compiler_anchors,
        "archive_sha256": EXPECTED_SHA,
        "members": 98,
        "expanded_bytes": 28416185,
        "kind": "independent offline assertions; not NPU validation",
        "versions": {k: plan[k] for k in ("plugin_sha", "core_sha")},
        "first_point": {
            "raw_sha256": hashlib.sha256(files[point_path]).hexdigest(),
            "tokens": 512,
            "generation_error": None,
            "snapshot_error": point["snapshot_error"],
        },
        "original_nan_scenario": "not reached",
        "ranks": ranks,
        "cleanup": cleanup,
        "worker_cleanup": workers,
        "cleanup_failure": read("runs/b64/cleanup-failure.json"),
        "supervisor": supervisor,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.archive), indent=2) + "\n")
    print("Verified archive, first point, all eight rank receipts and independent forced cleanup")


if __name__ == "__main__":
    main()
