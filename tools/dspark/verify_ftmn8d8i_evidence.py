# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable Ftmn8d8I raw archive assertions, no NPU execution or extraction."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

EXPECTED_SHA = "d74077045654f3a6488dc24462e7b193b7a12f9da330d840734724a64f4ea69f"


def verify(archive):
    """Verify original bytes, all point hashes, all rank histories and truncation."""
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA
    files = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        assert len(members) == 152 and sum(m.size for m in members) == 207083872
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

    def sha(name):
        return hashlib.sha256(files[name]).hexdigest()

    out = {
        "archive": "dspark-large-batch.Ftmn8d8I-evidence.tar.gz",
        "sha256": "d74077045654f3a6488dc24462e7b193b7a12f9da330d840734724a64f4ea69f",
        "members": 152,
        "expanded_bytes": 207083872,
        "kind": "independent offline archive assertions, not a new NPU run",
        "points": [],
        "ranks": [],
        "exit": [],
    }
    p = read("runs/b64/plan.json")
    assert (
        p["plugin_sha"] == "abd554d8d0ed6ff05e54ce0040db10887c9cb942"
        and p["core_sha"] == "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
    )
    out["versions"] = {k: p[k] for k in ["plugin_sha", "core_sha"]}
    ret = read("runs/b64/retained.json")
    assert len(ret) == 9
    for item in ret:
        path = "runs/b64/" + item["point"]["id"] + ".json"
        assert sha(path) == item["raw_sha256"]
        d = read(path)
        assert not d["performance_eligible"] and d["streaming"]["error"] is None
        assert all(x["error"] is None and len(x["output_token_ids"]) == 512 for x in d["streaming"]["requests"])
        out["points"].append(
            {
                "id": item["point"]["id"],
                "raw_sha256": sha(path),
                "requests": len(d["streaming"]["requests"]),
                "tokens_each": 512,
            }
        )
    d = read("runs/b64/ctx128-n4-t12-skewed.json")
    assert [len(x["output_token_ids"]) for x in d["streaming"]["requests"]] == [512, 397, 124, 95]
    out["failed_point"] = {k: d[k] for k in ["point", "streaming"]}
    out["failed_point"]["streaming"] = {k: v for k, v in d["streaming"].items() if k != "requests"}
    out["failed_point"]["requests"] = [
        {k: x.get(k) for k in ["request_id", "error", "finish_reason"]} | {"output_tokens": len(x["output_token_ids"])}
        for x in d["streaming"]["requests"]
    ]
    for rank in range(8):
        base = f"runs/b64/worker-first-failure/rank-{rank}-"
        d = read(base + "first-nan.json")
        assert d["recording_error"] is None and d["numeric"]["classification_counts"] == {
            "both_finite": 97,
            "hidden_nonfinite": 1,
        }
        assert d["numeric"]["compact_host_transfers_completed"] == d["numeric"]["compact_host_transfers"] == 98
        rounds = []
        for i, (a, n) in enumerate(zip(d["auxiliary"]["rounds"], d["numeric"]["rounds"])):
            e = 1802 + i
            epoch = 1790 + i
            assert a["execution"] == n["execution"] == e and a["proposal_epoch"] == n["proposal_epoch"] == epoch
            assert (
                a["coverage"] == a["target_internal"]["coverage"] == "FULL"
                and a["raw_replay_verified"]
                and a["target_mapping_matches_device"]
                and not a["missing_boundaries"]
            )
            ints = a["device_integers"]
            assert (
                ints["raw_receipts"] == ints["consume_receipts"] == [e] * 3
                and ints["target_internal.receipts"] == [e] * 9
            )
            bounds = a["target_internal"]["boundaries"]
            assert len(bounds) == 9
            for j, b in enumerate(bounds):
                assert [v["row"] for v in b["rows"] if v["nan"]] == ([0] if i == 2 and j >= 4 else [])
                assert not any(v["inf"] for v in b["rows"])
            for b in a["boundaries"]:
                expected = [0, 11] if b["boundary"] in ["raw", "persistent"] else [0]
                assert [v["row"] for v in b["rows"] if v["nan"]] == (expected if i == 2 else [])
                assert not any(v["inf"] or v.get("differs_from_raw", False) for v in b["rows"])
            for field in ["hidden_nan", "logits_nan"]:
                assert [v["candidate_row"] for v in n["rows"] if v[field]] == (list(range(5)) if i == 2 else [])
            assert not any(v["hidden_inf"] or v["logits_inf"] for v in n["rows"])
            rounds.append(
                {
                    k: a[k]
                    for k in [
                        "execution",
                        "proposal_epoch",
                        "request_ids",
                        "query_start_loc_cpu",
                        "pool_rows_cpu",
                        "target_rows",
                        "graph_capacity",
                    ]
                }
                | {
                    "integers": {
                        k: v
                        for k, v in ints.items()
                        if k.startswith("target.") or ".1.self_attn" in k or "receipts" in k
                    },
                    "cuts": [
                        {k: b[k] for k in ["name", "shape"]} | {"nan_rows": [v["row"] for v in b["rows"] if v["nan"]]}
                        for b in bounds
                    ],
                    "auxiliary_nan_rows": [
                        {
                            "boundary": b["boundary"],
                            "layer": b["layer"],
                            "nan_rows": [v["row"] for v in b["rows"] if v["nan"]],
                        }
                        for b in a["boundaries"]
                    ],
                }
            )
        assert (
            len(rounds) == 3
            and a["request_ids"] == ["batch10-3-b9c74bbc", "batch10-2-bfad06ee", "batch10-1-8467a66b"]
            and a["query_start_loc_cpu"] == [0, 1, 5, 11]
            and a["target_rows"] == 11
            and a["graph_capacity"] == 12
            and ints["target.positions"][0] == 222
        )
        events = read(base + "error-events.json")["events"]
        assert (
            [x["execution"] for x in events] == [1804, 1805]
            and events[0]["key"][2] == "Ascend DSpark Markov base logits contain NaN."
            and events[1]["key"][2] == "Scheduled candidates lack current proposal owners."
        )
        assert all(
            x["epochs"]["_proposal_step_epoch"] == 1792
            and x["epochs"]["_published_proposal_step_epoch"] is None
            and not x["owner_epochs"]
            for x in events
        )
        first = read(base + "first-failure.json")
        assert first["recording_error"] is None
        labels = [
            "first-nan",
            "first-failure",
            "first-nonfinite",
            "auxiliary-first-nan",
            "auxiliary-first-nonfinite",
            "target-first-nonfinite",
            "error-events",
        ]
        for f in labels[:-1]:
            snap = read(base + f + ".json")
            assert snap["auxiliary"]["rounds"] == d["auxiliary"]["rounds"]
        out["ranks"].append(
            {
                "rank": rank,
                "files": {f: sha(base + f + ".json") for f in labels},
                "rounds": rounds,
                "events": events,
                "transitions": d["transitions"],
            }
        )
        life = next(
            k for k in files if k.startswith(f"runs/b64/worker-exit/rank-{rank}-") and k.endswith("-lifetimes.jsonl")
        )
        ls = [json.loads(x) for x in files[life].splitlines()]
        assert len(ls) == 128 and ls[-1]["event"] == "event_limit_reached"
        out["exit"].append(
            {
                "rank": rank,
                "file": life,
                "raw_sha256": sha(life),
                "event_count": len(ls),
                "last_receipt": ls[-1],
                "coverage": "exhausted; no conclusion about close_fds or final lifetime",
            }
        )
    out["cleanup"] = read("runs/b64/cleanup.json")
    out["cleanup_failure"] = read("runs/b64/cleanup-failure.json")
    out["focused_result"] = next(x for x in files["focused.log"].decode().splitlines() if " passed, " in x)
    out["log_anchors"] = [
        {"line": i + 1, "text": x[:4000]}
        for i, x in enumerate(files["runs/b64.log"].decode().splitlines())
        if any(
            s in x
            for s in [
                "Markov base logits contain NaN",
                "Scheduled candidates lack",
                "force killing",
                "SIGKILL count",
                "enable_flashcomm1 falls",
                "multistream_dsv4",
            ]
        )
    ]
    supervisor = read("runs/b64-supervisor.json")
    command = supervisor["command"]
    assert command[command.index("--plugin-sha") + 1] == out["versions"]["plugin_sha"]
    assert command[command.index("--profile-target-layer") + 1] == "1"
    assert command[command.index("--profile-experiment") + 1] == "target-boundaries"
    assert supervisor["raw_returncode"] == 1 and supervisor["signals_sent"] == []
    assert files["source.pipestatus"].strip() == b"0 0"
    assert "vllm_ascend /workspace/vllm-ascend-hust" in files["source.log"].decode()
    assert "vllm /workspace/vllm-hust" in files["source.log"].decode()
    out["supervisor"] = supervisor
    out["worker_cleanup"] = read("runs/b64/worker-cleanup.json")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = verify(args.archive)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print("Verified archive bytes, nine raw point hashes, all eight rank histories and exit log exhaustion")


if __name__ == "__main__":
    main()
