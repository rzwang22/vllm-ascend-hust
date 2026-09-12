# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline assertions for the immutable vl70uUqj archive; no server execution."""

import argparse
import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath

EXPECTED_SHA = "8a07cd9f587479f89a4730ff222f9a970277739ed80d7c84d1eda2713f275910"


def verify(archive):
    """Verify raw bytes, point/rank receipts, then time-order shutdown evidence."""
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA
    files = {}
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        assert len(members) == 91 and sum(m.size for m in members) == 51153103
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
        "archive": "dspark-large-batch.vl70uUqj-evidence.tar.gz",
        "sha256": "8a07cd9f587479f89a4730ff222f9a970277739ed80d7c84d1eda2713f275910",
        "members": 91,
        "expanded_bytes": 51153103,
        "assertion_kind": "offline original archive verification, not a new NPU test",
        "points": [],
        "ranks": [],
    }
    plan = read("runs/b64/plan.json")
    assert (
        plan["plugin_sha"] == "06b2f60b7023a4bb675809775878ee7d629b6e60"
        and plan["core_sha"] == "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
    )
    out["versions"] = {k: plan[k] for k in ["plugin_sha", "core_sha"]}
    ret = read("runs/b64/retained.json")
    assert len(ret) == len(plan["points"]) == 10
    for item in ret:
        p = item["point"]["id"]
        h = sha(f"runs/b64/{p}.json")
        assert h == item["raw_sha256"]
        raw = read(f"runs/b64/{p}.json")
        assert raw["streaming"]["error"] is None and not raw["performance_eligible"]
        requests = raw["streaming"]["requests"]
        assert len(requests) == item["point"]["requests"]
        assert all(
            x["error"] is None and len(x["output_token_ids"]) == 512 and x["finish_reason"] == "length"
            for x in requests
        )
        assert len(raw["ranks"]) == 8
        for rank in raw["ranks"]:
            obs = rank["cost_profile"]["observation"]
            assert obs["recording_error"] is None and obs["target_internal"]["target_layer"] == 1
            assert (
                obs["numeric"]["nan_rounds"]
                == obs["target_internal"]["counts"]["nonfinite_rounds"]
                == obs["auxiliary"]["counts"]["nonfinite_rounds"]
                == 0
            )
            assert obs["numeric"]["compact_host_transfers_completed"] == obs["numeric"]["compact_host_transfers"]
        out["points"].append(
            {
                "id": p,
                "raw_sha256": h,
                "requests": len(requests),
                "output_tokens_each": 512,
                "stream_error": None,
                "all_eight_rank_observation_counters_checked": True,
            }
        )
    for rank in range(8):
        path = f"runs/b64/worker-first-failure/rank-{rank}-latest.json"
        d = read(path)
        assert d["recording_error"] is None and d["numeric"]["nan_rounds"] == 0
        obs = raw["ranks"][rank]["cost_profile"]["observation"]
        assert obs["rank"] == rank and obs["local_evidence"]["sha256"] == sha(path)
        assert (
            d["numeric"]["classification_counts"] == {"both_finite": 203}
            and d["numeric"]["compact_host_transfers_completed"] == 203
        )
        assert d["target_internal"]["counts"] == {"NON_FULL_UNOBSERVED": 3, "nonfinite_rounds": 0, "FULL": 200}
        rounds = []
        for a, n in zip(d["auxiliary"]["rounds"], d["numeric"]["rounds"]):
            e = a["execution"]
            p = a["proposal_epoch"]
            assert e == n["execution"] and p == n["proposal_epoch"]
            assert (
                a["coverage"] == a["target_internal"]["coverage"] == "FULL"
                and a["raw_replay_verified"]
                and not a["missing_boundaries"]
                and a["target_mapping_matches_device"]
            )
            ints = a["device_integers"]
            assert (
                ints["raw_receipts"] == ints["consume_receipts"] == [e] * 3
                and ints["target_internal.receipts"] == [e] * 9
            )
            assert len(a["target_internal"]["boundaries"]) == 9 and len(a["boundaries"]) == 9
            for b in a["boundaries"] + a["target_internal"]["boundaries"]:
                assert all(not x["nan"] and not x["inf"] and not x.get("differs_from_raw", False) for x in b["rows"])
            assert n["classification"] == "both_finite"
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
                        "raw_replay_verified",
                        "coverage",
                    ]
                }
                | {
                    "receipts": {k: ints[k] for k in ["raw_receipts", "consume_receipts", "target_internal.receipts"]},
                    "all_recorded_rows_finite": True,
                }
            )
        assert len(rounds) == 3 and all(
            b["execution"] == a["execution"] + 1 and b["proposal_epoch"] == a["proposal_epoch"] + 1
            for a, b in zip(rounds, rounds[1:])
        )
        out["ranks"].append(
            {
                "rank": rank,
                "latest_sha256": sha(path),
                "counts": d["target_internal"]["counts"],
                "numeric_counts": d["numeric"]["classification_counts"],
                "rounds": rounds,
            }
        )
    for name in [
        "runs/b64/cleanup.json",
        "runs/b64/lifecycle.json",
        "runs/b64/engine-failure.json",
        "runs/b64/diagnostic.json",
        "runs/b64-supervisor.json",
    ]:
        out[name] = read(name)
    c = out["runs/b64/cleanup.json"]
    assert (
        c["engine_returned"]
        and c["thread_completed"]
        and not c["timed_out"]
        and c["event_loop"] == "closed"
        and c["status"] == "forced_cleanup"
        and not c["success"]
    )
    assert 12 < c["elapsed_seconds"] < 12.1
    for name in ["worker-cleanup", "cleanup-thread", "cleanup-failure", "point-completion"]:
        out[f"runs/b64/{name}.json"] = read(f"runs/b64/{name}.json")
    w = out["runs/b64/worker-cleanup.json"]
    assert c["worker_cleanup"] == w
    assert [r["raw_exitcode"] for r in w["workers"]] == [None, -15, -15, -15, None, -15, None, -15]
    assert out["runs/b64/cleanup-failure.json"]["prior_error"] is None
    progress = out["runs/b64/point-completion.json"]
    assert progress["status"] == "completed" and len(progress["completed_points"]) == 10
    assert all(
        p["numeric_result"] == "no_nan_observed_at_enabled_boundaries"
        and p["generated_tokens"] == [512] * q["requests"]
        and p["raw_sha256"] == q["raw_sha256"]
        for p, q in zip(progress["completed_points"], out["points"])
    )
    assert (
        out["runs/b64-supervisor.json"]["raw_returncode"] == 1 and out["runs/b64-supervisor.json"]["signals_sent"] == []
    )
    assert files["status.txt"].decode().startswith("MAIN_RC=1")
    assert "runs/b64/cost-profile.json" not in files
    assert len([p for p in files if p.startswith("runs/b64/worker-first-failure/")]) == 8
    out["shutdown_log_anchors"] = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(files["runs/b64.log"].decode().splitlines())
        if "[shutdown]" in line or "VERIFICATION_FAILED=" in line or "EngineDeadError" in line
    ]
    assert any("force killing" in x["text"] for x in out["shutdown_log_anchors"])
    out["focused_result"] = next(line for line in files["focused.log"].decode().splitlines() if " passed, " in line)
    out["numeric_result"] = "not reproduced in observed boundaries; not a repair"
    out["cleanup_result"] = (
        "Explicit worker cleanup returned; all workers required SIGTERM, ranks 0/4/6 also SIGKILL. "
        "Natural exit unproven."
    )
    out["residual_check"] = (
        "No separate residual-process enumeration in archive; supervisor signals_sent=[] is consistent "
        "with group absent when checked, not proof of graceful exit."
    )

    request = read("runs/b64/worker-exit/request.json")
    start = datetime.fromisoformat(request["started_utc"])
    check = read("runs/b64/worker-exit/parent-checkpoint-1.json")
    pre1 = read("runs/b64/worker-exit/parent-before-escalation-1.json")
    pre2 = read("runs/b64/worker-exit/parent-before-escalation-2.json")
    assert [w["raw_exitcode"] for w in pre1["workers"]] == [None] * 8
    out["exit_stages"] = []
    for rank in range(8):
        ready = read(f"runs/b64/worker-exit/rank-{rank}-ready.json")
        pid = ready["pid"]
        prefix = f"runs/b64/worker-exit/rank-{rank}-pid-{pid}"
        rows = [json.loads(line) for line in files[prefix + "-steps.jsonl"].splitlines()]
        assert len(rows) == 181 and not any(r["event"] == "error" for r in rows)
        last = rows[-1]
        assert (last["stage"], last["event"]) == ("WorkerProc.shutdown", "returned")
        assert last["request"] == request["id"] and last["point"] == request["point"]
        elapsed = (datetime.fromisoformat(last["utc"]) - start).total_seconds()
        assert 3 < elapsed < 3.33
        for stage in ["model_runner.shutdown", "python.gc_collect", "distributed.destroy_distributed_environment"]:
            assert any(r["stage"] == stage and r["event"] == "returned" for r in rows)
        stack = files[prefix + "-stacks.txt"]
        n1 = pre1["workers"][rank]["stack_bytes_available_now"]
        n2 = pre2["workers"][rank]["stack_bytes_available_now"]
        assert len(stack) == n2 and "destroy_process_group" in stack[:n1].decode()
        second = stack[n1:n2].decode()
        item = {
            "rank": rank,
            "pid": pid,
            "events": len(rows),
            "last_step": last,
            "return_seconds_from_request": elapsed,
            "stack_bytes_before_term": n1,
            "stack_bytes_before_kill": n2,
            "appended_stack_after_term": second,
            "raw_exitcode": w["workers"][rank]["raw_exitcode"],
        }
        if rank in (0, 4, 6):
            assert n2 - n1 == 101 and "Garbage-collecting" in second and "<no Python frame>" in second
            worker = check["workers"][rank]
            assert worker["raw_exitcode"] is None
            thread = next(t for t in worker["procfs"]["threads"] if t["tid"] == pid)
            assert "State:\tR (running)" in thread["status"]["text"] and thread["wchan"]["text"] == "0"
            item["post_term_main_thread"] = {
                "state": "R",
                "wchan": "0",
                "checkpoint_elapsed_seconds": check["elapsed_seconds"],
            }
        else:
            assert not second and check["workers"][rank]["raw_exitcode"] == -15
        out["exit_stages"].append(item)
    out["checkpoint_times"] = {d["label"]: d["elapsed_seconds"] for d in [pre1, check, pre2]}
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(args.archive), indent=2) + "\n")
