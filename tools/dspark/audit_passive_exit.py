# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit completed B128 evidence without publishing a failed-cleanup run."""

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.dspark import formal_cost as formal
from tools.dspark.audit_formal_cost import extract, require
from tools.dspark.snapshot_transport_check import files

OUTER_SHA = "f4603a2a0254188cfd0e9172f6920237816b5c2e7946cea2089806dd7d96e817"
INNER_SHA = "00f594d3cacd5c687dca25dccb2f8745e123f3f75072d8127de5462e808c6307"


def audit(root):
    """Rebuild all raw samples and file transfers; retain the actual signal failure."""
    model = root / "dspark-large-batch.Hm043HFG"
    run = model / "b128/cost/runs/b128"
    read = formal.read
    plan, retained, progress = [read(run / n) for n in ("plan.json", "retained.json", "point-completion.json")]
    require(plan["plugin_sha"] == "ebdf9006e55cffdc6d8dc545102dc2f3d67d8d05", "Plugin differs")
    require(plan["core_sha"] == "71d2c1c436eba894a8e9eeb2c5af17e05cb42970", "Core differs")
    require(len(retained) == len(plan["points"]) == len(progress["completed_points"]) == 48, "Plan incomplete")
    cases = ET.parse(model / "host.xml").getroot().findall(".//testcase")
    require(
        len(cases) == 112 and not any(c.find(k) is not None for c in cases for k in ("failure", "error", "skipped")),
        "Host differs",
    )
    transport = read(model / "transport-preflight/transport-report.json")
    require(
        transport["status"] == "PASSED_TRANSPORT_ONLY"
        and transport["source_adapter"] is False
        and transport["exitcodes"] == [0] * 8,
        "Installed transport differs",
    )
    transfers = {}
    for folder in sorted((run / "snapshot-transfers").iterdir()):
        spec, frontend = read(folder / "request.json"), read(folder / "frontend.json")
        require(frontend["status"] == "validated" and spec["point"] not in transfers, "Transfer incomplete/duplicate")
        transfers[spec["point"]] = (folder, frontend)
    require(len(transfers) == 49 and None in transfers, "Initial/point transfers differ")
    initial, front = transfers[None]
    files.restore(run, initial.name, None, front["receipts"], 8)
    points, requests, samples = [], 0, 0
    for index, saved in enumerate(retained):
        point = saved["point"]
        require(point == plan["points"][index], "Plan changed")
        path = run / (point["id"] + ".json")
        digest = formal.sha(path)
        require(digest == saved["raw_sha256"] == progress["completed_points"][index]["raw_sha256"], "Raw hash differs")
        raw = read(path)
        require(formal.coverage.requests_complete(point, raw["streaming"]), "Requests incomplete")
        rebuilt = formal.profile.point_samples(point, raw["ranks"], 2, 5, 8)
        require(rebuilt == saved["retained"], "Retained samples differ")
        folder, frontend = transfers[point["id"]]
        transferred = files.restore(run, folder.name, point["id"], frontend["receipts"], 8)
        # The frontend adds sample_selection after the immutable transfer.
        # Re-run that exact annotator, never discard fields to make comparison pass.
        require(formal.profile.point_samples(point, transferred, 2, 5, 8) == rebuilt, "Transferred samples differ")
        require(transferred == raw["ranks"], "File evidence differs from annotated raw")
        for rank in range(8):
            state = read(folder / f"rank-{rank}-state.json")
            require(
                state["status"] == "committed" and state["receipt"] == frontend["receipts"][rank],
                "Worker commit differs",
            )
        count = len(raw["streaming"]["requests"])
        requests += count
        sample_count = sum(len(r["samples"]) for r in rebuilt)
        samples += sample_count
        points.append(
            dict(
                point=point["id"],
                requests=count,
                raw_sha256=digest,
                retained_samples=sample_count,
                transfer=folder.name,
            )
        )
    require(requests == 2258, "Request total differs")
    workers, cleanup = read(run / "worker-cleanup.json"), read(run / "cleanup.json")
    exits = [r["raw_exitcode"] for r in sorted(workers["workers"], key=lambda r: r["rank"])]
    require(exits == [0, 0, 0, 0, 0, -10, 0, 0], "Exit codes differ")
    require(
        cleanup["status"] == "worker_cleanup_incomplete"
        and not cleanup["success"]
        and not cleanup["forced_cleanup"]
        and not cleanup["timed_out"],
        "Cleanup differs",
    )
    checkpoint = read(run / "worker-exit/parent-checkpoint-0.json")
    row = next(r for r in checkpoint["workers"] if r["rank"] == 5)
    ready = read(run / "worker-exit/rank-5-ready.json")
    masks = dict(
        line.split(":", 1)
        for line in row["procfs"]["status"]["text"].splitlines()
        if line.startswith(("SigCgt:", "SigIgn:"))
    )
    require(all(not int(value.strip(), 16) & (1 << 9) for value in masks.values()), "SIGUSR1 disposition differs")
    require(
        ready["signal_registered"]
        and ready["pid"] == row["pid"] == 659146
        and row["stack_request"]["signal_sent"] == 10,
        "Signal identity differs",
    )
    stack = run / "worker-exit" / ready["stack_file"]
    require(stack.stat().st_size == 0, "Unexpected stack data")
    steps = [json.loads(line) for line in (run / "worker-exit/rank-5-pid-659146-steps.jsonl").read_text().splitlines()]
    returned = next(r for r in steps if r["stage"] == "WorkerProc.shutdown" and r["event"] == "returned")
    require(not list(run.glob("cost-profile*.json")), "Unexpected cost table")
    residual = read(run.parent / "b128-residual.json")
    expansion = read(model / "expansion-report.json")
    require(residual["success"] is True and residual["error"] is None, "Residual check differs")
    require(
        expansion["overall_pass"] is False and expansion["stages"][-1]["name"] == "b128-cost", "Stage order differs"
    )
    return dict(
        outer_sha256=OUTER_SHA,
        inner_sha256=INNER_SHA,
        plugin=plan["plugin_sha"],
        core=plan["core_sha"],
        host_tests=112,
        installed_transport="PASSED_TRANSPORT_ONLY",
        completed_points=points,
        requests=requests,
        retained_samples=samples,
        validated_transfers=49,
        validated_rank_files=392,
        collection="COMPLETE_NOT_PUBLISHED",
        cleanup_status=cleanup["status"],
        raw_exitcodes=exits,
        signal_evidence=dict(
            ready=ready,
            handler_masks=masks,
            shutdown_returned_utc=returned["utc"],
            checkpoint_elapsed_seconds=checkpoint["elapsed_seconds"],
            sent=row["stack_request"],
            stack_bytes=0,
        ),
        worker_elapsed_seconds=workers["elapsed_seconds"],
        frontend_elapsed_seconds=cleanup["elapsed_seconds"],
        forced_cleanup=False,
        timed_out=False,
        residual=residual,
        stages=[{k: s[k] for k in ("name", "rc", "elapsed_seconds")} for s in expansion["stages"]],
        first_error=expansion["first_error"],
        overall_pass=False,
        original_sigbus_cause="UNKNOWN",
        sigbus_this_run="NOT_REPRODUCED",
        handler_removal_call="NOT_CAPTURED",
        later_stages="NOT_RUN",
        pipestatus={str(p.relative_to(root)): p.read_text().strip() for p in root.rglob("*.pipestatus")},
        performance_eligible=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(formal.sha(args.archive) == OUTER_SHA, "Outer hash differs")
    with tempfile.TemporaryDirectory(prefix="dspark-passive-audit-") as tmp:
        root = Path(tmp).resolve()
        extract(args.archive, root)
        inner = root / "dspark-batch-expansion.IF4hUVja/model-evidence.tar.gz"
        require(formal.sha(inner) == INNER_SHA, "Embedded hash differs")
        extract(inner, root)
        args.output.write_text(json.dumps(audit(root), indent=2) + "\n")


if __name__ == "__main__":
    main()
