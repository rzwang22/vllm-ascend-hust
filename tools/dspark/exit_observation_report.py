# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent diagnostic result; never turns extended-budget exit into acceptance."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def report(root):
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}

    model = read(root / "model-acceptance.json")
    worker = read(root / "runs/b64/worker-cleanup.json")
    native = read(root / "runs/b64/worker-exit/native/observation.json")
    samples = native.get("native_samples", [])
    rows = worker.get("workers", [])
    natural = (
        len(rows) == 8
        and {r.get("rank") for r in rows} == set(range(8))
        and all(r.get("raw_exitcode") == 0 for r in rows)
        and worker.get("forced_cleanup") is False
    )
    state = (
        ("NATURAL_EXIT_NO_DEBUGGER" if not samples else "NATURAL_EXIT_DEBUGGER_AFFECTED")
        if natural
        else "FORCED_OR_INCOMPLETE"
    )
    observed = native.get("status") == "returned" and not worker.get("exit_observation_error")
    no_debugger = native.get("debugger_enabled") is False
    control_valid = None
    if read(root / "runs/b64/plan.json").get("exit_no_debugger"):
        disabled = read(root / "native-preflight-disabled.json")
        control_valid = bool(
            observed
            and no_debugger
            and native.get("attachment_count") == 0
            and not samples
            and native.get("native_sampling") == "disabled_by_configuration"
            and disabled.get("debugger_enabled") is False
            and disabled.get("attachment_count") == 0
            and disabled.get("attach_preflight") == "not_run_by_configuration"
            and not (root / "native-preflight/preflight.json").exists()
        )
        observed = observed and control_valid
    if no_debugger and (samples or native.get("attachment_count") != 0):
        observed = False  # contradictory receipts must not look like a clean control
    if natural and not observed:
        state = "NATURAL_EXIT_OBSERVATION_UNAVAILABLE"
    coverage = "UNAVAILABLE"
    if observed:
        if no_debugger:
            coverage = "DISABLED_BY_CONFIGURATION"
        elif natural and not samples:
            coverage = "NOT_NEEDED_EARLY_EXIT"
        elif samples and all(s["status"] == "captured" and s["detached"] for s in samples):
            coverage = "CAPTURED"
    stages = []
    for path in sorted((root / "runs/b64/worker-exit").glob("*steps.jsonl")):
        stages.extend(
            json.loads(line)
            for line in path.read_text().splitlines()
            if any(key in line for key in ("WorkerProc.shutdown", ".close", "model_runner.shutdown"))
        )
    finalization = [read(p) for p in sorted((root / "runs/b64/worker-exit").glob("*lifetimes-state.json"))]
    ranks = []
    for row in worker.get("workers", []):
        rank = row["rank"]
        returned = next(
            (
                s["utc"]
                for s in stages
                if s.get("rank") == rank and s["stage"] == "WorkerProc.shutdown" and s["event"] == "returned"
            ),
            None,
        )
        polled = native.get("exits", {}).get(str(rank), {})
        joined = next((r for r in worker.get("reap", {}).get("workers", []) if r["rank"] == rank), {})
        end = polled.get("observed_utc") or (
            joined.get("join_finished_utc")
            if joined.get("raw_exitcode") is not None
            else row.get("status_observed_utc")
        )
        if row.get("raw_exitcode") is None:
            end = None
        elapsed = (
            (datetime.fromisoformat(end) - datetime.fromisoformat(returned)).total_seconds()
            if end and returned
            else None
        )
        ranks.append(
            {
                **row,
                "shutdown_returned_utc": returned,
                "last_alive": native.get("last_alive", {}).get(str(rank)),
                "exit_observed_utc": end,
                "join": joined,
                "post_shutdown_observed_seconds_upper_bound": elapsed,
                "timing_source": "parent poll" if polled else "post-Core parent join; coarser upper bound",
            }
        )
    return {
        "performance_eligible": False,
        "diagnostic_only": True,
        "status": state,
        "worker_natural_exit_observed": natural,
        "frontend_cleanup_success": model.get("cleanup", {}).get("success"),
        "formal_acceptance": "NOT_EVALUATED",
        "original_budget_failure_preserved": True,
        "numerical_and_FULL_acceptance": model.get("numerical_and_FULL_acceptance", "UNAVAILABLE"),
        "ten_points_generation_complete": model.get("ten_points_generation_complete"),
        "native_observation": native,
        "worker_cleanup": worker,
        "rank_timings": ranks,
        "explicit_shutdown_stages": stages,
        "finalization": finalization,
        "frontend_cleanup": model.get("cleanup"),
        "native_coverage": coverage,
        "debugger_enabled": native.get("debugger_enabled"),
        "attachment_count": native.get("attachment_count"),
        "no_debugger_control_valid": control_valid,
        "output_handler_shutdown": read(root / "runs/b64/output-handler-shutdown.json"),
        "returncodes": {
            "supervisor": read(root / "runs/b64-supervisor.json").get("raw_returncode"),
            "child_and_log_scan_stages": read(root / "runs/stages.json"),
            "pipestatus_at_report_time": {p.name: p.read_text().strip() for p in sorted(root.glob("*.pipestatus"))},
            "note": "Report/export PIPESTATUS published by enclosing shell after this report; see archived sidecars",
        },
        "limits": "Exit timestamps are parent polling upper bounds. Debugger pause affects elapsed time. "
        "Repeated native frames alone do not prove deadlock; slow progress may exceed this window.",
        "root_cause": "PENDING_REVIEW",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = report(args.root)
    (args.root / "exit-observation-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], result["native_coverage"], "formal acceptance NOT_EVALUATED")
    if result["no_debugger_control_valid"] is False:
        raise SystemExit(1)  # Missing/contradictory control receipts cannot complete successfully.


if __name__ == "__main__":
    main()
