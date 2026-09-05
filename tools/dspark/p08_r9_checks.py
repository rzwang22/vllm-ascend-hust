# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""P08-R9 evidence status, separate from generation and root-cause claims."""

import json
import sys
from pathlib import Path

if __package__:
    from . import p08_r8_checks as common
else:
    # Also collect setup failures before the source gate exports PYTHONPATH.
    import p08_r8_checks as common


def diagnostics(root: Path, generation_rc: int, result_rc: int) -> dict:
    ranks, missing, insufficient = [], [], []
    for rank in common.RANKS:
        first = root / "rank-diagnostics" / f"rank-{rank}-first-failure.json"
        latest = first.with_name(f"rank-{rank}-latest.json")
        if not latest.is_file():
            missing.append(rank)
            continue
        report = json.loads((first if first.is_file() else latest).read_text())
        if report.get("rank") != rank or report.get("performance_eligible") is not False:
            insufficient.append(rank)
        current = report["current"]
        detail = current.get("replay_detail", {})
        configuration = report.get("replay_configuration", {})
        if first.is_file():
            covered = detail.get("status") == "ACTUAL_FULL_REPLAY_SNAPSHOTS"
            previous = report.get("previous_executions", [])
            if current.get("target_execution_epoch", 0) >= 3:
                covered = (
                    covered
                    and len(previous) == 2
                    and all(
                        item.get("replay_detail", {}).get("status") == "ACTUAL_FULL_REPLAY_SNAPSHOTS"
                        for item in previous
                    )
                )
        else:
            covered = configuration.get("completed_detailed_replays", 0) > 0
        covered = covered and configuration.get("unavailable_detailed_replays", 0) == 0
        if not covered:
            insufficient.append(rank)
        ranks.append(
            {
                "rank": rank,
                "first_failure": first.is_file(),
                "phase": current.get("phase"),
                "epoch": current.get("target_execution_epoch"),
                "stage": current.get("stage"),
                "detail_status": detail.get("status", "unavailable"),
                "request_ids": current.get("request_ids"),
                "request_mapping": detail.get("pre_replay", {}).get("request_mapping"),
                "graph_shape": detail.get("graph_shape"),
                "localization": detail.get("localization", {"status": "unavailable"}),
                "input_mismatches": detail.get("pre_replay", {}).get("mismatches"),
                "unavailable": detail.get("pre_replay", {}).get("unavailable"),
                "previous_executions": [
                    {
                        "epoch": item["target_execution_epoch"],
                        "phase": item["phase"],
                        "request_ids": item.get("request_ids"),
                        "detail_status": item.get("replay_detail", {}).get("status", "unavailable"),
                        "localization": item.get("replay_detail", {}).get("localization"),
                    }
                    for item in report.get("previous_executions", [])
                ],
                "configuration": configuration,
                "preserved_window_file": str(first.with_name(f"rank-{rank}-window.json"))
                if first.with_name(f"rank-{rank}-window.json").is_file()
                else None,
                "exception": current.get("exception"),
            }
        )
    generation_status = (
        "NOT_RUN"
        if generation_rc == 99
        else "FAILED"
        if generation_rc
        else "COMPLETED"
        if result_rc == 0
        else "RETURNED_BUT_RESULT_GATE_FAILED"
    )
    evidence_status = "PARTIAL" if missing else "COVERAGE_INSUFFICIENT" if insufficient else "AVAILABLE"
    if evidence_status != "AVAILABLE":
        localization_status = "UNAVAILABLE"
    elif any(row["localization"].get("status") == "OBSERVED_BOUNDARY_BRACKET" for row in ranks if row["first_failure"]):
        localization_status = "OBSERVED_BOUNDARY_BRACKET"
    elif generation_status == "COMPLETED" and not any(row["first_failure"] for row in ranks):
        localization_status = "NOT_REPRODUCED_OBSERVER_MAY_PERTURB"
    else:
        localization_status = "UNAVAILABLE"
    result = {
        "generation_status": generation_status,
        "diagnostic_evidence_status": evidence_status,
        "localization_status": localization_status,
        "root_cause_status": "ROOT_CAUSE_NOT_YET_PROVEN",
        "performance_eligible": False,
        "missing_ranks": missing,
        "insufficient_ranks": sorted(set(insufficient)),
        "ranks": ranks,
    }
    (root / "diagnostic-index.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "ranks"}, indent=2))
    return result


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostics":
        result = diagnostics(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
        if result["diagnostic_evidence_status"] != "AVAILABLE":
            raise RuntimeError("Replay localization evidence is incomplete; retained all available rank files.")
    elif len(sys.argv) > 1 and sys.argv[1] == "source":
        common.source(Path(sys.argv[2]), Path(sys.argv[3]))
        # Worker-only import verification, never reached by the Mac CPU fixtures.
        import inspect

        from vllm_ascend.diagnostics.dspark_replay import TargetLayerSnapshots

        expected = Path(sys.argv[2]) / "vllm_ascend/diagnostics/dspark_replay.py"
        assert Path(inspect.getfile(TargetLayerSnapshots)).resolve() == expected.resolve()
    else:
        common.main()


if __name__ == "__main__":
    main()
