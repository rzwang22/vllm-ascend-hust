# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconstruct the partial B128 run; never publish it as a cost table."""

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.dspark import formal_cost as formal
from tools.dspark.audit_formal_cost import extract, require

OUTER_SHA = "ffe56dd4d67c6d1f45b1e3f6a7f2256bd37f078d9b42cdeb1f95e8043df9110f"
INNER_SHA = "b0bdc8737e9a01d4d05a300e2e6ad47d2a0f34aaecf3da46fd1911018c8b6b8a"


def audit(archive):
    """Verify both bundles, rebuild retained samples, and separate failure from cleanup."""
    require(formal.sha(archive) == OUTER_SHA, "Outer SHA mismatch")
    with tempfile.TemporaryDirectory(prefix="dspark-sigbus-audit-") as tmp:
        base = Path(tmp).resolve()
        extract(archive, base)
        outer = base / "dspark-batch-expansion.qsQcxCRh"
        embedded = outer / "model-evidence.tar.gz"
        require(formal.sha(embedded) == INNER_SHA, "Inner SHA mismatch")
        extract(embedded, base)
        model = base / "dspark-large-batch.3eW01ldi"
        root = model / "b128/cost/runs/b128"
        plan = formal.read(root / "plan.json")
        require(plan["plugin_sha"] == "3d242eb3a80f942febb64acb52c40051e13bb492", "Producer differs")
        require(plan["core_sha"] == "71d2c1c436eba894a8e9eeb2c5af17e05cb42970", "Core differs")
        cases = ET.parse(model / "host.xml").getroot().findall(".//testcase")
        require(
            len(cases) == 92 and not any(c.find(k) is not None for c in cases for k in ("failure", "error", "skipped")),
            "Host check differs",
        )
        require(formal.read(model / "capacity-interface.json")["status"] == "PASSED_INTERFACE_ONLY", "Interface failed")
        capacities = formal.read(root / "capacity.json")
        require(
            sorted(r["rank"] for r in capacities) == list(range(8))
            and all(r["draft_max_requests"] == 128 for r in capacities),
            "Capacity differs",
        )
        retained = formal.read(root / "retained.json")
        progress = formal.read(root / "point-completion.json")
        require(
            len(plan["points"]) == 48 and len(retained) == len(progress["completed_points"]) == 35,
            "Partial count differs",
        )
        samples = []
        for index, row in enumerate(retained):
            point = row["point"]
            require(point == plan["points"][index], "Point order differs")
            path = root / (point["id"] + ".json")
            require(
                formal.sha(path) == row["raw_sha256"] == progress["completed_points"][index]["raw_sha256"],
                "Raw SHA differs",
            )
            raw = formal.read(path)
            require(
                formal.profile.point_samples(point, raw["ranks"], 2, 5, 8) == row["retained"],
                "Sample reconstruction differs",
            )
            require(formal.coverage.requests_complete(point, raw["streaming"]), "Partial completed point invalid")
            counts = []
            for rank in raw["ranks"]:
                events = rank["cost_profile"]["measurements"]
                require(all(e["point"] == point["id"] for e in events), "Cost event history crosses point")
                counts.append(len(events))
            samples.append(
                dict(
                    point=point["id"],
                    raw_sha256=row["raw_sha256"],
                    file_bytes=path.stat().st_size,
                    rank_measurements=counts,
                )
            )
        failed = formal.read(root / "ctx128-n96-t96-skewed.json")
        require(
            failed["streaming"]["error"] is None and len(failed["streaming"]["requests"]) == 96,
            "Failed-point generation differs",
        )
        require(all(len(r["output_token_ids"]) == 512 for r in failed["streaming"]["requests"]), "Output count differs")
        require("ranks" not in failed, "Unexpected complete snapshot at failed point")
        workers, cleanup = formal.read(root / "worker-cleanup.json"), formal.read(root / "cleanup.json")
        exits = [w["raw_exitcode"] for w in sorted(workers["workers"], key=lambda r: r["rank"])]
        require(
            exits == [0] + [-7] * 7
            and not cleanup["success"]
            and not cleanup["forced_cleanup"]
            and not cleanup["timed_out"],
            "Exit result differs",
        )
        require(not list(model.rglob("cost-profile*.json")), "Unexpected new table")
        anchors = [
            root / name
            for name in (
                "plan.json",
                "retained.json",
                "capacity.json",
                "point-completion.json",
                "engine-failure.json",
                "worker-cleanup.json",
                "cleanup.json",
            )
        ]
        return dict(
            archive_sha256=OUTER_SHA,
            embedded_sha256=INNER_SHA,
            plugin=plan["plugin_sha"],
            core=plan["core_sha"],
            host_tests=92,
            capacity_ranks=[
                {k: r[k] for k in ("rank", "draft_max_requests", "draft_max_tokens", "capture_sizes", "kv_bytes")}
                for r in capacities
            ],
            completed_points=samples,
            failed_point=failed["point"],
            failed_point_generation_complete=True,
            failed_point_snapshot_available=False,
            first_error=formal.read(root / "engine-failure.json"),
            raw_exitcodes=exits,
            cleanup_status=cleanup["status"],
            forced_cleanup=False,
            timed_out=False,
            no_new_cost_table=True,
            later_stages="NOT_RUN",
            original_sigbus_cause="UNKNOWN",
            anchors={str(p.relative_to(model)): formal.sha(p) for p in anchors},
            pipestatus={str(p.relative_to(base)): p.read_text().strip() for p in base.rglob("*.pipestatus")},
            performance_eligible=False,
            overall_pass=False,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.archive), indent=2) + "\n")


if __name__ == "__main__":
    main()
