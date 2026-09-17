# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only CPU reconstruction of the frozen formal cost archive (no weights loaded)."""

import argparse
import runpy
import shutil
import statistics
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.dspark import confidence_acceptance as acceptance
from tools.dspark import formal_cost as formal
from tools.dspark.profile_request_ids import validate_point_request_ids

OUTER_SHA = "c5ec9d9244191625bab5c7245298f165479b77b8cd378b35bdd1f754b08db5c0"
INNER_SHA = "bcb47fe104152bc184d13862f7be3ecf841c5d184c6aa8849d07c2de4ece61ee"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract(path, destination):
    with tarfile.open(path) as archive:
        for member in archive:
            target = destination / member.name
            require(target.resolve().is_relative_to(destination), "Unsafe archive path")
            require(member.isdir() or member.isfile(), "Unsupported archive member")
            if member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def audit(path):
    """Rebuild all raw points and require independent publication/exit/source receipts."""
    require(formal.sha(path) == OUTER_SHA, "Outer hash mismatch")
    with tempfile.TemporaryDirectory(prefix="dspark-cost-audit-") as tmp:
        base = Path(tmp).resolve()
        extract(path, base)
        outer = base / "dspark-formal-cost.Ta3Z6V3Z"
        inner = outer / "model-evidence.tar.gz"
        require(formal.sha(inner) == INNER_SHA, "Inner hash mismatch")
        extract(inner, base)
        model = base / "dspark-large-batch.8CR50Czp"
        root = model / "runs/b64"
        table, publication = acceptance.publication(root)
        preflight = formal.read(model / "formal-cost-preflight.json")
        saved = formal.read(root / "retained.json")
        require([r["point"] for r in saved] == formal.plan()["points"], "Point manifest mismatch")
        history, requests, samples = {}, 0, 0
        for record in saved:
            point = record["point"]
            rawpath = root / (point["id"] + ".json")
            require(formal.sha(rawpath) == record["raw_sha256"], "Raw hash mismatch")
            raw = formal.read(rawpath)
            require(formal.coverage.requests_complete(point, raw["streaming"]), "Incomplete prompt/output")
            require(
                validate_point_request_ids(point["id"], raw["streaming"], raw["ranks"], history)
                == raw["request_identity_validation"],
                "Request identity differs",
            )
            selected = formal.profile.point_samples(point, raw["ranks"], 2, 5, 8)
            require(selected == record["retained"], "Raw sample reconstruction differs")
            samples += sum(len(r["samples"]) for r in selected)
            requests += point["requests"]
            for rank in raw["ranks"]:
                require(rank["cost_profile"]["identity"] == table["identity"], "Runtime identity differs")
                require(rank["cost_profile"]["observation"] is None, "Diagnostic timings")
                require(
                    rank["confidence_verification"]["weights"] == table["loaded_confidence_weights"],
                    "Head identity differs",
                )
        host = formal.read(root / "scheduler-overhead.json")
        require(
            len(host["samples"]) == 20 and statistics.median(host["samples"]) == table["scheduler_seconds"],
            "Host timing mismatch",
        )
        rebuilt = formal.profile.compile_startup(
            saved,
            table["identity"],
            list(formal.REQUEST_GRID),
            checkpoint=formal.read(root / "checkpoint.json"),
            plugin_sha=acceptance.PRODUCER,
            raw_hashes=[r["raw_sha256"] for r in saved],
            overhead=table["scheduler_seconds"],
        )
        require(
            {**rebuilt, "source": "unpublished_startup_npu_event_profile"}
            == formal.read(root / "cost-profile.pending.json"),
            "Candidate reconstruction differs",
        )
        require(all(table[k] == v for k, v in rebuilt.items()), "Published samples differ")
        require(table["weight_provenance"] == preflight["weights"], "Weight provenance differs")
        policy = runpy.run_path(str(Path(__file__).parents[2] / "vllm_ascend/spec_decode/dspark_verification.py"))
        costs = policy["CostTable"].load_startup(table, table["identity"])
        lookups = 0
        for n in range(1, 65):
            for tokens in range(n, 6 * n + 1):
                costs.cost(n, tokens, 640)
                lookups += 1
        shutdown = formal.shutdown_acceptance.check(root, acceptance.shutdown_policy.POLICY_NAME)
        require(shutdown["shutdown_policy_evidence_valid"], "Shutdown not accepted")
        codes = {
            str(p.relative_to(base)): p.read_text().strip()
            for directory in (outer, model)
            for p in directory.rglob("*.pipestatus")
        }
        require(all(set(c.split()) == {"0"} for c in codes.values()), "Nonzero PIPESTATUS")
        cases = ET.parse(model / "formal-host-tests.xml").getroot().findall(".//testcase")
        require(
            len(cases) == 27
            and all(not any(c.find(k) is not None for k in ("error", "failure", "skipped")) for c in cases),
            "Host tests incomplete",
        )
        workers = formal.read(root / "worker-cleanup.json")
        cleanup = formal.read(root / "cleanup.json")
        require(
            workers["forced_cleanup"] is False
            and workers["force_events"] == []
            and cleanup["timed_out"] is False
            and cleanup["forced_cleanup"] is False,
            "Forced/timed-out cleanup",
        )
        return {
            "status": "INDEPENDENT_CPU_AUDIT_PASSED",
            "outer_sha256": OUTER_SHA,
            "inner_sha256": INNER_SHA,
            "producer_plugin": acceptance.PRODUCER,
            "core": formal.profile.suite.CORE_SHA,
            "cost_sha256": acceptance.TABLE_SHA,
            "points": len(saved),
            "requests": requests,
            "retained_samples": samples,
            "lookup_checks": lookups,
            "publication": publication,
            "shutdown": shutdown,
            "worker_codes": [[w["rank"], w["raw_exitcode"]] for w in workers["workers"]],
            "worker_elapsed_seconds": workers["elapsed_seconds"],
            "frontend_elapsed_seconds": cleanup["elapsed_seconds"],
            "core_source": formal.read(model / "core-source.json"),
            "host_tests": len(cases),
            "pipestatus": codes,
            "weight_verification": (
                "Archived full-shard preflight/publication hashes and loaded head receipts agree; "
                "actual model weight bytes absent locally, not independently rehashed here."
            ),
            "performance_eligible": False,
            "confidence_closed_loop": "NOT_RUN",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acceptance.write(args.output, audit(args.archive))


if __name__ == "__main__":
    main()
