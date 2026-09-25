# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline audit of the accepted expansion, never a model execution entry."""

import argparse
import json
import tempfile
from pathlib import Path

from tools.dspark import confidence_acceptance as acceptance
from tools.dspark import formal_cost as formal
from tools.dspark.audit_formal_cost import extract, require
from tools.dspark.performance_comparison import COSTS, PRODUCER
from tools.dspark.profile_request_ids import validate_point_request_ids

OUTER_SHA = "702e9ab73c4032fddd7bde50a2921875291f4e1062788b72013b9e2247a64d09"
MODEL_SHA = "98e744afa4b921d2a1bfc19456e37fe93856f6f78974bde5fb6472266a57fff3"


def audit(outer_path, model_path):
    """Reconstruct samples and execution first, then independently require clean exits."""
    require(formal.sha(outer_path) == OUTER_SHA and formal.sha(model_path) == MODEL_SHA, "Archive SHA differs")
    with tempfile.TemporaryDirectory(prefix="dspark-performance-audit-") as tmp:
        base = Path(tmp).resolve()
        extract(outer_path, base)
        outer = base / "dspark-batch-expansion.XUJieSKP"
        require(formal.sha(outer / "model-evidence.tar.gz") == MODEL_SHA, "Embedded archive differs")
        extract(model_path, base)
        model = base / "dspark-large-batch.6tf75BWi"
        report = {
            "outer_sha256": OUTER_SHA,
            "model_sha256": MODEL_SHA,
            "plugin": PRODUCER,
            "core": acceptance.suite.CORE_SHA,
            "tiers": [],
            "scope": "CPU reconstruction of archived results; no NPU execution",
        }
        for batch in (128, 256):
            root = model / f"b{batch}/cost/runs/b{batch}"
            confidence = model / f"b{batch}/confidence"
            cr = confidence / f"runs/b{batch}"
            table, _ = acceptance.publication(root, batch, PRODUCER)
            require(formal.sha(root / "cost-profile.json") == COSTS[batch][1], "Cost hash differs")
            saved = formal.read(root / "retained.json")
            require([r["point"] for r in saved] == formal.plan(batch)["points"], "Cost plan differs")
            requests = samples = 0
            history = {}
            for record in saved:
                point = record["point"]
                path = root / (point["id"] + ".json")
                require(formal.sha(path) == record["raw_sha256"], "Raw hash differs")
                raw = formal.read(path)
                require(formal.coverage.requests_complete(point, raw["streaming"]), "Request completion failed")
                require(
                    validate_point_request_ids(point["id"], raw["streaming"], raw["ranks"], history)
                    == raw["request_identity_validation"],
                    "Mapping differs",
                )
                selected = formal.profile.point_samples(point, raw["ranks"], 2, 5, 8)
                require(selected == record["retained"], "Sample reconstruction differs")
                samples += sum(len(r["samples"]) for r in selected)
                requests += point["requests"]
                for rank in raw["ranks"]:
                    require(
                        rank["cost_profile"]["identity"] == table["identity"]
                        and rank["cost_profile"]["observation"] is None,
                        "Timing identity differs",
                    )
            rebuilt = formal.profile.compile_startup(
                saved,
                table["identity"],
                formal.request_grid(batch),
                checkpoint=formal.read(root / "checkpoint.json"),
                plugin_sha=PRODUCER,
                raw_hashes=[r["raw_sha256"] for r in saved],
                overhead=table["scheduler_seconds"],
            )
            require(all(table[k] == v for k, v in rebuilt.items()), "Published table differs from raw reconstruction")
            require(
                table["weight_provenance"]
                == formal.read(model / f"b{batch}/cost/formal-cost-preflight.json")["weights"],
                "Weight identity differs",
            )
            records = [json.loads(line) for line in (confidence / "input.jsonl").read_text().splitlines()]
            require(
                formal.real_text_contract(confidence / "input/manifest.json") == formal.workload_contract(),
                "Frozen input manifest differs",
            )
            _, original, _ = formal.read_manifest(confidence / "input/manifest.json", 64)
            require(records == original * (batch // 64), "Actual tokens/order differ from frozen questions")
            preflight = formal.read(confidence / "preflight.json")
            require(
                formal.sha(confidence / "input.jsonl") == preflight["input_sha256"]
                and table["weight_provenance"] == preflight["weights"],
                "Confidence input/weight hashes differ",
            )
            stream = formal.read(cr / "stream.json")
            acceptance.validate_stream(stream, records, batch)
            execution = acceptance.validate_receipts(formal.read(cr / "after.json"), stream, table, batch)
            require(
                json.loads(json.dumps(execution))
                == formal.read(confidence / "confidence-acceptance.json")["generation"]["execution_acceptance"],
                "Execution differs",
            )
            exits = []
            for directory in (root, cr):
                require(
                    acceptance.shutdown_acceptance.check(directory, acceptance.shutdown_policy.POLICY_NAME)[
                        "shutdown_policy_evidence_valid"
                    ],
                    "Shutdown invalid",
                )
                acceptance.shutdown_acceptance.require_passive(directory)
                codes = sorted(
                    (w["rank"], w["raw_exitcode"]) for w in formal.read(directory / "worker-cleanup.json")["workers"]
                )
                require(codes == [(r, 0) for r in range(8)], "Worker exit failed")
                exits.append(codes)
            report["tiers"].append(
                {
                    "batch": batch,
                    "cost_points": len(saved),
                    "cost_requests": requests,
                    "samples": samples,
                    "cost_sha256": COSTS[batch][1],
                    "confidence": {k: v for k, v in execution.items() if k != "per_request"},
                    "worker_exitcodes": exits,
                }
            )
        codes = [p.read_text().split() for directory in (outer, model) for p in directory.rglob("*.pipestatus")]
        require(codes and all(c and set(c) == {"0"} for c in codes), "Nonzero PIPESTATUS")
        report["pipestatus_files"] = len(codes)
        report["core_source"] = formal.read(model / "core-source.json")
        report["expansion_report"] = {
            k: formal.read(model / "expansion-report.json")[k]
            for k in ("overall_pass", "first_error", "performance_eligible")
        }
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outer", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acceptance.write(args.output, audit(args.outer, args.model))


if __name__ == "__main__":
    main()
