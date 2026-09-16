# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only gates and independent model/cleanup results for the SWA repair."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import regex as re

from tools.dspark import functional_coverage as coverage
from tools.dspark import run_performance_suite as suite
from tools.dspark.profile_attention_validity import AttentionValidity
from tools.dspark.shutdown_acceptance import check
from tools.dspark.startup_cost_profile import diagnostic_points, grid, point_numeric_status, point_samples

LOCAL_ARCHIVE_SHA = "45e7dbe6679aa5192b644d542d0ac16bac67c7c735cd70502cb8cc6800023408"
LOCAL_PLUGIN = "116c26ccaca5acbf4ea9c4d6f0d4be462c0a5513"


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def audit_local(path):
    # Restricted CPU loading only; archive scripts and symlinks are never used.
    import torch

    assert hashlib.sha256(path.read_bytes()).hexdigest() == LOCAL_ARCHIVE_SHA
    with tarfile.open(path) as archive:
        files = {"/".join(m.name.split("/")[1:]): archive.extractfile(m).read() for m in archive if m.isfile()}
    status = dict(line.split("=", 1) for line in files["status.txt"].decode().splitlines())
    assert status == {"MAIN_RC": "0", "PLUGIN": LOCAL_PLUGIN, "CORE": suite.CORE_SHA, "PERFORMANCE_ELIGIBLE": "false"}
    counts = {}
    for stage, expected in (("core", 22), ("lifetime", 3), ("capsule", 23)):
        cases = ET.fromstring(files[stage + ".xml"]).findall(".//testcase")
        assert len(cases) == expected
        assert not any(c.find(k) is not None for c in cases for k in ("failure", "error", "skipped"))
        counts[stage] = len(cases)
    pipes = {n: d.decode().strip() for n, d in files.items() if "/" not in n and n.endswith(".pipestatus")}
    assert len(pipes) == 10 and all(v == "0 0" for v in pipes.values())
    snapshots = []
    for name, data in files.items():
        if not name.startswith("lifetime/") or not name.endswith("/lifetime.pt"):
            continue
        capsule = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        records = [r for r in capsule["records"] if "values" in r]
        assert [r["receipt"].tolist() for r in records] == [[1], [2], [3], [4]]
        assert [r["expected_calls"] for r in records] == [1, 2, 3, 4]
        assert all(torch.equal(r["values"], torch.full_like(r["values"], 3)) for r in records)
        assert capsule["history_page"] not in capsule["write_ids"]
        if capsule["mode"] == "aclgraph":
            capture = capsule["records"][1]
            assert capture == {"stage": "capture", "status": "not_executed", "valid_snapshot": False}
            assert [r["stage"] for r in records] == ["warmup", "replay-1", "replay-2", "replay-3"]
        snapshots.append(
            {"device": capsule["device"], "mode": capsule["mode"], "sha256": hashlib.sha256(data).hexdigest()}
        )
    assert {(s["device"], s["mode"]) for s in snapshots} == {("cpu", "eager"), ("npu", "eager"), ("npu", "aclgraph")}
    return {
        "archive_sha256": LOCAL_ARCHIVE_SHA,
        "versions": status,
        "passed": counts,
        "top_level_pipestatus": pipes,
        "snapshots": snapshots,
        "runtime": json.loads(files["runtime.json"]),
        "local_lifetime_validation": "PASSED",
        "full_model_NaN": "PENDING",
        "worker_natural_exit": "PENDING",
        "remote_note": (
            "Archive fetch log used origin with rzwang22 URL then; inspect current remote configuration anew."
        ),
        "fixture_note": (
            "Nested capsule test artifacts include deliberate nonzero exit codes; not top-level run stages."
        ),
        "performance_eligible": False,
    }


def core_source(root, remote, output):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    result = {"expected_head": suite.CORE_SHA, "selected_remote": remote, "status": "failed"}
    try:
        result["head_before"] = git("rev-parse", "HEAD")
        assert not git("status", "--porcelain"), "Dirty Core tree"
        assert git("branch", "--show-current") == "feat/dspark"
        assert re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", remote), "Invalid remote name"
        result["remote_url"] = git("remote", "get-url", remote)
        result["source_transport"] = (
            "local" if result["remote_url"].startswith(("file:", "/", ".")) else "network_remote"
        )
        result["verification_scope"] = (
            "Exact HEAD through selected remote URL; remote name alone does not verify GitHub provenance"
        )
        result["origin_url_before"] = git("remote", "get-url", "origin")
        git("fetch", remote, "feat/dspark")
        result["fetched_head"] = git("rev-parse", "FETCH_HEAD")
        git("merge-base", "--is-ancestor", suite.CORE_SHA, result["fetched_head"])
        git("merge", "--ff-only", suite.CORE_SHA)
        assert git("rev-parse", "HEAD") == suite.CORE_SHA
        assert git("remote", "get-url", "origin") == result["origin_url_before"]
        result["status"] = "verified"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["actual_head"] = git("rev-parse", "HEAD")
        write(output, result)


def model_report(root, raw_rc):
    def read(name, default):
        path = root / name
        return json.loads(path.read_text()) if path.exists() else default

    plan = read("plan.json", {})
    functional = plan.get("functional_coverage")
    phase = functional.get("phase") if isinstance(functional, dict) else None
    retained = read("retained.json", [])
    completion = read("point-completion.json", {})
    rows, errors = [], []
    for point in plan.get("points", []):
        path = root / (point["id"] + ".json")
        raw = read(path.name, {})
        stream = raw.get("streaming") or {}
        requests = stream.get("requests") or []
        generation = (
            len(requests) == point["requests"]
            and not stream.get("error")
            and all(
                isinstance(r, dict) and not r.get("error") and len(r.get("output_token_ids", [])) == 512
                for r in requests
            )
        )
        if phase:
            generation = generation and coverage.requests_complete(point, stream)
        observed = None
        graph = "UNAVAILABLE"
        numeric = point_numeric_status(raw.get("ranks", []))
        try:
            record = next(r for r in retained if r["point"]["id"] == point["id"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["raw_sha256"]
            assert raw.get("request_identity_validation"), "Request identity validation unavailable"
            selected = point_samples(point, raw["ranks"], 2, 5, 8)
            if phase:
                observed = coverage.observed_layout(point, raw["ranks"], selected, stream)
            graph = "VALIDATED_FULL_SAMPLES"
        except Exception as error:
            errors.append({"point": point["id"], "error": str(error)})
        rows.append(
            {
                **({"functional_coverage": observed} if phase else {}),
                "point": point["id"],
                "generation_complete": bool(generation),
                "tokens": [len(r.get("output_token_ids", [])) if isinstance(r, dict) else None for r in requests],
                "numeric": numeric,
                "graph": graph,
            }
        )
    try:
        gate = AttentionValidity(root / "worker-first-failure", 8)
        gate.check(plan["points"][0]["id"], finished=True)
        first_full = gate.passed
    except Exception as error:
        first_full = False
        errors.append({"early_FULL_gate": str(error)})
    cleanup, workers = read("cleanup.json", {}), read("worker-cleanup.json", {})
    exitcodes = [
        {
            "rank": rank,
            **next(
                (w for w in workers.get("workers", []) if w.get("rank") == rank),
                {"raw_exitcode": None, "exit_status": "unavailable: no worker receipt"},
            ),
        }
        for rank in range(8)
    ]
    natural = (
        cleanup.get("success") is True
        and cleanup.get("timed_out") is False
        and cleanup.get("forced_cleanup") is False
        and workers.get("forced_cleanup") is False
        and len(workers.get("workers", [])) == 8
        and {w.get("rank") for w in workers["workers"]} == set(range(8))
        and all(w["raw_exitcode"] == 0 for w in exitcodes)
    )
    failure = read("profile-failure.json", {})
    # Read saved logs/evidence only, no post-failure RPC and no synthetic exit codes.
    log = root.parent / "b64.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    nan = "Ascend DSpark Markov base logits contain NaN" in text
    owner = "Scheduled candidates lack current proposal owners" in text
    _, expected_points = grid(64, [6, 12, 24, 48, 96, 192, 384], [128, 2048], 512)
    expected_points = diagnostic_points(expected_points, "ctx128-n4-t12-skewed")
    if phase:
        expected_points = coverage.plan(phase)["points"]
    valid_plan = (
        plan.get("core_sha") == suite.CORE_SHA
        and plan.get("performance_eligible") is False
        and plan.get("points") == expected_points
        and len(rows) == len(expected_points)
        and (
            not phase
            or (
                functional == coverage.plan(phase)
                and plan.get("shutdown_policy") == "dspark-profile-25s-v1"
                and read("lifecycle.json", {}).get("engine_initializations") == 1
            )
        )
    )
    generated = valid_plan and all(r["generation_complete"] for r in rows)
    numerical = (
        generated
        and first_full
        and log.exists()
        and not nan
        and not owner
        and all(
            r["numeric"] == "no_nan_observed_at_enabled_boundaries" and r["graph"] == "VALIDATED_FULL_SAMPLES"
            for r in rows
        )
    )
    policy_result = {}
    if plan.get("shutdown_policy"):
        policy_result = check(root, plan["shutdown_policy"])
        policy_result["named_budget_acceptance"] = (
            "PASSED_THIS_RUN"
            if numerical and natural and raw_rc == 0 and policy_result["shutdown_policy_evidence_valid"]
            else "FAILED_OR_UNAVAILABLE"
        )
    return {
        **policy_result,
        "performance_eligible": False,
        "raw_generation_rc": raw_rc,
        "points": rows,
        **(
            {
                "functional_phase": phase,
                "planned_points_generation_complete": generated,
                "prior_baseline": coverage.plan(phase)["prior_baseline"],
                "coverage_plan": functional,
            }
            if phase
            else {"ten_points_generation_complete": generated}
        ),
        "point_acceptance_status": completion.get("status", "unavailable"),
        "numerical_and_FULL_acceptance": "PASSED_THIS_RUN" if numerical else "FAILED_OR_UNAVAILABLE",
        "first_point_FULL_receipts_valid": first_full,
        "markov_NaN_in_log": nan if log.exists() else None,
        "owner_error_in_log": owner if log.exists() else None,
        "primary_failure": failure,
        "cleanup": cleanup,
        "cleanup_failure": read("cleanup-failure.json", {}),
        "worker_force_events": workers.get("force_events"),
        "workers": exitcodes,
        "worker_natural_exit": bool(natural),
        "overall_pass": bool(
            numerical
            and natural
            and raw_rc == 0
            and not plan.get("exit_observation", False)
            and policy_result.get("shutdown_policy_evidence_valid", True)
        ),
        **(
            {
                "exit_observation": True,
                "original_budget_acceptance": "NOT_EVALUATED_BY_THIS_DIAGNOSTIC",
                "natural_exit_with_extended_budget": bool(natural),
            }
            if plan.get("exit_observation", False)
            else {}
        ),
        "evidence_errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    audit = sub.add_parser("audit-local")
    audit.add_argument("archive", type=Path)
    audit.add_argument("output", type=Path)
    source = sub.add_parser("core-source")
    source.add_argument("root", type=Path)
    source.add_argument("remote")
    source.add_argument("output", type=Path)
    report = sub.add_parser("report")
    report.add_argument("root", type=Path)
    report.add_argument("rc", type=int)
    report.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "audit-local":
        write(args.output, audit_local(args.archive))
    elif args.action == "core-source":
        core_source(args.root, args.remote, args.output)
    else:
        result = model_report(args.root, args.rc)
        write(args.output, result)
        if result.get("exit_observation"):
            # A successful diagnostic invocation is not original-budget acceptance.
            completed = (
                result["natural_exit_with_extended_budget"]
                and args.rc == 0
                and result["numerical_and_FULL_acceptance"] == "PASSED_THIS_RUN"
            )
            raise SystemExit(0 if completed else 1)
        raise SystemExit(0 if result["overall_pass"] else 1)


if __name__ == "__main__":
    main()
