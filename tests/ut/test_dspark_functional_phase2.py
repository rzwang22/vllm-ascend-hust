# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only phase2 entry/report gates; no claim of NPU coverage."""

import json
from pathlib import Path

import pytest

from tests.ut import test_dspark_functional_coverage as fixtures
from tools.dspark import functional_coverage as coverage
from tools.dspark import run_large_batch as large
from tools.dspark import startup_cost_profile as profile
from tools.dspark import swa_acceptance as acceptance

ROOT = Path(__file__).parents[2]


def test_exact_seven_legal_points_and_preserved_phase1():
    one = coverage.plan(coverage.PHASE)
    assert one == json.loads((ROOT / "tools/dspark/B64_FUNCTIONAL_PHASE1.json").read_text())
    two = coverage.plan(coverage.PHASE2)
    assert two == json.loads((ROOT / "tools/dspark/B64_FUNCTIONAL_PHASE2.json").read_text())
    _, matrix = profile.grid(64, list(coverage.CAPTURES), [128, 2048], 512)
    assert (two["point_count"], two["total_requests"], two["total_output_tokens"]) == (7, 288, 147456)
    assert two["max_runtime_seconds"] == large.TARGET_DIAGNOSTIC_RUNTIME_SECONDS == 3600
    assert [p["id"] for p in two["points"]] == [
        "ctx2048-n16-t48-balanced",
        "ctx2048-n16-t48-skewed",
        "ctx2048-n32-t96-balanced",
        "ctx2048-n32-t96-skewed",
        "ctx2048-n64-t192-balanced",
        "ctx2048-n64-t192-skewed",
        "ctx2048-n64-t384-balanced",
    ]
    assert coverage.select(matrix, coverage.PHASE2) == two["points"]
    assert not set(coverage.PHASE2_POINT_IDS) & (set(coverage.POINT_IDS) | {p["id"] for p in matrix[:10]})
    assert all(
        p in matrix
        and p["prompt_tokens"] == 2048
        and len(p["lengths"]) == p["requests"]
        and all(0 <= x <= 5 for x in p["lengths"])
        and sum(x + 1 for x in p["lengths"]) == p["actual_tokens"] == p["capacity"]
        for p in two["points"]
    )
    assert two["remaining_matrix_is_not_a_required_checklist"] is True
    assert not two["performance_eligible"] and two["model_initializations"] == 1


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--profile-operator-capture"],
        ["--profile-write-timeline"],
        ["--profile-exit-observation"],
        ["--profile-coverage-phase", "b64-functional-3"],
        ["--profile-contexts", "2048"],
        ["--profile-output-tokens", "511"],
    ],
)
def test_phase2_actual_driver_cli(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(large, "run", lambda args: args)
    args = fixtures.cli(tmp_path)
    args[args.index("--profile-coverage-phase") + 1] = coverage.PHASE2
    if extra:
        with pytest.raises(SystemExit):
            large.main(args + extra)
    else:
        config = large.main(args)
        command = large.command(config, 64, tmp_path)
        assert command[command.index("--profile-coverage-phase") + 1] == coverage.PHASE2
        assert "--profile-stop-after-point" not in command


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "partial_request",
        "short_input",
        "short_output",
        "fewer_scheduled",
        "stale_FULL",
        "NaN",
        "cleanup",
        "plan",
        "second_engine",
    ],
)
def test_phase2_real_report_and_closed_baselines(tmp_path, problem):
    fixtures.test_report_requires_real_full_concurrency_and_preserves_closed_baseline(
        tmp_path, problem, coverage.PHASE2
    )
    result = acceptance.model_report(tmp_path / "b64", 0)
    assert result["prior_functional_stage"]["status"] == "PHASE1_NAMED_BUDGET_PASSED_AND_FROZEN"
    assert result["synthetic_functional_exit_criterion"] == ("PASSED_THIS_RUN" if problem is None else "NOT_MET")
    assert result["real_text_validation_readiness"] == (
        "READY_FOR_SEPARATE_REAL_TEXT_VALIDATION" if problem is None else "BLOCKED_BY_PHASE2"
    )
    assert result["real_text_validation"] == "NOT_RUN"
    assert result["further_synthetic_matrix_expansion"] == "NOT_SCHEDULED"
    # A failed child cannot pass based solely on saved numerical samples.
    assert not acceptance.model_report(tmp_path / "b64", 1)["overall_pass"]


def test_phase2_shell_selection(tmp_path):
    fixtures.test_real_shell_selection_contains_only_phase_not_old_prefix(tmp_path, coverage.PHASE2)


def test_phase2_requires_accepted_archive_hash(tmp_path):
    invalid = tmp_path / "archive.tar.gz"
    invalid.write_bytes(b"wrong baseline")
    with pytest.raises(ValueError, match="hash mismatch"):
        coverage.audit_baseline(invalid, coverage.PHASE2)


@pytest.mark.parametrize("problem", ["missing_rank", "null_exitcode", "raw_hash", "owner"])
def test_incomplete_evidence_cannot_enable_real_text_readiness(tmp_path, problem):
    root = tmp_path / "b64"
    update = fixtures.functional_fixture(root, coverage.PHASE2)
    if problem == "null_exitcode":
        update(
            root / "worker-cleanup.json", workers=[{"rank": r, "raw_exitcode": None if r == 0 else 0} for r in range(8)]
        )
    elif problem == "owner":
        (root.parent / "b64.log").write_text("Scheduled candidates lack current proposal owners")
    else:
        path = root / (coverage.PHASE2_POINT_IDS[-1] + ".json")
        if problem == "raw_hash":
            path.write_text(path.read_text() + " ")
        else:
            raw = json.loads(path.read_text())
            raw["ranks"].pop()
            acceptance.write(path, raw)
            # Retained hash reflects these bytes; the missing rank itself must fail.
            retained = json.loads((root / "retained.json").read_text())
            retained[-1]["raw_sha256"] = fixtures.hashlib.sha256(path.read_bytes()).hexdigest()
            acceptance.write(root / "retained.json", retained)
    result = acceptance.model_report(root, 0)
    assert not result["overall_pass"] and result["real_text_validation_readiness"] == "BLOCKED_BY_PHASE2"
    assert result["prior_functional_stage"]["status"].endswith("PASSED_AND_FROZEN")
