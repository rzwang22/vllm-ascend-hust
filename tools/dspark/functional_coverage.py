# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded functional subsets of the existing B64 grid, never a cost table."""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path

PHASE = "b64-functional-1"
PHASE2 = "b64-functional-2"
PHASES = (PHASE, PHASE2)
PHASE1_PLUGIN = "c271239bc3bfc104309b61740476df6f2d07c9bc"
PHASE1_ARCHIVE_SHA = "ed690d6971ec777af5a43359ae79f9a24c9f082fb7105485d256022c32cb1fed"
PHASE2_POINT_IDS = (
    "ctx2048-n16-t48-balanced",
    "ctx2048-n16-t48-skewed",
    "ctx2048-n32-t96-balanced",
    "ctx2048-n32-t96-skewed",
    "ctx2048-n64-t192-balanced",
    "ctx2048-n64-t192-skewed",
    "ctx2048-n64-t384-balanced",
)
CAPTURES = (6, 12, 24, 48, 96, 192, 384)
CONTEXTS = (128, 2048)
RUNTIME_SECONDS = 3600
BASELINE_PLUGIN = "b2810899840141f1a71324d90e263008eecc4f06"
BASELINE_ARCHIVE_SHA = "7b91b1b039cd2de7cb020d802862583f0c34b4decdecede0e29477bb3e1c2f8e"
POINT_IDS = (
    "ctx128-n8-t24-balanced",
    "ctx128-n8-t24-skewed",
    "ctx128-n16-t48-balanced",
    "ctx128-n16-t48-skewed",
    "ctx128-n32-t96-balanced",
    "ctx128-n32-t96-skewed",
    "ctx128-n64-t192-balanced",
    "ctx128-n64-t192-skewed",
    "ctx128-n64-t384-balanced",
    "ctx2048-n8-t12-balanced",
    "ctx2048-n8-t24-skewed",
    "ctx2048-n8-t48-balanced",
)


def plan(phase):
    # startup_cost_profile consumes this selector; avoid a module import cycle.
    from tools.dspark.startup_cost_profile import grid

    if phase not in PHASES:
        raise ValueError("Unknown functional coverage phase")
    _, matrix = grid(64, list(CAPTURES), list(CONTEXTS), 512)
    ids = POINT_IDS if phase == PHASE else PHASE2_POINT_IDS
    selected = [p for p in matrix if p["id"] in ids]
    if tuple(p["id"] for p in selected) != ids:
        raise ValueError("Functional phase no longer matches the existing legal matrix")
    result = {
        "phase": phase,
        "performance_eligible": False,
        "engine_max_num_seqs": 64,
        "model_initializations": 1,
        "output_tokens_per_request": 512,
        "max_runtime_seconds": RUNTIME_SECONDS,
        "point_count": len(selected),
        "total_requests": sum(p["requests"] for p in selected),
        "total_output_tokens": sum(p["requests"] * 512 for p in selected),
        "matrix_point_count": len(matrix),
        "matrix_total_requests": sum(p["requests"] for p in matrix),
        "points": selected,
        "remaining_matrix_point_ids": [p["id"] for p in matrix[10:] if p["id"] not in POINT_IDS],
        "scope": (
            "Specified verification lengths, not confidence-policy quality/performance; "
            "actual concurrency requires FULL samples"
        ),
        "prior_baseline": {
            "plugin": BASELINE_PLUGIN,
            "archive_sha256": BASELINE_ARCHIVE_SHA,
            "status": "ORIGINAL_TEN_POINT_NAMED_BUDGET_PASSED_AND_CLOSED",
        },
    }

    if phase == PHASE2:
        result.update(
            prior_functional_stage={
                "phase": PHASE,
                "plugin": PHASE1_PLUGIN,
                "archive_sha256": PHASE1_ARCHIVE_SHA,
                "status": "PHASE1_NAMED_BUDGET_PASSED_AND_FROZEN",
            },
            remaining_matrix_point_ids=[p["id"] for p in matrix[10:] if p["id"] not in POINT_IDS + PHASE2_POINT_IDS],
            remaining_matrix_is_not_a_required_checklist=True,
            next_stage="Separate real-text validation after phase2 passes; no automatic synthetic expansion",
        )
    return result


def select(points, phase):
    expected = plan(phase)["points"]
    actual = [p for p in points if p["id"] in {p["id"] for p in expected}]
    if actual != expected:
        raise ValueError("Coverage requires unchanged B64 captures, contexts, lengths and 512-token output")
    return actual


def validate_args(args):
    if not getattr(args, "profile_coverage_phase", None):
        return
    if (
        args.profile_coverage_phase not in PHASES
        or args.stage != "profile"
        or getattr(args, "batch", 64) != 64
        or getattr(args, "batches", [64]) != [64]
        or args.profile_experiment != "target-boundaries"
        or args.profile_target_layer != 1
        or not args.profile_target_attention
        or not args.profile_worker_exit
        or args.profile_shutdown_policy != "dspark-profile-25s-v1"
        or args.profile_output_tokens != 512
        or args.profile_warmup != 2
        or args.profile_samples != 5
        or args.max_model_len != 8192
        or args.max_num_batched_tokens != 8192
        or args.gpu_memory_utilization != 0.9
        or args.profile_contexts != list(CONTEXTS)
        or (getattr(args, "capture_sizes", None) or getattr(args, "capture", None)) != list(CAPTURES)
        or any(
            getattr(args, k, False)
            for k in (
                "profile_exit_observation",
                "profile_operator_capture",
                "profile_write_timeline",
                "profile_nan_diagnostic",
            )
        )
    ):
        raise ValueError(
            "Functional coverage requires frozen B64 config, numeric/FULL/owner gates "
            "and named shutdown; no heavy diagnostics"
        )


def requests_complete(point, stream):
    requests = stream.get("requests") or []
    return (
        len(requests) == point["requests"]
        and not stream.get("error")
        and all(
            isinstance(r, dict)
            and not r.get("error")
            and len(r.get("output_token_ids", [])) == 512
            and len(r.get("observed_prompt_token_ids", [])) == point["prompt_tokens"]
            for r in requests
        )
    )


def observed_layout(point, snapshots, retained, stream=None):
    """Use validated saved events, not submitted request count, as coverage proof."""
    targets = [e for s in snapshots for e in s["cost_profile"]["measurements"] if e["kind"] == "target"]
    full = [r for r in retained if r["kind"] == "target"]
    return {
        "observed_prompt_lengths": sorted(
            {
                len(r.get("observed_prompt_token_ids", []))
                for r in ((stream or {}).get("requests") or [])
                if isinstance(r, dict)
            }
        ),
        "submitted_requests": point["requests"],
        "input_tokens_per_request": point["prompt_tokens"],
        "specified_draft_verification_lengths": point["lengths"],
        "planned_target_query_tokens": point["actual_tokens"],
        "planned_graph_capacity": point["capacity"],
        "observed_target_scheduled_requests": sorted({e["requests"] for e in targets}),
        "observed_target_query_tokens": sorted({e["actual_tokens"] for e in targets}),
        "observed_graph_capacities": sorted({e["capacity"] for e in targets if e["full_decode"]}),
        "matched_FULL_samples_by_rank": [
            {
                "rank": r["rank"],
                "count": len(r["samples"]),
                "scheduled_requests": sorted({e["requests"] for e in r["samples"]}),
            }
            for r in full
        ],
        "coverage_limit": "Saved target-event history and validated FULL samples; not every scheduler iteration",
    }


def audit_baseline(path, phase=PHASE):
    # Read scripts only as archive bytes; never execute or follow archived links.
    from tools.dspark.swa_acceptance import model_report

    plan(phase)  # Reject unknown stages before reading an archive.
    expected_sha = BASELINE_ARCHIVE_SHA if phase == PHASE else PHASE1_ARCHIVE_SHA
    expected_plugin = BASELINE_PLUGIN if phase == PHASE else PHASE1_PLUGIN
    directory = "dspark-large-batch.Ck6iA7rN" if phase == PHASE else "dspark-large-batch.XM02ngWZ"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_sha:
        raise ValueError("Accepted model baseline archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="dspark-accepted-baseline-") as temporary:
        root = Path(temporary).resolve()
        with tarfile.open(path) as archive:
            for member in archive:
                target = root / member.name
                if not target.resolve().is_relative_to(root):
                    raise ValueError("Unsafe baseline archive member")
                if member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        model = root / directory
        result = model_report(model / "runs/b64", 0)
        saved = json.loads((model / "model-acceptance.json").read_text())
        source = json.loads((model / "core-source.json").read_text())
        original_plan = json.loads((model / "runs/b64/plan.json").read_text())
        if result != saved or result.get("overall_pass") is not True or original_plan["plugin_sha"] != expected_plugin:
            raise ValueError("Accepted baseline failed independent report reconstruction")
        return {
            "archive_sha256": digest,
            "plugin": expected_plugin,
            "status": (
                "ORIGINAL_TEN_POINT_NAMED_BUDGET_PASSED_AND_CLOSED"
                if phase == PHASE
                else "PHASE1_NAMED_BUDGET_PASSED_AND_FROZEN"
            ),
            "core_source": source,
            "source_scope": "Recorded URL and exact HEAD; selected_remote=rzwang does not imply a GitHub fetch",
            "original_budget_acceptance": result["original_budget_acceptance"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline-archive", type=Path)
    args = parser.parse_args()
    data = plan(args.phase)
    receipt = (
        {"plan": data, "baseline_audit": audit_baseline(args.baseline_archive, args.phase)}
        if args.baseline_archive
        else data
    )
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "remaining_matrix_point_ids"}, indent=2))


if __name__ == "__main__":
    main()
