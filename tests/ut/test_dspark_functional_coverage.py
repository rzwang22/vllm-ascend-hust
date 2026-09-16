# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU functional-entry regressions; synthetic samples are not NPU coverage."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_attention_validity import publish
from tests.ut.test_dspark_shutdown_policy import configured_model
from tests.ut.test_dspark_startup_cost_profile import MAPPING_SOURCE, snapshots
from tools.dspark import functional_coverage as coverage
from tools.dspark import run_large_batch as large
from tools.dspark import startup_cost_profile as profile
from tools.dspark import swa_acceptance as acceptance

ROOT = Path(__file__).parents[2]


def test_exact_bounded_subset_is_legal_not_the_remaining_full_matrix():
    data = coverage.plan(coverage.PHASE)
    _, matrix = profile.grid(64, list(coverage.CAPTURES), [128, 2048], 512)
    assert (len(matrix), sum(p["requests"] for p in matrix)) == (136, 2948)
    assert (data["point_count"], data["total_requests"], data["total_output_tokens"]) == (12, 328, 167936)
    assert data["max_runtime_seconds"] == large.TARGET_DIAGNOSTIC_RUNTIME_SECONDS == 3600
    assert coverage.select(matrix, coverage.PHASE) == data["points"]
    assert not {p["id"] for p in matrix[:10]} & set(coverage.POINT_IDS)
    assert len(data["remaining_matrix_point_ids"]) == 114
    assert {p["requests"] for p in data["points"]} == {8, 16, 32, 64}
    assert all(
        p in matrix and sum(x + 1 for x in p["lengths"]) == p["actual_tokens"] <= p["capacity"] for p in data["points"]
    )
    for n in (8, 16, 32, 64):
        assert {p["layout"] for p in data["points"] if p["requests"] == n} == {"balanced", "skewed"}
    assert len([p for p in data["points"] if p["prompt_tokens"] == 2048]) == 3
    assert data["performance_eligible"] is False


@pytest.mark.parametrize("problem", ["unknown", "missing", "illegal", "order"])
def test_selector_cannot_expand_or_silently_change_shape(problem):
    points = coverage.plan(coverage.PHASE)["points"]
    if problem == "missing":
        points.pop()
    elif problem == "illegal":
        points[0]["capacity"] += 1
    elif problem == "order":
        points.reverse()
    with pytest.raises(ValueError):
        coverage.select(points, "unbounded" if problem == "unknown" else coverage.PHASE)


def cli(tmp_path):
    return [
        "--plugin-sha",
        "a" * 40,
        "--manifest",
        "manifest",
        "--output-dir",
        str(tmp_path),
        "--stage",
        "profile",
        "--batches",
        "64",
        "--profile-experiment",
        "target-boundaries",
        "--profile-target-layer",
        "1",
        "--profile-target-attention",
        "--profile-worker-exit",
        "--profile-shutdown-policy",
        "dspark-profile-25s-v1",
        "--profile-coverage-phase",
        coverage.PHASE,
        "--capture-sizes",
        *map(str, coverage.CAPTURES),
    ]


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--batches", "128"],
        ["--profile-output-tokens", "256"],
        ["--profile-operator-capture"],
        ["--profile-exit-observation"],
        ["--capture-sizes", "6", "12", "384"],
        ["--profile-contexts", "128"],
    ],
)
def test_real_driver_cli_forwards_only_valid_bounded_phase(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(large, "run", lambda args: args)
    if extra:
        with pytest.raises(SystemExit):
            large.main(cli(tmp_path) + extra)
    else:
        args = large.main(cli(tmp_path))
        command = large.command(args, 64, tmp_path)
        assert command[command.index("--profile-coverage-phase") + 1] == coverage.PHASE
        assert "--profile-stop-after-point" not in command
        assert "--profile-exit-observation" not in command


def rows(point, rank_count=8):
    data = snapshots(point, ranks=rank_count)
    for rank in data:
        cost = rank["cost_profile"]
        cost["identity"].update(capture_sizes=list(coverage.CAPTURES), max_num_seqs=64)
        cost["observation"] = {
            "recording_error": None,
            "numeric": {
                "enabled": True,
                "nan_rounds": 0,
                "compact_host_transfers": 8,
                "compact_host_transfers_completed": 8,
            },
        }
        for event in cost["measurements"]:
            event.update(
                request_capacity=min(64, point["capacity"]),
                context=point["prompt_tokens"] + 20,
                scheduler_computed_upper_bounds=[point["prompt_tokens"] + 20] * point["requests"],
                effective_kv_before_query=[point["prompt_tokens"] + 16] * point["requests"],
                attention_seq_lens=[point["prompt_tokens"] + 16 + q for q in event["query_lengths"]],
            )
    return data


def stream(point):
    return {
        "error": None,
        "requests": [
            {"error": None, "output_token_ids": [1] * 512, "observed_prompt_token_ids": [90] * point["prompt_tokens"]}
            for _ in range(point["requests"])
        ],
    }


def functional_fixture(root, phase=coverage.PHASE):
    update = configured_model(root)
    manifest = coverage.plan(phase)
    update(root / "plan.json", points=manifest["points"], functional_coverage=manifest)
    update(root / "lifecycle.json", engine_initializations=1)
    retained = []
    for point in manifest["points"]:
        path = root / (point["id"] + ".json")
        update(
            path,
            point=point,
            ranks=rows(point),
            streaming=stream(point),
            request_identity_validation={"source": "CPU fixture"},
        )
        retained.append({"point": point, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    acceptance.write(root / "retained.json", retained)
    for rank in range(8):
        publish(root / "worker-first-failure", rank, point=manifest["points"][0]["id"])
    return update


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
def test_report_requires_real_full_concurrency_and_preserves_closed_baseline(tmp_path, problem, phase=coverage.PHASE):
    root = tmp_path / "b64"
    update = functional_fixture(root, phase)
    point = coverage.plan(phase)["points"][-1]
    path = root / (point["id"] + ".json")
    raw = json.loads(path.read_text())
    if problem == "partial_request":
        raw["streaming"]["requests"][0] = None
    elif problem == "short_input":
        raw["streaming"]["requests"][0]["observed_prompt_token_ids"] = [90] * 128
    elif problem == "short_output":
        raw["streaming"]["requests"][0]["output_token_ids"].pop()
    elif problem == "fewer_scheduled":
        other = {**point, "requests": 4, "lengths": [5] * 4, "actual_tokens": 24}
        raw["ranks"] = rows(other)
    elif problem == "stale_FULL":
        for event in raw["ranks"][0]["cost_profile"]["measurements"]:
            event["full_decode"] = False
    elif problem == "NaN":
        raw["ranks"][0]["cost_profile"]["observation"]["numeric"]["nan_rounds"] = 1
    elif problem == "cleanup":
        update(root / "cleanup.json", forced_cleanup=True)
    elif problem == "plan":
        update(root / "plan.json", points=[point])
    elif problem == "second_engine":
        update(root / "lifecycle.json", engine_initializations=2)
    acceptance.write(path, raw)
    retained = json.loads((root / "retained.json").read_text())
    retained[-1]["raw_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    acceptance.write(root / "retained.json", retained)
    result = acceptance.model_report(root, 0)
    assert result["overall_pass"] is (problem is None)
    assert result["prior_baseline"]["status"].endswith("PASSED_AND_CLOSED")
    assert "ten_points_generation_complete" not in result
    if problem is None:
        proof = result["points"][7 if phase == coverage.PHASE else 5]["functional_coverage"]
        assert proof["submitted_requests"] == 64
        assert all(r["scheduled_requests"] == [64] and r["count"] == 5 for r in proof["matched_FULL_samples_by_rank"])
        assert result["points"][-1]["functional_coverage"]["observed_prompt_lengths"] == [2048]
        assert result["performance_eligible"] is False
    elif problem == "cleanup":
        assert result["numerical_and_FULL_acceptance"] == "PASSED_THIS_RUN"


@pytest.mark.parametrize("problem", ["numeric", "prompt", "none"])
def test_real_collect_single_engine_stops_at_failed_point_and_keeps_prior_results(tmp_path, problem):
    points = coverage.plan(coverage.PHASE)["points"][:3]
    instances = []

    class Engine:
        def __init__(self):
            self.calls = 0
            self.closed = False
            self.current = None
            instances.append(self)

        def get_tokenizer(self):
            return NS(encode=lambda *a, **k: [90])

        def collective_rpc(self, method, kwargs=None):
            if method == "dspark_benchmark_profile_point":
                self.current = next(p for p in points if p["id"] == kwargs["point"])
                return []
            if not self.current:
                return []
            result = rows(self.current)
            for rank in result:
                for event in rank["cost_profile"]["measurements"]:
                    event["request_ids"] = [m["internal_id"] for m in self.last_batch["request_id_mapping"]["mappings"]]
                if problem == "numeric" and self.calls == 2:
                    rank["cost_profile"]["observation"]["numeric"]["nan_rounds"] = 1
            return result

        def generate(self, prompts, sampling, use_tqdm, *, profile_point):
            self.calls += 1
            self.last_batch = stream(self.current)
            self.last_batch["scheduler"] = {}
            ids = [f"batch{self.calls}-{i}" for i in range(len(prompts))]
            for i, request in enumerate(self.last_batch["requests"]):
                request.update(request_id=ids[i], request_index=i)
            if problem == "prompt" and self.calls == 2:
                self.last_batch["requests"][0]["observed_prompt_token_ids"] = []
            self.last_batch["request_id_mapping"] = {
                "source": MAPPING_SOURCE,
                "point": profile_point,
                "hook_restored": True,
                "errors": [],
                "expected_external_ids": ids,
                "mappings": [
                    {"point": profile_point, "request_index": i, "external_id": key, "internal_id": key + "-abcdef01"}
                    for i, key in enumerate(ids)
                ],
            }
            return [None] * len(prompts)

        def shutdown(self):
            self.closed = True

    def run():
        return profile.collect(Engine, points, None, tmp_path, warmup=2, samples=5, require_numerical=True)

    if problem == "none":
        assert len(run()[0]) == 3
    else:
        with pytest.raises(ValueError, match="Functional point"):
            run()
        assert len(json.loads((tmp_path / "retained.json").read_text())) == 1
        assert json.loads((tmp_path / "profile-failure.json").read_text())["point"] == points[1]
        assert not (tmp_path / (points[2]["id"] + ".json")).exists()
    assert len(instances) == 1 and instances[0].closed


def test_real_shell_selection_contains_only_phase_not_old_prefix(tmp_path, phase=coverage.PHASE):
    shim = tmp_path / "bash"
    shim.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_OUT"\n')
    shim.chmod(0o755)
    out = tmp_path / "args"
    subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "tools/dspark/run_dspark_swa_acceptance.sh"),
            "a" * 40,
            "manifest",
            "rzwang",
            "--coverage=" + phase,
        ],
        check=True,
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "ARGS_OUT": str(out)},
    )
    args = out.read_text().splitlines()
    assert args[args.index("--profile-coverage-phase") + 1] == phase
    assert "--profile-stop-after-point" not in args
    assert args[args.index("--profile-shutdown-policy") + 1] == "dspark-profile-25s-v1"
    assert "--profile-operator-capture" not in args and "--profile-exit-observation" not in args
