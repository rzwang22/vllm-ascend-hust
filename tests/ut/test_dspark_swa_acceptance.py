# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host entry tests: actual Git remotes, original plan, and independent result gates."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.ut.test_dspark_attention_validity import publish
from tests.ut.test_dspark_startup_cost_profile import snapshots
from tools.dspark import startup_cost_profile as profile
from tools.dspark import swa_acceptance as acceptance

ROOT = Path(__file__).resolve().parents[2]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.mark.parametrize("remote", ["rzwang", "origin"])
def test_real_remote_selection_and_exact_head_without_origin_rewrite(tmp_path, monkeypatch, remote):
    origin, fork, checkout = [tmp_path / n for n in ("upstream", "fork", "checkout")]
    for repo in (origin, fork, checkout):
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        git(repo, "config", "commit.gpgsign", "false")
        git(repo, "config", "user.name", "Test")
        git(repo, "config", "user.email", "test@example.invalid")
    git(fork, "checkout", "-b", "feat/dspark")
    (fork / "code").write_text("old")
    git(fork, "add", ".")
    git(fork, "commit", "-qm", "base")
    git(checkout, "remote", "add", "origin", str(origin))
    git(checkout, "remote", "add", "rzwang", str(fork))
    git(checkout, "fetch", "rzwang", "feat/dspark")
    git(checkout, "checkout", "-b", "feat/dspark", "FETCH_HEAD")
    (fork / "code").write_text("fixed")
    git(fork, "commit", "-qam", "fix")
    sha = git(fork, "rev-parse", "HEAD")
    monkeypatch.setattr(acceptance.suite, "CORE_SHA", sha)
    output = tmp_path / "receipt.json"
    if remote == "origin":
        with pytest.raises(subprocess.CalledProcessError):
            acceptance.core_source(checkout, remote, output)
        assert git(checkout, "rev-parse", "HEAD") != sha
        assert json.loads(output.read_text())["status"] == "failed"
    else:
        acceptance.core_source(checkout, remote, output)
        data = json.loads(output.read_text())
        assert data["actual_head"] == data["expected_head"] == sha
        assert data["remote_url"] == str(fork)
        assert data["status"] == "verified"
    assert git(checkout, "remote", "get-url", "origin") == str(origin)


def model_fixture(root):
    root.mkdir()
    _, points = profile.grid(64, [6, 12, 24, 48, 96, 192, 384], [128, 2048], 512)
    points = profile.diagnostic_points(points, "ctx128-n4-t12-skewed")
    acceptance.write(
        root / "plan.json", {"points": points, "core_sha": acceptance.suite.CORE_SHA, "performance_eligible": False}
    )
    retained = []
    for point in points:
        ranks = snapshots(point, ranks=8)
        for rank in ranks:
            rank["cost_profile"]["observation"] = {
                "numeric": {
                    "enabled": True,
                    "nan_rounds": 0,
                    "compact_host_transfers": 10,
                    "compact_host_transfers_completed": 10,
                }
            }
        raw = {
            "point": point,
            "ranks": ranks,
            "request_identity_validation": {"source": "test"},
            "streaming": {
                "error": None,
                "requests": [{"error": None, "output_token_ids": [1] * 512} for _ in range(point["requests"])],
            },
        }
        path = root / (point["id"] + ".json")
        acceptance.write(path, raw)
        retained.append({"point": point, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    acceptance.write(root / "retained.json", retained)
    acceptance.write(root / "point-completion.json", {"status": "completed"})
    acceptance.write(root / "cleanup.json", {"success": True, "timed_out": False, "forced_cleanup": False})
    acceptance.write(
        root / "worker-cleanup.json",
        {"forced_cleanup": False, "workers": [{"rank": r, "raw_exitcode": 0} for r in range(8)]},
    )
    (root / "worker-first-failure").mkdir()
    for rank in range(8):
        publish(root / "worker-first-failure", rank, point=points[0]["id"])
    (root.parent / "b64.log").write_text("model completed\n")


@pytest.mark.parametrize(
    "problem", [None, "forced", "timeout", "missing_exit", "nan_owner", "stale", "hash", "short", "old_core"]
)
def test_generation_numerics_graph_and_cleanup_are_independent(tmp_path, problem):
    root = tmp_path / "b64"
    model_fixture(root)
    if problem in ("forced", "timeout"):
        acceptance.write(
            root / "cleanup.json",
            {"success": False, "timed_out": problem == "timeout", "forced_cleanup": problem == "forced"},
        )
    elif problem == "missing_exit":
        p = root / "worker-cleanup.json"
        data = json.loads(p.read_text())
        data["workers"][0]["raw_exitcode"] = None
        acceptance.write(p, data)
    elif problem == "nan_owner":
        (tmp_path / "b64.log").write_text(
            "Ascend DSpark Markov base logits contain NaN\nScheduled candidates lack current proposal owners"
        )
        acceptance.write(root / "profile-failure.json", {"error": "first NaN"})
    elif problem == "stale":
        p = root / "worker-first-failure/rank-0-attention-validity.json"
        data = json.loads(p.read_text())
        data["rounds"][0]["target_receipts"][0] = -1
        acceptance.write(p, data)
    elif problem in ("hash", "short"):
        p = root / "ctx128-n1-t6-balanced.json"
        data = json.loads(p.read_text())
        if problem == "short":
            data["streaming"]["requests"][0]["output_token_ids"].pop()
        else:
            data["extra"] = True
        acceptance.write(p, data)
    elif problem == "old_core":
        p = root / "plan.json"
        data = json.loads(p.read_text())
        data["core_sha"] = "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
        acceptance.write(p, data)
    result = acceptance.model_report(root, int(problem is not None))
    assert result["overall_pass"] is (problem is None)
    if problem in ("forced", "timeout", "missing_exit"):
        assert result["ten_points_generation_complete"]
        assert result["numerical_and_FULL_acceptance"] == "PASSED_THIS_RUN"
        assert not result["worker_natural_exit"]
    if problem == "missing_exit":
        assert result["workers"][0]["raw_exitcode"] is None
    if problem == "nan_owner":
        assert result["markov_NaN_in_log"] and result["owner_error_in_log"]
        assert result["primary_failure"]["error"] == "first NaN"


def test_actual_shell_wrapper_keeps_original_plan_and_remote(tmp_path):
    shim = tmp_path / "bash"
    shim.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_OUT"\n')
    shim.chmod(0o755)
    output = tmp_path / "arguments"
    subprocess.run(
        ["/bin/bash", str(ROOT / "tools/dspark/run_dspark_swa_acceptance.sh"), "a" * 40, "manifest", "rzwang"],
        check=True,
        env={**os.environ, "PATH": str(tmp_path) + ":" + os.environ["PATH"], "ARGS_OUT": str(output)},
    )
    args = output.read_text().splitlines()
    assert args[args.index("--core-remote") + 1] == "rzwang"
    assert args[args.index("--batches") + 1] == "64"
    assert args[args.index("--profile-output-tokens") + 1] == "512"
    assert args[args.index("--profile-stop-after-point") + 1] == "ctx128-n4-t12-skewed"
    assert args[args.index("--capture-sizes") + 1 : args.index("--profile-contexts")] == [
        "6",
        "12",
        "24",
        "48",
        "96",
        "192",
        "384",
    ]
    assert "--profile-target-attention" in args and "--profile-worker-exit" in args
    assert not {"--profile-operator-capture", "--profile-write-timeline"} & set(args)


def test_wrong_archive_hash_fails_before_loading(tmp_path):
    p = tmp_path / "invalid.tar.gz"
    p.write_bytes(b"not the verified local validation")
    with pytest.raises(AssertionError):
        acceptance.audit_local(p)


@pytest.mark.parametrize(
    "generation_rc,report_rc,observation_rc",
    [(0, 0, None), (0, 1, None), (7, 1, None), (7, 1, 9), (0, 1, 9), (0, 0, 9), (0, 0, 0)],
)
def test_actual_large_shell_filters_remote_and_preserves_generation_failure(
    tmp_path, generation_rc, report_rc, observation_rc
):
    workspace = tmp_path / "workspace"
    plugin = workspace / "vllm-ascend-hust"
    plugin.mkdir(parents=True)
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    (binary / "git").write_text(
        '#!/bin/sh\ncase "$*" in\n'
        '*"rev-parse HEAD"*) case "$*" in *vllm-ascend-hust*) echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;; '
        f"*) echo {acceptance.suite.CORE_SHA};; esac;;\n"
        '*"branch --show-current"*) echo feat/dspark;; esac\n'
    )
    (binary / "python").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALLS"\ncase "$*" in\n'
        f'*"run_large_batch.py"*) exit {generation_rc};;\n'
        f'*"swa_acceptance report"*) exit {report_rc};;\n'
        f'*"exit_observation_report"*) exit {observation_rc or 0};;\nesac\n'
    )
    (binary / "timeout").write_text('#!/bin/sh\nwhile test "$1" != python; do shift; done\nexec "$@"\n')
    (binary / "sha256sum").write_text('#!/bin/sh\nshasum -a 256 "$@"\n')
    for path in binary.iterdir():
        path.chmod(0o755)
    source = (ROOT / "tools/dspark/run_dspark_large_batch.sh").read_text().replace("/workspace", str(workspace))
    script = tmp_path / "run.sh"
    script.write_text(source)
    env = {
        **os.environ,
        "PATH": str(binary) + ":" + os.environ["PATH"],
        "CALLS": str(calls),
        "ASCEND_CUSTOM_OPP_PATH": "test",
    }
    run = subprocess.run(
        [
            "/bin/bash",
            str(script),
            "a" * 40,
            "manifest",
            "--core-remote",
            "rzwang",
            "--swa-acceptance-archive",
            "archive",
            "--profile-experiment",
            "target-boundaries",
            "--stage",
            "profile",
            "--batches",
            "64",
            *(["--profile-exit-observation"] if observation_rc is not None else []),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert run.returncode == (generation_rc or report_rc or observation_rc or 0), run.stdout + run.stderr
    commands = calls.read_text().splitlines()
    assert any("core-source" in c and "rzwang" in c for c in commands)
    generation = next(c for c in commands if "run_large_batch.py" in c)
    assert "--core-remote" not in generation and "--swa-acceptance-archive" not in generation
    assert not any("test_dspark_operator_capture.py" in c or "test_dspark_swa_lifecycle.py" in c for c in commands)
    assert any(f"b64 {generation_rc}" in c for c in commands if "swa_acceptance report" in c)
    if observation_rc is not None:
        assert any("exit_native_preflight" in c for c in commands)
        assert any("exit_observation_report" in c for c in commands)
    result = next((workspace / "dspark-results").glob("dspark-large-batch.*-evidence.tar.gz"))
    assert result.is_file()
