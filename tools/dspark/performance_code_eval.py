# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fixed Python-function judge. Generated code runs only in a restricted Docker container."""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from collections import Counter
from pathlib import Path

import regex as re

EXTRACTOR_VERSION = "single_python_fence_v1"
JUDGE_VERSION = "python_function_v1"
CONTAINER_RUNNER = """import json, pathlib, subprocess, sys
job = json.loads(pathlib.Path('/job/job.json').read_text())
code = pathlib.Path('/job/solution.py').read_text()
try:
    compile(code, 'solution.py', 'exec')
except (SyntaxError, ValueError, OverflowError) as error:
    print(json.dumps({'status': 'syntax_failure', 'syntax_pass': False, 'error': str(error)}))
    sys.exit(0)
if job['tests'] is None:
    print(json.dumps({'status': 'tests_unavailable', 'syntax_pass': True}))
    sys.exit(0)
tests = job['tests']
# The solution/test files and parent judge are read-only. Child output is bounded
# by RLIMIT_FSIZE; the parent's single JSON response is not candidate stdout.
script = ("import runpy\\nns = runpy.run_path('/job/solution.py')\\n" +
          tests['test'] + "\\ncheck(ns[" + repr(tests['entry_point']) + "])\\n" +
          "open('/tmp/completed', 'w').write('completed')\\n")
pathlib.Path('/tmp/test.py').write_text(script)
try:
    with open('/tmp/child.log', 'wb') as output:
        run = subprocess.run([sys.executable, '-I', '/tmp/test.py'], stdout=output,
                             stderr=subprocess.STDOUT, timeout=job['test_timeout'])
    passed = run.returncode == 0 and pathlib.Path('/tmp/completed').is_file()
    print(json.dumps({'status': 'passed' if passed else 'test_failure',
                      'syntax_pass': True, 'returncode': run.returncode,
                      'test_log': pathlib.Path('/tmp/child.log').read_text(errors='replace')[-8192:]}))
except subprocess.TimeoutExpired:
    print(json.dumps({'status': 'timeout', 'syntax_pass': True}))
"""


def extract_code(text):
    blocks = re.findall(r"```([^\n]*)\n(.*?)```", text, re.S)
    if len(blocks) != 1 or blocks[0][0].strip().lower() not in ("python", "python3", ""):
        raise ValueError("Require exactly one Python code fence")
    if not blocks[0][1].strip():
        raise ValueError("Empty code block")
    return blocks[0][1].strip() + "\n"


def sandbox_command(image, job_dir, container_name):
    if not re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Sandbox image must be pinned by sha256 digest")
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1",
        "--ulimit",
        "cpu=15:15",
        "--ulimit",
        "fsize=1048576:1048576",
        "--ulimit",
        "nofile=64:64",
        "--user=65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=32m,mode=1777",
        "--mount",
        f"type=bind,src={job_dir.resolve()},dst=/job,readonly",
        "--workdir=/tmp",
        "--entrypoint=python",
        image,
        "-I",
        "/job/judge.py",
    ]


def judge(code, tests, image, *, run=subprocess.run):
    if not image:
        return {"status": "infrastructure_unavailable", "syntax_pass": None, "reason": "No pinned sandbox image"}
    name = f"dspark-eval-{uuid.uuid4().hex}"
    try:
        with tempfile.TemporaryDirectory(prefix="dspark-eval-") as temporary:
            root = Path(temporary)
            root.chmod(0o755)
            (root / "solution.py").write_text(code)
            (root / "judge.py").write_text(CONTAINER_RUNNER)
            (root / "job.json").write_text(json.dumps({"tests": tests, "test_timeout": 10}))
            command = sandbox_command(image, root, name)
            try:
                completed = run(command, text=True, capture_output=True, timeout=30)
            except subprocess.TimeoutExpired:
                # Remove only this judge's UUID container, never an inference task.
                cleanup = run(["docker", "rm", "-f", name], text=True, capture_output=True, timeout=15)
                return {
                    "status": "infrastructure_failure",
                    "syntax_pass": None,
                    "reason": "container exceeded outer deadline",
                    "cleanup_returncode": cleanup.returncode,
                }
            if completed.returncode != 0:
                return {
                    "status": "infrastructure_failure",
                    "syntax_pass": None,
                    "returncode": completed.returncode,
                    "error": completed.stderr[-8192:],
                }
            result = json.loads(completed.stdout)
            if result.get("status") not in ("passed", "test_failure", "syntax_failure", "timeout", "tests_unavailable"):
                raise ValueError("Invalid judge receipt")
            return result
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"status": "infrastructure_unavailable", "syntax_pass": None, "reason": str(error)}


def quality_summary(rows, expected):
    observed = Counter(row["status"] for row in rows)
    counts = {
        name: observed[name]
        for name in (
            "passed",
            "test_failure",
            "syntax_failure",
            "timeout",
            "extraction_failure",
            "generation_failure",
            "tests_unavailable",
            "infrastructure_unavailable",
            "infrastructure_failure",
            "not_applicable",
        )
    }
    syntax_denominator = sum(row.get("syntax_pass") is not None for row in rows)
    syntax_passed = sum(row.get("syntax_pass") is True for row in rows)
    tested = sum(
        row.get("has_tests")
        and row["status"] not in ("infrastructure_unavailable", "infrastructure_failure", "generation_failure")
        for row in rows
    )
    passed = counts.get("passed", 0)
    complete = len(rows) == expected and tested == expected
    return {
        "status": "available" if complete else "unavailable",
        "task_count": expected,
        "generated_tasks": sum(row["status"] != "generation_failure" for row in rows),
        "counts": counts,
        "syntax_passed": syntax_passed,
        "syntax_denominator": syntax_denominator,
        "syntax_pass_rate": syntax_passed / syntax_denominator if syntax_denominator else None,
        "unit_test_tasks_passed": passed,
        "unit_test_task_denominator": tested,
        "unit_test_pass_rate": passed / tested if tested else None,
        "all_task_pass_fraction": passed / expected if complete and expected else None,
        "extractor_version": EXTRACTOR_VERSION,
        "judge_version": JUDGE_VERSION,
    }


def evaluate(records, stream_requests, output_dir, image, *, reference=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, task in enumerate(records):
        request = stream_requests[i] if i < len(stream_requests) else None
        tests = task["tests"]
        row = {"case_id": task["case_id"], "has_tests": tests is not None}
        if task.get("sample_kind") == "general":
            if request is not None:
                (output_dir / f"{i:06d}-generation.txt").write_text(request["text"])
            rows.append({**row, "status": "not_applicable", "syntax_pass": None})
            continue
        if reference and tests is None:
            rows.append({**row, "status": "tests_unavailable", "syntax_pass": None})
            continue
        if reference and tests is not None:
            code = tests["reference"]
        elif request is None or request.get("error") or request.get("completed_monotonic") is None:
            rows.append({**row, "status": "generation_failure", "syntax_pass": None})
            continue
        else:
            (output_dir / f"{i:06d}-generation.txt").write_text(request["text"])
            try:
                code = extract_code(request["text"])
            except ValueError as error:
                rows.append({**row, "status": "extraction_failure", "syntax_pass": None, "error": str(error)})
                continue
        (output_dir / f"{i:06d}-solution.py").write_text(code)
        rows.append({**row, **judge(code, tests, image)})
    result = {
        "summary": quality_summary(rows, len(records)),
        "tasks": rows,
        "sandbox_image": image,
        "reference_evaluation": reference,
        "inference_timing_included": False,
    }
    if not image:
        result["summary"]["status"] = "unavailable"
        result["summary"]["all_task_pass_fraction"] = None
    (output_dir / "quality.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
