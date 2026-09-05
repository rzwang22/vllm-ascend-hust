# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ut.test_p08_r8_runner import _PYTHON_SHIM, PLUGIN_SHA, _result, _write_executable
from tools.dspark import p08_r9_checks as checks

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "case", ["bracket", "unreproduced", "missing", "outside_window", "capture_only", "other_failure", "result_gate"]
)
def test_statuses_do_not_equate_files_capture_or_completion_to_root_cause(tmp_path, case):
    directory = tmp_path / "rank-diagnostics"
    directory.mkdir()
    for rank in range(8 if case != "missing" else 7):
        report = {
            "rank": rank,
            "performance_eligible": False,
            "replay_configuration": {"completed_detailed_replays": 0 if case == "capture_only" else 3},
            "previous_executions": [
                {
                    "target_execution_epoch": epoch,
                    "phase": "warmup",
                    "replay_detail": {"status": "ACTUAL_FULL_REPLAY_SNAPSHOTS"},
                }
                for epoch in (68, 69)
            ],
            "current": {
                "target_execution_epoch": 70,
                "phase": "warmup",
                "stage": "target_outputs",
                "replay_detail": {
                    "status": "unavailable" if case == "outside_window" else "ACTUAL_FULL_REPLAY_SNAPSHOTS",
                    "localization": {
                        "status": "OBSERVED_BOUNDARY_BRACKET"
                        if case == "bracket"
                        else "NO_NONFINITE_IN_OBSERVED_BOUNDARIES"
                    },
                },
            },
        }
        (directory / f"rank-{rank}-latest.json").write_text(json.dumps(report))
        if case in ("bracket", "outside_window", "other_failure"):
            (directory / f"rank-{rank}-first-failure.json").write_text(json.dumps(report))
    status = checks.diagnostics(
        tmp_path, 1 if case in ("bracket", "outside_window", "other_failure") else 0, 1 if case == "result_gate" else 0
    )
    assert status["root_cause_status"] == "ROOT_CAUSE_NOT_YET_PROVEN"
    assert status["performance_eligible"] is False
    expected = {"bracket": "OBSERVED_BOUNDARY_BRACKET", "unreproduced": "NOT_REPRODUCED_OBSERVER_MAY_PERTURB"}.get(
        case, "UNAVAILABLE"
    )
    assert status["localization_status"] == expected
    assert (tmp_path / "diagnostic-index.json").is_file()
    if case == "result_gate":
        assert status["generation_status"] == "RETURNED_BUT_RESULT_GATE_FAILED"


@pytest.mark.parametrize("fault", [None, "source", "dataset", "focused", "graph", "replay", "tee"])
def test_one_graph_shell_run_keeps_status_pipelines_and_archive(tmp_path, fault):
    # Execute the real shell runner. Only git/NPU/Python engine processes are
    # shims; dataset hashes, gates, pipeline statuses and tar run normally.
    workspace = tmp_path / "workspace"
    plugin, core = workspace / "vllm-ascend-hust", workspace / "vllm-hust"
    tools = plugin / "tools/dspark"
    tools.mkdir(parents=True)
    core.mkdir()
    for name in ("p08_r8_checks.py", "p08_r9_checks.py"):
        shutil.copyfile(ROOT / "tools/dspark" / name, tools / name)
    data = b'{"prompt_token_ids": [1, 2, 3]}\n' * 64
    dataset = workspace / "input-dataset.jsonl"
    dataset.write_bytes(data)
    cann = tmp_path / "cann.sh"
    cann.write_text("export P08_CANN_RESTORED=1\nexport ASCEND_CUSTOM_OPP_PATH=overwritten-by-cann\n")
    script = (ROOT / "tools/dspark/run_p08_r9.sh").read_text().replace("/workspace", str(workspace))
    script = script.replace("/usr/local/Ascend/ascend-toolkit/set_env.sh", str(cann))
    if fault != "dataset":
        script = script.replace(
            "6a2f629a5b5c9bbd9a3058b7a450fc18b2332f4699047f164cdde6a33b58d053", hashlib.sha256(data).hexdigest()
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        script = script.replace("P08_PORT=29888", f"P08_PORT={sock.getsockname()[1]}")
    runner = tmp_path / "run.sh"
    runner.write_text(script)
    shim = tmp_path / "bin"
    shim.mkdir()
    python_shim = _PYTHON_SHIM.replace(
        "'current': {'stage'", "'replay_configuration': {'completed_detailed_replays': 1}, 'current': {'stage'"
    )
    _write_executable(shim / "python", f"#!{sys.executable}\n" + python_shim)
    _write_executable(shim / "npu-smi", '#!/bin/sh\nprintf "CPU shell fixture, no NPU\\n"\n')
    _write_executable(
        shim / "git",
        f"""#!/bin/bash
if [[ "$*" == *"rev-parse HEAD"* ]]; then
    if [[ "$*" == *"/vllm-hust "* ]]; then
        printf '%s\\n' '{checks.common.CORE_SHA}'
    else
        printf '%s\\n' '{PLUGIN_SHA}'
    fi
elif [[ "$*" == *"branch --show-current"* ]]; then printf 'feat/dspark\\n'; fi
""",
    )
    if fault == "tee":
        _write_executable(
            shim / "tee", '#!/bin/bash\n/usr/bin/tee "$@"\nif [[ "$*" == *"focused.log"* ]]; then exit 9; fi\n'
        )
    result = tmp_path / "result.json"
    result.write_text(json.dumps(_result()))
    env = {
        **os.environ,
        "PATH": f"{shim}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "RESULT_FIXTURE": str(result),
        "FOCUSED_RC": "13" if fault == "focused" else "0",
        "GRAPH_RC": "7" if fault == "graph" else "0",
        "ASCEND_CUSTOM_OPP_PATH": "test-custom-opp",
    }
    if fault == "replay":
        env["BAD_REPLAY"] = "1"
    run = subprocess.run(
        ["bash", str(runner), "b" * 40 if fault == "source" else PLUGIN_SHA, str(dataset), "58", "82"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    out = next(path for path in (workspace / "dspark-results").glob("m2_5a-p08-r9-diagnostic.*") if path.is_dir())
    assert Path(str(out) + "-evidence.tar.gz").is_file(), run.stdout + run.stderr
    assert Path(str(out) + "-evidence.sha256").is_file()
    assert (out / "diagnostic-index.json").is_file()
    assert "PERFORMANCE_ELIGIBLE=false" in (out / "gate.txt").read_text()
    assert "DSPARK_PR_STYLE_BENCHMARK_PASS" not in run.stdout
    assert (run.returncode == 0) == (fault is None), run.stdout + run.stderr
    if fault in ("source", "dataset", "focused", "tee"):
        assert not (out / "actual-argv.json").exists()
    else:
        argv = json.loads((out / "actual-argv.json").read_text())
        index = argv.index("--dspark-nan-replay-window")
        assert argv[index + 1 : index + 3] == ["58", "82"]
        index = argv.index("--cudagraph-capture-sizes")
        assert argv[index + 1 : index + 5] == ["6", "12", "18", "24"]
        assert (out / "graph-b4-pipestatus.txt").read_text().strip() == ("7 0" if fault == "graph" else "0 0")
        assert (out / "diagnostic-index.json").is_file()
    if fault == "tee":
        assert (out / "focused-pipestatus.txt").read_text().strip() == "0 9"
