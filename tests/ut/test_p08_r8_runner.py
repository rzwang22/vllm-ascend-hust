# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import copy
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dspark import p08_r8_checks as checks

ROOT = Path(__file__).parents[2]
PLUGIN_SHA = "a" * 40


def _result():
    record = {"runtime_mode": "FULL", "num_unpadded_tokens": 24, "num_padded_tokens": 24, "count": 2}
    interval = {
        "graph_replay_count": 2,
        "records": [record],
        "workers": [
            {"rank": rank, "graph_replay_count": 2, "records": [record], "failed_execution_count": 0}
            for rank in range(8)
        ],
    }
    return {
        "performance_eligible": False,
        "nan_diagnostic": {"enabled": True},
        "plugin_sha": PLUGIN_SHA,
        "core_sha": checks.CORE_SHA,
        "target_enforce_eager": False,
        "dspark_enforce_eager": True,
        "cudagraph_mode_effective": "FULL_DECODE_ONLY",
        "observed_capture_sizes": [6, 12, 18, 24],
        "measured_request_count": 4,
        "warmup_request_count": 4,
        "sampling": {"ignore_eos": True, "output_len": 256},
        "throughput": {"total_output_tokens": 1024},
        "outputs": [{"output_token_count": 256, "finish_reason": "length"} for _ in range(4)],
        "cleanup": {"engine_shutdown_complete": True},
        "measured_graph_replay_count": 2,
        "measured_eager_fallback_count": None,
        "graph_execution": {
            "source": "mrv2_successful_execute_model_full_replay",
            "replay_evidence_status": "available",
            "boundary_snapshots": [[], [], []],
            "warmup_runtime": copy.deepcopy(interval),
            "measured_runtime": copy.deepcopy(interval),
        },
    }


@pytest.mark.parametrize(
    "fault", ["rank", "zero_measured", "only_warmup", "tp_sum", "shape", "shutdown", "perf", "tokens"]
)
def test_diagnostic_result_gate_rejects_invalid_evidence(tmp_path, fault):
    result = _result()
    measured = result["graph_execution"]["measured_runtime"]
    if fault == "rank":
        measured["workers"].pop()
    elif fault in ("zero_measured", "only_warmup"):
        measured["graph_replay_count"] = 0
        measured["records"] = []
    elif fault == "tp_sum":
        result["measured_graph_replay_count"] *= 8
    elif fault == "shape":
        measured["records"][0]["num_padded_tokens"] = 18
    elif fault == "shutdown":
        result["cleanup"]["engine_shutdown_complete"] = False
    elif fault == "perf":
        result["performance_eligible"] = True
    else:
        result["outputs"][0]["output_token_count"] = 255
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        checks.result(path, PLUGIN_SHA)


def test_rank_index_preserves_partial_evidence_and_log_scan_fails(tmp_path):
    folder = tmp_path / "rank-diagnostics"
    folder.mkdir()
    report = {
        "rank": 7,
        "performance_eligible": False,
        "status": "ROOT_CAUSE_NOT_YET_PROVEN",
        "current": {"stage": "draft_hidden", "checks": []},
    }
    (folder / "rank-7-latest.json").write_text(json.dumps(report))
    (folder / "rank-7-first-failure.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="incomplete"):
        checks.diagnostics(tmp_path)
    index = json.loads((tmp_path / "diagnostic-index.json").read_text())
    assert index["missing_ranks"] == list(range(7))
    assert index["ranks"][0]["first_failure"] is True
    path = tmp_path / "graph.log"
    path.write_text("RuntimeError: Ascend DSpark Markov base logits contain NaN.\n")
    with pytest.raises(AssertionError, match="Runtime errors"):
        checks.scan(path)


_PYTHON_SHIM = r"""import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
if args[:2] == ['-m', 'pytest']:
    print('CPU shell-flow fixture: no hardware focused test executed')
    sys.exit(int(os.environ.get('FOCUSED_RC', '0')))
if len(args) > 1 and args[1] == 'source':
    assert os.environ['P08_CANN_RESTORED'] == '1'
    assert os.environ['ASCEND_CUSTOM_OPP_PATH'] == 'test-custom-opp'
    sys.exit(0)
if args[0].endswith('benchmark_dspark_acceptance.py'):
    assert os.environ['P08_CANN_RESTORED'] == '1'
    assert os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] == '0'
    out = pathlib.Path(args[args.index('--result-json') + 1])
    (out.parent / 'actual-argv.json').write_text(json.dumps(args))
    diag = pathlib.Path(args[args.index('--dspark-nan-diagnostic-dir') + 1])
    diag.mkdir()
    for rank in range(8):
        report = {'rank': rank, 'performance_eligible': False, 'status': 'ROOT_CAUSE_NOT_YET_PROVEN',
                  'current': {'stage': 'target_outputs', 'checks': []}}
        (diag / f'rank-{rank}-latest.json').write_text(json.dumps(report))
    rc = int(os.environ.get('GRAPH_RC', '0'))
    if rc:
        print('RuntimeError: injected diagnostic first failure')
    else:
        result = json.loads(pathlib.Path(os.environ['RESULT_FIXTURE']).read_text())
        if os.environ.get('BAD_REPLAY'):
            result['measured_graph_replay_count'] = 0
        out.write_text(json.dumps(result))
    sys.exit(rc)
sys.exit(subprocess.call([os.environ['REAL_PYTHON'], *args]))
"""


def _write_executable(path, source):
    path.write_text(source)
    path.chmod(0o755)


@pytest.mark.parametrize("fault", [None, "source", "dataset", "focused", "graph", "replay", "tee"])
def test_whole_shell_flow_preserves_environment_pipelines_and_failure_archive(tmp_path, fault):
    workspace = tmp_path / "workspace"
    plugin, core = workspace / "vllm-ascend-hust", workspace / "vllm-hust"
    tools = plugin / "tools/dspark"
    tools.mkdir(parents=True)
    core.mkdir()
    shutil.copyfile(ROOT / "tools/dspark/p08_r8_checks.py", tools / "p08_r8_checks.py")
    source_dir = workspace / "dspark-results/m2_5a-p08-r7-smallbatch.3AgHPg"
    source_dir.mkdir(parents=True)
    data = b'{"prompt_token_ids": [1, 2, 3]}\n' * 64
    (source_dir / "input-dataset.jsonl").write_bytes(data)
    cann = tmp_path / "cann.sh"
    cann.write_text("export P08_CANN_RESTORED=1\nexport ASCEND_CUSTOM_OPP_PATH=overwritten-by-cann\n")
    opp = plugin / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin"
    opp.mkdir(parents=True)
    (opp / "set_env.bash").write_text("export ASCEND_CUSTOM_OPP_PATH=test-custom-opp\n")
    script = (ROOT / "tools/dspark/run_p08_r8.sh").read_text().replace("/workspace", str(workspace))
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
    _write_executable(shim / "python", f"#!{sys.executable}\n" + _PYTHON_SHIM)
    _write_executable(shim / "npu-smi", '#!/bin/sh\nprintf "Mock NPU: shell control-flow only\\n"\n')
    _write_executable(
        shim / "git",
        f"""#!/bin/bash
if [[ "$*" == *"rev-parse HEAD"* ]]; then
    if [[ "$*" == *"/vllm-hust "* ]]; then printf '%s\\n' '{checks.CORE_SHA}'; else printf '%s\\n' '{PLUGIN_SHA}'; fi
elif [[ "$*" == *"branch --show-current"* ]]; then printf 'feat/dspark\\n'; fi
""",
    )
    if fault == "tee":
        _write_executable(
            shim / "tee",
            '#!/bin/bash\n/usr/bin/tee "$@"\nif [[ "$*" == *"focused.log"* ]]; then exit 9; fi\n',
        )
    fixture = tmp_path / "result.json"
    fixture.write_text(json.dumps(_result()))
    env = {
        **os.environ,
        "PATH": f"{shim}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "RESULT_FIXTURE": str(fixture),
        "FOCUSED_RC": "13" if fault == "focused" else "0",
        "GRAPH_RC": "7" if fault == "graph" else "0",
    }
    # Test restoration of an existing custom OPP value, not only the fallback.
    env["ASCEND_CUSTOM_OPP_PATH"] = "test-custom-opp"
    if fault == "replay":
        env["BAD_REPLAY"] = "1"
    completed = subprocess.run(
        ["bash", str(runner), "b" * 40 if fault == "source" else PLUGIN_SHA],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    dirs = list((workspace / "dspark-results").glob("m2_5a-p08-r8-diagnostic.*"))
    out = next(path for path in dirs if path.is_dir())
    assert Path(str(out) + "-evidence.tar.gz").is_file(), completed.stdout + completed.stderr
    assert Path(str(out) + "-evidence.sha256").is_file()
    gate = (out / "gate.txt").read_text()
    assert "PERFORMANCE_ELIGIBLE=false" in gate
    assert "DSPARK_PR_STYLE_BENCHMARK_PASS" not in completed.stdout
    if fault is None:
        assert completed.returncode == 0, completed.stdout + completed.stderr
    else:
        assert completed.returncode != 0
    if fault in ("source", "dataset", "focused", "tee"):
        assert not (out / "actual-argv.json").exists()
    else:
        argv = json.loads((out / "actual-argv.json").read_text())
        assert argv.count("--target-execution-mode") == 1
        assert argv[argv.index("--num-prompts") + 1] == argv[argv.index("--warmup-prompts") + 1] == "4"
        assert argv[argv.index("--output-len") + 1] == "256"
        i = argv.index("--cudagraph-capture-sizes")
        assert argv[i + 1 : i + 5] == ["6", "12", "18", "24"]
        assert (out / "graph-b4-pipestatus.txt").read_text().strip() == ("7 0" if fault == "graph" else "0 0")
    if fault == "tee":
        assert (out / "focused-pipestatus.txt").read_text().strip() == "0 9"
    if fault == "graph":
        assert "RESULT_GATE_RC=99" in gate and "ERROR_SCAN_RC=1" in gate
        assert not (out / "graph-b4-diagnostic.json").exists()


@pytest.mark.parametrize(
    "command,expected",
    [
        ("/env/bin/python /repo/tools/dspark/benchmark_dspark_acceptance.py --model-dir /model", True),
        ("python3 -u /repo/tools/dspark/benchmark_dspark_acceptance.py --model-dir /model", True),
        ("VLLM::EngineCore_DP0", True),
        ("VLLM::Worker_TP7_EP7", True),
        ("python /env/bin/pre-commit --files tools/dspark/benchmark_dspark_acceptance.py", False),
        ("python -m pytest tests/ut/test_dspark_acceptance_benchmark.py", False),
    ],
)
def test_idle_gate_matches_running_program_instead_of_lint_arguments(command, expected):
    assert checks._is_inference_command(command) is expected
