# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Execute the server script's control flow locally, with no server access."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
CORE_SHA = "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
PLUGIN_SHA = "25a1ceba0f3b222f881eeddb058c85d88b980e23"

SHIM = r"""
import hashlib
import json
import os
import sys
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
failure = os.environ['P08_TEST_FAILURE']
if name == 'git':
    if 'status' in args:
        pass
    elif 'rev-parse' in args:
        print(os.environ['P08_TEST_CORE'] if '-C' in args and args[1].endswith('/vllm-hust')
              else os.environ['P08_TEST_PLUGIN'])
    elif not any(x in args for x in ('switch', 'pull', 'merge-base')):
        raise AssertionError(args)
elif name == 'npu-smi':
    print('CPU test: no device queried')
elif name == 'sha256sum':
    digest, path = sys.stdin.read().strip().split('  ')
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
elif name == 'tee':
    if failure == 'tee' and args[0].endswith('/focused.log'):
        data = sys.stdin.read()
        Path(args[0]).write_text(data)
        print(data, end='')
        sys.exit(23)
    os.execv(os.environ['P08_TEST_TEE'], [os.environ['P08_TEST_TEE'], *args])
elif name == 'python':
    if args[:2] == ['-m', 'pytest']:
        assert 'tests/ut/attention/test_dsa_padding_contract.py' in args
        if os.environ['P08_TEST_REVISION'] in ('r5', 'r6'):
            assert 'tests/ut/attention/test_dsa_capture_validation.py' in args
        if os.environ['P08_TEST_REVISION'] == 'r6':
            assert 'tests/ut/test_dspark_graph_rpc.py' in args
            assert os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] == '0'
        print('CPU test: focused invocation recorded')
        sys.exit(7 if failure == 'focused' else 0)
    elif args[0].endswith('/benchmark_dspark_acceptance.py'):
        for option, expected in (
            ('--tensor-parallel-size', '8'), ('--num-spec-tokens', '5'),
            ('--cudagraph-capture-sizes', '6'), ('--num-prompts', '1'),
            ('--target-execution-mode', 'full_decode_only'),
        ):
            assert args[args.index(option) + 1] == expected
        assert '--enable-expert-parallel' in args and '--ignore-eos' in args
        result = {
            'plugin_sha': os.environ['P08_TEST_PLUGIN'],
            'core_sha': os.environ['P08_TEST_CORE'],
            'target_enforce_eager': False, 'dspark_enforce_eager': True,
            'cudagraph_mode_effective': 'FULL_DECODE_ONLY',
            'graph_capture_count': 1, 'observed_capture_sizes': [6],
            'measured_graph_replay_count': 2,
            'graph_execution': {'measured_runtime': {'records': [
                {'runtime_mode': 'FULL', 'num_unpadded_tokens': 6, 'num_padded_tokens': 6}]}},
            'acceptance': {'num_drafts': 1}, 'cleanup': {'engine_shutdown_complete': True},
        }
        if failure == 'replay_zero':
            result['measured_graph_replay_count'] = 0
        if failure == 'replay_eager':
            result['graph_execution']['measured_runtime']['records'][0]['runtime_mode'] = 'NONE'
        if failure == 'replay_shape':
            result['graph_execution']['measured_runtime']['records'][0]['num_unpadded_tokens'] = 1
        Path(args[args.index('--result-json') + 1]).write_text(json.dumps(result))
        print('CPU test: benchmark invocation recorded; no model executed')
        sys.exit(9 if failure == 'graph' else 0)
    elif args[0] == '-' and args[1].endswith('/vllm-ascend-hust'):
        sys.stdin.read()
        if os.environ['P08_TEST_REVISION'] == 'r6':
            assert os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] == '0'
        print('CPU test: source imports replaced')
    elif args[0] == '-' and args[1].endswith('/graph-p1.json'):
        # Execute the script's actual JSON assertions, including measured FULL.
        os.execv(sys.executable, [sys.executable, *args])
    else:
        raise AssertionError(args)
else:
    raise AssertionError(name)
"""


@pytest.mark.parametrize("failure", ["none", "focused", "graph", "tee", "replay_zero", "replay_eager", "replay_shape"])
@pytest.mark.parametrize("revision", ["r4", "r5", "r6"])
def test_server_script_preserves_failures_and_requires_measured_full(tmp_path, failure, revision):
    workspace = tmp_path / "workspace"
    plugin = workspace / "vllm-ascend-hust"
    plugin.mkdir(parents=True)
    (workspace / "vllm-hust").mkdir()
    data = workspace / "dspark-results/m2_5a-p08-r3-25a1ceba0.7jgir6/input-dataset.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"prompt_token_ids": [1, 2, 3]}\n')
    source = (ROOT / f"tools/dspark/run_p08_{revision}.sh").read_text()
    assert "set -e" not in source and "exit" not in source
    source = source.replace("/workspace", str(workspace)).replace(
        "6a2f629a5b5c9bbd9a3058b7a450fc18b2332f4699047f164cdde6a33b58d053",
        hashlib.sha256(data.read_bytes()).hexdigest(),
    )
    script = tmp_path / "run.sh"
    script.write_text(source)
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("git", "python", "npu-smi", "sha256sum", "tee"):
        path = commands / name
        path.write_text(f"#!{sys.executable}\n{SHIM}")
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "P08_TEST_FAILURE": failure,
        "P08_TEST_REVISION": revision,
        "P08_TEST_TEE": shutil.which("tee"),
        "P08_TEST_CORE": CORE_SHA,
        "P08_TEST_PLUGIN": PLUGIN_SHA,
    }
    command = ["bash", str(script)]
    if revision == "r6":
        command.append(PLUGIN_SHA)
        env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # Runner must explicitly pin 0.
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert (result.returncode == 0) == (failure == "none"), result.stdout + result.stderr
    outputs = list((workspace / "dspark-results").glob(f"m2_5a-p08-{revision}.*"))
    assert len(outputs) == 1
    out = outputs[0]
    gate = dict(line.split("=", 1) for line in (out / "gate.txt").read_text().splitlines())
    assert gate["SOURCE_GATE_RC"] == gate["DATASET_GATE_RC"] == "0"
    assert (out / "process-after.log").is_file() and (out / "npu-after.log").is_file()
    assert (out / "server-command.sh").read_text() == source
    assert (out / "focused-pipestatus.txt").read_text().strip() == {"focused": "7 0", "tee": "0 23"}.get(failure, "0 0")
    if failure in ("focused", "tee"):
        assert gate["GRAPH_P1_RC"] == "99" and not (out / "graph-p1.json").exists()
    else:
        assert (out / "graph-pipestatus.txt").read_text().strip() == ("9 0" if failure == "graph" else "0 0")
        assert (out / "graph-p1.json").is_file()
        assert gate["REPLAY_GATE_RC"] == ("99" if failure == "graph" else "0" if failure == "none" else "1")
    assert data.is_file()  # The previous run's evidence is never removed.


@pytest.mark.parametrize("fault", ["missing_plugin_sha", "wrong_plugin_sha", "wrong_core_sha"])
def test_r6_exact_source_gate_preserves_logs_and_does_not_run_graph(tmp_path, fault):
    # Exercise the complete runner first to create the strict command shims.
    test_server_script_preserves_failures_and_requires_measured_full(tmp_path, "none", "r6")
    workspace = tmp_path / "workspace"
    command = ["bash", str(tmp_path / "run.sh")]
    if fault != "missing_plugin_sha":
        command.append("0" * 40 if fault == "wrong_plugin_sha" else PLUGIN_SHA)
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "P08_TEST_FAILURE": "none",
        "P08_TEST_REVISION": "r6",
        "P08_TEST_TEE": shutil.which("tee"),
        "P08_TEST_CORE": "0" * 40 if fault == "wrong_core_sha" else CORE_SHA,
        "P08_TEST_PLUGIN": PLUGIN_SHA,
    }
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    outputs = list((workspace / "dspark-results").glob("m2_5a-p08-r6.*"))
    assert len(outputs) == 2  # Each invocation keeps a new directory, including failure.
    failed = next(path for path in outputs if not (path / "focused.log").exists())
    gate = dict(line.split("=", 1) for line in (failed / "gate.txt").read_text().splitlines())
    assert gate["SOURCE_GATE_RC"] != "0" and gate["GRAPH_P1_RC"] == "99"
    assert (failed / "source-pipestatus.txt").read_text().strip() == "1 0"
    assert (failed / "process-after.log").is_file() and (failed / "source.log").is_file()
