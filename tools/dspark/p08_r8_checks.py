# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Gates for the one-case P08-R8 diagnostic; never publish performance PASS."""

import argparse
import importlib.util
import json
import os
import socket
import subprocess
from pathlib import Path

import regex as re

CORE_SHA = "897306c43bf800e2480cb5c0f3e2da408d85a2fd"
RANKS = list(range(8))


def source(plugin: Path, core: Path) -> None:
    for name, root in (("vllm_ascend", plugin), ("vllm", core)):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.submodule_search_locations, name
        assert Path(next(iter(spec.submodule_search_locations))).resolve() == (root / name).resolve(), name
        print(name, root)
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "0"
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"], "Custom OPP environment is unavailable."
    # Worker-only dependencies: source validation runs in the server environment.
    import inspect

    import torch
    import torch_npu
    from vllm.utils.import_utils import resolve_obj_by_qualname

    for module, name in (
        ("dspark_benchmark_worker", "DSparkBenchmarkWorkerExtension"),
        ("dspark_nan", "DSparkNaNDiagnostics"),
    ):
        cls = resolve_obj_by_qualname(f"vllm_ascend.diagnostics.{module}.{name}")
        assert Path(inspect.getfile(cls)).resolve() == (plugin / f"vllm_ascend/diagnostics/{module}.py").resolve()
    assert callable(torch.npu.is_current_stream_capturing)
    print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
    print("ASCEND_CUSTOM_OPP_PATH", os.environ["ASCEND_CUSTOM_OPP_PATH"])


def _is_inference_command(command: str) -> bool:
    words = command.split()
    if not words:
        return False
    executable = Path(words[0]).name
    if any(marker in executable for marker in ("EngineCore", "VLLM::Worker", "Worker_TP")):
        return True
    if re.fullmatch(r"python[0-9.]*", executable):
        words = words[1:]
        while words and words[0].startswith("-"):
            words = words[1:]
        executable = Path(words[0]).name if words else ""
    # A lint command listing the benchmark filename is not a running engine.
    return executable == "benchmark_dspark_acceptance.py"


def idle() -> None:
    own = {os.getpid(), os.getppid()}
    rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
    residual = []
    for row in rows:
        fields = row.strip().split(maxsplit=1)
        if len(fields) == 2 and int(fields[0]) not in own and _is_inference_command(fields[1]):
            residual.append(row)
    print("RESIDUAL_PROCESSES", json.dumps(residual))
    assert not residual, "Inference processes remain; no dependent benchmark may start."


def scan(path: Path) -> None:
    pattern = re.compile(
        r"507057|MTE DDR address out of range|(?:NPU )?out of memory|"
        r"Traceback \(most recent call last\)|(?:RuntimeError|TypeError|AssertionError|ValueError):|"
        r"EngineDeadError|ChildFailedError|DSPARK_PREPARE_FAILURE|DSPARK_M2_5A_FAILURE|"
        r"contain(?:s)? NaN|non-finite values|without published ownership",
        re.I,
    )
    hits = [
        (number, line)
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        if pattern.search(line)
    ]
    print("ERROR_COUNT", len(hits))
    for number, line in hits:
        print(number, line)
    assert not hits, "Runtime errors found; diagnostic evidence is retained, not a successful graph run."


def diagnostics(root: Path) -> None:
    ranks = []
    missing = []
    for rank in RANKS:
        latest = root / "rank-diagnostics" / f"rank-{rank}-latest.json"
        first = latest.with_name(f"rank-{rank}-first-failure.json")
        if not latest.is_file():
            missing.append(rank)
            continue
        result = json.loads((first if first.is_file() else latest).read_text())
        assert result["rank"] == rank and result["performance_eligible"] is False
        assert result["status"] == "ROOT_CAUSE_NOT_YET_PROVEN"
        current = result["current"]
        ranks.append(
            {
                "rank": rank,
                "first_failure": first.is_file(),
                "phase": current.get("phase"),
                "stage": current.get("stage"),
                "target_execution_epoch": current.get("target_execution_epoch"),
                "proposal_epoch": current.get("proposal_epoch"),
                "request_ids": current.get("request_ids"),
                "actual_full_shapes": current.get("actual_full_shapes"),
                "checks": current.get("checks"),
                "exception": current.get("exception"),
            }
        )
    report = {
        "status": "ROOT_CAUSE_NOT_YET_PROVEN",
        "performance_eligible": False,
        "missing_ranks": missing,
        "ranks": ranks,
    }
    (root / "diagnostic-index.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("DIAGNOSTIC_RANKS", [item["rank"] for item in ranks], "MISSING_RANKS", missing)
    print("FIRST_FAILURE_RANKS", [item["rank"] for item in ranks if item["first_failure"]])
    assert not missing, "Rank diagnostic evidence incomplete; available files were retained."


def result(path: Path, plugin_sha: str) -> None:
    r = json.loads(path.read_text())
    assert r["performance_eligible"] is False and r["nan_diagnostic"]["enabled"] is True
    assert (r["plugin_sha"], r["core_sha"]) == (plugin_sha, CORE_SHA)
    assert r["target_enforce_eager"] is False and r["dspark_enforce_eager"] is True
    assert r["cudagraph_mode_effective"] == "FULL_DECODE_ONLY"
    assert r["observed_capture_sizes"] == [6, 12, 18, 24]
    assert r["measured_request_count"] == r["warmup_request_count"] == 4
    assert r["sampling"]["ignore_eos"] is True and r["sampling"]["output_len"] == 256
    assert r["throughput"]["total_output_tokens"] == 1024
    assert len(r["outputs"]) == 4
    assert all(row["output_token_count"] == 256 and row["finish_reason"] == "length" for row in r["outputs"])
    assert r["cleanup"]["engine_shutdown_complete"] is True
    graph = r["graph_execution"]
    assert graph["replay_evidence_status"] == "available"
    assert graph["source"] == "mrv2_successful_execute_model_full_replay"
    assert len(graph["boundary_snapshots"]) == 3
    for phase in ("warmup_runtime", "measured_runtime"):
        interval = graph[phase]
        assert sorted(worker["rank"] for worker in interval["workers"]) == RANKS
        assert interval["graph_replay_count"] > 0
        assert sum(row["count"] for row in interval["records"]) == interval["graph_replay_count"]
        assert all(
            worker["records"] == interval["records"]
            and worker["graph_replay_count"] == interval["graph_replay_count"]
            and worker["failed_execution_count"] == 0
            for worker in interval["workers"]
        )
        assert any(
            row["runtime_mode"] == "FULL"
            and row["count"] > 0
            and row["num_unpadded_tokens"] == 24
            and row["num_padded_tokens"] == 24
            for row in interval["records"]
        )
    assert graph["measured_runtime"]["graph_replay_count"] == r["measured_graph_replay_count"] > 0
    assert r["measured_eager_fallback_count"] is None
    print("DIAGNOSTIC_GENERATION_COMPLETED; ROOT_CAUSE_NOT_YET_PROVEN; NOT_PERFORMANCE_DATA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("source", "idle", "port", "scan", "diagnostics", "result"))
    parser.add_argument("values", nargs="*")
    args = parser.parse_args()
    if args.action == "source":
        source(Path(args.values[0]), Path(args.values[1]))
    elif args.action == "idle":
        idle()
    elif args.action == "port":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", int(args.values[0])))
    elif args.action == "scan":
        scan(Path(args.values[0]))
    elif args.action == "diagnostics":
        diagnostics(Path(args.values[0]))
    else:
        result(Path(args.values[0]), args.values[1])


if __name__ == "__main__":
    main()
