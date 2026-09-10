# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU behavior tests; artificial timings cannot be used as server measurements."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools.dspark import run_large_batch as large
from tools.dspark import run_performance_suite as suite
from tools.dspark import startup_cost_profile as profile

POLICY = runpy.run_path(str(Path(__file__).resolve().parents[2] / "vllm_ascend/spec_decode/dspark_verification.py"))
CostTable = POLICY["CostTable"]


def snapshots(point, *, count=8, ranks=2):
    result = []
    for rank in range(ranks):
        rows = []
        for kind in ("target", "draft"):
            for index in range(count):
                rows.append(
                    {
                        "point": point["id"],
                        "kind": kind,
                        "full_decode": True,
                        "requests": point["requests"],
                        "actual_tokens": point["actual_tokens"],
                        "capacity": point["capacity"],
                        "request_capacity": min(4, point["capacity"]),
                        "query_lengths": [ell + 1 for ell in point["lengths"]],
                        "context": 20,
                        "seconds": (100 if index < 2 else index + 1) / 1000,
                        "request_ids": [f"batch1-{i}" for i in range(point["requests"])],
                        "size": point["capacity"] if kind == "target" else point["requests"],
                    }
                )
        result.append(
            {
                "rank": rank,
                "failed_execution_count": 0,
                "cost_profile": {
                    "source": "isolated_npu_event_profile",
                    "measurements": rows,
                    "identity": {"tp": ranks},
                },
            }
        )
    return result


def table_data():
    counts, points = profile.grid(4, [6, 12, 24], [16, 128], 64)
    records = [{"point": p, "retained": profile.point_samples(p, snapshots(p), 2, 5, 2)} for p in points]
    identity = {"max_num_seqs": 4, "capture_sizes": [6, 12, 24], "tp": 2}
    return profile.compile_startup(
        records,
        identity,
        counts,
        checkpoint={"index": "quant"},
        plugin_sha="abc",
        raw_hashes=["real-in-production"],
        overhead=0.001,
    ), identity


def test_warmup_raw_samples_and_median():
    _, points = profile.grid(4, [6, 12, 24], [16], 64)
    rows = profile.point_samples(points[0], snapshots(points[0]), 2, 5, 2)
    assert len(rows) == 4
    assert rows[0]["median_seconds"] == 0.005
    assert [x["seconds"] for x in rows[0]["warmup"]] == [0.1, 0.1]
    assert rows[0]["extra_sample_count"] == 1
    with pytest.raises(ValueError, match="five"):
        profile.point_samples(points[0], snapshots(points[0]), 2, 4, 2)


@pytest.mark.parametrize("problem", ["missing_rank", "short", "failed", "nan", "eager", "wrong_layout"])
def test_invalid_measurements_not_credited(problem):
    _, points = profile.grid(4, [6, 12, 24], [16], 64)
    point = points[0]
    rows = snapshots(point)
    if problem == "missing_rank":
        rows.pop()
    elif problem == "short":
        rows[0]["cost_profile"]["measurements"] = rows[0]["cost_profile"]["measurements"][:2]
    elif problem == "failed":
        rows[0]["failed_execution_count"] = 1
    else:
        for r in rows[0]["cost_profile"]["measurements"]:
            if problem == "nan":
                r["seconds"] = float("nan")
            elif problem == "eager":
                r["full_decode"] = False
            else:
                r["query_lengths"] = [2]
    with pytest.raises(ValueError):
        profile.point_samples(point, rows, 2, 5, 2)


def test_cost_dimensions_padding_bounds_and_processing():
    data, identity = table_data()
    costs = CostTable.load_startup(data, identity)
    assert costs.cost(3, 15, 20) == (24, pytest.approx(0.011))
    assert costs.cost(3, 13, 20) == costs.cost(3, 18, 20)  # same physical graph tier
    assert costs.cost(4, 4, 180)[0] == 6  # no tokens//6 request inference
    with pytest.raises(ValueError, match="Context"):
        costs.cost(4, 24, 193)
    with pytest.raises(ValueError, match="coverage"):
        costs.cost(5, 25, 20)
    with pytest.raises(ValueError, match="layout"):
        costs.cost(3, 19, 20)
    assert data["cells"][0]["raw_layout_medians"]["balanced"]["target"] == 0.005
    assert data["model_initializations"] == 1 and data["performance_eligible"] is False


@pytest.mark.parametrize("problem", ["identity", "unit", "coverage", "load_count", "nan"])
def test_incompatible_profiles_rejected(problem):
    data, identity = table_data()
    if problem == "identity":
        identity = {**identity, "tp": 8}
    elif problem == "unit":
        data["unit"] = "milliseconds"
    elif problem == "coverage":
        data["cells"].pop()
    elif problem == "load_count":
        data["model_initializations"] = 4
    else:
        data["cells"][0]["draft_seconds"] = float("nan")
    with pytest.raises(ValueError):
        CostTable.load_startup(data, identity)


def test_same_engine_drains_points_and_always_shuts_down(tmp_path):
    _, points = profile.grid(2, [6, 12], [16], 64)
    points = points[:3]
    created = []

    class Engine:
        def __init__(self):
            created.append(self)
            self.current = None
            self.active = False
            self.ids = set()
            self.calls = 0
            self.closed = False
            self.last_batch = {"scheduler": {}}

        def get_tokenizer(self):
            return NS(encode=lambda text, add_special_tokens: [10])

        def collective_rpc(self, method, kwargs=None):
            assert isinstance(method, str) and not self.active
            if method == "dspark_benchmark_profile_point":
                assert set(kwargs) == {"point", "lengths"}
                self.current = next(p for p in points if p["id"] == kwargs["point"])
                return [{"rank": i, **kwargs} for i in range(2)]
            assert method == "dspark_benchmark_replay_snapshot" and kwargs is None
            result = snapshots(self.current) if self.current else []
            for row in result:
                for measurement in row["cost_profile"]["measurements"]:
                    measurement["request_ids"] = [r["request_id"] for r in self.last_batch["requests"]]
            return result

        def generate(self, prompts, sampling, use_tqdm):
            self.active = True
            self.calls += 1
            ids = {f"batch{self.calls}-{i}" for i in range(len(prompts))}
            assert not self.ids.intersection(ids)
            self.ids.update(ids)
            self.last_batch["requests"] = [{"request_id": key} for key in sorted(ids)]
            assert prompts[0]["prompt_token_ids"] == [10] * self.current["prompt_tokens"]
            self.active = False  # completion drained before next point RPC
            return [None] * len(prompts)

        def shutdown(self):
            self.closed = True

    records, _ = profile.collect(Engine, points, None, tmp_path, warmup=2, samples=5, ranks=2)
    assert len(created) == 1 and created[0].calls == 3 and created[0].closed
    assert len(records) == 3
    assert json.loads((tmp_path / "lifecycle.json").read_text())["engine_initializations"] == 1
    failing = tmp_path / "failed"
    failing.mkdir()
    with pytest.raises(ValueError):
        profile.collect(Engine, points, None, failing, warmup=2, samples=50, ranks=2)
    assert created[-1].closed and (failing / f"{points[0]['id']}.json").is_file()
    assert (failing / "profile-failure.json").is_file()


def test_large_grid_covers_every_request_and_capacity():
    for maximum in (64, 128, 256):
        sizes = large.captures(maximum)
        counts, points = profile.grid(maximum, sizes, [128], 512)
        cells = {(p["requests"], p["capacity"]) for p in points}
        for n in range(1, maximum + 1):
            for previous, cap in zip([0] + sizes[:-1], sizes):
                if n <= cap and 6 * n > previous:
                    assert any(req >= n and c == cap for req, c in cells)
        assert sizes[-1] == maximum * 6
        assert all(sum(x + 1 for x in p["lengths"]) == p["actual_tokens"] <= p["capacity"] for p in points)
        assert counts[-1] == maximum


def test_profile_wrapper_does_not_reset_production_state(monkeypatch):
    import sys

    import numpy as np
    import torch

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            pass

        def elapsed_time(self, end):
            return 2.5

    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "Event", Event)
        monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    else:
        monkeypatch.setattr(torch, "npu", NS(Event=Event, synchronize=lambda: None), raising=False)
    module_name = "vllm_ascend.worker.v2.spec_decode.dspark.verification_runtime"
    monkeypatch.setitem(sys.modules, module_name, NS(runtime_identity=lambda *args: {}))
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(**POLICY))
    Profiler = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "vllm_ascend/diagnostics/dspark_cost_profile.py")
    )["IsolatedCostProfiler"]
    slots = np.array([[-1, 31], [2, 4]])
    owners = {"old": 2}
    adaptive = NS(options={"mode": "specified_lengths", "profile": True})
    batch = NS(
        req_ids=["new"],
        num_reqs=1,
        num_tokens=6,
        num_reqs_after_padding=4,
        num_tokens_after_padding=6,
        num_computed_tokens_np=np.array([32]),
        num_scheduled_tokens=np.array([6]),
        is_prefilling_np=np.array([False]),
    )
    runner = NS(
        input_batch=batch,
        slots=slots,
        speculator=NS(
            confidence_verification=adaptive, _published_proposal_owners=owners, _execute_draft=lambda inputs: "draft"
        ),
        cudagraph_manager=NS(run_fullgraph=lambda desc: "target"),
    )
    profiler = Profiler(runner)
    profiler.begin_point("a", [5])
    assert profiler.target(NS(num_tokens=6)) == "target"
    assert profiler.propose(NS(num_reqs=1)) == "draft"
    assert all(r[0]["full_decode"] for r in profiler.events)
    profiler.begin_point("b", [0])
    assert not profiler.events
    assert runner.slots is slots and owners == {"old": 2}  # scheduler alone retires it
    profiler.propose(NS(num_reqs=1))  # no adjacent real FULL target
    assert not profiler.events[0][0]["full_decode"]
    with pytest.raises(ValueError):
        profiler.begin_point("bad", [-1])


def test_large_driver_parameters_and_alternating_processes(tmp_path):
    args = suite.parse_args(
        [
            "--plugin-sha",
            "abc",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path),
            "--max-num-seqs",
            "64",
            "--num-prompts",
            "400",
            "--client-outstanding",
            "128",
            "--modes",
            "dspark_graph",
            "dspark_confidence_graph",
            "--confidence-verification",
            str(tmp_path / "v.json"),
            "--repeats",
            "3",
        ]
    )
    plan = suite.create_plan(args, tmp_path / "requests.jsonl", tmp_path)
    assert len(plan["runs"]) == 6
    assert [r["mode"] for r in plan["runs"]] == [
        "dspark_graph",
        "dspark_confidence_graph",
        "dspark_confidence_graph",
        "dspark_graph",
        "dspark_graph",
        "dspark_confidence_graph",
    ]
    for row in plan["runs"]:
        cmd = row["command"]
        assert cmd[cmd.index("--num-prompts") + 1] == "400"
        assert cmd[cmd.index("--max-num-seqs") + 1] == "64"
        assert cmd[cmd.index("--client-outstanding") + 1] == "128"
    assert len({r["directory"] for r in plan["runs"]}) == 6
    assert copy.deepcopy(plan) == plan


def test_import_preserves_frozen_order_tokens_and_rejects_short_duplicate_data():
    from tools.dspark.prepare_performance_data import digest, import_frozen_records

    tokenizer = NS(decode=lambda tokens, skip_special_tokens: str(tokens))
    kwargs = dict(
        count=2, max_input_tokens=32, source="frozen-gsm", revision="source-hash", token_field="ids", id_field="id"
    )
    raw = [{"ids": [3, 2], "id": "z"}, {"ids": [7, 1], "id": "a"}]
    rows, _ = import_frozen_records(raw, tokenizer, **kwargs)
    assert [r["case_id"] for r in rows] == ["z", "a"]
    assert [r["prompt_token_ids"] for r in rows] == [[3, 2], [7, 1]]
    assert rows[0]["prompt_token_sha256"] == digest([3, 2])
    with pytest.raises(ValueError, match="Insufficient"):
        import_frozen_records(raw[:1], tokenizer, **kwargs)
    with pytest.raises(ValueError, match="Duplicate"):
        import_frozen_records([raw[0], raw[0]], tokenizer, **kwargs)
