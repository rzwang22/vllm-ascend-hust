# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts and real host publication path; timings here are mock data."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut import test_dspark_functional_coverage as fixtures
from tests.ut.test_dspark_shutdown_policy import configured_model
from tests.ut.test_dspark_startup_cost_profile import POLICY
from tools.dspark import formal_cost as formal
from tools.dspark import run_confidence_verification as child
from tools.dspark import run_large_batch as driver
from tools.dspark import shutdown_policy
from tools.dspark import startup_cost_profile as profile

CostTable = POLICY["CostTable"]
ROOT = Path(__file__).parents[2]


def identity():
    return dict(
        max_num_seqs=64,
        max_num_batched_tokens=8192,
        max_model_len=8192,
        capture_sizes=list(formal.coverage.CAPTURES),
        tp=8,
        ep=True,
        K=5,
        target_mode="FULL_DECODE_ONLY",
        draft_mode="eager",
        gpu_memory_utilization=0.9,
        hardware="CPU fixture",
        torch_version="test",
        torch_npu_version="test",
        confidence_weights_sha256="a" * 64,
        model="/fixture/model",
        revision=profile.suite.MODEL_REVISION,
        block_size=32,
        hf_config="hash",
        dtype="torch.bfloat16",
        quantization="ascend",
        cost_context_semantics=POLICY["COST_CONTEXT_SEMANTICS"],
    )


def raw(point):
    ranks = fixtures.rows(point)
    for rank in ranks:
        rank["cost_profile"].update(identity=identity(), observation=None)
        rank["confidence_verification"] = {
            "weights": {"loaded_parameters": ["head.weight"], "weights_sha256": "a" * 64}
        }
    return dict(
        point=point, ranks=ranks, streaming=fixtures.stream(point), request_identity_validation={"validated": True}
    )


def records_table():
    rows = [
        {"point": p, "retained": profile.point_samples(p, raw(p)["ranks"], 2, 5, 8)} for p in formal.plan()["points"]
    ]
    return profile.compile_startup(
        rows,
        identity(),
        list(formal.REQUEST_GRID),
        checkpoint={},
        plugin_sha="b" * 40,
        raw_hashes=["c" * 64] * len(rows),
        overhead=0.001,
    )


def test_minimal_bounded_plan_covers_every_reachable_candidate_and_tail():
    plan = formal.plan()
    assert plan == json.loads((ROOT / "tools/dspark/B64_FORMAL_COST_PLAN.json").read_text())
    assert (plan["point_count"], plan["total_requests"], plan["total_output_tokens"]) == (40, 1106, 566272)
    table = records_table()
    costs = CostTable.load_startup(table, identity())
    for n in range(64, 0, -1):
        for tokens in range(n, n * 6 + 1):
            for ctx in (0, 29, 116, 372, 640):
                cap, seconds = costs.cost(n, tokens, ctx)
                assert cap == next(c for c in formal.coverage.CAPTURES if c >= tokens) and seconds > 0
    for args in ((1, 1, 641), (65, 65, 128), (0, 1, 128), (1, 7, 128)):
        with pytest.raises(ValueError):
            costs.cost(*args)
    incomplete = copy.deepcopy(table)
    incomplete["cells"].pop()
    with pytest.raises(ValueError, match="coverage"):
        CostTable.load_startup(incomplete, identity())


def cli(tmp_path):
    return [
        "--plugin-sha",
        "b" * 40,
        "--manifest",
        "manifest",
        "--output-dir",
        str(tmp_path),
        "--stage",
        "profile",
        "--batches",
        "64",
        "--formal-cost-plan",
        formal.NAME,
        "--profile-worker-exit",
        "--profile-shutdown-policy",
        shutdown_policy.POLICY_NAME,
        "--profile-contexts",
        "128",
        "--capture-sizes",
        *map(str, formal.coverage.CAPTURES),
    ]


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--profile-experiment", "target-boundaries"],
        ["--profile-operator-capture"],
        ["--batches", "128"],
        ["--profile-contexts", "128", "2048"],
        ["--profile-exit-observation"],
        ["--profile-output-tokens", "128"],
    ],
)
def test_parent_and_child_cli_enforce_clean_timing_plan(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(driver, "run", lambda args: args)
    if extra:
        with pytest.raises(SystemExit):
            driver.main(cli(tmp_path) + extra)
    else:
        args = driver.main(cli(tmp_path))
        command = driver.command(args, 64, tmp_path)
        monkeypatch.setattr(child, "run", lambda args: args)
        parsed = child.main(command[2:])
        assert parsed.formal_cost_plan == formal.NAME and not parsed.profile_experiment
        assert not parsed.profile_operator_capture and parsed.profile_shutdown_policy == shutdown_policy.POLICY_NAME


def test_actual_engine_kwargs_reuse_exit_without_numeric_observers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        profile, "installed_budget", lambda *a, **k: shutdown_policy.budget(shutdown_policy.POLICY_NAME)
    )
    monkeypatch.setattr(
        profile.benchmark,
        "build_engine_kwargs",
        lambda _: {
            "additional_config": {"dspark_confidence_verification": {"mode": "specified_lengths", "profile": True}}
        },
    )
    kwargs = profile.profile_engine_kwargs(
        NS(),
        tmp_path / "worker",
        False,
        worker_exit=True,
        shutdown_policy=shutdown_policy.POLICY_NAME,
        formal_cost=True,
    )
    assert kwargs["distributed_executor_backend"].endswith("ProfileMultiprocExecutor")
    assert kwargs["worker_cls"].endswith("ProfileNPUWorker")
    config = kwargs["additional_config"]
    assert config["dspark_profile_stack_signals"] is False
    assert config["dspark_profile_exit_debugger"] is False
    assert "dspark_profile_observation" not in config and "dspark_profile_nan_diagnostic_dir" not in config
    assert config["dspark_profile_failure_dir"] == str(tmp_path.resolve())
    with pytest.raises(ValueError):
        profile.profile_engine_kwargs(
            NS(),
            tmp_path,
            False,
            experiment="baseline",
            worker_exit=True,
            shutdown_policy=shutdown_policy.POLICY_NAME,
            formal_cost=True,
        )


def setup_publication(tmp_path, monkeypatch):
    root = tmp_path / "runs/b64"
    root.parent.mkdir()
    update = configured_model(root)
    update(root / "worker-cleanup.json", force_events=[])
    update(root / "plan.json", formal_cost=formal.plan(), points=formal.plan()["points"], plugin_sha="b" * 40)
    update(root / "lifecycle.json", engine_initializations=1, shutdown=True)
    update(root / "checkpoint.json", source="test")
    capture = dict(
        configured_capture_sizes=list(formal.coverage.CAPTURES),
        observed_capture_sizes=list(formal.coverage.CAPTURES),
        npugraph_ex_enabled=True,
        workers=[
            dict(
                rank=r,
                target_cudagraph_mode="FULL_DECODE_ONLY",
                dspark_cudagraph_mode="NONE",
                observed_capture_sizes=list(formal.coverage.CAPTURES),
            )
            for r in range(8)
        ],
    )
    formal.benchmark._atomic_write_json(root / "capture.json", capture)
    checkpoint = formal.read(root / "checkpoint.json")
    provenance = dict(
        plan=formal.plan(),
        baseline={"archive_sha256": formal.ACCEPTED_SHA},
        plugin_sha="b" * 40,
        core_sha=profile.suite.CORE_SHA,
        weights={"model": "/fixture/model", "shards": [], "checkpoint": checkpoint},
        future_workload=formal.read(ROOT / "tools/dspark/B64_REAL_TEXT_PLAN.json"),
    )
    formal.benchmark._atomic_write_json(tmp_path / "formal-cost-preflight.json", provenance)
    monkeypatch.setattr(formal, "weight_identity", lambda _: provenance["weights"])
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(CostTable=CostTable))
    records = []
    for p in formal.plan()["points"]:
        data = raw(p)
        selected = profile.point_samples(p, data["ranks"], 2, 5, 8)
        path = root / (p["id"] + ".json")
        formal.benchmark._atomic_write_json(path, data)
        records.append(dict(point=p, retained=selected, raw_sha256=formal.sha(path)))
    formal.benchmark._atomic_write_json(root / "retained.json", records)
    table = profile.compile_startup(
        records,
        identity(),
        list(formal.REQUEST_GRID),
        checkpoint=checkpoint,
        plugin_sha="b" * 40,
        raw_hashes=[r["raw_sha256"] for r in records],
        overhead=0.001,
    )
    formal.benchmark._atomic_write_json(
        root / "scheduler-overhead.json",
        {
            "source": "host_allocate_prefixes_perf_counter",
            "unit": "seconds",
            "samples": [0.001] * 20,
            "median_seconds": 0.001,
        },
    )
    table["source"] = "unpublished_startup_npu_event_profile"
    formal.benchmark._atomic_write_json(root / "cost-profile.pending.json", table)
    with pytest.raises(ValueError):
        CostTable.load(str(root / "cost-profile.pending.json"), identity())
    return root, update


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "child_rc",
        "cleanup",
        "missing_exitcode",
        "log_scan",
        "raw_hash",
        "missing_cell",
        "candidate",
        "diagnostic",
        "unloaded_head",
        "capture",
        "weights",
        "host_overhead",
    ],
)
def test_real_publication_requires_samples_compatibility_and_cleanup(tmp_path, monkeypatch, fault):
    root, update = setup_publication(tmp_path, monkeypatch)
    if fault == "cleanup":
        update(root / "cleanup.json", forced_cleanup=True)
    elif fault == "missing_exitcode":
        update(
            root / "worker-cleanup.json", workers=[dict(rank=r, raw_exitcode=None if r == 0 else 0) for r in range(8)]
        )
    elif fault == "log_scan":
        update(root.parent / "b64-command.json", log_scan_rc=1)
    elif fault == "capture":
        update(root / "capture.json", observed_capture_sizes=[6])
    elif fault == "weights":
        monkeypatch.setattr(formal, "weight_identity", lambda _: {"changed": True})
    elif fault == "host_overhead":
        update(root / "scheduler-overhead.json", samples=[0.1])
    elif fault == "candidate":
        table = formal.read(root / "cost-profile.pending.json")
        table["cells"][0]["target_seconds"] *= 2
        formal.benchmark._atomic_write_json(root / "cost-profile.pending.json", table)
    elif fault == "missing_cell":
        records = formal.read(root / "retained.json")
        records.pop()
        formal.benchmark._atomic_write_json(root / "retained.json", records)
    elif fault in ("raw_hash", "diagnostic", "unloaded_head"):
        records = formal.read(root / "retained.json")
        p = root / (records[0]["point"]["id"] + ".json")
        if fault == "raw_hash":
            p.write_text(p.read_text() + " ")
        else:
            data = formal.read(p)
            if fault == "diagnostic":
                data["ranks"][0]["cost_profile"]["identity"]["diagnostic_only"] = True
            else:
                data["ranks"][0]["confidence_verification"]["weights"]["loaded_parameters"] = []
            formal.benchmark._atomic_write_json(p, data)
            records[0]["raw_sha256"] = formal.sha(p)
            formal.benchmark._atomic_write_json(root / "retained.json", records)
    if fault:
        with pytest.raises(ValueError):
            formal.publish(root, int(fault == "child_rc"), "b" * 40)
        assert not (root / "cost-profile.json").exists()
        assert formal.read(root / "cost-publication.json")["cost_table_usable"] is False
        assert (root / "cost-profile.pending.json").exists()
    else:
        result = formal.publish(root, 0, "b" * 40)
        assert result["cost_table_usable"] and result["validated_candidate_lookups"] == 10464
        assert result["table_sha256"] == formal.sha(root / "cost-profile.json")
        loaded = CostTable.load(str(root / "cost-profile.json"), identity())
        assert loaded.cost(63, 378, 372)[0] == 384
        with pytest.raises(ValueError, match="overwrite"):
            formal.publish(root, 0, "b" * 40)


def test_full_shard_hashing_is_not_just_head_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(formal, "checkpoint_preflight", lambda _: {"head": "same"})
    (tmp_path / "one.safetensors").write_bytes(b"weight-one")
    (tmp_path / "two.safetensors").write_bytes(b"weight-two")
    before = formal.weight_identity(tmp_path)
    (tmp_path / "two.safetensors").write_bytes(b"weight-TWO")
    after = formal.weight_identity(tmp_path)
    assert len(before["shards"]) == 2 and before["checkpoint"] == after["checkpoint"]
    assert before["shards"][1]["sha256"] != after["shards"][1]["sha256"]


def test_driver_uses_bounded_supervisor_without_debugger(tmp_path, monkeypatch):
    run = driver.run
    monkeypatch.setattr(driver, "run", lambda args: args)
    args = driver.main(cli(tmp_path / "result"))
    commands = []
    monkeypatch.setattr(driver.suite, "source_gate", lambda _: None)
    monkeypatch.setattr(driver, "read_manifest", lambda *a: ({}, [{}] * args.num_prompts, None))
    monkeypatch.setattr(driver, "input_population", lambda *a: {})
    monkeypatch.setattr(driver, "copy_manifest_assets", lambda *a: None)
    monkeypatch.setattr(driver.benchmark, "_sha256_file", lambda _: "hash")
    monkeypatch.setattr(driver.suite, "resources_idle", lambda *a: None)
    monkeypatch.setattr(driver, "scan", lambda *a: None)
    monkeypatch.setattr(driver.suite, "logged", lambda command, _: commands.append(command) or 0)
    assert run(args) == 0 and len(commands) == 1
    command = commands[0]
    assert "tools.dspark.profile_process_guard" in command
    assert command[command.index("--max-runtime-seconds") + 1] == "7200"
    assert "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=25" in command
    assert "--exit-observation" not in command and "--profile-experiment" not in command
    assert formal.read(args.output_dir / "summary.json")["performance_eligible"] is False


def test_publication_receipt_failure_leaves_no_usable_table(tmp_path, monkeypatch):
    root, _ = setup_publication(tmp_path, monkeypatch)
    write = formal.benchmark._atomic_write_json

    def fail_receipt(path, value):
        if path.name == "cost-publication.json":
            raise OSError("receipt disk error")
        return write(path, value)

    monkeypatch.setattr(formal.benchmark, "_atomic_write_json", fail_receipt)
    with pytest.raises(OSError, match="receipt disk error"):
        formal.publish(root, 0, "b" * 40)
    assert not (root / "cost-profile.json").exists()
    assert (root / "cost-profile.pending.json").exists()


def test_formal_collection_rejects_short_request_before_next_point(tmp_path, monkeypatch):
    points = formal.plan()["points"][:2]
    called = []

    class Engine:
        last_batch = {**fixtures.stream(points[0]), "scheduler": {}}

        def get_tokenizer(self):
            return NS(encode=lambda *a, **k: [90])

        def collective_rpc(self, method, kwargs=None):
            return raw(points[0])["ranks"]

        def generate(self, *a, **kw):
            called.append("generate")
            self.last_batch["requests"][0]["output_token_ids"].pop()
            return [None] * points[0]["requests"]

        def shutdown(self):
            called.append("shutdown")

    monkeypatch.setattr(profile, "validate_point_request_ids", lambda *a: {"test": True})
    with pytest.raises(ValueError, match="completion mismatch"):
        profile.collect(Engine, points, None, tmp_path, warmup=2, samples=5, require_completion=True)
    assert called == ["generate", "shutdown"]
    assert not (tmp_path / (points[1]["id"] + ".json")).exists()
    assert formal.read(tmp_path / "profile-failure.json")["point"] == points[0]


def test_host_overhead_retains_actual_samples_and_median(monkeypatch):
    from tools.dspark.verification_tools import measured_scheduler_overhead

    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(**POLICY))
    receipt = {}
    result = measured_scheduler_overhead(
        identity(), CostTable.load_startup(records_table(), identity()), receipt=receipt
    )
    assert len(receipt["samples"]) == 20 and all(v > 0 for v in receipt["samples"])
    assert result == formal.statistics.median(receipt["samples"]) == receipt["median_seconds"]
    assert "TP broadcast" in receipt["excludes"]
