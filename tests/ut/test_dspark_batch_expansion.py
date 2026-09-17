# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU CostTable/CLI/receipt paths. Mock timings are not NPU evidence."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tests.ut import test_dspark_confidence_acceptance as confidence_fixtures
from tests.ut import test_dspark_formal_cost as fixtures
from tests.ut.test_dspark_confidence_acceptance import observer  # noqa: F401
from tools.dspark import batch_expansion as expansion
from tools.dspark import confidence_acceptance as confidence
from tools.dspark import formal_cost as formal
from tools.dspark import run_confidence_verification as child
from tools.dspark import run_large_batch as driver


def identity(batch):
    return {**fixtures.identity(), "max_num_seqs": batch, "capture_sizes": formal.captures(batch)}


def tier_raw(point, batch):
    raw = fixtures.raw(point)
    for rank in raw["ranks"]:
        rank["cost_profile"]["identity"] = identity(batch)
        for row in rank["cost_profile"]["measurements"]:
            row["request_capacity"] = min(batch, point["capacity"])
    return raw


def records_table(batch):
    rows = [
        {"point": p, "retained": formal.profile.point_samples(p, tier_raw(p, batch)["ranks"], 2, 5, 8)}
        for p in formal.plan(batch)["points"]
    ]
    return formal.profile.compile_startup(
        rows,
        identity(batch),
        formal.request_grid(batch),
        checkpoint={},
        plugin_sha="b" * 40,
        raw_hashes=["c" * 64] * len(rows),
        overhead=0.001,
    )


@pytest.mark.parametrize("batch,points,requests", [(128, 48, 2258), (256, 56, 4562)])
def test_all_cost_candidates_and_completion_tails(batch, points, requests):
    p = formal.plan(batch)
    assert p["point_count"] == points and p["total_requests"] == requests
    table = records_table(batch)
    costs = fixtures.CostTable.load_startup(table, identity(batch))
    for n in range(1, batch + 1):
        for tokens in range(n, 6 * n + 1):
            assert costs.cost(n, tokens, 640)[0] == next(c for c in formal.captures(batch) if c >= tokens)
    for args in ((batch + 1, batch + 1, 0), (batch, 6 * batch, 641)):
        with pytest.raises(ValueError):
            costs.cost(*args)
    with pytest.raises(ValueError):
        fixtures.CostTable.load_startup(table, fixtures.identity())
    del table["cells"][-1]
    with pytest.raises(ValueError):
        fixtures.CostTable.load_startup(table, identity(batch))


@pytest.mark.parametrize("batch", [128, 256])
def test_instances_preserve_all_source_tokens_and_sampling(batch):
    base = formal.workload_contract()
    rows = formal.workload_contract(batch)["records"]
    assert len(rows) == len({r["request_id"] for r in rows}) == batch
    assert len({r["source_case_id"] for r in rows}) == 64
    assert formal.workload_contract(batch)["sampling"] == base["sampling"]
    for i, row in enumerate(rows):
        old = base["records"][i % 64]
        assert row["original_request_id"] == old["request_id"]
        assert {k: row[k] for k in old if k != "request_id"} == {k: v for k, v in old.items() if k != "request_id"}


def args(tmp_path, batch=128):
    return NS(
        plugin=fixtures.ROOT,
        core=fixtures.ROOT.parent / "vllm-hust",
        model=Path("/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8"),
        plugin_sha="b" * 40,
        manifest=tmp_path / "manifest",
        output_dir=tmp_path,
        batch=batch,
        baseline=tmp_path / "baseline.tar.gz",
    )


@pytest.mark.parametrize("batch", [128, 256])
def test_real_parent_child_and_confidence_engine_configuration(tmp_path, monkeypatch, batch):
    a = args(tmp_path, batch)
    monkeypatch.setattr(driver, "run", lambda x: x)
    parsed = driver.main(expansion.cost_command(a, batch, tmp_path)[3:])
    monkeypatch.setattr(child, "run", lambda x: x)
    inner = child.main(driver.command(parsed, batch, tmp_path)[2:])
    assert inner.batch == batch and inner.capture == formal.captures(batch)
    assert inner.formal_cost_plan == formal.plan(batch)["name"] and inner.profile_samples == 5
    assert not inner.profile_experiment and not inner.profile_operator_capture
    _, kwargs = confidence.engine_config(a, tmp_path)
    assert kwargs["max_num_seqs"] == batch
    options = kwargs["additional_config"]["dspark_confidence_verification"]
    assert options["mode"] == "confidence" and options["profile"] is False
    assert kwargs["max_num_batched_tokens"] == 8192


def receipts(batch, executed):
    table = records_table(batch)
    table["loaded_confidence_weights"] = {"loaded_parameters": ["real.weight"], "weights_sha256": "hash"}
    pairs = [
        {"external_id": r["request_id"], "internal_id": f"internal-{i}"}
        for i, r in enumerate(formal.workload_contract(batch)["records"])
    ]
    all_ids = [p["internal_id"] for p in pairs]
    ids = all_ids[:executed]
    cap = next(c for c in formal.captures(batch) if c >= executed * 6)
    cell = min(
        (c for c in table["cells"] if c["capacity"] == cap and c["requests"] >= executed), key=lambda c: c["requests"]
    )
    seconds = cell["target_seconds"] + cell["draft_seconds"] + table["scheduler_seconds"]
    r, _, _ = confidence_fixtures.receipts()
    prefill = r[0]["confidence_execution_receipts"]["records"][0]
    prefill.update(
        context_upper=dict.fromkeys(all_ids, 0),
        scheduled_queries=dict.fromkeys(all_ids, 1),
        target=dict(
            request_ids=all_ids, query_lengths=[1] * batch, valid_tokens=batch, capacity=batch, full_replay=False
        ),
    )
    row = r[0]["confidence_execution_receipts"]["records"][1]
    row.update(
        context_upper=dict.fromkeys(ids, 100),
        scheduled_queries=dict.fromkeys(ids, 6),
        confidence={i: dict(producer_epoch=7, conditional=[0.9] * 5) for i in ids},
        graph=dict(mode="FULL", capacity=cap),
        target=dict(
            request_ids=ids, query_lengths=[6] * executed, valid_tokens=6 * executed, capacity=cap, full_replay=True
        ),
        accepted=dict(
            request_ids=ids,
            verified=[5] * executed,
            producer_epochs=[7] * executed,
            num_sampled=[3] * executed,
            status="available",
        ),
    )
    row["selection"].update(
        lengths=dict.fromkeys(ids, 5),
        producer_epochs=dict.fromkeys(ids, 7),
        confidence_epochs=dict.fromkeys(ids, 7),
        actual_tokens=6 * executed,
        selected_graph_capacity=cap,
        estimated_seconds=seconds,
    )
    row["lookup"].update(
        requests=executed,
        sampled_requests=cell["requests"],
        capacity=cap,
        estimated_seconds=seconds,
        token_budget=6 * executed,
    )
    for rank in r:
        rank["confidence_verification"].update(
            weights=table["loaded_confidence_weights"], cost_profile=dict(identity=table["identity"])
        )
        rank["confidence_execution_receipts"]["records"] = copy.deepcopy([prefill, row])
    return r, dict(request_id_mapping=dict(mappings=pairs, errors=[], hook_restored=True)), table


@pytest.mark.parametrize("batch", [128, 256])
def test_actual_concurrency_required_not_client_or_prefill_only(batch):
    ranks, stream, table = receipts(batch, batch)
    result = confidence.validate_receipts(ranks, stream, table, batch)
    assert result["capacity_coverage"][0]["requests"] == batch
    assert result["capacity_coverage"][0]["target_query_tokens"] == 6 * batch
    assert result["confidence_length_histogram"] == {5: batch}  # no manufactured variability
    ranks[7]["confidence_execution_receipts"]["records"][1]["accepted"]["producer_epochs"][0] = 6
    with pytest.raises(ValueError):
        confidence.validate_receipts(ranks, stream, table, batch)
    ranks, stream, table = receipts(batch, 64)
    with pytest.raises(ValueError, match="concurrency coverage incomplete"):
        confidence.validate_receipts(ranks, stream, table, batch)


@pytest.mark.parametrize("batch", [128, 256])
def test_real_publication_rebuild_and_same_commit_tier_load(tmp_path, monkeypatch, batch):
    root, update = fixtures.setup_publication(tmp_path, monkeypatch)
    new = root.with_name(f"b{batch}")
    root.rename(new)
    root = new
    for suffix in ("supervisor", "command", "residual"):
        (root.parent / f"b64-{suffix}.json").rename(root.parent / f"b{batch}-{suffix}.json")
    update(root / "capacity.json", unused=True)
    formal.benchmark._atomic_write_json(root / "capacity.json", capacity_rows(batch))
    p = formal.plan(batch)
    update(root / "plan.json", formal_cost=p, points=p["points"])
    preflight = formal.read(tmp_path / "formal-cost-preflight.json")
    preflight.update(
        plan=p,
        baseline={"archive_sha256": formal.CONFIDENCE_BASELINE_SHA},
        future_workload=formal.workload_contract(batch),
    )
    formal.benchmark._atomic_write_json(tmp_path / "formal-cost-preflight.json", preflight)
    capture = formal.read(root / "capture.json")
    capture.update(configured_capture_sizes=formal.captures(batch), observed_capture_sizes=formal.captures(batch))
    for w in capture["workers"]:
        w["observed_capture_sizes"] = formal.captures(batch)
    formal.benchmark._atomic_write_json(root / "capture.json", capture)
    records = []
    for point in p["points"]:
        raw = tier_raw(point, batch)
        path = root / (point["id"] + ".json")
        formal.benchmark._atomic_write_json(path, raw)
        records.append(
            dict(
                point=point,
                retained=formal.profile.point_samples(point, raw["ranks"], 2, 5, 8),
                raw_sha256=formal.sha(path),
            )
        )
    formal.benchmark._atomic_write_json(root / "retained.json", records)
    table = formal.profile.compile_startup(
        records,
        identity(batch),
        formal.request_grid(batch),
        checkpoint=preflight["weights"]["checkpoint"],
        plugin_sha="b" * 40,
        raw_hashes=[r["raw_sha256"] for r in records],
        overhead=0.001,
    )
    table["source"] = "unpublished_startup_npu_event_profile"
    formal.benchmark._atomic_write_json(root / "cost-profile.pending.json", table)
    proof = formal.publish(root, 0, "b" * 40, batch)
    assert proof["validated_candidate_lookups"] == 5 * batch * (batch + 1) // 2 + batch
    loaded, _ = confidence.publication(root, batch, "b" * 40)
    assert loaded["identity"]["max_num_seqs"] == batch
    with pytest.raises(ValueError):
        confidence.publication(root, batch, "c" * 40)
    with pytest.raises(ValueError):
        confidence.publication(root, 64)


def capacity_rows(batch):
    return [
        dict(
            rank=r,
            max_requests=batch,
            max_tokens=8192,
            draft_max_requests=batch,
            kv_num_blocks=1000,
            kv_bytes=1024000,
            groups=[{}],
            capture_sizes=formal.captures(batch),
        )
        for r in range(8)
    ]


def test_capacity_failure_and_bounded_receipts(observer, tmp_path):  # noqa: F811
    rows = capacity_rows(256)
    expansion.capacity_check(rows, 256)
    rows[1]["max_requests"] = 64
    with pytest.raises(ValueError):
        expansion.capacity_check(rows, 256)
    runner, state, old = confidence_fixtures.runner_fixture(observer, tmp_path)
    old.close()
    runner.vllm_config.scheduler_config = NS(max_num_seqs=256)
    obj = observer[0].ConfidenceReceipts(runner)
    assert obj.max_request_rows == 131072 and obj.max_log_bytes == 256 * 1024 * 1024
    obj.request_rows = obj.max_request_rows
    with pytest.raises(ValueError, match="bound"):
        runner.execute_model(
            NS(num_scheduled_tokens={"r": 6}, total_num_scheduled_tokens=6, scheduled_spec_decode_tokens={"r": [1] * 5})
        )
    obj.close()


@pytest.mark.parametrize("failure", [None, "b128-cost", "b128-confidence", "b256-cost"])
def test_sequential_driver_first_failure_no_retry(tmp_path, monkeypatch, failure):
    calls = []
    monkeypatch.setattr(confidence.suite, "source_gate", lambda _: None)

    def logged(cmd, log):
        name = log.stem
        calls.append(name)
        if name == "host":
            (tmp_path / "host.xml").write_text('<testsuite><testcase name="host"/></testsuite>')
        if name in ("b128-publish", "b256-publish"):
            batch = name.split("-")[0]
            confidence.write(tmp_path / batch / f"cost/runs/{batch}/cost-publication.json", {"status": "PASSED"})
        if name in ("b128-confidence", "b256-confidence") and name != failure:
            batch = name.split("-")[0]
            confidence.write(tmp_path / batch / "confidence/confidence-acceptance.json", {"overall_pass": True})
            confidence.write(tmp_path / batch / f"cost/runs/{batch}/cost-publication.json", {"status": "PASSED"})
        assert cmd[:3] == ["timeout", "--signal=TERM", "--kill-after=15s"]
        return int(name == failure)

    monkeypatch.setattr(confidence.suite, "logged", logged)
    assert expansion.run(args(tmp_path)) == int(failure is not None)
    assert len(calls) == len(set(calls))
    report = formal.read(tmp_path / "expansion-report.json")
    if failure:
        assert calls[-1] == failure and report["first_error"]
    else:
        assert list(report["tiers"]) == ["128", "256"] and report["overall_pass"]
    if failure == "b256-cost":
        assert report["tiers"]["128"]["status"] == "PASSED_THIS_RUN"


def test_capacity_rpc_reads_real_host_descriptors(observer):  # noqa: F811
    extension = sys.modules["vllm_ascend.diagnostics.dspark_benchmark_worker"].DSparkBenchmarkWorkerExtension
    cache = NS(
        num_blocks=100,
        kv_cache_tensors=[NS(size=4096, shared_by=["layer"], offset=0, block_stride=1024)],
        kv_cache_groups=[
            NS(layer_names=["layer"], kv_cache_spec=NS(block_size=32, page_size_bytes=1024), is_eagle_group=False)
        ],
    )
    worker = NS(
        rank=7,
        model_runner=NS(max_num_reqs=256, max_num_tokens=8192, speculator=NS(max_num_reqs=256), kv_cache_config=cache),
        dspark_benchmark_graph_runtime=lambda: {"observed_capture_sizes": formal.captures(256)},
    )
    result = extension.dspark_benchmark_capacity(worker)
    assert result["max_requests"] == 256 and result["kv_bytes"] == 4096
    assert result["groups"][0]["page_bytes"] == 1024 and result["rank"] == 7


def test_frozen_manifest_and_shell_bounds():
    root = fixtures.ROOT
    saved = formal.read(root / "tools/dspark/B128_B256_EXPANSION_PLAN.json")
    assert saved == expansion.plan()
    assert saved["model_initializations"] == 4 and saved["stage_budget_sum_seconds"] == 35780
    assert saved["group_runtime_limit_seconds"] == 36000
    assert saved["group_kill_margin_seconds"] == 65
    for name in (
        "run_dspark_batch_expansion.sh",
        "run_dspark_large_batch.sh",
        "run_dspark_swa_acceptance.sh",
        "run_dspark_exit_observation.sh",
    ):
        subprocess.run(["bash", "-n", str(root / "tools/dspark" / name)], check=True)
    assert "b64-gsm8k-confidence-v1" not in json.dumps(expansion.plan())
