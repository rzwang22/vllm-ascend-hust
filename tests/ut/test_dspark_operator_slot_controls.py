# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counterfactual byte accounting; no claim of native NPU execution."""

import copy
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.ut.test_dspark_operator_capture import ROOT, fixture
from tools.dspark import operator_replay as replay
from tools.dspark.operator_slot_controls import raw_bytes, replace_slot, save_watch


def pair(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    base, _ = f.run(1)
    donor = copy.deepcopy(base)
    # Make page3 a duplicate in the saved table; both copies must change.
    for c in (base, donor):
        v = c["values"]
        v["ori_block_table"][0, 3] = 3
        v["page_ids"][0, 3] = 3
        v["pages"][3].copy_(v["pages"][2])
    donor["values"]["pages"][[2, 3], 2] = torch.inf
    return base, donor


def test_counterfactual_changes_only_all_copies_of_one_slot(tmp_path, monkeypatch):
    base, donor = pair(tmp_path, monkeypatch)
    before = {k: raw_bytes(v) for k, v in base["values"].items()}
    result, record = replace_slot(base, donor, 3, 2)
    assert record["captured_page_indices_updated"] == [2, 3]
    assert record["slot_bytes"] == 16
    assert record["cache_storage_byte_range"] == [224, 240]
    assert result["identity"] == base["identity"] and result["layouts"] == base["layouts"]
    assert result["scalars"] == base["scalars"]
    for k, v in base["values"].items():
        assert raw_bytes(v) == before[k]
        if k != "pages":
            assert raw_bytes(result["values"][k]) == before[k]
    a, b = before["pages"], raw_bytes(result["values"]["pages"])
    allowed = {i for lo, hi in record["pages_payload_byte_ranges"] for i in range(lo, hi)}
    assert all(i in allowed for i, (x, y) in enumerate(zip(a, b)) if x != y)
    replay.validate(result)
    path = tmp_path / "counterfactual.pt"
    torch.save(result, path)
    assert torch.serialization.get_unsafe_globals_in_checkpoint(path) == []
    restored = replay.restore(torch.load(path, map_location="cpu", weights_only=True), "cpu")
    assert restored["ori_kv"][3, 2].isinf().all()
    reversed_result, _ = replace_slot(result, base, 3, 2)
    assert raw_bytes(reversed_result["values"]["pages"]) == before["pages"]


@pytest.mark.parametrize("fault", ["missing", "offset", "layout", "duplicates", "receipt", "dtype", "shape"])
def test_counterfactual_rejects_invalid_evidence(tmp_path, monkeypatch, fault):
    base, donor = pair(tmp_path, monkeypatch)
    block, offset = 3, 2
    if fault == "missing":
        block = 29
    elif fault == "offset":
        offset = 4
    elif fault == "layout":
        donor["layouts"]["ori_kv"]["npu_format"] = 29
    elif fault == "duplicates":
        donor["values"]["pages"][3, 2] = 0
    elif fault == "receipt":
        donor["values"]["receipts"].fill_(-1)
    elif fault == "shape":
        donor["values"]["pages"] = donor["values"]["pages"][..., :1].contiguous()
    else:
        donor["values"]["pages"] = donor["values"]["pages"].double()
    with pytest.raises(ValueError):
        replace_slot(base, donor, block, offset)


def test_watch_copies_before_later_reuse(tmp_path):
    watch = {"records": [], "snapshots": []}
    before = torch.ones(1, 512, dtype=torch.bfloat16)
    after = before.clone()
    after[0, 17] = torch.inf
    save_watch(watch, "replay-0", before, after)
    after.zero_()
    before.zero_()
    assert watch["snapshots"][0]["values"][1, 0, 17].isinf()
    assert watch["records"][0]["changed_byte_ranges"]
    assert watch["records"][0]["d2h_bytes"] == 2048


def test_formal_reference_cli_keeps_counterfactual_separate(tmp_path, monkeypatch):
    base, donor = pair(tmp_path, monkeypatch)
    # Finite intervention keeps the semantic reference well-defined.
    donor["values"]["pages"][[2, 3], 2] = 3
    for name, c in (("base", base), ("donor", donor)):
        torch.save(c, tmp_path / f"{name}.pt")
    base_bytes = (tmp_path / "base.pt").read_bytes()
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/dspark/operator_replay.py"),
            str(tmp_path / "base.pt"),
            "--slot-source",
            str(tmp_path / "donor.pt"),
            "--slot-block",
            "3",
            "--slot-offset",
            "2",
            "--mode",
            "reference",
            "--output",
            str(tmp_path / "result"),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "base.pt").read_bytes() == base_bytes
    report = json.loads((tmp_path / "result/result.json").read_text())
    assert report["intervention"]["slot_bytes"] == 16
    assert "not an observation" in report["captured_output_scope"]
    assert report["slot_watch"] is None
    assert not report["performance_eligible"]


def native_capture_semantics(directory):
    """Built-in operations only: distinguish deferred capture from a missing stream wait."""
    import torch_npu

    state = {
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "stages": [],
        "replays_submitted": 0,
        "replays_completed": 0,
    }
    # Record the actual installed wrapper bytes, not an assumed release implementation.
    state["graph_wrapper"] = replay.file_identity(inspect.getfile(torch_npu.npu.graph))
    try:
        counter = torch.zeros(1, dtype=torch.int32, device="npu")
        state["active_phase"] = "builtin-warmup"
        warm_counter = counter.clone()
        warm_before = warm_counter.clone()
        warm_counter.add_(1)
        warm_after = warm_counter.clone()
        state["warmup_values"] = torch.stack((warm_before, warm_after)).cpu()
        state["warmup_expected"] = [0, 1]
        state["warmup_completed"] = 1
        assert state["warmup_values"].flatten().tolist() == state["warmup_expected"]
        caller, capture_stream = torch.npu.current_stream(), torch.npu.Stream()
        capture_stream.wait_stream(caller)
        graph = torch.npu.NPUGraph()
        state["active_phase"] = "capture"
        with torch.npu.graph(graph, stream=capture_stream):
            before = counter.clone()
            counter.add_(1)
            after = counter.clone()
        # This local dependency rules out pending capture-stream work for the counter read.
        caller.wait_stream(capture_stream)
        captured_counter = counter.cpu().clone()
        state["stages"].append(
            {
                "phase": "capture",
                "counter": captured_counter,
                "expected_counter": 0,
                "replays_submitted": 0,
                "replays_completed": 0,
                "snapshot_valid": False,
                "values": None,
                "reason": "graph clone has not been executed by replay",
                "before_ptr": before.data_ptr(),
                "after_ptr": after.data_ptr(),
            }
        )
        assert captured_counter.item() == 0, "capture executed work: deferred-capture contract not established"
        for _ in range(3):
            prior = state["replays_completed"]
            state["active_phase"] = f"replay-{prior}"
            state["replays_submitted"] += 1
            graph.replay()
            packet = torch.stack((before, after, counter)).cpu()
            state["replays_completed"] += 1
            state["stages"].append(
                {
                    "phase": state["active_phase"],
                    "values": packet,
                    "snapshot_valid": True,
                    "expected": [prior, prior + 1, prior + 1],
                    "replays_submitted": state["replays_submitted"],
                    "replays_completed": state["replays_completed"],
                }
            )
            assert packet.flatten().tolist() == [prior, prior + 1, prior + 1]
        # Recheck owned CPU history after all graph-buffer reuse.
        for stage in state["stages"][1:]:
            assert stage["values"].flatten().tolist() == stage["expected"]
        state["active_phase"] = "complete"
    except BaseException as error:
        state["error"] = repr(error)
        raise
    finally:
        torch.save(state, directory / "capture-semantics.pt")


def test_npu_watch_observes_each_graph_replay(monkeypatch, tmp_path):
    """First prove capture semantics, then exercise the actual watched SWA/ACLGraph path."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires actual Ascend SWA and ACLGraph")
    from tests.ut.test_dspark_operator_npu import synthetic
    from tools.dspark import operator_slot_controls as controls

    watch = {"block": 70, "offset": 31, "snapshots": [], "records": []}
    checks = []
    state = {"expected_guard": 0, "guard_updates_submitted": 0, "checks": checks}
    try:
        native_capture_semantics(tmp_path)
        capture = synthetic(False)
        restored = {}
        original_restore, original_save = replay.restore, controls.save_watch

        def restore(capsule, device):
            result = original_restore(capsule, device)
            restored.update(result)
            return result

        def save(watch, phase, before, after):
            expected = state["expected_guard"]
            original_save(watch, phase, before, after)
            checks.append(
                {"phase": phase, "expected_guard": expected, "completed_call": watch["records"][-1]["completed_call"]}
            )
            # Advances only after a completed observation, never merely after capture.
            restored["ori_kv"][70, 31].add_(1)
            state["guard_updates_submitted"] += 1
            state["expected_guard"] += 1

        monkeypatch.setattr(replay, "restore", restore)
        monkeypatch.setattr(controls, "save_watch", save)
        outputs, _, _ = replay.run(capture, "aclgraph", "regenerated", watch=watch)
        state["outputs"] = outputs
        assert [r["phase"] for r in watch["records"]] == ["warmup", "replay-0", "replay-1", "replay-2"]
        assert watch["stages"][1]["status"] == "captured_not_replayed"
        assert not watch["stages"][1]["snapshot_valid"]
        assert watch["replays_submitted"] == 3 and watch["python_invocations"] == 2
        assert len({r["before_ptr"] for r in watch["records"][1:]}) == 1
        assert len({r["after_ptr"] for r in watch["records"][1:]}) == 1
        assert all(r["before_ptr"] != r["after_ptr"] for r in watch["records"])
        for snapshot, check in zip(watch["snapshots"], checks):
            assert snapshot["phase"] == check["phase"]
            assert (snapshot["values"] == check["expected_guard"]).all(), "completed call snapshot differs from guard"
        assert len(checks) == 4 and state["guard_updates_submitted"] == 4
        assert all(not r["changed_byte_ranges"] for r in watch["records"])
        assert all(x[:11].isfinite().all() for x in outputs)
    except BaseException as error:
        state["error"] = repr(error)
        raise
    finally:
        # Evidence precedes test success and survives assertion/dispatch failures.
        torch.save({"watch": watch, "expectations": state}, tmp_path / "native-watch-evidence.pt")
        controls.write_watch(watch, tmp_path)


@pytest.mark.parametrize("fail_phase", [None, "replay-1"])
def test_deferred_capture_never_observed_and_failure_preserves_history(tmp_path, fail_phase):
    """CPU scheduling regression with deliberately unexecuted capture buffers, not NPU proof."""
    from tools.dspark import operator_slot_controls as controls

    watch = {"records": [], "snapshots": []}
    state = {"capturing": False, "guard": 0, "before": None, "after": None}
    graph = SimpleNamespace()
    stream = SimpleNamespace(wait_stream=lambda other: None)

    def invoke():
        if state["capturing"]:
            state["before"] = torch.full((1, 512), torch.nan)
            state["after"] = torch.full((1, 512), torch.nan)
            graph.output = torch.full((1,), torch.nan)
            return graph.output
        state["before"] = torch.full((1, 512), float(state["guard"]))
        state["after"] = state["before"].clone()
        return torch.ones(1)

    def execute():
        if watch["active_phase"] == fail_phase:
            raise RuntimeError("injected replay failure")
        state["before"].fill_(state["guard"])
        state["after"].fill_(state["guard"])
        graph.output.fill_(1)

    @contextmanager
    def capture(g):
        state["capturing"] = True
        try:
            yield
        finally:
            state["capturing"] = False

    def observe(phase, executed):
        if not executed:
            controls.record_capture(watch)
            return
        controls.save_watch(watch, phase, state["before"], state["after"])
        state["guard"] += 1

    graph.replay = execute
    backend = SimpleNamespace(
        Stream=lambda: stream,
        current_stream=lambda: stream,
        stream=lambda s: nullcontext(),
        NPUGraph=lambda: graph,
        graph=capture,
    )
    try:
        with pytest.raises(RuntimeError, match="injected") if fail_phase else nullcontext():
            replay.execute_replays(invoke, observe, "aclgraph", backend, watch)
    finally:
        controls.write_watch(watch, tmp_path)
    saved = torch.load(tmp_path / "slot-watch.pt", weights_only=True)
    report = json.loads((tmp_path / "slot-watch.json").read_text())
    assert report["stages"][1]["snapshot_valid"] is False
    assert all(s["phase"] != "capture" for s in saved)
    assert len(saved) == (2 if fail_phase else 4)
    for s in saved:
        assert (s["values"] == s["completed_call"] - 1).all()
    assert report["replays_completed"] == (1 if fail_phase else 3)
    assert report["replays_submitted"] == (2 if fail_phase else 3)


def test_cli_failure_exports_partial_watch_and_preserves_error(tmp_path, monkeypatch):
    from tools.dspark import operator_slot_controls as controls

    base, _ = pair(tmp_path, monkeypatch)
    torch.save(base, tmp_path / "input.pt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "operator_replay",
            str(tmp_path / "input.pt"),
            "--output",
            str(tmp_path / "out"),
            "--watch-slot",
            "--slot-block",
            "3",
            "--slot-offset",
            "2",
        ],
    )

    def fail(capsule, mode, metadata, watch):
        before = torch.ones(1, 512, dtype=torch.bfloat16)
        controls.save_watch(watch, "warmup", before, before)
        controls.record_capture(watch)
        watch["active_phase"] = "replay-0"
        watch["replays_submitted"] = 1
        raise RuntimeError("original native failure")

    monkeypatch.setattr(replay, "run", fail)
    with pytest.raises(RuntimeError, match="original native failure"):
        replay.main()
    record = json.loads((tmp_path / "out/slot-watch.json").read_text())
    assert record["error"] == "RuntimeError('original native failure')"
    assert record["active_phase"] == "replay-0" and record["replays_completed"] == 0
    assert record["completed_calls"] == 1 and not record["stages"][1]["snapshot_valid"]
    packet = torch.load(tmp_path / "out/slot-watch.pt", map_location="cpu", weights_only=True)
    assert len(packet) == 1 and (packet[0]["values"] == 1).all()
