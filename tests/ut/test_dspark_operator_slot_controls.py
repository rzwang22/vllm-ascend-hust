# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Counterfactual byte accounting; no claim of native NPU execution."""

import copy
import json
import os
import subprocess
import sys

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


def test_npu_watch_observes_each_graph_replay(monkeypatch, tmp_path):
    """Native SWA and ACLGraph, changing only a synthetic guard slot between calls."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires actual Ascend SWA and ACLGraph")
    from tests.ut.test_dspark_operator_npu import synthetic
    from tools.dspark import operator_slot_controls as controls

    capture = synthetic(False)
    restored = {}
    original_restore, original_save = replay.restore, controls.save_watch

    def restore(capsule, device):
        result = original_restore(capsule, device)
        restored.update(result)
        return result

    def save(watch, phase, before, after):
        original_save(watch, phase, before, after)
        restored["ori_kv"][70, 31].add_(1)  # outside synthetic semantic reads

    monkeypatch.setattr(replay, "restore", restore)
    monkeypatch.setattr(controls, "save_watch", save)
    watch = {"block": 70, "offset": 31, "snapshots": [], "records": []}
    outputs, _, _ = replay.run(capture, "aclgraph", "regenerated", watch=watch)
    assert [r["phase"] for r in watch["records"]] == ["warmup", "capture", "replay-0", "replay-1", "replay-2"]
    for i, snapshot in enumerate(watch["snapshots"]):
        assert (snapshot["values"] == i).all(), "graph watch is stale or overwritten"
    assert all(not r["changed_byte_ranges"] for r in watch["records"])
    assert all(x[:11].isfinite().all() for x in outputs)
    torch.save(watch, tmp_path / "native-watch-evidence.pt")
