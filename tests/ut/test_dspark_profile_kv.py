# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local KV write/window ownership, through captured ATen and actual DSA bodies."""

import json
import sys
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_attention import attention_factory
from tests.ut.test_dspark_profile_target import load_target, make_bank, target_fixture, torch
from tests.ut.test_dspark_replay_diagnostics import CPURecordedGraph


def fixture(monkeypatch, *, skip=False):
    bank = make_bank(load_target(monkeypatch), sizes=(6,), target_layer=1, attention=True)
    mod = sys.modules["vllm_ascend.diagnostics.dspark_profile_attention"]
    probe = mod.AttentionProbe(bank, 1)
    cache = torch.ones(12, 4, 1, 4)
    source = torch.full((6, 1, 4), 2.0)
    meta = NS(
        query_start_loc=torch.tensor([0, 1, 3, 3, 3, 3, 3]),
        seq_lens=torch.tensor([8, 5, 0, 0, 0, 0]),
        block_table=torch.tensor([[1, 2, 5], [3, 4, 6], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        slot_mapping=torch.tensor([[2, 3], [3, 3], [4, 0], [-1, -1], [-1, -1], [-1, -1]]),
    )
    graph = CPURecordedGraph()
    with graph:
        probe.kv.bind("model.layers.1.self_attn.attn", cache, meta, 4)
        probe.kv.scatter(cache, source, meta.slot_mapping, 0)
        if not skip:
            # Fixed valid rows avoid a test-only host read of dynamic indices.
            idx = meta.slot_mapping[:3]
            cache.index_put_((idx[:, 0], idx[:, 1]), source[:3])
        probe.kv.scatter(cache, source, meta.slot_mapping, 1)
        probe.window(cache, meta, 6, 4)

    def run(execution, names=("first", "second")):
        bank.epoch_input.fill_(execution)
        probe.kv.receipts.fill_(-1)
        graph.replay()
        packet = {}
        observer = NS(integer=lambda name, value: packet.update({name: value.clone().flatten().tolist()}))
        probe.kv.packet(observer, 6)
        assert probe.kv.fresh(packet, 6, execution)
        rows = [
            {
                "row": i,
                "request_row": int(probe.state[i, 0]),
                "request_id": names[int(probe.state[i, 0])],
                "valid_target_row": True,
                "position": int(probe.state[i, 1]),
            }
            for i in range(3)
        ]
        return probe.kv.decode(packet, rows, 6), packet

    return NS(bank=bank, probe=probe, cache=cache, source=source, meta=meta, graph=graph, run=run)


@pytest.mark.parametrize("skip", [False, True])
def test_source_vs_slot_before_after_and_window(monkeypatch, skip):
    f = fixture(monkeypatch, skip=skip)
    f.cache[2, 3] = torch.nan
    data, packet = f.run(10)
    row = data["rows"][0]
    assert row["before_scatter"]["source_nan"] == 0
    assert row["before_scatter"]["target_nan"] == 1
    assert row["after_scatter"]["target_nan"] == int(skip)
    assert row["after_scatter"]["differs_source"] == int(skip)
    if skip:
        bad = row["first_nonfinite"]
        assert (bad["logical_position"], bad["physical_block"], bad["offset"]) == (7, 2, 3)
        assert bad["current_writers"][0]["request_id"] == "first"
    else:
        assert row["first_nonfinite"] is None
    f.probe.kv.receipts.fill_(-1)
    assert f.probe.kv.fresh(packet, 6, 10)  # owned packet unaffected
    packet["kv.receipts"][2] = -1
    assert not f.probe.kv.fresh(packet, 6, 10)


def test_historical_slot_page_remap_and_first_logical_nonfinite(monkeypatch):
    f = fixture(monkeypatch)
    before, packet = f.run(20)
    f.cache[2, 0] = torch.nan
    f.cache[2, 1] = torch.inf
    after, _ = f.run(21)
    row = after["rows"][0]
    assert row["after_scatter"]["target_nan"] == row["after_scatter"]["differs_source"] == 0
    assert [s["logical_position"] for s in row["nonfinite_slots"]] == [4, 5]
    assert row["nonfinite_slots"][0]["nan"] and row["nonfinite_slots"][1]["inf"]
    assert row["first_nonfinite"]["current_writers"] == []
    saved = json.dumps(after)
    f.meta.block_table[0, 1] = 7
    f.meta.slot_mapping[0] = torch.tensor([7, 3])
    remapped, _ = f.run(22, names=("replacement", "second"))
    assert remapped["rows"][0]["first_nonfinite"] is None
    assert remapped["rows"][0]["page_runs"][0]["physical_block"] == 7
    assert remapped["rows"][0]["request_id"] == "replacement"
    assert before["rows"][0]["page_runs"][0]["physical_block"] == 2
    assert json.dumps(after) == saved and f.probe.kv.fresh(packet, 6, 20)


def test_duplicate_slots_and_invalid_padding_are_not_normal_writes(monkeypatch):
    f = fixture(monkeypatch, skip=True)
    f.meta.slot_mapping[1] = f.meta.slot_mapping[0]
    f.cache[0] = torch.nan  # safe diagnostic padding read must not contaminate window
    data, packet = f.run(30)
    assert data["rows"][0]["before_scatter"]["duplicate_slot"] == 1
    assert data["rows"][1]["before_scatter"]["duplicate_slot"] == 1
    assert not any(r["first_nonfinite"] for r in data["rows"])
    assert len(data["rows"]) == 3
    assert packet["kv.writes"][3 * 9 + 2] == 0  # padding slot_valid


def test_binding_catalog_cannot_be_relabelled_by_later_eager_call(monkeypatch):
    f = fixture(monkeypatch)
    f.probe.kv.bind("other.eager.attn", f.cache.clone(), f.meta, 4)
    data, _ = f.run(40)
    assert data["binding"]["layer_name"] == "model.layers.1.self_attn.attn"
    assert data["binding"]["tensors"]["cache"]["data_ptr"] == f.cache.data_ptr()
    assert len(f.probe.kv.bindings) == 2


@pytest.mark.parametrize("multistream", [False, True])
def test_actual_dsa_scatter_receipts_and_first_error_history(tmp_path, monkeypatch, multistream):
    f = target_fixture(
        tmp_path,
        monkeypatch,
        target_layer=1,
        attention_factory=attention_factory(monkeypatch, "raw_attention", multistream=multistream),
    )
    for epoch in (50, 51):
        f.run(epoch)
    f.fault[0] = torch.nan
    with pytest.raises(RuntimeError, match="Markov NaN"):
        f.run(52)
    d = json.loads((tmp_path / "rank-0-first-nan.json").read_text())
    rounds = d["auxiliary"]["rounds"]
    assert len(rounds) == 3
    for row in rounds:
        detail = row["target_internal"]["attention"]["kv"]
        assert detail["receipts"] == [row["execution"]] * 4
        first = detail["rows"][0]
        assert first["request_id"] == "third"
        assert first["after_scatter"]["differs_source"] == 0
    f.observer.close()


def test_missing_kv_receipt_fails_first_full_gate(tmp_path, monkeypatch):
    f = target_fixture(
        tmp_path, monkeypatch, target_layer=1, attention_factory=attention_factory(monkeypatch, "raw_attention")
    )
    replay = f.graph.replay

    def missing():
        replay()
        f.bank.attention_probe.kv.receipts[2] = -1

    monkeypatch.setattr(f.graph, "replay", missing)
    f.run(60)
    gate = json.loads((tmp_path / "rank-0-attention-validity.json").read_text())
    assert gate["status"] == "failed" and gate["kv_required"]
    assert gate["rounds"][0]["target_receipts"] == [1] * 21
    assert gate["rounds"][0]["kv_receipts"] == [1, 1, -1, 1]
    f.observer.close()
