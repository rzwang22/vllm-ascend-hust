# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-call capsules on CPU; no claim of custom NPU kernel reproduction."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.test_dspark_replay_diagnostics import CPURecordedGraph
from tools.dspark import operator_replay as replay

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "operator_capture_test", ROOT / "vllm_ascend/diagnostics/dspark_profile_operator.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "runtime_identity", lambda: {"test": "CPU"})
    bank = NS(epoch_input=torch.zeros(1, dtype=torch.int64))
    capture = MODULE.OperatorCapture(bank, {"point": "selected", "max_tokens": 4, "max_seq_len": 12})
    q = torch.ones(4, 2, 4)
    cache = torch.randn(30, 4, 1, 4)
    # Extra current-page position and guard pages stay NaN, not zeroed away.
    cache[3, 3] = torch.nan
    kwargs = dict(
        ori_kv=cache,
        ori_block_table=torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [0] * 6, [0] * 6]),
        cu_seqlens_q=torch.tensor([0, 1, 3, 3, 3]),
        seqused_kv=torch.tensor([11, 6, 0, 0]),
        sinks=torch.zeros(2),
        metadata=torch.zeros(1024, dtype=torch.int32),
        softmax_scale=0.5,
        cmp_ratio=1,
        ori_mask_mode=4,
        ori_win_left=3,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
    )
    graph = CPURecordedGraph()
    output = torch.ones_like(q)
    capture.before(q, kwargs)  # allocate persistent buffers during warmup
    with graph:
        plan = capture.before(q, kwargs)
        capture.after(plan, output)

    def run(epoch, bad=False):
        bank.epoch_input.fill_(epoch)
        capture.reset()
        graph.replay()
        record = dict(
            point="selected",
            rank=0,
            execution=epoch,
            proposal_epoch=epoch - 1,
            request_ids=["short", "long"],
            query_start_loc_cpu=[0, 1, 3],
            graph_capacity=4,
            graph_object_id=1,
            head_flags=[[bad]],
            target_internal={"attention": {"kv": {"binding": {}}}},
        )
        pending = {"identity": record}
        capture.own(pending)
        capture.save(record, pending, tmp_path)
        return torch.load(tmp_path / f"rank-0-operator-{epoch}.pt", weights_only=True), pending

    return NS(capture=capture, bank=bank, q=q, cache=cache, kwargs=kwargs, graph=graph, run=run)


def test_real_values_full_tail_pages_and_repeated_receipts(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    for epoch in (20, 21, 22):
        f.q.add_(1)
        capsule, _ = f.run(epoch)
        assert capsule["values"]["receipts"].tolist() == [epoch] * 2
        assert torch.equal(capsule["values"]["q"], f.q)
        assert capsule["values"]["pages"][2, 3].isnan().all()
        assert capsule["values"]["page_ids"][0].tolist() == [1, 2, 3, 4, 5]
        restored = replay.restore(capsule, "cpu")
        assert restored["ori_kv"].stride() == f.cache.stride()
        assert restored["ori_kv"][3, 3].isnan().all()
        assert restored["ori_kv"][29].isnan().all()  # explicitly unknown, never original
        assert torch.isfinite(replay.reference(capsule)[:3]).all()


def test_owned_packet_before_reuse_and_first_error_freeze(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    for epoch in range(10, 15):
        c, pending = f.run(epoch, bad=epoch == 14)
    assert len(list(tmp_path.glob("*.pt"))) == 3
    assert [p.name for p in f.capture.history] == [f"rank-0-operator-{e}.pt" for e in (12, 13, 14)]
    saved = pending["operator_packets"][0]["packet"].clone()
    f.q.fill_(float("nan"))
    f.graph.replay()
    assert torch.equal(saved, pending["operator_packets"][0]["packet"])
    later = {"identity": {"point": "selected", "graph_capacity": 4}}
    f.capture.own(later)
    assert "operator_packets" not in later and f.capture.frozen
    assert json.loads((tmp_path / "rank-0-operator-index.json").read_text())["first_error_frozen"]


def test_missing_receipt_and_coverage_rejected(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    capsule, _ = f.run(1)
    capsule["values"]["receipts"][0] = -1
    with pytest.raises(ValueError, match="receipt"):
        replay.validate(capsule)
    capsule["values"]["receipts"].fill_(1)
    capsule["values"]["seqused_kv"][0] = 13
    with pytest.raises(ValueError, match="coverage"):
        replay.validate(capsule)


def test_reference_sink_scaling_and_masked_nan(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.q.zero_()
    f.cache.fill_(2)
    f.cache[3, 3] = torch.nan  # outside request0 at position10
    capsule, _ = f.run(1)
    ref = replay.reference(capsule)
    torch.testing.assert_close(ref[:3], torch.full((3, 2, 4), 2 * 4 / 5, dtype=torch.float64))
    assert ref[3].isnan().all()


def test_layout_id_not_relabelled_by_eager_and_default_scope(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.capture.before(f.q.clone(), f.kwargs)  # same shape, different source pointer
    c, _ = f.run(7)
    assert c["layouts"]["q"]["data_ptr"] == f.q.data_ptr()
    assert f.capture.before(torch.ones(5, 2, 4), f.kwargs) is None
    p = {"identity": {"point": "not-selected", "graph_capacity": 4}}
    f.capture.own(p)
    assert "operator_packets" not in p


@pytest.mark.parametrize("disk_error", [False, True])
def test_actual_target_path_saves_before_markov_and_preserves_first_error(tmp_path, monkeypatch, disk_error):
    from tests.ut.test_dspark_profile_attention import attention_factory
    from tests.ut.test_dspark_profile_target import target_fixture

    create = attention_factory(monkeypatch, "raw_attention")

    def with_capture(bank, fault):
        call = create(bank, fault)
        bank.attention_probe.operator = MODULE.OperatorCapture(
            bank, {"point": "test-point", "max_tokens": 12, "max_seq_len": 16}
        )
        return call

    monkeypatch.setattr(replay, "runtime_identity", lambda: {"test": "CPU actual DSA path"})
    f = target_fixture(tmp_path, monkeypatch, target_layer=1, attention_factory=with_capture)
    f.run(71)
    f.run(72)
    if disk_error:
        monkeypatch.setattr(MODULE.torch, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("disk test failure")))
    f.fault[0].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="Markov NaN"):
        f.run(73)
    first = json.loads((tmp_path / "rank-0-first-nan.json").read_text())
    if disk_error:
        assert "disk test failure" in first["recording_error"]
    else:
        assert first["recording_error"] is None
        paths = sorted(tmp_path.glob("rank-0-operator-*.pt"))
        assert len(paths) == 3
        c = torch.load(paths[-1], weights_only=True)
        assert c["identity"]["proposal_epoch"] == 73
        assert c["identity"]["request_ids"][0] == "third"
        assert c["values"]["output"][0].isnan().any()
    f.observer.close()


def test_operator_option_flows_through_driver_and_engine(tmp_path, monkeypatch):
    from tools.dspark import run_confidence_verification as driver
    from tools.dspark import run_large_batch as large
    from tools.dspark import startup_cost_profile as profile

    seen = []
    monkeypatch.setattr(large, "run", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(driver, "run", lambda args: seen.append(args) or 0)
    assert (
        large.main(
            [
                "--plugin-sha",
                "abc",
                "--manifest",
                str(tmp_path / "manifest"),
                "--output-dir",
                str(tmp_path),
                "--stage",
                "profile",
                "--batches",
                "64",
                "--profile-experiment",
                "target-boundaries",
                "--profile-target-layer",
                "1",
                "--profile-target-attention",
                "--profile-operator-capture",
            ]
        )
        == 0
    )
    command = large.command(seen[-1], 64, tmp_path)
    assert "--profile-operator-capture" in command
    assert driver.main(command[2:]) == 0 and seen[-1].profile_operator_capture
    monkeypatch.setattr(
        profile.benchmark,
        "build_engine_kwargs",
        lambda _: {
            "additional_config": {"dspark_confidence_verification": {"mode": "specified_lengths", "profile": True}}
        },
    )
    options = {"point": "chosen", "max_tokens": 12, "max_seq_len": 640}
    kw = profile.profile_engine_kwargs(None, tmp_path, False, "target-boundaries", 1, False, True, options)
    assert kw["additional_config"]["dspark_profile_observation"]["operator_capture"] == options
    with pytest.raises(ValueError, match="requires attention"):
        profile.profile_engine_kwargs(None, tmp_path, False, "target-boundaries", 1, False, False, options)


def test_duplicate_page_instability_is_not_a_valid_replay(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    capsule, _ = f.run(1)
    # Padded descriptor rows reference page zero repeatedly.
    capsule["values"]["pages"][-1].fill_(123)
    with pytest.raises(ValueError, match="Repeated page"):
        replay.validate(capsule)


def test_storage_offset_stride_and_alias_restore(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    capsule, _ = f.run(1)
    backing = torch.arange(30, dtype=torch.float32)
    q = backing.as_strided((4, 2, 4), (4, 0, 1), 2)
    sink = backing[2:4]
    capsule["layouts"]["q"] = MODULE.descriptor(q)
    capsule["layouts"]["sinks"] = MODULE.descriptor(sink)
    capsule["values"]["q"] = q.clone()
    capsule["values"]["sinks"] = sink.clone()
    # An overlapping destination cannot copy_ safely: reject unsupported aliases
    # explicitly rather than silently flattening the original layout.
    with pytest.raises(RuntimeError, match="single memory location"):
        replay.restore(capsule, "cpu")
    # A non-overlapping strided input retains its original offset.
    backing = torch.arange(80, dtype=torch.float32)
    q = backing.as_strided((4, 2, 4), (16, 4, 1), 2)
    sink = backing[2:4]
    capsule["layouts"]["q"], capsule["layouts"]["sinks"] = MODULE.descriptor(q), MODULE.descriptor(sink)
    capsule["values"]["q"], capsule["values"]["sinks"] = q.clone(), sink.clone()
    restored = replay.restore(capsule, "cpu")
    assert restored["q"].stride() == q.stride() and restored["q"].storage_offset() == 2
    assert restored["q"].untyped_storage().data_ptr() == restored["sinks"].untyped_storage().data_ptr()


@pytest.mark.parametrize("poison", [False, True])
def test_synthetic_control_has_finite_semantic_window_only(poison):
    from tests.ut.test_dspark_operator_npu import synthetic

    capsule = synthetic(poison)
    assert torch.isfinite(replay.reference(capsule)[:11]).all()
    assert capsule["identity"]["origin"] == "synthetic, not archived failure"


def test_unmapped_guard_is_preserved_but_required_page_is_rejected(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    f.kwargs["ori_block_table"][0, 4] = -1
    capsule, _ = f.run(1)
    assert capsule["unmapped_guard_entries"] == 1
    assert replay.restore(capsule, "cpu")["ori_block_table"][0, 4] == -1
    capsule["values"]["ori_block_table"][0, 1] = -1
    capsule["values"]["page_ids"][0, 1] = -1
    with pytest.raises(ValueError, match="required page"):
        replay.validate(capsule)
