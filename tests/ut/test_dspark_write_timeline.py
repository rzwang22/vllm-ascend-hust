# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded writer timeline: CPU mechanisms and optional real opaque NPU replay."""

import ast
import importlib.util
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.ut.test_dspark_operator_capture import MODULE, ROOT


def load(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_operator", MODULE)
    spec = importlib.util.spec_from_file_location(
        "write_timeline_test", ROOT / "vllm_ascend/diagnostics/dspark_write_timeline.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Target:
    window_size = 128


def setup(monkeypatch, device="cpu"):
    mod = load(monkeypatch)
    if device == "cpu":
        monkeypatch.setattr(
            torch,
            "npu",
            NS(is_current_stream_capturing=lambda: False, current_stream=lambda: NS(npu_stream=0)),
            raising=False,
        )
    target = Target()
    cache = torch.ones((8, 32, 1, 512), dtype=torch.bfloat16, device=device)
    target.swa_cache_layer = NS(prefix="layer.swa_cache", kv_cache=cache)
    epoch = torch.tensor([7], dtype=torch.int64, device=device)
    watch = mod.SlotWriteTimeline(NS(epoch_input=epoch), target, {"point": "point", "max_tokens": 12})
    md = NS(
        query_start_loc=torch.tensor([0, 1, 5, 11], device=device),
        seq_lens=torch.tensor([223, 255, 530], device=device),
        input_positions=torch.tensor([222, 251, 252, 253, 254, 524, 525, 526, 527, 528, 529, 0], device=device),
        block_table=torch.tensor([[0, 1, 3, 4, 5, 6, 7] * 3] * 3, device=device),
    )
    context = NS(attn_metadata={"layer.swa_cache": NS(decode=md)})
    watch.control.copy_(torch.tensor([0, 1], device=device))
    return mod, target, watch, context


def test_dynamic_binding_and_no_scalar_extraction(monkeypatch):
    mod, target, w, c = setup(monkeypatch)

    class NoScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert "_local_scalar_dense" not in str(func)
            return func(*args, **(kwargs or {}))

    with NoScalar():
        b = w.binding(c.attn_metadata)
        v = w.read(b)
    assert b.tolist() == [0, 222, 95, 3, 31, 1, 7] and v.numel() == 1024
    c.attn_metadata["layer.swa_cache"].decode.block_table[0, 2] = 6
    assert w.binding(c.attn_metadata)[3].item() == 6
    w.control[0] = 1
    assert w.binding(c.attn_metadata)[5].item() == 0  # four queries, not selected single-query request
    assert mod.overlap([0, 8], [7, 10]) and not mod.overlap([0, 8], [8, 10])


def test_mutable_alias_detected_readonly_and_disjoint_ignored(monkeypatch):
    mod, target, w, c = setup(monkeypatch)
    with w.observe("layer", c, 12):
        target.swa_cache_layer.kv_cache.sum()
        torch.ones(2).add_(1)
    assert not w.plans
    slot = target.swa_cache_layer.kv_cache[3, 31].view(torch.float32)
    with w.observe("layer", c, 12):
        slot.fill_(2)
    p = next(iter(w.plans.values()))
    assert not torch.equal(p["buffers"]["before"], p["buffers"]["after"])
    assert p["buffers"]["receipt"].tolist() == [7, 7]
    assert p["targets"][0][1]["dtype"] == "torch.float32"
    w.options["max_tokens"] = 6
    with w.observe("disabled-capacity", c, 12):
        slot.fill_(3)
    assert len(w.plans) == 1


def test_owned_packet_survives_reuse_and_first_change_is_retained(tmp_path, monkeypatch):
    _, target, w, c = setup(monkeypatch)
    runner = NS(
        input_batch=NS(seq_lens_np=np.array([223, 255, 530])),
        compilation_config=NS(static_forward_context={"layer.swa_cache": target.swa_cache_layer}),
        kv_cache_config=NS(kv_cache_groups=[]),
    )
    for execution in range(1, 6):
        w.epoch.fill_(execution)
        identity = {
            "point": "point",
            "execution": execution,
            "rank": 0,
            "request_ids": ["a", "b", "c"],
            "query_start_loc_cpu": [0, 1, 5, 11],
            "graph_capacity": 12,
            "proposal_epoch": execution + 10,
        }
        pending = {"identity": identity}
        w.begin(pending, runner, execution + 20, c.attn_metadata)
        # Same bytes through first two calls, corruption on third.
        with w.observe("writer", c, 12):
            target.swa_cache_layer.kv_cache[3, 31].fill_(2 if execution >= 3 else 1)
        w.own(pending)
        if w.pending is None:
            continue
        for p in w.plans.values():
            p["buffers"]["before"].zero_()
        w.host("draft.after")
        w.save(identity | {"device_integers": {"target.positions": [222]}}, pending, tmp_path)
    assert w.first_change == 3 and w.following_saved
    paths = sorted(tmp_path.glob("*-writes-*.pt"))
    assert len(paths) == 4
    report = torch.load(paths[2], weights_only=True)
    assert report["sites"][0]["changed"] and report["identity"]["execution"] == 3
    assert (tmp_path / "rank-0-writes-catalog.json").exists()


def load_page_trace():
    # Only PageTrace is exercised here; actual AsyncScheduler remains installation-only.
    path = ROOT / "vllm_ascend/diagnostics/dspark_write_scheduler.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.module != "vllm.v1.core.sched.async_scheduler"
    ]
    namespace = {"AsyncScheduler": object}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["PageTrace"]


def test_real_computed_forwarded_and_free_iterator_not_eagerly_consumed(tmp_path):
    PageTrace = load_page_trace()

    class Scheduler:
        pass

    s = Scheduler()
    trace = PageTrace(s, tmp_path, "point")
    trace.active = True

    class Manager:
        def remove_skipped_blocks(self, request_id, total_computed_tokens):
            self.received = total_computed_tokens

        def free_blocks(self, ordered_blocks, prepend=False):
            self.order = []
            for block in ordered_blocks:
                assert block.ref_cnt == 1
                block.ref_cnt -= 1
                self.order.append(block.block_id)

    m = Manager()
    trace.wrap(m, "remove_skipped_blocks", "remove_skipped_blocks", 2)
    m.remove_skipped_blocks("a", 228)
    assert m.received == 228
    trace.wrap(m, "free_blocks", "pool.free")
    seen = []

    def values():
        for i in (9, 7):
            seen.append(i)
            yield NS(block_id=i, ref_cnt=1, is_null=False, block_hash=None)
            assert m.order[-1] == i

    m.free_blocks(values(), prepend=True)
    assert seen == m.order == [9, 7]
    events = [json.loads(x) for x in trace.path.read_text().splitlines()]
    assert events[0]["arguments"]["total_computed_tokens"] == 228
    assert events[-1]["freed_before"][0]["ref_cnt"] == 1 and events[-1]["freed_blocks"][0]["ref_cnt"] == 0


def test_npu_writer_probe_inside_opaque_dispatch_replays(monkeypatch, tmp_path):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires actual NPU ACLGraph")
    _, target, w, c = setup(monkeypatch, "npu")
    lib = torch.library.Library("dspark_write_probe_test", "FRAGMENT")
    lib.define("write(Tensor(a!) cache) -> ()")

    def write(cache):
        with w.observe("opaque", c, 12):
            cache[3, 31].view(torch.float32).fill_(2)

    lib.impl("write", write, "PrivateUse1")
    op = torch.ops.dspark_write_probe_test.write
    data = {"records": []}
    try:
        op(target.swa_cache_layer.kv_cache)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            op(target.swa_cache_layer.kv_cache)
        data["capture"] = "not consumed; no valid replay receipt yet"
        for epoch in (11, 12, 13):
            w.epoch.fill_(epoch)
            for p in w.plans.values():
                p["buffers"]["receipt"].fill_(-1)
            target.swa_cache_layer.kv_cache[3, 31].fill_(epoch)
            graph.replay()
            p = next(iter(w.plans.values()))
            owned = {k: v.cpu().clone() for k, v in p["buffers"].items()}
            data["records"].append(owned)
            assert owned["receipt"].tolist() == [epoch, epoch]
            assert not torch.equal(owned["before"], owned["after"])
            assert owned["before"].view(torch.bfloat16).eq(epoch).all()
        for epoch, packet in zip((11, 12, 13), data["records"]):
            assert packet["before"].view(torch.bfloat16).eq(epoch).all()
    except BaseException as error:
        data["error"] = repr(error)
        raise
    finally:
        torch.save(data, tmp_path / "writer-preflight.pt")


def test_installed_core_page_removal_uses_actual_computed(tmp_path):
    pytest.importorskip("vllm")
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

    from vllm_ascend.diagnostics.dspark_write_scheduler import PageTrace

    class Scheduler:
        pass

    class Pool:
        def free_blocks(self, ordered_blocks, prepend=False):
            for b in ordered_blocks:
                b.ref_cnt -= 1

    scheduler = Scheduler()
    trace = PageTrace(scheduler, tmp_path, "point")
    trace.active = True
    manager = SlidingWindowManager.__new__(SlidingWindowManager)
    manager.block_size = 32
    manager.sliding_window = 128
    manager.block_pool = Pool()
    manager._null_block = NS(block_id=0, ref_cnt=0, is_null=True, block_hash=None)
    owned = [NS(block_id=i + 1, ref_cnt=1, is_null=False, block_hash=None) for i in range(8)]
    manager.req_to_blocks = {"a": list(owned)}
    trace.wrap(manager, "remove_skipped_blocks", "remove_skipped_blocks", 1)
    trace.wrap(manager.block_pool, "free_blocks", "pool.free")
    manager.remove_skipped_blocks("a", 222)
    assert manager.req_to_blocks["a"][2] is owned[2] and owned[2].ref_cnt == 1
    manager.remove_skipped_blocks("a", 228)
    assert manager.req_to_blocks["a"][2] is manager._null_block and owned[2].ref_cnt == 0
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    calls = [e for e in events if e["event"] == "call.begin" and e["call"] == "remove_skipped_blocks"]
    assert [e["arguments"]["total_computed_tokens"] for e in calls] == [222, 228]


@pytest.mark.parametrize("fault", ["receipt", "mapping", "missing"])
def test_invalid_packets_saved_before_failure(tmp_path, monkeypatch, fault):
    _, target, w, c = setup(monkeypatch)
    runner = NS(
        input_batch=NS(seq_lens_np=np.array([223, 255, 530])),
        compilation_config=NS(static_forward_context={}),
        kv_cache_config=NS(kv_cache_groups=[]),
    )
    identity = {
        "point": "point",
        "execution": 7,
        "rank": 0,
        "request_ids": ["a", "b", "c"],
        "query_start_loc_cpu": [0, 1, 5, 11],
        "graph_capacity": 12,
    }
    pending = {"identity": identity}
    w.begin(pending, runner, 20, c.attn_metadata)
    if fault != "missing":
        with w.observe("writer", c, 12):
            target.swa_cache_layer.kv_cache[3, 31].fill_(1)
    if fault == "receipt":
        next(iter(w.plans.values()))["buffers"]["receipt"].fill_(-1)
    w.own(pending)
    record = identity | {"device_integers": {"target.positions": [221 if fault == "mapping" else 222]}}
    with pytest.raises(ValueError, match="raw packets saved"):
        w.save(record, pending, tmp_path)
    saved = torch.load(tmp_path / "rank-0-writes-7.pt", weights_only=True)
    assert saved["coverage"] == "UNAVAILABLE" and saved["unavailable"]
    assert w.first_unavailable == 7


def test_prefill_and_unselected_point_no_packets(monkeypatch):
    _, target, w, c = setup(monkeypatch)
    c.attn_metadata["layer.swa_cache"].decode = None
    with w.observe("prefill", c, 12):
        target.swa_cache_layer.kv_cache.fill_(1)
    assert not w.plans


def test_cpu_opaque_dispatch_and_epoch_reuse(monkeypatch):
    _, target, w, c = setup(monkeypatch)
    lib = torch.library.Library("dspark_cpu_write_probe_test", "FRAGMENT")
    lib.define("write(Tensor(a!) cache) -> ()")

    def write(cache):
        with w.observe("opaque", c, 12):
            cache[3, 31].fill_(2)

    lib.impl("write", write, "CPU")
    for epoch in (1, 2, 3):
        w.epoch.fill_(epoch)
        for p in w.plans.values():
            p["buffers"]["receipt"].fill_(-1)
        torch.ops.dspark_cpu_write_probe_test.write(target.swa_cache_layer.kv_cache)
        p = next(iter(w.plans.values()))
        assert p["buffers"]["receipt"].tolist() == [epoch, epoch]


def test_budget_error_does_not_mask_original(tmp_path):
    PageTrace = load_page_trace()

    class Scheduler:
        pass

    class Manager:
        def fail(self):
            trace.sequence = 30000
            raise RuntimeError("original engine failure")

    s = Scheduler()
    trace = PageTrace(s, tmp_path, "point")
    trace.active = True
    m = Manager()
    trace.wrap(m, "fail", "fail")
    with pytest.raises(RuntimeError, match="original engine failure") as exc:
        m.fail()
    assert "Page trace error" in exc.value.__notes__[0]
    assert json.loads(trace.path.read_text().splitlines()[-1])["event"] == "TRUNCATED"


@pytest.mark.parametrize("enabled", [False, True])
def test_writer_scheduler_is_opt_in(tmp_path, monkeypatch, enabled):
    from tools.dspark import startup_cost_profile as profile

    monkeypatch.setattr(
        profile.benchmark,
        "build_engine_kwargs",
        lambda _: {
            "additional_config": {"dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"}}
        },
    )
    options = {"point": "point", "max_tokens": 12, "write_timeline": True} if enabled else None
    result = profile.profile_engine_kwargs(
        None, tmp_path, False, "target-boundaries", 1, attention=True, operator_capture=options
    )
    assert ("scheduler_cls" in result) == enabled


def test_writer_preflight_gates_model_and_preserves_point_order():
    text = (ROOT / "tools/dspark/run_dspark_large_batch.sh").read_text()
    assert text.index("logged writer-preflight") < text.index("logged focused") < text.index("logged generation")
    assert "('failure', 'error', 'skipped')" in text
    control = (ROOT / "tools/dspark/run_dspark_profile_control.sh").read_text()
    assert "--profile-stop-after-point ctx128-n4-t12-skewed" in control
    assert "--write-timeline) detail_args+=(--profile-write-timeline)" in control
