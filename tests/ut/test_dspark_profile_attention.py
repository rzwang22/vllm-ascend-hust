# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual DSA/projection bodies and ATen replay with CPU kernel leaves."""

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_failure import core_path
from tests.ut.test_dspark_profile_target import ROOT, load_target, make_bank, target_fixture, torch
from tests.ut.test_dspark_replay_diagnostics import CPURecordedGraph, _production_method
from tools.dspark import run_confidence_verification as driver
from tools.dspark import run_large_batch as large
from tools.dspark import startup_cost_profile as profile


def source(path, cls, name, ns):
    return _production_method(ROOT / path, cls, name, ns)


def attention_factory(monkeypatch, failure, *, multistream=True, quantized=True):
    """Bind actual DSA/projection bodies; only accelerator leaves are CPU fixtures."""

    def create(bank, fault):
        """Prepare CPU leaf substitutes, then bind unmodified production method bodies."""
        mod = sys.modules["vllm_ascend.diagnostics.dspark_profile_attention"]
        probe = mod.AttentionProbe(bank, 1)
        bank.attention_probe = probe
        n = fault.shape[0]
        cos, sin = torch.ones(n, 1, 1, 2), torch.zeros(n, 1, 1, 2)
        starts = torch.tensor([0, 1, 5, 11] + [11] * (n - 3), dtype=torch.int32)
        seq = torch.tensor([1, 4, 6] + [0] * (n - 3), dtype=torch.int32)
        table = torch.arange(n * 4, dtype=torch.int32).reshape(n, 4)
        slots = torch.tensor(
            [[0, 0], [4, 0], [4, 1], [4, 2], [4, 3], [8, 0], [8, 1], [8, 2], [8, 3], [9, 0], [9, 1], [-1, -1]]
        )
        metadata = NS(
            cos={"attn": cos},
            sin={"attn": sin},
            query_start_loc=starts,
            seq_lens=seq,
            slot_mapping=slots,
            block_table=table,
            block_size=4,
            sas_metadata=torch.zeros(1024),
        )
        cache = torch.ones(n * 4, 4, 1, 2)
        route = NS(
            decode=metadata,
            num_prefills=0,
            num_decodes=n,
            num_decode_tokens=n,
            num_actual_tokens=n,
            cos=metadata.cos,
            sin=metadata.sin,
        )

        class Linear:
            def __init__(self, width):
                self.width = width
                self.weight = torch.eye(4)[:, :width]
                self.weight_scale = None
                self.bias = None
                self.quant_method = NS(quant_method=NS())

            def __call__(self, x):
                return x @ self.weight

        class CV:
            def __init__(self, linear):
                self.linear = linear

            def quantize(self, x):
                return x, None

            def matmul(self, x, scale):
                return self.linear(x)

        def qnorm(x):
            return x + (fault if failure == "q_normalized" else 0)

        qnorm.weight = torch.ones(4)
        stream = NS(record_event=lambda: None, wait_stream=lambda s: None, wait_event=lambda e: None)
        rotations = []

        def rope(x, c, s, **kwargs):
            i = len(rotations) % 3
            rotations.append(i)
            x.mul_(c)
            bad = failure == "inverse_rope" and i == 2
            bad |= failure == "q_rope" and x.shape[-2] == 2 and i != 2
            bad |= failure == "kv_rope" and x.shape[-2] == 1
            if bad:
                x.add_(fault[:, : x.shape[-2] * 2].reshape(x.shape))

        def scatter(c, values, mapping):
            valid = mapping[:, 0] >= 0
            c[mapping[valid, 0], mapping[valid, 1]] = values[valid]

        device = NS(
            unpack_dsa_forward_kv_cache=lambda caches, ratio: caches,
            apply_dsa_q_rms=lambda q, *a: q,
            dsa_kv_compress_scatter=scatter,
            get_dsa_sparse_attn_base_kwargs=lambda: {},
            get_dsa_sparse_attn_op=lambda: lambda q, **kw: (
                q + (fault.reshape(n, 2, 2) if failure == "raw_attention" else 0),
                None,
            ),
        )
        npu = NS(
            npu_dynamic_quant=lambda x: (x, None),
            npu_quant_matmul=lambda x, weight, *a, **kw: x @ weight,
            npu_transpose_batchmatmul=lambda x, *a, **kw: x + (fault[:, None, :] if failure == "wo_a" else 0),
        )
        backend = getattr(torch, "npu", None)
        if backend is None:
            backend = NS()
            monkeypatch.setattr(torch, "npu", backend, raising=False)
        monkeypatch.setattr(backend, "current_stream", lambda: stream, raising=False)
        monkeypatch.setattr(backend, "is_current_stream_capturing", lambda: False, raising=False)
        monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", rope, raising=False)
        monkeypatch.setattr(
            torch.ops._C_ascend, "npu_rms_norm_dynamic_quant", lambda x, *a, **kw: (qnorm(x), None), raising=False
        )
        monkeypatch.setattr(torch.ops.vllm, "maybe_all_gather_and_maybe_unpad", lambda x, gather: x, raising=False)
        ns = dict(
            torch=torch,
            torch_npu=npu,
            DeviceOperator=device,
            _EXTRA_CTX=NS(num_tokens=n),
            _require_decode_metadata=lambda m: m.decode,
            _is_w8a8_dynamic=lambda layer: quantized,
            dsv4_dsa_overlap_stream=lambda: stream,
            npu_stream_switch=lambda *a, **kw: nullcontext(),
            get_ascend_device_type=lambda: "A2",
            AscendDeviceType=NS(A5="A5"),
            oproj_tp_enable=lambda: False,
            olora_tp_enable=lambda: False,
        )
        names = ("forward", "_forward_decode", "_mla_prolog_multistream", "_forward_o_proj")
        cls = type(
            "CPUActualDSA",
            (),
            {name: source("vllm_ascend/attention/dsa_v1.py", "AscendDSAImpl", name, ns) for name in names},
        )
        impl = cls()
        impl._dspark_attn_probe = probe
        impl.n_local_heads, impl.head_dim, impl.nope_head_dim, impl.rope_head_dim = 2, 2, 0, 2
        impl.n_local_groups, impl.eps, impl.compress_ratio, impl.window_size, impl.softmax_scale = 1, 1e-6, 1, 4, 0.5
        impl.validate_dspark_sharedkv_contract = False
        impl.multistream_dsv4_dsa_overlap = multistream
        impl.wq_a, impl.wq_b, impl.wkv = Linear(4), Linear(4), Linear(2)
        impl.cv_wq_a, impl.cv_wq_b, impl.cv_wkv = CV(impl.wq_a), CV(impl.wq_b), CV(impl.wkv)
        impl.q_norm, impl.q_norm_without_weight = qnorm, None
        impl.kv_norm = lambda x: x + (fault[:, :2] if failure == "kv_normalized" else 0)
        impl.attn_sink, impl.wo_a = torch.zeros(2), Linear(4)
        row_forward = _production_method(
            core_path("model_executor/layers/linear.py"),
            "RowParallelLinear",
            "forward",
            {"tensor_model_parallel_all_reduce": lambda x: x * 2 + (fault if failure == "tp" else 0)},
        )
        row = type("CPUActualRow", (), {"forward": row_forward})
        row.__call__ = row.forward
        impl.wo_b = row()
        impl.wo_b._dspark_attn_probe = probe
        impl.wo_b.prefix, impl.wo_b.tp_size, impl.wo_b.tp_rank = "model.layers.1.self_attn.wo_b", 2, 0
        impl.wo_b.input_is_parallel, impl.wo_b.reduce_results = True, True
        impl.wo_b.skip_bias_add, impl.wo_b.return_bias, impl.wo_b.bias = False, False, None
        apply = source(
            "vllm_ascend/quantization/method_adapters.py",
            "AscendLinearMethod",
            "apply",
            {"torch": torch, "RowParallelLinear": row, "get_tensor_model_parallel_rank": lambda: 0},
        )
        adapter = type("Adapter", (), {"apply": apply})()
        adapter.quant_method = NS(apply=lambda layer, x, bias, rank: x + (fault if failure == "wo_b_local" else 0))
        impl.wo_b.quant_method = adapter
        impl.wo_b.custom_op = None
        probe.test_metadata, probe.test_cache = metadata, cache

        def call(*, hidden_states, **kwargs):
            output = torch.empty_like(hidden_states)
            return impl.forward("attn", hidden_states, (None, cache, None, None, None, None), [route], False, output)

        return call

    return create


@pytest.mark.parametrize(
    "failure",
    ["q_normalized", "q_rope", "kv_normalized", "kv_rope", "raw_attention", "inverse_rope", "wo_a", "wo_b_local", "tp"],
)
def test_actual_attention_replay_cuts_and_first_history(tmp_path, monkeypatch, failure):
    f = target_fixture(tmp_path, monkeypatch, target_layer=1, attention_factory=attention_factory(monkeypatch, failure))
    f.run(90)
    f.run(91)
    f.fault[0].fill_(torch.nan)
    # KV input failure is a mocked leaf; this CPU attention kernel deliberately
    # does not implement the hardware's KV reads. Inspect its input flags anyway.
    if failure.startswith("kv_"):
        f.run(92)
        d = f.observer.snapshot()
    else:
        with pytest.raises(RuntimeError, match="original Markov NaN"):
            f.run(92)
        d = json.loads((tmp_path / "rank-0-first-nan.json").read_text())
    rounds = d["auxiliary"]["rounds"]
    assert [a["proposal_epoch"] for a in rounds] == [90, 91, 92]
    a = rounds[-1]["target_internal"]["attention"]
    assert a["coverage"] == "FULL" and d["recording_error"] is None
    cuts = {b["name"].rsplit(".", 1)[1]: b for b in a["boundaries"]}
    assert len(cuts) == 12 and f.graph.calls == 3
    for r in rounds[:2]:
        assert not any(x["nan"] or x["inf"] for b in r["target_internal"]["boundaries"] for x in b["rows"])
    if failure == "tp":
        assert not any(x["nan"] for x in cuts["wo_b_local"]["rows"])
    else:
        row = cuts[failure]["rows"][0]
        assert row["nan"] and row["request_id"] == "third" and row["request_row"] == 0
    assert a["rows"][0]["slot_mismatch"] == 0
    assert a["rows"][11]["request_row"] == -1
    f.observer.close()


@pytest.mark.parametrize("multistream,quantized", [(False, False), (False, True), (True, False)])
def test_other_prolog_branches_preserve_receipts(tmp_path, monkeypatch, multistream, quantized):
    f = target_fixture(
        tmp_path,
        monkeypatch,
        target_layer=1,
        attention_factory=attention_factory(monkeypatch, "raw_attention", multistream=multistream, quantized=quantized),
    )
    f.run(11)
    a = f.observer.snapshot()["auxiliary"]["rounds"][-1]
    assert a["target_internal"]["attention"]["coverage"] == "FULL"
    assert a["target_internal"]["attention"]["route"]["multistream"] == multistream
    f.observer.close()


def test_kv_window_read_indices_padding_and_owned_receipts(monkeypatch):
    module = load_target(monkeypatch)
    bank = make_bank(module, target_layer=1, attention=True)
    mod = sys.modules["vllm_ascend.diagnostics.dspark_profile_attention"]
    probe = mod.AttentionProbe(bank, 1)
    cache = torch.ones(8, 4, 1, 2)
    metadata = NS(
        query_start_loc=torch.tensor([0, 1, 3, 3]),
        seq_lens=torch.tensor([8, 5, 0]),
        block_table=torch.tensor([[1, 2], [3, 4], [0, 0]]),
        slot_mapping=torch.tensor([[2, 3], [3, 3], [4, 0], [-1, -1]]),
    )
    graph = CPURecordedGraph()
    with graph:
        probe.window(cache, metadata, 4, 4)
    bank.epoch_input.fill_(7)
    cache[2, 0].fill_(torch.nan)
    graph.replay()
    index = bank.names.index("layer.1.attention.kv_window") - len(bank.outer_names)
    assert bank.attention_receipts[index].item() == 7 and bank.attention_flags[index, 0, 0]
    assert not bank.attention_flags[index, 3].any()
    saved = probe.state.clone()
    metadata.block_table[0, 1] = 1000
    bank.epoch_input.fill_(8)
    graph.replay()
    assert probe.state[0, -2] > 0 and probe.state[0, -1] == 1
    assert saved[0, -2] == 0 and saved[0, -1] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_attention_option_flows_to_actual_profile_kwargs(tmp_path, monkeypatch, enabled):
    args = [
        "--plugin-sha",
        "abc",
        "--manifest",
        str(tmp_path),
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
    ]
    seen = []
    monkeypatch.setattr(large, "run", lambda a: seen.append(a) or 0)
    monkeypatch.setattr(driver, "run", lambda a: seen.append(a) or 0)
    large.main(args + (["--profile-target-attention"] if enabled else []))
    cmd = large.command(seen[-1], 64, tmp_path)
    driver.main(cmd[2:])
    assert seen[-1].profile_target_attention == enabled
    monkeypatch.setattr(
        profile.benchmark,
        "build_engine_kwargs",
        lambda _: {
            "additional_config": {"dspark_confidence_verification": {"mode": "specified_lengths", "profile": True}}
        },
    )
    kw = profile.profile_engine_kwargs(None, tmp_path, False, "target-boundaries", 1, attention=enabled)
    assert kw["additional_config"]["dspark_profile_observation"].get("attention", False) == enabled


@pytest.mark.parametrize("mode,layer", [("baseline", 1), ("target-boundaries", None)])
def test_attention_option_rejects_wrong_plan(tmp_path, mode, layer):
    with pytest.raises(ValueError, match="requires target-boundaries"):
        profile.profile_engine_kwargs(None, tmp_path, False, mode, layer, attention=True)


@pytest.mark.parametrize("quantized", [False, True])
def test_projection_default_off_has_only_original_leaf(monkeypatch, quantized):
    calls = []
    x = torch.ones(2, 4)
    row = type("Row", (), {})
    layer = row()
    layer.prefix = "wo_b"
    layer.weight = x
    layer.tp_size = 2

    def gemm(*args, **kwargs):
        calls.append(args)
        return x

    if quantized:
        method = source(
            "vllm_ascend/quantization/method_adapters.py",
            "AscendLinearMethod",
            "apply",
            {"torch": torch, "RowParallelLinear": row, "get_tensor_model_parallel_rank": lambda: 0},
        )
        owner = NS(quant_method=NS(apply=gemm))
    else:
        method = source("vllm_ascend/ops/linear.py", "AscendUnquantizedLinearMethod", "apply", {"torch": torch})
        monkeypatch.setattr(torch.ops.vllm, "unquantized_gemm", gemm, raising=False)
        owner = NS()
    graph = CPURecordedGraph()
    with graph:
        result = method(owner, layer, x)
    assert result is x and len(calls) == 1 and not graph.operations


@pytest.mark.parametrize("fault", ["missing_cut", "stale_replay", "missing_route"])
def test_attention_missing_receipt_is_unavailable(tmp_path, monkeypatch, fault):
    f = target_fixture(
        tmp_path, monkeypatch, target_layer=1, attention_factory=attention_factory(monkeypatch, "raw_attention")
    )
    original = f.graph.replay

    def replay():
        if fault == "stale_replay":
            return
        original()
        if fault == "missing_cut":
            f.bank.receipts[-1].fill_(-1)
        else:
            f.bank.attention_probe.routes.clear()

    monkeypatch.setattr(f.graph, "replay", replay)
    f.run(10)
    d = f.observer.auxiliary_records[-1]["target_internal"]
    assert d["coverage"] == "INVALID_RECEIPT" and not d["boundaries"]
    assert not d["attention"]["rows"] and f.observer.recording_error
    f.observer.close()


def test_attention_inf_mapping_reorder_padding_and_owned_copy(tmp_path, monkeypatch):
    f = target_fixture(
        tmp_path, monkeypatch, target_layer=1, attention_factory=attention_factory(monkeypatch, "raw_attention")
    )
    f.run(9)
    f.fault[1].fill_(torch.inf)
    f.fault[-1].fill_(torch.nan)
    meta = f.bank.attention_probe.test_metadata
    meta.query_start_loc.fill_(5)
    meta.query_start_loc[:3] = torch.tensor([0, 2, 5])
    meta.seq_lens[:3] = torch.tensor([2, 3, 0])
    f.run(20, lengths=(2, 3), names=("replacement", "third"))
    r = f.observer.auxiliary_records[-1]
    b = next(b for b in r["target_internal"]["attention"]["boundaries"] if b["name"].endswith("raw_attention"))
    assert b["rows"][1]["inf"] and b["rows"][1]["request_id"] == "replacement"
    assert b["rows"][-1]["nan"] and b["rows"][-1]["request_id"] is None
    assert r["pool_rows_cpu"] == [1, 0]
    saved = json.dumps(r)
    f.bank.flags.zero_()
    f.bank.attention_probe.state.fill_(1000)
    f.bank.receipts.zero_()
    assert json.dumps(r) == saved
    first = (tmp_path / "rank-0-target-first-nonfinite.json").read_bytes()
    f.observer.failed("markov", RuntimeError("later duplicate"))
    assert (tmp_path / "rank-0-target-first-nonfinite.json").read_bytes() == first
    f.observer.close()


@pytest.mark.parametrize("unsupported", [None, "A5", "compressor", "oproj", "olora", "fused", "different_impl"])
def test_installation_selects_real_nested_impl_and_rejects_other_routes(monkeypatch, unsupported):
    module = load_target(monkeypatch)
    attention = sys.modules["vllm_ascend.diagnostics.dspark_profile_attention"]
    impl_type = type("AscendDSAImpl", (), {})
    impl = impl_type()
    impl.compress_ratio = 4 if unsupported == "compressor" else 1
    impl.wo_b = NS(custom_op=NS() if unsupported == "fused" else None)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.dsa_v1",
        NS(AscendDSAImpl=impl_type if unsupported != "different_impl" else type("Other", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.utils",
        NS(
            AscendDeviceType=NS(A5="A5"),
            get_ascend_device_type=lambda: unsupported,
            oproj_tp_enable=lambda: unsupported == "oproj",
            olora_tp_enable=lambda: unsupported == "olora",
        ),
    )
    model = NS(layers=[NS(), NS(self_attn=NS(dsa_attn=NS(dsa_attn=NS(impl=impl))))])
    bank = make_bank(module, target_layer=1, attention=True)
    if unsupported:
        with pytest.raises(ValueError, match="non-A5 SWA"):
            attention.install_attention_probe(bank, model, 1)
        assert bank.attention_probe is None
    else:
        attention.install_attention_probe(bank, model, 1)
        assert impl._dspark_attn_probe is impl.wo_b._dspark_attn_probe is bank.attention_probe
        assert bank.allocated_bytes == 47024


@pytest.mark.parametrize("stale", [False, True])
def test_first_full_validity_publication_is_owned_and_bounded(tmp_path, monkeypatch, stale):
    f = target_fixture(
        tmp_path,
        monkeypatch,
        target_layer=1,
        attention_factory=attention_factory(monkeypatch, "raw_attention"),
    )
    if stale:
        original = f.observer.after_replay

        def corrupt(desc):
            f.observer.bank.attention_receipts.fill_(-1)
            original(desc)

        monkeypatch.setattr(f.observer, "after_replay", corrupt)
    for epoch in (10, 11, 12):
        f.run(epoch)
    path = tmp_path / "rank-0-attention-validity.json"
    saved = path.read_bytes()
    data = json.loads(saved)
    assert data["status"] == ("failed" if stale else "passed")
    assert len(data["rounds"]) == (1 if stale else 3)
    assert data["rounds"][0]["target_receipts"][9:] == ([-1] * 12 if stale else [1] * 12)
    f.run(13)
    assert path.read_bytes() == saved
    f.observer.close()
