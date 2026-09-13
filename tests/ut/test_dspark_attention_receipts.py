# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual dispatcher + plugin AOT entry, with CPU and opt-in-device replay."""

import ast
import functools
import json
import os
import sys
from types import SimpleNamespace as NS

import pytest
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_func
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple

from tests.ut.test_dspark_profile_target import ROOT, load_target, torch
from tests.ut.test_dspark_replay_diagnostics import bank as replay_bank


def function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(namespace)
    exec(
        compile(
            ast.Module(body=[*ast.parse("from __future__ import annotations").body, node], type_ignores=[]),
            path,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_real_dispatch_aot_copyback_and_repeated_replay(tmp_path, monkeypatch, shared, device):
    """Legacy shared storage must reproduce failure; separated storage must pass.

    CPU executes the real functionalized graph repeatedly. NPU additionally
    captures that graph and replays without Python model/dispatcher entry.
    Actual DSA dispatcher/metadata/cache selection are used with a tiny leaf;
    no model weights, sparse-attention kernel or original NaN reproduction.
    """
    if device == "npu":
        pytest.importorskip("torch_npu")
        pytest.importorskip("npugraph_ex")
        if not torch.npu.is_available():
            pytest.skip("requires an available Ascend NPU for actual ACLGraph replay")
    sym_min_binding = torch.sym_min
    module = load_target(monkeypatch)
    assert module.torch is torch and sys.modules["torch"] is torch
    assert module.torch.sym_min is sym_min_binding
    attention = sys.modules["vllm_ascend.diagnostics.dspark_profile_attention"]
    bank = module.TargetBoundaryFlags(
        sizes=(6,),
        auxiliary_layers=(40, 41, 42),
        start_layer=0,
        end_layer=43,
        hidden_size=4,
        hc_mult=2,
        device=device,
        target_layer=1,
        attention=True,
    )
    if shared:
        # Precisely the old storage relation: outer copy-back spans all 21 rows.
        bank.flags = torch.zeros((21, 6, 2), dtype=torch.bool, device=device)
        bank.receipts = torch.full((21, 1), -1, dtype=torch.int64, device=device)
        bank.attention_flags = bank.flags[9:]
        bank.attention_receipts = bank.receipts[9:]
    probe = attention.AttentionProbe(bank, 1)
    meta = NS(
        query_start_loc=torch.tensor([0, 6], device=device),
        seq_lens=torch.tensor([12], device=device),
        block_table=torch.arange(64, device=device).reshape(1, 64),
        slot_mapping=torch.zeros((6, 2), dtype=torch.int32, device=device),
    )
    cache = torch.ones(64, 4, 1, 4, device=device)
    calls = []

    def leaf(layer, x, caches, metadata, gather, output):
        calls.append(1)
        assert caches[1] is cache and metadata[0] is meta
        probe.kv.bind("selected.attn", cache, meta, 4)
        source = x[:, None, :]
        probe.kv.scatter(cache, source, meta.slot_mapping, 0)
        indices = meta.slot_mapping.to(torch.int64)
        cache.index_put_((indices[:, 0], indices[:, 1]), source)
        probe.kv.scatter(cache, source, meta.slot_mapping, 1)
        for stage in attention.ATTENTION_STAGES:
            if stage != "kv_window":
                probe.write(stage, x)
        probe.window(cache, meta, 6, 4)
        output.copy_(x)

    wrapper = NS(
        prefix="selected",
        compress_ratio=1,
        swa_cache_layer=NS(kv_cache=cache),
        dsa_attn=NS(layer_name="selected.attn", impl=NS(forward=leaf)),
    )
    context = NS(no_compile_layers={"selected": wrapper}, attn_metadata={"selected.swa_cache": meta})
    op_path = "vllm_ascend/ops/dsa.py"
    unpack = function(
        op_path,
        "_build_kv_cache",
        {"get_ascend_device_type": lambda: None, "AscendDeviceType": NS(A5="A5"), "unfold_kvcache": lambda x: x},
    )
    body = function(
        op_path,
        "dsa_forward",
        {
            "get_forward_context": lambda: context,
            "filter_metadata": function(op_path, "filter_metadata", {}),
            "_build_kv_cache": unpack,
        },
    )
    # Same mutation schema as vllm::dsa_forward; an isolated name avoids
    # replacing installed CPU/PrivateUse1 implementations in other tests.
    lib = torch.library.Library("dspark_receipt_test", "FRAGMENT")
    lib.define("dsa_forward(Tensor hidden_states, bool need_gather_q_kv, Tensor(a!) output, str layer_name) -> ()")
    lib.impl("dsa_forward", body, "CPU" if device == "cpu" else "PrivateUse1")
    torch.library.register_fake("dspark_receipt_test::dsa_forward", function(op_path, "dsa_forward_fake", {}), lib=lib)

    def model(x):
        for name in bank.outer_names[:-1]:
            bank.write(name, x)
        output = torch.empty_like(x)
        torch.ops.dspark_receipt_test.dsa_forward(x, False, output, "selected")
        bank.write(bank.outer_names[-1], output)
        return output

    compile_fx = function(
        "vllm_ascend/compilation/compiler_interface.py",
        "compile_fx",
        dict(
            functools=functools,
            graph_returns_tuple=graph_returns_tuple,
            make_graph_return_tuple=make_graph_return_tuple,
            aot_autograd=aot_autograd,
        ),
    )
    compile_fx.__globals__["compile_fx"] = compile_fx
    graphs = []

    def inner(gm, inputs):
        graphs.append(gm.code)
        return make_boxed_func(gm.forward)

    if device == "npu":
        # Actual archived backend, including its own AOT and compiler passes.
        import npugraph_ex.npu_fx_compiler as nfx

        monkeypatch.setattr(nfx._NpuFxCompiler, "_get_compiled_gm", nfx._NpuFxCompiler._get_compiled_gm)
        config = NS(enable_static_kernel=False, enable_npugraph_ex=True)
        namespace = dict(
            torch=torch,
            os=os,
            logger=NS(info=lambda *a: None),
            graph_returns_tuple=graph_returns_tuple,
            make_graph_return_tuple=make_graph_return_tuple,
        )
        namespace["_configure_backend"] = function(
            "vllm_ascend/compilation/compiler_interface.py", "_configure_backend", namespace
        )
        npu_compile = function("vllm_ascend/compilation/compiler_interface.py", "npugraph_ex_compile", namespace)

    def backend(gm, inputs):
        if device == "npu":
            compiled, _ = npu_compile(
                gm, inputs, {}, NS(), config, None, key="functionalized-graph.txt", cache_dir=str(tmp_path)
            )
            graphs.append((tmp_path / "functionalized-graph.txt").read_text())
            return compiled
        return compile_fx(gm, inputs, inner, {})

    torch._dynamo.reset()
    run = torch.compile(model, backend=backend, fullgraph=True)
    x = torch.ones(6, 4, device=device)
    try:
        run(x)
        if device == "npu":
            torch.npu.synchronize()  # test harness only, never production repair
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run(x)
            replay = graph.replay
        else:
            replay = lambda: run(x)
        count = len(calls)
        pointers = [
            t.data_ptr()
            for t in (
                bank.flags,
                bank.attention_flags,
                bank.receipts,
                bank.attention_receipts,
                bank.epoch_input,
                *probe.kv.tensors,
            )
        ]
        evidence = []
        for epoch in (11, 12, 13):
            bank.receipts.fill_(-1)
            bank.attention_receipts.fill_(-1)
            probe.kv.receipts.fill_(-1)
            bank.epoch_input.fill_(epoch)
            meta.seq_lens.add_(6)
            if epoch == 13:
                x[0].fill_(torch.nan)
            if device == "cpu":
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
                ) as prof:
                    replay()
                copy_shapes = [event.input_shapes[0] for event in prof.events() if event.name == "aten::copy_"]
                # AOT's runtime wrapper copies returned mutations back to the
                # entire input bank, beyond the rows touched by the outer model.
                assert list(bank.receipts.shape) in copy_shapes
                assert list(bank.flags.shape) in copy_shapes
            else:
                replay()
                copy_shapes = None
            evidence.append(
                {
                    "execution": epoch,
                    "shared": shared,
                    "whole_bank_copy_shapes": copy_shapes,
                    "attention_receipts": bank.attention_receipts.cpu().flatten().tolist(),
                }
            )
            assert probe.kv.receipts.cpu().flatten().tolist() == [epoch] * 4
            assert bank.receipts[:9].cpu().flatten().tolist() == [epoch] * 9
            assert bank.attention_receipts.cpu().flatten().tolist() == ([-1] * 12 if shared else [epoch] * 12)
            assert probe.state[0, 1].cpu().item() == 12 + 6 * (epoch - 10) - 6
            if not shared:
                assert bool(bank.attention_flags[0, 0, 0].cpu()) == (epoch == 13)
            assert pointers == [
                t.data_ptr()
                for t in (
                    bank.flags,
                    bank.attention_flags,
                    bank.receipts,
                    bank.attention_receipts,
                    bank.epoch_input,
                    *probe.kv.tensors,
                )
            ]
        if device == "npu":
            assert len(calls) == count
        assert len(graphs) == 1
        (tmp_path / "copyback-evidence.json").write_text(json.dumps(evidence, indent=2))
        (tmp_path / "functionalized-graph.txt").write_text(graphs[0])
    finally:
        torch._dynamo.reset()
        lib._destroy()


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("writer", ["flags", "snapshots"])
def test_compiled_row_bound_static_and_symbolic(monkeypatch, writer, dynamic):
    """Compile actual writers through AOT, below/at/above capacity, no shape coercion."""
    module = load_target(monkeypatch)
    if writer == "flags":
        bank = module.TargetBoundaryFlags(
            sizes=(6, 12, 18, 24),
            auxiliary_layers=(40, 41, 42),
            start_layer=0,
            end_layer=43,
            hidden_size=8,
            hc_mult=4,
            device="cpu",
            target_layer=1,
        )
    else:
        bank = replay_bank(layers=[])
    graphs = []

    def model(x):
        bank.write("embedding", x)
        return x * 2

    def backend(gm, examples):
        graphs.append(gm)
        return aot_autograd(fw_compiler=lambda g, _: make_boxed_func(g.forward))(gm, examples)

    torch._dynamo.reset()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=dynamic)
    try:
        for execution, size in enumerate((29, 5, 24, 30), 1):
            n = min(size, bank.max_tokens)
            x = torch.ones(size, 8)
            x[0] = torch.nan
            x[n - 1] = torch.inf
            bank.epoch_input.fill_(execution)
            if writer == "flags":
                bank.flags.fill_(True)
                bank.receipts.fill_(-1)
            else:
                bank.buffers["embedding"].fill_(-123)
                bank.receipts["embedding"].fill_(-1)
            output = compiled(x)
            torch.testing.assert_close(output, x * 2, equal_nan=True)
            if writer == "flags":
                expected = torch.stack((torch.isnan(x[:n]).any(1), torch.isinf(x[:n]).any(1)), 1)
                assert torch.equal(bank.flags[0, :n], expected)
                assert bank.flags[0, n:].all()
                assert bank.receipts[0].item() == execution
            else:
                bucket = (n - 1) // bank.query_len + 1
                offset = bucket * (bucket - 1) * bank.query_len // 2
                expected = torch.full_like(bank.buffers["embedding"], -123)
                expected[offset : offset + n] = x[:n]
                torch.testing.assert_close(bank.buffers["embedding"], expected, equal_nan=True)
                receipts = [-1] * len(bank.sizes)
                receipts[bucket - 1] = execution
                assert bank.receipts["embedding"].tolist() == receipts
        if dynamic:
            assert any(node.target is torch.sym_min for graph in graphs for node in graph.graph.nodes)
        else:
            assert not any(node.target is torch.sym_min for graph in graphs for node in graph.graph.nodes)
    finally:
        torch._dynamo.reset()
