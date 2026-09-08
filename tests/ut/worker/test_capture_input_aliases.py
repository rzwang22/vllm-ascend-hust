# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""CPU execution of real core/plugin batch constructors and the DSA consumer.

Only imports/rotary setup and NPU leaf operators are substituted. These tests
prove captured reference updates, not the numerical behavior of the NPU kernel.
"""

import ast
import dataclasses
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from uuid import uuid4

import numpy as np
import pytest

from tests.ut.worker.test_aclgraph_capture import _core_source

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[3]


def _classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *body], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture
def batches(monkeypatch):
    module = ModuleType("p08_capture_input_aliases")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    namespace = vars(module)
    namespace.update(
        torch=torch,
        np=np,
        dataclass=dataclasses.dataclass,
        asdict=dataclasses.asdict,
        fields=dataclasses.fields,
        random_uuid=lambda: uuid4().hex,
    )
    _classes(_core_source().with_name("input_batch.py"), {"InputBatch", "InputBuffers"}, namespace)
    parent = module.InputBatch
    namespace.update(AscendAttentionState=NS(DecodeOnly=object()), update_cos_sin=lambda positions: None)
    _classes(ROOT / "vllm_ascend/worker/v2/input_batch.py", {"AscendInputBatch", "AscendInputBuffers"}, namespace)
    return NS(parent=parent, cls=module.AscendInputBatch, buffers=module.AscendInputBuffers(4, 24, torch.device("cpu")))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_dummy_capture_retains_live_input_storage(batches, batch_size):
    buffers = batches.buffers
    warmup = batches.cls.make_dummy(batch_size, batch_size * 6, buffers)
    captured = batches.cls.make_dummy(batch_size, batch_size * 6, buffers)
    assert warmup is not captured
    for name in ("input_ids", "positions", "query_start_loc", "seq_lens", "is_padding"):
        live = getattr(buffers, name)
        for batch in (warmup, captured):
            held = getattr(batch, name)
            assert held.untyped_storage().data_ptr() == live.untyped_storage().data_ptr(), name
            assert held.storage_offset() == 0
            assert held.stride() == live.stride()
    assert np.shares_memory(captured.seq_lens_np, buffers.seq_lens_np)
    # Fields already aliased by the core constructor must retain that identity.
    assert captured.idx_mapping is captured.expanded_idx_mapping
    assert captured.num_reqs == batch_size and captured.num_tokens == batch_size * 6
    for step in range(2):
        seq = torch.arange(batch_size, dtype=torch.int32) * 13 + 41 + step * 7
        buffers.seq_lens[:batch_size].copy_(seq)
        buffers.positions[: batch_size * 6].copy_(torch.cat([torch.arange(s - 6, s) for s in seq]))
        buffers.input_ids[: batch_size * 6].fill_(step + 17)
        buffers.is_padding[: batch_size * 6].zero_()
        for batch in (warmup, captured):
            torch.testing.assert_close(batch.seq_lens, seq)
            torch.testing.assert_close(batch.positions, buffers.positions[: batch_size * 6])
            assert not batch.is_padding.any()
            assert torch.all(batch.input_ids == step + 17)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_dsa_capture_consumer_observes_real_replay_sequence_updates(batches, batch_size):
    captured = batches.cls.make_dummy(batch_size, batch_size * 6, batches.buffers)
    path = ROOT / "vllm_ascend/attention/dsa_v1.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDSAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_decode")
    calls = []

    def attention(query, **kwargs):
        # Record the exact tensor references passed to the real leaf ABI.
        assert kwargs["layout_q"] == "TND" and kwargs["layout_kv"] == "PA_ND"
        calls.append(kwargs)
        return (torch.zeros_like(query),)

    device = NS(
        unpack_dsa_forward_kv_cache=lambda cache, ratio: (None, cache, None, None, None, None),
        get_dsa_sparse_attn_op=lambda: attention,
        get_dsa_sparse_attn_base_kwargs=lambda: {},
    )
    namespace = {"DeviceOperator": device, "torch": torch, "_require_decode_metadata": lambda metadata: metadata.decode}
    exec(
        compile(
            ast.Module(body=[*ast.parse("from __future__ import annotations").body, method], type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )
    metadata = NS(
        decode=NS(
            cos={"layer": None},
            sin={"layer": None},
            query_start_loc=captured.query_start_loc,
            seq_lens=captured.seq_lens,
            slot_mapping=None,
            block_table=torch.zeros(batch_size, 8, dtype=torch.int32),
            sas_metadata=torch.zeros(1024, dtype=torch.int32),
        )
    )
    impl = NS(
        compress_ratio=1,
        validate_dspark_sharedkv_contract=False,
        multistream_dsv4_dsa_overlap=True,
        _mla_prolog_multistream=lambda *args, **kwargs: (torch.ones(batch_size * 6, 1, 8), None, None),
        attn_sink=torch.zeros(1),
        softmax_scale=0.5,
        window_size=128,
    )
    namespace["_forward_decode"](impl, "layer", torch.ones(batch_size * 6, 8), torch.ones(8, 32, 8), [metadata])
    assert len(calls) == 1
    held = calls[0]
    # Replay reuses these leaf references without another Python forward.
    for step in range(2):
        seq = torch.arange(batch_size, dtype=torch.int32) * 11 + 59 + step * 5
        batches.buffers.seq_lens[:batch_size].copy_(seq)
        actual = held["seqused_kv"]
        torch.testing.assert_close(actual, seq)
        query_lengths = held["cu_seqlens_q"].diff()
        torch.testing.assert_close(actual - query_lengths, seq - 6)
    assert len(calls) == 1


def test_core_batch_constructor_remains_unchanged(batches):
    batch = batches.parent.make_dummy(2, 12, batches.buffers)
    assert batch.seq_lens.untyped_storage().data_ptr() == batches.buffers.seq_lens.untyped_storage().data_ptr()
    assert not hasattr(batch, "seq_lens_np")


def test_capture_shapes_share_updates_and_clear_unused_rows(batches):
    captured = {size: batches.cls.make_dummy(size, size * 6, batches.buffers) for size in (1, 2, 4)}
    for size in (4, 2, 1):
        batches.buffers.seq_lens.zero_()
        batches.buffers.seq_lens[:size].copy_(torch.arange(size, dtype=torch.int32) + 71)
        for graph_size, batch in captured.items():
            torch.testing.assert_close(batch.seq_lens, batches.buffers.seq_lens[:graph_size])
