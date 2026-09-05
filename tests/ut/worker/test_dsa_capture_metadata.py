# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Exercise MRV2 capture metadata through real function bodies, without NPU ops.

The source variant runs with --noconftest and NumPy on hosts without Torch.
The runtime variant imports the actual classes under the usual CPU UT setup.
Both execute prepare_inputs_to_capture -> prepare_attn -> build_attn_metadata
-> DSA build_for_cudagraph_capture -> DSA build, including its real assertions.
Only tensor/runtime dependencies and the leaf SAS/rope operations are mocked.
"""

import ast
import importlib.util
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
DSA_SOURCE = ROOT / "vllm_ascend/attention/dsa_v1.py"
ATTN_SOURCE = ROOT / "vllm_ascend/worker/v2/attn_utils.py"
STATE_SOURCE = ROOT / "vllm_ascend/worker/v2/model_states/default.py"
CONTEXT_KEYS = (
    "num_reqs_actual",
    "prefill_ratio_to_sas_metadata",
    "decode_ratio_to_sas_metadata",
    "common_ratio_to_sas_metadata",
    "block_size",
)


class _Tensor(np.ndarray):
    """Only the CPU tensor operations used by the unchanged build body."""

    def long(self):
        return self.astype(np.int64)

    def int(self):
        return self.astype(np.int32)

    def argmax(self, dim=None):
        return super().argmax(axis=dim)


def _tensor(value):
    return np.asarray(value).view(_Tensor)


class _Mode(Enum):
    NONE = 0
    FULL = 1
    PIECEWISE = 2


def _load_functions(path, namespace, names, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *selected], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def _core_graph_source():
    spec = importlib.util.find_spec("vllm")
    root = Path(next(iter(spec.submodule_search_locations))) if spec else ROOT.parent / "vllm-hust/vllm"
    source = root / "v1/worker/gpu/cudagraph_utils.py"
    assert source.is_file(), "Tests need the real installed core source or sibling vllm-hust"
    return source


@pytest.fixture(params=["source", "runtime"])
def api(request, monkeypatch):
    if request.param == "source":
        mode = _Mode
        namespace = {
            "np": np,
            "torch": SimpleNamespace(from_numpy=_tensor, any=np.any, all=np.all),
            "CUDAGraphMode": mode,
            "AscendCommonAttentionMetadata": SimpleNamespace,
            "BUILD_METADATA_STEP_PREFILL": 0,
            "is_pd_decode_recompute_scheduler_enabled": lambda: False,
        }
        _load_functions(ROOT / "vllm_ascend/attention/utils.py", namespace, ["split_decodes_and_prefills"])
        names = ["build", "build_for_cudagraph_capture", "set_num_actual_tokens", "get_block_table_size"]
        _load_functions(DSA_SOURCE, namespace, names, "AscendDSAMetadataBuilder")
        builder = type("AscendDSAMetadataBuilder", (), {name: namespace[name] for name in names})
        builder.hadamard = None
        namespace["AscendDSAMetadataBuilder"] = builder
        _load_functions(ATTN_SOURCE, namespace, ["build_attn_metadata"])
        _load_functions(STATE_SOURCE, namespace, ["prepare_attn"], "AscendModelState")
        state_cls = type("AscendModelState", (), {"prepare_attn": namespace["prepare_attn"]})
        _load_functions(_core_graph_source(), namespace, ["prepare_inputs_to_capture"])
        result = SimpleNamespace(
            builder=builder,
            state_cls=state_cls,
            tensor=_tensor,
            mode=mode,
            build_attn=namespace["build_attn_metadata"],
            prepare=namespace["prepare_inputs_to_capture"],
            dsa_globals=namespace,
            graph_globals=namespace,
        )
    else:
        if importlib.util.find_spec("torch") is None:
            pytest.skip("Torch unavailable; real-class CPU/mock metadata tests require the server environment")
        import torch
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu import cudagraph_utils

        from vllm_ascend.attention import dsa_v1
        from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
        from vllm_ascend.worker.v2.model_states.default import AscendModelState

        result = SimpleNamespace(
            builder=dsa_v1.AscendDSAMetadataBuilder,
            state_cls=AscendModelState,
            tensor=torch.as_tensor,
            mode=CUDAGraphMode,
            build_attn=build_attn_metadata,
            prepare=cudagraph_utils.prepare_inputs_to_capture,
            dsa_globals=vars(dsa_v1),
            graph_globals=vars(cudagraph_utils),
        )
    result.rope_calls = []

    def rope(positions, decode=False):
        result.rope_calls.append((positions, decode))
        return positions, positions

    def format_slots(slots, block_size):
        assert block_size in (32, 64, 256)
        return result.tensor(np.stack((np.asarray(slots) // block_size, np.asarray(slots) % block_size), axis=-1))

    monkeypatch.setitem(result.dsa_globals, "get_cos_and_sin_dsa", rope)
    monkeypatch.setitem(result.dsa_globals, "DeviceOperator", SimpleNamespace(format_dsa_slot_mapping=format_slots))
    monkeypatch.setitem(result.graph_globals, "AttentionState", lambda metadata, slots: (metadata, slots))
    monkeypatch.setitem(result.graph_globals, "build_slot_mappings_by_layer", lambda slots, config: slots)
    return result


def _setup(api, monkeypatch, batch=1, *, replace_first=False):
    """Reuse builders across preparations to expose stale per-builder caches."""
    calls = []
    builders = []

    class Builder(api.builder):
        def __init__(self, ratio):
            self.ratio = ratio
            self.decode_threshold = 6  # K=5 target verification.
            self.model_config = SimpleNamespace(get_head_size=lambda: 128)
            self.metadata_cls = SimpleNamespace
            self.slot_mapping = api.tensor(np.zeros((batch * 6, 2), dtype=np.int32))
            self.fail = False
            self.capture_calls = 0

        def build_for_cudagraph_capture(self, common_attn_metadata, **context):
            self.capture_calls += 1
            return super().build_for_cudagraph_capture(common_attn_metadata, **context)

        def _leaf(self, common_prefix_len, common_attn_metadata, num_reqs_actual, kind):
            assert common_prefix_len == 0
            assert num_reqs_actual == common_attn_metadata.num_reqs == batch
            assert common_attn_metadata.max_query_len == 6
            assert self.block_size in (32, 64, 256)
            assert self.num_actual_tokens == batch * 6
            if replace_first and self.ratio == 1:
                for key in CONTEXT_KEYS[1:4]:
                    setattr(self, key, dict(getattr(self, key)))
            caches = [getattr(self, key) for key in CONTEXT_KEYS[1:4]]
            assert all(isinstance(cache, dict) for cache in caches)
            caches[kind].setdefault(self.ratio, object())
            leaf = SimpleNamespace(
                caches=caches,
                block_size=self.block_size,
                num_reqs_actual=num_reqs_actual,
                common_attn_metadata=common_attn_metadata,
            )
            calls.append(leaf)
            if self.fail:
                raise RuntimeError("SAS metadata preparation failed")
            return leaf

        def build_prefill_metadata(self, common_prefix_len, common_attn_metadata, num_reqs_actual):
            return self._leaf(common_prefix_len, common_attn_metadata, num_reqs_actual, 0)

        def build_decode_metadata(self, common_prefix_len, common_attn_metadata, num_reqs_actual):
            return self._leaf(common_prefix_len, common_attn_metadata, num_reqs_actual, 1)

    groups = []
    for index, block_size in enumerate((32, 64, 256)):
        builder = Builder((1, 4, 128)[index])
        builders.append(builder)
        # Deliberately distinct from the outer KV group's block_size.
        group = SimpleNamespace(
            layer_names=[f"layer{index}", f"layer{index}.shared"],
            kv_cache_spec=SimpleNamespace(block_size=block_size),
            get_metadata_builder=lambda index, builder=builder: builder,
        )
        groups.append([group])
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(block_size=128) for _ in groups])
    block_tables = tuple(api.tensor(np.full((batch, 2), i + 10, dtype=np.int32)) for i in range(3))
    slots = [api.tensor(np.arange(batch * 6, dtype=np.int32) + i * 100) for i in range(3)]
    batch_input = SimpleNamespace(
        num_reqs=batch,
        num_reqs_after_padding=batch,
        num_tokens=batch * 6,
        num_tokens_after_padding=batch * 6,
        query_start_loc_np=np.arange(batch + 1, dtype=np.int32) * 6,
        query_start_loc=api.tensor(np.arange(batch + 1, dtype=np.int32) * 6),
        num_scheduled_tokens=np.full(batch, 6, dtype=np.int32),
        seq_lens=api.tensor(np.full(batch, 6, dtype=np.int32)),
        seq_lens_np=np.full(batch, 6, dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=api.tensor(np.zeros(batch * 6, dtype=np.int64)),
        is_prefilling_np=np.zeros(batch, dtype=np.bool_),
        attn_state=object(),
    )

    def make_dummy(num_reqs, num_tokens, buffers):
        assert (num_reqs, num_tokens) == (batch, batch * 6)
        assert buffers is batch_input
        return buffers

    monkeypatch.setitem(api.graph_globals, "InputBatch", SimpleNamespace(make_dummy=make_dummy))
    tables = SimpleNamespace(
        cp_size=1,
        get_dummy_block_tables=lambda count: block_tables,
        get_dummy_slot_mappings=lambda count: slots,
    )
    state = object.__new__(api.state_cls)
    state.max_model_len = 8192
    return SimpleNamespace(
        calls=calls,
        builders=builders,
        groups=groups,
        config=config,
        tables=tables,
        block_tables=block_tables,
        slots=slots,
        batch_input=batch_input,
        state=state,
        batch=batch,
    )


def _prepare(api, run, capture=True):
    if capture:
        metadata, slots = api.prepare(
            run.batch,
            run.batch * 6,
            run.state,
            run.batch_input,
            run.tables,
            run.groups,
            run.config,
        )
        assert slots is run.slots
        return metadata
    return run.state.prepare_attn(
        run.batch_input,
        api.mode.NONE,
        run.block_tables,
        run.slots,
        run.groups,
        run.config,
        for_capture=False,
    )


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("capture", [False, True])
def test_full_context_group_identity_and_real_block_sizes(api, monkeypatch, batch, capture):
    run = _setup(api, monkeypatch, batch)
    metadata = _prepare(api, run, capture)
    assert run.state.attn_metadata is metadata
    assert [call.block_size for call in run.calls] == [32, 64, 256]
    assert len(api.rope_calls) == 1  # Common metadata actually reused by DSA build.
    for index, call in enumerate(run.calls):
        common = call.common_attn_metadata
        assert common.block_table_tensor is run.block_tables[index]
        assert common.slot_mapping is run.slots[index]
        assert common.positions is run.batch_input.positions
        assert common.attn_state is run.batch_input.attn_state
        assert common.query_start_loc_cpu.tolist() == list(range(0, batch * 6 + 1, 6))
        assert all(left is right for left, right in zip(call.caches, run.calls[0].caches))
        assert metadata[f"layer{index}"] is metadata[f"layer{index}.shared"]
        assert metadata[f"layer{index}"].num_decode_tokens == batch * 6
        assert run.builders[index].capture_calls == int(capture)
        expected_slots = np.stack(
            (np.asarray(run.slots[index]) // call.block_size, np.asarray(run.slots[index]) % call.block_size), axis=-1
        )
        np.testing.assert_array_equal(run.builders[index].slot_mapping, expected_slots)
    assert run.calls[0].caches[0] == {}
    assert set(run.calls[0].caches[1]) == {1, 4, 128}


@pytest.mark.parametrize("capture", [False, True])
def test_replaced_dictionaries_are_forwarded_to_later_groups(api, monkeypatch, capture):
    run = _setup(api, monkeypatch, replace_first=True)
    _prepare(api, run, capture)
    for call in run.calls[1:]:
        assert all(left is right for left, right in zip(call.caches, run.calls[0].caches))


@pytest.mark.parametrize("capture", [False, True])
def test_failed_preparation_does_not_publish_or_reuse_partial_state(api, monkeypatch, capture):
    run = _setup(api, monkeypatch)
    previous = object()
    run.state.attn_metadata = previous
    run.builders[1].fail = True
    with pytest.raises(RuntimeError, match="SAS metadata preparation failed"):
        _prepare(api, run, capture)
    assert run.state.attn_metadata is previous
    failed_caches = run.calls[0].caches
    run.builders[1].fail = False
    run.calls.clear()
    _prepare(api, run, capture)
    assert all(left is not right for left, right in zip(failed_caches, run.calls[0].caches))
    assert len(api.rope_calls) == 2


def test_eager_prefill_state_is_not_reused_by_capture(api, monkeypatch):
    run = _setup(api, monkeypatch)
    run.batch_input.is_prefilling_np[:] = True
    eager = _prepare(api, run, capture=False)
    eager_caches = run.calls[0].caches
    assert eager["layer0"].num_prefills == 1
    assert eager_caches[0] and eager_caches[1] == {}
    run.batch_input.is_prefilling_np[:] = False
    run.calls.clear()
    captured = _prepare(api, run)
    assert captured["layer0"].num_decodes == 1
    assert captured["layer0"].num_prefills == 0
    assert run.calls[0].caches[0] == {} and run.calls[0].caches[1]
    assert eager["layer0"].num_prefills == 1
    assert all(old is not new for old, new in zip(eager_caches, run.calls[0].caches))


@pytest.mark.parametrize("missing", CONTEXT_KEYS)
def test_capture_requires_every_context_argument(api, missing):
    builder = object.__new__(api.builder)
    context = dict(zip(CONTEXT_KEYS, (1, {}, {}, {}, 32)))
    del context[missing]
    with pytest.raises(TypeError, match=missing):
        builder.build_for_cudagraph_capture(SimpleNamespace(), **context)


def test_build_does_not_default_missing_block_size_to_128(api):
    builder = object.__new__(api.builder)
    builder.block_size = 64  # A previous build must not supply the missing value.
    common = SimpleNamespace(num_reqs=1, query_start_loc=api.tensor([0, 6]))
    with pytest.raises(KeyError, match="block_size"):
        builder.build(
            0,
            common,
            num_reqs_actual=1,
            prefill_ratio_to_sas_metadata={},
            decode_ratio_to_sas_metadata={},
            common_ratio_to_sas_metadata={},
        )


@pytest.mark.parametrize("missing", CONTEXT_KEYS[1:4])
def test_capture_retains_real_build_assertions_for_missing_metadata(api, missing):
    builder = object.__new__(api.builder)
    common = SimpleNamespace(num_reqs=1, query_start_loc=api.tensor([0, 6]))
    context = dict(zip(CONTEXT_KEYS, (1, {}, {}, {}, 32)))
    context[missing] = None
    with pytest.raises(AssertionError):
        builder.build_for_cudagraph_capture(common, **context)


@dataclass(frozen=True)
class _Descriptor:
    cg_mode: object
    num_reqs: int
    num_tokens: int


def test_core_full_capture_uses_fresh_warmup_and_capture_contexts(api, monkeypatch):
    run = _setup(api, monkeypatch)
    namespace = {
        "CUDAGraphMode": api.mode,
        "graph_capture": lambda device: nullcontext(),
        "is_global_first_rank": lambda: False,
        "logger": SimpleNamespace(debug=lambda *args: None),
        "get_offloader": lambda: SimpleNamespace(sync_prev_onload=lambda: None, join_after_forward=lambda: None),
        "torch": SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=object, graph=lambda graph, pool: nullcontext())),
        "AttentionStatePair": lambda warmup, captured: SimpleNamespace(warmup=warmup, captured=captured),
        "compilation_counter": SimpleNamespace(num_cudagraph_captured=0),
    }
    # Execute the frozen core lifecycle itself; only hardware capture is mocked.
    _load_functions(_core_graph_source(), namespace, ["capture"], "CudaGraphManager")
    rounds = []
    for _ in range(2):
        desc = _Descriptor(api.mode.FULL, 1, 6)
        manager = SimpleNamespace(device="cpu", _capture_descs={api.mode.FULL: [desc]}, graphs={}, pool=None)
        phases = []

        def create_forward_fn(descriptor, warmup, desc=desc, phases=phases):
            assert descriptor is desc
            metadata = _prepare(api, run)
            phases.append(warmup)
            rounds.append(run.calls[-3].caches)

            def forward(mode):
                assert mode == api.mode.NONE
                assert metadata["layer0"].num_decode_tokens == 6

            return forward, metadata

        states = namespace["capture"](manager, create_forward_fn)
        assert phases == [True, False]
        assert states[desc].warmup is not states[desc].captured
        assert manager._graphs_captured and desc in manager.graphs
    assert len(api.rope_calls) == 4
    for cache_index in range(3):
        assert len({id(context[cache_index]) for context in rounds}) == 4


@pytest.mark.parametrize("capture", [False, True])
def test_non_dsa_keeps_capture_entry_and_eager_model_specific_options(api, monkeypatch, capture):
    run = _setup(api, monkeypatch)
    calls = []
    marker = object()

    class GenericBuilder:
        def build_for_cudagraph_capture(self, common_attn_metadata):
            calls.append("capture")
            assert common_attn_metadata.max_query_len == 6
            return marker

        def build(self, common_prefix_len, common_attn_metadata, *, model_marker):
            calls.append("eager")
            assert common_prefix_len == 0 and common_attn_metadata.max_query_len == 6
            assert model_marker is marker
            return marker

    class ModelOptions:
        def get_extra_common_attn_kwargs(self, group_id, num_reqs):
            assert group_id in (0, 1, 2) and num_reqs == 1
            return {}

        def get_extra_attn_kwargs(self, builder, num_reqs):
            assert not capture  # Capture did not call this callback before R3.
            assert isinstance(builder, GenericBuilder) and num_reqs == 1
            return {"model_marker": marker}

    for group in run.groups:
        group[0].get_metadata_builder = lambda index: GenericBuilder()
    metadata = api.build_attn(
        attn_groups=run.groups,
        num_reqs=1,
        num_tokens=6,
        query_start_loc_gpu=run.batch_input.query_start_loc,
        query_start_loc_cpu=api.tensor(run.batch_input.query_start_loc_np),
        max_query_len=6,
        seq_lens=run.batch_input.seq_lens,
        max_seq_len=6,
        block_tables=run.block_tables,
        slot_mappings=run.slots,
        kv_cache_config=run.config,
        positions=run.batch_input.positions,
        for_cudagraph_capture=capture,
        model_specific_attn_metadata=ModelOptions(),
    )
    assert calls == ["capture" if capture else "eager"] * 3
    assert all(value is marker for value in metadata.values())
