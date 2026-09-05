# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Capture ABI tests against the actual core signature, without NPU execution.

The source variant runs with --noconftest without Torch. It executes the plugin
classes from AST and reads the parent signature from installed core source (or
the sibling vllm-hust checkout). The runtime variant imports the real classes
under the ordinary CPU UT mocks and is skipped only when Torch is unavailable.
"""

import ast
import copy
import importlib.util
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parents[3]
PLUGIN_SOURCE = REPO_ROOT / "vllm_ascend/worker/v2/aclgraph_utils.py"


def _core_source() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
    else:
        root = REPO_ROOT.parent / "vllm-hust/vllm"
    source = root / "v1/worker/gpu/cudagraph_utils.py"
    assert source.is_file(), "Tests require the real installed core source or sibling vllm-hust checkout"
    return source


def _source_classes():
    core_tree = ast.parse(_core_source().read_text())
    core_class = next(n for n in core_tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelCudaGraphManager")
    core_capture = copy.deepcopy(
        next(n for n in core_class.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
    )
    # Keep the actual parent signature, including its defaults and annotation.
    # Only the body is replaced; no permissive *args/**kwargs mock is used.
    core_capture.body = ast.parse("raise AssertionError('parent spy not installed')").body
    core_capture.decorator_list = []
    namespace = {"nn": SimpleNamespace(Module=object)}
    future = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=[*future, core_capture], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(_core_source()), "exec"), namespace)
    parent = type("ModelCudaGraphManager", (), {"capture": namespace["capture"]})
    namespace["ModelCudaGraphManager"] = parent
    plugin_tree = ast.parse(PLUGIN_SOURCE.read_text())
    classes = [n for n in plugin_tree.body if isinstance(n, ast.ClassDef)]
    exec(compile(ast.Module(body=[*future, *classes], type_ignores=[]), str(PLUGIN_SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace), parent


def _parameters(function):
    return [(p.name, p.kind, p.default) for p in signature(function).parameters.values()]


@pytest.fixture(params=["source", "runtime"])
def capture_api(request, monkeypatch):
    if request.param == "source":
        module, parent = _source_classes()
    else:
        if importlib.util.find_spec("torch") is None:
            pytest.skip("Torch is unavailable; real-class CPU/mock capture tests require the server environment")
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

        from vllm_ascend.worker.v2 import aclgraph_utils

        module, parent = aclgraph_utils, ModelCudaGraphManager

    state = SimpleNamespace(calls=[], events=[], result={object(): object()}, error=None)
    state.parent_capture = parent.capture
    state.plugin_capture = module.ModelAclGraphManager.capture
    state.wrapper_type = module.ModelWithContext

    @contextmanager
    def communicator_switch():
        state.events.append("enter")
        try:
            yield
        finally:
            state.events.append("exit")

    def parent_capture(
        self,
        model,
        model_state,
        input_buffers,
        intermediate_tensors,
        block_tables,
        attn_groups,
        kv_cache_config,
        has_lora=False,
        use_aux_hidden_state_outputs=False,
        lora_capture_hook=None,
        progress_bar_desc="Capturing CUDA graphs",
    ):
        state.calls.append(locals().copy())
        assert state.events == ["enter"]
        state.events.append("parent")
        if state.error is not None:
            raise state.error
        return state.result

    # Fail on core ABI drift before replacing the method, including new options,
    # positional/keyword kinds and defaults. The spy itself is a strict function.
    assert _parameters(parent_capture) == _parameters(parent.capture)
    monkeypatch.setattr(parent, "capture", parent_capture)
    monkeypatch.setattr(module, "communicator_switch", communicator_switch, raising=False)
    if request.param == "source":
        # exec-created functions use their original globals, not the namespace view.
        module.ModelAclGraphManager.capture.__globals__["communicator_switch"] = communicator_switch
    state.manager = object.__new__(module.ModelAclGraphManager)
    state.inputs = [object() for _ in range(7)]
    return state


def test_capture_signature_and_return_annotation_match_real_core(capture_api):
    assert signature(capture_api.plugin_capture) == signature(capture_api.parent_capture)


def test_capture_defaults_preserve_wrapper_lifecycle_and_attention_states(capture_api):
    state = capture_api
    result = state.manager.capture(*state.inputs)
    assert result is state.result
    assert state.events == ["enter", "parent", "exit"]
    assert len(state.calls) == 1
    call = state.calls[0]
    assert isinstance(call["model"], state.wrapper_type)
    assert call["model"].get_original_model() is state.inputs[0]
    assert call["has_lora"] is False
    assert call["use_aux_hidden_state_outputs"] is False
    assert call["lora_capture_hook"] is None
    assert call["progress_bar_desc"] == "Capturing CUDA graphs"


@pytest.mark.parametrize("has_lora", [False, True])
@pytest.mark.parametrize("use_aux", [False, True])
@pytest.mark.parametrize("with_hook", [False, True])
def test_capture_forwards_options_without_positional_shift(capture_api, has_lora, use_aux, with_hook):
    state = capture_api
    hook_calls = []

    def hook(num_loras, num_reqs, num_tokens):
        hook_calls.append((num_loras, num_reqs, num_tokens))

    expected_hook = hook if with_hook else None
    result = state.manager.capture(
        *state.inputs,
        has_lora=has_lora,
        use_aux_hidden_state_outputs=use_aux,
        lora_capture_hook=expected_hook,
        progress_bar_desc="P08-R2 target capture",
    )
    assert result is state.result
    assert len(state.calls) == 1
    call = state.calls[0]
    required_names = list(signature(state.parent_capture).parameters)[2:8]
    assert all(call[name] is value for name, value in zip(required_names, state.inputs[1:]))
    assert call["model"].get_original_model() is state.inputs[0]
    assert call["has_lora"] is has_lora
    assert call["use_aux_hidden_state_outputs"] is use_aux
    assert call["lora_capture_hook"] is expected_hook
    assert call["progress_bar_desc"] == "P08-R2 target capture"
    assert hook_calls == []  # Only the parent lifecycle should invoke the hook.
    if with_hook:
        call["lora_capture_hook"](2, 1, 6)
        assert hook_calls == [(2, 1, 6)]
    assert state.events == ["enter", "parent", "exit"]


def test_parent_capture_failure_restores_communicator_and_propagates(capture_api):
    state = capture_api
    state.error = RuntimeError("capture failed")
    with pytest.raises(RuntimeError, match="capture failed") as caught:
        state.manager.capture(*state.inputs, lora_capture_hook=None)
    assert caught.value is state.error
    assert state.events == ["enter", "parent", "exit"]


def test_capture_rejects_unknown_options(capture_api):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        capture_api.manager.capture(*capture_api.inputs, unknown_capture_option=True)
    assert capture_api.calls == []
    assert capture_api.events == []


def test_optional_parent_arguments_are_forwarded_by_keyword():
    tree = ast.parse(PLUGIN_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelAclGraphManager")
    capture = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "capture")
    call = next(
        n
        for n in ast.walk(capture)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "capture"
    )
    assert len(call.args) == 7
    assert {kw.arg for kw in call.keywords} == {
        "has_lora",
        "use_aux_hidden_state_outputs",
        "lora_capture_hook",
        "progress_bar_desc",
    }
    assert capture.args.kwarg is None
