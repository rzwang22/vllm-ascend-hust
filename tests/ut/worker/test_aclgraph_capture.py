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
import sys
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

    state = SimpleNamespace(
        calls=[], events=[], result={object(): object()}, error=None, source=request.param == "source"
    )
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


@pytest.fixture
def diagnostic_capture(capture_api, monkeypatch, tmp_path):
    # Execute the real diagnostics and capture entry, not permissive constructor
    # mocks. The source variant needs CPU Torch but no installed vLLM/NPU stack.
    pytest.importorskip("torch")
    from tests.ut.test_dspark_nan_diagnostics import _NAN
    from tests.ut.test_dspark_replay_diagnostics import bank, replay

    state = capture_api
    if state.source:
        core_path = _core_source().parents[3] / "forward_context.py"
        core_tree = ast.parse(core_path.read_text())
        core_functions = [
            n
            for n in core_tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("get_forward_context", "is_forward_context_available")
        ]
        context = {"_forward_context": None}
        future = ast.parse("from __future__ import annotations").body
        exec(compile(ast.Module(body=[*future, *core_functions], type_ignores=[]), str(core_path), "exec"), context)
        proxy_path = REPO_ROOT / "vllm_ascend/ascend_forward_context.py"
        proxy_class = next(
            n
            for n in ast.parse(proxy_path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "_ExtraForwardContextProxy"
        )
        exec(compile(ast.Module(body=[*future, proxy_class], type_ignores=[]), str(proxy_path), "exec"), context)
        proxy = context["_ExtraForwardContextProxy"]()
        get_context = context["get_forward_context"]
        available = context["is_forward_context_available"]
        monkeypatch.setitem(state.plugin_capture.__globals__, "_EXTRA_CTX", proxy)
    else:
        from vllm import forward_context

        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        # No synthetic ForwardContext: exercise the real proxy's unset guard.
        monkeypatch.setattr(forward_context, "_forward_context", None)
        proxy, get_context = _EXTRA_CTX, forward_context.get_forward_context
        available = forward_context.is_forward_context_available

    def assert_no_context():
        assert not available()
        with pytest.raises(AssertionError, match="Forward context is not set"):
            get_context()
        with pytest.raises(AssertionError, match="Forward context is not set"):
            _ = proxy.flash_comm_v1_enabled

    state.assert_no_context = assert_no_context
    state.bank = bank(layers=[])
    state.inputs[0] = SimpleNamespace(model=SimpleNamespace(_dspark_layer_snapshots=state.bank))
    state.manager.vllm_config = SimpleNamespace(
        additional_config={"dspark_nan_replay_window": [60, 80], "dspark_nan_diagnostic_dir": str(tmp_path / "ranks")},
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    # This is the resolved AscendConfig field already retained by NPUModelRunner,
    # including environment-based configuration absent from additional_config.
    state.manager.model_runner = SimpleNamespace(ascend_config=SimpleNamespace(enable_flashcomm1=False), dp_size=1)
    state.result = {}  # Parent transport is spied; no NPU capture in this test.
    state.finished = []
    finish_capture = replay.ReplaySnapshots.finish_capture

    def finish(self, states):
        assert state.events == ["enter", "parent", "exit"]
        state.finished.append((self, states))
        return finish_capture(self, states)

    monkeypatch.setattr(replay.ReplaySnapshots, "finish_capture", finish)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_nan", _NAN)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_replay", replay)
    state.assert_no_context()
    yield state
    state.assert_no_context()


@pytest.mark.parametrize("with_hook", [False, True])
def test_diagnostic_capture_initializes_without_forward_context(diagnostic_capture, with_hook):
    state = diagnostic_capture
    hook = (lambda num_loras, num_reqs, num_tokens: None) if with_hook else None
    result = state.manager.capture(
        *state.inputs,
        has_lora=True,
        use_aux_hidden_state_outputs=True,
        lora_capture_hook=hook,
        progress_bar_desc="P08-R9B capture",
    )
    assert result is state.result
    call = state.calls[0]
    assert call["model"].get_original_model() is state.inputs[0]
    assert all(
        call[name] is value
        for name, value in zip(list(signature(state.parent_capture).parameters)[2:8], state.inputs[1:])
    )
    assert call["has_lora"] is True and call["use_aux_hidden_state_outputs"] is True
    assert call["lora_capture_hook"] is hook
    assert call["progress_bar_desc"] == "P08-R9B capture"
    snapshots = call["model"].replay_diagnostics
    diagnostic = state.manager._dspark_nan_diagnostic
    assert snapshots.bank is state.bank
    assert snapshots.diagnostic is diagnostic
    assert diagnostic.execution_epoch == 0 and diagnostic.current == {}
    assert diagnostic.phase == "not_started"
    assert diagnostic.replay_configuration["completed_detailed_replays"] == 0
    assert state.finished == [(snapshots, result)]


@pytest.mark.parametrize("flashcomm1,dp", [(True, 1), (False, 2), (True, 2)])
def test_diagnostic_capture_rejects_unsupported_static_config(diagnostic_capture, flashcomm1, dp):
    state = diagnostic_capture
    state.manager.model_runner.ascend_config.enable_flashcomm1 = flashcomm1
    state.manager.vllm_config.parallel_config.data_parallel_size = dp
    state.manager.model_runner.dp_size = dp
    with pytest.raises(ValueError, match="FlashComm1 off and DP1"):
        state.manager.capture(*state.inputs)
    assert state.calls == [] and state.events == [] and state.finished == []
    assert not hasattr(state.manager, "_dspark_nan_diagnostic")


def test_diagnostic_capture_parent_failure_preserves_lifecycle(diagnostic_capture):
    state = diagnostic_capture
    state.error = RuntimeError("parent capture failed")
    with pytest.raises(RuntimeError, match="parent capture failed") as caught:
        state.manager.capture(*state.inputs)
    assert caught.value is state.error
    assert state.events == ["enter", "parent", "exit"]
    assert state.finished == []


def test_diagnostic_disabled_does_not_apply_support_gate(diagnostic_capture):
    state = diagnostic_capture
    state.manager.vllm_config.additional_config = {}
    state.manager.model_runner.ascend_config.enable_flashcomm1 = True
    state.manager.vllm_config.parallel_config.data_parallel_size = 2
    state.manager.model_runner.dp_size = 2
    assert state.manager.capture(*state.inputs) is state.result
    assert state.calls[0]["model"].replay_diagnostics is None
    assert not hasattr(state.manager, "_dspark_nan_diagnostic")
    assert state.finished == []
    assert not hasattr(state.manager, "_dspark_auxiliary_capture")


def test_auxiliary_capture_reuses_output_wrapper_without_layer_banks(capture_api, monkeypatch):
    pytest.importorskip("torch")
    from tests.ut.test_dspark_profile_auxiliary import load_auxiliary

    state = capture_api
    module = load_auxiliary(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_auxiliary", module)
    state.manager.vllm_config = SimpleNamespace(
        additional_config={
            "dspark_profile_observation": {"mode": "auxiliary-transfers"},
            "dspark_confidence_verification": {"mode": "specified_lengths", "profile": True},
        },
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    state.manager.model_runner = SimpleNamespace(
        ascend_config=SimpleNamespace(enable_flashcomm1=False),
        speculator=SimpleNamespace(target_layer_ids=(40, 41, 42)),
    )
    state.result = {}
    state.manager.capture(*state.inputs, use_aux_hidden_state_outputs=True)
    snapshots = state.calls[0]["model"].replay_diagnostics
    assert snapshots is state.manager._dspark_auxiliary_capture
    assert isinstance(snapshots, module.AuxiliaryCapture)
    assert not hasattr(snapshots, "bank") and not snapshots.shapes
    assert not hasattr(state.manager, "_dspark_nan_diagnostic")


@pytest.mark.parametrize("override", ["no_aux", "flashcomm", "dp", "lora", "not_profile", "old_diagnostic"])
def test_auxiliary_capture_rejects_incompatible_scopes(capture_api, monkeypatch, override):
    pytest.importorskip("torch")
    from tests.ut.test_dspark_profile_auxiliary import load_auxiliary

    state = capture_api
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_auxiliary", load_auxiliary(monkeypatch))
    additional = {
        "dspark_profile_observation": {"mode": "auxiliary-transfers"},
        "dspark_confidence_verification": {"mode": "specified_lengths", "profile": override != "not_profile"},
    }
    if override == "old_diagnostic":
        additional["dspark_nan_replay_window"] = [1, 3]
    state.manager.vllm_config = SimpleNamespace(
        additional_config=additional, parallel_config=SimpleNamespace(data_parallel_size=2 if override == "dp" else 1)
    )
    state.manager.model_runner = SimpleNamespace(
        ascend_config=SimpleNamespace(enable_flashcomm1=override == "flashcomm")
    )
    with pytest.raises(ValueError, match="Auxiliary transfers require"):
        state.manager.capture(
            *state.inputs, use_aux_hidden_state_outputs=override != "no_aux", has_lora=override == "lora"
        )
    assert not state.calls and not hasattr(state.manager, "_dspark_auxiliary_capture")
