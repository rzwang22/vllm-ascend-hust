import builtins
import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner as vllm_model_runner
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_ascend.spec_decode import (
    DSPARK_PROPOSER_IDENTITY,
    DSparkRuntimeNotWiredError,
)
from vllm_ascend.worker.v2.spec_decode.dspark import (
    AscendDSparkSpeculator,
    create_dspark_speculator,
)
from vllm_ascend.worker.v2.spec_decode.eagle import init_speculator
from vllm_ascend.worker.v2.spec_decode.runner_init import (
    initialize_ascend_speculator,
    override_core_dspark_speculator_factory,
)


def _dspark_config(
    *,
    method: str = "dspark",
    draft_model_config=...,
    num_speculative_tokens: int = 3,
):
    if draft_model_config is ...:
        draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(dspark_noise_token_id=128799))
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=draft_model_config,
        num_speculative_tokens=num_speculative_tokens,
        use_dspark=lambda: method == "dspark",
        use_eagle=lambda: method in {"mtp", "eagle", "eagle3", "dflash"},
    )
    return SimpleNamespace(speculative_config=speculative_config)


def _proposal_kwargs(*, dummy_run: bool) -> dict:
    return {
        "input_batch": None,
        "attn_metadata": {},
        "slot_mappings": {},
        "last_hidden_states": None,
        "aux_hidden_states": None,
        "num_sampled": None,
        "num_rejected": None,
        "last_sampled": None,
        "next_prefill_tokens": None,
        "temperature": None,
        "seeds": None,
        "dummy_run": dummy_run,
    }


def test_factory_constructs_dedicated_dspark_speculator() -> None:
    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))

    assert type(speculator) is AscendDSparkSpeculator
    assert isinstance(speculator, BaseSpeculator)
    assert not {
        "AscendEagleSpeculator",
        "DFlashSpeculator",
        "DraftModelSpeculator",
    }.intersection(cls.__name__ for cls in type(speculator).__mro__)
    assert speculator.parallel_drafting_token_id == 128799
    assert speculator.supports_mm_inputs is False
    assert speculator.draft_logits is None


def test_selector_identity_matches_constructed_type() -> None:
    module_name, class_name = DSPARK_PROPOSER_IDENTITY.rsplit(".", 1)
    registered_type = getattr(importlib.import_module(module_name), class_name)

    speculator = init_speculator(_dspark_config(), torch.device("cpu"))

    assert registered_type is AscendDSparkSpeculator
    assert type(speculator) is registered_type


@pytest.mark.parametrize(
    ("vllm_config", "error"),
    [
        (SimpleNamespace(speculative_config=None), "speculative_config"),
        (
            SimpleNamespace(speculative_config=SimpleNamespace()),
            "method='dspark'",
        ),
        (_dspark_config(method="eagle"), "method='dspark'"),
        (_dspark_config(draft_model_config=None), "draft_model_config"),
        (
            _dspark_config(draft_model_config=SimpleNamespace(hf_config=None)),
            "draft_model_config.hf_config",
        ),
        (_dspark_config(num_speculative_tokens=0), "positive integer"),
        (
            _dspark_config(draft_model_config=SimpleNamespace(hf_config=SimpleNamespace())),
            "parallel-drafting token id",
        ),
    ],
)
def test_factory_rejects_missing_dspark_configuration(vllm_config, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        create_dspark_speculator(vllm_config, torch.device("cpu"))


@pytest.mark.parametrize("flag", ("dummy_run", "is_profile", "skip_attn_for_dummy_run"))
def test_incomplete_profile_protocol_fails_closed(flag: str) -> None:
    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))
    kwargs = _proposal_kwargs(dummy_run=False)
    kwargs[flag] = True

    with pytest.raises(
        ValueError,
        match="profile execution requires dummy_run=True",
    ):
        speculator.propose(**kwargs)


def test_draft_model_boundary_fails_closed() -> None:
    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))

    with pytest.raises(DSparkRuntimeNotWiredError, match="V2 draft-model loading"):
        _ = speculator.model


class _StopAfterSpeculatorInitialization(Exception):
    pass


def _run_npu_runner_until_after_speculator_initialization(
    monkeypatch: pytest.MonkeyPatch,
    vllm_config,
):
    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    def fake_gpu_model_runner_init(runner, config, device) -> None:
        runner.vllm_config = config
        runner.device = device
        runner.speculative_config = config.speculative_config
        runner.req_states = object()
        runner.input_buffers = object()
        runner.speculator = None
        runner.max_num_reqs = 1
        runner.max_model_len = 8
        runner.max_num_tokens = 8
        runner.num_speculative_steps = 3
        runner.vocab_size = 16
        if config.speculative_config is not None:
            runner.speculator = vllm_model_runner.init_speculator(config, device)

    def stop_before_request_state_initialization(**_kwargs):
        raise _StopAfterSpeculatorInitialization

    vllm_config.parallel_config = SimpleNamespace(
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    monkeypatch.setattr(
        ascend_model_runner,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False)),
    )
    monkeypatch.setattr(ascend_model_runner, "torch_cuda_wrapper", nullcontext)
    monkeypatch.setattr(vllm_model_runner.GPUModelRunner, "__init__", fake_gpu_model_runner_init)
    monkeypatch.setattr(ascend_model_runner, "AscendRequestState", stop_before_request_state_initialization)

    runner = object.__new__(ascend_model_runner.NPUModelRunner)
    with pytest.raises(_StopAfterSpeculatorInitialization):
        ascend_model_runner.NPUModelRunner.__init__(runner, vllm_config, torch.device("cpu"))
    return runner


def test_npu_runner_constructs_only_one_ascend_dspark_speculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2.spec_decode import dspark as dspark_module

    core_selector_calls = 0
    core_constructor_calls = 0
    ascend_factory_calls = 0
    original_core_selector = vllm_model_runner.init_speculator
    original_ascend_factory = dspark_module.create_dspark_speculator

    def core_selector(config, device):
        nonlocal core_selector_calls
        core_selector_calls += 1
        return original_core_selector(config, device)

    def core_constructor(_config, _device):
        nonlocal core_constructor_calls
        core_constructor_calls += 1
        raise AssertionError("core DSpark constructor must not run")

    def ascend_factory(config, device):
        nonlocal ascend_factory_calls
        ascend_factory_calls += 1
        return original_ascend_factory(config, device)

    core_module_name = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
    fake_core_module = ModuleType(core_module_name)
    fake_core_module.DSparkSpeculator = core_constructor
    monkeypatch.setitem(sys.modules, core_module_name, fake_core_module)
    monkeypatch.setattr(vllm_model_runner, "init_speculator", core_selector)
    monkeypatch.setattr(dspark_module, "create_dspark_speculator", ascend_factory)

    runner = _run_npu_runner_until_after_speculator_initialization(monkeypatch, _dspark_config())

    assert core_selector_calls == 0
    assert core_constructor_calls == 0
    assert ascend_factory_calls == 1
    assert type(runner.speculator) is AscendDSparkSpeculator
    assert vllm_model_runner.init_speculator is core_selector


def test_dspark_runner_initialization_does_not_import_core_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_dspark_module = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
    import_attempts: list[str] = []
    original_import = builtins.__import__

    def import_spy(name, *args, **kwargs):
        if name.startswith("vllm.v1.worker.gpu.spec_decode.dspark"):
            import_attempts.append(name)
            raise AssertionError(f"unexpected core DSpark import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, core_dspark_module, raising=False)
    monkeypatch.setattr(builtins, "__import__", import_spy)

    runner = _run_npu_runner_until_after_speculator_initialization(monkeypatch, _dspark_config())

    assert type(runner.speculator) is AscendDSparkSpeculator
    assert import_attempts == []
    assert core_dspark_module not in sys.modules


@pytest.mark.parametrize("method", ["mtp", "eagle", "dflash"])
def test_non_dspark_runner_initialization_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    core_speculator = object()
    ascend_speculator = object()
    calls = {"core": 0, "ascend": 0}

    def core_factory(_config, _device):
        calls["core"] += 1
        return core_speculator

    def ascend_factory(_config, _device):
        calls["ascend"] += 1
        return ascend_speculator

    monkeypatch.setattr(vllm_model_runner, "init_speculator", core_factory)
    monkeypatch.setattr(ascend_model_runner, "init_speculator", ascend_factory)

    runner = _run_npu_runner_until_after_speculator_initialization(
        monkeypatch,
        _dspark_config(method=method),
    )

    assert calls == {"core": 1, "ascend": 1}
    assert runner.speculator is ascend_speculator
    assert vllm_model_runner.init_speculator is core_factory


def test_runner_without_speculative_config_constructs_no_speculator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.worker.v2 import model_runner as ascend_model_runner

    calls = {"core": 0, "ascend": 0}

    def core_factory(_config, _device):
        calls["core"] += 1
        return object()

    def ascend_factory(_config, _device):
        calls["ascend"] += 1
        return object()

    monkeypatch.setattr(vllm_model_runner, "init_speculator", core_factory)
    monkeypatch.setattr(ascend_model_runner, "init_speculator", ascend_factory)

    runner = _run_npu_runner_until_after_speculator_initialization(
        monkeypatch,
        SimpleNamespace(speculative_config=None),
    )

    assert calls == {"core": 0, "ascend": 0}
    assert runner.speculator is None


@pytest.mark.parametrize(
    "ascend_factory",
    [
        pytest.param(
            init_speculator,
            id="active-selector",
        ),
        pytest.param(
            importlib.import_module("vllm_ascend.worker.v2.spec_decode").init_speculator,
            id="package-selector",
        ),
    ],
)
def test_both_ascend_selectors_bypass_core_dspark_factory(
    monkeypatch: pytest.MonkeyPatch,
    ascend_factory,
) -> None:
    config = _dspark_config()
    core_factory_calls = 0

    def core_factory(_config, _device):
        nonlocal core_factory_calls
        core_factory_calls += 1
        raise AssertionError("core DSpark factory must not run")

    monkeypatch.setattr(vllm_model_runner, "init_speculator", core_factory)

    with override_core_dspark_speculator_factory(config, ascend_factory):
        core_initialized_speculator = vllm_model_runner.init_speculator(config, torch.device("cpu"))

    selected_speculator = initialize_ascend_speculator(
        config,
        torch.device("cpu"),
        core_initialized_speculator,
        ascend_factory,
    )

    assert core_factory_calls == 0
    assert selected_speculator is core_initialized_speculator
    assert type(selected_speculator) is AscendDSparkSpeculator
    assert vllm_model_runner.init_speculator is core_factory


def test_graph_lifecycle_remains_eager_before_proposal() -> None:
    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL)
    speculator.capture({})

    assert speculator.requested_cudagraph_mode == CUDAGraphMode.FULL
    assert speculator.cudagraph_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize("method", ["mtp", "eagle", "dflash"])
def test_non_dspark_methods_keep_eagle_selection(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    sentinel = object()
    module_name = "vllm_ascend.worker.v2.spec_decode.eagle.speculator"
    fake_module = ModuleType(module_name)
    fake_module.AscendEagleSpeculator = lambda _config, _device: sentinel
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    selected = init_speculator(_dspark_config(method=method), None)

    assert selected is sentinel
