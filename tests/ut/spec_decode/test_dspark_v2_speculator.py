import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
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


@pytest.mark.parametrize(
    ("dummy_run", "boundary"),
    [(False, "V2 proposal"), (True, "V2 dummy-run proposal")],
)
def test_proposal_boundaries_fail_closed(dummy_run: bool, boundary: str) -> None:
    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))

    with pytest.raises(DSparkRuntimeNotWiredError, match=boundary):
        speculator.propose(**_proposal_kwargs(dummy_run=dummy_run))


def test_draft_model_boundary_fails_closed() -> None:
    speculator = create_dspark_speculator(_dspark_config(), torch.device("cpu"))

    with pytest.raises(DSparkRuntimeNotWiredError, match="V2 draft-model loading"):
        _ = speculator.model


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
