# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
import builtins
import json
import sys
import types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tests.e2e.nightly.single_node.spec_decode.test_dspark_proposal_inputs_prepare as prepare_harness
import tests.e2e.nightly.single_node.spec_decode.test_dspark_single_request_realdata as realdata_harness
from tools.dspark import build_m2_5a_dataset_assets as asset_builder
from tools.dspark import summarize_m2_5a_performance as performance_summary
from tools.dspark.build_m2_5a_dataset_assets import build_assets
from tools.dspark.m2_5a_common import (
    ASSET_FILES,
    build_execution_plan,
    read_jsonl,
    sha256_file,
    token_ids_sha256,
    verify_asset_bundle,
    write_jsonl,
)
from tools.dspark.summarize_m2_5a_performance import summarize_performance
from tools.dspark.validate_m2_5a_results import validate_result_pair


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, template: str = "CHAT {messages} ASSISTANT") -> None:
        self.chat_template = template

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        rendered_messages = " ".join(f"{message['role']} {message['content']}" for message in messages)
        return self.chat_template.format(messages=rendered_messages)

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": list(range(10, 10 + len(text.split())))}


class _CleanupWorker:
    def __init__(self) -> None:
        self.outputs: list[Any] = []
        self.runner_outputs: list[Any] = []

    def execute_model(self, scheduler_output: Any) -> Any:
        self.outputs.append(scheduler_output)
        return self.runner_outputs.pop(0) if self.runner_outputs else None


class _TensorLikeTraceValue:
    def __init__(self, value: list[int]) -> None:
        self.value = value
        self.detach_calls = 0
        self.cpu_calls = 0
        self.tolist_calls = 0

    def detach(self) -> _TensorLikeTraceValue:
        self.detach_calls += 1
        return self

    def cpu(self) -> _TensorLikeTraceValue:
        self.cpu_calls += 1
        return self

    def tolist(self) -> list[int]:
        self.tolist_calls += 1
        return self.value


class _NumpyNdarrayTraceValue:
    """Dependency-free stand-in for NumPy's host-resident tolist protocol."""

    __module__ = "numpy"

    def __init__(self, value: Any) -> None:
        self.value = value
        self.tolist_calls = 0

    def tolist(self) -> Any:
        self.tolist_calls += 1
        return self.value


class _TraceTokenMatrix:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def __getitem__(self, key: tuple[int, slice]) -> list[int]:
        row, token_slice = key
        return self.rows[row][token_slice]


def _cleanup_runtime(mode: str) -> SimpleNamespace:
    runner = SimpleNamespace(
        req_states=SimpleNamespace(req_id_to_index={}),
        input_batch=SimpleNamespace(req_id_to_index={}),
        requests={},
    )
    runtime = SimpleNamespace(runner=runner, worker=_CleanupWorker())
    if mode == "dspark":
        runtime.speculator = SimpleNamespace(
            _published_candidate_tokens=None,
            _published_proposal_step_epoch=None,
            _published_proposal_request_ids=None,
            _published_proposal_request_state_indices=None,
            _current_proposal_lifecycle=None,
            _prepared_step_epoch=None,
            _context_kv_step_epoch=None,
            _draft_forward_step_epoch=None,
            _markov_attempt_step_epoch=None,
            _markov_step_epoch=None,
            _markov_result=None,
        )
    return runtime


def _cleanup_scheduler(*outputs: Any) -> SimpleNamespace:
    scheduled_outputs = list(outputs)
    manager = SimpleNamespace(req_to_blocks={})
    kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=[manager]),
        get_block_ids=lambda _request_id: ([], []),
        _compressed_request_physical_tokens={},
        _compression_destination_reservations={},
    )

    def schedule() -> Any:
        return scheduled_outputs.pop(0)

    return SimpleNamespace(
        schedule=schedule,
        requests={},
        running=[],
        waiting=[],
        skipped_waiting=[],
        finished_req_ids=set(),
        kv_cache_manager=kv_cache_manager,
    )


def _scheduler_output(*, finished_req_ids: set[str]) -> SimpleNamespace:
    return SimpleNamespace(
        total_num_scheduled_tokens=0,
        finished_req_ids=finished_req_ids,
    )


def _empty_model_runner_output(**overrides: Any) -> SimpleNamespace:
    fields = {
        "req_ids": [],
        "req_id_to_index": {},
        "sampled_token_ids": [],
        "logprobs": None,
        "prompt_logprobs_dict": {},
        "pooler_output": None,
        "kv_connector_output": None,
        "ec_connector_output": None,
        "kv_cache_compression_plans": None,
        "num_nans_in_logits": None,
        "cudagraph_stats": None,
        "spec_decode_proposer_latency_seconds": 0.0,
        "spec_decode_verification_latency_seconds": 0.0,
        "spec_decode_num_forwards": 0,
        "routed_experts": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _install_deepseek_tokenizer_module(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer_class: type,
) -> None:
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    tokenizers = types.ModuleType("vllm.tokenizers")
    tokenizers.__path__ = []
    deepseek_v4 = types.ModuleType("vllm.tokenizers.deepseek_v4")
    deepseek_v4.DeepseekV4Tokenizer = tokenizer_class
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers.deepseek_v4", deepseek_v4)


def test_tokenizer_loader_uses_local_vllm_deepseek_v4_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    expected_tokenizer = object()

    class _DeepseekV4Tokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> object:
            calls.append((path, kwargs))
            return expected_tokenizer

    _install_deepseek_tokenizer_module(monkeypatch, _DeepseekV4Tokenizer)
    checkpoint = tmp_path / "checkpoint"

    tokenizer = asset_builder._load_tokenizer(checkpoint)

    assert tokenizer is expected_tokenizer
    assert calls == [(str(checkpoint.resolve()), {"local_files_only": True})]


def test_tokenizer_loader_reports_missing_vllm_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_deepseek_tokenizer(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "vllm.tokenizers.deepseek_v4":
            raise ImportError("DeepSeek V4 tokenizer is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_deepseek_tokenizer)
    for name in ("vllm.tokenizers.deepseek_v4", "vllm.tokenizers", "vllm"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(RuntimeError, match="requires the vLLM DeepSeek V4 tokenizer") as error:
        asset_builder._load_tokenizer(tmp_path)

    assert isinstance(error.value.__cause__, ImportError)


def test_tokenizer_loader_reports_checkpoint_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingDeepseekV4Tokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> object:
            raise ValueError(f"invalid tokenizer at {path}: {kwargs}")

    _install_deepseek_tokenizer_module(monkeypatch, _FailingDeepseekV4Tokenizer)

    with pytest.raises(RuntimeError, match="Unable to load the DeepSeek V4 tokenizer") as error:
        asset_builder._load_tokenizer(tmp_path)

    assert isinstance(error.value.__cause__, ValueError)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _assets(tmp_path: Path, *, template: str = "CHAT {messages} ASSISTANT", name: str = "assets") -> Path:
    livecodebench_path = tmp_path / "test6.jsonl"
    sharegpt_path = tmp_path / "sharegpt.json"
    gsm8k_path = tmp_path / "gsm8k.jsonl"
    tokenizer_path = tmp_path / f"tokenizer-{name}"
    output_path = tmp_path / name
    _write_jsonl(
        livecodebench_path,
        [
            {
                "question_id": f"lcb-{index:03d}",
                "question_content": f"Solve deterministic program {index} with proof.",
                "difficulty": ("easy", "medium", "hard")[index % 3],
                "platform": ("leetcode", "codeforces")[index % 2],
                "contest_date": f"2025-01-{index % 28 + 1:02d}",
                "test": ["hidden"],
            }
            for index in range(70)
        ],
    )
    sharegpt_path.write_text(
        json.dumps(
            [
                {
                    "id": f"conversation-{index:03d}",
                    "conversations": [
                        {"from": "human", "value": " ".join([f"short{index}"] * 995)},
                        {"from": "gpt", "value": " ".join([f"answer{index}"] * 1000)},
                        {"from": "human", "value": " ".join([f"long{index}"] * 2100)},
                    ],
                }
                for index in range(40)
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        gsm8k_path,
        [
            {
                "id": f"gsm-{index:03d}",
                "question": f"If Alice owns {index + 1} books and gets two, how many?",
                "answer": f"She has {index + 3}. #### {index + 3}",
            }
            for index in range(70)
        ],
    )
    tokenizer_path.mkdir()
    (tokenizer_path / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    (tokenizer_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}) + "\n", encoding="utf-8"
    )
    sources = {
        "livecodebench": {
            "repo": "livecodebench/code_generation_lite",
            "revision": "7e47b68262a1da6d8634e205b69d88d978d53dc9",
            "raw_file_sha256": sha256_file(livecodebench_path),
        },
        "sharegpt": {
            "repo": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "revision": "044ca94aec8d8cdee04973000738431161247677",
            "raw_file_sha256": sha256_file(sharegpt_path),
        },
        "gsm8k": {
            "repo": "openai/gsm8k",
            "revision": "cc7b047b6e5bb11b4f1af84efc572db110a51b3c",
            "raw_file_sha256": sha256_file(gsm8k_path),
        },
    }
    source_revisions = tmp_path / "source_revisions.json"
    source_revisions.write_text(json.dumps(sources, sort_keys=True) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        livecodebench=livecodebench_path,
        sharegpt=sharegpt_path,
        gsm8k=gsm8k_path,
        source_revisions=source_revisions,
        tokenizer=tokenizer_path,
        output_dir=output_path,
        seed=0,
    )
    return build_assets(args, tokenizer=_FakeTokenizer(template))


@pytest.fixture(scope="module")
def frozen_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _assets(tmp_path_factory.mktemp("m25a"))


def test_verify_only_does_not_import_tokenizer_runtime(
    frozen_assets: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def reject_runtime_imports(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "vllm" or name.startswith(("vllm.", "torch", "transformers")):
            raise AssertionError(f"verify-only imported runtime dependency {name}")
        return original_import(name, *args, **kwargs)

    def reject_tokenizer_load(path: Path) -> Any:
        raise AssertionError(f"verify-only loaded tokenizer {path}")

    monkeypatch.setattr(builtins, "__import__", reject_runtime_imports)
    monkeypatch.setattr(asset_builder, "_load_tokenizer", reject_tokenizer_load)
    monkeypatch.setattr(sys, "argv", [str(asset_builder.__file__), "--verify-only", str(frozen_assets)])

    assert asset_builder.main() == 0


def _runtime_contract_fixture(
    mode: str,
    *,
    prefix_caching: bool = False,
    block_size: int = realdata_harness.EXPECTED_BLOCK_SIZE,
    rank: int = 0,
) -> SimpleNamespace:
    speculative_config = None
    if mode == "dspark":
        speculative_config = SimpleNamespace(
            method="dspark",
            num_speculative_tokens=realdata_harness.EXPECTED_K,
        )
    return SimpleNamespace(
        launch=SimpleNamespace(rank=rank),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=realdata_harness.EXPECTED_TP_SIZE,
                pipeline_parallel_size=1,
                enable_expert_parallel=True,
            ),
            cache_config=SimpleNamespace(
                enable_prefix_caching=prefix_caching,
                block_size=block_size,
            ),
            model_config=SimpleNamespace(
                enforce_eager=True,
                max_model_len=realdata_harness.EXPECTED_MAX_MODEL_LEN,
            ),
            speculative_config=speculative_config,
        ),
    )


def test_m2_5a_kv_budget_preflight_is_explicit_and_preserves_two_gib() -> None:
    with pytest.raises(ValueError, match="explicitly set before model loading"):
        realdata_harness._m2_5a_kv_cache_budget({})
    with pytest.raises(ValueError, match=r"got 536870912 bytes.*2147483648 bytes.*max_model_len=8192"):
        realdata_harness._m2_5a_kv_cache_budget({"DSPARK_KV_CACHE_BYTES": "536870912"})

    assert realdata_harness._m2_5a_kv_cache_budget({"DSPARK_KV_CACHE_BYTES": "2147483648"}) == 2147483648


@pytest.mark.parametrize("mode", ["target_only", "dspark"])
def test_m2_5a_runtime_contract_disables_prefix_cache_for_both_modes(
    mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _runtime_contract_fixture(mode, rank=3)

    contract = realdata_harness._validate_runtime_contract(
        runtime,
        mode,
        "smoke",
        2147483648,
    )

    assert contract == {
        "rank": 3,
        "mode": mode,
        "profile": "smoke",
        "max_model_len": 8192,
        "kv_cache_bytes": 2147483648,
        "block_size": 128,
        "prefix_caching_enabled": False,
        "enforce_eager": True,
        "tp_size": 8,
        "pp_size": 1,
        "expert_parallel": True,
    }
    assert f"{realdata_harness.RUNTIME_CONTRACT}=" in capsys.readouterr().out


def test_m2_5a_runtime_contract_reports_prefix_and_block_failures_separately() -> None:
    with pytest.raises(RuntimeError, match=r"prefix caching disabled, got True\.$"):
        realdata_harness._validate_runtime_contract(
            _runtime_contract_fixture("target_only", prefix_caching=True),
            "target_only",
            "smoke",
            2147483648,
        )
    with pytest.raises(RuntimeError, match=r"block_size=128, got 64\.$"):
        realdata_harness._validate_runtime_contract(
            _runtime_contract_fixture("target_only", block_size=64),
            "target_only",
            "smoke",
            2147483648,
        )


@pytest.mark.parametrize("mode", ["target_only", "dspark"])
def test_finished_request_event_is_delivered_once_before_empty_flush(mode: str) -> None:
    request_id = f"request-{mode}"
    runtime = _cleanup_runtime(mode)
    empty_output = _scheduler_output(finished_req_ids=set())
    scheduler = _cleanup_scheduler(empty_output)
    lifecycle = realdata_harness._FinishedRequestLifecycle(request_id)
    finished_output = _scheduler_output(finished_req_ids={request_id})

    assert realdata_harness._execute_scheduler_output(runtime, finished_output, lifecycle) is None
    realdata_harness._flush_finished_request(runtime, scheduler, lifecycle)

    assert lifecycle.observed_count == 1
    assert lifecycle.worker_delivery_count == 1
    assert runtime.worker.outputs == [finished_output, empty_output]


@pytest.mark.parametrize(
    "connector_wrapped",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize("mode", ["target_only", "dspark"])
def test_zero_token_cleanup_accepts_canonical_and_connector_empty_outputs(
    mode: str,
    connector_wrapped: bool,
) -> None:
    request_id = f"request-{mode}"
    connector_output = (
        SimpleNamespace(finished_sending={request_id}, finished_recving=None) if connector_wrapped else None
    )
    runner_output = _empty_model_runner_output(
        kv_connector_output=connector_output,
    )
    runtime = _cleanup_runtime(mode)
    runtime.worker.runner_outputs.extend([runner_output, _empty_model_runner_output()])
    scheduler = _cleanup_scheduler(_scheduler_output(finished_req_ids=set()))
    lifecycle = realdata_harness._FinishedRequestLifecycle(request_id)
    finished_output = _scheduler_output(finished_req_ids={request_id})

    first_runner_output = realdata_harness._execute_scheduler_output(
        runtime,
        finished_output,
        lifecycle,
    )
    realdata_harness._assert_canonical_zero_token_runner_output(first_runner_output)
    realdata_harness._flush_finished_request(runtime, scheduler, lifecycle)

    assert lifecycle.observed_count == 1
    assert lifecycle.worker_delivery_count == 1


@pytest.mark.parametrize(
    ("runner_output", "message"),
    [
        (_empty_model_runner_output(req_ids=["request"]), "req_ids"),
        (
            _empty_model_runner_output(
                req_id_to_index={"request": 0},
            ),
            "req_id_to_index",
        ),
        (_empty_model_runner_output(sampled_token_ids=[[7]]), "sampled_token_ids"),
        (_empty_model_runner_output(logprobs=object()), "logprobs"),
        (_empty_model_runner_output(pooler_output=[object()]), "pooler"),
        (_empty_model_runner_output(spec_decode_num_forwards=1), "model forwards"),
    ],
)
def test_zero_token_output_with_execution_payload_fails_closed(
    runner_output: Any,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        realdata_harness._assert_canonical_zero_token_runner_output(runner_output)


def test_missing_finished_event_with_live_request_fails_closed() -> None:
    request_id = "request-missing-event"
    runtime = _cleanup_runtime("target_only")
    scheduler = _cleanup_scheduler(_scheduler_output(finished_req_ids=set()))
    scheduler.requests[request_id] = SimpleNamespace(request_id=request_id)
    lifecycle = realdata_harness._FinishedRequestLifecycle(request_id)

    with pytest.raises(RuntimeError, match="published and delivered.*exactly once"):
        realdata_harness._flush_finished_request(runtime, scheduler, lifecycle)


def test_repeated_finished_event_is_rejected_without_second_worker_cleanup() -> None:
    request_id = "request-duplicate-event"
    runtime = _cleanup_runtime("target_only")
    duplicate_output = _scheduler_output(finished_req_ids={request_id})
    scheduler = _cleanup_scheduler(duplicate_output)
    lifecycle = realdata_harness._FinishedRequestLifecycle(request_id)
    first_output = _scheduler_output(finished_req_ids={request_id})
    realdata_harness._execute_scheduler_output(runtime, first_output, lifecycle)

    with pytest.raises(RuntimeError, match="must not repeat or retain"):
        realdata_harness._flush_finished_request(runtime, scheduler, lifecycle)

    assert runtime.worker.outputs == [first_output]


@pytest.mark.parametrize(
    ("state_owner", "message"),
    [
        ("runner_requests", "runner cached requests"),
        ("runner_req_states", "runner request tensors"),
        ("runner_input_batch", "runner input batch"),
        ("kv_blocks", "retains KV block ownership"),
        ("kv_manager", "remains in KV manager groups"),
    ],
)
def test_finished_request_must_be_absent_from_runner_and_kv_ownership(
    state_owner: str,
    message: str,
) -> None:
    request_id = f"request-stale-{state_owner}"
    runtime = _cleanup_runtime("target_only")
    scheduler = _cleanup_scheduler(_scheduler_output(finished_req_ids=set()))
    if state_owner == "runner_requests":
        runtime.runner.requests[request_id] = object()
    elif state_owner == "runner_req_states":
        runtime.runner.req_states.req_id_to_index[request_id] = 0
    elif state_owner == "runner_input_batch":
        runtime.runner.input_batch.req_id_to_index[request_id] = 0
    elif state_owner == "kv_blocks":
        scheduler.kv_cache_manager.get_block_ids = lambda _request_id: ([7], [])
    else:
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[request_id] = [object()]
    lifecycle = realdata_harness._FinishedRequestLifecycle(request_id)
    realdata_harness._execute_scheduler_output(
        runtime,
        _scheduler_output(finished_req_ids={request_id}),
        lifecycle,
    )

    with pytest.raises(RuntimeError, match=message):
        realdata_harness._flush_finished_request(runtime, scheduler, lifecycle)


@pytest.mark.parametrize("mode", ["target_only", "dspark"])
def test_consecutive_request_cleanup_does_not_retain_prior_event(mode: str) -> None:
    request_ids = tuple(f"request-{mode}-{index}" for index in range(10))
    runtime = _cleanup_runtime(mode)
    scheduler = _cleanup_scheduler(*(_scheduler_output(finished_req_ids=set()) for _ in request_ids))

    for request_id in request_ids:
        lifecycle = realdata_harness._FinishedRequestLifecycle(request_id)
        realdata_harness._execute_scheduler_output(
            runtime,
            _scheduler_output(finished_req_ids={request_id}),
            lifecycle,
        )
        realdata_harness._flush_finished_request(runtime, scheduler, lifecycle)

    delivered_events = [output.finished_req_ids for output in runtime.worker.outputs]
    assert delivered_events == [event for request_id in request_ids for event in ({request_id}, set())]


def test_m2_5a_prepare_launch_config_restores_default_after_failure() -> None:
    assert prepare_harness._PREPARE_ONLY_LAUNCH_CONFIG.get() is None

    with (
        pytest.raises(RuntimeError, match="injected failure"),
        prepare_harness.prepare_only_launch_config(
            enable_prefix_caching=False,
            kv_cache_bytes=2147483648,
        ),
    ):
        launch_config = prepare_harness._PREPARE_ONLY_LAUNCH_CONFIG.get()
        assert launch_config is not None
        assert launch_config.enable_prefix_caching is False
        assert launch_config.kv_cache_bytes == 2147483648
        raise RuntimeError("injected failure")

    assert prepare_harness._PREPARE_ONLY_LAUNCH_CONFIG.get() is None


def test_repeated_asset_build_is_byte_identical(tmp_path: Path) -> None:
    first = _assets(tmp_path, name="first")
    second = _assets(tmp_path, name="second")

    assert first.read_bytes() == second.read_bytes()
    for filename in (*ASSET_FILES, "SHA256SUMS"):
        assert (first.parent / filename).read_bytes() == (second.parent / filename).read_bytes()


def test_source_revision_or_hash_is_required(tmp_path: Path) -> None:
    manifest = _assets(tmp_path)
    sources_path = tmp_path / "source_revisions.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    del sources["gsm8k"]["raw_file_sha256"]
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    args = argparse.Namespace(
        livecodebench=tmp_path / "test6.jsonl",
        sharegpt=tmp_path / "sharegpt.json",
        gsm8k=tmp_path / "gsm8k.jsonl",
        source_revisions=sources_path,
        tokenizer=tmp_path / "tokenizer-assets",
        output_dir=manifest.parent / "invalid",
        seed=0,
    )
    with pytest.raises(ValueError, match="raw_file_sha256"):
        build_assets(args, tokenizer=_FakeTokenizer())


def test_dataset_contracts_and_smoke_subset(frozen_assets: Path) -> None:
    manifest = verify_asset_bundle(frozen_assets)
    root = frozen_assets.parent
    livecodebench = read_jsonl(root / "livecodebench_64.jsonl")
    sharegpt = read_jsonl(root / "sharegpt_64.jsonl")
    gsm8k = read_jsonl(root / "gsm8k_64.jsonl")
    synthetic = read_jsonl(root / "synthetic_lengths.jsonl")
    smoke = read_jsonl(root / "smoke_cases.jsonl")
    full = read_jsonl(root / "full_cases.jsonl")

    assert [len(livecodebench), len(sharegpt), len(gsm8k), len(synthetic), len(smoke), len(full)] == [
        64,
        64,
        64,
        16,
        10,
        208,
    ]
    assert all(case["prompt_token_count"] <= 2048 and case["output_cap"] == 1024 for case in livecodebench)
    assert {case["metadata"]["difficulty"] for case in livecodebench} == {"easy", "medium", "hard"}
    assert {case["metadata"]["platform"] for case in livecodebench} == {"leetcode", "codeforces"}
    assert sum(922 <= case["prompt_token_count"] <= 1126 and case["output_cap"] == 128 for case in sharegpt) == 32
    assert sum(3686 <= case["prompt_token_count"] <= 4506 and case["output_cap"] == 256 for case in sharegpt) == 32
    assert all(case["messages"][-1]["role"] == "user" for case in sharegpt)
    assert all(case["prompt_token_count"] <= 1024 and case["metadata"]["final_answer"] for case in gsm8k)
    assert [(case["prompt_token_count"], case["output_cap"]) for case in synthetic] == [
        (prompt_tokens, output_cap)
        for prompt_tokens, output_cap in ((128, 16), (1024, 256), (2048, 512), (4096, 1024))
        for _ in range(4)
    ]
    assert all(case["ignore_eos"] for case in synthetic)
    assert {case["case_id"] for case in smoke}.issubset({case["case_id"] for case in full})
    assert len(manifest["profiles"]["smoke"]) == 10


def test_tokenizer_template_change_changes_manifest(tmp_path: Path) -> None:
    first = _assets(tmp_path, template="CHAT {messages} ASSISTANT", name="first")
    second = _assets(tmp_path, template="ALT CHAT {messages} ASSISTANT", name="second")

    assert sha256_file(first) != sha256_file(second)


def _result_record(
    case: dict[str, Any], *, rank: int, mode: str, sequence: int, repeat: int, manifest_hash: str
) -> dict[str, Any]:
    output_ids = list(range(case["output_cap"])) if case["dataset"] == "synthetic" else [100 + sequence, 200 + sequence]
    is_dspark = mode == "dspark"
    return {
        "rank": rank,
        "mode": mode,
        "profile": "smoke",
        "dataset": case["dataset"],
        "case_id": case["case_id"],
        "lifecycle_repeat": repeat,
        "profile_case_index": sequence % 10,
        "request_sequence_index": sequence,
        "request_id": f"{mode}-{sequence}",
        "proposal_epoch_start": sequence * 10 + 1 if is_dspark else None,
        "proposal_epoch_end": sequence * 10 + 5 if is_dspark else None,
        "consumer_epoch_start": sequence * 10 + 2 if is_dspark else None,
        "consumer_epoch_end": sequence * 10 + 6 if is_dspark else None,
        "prompt_token_count": case["prompt_token_count"],
        "prompt_token_sha256": case["ordered_prompt_token_sha256"],
        "output_cap": case["output_cap"],
        "ignore_eos": case["ignore_eos"],
        "output_token_count": len(output_ids),
        "output_token_ids": output_ids,
        "output_token_sha256": token_ids_sha256(output_ids),
        "stop_reason": {"finish_reason": "length", "stop_reason": None},
        "usage_prompt_tokens": case["prompt_token_count"],
        "usage_completion_tokens": len(output_ids),
        "completed_rounds": 1 if is_dspark else 0,
        "proposal_generated_count": 1 if is_dspark else 0,
        "proposal_installed_count": 1 if is_dspark else 0,
        "proposal_consumed_count": 1 if is_dspark else 0,
        "terminal_discarded_proposal_count": 1 if is_dspark and sequence == 0 else 0,
        "terminal_partial_commit": is_dspark and sequence == 0,
        "post_finish_target_forward_count": 0,
        "post_finish_verification_count": 0,
        "cleanup_complete": True,
        "state_isolation_verified": True,
        "historical_error_count": 0,
        "manifest_sha256": manifest_hash,
    }


def _result_dirs(tmp_path: Path, manifest: Path, expected_ranks: int = 2) -> tuple[Path, Path]:
    cases = read_jsonl(manifest.parent / "smoke_cases.jsonl")
    manifest_hash = sha256_file(manifest)
    target_root = tmp_path / "target"
    dspark_root = tmp_path / "dspark"
    target_root.mkdir(parents=True)
    dspark_root.mkdir(parents=True)
    for rank in range(expected_ranks):
        for root, mode in ((target_root, "target_only"), (dspark_root, "dspark")):
            records = [
                _result_record(case, rank=rank, mode=mode, sequence=index, repeat=0, manifest_hash=manifest_hash)
                for index, case in enumerate(cases)
            ]
            path = root / f"rank-{rank}.jsonl"
            write_jsonl(path, records)
            path.with_suffix(".jsonl.sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")
    return target_root, dspark_root


def _rewrite_results(path: Path, records: list[dict[str, Any]]) -> None:
    write_jsonl(path, records)
    path.with_suffix(".jsonl.sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")


def _performance_result_dirs(
    tmp_path: Path,
    manifest: Path,
    expected_ranks: int = 2,
) -> tuple[Path, Path]:
    target_root, dspark_root = _result_dirs(tmp_path, manifest, expected_ranks)
    for root, mode in ((target_root, "target_only"), (dspark_root, "dspark")):
        for rank in range(expected_ranks):
            path = root / f"rank-{rank}.jsonl"
            records = read_jsonl(path)
            for record in records:
                if mode == "dspark":
                    record["output_token_ids"][0] += 1
                    record["output_token_sha256"] = token_ids_sha256(record["output_token_ids"])
                output_count = record["output_token_count"]
                verification_count = 1 if mode == "dspark" else 0
                accepted_count = 4 if mode == "dspark" else 0
                verification_committed = min(output_count, accepted_count + 1) if mode == "dspark" else 0
                record.update(
                    target_forward_count=2 if mode == "target_only" else 1,
                    verification_count=verification_count,
                    diagnostic_only=False,
                    performance_validated=True,
                    performance_provisional=True,
                    bit_exact_validated=False,
                    performance={
                        "timing_boundary": "test",
                        "prefill_definition": "time_to_first_scheduler_commit",
                        "prefill_latency_seconds": 0.5,
                        "prefill_output_token_count": 1,
                        "decode_latency_seconds": 2.0 if mode == "target_only" else 1.0,
                        "decode_output_token_count": output_count - 1,
                        "inference_latency_seconds": 2.5 if mode == "target_only" else 1.5,
                        "milliseconds_per_output_token": 1.0,
                        "output_tokens_per_second": 100.0,
                        "decode_milliseconds_per_output_token": 1.0,
                        "decode_output_tokens_per_second": 100.0,
                        "scheduler_seconds": 0.01,
                        "scheduler_update_seconds": 0.02,
                        "model_execute_host_seconds": 0.03,
                        "sample_materialize_seconds": 0.04,
                        "draft_install_seconds": 0.01 if mode == "dspark" else 0.0,
                        "spec_decode_proposer_latency_seconds": 0.1 if mode == "dspark" else 0.0,
                        "spec_decode_verification_latency_seconds": 0.2 if mode == "dspark" else 0.0,
                        "scheduled_draft_token_count": 5 if mode == "dspark" else 0,
                        "accepted_candidate_metrics_source": (performance_summary.ACCEPTED_METRICS_SOURCE),
                        "accepted_candidate_tokens_total": accepted_count,
                        "average_accepted_candidate_tokens_per_verification": (4.0 if mode == "dspark" else None),
                        "replacement_tokens_total": verification_count,
                        "bonus_tokens_total": 0,
                        "committed_tokens_total": output_count,
                        "verification_committed_tokens_total": verification_committed,
                        "effective_committed_tokens_per_verification": (
                            float(verification_committed) if mode == "dspark" else None
                        ),
                        "accepted_draft_token_count": accepted_count,
                        "average_accepted_tokens_per_verification": 4.0 if mode == "dspark" else 0.0,
                        "draft_token_acceptance_rate": 0.8 if mode == "dspark" else 0.0,
                        "npu_memory": {
                            "allocated_before": 100,
                            "reserved_before": 120,
                            "allocated_after": 101,
                            "reserved_after": 120,
                            "peak_allocated": 110,
                            "peak_reserved": 130,
                            "peak_allocated_increment": 10,
                            "peak_reserved_increment": 10,
                        },
                    },
                )
            _rewrite_results(path, records)
            summary = {
                "rank": rank,
                "mode": mode,
                "profile": "smoke",
                "performance_validated": True,
                "performance_provisional": True,
                "phase_timings": {
                    "init_device_seconds": 0.25,
                    "model_load_seconds": 10.0,
                    "kv_cache_init_seconds": 0.5,
                },
            }
            (root / f"rank-{rank}.summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return target_root, dspark_root


def _steady_state_performance_result_dirs(
    tmp_path: Path,
    manifest: Path,
    expected_ranks: int = 2,
) -> tuple[Path, Path]:
    target_root, dspark_root = _performance_result_dirs(tmp_path, manifest, expected_ranks)
    selected_indices = (4, 5, 9)
    identity = {
        "plugin_head": "plugin-sha",
        "core_head": "core-sha",
        "target_model": {"path": "/checkpoint", "config.json_sha256": "config-sha"},
        "draft_model": {"path": "/checkpoint", "config.json_sha256": "config-sha"},
        "manifest_sha256": sha256_file(manifest),
    }
    for root, mode in ((target_root, "target_only"), (dspark_root, "dspark")):
        for rank in range(expected_ranks):
            original = read_jsonl(root / f"rank-{rank}.jsonl")
            records: list[dict[str, Any]] = []
            sequence = 0
            for case_index in selected_indices:
                for repeat_kind, repeat_count in (("warmup", 1), ("measured", 3)):
                    for repeat_index in range(repeat_count):
                        record = deepcopy(original[case_index])
                        record.update(
                            request_sequence_index=sequence,
                            request_id=f"m25a-{mode}-{sequence}-{repeat_kind}-{repeat_index}-{record['case_id']}",
                            performance_protocol="per_case_steady_state_v1",
                            performance_repeat_kind=repeat_kind,
                            performance_repeat_index=repeat_index,
                            performance_artifact_identity=identity,
                        )
                        latency_multiplier = 4.0 if repeat_kind == "warmup" else (0.95, 1.0, 1.05)[repeat_index]
                        base_decode = 2.0 if mode == "target_only" else 1.0
                        decode = base_decode * latency_multiplier
                        performance = record["performance"]
                        performance["prefill_latency_seconds"] = 0.5 * latency_multiplier
                        performance["decode_latency_seconds"] = decode
                        performance["inference_latency_seconds"] = performance["prefill_latency_seconds"] + decode
                        output_count = record["output_token_count"]
                        decode_count = performance["decode_output_token_count"]
                        performance["milliseconds_per_output_token"] = (
                            1000.0 * performance["inference_latency_seconds"] / output_count
                        )
                        performance["output_tokens_per_second"] = (
                            output_count / performance["inference_latency_seconds"]
                        )
                        performance["decode_milliseconds_per_output_token"] = 1000.0 * decode / decode_count
                        performance["decode_output_tokens_per_second"] = decode_count / decode
                        records.append(record)
                        sequence += 1
            _rewrite_results(root / f"rank-{rank}.jsonl", records)
            summary_path = root / f"rank-{rank}.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(
                performance_protocol="per_case_steady_state_v1",
                performance_warmup_repeats=1,
                performance_measured_repeats=3,
                performance_case_ids=[original[index]["case_id"] for index in selected_indices],
                performance_artifact_identity=identity,
            )
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return target_root, dspark_root


def test_performance_mode_is_explicit_and_rejects_trace_or_launch_blocking() -> None:
    trace = realdata_harness._ForensicTraceConfig(None, False, None)

    assert realdata_harness._performance_config({}, trace, 1).enabled is False
    default = realdata_harness._performance_config({"DSPARK_M25A_PERFORMANCE": "1"}, trace, 1)
    assert default == realdata_harness._PerformanceConfig(enabled=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        realdata_harness._performance_config(
            {"DSPARK_M25A_PERFORMANCE": "1"},
            realdata_harness._ForensicTraceConfig("synthetic:1024:0", False, None),
            1,
        )
    with pytest.raises(ValueError, match="ASCEND_LAUNCH_BLOCKING=0"):
        realdata_harness._performance_config(
            {
                "DSPARK_M25A_PERFORMANCE": "1",
                "ASCEND_LAUNCH_BLOCKING": "1",
            },
            trace,
            1,
        )
    with pytest.raises(ValueError, match="independent launches"):
        realdata_harness._performance_config({"DSPARK_M25A_PERFORMANCE": "1"}, trace, 3)
    with pytest.raises(ValueError, match="require DSPARK_M25A_PERFORMANCE=1"):
        realdata_harness._performance_config(
            {"DSPARK_M25A_PERFORMANCE_WARMUP_REPEATS": "1"},
            trace,
            1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        realdata_harness._performance_config(
            {
                "DSPARK_M25A_PERFORMANCE": "1",
                "DSPARK_M25A_PERFORMANCE_WARMUP_REPEATS": "-1",
            },
            trace,
            1,
        )
    with pytest.raises(ValueError, match="positive"):
        realdata_harness._performance_config(
            {
                "DSPARK_M25A_PERFORMANCE": "1",
                "DSPARK_M25A_PERFORMANCE_MEASURED_REPEATS": "0",
            },
            trace,
            1,
        )


def test_performance_plan_runs_each_case_warmup_then_measured_with_unique_requests(
    frozen_assets: Path,
) -> None:
    cases = read_jsonl(frozen_assets.parent / "smoke_cases.jsonl")
    base_plan = build_execution_plan(cases, 1)
    assert (
        realdata_harness._build_performance_plan(
            base_plan,
            realdata_harness._PerformanceConfig(enabled=False),
        )
        is base_plan
    )
    selected = (cases[4]["case_id"], cases[5]["case_id"], cases[9]["case_id"])
    config = realdata_harness._PerformanceConfig(
        enabled=True,
        warmup_repeats=1,
        measured_repeats=3,
        case_ids=selected,
    )

    plan = realdata_harness._build_performance_plan(base_plan, config)

    assert len(plan) == 12
    assert [case["case_id"] for case in plan] == [case_id for case_id in selected for _ in range(4)]
    assert [case["performance_repeat_kind"] for case in plan] == [
        kind for _ in selected for kind in ("warmup", "measured", "measured", "measured")
    ]
    assert [case["performance_repeat_index"] for case in plan] == [index for _ in selected for index in (0, 0, 1, 2)]
    assert [case["request_sequence_index"] for case in plan] == list(range(12))
    request_ids = [realdata_harness._request_id("dspark", case) for case in plan]
    assert len(set(request_ids)) == 12
    assert all(case["lifecycle_repeat"] == 0 for case in plan)


def test_performance_plan_rejects_missing_or_duplicate_case_ids(
    frozen_assets: Path,
) -> None:
    cases = read_jsonl(frozen_assets.parent / "smoke_cases.jsonl")
    plan = build_execution_plan(cases, 1)
    with pytest.raises(ValueError, match="outside the selected profile"):
        realdata_harness._build_performance_plan(
            plan,
            realdata_harness._PerformanceConfig(True, 1, 3, ("missing",)),
        )
    with pytest.raises(ValueError, match="duplicate"):
        realdata_harness._performance_config(
            {
                "DSPARK_M25A_PERFORMANCE": "1",
                "DSPARK_M25A_PERFORMANCE_CASE_IDS": "synthetic:1024:0,synthetic:1024:0",
            },
            realdata_harness._ForensicTraceConfig(None, False, None),
            1,
        )


def test_verification_telemetry_uses_rejection_output_length_not_scheduler_placeholders() -> None:
    assert realdata_harness._verification_token_telemetry(5, [99]) == (0, 1, 0)
    assert realdata_harness._verification_token_telemetry(5, [10, 20, 99]) == (2, 1, 0)
    assert realdata_harness._verification_token_telemetry(5, [10, 20, 30, 40, 50, 60]) == (5, 0, 1)
    with pytest.raises(ValueError, match="positive"):
        realdata_harness._verification_token_telemetry(0, [99])
    with pytest.raises(ValueError, match="replacement or bonus"):
        realdata_harness._verification_token_telemetry(5, [])
    with pytest.raises(ValueError, match="replacement or bonus"):
        realdata_harness._verification_token_telemetry(5, list(range(7)))


def test_performance_summary_keeps_exact_tokens_non_blocking(
    frozen_assets: Path,
    tmp_path: Path,
) -> None:
    target, dspark = _performance_result_dirs(tmp_path, frozen_assets)

    summary = summarize_performance(
        frozen_assets,
        [("target_only", target), ("dspark", dspark)],
        expected_ranks=2,
        min_runs_per_mode=1,
    )

    assert summary["performance_provisional"] is True
    assert summary["exact_token_cross_mode_blocking"] is False
    assert summary["run_aggregation"] == "single-run aggregate"
    assert summary["speedup"]["all_case_decode"] == 2.0
    assert summary["speedup"]["all_case_inference"] == pytest.approx(5 / 3)
    assert summary["speedup"]["primary_warmup_excluded_decode"] == 2.0
    dspark_run = next(run for run in summary["runs"] if run["mode"] == "dspark")
    assert dspark_run["accepted_candidate_metrics_available"] is True
    assert dspark_run["average_accepted_candidate_tokens_per_verification"] == 4.0
    assert dspark_run["warmup_excluded"]["case_count"] == 9
    diagnostics = summary["cross_mode_exact_token_diagnostics"][0]["cases"]
    assert len(diagnostics) == 10
    assert all(item["exact_token_match"] is False for item in diagnostics)
    assert all(item["first_different_token_index"] == 0 for item in diagnostics)
    csv_path = tmp_path / "summary.csv"
    performance_summary._write_csv(csv_path, summary["runs"])
    csv_rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(csv_rows) == 3
    assert csv_rows[0].startswith("mode,root,profile,case_count")
    case_csv = tmp_path / "cases.csv"
    case_markdown = tmp_path / "cases.md"
    performance_summary._write_case_csv(case_csv, summary["matched_case_performance"])
    performance_summary._write_case_markdown(case_markdown, summary["matched_case_performance"])
    assert len(case_csv.read_text(encoding="utf-8").splitlines()) == 11
    assert len(case_markdown.read_text(encoding="utf-8").splitlines()) == 12
    assert "accepted/ver" in case_markdown.read_text(encoding="utf-8")


def test_performance_summary_marks_legacy_placeholder_acceptance_unavailable(
    frozen_assets: Path,
    tmp_path: Path,
) -> None:
    target, dspark = _performance_result_dirs(tmp_path, frozen_assets)
    for rank in range(2):
        path = dspark / f"rank-{rank}.jsonl"
        records = read_jsonl(path)
        for record in records:
            performance = record["performance"]
            for field in (
                "accepted_candidate_metrics_source",
                "accepted_candidate_tokens_total",
                "average_accepted_candidate_tokens_per_verification",
                "replacement_tokens_total",
                "bonus_tokens_total",
                "committed_tokens_total",
                "verification_committed_tokens_total",
                "effective_committed_tokens_per_verification",
            ):
                performance.pop(field)
            performance["accepted_draft_token_count"] = 0
            performance["average_accepted_tokens_per_verification"] = 0.0
            performance["draft_token_acceptance_rate"] = 0.0
            record["target_forward_count"] = record["output_token_count"]
        _rewrite_results(path, records)

    summary = summarize_performance(
        frozen_assets,
        [("target_only", target), ("dspark", dspark)],
        expected_ranks=2,
        min_runs_per_mode=1,
    )

    dspark_run = next(run for run in summary["runs"] if run["mode"] == "dspark")
    assert dspark_run["accepted_candidate_metrics_available"] is False
    assert dspark_run["accepted_candidate_tokens_total"] is None
    assert dspark_run["average_accepted_candidate_tokens_per_verification"] is None
    assert dspark_run["accepted_candidate_metrics_source"] == ("unavailable_legacy_scheduler_placeholder_comparison")


def test_steady_state_summary_excludes_warmup_and_reports_per_case_statistics(
    frozen_assets: Path,
    tmp_path: Path,
) -> None:
    target, dspark = _steady_state_performance_result_dirs(tmp_path, frozen_assets)

    summary = summarize_performance(
        frozen_assets,
        [("target_only", target), ("dspark", dspark)],
        expected_ranks=2,
        min_runs_per_mode=1,
    )

    assert summary["run_aggregation"] == "per-case steady-state measured repeats"
    assert summary["speedup"]["primary_warmup_excluded_decode"] == pytest.approx(2.0)
    assert summary["performance_stability_gate"] == {
        "applicable": True,
        "passed": True,
        "thresholds": {
            "stable": "CV <= 0.05",
            "annotate": "0.05 < CV <= 0.10",
            "not_formal": "CV > 0.10",
        },
        "metric": "max(decode_latency_cv, inference_latency_cv)",
    }
    target_run = next(run for run in summary["runs"] if run["mode"] == "target_only")
    assert target_run["case_count"] == 9
    assert target_run["steady_state_protocol"]["warmup_repeats"] == 1
    assert target_run["steady_state_protocol"]["measured_repeats"] == 3
    cases = summary["steady_state_case_performance"][0]["cases"]
    assert len(cases) == 3
    first = cases[0]
    assert first["target_only"]["cold_first_use"]["decode_latency_seconds"] == 8.0
    assert [repeat["decode_latency_seconds"] for repeat in first["target_only"]["measured_repeats"]] == [
        1.9,
        2.0,
        2.1,
    ]
    target_decode = first["target_only"]["statistics"]["decode_latency_seconds"]
    assert target_decode["median"] == 2.0
    assert target_decode["mean"] == 2.0
    assert target_decode["min"] == 1.9
    assert target_decode["max"] == 2.1
    assert target_decode["coefficient_of_variation"] == pytest.approx(0.040824829)
    assert first["decode_speedup"] == 2.0
    assert first["dspark"]["average_accepted_candidate_tokens_per_verification"] == 4.0
    assert len({repeat["request_id"] for repeat in first["dspark"]["measured_repeats"]}) == 3

    csv_path = tmp_path / "steady.csv"
    markdown_path = tmp_path / "steady.md"
    performance_summary._write_steady_state_csv(csv_path, summary["steady_state_case_performance"])
    performance_summary._write_steady_state_markdown(markdown_path, summary["steady_state_case_performance"])
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 4
    assert len(markdown_path.read_text(encoding="utf-8").splitlines()) == 5
    assert "target_decode_cv" in csv_path.read_text(encoding="utf-8").splitlines()[0]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records.pop(2), "missing|out of order"),
        (
            lambda records: records[2].update(performance_repeat_index=records[1]["performance_repeat_index"]),
            "missing or duplicated",
        ),
        (lambda records: records.__setitem__(slice(0, 2), reversed(records[0:2])), "out of order"),
    ],
)
def test_steady_state_summary_rejects_missing_duplicate_or_out_of_order_repeats(
    frozen_assets: Path,
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    target, dspark = _steady_state_performance_result_dirs(tmp_path, frozen_assets)
    for rank in range(2):
        path = dspark / f"rank-{rank}.jsonl"
        records = read_jsonl(path)
        mutation(records)
        for sequence, record in enumerate(records):
            record["request_sequence_index"] = sequence
        _rewrite_results(path, records)

    with pytest.raises(ValueError, match=message):
        summarize_performance(
            frozen_assets,
            [("target_only", target), ("dspark", dspark)],
            expected_ranks=2,
            min_runs_per_mode=1,
        )


@pytest.mark.parametrize("record_index", [0, 1], ids=["warmup", "measured"])
def test_steady_state_summary_rejects_tp_rank_repeat_mismatch(
    frozen_assets: Path,
    tmp_path: Path,
    record_index: int,
) -> None:
    target, dspark = _steady_state_performance_result_dirs(tmp_path, frozen_assets)
    path = dspark / "rank-1.jsonl"
    records = read_jsonl(path)
    records[record_index]["performance_repeat_index"] = 9
    _rewrite_results(path, records)

    with pytest.raises(ValueError, match="Rank 1 disagrees"):
        summarize_performance(
            frozen_assets,
            [("target_only", target), ("dspark", dspark)],
            expected_ranks=2,
            min_runs_per_mode=1,
        )


def test_steady_state_summary_rejects_repeat_workload_mismatch(
    frozen_assets: Path,
    tmp_path: Path,
) -> None:
    target, dspark = _steady_state_performance_result_dirs(tmp_path, frozen_assets)
    for rank in range(2):
        path = dspark / f"rank-{rank}.jsonl"
        records = read_jsonl(path)
        records[1]["stop_reason"] = {
            "finish_reason": "length",
            "stop_reason": "repeat-workload-mismatch",
        }
        _rewrite_results(path, records)

    with pytest.raises(ValueError, match="repeats.*differ for stop_reason"):
        summarize_performance(
            frozen_assets,
            [("target_only", target), ("dspark", dspark)],
            expected_ranks=2,
            min_runs_per_mode=1,
        )


def test_steady_state_stability_gate_rejects_high_cv(
    frozen_assets: Path,
    tmp_path: Path,
) -> None:
    target, dspark = _steady_state_performance_result_dirs(tmp_path, frozen_assets)
    for rank in range(2):
        path = dspark / f"rank-{rank}.jsonl"
        records = read_jsonl(path)
        performance = records[1]["performance"]
        performance["decode_latency_seconds"] = 0.1
        performance["inference_latency_seconds"] = performance["prefill_latency_seconds"] + 0.1
        _rewrite_results(path, records)

    summary = summarize_performance(
        frozen_assets,
        [("target_only", target), ("dspark", dspark)],
        expected_ranks=2,
        min_runs_per_mode=1,
    )

    assert summary["performance_stability_gate"]["passed"] is False
    assert summary["steady_state_case_performance"][0]["cases"][0]["dspark"]["stability"] == "not_formal"


def test_performance_summary_rejects_unconsumed_proposal(
    frozen_assets: Path,
    tmp_path: Path,
) -> None:
    target, dspark = _performance_result_dirs(tmp_path, frozen_assets)
    path = dspark / "rank-0.jsonl"
    records = read_jsonl(path)
    records[0]["proposal_consumed_count"] = 0
    _rewrite_results(path, records)

    with pytest.raises(ValueError, match="consumed exactly once|disagrees"):
        summarize_performance(
            frozen_assets,
            [("target_only", target), ("dspark", dspark)],
            expected_ranks=2,
            min_runs_per_mode=1,
        )


def test_exact_token_result_gate_accepts_complete_rank_consistent_artifacts(
    frozen_assets: Path, tmp_path: Path
) -> None:
    target, dspark = _result_dirs(tmp_path, frozen_assets)

    summary = validate_result_pair(frozen_assets, target, dspark, expected_ranks=2)

    assert summary["exact_token_match"] is True
    assert summary["case_executions"] == 10
    assert summary["performance_validated"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records.pop(), "sizes|partial"),
        (lambda records: records.append(deepcopy(records[0])), "duplicate"),
        (lambda records: records.reverse(), "order"),
        (
            lambda records: records[0].update(
                output_token_ids=[999],
                output_token_count=1,
                output_token_sha256=token_ids_sha256([999]),
                usage_completion_tokens=1,
            ),
            "disagrees",
        ),
        (lambda records: records[0].update(stop_reason={"finish_reason": "stop", "stop_reason": 2}), "disagrees"),
        (lambda records: records[0].update(usage_completion_tokens=1), "accounting"),
        (lambda records: records[0].update(cleanup_complete=False), "cleanup"),
    ],
)
def test_exact_token_gate_fails_closed(
    frozen_assets: Path,
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    target, dspark = _result_dirs(tmp_path, frozen_assets)
    path = dspark / "rank-0.jsonl"
    records = read_jsonl(path)
    mutation(records)
    _rewrite_results(path, records)

    with pytest.raises(ValueError, match=message):
        validate_result_pair(frozen_assets, target, dspark, expected_ranks=2)


def test_rank_inconsistency_and_stale_epochs_fail_closed(frozen_assets: Path, tmp_path: Path) -> None:
    target, dspark = _result_dirs(tmp_path, frozen_assets)
    rank_one = dspark / "rank-1.jsonl"
    records = read_jsonl(rank_one)
    records[0]["output_token_ids"] = [7]
    records[0]["output_token_count"] = 1
    records[0]["output_token_sha256"] = token_ids_sha256([7])
    records[0]["usage_completion_tokens"] = 1
    _rewrite_results(rank_one, records)
    with pytest.raises(ValueError, match="disagrees"):
        validate_result_pair(frozen_assets, target, dspark, expected_ranks=2)

    target, dspark = _result_dirs(tmp_path / "stale", frozen_assets)
    for rank in range(2):
        rank_path = dspark / f"rank-{rank}.jsonl"
        records = read_jsonl(rank_path)
        records[1]["proposal_epoch_start"] = records[0]["proposal_epoch_end"]
        _rewrite_results(rank_path, records)
    with pytest.raises(ValueError, match="epochs"):
        validate_result_pair(frozen_assets, target, dspark, expected_ranks=2)


def test_execution_plan_is_three_complete_passes_with_a_b_a_separation(frozen_assets: Path) -> None:
    cases = read_jsonl(frozen_assets.parent / "smoke_cases.jsonl")
    plan = build_execution_plan(cases, 3)

    assert len(plan) == 30
    assert [record["case_id"] for record in plan[:10]] == [record["case_id"] for record in plan[10:20]]
    assert plan[0]["case_id"] == plan[10]["case_id"] == plan[20]["case_id"]
    assert plan[0]["case_id"] != plan[1]["case_id"]
    assert [record["request_sequence_index"] for record in plan] == list(range(30))


def test_forensic_case_filter_is_exact_and_preserves_original_case_identity(
    frozen_assets: Path,
) -> None:
    cases = read_jsonl(frozen_assets.parent / "smoke_cases.jsonl")
    plan = build_execution_plan(cases, 1)

    selected, config = realdata_harness._select_forensic_cases(
        plan,
        {
            "DSPARK_M25A_CASE_ID": "synthetic:1024:0",
            "DSPARK_M25A_TRACE_FIRST_ROUND": "1",
        },
    )

    assert [case["case_id"] for case in selected] == ["synthetic:1024:0"]
    assert selected[0]["request_sequence_index"] == 7
    assert config.case_id == "synthetic:1024:0"
    assert config.first_round is True
    assert config.output_index is None
    assert plan[7] is selected[0]


def test_output_index_trace_requires_exact_case_and_parses_non_negative_index(
    frozen_assets: Path,
) -> None:
    cases = read_jsonl(frozen_assets.parent / "smoke_cases.jsonl")
    plan = build_execution_plan(cases, 1)

    selected, config = realdata_harness._select_forensic_cases(
        plan,
        {
            "DSPARK_M25A_CASE_ID": "synthetic:1024:0",
            "DSPARK_M25A_TRACE_OUTPUT_INDEX": "6",
        },
    )

    assert [case["case_id"] for case in selected] == ["synthetic:1024:0"]
    assert config.case_id == "synthetic:1024:0"
    assert config.first_round is False
    assert config.output_index == 6

    with pytest.raises(ValueError, match="requires an exact DSPARK_M25A_CASE_ID"):
        realdata_harness._select_forensic_cases(
            plan,
            {"DSPARK_M25A_TRACE_OUTPUT_INDEX": "6"},
        )


@pytest.mark.parametrize(
    ("output_length_before", "committed_tokens", "output_index", "offset"),
    [
        (6, [9045], 6, 0),
        (4, [10, 11, 12, 13], 6, 2),
        (4, [10, 11, 12, 13], 4, 0),
        (4, [10, 11, 12, 13], 7, 3),
    ],
)
def test_output_index_trace_locates_single_and_multi_token_commit_boundaries(
    output_length_before: int,
    committed_tokens: list[int],
    output_index: int,
    offset: int,
) -> None:
    trace = realdata_harness._commit_output_index_trace(
        output_length_before,
        committed_tokens,
        output_index,
    )

    assert trace == {
        "commit_start_output_index": output_length_before,
        "commit_end_output_index_exclusive": output_length_before + len(committed_tokens),
        "commit_covers_traced_output_index": True,
        "traced_commit_offset": offset,
        "traced_committed_token": committed_tokens[offset],
    }


def test_output_index_trace_retries_when_rejection_shortens_commit() -> None:
    trace = realdata_harness._commit_output_index_trace(4, [10, 11], 6)

    assert trace["commit_covers_traced_output_index"] is False
    assert trace["traced_commit_offset"] is None
    assert trace["traced_committed_token"] is None


def test_output_index_trace_preserves_replacement_and_bonus_contracts() -> None:
    replacement = realdata_harness._expected_greedy_verification(
        [16, 223, 5769, 22, 28],
        [16, 88338, 7, 8, 9, 10],
    )
    bonus = realdata_harness._expected_greedy_verification(
        [16, 223],
        [16, 223, 9045],
    )

    assert replacement == ([16, 88338], 1, True, False)
    assert bonus == ([16, 223, 9045], 2, False, True)
    assert realdata_harness._commit_output_index_trace(5, replacement[0], 6)["traced_committed_token"] == 88338
    assert realdata_harness._commit_output_index_trace(4, bonus[0], 6)["traced_committed_token"] == 9045


def test_output_index_trace_observes_next_input_and_runner_prefix_identity() -> None:
    prompt_tokens = [101, 102]
    output_tokens = [10, 11, 9045, 13, 14]
    input_batch = SimpleNamespace(
        query_start_loc=[0, 2],
        num_reqs=1,
        input_ids=[9045, 223],
        positions=[4, 5],
        num_tokens=2,
        num_draft_tokens=1,
        num_computed_tokens_np=[4],
    )
    req_states = SimpleNamespace(
        req_id_to_index={"request": 0},
        all_token_ids=SimpleNamespace(gpu=_TraceTokenMatrix([[*prompt_tokens, *output_tokens]])),
        num_computed_tokens_np=[4],
    )
    runtime = SimpleNamespace(
        runner=SimpleNamespace(
            execute_model_state=SimpleNamespace(input_batch=input_batch),
            req_states=req_states,
        ),
        torch=SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)),
    )
    request = SimpleNamespace(
        prompt_token_ids=prompt_tokens,
        output_token_ids=output_tokens,
    )

    trace = realdata_harness._next_model_input_trace(
        runtime,
        request,
        "request",
        output_index=2,
        traced_token=9045,
    )

    assert trace["next_model_input_ids"] == [9045, 223]
    assert trace["next_model_positions"] == [4, 5]
    assert trace["next_model_input_contains_traced_token"] is True
    assert trace["next_runner_traced_token"] == 9045
    assert trace["next_runner_contains_traced_token_at_output_index"] is True
    assert trace["next_runner_output_window"] == output_tokens
    assert trace["next_runner_window_matches_request"] is True


def test_trace_host_serialization_keeps_numpy_on_host() -> None:
    value = _NumpyNdarrayTraceValue([3, 5])
    synchronize_calls = 0

    def synchronize() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1

    serialized = realdata_harness._host_json_value(
        value,
        synchronize_tensor=synchronize,
    )

    assert serialized == [3, 5]
    assert synchronize_calls == 0
    assert not hasattr(value, "detach")
    assert not hasattr(value, "cpu")
    assert value.tolist_calls == 1


def test_trace_host_serialization_uses_tensor_detach_cpu_boundary() -> None:
    value = _TensorLikeTraceValue([7, 11])
    synchronize_calls = 0

    def synchronize() -> None:
        nonlocal synchronize_calls
        synchronize_calls += 1

    serialized = realdata_harness._host_json_value(
        value,
        synchronize_tensor=synchronize,
    )

    assert serialized == [7, 11]
    assert synchronize_calls == 1
    assert value.detach_calls == 1
    assert value.cpu_calls == 1
    assert value.tolist_calls == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (17, 17),
        (True, True),
        ([1, _NumpyNdarrayTraceValue(2)], [1, 2]),
        (("producer", 1), ["producer", 1]),
        ({"tokens": _NumpyNdarrayTraceValue([13])}, {"tokens": [13]}),
    ],
)
def test_trace_host_serialization_accepts_python_values(
    value: Any,
    expected: Any,
) -> None:
    serialized = realdata_harness._host_json_value(value)

    assert serialized == expected
    assert json.loads(json.dumps(serialized)) == expected


def test_trace_marker_payload_is_json_serializable(capsys: pytest.CaptureFixture[str]) -> None:
    payload = realdata_harness._host_json_value(
        {
            "scheduled": _NumpyNdarrayTraceValue([1]),
            "tokens": _TensorLikeTraceValue([23]),
            "step": "producer",
        }
    )

    realdata_harness._marker(realdata_harness.FIRST_ROUND_TRACE, payload)

    marker = capsys.readouterr().out.strip()
    assert marker.startswith(realdata_harness.FIRST_ROUND_TRACE + "=")
    assert json.loads(marker.partition("=")[2]) == {
        "scheduled": [1],
        "step": "producer",
        "tokens": [23],
    }


def test_forensic_trace_is_disabled_without_an_explicit_case_filter(
    frozen_assets: Path,
) -> None:
    cases = read_jsonl(frozen_assets.parent / "smoke_cases.jsonl")
    plan = build_execution_plan(cases, 1)

    selected, config = realdata_harness._select_forensic_cases(plan, {})

    assert selected is plan
    assert config.case_id is None
    assert config.first_round is False
    assert config.output_index is None
    with pytest.raises(ValueError, match="requires an exact DSPARK_M25A_CASE_ID"):
        realdata_harness._select_forensic_cases(
            plan,
            {"DSPARK_M25A_TRACE_FIRST_ROUND": "1"},
        )


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        (
            {"DSPARK_M25A_TRACE_FIRST_ROUND": "true"},
            "must be 0 or 1",
        ),
        (
            {"DSPARK_M25A_CASE_ID": "missing:case"},
            "is not in the selected profile",
        ),
        (
            {
                "DSPARK_M25A_CASE_ID": "synthetic:1024:0",
                "DSPARK_M25A_TRACE_OUTPUT_INDEX": "not-an-index",
            },
            "must be a non-negative integer",
        ),
        (
            {
                "DSPARK_M25A_CASE_ID": "synthetic:1024:0",
                "DSPARK_M25A_TRACE_OUTPUT_INDEX": "-1",
            },
            "must be non-negative",
        ),
    ],
)
def test_forensic_case_filter_rejects_ambiguous_or_unknown_configuration(
    frozen_assets: Path,
    environ: dict[str, str],
    message: str,
) -> None:
    cases = read_jsonl(frozen_assets.parent / "smoke_cases.jsonl")
    plan = build_execution_plan(cases, 1)

    with pytest.raises(ValueError, match=message):
        realdata_harness._select_forensic_cases(plan, environ)


def test_long_output_token_artifact_is_not_truncated(frozen_assets: Path, tmp_path: Path) -> None:
    target, _ = _result_dirs(tmp_path, frozen_assets)
    path = target / "rank-0.jsonl"
    records = read_jsonl(path)
    output_ids = list(range(1024))
    records[0].update(
        output_cap=1024,
        output_token_ids=output_ids,
        output_token_count=1024,
        output_token_sha256=token_ids_sha256(output_ids),
        usage_completion_tokens=1024,
    )
    _rewrite_results(path, records)

    assert len(read_jsonl(path)[0]["output_token_ids"]) == 1024
