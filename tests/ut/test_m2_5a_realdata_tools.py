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

    def execute_model(self, scheduler_output: Any) -> None:
        self.outputs.append(scheduler_output)


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
