# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import summarize_dspark_acceptance_benchmark as summary


class _Counter:
    def __init__(self, name: str, value: int, labels: dict[str, str] | None = None):
        self.name = name
        self.value = value
        self.labels = labels or {"engine": "0"}


class _Vector:
    def __init__(self, values: list[int], labels: dict[str, str] | None = None):
        self.name = benchmark.VECTOR_METRIC_NAME
        self.values = values
        self.labels = labels or {"engine": "0"}


def _metrics(drafts: int, draft_tokens: int, accepted: int, positions: list[int]) -> list[Any]:
    return [
        _Counter("vllm:spec_decode_num_drafts", drafts),
        _Counter("vllm:spec_decode_num_draft_tokens", draft_tokens),
        _Counter("vllm:spec_decode_num_accepted_tokens", accepted),
        _Vector(positions),
    ]


class _Completion:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids
        self.finish_reason = "length"
        self.stop_reason = None


class _Output:
    def __init__(self, prompt_token_ids: list[int], token_ids: list[int]):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [_Completion(token_ids)]


class _Tokenizer:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, prompt: str) -> list[int]:
        self.encoded.append(prompt)
        return [0, len(prompt), 1]


class _FakeEngine:
    def __init__(self, args: argparse.Namespace):
        speculative = (
            SimpleNamespace(method="dspark", num_speculative_tokens=args.num_spec_tokens)
            if args.mode == "dspark"
            else None
        )
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                use_v2_model_runner=True,
                model_config=SimpleNamespace(
                    enforce_eager=args.enforce_eager,
                    max_model_len=args.max_model_len,
                    dtype="torch.bfloat16",
                    quantization=args.quantization,
                ),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=args.tensor_parallel_size,
                    pipeline_parallel_size=1,
                    enable_expert_parallel=args.enable_expert_parallel,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=args.async_scheduling,
                    max_num_seqs=args.max_num_seqs,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                ),
                cache_config=SimpleNamespace(block_size=args.block_size, enable_prefix_caching=False),
                speculative_config=speculative,
            ),
            shutdown=self._shutdown,
        )
        self.tokenizer = _Tokenizer()
        self.generate_calls: list[list[dict[str, list[int]]]] = []
        self.shutdown_called = False
        self.metric_call = 0
        self.mode = args.mode

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer

    def generate(self, prompts: list[dict[str, list[int]]], _params: object, *, use_tqdm: bool) -> list[_Output]:
        assert use_tqdm is False
        self.generate_calls.append(copy.deepcopy(prompts))
        return [
            _Output(prompt["prompt_token_ids"], [request_index + 10] * 4)
            for request_index, prompt in enumerate(prompts)
        ]

    def get_metrics(self) -> list[Any]:
        self.metric_call += 1
        if self.mode == "target_only":
            return []
        if self.metric_call == 1:
            return _metrics(2, 10, 7, [2, 2, 1, 1, 1])
        return _metrics(5, 23, 16, [5, 5, 3, 2, 1])

    def _shutdown(self) -> None:
        self.shutdown_called = True


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]}),
        encoding="utf-8",
    )
    (model / "quant_model_description.json").write_text("{}\n", encoding="utf-8")
    return model


def _dataset(tmp_path: Path, count: int = 2) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"question": f"prompt-{index}"}) + "\n" for index in range(count)),
        encoding="utf-8",
    )
    return path


def _args(tmp_path: Path, mode: str = "dspark", count: int = 2) -> argparse.Namespace:
    return benchmark.parse_args(
        [
            "--model-dir",
            str(_model_dir(tmp_path)),
            "--mode",
            mode,
            "--dataset-name",
            "jsonl",
            "--dataset-path",
            str(_dataset(tmp_path, count)),
            "--num-prompts",
            str(count),
            "--result-json",
            str(tmp_path / f"{mode}.json"),
        ]
    )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> tuple[dict[str, Any], _FakeEngine]:
    args = _args(tmp_path, mode)
    engine = _FakeEngine(args)
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)
    times = iter((10.0, 12.0))
    result = benchmark.run_benchmark(
        args,
        engine_factory=lambda _kwargs: engine,
        sampling_factory=lambda _args: object(),
        clock=lambda: next(times),
        plugin_root=tmp_path,
        core_root=tmp_path,
    )
    return result, engine


def test_cli_defaults_force_pr_style_mrv2_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)

    assert args.num_spec_tokens == 5
    assert args.temperature == 0
    assert args.top_p == 1
    assert args.top_k == -1
    assert args.output_len == 256
    assert args.tensor_parallel_size == 8
    assert args.max_model_len == 8192
    assert args.enforce_eager is True
    assert args.enable_expert_parallel is True
    assert args.async_scheduling is True


def test_engine_kwargs_select_target_only_or_exact_dspark_config(tmp_path: Path) -> None:
    target = benchmark.build_engine_kwargs(_args(tmp_path, "target_only"))
    dspark_tmp = tmp_path / "dspark"
    dspark_tmp.mkdir()
    dspark = benchmark.build_engine_kwargs(_args(dspark_tmp, "dspark"))

    assert target["speculative_config"] is None
    assert dspark["speculative_config"] == {"method": "dspark", "num_speculative_tokens": 5}
    assert dspark["tensor_parallel_size"] == 8
    assert dspark["enable_expert_parallel"] is True
    assert dspark["async_scheduling"] is True
    assert dspark["disable_log_stats"] is False


def test_mrv2_environment_is_set_before_dynamic_vllm_import(monkeypatch: pytest.MonkeyPatch) -> None:
    environ: dict[str, str] = {}
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    benchmark._require_mrv2_environment(environ)

    assert environ == {"VLLM_USE_V2_MODEL_RUNNER": "1"}


def test_jsonl_and_hf_prompt_loading_are_deterministic(tmp_path: Path) -> None:
    args = _args(tmp_path)
    jsonl_sources, jsonl_descriptor = benchmark.load_prompt_sources(args)
    hf_args = copy.copy(args)
    hf_args.dataset_name = "hf"
    hf_args.dataset_path = "openai/gsm8k"
    hf_args.dataset_revision = benchmark.DEFAULT_GSM8K_REVISION
    seen: dict[str, Any] = {}

    def loader(path: str, **kwargs: Any) -> list[dict[str, str]]:
        seen.update(path=path, **kwargs)
        return [{"question": "a"}, {"question": "b"}, {"question": "c"}]

    hf_sources, hf_descriptor = benchmark.load_prompt_sources(hf_args, hf_loader=loader)

    assert [source.source_index for source in jsonl_sources] == [0, 1]
    assert [source.value for source in hf_sources] == ["a", "b"]
    assert seen == {
        "path": "openai/gsm8k",
        "split": "test",
        "revision": benchmark.DEFAULT_GSM8K_REVISION,
        "name": "main",
    }
    assert jsonl_descriptor["file_sha256"]
    assert hf_descriptor["selection"] == "source_order_first_n"


def test_dataset_never_cycles_insufficient_prompts(tmp_path: Path) -> None:
    args = _args(tmp_path, count=2)
    args.num_prompts = 3

    with pytest.raises(ValueError, match="never cycled"):
        benchmark.load_prompt_sources(args)


def test_warmup_prompt_count_cannot_be_silently_clamped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path, count=1)
    args.warmup_prompts = 2
    engine = _FakeEngine(args)
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)

    with pytest.raises(ValueError, match="cannot exceed"):
        benchmark.run_benchmark(
            args,
            engine_factory=lambda _kwargs: engine,
            sampling_factory=lambda _args: object(),
            plugin_root=tmp_path,
            core_root=tmp_path,
        )

    assert engine.shutdown_called is True


def test_hf_source_requires_revision(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.dataset_name = "hf"
    args.dataset_revision = None

    with pytest.raises(ValueError, match="dataset-revision"):
        benchmark.load_prompt_sources(args, hf_loader=lambda *_args, **_kwargs: [])


def test_tokenization_precomputes_stable_prompt_fingerprint() -> None:
    sources = [
        benchmark.PromptSource(0, "abc", "a" * 64),
        benchmark.PromptSource(1, [1, 2, 3], "b" * 64),
    ]
    tokenizer = _Tokenizer()

    first = benchmark.tokenize_prompt_sources(sources, tokenizer)
    second = benchmark.tokenize_prompt_sources(sources, _Tokenizer())

    assert first == second
    assert first[0] == [{"prompt_token_ids": [0, 3, 1]}, {"prompt_token_ids": [1, 2, 3]}]
    assert tokenizer.encoded == ["abc"]


def test_counter_vector_snapshot_delta_excludes_warmup() -> None:
    start = benchmark.capture_spec_metrics(_metrics(2, 10, 7, [2, 2, 1, 1, 1]))
    end = benchmark.capture_spec_metrics(_metrics(5, 23, 16, [5, 5, 3, 2, 1]))

    delta = benchmark.metric_snapshot_delta(start, end, 5)
    acceptance = benchmark.acceptance_from_delta(delta, 5)

    assert delta["totals"] == {
        "vllm:spec_decode_num_drafts": 3,
        "vllm:spec_decode_num_draft_tokens": 13,
        "vllm:spec_decode_num_accepted_tokens": 9,
        "vllm:spec_decode_num_accepted_tokens_per_pos": [3, 3, 2, 1, 0],
    }
    assert acceptance["accepted_candidate_tokens_per_verification"] == 3
    assert acceptance["effective_acceptance_length"] == 4
    assert acceptance["acceptance_per_position"] == [1, 1, 2 / 3, 1 / 3, 0]


def test_pr_12968_example_metric_semantics() -> None:
    drafts = 15588
    accepted = 51685

    assert accepted / drafts == pytest.approx(3.315691557608417)
    assert 1 + accepted / drafts == pytest.approx(4.315691557608417)


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        (_metrics(2, 10, 7, [2, 2, 1, 1, 1]), [], "missing or changed"),
        (_metrics(2, 10, 7, [2, 2, 1, 1, 1]), _metrics(1, 10, 7, [2, 2, 1, 1, 1]), "decreased"),
    ],
)
def test_metric_missing_or_counter_reset_fails_closed(start: list[Any], end: list[Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        benchmark.metric_snapshot_delta(
            benchmark.capture_spec_metrics(start),
            benchmark.capture_spec_metrics(end),
            5,
        )


@pytest.mark.parametrize(
    ("positions", "accepted", "match"),
    [
        ([3, 2, 3, 1, 0], 9, "non-increasing"),
        ([3, 3, 2, 1, 0], 8, "vector sum"),
    ],
)
def test_invalid_position_metrics_fail_closed(positions: list[int], accepted: int, match: str) -> None:
    start = benchmark.capture_spec_metrics(_metrics(0, 0, 0, [0, 0, 0, 0, 0]))
    end = benchmark.capture_spec_metrics(_metrics(3, 15, accepted, positions))

    with pytest.raises(ValueError, match=match):
        benchmark.acceptance_from_delta(benchmark.metric_snapshot_delta(start, end, 5), 5)


def test_metric_type_and_duplicate_series_are_strict() -> None:
    bad = SimpleNamespace(
        name="vllm:spec_decode_num_drafts",
        labels={"engine": "0"},
        value=1.5,
    )
    with pytest.raises(TypeError, match="Counter"):
        benchmark.capture_spec_metrics([bad])
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.capture_spec_metrics(
            [
                _Counter("vllm:spec_decode_num_drafts", 1),
                _Counter("vllm:spec_decode_num_drafts", 2),
            ]
        )


def test_metric_snapshot_sums_distinct_labeled_series() -> None:
    metrics = benchmark.capture_spec_metrics(
        [
            _Counter("vllm:spec_decode_num_drafts", 2, {"engine": "0"}),
            _Counter("vllm:spec_decode_num_drafts", 3, {"engine": "1"}),
        ]
    )

    assert [series["value"] for series in metrics["vllm:spec_decode_num_drafts"]] == [2, 3]


def test_effective_config_rejects_non_mrv2_engine(tmp_path: Path) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    engine.llm_engine.vllm_config.use_v2_model_runner = False

    with pytest.raises(RuntimeError, match="MRV2"):
        benchmark._effective_engine_config(engine, args)


def test_run_benchmark_batches_measured_prompts_and_excludes_load_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, engine = _run(tmp_path, monkeypatch, "dspark")

    assert [len(call) for call in engine.generate_calls] == [1, 2]
    assert result["timing"]["elapsed_seconds"] == 2
    assert result["throughput"]["total_output_tokens"] == 8
    assert result["throughput"]["output_tokens_per_second"] == 4
    assert result["throughput"]["requests_per_second"] == 1
    assert result["metrics"]["delta_boundary"] == "post_warmup_snapshot_to_post_measured_snapshot"
    assert result["acceptance"]["num_drafts"] == 3
    assert result["cleanup"]["engine_shutdown_complete"] is True
    assert engine.shutdown_called is True
    json.dumps(result)


def test_target_only_records_speculative_metrics_as_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _engine = _run(tmp_path, monkeypatch, "target_only")

    assert result["effective_config"]["speculative_config"] is None
    assert result["metrics"]["measured_delta"] is None
    assert result["acceptance"]["num_drafts"] is None
    assert result["acceptance"]["acceptance_per_position"] == [None] * 5


def _independent_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    dsparks: list[dict[str, Any]] = []
    for index in range(3):
        target_root = tmp_path / f"target-{index}"
        dspark_root = tmp_path / f"dspark-{index}"
        target_root.mkdir(parents=True)
        dspark_root.mkdir(parents=True)
        target, _ = _run(target_root, monkeypatch, "target_only")
        dspark, _ = _run(dspark_root, monkeypatch, "dspark")
        # Normalize source paths so only the output throughput differs.
        dspark["comparison_config"] = copy.deepcopy(target["comparison_config"])
        dspark["comparison_config_fingerprint"] = target["comparison_config_fingerprint"]
        targets.append(target)
        dsparks.append(dspark)
    shared_comparison = copy.deepcopy(targets[0]["comparison_config"])
    shared_fingerprint = benchmark._sha256_bytes(benchmark._canonical_json_bytes(shared_comparison))
    for result in [*targets, *dsparks]:
        result["comparison_config"] = copy.deepcopy(shared_comparison)
        result["comparison_config_fingerprint"] = shared_fingerprint
    for index, result in enumerate(targets):
        result["timing"]["elapsed_seconds"] = (4.0, 2.0, 1.0)[index]
        result["throughput"]["output_tokens_per_second"] = (2.0, 4.0, 8.0)[index]
        result["throughput"]["requests_per_second"] = (0.5, 1.0, 2.0)[index]
        result["throughput"]["milliseconds_per_output_token"] = (500.0, 250.0, 125.0)[index]
    for index, result in enumerate(dsparks):
        result["timing"]["elapsed_seconds"] = (2.0, 1.0, 0.5)[index]
        result["throughput"]["output_tokens_per_second"] = (4.0, 8.0, 16.0)[index]
        result["throughput"]["requests_per_second"] = (1.0, 2.0, 4.0)[index]
        result["throughput"]["milliseconds_per_output_token"] = (250.0, 125.0, 62.5)[index]
    return targets, dsparks


def test_three_run_summary_uses_independent_median_cv_and_nonblocking_token_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch)
    dsparks[0]["outputs"][0]["output_token_sha256"] = "f" * 64

    result = summary.summarize_results(targets, dsparks)

    assert result["target_only"]["output_tokens_per_second"]["median"] == 4
    assert result["dspark"]["output_tokens_per_second"]["median"] == 8
    assert result["dspark_over_target_output_throughput_speedup"] == 2
    assert result["target_only"]["output_tokens_per_second"]["population_coefficient_of_variation"] > 0
    assert result["exact_token_comparison"]["blocking"] is False
    assert result["exact_token_comparison"]["mismatched_request_pair_count"] > 0


def test_run_csv_preserves_absolute_throughput_and_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch)
    path = tmp_path / "runs.csv"

    summary._write_run_csv(path, [targets[0], dsparks[0]])
    rows = path.read_text(encoding="utf-8").splitlines()

    assert len(rows) == 3
    assert "output_tokens_per_second" in rows[0]
    assert "accepted_candidate_tokens_per_verification" in rows[0]
    assert rows[1].startswith("target_only,")
    assert rows[2].startswith("dspark,")


def test_summary_rejects_prompt_config_mismatch_and_duplicate_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch)
    dsparks[0]["comparison_config"]["prompt_set_sha256"] = "b" * 64
    dsparks[0]["comparison_config_fingerprint"] = benchmark._sha256_bytes(
        benchmark._canonical_json_bytes(dsparks[0]["comparison_config"])
    )
    with pytest.raises(ValueError, match="configuration differs"):
        summary.summarize_results(targets, dsparks)

    targets, dsparks = _independent_results(tmp_path / "again", monkeypatch)
    dsparks[0]["run_id"] = targets[0]["run_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        summary.summarize_results(targets, dsparks)


def test_summary_rejects_output_length_or_stop_reason_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch)
    dsparks[0]["outputs"][0]["output_token_count"] = 3
    dsparks[0]["throughput"]["total_output_tokens"] = 7
    dsparks[0]["throughput"]["output_tokens_per_second"] = 3.5

    with pytest.raises(ValueError, match="output count, or stop reason"):
        summary.summarize_results(targets, dsparks)


def test_summary_revalidates_warmup_metric_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch)
    dsparks[0]["metrics"]["measured_delta"]["totals"]["vllm:spec_decode_num_drafts"] = 99

    with pytest.raises(ValueError, match="differs from its snapshots"):
        summary.summarize_results(targets, dsparks)


def test_malformed_result_json_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read"):
        summary._read_result(path)


def test_public_generate_path_has_no_outer_torchrun_or_single_request_loop() -> None:
    source = Path(benchmark.__file__).read_text(encoding="utf-8")

    assert "torchrun" not in source
    assert "_run_case" not in source
    assert "engine.generate(prompts, sampling_params, use_tqdm=False)" in source
    assert "time.perf_counter" in source


def test_cli_help_is_dependency_free() -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(benchmark.__file__)), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--num-spec-tokens" in completed.stdout
    assert "--async-scheduling" in completed.stdout
