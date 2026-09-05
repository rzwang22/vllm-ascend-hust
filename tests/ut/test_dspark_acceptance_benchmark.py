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

from tests.ut.test_dspark_graph_replay import batch, replay_worker
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


class _FakeEngineCore:
    def __init__(self, block_sizes: list[int]):
        self.block_sizes = block_sizes
        self.utility_calls: list[str] = []

    def call_utility(self, method: str) -> list[dict[str, Any]]:
        self.utility_calls.append(method)
        assert method == "get_kv_cache_group_metadata"
        return [
            {
                "group_idx": index,
                "kind": "full_attention",
                "block_size": block_size,
                "sliding_window": None,
            }
            for index, block_size in enumerate(self.block_sizes)
        ]


class _FakeEngine:
    def __init__(self, args: argparse.Namespace):
        graph_enabled = args.target_execution_mode == "full_decode_only"
        configured_capture_sizes = args.cudagraph_capture_sizes or ([6, 24] if graph_enabled else [1])
        speculative = (
            SimpleNamespace(
                method="dspark",
                num_speculative_tokens=args.num_spec_tokens,
                enforce_eager=True,
            )
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
                    hf_config=SimpleNamespace(model_type="uniform_test"),
                ),
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=args.tensor_parallel_size,
                    pipeline_parallel_size=1,
                    enable_expert_parallel=args.enable_expert_parallel,
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=1,
                ),
                scheduler_config=SimpleNamespace(
                    async_scheduling=args.async_scheduling,
                    max_num_seqs=args.max_num_seqs,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                ),
                cache_config=SimpleNamespace(block_size=args.block_size, enable_prefix_caching=False),
                speculative_config=speculative,
                compilation_config=SimpleNamespace(
                    cudagraph_mode="FULL_DECODE_ONLY" if graph_enabled else "NONE",
                    cudagraph_capture_sizes=configured_capture_sizes,
                ),
                observability_config=SimpleNamespace(cudagraph_metrics=graph_enabled),
                additional_config={
                    "ascend_compilation_config": {
                        "enable_npugraph_ex": graph_enabled,
                        "enable_static_kernel": False,
                    }
                },
            ),
            engine_core=_FakeEngineCore([args.block_size]),
            model_executor=None,
            logger_manager=SimpleNamespace(stat_loggers=[]),
            shutdown=self._shutdown,
        )
        self.tokenizer = _Tokenizer()
        self.generate_calls: list[list[dict[str, list[int]]]] = []
        self.shutdown_called = False
        self.metric_call = 0
        self.mode = args.mode
        self.tensor_parallel_size = args.tensor_parallel_size
        self.graph_enabled = graph_enabled
        self.graph_runtime_mode = "FULL"
        self.emit_graph_metrics = False  # Frozen MRV2 has no scheduler graph-stat producer.
        self.replay_workers = [replay_worker(rank) for rank in range(args.tensor_parallel_size)]
        self.rpc_calls = []
        self.graph_capture_count = len(configured_capture_sizes) if graph_enabled else 0
        self.configured_capture_sizes = list(configured_capture_sizes)
        self.observed_capture_sizes = list(configured_capture_sizes) if graph_enabled else []

    def get_tokenizer(self) -> _Tokenizer:
        return self.tokenizer

    def generate(self, prompts: list[dict[str, list[int]]], _params: object, *, use_tqdm: bool) -> list[_Output]:
        assert use_tqdm is False
        self.generate_calls.append(copy.deepcopy(prompts))
        if self.graph_enabled and self.emit_graph_metrics:
            graph_stats = SimpleNamespace(
                runtime_mode=self.graph_runtime_mode,
                num_unpadded_tokens=len(prompts) * (6 if self.mode == "dspark" else 1),
                num_padded_tokens=len(prompts) * (6 if self.mode == "dspark" else 1),
                num_paddings=0,
            )
            scheduler_stats = SimpleNamespace(cudagraph_stats=graph_stats)
            for stat_logger in self.llm_engine.logger_manager.stat_loggers:
                stat_logger.record(scheduler_stats, None)
        if self.graph_enabled:
            for worker in self.replay_workers:
                tokens = len(prompts) * (6 if self.mode == "dspark" else 1)
                worker.model_runner.execute_model(batch(tokens, tokens, self.graph_runtime_mode))
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

    def collective_rpc(self, method: str) -> list[dict[str, Any]]:
        assert type(method) is str
        self.rpc_calls.append(method)
        if method == benchmark._REPLAY_SNAPSHOT_METHOD:
            return [worker.dspark_benchmark_replay_snapshot() for worker in self.replay_workers]
        assert method == benchmark._GRAPH_SNAPSHOT_METHOD
        speculator_requested = "FULL_DECODE_ONLY" if self.mode == "dspark" else None
        speculator_effective = "NONE" if self.mode == "dspark" else None
        return [
            {
                "rank": rank,
                "target_cudagraph_mode": (self.llm_engine.vllm_config.compilation_config.cudagraph_mode),
                "configured_capture_sizes": list(self.configured_capture_sizes),
                "observed_capture_sizes": list(self.observed_capture_sizes),
                "graph_capture_count": self.graph_capture_count,
                "npugraph_ex_enabled": self.graph_enabled,
                "static_kernel_enabled": False,
                "dspark_requested_cudagraph_mode": speculator_requested,
                "dspark_cudagraph_mode": speculator_effective,
            }
            for rank in range(self.tensor_parallel_size)
        ]


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(
        json.dumps({"model_type": "uniform_test", "architectures": ["UniformTestForCausalLM"]}),
        encoding="utf-8",
    )
    (model / "quant_model_description.json").write_text("{}\n", encoding="utf-8")
    return model


def _set_model_type(model_dir: Path, model_type: str, architecture: str) -> None:
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [architecture]}),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path, count: int = 2) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"question": f"prompt-{index}"}) + "\n" for index in range(count)),
        encoding="utf-8",
    )
    return path


def _argv(tmp_path: Path, mode: str = "dspark", count: int = 2) -> list[str]:
    return [
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


def _args(tmp_path: Path, mode: str = "dspark", count: int = 2) -> argparse.Namespace:
    return benchmark.parse_args(_argv(tmp_path, mode, count))


def _graph_args(tmp_path: Path, mode: str = "dspark", count: int = 2) -> argparse.Namespace:
    return benchmark.parse_args([*_argv(tmp_path, mode, count), "--target-execution-mode", "full_decode_only"])


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


def _run_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    configure: Any = None,
) -> tuple[dict[str, Any], _FakeEngine]:
    args = _graph_args(tmp_path, mode)
    engine = _FakeEngine(args)
    if configure is not None:
        configure(engine)
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)
    result = benchmark.run_benchmark(
        args,
        engine_factory=lambda _kwargs: engine,
        sampling_factory=lambda _args: object(),
        clock=iter((10.0, 12.0)).__next__,
        plugin_root=tmp_path,
        core_root=tmp_path,
    )
    return result, engine


_DSV4_BLOCK_SIZES = {
    32: [[32, 32, 2, 8], [4160, 32768]],
    64: [[64, 64, 4, 16], [8320, 65536]],
    128: [[128, 128, 8, 32], [16640, 131072]],
}


def _configure_dsv4(
    monkeypatch: pytest.MonkeyPatch,
    engine: _FakeEngine,
    args: argparse.Namespace,
    *,
    group_block_sizes: list[int] | None = None,
    frontend_ready_block_size: int | None = None,
) -> list[int]:
    mapped = list(_DSV4_BLOCK_SIZES[args.block_size][0])
    groups = mapped if group_block_sizes is None else group_block_sizes
    engine.llm_engine.vllm_config.model_config.hf_config.model_type = "deepseek_v4"
    engine.llm_engine.vllm_config.cache_config.block_size = (
        min(groups) if frontend_ready_block_size is None else frontend_ready_block_size
    )
    engine.llm_engine.engine_core.block_sizes = groups
    monkeypatch.setattr(benchmark, "_runtime_dsv4_block_sizes", lambda: _DSV4_BLOCK_SIZES)
    return mapped


def test_cli_defaults_force_pr_style_mrv2_contract(tmp_path: Path) -> None:
    args = _args(tmp_path)

    assert args.num_spec_tokens == 5
    assert args.temperature == 0
    assert args.top_p == 1
    assert args.top_k == -1
    assert args.output_len == 256
    assert args.tensor_parallel_size == 8
    assert args.max_model_len == 8192
    assert args.block_size == 32
    assert args.target_execution_mode == "eager"
    assert args.enforce_eager is True
    assert args.enable_expert_parallel is True
    assert args.async_scheduling is True


def test_engine_kwargs_select_target_only_or_exact_dspark_config(tmp_path: Path) -> None:
    target = benchmark.build_engine_kwargs(_args(tmp_path, "target_only"))
    dspark_tmp = tmp_path / "dspark"
    dspark_tmp.mkdir()
    dspark = benchmark.build_engine_kwargs(_args(dspark_tmp, "dspark"))

    assert target["speculative_config"] is None
    assert dspark["speculative_config"] == {
        "method": "dspark",
        "num_speculative_tokens": 5,
        "enforce_eager": True,
    }
    assert target["enforce_eager"] is True
    assert "compilation_config" not in target
    assert "worker_extension_cls" not in target and "worker_extension_cls" not in dspark
    assert dspark["tensor_parallel_size"] == 8
    assert dspark["enable_expert_parallel"] is True
    assert dspark["async_scheduling"] is True
    assert dspark["disable_log_stats"] is False


@pytest.mark.parametrize("mode", ["target_only", "dspark"])
def test_full_decode_only_configures_target_graph_and_eager_draft(tmp_path: Path, mode: str) -> None:
    args = _graph_args(tmp_path, mode)

    kwargs = benchmark.build_engine_kwargs(args)

    assert args.enforce_eager is False
    assert "enforce_eager" not in kwargs
    assert kwargs["compilation_config"] == {"cudagraph_mode": "FULL_DECODE_ONLY"}
    assert kwargs["cudagraph_metrics"] is True
    assert kwargs["worker_extension_cls"] == benchmark._GRAPH_WORKER_EXTENSION
    if mode == "target_only":
        assert kwargs["speculative_config"] is None
    else:
        assert kwargs["speculative_config"] == {
            "method": "dspark",
            "num_speculative_tokens": 5,
            "enforce_eager": True,
        }


def test_explicit_graph_capture_sizes_are_target_only_configuration(tmp_path: Path) -> None:
    args = benchmark.parse_args(
        [
            *_argv(tmp_path),
            "--target-execution-mode",
            "full_decode_only",
            "--cudagraph-capture-sizes",
            "6",
            "24",
        ]
    )

    kwargs = benchmark.build_engine_kwargs(args)

    assert kwargs["compilation_config"] == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [6, 24],
    }


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--target-execution-mode", "eager", "--no-enforce-eager"],
        ["--target-execution-mode", "full_decode_only", "--enforce-eager"],
        ["--target-execution-mode", "eager", "--cudagraph-capture-sizes", "1"],
    ],
)
def test_target_execution_options_reject_conflicts(tmp_path: Path, extra_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args([*_argv(tmp_path), *extra_args])


def test_full_decode_only_rejects_launch_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _graph_args(tmp_path)
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")

    with pytest.raises(RuntimeError, match="rejects ASCEND_LAUNCH_BLOCKING=1"):
        benchmark.build_engine_kwargs(args)


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


def test_uniform_kv_groups_require_requested_frontend_and_scheduler_size(tmp_path: Path) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    engine.llm_engine.engine_core.block_sizes = [32, 32]

    effective = benchmark._effective_engine_config(engine, args)

    resolved = effective["resolved_kv_block_config"]
    assert effective["block_size"] == 32
    assert resolved["frontend_ready_block_size"] == 32
    assert resolved["scheduler_block_size"] == 32
    assert resolved["group_block_sizes"] == [32, 32]
    assert resolved["validation"] == "uniform_kv_exact"


@pytest.mark.parametrize(
    ("requested", "mapped"),
    [
        (32, [32, 32, 2, 8]),
        (64, [64, 64, 4, 16]),
        (128, [128, 128, 8, 32]),
    ],
)
def test_deepseek_v4_hybrid_kv_block_identity_is_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: int,
    mapped: list[int],
) -> None:
    args = _args(tmp_path)
    args.block_size = requested
    engine = _FakeEngine(args)
    assert _configure_dsv4(monkeypatch, engine, args) == mapped

    effective = benchmark._effective_engine_config(
        engine,
        args,
        expected_model_type="deepseek_v4",
    )

    resolved = effective["resolved_kv_block_config"]
    assert effective["block_size"] == min(mapped)
    assert effective["block_size_semantics"] == "frontend_ready_block_size_legacy_alias"
    assert resolved["requested_base_block_size"] == requested
    assert resolved["platform_normalized_base_block_size"] == requested
    assert resolved["frontend_ready_block_size"] == min(mapped)
    assert resolved["scheduler_block_size"] == requested
    assert resolved["decode_context_parallel_size"] == 1
    assert resolved["prefill_context_parallel_size"] == 1
    assert resolved["group_block_sizes"] == mapped
    assert resolved["model_block_size_mapping"] == {
        "mla": mapped[0],
        "indexer": mapped[0],
        "sliding_window_mla": mapped[1],
        "c4_compressor_state": mapped[2],
        "c128_compressor_state": mapped[3],
    }
    assert resolved["validation"] == "deepseek_v4_hybrid_mapping"
    assert engine.llm_engine.engine_core.utility_calls == ["get_kv_cache_group_metadata"]


def test_deepseek_v4_resolved_kv_identity_is_serialized_and_shared_across_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results: list[dict[str, Any]] = []
    for mode in ("target_only", "dspark"):
        mode_root = tmp_path / mode
        mode_root.mkdir()
        args = _args(mode_root, mode)
        _set_model_type(Path(args.model_dir), "deepseek_v4", "DeepseekV4ForCausalLM")
        engine = _FakeEngine(args)
        _configure_dsv4(monkeypatch, engine, args)
        monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)
        result = benchmark.run_benchmark(
            args,
            engine_factory=lambda _kwargs, engine=engine: engine,
            sampling_factory=lambda _args: object(),
            clock=iter((10.0, 12.0)).__next__,
            plugin_root=mode_root,
            core_root=mode_root,
        )
        results.append(result)

    target, dspark = results
    for result in results:
        resolved = result["resolved_kv_block_config"]
        assert result["requested_engine_config"]["block_size"] == 32
        assert result["effective_engine_config"]["block_size"] == 2
        assert resolved == result["effective_engine_config"]["resolved_kv_block_config"]
        assert resolved["group_block_sizes"] == [32, 32, 2, 8]
        assert resolved["scheduler_block_size"] == 32
        assert resolved["frontend_ready_block_size"] == 2
        json.dumps(result)
    assert target["comparison_config"]["effective_engine"] == dspark["comparison_config"]["effective_engine"]


def test_deepseek_v4_rejects_unexplained_frontend_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    _configure_dsv4(monkeypatch, engine, args, frontend_ready_block_size=4)

    with pytest.raises(
        RuntimeError,
        match=r"frontend_ready_block_size=4, group_block_sizes=\[32, 32, 2, 8\]",
    ):
        benchmark._effective_engine_config(engine, args)


def test_deepseek_v4_rejects_group_outside_active_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    _configure_dsv4(monkeypatch, engine, args, group_block_sizes=[32, 32, 2, 8, 16])

    with pytest.raises(RuntimeError, match=r"unexpected=\[16\]"):
        benchmark._effective_engine_config(engine, args)


def test_scheduler_alignment_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    _configure_dsv4(monkeypatch, engine, args)
    monkeypatch.setattr(benchmark, "_scheduler_block_size_from_groups", lambda *_args: 16)

    with pytest.raises(
        RuntimeError,
        match=r"scheduler_block_size=16, expected_scheduler_block_size=32",
    ):
        benchmark._effective_engine_config(engine, args)


def test_non_deepseek_model_cannot_use_hybrid_mapping_exception(tmp_path: Path) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    engine.llm_engine.vllm_config.cache_config.block_size = 2
    engine.llm_engine.engine_core.block_sizes = [32, 32, 2, 8]

    with pytest.raises(RuntimeError, match="without the DeepSeek-V4 hybrid mapping"):
        benchmark._effective_engine_config(engine, args)


def test_missing_engine_core_group_metadata_abi_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    del engine.llm_engine.engine_core

    with pytest.raises(RuntimeError, match="KV-group metadata utility"):
        benchmark._effective_engine_config(engine, args)


def test_effective_config_reports_prefix_cache_mismatch_separately(tmp_path: Path) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    engine.llm_engine.vllm_config.cache_config.enable_prefix_caching = True

    with pytest.raises(
        RuntimeError,
        match=r"requested enable_prefix_caching=False, effective enable_prefix_caching=True",
    ):
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
    assert result["requested_engine_config"]["block_size"] == 32
    assert result["effective_engine_config"]["block_size"] == 32
    assert result["effective_engine_config"]["block_size_semantics"] == "frontend_ready_block_size_legacy_alias"
    assert result["effective_config"] == result["effective_engine_config"]
    assert result["comparison_config"]["effective_engine"]["block_size"] == 32
    assert result["resolved_kv_block_config"] == result["effective_engine_config"]["resolved_kv_block_config"]
    assert result["resolved_kv_block_config"]["group_block_sizes"] == [32]
    assert result["resolved_kv_block_config"]["scheduler_block_size"] == 32
    assert result["cleanup"]["engine_shutdown_complete"] is True
    assert engine.shutdown_called is True
    json.dumps(result)


@pytest.mark.parametrize("mode", ["target_only", "dspark"])
def test_graph_run_records_capture_replay_and_eager_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    result, engine = _run_graph(tmp_path, monkeypatch, mode)

    assert result["target_execution_mode_requested"] == "full_decode_only"
    assert result["target_execution_mode_effective"] == "full_decode_only"
    assert result["target_enforce_eager"] is False
    assert result["dspark_enforce_eager"] is (True if mode == "dspark" else None)
    assert result["cudagraph_mode_requested"] == "FULL_DECODE_ONLY"
    assert result["cudagraph_mode_effective"] == "FULL_DECODE_ONLY"
    assert result["npugraph_ex_enabled"] is True
    assert result["static_kernel_enabled"] is False
    assert result["configured_capture_sizes"] == [6, 24]
    assert result["observed_capture_sizes"] == [6, 24]
    assert result["graph_capture_count"] == 2
    assert result["graph_replay_count"] == 2
    assert result["graph_fallback_count"] is None
    assert result["measured_graph_replay_count"] == 1
    assert result["measured_eager_fallback_count"] is None
    assert result["graph_execution"]["warmup_runtime"]["record_count"] == 1
    assert result["graph_execution"]["measured_runtime"]["record_count"] == 1
    assert result["timing"]["graph_capture_included"] is False
    assert result["timing"]["graph_warmup_included"] is False
    assert [len(call) for call in engine.generate_calls] == [1, 2]
    json.dumps(result)


def test_graph_run_fails_before_generation_without_capture(tmp_path, monkeypatch):
    def configure(engine):
        engine.graph_capture_count = 0
        engine.observed_capture_sizes = []

    with pytest.raises(RuntimeError, match="without captured target graph descriptors"):
        _run_graph(tmp_path, monkeypatch, "dspark", configure)


def test_graph_run_preserves_diagnostics_without_actual_replay(tmp_path, monkeypatch):
    result, engine = _run_graph(
        tmp_path, monkeypatch, "dspark", lambda engine: setattr(engine, "graph_runtime_mode", "NONE")
    )
    assert result["graph_execution"]["replay_evidence_status"] == "unavailable"
    assert "performed no target graph replay" in result["graph_execution"]["error"]
    assert result["measured_graph_replay_count"] is None
    assert result["outputs"] and result["timing"]["elapsed_seconds"] == 2.0
    assert engine.shutdown_called
    with pytest.raises(ValueError, match="Replay evidence unavailable"):
        summary._validate_graph_execution(result, "dspark")


def test_capture_and_warmup_full_do_not_replace_measured_replay(tmp_path, monkeypatch):
    args = _graph_args(tmp_path)
    engine = _FakeEngine(args)
    generate = engine.generate

    def generate_phase(prompts, params, *, use_tqdm):
        engine.graph_runtime_mode = "NONE" if engine.generate_calls else "FULL"
        return generate(prompts, params, use_tqdm=use_tqdm)

    engine.generate = generate_phase
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)
    result = benchmark.run_benchmark(
        args,
        engine_factory=lambda kwargs: engine,
        sampling_factory=lambda args: object(),
        clock=iter((10.0, 12.0)).__next__,
        plugin_root=tmp_path,
        core_root=tmp_path,
    )
    assert result["graph_execution"]["replay_evidence_status"] == "unavailable"
    assert result["graph_execution"]["warmup_runtime"]["graph_replay_count"] == 1
    assert result["graph_execution"]["measured_runtime"]["graph_replay_count"] == 0
    assert result["outputs"] and result["timing"]["elapsed_seconds"] == 2.0
    assert len(engine.generate_calls) == 2 and engine.graph_capture_count > 0
    assert engine.shutdown_called


def test_graph_run_rejects_non_eager_dspark_config(tmp_path: Path) -> None:
    args = _graph_args(tmp_path)
    engine = _FakeEngine(args)
    engine.llm_engine.vllm_config.speculative_config.enforce_eager = False

    with pytest.raises(RuntimeError, match="eager draft execution"):
        benchmark._effective_engine_config(engine, args)


def test_target_only_records_speculative_metrics_as_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _engine = _run(tmp_path, monkeypatch, "target_only")

    assert result["effective_config"]["speculative_config"] is None
    assert result["metrics"]["measured_delta"] is None
    assert result["acceptance"]["num_drafts"] is None
    assert result["acceptance"]["acceptance_per_position"] == [None] * 5


class _TrackedRenderer:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class SyncMPClient:
    def __init__(self, *, fail_shutdown: bool = False):
        self.shutdown_calls = 0
        self.fail_shutdown = fail_shutdown

    def shutdown(self, timeout: float | None = None) -> None:
        assert timeout is None
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError("EngineCore cleanup failed")


SyncMPClient.__module__ = "vllm.v1.engine.core_client"


class LLMEngine:
    def __init__(self, engine_core: Any):
        self.shutdown_calls = 0
        self.renderer = _TrackedRenderer()
        self.engine_core = engine_core
        self.dp_group = None
        self.external_launcher_dp = False

    def shutdown(self, timeout: float | None = None) -> None:
        assert timeout is None
        self.shutdown_calls += 1
        # Mirrors the current core's failing frontend-model probe and the
        # renderer/EngineCore ordering that follows it.
        getattr(self.model_executor, "driver_worker", None)
        if self.renderer is not None:
            self.renderer.shutdown()
            self.renderer = None
        if self.engine_core is not None:
            self.engine_core.shutdown(timeout=timeout)
            self.engine_core = None


LLMEngine.__module__ = "vllm.v1.engine.llm_engine"


class LLM:
    def __init__(self, engine_core: Any | None = None):
        self.llm_engine = LLMEngine(engine_core or SyncMPClient())


LLM.__module__ = "vllm.entrypoints.llm"


def test_current_public_llm_mp_layout_shutdown_is_complete_and_idempotent() -> None:
    engine = LLM()
    llm_engine = engine.llm_engine
    renderer = llm_engine.renderer
    engine_core = llm_engine.engine_core
    cleanup = benchmark._BenchmarkEngineCleanup(engine)

    assert not hasattr(engine, "shutdown")
    assert not hasattr(llm_engine, "model_executor")

    cleanup.shutdown()
    cleanup.shutdown()

    assert cleanup.complete is True
    assert llm_engine.model_executor is None
    assert llm_engine.shutdown_calls == 1
    assert renderer.shutdown_calls == 1
    assert engine_core.shutdown_calls == 1
    assert llm_engine.renderer is None
    assert llm_engine.engine_core is None


def test_unknown_missing_model_executor_layout_fails_closed() -> None:
    class UnknownEngineCore:
        def shutdown(self, timeout: float | None = None) -> None:
            raise AssertionError("unknown EngineCore must not be called")

    engine = LLM(UnknownEngineCore())
    cleanup = benchmark._BenchmarkEngineCleanup(engine)

    with pytest.raises(RuntimeError, match="does not match the supported"):
        cleanup.shutdown()
    with pytest.raises(RuntimeError, match="refusing to repeat"):
        cleanup.shutdown()

    assert cleanup.complete is False
    assert engine.llm_engine.shutdown_calls == 0


def test_failed_engine_core_cleanup_is_not_repeated() -> None:
    engine_core = SyncMPClient(fail_shutdown=True)
    engine = LLM(engine_core)
    renderer = engine.llm_engine.renderer
    cleanup = benchmark._BenchmarkEngineCleanup(engine)

    with pytest.raises(RuntimeError, match="EngineCore cleanup failed"):
        cleanup.shutdown()
    with pytest.raises(RuntimeError, match="refusing to repeat"):
        cleanup.shutdown()

    assert cleanup.complete is False
    assert renderer.shutdown_calls == 1
    assert engine_core.shutdown_calls == 1


def test_configuration_failure_still_cleans_up_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    engine.llm_engine.vllm_config.cache_config.block_size = 64
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)

    with pytest.raises(RuntimeError, match=r"frontend_ready_block_size=64, group_block_sizes=\[32\]"):
        benchmark.run_benchmark(
            args,
            engine_factory=lambda _kwargs: engine,
            sampling_factory=lambda _args: object(),
            plugin_root=tmp_path,
            core_root=tmp_path,
        )

    assert engine.shutdown_called is True


def test_measured_generation_failure_still_cleans_up_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    original_generate = engine.generate
    calls = 0

    def fail_measured_generate(
        prompts: list[dict[str, list[int]]],
        params: object,
        *,
        use_tqdm: bool,
    ) -> list[_Output]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("measured generation failed")
        return original_generate(prompts, params, use_tqdm=use_tqdm)

    engine.generate = fail_measured_generate  # type: ignore[method-assign]
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)

    with pytest.raises(ValueError, match="measured generation failed"):
        benchmark.run_benchmark(
            args,
            engine_factory=lambda _kwargs: engine,
            sampling_factory=lambda _args: object(),
            plugin_root=tmp_path,
            core_root=tmp_path,
        )

    assert engine.shutdown_called is True


def test_cleanup_failure_does_not_replace_primary_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)
    engine.llm_engine.vllm_config.cache_config.block_size = 64

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    engine.llm_engine.shutdown = fail_cleanup
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)

    with pytest.raises(RuntimeError, match=r"frontend_ready_block_size=64, group_block_sizes=\[32\]") as caught:
        benchmark.run_benchmark(
            args,
            engine_factory=lambda _kwargs: engine,
            sampling_factory=lambda _args: object(),
            plugin_root=tmp_path,
            core_root=tmp_path,
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "cleanup failed"


def test_successful_run_fails_if_cleanup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    engine = _FakeEngine(args)

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    engine.llm_engine.shutdown = fail_cleanup
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        benchmark.run_benchmark(
            args,
            engine_factory=lambda _kwargs: engine,
            sampling_factory=lambda _args: object(),
            clock=iter((10.0, 12.0)).__next__,
            plugin_root=tmp_path,
            core_root=tmp_path,
        )


def _independent_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    graph: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    dsparks: list[dict[str, Any]] = []
    for index in range(3):
        target_root = tmp_path / f"target-{index}"
        dspark_root = tmp_path / f"dspark-{index}"
        target_root.mkdir(parents=True)
        dspark_root.mkdir(parents=True)
        runner = _run_graph if graph else _run
        target, _ = runner(target_root, monkeypatch, "target_only")
        dspark, _ = runner(dspark_root, monkeypatch, "dspark")
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


def test_graph_summary_preserves_target_replay_evidence_across_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch, graph=True)

    result = summary.summarize_results(targets, dsparks)

    assert result["target_execution"]["target_only"]["mode"] == "full_decode_only"
    assert result["target_execution"]["dspark"]["mode"] == "full_decode_only"
    assert result["target_execution"]["target_only"]["measured_graph_replay_count"]["minimum"] == 1
    assert result["target_execution"]["dspark"]["measured_graph_replay_count"]["minimum"] == 1


def test_run_csv_preserves_absolute_throughput_and_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch)
    path = tmp_path / "runs.csv"

    summary._write_run_csv(path, [targets[0], dsparks[0]])
    rows = path.read_text(encoding="utf-8").splitlines()

    assert len(rows) == 3
    assert "output_tokens_per_second" in rows[0]
    assert "accepted_candidate_tokens_per_verification" in rows[0]
    assert "comparison_config_fingerprint" in rows[0]
    assert "scheduler_block_size" in rows[0]
    assert "group_block_sizes" in rows[0]
    assert "target_execution_mode" in rows[0]
    assert "measured_graph_replay_count" in rows[0]
    assert rows[1].startswith("target_only,")
    assert rows[2].startswith("dspark,")


def test_summary_accepts_legacy_eager_result_without_graph_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _engine = _run(tmp_path, monkeypatch, "target_only")
    for field in summary._GRAPH_RESULT_FIELDS:
        result.pop(field)
    result.pop("graph_execution")

    summary._validate_result(result, "target_only")


def test_summary_rejects_graph_result_without_measured_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, _engine = _run_graph(tmp_path, monkeypatch, "dspark")
    result["measured_graph_replay_count"] = 0
    result["graph_execution"]["measured_graph_replay_count"] = 0

    with pytest.raises(ValueError, match="no measured target graph replay"):
        summary._validate_result(result, "dspark")


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


def test_summary_rejects_different_resolved_kv_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets, dsparks = _independent_results(tmp_path, monkeypatch)
    dspark = dsparks[0]
    for config in (dspark["effective_engine_config"], dspark["effective_config"]):
        config["block_size"] = 64
        config["resolved_kv_block_config"]["requested_base_block_size"] = 64
        config["resolved_kv_block_config"]["platform_normalized_base_block_size"] = 64
        config["resolved_kv_block_config"]["frontend_ready_block_size"] = 64
        config["resolved_kv_block_config"]["scheduler_block_size"] = 64
        config["resolved_kv_block_config"]["group_block_sizes"] = [64]
        config["resolved_kv_block_config"]["group_metadata"][0]["block_size"] = 64
    dspark["requested_engine_config"]["block_size"] = 64
    dspark["resolved_kv_block_config"] = copy.deepcopy(dspark["effective_engine_config"]["resolved_kv_block_config"])
    dspark["comparison_config"]["effective_engine"] = {
        key: value for key, value in dspark["effective_engine_config"].items() if key != "speculative_config"
    }
    dspark["comparison_config_fingerprint"] = benchmark._sha256_bytes(
        benchmark._canonical_json_bytes(dspark["comparison_config"])
    )

    with pytest.raises(ValueError, match="configuration differs"):
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


def test_cleanup_has_no_process_kill_or_global_shutdown_patch() -> None:
    source = Path(benchmark.__file__).read_text(encoding="utf-8")

    assert "pkill" not in source
    assert "killall" not in source
    assert "os._exit" not in source
    assert "LLMEngine.shutdown =" not in source


@pytest.mark.parametrize("fault", ["missing_rank", "rpc_failure"])
def test_post_generate_telemetry_failure_keeps_json_and_has_no_pass_marker(tmp_path, monkeypatch, capsys, fault):
    args = _graph_args(tmp_path)
    engine = _FakeEngine(args)
    rpc = engine.collective_rpc
    events = []

    def observed_rpc(method):
        events.append("rpc")
        states = rpc(method)
        if method == benchmark._REPLAY_SNAPSHOT_METHOD and len(engine.generate_calls) == 2:
            if fault == "rpc_failure":
                raise RuntimeError("telemetry transport failed")
            states.pop()
        return states

    def clock():
        events.append("clock")
        return float(events.count("clock"))

    engine.collective_rpc = observed_rpc
    monkeypatch.setattr(benchmark, "_git_head", lambda _path: "a" * 40)
    result = benchmark.run_benchmark(
        args,
        engine_factory=lambda kwargs: engine,
        sampling_factory=lambda args: object(),
        clock=clock,
        plugin_root=tmp_path,
        core_root=tmp_path,
    )
    assert events == ["rpc", "rpc", "rpc", "clock", "clock", "rpc"]
    assert result["graph_execution"]["replay_evidence_status"] == "unavailable"
    assert result["outputs"] and result["throughput"]["total_output_tokens"] == 8
    monkeypatch.setattr(benchmark, "parse_args", lambda argv: args)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda args: result)
    assert benchmark.main([]) == 1
    saved = json.loads(args.result_json.read_text())
    assert saved == result and saved["cleanup"]["engine_shutdown_complete"]
    output = capsys.readouterr()
    assert "BENCHMARK_PASS" not in output.out and "Replay evidence unavailable" in output.err
