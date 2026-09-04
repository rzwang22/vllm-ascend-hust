#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""PR-style public-LLM throughput and acceptance benchmark for DSpark MRV2.

The module deliberately has no import-time dependency on vLLM, torch, NPU, or
datasets.  Server-only dependencies are loaded after the MRV2 environment has
been fixed and only when the corresponding execution path is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUNNER = "mrv2"
MODES = ("target_only", "dspark")
SPEC_METRIC_NAMES = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos",
)
VECTOR_METRIC_NAME = "vllm:spec_decode_num_accepted_tokens_per_pos"
DEFAULT_GSM8K_REVISION = "cc7b047b6e5bb11b4f1af84efc572db110a51b3c"
MODEL_FINGERPRINT_FILES = (
    "config.json",
    "quant_model_description.json",
    "model.safetensors.index.json",
    "quant_model_weights.safetensors.index.json",
)
_PUBLIC_LLM_TYPE = ("vllm.entrypoints.llm", "LLM")
_MULTIPROCESS_LLM_ENGINE_TYPE = ("vllm.v1.engine.llm_engine", "LLMEngine")
_SYNC_ENGINE_CORE_CLIENT_TYPE = ("vllm.v1.engine.core_client", "SyncMPClient")


@dataclass(frozen=True)
class PromptSource:
    source_index: int
    value: str | list[int]
    source_record_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json_bytes(value))
    temporary.replace(path)


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to resolve Git HEAD for {root}.") from exc


def _nested_field(record: Mapping[str, Any], field: str) -> Any:
    value: Any = record
    for component in field.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"Dataset record has no prompt field {field!r}.")
        value = value[component]
    return value


def _prompt_value(value: Any, field: str) -> str | list[int]:
    if isinstance(value, str):
        if not value:
            raise ValueError(f"Prompt field {field!r} must not be empty.")
        return value
    if isinstance(value, list) and value:
        if any(isinstance(item, bool) or not isinstance(item, Integral) or item < 0 for item in value):
            raise ValueError(f"Token prompt field {field!r} contains invalid token IDs.")
        return [int(item) for item in value]
    raise TypeError(f"Prompt field {field!r} must be non-empty text or a token-ID list.")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Partial JSONL dataset (missing final newline): {path}.")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
        if not isinstance(value, dict):
            raise TypeError(f"JSONL record at {path}:{line_number} must be an object.")
        records.append(value)
    return records


def load_prompt_sources(
    args: argparse.Namespace,
    *,
    hf_loader: Callable[..., Any] | None = None,
) -> tuple[list[PromptSource], dict[str, Any]]:
    """Load the first N prompts in source order without cycling the dataset."""
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    if args.dataset_name == "jsonl":
        path = Path(args.dataset_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"JSONL dataset does not exist: {path}.")
        raw_records = _read_jsonl(path)
        descriptor = {
            "kind": "jsonl",
            "path": str(path),
            "file_sha256": _sha256_file(path),
            "split": args.dataset_split,
            "prompt_field": args.prompt_field,
            "selection": "source_order_first_n",
        }
    elif args.dataset_name == "hf":
        if not args.dataset_revision:
            raise ValueError("HF datasets require --dataset-revision for a reproducible source identity.")
        if hf_loader is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("HF dataset mode requires the optional 'datasets' package.") from exc
            hf_loader = load_dataset
        load_kwargs: dict[str, Any] = {
            "split": args.dataset_split,
            "revision": args.dataset_revision,
        }
        if args.dataset_config:
            load_kwargs["name"] = args.dataset_config
        dataset = hf_loader(args.dataset_path, **load_kwargs)
        raw_records = [dict(dataset[index]) for index in range(len(dataset))]
        descriptor = {
            "kind": "hf",
            "repository": args.dataset_path,
            "config": args.dataset_config,
            "revision": args.dataset_revision,
            "split": args.dataset_split,
            "prompt_field": args.prompt_field,
            "file_sha256": None,
            "selection": "source_order_first_n",
        }
    else:
        raise ValueError(f"Unsupported dataset mode: {args.dataset_name!r}.")

    if len(raw_records) < args.num_prompts:
        raise ValueError(
            f"Dataset contains {len(raw_records)} records, fewer than --num-prompts={args.num_prompts}; "
            "prompts are never cycled."
        )
    selected: list[PromptSource] = []
    for source_index, record in enumerate(raw_records[: args.num_prompts]):
        selected.append(
            PromptSource(
                source_index=source_index,
                value=_prompt_value(_nested_field(record, args.prompt_field), args.prompt_field),
                source_record_sha256=_sha256_bytes(_canonical_json_bytes(record)),
            )
        )
    descriptor["available_record_count"] = len(raw_records)
    descriptor["selected_record_count"] = len(selected)
    descriptor["selected_source_records_sha256"] = _sha256_bytes(
        _canonical_json_bytes([record.source_record_sha256 for record in selected])
    )
    descriptor["scale"] = "pr_scale_400_prompt" if len(selected) >= 400 else f"{len(selected)}-prompt local smoke"
    return selected, descriptor


def tokenize_prompt_sources(
    sources: Sequence[PromptSource],
    tokenizer: Any,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]], str]:
    prompts: list[dict[str, list[int]]] = []
    identities: list[dict[str, Any]] = []
    for request_index, source in enumerate(sources):
        if isinstance(source.value, str):
            token_ids = tokenizer.encode(source.value)
        else:
            token_ids = source.value
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"Tokenizer returned no token IDs for request {request_index}.")
        if any(isinstance(token, bool) or not isinstance(token, Integral) or token < 0 for token in token_ids):
            raise ValueError(f"Tokenizer returned invalid token IDs for request {request_index}.")
        normalized = [int(token) for token in token_ids]
        prompt_hash = _sha256_bytes(_canonical_json_bytes(normalized))
        prompts.append({"prompt_token_ids": normalized})
        identities.append(
            {
                "request_index": request_index,
                "source_index": source.source_index,
                "source_record_sha256": source.source_record_sha256,
                "prompt_token_count": len(normalized),
                "prompt_token_sha256": prompt_hash,
            }
        )
    prompt_set_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            [
                {
                    "request_index": item["request_index"],
                    "prompt_token_sha256": item["prompt_token_sha256"],
                }
                for item in identities
            ]
        )
    )
    return prompts, identities, prompt_set_sha256


def _metric_labels(metric: Any) -> dict[str, str]:
    labels = getattr(metric, "labels", None)
    if not isinstance(labels, Mapping):
        raise TypeError(f"Metric {getattr(metric, 'name', None)!r} has invalid labels.")
    return {str(key): str(value) for key, value in sorted(labels.items())}


def capture_spec_metrics(metrics: Sequence[Any]) -> dict[str, list[dict[str, Any]]]:
    """Capture exact Counter/Vector series without importing vLLM reader types."""
    captured = {name: [] for name in SPEC_METRIC_NAMES}
    identities: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    for metric in metrics:
        name = getattr(metric, "name", None)
        if name not in captured:
            continue
        labels = _metric_labels(metric)
        identity = (name, tuple(labels.items()))
        if identity in identities:
            raise ValueError(f"Duplicate metric series for {name!r} and labels {labels!r}.")
        identities.add(identity)
        if name == VECTOR_METRIC_NAME:
            values = getattr(metric, "values", None)
            if not isinstance(values, list) or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values
            ):
                raise TypeError(f"Metric {name!r} must be a non-negative integer Vector.")
            captured[name].append({"kind": "vector", "labels": labels, "values": [int(value) for value in values]})
        else:
            value = getattr(metric, "value", None)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise TypeError(f"Metric {name!r} must be a non-negative integer Counter.")
            captured[name].append({"kind": "counter", "labels": labels, "value": int(value)})
    return captured


def _series_map(series: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {json.dumps(item["labels"], sort_keys=True, separators=(",", ":")): item for item in series}


def metric_snapshot_delta(
    start: Mapping[str, Sequence[Mapping[str, Any]]],
    end: Mapping[str, Sequence[Mapping[str, Any]]],
    num_spec_tokens: int,
) -> dict[str, Any]:
    per_series: dict[str, list[dict[str, Any]]] = {}
    totals: dict[str, int | list[int]] = {}
    for name in SPEC_METRIC_NAMES:
        start_series = _series_map(start.get(name, []))
        end_series = _series_map(end.get(name, []))
        if not start_series or start_series.keys() != end_series.keys():
            raise ValueError(f"Required measured metric series {name!r} is missing or changed after warmup.")
        deltas: list[dict[str, Any]] = []
        if name == VECTOR_METRIC_NAME:
            total_vector = [0] * num_spec_tokens
            for identity in sorted(start_series):
                before = start_series[identity]
                after = end_series[identity]
                before_values = before["values"]
                after_values = after["values"]
                if len(before_values) != num_spec_tokens or len(after_values) != num_spec_tokens:
                    raise ValueError(f"Metric {name!r} must contain exactly K={num_spec_tokens} positions.")
                values = [right - left for left, right in zip(before_values, after_values)]
                if any(value < 0 for value in values):
                    raise ValueError(f"Metric {name!r} decreased during the measured interval.")
                total_vector = [left + right for left, right in zip(total_vector, values)]
                deltas.append({"kind": "vector", "labels": before["labels"], "values": values})
            totals[name] = total_vector
        else:
            total = 0
            for identity in sorted(start_series):
                before = start_series[identity]
                after = end_series[identity]
                value = after["value"] - before["value"]
                if value < 0:
                    raise ValueError(f"Metric {name!r} decreased during the measured interval.")
                total += value
                deltas.append({"kind": "counter", "labels": before["labels"], "value": value})
            totals[name] = total
        per_series[name] = deltas
    return {"series": per_series, "totals": totals}


def acceptance_from_delta(delta: Mapping[str, Any], num_spec_tokens: int) -> dict[str, Any]:
    totals = delta["totals"]
    drafts = totals["vllm:spec_decode_num_drafts"]
    draft_tokens = totals["vllm:spec_decode_num_draft_tokens"]
    accepted = totals["vllm:spec_decode_num_accepted_tokens"]
    per_position = totals[VECTOR_METRIC_NAME]
    if not isinstance(drafts, int) or drafts <= 0:
        raise ValueError("The measured DSpark interval contains no real draft verification metrics.")
    if not isinstance(draft_tokens, int) or draft_tokens <= 0:
        raise ValueError("The measured DSpark interval contains no real draft tokens.")
    if not isinstance(accepted, int) or not isinstance(per_position, list):
        raise TypeError("Unexpected speculative metric aggregate types.")
    if len(per_position) != num_spec_tokens:
        raise ValueError(f"Accepted-position vector length differs from K={num_spec_tokens}.")
    if any(left < right for left, right in zip(per_position, per_position[1:])):
        raise ValueError("Accepted-token counts must be non-increasing by draft position.")
    if any(value > drafts for value in per_position):
        raise ValueError("Accepted-token position count exceeds the number of drafts.")
    if sum(per_position) != accepted:
        raise ValueError("Accepted-token counter differs from the per-position vector sum.")
    if accepted > draft_tokens or draft_tokens > drafts * num_spec_tokens:
        raise ValueError("Draft/accepted token counters violate the K-token proposal bounds.")
    accepted_per_verification = accepted / drafts
    return {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_candidate_tokens": accepted,
        "accepted_candidate_tokens_per_verification": accepted_per_verification,
        "effective_committed_tokens_per_verification": 1.0 + accepted_per_verification,
        "effective_acceptance_length": 1.0 + accepted_per_verification,
        "accepted_candidate_tokens_per_position": per_position,
        "acceptance_per_position": [value / drafts for value in per_position],
    }


def _not_applicable_acceptance(num_spec_tokens: int) -> dict[str, Any]:
    return {
        "num_drafts": None,
        "num_draft_tokens": None,
        "num_accepted_candidate_tokens": None,
        "accepted_candidate_tokens_per_verification": None,
        "effective_committed_tokens_per_verification": None,
        "effective_acceptance_length": None,
        "accepted_candidate_tokens_per_position": [None] * num_spec_tokens,
        "acceptance_per_position": [None] * num_spec_tokens,
    }


def _model_descriptor(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise ValueError(f"--model-dir must contain config.json: {model_dir}.")
    files: dict[str, str] = {}
    for name in MODEL_FINGERPRINT_FILES:
        path = model_dir / name
        if path.is_file():
            files[name] = _sha256_file(path)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    descriptor = {
        "path": str(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "files": files,
    }
    descriptor["fingerprint_sha256"] = _sha256_bytes(_canonical_json_bytes(descriptor))
    return descriptor


def _require_mrv2_environment(environ: dict[str, str] | os._Environ[str] = os.environ) -> None:
    if "vllm" in sys.modules and environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise RuntimeError("vLLM was imported before VLLM_USE_V2_MODEL_RUNNER=1 was established.")
    environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"


def build_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    speculative_config = None
    if args.mode == "dspark":
        speculative_config = {
            "method": "dspark",
            "num_speculative_tokens": args.num_spec_tokens,
        }
    kwargs: dict[str, Any] = {
        "model": str(Path(args.model_dir).expanduser().resolve()),
        "tokenizer": str(Path(args.model_dir).expanduser().resolve()),
        "tokenizer_mode": args.tokenizer_mode,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "enable_expert_parallel": args.enable_expert_parallel,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": args.async_scheduling,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": False,
        "speculative_config": speculative_config,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if args.kv_cache_memory_bytes is not None:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    return kwargs


def _requested_engine_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "use_v2_model_runner": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "enable_expert_parallel": args.enable_expert_parallel,
        "enforce_eager": args.enforce_eager,
        "async_scheduling": args.async_scheduling,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "block_size": args.block_size,
        "enable_prefix_caching": False,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "speculative_config": build_engine_kwargs(args)["speculative_config"],
    }


def _public_engine_factory(kwargs: Mapping[str, Any]) -> Any:
    from vllm import LLM

    return LLM(**kwargs)


def _sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.output_len,
        seed=args.seed,
        ignore_eos=args.ignore_eos,
    )


def _runtime_dsv4_block_sizes() -> Mapping[int, Sequence[Sequence[int]]]:
    """Load the active Ascend DeepSeek-V4 block mapping after platform init."""
    from vllm_ascend.models.layer.attention.layer import get_dsv4_block_sizes

    return get_dsv4_block_sizes()


def _engine_core_kv_group_metadata(engine: Any) -> list[dict[str, Any]]:
    """Read scheduler-owned KV group metadata through the current core utility."""
    llm_engine = getattr(engine, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    call_utility = getattr(engine_core, "call_utility", None)
    if not callable(call_utility):
        raise RuntimeError(
            "The public LLM does not expose the current EngineCore KV-group "
            "metadata utility; hybrid KV configuration cannot be validated."
        )

    raw_metadata = call_utility("get_kv_cache_group_metadata")
    if not isinstance(raw_metadata, list) or not raw_metadata:
        raise RuntimeError("EngineCore returned no scheduler KV cache group metadata.")

    metadata: list[dict[str, Any]] = []
    for expected_index, raw_group in enumerate(raw_metadata):
        if not isinstance(raw_group, Mapping):
            raise TypeError("EngineCore KV cache group metadata entries must be mappings.")
        group_index = raw_group.get("group_idx")
        block_size = raw_group.get("block_size")
        kind = raw_group.get("kind")
        if isinstance(group_index, bool) or not isinstance(group_index, int) or group_index != expected_index:
            raise RuntimeError("EngineCore KV cache group indexes are incomplete or out of order.")
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise RuntimeError("EngineCore reported an invalid KV cache group block size.")
        if not isinstance(kind, str) or not kind:
            raise RuntimeError("EngineCore reported an invalid KV cache group kind.")
        metadata.append(
            {
                "group_idx": group_index,
                "kind": kind,
                "block_size": block_size,
                "sliding_window": raw_group.get("sliding_window"),
            }
        )
    return metadata


def _scheduler_block_size_from_groups(group_block_sizes: Sequence[int], dcp: int, pcp: int) -> int:
    if not group_block_sizes:
        raise ValueError("At least one KV cache group block size is required.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (*group_block_sizes, dcp, pcp)
    ):
        raise ValueError("KV group block sizes and context-parallel sizes must be positive integers.")
    if len(group_block_sizes) == 1:
        return group_block_sizes[0] * dcp * pcp
    return math.lcm(*group_block_sizes) * dcp * pcp


def _resolved_kv_block_config(
    engine: Any,
    args: argparse.Namespace,
    *,
    model_type: str,
    frontend_ready_block_size: int,
    dcp: int,
    pcp: int,
) -> dict[str, Any]:
    group_metadata = _engine_core_kv_group_metadata(engine)
    group_block_sizes = [group["block_size"] for group in group_metadata]
    expected_frontend_ready = min(group_block_sizes)
    if frontend_ready_block_size != expected_frontend_ready:
        raise RuntimeError(
            "The frontend ready-response KV block size does not match the "
            "EngineCore group minimum: "
            f"frontend_ready_block_size={frontend_ready_block_size}, "
            f"group_block_sizes={group_block_sizes}."
        )

    requested_base = args.block_size
    platform_normalized_base = requested_base
    model_block_size_mapping: dict[str, int] | None = None
    validation_mode = "uniform_kv_exact"
    if model_type == "deepseek_v4":
        block_sizes = _runtime_dsv4_block_sizes()
        active_mapping = block_sizes.get(requested_base)
        if (
            not isinstance(active_mapping, Sequence)
            or len(active_mapping) < 1
            or not isinstance(active_mapping[0], Sequence)
            or len(active_mapping[0]) != 4
        ):
            raise RuntimeError(
                "The requested DeepSeek-V4 base block size has no active Ascend mapping: "
                f"requested_base_block_size={requested_base}."
            )
        mapped_sizes = list(active_mapping[0])
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in mapped_sizes):
            raise RuntimeError("The active Ascend DeepSeek-V4 block-size mapping is invalid.")
        platform_normalized_base = mapped_sizes[0]
        model_block_size_mapping = {
            "mla": mapped_sizes[0],
            "indexer": mapped_sizes[0],
            "sliding_window_mla": mapped_sizes[1],
            "c4_compressor_state": mapped_sizes[2],
            "c128_compressor_state": mapped_sizes[3],
        }
        missing = {
            size: mapped_sizes.count(size) - group_block_sizes.count(size)
            for size in set(mapped_sizes)
            if group_block_sizes.count(size) < mapped_sizes.count(size)
        }
        unexpected = sorted(set(group_block_sizes).difference(mapped_sizes))
        if platform_normalized_base != requested_base or missing or unexpected:
            raise RuntimeError(
                "The EngineCore DeepSeek-V4 KV groups do not match the active "
                "Ascend block-size mapping: "
                f"requested_base_block_size={requested_base}, "
                f"platform_normalized_base_block_size={platform_normalized_base}, "
                f"mapped_block_sizes={mapped_sizes}, group_block_sizes={group_block_sizes}, "
                f"missing={missing}, unexpected={unexpected}."
            )
        validation_mode = "deepseek_v4_hybrid_mapping"
    elif any(block_size != requested_base for block_size in group_block_sizes):
        raise RuntimeError(
            "The public LLM resolved different KV group block sizes for a model "
            "without the DeepSeek-V4 hybrid mapping: "
            f"model_type={model_type!r}, requested_base_block_size={requested_base}, "
            f"group_block_sizes={group_block_sizes}."
        )

    scheduler_block_size = _scheduler_block_size_from_groups(group_block_sizes, dcp, pcp)
    expected_scheduler_block_size = platform_normalized_base * dcp * pcp
    if scheduler_block_size != expected_scheduler_block_size:
        raise RuntimeError(
            "The scheduler KV alignment does not match the platform-normalized "
            "base block size: "
            f"scheduler_block_size={scheduler_block_size}, "
            f"expected_scheduler_block_size={expected_scheduler_block_size}, "
            f"group_block_sizes={group_block_sizes}, dcp={dcp}, pcp={pcp}."
        )

    return {
        "model_type": model_type,
        "requested_base_block_size": requested_base,
        "platform_normalized_base_block_size": platform_normalized_base,
        "frontend_ready_block_size": frontend_ready_block_size,
        "scheduler_block_size": scheduler_block_size,
        "decode_context_parallel_size": dcp,
        "prefill_context_parallel_size": pcp,
        "group_block_sizes": group_block_sizes,
        "group_metadata": group_metadata,
        "model_block_size_mapping": model_block_size_mapping,
        "validation": validation_mode,
        "source": {
            "frontend_ready_block_size": "EngineCoreReadyResponse.block_size via SyncMPClient",
            "group_metadata": "EngineCore.get_kv_cache_group_metadata",
            "scheduler_block_size": "derived from current resolve_kv_cache_block_sizes contract",
            "platform_mapping": (
                "vllm_ascend.models.layer.attention.layer.get_dsv4_block_sizes" if model_type == "deepseek_v4" else None
            ),
        },
    }


def _effective_engine_config(
    engine: Any,
    args: argparse.Namespace,
    *,
    expected_model_type: str | None = None,
) -> dict[str, Any]:
    config = engine.llm_engine.vllm_config
    model = config.model_config
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    speculative = config.speculative_config
    hf_config = getattr(model, "hf_config", None)
    model_type = getattr(hf_config, "model_type", None)
    if not isinstance(model_type, str) or not model_type:
        raise RuntimeError("The public LLM did not expose a model_type for KV configuration validation.")
    if expected_model_type is not None and model_type != expected_model_type:
        raise RuntimeError(
            "The public LLM resolved a different model type: "
            f"expected={expected_model_type!r}, effective={model_type!r}."
        )
    if config.use_v2_model_runner is not True:
        raise RuntimeError("The public LLM did not resolve VLLM_USE_V2_MODEL_RUNNER=1 to MRV2.")
    if parallel.tensor_parallel_size != args.tensor_parallel_size or parallel.pipeline_parallel_size != 1:
        raise RuntimeError("The public LLM resolved an unexpected TP/PP topology.")
    if parallel.enable_expert_parallel is not args.enable_expert_parallel:
        raise RuntimeError("The public LLM resolved an unexpected expert-parallel setting.")
    if model.enforce_eager is not args.enforce_eager:
        raise RuntimeError("The public LLM resolved an unexpected eager setting.")
    if scheduler.async_scheduling is not args.async_scheduling:
        raise RuntimeError("The public LLM resolved an unexpected async-scheduling setting.")
    if model.max_model_len != args.max_model_len:
        raise RuntimeError("The public LLM resolved an unexpected maximum model length.")
    if scheduler.max_num_seqs != args.max_num_seqs:
        raise RuntimeError("The public LLM resolved an unexpected maximum sequence count.")
    if scheduler.max_num_batched_tokens != args.max_num_batched_tokens:
        raise RuntimeError("The public LLM resolved an unexpected batched-token limit.")
    if cache.enable_prefix_caching is not False:
        raise RuntimeError(
            "The public LLM resolved a different prefix-cache setting: "
            "requested enable_prefix_caching=False, "
            f"effective enable_prefix_caching={cache.enable_prefix_caching}."
        )
    dcp = getattr(parallel, "decode_context_parallel_size", 1)
    pcp = getattr(parallel, "prefill_context_parallel_size", 1)
    resolved_kv_blocks = _resolved_kv_block_config(
        engine,
        args,
        model_type=model_type,
        frontend_ready_block_size=cache.block_size,
        dcp=dcp,
        pcp=pcp,
    )
    if args.mode == "target_only":
        if speculative is not None:
            raise RuntimeError("target_only unexpectedly constructed a speculative decoder.")
        speculative_value = None
    else:
        if (
            speculative is None
            or speculative.method != "dspark"
            or speculative.num_speculative_tokens != args.num_spec_tokens
        ):
            raise RuntimeError("DSpark did not resolve method='dspark' with the requested K.")
        speculative_value = {
            "method": speculative.method,
            "num_speculative_tokens": speculative.num_speculative_tokens,
        }
    return {
        "use_v2_model_runner": True,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "enable_expert_parallel": parallel.enable_expert_parallel,
        "enforce_eager": model.enforce_eager,
        "async_scheduling": scheduler.async_scheduling,
        "max_model_len": model.max_model_len,
        "max_num_seqs": scheduler.max_num_seqs,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "model_type": model_type,
        "block_size": cache.block_size,
        "block_size_semantics": "frontend_ready_block_size_legacy_alias",
        "resolved_kv_block_config": resolved_kv_blocks,
        "enable_prefix_caching": cache.enable_prefix_caching,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
        "speculative_config": speculative_value,
    }


def _output_records(outputs: Sequence[Any], prompt_identities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(outputs) != len(prompt_identities):
        raise RuntimeError(f"generate() returned {len(outputs)} outputs for {len(prompt_identities)} prompts.")
    records: list[dict[str, Any]] = []
    for request_index, (output, prompt_identity) in enumerate(zip(outputs, prompt_identities)):
        completions = output.outputs
        if len(completions) != 1:
            raise RuntimeError("The benchmark requires exactly one completion per measured request.")
        completion = completions[0]
        token_ids = [int(token) for token in completion.token_ids]
        if output.prompt_token_ids is not None:
            observed_prompt_hash = _sha256_bytes(_canonical_json_bytes(list(output.prompt_token_ids)))
            if observed_prompt_hash != prompt_identity["prompt_token_sha256"]:
                raise RuntimeError(f"Public LLM changed measured prompt {request_index} token IDs.")
        records.append(
            {
                **prompt_identity,
                "output_token_count": len(token_ids),
                "output_token_sha256": _sha256_bytes(_canonical_json_bytes(token_ids)),
                "finish_reason": str(completion.finish_reason),
                "stop_reason": completion.stop_reason,
            }
        )
    return records


def _type_identity(value: Any) -> tuple[str, str]:
    value_type = type(value)
    return value_type.__module__, value_type.__name__


class _BenchmarkEngineCleanup:
    """One-shot public-LLM cleanup with a current-core MP compatibility path."""

    def __init__(self, engine: Any):
        self._engine = engine
        self._attempted = False
        self._complete = False

    @property
    def complete(self) -> bool:
        return self._complete

    def shutdown(self) -> None:
        if self._complete:
            return
        if self._attempted:
            raise RuntimeError("Benchmark engine cleanup already failed; refusing to repeat partial shutdown.")
        self._attempted = True

        direct_shutdown = getattr(self._engine, "shutdown", None)
        if callable(direct_shutdown):
            direct_shutdown()
            self._complete = True
            return

        llm_engine = getattr(self._engine, "llm_engine", None)
        shutdown = getattr(llm_engine, "shutdown", None)
        if not callable(shutdown):
            raise RuntimeError("Public LLM exposes no engine shutdown boundary.")

        if not hasattr(llm_engine, "model_executor"):
            engine_core = getattr(llm_engine, "engine_core", None)
            renderer = getattr(llm_engine, "renderer", None)
            is_current_mp_client = (
                _type_identity(self._engine) == _PUBLIC_LLM_TYPE
                and _type_identity(llm_engine) == _MULTIPROCESS_LLM_ENGINE_TYPE
                and _type_identity(engine_core) == _SYNC_ENGINE_CORE_CLIENT_TYPE
                and callable(getattr(engine_core, "shutdown", None))
                and callable(getattr(renderer, "shutdown", None))
                and hasattr(llm_engine, "dp_group")
                and hasattr(llm_engine, "external_launcher_dp")
            )
            if not is_current_mp_client:
                raise RuntimeError(
                    "Public LLM has no model_executor and does not match the supported "
                    "LLM/LLMEngine/SyncMPClient cleanup layout."
                )
            # Multiprocess LLMEngine deliberately has no frontend model executor:
            # the model is owned and released by EngineCore.  Supplying None lets
            # the current core shutdown method skip only its in-process compiled-
            # model cleanup, then retain its normal Prometheus, renderer,
            # EngineCore-client, and DP-group shutdown sequence.
            llm_engine.model_executor = None

        shutdown()
        self._complete = True


def _comparison_config(
    effective: Mapping[str, Any],
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    prompt_set_sha256: str,
    sampling: Mapping[str, Any],
    num_spec_tokens: int,
) -> dict[str, Any]:
    shared_effective = dict(effective)
    shared_effective.pop("speculative_config", None)
    return {
        "effective_engine": shared_effective,
        "model_fingerprint_sha256": model["fingerprint_sha256"],
        "dataset": dataset,
        "prompt_set_sha256": prompt_set_sha256,
        "sampling": sampling,
        "num_spec_tokens": num_spec_tokens,
    }


def run_benchmark(
    args: argparse.Namespace,
    *,
    engine_factory: Callable[[Mapping[str, Any]], Any] = _public_engine_factory,
    sampling_factory: Callable[[argparse.Namespace], Any] = _sampling_params,
    clock: Callable[[], float] = time.perf_counter,
    hf_loader: Callable[..., Any] | None = None,
    plugin_root: Path | None = None,
    core_root: Path | None = None,
) -> dict[str, Any]:
    _require_mrv2_environment()
    sources, dataset_descriptor = load_prompt_sources(args, hf_loader=hf_loader)
    model_descriptor = _model_descriptor(Path(args.model_dir))
    engine_kwargs = build_engine_kwargs(args)
    engine = engine_factory(engine_kwargs)
    cleanup = _BenchmarkEngineCleanup(engine)
    try:
        effective = _effective_engine_config(
            engine,
            args,
            expected_model_type=model_descriptor["model_type"],
        )
        prompts, prompt_identities, prompt_set_sha256 = tokenize_prompt_sources(sources, engine.get_tokenizer())
        sampling_params = sampling_factory(args)
        if args.warmup_prompts > len(prompts):
            raise ValueError("--warmup-prompts cannot exceed the measured prompt count.")
        warmup_count = args.warmup_prompts
        if warmup_count:
            engine.generate(prompts[:warmup_count], sampling_params, use_tqdm=False)
        metrics_start = capture_spec_metrics(engine.get_metrics())
        measured_started_at = clock()
        outputs = engine.generate(prompts, sampling_params, use_tqdm=False)
        measured_finished_at = clock()
        metrics_end = capture_spec_metrics(engine.get_metrics())
        elapsed_seconds = measured_finished_at - measured_started_at
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
            raise RuntimeError(f"Measured generate() duration must be positive, got {elapsed_seconds!r}.")
        output_records = _output_records(outputs, prompt_identities)
        total_output_tokens = sum(record["output_token_count"] for record in output_records)
        total_prompt_tokens = sum(record["prompt_token_count"] for record in output_records)
        if total_output_tokens <= 0:
            raise RuntimeError("Measured generate() produced no output tokens.")
        if args.mode == "dspark":
            metric_delta = metric_snapshot_delta(metrics_start, metrics_end, args.num_spec_tokens)
            acceptance = acceptance_from_delta(metric_delta, args.num_spec_tokens)
        else:
            metric_delta = None
            acceptance = _not_applicable_acceptance(args.num_spec_tokens)
        sampling = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "output_len": args.output_len,
            "seed": args.seed,
            "ignore_eos": args.ignore_eos,
        }
        comparison = _comparison_config(
            effective,
            model_descriptor,
            dataset_descriptor,
            prompt_set_sha256,
            sampling,
            args.num_spec_tokens,
        )
        if plugin_root is None:
            plugin_root = Path(__file__).resolve().parents[2]
        if core_root is None:
            import vllm

            core_root = Path(vllm.__file__).resolve().parents[1]
        result = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "dspark_pr_style_batch_throughput",
            "run_id": str(uuid.uuid4()),
            "runner": RUNNER,
            "mode": args.mode,
            "plugin_sha": _git_head(plugin_root),
            "core_sha": _git_head(core_root),
            "model": model_descriptor,
            "requested_engine_config": _requested_engine_config(args),
            "effective_engine_config": effective,
            "effective_config": effective,
            "resolved_kv_block_config": effective["resolved_kv_block_config"],
            "dataset": dataset_descriptor,
            "prompt_set_sha256": prompt_set_sha256,
            "prompt_identities": prompt_identities,
            "warmup_request_count": warmup_count,
            "measured_request_count": len(prompts),
            "sampling": sampling,
            "timing": {
                "boundary": "immediately_before_measured_batch_generate_to_synchronous_generate_return",
                "model_load_included": False,
                "dataset_load_included": False,
                "prompt_tokenization_included": False,
                "warmup_included": False,
                "explicit_device_synchronization": False,
                "elapsed_seconds": elapsed_seconds,
            },
            "throughput": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_output_tokens": total_output_tokens,
                "requests_per_second": len(prompts) / elapsed_seconds,
                "output_tokens_per_second": total_output_tokens / elapsed_seconds,
                "milliseconds_per_output_token": 1000.0 * elapsed_seconds / total_output_tokens,
            },
            "acceptance": acceptance,
            "metrics": {
                "api": "LLM.get_metrics/vllm.v1.metrics.reader.Counter+Vector",
                "delta_boundary": "post_warmup_snapshot_to_post_measured_snapshot",
                "warmup_snapshot": metrics_start,
                "measured_end_snapshot": metrics_end,
                "measured_delta": metric_delta,
            },
            "outputs": output_records,
            "comparison_config": comparison,
            "comparison_config_fingerprint": _sha256_bytes(_canonical_json_bytes(comparison)),
            "peak_npu_memory": None,
            "peak_npu_memory_provenance": "not exposed by LLM.get_metrics",
            "historical_error_count": None,
            "historical_error_count_provenance": "external merged-log scan required",
            "cleanup": {"engine_shutdown_complete": False},
        }
    except BaseException as primary_error:
        try:
            cleanup.shutdown()
        except BaseException as cleanup_error:
            raise primary_error from cleanup_error
        raise

    cleanup.shutdown()
    if not cleanup.complete:
        raise RuntimeError("Engine cleanup did not complete.")
    result["cleanup"]["engine_shutdown_complete"] = True
    return result


def _print_summary(result: Mapping[str, Any]) -> None:
    throughput = result["throughput"]
    acceptance = result["acceptance"]
    resolved_kv_blocks = result["resolved_kv_block_config"]
    print("-" * 60)
    print(f"runner: {result['runner']}")
    print(f"mode: {result['mode']}")
    print(f"total_requests: {result['measured_request_count']}")
    print(f"total_output_tokens: {throughput['total_output_tokens']}")
    print(f"requested base KV block size: {resolved_kv_blocks['requested_base_block_size']}")
    print(f"frontend ready-response KV block size: {resolved_kv_blocks['frontend_ready_block_size']}")
    print(f"scheduler KV block size: {resolved_kv_blocks['scheduler_block_size']}")
    print(f"KV group block sizes: {resolved_kv_blocks['group_block_sizes']}")
    print(f"num_drafts: {acceptance['num_drafts']}")
    print(f"num_draft_tokens: {acceptance['num_draft_tokens']}")
    print(f"num_accepted_tokens: {acceptance['num_accepted_candidate_tokens']}")
    print(f"mean accepted candidate length: {acceptance['accepted_candidate_tokens_per_verification']}")
    print(f"effective acceptance length: {acceptance['effective_acceptance_length']}")
    print(f"output token throughput: {throughput['output_tokens_per_second']:.6f} tok/s")
    print("-" * 60)
    for position, value in enumerate(acceptance["acceptance_per_position"]):
        print(f"acceptance at token {position}: {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a public-LLM MRV2 target-only or DSpark batch throughput benchmark."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--num-spec-tokens", type=int, default=5)
    parser.add_argument("--dataset-name", choices=("hf", "jsonl"), required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--prompt-field", default="question")
    parser.add_argument("--num-prompts", type=int, default=400)
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=400)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default="ascend")
    parser.add_argument("--tokenizer-mode", default="deepseek_v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-expert-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("num_spec_tokens", "warmup_prompts", "output_len", "tensor_parallel_size"):
        value = getattr(args, name)
        minimum = 0 if name == "warmup_prompts" else 1
        if value < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(args)
    _atomic_write_json(args.result_json.expanduser().resolve(), result)
    _print_summary(result)
    print(f"DSPARK_PR_STYLE_BENCHMARK_PASS={args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
