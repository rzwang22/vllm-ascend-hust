#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dspark.m2_5a_common import (
    ASSET_FILES,
    EXPECTED_SOURCE_REVISIONS,
    SCHEMA_VERSION,
    canonical_json_bytes,
    content_sha256,
    normalized_text,
    sha256_bytes,
    sha256_file,
    stable_case_sort_key,
    token_ids_sha256,
    verify_asset_bundle,
    write_json,
    write_jsonl,
)

TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "tokenizer.model",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic offline assets for the M2.5A DSpark matrix.")
    parser.add_argument("--livecodebench", type=Path)
    parser.add_argument("--sharegpt", type=Path)
    parser.add_argument("--gsm8k", type=Path)
    parser.add_argument("--source-revisions", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only is None:
        missing = [
            name
            for name in ("livecodebench", "sharegpt", "gsm8k", "source_revisions", "tokenizer", "output_dir")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("asset construction requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    return args


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Building assets requires transformers, but verification does not.") from exc
    return AutoTokenizer.from_pretrained(
        str(path.expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )


def _render_and_tokenize(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> tuple[str, list[int]]:
    normalized_messages = [
        {"role": normalized_text(message["role"]), "content": normalized_text(message["content"])}
        for message in messages
    ]
    rendered = tokenizer.apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("The checkpoint chat template must render text.")
    rendered = normalized_text(rendered)
    encoded = tokenizer(rendered, add_special_tokens=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Tokenizer unexpectedly returned a batch for one rendered prompt.")
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    if not token_ids:
        raise ValueError("The rendered prompt produced no token IDs.")
    return rendered, token_ids


def _read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value if isinstance(value, list) else value.get("data")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Expected a list of objects in {path}.")
    return records


def _read_gsm8k(path: Path) -> list[dict[str, Any]]:
    if path.suffix in {".json", ".jsonl"}:
        return _read_json_or_jsonl(path)
    if path.suffix != ".parquet":
        raise ValueError("GSM8K input must be the fixed parquet source file.")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Reading the fixed GSM8K parquet file requires pyarrow.") from exc
    return parquet.read_table(path).to_pylist()


def _source_contracts(path: Path, source_paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read source revision contract: {path}.") from exc
    if not isinstance(raw, dict):
        raise ValueError("source_revisions.json must contain an object.")
    contracts: dict[str, dict[str, str]] = {}
    for dataset, expected in EXPECTED_SOURCE_REVISIONS.items():
        value = raw.get(dataset)
        if not isinstance(value, dict):
            raise ValueError(f"source_revisions.json is missing {dataset!r}.")
        repo = value.get("repo")
        revision = value.get("revision")
        source_hash = value.get("raw_file_sha256", value.get("sha256"))
        if repo != expected["repo"] or revision != expected["revision"]:
            raise ValueError(f"{dataset} source must be frozen at {expected['repo']}@{expected['revision']}.")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise ValueError(f"{dataset} must provide raw_file_sha256/sha256.")
        actual_hash = sha256_file(source_paths[dataset])
        if actual_hash != source_hash:
            raise ValueError(f"{dataset} raw source hash mismatch.")
        contracts[dataset] = {
            "repo": repo,
            "revision": revision,
            "split": expected["split"],
            "raw_file": source_paths[dataset].name,
            "raw_file_sha256": source_hash,
        }
    return contracts


def _case(
    *,
    tokenizer: Any,
    dataset: str,
    source: Mapping[str, str],
    case_id: str,
    messages: Sequence[Mapping[str, str]],
    output_cap: int,
    ignore_eos: bool,
    selection_reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rendered, token_ids = _render_and_tokenize(tokenizer, messages)
    return {
        "dataset": dataset,
        "source_repo": source["repo"],
        "source_revision": source["revision"],
        "source_split": source["split"],
        "case_id": normalized_text(case_id),
        "messages": [
            {"role": normalized_text(message["role"]), "content": normalized_text(message["content"])}
            for message in messages
        ],
        "normalized_content_sha256": content_sha256(messages),
        "rendered_prompt_sha256": sha256_bytes(rendered.encode()),
        "ordered_prompt_token_sha256": token_ids_sha256(token_ids),
        "prompt_token_ids": token_ids,
        "prompt_token_count": len(token_ids),
        "output_cap": output_cap,
        "ignore_eos": ignore_eos,
        "selection_rank": -1,
        "selection_reason": selection_reason,
        "metadata": dict(metadata or {}),
    }


def _first_text(record: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return normalized_text(value)
    raise ValueError(f"Source record has none of the required text fields: {names}.")


def _stable_select(candidates: Iterable[dict[str, Any]], count: int, dataset: str) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=stable_case_sort_key)
    if len(ordered) < count:
        raise ValueError(f"{dataset} yielded only {len(ordered)} eligible cases; {count} are required.")
    selected = ordered[:count]
    for rank, case in enumerate(selected):
        case["selection_rank"] = rank
    return selected


def build_livecodebench(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, source: Mapping[str, str]
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        prompt = _first_text(record, ("question_content", "prompt", "question", "text"))
        raw_id = record.get("question_id", record.get("task_id", record.get("id")))
        case_id = str(raw_id) if raw_id is not None else content_sha256([{"role": "user", "content": prompt}])[:24]
        metadata = {
            name: record[name]
            for name in ("difficulty", "platform", "contest_date", "date", "starter_code", "test")
            if name in record
        }
        case = _case(
            tokenizer=tokenizer,
            dataset="livecodebench",
            source=source,
            case_id=f"livecodebench:{case_id}",
            messages=[{"role": "user", "content": prompt}],
            output_cap=1024,
            ignore_eos=False,
            selection_reason="stable_id_hash_order;rendered_prompt_tokens<=2048",
            metadata=metadata,
        )
        if case["prompt_token_count"] <= 2048:
            candidates.append(case)
    return _stable_select(candidates, 64, "LiveCodeBench")


def _sharegpt_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_messages = record.get("conversations", record.get("messages"))
    if not isinstance(raw_messages, list):
        raise ValueError("ShareGPT record has no conversations list.")
    result = []
    expected = "user"
    for raw in raw_messages:
        if not isinstance(raw, Mapping):
            break
        source_role = raw.get("from", raw.get("role"))
        role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}.get(source_role)
        content = raw.get("value", raw.get("content"))
        if role != expected or not isinstance(content, str) or not content.strip():
            break
        result.append({"role": role, "content": normalized_text(content)})
        expected = "assistant" if expected == "user" else "user"
    return result


def build_sharegpt(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, source: Mapping[str, str]
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"1024": [], "4096": []}
    ranges = {"1024": (922, 1126, 128), "4096": (3686, 4506, 256)}
    for record in records:
        messages = _sharegpt_messages(record)
        base_id = str(record.get("id", record.get("conversation_id", content_sha256(messages)[:24])))
        for prefix_length in range(1, len(messages) + 1, 2):
            prefix = messages[:prefix_length]
            for bucket, (lower, upper, output_cap) in ranges.items():
                case = _case(
                    tokenizer=tokenizer,
                    dataset="sharegpt",
                    source=source,
                    case_id=f"sharegpt:{base_id}:turns-{prefix_length}",
                    messages=prefix,
                    output_cap=output_cap,
                    ignore_eos=False,
                    selection_reason=f"valid_user_ending_prefix;{bucket}_token_bucket;stable_id_hash_order",
                    metadata={"conversation_turns": prefix_length, "length_bucket": bucket},
                )
                if lower <= case["prompt_token_count"] <= upper:
                    buckets[bucket].append(case)
    selected = _stable_select(buckets["1024"], 32, "ShareGPT 1024-token bucket")
    selected.extend(_stable_select(buckets["4096"], 32, "ShareGPT 4096-token bucket"))
    for rank, case in enumerate(selected):
        case["selection_rank"] = rank
    return selected


def build_gsm8k(
    records: Sequence[Mapping[str, Any]], tokenizer: Any, source: Mapping[str, str]
) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        question = _first_text(record, ("question", "prompt"))
        answer = _first_text(record, ("answer",))
        content_hash = content_sha256([{"role": "user", "content": question}])
        final_answer = answer.rsplit("####", 1)[-1].strip() if "####" in answer else answer.strip()
        case = _case(
            tokenizer=tokenizer,
            dataset="gsm8k",
            source=source,
            case_id=f"gsm8k:{record.get('id', content_hash[:24])}",
            messages=[{"role": "user", "content": question}],
            output_cap=512,
            ignore_eos=False,
            selection_reason="stable_id_hash_order;rendered_prompt_tokens<=1024",
            metadata={"answer": answer, "final_answer": final_answer},
        )
        if case["prompt_token_count"] <= 1024:
            candidates.append(case)
    return _stable_select(candidates, 64, "GSM8K")


def _fit_synthetic_prompt(tokenizer: Any, target_tokens: int, case_index: int) -> tuple[str, list[int]]:
    base = (
        f"Deterministic DSpark boundary case {target_tokens}-{case_index}.\n"
        "Read the following repeated audit words and continue with concise numbered observations.\n"
    )
    units = (" audit", " token", " datum", " check")
    for unit in units:
        cache: dict[int, tuple[str, list[int]]] = {}

        def rendered(
            repetitions: int,
            unit: str = unit,
            cache: dict[int, tuple[str, list[int]]] = cache,
        ) -> tuple[str, list[int]]:
            if repetitions not in cache:
                content = base + unit * repetitions
                _, token_ids = _render_and_tokenize(tokenizer, [{"role": "user", "content": content}])
                cache[repetitions] = content, token_ids
            return cache[repetitions]

        low, high = 0, max(target_tokens * 2, 256)
        while low <= high:
            middle = (low + high) // 2
            content, token_ids = rendered(middle)
            if len(token_ids) == target_tokens:
                return content, token_ids
            if len(token_ids) < target_tokens:
                low = middle + 1
            else:
                high = middle - 1
        for repetitions in range(max(0, high - 8), low + 9):
            content, token_ids = rendered(repetitions)
            if len(token_ids) == target_tokens:
                return content, token_ids
    raise ValueError(f"Unable to construct a reviewable synthetic prompt with exactly {target_tokens} tokens.")


def build_synthetic(tokenizer: Any) -> list[dict[str, Any]]:
    source = {"repo": "repository-local/deterministic-hybrid", "revision": "m2.5a-v1", "split": "synthetic"}
    output_caps = {128: 16, 1024: 256, 2048: 512, 4096: 1024}
    result = []
    for target_tokens, output_cap in output_caps.items():
        for case_index in range(4):
            content, _ = _fit_synthetic_prompt(tokenizer, target_tokens, case_index)
            case = _case(
                tokenizer=tokenizer,
                dataset="synthetic",
                source=source,
                case_id=f"synthetic:{target_tokens}:{case_index}",
                messages=[{"role": "user", "content": content}],
                output_cap=output_cap,
                ignore_eos=True,
                selection_reason="deterministic_reviewable_text;exact_rendered_token_boundary",
                metadata={"synthetic": True, "target_prompt_tokens": target_tokens},
            )
            if case["prompt_token_count"] != target_tokens:
                raise AssertionError("Synthetic prompt builder returned the wrong token count.")
            case["selection_rank"] = len(result)
            result.append(case)
    return result


def _tokenizer_identity(tokenizer_path: Path, tokenizer: Any) -> dict[str, Any]:
    files = {
        filename: sha256_file(tokenizer_path / filename)
        for filename in TOKENIZER_FILES
        if (tokenizer_path / filename).is_file()
    }
    if not files:
        raise ValueError("The tokenizer directory contains none of the frozen tokenizer identity files.")
    chat_template = getattr(tokenizer, "chat_template", None)
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "files": files,
        "chat_template_sha256": sha256_bytes(normalized_text(chat_template or "").encode()),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }


def _distribution(cases: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(case.get("metadata", {}).get(field, "unknown")) for case in cases).items()))


def build_assets(args: argparse.Namespace, tokenizer: Any | None = None) -> Path:
    if args.seed != 0:
        raise ValueError("M2.5A uses the frozen deterministic seed 0.")
    source_paths = {
        "livecodebench": args.livecodebench.expanduser().resolve(),
        "sharegpt": args.sharegpt.expanduser().resolve(),
        "gsm8k": args.gsm8k.expanduser().resolve(),
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise ValueError("All fixed source files must exist locally; downloads are forbidden.")
    contracts = _source_contracts(args.source_revisions.expanduser().resolve(), source_paths)
    tokenizer_path = args.tokenizer.expanduser().resolve()
    if not tokenizer_path.is_dir():
        raise ValueError("--tokenizer must point to the local checkpoint tokenizer directory.")
    tokenizer = tokenizer or _load_tokenizer(tokenizer_path)

    livecodebench = build_livecodebench(
        _read_json_or_jsonl(source_paths["livecodebench"]), tokenizer, contracts["livecodebench"]
    )
    sharegpt = build_sharegpt(_read_json_or_jsonl(source_paths["sharegpt"]), tokenizer, contracts["sharegpt"])
    gsm8k = build_gsm8k(_read_gsm8k(source_paths["gsm8k"]), tokenizer, contracts["gsm8k"])
    synthetic = build_synthetic(tokenizer)
    full = [*livecodebench, *sharegpt, *gsm8k, *synthetic]
    smoke = [*livecodebench[:2], *sharegpt[:2], *gsm8k[:2], synthetic[0], synthetic[4], synthetic[8], synthetic[12]]
    if not {case["case_id"] for case in smoke}.issubset({case["case_id"] for case in full}):
        raise AssertionError("Smoke profile must be a subset of full profile.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_by_file = {
        "livecodebench_64.jsonl": livecodebench,
        "sharegpt_64.jsonl": sharegpt,
        "gsm8k_64.jsonl": gsm8k,
        "synthetic_lengths.jsonl": synthetic,
        "smoke_cases.jsonl": smoke,
        "full_cases.jsonl": full,
    }
    for filename, records in records_by_file.items():
        write_jsonl(output_dir / filename, records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": 0,
        "sources": contracts,
        "source_revisions_sha256": sha256_file(args.source_revisions.expanduser().resolve()),
        "tokenizer": _tokenizer_identity(tokenizer_path, tokenizer),
        "rendering": {"apply_chat_template": True, "add_generation_prompt": True, "add_special_tokens": False},
        "profiles": {"smoke": [case["case_id"] for case in smoke], "full": [case["case_id"] for case in full]},
        "distributions": {
            "livecodebench_difficulty": _distribution(livecodebench, "difficulty"),
            "livecodebench_platform": _distribution(livecodebench, "platform"),
            "sharegpt_input_length": dict(
                sorted(Counter(case["metadata"]["length_bucket"] for case in sharegpt).items())
            ),
            "prompt_token_ranges": {
                dataset: [
                    min(case["prompt_token_count"] for case in cases),
                    max(case["prompt_token_count"] for case in cases),
                ]
                for dataset, cases in (
                    ("livecodebench", livecodebench),
                    ("sharegpt", sharegpt),
                    ("gsm8k", gsm8k),
                    ("synthetic", synthetic),
                )
            },
        },
        "artifacts": {
            filename: {"record_count": len(records), "sha256": sha256_file(output_dir / filename)}
            for filename, records in records_by_file.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    sum_files = ("manifest.json", *ASSET_FILES)
    sums = "".join(f"{sha256_file(output_dir / filename)}  {filename}\n" for filename in sum_files)
    (output_dir / "SHA256SUMS").write_bytes(sums.encode())
    verify_asset_bundle(manifest_path)
    return manifest_path


def main() -> int:
    args = _parse_args()
    if args.verify_only is not None:
        manifest = verify_asset_bundle(args.verify_only)
        print(
            "M2_5A_ASSET_VERIFY_PASS="
            + canonical_json_bytes(
                {
                    "manifest": str(args.verify_only),
                    "profiles": {key: len(value) for key, value in manifest["profiles"].items()},
                }
            )
            .decode()
            .strip()
        )
        return 0
    manifest_path = build_assets(args)
    print(f"M2_5A_ASSET_BUILD_PASS={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
