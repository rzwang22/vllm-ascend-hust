# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXPECTED_SOURCE_REVISIONS = {
    "livecodebench": {
        "repo": "livecodebench/code_generation_lite",
        "revision": "7e47b68262a1da6d8634e205b69d88d978d53dc9",
        "split": "release_v6/test6.jsonl",
    },
    "sharegpt": {
        "repo": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "revision": "044ca94aec8d8cdee04973000738431161247677",
        "split": "ShareGPT_V3_unfiltered_cleaned_split.json",
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "revision": "cc7b047b6e5bb11b4f1af84efc572db110a51b3c",
        "split": "main/test",
    },
}

ASSET_FILES = (
    "livecodebench_64.jsonl",
    "sharegpt_64.jsonl",
    "gsm8k_64.jsonl",
    "synthetic_lengths.jsonl",
    "smoke_cases.jsonl",
    "full_cases.jsonl",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected text, got {type(value).__name__}.")
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("Token IDs must be non-negative integers.")
    return sha256_bytes(canonical_json_bytes(list(token_ids)))


def content_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    normalized = [
        {
            "role": normalized_text(message["role"]),
            "content": normalized_text(message["content"]),
        }
        for message in messages
    ]
    return sha256_bytes(canonical_json_bytes(normalized))


def stable_case_sort_key(case: Mapping[str, Any]) -> tuple[str, str]:
    return str(case["case_id"]), str(case["normalized_content_sha256"])


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("wb") as stream:
        for record in records:
            stream.write(canonical_json_bytes(record))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Partial JSONL artifact (missing final newline): {path}.")
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {path}:{line_number} must be an object.")
        records.append(value)
    return records


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_jsonl(temporary, records)
    temporary.replace(path)


def validate_case_record(case: Mapping[str, Any]) -> None:
    required = {
        "dataset",
        "source_repo",
        "source_revision",
        "source_split",
        "case_id",
        "normalized_content_sha256",
        "rendered_prompt_sha256",
        "ordered_prompt_token_sha256",
        "prompt_token_ids",
        "prompt_token_count",
        "output_cap",
        "ignore_eos",
        "selection_rank",
        "selection_reason",
    }
    missing = sorted(required.difference(case))
    if missing:
        raise ValueError(f"Case {case.get('case_id')!r} is missing fields: {missing}.")
    prompt_token_ids = case["prompt_token_ids"]
    if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
        raise TypeError("prompt_token_ids must be a non-empty list.")
    if case["prompt_token_count"] != len(prompt_token_ids):
        raise ValueError(f"Case {case['case_id']!r} prompt token count is inconsistent.")
    if case["ordered_prompt_token_sha256"] != token_ids_sha256(prompt_token_ids):
        raise ValueError(f"Case {case['case_id']!r} prompt token hash is inconsistent.")
    messages = case.get("messages")
    if not isinstance(messages, list) or case["normalized_content_sha256"] != content_sha256(messages):
        raise ValueError(f"Case {case['case_id']!r} normalized content hash is inconsistent.")
    if not isinstance(case["rendered_prompt_sha256"], str) or len(case["rendered_prompt_sha256"]) != 64:
        raise ValueError(f"Case {case['case_id']!r} rendered prompt hash is invalid.")
    if type(case["output_cap"]) is not int or case["output_cap"] <= 0:
        raise ValueError(f"Case {case['case_id']!r} has an invalid output cap.")
    if type(case["ignore_eos"]) is not bool:
        raise TypeError(f"Case {case['case_id']!r} ignore_eos must be boolean.")


def verify_asset_bundle(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read manifest: {manifest_path}.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or invalid M2.5A manifest schema.")
    root = manifest_path.parent
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Manifest artifacts must be an object.")
    for filename in ASSET_FILES:
        expected = artifacts.get(filename)
        path = root / filename
        if not isinstance(expected, dict) or not path.is_file():
            raise ValueError(f"Manifest artifact is missing: {filename}.")
        actual_hash = sha256_file(path)
        if expected.get("sha256") != actual_hash:
            raise ValueError(f"Artifact hash mismatch for {filename}.")
        records = read_jsonl(path)
        if expected.get("record_count") != len(records):
            raise ValueError(f"Artifact record count mismatch for {filename}.")
        for case in records:
            validate_case_record(case)
    full_cases = read_jsonl(root / "full_cases.jsonl")
    smoke_cases = read_jsonl(root / "smoke_cases.jsonl")
    full_ids = [case["case_id"] for case in full_cases]
    smoke_ids = [case["case_id"] for case in smoke_cases]
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or profiles.get("full") != full_ids or profiles.get("smoke") != smoke_ids:
        raise ValueError("Manifest profile order differs from the frozen case artifacts.")
    if len(full_ids) != len(set(full_ids)) or len(smoke_ids) != len(set(smoke_ids)):
        raise ValueError("Frozen profiles contain duplicate case IDs.")
    if not set(smoke_ids).issubset(full_ids):
        raise ValueError("Smoke cases must be a deterministic subset of full cases.")
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise ValueError("SHA256SUMS is missing.")
    parsed_sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, filename = line.partition("  ")
        if not separator or filename in parsed_sums:
            raise ValueError("SHA256SUMS contains a malformed or duplicate entry.")
        parsed_sums[filename] = digest
    expected_sum_files = {"manifest.json", *ASSET_FILES}
    if set(parsed_sums) != expected_sum_files:
        raise ValueError("SHA256SUMS does not describe the complete frozen bundle.")
    for filename, expected_hash in parsed_sums.items():
        if sha256_file(root / filename) != expected_hash:
            raise ValueError(f"SHA256SUMS mismatch for {filename}.")
    return manifest


def load_profile_cases(manifest_path: Path, profile: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = verify_asset_bundle(manifest_path)
    if profile not in {"smoke", "full"}:
        raise ValueError("DSPARK_M25A_PROFILE must be 'smoke' or 'full'.")
    cases = read_jsonl(manifest_path.expanduser().resolve().parent / f"{profile}_cases.jsonl")
    expected_count = 10 if profile == "smoke" else 208
    if len(cases) != expected_count:
        raise ValueError(f"The {profile} profile must contain {expected_count} cases, got {len(cases)}.")
    return manifest, cases


def build_execution_plan(
    cases: Sequence[Mapping[str, Any]],
    lifecycle_repeat: int,
) -> list[dict[str, Any]]:
    if lifecycle_repeat not in {1, 3}:
        raise ValueError("DSPARK_M25A_LIFECYCLE_REPEAT must be 1 or 3.")
    if lifecycle_repeat == 3 and len(cases) < 2:
        raise ValueError("Three-pass lifecycle validation requires at least two distinct cases.")
    plan: list[dict[str, Any]] = []
    sequence_index = 0
    for repeat_index in range(lifecycle_repeat):
        for profile_index, case in enumerate(cases):
            record = dict(case)
            record["lifecycle_repeat"] = repeat_index
            record["profile_case_index"] = profile_index
            record["request_sequence_index"] = sequence_index
            plan.append(record)
            sequence_index += 1
    return plan
