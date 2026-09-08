# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Freeze real, unique tasks and the exact DSV4 rendered inputs; never cycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import regex as re

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dspark.benchmark_dspark_acceptance import _canonical_json_bytes, _read_jsonl, _sha256_bytes, _sha256_file
from tools.dspark.build_m2_5a_dataset_assets import _load_tokenizer, _render_and_tokenize


def digest(value):
    return _sha256_bytes(_canonical_json_bytes(value))


def build_records(
    raw, tokenizer, *, count, max_input_tokens, source, revision, kind, prompt_field="prompt", id_field="task_id"
):
    if count < 1 or max_input_tokens < 1 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Positive counts and an immutable 40/64 hex source revision are required")
    candidates, dispositions, ids, prompts = [], [], set(), set()
    for index, task in enumerate(raw):
        case_id = f"{source}:{index}" if id_field == "@index" else str(task[id_field])
        if not case_id or case_id in ids:
            raise ValueError(f"Duplicate or empty case ID: {case_id!r}")
        ids.add(case_id)
        original = task[prompt_field]
        if not isinstance(original, str) or not original.strip():
            raise ValueError(f"Empty/non-text task {case_id}")
        tests = None
        if kind == "humaneval":
            # HumanEval import requires its real harness and canonical completion.
            for key in ("test", "entry_point", "canonical_solution"):
                if not isinstance(task.get(key), str) or not task[key].strip():
                    raise ValueError(f"{case_id}: missing real HumanEval field {key}")
            tests = {
                "kind": "python_function_v1",
                "test": task["test"],
                "entry_point": task["entry_point"],
                "reference": original + task["canonical_solution"],
            }
        elif kind == "code":
            tests = task.get("tests")
            if tests is not None:
                if tests.get("kind") != "python_function_v1" or any(
                    not isinstance(tests.get(key), str) or not tests[key].strip()
                    for key in ("test", "entry_point", "reference")
                ):
                    raise ValueError(f"{case_id}: unsupported/incomplete executable tests")
        if tests is not None and not re.fullmatch(r"[A-Za-z_]\w*", tests["entry_point"]):
            raise ValueError("Invalid entry point")
        user_prompt = original
        if kind in ("code", "humaneval"):
            user_prompt += "\n\nReturn the complete Python solution in one fenced python code block."
        rendered, tokens = _render_and_tokenize(tokenizer, [{"role": "user", "content": user_prompt}])
        prompt_hash = _sha256_bytes(rendered.encode())
        if prompt_hash in prompts:
            raise ValueError(f"Duplicate task prompt: {case_id}; replay is not enabled")
        prompts.add(prompt_hash)
        if len(tokens) > max_input_tokens:
            dispositions.append({"case_id": case_id, "reason": "rendered_input_too_long", "tokens": len(tokens)})
            continue
        record = {
            "schema_version": 1,
            "case_id": case_id,
            "source_index": index,
            "source": source,
            "source_revision": revision,
            "raw_task": task,
            "raw_task_sha256": digest(task),
            "prompt": rendered,
            "prompt_sha256": prompt_hash,
            "prompt_token_ids": tokens,
            "prompt_token_sha256": digest(tokens),
            "prompt_token_count": len(tokens),
            "tests": tests,
            "tests_sha256": digest(tests) if tests is not None else None,
            "sample_kind": "code" if kind in ("code", "humaneval") else "general",
            "replay_of": None,
        }
        record["record_sha256"] = digest(record)
        candidates.append(record)
    candidates.sort(key=lambda record: (digest(record["case_id"]), record["case_id"]))
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} unique legal samples; requested {count}; no truncation/repetition")
    selected = candidates[:count]
    dispositions.extend({"case_id": row["case_id"], "reason": "outside_frozen_selection"} for row in candidates[count:])
    return selected, dispositions


def read_manifest(path, count=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported input manifest")
    data_path = (path.parent / manifest["records_file"]).resolve()
    if data_path.parent != path.parent or _sha256_file(data_path) != manifest["records_sha256"]:
        raise ValueError("Input manifest path/hash mismatch")
    rows = _read_jsonl(data_path)
    if len(rows) != manifest["num_unique_samples"] or count is not None and not 0 < count <= len(rows):
        raise ValueError("Insufficient data or manifest count mismatch")
    ids, hashes = set(), set()
    for row in rows:
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        if digest(body) != row["record_sha256"] or digest(row["raw_task"]) != row["raw_task_sha256"]:
            raise ValueError("Record/raw task hash mismatch")
        if (
            digest(row["prompt_token_ids"]) != row["prompt_token_sha256"]
            or _sha256_bytes(row["prompt"].encode()) != row["prompt_sha256"]
            or len(row["prompt_token_ids"]) != row["prompt_token_count"]
            or not 0 < len(row["prompt_token_ids"]) <= manifest["max_input_tokens"]
            or any(type(token) is not int or token < 0 for token in row["prompt_token_ids"])
        ):
            raise ValueError("Prompt/token hash or length mismatch")
        if row["case_id"] in ids or row["prompt_sha256"] in hashes or row["replay_of"] is not None:
            raise ValueError("Repeated samples are not independent; replay unsupported")
        ids.add(row["case_id"])
        hashes.add(row["prompt_sha256"])
        if row["tests_sha256"] != (digest(row["tests"]) if row["tests"] is not None else None):
            raise ValueError("Tests hash mismatch")
    return manifest, rows[:count] if count is not None else rows, data_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--hf-repo")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--hf-config")
    parser.add_argument("--split", default="test")
    parser.add_argument("--kind", choices=("general", "code", "humaneval"), required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--id-field", default="task_id")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.kind != "general" and args.max_input_tokens > 2048:
        parser.error("Code workload inputs may not exceed 2048 tokens")
    if args.input_jsonl:
        raw = _read_jsonl(args.input_jsonl)
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.hf_repo, name=args.hf_config, split=args.split, revision=args.source_revision)
        raw = [dict(row) for row in dataset]
    records, dispositions = build_records(
        raw,
        _load_tokenizer(args.tokenizer),
        count=args.num_samples,
        max_input_tokens=args.max_input_tokens,
        source=args.source_name,
        revision=args.source_revision,
        kind=args.kind,
        prompt_field=args.prompt_field,
        id_field=args.id_field,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_dir / "source.jsonl"
    snapshot.write_bytes(b"".join(_canonical_json_bytes(row) for row in raw))
    data = args.output_dir / "requests.jsonl"
    data.write_bytes(b"".join(_canonical_json_bytes(row) for row in records))
    tokenizer_files = {
        str(path.relative_to(args.tokenizer)): _sha256_file(path)
        for pattern in ("*token*", "*vocab*", "*merges*", "*chat_template*", "config.json")
        for path in args.tokenizer.glob(pattern)
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "records_file": data.name,
        "records_sha256": _sha256_file(data),
        "source_snapshot_sha256": _sha256_file(snapshot),
        "source": args.source_name,
        "source_revision": args.source_revision,
        "num_unique_samples": len(records),
        "selection": "sha256(case_id) order after full-render length filtering; no replay",
        "max_input_tokens": args.max_input_tokens,
        "kind": args.kind,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_files_sha256": tokenizer_files,
        "rendering": "DSV4 chat template, generation prompt, add_special_tokens=False",
        "quality_test_tasks": sum(row["tests"] is not None for row in records),
    }
    (args.output_dir / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    (args.output_dir / "dispositions.json").write_bytes(_canonical_json_bytes(dispositions))
    read_manifest(args.output_dir / "manifest.json")
    print(args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
