# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Freeze rendered tasks or import exact request instances; never synthesize repeats."""

from __future__ import annotations

import argparse
import json
import shutil
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
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError("Unsupported input manifest")
    data_path = (path.parent / manifest["records_file"]).resolve()
    if data_path.parent != path.parent or _sha256_file(data_path) != manifest["records_sha256"]:
        raise ValueError("Input manifest path/hash mismatch")
    rows = _read_jsonl(data_path)
    repeated = manifest["schema_version"] == 2
    declared = manifest["request_instance_count"] if repeated else manifest["num_unique_samples"]
    if len(rows) != declared or count is not None and not 0 < count <= len(rows):
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
        if row["case_id"] in ids or (not repeated and (row["prompt_sha256"] in hashes or row["replay_of"] is not None)):
            raise ValueError("Repeated samples are not independent; replay unsupported")
        ids.add(row["case_id"])
        hashes.add(row["prompt_sha256"])
        if row["tests_sha256"] != (digest(row["tests"]) if row["tests"] is not None else None):
            raise ValueError("Tests hash mismatch")
    if repeated:
        validate_repeated_manifest(path, manifest, rows)
    return manifest, rows[:count] if count is not None else rows, data_path


def input_population(manifest, rows):
    """Selected request population, distinct from independent engine repeats."""
    unique = len({r["prompt_token_sha256"] for r in rows})
    return {
        "request_instance_count": len(rows),
        "unique_prompt_count": unique,
        "label": f"{len(rows)} request instances / {unique} unique prompts",
        "repeated_prompt_policy": manifest.get("repeated_prompt_policy", "reject"),
        "ordered_token_sequences_sha256": digest([r["prompt_token_ids"] for r in rows]),
        "quality_scope": "prompt repetitions are not independent quality samples",
    }


def repetition_mapping(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["prompt_token_sha256"], []).append(row["request_instance_id"])
    return groups


def validate_repeated_manifest(path, manifest, rows):
    if (
        manifest.get("repeated_prompt_policy") != "preserve_source_occurrences"
        or manifest.get("allow_repeated_prompts") is not True
        or "num_unique_samples" in manifest
        or manifest.get("kind") != "general"
        or type(manifest.get("request_instance_count")) is not int
        or type(manifest.get("unique_prompt_count")) is not int
    ):
        raise ValueError("Explicit repeated manifest policy/count schema required")
    source_path = (path.parent / manifest["source_snapshot_file"]).resolve()
    if (
        source_path.parent != path.parent
        or _sha256_file(source_path) != manifest["source_snapshot_sha256"]
        or manifest["source_snapshot_sha256"] != manifest["original_source_file_sha256"]
    ):
        raise ValueError("Original source snapshot path/hash mismatch")
    source = _read_jsonl(source_path)
    if len(source) < len(rows):
        raise ValueError("Source lacks declared request instances")
    first, occurrences = {}, {}
    for index, row in enumerate(rows):
        instance = f"request:{manifest['original_source_file_sha256']}:{index}"
        hashed = row["prompt_token_sha256"]
        expected_case = source[index].get(manifest["source_id_field"])
        if (
            row.get("schema_version") != 2
            or row.get("request_instance_id") != instance
            or row["case_id"] != instance
            or row["source_index"] != index
            or row["source"] != manifest["source"]
            or row["source_revision"] != manifest["source_revision"]
            or row["sample_kind"] != "general"
            or row["tests"] is not None
            or row["raw_task"] != source[index]
            or row.get("original_case_id") != expected_case
            or row["prompt_token_ids"] != source[index][manifest["source_token_field"]]
            or row["replay_of"] != first.get(hashed)
            or row.get("prompt_occurrence_index") != occurrences.get(hashed, 0)
        ):
            raise ValueError("Invalid request instance identity, source sequence or repetition reference")
        first.setdefault(hashed, instance)
        occurrences[hashed] = occurrences.get(hashed, 0) + 1
    if (
        manifest["unique_prompt_count"] != len(first)
        or manifest["repeated_prompt_mapping"] != repetition_mapping(rows)
        or manifest["ordered_token_sequences_sha256"] != digest([r["prompt_token_ids"] for r in rows])
    ):
        raise ValueError("Repeated prompt mapping/count/sequence hash mismatch")


def copy_manifest_assets(path, directory):
    """Copy the full verifiable input, even when a run selects only its prefix."""
    path = Path(path)
    manifest, _, records = read_manifest(path)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, directory / "manifest.json")
    shutil.copyfile(records, directory / manifest["records_file"])
    if manifest["schema_version"] == 2:
        name = manifest["source_snapshot_file"]
        shutil.copyfile(path.parent / name, directory / name)


def import_frozen_records(
    raw,
    tokenizer,
    *,
    count,
    max_input_tokens,
    source,
    revision,
    token_field,
    id_field,
    allow_repeated_prompts=False,
    source_file_sha256=None,
):
    """Preserve existing token IDs and order; decoded text is labelled, not re-rendered."""
    if count < 1 or len(raw) < count:
        raise ValueError("Insufficient frozen requests; no replay")
    if allow_repeated_prompts and not re.fullmatch(r"[0-9a-f]{64}", source_file_sha256 or ""):
        raise ValueError("Repeated input import requires the original file SHA256")
    rows = []
    seen = {}
    occurrences = {}
    for index, task in enumerate(raw[:count]):
        tokens = task.get(token_field)
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= max_input_tokens
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("Invalid frozen token IDs or length; no truncation")
        hashed = digest(tokens)
        if hashed in seen and not allow_repeated_prompts:
            raise ValueError("Duplicate frozen token sequence; use explicit --allow-repeated-prompts")
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        row = {
            "schema_version": 1,
            "case_id": str(task.get(id_field, f"{revision}:{index}")),
            "source_index": index,
            "source": source,
            "source_revision": revision,
            "raw_task": task,
            "raw_task_sha256": digest(task),
            "prompt": text,
            "prompt_sha256": _sha256_bytes(text.encode()),
            "prompt_token_ids": list(tokens),
            "prompt_token_sha256": hashed,
            "prompt_token_count": len(tokens),
            "prompt_text_origin": "decoded frozen token IDs; not re-rendered or re-tokenized",
            "tests": None,
            "tests_sha256": None,
            "sample_kind": "general",
            "replay_of": None,
        }
        if allow_repeated_prompts:
            instance = f"request:{source_file_sha256}:{index}"
            row.update(
                schema_version=2,
                case_id=instance,
                request_instance_id=instance,
                original_case_id=task.get(id_field),
                replay_of=seen.get(hashed),
                prompt_occurrence_index=occurrences.get(hashed, 0),
            )
        seen.setdefault(hashed, row["case_id"])
        occurrences[hashed] = occurrences.get(hashed, 0) + 1
        row["record_sha256"] = digest(row)
        rows.append(row)
    return rows, []


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
    parser.add_argument("--frozen-token-field", help="Import final token IDs verbatim, in source order; general only")
    parser.add_argument(
        "--allow-repeated-prompts",
        action="store_true",
        help="Explicitly preserve existing frozen prompt occurrences; never generate extra requests",
    )
    parser.add_argument("--expected-source-sha256", help="Required for frozen token import")
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
    if args.allow_repeated_prompts and not args.frozen_token_field:
        parser.error("--allow-repeated-prompts requires --frozen-token-field")
    if args.input_jsonl:
        raw = _read_jsonl(args.input_jsonl)
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.hf_repo, name=args.hf_config, split=args.split, revision=args.source_revision)
        raw = [dict(row) for row in dataset]
    if args.frozen_token_field:
        if (
            args.kind != "general"
            or not args.input_jsonl
            or not args.expected_source_sha256
            or _sha256_file(args.input_jsonl) != args.expected_source_sha256
        ):
            parser.error("Frozen import requires general JSONL and matching --expected-source-sha256")
        records, dispositions = import_frozen_records(
            raw,
            _load_tokenizer(args.tokenizer),
            count=args.num_samples,
            max_input_tokens=args.max_input_tokens,
            source=args.source_name,
            revision=args.source_revision,
            token_field=args.frozen_token_field,
            id_field=args.id_field,
            allow_repeated_prompts=args.allow_repeated_prompts,
            source_file_sha256=args.expected_source_sha256,
        )
    else:
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
    if args.allow_repeated_prompts:
        shutil.copyfile(args.input_jsonl, snapshot)  # Preserve original bytes, including JSONL formatting.
        if _sha256_file(snapshot) != args.expected_source_sha256:
            raise ValueError("Source changed during import")
    else:
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
        "selection": "original frozen token order; no replay"
        if args.frozen_token_field
        else "sha256(case_id) order after full-render length filtering; no replay",
        "original_source_file_sha256": _sha256_file(args.input_jsonl) if args.input_jsonl else None,
        "max_input_tokens": args.max_input_tokens,
        "kind": args.kind,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_files_sha256": tokenizer_files,
        "rendering": "verbatim frozen token IDs; display text decoded only"
        if args.frozen_token_field
        else "DSV4 chat template, generation prompt, add_special_tokens=False",
        "quality_test_tasks": sum(row["tests"] is not None for row in records),
    }
    if args.allow_repeated_prompts:
        manifest.pop("num_unique_samples")
        manifest.update(
            schema_version=2,
            allow_repeated_prompts=True,
            request_instance_count=len(records),
            unique_prompt_count=len(repetition_mapping(records)),
            repeated_prompt_policy="preserve_source_occurrences",
            repeated_prompt_mapping=repetition_mapping(records),
            source_snapshot_file=snapshot.name,
            source_token_field=args.frozen_token_field,
            source_id_field=args.id_field,
            ordered_token_sequences_sha256=digest([r["prompt_token_ids"] for r in records]),
            selection="original first N request instances; existing repetitions preserved",
        )
    (args.output_dir / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    (args.output_dir / "dispositions.json").write_bytes(_canonical_json_bytes(dispositions))
    read_manifest(args.output_dir / "manifest.json")
    print(args.output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
