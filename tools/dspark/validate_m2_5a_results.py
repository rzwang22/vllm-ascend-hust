#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dspark.m2_5a_common import (
    read_jsonl,
    sha256_file,
    token_ids_sha256,
    verify_asset_bundle,
)

EXPECTED_RANKS = 8
HISTORICAL_ERROR_MARKERS = (
    "507057",
    "MTE DDR address out of range",
    "aclnnScatterNdUpdateV2",
    "StorageShape got",
    "cmp_block_size",
    "block_size should be",
    "ChildFailedError",
    "DSPARK_PREPARE_FAILURE",
    "DSPARK_DRAFT_FORWARD_FAILURE",
    "DSPARK_MARKOV_FAILURE",
    "DSPARK_M2_4A_FAILURE",
    "DSPARK_M2_4B_FAILURE",
    "DSPARK_M2_5A_FAILURE",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exact target-only/DSpark M2.5A result artifacts.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-results", type=Path, required=True)
    parser.add_argument("--dspark-results", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, default=EXPECTED_RANKS)
    return parser.parse_args()


def _rank_files(root: Path, expected_ranks: int) -> dict[int, Path]:
    root = root.expanduser().resolve()
    failures = sorted(root.glob("rank-*.failure.json"))
    partials = sorted(root.glob(".*.tmp"))
    if failures or partials:
        raise ValueError(f"Result directory contains failure/partial artifacts: {failures + partials}.")
    files: dict[int, Path] = {}
    for path in sorted(root.glob("rank-*.jsonl")):
        try:
            rank = int(path.stem.removeprefix("rank-"))
        except ValueError as exc:
            raise ValueError(f"Invalid rank artifact name: {path.name}.") from exc
        if rank in files:
            raise ValueError(f"Duplicate rank artifact for rank {rank}.")
        files[rank] = path
    if set(files) != set(range(expected_ranks)):
        raise ValueError(f"Expected rank artifacts 0..{expected_ranks - 1}, got {sorted(files)}.")
    for path in files.values():
        digest_path = path.with_suffix(path.suffix + ".sha256")
        if not digest_path.is_file() or digest_path.read_text(encoding="utf-8").strip() != sha256_file(path):
            raise ValueError(f"Missing or invalid artifact digest: {digest_path}.")
    return files


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(record["request_sequence_index"]),
        int(record["lifecycle_repeat"]),
        str(record["case_id"]),
    )


def _validate_record(record: Mapping[str, Any], *, mode: str, manifest_hash: str, rank: int) -> None:
    required = {
        "rank",
        "mode",
        "dataset",
        "case_id",
        "lifecycle_repeat",
        "request_sequence_index",
        "proposal_epoch_start",
        "proposal_epoch_end",
        "consumer_epoch_start",
        "consumer_epoch_end",
        "request_id",
        "prompt_token_count",
        "prompt_token_sha256",
        "output_cap",
        "ignore_eos",
        "output_token_count",
        "output_token_ids",
        "output_token_sha256",
        "stop_reason",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "completed_rounds",
        "proposal_generated_count",
        "proposal_installed_count",
        "proposal_consumed_count",
        "terminal_discarded_proposal_count",
        "terminal_partial_commit",
        "post_finish_target_forward_count",
        "post_finish_verification_count",
        "cleanup_complete",
        "state_isolation_verified",
        "historical_error_count",
        "manifest_sha256",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Result record is missing fields: {missing}.")
    if record["rank"] != rank or record["mode"] != mode:
        raise ValueError("Result rank/mode ownership is inconsistent.")
    if record["manifest_sha256"] != manifest_hash:
        raise ValueError("Target/result manifest hash does not match the frozen input.")
    output_ids = record["output_token_ids"]
    if not isinstance(output_ids, list) or record["output_token_count"] != len(output_ids):
        raise ValueError("Result output token count is inconsistent.")
    if record["output_token_sha256"] != token_ids_sha256(output_ids):
        raise ValueError("Result output token hash is inconsistent.")
    if record["usage_prompt_tokens"] != record["prompt_token_count"]:
        raise ValueError("Prompt usage accounting is inconsistent.")
    if record["usage_completion_tokens"] != record["output_token_count"]:
        raise ValueError("Completion usage accounting is inconsistent.")
    if record["post_finish_target_forward_count"] != 0 or record["post_finish_verification_count"] != 0:
        raise ValueError("A finished request executed another target/verification step.")
    if record["cleanup_complete"] is not True or record["state_isolation_verified"] is not True:
        raise ValueError("Per-request cleanup/state isolation was not verified.")
    if record["historical_error_count"] != 0:
        raise ValueError("A historical NPU/runtime error was recorded.")
    if record["ignore_eos"] and record["output_token_count"] != record["output_cap"]:
        raise ValueError("Synthetic ignore_eos request did not reach its fixed output cap.")
    if mode == "target_only" and any(
        record[field] != 0
        for field in (
            "completed_rounds",
            "proposal_generated_count",
            "proposal_installed_count",
            "proposal_consumed_count",
            "terminal_discarded_proposal_count",
        )
    ):
        raise ValueError("Target-only result contains DSpark proposal activity.")


def load_results(
    root: Path,
    *,
    mode: str,
    manifest_hash: str,
    expected_ranks: int = EXPECTED_RANKS,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for rank, path in _rank_files(root, expected_ranks).items():
        records = read_jsonl(path)
        keys = [_record_key(record) for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Rank {rank} contains duplicate result cases.")
        if [key[0] for key in keys] != list(range(len(keys))):
            raise ValueError(f"Rank {rank} result order is incomplete or non-canonical.")
        for record in records:
            _validate_record(record, mode=mode, manifest_hash=manifest_hash, rank=rank)
        result[rank] = records
    return result


def _rank_consistency(results: Mapping[int, Sequence[Mapping[str, Any]]]) -> None:
    baseline = results[0]
    comparable_fields = (
        "case_id",
        "dataset",
        "lifecycle_repeat",
        "request_sequence_index",
        "proposal_epoch_start",
        "proposal_epoch_end",
        "consumer_epoch_start",
        "consumer_epoch_end",
        "prompt_token_count",
        "prompt_token_sha256",
        "output_cap",
        "ignore_eos",
        "output_token_count",
        "output_token_ids",
        "output_token_sha256",
        "stop_reason",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "completed_rounds",
        "proposal_generated_count",
        "proposal_installed_count",
        "proposal_consumed_count",
        "terminal_discarded_proposal_count",
        "terminal_partial_commit",
    )
    for rank, records in results.items():
        if len(records) != len(baseline):
            raise ValueError(f"Rank {rank} has a missing/extra/partial result set.")
        for expected, actual in zip(baseline, records):
            for field in comparable_fields:
                if actual[field] != expected[field]:
                    raise ValueError(f"Rank {rank} disagrees with rank 0 for {actual['case_id']} field {field}.")


def _validate_manifest_order(manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    profile = records[0].get("profile") if records else None
    if profile not in {"smoke", "full"}:
        raise ValueError("Result profile must be smoke or full.")
    expected = manifest["profiles"][profile]
    grouped: dict[int, list[str]] = defaultdict(list)
    for record in records:
        grouped[int(record["lifecycle_repeat"])].append(str(record["case_id"]))
    if sorted(grouped) not in ([0], [0, 1, 2]):
        raise ValueError("Lifecycle repeats must be exactly one or three complete passes.")
    for repeat, case_ids in grouped.items():
        if case_ids != expected:
            raise ValueError(f"Lifecycle repeat {repeat} does not match the frozen profile order.")


def _validate_frozen_case_contract(manifest_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    frozen_cases = {case["case_id"]: case for case in read_jsonl(manifest_path.parent / "full_cases.jsonl")}
    for record in records:
        frozen = frozen_cases.get(record["case_id"])
        if frozen is None:
            raise ValueError(f"Result contains a case absent from the frozen manifest: {record['case_id']}.")
        field_pairs = {
            "dataset": "dataset",
            "prompt_token_count": "prompt_token_count",
            "prompt_token_sha256": "ordered_prompt_token_sha256",
            "output_cap": "output_cap",
            "ignore_eos": "ignore_eos",
        }
        for result_field, frozen_field in field_pairs.items():
            if record[result_field] != frozen[frozen_field]:
                raise ValueError(f"Result case {record['case_id']} differs from the frozen {result_field} contract.")


def _validate_repeat_determinism(records: Sequence[Mapping[str, Any]]) -> None:
    per_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        per_case[str(record["case_id"])].append(record)
    previous_epoch_end: int | None = None
    previous_consumer_epoch_end: int | None = None
    for record in records:
        epoch_start = record["proposal_epoch_start"]
        epoch_end = record["proposal_epoch_end"]
        if epoch_start is not None:
            if previous_epoch_end is not None and epoch_start <= previous_epoch_end:
                raise ValueError("Proposal epochs did not advance across sequential requests.")
            previous_epoch_end = epoch_end
        consumer_epoch_start = record["consumer_epoch_start"]
        consumer_epoch_end = record["consumer_epoch_end"]
        if consumer_epoch_start is not None:
            if previous_consumer_epoch_end is not None and consumer_epoch_start <= previous_consumer_epoch_end:
                raise ValueError("Consumer epochs did not advance across sequential requests.")
            previous_consumer_epoch_end = consumer_epoch_end
    for case_id, repeats in per_case.items():
        expected = repeats[0]
        for actual in repeats[1:]:
            for field in (
                "prompt_token_sha256",
                "output_token_ids",
                "output_token_sha256",
                "output_token_count",
                "stop_reason",
                "usage_prompt_tokens",
                "usage_completion_tokens",
            ):
                if actual[field] != expected[field]:
                    raise ValueError(f"Repeated lifecycle output changed for {case_id} field {field}.")


def validate_result_pair(
    manifest_path: Path,
    target_root: Path,
    dspark_root: Path,
    *,
    expected_ranks: int = EXPECTED_RANKS,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = verify_asset_bundle(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    target = load_results(
        target_root,
        mode="target_only",
        manifest_hash=manifest_hash,
        expected_ranks=expected_ranks,
    )
    dspark = load_results(
        dspark_root,
        mode="dspark",
        manifest_hash=manifest_hash,
        expected_ranks=expected_ranks,
    )
    _rank_consistency(target)
    _rank_consistency(dspark)
    target_records = target[0]
    dspark_records = dspark[0]
    _validate_manifest_order(manifest, target_records)
    _validate_manifest_order(manifest, dspark_records)
    _validate_frozen_case_contract(manifest_path, target_records)
    _validate_frozen_case_contract(manifest_path, dspark_records)
    _validate_repeat_determinism(target_records)
    _validate_repeat_determinism(dspark_records)
    if len(target_records) != len(dspark_records):
        raise ValueError("Target-only and DSpark result sets have different sizes.")
    compared_fields = (
        "case_id",
        "dataset",
        "lifecycle_repeat",
        "request_sequence_index",
        "prompt_token_count",
        "prompt_token_sha256",
        "output_cap",
        "ignore_eos",
        "output_token_count",
        "output_token_ids",
        "output_token_sha256",
        "stop_reason",
        "usage_prompt_tokens",
        "usage_completion_tokens",
    )
    for target_record, dspark_record in zip(target_records, dspark_records):
        for field in compared_fields:
            if target_record[field] != dspark_record[field]:
                raise ValueError(f"Exact-token gate mismatch for {target_record['case_id']} field {field}.")
    if not any(
        record["dataset"] != "synthetic"
        and not record["ignore_eos"]
        and record["output_token_count"] < record["output_cap"]
        for record in target_records
    ):
        raise ValueError("The real-data matrix observed no natural EOS/stop completion.")
    synthetic = [record for record in target_records if record["dataset"] == "synthetic"]
    if not synthetic or any(record["output_token_count"] != record["output_cap"] for record in synthetic):
        raise ValueError("Synthetic output-cap boundary coverage is incomplete.")
    configured_caps = {record["output_cap"] for record in target_records}
    if not {16, 128, 256, 512, 1024}.issubset(configured_caps):
        raise ValueError("The result matrix does not cover all frozen output-cap boundaries.")
    if not any(record.get("terminal_partial_commit") is True for record in dspark_records):
        raise ValueError("DSpark did not observe a terminal partial raw-verified commit.")
    if not any(record["terminal_discarded_proposal_count"] > 0 for record in dspark_records):
        raise ValueError("DSpark did not observe terminal proposal discard cleanup.")
    return {
        "case_executions": len(target_records),
        "unique_cases": len({record["case_id"] for record in target_records}),
        "ranks": expected_ranks,
        "manifest_sha256": manifest_hash,
        "historical_error_markers": list(HISTORICAL_ERROR_MARKERS),
        "exact_token_match": True,
        "performance_validated": False,
    }


def main() -> int:
    args = _parse_args()
    summary = validate_result_pair(
        args.manifest,
        args.target_results,
        args.dspark_results,
        expected_ranks=args.expected_ranks,
    )
    print("M2_5A_EXACT_TOKEN_GATE_PASS=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
