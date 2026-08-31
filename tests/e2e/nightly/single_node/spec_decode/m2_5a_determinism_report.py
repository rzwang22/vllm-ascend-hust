# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Offline, full-token forensic report, NOT a replacement for the exact gate.

Run with ``python -m tests.e2e.nightly.single_node.spec_decode.m2_5a_determinism_report``.
Each --stream NAME=DIR contains one selected case and rank-*.jsonl artifacts;
optional --trace NAME=DIR associates rank-local traces from that model launch.
Legacy NAME=LOG is exploratory only and cannot satisfy semantic gates.
Missing trace coverage is reported as unknown, never inferred from another run.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import re
from pathlib import Path
from typing import Any

from tests.e2e.nightly.single_node.spec_decode.m2_5a_matched_prefix import matched_prefix_report, semantic_checks
from tests.e2e.nightly.single_node.spec_decode.m2_5a_trace_io import load_rank_traces
from tools.dspark.m2_5a_common import read_jsonl, sha256_file, token_ids_sha256

TRACE_MARKERS = (
    "DSPARK_M2_5A_EARLY_RANGE_TRACE=",
    "DSPARK_M2_5A_OUTPUT_INDEX_TRACE=",
)
PAIR_FIELDS = (
    "case_id",
    "lifecycle_repeat",
    "request_sequence_index",
    "prompt_token_count",
    "prompt_token_sha256",
    "manifest_sha256",
    "output_cap",
    "ignore_eos",
)


def align_tokens(left: list[int], right: list[int]) -> dict[str, Any]:
    common = 0
    for a, b in zip(left, right):
        if a != b:
            break
        common += 1
    identical = left == right
    opcodes = difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
    return {
        "identical": identical,
        "first_different_index": None if identical else common,
        "first_tokens": None if identical else [left[common : common + 1], right[common : common + 1]],
        "longest_common_prefix_length": common,
        "longest_common_prefix_tokens": left[:common],
        "opcodes": opcodes,
        "edits": [
            {"tag": tag, "left_range": [i, j], "right_range": [k, end], "left": left[i:j], "right": right[k:end]}
            for tag, i, j, k, end in opcodes
            if tag != "equal"
        ],
    }


def load_stream(root: Path, expected_ranks: int | None = None) -> dict[str, Any]:
    records: dict[int, dict[str, Any]] = {}
    digests = {}
    for path in sorted(root.glob("rank-*.jsonl")):
        rank = int(path.stem.removeprefix("rank-"))
        rows = read_jsonl(path)
        if len(rows) != 1 or rows[0]["rank"] != rank:
            raise ValueError(f"Expected one exact case with matching rank in {path}.")
        row = rows[0]
        tokens = row["output_token_ids"]
        if (
            not isinstance(tokens, list)
            or any(type(token) is not int for token in tokens)
            or len(tokens) != row["output_token_count"]
            or token_ids_sha256(tokens) != row["output_token_sha256"]
        ):
            raise ValueError(f"Invalid token count/hash/content in {path}.")
        digest_path = path.with_suffix(".jsonl.sha256")
        digest = sha256_file(path)
        if digest_path.exists() and digest_path.read_text().strip() != digest:
            raise ValueError(f"Invalid artifact SHA256 in {digest_path}.")
        digests[str(rank)] = {"sha256": digest, "sidecar_present": digest_path.exists()}
        records[rank] = row
    if 0 not in records or (expected_ranks is not None and set(records) != set(range(expected_ranks))):
        raise ValueError(f"Incomplete rank artifacts at {root}: {sorted(records)}.")
    baseline = {key: value for key, value in records[0].items() if key != "rank"}
    return {
        "path": str(root.resolve()),
        "artifact_digests": digests,
        "records": {str(rank): row for rank, row in records.items()},
        "rank_identity": all({k: v for k, v in row.items() if k != "rank"} == baseline for row in records.values()),
        "rank_alignments": {
            str(rank): align_tokens(records[0]["output_token_ids"], row["output_token_ids"])
            for rank, row in records.items()
        },
        "token_9045_positions": [i for i, token in enumerate(records[0]["output_token_ids"]) if token == 9045],
    }


def load_traces(path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    traces = {}
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    for match in re.finditer("|".join(map(re.escape, TRACE_MARKERS)), text):
        trace, _ = decoder.raw_decode(text[match.end() :].lstrip())
        if trace["mode"] != record["mode"]:
            continue
        for field in ("case_id", "lifecycle_repeat", "request_sequence_index", "request_id"):
            if trace[field] != record[field]:
                raise ValueError(f"Trace/artifact ownership mismatch for {field} in {path}.")
        key = (trace["rank"], trace["target_step_index"])
        if key in traces:
            raise ValueError(f"Duplicate trace {key} in {path}.")
        traces[key] = trace
    if not traces:
        raise ValueError(f"No matching early/output-index traces in {path}.")
    return [traces[key] for key in sorted(traces)]


def trace_at_index(traces: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    for record in traces:
        start = record["commit_start_output_index"]
        end = record["commit_end_output_index_exclusive"]
        if record["rank"] != 0 or not start <= index < end:
            continue
        row = index - start
        prefixes = record.get("logit_row_prefix_sha256", [])
        matches = record.get("logit_row_matches_committed_prefix", [])
        return {
            "output_index": index,
            "logit_row": row,
            "logit_prefix_sha256": prefixes[row] if prefixes else record["prefix_token_sha256"] if row == 0 else None,
            "logit_prefix_matches_committed": matches[row] if matches else True if row == 0 else None,
            "top1": record["target_top1_token_ids"][row],
            "top2": record["target_top2_token_ids"][row],
            "margin": record["target_top1_top2_margins"][row],
            "position": record["logits_positions"][row],
            "input_token": record["logits_input_ids"][row],
            "record": record,
        }
    return None


def build_report(
    streams: dict[str, Path],
    trace_paths: dict[str, Path],
    expected_ranks: int | None = None,
) -> dict[str, Any]:
    if len(streams) < 2 or set(trace_paths).difference(streams):
        raise ValueError("Need at least two named streams; each trace must belong to a stream.")
    if len({path.resolve() for path in streams.values()}) != len(streams):
        raise ValueError("Aliased stream directories are not independent launches.")
    loaded = {name: load_stream(path, expected_ranks) for name, path in streams.items()}
    for name, stream in loaded.items():
        traces = []
        if name in trace_paths:
            if trace_paths[name].is_dir():
                traces, summaries = load_rank_traces(
                    trace_paths[name], stream["records"], expected_ranks or len(stream["records"])
                )
                stream["rank_trace_summaries"] = summaries
            else:
                traces = load_traces(trace_paths[name], stream["records"]["0"])
        stream["traces"] = traces
        by_rank = {
            rank: [{k: v for k, v in row.items() if k != "rank"} for row in traces if row["rank"] == int(rank)]
            for rank in stream["records"]
        }
        stream["trace_rank_identity"] = (
            all(rows == by_rank["0"] and bool(rows) for rows in by_rank.values()) if traces else None
        )
        stream["hard_semantic_gates"] = semantic_checks(stream)
    comparisons = []
    for left_name, right_name in itertools.combinations(loaded, 2):
        left, right = loaded[left_name], loaded[right_name]
        a, b = left["records"]["0"], right["records"]["0"]
        alignment = align_tokens(a["output_token_ids"], b["output_token_ids"])
        ltrace = trace_at_index(left["traces"], alignment["first_different_index"])
        rtrace = trace_at_index(right["traces"], alignment["first_different_index"])
        same_logit_prefix = None
        if ltrace and rtrace and ltrace["logit_prefix_sha256"] and rtrace["logit_prefix_sha256"]:
            same_logit_prefix = (
                ltrace["logit_prefix_sha256"] == rtrace["logit_prefix_sha256"]
                and ltrace["logit_prefix_matches_committed"] is True
                and rtrace["logit_prefix_matches_committed"] is True
            )
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "pairing_mismatches": {field: [a[field], b[field]] for field in PAIR_FIELDS if a[field] != b[field]},
                **alignment,
                "cross_launch_exact_match": alignment["identical"],
                "output_sha256": [a["output_token_sha256"], b["output_token_sha256"]],
                "first_difference_traces": [ltrace, rtrace],
                "same_logit_prefix": same_logit_prefix,
            }
        )
    return {
        "diagnostic_only": True,
        "exact_gate_changed": False,
        "alignment": "difflib.SequenceMatcher(autojunk=False), directional, not a minimal edit-distance proof",
        "logits_caveat": "Trace recomputes LM-head logits; observer effect and sampler-logit identity are not proven.",
        "streams": loaded,
        "comparisons": comparisons,
        "matched_prefix": matched_prefix_report(loaded),
        "hard_semantic_gates_pass": (
            all(stream["hard_semantic_gates"]["status"] == "OBSERVED_GATES_PASS" for stream in loaded.values())
            and all(not pair["pairing_mismatches"] for pair in comparisons)
        ),
        "whole_generation_correctness_proven": False,
    }


def _named_paths(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not name or not sep or not path or name in paths:
            raise ValueError(f"Expected a unique NAME=PATH, got {value!r}.")
        paths[name] = Path(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", action="append", required=True)
    parser.add_argument("--trace", action="append", default=[])
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--require-semantic-gates", action="store_true")
    parser.add_argument("--require-matched-prefix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(_named_paths(args.stream), _named_paths(args.trace), args.expected_ranks)
    # Exclusive creation protects all historical evidence and prior reports.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for comparison in report["comparisons"]:
        print(
            json.dumps(
                {
                    k: comparison[k]
                    for k in (
                        "left",
                        "right",
                        "identical",
                        "first_different_index",
                        "first_tokens",
                        "pairing_mismatches",
                        "same_logit_prefix",
                    )
                }
            )
        )
    print("M2_5A_DETERMINISM_REPORT_WRITTEN=" + str(args.output))
    if any(stream["hard_semantic_gates"]["status"] == "FAIL" for stream in report["streams"].values()):
        raise ValueError("Observed semantic violation; inspect the written evidence report.")
    if args.require_semantic_gates and not report["hard_semantic_gates_pass"]:
        raise ValueError("Missing/failed observed semantic gates; inspect the written evidence report.")
    if args.require_matched_prefix and report["matched_prefix"]["eligible_cross_path_pair_count"] == 0:
        raise ValueError("No exact matched-prefix q_len=1/verification pair; no numerical conclusion is possible.")


if __name__ == "__main__":
    main()
