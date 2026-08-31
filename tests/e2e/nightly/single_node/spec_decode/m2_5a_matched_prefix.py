# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Dependency-free evidence checks. No tolerance or production correctness waiver."""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from typing import Any

from tools.dspark.m2_5a_common import token_ids_sha256


def semantic_checks(stream: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on observed contract violations; report the observation scope.

    Recomputed diagnostic logits are an independent consistency check, NOT
    proof of actual sampler-logit identity. No whole-generation correctness
    PASS is possible from a bounded early-range observer alone.
    """
    errors: list[str] = []

    def require(ok: bool, label: str) -> None:
        if not ok:
            errors.append(label)

    if not stream.get("rank_trace_summaries"):
        return {"status": "UNAVAILABLE", "errors": ["Completed rank-local evidence required."]}
    require(stream["rank_identity"] is True, "artifact rank identity")
    require(stream["trace_rank_identity"] is True, "trace rank identity")
    for rank, artifact in stream["records"].items():
        try:
            summary = stream["rank_trace_summaries"][rank]
            completion = summary["completion"]
            require(completion["finished_event_observed_count"] == 1, f"rank {rank}: finished event publication")
            require(completion["finished_event_worker_delivery_count"] == 1, f"rank {rank}: finished event delivery")
            require(completion["released_scheduler_runner_kv_proposal_state"] is True, f"rank {rank}: released state")
            require(artifact["cleanup_complete"] is True, f"rank {rank}: cleanup")
            require(artifact["state_isolation_verified"] is True, f"rank {rank}: state isolation")
            epochs = completion["consumer_epochs"]
            require(epochs == sorted(set(epochs)), f"rank {rank}: repeated/nonmonotonic consumption epoch")
            require(len(epochs) == artifact["proposal_consumed_count"], f"rank {rank}: consumption count")
            if artifact["mode"] == "dspark":
                require(len(epochs) == artifact["verification_count"], f"rank {rank}: verification count")
                terminal = completion["terminal_lifecycle"]
                require(
                    terminal is not None
                    and terminal["request_ids"] == [artifact["request_id"]]
                    and terminal["discarded_terminal"] is True
                    and terminal["dropped"] is True
                    and terminal["consumed"] is False
                    and terminal["drop_reason"] == "terminal"
                    and terminal["proposal_epoch"] == terminal["owner_epoch"] == artifact["proposal_epoch_end"]
                    and artifact["terminal_discarded_proposal_count"] == 1,
                    f"rank {rank}: optimistic terminal proposal retirement",
                )
            end = 0
            seen = set()
            for row in (r for r in stream["traces"] if r["rank"] == int(rank)):
                label = f"rank {rank}, step {row['target_step_index']}"
                require(
                    row["sampling_state"]["temperature"] == [0.0] and row["sampling_state"]["seeds"] == [0],
                    label + ": observed greedy/seed contract",
                )
                start, end_next = row["commit_start_output_index"], row["commit_end_output_index_exclusive"]
                commit, raw = row["scheduler_committed_tokens"], row["raw_sampled_tokens"]
                require(start == end and end_next == start + len(commit), label + ": contiguous trace coverage")
                require(start < summary["header"]["early_output_token_limit"], label + ": trace bound")
                require(raw == commit, label + ": raw/commit (terminal truncation is not silently accepted)")
                require(
                    commit == row["artifact_appended_tokens"] == artifact["output_token_ids"][start:end_next],
                    label + ": commit/artifact",
                )
                require(row["logit_row_matches_committed_prefix"] == [True] * len(commit), label + ": row prefix")
                require(
                    row["prompt_token_sha256"] == artifact["prompt_token_sha256"]
                    and row["logit_row_output_prefix_sha256"]
                    == [token_ids_sha256(artifact["output_token_ids"][: start + i]) for i in range(len(commit))]
                    and row["logit_row_request_logical_lengths"]
                    == [artifact["prompt_token_count"] + start + i for i in range(len(commit))],
                    label + ": exact artifact prefix identity",
                )
                require(
                    row["logits_input_ids"][: len(commit)] == row["logit_row_request_input_token_ids"]
                    and row["logits_positions"][: len(commit)]
                    == [length - 1 for length in row["logit_row_request_logical_lengths"]],
                    label + ": actual input token/position versus logical request prefix",
                )
                if row["next_model_input_available"]:
                    require(row["next_runner_prefix_matches_request"] is True, label + ": next full runner prefix")
                    require(row["next_runner_prefix_sha256"] == row["next_prefix_token_sha256"], label + ": next hash")
                    require(
                        row["next_request_logical_length"] == artifact["prompt_token_count"] + end_next,
                        label + ": next logical length",
                    )
                else:
                    require(
                        end_next == artifact["output_token_count"]
                        and row["next_model_input_terminal_reason"] == "request_finished",
                        label + ": terminal next state",
                    )
                if row["step_kind"] == "verification":
                    size = row["scheduled_proposal_length"]
                    candidates = row["published_candidate_tokens"][:size]
                    require(0 < size <= len(row["published_candidate_tokens"]), label + ": scheduled length")
                    require(candidates == row["consumed_candidate_tokens"], label + ": real scheduled device prefix")
                    scheduled = row["scheduled_candidate_tokens"]
                    require(
                        len(scheduled) == size and (scheduled == [-1] * size or scheduled == candidates),
                        label + ": scheduler placeholder/token contract",
                    )
                    before, consumed = row["lifecycle_before_sample"], row["consumed_lifecycle"]
                    epoch = row["proposal_epoch"]
                    require(epoch not in seen, label + ": consumed twice")
                    seen.add(epoch)
                    for state in (before, consumed):
                        require(state["proposal_epoch"] == state["owner_epoch"] == epoch, label + ": lifecycle epoch")
                        require(state["request_ids"] == [row["request_id"]], label + ": lifecycle owner")
                        require(
                            state["scheduled_lengths"] == [size] and state["installed"] is True,
                            label + ": lifecycle installation",
                        )
                    require(
                        before["consumed"] is False
                        and consumed["consumed"] is True
                        and consumed["token_prefix_match"] is True,
                        label + ": consumption transition",
                    )
                    require(
                        row["consumer_epoch"] == consumed["consumer_epoch"] == epoch + 1, label + ": consumer epoch"
                    )
                    require(
                        row["published_owner_request_ids"] == row["verification_request_ids"] == [row["request_id"]],
                        label + ": request ownership",
                    )
                    require(
                        row["published_owner_state_indices"] == row["verification_state_indices"],
                        label + ": rank slot owner",
                    )
                    require(
                        row["counters_after_sample"]["consumed"] - row["counters_before_sample"]["consumed"] == 1,
                        label + ": consume once",
                    )
                    top1 = row["target_top1_token_ids"]
                    require(len(top1) == size + 1, label + ": verification logit rows")
                    accepted = 0
                    while accepted < size and candidates[accepted] == top1[accepted]:
                        accepted += 1
                    expected = candidates[:accepted] + [top1[accepted]]
                    bonus = accepted == size
                    require(raw == expected == row["expected_greedy_tokens"], label + ": observed greedy verification")
                    require(row["accepted_prefix_length"] == accepted, label + ": accepted prefix")
                    require(
                        row["bonus_used"] is bonus and row["replacement_used"] is not bonus,
                        label + ": replacement/bonus flag",
                    )
                    require(
                        row["bonus_token"] == (top1[accepted] if bonus else None)
                        and row["replacement_token"] == (None if bonus else top1[accepted]),
                        label + ": replacement/bonus token",
                    )
                    require(
                        row["num_sampled"] == [len(expected)] and row["num_rejected"] == [size + 1 - len(expected)],
                        label + ": raw sampled/rejected counts",
                    )
                else:
                    require(raw == row["target_top1_token_ids"][: len(raw)], label + ": observed producer greedy")
                end = end_next
            require(
                end >= min(summary["header"]["early_output_token_limit"], artifact["output_token_count"]),
                f"rank {rank}: missing early coverage",
            )
        except (KeyError, TypeError, IndexError) as exc:
            errors.append(f"rank {rank}: incomplete semantic evidence: {type(exc).__name__}: {exc}")
    return {
        "status": "FAIL" if errors else "OBSERVED_GATES_PASS",
        "errors": errors,
        "scope": "early traced commits + whole-request consumption counts/terminal cleanup",
        "actual_sampler_logit_identity_proven": False,
        "whole_generation_correctness_proven": False,
    }


def matched_prefix_report(streams: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    for launch, stream in streams.items():
        artifact = stream["records"]["0"]
        for row in stream["traces"]:
            if row["rank"] != 0:
                continue  # TP ranks are replicas, not independent error-envelope samples.
            if "scheduler_committed_tokens" not in row:
                excluded.append(
                    {"launch": launch, "step": row["target_step_index"], "reason": "legacy trace has no commit tokens"}
                )
                continue
            for i in range(len(row["scheduler_committed_tokens"])):
                try:
                    if row["logit_row_matches_committed_prefix"][i] is not True:
                        raise ValueError("rejected/noncommitted prefix")
                    index = row["output_length_before"] + i
                    start = row["output_length_before"]
                    commit = row["scheduler_committed_tokens"]
                    if commit != artifact["output_token_ids"][start : start + len(commit)]:
                        raise ValueError("trace commit differs from its own artifact")
                    if row["logit_row_prefix_sha256"][i] != row["committed_prefix_sha256"][i]:
                        raise ValueError("logit/committed prefix hash mismatch")
                    # R6D had full prefix hashes but not a separate output
                    # prefix hash. Derive only after the checks above, from
                    # THIS launch's artifact; never substitute another run.
                    output_prefix = token_ids_sha256(artifact["output_token_ids"][:index])
                    logical_length = artifact["prompt_token_count"] + index
                    if "logit_row_output_prefix_sha256" in row and (
                        row["logit_row_output_prefix_sha256"][i] != output_prefix
                        or row["logit_row_request_logical_lengths"][i] != logical_length
                        or row["prompt_token_sha256"] != artifact["prompt_token_sha256"]
                    ):
                        raise ValueError("explicit prefix identity differs from its own artifact")
                    identity = (
                        row["case_id"],
                        row["lifecycle_repeat"],
                        artifact["prompt_token_sha256"],
                        output_prefix,
                        row["logits_input_ids"][i],
                        row["logits_positions"][i],
                        logical_length,
                    )
                    if identity[-2] != identity[-1] - 1:
                        raise ValueError("position/logical length mismatch")
                    q_len = row["query_lengths"][0]
                    # Configuration/manifest are additional guards, never a
                    # replacement for the exact per-row prefix identity above.
                    config = {
                        "manifest": artifact["manifest_sha256"],
                        "sampling": row["sampling_state"],
                        "model_seed": row["model_seed"],
                        "path": row.get("execution_path"),
                    }
                    context = json.dumps(config, sort_keys=True)
                    sample = {
                        "launch": launch,
                        "mode": artifact["mode"],
                        "step": row["target_step_index"],
                        "row": i,
                        "output_index": row["output_length_before"] + i,
                        "q_len": q_len,
                        "step_kind": row["step_kind"],
                        "top1": row["target_top1_token_ids"][i],
                        "top2": row["target_top2_token_ids"][i],
                        "top1_logit": row["target_top1_logits"][i],
                        "top2_logit": row["target_top2_logits"][i],
                        "margin": row["target_top1_top2_margins"][i],
                        "logits_observation": row["logits_observation"],
                        "configuration": config,
                        "legacy_derived_identity": "logit_row_output_prefix_sha256" not in row,
                        "execution_signature_complete": config["path"] is not None,
                    }
                    groups[(*identity, context)].append(sample)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    # Legacy data may be explored, but never silently counted
                    # as an eligible match or an empirical tolerance sample.
                    excluded.append({"launch": launch, "step": row["target_step_index"], "row": i, "reason": str(exc)})
    matched, cross_path_count = [], 0
    for key, samples in groups.items():
        paths = defaultdict(list)
        for sample in samples:
            paths[(sample["mode"], sample["step_kind"], sample["q_len"], sample["logits_observation"])].append(sample)
        envelopes = []
        for path, observations in paths.items():
            by_token = defaultdict(list)
            for obs in observations:
                for slot in ("top1", "top2"):
                    by_token[obs[slot]].append({"launch": obs["launch"], "logit": obs[slot + "_logit"]})
            envelopes.append(
                {
                    "path": path,
                    "independent_launches": sorted({obs["launch"] for obs in observations}),
                    "execution_signature_complete": all(obs["execution_signature_complete"] for obs in observations),
                    "tokens": {
                        str(token): {
                            "raw_samples": values,
                            "observed_spread": max(v["logit"] for v in values) - min(v["logit"] for v in values)
                            if len({v["launch"] for v in values}) >= 2
                            and all(obs["execution_signature_complete"] for obs in observations)
                            else None,
                        }
                        for token, values in by_token.items()
                    },
                }
            )
        comparisons = []
        for left, right in itertools.combinations(samples, 2):
            if left["launch"] == right["launch"]:
                continue
            eligible = (
                left["mode"] == "target_only"
                and left["q_len"] == 1
                and right["mode"] == "dspark"
                and right["step_kind"] == "verification"
                and right["q_len"] > 1
            ) or (
                right["mode"] == "target_only"
                and right["q_len"] == 1
                and left["mode"] == "dspark"
                and left["step_kind"] == "verification"
                and left["q_len"] > 1
            )
            if not eligible:
                continue
            a = {left["top1"]: left["top1_logit"], left["top2"]: left["top2_logit"]}
            b = {right["top1"]: right["top1_logit"], right["top2"]: right["top2_logit"]}
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "same_prefix": True,
                    "top1_equal": left["top1"] == right["top1"],
                    "common_token_logit_deltas_right_minus_left": {
                        str(token): b[token] - a[token] for token in a.keys() & b.keys()
                    },
                    "full_vocabulary_logits_observed": False,
                }
            )
        if comparisons or any(len(group["independent_launches"]) >= 2 for group in envelopes):
            matched.append(
                {
                    "identity": dict(
                        zip(
                            (
                                "case_id",
                                "lifecycle_repeat",
                                "prompt_sha256",
                                "output_prefix_sha256",
                                "input_token",
                                "position",
                                "request_logical_length",
                            ),
                            key[:-1],
                        )
                    ),
                    "samples": samples,
                    "same_path_empirical_envelopes": envelopes,
                    "cross_path_comparisons": comparisons,
                }
            )
            cross_path_count += len(comparisons)
    return {
        "groups": matched,
        "eligible_cross_path_pair_count": cross_path_count,
        "exact_prefix_group_count": sum(bool(group["cross_path_comparisons"]) for group in matched),
        "excluded_rows": excluded,
        "tolerance": None,
        "envelope_caveat": "Empirical same-prefix/same-path spreads only; not a bound or correctness tolerance.",
        "matched_prefix_abi": "natural real scheduler/runner commits; no forced prefix or forged verification",
        "matching_unit": "rank-0 row per independent launch; TP ranks are identity checks, not independent samples",
        "d2_observer_effect_proven": False,
        "d3_numerical_path_causality_proven": False,
        "production_correctness_proven": False,
    }
