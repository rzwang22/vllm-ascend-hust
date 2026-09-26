# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B256 fixed K5/K8 experiment contracts; no cost lookup or confidence runtime."""

from tools.dspark import formal_cost as formal
from tools.dspark.audit_formal_cost import require

GROUP_SECONDS = 10000


def captures(k):
    require(k in (5, 8), "Only fixed K5 and K8 are in this experiment")
    return [(k + 1) * 2**i for i in range(9)]


def plan(base):
    cases = []
    for k in (5, 8):
        row = dict(next(c for c in base["cases"] if c["batch"] == 256 and c["mode"] == "fixed"))
        row.update(
            name=f"b256-fixed-k{k}",
            draft_k=k,
            capture_sizes=captures(k),
            full_target_query_tokens=256 * (k + 1),
            draft_query_tokens=256 * k,
            training_length="checkpoint dspark_block_size=5; K8 extrapolation, training support UNKNOWN",
        )
        cases.append(row)
    return {
        **base,
        "name": "dspark-b256-fixed-k5-k8-v1",
        "cases": cases,
        "model_initializations": 2,
        "total_limit_seconds": GROUP_SECONDS,
        "stage_sum_upper_seconds": 600 + 1800 + 2 * (3600 + 48 + 15),
        "order": "B256 fixed K5 then B256 fixed K8; no retries",
        "cost_tables": (
            "NO_RUNTIME_LOOKUP; immutable B256 publication used solely as frozen weight/environment provenance"
        ),
        "confidence_head": "weights loaded, no fixed-mode head forward or adaptive policy/D2H/lookup",
    }


def capacity_check(rows, k):
    require(sorted(r["rank"] for r in rows) == list(range(8)), "Missing capacity ranks")
    for r in rows:
        require(r["max_requests"] == r["draft_max_requests"] == 256, "Wrong real request capacity")
        require(min(r["max_tokens"], r["draft_max_tokens"]) >= 256 * (k + 1), "Insufficient real token buffers")
        source = r["draft_capacity_source"]
        require(
            source["kind"] == "allocated_shared_block_tables" and source["shared_with_target"], "Not actual buffers"
        )
        require(
            source["groups"] and all(g["stored_shape"][0] == g["input_shape"][0] == 256 for g in source["groups"]),
            "Bad block table shape",
        )
        require(source["slot_mapping_shape"][1] == r["draft_max_tokens"], "Slot buffer mismatch")
        require(r["kv_num_blocks"] > 0 and r["kv_bytes"] > 0 and r["groups"], "KV unavailable")
        require(r["capture_sizes"] == captures(k), "Wrong actual captures")


def validate_full(execution, k):
    proposals = execution["FULL"].get("published_proposals", {})
    require(
        proposals.get("calls", 0) > 0
        and proposals.get("requests", 0) > 0
        and proposals.get("candidates") == proposals["requests"] * k
        and proposals.get("first_epoch") is not None
        and proposals.get("last_epoch") is not None,
        "Missing real K-specific candidate publication evidence",
    )
    require(
        any(
            r["requests"] == 256 and r["query_tokens"] == r["capacity"] == 256 * (k + 1) and r["count"] > 0
            for r in execution["FULL"]["layouts"]
        ),
        "No actual full-population K-specific FULL execution; configured capacity is insufficient evidence",
    )


def runtime_expected(frozen, k):
    # Only the intended K/capture differences; all other frozen runtime fields exact.
    return {**frozen, "K": k, "capture_sizes": captures(k)}


def progress(scheduler, delta, execution):
    totals = delta["totals"]
    verified = totals["vllm:spec_decode_num_drafts"]
    accepted = totals["vllm:spec_decode_num_accepted_tokens"]
    return {
        "request_verifications": verified,
        "verified_candidates": totals["vllm:spec_decode_num_draft_tokens"],
        "accepted_candidates": accepted,
        "sampler_progress_per_request_verification": (verified + accepted) / verified if verified else None,
        "progress_semantics": (
            "accepted candidates + one bonus/replacement per request verification, "
            "before EOS truncation; not delivered output"
        ),
        "verification_batch_count": scheduler["verification_batch_count"],
        "sampler_progress_per_verification_batch": (verified + accepted) / scheduler["verification_batch_count"]
        if scheduler["verification_batch_count"]
        else None,
        "verification_steps": scheduler["verification_steps"],
        "step_semantics": (
            "Core SchedulerStats per output batch with num_drafts>0; delivered tokens may be EOS-truncated"
        ),
        "published_proposals": execution["FULL"]["published_proposals"],
    }


def summarize(root, reports, distribution):
    valid = len(reports) == 2 and all(r["valid"] for r in reports)
    result = {"all_two_valid": valid, "cases": reports, "comparison": None}
    if not valid:
        return result
    a, b = reports
    per_mode = {
        r["case"]["name"]: distribution([v["output_tokens_per_second"] for v in r["generation"]["rounds"]])
        for r in reports
    }
    differences = []
    for i in range(1, 4):
        x, y = [formal.read(root / "runs" / r["case"]["name"] / f"round-{i}-stream.json")["requests"] for r in reports]
        require([r["request_id"] for r in x] == [r["request_id"] for r in y], "Comparison identity mismatch")
        differences.append(
            {
                "round": i,
                "different_output_sequences": sum(p["output_token_ids"] != q["output_token_ids"] for p, q in zip(x, y)),
                "k5_tokens": sum(len(r["output_token_ids"]) for r in x),
                "k8_tokens": sum(len(r["output_token_ids"]) for r in y),
            }
        )
    result["comparison"] = {
        "throughput": per_mode,
        "speedup": per_mode[b["case"]["name"]]["median"] / per_mode[a["case"]["name"]]["median"],
        "speedup_formula": "median(K8 actual tokens/s) / median(K5 actual tokens/s); all three rounds",
        "output_differences": differences,
        "quality_equivalence": "NOT_EVALUATED",
        "warning": "OUTPUTS_DIFFER" if any(v["different_output_sequences"] for v in differences) else None,
        "order_bias": "K5 before K8, separate engines; order not randomized",
        "operator_compute_time": "UNAVAILABLE: no extra synchronization or layer timers",
        "structural_work": {"draft_rows_K5_K8": [1280, 2048], "target_full_rows_K5_K8": [1536, 2304]},
        "sampler_progress_per_request_verification": {
            r["case"]["name"]: distribution(
                [v["progress"]["sampler_progress_per_request_verification"] for v in r["generation"]["rounds"]]
            )
            for r in reports
        },
        "sampler_progress_per_verification_batch": {
            r["case"]["name"]: distribution(
                [v["progress"]["sampler_progress_per_verification_batch"] for v in r["generation"]["rounds"]]
            )
            for r in reports
        },
        "allocator_peak_by_mode": {
            r["case"]["name"]: [v["memory"]["after"] for v in r["generation"]["rounds"]] for r in reports
        },
        "allocator_peak_delta_k8_minus_k5_bytes": [
            {
                "rank": rank,
                **{
                    key: max(
                        row[key]
                        for v in b["generation"]["rounds"]
                        for row in v["memory"]["after"]
                        if row["rank"] == rank
                    )
                    - max(
                        row[key]
                        for v in a["generation"]["rounds"]
                        for row in v["memory"]["after"]
                        if row["rank"] == rank
                    )
                    for key in ("peak_allocated_bytes", "peak_reserved_bytes")
                },
            }
            for rank in range(8)
        ],
        "memory_limit": "Allocator counters exclude external CANN/HCCL; full device peak UNAVAILABLE",
    }
    return result
