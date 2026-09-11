# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host metadata and deterministic profile-window regression; no NPU execution."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.ut.test_dspark_startup_cost_profile import POLICY, snapshots, table_data
from tools.dspark import startup_cost_profile as profile

ROOT = Path(__file__).parents[2]


def tail_evidence():
    point = next(
        p for p in profile.grid(64, [6, 12, 24, 48, 96, 192, 384], [128], 512)[1] if p["id"] == "ctx128-n2-t6-balanced"
    )
    ranks = snapshots(point, count=171, ranks=8)
    for rank in ranks:
        for index, event in enumerate(rank["cost_profile"]["measurements"]):
            i = index % 171
            # Synthetic legal physical lengths distinguish the scheduler bound
            # from corrected KV. Actual server physical lengths are unavailable.
            context = 643 if i == 170 else 128 + i * 3
            event.update(
                context=context,
                scheduler_computed_upper_bounds=[context, context],
                effective_kv_before_query=[min(context, 637)] * 2,
                attention_seq_lens=[min(context, 637) + 3] * 2,
            )
    return point, ranks


def test_171_events_tail_outside_domain_preserves_raw_and_windows():
    point, ranks = tail_evidence()
    original = copy.deepcopy(ranks)
    selected = profile.point_samples(point, ranks, 2, 5, 8)
    assert len(selected) == 16
    for row in selected:
        offset = 0 if row["kind"] == "target" else 171
        assert row["selected_raw_indices"] == list(range(offset + 2, offset + 7))
        assert row["extra_sample_count"] == 163
        assert row["excluded_sample_count"] == 1
        assert row["excluded_counts_by_reason"]["outside_context_domain"] == 1
        assert row["excluded_samples"] == [
            {"raw_index": offset + 170, "reason": "outside_context_domain", "context": 643}
        ]
        assert [r["sample_selection"]["classification"] for r in row["warmup"]] == ["warmup"] * 2
        assert all(r["sample_selection"]["classification"] == "retained" for r in row["samples"])
        assert row["median_seconds"] == 0.005
    for before, after in zip(original, json.loads(json.dumps(ranks))):
        for a, b in zip(before["cost_profile"]["measurements"], after["cost_profile"]["measurements"]):
            b.pop("sample_selection")
            assert a == b  # Never clamp or overwrite 643, timing, or raw IDs.


@pytest.mark.parametrize(
    "invalid",
    [
        "nan",
        "zero",
        "negative_context",
        "negative_kv",
        "physical_overflow",
        "bad_shape",
        "bad_capacity",
        "wrong_owner",
        "failed",
    ],
)
def test_illegal_extra_or_outside_domain_event_still_fails(invalid):
    point, ranks = tail_evidence()
    row = ranks[0]["cost_profile"]["measurements"][170]
    if invalid == "nan":
        row["seconds"] = float("nan")
    elif invalid == "zero":
        row["seconds"] = 0
    elif invalid == "negative_context":
        row["context"] = -1
        row["scheduler_computed_upper_bounds"] = [-1, -1]
    elif invalid == "negative_kv":
        row["effective_kv_before_query"][0] = -1
    elif invalid == "physical_overflow":
        row["effective_kv_before_query"][0] = 8191
        row["attention_seq_lens"][0] = 8194
    elif invalid == "bad_shape":
        row["query_lengths"] = [3]  # formerly dropped by the matching predicate
    elif invalid == "bad_capacity":
        row["capacity"] = row["size"] = 99999
    elif invalid == "wrong_owner":
        row["point"] = "previous"
    else:
        ranks[0]["failed_execution_count"] = 1
    with pytest.raises(ValueError):
        profile.point_samples(point, ranks, 2, 5, 8)
    if invalid != "failed":
        assert row["sample_selection"]["classification"] == "invalid"


def test_domain_filter_precedes_warmup_and_never_depends_on_speed():
    point, ranks = tail_evidence()
    ranks = ranks[:1]
    events = ranks[0]["cost_profile"]["measurements"]
    for kind in ("target", "draft"):
        rows = [e for e in events if e["kind"] == kind]
        # Exclude the first event, not just a special-cased terminal suffix.
        rows[0].update(context=700, scheduler_computed_upper_bounds=[700, 700])
        rows[3]["seconds"] = 1e6
    selected = profile.point_samples(point, ranks, 2, 5, 1)
    assert selected[0]["selected_raw_indices"] == [3, 4, 5, 6, 7]
    assert selected[0]["samples"][0]["seconds"] == 1e6
    assert selected[0]["excluded_sample_count"] == 2
    ranks[0]["cost_profile"]["measurements"] = events[:7] + events[171:178]
    with pytest.raises(ValueError, match="Insufficient real FULL samples"):
        profile.point_samples(point, ranks, 2, 5, 1)


def test_same_capacity_different_queries_and_valid_layout_exclusion():
    points = profile.grid(2, [6, 12], [128], 512)[1]
    for requests, lengths in [(1, [5]), (2, [2, 2])]:
        p = next(p for p in points if p["capacity"] == 6 and p["requests"] == requests and p["lengths"] == lengths)
        ranks = snapshots(p)
        extra = copy.deepcopy(ranks[0]["cost_profile"]["measurements"][-1])
        extra.update(
            requests=1,
            query_lengths=[6],
            scheduler_computed_upper_bounds=[20],
            effective_kv_before_query=[16],
            attention_seq_lens=[22],
            size=1,
        )
        ranks[0]["cost_profile"]["measurements"].append(extra)
        selected = profile.point_samples(p, ranks, 2, 5, 2)
        if requests == 2:
            assert selected[1]["excluded_samples"][0]["reason"] == "outside_layout_domain"


def test_corrected_cpu_lengths_are_distinct_from_runtime_scheduler_context():
    source = ast.parse((ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text())
    method = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "_update_seq_lens_cpu")
    helper_source = ast.parse((ROOT / "vllm_ascend/diagnostics/dspark_cost_profile.py").read_text())
    helper = next(n for n in helper_source.body if isinstance(n, ast.FunctionDef) and n.name == "profile_context")
    ns = {"COST_CONTEXT_SEMANTICS": POLICY["COST_CONTEXT_SEMANTICS"]}
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, method, helper], type_ignores=[])
    exec(compile(module, "actual-Ascend-host-context-bodies", "exec"), ns)
    waits = []
    states = NS(
        req_id_to_index={"a": 0, "b": 1},
        num_computed_tokens_cpu=np.zeros(2),
        num_computed_tokens_np=np.array([900, 800]),
    )
    output = NS(
        num_scheduled_tokens={"a": 3, "b": 3},
        scheduled_cached_reqs=NS(req_ids=["a", "b"], num_computed_tokens=[643, 639]),
    )
    runner = NS(
        req_states=states,
        num_computed_tokens_event=NS(synchronize=lambda: waits.append("existing")),
        num_computed_tokens_cpu=np.array([637, 635]),
        input_buffers=NS(seq_lens_cpu=np.zeros(8, dtype=np.int32)),
    )
    ns["_update_seq_lens_cpu"](runner, output, ["b", "a"])
    assert waits == ["existing"]  # only the already-required runner wait
    assert POLICY["current_host_contexts"](states, output) == {"a": 643, "b": 639}
    batch = NS(
        num_reqs=2,
        num_scheduled_tokens=np.array([3, 3, 99]),
        num_computed_tokens_np=np.array([639, 643, 99999]),
        seq_lens_np=runner.input_buffers.seq_lens_cpu,
    )
    context = ns["profile_context"](batch, 8192)
    assert context["context"] == 643
    assert context["effective_kv_before_query"] == [635, 637]
    assert context["attention_seq_lens"] == [638, 640]
    assert len(waits) == 1  # helper performs no sync/D2H
    # Next point/input updates the same buffer; old receipt is a detached list.
    identity = runner.input_buffers.seq_lens_cpu.ctypes.data
    runner.num_computed_tokens_cpu[:] = [130, 132]
    ns["_update_seq_lens_cpu"](runner, output, ["a", "b"])
    batch.num_computed_tokens_np[:2] = [134, 135]
    fresh = ns["profile_context"](batch, 8192)
    assert fresh["context"] == 135 and fresh["effective_kv_before_query"] == [130, 132]
    assert context["attention_seq_lens"] == [638, 640]
    assert runner.input_buffers.seq_lens_cpu.ctypes.data == identity


@pytest.mark.parametrize("old", ["missing_identity", "missing_table", "wrong_semantics"])
def test_cost_table_rejects_ambiguous_or_different_context_semantics(old):
    table, identity = table_data()
    if old == "missing_identity":
        del identity["cost_context_semantics"]
    elif old == "missing_table":
        del table["context_semantics"]
    else:
        table["context_semantics"] = "effective_kv"
    with pytest.raises(ValueError, match="context semantics"):
        POLICY["CostTable"].load_startup(table, identity)


def test_cost_bucket_is_explicit_estimate_not_measured_ceiling():
    table, identity = table_data()
    assert table["context_semantics"] == identity["cost_context_semantics"] == POLICY["COST_CONTEXT_SEMANTICS"]
    assert all(
        r["range"] == [20, 20] for c in table["cells"] for rows in c["selected_context_ranges"].values() for r in rows
    )
    costs = POLICY["CostTable"].load_startup(table, identity)
    assert costs.cost(2, 6, 20) == costs.cost(2, 6, 80)
    with pytest.raises(ValueError, match="Context outside"):
        costs.cost(2, 6, max(table["context_ceilings"]) + 1)


def test_unknown_request_in_outside_tail_is_not_filtered():
    from tools.dspark.profile_request_ids import MAPPING_SOURCE, RequestIdentityError, validate_point_request_ids

    point, ranks = tail_evidence()
    for rank in ranks:
        for row in rank["cost_profile"]["measurements"]:
            row["request_ids"] = ["batch3-0-88cf6394", "batch3-1-b4796666"]
    ranks[0]["cost_profile"]["measurements"][170]["request_ids"][0] = "prior-point-internal"
    stream = {
        "requests": [{"request_id": f"batch3-{i}", "request_index": i} for i in range(2)],
        "request_id_mapping": {
            "point": point["id"],
            "source": MAPPING_SOURCE,
            "hook_restored": True,
            "expected_external_ids": ["batch3-0", "batch3-1"],
            "errors": [],
            "mappings": [
                {"point": point["id"], "external_id": f"batch3-{i}", "request_index": i, "internal_id": internal}
                for i, internal in enumerate(["batch3-0-88cf6394", "batch3-1-b4796666"])
            ],
        },
    }
    with pytest.raises(RequestIdentityError) as error:
        validate_point_request_ids(point["id"], stream, ranks, {"prior-point-internal": "previous"})
    assert error.value.evidence["event_failures"][0]["reason"] == "previous_point_event"
