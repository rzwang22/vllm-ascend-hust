# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early validity gate reads actual device receipts, never manufactured epochs."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from tools.dspark import profile_failure
from tools.dspark.performance_stream import stream_batch
from tools.dspark.profile_attention_validity import AttentionValidity
from tools.dspark.profile_failure import ProfileFailureGuard


def publish(directory, rank, **changes):
    data = {
        "point": "first",
        "rank": rank,
        "performance_eligible": False,
        "status": "passed",
        "required_boundaries": [f"outer{i}" for i in range(9)] + [f"layer.1.attention.{i}" for i in range(12)],
        "rounds": [
            {
                "execution": e,
                "proposal_epoch": e - 2,
                "valid": True,
                "target_receipts": [e] * 21,
                "raw_receipts": [e] * 3,
                "consume_receipts": [e] * 3,
            }
            for e in range(5, 8)
        ],
    }
    data.update(changes)
    (directory / f"rank-{rank}-attention-validity.json").write_text(json.dumps(data))
    return data


def test_all_ranks_pass_then_same_engine_continues_without_reads(tmp_path):
    gate = AttentionValidity(tmp_path, 8)
    for rank in range(8):
        publish(tmp_path, rank)
        gate.check("first")
        assert gate.passed == (rank == 7)
    for path in tmp_path.iterdir():
        path.unlink()
    gate.check("tenth", tokens=512, finished=True)


@pytest.mark.parametrize("change", ["stale", "missing", "nonconsecutive", "identity", "failed"])
def test_invalid_receipts_fail_before_point_completion(tmp_path, change):
    data = publish(tmp_path, 0)
    if change == "stale":
        data["rounds"][0]["target_receipts"][9:] = [-1] * 12
    elif change == "missing":
        data["rounds"][0]["consume_receipts"] = []
    elif change == "nonconsecutive":
        data["rounds"][1]["execution"] = 99
    elif change == "identity":
        data["point"] = "old"
    else:
        data["status"] = "failed"
    publish(tmp_path, **data)
    with pytest.raises(RuntimeError):
        AttentionValidity(tmp_path, 1).check("first", tokens=6)


def test_missing_receipts_bounded_before_512(tmp_path):
    gate = AttentionValidity(tmp_path, 8)
    gate.check("first", tokens=63)
    with pytest.raises(RuntimeError, match="unavailable"):
        gate.check("first", tokens=64)
    with pytest.raises(RuntimeError, match="unavailable"):
        AttentionValidity(tmp_path, 8).check("first", finished=True)


@pytest.mark.parametrize("receipts", [True, False])
def test_real_stream_stops_early_or_continues_same_engine(tmp_path, receipts):
    gate = AttentionValidity(tmp_path, 1)
    if receipts:
        publish(tmp_path, 0)

    class Engine:
        calls = 0

        async def generate(self, prompt, sampling, request_id):
            self.calls += 1
            for step in range(512):
                yield NS(
                    outputs=[NS(index=0, token_ids=[1], text="", finish_reason="length", stop_reason=None)],
                    finished=step == 511,
                )

    engine = Engine()
    result = asyncio.run(
        stream_batch(
            engine,
            [{"prompt_token_ids": [1]}],
            NS(),
            None,
            "first",
            bounded_cancel=True,
            validate_progress=lambda tokens: gate.check("first", tokens=tokens),
        )
    )
    assert engine.calls == 1
    assert len(result["requests"][0]["output_token_ids"]) == (512 if receipts else 64)
    assert (result["error"] is None) == receipts


def test_generation_error_is_not_replaced_by_missing_gate(tmp_path):
    async def run():
        guard = ProfileFailureGuard(NS(errored=False), tmp_path)
        guard.attention_validity = AttentionValidity(tmp_path, 8)
        guard.point = "first"

        async def failed():
            return {"error": "original Markov NaN"}

        return await guard.run("generate", failed())

    assert asyncio.run(run())["error"] == "original Markov NaN"


def test_guard_detects_invalid_packet_without_rpc_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_failure, "POLL_SECONDS", 0.001)
    publish(tmp_path, 0, status="failed", error="INVALID_RECEIPT")
    cancelled = []

    async def run():
        guard = ProfileFailureGuard(NS(errored=False), tmp_path)
        guard.attention_validity = AttentionValidity(tmp_path, 1)
        guard.point = "first"

        async def waiting():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

        with pytest.raises(profile_failure.ProfileEngineFailed, match="INVALID_RECEIPT"):
            await guard.run("generate", waiting())
        assert "INVALID_RECEIPT" in guard.first["error"]

    asyncio.run(run())
    assert cancelled == [True]


def test_required_kv_receipts_cannot_be_omitted_or_forged_finite(tmp_path):
    data = publish(tmp_path, 0, kv_required=True)
    with pytest.raises(RuntimeError, match="stale/missing"):
        AttentionValidity(tmp_path, 1).check("first")
    for row in data["rounds"]:
        row["kv_receipts"] = [row["execution"]] * 4
    publish(tmp_path, **data)
    gate = AttentionValidity(tmp_path, 1)
    gate.check("first")
    assert gate.passed


@pytest.mark.parametrize("kind", ["missing", "stale", "duplicate", "current"])
def test_operator_receipt_gate_requires_one_current_call(tmp_path, kind):
    data = publish(tmp_path, 0, operator_required=True)
    for row in data["rounds"]:
        e = row["execution"]
        row["operator_receipts"] = {
            "missing": None,
            "stale": [-1, -1],
            "duplicate": [e] * 4,
            "current": [e, e, -1, -1],
        }[kind]
    publish(tmp_path, **data)
    if kind == "current":
        assert AttentionValidity(tmp_path, 1).check("first") is None
    else:
        with pytest.raises(RuntimeError):
            AttentionValidity(tmp_path, 1).check("first")
