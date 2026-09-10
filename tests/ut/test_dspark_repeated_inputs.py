# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic 400/64 fixture: real import/hash/reader/driver behavior, no NPU claim."""

import json
from collections import Counter

import pytest

from tools.dspark import performance_report as report
from tools.dspark import prepare_performance_data as data
from tools.dspark import run_large_batch as large
from tools.dspark import run_performance_suite as suite


class DecoderOnly:
    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return str(tokens)


def import_inputs(tmp_path, monkeypatch, *, allow=True, count=400):
    raw = [{"task_id": f"case-{i % 64}", "prompt_token_ids": [100 + i % 64] + [42] * 115} for i in range(401)]
    source = tmp_path / "input-400.jsonl"
    # Noncanonical whitespace tests preservation of the original bytes/hash.
    source.write_text("".join(json.dumps(row, separators=(", ", ": ")) + "\n" for row in raw))
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    monkeypatch.setattr(data, "_load_tokenizer", lambda _: DecoderOnly())
    args = [
        "--input-jsonl",
        str(source),
        "--expected-source-sha256",
        data._sha256_file(source),
        "--source-name",
        "historical-fixture",
        "--source-revision",
        data._sha256_file(source),
        "--kind",
        "general",
        "--frozen-token-field",
        "prompt_token_ids",
        "--num-samples",
        str(count),
        "--max-input-tokens",
        "116",
        "--tokenizer",
        str(model),
        "--tokenizer-revision",
        suite.MODEL_REVISION,
        "--output-dir",
        str(tmp_path / "imported"),
    ]
    if allow:
        args += ["--allow-repeated-prompts"]
    data.main(args)
    return tmp_path / "imported/manifest.json", raw, source


def test_opt_in_preserves_400_instances_64_prompts(tmp_path, monkeypatch):
    path, raw, source = import_inputs(tmp_path, monkeypatch)
    manifest, rows, _ = data.read_manifest(path, 400)
    assert manifest["schema_version"] == 2 and "num_unique_samples" not in manifest
    assert manifest["request_instance_count"] == 400 and manifest["unique_prompt_count"] == 64
    assert data._sha256_file(source) == manifest["original_source_file_sha256"]
    assert (path.parent / "source.jsonl").read_bytes() == source.read_bytes()
    assert [r["prompt_token_ids"] for r in rows] == [r["prompt_token_ids"] for r in raw[:400]]
    assert Counter(tuple(r["prompt_token_ids"]) for r in rows) == Counter(
        tuple(r["prompt_token_ids"]) for r in raw[:400]
    )
    assert len({r["request_instance_id"] for r in rows}) == 400
    assert len({r["original_case_id"] for r in rows}) == 64
    assert rows[64]["replay_of"] == rows[0]["request_instance_id"]
    assert rows[128]["prompt_occurrence_index"] == 2
    assert sum(map(len, manifest["repeated_prompt_mapping"].values())) == 400
    assert data.input_population(manifest, rows)["label"] == "400 request instances / 64 unique prompts"
    assert data.input_population(manifest, data.read_manifest(path, 32)[1])["unique_prompt_count"] == 32
    with pytest.raises(ValueError, match="Insufficient"):
        data.read_manifest(path, 401)


def test_default_still_rejects_duplicates_and_legacy_unique_import_works(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Duplicate frozen"):
        import_inputs(tmp_path, monkeypatch, allow=False)
    path, _, _ = import_inputs(tmp_path, monkeypatch, allow=False, count=64)
    manifest, rows, _ = data.read_manifest(path, 64)
    assert manifest["schema_version"] == 1 and manifest["num_unique_samples"] == len(rows) == 64


@pytest.mark.parametrize(
    "damage",
    [
        "record_hash",
        "token_hash",
        "instance",
        "reference",
        "order",
        "mapping",
        "count",
        "policy",
        "source",
        "source_path",
        "original_case",
    ],
)
def test_corruption_rejected_even_with_rehashed_records(tmp_path, monkeypatch, damage):
    path, _, _ = import_inputs(tmp_path, monkeypatch)
    manifest, rows, records = data.read_manifest(path)
    if damage == "record_hash":
        rows[0]["prompt_token_ids"][0] += 1
    elif damage == "token_hash":
        rows[0]["prompt_token_sha256"] = "a" * 64
    elif damage == "instance":
        rows[64]["request_instance_id"] = rows[0]["request_instance_id"]
    elif damage == "reference":
        rows[64]["replay_of"] = rows[1]["request_instance_id"]
    elif damage == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif damage == "original_case":
        rows[0]["original_case_id"] = "invented"
    elif damage == "mapping":
        manifest["repeated_prompt_mapping"] = {}
    elif damage == "count":
        manifest["unique_prompt_count"] = 400
    elif damage == "policy":
        manifest["allow_repeated_prompts"] = False
    elif damage == "source":
        (path.parent / "source.jsonl").write_text("{}\n")
    else:
        manifest["source_snapshot_file"] = "../input-400.jsonl"
    if damage != "record_hash":
        for row in rows:
            row["record_sha256"] = data.digest({k: v for k, v in row.items() if k != "record_sha256"})
    records.write_bytes(b"".join(data._canonical_json_bytes(row) for row in rows))
    manifest["records_sha256"] = data._sha256_file(records)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        data.read_manifest(path)


def test_profile_and_performance_entrypoints_accept_instance_counts(tmp_path, monkeypatch):
    path, raw, _ = import_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "source_gate", lambda _: None)
    monkeypatch.setattr(suite, "resources_idle", lambda _: None)
    calls = []

    def logged(command, log):
        calls.append(command)
        log.write_text("CPU profile-entry test; no engine launched\n")
        return 0

    monkeypatch.setattr(suite, "logged", logged)
    monkeypatch.setattr(large, "scan", lambda _: None)
    root = tmp_path / "profile"
    assert (
        large.main(
            [
                "--stage",
                "profile",
                "--plugin-sha",
                "a" * 40,
                "--manifest",
                str(path),
                "--output-dir",
                str(root),
                "--batches",
                "64",
                "128",
                "256",
            ]
        )
        == 0
    )
    assert len(calls) == 3
    assert all(c[c.index("--num-prompts") + 1] == "400" for c in calls)
    identity = json.loads((root / "input-identity.json").read_text())
    assert identity["request_instance_count"] == 400 and identity["unique_prompt_count"] == 64
    data.read_manifest(root / "input/manifest.json", 400)
    options = tmp_path / "verification.json"
    cost = tmp_path / "cost.json"
    cost.write_text("{}")  # Planning only, no invented runtime cost accepted.
    options.write_text(json.dumps({"mode": "confidence", "cost_profile": str(cost)}))
    planned = tmp_path / "performance"
    assert (
        suite.main(
            [
                "--plugin-sha",
                "a" * 40,
                "--manifest",
                str(path),
                "--output-dir",
                str(planned),
                "--num-prompts",
                "400",
                "--max-num-seqs",
                "64",
                "--repeats",
                "3",
                "--modes",
                "dspark_graph",
                "dspark_confidence_graph",
                "--confidence-verification",
                str(options),
            ]
        )
        == 0
    )
    _, rows, _ = data.read_manifest(planned / "input/manifest.json", 400)
    assert [r["prompt_token_ids"] for r in rows] == [r["prompt_token_ids"] for r in raw[:400]]
    plan = json.loads((planned / "plan.json").read_text())
    assert plan["repeats"] == 3 and len(plan["runs"]) == 6
    assert plan["enable_prefix_caching"] is False
    summary = report.summarize_suite(planned)
    assert summary["status"] == "incomplete"  # Planning is never performance PASS.
    report.write_reports(summary, planned / "summary.json", planned / "summary.csv", planned / "summary.md")
    assert "400 request instances / 64 unique prompts" in (planned / "summary.md").read_text()
    assert "request_instance_count" in (planned / "summary.csv").read_text()
    assert report.run_statistics([100, 110, 120])["n"] == 3


def test_repeated_inputs_have_distinct_async_requests(tmp_path, monkeypatch):
    import asyncio

    from tests.ut.test_dspark_performance_delivery import DeltaEngine
    from tools.dspark.performance_stream import stream_batch

    path, _, _ = import_inputs(tmp_path, monkeypatch)
    _, rows, _ = data.read_manifest(path)
    prompts = [{"prompt_token_ids": row["prompt_token_ids"]} for row in rows]
    engine = DeltaEngine()
    warmup = asyncio.run(stream_batch(engine, prompts[:4], object(), 128, "batch1"))
    measured = asyncio.run(stream_batch(engine, prompts, object(), 128, "batch2"))
    assert len(set(engine.ids)) == 404
    assert len(measured["requests"]) == 400 and measured["error"] is None
    assert engine.peak == 128
    assert [r["request_index"] for r in measured["requests"]] == list(range(400))
    assert not {r["request_id"] for r in warmup["requests"]}.intersection(r["request_id"] for r in measured["requests"])


def test_documented_server_sequence_check_executes(tmp_path, monkeypatch, capsys):
    import sys
    from pathlib import Path

    path, _, source = import_inputs(tmp_path, monkeypatch)
    doc = (Path(__file__).parents[2] / "tools/dspark/REPEATED_INPUTS.md").read_text()
    script = doc.split("<<'PY' | tee \"$DATA_ROOT/check.json\"\n", 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.setattr(sys, "argv", ["check", str(path), str(source)])
    capsys.readouterr()
    exec(compile(script, "documented-server-input-check", "exec"), {})
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "valid"
    assert receipt["request_instance_count"] == 400 and receipt["unique_prompt_count"] == 64
