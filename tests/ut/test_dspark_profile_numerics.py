# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU/mock boundary ordering and evidence; these are not NPU reproductions."""

import ast
import json
import subprocess
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_observation import OBS, ROOT, Runner, torch


def installed(tmp_path, head=None, real_markov=False, mode="numeric-boundaries"):
    runner = Runner()
    spec = runner.speculator
    spec.model.compute_draft_logits = head or (lambda hidden_states: hidden_states.clone())
    spec._build_draft_forward_metadata = lambda execution: {
        "mtp.0": NS(decode=NS(seq_lens_list=[101, 205, 509])),
        "mtp.1": NS(decode=None),
    }

    def checked_markov(proposal_inputs, hidden_states):
        spec._markov_attempt_step_epoch = proposal_inputs.step_epoch
        logits = spec.model.compute_draft_logits(hidden_states=hidden_states)
        # Stand in for the existing assertion; observation must have saved first.
        if torch.isnan(logits).any():
            evidence = json.loads((tmp_path / "rank-0-first-nan.json").read_text())
            assert evidence["numeric"]["rounds"][-1]["proposal_epoch"] == proposal_inputs.step_epoch
            raise RuntimeError("Ascend DSpark Markov base logits contain NaN.")
        return logits

    spec._execute_sequential_markov_sampling = checked_markov
    if real_markov:
        source = ast.parse((ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py").read_text())
        method = next(
            n
            for n in ast.walk(source)
            if isinstance(n, ast.FunctionDef) and n.name == "_execute_sequential_markov_sampling"
        )
        method.decorator_list = []
        # Execute the production method through its actual NaN guard. Helpers
        # model installed-state validation; no importing vLLM/Ascend on this host.
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )

        def guard(condition, message):
            assert "base logits contain NaN" in message
            assert (tmp_path / "rank-0-first-nonfinite.json").exists()
            if not bool(condition):
                raise RuntimeError(message)
            raise AssertionError("Test expected the original NaN guard to fail")

        namespace = {"torch": torch, "_assert_markov_tensor_contract": guard}
        exec(compile(ast.fix_missing_locations(module), "production_markov", "exec"), namespace)
        spec._execute_sequential_markov_sampling = namespace[method.name].__get__(spec)
        spec._validate_markov_inputs = lambda p, h: (p.num_reqs, p.num_speculative_tokens, p.num_query_tokens)
        spec._require_greedy_markov_sampling = lambda p: None
        spec._validate_step_tensor = lambda *a, **k: None
        spec._markov_attempt_step_epoch = None
        spec._markov_module_contract = {"lm_head_id": 1, "markov_head_id": 2, "confidence_head_id": 3}
        spec._inspect_markov_modules = lambda model: spec._markov_module_contract
        spec.draft_model_config = NS(hf_config=NS(vocab_size=8))
    return runner, OBS.ProfileObservation(runner, {"mode": mode, "directory": str(tmp_path)})


def run_round(runner, epoch, ids=("r3", "r2", "r1"), k=3, hidden=None):
    spec = runner.speculator
    runner.execute_model(NS(num_scheduled_tokens=dict(zip(ids, [1, 4, 6, 1])), finished_req_ids=[]))
    proposal = NS(
        step_epoch=epoch,
        rank=0,
        request_ids=tuple(ids),
        num_reqs=len(ids),
        num_speculative_tokens=k,
        num_query_tokens=len(ids) * k,
        num_target_tokens=int(runner.input_batch.num_tokens),
    )
    spec._proposal_step_epoch = epoch
    spec._prepared_step_epoch = epoch
    spec._build_draft_forward_metadata(NS(proposal_inputs=proposal))
    return spec._execute_sequential_markov_sampling(proposal, torch.ones(len(ids) * k, 8) if hidden is None else hidden)


@pytest.mark.parametrize("source", ["hidden", "logits"])
def test_nan_producer_classification_mapping_history_and_original_guard(tmp_path, source):
    inject = [False]

    def head(hidden_states):
        result = hidden_states.clone()
        if inject[0] and source == "logits":
            result[4, 2] = torch.nan
        return result

    runner, observer = installed(tmp_path, head)
    observer.begin_point("arbitrary-point")
    for epoch in (41, 42, 43, 44):
        run_round(runner, epoch)
    inject[0] = True
    hidden = torch.ones(9, 8)
    if source == "hidden":
        hidden[4, 2] = torch.nan
    with pytest.raises(RuntimeError, match="base logits contain NaN"):
        run_round(runner, 45, hidden=hidden)
    for suffix in ("first-nan", "first-nonfinite", "first-failure"):
        data = json.loads((tmp_path / f"rank-0-{suffix}.json").read_text())
        rounds = data["numeric"]["rounds"]
        assert [r["proposal_epoch"] for r in rounds] == [43, 44, 45]
        assert [r["execution"] for r in rounds] == [3, 4, 5]
        last = rounds[-1]
        assert last["rank"] == 0 and last["point"] == "arbitrary-point"
        assert last["classification"] == (
            "hidden_nonfinite" if source == "hidden" else "hidden_finite_logits_nonfinite"
        )
        assert last["candidate_rows"] == 9 and last["target_valid_tokens"] == 11
        row = last["rows"][4]
        assert (row["request_id"], row["request_row"], row["candidate_position"]) == ("r2", 1, 1)
        assert row["hidden_nan"] == (source == "hidden") and row["logits_nan"]
        assert not row["hidden_inf"] and not row["logits_inf"]
        assert last["epochs"]["_markov_attempt_step_epoch"] == 45
        assert last["draft_decode_seq_lens"]["layers"]["mtp.0"] == [101, 205, 509]
        assert data["performance_eligible"] is False and data["recording_error"] is None
        assert data["numeric"]["compact_host_transfers"] == 5
    original = (tmp_path / "rank-0-first-failure.json").read_bytes()
    observer.failed("downstream_ownership", ValueError("owners empty"))
    assert (tmp_path / "rank-0-first-failure.json").read_bytes() == original
    observer.close()


def test_production_markov_nan_guard_receives_unchanged_logits_after_evidence(tmp_path):
    runner, observer = installed(tmp_path, real_markov=True)
    observer.begin_point("production-guard")
    hidden = torch.ones(5, 8)
    hidden[2, 0] = torch.nan
    with pytest.raises(RuntimeError, match="Ascend DSpark Markov base logits contain NaN"):
        run_round(runner, 81, ids=("actual-id",), k=5, hidden=hidden)
    assert torch.isnan(hidden[2, 0])
    assert json.loads((tmp_path / "rank-0-first-failure.json").read_text())["failure"]["stage"] == "markov"
    observer.close()


def test_reused_hidden_reduced_before_in_place_head_and_single_transfer(tmp_path, monkeypatch):
    operations = []
    original_to = torch.Tensor.to

    def copy(tensor, *args, **kwargs):
        operations.append(("copy", tuple(tensor.shape), kwargs))
        return original_to(tensor, *args, **kwargs)

    def head(hidden_states):
        operations.append(("head",))
        hidden_states.fill_(0)  # fake norm/head reusing its input storage
        result = hidden_states.clone()
        result[0, 0] = float("-inf")
        result[3, 0] = float("inf")
        return result

    runner, observer = installed(tmp_path, head)
    observer.begin_point("reuse")
    hidden = torch.ones(4, 8)
    hidden[1, 0], hidden[2, 0] = float("inf"), float("-inf")
    monkeypatch.setattr(torch.Tensor, "to", copy)
    result = run_round(runner, 97, ids=("b", "a"), k=2, hidden=hidden)
    assert operations == [("head",), ("copy", (4, 4), {"device": "cpu", "non_blocking": False})]
    rows = observer.numeric_records[-1]["rows"]
    assert [r["hidden_inf"] for r in rows] == [False, True, True, False]
    assert [r["logits_inf"] for r in rows] == [True, False, False, True]
    assert [r["candidate_position"] for r in rows] == [0, 1, 0, 1]
    assert torch.isinf(result).any() and torch.isfinite(hidden).all()
    assert not (tmp_path / "rank-0-first-failure.json").exists()  # Inf observed, never rejected
    frozen = json.dumps(list(observer.numeric_records))
    hidden.fill_(float("nan"))
    result.fill_(float("nan"))
    assert json.dumps(list(observer.numeric_records)) == frozen
    observer.close()


def test_head_exception_saves_hidden_only_and_preserves_exception(tmp_path):
    error = RuntimeError("head failed")

    def head(hidden_states):
        raise error

    runner, observer = installed(tmp_path, head)
    observer.begin_point("head-error")
    with pytest.raises(RuntimeError) as caught:
        run_round(runner, 122, ids=("request",), k=2)
    assert caught.value is error
    data = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    last = data["numeric"]["rounds"][-1]
    assert last["classification"] == "logits_unavailable"
    assert all(r["logits_nan"] is None and r["logits_inf"] is None for r in last["rows"])
    assert data["numeric"]["compact_host_transfers"] == 1
    observer.close()


def test_first_nan_is_preserved_even_after_earlier_inf(tmp_path):
    runner, observer = installed(tmp_path)
    observer.begin_point("inf-then-nan")
    hidden = torch.ones(9, 8)
    hidden[0, 0] = float("-inf")
    run_round(runner, 6, hidden=hidden)
    first_inf = (tmp_path / "rank-0-first-nonfinite.json").read_bytes()
    run_round(runner, 7)
    run_round(runner, 8)
    hidden[0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        run_round(runner, 9, hidden=hidden)
    first_nan = json.loads((tmp_path / "rank-0-first-nan.json").read_text())
    assert [r["proposal_epoch"] for r in first_nan["numeric"]["rounds"]] == [7, 8, 9]
    assert (tmp_path / "rank-0-first-nonfinite.json").read_bytes() == first_inf
    assert first_nan["numeric"]["nan_rounds"] == 1
    observer.close()


def test_stale_numeric_context_is_reported_without_mislabeling_or_changing_output(tmp_path):
    runner, observer = installed(tmp_path)
    observer.numeric_context = NS(step_epoch=90)
    runner.speculator._markov_attempt_step_epoch = 91
    hidden = torch.ones(2, 8)
    assert torch.equal(runner.speculator.model.compute_draft_logits(hidden), hidden)
    assert "current Markov proposal epoch" in observer.recording_error
    assert not observer.numeric_records and observer.numeric_transfers == 0
    with pytest.raises(RuntimeError, match="evidence unavailable"):
        observer.finish_point()
    observer.close()


def test_compact_receipt_counts_all_rounds_and_close_restores_methods(tmp_path):
    runner, observer = installed(tmp_path)
    observer.begin_point("finite")
    for epoch in range(5):
        run_round(runner, epoch)
    receipt = observer.finish_point()
    assert receipt["numeric"]["classification_counts"] == {"both_finite": 5}
    assert receipt["numeric"]["compact_host_transfers_completed"] == 5
    assert "rounds" not in receipt["numeric"]
    observer.close()
    run_round(runner, 8)
    assert observer.numeric_transfers == 5 and not observer.numeric_records


def test_save_error_cannot_mask_original_model_error(tmp_path, monkeypatch):
    error = RuntimeError("original error")
    runner, observer = installed(tmp_path, lambda hidden_states: (_ for _ in ()).throw(error))
    monkeypatch.setattr(observer, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(RuntimeError) as caught:
        run_round(runner, 3)
    assert caught.value is error
    observer.close()


@pytest.mark.parametrize("mode", ["metadata-only", "context-kv-sync"])
def test_existing_modes_do_not_collect_numeric_values(tmp_path, monkeypatch, mode):
    runner, observer = installed(tmp_path, mode=mode)
    spec = runner.speculator
    # Avoid the mock Markov check; only the observer is under test here.
    forbidden = lambda *a, **k: pytest.fail("Disabled numeric diagnostic launched work")
    monkeypatch.setattr(torch, "isnan", forbidden)
    monkeypatch.setattr(torch, "isinf", forbidden)
    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    assert torch.equal(spec.model.compute_draft_logits(torch.ones(2, 8)), torch.ones(2, 8))
    assert not observer.numeric_records and observer.numeric_transfers == 0
    observer.close()


def test_cpu_seq_lens_and_id_transition_survive_stage_ring_eviction(tmp_path):
    runner, observer = installed(tmp_path)
    observer.begin_point("transitions")
    run_round(runner, 20, ids=("a", "b", "c", "d"))
    spec = runner.speculator
    spec._published_proposal_owners = {
        key: NS(producer_epoch=20, publication_row=i) for i, key in enumerate(("a", "b", "c", "d"))
    }
    spec.confidence_verification.last_selection = {"lengths": {"a": 0, "b": 1}, "producer_epochs": {"a": 19, "b": 19}}
    observer.record("proposal_publish.return")
    spec.confidence_verification.last_selection = {
        "lengths": {"d": 0, "c": 3, "b": 5},
        "producer_epochs": {"d": 20, "c": 20, "b": 20},
    }
    run_round(runner, 21, ids=("d", "c", "b"))
    observer.record("proposal_prepare.return")
    spec._published_proposal_owners = {}
    for _ in range(OBS.RING_RECORDS + 2):
        observer.record("unrelated")
    assert all(r["stage"] == "unrelated" for r in observer.records)
    data = observer.snapshot()
    transition = data["transitions"][0]
    assert transition["before"]["request_ids"] == ["a", "b", "c", "d"]
    assert transition["after"]["request_ids"] == ["d", "c", "b"]
    assert transition["before"]["owner_rows"]["c"] == {"producer_epoch": 20, "publication_row": 2}
    assert transition["after"]["selection"]["producer_epochs"]["c"] == 20
    assert transition["after"]["state_indices"] == [2, 0, 3]
    assert transition["proposal_prepare.return"]["epochs"]["_prepared_step_epoch"] == 21
    for index in range(OBS.TRANSITION_RECORDS + 3):
        run_round(runner, 30 + index, ids=(f"new-{index}",))
    assert len(observer.transitions) == OBS.TRANSITION_RECORDS
    assert observer.snapshot()["transitions_dropped"] == 4
    observer.begin_point("next")
    assert not observer.transitions and not observer.numeric_records
    assert observer.draft_seq_lens is None
    observer.close()


def test_server_entry_invokes_only_original_b64_prefix_with_numeric_mode(tmp_path):
    # Execute the real shell entry but replace its child bash with an argv
    # recorder. No server path, model load or NPU access is attempted on CPU.
    child = tmp_path / "bash"
    child.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    child.chmod(0o755)
    result = subprocess.run(
        [
            "/bin/bash",
            str(ROOT / "tools/dspark/run_dspark_profile_control.sh"),
            "test-sha",
            "manifest",
            "numeric-boundaries",
        ],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "/workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh",
        "test-sha",
        "manifest",
        "--stage",
        "profile",
        "--batches",
        "64",
        "--num-prompts",
        "400",
        "--profile-experiment",
        "numeric-boundaries",
        "--profile-stop-after-point",
        "ctx128-n4-t12-skewed",
        "--capture-sizes",
        "6",
        "12",
        "24",
        "48",
        "96",
        "192",
        "384",
        "--profile-contexts",
        "128",
        "2048",
        "--profile-output-tokens",
        "512",
        "--profile-warmup",
        "2",
        "--profile-samples",
        "5",
        "--max-model-len",
        "8192",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        "0.9",
    ]
