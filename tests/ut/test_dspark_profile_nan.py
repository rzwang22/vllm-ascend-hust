# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile-only observer behavior, real CPU reductions and local failure writes."""

import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.ut.test_dspark_graph_replay import _EXTENSION, ReplayManager
from tests.ut.test_dspark_nan_diagnostics import _NAN, _proposal
from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import run_confidence_verification as driver
from tools.dspark import run_large_batch as large
from tools.dspark import startup_cost_profile as profile

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    else:
        monkeypatch.setattr(torch, "npu", NS(is_current_stream_capturing=lambda: False), raising=False)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_nan", _NAN)


def test_default_prefix_has_all_nine_predecessors():
    points = profile.grid(64, [6, 12, 24, 48, 96, 192, 384], [128, 2048], 512)[1]
    chosen = profile.diagnostic_points(points, "ctx128-n4-t12-skewed")
    assert len(chosen) == 10 and chosen == points[:10]
    assert chosen[-2]["id"] == "ctx128-n4-t12-balanced"
    assert chosen[-1]["lengths"] == [5, 3, 0, 0]
    with pytest.raises(ValueError):
        profile.diagnostic_points(points, "missing")


def test_entrypoint_is_separate_from_streaming_performance_guard(tmp_path, monkeypatch):
    options = {"mode": "specified_lengths", "profile": True, "lengths": [5]}
    monkeypatch.setattr(
        benchmark, "build_engine_kwargs", lambda _: {"additional_config": {"dspark_confidence_verification": options}}
    )
    normal = profile.profile_engine_kwargs(None, tmp_path, False)
    assert "dspark_profile_nan_diagnostic_dir" not in normal["additional_config"]
    diagnostic = profile.profile_engine_kwargs(None, tmp_path, True)
    assert diagnostic["additional_config"]["dspark_profile_nan_diagnostic_dir"] == str(tmp_path)
    options["profile"] = False
    with pytest.raises(ValueError):
        profile.profile_engine_kwargs(None, tmp_path, True)
    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--model-dir",
                "/model",
                "--mode",
                "dspark",
                "--measurement-protocol",
                "async_stream",
                "--dspark-nan-diagnostic-dir",
                str(tmp_path),
            ]
        )


def test_cli_forwards_only_to_isolated_profile(tmp_path, monkeypatch):
    base = ["--plugin-sha", "abc", "--manifest", str(tmp_path / "manifest"), "--output-dir", str(tmp_path)]
    observed = []
    monkeypatch.setattr(large, "run", lambda args: observed.append(args) or 0)
    assert large.main(base + ["--stage", "profile", "--batches", "64", "--profile-nan-diagnostic"]) == 0
    args = observed[0]
    cmd = large.command(args, 64, tmp_path)
    assert (
        "--profile-nan-diagnostic" in cmd and cmd[cmd.index("--profile-stop-after-point") + 1] == "ctx128-n4-t12-skewed"
    )
    for stage, batches in [("validate", ["64"]), ("profile", ["64", "128"])]:
        with pytest.raises(SystemExit):
            large.main(base + ["--stage", stage, "--batches", *batches, "--profile-nan-diagnostic"])
    with pytest.raises(SystemExit):
        driver.main(base + ["--stage", "extend", "--batch", "64", "--profile-nan-diagnostic"])
    with pytest.raises(ValueError, match="cannot produce a cost table"):
        profile.compile_startup(
            [], {"diagnostic_only": True}, [], checkpoint={}, plugin_sha="abc", raw_hashes=[], overhead=0
        )


def make_runner(directory):
    class Runner:
        def __init__(self):
            self.vllm_config = NS(additional_config={"dspark_profile_nan_diagnostic_dir": str(directory)})
            self.speculator = NS(
                rank=0,
                _proposal_step_epoch=0,
                _published_proposal_owners={},
                confidence_verification=NS(options={"profile": True, "mode": "specified_lengths"}, last_selection=None),
            )
            self.cudagraph_manager = ReplayManager()
            self.block_tables = NS(
                input_block_tables=[torch.tensor([[1, 2]] * 4)],
                num_blocks=NS(np=np.ones((1, 4), dtype=int)),
                block_sizes=[32],
                kernel_block_sizes=[32],
            )
            self.positions = torch.arange(12)
            self.offsets = torch.zeros(5, dtype=torch.int32)

        def execute_model(
            self, scheduled, intermediate_tensors=None, dummy_run=False, skip_attn_for_dummy_run=False, is_profile=False
        ):
            names, queries = list(scheduled.num_scheduled_tokens), list(scheduled.num_scheduled_tokens.values())
            self.offsets[: len(names) + 1] = torch.tensor([0, *np.cumsum(queries)], dtype=torch.int32)
            self.cudagraph_manager.run_fullgraph(NS(cg_mode="FULL", num_tokens=12))
            self.speculator.confidence_verification.last_selection = {
                "lengths": {k: q - 1 for k, q in zip(names, queries)}
            }
            batch = NS(
                req_ids=names,
                num_reqs=len(names),
                num_tokens=sum(queries),
                num_tokens_after_padding=12,
                query_start_loc_np=self.offsets.numpy(),
                query_start_loc=self.offsets,
                seq_lens=torch.tensor([200] * len(names)),
                seq_lens_np=np.array([200] * len(names)),
                num_computed_tokens_np=np.array([203] * len(names)),
                idx_mapping_np=np.array([2, 1, 0, 3][: len(names)]),
                num_scheduled_tokens=np.array(queries),
                positions=self.positions,
                logits_indices=torch.arange(sum(queries)),
                cu_num_logits=self.offsets,
            )
            self.execute_model_state = NS(
                input_batch=batch,
                hidden_states=torch.ones(12, 8),
                aux_hidden_states=[],
                slot_mappings_by_layer={"target": torch.arange(12, dtype=torch.int32)},
            )
            return "target"

    return Runner()


def test_local_first_failure_tracks_exit_reorder_epoch_and_boundaries(tmp_path):
    runner = make_runner(tmp_path)
    observer = _EXTENSION._FullReplayObserver(runner)
    diag = observer.nan_diagnostic
    diag.profile_point = {"id": "ctx128-n4-t12-skewed", "specified_lengths": [5, 3, 0, 0]}
    for epoch, mapping in enumerate([{"r2": 1, "r3": 1, "r1": 4, "r0": 6}, {"r3": 1, "r2": 4, "r1": 6}], 1):
        scheduled = NS(
            num_scheduled_tokens=mapping,
            total_num_scheduled_tokens=sum(mapping.values()),
            finished_req_ids=["r0"] if epoch == 2 else [],
        )
        assert runner.execute_model(scheduled) == "target"
        proposal = _proposal(size=len(mapping), epoch=epoch)
        proposal.request_ids = tuple(mapping)
        proposal.num_target_tokens = sum(mapping.values())
        proposal.request_state_indices = torch.tensor([2, 1, 0, 3][: len(mapping)])
        proposal.target_query_start_loc = runner.offsets[: len(mapping) + 1]
        proposal.target_sequence_lengths = torch.tensor([200] * len(mapping))
        proposal.num_sampled = torch.tensor(list(mapping.values()))
        proposal.num_rejected = torch.zeros(len(mapping), dtype=torch.int32)
        proposal.draft_query_slot_mappings = {"draft": torch.arange(len(mapping) * 5)}
        proposal.draft_block_tables = {0: torch.tensor([[1, 2]] * len(mapping))}
        runner.speculator._proposal_step_epoch = epoch
        diag.proposal_inputs(proposal)
        diag.check("draft_hidden", {"hidden": torch.ones(len(mapping) * 5, 8)}, len(mapping) * 5)
        logits = torch.ones(len(mapping) * 5, 8)
        if epoch == 2:
            logits[6, 0] = torch.nan
            with pytest.raises(RuntimeError, match="base_logits"):
                diag.check("base_logits", {"logits": logits}, len(mapping) * 5)
        else:
            diag.check("base_logits", {"logits": logits}, len(mapping) * 5)
    path = tmp_path / "rank-0-first-failure.json"
    first = path.read_bytes()
    evidence = json.loads(first)
    current = evidence["current"]
    assert current["target_execution_epoch"] == current["proposal_epoch"] == 2
    assert current["target_runtime"] == "FULL" and current["actual_row_mapping"]["query_lengths"] == [1, 4, 6]
    assert current["finished_request_ids"] == ["r0"]
    assert current["proposal_request_ids"] == ["r3", "r2", "r1"]
    assert current["profile_point"]["id"] == "ctx128-n4-t12-skewed"
    assert evidence["previous_executions"][0]["actual_row_mapping"]["query_lengths"] == [1, 1, 4, 6]
    assert evidence["previous_executions"][0]["actual_row_mapping"]["query_start_loc"]["values"] == [0, 1, 2, 6, 12]
    assert (
        current["actual_row_mapping"]["query_start_loc"]["storage_data_ptr"]
        == runner.offsets.untyped_storage().data_ptr()
    )
    diag.failed_execution("ownership_cascade", ValueError("missing owner"))
    assert path.read_bytes() == first  # saved before propagation/RPC/next error
    assert evidence["performance_eligible"] is False
    with pytest.raises(ValueError, match="fresh"):
        _NAN.DSparkNaNDiagnostics(str(tmp_path), 0)


def test_worker_rejects_profile_observer_for_formal_mode(tmp_path):
    runner = make_runner(tmp_path)
    runner.speculator.confidence_verification.options = {"mode": "confidence"}
    with pytest.raises(ValueError, match="isolated"):
        _EXTENSION._FullReplayObserver(runner)


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("mode", ["full", "baseline", "metadata-only", "context-kv-sync", "numeric-boundaries"])
def test_diagnostic_run_never_compiles_costs(tmp_path, monkeypatch, fails, mode):
    args = NS(
        output_dir=tmp_path / "run",
        model=tmp_path / "model",
        batch=64,
        capture=[6, 12, 24, 48, 96, 192, 384],
        profile_contexts=[128, 2048],
        profile_output_tokens=512,
        profile_warmup=2,
        profile_samples=5,
        max_model_len=8192,
        profile_nan_diagnostic=mode == "full",
        profile_experiment=None if mode == "full" else mode,
        profile_stop_after_point="ctx128-n4-t12-skewed",
    )
    monkeypatch.setattr(profile.suite, "source_gate", lambda _: None)
    monkeypatch.setattr(profile, "checkpoint_preflight", lambda _: {})
    monkeypatch.setattr(profile.suite, "resources_idle", lambda _: None)
    monkeypatch.setattr(
        profile.suite, "create_plan", lambda *args: {"runs": [{"command": ["python", "benchmark", "--no-ignore-eos"]}]}
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda _: NS())
    monkeypatch.setattr(benchmark, "_sampling_params", lambda _: None)
    collected = []

    def collect(factory, points, sampling, root, **kwargs):
        collected.append(points)
        if fails:
            raise RuntimeError("synthetic worker failure")
        return [], {}

    monkeypatch.setattr(profile, "collect", collect)
    monkeypatch.setattr(profile, "compile_startup", lambda *args, **kwargs: pytest.fail("Diagnostic costs compiled"))
    if fails:
        with pytest.raises(RuntimeError, match="worker failure"):
            profile.run(args)
    else:
        assert profile.run(args) == 0
    receipt = json.loads((args.output_dir / "diagnostic.json").read_text())
    assert receipt["performance_eligible"] is False
    assert receipt["status"] == ("failed" if fails else "completed_without_observed_failure")
    assert len(collected) == 1 and len(collected[0]) == 10
    assert not (args.output_dir / "cost-profile.json").exists()
