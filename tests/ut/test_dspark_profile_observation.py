# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU tensors/arrays; no claim of NPU execution or numerical repair."""

import hashlib
import importlib.util
import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark import run_large_batch as large
from tools.dspark import startup_cost_profile as profile

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "profile_observation_under_test", ROOT / "vllm_ascend/diagnostics/dspark_profile_observation.py"
)
OBS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBS)


@dataclass
class Proposal:
    step_epoch: int
    request_ids: tuple
    request_state_indices: object
    target_query_start_loc: object
    num_sampled: object
    num_rejected: object


class Model:
    def combine_hidden_states(self, hidden):
        return hidden

    def precompute_and_store_context_kv(self, hidden, positions, slots):
        return hidden

    def compute_draft_logits(self, hidden):
        return hidden


class Speculator:
    def __init__(self):
        self.rank = 0
        self.model = Model()
        self._proposal_step_epoch = 0
        self._published_proposal_owners = {}
        self.confidence_verification = NS(options={"profile": True, "mode": "specified_lengths"})
        self.hidden = torch.ones(20, 8)
        self.error = None

    def prepare_proposal_inputs(self, batch):
        self._proposal_step_epoch += 1
        return Proposal(
            self._proposal_step_epoch,
            tuple(batch.req_ids),
            batch.idx_mapping,
            batch.query_start_loc,
            torch.ones(batch.num_reqs),
            torch.zeros(batch.num_reqs),
        )

    def _run_draft_model_forward(self, proposal, metadata):
        return self.hidden

    def _build_draft_forward_metadata(self, execution):
        return {}

    def _execute_sequential_markov_sampling(self, proposal, hidden):
        logits = self.model.compute_draft_logits(hidden)
        if self.error is not None:
            raise self.error
        return logits

    def propose(self, proposal):
        hidden = self.model.combine_hidden_states(self.hidden)
        self.model.precompute_and_store_context_kv(hidden, proposal.target_query_start_loc, {})
        hidden = self._run_draft_model_forward(proposal, {})
        return self._execute_sequential_markov_sampling(proposal, hidden)


class Runner:
    def __init__(self):
        self.vllm_config = NS(additional_config={})
        self.speculator = Speculator()
        self.cudagraph_manager = NS(run_fullgraph=lambda desc: self.speculator.hidden)
        self.query_host = np.zeros(6, dtype=np.int32)
        self.seq_host = np.zeros(5, dtype=np.int32)
        self.query_device = torch.zeros(6, dtype=torch.int32)
        self.input_batch = None

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
    ):
        ids = list(scheduler_output.num_scheduled_tokens)
        q = list(scheduler_output.num_scheduled_tokens.values())
        n = len(ids)
        self.query_host[: n + 1] = [0, *np.cumsum(q)]
        self.seq_host[:n] = np.arange(n) + 128 + self.speculator._proposal_step_epoch
        batch = NS(
            req_ids=ids,
            num_reqs=n,
            num_tokens=sum(q),
            num_tokens_after_padding=12,
            num_reqs_after_padding=6,
            query_start_loc_np=self.query_host,
            seq_lens_np=self.seq_host,
            num_scheduled_tokens=np.array(q),
            is_prefilling_np=np.zeros(n, dtype=bool),
            idx_mapping_np=np.array([2, 0, 3, 1][:n]),
            idx_mapping=torch.arange(n),
            query_start_loc=self.query_device,
            num_computed_tokens_np=np.ones(n, dtype=int) * 140,
        )
        self.input_batch = batch
        self.cudagraph_manager.run_fullgraph(NS(num_tokens=12, cg_mode="FULL"))
        self.execute_model_state = NS(input_batch=batch, hidden_states=self.speculator.hidden, aux_hidden_states=[])
        return "target"


def installed(tmp_path, mode="metadata-only"):
    runner = Runner()
    return runner, OBS.ProfileObservation(runner, {"mode": mode, "directory": str(tmp_path)})


def execute(runner, mapping):
    scheduler = NS(num_scheduled_tokens=mapping, finished_req_ids=["old-request"])
    assert runner.execute_model(scheduler) == "target"
    proposal = runner.speculator.prepare_proposal_inputs(runner.input_batch)
    return runner.speculator.propose(proposal)


def test_metadata_only_no_device_reads_waits_or_hot_path_writes(tmp_path, monkeypatch):
    runner, observer = installed(tmp_path)
    observer.begin_point("skewed")
    forbidden = lambda *a, **k: pytest.fail("metadata observation invoked a forbidden operation")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "item", forbidden)
        patch.setattr(torch, "isnan", forbidden)
        patch.setattr(torch, "isfinite", forbidden)
        # Preserve a real backend module if installed.
        if hasattr(torch, "npu"):
            patch.setattr(torch.npu, "synchronize", forbidden)
            patch.setattr(torch.npu, "current_stream", forbidden)
        else:
            patch.setattr(torch, "npu", NS(synchronize=forbidden, current_stream=forbidden), raising=False)
        patch.setattr(observer, "write", forbidden)
        execute(runner, {"r2": 1, "r3": 1, "r1": 4, "r0": 6})
        first = observer.snapshot()
        before = json.dumps(first)
        execute(runner, {"r3": 1, "r2": 4, "r1": 6})
        assert json.dumps(first) == before  # not numpy/tensor views
    assert list(tmp_path.iterdir()) == []
    receipt = observer.finish_point()
    records = json.loads(Path(receipt["local_evidence"]["path"]).read_text())["records"]
    prepared = [r for r in records if r["stage"] == "proposal_prepare.return"]
    assert prepared[0]["batch"]["query_start_loc_np"] == [0, 1, 2, 6, 12]
    assert prepared[1]["batch"]["query_start_loc_np"] == [0, 1, 5, 11]
    assert prepared[1]["batch"]["request_ids"] == ["r3", "r2", "r1"]
    for row in prepared:
        assert row["batch"]["buffers"]["query_start_loc"]["data_ptr"] == runner.query_device.data_ptr()
        assert row["payload"]["result"]["num_sampled"]["values"] == "unavailable"
    assert observer.snapshot()["recording_error"] is None
    assert observer.snapshot()["sync"]["calls"] == 0


def test_bounded_ring_local_error_before_propagation_preserves_original(tmp_path):
    runner, observer = installed(tmp_path)
    observer.begin_point("p1")
    execute(runner, {"r0": 6, "r1": 4, "r2": 1, "r3": 1})
    observer.begin_point("p2")
    for _ in range(12):
        execute(runner, {"r3": 1, "r2": 4, "r1": 6})
    error = RuntimeError("Ascend DSpark Markov base logits contain NaN.")
    runner.speculator.error = error
    with pytest.raises(RuntimeError) as caught:
        execute(runner, {"r2": 4, "r1": 6})
    assert caught.value is error
    path = tmp_path / "rank-0-first-failure.json"
    evidence = json.loads(path.read_text())
    assert len(evidence["records"]) == OBS.RING_RECORDS
    assert {r["point"] for r in evidence["records"]} == {"p2"}
    assert evidence["failure"]["stage"] == "markov"
    assert any(r["stage"] == "base_logits.return" for r in evidence["records"])
    assert any(r["stage"] == "markov.return" for r in evidence["records"])
    assert evidence["performance_eligible"] is False
    saved = path.read_bytes()
    observer.failed("ownership", RuntimeError("Scheduled candidates lack current proposal owners"))
    assert path.read_bytes() == saved
    observer.close()
    assert observer.point is None and not observer.records and not observer.hooks
    assert "propose" not in vars(runner.speculator)
    assert "compute_draft_logits" not in vars(runner.speculator.model)


def test_error_in_recorder_cannot_replace_original_exception(tmp_path, monkeypatch):
    runner, observer = installed(tmp_path)
    observer.begin_point("point")
    monkeypatch.setattr(observer, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    error = RuntimeError("device execution failed")
    runner.speculator.error = error
    with pytest.raises(RuntimeError) as caught:
        execute(runner, {"same-prompt-instance-1": 1})
    assert caught.value is error
    observer.close()


def test_sync_control_changes_only_context_return_and_counts(tmp_path, monkeypatch):
    runner, observer = installed(tmp_path, "context-kv-sync")
    waits = []
    current = lambda: NS(npu_stream=17, synchronize=lambda: waits.append("current"))
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "current_stream", current)
    else:
        monkeypatch.setattr(torch, "npu", NS(current_stream=current), raising=False)
    assert len(observer.hooks) == 1
    observer.begin_point("one")
    for _ in range(2):
        execute(runner, {"instance-a": 1, "instance-b": 6})
    snapshot = observer.finish_point()
    assert snapshot["records_count"] == 0
    assert snapshot["sync"]["calls"] == snapshot["sync"]["completed"] == 2
    assert snapshot["sync"]["stream_handle"] == "17"
    observer.begin_point("two")
    execute(runner, {"instance-b": 6})
    assert observer.snapshot()["sync"]["calls"] == 1
    observer.close()
    execute(runner, {"instance-b": 6})
    assert len(waits) == 3


@pytest.mark.parametrize("mode", ["baseline", "metadata-only", "context-kv-sync", "numeric-boundaries"])
def test_prefix_controls_and_cli_do_not_enable_full_diagnostic(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(
        benchmark,
        "build_engine_kwargs",
        lambda _: {
            "additional_config": {"dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"}}
        },
    )
    kwargs = profile.profile_engine_kwargs(None, tmp_path, False, mode)
    assert "dspark_profile_nan_diagnostic_dir" not in kwargs["additional_config"]
    assert ("dspark_profile_observation" in kwargs["additional_config"]) == (mode != "baseline")
    assert ("distributed_executor_backend" in kwargs) == (mode in ("metadata-only", "numeric-boundaries"))
    assert ("dspark_profile_failure_dir" in kwargs["additional_config"]) == (
        mode in ("metadata-only", "numeric-boundaries")
    )
    if mode in ("metadata-only", "numeric-boundaries"):
        assert kwargs["distributed_executor_backend"].endswith("dspark_profile_executor.ProfileMultiprocExecutor")
    base = ["--plugin-sha", "abc", "--manifest", str(tmp_path / "manifest"), "--output-dir", str(tmp_path)]
    observed = []
    monkeypatch.setattr(large, "run", lambda args: observed.append(args) or 0)
    large.main(base + ["--stage", "profile", "--batches", "64", "--profile-experiment", mode])
    cmd = large.command(observed[-1], 64, tmp_path)
    assert cmd[cmd.index("--profile-experiment") + 1] == mode
    assert "--profile-nan-diagnostic" not in cmd
    assert cmd[cmd.index("--profile-stop-after-point") + 1] == "ctx128-n4-t12-skewed"
    with pytest.raises(SystemExit):
        large.main(base + ["--stage", "validate", "--batches", "64", "--profile-experiment", mode])
    with pytest.raises(SystemExit):
        large.main(
            base + ["--stage", "profile", "--batches", "64", "--profile-experiment", mode, "--profile-nan-diagnostic"]
        )


@pytest.mark.parametrize("mode", ["metadata-only", "context-kv-sync", "numeric-boundaries"])
def test_reject_performance_and_full_diagnostic_engines(tmp_path, mode):
    runner = Runner()
    runner.speculator.confidence_verification.options["profile"] = False
    with pytest.raises(ValueError, match="isolated"):
        OBS.ProfileObservation(runner, {"mode": mode, "directory": str(tmp_path)})
    runner.speculator.confidence_verification.options["profile"] = True
    runner.vllm_config.additional_config["dspark_profile_nan_diagnostic_dir"] = str(tmp_path)
    with pytest.raises(ValueError, match="isolated"):
        OBS.ProfileObservation(runner, {"mode": mode, "directory": str(tmp_path)})


def test_real_cpu_nan_assertion_still_rejects_and_saves_first_boundary(tmp_path):
    runner = Runner()

    def checked_markov(proposal, hidden):
        logits = runner.speculator.model.compute_draft_logits(hidden)
        torch._assert_async(~torch.isnan(logits).any(), "base logits contain NaN")
        return logits

    runner.speculator._execute_sequential_markov_sampling = checked_markov
    observer = OBS.ProfileObservation(runner, {"mode": "metadata-only", "directory": str(tmp_path)})
    observer.begin_point("skewed")
    execute(runner, {"r3": 1, "r2": 4, "r1": 6})
    runner.speculator.hidden[5, 0] = torch.nan
    with pytest.raises(RuntimeError, match="base logits contain NaN"):
        execute(runner, {"r2": 4, "r1": 6})
    first = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    assert first["failure"]["stage"] == "markov"
    assert first["records"][-1]["epochs"]["_proposal_step_epoch"] == 2
    assert first["records"][-1]["batch"]["request_ids"] == ["r2", "r1"]
    assert first["records"][-2]["stage"] == "base_logits.return"
    assert torch.isnan(runner.speculator.hidden[5, 0])  # never masked/replaced
    observer.close()


def test_no_forward_or_early_failure_cannot_label_previous_batch_as_current(tmp_path):
    runner = Runner()
    original = runner.execute_model

    def can_skip(scheduled):
        if not scheduled.num_scheduled_tokens:
            runner.execute_model_state = None
            return None
        if "unknown" in scheduled.num_scheduled_tokens:
            raise ValueError("Scheduled candidates lack current proposal owners")
        return original(scheduled)

    runner.execute_model = can_skip
    observer = OBS.ProfileObservation(runner, {"mode": "metadata-only", "directory": str(tmp_path)})
    observer.begin_point("first")
    execute(runner, {"r0": 6, "r1": 4, "r2": 1, "r3": 1})
    runner.execute_model(NS(num_scheduled_tokens={}, finished_req_ids=["r0"]))
    assert observer.records[-1]["batch"] is None
    observer.begin_point("second")
    with pytest.raises(ValueError):
        runner.execute_model(NS(num_scheduled_tokens={"unknown": 6}, finished_req_ids=[]))
    evidence = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    assert evidence["records"][-1]["batch"] is None
    assert evidence["records"][-2]["scheduler"]["scheduled"] == {"unknown": 6}
    observer.close()


def test_installs_through_real_profile_wrapper_and_replay_observer(tmp_path, monkeypatch):
    from tests.ut.test_dspark_graph_replay import _EXTENSION
    from tests.ut.test_dspark_startup_cost_profile import POLICY

    recorded = []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            recorded.append("record")

        def elapsed_time(self, end):
            return 2.5

    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "Event", Event)
        monkeypatch.setattr(torch.npu, "synchronize", lambda: recorded.append("boundary"))
    else:
        monkeypatch.setattr(
            torch, "npu", NS(Event=Event, synchronize=lambda: recorded.append("boundary")), raising=False
        )
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_observation", OBS)
    monkeypatch.setitem(sys.modules, "vllm_ascend.spec_decode.dspark_verification", NS(**POLICY))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.worker.v2.spec_decode.dspark.verification_runtime", NS(runtime_identity=lambda *a: {})
    )
    Profiler = runpy.run_path(str(ROOT / "vllm_ascend/diagnostics/dspark_cost_profile.py"))["IsolatedCostProfiler"]
    runner = Runner()
    runner.vllm_config.model_config = NS(max_model_len=8192)
    runner.vllm_config.additional_config["dspark_profile_observation"] = {
        "mode": "metadata-only",
        "directory": str(tmp_path),
    }
    runner.device = "cpu"
    if hasattr(torch.npu, "get_device_name"):
        monkeypatch.setattr(torch.npu, "get_device_name", lambda _: "mock NPU")
    else:
        monkeypatch.setattr(torch.npu, "get_device_name", lambda _: "mock NPU", raising=False)
    runner.speculator.confidence_verification.receipt = {"weights_sha256": "test"}
    runner.speculator._execute_draft = runner.speculator.propose
    profiler = Profiler(runner)
    observer = _EXTENSION._FullReplayObserver(runner)  # actual installation order
    assert observer.nan_diagnostic is None
    profiler.begin_point("point", [5, 3, 0, 0])
    runner.execute_model(NS(num_scheduled_tokens={"r2": 1, "r3": 1, "r1": 4, "r0": 6}, finished_req_ids=[]))
    runner.input_batch.is_prefilling_np = np.zeros(4, dtype=bool)
    proposal = runner.speculator.prepare_proposal_inputs(runner.input_batch)
    # num_reqs is a real ProposalInputs scalar in production.
    proposal.num_reqs = 4
    runner.speculator._execute_draft(proposal)
    snapshot = profiler.snapshot()
    assert snapshot["identity"]["diagnostic_only"] is True
    assert snapshot["observation"]["point"] == "point"
    assert snapshot["observation"]["recording_error"] is None
    assert snapshot["observation"]["stage_counts"]["base_logits.return"] == 1
    assert [m["kind"] for m in snapshot["measurements"]] == ["target", "draft"]
    assert observer.snapshot()["records"][0]["count"] == 1
    # Exactly the original timer events and point-boundary synchronizations.
    assert recorded == ["boundary", "record", "record", "record", "record", "boundary"]
    profiler.observation.close()


def test_point_rpc_is_compact_hashed_and_safe_with_full_history_local(tmp_path, monkeypatch):
    msgspec = pytest.importorskip("msgspec")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    runner, observer = installed(tmp_path)
    observer.begin_point("point-two")
    for _ in range(15):
        execute(runner, {"r3": 1, "r2": 4, "r1": 6})
    receipt = observer.finish_point()
    raw = Path(receipt["local_evidence"]["path"]).read_bytes()
    data = json.loads(raw)
    assert receipt["local_evidence"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["local_evidence"]["bytes"] == len(raw)
    assert receipt["records_count"] == len(data["records"]) == OBS.RING_RECORDS
    assert data["point"] == "point-two" and data["pid"] > 0
    assert data["observed_utc"] and data["recording_error"] is None
    wire = msgspec.msgpack.encode(receipt)
    assert len(wire) < 4096 and msgspec.msgpack.decode(wire) == receipt
    observer.begin_point("point-three")
    assert not observer.records and not observer.counts
    assert json.loads(raw)["point"] == "point-two"
    observer.close()


def test_descriptors_limit_total_expansion_and_represent_aliases():
    leaf = {str(i): torch.zeros(1) for i in range(128)}
    # Many layers can share one metadata object. Record the identity link once
    # instead of recursively multiplying the same payload for every layer.
    aliases = OBS.describe({str(i): leaf for i in range(128)})
    assert aliases["0"]["object_id"] == aliases["127"]["object_id"]
    assert aliases["127"]["fields"] == "already_described"
    nested = {str(i): {str(j): {str(k): k for k in range(128)} for j in range(128)} for i in range(10)}
    described = OBS.describe(nested)
    assert described["truncated_fields"] > 0
    assert len(json.dumps(described)) < 20000
