# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual decoder statements and recorded ATen replay; NPU validation pending."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import pytest

from tests.ut.test_dspark_profile_auxiliary import fixture, load_auxiliary
from tests.ut.test_dspark_profile_observation import ROOT, torch
from tests.ut.test_dspark_replay_diagnostics import CPURecordedGraph, _production_method, decoder
from tools.dspark import run_confidence_verification as driver
from tools.dspark import run_large_batch as large
from tools.dspark import startup_cost_profile as profile


def load_target(monkeypatch):
    name = "vllm_ascend.diagnostics.dspark_profile_attention"
    attention_spec = importlib.util.spec_from_file_location(
        name, ROOT / "vllm_ascend/diagnostics/dspark_profile_attention.py"
    )
    attention_module = importlib.util.module_from_spec(attention_spec)
    monkeypatch.setitem(sys.modules, name, attention_module)
    attention_spec.loader.exec_module(attention_module)
    auxiliary = load_auxiliary(monkeypatch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.diagnostics.dspark_profile_auxiliary", auxiliary)
    spec = importlib.util.spec_from_file_location(
        "dspark_target_test", ROOT / "vllm_ascend/diagnostics/dspark_profile_target.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def make_bank(module, sizes=(6, 12, 24, 48, 96, 192, 384), auxiliary=(40, 41, 42), target_layer=None, attention=False):
    return module.TargetBoundaryFlags(
        sizes=sizes,
        auxiliary_layers=auxiliary,
        start_layer=0,
        end_layer=43,
        hidden_size=4,
        hc_mult=2,
        device="cpu",
        target_layer=target_layer,
        attention=attention,
    )


def target_fixture(
    tmp_path, monkeypatch, *, capacity=12, layer=None, stage="attn_output", target_layer=None, attention_factory=None
):
    module = load_target(monkeypatch)
    bank = make_bank(module, target_layer=target_layer, attention=attention_factory is not None)
    fault_layer = layer if layer is not None else (40 if target_layer is None else target_layer)
    fault = torch.zeros(capacity, 4)
    layers = [decoder(i, bank if i in bank.layers else None, torch.zeros_like(fault)) for i in range(43)]
    leaf = layers[fault_layer]
    if attention_factory is not None:
        leaf.self_attn = attention_factory(bank, fault)
    elif stage == "attn_input":
        leaf.input_layernorm.forward = lambda x: x + fault
    elif stage == "attn_output":
        leaf.self_attn = lambda *, hidden_states, **kwargs: hidden_states + fault
    elif stage == "ffn_input":
        leaf.post_attention_layernorm.forward = lambda x: x + fault
    elif stage == "ffn_output":
        leaf.mlp = lambda x, **kwargs: x * 0.5 + fault
    elif stage != "input":
        # CPURecordedGraph freezes actual ATen operations in their two source
        # call sites. The counter is only a fixture for the custom HC kernel.
        post_calls = []

        def post(x, residual, *args):
            index = len(post_calls) % 2
            post_calls.append(index)
            inject = (index == 0) == (stage == "residual")
            return residual + (x + fault if inject else x)[:, None, :]

        leaf.hc_post = post

    def forward(raw, input_ids, positions):
        hidden = raw[0] + 1
        bank.write("embedding", hidden)
        hidden = hidden[:, None, :].repeat(1, 2, 1)
        auxiliary = []
        for i, layer in enumerate(layers):
            if stage == "input" and i == fault_layer:
                # Model a write between the previous layer and actual input.
                hidden = hidden + fault[:, None, :]
            hidden, _ = layer(positions, hidden, None, input_ids=input_ids)
            if i in (40, 41, 42):
                auxiliary.append(hidden.mean(1))
        return hidden.mean(1), auxiliary

    f = fixture(
        tmp_path,
        monkeypatch,
        capacity=capacity,
        mode="target-boundaries",
        target=NS(module=module, bank=bank, forward=forward),
    )
    f.bank, f.fault, f.target_module = bank, fault, module
    return f


def test_plan_is_small_and_default_off_does_not_allocate(monkeypatch):
    module = load_target(monkeypatch)
    bank = make_bank(module)
    assert len(bank.names) == 15 and bank.allocated_bytes == 11648
    assert bank.layers == (0, 1, 2, 3, 9, 19, 29, 39, 40)
    assert bank.names[-6:] == tuple(f"layer.40.{x}" for x in module.DETAILED_BOUNDARIES)
    monkeypatch.setattr(module.torch, "empty", lambda *a, **k: pytest.fail("off mode allocated a bank"))
    for mode in (None, "metadata-only", "numeric-boundaries", "auxiliary-transfers"):
        model = NS(_dspark_layer_snapshots=None)
        config = NS(additional_config={"dspark_profile_observation": {"mode": mode}})
        module.install_target_boundaries(model, config)
        assert model._dspark_layer_snapshots is None


@pytest.mark.parametrize("sizes,auxiliary", [((), (40,)), ((0,), (40,)), ((385,), (40,)), ((12,), (43,)), ((12,), ())])
def test_budget_and_scope_are_checked(monkeypatch, sizes, auxiliary):
    with pytest.raises(ValueError):
        make_bank(load_target(monkeypatch), sizes, auxiliary)


@pytest.mark.parametrize("target_layer", [-1, 43, True, 1.5, "1"])
def test_local_plan_rejects_invalid_layer(monkeypatch, target_layer):
    with pytest.raises(ValueError, match="decoder index"):
        make_bank(load_target(monkeypatch), target_layer=target_layer)


def test_local_plan_reuses_bank_and_omits_distant_cuts(monkeypatch):
    module = load_target(monkeypatch)
    bank = make_bank(module, target_layer=1)
    assert bank.layers == (0, 1) and bank.allocated_bytes == 6992
    assert bank.names == ("embedding", "layer.0.output", "layer.1.input") + tuple(
        f"layer.1.{stage}" for stage in module.DETAILED_BOUNDARIES
    )
    assert bank.tails["layer.1.input"] == [2, 4] and bank.tails["layer.1.attn_input"] == [4]
    graph = CPURecordedGraph()
    value = torch.ones(12, 2, 4)
    with graph:
        for stage in module.DETAILED_BOUNDARIES:
            bank.write(f"layer.40.{stage}", value)
    assert not graph.operations


def test_actual_layer_input_distinguishes_inter_layer_write(tmp_path, monkeypatch):
    f = target_fixture(tmp_path, monkeypatch, target_layer=1, stage="input")
    for epoch in (90, 91):
        f.run(epoch)
    f.fault[0].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="original Markov NaN"):
        f.run(92)
    d = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    rounds = d["auxiliary"]["rounds"]
    assert [x["proposal_epoch"] for x in rounds] == [90, 91, 92]
    assert d["target_internal"]["target_layer"] == 1 and d["recording_error"] is None
    b = rounds[-1]["target_internal"]["valid_row_brackets"][0]
    assert (b["last_observed_finite"], b["first_observed_nonfinite"]) == ("layer.0.output", "layer.1.input")
    assert b["request_id"] == "third" and b["request_row"] == 0
    assert [x["candidate_row"] for x in d["numeric"]["rounds"][-1]["rows"] if x["hidden_nan"]] == list(range(5))
    assert f.graph.calls == 3 and len(f.calls) == 2
    f.observer.close()


@pytest.mark.parametrize("target_layer", [None, 0, 1, 42])
def test_local_layer_flows_through_server_entrypoints(tmp_path, monkeypatch, target_layer):
    base = ["--plugin-sha", "abc", "--manifest", str(tmp_path / "manifest"), "--output-dir", str(tmp_path)]
    extra = [] if target_layer is None else ["--profile-target-layer", str(target_layer)]
    seen = []
    monkeypatch.setattr(large, "run", lambda args: seen.append(args) or 0)
    monkeypatch.setattr(driver, "run", lambda args: seen.append(args) or 0)
    assert (
        large.main(
            base + ["--stage", "profile", "--batches", "64", "--profile-experiment", "target-boundaries"] + extra
        )
        == 0
    )
    cmd = large.command(seen[-1], 64, tmp_path)
    assert ("--profile-target-layer" in cmd) == (target_layer is not None)
    assert driver.main(cmd[2:]) == 0
    assert seen[-1].profile_target_layer == target_layer
    options = {"mode": "specified_lengths", "profile": True}
    monkeypatch.setattr(
        profile.benchmark,
        "build_engine_kwargs",
        lambda _: {"additional_config": {"dspark_confidence_verification": options}},
    )
    kw = profile.profile_engine_kwargs(None, tmp_path, False, "target-boundaries", seen[-1].profile_target_layer)
    observation = kw["additional_config"]["dspark_profile_observation"]
    assert observation.get("target_layer") == target_layer
    assert observation["mode"] == "target-boundaries"
    if target_layer is None:
        assert "target_layer" not in observation


@pytest.mark.parametrize("mode,layer", [("baseline", 1), ("numeric-boundaries", 1), ("target-boundaries", -1)])
def test_layer_option_requires_target_profile(tmp_path, monkeypatch, mode, layer):
    base = [
        "--plugin-sha",
        "abc",
        "--manifest",
        str(tmp_path),
        "--output-dir",
        str(tmp_path),
        "--stage",
        "profile",
        "--profile-experiment",
        mode,
        "--profile-target-layer",
        str(layer),
    ]
    for entry, batcharg in ((large, "--batches"), (driver, "--batch")):
        monkeypatch.setattr(entry, "run", lambda args: pytest.fail("invalid option started a run"))
        with pytest.raises(SystemExit):
            entry.main(base + [batcharg, "64"])
    with pytest.raises(ValueError, match="target-boundaries"):
        profile.profile_engine_kwargs(None, tmp_path, False, mode, layer)


@pytest.mark.parametrize("layer", [None, "0", "1"])
@pytest.mark.parametrize("attention", [False, True])
def test_shell_control_keeps_single_original_prefix(tmp_path, layer, attention):
    # Intercept only the child Bash command; execute the real control script.
    child = tmp_path / "bash"
    child.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    child.chmod(0o755)
    args = [
        "/bin/bash",
        str(ROOT / "tools/dspark/run_dspark_profile_control.sh"),
        "sha",
        "manifest",
        "target-boundaries",
    ]
    result = subprocess.run(
        args
        + ([] if layer is None else [layer])
        + (["--attention", "--worker-exit"] if attention and layer is not None else []),
        env=dict(os.environ, PATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    cmd = result.stdout.splitlines()
    assert ("--profile-target-attention" in cmd) == (attention and layer is not None)
    assert cmd[cmd.index("--batches") + 1] == "64"
    assert cmd[cmd.index("--profile-stop-after-point") + 1] == "ctx128-n4-t12-skewed"
    assert cmd[cmd.index("--profile-output-tokens") + 1] == "512"
    assert ("--profile-target-layer" in cmd) == (layer is not None)
    if layer is not None:
        assert cmd[cmd.index("--profile-target-layer") + 1] == layer


@pytest.mark.parametrize("stage", ["attn_input", "attn_output", "residual", "ffn_input", "ffn_output", "output"])
@pytest.mark.parametrize("target_layer", [None, 1])
def test_actual_decoder_replay_locates_selected_submodule(tmp_path, monkeypatch, stage, target_layer):
    f = target_fixture(tmp_path, monkeypatch, stage=stage, target_layer=target_layer)
    for epoch in (40, 41, 42):
        f.run(epoch)
    f.fault[0].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="original Markov NaN"):
        f.run(43)
    d = json.loads((tmp_path / "rank-0-first-failure.json").read_text())
    rounds = d["auxiliary"]["rounds"]
    assert [(x["execution"], x["proposal_epoch"]) for x in rounds] == [(2, 41), (3, 42), (4, 43)]
    internal = rounds[-1]["target_internal"]
    assert internal["coverage"] == "FULL" and d["recording_error"] is None
    anchor = 40 if target_layer is None else target_layer
    assert internal["valid_row_brackets"][0]["first_observed_nonfinite"] == f"layer.{anchor}.{stage}"
    assert internal["valid_row_brackets"][0]["request_id"] == "third"
    assert internal["valid_row_brackets"][0]["last_observed_finite"] is not None
    assert all(not x["target_internal"]["valid_row_brackets"] for x in rounds[:-1])
    assert f.observer.numeric_transfers_completed == 4
    assert len(f.calls) == 2 and f.graph.calls == 4  # replay never calls Python model.forward
    saved = (tmp_path / "rank-0-target-first-nonfinite.json").read_bytes()
    f.observer.failed("markov", RuntimeError("original Markov NaN"))
    assert (tmp_path / "rank-0-target-first-nonfinite.json").read_bytes() == saved
    f.observer.execution += 1
    f.observer.failed("target_execute", ValueError("Scheduled candidates lack current proposal owners."))
    events = json.loads((tmp_path / "rank-0-error-events.json").read_text())["events"]
    assert [x["execution"] for x in events] == [4, 5]
    assert events[-1]["owner_epochs"] == {}
    for i in range(10):
        f.observer.execution += 1
        f.observer.failed("target_execute", RuntimeError(str(i)))
    assert len(f.observer.error_events) == f.target_module.MAX_ERROR_EVENTS
    f.observer.close()


@pytest.mark.parametrize(
    "layer,previous", [(0, "embedding"), (2, "layer.1.output"), (15, "layer.9.output"), (35, "layer.29.output")]
)
def test_coarse_checkpoints_do_not_misidentify_layer_40(tmp_path, monkeypatch, layer, previous):
    f = target_fixture(tmp_path, monkeypatch, layer=layer)
    f.fault[1].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="original Markov NaN"):
        f.run(501)
    bracket = f.observer.auxiliary_records[-1]["target_internal"]["valid_row_brackets"][0]
    assert bracket["last_observed_finite"] == previous and bracket["request_id"] == "second"
    assert bracket["first_observed_nonfinite"] != "layer.40.attn_input"
    f.observer.close()


@pytest.mark.parametrize("target_layer", [None, 1])
def test_nan_inf_mapping_padding_reorder_exit_and_reused_pool(tmp_path, monkeypatch, target_layer):
    f = target_fixture(tmp_path, monkeypatch, target_layer=target_layer)
    f.run(6)
    # Two requests now occupy the same local pool indices used by other IDs.
    f.fault[1].fill_(-torch.inf)
    f.fault[-1].fill_(torch.nan)  # padding alone does not create an owner or bracket
    f.run(20, lengths=(2, 3), names=("replacement", "third"))
    r = f.observer.auxiliary_records[-1]
    brackets = r["target_internal"]["valid_row_brackets"]
    assert [(x["row"], x["request_id"], x["position_in_request"], x["inf"]) for x in brackets] == [
        (1, "replacement", 1, True)
    ]
    assert r["pool_rows_cpu"] == [1, 0]
    assert r["device_integers"]["target.query_start_loc"] == [0, 2, 5]
    for b in r["target_internal"]["boundaries"]:
        assert all(x["request_id"] is None for x in b["rows"][5:])
    assert r["target_internal"]["boundaries"][-1]["rows"][-1]["nan"]
    snapshot = json.dumps(r)
    f.bank.flags.zero_()
    f.bank.receipts.fill_(900)
    assert json.dumps(r) == snapshot
    f.observer.close()


@pytest.mark.parametrize("target_layer", [None, 1])
def test_owned_packet_survives_bank_reuse_before_head_and_partial_point_drain(tmp_path, monkeypatch, target_layer):
    f = target_fixture(tmp_path, monkeypatch, target_layer=target_layer)
    f.fault[0].fill_(torch.inf)
    f.run(70, proposal=False)
    f.bank.flags.zero_()
    f.bank.receipts.zero_()
    f.observer.begin_point("next-point")
    record = json.loads((tmp_path / "rank-0-target-first-nonfinite.json").read_text())["auxiliary"]["rounds"][-1]
    assert record["point"] == "test-point" and record["proposal_epoch"] is None
    assert record["target_internal"]["coverage"] == "FULL"
    assert record["target_internal"]["valid_row_brackets"][0]["inf"]
    f.observer.close()


@pytest.mark.parametrize("fault", ["no_replay", "missing_cut", "replay_error"])
@pytest.mark.parametrize("target_layer", [None, 1])
def test_missing_or_failed_replay_is_not_finite_evidence(tmp_path, monkeypatch, fault, target_layer):
    f = target_fixture(tmp_path, monkeypatch, target_layer=target_layer)
    original = f.graph.replay

    def replay():
        if fault == "no_replay":
            return
        original()
        if fault == "missing_cut":
            f.bank.receipts[0].fill_(0)
        else:
            raise RuntimeError("original target failure")

    monkeypatch.setattr(f.graph, "replay", replay)
    if fault == "replay_error":
        with pytest.raises(RuntimeError, match="original target failure"):
            f.run(9)
        assert (
            json.loads((tmp_path / "rank-0-first-failure.json").read_text())["failure"]["message"]
            == "original target failure"
        )
    else:
        f.run(9)
    r = f.observer.auxiliary_records[-1]
    assert r["target_internal"]["coverage"] == "INVALID_RECEIPT" and not r["target_internal"]["boundaries"]
    assert f.observer.recording_error
    f.observer.close()


def test_unselected_boundaries_have_no_recorded_operators(monkeypatch):
    bank = make_bank(load_target(monkeypatch))
    graph = CPURecordedGraph()
    value = torch.ones(12, 2, 4)
    with graph:
        bank.write("pre_hc", value)
        bank.write("layer.8.output", value)
    assert not graph.operations


@pytest.mark.parametrize("boundary,tail", [("embedding", (4,)), ("layer.1.input", (2, 4))])
def test_large_profile_fx_graph_replays_small_shapes_without_dynamo_guards(monkeypatch, boundary, tail):
    bank = make_bank(load_target(monkeypatch), target_layer=1)
    index = bank.names.index(boundary)
    graphs = []

    def backend(graph, examples):
        graphs.append((graph, examples))
        return graph.forward

    def write(value):
        bank.write(boundary, value)
        return value * 2

    profile_tokens = 8192
    torch.compile(write, backend=backend, fullgraph=True, dynamic=True)(torch.ones(profile_tokens, *tail))
    assert len(graphs) == 1
    graph, examples = graphs[0]
    for size in (6, 12, 384, 24, 1):
        value = torch.ones(size, *tail)
        value[-1, 0] = torch.nan
        args = []
        for example in examples:
            if isinstance(example, torch.SymInt):
                args.append(size if int(example) == profile_tokens else int(example))
            elif isinstance(example, torch.Tensor) and tuple(example.shape) == (profile_tokens, *tail):
                args.append(value)
            else:
                args.append(example)
        bank.epoch_input.fill_(size)
        bank.flags.zero_()
        graph(*args)
        assert bank.receipts[index].item() == size
        assert bank.flags[index, :size, 0].nonzero().flatten().tolist() == [size - 1]
        assert not bank.flags[index, size:].any()
    assert len(graphs) == 1


def test_storage_error_cannot_replace_original_nan(tmp_path, monkeypatch):
    f = target_fixture(tmp_path, monkeypatch)
    f.fault[0].fill_(torch.nan)
    original = f.observer.write

    def write(suffix, *a, **kw):
        if suffix.startswith("target-") or suffix == "error-events":
            raise OSError("disk failure")
        return original(suffix, *a, **kw)

    monkeypatch.setattr(f.observer, "write", write)
    with pytest.raises(RuntimeError, match="original Markov NaN"):
        f.run(5)
    assert "disk failure" in f.observer.recording_error
    f.observer.close()


@pytest.mark.parametrize("target_layer", [None, 0, 1, 42])
def test_installation_binds_only_selected_target_layers_before_compile(monkeypatch, target_layer):
    module = load_target(monkeypatch)
    layers = [NS(_dspark_layer_snapshots=None) for _ in range(43)]
    model = NS(layers=layers, start_layer=0, end_layer=43, config=NS(hidden_size=4), hc_mult=2, device="cpu")
    config = NS(
        additional_config={
            "dspark_profile_observation": {"mode": "target-boundaries", "target_layer": target_layer},
            "dspark_confidence_verification": {"profile": True, "mode": "specified_lengths"},
        },
        parallel_config=NS(pipeline_parallel_size=1),
        speculative_config=NS(draft_model_config=NS(hf_config=NS(dspark_target_layer_ids=[40, 41, 42]))),
        compilation_config=NS(cudagraph_capture_sizes=[6, 12, 24, 48, 96, 192, 384]),
    )
    module.install_target_boundaries(model, config)
    bank = model._dspark_layer_snapshots
    assert bank.target_layer == target_layer
    assert all((layer._dspark_layer_snapshots is bank) == (i in bank.layers) for i, layer in enumerate(layers))
    config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="PP1"):
        module.install_target_boundaries(model, config)


def test_capture_integer_inputs_are_owned_and_do_not_claim_kv_coverage(tmp_path, monkeypatch):
    f = target_fixture(tmp_path, monkeypatch)
    f.run(88, proposal=False)
    observed_positions = f.positions.tolist()
    f.positions.fill_(999)
    f.observer.drain()
    r = f.observer.auxiliary_records[-1]
    fields = r["target_internal"]["metadata_fields"]
    assert r["device_integers"][fields["target.attn.input_positions"]] == observed_positions
    assert not any("slot_mapping" in k or "block_table" in k for k in fields)
    f.observer.close()


@pytest.mark.parametrize("changed", [None, "config_sha256", "index_sha256", "checkpoint_weight_sha256", "manifest"])
def test_preflight_matches_only_archived_provenance_fields(tmp_path, monkeypatch, changed):
    import hashlib

    from tools.dspark import check_target_profile_inputs as preflight

    manifest = tmp_path / "manifest.json"
    manifest.write_text("frozen input")
    checkpoint = {name: name for name in preflight.CHECKPOINT_FIELDS}
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {"checkpoint": checkpoint, "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        )
    )
    monkeypatch.setattr(preflight, "AUDIT_PATH", audit)
    if changed == "manifest":
        manifest.write_text("changed")
    elif changed is not None:
        checkpoint[changed] = "changed"
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    if changed is None:
        assert preflight.check(path, manifest)["full_target_weight_hashes"] == "UNAVAILABLE_IN_PRIOR_ARCHIVE"
    else:
        with pytest.raises(ValueError, match="changed"):
            preflight.check(path, manifest)


def test_actual_owner_selection_after_nan_keeps_first_error_and_counts_new_execution(tmp_path, monkeypatch):
    f = target_fixture(tmp_path, monkeypatch)
    f.fault[0].fill_(torch.nan)
    with pytest.raises(RuntimeError, match="original Markov NaN"):
        f.run(61)
    path = tmp_path / "rank-0-first-failure.json"
    first = path.read_bytes()
    select = _production_method(
        ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/verification_runtime.py",
        "ConfidenceVerification",
        "select",
        {"time": time},
    )

    def before_execute(scheduler):
        scheduler.scheduled_spec_decode_tokens = {"first": [1, 2, 3, 4, 5]}
        # Replay the actual selector's missing-owner branch. This reproduces
        # the downstream check, not an independent owner-publication defect.
        select(NS(rows={}), f.runner, scheduler)

    f.target.before_execute = before_execute
    with pytest.raises(ValueError, match="Scheduled candidates lack current proposal owners"):
        f.run(62)
    events = json.loads((tmp_path / "rank-0-error-events.json").read_text())["events"]
    assert [(x["execution"], x["stage"]) for x in events] == [(1, "markov"), (2, "target_execute")]
    assert f.graph.calls == 1  # owner failure occurs before a second target replay
    assert path.read_bytes() == first
    f.observer.close()
