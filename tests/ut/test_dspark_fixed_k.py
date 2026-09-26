# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CPU tensor recurrence + source methods; full installed/NPU path is separate."""

import ast
import importlib.util
import runpy
import sys
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace as NS

import pytest
import torch

from tests.ut.test_dspark_performance_comparison import args, observation, runner_fixture  # noqa: F401
from tools.dspark import fixed_k_comparison as fixed
from tools.dspark import performance_comparison as task
from tools.dspark.performance_stream import SchedulerCollector

ROOT = Path(__file__).parents[2]
validate_draft_length = runpy.run_path(str(ROOT / "vllm_ascend/spec_decode/dspark_fixed_k.py"))["validate_draft_length"]
SPECULATOR = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"


@pytest.mark.parametrize(
    "k,options,valid",
    [
        (5, {}, True),
        (8, {}, False),
        (8, {"dspark_fixed_k8_experiment": True}, True),
        (8, {"dspark_fixed_k8_experiment": True, "dspark_confidence_verification": {"mode": "confidence"}}, False),
        (7, {"dspark_fixed_k8_experiment": True}, False),
        (True, {}, False),
    ],
)
def test_length_opt_in(k, options, valid):
    if valid:
        validate_draft_length(k, options)
    else:
        with pytest.raises(ValueError):
            validate_draft_length(k, options)


@pytest.mark.parametrize("k", [5, 8])
def test_real_config_builder(tmp_path, monkeypatch, k):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0")
    monkeypatch.setenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", "0")
    a = args(tmp_path, 256)
    a.fixed_k_comparison, a.draft_k = True, k
    parsed, kw = task.case_config(a, tmp_path)
    assert parsed.num_spec_tokens == k
    assert kw["speculative_config"]["num_speculative_tokens"] == k
    assert kw["compilation_config"]["cudagraph_capture_sizes"] == fixed.captures(k)
    assert kw["max_num_batched_tokens"] >= 256 * (k + 1)
    assert "dspark_confidence_verification" not in kw["additional_config"]
    assert kw["additional_config"]["dspark_fixed_k8_experiment"] == (k == 8)
    assert not parsed.ignore_eos and parsed.output_len == 256


def test_two_cases_and_full_capacity_gate():
    p = fixed.plan(task.plan())
    assert [c["draft_k"] for c in p["cases"]] == [5, 8]
    assert all(c["batch"] == 256 and c["mode"] == "fixed" for c in p["cases"])
    assert p["model_initializations"] == 2
    assert p["stage_sum_upper_seconds"] < p["total_limit_seconds"] == 10000
    assert sum(c["max_requests_total"] for c in p["cases"]) == 2048
    good = {
        "FULL": {
            "layouts": [dict(requests=256, query_tokens=2304, capacity=2304, count=1)],
            "published_proposals": dict(calls=1, requests=256, candidates=2048, first_epoch=5, last_epoch=5),
        }
    }
    fixed.validate_full(good, 8)
    for key, value in (("requests", 64), ("query_tokens", 1536), ("capacity", 1536), ("count", 0)):
        row = {**good["FULL"]["layouts"][0], key: value}
        with pytest.raises(ValueError):
            fixed.validate_full({"FULL": {**good["FULL"], "layouts": [row]}}, 8)


def source_runtime():
    """Compile unchanged real methods, avoiding import of unavailable Ascend/vLLM.

    Uses actual Torch math and actual Markov input/epoch validation; fake tiny
    weight heads stand in for the checkpoint. Does not emulate NPU attention.
    """
    path = SPECULATOR.with_name("proposal_inputs.py")
    spec = importlib.util.spec_from_file_location("k_test_proposal", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tree = ast.parse(SPECULATOR.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendDSparkSpeculator")
    names = {
        "_execute_sequential_markov_sampling",
        "_validate_markov_inputs",
        "_validate_step_tensor",
        "_require_greedy_markov_sampling",
        "_build_query_slot_mappings",
    }
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    predicate = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_assert_markov_tensor_contract"
    )
    selected = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            predicate,
            ast.ClassDef(name="Runtime", bases=[], keywords=[], body=methods, decorator_list=[]),
        ],
        type_ignores=[],
    )
    ns = {**vars(module), "validate_draft_length": validate_draft_length, "MappingProxyType": MappingProxyType}
    exec(compile(ast.fix_missing_locations(selected), str(SPECULATOR), "exec"), ns)
    return ns["Runtime"]


@pytest.mark.parametrize("k", [5, 8])
@pytest.mark.parametrize("nan_last", [False, True])
def test_real_recurrence_and_epoch_no_padding(k, nan_last):
    runtime = source_runtime()()
    runtime.device = torch.device("cpu")
    runtime.vllm_config = NS(additional_config={"dspark_fixed_k8_experiment": k == 8})
    runtime.rank = 0
    runtime._proposal_step_epoch = runtime._context_kv_step_epoch = runtime._draft_forward_step_epoch = 17
    runtime._markov_attempt_step_epoch = None
    runtime.confidence_verification = None
    runtime.draft_model_config = NS(hf_config=NS(vocab_size=32, hidden_size=4))
    contract = dict(
        lm_head_id=1,
        markov_head_id=2,
        confidence_head_id=3,
        lm_head_class="tiny",
        markov_head_class="tiny",
        lm_head_parameter_names=("head",),
        markov_parameter_names=("bias",),
        confidence_head=object(),
    )
    runtime._markov_module_contract = contract
    runtime._inspect_markov_modules = lambda m: contract
    calls = []

    def bias(x):
        calls.append(x.clone())
        values = torch.zeros(2, 32)
        values.scatter_(1, ((x.long() + 1) % 32), 10)
        if nan_last and len(calls) == k:
            values[0, 0] = float("nan")
        return values

    runtime.model = NS(
        compute_draft_logits=lambda h: torch.zeros(len(h), 32),
        markov_embed=lambda t: t[:, None].float(),
        markov_bias=bias,
        map_draft_to_target=lambda t: t,
    )
    proposal = NS(
        step_epoch=17,
        rank=0,
        request_ids=("r1", "r0"),
        num_reqs=2,
        num_speculative_tokens=k,
        num_query_tokens=2 * k,
        draft_query_start_loc=torch.arange(3, dtype=torch.int32) * k,
        anchor_token_ids=torch.tensor([2, 10]),
        request_state_indices=torch.tensor([63, 12], dtype=torch.int32),
        temperature=torch.zeros(64),
    )
    hidden = torch.zeros(2 * k, 4)
    if nan_last:
        with pytest.raises(ValueError, match="NaN"):
            runtime._execute_sequential_markov_sampling(proposal, hidden)
        assert runtime._markov_result is None and runtime._markov_step_epoch is None
    else:
        result = runtime._execute_sequential_markov_sampling(proposal, hidden)
        assert result.candidate_tokens.tolist() == [list(range(3, 3 + k)), list(range(11, 11 + k))]
        assert result.num_speculative_tokens == len(result.steps) == len(calls) == k
        assert result.request_ids == ("r1", "r0") and result.step_epoch == 17
        assert result.confidence_head_used is False
    with pytest.raises(RuntimeError, match="already attempted"):
        runtime._execute_sequential_markov_sampling(proposal, hidden)


def test_k8_completed_replay_records_actual_nine_rows(observation):  # noqa: F811 -- imported pytest fixture
    runner = runner_fixture()
    runner.speculator.num_speculative_steps = 8
    o = observation.PerformanceReplayObserver(runner)
    runner.execute_model(([9] * 256, 2304))
    assert o.snapshot()["performance"]["layouts"] == [dict(requests=256, query_tokens=2304, capacity=2304, count=1)]
    o.reset()
    runner.execute_model(([10], 18))
    assert o.snapshot()["error"]
    o.close()


def test_core_metrics_distinguish_verified_and_delivered():
    c = SchedulerCollector(8, record_verifications=True)
    spec = NS(
        num_spec_tokens=8,
        num_accepted_tokens_per_pos=[2, 1, 0, 0, 0, 0, 0, 0],
        num_drafts=2,
        num_draft_tokens=16,
        num_accepted_tokens=3,
        num_forwards=1,
        num_committed_tokens=5,
    )
    c.record(
        NS(num_running_reqs=2, num_waiting_reqs=0, kv_cache_usage=0.1, spec_decoding_stats=spec),
        NS(num_preempted_reqs=0, num_corrupted_reqs=0, num_generation_tokens=4),
    )
    assert c.verification_steps[0]["sampler_progress_before_eos"] == 5
    assert c.verification_steps[0]["frontend_output_tokens"] == 4
    assert c.totals == [2, 16, 3]


def test_real_k8_slot_mapping_crosses_page_boundary_and_preserves_tail():
    r = source_runtime()()
    r.device = torch.device("cpu")
    r.num_speculative_steps = 8
    r.draft_attn_layer_order = ("a", "b")
    r.draft_layer_group_ids = {"a": 1, "b": 1}
    table = torch.tensor([[8, 11], [3, 7]], dtype=torch.int32)
    r.block_tables = NS(
        slot_mappings=torch.full((2, 32), -1, dtype=torch.int32),
        kernel_block_sizes=[32, 32],
        input_block_tables=[table, table],
    )
    positions = torch.tensor(list(range(29, 37)) + list(range(31, 39)))
    groups, tables, slots = r._build_query_slot_mappings(positions, 2)
    expected = (
        [8 * 32 + 29, 8 * 32 + 30, 8 * 32 + 31]
        + list(range(11 * 32, 11 * 32 + 5))
        + [3 * 32 + 31]
        + list(range(7 * 32, 7 * 32 + 7))
    )
    assert slots["a"].tolist() == expected
    assert slots["a"].data_ptr() == slots["b"].data_ptr()
    assert r.block_tables.slot_mappings[1, 16:].tolist() == [-1] * 16
    assert groups == {"a": 1, "b": 1} and tables[1] is table


def test_candidate_receipt_and_wrapper_release(observation):  # noqa: F811
    r = runner_fixture()
    r.speculator.num_speculative_steps = 8
    r.vllm_config.additional_config["dspark_fixed_k_comparison"] = True
    tensor = torch.arange(16).reshape(2, 8)
    original = lambda p, v: tensor
    r.speculator._build_core_proposal = original
    observer = observation.PerformanceReplayObserver(r)
    for epoch in (5, 6, 7):
        assert r.speculator._build_core_proposal(None, NS(num_reqs=2, step_epoch=epoch)) is tensor
    receipt = observer.snapshot()["performance"]["published_proposals"]
    assert receipt == dict(calls=3, requests=6, candidates=48, first_epoch=5, last_epoch=7)
    observer.reset()
    assert receipt["candidates"] == 48
    assert observer.snapshot()["performance"]["published_proposals"]["candidates"] == 0
    observer.close()
    assert r.speculator._build_core_proposal is original
    assert observer.original_publish is None and observer.runner is None


@pytest.mark.parametrize("k", [5, 8])
def test_acceptance_uses_runtime_candidate_count(k):
    totals = {
        "vllm:spec_decode_num_drafts": 2,
        "vllm:spec_decode_num_draft_tokens": 2 * k,
        "vllm:spec_decode_num_accepted_tokens": 2 * k,
        task.benchmark.VECTOR_METRIC_NAME: [2] * k,
    }
    task.acceptance_metrics({"totals": totals}, "fixed", k)
    with pytest.raises(ValueError):
        task.acceptance_metrics({"totals": totals}, "fixed", k + 1)


def test_passive_metrics_default_does_not_keep_step_history():
    c = SchedulerCollector(5)
    spec = NS(
        num_spec_tokens=5,
        num_accepted_tokens_per_pos=[1, 0, 0, 0, 0],
        num_drafts=1,
        num_draft_tokens=5,
        num_accepted_tokens=1,
        num_forwards=1,
        num_committed_tokens=2,
    )
    c.record(NS(num_running_reqs=1, num_waiting_reqs=0, kv_cache_usage=0.1, spec_decoding_stats=spec), None)
    assert c.totals == [1, 5, 1] and c.verification_steps == []


def test_k8_runtime_comparison_only_changes_declared_dimensions():
    frozen = dict(K=5, capture_sizes=fixed.captures(5), dtype="torch.bfloat16", tp=8, model="frozen")
    expected = fixed.runtime_expected(frozen, 8)
    assert expected == {**frozen, "K": 8, "capture_sizes": fixed.captures(8)}
    assert frozen["K"] == 5 and frozen["capture_sizes"] == fixed.captures(5)


def test_fixed_k_driver_stops_after_first_failed_case(tmp_path, monkeypatch):
    a = args(tmp_path, 256)
    a.fixed_k_comparison, a.draft_k = True, 5
    seen = []

    def logged(command, log):
        if log.name == "host.log":
            (tmp_path / "host.xml").write_text(
                '<testsuites><testsuite><testcase name="test"/></testsuite></testsuites>'
            )
        return 0

    def fail_case(a, case):
        seen.append(case["name"])
        return dict(case=case, valid=False, error="fixture failure")

    monkeypatch.setattr(task.suite, "logged", logged)
    monkeypatch.setattr(task, "supervise", fail_case)
    assert task.run(a) == 1
    assert seen == ["b256-fixed-k5"]
    assert task.read(tmp_path / "performance-summary.json")["all_two_valid"] is False


@pytest.mark.parametrize("bad", ["capacity", "buffers", "rank"])
def test_k8_actual_capacity_rejects_config_only_claim(bad):
    rows = [
        dict(
            rank=i,
            max_requests=256,
            draft_max_requests=256,
            max_tokens=8192,
            draft_max_tokens=8192,
            draft_capacity_source=dict(
                kind="allocated_shared_block_tables",
                shared_with_target=True,
                groups=[dict(stored_shape=[256, 256], input_shape=[256, 256])],
                slot_mapping_shape=[3, 8192],
            ),
            kv_num_blocks=100,
            kv_bytes=1048576,
            groups=["real"],
            capture_sizes=fixed.captures(8),
        )
        for i in range(8)
    ]
    fixed.capacity_check(rows, 8)
    if bad == "capacity":
        rows[7]["capture_sizes"] = fixed.captures(5)
    elif bad == "buffers":
        rows[7]["max_tokens"] = 1536
    else:
        rows.pop()
    with pytest.raises(ValueError):
        fixed.capacity_check(rows, 8)


def test_summary_keeps_outputs_progress_and_memory_cost_separate(tmp_path):
    reports = []
    for k, rate in ((5, 100), (8, 110)):
        name = f"b256-fixed-k{k}"
        rows = [
            dict(
                output_tokens_per_second=rate,
                progress=dict(
                    sampler_progress_per_request_verification=k, sampler_progress_per_verification_batch=k * 2
                ),
                memory=dict(
                    after=[dict(rank=i, peak_allocated_bytes=k * 100, peak_reserved_bytes=k * 200) for i in range(8)]
                ),
            )
            for _ in range(3)
        ]
        reports.append(dict(valid=True, case=dict(name=name), generation=dict(rounds=rows)))
        for i in range(1, 4):
            task.write(
                tmp_path / "runs" / name / f"round-{i}-stream.json",
                dict(requests=[dict(request_id="r", output_token_ids=[k])]),
            )
    result = fixed.summarize(tmp_path, reports, task.distribution)
    assert result["all_two_valid"]
    comparison = result["comparison"]
    assert comparison["speedup"] == 1.1 and comparison["warning"] == "OUTPUTS_DIFFER"
    assert comparison["quality_equivalence"] == "NOT_EVALUATED"
    assert comparison["allocator_peak_delta_k8_minus_k5_bytes"] == [
        dict(rank=i, peak_allocated_bytes=300, peak_reserved_bytes=600) for i in range(8)
    ]
