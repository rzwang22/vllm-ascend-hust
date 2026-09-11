# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU execution of frozen Core admission bodies; no model/NPU initialization."""

import ast
import asyncio
import copy
import importlib.util
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast

import pytest

from tools.dspark.performance_stream import SchedulerCollector, StreamingEngine
from tools.dspark.profile_request_ids import RequestIdentityError, RequestIdObserver, validate_point_request_ids
from tools.dspark.startup_cost_profile import collect


def load_methods(path, namespace, name, methods):
    if not path.is_file():
        pytest.skip(f"Frozen Core source unavailable: {path}")
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    assert {n.name for n in cls.body} == set(methods)
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, cls], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def core_engine():
    spec = importlib.util.find_spec("vllm")
    core = Path(next(iter(spec.submodule_search_locations))) if spec else Path(__file__).parents[3] / "vllm-hust/vllm"
    suffixes = iter(["bea9330c01234567", "deadcafe01234567", *[f"{i:08x}" for i in range(100)]])
    ns = {
        "asyncio": asyncio,
        "AsyncGenerator": AsyncGenerator,
        "envs": NS(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),
        "random_uuid": lambda: next(suffixes),
        "PoolingParams": type("PoolingParams", (), {}),
        "EngineCoreRequest": type("EngineCoreRequest", (NS,), {}),
        "RequestOutputKind": NS(DELTA="delta"),
        "RequestOutput": NS,
        "PoolingOutput": type("PoolingOutput", (), {}),
        "CompletionOutput": NS,
        "cast": cast,
        "extract_prompt_components": lambda model, prompt: (None, prompt["prompt_token_ids"], None),
    }
    processor = load_methods(core / "v1/engine/input_processor.py", ns, "InputProcessor", ["assign_request_id"])()
    load_methods(core / "v1/engine/output_processor.py", ns, "RequestOutputCollector", ["__init__"])
    output_state = load_methods(core / "v1/engine/output_processor.py", ns, "RequestState", ["_new_request_output"])
    async_llm = load_methods(core / "v1/engine/async_llm.py", ns, "AsyncLLM", ["add_request", "_add_request"])

    class Engine(async_llm):
        def __init__(self):
            self.errored = False
            self.vllm_config = NS(cache_config=NS(kv_sharing_fast_prefill=False))
            self.model_config = None
            self.input_processor = processor
            self.input_processor.process_inputs = self.process_inputs
            self.log_requests = False
            self.admitted = []
            self.registered = []
            self.active = {}
            self.fail_enqueue = False
            self.output_processor = NS(add_request=self.register)
            self.engine_core = NS(add_request_async=self.enqueue)

        def process_inputs(self, request_id, prompt, params, **kwargs):
            return ns["EngineCoreRequest"](
                request_id=request_id, external_req_id=None, params=params, prompt_token_ids=prompt["prompt_token_ids"]
            )

        async def get_supported_tasks(self):
            return ("generate",)

        def _run_output_handler(self):
            pass

        def register(self, request, prompt, parent, index, collector):
            assert parent is None and index == 0
            assert collector.request_id == request.request_id
            self.registered.append(request)

        async def enqueue(self, request):
            if self.fail_enqueue:
                raise RuntimeError("enqueue failed")
            self.admitted.append(request)
            self.active[request.request_id] = request

        async def generate(self, prompt, params, request_id):
            collector = await self.add_request(request_id, prompt, params)
            request = self.active.pop(collector.request_id)
            state = output_state()
            state.prompt_token_ids = request.prompt_token_ids
            state.output_kind = "delta"
            state.logprobs_processor = NS(pop_prompt_logprobs=lambda: None)
            state.lora_request = state.prompt = state.stats = None
            state.num_cached_tokens = 0
            completion = NS(index=0, token_ids=[9, 10], text="ok", finish_reason="length", stop_reason=None)
            yield state._new_request_output(request.external_req_id, [completion], True)

    return Engine()


def facade(engine):
    result = StreamingEngine.__new__(StreamingEngine)
    result.engine = engine
    result.loop = asyncio.new_event_loop()
    result.collector = SchedulerCollector(5)
    result.args = NS(client_outstanding=None)
    result.delta_kind = "delta"
    result.batch_number = 0
    result.last_batch = None
    result.profile_guard = None
    return result


def events(point, ids):
    return [
        {
            "rank": rank,
            "cost_profile": {
                "measurements": [{"kind": "draft", "point": point, "request_ids": list(ids), "requests": len(ids)}]
            },
        }
        for rank in range(8)
    ]


def run_point(engine, point, count=1):
    engine.generate([{"prompt_token_ids": [12, 13]}] * count, NS(n=1), profile_point=point)
    return copy.deepcopy(engine.last_batch)


def test_frozen_assignment_enqueue_collector_and_public_output(core_engine):
    engine = facade(core_engine)
    try:
        stream = run_point(engine, "point1")
        receipt = stream["request_id_mapping"]
        assert receipt["mappings"] == [
            {"point": "point1", "request_index": 0, "external_id": "batch1-0", "internal_id": "batch1-0-bea9330c"}
        ]
        assert core_engine.admitted[0].request_id == core_engine.registered[0].request_id == "batch1-0-bea9330c"
        assert core_engine.admitted[0].external_req_id == stream["requests"][0]["request_id"] == "batch1-0"
        assert stream["requests"][0]["output_token_ids"] == [9, 10]
        assert receipt["hook_restored"] and "add_request" not in vars(core_engine)
        history = {}
        validate_point_request_ids("point1", stream, events("point1", ["batch1-0-bea9330c"]), history)
        second = run_point(engine, "point2")
        validate_point_request_ids("point2", second, events("point2", ["batch2-0-deadcafe"]), history)
        assert history == {"batch1-0-bea9330c": "point1", "batch2-0-deadcafe": "point2"}
        assert not core_engine.active
    finally:
        engine.loop.close()


def test_prior_point_and_similar_prefix_are_not_accepted(core_engine):
    engine = facade(core_engine)
    try:
        first = run_point(engine, "first")
        history = {}
        old = first["request_id_mapping"]["mappings"][0]["internal_id"]
        validate_point_request_ids("first", first, events("first", [old]), history)
        second = run_point(engine, "second")
        for bad, reason in [(old, "previous_point_event"), ("batch2-0-deadcafe-extra", "unknown_internal_id")]:
            with pytest.raises(RequestIdentityError) as error:
                validate_point_request_ids("second", second, events("second", [bad]), history)
            evidence = json.loads(json.dumps(error.value.evidence))
            assert len(evidence["event_failures"]) == 8
            assert evidence["event_failures"][0]["reason"] == reason
            assert evidence["event_failures"][0]["unknown_internal_ids"] == [bad]
            assert evidence["event_failures"][0]["event_kind"] == "draft"
            assert evidence["expected_internal_ids"] == ["batch2-0-deadcafe"]
    finally:
        engine.loop.close()


def test_external_reuse_hyphens_and_repeated_prompt_instances(core_engine):
    history = {}
    for point in ("first", "second"):
        expected = {"batch-1-0": 0, "batch-1-0-extra": 1}
        observer = RequestIdObserver(core_engine, point, expected)

        async def admit(observer=observer, expected=expected):
            with observer:
                for external in expected:
                    await core_engine.add_request(external, {"prompt_token_ids": [3, 3]}, NS(n=1, output_kind="delta"))

        asyncio.run(admit())
        stream = {
            "requests": [{"request_id": key, "request_index": index} for key, index in expected.items()],
            "request_id_mapping": observer.receipt,
        }
        internal = [row["internal_id"] for row in observer.receipt["mappings"]]
        validate_point_request_ids(point, stream, events(point, internal), history)
        assert observer.engine is observer.point is observer.original is None
        assert not observer.mappings and not observer.expected
    assert len(history) == 4  # external reuse is legal, internal reuse is not


@pytest.mark.parametrize(
    "problem", ["missing", "absent_receipt", "conflict", "wrong_point", "wrong_index", "wrong_event_point", "empty_ids"]
)
def test_invalid_receipts_keep_readable_evidence(core_engine, problem):
    engine = facade(core_engine)
    try:
        stream = run_point(engine, "point", 2)
        receipt = stream["request_id_mapping"]
        worker = events("point", [row["internal_id"] for row in receipt["mappings"]])
        if problem == "missing":
            receipt["mappings"].pop()
        elif problem == "absent_receipt":
            del stream["request_id_mapping"]
        elif problem == "empty_ids":
            worker[-1]["cost_profile"]["measurements"][0]["request_ids"] = []
        elif problem == "conflict":
            receipt["mappings"][1]["internal_id"] = receipt["mappings"][0]["internal_id"]
        elif problem == "wrong_point":
            receipt["point"] = "old"
        elif problem == "wrong_index":
            receipt["mappings"][0]["request_index"] = 3
        else:
            worker[0]["cost_profile"]["measurements"][0]["point"] = "old"
        with pytest.raises(RequestIdentityError) as error:
            validate_point_request_ids("point", stream, worker, {})
        evidence = json.loads(json.dumps(error.value.evidence))
        assert evidence["expected_external_ids"] == ["batch1-0", "batch1-1"]
        assert evidence["observed_events"][0]["actual_internal_ids"] == ["batch1-0-bea9330c", "batch1-1-deadcafe"]
        assert evidence["point"] == "point"
    finally:
        engine.loop.close()


@pytest.mark.parametrize("failure", ["enqueue", "cancel", "conflict", "body", "missing_collector"])
def test_observer_restores_existing_instance_override_on_failure(core_engine, failure):
    original = core_engine.add_request

    async def override(request_id, prompt, params):
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "missing_collector":
            return NS()
        return await original(request_id, prompt, params)

    core_engine.add_request = override
    core_engine.fail_enqueue = failure == "enqueue"
    observer = RequestIdObserver(core_engine, "point", {"external-with-hyphens": 0})

    async def admit():
        with observer:
            for _ in range(2 if failure == "conflict" else 1):
                await core_engine.add_request(
                    "external-with-hyphens", {"prompt_token_ids": [1]}, NS(n=1, output_kind="delta")
                )
            raise RuntimeError("body failed")

    with pytest.raises((RuntimeError, ValueError, asyncio.CancelledError)):
        asyncio.run(admit())
    assert core_engine.add_request is override
    assert observer.receipt["hook_restored"]
    assert observer.point is observer.engine is observer.original is None
    assert not observer.mappings and not observer.expected
    if failure == "enqueue":
        assert not observer.receipt["mappings"] and not core_engine.admitted
    if failure == "conflict":
        assert observer.receipt["errors"][0]["reason"] == "mapping_conflict"


def test_failed_admission_persists_point_artifacts_and_closes_engine(core_engine, tmp_path):
    engine = facade(core_engine)
    core_engine.fail_enqueue = True
    closed = []
    engine.get_tokenizer = lambda: NS(encode=lambda *args, **kwargs: [3])
    engine.collective_rpc = lambda *args, **kwargs: events("point", ["unknown-internal"])
    engine.shutdown = lambda: (closed.append(True), engine.loop.close())
    point = {"id": "point", "lengths": [5], "requests": 1, "prompt_tokens": 128}
    with pytest.raises(RuntimeError, match="enqueue failed"):
        collect(lambda: engine, [point], NS(n=1), tmp_path, warmup=2, samples=5)
    failure = json.loads((tmp_path / "profile-failure.json").read_text())
    assert failure["request_identity_failure"]["reason"] == "mapping_missing"
    assert failure["request_identity_failure"]["observed_events"][0]["actual_internal_ids"] == ["unknown-internal"]
    assert json.loads((tmp_path / "point.json").read_text())["streaming"]["request_id_mapping"]["hook_restored"]
    assert closed == [True] and "add_request" not in vars(core_engine)


@pytest.mark.parametrize("problem", [None, "prior_point", "mapping_missing", "mapping_conflict"])
def test_collect_actual_admission_receipts_and_failure_artifacts(core_engine, tmp_path, problem):
    from tests.ut.test_dspark_startup_cost_profile import snapshots
    from tools.dspark.startup_cost_profile import grid

    engine = facade(core_engine)
    engine.get_tokenizer = lambda: NS(encode=lambda *args, **kwargs: [3])
    points = grid(1, [6], [128], 64)[1]
    current = None
    first_id = None
    closed = []

    def rpc(method, kwargs=None):
        nonlocal current, first_id
        if method == "dspark_benchmark_profile_point":
            current = next(p for p in points if p["id"] == kwargs["point"])
            return []
        if current is None:
            return []
        ids = [row["internal_id"] for row in engine.last_batch["request_id_mapping"]["mappings"]]
        if first_id is None:
            first_id = ids[0]
        elif problem == "prior_point":
            ids = [first_id]
        elif problem == "mapping_missing":
            engine.last_batch["request_id_mapping"]["mappings"] = []
        elif problem == "mapping_conflict":
            mapping = engine.last_batch["request_id_mapping"]["mappings"]
            mapping.append(dict(mapping[0]))
        result = snapshots(current)
        for rank in result:
            for event in rank["cost_profile"]["measurements"]:
                event["request_ids"] = ids
        return result

    engine.collective_rpc = rpc
    engine.shutdown = lambda: (closed.append(True), engine.loop.close())
    if problem is None:
        retained, _ = collect(lambda: engine, points, NS(n=1), tmp_path, warmup=2, samples=5, ranks=2)
        assert len(retained) == 2
        assert not (tmp_path / "profile-failure.json").exists()
    else:
        with pytest.raises(RequestIdentityError):
            collect(lambda: engine, points, NS(n=1), tmp_path, warmup=2, samples=5, ranks=2)
        failure = json.loads((tmp_path / "profile-failure.json").read_text())["request_identity_failure"]
        if problem == "prior_point":
            assert failure["event_failures"][0]["previous_points"] == {first_id: points[0]["id"]}
        else:
            assert failure["reason"] == problem
        assert failure["observed_events"] and failure["expected_internal_ids"] is not None
        raw = json.loads((tmp_path / f"{points[1]['id']}.json").read_text())
        assert raw["request_identity_failure"] == failure
        assert raw["ranks"][0]["cost_profile"]["measurements"][0]["request_ids"]
    assert len(core_engine.admitted) == 2 and not core_engine.active
    assert closed == [True] and "add_request" not in vars(core_engine)
    assert json.loads((tmp_path / "lifecycle.json").read_text()) == {
        "engine_initialization_attempts": 1,
        "engine_initializations": 1,
        "shutdown": True,
    }
