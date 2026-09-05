# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Frozen-core RPC bodies, real Msgpack, imported extension and CPU workers.

Only process transport/model initialization are replaced. No permissive fake
collective_rpc, serializer, extension injection or worker dispatch is used.
Run with --noconftest on Mac; this does not execute an NPU graph.
"""

import ast
import enum
import importlib
import importlib.util
import inspect
import os
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.dspark import benchmark_dspark_acceptance as benchmark

msgspec = pytest.importorskip("msgspec")
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
ROOT = Path(__file__).parents[2]


def _core():
    spec = importlib.util.find_spec("vllm")
    return Path(next(iter(spec.submodule_search_locations))) if spec else ROOT.parent / "vllm-hust/vllm"


def _load(path, namespace, *, whole=(), methods=None, functions=()):
    """Execute complete source bodies; retain real signatures and dispatch branches."""
    selected = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef):
            if node.name in whole:
                selected.append(node)
            elif methods and node.name in methods:
                node.bases = []
                node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods[node.name]]
                assert {n.name for n in node.body} == set(methods[node.name])
                for method in node.body:
                    method.decorator_list = [
                        d for d in method.decorator_list if isinstance(d, ast.Name) and d.id == "staticmethod"
                    ]
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    assert {n.name for n in selected} == set(whole) | set(methods or {}) | set(functions)
    module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *selected], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def _boundary(monkeypatch):
    core = _core()
    module = ModuleType("p08_core_rpc_boundary")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = vars(module)
    # Evaluate the frozen core's real environment reader for unset and 0 cases.
    key = "VLLM_ALLOW_INSECURE_SERIALIZATION"
    reader = next(
        value
        for node in ast.walk(ast.parse((core / "envs.py").read_text()))
        if isinstance(node, ast.Dict)
        for name, value in zip(node.keys, node.values)
        if isinstance(name, ast.Constant) and name.value == key
    )
    read_insecure = eval(compile(ast.Expression(reader), "core-env", "eval"), {"os": os})
    assert read_insecure() is False

    def forbidden_pickle(*args, **kwargs):
        raise AssertionError("Telemetry must not use pickle fallback")

    ns.update(
        torch=torch,
        np=np,
        msgspec=msgspec,
        msgpack=msgspec.msgpack,
        envs=SimpleNamespace(VLLM_ALLOW_INSECURE_SERIALIZATION=read_insecure(), VLLM_MSGPACK_ZERO_COPY_THRESHOLD=256),
        MultiModalKwargsItem=type("MMItem", (), {}),
        MultiModalKwargsItems=type("MMItems", (), {}),
        PIN_MEMORY=False,
        bytestr=bytes | bytearray | memoryview,
        isclass=inspect.isclass,
        signature=inspect.signature,
        Future=Future,
        InvalidStateError=InvalidStateError,
        suppress=suppress,
        partial=partial,
        time=time,
        uuid=uuid,
        enum=enum,
        traceback=traceback,
        importlib=importlib,
        cloudpickle=SimpleNamespace(dumps=forbidden_pickle, loads=forbidden_pickle),
        logger=SimpleNamespace(info=lambda *a: None, warning_once=lambda *a: None, exception=lambda *a: None),
        set_current_vllm_config=lambda config: nullcontext(),
    )
    _load(core / "v1/serial_utils.py", ns, whole=("MsgpackEncoder", "MsgpackDecoder", "UtilityResult"))
    _load(core / "v1/engine/__init__.py", ns, whole=("EngineCoreRequestType", "UtilityOutput"))
    _load(core / "utils/import_utils.py", ns, functions=("resolve_obj_by_qualname",))
    _load(core / "v1/worker/worker_base.py", ns, methods={"WorkerWrapperBase": ("init_worker", "__getattr__")})
    _load(
        core / "v1/executor/multiproc_executor.py",
        ns,
        whole=("FutureWrapper",),
        methods={"MultiprocExecutor": ("collective_rpc",), "WorkerProc": ("worker_busy_loop",)},
    )
    ns["WorkerProc"].ResponseStatus = SimpleNamespace(SUCCESS=0, FAILURE=1)
    _load(
        core / "v1/engine/core.py",
        ns,
        methods={
            "EngineCore": ("collective_rpc",),
            "EngineCoreProc": ("_invoke_utility_method", "_convert_msgspec_args"),
        },
    )
    _load(
        core / "v1/engine/core_client.py",
        ns,
        methods={"SyncMPClient": ("collective_rpc", "call_utility", "_send_input")},
    )
    _load(core / "v1/engine/llm_engine.py", ns, methods={"LLMEngine": ("collective_rpc",)})
    _load(core / "entrypoints/llm.py", ns, methods={"LLM": ("collective_rpc",)})
    return SimpleNamespace(**ns)


@dataclass(frozen=True)
class _Descriptor:
    num_tokens: int


class _Mode(enum.Enum):
    NONE = 0
    FULL_DECODE_ONLY = 1


def _workers(api, monkeypatch, extension, count=8):
    # The plugin package and extension module are imported normally. Stub only
    # its unrelated logging bootstrap on hosts without installed vLLM.
    if "vllm" not in sys.modules:
        parent = ModuleType("vllm")
        parent.__path__ = [str(_core())]
        monkeypatch.setitem(sys.modules, "vllm", parent)
    plugins = ModuleType("vllm.plugins")
    plugins.load_general_plugins = lambda: None
    monkeypatch.setitem(sys.modules, "vllm.plugins", plugins)
    if "vllm_ascend" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vllm_ascend.logger", ModuleType("vllm_ascend.logger"))

    class Base:
        pass

    class Worker(Base):
        def __init__(self, vllm_config, rank):
            self.rank = np.int64(rank)
            self.model_runner = SimpleNamespace(
                compilation_config=SimpleNamespace(
                    cudagraph_mode=_Mode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[np.int64(6)]
                ),
                cudagraph_manager=SimpleNamespace(graphs={_Descriptor(np.int64(6)): object()}),
                ascend_config=SimpleNamespace(
                    ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=True, enable_static_kernel=False)
                ),
                speculator=SimpleNamespace(requested_cudagraph_mode=_Mode.FULL_DECODE_ONLY, cudagraph_mode=_Mode.NONE),
            )

    worker_module = ModuleType("p08_cpu_worker")
    worker_module.Worker = Worker
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(worker_cls="p08_cpu_worker.Worker", worker_extension_cls=extension),
        model_config=SimpleNamespace(multimodal_config=None),
        enable_trace_function_call_for_thread=lambda: None,
    )
    kwargs = [{"vllm_config": config, "rank": rank} for rank in range(count)]
    workers = []
    for rank in range(count):
        wrapper = api.WorkerWrapperBase()
        wrapper.rpc_rank = rank
        wrapper.init_worker(kwargs)
        workers.append(wrapper)
    if extension:
        cls = api.resolve_obj_by_qualname(extension)
        assert cls.__module__ == "vllm_ascend.diagnostics.dspark_benchmark_worker"
        assert Path(sys.modules[cls.__module__].__file__).resolve() == (
            ROOT / "vllm_ascend/diagnostics/dspark_benchmark_worker.py"
        )
        assert cls in Worker.__bases__  # Actual core injection, not a fake registration.
    return workers


def _executor(api, workers):
    responses = [deque() for worker in workers]

    def enqueue(message):
        assert type(message[0]) is str  # Executor's callable/cloudpickle branch must never run.
        for rank, worker in enumerate(workers):
            proc = api.WorkerProc()
            proc.worker, proc.rank = worker, rank
            messages = iter([message])
            proc.rpc_broadcast_mq = SimpleNamespace(dequeue=lambda messages=messages, **kw: next(messages))
            proc.handle_output = lambda value, rank=rank: responses[rank].append(
                (1 if isinstance(value, Exception) else 0, value)
            )
            # End the finite queue outside the actual dispatch try block.
            with suppress(StopIteration):
                proc.worker_busy_loop()  # Real getattr dispatch inside the worker loop.

    executor = api.MultiprocExecutor()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=enqueue)
    executor.is_failed = False
    executor.futures_queue = deque()
    executor.response_mqs = [SimpleNamespace(dequeue=lambda timeout, q=q: q.popleft()) for q in responses]
    return executor


def _serve(payload, extension=None):
    with pytest.MonkeyPatch.context() as patch:
        api = _boundary(patch)
        if extension is None:
            args = benchmark.parse_args(
                [
                    "--model-dir",
                    str(ROOT),
                    "--mode",
                    "dspark",
                    "--dataset-name",
                    "jsonl",
                    "--dataset-path",
                    "unused.jsonl",
                    "--result-json",
                    "unused.json",
                    "--target-execution-mode",
                    "full_decode_only",
                    "--cudagraph-capture-sizes",
                    "6",
                ]
            )
            extension = benchmark.build_engine_kwargs(args)["worker_extension_cls"]
        workers = _workers(api, patch, extension)
        engine = api.EngineCore()
        engine.model_executor = _executor(api, workers)
        client_index, call_id, utility_method, args = api.MsgpackDecoder().decode(payload)
        assert client_index == 0 and utility_method == "collective_rpc"
        method = getattr(engine, utility_method)
        output = api.UtilityOutput(call_id)
        results = []
        api.EngineCoreProc._invoke_utility_method(
            utility_method,
            lambda: method(*api.EngineCoreProc._convert_msgspec_args(method, args)),
            output,
            results.append,
        )
        assert len(results) == 1
        encoded = api.MsgpackEncoder().encode(output)
        assert len(encoded) == 1
        return encoded[0]


def _frontend(api, *, subprocess_worker=False, extension=None):
    client = api.SyncMPClient()
    client.utility_results = {}
    client.encoder = api.MsgpackEncoder()
    client.core_engine = b"cpu-engine"
    client.ensure_alive = client.free_pending_messages = lambda: None
    sent = []

    def send_multipart(message, *, copy):
        assert copy is False and len(message) == 3 and message[1] == api.EngineCoreRequestType.UTILITY.value
        sent.append(message)
        if subprocess_worker:
            command = (
                "import sys; from tests.ut.test_dspark_graph_rpc import _serve; "
                "sys.stdout.buffer.write(_serve(sys.stdin.buffer.read()))"
            )
            result = subprocess.run(
                [sys.executable, "-c", command], cwd=ROOT, input=message[2], capture_output=True, timeout=60
            )
            assert result.returncode == 0, result.stderr.decode()
            payload = result.stdout
        else:
            payload = _serve(message[2], extension)
        output = api.MsgpackDecoder(api.UtilityOutput).decode(payload)
        future = client.utility_results.pop(output.call_id)
        if output.failure_message is not None:
            future.set_exception(RuntimeError(output.failure_message))
        else:
            future.set_result(output.result.result)

    client.input_socket = SimpleNamespace(send_multipart=send_multipart)
    engine = api.LLMEngine()
    engine.engine_core = client
    frontend = api.LLM()
    frontend.llm_engine = engine
    return frontend, sent


def _assert_basic(value):
    assert type(value) in (str, int, bool, type(None), list, dict)
    if type(value) is dict:
        for key, child in value.items():
            assert type(key) is str
            _assert_basic(child)
    elif type(value) is list:
        for child in value:
            _assert_basic(child)


@pytest.mark.parametrize("insecure", [None, "0"])
def test_frontend_rpc_roundtrip_imports_and_registers_extension_in_fresh_process(monkeypatch, insecure):
    if insecure is None:
        monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    else:
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", insecure)
    api = _boundary(monkeypatch)
    frontend, sent = _frontend(api, subprocess_worker=True)
    args = SimpleNamespace(tensor_parallel_size=8, mode="dspark", cudagraph_capture_sizes=[6])
    result = benchmark._collect_worker_graph_runtime(frontend, args)
    _assert_basic(result)
    assert len(sent) == 1
    assert result["observed_capture_sizes"] == result["configured_capture_sizes"] == [6]
    assert result["graph_capture_count"] == 1
    assert [worker["rank"] for worker in result["workers"]] == list(range(8))
    assert all(
        worker["target_cudagraph_mode"] == "FULL_DECODE_ONLY" and worker["dspark_cudagraph_mode"] == "NONE"
        for worker in result["workers"]
    )
    # Reproduce the original error at actual frontend encoding, before send.
    with pytest.raises(TypeError, match="function.*not serializable"):
        frontend.collective_rpc(lambda worker: {})
    assert len(sent) == 1


def test_string_method_without_actual_extension_registration_fails(monkeypatch):
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    api = _boundary(monkeypatch)
    frontend, _ = _frontend(api, extension="")
    with pytest.raises(RuntimeError, match="has no attribute.*dspark_benchmark_graph_runtime"):
        frontend.collective_rpc(benchmark._GRAPH_SNAPSHOT_METHOD)


@pytest.mark.parametrize(
    "field,value", [("rank", True), ("rank", None), ("npugraph_ex_enabled", torch.tensor(True)), ("capture_size", -1)]
)
def test_extension_rejects_nonprimitive_or_invalid_runtime_values(monkeypatch, field, value):
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    api = _boundary(monkeypatch)
    worker = _workers(api, monkeypatch, benchmark._GRAPH_WORKER_EXTENSION, count=1)[0]
    if field == "rank":
        worker.worker.rank = value
    elif field == "capture_size":
        worker.model_runner.cudagraph_manager.graphs = {_Descriptor(value): object()}
    else:
        worker.model_runner.ascend_config.ascend_compilation_config.enable_npugraph_ex = value
    with pytest.raises(RuntimeError):
        getattr(worker, benchmark._GRAPH_SNAPSHOT_METHOD)()


@pytest.mark.parametrize(
    "fault", ["duplicate_rank", "rank_target_mode", "rank_draft_mode", "rank_capture", "wrong_sizes"]
)
def test_real_rank_snapshots_must_agree_and_cover_requested_graphs(monkeypatch, fault):
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)
    api = _boundary(monkeypatch)
    workers = _workers(api, monkeypatch, benchmark._GRAPH_WORKER_EXTENSION)
    last = workers[-1].worker
    if fault == "duplicate_rank":
        last.rank = 0
    elif fault == "rank_target_mode":
        last.model_runner.compilation_config.cudagraph_mode = _Mode.NONE
    elif fault == "rank_draft_mode":
        last.model_runner.speculator.cudagraph_mode = _Mode.FULL_DECODE_ONLY
    elif fault == "rank_capture":
        last.model_runner.cudagraph_manager.graphs.clear()
    else:
        for worker in workers:
            worker.model_runner.compilation_config.cudagraph_capture_sizes = [12]
            worker.model_runner.cudagraph_manager.graphs = {_Descriptor(12): object()}
    args = SimpleNamespace(tensor_parallel_size=8, mode="dspark", cudagraph_capture_sizes=[6])
    with pytest.raises(RuntimeError, match="rank|capture sizes"):
        benchmark._collect_worker_graph_runtime(_executor(api, workers), args)
