# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only source adapter: real frozen queue code, dependency imports isolated.

Not an installed-vLLM/NPU claim. Server preflight uses normal installed imports.
The queue, serializer, response wrapper and collective RPC bodies are unmodified.
"""

import ast
import importlib.util
import logging
import sys
import types
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from enum import Enum, auto
from functools import partial
from pathlib import Path
from uuid import uuid4


def load(core):
    def module(name, **attrs):
        value = types.ModuleType(name)
        value.__dict__.update(attrs)
        value.__path__ = []
        sys.modules[name] = value
        return value

    for name in (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.device_communicators",
        "vllm.utils",
        "vllm.v1",
        "vllm.v1.executor",
    ):
        module(name)
    module("vllm.envs", VLLM_USE_SPINLOOP_EXT=False, VLLM_RINGBUFFER_WARNING_INTERVAL=60)
    module("vllm.distributed.utils", StatelessProcessGroup=object, sched_yield=lambda: __import__("time").sleep(0))
    module("vllm.logger", init_logger=logging.getLogger)
    module("vllm.platforms", current_platform=None)
    module(
        "vllm.utils.network_utils",
        get_ip=lambda: "127.0.0.1",
        get_open_port=lambda: 0,
        get_open_zmq_inproc_path=lambda: "inproc://" + uuid4().hex,
        get_open_zmq_ipc_path=lambda: "ipc:///tmp/dspark-mq-" + uuid4().hex,
        is_valid_ipv6_address=lambda _: False,
    )
    path = Path(core) / "vllm/distributed/device_communicators/shm_broadcast.py"
    spec = importlib.util.spec_from_file_location("vllm.distributed.device_communicators.shm_broadcast", path)
    queue = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = queue
    spec.loader.exec_module(queue)
    path = Path(core) / "vllm/v1/executor/multiproc_executor.py"
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "FutureWrapper":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in ("WorkerProc", "MultiprocExecutor"):
            names = {"ResponseStatus", "enqueue_output"} if node.name == "WorkerProc" else {"collective_rpc"}
            node.bases = []
            node.decorator_list = []
            node.body = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
            nodes.append(node)
    executor = module(
        "vllm.v1.executor.multiproc_executor",
        Future=Future,
        InvalidStateError=InvalidStateError,
        suppress=suppress,
        deque=deque,
        Enum=Enum,
        auto=auto,
        partial=partial,
        time=__import__("time"),
        AsyncModelRunnerOutput=type("UnusedAsyncOutput", (), {}),
        logger=logging.getLogger("source-adapter"),
    )
    exec(
        compile(
            ast.Module(body=[*ast.parse("from __future__ import annotations").body, *nodes], type_ignores=[]),
            str(path),
            "exec",
        ),
        executor.__dict__,
    )
    return queue, executor
