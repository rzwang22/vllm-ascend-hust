# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Worker extension used only by the public-LLM DSpark benchmark.

Keep this module importable by its installed package path in spawned workers.
No tensors, graph objects, callables or enum objects cross the telemetry RPC.
"""

from numbers import Integral
from typing import Any


def _cudagraph_mode_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.rsplit(".", 1)[-1]


class DSparkBenchmarkWorkerExtension:
    """Named telemetry RPC; no worker initialization or compute overrides."""

    def dspark_benchmark_graph_runtime(self) -> dict[str, Any]:
        """Return JSON-safe target/draft graph state from one real worker."""
        rank = getattr(self, "rank", None)
        if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 0:
            raise RuntimeError("Worker does not expose a valid rank for graph evidence.")
        runner = getattr(self, "model_runner", None)
        if runner is None:
            raise RuntimeError("Worker does not expose its MRV2 model runner.")
        compilation_config = getattr(runner, "compilation_config", None)
        graph_manager = getattr(runner, "cudagraph_manager", None)
        graphs = getattr(graph_manager, "graphs", None)
        if not isinstance(graphs, dict):
            raise RuntimeError("MRV2 model runner does not expose captured graph descriptors.")
        descriptors = list(graphs)
        capture_sizes = [getattr(descriptor, "num_tokens", None) for descriptor in descriptors]
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in capture_sizes):
            raise RuntimeError("MRV2 model runner exposes invalid captured graph descriptors.")
        observed_capture_sizes = sorted({int(value) for value in capture_sizes})
        configured_capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
        if not isinstance(configured_capture_sizes, list) or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in configured_capture_sizes
        ):
            raise RuntimeError("MRV2 model runner exposes invalid configured capture sizes.")
        ascend_config = getattr(runner, "ascend_config", None)
        ascend_compilation = getattr(ascend_config, "ascend_compilation_config", None)
        speculator = getattr(runner, "speculator", None)
        flags = {
            "npugraph_ex_enabled": getattr(ascend_compilation, "enable_npugraph_ex", None),
            "static_kernel_enabled": getattr(ascend_compilation, "enable_static_kernel", None),
        }
        if any(value is not None and type(value) is not bool for value in flags.values()):
            raise RuntimeError("MRV2 graph flags must be bool or None for telemetry serialization.")
        return {
            "rank": int(rank),
            "target_cudagraph_mode": _cudagraph_mode_name(getattr(compilation_config, "cudagraph_mode", None)),
            "configured_capture_sizes": [int(value) for value in configured_capture_sizes],
            "observed_capture_sizes": observed_capture_sizes,
            "graph_capture_count": len(descriptors),
            **flags,
            "dspark_requested_cudagraph_mode": _cudagraph_mode_name(
                getattr(speculator, "requested_cudagraph_mode", None)
            ),
            "dspark_cudagraph_mode": _cudagraph_mode_name(getattr(speculator, "cudagraph_mode", None)),
        }
