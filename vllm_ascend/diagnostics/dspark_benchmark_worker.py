# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Worker extension used only by the public-LLM DSpark benchmark.

Keep this module importable by its installed package path in spawned workers.
No tensors, graph objects, callables or enum objects cross the telemetry RPC.
"""

from numbers import Integral
from typing import Any
from uuid import uuid4


def _cudagraph_mode_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _host_count(value: Any) -> int:
    # Never convert a tensor: telemetry must not synchronize or copy from NPU.
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Replay token counts must be nonnegative host integers.")
    return int(value)


class _FullReplayObserver:
    """Benchmark-local wrappers; delegate each call once, preserve its return/exception.

    Installed after model capture. A successful run_fullgraph is committed only
    after its enclosing real execute_model returns successfully. Shapes come
    from that call's completed InputBatch, not the scheduler's graph statistics.
    Histograms bound storage by shape count rather than generation length.
    """

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.manager = runner.cudagraph_manager
        self.original_execute = runner.execute_model
        self.original_fullgraph = self.manager.run_fullgraph
        self.observer_id = str(uuid4())
        self.pending: list[Any] | None = None
        self.shapes: dict[tuple[int, int], int] = {}
        self.excluded_dummy_replay_count = 0
        self.excluded_unscoped_replay_count = 0
        self.failed_execution_count = 0
        self.error: str | None = None
        self.nan_diagnostic = None
        additional = getattr(getattr(runner, "vllm_config", None), "additional_config", None) or {}
        directory = additional.get("dspark_nan_diagnostic_dir")
        if directory is not None:
            # Explicit benchmark-only instrumentation, installed after capture.
            from vllm_ascend.diagnostics.dspark_nan import DSparkNaNDiagnostics

            speculator = runner.speculator
            self.nan_diagnostic = DSparkNaNDiagnostics(directory, speculator.rank)
            speculator._nan_diagnostic = self.nan_diagnostic
        runner.execute_model = self.execute_model
        self.manager.run_fullgraph = self.run_fullgraph

    def execute_model(
        self,
        scheduler_output: Any,
        intermediate_tensors: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ) -> Any:
        if self.pending is not None:
            self.error = "Nested execute_model is unsupported by replay telemetry."
        pending: list[Any] = []
        previous = self.pending
        self.pending = pending
        diagnostic = self.nan_diagnostic if not (dummy_run or is_profile) else None
        try:
            if diagnostic is not None:
                diagnostic.begin_execution(scheduler_output)
            result = self.original_execute(
                scheduler_output,
                intermediate_tensors=intermediate_tensors,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                is_profile=is_profile,
            )
            if diagnostic is not None:
                diagnostic.target_completed(self.runner, pending)
        except BaseException as error:
            self.failed_execution_count += 1
            if diagnostic is not None:
                diagnostic.failed_execution("target_execute", error)
            raise
        else:
            if dummy_run or is_profile:
                self.excluded_dummy_replay_count += len(pending)
            elif pending:
                try:
                    batch = self.runner.execute_model_state.input_batch
                    unpadded = _host_count(batch.num_tokens)
                    padded = _host_count(batch.num_tokens_after_padding)
                    if (
                        len(pending) != 1
                        or _cudagraph_mode_name(pending[0].cg_mode) != "FULL"
                        or _host_count(pending[0].num_tokens) != padded
                        or not 0 < unpadded <= padded
                    ):
                        raise ValueError("FULL replay descriptor disagrees with the completed input batch.")
                    shape = (unpadded, padded)
                    self.shapes[shape] = self.shapes.get(shape, 0) + 1
                except (AttributeError, TypeError, ValueError) as error:
                    # Retain output/timing even if the evidence ABI is unavailable.
                    self.error = str(error)
            return result
        finally:
            self.pending = previous

    def run_fullgraph(self, desc: Any) -> Any:
        result = self.original_fullgraph(desc)
        if self.pending is None:
            self.excluded_unscoped_replay_count += 1
        else:
            self.pending.append(desc)
        return result

    def snapshot(self) -> dict[str, Any]:
        if self.runner.execute_model != self.execute_model or self.manager.run_fullgraph != self.run_fullgraph:
            self.error = "Replay observer was replaced after installation."
        if self.pending is not None:
            self.error = "Replay snapshot requested inside execute_model."
        return {
            "observer_id": self.observer_id,
            "source": "mrv2_successful_execute_model_full_replay",
            "error": self.error,
            "records": [
                {
                    "runtime_mode": "FULL",
                    "num_unpadded_tokens": unpadded,
                    "num_padded_tokens": padded,
                    "num_paddings": padded - unpadded,
                    "count": count,
                }
                for (unpadded, padded), count in sorted(self.shapes.items())
            ],
            "excluded_dummy_replay_count": self.excluded_dummy_replay_count,
            "excluded_unscoped_replay_count": self.excluded_unscoped_replay_count,
            "failed_execution_count": self.failed_execution_count,
            "eager_fallback_count": None,
            "eager_fallback_evidence": "unavailable: observer covers FULL replay only",
        }


class DSparkBenchmarkWorkerExtension:
    """Named benchmark telemetry RPC, installed through worker_extension_cls."""

    def dspark_benchmark_replay_snapshot(self, diagnostic_phase: str | None = None) -> dict[str, Any]:
        """Install once at a quiescent boundary, then read cumulative host counters."""
        rank = _host_count(self.rank)
        runner = self.model_runner
        observer = getattr(runner, "_dspark_benchmark_replay_observer", None)
        if observer is None:
            observer = _FullReplayObserver(runner)
            runner._dspark_benchmark_replay_observer = observer
        if diagnostic_phase is not None:
            if diagnostic_phase not in ("warmup", "measured", "complete") or observer.nan_diagnostic is None:
                raise ValueError("Diagnostic phase requires an installed DSpark NaN observer and a valid phase.")
            observer.nan_diagnostic.phase = diagnostic_phase
        return {"rank": rank, **observer.snapshot()}

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
