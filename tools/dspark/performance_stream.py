# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Benchmark-only AsyncLLM facade; no changes to engine execution or sampling.

The private event loop stays alive across warmup, phase RPCs and measurement.
Only an independent benchmark process owns this facade. DELTA events are
observed at the frontend iterator, including any output-queue coalescing.
"""

from __future__ import annotations

import asyncio
import copy
import time
from types import SimpleNamespace

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark.profile_attention_validity import AttentionValidity
from tools.dspark.profile_failure import CANCEL_TIMEOUT_SECONDS, RPC_TIMEOUT_SECONDS, ProfileFailureGuard
from tools.dspark.profile_request_ids import RequestIdObserver


def request_latency(record):
    first = record["first_output_monotonic"]
    finish = record["completed_monotonic"]
    submit = record["submitted_monotonic"]
    count = len(record["output_token_ids"])
    if finish is None or finish < submit or (first is not None and not submit <= first <= finish):
        raise ValueError("Invalid request timing boundaries")
    events = record["events"]
    if sum(event["new_tokens"] for event in events) != count:
        raise ValueError("Stream event token counts do not match output")
    return {
        "ttft_seconds": first - submit if first is not None else None,
        "completion_seconds": finish - submit,
        "mean_tpot_seconds": (finish - first) / (count - 1) if count > 1 and first is not None else None,
        "event_intervals_seconds": [b["monotonic"] - a["monotonic"] for a, b in zip(events, events[1:])],
    }


async def stream_batch(
    engine,
    prompts,
    sampling,
    outstanding,
    batch_id,
    *,
    clock=time.monotonic,
    progress=None,
    bounded_cancel=False,
    validate_progress=None,
):
    """Closed-loop admission in source order, or all-at-once when limit is None."""
    if outstanding is not None and outstanding <= 0:
        raise ValueError("client outstanding must be positive or None")
    semaphore = asyncio.Semaphore(outstanding or len(prompts))
    started = clock()
    records = [None] * len(prompts)
    result = progress if progress is not None else {}
    result.update(started_monotonic=started, requests=records, error=None)

    async def consume(index, prompt):
        queued = clock()
        async with semaphore:
            record = {
                "request_id": f"{batch_id}-{index}",
                "request_index": index,
                "client_ready_monotonic": queued,
                "submitted_monotonic": clock(),
                "first_output_monotonic": None,
                "completed_monotonic": None,
                "events": [],
                "output_token_ids": [],
                "text_parts": [],
                "finish_reason": None,
                "stop_reason": None,
                "error": None,
                "observed_prompt_token_ids": None,
            }
            records[index] = record
            try:
                async for output in engine.generate(prompt, copy.copy(sampling), record["request_id"]):
                    observed = clock()
                    observed_prompt = getattr(output, "prompt_token_ids", None)
                    if observed_prompt is not None:
                        if list(observed_prompt) != prompt["prompt_token_ids"]:
                            raise RuntimeError("Engine changed the frozen prompt token IDs")
                        record["observed_prompt_token_ids"] = list(observed_prompt)
                    if len(output.outputs) != 1 or output.outputs[0].index != 0:
                        raise RuntimeError("Streaming benchmark requires one completion")
                    completion = output.outputs[0]
                    tokens = list(completion.token_ids)
                    if any(type(token) is not int or token < 0 for token in tokens):
                        raise RuntimeError("Corrupt output token IDs")
                    if tokens:
                        if record["first_output_monotonic"] is None:
                            record["first_output_monotonic"] = observed
                        record["events"].append({"monotonic": observed, "new_tokens": len(tokens)})
                    record["output_token_ids"].extend(tokens)
                    record["text_parts"].append(completion.text)
                    if validate_progress is not None:
                        validate_progress(sum(len(r["output_token_ids"]) for r in records if r is not None))
                    if output.finished:
                        record["completed_monotonic"] = observed
                        record["finish_reason"] = completion.finish_reason
                        record["stop_reason"] = completion.stop_reason
                if record["completed_monotonic"] is None or record["finish_reason"] not in ("stop", "length"):
                    raise RuntimeError("Stream ended without a normal completion")
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                raise

    tasks = [asyncio.create_task(consume(i, prompt)) for i, prompt in enumerate(prompts)]
    error = None
    try:
        await asyncio.gather(*tasks)
    except (Exception, asyncio.CancelledError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        for task in tasks:
            task.cancel()
        if bounded_cancel:
            _, pending = await asyncio.wait(tasks, timeout=CANCEL_TIMEOUT_SECONDS)
            result["pending_cancelled_requests"] = len(pending)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(exc, asyncio.CancelledError):
                raise
    finished = clock()
    # Text joins and latency calculations occur after the measured boundary.
    for record in records:
        if record is not None:
            record["text"] = "".join(record.pop("text_parts"))
            if record["completed_monotonic"] is not None:
                record.update(request_latency(record))
    result.update(finished_monotonic=finished, elapsed_seconds=finished - started, error=error)
    return result


class SchedulerCollector:
    """CPU fields already delivered by AsyncLLM's StatLoggerManager.

    SchedulerStats is produced once per EngineCore step, not once per TP rank.
    The frozen output_handler calls record before completing a chunk's waiters.
    """

    def __init__(self, k):
        self.k = k
        self.totals = [0, 0, 0]
        self.positions = [0] * k
        self.preemptions = 0
        self.forwards = 0
        self.committed = 0
        self.corrupted = 0
        self.rows = []

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if engine_idx != 0:
            raise ValueError("This TP benchmark requires a single DP EngineCore")
        if iteration_stats is not None:
            self.preemptions += iteration_stats.num_preempted_reqs
            self.corrupted += iteration_stats.num_corrupted_reqs
        if scheduler_stats is not None:
            self.rows.append(
                {
                    "num_running_reqs": scheduler_stats.num_running_reqs,
                    "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
                    "kv_cache_usage": scheduler_stats.kv_cache_usage,
                }
            )
            spec = scheduler_stats.spec_decoding_stats
            if spec is not None:
                if spec.num_spec_tokens != self.k or len(spec.num_accepted_tokens_per_pos) != self.k:
                    raise ValueError("Invalid speculative metrics vector")
                for i, value in enumerate((spec.num_drafts, spec.num_draft_tokens, spec.num_accepted_tokens)):
                    self.totals[i] += int(value)
                self.positions = [a + int(b) for a, b in zip(self.positions, spec.num_accepted_tokens_per_pos)]
                self.forwards += spec.num_forwards
                self.committed += spec.num_committed_tokens

    def metrics(self):
        result = [
            SimpleNamespace(name=name, labels={"engine": "0"}, value=value)
            for name, value in zip(benchmark.SPEC_METRIC_NAMES[:3], self.totals)
        ]
        result.append(
            SimpleNamespace(name=benchmark.VECTOR_METRIC_NAME, labels={"engine": "0"}, values=list(self.positions))
        )
        return result

    def log_engine_initialized(self):
        pass

    def log(self):
        pass

    def record_sleep_state(self, is_awake=1, level=0):
        pass


class StreamingEngine:
    def __init__(self, kwargs, args):
        # Deferred imports keep data preparation, statistics and CPU tests independent of NPU.
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.sampling_params import RequestOutputKind
        from vllm.v1.engine.async_llm import AsyncLLM

        self.loop = asyncio.new_event_loop()
        self.collector = SchedulerCollector(args.num_spec_tokens)
        self.args = args
        self.batch_number = 0
        self.last_batch = None
        self.delta_kind = RequestOutputKind.DELTA
        self.profile_guard = None
        self.cleanup_result = None

        def logger_factory(vllm_config, engine_index=0):
            if engine_index != 0:
                raise ValueError("Only DP=1 is supported")
            return self.collector

        async def initialize():
            return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs), stat_loggers=[logger_factory])

        try:
            self.engine = self.loop.run_until_complete(initialize())
        except BaseException:
            self.loop.close()
            raise
        failure_directory = (kwargs.get("additional_config") or {}).get("dspark_profile_failure_dir")
        if failure_directory is not None:
            self.profile_guard = ProfileFailureGuard(self.engine, failure_directory, require_worker_receipt=True)
        observation = (kwargs.get("additional_config") or {}).get("dspark_profile_observation", {})
        if observation.get("attention"):
            if self.profile_guard is None:
                raise ValueError("Attention validation requires the bounded profile guard")
            self.profile_guard.attention_validity = AttentionValidity(
                observation["directory"], kwargs["tensor_parallel_size"]
            )
        # Adapt only the read-only config/utility interface used by the existing benchmark.
        self.llm_engine = SimpleNamespace(
            vllm_config=self.engine.vllm_config, engine_core=SimpleNamespace(call_utility=self.call_utility)
        )

    def _run(self, operation, awaitable, timeout=None):
        if self.profile_guard is None:
            return self.loop.run_until_complete(awaitable)
        try:
            return self.loop.run_until_complete(self.profile_guard.run(operation, awaitable, timeout))
        except BaseException as error:
            self.profile_guard.remember(error)
            raise

    def call_utility(self, method):
        return self._run(method, self.engine.engine_core.call_utility_async(method), RPC_TIMEOUT_SECONDS)

    def collective_rpc(self, method, kwargs=None):
        if not isinstance(method, str):
            raise TypeError("Only named RPC methods are permitted")
        options = {}
        if self.profile_guard is not None:
            options["timeout"] = RPC_TIMEOUT_SECONDS
            if method == "dspark_benchmark_profile_point":
                self.profile_guard.point = (kwargs or {}).get("point")
        return self._run(
            "collective_rpc:" + method,
            self.engine.collective_rpc(method, kwargs=kwargs, **options),
            RPC_TIMEOUT_SECONDS,
        )

    def get_tokenizer(self):
        return self.engine.get_tokenizer()

    def get_metrics(self):
        return self.collector.metrics()

    def generate(self, prompts, sampling_params, use_tqdm=False, *, profile_point=None):
        sampling = copy.copy(sampling_params)
        sampling.output_kind = self.delta_kind
        self.batch_number += 1
        before = (
            len(self.collector.rows),
            self.collector.preemptions,
            self.collector.forwards,
            self.collector.committed,
            self.collector.corrupted,
            self.collector.totals[0],
        )
        batch_id = f"batch{self.batch_number}"
        if profile_point is None:
            self.last_batch = self.loop.run_until_complete(
                stream_batch(self.engine, prompts, sampling, self.args.client_outstanding, batch_id)
            )
        else:
            observer = RequestIdObserver(
                self.engine, profile_point, {f"{batch_id}-{i}": i for i in range(len(prompts))}
            )
            self.last_batch = {}
            gate = getattr(self.profile_guard, "attention_validity", None)
            validate = (
                (lambda tokens: gate.check(profile_point, tokens=tokens))
                if gate is not None and not gate.passed
                else None
            )
            try:
                with observer:
                    self.last_batch = self._run(
                        "generate",
                        stream_batch(
                            self.engine,
                            prompts,
                            sampling,
                            self.args.client_outstanding,
                            batch_id,
                            progress=self.last_batch,
                            bounded_cancel=self.profile_guard is not None,
                            validate_progress=validate,
                        ),
                    )
            finally:
                self.last_batch["request_id_mapping"] = observer.receipt
        self.last_batch["scheduler"] = {
            "observations": self.collector.rows[before[0] :],
            "preemptions": self.collector.preemptions - before[1],
            "proposal_publication_steps": self.collector.forwards - before[2],
            "committed_tokens_on_proposal_publication_steps": self.collector.committed - before[3],
            "request_verifications": self.collector.totals[0] - before[5],
            "verification_batch_count": None,
            "spec_count_note": (
                "Frozen MRV2 num_forwards counts successful proposal publication, not all target forwards"
            ),
            "corrupted_requests": self.collector.corrupted - before[4],
            "corrupted_request_note": "Only errors reported by the core; internal NaN instrumentation is not enabled",
            "recomputed_tokens": None,
            "peak_device_memory_bytes": None,
            "source": "AsyncLLM StatLogger SchedulerStats/IterationStats; missing fields unavailable",
        }
        if self.last_batch["error"]:
            raise RuntimeError(self.last_batch["error"])
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(
                        token_ids=record["output_token_ids"],
                        text=record["text"],
                        finish_reason=record["finish_reason"],
                        stop_reason=record["stop_reason"],
                    )
                ],
            )
            for prompt, record in zip(prompts, self.last_batch["requests"])
        ]

    def shutdown(self):
        async def close():
            if self.profile_guard is None:
                self.engine.shutdown()
            else:
                self.cleanup_result = await self.profile_guard.shutdown()
                if not self.cleanup_result["thread_completed"]:
                    return  # the supervisor owns the stuck thread/process bound
                # Frozen Core schedules socket/task cleanup with
                # call_soon_threadsafe immediately before shutdown returns.
                await asyncio.sleep(0)
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            if pending:
                if self.profile_guard is None:
                    await asyncio.gather(*pending, return_exceptions=True)
                else:
                    _, remaining = await asyncio.wait(pending, timeout=CANCEL_TIMEOUT_SECONDS)
                    self.cleanup_result["pending_loop_tasks"] = len(remaining)
                    if remaining:
                        self.cleanup_result["success"] = False
                        self.cleanup_result["loop_error"] = "Task cancellation exceeded the loop drain budget"
                        self.profile_guard.remember(RuntimeError(self.cleanup_result["loop_error"]))
            if self.profile_guard is not None:
                await asyncio.sleep(0)  # run completion callbacks before closing the loop

        try:
            self.loop.run_until_complete(close())
        finally:
            if self.profile_guard is None:
                self.loop.close()
            elif self.cleanup_result is not None:
                safe = self.cleanup_result["thread_completed"] and not self.cleanup_result.get("pending_loop_tasks")
                if safe:
                    self.loop.close()
                self.cleanup_result["event_loop"] = "closed" if safe else "retained_for_supervisor"
                self.profile_guard.save_cleanup(self.cleanup_result)
