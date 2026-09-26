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
from datetime import datetime, timezone
from types import SimpleNamespace

from tools.dspark import benchmark_dspark_acceptance as benchmark
from tools.dspark.profile_attention_validity import AttentionValidity
from tools.dspark.profile_failure import CANCEL_TIMEOUT_SECONDS, RPC_TIMEOUT_SECONDS, ProfileFailureGuard, write_json
from tools.dspark.profile_request_ids import RequestIdObserver

MAX_VERIFICATION_OBSERVATIONS = 65536


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
    request_ids=None,
):
    """Closed-loop admission in source order, or all-at-once when limit is None."""
    if outstanding is not None and outstanding <= 0:
        raise ValueError("client outstanding must be positive or None")
    if request_ids is not None and (len(request_ids) != len(prompts) or len(set(request_ids)) != len(prompts)):
        raise ValueError("Explicit request IDs must uniquely match prompts")
    semaphore = asyncio.Semaphore(outstanding or len(prompts))
    started = clock()
    records = [None] * len(prompts)
    result = progress if progress is not None else {}
    result.update(started_monotonic=started, requests=records, error=None)

    async def consume(index, prompt):
        queued = clock()
        async with semaphore:
            record = {
                "request_id": request_ids[index] if request_ids is not None else f"{batch_id}-{index}",
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

    def __init__(self, k, record_verifications=False):
        self.k = k
        self.record_verifications = record_verifications
        self.totals = [0, 0, 0]
        self.positions = [0] * k
        self.preemptions = 0
        self.forwards = 0
        self.committed = 0
        self.corrupted = 0
        self.rows = []
        self.verification_steps = []

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
                if self.record_verifications and spec.num_drafts:
                    if len(self.verification_steps) >= MAX_VERIFICATION_OBSERVATIONS:
                        raise ValueError("Verification step telemetry capacity exceeded")
                    self.verification_steps.append(
                        {
                            "requests": int(spec.num_drafts),
                            "candidates": int(spec.num_draft_tokens),
                            "accepted": int(spec.num_accepted_tokens),
                            "sampler_progress_before_eos": int(spec.num_drafts + spec.num_accepted_tokens),
                            "frontend_output_tokens": getattr(iteration_stats, "num_generation_tokens", None),
                            "scope": "one Core stats delivery; frontend output may also include prefill requests",
                        }
                    )

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
        self.collector = SchedulerCollector(
            args.num_spec_tokens, kwargs.get("additional_config", {}).get("dspark_fixed_k_comparison") is True
        )
        self.args = args
        observation = kwargs.get("additional_config", {}).get("dspark_profile_observation", {})
        self.write_timeline_point = (
            (observation.get("operator_capture") or {}).get("point")
            if (observation.get("operator_capture") or {}).get("write_timeline")
            else None
        )
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
            self.profile_guard = ProfileFailureGuard(
                self.engine,
                failure_directory,
                require_worker_receipt=True,
                shutdown_policy=(kwargs.get("additional_config") or {}).get("dspark_profile_shutdown_policy"),
                exit_observation=(kwargs.get("additional_config") or {}).get("dspark_profile_exit_observation", False),
            )
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

    def generate(self, prompts, sampling_params, use_tqdm=False, *, profile_point=None, request_ids=None):
        if request_ids is not None and (len(request_ids) != len(prompts) or len(set(request_ids)) != len(prompts)):
            raise ValueError("Explicit request IDs must uniquely match prompts")
        sampling = copy.copy(sampling_params)
        sampling.output_kind = self.delta_kind
        if profile_point is not None and profile_point == getattr(self, "write_timeline_point", None):
            sampling.extra_args = {**(sampling.extra_args or {}), "dspark_write_point": profile_point}
        self.batch_number += 1
        before = (
            len(self.collector.rows),
            self.collector.preemptions,
            self.collector.forwards,
            self.collector.committed,
            self.collector.corrupted,
            self.collector.totals[0],
            len(self.collector.verification_steps),
        )
        batch_id = f"batch{self.batch_number}"
        if profile_point is None:
            self.last_batch = self.loop.run_until_complete(
                stream_batch(
                    self.engine, prompts, sampling, self.args.client_outstanding, batch_id, request_ids=request_ids
                )
            )
        else:
            observer = RequestIdObserver(
                self.engine,
                profile_point,
                {(request_ids[i] if request_ids is not None else f"{batch_id}-{i}"): i for i in range(len(prompts))},
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
                            request_ids=request_ids,
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
            "verification_batch_count": len(self.collector.verification_steps) - before[6]
            if self.collector.record_verifications
            else None,
            "verification_steps": self.collector.verification_steps[before[6] :],
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

    async def _stop_profile_output(self):
        """Stop the idle AsyncLLM consumer on its loop, before Core closes its producer.

        Unfinished or unavailable request state is an abort/error, never a
        successful drain. Leave that consumer for the original abort cleanup.
        Frozen AsyncLLM catches output errors internally, so a task that already
        returned (even without a raised exception) is not proof of healthy exit.
        """
        guard = self.profile_guard
        started = time.monotonic()
        result = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "performance_eligible": False,
            "timeout_seconds": CANCEL_TIMEOUT_SECONDS,
            "success": False,
            "status": "checking",
            "prior_error": guard.first,
            "error": None,
        }
        try:
            result["unfinished_requests"] = self.engine.output_processor.get_num_unfinished_requests()
            result["engine_errored_before_shutdown"] = self.engine.errored
            batch = getattr(self, "last_batch", None)
            result["last_batch_complete"] = batch is None or (
                not batch.get("error")
                and bool(batch.get("requests"))
                and all(
                    r is not None
                    and r.get("completed_monotonic") is not None
                    and not r.get("error")
                    and r.get("finish_reason") in ("stop", "length")
                    for r in batch["requests"]
                )
            )
            task = self.engine.output_handler
            if task is not None and task.done():
                if not task.cancelled():
                    task.result()  # collect a pre-existing task exception even if EngineCore is dead
                raise RuntimeError("Output handler stopped before the intentional drain")
            producer = self.engine.engine_core.resources.output_queue_task
            if producer is not None and producer.done():
                if not producer.cancelled():
                    producer.result()
                # The frozen socket task catches transport exceptions, queues
                # them for the consumer, then returns. Do not cancel that
                # consumer and hide an error it has not read yet.
                raise RuntimeError("Core output socket task stopped before the intentional drain")
            if result["unfinished_requests"] != 0 or not result["last_batch_complete"] or guard.pending is not None:
                raise RuntimeError("Requests/operation still pending at profile shutdown; refusing normal output drain")
            if result["engine_errored_before_shutdown"]:
                raise RuntimeError("AsyncLLM was already errored before profile output drain")
            if task is None:
                result["status"] = "not_started"
            else:
                if task.get_loop() is not asyncio.get_running_loop():
                    raise RuntimeError("Output handler belongs to a different event loop")
                result["status"] = "cancelling"
                task.cancel()
                done, _ = await asyncio.wait({task}, timeout=CANCEL_TIMEOUT_SECONDS)
                if not done:
                    raise TimeoutError("Output handler cancellation exceeded its drain budget")
                result["cancelled"] = task.cancelled()
                if not task.cancelled():
                    task.result()  # do not hide errors raised while cancelling
                if self.engine.engine_core.resources.engine_dead or (producer is not None and producer.done()):
                    if producer is not None and producer.done() and not producer.cancelled():
                        producer.result()
                    raise RuntimeError("Core output producer failed during the intentional drain")
                result["status"] = "drained"
            result["success"] = guard.first is None
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            guard.remember(error)
        finally:
            result["finished_utc"] = datetime.now(timezone.utc).isoformat()
            result["elapsed_seconds"] = time.monotonic() - started
            try:
                write_json(guard.directory / "output-handler-shutdown.json", result)
            except OSError as error:
                result.update(success=False, recording_error=f"{type(error).__name__}: {error}")
                guard.remember(error)
        return result

    def shutdown(self):
        async def close():
            if self.profile_guard is None:
                self.engine.shutdown()
            else:
                self.profile_guard.phase = "cleanup"
                frontend_started = (time.monotonic(), datetime.now(timezone.utc).isoformat())
                output_drain = await self._stop_profile_output()
                self.cleanup_result = await self.profile_guard.shutdown(frontend_started=frontend_started)
                self.cleanup_result["output_handler_shutdown"] = output_drain
                if not output_drain["success"]:
                    self.cleanup_result.update(success=False, frontend_error="Output handler drain failed; see receipt")
                    if self.cleanup_result["status"] == "returned":
                        self.cleanup_result["status"] = "output_handler_failed"
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
                    done, remaining = await asyncio.wait(pending, timeout=CANCEL_TIMEOUT_SECONDS)
                    for task in done:
                        if not task.cancelled():
                            try:
                                task.result()
                            except Exception as error:
                                self.cleanup_result.update(success=False, loop_error=f"{type(error).__name__}: {error}")
                                self.profile_guard.remember(error)
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
