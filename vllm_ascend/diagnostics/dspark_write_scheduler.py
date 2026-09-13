# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic AsyncScheduler subclass; forwards all Core allocation/accounting unchanged."""

import inspect
import json
import time
import weakref
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler

MAX_EVENTS = 30000
MAX_BLOCKS = 4096


def request_state(request):
    return {
        k: getattr(request, k, None)
        for k in (
            "request_id",
            "num_computed_tokens",
            "num_output_placeholders",
            "num_tokens",
            "num_tokens_with_spec",
            "num_prompt_tokens",
            "async_tokens_to_discard",
        )
    }


def blocks(values):
    values = list(values)
    if len(values) > MAX_BLOCKS:
        raise ValueError("Page diagnostic block bound exceeded")
    return [
        {"block_id": b.block_id, "ref_cnt": b.ref_cnt, "is_null": b.is_null, "hash": repr(b.block_hash)} for b in values
    ]


class PageTrace:
    def __init__(self, scheduler, directory, point):
        self.scheduler = weakref.ref(scheduler)
        self.path = Path(directory) / "scheduler-page-timeline.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.point = point
        self.sequence = 0
        self.step = 0
        self.active = False
        self.context = []
        self.truncated = False

    def record(self, event, **values):
        if not self.active:
            return
        if self.sequence >= MAX_EVENTS:
            if not self.truncated:
                self.truncated = True
                with self.path.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "event": "TRUNCATED",
                                "sequence": self.sequence,
                                "schedule_id": self.step,
                                "active_calls": self.context,
                                "coverage": "UNAVAILABLE",
                            }
                        )
                        + "\n"
                    )
            raise ValueError("Page timeline event budget exhausted; coverage unavailable")
        self.sequence += 1
        with self.path.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "sequence": self.sequence,
                        "point": self.point,
                        "schedule_id": self.step,
                        "event": event,
                        "monotonic_ns": time.monotonic_ns(),
                        "context": list(self.context),
                        **values,
                    }
                )
                + "\n"
            )

    def wrap(self, obj, name, label, group=None):
        original = getattr(obj, name)
        ref = weakref.WeakMethod(original)
        sig = inspect.signature(original)
        owner = weakref.ref(obj)

        def observed(*args, **kwargs):
            if not self.active:
                return ref()(*args, **kwargs)
            params = sig.bind(*args, **kwargs).arguments
            seen, freed_before = [], []
            if name == "free_blocks":
                ordered = params["ordered_blocks"]

                def observing_iterator():
                    for block in ordered:
                        seen.append(block)
                        freed_before.extend(blocks([block]))
                        yield block

                params["ordered_blocks"] = observing_iterator()
                args, kwargs = (), dict(params)
            clean = {}
            for k, v in params.items():
                if k == "ordered_blocks":
                    clean[k] = {"scope": "recorded as Core consumes iterator"}
                elif hasattr(v, "request_id"):
                    clean[k] = request_state(v)
                elif type(v) in (str, int, bool, float) or v is None:
                    clean[k] = v
                elif k == "new_computed_blocks":
                    clean[k] = repr(v)
                else:
                    clean[k] = {"type": type(v).__name__, "scope": "not serialized"}
            req = params.get("request_id") or params.get("request")
            rid = req if isinstance(req, str) else getattr(req, "request_id", None)
            mapping = getattr(owner(), "req_to_blocks", {})
            before = blocks(mapping.get(rid, ())) if rid else None
            self.context.append({"call": label, "group": group, "request_id": rid})
            self.record("call.begin", call=label, group=group, arguments=clean, request_blocks=before)
            try:
                result = ref()(*args, **kwargs)
                self.record(
                    "call.return",
                    call=label,
                    group=group,
                    request_blocks=blocks(mapping.get(rid, ())) if rid else None,
                    result_blocks=blocks(result) if name == "get_new_blocks" else None,
                    freed_before=freed_before if name == "free_blocks" else None,
                    freed_blocks=blocks(seen) if name == "free_blocks" else None,
                    allocated_groups=[blocks(v) for v in result.blocks]
                    if name == "allocate_slots" and result is not None
                    else None,
                )
                return result
            except BaseException as error:
                try:
                    self.record("call.error", call=label, error=repr(error))
                except Exception as diagnostic_error:
                    error.add_note(f"Page trace error (original preserved): {diagnostic_error!r}")
                raise
            finally:
                self.context.pop()

        setattr(obj, name, observed)

    def install(self):
        scheduler = self.scheduler()
        manager = scheduler.kv_cache_manager
        self.wrap(manager, "allocate_slots", "allocate_slots")
        self.wrap(manager, "free", "request.free")
        self.wrap(manager.block_pool, "get_new_blocks", "pool.allocate")
        self.wrap(manager.block_pool, "free_blocks", "pool.free")
        config = scheduler.kv_cache_config
        layer = scheduler.vllm_config.additional_config["dspark_profile_observation"]["target_layer"]
        targets = {
            name
            for g in config.kv_cache_groups
            for name in g.layer_names
            if f"layers.{layer}." in name and name.endswith(".swa_cache")
        }
        if len(targets) != 1:
            raise ValueError("Writer scheduler target group unavailable")
        related = set()
        for t in config.kv_cache_tensors:
            if targets.intersection(t.shared_by):
                related.update(t.shared_by)
        if not related:
            raise ValueError("Writer scheduler shared backing catalog unavailable")
        if any(t.block_stride > 0 and targets.intersection(t.shared_by) for t in config.kv_cache_tensors):
            related.update(name for t in config.kv_cache_tensors if t.block_stride > 0 for name in t.shared_by)
        self.catalog = {
            "target_layers": sorted(targets),
            "related_layers": sorted(related),
            "groups": [
                {"id": i, "layers": g.layer_names, "spec": str(g.kv_cache_spec)}
                for i, g in enumerate(config.kv_cache_groups)
            ],
            "allocations": [
                {"shared_by": t.shared_by, "bytes": t.size, "offset": t.offset, "block_stride": t.block_stride}
                for t in config.kv_cache_tensors
            ],
        }
        self.catalog_written = False
        for i, m in enumerate(manager.coordinator.single_type_managers):
            if not related.intersection(config.kv_cache_groups[i].layer_names):
                continue
            self.wrap(m, "free", "group.free", i)
            self.wrap(m, "remove_skipped_blocks", "remove_skipped_blocks", i)
            self.wrap(m, "allocate_new_blocks", "allocate_new_blocks", i)

    def state(self):
        return [
            request_state(r)
            for r in self.scheduler().requests.values()
            if (getattr(r.sampling_params, "extra_args", None) or {}).get("dspark_write_point") == self.point
        ]


class WriteTimelineScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.vllm_config.additional_config["dspark_profile_observation"]
        options = cfg["operator_capture"]
        if not options.get("write_timeline") or not self.scheduler_config.async_scheduling:
            raise ValueError("Writer scheduler is only for explicit async diagnostic profiling")
        self.write_trace = PageTrace(self, cfg["directory"], options["point"])
        self.write_trace.install()

    def schedule(self, *args, **kwargs):
        trace = self.write_trace
        trace.step += 1
        trace.active = bool(trace.state())
        if trace.active and not trace.catalog_written:
            trace.record("catalog", **trace.catalog)
            trace.catalog_written = True
        trace.record("schedule.before", requests=trace.state())
        result = super().schedule(*args, **kwargs)
        result._dspark_write_schedule_id = trace.step
        trace.record(
            "schedule.after",
            requests=trace.state(),
            scheduled=result.num_scheduled_tokens,
            spec_lengths={k: len(v) for k, v in result.scheduled_spec_decode_tokens.items()},
            finished=sorted(result.finished_req_ids),
            preempted=sorted(result.preempted_req_ids or ()),
        )
        return result

    def update_from_output(self, scheduler_output, model_output):
        trace = self.write_trace
        trace.active = bool(trace.state())
        trace.record(
            "output.before",
            output_schedule_id=getattr(scheduler_output, "_dspark_write_schedule_id", None),
            requests=trace.state(),
            req_ids=model_output.req_ids,
            sampled_lengths=[len(v) for v in model_output.sampled_token_ids],
        )
        result = super().update_from_output(scheduler_output, model_output)
        trace.record(
            "output.after",
            output_schedule_id=getattr(scheduler_output, "_dspark_write_schedule_id", None),
            requests=trace.state(),
        )
        return result
