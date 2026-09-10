# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile-only admission receipts, scoped to one AsyncLLM instance and point.

Frozen Core's public RequestOutput uses external IDs. The collector returned by
AsyncLLM.add_request carries the internal ID after _add_request has enqueued it.
Only the benchmark's token prompts with n=1 are supported (no child fan-out).
"""

import copy
import inspect

MAPPING_SOURCE = "AsyncLLM.add_request returned RequestOutputCollector.request_id after enqueue (n=1)"


class RequestIdentityError(ValueError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(f"Profile request identity: {evidence['reason']} at {evidence['point']}")


class RequestIdObserver:
    def __init__(self, engine, point, expected):
        self.engine = engine
        self.point = point
        self.expected = dict(expected)
        self.mappings = []
        self.errors = []
        self.receipt = None

    def _fail(self, reason, **details):
        evidence = {"reason": reason, "point": self.point, **details}
        self.errors.append(evidence)
        raise RequestIdentityError(evidence)

    def __enter__(self):
        self.had_override = "add_request" in vars(self.engine)
        self.previous_override = vars(self.engine).get("add_request")
        self.original = self.engine.add_request
        signature = inspect.signature(self.original)

        async def observed(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            external = bound.arguments["request_id"]
            params = bound.arguments["params"]
            prompt = bound.arguments["prompt"]
            if external not in self.expected:
                self._fail("unexpected_external_id", external_id=external)
            if getattr(params, "n", None) != 1 or not isinstance(prompt, dict) or "prompt_token_ids" not in prompt:
                self._fail("unsupported_admission", external_id=external)
            collector = await self.original(*args, **kwargs)
            internal = getattr(collector, "request_id", None)
            observation = {
                "point": self.point,
                "request_index": self.expected[external],
                "external_id": external,
                "internal_id": internal,
            }
            if not isinstance(internal, str) or not internal:
                self._fail("mapping_missing", observation=observation)
            if any(row["external_id"] == external or row["internal_id"] == internal for row in self.mappings):
                self._fail("mapping_conflict", observation=observation, prior=copy.deepcopy(self.mappings))
            self.mappings.append(observation)
            return collector

        self.wrapper = observed
        self.engine.add_request = observed
        return self

    def __exit__(self, exc_type, exc, traceback):
        restored = self.engine.add_request is self.wrapper
        if restored:
            if self.had_override:
                self.engine.add_request = self.previous_override
            else:
                del self.engine.add_request
        else:
            self.errors.append({"reason": "observer_replaced", "point": self.point})
        self.receipt = copy.deepcopy(
            {
                "schema_version": 1,
                "source": MAPPING_SOURCE,
                "point": self.point,
                "expected_external_ids": list(self.expected),
                "mappings": self.mappings,
                "errors": self.errors,
                "hook_restored": restored,
            }
        )
        # No live request/point state survives the observation lifetime. The
        # detached JSON receipt is deliberately retained as diagnostic evidence.
        self.engine = self.original = self.previous_override = self.wrapper = None
        self.point = None
        self.expected.clear()
        self.mappings.clear()
        self.errors.clear()
        if not restored and exc is None:
            raise RequestIdentityError({"reason": "observer_replaced", "point": self.receipt["point"]})


def validate_point_request_ids(point, stream, snapshots, history):
    """Validate exact internal IDs for every event; history is local to collect()."""
    receipt = (stream or {}).get("request_id_mapping")
    records = (stream or {}).get("requests", [])
    expected = {r["request_id"]: r["request_index"] for r in records if r is not None}
    evidence = {
        "point": point,
        "expected_external_ids": list(expected),
        "expected_internal_ids": [],
        "mapping": receipt,
        "observed_events": [
            {
                "rank": snapshot["rank"],
                "event_kind": event["kind"],
                "event_point": event["point"],
                "actual_internal_ids": event["request_ids"],
                "unknown_internal_ids": sorted(
                    set(event["request_ids"]) - {row.get("internal_id") for row in (receipt or {}).get("mappings", [])}
                ),
            }
            for snapshot in snapshots
            for event in snapshot["cost_profile"]["measurements"]
        ],
        "event_failures": [],
    }

    def fail(reason):
        raise RequestIdentityError({**evidence, "reason": reason})

    if not receipt:
        fail("mapping_missing")
    if receipt.get("source") != MAPPING_SOURCE or receipt.get("point") != point or not receipt.get("hook_restored"):
        fail("invalid_mapping_provenance")
    if receipt.get("errors"):
        fail(receipt["errors"][0]["reason"])
    rows = receipt.get("mappings", [])
    internals = [r.get("internal_id") for r in rows]
    externals = [r.get("external_id") for r in rows]
    evidence["expected_internal_ids"] = internals
    if any(not isinstance(key, str) or not key for key in internals):
        fail("mapping_missing")
    if len(set(internals)) != len(internals) or len(set(externals)) != len(externals):
        fail("mapping_conflict")
    if set(externals) != set(expected) or set(receipt["expected_external_ids"]) != set(expected):
        fail("mapping_missing")
    if len(expected) != len(records) or len(set(expected.values())) != len(records):
        fail("request_instance_conflict")
    if any(row["point"] != point or row["request_index"] != expected[row["external_id"]] for row in rows):
        fail("mapping_instance_conflict")
    if set(internals).intersection(history):
        fail("internal_id_reused_across_points")
    for snapshot in snapshots:
        for event in snapshot["cost_profile"]["measurements"]:
            actual = event["request_ids"]
            unknown = sorted(set(actual) - set(internals))
            if (
                unknown
                or event["point"] != point
                or len(set(actual)) != len(actual)
                or len(actual) != event["requests"]
            ):
                prior = {key: history[key] for key in unknown if key in history}
                evidence["event_failures"].append(
                    {
                        "rank": snapshot["rank"],
                        "event_kind": event["kind"],
                        "event_point": event["point"],
                        "actual_internal_ids": actual,
                        "unknown_internal_ids": unknown,
                        "previous_points": prior,
                        "reason": (
                            "previous_point_event"
                            if prior
                            else "unknown_internal_id"
                            if unknown
                            else "wrong_event_point"
                            if event["point"] != point
                            else "invalid_event_request_ids"
                        ),
                    }
                )
    if evidence["event_failures"]:
        fail("event_ownership_failed")
    history.update({key: point for key in internals})
    return {"point": point, "source": MAPPING_SOURCE, "validated_internal_ids": internals}
