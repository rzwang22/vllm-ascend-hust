# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate actual named-policy receipts, independently of numerical acceptance."""

import json

from tools.dspark.shutdown_policy import budget


def check(root, name):
    expected = budget(name)
    failures = []

    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            failures.append(f"{path.name}: {error}")
            return {}

    cleanup = read(root / "cleanup.json")
    workers = read(root / "worker-cleanup.json")
    supervisor = read(root.parent / "b64-supervisor.json")
    command = read(root.parent / "b64-command.json")
    residual = read(root.parent / "b64-residual.json")
    drain = read(root / "output-handler-shutdown.json")
    plan = read(root / "plan.json")
    for label, receipt in (("frontend", cleanup), ("workers", workers), ("supervisor", supervisor)):
        actual = receipt.get("shutdown_budget", {})
        if any(actual.get(k) != v for k, v in expected.items()):
            failures.append(f"{label}: policy receipt missing/mismatched")
    requirements = {
        "effective Core configuration": workers.get("shutdown_budget", {}).get("effective_core_worker_seconds") == 25,
        "no observation wrapper": plan.get("exit_observation") is False and workers.get("exit_observation") is False,
        "no native observation": not (root / "worker-exit/native/observation.json").exists(),
        "frontend budgets": cleanup.get("timeout_seconds") == 36 and cleanup.get("outer_timeout_seconds") == 40,
        "supervisor budget": supervisor.get("failure_grace_seconds") == 48,
        "frontend completion": cleanup.get("success") is True
        and cleanup.get("event_loop") == "closed"
        and cleanup.get("error") is None
        and cleanup.get("recording_error") is None,
        "drained output": drain.get("success") is True and drain.get("error") is None,
        "worker recording": workers.get("error") is None and workers.get("recording_error") is None,
        "reaped workers": workers.get("reap", {}).get("budget_seconds") == 1
        and len(workers.get("reap", {}).get("workers", [])) == 8
        and {r.get("rank") for r in workers["reap"]["workers"]} == set(range(8))
        and all(r.get("status") == "reaped" and r.get("raw_exitcode") == 0 for r in workers["reap"]["workers"]),
        "supervisor completion": supervisor.get("raw_returncode") == 0
        and supervisor.get("first_failure") is None
        and supervisor.get("signals_sent") == []
        and supervisor.get("owned_group_remaining") is False,
        "strict log scan": command.get("rc") == 0 and command.get("log_scan_rc") == 0,
        "post-run resources": residual.get("success") is True and residual.get("error") is None,
    }
    failures.extend(k for k, ok in requirements.items() if not ok)
    return {
        "acceptance_policy": name,
        "original_budget_reference": "profile-default-5s",
        "original_budget_acceptance": "NOT_EVALUATED",
        "shutdown_policy_evidence_valid": not failures,
        "shutdown_policy_errors": failures,
    }
