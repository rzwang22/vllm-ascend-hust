# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named, opt-in profile shutdown budgets. No environment mutation on import."""

POLICY_NAME = "dspark-profile-25s-v1"
# Existing Core option, registered in vllm/envs.py; not a new plugin variable.
CORE_WORKER_ENV = "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"
WORKER_SECONDS = 25
ENGINE_SECONDS = 36
SUPERVISOR_SECONDS = 48


def stack_signals_enabled(additional):
    """Active stack signals are diagnosis only, independent of gdb and receipts."""
    enabled = additional.get("dspark_profile_stack_signals", False)
    if type(enabled) is not bool:
        raise ValueError("dspark_profile_stack_signals must be boolean")
    if enabled and (
        not additional.get("dspark_profile_exit_observation")
        or not additional.get("dspark_profile_worker_exit")
        or additional.get("dspark_profile_shutdown_policy")
        or additional.get("dspark_confidence_acceptance")
    ):
        raise ValueError("Stack signals require explicit exit diagnosis, not formal acceptance")
    return enabled


def budget(name, *, exit_observation=False, worker_exit=True):
    if name is None:
        return None
    if name != POLICY_NAME or exit_observation or not worker_exit:
        raise ValueError("Named shutdown policy requires worker receipts and excludes exit observation")
    return dict(
        name=name,
        worker_seconds=WORKER_SECONDS,
        term_seconds=4,
        shared_reap_seconds=1,
        engine_seconds=ENGINE_SECONDS,
        frontend_outer_seconds=40,
        supervisor_seconds=SUPERVISOR_SECONDS,
        debugger_enabled=False,
        attachment_count=0,
        observation_extra_wait_seconds=0,
    )


def installed_budget(name, **kwargs):
    result = budget(name, **kwargs)
    if result is not None:
        # Read Core's effective accessor in the owning process, including its
        # environment cache. Never patch envs or reinterpret an old receipt.
        from vllm import envs

        actual = getattr(envs, CORE_WORKER_ENV)
        if actual != result["worker_seconds"]:
            raise ValueError(f"Effective Core worker grace {actual!r} != {result['worker_seconds']}")
        result["effective_core_worker_seconds"] = actual
    return result


def child_command(name, command):
    selected = budget(name)
    if selected is None:
        return command
    # Fresh child sees the existing Core setting before importing/caching envs.
    # The invoking shell and unrelated processes keep their own defaults.
    return ["env", f"{CORE_WORKER_ENV}={selected['worker_seconds']}", *command]


def confidence_acceptance_enabled(additional):
    """Explicit workload gate, independent of installed device modules."""
    options = additional.get("dspark_confidence_verification", {})
    return (
        additional.get("dspark_confidence_acceptance") is True
        and additional.get("dspark_profile_worker_exit") is True
        and additional.get("dspark_profile_shutdown_policy") == POLICY_NAME
        and not additional.get("dspark_profile_exit_observation")
        and options.get("mode") == "confidence"
        and options.get("profile") is False
    )


def performance_enabled(additional):
    """Six-case consumer only: passive exit receipts without profile probes."""
    options = additional.get("dspark_confidence_verification")
    return (
        additional.get("dspark_performance_comparison") is True
        and additional.get("dspark_profile_worker_exit") is True
        and additional.get("dspark_profile_shutdown_policy") == POLICY_NAME
        and additional.get("dspark_profile_stack_signals") is False
        and additional.get("dspark_profile_exit_debugger") is False
        and not additional.get("dspark_profile_exit_observation")
        and not additional.get("dspark_confidence_acceptance")
        and (options is None or (options.get("mode") == "confidence" and options.get("profile") is False))
    )
