# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit fixed-length extrapolation; never enables adaptive K8 or changes weights."""


def validate_draft_length(k, additional_config):
    if type(k) is int and k == 5:
        return
    options = additional_config or {}
    if (
        type(k) is int
        and k == 8
        and options.get("dspark_fixed_k8_experiment") is True
        and not options.get("dspark_confidence_verification")
    ):
        return
    raise ValueError("DSpark requires fixed K=5; experimental fixed K=8 needs explicit opt-in without confidence")
