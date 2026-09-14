# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-free native debugger preflight for the opt-in exit observation."""

import argparse
from pathlib import Path

from vllm import envs

from vllm_ascend.diagnostics.dspark_exit_observation import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS != 5:
        raise ValueError("Expected unchanged original VLLM worker grace=5 seconds")
    result = preflight(args.directory, import_runtime=True)
    print(result.get("error", "Native backtrace and forced debugger detach preflight passed"))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
