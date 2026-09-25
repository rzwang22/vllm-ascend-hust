#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Six cases only; reuse source, environment, PIPESTATUS and unconditional export.
set -euo pipefail
test "$#" -eq 3 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE' >&2; exit 1; }
exec bash "$(dirname "$0")/run_dspark_exit_observation.sh" "$@" --performance-comparison
