#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fresh B128 costs/confidence, then B256; stop on first failure, always export.
set -euo pipefail
test "$#" -eq 3 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE' >&2; exit 1; }
exec bash "$(dirname "$0")/run_dspark_exit_observation.sh" "$@" --batch-expansion
