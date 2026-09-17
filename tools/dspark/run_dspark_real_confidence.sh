#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One natural-EOS frozen workload; no calibration/model reprofile or comparison.
set -euo pipefail
test "$#" -eq 3 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE' >&2; exit 1; }
exec bash "$(dirname "$0")/run_dspark_exit_observation.sh" "$@" --confidence-acceptance
