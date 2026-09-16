#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One named, bounded B64 functional phase. No automatic subsequent phases.
set -euo pipefail
test "$#" -eq 3 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE' >&2; exit 1; }
exec bash "$(dirname "$0")/run_dspark_exit_observation.sh" "$@" --coverage=b64-functional-1
