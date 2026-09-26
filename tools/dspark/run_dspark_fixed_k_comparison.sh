#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Exactly B256 fixed K5 then K8; retain bounded exit and unconditional evidence export.
set -euo pipefail
test "$#" -eq 3 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE' >&2; exit 1; }
exec bash "$(dirname "$0")/run_dspark_exit_observation.sh" "$@" --fixed-k-comparison
