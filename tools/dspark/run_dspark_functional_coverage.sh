#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One named, bounded B64 functional phase. No automatic subsequent phases.
set -euo pipefail
test "$#" -ge 3 && test "$#" -le 4 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE [b64-functional-1|b64-functional-2]' >&2; exit 1; }
phase=${4:-b64-functional-1}
case "$phase" in b64-functional-1|b64-functional-2) ;; *) exit 1 ;; esac
exec bash "$(dirname "$0")/run_dspark_exit_observation.sh" "$1" "$2" "$3" "--coverage=$phase"
