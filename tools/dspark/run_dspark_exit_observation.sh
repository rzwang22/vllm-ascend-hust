#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Child Bash only: archive the outer log/status and the new model evidence.
set -uo pipefail
out=''
logged() {
    local label=$1; shift
    "$@" 2>&1 | tee "$out/$label.log"
    local codes=("${PIPESTATUS[@]}")
    printf '%s\n' "${codes[*]}" > "$out/$label.pipestatus"
    if test "${codes[0]}" -ne 0; then return "${codes[0]}"; fi
    return "${codes[1]}"
}
main() {
    test "$#" -ge 3 && test "$#" -le 4 || return 1
    local sha=$1 manifest=$2 remote=$3 mode=--exit-observation
    if test "$#" -eq 4; then
        test "$4" = --no-debugger || return 1
        mode=--exit-observation-no-debugger
    fi
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || return 1
    mkdir -p /workspace/dspark-results || return 1
    out=$(mktemp -d /workspace/dspark-results/dspark-exit-observation.XXXXXXXX) || return 1
    printf 'EXIT_OBSERVATION_DIR=%s\n' "$out"
    cd /workspace/vllm-ascend-hust || return 1
    test -z "$(git status --porcelain)" || return 1
    logged fetch git fetch origin feat/dspark || return "$?"
    logged checkout git merge --ff-only "$sha" || return "$?"
    test "$(git rev-parse HEAD)" = "$sha" || return 1
    logged driver bash tools/dspark/run_dspark_swa_acceptance.sh "$sha" "$manifest" "$remote" "$mode"
}
main "$@"
rc=$?
export_rc=0
if test -n "$out"; then
    printf 'MAIN_RC=%s\nDIAGNOSTIC_ONLY=true\nFORMAL_ACCEPTANCE=NOT_EVALUATED\n' "$rc" > "$out/status.txt"
    if test -f "$out/driver.log"; then
        model_dir=$(sed -n 's/^SERVER_RESULT_DIR=//p' "$out/driver.log" | head -1)
        if [[ "$model_dir" =~ ^/workspace/dspark-results/dspark-large-batch\.[A-Za-z0-9]+$ ]] && test -f "$model_dir-evidence.tar.gz"; then
            cp "$model_dir-evidence.tar.gz" "$out/model-evidence.tar.gz" || export_rc=$?
            cp "$model_dir-evidence.sha256" "$out/model-evidence.original.sha256" || export_rc=$?
            printf '%s\n' "$model_dir" > "$out/model-source.txt"
        else
            printf 'Model evidence unavailable; inspect driver log (possibly preflight failed)\n' > "$out/export-error.txt"
            export_rc=1
        fi
    fi
    printf 'EXPORT_RC=%s\n' "$export_rc" >> "$out/status.txt"
    tar -czf "$out-evidence.tar.gz" -C "$(dirname "$out")" "$(basename "$out")" || export_rc=$?
    sha256sum "$out-evidence.tar.gz" | tee "$out-evidence.sha256"
    hash_codes=("${PIPESTATUS[@]}")
    printf '%s\n' "${hash_codes[*]}" > "$out-evidence.sha256.pipestatus"
    if test "${hash_codes[0]}" -ne 0; then export_rc=${hash_codes[0]}; fi
    if test "${hash_codes[1]}" -ne 0 && test "$export_rc" -eq 0; then export_rc=${hash_codes[1]}; fi
    printf 'DIAGNOSTIC_MAIN_RC=%s EXPORT_RC=%s\n' "$rc" "$export_rc"
fi
if test "$rc" -ne 0; then exit "$rc"; fi
exit "$export_rc"
