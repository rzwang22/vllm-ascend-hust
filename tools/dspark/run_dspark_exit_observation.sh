#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Child Bash only: archive the outer log/status and the new model evidence.
set -uo pipefail
out=''
formal=false
coverage=false
formal_cost=false
confidence=false
expansion=false
performance=false
fixed_k=false
coverage_phase=''
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
        case "$4" in
            --fixed-k-comparison) mode=$4; formal=true; performance=true; fixed_k=true ;;
            --performance-comparison) mode=$4; formal=true; performance=true ;;
            --batch-expansion) mode=$4; formal=true; expansion=true ;;
            --confidence-acceptance) mode=$4; formal=true; confidence=true ;;
            --formal-cost=b64-confidence-cost-v1) mode=$4; formal=true; formal_cost=true ;;
            --coverage=b64-functional-1|--coverage=b64-functional-2) mode=$4; formal=true; coverage=true; coverage_phase=${4#--coverage=} ;;
            --no-debugger) mode=--exit-observation-no-debugger ;;
            --shutdown-policy=dspark-profile-25s-v1) mode=$4; formal=true ;;
            *) return 1 ;;
        esac
    fi
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || return 1
    mkdir -p /workspace/dspark-results || return 1
    local prefix=dspark-exit-observation
    if test "$formal" = true; then prefix=dspark-model-acceptance; fi
    if test "$coverage" = true; then prefix=dspark-functional-coverage; fi
    if test "$formal_cost" = true; then prefix=dspark-formal-cost; fi
    if test "$confidence" = true; then prefix=dspark-confidence-acceptance; fi
    if test "$expansion" = true; then prefix=dspark-batch-expansion; fi
    if test "$performance" = true; then prefix=dspark-performance-comparison; fi
    if test "$fixed_k" = true; then prefix=dspark-fixed-k-comparison; fi
    out=$(mktemp -d "/workspace/dspark-results/$prefix.XXXXXXXX") || return 1
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
    if test "$formal" = true; then
        printf 'MAIN_RC=%s\nACCEPTANCE_POLICY=dspark-profile-25s-v1\nFORMAL_ACCEPTANCE=SEE_MODEL_REPORT\nORIGINAL_BUDGET=NOT_EVALUATED\n' "$rc" > "$out/status.txt"
    fi
    if test "$formal_cost" = true; then
        printf 'MAIN_RC=%s\nCOST_PLAN=b64-confidence-cost-v1\nCOST_TABLE_USABLE=SEE_COST_PUBLICATION\nPERFORMANCE_ELIGIBLE=false\n' "$rc" > "$out/status.txt"
    fi
    if test "$performance" = true; then
        printf 'MAIN_RC=%s\nRESULT=SEE_PERFORMANCE_SUMMARY\nCOST_TABLES=FROZEN_UNMODIFIED\nORIGINAL_5S_BUDGET=NOT_EVALUATED\n' "$rc" > "$out/status.txt"
    fi
    if test "$expansion" = true; then printf 'BATCH_ORDER=128,256\nRESULT=SEE_EXPANSION_REPORT\nPERFORMANCE_ELIGIBLE=false\n' >> "$out/status.txt"; fi
    if test "$confidence" = true; then printf 'CONFIDENCE_CLOSED_LOOP=SEE_CONFIDENCE_REPORT\nPERFORMANCE_ELIGIBLE=false\n' >> "$out/status.txt"; fi
    if test "$coverage" = true; then printf 'FUNCTIONAL_PHASE=%s\nORIGINAL_TEN_POINT_BLOCKER=CLOSED\n' "$coverage_phase" >> "$out/status.txt"; fi
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
