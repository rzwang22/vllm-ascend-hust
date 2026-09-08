#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Run in the caller's existing CANN/OPP environment; no builds or resets.
set -o pipefail
PERF_SHA=${1:-}
PERF_MANIFEST=${2:-}
PERF_STAGE=${3:-smoke}
if test "$#" -ge 3; then shift 3; else set --; fi
PERF_PLUGIN=/workspace/vllm-ascend-hust
PERF_CORE=/workspace/vllm-hust
PERF_OUT=

logged() {
    local label=$1
    shift
    "$@" 2>&1 | tee "$PERF_OUT/$label.log"
    local codes=("${PIPESTATUS[@]}")
    printf '%s\n' "${codes[*]}" > "$PERF_OUT/$label.pipestatus" || return 1
    test "${codes[1]}" -eq 0 || return 1
    return "${codes[0]}"
}

main() {
    mkdir -p /workspace/dspark-results || return 1
    PERF_OUT=$(mktemp -d /workspace/dspark-results/dspark-performance.XXXXXX) || return 1
    cp -- "$0" "$PERF_OUT/server-command.sh" || return 1
    printf 'STAGE=%s\nSHA=%s\nMANIFEST=%s\nRESULT_DIR=%s\n' \
        "$PERF_STAGE" "$PERF_SHA" "$PERF_MANIFEST" "$PERF_OUT"
    [[ "$PERF_SHA" =~ ^[0-9a-f]{40}$ ]] || return 1
    test "$(git -C "$PERF_PLUGIN" rev-parse HEAD)" = "$PERF_SHA" || return 1
    test "$(git -C "$PERF_CORE" rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd || return 1
    test "$(git -C "$PERF_PLUGIN" branch --show-current)" = feat/dspark || return 1
    test -z "$(git -C "$PERF_PLUGIN" status --porcelain)" || return 1
    test -z "$(git -C "$PERF_CORE" status --porcelain)" || return 1
    test -n "${ASCEND_CUSTOM_OPP_PATH:-}" || return 1
    export PYTHONPATH="$PERF_PLUGIN:$PERF_CORE:${PYTHONPATH:-}"
    export VLLM_ALLOW_INSECURE_SERIALIZATION=0 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_LAUNCH_BLOCKING=0
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    cd "$PERF_PLUGIN" || return 1
    logged source python tools/dspark/p08_r8_checks.py source "$PERF_PLUGIN" "$PERF_CORE" || return 1
    logged focused python -m pytest -q -ra \
        tests/ut/test_dspark_performance_delivery.py \
        tests/ut/test_dspark_acceptance_benchmark.py tests/ut/test_dspark_graph_rpc.py \
        tests/ut/test_dspark_graph_replay.py tests/ut/test_dspark_draft_config.py \
        tests/ut/worker/test_dsa_padded_requests.py || return 1
    local options=()
    case "$PERF_STAGE" in
        smoke) options=(--num-prompts 4 --max-num-seqs 2 --repeats 1 --output-len 128 --warmup-prompts 1) ;;
        code64) options=(--num-prompts 64 --max-num-seqs 4 --repeats 3 --output-len 1024 --warmup-prompts 1) ;;
        explore) options=(--num-prompts 2048 --max-num-seqs 400 --repeats 3 --output-len 256 --warmup-prompts 4) ;;
        *) printf 'Unknown stage: %s\n' "$PERF_STAGE"; return 1 ;;
    esac
    # Additional explicit options can override defaults; nothing escalates automatically.
    logged suite python tools/dspark/run_performance_suite.py \
        --plugin-sha "$PERF_SHA" --manifest "$PERF_MANIFEST" --output-dir "$PERF_OUT/runs" \
        --execute "${options[@]}" "$@"
}

main "$@"
PERF_RC=$?
if test -n "$PERF_OUT"; then
    printf 'MAIN_RC=%s\nSERVER_RESULT_DIR=%s\n' "$PERF_RC" "$PERF_OUT" > "$PERF_OUT/status.txt"
    cat "$PERF_OUT/status.txt"
    tar -czf "$PERF_OUT-evidence.tar.gz" -C "$(dirname "$PERF_OUT")" "$(basename "$PERF_OUT")"
    PERF_ARCHIVE_RC=$?
    if test "$PERF_ARCHIVE_RC" -eq 0; then
        sha256sum "$PERF_OUT-evidence.tar.gz" | tee "$PERF_OUT-evidence.sha256"
        PERF_CODES=("${PIPESTATUS[@]}")
        printf '%s\n' "${PERF_CODES[*]}" > "$PERF_OUT-evidence.sha256.pipestatus"
        test "${PERF_CODES[0]}" -eq 0 && test "${PERF_CODES[1]}" -eq 0 || PERF_ARCHIVE_RC=1
    fi
    printf 'ARCHIVE_RC=%s\nEVIDENCE=%s-evidence.tar.gz\n' "$PERF_ARCHIVE_RC" "$PERF_OUT"
else
    PERF_ARCHIVE_RC=1
fi
test "$PERF_RC" -eq 0 && test "$PERF_ARCHIVE_RC" -eq 0
