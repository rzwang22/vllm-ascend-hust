#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Run under the existing CANN/custom OPP environment. No build or reset.
set -o pipefail
G64_SHA=${1:-}
G64_EVIDENCE=${2:-/workspace/dspark-results/p08-r9c-400request-sweep._fb6g2_4}
G64_PLUGIN=/workspace/vllm-ascend-hust
G64_CORE=/workspace/vllm-hust
G64_CORE_SHA=897306c43bf800e2480cb5c0f3e2da408d85a2fd
G64_CHECKS=$G64_PLUGIN/tools/dspark/graph64_checks.py
G64_COMMON=$G64_PLUGIN/tools/dspark/p08_r8_checks.py
G64_OUT=
G64_STAGE=initialization
G64_SOURCE_RC=99 G64_DATA_RC=99 G64_FOCUSED_RC=99 G64_GRAPH_RC=99 G64_EAGER_RC=99 G64_COMPARE_RC=99
G64_RESIDUAL_RC=99 G64_NPU_RC=99 G64_ARCHIVE_RC=99 G64_SHA_RC=99

logged() {
    local label=$1
    shift
    "$@" 2>&1 | tee "$G64_OUT/$label.log"
    local codes=("${PIPESTATUS[@]}")
    printf '%s\n' "${codes[*]}" > "$G64_OUT/$label-pipestatus.txt" || return 1
    test "${codes[1]}" -eq 0 || return 1
    return "${codes[0]}"
}

source_gate() {
    [[ "$G64_SHA" =~ ^[0-9a-f]{40}$ ]] || return 1
    test "$(git -C "$G64_PLUGIN" rev-parse HEAD)" = "$G64_SHA" || return 1
    test "$(git -C "$G64_CORE" rev-parse HEAD)" = "$G64_CORE_SHA" || return 1
    test "$(git -C "$G64_PLUGIN" branch --show-current)" = feat/dspark || return 1
    test -z "$(git -C "$G64_PLUGIN" status --porcelain)" || return 1
    test -z "$(git -C "$G64_CORE" status --porcelain)" || return 1
    git -C "$G64_PLUGIN" merge-base --is-ancestor 7e23a859defa5a12b3e583fcf4ce57a52da94c71 HEAD || return 1
    printf 'PLUGIN_SHA=%s\nCORE_SHA=%s\n' "$G64_SHA" "$G64_CORE_SHA"
    # Preserve the caller's CANN paths/custom OPP rather than sourcing a new environment.
    test -n "${ASCEND_CUSTOM_OPP_PATH:-}" || return 1
    export PYTHONPATH="$G64_PLUGIN:$G64_CORE:${PYTHONPATH:-}"
    export VLLM_ALLOW_INSECURE_SERIALIZATION=0 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_LAUNCH_BLOCKING=0
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    python "$G64_COMMON" source "$G64_PLUGIN" "$G64_CORE"
}

run_case() {
    local mode=$1
    local launch_rc=99 scan_rc=99 idle_rc=99 npu_rc=99 result_rc=99
    logged "$mode-idle-before" python "$G64_COMMON" idle || return 1
    logged "$mode-npu-before" npu-smi info || return 1
    logged "$mode-run" python "$G64_CHECKS" launch "$G64_PLUGIN" "$G64_OUT" "$mode"
    launch_rc=$?
    # All post-run gates run even if generation failed; no processes are killed.
    logged "$mode-errors" python "$G64_CHECKS" scan "$G64_OUT/$mode-run.log"
    scan_rc=$?
    logged "$mode-idle-after" python "$G64_COMMON" idle
    idle_rc=$?
    logged "$mode-npu-after" npu-smi info
    npu_rc=$?
    if test "$launch_rc" -eq 0; then
        logged "$mode-result" python "$G64_CHECKS" result "$G64_OUT/$mode/result.json" "$G64_SHA" "$mode"
        result_rc=$?
    fi
    printf 'LAUNCH_RC=%s\nERROR_SCAN_RC=%s\nIDLE_RC=%s\nNPU_RC=%s\nRESULT_RC=%s\n' \
        "$launch_rc" "$scan_rc" "$idle_rc" "$npu_rc" "$result_rc" > "$G64_OUT/$mode-gate.txt"
    test "$launch_rc" -eq 0 && test "$scan_rc" -eq 0 && test "$idle_rc" -eq 0 &&
        test "$npu_rc" -eq 0 && test "$result_rc" -eq 0
}

main() {
    mkdir -p /workspace/dspark-results || return 1
    G64_OUT=$(mktemp -d /workspace/dspark-results/dspark-graph64-retest.XXXXXX) || return 1
    cp -- "$0" "$G64_OUT/server-command.sh" || return 1
    cp -- "$G64_CHECKS" "$G64_OUT/graph64_checks.py" || return 1
    printf 'RESULT_DIR=%s\nPLUGIN_SHA=%s\nORIGINAL_EVIDENCE=%s\n' "$G64_OUT" "$G64_SHA" "$G64_EVIDENCE"
    G64_STAGE=source
    source_gate > "$G64_OUT/source.log" 2>&1
    G64_SOURCE_RC=$?
    cat "$G64_OUT/source.log"
    test "$G64_SOURCE_RC" -eq 0 || return "$G64_SOURCE_RC"
    G64_STAGE=dataset
    logged dataset python "$G64_CHECKS" prepare "$G64_EVIDENCE" "$G64_OUT" "$G64_PLUGIN"
    G64_DATA_RC=$?
    test "$G64_DATA_RC" -eq 0 || return "$G64_DATA_RC"
    G64_STAGE=focused
    cd "$G64_PLUGIN" || return 1
    logged focused python -m pytest -q -ra \
        tests/ut/worker/test_dsa_padded_requests.py tests/ut/test_dspark_graph64_runner.py \
        tests/ut/worker/test_model_runner_v2_input_batch.py tests/ut/worker/test_dsa_model_runner_v2_metadata.py \
        tests/ut/worker/test_capture_input_aliases.py tests/ut/worker/test_dsa_capture_metadata.py \
        tests/ut/attention/test_dsa_padding_contract.py tests/ut/attention/test_dsa_capture_validation.py \
        tests/ut/test_dspark_draft_config.py tests/ut/worker/test_aclgraph_capture.py \
        tests/ut/test_dspark_graph_rpc.py tests/ut/test_dspark_graph_replay.py \
        tests/ut/test_dspark_acceptance_benchmark.py tests/ut/spec_decode/test_dspark_v2_*.py
    G64_FOCUSED_RC=$?
    test "$G64_FOCUSED_RC" -eq 0 || return "$G64_FOCUSED_RC"
    G64_STAGE=graph64
    run_case graph
    G64_GRAPH_RC=$?
    test "$G64_GRAPH_RC" -eq 0 || return "$G64_GRAPH_RC"
    G64_STAGE=eager64
    run_case eager
    G64_EAGER_RC=$?
    test "$G64_EAGER_RC" -eq 0 || return "$G64_EAGER_RC"
    G64_STAGE=comparison
    logged comparison python "$G64_CHECKS" compare "$G64_OUT" "$G64_SHA"
    G64_COMPARE_RC=$?
    return "$G64_COMPARE_RC"
}

main
G64_MAIN_RC=$?
if test -n "$G64_OUT"; then
    logged final-idle python "$G64_COMMON" idle
    G64_RESIDUAL_RC=$?
    logged final-npu npu-smi info
    G64_NPU_RC=$?
    G64_STATUS=FAILED
    if test "$G64_MAIN_RC" -eq 0 && test "$G64_RESIDUAL_RC" -eq 0 && test "$G64_NPU_RC" -eq 0; then
        G64_STATUS=PASS
    fi
    printf 'STATUS=%s\nLAST_STAGE=%s\nSOURCE_RC=%s\nDATASET_RC=%s\nFOCUSED_RC=%s\nGRAPH64_RC=%s\nEAGER64_RC=%s\nCOMPARE_RC=%s\nRESIDUAL_RC=%s\nNPU_RC=%s\nMAIN_RC=%s\n' \
        "$G64_STATUS" "$G64_STAGE" "$G64_SOURCE_RC" "$G64_DATA_RC" "$G64_FOCUSED_RC" "$G64_GRAPH_RC" \
        "$G64_EAGER_RC" "$G64_COMPARE_RC" "$G64_RESIDUAL_RC" "$G64_NPU_RC" "$G64_MAIN_RC" > "$G64_OUT/gate.txt"
    cat "$G64_OUT/gate.txt"
    tar -czf "$G64_OUT-evidence.tar.gz" -C "$(dirname "$G64_OUT")" "$(basename "$G64_OUT")"
    G64_ARCHIVE_RC=$?
    if test "$G64_ARCHIVE_RC" -eq 0; then
        sha256sum "$G64_OUT-evidence.tar.gz" | tee "$G64_OUT-evidence.sha256"
        G64_PIPE=("${PIPESTATUS[@]}")
        printf '%s\n' "${G64_PIPE[*]}" > "$G64_OUT-evidence.sha256-pipestatus.txt"
        G64_SHA_RC=${G64_PIPE[0]}
        test "${G64_PIPE[1]}" -eq 0 || G64_SHA_RC=1
    fi
    printf 'ARCHIVE_RC=%s\nSHA_RC=%s\nEVIDENCE=%s-evidence.tar.gz\n' "$G64_ARCHIVE_RC" "$G64_SHA_RC" "$G64_OUT"
fi
test "$G64_MAIN_RC" -eq 0 && test "$G64_RESIDUAL_RC" -eq 0 && test "$G64_NPU_RC" -eq 0 &&
    test "$G64_ARCHIVE_RC" -eq 0 && test "$G64_SHA_RC" -eq 0
