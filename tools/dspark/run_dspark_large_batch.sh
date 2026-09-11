#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Preserve CANN/custom OPP; never build, reset a device or terminate other jobs.
set -o pipefail

logged() {
    local name=$1
    shift
    "$@" 2>&1 | tee "$CONF_OUT/$name.log"
    local codes=("${PIPESTATUS[@]}")
    printf '%s\n' "${codes[*]}" > "$CONF_OUT/$name.pipestatus"
    test "${codes[0]}" -eq 0 && test "${codes[1]}" -eq 0
}

main() {
    test "$#" -ge 2 || return 1
    local sha=$1 manifest=$2
    shift 2
    local plugin=/workspace/vllm-ascend-hust core=/workspace/vllm-hust
    local model=/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 previous='' argument
    for argument in "$@"; do
        if test "$previous" = --model; then model=$argument; fi
        case "$argument" in --model=*) model=${argument#--model=} ;; esac
        previous=$argument
    done
    mkdir -p /workspace/dspark-results || return 1
    CONF_OUT=$(mktemp -d /workspace/dspark-results/dspark-large-batch.XXXXXXXX) || return 1
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || return 1
    test "$(git -C "$plugin" rev-parse HEAD)" = "$sha" || return 1
    test "$(git -C "$core" rev-parse HEAD)" = 897306c43bf800e2480cb5c0f3e2da408d85a2fd || return 1
    test "$(git -C "$plugin" branch --show-current)" = feat/dspark || return 1
    test -z "$(git -C "$plugin" status --porcelain)" || return 1
    test -z "$(git -C "$core" status --porcelain)" || return 1
    test -n "${ASCEND_CUSTOM_OPP_PATH:-}" || return 1
    export PYTHONPATH="$plugin:$core:${PYTHONPATH:-}"
    export VLLM_ALLOW_INSECURE_SERIALIZATION=0 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_LAUNCH_BLOCKING=0
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    cd "$plugin" || return 1
    logged source python tools/dspark/p08_r8_checks.py source "$plugin" "$core" || return 1
    logged checkpoint python tools/dspark/verification_tools.py checkpoint \
        --model "$model" \
        --output "$CONF_OUT/checkpoint.json" || return 1
    logged focused python -m pytest -q -ra \
        tests/ut/test_dspark_repeated_inputs.py tests/ut/test_dspark_startup_cost_profile.py \
        tests/ut/test_dspark_profile_request_ids.py tests/ut/test_dspark_profile_context.py \
        tests/ut/test_dspark_profile_nan.py tests/ut/test_dspark_nan_diagnostics.py \
        tests/ut/test_dspark_confidence_verification.py tests/ut/worker/test_dsa_padded_requests.py \
        tests/ut/worker/test_capture_input_aliases.py tests/ut/worker/test_dsa_capture_metadata.py \
        tests/ut/attention/test_dsa_padding_contract.py tests/ut/attention/test_dsa_capture_validation.py \
        tests/ut/worker/test_aclgraph_capture.py tests/ut/test_dspark_draft_config.py \
        tests/ut/test_dspark_graph_rpc.py tests/ut/test_dspark_graph_replay.py \
        tests/ut/test_dspark_acceptance_benchmark.py tests/ut/test_dspark_performance_delivery.py \
        tests/ut/spec_decode/test_dspark_v2_*.py || return 1
    logged generation python tools/dspark/run_large_batch.py \
        --plugin-sha "$sha" --manifest "$manifest" --output-dir "$CONF_OUT/runs" "$@"
}

main "$@"
CONF_RC=$?
if test -n "${CONF_OUT:-}"; then
    printf 'MAIN_RC=%s\nSERVER_RESULT_DIR=%s\n' "$CONF_RC" "$CONF_OUT" > "$CONF_OUT/status.txt"
    cat "$CONF_OUT/status.txt"
    tar -czf "$CONF_OUT-evidence.tar.gz" -C "$(dirname "$CONF_OUT")" "$(basename "$CONF_OUT")"
    CONF_ARCHIVE_RC=$?
    sha256sum "$CONF_OUT-evidence.tar.gz" | tee "$CONF_OUT-evidence.sha256"
    CONF_HASH_CODES=("${PIPESTATUS[@]}")
    printf '%s\n' "${CONF_HASH_CODES[*]}" > "$CONF_OUT-evidence.sha256.pipestatus"
else
    CONF_ARCHIVE_RC=1
    CONF_HASH_CODES=(1 1)
fi
test "$CONF_RC" -eq 0 && test "$CONF_ARCHIVE_RC" -eq 0 && \
    test "${CONF_HASH_CODES[0]}" -eq 0 && test "${CONF_HASH_CODES[1]}" -eq 0
