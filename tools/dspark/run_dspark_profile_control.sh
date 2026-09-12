#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One experiment per invocation. No automatic control sweep or later batches.
set -o pipefail
main() {
    test "$#" -eq 3 || return 1
    local sha=$1 manifest=$2 mode=$3
    case "$mode" in baseline|metadata-only|context-kv-sync|numeric-boundaries|upstream-boundaries|auxiliary-transfers|target-boundaries) ;; *) return 1 ;; esac
    bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh "$sha" "$manifest" \
        --stage profile --batches 64 --num-prompts 400 \
        --profile-experiment "$mode" --profile-stop-after-point ctx128-n4-t12-skewed \
        --capture-sizes 6 12 24 48 96 192 384 \
        --profile-contexts 128 2048 --profile-output-tokens 512 \
        --profile-warmup 2 --profile-samples 5 \
        --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9
}
main "$@"
