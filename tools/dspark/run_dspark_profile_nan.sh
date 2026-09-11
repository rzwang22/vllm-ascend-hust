#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Independent diagnostic, one B64 engine, all predecessors through the failed point.
set -o pipefail
main() {
    test "$#" -eq 2 || return 1
    bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh "$1" "$2" \
        --stage profile --batches 64 --num-prompts 400 \
        --profile-nan-diagnostic --profile-stop-after-point ctx128-n4-t12-skewed \
        --capture-sizes 6 12 24 48 96 192 384 \
        --profile-contexts 128 2048 --profile-output-tokens 512 \
        --profile-warmup 2 --profile-samples 5 \
        --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9
}
main "$@"
