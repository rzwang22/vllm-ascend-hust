#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One model initialization, original B64 ten points; strict options in child Bash.
set -euo pipefail
test "$#" -ge 3 && test "$#" -le 4 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE [--exit-observation]' >&2; exit 1; }
sha=$1 manifest=$2 core_remote=$3
extra=(--profile-worker-exit)
if test "$#" -eq 4; then
    test "$4" = --exit-observation
    extra+=(--profile-exit-observation)
fi
bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh "$sha" "$manifest" \
    --core-remote "$core_remote" \
    --swa-acceptance-archive /workspace/dspark-results/dspark-swa-lifecycle.sGR4YXDq-evidence.tar.gz \
    --stage profile --batches 64 --num-prompts 400 \
    --profile-experiment target-boundaries --profile-target-layer 1 \
    --profile-target-attention \
    "${extra[@]}" \
    --profile-stop-after-point ctx128-n4-t12-skewed \
    --capture-sizes 6 12 24 48 96 192 384 \
    --profile-contexts 128 2048 --profile-output-tokens 512 \
    --profile-warmup 2 --profile-samples 5 \
    --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9
