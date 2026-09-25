#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One model initialization, original B64 ten points; strict options in child Bash.
set -euo pipefail
test "$#" -ge 3 && test "$#" -le 4 || { echo 'Usage: script PLUGIN_SHA MANIFEST CORE_REMOTE [--exit-observation|--exit-observation-no-debugger]' >&2; exit 1; }
sha=$1 manifest=$2 core_remote=$3
extra=(--profile-worker-exit)
selection=(--profile-stop-after-point ctx128-n4-t12-skewed)
experiment=(--profile-experiment target-boundaries --profile-target-layer 1 --profile-target-attention)
acceptance=(--swa-acceptance-archive /workspace/dspark-results/dspark-swa-lifecycle.sGR4YXDq-evidence.tar.gz)
contexts=(128 2048)
if test "$#" -eq 4; then
    case "$4" in
        --performance-comparison)
            exec bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh "$sha" "$manifest" --core-remote "$core_remote" --performance-comparison ;;
        --batch-expansion)
            exec bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh "$sha" "$manifest" --core-remote "$core_remote" --batch-expansion ;;
        --confidence-acceptance)
            exec bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh "$sha" "$manifest" --core-remote "$core_remote" --confidence-acceptance ;;
        --formal-cost=b64-confidence-cost-v1)
            extra+=(--profile-shutdown-policy dspark-profile-25s-v1)
            selection=(--formal-cost-plan b64-confidence-cost-v1)
            experiment=(); acceptance=(); contexts=(128) ;;
        --shutdown-policy=dspark-profile-25s-v1) extra+=(--profile-shutdown-policy dspark-profile-25s-v1) ;;
        --coverage=b64-functional-1|--coverage=b64-functional-2)
            extra+=(--profile-shutdown-policy dspark-profile-25s-v1)
            selection=(--profile-coverage-phase "${4#--coverage=}") ;;
        --exit-observation) extra+=(--profile-exit-observation) ;;
        --exit-observation-no-debugger) extra+=(--profile-exit-observation --profile-exit-no-debugger) ;;
        *) exit 1 ;;
    esac
fi
bash /workspace/vllm-ascend-hust/tools/dspark/run_dspark_large_batch.sh "$sha" "$manifest" \
    --core-remote "$core_remote" \
    "${acceptance[@]}" \
    --stage profile --batches 64 --num-prompts 400 \
    "${experiment[@]}" \
    "${extra[@]}" \
    "${selection[@]}" \
    --capture-sizes 6 12 24 48 96 192 384 \
    --profile-contexts "${contexts[@]}" --profile-output-tokens 512 \
    --profile-warmup 2 --profile-samples 5 \
    --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9
