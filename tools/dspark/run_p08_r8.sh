#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# One Graph B4 diagnostic only. Server/NPU execution is owned by the user.
set -o pipefail

P08_PLUGIN=/workspace/vllm-ascend-hust
P08_CORE=/workspace/vllm-hust
P08_CORE_SHA=897306c43bf800e2480cb5c0f3e2da408d85a2fd
P08_BASELINE=9b40a187b070cbe0efa32f91cffa9889a7aaf51d
P08_PLUGIN_SHA=${1:-}
P08_DATA=/workspace/dspark-results/m2_5a-p08-r7-smallbatch.3AgHPg/input-dataset.jsonl
P08_DATA_SHA=6a2f629a5b5c9bbd9a3058b7a450fc18b2332f4699047f164cdde6a33b58d053
P08_CHECKS=$P08_PLUGIN/tools/dspark/p08_r8_checks.py
P08_PORT=29888
P08_OUT=
P08_STAGE=setup
P08_SOURCE_RC=99 P08_DATA_RC=99 P08_FOCUSED_RC=99 P08_GRAPH_RC=99 P08_RESULT_RC=99
P08_DIAG_RC=99 P08_ERRORS_RC=99 P08_RESIDUAL_RC=99 P08_NPU_RC=99
P08_ARCHIVE_RC=99 P08_SHA_RC=99

# Every tee pipeline captures all PIPESTATUS entries before any other command.
p08_logged() {
    local name=$1
    shift
    "$@" 2>&1 | tee "$P08_OUT/$name.log"
    local codes=("${PIPESTATUS[@]}")
    printf '%s\n' "${codes[*]}" > "$P08_OUT/$name-pipestatus.txt" || return 1
    test "${codes[1]}" -eq 0 || return 1
    return "${codes[0]}"
}

p08_source() {
    [[ "$P08_PLUGIN_SHA" =~ ^[0-9a-f]{40}$ ]] || return 1
    test "$(git -C "$P08_PLUGIN" rev-parse HEAD)" = "$P08_PLUGIN_SHA" || return 1
    test "$(git -C "$P08_CORE" rev-parse HEAD)" = "$P08_CORE_SHA" || return 1
    test "$(git -C "$P08_PLUGIN" branch --show-current)" = feat/dspark || return 1
    test -z "$(git -C "$P08_PLUGIN" status --porcelain)" || return 1
    test -z "$(git -C "$P08_CORE" status --porcelain)" || return 1
    git -C "$P08_PLUGIN" merge-base --is-ancestor "$P08_BASELINE" HEAD || return 1
    printf 'PLUGIN_HEAD=%s\nCORE_HEAD=%s\n' "$P08_PLUGIN_SHA" "$P08_CORE_SHA"
    local saved_opp=${ASCEND_CUSTOM_OPP_PATH:-}
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh || return 1
    if test -n "$saved_opp"; then
        export ASCEND_CUSTOM_OPP_PATH="$saved_opp"
    else
        # shellcheck disable=SC1091
        source "$P08_PLUGIN/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash" || return 1
    fi
    export PYTHONPATH="$P08_PLUGIN:$P08_CORE:${PYTHONPATH:-}"
    export VLLM_ALLOW_INSECURE_SERIALIZATION=0
    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_LAUNCH_BLOCKING=0
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
    python "$P08_CHECKS" source "$P08_PLUGIN" "$P08_CORE"
}

p08_dataset() {
    python - "$P08_DATA" "$P08_DATA_SHA" "$P08_OUT/input-dataset.jsonl" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(data).hexdigest() == sys.argv[2]
rows = [json.loads(line) for line in data.splitlines()]
assert len(rows) == 64
assert all(isinstance(row['prompt_token_ids'], list) and row['prompt_token_ids'] for row in rows[:4])
Path(sys.argv[3]).write_bytes(data)
print('INPUT_SHA256', sys.argv[2], 'ROWS', len(rows))
PY
}

p08_graph() {
    cd /workspace || return 1
    VLLM_PORT="$P08_PORT" MASTER_PORT="$P08_PORT" python "$P08_PLUGIN/tools/dspark/benchmark_dspark_acceptance.py" \
        --model-dir /workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
        --revision 9e8679a9db7eec11efed9925f7efb96549077545 \
        --mode dspark --num-spec-tokens 5 \
        --dataset-name jsonl --dataset-path "$P08_OUT/input-dataset.jsonl" --prompt-field prompt_token_ids \
        --num-prompts 4 --warmup-prompts 4 --output-len 256 --ignore-eos \
        --tensor-parallel-size 8 --enable-expert-parallel --async-scheduling \
        --max-num-seqs 4 --max-model-len 8192 --max-num-batched-tokens 8192 \
        --block-size 32 --gpu-memory-utilization 0.9 \
        --dtype bfloat16 --quantization ascend --tokenizer-mode deepseek_v4 \
        --seed 0 --temperature 0 --top-p 1 --top-k -1 \
        --target-execution-mode full_decode_only --cudagraph-capture-sizes 6 12 18 24 \
        --dspark-nan-diagnostic-dir "$P08_OUT/rank-diagnostics" --result-json "$P08_OUT/graph-b4-diagnostic.json"
}

p08_main() {
    mkdir -p /workspace/dspark-results || return 1
    P08_OUT=$(mktemp -d /workspace/dspark-results/m2_5a-p08-r8-diagnostic.XXXXXX) || return 1
    cp -- "$0" "$P08_OUT/server-command.sh" || return 1
    printf 'P08_OUT=%s\n' "$P08_OUT"
    P08_STAGE=source
    # A pipeline would lose the sourced CANN environment in a subshell.
    p08_source > "$P08_OUT/source.log" 2>&1
    P08_SOURCE_RC=$?
    cat "$P08_OUT/source.log"
    test "$P08_SOURCE_RC" -eq 0 || return "$P08_SOURCE_RC"
    cp -- "$P08_CHECKS" "$P08_OUT/p08_r8_checks.py" || return 1
    P08_STAGE=dataset
    p08_logged dataset p08_dataset
    P08_DATA_RC=$?
    test "$P08_DATA_RC" -eq 0 || return "$P08_DATA_RC"
    P08_STAGE=focused
    cd "$P08_PLUGIN" || return 1
    p08_logged focused python -m pytest -q -ra \
        tests/ut/test_dspark_nan_diagnostics.py tests/ut/test_p08_r8_runner.py \
        tests/ut/test_dspark_graph_rpc.py tests/ut/test_dspark_graph_replay.py \
        tests/ut/test_dspark_acceptance_benchmark.py tests/ut/test_dspark_draft_config.py \
        tests/ut/worker/test_aclgraph_capture.py tests/ut/worker/test_dsa_capture_metadata.py \
        tests/ut/attention/test_dsa_capture_validation.py tests/ut/attention/test_dsa_padding_contract.py \
        tests/ut/spec_decode/test_dspark_v2_*.py
    P08_FOCUSED_RC=$?
    test "$P08_FOCUSED_RC" -eq 0 || return "$P08_FOCUSED_RC"
    P08_STAGE=idle_before
    p08_logged process-before python "$P08_CHECKS" idle || return 1
    p08_logged port python "$P08_CHECKS" port "$P08_PORT" || return 1
    p08_logged npu-before npu-smi info || return 1
    P08_STAGE=graph_b4_diagnostic
    p08_logged graph-b4 p08_graph
    P08_GRAPH_RC=$?
    test "$P08_GRAPH_RC" -eq 0 || return "$P08_GRAPH_RC"
    P08_STAGE=result
    p08_logged result python "$P08_CHECKS" result "$P08_OUT/graph-b4-diagnostic.json" "$P08_PLUGIN_SHA"
    P08_RESULT_RC=$?
    test "$P08_RESULT_RC" -eq 0 || return "$P08_RESULT_RC"
    P08_STAGE=diagnostic_generation_complete
}

p08_main
P08_MAIN_RC=$?
if test -n "$P08_OUT"; then
    # Always preserve diagnostics and cleanup evidence, including failed warmup.
    p08_logged diagnostics python "$P08_CHECKS" diagnostics "$P08_OUT"
    P08_DIAG_RC=$?
    if test -f "$P08_OUT/graph-b4.log"; then
        p08_logged errors python "$P08_CHECKS" scan "$P08_OUT/graph-b4.log"
        P08_ERRORS_RC=$?
    fi
    p08_logged process-after python "$P08_CHECKS" idle
    P08_RESIDUAL_RC=$?
    p08_logged npu-after npu-smi info
    P08_NPU_RC=$?
    printf 'STATUS=ROOT_CAUSE_NOT_YET_PROVEN\nPERFORMANCE_ELIGIBLE=false\nLAST_STAGE=%s\nSOURCE_RC=%s\nDATASET_RC=%s\nFOCUSED_RC=%s\nGRAPH_B4_RC=%s\nRESULT_GATE_RC=%s\nDIAGNOSTIC_EVIDENCE_RC=%s\nERROR_SCAN_RC=%s\nRESIDUAL_GATE_RC=%s\nNPU_RC=%s\nMAIN_RC=%s\n' \
        "$P08_STAGE" "$P08_SOURCE_RC" "$P08_DATA_RC" "$P08_FOCUSED_RC" "$P08_GRAPH_RC" "$P08_RESULT_RC" \
        "$P08_DIAG_RC" "$P08_ERRORS_RC" "$P08_RESIDUAL_RC" "$P08_NPU_RC" "$P08_MAIN_RC" > "$P08_OUT/gate.txt"
    cat "$P08_OUT/gate.txt"
    tar -czf "$P08_OUT-evidence.tar.gz" -C "$(dirname "$P08_OUT")" "$(basename "$P08_OUT")"
    P08_ARCHIVE_RC=$?
    if test "$P08_ARCHIVE_RC" -eq 0; then
        sha256sum "$P08_OUT-evidence.tar.gz" | tee "$P08_OUT-evidence.sha256"
        P08_PIPE=("${PIPESTATUS[@]}")
        printf '%s\n' "${P08_PIPE[*]}" > "$P08_OUT-evidence.sha256-pipestatus.txt"
        P08_SHA_RC=${P08_PIPE[0]}
        test "${P08_PIPE[1]}" -eq 0 || P08_SHA_RC=1
    fi
    printf 'ARCHIVE_RC=%s\nARCHIVE_SHA_RC=%s\nEVIDENCE=%s-evidence.tar.gz\n' "$P08_ARCHIVE_RC" "$P08_SHA_RC" "$P08_OUT"
fi
test "$P08_MAIN_RC" -eq 0 && test "$P08_DIAG_RC" -eq 0 && test "$P08_ERRORS_RC" -eq 0 &&
    test "$P08_RESIDUAL_RC" -eq 0 && test "$P08_NPU_RC" -eq 0 &&
    test "$P08_ARCHIVE_RC" -eq 0 && test "$P08_SHA_RC" -eq 0
