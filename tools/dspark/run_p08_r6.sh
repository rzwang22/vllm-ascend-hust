#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Run in the same initialized CANN/custom-op container as P08-R5.
set -u
set -o pipefail

P08_PLUGIN_REPO=/workspace/vllm-ascend-hust
P08_CORE_REPO=/workspace/vllm-hust
P08_CORE_SHA=897306c43bf800e2480cb5c0f3e2da408d85a2fd
P08_BASELINE=15ea9d134b6284dc365dcdda56afa6502161bd04
P08_EXPECTED_PLUGIN_SHA=${1:-}
export VLLM_ALLOW_INSECURE_SERIALIZATION=0
P08_DATA=/workspace/dspark-results/m2_5a-p08-r3-25a1ceba0.7jgir6/input-dataset.jsonl
P08_DATA_SHA=6a2f629a5b5c9bbd9a3058b7a450fc18b2332f4699047f164cdde6a33b58d053
P08_OUT=
P08_STAGE=setup
P08_SOURCE_RC=99
P08_DATASET_RC=99
P08_FOCUSED_RC=99
P08_GRAPH_RC=99
P08_REPLAY_RC=99

p08_source_gate() {
    cd "$P08_PLUGIN_REPO" || return 1
    printf 'EXPECTED_PLUGIN_HEAD=%s\nEXPECTED_CORE_HEAD=%s\n' "$P08_EXPECTED_PLUGIN_SHA" "$P08_CORE_SHA"
    [[ "$P08_EXPECTED_PLUGIN_SHA" =~ ^[0-9a-f]{40}$ ]] || return 1
    test -z "$(git status --porcelain)" || return 1
    git switch feat/dspark || return 1
    git pull --ff-only origin feat/dspark || return 1
    P08_ACTUAL_PLUGIN_SHA=$(git rev-parse HEAD) || return 1
    printf 'ACTUAL_PLUGIN_HEAD=%s\n' "$P08_ACTUAL_PLUGIN_SHA"
    test "$P08_ACTUAL_PLUGIN_SHA" = "$P08_EXPECTED_PLUGIN_SHA" || return 1
    git merge-base --is-ancestor "$P08_BASELINE" HEAD || return 1
    test -z "$(git status --porcelain)" || return 1
    test -z "$(git -C "$P08_CORE_REPO" status --porcelain)" || return 1
    P08_ACTUAL_CORE_SHA=$(git -C "$P08_CORE_REPO" rev-parse HEAD) || return 1
    printf 'ACTUAL_CORE_HEAD=%s\n' "$P08_ACTUAL_CORE_SHA"
    test "$P08_ACTUAL_CORE_SHA" = "$P08_CORE_SHA" || return 1
    git rev-parse HEAD || return 1
    python - "$P08_PLUGIN_REPO" "$P08_CORE_REPO" <<'PY'
import importlib.util
import sys
from pathlib import Path

for name, root in (('vllm_ascend', sys.argv[1]), ('vllm', sys.argv[2])):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.submodule_search_locations, name
    actual = Path(next(iter(spec.submodule_search_locations))).resolve()
    assert actual == (Path(root) / name).resolve(), (name, actual)
    print(name, actual)

import torch
import torch_npu

print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__)
import inspect
import os
from vllm.utils.import_utils import resolve_obj_by_qualname

assert os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] == '0'
worker_extension = resolve_obj_by_qualname(
    'vllm_ascend.diagnostics.dspark_benchmark_worker.DSparkBenchmarkWorkerExtension'
)
extension_file = Path(inspect.getfile(worker_extension)).resolve()
assert extension_file == (Path(sys.argv[1]) / 'vllm_ascend/diagnostics/dspark_benchmark_worker.py').resolve()
print('benchmark worker extension:', extension_file)
print('VLLM_ALLOW_INSECURE_SERIALIZATION=0')
PY
}

p08_main() {
    mkdir -p /workspace/dspark-results || return 1
    P08_OUT=$(mktemp -d /workspace/dspark-results/m2_5a-p08-r6.XXXXXX) || return 1
    cp -- "$0" "$P08_OUT/server-command.sh" || return 1
    printf 'P08_OUT=%s\n' "$P08_OUT"

    P08_STAGE=source
    p08_source_gate 2>&1 | tee "$P08_OUT/source.log"
    P08_PIPE=("${PIPESTATUS[@]}")
    printf '%s\n' "${P08_PIPE[*]}" > "$P08_OUT/source-pipestatus.txt"
    P08_SOURCE_RC=${P08_PIPE[0]}
    test "${P08_PIPE[1]}" -eq 0 || return 1
    test "$P08_SOURCE_RC" -eq 0 || return "$P08_SOURCE_RC"
    P08_PLUGIN_SHA=$(git -C "$P08_PLUGIN_REPO" rev-parse HEAD) || return 1
    printf 'PLUGIN_HEAD=%s\nCORE_HEAD=%s\n' "$P08_PLUGIN_SHA" "$P08_CORE_SHA" > "$P08_OUT/source.txt"

    P08_STAGE=dataset
    printf '%s  %s\n' "$P08_DATA_SHA" "$P08_DATA" | sha256sum -c - 2>&1 | tee "$P08_OUT/dataset.log"
    P08_PIPE=("${PIPESTATUS[@]}")
    printf '%s\n' "${P08_PIPE[*]}" > "$P08_OUT/dataset-pipestatus.txt"
    P08_DATASET_RC=${P08_PIPE[1]}
    test "${P08_PIPE[0]}" -eq 0 && test "${P08_PIPE[2]}" -eq 0 || return 1
    test "$P08_DATASET_RC" -eq 0 || return "$P08_DATASET_RC"
    cp -- "$P08_DATA" "$P08_OUT/input-dataset.jsonl" || return 1

    export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=0
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ASCEND_LAUNCH_BLOCKING=0
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    unset RANK LOCAL_RANK WORLD_SIZE GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT

    P08_STAGE=focused
    cd "$P08_PLUGIN_REPO" || return 1
    python -m pytest -q -ra \
        tests/ut/test_dspark_graph_rpc.py \
        tests/ut/attention/test_dsa_capture_validation.py \
        tests/ut/attention/test_dsa_padding_contract.py \
        tests/ut/attention/test_dsa_sharedkv_contract.py \
        tests/ut/worker/test_dsa_capture_metadata.py \
        tests/ut/worker/test_dsa_model_runner_v2_metadata.py \
        tests/ut/test_ascend_forward_context_v2_dsa.py \
        tests/ut/worker/test_aclgraph_capture.py \
        tests/ut/test_dspark_draft_config.py \
        tests/ut/test_dspark_acceptance_benchmark.py \
        tests/ut/spec_decode/test_dspark*.py 2>&1 | tee "$P08_OUT/focused.log"
    P08_PIPE=("${PIPESTATUS[@]}")
    printf '%s\n' "${P08_PIPE[*]}" > "$P08_OUT/focused-pipestatus.txt"
    P08_FOCUSED_RC=${P08_PIPE[0]}
    test "${P08_PIPE[1]}" -eq 0 || return 1
    test "$P08_FOCUSED_RC" -eq 0 || return "$P08_FOCUSED_RC"

    P08_STAGE=graph_p1
    cd /workspace || return 1
    ps -eo pid,ppid,args > "$P08_OUT/process-before.log"
    npu-smi info > "$P08_OUT/npu-before.log" 2>&1 || return 1
    VLLM_PORT=29866 MASTER_PORT=29866 python "$P08_PLUGIN_REPO/tools/dspark/benchmark_dspark_acceptance.py" \
        --model-dir /workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 \
        --revision 9e8679a9db7eec11efed9925f7efb96549077545 \
        --mode dspark --num-spec-tokens 5 \
        --dataset-name jsonl --dataset-path "$P08_OUT/input-dataset.jsonl" --prompt-field prompt_token_ids \
        --num-prompts 1 --warmup-prompts 1 --output-len 16 --ignore-eos \
        --tensor-parallel-size 8 --enable-expert-parallel --async-scheduling \
        --max-num-seqs 1 --max-model-len 8192 --max-num-batched-tokens 8192 \
        --block-size 32 --gpu-memory-utilization 0.9 \
        --dtype bfloat16 --quantization ascend --tokenizer-mode deepseek_v4 \
        --seed 0 --temperature 0 --top-p 1 --top-k -1 \
        --target-execution-mode full_decode_only --cudagraph-capture-sizes 6 \
        --result-json "$P08_OUT/graph-p1.json" 2>&1 | tee "$P08_OUT/graph-p1.log"
    P08_PIPE=("${PIPESTATUS[@]}")
    printf '%s\n' "${P08_PIPE[*]}" > "$P08_OUT/graph-pipestatus.txt"
    P08_GRAPH_RC=${P08_PIPE[0]}
    test "${P08_PIPE[1]}" -eq 0 || return 1
    test "$P08_GRAPH_RC" -eq 0 || return "$P08_GRAPH_RC"

    P08_STAGE=measured_replay
    python - "$P08_OUT/graph-p1.json" "$P08_PLUGIN_SHA" "$P08_CORE_SHA" <<'PY' 2>&1 | tee "$P08_OUT/replay.log"
import json
import sys

r = json.load(open(sys.argv[1]))
assert r['plugin_sha'] == sys.argv[2] and r['core_sha'] == sys.argv[3]
assert r['target_enforce_eager'] is False and r['dspark_enforce_eager'] is True
assert r['cudagraph_mode_effective'] == 'FULL_DECODE_ONLY'
assert r['graph_capture_count'] > 0 and r['observed_capture_sizes'] == [6]
assert r['measured_graph_replay_count'] > 0
assert any(x['runtime_mode'] == 'FULL' and x['num_unpadded_tokens'] == 6
           and x['num_padded_tokens'] == 6
           for x in r['graph_execution']['measured_runtime']['records'])
assert r['acceptance']['num_drafts'] > 0
assert r['cleanup']['engine_shutdown_complete'] is True
print('Graph P1 measured verification FULL replay passed')
PY
    P08_PIPE=("${PIPESTATUS[@]}")
    printf '%s\n' "${P08_PIPE[*]}" > "$P08_OUT/replay-pipestatus.txt"
    P08_REPLAY_RC=${P08_PIPE[0]}
    test "${P08_PIPE[1]}" -eq 0 || return 1
    test "$P08_REPLAY_RC" -eq 0 || return "$P08_REPLAY_RC"
    P08_STAGE=complete
}

p08_main
P08_MAIN_RC=$?
if test -n "$P08_OUT"; then
    ps -eo pid,ppid,args > "$P08_OUT/process-after.log"
    npu-smi info > "$P08_OUT/npu-after.log" 2>&1
    P08_NPU_RC=$?
    printf 'LAST_STAGE=%s\nSOURCE_GATE_RC=%s\nDATASET_GATE_RC=%s\nFOCUSED_RC=%s\nGRAPH_P1_RC=%s\nREPLAY_GATE_RC=%s\nNPU_SMI_RC=%s\nP08_MAIN_RC=%s\nP08_OUT=%s\n' \
        "$P08_STAGE" "$P08_SOURCE_RC" "$P08_DATASET_RC" "$P08_FOCUSED_RC" "$P08_GRAPH_RC" "$P08_REPLAY_RC" \
        "$P08_NPU_RC" "$P08_MAIN_RC" "$P08_OUT" | tee "$P08_OUT/gate.txt"
fi
test "$P08_MAIN_RC" -eq 0
