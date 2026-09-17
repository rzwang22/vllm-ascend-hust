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
    if test "${codes[0]}" -ne 0; then return "${codes[0]}"; fi
    return "${codes[1]}"
}

main() {
    test "$#" -ge 2 || return 1
    local sha=$1 manifest=$2
    shift 2
    local plugin=/workspace/vllm-ascend-hust core=/workspace/vllm-hust
    local core_remote=origin acceptance_archive='' forwarded=()
    while test "$#" -gt 0; do
        case "$1" in
            --core-remote) test "$#" -ge 2 || return 1; core_remote=$2; shift 2 ;;
            --swa-acceptance-archive) test "$#" -ge 2 || return 1; acceptance_archive=$2; shift 2 ;;
            *) forwarded+=("$1"); shift ;;
        esac
    done
    set -- "${forwarded[@]}"
    local model=/workspace/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8 previous='' argument experiment='' writer=false exit_observation=false no_debugger=false shutdown_policy='' coverage_phase='' formal_cost='' confidence_acceptance=false expansion=false
    for argument in "$@"; do
        if test "$argument" = --batch-expansion; then expansion=true; fi
        if test "$argument" = --confidence-acceptance; then confidence_acceptance=true; fi
        if test "$previous" = --formal-cost-plan; then formal_cost=$argument; fi
        case "$argument" in --formal-cost-plan=*) formal_cost=${argument#--formal-cost-plan=} ;; esac
        if test "$previous" = --profile-coverage-phase; then coverage_phase=$argument; fi
        case "$argument" in --profile-coverage-phase=*) coverage_phase=${argument#--profile-coverage-phase=} ;; esac
        if test "$previous" = --profile-shutdown-policy; then shutdown_policy=$argument; fi
        case "$argument" in --profile-shutdown-policy=*) shutdown_policy=${argument#--profile-shutdown-policy=} ;; esac
        if test "$previous" = --model; then model=$argument; fi
        if test "$previous" = --profile-experiment; then experiment=$argument; fi
        case "$argument" in --model=*) model=${argument#--model=} ;; esac
        case "$argument" in --profile-experiment=*) experiment=${argument#--profile-experiment=} ;; esac
        if test "$argument" = --profile-write-timeline; then writer=true; fi
        if test "$argument" = --profile-exit-observation; then exit_observation=true; fi
        if test "$argument" = --profile-exit-no-debugger; then no_debugger=true; fi
        previous=$argument
    done
    mkdir -p /workspace/dspark-results || return 1
    CONF_OUT=$(mktemp -d /workspace/dspark-results/dspark-large-batch.XXXXXXXX) || return 1
    printf 'SERVER_RESULT_DIR=%s\n' "$CONF_OUT"
    cd "$plugin" || return 1
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || return 1
    test "$(git -C "$plugin" rev-parse HEAD)" = "$sha" || return 1
    logged core-source python -m tools.dspark.swa_acceptance core-source \
        "$core" "$core_remote" "$CONF_OUT/core-source.json" || return "$?"
    test "$(git -C "$core" rev-parse HEAD)" = 71d2c1c436eba894a8e9eeb2c5af17e05cb42970 || return 1
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
    logged source python tools/dspark/p08_r8_checks.py source "$plugin" "$core" || return "$?"
    if test "$expansion" = true; then
        test "$#" -eq 1 || return 1
        logged batch-expansion timeout --signal=TERM --kill-after=65s 36000s python -m tools.dspark.batch_expansion run \
            --plugin-sha "$sha" --manifest "$manifest" --output-dir "$CONF_OUT"
        return "$?"
    fi
    if test "$confidence_acceptance" = true; then
        test "$#" -eq 1 || return 1
        logged confidence-host-tests python -m pytest --noconftest -q -ra tests/ut/test_dspark_confidence_acceptance.py \
            --junitxml "$CONF_OUT/confidence-host-tests.xml" || return "$?"
        logged confidence-host-check python - "$CONF_OUT/confidence-host-tests.xml" <<'PYTEST'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert cases and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print(f'Confidence host tests: {len(cases)} passed; no model initialized')
PYTEST
        test "$?" -eq 0 || return 1
        logged confidence-preflight timeout --signal=TERM --kill-after=15s 1800s python -m tools.dspark.confidence_acceptance prepare \
            --plugin-sha "$sha" --manifest "$manifest" --output-dir "$CONF_OUT" || return "$?"
        logged confidence-run python -m tools.dspark.confidence_acceptance run \
            --plugin-sha "$sha" --manifest "$manifest" --output-dir "$CONF_OUT"
        return "$?"
    fi
    if test -n "$coverage_phase"; then
        local baseline_archive=/workspace/dspark-results/dspark-large-batch.Ck6iA7rN-evidence.tar.gz
        if test "$coverage_phase" = b64-functional-2; then
            baseline_archive=/workspace/dspark-results/dspark-large-batch.XM02ngWZ-evidence.tar.gz
        fi
        logged coverage-plan python -m tools.dspark.functional_coverage "$coverage_phase" \
            "$CONF_OUT/coverage-plan.json" \
            --baseline-archive "$baseline_archive" || return "$?"
    fi
    if test -n "$formal_cost"; then
        test "$formal_cost" = b64-confidence-cost-v1 && test -z "$acceptance_archive" || return 1
        logged formal-preflight timeout --signal=TERM --kill-after=15s 1800s python -m tools.dspark.formal_cost prepare \
            "$model" /workspace/dspark-results/dspark-large-batch.v8vohAeE-evidence.tar.gz \
            "$CONF_OUT/formal-cost-preflight.json" "$sha" "$manifest" || return "$?"
    fi
    logged checkpoint python tools/dspark/verification_tools.py checkpoint \
        --model "$model" \
        --output "$CONF_OUT/checkpoint.json" || return "$?"
    if test "$experiment" = target-boundaries; then
        logged frozen-inputs python tools/dspark/check_target_profile_inputs.py \
            "$CONF_OUT/checkpoint.json" "$manifest" || return "$?"
    fi
    if test "$writer" = true; then
        logged writer-preflight timeout --signal=TERM --kill-after=15s 300s python -m pytest -q -ra \
            tests/ut/test_dspark_write_timeline.py::test_npu_writer_probe_inside_opaque_dispatch_replays \
            tests/ut/test_dspark_write_timeline.py::test_installed_core_page_removal_uses_actual_computed \
            --basetemp "$CONF_OUT/writer-preflight" --junitxml "$CONF_OUT/writer-preflight.xml" || return "$?"
        logged writer-acceptance python - "$CONF_OUT/writer-preflight.xml" <<'PYTEST'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert len(cases) == 2 and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print('Writer probe: real opaque NPU replay and installed Core forwarding passed')
PYTEST
        test "$?" -eq 0 || return 1
    fi
    if test -n "$formal_cost"; then
        logged formal-host-tests python -m pytest --noconftest -q -ra tests/ut/test_dspark_formal_cost.py \
            --junitxml "$CONF_OUT/formal-host-tests.xml" || return "$?"
        logged formal-host-check python - "$CONF_OUT/formal-host-tests.xml" <<'PYTEST'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert cases and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print(f'Formal cost host tests: {len(cases)} passed, zero skips; no NPU model initialized')
PYTEST
        test "$?" -eq 0 || return 1
    elif test -n "$acceptance_archive"; then
        test "$experiment" = target-boundaries && test "$writer" = false || return 1
        [[ " $* " != *" --profile-operator-capture "* ]] || return 1
        if test -z "$coverage_phase"; then
            logged local-validation python -m tools.dspark.swa_acceptance audit-local \
                "$acceptance_archive" "$CONF_OUT/local-validation.json" || return "$?"
        fi
        if test -n "$shutdown_policy"; then
            test "$shutdown_policy" = dspark-profile-25s-v1 && test "$exit_observation" = false || return 1
            local policy_tests=tests/ut/test_dspark_shutdown_policy.py
            if test -n "$coverage_phase"; then policy_tests=tests/ut/test_dspark_functional_coverage.py; fi
            if test "$coverage_phase" = b64-functional-2; then policy_tests=tests/ut/test_dspark_functional_phase2.py; fi
            logged shutdown-policy-tests python -m pytest --noconftest -q -ra \
                "$policy_tests" \
                --basetemp "$CONF_OUT/policy-tests" --junitxml "$CONF_OUT/shutdown-policy.xml" || return "$?"
            logged shutdown-policy-test-check python - "$CONF_OUT/shutdown-policy.xml" <<'PYTEST'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert cases and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print(f'Shutdown policy host tests: {len(cases)} passed; zero failures/skips; no model initialization')
PYTEST
            test "$?" -eq 0 || return 1
        elif test "$exit_observation" = true; then
            local exit_tests=tests/ut/test_dspark_exit_observation.py
            if test "$no_debugger" = true; then exit_tests=tests/ut/test_dspark_exit_no_debugger.py; fi
            logged exit-observation-tests python -m pytest --noconftest -q -ra \
                "$exit_tests" \
                --basetemp "$CONF_OUT/exit-tests" --junitxml "$CONF_OUT/exit-observation.xml" || return "$?"
            logged exit-observation-test-check python - "$CONF_OUT/exit-observation.xml" <<'PYTEST'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert cases and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print(f'Exit observation host tests: {len(cases)} passed; zero failures/skips')
PYTEST
            test "$?" -eq 0 || return 1
            if test "$no_debugger" = true; then
                logged native-disabled python - "$CONF_OUT/native-preflight-disabled.json" <<'PYNOATTACH'
import json
import sys
from vllm import envs
assert envs.VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS == 5
with open(sys.argv[1], 'w') as out:
    json.dump(dict(debugger_enabled=False, attachment_count=0, native_sampling='disabled_by_configuration',
                   attach_preflight='not_run_by_configuration', performance_eligible=False), out)
PYNOATTACH
                test "$?" -eq 0 || return 1
            else
            logged native-preflight timeout --signal=TERM --kill-after=2s 30s python -m tools.dspark.exit_native_preflight \
                "$CONF_OUT/native-preflight" || return "$?"
            fi
        else
        # The archived 22/3/23 tests already passed; validate entry and host teardown.
        logged acceptance-entry python -m pytest --noconftest -q -ra \
            tests/ut/test_dspark_swa_acceptance.py tests/ut/test_dspark_profile_teardown.py \
            tests/ut/test_dspark_worker_exit.py tests/ut/test_dspark_profile_failure.py \
            tests/ut/test_dspark_post_shutdown.py \
            --basetemp "$CONF_OUT/entry-tests" --junitxml "$CONF_OUT/acceptance-entry.xml" || return "$?"
        logged acceptance-entry-check python - "$CONF_OUT/acceptance-entry.xml" <<'PYTEST'
import sys
import xml.etree.ElementTree as ET
cases = ET.parse(sys.argv[1]).getroot().findall('.//testcase')
assert cases and not any(c.find(k) is not None for c in cases for k in ('failure', 'error', 'skipped'))
print(f'Host entry/teardown: {len(cases)} passed, zero failures/skips; no NPU or model verification claimed')
PYTEST
        test "$?" -eq 0 || return 1
        fi
    else
    logged focused python -m pytest -q -ra \
        tests/ut/test_dspark_repeated_inputs.py tests/ut/test_dspark_startup_cost_profile.py \
        tests/ut/test_dspark_profile_request_ids.py tests/ut/test_dspark_profile_context.py \
        tests/ut/test_dspark_profile_nan.py tests/ut/test_dspark_profile_observation.py \
        tests/ut/test_dspark_profile_numerics.py tests/ut/test_dspark_profile_upstream.py \
        tests/ut/test_dspark_profile_auxiliary.py tests/ut/test_dspark_profile_target.py \
        tests/ut/test_dspark_attention_receipts.py \
        tests/ut/test_dspark_attention_validity.py \
        tests/ut/test_dspark_profile_attention.py \
        tests/ut/test_dspark_profile_kv.py tests/ut/test_dspark_operator_capture.py tests/ut/test_dspark_operator_npu.py \
        tests/ut/test_dspark_operator_serialization.py tests/ut/test_dspark_write_timeline.py \
        tests/ut/test_dspark_replay_diagnostics.py \
        tests/ut/test_dspark_nan_diagnostics.py tests/ut/test_dspark_profile_failure.py tests/ut/test_dspark_worker_exit.py \
        tests/ut/test_dspark_post_shutdown.py \
        tests/ut/test_dspark_confidence_verification.py tests/ut/worker/test_dsa_padded_requests.py \
        tests/ut/worker/test_capture_input_aliases.py tests/ut/worker/test_dsa_capture_metadata.py \
        tests/ut/attention/test_dsa_padding_contract.py tests/ut/attention/test_dsa_capture_validation.py \
        tests/ut/worker/test_aclgraph_capture.py tests/ut/test_dspark_draft_config.py \
        tests/ut/test_dspark_graph_rpc.py tests/ut/test_dspark_graph_replay.py \
        tests/ut/test_dspark_acceptance_benchmark.py tests/ut/test_dspark_performance_delivery.py \
        tests/ut/spec_decode/test_dspark_v2_*.py || return "$?"
    fi
    if test "$experiment" = target-boundaries && test -f "$CONF_OUT/STOP"; then return 130; fi
    logged generation python tools/dspark/run_large_batch.py \
        --plugin-sha "$sha" --manifest "$manifest" --output-dir "$CONF_OUT/runs" "$@"
    local generation_rc=$? report_rc=0 observation_rc=0
    if test -n "$acceptance_archive"; then
        logged acceptance-report python -m tools.dspark.swa_acceptance report \
            "$CONF_OUT/runs/b64" "$generation_rc" "$CONF_OUT/model-acceptance.json"
        report_rc=$?
        if test "$exit_observation" = true; then
            logged exit-observation-report python -m tools.dspark.exit_observation_report \
                "$CONF_OUT"
            observation_rc=$?
        fi
    fi
    if test -n "$formal_cost"; then
        logged cost-publication timeout --signal=TERM --kill-after=15s 1800s python -m tools.dspark.formal_cost publish \
            "$CONF_OUT/runs/b64" "$generation_rc" "$sha"
        report_rc=$?
    fi
    if test "$generation_rc" -ne 0; then return "$generation_rc"; fi
    if test "$report_rc" -ne 0; then return "$report_rc"; fi
    return "$observation_rc"
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
# Export errors never replace the first failed phase's exit status.
if test "$CONF_RC" -ne 0; then exit "$CONF_RC"; fi
if test "$CONF_ARCHIVE_RC" -ne 0; then exit "$CONF_ARCHIVE_RC"; fi
if test "${CONF_HASH_CODES[0]}" -ne 0; then exit "${CONF_HASH_CODES[0]}"; fi
exit "${CONF_HASH_CODES[1]}"
