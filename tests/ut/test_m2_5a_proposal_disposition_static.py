# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "vllm_ascend/worker/v2/model_runner.py"
SPECULATOR = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
LIFECYCLE = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/proposal_inputs.py"
HARNESS = ROOT / ("tests/e2e/nightly/single_node/spec_decode/test_dspark_single_request_realdata.py")


def test_runner_reconciles_scheduler_truth_before_base_request_updates() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    finish = source[source.index("    def finish_requests(") : source.index("    def get_kv_cache_spec(")]

    assert "speculator.reconcile_scheduler_proposal(" in finish
    assert "scheduler_output.scheduled_spec_decode_tokens" in finish
    assert "scheduler_output.num_scheduled_tokens" in finish
    assert "scheduler_output.finished_req_ids" in finish
    assert "scheduler_output.preempted_req_ids" in finish
    assert finish.index("speculator.reconcile_scheduler_proposal(") < finish.index(
        "super().finish_requests(scheduler_output)"
    )


def test_disposition_tracks_real_lengths_without_mutating_input_batch() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    reconcile = source[
        source.index("    def reconcile_scheduler_proposal(") : source.index(
            "    def _consume_published_proposal_after_verification("
        )
    ]

    assert "scheduled_lengths_by_request" in reconcile
    assert 'disposition = "INSTALLED"' in reconcile
    assert 'else "TRUNCATED"' in reconcile
    assert 'reason="scheduled_without_proposal"' in reconcile
    assert 'reason="preempted"' in reconcile
    assert "unresolved_owners" in reconcile
    assert "delayed_owners" in reconcile
    assert "_defer_published_proposal_rows(delayed_owners)" in reconcile
    assert "conflicting proposal owner dispositions" in reconcile
    assert "num_draft_tokens_per_req" not in reconcile
    assert "input_batch" not in reconcile


def test_verification_compares_only_the_scheduled_candidate_prefix() -> None:
    source = SPECULATOR.read_text(encoding="utf-8")
    consume = source[
        source.index("    def _consume_published_proposal_after_verification(") : source.index(
            "    def _skip_next_proposal_after_verification("
        )
    ]

    assert "if length > 0" in consume
    assert "verification_batch_rows" in consume
    assert "input_request_ids[row]" in consume
    assert "verification_candidate_tokens[:, :max_scheduled_length][scheduled_mask]" in consume
    assert "sum(verification_scheduled_lengths)" in consume
    assert "expected_query_lengths = scheduled_lengths_tensor + 1" in consume
    assert "published candidate set prefix scheduled by core" in consume
    assert "self.num_speculative_steps" not in consume


def test_scheduler_spec_tokens_are_the_proposal_installation_truth() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    speculator = SPECULATOR.read_text(encoding="utf-8")
    finish = runner[runner.index("    def finish_requests(") : runner.index("    def get_kv_cache_spec(")]
    reconcile = speculator[
        speculator.index("    def reconcile_scheduler_proposal(") : speculator.index(
            "    def _consume_published_proposal_after_verification("
        )
    ]

    assert "scheduled_spec_decode_tokens=(scheduler_output.scheduled_spec_decode_tokens)" in finish
    assert "set(scheduler_output.num_scheduled_tokens)" in finish
    assert "scheduled_spec_owners = set(scheduled_spec_decode_tokens)" in reconcile
    assert "scheduled_owners = owners.intersection(scheduled_spec_owners)" in reconcile
    assert "scheduled_spec_owners.difference(owners)" in reconcile
    assert "scheduled_lengths_by_request" in reconcile


def test_lifecycle_and_diagnostic_expose_terminal_disposition_fields() -> None:
    lifecycle = LIFECYCLE.read_text(encoding="utf-8")
    speculator = SPECULATOR.read_text(encoding="utf-8")
    logging_method = speculator[
        speculator.index("    def _log_proposal_disposition(") : speculator.index(
            "    def _clear_published_proposal_state("
        )
    ]

    for field in (
        "scheduled_lengths",
        "disposition",
        "token_prefix_match",
        "truncated",
        "dropped",
        "drop_reason",
    ):
        assert field in lifecycle
    assert "DSPARK_PROPOSAL_DISPOSITION=" in speculator
    assert "if not logger.isEnabledFor(logging.DEBUG):" in logging_method
    assert logging_method.index("if not logger.isEnabledFor(logging.DEBUG):") < logging_method.index(
        "published_length ="
    )
    assert "logger.debug(" in logging_method
    assert "logger.info(" not in logging_method
    for field in (
        '"rank"',
        '"request_id"',
        '"producer_epoch"',
        '"consumer_epoch"',
        '"published_length"',
        '"scheduled_length"',
        '"token_prefix_match"',
        '"consumed"',
    ):
        assert field in speculator


def test_realdata_harness_checks_reconciliation_before_sampling() -> None:
    source = HARNESS.read_text(encoding="utf-8")

    assert "def _assert_scheduler_proposal_disposition(" in source
    assert "proposal_pending_before_execute" in source
    assert "scheduler_output.scheduled_spec_decode_tokens" in source
    assert "_assert_scheduler_proposal_disposition(" in source
    assert "An uninstalled DSpark proposal was not atomically retired" in source
    assert source.index("_assert_scheduler_proposal_disposition(") < source.index(
        "async_output = runtime.worker.sample_tokens(None)"
    )
