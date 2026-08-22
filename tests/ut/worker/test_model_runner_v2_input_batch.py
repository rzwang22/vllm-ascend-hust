# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from inspect import Parameter, signature
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2 import model_runner as ascend_model_runner
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class _MirroredTensor:
    def __init__(self, values: list[int]) -> None:
        self.gpu = torch.tensor(values, dtype=torch.int32)
        self.np = self.gpu.numpy()


class _RequestStates:
    def __init__(
        self,
        request_ids: list[str],
        prompt_lens: list[int],
        computed_tokens: list[int],
        computed_prefill_tokens: list[int],
    ) -> None:
        max_num_reqs = len(request_ids)
        self.req_id_to_index = {request_id: index for index, request_id in enumerate(request_ids)}
        self.prompt_len = _MirroredTensor(prompt_lens)
        self.prefill_len = _MirroredTensor(prompt_lens)
        self.num_computed_prefill_tokens = np.array(
            computed_prefill_tokens,
            dtype=np.int32,
        )
        self.num_computed_tokens_np = np.array(computed_tokens, dtype=np.int32)
        self.num_computed_tokens = SimpleNamespace(
            gpu=torch.tensor(computed_tokens, dtype=torch.int32),
        )
        self.next_prefill_tokens = torch.zeros(max_num_reqs, dtype=torch.int32)
        self.all_token_ids = SimpleNamespace(
            gpu=torch.zeros((max_num_reqs, 32), dtype=torch.int32),
        )
        self.last_sampled_tokens = torch.zeros(
            (max_num_reqs, 1),
            dtype=torch.int64,
        )
        self.draft_tokens = torch.zeros((max_num_reqs, 5), dtype=torch.int64)
        self.max_seq_len = np.array(prompt_lens, dtype=np.int32) + 8


def _copy_to_device(value, out=None, device=None):
    del device
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        out.copy_(tensor)
        return out
    return tensor.clone()


def _prepare_pos_seq_lens(
    idx_mapping,
    query_start_loc,
    num_computed_tokens,
    positions,
    seq_lens,
) -> None:
    for batch_index, request_index in enumerate(idx_mapping.tolist()):
        query_start = int(query_start_loc[batch_index])
        query_end = int(query_start_loc[batch_index + 1])
        num_computed = int(num_computed_tokens[request_index])
        positions[query_start:query_end] = torch.arange(
            num_computed,
            num_computed + query_end - query_start,
            dtype=positions.dtype,
        )
        seq_lens[batch_index] = num_computed + query_end - query_start


def _combine_sampled_and_draft_tokens(*args):
    total_num_logits = args[-1]
    return torch.arange(total_num_logits, dtype=torch.int64)


def _make_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    request_ids: list[str],
    prompt_lens: list[int],
    computed_tokens: list[int],
    computed_prefill_tokens: list[int],
    rswa: bool = True,
    speculative_method: str | None = None,
):
    runner = object.__new__(ascend_model_runner.NPUModelRunner)
    max_num_reqs = max(8, len(request_ids))
    max_num_tokens = 32
    seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
    runner.device = torch.device("cpu")
    runner.max_num_reqs = max_num_reqs
    runner.max_num_tokens = max_num_tokens
    runner.num_speculative_steps = 5
    runner.decode_query_len = 1
    runner.use_pp = False
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.model_config = SimpleNamespace(rswa_window=8 if rswa else None)
    runner.vllm_config = SimpleNamespace(
        model_config=runner.model_config,
        speculative_config=(SimpleNamespace(method=speculative_method) if speculative_method is not None else None),
    )
    runner.req_states = _RequestStates(
        request_ids,
        prompt_lens,
        computed_tokens,
        computed_prefill_tokens,
    )
    runner.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int32),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64),
        is_padding=torch.zeros(max_num_tokens, dtype=torch.bool),
        query_start_loc=torch.zeros(max_num_reqs + 2, dtype=torch.int32),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_np=seq_lens_cpu.numpy(),
    )
    runner.input_batch = None

    def update_seq_lens(_scheduler_output, req_ids) -> None:
        for batch_index, request_id in enumerate(req_ids):
            request_index = runner.req_states.req_id_to_index[request_id]
            runner.input_buffers.seq_lens_np[batch_index] = (
                runner.req_states.num_computed_tokens_np[request_index]
                + _scheduler_output.num_scheduled_tokens[request_id]
            )

    runner._update_seq_lens_cpu = update_seq_lens
    monkeypatch.setattr(ascend_model_runner, "async_copy_to_gpu", _copy_to_device)
    monkeypatch.setattr(
        ascend_model_runner,
        "build_attn_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        ascend_model_runner,
        "prepare_prefill_inputs",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        ascend_model_runner,
        "prepare_pos_seq_lens",
        _prepare_pos_seq_lens,
    )
    monkeypatch.setattr(
        ascend_model_runner,
        "combine_sampled_and_draft_tokens",
        _combine_sampled_and_draft_tokens,
    )
    monkeypatch.setattr(ascend_model_runner, "update_cos_sin", lambda *_args: None)
    return runner


def _scheduler(num_scheduled_tokens: dict[str, int]):
    return SimpleNamespace(
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        has_structured_output_requests=False,
    )


def _batch_desc(
    num_tokens: int,
    *,
    cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    num_reqs: int | None = None,
) -> BatchExecutionDescriptor:
    return BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
    )


def test_ascend_input_batch_matches_current_core_required_abi() -> None:
    parameters = signature(AscendInputBatch).parameters

    for field_name in ("is_padding", "prompt_lens"):
        assert field_name in parameters
        assert parameters[field_name].default is Parameter.empty


@pytest.mark.parametrize("speculative_method", [None, "dspark"])
@pytest.mark.parametrize(
    ("scheduled", "prompt_lens", "computed", "computed_prefill", "expected_prefill"),
    [
        ({"prefill": 4}, [4], [0], [0], [True]),
        ({"decode": 1}, [4], [4], [4], [False]),
        (
            {"prefill": 2, "decode": 1},
            [6, 4],
            [0, 4],
            [0, 4],
            [False, True],
        ),
    ],
)
def test_prepare_inputs_populates_current_abi_for_prefill_decode_and_mixed(
    monkeypatch: pytest.MonkeyPatch,
    speculative_method: str | None,
    scheduled: dict[str, int],
    prompt_lens: list[int],
    computed: list[int],
    computed_prefill: list[int],
    expected_prefill: list[bool],
) -> None:
    runner = _make_runner(
        monkeypatch,
        request_ids=list(scheduled),
        prompt_lens=prompt_lens,
        computed_tokens=computed,
        computed_prefill_tokens=computed_prefill,
        speculative_method=speculative_method,
    )
    monkeypatch.setattr(ascend_model_runner.envs, "VLLM_MOE_SKIP_PADDING", True)

    batch = runner.prepare_inputs(
        _scheduler(scheduled),
        _batch_desc(sum(scheduled.values())),
    )

    expected_prompt_lens = [prompt_lens[runner.req_states.req_id_to_index[request_id]] for request_id in batch.req_ids]
    assert batch.prompt_lens is not None
    assert batch.prompt_lens.tolist() == expected_prompt_lens
    assert batch.prompt_lens.shape == (len(scheduled),)
    assert batch.prompt_lens.dtype is torch.int32
    assert batch.prompt_lens.device.type == "cpu"
    assert batch.is_prefilling_np.tolist() == expected_prefill
    assert batch.is_padding.shape == (sum(scheduled.values()),)
    assert not batch.is_padding.any()


def test_prepare_inputs_marks_graph_token_padding_and_ignores_padded_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_runner(
        monkeypatch,
        request_ids=["decode-a", "decode-b"],
        prompt_lens=[7, 11],
        computed_tokens=[7, 11],
        computed_prefill_tokens=[7, 11],
    )
    monkeypatch.setattr(ascend_model_runner.envs, "VLLM_MOE_SKIP_PADDING", True)

    batch = runner.prepare_inputs(
        _scheduler({"decode-a": 1, "decode-b": 1}),
        _batch_desc(4, cg_mode=CUDAGraphMode.FULL, num_reqs=4),
    )

    assert batch.is_padding.tolist() == [False, False, True, True]
    assert batch.prompt_lens is not None
    assert batch.prompt_lens.tolist() == [7, 11]
    assert batch.prompt_lens.shape == (batch.num_reqs,)
    assert batch.prompt_lens.shape != (batch.num_reqs_after_padding,)


def test_prepare_inputs_refreshes_prompt_lens_and_padding_each_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_runner(
        monkeypatch,
        request_ids=["request-a", "request-b"],
        prompt_lens=[5, 9],
        computed_tokens=[5, 9],
        computed_prefill_tokens=[5, 9],
    )
    monkeypatch.setattr(ascend_model_runner.envs, "VLLM_MOE_SKIP_PADDING", True)

    first = runner.prepare_inputs(
        _scheduler({"request-a": 1}),
        _batch_desc(4, cg_mode=CUDAGraphMode.FULL, num_reqs=4),
    )
    first_prompt_lens = first.prompt_lens
    assert first_prompt_lens is not None
    assert first_prompt_lens.tolist() == [5]
    assert first.is_padding.tolist() == [False, True, True, True]

    second = runner.prepare_inputs(
        _scheduler({"request-b": 1}),
        _batch_desc(1),
    )

    assert second.prompt_lens is not None
    assert second.prompt_lens.tolist() == [9]
    assert second.prompt_lens.data_ptr() != first_prompt_lens.data_ptr()
    assert second.is_padding.tolist() == [False]


def test_prepare_inputs_uses_none_prompt_lens_outside_rswa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_runner(
        monkeypatch,
        request_ids=["decode"],
        prompt_lens=[4],
        computed_tokens=[4],
        computed_prefill_tokens=[4],
        rswa=False,
    )
    monkeypatch.setattr(ascend_model_runner.envs, "VLLM_MOE_SKIP_PADDING", False)

    batch = runner.prepare_inputs(_scheduler({"decode": 1}), _batch_desc(1))

    assert batch.prompt_lens is None
    assert batch.is_padding.tolist() == [False]


def test_prepare_inputs_keeps_previous_batch_if_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _make_runner(
        monkeypatch,
        request_ids=["decode"],
        prompt_lens=[4],
        computed_tokens=[4],
        computed_prefill_tokens=[4],
    )
    previous_batch = object()
    runner.input_batch = previous_batch
    monkeypatch.setattr(ascend_model_runner.envs, "VLLM_MOE_SKIP_PADDING", True)

    def reject_batch(**_kwargs):
        raise RuntimeError("input batch construction failed")

    monkeypatch.setattr(ascend_model_runner, "AscendInputBatch", reject_batch)

    with pytest.raises(RuntimeError, match="input batch construction failed"):
        runner.prepare_inputs(_scheduler({"decode": 1}), _batch_desc(1))

    assert runner.input_batch is previous_batch
