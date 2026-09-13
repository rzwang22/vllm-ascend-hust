# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real Core page lifetime plus synthetic writes, no weights or attention kernel.

The NPU cases exercise actual ACLGraph, not a model or compressor reproduction.
Run separately from mock-heavy plugin tests so the installed Core is used.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("device,mode", [("cpu", "eager"), ("npu", "eager"), ("npu", "aclgraph")])
def test_live_swa_page_survives_cross_group_writes(tmp_path, device, mode):
    if device == "npu":
        pytest.importorskip("torch_npu")
        assert torch.npu.is_available(), "NPU case must execute on actual hardware"
        torch.npu.set_device(0)
    vllm = pytest.importorskip("vllm")
    source = Path(vllm.__file__).resolve().parents[1] / "tests/v1/core/test_async_swa_reclamation.py"
    assert source.is_file(), "Run against the pinned Core checkout containing its real allocator regression"
    spec = importlib.util.spec_from_file_location("swa_core_regression", source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    scheduler = core.make_scheduler(tmp_path)
    request = core.ready_request(scheduler)
    manager = scheduler.kv_cache_manager
    swa, state = manager.coordinator.single_type_managers
    history_page = swa.req_to_blocks[request.request_id][2].block_id
    # Persistent worker block table is intentionally not rewritten by scheduler reclaim.
    table = torch.tensor([b.block_id for b in swa.req_to_blocks[request.request_id]], device=device)
    table_address = table.data_ptr()
    backing = torch.full((256, 32, 1, 512), 3.0, dtype=torch.bfloat16, device=device)
    state_view = backing.view(torch.float32).reshape(256, 8, 1024)
    first, second = scheduler.schedule(), scheduler.schedule()
    other = state.allocate_new_blocks("other", 8, 8)
    # Both a different group of this request and another request may reuse free pages.
    write_ids = sorted({b.block_id for b in state.req_to_blocks[request.request_id] + other if not b.is_null})
    destinations = torch.tensor(write_ids, device=device)
    calls = torch.zeros(1, dtype=torch.int64, device=device)
    records = []
    graph = None

    def invoke():
        # Deliberately FP32, exercising the same byte alias layout, not compressor math.
        state_view[destinations] = 2.97
        calls.add_(1)
        return backing.index_select(0, table[2:3])[:, 31].clone(), calls.clone()

    def observe(stage, result, expected):
        values, receipt = (t.cpu().clone() for t in result)
        records.append({"stage": stage, "values": values, "receipt": receipt, "expected_calls": expected})
        assert int(receipt[0]) == expected
        assert torch.equal(values, torch.full_like(values, 3.0))
        assert history_page not in write_ids
        assert table.data_ptr() == table_address

    try:
        observe("warmup", invoke(), 1)
        if mode == "aclgraph":
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = invoke()
            records.append({"stage": "capture", "status": "not_executed", "valid_snapshot": False})
        for iteration in range(3):
            if graph is not None:
                graph.replay()
            else:
                result = invoke()
            observe(f"replay-{iteration + 1}" if graph else f"eager-{iteration + 1}", result, iteration + 2)
        scheduler.update_from_output(first, core.output_for(first, 1))
        scheduler.update_from_output(second, core.output_for(second, 4))
        assert request.num_output_placeholders == 0
        assert swa.req_to_blocks[request.request_id][2].block_id == history_page
        assert all(torch.equal(r["values"], torch.full_like(r["values"], 3.0)) for r in records if "values" in r)
    finally:
        torch.save(
            {
                "device": device,
                "mode": mode,
                "synthetic_writer": True,
                "performance_eligible": False,
                "history_page": history_page,
                "write_ids": write_ids,
                "records": records,
            },
            tmp_path / "lifetime.pt",
        )
