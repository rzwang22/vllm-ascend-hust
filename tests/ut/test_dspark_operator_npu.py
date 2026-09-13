# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small synthetic operator controls. Never label these as the archived call."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tools.dspark import operator_replay as replay


def synthetic(poison):
    generator = torch.Generator().manual_seed(17)
    q = torch.randn((12, 8, 512), generator=generator).to(torch.bfloat16)
    cache = torch.full((72, 32, 1, 512), float("nan") if poison else 0, dtype=torch.bfloat16)
    table = torch.zeros(12, 256, dtype=torch.int32)
    starts = torch.tensor([0, 1, 5, 11] + [11] * 9, dtype=torch.int32)
    seq = torch.tensor([223, 255, 530] + [0] * 9, dtype=torch.int32)
    for r, (start, end) in enumerate(zip(starts[:3], starts[1:4])):
        table[r, :22] = torch.arange(1 + r * 24, 23 + r * 24)
        for pos in range(int(seq[r] - (end - start)) - 127, int(seq[r])):
            cache[table[r, pos // 32], pos % 32] = torch.randn((1, 512), generator=generator).to(torch.bfloat16)
    spec = importlib.util.spec_from_file_location(
        "synthetic_operator_capture", Path(__file__).parents[2] / "vllm_ascend/diagnostics/dspark_profile_operator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inputs = {
        "q": q,
        "ori_kv": cache,
        "ori_block_table": table,
        "cu_seqlens_q": starts,
        "seqused_kv": seq,
        "sinks": torch.zeros(8),
        "metadata": torch.zeros(1024, dtype=torch.int32),
    }
    values = {k: v.clone() for k, v in inputs.items() if k != "ori_kv"}
    values.update(
        page_ids=table[:, :22].long().clone(),
        pages=cache[table[:, :22].long()].flatten(0, 1).clone(),
        output=torch.empty_like(q),
        receipts=torch.tensor([1, 1]),
    )
    return {
        "schema": 1,
        "performance_eligible": False,
        "coverage": "PREFIX_AND_FULL_GUARD_PAGES",
        "query_mapping_matches": True,
        "columns": 22,
        "options": {"max_seq_len": 640},
        "identity": {"execution": 1, "origin": "synthetic, not archived failure", "poison_outside_window": poison},
        "values": values,
        "layouts": {k: module.descriptor(v) for k, v in inputs.items()},
        "scalars": {
            "softmax_scale": 512**-0.5,
            "cmp_ratio": 1,
            "ori_mask_mode": 4,
            "ori_win_left": 127,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        },
    }


@pytest.mark.parametrize("poison", [False, True])
@pytest.mark.parametrize("mode", ["aclgraph", "eager"])
def test_native_swa_tail_control(tmp_path, poison, mode):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires actual Ascend sparse attention and ACLGraph")
    capsule = synthetic(poison)
    outputs, metadata, runtime = replay.run(capsule, mode, "regenerated")
    capsule["values"]["metadata"] = metadata
    ref = replay.reference(capsule)
    torch.save(capsule, tmp_path / "synthetic.pt")
    torch.save({"outputs": outputs, "reference": ref}, tmp_path / "outputs.pt")

    (tmp_path / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    for output in outputs:
        assert torch.isfinite(output[:11]).all(), "Synthetic native SWA produced nonfinite valid output"
        torch.testing.assert_close(output[:11].double(), ref[:11], atol=0.02, rtol=0.02)
