# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Installed interface smoke test before weights; CPU buffers are NOT fit evidence.

Run in a fresh process before mock-heavy pytest files. Its timeout is included
in the existing 600-second host stage. Import/constructor/RPC failures are fatal.
"""

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS


def exercise(speculator_type, block_tables_type, staged_type, config_type, group_type, tensor_type, attention_type):
    # Lazy imports keep host-only tooling importable without installed vLLM/NPU.
    import torch

    from vllm_ascend.diagnostics.dspark_benchmark_worker import DSparkBenchmarkWorkerExtension

    config = NS(
        speculative_config=NS(
            method="dspark",
            num_speculative_tokens=5,
            draft_model_config=NS(hf_config=NS(dspark_noise_token_id=128799)),
        )
    )
    receipts = []
    for requests in (128, 256):
        draft = speculator_type(config, torch.device("cpu"))
        spec = attention_type(block_size=32, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
        cache = config_type(
            num_blocks=2,
            kv_cache_tensors=[tensor_type(size=2 * spec.page_size_bytes, shared_by=["target", "draft"])],
            kv_cache_groups=[group_type(["target"], spec), group_type(["draft"], spec, is_eagle_group=True)],
        )
        # Hardware/UVA initialization is deliberately not simulated as an NPU
        # success. Bind small CPU allocations to the real container types; the
        # real drafter constructor and complete RPC interface are exercised.
        tables = block_tables_type.__new__(block_tables_type)
        tables.block_tables = []
        for _ in cache.kv_cache_groups:
            staged = staged_type.__new__(staged_type)
            staged.gpu = torch.empty((requests, 2), dtype=torch.int32)
            tables.block_tables.append(staged)
        tables.input_block_tables = [torch.empty_like(t.gpu) for t in tables.block_tables]
        tables.slot_mappings = torch.empty((2, 6 * requests), dtype=torch.int64)
        draft.block_tables = tables
        draft.kv_cache_config = cache
        draft.draft_kv_cache_group_ids = (1,)
        worker = NS(
            rank=0,
            model_runner=NS(
                max_num_reqs=requests,
                max_num_tokens=6 * requests,
                speculator=draft,
                block_tables=tables,
                kv_cache_config=cache,
            ),
            dspark_benchmark_graph_runtime=lambda n=requests: {"observed_capture_sizes": [6 * n]},
        )
        result = DSparkBenchmarkWorkerExtension.dspark_benchmark_capacity(worker)
        if result["draft_max_requests"] != requests or result["draft_max_tokens"] != 6 * requests:
            raise ValueError("Installed capacity interface returned incorrect allocated dimensions")
        # Prove dimensions win over configuration or an invented scalar default.
        tables.input_block_tables[1] = torch.empty((requests - 1, 2), dtype=torch.int32)
        reduced = DSparkBenchmarkWorkerExtension.dspark_benchmark_capacity(worker)
        if reduced["draft_max_requests"] != requests - 1:
            raise ValueError("Capacity RPC did not measure the actual input table")
        receipts.append({"complete_rpc": result, "reduced_input_rows": reduced["draft_max_requests"]})
    return receipts


def installed_check():
    # Actual installed imports, not extracted methods or patched module globals.
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

    types = (
        AscendDSparkSpeculator,
        BlockTables,
        StagedWriteTensor,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        FullAttentionSpec,
    )
    identities = []
    for cls in types:
        path = Path(inspect.getfile(cls)).resolve()
        identities.append(
            {
                "class": cls.__qualname__,
                "module": cls.__module__,
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "status": "PASSED_INTERFACE_ONLY",
        "scope": "Real installed drafter constructor and RPC; synthetic CPU KV bindings, no weights/NPU fit test",
        "mro": [f"{c.__module__}.{c.__qualname__}" for c in AscendDSparkSpeculator.__mro__],
        "sources": identities,
        "cases": exercise(*types),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        report = installed_check()
    except Exception as exc:
        report = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    remaining = args.pytest_args
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if not remaining:
        return 0
    return subprocess.run([sys.executable, "-m", "pytest", *remaining], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
