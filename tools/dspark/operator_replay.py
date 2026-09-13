# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay bounded real SWA capsules without model weights or TP initialization."""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import torch

MAX_RESTORE_BYTES = 2 * 1024**3


def file_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def runtime_identity():
    maps = Path("/proc/self/maps")
    loaded = (
        sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if "/" in line and any(s in line.lower() for s in ("custom", "optiling", "vllm_ascend_c", "aicpu"))
            }
        )
        if maps.exists()
        else []
    )
    artifacts = {Path(p) for p in loaded if Path(p).is_file()}
    for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if not entry:
            continue
        root = Path(entry)
        artifacts.update(
            p
            for p in root.rglob("*")
            if p.is_file()
            and (
                "sparse_attn_sharedkv" in str(p).lower()
                or "sparseattnsharedkv" in str(p).lower()
                or p.name in ("version.info", "version.json", "libcust_opapi.so")
            )
        )
    repo = Path(__file__).resolve().parents[2]
    sources = [
        repo / "csrc/torch_binding.cpp",
        *sorted((repo / "csrc/attention/sparse_attn_sharedkv").rglob("*.h")),
        *sorted((repo / "csrc/attention/sparse_attn_sharedkv").rglob("*.cpp")),
    ]
    sources += sorted((repo / "csrc/attention/sparse_attn_sharedkv_metadata").rglob("*.cpp"))
    sources += sorted((repo / "csrc/attention/sparse_attn_sharedkv_metadata").rglob("*.h"))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_npu": str(getattr(sys.modules.get("torch_npu"), "__version__", "unavailable")),
        "loaded_paths": loaded,
        "artifacts": [file_identity(p) for p in sorted(artifacts)[:256]],
        "artifact_truncated": len(artifacts) > 256,
        "sources": [file_identity(p) for p in sources if p.exists()],
        "source_binary_correspondence": (
            "UNKNOWN without reproducible build manifest; hashes identify files, not build provenance"
        ),
    }


def validate(capsule):
    if (
        capsule["schema"] != 1
        or capsule["performance_eligible"]
        or capsule["coverage"] != "PREFIX_AND_FULL_GUARD_PAGES"
    ):
        raise ValueError("Unavailable/non-diagnostic capsule")
    values = capsule["values"]
    if values["receipts"].tolist() != [capsule["identity"]["execution"]] * 2 or not capsule["query_mapping_matches"]:
        raise ValueError("Stale/missing call receipt or mapping")
    starts, seq = values["cu_seqlens_q"], values["seqused_kv"]
    if len(starts) != len(seq) + 1 or starts[0] != 0 or (starts.diff() < 0).any() or starts[-1] > values["q"].shape[0]:
        raise ValueError("Invalid padded query descriptor")
    if (seq < starts.diff()).any() or (seq > capsule["options"]["max_seq_len"]).any():
        raise ValueError("Prefix coverage insufficient")
    table, pages = values["ori_block_table"], values["page_ids"]
    if not torch.equal(table[:, : capsule["columns"]].long(), pages):
        raise ValueError("Captured page mapping mismatch")
    shape = capsule["layouts"]["ori_kv"]["shape"]
    required = torch.arange(pages.shape[1])[None, :] * shape[1] < seq[:, None]
    valid_pages = (pages >= 0) & (pages < shape[0])
    if not (valid_pages | ~required).all():
        raise ValueError("Invalid required page address")
    # Duplicate physical pages must hold identical bytes in this same-call copy.
    first = {}
    for index, page in enumerate(pages.flatten().tolist()):
        if not 0 <= page < shape[0]:
            continue  # unmapped guard entry, not a physical page
        bits = values["pages"][index].contiguous().view(torch.uint8)
        if page in first and not torch.equal(bits, first[page]):
            raise ValueError("Repeated page content differs: concurrent write/capture instability")
        first[page] = bits
    return capsule


def restore(capsule, device):
    """Keep original page numbers, descriptor sizes, strides and small-input aliases.

    Uncaptured pages are NaN sentinels, explicitly NOT original bytes. Valid
    source-contract reads must be covered; an out-of-contract kernel cannot
    be called an exact replay. No address compression or zero substitution.
    """
    validate(capsule)
    values, layouts = capsule["values"], capsule["layouts"]
    groups, result = {}, {}
    for name, spec in layouts.items():
        span = spec["storage_offset"] + 1 + sum((n - 1) * s for n, s in zip(spec["shape"], spec["stride"]))
        key = (spec["storage_ptr"], spec["dtype"])
        groups.setdefault(key, {"span": 0, "names": []})
        groups[key]["span"] = max(groups[key]["span"], span)
        groups[key]["nbytes"] = max(groups[key].get("nbytes", 0), spec.get("storage_nbytes", 0))
        groups[key]["base_dtype"] = spec.get("base_dtype", spec["dtype"])
        groups[key]["names"].append(name)
    pointers = [key[0] for key in groups]
    if len(set(pointers)) != len(pointers):
        raise ValueError("Cross-dtype input alias unsupported; not silently split")
    used = 0
    for (_, dtype_name), group in groups.items():
        dtype = getattr(torch, dtype_name.removeprefix("torch."))
        element_size = torch.empty((), dtype=dtype).element_size()
        nbytes = max(group["nbytes"], group["span"] * element_size)
        used += nbytes
        if used > MAX_RESTORE_BYTES:
            raise ValueError("Replay allocation budget exceeded")
        base_dtype = getattr(torch, group["base_dtype"].removeprefix("torch."))
        base_size = torch.empty((), dtype=base_dtype).element_size()
        backing = torch.empty((nbytes // base_size,), dtype=base_dtype, device=device)
        storage = backing.view(dtype)
        storage.fill_(float("nan") if dtype.is_floating_point else -1)
        if "ori_kv" in group["names"] and len(group["names"]) != 1:
            raise ValueError("Uncovered cache/input alias")
        for name in group["names"]:
            spec = layouts[name]
            view = storage.as_strided(spec["shape"], spec["stride"], spec["storage_offset"])
            if name != "ori_kv":
                view.copy_(values[name].to(device))
            result[name] = view
    seen = set()
    for index, page in enumerate(values["page_ids"].flatten().tolist()):
        if 0 <= page < result["ori_kv"].shape[0] and page not in seen:
            result["ori_kv"][page].copy_(values["pages"][index].to(device))
            seen.add(page)
    for name, value in result.items():
        expected_format = layouts[name].get("npu_format")
        if expected_format is not None and device != "cpu":
            import torch_npu

            if torch_npu.get_npu_format(value) != expected_format:
                raise ValueError("Original NPU format unavailable; no silent format conversion")
    # Verify all restored views, including aliases, after every copy is complete.
    for name in result.keys() - {"ori_kv"}:
        if not torch.equal(
            result[name].cpu().contiguous().view(torch.uint8), values[name].contiguous().view(torch.uint8)
        ):
            raise ValueError("Input alias/layout restore mismatch")
    return result


def reference(capsule):
    """Float64 SWA causal attention with the per-head sink in the denominator."""
    validate(capsule)
    v, params = capsule["values"], capsule["scalars"]
    q, sink = v["q"].double(), v["sinks"].double()
    result = torch.full_like(q, float("nan"))  # padding has no semantic reference
    lookup = {page: v["pages"][i].double() for i, page in enumerate(v["page_ids"].flatten().tolist())}
    block_size = capsule["layouts"]["ori_kv"]["shape"][1]
    starts = v["cu_seqlens_q"].tolist()
    for request, (start, end) in enumerate(zip(starts, starts[1:])):
        seq = int(v["seqused_kv"][request])
        for row in range(start, end):
            position = seq - (end - row)
            positions = range(
                max(0, position - params["ori_win_left"]), min(seq, position + params["ori_win_right"] + 1)
            )
            kv = torch.stack(
                [
                    lookup[int(v["ori_block_table"][request, pos // block_size])][pos % block_size, 0]
                    for pos in positions
                ]
            )
            scores = q[row] @ kv.T * params["softmax_scale"]
            weights = torch.softmax(torch.cat((scores, sink[:, None]), dim=1), dim=1)[:, :-1]
            result[row] = weights @ kv
    return result


def regenerate(inputs, params):
    starts, seq = inputs["cu_seqlens_q"], inputs["seqused_kv"]
    return torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=inputs["q"].shape[1],
        num_heads_kv=1,
        head_dim=inputs["q"].shape[2],
        cu_seqlens_q=starts,
        seqused_kv=seq,
        batch_size=seq.numel(),
        max_seqlen_q=int(starts.diff().max().cpu()),
        max_seqlen_kv=int(seq.max().cpu()),
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=params["ori_win_left"],
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(starts.device),
    )


def run(capsule, mode, metadata):
    import torch_npu  # noqa: F401

    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env(include_vendor_lib=True)
    from vllm_ascend import vllm_ascend_C  # noqa: F401

    inputs = restore(capsule, "npu")
    q = inputs.pop("q")
    params = capsule["scalars"]
    if metadata == "regenerated":
        inputs["metadata"] = regenerate({"q": q, **inputs}, params)
    op = torch.ops._C_ascend.npu_sparse_attn_sharedkv

    def invoke():
        return op(q, **inputs, **params)[0]

    if mode == "aclgraph":
        warm_stream = torch.npu.Stream()
        warm_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(warm_stream):
            invoke()
        torch.npu.current_stream().wait_stream(warm_stream)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = invoke()
        snapshots = []
        for _ in range(3):
            graph.replay()
            snapshots.append(output.cpu().clone())
    else:
        snapshots = [invoke().cpu() for _ in range(3)]
    return snapshots, inputs["metadata"].cpu(), runtime_identity()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("aclgraph", "eager", "reference"), default="aclgraph")
    parser.add_argument("--metadata", choices=("saved", "regenerated"), default="saved")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    capsule = validate(torch.load(args.capsule, map_location="cpu", weights_only=True))
    ref = reference(capsule)
    torch.save(ref, args.output / "reference.pt")
    outputs, metadata, runtime = (
        ([], None, runtime_identity()) if args.mode == "reference" else run(capsule, args.mode, args.metadata)
    )
    if outputs:
        torch.save({"outputs": outputs, "metadata": metadata}, args.output / "replay.pt")
    valid = int(capsule["values"]["cu_seqlens_q"][-1])
    result = {
        "capsule": file_identity(args.capsule),
        "identity": capsule["identity"],
        "mode": args.mode,
        "metadata": args.metadata,
        "performance_eligible": False,
        "root_cause": "UNKNOWN",
        "scope": "local operator, preserved covered pages/physical ids; not full graph/stream history or unknown pages",
        "reference_nonfinite_rows": (~torch.isfinite(ref[:valid]).flatten(1).all(1)).nonzero().flatten().tolist(),
        "replay_nonfinite_rows": [
            (~torch.isfinite(x[:valid]).flatten(1).all(1)).nonzero().flatten().tolist() for x in outputs
        ],
        "captured_nonfinite_rows": (~torch.isfinite(capsule["values"]["output"][:valid]).flatten(1).all(1))
        .nonzero()
        .flatten()
        .tolist(),
        "max_abs_error": [
            float((x[:valid].double() - ref[:valid]).abs().max()) if torch.isfinite(x[:valid]).all() else None
            for x in outputs
        ],
        "runtime": runtime,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runtime"}, indent=2))


if __name__ == "__main__":
    main()
