# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Confidence preflight, separate calibration, profile compilation and phase evidence."""

import argparse
import hashlib
import json
import math
import statistics
import struct
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def summarize_verification(before, after, ranks):
    """Validate phase deltas and rank agreement; count each logical replay once."""

    def indexed(snapshot):
        if not isinstance(snapshot, list) or len(snapshot) != ranks:
            raise ValueError("Missing rank verification evidence.")
        result = {row["rank"]: row for row in snapshot}
        if set(result) != set(range(ranks)):
            raise ValueError("Duplicate/missing rank verification evidence.")
        return result

    starts, ends = indexed(before), indexed(after)
    evidence = []
    for rank in range(ranks):
        first, last = starts[rank], ends[rank]
        if last.get("error") or last["failed_execution_count"] != first["failed_execution_count"]:
            raise ValueError("Failed execution cannot publish adaptive replay evidence.")
        initial = first["confidence_verification"]
        final = last["confidence_verification"]
        if final["mode"] != initial["mode"] or final["weights"] != initial["weights"]:
            raise ValueError("Verification mode or weight provenance changed within a run.")
        if final["mode"] == "confidence" and (
            not final["weights"]
            or not final["weights"].get("loaded_parameters")
            or not final["weights"].get("weights_sha256")
        ):
            raise ValueError("Missing loaded confidence weight provenance.")
        delta = {}
        for key in (
            "generated",
            "scheduled",
            "verified",
            "accepted",
            "policy_seconds",
            "confidence_transfer_seconds",
            "confidence_head_calls",
            "confidence_batches",
            "specified_batches",
            "fixed_admission_batches",
        ):
            delta[key] = final[key] - initial[key]
            if not math.isfinite(delta[key]) or delta[key] < 0:
                raise ValueError("Verification counters regressed.")
        for key in ("length_histogram", "verified_by_position", "generated_by_position", "accepted_by_position"):
            delta[key] = [end - start for start, end in zip(initial[key], final[key])]
            if len(delta[key]) != (6 if key == "length_histogram" else 5) or any(v < 0 for v in delta[key]):
                raise ValueError("Invalid position counter widths/deltas.")
        if (
            sum(delta["accepted_by_position"]) != delta["accepted"]
            or sum(delta["verified_by_position"]) != delta["verified"]
            or sum(delta["generated_by_position"]) != delta["generated"]
            or sum(i * n for i, n in enumerate(delta["length_histogram"])) != delta["verified"]
        ):
            raise ValueError("Inconsistent verified/generated/accepted position accounting.")
        if any(a > v for a, v in zip(delta["accepted_by_position"], delta["verified_by_position"])):
            raise ValueError("Accepted candidates exceed actual verification denominators.")
        layouts = {}
        for row in first["query_layouts"]:
            layouts[(row["capacity"], tuple(row["query_lengths"]))] = row["count"]
        measured = []
        for row in last["query_layouts"]:
            lengths = row["query_lengths"]
            count = row["count"] - layouts.get((row["capacity"], tuple(lengths)), 0)
            if (
                count < 0
                or any(type(q) is not int or not 1 <= q <= 6 for q in lengths)
                or sum(lengths) > row["capacity"]
            ):
                raise ValueError("Invalid actual FULL query layout.")
            if count:
                measured.append({**row, "count": count})
        if not measured:
            raise ValueError("No successful measured variable FULL execution.")
        delta["layouts"] = measured
        delta["rank"] = rank
        evidence.append(delta)
    for rank in range(1, ranks):
        if any(
            ends[rank]["confidence_verification"][key] != ends[0]["confidence_verification"][key]
            for key in ("mode", "weights", "calibration")
        ):
            raise ValueError("TP ranks disagree on confidence provenance.")
    comparable = (
        "generated",
        "scheduled",
        "verified",
        "accepted",
        "length_histogram",
        "layouts",
        "verified_by_position",
        "generated_by_position",
        "accepted_by_position",
        "confidence_head_calls",
        "confidence_batches",
        "specified_batches",
        "fixed_admission_batches",
    )
    if any(any(row[key] != evidence[0][key] for key in comparable) for row in evidence[1:]):
        raise ValueError("TP rank verification counters/layouts disagree; never sum TP replay counts.")
    first = evidence[0]
    if ends[0]["confidence_verification"]["mode"] == "confidence" and (
        first["confidence_batches"] <= 0 or first["confidence_head_calls"] <= 0
    ):
        raise ValueError("No measured confidence budget decision; fixed admission is not adaptive execution.")
    return {
        "status": "available",
        "mode": ends[0]["confidence_verification"]["mode"],
        "per_rank": evidence,
        "logical_full_replays": sum(row["count"] for row in first["layouts"]),
        "mixed_length_full_replays": sum(
            row["count"] for row in first["layouts"] if len(set(row["query_lengths"])) > 1
        ),
        "effective_target_tokens": sum(sum(row["query_lengths"]) * row["count"] for row in first["layouts"]),
        "padded_target_tokens": sum(row["capacity"] * row["count"] for row in first["layouts"]),
        "accepted_per_verified_position": [
            a / v if v else None for a, v in zip(first["accepted_by_position"], first["verified_by_position"])
        ],
        "accepted_per_generated_position": [
            a / v if v else None for a, v in zip(first["accepted_by_position"], first["generated_by_position"])
        ],
        "weights": ends[0]["confidence_verification"]["weights"],
        "calibration": ends[0]["confidence_verification"]["calibration"],
    }


def freeze_verification_config(path, directory):
    """Snapshot policy assets before engines start; paths in the copy are absolute."""
    directory.mkdir(parents=True, exist_ok=False)
    path = Path(path).resolve()
    options = json.loads(path.read_text())
    hashes = {"original_config": hashlib.sha256(path.read_bytes()).hexdigest()}
    for key in ("cost_profile", "calibration"):
        if options.get(key):
            source = Path(options[key])
            if not source.is_absolute():
                source = path.parent / source
            payload = source.read_bytes()
            target = directory / f"{key}.json"
            target.write_bytes(payload)
            hashes[key] = hashlib.sha256(payload).hexdigest()
            options[key] = str(target.resolve())
    target = directory / "verification.json"
    target.write_text(json.dumps(options, indent=2, allow_nan=False) + "\n")
    (directory / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return target


def _checkpoint_index(model):
    """Mirror DefaultModelLoader's standard-index precedence, without changing files.

    Without the standard index, the frozen loader scans top-level *.safetensors;
    the ModelSlim index is then an audit manifest, not a loader filtering override.
    Ambiguous manifests fail closed even if the loader would pick the standard one.
    """
    names = ("model.safetensors.index.json", "quant_model_weights.safetensors.index.json")
    found = []
    for name in names:
        path = model / name
        if path.is_file():
            payload = path.read_bytes()
            index = json.loads(payload)
            mapping = index.get("weight_map")
            if (
                not isinstance(mapping, dict)
                or not mapping
                or any(not isinstance(key, str) or not isinstance(value, str) for key, value in mapping.items())
            ):
                raise ValueError(f"Invalid weight_map in checkpoint index {name}.")
            found.append((path, mapping, hashlib.sha256(payload).hexdigest()))
    if not found:
        raise ValueError(f"Missing checkpoint index; expected one of {names}.")
    if len(found) > 1 and found[0][1] != found[1][1]:
        raise ValueError("Conflicting checkpoint index weight_map entries; refusing ambiguous model files.")
    path, mapping, digest = found[0]
    return mapping, {
        "index_file": path.name,
        "index_sha256": digest,
        "index_selection": (
            "standard index takes loader precedence"
            if path.name == names[0]
            else "quantized audit index; loader scans top-level safetensors"
        ),
        "available_indices_sha256": {item[0].name: item[2] for item in found},
    }


def checkpoint_preflight(model):
    config = json.loads((model / "config.json").read_text())
    weight_map, index_receipt = _checkpoint_index(model)
    layers = config.get("n_mtp_layers")
    if layers is None:
        layers = config.get("dspark_num_mtp_layers", 3)
    stage = int(layers or 3) - 1
    name = f"mtp.{stage}.confidence_head.proj.weight"
    if name not in weight_map:
        raise ValueError(f"Checkpoint lacks real confidence weight {name}; adaptive mode is unavailable.")
    shard = model / weight_map[name]
    if shard.parent != model or shard.suffix != ".safetensors":
        raise ValueError("Confidence shard must match the loader top-level *.safetensors selection.")
    if not shard.is_file():
        raise ValueError(f"Missing confidence checkpoint shard: {shard.name}.")
    if model.resolve() not in shard.resolve().parents:
        raise ValueError("Unsafe checkpoint shard path.")
    with shard.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= 128 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length.")
        header = json.loads(stream.read(length))
        weight = header[name]
        expected = [1, config["hidden_size"] + config["dspark_markov_rank"]]
        if weight["shape"] != expected or weight["dtype"] not in {"BF16", "F16", "F32"}:
            raise ValueError("Unexpected confidence projection shape/dtype.")
        start, end = weight["data_offsets"]
        if not 0 <= start < end <= shard.stat().st_size - 8 - length:
            raise ValueError("Invalid confidence weight byte range.")
        stream.seek(8 + length + start)
        payload = stream.read(end - start)
    description = json.loads((model / "quant_model_description.json").read_text())
    if description.get(name) != "FLOAT":
        raise ValueError("Confidence projection must be declared FLOAT in ModelSlim metadata.")
    return {
        "status": "present_not_runtime_loaded",
        "weight": name,
        "shard": shard.name,
        "shape": expected,
        "dtype": weight["dtype"],
        "checkpoint_weight_sha256": hashlib.sha256(payload).hexdigest(),
        "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        **index_receipt,
    }


def compile_profile(results):
    if not results:
        raise ValueError("No isolated profile results.")
    profiles = []
    raw = []
    for path in results:
        data = json.loads(path.read_text())
        if not data.get("cleanup", {}).get("engine_shutdown_complete"):
            raise ValueError("Profile engine did not shut down cleanly.")
        if data.get("performance_eligible", True):
            raise ValueError("A performance run cannot be relabeled as an isolated profile.")
        snapshots = data["graph_execution"]["boundary_snapshots"]
        final = snapshots[-1]
        tp = data["effective_config"]["tensor_parallel_size"]
        if len(final) != tp or {row["rank"] for row in final} != set(range(tp)):
            raise ValueError("Missing profile rank evidence.")
        for row in snapshots[-1]:
            if row.get("error") or row.get("failed_execution_count", 0):
                raise ValueError("Failed profile execution.")
            profile = row["cost_profile"]
            if profile["source"] != "isolated_npu_event_profile":
                raise ValueError("No real profile event evidence.")
            profiles.append(profile)
        raw.append(hashlib.sha256(path.read_bytes()).hexdigest())
    identity = profiles[0]["identity"]
    if any(profile["identity"] != identity for profile in profiles):
        raise ValueError("Cannot merge different model/hardware/TP/EP/capture profile identities.")
    timings = {"target": {}, "draft": {}}
    contexts = []
    for profile in profiles:
        for row in profile["measurements"]:
            seconds = row["seconds"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Nonpositive/nonfinite NPU timing.")
            timings[row["kind"]].setdefault(row["size"], []).append(seconds)
            contexts.append(row["context"])
    # Rank maxima, conservatively represented by upper observed cost. They
    # are not independent repeats and are never summed as logical work.
    if set(timings["target"]) != set(identity["capture_sizes"]):
        raise ValueError("Profile did not measure every requested graph tier; add isolated specified-length runs.")
    return {
        "schema_version": 1,
        "source": "isolated_npu_event_profile",
        "identity": identity,
        "raw_measurements_sha256": raw,
        "context_range": [min(contexts), max(contexts)],
        "target_seconds": {k: max(v) for k, v in timings["target"].items()},
        "draft_seconds": {k: max(v) for k, v in timings["draft"].items()},
        "scheduler_seconds": measured_scheduler_overhead(identity),
        "aggregation": "maximum observed event duration across samples and TP ranks; no extrapolation",
    }


def measured_scheduler_overhead(identity):
    # Profile CPU scheduling separately, without touching worker/request state.
    import time

    from vllm_ascend.spec_decode.dspark_verification import ConfidenceRow, CostTable, allocate_prefixes

    count = identity["max_num_seqs"]
    rows = [ConfidenceRow(str(i), 1, (0.9,) * 5) for i in range(count)]
    table = CostTable({size: 1.0 for size in identity["capture_sizes"]}, {count: 1.0}, 0, (0, 1), {})
    values = []
    for _ in range(20):
        start = time.perf_counter()
        allocate_prefixes(
            rows,
            {str(i): 5 for i in range(count)},
            base_tokens=count,
            sampling_requests=count,
            draft_requests=count,
            context=0,
            costs=table,
        )
        values.append(time.perf_counter() - start)
    return statistics.median(values)


def calibrate(path, weight_hash):
    import torch

    data = json.loads(path.read_text())
    if data.get("split") != "calibration" or data.get("weights_sha256") != weight_hash:
        raise ValueError("Require separate calibration split and exact loaded-weight hash.")
    logits = torch.tensor(data["conditional_logits"], dtype=torch.float64)
    accepted = torch.tensor(data["conditional_accepted"], dtype=torch.float64)
    if logits.ndim != 1 or logits.numel() < 2 or logits.shape != accepted.shape:
        raise ValueError("Calibration needs paired logits and conditional acceptance labels.")
    if not torch.isfinite(logits).all() or not ((accepted == 0) | (accepted == 1)).all():
        raise ValueError("Invalid calibration samples.")
    # Rows must include only positions reached by actual verification: later
    # positions after an earlier rejection have no conditional ground truth.
    scale_log = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([scale_log, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits * scale_log.exp() + bias, accepted)
        loss.backward()
        return loss

    optimizer.step(closure)
    scale, intercept = float(scale_log.detach().exp()), float(bias.detach())
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(intercept):
        raise ValueError("Calibration fit failed.")
    return {
        "schema_version": 1,
        "split": "calibration",
        "weights_sha256": weight_hash,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n": logits.numel(),
        "scale": scale,
        "bias": intercept,
        "method": "positive-slope logistic calibration on conditional labels",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("checkpoint", "profile", "calibrate"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--inputs", nargs="+", type=Path)
    parser.add_argument("--weights-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "checkpoint":
        result = checkpoint_preflight(args.model)
    elif args.action == "profile":
        result = compile_profile(args.inputs)
    else:
        result = calibrate(args.inputs[0], args.weights_sha256)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output)


def verification_acceptance(summary):
    row = summary["per_rank"][0]
    requests = sum(row["length_histogram"])
    accepted = row["accepted"]
    average = accepted / requests if requests else None
    return {
        "num_drafts": requests,
        "num_draft_tokens": row["verified"],
        "num_accepted_candidate_tokens": accepted,
        "accepted_candidate_tokens_per_verification": average,
        "effective_committed_tokens_per_verification": 1 + average if average is not None else None,
        "effective_acceptance_length": 1 + average if average is not None else None,
        "accepted_candidate_tokens_per_position": row["accepted_by_position"],
        "acceptance_per_position": summary["accepted_per_verified_position"],
    }


if __name__ == "__main__":
    main()
