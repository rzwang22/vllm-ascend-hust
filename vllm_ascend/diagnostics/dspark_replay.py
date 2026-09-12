# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""P08-R9 opt-in target snapshots. No changes to target arithmetic or KV writes.

Layer banks exist BEFORE torch.compile/profile. Only copy_ and views are added
to the model graph. Host inspection is confined to the NPUGraph replay proxy.
This module is lazily imported only with dspark_nan_replay_window configured.
"""

from typing import Any

import torch

from vllm_ascend.diagnostics.dspark_nan import DSparkNaNDiagnostics

_LAYER_BOUNDARIES = ("attn_input", "attn_output", "residual", "ffn_input", "ffn_output", "output")
_SMALL_METADATA_LIMIT = 4096


def tensor_layout(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "data_ptr": value.data_ptr(),
        "storage_data_ptr": value.untyped_storage().data_ptr(),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


class TargetLayerSnapshots:
    """Strong, disjoint banks for each shape/layer in one worker address space.

    Consecutive uniform-decode buckets permit a symbolic triangular offset in
    the already dynamic compiled model. Profile/eager calls may write a bounded
    prefix, but only an explicitly armed, completed FULL replay is inspected.
    No Python hook, allocator, .item(), .cpu(), assertion or reduction in write().
    """

    def __init__(self, *, sizes, query_len, hidden_size, hc_mult, layers, dtype, device, rank):
        self.sizes = tuple(sizes)
        self.query_len = query_len
        if not sizes or list(sizes) != list(range(query_len, max(sizes) + 1, query_len)):
            raise ValueError("DSpark replay diagnostics require consecutive uniform-query capture sizes.")
        self.max_tokens = max(sizes)
        self.rank = rank
        self.buffers = {}
        self.receipts = {}
        self.epoch_input = torch.zeros(1, dtype=torch.int32, device=device)
        shapes = {"embedding": (hidden_size,)}
        for layer in layers:
            for boundary in _LAYER_BOUNDARIES:
                tail = (hc_mult, hidden_size) if boundary in ("residual", "output") else (hidden_size,)
                shapes[f"layer.{layer}.{boundary}"] = tail
        shapes.update(pre_hc=(hc_mult, hidden_size), post_hc=(hidden_size,), post_norm=(hidden_size,))
        for name, tail in shapes.items():
            self.buffers[name] = torch.empty((sum(sizes), *tail), dtype=dtype, device=device)
            self.receipts[name] = torch.zeros(len(sizes), dtype=torch.int32, device=device)

    def write(self, name: str, value: torch.Tensor) -> None:
        # Keep the symbolic minimum explicitly. Reading value[:max].shape[0]
        # during a large profile specializes it to max and relies on a Dynamo
        # guard; the frozen vLLM wrapper intentionally bypasses those guards.
        # Dynamo 2.10 folds builtin min for static ints and emits sym_min
        # for SymInt, preserving the runtime bound without an int conversion.
        n = min(value.shape[0], self.max_tokens)
        bucket = (n - 1) // self.query_len + 1
        offset = bucket * (bucket - 1) * self.query_len // 2
        self.buffers[name][offset : offset + n].copy_(value[:n])
        self.receipts[name][bucket - 1 : bucket].copy_(self.epoch_input)

    def views(self, size: int) -> dict[str, torch.Tensor]:
        if size not in self.sizes:
            raise ValueError(f"Uncaptured diagnostic shape {size}.")
        offset = sum(s for s in self.sizes if s < size)
        return {name: value[offset : offset + size] for name, value in self.buffers.items()}

    @property
    def allocated_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (*self.buffers.values(), *self.receipts.values(), self.epoch_input)
        )


def finite_summaries(values: dict[str, torch.Tensor], valid_rows: int) -> dict[str, Any]:
    """One D2H for all row flags; no hidden/KV/logit contents leave the device."""
    DSparkNaNDiagnostics._outside_capture()
    flags = []
    for value in values.values():
        if not value.is_floating_point() or not 0 < valid_rows <= value.shape[0]:
            raise ValueError("Replay snapshot does not cover floating-point valid rows.")
        flat = value[:valid_rows].reshape(valid_rows, -1)
        flags.append(
            torch.stack((torch.isnan(flat).any(1), torch.isposinf(flat).any(1), torch.isneginf(flat).any(1)), 1)
        )
    host = torch.stack(flags).cpu().tolist() if flags else []
    result = {}
    for (name, value), rows in zip(values.items(), host):
        stats = {**tensor_layout(value), "valid_row_range": [0, valid_rows]}
        for column, kind in enumerate(("nan", "positive_inf", "negative_inf")):
            affected = [i for i, row in enumerate(rows) if row[column]]
            stats[f"{kind}_row_count"] = len(affected)
            stats[f"{kind}_rows"] = affected
        stats["invalid"] = any(any(row) for row in rows)
        result[name] = stats
    return result


def localize(boundaries: dict[str, Any], transfers: dict[str, Any]) -> dict[str, Any]:
    last_finite, first_invalid = None, None
    for name, stats in boundaries.items():
        if stats["invalid"]:
            first_invalid = name
            break
        last_finite = name
    bad_sources = [name for name, item in transfers.items() if item["source"]["invalid"]]
    changed = [name for name, item in transfers.items() if item["different_rows"]]
    bad_destinations = [name for name, item in transfers.items() if item["destination"]["invalid"]]
    return {
        "status": "OBSERVED_BOUNDARY_BRACKET"
        if first_invalid or bad_sources or bad_destinations
        else "NO_NONFINITE_IN_OBSERVED_BOUNDARIES",
        "last_finite_boundary": last_finite,
        "first_invalid_boundary": first_invalid,
        "raw_output_nonfinite": bad_sources,
        "transfer_changed_rows": changed,
        "finite_source_nonfinite_destination": [name for name in bad_destinations if name not in bad_sources],
        "root_cause_status": "ROOT_CAUSE_NOT_YET_PROVEN",
        "unseparated": (
            "HC pre/norm, attention internals/communication, MoE internals, HC post and "
            "clone remain grouped between observed boundaries."
        ),
    }


def _host(value):
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value.tolist()


def _record_tensor(value, *, source, rows=None):
    if not isinstance(value, torch.Tensor):
        return {"source": source, "status": "unavailable", "reason": "ABI field is not a tensor"}
    selected = value if rows is None else value[:rows]
    result = {"source": source, **tensor_layout(value), "observed_shape": list(selected.shape)}
    if selected.numel() <= _SMALL_METADATA_LIMIT and not selected.is_floating_point():
        result["values"] = _host(selected)
    elif selected.is_floating_point() and selected.ndim and selected.shape[0]:
        result["finite"] = finite_summaries({"metadata": selected}, selected.shape[0])["metadata"]
        result["values_status"] = "not_dumped_floating_metadata"
    else:
        result["values_status"] = "unavailable: bounded metadata limit"
    return result


def inspect_replay_inputs(runner, captured, captured_inputs, size):
    """Called AFTER prepare_attn/copies and offloader sync, immediately pre replay.

    Read allocated block-table prefixes using actual request-state num_blocks.
    We do not scan KV capacity: sparse/compressor kernels determine read ranges
    inside the graph, so a complete target historical-KV check is unavailable.
    """
    DSparkNaNDiagnostics._outside_capture()
    batch = runner.input_batch
    n, requests = batch.num_tokens, batch.num_reqs
    mismatches = []
    unavailable = []

    def compare(field, expected, actual, relation="equal"):
        passed = (
            expected == actual
            if relation == "equal"
            else (
                isinstance(actual, list)
                and len(expected) == len(actual)
                and all(a <= e for e, a in zip(expected, actual))
            )
        )
        if not passed:
            mismatches.append({"field": field, "expected": expected, "actual": actual, "relation": relation})

    inputs = {}
    for name, rows in (
        ("input_ids", n),
        ("positions", n),
        ("query_start_loc", requests + 1),
        ("seq_lens", requests),
        ("is_padding", size),
    ):
        actual = getattr(runner.input_buffers, name, None)
        inputs[name] = _record_tensor(
            actual, source=f"runner.input_buffers.{name} immediately before NPUGraph.replay", rows=rows
        )
    offsets = _host(batch.query_start_loc_np[: requests + 1])
    device_offsets = inputs["query_start_loc"].get("values")
    compare("input_buffers.query_start_loc vs input_batch.query_start_loc_np", offsets, device_offsets)
    indices = _host(batch.idx_mapping_np)
    compare("input_batch.idx_mapping", indices, _host(batch.idx_mapping[:requests]))
    request_index = getattr(getattr(runner, "req_states", None), "req_id_to_index", None)
    if request_index is not None:
        compare(
            "request identity vs req_states.req_id_to_index",
            [request_index.get(request) for request in batch.req_ids],
            indices,
        )
    else:
        unavailable.append("req_states.req_id_to_index is absent from this runner ABI")
    seq = inputs["seq_lens"].get("values")
    expected_upper = _host(batch.seq_lens_cpu_upper_bound[:requests])
    compare("seq_lens <= CPU upper bound (not equality)", expected_upper, seq, "upper_bound")
    positions = inputs["positions"].get("values")
    if not isinstance(seq, list) or len(seq) != requests or not isinstance(positions, list) or len(positions) != n:
        return {
            "source": "pre-replay device inputs",
            "match_status": "unavailable",
            "device_inputs": inputs,
            "request_mapping": [],
            "mismatches": mismatches,
            "unavailable": ["actual seq_lens/positions do not cover this batch; dependent comparisons not performed"],
        }
    expected_positions = [p for i in range(requests) for p in range(seq[i] - (offsets[i + 1] - offsets[i]), seq[i])]
    compare("positions vs device seq_lens/query spans", expected_positions, positions)
    mapping = [
        {
            "request_id": request,
            "runner_row": row,
            "request_state_row": indices[row],
            "query_span": offsets[row : row + 2],
            "positions": positions[offsets[row] : offsets[row + 1]],
        }
        for row, request in enumerate(batch.req_ids)
    ]
    # Input IDs can be assembled from asynchronous device sampling. There is no
    # authoritative CPU token array in this frozen ABI; never invent one.
    unavailable.append(
        "CPU expected input_ids: async sampled/draft tokens are device-produced; actual device IDs are recorded"
    )
    captured_input_records = {}
    for name, value in captured_inputs.items():
        captured_input_records[name] = _record_tensor(value, source="strong capture model input reference", rows=n)
        if name in inputs:
            compare(
                f"captured_inputs.{name}.values", inputs[name].get("values"), captured_input_records[name].get("values")
            )
    tables = runner.block_tables
    group_records, metadata_records, metadata_owners = [], {}, {}
    current_metadata = runner.model_state.attn_metadata
    for group_id, groups in enumerate(runner.attn_groups):
        counts = [int(tables.num_blocks.np[group_id, index]) for index in indices]
        input_table = tables.input_block_tables[group_id]
        allocated = []
        for row, (index, count) in enumerate(zip(indices, counts)):
            if not 0 <= count <= input_table.shape[1]:
                mismatches.append(
                    {
                        "field": f"group.{group_id}.num_blocks[{row}]",
                        "actual": count,
                        "expected": f"0..{input_table.shape[1]}",
                    }
                )
                allocated.append(None)
                continue
            source = _host(tables.block_tables[group_id].gpu[index, :count])
            actual = _host(input_table[row, :count])
            compare(f"group.{group_id}.block_table[{row}] gathered vs request-state device table", source, actual)
            allocated.append(actual)
        raw_slots = _record_tensor(
            tables.slot_mappings[group_id], source="AscendBlockTables.compute_slot_mappings", rows=size
        )
        raw_slot_check = {"status": "unavailable", "reason": "CP mapping is outside this diagnostic"}
        if getattr(tables, "cp_size", 1) == 1:
            expected_slots = []
            block_size = tables.kernel_block_sizes[group_id]
            for row in range(requests):
                for position in positions[offsets[row] : offsets[row + 1]]:
                    column, offset = divmod(position, block_size)
                    if allocated[row] is None or not 0 <= column < len(allocated[row]):
                        expected_slots.append(None)
                    else:
                        expected_slots.append(allocated[row][column] * block_size + offset)
            expected_slots += [-1] * (size - n)
            known = [i for i, value in enumerate(expected_slots) if value is not None]
            unknown = [i for i, value in enumerate(expected_slots) if value is None]
            actual_slots = raw_slots.get("values")
            raw_slot_check = {
                "status": "PARTIAL" if unknown and known else "unavailable" if unknown else "CHECKED",
                "compared_rows": known,
                "unavailable_rows": unknown,
                "source": "this KV group's allocated table prefix and kernel_block_size",
            }
            if unknown:
                unavailable.append(
                    f"group.{group_id}.raw_slots rows {unknown}: logical position outside the observed allocated "
                    "prefix; compressed-cache raw slots may be unused. No expected physical block inferred."
                )
            if known:
                compare(
                    f"group.{group_id}.raw_slots (exact -1 padding)",
                    [expected_slots[i] for i in known],
                    [actual_slots[i] for i in known]
                    if isinstance(actual_slots, list) and len(actual_slots) == size
                    else actual_slots,
                )
        group_records.append(
            {
                "group_id": group_id,
                "block_size": tables.block_sizes[group_id],
                "kernel_block_size": tables.kernel_block_sizes[group_id],
                "allocated_prefix_counts": counts,
                "source": "BlockTables.num_blocks.np at actual idx_mapping; no unused columns read",
                "input_block_table_layout": tensor_layout(input_table),
                "allocated_block_ids": allocated,
                "raw_slot_mapping": raw_slots,
                "raw_slot_check": raw_slot_check,
                "cache_owners": [
                    {
                        "layer_names": list(group.layer_names),
                        "spec_type": type(group.kv_cache_spec).__name__,
                        "block_size": group.kv_cache_spec.block_size,
                        "compress_ratio": getattr(group.kv_cache_spec, "compress_ratio", None),
                    }
                    for group in groups
                ],
            }
        )
        for group in groups:
            for layer_name in group.layer_names:
                if not layer_name.startswith("model.layers."):
                    continue
                current = current_metadata.get(layer_name)
                held = (captured.attn_metadata or {}).get(layer_name)
                identity = (group_id, id(current), id(held))
                if identity in metadata_owners:
                    metadata_records[layer_name] = {
                        "group_id": group_id,
                        "same_metadata_objects_as": metadata_owners[identity],
                    }
                    continue
                metadata_owners[identity] = layer_name
                record = {"group_id": group_id, "kv_spec_block_size": group.kv_cache_spec.block_size, "fields": {}}
                record["reshape_cache_event"] = {
                    "current_id": id(current.reshape_cache_event)
                    if getattr(current, "reshape_cache_event", None) is not None
                    else None,
                    "captured_id": id(held.reshape_cache_event)
                    if getattr(held, "reshape_cache_event", None) is not None
                    else None,
                    "completion_status": "unavailable: event not queried or synchronized by diagnostics",
                }
                for branch in ("decode", "prefill"):
                    live_branch, held_branch = getattr(current, branch, None), getattr(held, branch, None)
                    if live_branch is None and held_branch is None:
                        continue
                    if live_branch is None or held_branch is None:
                        unavailable.append(f"{layer_name}.{branch}: absent in current or captured metadata")
                        continue
                    fields = record["fields"]
                    for name in ("full_compress_cos", "full_compress_sin"):
                        fields[f"{branch}.{name}"] = {
                            which: tensor_layout(value)
                            if isinstance(value := getattr(obj, name, None), torch.Tensor)
                            else None
                            for which, obj in (("current", live_branch), ("captured", held_branch))
                        }
                        fields[f"{branch}.{name}"]["content_status"] = (
                            "unavailable: full RoPE table values not dumped; compressor selects rows inside graph"
                        )
                    for name, limit in (
                        ("query_start_loc", requests + 1),
                        ("seq_lens", requests),
                        ("input_positions", n),
                        ("start_pos", requests),
                        ("slot_mapping", n),
                        ("sas_metadata", None),
                        ("qli_metadata", None),
                        ("sin", n),
                        ("cos", n),
                    ):
                        live, ref = getattr(live_branch, name, None), getattr(held_branch, name, None)
                        key = f"{branch}.{name}"
                        if live is None and ref is None:
                            fields[key] = {"status": "unavailable", "reason": "not used in this metadata branch"}
                            continue
                        fields[key] = {
                            "current": _record_tensor(
                                live, source=f"prepare_attn result: {layer_name}.{key}", rows=limit
                            ),
                            "captured": _record_tensor(
                                ref, source=f"AttentionStatePair.captured: {layer_name}.{key}", rows=limit
                            ),
                        }
                        a, b = fields[key]["current"], fields[key]["captured"]
                        if "values" in a and "values" in b:
                            compare(f"{layer_name}.{key}.captured_values", a["values"], b["values"])
                        elif (
                            isinstance(live, torch.Tensor) and isinstance(ref, torch.Tensor) and live.shape == ref.shape
                        ):
                            # Exact diagnostic comparison only within one replay's
                            # metadata, not an eager/graph output correctness gate.
                            equal = bool(torch.equal(live[:limit] if limit else live, ref[:limit] if limit else ref))
                            fields[key]["content_equal"] = equal
                            if not equal:
                                mismatches.append(
                                    {
                                        "field": f"{layer_name}.{key}.captured_content",
                                        "expected": "equal current metadata",
                                        "actual": "different",
                                    }
                                )
                    fields[f"{branch}.scalars"] = {
                        "current": {
                            key: getattr(live_branch, key, None)
                            for key in (
                                "block_size",
                                "num_reqs_actual",
                                "num_compressed_tokens",
                                "max_seqlen_q",
                                "max_seqlen_kv",
                            )
                        },
                        "captured": {
                            key: getattr(held_branch, key, None)
                            for key in (
                                "block_size",
                                "num_reqs_actual",
                                "num_compressed_tokens",
                                "max_seqlen_q",
                                "max_seqlen_kv",
                            )
                        },
                        "source": (
                            "host fields captured into op arguments; differences are observations, not "
                            "automatically invalid"
                        ),
                    }
                    for which, obj in (("current", live_branch), ("captured", held_branch)):
                        table = obj.block_table
                        fields[f"{branch}.{which}.block_table"] = {
                            **tensor_layout(table),
                            "allocated_rows": [
                                _host(table[row, :count])
                                if 0 <= count <= table.shape[1] and row < table.shape[0]
                                else None
                                for row, count in enumerate(counts)
                            ],
                        }
                    compare(
                        f"{layer_name}.{branch}.block_table.captured_values",
                        fields[f"{branch}.current.block_table"]["allocated_rows"],
                        fields[f"{branch}.captured.block_table"]["allocated_rows"],
                    )
                metadata_records[layer_name] = record
    unavailable.append(
        "Complete target historical KV read range: sparse topk/compressor indices are produced inside graph; "
        "no KV capacity scanned"
    )
    return {
        "source": "immediately before manager.graphs[desc].replay, after core offloader sync and prepare_attn",
        "request_mapping": mapping,
        "valid_query_range": [0, n],
        "padding_query_range": [n, size],
        "host_expected_query_offsets": offsets,
        "host_sequence_upper_bound": expected_upper,
        "device_inputs": inputs,
        "device_idx_mapping": _record_tensor(batch.idx_mapping, source="input_batch.idx_mapping", rows=requests),
        "captured_inputs": captured_input_records,
        "groups": group_records,
        "attention_metadata": metadata_records,
        "mismatches": mismatches,
        "unavailable": unavailable,
        "match_status": "MISMATCH" if mismatches else "MATCH_FOR_CHECKED_FIELDS_ONLY",
        "ordering": {
            "metadata_updates": (
                "prepare_inputs -> gather/compute slots -> prepare_attn -> DSA persistent "
                "copy_ -> this boundary; current stream"
            ),
            "offloader": "core sync_prev_onload before proxy.replay",
            "DSA_task_handle_updates": "not used: AscendDSAImpl.update_graph_params is a no-op",
            "other_backend_task_handle_updates": "unavailable; diagnostic scope is MRV2 DSA",
            "diagnostic_dependency": (
                "pre-replay small D2H reads wait on current-stream writes; post-replay flag "
                "D2H waits on graph in same stream"
            ),
            "current_stream_id": int(torch.npu.current_stream().npu_stream)
            if batch.input_ids.device.type != "cpu"
            else "CPU mock",
        },
    }


class ReplaySnapshots:
    def __init__(self, diagnostic, manager, bank, window):
        self.diagnostic, self.manager, self.bank = diagnostic, manager, bank
        self.window = tuple(window)
        self.shapes = {}
        self.captured = {}
        self.completed = 0
        self.diagnostic.replay_configuration = {
            "window": list(window),
            "capture_shapes": list(bank.sizes),
            "layer_snapshot_bytes_per_rank": bank.allocated_bytes,
            "completed_detailed_replays": 0,
            "attempted_detailed_replays": 0,
            "unavailable_detailed_replays": 0,
            "performance_eligible": False,
            "perturbation": (
                "Additional compiled copy_ nodes, persistent memory before KV profiling, "
                "pre/post-replay D2H and comparisons. May change reproduction; never performance data."
            ),
        }

    def model_inputs(self, kwargs):
        size = kwargs["input_ids"].shape[0]
        if size not in self.shapes:
            DSparkNaNDiagnostics._outside_capture()
            self.shapes[size] = {
                "sources": {},
                "source_layouts": {},
                "inputs": {},
                "epoch_input": torch.zeros(1, dtype=torch.int32, device=kwargs["input_ids"].device),
                "epoch_output": torch.zeros(1, dtype=torch.int32, device=kwargs["input_ids"].device),
            }
        state = self.shapes[size]
        state["inputs"] = {
            name: value
            for name, value in kwargs.items()
            if name in ("input_ids", "positions") and isinstance(value, torch.Tensor)
        }
        return size

    def model_outputs(self, size, output):
        state = self.shapes[size]
        hidden, aux = output if isinstance(output, tuple) else (output, [])
        values = {"hidden": hidden, **{f"aux.{i}": value for i, value in enumerate(aux)}}
        for name, value in values.items():
            if name not in state["sources"]:
                DSparkNaNDiagnostics._outside_capture()
                state["sources"][name] = torch.empty_like(value)
            # This copy precedes core's persistent output copy in the captured
            # forward closure, not a read of two aliases after replay.
            state["sources"][name].copy_(value)
            state["source_layouts"][name] = tensor_layout(value)
        state["epoch_output"].copy_(state["epoch_input"])

    def finish_capture(self, states):
        for desc, pair in states.items():
            if desc.cg_mode.name != "FULL":
                continue
            if desc.num_tokens not in self.shapes:
                raise RuntimeError("Target diagnostic capture wrapper did not observe a requested graph shape.")
            graph = self.manager.graphs[desc]
            if isinstance(graph, DiagnosticReplayGraph):
                raise RuntimeError("Duplicate replay diagnostic graph installation.")
            self.captured[desc] = pair.captured
            self.manager.graphs[desc] = DiagnosticReplayGraph(graph, self, desc)

    def before(self, desc, graph):
        diagnostic = self.diagnostic
        if not diagnostic.current:
            return None  # capture/profile replay cannot claim an execution epoch
        epoch = diagnostic.execution_epoch
        if not self.window[0] <= epoch <= self.window[1]:
            diagnostic.current["replay_detail"] = {
                "status": "unavailable",
                "reason": "execution outside configured detailed window",
                "window": list(self.window),
            }
            return None
        state = self.shapes[desc.num_tokens]
        diagnostic.replay_configuration["attempted_detailed_replays"] += 1
        detail = {
            "status": "REPLAY_NOT_COMPLETED",
            "epoch": epoch,
            "phase": diagnostic.phase,
            "graph_shape": desc.num_tokens,
            "graph_object_id": id(graph),
            "graph_class": f"{type(graph).__module__}.{type(graph).__name__}",
            "window": list(self.window),
            "layer_snapshot_bytes_per_rank": self.bank.allocated_bytes,
            "output_snapshot_bytes_per_rank": sum(
                t.numel() * t.element_size()
                for s in self.shapes.values()
                for t in (*s["sources"].values(), s["epoch_input"], s["epoch_output"])
            ),
        }
        diagnostic.current["replay_detail"] = detail
        detail["pre_replay"] = inspect_replay_inputs(
            self.manager.model_runner, self.captured[desc], state["inputs"], desc.num_tokens
        )
        prepared = self.manager.model_runner.input_batch
        scheduled = diagnostic.current["scheduled_tokens"]
        mapping = detail["pre_replay"]["request_mapping"]
        actual = {row["request_id"]: row["query_span"][1] - row["query_span"][0] for row in mapping}
        if mapping and scheduled != actual:
            detail["pre_replay"]["mismatches"].append(
                {
                    "field": "scheduler requests/query counts vs prepared input batch",
                    "expected": scheduled,
                    "actual": actual,
                }
            )
            detail["pre_replay"]["match_status"] = "MISMATCH"
        detail["request_ids"] = list(prepared.req_ids)
        detail["valid_query_range"] = [0, prepared.num_tokens]
        # Clear the receipt OUTSIDE graph and arm this execution. Capture data
        # and a failed/no-op replay cannot pass the fresh epoch receipt check.
        state["epoch_output"].fill_(-1)
        state["epoch_input"].fill_(epoch)
        bucket = self.bank.sizes.index(desc.num_tokens)
        for receipt in self.bank.receipts.values():
            receipt[bucket : bucket + 1].fill_(-1)
        self.bank.epoch_input.fill_(epoch)
        return detail

    def unavailable(self, detail, reason, *, failed=False):
        detail.update(status="REPLAY_NOT_COMPLETED" if failed else "unavailable", reason=reason)
        configuration = self.diagnostic.replay_configuration
        configuration["unavailable_detailed_replays"] += 1
        configuration.setdefault(
            "first_unavailable_detail",
            {"epoch": detail["epoch"], "phase": detail["phase"], "reason": reason},
        )
        self.diagnostic._write(window_snapshot=True)

    def after(self, desc, detail):
        if detail is None:
            return
        state = self.shapes[desc.num_tokens]
        receipt = _host(state["epoch_output"])
        detail["replay_epoch_receipt"] = receipt
        if receipt != [detail["epoch"]]:
            self.unavailable(detail, "captured snapshot nodes did not publish this replay epoch")
            return
        bucket = self.bank.sizes.index(desc.num_tokens)
        layer_receipts = torch.stack([value[bucket] for value in self.bank.receipts.values()]).cpu().tolist()
        missing = [name for name, epoch in zip(self.bank.receipts, layer_receipts) if epoch != detail["epoch"]]
        detail["missing_replay_boundaries"] = missing
        if missing:
            self.unavailable(detail, "compiled layer snapshot writes did not execute for this epoch")
            return
        if detail["pre_replay"]["match_status"] == "unavailable":
            self.unavailable(detail, "pre-replay input ABI coverage is incomplete")
            return
        n = self.manager.model_runner.input_batch.num_tokens
        boundaries = finite_summaries(self.bank.views(desc.num_tokens), n)
        destinations = {
            "hidden": self.manager.hidden_states[: desc.num_tokens],
            **{f"aux.{i}": value[: desc.num_tokens] for i, value in enumerate(self.manager.aux_hidden_states)},
        }
        source_stats = finite_summaries(state["sources"], n)
        destination_stats = finite_summaries(destinations, n)
        transfers = {}
        for name, source in state["sources"].items():
            destination = destinations[name]
            # NaN == NaN for transfer comparison only: NaNs remain failures in
            # finite_summaries; no tensor content is repaired or replaced.
            same = (
                ((source[:n] == destination[:n]) | (torch.isnan(source[:n]) & torch.isnan(destination[:n])))
                .reshape(n, -1)
                .all(1)
                .cpu()
                .tolist()
            )
            transfers[name] = {
                "source": source_stats[name],
                "destination": destination_stats[name],
                "different_rows": [i for i, equal in enumerate(same) if not equal],
                "original_source_layout_at_capture": state["source_layouts"][name],
                "source_ownership": "copy_ snapshot before core output transfer; separate strong buffer per shape",
                "destination_ownership": "core ModelCudaGraphManager persistent maximum-shape output prefix",
            }
        # The pre-HC snapshot is taken before the model's own copy_.
        pre_hc = self.manager.model_runner.model.model._mtp_hidden_buffer[:n]
        pre_source = self.bank.views(desc.num_tokens)["pre_hc"][:n].flatten(1)
        same = ((pre_source == pre_hc) | (torch.isnan(pre_source) & torch.isnan(pre_hc))).all(1).cpu().tolist()
        transfers["pre_hc"] = {
            "source": finite_summaries({"source": pre_source}, n)["source"],
            "destination": finite_summaries({"destination": pre_hc}, n)["destination"],
            "different_rows": [i for i, equal in enumerate(same) if not equal],
            "source_ownership": "layer bank before _mtp_hidden_buffer.copy_",
            "destination_ownership": "target pre-HC persistent buffer; flattened HC residual",
        }
        detail.update(
            status="ACTUAL_FULL_REPLAY_SNAPSHOTS",
            boundaries=boundaries,
            output_transfers=transfers,
            localization=localize(boundaries, transfers),
        )
        self.completed += 1
        self.diagnostic.replay_configuration["completed_detailed_replays"] = self.completed
        # Preserve the last detailed round and its prior two even if generation
        # continues outside the window. This never replaces first-failure.
        self.diagnostic._write(window_snapshot=True)


class DiagnosticReplayGraph:
    """Instance-only proxy at the frozen MRV2 graph.replay ABI; no core patch."""

    def __init__(self, graph, snapshots, desc):
        self.graph, self.snapshots, self.desc = graph, snapshots, desc

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def replay(self):
        detail = self.snapshots.before(self.desc, self.graph)
        try:
            result = self.graph.replay()
        except BaseException as error:
            if detail is not None:
                self.snapshots.unavailable(detail, f"{type(error).__name__}: {error}", failed=True)
            raise
        self.snapshots.after(self.desc, detail)
        return result
