# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One selected SWA layer: actual graph prolog, KV window and projection cuts."""

import torch

ATTENTION_STAGES = (
    "rope_cos",
    "rope_sin",
    "sink",
    "q_normalized",
    "q_rope",
    "kv_normalized",
    "kv_rope",
    "kv_window",
    "raw_attention",
    "inverse_rope",
    "wo_a",
    "wo_b_local",
)
MAX_WINDOW = 128
STATE_COLUMNS = (
    "request_row",
    "position",
    "window_start",
    "window_end",
    "slot_block",
    "slot_offset",
    "expected_block",
    "expected_offset",
    "invalid_window_indices",
    "slot_mismatch",
)


class AttentionProbe:
    def __init__(self, bank, layer):
        self.bank = bank
        self.prefix = f"layer.{layer}.attention."
        self.routes = {}
        self.state = torch.zeros((bank.max_tokens, len(STATE_COLUMNS)), dtype=torch.int64, device=bank.flags.device)

    def write(self, stage, value):
        name = self.prefix + stage
        # DSA's opaque custom op executes Python while capturing real NPU ops;
        # this layout is static capture provenance, not a Python replay receipt.
        self.bank.tails[name] = list(value.shape[1:])
        self.bank.write(name, value)

    def inputs(self, impl, layer_name, hidden, cache, metadata):
        if impl.compress_ratio > 1 or not 0 < impl.window_size <= MAX_WINDOW:
            raise ValueError("Attention detail supports only one SWA layer with window <= 128")
        self.routes[hidden.shape[0]] = {
            "layer_name": layer_name,
            "implementation": type(impl).__qualname__,
            "branch": "decode_swa",
            "multistream": impl.multistream_dsv4_dsa_overlap,
            "compress_ratio": impl.compress_ratio,
            "window": impl.window_size,
            "block_size": metadata.block_size,
            "kv_shape": list(cache.shape),
            "slot_shape": list(metadata.slot_mapping.shape),
            "block_table_shape": list(metadata.block_table.shape),
            "mask": {"ori_mask_mode": 4, "left": impl.window_size - 1, "right": 0},
            "operator": "_C_ascend.npu_sparse_attn_sharedkv",
            "quant_methods": {
                key: type(getattr(getattr(impl, key).quant_method, "quant_method", None)).__qualname__
                for key in ("wq_a", "wq_b", "wkv", "wo_b")
            },
            "wo_b_custom_op": type(getattr(impl.wo_b, "custom_op", None)).__qualname__,
            "wo_b_tp_size": impl.wo_b.tp_size,
        }
        self.write("rope_cos", metadata.cos[layer_name])
        self.write("rope_sin", metadata.sin[layer_name])
        self.write("sink", impl.attn_sink.reshape(1, -1).expand(hidden.shape[0], -1))

    def window(self, cache, metadata, tokens, window):
        """Read the contract's causal SWA window before attention, not all KV.

        Invalid indices use safe reads ONLY for diagnostics and are reported.
        This never modifies the original kernel's indices, cache or mask.
        It does not prove the native kernel obeyed this semantic read range.
        """
        if len(cache.shape) != 4 or cache.shape[2] != 1 or metadata.slot_mapping.shape[-1] != 2:
            raise ValueError("Attention detail requires non-A5 PA_ND KV and block/offset slots")
        n = min(tokens, self.bank.max_tokens)
        rows = torch.arange(n, device=cache.device, dtype=torch.int64)
        starts = metadata.query_start_loc.to(torch.int64)
        table = metadata.block_table
        req = (rows[:, None] >= starts[None, 1:]).sum(1)
        safe_req = req.clamp(0, table.shape[0] - 1)
        end = starts[safe_req + 1]
        seq = metadata.seq_lens[safe_req].to(torch.int64)
        position = seq - (end - rows)
        valid = (rows < starts[-1]) & (req < table.shape[0])
        positions = position[:, None] - torch.arange(window - 1, -1, -1, device=cache.device)
        needed = valid[:, None] & (positions >= 0) & (positions < seq[:, None])
        logical = torch.div(positions, cache.shape[1], rounding_mode="floor")
        pages = table[safe_req[:, None], logical.clamp(0, table.shape[1] - 1)].to(torch.int64)
        bad = (logical < 0) | (logical >= table.shape[1]) | (pages < 0) | (pages >= cache.shape[0])
        values = cache[pages.clamp(0, cache.shape[0] - 1), positions.remainder(cache.shape[1])]
        # Mask unneeded values before reduction: unused cache may legitimately
        # contain NaN, and padding has no request identity.
        values = torch.where(needed[:, :, None, None], values, 0)
        self.write("kv_window", values.flatten(1))
        slots = metadata.slot_mapping[:n].to(torch.int64)
        expected = torch.stack((pages[:, -1], position.remainder(cache.shape[1])), 1)
        mismatch = valid & (slots != expected).any(1)
        state = torch.stack(
            (
                req,
                position,
                (position - window + 1).clamp_min(0),
                position + 1,
                slots[:, 0],
                slots[:, 1],
                expected[:, 0],
                expected[:, 1],
                (needed & bad).sum(1),
                mismatch.to(torch.int64),
            ),
            1,
        )
        self.state[:n].copy_(torch.where(valid[:, None], state, -1))


def install_attention_probe(bank, model, layer):
    # Installation is opt-in and before compile/capture; no new wrapper framework.
    from vllm_ascend.attention.dsa_v1 import AscendDSAImpl
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, olora_tp_enable, oproj_tp_enable

    impl = model.layers[layer].self_attn.dsa_attn.dsa_attn.impl
    if (
        type(impl) is not AscendDSAImpl
        or get_ascend_device_type() == AscendDeviceType.A5
        or impl.compress_ratio > 1
        or oproj_tp_enable()
        or olora_tp_enable()
        or getattr(impl.wo_b, "custom_op", None) is not None
    ):
        raise ValueError("Attention detail requires non-A5 SWA, standard row-TP output projection and no CP")
    probe = AttentionProbe(bank, layer)
    impl._dspark_attn_probe = probe
    impl.wo_b._dspark_attn_probe = probe
    bank.attention_probe = probe
