"""Feed vLLM's decode tensors to the PyPTO attention-only CSA kernel.

The kernel is ``pto_kernels.dspark.decode_csa.decode_csa_attn_tp1_test``: it takes
``x_normed [T, D] BF16`` and fills ``attn_out [T, D] BF16``, leaving ``npu_hc_pre``,
the input RMSNorm and ``npu_hc_post`` to the native path. Everything here is the
translation between vLLM's decode state and that kernel's 46 arguments.

Two properties the per-step path must keep, because it has to survive ACLGraph
capture: no device-to-host read (no ``.item()``, no boolean-mask indexing, no
``torch.unique``), and no allocation whose shape depends on a device value.

``PTO_ATTN_COMPARE=<dir>`` runs one decode step through both the native path and
this kernel and writes the comparison; it is a diagnostic, never a serving path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

# --- kernel import -----------------------------------------------------------
# decode_csa fixes its TP specialization at import time from sys.argv, which a
# vLLM process never carries. Inject it so B = DECODE_BATCH // TP and T = B * S
# land on the intended shape instead of the module's own default of 2.
def _env_int(name: str, default: int) -> int:
    """The launcher forwards unset switches as empty strings, so a present-but-empty
    variable has to fall back the same way an absent one does."""
    return int(os.environ.get(name, "") or default)


_TP = _env_int("PTO_ATTN_TP", 4)


def _import_kernel():
    argv = sys.argv
    if not any(a == "--tp" or a.startswith("--tp=") for a in argv):
        sys.argv = [*argv, "--tp", str(_TP)]
    try:
        from .pto_kernels.dspark import config as kcfg
        from .pto_kernels.dspark import decode_csa as kcsa
    finally:
        sys.argv = argv
    return kcsa, kcfg


_KCSA = None
_KCFG = None


def kernel():
    global _KCSA, _KCFG
    if _KCSA is None:
        _KCSA, _KCFG = _import_kernel()
    return _KCSA, _KCFG


# --- constants ---------------------------------------------------------------

VLLM_PAGE = 128          # swa / compressed / indexer KV page, in slots
VLLM_STATE_PAGE = 8      # both compressor state caches, in rows
COMPRESS_RATIO = 4


def _enabled(name: str) -> str:
    return os.environ.get(name, "").strip()


# --- weights: one-time, cached on the impl -----------------------------------


def _to_nd(w: torch.Tensor) -> torch.Tensor:
    """Undo the FRACTAL_NZ layout vLLM applies to quantized weights.

    ``weight_nz_mode`` defaults to 1, and a NZ-laid-out INT8 weight read as ND is
    silently wrong rather than an error, so this runs unconditionally.
    """
    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_ND

    return torch_npu.npu_format_cast(w, ACL_FORMAT_ND)


def _quant_int8_per_channel(w: torch.Tensor, kcfg):
    """BF16 -> INT8 plus per-channel scale, matching the kernel's contract."""
    amax = w.float().abs().amax(dim=-1).clamp_min(kcfg.INT8_AMAX_EPS)
    sq = kcfg.INT8_SCALE_MAX / amax
    q = torch.round(w.float() * sq.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q, (1.0 / sq).float()


def _oriented(w: torch.Tensor, want: tuple) -> torch.Tensor:
    """Give the kernel its orientation, whichever one vLLM stored.

    A W8A8 linear comes out of process_weights_after_loading already transposed to
    [in, out], while an unquantized one keeps torch's [out, in]. Deciding by shape
    rather than by quantization keeps both checkpoints working.
    """
    shape = tuple(w.shape)
    if shape == want:
        return w
    if shape == want[::-1]:
        return w.t().contiguous()
    raise ValueError(f"weight is {shape}, expected {want} or its transpose")


def _dense(linear, want: tuple) -> torch.Tensor:
    """A BF16 weight in the kernel's orientation, dequantizing if vLLM quantized it."""
    w = _to_nd(linear.weight.detach())
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if scale is not None and w.dtype in (torch.int8, torch.uint8):
        w = w.float() * scale.detach().float().view(1, -1)
    return _oriented(w.to(torch.bfloat16), want)


def _int8(linear, want: tuple, kcfg):
    """An INT8 weight plus per-channel scale, quantizing if vLLM kept it dense.

    The kernel's signature fixes INT8 for wq_b / idx_wq_b / wo_b, so an
    unquantized checkpoint has to be quantized here. That is a real numerical
    difference, not an adapter artifact.
    """
    w = _to_nd(linear.weight.detach())
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if w.dtype in (torch.int8, torch.uint8) and scale is not None:
        return _oriented(w, want), scale.detach().float()
    q, sc = _quant_int8_per_channel(_oriented(w, want).t().contiguous(), kcfg)
    return q.t().contiguous(), sc


def _hadamard(dim: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Sylvester Hadamard, normalized.

    The kernel treats this as a bare right-hand matmul operand with no scaling of
    its own, so the 1/sqrt(dim) that vLLM's rotate_activation applies separately
    has to be folded in here. Mirrors decode_csa.py::init_hadamard_idx.
    """
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return (h / (dim**0.5)).to(dtype).to(device)


def prepare_weights(impl):
    """Build and cache the kernel's weight arguments on the impl."""
    cached = getattr(impl, "_pto_attn_weights", None)
    if cached is not None:
        return cached

    kcsa, kcfg = kernel()
    D, QL, HD = kcsa.D, kcsa.Q_LORA, kcsa.HEAD_DIM
    IH, ID = kcsa.IDX_N_HEADS, kcsa.IDX_HEAD_DIM
    wq_b, wq_b_scale = _int8(impl.wq_b, (QL, kcsa.H * HD), kcfg)
    idx_wq_b, idx_wq_b_scale = _int8(impl.inderxer_wq_b, (QL, IH * ID), kcfg)
    wo_b, wo_b_scale = _int8(impl.wo_b, (D, kcsa.O_GROUPS * kcsa.O_LORA), kcfg)

    w = {
        "wq_a": _dense(impl.wq_a, (D, QL)),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": _dense(impl.wkv, (D, HD)),
        "gamma_cq": impl.q_norm.weight.detach().to(torch.bfloat16),
        "gamma_ckv": impl.kv_norm.weight.detach().to(torch.bfloat16),
        "cmp_wkv": _dense(impl.compressor_wkv, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_wgate": _dense(impl.compressor_wgate, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_ape": impl.compressor_ape.detach().float(),
        "cmp_norm_w": impl.compressor_norm.weight.detach().to(torch.bfloat16),
        "idx_wq_b": idx_wq_b,
        "idx_wq_b_scale": idx_wq_b_scale,
        "weights_proj": _dense(impl.weights_proj, (D, IH)),
        "hadamard_idx": _hadamard(ID, impl.wo_b.weight.device),
        "inner_wkv": _dense(impl.indexcom_wkv, (kcsa.INNER_OUT_DIM, D)),
        "inner_wgate": _dense(impl.indexcom_wgate, (kcsa.INNER_OUT_DIM, D)),
        "inner_ape": impl.indexcom_ape.detach().float(),
        "inner_norm_w": impl.indexcom_norm.weight.detach().to(torch.bfloat16),
        "attn_sink": impl.attn_sink.detach().float(),
        # vLLM keeps [G, O_GROUP_IN, O_LORA]; the kernel wants the transpose.
        "wo_a": impl.wo_a.weight.detach().transpose(1, 2).contiguous().to(torch.bfloat16),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
    }
    impl._pto_attn_weights = w
    return w


# --- per-step derivation: everything below must stay device-only ------------


def _rope_to_pto(t: torch.Tensor) -> torch.Tensor:
    """vLLM's interleaved RoPE table -> the kernel's 'first half real, second half copied'.

    vLLM emits ``[c0, c0, c1, c1, ...]`` over 64 lanes, so only 32 frequencies are
    distinct; the kernel reads ``[:, :HALF_ROPE]`` and expects those 32 followed by
    a copy of themselves.
    """
    flat = t.reshape(t.shape[0], -1)
    uniq = flat[:, 0::2][:, :32]
    return torch.cat([uniq, uniq], dim=1).to(torch.bfloat16)


def _flat_slots(sm_2d: torch.Tensor, page: int) -> torch.Tensor:
    """vLLM's (block, intra) INT32 pair -> the kernel's flat INT64 row index."""
    return sm_2d[:, 0].long() * page + sm_2d[:, 1].long()


def _repage_block_table(bt: torch.Tensor, n_cols: int, per: int, n_blocks: int) -> torch.Tensor:
    """Restate a 128-slot-page block table over ``per``-times-smaller pages.

    Fixed column count on purpose: deriving the width from the table's contents
    would need a device read, which capture forbids.
    """
    b = bt.shape[0]
    j = torch.arange(n_cols, device=bt.device)
    src = torch.div(j, per, rounding_mode="floor").clamp(max=bt.shape[1] - 1)
    phys = (
        torch.gather(bt.long(), 1, src.unsqueeze(0).expand(b, -1)) * per
        + (j % per).unsqueeze(0)
    )
    return phys.clamp(0, n_blocks - 1).to(torch.int32)


def _window_indices(positions: torch.Tensor, swa_bt: torch.Tensor, seq: int, win: int) -> torch.Tensor:
    """Per-token physical KV rows for the sliding window, -1 where unmapped.

    The row is ordered oldest-to-current with the padding on the left; the kernel
    only takes a row-wise max over the validity mask, so the side the padding sits
    on does not matter.
    """
    dev = positions.device
    t = positions.shape[0]
    req = (torch.arange(t, device=dev) // seq).clamp(max=swa_bt.shape[0] - 1)
    bt_t = swa_bt.long().index_select(0, req)

    offs = torch.arange(win, device=dev) - (win - 1)
    abs_pos = positions.long().unsqueeze(1) + offs.unsqueeze(0)
    lblk = torch.div(abs_pos, VLLM_PAGE, rounding_mode="floor")
    intra = abs_pos - lblk * VLLM_PAGE
    ok = (abs_pos >= 0) & (lblk >= 0) & (lblk < bt_t.shape[1])
    pblk = torch.gather(bt_t, 1, lblk.clamp(0, bt_t.shape[1] - 1))
    ok = ok & (pblk >= 0)
    return torch.where(
        ok, pblk.clamp_min(0) * VLLM_PAGE + intra, torch.full_like(abs_pos, -1)
    ).to(torch.int32)


def _compressed_rows(positions: torch.Tensor):
    """Which compressed row each token maps to, and whether it is a boundary.

    vLLM packs the compressed tables by boundary token; the kernel indexes them by
    token. ``cumsum`` gives the mapping without a device read.
    """
    pos = positions.long()
    boundary = ((pos + 1) % COMPRESS_RATIO) == 0
    row = (torch.cumsum(boundary.long(), 0) - 1).clamp_min(0)
    return boundary, row


def _to_token_rows(src: torch.Tensor, row: torch.Tensor, boundary: torch.Tensor, fill):
    g = src.index_select(0, row.clamp(max=src.shape[0] - 1))
    return torch.where(boundary, g, torch.full_like(g, fill))


def _state_slots(positions: torch.Tensor, state_bt: torch.Tensor, seq: int,
                 logical_page: int, physical_page: int) -> torch.Tensor:
    """Per-token compressor-state rows.

    vLLM never materializes these: its compressor op takes the block table plus a
    start position and resolves slots inside the closed-source kernel. The formula
    is the ordinary paged lookup -- no modulo, no ring -- so wrap-around follows
    whatever vLLM's allocator wrote into the block table.

    ``logical_page`` is vLLM's rows-per-page; ``physical_page`` is the stride of the
    view handed to the kernel, which differs when the cache is reinterpreted to
    absorb vLLM's page padding.
    """
    dev = positions.device
    t = positions.shape[0]
    req = (torch.arange(t, device=dev) // seq).clamp(max=state_bt.shape[0] - 1)
    bt_t = state_bt.long().index_select(0, req)

    pos = positions.long()
    lblk = torch.div(pos, logical_page, rounding_mode="floor")
    intra = pos - lblk * logical_page
    ok = (lblk >= 0) & (lblk < bt_t.shape[1])
    blk = torch.gather(bt_t, 1, lblk.clamp(0, bt_t.shape[1] - 1))
    ok = ok & (blk >= 0)
    return torch.where(
        ok, blk * physical_page + intra, torch.full_like(pos, -1)
    ).to(torch.int64)


# --- compressor state: a private ring per request ----------------------------
#
# Two things rule out handing vLLM's state cache to the kernel directly.
#
# Layout: vLLM pads each state page to a fixed byte size and builds the cache
# with torch.as_strided, so the tensor is non-contiguous -- main carries 65536 B
# of content in a 131072 B stride, inner 16384 B in 16640 B. PyPTO's binding
# rejects any tensor whose strides are not canonical row-major, in three
# independent places, the innermost being torch_npu_adapter.cpp's
# Require(tensor.is_contiguous()).
#
# Addressing: the kernel's state is a per-request ring of STATE_STORAGE_LEN rows
# reached through a fixed-width block table of 2-row pages --
#   ring_row  = logical_pos % 16                       (ratio4:191)
#   state_row = bt[req, ring_row // 2] * 2 + ring_row % 2   (ratio4:192-197)
# while vLLM's block table holds absolute page indices with null_block == 0 for
# holes. Feeding vLLM's table in would make every request address pages 0..7,
# i.e. positions 0..63, and then collide on the null block -- silently.
#
# So the kernel gets its own contiguous ring, seeded from vLLM's cache before the
# call and written back after. Sixteen consecutive positions cover the ring
# exactly once, which makes the seed a single gather with no device read.

KERNEL_STATE_PAGE = 2   # C4A_COMPRESSOR_BLOCK_SIZE


def state_ring_len() -> int:
    """STATE_STORAGE_LEN: the history window plus this step's S rows."""
    kcsa, _ = kernel()
    return kcsa.MAIN_STATE_STORAGE_LEN


def state_ring_plan(positions: torch.Tensor, seq: int, vllm_bt: torch.Tensor):
    """Which vLLM page and row seed which ring row, for every request.

    Returns ``(blk, intra, ring_rows, valid)``, each ``[B * _RR]``. Pages and
    rows stay separate because vLLM's cache is strided on dim 0: selecting whole
    pages keeps the copy proportional to the ring, while flattening it to rows
    first would materialize the entire cache.
    """
    dev = positions.device
    _RR = state_ring_len()
    b = positions.shape[0] // seq
    first = positions.long().view(b, seq)[:, 0]                    # [B]

    i = torch.arange(_RR, device=dev)
    pos = first.unsqueeze(1) - (_RR - seq) + i.unsqueeze(0)  # [B, _RR]

    lblk = torch.div(pos, VLLM_STATE_PAGE, rounding_mode="floor")
    intra = pos - lblk * VLLM_STATE_PAGE
    ok = (pos >= 0) & (lblk >= 0) & (lblk < vllm_bt.shape[1])
    blk = torch.gather(vllm_bt.long(), 1, lblk.clamp(0, vllm_bt.shape[1] - 1))
    ok = ok & (blk >= 0)

    ring = (torch.arange(b, device=dev).unsqueeze(1) * _RR + (pos % _RR))
    return (blk.clamp_min(0).reshape(-1), intra.clamp_min(0).reshape(-1),
            ring.reshape(-1), ok.reshape(-1))


def _pick_rows(cache: torch.Tensor, blk: torch.Tensor, intra: torch.Tensor, dim: int):
    """One row per (page, offset) pair, as a contiguous [k, dim]."""
    pages = cache.index_select(0, blk.clamp(0, cache.shape[0] - 1))   # [k, rows, ..., dim]
    pages = pages.reshape(pages.shape[0], VLLM_STATE_PAGE, dim)
    return pages[torch.arange(blk.shape[0], device=cache.device), intra]


def make_state_ring(cache: torch.Tensor, plan, b: int, dim: int):
    """Seed a contiguous per-request ring from vLLM's padded, strided cache."""
    _RR = state_ring_len()
    blk, intra, ring_rows, valid = plan
    rows = _pick_rows(cache, blk, intra, dim)
    rows = torch.where(valid.unsqueeze(1), rows, torch.zeros_like(rows))

    ring = torch.zeros(b * _RR, dim, dtype=cache.dtype, device=cache.device)
    ring.index_copy_(0, ring_rows, rows)
    return ring.view(b * _RR // KERNEL_STATE_PAGE, KERNEL_STATE_PAGE, dim)


def write_state_ring(cache: torch.Tensor, ring: torch.Tensor, plan, seq: int,
                     positions: torch.Tensor, dim: int) -> None:
    """Return the ring's rows to vLLM's cache.

    The pages are addressed by construction rather than by de-duplicating the
    plan's page list: several ring rows share a vLLM page, and ``index_copy_``
    writes whole pages, so duplicate page indices would make the last entry's
    copy overwrite the earlier entries' edits. A request's ring spans
    ``RING/VSP + 1`` consecutive logical pages, so gathering exactly those gives a
    duplicate-free index without a device read.
    """
    blk, intra, ring_rows, valid = plan
    rr = state_ring_len()
    dev = cache.device
    b = positions.shape[0] // seq
    span = rr // VLLM_STATE_PAGE + 1

    first = positions.long().view(b, seq)[:, 0]
    base = torch.div(first - (rr - seq), VLLM_STATE_PAGE, rounding_mode="floor")
    lb = base.unsqueeze(1) + torch.arange(span, device=dev).unsqueeze(0)    # [B, span]

    pages_bt = blk.view(b, rr)
    lblk_of = torch.div(
        (first.unsqueeze(1) - (rr - seq) + torch.arange(rr, device=dev).unsqueeze(0)),
        VLLM_STATE_PAGE, rounding_mode="floor")                            # [B, rr]
    # Physical page for each of the span slots, taken from the row that uses it.
    slot_of_row = (lblk_of - base.unsqueeze(1)).clamp(0, span - 1)          # [B, rr]
    phys = torch.zeros(b, span, dtype=torch.long, device=dev)
    phys.scatter_(1, slot_of_row, pages_bt)

    flat_pages = phys.reshape(-1).clamp(0, cache.shape[0] - 1)
    buf = cache.index_select(0, flat_pages).reshape(b * span, VLLM_STATE_PAGE, dim).clone()

    rows = ring.reshape(-1, dim).index_select(0, ring_rows)
    tgt_page = (torch.arange(b, device=dev).unsqueeze(1) * span + slot_of_row).reshape(-1)
    keep = buf[tgt_page, intra]
    buf[tgt_page, intra] = torch.where(valid.unsqueeze(1), rows, keep)

    cache.index_copy_(0, flat_pages, buf.reshape(-1, *cache.shape[1:]))


def state_block_table(b: int, device) -> torch.Tensor:
    """The ring's own block table: request r owns its whole slice of pages."""
    pages = state_ring_len() // KERNEL_STATE_PAGE
    return (torch.arange(b, device=device).unsqueeze(1) * pages
            + torch.arange(pages, device=device).unsqueeze(0)).to(torch.int32)


def state_slots(positions: torch.Tensor, seq: int) -> torch.Tensor:
    """Flat ring rows the kernel writes this step, one per token.

    Matches decode_metadata.py:225-248 with the ring's own block table folded in:
    ``bt[r, (pos//2) % 8] * 2 + pos % 2`` reduces to ``r*16 + pos % 16``.
    """
    dev = positions.device
    t = positions.shape[0]
    req = torch.arange(t, device=dev) // seq
    pos = positions.long()
    rr = state_ring_len()
    return (req * rr + (pos % rr)).to(torch.int64)


def repage_kv(cache: torch.Tensor):
    """Restate a 128-slot KV cache as the kernel's 32-slot pages, as a view."""
    return cache.view(-1, VLLM_PAGE // COMPRESS_RATIO, *cache.shape[2:])


# --- structure probe ---------------------------------------------------------


def describe(obj, depth: int = 0, limit: int = 3):
    """Shape/dtype sketch of a metadata object, for the first-call dump."""
    if isinstance(obj, torch.Tensor):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype),
                "contig": bool(obj.is_contiguous()), "stride": list(obj.stride())}
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [describe(x, depth + 1, limit) for x in obj[:8]]
    if isinstance(obj, dict):
        return {str(k): describe(v, depth + 1, limit) for k, v in list(obj.items())[:24]}
    if depth < limit:
        out = {"__class__": type(obj).__name__}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            if callable(v):
                continue
            out[name] = describe(v, depth + 1, limit)
        return out
    return {"__class__": type(obj).__name__}


def dump_structure(path, hidden_states, kv_cache, attn_metadata, impl) -> None:
    payload = {
        "hidden_states": describe(hidden_states),
        "kv_cache": [describe(c, 2) for c in kv_cache],
        "attn_metadata": describe(attn_metadata, 0, 4),
        "impl_shapes": {
            k: describe(getattr(impl, k, None), 2)
            for k in ("wq_a", "wq_b", "wkv", "q_norm", "kv_norm", "wo_a", "wo_b",
                      "attn_sink", "weights_proj", "inderxer_wq_b",
                      "compressor_wkv", "compressor_wgate", "compressor_ape", "compressor_norm",
                      "indexcom_wkv", "indexcom_wgate", "indexcom_ape", "indexcom_norm")
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --- the 46 arguments --------------------------------------------------------

ARG_ORDER = (
    "x_normed",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state", "compress_state_block_table",
    "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
    "inner_compress_state", "inner_compress_state_block_table",
    "kv_cache", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "ori_slot_mapping", "window_swa_indices",
    "cmp_slot_mapping", "idx_slot_mapping",
    "state_slot_mapping", "inner_state_slot_mapping",
    "position_ids", "kv_seq_lens", "attn_sink",
    "wo_a", "wo_b", "wo_b_scale",
    "attn_out",
)


def rectangular(n_real: int, kernel_seq: int, device):
    """Where each of the kernel's T slots takes its data from.

    The kernel's token-to-request map is a compile-time ``t // S``
    (decode_sparse_attn_csa.py:202), and S cannot simply be lowered to match a
    host that submits one token per request: T_PAD would drop below the 128-row
    O-B tile and decode_o_proj refuses to build. So a host token occupies slot
    ``r * S`` and the other S-1 slots of that request are padding, kept inert by
    writing -1 into every slot mapping.

    Returns ``(src, real)``: ``src[t]`` is the host row slot ``t`` reads, and
    ``real[t]`` marks the slots that are not padding.
    """
    t = n_real * kernel_seq
    idx = torch.arange(t, device=device)
    req = torch.div(idx, kernel_seq, rounding_mode="floor")
    real = (idx - req * kernel_seq) == 0
    return req, real


def build_args(impl, hidden_states, kv_cache, metadata_list, seq: int):
    """Translate one decode step into the kernel's 46 arguments.

    ``metadata_list`` is what ``filter_metadata`` returns for a ratio-4 layer:
    five per-cache metadata objects sorted by key -- attn, compressor state,
    indexer-compressor state, indexer k, sliding window.
    """
    kcsa, kcfg = kernel()
    cmp_md, cst_md, ist_md, idx_md, swa_md = (m.decode for m in metadata_list)
    cmp_kv_c, swa_kv_c, state_c, ist_c, idx_k_c, idx_s_c = kv_cache

    layer = impl.layer_name if hasattr(impl, "layer_name") else None
    host_pos = metadata_list[0].decode.input_positions
    n_real = host_pos.shape[0] // seq            # host requests this step
    ks = kcsa.S                                  # the kernel's compile-time S
    if n_real > kcsa.B:
        raise ValueError(f"{n_real} requests exceed the kernel's B={kcsa.B}")

    src, real = rectangular(n_real, ks, host_pos.device)
    b, t = n_real, n_real * ks
    pos = host_pos.index_select(0, src * seq)    # [T], padding repeats its request

    a = dict(prepare_weights(impl))
    a["x_normed"] = hidden_states.index_select(0, src * seq).to(torch.bfloat16)
    a["attn_out"] = torch.empty(t, hidden_states.shape[-1],
                                dtype=torch.bfloat16, device=hidden_states.device)

    # RoPE: vLLM keeps the compressed table packed by boundary row.
    boundary, row = _compressed_rows(pos)
    # The per-token tables are indexed by host row, so they follow the same
    # rectangle as x_normed; the compressed ones are indexed by boundary row and
    # are expanded through `row` below.
    a["freqs_cos"] = _rope_to_pto(_pick(cmp_md.cos, layer)).index_select(0, src * seq)
    a["freqs_sin"] = _rope_to_pto(_pick(cmp_md.sin, layer)).index_select(0, src * seq)
    cc = _rope_to_pto(_pick(cmp_md.compress_cos, layer))
    cs = _rope_to_pto(_pick(cmp_md.compress_sin, layer))
    rc = row.clamp(max=cc.shape[0] - 1)
    a["cmp_freqs_cos"] = cc.index_select(0, rc)
    a["cmp_freqs_sin"] = cs.index_select(0, rc)

    # Paged KV: the kernel's page is a quarter of vLLM's.
    per = COMPRESS_RATIO
    a["kv_cache"] = repage_kv(swa_kv_c)
    a["cmp_kv"] = repage_kv(cmp_kv_c)
    a["idx_kv_cache"] = repage_kv(idx_k_c)
    a["idx_kv_scale"] = repage_kv(idx_s_c)
    a["cmp_block_table"] = _repage_block_table(
        cmp_md.block_table, kcsa.CMP_MAX_BLOCKS, per, a["cmp_kv"].shape[0])
    a["idx_block_table"] = _repage_block_table(
        idx_md.block_table, kcsa.IDX_MAX_BLOCKS, per, a["idx_kv_cache"].shape[0])

    # Compressor state: a private ring per request, seeded from vLLM's cache.
    main_dim = kcsa.MAIN_STATE_DIM
    inner_dim = kcsa.INNER_STATE_DIM
    plan_m = state_ring_plan(pos, ks, cst_md.block_table)
    plan_i = state_ring_plan(pos, ks, ist_md.block_table)
    a["compress_state"] = make_state_ring(state_c, plan_m, b, main_dim)
    a["inner_compress_state"] = make_state_ring(ist_c, plan_i, b, inner_dim)
    a["compress_state_block_table"] = state_block_table(b, pos.device)
    a["inner_compress_state_block_table"] = state_block_table(b, pos.device)
    a["state_slot_mapping"] = _inert_later = state_slots(pos, ks)
    a["inner_state_slot_mapping"] = a["state_slot_mapping"]

    # Slots and window.
    def _inert(x):
        return torch.where(real, x, torch.full_like(x, -1))

    a["ori_slot_mapping"] = _inert(
        _flat_slots(swa_md.slot_mapping, VLLM_PAGE).index_select(0, src * seq))
    a["cmp_slot_mapping"] = _inert(_to_token_rows(
        _flat_slots(cmp_md.slot_mapping, VLLM_PAGE), row, boundary, -1))
    a["idx_slot_mapping"] = _inert(_to_token_rows(
        _flat_slots(idx_md.slot_mapping, VLLM_PAGE), row, boundary, -1))
    a["window_swa_indices"] = _window_indices(pos, swa_md.block_table, ks, kcsa.WIN)

    a["position_ids"] = pos.to(torch.int32)
    a["kv_seq_lens"] = cmp_md.seq_lens.to(torch.int32)[:b]
    a["state_slot_mapping"] = _inert(a["state_slot_mapping"])
    a["inner_state_slot_mapping"] = a["state_slot_mapping"]

    return [a[name] for name in ARG_ORDER], (plan_m, plan_i, state_c, ist_c, main_dim, inner_dim)


def _pick(table, layer):
    """vLLM keeps some RoPE tables per layer in a dict."""
    if isinstance(table, dict):
        return table[layer] if layer in table else next(iter(table.values()))
    return table


# --- one-shot comparison -----------------------------------------------------

_OP = None
_DONE: set = set()


def _registered():
    global _OP
    if _OP is None:
        from pypto.torch import init, register

        kcsa, _ = kernel()
        init()
        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
    return _OP


def compare_once(self, hidden_states, kv_cache, metadata_list, native_out, out_dir: str) -> None:
    """Run the kernel on this step's real tensors and record how it compares.

    The kernel writes six caches, so this runs after the native path and reads
    caches the native path has already advanced: the output is not numerically
    comparable, and is not meant to be. What it answers is whether the 46
    arguments assemble, bind and execute on live vLLM state at all.
    """
    impl = self.dsa_attn.impl
    layer = self.dsa_attn.layer_name
    rec = {"layer": layer, "stage": "start"}
    path = Path(out_dir) / f"compare__{layer.replace('.', '_')}.json"

    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")

    try:
        seq = _env_int("PTO_ATTN_SEQ", 1)
        rec["seq"] = seq
        rec["stage"] = "build_args"
        save()
        args, plan = build_args(impl, hidden_states, kv_cache, metadata_list, seq)
        rec["arg_shapes"] = {
            n: [list(a.shape), str(a.dtype), bool(a.is_contiguous())]
            for n, a in zip(ARG_ORDER, args)
        }
        rec["stage"] = "register"
        save()
        op = _registered()

        rec["stage"] = "launch"
        save()
        op(*args)
        torch.npu.synchronize()

        pto = args[-1].float()
        nat = native_out[: pto.shape[0]].float()
        rec["pto"] = {"finite": bool(torch.isfinite(pto).all().item()),
                      "absmax": pto.abs().max().item(),
                      "absmean": pto.abs().mean().item()}
        rec["native"] = {"absmax": nat.abs().max().item(),
                         "absmean": nat.abs().mean().item()}
        rec["max_abs_diff"] = (pto - nat).abs().max().item()
        denom = (pto.norm() * nat.norm()).clamp_min(1e-12)
        rec["cosine"] = ((pto * nat).sum() / denom).item()
        rec["ok"] = True
        rec["stage"] = "complete"
    except Exception:
        import traceback

        rec["ok"] = False
        rec["error"] = traceback.format_exc()
    finally:
        save()
        print(f"[pto-attn-compare] {layer} stage={rec['stage']} ok={rec.get('ok')}", flush=True)
