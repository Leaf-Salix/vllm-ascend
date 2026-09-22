"""Feed vLLM's decode tensors to the PyPTO attention-only CSA kernel.

The kernel is ``ops.pypto.dspark.decode_csa.decode_csa_attn_tp1_test``: it takes
``x_normed [T, D] BF16`` and fills ``attn_out [T, D] BF16``, leaving ``npu_hc_pre``,
the input RMSNorm and ``npu_hc_post`` to the native path. Everything here is the
translation between vLLM's decode state and that kernel's 46 arguments.

Two properties the per-step path must keep, because it has to survive ACLGraph
capture: no device-to-host read (no ``.item()``, no boolean-mask indexing, no
``torch.unique``), and no allocation whose shape depends on a device value.

The public ``forward`` matches ``DeepseekV4Attention.forward`` and falls back to
native attention for paths not covered by this first decode specialization.
"""

from __future__ import annotations

import sys

import torch

# --- kernel import -----------------------------------------------------------
# decode_csa fixes its TP specialization at import time from sys.argv, which a
# vLLM process never carries. Inject it so B = DECODE_BATCH // TP and T = B * S
# land on the intended shape instead of the module's own default of 2.
_TP = 4  # Old kernel specialization: B=16 on one physical NPU.


def _import_kernel():
    argv = sys.argv
    sys.argv = [argv[0], "--tp", str(_TP), *argv[1:]]
    try:
        from .pypto.dspark import config as kcfg
        from .pypto.dspark import decode_csa as kcsa
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

# The first supported target is the A3 32-token cache layout. Larger page
# configurations must take the native path until their layout is validated.
VLLM_PAGE = 32
VLLM_STATE_PAGE = 2
COMPRESS_RATIO = 4


# --- weights: one-time, cached on the model attention ------------------------


def _to_nd(w: torch.Tensor) -> torch.Tensor:
    """Undo the FRACTAL_NZ layout vLLM gives quantized weights.

    ``weight_nz_mode`` defaults to 1, and an NZ-laid-out weight read as ND is
    silently wrong rather than an error, so every INT8 weight goes through this.
    Dense weights are never converted, so they are returned untouched.
    """
    if w.dtype not in (torch.int8, torch.uint8):
        return w
    import torch_npu

    from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

    return torch_npu.npu_format_cast(w, ACL_FORMAT_FRACTAL_ND)


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


def _int8(linear, want: tuple, scale_len: int, kcfg):
    """An INT8 weight plus its per-channel scale, quantizing if vLLM kept it dense.

    The scale's axis is not the same for every slot: wq_b and idx_wq_b carry one
    scale per output column, wo_b one per output row. ``scale_len`` picks which,
    so a mismatch is a shape error here rather than at kernel launch.
    """
    w = _to_nd(linear.weight.detach())
    scale = getattr(linear, "weight_scale_fp32", None)
    if scale is None:
        scale = getattr(linear, "weight_scale", None)
    if w.dtype in (torch.int8, torch.uint8) and scale is not None:
        return _oriented(w, want), scale.detach().float().reshape(-1)

    dense = _oriented(w, want)
    if scale_len == want[1]:
        q, sc = _quant_int8_per_channel(dense.t().contiguous(), kcfg)
        return q.t().contiguous(), sc
    if scale_len == want[0]:
        return _quant_int8_per_channel(dense, kcfg)
    raise ValueError(f"scale length {scale_len} matches neither axis of {want}")


def _hadamard(dim: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    """Sylvester Hadamard, normalized.

    The kernel treats this as a bare right-hand matmul operand with no scaling of
    its own, so the 1/sqrt(dim) that vLLM's rotate_activation applies separately
    has to be folded in here. Mirrors decode_csa.py::init_hadamard_idx.
    """
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / (dim**0.5)).to(dtype).to(device)


def prepare_weights(attn):
    """Build and cache the kernel's weight arguments on the model attention."""
    cached = getattr(attn, "_pto_attn_weights", None)
    if cached is not None:
        return cached

    kcsa, kcfg = kernel()
    D, QL, HD = kcsa.D, kcsa.Q_LORA, kcsa.HEAD_DIM
    IH, ID = kcsa.IDX_N_HEADS, kcsa.IDX_HEAD_DIM
    indexer = attn.indexer
    compressor = attn.compressor
    assert indexer is not None and indexer.compressor is not None and compressor is not None
    inner_compressor = indexer.compressor
    wq_b, wq_b_scale = _int8(attn.wq_b, (QL, kcsa.H * HD), kcsa.H * HD, kcfg)
    idx_wq_b, idx_wq_b_scale = _int8(indexer.wq_b, (QL, IH * ID), IH * ID, kcfg)
    wo_b, wo_b_scale = _int8(attn.wo_b, (D, kcsa.O_GROUPS * kcsa.O_LORA), D, kcfg)

    w = {
        "wq_a": _dense(attn.wq_a, (D, QL)),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": _dense(attn.wkv, (D, HD)),
        "gamma_cq": attn.q_norm.weight.detach().to(torch.bfloat16),
        "gamma_ckv": attn.kv_norm.weight.detach().to(torch.bfloat16),
        "cmp_wkv": _dense(compressor.wkv, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_wgate": _dense(compressor.wgate, (kcsa.MAIN_OUT_DIM, D)),
        "cmp_ape": compressor.ape.detach().float(),
        "cmp_norm_w": compressor.norm.weight.detach().to(torch.bfloat16),
        "idx_wq_b": idx_wq_b,
        "idx_wq_b_scale": idx_wq_b_scale,
        "weights_proj": _dense(indexer.weights_proj, (D, IH)),
        "hadamard_idx": _hadamard(ID, attn.wo_b.weight.device),
        "inner_wkv": _dense(inner_compressor.wkv, (kcsa.INNER_OUT_DIM, D)),
        "inner_wgate": _dense(inner_compressor.wgate, (kcsa.INNER_OUT_DIM, D)),
        "inner_ape": inner_compressor.ape.detach().float(),
        "inner_norm_w": inner_compressor.norm.weight.detach().to(torch.bfloat16),
        "attn_sink": attn.attn_sink.detach().float(),
        # vLLM keeps [G, O_GROUP_IN, O_LORA]; the kernel wants the transpose.
        "wo_a": attn.wo_a.weight.detach().transpose(1, 2).contiguous().to(torch.bfloat16),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
    }
    attn._pto_attn_weights = w
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
    """Restate a vLLM block table over ``per``-times-smaller kernel pages.

    Fixed column count on purpose: deriving the width from the table's contents
    would need a device read, which capture forbids.
    """
    b = bt.shape[0]
    j = torch.arange(n_cols, device=bt.device)
    src = torch.div(j, per, rounding_mode="floor").clamp(max=bt.shape[1] - 1)
    phys = torch.gather(bt.long(), 1, src.unsqueeze(0).expand(b, -1)) * per + (j % per).unsqueeze(0)
    return phys.clamp(0, n_blocks - 1).to(torch.int32)


def _window_indices(positions: torch.Tensor, swa_bt: torch.Tensor, seq: int, win: int, paged=None) -> torch.Tensor:
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
    if paged is not None and paged.compacted:
        # The logical column is the block-table column, so the row a compacted
        # cache put this block in follows without searching for it.
        col = lblk.clamp(0, bt_t.shape[1] - 1)
        flat = (req.unsqueeze(1) * bt_t.shape[1] + col) * VLLM_PAGE + intra
    else:
        flat = pblk.clamp_min(0) * VLLM_PAGE + intra
    return torch.where(ok, flat, torch.full_like(abs_pos, -1)).to(torch.int32)


def _compressed_rows(positions: torch.Tensor):
    """Which compressed row each token maps to, and whether it is a boundary.

    vLLM packs the compressed tables by boundary token; the kernel indexes them by
    token. ``cumsum`` gives the mapping without a device read.
    """
    pos = positions.long()
    boundary = ((pos + 1) % COMPRESS_RATIO) == 0
    row = (torch.cumsum(boundary.long(), 0) - 1).clamp_min(0)
    return boundary, row


def _expand_compressed_rows(host_positions: torch.Tensor, src: torch.Tensor):
    """Expand packed compressed rows after counting only real host tokens."""
    boundary, row = _compressed_rows(host_positions)
    return boundary.index_select(0, src), row.index_select(0, src)


def _to_token_rows(src: torch.Tensor, row: torch.Tensor, boundary: torch.Tensor, fill):
    g = src.index_select(0, row.clamp(max=src.shape[0] - 1))
    return torch.where(boundary, g, torch.full_like(g, fill))


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

KERNEL_STATE_PAGE = 2  # C4A_COMPRESSOR_BLOCK_SIZE


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
    first = positions.long().view(b, seq)[:, 0]  # [B]

    i = torch.arange(_RR, device=dev)
    pos = first.unsqueeze(1) - (_RR - seq) + i.unsqueeze(0)  # [B, _RR]

    lblk = torch.div(pos, VLLM_STATE_PAGE, rounding_mode="floor")
    intra = pos - lblk * VLLM_STATE_PAGE
    ok = (pos >= 0) & (lblk >= 0) & (lblk < vllm_bt.shape[1])
    blk = torch.gather(vllm_bt.long(), 1, lblk.clamp(0, vllm_bt.shape[1] - 1))
    ok = ok & (blk >= 0)

    ring = torch.arange(b, device=dev).unsqueeze(1) * _RR + (pos % _RR)
    return (blk.clamp_min(0).reshape(-1), intra.clamp_min(0).reshape(-1), ring.reshape(-1), ok.reshape(-1))


def _pick_rows(cache: torch.Tensor, blk: torch.Tensor, intra: torch.Tensor, dim: int):
    """One row per (page, offset) pair, as a contiguous [k, dim]."""
    pages = cache.index_select(0, blk.clamp(0, cache.shape[0] - 1))  # [k, rows, ..., dim]
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


def write_state_ring(
    cache: torch.Tensor, ring: torch.Tensor, plan, seq: int, positions: torch.Tensor, dim: int
) -> None:
    """Return the ring's rows to vLLM's cache.

    The pages are addressed by construction rather than by de-duplicating the
    plan's page list: several ring rows share a vLLM page, and ``index_copy_``
    writes whole pages. Inactive columns are redirected to a page updated by
    that same request, with identical data, so padded/null pages cannot
    overwrite a real write even when their physical page number is zero.
    """
    blk, intra, ring_rows, valid = plan
    rr = state_ring_len()
    dev = cache.device
    b = positions.shape[0] // seq
    span = rr // VLLM_STATE_PAGE + 1

    first = positions.long().view(b, seq)[:, 0]
    base = torch.div(first - (rr - seq), VLLM_STATE_PAGE, rounding_mode="floor")
    pages_bt = blk.view(b, rr)
    lblk_of = torch.div(
        (first.unsqueeze(1) - (rr - seq) + torch.arange(rr, device=dev).unsqueeze(0)),
        VLLM_STATE_PAGE,
        rounding_mode="floor",
    )  # [B, rr]
    # Physical page for each of the span slots, taken from the row that uses it.
    slot_of_row = (lblk_of - base.unsqueeze(1)).clamp(0, span - 1)  # [B, rr]
    phys = torch.zeros(b, span, dtype=torch.long, device=dev)
    phys.scatter_(1, slot_of_row, pages_bt)

    flat_pages = phys.reshape(-1).clamp(0, cache.shape[0] - 1)
    buf = cache.index_select(0, flat_pages).reshape(b * span, VLLM_STATE_PAGE, dim).clone()

    rows = ring.reshape(-1, dim).index_select(0, ring_rows)
    tgt_page = (torch.arange(b, device=dev).unsqueeze(1) * span + slot_of_row).reshape(-1)
    keep = buf[tgt_page, intra]
    buf[tgt_page, intra] = torch.where(valid.unsqueeze(1), rows, keep)
    used = torch.zeros((b, span), dtype=torch.int32, device=dev)
    used.scatter_add_(1, slot_of_row, valid.view(b, rr).to(torch.int32))
    active = used > 0
    first = active.to(torch.int32).argmax(dim=1, keepdim=True)
    safe_pages = torch.where(active, phys, phys.gather(1, first))
    pages = buf.view(b, span, VLLM_STATE_PAGE, dim)
    first_page = pages.gather(1, first[:, :, None, None].expand(b, 1, VLLM_STATE_PAGE, dim))
    safe_data = torch.where(active[:, :, None, None], pages, first_page)
    cache.index_copy_(0, safe_pages.reshape(-1), safe_data.reshape(-1, *cache.shape[1:]))


def state_block_table(b: int, device) -> torch.Tensor:
    """The ring's own block table: request r owns its whole slice of pages."""
    pages = state_ring_len() // KERNEL_STATE_PAGE
    return (torch.arange(b, device=device).unsqueeze(1) * pages + torch.arange(pages, device=device).unsqueeze(0)).to(
        torch.int32
    )


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


class Paged:
    """A KV cache as the kernel addresses it, plus the translation that implies.

    A contiguous cache is only reinterpreted, so a vLLM slot still names the same
    row and the kernel writes into vLLM's own pages. A strided one has to be
    compacted into a private buffer, and that moves every row: slots computed
    against the original cache name rows the buffer does not have, and whatever
    the kernel writes there is invisible to vLLM until it is copied back.

    ``remap`` records what it translated, so ``commit`` needs no arguments; a
    cache the kernel writes but is given no slot mapping for is declared with
    ``track``.
    """

    def __init__(self, view, table, block_table, cache=None):
        self.view = view
        self.table = table
        self._bt = block_table
        self._cache = cache
        self._slots = None

    @property
    def compacted(self) -> bool:
        return self._cache is not None

    def _located(self, flat, req):
        """Where a row of the original cache sits after compaction, and if at all.

        The physical block is searched for in the request's own block table rather
        than derived from the position, so this holds whatever rule vLLM used to
        assign the slot.
        """
        phys = torch.div(flat, VLLM_PAGE, rounding_mode="floor")
        intra = flat - phys * VLLM_PAGE
        bt = self._bt.long()
        rows = bt.index_select(0, req.clamp(max=bt.shape[0] - 1))
        hit = rows == phys.unsqueeze(1)
        col = torch.argmax(hit.to(torch.int32), dim=1)
        ok = hit.any(dim=1) & (flat >= 0)
        return (req * bt.shape[1] + col) * VLLM_PAGE + intra, ok

    def track(self, flat, req):
        """Declare the slots the kernel writes, for a cache with no mapping arg."""
        self._slots = (flat.long(), req.long())

    def remap(self, flat, req):
        """Restate vLLM slots against the buffer the kernel was actually handed."""
        self.track(flat, req)
        if not self.compacted:
            return flat
        moved, ok = self._located(flat.long(), req.long())
        return torch.where(ok, moved, torch.full_like(moved, -1)).to(flat.dtype)

    def commit(self):
        """Copy rows the kernel wrote in a private buffer back into vLLM's cache.

        Every shape here is a function of the token count alone, so this is legal
        under graph capture. Rows the kernel did not write are parked on the
        cache's first row, which makes the scatter an identity for them -- unless
        a real slot is parked there too, in which case they all write what that
        slot's owner writes, so the duplicates agree and the real write cannot be
        the one the scatter discards. Nothing here assumes vLLM keeps row 0 free.
        """
        if not self.compacted or self._slots is None:
            return
        flat, req = self._slots
        moved, ok = self._located(flat, req)
        src = self.view.reshape(-1, *self.view.shape[2:])
        rows = src.index_select(0, moved.clamp(0, src.shape[0] - 1)).to(self._cache.dtype)
        # vLLM's page is strided, so the destination is indexed as (block, row)
        # rather than flattened: a reshape there would write to a copy instead.
        d = flat.clamp_min(0)
        blk = torch.div(d, VLLM_PAGE, rounding_mode="floor")
        row = d - blk * VLLM_PAGE
        blk = torch.where(ok, blk, torch.zeros_like(blk))
        row = torch.where(ok, row, torch.zeros_like(row))
        mask = ok.reshape(-1, *([1] * (rows.dim() - 1)))

        owns_park = ok & (d == 0)
        first = torch.argmax(owns_park.to(torch.int32)).reshape(1)
        parked = torch.where(
            owns_park.any(),
            rows.index_select(0, first).squeeze(0),
            self._cache[0, 0].to(rows.dtype),
        )
        self._cache[blk, row] = torch.where(mask, rows, parked)


def repage_kv(cache: torch.Tensor, block_table: torch.Tensor, kernel_cols: int, dtype: torch.dtype | None = None):
    """Give the kernel a 32-slot-page view of a vLLM cache.

    A contiguous cache reshapes for free. The indexer's pair does not: vLLM lays
    the key and its scale into one padded 16640-byte page, so each is strided and
    the gap in one holds the other's data. Those are compacted to the pages the
    block table names, which both makes them contiguous and drops the sibling.

    Returns a :class:`Paged`.
    """
    per = VLLM_PAGE // 32
    rows = VLLM_PAGE // per
    b, ncols = block_table.shape
    want = min(kernel_cols, ncols * per)

    if cache.is_contiguous() and dtype in (None, cache.dtype):
        view = cache.view(-1, rows, *cache.shape[2:])
        bt = _repage_block_table(block_table, want, per, view.shape[0])
        return Paged(view, _pad_cols(bt, kernel_cols), block_table)

    src = block_table.long().reshape(-1).clamp(0, cache.shape[0] - 1)
    packed = cache.index_select(0, src).contiguous()
    if dtype is not None:
        packed = packed.to(dtype)
    view = packed.view(b * ncols * per, rows, *packed.shape[2:])
    compact = torch.arange(b * ncols, device=cache.device).view(b, ncols, 1) * per
    bt = compact + torch.arange(per, device=cache.device).view(1, 1, per)
    return Paged(view, _pad_cols(bt.reshape(b, ncols * per).to(torch.int32), kernel_cols), block_table, cache=cache)


def _pad_cols(bt: torch.Tensor, cols: int) -> torch.Tensor:
    """Fixed column count: the kernel's table width is a compile-time constant."""
    have = bt.shape[1]
    if have == cols:
        return bt
    if have > cols:
        return bt[:, :cols].contiguous()
    pad = torch.full((bt.shape[0], cols - have), -1, dtype=bt.dtype, device=bt.device)
    return torch.cat([bt, pad], dim=1)


# --- the 46 arguments --------------------------------------------------------

ARG_ORDER = (
    "x_normed",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "cmp_freqs_cos",
    "cmp_freqs_sin",
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "compress_state",
    "compress_state_block_table",
    "idx_wq_b",
    "idx_wq_b_scale",
    "weights_proj",
    "hadamard_idx",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
    "inner_compress_state",
    "inner_compress_state_block_table",
    "kv_cache",
    "cmp_kv",
    "cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "cmp_slot_mapping",
    "idx_slot_mapping",
    "state_slot_mapping",
    "inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
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


def build_args(attn, positions, hidden_states, kv_cache, layer_metadata, seq: int, layer: str):
    """Translate one decode step into the kernel's 46 arguments.

    Resolve the five cache owners by name rather than metadata dictionary order.
    """
    if not isinstance(layer, str):
        # RopeDataProxy takes a non-string key as a slice and hands back another
        # proxy, so a wrong name surfaces two frames later as a missing reshape.
        raise TypeError(f"layer must be the layer's name, got {type(layer).__name__}")
    kcsa, kcfg = kernel()
    assert layer_metadata.attention is not None
    assert layer_metadata.compressor is not None
    assert layer_metadata.indexer is not None
    cmp_md = layer_metadata.attention.req_metadata
    cst_md = layer_metadata.compressor.state.req_metadata
    idx_md = layer_metadata.indexer.compressor.cache.req_metadata
    inner_state_md = layer_metadata.indexer.compressor.state.req_metadata
    swa_md = layer_metadata.swa.req_metadata
    assert cmp_md is not None and cst_md is not None
    assert idx_md is not None and inner_state_md is not None and swa_md is not None
    assert attn.compressor is not None and attn.indexer is not None
    assert attn.indexer.compressor is not None
    cmp_cos, cmp_sin, cmp_slot = attn.compressor._compute_metadata(cmp_md)
    _, _, idx_slot = attn.indexer.compressor._compute_metadata(idx_md)
    cmp_kv_c, swa_kv_c, state_c, inner_state_cache, idx_k_c, idx_s_c = kv_cache

    # The RoPE proxy resolves by layer name and quietly returns another proxy for a
    # non-string key, so the name has to come from the wrapper, not the impl.
    host_rows = cmp_md.query_start_loc[:-1].long()
    host_pos = positions.index_select(0, host_rows)
    n_real = host_pos.shape[0] // seq  # host requests this step
    ks = kcsa.S  # the kernel's compile-time S
    if n_real > kcsa.B:
        raise ValueError(f"{n_real} requests exceed the kernel's B={kcsa.B}")

    src, real = rectangular(n_real, ks, host_pos.device)
    b, t = n_real, n_real * ks
    pos = host_pos.index_select(0, src * seq)  # [T], padding repeats its request

    a = dict(prepare_weights(attn))
    token_rows = host_rows.index_select(0, src)
    a["x_normed"] = hidden_states.index_select(0, token_rows).to(torch.bfloat16)
    a["attn_out"] = torch.empty(t, hidden_states.shape[-1], dtype=torch.bfloat16, device=hidden_states.device)

    # RoPE: vLLM keeps the compressed table packed by boundary row.
    boundary, row = _expand_compressed_rows(host_pos, src)
    # The per-token tables are indexed by host row, so they follow the same
    # rectangle as x_normed; the compressed ones are indexed by boundary row and
    # are expanded through `row` below.
    a["freqs_cos"] = _rope_to_pto(_pick(cmp_md.cos, layer)).index_select(0, token_rows)
    a["freqs_sin"] = _rope_to_pto(_pick(cmp_md.sin, layer)).index_select(0, token_rows)
    cc = _rope_to_pto(cmp_cos)
    cs = _rope_to_pto(cmp_sin)
    rc = row.clamp(max=cc.shape[0] - 1)
    a["cmp_freqs_cos"] = cc.index_select(0, rc)
    a["cmp_freqs_sin"] = cs.index_select(0, rc)

    # Paged KV: the supported A3 layout already uses 32-slot pages.
    pg_swa = repage_kv(swa_kv_c, swa_md.block_table, kcsa.CMP_MAX_BLOCKS)
    pg_cmp = repage_kv(cmp_kv_c, cmp_md.block_table, kcsa.CMP_MAX_BLOCKS)
    pg_idx = repage_kv(idx_k_c, idx_md.block_table, kcsa.IDX_MAX_BLOCKS)
    # The scale cache shares its page with the key cache and is FP16 there; the
    # kernel's signature is FP32, and it cannot be cast in place.
    pg_ids = repage_kv(idx_s_c, idx_md.block_table, kcsa.IDX_MAX_BLOCKS, dtype=torch.float32)
    paged = (pg_swa, pg_cmp, pg_idx, pg_ids)
    a["kv_cache"] = pg_swa.view
    a["cmp_kv"], a["cmp_block_table"] = pg_cmp.view, pg_cmp.table
    a["idx_kv_cache"], a["idx_block_table"] = pg_idx.view, pg_idx.table
    a["idx_kv_scale"] = pg_ids.view

    # Compressor state: a private ring per request, seeded from vLLM's cache.
    main_dim = kcsa.MAIN_STATE_DIM
    inner_dim = kcsa.INNER_STATE_DIM
    plan_m = state_ring_plan(pos, ks, cst_md.block_table)
    plan_i = state_ring_plan(pos, ks, inner_state_md.block_table)
    a["compress_state"] = make_state_ring(state_c, plan_m, b, main_dim)
    a["inner_compress_state"] = make_state_ring(inner_state_cache, plan_i, b, inner_dim)
    a["compress_state_block_table"] = state_block_table(b, pos.device)
    a["inner_compress_state_block_table"] = state_block_table(b, pos.device)
    a["state_slot_mapping"] = state_slots(pos, ks)
    a["inner_state_slot_mapping"] = a["state_slot_mapping"]

    # Slots and window.
    def _inert(x):
        return torch.where(real, x, torch.full_like(x, -1))

    ori_flat = _inert(_flat_slots(swa_md.slot_mapping, VLLM_PAGE).index_select(0, token_rows))
    cmp_flat = _inert(_to_token_rows(_flat_slots(cmp_slot, VLLM_PAGE), row, boundary, -1))
    idx_flat = _inert(_to_token_rows(_flat_slots(idx_slot, VLLM_PAGE), row, boundary, -1))
    a["ori_slot_mapping"] = pg_swa.remap(ori_flat, src)
    a["cmp_slot_mapping"] = pg_cmp.remap(cmp_flat, src)
    a["idx_slot_mapping"] = pg_idx.remap(idx_flat, src)
    # The scale is written at the key's slots but takes no mapping of its own.
    pg_ids.track(idx_flat, src)
    a["window_swa_indices"] = _window_indices(pos, swa_md.block_table, ks, kcsa.WIN, paged=pg_swa)

    a["position_ids"] = pos.to(torch.int32)
    a["kv_seq_lens"] = cmp_md.seq_lens.to(torch.int32)[:b]
    a["state_slot_mapping"] = _inert(a["state_slot_mapping"])
    a["inner_state_slot_mapping"] = a["state_slot_mapping"]

    return [a[name] for name in ARG_ORDER], (
        plan_m,
        plan_i,
        state_c,
        inner_state_cache,
        main_dim,
        inner_dim,
        pos,
        ks,
        n_real,
        host_rows,
        paged,
    )


def _pick(table, layer):
    """The RoPE tables are keyed by layer, behind a dict or a RopeDataProxy.

    The proxy resolves a layer name to its registered cache group and hands back
    the tensor; only a layer registered for several groups gets a dict, and the
    DSA layers are single-group.
    """
    if isinstance(table, torch.Tensor):
        return table
    value = table[layer]
    if isinstance(value, dict):
        value = next(iter(value.values()))
    return value


# --- opt-in attention boundary ------------------------------------------------

_OP = None


def _registered():
    global _OP
    if _OP is None:
        from pypto.torch import init, register

        kcsa, _ = kernel()
        init()
        _OP = register(kcsa.decode_csa_attn_tp1_test, "pypto_csa::attention_csa")
    return _OP


def forward(
    attn,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None,
) -> torch.Tensor:
    """Match DeepseekV4Attention.forward's tensor input and output contract."""
    output = _try_pypto(attn, positions, hidden_states, llama_4_scaling)
    if output is not None:
        return output
    return attn.dsa_attn(positions, hidden_states, llama_4_scaling)


def _try_pypto(attn, positions, hidden_states, llama_4_scaling) -> torch.Tensor | None:
    """Replace one supported DSV4 attention call; return None for native fallback.

    The boundary is DeepseekV4Attention.forward. All HC, outer normalization,
    and MoE work stays in the decoder layer. A launch failure is propagated:
    falling back after a partially mutated KV cache would corrupt the step.
    """
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.forward_context import get_forward_context

    from vllm_ascend.attention.utils import (
        maybe_save_kv_layer_to_connector,
        notify_kv_cache_written,
        wait_for_kv_layer_from_connector,
    )
    from vllm_ascend.ops.dsa import _build_kv_cache

    if (
        attn.compress_ratio != COMPRESS_RATIO
        or attn.dsa_attn.need_gather_q_kv
        or attn.indexer is None
        or attn.indexer.skip_topk
        or get_tensor_model_parallel_world_size() != 1
        or llama_4_scaling is not None
    ):
        return None

    context = get_forward_context()
    metadata = context.attn_metadata
    if not isinstance(metadata, dict):
        return None
    layer = attn.dsa_attn.dsa_attn.layer_name
    owners = (
        layer,
        attn.dsa_attn.swa_cache_layer.prefix,
        attn.compressor.state_cache.prefix,
        attn.indexer.k_cache.prefix,
        attn.indexer.compressor.state_cache.prefix,
    )
    if any(owner not in metadata for owner in owners):
        return None
    impl = attn.dsa_attn.dsa_attn.impl
    if impl.vllm_config.speculative_config is not None:
        # Adaptive draft/verify boundaries can differ from the host token count.
        return None
    layer_metadata = impl._get_layer_metadata(layer, metadata)
    attention_metadata = layer_metadata.attention
    if attention_metadata is None or attention_metadata.req_metadata is None or layer_metadata.indexer is None:
        return None
    num_reqs = attention_metadata.req_metadata.query_start_loc.shape[0] - 1
    if (
        attention_metadata.num_prefills != 0
        or attention_metadata.num_decodes != num_reqs
        or attention_metadata.num_decode_tokens != num_reqs
        or attention_metadata.num_actual_tokens != num_reqs
        or num_reqs == 0
        or hidden_states.shape[0] != num_reqs
        or positions.shape[0] != num_reqs
        or hidden_states.dtype != torch.bfloat16
    ):
        return None

    req = attention_metadata.req_metadata
    if req.storage_block_size != VLLM_PAGE:
        return None
    kv_cache = _build_kv_cache(attn.dsa_attn, context)
    if (
        len(kv_cache) != 6
        or any(not isinstance(cache, torch.Tensor) for cache in kv_cache)
        or any(cache.shape[1] != VLLM_PAGE for cache in (kv_cache[0], kv_cache[1], kv_cache[4], kv_cache[5]))
        or any(cache.shape[1] != VLLM_STATE_PAGE for cache in (kv_cache[2], kv_cache[3]))
    ):
        return None

    kcsa, _ = kernel()
    if (
        num_reqs > kcsa.B
        or attn.dim != kcsa.D
        or attn.n_heads != kcsa.H
        or attn.head_dim != kcsa.HEAD_DIM
        or attn.rope_head_dim != kcsa.ROPE_HEAD_DIM
        or attn.q_lora_rank != kcsa.Q_LORA
        or attn.o_lora_rank != kcsa.O_LORA
        or attn.n_groups != kcsa.O_GROUPS
        or attn.window_size != kcsa.WIN
        or attn.indexer.n_heads != kcsa.IDX_N_HEADS
        or attn.indexer.head_dim != kcsa.IDX_HEAD_DIM
        or attn.indexer.index_topk != kcsa.IDX_TOPK
    ):
        return None
    wait_for_kv_layer_from_connector(layer)
    args, plan = build_args(attn, positions, hidden_states, kv_cache, layer_metadata, 1, layer)
    plan_m, plan_i, state_c, inner_state_cache, main_dim, inner_dim, pos, ks, n_real, host_rows, paged = plan
    _registered()(*args)
    write_state_ring(state_c, args[ARG_ORDER.index("compress_state")], plan_m, ks, pos, main_dim)
    write_state_ring(inner_state_cache, args[ARG_ORDER.index("inner_compress_state")], plan_i, ks, pos, inner_dim)
    for page in paged:
        page.commit()
    notify_kv_cache_written(layer)

    output = torch.empty_like(hidden_states)
    take = torch.arange(n_real, device=output.device) * ks
    output.index_copy_(0, host_rows, args[-1].index_select(0, take).to(output.dtype))
    maybe_save_kv_layer_to_connector(layer, list(kv_cache))
    return output
