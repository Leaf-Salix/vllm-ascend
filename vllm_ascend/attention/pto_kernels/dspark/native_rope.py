# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Read the native compressor's compact RoPE at the point of consumption."""

import pypto.language as pl

from .config import DECODE_SEQ, FLASH

COMPACT_ROWS = pl.dynamic("NATIVE_ROPE_COMPACT_ROWS")
TOKENS = pl.dynamic("NATIVE_ROPE_TOKENS")
REQUESTS = pl.dynamic("NATIVE_ROPE_REQUESTS")
ROPE_DIM = FLASH.qk_rope_head_dim
TILE_ROWS = 16  # Both compressor RMS consumers use a 16-row vector tile.


@pl.jit.inline(auto_scope=False)
def load_compact_rope(
    cos: pl.Tensor[[COMPACT_ROWS, ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS, ROPE_DIM], pl.FP32],
    positions: pl.Tensor[[TOKENS], pl.INT32],
    valid: pl.Tensor[[TOKENS], pl.INT32],
    offsets: pl.Tensor[[REQUESTS], pl.INT32],
    begin: pl.Scalar[pl.INDEX],
    rows: pl.Scalar[pl.INDEX],
):
    """Fill consumer-local tiles; invalid/non-closing rows never read compact GM.

    The native producer packs closing boundaries in request order. Offsets are
    computed from its original query bounds and start_pos. No token-sized
    frequency buffers or extra GM write/read roundtrip are introduced here.
    """
    cosine = pl.tile.full([TILE_ROWS, ROPE_DIM], dtype=pl.FP32, value=1.0)
    sine = pl.tile.full([TILE_ROWS, ROPE_DIM], dtype=pl.FP32, value=0.0)
    for row in pl.range(rows):
        token = begin + row
        position = pl.read(positions, [token])
        if pl.read(valid, [token]) != 0 and (position + 1) % 4 == 0:
            compact = pl.cast(pl.read(offsets, [token // DECODE_SEQ]), pl.INDEX) + (position + 1) // 4
            cosine = pl.gather_row(cosine, cos, [row, 0], [compact, 0], [1, ROPE_DIM])
            sine = pl.gather_row(sine, sin, [row, 0], [compact, 0], [1, ROPE_DIM])
    return cosine, sine
