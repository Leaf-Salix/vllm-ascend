# SPDX-License-Identifier: Apache-2.0
"""Isolated HC input prototype: four column owners and ordered RMS combination.

Only frozen experiment packages import this module. The existing widen_tail
root is repurposed as a partial-sum buffer; it is not also used as tail scratch.
Each (column split, row tile) owns a 512-byte slot, of which 256 bytes hold eight
unchanged K512 row sums. Capacity is 128 M8 tiles (1024 rows), while this pilot
uses the fixed eight-case specification with T<=192. Adoption needs a guarded
resource policy and dynamic-state/service checks, not merely this prototype.
"""

import pypto.language as pl

from .hc_stages import (
    HC_DIM,
    HC_DIM_INV,
    LINEAR_K_PER_SPLIT,
    LINEAR_OK,
    NORM_EPS,
    PAD_ROWS_DYN,
    RMS_K_TILE,
    T_DYN,
    T_TILE,
)

RMS_SLOT_WORDS = 128
RMS_CHUNKS_PER_SPLIT = LINEAR_K_PER_SPLIT // RMS_K_TILE


@pl.jit.inline(auto_scope=False)
def hc_widen_columns(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],
    rms_parts: pl.InOut[pl.Tensor[[T_TILE, HC_DIM], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    workers: pl.Scalar[pl.INDEX],
):
    """Write disjoint K4096 slices and eight separate K512 sums per owner."""
    tokens = pl.tensor.dim(x, 0)
    row_tiles = (tokens + T_TILE - 1) // T_TILE
    for work in pl.range(worker, row_tiles * LINEAR_OK, workers):
        row_tile = work // LINEAR_OK
        split = work % LINEAR_OK
        row = row_tile * T_TILE
        rows = pl.min(T_TILE, tokens - row)
        for local_chunk in pl.pipeline(RMS_CHUNKS_PER_SPLIT, stage=4):
            column = split * LINEAR_K_PER_SPLIT + local_chunk * RMS_K_TILE
            source = pl.slice(x, [T_TILE, RMS_K_TILE], [row, column], valid_shape=[rows, RMS_K_TILE])
            value = pl.cast(source, pl.FP32)
            if rows == T_TILE:
                squared = pl.mul(value, value)
                row_sum = pl.reshape(pl.row_sum(squared), [1, T_TILE])
            else:
                valid = pl.set_validshape(value, rows, RMS_K_TILE)
                clean = pl.fillpad(valid, pad_value=pl.PadValue.zero)
                squared = pl.mul(clean, clean)
                row_sum = pl.reshape(pl.row_sum(squared), [1, T_TILE])
            output = pl.set_validshape(value, rows, RMS_K_TILE)
            x_fp32 = pl.assemble(x_fp32, output, [row, column])
            rms_parts = pl.assemble(rms_parts, row_sum, [split, row_tile * RMS_SLOT_WORDS + local_chunk * T_TILE])
    return x_fp32, rms_parts


@pl.jit.inline(auto_scope=False)
def hc_combine_rms(
    rms_parts: pl.Tensor[[T_TILE, HC_DIM], pl.FP32],
    inv_rms: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    workers: pl.Scalar[pl.INDEX],
):
    """Original +0, K512[0..31] left fold; no split-level sum or reassociation."""
    row_tiles = (tokens + T_TILE - 1) // T_TILE
    for row_tile in pl.range(worker, row_tiles, workers):
        total = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for chunk in pl.pipeline(HC_DIM // RMS_K_TILE, stage=2):
            split = chunk // RMS_CHUNKS_PER_SPLIT
            local_chunk = chunk % RMS_CHUNKS_PER_SPLIT
            part = pl.slice(rms_parts, [1, T_TILE], [split, row_tile * RMS_SLOT_WORDS + local_chunk * T_TILE])
            total = pl.add(total, part)
        mean = pl.add(pl.mul(total, HC_DIM_INV), NORM_EPS)
        inverse = pl.reshape(pl.rsqrt(mean, high_precision=True), [T_TILE, 1])
        inv_rms = pl.assemble(inv_rms, inverse, [row_tile * T_TILE, 0])
    return inv_rms
