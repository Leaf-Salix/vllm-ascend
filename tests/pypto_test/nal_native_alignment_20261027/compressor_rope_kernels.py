# SPDX-License-Identifier: Apache-2.0
"""固定相同pooled，仅改变Norm除法；完整保留FP32 RoPE及末端BF16边界。"""

import pypto.language as pl

ROWS = pl.dynamic("COMPRESSOR_ROPE_PROBE_ROWS")


@pl.jit
def compressor_rope_arm(
    pooled: pl.Tensor[[ROWS, 512], pl.FP32],
    gamma: pl.Tensor[[512], pl.BF16],
    cosine: pl.Tensor[[ROWS, 64], pl.FP32],
    sine: pl.Tensor[[ROWS, 64], pl.FP32],
    output: pl.InOut[pl.Tensor[[ROWS, 512], pl.FP32]],
    DIRECT: pl.constexpr,
    EPS: pl.constexpr,
):
    for worker in pl.spmd(48, name_hint="compressor_same_pool_rope", allow_early_resolve=True):
        for row in pl.range(worker * 16, pl.tensor.dim(pooled, 0), 48 * 16):
            rows = pl.min(16, pl.tensor.dim(pooled, 0) - row)
            value = pl.load(pooled, [row, 0], [16, 512], valid_shape=[rows, 512])
            square = pl.mul(value, value)
            fold256 = pl.add(square[:, :256], square[:, 256:512])
            fold128 = pl.add(fold256[:, :128], fold256[:, 128:256])
            fold64 = pl.add(fold128[:, :64], fold128[:, 64:128])
            total = pl.row_sum(fold64, pl.create_tile([16, 64], dtype=pl.FP32))
            rms = pl.sqrt(pl.add(pl.mul(total, 1.0 / 512.0), EPS))
            weight = pl.cast(pl.reshape(pl.load(gamma, [0], [512]), [1, 512]), pl.FP32)
            if DIRECT:
                normalized = pl.col_expand_mul(pl.row_expand_div(value, rms), weight)
            else:
                normalized = pl.col_expand_mul(pl.row_expand_mul(value, pl.recip(rms)), weight)
            pl.store(pl.set_validshape(normalized, rows, 512), [row, 0], output)
            # TGATHER indices use a compact 64-column row stride, not a view
            # retaining the 512-column parent stride.
            rope = pl.tile.extract(normalized, 0, 448, [16, 64], target_memory=pl.MemorySpace.Vec)
            cos = pl.load(cosine, [row, 0], [16, 64], valid_shape=[rows, 64])
            sin = pl.load(sine, [row, 0], [16, 64], valid_shape=[rows, 64])
            ones = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
            index = pl.cast(pl.tile.arange(0, [1, 64], dtype=pl.INT32), pl.FP32)
            columns = pl.col_expand_mul(ones, index)
            halves = pl.cast(pl.cast(pl.mul(columns, 0.5), pl.INT32, mode="trunc"), pl.FP32)
            lane = pl.sub(columns, pl.mul(halves, 2.0))
            swap = pl.cast(pl.sub(pl.add(columns, 1.0), pl.mul(lane, 2.0)), pl.INT32)
            row_offsets = pl.reshape(pl.tile.muls(pl.tile.ci(0, [1, 16], dtype=pl.INT32), 64), [16, 1])
            addresses = pl.set_validshape(pl.row_expand_add(swap, row_offsets), rows, 64)
            swapped = pl.tile.gather(rope, addresses, pl.tile.create([16, 64], dtype=pl.INT32))
            signed_sin = pl.mul(sin, pl.sub(pl.mul(lane, 2.0), 1.0))
            rotated = pl.add(pl.mul(rope, cos), pl.mul(swapped, signed_sin))
            pl.store(pl.set_validshape(rotated, rows, 64), [row, 448], output)
    return output
