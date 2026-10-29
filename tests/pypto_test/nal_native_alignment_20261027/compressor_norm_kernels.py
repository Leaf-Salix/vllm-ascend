# SPDX-License-Identifier: Apache-2.0
"""相同pooled下，仅对比乘倒数与Native直接除sqrt的舍入边界。"""

import pypto.language as pl

ROWS = pl.dynamic("COMPRESSOR_NORM_PROBE_ROWS")


@pl.jit
def compressor_norm_pair(
    pooled: pl.Tensor[[ROWS, 512], pl.FP32],
    gamma: pl.Tensor[[512], pl.BF16],
    baseline: pl.InOut[pl.Tensor[[ROWS, 512], pl.FP32]],
    direct: pl.InOut[pl.Tensor[[ROWS, 512], pl.FP32]],
    EPS: pl.constexpr,
):
    for worker in pl.spmd(48, name_hint="compressor_norm_same_input", allow_early_resolve=True):
        for row in pl.range(worker * 8, pl.tensor.dim(pooled, 0), 48 * 8):
            rows = pl.min(8, pl.tensor.dim(pooled, 0) - row)
            value = pl.load(pooled, [row, 0], [8, 512], valid_shape=[rows, 512])
            square = pl.mul(value, value)
            fold256 = pl.add(square[:, :256], square[:, 256:512])
            fold128 = pl.add(fold256[:, :128], fold256[:, 128:256])
            fold64 = pl.add(fold128[:, :64], fold128[:, 64:128])
            total = pl.row_sum(fold64, pl.create_tile([8, 64], dtype=pl.FP32))
            rms = pl.sqrt(pl.add(pl.mul(total, 1.0 / 512.0), EPS))
            weight = pl.cast(pl.reshape(pl.load(gamma, [0], [512]), [1, 512]), pl.FP32)
            old = pl.col_expand_mul(pl.row_expand_mul(value, pl.recip(rms)), weight)
            new = pl.col_expand_mul(pl.row_expand_div(value, rms), weight)
            pl.store(old, [row, 0], baseline)
            pl.store(new, [row, 0], direct)
    return baseline, direct
