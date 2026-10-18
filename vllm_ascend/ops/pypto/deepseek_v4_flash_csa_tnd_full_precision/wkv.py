# SPDX-License-Identifier: Apache-2.0
"""按冻结的Native K256次序执行WKV，保留FP32累加及最终BF16边界。"""

import pypto.language as pl

T = pl.dynamic("WKV_INPUT_T")


def _wkv_test(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    output: pl.Out[pl.Tensor[[T, 512], pl.BF16]],
):
    x.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    rows = pl.tensor.dim(x, 0)
    row_tiles = (rows + 15) // 16
    accum = pl.create_tensor([row_tiles * 16, 512], dtype=pl.FP32)
    for worker in pl.spmd(row_tiles * 32, name_hint="wkv_matmul"):
        row = worker // 32 * 16
        group = worker % 32
        col = group * 16
        valid = pl.min(16, rows - row)
        tile = pl.create_tensor([16, 16], dtype=pl.FP32)
        for step in pl.range(16):
            k_block = pl.read(order, [group, step])
            offset = pl.max(k_block, 0) * 256
            a = pl.slice(x, [16, 256], [row, offset], valid_shape=[valid, 256])
            b = weight[col : col + 16, offset : offset + 256]
            tile = pl.matmul_acc(tile, a, b, b_trans=True, init_cond=(step == 0))
        accum[row : row + 16, col : col + 16] = tile
    for block in pl.spmd(row_tiles, name_hint="wkv_round"):
        out_row = block * 16
        out_rows = pl.min(16, rows - out_row)
        projected = pl.load(accum, [out_row, 0], [16, 512], valid_shape=[out_rows, 512])
        rounded = pl.cast(projected, pl.BF16, mode="rint")
        pl.store(rounded, [out_row, 0], output)
    return output


wkv_entry = pl.jit.inline(auto_scope=False)(_wkv_test)
wkv_test = pl.jit(_wkv_test)
