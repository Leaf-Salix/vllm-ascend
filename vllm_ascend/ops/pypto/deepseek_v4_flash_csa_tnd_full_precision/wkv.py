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
    panels = (rows + 63) // 64
    accum = pl.create_tensor([rows + 63, 512], dtype=pl.FP32)
    if rows <= 16:
        for m16_worker in pl.spmd(row_tiles * 32, name_hint="wkv_matmul"):
            m16_row = m16_worker // 32 * 16
            m16_group = m16_worker % 32
            m16_col = m16_group * 16
            m16_valid = pl.min(16, rows - m16_row)
            m16_tile = pl.create_tensor([16, 16], dtype=pl.FP32)
            for m16_step in pl.range(16):
                m16_k_block = pl.read(order, [m16_group, m16_step])
                m16_offset = pl.max(m16_k_block, 0) * 256
                m16_a = pl.slice(x, [16, 256], [m16_row, m16_offset], valid_shape=[m16_valid, 256])
                m16_b = weight[m16_col : m16_col + 16, m16_offset : m16_offset + 256]
                m16_tile = pl.matmul_acc(m16_tile, m16_a, m16_b, b_trans=True, init_cond=m16_step == 0)
            accum[m16_row : m16_row + 16, m16_col : m16_col + 16] = m16_tile
    else:
        for worker in pl.spmd(panels * 16, name_hint="wkv64n32_matmul"):
            row = worker // 16 * 64
            first_group = worker % 16 * 2
            col = first_group * 16
            valid = pl.min(64, rows - row)
            same_order = 1
            for check_step in pl.range(16):
                if pl.read(order, [first_group, check_step]) != pl.read(order, [first_group + 1, check_step]):
                    same_order = 0
            if same_order == 1:
                first_block = pl.read(order, [first_group, 0])
                first_offset = pl.max(first_block, 0) * 256
                first_a = pl.slice(x, [64, 256], [row, first_offset], valid_shape=[valid, 256])
                first_b = weight[col : col + 32, first_offset : first_offset + 256]
                wide_tile = pl.matmul(first_a, first_b, b_trans=True, out_dtype=pl.FP32)
                for step in pl.range(1, 16):
                    k_block = pl.read(order, [first_group, step])
                    offset = pl.max(k_block, 0) * 256
                    a = pl.slice(x, [64, 256], [row, offset], valid_shape=[valid, 256])
                    b = weight[col : col + 32, offset : offset + 256]
                    wide_tile = pl.matmul_acc(wide_tile, a, b, b_trans=True, init_cond=False)
                accum = pl.assemble(accum, wide_tile, [row, col])
            else:
                for local_group in pl.range(2):
                    narrow_group = first_group + local_group
                    narrow_col = narrow_group * 16
                    narrow_first_block = pl.read(order, [narrow_group, 0])
                    narrow_first_offset = pl.max(narrow_first_block, 0) * 256
                    narrow_first_a = pl.slice(x, [64, 256], [row, narrow_first_offset], valid_shape=[valid, 256])
                    narrow_first_b = weight[
                        narrow_col : narrow_col + 16, narrow_first_offset : narrow_first_offset + 256
                    ]
                    narrow_tile = pl.matmul(narrow_first_a, narrow_first_b, b_trans=True, out_dtype=pl.FP32)
                    for narrow_step in pl.range(1, 16):
                        narrow_block = pl.read(order, [narrow_group, narrow_step])
                        narrow_offset = pl.max(narrow_block, 0) * 256
                        narrow_a = pl.slice(x, [64, 256], [row, narrow_offset], valid_shape=[valid, 256])
                        narrow_b = weight[narrow_col : narrow_col + 16, narrow_offset : narrow_offset + 256]
                        narrow_tile = pl.matmul_acc(narrow_tile, narrow_a, narrow_b, b_trans=True, init_cond=False)
                    accum = pl.assemble(accum, narrow_tile, [row, narrow_col])
    for block in pl.spmd(row_tiles, name_hint="wkv_round"):
        out_row = block * 16
        out_rows = pl.min(16, rows - out_row)
        projected = pl.load(accum, [out_row, 0], [16, 512], valid_shape=[out_rows, 512])
        rounded = pl.cast(projected, pl.BF16, mode="rint")
        pl.store(rounded, [out_row, 0], output)
    return output


wkv_entry = pl.jit.inline(auto_scope=False)(_wkv_test)
wkv_test = pl.jit(_wkv_test)
