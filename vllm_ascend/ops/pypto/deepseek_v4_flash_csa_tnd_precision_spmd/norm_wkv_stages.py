# SPDX-License-Identifier: Apache-2.0
"""Native-order Norm/WKV stage bodies from the preserved precision baseline.

Only the worker loops change to 48 AIV / 24 AIC striding. WKV keeps the
M16/M64 selection, K256 qualification table, FP32 accumulation and BF16 rint.
No nested task/SPMD or workspace reuse is introduced in these inline stages.
"""

import pypto.language as pl

T = pl.dynamic("PRECISION_SPMD_T")
ACC_ROWS = pl.dynamic("PRECISION_SPMD_ACC_ROWS")
FRACTION_UNIT = 1 << 23
SQRT_COMPARE_SHIFT = 152
SQRT_CORRECTION_STEPS = 2
FP32_INFINITY_BITS = 0x7F800000


@pl.jit.inline(auto_scope=False)
def norm_rows(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[4096], pl.BF16],
    output: pl.Out[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.Out[pl.Tensor[[T, 8], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    rows = pl.tensor.dim(x, 0)
    for row in pl.range(worker, rows, 48):
        raw = pl.load(x, [row, 0], [1, 4096], target_memory=pl.MemorySpace.Vec)
        wide = pl.cast(raw, pl.FP32)
        square = pl.mul(wide, wide)
        fold2048 = pl.add(pl.slice(square, [1, 2048], [0, 0]), pl.slice(square, [1, 2048], [0, 2048]))
        fold1024 = pl.add(pl.slice(fold2048, [1, 1024], [0, 0]), pl.slice(fold2048, [1, 1024], [0, 1024]))
        fold512 = pl.add(pl.slice(fold1024, [1, 512], [0, 0]), pl.slice(fold1024, [1, 512], [0, 512]))
        fold256 = pl.add(pl.slice(fold512, [1, 256], [0, 0]), pl.slice(fold512, [1, 256], [0, 256]))
        fold128 = pl.add(pl.slice(fold256, [1, 128], [0, 0]), pl.slice(fold256, [1, 128], [0, 128]))
        fold64 = pl.add(pl.slice(fold128, [1, 64], [0, 0]), pl.slice(fold128, [1, 64], [0, 64]))
        repeated = pl.col_expand(pl.create_tile([8, 64], dtype=pl.FP32), fold64)
        total = pl.row_sum(repeated, pl.create_tile([8, 64], dtype=pl.FP32))
        total_scalar = pl.read(total, [0, 0])
        total_vector = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), total_scalar)
        variance = pl.add(pl.mul(total_vector, 1.0 / 4096.0), 1e-06)
        root_value = pl.sqrt(variance)
        variance_bits = pl.reinterpret_view(variance, pl.INT32)
        approx_bits = pl.reinterpret_view(root_value, pl.INT32)
        x_bits = pl.cast(pl.read(variance_bits, [0, 0]), pl.INT64)
        y_bits = pl.cast(pl.read(approx_bits, [0, 0]), pl.INT64)
        fraction_unit = FRACTION_UNIT
        if x_bits >= fraction_unit and x_bits < FP32_INFINITY_BITS:
            x_exp = x_bits // fraction_unit
            x_sig = x_bits % fraction_unit + fraction_unit
            for correction_step in pl.range(SQRT_CORRECTION_STEPS):
                y_exp = y_bits // fraction_unit
                y_sig = y_bits % fraction_unit + fraction_unit
                upper_mid = 2 * y_sig + 1
                upper_square = upper_mid * upper_mid
                x_upper = x_sig << x_exp - 2 * y_exp + SQRT_COMPARE_SHIFT
                previous_bits = y_bits - 1
                previous_exp = previous_bits // fraction_unit
                previous_sig = previous_bits % fraction_unit + fraction_unit
                lower_mid = 2 * previous_sig + 1
                lower_square = lower_mid * lower_mid
                x_lower = x_sig << x_exp - 2 * previous_exp + SQRT_COMPARE_SHIFT
                if x_upper > upper_square or (x_upper == upper_square and y_bits % 2 == 1):
                    y_bits = y_bits + 1
                elif x_lower < lower_square or (x_lower == lower_square and y_bits % 2 == 1):
                    y_bits = y_bits - 1
        rounded_bits = pl.full([1, 8], dtype=pl.INT32, value=0)
        for root_lane in pl.range(8):
            pl.write(rounded_bits, [0, root_lane], pl.cast(y_bits, pl.INT32))
        rounded_root = pl.reinterpret_view(rounded_bits, pl.FP32)
        scalar_root_value = pl.read(rounded_root, [0, 0])
        scalar_inv = 1.0 / scalar_root_value
        inverse = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), scalar_inv)
        rstd[row : row + 1, 0:8] = inverse
        apply_value = wide
        w_raw = pl.reshape(pl.load(weight, [0], [4096], target_memory=pl.MemorySpace.Vec), [1, 4096])
        w = pl.cast(w_raw, pl.FP32)
        inv = pl.read(inverse, [0, 0])
        scaled = pl.mul(apply_value, inv)
        result = pl.col_expand_mul(scaled, w)
        pl.store(pl.cast(result, pl.BF16, mode="rint"), [row, 0], output)
    return (output, rstd)


@pl.jit.inline(auto_scope=False)
def wkv_matmul(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    logical_worker: pl.Scalar[pl.INDEX],
):
    rows = pl.tensor.dim(x, 0)
    row_tiles = (rows + 15) // 16
    panels = (rows + 63) // 64
    if rows <= 16:
        for m16_worker in pl.range(logical_worker, row_tiles * 32, 24):
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
        for worker in pl.range(logical_worker, panels * 16, 24):
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
    return accum


@pl.jit.inline(auto_scope=False)
def wkv_round(
    accum: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.Out[pl.Tensor[[T, 512], pl.BF16]],
    worker: pl.Scalar[pl.INDEX],
):
    rows = pl.tensor.dim(output, 0)
    row_tiles = (rows + 15) // 16
    for block in pl.range(worker, row_tiles, 48):
        out_row = block * 16
        out_rows = pl.min(16, rows - out_row)
        projected = pl.load(accum, [out_row, 0], [16, 512], valid_shape=[out_rows, 512])
        rounded = pl.cast(projected, pl.BF16, mode="rint")
        pl.store(rounded, [out_row, 0], output)
    return output
