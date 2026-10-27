# SPDX-License-Identifier: Apache-2.0
"""按本核M8 mixed所有权执行Native的4096维Norm；不跨owner读取mixed。

采用旧精度版已验的fold2048→64/F8树、sqrt舍入校正、scalar div和乘法边界。
与HC原始输入16384维RMS是两套不同的归约，不可互相替换。
"""

import pypto.language as pl

T = pl.dynamic("HC_INPUT_NORM_T")
P8 = pl.dynamic("HC_INPUT_NORM_P8")
FRACTION_UNIT = 1 << 23
SQRT_COMPARE_SHIFT = 152
SQRT_CORRECTION_STEPS = 2
FP32_INFINITY_BITS = 0x7F800000


@pl.jit.inline(auto_scope=False)
def hc_input_norm_rows(
    x: pl.Tensor[[P8, 4096], pl.BF16],
    weight: pl.Tensor[[4096], pl.BF16],
    output: pl.Out[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.Out[pl.Tensor[[T, 8], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    rows = pl.tensor.dim(output, 0)
    for row_block in pl.range(worker, (rows + 7) // 8, 48):
        for row_lane in pl.range(pl.min(8, rows - row_block * 8)):
            row = row_block * 8 + row_lane
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
