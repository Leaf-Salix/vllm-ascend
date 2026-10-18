# SPDX-License-Identifier: Apache-2.0
"""Native-order input RMSNorm for the A3 full-PTO TND candidate.

The sum follows the bundled Native 4096-column half-interval reduction. Match
its scalar FP32 square root/division boundary before the final BF16 rounding.
"""

import pypto.language as pl

T = pl.dynamic("INPUT_NORM_T")
FRACTION_UNIT = 1 << 23
SQRT_COMPARE_SHIFT = 152
SQRT_CORRECTION_STEPS = 2
FP32_INFINITY_BITS = 0x7F800000


def _square_sum(x: pl.Tensor[[T, 4096], pl.BF16], sums: pl.Out[pl.Tensor[[T, 8], pl.FP32]]):
    x.bind_dynamic(0, T)
    sums.bind_dynamic(0, T)
    rows = pl.tensor.dim(x, 0)
    for row in pl.spmd(rows, name_hint="input_rms_sum_dsl", allow_early_resolve=True):
        raw = pl.load(x, [row, 0], [1, 4096], target_memory=pl.MemorySpace.Vec)
        wide = pl.cast(raw, pl.FP32)
        square = pl.mul(wide, wide)
        fold2048 = pl.add(pl.slice(square, [1, 2048], [0, 0]), pl.slice(square, [1, 2048], [0, 2048]))
        fold1024 = pl.add(pl.slice(fold2048, [1, 1024], [0, 0]), pl.slice(fold2048, [1, 1024], [0, 1024]))
        fold512 = pl.add(pl.slice(fold1024, [1, 512], [0, 0]), pl.slice(fold1024, [1, 512], [0, 512]))
        fold256 = pl.add(pl.slice(fold512, [1, 256], [0, 0]), pl.slice(fold512, [1, 256], [0, 256]))
        fold128 = pl.add(pl.slice(fold256, [1, 128], [0, 0]), pl.slice(fold256, [1, 128], [0, 128]))
        fold64 = pl.add(pl.slice(fold128, [1, 64], [0, 0]), pl.slice(fold128, [1, 64], [0, 64]))
        # ColMajor 8x1 is a full 32B result. Replicate the final 64-value
        # reduction input without changing its order; no 8x4096 UB expansion.
        repeated = pl.col_expand(pl.create_tile([8, 64], dtype=pl.FP32), fold64)
        total = pl.row_sum(repeated, pl.create_tile([8, 64], dtype=pl.FP32))
        pl.store(pl.reshape(total, [1, 8]), [row, 0], sums)
    return sums


square_sum = pl.jit.inline(auto_scope=False)(_square_sum)
square_sum_test = pl.jit(_square_sum)


def _input_norm_test(
    x: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[4096], pl.BF16],
    output: pl.Out[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.Out[pl.Tensor[[T, 8], pl.FP32]],
):
    x.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    rstd.bind_dynamic(0, T)
    rows = pl.tensor.dim(x, 0)
    sums = pl.create_tensor([rows, 8], dtype=pl.FP32)
    sums = square_sum(x, sums)
    for row in pl.spmd(rows, name_hint="input_rms_inverse", allow_early_resolve=True):
        total = pl.read(sums, [row, 0])
        total_vector = pl.mul(pl.full([1, 8], dtype=pl.FP32, value=1.0), total)
        variance = pl.add(pl.mul(total_vector, 1.0 / 4096.0), 1e-6)
        root_value = pl.sqrt(variance)
        # Native uses a device scalar square root, while A3 VSQRT may be 1 ULP off.
        # Round its candidate with exact INT64 midpoint comparisons (normal positive
        # variance; eps=1e-6), then use device scalar division, as in the QR path.
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
                x_upper = x_sig << (x_exp - 2 * y_exp + SQRT_COMPARE_SHIFT)
                previous_bits = y_bits - 1
                previous_exp = previous_bits // fraction_unit
                previous_sig = previous_bits % fraction_unit + fraction_unit
                lower_mid = 2 * previous_sig + 1
                lower_square = lower_mid * lower_mid
                x_lower = x_sig << (x_exp - 2 * previous_exp + SQRT_COMPARE_SHIFT)
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
    for row in pl.spmd(rows, name_hint="input_rms_apply", allow_early_resolve=True):
        apply_value = pl.cast(x[row : row + 1, 0:4096], pl.FP32)
        w = pl.cast(pl.reshape(weight, [1, 4096]), pl.FP32)
        inv = pl.read(rstd, [row, 0])
        scaled = pl.mul(apply_value, inv)
        result = pl.col_expand_mul(scaled, w)
        output[row : row + 1, 0:4096] = pl.cast(result, pl.BF16, mode="rint")
    return output, rstd


input_norm_entry = pl.jit.inline(auto_scope=False)(_input_norm_test)
input_norm_test = pl.jit(_input_norm_test)
