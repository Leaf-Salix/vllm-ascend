# SPDX-License-Identifier: Apache-2.0
"""借鉴nalinaly输出融合编排，保留我方Native数值体；独立诊断候选。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.nz_mode import QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT

T = pl.dynamic("FULL_PRECISION_OPROJ_T")


def _o_proj_hcpost(
    heads: pl.Tensor[[T, 32768], pl.BF16],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    start: pl.Tensor[[16, 8, 8], pl.INT32],
    projected: pl.Out[pl.Tensor[[T, 8192], pl.BF16]],
    quantized: pl.Out[pl.Tensor[[T, 8192], pl.INT8]],
    token_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    residual: pl.Tensor[[T, 4, 4096], pl.BF16],
    post: pl.Tensor[[T, 4], pl.FP32],
    comb: pl.Tensor[[T, 16], pl.FP32],
    output: pl.Out[pl.Tensor[[T, 4, 4096], pl.BF16]],
):
    heads.bind_dynamic(0, T)
    projected.bind_dynamic(0, T)
    quantized.bind_dynamic(0, T)
    token_scale.bind_dynamic(0, T)
    residual.bind_dynamic(0, T)
    post.bind_dynamic(0, T)
    comb.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    rows = pl.tensor.dim(heads, 0)
    row_tiles = (rows + 15) // 16
    accum = pl.create_tensor([rows + 127, 8192], dtype=pl.FP32)
    if rows >= 192:
        for seed_group in pl.parallel(8):
            for seed_block in pl.spmd(8, name_hint="oproj_a"):
                seed_col = seed_block * 128
                for seed_panel in pl.range((row_tiles + 7) // 8):
                    seed_first_rb = seed_panel * 8
                    seed_first_start = pl.max(pl.read(start, [seed_first_rb, seed_group, seed_block]), 0)
                    seed_same_start = 1
                    if seed_first_rb + 1 < row_tiles:
                        if pl.max(pl.read(start, [seed_first_rb + 1, seed_group, seed_block]), 0) != seed_first_start:
                            seed_same_start = 0
                    if seed_first_rb + 2 < row_tiles:
                        if pl.max(pl.read(start, [seed_first_rb + 2, seed_group, seed_block]), 0) != seed_first_start:
                            seed_same_start = 0
                    if seed_first_rb + 3 < row_tiles:
                        if pl.max(pl.read(start, [seed_first_rb + 3, seed_group, seed_block]), 0) != seed_first_start:
                            seed_same_start = 0
                    if seed_first_rb + 4 < row_tiles:
                        if pl.max(pl.read(start, [seed_first_rb + 4, seed_group, seed_block]), 0) != seed_first_start:
                            seed_same_start = 0
                    if seed_first_rb + 5 < row_tiles:
                        if pl.max(pl.read(start, [seed_first_rb + 5, seed_group, seed_block]), 0) != seed_first_start:
                            seed_same_start = 0
                    if seed_first_rb + 6 < row_tiles:
                        if pl.max(pl.read(start, [seed_first_rb + 6, seed_group, seed_block]), 0) != seed_first_start:
                            seed_same_start = 0
                    if seed_first_rb + 7 < row_tiles:
                        if pl.max(pl.read(start, [seed_first_rb + 7, seed_group, seed_block]), 0) != seed_first_start:
                            seed_same_start = 0
                    if rows <= 16:
                        seed_same_start = 0
                    if seed_first_start > 240:
                        seed_same_start = 0
                    if seed_same_start == 1:
                        seed_fast_row = seed_first_rb * 16
                        seed_fast_valid = pl.min(128, rows - seed_fast_row)
                        seed_fast_seed_k = seed_first_start * 16
                        seed_fast_seed_x = pl.slice(
                            heads,
                            [128, 256],
                            [seed_fast_row, seed_group * 4096 + seed_fast_seed_k],
                            valid_shape=[seed_fast_valid, 256],
                        )
                        seed_fast_seed_w = wo_a[
                            seed_group : seed_group + 1,
                            seed_fast_seed_k : seed_fast_seed_k + 256,
                            seed_col : seed_col + 128,
                        ]
                        seed_fast_acc = pl.matmul(seed_fast_seed_x, seed_fast_seed_w, out_dtype=pl.FP32)
                        for seed_fast_packet in pl.pipeline(1, 16, stage=2):
                            seed_fast_packet_kb = (seed_first_start + seed_fast_packet * 16) % 256
                            if seed_fast_packet_kb <= 240:
                                seed_fast_packet_k = seed_fast_packet_kb * 16
                                seed_fast_packet_x = pl.slice(
                                    heads,
                                    [128, 256],
                                    [seed_fast_row, seed_group * 4096 + seed_fast_packet_k],
                                    valid_shape=[seed_fast_valid, 256],
                                )
                                seed_fast_packet_w = wo_a[
                                    seed_group : seed_group + 1,
                                    seed_fast_packet_k : seed_fast_packet_k + 256,
                                    seed_col : seed_col + 128,
                                ]
                                seed_fast_acc = pl.matmul_acc(
                                    seed_fast_acc, seed_fast_packet_x, seed_fast_packet_w, init_cond=False
                                )
                            else:
                                for seed_fast_inner in pl.range(16):
                                    seed_fast_wrap_k = (seed_fast_packet_kb + seed_fast_inner) % 256 * 16
                                    seed_fast_wrap_x = pl.slice(
                                        heads,
                                        [128, 16],
                                        [seed_fast_row, seed_group * 4096 + seed_fast_wrap_k],
                                        valid_shape=[seed_fast_valid, 16],
                                    )
                                    seed_fast_wrap_w = wo_a[
                                        seed_group : seed_group + 1,
                                        seed_fast_wrap_k : seed_fast_wrap_k + 16,
                                        seed_col : seed_col + 128,
                                    ]
                                    seed_fast_acc = pl.matmul_acc(
                                        seed_fast_acc, seed_fast_wrap_x, seed_fast_wrap_w, init_cond=False
                                    )
                        accum = pl.assemble(accum, seed_fast_acc, [seed_fast_row, seed_group * 1024 + seed_col])
                    else:
                        for seed_local_rb in pl.range(8):
                            seed_narrow_rb = seed_first_rb + seed_local_rb
                            if seed_narrow_rb < row_tiles:
                                seed_narrow_row = seed_narrow_rb * 16
                                seed_narrow_valid = pl.min(16, rows - seed_narrow_row)
                                seed_narrow_first_block = pl.max(
                                    pl.read(start, [seed_narrow_rb, seed_group, seed_block]), 0
                                )
                                seed_narrow_acc = pl.create_tensor([1, 16, 128], dtype=pl.FP32)
                                for seed_narrow_packet in pl.range(16):
                                    seed_narrow_packet_kb = (seed_narrow_first_block + seed_narrow_packet * 16) % 256
                                    if seed_narrow_packet_kb <= 240:
                                        seed_narrow_packet_k = seed_narrow_packet_kb * 16
                                        seed_narrow_packet_x = pl.slice(
                                            heads,
                                            [16, 256],
                                            [seed_narrow_row, seed_group * 4096 + seed_narrow_packet_k],
                                            valid_shape=[seed_narrow_valid, 256],
                                        )
                                        seed_narrow_packet_w = wo_a[
                                            seed_group : seed_group + 1,
                                            seed_narrow_packet_k : seed_narrow_packet_k + 256,
                                            seed_col : seed_col + 128,
                                        ]
                                        seed_narrow_acc = pl.matmul_acc(
                                            seed_narrow_acc,
                                            seed_narrow_packet_x,
                                            seed_narrow_packet_w,
                                            init_cond=seed_narrow_packet == 0,
                                        )
                                    else:
                                        for seed_narrow_inner in pl.range(16):
                                            seed_narrow_wrap_k = (seed_narrow_packet_kb + seed_narrow_inner) % 256 * 16
                                            seed_narrow_wrap_x = pl.slice(
                                                heads,
                                                [16, 16],
                                                [seed_narrow_row, seed_group * 4096 + seed_narrow_wrap_k],
                                                valid_shape=[seed_narrow_valid, 16],
                                            )
                                            seed_narrow_wrap_w = wo_a[
                                                seed_group : seed_group + 1,
                                                seed_narrow_wrap_k : seed_narrow_wrap_k + 16,
                                                seed_col : seed_col + 128,
                                            ]
                                            seed_narrow_acc = pl.matmul_acc(
                                                seed_narrow_acc,
                                                seed_narrow_wrap_x,
                                                seed_narrow_wrap_w,
                                                init_cond=seed_narrow_packet * 16 + seed_narrow_inner == 0,
                                            )
                                accum = pl.assemble(
                                    accum, seed_narrow_acc, [seed_narrow_row, seed_group * 1024 + seed_col]
                                )
    else:
        for v4_group in pl.parallel(8):
            for v4_block in pl.spmd(8, name_hint="oproj_a"):
                v4_col = v4_block * 128
                for v4_panel in pl.range((row_tiles + 7) // 8):
                    v4_first_rb = v4_panel * 8
                    v4_first_start = pl.max(pl.read(start, [v4_first_rb, v4_group, v4_block]), 0)
                    v4_same_start = 1
                    if v4_first_rb + 1 < row_tiles:
                        if pl.max(pl.read(start, [v4_first_rb + 1, v4_group, v4_block]), 0) != v4_first_start:
                            v4_same_start = 0
                    if v4_first_rb + 2 < row_tiles:
                        if pl.max(pl.read(start, [v4_first_rb + 2, v4_group, v4_block]), 0) != v4_first_start:
                            v4_same_start = 0
                    if v4_first_rb + 3 < row_tiles:
                        if pl.max(pl.read(start, [v4_first_rb + 3, v4_group, v4_block]), 0) != v4_first_start:
                            v4_same_start = 0
                    if v4_first_rb + 4 < row_tiles:
                        if pl.max(pl.read(start, [v4_first_rb + 4, v4_group, v4_block]), 0) != v4_first_start:
                            v4_same_start = 0
                    if v4_first_rb + 5 < row_tiles:
                        if pl.max(pl.read(start, [v4_first_rb + 5, v4_group, v4_block]), 0) != v4_first_start:
                            v4_same_start = 0
                    if v4_first_rb + 6 < row_tiles:
                        if pl.max(pl.read(start, [v4_first_rb + 6, v4_group, v4_block]), 0) != v4_first_start:
                            v4_same_start = 0
                    if v4_first_rb + 7 < row_tiles:
                        if pl.max(pl.read(start, [v4_first_rb + 7, v4_group, v4_block]), 0) != v4_first_start:
                            v4_same_start = 0
                    if rows <= 16:
                        v4_same_start = 0
                    if v4_same_start == 1:
                        v4_wide_row = v4_first_rb * 16
                        v4_wide_valid = pl.min(128, rows - v4_wide_row)
                        v4_wide_first_block = v4_first_start
                        v4_wide_seed_k = v4_wide_first_block % 256 * 16
                        v4_wide_seed_x = pl.slice(
                            heads,
                            [128, 16],
                            [v4_wide_row, v4_group * 4096 + v4_wide_seed_k],
                            valid_shape=[v4_wide_valid, 16],
                        )
                        v4_wide_seed_w = wo_a[
                            v4_group : v4_group + 1, v4_wide_seed_k : v4_wide_seed_k + 16, v4_col : v4_col + 128
                        ]
                        v4_wide_acc = pl.matmul(v4_wide_seed_x, v4_wide_seed_w, out_dtype=pl.FP32)
                        for v4_wide_packet in pl.pipeline(15, stage=2):
                            v4_wide_packet_kb = (v4_wide_first_block + 1 + v4_wide_packet * 16) % 256
                            if v4_wide_packet_kb <= 240:
                                v4_wide_packet_k = v4_wide_packet_kb * 16
                                v4_wide_packet_x = pl.slice(
                                    heads,
                                    [128, 256],
                                    [v4_wide_row, v4_group * 4096 + v4_wide_packet_k],
                                    valid_shape=[v4_wide_valid, 256],
                                )
                                v4_wide_packet_w = wo_a[
                                    v4_group : v4_group + 1,
                                    v4_wide_packet_k : v4_wide_packet_k + 256,
                                    v4_col : v4_col + 128,
                                ]
                                v4_wide_acc = pl.matmul_acc(
                                    v4_wide_acc, v4_wide_packet_x, v4_wide_packet_w, init_cond=False
                                )
                            else:
                                for v4_wide_inner in pl.range(16):
                                    v4_wide_wrap_k = (v4_wide_packet_kb + v4_wide_inner) % 256 * 16
                                    v4_wide_wrap_x = pl.slice(
                                        heads,
                                        [128, 16],
                                        [v4_wide_row, v4_group * 4096 + v4_wide_wrap_k],
                                        valid_shape=[v4_wide_valid, 16],
                                    )
                                    v4_wide_wrap_w = wo_a[
                                        v4_group : v4_group + 1,
                                        v4_wide_wrap_k : v4_wide_wrap_k + 16,
                                        v4_col : v4_col + 128,
                                    ]
                                    v4_wide_acc = pl.matmul_acc(
                                        v4_wide_acc, v4_wide_wrap_x, v4_wide_wrap_w, init_cond=False
                                    )
                        for v4_wide_tail in pl.range(241, 256):
                            v4_wide_tail_k = (v4_wide_first_block + v4_wide_tail) % 256 * 16
                            v4_wide_tail_x = pl.slice(
                                heads,
                                [128, 16],
                                [v4_wide_row, v4_group * 4096 + v4_wide_tail_k],
                                valid_shape=[v4_wide_valid, 16],
                            )
                            v4_wide_tail_w = wo_a[
                                v4_group : v4_group + 1, v4_wide_tail_k : v4_wide_tail_k + 16, v4_col : v4_col + 128
                            ]
                            v4_wide_acc = pl.matmul_acc(v4_wide_acc, v4_wide_tail_x, v4_wide_tail_w, init_cond=False)
                        accum = pl.assemble(accum, v4_wide_acc, [v4_wide_row, v4_group * 1024 + v4_col])
                    else:
                        for v4_local_rb in pl.range(8):
                            v4_narrow_rb = v4_first_rb + v4_local_rb
                            if v4_narrow_rb < row_tiles:
                                v4_narrow_row = v4_narrow_rb * 16
                                v4_narrow_valid = pl.min(16, rows - v4_narrow_row)
                                v4_narrow_first_block = pl.max(pl.read(start, [v4_narrow_rb, v4_group, v4_block]), 0)
                                v4_narrow_acc = pl.create_tensor([1, 16, 128], dtype=pl.FP32)
                                for v4_narrow_packet in pl.range(16):
                                    v4_narrow_packet_kb = (v4_narrow_first_block + v4_narrow_packet * 16) % 256
                                    if v4_narrow_packet_kb <= 240:
                                        v4_narrow_packet_k = v4_narrow_packet_kb * 16
                                        v4_narrow_packet_x = pl.slice(
                                            heads,
                                            [16, 256],
                                            [v4_narrow_row, v4_group * 4096 + v4_narrow_packet_k],
                                            valid_shape=[v4_narrow_valid, 256],
                                        )
                                        v4_narrow_packet_w = wo_a[
                                            v4_group : v4_group + 1,
                                            v4_narrow_packet_k : v4_narrow_packet_k + 256,
                                            v4_col : v4_col + 128,
                                        ]
                                        v4_narrow_acc = pl.matmul_acc(
                                            v4_narrow_acc,
                                            v4_narrow_packet_x,
                                            v4_narrow_packet_w,
                                            init_cond=v4_narrow_packet == 0,
                                        )
                                    else:
                                        for v4_narrow_inner in pl.range(16):
                                            v4_narrow_wrap_k = (v4_narrow_packet_kb + v4_narrow_inner) % 256 * 16
                                            v4_narrow_wrap_x = pl.slice(
                                                heads,
                                                [16, 16],
                                                [v4_narrow_row, v4_group * 4096 + v4_narrow_wrap_k],
                                                valid_shape=[v4_narrow_valid, 16],
                                            )
                                            v4_narrow_wrap_w = wo_a[
                                                v4_group : v4_group + 1,
                                                v4_narrow_wrap_k : v4_narrow_wrap_k + 16,
                                                v4_col : v4_col + 128,
                                            ]
                                            v4_narrow_acc = pl.matmul_acc(
                                                v4_narrow_acc,
                                                v4_narrow_wrap_x,
                                                v4_narrow_wrap_w,
                                                init_cond=v4_narrow_packet * 16 + v4_narrow_inner == 0,
                                            )
                                accum = pl.assemble(accum, v4_narrow_acc, [v4_narrow_row, v4_group * 1024 + v4_col])
    for block in pl.spmd(row_tiles * 8, name_hint="oproj_a_bf16"):
        row = block // 8 * 16
        col = block % 8 * 1024
        valid = pl.min(16, rows - row)
        oa_value = pl.load(accum, [row, col], [16, 1024], valid_shape=[valid, 1024])
        pl.store(pl.cast(oa_value, pl.BF16, mode="rint"), [row, col], projected)
    quant_rows = (rows + 7) // 8
    for block in pl.spmd(quant_rows, name_hint="oproj_quant"):
        row = block * 8
        valid = pl.min(8, rows - row)
        first_quant = pl.load(projected, [row, 0], [8, 1024], valid_shape=[valid, 1024])
        first_abs = pl.abs(pl.cast(first_quant, pl.FP32))
        amax = pl.row_max(first_abs, pl.create_tile([8, 1024], dtype=pl.FP32))
        for group in pl.range(1, 8):
            q_bf16 = pl.load(projected, [row, group * 1024], [8, 1024], valid_shape=[valid, 1024])
            q_fp32 = pl.cast(q_bf16, pl.FP32)
            amax = pl.maximum(amax, pl.row_max(pl.abs(q_fp32), pl.create_tile([8, 1024], dtype=pl.FP32)))
        dequant = pl.mul(amax, 1.0 / 127.0)
        denominator = pl.add(amax, 0.0)
        for lane in pl.range(valid):
            magnitude = pl.read(denominator, [lane, 0])
            if magnitude == 0.0:
                pl.write(denominator, [lane, 0], 1.0)
        multiplier = pl.div(pl.add(pl.mul(amax, 0.0), 127.0), denominator)
        pl.store(pl.set_validshape(dequant, valid, 1), [row, 0], token_scale)
        for group in pl.range(8):
            col = group * 1024
            quant_bf16 = pl.load(projected, [row, col], [8, 1024], valid_shape=[valid, 1024])
            quant_scaled = pl.row_expand_mul(pl.cast(quant_bf16, pl.FP32), multiplier)
            integer = pl.cast(quant_scaled, pl.INT32, mode="rint")
            half = pl.cast(integer, pl.FP16, mode="round")
            result = pl.cast(half, pl.INT8, mode="trunc")
            pl.store(pl.set_validshape(result, valid, 1024), [row, col], quantized)
    integer_accum = pl.create_tensor([rows + 63, 4096], dtype=pl.INT32)
    if rows <= 16:
        for block in pl.spmd(row_tiles * 32, name_hint="oproj_b"):
            row = block // 32 * 16
            col = pl.max(block % 32, 0) * 128
            valid = pl.min(16, rows - row)
            b_acc = pl.create_tensor([16, 128], dtype=pl.INT32)
            for step in pl.range(16):
                k = step * 512
                b_input = pl.slice(quantized, [16, 512], [row, k], valid_shape=[valid, 512])
                b_weight = wo_b[k : k + 512, col : col + 128]
                b_acc = pl.matmul_acc(b_acc, b_input, b_weight, init_cond=step == 0)
            integer_accum[row : row + 16, col : col + 128] = b_acc
    else:
        for wide_block in pl.spmd((rows + 63) // 64 * 32, name_hint="oproj_b64"):
            wide_row = wide_block // 32 * 64
            wide_col = pl.max(wide_block % 32, 0) * 128
            wide_valid = pl.min(64, rows - wide_row)
            wide_first_input = pl.slice(quantized, [64, 512], [wide_row, 0], valid_shape=[wide_valid, 512])
            wide_first_weight = wo_b[0:512, wide_col : wide_col + 128]
            wide_b_acc = pl.matmul(wide_first_input, wide_first_weight, out_dtype=pl.INT32)
            for wide_step in pl.range(1, 16):
                wide_k = wide_step * 512
                wide_b_input = pl.slice(quantized, [64, 512], [wide_row, wide_k], valid_shape=[wide_valid, 512])
                wide_b_weight = wo_b[wide_k : wide_k + 512, wide_col : wide_col + 128]
                wide_b_acc = pl.matmul_acc(wide_b_acc, wide_b_input, wide_b_weight, init_cond=False)
            integer_accum = pl.assemble(integer_accum, wide_b_acc, [wide_row, wide_col])
    residual_flat = pl.reshape(residual, [rows, 16384])
    output_flat = pl.reshape(output, [rows, 16384])
    # Match Native channel*token scale and the attended BF16 publication.
    # Fusion only removes the attended GM store/reload and task handoff.
    for token_block in pl.spmd((rows + 3) // 4, name_hint="oproj_dequant_hcpost"):
        first = token_block * 4
        for row in pl.pipeline(first, first + 4, stage=2):
            if row < rows:
                ob_integer = pl.load(integer_accum, [row, 0], [1, 4096])
                channel = pl.reshape(pl.load(wo_b_scale, [0], [4096]), [1, 4096])
                scale = pl.read(token_scale, [row, 0])
                coefficient_one = pl.tile.full([1, 4096], dtype=pl.FP32, value=1.0)
                coefficient_channel = pl.col_expand_mul(coefficient_one, channel)
                coefficient = pl.mul(coefficient_channel, scale)
                ob_scaled = pl.mul(pl.cast(ob_integer, pl.FP32), coefficient)
                attended_bf16 = pl.cast(ob_scaled, pl.BF16, mode="rint")
                x_f32 = pl.cast(attended_bf16, pl.FP32)
                res_0 = pl.cast(pl.load(residual_flat, [row, 0], [1, 4096], valid_shape=[1, 4096]), pl.FP32)
                res_1 = pl.cast(pl.load(residual_flat, [row, 4096 + 0], [1, 4096], valid_shape=[1, 4096]), pl.FP32)
                res_2 = pl.cast(pl.load(residual_flat, [row, 2 * 4096 + 0], [1, 4096], valid_shape=[1, 4096]), pl.FP32)
                res_3 = pl.cast(pl.load(residual_flat, [row, 3 * 4096 + 0], [1, 4096], valid_shape=[1, 4096]), pl.FP32)
                for out_h in pl.unroll(4):
                    post_scalar = pl.read(post, [row, out_h])
                    single_y = pl.mul(x_f32, post_scalar)
                    comb_scalar_0 = pl.read(comb, [row, out_h])
                    single_weighted_0 = pl.mul(res_0, comb_scalar_0)
                    single_y = pl.add(single_y, single_weighted_0)
                    comb_scalar_1 = pl.read(comb, [row, 4 + out_h])
                    single_weighted_1 = pl.mul(res_1, comb_scalar_1)
                    single_y = pl.add(single_y, single_weighted_1)
                    comb_scalar_2 = pl.read(comb, [row, 2 * 4 + out_h])
                    single_weighted_2 = pl.mul(res_2, comb_scalar_2)
                    single_y = pl.add(single_y, single_weighted_2)
                    comb_scalar_3 = pl.read(comb, [row, 3 * 4 + out_h])
                    single_weighted_3 = pl.mul(res_3, comb_scalar_3)
                    single_y = pl.add(single_y, single_weighted_3)
                    single_result = pl.cast(single_y, pl.BF16, mode="rint")
                    pl.store(pl.set_validshape(single_result, 1, 4096), [row, out_h * 4096], output_flat)
    return output


o_proj_hcpost_entry = pl.jit.inline(auto_scope=False)(_o_proj_hcpost)
o_proj_hcpost_test = pl.jit(_o_proj_hcpost)
