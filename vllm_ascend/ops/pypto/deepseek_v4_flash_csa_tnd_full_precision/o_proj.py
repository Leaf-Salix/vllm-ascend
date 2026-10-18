# SPDX-License-Identifier: Apache-2.0
"""Native WO-A BF16 → 整行量化 → INT32 WO-B → 通道/行 scale → BF16。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.nz_mode import (
    QUANT_WEIGHT_LAYOUT,
    WO_A_WEIGHT_LAYOUT,
)

T = pl.dynamic("FULL_PRECISION_OPROJ_T")


def _o_proj_test(
    heads: pl.Tensor[[T, 32768], pl.BF16],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    start: pl.Tensor[[16, 8, 8], pl.INT32],
    projected: pl.Out[pl.Tensor[[T, 8192], pl.BF16]],
    quantized: pl.Out[pl.Tensor[[T, 8192], pl.INT8]],
    token_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    output: pl.Out[pl.Tensor[[T, 4096], pl.BF16]],
):
    heads.bind_dynamic(0, T)
    projected.bind_dynamic(0, T)
    quantized.bind_dynamic(0, T)
    token_scale.bind_dynamic(0, T)
    output.bind_dynamic(0, T)
    rows = pl.tensor.dim(heads, 0)
    row_tiles = (rows + 15) // 16
    accum = pl.create_tensor([rows + 127, 8192], dtype=pl.FP32)
    for group in pl.parallel(8):
        for block in pl.spmd(8, name_hint="oproj_a"):
            col = block * 128
            for panel in pl.range((row_tiles + 7) // 8):
                first_rb = panel * 8
                first_start = pl.max(pl.read(start, [first_rb, group, block]), 0)
                same_start = 1
                if first_rb + 1 < row_tiles:
                    if pl.max(pl.read(start, [first_rb + 1, group, block]), 0) != first_start:
                        same_start = 0
                if first_rb + 2 < row_tiles:
                    if pl.max(pl.read(start, [first_rb + 2, group, block]), 0) != first_start:
                        same_start = 0
                if first_rb + 3 < row_tiles:
                    if pl.max(pl.read(start, [first_rb + 3, group, block]), 0) != first_start:
                        same_start = 0
                if first_rb + 4 < row_tiles:
                    if pl.max(pl.read(start, [first_rb + 4, group, block]), 0) != first_start:
                        same_start = 0
                if first_rb + 5 < row_tiles:
                    if pl.max(pl.read(start, [first_rb + 5, group, block]), 0) != first_start:
                        same_start = 0
                if first_rb + 6 < row_tiles:
                    if pl.max(pl.read(start, [first_rb + 6, group, block]), 0) != first_start:
                        same_start = 0
                if first_rb + 7 < row_tiles:
                    if pl.max(pl.read(start, [first_rb + 7, group, block]), 0) != first_start:
                        same_start = 0
                if rows <= 16:
                    same_start = 0
                if same_start == 1:
                    wide_row = first_rb * 16
                    wide_valid = pl.min(128, rows - wide_row)
                    wide_first_block = first_start
                    wide_seed_k = wide_first_block % 256 * 16
                    wide_seed_x = pl.slice(
                        heads, [128, 16], [wide_row, group * 4096 + wide_seed_k], valid_shape=[wide_valid, 16]
                    )
                    wide_seed_w = wo_a[group : group + 1, wide_seed_k : wide_seed_k + 16, col : col + 128]
                    wide_acc = pl.matmul(wide_seed_x, wide_seed_w, out_dtype=pl.FP32)
                    for wide_packet in pl.pipeline(15, stage=2):
                        wide_packet_kb = (wide_first_block + 1 + wide_packet * 16) % 256
                        if wide_packet_kb <= 240:
                            wide_packet_k = wide_packet_kb * 16
                            wide_packet_x = pl.slice(
                                heads,
                                [128, 256],
                                [wide_row, group * 4096 + wide_packet_k],
                                valid_shape=[wide_valid, 256],
                            )
                            wide_packet_w = wo_a[
                                group : group + 1, wide_packet_k : wide_packet_k + 256, col : col + 128
                            ]
                            wide_acc = pl.matmul_acc(wide_acc, wide_packet_x, wide_packet_w, init_cond=False)
                        else:
                            for wide_inner in pl.range(16):
                                wide_wrap_k = (wide_packet_kb + wide_inner) % 256 * 16
                                wide_wrap_x = pl.slice(
                                    heads,
                                    [128, 16],
                                    [wide_row, group * 4096 + wide_wrap_k],
                                    valid_shape=[wide_valid, 16],
                                )
                                wide_wrap_w = wo_a[group : group + 1, wide_wrap_k : wide_wrap_k + 16, col : col + 128]
                                wide_acc = pl.matmul_acc(wide_acc, wide_wrap_x, wide_wrap_w, init_cond=False)
                    for wide_tail in pl.range(241, 256):
                        wide_tail_k = (wide_first_block + wide_tail) % 256 * 16
                        wide_tail_x = pl.slice(
                            heads, [128, 16], [wide_row, group * 4096 + wide_tail_k], valid_shape=[wide_valid, 16]
                        )
                        wide_tail_w = wo_a[group : group + 1, wide_tail_k : wide_tail_k + 16, col : col + 128]
                        wide_acc = pl.matmul_acc(wide_acc, wide_tail_x, wide_tail_w, init_cond=False)
                    accum = pl.assemble(accum, wide_acc, [wide_row, group * 1024 + col])
                else:
                    for local_rb in pl.range(8):
                        narrow_rb = first_rb + local_rb
                        if narrow_rb < row_tiles:
                            narrow_row = narrow_rb * 16
                            narrow_valid = pl.min(16, rows - narrow_row)
                            narrow_first_block = pl.max(pl.read(start, [narrow_rb, group, block]), 0)
                            narrow_acc = pl.create_tensor([1, 16, 128], dtype=pl.FP32)
                            for narrow_packet in pl.range(16):
                                narrow_packet_kb = (narrow_first_block + narrow_packet * 16) % 256
                                if narrow_packet_kb <= 240:
                                    narrow_packet_k = narrow_packet_kb * 16
                                    narrow_packet_x = pl.slice(
                                        heads,
                                        [16, 256],
                                        [narrow_row, group * 4096 + narrow_packet_k],
                                        valid_shape=[narrow_valid, 256],
                                    )
                                    narrow_packet_w = wo_a[
                                        group : group + 1, narrow_packet_k : narrow_packet_k + 256, col : col + 128
                                    ]
                                    narrow_acc = pl.matmul_acc(
                                        narrow_acc, narrow_packet_x, narrow_packet_w, init_cond=narrow_packet == 0
                                    )
                                else:
                                    for narrow_inner in pl.range(16):
                                        narrow_wrap_k = (narrow_packet_kb + narrow_inner) % 256 * 16
                                        narrow_wrap_x = pl.slice(
                                            heads,
                                            [16, 16],
                                            [narrow_row, group * 4096 + narrow_wrap_k],
                                            valid_shape=[narrow_valid, 16],
                                        )
                                        narrow_wrap_w = wo_a[
                                            group : group + 1, narrow_wrap_k : narrow_wrap_k + 16, col : col + 128
                                        ]
                                        narrow_acc = pl.matmul_acc(
                                            narrow_acc,
                                            narrow_wrap_x,
                                            narrow_wrap_w,
                                            init_cond=narrow_packet * 16 + narrow_inner == 0,
                                        )
                            accum = pl.assemble(accum, narrow_acc, [narrow_row, group * 1024 + col])
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
                b_acc = pl.matmul_acc(b_acc, b_input, b_weight, init_cond=(step == 0))
            integer_accum[row : row + 16, col : col + 128] = b_acc
    else:
        for wide_block in pl.spmd(((rows + 63) // 64) * 32, name_hint="oproj_b64"):
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
    for block in pl.spmd(row_tiles * 8, name_hint="oproj_b_dequant"):
        row = block // 8 * 16
        col = block % 8 * 512
        valid = pl.min(16, rows - row)
        ob_integer = pl.load(integer_accum, [row, col], [16, 512], valid_shape=[valid, 512])
        channel = pl.reshape(pl.load(wo_b_scale, [col], [512]), [1, 512])
        scale = pl.load(token_scale, [row, 0], [16, 1], valid_shape=[valid, 1])
        coefficient_one = pl.tile.full([16, 512], dtype=pl.FP32, value=1.0)
        coefficient_channel = pl.col_expand_mul(coefficient_one, channel)
        coefficient = pl.row_expand_mul(coefficient_channel, scale)
        ob_scaled = pl.mul(pl.cast(ob_integer, pl.FP32), coefficient)
        pl.store(pl.cast(ob_scaled, pl.BF16, mode="rint"), [row, col], output)
    return output


o_proj_entry = pl.jit.inline(auto_scope=False)(_o_proj_test)
o_proj_test = pl.jit(_o_proj_test)
