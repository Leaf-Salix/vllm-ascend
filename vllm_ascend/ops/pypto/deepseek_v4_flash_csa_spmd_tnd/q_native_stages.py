# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""单SPMD内部的Native QR边界；保留nalinaly的M8所有权和暂存布局。

归约/开方与量化顺序参考旧精度版q_proj_qr_normalize，不创建子任务或Native调用。
"""

import pypto.language as pl

from .qkv_stages import (
    EPS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    MATMUL_ROWS_DYN,
    Q_LORA,
    Q_LORA_TILE,
    QA_ROWS_DYN,
    QUANT_TILE,
    SCALE_ROWS_DYN,
    T_DYN,
    T_TILE,
)

FP32_FRACTION_UNIT = 1 << 23
FP32_SQRT_COMPARE_SHIFT = 152
FP32_INFINITY_BITS = 0x7F800000
QR_SQRT_CORRECTION_STEPS = 2


@pl.jit.inline(auto_scope=False)
def q_r_native_inverse(
    qr_fp32: pl.Tensor[[QA_ROWS_DYN, Q_LORA], pl.FP32],
    row: pl.Scalar[pl.INDEX],
):
    """Native QR reduction and device scalar reciprocal, shared by diagnostics."""
    qr_rms_input = pl.load(qr_fp32, [row, 0], [T_TILE, Q_LORA], target_memory=pl.MemorySpace.Vec)
    qr_rms_full = pl.cast(pl.cast(qr_rms_input, pl.BF16, mode="rint"), pl.FP32)
    qr_square_full = pl.mul(qr_rms_full, qr_rms_full)
    qr_square_half = pl.add(qr_square_full[:, 0:512], qr_square_full[:, 512:1024])
    qr_square_quarter = pl.add(qr_square_half[:, 0:256], qr_square_half[:, 256:512])
    qr_square_eighth = pl.add(qr_square_quarter[:, 0:128], qr_square_quarter[:, 128:256])
    qr_square_sixteenth = pl.add(qr_square_eighth[:, 0:64], qr_square_eighth[:, 64:128])
    qr_sq_sum = pl.reshape(
        pl.row_sum(qr_square_sixteenth, tmp_tile=pl.tile.create([T_TILE, 64], dtype=pl.FP32)), [1, T_TILE]
    )
    qr_variance = pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)
    qr_rms_approx = pl.sqrt(qr_variance)
    qr_variance_bits = pl.reinterpret_view(qr_variance, pl.INT32)
    qr_approx_bits = pl.reinterpret_view(qr_rms_approx, pl.INT32)
    qr_rounded_bits = pl.tile.full([1, T_TILE], dtype=pl.INT32, value=0)
    # A3 VSQRT can differ from Native scalar sqrt by one FP32 ULP.
    # Compare x against the squared midpoints around the candidate root.
    # Their integer significands fit in INT64, so this rounding decision
    # introduces no additional floating-point error. EPS keeps x normal.
    for qr_sqrt_row in pl.range(T_TILE):
        qr_x_bits = pl.cast(pl.read(qr_variance_bits, [0, qr_sqrt_row]), pl.INT64)
        qr_y_bits = pl.cast(pl.read(qr_approx_bits, [0, qr_sqrt_row]), pl.INT64)
        if qr_x_bits >= FP32_FRACTION_UNIT and qr_x_bits < FP32_INFINITY_BITS:
            qr_x_exp = qr_x_bits // FP32_FRACTION_UNIT
            qr_x_sig = qr_x_bits % FP32_FRACTION_UNIT + FP32_FRACTION_UNIT
            for _qr_sqrt_step in pl.range(QR_SQRT_CORRECTION_STEPS):
                qr_y_exp = qr_y_bits // FP32_FRACTION_UNIT
                qr_y_sig = qr_y_bits % FP32_FRACTION_UNIT + FP32_FRACTION_UNIT
                qr_upper_mid = 2 * qr_y_sig + 1
                qr_upper_square = qr_upper_mid * qr_upper_mid
                qr_x_upper = qr_x_sig << (qr_x_exp - 2 * qr_y_exp + FP32_SQRT_COMPARE_SHIFT)
                qr_previous_bits = qr_y_bits - 1
                qr_previous_exp = qr_previous_bits // FP32_FRACTION_UNIT
                qr_previous_sig = qr_previous_bits % FP32_FRACTION_UNIT + FP32_FRACTION_UNIT
                qr_lower_mid = 2 * qr_previous_sig + 1
                qr_lower_square = qr_lower_mid * qr_lower_mid
                qr_x_lower = qr_x_sig << (qr_x_exp - 2 * qr_previous_exp + FP32_SQRT_COMPARE_SHIFT)
                if qr_x_upper > qr_upper_square or (qr_x_upper == qr_upper_square and qr_y_bits % 2 == 1):
                    qr_y_bits = qr_y_bits + 1
                elif qr_x_lower < qr_lower_square or (qr_x_lower == qr_lower_square and qr_y_bits % 2 == 1):
                    qr_y_bits = qr_y_bits - 1
            pl.write(qr_rounded_bits, [0, qr_sqrt_row], pl.cast(qr_y_bits, pl.INT32))
        else:
            pl.write(qr_rounded_bits, [0, qr_sqrt_row], pl.cast(qr_y_bits, pl.INT32))
    qr_rms = pl.reinterpret_view(qr_rounded_bits, pl.FP32)
    qr_inv_rms = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
    # A2/A3 TDIV uses vdiv even when high_precision is requested.
    # Match Native's scalar device division, without a host value read.
    for qr_rms_row in pl.range(T_TILE):
        qr_rms_value = pl.read(qr_rms, [0, qr_rms_row])
        pl.write(qr_inv_rms, [0, qr_rms_row], 1.0 / qr_rms_value)
    return qr_sq_sum, qr_inv_rms, qr_rms


@pl.jit.inline(auto_scope=False)
def q_r_native_stage(
    qa_fp32: pl.Tensor[[QA_ROWS_DYN, Q_LORA], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Tensor[[MATMUL_ROWS_DYN, Q_LORA], pl.INT8],
    qr_scale_pad: pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    # Keep the original M8 worker mapping, tail stores and publication boundary.
    tokens = pl.tensor.dim(qr, 0)
    token_tiles = (tokens + T_TILE - 1) // T_TILE
    workers = pl.min(token_tiles, WORKERS)
    if worker_id < workers:
        gamma_row = pl.reshape(pl.load(gamma_cq, [0], [Q_LORA], target_memory=pl.MemorySpace.Vec), [1, Q_LORA])
        gamma_fp32 = pl.cast(gamma_row, pl.FP32)
        for tile_index in pl.range(worker_id, token_tiles, workers):
            tg = tile_index * T_TILE
            valid_rows = pl.min(T_TILE, tokens - tg)
            raw = pl.load(qa_fp32, [tg, 0], [T_TILE, Q_LORA], target_memory=pl.MemorySpace.Vec)
            qa_bf16 = pl.cast(pl.cast(raw, pl.BF16, mode="rint"), pl.FP32)
            _square, inv_row, _root = q_r_native_inverse(qa_fp32, tg)
            inv = pl.reshape(inv_row, [T_TILE, 1])
            tile_amax = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            # Native takes amax after (QA * inverse) * gamma. Do not factor out inverse.
            for col in pl.pipeline(0, Q_LORA, Q_LORA_TILE, stage=2):
                chunk = pl.tile.extract(qa_bf16, 0, col, [T_TILE, Q_LORA_TILE], target_memory=pl.MemorySpace.Vec)
                weight = pl.tile.extract(gamma_fp32, 0, col, [1, Q_LORA_TILE], target_memory=pl.MemorySpace.Vec)
                normalized = pl.col_expand_mul(pl.row_expand_mul(chunk, inv), weight)
                chunk_amax = pl.reshape(
                    pl.row_max(pl.abs(normalized), tmp_tile=pl.tile.create([T_TILE, Q_LORA_TILE], dtype=pl.FP32)),
                    [1, T_TILE],
                )
                tile_amax = pl.maximum(tile_amax, chunk_amax)
            quant_row = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            dequant_row = pl.tile.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for lane in pl.range(T_TILE):
                amax = pl.read(tile_amax, [0, lane])
                quant = INT8_SCALE_MAX / amax
                pl.write(quant_row, [0, lane], quant)
                pl.write(dequant_row, [0, lane], 1.0 / quant)
            quant_column = pl.reshape(quant_row, [T_TILE, 1])
            dequant_column = pl.reshape(dequant_row, [T_TILE, 1])
            pl.store(dequant_column, [tg, 0], qr_scale_pad)
            if valid_rows == T_TILE:
                pl.store(dequant_column, [tg, 0], qr_scale)
            else:
                pl.store(pl.set_validshape(dequant_column, valid_rows, 1), [tg, 0], qr_scale)
            for col in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
                chunk = pl.tile.extract(qa_bf16, 0, col, [T_TILE, QUANT_TILE], target_memory=pl.MemorySpace.Vec)
                weight = pl.tile.extract(gamma_fp32, 0, col, [1, QUANT_TILE], target_memory=pl.MemorySpace.Vec)
                normalized = pl.col_expand_mul(pl.row_expand_mul(chunk, inv), weight)
                scaled = pl.row_expand_mul(normalized, quant_column)
                rounded = pl.cast(scaled, pl.INT32, mode="rint")
                half = pl.cast(rounded, pl.FP16, mode="round")
                quantized = pl.cast(half, pl.INT8, mode="trunc")
                pl.store(quantized, [tg, col], qr_i8_matmul)
                if valid_rows == T_TILE:
                    pl.store(quantized, [tg, col], qr)
                else:
                    pl.store(pl.set_validshape(quantized, valid_rows, QUANT_TILE), [tg, col], qr)
    return qr, qr_scale, qr_i8_matmul, qr_scale_pad
