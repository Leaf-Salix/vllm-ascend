# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/qkv_proj_rope.py."""

import pypto.language as pl

from .config import (
    FLASH as M,
)
from .config import (
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)

T_DYN = pl.dynamic("QKV_Q_T_DYN")  # T = B * S

KV_T_DYN = pl.dynamic("QKV_KV_T_DYN")

ROPE_T_DYN = pl.dynamic("QKV_ROPE_T_DYN")

QPROJ_MM_T_DYN = pl.dynamic("QKV_QPROJ_MM_T_DYN")

PREFILL_DENSE_TILE = 512

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

ROPE_DIM_SCALE = float(ROPE_DIM)

ROPE_HALF = ROPE_DIM // 2

NOPE_DIM = M.nope_head_dim

Q_LORA = M.q_lora_rank

EPS = M.rms_norm_eps

MAX_SEQ_LEN = M.max_position_embeddings

Q_PROJ_TILE = 256  # qproj K-tile (Q_LORA reduction)

QPROJ_MM_N_TILE = 512  # qproj output-column tile

Q_LORA_TILE = 256  # qr rms-norm / quant N granularity

KV_TILE = 64  # kv rms-norm / rope / NOPE N granularity

QUANT_TILE = 256

T_TILE = 8

FP32_FRACTION_UNIT = 1 << 23
FP32_SQRT_COMPARE_SHIFT = 127 + 25  # Exponent bias plus midpoint-square alignment.
FP32_INFINITY_BITS = 0x7F800000
QR_SQRT_CORRECTION_STEPS = 2

MATMUL_T_TILE = 16

QR_M_TILE = MATMUL_T_TILE  # qr_proj token (M) tile; cube rows must be a 16-row boxed tile

QR_DENSE_M_TILE = 64

QR_N_TILE = 32  # Divide Native's 96-column groups without crossing K-order boundaries.

QR_K_TILE = 256  # qr_proj D (K) reduction tile   | divides QR_SPLIT_K_TILE

QR_OK = 1  # One FP32 accumulator; split-K changes Native QR rounding.

QR_SPLIT_K_TILE = D // QR_OK

# CANN 9.0 MatMulV2 NT at M96/144/192/240: paired forward/reverse K256 traversals.
# Sources: tests/pypto_test/results/cann90_20260921/qa_native_codegen_six/.
QR_NATIVE_N_GROUP = 96
QR_NATIVE_SHIFT_MIN_ROWS = 96
QR_NATIVE_SHIFT_MAX_ROWS = 240
QR_NATIVE_SHIFT_ROW_STEP = 48
QR_NATIVE_SHIFT_GROUPS = 9
QR_NATIVE_PAIR_SHIFT = 3
QR_NATIVE_K_BLOCKS = D // QR_K_TILE

KV_M_TILE = MATMUL_T_TILE  # kv_proj token (M) tile; decode pads from 8 real rows to 16

KV_DENSE_M_TILE = 64

KV_N_TILE = 128  # kv_proj HEAD_DIM (N) per matmul

KV_K_TILE = 256  # kv_proj D (K) reduction tile   | divides KV_SPLIT_K_TILE

KV_OK = 2  # kv_proj split-K factor         | D//KV_OK cores share each N-group

KV_OM = 3  # maximum kv_proj split-M factor

KV_SPLIT_K_TILE = D // KV_OK  # kv_proj K per split (=2048)

# CANN 9.0 MatMulV2 NT, M240/N512/K4096. Keep the existing schedules
# for other token counts; Native chooses different tilings for those shapes.
KV_NATIVE_ROWS = 240
KV_NATIVE_N_TILE = 32  # Does not cross a Native 96-column group.
KV_NATIVE_K_TILE = 64
KV_NATIVE_K_BLOCK = 256
KV_NATIVE_N_GROUP = 96
KV_NATIVE_SHIFT_GROUPS = 4
KV_NATIVE_PAIR_SHIFT = 5

QPROJ_M_TILE = 64  # dense qproj token tile; fills the 128 KiB L0C accumulator

QPROJ_WORKERS = 24

QPROJ_TAIL_M_TILE = MATMUL_T_TILE  # partial-M path validated by decode/small physical T

QPROJ_T_PAD = ((PREFILL_DENSE_TILE + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE

KV_RMS_T_TILE = 32  # kv rms-norm + rope fused token (T) tile

Q_ROPE_T_TILE = 8

Q_ROPE_WORKERS = 48

Q_ROPE_H_TILE = 4  # heads per fused qproj dequant/rms/rope task

Q_DEQUANT_WORKERS = 48

assert QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap

assert QPROJ_M_TILE % QPROJ_TAIL_M_TILE == 0


@pl.jit.inline
def rope_prepare(
    rope_sin: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[ROPE_T_DYN, ROPE_DIM], pl.INT32],
):
    """Fold Native interleaved sin signs and build the pair-swap indices."""
    t_dim = pl.tensor.dim(rope_sin, 0)
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    # The tail guard below writes the outputs from inside a conditional region.
    # Binding them to the scalar extent keeps the region result off the dynamic
    # symbol, which has no definition in an inlining caller that passes a
    # statically shaped token axis.
    rope_sin_signed_view = pl.reshape(rope_sin_signed, [t_dim, ROPE_DIM])
    rope_swap_idx_view = pl.reshape(rope_swap_idx, [t_dim, ROPE_DIM])

    token_tiles = (t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE
    for qrp_worker in pl.spmd(
        pl.min(Q_ROPE_WORKERS, token_tiles), name_hint="q_rope_prepare", allow_early_resolve=True
    ):
        # The pair lane, swap index and sign fold only depend on the column
        # index, so they are built once per worker instead of once per tile.
        qrp_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        for qrp_idx in pl.range(qrp_worker, token_tiles, pl.min(Q_ROPE_WORKERS, token_tiles)):
            qrp_t0 = qrp_idx * Q_ROPE_T_TILE
            qrp_valid_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
            if qrp_valid_rows == Q_ROPE_T_TILE:
                qrp_sin_rows_full = rope_sin_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :]
                qrp_sin_signed_full = pl.mul(qrp_sin_rows_full, qrp_sign)
                rope_sin_signed_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_sin_signed_full
                rope_swap_idx_view[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = qrp_swap_idx
            else:
                qrp_sin_rows_tail = pl.load(
                    rope_sin_view,
                    [qrp_t0, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[qrp_valid_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                qrp_tail_col = pl.col_expand_mul(
                    pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                    pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                qrp_tail_dup_f = pl.cast(
                    pl.cast(pl.mul(qrp_tail_col, 0.5), target_type=pl.INT32, mode="trunc"),
                    target_type=pl.FP32,
                )
                qrp_tail_lane = pl.sub(qrp_tail_col, pl.mul(qrp_tail_dup_f, 2.0))
                qrp_tail_swap_f = pl.sub(pl.add(qrp_tail_col, 1.0), pl.mul(qrp_tail_lane, 2.0))
                qrp_tail_sign = pl.sub(pl.mul(qrp_tail_lane, 2.0), 1.0)
                qrp_tail_sign = pl.set_validshape(qrp_tail_sign, qrp_valid_rows, ROPE_DIM)
                qrp_sin_signed_tail = pl.mul(qrp_sin_rows_tail, qrp_tail_sign)
                pl.store(
                    pl.set_validshape(qrp_sin_signed_tail, qrp_valid_rows, ROPE_DIM),
                    [qrp_t0, 0],
                    rope_sin_signed_view,
                )
                pl.store(
                    pl.set_validshape(
                        pl.cast(qrp_tail_swap_f, target_type=pl.INT32),
                        qrp_valid_rows,
                        ROPE_DIM,
                    ),
                    [qrp_t0, 0],
                    rope_swap_idx_view,
                )


@pl.jit.inline(auto_scope=False)
def q_proj_qa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    qr_fp32: pl.Out[pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Reference QA matmul, shared by the production QR and its diagnostic."""
    qa_tokens = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [qa_tokens, D])
    qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
    qr_full_rows = (tile_rows // QR_DENSE_M_TILE) * QR_DENSE_M_TILE
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
        for ts0 in pl.range(0, qr_t_matmul, QR_M_TILE):
            for nseed0 in pl.range(0, Q_LORA, QR_N_TILE):
                qr_seed = pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0)
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = qr_seed

    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_SPLIT_K_TILE
        qr_native_group = q_a_col0 // QR_NATIVE_N_GROUP
        for dense_t0 in pl.range(0, qr_full_rows, QR_DENSE_M_TILE):
            dense_x0 = tile_base + dense_t0
            dense_acc = pl.create_tensor([QR_DENSE_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for dense_k in pl.pipeline(0, QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                dense_k_order = dense_k
                if (
                    tile_rows >= QR_NATIVE_SHIFT_MIN_ROWS
                    and tile_rows <= QR_NATIVE_SHIFT_MAX_ROWS
                    and tile_rows % QR_NATIVE_SHIFT_ROW_STEP == 0
                    and qr_native_group < QR_NATIVE_SHIFT_GROUPS
                ):
                    dense_k_direction = dense_k
                    if qr_native_group % 2 == 1:
                        dense_k_direction = QR_NATIVE_K_BLOCKS - 1 - dense_k
                    dense_k_order = (
                        qr_native_group // 2 * QR_NATIVE_PAIR_SHIFT + dense_k_direction
                    ) % QR_NATIVE_K_BLOCKS
                dense_d0 = qr_k_base + dense_k_order * QR_K_TILE
                dense_x = x_view[dense_x0 : dense_x0 + QR_DENSE_M_TILE, dense_d0 : dense_d0 + QR_K_TILE]
                dense_w = wq_a[dense_d0 : dense_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, init_cond=(dense_k == 0))
            qr_fp32 = pl.assemble(qr_fp32, dense_acc, [dense_t0, q_a_col0], atomic=pl.AtomicType.Add)
        for t0 in pl.range(qr_full_rows, qr_t_matmul, QR_M_TILE):
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                qr_k_order = db
                if (
                    tile_rows >= QR_NATIVE_SHIFT_MIN_ROWS
                    and tile_rows <= QR_NATIVE_SHIFT_MAX_ROWS
                    and tile_rows % QR_NATIVE_SHIFT_ROW_STEP == 0
                    and qr_native_group < QR_NATIVE_SHIFT_GROUPS
                ):
                    qr_k_direction = db
                    if qr_native_group % 2 == 1:
                        qr_k_direction = QR_NATIVE_K_BLOCKS - 1 - db
                    qr_k_order = (qr_native_group // 2 * QR_NATIVE_PAIR_SHIFT + qr_k_direction) % QR_NATIVE_K_BLOCKS
                qr_d0 = qr_k_base + qr_k_order * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, tile_rows - t0)
                x_t0 = tile_base + t0
                q_x_chunk_bf16 = pl.slice(
                    x_view,
                    [QR_M_TILE, QR_K_TILE],
                    [x_t0, qr_d0],
                    valid_shape=[qr_rows, QR_K_TILE],
                )
                w_chunk = wq_a[qr_d0 : qr_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk, init_cond=(db == 0))
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)


@pl.jit.inline(auto_scope=False)
def q_proj_qr(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Out[pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8]],
    qr_scale_pad_store: pl.Out[pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32]],
):
    """Q LoRA, RMSNorm and quantization -- the half the indexer query chain needs.

    The two staging tensors are held by the caller so the qproj half can be issued
    separately; they are rewritten per dense tile, so a caller that runs more than one
    tile iteration must let the qproj half finish before the next one starts. Every
    current caller has t_dim <= PREFILL_DENSE_TILE, i.e. a single iteration.
    """
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
            qr_fp32 = pl.create_tensor([qr_t_matmul, Q_LORA], dtype=pl.FP32)
            q_proj_qa(x, wq_a, qr_fp32, tile_base, tile_rows)

            q_proj_qr_normalize(qr_fp32, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store, tile_base, tile_rows)


@pl.jit.inline(auto_scope=False)
def q_proj_qr_rms(
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    row: pl.Scalar[pl.INDEX],
):
    """Native QR reduction and device scalar reciprocal, shared by diagnostics."""
    qr_rms_full = pl.cast(pl.cast(qr_fp32[row : row + T_TILE, 0:Q_LORA], pl.BF16, mode="rint"), pl.FP32)
    qr_square_full = pl.mul(qr_rms_full, qr_rms_full)
    qr_square_half = pl.add(qr_square_full[:, 0:512], qr_square_full[:, 512:1024])
    qr_square_quarter = pl.add(qr_square_half[:, 0:256], qr_square_half[:, 256:512])
    qr_square_eighth = pl.add(qr_square_quarter[:, 0:128], qr_square_quarter[:, 128:256])
    qr_square_sixteenth = pl.add(qr_square_eighth[:, 0:64], qr_square_eighth[:, 64:128])
    qr_sq_sum = pl.reshape(pl.row_sum(qr_square_sixteenth), [1, T_TILE])
    qr_variance = pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)
    qr_rms_approx = pl.sqrt(qr_variance)
    qr_variance_bits = pl.reinterpret_view(qr_variance, pl.INT32)
    qr_approx_bits = pl.reinterpret_view(qr_rms_approx, pl.INT32)
    qr_rounded_bits = pl.full([1, T_TILE], dtype=pl.INT32, value=0)
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
    qr_inv_rms = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
    # A2/A3 TDIV uses vdiv even when high_precision is requested.
    # Match Native's scalar device division, without a host value read.
    for qr_rms_row in pl.range(T_TILE):
        qr_rms_value = pl.read(qr_rms, [0, qr_rms_row])
        pl.write(qr_inv_rms, [0, qr_rms_row], 1.0 / qr_rms_value)
    return qr_sq_sum, qr_inv_rms, qr_rms


@pl.jit.inline(auto_scope=False)
def q_proj_qr_normalize(
    qr_fp32: pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Out[pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8]],
    qr_scale_pad_store: pl.Out[pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Native QR normalization boundary, also reused by numerical diagnostics."""
    t_dim = pl.tensor.dim(qr, 0)
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])

    # Native QA materializes BF16 before fused RMSNorm + dynamic quant.
    # Preserve that rounding in both passes over the FP32 accumulation.
    # Preserve Native's operation order: RMS, normalized amax, quant.
    qr_token_tiles = (tile_rows + T_TILE - 1) // T_TILE
    for tg_idx in pl.spmd(qr_token_tiles, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        tg = tg_idx * T_TILE
        valid_rows = pl.min(T_TILE, tile_rows - tg)
        out_tg = tile_base + tg
        # Native ReduceSumHalfInterval folds 1024 -> 512 -> 256
        # -> 128 -> 64 before the hardware row reduction.
        _qr_sq_sum, qr_inv_rms, _qr_rms = q_proj_qr_rms(qr_fp32, tg)
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
        qr_tile_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for qr_max_col0 in pl.pipeline(0, Q_LORA, Q_LORA_TILE, stage=2):
            qr_max_chunk = pl.cast(
                pl.cast(qr_fp32[tg : tg + T_TILE, qr_max_col0 : qr_max_col0 + Q_LORA_TILE], pl.BF16, mode="rint"),
                pl.FP32,
            )
            gamma_max_cast = pl.cast(gamma_cq[qr_max_col0 : qr_max_col0 + Q_LORA_TILE], pl.FP32)
            gamma_max_chunk = pl.reshape(gamma_max_cast, [1, Q_LORA_TILE])
            qr_normalized = pl.col_expand_mul(pl.row_expand_mul(qr_max_chunk, qr_inv_rms_t), gamma_max_chunk)
            qr_normalized_max = pl.reshape(pl.row_max(pl.abs(qr_normalized)), [1, T_TILE])
            qr_tile_amax = pl.maximum(qr_tile_amax, qr_normalized_max)

        qr_scale_quant_row = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_scale_dq_row = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_scale_row in pl.range(T_TILE):
            qr_amax_value = pl.read(qr_tile_amax, [0, qr_scale_row])
            qr_quant_value = INT8_SCALE_MAX / qr_amax_value
            pl.write(qr_scale_quant_row, [0, qr_scale_row], qr_quant_value)
            pl.write(qr_scale_dq_row, [0, qr_scale_row], 1.0 / qr_quant_value)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_tile_scale_dq = pl.reshape(qr_scale_dq_row, [T_TILE, 1])
        qr_scale_pad_store = pl.assemble(qr_scale_pad_store, qr_tile_scale_dq, [tg, 0])
        if valid_rows == T_TILE:
            qr_scale_view[out_tg : out_tg + T_TILE, :] = qr_tile_scale_dq
        else:
            qr_scale_tail = pl.load(
                qr_scale_pad_store,
                [tg, 0],
                [T_TILE, 1],
                valid_shape=[valid_rows, 1],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(qr_scale_tail, [out_tg, 0], qr_scale_view)

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = pl.cast(pl.cast(qr_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE], pl.BF16, mode="rint"), pl.FP32)
            gamma_q_cast = pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc")
            qr_i8_matmul[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8
            if valid_rows == T_TILE:
                qr_view[out_tg : out_tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8
            else:
                qr_q_tail = pl.load(
                    qr_i8_matmul,
                    [tg, qa],
                    [T_TILE, QUANT_TILE],
                    valid_shape=[valid_rows, QUANT_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(qr_q_tail, [out_tg, qa], qr_view)


@pl.jit.inline(auto_scope=False)
def q_proj_q_matmul(
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], pl.INT32],
    tile_rows: pl.Scalar[pl.INDEX],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """Project one bounded Q tile and expose its cube task ID."""
    qproj_t_matmul = pl.tensor.dim(q_proj_i32, 0)
    qproj_full_rows = (tile_rows // QPROJ_M_TILE) * QPROJ_M_TILE
    with pl.spmd(
        QPROJ_WORKERS,
        name_hint="qproj_matmul",
        deps=[qproj_dep],
    ) as qproj_tid:
        qproj_worker = pl.tile.get_block_idx()
        for qproj_n_idx in pl.range(
            qproj_worker,
            (H * HEAD_DIM) // QPROJ_MM_N_TILE,
            QPROJ_WORKERS,
        ):
            w_col0 = qproj_n_idx * QPROJ_MM_N_TILE
            for t0 in pl.range(0, qproj_full_rows, QPROJ_M_TILE):
                col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                for qr_proj_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                    qr_i8_chunk = qr_i8_matmul[
                        t0 : t0 + QPROJ_M_TILE,
                        qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE,
                    ]
                    wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                    col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk, init_cond=(qr_proj_col0 == 0))
                q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc

            tail_w_col0 = w_col0
            for tail_t0 in pl.range(qproj_full_rows, qproj_t_matmul, QPROJ_TAIL_M_TILE):
                qproj_tail_rows = pl.min(QPROJ_TAIL_M_TILE, tile_rows - tail_t0)
                tail_acc = pl.create_tensor([QPROJ_TAIL_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
                for tail_qr_col0 in pl.pipeline(0, Q_LORA, Q_PROJ_TILE, stage=2):
                    qr_i8_tail = pl.slice(
                        qr_i8_matmul,
                        [QPROJ_TAIL_M_TILE, Q_PROJ_TILE],
                        [tail_t0, tail_qr_col0],
                        valid_shape=[qproj_tail_rows, Q_PROJ_TILE],
                    )
                    wq_tail = wq_b[
                        tail_qr_col0 : tail_qr_col0 + Q_PROJ_TILE,
                        tail_w_col0 : tail_w_col0 + QPROJ_MM_N_TILE,
                    ]
                    tail_acc = pl.matmul_acc(tail_acc, qr_i8_tail, wq_tail, init_cond=(tail_qr_col0 == 0))
                q_proj_i32[
                    tail_t0 : tail_t0 + QPROJ_TAIL_M_TILE,
                    tail_w_col0 : tail_w_col0 + QPROJ_MM_N_TILE,
                ] = tail_acc
    return q_proj_i32, qproj_tid


@pl.jit.inline(auto_scope=False)
def q_proj_q_dequant(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, H * HEAD_DIM], pl.INT32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Dequantize, normalize, and rotate one projected Q tile."""
    t_dim = pl.tensor.dim(q, 0)
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for dq_worker in pl.spmd(
        Q_DEQUANT_WORKERS,
        name_hint="qproj_dequant_rms_nope_rope",
        allow_early_resolve=True,
    ):
        for dq_work in pl.range(
            dq_worker,
            ((tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE) * (H // Q_ROPE_H_TILE),
            Q_DEQUANT_WORKERS,
        ):
            hg = (dq_work % (H // Q_ROPE_H_TILE)) * Q_ROPE_H_TILE
            tg = (dq_work // (H // Q_ROPE_H_TILE)) * Q_ROPE_T_TILE
            out_tg = tile_base + tg
            if tg + Q_ROPE_T_TILE <= tile_rows:
                qr_scale_dq_t = qr_scale_pad_store[tg : tg + Q_ROPE_T_TILE, :]
                q_cos_il = rope_cos_il[out_tg : out_tg + Q_ROPE_T_TILE, :]
                q_sin_signed = rope_sin_signed[out_tg : out_tg + Q_ROPE_T_TILE, :]
                q_swap_idx = rope_swap_idx[out_tg : out_tg + Q_ROPE_T_TILE, :]
                for h_inner in pl.pipeline(Q_ROPE_H_TILE, stage=2):
                    h = hg + h_inner
                    h0 = h * HEAD_DIM
                    q_head_acc = q_proj_i32[tg : tg + Q_ROPE_T_TILE, h0 : h0 + HEAD_DIM]
                    q_head_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM], [1, HEAD_DIM])
                    q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    # Native Q's pertoken kernel combines the two scales
                    # before dequantizing the integer accumulator.
                    q_head_scale_combined = pl.col_expand_mul(
                        pl.row_expand_mul(pl.full([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, value=1.0), qr_scale_dq_t),
                        q_head_scale,
                    )
                    q_head_dq = pl.mul(q_head_acc_fp32, q_head_scale_combined)
                    q_head_dq = pl.cast(pl.cast(q_head_dq, pl.BF16, mode="rint"), pl.FP32)
                    q_head_sq = pl.mul(q_head_dq, q_head_dq)
                    q_head_sq_row = pl.row_sum(q_head_sq)
                    q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                    q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                    q_head_var = pl.add(q_head_sq_mean, EPS)
                    q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
                    q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

                    q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                    q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

                    q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                    q_rope_chunk = pl.cast(pl.cast(q_rope_chunk, pl.BF16, mode="rint"), pl.FP32)
                    q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)
                    q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                    q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                    q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + HEAD_DIM] = q_rope_bf16
            else:
                valid_tail_rows = tile_rows - tg
                # q_proj and its dequant scale are padded to the cube boundary.
                # Keep the math on a static eight-row tile and crop public stores.
                qr_scale_dq_tail = pl.load(
                    qr_scale_pad_store,
                    [tg, 0],
                    [Q_ROPE_T_TILE, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_cos_il_tail = pl.load(
                    rope_cos_il,
                    [out_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[valid_tail_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_sin_signed_tail = pl.load(
                    rope_sin_signed,
                    [out_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[valid_tail_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_col = pl.col_expand_mul(
                    pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                    pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                q_dup_f = pl.cast(pl.cast(pl.mul(q_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
                q_lane = pl.sub(q_col, pl.mul(q_dup_f, 2.0))
                q_swap_f = pl.sub(pl.add(q_col, 1.0), pl.mul(q_lane, 2.0))
                # Row-major flattening offsets stay in fp32: col-expand is defined
                # for half / float only.
                q_row_seed = pl.mul(
                    pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                    ROPE_DIM_SCALE,
                )
                q_row_grid = pl.col_expand_mul(
                    pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0),
                    q_row_seed,
                )
                q_row_offset = pl.transpose(q_row_grid, axis1=0, axis2=1)
                q_swap_idx_tail = pl.cast(pl.add(q_swap_f, q_row_offset), target_type=pl.INT32)
                q_head_reduce_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, HEAD_DIM],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                q_gather_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    dtype=pl.INT32,
                    target_memory=pl.MemorySpace.Vec,
                )
                for h_inner_tail in pl.range(Q_ROPE_H_TILE):
                    h_tail = hg + h_inner_tail
                    h0_tail = h_tail * HEAD_DIM
                    q_head_acc_tail = pl.load(
                        q_proj_i32,
                        [tg, h0_tail],
                        [Q_ROPE_T_TILE, HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale_input_tail = pl.load(
                        wq_b_scale,
                        [h0_tail],
                        [HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale_tail = pl.reshape(q_head_scale_input_tail, [1, HEAD_DIM])
                    q_head_acc_fp32_tail = pl.cast(q_head_acc_tail, target_type=pl.FP32, mode="none")
                    q_head_scale_combined_tail = pl.col_expand_mul(
                        pl.row_expand_mul(
                            pl.tile.full([Q_ROPE_T_TILE, HEAD_DIM], dtype=pl.FP32, value=1.0), qr_scale_dq_tail
                        ),
                        q_head_scale_tail,
                    )
                    q_head_dq_tail = pl.mul(q_head_acc_fp32_tail, q_head_scale_combined_tail)
                    q_head_dq_tail = pl.cast(pl.cast(q_head_dq_tail, pl.BF16, mode="rint"), pl.FP32)

                    q_head_sq_tail = pl.mul(q_head_dq_tail, q_head_dq_tail)
                    q_head_sq_sum_tail = pl.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                    q_head_inv_rms_tail = pl.recip(
                        pl.sqrt(pl.add(pl.mul(q_head_sq_sum_tail, 1.0 / HEAD_DIM), EPS)),
                    )

                    q_nope_normed_tail = pl.row_expand_mul(q_head_dq_tail[:, 0:NOPE_DIM], q_head_inv_rms_tail)
                    q_nope_bf16_tail = pl.cast(q_nope_normed_tail, target_type=pl.BF16, mode="rint")
                    q_nope_valid = pl.set_validshape(q_nope_bf16_tail, valid_tail_rows, NOPE_DIM)
                    pl.store(q_nope_valid, [out_tg, h0_tail], q_flat)

                    q_rope_chunk_raw_tail = q_head_dq_tail[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk_tail = pl.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                    q_rope_chunk_tail = pl.cast(pl.cast(q_rope_chunk_tail, pl.BF16, mode="rint"), pl.FP32)
                    q_rope_swapped_tail = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                    # RoPE inputs contain only the live rows of this tail.
                    q_rope_chunk_tail = pl.set_validshape(q_rope_chunk_tail, valid_tail_rows, ROPE_DIM)
                    q_rope_swapped_tail = pl.set_validshape(q_rope_swapped_tail, valid_tail_rows, ROPE_DIM)
                    q_rope_base_tail = pl.mul(q_rope_chunk_tail, q_cos_il_tail)
                    q_rope_delta_tail = pl.mul(q_rope_swapped_tail, q_sin_signed_tail)
                    q_rope_rot_tail = pl.add(q_rope_base_tail, q_rope_delta_tail)
                    q_rope_bf16_tail = pl.cast(q_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    q_rope_valid = pl.set_validshape(q_rope_bf16_tail, valid_tail_rows, ROPE_DIM)
                    pl.store(q_rope_valid, [out_tg, h0_tail + NOPE_DIM], q_flat)
    return q


@pl.jit.inline(auto_scope=False)
def q_proj_q(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    qproj_dep: pl.Scalar[pl.TASK_ID],
):
    """Q projection and its dequant + RMSNorm + RoPE over bounded tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            qproj_t_matmul = ((tile_rows + QPROJ_TAIL_M_TILE - 1) // QPROJ_TAIL_M_TILE) * QPROJ_TAIL_M_TILE
            q_proj_i32 = pl.create_tensor([qproj_t_matmul, H * HEAD_DIM], dtype=pl.INT32)
            q_proj_i32, _qproj_tid = q_proj_q_matmul(
                wq_b,
                qr_i8_matmul,
                q_proj_i32,
                tile_rows,
                qproj_dep,
            )
            q_proj_q_dequant(
                wq_b_scale,
                rope_cos_il,
                rope_sin_signed,
                rope_swap_idx,
                q,
                qr_scale_pad_store,
                q_proj_i32,
                tile_base,
                tile_rows,
            )
    return q


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
):
    """Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles."""
    qr_i8_matmul = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    # The quant scale rides the qr_i8 -> qproj_matmul -> dequant chain.
    qr_scale_pad_store = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
    q_proj_qr(x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store)
    q_seq_dep = pl.system.task_dummy(deps=[])
    q_proj_q(
        x,
        wq_b,
        wq_b_scale,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        q,
        qr_i8_matmul,
        qr_scale_pad_store,
        q_seq_dep,
    )


@pl.jit.inline(auto_scope=False)
def kv_project_native_240(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    projected: pl.Tensor[[QPROJ_MM_T_DYN, HEAD_DIM], pl.FP32],
    tile_base: pl.Scalar[pl.INDEX],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Match Native's paired K256 traversal and its single K64 accumulator."""
    with pl.spmd(HEAD_DIM // KV_NATIVE_N_TILE, name_hint="kv_proj_native_240", deps=[late_dep]):
        column = pl.tile.get_block_idx() * KV_NATIVE_N_TILE
        group = column // KV_NATIVE_N_GROUP
        for row in pl.range(0, (KV_NATIVE_ROWS // KV_DENSE_M_TILE) * KV_DENSE_M_TILE, KV_DENSE_M_TILE):
            accumulator = pl.create_tensor([KV_DENSE_M_TILE, KV_NATIVE_N_TILE], dtype=pl.FP32)
            for step in pl.pipeline(D // KV_NATIVE_K_TILE, stage=2):
                block = step // (KV_NATIVE_K_BLOCK // KV_NATIVE_K_TILE)
                if group < KV_NATIVE_SHIFT_GROUPS:
                    direction = block
                    if group % 2 == 1:
                        direction = D // KV_NATIVE_K_BLOCK - 1 - block
                    block = (group // 2 * KV_NATIVE_PAIR_SHIFT + direction) % (D // KV_NATIVE_K_BLOCK)
                offset = block * KV_NATIVE_K_BLOCK + step % (KV_NATIVE_K_BLOCK // KV_NATIVE_K_TILE) * KV_NATIVE_K_TILE
                a = x[tile_base + row : tile_base + row + KV_DENSE_M_TILE, offset : offset + KV_NATIVE_K_TILE]
                b = wkv[offset : offset + KV_NATIVE_K_TILE, column : column + KV_NATIVE_N_TILE]
                accumulator = pl.matmul_acc(accumulator, a, b, init_cond=(step == 0))
            projected[row : row + KV_DENSE_M_TILE, column : column + KV_NATIVE_N_TILE] = accumulator
        for row in pl.range((KV_NATIVE_ROWS // KV_DENSE_M_TILE) * KV_DENSE_M_TILE, KV_NATIVE_ROWS, KV_M_TILE):
            accumulator_tail = pl.create_tensor([KV_M_TILE, KV_NATIVE_N_TILE], dtype=pl.FP32)
            for step_tail in pl.pipeline(D // KV_NATIVE_K_TILE, stage=2):
                block_tail = step_tail // (KV_NATIVE_K_BLOCK // KV_NATIVE_K_TILE)
                if group < KV_NATIVE_SHIFT_GROUPS:
                    direction_tail = block_tail
                    if group % 2 == 1:
                        direction_tail = D // KV_NATIVE_K_BLOCK - 1 - block_tail
                    block_tail = (group // 2 * KV_NATIVE_PAIR_SHIFT + direction_tail) % (D // KV_NATIVE_K_BLOCK)
                offset_tail = (
                    block_tail * KV_NATIVE_K_BLOCK
                    + step_tail % (KV_NATIVE_K_BLOCK // KV_NATIVE_K_TILE) * KV_NATIVE_K_TILE
                )
                a_tail = x[tile_base + row : tile_base + row + KV_M_TILE, offset_tail : offset_tail + KV_NATIVE_K_TILE]
                b_tail = wkv[offset_tail : offset_tail + KV_NATIVE_K_TILE, column : column + KV_NATIVE_N_TILE]
                accumulator_tail = pl.matmul_acc(accumulator_tail, a_tail, b_tail, init_cond=(step_tail == 0))
            projected[row : row + KV_M_TILE, column : column + KV_NATIVE_N_TILE] = accumulator_tail
    return projected


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[KV_T_DYN, HEAD_DIM], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """KV LoRA, RMSNorm, and RoPE over bounded dense tiles."""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            x_view = pl.reshape(x, [t_dim, D])
            t_matmul = ((tile_rows + MATMUL_T_TILE - 1) // MATMUL_T_TILE) * MATMUL_T_TILE
            kv_full_rows = (tile_rows // KV_DENSE_M_TILE) * KV_DENSE_M_TILE
            kv_m_groups = pl.min(KV_OM, pl.max(1, tile_rows // (2 * KV_DENSE_M_TILE)))

            # Split-K kv_proj: KV_N_TILE N-groups expanded KV_OK-fold into cube blocks that
            # atomic-add their K partials into a zero-seeded output.
            kv_fp32 = pl.create_tensor([t_matmul, HEAD_DIM], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed"):
                for kts0 in pl.range(0, t_matmul, KV_M_TILE):
                    for kvseed0 in pl.range(0, HEAD_DIM, KV_N_TILE):
                        kv_seed = pl.full([KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0)
                        kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = kv_seed

            # Native changes the K traversal at the B40/S6 shape.
            if tile_rows == KV_NATIVE_ROWS:
                kv_fp32 = kv_project_native_240(x, wkv, kv_fp32, tile_base, late_dep)
            else:
                # KV projection consumes the caller's readiness dependency.
                with pl.spmd(
                    (HEAD_DIM // KV_N_TILE) * KV_OK * kv_m_groups,
                    name_hint="kv_proj_matmul",
                    deps=[late_dep],
                ) as _kv_tid:
                    kbg = pl.tile.get_block_idx()
                    kv_col0 = (kbg // (KV_OK * kv_m_groups)) * KV_N_TILE
                    kv_k_base = ((kbg // kv_m_groups) % KV_OK) * KV_SPLIT_K_TILE
                    kv_m_group = kbg % kv_m_groups
                    for dense_t0 in pl.range(kv_m_group * KV_DENSE_M_TILE, kv_full_rows, kv_m_groups * KV_DENSE_M_TILE):
                        dense_x0 = tile_base + dense_t0
                        dense_acc = pl.create_tensor([KV_DENSE_M_TILE, KV_N_TILE], dtype=pl.FP32)
                        for dense_k in pl.pipeline(0, KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                            dense_d0 = kv_k_base + dense_k * KV_K_TILE
                            dense_x = x_view[dense_x0 : dense_x0 + KV_DENSE_M_TILE, dense_d0 : dense_d0 + KV_K_TILE]
                            dense_w = wkv[dense_d0 : dense_d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                            dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, init_cond=(dense_k == 0))
                        kv_fp32 = pl.assemble(kv_fp32, dense_acc, [dense_t0, kv_col0], atomic=pl.AtomicType.Add)
                    for t0 in pl.range(kv_full_rows + kv_m_group * KV_M_TILE, t_matmul, kv_m_groups * KV_M_TILE):
                        kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
                        for db in pl.pipeline(KV_SPLIT_K_TILE // KV_K_TILE, stage=2):
                            d0 = kv_k_base + db * KV_K_TILE
                            kv_rows = pl.min(KV_M_TILE, tile_rows - t0)
                            x_t0 = tile_base + t0
                            kv_x_chunk_bf16 = pl.slice(
                                x_view,
                                [KV_M_TILE, KV_K_TILE],
                                [x_t0, d0],
                                valid_shape=[kv_rows, KV_K_TILE],
                            )
                            wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                            kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, init_cond=(db == 0))
                        kv_fp32 = pl.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=pl.AtomicType.Add)

            kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])

            # Fused KV RMSNorm + interleaved (CANN A3) RoPE, one spmd task per
            # [KV_RMS_T_TILE, HEAD_DIM] row block. NOPE columns [0:NOPE_DIM) and rope columns
            # [NOPE_DIM:HEAD_DIM) are disjoint, so each task writes a conflict-free row block.
            kv_token_tiles = (tile_rows + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
            for tg_idx in pl.spmd(
                kv_token_tiles,
                name_hint="kv_rms_norm_rope",
                sync_start=True,
            ):
                tg = tg_idx * KV_RMS_T_TILE
                valid_rows = pl.min(KV_RMS_T_TILE, tile_rows - tg)
                out_tg = tile_base + tg
                if valid_rows == KV_RMS_T_TILE:
                    # Native RMSNorm ReduceSumMultiN accumulates the eight
                    # 64-column vectors lane-wise, then WholeReduceSum.
                    # Reducing each vector first changes FP32 rounding.
                    kv_sq_lanes = pl.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
                        # Native WKV publishes BF16 before KV RMSNorm.
                        kv_chunk = pl.cast(pl.cast(kv_chunk, pl.BF16, mode="rint"), pl.FP32)
                        kv_sq = pl.mul(kv_chunk, kv_chunk)
                        kv_sq_lanes = pl.add(kv_sq_lanes, kv_sq)
                    kv_sq_sum = pl.reshape(pl.row_sum(kv_sq_lanes), [1, KV_RMS_T_TILE])
                    kv_rms = pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS))
                    kv_inv_rms = pl.div(pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms)
                    kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

                    for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
                        kv_chunk = pl.cast(pl.cast(kv_chunk, pl.BF16, mode="rint"), pl.FP32)
                        gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
                        gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
                        kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                        kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
                        kv_view[out_tg : out_tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16

                    gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
                    gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
                    kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
                    kv_rope_chunk = pl.cast(pl.cast(kv_rope_chunk, pl.BF16, mode="rint"), pl.FP32)
                    kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
                    # Native KV RMSNorm also publishes BF16 before RoPE.
                    kv_rope_norm_chunk = pl.cast(pl.cast(kv_rope_norm_chunk, pl.BF16, mode="rint"), pl.FP32)
                    kv_cos_il_full = rope_cos_il[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_sin_signed_full = rope_sin_signed[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swap_idx_full = rope_swap_idx[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swapped_full = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx_full)
                    kv_rope_rot_full = pl.add(
                        pl.mul(kv_rope_norm_chunk, kv_cos_il_full),
                        pl.mul(kv_swapped_full, kv_sin_signed_full),
                    )
                    kv_rope_i16_full = pl.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
                    kv_view[out_tg : out_tg + KV_RMS_T_TILE, NOPE_DIM:HEAD_DIM] = kv_rope_i16_full
                else:
                    kv_reduce_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
                    kv_sq_lanes_tail = pl.tile.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0_tail in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, kv_sq_col0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_chunk_tail = pl.cast(pl.cast(kv_chunk_tail, pl.BF16, mode="rint"), pl.FP32)
                        kv_sq_tail = pl.mul(kv_chunk_tail, kv_chunk_tail)
                        kv_sq_lanes_tail = pl.add(kv_sq_lanes_tail, kv_sq_tail)
                    kv_sq_sum_tail = pl.reshape(pl.row_sum(kv_sq_lanes_tail, kv_reduce_tmp), [1, KV_RMS_T_TILE])
                    kv_rms_tail = pl.sqrt(pl.add(pl.mul(kv_sq_sum_tail, 1.0 / HEAD_DIM), EPS))
                    kv_inv_rms_tail = pl.div(pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms_tail)
                    kv_inv_rms_t_tail = pl.reshape(kv_inv_rms_tail, [KV_RMS_T_TILE, 1])

                    for n0_tail in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                        kv_chunk_tail = pl.load(
                            kv_fp32,
                            [tg, n0_tail],
                            [KV_RMS_T_TILE, KV_TILE],
                            valid_shape=[valid_rows, KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        kv_chunk_tail = pl.cast(pl.cast(kv_chunk_tail, pl.BF16, mode="rint"), pl.FP32)
                        gamma_kv_input_tail = pl.load(
                            gamma_ckv,
                            [n0_tail],
                            [KV_TILE],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        gamma_kv_cast_tail = pl.cast(gamma_kv_input_tail, target_type=pl.FP32)
                        gamma_kv_chunk_tail = pl.reshape(gamma_kv_cast_tail, [1, KV_TILE])
                        kv_normed_tail = pl.col_expand_mul(
                            pl.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail),
                            gamma_kv_chunk_tail,
                        )
                        kv_normed_bf16_tail = pl.cast(kv_normed_tail, target_type=pl.BF16, mode="rint")
                        kv_normed_valid = pl.set_validshape(kv_normed_bf16_tail, valid_rows, KV_TILE)
                        pl.store(kv_normed_valid, [out_tg, n0_tail], kv_view)

                    gamma_rope_input_tail = pl.load(
                        gamma_ckv,
                        [NOPE_DIM],
                        [ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    gamma_rope_cast_tail = pl.cast(gamma_rope_input_tail, target_type=pl.FP32)
                    gamma_rope_tail = pl.reshape(gamma_rope_cast_tail, [1, ROPE_DIM])
                    kv_rope_chunk_tail = pl.load(
                        kv_fp32,
                        [tg, NOPE_DIM],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_rope_chunk_tail = pl.cast(pl.cast(kv_rope_chunk_tail, pl.BF16, mode="rint"), pl.FP32)
                    kv_rope_norm_tail = pl.col_expand_mul(
                        pl.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail),
                        gamma_rope_tail,
                    )
                    kv_rope_norm_tail = pl.cast(pl.cast(kv_rope_norm_tail, pl.BF16, mode="rint"), pl.FP32)
                    kv_cos_il_tail = pl.load(
                        rope_cos_il,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_sin_signed_tail = pl.load(
                        rope_sin_signed,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_col = pl.col_expand_mul(
                        pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                        pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                    )
                    kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
                    kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
                    kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))
                    # Row-major flattening offsets stay in fp32: col-expand is defined
                    # for half / float only.
                    kv_row_seed = pl.mul(
                        pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                        ROPE_DIM_SCALE,
                    )
                    kv_row_grid = pl.col_expand_mul(
                        pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0),
                        kv_row_seed,
                    )
                    kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
                    kv_swap_idx_tail = pl.cast(pl.add(kv_swap_f, kv_row_offset), target_type=pl.INT32)
                    kv_gather_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, ROPE_DIM],
                        dtype=pl.INT32,
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_swapped_tail = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
                    kv_rope_rot_tail = pl.add(
                        pl.mul(kv_rope_norm_tail, kv_cos_il_tail),
                        pl.mul(kv_swapped_tail, kv_sin_signed_tail),
                    )
                    kv_rope_i16_tail = pl.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    kv_rope_valid = pl.set_validshape(kv_rope_i16_tail, valid_rows, ROPE_DIM)
                    pl.store(kv_rope_valid, [out_tg, NOPE_DIM], kv_view)


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Fused q + kv projection: both branches share one token axis and one rope table."""
    t_dim = pl.tensor.dim(x, 0)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, q_rope_sin_signed, q_rope_swap_idx)
    q_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        rope_cos,
        q_rope_sin_signed,
        q_rope_swap_idx,
        q,
        qr,
        qr_scale,
    )
    kv_proj_rope(
        x,
        wkv,
        gamma_ckv,
        rope_cos,
        q_rope_sin_signed,
        q_rope_swap_idx,
        kv,
        late_dep,
    )
    return q
