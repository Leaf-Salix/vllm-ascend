# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CSA integration dependency adapted from pypto-lib 205255b4/decode_o_proj.py."""

import pypto.language as pl

from .config import (
    DECODE_TOKENS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from .config import (
    FLASH as M,
)
from .config import (
    TP as TP_SIZE,
)

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

LOCAL_T = DECODE_TOKENS // TP_SIZE

LOCAL_O_GROUPS = O_GROUPS // TP_SIZE

LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA

T_DYN = pl.dynamic("T_DYN")

TOKEN_TILE = 16

COMM_ROW_TILE = 8

ATTENTION_PUBLISH_WORKERS = 48

O_RS_REDUCE_WORKERS = 48  # 128 row blocks; 8 left 40 of 48 AIV idle

O_RS_PUBLISH_WORKERS = 32

O_RS_DEQUANT_WORKERS = 12  # per owner; 4 owners -> one AIV wave

O_RS_PUT_T_TILE = 8

O_RS_D_TILE = 4096

LOCAL_T_PAD = (LOCAL_T + TOKEN_TILE - 1) // TOKEN_TILE * TOKEN_TILE

T_PAD = LOCAL_T_PAD

GROUP_T_PAD = TP_SIZE * LOCAL_T_PAD

ATTENTION_WINDOW_ROWS = LOCAL_O_GROUPS * GROUP_T_PAD

O_WINDOW_ROWS = TP_SIZE * LOCAL_T_PAD

A_K_TILE = 256

PROJ_A_MM_N_TILE = 128

PROJ_A_ROW_TILE = 128  # proj_a token block; one block covers T_PAD, 8 tasks/group

B_K_TILE = 256

PROJ_B_MM_T_TILE = 128

PROJ_B_MM_N_TILE = 256

PROJ_B_ACT_N_TILE = 512

QUANT_TOKEN_TILE = 8

PROJ_B_D_TILE = 512  # proj_b_mm D chunk per task; coarser starves the 24 AIC cores

PROJ_B_ACT_T_TILE = 8

PROJ_B_ACT_TASK_T_TILE = 32  # proj_b_act token block per task

O_A_T_TILE = 128

O_A_K_TILE = 256

O_A_N_TILE = 128

QUANT_T_TILE = 8

O_A_QUANT_WORKERS = 6  # per owner-group; 4 x 2 x 6 -> one AIV wave

O_B_T_TILE = 128

O_B_K_TILE = 256

O_B_N_TILE = 256

O_B_D_TILE = 512

ACT_T_TILE = 16

ACT_N_TILE = 512

FIXTURE_LOCAL_T = max(1, LOCAL_T - 1)

FIXTURE_OUTPUT_SENTINEL = -7.0

if DECODE_TOKENS % TP_SIZE != 0:
    raise ValueError(f"decode tokens {DECODE_TOKENS} must be divisible by TP size {TP_SIZE}")

if O_GROUPS % TP_SIZE != 0:
    raise ValueError(f"output groups {O_GROUPS} must be divisible by TP size {TP_SIZE}")

if O_GROUP_IN % O_A_K_TILE != 0:
    raise ValueError(f"O-A input {O_GROUP_IN} must be divisible by K tile {O_A_K_TILE}")

if O_LORA % O_A_N_TILE != 0:
    raise ValueError(f"O-A output {O_LORA} must be divisible by N tile {O_A_N_TILE}")

if O_LORA % O_B_K_TILE != 0:
    raise ValueError(f"O-B group width {O_LORA} must be divisible by K tile {O_B_K_TILE}")

if D % O_B_D_TILE != 0 or O_B_D_TILE % O_B_N_TILE != 0:
    raise ValueError("O-B output tiles must divide the hidden dimension")

if D % ACT_N_TILE != 0:
    raise ValueError(f"O-B activation tile {ACT_N_TILE} must divide hidden size {D}")

if D % O_RS_D_TILE != 0:
    raise ValueError(f"O-B ReduceScatter tile {O_RS_D_TILE} must divide hidden size {D}")

if O_RS_REDUCE_WORKERS % TP_SIZE != 0:
    raise ValueError(f"TP size {TP_SIZE} must divide O-B ReduceScatter workers {O_RS_REDUCE_WORKERS}")

if O_RS_PUBLISH_WORKERS % TP_SIZE != 0:
    raise ValueError(f"TP size {TP_SIZE} must divide O-B publish workers {O_RS_PUBLISH_WORKERS}")

if GROUP_T_PAD % O_B_T_TILE != 0:
    raise ValueError(f"O-B token tile {O_B_T_TILE} must divide token capacity {GROUP_T_PAD}")

if T_PAD % PROJ_B_MM_T_TILE != 0:
    raise ValueError(f"proj_b_mm token tile {PROJ_B_MM_T_TILE} must divide token capacity {T_PAD}")


@pl.jit.inline
def decode_o_proj_tp1(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    """Project local-token, full-group attention heads into BF16 hidden rows."""
    t_dim = pl.tensor.dim(attn_out, 0)
    act_t_blks = (t_dim + PROJ_B_ACT_TASK_T_TILE - 1) // PROJ_B_ACT_TASK_T_TILE
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    proj_b_t_rows = (t_dim + PROJ_B_MM_T_TILE - 1) // PROJ_B_MM_T_TILE
    proj_b_padded_rows = proj_b_t_rows * PROJ_B_MM_T_TILE

    # Native WO-A materializes BF16 across all eight groups, then WO-B
    # quantizes the concatenated 8192 columns with one scale per token.
    # Keep the reference's grouped matmul tiles and INT32 group partials.
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([1, T_PAD], dtype=pl.FP32)
    act_scale_q = pl.create_tensor([1, T_PAD], dtype=pl.FP32)
    # Per-group INT32 partials: proj_b_mm writes group g's contribution to output
    # channel n at partials[:, g*D + n]. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_a_tids = pl.array.create(O_GROUPS, pl.TASK_ID)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA

            with pl.spmd(
                proj_a_rows * (O_LORA // PROJ_A_MM_N_TILE),
                name_hint="proj_a_mm",
                deps=[heads_dep],
                allow_early_resolve=True,
            ) as pa_tid:
                pa_unit = pl.tile.get_block_idx()
                pa_rb = pa_unit // (O_LORA // PROJ_A_MM_N_TILE)  # row block outermost
                nf = pa_unit - pa_rb * (O_LORA // PROJ_A_MM_N_TILE)
                pa_r0 = pa_rb * PROJ_A_ROW_TILE
                pa_rows = pl.min(PROJ_A_ROW_TILE, t_dim - pa_r0)
                pa_src0 = row_base_o + pa_r0
                n0 = nf * PROJ_A_MM_N_TILE
                xa_first = pl.slice(
                    o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
                )
                wa_first = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, 0:A_K_TILE]
                acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32, b_trans=True)
                for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
                    k0 = kb * A_K_TILE
                    xa_k_chunk = pl.slice(
                        o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
                    )
                    wa_k_chunk = wo_a[g : g + 1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)
                # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
                o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])

            proj_a_tids[g] = pa_tid

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="oproj_token_scale",
        deps=[proj_a_tids[i] for i in range(O_GROUPS)],
        allow_early_resolve=True,
    ) as scale_tid:
        for qt in pl.pipeline(0, t_dim, QUANT_TOKEN_TILE, stage=2):
            token_amax = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for scale_group in pl.range(O_GROUPS):
                scale_col = scale_group * O_LORA
                projected = o_r_pad[qt : qt + QUANT_TOKEN_TILE, scale_col : scale_col + O_LORA]
                projected = pl.cast(pl.cast(projected, pl.BF16, mode="rint"), pl.FP32)
                group_amax = pl.reshape(pl.row_max(pl.abs(projected)), [1, QUANT_TOKEN_TILE])
                token_amax = pl.maximum(token_amax, group_amax)
            act_scale_dq[0:1, qt : qt + QUANT_TOKEN_TILE] = pl.mul(token_amax, 1.0 / INT8_SCALE_MAX)
            # Native A3 dynamic_quant computes these independently: 127/amax
            # for quantization, amax*(1/127) for dequantization. Reciprocating
            # the rounded dequant scale changes INT8 half-way boundaries.
            quant_numerator = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
            act_scale_q[0:1, qt : qt + QUANT_TOKEN_TILE] = pl.div(quant_numerator, token_amax)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            col_g = g * O_LORA
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="quant", deps=[scale_tid], allow_early_resolve=True
            ) as q_tid:
                for qt in pl.pipeline(0, t_dim, QUANT_TOKEN_TILE, stage=2):
                    token_multiplier = act_scale_q[0:1, qt : qt + QUANT_TOKEN_TILE]
                    g_sq_col = pl.reshape(token_multiplier, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oc_q = pl.cast(pl.cast(oc_q, pl.BF16, mode="rint"), pl.FP32)
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                # Zero the tail of the final active proj_b_mm row tile.
                for zt in pl.range(t_dim, proj_b_padded_rows, QUANT_TOKEN_TILE):
                    zero_half = pl.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                    zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
                    zero_rows = pl.min(QUANT_TOKEN_TILE, proj_b_padded_rows - zt)
                    o_r_i8_pad = pl.assemble(o_r_i8_pad, pl.set_validshape(zero_i8, zero_rows, O_LORA), [zt, col_g])

            with pl.spmd(
                proj_b_t_rows * (D // PROJ_B_D_TILE), name_hint="proj_b_mm", deps=[q_tid], allow_early_resolve=True
            ) as pb_tid:
                pb_unit = pl.tile.get_block_idx()
                tb = pb_unit // (D // PROJ_B_D_TILE)
                dc = pb_unit - tb * (D // PROJ_B_D_TILE)
                t0 = tb * PROJ_B_MM_T_TILE
                d0 = dc * PROJ_B_D_TILE
                for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
                    n0 = d0 + nf * PROJ_B_MM_N_TILE
                    acc_b = pl.create_tensor([PROJ_B_MM_T_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                        k0 = col_g + kb * B_K_TILE
                        b_act = o_r_i8_pad[t0 : t0 + PROJ_B_MM_T_TILE, k0 : k0 + B_K_TILE]
                        b_weight = wo_b[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE]
                        acc_b = pl.matmul_acc(acc_b, b_act, b_weight, b_trans=True, init_cond=(kb == 0))
                    partials[t0 : t0 + PROJ_B_MM_T_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
            proj_b_tids[g] = pb_tid

    # Sum INT32 group partials before dequantizing by the common token scale,
    # matching Native's single quantized WO-B matmul.
    with pl.spmd(
        act_t_blks * (D // PROJ_B_ACT_N_TILE),
        name_hint="proj_b_act",
        deps=[proj_b_tids[i] for i in range(O_GROUPS)],
        allow_early_resolve=True,
    ) as _act_tid:
        act_idx = pl.tile.get_block_idx()
        tblk = act_idx // (D // PROJ_B_ACT_N_TILE)  # token block outermost
        nreg = act_idx - tblk * (D // PROJ_B_ACT_N_TILE)
        ob_n0 = nreg * PROJ_B_ACT_N_TILE
        t0 = tblk * PROJ_B_ACT_TASK_T_TILE
        wb_scale = wo_b_scale[ob_n0 : ob_n0 + PROJ_B_ACT_N_TILE]
        wb_scale_chunk = pl.reshape(wb_scale, [1, PROJ_B_ACT_N_TILE])
        for b_tb in pl.range(t0, pl.min(t0 + PROJ_B_ACT_TASK_T_TILE, t_dim), PROJ_B_ACT_T_TILE):
            acc_i32 = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.INT32, value=0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                acc_i32 = pl.add(acc_i32, p_g)
            output_token_scale = pl.reshape(act_scale_dq[0:1, b_tb : b_tb + PROJ_B_ACT_T_TILE], [PROJ_B_ACT_T_TILE, 1])
            # Native WO-B dequantizes the INT32 result by channel first,
            # then applies the token scale before its BF16 cast.
            acc = pl.col_expand_mul(pl.cast(acc_i32, pl.FP32), wb_scale_chunk)
            out_t = pl.row_expand_mul(acc, output_token_scale)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            output_rows = pl.min(PROJ_B_ACT_T_TILE, t_dim - b_tb)
            attn_out = pl.assemble(attn_out, pl.set_validshape(out_bf16, output_rows, PROJ_B_ACT_N_TILE), [b_tb, ob_n0])

    return attn_out
