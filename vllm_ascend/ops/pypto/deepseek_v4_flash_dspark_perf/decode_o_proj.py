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
from .nz_mode import (
    BF16_WEIGHT_LAYOUT,
    BF16_WEIGHT_NZ,
    QUANT_WEIGHT_LAYOUT,
    QUANT_WEIGHT_NZ,
    WO_A_WEIGHT_LAYOUT,
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
PROJ_A_LARGE_N_TILE = 256

PROJ_A_ROW_TILE = 128  # Parallel row tile; the last block may contain fewer valid rows.

# Match upstream's adaptive row tiles and N256; INT8 accumulation is exact.
B_K_TILE = 256
PROJ_B_SMALL_T_TILE = 32
PROJ_B_MEDIUM_T_TILE = 96
PROJ_B_MM_T_TILE = 128
PROJ_B_MM_N_TILE = 256

PROJ_B_ACT_N_TILE = 512

QUANT_TOKEN_TILE = 8
QUANT_TASK_T_TILE = 32  # 每个量化任务负责的 token 跨度，沿用上游 e68e091 的调优值

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


# NZ 与 ND 都按行块×列块分配；NZ 单独显式约束列偏移非负，下面按开关绑定。
# 不要写成同一个函数里的 `if BF16_WEIGHT_NZ`——@pl.jit 读源文件做 AST 分析，两个分支
# 都会被 trace，SSA 会冲突（实测报 Error Code: 6）。
@pl.jit.inline
def _proj_a_mm_nz(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    o_r_pad: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.FP32],
    g: pl.Scalar[pl.INDEX],
    row_base_o: pl.Scalar[pl.INDEX],
    out_col_g: pl.Scalar[pl.INDEX],
    t_dim: pl.Scalar[pl.INDEX],
    proj_a_rows: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    A_COL_TILE: pl.constexpr,
):
    """Parallelize row and column tiles as upstream; retain Native NZ weights."""
    with pl.spmd(
        proj_a_rows * (O_LORA // A_COL_TILE),
        name_hint="proj_a_mm",
        deps=[heads_dep],
        allow_early_resolve=True,
    ) as pa_tid:
        pl.set_cache_policy(wo_a, pl.CachePolicy.BYPASS)
        pa_unit = pl.tile.get_block_idx()
        pa_rb = pa_unit // (O_LORA // A_COL_TILE)  # row block outermost
        nf = pa_unit % (O_LORA // A_COL_TILE)
        pa_r0 = pa_rb * PROJ_A_ROW_TILE
        pa_rows = pl.min(PROJ_A_ROW_TILE, t_dim - pa_r0)
        pa_src0 = row_base_o + pa_r0
        # The block index remainder is nonnegative; make the NZ bound explicit.
        n0 = pl.max(nf, 0) * A_COL_TILE
        xa_first = pl.slice(
            o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
        )
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
            )
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
    return o_r_pad, pa_tid


@pl.jit.inline
def _proj_a_mm_nd(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    o_r_pad: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.FP32],
    g: pl.Scalar[pl.INDEX],
    row_base_o: pl.Scalar[pl.INDEX],
    out_col_g: pl.Scalar[pl.INDEX],
    t_dim: pl.Scalar[pl.INDEX],
    proj_a_rows: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    A_COL_TILE: pl.constexpr,
):
    """ND 版：与上游 _decode_o_proj 同形，(行块 x N 块) 二维展开、行块最外。"""
    with pl.spmd(
        proj_a_rows * (O_LORA // A_COL_TILE),
        name_hint="proj_a_mm",
        deps=[heads_dep],
        allow_early_resolve=True,
    ) as pa_tid:
        pl.set_cache_policy(wo_a, pl.CachePolicy.BYPASS)
        pa_unit = pl.tile.get_block_idx()
        pa_rb = pa_unit // (O_LORA // A_COL_TILE)  # row block outermost
        nf = pa_unit % (O_LORA // A_COL_TILE)
        pa_r0 = pa_rb * PROJ_A_ROW_TILE
        pa_rows = pl.min(PROJ_A_ROW_TILE, t_dim - pa_r0)
        pa_src0 = row_base_o + pa_r0
        n0 = nf * A_COL_TILE
        xa_first = pl.slice(
            o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
        )
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
            )
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        # acc_a is 3D (wo_a keeps its group axis), which subscript-write cannot express.
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
    return o_r_pad, pa_tid


proj_a_mm = _proj_a_mm_nz if BF16_WEIGHT_NZ else _proj_a_mm_nd


@pl.jit.inline
def _proj_b_mm_nd(
    o_r_i8_pad: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8],
    partials: pl.Tensor[[T_PAD, O_GROUPS * D], pl.INT32],
    g: pl.Scalar[pl.INDEX],
    col_g: pl.Scalar[pl.INDEX],
    proj_b_t_rows: pl.Scalar[pl.INDEX],
    q_tid: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
):
    """ND 版：(token 块 x D 块) 二维展开，与上游同形。"""
    with pl.spmd(
        proj_b_t_rows * (D // PROJ_B_D_TILE), name_hint="proj_b_mm", deps=[q_tid], allow_early_resolve=True
    ) as pb_tid:
        pl.set_cache_policy(wo_b, pl.CachePolicy.BYPASS)
        pb_unit = pl.tile.get_block_idx()
        tb = pb_unit // (D // PROJ_B_D_TILE)
        dc = pb_unit - tb * (D // PROJ_B_D_TILE)
        t0 = tb * ROW_TILE
        d0 = dc * PROJ_B_D_TILE
        for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
            n0 = d0 + nf * PROJ_B_MM_N_TILE
            acc_b = pl.create_tensor([ROW_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                k0 = col_g + kb * B_K_TILE
                b_act = o_r_i8_pad[t0 : t0 + ROW_TILE, k0 : k0 + B_K_TILE]
                b_weight = wo_b[
                    k0 : k0 + B_K_TILE,
                    n0 : n0 + PROJ_B_MM_N_TILE,
                ]
                acc_b = pl.matmul_acc(acc_b, b_act, b_weight, init_cond=(kb == 0))
            partials[t0 : t0 + ROW_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
    return partials, pb_tid


@pl.jit.incore
def _proj_b_mm_nz_kernel(
    o_r_i8_pad: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, pl.NZ],
    partials: pl.Tensor[[T_PAD, O_GROUPS * D], pl.INT32],
    g: pl.Scalar[pl.INDEX],
    col_g: pl.Scalar[pl.INDEX],
    proj_b_t_rows: pl.Scalar[pl.INDEX],
    ROW_TILE: pl.constexpr,
):
    # Keep Native [G*K,N] NZ storage; all group offsets remain inside this kernel.
    d0 = pl.tile.get_block_idx() * PROJ_B_D_TILE
    for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
        n0 = d0 + nf * PROJ_B_MM_N_TILE
        if ROW_TILE > PROJ_B_MEDIUM_T_TILE:
            weight_k0 = pl.max(g, 0) * O_LORA
            weight_resident = wo_b[weight_k0 : weight_k0 + O_LORA, n0 : n0 + PROJ_B_MM_N_TILE]
            for tb in pl.range(proj_b_t_rows):
                t0 = tb * ROW_TILE
                b_act_full = o_r_i8_pad[t0 : t0 + ROW_TILE, col_g : col_g + O_LORA]
                full_acc = pl.matmul(b_act_full, weight_resident, out_dtype=pl.INT32)
                partials[t0 : t0 + ROW_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = full_acc
        else:
            for tb in pl.range(proj_b_t_rows):
                t0 = tb * ROW_TILE
                acc_b = pl.create_tensor([ROW_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
                for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                    k0 = col_g + kb * B_K_TILE
                    b_act = o_r_i8_pad[t0 : t0 + ROW_TILE, k0 : k0 + B_K_TILE]
                    wk0 = pl.max(g, 0) * O_LORA + kb * B_K_TILE
                    b_weight = wo_b[wk0 : wk0 + B_K_TILE, n0 : n0 + PROJ_B_MM_N_TILE]
                    acc_b = pl.matmul_acc(acc_b, b_act, b_weight, init_cond=(kb == 0))
                partials[t0 : t0 + ROW_TILE, g * D + n0 : g * D + n0 + PROJ_B_MM_N_TILE] = acc_b
    return partials


@pl.jit.inline
def _proj_b_mm_nz(
    o_r_i8_pad: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, pl.NZ],
    partials: pl.Tensor[[T_PAD, O_GROUPS * D], pl.INT32],
    g: pl.Scalar[pl.INDEX],
    col_g: pl.Scalar[pl.INDEX],
    proj_b_t_rows: pl.Scalar[pl.INDEX],
    q_tid: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
):
    """复用 Native 二维 NZ 权重，组号仅改变核内 K 偏移。"""
    with pl.spmd(
        D // PROJ_B_D_TILE, name_hint="proj_b_mm", deps=[q_tid], allow_early_resolve=True
    ) as pb_tid:
        partials = _proj_b_mm_nz_kernel(o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, ROW_TILE)
    return partials, pb_tid


proj_b_mm = _proj_b_mm_nz if QUANT_WEIGHT_NZ else _proj_b_mm_nd


@pl.jit.inline
def _decode_o_proj_tp1_tiled(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    """Project local-token, full-group attention heads into BF16 hidden rows."""
    t_dim = pl.tensor.dim(attn_out, 0)
    act_t_blks = (t_dim + PROJ_B_ACT_TASK_T_TILE - 1) // PROJ_B_ACT_TASK_T_TILE
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    proj_b_t_rows = (t_dim + ROW_TILE - 1) // ROW_TILE
    proj_b_padded_rows = proj_b_t_rows * ROW_TILE

    # Upstream performance arithmetic: per-group quantization and INT32 partials.
    # The precision entry keeps Native's whole-token quantization separately.
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    # Per-group INT32 partials: proj_b_mm writes group g's contribution to output
    # channel n at partials[:, g*D + n]. No atomic-add -> no zero-seed.
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA

            o_r_pad, pa_tid = proj_a_mm(
                o_packed, wo_a, o_r_pad, g, row_base_o, out_col_g,
                t_dim, proj_a_rows, heads_dep, A_COL_TILE,
            )

            col_g = g * O_LORA
            # 性能版按上游把 amax 与量化融进同一个 SPMD，并让每个 group 用自己的标度。
            # 精度版另起 oproj_token_scale 任务、跨全部 O_GROUPS 取同一个 amax，并多做
            # 两次 BF16 往返，那是为复刻 Native A3 dynamic_quant 的语义；本版本不要求
            # 与 Native 逐 bit 一致，于是省掉那遍全量扫描和往返。
            with pl.spmd(
                (t_dim + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE,
                name_hint="quant",
                deps=[pa_tid],
                allow_early_resolve=True,
            ) as q_tid:
                quant_start = pl.tile.get_block_idx() * QUANT_TASK_T_TILE
                for qt in pl.pipeline(
                    quant_start, pl.min(quant_start + QUANT_TASK_T_TILE, t_dim), QUANT_TOKEN_TILE, stage=2
                ):
                    oc_amax = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    g_row_max = pl.reshape(pl.row_max(pl.abs(oc_amax)), [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = pl.mul(g_amax, 1.0 / INT8_SCALE_MAX)
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                # Zero the tail of the final active proj_b_mm row tile.
                if quant_start + QUANT_TASK_T_TILE >= t_dim:
                    for zt in pl.range(t_dim, proj_b_padded_rows, QUANT_TOKEN_TILE):
                        zero_half = pl.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                        zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
                        zero_rows = pl.min(QUANT_TOKEN_TILE, proj_b_padded_rows - zt)
                        o_r_i8_pad = pl.assemble(o_r_i8_pad, pl.set_validshape(zero_i8, zero_rows, O_LORA), [zt, col_g])

            partials, pb_tid = proj_b_mm(
                o_r_i8_pad, wo_b, partials, g, col_g, proj_b_t_rows, q_tid, ROW_TILE,
            )
            proj_b_tids[g] = pb_tid

    # Dequantize each group with its own scale, then sum in FP32.
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
            # 每个 group 先按自己的 token 标度反量化到 FP32 再相加（上游写法）。
            acc = pl.full([PROJ_B_ACT_T_TILE, PROJ_B_ACT_N_TILE], dtype=pl.FP32, value=0.0)
            for act_g in pl.pipeline(O_GROUPS, stage=2):
                p_col0 = act_g * D + ob_n0
                p_g = partials[b_tb : b_tb + PROJ_B_ACT_T_TILE, p_col0 : p_col0 + PROJ_B_ACT_N_TILE]
                g_scale_row = act_scale_dq[act_g : act_g + 1, b_tb : b_tb + PROJ_B_ACT_T_TILE]
                g_scale = pl.reshape(g_scale_row, [PROJ_B_ACT_T_TILE, 1])
                p_g_f32 = pl.cast(p_g, target_type=pl.FP32, mode="none")
                acc = pl.add(acc, pl.row_expand_mul(p_g_f32, g_scale))
            out_t = pl.col_expand_mul(acc, wb_scale_chunk)
            out_bf16 = pl.cast(out_t, target_type=pl.BF16, mode="rint")
            output_rows = pl.min(PROJ_B_ACT_T_TILE, t_dim - b_tb)
            attn_out = pl.assemble(attn_out, pl.set_validshape(out_bf16, output_rows, PROJ_B_ACT_N_TILE), [b_tb, ob_n0])

    return attn_out


@pl.jit.inline
def decode_o_proj_tp1(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    """Use upstream row/column tiles while retaining Native weight storage."""
    t_dim = pl.tensor.dim(attn_out, 0)
    if t_dim <= PROJ_B_SMALL_T_TILE:
        attn_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, attn_out, heads_dep, PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE,
        )
    elif t_dim <= PROJ_B_MEDIUM_T_TILE:
        attn_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, attn_out, heads_dep, PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE,
        )
    else:
        attn_out = _decode_o_proj_tp1_tiled(
            o_packed, wo_a, wo_b, wo_b_scale, attn_out, heads_dep, PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE,
        )
    return attn_out
