# SPDX-License-Identifier: Apache-2.0
"""复用公共 O 投影主体，将反量化与 mHC post 放在同一个 Vector 任务内。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.decode_o_proj import (
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    PROJ_A_LARGE_N_TILE,
    PROJ_A_MM_N_TILE,
    PROJ_A_ROW_TILE,
    PROJ_B_MEDIUM_T_TILE,
    PROJ_B_MM_T_TILE,
    PROJ_B_SMALL_T_TILE,
    QUANT_TASK_T_TILE,
    QUANT_TOKEN_TILE,
    T_PAD,
    D,
    proj_a_mm,
    proj_b_mm,
)
from ..deepseek_v4_flash_csa.nz_mode import QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT

T_DYN = pl.dynamic("HCA_O_POST_TOKENS")
HC_MULT = 4
HC_DIM = HC_MULT * D
TOKEN_TILE = 8
# B16/S6 按 16 行分工，覆盖 48 个 Vector 核，避免收尾只使用半数核。
TASK_TOKENS = 16
COL_TILE = 512


@pl.jit.inline
def _decode_o_proj_tp1_parts(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    t_dim: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    PIPELINE_OA: pl.constexpr,
):
    """HCA 的分组 O 投影任务组织，复用 CSA 矩阵乘，保留 HCA 收尾依赖。"""
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    proj_b_t_rows = (t_dim + ROW_TILE - 1) // ROW_TILE
    proj_b_padded_rows = proj_b_t_rows * ROW_TILE

    # Upstream performance arithmetic: per-group quantization and INT32 partials.
    # The archived precision entry kept Native's whole-token quantization separately.
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
                t_dim, proj_a_rows, heads_dep, A_COL_TILE, PIPELINE_OA,
            )

            col_g = g * O_LORA
            # 性能版按上游把 amax 与量化融进同一个 SPMD，并让每个 group 用自己的标度。
            # 已封存的精度版另起 oproj_token_scale 任务、跨全部 O_GROUPS 取同一个 amax，并多做
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

    parts_ready = pl.system.task_dummy(deps=[proj_b_tids[i] for i in range(O_GROUPS)])
    return partials, act_scale_dq, parts_ready


@pl.jit.inline
def _o_proj_hc_post_tiled(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
    MM_ROWS: pl.constexpr,
    MM_COLS: pl.constexpr,
):
    tokens = pl.tensor.dim(residual, 0)
    partials, scales, ready = _decode_o_proj_tp1_parts(
        packed, wo_a, wo_b, tokens, heads_dep, MM_ROWS, MM_COLS, PIPELINE_OA=True,
    )
    residual_flat = pl.reshape(residual, [tokens, HC_DIM])
    output_flat = pl.reshape(output, [tokens, HC_DIM])
    blocks = (tokens + TASK_TOKENS - 1) // TASK_TOKENS
    with pl.spmd(
        blocks * (D // COL_TILE), name_hint="hca_oproj_hc_post",
        deps=[ready], allow_early_resolve=True,
    ):
        worker = pl.tile.get_block_idx()
        row0 = worker // (D // COL_TILE) * TASK_TOKENS
        col0 = worker % (D // COL_TILE) * COL_TILE
        weight_scale = pl.reshape(pl.load(wo_scale, [col0], [COL_TILE]), [1, COL_TILE])
        for row in pl.range(row0, pl.min(row0 + TASK_TOKENS, tokens), TOKEN_TILE):
            valid = pl.min(TOKEN_TILE, tokens - row)
            acc = pl.tile.full([TOKEN_TILE, COL_TILE], dtype=pl.FP32, value=0.0)
            for group in pl.pipeline(O_GROUPS, stage=2):
                part = pl.load(partials, [row, group * D + col0], [TOKEN_TILE, COL_TILE])
                scale = pl.reshape(pl.load(scales, [group, row], [1, TOKEN_TILE]), [TOKEN_TILE, 1])
                acc = pl.add(acc, pl.row_expand_mul(pl.cast(part, pl.FP32), scale))
            attention = pl.col_expand_mul(acc, weight_scale)
            # 保留原先 attention 输出落 BF16、mHC post 再转 FP32 的舍入点。
            attention = pl.cast(pl.cast(attention, pl.BF16, mode="rint"), pl.FP32)
            # Native 门控是行主序；整块读取后收集各列，生成独立的连续权重向量。
            post_rows = pl.load(post, [row, 0], [TOKEN_TILE, 8], valid_shape=[valid, HC_MULT])
            comb_rows = pl.load(comb, [row, 0], [TOKEN_TILE, HC_MULT * HC_MULT],
                                valid_shape=[valid, HC_MULT * HC_MULT])
            # 四个输入HC分量只加载、转FP32一次，供四个输出顺序复用。
            residual_0 = pl.cast(pl.load(
                residual_flat, [row, 0 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            residual_1 = pl.cast(pl.load(
                residual_flat, [row, 1 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            residual_2 = pl.cast(pl.load(
                residual_flat, [row, 2 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            residual_3 = pl.cast(pl.load(
                residual_flat, [row, 3 * D + col0], [TOKEN_TILE, COL_TILE],
                valid_shape=[valid, COL_TILE],
            ), pl.FP32)
            post_columns = pl.transpose(post_rows, axis1=0, axis2=1)
            comb_columns = pl.transpose(comb_rows, axis1=0, axis2=1)
            for out_h in pl.unroll(HC_MULT):
                post_weight = pl.reshape(
                    pl.tile.slice(post_columns, [1, TOKEN_TILE], [out_h, 0]), [TOKEN_TILE, 1],
                )
                value = pl.row_expand_mul(attention, post_weight)
                coefficient_0 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [0 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_0, coefficient_0))
                coefficient_1 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [1 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_1, coefficient_1))
                coefficient_2 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [2 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_2, coefficient_2))
                coefficient_3 = pl.reshape(
                    pl.tile.slice(comb_columns, [1, TOKEN_TILE], [3 * HC_MULT + out_h, 0]),
                    [TOKEN_TILE, 1],
                )
                value = pl.add(value, pl.row_expand_mul(residual_3, coefficient_3))
                result = pl.cast(value, pl.BF16, mode="rint")
                pl.store(pl.set_validshape(result, valid, COL_TILE), [row, out_h * D + col0], output_flat)
    return output


@pl.jit.inline
def o_proj_hc_post(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(residual, 0)
    if tokens <= PROJ_B_SMALL_T_TILE:
        output = _o_proj_hc_post_tiled(
            packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
            PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE,
        )
    elif tokens <= PROJ_B_MEDIUM_T_TILE:
        output = _o_proj_hc_post_tiled(
            packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
            PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE,
        )
    else:
        output = _o_proj_hc_post_tiled(
            packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep,
            PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE,
        )
    return output
