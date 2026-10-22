# SPDX-License-Identifier: Apache-2.0
"""沿用分组O_A/量化/O_B，独立HC_post按整行分工的候选。

参考同仓CSA三个session的逐token收尾：一个AIV处理一整行4096列，使本行
8个量化scale、20个HC门值跨整行复用，消除8行窄块的门控转置与广播。
八组浮点累加、两次BF16舍入及四路残差加法顺序不变；保留O_B原来的
分组提前开工，不采用上一轮等待全组量化的MIX尾链。
T<48保留原二维分工，避免小token数量下整行方案令大量AIV空闲。
该文件仅在冻结候选包中替换output.py，output_base.py为同时冻结的正式原版。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.hc_post import hc_post_block
from .output_base import (
    D, HC_MULT, HC_DIM, O_GROUPS, O_GROUP_IN, O_LORA, T_PAD, T_DYN,
    PROJ_B_MEDIUM_T_TILE, PROJ_B_MM_T_TILE, PROJ_A_MM_N_TILE, PROJ_A_LARGE_N_TILE,
    QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT, _decode_o_proj_tp1_parts,
    o_proj_hc_post as baseline_output,
)


@pl.jit.inline(auto_scope=False)
def _output_rows(
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
    with pl.spmd(48, name_hint="hca_post_rows", deps=[ready], allow_early_resolve=True):
        scale_w = pl.reshape(pl.load(wo_scale, [0], [D]), [1, D])
        for token in pl.range(pl.tile.get_block_idx(), tokens, 48):
            total = pl.tile.full([1, D], dtype=pl.FP32, value=0.0)
            for group in pl.pipeline(O_GROUPS, stage=2):
                part = pl.load(partials, [token, group * D], [1, D])
                activation_scale = pl.read(scales, [group, token])
                contribution = pl.mul(pl.cast(part, pl.FP32), activation_scale)
                total = pl.add(total, contribution)
            projected = pl.col_expand_mul(total, scale_w)
            # 显式Tensor元数据种子供原公共hc_post helper绑定，实际值仍由一次cast产生。
            rounded = pl.create_tensor([1, D], dtype=pl.BF16)
            rounded = pl.cast(projected, pl.BF16, mode="rint")
            output_flat = hc_post_block(rounded, residual_flat, post, comb, output_flat, token, 0, 1, 1, D)
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
    if tokens < 48:
        output = baseline_output(packed, wo_a, wo_b, wo_scale, residual, post, comb, output, heads_dep)
    elif tokens <= PROJ_B_MEDIUM_T_TILE:
        output = _output_rows(packed, wo_a, wo_b, wo_scale, residual, post, comb, output,
                              heads_dep, PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE)
    else:
        output = _output_rows(packed, wo_a, wo_b, wo_scale, residual, post, comb, output,
                              heads_dep, PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE)
    return output
