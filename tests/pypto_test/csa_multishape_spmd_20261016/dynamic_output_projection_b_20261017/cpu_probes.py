# SPDX-License-Identifier: Apache-2.0
"""动态输出后段的CPU编译壳；只提供显式worker，不改变正式helper的数值或形状。

actual T从out/residual的动态描述符读取，MM与SCALE保持独立动态轴。
三个RB壳覆盖NZ的两类循环布局和实际32/96/128 Acc形状；不传具体batch。
这里的独立SPMD仅用于后端编译，不代表整层已完成同步或调用接入。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.output_projection_b_stages import (
    MM_ROWS_DYN, QUANT_WEIGHT_LAYOUT, SCALE_ROWS_DYN, T_DYN,
    o_b_stage, o_dequant_hc_post_stage,
)


def make_ob_probe(row_tile):
    """只specialize数学tile；实际T/P_MM仍保持动态。"""
    @pl.jit(auto_scope=False)
    def entry(
        activation: pl.Tensor[[MM_ROWS_DYN, 8192], pl.INT8],
        weight: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
        residual: pl.Tensor[[T_DYN, 16384], pl.BF16],
        partials: pl.Out[pl.Tensor[[MM_ROWS_DYN, 32768], pl.INT32]],
    ):
        tokens = pl.tensor.dim(residual, 0)
        with pl.spmd(24, name_hint="dynamic_output_b"):
            pl.set_cache_policy(weight, pl.CachePolicy.BYPASS)
            worker = pl.tile.get_block_idx()
            partials = o_b_stage(activation, weight, partials, tokens, worker, 24, row_tile)
        return partials
    return entry


ob32 = make_ob_probe(32)
ob96 = make_ob_probe(96)
ob128 = make_ob_probe(128)


@pl.jit(auto_scope=False)
def dequant(
    partials: pl.Tensor[[MM_ROWS_DYN, 32768], pl.INT32],
    scale: pl.Tensor[[8, SCALE_ROWS_DYN], pl.FP32],
    weight_scale: pl.Tensor[[4096], pl.FP32],
    residual: pl.Tensor[[T_DYN, 16384], pl.BF16],
    post: pl.Tensor[[T_DYN, 4], pl.FP32],
    comb: pl.Tensor[[T_DYN, 16], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, 16384], pl.BF16]],
):
    with pl.spmd(48, name_hint="dynamic_output_dequant_hc_post"):
        worker = pl.tile.get_block_idx()
        out = o_dequant_hc_post_stage(partials, scale, weight_scale, residual, post, comb, out, worker, 48)
    return out
