# SPDX-License-Identifier: Apache-2.0
"""HC-pre Native-order candidate, T<=256/A3 only.

The HC projection reproduces HF32 operand conversion and exact-eight-product
accumulation using existing FP32 DSL operations; no extern or Native arithmetic.
All remaining arithmetic is DSL; input RMSNorm is outside this module.
Original Native weights are used unchanged. No compiler interception.
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.config import FLASH as M
from .hc_pre_panel import hcpre_linear_panel

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # actual token rows, independent request lengths
HC_PAD_ROWS_DYN = pl.dynamic("HC_PAD_ROWS_DYN")
LINEAR_ROWS_DYN = pl.dynamic("LINEAR_ROWS_DYN")

# model config
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# tiling
MIX_PAD = 32  # mix_hc (24) padded to a 32-wide cube N / vector row
HC_PAD = 8  # hc (4) padded for 32B-aligned vector ops
T_TILE = 8  # other values miscompare
LINEAR_T_TILE = 16  # cube matmul rows must be a 16-row boxed tile
COMB_T_TILE = 8
PRE_POST_WORKERS = 16
RMS_K_TILE = 512
LINEAR_K_TILE = 32
D_TILE = 256
D_SPMD = 4096
LINEAR_OK = 22
LINEAR_K_PER_SPLIT = 768
LINEAR_WORKERS = 24  # hc_pre_linear workers, one per AIC
PANEL_MIN_TOKENS = 1  # All qualified nonempty queries use the Cube panel path.

# pre0..pre3 / row0..row3 are hand-unrolled over the hc lanes.
assert HC_MULT == 4, f"hc_pre is specialized to HC_MULT == 4, got {HC_MULT}"


def _hcpre_linear_dsl(
    flat: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    weight: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    partials: pl.Out[pl.Tensor[[LINEAR_ROWS_DYN, MIX_PAD], pl.FP32]],
):
    flat.bind_dynamic(0, T_DYN)
    partials.bind_dynamic(0, LINEAR_ROWS_DYN)
    tokens = pl.tensor.dim(flat, 0)
    padded_tokens = pl.tensor.dim(partials, 0) // LINEAR_OK
    # HF32 operand conversion is explicit bit arithmetic on original FP32 weight.
    # Transpose is DSL layout work, not a Native arithmetic callback.
    rounded_weight = pl.create_tensor([HC_DIM, MIX_PAD], dtype=pl.FP32)
    for block in pl.spmd(HC_DIM // 32, name_hint="hcpre_hf32_weight", allow_early_resolve=True):
        raw: pl.Tile[[32, 32], pl.FP32] = pl.load(weight, [0, block * 32], [32, 32], valid_shape=[MIX_HC, 32])
        transposed = pl.tile.transpose(raw, axis1=0, axis2=1)
        bits = pl.tile.reinterpret_view(transposed, pl.INT32)
        converted = pl.tile.reinterpret_view(
            pl.tile.and_(pl.tile.add(bits, 2048), pl.tile.full([32, 32], dtype=pl.INT32, value=-4096)), pl.FP32
        )
        for pad_column in pl.range(MIX_HC, MIX_PAD):
            for pad_k in pl.range(32):
                pl.write(converted, [pad_k, pad_column], 0.0)
        pl.store(pl.set_validshape(converted, 32, 32), [block * 32, 0], rounded_weight)
    for unit in pl.spmd(LINEAR_OK * padded_tokens, name_hint="hcpre_hf32_exact8", allow_early_resolve=True):
        split = unit // padded_tokens
        row = unit % padded_tokens
        zero = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
        one = pl.tile.full([1, 32], dtype=pl.FP32, value=1.0)
        sel_tmp = pl.create_tile([1, 16], dtype=pl.UINT32)
        accumulator = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
        if row < tokens:
            count = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - split * LINEAR_K_PER_SPLIT) // 8
            for kb in pl.range(count):
                k = split * LINEAR_K_PER_SPLIT + kb * 8
                weight0: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 0, 0], [1, 32])
                input0 = pl.read(flat, [row, k + 0])
                product0 = pl.tile.mul(weight0, input0)
                weight1: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 1, 0], [1, 32])
                input1 = pl.read(flat, [row, k + 1])
                product1 = pl.tile.mul(weight1, input1)
                weight2: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 2, 0], [1, 32])
                input2 = pl.read(flat, [row, k + 2])
                product2 = pl.tile.mul(weight2, input2)
                weight3: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 3, 0], [1, 32])
                input3 = pl.read(flat, [row, k + 3])
                product3 = pl.tile.mul(weight3, input3)
                weight4: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 4, 0], [1, 32])
                input4 = pl.read(flat, [row, k + 4])
                product4 = pl.tile.mul(weight4, input4)
                weight5: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 5, 0], [1, 32])
                input5 = pl.read(flat, [row, k + 5])
                product5 = pl.tile.mul(weight5, input5)
                weight6: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 6, 0], [1, 32])
                input6 = pl.read(flat, [row, k + 6])
                product6 = pl.tile.mul(weight6, input6)
                weight7: pl.Tile[[1, 32], pl.FP32] = pl.load(rounded_weight, [k + 7, 0], [1, 32])
                input7 = pl.read(flat, [row, k + 7])
                product7 = pl.tile.mul(weight7, input7)
                e0_0_hi = pl.tile.add(product0, accumulator)
                e0_0_v = pl.tile.sub(e0_0_hi, product0)
                e0_0_lo = pl.tile.add(
                    pl.tile.sub(product0, pl.tile.sub(e0_0_hi, e0_0_v)), pl.tile.sub(accumulator, e0_0_v)
                )
                e1_0_hi = pl.tile.add(product1, e0_0_lo)
                e1_0_v = pl.tile.sub(e1_0_hi, product1)
                e1_0_lo = pl.tile.add(pl.tile.sub(product1, pl.tile.sub(e1_0_hi, e1_0_v)), pl.tile.sub(e0_0_lo, e1_0_v))
                e1_1_hi = pl.tile.add(e1_0_hi, e0_0_hi)
                e1_1_v = pl.tile.sub(e1_1_hi, e1_0_hi)
                e1_1_lo = pl.tile.add(pl.tile.sub(e1_0_hi, pl.tile.sub(e1_1_hi, e1_1_v)), pl.tile.sub(e0_0_hi, e1_1_v))
                e2_0_hi = pl.tile.add(product2, e1_0_lo)
                e2_0_v = pl.tile.sub(e2_0_hi, product2)
                e2_0_lo = pl.tile.add(pl.tile.sub(product2, pl.tile.sub(e2_0_hi, e2_0_v)), pl.tile.sub(e1_0_lo, e2_0_v))
                e2_1_hi = pl.tile.add(e2_0_hi, e1_1_lo)
                e2_1_v = pl.tile.sub(e2_1_hi, e2_0_hi)
                e2_1_lo = pl.tile.add(pl.tile.sub(e2_0_hi, pl.tile.sub(e2_1_hi, e2_1_v)), pl.tile.sub(e1_1_lo, e2_1_v))
                e2_2_hi = pl.tile.add(e2_1_hi, e1_1_hi)
                e2_2_v = pl.tile.sub(e2_2_hi, e2_1_hi)
                e2_2_lo = pl.tile.add(pl.tile.sub(e2_1_hi, pl.tile.sub(e2_2_hi, e2_2_v)), pl.tile.sub(e1_1_hi, e2_2_v))
                e3_0_hi = pl.tile.add(product3, e2_0_lo)
                e3_0_v = pl.tile.sub(e3_0_hi, product3)
                e3_0_lo = pl.tile.add(pl.tile.sub(product3, pl.tile.sub(e3_0_hi, e3_0_v)), pl.tile.sub(e2_0_lo, e3_0_v))
                e3_1_hi = pl.tile.add(e3_0_hi, e2_1_lo)
                e3_1_v = pl.tile.sub(e3_1_hi, e3_0_hi)
                e3_1_lo = pl.tile.add(pl.tile.sub(e3_0_hi, pl.tile.sub(e3_1_hi, e3_1_v)), pl.tile.sub(e2_1_lo, e3_1_v))
                e3_2_hi = pl.tile.add(e3_1_hi, e2_2_lo)
                e3_2_v = pl.tile.sub(e3_2_hi, e3_1_hi)
                e3_2_lo = pl.tile.add(pl.tile.sub(e3_1_hi, pl.tile.sub(e3_2_hi, e3_2_v)), pl.tile.sub(e2_2_lo, e3_2_v))
                e3_3_hi = pl.tile.add(e3_2_hi, e2_2_hi)
                e3_3_v = pl.tile.sub(e3_3_hi, e3_2_hi)
                e3_3_lo = pl.tile.add(pl.tile.sub(e3_2_hi, pl.tile.sub(e3_3_hi, e3_3_v)), pl.tile.sub(e2_2_hi, e3_3_v))
                e4_0_hi = pl.tile.add(product4, e3_0_lo)
                e4_0_v = pl.tile.sub(e4_0_hi, product4)
                e4_0_lo = pl.tile.add(pl.tile.sub(product4, pl.tile.sub(e4_0_hi, e4_0_v)), pl.tile.sub(e3_0_lo, e4_0_v))
                e4_1_hi = pl.tile.add(e4_0_hi, e3_1_lo)
                e4_1_v = pl.tile.sub(e4_1_hi, e4_0_hi)
                e4_1_lo = pl.tile.add(pl.tile.sub(e4_0_hi, pl.tile.sub(e4_1_hi, e4_1_v)), pl.tile.sub(e3_1_lo, e4_1_v))
                e4_2_hi = pl.tile.add(e4_1_hi, e3_2_lo)
                e4_2_v = pl.tile.sub(e4_2_hi, e4_1_hi)
                e4_2_lo = pl.tile.add(pl.tile.sub(e4_1_hi, pl.tile.sub(e4_2_hi, e4_2_v)), pl.tile.sub(e3_2_lo, e4_2_v))
                e4_3_hi = pl.tile.add(e4_2_hi, e3_3_lo)
                e4_3_v = pl.tile.sub(e4_3_hi, e4_2_hi)
                e4_3_lo = pl.tile.add(pl.tile.sub(e4_2_hi, pl.tile.sub(e4_3_hi, e4_3_v)), pl.tile.sub(e3_3_lo, e4_3_v))
                e4_4_hi = pl.tile.add(e4_3_hi, e3_3_hi)
                e4_4_v = pl.tile.sub(e4_4_hi, e4_3_hi)
                e4_4_lo = pl.tile.add(pl.tile.sub(e4_3_hi, pl.tile.sub(e4_4_hi, e4_4_v)), pl.tile.sub(e3_3_hi, e4_4_v))
                e5_0_hi = pl.tile.add(product5, e4_0_lo)
                e5_0_v = pl.tile.sub(e5_0_hi, product5)
                e5_0_lo = pl.tile.add(pl.tile.sub(product5, pl.tile.sub(e5_0_hi, e5_0_v)), pl.tile.sub(e4_0_lo, e5_0_v))
                e5_1_hi = pl.tile.add(e5_0_hi, e4_1_lo)
                e5_1_v = pl.tile.sub(e5_1_hi, e5_0_hi)
                e5_1_lo = pl.tile.add(pl.tile.sub(e5_0_hi, pl.tile.sub(e5_1_hi, e5_1_v)), pl.tile.sub(e4_1_lo, e5_1_v))
                e5_2_hi = pl.tile.add(e5_1_hi, e4_2_lo)
                e5_2_v = pl.tile.sub(e5_2_hi, e5_1_hi)
                e5_2_lo = pl.tile.add(pl.tile.sub(e5_1_hi, pl.tile.sub(e5_2_hi, e5_2_v)), pl.tile.sub(e4_2_lo, e5_2_v))
                e5_3_hi = pl.tile.add(e5_2_hi, e4_3_lo)
                e5_3_v = pl.tile.sub(e5_3_hi, e5_2_hi)
                e5_3_lo = pl.tile.add(pl.tile.sub(e5_2_hi, pl.tile.sub(e5_3_hi, e5_3_v)), pl.tile.sub(e4_3_lo, e5_3_v))
                e5_4_hi = pl.tile.add(e5_3_hi, e4_4_lo)
                e5_4_v = pl.tile.sub(e5_4_hi, e5_3_hi)
                e5_4_lo = pl.tile.add(pl.tile.sub(e5_3_hi, pl.tile.sub(e5_4_hi, e5_4_v)), pl.tile.sub(e4_4_lo, e5_4_v))
                e5_5_hi = pl.tile.add(e5_4_hi, e4_4_hi)
                e5_5_v = pl.tile.sub(e5_5_hi, e5_4_hi)
                e5_5_lo = pl.tile.add(pl.tile.sub(e5_4_hi, pl.tile.sub(e5_5_hi, e5_5_v)), pl.tile.sub(e4_4_hi, e5_5_v))
                e6_0_hi = pl.tile.add(product6, e5_0_lo)
                e6_0_v = pl.tile.sub(e6_0_hi, product6)
                e6_0_lo = pl.tile.add(pl.tile.sub(product6, pl.tile.sub(e6_0_hi, e6_0_v)), pl.tile.sub(e5_0_lo, e6_0_v))
                e6_1_hi = pl.tile.add(e6_0_hi, e5_1_lo)
                e6_1_v = pl.tile.sub(e6_1_hi, e6_0_hi)
                e6_1_lo = pl.tile.add(pl.tile.sub(e6_0_hi, pl.tile.sub(e6_1_hi, e6_1_v)), pl.tile.sub(e5_1_lo, e6_1_v))
                e6_2_hi = pl.tile.add(e6_1_hi, e5_2_lo)
                e6_2_v = pl.tile.sub(e6_2_hi, e6_1_hi)
                e6_2_lo = pl.tile.add(pl.tile.sub(e6_1_hi, pl.tile.sub(e6_2_hi, e6_2_v)), pl.tile.sub(e5_2_lo, e6_2_v))
                e6_3_hi = pl.tile.add(e6_2_hi, e5_3_lo)
                e6_3_v = pl.tile.sub(e6_3_hi, e6_2_hi)
                e6_3_lo = pl.tile.add(pl.tile.sub(e6_2_hi, pl.tile.sub(e6_3_hi, e6_3_v)), pl.tile.sub(e5_3_lo, e6_3_v))
                e6_4_hi = pl.tile.add(e6_3_hi, e5_4_lo)
                e6_4_v = pl.tile.sub(e6_4_hi, e6_3_hi)
                e6_4_lo = pl.tile.add(pl.tile.sub(e6_3_hi, pl.tile.sub(e6_4_hi, e6_4_v)), pl.tile.sub(e5_4_lo, e6_4_v))
                e6_5_hi = pl.tile.add(e6_4_hi, e5_5_lo)
                e6_5_v = pl.tile.sub(e6_5_hi, e6_4_hi)
                e6_5_lo = pl.tile.add(pl.tile.sub(e6_4_hi, pl.tile.sub(e6_5_hi, e6_5_v)), pl.tile.sub(e5_5_lo, e6_5_v))
                e6_6_hi = pl.tile.add(e6_5_hi, e5_5_hi)
                e6_6_v = pl.tile.sub(e6_6_hi, e6_5_hi)
                e6_6_lo = pl.tile.add(pl.tile.sub(e6_5_hi, pl.tile.sub(e6_6_hi, e6_6_v)), pl.tile.sub(e5_5_hi, e6_6_v))
                e7_0_hi = pl.tile.add(product7, e6_0_lo)
                e7_0_v = pl.tile.sub(e7_0_hi, product7)
                e7_0_lo = pl.tile.add(pl.tile.sub(product7, pl.tile.sub(e7_0_hi, e7_0_v)), pl.tile.sub(e6_0_lo, e7_0_v))
                e7_1_hi = pl.tile.add(e7_0_hi, e6_1_lo)
                e7_1_v = pl.tile.sub(e7_1_hi, e7_0_hi)
                e7_1_lo = pl.tile.add(pl.tile.sub(e7_0_hi, pl.tile.sub(e7_1_hi, e7_1_v)), pl.tile.sub(e6_1_lo, e7_1_v))
                e7_2_hi = pl.tile.add(e7_1_hi, e6_2_lo)
                e7_2_v = pl.tile.sub(e7_2_hi, e7_1_hi)
                e7_2_lo = pl.tile.add(pl.tile.sub(e7_1_hi, pl.tile.sub(e7_2_hi, e7_2_v)), pl.tile.sub(e6_2_lo, e7_2_v))
                e7_3_hi = pl.tile.add(e7_2_hi, e6_3_lo)
                e7_3_v = pl.tile.sub(e7_3_hi, e7_2_hi)
                e7_3_lo = pl.tile.add(pl.tile.sub(e7_2_hi, pl.tile.sub(e7_3_hi, e7_3_v)), pl.tile.sub(e6_3_lo, e7_3_v))
                e7_4_hi = pl.tile.add(e7_3_hi, e6_4_lo)
                e7_4_v = pl.tile.sub(e7_4_hi, e7_3_hi)
                e7_4_lo = pl.tile.add(pl.tile.sub(e7_3_hi, pl.tile.sub(e7_4_hi, e7_4_v)), pl.tile.sub(e6_4_lo, e7_4_v))
                e7_5_hi = pl.tile.add(e7_4_hi, e6_5_lo)
                e7_5_v = pl.tile.sub(e7_5_hi, e7_4_hi)
                e7_5_lo = pl.tile.add(pl.tile.sub(e7_4_hi, pl.tile.sub(e7_5_hi, e7_5_v)), pl.tile.sub(e6_5_lo, e7_5_v))
                e7_6_hi = pl.tile.add(e7_5_hi, e6_6_lo)
                e7_6_v = pl.tile.sub(e7_6_hi, e7_5_hi)
                e7_6_lo = pl.tile.add(pl.tile.sub(e7_5_hi, pl.tile.sub(e7_6_hi, e7_6_v)), pl.tile.sub(e6_6_lo, e7_6_v))
                e7_7_hi = pl.tile.add(e7_6_hi, e6_6_hi)
                e7_7_v = pl.tile.sub(e7_7_hi, e7_6_hi)
                e7_7_lo = pl.tile.add(pl.tile.sub(e7_6_hi, pl.tile.sub(e7_7_hi, e7_7_v)), pl.tile.sub(e6_6_hi, e7_7_v))
                tail_prefix_1 = pl.tile.sel(pl.tile.cmp(e7_1_lo, zero, cmp_type=1), e7_1_lo, e7_0_lo, sel_tmp)
                tail_prefix_2 = pl.tile.sel(pl.tile.cmp(e7_2_lo, zero, cmp_type=1), e7_2_lo, tail_prefix_1, sel_tmp)
                tail_prefix_3 = pl.tile.sel(pl.tile.cmp(e7_3_lo, zero, cmp_type=1), e7_3_lo, tail_prefix_2, sel_tmp)
                tail_prefix_4 = pl.tile.sel(pl.tile.cmp(e7_4_lo, zero, cmp_type=1), e7_4_lo, tail_prefix_3, sel_tmp)
                tail_prefix_5 = pl.tile.sel(pl.tile.cmp(e7_5_lo, zero, cmp_type=1), e7_5_lo, tail_prefix_4, sel_tmp)
                tail_prefix_6 = pl.tile.sel(pl.tile.cmp(e7_6_lo, zero, cmp_type=1), e7_6_lo, tail_prefix_5, sel_tmp)
                rounded = e7_7_hi
                residual = zero
                remaining = zero
                active = one
                round7_sum = pl.tile.add(rounded, e7_7_lo)
                round7_error = pl.tile.sub(e7_7_lo, pl.tile.sub(round7_sum, rounded))
                round7_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round7_active, round7_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round7_active, round7_error, residual, sel_tmp)
                remaining = pl.tile.sel(round7_active, tail_prefix_6, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round7_error, zero, cmp_type=0), active, zero, sel_tmp)
                round6_sum = pl.tile.add(rounded, e7_6_lo)
                round6_error = pl.tile.sub(e7_6_lo, pl.tile.sub(round6_sum, rounded))
                round6_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round6_active, round6_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round6_active, round6_error, residual, sel_tmp)
                remaining = pl.tile.sel(round6_active, tail_prefix_5, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round6_error, zero, cmp_type=0), active, zero, sel_tmp)
                round5_sum = pl.tile.add(rounded, e7_5_lo)
                round5_error = pl.tile.sub(e7_5_lo, pl.tile.sub(round5_sum, rounded))
                round5_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round5_active, round5_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round5_active, round5_error, residual, sel_tmp)
                remaining = pl.tile.sel(round5_active, tail_prefix_4, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round5_error, zero, cmp_type=0), active, zero, sel_tmp)
                round4_sum = pl.tile.add(rounded, e7_4_lo)
                round4_error = pl.tile.sub(e7_4_lo, pl.tile.sub(round4_sum, rounded))
                round4_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round4_active, round4_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round4_active, round4_error, residual, sel_tmp)
                remaining = pl.tile.sel(round4_active, tail_prefix_3, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round4_error, zero, cmp_type=0), active, zero, sel_tmp)
                round3_sum = pl.tile.add(rounded, e7_3_lo)
                round3_error = pl.tile.sub(e7_3_lo, pl.tile.sub(round3_sum, rounded))
                round3_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round3_active, round3_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round3_active, round3_error, residual, sel_tmp)
                remaining = pl.tile.sel(round3_active, tail_prefix_2, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round3_error, zero, cmp_type=0), active, zero, sel_tmp)
                round2_sum = pl.tile.add(rounded, e7_2_lo)
                round2_error = pl.tile.sub(e7_2_lo, pl.tile.sub(round2_sum, rounded))
                round2_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round2_active, round2_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round2_active, round2_error, residual, sel_tmp)
                remaining = pl.tile.sel(round2_active, tail_prefix_1, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round2_error, zero, cmp_type=0), active, zero, sel_tmp)
                round1_sum = pl.tile.add(rounded, e7_1_lo)
                round1_error = pl.tile.sub(e7_1_lo, pl.tile.sub(round1_sum, rounded))
                round1_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round1_active, round1_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round1_active, round1_error, residual, sel_tmp)
                remaining = pl.tile.sel(round1_active, e7_0_lo, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round1_error, zero, cmp_type=0), active, zero, sel_tmp)
                round0_sum = pl.tile.add(rounded, e7_0_lo)
                round0_error = pl.tile.sub(e7_0_lo, pl.tile.sub(round0_sum, rounded))
                round0_active = pl.tile.cmp(active, zero, cmp_type=1)
                rounded = pl.tile.sel(round0_active, round0_sum, rounded, sel_tmp)
                residual = pl.tile.sel(round0_active, round0_error, residual, sel_tmp)
                remaining = pl.tile.sel(round0_active, zero, remaining, sel_tmp)
                active = pl.tile.sel(pl.tile.cmp(round0_error, zero, cmp_type=0), active, zero, sel_tmp)
                positive_tail = pl.tile.sel(pl.tile.cmp(remaining, zero, cmp_type=4), one, zero, sel_tmp)
                negative_tail = pl.tile.sel(pl.tile.cmp(remaining, zero, cmp_type=2), one, zero, sel_tmp)
                positive = pl.tile.sel(pl.tile.cmp(residual, zero, cmp_type=4), positive_tail, zero, sel_tmp)
                negative = pl.tile.sel(pl.tile.cmp(residual, zero, cmp_type=2), negative_tail, zero, sel_tmp)
                same_sign = pl.tile.cmp(pl.tile.add(positive, negative), zero, cmp_type=4)
                twice = pl.tile.mul(residual, 2.0)
                adjusted = pl.tile.add(rounded, twice)
                exact_half = pl.tile.cmp(pl.tile.sub(adjusted, rounded), twice, cmp_type=0)
                tie_adjusted = pl.tile.sel(exact_half, adjusted, rounded, sel_tmp)
                rounded = pl.tile.sel(same_sign, tie_adjusted, rounded, sel_tmp)
                accumulator = rounded
        pl.store(accumulator, [unit, 0], partials)
    return partials


hcpre_linear_dsl = pl.jit.inline(auto_scope=False)(_hcpre_linear_dsl)
hcpre_linear_test = pl.jit(_hcpre_linear_dsl)


@pl.jit.inline
def hc_pre_gates_from_rms(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[HC_PAD_ROWS_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
    inv_rms: pl.Tensor[[HC_PAD_ROWS_DYN, 1], pl.FP32],
):
    """共享门控与Sinkhorn算术，复用调用方按原规则生成的RMS统计。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE  # pad t_dim up to whole 16-row cube tiles
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])  # for per-group comb base loads in comb_sinkhorn

    # Both paths zero-fill padded partial rows. The panel path explicitly
    # initializes the transposed input tail before sharing weight loads.
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    if t_dim < PANEL_MIN_TOKENS:
        mixes_partials = hcpre_linear_dsl(x_flat, hc_fn, mixes_partials)
    else:
        mixes_partials = hcpre_linear_panel(x_flat, hc_fn, mixes_partials)

    # Partials are reduced in ascending K order.
    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for linear_block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="hc_pre_linear_reduce", allow_early_resolve=True):
        linear_t0 = linear_block * LINEAR_T_TILE
        mixes_total = mixes_partials[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for linear_split in pl.range(1, LINEAR_OK):
            partial_t0 = linear_split * t_linear + linear_t0
            partial_tile = mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:MIX_PAD]
            mixes_total = pl.add(mixes_total, partial_tile)
        mixes_raw[linear_t0 : linear_t0 + LINEAR_T_TILE, 0:MIX_PAD] = mixes_total

    # split_pre_post: inv_rms-scaled pre gate -> pre_val_store (for mix_x), post gate -> post.
    # Both compute at HC_PAD width; post narrows to HC_MULT via a valid-shape slice (an 8-wide
    # 32B tile, 4 cols valid -- a bare 4-wide slice allocs a 16B tile ptoas rejects). comb gate
    # lives in comb_sinkhorn.
    # Only the final partial token tile uses these fixed-size staging buffers.
    post_tail_store = pl.create_tensor([T_TILE, HC_PAD], dtype=pl.FP32)
    for ob_worker in pl.spmd(
        pl.min(token_tiles, PRE_POST_WORKERS), name_hint="split_pre_post", allow_early_resolve=True
    ):
        scale0 = pl.read(hc_scale, [0])
        scale1 = pl.read(hc_scale, [1])
        for ob in pl.range(ob_worker, token_tiles, pl.min(token_tiles, PRE_POST_WORKERS)):
            t0 = ob * T_TILE
            valid_rows = pl.min(T_TILE, t_dim - t0)
            inv_col = inv_rms[t0 : t0 + T_TILE, 0:1]

            pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
            pre_scaled = pl.mul(pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], inv_col), scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.div(
                pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(pre_logits)), 1.0)
            )
            pre_val = pl.add(pre_sig, HC_EPS)
            pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD] = pre_val

            post_base = pl.reshape(hc_base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
            post_scaled = pl.mul(
                pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], inv_col), scale1
            )
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.div(
                pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), pl.add(pl.exp(pl.neg(post_logits)), 1.0)
            )
            post_pad = pl.mul(post_sig, 2.0)
            if valid_rows == T_TILE:
                post[t0 : t0 + T_TILE, 0:HC_MULT] = pl.slice(
                    post_pad, [T_TILE, HC_PAD], [0, 0], valid_shape=[T_TILE, HC_MULT]
                )
            else:
                post_tail_store[0:T_TILE, 0:HC_PAD] = post_pad
                post_tile = pl.load(
                    post_tail_store,
                    [0, 0],
                    [T_TILE, HC_PAD],
                    valid_shape=[valid_rows, HC_MULT],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(post_tile, [t0, 0], post)

    # comb_sinkhorn: comb gate from mixes_raw cols 8/12/16/20, softmax, then a
    # column-first 20-iteration Sinkhorn -> comb.
    comb_tail_store = pl.create_tensor([COMB_T_TILE, HC_PAD * HC_MULT], dtype=pl.FP32)
    for ob in pl.spmd(token_tiles, name_hint="comb_sinkhorn", allow_early_resolve=True):
        scale2 = pl.read(hc_scale, [2])
        t0 = ob * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        inv_col_t = pl.load(
            inv_rms, [t0, 0], [COMB_T_TILE, 1], valid_shape=[valid_rows, 1], target_memory=pl.MemorySpace.Vec
        )
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(
            mixes_raw,
            [t0, comb_off + 0 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g1 = pl.load(
            mixes_raw,
            [t0, comb_off + 1 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g2 = pl.load(
            mixes_raw,
            [t0, comb_off + 2 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix_g3 = pl.load(
            mixes_raw,
            [t0, comb_off + 3 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb0 = pl.load(
            hc_base_2d,
            [0, comb_off + 0 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb1 = pl.load(
            hc_base_2d,
            [0, comb_off + 1 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb2 = pl.load(
            hc_base_2d,
            [0, comb_off + 2 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        cb3 = pl.load(
            hc_base_2d,
            [0, comb_off + 3 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        row0 = pl.add(pl.mul(pl.row_expand_mul(mix_g0, inv_col_t), scale2), pl.col_expand(mix_g0, cb0))
        row1 = pl.add(pl.mul(pl.row_expand_mul(mix_g1, inv_col_t), scale2), pl.col_expand(mix_g1, cb1))
        row2 = pl.add(pl.mul(pl.row_expand_mul(mix_g2, inv_col_t), scale2), pl.col_expand(mix_g2, cb2))
        row3 = pl.add(pl.mul(pl.row_expand_mul(mix_g3, inv_col_t), scale2), pl.col_expand(mix_g3, cb3))
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_max = pl.row_max(row0_p, row_max_tmp)
        row1_max = pl.row_max(row1_p, row_max_tmp)
        row2_max = pl.row_max(row2_p, row_max_tmp)
        row3_max = pl.row_max(row3_p, row_max_tmp)
        row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
        row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
        row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
        row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
        row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
        row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
        row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
        row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
        if row_recip:
            row0_inv = pl.recip(row0_sum)
            row0_prob = pl.row_expand_mul(row0_exp, row0_inv)
            row0_soft = pl.add(row0_prob, HC_EPS)
            row1_inv = pl.recip(row1_sum)
            row1_prob = pl.row_expand_mul(row1_exp, row1_inv)
            row1_soft = pl.add(row1_prob, HC_EPS)
            row2_inv = pl.recip(row2_sum)
            row2_prob = pl.row_expand_mul(row2_exp, row2_inv)
            row2_soft = pl.add(row2_prob, HC_EPS)
            row3_inv = pl.recip(row3_sum)
            row3_prob = pl.row_expand_mul(row3_exp, row3_inv)
            row3_soft = pl.add(row3_prob, HC_EPS)
        else:
            row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)

        row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        col_sum = pl.add(pl.add(pl.add(row0_eff, row1_eff), row2_eff), row3_eff)
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)

        for _sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
            if row_recip:
                row0_inv = pl.recip(row0_rowsum)
                row0_norm = pl.row_expand_mul(row0_cur, row0_inv)
                row1_inv = pl.recip(row1_rowsum)
                row1_norm = pl.row_expand_mul(row1_cur, row1_inv)
                row2_inv = pl.recip(row2_rowsum)
                row2_norm = pl.row_expand_mul(row2_cur, row2_inv)
                row3_inv = pl.recip(row3_rowsum)
                row3_norm = pl.row_expand_mul(row3_cur, row3_inv)
            else:
                row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
            col_sum = pl.add(pl.add(pl.add(row0_norm, row1_norm), row2_norm), row3_norm)
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_norm, col_sum)
            row1_cur = pl.div(row1_norm, col_sum)
            row2_cur = pl.div(row2_norm, col_sum)
            row3_cur = pl.div(row3_norm, col_sum)

        if valid_rows == COMB_T_TILE:
            row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
            row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
            row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
            row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
            pl.store(row0_out, [t0, 0 * HC_MULT], comb)
            pl.store(row1_out, [t0, 1 * HC_MULT], comb)
            pl.store(row2_out, [t0, 2 * HC_MULT], comb)
            pl.store(row3_out, [t0, 3 * HC_MULT], comb)
        else:
            pl.store(row0_cur, [0, 0 * HC_PAD], comb_tail_store)
            pl.store(row1_cur, [0, 1 * HC_PAD], comb_tail_store)
            pl.store(row2_cur, [0, 2 * HC_PAD], comb_tail_store)
            pl.store(row3_cur, [0, 3 * HC_PAD], comb_tail_store)
            row0_tail = pl.load(
                comb_tail_store,
                [0, 0 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row1_tail = pl.load(
                comb_tail_store,
                [0, 1 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row2_tail = pl.load(
                comb_tail_store,
                [0, 2 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row3_tail = pl.load(
                comb_tail_store,
                [0, 3 * HC_PAD],
                [COMB_T_TILE, HC_PAD],
                valid_shape=[valid_rows, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(row0_tail, [t0, 0 * HC_MULT], comb)
            pl.store(row1_tail, [t0, 1 * HC_MULT], comb)
            pl.store(row2_tail, [t0, 2 * HC_MULT], comb)
            pl.store(row3_tail, [t0, 3 * HC_MULT], comb)

    return pre_val_store


@pl.jit.inline
def hc_pre_gates(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_val_store: pl.Tensor[[T_DYN, HC_PAD], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    row_recip: pl.Scalar[pl.BOOL],
):
    """保留原独立RMS任务及其归约次序。"""
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)

    # Diagnostic scope T<=256/AIC24; matched Native Gram partials, then exact K order.
    sq_partials = pl.create_tensor([LINEAR_OK * t_linear, 16], dtype=pl.FP32)
    sq_group = 1
    if t_dim >= 64:
        sq_group = 4
    sq_groups = (LINEAR_OK + sq_group - 1) // sq_group
    sq_units = (t_linear // 16) * sq_groups
    for sq_unit in pl.spmd(sq_units, name_hint="hc_pre_square_partial", allow_early_resolve=True):
        sq_t0 = (sq_unit // sq_groups) * 16
        sq_part_base = (sq_unit % sq_groups) * sq_group
        sq_rows = pl.min(16, t_dim - sq_t0)
        for sq_group_item in pl.range(pl.min(sq_group, LINEAR_OK - sq_part_base)):
            sq_part = sq_part_base + sq_group_item
            sq_count = pl.min(768, HC_DIM - sq_part * 768) // 32
            native_sq_acc = pl.create_tensor([16, 16], dtype=pl.FP32)
            for sq_kb in pl.pipeline(0, sq_count, stage=2):
                sq_k = sq_part * 768 + sq_kb * 32
                sq_lhs = pl.slice(x_flat, [16, 32], [sq_t0, sq_k], valid_shape=[sq_rows, 32])
                native_sq_acc = pl.matmul_acc(native_sq_acc, sq_lhs, sq_lhs, b_trans=True, init_cond=(sq_kb == 0))
            sq_out_row = sq_part * t_linear + sq_t0
            sq_partials[sq_out_row : sq_out_row + 16, 0:16] = native_sq_acc
    for rms_block in pl.spmd(token_tiles, name_hint="hc_pre_native_rms", allow_early_resolve=True):
        rms_t0 = rms_block * T_TILE
        rms_lanes = pl.arange(0, [1, T_TILE], dtype=pl.INT32)
        rms_rows = pl.add(rms_lanes, pl.cast(rms_t0, pl.INT32))
        rms_local = pl.add(rms_lanes, pl.cast(rms_t0 % 16, pl.INT32))
        rms_index = pl.add(pl.mul(rms_rows, 16), rms_local)
        rms_total = pl.gather(sq_partials, index=rms_index)
        for rms_part in pl.range(1, LINEAR_OK):
            rms_part_index = pl.add(rms_index, pl.cast(rms_part * t_linear * 16, pl.INT32))
            rms_value = pl.gather(sq_partials, index=rms_part_index)
            rms_total = pl.add(rms_total, rms_value)
        rms_variance = pl.add(pl.mul(rms_total, HC_DIM_INV), NORM_EPS)
        rms_root = pl.sqrt(rms_variance)
        native_inverse = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=1.0), rms_root)
        inv_rms[rms_t0 : rms_t0 + T_TILE, 0:1] = pl.reshape(native_inverse, [T_TILE, 1])

    return hc_pre_gates_from_rms(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        pre_val_store,
        post,
        comb,
        row_recip,
        inv_rms,
    )


def _hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Compute HC gates and BF16 pre-mixed activations."""
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(x, 0)
    token_tiles = (t_dim + T_TILE - 1) // T_TILE
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    pre_val_store = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    hc_pre_gates(x, hc_fn, hc_scale, hc_base, pre_val_store, post, comb, False)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])

    # mix_x: x_mixed = sum_h pre[:,h]*x[:,h,:], fanned over D/D_SPMD blocks per token tile.
    x_mixed_tail_store = pl.create_tensor([T_TILE, D], dtype=pl.BF16)
    for blk in pl.spmd(token_tiles * (D // D_SPMD), name_hint="mix_x", allow_early_resolve=True):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        valid_rows = pl.min(T_TILE, t_dim - t0)
        pre_tile_t = pl.transpose(pre_val_store[t0 : t0 + T_TILE, 0:HC_PAD], axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[valid_rows, D_TILE])
            x1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[valid_rows, D_TILE])
            x2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[valid_rows, D_TILE])
            x3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[valid_rows, D_TILE])
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(pl.add(y0, y1), y2), y3)
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            if valid_rows == T_TILE:
                x_mixed[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = y_bf16
            else:
                x_mixed_tail_store[0:T_TILE, d0 : d0 + D_TILE] = y_bf16
                y_out = pl.load(
                    x_mixed_tail_store,
                    [0, d0],
                    [T_TILE, D_TILE],
                    valid_shape=[valid_rows, D_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(y_out, [t0, d0], x_mixed)
    return x_mixed


hc_pre = pl.jit.inline(_hc_pre)
hc_pre_fp32_test = pl.jit(_hc_pre)


def _hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)
    tokens = pl.tensor.dim(x, 0)
    wide = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
    flat = pl.reshape(x, [tokens, HC_DIM])
    wide_flat = pl.reshape(wide, [tokens, HC_DIM])
    # A single valid row per task avoids losing a tail-row valid_shape on cast.
    # This is an exact widening, not a change to the Native BF16 math boundary.
    for row in pl.spmd(tokens, name_hint="hc_pre_widen", allow_early_resolve=True):
        for column in pl.pipeline(HC_DIM // RMS_K_TILE, stage=2):
            start = column * RMS_K_TILE
            value = pl.cast(flat[row : row + 1, start : start + RMS_K_TILE], pl.FP32)
            wide_flat[row : row + 1, start : start + RMS_K_TILE] = value
    hc_pre(wide, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed, post, comb


hc_pre_entry = pl.jit.inline(auto_scope=False)(_hc_pre_test)
hc_pre_test = pl.jit(_hc_pre_test)
