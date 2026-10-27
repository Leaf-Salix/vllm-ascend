# SPDX-License-Identifier: Apache-2.0
"""隔离HC模块：保留nalinaly生产helper的算术，单独替换原始输入RMS。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_input_norm_stages import hc_input_norm_rows
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_input_rms_stages import (
    SQUARE_ROWS,
    hc_input_rms_range,
    hc_input_square_range,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_native_projection_stages import (
    hc_linear,
    hc_linear_reduce,
    hc_pack_input,
    hc_pack_input_from_bf16,
    hc_pack_weight,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_stages import (
    MIX_ROWS_DYN,
    PAD_ROWS_DYN,
    PARTIAL_ROWS_DYN,
    T_DYN,
    hc_comb_range,
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
    hc_widen_rms_range,
)


@pl.jit
def hc_pair_arm(
    x: pl.Tensor[[T_DYN, 16384], pl.BF16],
    fn: pl.Tensor[[24, 16384], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[24], pl.FP32],
    base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wide: pl.InOut[pl.Tensor[[T_DYN, 16384], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    partials: pl.InOut[pl.Tensor[[PARTIAL_ROWS_DYN, 32], pl.FP32]],
    squares: pl.InOut[pl.Tensor[[SQUARE_ROWS, 16], pl.FP32]],
    raw: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 32], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T_DYN, 4], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T_DYN, 16], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T_DYN, 4096], pl.BF16]],
    NATIVE_RMS: pl.constexpr,
    NATIVE_LINEAR: pl.constexpr = False,
    NATIVE_FOLD: pl.constexpr = False,
    NATIVE_NORM: pl.constexpr = False,
    DIRECT_PACK: pl.constexpr = False,
):
    widen_tail = pl.create_tensor([8, 16384], dtype=pl.FP32)
    pre = pl.create_tensor([pl.tensor.dim(inv, 0), 8], dtype=pl.FP32)
    post_tail = pl.create_tensor([8, 8], dtype=pl.FP32)
    comb_tail = pl.create_tensor([8, 32], dtype=pl.FP32)
    norm_tail = pl.create_tensor([8, 4096], dtype=pl.BF16)
    for worker in pl.spmd(48, name_hint="nal_hc_widen", allow_early_resolve=True):
        wide, inv = hc_widen_rms_range(x, wide, inv, widen_tail, worker, 48)
    packed_input = pl.create_tensor([pl.tensor.dim(inv, 0), 32768], dtype=pl.BF16)
    packed_weight = pl.create_tensor([32768, 32], dtype=pl.BF16)
    if NATIVE_LINEAR:
        for worker in pl.spmd(48, name_hint="native_order_hc_pack", allow_early_resolve=True):
            if DIRECT_PACK:
                packed_input = hc_pack_input_from_bf16(x, packed_input, worker)
            else:
                packed_input = hc_pack_input(wide, packed_input, worker)
            packed_weight = hc_pack_weight(fn, packed_weight, worker)
    for worker in pl.spmd(24, name_hint="nal_hc_project", allow_early_resolve=True):
        if NATIVE_LINEAR:
            partials = hc_linear(wide, packed_input, packed_weight, partials, worker)
        else:
            partials = hc_project_range(wide, fn, partials, worker, 24)
        if NATIVE_RMS:
            squares = hc_input_square_range(wide, squares, worker)
    for worker in pl.spmd(48, name_hint="nal_hc_reduce", allow_early_resolve=True):
        if NATIVE_LINEAR:
            raw = hc_linear_reduce(partials, raw, worker)
        else:
            raw = hc_reduce_range(partials, raw, worker, 48)
        if NATIVE_RMS:
            inv = hc_input_rms_range(squares, inv, wide, worker)
    for worker in pl.spmd(48, name_hint="nal_hc_gates", allow_early_resolve=True):
        pre, post = hc_gates_range(inv, raw, scale, base, pre, post, post_tail, worker, 48, partials, False)
        comb = hc_comb_range(inv, raw, scale, base_2d, comb, comb_tail, worker, 48, partials, False, NATIVE_FOLD)
        mixed, normed = hc_mix_norm_range(
            wide, pre, norm_w, mixed, normed, norm_tail, worker, 48, NATIVE_FOLD, not NATIVE_NORM
        )
    if NATIVE_NORM:
        rstd = pl.create_tensor([pl.tensor.dim(x, 0), 8], dtype=pl.FP32)
        for worker in pl.spmd(48, name_hint="native_input_norm", allow_early_resolve=True):
            normed, rstd = hc_input_norm_rows(mixed, norm_w, normed, rstd, worker)
    return wide, inv, partials, squares, raw, post, comb, mixed, normed
