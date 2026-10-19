# SPDX-License-Identifier: Apache-2.0
"""Single-SPMD HC-pre/Norm/WKV prefix; not a complete CSA replacement."""

import pypto.language as pl

from .hc_prefix_stages import (
    P16,
    PARTIAL_ROWS,
    hc_comb,
    hc_linear,
    hc_linear_reduce,
    hc_mix,
    hc_pack_input,
    hc_pack_weight,
    hc_pre_post,
    hc_rms,
    hc_square,
    hc_widen,
)
from .norm_wkv_stages import (
    ACC_ROWS,
    T,
    norm_rows,
    wkv_matmul,
    wkv_round,
)
from .rendezvous import GROUPS, handoff


@pl.jit.incore
def prefix_cube(
    wide: pl.Tensor[[T, 16384], pl.FP32],
    square: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    packed_weight: pl.Tensor[[32768, 32], pl.BF16],
    packed_flat: pl.Tensor[[P16, 32768], pl.BF16],
    linear: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32]],
    normalized: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    pl.system.set_ffts(ffts)
    worker = pl.tile.get_block_idx()
    handoff()
    square = hc_square(wide, square, worker)
    handoff()
    handoff()
    linear = hc_linear(wide, packed_flat, packed_weight, linear, worker)
    handoff()
    handoff()
    handoff()
    handoff()
    handoff()
    accum = wkv_matmul(normalized, weight, order, accum, worker)
    handoff()
    handoff()
    return square, linear, accum


@pl.jit.incore
def prefix_vector(
    flat: pl.Tensor[[T, 16384], pl.BF16],
    wide: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    square_read: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    inverse: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    packed_weight: pl.InOut[pl.Tensor[[32768, 32], pl.BF16]],
    packed_flat: pl.InOut[pl.Tensor[[P16, 32768], pl.BF16]],
    linear_read: pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    mixed_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    gamma: pl.Tensor[[4096], pl.BF16],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.InOut[pl.Tensor[[T, 8], pl.FP32]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    wide = hc_widen(flat, wide, worker)
    handoff()
    handoff()
    inverse = hc_rms(square_read, inverse, wide, worker)
    packed_weight = hc_pack_weight(hc_fn, packed_weight, worker)
    packed_flat = hc_pack_input(wide, packed_flat, worker)
    handoff()
    handoff()
    raw = hc_linear_reduce(linear_read, raw, worker)
    handoff()
    pre, post, post_tail = hc_pre_post(raw, inverse, hc_scale, hc_base, pre, post, post_tail, worker)
    comb, comb_tail = hc_comb(raw, inverse, hc_scale, hc_base_2d, comb, comb_tail, False, worker)
    handoff()
    mixed, mixed_tail = hc_mix(wide, pre, mixed, mixed_tail, worker)
    handoff()
    normalized, rstd = norm_rows(mixed, gamma, normalized, rstd, worker)
    handoff()
    handoff()
    output = wkv_round(accum_read, output, worker)
    handoff()
    return (
        wide,
        inverse,
        packed_weight,
        packed_flat,
        raw,
        pre,
        post,
        post_tail,
        comb,
        comb_tail,
        mixed,
        mixed_tail,
        normalized,
        rstd,
        output,
    )


@pl.jit.inline(auto_scope=False)
def prefix_group(
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    gamma: pl.Tensor[[4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    wide: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    square: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    square_read: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    inverse: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    packed_weight: pl.InOut[pl.Tensor[[32768, 32], pl.BF16]],
    packed_flat: pl.InOut[pl.Tensor[[P16, 32768], pl.BF16]],
    linear: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32]],
    linear_read: pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    mixed_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.InOut[pl.Tensor[[T, 8], pl.FP32]],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
    flat: pl.Tensor[[T, 16384], pl.BF16],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
):
    with pl.cluster(name_hint="precision_hc_prefix"):
        square, linear, accum = prefix_cube(
            wide, square, packed_weight, packed_flat, linear, normalized, weight, order, accum, ffts
        )
        (
            wide,
            inverse,
            packed_weight,
            packed_flat,
            raw,
            pre,
            post,
            post_tail,
            comb,
            comb_tail,
            mixed,
            mixed_tail,
            normalized,
            rstd,
            output,
        ) = prefix_vector(
            flat,
            wide,
            square_read,
            inverse,
            hc_fn,
            packed_weight,
            packed_flat,
            linear_read,
            raw,
            hc_scale,
            hc_base,
            hc_base_2d,
            pre,
            post,
            post_tail,
            comb,
            comb_tail,
            mixed,
            mixed_tail,
            gamma,
            normalized,
            rstd,
            accum_read,
            output,
            ffts,
        )
    return mixed, post, comb, normalized, rstd, accum, output


@pl.jit
def precision_hc_prefix(
    x: pl.Tensor[[T, 4, 4096], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    gamma: pl.Tensor[[4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    wide: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    square: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    square_read: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    inverse: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    packed_weight: pl.InOut[pl.Tensor[[32768, 32], pl.BF16]],
    packed_flat: pl.InOut[pl.Tensor[[P16, 32768], pl.BF16]],
    linear: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32]],
    linear_read: pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    mixed_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    rstd: pl.InOut[pl.Tensor[[T, 8], pl.FP32]],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    x.bind_dynamic(0, T)
    wide.bind_dynamic(0, T)
    square.bind_dynamic(0, PARTIAL_ROWS)
    square_read.bind_dynamic(0, PARTIAL_ROWS)
    inverse.bind_dynamic(0, P16)
    packed_flat.bind_dynamic(0, P16)
    linear.bind_dynamic(0, PARTIAL_ROWS)
    linear_read.bind_dynamic(0, PARTIAL_ROWS)
    raw.bind_dynamic(0, P16)
    pre.bind_dynamic(0, P16)
    post.bind_dynamic(0, T)
    comb.bind_dynamic(0, T)
    mixed.bind_dynamic(0, T)
    normalized.bind_dynamic(0, T)
    rstd.bind_dynamic(0, T)
    accum.bind_dynamic(0, ACC_ROWS)
    accum_read.bind_dynamic(0, ACC_ROWS)
    output.bind_dynamic(0, T)
    tokens = pl.tensor.dim(x, 0)
    flat = pl.reshape(x, [tokens, 16384])
    hc_base_2d = pl.reshape(hc_base, [1, 24])
    with pl.spmd(GROUPS, sync_start=True):
        mixed, post, comb, normalized, rstd, accum, output = prefix_group(
            hc_fn,
            hc_scale,
            hc_base,
            gamma,
            weight,
            order,
            wide,
            square,
            square_read,
            inverse,
            packed_weight,
            packed_flat,
            linear,
            linear_read,
            raw,
            pre,
            post,
            post_tail,
            comb,
            comb_tail,
            mixed,
            mixed_tail,
            normalized,
            rstd,
            accum,
            accum_read,
            output,
            ffts,
            flat,
            hc_base_2d,
        )
    return mixed, post, comb, normalized, rstd, accum, output
