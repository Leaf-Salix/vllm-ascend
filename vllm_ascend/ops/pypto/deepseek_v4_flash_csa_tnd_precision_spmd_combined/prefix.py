# SPDX-License-Identifier: Apache-2.0
"""Single-SPMD HC/Norm/WKV/Indexer-weight prefix with a 32-tensor MIX ABI."""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_indexer_stages import (
    VEC_ROWS,
    indexer_projection_round,
    packed_hc_comb,
    packed_hc_linear_reduce,
    packed_hc_mix,
    packed_hc_pre_post,
    packed_hc_rms,
    packed_norm_rows,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_stages import (
    P16,
    PARTIAL_ROWS,
    hc_pack_input,
    hc_pack_weight,
    hc_widen,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.norm_wkv_stages import (
    ACC_ROWS,
    T,
    wkv_round,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.rendezvous import GROUPS, handoff
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd_bf16layout.input_layout import hc_pack_input_bf16

from .prefix_cube import packed_prefix_indexer_cube


@pl.jit.incore
def prepacked_prefix_indexer_vector(
    flat: pl.Tensor[[T, 16384], pl.BF16],
    wide: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    square_read: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    packed_flat: pl.InOut[pl.Tensor[[P16, 32768], pl.BF16]],
    linear_read: pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    mixed_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    gamma: pl.Tensor[[4096], pl.BF16],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
    idx_accum_read: pl.Tensor[[P16, 64], pl.FP32],
    idx_raw: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    idx_scaled: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    idx_output: pl.InOut[pl.Tensor[[T, 64], pl.FP16]],
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    wide = hc_widen(flat, wide, worker)
    handoff()
    # Independent input packing overlaps AIC hc_square, before its completion wait.
    if pl.tensor.dim(flat, 0) >= 64:
        packed_flat = hc_pack_input_bf16(wide, packed_flat, worker)
    else:
        packed_flat = hc_pack_input(wide, packed_flat, worker)
    handoff()
    vec_ws = packed_hc_rms(square_read, vec_ws, wide, worker)
    handoff()
    vec_ws = packed_hc_linear_reduce(linear_read, vec_ws, worker)
    handoff()
    vec_ws, post = packed_hc_pre_post(vec_ws, hc_scale, hc_base, post, worker)
    handoff()
    mixed, mixed_tail = packed_hc_mix(wide, vec_ws, mixed, mixed_tail, worker)
    handoff()
    normalized, vec_ws = packed_norm_rows(mixed, gamma, normalized, vec_ws, worker)
    handoff()
    vec_ws, comb = packed_hc_comb(vec_ws, hc_scale, hc_base_2d, comb, False, worker)
    handoff()
    output = wkv_round(accum_read, output, worker)
    idx_raw, idx_scaled, idx_output = indexer_projection_round(idx_accum_read, idx_raw, idx_scaled, idx_output, worker)
    handoff()
    return (
        wide,
        vec_ws,
        packed_flat,
        post,
        comb,
        mixed,
        mixed_tail,
        normalized,
        output,
        idx_raw,
        idx_scaled,
        idx_output,
    )


@pl.jit.inline(auto_scope=False)
def prepacked_prefix_indexer_group(
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    gamma: pl.Tensor[[4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    wide: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    square: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    square_read: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    packed_weight: pl.Tensor[[32768, 32], pl.BF16],
    packed_flat: pl.InOut[pl.Tensor[[P16, 32768], pl.BF16]],
    linear: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32]],
    linear_read: pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    mixed_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
    flat: pl.Tensor[[T, 16384], pl.BF16],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    idx_weight: pl.Tensor[[64, 4096], pl.BF16],
    idx_order: pl.Tensor[[16, 4, 16], pl.INT32],
    idx_accum: pl.InOut[pl.Tensor[[P16, 64], pl.FP32]],
    idx_accum_read: pl.Tensor[[P16, 64], pl.FP32],
    idx_raw: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    idx_scaled: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    idx_output: pl.InOut[pl.Tensor[[T, 64], pl.FP16]],
):
    with pl.cluster(name_hint="precision_hc_prefix"):
        square, linear, accum, idx_accum = packed_prefix_indexer_cube(
            wide,
            square,
            packed_weight,
            packed_flat,
            linear,
            normalized,
            weight,
            order,
            accum,
            ffts,
            idx_weight,
            idx_order,
            idx_accum,
        )
        (
            wide,
            vec_ws,
            packed_flat,
            post,
            comb,
            mixed,
            mixed_tail,
            normalized,
            output,
            idx_raw,
            idx_scaled,
            idx_output,
        ) = prepacked_prefix_indexer_vector(
            flat,
            wide,
            square_read,
            vec_ws,
            packed_flat,
            linear_read,
            hc_scale,
            hc_base,
            hc_base_2d,
            post,
            comb,
            mixed,
            mixed_tail,
            gamma,
            normalized,
            accum_read,
            output,
            ffts,
            idx_accum_read,
            idx_raw,
            idx_scaled,
            idx_output,
        )
    return (mixed, post, comb, normalized, vec_ws, accum, output, idx_raw, idx_scaled, idx_output)


@pl.jit
def prepacked_precision_hc_prefix_indexer(
    x: pl.Tensor[[T, 4, 4096], pl.BF16],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    gamma: pl.Tensor[[4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    wide: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    square: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    square_read: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
    packed_weight: pl.Tensor[[32768, 32], pl.BF16],
    packed_flat: pl.InOut[pl.Tensor[[P16, 32768], pl.BF16]],
    linear: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32]],
    linear_read: pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    mixed_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    normalized: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
    idx_weight: pl.Tensor[[64, 4096], pl.BF16],
    idx_order: pl.Tensor[[16, 4, 16], pl.INT32],
    idx_accum: pl.InOut[pl.Tensor[[P16, 64], pl.FP32]],
    idx_accum_read: pl.Tensor[[P16, 64], pl.FP32],
    idx_raw: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    idx_scaled: pl.InOut[pl.Tensor[[T, 64], pl.BF16]],
    idx_output: pl.InOut[pl.Tensor[[T, 64], pl.FP16]],
):
    vec_ws.bind_dynamic(0, VEC_ROWS)
    idx_output.bind_dynamic(0, T)
    idx_scaled.bind_dynamic(0, T)
    idx_raw.bind_dynamic(0, T)
    idx_accum_read.bind_dynamic(0, P16)
    idx_accum.bind_dynamic(0, P16)
    x.bind_dynamic(0, T)
    wide.bind_dynamic(0, T)
    square.bind_dynamic(0, PARTIAL_ROWS)
    square_read.bind_dynamic(0, PARTIAL_ROWS)
    packed_flat.bind_dynamic(0, P16)
    linear.bind_dynamic(0, PARTIAL_ROWS)
    linear_read.bind_dynamic(0, PARTIAL_ROWS)
    post.bind_dynamic(0, T)
    comb.bind_dynamic(0, T)
    mixed.bind_dynamic(0, T)
    normalized.bind_dynamic(0, T)
    accum.bind_dynamic(0, ACC_ROWS)
    accum_read.bind_dynamic(0, ACC_ROWS)
    output.bind_dynamic(0, T)
    tokens = pl.tensor.dim(x, 0)
    flat = pl.reshape(x, [tokens, 16384])
    hc_base_2d = pl.reshape(hc_base, [1, 24])
    with pl.spmd(GROUPS, sync_start=True):
        mixed, post, comb, normalized, vec_ws, accum, output, idx_raw, idx_scaled, idx_output = (
            prepacked_prefix_indexer_group(
                hc_scale,
                hc_base,
                gamma,
                weight,
                order,
                wide,
                square,
                square_read,
                vec_ws,
                packed_weight,
                packed_flat,
                linear,
                linear_read,
                post,
                comb,
                mixed,
                mixed_tail,
                normalized,
                accum,
                accum_read,
                output,
                ffts,
                flat,
                hc_base_2d,
                idx_weight,
                idx_order,
                idx_accum,
                idx_accum_read,
                idx_raw,
                idx_scaled,
                idx_output,
            )
        )
    return (mixed, post, comb, normalized, vec_ws, accum, output, idx_raw, idx_scaled, idx_output)


@pl.jit.incore
def _prepack_fixed_hc_weight(
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    packed: pl.InOut[pl.Tensor[[32768, 32], pl.BF16]],
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    packed = hc_pack_weight(hc_fn, packed, worker)
    return packed


@pl.jit
def prepack_fixed_hc_weight(
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    packed: pl.InOut[pl.Tensor[[32768, 32], pl.BF16]],
):
    # PyPTO parses one context per statement; keep the two DSL scopes nested.
    # fmt: off
    with pl.spmd(GROUPS):  # noqa: SIM117
        with pl.cluster(name_hint="fixed_hc_weight"):
            packed = _prepack_fixed_hc_weight(hc_fn, packed)
    # fmt: on
    return packed
