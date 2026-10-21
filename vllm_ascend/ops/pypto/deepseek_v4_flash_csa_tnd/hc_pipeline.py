# SPDX-License-Identifier: Apache-2.0
"""独立HC局部流水；信号一次清零、按调用代际推进。"""

import pypto.language as pl

from .hc_rows import row_hc_gates_range, row_hc_mix_norm_range, row_hc_widen_rms_range
from .hc_stages import (
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
)
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
from .runtime_abi import (
    P8,
    P16,
    PARTIAL,
    T,
)

HC_SIGNAL_WORDS = pl.dynamic("CSA_HC_PREFIX_SIGNAL_WORDS")
H_WIDE = 1
H_PROJECT = 2
H_RAW = 3


@pl.jit.incore
def hc_prefix_cube(
    x_fp32: pl.Tensor[[T, 16384], pl.FP32],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    hc_signal: pl.InOut[pl.Tensor[[HC_SIGNAL_WORDS], pl.INT32]],
    HC_GROUPS: pl.constexpr,
    HC_ROW_TILE: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    tokens = pl.tensor.dim(x_fp32, 0)
    (hc_signal, generation) = begin_invocation(hc_signal, worker)
    units = (tokens + 15) // 16 * 4
    for unit in pl.range(worker, units, HC_GROUPS):
        first = unit // 4 * 16
        hc_signal = wait_row_owners(
            hc_signal,
            H_WIDE,
            3 * HC_GROUPS,
            generation,
            first,
            pl.min(16, tokens - first),
            HC_ROW_TILE,
            2 * HC_GROUPS,
            HC_GROUPS,
            1,
        )
    partials = hc_project_range(x_fp32, hc_fn, partials, worker, HC_GROUPS)
    hc_signal = publish_phase(hc_signal, H_PROJECT, worker, 3 * HC_GROUPS, generation)
    return (partials, hc_signal)


@pl.jit.incore
def hc_prefix_vector(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    partials: pl.Tensor[[PARTIAL, 32], pl.FP32],
    x_fp32: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    hc_signal: pl.InOut[pl.Tensor[[HC_SIGNAL_WORDS], pl.INT32]],
    HC_GROUPS: pl.constexpr,
    HC_ROW_TILE: pl.constexpr,
    HC_CONSUMER_REDUCE: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = HC_GROUPS + worker
    tokens = pl.tensor.dim(x, 0)
    (hc_signal, generation) = begin_invocation(hc_signal, core)
    (x_fp32, inv) = row_hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, 2 * HC_GROUPS, HC_ROW_TILE)
    hc_signal = publish_phase(hc_signal, H_WIDE, core, 3 * HC_GROUPS, generation)
    for block in pl.range(worker, (tokens + 15) // 16, 2 * HC_GROUPS):
        hc_signal = wait_row_owners(hc_signal, H_PROJECT, 3 * HC_GROUPS, generation, block * 4, 4, 1, HC_GROUPS, 0, 1)
    raw = hc_reduce_range(partials, raw, worker, 2 * HC_GROUPS)
    hc_signal = publish_phase(hc_signal, H_RAW, core, 3 * HC_GROUPS, generation)
    if tokens < 8:
        for block in pl.range(worker, tokens, 2 * HC_GROUPS):
            hc_signal = wait_row_owners(
                hc_signal, H_RAW, 3 * HC_GROUPS, generation, block, 1, 16, 2 * HC_GROUPS, HC_GROUPS, 1
            )
        (pre, post) = row_hc_gates_range(
            inv, raw, hc_scale, hc_base, pre, post, post_tail, worker, 2 * HC_GROUPS, partials, HC_CONSUMER_REDUCE, 1
        )
        pl.system.bar_all()
        (mixed, normed) = row_hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail, worker, 2 * HC_GROUPS, 1)
    else:
        for block in pl.range(worker, (tokens + 7) // 8, 2 * HC_GROUPS):
            first = block * 8
            hc_signal = wait_row_owners(
                hc_signal,
                H_RAW,
                3 * HC_GROUPS,
                generation,
                first,
                pl.min(8, tokens - first),
                16,
                2 * HC_GROUPS,
                HC_GROUPS,
                1,
            )
        (pre, post) = hc_gates_range(
            inv, raw, hc_scale, hc_base, pre, post, post_tail, worker, 2 * HC_GROUPS, partials, HC_CONSUMER_REDUCE
        )
        pl.system.bar_all()
        (mixed, normed) = hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail, worker, 2 * HC_GROUPS)
    return (x_fp32, inv, widen_tail, raw, pre, post, post_tail, mixed, normed, norm_tail, hc_signal)


@pl.jit.inline(auto_scope=False)
def hc_prefix_chain(
    x_fp32: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    hc_signal: pl.InOut[pl.Tensor[[HC_SIGNAL_WORDS], pl.INT32]],
    HC_GROUPS: pl.constexpr,
    HC_ROW_TILE: pl.constexpr,
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    HC_CONSUMER_REDUCE: pl.constexpr,
):
    with pl.cluster(name_hint="hc_prefix_chain"):
        (partials, hc_signal) = hc_prefix_cube(x_fp32, hc_fn, partials, hc_signal, HC_GROUPS, HC_ROW_TILE)
        (x_fp32, inv, widen_tail, raw, pre, post, post_tail, mixed, normed, norm_tail, hc_signal) = hc_prefix_vector(
            x,
            hc_scale,
            hc_base,
            norm_w,
            partials,
            x_fp32,
            inv,
            widen_tail,
            raw,
            pre,
            post,
            post_tail,
            mixed,
            normed,
            norm_tail,
            hc_signal,
            HC_GROUPS,
            HC_ROW_TILE,
            HC_CONSUMER_REDUCE,
        )
    return (x_fp32, partials, hc_signal, inv, widen_tail, raw, pre, post, post_tail, mixed, normed, norm_tail)
