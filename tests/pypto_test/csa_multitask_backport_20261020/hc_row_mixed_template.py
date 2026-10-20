# SPDX-License-Identifier: Apache-2.0
"""将冻结单SPMD的HC行依赖适配为多task前段局部MIX。

数学取自13259f92的hc_stages，CONSUMER_REDUCE仍关闭。24个AIC/48个
AIV按来源实际owner等待输入、四个K分片和RAW行；gate与norm归同一AIV。
Sinkhorn留作独立task，不让只有最终HCpost需要的comb阻挡QKV入口。
每次调用显式清零独立信号，清零时间计入完整CSA，不复用未初始化代际。
"""

import pypto.language as pl

from .hc_row_phase_sync import publish_phase, wait_row_owners
from .hc_stages import (
    HC_DIM,
    HC_MULT,
    HC_PAD,
    MIX_HC,
    MIX_PAD,
    MIX_ROWS_DYN,
    PAD_ROWS_DYN,
    PARTIAL_ROWS_DYN,
    T_DYN,
    D,
    hc_comb_range,
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
    hc_widen_rms_range,
)

HC_CUBES = 24
HC_VECTORS = 2 * HC_CUBES
HC_CORES = 3 * HC_CUBES
HC_INPUT_PHASE = 0
HC_PARTIAL_PHASE = 1
HC_RAW_PHASE = 2
HC_GATES_PHASE = 3
HC_SIGNAL_WORDS = 4 * HC_CORES * 128
HC_INIT_WORKERS = 16
HC_INIT_WORDS = 2048


@pl.jit.incore
def hc_row_cube(
    x_fp32: pl.Tensor[[T_DYN, HC_DIM], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    partials: pl.InOut[pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[HC_SIGNAL_WORDS], pl.INT32]],
):
    partials_gm = partials
    signal_gm = signal
    worker = pl.tile.get_block_idx()
    tokens = pl.tensor.dim(x_fp32, 0)
    generation = pl.cast(1, pl.INT32)
    hc_tasks = ((tokens + 15) // 16) * 4
    hc_waves = (hc_tasks + HC_CUBES - 1 - worker) // HC_CUBES
    # 来源HC_ROW_DEPENDENCIES：每个M16只等待两个实际M8输入owner。
    for wave in pl.range(pl.min(hc_waves, HC_VECTORS // (2 * (HC_CUBES // 4)))):
        first = ((worker + wave * HC_CUBES) // 4) * 16
        signal = wait_row_owners(
            signal,
            HC_INPUT_PHASE,
            HC_CORES,
            generation,
            first,
            pl.min(16, tokens - first),
            8,
            HC_VECTORS,
            HC_CUBES,
            1,
        )
    partials = hc_project_range(x_fp32, hc_fn, partials, worker, HC_CUBES)
    signal = publish_phase(signal, HC_PARTIAL_PHASE, worker, HC_CORES, generation)
    return partials_gm, signal_gm


@pl.jit.incore
def hc_row_vector(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, HC_DIM], pl.FP32]],
    partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    raw: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32]],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    pre: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, HC_PAD], pl.FP32]],
    norm_w: pl.Tensor[[D], pl.BF16],
    mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, D], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, D], pl.BF16]],
    signal: pl.InOut[pl.Tensor[[HC_SIGNAL_WORDS], pl.INT32]],
):
    pl.func_attr({"split_aiv": True})
    x_fp32_gm = x_fp32
    inv_gm = inv
    widen_tail_gm = widen_tail
    raw_gm = raw
    pre_gm = pre
    post_gm = post
    post_tail_gm = post_tail
    mixed_gm = mixed
    normed_gm = normed
    norm_tail_gm = norm_tail
    signal_gm = signal
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = HC_CUBES + worker
    tokens = pl.tensor.dim(x, 0)
    generation = pl.cast(1, pl.INT32)
    x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, HC_VECTORS)
    signal = publish_phase(signal, HC_INPUT_PHASE, core, HC_CORES, generation)
    if worker < (tokens + 15) // 16:
        signal = wait_row_owners(
            signal,
            HC_PARTIAL_PHASE,
            HC_CORES,
            generation,
            worker * 4,
            4,
            1,
            HC_CUBES,
            0,
            1,
        )
    raw = hc_reduce_range(partials, raw, worker, HC_VECTORS)
    if worker < (tokens + 15) // 16:
        signal = publish_phase(signal, HC_RAW_PHASE, core, HC_CORES, generation)
    gate_tiles = (tokens + 7) // 8
    if worker < gate_tiles:
        signal = wait_row_owners(
            signal,
            HC_RAW_PHASE,
            HC_CORES,
            generation,
            worker // 2,
            1,
            1,
            HC_VECTORS,
            HC_CUBES,
            1,
        )
        if worker + HC_VECTORS < gate_tiles:
            signal = wait_row_owners(
                signal,
                HC_RAW_PHASE,
                HC_CORES,
                generation,
                worker // 2 + HC_VECTORS // 2,
                1,
                1,
                HC_VECTORS,
                HC_CUBES,
                1,
            )
    pre, post = hc_gates_range(
        inv,
        raw,
        hc_scale,
        hc_base,
        pre,
        post,
        post_tail,
        worker,
        HC_VECTORS,
        partials,
        False,
    )
    if worker < gate_tiles:
        signal = publish_phase(signal, HC_GATES_PHASE, core, HC_CORES, generation)
    mixed, normed = hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail, worker, HC_VECTORS)
    return (
        x_fp32_gm,
        inv_gm,
        widen_tail_gm,
        raw_gm,
        pre_gm,
        post_gm,
        post_tail_gm,
        mixed_gm,
        normed_gm,
        norm_tail_gm,
        signal_gm,
    )


@pl.jit.inline(auto_scope=False)
def hc_row_cluster(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, HC_DIM], pl.FP32]],
    partials: pl.InOut[pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32]],
    raw: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, HC_PAD], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, HC_PAD], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, D], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, D], pl.BF16]],
    signal: pl.InOut[pl.Tensor[[HC_SIGNAL_WORDS], pl.INT32]],
):
    with pl.cluster(name_hint="hc_row_pipeline"):
        partials, signal = hc_row_cube(x_fp32, hc_fn, partials, signal)
        (
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
            signal,
        ) = hc_row_vector(
            x,
            x_fp32,
            inv,
            widen_tail,
            partials,
            raw,
            hc_scale,
            hc_base,
            pre,
            post,
            post_tail,
            norm_w,
            mixed,
            normed,
            norm_tail,
            signal,
        )
    return post, normed, inv, raw, partials


@pl.jit.inline(auto_scope=False)
def hc_row_prefix(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    normed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    tokens = pl.tensor.dim(x, 0)
    padded16 = (tokens + 15) // 16 * 16
    padded8 = (tokens + 7) // 8 * 8
    x_flat = pl.reshape(x, [tokens, HC_DIM])
    x_fp32 = pl.create_tensor([tokens, HC_DIM], dtype=pl.FP32)
    inv = pl.create_tensor([padded16, 1], dtype=pl.FP32)
    widen_tail = pl.create_tensor([8, HC_DIM], dtype=pl.FP32)
    partials = pl.create_tensor([4 * padded16, MIX_PAD], dtype=pl.FP32)
    raw = pl.create_tensor([padded16, MIX_PAD], dtype=pl.FP32)
    pre = pl.create_tensor([padded16, HC_PAD], dtype=pl.FP32)
    post_tail = pl.create_tensor([8, HC_PAD], dtype=pl.FP32)
    mixed = pl.create_tensor([padded8, D], dtype=pl.BF16)
    norm_tail = pl.create_tensor([8, D], dtype=pl.BF16)
    signal = pl.create_tensor([HC_SIGNAL_WORDS], dtype=pl.INT32)
    signal_matrix = pl.reshape(signal, [1, HC_SIGNAL_WORDS])
    with pl.spmd(HC_INIT_WORKERS, name_hint="hc_row_flags_init", allow_early_resolve=True) as init_tid:
        zero = pl.tile.full([1, HC_INIT_WORDS], dtype=pl.INT32, value=0)
        for offset in pl.range(
            pl.tile.get_block_idx() * HC_INIT_WORDS, HC_SIGNAL_WORDS, HC_INIT_WORKERS * HC_INIT_WORDS
        ):
            live = pl.min(HC_INIT_WORDS, HC_SIGNAL_WORDS - offset)
            pl.store(pl.set_validshape(zero, 1, live), [0, offset], signal_matrix)
    with pl.spmd(HC_CUBES, deps=[init_tid], sync_start=True, allow_early_resolve=True) as prefix_tid:
        _post_result, _normed_result, _inv_result, _raw_result, _partials_result = hc_row_cluster(
            x_flat,
            hc_fn,
            hc_scale,
            hc_base,
            norm_w,
            x_fp32,
            inv,
            widen_tail,
            partials,
            raw,
            pre,
            post,
            post_tail,
            mixed,
            normed,
            norm_tail,
            signal,
        )
    return post, normed, inv, raw, partials, prefix_tid


@pl.jit.inline(auto_scope=False)
def hc_comb_after_prefix(
    inv: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    raw: pl.Tensor[[PAD_ROWS_DYN, MIX_PAD], pl.FP32],
    partials: pl.Tensor[[PARTIAL_ROWS_DYN, MIX_PAD], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
    prefix_tid: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(comb, 0)
    base_matrix = pl.reshape(hc_base, [1, MIX_HC])
    comb_tail = pl.create_tensor([8, HC_PAD * HC_MULT], dtype=pl.FP32)
    dependencies = pl.array.create(1, pl.TASK_ID)
    dependencies[0] = prefix_tid
    with pl.spmd(
        pl.min((tokens + 7) // 8, HC_VECTORS),
        name_hint="comb_sinkhorn",
        deps=[dependencies[0]],
        allow_early_resolve=True,
    ):
        comb = hc_comb_range(
            inv,
            raw,
            hc_scale,
            base_matrix,
            comb,
            comb_tail,
            pl.tile.get_block_idx(),
            HC_VECTORS,
            partials,
            False,
        )
    return comb
