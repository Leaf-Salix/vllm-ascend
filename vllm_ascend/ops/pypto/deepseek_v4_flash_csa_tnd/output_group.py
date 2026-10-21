# SPDX-License-Identifier: Apache-2.0
"""冻结84751e34的输出组流水，独立于Attention与其它CSA task。

24 AIC/48 AIV每次各推进一次代际；AIC每组写完OA后发布，AIV只等真实owner，
量化通知包含INT8尾零。OB完成后HCpost仍按原八组左折叠规约。
局部MIX使用新分配且首次清零的20阶段signal，不能复用query阶段的代际布局。
跨task依赖由TMR表达；这里只复用输出尾段的核内通知，未合并完整CSA。
"""

import pypto.language as pl

from .output_projection_a_stages import (
    o_projection_a_groups_stage_nz,
    o_projection_a_stage_nz,
    o_projection_group_quant_stage,
    o_projection_groups_quant_stage,
)
from .output_projection_b_stages import o_b_groups_stage, o_b_stage, o_dequant_hc_post_stage
from .qkv_stages import BF16_WEIGHT_LAYOUT
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
from .query_projection_stages import QUANT_WEIGHT_LAYOUT
from .runtime_abi import CORES, GROUPS, O_MM_ROWS, O_SCALE_ROWS, PACK_ROWS, VECTORS, T

OUTPUT_SIGNAL_WORDS = pl.dynamic("CSA_OUTPUT_SIGNAL_WORDS")
OUTPUT_SIGNAL_COUNT = 20 * CORES * 128
OUTPUT_GROUP_PIPELINE = True
PHASE_OA = 1
PHASE_O_QUANT = 2
PHASE_OB = 3
PHASE_OA_GROUP_BASE = 4
PHASE_O_QUANT_GROUP_BASE = 12
OUTPUT_BALANCE_HEAVY_CORES = 16
OUTPUT_BALANCE_VIRTUAL_WORKERS = 32
OUTPUT_BALANCE_LAST_RANK = 31
OUTPUT_BALANCE_COLUMNS_PER_GROUP = 8


@pl.jit.incore
def output_chain_cube(
    positions: pl.Tensor[[T], pl.INT64],
    packed: pl.Tensor[[PACK_ROWS, 4096], pl.BF16],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, BF16_WEIGHT_LAYOUT],
    o_quant: pl.Tensor[[O_MM_ROWS, 8192], pl.INT8],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    o_r: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.FP32]],
    o_partials: pl.InOut[pl.Tensor[[O_MM_ROWS, 32768], pl.INT32]],
    output_signal: pl.InOut[pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
    OUTPUT_WORK_BALANCE: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    tokens = pl.tensor.dim(positions, 0)
    (output_signal, generation) = begin_invocation(output_signal, worker)
    if OUTPUT_GROUP_PIPELINE:
        oa_per_group = (tokens + 127) // 128 * (1024 // OA_COLS)
        for oa_group in pl.range(8):
            oa_first = oa_group * oa_per_group
            oa_owned = worker + (oa_first + GROUPS - 1 - worker) // GROUPS * GROUPS
            if oa_owned < oa_first + oa_per_group:
                o_r = o_projection_a_groups_stage_nz(
                    packed, wo_a, o_r, tokens, worker, GROUPS, OA_COLS, oa_group, oa_group + 1
                )
                output_signal = publish_phase(output_signal, PHASE_OA_GROUP_BASE + oa_group, worker, CORES, generation)
        quant_per_group = (tokens + 31) // 32
        if OUTPUT_WORK_BALANCE:
            for ob_group in pl.range(8):
                ob_worker = worker + OUTPUT_BALANCE_HEAVY_CORES
                if worker >= OUTPUT_BALANCE_HEAVY_CORES:
                    ob_worker = worker - OUTPUT_BALANCE_HEAVY_CORES + ob_group % 2 * OUTPUT_BALANCE_COLUMNS_PER_GROUP
                ob_first = ob_group * OUTPUT_BALANCE_COLUMNS_PER_GROUP
                ob_owned = (
                    ob_worker
                    + (ob_first + OUTPUT_BALANCE_LAST_RANK - ob_worker)
                    // OUTPUT_BALANCE_VIRTUAL_WORKERS
                    * OUTPUT_BALANCE_VIRTUAL_WORKERS
                )
                if ob_owned < ob_first + OUTPUT_BALANCE_COLUMNS_PER_GROUP:
                    output_signal = wait_row_owners(
                        output_signal,
                        PHASE_O_QUANT_GROUP_BASE + ob_group,
                        CORES,
                        generation,
                        ob_group * quant_per_group,
                        quant_per_group,
                        1,
                        VECTORS,
                        GROUPS,
                        1,
                    )
                    o_partials = o_b_groups_stage(
                        o_quant,
                        wo_b,
                        o_partials,
                        tokens,
                        ob_worker,
                        OUTPUT_BALANCE_VIRTUAL_WORKERS,
                        OB_ROWS,
                        ob_group,
                        ob_group + 1,
                    )
        else:
            for ob_group in pl.range(8):
                ob_first = ob_group * 8
                ob_owned = worker + (ob_first + GROUPS - 1 - worker) // GROUPS * GROUPS
                if ob_owned < ob_first + 8:
                    output_signal = wait_row_owners(
                        output_signal,
                        PHASE_O_QUANT_GROUP_BASE + ob_group,
                        CORES,
                        generation,
                        ob_group * quant_per_group,
                        quant_per_group,
                        1,
                        VECTORS,
                        GROUPS,
                        1,
                    )
                    o_partials = o_b_groups_stage(
                        o_quant, wo_b, o_partials, tokens, worker, GROUPS, OB_ROWS, ob_group, ob_group + 1
                    )
    else:
        o_r = o_projection_a_stage_nz(packed, wo_a, o_r, tokens, worker, GROUPS, OA_COLS)
        output_signal = publish_phase(output_signal, PHASE_OA, worker, CORES, generation)
        quant_tasks = 8 * ((tokens + 31) // 32)
        output_signal = wait_row_owners(
            output_signal, PHASE_O_QUANT, CORES, generation, 0, quant_tasks, 1, VECTORS, GROUPS, 1
        )
        o_partials = o_b_stage(o_quant, wo_b, o_partials, tokens, worker, GROUPS, OB_ROWS)
    output_signal = publish_phase(output_signal, PHASE_OB, worker, CORES, generation)
    return (o_r, o_partials, output_signal)


@pl.jit.incore
def output_chain_vector(
    positions: pl.Tensor[[T], pl.INT64],
    x: pl.Tensor[[T, 16384], pl.BF16],
    post: pl.Tensor[[T, 4], pl.FP32],
    comb: pl.Tensor[[T, 16], pl.FP32],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    o_r: pl.Tensor[[O_MM_ROWS, 8192], pl.FP32],
    o_partials: pl.Tensor[[O_MM_ROWS, 32768], pl.INT32],
    o_quant: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.INT8]],
    o_scales: pl.InOut[pl.Tensor[[8, O_SCALE_ROWS], pl.FP32]],
    output: pl.InOut[pl.Tensor[[T, 16384], pl.BF16]],
    output_signal: pl.InOut[pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = GROUPS + worker
    tokens = pl.tensor.dim(positions, 0)
    (output_signal, generation) = begin_invocation(output_signal, core)
    if OUTPUT_GROUP_PIPELINE:
        oa_per_group = (tokens + 127) // 128 * (1024 // OA_COLS)
        quant_per_group = (tokens + 31) // 32
        for quant_group in pl.range(8):
            quant_first = quant_group * quant_per_group
            quant_owned = worker + (quant_first + VECTORS - 1 - worker) // VECTORS * VECTORS
            if quant_owned < quant_first + quant_per_group:
                output_signal = wait_row_owners(
                    output_signal,
                    PHASE_OA_GROUP_BASE + quant_group,
                    CORES,
                    generation,
                    quant_group * oa_per_group,
                    oa_per_group,
                    1,
                    GROUPS,
                    0,
                    1,
                )
                (o_quant, o_scales) = o_projection_groups_quant_stage(
                    o_r, o_quant, o_scales, tokens, worker, VECTORS, OB_ROWS, quant_group, quant_group + 1
                )
                output_signal = publish_phase(
                    output_signal, PHASE_O_QUANT_GROUP_BASE + quant_group, core, CORES, generation
                )
    else:
        oa_tasks = 8 * ((tokens + 127) // 128) * (1024 // OA_COLS)
        output_signal = wait_row_owners(output_signal, PHASE_OA, CORES, generation, 0, oa_tasks, 1, GROUPS, 0, 1)
        (o_quant, o_scales) = o_projection_group_quant_stage(o_r, o_quant, o_scales, tokens, worker, VECTORS, OB_ROWS)
        output_signal = publish_phase(output_signal, PHASE_O_QUANT, core, CORES, generation)
    output_signal = wait_row_owners(output_signal, PHASE_OB, CORES, generation, 0, 64, 1, GROUPS, 0, 1)
    output = o_dequant_hc_post_stage(o_partials, o_scales, wo_b_scale, x, post, comb, output, worker, VECTORS)
    return (o_quant, o_scales, output, output_signal)


@pl.jit.inline(auto_scope=False)
def output_group_chain(
    positions: pl.Tensor[[T], pl.INT64],
    packed: pl.Tensor[[PACK_ROWS, 4096], pl.BF16],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, BF16_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    x: pl.Tensor[[T, 16384], pl.BF16],
    post: pl.Tensor[[T, 4], pl.FP32],
    comb: pl.Tensor[[T, 16], pl.FP32],
    o_r: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.FP32]],
    o_quant: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.INT8]],
    o_scales: pl.InOut[pl.Tensor[[8, O_SCALE_ROWS], pl.FP32]],
    o_partials: pl.InOut[pl.Tensor[[O_MM_ROWS, 32768], pl.INT32]],
    output: pl.InOut[pl.Tensor[[T, 16384], pl.BF16]],
    output_signal: pl.InOut[pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
    OUTPUT_WORK_BALANCE: pl.constexpr,
):
    with pl.cluster(name_hint="output_group_chain"):
        (o_r, o_partials, output_signal) = output_chain_cube(
            positions,
            packed,
            wo_a,
            o_quant,
            wo_b,
            o_r,
            o_partials,
            output_signal,
            OA_COLS,
            OB_ROWS,
            OUTPUT_WORK_BALANCE,
        )
        (o_quant, o_scales, output, output_signal) = output_chain_vector(
            positions,
            x,
            post,
            comb,
            wo_b_scale,
            o_r,
            o_partials,
            o_quant,
            o_scales,
            output,
            output_signal,
            OA_COLS,
            OB_ROWS,
        )
    return (o_r, o_quant, o_scales, o_partials, output, output_signal)
