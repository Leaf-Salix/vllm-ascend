# SPDX-License-Identifier: Apache-2.0
"""私有候选：以冻结单SPMD的组通知流水承载多task的输出尾段。

数学helper直接取自13259f92；整体CSA仍由原有多task执行。只有O_A、
按组量化、O_B和HCpost放在同一个24组MIX任务里。每个AIC先完成本核全部
O_A，再等待量化并做本核O_B，避免占核等待尚未执行的自身生产者。

生命周期采用每调用独立的运行时arena缓冲，在CSA早期显式清零全部信号。
尾段task依赖该清零task及packed producer；不读取未初始化的代际，不把
外部persistent workspace或capture外清零偷偷排除在完整CSA计时之外。
所有phase在本次调用只发布一次，等待值固定为1，退出后不跨调用保留标记。
"""

import pypto.language as pl

from .decode_o_proj import (
    HC_MULT,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    PROJ_A_LARGE_N_TILE,
    PROJ_A_MM_N_TILE,
    PROJ_B_MEDIUM_T_TILE,
    PROJ_B_MM_T_TILE,
    PROJ_B_SMALL_T_TILE,
    T_DYN,
    T_PAD,
    D,
)
from .nz_mode import QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT
from .output_projection_a_stages import (
    o_projection_a_groups_stage_nz,
    o_projection_groups_quant_stage,
)
from .output_projection_b_stages import o_b_groups_stage, o_dequant_hc_post_stage
from .phase_sync import publish_phase, wait_row_owners

OUTPUT_CUBES = 24
OUTPUT_VECTORS = 2 * OUTPUT_CUBES
OUTPUT_PARTICIPANTS = 3 * OUTPUT_CUBES
OUTPUT_OA_PHASE = 0
OUTPUT_QUANT_PHASE = O_GROUPS
OUTPUT_OB_PHASE = 2 * O_GROUPS
OUTPUT_PHASE_COUNT = OUTPUT_OB_PHASE + 1
OUTPUT_SIGNAL_SLOT_WORDS = 128
OUTPUT_SIGNAL_WORDS = OUTPUT_PHASE_COUNT * OUTPUT_PARTICIPANTS * OUTPUT_SIGNAL_SLOT_WORDS
OUTPUT_INIT_WORKERS = 16
OUTPUT_INIT_WORDS = 2048

# 私有实验仅验证CANN9.2/NZ2的Native实际NZ权重；不改变共享mode选择。
assert WO_A_WEIGHT_LAYOUT == pl.NZ and QUANT_WEIGHT_LAYOUT == pl.NZ


@pl.jit.inline(auto_scope=False)
def output_flags_init(flags: pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]):
    flags_matrix = pl.reshape(flags, [1, OUTPUT_SIGNAL_WORDS])
    with pl.spmd(OUTPUT_INIT_WORKERS, name_hint="output_flags_init", allow_early_resolve=True) as init_tid:
        zero = pl.tile.full([1, OUTPUT_INIT_WORDS], dtype=pl.INT32, value=0)
        for offset in pl.range(
            pl.tile.get_block_idx() * OUTPUT_INIT_WORDS,
            OUTPUT_SIGNAL_WORDS,
            OUTPUT_INIT_WORKERS * OUTPUT_INIT_WORDS,
        ):
            live = pl.min(OUTPUT_INIT_WORDS, OUTPUT_SIGNAL_WORDS - offset)
            pl.store(pl.set_validshape(zero, 1, live), [0, offset], flags_matrix)
    return flags, init_tid


@pl.jit.incore
def output_cube(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, pl.NZ],
    o_r: pl.InOut[pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.FP32]],
    o_quant: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8],
    partials: pl.InOut[pl.Tensor[[T_PAD, O_GROUPS * D], pl.INT32]],
    flags: pl.InOut[pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
):
    # 对应来源cube_schedule的wo_a/wo_b参数缓存属性，索引按本函数签名重映射。
    pl.func_attr({"cache_policy": [(1, 1), (2, 1)]})
    o_r_gm = o_r
    partials_gm = partials
    flags_gm = flags
    worker = pl.tile.get_block_idx()
    generation = pl.cast(1, pl.INT32)
    oa_per_group = ((tokens + 127) // 128) * (O_LORA // OA_COLS)
    for group in pl.range(O_GROUPS):
        first = group * oa_per_group
        owned = worker + ((first + OUTPUT_CUBES - 1 - worker) // OUTPUT_CUBES) * OUTPUT_CUBES
        if owned < first + oa_per_group:
            o_r = o_projection_a_groups_stage_nz(
                packed, wo_a, o_r, tokens, worker, OUTPUT_CUBES, OA_COLS, group, group + 1
            )
            flags = publish_phase(flags, OUTPUT_OA_PHASE + group, worker, OUTPUT_PARTICIPANTS, generation)
    quant_per_group = (tokens + 31) // 32
    for group in pl.range(O_GROUPS):
        first = group * (D // 512)
        owned = worker + ((first + OUTPUT_CUBES - 1 - worker) // OUTPUT_CUBES) * OUTPUT_CUBES
        if owned < first + D // 512:
            flags = wait_row_owners(
                flags,
                OUTPUT_QUANT_PHASE + group,
                OUTPUT_PARTICIPANTS,
                generation,
                group * quant_per_group,
                quant_per_group,
                1,
                OUTPUT_VECTORS,
                OUTPUT_CUBES,
                1,
            )
            partials = o_b_groups_stage(
                o_quant, wo_b, partials, tokens, worker, OUTPUT_CUBES, OB_ROWS, group, group + 1
            )
    flags = publish_phase(flags, OUTPUT_OB_PHASE, worker, OUTPUT_PARTICIPANTS, generation)
    return o_r_gm, partials_gm, flags_gm


@pl.jit.incore
def output_vector(
    o_r: pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.FP32],
    o_quant: pl.InOut[pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8]],
    scales: pl.InOut[pl.Tensor[[O_GROUPS, T_PAD], pl.FP32]],
    partials: pl.Tensor[[T_PAD, O_GROUPS * D], pl.INT32],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    out: pl.InOut[pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16]],
    flags: pl.InOut[pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
):
    # 必须与来源vector_schedule相同；读取subblock_idx本身不会启动第二条AIV。
    pl.func_attr({"split_aiv": True})
    quant_gm = o_quant
    scales_gm = scales
    out_gm = out
    flags_gm = flags
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = OUTPUT_CUBES + worker
    generation = pl.cast(1, pl.INT32)
    oa_per_group = ((tokens + 127) // 128) * (O_LORA // OA_COLS)
    quant_per_group = (tokens + 31) // 32
    for group in pl.range(O_GROUPS):
        first = group * quant_per_group
        owned = worker + ((first + OUTPUT_VECTORS - 1 - worker) // OUTPUT_VECTORS) * OUTPUT_VECTORS
        if owned < first + quant_per_group:
            flags = wait_row_owners(
                flags,
                OUTPUT_OA_PHASE + group,
                OUTPUT_PARTICIPANTS,
                generation,
                group * oa_per_group,
                oa_per_group,
                1,
                OUTPUT_CUBES,
                0,
                1,
            )
            o_quant, scales = o_projection_groups_quant_stage(
                o_r, o_quant, scales, tokens, worker, OUTPUT_VECTORS, OB_ROWS, group, group + 1
            )
            flags = publish_phase(flags, OUTPUT_QUANT_PHASE + group, core, OUTPUT_PARTICIPANTS, generation)
    flags = wait_row_owners(
        flags,
        OUTPUT_OB_PHASE,
        OUTPUT_PARTICIPANTS,
        generation,
        0,
        O_GROUPS * (D // 512),
        1,
        OUTPUT_CUBES,
        0,
        1,
    )
    out = o_dequant_hc_post_stage(partials, scales, wo_b_scale, residual, post, comb, out, worker, OUTPUT_VECTORS)
    return quant_gm, scales_gm, out_gm, flags_gm


@pl.jit.inline(auto_scope=False)
def output_group_cluster(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, pl.NZ],
    o_r: pl.InOut[pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.FP32]],
    o_quant: pl.InOut[pl.Tensor[[T_PAD, O_GROUPS * O_LORA], pl.INT8]],
    scales: pl.InOut[pl.Tensor[[O_GROUPS, T_PAD], pl.FP32]],
    partials: pl.InOut[pl.Tensor[[T_PAD, O_GROUPS * D], pl.INT32]],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    out: pl.InOut[pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16]],
    flags: pl.InOut[pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
):
    with pl.cluster(name_hint="csa_output_groups"):
        o_r, partials, flags = output_cube(packed, wo_a, wo_b, o_r, o_quant, partials, flags, tokens, OA_COLS, OB_ROWS)
        o_quant, scales, out, flags = output_vector(
            o_r,
            o_quant,
            scales,
            partials,
            wo_b_scale,
            residual,
            post,
            comb,
            out,
            flags,
            tokens,
            OA_COLS,
            OB_ROWS,
        )
    return out


@pl.jit.inline(auto_scope=False)
def output_group_tiled(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    out: pl.InOut[pl.Tensor[[T_DYN, HC_MULT * D], pl.BF16]],
    flags: pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32],
    heads_dep: pl.Scalar[pl.TASK_ID],
    init_dep: pl.Scalar[pl.TASK_ID],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
):
    tokens = pl.tensor.dim(out, 0)
    o_r = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_quant = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    scales = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    partials = pl.create_tensor([T_PAD, O_GROUPS * D], dtype=pl.INT32)
    dependencies = pl.array.create(2, pl.TASK_ID)
    dependencies[0] = heads_dep
    dependencies[1] = init_dep
    with pl.spmd(
        OUTPUT_CUBES,
        deps=[dependencies[0], dependencies[1]],
        sync_start=True,
        allow_early_resolve=True,
    ):
        out = output_group_cluster(
            packed,
            wo_a,
            wo_b,
            o_r,
            o_quant,
            scales,
            partials,
            wo_b_scale,
            residual,
            post,
            comb,
            out,
            flags,
            tokens,
            OA_COLS,
            OB_ROWS,
        )
    return out


@pl.jit.inline(auto_scope=False)
def decode_output_groups(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, pl.NZ],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
    heads_dep: pl.Scalar[pl.TASK_ID],
    flags: pl.Tensor[[OUTPUT_SIGNAL_WORDS], pl.INT32],
    init_dep: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(out, 0)
    # 在策略条件之前固定二维GM描述符，避免把分支内reshape带入MIX的循环phi。
    residual_flat = pl.reshape(residual, [tokens, HC_MULT * D])
    out_flat = pl.reshape(out, [tokens, HC_MULT * D])
    if tokens <= PROJ_B_SMALL_T_TILE:
        out_flat = output_group_tiled(
            packed,
            wo_a,
            wo_b,
            wo_b_scale,
            residual_flat,
            post,
            comb,
            out_flat,
            flags,
            heads_dep,
            init_dep,
            PROJ_A_MM_N_TILE,
            PROJ_B_SMALL_T_TILE,
        )
    elif tokens <= PROJ_B_MEDIUM_T_TILE:
        out_flat = output_group_tiled(
            packed,
            wo_a,
            wo_b,
            wo_b_scale,
            residual_flat,
            post,
            comb,
            out_flat,
            flags,
            heads_dep,
            init_dep,
            PROJ_A_MM_N_TILE,
            PROJ_B_MEDIUM_T_TILE,
        )
    else:
        out_flat = output_group_tiled(
            packed,
            wo_a,
            wo_b,
            wo_b_scale,
            residual_flat,
            post,
            comb,
            out_flat,
            flags,
            heads_dep,
            init_dep,
            PROJ_A_LARGE_N_TILE,
            PROJ_B_MM_T_TILE,
        )
    return out
