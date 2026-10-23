# SPDX-License-Identifier: Apache-2.0
"""沿BSH-MPMD按8个输出组提交O_A→量化→O_B，AICPU管理全部依赖。

数学复用TND已验证的投影和量化helper。每组只写自身的列/scale行，manual_scope
内使用明确的TaskId依赖，避免全Tensor保守重叠把互不相交的组串行化。
HCpost等待全部O_B，并通过自动Tensor依赖等待HC pre的post/comb。
没有GM进度变量、核内生产者等待或常驻阶段循环。
"""

import pypto.language as pl

from .output_projection_a_stages import o_projection_a_groups_stage_nz, o_projection_groups_quant_stage
from .output_projection_b_stages import o_b_group_task, o_dequant_hc_post_stage
from .query_projection_stages import QUANT_WEIGHT_LAYOUT
from .runtime_abi import O_MM_ROWS, O_SCALE_ROWS, PACK_ROWS, VECTORS, T

OUTPUT_GROUPS = 8
OA_WORKERS_PER_GROUP = 8
QUANT_WORKERS_PER_GROUP = 4
OB_WORKERS_PER_GROUP = 8


@pl.jit.inline(auto_scope=False)
def output_tasks(
    packed: pl.Tensor[[PACK_ROWS, 4096], pl.BF16],
    wo_a: pl.Tensor[[OUTPUT_GROUPS, 4096, 1024], pl.BF16, pl.NZ],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    residual: pl.Tensor[[T, 16384], pl.BF16],
    post: pl.Tensor[[T, 4], pl.FP32],
    comb: pl.Tensor[[T, 16], pl.FP32],
    o_r: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.FP32]],
    o_quant: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.INT8]],
    o_scales: pl.InOut[pl.Tensor[[8, O_SCALE_ROWS], pl.FP32]],
    o_partials: pl.InOut[pl.Tensor[[O_MM_ROWS, 32768], pl.INT32]],
    output: pl.InOut[pl.Tensor[[T, 16384], pl.BF16]],
    attention_tid: pl.Scalar[pl.TASK_ID],
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
):
    tokens = pl.tensor.dim(residual, 0)
    if OA_COLS == 192:
        oa_units = (tokens + 159) // 160 * 6
    else:
        oa_units = (tokens + 127) // 128 * (1024 // OA_COLS)
    quant_units = (tokens + 31) // 32
    ob_tids = pl.array.create(OUTPUT_GROUPS, pl.TASK_ID)
    # 对齐BSH：各组按就绪核启动，O_A→quant→O_B仍严格依赖；不为小组额外等待整组准入。
    with pl.manual_scope():
        for group in pl.parallel(OUTPUT_GROUPS):
            with pl.spmd(
                pl.min(OA_WORKERS_PER_GROUP, oa_units),
                name_hint="o_proj_a",
                deps=[attention_tid],
                allow_early_resolve=True,
                sync_start=False,
            ) as oa_tid:
                o_r = o_projection_a_groups_stage_nz(
                    packed,
                    wo_a,
                    o_r,
                    tokens,
                    pl.tile.get_block_idx(),
                    OA_WORKERS_PER_GROUP,
                    OA_COLS,
                    group,
                    group + 1,
                )
            with pl.spmd(
                pl.min(QUANT_WORKERS_PER_GROUP, quant_units),
                name_hint="o_quant",
                deps=[oa_tid],
                allow_early_resolve=True,
                sync_start=False,
            ) as quant_tid:
                (o_quant, o_scales) = o_projection_groups_quant_stage(
                    o_r,
                    o_quant,
                    o_scales,
                    tokens,
                    pl.tile.get_block_idx(),
                    QUANT_WORKERS_PER_GROUP,
                    OB_ROWS,
                    group,
                    group + 1,
                )
            with pl.spmd(
                OB_WORKERS_PER_GROUP,
                name_hint="o_proj_b",
                deps=[quant_tid],
                allow_early_resolve=True,
                sync_start=False,
            ) as ob_tid:
                o_partials = o_b_group_task(o_quant, wo_b, o_partials, tokens, group, OB_ROWS)
            ob_tids[group] = ob_tid
    with pl.spmd(
        pl.min(tokens, VECTORS),
        name_hint="hc_post",
        deps=[ob_tids[i] for i in range(OUTPUT_GROUPS)],
        allow_early_resolve=True,
        sync_start=True,
    ):
        output = o_dequant_hc_post_stage(
            o_partials, o_scales, wo_b_scale, residual, post, comb, output, pl.tile.get_block_idx(), VECTORS
        )
    return o_r, o_quant, o_scales, o_partials, output
