# SPDX-License-Identifier: Apache-2.0
"""仅编译动态Indexer组计划/系数；不建立完整CSA、不执行设备。

geometry由一个worker按顺序写计划再查询分组，避免探针自己引入跨核依赖。
coeff2/6按48个worker写不同组，前缀、scale和weights视为已就绪输入。
不同G只选矩阵tile，B/T/各容量仍为动态签名，未绑定任何样本形状。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_group_stages import (
    COEFF_COLS_DYN,
    COEFF_ROWS_DYN,
    HEADS,
    OFFSET_ROWS_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    SCALE_ROWS_DYN,
    TOKENS_DYN,
    WEIGHT_ROWS_DYN,
    indexer_decode_group,
    indexer_group_plan_stage,
    indexer_head_coefficients_stage,
    indexer_leaf_range,
    indexer_work_item,
)

REPORT_ROWS = pl.dynamic("CSA_GROUP_PROBE_REPORT_ROWS")


@pl.jit
def geometry_entry(
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
    report: pl.InOut[pl.Tensor[[REPORT_ROWS, 10], pl.INT32]],
):
    with pl.spmd(1):
        worker = pl.tile.get_block_idx()
        offsets, plan = indexer_group_plan_stage(
            bounds, lengths, positions, offsets, plan, pl.tensor.dim(positions, 0), worker, 6, True, 1
        )
        groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
        leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
        leaf_tiles = pl.cast(pl.read(plan, [3]), pl.INDEX)
        extra = pl.cast(pl.read(plan, [4]), pl.INDEX)
        for item in pl.range(groups * leaves):
            group, leaf = indexer_work_item(item, groups, leaves)
            request, first, live, visible = indexer_decode_group(bounds, lengths, positions, offsets, group, 6)
            begin, capacity, valid = indexer_leaf_range(leaf, leaf_tiles, extra, visible)
            pl.write(report, [item, 0], pl.cast(group, pl.INT32))
            pl.write(report, [item, 1], pl.cast(leaf, pl.INT32))
            pl.write(report, [item, 2], pl.cast(request, pl.INT32))
            pl.write(report, [item, 3], pl.cast(first, pl.INT32))
            pl.write(report, [item, 4], pl.cast(live, pl.INT32))
            pl.write(report, [item, 5], pl.cast(visible, pl.INT32))
            pl.write(report, [item, 6], pl.cast(begin, pl.INT32))
            pl.write(report, [item, 7], pl.cast(capacity, pl.INT32))
            pl.write(report, [item, 8], pl.cast(valid, pl.INT32))
            pl.write(report, [item, 9], pl.cast(worker, pl.INT32))
    return offsets, plan, report


@pl.jit
def coeff2_entry(
    scales: pl.Tensor[[SCALE_ROWS_DYN, HEADS], pl.FP32],
    weights: pl.Tensor[[WEIGHT_ROWS_DYN, HEADS], pl.FP32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16]],
):
    with pl.spmd(48):
        worker = pl.tile.get_block_idx()
        coefficients = indexer_head_coefficients_stage(
            scales, weights, positions, bounds, lengths, offsets, coefficients, worker, 48, 2
        )
    return coefficients


@pl.jit
def coeff6_entry(
    scales: pl.Tensor[[SCALE_ROWS_DYN, HEADS], pl.FP32],
    weights: pl.Tensor[[WEIGHT_ROWS_DYN, HEADS], pl.FP32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16]],
):
    with pl.spmd(48):
        worker = pl.tile.get_block_idx()
        coefficients = indexer_head_coefficients_stage(
            scales, weights, positions, bounds, lengths, offsets, coefficients, worker, 48, 6
        )
    return coefficients
