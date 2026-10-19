# SPDX-License-Identifier: Apache-2.0
"""Score/Attention 交错流水的动态叶游标：整体思路与阅读顺序。

本模块只在 Scalar 上读本次已发布的 group 描述符和叶划分计划，不读回 Host、
不改变数学 tile，也不重新分核。先读 next_score_leaf，再读 score_wave_step_count。
G6/1024 是已有平衡 Score 策略的数学形状，B、T、请求长度、query 位置、组数、
叶数及每叶循环长度仍来自运行时 metadata。wave 只筛选全局 group 半开区间，
所有权始终是原始 item % 24；少组时叶优先、多组时组优先的规则完整保留。

每个 wave 从本核 worker 开始扫描。不能把前一 wave 的 cursor 继续往后用，
因为少组路径的不同 wave 在全局 item 序列中交错。next_item 指向命中叶后的
下一个本核 item；found=0 时其他返回值均为安全的零值，不得初始化或读取叶。
计数遍历只用于建立明确的 pl.range 循环边界，随后真正执行仍重新读取同份
不可变计划；它有 Scalar 成本，不能把源码重排直接当成性能收益。

外层负责原有 metadata acquire、A3 业务缓存维护、FFTS ready/ack 和根节点通知。
本模块没有同步副作用。每叶末 AIC 必须排空 min(steps,2) 个 ACK，AIV 必须先
写完各自有效 query 的根，才允许推进叶或发布整 wave 完成。
当前为候选联合实现，尚未经整层 lowering 或设备验证。
"""

import pypto.language as pl

from .indexer_group_stages import (
    OFFSET_ROWS_DYN, PLAN_WORDS, TOKENS_DYN,
    indexer_decode_group, indexer_leaf_range, indexer_work_item,
)
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN

SCORE_WORKERS = 24
QUERY_GROUP = 6
SCORE_TILE = 1024
LEAF_LANE_CAPACITY = 4096


@pl.jit.inline(auto_scope=False)
def next_score_leaf(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    item_start: pl.Scalar[pl.INDEX],
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """返回下一叶十二元组；仅 found=1 才允许使用其业务地址。"""
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    leaf_tiles = pl.cast(pl.read(plan, [3]), pl.INDEX)
    extra = pl.cast(pl.read(plan, [4]), pl.INDEX)
    zero = pl.cast(0, pl.INDEX)
    # 显式携带全部状态，避免JIT自然while将cursor更新提前版本化却不更新条件。
    # 即使零次迭代，循环结果也等于init_values；无叶时十二项均有定义。
    for (
        cursor_i, found_i, group_i, leaf_i, batch_i, query_i, live_i,
        cache_i, begin_i, valid_i, steps_i, span_i,
    ) in pl.while_(init_values=(item_start, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)):
        pl.cond(cursor_i < groups * leaves and found_i == 0)
        candidate_group, candidate_leaf = indexer_work_item(cursor_i, groups, leaves)
        next_found = found_i
        next_group = group_i
        next_leaf = leaf_i
        next_batch = batch_i
        next_query = query_i
        next_live = live_i
        next_cache = cache_i
        next_begin = begin_i
        next_valid = valid_i
        next_steps = steps_i
        next_span = span_i
        if candidate_group >= group_begin and candidate_group < group_end:
            owner, first, count, visible = indexer_decode_group(
                bounds, lengths, positions, offsets, candidate_group, QUERY_GROUP
            )
            begin, capacity, valid = indexer_leaf_range(candidate_leaf, leaf_tiles, extra, visible)
            if valid > 0 and count > 0:
                next_found = pl.cast(1, pl.INDEX)
                next_group = candidate_group
                next_leaf = candidate_leaf
                next_batch = owner
                next_query = first
                next_live = count
                next_cache = pl.cast(pl.read(lengths, [owner]), pl.INDEX) // 4
                next_begin = begin
                next_valid = pl.min(capacity, visible - begin)
                tile_count = (next_valid + SCORE_TILE - 1) // SCORE_TILE
                next_span = pl.min(tile_count * (SCORE_TILE // 2), LEAF_LANE_CAPACITY)
                next_steps = (next_span + SCORE_TILE // 2 - 1) // (SCORE_TILE // 2)
        cursor, found, group, leaf, batch, query, live, cache_len, logical_begin, valid_count, steps, lane_span = pl.yield_(
            cursor_i + SCORE_WORKERS, next_found, next_group, next_leaf, next_batch,
            next_query, next_live, next_cache, next_begin, next_valid, next_steps, next_span,
        )
    return cursor, found, group, leaf, batch, query, live, cache_len, logical_begin, valid_count, steps, lane_span


@pl.jit.inline(auto_scope=False)
def score_wave_step_count(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    worker: pl.Scalar[pl.INDEX],
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """遍历本核真实有效叶求总步数；空wave严格为0，不触发FFT​​S。"""
    zero = pl.cast(0, pl.INDEX)
    one = pl.cast(1, pl.INDEX)
    for cursor_i, found_i, total_i in pl.while_(init_values=(worker, one, zero)):
        pl.cond(found_i > 0)
        cursor_next, found_next, group, leaf, batch, query, live, cache_len, begin, valid, steps, lane_span = next_score_leaf(
            bounds, lengths, positions, offsets, plan, cursor_i, group_begin, group_end
        )
        cursor, found, total = pl.yield_(cursor_next, found_next, total_i + steps)
    return total
