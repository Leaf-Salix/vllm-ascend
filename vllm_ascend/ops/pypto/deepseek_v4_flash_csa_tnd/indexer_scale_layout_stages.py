# SPDX-License-Identifier: Apache-2.0
"""AICPU编排层校验Native页表的正/逆序连续布局，供Score批量读取scale。

复用Native的Q/L和页表，无Host回读。先筛首尾及合法范围，再逐页校验；
失败、padding或非法容量写(0,0)，每次调用覆盖全部请求的两项证明。
沿用已验证Score的保守2^24页号上限和workspace分区，不改变scale/Key布局。
本函数只有Scalar读写，不启动AICore，不使用阶段信号；消费者由AICPU提交。
证明的真实运行成本计入CSA含AICPU跨度，不能用L1下降代表该成本消失。
"""

import pypto.language as pl

from .indexer_group_stages import OFFSET_ROWS_DYN
from .indexer_scale_plan_stages import SCALE_JOB_WORDS, indexer_scale_plan_base
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

LAYOUT_TABLE_COLS = pl.dynamic("CSA_SCALE_LAYOUT_TABLE_COLS")
LAYOUT_WORDS_PER_REQUEST = 2
LAYOUT_EXACT_INTEGER_LIMIT = 16777216
LAYOUT_COMPRESS_RATIO = 4
LAYOUT_ROWS_PER_PAGE = 32


@pl.jit.inline(auto_scope=False)
def indexer_scale_layout_base(
    requests: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """追加到容量分区尾部；不能用本轮实际 scale job 数缩短前面的保留区。"""
    base = indexer_scale_plan_base(requests, tokens)
    return base + WORKERS + SCALE_JOB_WORDS * tokens


@pl.jit.inline(auto_scope=False)
def indexer_scale_layout_plan_stage(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    block_table: pl.Tensor[[REQUESTS_DYN, LAYOUT_TABLE_COLS], pl.INT32],
    workspace: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    tokens: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """在AICPU编排执行，每个请求只写一次结果。"""
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    base = indexer_scale_layout_base(requests, tokens, WORKERS)
    for request in pl.range(requests):
        first_page = pl.cast(0, pl.INT32)
        direction = pl.cast(0, pl.INT32)
        begin, end, active = request_bounds(bounds, lengths, request, tokens)
        if active != 0:
            length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
            compressed = length // LAYOUT_COMPRESS_RATIO
            pages = (compressed + LAYOUT_ROWS_PER_PAGE - 1) // LAYOUT_ROWS_PER_PAGE
            if pages > 0:
                if pages <= pl.tensor.dim(block_table, 1):
                    if physical_pages > 0:
                        if physical_pages <= LAYOUT_EXACT_INTEGER_LIMIT:
                            first = pl.cast(pl.read(block_table, [request, 0]), pl.INDEX)
                            last = pl.cast(pl.read(block_table, [request, pages - 1]), pl.INDEX)
                            trial_direction = pl.cast(1, pl.INT32)
                            if last < first:
                                trial_direction = pl.cast(-1, pl.INT32)
                            expected_last = first + pl.cast(trial_direction, pl.INDEX) * (pages - 1)
                            if last == expected_last:
                                if pl.min(first, expected_last) >= 0:
                                    if pl.max(first, expected_last) < physical_pages:
                                        first_page = pl.cast(first, pl.INT32)
                                        direction = trial_direction
                                        # 首尾不构成完整证明，仍检查每个中间页；失配后不继续读表。
                                        for page in pl.range(1, pages - 1):
                                            if direction != 0:
                                                actual = pl.cast(pl.read(block_table, [request, page]), pl.INDEX)
                                                expected = first + pl.cast(trial_direction, pl.INDEX) * page
                                                if actual != expected:
                                                    direction = pl.cast(0, pl.INT32)
        if direction == 0:
            first_page = pl.cast(0, pl.INT32)
        # 两项都无条件覆盖。相邻请求共享缓存行，由唯一 Scalar 写者发布。
        row = base + LAYOUT_WORDS_PER_REQUEST * request
        pl.write(workspace, [row], first_page)
        pl.write(workspace, [row + 1], direction)
    return workspace
