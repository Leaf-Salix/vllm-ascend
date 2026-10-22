# SPDX-License-Identifier: Apache-2.0
"""Score消费者在核内校验本叶需要读取的Native scale页。

沿用已验证SPMD的整数量化域、正/逆序检查及Vector差值归约，只把校验范围
缩至实际叶的至多256页。结果为当前AIV的两个Scalar，不写GM，不提交规划task，
不与其他核交换状态；AICPU仍负责请求/叶分工和所有跨task依赖。

首尾用于筛选，随后加载全部有效页证明连续性。失败时返回方向0，由Score沿用
原逐页gather。物理页号域限制为2^24，保证FP32比较不丢失有效页号的整数精度。
每个叶的两条AIV各自校验，不复用上一请求、上一叶或上次图重放的结果。
"""

import pypto.language as pl

from .request_geometry import REQUESTS_DYN

LAYOUT_TABLE_COLS = pl.dynamic("CSA_SCALE_LAYOUT_TABLE_COLS")
LAYOUT_PROOF_PAGES = 256  # 一个8192候选叶最多读取256个32行页。
LAYOUT_PROOF_ROWS = 8  # 沿用A3已验证的行归约布局。
LAYOUT_EXACT_INTEGER_LIMIT = 16777216


@pl.jit.inline(auto_scope=False)
def indexer_leaf_scale_layout(
    block_table: pl.Tensor[[REQUESTS_DYN, LAYOUT_TABLE_COLS], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    page_begin: pl.Scalar[pl.INDEX],
    pages: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
):
    """返回本叶首物理页和方向；仅在实际Score AIV内调用，范围不合法则回退。"""
    first_page = pl.cast(0, pl.INDEX)
    direction = pl.cast(0, pl.INT32)
    if request >= 0 and request < pl.tensor.dim(block_table, 0):
        if page_begin >= 0 and pages > 0 and pages <= LAYOUT_PROOF_PAGES:
            if page_begin + pages <= pl.tensor.dim(block_table, 1):
                if physical_pages > 0 and physical_pages <= LAYOUT_EXACT_INTEGER_LIMIT:
                    first = pl.cast(pl.read(block_table, [request, page_begin]), pl.INDEX)
                    last = pl.cast(pl.read(block_table, [request, page_begin + pages - 1]), pl.INDEX)
                    trial_direction = pl.cast(1, pl.INT32)
                    if last < first:
                        trial_direction = pl.cast(-1, pl.INT32)
                    expected_last = first + pl.cast(trial_direction, pl.INDEX) * (pages - 1)
                    if last == expected_last:
                        if pl.min(first, expected_last) >= 0 and pl.max(first, expected_last) < physical_pages:
                            page_ids = pl.load(
                                block_table,
                                [request, page_begin],
                                [1, LAYOUT_PROOF_PAGES],
                                valid_shape=[1, pages],
                            )
                            expected = pl.set_validshape(
                                pl.add(
                                    pl.mul(
                                        pl.cast(pl.tile.ci(0, [1, LAYOUT_PROOF_PAGES], dtype=pl.INT32), pl.FP32),
                                        pl.cast(trial_direction, pl.FP32),
                                    ),
                                    pl.cast(pl.cast(first, pl.INT32), pl.FP32),
                                ),
                                1,
                                pages,
                            )
                            difference = pl.abs(pl.sub(pl.cast(page_ids, pl.FP32), expected))
                            ones = pl.set_validshape(
                                pl.tile.full([LAYOUT_PROOF_ROWS, LAYOUT_PROOF_PAGES], dtype=pl.FP32, value=1.0),
                                LAYOUT_PROOF_ROWS,
                                pages,
                            )
                            repeated = pl.set_validshape(
                                pl.col_expand_mul(ones, difference), LAYOUT_PROOF_ROWS, pages
                            )
                            scratch = pl.create_tile([LAYOUT_PROOF_ROWS, LAYOUT_PROOF_PAGES], dtype=pl.FP32)
                            max_difference = pl.row_max(repeated, scratch)
                            if pl.tile.read(max_difference, [0, 0]) == 0.0:
                                first_page = first
                                direction = trial_direction
    return first_page, direction
