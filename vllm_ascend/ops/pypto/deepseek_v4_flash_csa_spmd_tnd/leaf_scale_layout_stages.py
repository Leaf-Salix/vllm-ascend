# SPDX-License-Identifier: Apache-2.0
"""Score AIV对本叶需要的Native scale页做局部布局证明。

只在policy选择的长G6、T容量64～96、压缩长度上界不超过6×8192的资源区间
使用。每个实际消费者各自验证至多256页的正序或逆序连续性；结果是本核的
首物理页及方向，不写GM证明表，不增加task、通知或跨核等待。

首尾只用于筛选，随后检查全部有效页。物理页号域不超过2^24时，INT32到
FP32比较保持整数精度。归约保留物理8×256布局，但只使用一个有效行，避免
重复计算七行相同数据。范围或连续性验证失败返回方向0，调用方执行原逐页
读取；不复用上次调用、上一请求或上一叶的证明。

源机制来自MPMD叶内证明；单SPMD的启用范围和完整L1/动态状态/服务证据见
DSV4_CSA_SPMD_TND_IMPLEMENTATION_LOG.md，其他区间保留原策略。
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
                                1,
                                pages,
                            )
                            repeated = pl.set_validshape(pl.col_expand_mul(ones, difference), 1, pages)
                            scratch = pl.create_tile([LAYOUT_PROOF_ROWS, LAYOUT_PROOF_PAGES], dtype=pl.FP32)
                            max_difference = pl.row_max(repeated, scratch)
                            if pl.tile.read(max_difference, [0, 0]) == 0.0:
                                first_page = first
                                direction = trial_direction
    return first_page, direction
