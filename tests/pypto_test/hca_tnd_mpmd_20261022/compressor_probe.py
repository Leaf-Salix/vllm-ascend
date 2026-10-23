# SPDX-License-Identifier: Apache-2.0
"""给定 Native 归一化结果，单独验证 TND C128 的请求边界、state 与 compact 输出。"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd import compressor as c
from vllm_ascend.ops.pypto.deepseek_v4_flash_hca_tnd.compressor import compressor_ratio128

@pl.jit
def hca_compressor_test(
    x: pl.Tensor[[c.T_DYN, c.D], pl.BF16],
    wkv: pl.Tensor[[c.HEAD_DIM, c.D], pl.BF16],
    wgate: pl.Tensor[[c.HEAD_DIM, c.D], pl.BF16],
    ape: pl.Tensor[[c.RATIO, c.HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[c.HEAD_DIM], pl.BF16],
    state: pl.InOut[pl.Tensor[[c.STATE_PAGES_DYN, c.STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    state_table: pl.Tensor[[c.B_DYN, c.STATE_COLUMNS_DYN], pl.INT32],
    state_slots: pl.Tensor[[c.T_DYN, 2], pl.INT32],
    positions: pl.Tensor[[c.T_DYN], pl.INT64],
    query_bounds: pl.Tensor[[c.BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[c.B_DYN], pl.INT32],
    cos: pl.Tensor[[c.COMPACT_ROWS_DYN, c.ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[c.COMPACT_ROWS_DYN, c.ROPE_DIM], pl.FP32],
    cmp_slots: pl.Tensor[[c.COMPACT_ROWS_DYN, 2], pl.INT32],
    cmp_cache: pl.InOut[pl.Tensor[[c.CMP_PAGES_DYN, c.CMP_BLOCK, 1, c.HEAD_DIM], pl.BF16]],
):
    x.bind_dynamic(0, c.T_DYN)
    state.bind_dynamic(0, c.STATE_PAGES_DYN)
    state.bind_dynamic(1, c.STATE_PAGE_ELEMENTS_DYN)
    state_table.bind_dynamic(0, c.B_DYN)
    state_table.bind_dynamic(1, c.STATE_COLUMNS_DYN)
    state_slots.bind_dynamic(0, c.T_DYN)
    positions.bind_dynamic(0, c.T_DYN)
    query_bounds.bind_dynamic(0, c.BOUNDS_DYN)
    seq_lens.bind_dynamic(0, c.B_DYN)
    cos.bind_dynamic(0, c.COMPACT_ROWS_DYN)
    sin.bind_dynamic(0, c.COMPACT_ROWS_DYN)
    cmp_slots.bind_dynamic(0, c.COMPACT_ROWS_DYN)
    cmp_cache.bind_dynamic(0, c.CMP_PAGES_DYN)
    ready = pl.system.task_dummy(deps=[])
    compressor_ratio128(
        x, wkv, wgate, ape, norm_w, state, state_table, state_slots, positions,
        query_bounds, seq_lens, cos, sin, cmp_slots, cmp_cache, ready,
    )
    return state, cmp_cache
