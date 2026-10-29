# SPDX-License-Identifier: Apache-2.0
"""WKV K256/N32候选；保留原N128生产者的全部行/列所有权。

采用当前小M下的paired/natural顺序，以及单独验收的T240列分组顺序；
数值资格必须逐档验证，不能跨环境沿用。
每个原N128生产者顺序计算四个N32子块；返回前全部写完，不创建任务或信号。
"""

import pypto.language as pl

from .kv_stages import (
    BF16_WEIGHT_LAYOUT,
    HEAD_DIM,
    KV_DENSE_M_TILE,
    KV_FIXED_SMALL_M_TILE,
    KV_FIXED_SMALL_ROWS,
    KV_M_TILE,
    KV_N_TILE,
    KV_OM,
    PAD_ROWS_DYN,
    PREFILL_DENSE_TILE,
    T_DYN,
    D,
)

NATIVE_N = 32
NATIVE_K = 256
K_BLOCKS = D // NATIVE_K


@pl.jit.inline(auto_scope=False)
def paired_k_block(column: pl.Scalar[pl.INDEX], step: pl.Scalar[pl.INDEX], ORDER: pl.Scalar[pl.BOOL]):
    block = step
    if ORDER:
        group = column // NATIVE_N
        pair = step // 2
        if group % 2 == 1:
            pair = K_BLOCKS // 2 - 1 - pair
        block = (2 * pair + 2 * (group // 2) + step % 2) % K_BLOCKS
    return block


@pl.jit.inline(auto_scope=False)
def _kv_t240_k_block(column: pl.Scalar[pl.INDEX], step: pl.Scalar[pl.INDEX]):
    block = step
    if column >= 96 and column < 192:
        block = K_BLOCKS - 1 - step
    elif column >= 192 and column < 288:
        block = (step + 5) % K_BLOCKS
    elif column >= 288 and column < 384:
        block = (20 - step) % K_BLOCKS
    return block


@pl.jit.inline(auto_scope=False)
def kv_t240_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, HEAD_DIM], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    # Preserve the original N128/M64 owner and its M16 tail. All N32 pieces
    # complete before the caller publishes this owner's KV completion signal.
    for logical in pl.range(worker, (HEAD_DIM // KV_N_TILE) * KV_OM, WORKERS):
        col128 = logical // KV_OM * KV_N_TILE
        m_group = logical % KV_OM
        for sub in pl.range(KV_N_TILE // NATIVE_N):
            col = col128 + sub * NATIVE_N
            acc = pl.create_tensor([KV_DENSE_M_TILE, NATIVE_N], dtype=pl.FP32)
            for step in pl.pipeline(K_BLOCKS, stage=2):
                k = pl.max(_kv_t240_k_block(col, step), 0) * NATIVE_K
                a = x[m_group * KV_DENSE_M_TILE : m_group * KV_DENSE_M_TILE + KV_DENSE_M_TILE, k : k + NATIVE_K]
                b = weight[col : col + NATIVE_N, k : k + NATIVE_K]
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=(step == 0))
            projected = pl.assemble(projected, acc, [m_group * KV_DENSE_M_TILE, col])
            tail = pl.create_tensor([KV_M_TILE, NATIVE_N], dtype=pl.FP32)
            row = KV_OM * KV_DENSE_M_TILE + m_group * KV_M_TILE
            for tail_step in pl.pipeline(K_BLOCKS, stage=2):
                tail_k = pl.max(_kv_t240_k_block(col, tail_step), 0) * NATIVE_K
                tail_a = x[row : row + KV_M_TILE, tail_k : tail_k + NATIVE_K]
                tail_b = weight[col : col + NATIVE_N, tail_k : tail_k + NATIVE_K]
                tail = pl.matmul_acc(tail, tail_a, tail_b, b_trans=True, init_cond=(tail_step == 0))
            projected = pl.assemble(projected, tail, [row, col])
    return projected


@pl.jit.inline(auto_scope=False)
def _kv_order_worker(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, HEAD_DIM], pl.FP32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    DENSE_M: pl.constexpr,
    ORDER: pl.Scalar[pl.BOOL],
):
    padded = (tile_rows + KV_M_TILE - 1) // KV_M_TILE * KV_M_TILE
    full = tile_rows // DENSE_M * DENSE_M
    groups = pl.min(KV_OM, pl.max(1, tile_rows // DENSE_M))
    # Same logical N128/M producer as Nal; only subdivide inside this owner.
    for logical in pl.range(worker, (HEAD_DIM // KV_N_TILE) * groups, WORKERS):
        col128 = (logical // groups) * KV_N_TILE
        m_group = logical % groups
        for sub in pl.range(KV_N_TILE // NATIVE_N):
            column = pl.max(0, col128 + sub * NATIVE_N)
            for row in pl.range(m_group * DENSE_M, full, groups * DENSE_M):
                acc = pl.create_tensor([DENSE_M, NATIVE_N], dtype=pl.FP32)
                for step in pl.pipeline(K_BLOCKS, stage=2):
                    block = paired_k_block(column, step, ORDER)
                    offset = pl.max(block, 0) * NATIVE_K
                    a = x[tile_base + row : tile_base + row + DENSE_M, offset : offset + NATIVE_K]
                    b = weight[column : column + NATIVE_N, offset : offset + NATIVE_K]
                    acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=(step == 0))
                projected = pl.assemble(projected, acc, [tile_base + row, column])
            for row_tail in pl.range(full + m_group * KV_M_TILE, padded, groups * KV_M_TILE):
                tail = pl.create_tensor([KV_M_TILE, NATIVE_N], dtype=pl.FP32)
                for step_tail in pl.pipeline(K_BLOCKS, stage=2):
                    block_tail = paired_k_block(column, step_tail, ORDER)
                    offset_tail = pl.max(block_tail, 0) * NATIVE_K
                    valid = pl.min(KV_M_TILE, tile_rows - row_tail)
                    a_tail = pl.slice(
                        x, [KV_M_TILE, NATIVE_K], [tile_base + row_tail, offset_tail], valid_shape=[valid, NATIVE_K]
                    )
                    b_tail = weight[column : column + NATIVE_N, offset_tail : offset_tail + NATIVE_K]
                    tail = pl.matmul_acc(tail, a_tail, b_tail, b_trans=True, init_cond=(step_tail == 0))
                projected = pl.assemble(projected, tail, [tile_base + row_tail, column])
    return projected


@pl.jit.inline(auto_scope=False)
def kv_order_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[HEAD_DIM, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, HEAD_DIM], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ORDER: pl.Scalar[pl.BOOL],
):
    tokens = pl.tensor.dim(x, 0)
    for first in pl.range(0, tokens, PREFILL_DENSE_TILE):
        rows = pl.min(PREFILL_DENSE_TILE, tokens - first)
        if rows < KV_FIXED_SMALL_ROWS:
            projected = _kv_order_worker(
                x, weight, projected, first, rows, worker, WORKERS, KV_FIXED_SMALL_M_TILE, ORDER
            )
        else:
            projected = _kv_order_worker(x, weight, projected, first, rows, worker, WORKERS, KV_DENSE_M_TILE, ORDER)
    return projected
