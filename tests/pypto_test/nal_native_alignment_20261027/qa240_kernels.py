# SPDX-License-Identifier: Apache-2.0
"""T240 Q-A：保持原N128/M64+M16所有权，区分N32细分和Native K次序。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.qkv_stages import (
    BF16_WEIGHT_LAYOUT,
    QA_ROWS_DYN,
    T_DYN,
    _q_a_worker,
)


@pl.jit.inline(auto_scope=False)
def native_block(column: pl.Scalar[pl.INDEX], step: pl.Scalar[pl.INDEX], ORDER: pl.constexpr):
    block = step
    group = column // 96
    if ORDER and group < 9:
        direction = step
        if group % 2 == 1:
            direction = 15 - step
        block = (group // 2 * 3 + direction) % 16
    return block


@pl.jit.inline(auto_scope=False)
def narrow240(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[QA_ROWS_DYN, 1024], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ORDER: pl.constexpr,
):
    for logical in pl.range(pl.max(worker, 0), 8 * 3, WORKERS):
        col128 = logical % 8 * 128
        m_group = logical // 8
        for sub in pl.range(4):
            col = col128 + sub * 32
            acc = pl.create_tensor([64, 32], dtype=pl.FP32)
            for step in pl.pipeline(16, stage=2):
                k = pl.max(native_block(col, step, ORDER), 0) * 256
                a = x[m_group * 64 : m_group * 64 + 64, k : k + 256]
                b = weight[col : col + 32, k : k + 256]
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=(step == 0))
            projected = pl.assemble(projected, acc, [m_group * 64, col])
            row = 192 + m_group * 16
            tail = pl.create_tensor([16, 32], dtype=pl.FP32)
            for tail_step in pl.pipeline(16, stage=2):
                tail_k = pl.max(native_block(col, tail_step, ORDER), 0) * 256
                tail_a = x[row : row + 16, tail_k : tail_k + 256]
                tail_b = weight[col : col + 32, tail_k : tail_k + 256]
                tail = pl.matmul_acc(tail, tail_a, tail_b, b_trans=True, init_cond=(tail_step == 0))
            projected = pl.assemble(projected, tail, [row, col])
    return projected


@pl.jit
def qa240_arm(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[QA_ROWS_DYN, 1024], pl.FP32]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(24, name_hint="qa240_same_owner", allow_early_resolve=True):
        if MODE == 0:
            # Freeze the pre-fix T240 branch as a negative control even after
            # q_a_stage dispatches T240 to the corrected implementation.
            projected = _q_a_worker(x, weight, projected, worker, 24, 64)
        else:
            projected = narrow240(x, weight, projected, worker, 24, MODE == 2)
    return projected
