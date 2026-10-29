# SPDX-License-Identifier: Apache-2.0
"""T240三臂：保留N128生产者归属，仅改变内部N/K分块与累加次序。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_native_stages import kv_order_stage
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_stages import (
    BF16_WEIGHT_LAYOUT,
    PAD_ROWS_DYN,
    T_DYN,
    kv_project_stage,
)


@pl.jit.inline(auto_scope=False)
def column_k_block(column: pl.Scalar[pl.INDEX], step: pl.Scalar[pl.INDEX]):
    block = step
    if column >= 96 and column < 192:
        block = 15 - step
    elif column >= 192 and column < 288:
        block = (step + 5) % 16
    elif column >= 288 and column < 384:
        block = (20 - step) % 16
    return block


@pl.jit.inline(auto_scope=False)
def ordered240(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
):
    # T240: three M64 groups, then one M16 tail per group, as in production.
    for logical in pl.range(worker, 4 * 3, 24):
        col128 = logical // 3 * 128
        m_group = logical % 3
        for sub in pl.range(4):
            col = col128 + sub * 32
            acc = pl.create_tensor([64, 32], dtype=pl.FP32)
            for step in pl.pipeline(16, stage=2):
                k = pl.max(column_k_block(col, step), 0) * 256
                a = x[m_group * 64 : m_group * 64 + 64, k : k + 256]
                b = weight[col : col + 32, k : k + 256]
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=(step == 0))
            projected = pl.assemble(projected, acc, [m_group * 64, col])
            tail = pl.create_tensor([16, 32], dtype=pl.FP32)
            row = 192 + m_group * 16
            for tail_step in pl.pipeline(16, stage=2):
                tail_k = pl.max(column_k_block(col, tail_step), 0) * 256
                tail_a = x[row : row + 16, tail_k : tail_k + 256]
                tail_b = weight[col : col + 32, tail_k : tail_k + 256]
                tail = pl.matmul_acc(tail, tail_a, tail_b, b_trans=True, init_cond=(tail_step == 0))
            projected = pl.assemble(projected, tail, [row, col])
    return projected


@pl.jit
def wkv240_arm(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(24, name_hint="wkv240_same_owner", allow_early_resolve=True):
        if MODE == 0:
            projected = kv_project_stage(x, weight, projected, worker, 24)
        elif MODE == 1:
            projected = kv_order_stage(x, weight, projected, worker, 24, False)
        else:
            projected = ordered240(x, weight, projected, worker)
    return projected
