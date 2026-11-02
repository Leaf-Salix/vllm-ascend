# SPDX-License-Identifier: Apache-2.0
"""T192 WKV：原N128/K512、N32/K256及已有T240列累加顺序假设。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_native_stages import _kv_t240_k_block, kv_order_stage
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_stages import (
    BF16_WEIGHT_LAYOUT,
    PAD_ROWS_DYN,
    T_DYN,
    kv_project_stage,
)


@pl.jit.inline(auto_scope=False)
def group64_block(column: pl.Scalar[pl.INDEX], step: pl.Scalar[pl.INDEX]):
    # 候选假设：实际Native为N64、8列组；轮转/方向仍须逐bit对拍确认。
    group = column // 64
    direction = step
    if group % 2 == 1:
        direction = 15 - step
    return (group // 2 * 4 + direction) % 16


@pl.jit.inline(auto_scope=False)
def ordered192(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    GROUP64: pl.constexpr = False,
):
    # 与原WKV相同的column-major owner；T192恰三个M64块，无尾块。
    for logical in pl.range(worker, 4 * 3, 24):
        col128 = logical // 3 * 128
        m_group = logical % 3
        for sub in pl.range(4):
            col = col128 + sub * 32
            acc = pl.create_tensor([64, 32], dtype=pl.FP32)
            for step in pl.pipeline(16, stage=2):
                block = _kv_t240_k_block(col, step)
                if GROUP64:
                    block = group64_block(col, step)
                k = pl.max(block, 0) * 256
                a = x[m_group * 64 : m_group * 64 + 64, k : k + 256]
                b = weight[col : col + 32, k : k + 256]
                acc = pl.matmul_acc(acc, a, b, b_trans=True, init_cond=(step == 0))
            projected = pl.assemble(projected, acc, [m_group * 64, col])
    return projected


@pl.jit
def wkv192_arm(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(24, name_hint="wkv192_same_owner", allow_early_resolve=True):
        if MODE == 0:
            projected = kv_project_stage(x, weight, projected, worker, 24)
        elif MODE == 1:
            projected = kv_order_stage(x, weight, projected, worker, 24, False)
        else:
            projected = ordered192(x, weight, projected, worker, MODE == 3)
    return projected
