# SPDX-License-Identifier: Apache-2.0
"""HCA TND 的四组 Q_B 发布候选：扩大 Cube 并行度，保持逐行数学。

Q_A/QR 和 KV 继续使用已验证的公共函数；本文件只重新组织 Q_B 的整数矩阵乘。
每个组处理16个 head，保持 K512 搬入、K128 累加、N256 列块以及原来的 Vector
反量化/RMS/RoPE。四组各6个 Cube worker，总共24个；不按单个 B 特化。
INT8×INT8 在 K1024 内不会溢出 INT32，分工变化不改变同一输出的累加次序。

TaskId 与 Tensor 输入输出表达所有跨任务依赖；没有核间 GM 轮询。
本候选在完整层配对性能与状态等价通过后才由正式 entry 采用。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT
from ..deepseek_v4_flash_csa.qkv_proj_rope import (
    D, H, HEAD_DIM, PREFILL_DENSE_TILE, Q_LORA, QPROJ_T_PAD, ROPE_DIM, T_DYN,
    kv_proj_rope, q_proj_qr, rope_prepare,
)
from ..deepseek_v4_flash_hca.q_projection_streamed import q_proj_q_dequant_stream

GROUPS = 4
GROUP_HEADS = H // GROUPS
CUBE_WORKERS = 6
M_TILE = 128
N_TILE = 256
L1_K = 512
L0_K = 128


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    q_ready: pl.Array[4, pl.TASK_ID],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(x, 0)
    sin_signed = pl.create_tensor([tokens, ROPE_DIM], dtype=pl.FP32)
    swap_indices = pl.create_tensor([tokens, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, sin_signed, swap_indices)
    qr_i8 = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    scale_pad = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
    qa_tid = q_proj_qr(x, wq_a, gamma_cq, qr, qr_scale, qr_i8, scale_pad)
    qproj_dep = pl.system.task_dummy(deps=[])
    for tile_base in pl.range(0, tokens, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, tokens - tile_base)
        matrix_rows = (tile_rows + M_TILE - 1) // M_TILE * M_TILE
        # unroll 只展开固定的四个 head 组；T、有效行以及任务循环仍为运行时值。
        for group in pl.unroll(GROUPS):
            with pl.scope():
                head_base = group * GROUP_HEADS
                projected = pl.create_tensor([matrix_rows, GROUP_HEADS * HEAD_DIM], dtype=pl.INT32)
                with pl.spmd(CUBE_WORKERS, name_hint="hca_tnd_qb_group",
                             deps=[qproj_dep, q_ready[group]]):
                    pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
                    worker = pl.tile.get_block_idx()
                    n_blocks = GROUP_HEADS * HEAD_DIM // N_TILE
                    for step in pl.range((n_blocks - worker + CUBE_WORKERS - 1) // CUBE_WORKERS):
                        local_col = (worker + step * CUBE_WORKERS) * N_TILE
                        # unroll 组号跨 outlining 后不一定保留符号范围，显式证明 NZ 列非负。
                        col = pl.max(head_base * HEAD_DIM + local_col, 0)
                        for row in pl.range(0, matrix_rows, M_TILE):
                            count = pl.min(M_TILE, tile_rows - row)
                            storage = pl.create_tile([M_TILE, N_TILE], dtype=pl.INT32,
                                                     target_memory=pl.MemorySpace.Acc, compact=True)
                            seed = pl.tile.set_validshape(storage, count, N_TILE)
                            for outer, (outer_acc,) in pl.pipeline(
                                0, Q_LORA // L1_K, stage=2, init_values=(seed,),
                            ):
                                k_base = outer * L1_K
                                a_l1 = pl.load(qr_i8, [row, k_base], [M_TILE, L1_K],
                                               valid_shape=[count, L1_K], target_memory=pl.MemorySpace.Mat)
                                b_l1 = pl.load(wq_b, [k_base, col], [L1_K, N_TILE],
                                               target_memory=pl.MemorySpace.Mat)
                                for inner, (inner_acc,) in pl.pipeline(
                                    0, L1_K // L0_K, stage=2, init_values=(outer_acc,),
                                ):
                                    k_inner = inner * L0_K
                                    a_l0 = pl.tile.extract(a_l1, 0, k_inner, [M_TILE, L0_K],
                                                          target_memory=pl.MemorySpace.Left)
                                    b_l0 = pl.tile.extract(b_l1, k_inner, 0, [L0_K, N_TILE],
                                                          target_memory=pl.MemorySpace.Right)
                                    updated = pl.tile.matmul_acc(inner_acc, a_l0, b_l0,
                                                                init_cond=(outer == 0 and inner == 0))
                                    inner_done = pl.yield_(updated)
                                outer_done = pl.yield_(inner_done)
                            projected = pl.store(outer_done, [row, local_col], projected)
                ready = q_proj_q_dequant_stream(
                    wq_b_scale, rope_cos, sin_signed, swap_indices, q, scale_pad,
                    projected, tile_base, tile_rows, head_base,
                )
                q_ready[group] = ready
    kv_proj_rope(x, wkv, gamma_ckv, rope_cos, sin_signed, swap_indices, kv, late_dep)
    return q, qa_tid
