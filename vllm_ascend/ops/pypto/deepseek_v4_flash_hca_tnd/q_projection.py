# SPDX-License-Identifier: Apache-2.0
"""HCA大T Q_B：保留动态行数，用更高的M块减少NZ权重重复加载。

整体思路：Native适配层用已有Host max_seq_lens和输入shape选择独立入口。
长历史且129≤T≤192采用M192/N128，一轮M覆盖有效token；其他场景走原入口。
选择发生在绑定阶段，不在每次TMR编排中展开两个分支，不新增设备到Host读取。
T、B、Q/L及页表仍然动态；本文件不使用T=B*S，也不固化历史长度或请求排列。

Q_B沿四个head组分别发布，每组4个Cube；L1 K512和L0 K128均双缓冲。
相较原M128/N256跨两轮M，本区间的32MiB权重只遍历一轮。
L1两组A/B缓冲共320KiB，L0A双槽48KiB、L0B双槽32KiB，Acc96KiB。
INT8点积仍完整累加到INT32；反量化、RMS、RoPE复用已验证原数学函数。
任务依赖、head分组和Q_ready发布保持原样，供整层attention消费。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT
from ..deepseek_v4_flash_csa.qkv_proj_rope import (
    D, H, HEAD_DIM, PREFILL_DENSE_TILE, Q_LORA, QPROJ_T_PAD, ROPE_DIM, T_DYN,
    kv_proj_rope, q_proj_qr, rope_prepare,
)
from ..deepseek_v4_flash_hca.q_projection_streamed import (
    q_proj_q_dequant_stream,
)

STREAM_HEADS = 16
STREAM_CUBE_WORKERS = 4
STREAM_M = 192
STREAM_N = 128
STREAM_L1_K = 512
STREAM_L0_K = 128


@pl.jit.inline(auto_scope=False)
def _q_proj_q_tall(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    qproj_dep: pl.Scalar[pl.TASK_ID],
    q_ready: pl.Array[4, pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        matrix_rows = ((tile_rows + STREAM_M - 1) // STREAM_M) * STREAM_M
        with pl.scope():
            head_base = 0
            q_proj_i32 = pl.create_tensor([matrix_rows, STREAM_HEADS * HEAD_DIM], dtype=pl.INT32)
            with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep, q_ready[0]]):
                pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
                worker = pl.tile.get_block_idx()
                n_blocks = STREAM_HEADS * HEAD_DIM // STREAM_N
                for step in pl.range((n_blocks - worker + STREAM_CUBE_WORKERS - 1) // STREAM_CUBE_WORKERS):
                    col_local = (worker + step * STREAM_CUBE_WORKERS) * STREAM_N
                    col = 0 + col_local
                    for t0 in pl.range(0, matrix_rows, STREAM_M):
                        count = pl.min(STREAM_M, tile_rows - t0)
                        seed_storage = pl.create_tile([STREAM_M, STREAM_N], dtype=pl.INT32,
                                                      target_memory=pl.MemorySpace.Acc, compact=True)
                        seed = pl.tile.set_validshape(seed_storage, count, STREAM_N)
                        for outer, (outer_acc,) in pl.pipeline(0, Q_LORA // STREAM_L1_K, stage=2, init_values=(seed,)):
                            k_base = outer * STREAM_L1_K
                            a_l1 = pl.load(qr_i8_matmul, [t0, k_base], [STREAM_M, STREAM_L1_K],
                                           valid_shape=[count, STREAM_L1_K], target_memory=pl.MemorySpace.Mat)
                            b_l1 = pl.load(wq_b, [k_base, col], [STREAM_L1_K, STREAM_N],
                                           target_memory=pl.MemorySpace.Mat)
                            for inner, (inner_acc,) in pl.pipeline(
                                0, STREAM_L1_K // STREAM_L0_K, stage=2, init_values=(outer_acc,),
                            ):
                                k_inner = inner * STREAM_L0_K
                                a_l0 = pl.tile.extract(a_l1, 0, k_inner, [STREAM_M, STREAM_L0_K],
                                                      target_memory=pl.MemorySpace.Left)
                                b_l0 = pl.tile.extract(b_l1, k_inner, 0, [STREAM_L0_K, STREAM_N],
                                                      target_memory=pl.MemorySpace.Right)
                                updated = pl.tile.matmul_acc(inner_acc, a_l0, b_l0,
                                                            init_cond=(outer == 0 and inner == 0))
                                inner_done = pl.yield_(updated)
                            outer_done = pl.yield_(inner_done)
                        q_proj_i32 = pl.store(outer_done, [t0, col_local], q_proj_i32)
            dq_tid = q_proj_q_dequant_stream(
                wq_b_scale,
                rope_cos_il,
                rope_sin_signed,
                rope_swap_idx,
                q,
                qr_scale_pad_store,
                q_proj_i32,
                tile_base,
                tile_rows,
                head_base,
            )
            q_ready[0] = dq_tid
        with pl.scope():
            head_base = 16
            q_proj_i32 = pl.create_tensor([matrix_rows, STREAM_HEADS * HEAD_DIM], dtype=pl.INT32)
            with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep, q_ready[1]]):
                pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
                worker = pl.tile.get_block_idx()
                n_blocks = STREAM_HEADS * HEAD_DIM // STREAM_N
                for step in pl.range((n_blocks - worker + STREAM_CUBE_WORKERS - 1) // STREAM_CUBE_WORKERS):
                    col_local = (worker + step * STREAM_CUBE_WORKERS) * STREAM_N
                    col = 8192 + col_local
                    for t0 in pl.range(0, matrix_rows, STREAM_M):
                        count = pl.min(STREAM_M, tile_rows - t0)
                        seed_storage = pl.create_tile([STREAM_M, STREAM_N], dtype=pl.INT32,
                                                      target_memory=pl.MemorySpace.Acc, compact=True)
                        seed = pl.tile.set_validshape(seed_storage, count, STREAM_N)
                        for outer, (outer_acc,) in pl.pipeline(0, Q_LORA // STREAM_L1_K, stage=2, init_values=(seed,)):
                            k_base = outer * STREAM_L1_K
                            a_l1 = pl.load(qr_i8_matmul, [t0, k_base], [STREAM_M, STREAM_L1_K],
                                           valid_shape=[count, STREAM_L1_K], target_memory=pl.MemorySpace.Mat)
                            b_l1 = pl.load(wq_b, [k_base, col], [STREAM_L1_K, STREAM_N],
                                           target_memory=pl.MemorySpace.Mat)
                            for inner, (inner_acc,) in pl.pipeline(
                                0, STREAM_L1_K // STREAM_L0_K, stage=2, init_values=(outer_acc,),
                            ):
                                k_inner = inner * STREAM_L0_K
                                a_l0 = pl.tile.extract(a_l1, 0, k_inner, [STREAM_M, STREAM_L0_K],
                                                      target_memory=pl.MemorySpace.Left)
                                b_l0 = pl.tile.extract(b_l1, k_inner, 0, [STREAM_L0_K, STREAM_N],
                                                      target_memory=pl.MemorySpace.Right)
                                updated = pl.tile.matmul_acc(inner_acc, a_l0, b_l0,
                                                            init_cond=(outer == 0 and inner == 0))
                                inner_done = pl.yield_(updated)
                            outer_done = pl.yield_(inner_done)
                        q_proj_i32 = pl.store(outer_done, [t0, col_local], q_proj_i32)
            dq_tid = q_proj_q_dequant_stream(
                wq_b_scale,
                rope_cos_il,
                rope_sin_signed,
                rope_swap_idx,
                q,
                qr_scale_pad_store,
                q_proj_i32,
                tile_base,
                tile_rows,
                head_base,
            )
            q_ready[1] = dq_tid
        with pl.scope():
            head_base = 32
            q_proj_i32 = pl.create_tensor([matrix_rows, STREAM_HEADS * HEAD_DIM], dtype=pl.INT32)
            with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep, q_ready[2]]):
                pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
                worker = pl.tile.get_block_idx()
                n_blocks = STREAM_HEADS * HEAD_DIM // STREAM_N
                for step in pl.range((n_blocks - worker + STREAM_CUBE_WORKERS - 1) // STREAM_CUBE_WORKERS):
                    col_local = (worker + step * STREAM_CUBE_WORKERS) * STREAM_N
                    col = 16384 + col_local
                    for t0 in pl.range(0, matrix_rows, STREAM_M):
                        count = pl.min(STREAM_M, tile_rows - t0)
                        seed_storage = pl.create_tile([STREAM_M, STREAM_N], dtype=pl.INT32,
                                                      target_memory=pl.MemorySpace.Acc, compact=True)
                        seed = pl.tile.set_validshape(seed_storage, count, STREAM_N)
                        for outer, (outer_acc,) in pl.pipeline(0, Q_LORA // STREAM_L1_K, stage=2, init_values=(seed,)):
                            k_base = outer * STREAM_L1_K
                            a_l1 = pl.load(qr_i8_matmul, [t0, k_base], [STREAM_M, STREAM_L1_K],
                                           valid_shape=[count, STREAM_L1_K], target_memory=pl.MemorySpace.Mat)
                            b_l1 = pl.load(wq_b, [k_base, col], [STREAM_L1_K, STREAM_N],
                                           target_memory=pl.MemorySpace.Mat)
                            for inner, (inner_acc,) in pl.pipeline(
                                0, STREAM_L1_K // STREAM_L0_K, stage=2, init_values=(outer_acc,),
                            ):
                                k_inner = inner * STREAM_L0_K
                                a_l0 = pl.tile.extract(a_l1, 0, k_inner, [STREAM_M, STREAM_L0_K],
                                                      target_memory=pl.MemorySpace.Left)
                                b_l0 = pl.tile.extract(b_l1, k_inner, 0, [STREAM_L0_K, STREAM_N],
                                                      target_memory=pl.MemorySpace.Right)
                                updated = pl.tile.matmul_acc(inner_acc, a_l0, b_l0,
                                                            init_cond=(outer == 0 and inner == 0))
                                inner_done = pl.yield_(updated)
                            outer_done = pl.yield_(inner_done)
                        q_proj_i32 = pl.store(outer_done, [t0, col_local], q_proj_i32)
            dq_tid = q_proj_q_dequant_stream(
                wq_b_scale,
                rope_cos_il,
                rope_sin_signed,
                rope_swap_idx,
                q,
                qr_scale_pad_store,
                q_proj_i32,
                tile_base,
                tile_rows,
                head_base,
            )
            q_ready[2] = dq_tid
        with pl.scope():
            head_base = 48
            q_proj_i32 = pl.create_tensor([matrix_rows, STREAM_HEADS * HEAD_DIM], dtype=pl.INT32)
            with pl.spmd(STREAM_CUBE_WORKERS, name_hint="hca_qb_stream", deps=[qproj_dep, q_ready[3]]):
                pl.set_cache_policy(wq_b, pl.CachePolicy.BYPASS)
                worker = pl.tile.get_block_idx()
                n_blocks = STREAM_HEADS * HEAD_DIM // STREAM_N
                for step in pl.range((n_blocks - worker + STREAM_CUBE_WORKERS - 1) // STREAM_CUBE_WORKERS):
                    col_local = (worker + step * STREAM_CUBE_WORKERS) * STREAM_N
                    col = 24576 + col_local
                    for t0 in pl.range(0, matrix_rows, STREAM_M):
                        count = pl.min(STREAM_M, tile_rows - t0)
                        seed_storage = pl.create_tile([STREAM_M, STREAM_N], dtype=pl.INT32,
                                                      target_memory=pl.MemorySpace.Acc, compact=True)
                        seed = pl.tile.set_validshape(seed_storage, count, STREAM_N)
                        for outer, (outer_acc,) in pl.pipeline(0, Q_LORA // STREAM_L1_K, stage=2, init_values=(seed,)):
                            k_base = outer * STREAM_L1_K
                            a_l1 = pl.load(qr_i8_matmul, [t0, k_base], [STREAM_M, STREAM_L1_K],
                                           valid_shape=[count, STREAM_L1_K], target_memory=pl.MemorySpace.Mat)
                            b_l1 = pl.load(wq_b, [k_base, col], [STREAM_L1_K, STREAM_N],
                                           target_memory=pl.MemorySpace.Mat)
                            for inner, (inner_acc,) in pl.pipeline(
                                0, STREAM_L1_K // STREAM_L0_K, stage=2, init_values=(outer_acc,),
                            ):
                                k_inner = inner * STREAM_L0_K
                                a_l0 = pl.tile.extract(a_l1, 0, k_inner, [STREAM_M, STREAM_L0_K],
                                                      target_memory=pl.MemorySpace.Left)
                                b_l0 = pl.tile.extract(b_l1, k_inner, 0, [STREAM_L0_K, STREAM_N],
                                                      target_memory=pl.MemorySpace.Right)
                                updated = pl.tile.matmul_acc(inner_acc, a_l0, b_l0,
                                                            init_cond=(outer == 0 and inner == 0))
                                inner_done = pl.yield_(updated)
                            outer_done = pl.yield_(inner_done)
                        q_proj_i32 = pl.store(outer_done, [t0, col_local], q_proj_i32)
            dq_tid = q_proj_q_dequant_stream(
                wq_b_scale,
                rope_cos_il,
                rope_sin_signed,
                rope_swap_idx,
                q,
                qr_scale_pad_store,
                q_proj_i32,
                tile_base,
                tile_rows,
                head_base,
            )
            q_ready[3] = dq_tid
    return q



@pl.jit.inline(auto_scope=False)
def qkv_proj_rope_tall(
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
    """沿用原Q_A/QR/KV链，只替换Q_B分块；所有跨任务依赖保留。"""
    tokens = pl.tensor.dim(x, 0)
    signed_sin = pl.create_tensor([tokens, ROPE_DIM], dtype=pl.FP32)
    swap_index = pl.create_tensor([tokens, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, signed_sin, swap_index)
    qr_i8 = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    qr_scales = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
    qa_tid = q_proj_qr(x, wq_a, gamma_cq, qr, qr_scale, qr_i8, qr_scales)
    ready = pl.system.task_dummy(deps=[])
    # 与公共QKV入口一致：四组任务写入传入的manual_dep缓冲，以q_ready发布。
    _q_proj_q_tall(
        x, wq_b, wq_b_scale, rope_cos, signed_sin, swap_index,
        q, qr_i8, qr_scales, ready, q_ready,
    )
    kv_proj_rope(x, wkv, gamma_ckv, rope_cos, signed_sin, swap_index, kv, late_dep)
    return q, qa_tid
