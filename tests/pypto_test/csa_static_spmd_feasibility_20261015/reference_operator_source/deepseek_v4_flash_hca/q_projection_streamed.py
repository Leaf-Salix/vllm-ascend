# SPDX-License-Identifier: Apache-2.0
"""Q-B每16个head独立发布，以分层双缓冲保留原整数结果和逐行数值策略。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.nz_mode import QUANT_WEIGHT_LAYOUT
from ..deepseek_v4_flash_csa.qkv_proj_rope import (
    EPS,
    HEAD_DIM,
    NOPE_DIM,
    PREFILL_DENSE_TILE,
    Q_LORA,
    Q_ROPE_H_TILE,
    Q_ROPE_T_TILE,
    QPROJ_MM_T_DYN,
    QPROJ_T_PAD,
    ROPE_DIM,
    ROPE_DIM_SCALE,
    T_DYN,
    D,
    H,
)

STREAM_GROUPS = 4
STREAM_HEADS = H // STREAM_GROUPS
STREAM_CUBE_WORKERS = 16 // STREAM_GROUPS
STREAM_VEC_WORKERS = 48 // STREAM_GROUPS
STREAM_M = 128
STREAM_N = 256
# 对齐Native A3 MDL：L1搬运K512，L0以K128双缓冲，保留N256和四组交接。
# compact Acc匹配动态有效行，是当前PyPTO内部接口；B4/补位验证见
# tests/pypto_test/hca_native_five_20260930/README.md。
STREAM_L1_K = 512
STREAM_L0_K = 128



@pl.jit.inline(auto_scope=False)
def q_proj_q_dequant_stream(
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    q_proj_i32: pl.Tensor[[QPROJ_MM_T_DYN, STREAM_HEADS * HEAD_DIM], pl.INT32],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
    head_base: pl.Scalar[pl.INDEX],
):
    """Dequantize, normalize, and rotate one projected Q tile."""
    q_flat = q
    with pl.spmd(
        STREAM_VEC_WORKERS,
        name_hint="qproj_dequant_rms_nope_rope",
        allow_early_resolve=True,
    ) as dq_tid:
        dq_worker = pl.tile.get_block_idx()
        # 按本档位的 token 数自适应放大 head 块（上游写法）。恒用 Q_ROPE_H_TILE=4 时
        # Vector 块太小、指令条数多，in-core 实测 VECTOR 占 82%。生产档位 tile_rows=96
        # 正好命中下面第二个分支，head 块 4 -> 16。
        dq_head_tile = Q_ROPE_H_TILE
        if tile_rows >= STREAM_VEC_WORKERS * Q_ROPE_T_TILE and tile_rows % (STREAM_VEC_WORKERS * Q_ROPE_T_TILE) == 0:
            dq_head_tile = STREAM_HEADS
        elif (
            tile_rows >= STREAM_VEC_WORKERS * Q_ROPE_T_TILE * 16 // STREAM_HEADS
            and tile_rows % (STREAM_VEC_WORKERS * Q_ROPE_T_TILE * 16 // STREAM_HEADS) == 0
        ):
            dq_head_tile = 16
        for dq_work in pl.range(
            dq_worker,
            ((tile_rows + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE) * (STREAM_HEADS // dq_head_tile),
            STREAM_VEC_WORKERS,
        ):
            hg = (dq_work % (STREAM_HEADS // dq_head_tile)) * dq_head_tile
            tg = (dq_work // (STREAM_HEADS // dq_head_tile)) * Q_ROPE_T_TILE
            out_tg = tile_base + tg
            if tg + Q_ROPE_T_TILE <= tile_rows:
                qr_scale_dq_t = qr_scale_pad_store[tg : tg + Q_ROPE_T_TILE, :]
                q_cos_il = rope_cos_il[out_tg : out_tg + Q_ROPE_T_TILE, :]
                q_sin_signed = rope_sin_signed[out_tg : out_tg + Q_ROPE_T_TILE, :]
                q_swap_idx = rope_swap_idx[out_tg : out_tg + Q_ROPE_T_TILE, :]
                # Flat gather handles all eight rows in one vector operation.
                # Axis gather lowers to a scalar loop with one gather per row.
                q_gather_row_seed = pl.mul(
                    pl.cast(pl.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), pl.FP32),
                    ROPE_DIM_SCALE,
                )
                q_gather_row_grid = pl.col_expand_mul(
                    pl.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0),
                    q_gather_row_seed,
                )
                q_gather_row_offsets = pl.cast(
                    pl.transpose(q_gather_row_grid, axis1=0, axis2=1),
                    pl.INT32,
                )
                q_swap_flat_idx = pl.add(q_swap_idx, q_gather_row_offsets)
                for h_inner in pl.pipeline(dq_head_tile, stage=2):
                    h = hg + h_inner
                    source_h0 = h * HEAD_DIM
                    h0 = head_base * HEAD_DIM + source_h0
                    q_head_acc = q_proj_i32[tg : tg + Q_ROPE_T_TILE, source_h0 : source_h0 + HEAD_DIM]
                    q_head_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM], [1, HEAD_DIM])
                    q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                    q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                    q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                    q_head_sq = pl.mul(q_head_dq, q_head_dq)
                    q_head_sq_row = pl.row_sum(q_head_sq)
                    q_head_sq_sum = pl.reshape(q_head_sq_row, [1, Q_ROPE_T_TILE])
                    q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                    q_head_var = pl.add(q_head_sq_mean, EPS)
                    q_head_inv_rms = pl.rsqrt(q_head_var, high_precision=True)
                    q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

                    q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                    q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = q_nope_bf16

                    q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                    q_rope_swapped = pl.gather(q_rope_chunk, index=q_swap_flat_idx)
                    q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                    q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                    q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                    q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                    q_flat[out_tg : out_tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + HEAD_DIM] = q_rope_bf16
            else:
                valid_tail_rows = tile_rows - tg
                # q_proj and its dequant scale are padded to the cube boundary.
                # Keep the math on a static eight-row tile and crop public stores.
                qr_scale_dq_tail = pl.load(
                    qr_scale_pad_store,
                    [tg, 0],
                    [Q_ROPE_T_TILE, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_cos_il_tail = pl.load(
                    rope_cos_il,
                    [out_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[valid_tail_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_sin_signed_tail = pl.load(
                    rope_sin_signed,
                    [out_tg, 0],
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    valid_shape=[valid_tail_rows, ROPE_DIM],
                    target_memory=pl.MemorySpace.Vec,
                )
                q_col = pl.col_expand_mul(
                    pl.tile.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                    pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                q_dup_f = pl.cast(pl.cast(pl.mul(q_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
                q_lane = pl.sub(q_col, pl.mul(q_dup_f, 2.0))
                q_swap_f = pl.sub(pl.add(q_col, 1.0), pl.mul(q_lane, 2.0))
                # Row-major flattening offsets stay in fp32: col-expand is defined
                # for half / float only.
                q_row_seed = pl.mul(
                    pl.cast(pl.tile.arange(0, [1, Q_ROPE_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                    ROPE_DIM_SCALE,
                )
                q_row_grid = pl.col_expand_mul(
                    pl.tile.full([ROPE_DIM, Q_ROPE_T_TILE], dtype=pl.FP32, value=1.0),
                    q_row_seed,
                )
                q_row_offset = pl.transpose(q_row_grid, axis1=0, axis2=1)
                q_swap_idx_tail = pl.cast(pl.add(q_swap_f, q_row_offset), target_type=pl.INT32)
                q_head_reduce_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, HEAD_DIM],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                q_gather_tmp = pl.create_tile(
                    [Q_ROPE_T_TILE, ROPE_DIM],
                    dtype=pl.INT32,
                    target_memory=pl.MemorySpace.Vec,
                )
                for h_inner_tail in pl.range(Q_ROPE_H_TILE):
                    h_tail = hg + h_inner_tail
                    source_h0_tail = h_tail * HEAD_DIM
                    h0_tail = head_base * HEAD_DIM + source_h0_tail
                    q_head_acc_tail = pl.load(
                        q_proj_i32,
                        [tg, source_h0_tail],
                        [Q_ROPE_T_TILE, HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale_input_tail = pl.load(
                        wq_b_scale,
                        [h0_tail],
                        [HEAD_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_head_scale_tail = pl.reshape(q_head_scale_input_tail, [1, HEAD_DIM])
                    q_head_acc_fp32_tail = pl.cast(q_head_acc_tail, target_type=pl.FP32, mode="none")
                    q_head_row_scaled_tail = pl.row_expand_mul(q_head_acc_fp32_tail, qr_scale_dq_tail)
                    q_head_dq_tail = pl.col_expand_mul(q_head_row_scaled_tail, q_head_scale_tail)

                    q_head_sq_tail = pl.mul(q_head_dq_tail, q_head_dq_tail)
                    q_head_sq_sum_tail = pl.row_sum(q_head_sq_tail, q_head_reduce_tmp)
                    q_head_inv_rms_tail = pl.recip(
                        pl.sqrt(pl.add(pl.mul(q_head_sq_sum_tail, 1.0 / HEAD_DIM), EPS)),
                    )

                    q_nope_normed_tail = pl.row_expand_mul(q_head_dq_tail[:, 0:NOPE_DIM], q_head_inv_rms_tail)
                    q_nope_bf16_tail = pl.cast(q_nope_normed_tail, target_type=pl.BF16, mode="rint")
                    q_nope_valid = pl.set_validshape(q_nope_bf16_tail, valid_tail_rows, NOPE_DIM)
                    pl.store(q_nope_valid, [out_tg, h0_tail], q_flat)

                    q_rope_chunk_raw_tail = q_head_dq_tail[:, NOPE_DIM:HEAD_DIM]
                    q_rope_chunk_tail = pl.row_expand_mul(q_rope_chunk_raw_tail, q_head_inv_rms_tail)
                    q_rope_swapped_tail = pl.tile.gather(q_rope_chunk_tail, q_swap_idx_tail, q_gather_tmp)
                    # RoPE inputs contain only the live rows of this tail.
                    q_rope_chunk_tail = pl.set_validshape(q_rope_chunk_tail, valid_tail_rows, ROPE_DIM)
                    q_rope_swapped_tail = pl.set_validshape(q_rope_swapped_tail, valid_tail_rows, ROPE_DIM)
                    q_rope_base_tail = pl.mul(q_rope_chunk_tail, q_cos_il_tail)
                    q_rope_delta_tail = pl.mul(q_rope_swapped_tail, q_sin_signed_tail)
                    q_rope_rot_tail = pl.add(q_rope_base_tail, q_rope_delta_tail)
                    q_rope_bf16_tail = pl.cast(q_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    q_rope_valid = pl.set_validshape(q_rope_bf16_tail, valid_tail_rows, ROPE_DIM)
                    pl.store(q_rope_valid, [out_tg, h0_tail + NOPE_DIM], q_flat)
    return dq_tid


@pl.jit.inline(auto_scope=False)
def q_proj_q_streamed(
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
