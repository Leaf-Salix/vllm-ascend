# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""保留nalinaly的KV M32所有权和数学，仅恢复WKV/Norm的Native BF16边界。

独立候选；归约、sqrt/div和RoPE顺序仍沿用原helper，须先同输入验证。
"""

import pypto.language as pl

from .kv_stages import (
    EPS,
    HEAD_DIM,
    KV_RMS_T_TILE,
    KV_TILE,
    NOPE_DIM,
    PAD_ROWS_DYN,
    ROPE_DIM,
    ROPE_DIM_SCALE,
    T_DYN,
)


@pl.jit.inline(auto_scope=False)
def kv_native_norm_rope_stage(
    kv_fp32: pl.Tensor[[PAD_ROWS_DYN, HEAD_DIM], pl.FP32],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """AIV 独占 32 行的全部 512 列；调用前由外层等齐投影和 RoPE 输入。

    本 helper 内没有 wait。完整块先对 512 列做同一 RMS，再分别处理 NOPE
    和 RoPE；尾块走显式 load/store 的有效行路径，不访问 pad16 之外的 GM。
    """
    tokens = pl.tensor.dim(kv, 0)
    token_tiles = (tokens + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
    # 原512行分段边界与32行tile对齐，因此扁平化此循环不改变完整/尾块数学。
    # 这是worker条带循环，不派发任务；空T和未分到tile的核不读gamma/RoPE。
    for tg_idx in pl.range(worker_id, token_tiles, WORKERS):
        tg = tg_idx * KV_RMS_T_TILE
        valid_rows = pl.min(KV_RMS_T_TILE, tokens - tg)
        if valid_rows == KV_RMS_T_TILE:
            # 八段64列先按lane累加，再一次row_sum，保持Native的归约树。
            kv_sq_lanes = pl.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
            for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
                kv_chunk = pl.cast(pl.cast(kv_chunk, pl.BF16, mode="rint"), pl.FP32)
                kv_sq = pl.mul(kv_chunk, kv_chunk)
                kv_sq_lanes = pl.add(kv_sq_lanes, kv_sq)
            kv_sq_sum = pl.reshape(pl.row_sum(kv_sq_lanes), [1, KV_RMS_T_TILE])
            # 保留sqrt后用1除，不替换为rsqrt/recip，以免移动舍入边界。
            kv_rms = pl.sqrt(pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS))
            kv_inv_rms = pl.div(pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms)
            kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])
            for n0 in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
                kv_chunk = pl.cast(pl.cast(kv_chunk, pl.BF16, mode="rint"), pl.FP32)
                gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
                gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
                kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
                kv[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16
            gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
            gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
            kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
            kv_rope_chunk = pl.cast(pl.cast(kv_rope_chunk, pl.BF16, mode="rint"), pl.FP32)
            kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
            kv_rope_norm_chunk = pl.cast(pl.cast(kv_rope_norm_chunk, pl.BF16, mode="rint"), pl.FP32)
            kv_cos_il_full = rope_cos_il[tg : tg + KV_RMS_T_TILE, :]
            kv_sin_signed_full = rope_sin_signed[tg : tg + KV_RMS_T_TILE, :]
            kv_swap_idx_full = rope_swap_idx[tg : tg + KV_RMS_T_TILE, :]
            kv_swapped_full = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx_full)
            kv_rope_rot_full = pl.add(
                pl.mul(kv_rope_norm_chunk, kv_cos_il_full), pl.mul(kv_swapped_full, kv_sin_signed_full)
            )
            kv_rope_i16_full = pl.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
            kv[tg : tg + KV_RMS_T_TILE, NOPE_DIM:HEAD_DIM] = kv_rope_i16_full
        else:
            # 最后不足 32 行时，pad16 可能只剩 16 行，不能直接读完整 32 行
            # GM 窗口。每次 load/store 都带 valid_rows；无效 lane 的中间数值
            # 不属于模型结果，也不能用来推断 padding 已经初始化。
            kv_reduce_tmp = pl.create_tile([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            kv_sq_lanes_tail = pl.tile.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
            for kv_sq_col0_tail in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                kv_chunk_tail = pl.load(
                    kv_fp32,
                    [tg, kv_sq_col0_tail],
                    [KV_RMS_T_TILE, KV_TILE],
                    valid_shape=[valid_rows, KV_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                kv_chunk_tail = pl.cast(pl.cast(kv_chunk_tail, pl.BF16, mode="rint"), pl.FP32)
                kv_sq_tail = pl.mul(kv_chunk_tail, kv_chunk_tail)
                kv_sq_lanes_tail = pl.add(kv_sq_lanes_tail, kv_sq_tail)
            kv_sq_sum_tail = pl.reshape(pl.row_sum(kv_sq_lanes_tail, kv_reduce_tmp), [1, KV_RMS_T_TILE])
            kv_rms_tail = pl.sqrt(pl.add(pl.mul(kv_sq_sum_tail, 1.0 / HEAD_DIM), EPS))
            kv_inv_rms_tail = pl.div(pl.tile.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_rms_tail)
            kv_inv_rms_t_tail = pl.reshape(kv_inv_rms_tail, [KV_RMS_T_TILE, 1])
            for n0_tail in pl.pipeline(0, NOPE_DIM, KV_TILE, stage=2):
                kv_chunk_tail = pl.load(
                    kv_fp32,
                    [tg, n0_tail],
                    [KV_RMS_T_TILE, KV_TILE],
                    valid_shape=[valid_rows, KV_TILE],
                    target_memory=pl.MemorySpace.Vec,
                )
                kv_chunk_tail = pl.cast(pl.cast(kv_chunk_tail, pl.BF16, mode="rint"), pl.FP32)
                gamma_kv_input_tail = pl.load(gamma_ckv, [n0_tail], [KV_TILE], target_memory=pl.MemorySpace.Vec)
                gamma_kv_cast_tail = pl.cast(gamma_kv_input_tail, target_type=pl.FP32)
                gamma_kv_chunk_tail = pl.reshape(gamma_kv_cast_tail, [1, KV_TILE])
                kv_normed_tail = pl.col_expand_mul(
                    pl.row_expand_mul(kv_chunk_tail, kv_inv_rms_t_tail), gamma_kv_chunk_tail
                )
                kv_normed_bf16_tail = pl.cast(kv_normed_tail, target_type=pl.BF16, mode="rint")
                kv_normed_valid = pl.set_validshape(kv_normed_bf16_tail, valid_rows, KV_TILE)
                pl.store(kv_normed_valid, [tg, n0_tail], kv)
            gamma_rope_input_tail = pl.load(gamma_ckv, [NOPE_DIM], [ROPE_DIM], target_memory=pl.MemorySpace.Vec)
            gamma_rope_cast_tail = pl.cast(gamma_rope_input_tail, target_type=pl.FP32)
            gamma_rope_tail = pl.reshape(gamma_rope_cast_tail, [1, ROPE_DIM])
            kv_rope_chunk_tail = pl.load(
                kv_fp32,
                [tg, NOPE_DIM],
                [KV_RMS_T_TILE, ROPE_DIM],
                valid_shape=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            kv_rope_chunk_tail = pl.cast(pl.cast(kv_rope_chunk_tail, pl.BF16, mode="rint"), pl.FP32)
            kv_rope_norm_tail = pl.col_expand_mul(
                pl.row_expand_mul(kv_rope_chunk_tail, kv_inv_rms_t_tail), gamma_rope_tail
            )
            kv_rope_norm_tail = pl.cast(pl.cast(kv_rope_norm_tail, pl.BF16, mode="rint"), pl.FP32)
            kv_cos_il_tail = pl.load(
                rope_cos_il,
                [tg, 0],
                [KV_RMS_T_TILE, ROPE_DIM],
                valid_shape=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            kv_sin_signed_tail = pl.load(
                rope_sin_signed,
                [tg, 0],
                [KV_RMS_T_TILE, ROPE_DIM],
                valid_shape=[valid_rows, ROPE_DIM],
                target_memory=pl.MemorySpace.Vec,
            )
            kv_col = pl.col_expand_mul(
                pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
            )
            # pair=floor(col/2)，相邻交换为(1,0,3,2,...)；只改变索引，不改RoPE算术。
            kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
            kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
            kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))
            # tile.gather按整个Tile展开寻址，补上每行64列偏移，防止跨行取错。
            kv_row_seed = pl.mul(
                pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32), target_type=pl.FP32), ROPE_DIM_SCALE
            )
            kv_row_grid = pl.col_expand_mul(
                pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_row_seed
            )
            kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
            kv_swap_idx_tail = pl.cast(pl.add(kv_swap_f, kv_row_offset), target_type=pl.INT32)
            kv_gather_tmp = pl.create_tile([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
            kv_swapped_tail = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
            kv_rope_rot_tail = pl.add(
                pl.mul(kv_rope_norm_tail, kv_cos_il_tail), pl.mul(kv_swapped_tail, kv_sin_signed_tail)
            )
            # BF16舍入仍在RoPE两项乘加之后，只有有效行写回最终kv。
            kv_rope_i16_tail = pl.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
            kv_rope_valid = pl.set_validshape(kv_rope_i16_tail, valid_rows, ROPE_DIM)
            pl.store(kv_rope_valid, [tg, NOPE_DIM], kv)
    return kv
