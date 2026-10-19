# SPDX-License-Identifier: Apache-2.0
"""Remove only dead forwarded weight arguments; retain the Native-order math."""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.qkv_proj_rope import (  # noqa: F401
    BF16_WEIGHT_LAYOUT,
    EPS,
    HEAD_DIM,
    KV_M_TILE,
    KV_OK,
    KV_RMS_T_TILE,
    KV_T_DYN,
    KV_TILE,
    MATMUL_T_TILE,
    NOPE_DIM,
    PREFILL_DENSE_TILE,
    Q_LORA,
    QR_OK,
    QUANT_WEIGHT_LAYOUT,
    ROPE_DIM,
    ROPE_DIM_SCALE,
    T_DYN,
    D,
    H,
    q_proj_rope,
    rope_prepare,
)


@pl.jit.inline(auto_scope=False)
def kv_proj_rope(
    x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_il: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[KV_T_DYN, ROPE_DIM], pl.INT32],
    kv: pl.Tensor[[KV_T_DYN, HEAD_DIM], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
    native_kv_projected: pl.Tensor[[KV_T_DYN, HEAD_DIM], pl.BF16],
):
    """消费 Native BF16 投影，按原顺序执行 TND KV RMSNorm/RoPE。"""
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            t_matmul = (tile_rows + MATMUL_T_TILE - 1) // MATMUL_T_TILE * MATMUL_T_TILE
            kv_fp32 = pl.create_tensor([t_matmul, HEAD_DIM], dtype=pl.FP32)
            copy_blocks = (tile_rows + KV_M_TILE - 1) // KV_M_TILE
            with pl.spmd(copy_blocks, name_hint="kv_native_projection", deps=[late_dep]):
                copy_row = pl.tile.get_block_idx() * KV_M_TILE
                copy_rows = pl.min(KV_M_TILE, tile_rows - copy_row)
                projected = pl.load(
                    native_kv_projected,
                    [tile_base + copy_row, 0],
                    [KV_M_TILE, HEAD_DIM],
                    valid_shape=[copy_rows, HEAD_DIM],
                )
                pl.store(pl.cast(projected, pl.FP32), [copy_row, 0], kv_fp32)
            kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
            kv_token_tiles = (tile_rows + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE
            for tg_idx in pl.spmd(kv_token_tiles, name_hint="kv_rms_norm_rope", sync_start=True):
                tg = tg_idx * KV_RMS_T_TILE
                valid_rows = pl.min(KV_RMS_T_TILE, tile_rows - tg)
                out_tg = tile_base + tg
                if valid_rows == KV_RMS_T_TILE:
                    kv_sq_lanes = pl.full([KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, value=0.0)
                    for kv_sq_col0 in pl.pipeline(0, HEAD_DIM, KV_TILE, stage=2):
                        kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, kv_sq_col0 : kv_sq_col0 + KV_TILE]
                        kv_chunk = pl.cast(pl.cast(kv_chunk, pl.BF16, mode="rint"), pl.FP32)
                        kv_sq = pl.mul(kv_chunk, kv_chunk)
                        kv_sq_lanes = pl.add(kv_sq_lanes, kv_sq)
                    kv_sq_sum = pl.reshape(pl.row_sum(kv_sq_lanes), [1, KV_RMS_T_TILE])
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
                        kv_view[out_tg : out_tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = kv_normed_bf16
                    gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
                    gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
                    kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
                    kv_rope_chunk = pl.cast(pl.cast(kv_rope_chunk, pl.BF16, mode="rint"), pl.FP32)
                    kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
                    kv_rope_norm_chunk = pl.cast(pl.cast(kv_rope_norm_chunk, pl.BF16, mode="rint"), pl.FP32)
                    kv_cos_il_full = rope_cos_il[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_sin_signed_full = rope_sin_signed[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swap_idx_full = rope_swap_idx[out_tg : out_tg + KV_RMS_T_TILE, :]
                    kv_swapped_full = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx_full)
                    kv_rope_rot_full = pl.add(
                        pl.mul(kv_rope_norm_chunk, kv_cos_il_full), pl.mul(kv_swapped_full, kv_sin_signed_full)
                    )
                    kv_rope_i16_full = pl.cast(kv_rope_rot_full, target_type=pl.BF16, mode="rint")
                    kv_view[out_tg : out_tg + KV_RMS_T_TILE, NOPE_DIM:HEAD_DIM] = kv_rope_i16_full
                else:
                    kv_reduce_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, KV_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                    )
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
                        pl.store(kv_normed_valid, [out_tg, n0_tail], kv_view)
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
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_sin_signed_tail = pl.load(
                        rope_sin_signed,
                        [out_tg, 0],
                        [KV_RMS_T_TILE, ROPE_DIM],
                        valid_shape=[valid_rows, ROPE_DIM],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    kv_col = pl.col_expand_mul(
                        pl.tile.full([KV_RMS_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
                        pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                    )
                    kv_dup_f = pl.cast(pl.cast(pl.mul(kv_col, 0.5), target_type=pl.INT32, mode="trunc"), pl.FP32)
                    kv_lane = pl.sub(kv_col, pl.mul(kv_dup_f, 2.0))
                    kv_swap_f = pl.sub(pl.add(kv_col, 1.0), pl.mul(kv_lane, 2.0))
                    kv_row_seed = pl.mul(
                        pl.cast(pl.tile.arange(0, [1, KV_RMS_T_TILE], dtype=pl.INT32), target_type=pl.FP32),
                        ROPE_DIM_SCALE,
                    )
                    kv_row_grid = pl.col_expand_mul(
                        pl.tile.full([ROPE_DIM, KV_RMS_T_TILE], dtype=pl.FP32, value=1.0), kv_row_seed
                    )
                    kv_row_offset = pl.transpose(kv_row_grid, axis1=0, axis2=1)
                    kv_swap_idx_tail = pl.cast(pl.add(kv_swap_f, kv_row_offset), target_type=pl.INT32)
                    kv_gather_tmp = pl.create_tile(
                        [KV_RMS_T_TILE, ROPE_DIM], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec
                    )
                    kv_swapped_tail = pl.tile.gather(kv_rope_norm_tail, kv_swap_idx_tail, kv_gather_tmp)
                    kv_rope_rot_tail = pl.add(
                        pl.mul(kv_rope_norm_tail, kv_cos_il_tail), pl.mul(kv_swapped_tail, kv_sin_signed_tail)
                    )
                    kv_rope_i16_tail = pl.cast(kv_rope_rot_tail, target_type=pl.BF16, mode="rint")
                    kv_rope_valid = pl.set_validshape(kv_rope_i16_tail, valid_rows, ROPE_DIM)
                    pl.store(kv_rope_valid, [out_tg, NOPE_DIM], kv_view)


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
    native_kv_projected: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
):
    """Q 投影和 Native KV 后处理共用真实 TND 行数与 RoPE 表。"""
    t_dim = pl.tensor.dim(x, 0)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, q_rope_sin_signed, q_rope_swap_idx)
    q_proj_rope(x, wq_a, wq_b, wq_b_scale, gamma_cq, rope_cos, q_rope_sin_signed, q_rope_swap_idx, q, qr, qr_scale)
    kv_proj_rope(x, gamma_ckv, rope_cos, q_rope_sin_signed, q_rope_swap_idx, kv, late_dep, native_kv_projected)
    projection_marker = pl.system.task_dummy(deps=[late_dep])
    return (q, projection_marker)
