# SPDX-License-Identifier: Apache-2.0
"""两条独立前段链的局部流水，不合并整个CSA。

Indexer默认8 AIC/16 AIV，主压缩/Q_B默认16 AIC/32 AIV，共同填满硬件。
POOL阶段必须先发布再写state，避免覆盖其它角色尚未读完的历史；Norm只读本owner pooled。
Indexer scale保留48个logical page writer，当前AIV按条带接管，维持每页唯一64B RMW。
信号首次清零、各组顺序独占，每角色每调用推进一次代际；数值/图/性能以设备结果为准。
"""

import pypto.language as pl

from .cache_stages import original_kv_writeback_stage
from .compressor_output_stages import compressor_norm_rope_cache_stage
from .compressor_pool_stages import (
    compressor_pool_stage,
    compressor_state_commit_stage,
    indexer_compressor_pool_stage,
    indexer_compressor_state_commit_stage,
)
from .compressor_projection_stages import (
    CMP_M_TILE,
    CMP_N_TILE,
    CMP_OUT_DIM,
    INDEXER_M_TILE,
    INDEXER_N_TILE,
    INDEXER_OUT_DIM,
    compressor_project_stage,
    indexer_compressor_project_stage,
)
from .indexer_cache_stages import indexer_compact_quant_cache_stage
from .indexer_compressor_output_stages import HEAD_DIM as COMPACT_HEAD_DIM
from .indexer_compressor_output_stages import OUT_TILE as COMPACT_HAD_COL_TILE
from .indexer_compressor_output_stages import RMS_PAD_TILE as COMPACT_HAD_ROW_TILE
from .indexer_compressor_output_stages import indexer_compressor_hadamard_stage, indexer_compressor_norm_rope_stage
from .indexer_group_stages import indexer_head_coefficients_stage
from .indexer_query_stages import (
    DEQUANT_T_TILE,
    DQ_ROPE_H_TILE,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    MM_ROW_TILE,
    QR_MM_N_TILE,
    WEIGHTS_OK,
    indexer_query_dequant_rope_stage,
    indexer_query_dequant_rope_wide_stage,
    indexer_query_project_stage,
    indexer_query_quant_stage,
    indexer_weights_project_stage,
    indexer_weights_reduce_stage,
)
from .indexer_scale_plan_stages import indexer_scale_page_commit_stage
from .kv_stages import HEAD_DIM as KV_HEAD_DIM
from .kv_stages import KV_N_TILE, KV_OM, KV_RMS_T_TILE, kv_norm_rope_stage, kv_project_stage
from .query_group import _query_hadamard_four_tokens
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
from .query_projection_stages import (
    QPROJ_N_BLOCKS,
    QUANT_WEIGHT_LAYOUT,
    q_b_stage,
    q_dequant_rope_stage,
)
from .runtime_abi import (
    CACHE_ROWS,
    COEFF_COLS,
    COEFF_ROWS,
    GROUP_OFFSET_ROWS,
    IC,
    ICACHE_BYTES,
    ICACHE_PAGES,
    ICPAD,
    IDX_HEAD_ROWS,
    IDX_PROJ_ROWS,
    IDX_ROPE_HEAD_ROWS,
    IDX_ROPE_ROWS,
    IDX_SCALE_T_ROWS,
    IPOOL,
    IPROJ,
    IQ,
    ISCALES,
    ISLOTS,
    ISTATE_ELEMS,
    ISTATE_PAGES,
    ITABLE_COLS,
    KV_ROWS,
    MC,
    MCACHE_ROWS,
    MNORM,
    MPOOL,
    MPROJ,
    MQ,
    MSLOTS,
    MSTATE_ELEMS,
    MSTATE_PAGES,
    MTABLE_COLS,
    P8,
    QPROJ_ROWS,
    QR_ROWS,
    WEIGHTS_PARTIAL_ROWS,
    WEIGHTS_ROWS,
    B,
    T,
)

IDX_FRONT_WORDS = pl.dynamic("CSA_IDX_FRONT_WORDS")
MAIN_FRONT_WORDS = pl.dynamic("CSA_MAIN_FRONT_WORDS")
IDX_FRONT_PHASES = 11
MAIN_FRONT_PHASES = 6
SCALE_WRITERS = 48
I_PROJECT = 1
I_POOL = 2
I_NORM = 3
I_QUERY_PROJECT = 4
I_WEIGHTS = 5
I_HAD = 6
I_QUERY_ROPE = 7
I_QUERY_HAD = 8
I_QUANT = 9
I_QUERY_QUANT = 10
M_PROJECT = 1
M_POOL = 2
M_QUERY = 3


@pl.jit.incore
def index_front_cube(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    qr: pl.Tensor[[T, 1024], pl.INT8],
    positions: pl.Tensor[[T], pl.INT64],
    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_count: pl.Tensor[[1], pl.INT32],
    idx_normed: pl.Tensor[[ICPAD, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
    idx_query_rope_heads: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    idx_value: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_score: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_query_acc: pl.InOut[pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32]],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    idx_had: pl.InOut[pl.Tensor[[ICPAD, 128], pl.FP32]],
    idx_query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    idx_front_signal: pl.InOut[pl.Tensor[[IDX_FRONT_WORDS], pl.INT32]],
    FRONT_GROUPS: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    core = worker
    tokens = pl.tensor.dim(positions, 0)
    (idx_front_signal, generation) = begin_invocation(idx_front_signal, core)
    (idx_value, idx_score) = indexer_compressor_project_stage(
        normed, idx_wkv, idx_wgate, idx_value, idx_score, worker, FRONT_GROUPS
    )
    idx_front_signal = publish_phase(idx_front_signal, I_PROJECT, core, 3 * FRONT_GROUPS, generation)
    idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, worker, FRONT_GROUPS)
    idx_front_signal = publish_phase(idx_front_signal, I_QUERY_PROJECT, core, 3 * FRONT_GROUPS, generation)
    weights_partial = indexer_weights_project_stage(normed, weights_proj, weights_partial, worker, FRONT_GROUPS)
    idx_front_signal = publish_phase(idx_front_signal, I_WEIGHTS, core, 3 * FRONT_GROUPS, generation)
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_NORM,
        3 * FRONT_GROUPS,
        generation,
        0,
        pl.tensor.dim(idx_lengths, 0),
        1,
        2 * FRONT_GROUPS,
        FRONT_GROUPS,
        1,
    )
    compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
    idx_had = indexer_compressor_hadamard_stage(idx_normed, hadamard, idx_had, compact_count, worker, FRONT_GROUPS)
    idx_front_signal = publish_phase(idx_front_signal, I_HAD, core, 3 * FRONT_GROUPS, generation)
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_QUERY_ROPE,
        3 * FRONT_GROUPS,
        generation,
        0,
        (tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE * (IDX_N_HEADS // DQ_ROPE_H_TILE),
        1,
        2 * FRONT_GROUPS,
        FRONT_GROUPS,
        1,
    )
    idx_query_had = _query_hadamard_four_tokens(
        idx_query_rope_heads, hadamard, idx_query_had, tokens, worker, FRONT_GROUPS
    )
    idx_front_signal = publish_phase(idx_front_signal, I_QUERY_HAD, core, 3 * FRONT_GROUPS, generation)
    return (idx_value, idx_score, idx_query_acc, weights_partial, idx_had, idx_query_had, idx_front_signal)


@pl.jit.incore
def index_front_vector(
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.Tensor[[T, 64], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    idx_value: pl.Tensor[[IPROJ, 256], pl.FP32],
    idx_score: pl.Tensor[[IPROJ, 256], pl.FP32],
    idx_state_table: pl.Tensor[[B, ITABLE_COLS], pl.INT32],
    idx_ape: pl.Tensor[[4, 256], pl.FP32],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_state_slots: pl.Tensor[[T, 2], pl.INT32],
    idx_gamma: pl.Tensor[[128], pl.BF16],
    idx_cos: pl.Tensor[[IC, 64], pl.FP32],
    idx_sin: pl.Tensor[[IC, 64], pl.FP32],
    idx_offsets: pl.Tensor[[B], pl.INT32],
    idx_count: pl.Tensor[[1], pl.INT32],
    idx_slots: pl.Tensor[[ISLOTS, 2], pl.INT32],
    idx_had: pl.Tensor[[ICPAD, 128], pl.FP32],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    idx_query_acc: pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32],
    idx_query_had: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32],
    idx_query_scale_heads: pl.Tensor[[IDX_SCALE_T_ROWS, 64], pl.FP32],
    weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    QUERY_GROUP: pl.constexpr,
    idx_pooled: pl.InOut[pl.Tensor[[IPOOL, 128], pl.FP32]],
    idx_normed: pl.InOut[pl.Tensor[[ICPAD, 128], pl.BF16]],
    idx_state: pl.InOut[pl.Tensor[[ISTATE_PAGES, ISTATE_ELEMS], pl.FP32]],
    idx_query_rope: pl.InOut[pl.Tensor[[IDX_ROPE_ROWS, 8192], pl.BF16]],
    head_weights: pl.InOut[pl.Tensor[[WEIGHTS_ROWS, 64], pl.FP32]],
    idx_scales: pl.InOut[pl.Tensor[[ISCALES, 1], pl.FP32]],
    idx_debug: pl.InOut[pl.Tensor[[T, 128], pl.FP32]],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    idx_query_i8: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8]],
    idx_query_scale: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 1], pl.FP32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16]],
    idx_front_signal: pl.InOut[pl.Tensor[[IDX_FRONT_WORDS], pl.INT32]],
    FRONT_GROUPS: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = FRONT_GROUPS + worker
    tokens = pl.tensor.dim(positions, 0)
    (idx_front_signal, generation) = begin_invocation(idx_front_signal, core)
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_PROJECT,
        3 * FRONT_GROUPS,
        generation,
        0,
        INDEXER_OUT_DIM // INDEXER_N_TILE * ((tokens + INDEXER_M_TILE - 1) // INDEXER_M_TILE),
        1,
        FRONT_GROUPS,
        0,
        1,
    )
    owner_worker = worker
    idx_pooled = indexer_compressor_pool_stage(
        idx_value,
        idx_score,
        idx_state,
        idx_state_table,
        idx_ape,
        positions,
        idx_bounds,
        idx_lengths,
        idx_pooled,
        owner_worker,
        2 * FRONT_GROUPS,
    )
    idx_front_signal = publish_phase(idx_front_signal, I_POOL, core, 3 * FRONT_GROUPS, generation)
    idx_normed = indexer_compressor_norm_rope_stage(
        idx_pooled,
        idx_gamma,
        idx_cos,
        idx_sin,
        positions,
        idx_bounds,
        idx_lengths,
        idx_offsets,
        idx_normed,
        owner_worker,
        2 * FRONT_GROUPS,
    )
    idx_front_signal = publish_phase(idx_front_signal, I_NORM, core, 3 * FRONT_GROUPS, generation)
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_POOL,
        3 * FRONT_GROUPS,
        generation,
        0,
        pl.tensor.dim(idx_lengths, 0),
        1,
        2 * FRONT_GROUPS,
        FRONT_GROUPS,
        1,
    )
    idx_state = indexer_compressor_state_commit_stage(
        idx_value,
        idx_score,
        idx_ape,
        positions,
        idx_bounds,
        idx_lengths,
        idx_state_slots,
        idx_state,
        worker,
        2 * FRONT_GROUPS,
    )
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_QUERY_PROJECT,
        3 * FRONT_GROUPS,
        generation,
        0,
        IDX_N_HEADS * IDX_HEAD_DIM // QR_MM_N_TILE,
        1,
        FRONT_GROUPS,
        0,
        1,
    )
    if tokens >= 2 * DEQUANT_T_TILE and tokens % DEQUANT_T_TILE == 0:
        idx_query_rope = indexer_query_dequant_rope_wide_stage(
            qr_scale, idx_wq_b_scale, rope_cos, rope_signed, idx_query_acc, idx_query_rope, worker, 2 * FRONT_GROUPS
        )
    else:
        idx_query_rope = indexer_query_dequant_rope_stage(
            qr_scale, idx_wq_b_scale, rope_cos, rope_signed, idx_query_acc, idx_query_rope, worker, 2 * FRONT_GROUPS
        )
    idx_front_signal = publish_phase(idx_front_signal, I_QUERY_ROPE, core, 3 * FRONT_GROUPS, generation)
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_WEIGHTS,
        3 * FRONT_GROUPS,
        generation,
        0,
        WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE),
        1,
        FRONT_GROUPS,
        0,
        1,
    )
    head_weights = indexer_weights_reduce_stage(weights_partial, head_weights, tokens, worker, 2 * FRONT_GROUPS)
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_HAD,
        3 * FRONT_GROUPS,
        generation,
        0,
        COMPACT_HEAD_DIM
        // COMPACT_HAD_COL_TILE
        * ((pl.cast(pl.read(idx_count, [0]), pl.INDEX) + COMPACT_HAD_ROW_TILE - 1) // COMPACT_HAD_ROW_TILE),
        1,
        FRONT_GROUPS,
        0,
        1,
    )
    (idx_scales, idx_debug, idx_cache) = indexer_compact_quant_cache_stage(
        idx_had,
        positions,
        idx_bounds,
        idx_lengths,
        idx_offsets,
        idx_slots,
        idx_scales,
        idx_debug,
        idx_cache,
        worker,
        2 * FRONT_GROUPS,
    )
    idx_front_signal = publish_phase(idx_front_signal, I_QUANT, core, 3 * FRONT_GROUPS, generation)
    idx_front_signal = wait_row_owners(
        idx_front_signal,
        I_QUANT,
        3 * FRONT_GROUPS,
        generation,
        0,
        pl.tensor.dim(idx_lengths, 0),
        1,
        2 * FRONT_GROUPS,
        FRONT_GROUPS,
        1,
    )
    for scale_owner in pl.range(worker, SCALE_WRITERS, 2 * FRONT_GROUPS):
        idx_cache = indexer_scale_page_commit_stage(
            idx_scales, group_offsets, idx_cache, tokens, pl.tensor.dim(idx_lengths, 0), scale_owner, SCALE_WRITERS
        )
    idx_front_signal = wait_row_owners(
        idx_front_signal, I_QUERY_HAD, 3 * FRONT_GROUPS, generation, 0, (tokens + 3) // 4, 1, FRONT_GROUPS, 0, 1
    )
    (idx_query_i8, idx_query_scale) = indexer_query_quant_stage(
        idx_query_had, idx_query_i8, idx_query_scale, tokens, worker, 2 * FRONT_GROUPS
    )
    idx_front_signal = publish_phase(idx_front_signal, I_QUERY_QUANT, core, 3 * FRONT_GROUPS, generation)
    idx_front_signal = wait_row_owners(
        idx_front_signal, I_QUERY_QUANT, 3 * FRONT_GROUPS, generation, 0, tokens, 1, 2 * FRONT_GROUPS, FRONT_GROUPS, 1
    )
    coefficients = indexer_head_coefficients_stage(
        idx_query_scale_heads,
        head_weights,
        positions,
        idx_bounds,
        idx_lengths,
        group_offsets,
        coefficients,
        worker,
        2 * FRONT_GROUPS,
        QUERY_GROUP,
    )
    return (
        idx_pooled,
        idx_normed,
        idx_state,
        idx_query_rope,
        head_weights,
        idx_scales,
        idx_debug,
        idx_cache,
        idx_query_i8,
        idx_query_scale,
        coefficients,
        idx_front_signal,
    )


@pl.jit.incore
def main_front_cube(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    qr_matmul: pl.Tensor[[QR_ROWS, 1024], pl.INT8],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    positions: pl.Tensor[[T], pl.INT64],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    main_front_signal: pl.InOut[pl.Tensor[[MAIN_FRONT_WORDS], pl.INT32]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    FRONT_GROUPS: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    core = worker
    tokens = pl.tensor.dim(positions, 0)
    (main_front_signal, generation) = begin_invocation(main_front_signal, core)
    kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, worker, FRONT_GROUPS)
    main_front_signal = publish_phase(main_front_signal, M_KV_PROJECT, core, 3 * FRONT_GROUPS, generation)
    (main_value, main_score) = compressor_project_stage(
        normed, main_wkv, main_wgate, main_value, main_score, worker, FRONT_GROUPS
    )
    main_front_signal = publish_phase(main_front_signal, M_PROJECT, core, 3 * FRONT_GROUPS, generation)
    q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, FRONT_GROUPS)
    main_front_signal = publish_phase(main_front_signal, M_QUERY, core, 3 * FRONT_GROUPS, generation)
    return (main_value, main_score, q_proj, main_front_signal, kv_fp32)


@pl.jit.incore
def main_front_vector(
    qr_scale_pad: pl.Tensor[[P8, 1], pl.FP32],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.Tensor[[T, 64], pl.FP32],
    rope_swap: pl.Tensor[[T, 64], pl.INT32],
    q_proj: pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32],
    main_value: pl.Tensor[[MPROJ, 1024], pl.FP32],
    main_score: pl.Tensor[[MPROJ, 1024], pl.FP32],
    main_state_table: pl.Tensor[[B, MTABLE_COLS], pl.INT32],
    main_ape: pl.Tensor[[4, 1024], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    main_bounds: pl.Tensor[[MQ], pl.INT32],
    main_lengths: pl.Tensor[[B], pl.INT32],
    main_state_slots: pl.Tensor[[T, 2], pl.INT32],
    main_gamma: pl.Tensor[[512], pl.BF16],
    main_cos: pl.Tensor[[MC, 64], pl.FP32],
    main_sin: pl.Tensor[[MC, 64], pl.FP32],
    main_offsets: pl.Tensor[[B], pl.INT32],
    main_slots: pl.Tensor[[MSLOTS, 2], pl.INT32],
    main_pooled: pl.InOut[pl.Tensor[[MPOOL, 512], pl.FP32]],
    main_normed: pl.InOut[pl.Tensor[[MNORM, 512], pl.FP32]],
    main_debug: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    main_state: pl.InOut[pl.Tensor[[MSTATE_PAGES, MSTATE_ELEMS], pl.FP32]],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    main_front_signal: pl.InOut[pl.Tensor[[MAIN_FRONT_WORDS], pl.INT32]],
    kv_fp32: pl.Tensor[[KV_ROWS, 512], pl.FP32],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    FRONT_GROUPS: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = FRONT_GROUPS + worker
    tokens = pl.tensor.dim(positions, 0)
    (main_front_signal, generation) = begin_invocation(main_front_signal, core)
    main_front_signal = wait_row_owners(
        main_front_signal,
        M_KV_PROJECT,
        3 * FRONT_GROUPS,
        generation,
        0,
        KV_HEAD_DIM // KV_N_TILE * KV_OM,
        1,
        FRONT_GROUPS,
        0,
        1,
    )
    kv = kv_norm_rope_stage(kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, worker, 2 * FRONT_GROUPS)
    main_front_signal = publish_phase(main_front_signal, M_KV_NORM, core, 3 * FRONT_GROUPS, generation)
    main_front_signal = wait_row_owners(
        main_front_signal,
        M_KV_NORM,
        3 * FRONT_GROUPS,
        generation,
        0,
        (tokens + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE,
        1,
        2 * FRONT_GROUPS,
        FRONT_GROUPS,
        1,
    )
    cache_rows = original_kv_writeback_stage(kv, native_slots, cache_rows, worker, 2 * FRONT_GROUPS)
    main_front_signal = wait_row_owners(
        main_front_signal,
        M_PROJECT,
        3 * FRONT_GROUPS,
        generation,
        0,
        CMP_OUT_DIM // CMP_N_TILE * ((tokens + CMP_M_TILE - 1) // CMP_M_TILE),
        1,
        FRONT_GROUPS,
        0,
        1,
    )
    owner_worker = worker
    main_pooled = compressor_pool_stage(
        main_value,
        main_score,
        main_state,
        main_state_table,
        main_ape,
        positions,
        main_bounds,
        main_lengths,
        main_pooled,
        owner_worker,
        2 * FRONT_GROUPS,
    )
    main_front_signal = publish_phase(main_front_signal, M_POOL, core, 3 * FRONT_GROUPS, generation)
    (main_normed, main_debug, main_cache) = compressor_norm_rope_cache_stage(
        main_pooled,
        main_gamma,
        main_cos,
        main_sin,
        main_bounds,
        main_lengths,
        positions,
        main_offsets,
        main_slots,
        main_normed,
        main_debug,
        main_cache,
        owner_worker,
        2 * FRONT_GROUPS,
    )
    main_front_signal = wait_row_owners(
        main_front_signal,
        M_POOL,
        3 * FRONT_GROUPS,
        generation,
        0,
        pl.tensor.dim(main_lengths, 0),
        1,
        2 * FRONT_GROUPS,
        FRONT_GROUPS,
        1,
    )
    main_state = compressor_state_commit_stage(
        main_value,
        main_score,
        main_ape,
        positions,
        main_bounds,
        main_lengths,
        main_state_slots,
        main_state,
        worker,
        2 * FRONT_GROUPS,
    )
    main_front_signal = wait_row_owners(
        main_front_signal, M_QUERY, 3 * FRONT_GROUPS, generation, 0, QPROJ_N_BLOCKS, 1, FRONT_GROUPS, 0, 1
    )
    q_flat = q_dequant_rope_stage(
        wq_b_scale, rope_cos, rope_signed, rope_swap, q_flat, qr_scale_pad, q_proj, worker, 2 * FRONT_GROUPS
    )
    return (main_pooled, main_normed, main_debug, main_cache, main_state, q_flat, main_front_signal, kv, cache_rows)


@pl.jit.inline(auto_scope=False)
def index_front_chain(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    qr: pl.Tensor[[T, 1024], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.Tensor[[T, 64], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
    idx_state_table: pl.Tensor[[B, ITABLE_COLS], pl.INT32],
    idx_ape: pl.Tensor[[4, 256], pl.FP32],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_state_slots: pl.Tensor[[T, 2], pl.INT32],
    idx_gamma: pl.Tensor[[128], pl.BF16],
    idx_cos: pl.Tensor[[IC, 64], pl.FP32],
    idx_sin: pl.Tensor[[IC, 64], pl.FP32],
    idx_offsets: pl.Tensor[[B], pl.INT32],
    idx_count: pl.Tensor[[1], pl.INT32],
    idx_slots: pl.Tensor[[ISLOTS, 2], pl.INT32],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    idx_query_rope_heads: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    idx_query_scale_heads: pl.Tensor[[IDX_SCALE_T_ROWS, 64], pl.FP32],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    QUERY_GROUP: pl.constexpr,
    idx_value: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_score: pl.InOut[pl.Tensor[[IPROJ, 256], pl.FP32]],
    idx_query_acc: pl.InOut[pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32]],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    idx_had: pl.InOut[pl.Tensor[[ICPAD, 128], pl.FP32]],
    idx_query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    idx_front_signal: pl.InOut[pl.Tensor[[IDX_FRONT_WORDS], pl.INT32]],
    idx_pooled: pl.InOut[pl.Tensor[[IPOOL, 128], pl.FP32]],
    idx_normed: pl.InOut[pl.Tensor[[ICPAD, 128], pl.BF16]],
    idx_state: pl.InOut[pl.Tensor[[ISTATE_PAGES, ISTATE_ELEMS], pl.FP32]],
    idx_query_rope: pl.InOut[pl.Tensor[[IDX_ROPE_ROWS, 8192], pl.BF16]],
    head_weights: pl.InOut[pl.Tensor[[WEIGHTS_ROWS, 64], pl.FP32]],
    idx_scales: pl.InOut[pl.Tensor[[ISCALES, 1], pl.FP32]],
    idx_debug: pl.InOut[pl.Tensor[[T, 128], pl.FP32]],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    idx_query_i8: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8]],
    idx_query_scale: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 1], pl.FP32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16]],
    FRONT_GROUPS: pl.constexpr,
):
    with pl.cluster(name_hint="index_front_chain"):
        (idx_value, idx_score, idx_query_acc, weights_partial, idx_had, idx_query_had, idx_front_signal) = (
            index_front_cube(
                normed,
                qr,
                positions,
                idx_wkv,
                idx_wgate,
                idx_lengths,
                idx_count,
                idx_normed,
                hadamard,
                idx_wq_b,
                idx_query_rope_heads,
                weights_proj,
                idx_value,
                idx_score,
                idx_query_acc,
                weights_partial,
                idx_had,
                idx_query_had,
                idx_front_signal,
                FRONT_GROUPS,
            )
        )
        (
            idx_pooled,
            idx_normed,
            idx_state,
            idx_query_rope,
            head_weights,
            idx_scales,
            idx_debug,
            idx_cache,
            idx_query_i8,
            idx_query_scale,
            coefficients,
            idx_front_signal,
        ) = index_front_vector(
            qr_scale,
            rope_cos,
            rope_signed,
            positions,
            idx_value,
            idx_score,
            idx_state_table,
            idx_ape,
            idx_bounds,
            idx_lengths,
            idx_state_slots,
            idx_gamma,
            idx_cos,
            idx_sin,
            idx_offsets,
            idx_count,
            idx_slots,
            idx_had,
            idx_wq_b_scale,
            idx_query_acc,
            idx_query_had,
            idx_query_scale_heads,
            weights_partial,
            group_offsets,
            QUERY_GROUP,
            idx_pooled,
            idx_normed,
            idx_state,
            idx_query_rope,
            head_weights,
            idx_scales,
            idx_debug,
            idx_cache,
            idx_query_i8,
            idx_query_scale,
            coefficients,
            idx_front_signal,
            FRONT_GROUPS,
        )
    return (
        idx_value,
        idx_score,
        idx_query_acc,
        weights_partial,
        idx_had,
        idx_query_had,
        idx_front_signal,
        idx_pooled,
        idx_normed,
        idx_state,
        idx_query_rope,
        head_weights,
        idx_scales,
        idx_debug,
        idx_cache,
        idx_query_i8,
        idx_query_scale,
        coefficients,
    )


@pl.jit.inline(auto_scope=False)
def main_front_chain(
    normed: pl.Tensor[[T, 4096], pl.BF16],
    qr_matmul: pl.Tensor[[QR_ROWS, 1024], pl.INT8],
    qr_scale_pad: pl.Tensor[[P8, 1], pl.FP32],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.Tensor[[T, 64], pl.FP32],
    rope_swap: pl.Tensor[[T, 64], pl.INT32],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_state_table: pl.Tensor[[B, MTABLE_COLS], pl.INT32],
    main_ape: pl.Tensor[[4, 1024], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    main_bounds: pl.Tensor[[MQ], pl.INT32],
    main_lengths: pl.Tensor[[B], pl.INT32],
    main_state_slots: pl.Tensor[[T, 2], pl.INT32],
    main_gamma: pl.Tensor[[512], pl.BF16],
    main_cos: pl.Tensor[[MC, 64], pl.FP32],
    main_sin: pl.Tensor[[MC, 64], pl.FP32],
    main_offsets: pl.Tensor[[B], pl.INT32],
    main_slots: pl.Tensor[[MSLOTS, 2], pl.INT32],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    main_front_signal: pl.InOut[pl.Tensor[[MAIN_FRONT_WORDS], pl.INT32]],
    main_pooled: pl.InOut[pl.Tensor[[MPOOL, 512], pl.FP32]],
    main_normed: pl.InOut[pl.Tensor[[MNORM, 512], pl.FP32]],
    main_debug: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    main_state: pl.InOut[pl.Tensor[[MSTATE_PAGES, MSTATE_ELEMS], pl.FP32]],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    FRONT_GROUPS: pl.constexpr,
):
    with pl.cluster(name_hint="main_front_chain"):
        (main_value, main_score, q_proj, main_front_signal, kv_fp32) = main_front_cube(
            normed,
            qr_matmul,
            wq_b,
            main_wkv,
            main_wgate,
            positions,
            main_value,
            main_score,
            q_proj,
            main_front_signal,
            wkv,
            kv_fp32,
            FRONT_GROUPS,
        )
        (main_pooled, main_normed, main_debug, main_cache, main_state, q_flat, main_front_signal, kv, cache_rows) = (
            main_front_vector(
                qr_scale_pad,
                wq_b_scale,
                rope_cos,
                rope_signed,
                rope_swap,
                q_proj,
                main_value,
                main_score,
                main_state_table,
                main_ape,
                positions,
                main_bounds,
                main_lengths,
                main_state_slots,
                main_gamma,
                main_cos,
                main_sin,
                main_offsets,
                main_slots,
                main_pooled,
                main_normed,
                main_debug,
                main_cache,
                main_state,
                q_flat,
                main_front_signal,
                kv_fp32,
                gamma_ckv,
                native_slots,
                kv,
                cache_rows,
                FRONT_GROUPS,
            )
        )
    return (
        main_value,
        main_score,
        q_proj,
        main_front_signal,
        main_pooled,
        main_normed,
        main_debug,
        main_cache,
        main_state,
        q_flat,
        kv_fp32,
        kv,
        cache_rows,
    )


M_KV_PROJECT = 4
M_KV_NORM = 5
