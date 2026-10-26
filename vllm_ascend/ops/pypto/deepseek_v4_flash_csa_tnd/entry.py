# SPDX-License-Identifier: Apache-2.0
"""原生 TND 多 task CSA：沿用 BSH MPMD 的独立计算链与 TMR 依赖。

Q/L、slots、页表和 compact metadata 保持 Native 原存储。核内数学来自
84751e34 的 TND SPMD；每个 pl.spmd/pl.cluster 是独立调度单元，完整 CSA
由AICPU的Tensor/TaskId依赖调度。标量规划在编排层执行，局部数学MIX内保留FFTS；
不使用GM阶段信号、代际计数或轮询，不把多段业务计算融合成常驻任务。

普通入口在调用内创建全部临时Tensor；_debug入口仅额外接收Top-K观察输出。
两者沿用同一直接编排与数学helper，CPU合同检查防止数学和调度意外分叉。
"""

import pypto.language as pl

from .attention_plan_stages import (
    PHYSICAL_PLAN_COLUMNS,
    attention_geometry_prepare_stage,
    attention_inverse_rope_sign_stage,
    attention_slot_plan_stage,
)
from .attention_stream_stages import attention_qk_pv_cube_stage, attention_qk_pv_vector_lane_stage
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
from .hc_stages import (
    hc_comb_range,
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
    hc_widen_rms_range,
)
from .indexer_cache_stages import indexer_compact_quant_cache_stage
from .indexer_compressor_output_stages import (
    HEAD_DIM as COMPACT_HEAD_DIM,
)
from .indexer_compressor_output_stages import (
    OUT_TILE as COMPACT_HAD_COL_TILE,
)
from .indexer_compressor_output_stages import (
    RMS_PAD_TILE as COMPACT_HAD_ROW_TILE,
)
from .indexer_compressor_output_stages import (
    indexer_compressor_hadamard_stage,
    indexer_compressor_norm_rope_stage,
)
from .indexer_group_stages import indexer_group_plan_stage, indexer_head_coefficients_stage
from .indexer_publish_stages import indexer_pair_publish_stage
from .indexer_query_stages import (
    DEQUANT_T_TILE,
    DQ_ROPE_H_TILE,
    IDX_N_HEADS,
    MM_ROW_TILE,
    WEIGHTS_OK,
    indexer_query_dequant_rope_stage,
    indexer_query_hadamard_four_tokens_stage,
    indexer_query_project_stage,
    indexer_query_quant_stage,
    indexer_weights_project_stage,
    indexer_weights_reduce_stage,
)
from .indexer_scale_plan_stages import indexer_scale_page_commit_stage, indexer_scale_page_plan_stage
from .indexer_score_stages import indexer_score_cube_stage, indexer_score_vector_lane_stage
from .kv_stages import (
    HEAD_DIM as KV_HEAD_DIM,
)
from .kv_stages import (
    KV_N_TILE,
    KV_OM,
    kv_norm_rope_stage,
    kv_project_stage,
)
from .metadata_stages import compact_offsets_and_count_stage
from .output_tasks import output_tasks
from .qkv_stages import (
    BF16_WEIGHT_LAYOUT,
    QR_DENSE_M_TILE,
    QR_FIXED_M_GROUPS,
    QR_FIXED_SMALL_M_TILE,
    QR_FIXED_SMALL_ROWS,
    QR_M_TILE,
    QR_N_BLOCKS,
    q_a_stage,
    q_r_stage,
)
from .query_projection_stages import QUANT_WEIGHT_LAYOUT, q_b_stage, q_dequant_rope_stage, q_rope_prepare_stage
from .runtime_abi import (
    ATTN_PLAN_ROWS,
    CACHE_ROWS,
    COEFF_COLS,
    COEFF_ROWS,
    COMPRESSED_COLS,
    GROUP_OFFSET_ROWS,
    GROUPS,
    HEAD_RING_ROWS,
    IC,
    ICACHE_BYTES,
    ICACHE_PAGES,
    IDX_HEAD_ROWS,
    INDEX_TRANSFER_COLS,
    INDEX_TRANSFER_ROWS,
    INDEXER_COLS,
    IQ,
    ISCALES,
    ISLOTS,
    ISTATE_ELEMS,
    ISTATE_PAGES,
    ITABLE_COLS,
    KEY0_ROWS,
    KEY64_ROWS,
    KV_RING_ROWS,
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
    ORIGINAL_COLS,
    P8,
    PACK_ROWS,
    PAIR_ROWS,
    QPROJ_ROWS,
    QR_ROWS,
    QUERY_HEAD_ROWS,
    SCORE_ROWS,
    VECTORS,
    B,
    T,
)


@pl.jit.incore
def index_score_topk_cube(
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_query_i8: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    group_plan: pl.Tensor[[8], pl.INT32],
    coefficients: pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16],
    idx_key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    idx_key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    indexer_transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    SPLIT_SCORE_ROWS: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    pl.system.set_ffts(ffts)
    indexer_transfer = indexer_score_cube_stage(
        idx_query_i8,
        coefficients,
        idx_key0,
        idx_key64,
        idx_cache,
        indexer_table,
        positions,
        idx_lengths,
        idx_bounds,
        group_offsets,
        group_plan,
        indexer_transfer,
        worker,
        QUERY_GROUP,
        QK_PANEL,
        SCORE_TILE,
        PREFETCH,
        TO_L0,
        BALANCED,
        0,
        1,
        SPLIT_SCORE_ROWS,
    )
    return indexer_transfer


@pl.jit.incore
def index_score_topk_vector(
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    group_offsets: pl.InOut[pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32]],
    group_plan: pl.InOut[pl.Tensor[[8], pl.INT32]],
    indexer_transfer: pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32],
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS, 8192], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS, 1024], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    SCALE_PAGE_COALESCING: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    (score_arena, pair_arena) = indexer_score_vector_lane_stage(
        idx_cache,
        indexer_table,
        positions,
        idx_lengths,
        idx_bounds,
        group_offsets,
        group_plan,
        indexer_transfer,
        score_arena,
        pair_arena,
        pl.tile.get_subblock_idx(),
        pl.tile.get_block_idx(),
        QUERY_GROUP,
        QK_PANEL,
        SCORE_TILE,
        PREFETCH,
        TO_L0,
        BALANCED,
        SCALE_PAGE_COALESCING,
        0,
        1,
    )
    return (score_arena, pair_arena)


@pl.jit.inline(auto_scope=False)
def index_score_topk_group(
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_query_i8: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    group_plan: pl.Tensor[[8], pl.INT32],
    coefficients: pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16],
    idx_key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    idx_key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    indexer_transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    SPLIT_SCORE_ROWS: pl.constexpr,
    score_arena: pl.InOut[pl.Tensor[[SCORE_ROWS, 8192], pl.FP32]],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS, 1024], pl.FP32]],
    SCALE_PAGE_COALESCING: pl.constexpr,
):
    with pl.cluster(name_hint="indexer_score"):
        indexer_transfer = index_score_topk_cube(
            positions,
            idx_bounds,
            idx_lengths,
            idx_cache,
            indexer_table,
            idx_query_i8,
            group_offsets,
            group_plan,
            coefficients,
            idx_key0,
            idx_key64,
            indexer_transfer,
            ffts,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            SPLIT_SCORE_ROWS,
        )
        (score_arena, pair_arena) = index_score_topk_vector(
            positions,
            idx_bounds,
            idx_lengths,
            idx_cache,
            indexer_table,
            group_offsets,
            group_plan,
            indexer_transfer,
            score_arena,
            pair_arena,
            ffts,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            SCALE_PAGE_COALESCING,
        )
    return (indexer_transfer, score_arena, pair_arena)


@pl.jit.incore
def sparse_attn_cube(
    q_heads: pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16],
    positions: pl.Tensor[[T], pl.INT64],
    attention_mask: pl.Tensor[[ATTN_PLAN_ROWS, 16], pl.INT32],
    kv_ring: pl.Tensor[[KV_RING_ROWS, 512], pl.BF16],
    score_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32]],
    probability_ring: pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16],
    pv_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
):
    worker = pl.tile.get_block_idx()
    pl.system.set_ffts(ffts)
    (score_ring, pv_ring) = attention_qk_pv_cube_stage(
        q_heads,
        attention_mask,
        kv_ring,
        score_ring,
        probability_ring,
        pv_ring,
        ffts,
        worker,
        pl.tensor.dim(positions, 0),
        GROUPS,
        2,
    )
    return (score_ring, pv_ring)


@pl.jit.incore
def sparse_attn_vector(
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    positions: pl.Tensor[[T], pl.INT64],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
    attention_indices: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 512], pl.INT32]],
    attention_bias: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 640], pl.FP32]],
    attention_mask: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 16], pl.INT32]],
    inverse_sin: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    attn_sink: pl.Tensor[[64, 1], pl.FP32],
    kv_ring: pl.InOut[pl.Tensor[[KV_RING_ROWS, 512], pl.BF16]],
    score_ring: pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32],
    probability_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16]],
    pv_ring: pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32],
    mi_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    li_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    packed: pl.InOut[pl.Tensor[[PACK_ROWS, 4096], pl.BF16]],
    ffts: pl.Tensor[[256], pl.INT64],
    ATTENTION_QUERY_METADATA: pl.constexpr,
    LOCAL_ATTENTION_STATS: pl.constexpr,
    PLAN_PHYSICAL_ROWS: pl.constexpr,
    PLAN_TABLE_COLUMNS: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    (kv_ring, probability_ring, mi_ring, li_ring, packed) = attention_qk_pv_vector_lane_stage(
        cache_rows,
        main_cache,
        original_table,
        compressed_table,
        idx_bounds,
        idx_lengths,
        positions,
        attention_indices,
        attention_bias,
        attention_mask,
        attn_sink,
        rope_cos,
        inverse_sin,
        kv_ring,
        score_ring,
        probability_ring,
        pv_ring,
        mi_ring,
        li_ring,
        packed,
        ffts,
        pl.tile.get_block_idx(),
        pl.tile.get_subblock_idx(),
        GROUPS,
        2,
        ATTENTION_QUERY_METADATA,
        LOCAL_ATTENTION_STATS,
        PLAN_PHYSICAL_ROWS,
        PLAN_TABLE_COLUMNS,
    )
    return (kv_ring, probability_ring, mi_ring, li_ring, packed)


@pl.jit.inline(auto_scope=False)
def sparse_attn_group(
    q_heads: pl.Tensor[[QUERY_HEAD_ROWS, 512], pl.BF16],
    attention_mask: pl.Tensor[[ATTN_PLAN_ROWS, 16], pl.INT32],
    kv_ring: pl.Tensor[[KV_RING_ROWS, 512], pl.BF16],
    score_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 128], pl.FP32]],
    probability_ring: pl.Tensor[[HEAD_RING_ROWS, 128], pl.BF16],
    pv_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 512], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    positions: pl.Tensor[[T], pl.INT64],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
    attention_indices: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 512], pl.INT32]],
    attention_bias: pl.InOut[pl.Tensor[[ATTN_PLAN_ROWS, 640], pl.FP32]],
    inverse_sin: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    attn_sink: pl.Tensor[[64, 1], pl.FP32],
    mi_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    li_ring: pl.InOut[pl.Tensor[[HEAD_RING_ROWS, 1], pl.FP32]],
    packed: pl.InOut[pl.Tensor[[PACK_ROWS, 4096], pl.BF16]],
    ATTENTION_QUERY_METADATA: pl.constexpr,
    LOCAL_ATTENTION_STATS: pl.constexpr,
    PLAN_PHYSICAL_ROWS: pl.constexpr,
    PLAN_TABLE_COLUMNS: pl.constexpr,
):
    with pl.cluster(name_hint="sparse_attn"):
        (score_ring, pv_ring) = sparse_attn_cube(
            q_heads, positions, attention_mask, kv_ring, score_ring, probability_ring, pv_ring, ffts
        )
        (kv_ring, probability_ring, mi_ring, li_ring, packed) = sparse_attn_vector(
            rope_cos,
            cache_rows,
            positions,
            main_cache,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            attention_mask,
            inverse_sin,
            attn_sink,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            mi_ring,
            li_ring,
            packed,
            ffts,
            ATTENTION_QUERY_METADATA,
            LOCAL_ATTENTION_STATS,
            PLAN_PHYSICAL_ROWS,
            PLAN_TABLE_COLUMNS,
        )
    return (score_ring, pv_ring, kv_ring, probability_ring, mi_ring, li_ring, packed)


@pl.jit.inline(auto_scope=False)
def qkv_tail_tasks(
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    qr_matmul: pl.InOut[pl.Tensor[[QR_ROWS, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    rope_swap: pl.InOut[pl.Tensor[[T, 64], pl.INT32]],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    wkv: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    tokens: pl.Scalar[pl.INDEX],
    kv_workers: pl.Scalar[pl.INDEX],
):
    with pl.spmd(GROUPS, name_hint="qproj_b", allow_early_resolve=True, sync_start=False):
        q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, pl.tile.get_block_idx(), GROUPS)
    with pl.spmd(VECTORS, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True, sync_start=False):
        q_flat = q_dequant_rope_stage(
            wq_b_scale,
            rope_cos,
            rope_signed,
            rope_swap,
            q_flat,
            qr_scale_pad,
            q_proj,
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(kv_workers, name_hint="kv_proj", allow_early_resolve=True, sync_start=False):
        kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, pl.tile.get_block_idx(), GROUPS)
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="kv_norm_rope", allow_early_resolve=True, sync_start=False
    ):
        kv = kv_norm_rope_stage(
            kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, pl.tile.get_block_idx(), VECTORS
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS),
        name_hint="csa_cache_writeback",
        allow_early_resolve=True,
        sync_start=False,
    ):
        cache_rows = original_kv_writeback_stage(kv, native_slots, cache_rows, pl.tile.get_block_idx(), VECTORS)
    return q_proj, q_flat, kv_fp32, kv, cache_rows


@pl.jit.inline(auto_scope=False)
def main_compressor_tasks(
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_value: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_score: pl.InOut[pl.Tensor[[MPROJ, 1024], pl.FP32]],
    main_state: pl.InOut[pl.Tensor[[MSTATE_PAGES, MSTATE_ELEMS], pl.FP32]],
    main_state_table: pl.Tensor[[B, MTABLE_COLS], pl.INT32],
    main_ape: pl.Tensor[[4, 1024], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    main_bounds: pl.Tensor[[MQ], pl.INT32],
    main_lengths: pl.Tensor[[B], pl.INT32],
    main_state_slots: pl.Tensor[[T, 2], pl.INT32],
    main_pooled: pl.InOut[pl.Tensor[[MPOOL, 512], pl.FP32]],
    main_gamma: pl.Tensor[[512], pl.BF16],
    main_cos: pl.Tensor[[MC, 64], pl.FP32],
    main_sin: pl.Tensor[[MC, 64], pl.FP32],
    main_offsets: pl.InOut[pl.Tensor[[B], pl.INT32]],
    main_slots: pl.Tensor[[MSLOTS, 2], pl.INT32],
    main_normed: pl.InOut[pl.Tensor[[MNORM, 512], pl.FP32]],
    main_debug: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    main_requests: pl.Scalar[pl.INDEX],
    main_project_workers: pl.Scalar[pl.INDEX],
    qa_tid: pl.Scalar[pl.TASK_ID],
):
    with pl.spmd(
        main_project_workers,
        name_hint="compressor_kv_score",
        allow_early_resolve=True,
        sync_start=False,
        deps=[qa_tid],
    ) as main_project_tid:
        (main_value, main_score) = compressor_project_stage(
            normed, main_wkv, main_wgate, main_value, main_score, pl.tile.get_block_idx(), GROUPS
        )
    with pl.spmd(
        pl.min(main_requests, VECTORS), name_hint="compressor_pool", allow_early_resolve=True, sync_start=False
    ) as main_pool_tid:
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        pl.min(main_requests, VECTORS),
        name_hint="compressor_state_commit",
        allow_early_resolve=True,
        sync_start=False,
        deps=[main_pool_tid],
    ):
        main_state = compressor_state_commit_stage(
            main_value,
            main_score,
            main_ape,
            positions,
            main_bounds,
            main_lengths,
            main_state_slots,
            main_state,
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        pl.min(main_requests, VECTORS),
        name_hint="compressor_norm_rope_cache",
        allow_early_resolve=True,
        sync_start=False,
    ):
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    return main_value, main_score, main_pooled, main_state, main_normed, main_debug, main_cache, main_project_tid


@pl.jit.incore
def indexer_scale_page_commit_task(
    scales: pl.Tensor[[ISCALES, 1], pl.FP32],
    offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    tokens: pl.Scalar[pl.INDEX],
    requests: pl.Scalar[pl.INDEX],
):
    cache = indexer_scale_page_commit_stage(scales, offsets, cache, tokens, requests, pl.tile.get_block_idx(), VECTORS)
    return cache


@pl.jit
def decode_csa_tnd_debug(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_cq: pl.Tensor[[1024], pl.BF16],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_sin: pl.Tensor[[T, 64], pl.FP32],
    wkv: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_state: pl.InOut[pl.Tensor[[MSTATE_PAGES, MSTATE_ELEMS], pl.FP32]],
    main_state_table: pl.Tensor[[B, MTABLE_COLS], pl.INT32],
    main_ape: pl.Tensor[[4, 1024], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    main_bounds: pl.Tensor[[MQ], pl.INT32],
    main_lengths: pl.Tensor[[B], pl.INT32],
    main_state_slots: pl.Tensor[[T, 2], pl.INT32],
    main_gamma: pl.Tensor[[512], pl.BF16],
    main_cos: pl.Tensor[[MC, 64], pl.FP32],
    main_sin: pl.Tensor[[MC, 64], pl.FP32],
    main_slots: pl.Tensor[[MSLOTS, 2], pl.INT32],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
    idx_state: pl.InOut[pl.Tensor[[ISTATE_PAGES, ISTATE_ELEMS], pl.FP32]],
    idx_state_table: pl.Tensor[[B, ITABLE_COLS], pl.INT32],
    idx_ape: pl.Tensor[[4, 256], pl.FP32],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_state_slots: pl.Tensor[[T, 2], pl.INT32],
    idx_gamma: pl.Tensor[[128], pl.BF16],
    idx_cos: pl.Tensor[[IC, 64], pl.FP32],
    idx_sin: pl.Tensor[[IC, 64], pl.FP32],
    idx_slots: pl.Tensor[[ISLOTS, 2], pl.INT32],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8, QUANT_WEIGHT_LAYOUT],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    weights_proj: pl.Tensor[[64, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    topk_scores: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    topk_indices: pl.InOut[pl.Tensor[[T, 512], pl.INT32]],
    attn_sink: pl.Tensor[[64, 1], pl.FP32],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, BF16_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 16384], pl.BF16]],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    ROOTS: pl.constexpr,
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
    ATTENTION_QUERY_METADATA: pl.constexpr,
    HC_CONSUMER_REDUCE: pl.constexpr,
    QA_M16_PARALLEL: pl.constexpr = False,
    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    SCALE_PAGE_COALESCING: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,
    SPLIT_SCORE_ROWS: pl.constexpr = False,
    LIMIT_EMPTY_WORKERS: pl.constexpr = False,
    BSH_PROJECTION_ORDER: pl.constexpr = False,
):
    t = pl.tensor.dim(x, 0)
    b = pl.tensor.dim(idx_lengths, 0)
    p8 = (t + 7) // 8 * 8
    p16 = (t + 15) // 16 * 16
    p64 = (t + 63) // 64 * 64
    p128 = (t + 127) // 128 * 128
    group = QUERY_GROUP
    mm = pl.max(p128, (t + OB_ROWS - 1) // OB_ROWS * OB_ROWS)
    x_fp32 = pl.create_tensor([t, 16384], dtype=pl.FP32)
    inv = pl.create_tensor([p16, 1], dtype=pl.FP32)
    widen_tail = pl.create_tensor([8, 16384], dtype=pl.FP32)
    partials = pl.create_tensor([p16 * 4, 32], dtype=pl.FP32)
    raw = pl.create_tensor([p16, 32], dtype=pl.FP32)
    pre = pl.create_tensor([p16, 8], dtype=pl.FP32)
    post = pl.create_tensor([t, 4], dtype=pl.FP32)
    post_tail = pl.create_tensor([8, 8], dtype=pl.FP32)
    comb = pl.create_tensor([t, 16], dtype=pl.FP32)
    comb_tail = pl.create_tensor([8, 32], dtype=pl.FP32)
    mixed = pl.create_tensor([p8, 4096], dtype=pl.BF16)
    normed = pl.create_tensor([t, 4096], dtype=pl.BF16)
    norm_tail = pl.create_tensor([8, 4096], dtype=pl.BF16)
    qa = pl.create_tensor([p16, 1024], dtype=pl.FP32)
    qr = pl.create_tensor([t, 1024], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t, 1], dtype=pl.FP32)
    qr_matmul = pl.create_tensor([p128, 1024], dtype=pl.INT8)
    qr_scale_pad = pl.create_tensor([p8, 1], dtype=pl.FP32)
    rope_signed = pl.create_tensor([t, 64], dtype=pl.FP32)
    rope_swap = pl.create_tensor([t, 64], dtype=pl.INT32)
    q_proj = pl.create_tensor([p128, 32768], dtype=pl.INT32)
    q_flat = pl.create_tensor([t, 32768], dtype=pl.BF16)
    kv_fp32 = pl.create_tensor([p16, 512], dtype=pl.FP32)
    kv = pl.create_tensor([t, 512], dtype=pl.BF16)
    main_value = pl.create_tensor([p64, 1024], dtype=pl.FP32)
    main_score = pl.create_tensor([p64, 1024], dtype=pl.FP32)
    main_pooled = pl.create_tensor([t, 512], dtype=pl.FP32)
    main_offsets = pl.create_tensor([b], dtype=pl.INT32)
    main_count = pl.create_tensor([1], dtype=pl.INT32)
    main_normed = pl.create_tensor([t, 512], dtype=pl.FP32)
    main_debug = pl.create_tensor([t, 512], dtype=pl.FP32)
    idx_value = pl.create_tensor([p16, 256], dtype=pl.FP32)
    idx_score = pl.create_tensor([p16, 256], dtype=pl.FP32)
    idx_pooled = pl.create_tensor([t, 128], dtype=pl.FP32)
    idx_offsets = pl.create_tensor([b], dtype=pl.INT32)
    idx_count = pl.create_tensor([1], dtype=pl.INT32)
    idx_normed = pl.create_tensor([p16, 128], dtype=pl.BF16)
    idx_had = pl.create_tensor([p16, 128], dtype=pl.FP32)
    idx_scales = pl.create_tensor([p16, 1], dtype=pl.FP32)
    idx_debug = pl.create_tensor([t, 128], dtype=pl.FP32)
    idx_query_acc = pl.create_tensor([p64, 8192], dtype=pl.INT32)
    idx_query_rope = pl.create_tensor([p16, 8192], dtype=pl.BF16)
    idx_query_had = pl.create_tensor([t * 64, 128], dtype=pl.FP32)
    idx_query_i8 = pl.create_tensor([t * 64, 128], dtype=pl.INT8)
    idx_query_scale = pl.create_tensor([t * 64, 1], dtype=pl.FP32)
    weights_partial = pl.create_tensor([p16 * 4, 64], dtype=pl.FP32)
    head_weights = pl.create_tensor([p16, 64], dtype=pl.FP32)
    group_offsets = pl.create_tensor(
        [((b + 1 + 16 - 1) // 16 * 16 + 4 * t + 16 - 1) // 16 * 16 + 48 + 4 * t], dtype=pl.INT32
    )
    group_plan = pl.create_tensor([8], dtype=pl.INT32)
    coefficients = pl.create_tensor([t * 16, group * 64], dtype=pl.FP16)
    indexer_transfer = pl.create_tensor([24 * 2 * group, SCORE_TILE], dtype=pl.FP32)
    score_arena = pl.create_tensor([24 * 2 * group, 8192], dtype=pl.FP32)
    pair_arena = pl.create_tensor([t * 64, 1024], dtype=pl.FP32)
    attention_indices = pl.create_tensor([p8, 512], dtype=pl.INT32)
    attention_bias = pl.create_tensor([p8, 640], dtype=pl.FP32)
    attention_mask = pl.create_tensor([p8, 16], dtype=pl.INT32)
    inverse_sin = pl.create_tensor([t, 64], dtype=pl.FP32)
    kv_ring = pl.create_tensor([24 * 3 * 128, 512], dtype=pl.BF16)
    score_ring = pl.create_tensor([24 * 3 * 64, 128], dtype=pl.FP32)
    probability_ring = pl.create_tensor([24 * 3 * 64, 128], dtype=pl.BF16)
    pv_ring = pl.create_tensor([24 * 3 * 64, 512], dtype=pl.FP32)
    mi_ring = pl.create_tensor([24 * 3 * 64, 1], dtype=pl.FP32)
    li_ring = pl.create_tensor([24 * 3 * 64, 1], dtype=pl.FP32)
    packed = pl.create_tensor([8 * p128, 4096], dtype=pl.BF16)
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    o_r = pl.create_tensor([mm, 8192], dtype=pl.FP32)
    o_quant = pl.create_tensor([mm, 8192], dtype=pl.INT8)
    o_scales = pl.create_tensor([8, p128], dtype=pl.FP32)
    o_partials = pl.create_tensor([mm, 32768], dtype=pl.INT32)
    q_heads = pl.reshape(q_flat, [t * 64, 512])
    rope_rows = pl.tensor.dim(idx_query_rope, 0)
    idx_query_rope_heads = pl.reshape(idx_query_rope, [rope_rows * 64, 128])
    idx_query_scale_heads = pl.reshape(idx_query_scale, [t, 64])
    tokens = pl.tensor.dim(x, 0)
    requests = pl.tensor.dim(idx_lengths, 0)
    main_requests = pl.tensor.dim(main_lengths, 0)
    q_heads = pl.reshape(q_flat, [tokens * 64, 512])
    idx_rope_tokens = pl.tensor.dim(idx_query_rope, 0)
    idx_query_rope_heads = pl.reshape(idx_query_rope, [idx_rope_tokens * 64, 128])
    idx_query_scale_heads = pl.reshape(idx_query_scale, [tokens, 64])
    pages = pl.tensor.dim(idx_cache, 0)
    page_bytes = pl.tensor.dim(idx_cache, 1)
    cache_flat = pl.reshape(idx_cache, [pages * page_bytes])
    key_prefix = cache_flat[0 : pages * page_bytes // 128 * 128]
    key_shifted = cache_flat[64 : 64 + (pages * page_bytes - 64) // 128 * 128]
    idx_key0 = pl.reshape(key_prefix, [pages * page_bytes // 128, 128])
    idx_key64 = pl.reshape(key_shifted, [(pages * page_bytes - 64) // 128, 128])
    qa_workers = GROUPS
    idx_project_workers = GROUPS
    idx_hadamard_workers = GROUPS
    query_rope_workers = VECTORS
    query_hadamard_workers = GROUPS
    weight_workers = GROUPS
    main_project_workers = GROUPS
    kv_workers = GROUPS
    if LIMIT_EMPTY_WORKERS:
        if QA_M16_PARALLEL:
            qa_units = QR_N_BLOCKS * ((tokens + QR_M_TILE - 1) // QR_M_TILE)
        elif tokens < QR_FIXED_SMALL_ROWS:
            qa_units = QR_N_BLOCKS * pl.min(QR_FIXED_M_GROUPS, pl.max(1, tokens // QR_FIXED_SMALL_M_TILE))
        else:
            qa_units = QR_N_BLOCKS * pl.min(QR_FIXED_M_GROUPS, pl.max(1, tokens // QR_DENSE_M_TILE))
        qa_workers = pl.min(GROUPS, qa_units)
        idx_project_workers = pl.min(
            GROUPS, INDEXER_OUT_DIM // INDEXER_N_TILE * ((tokens + INDEXER_M_TILE - 1) // INDEXER_M_TILE)
        )
        compact_hadamard_units = (
            COMPACT_HEAD_DIM
            // COMPACT_HAD_COL_TILE
            * ((pl.tensor.dim(idx_normed, 0) + COMPACT_HAD_ROW_TILE - 1) // COMPACT_HAD_ROW_TILE)
        )
        idx_hadamard_workers = pl.min(GROUPS, compact_hadamard_units)
        query_rope_workers = pl.min(
            VECTORS, IDX_N_HEADS // DQ_ROPE_H_TILE * ((tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE)
        )
        query_hadamard_workers = pl.min(GROUPS, (tokens + 3) // 4)
        weight_workers = pl.min(GROUPS, WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE))
        main_project_workers = pl.min(GROUPS, CMP_OUT_DIM // CMP_N_TILE * ((tokens + CMP_M_TILE - 1) // CMP_M_TILE))
        kv_workers = KV_HEAD_DIM // KV_N_TILE * KV_OM
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_widen_rms", allow_early_resolve=True, sync_start=True
    ):
        (x_fp32, inv) = hc_widen_rms_range(x, x_fp32, inv, widen_tail, pl.tile.get_block_idx(), VECTORS)
    with pl.spmd(
        pl.min((tokens + 15) // 16 * 4, GROUPS), name_hint="hc_linear", allow_early_resolve=True, sync_start=True
    ):
        partials = hc_project_range(x_fp32, hc_fn, partials, pl.tile.get_block_idx(), GROUPS)
    with pl.spmd(
        pl.min((tokens + 15) // 16, VECTORS), name_hint="hc_reduce", allow_early_resolve=True, sync_start=True
    ):
        raw = hc_reduce_range(partials, raw, pl.tile.get_block_idx(), VECTORS)
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_pre_gates", allow_early_resolve=True, sync_start=True
    ):
        (pre, post) = hc_gates_range(
            inv,
            raw,
            hc_scale,
            hc_base,
            pre,
            post,
            post_tail,
            pl.tile.get_block_idx(),
            VECTORS,
            partials,
            HC_CONSUMER_REDUCE,
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_mix_norm", allow_early_resolve=True, sync_start=True
    ):
        (mixed, normed) = hc_mix_norm_range(
            x_fp32, pre, norm_w, mixed, normed, norm_tail, pl.tile.get_block_idx(), VECTORS
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_sinkhorn", allow_early_resolve=True, sync_start=True
    ):
        comb = hc_comb_range(
            inv,
            raw,
            hc_scale,
            hc_base_2d,
            comb,
            comb_tail,
            pl.tile.get_block_idx(),
            VECTORS,
            partials,
            HC_CONSUMER_REDUCE,
        )
    (main_offsets, main_count) = compact_offsets_and_count_stage(
        main_bounds, main_lengths, main_offsets, main_count, tokens, 0
    )
    (idx_offsets, idx_count) = compact_offsets_and_count_stage(
        idx_bounds, idx_lengths, idx_offsets, idx_count, tokens, 0
    )
    with pl.spmd(
        pl.min((tokens + 3) // 4, VECTORS), name_hint="csa_rope_sign", allow_early_resolve=True, sync_start=False
    ):
        (rope_signed, rope_swap) = q_rope_prepare_stage(
            rope_sin, rope_signed, rope_swap, pl.tile.get_block_idx(), VECTORS
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="csa_inverse_rope", allow_early_resolve=True, sync_start=False
    ):
        inverse_sin = attention_inverse_rope_sign_stage(rope_sin, inverse_sin, pl.tile.get_block_idx(), VECTORS)
    if EARLY_ATTENTION_GEOMETRY:
        with pl.spmd(
            pl.min((tokens + 7) // 8, VECTORS),
            name_hint="csa_query_geometry",
            allow_early_resolve=True,
            sync_start=False,
        ):
            attention_indices = attention_geometry_prepare_stage(
                positions,
                idx_bounds,
                idx_lengths,
                compressed_table,
                attention_indices,
                pl.tensor.dim(main_cache, 0) // 32,
                pl.tile.get_block_idx(),
                VECTORS,
                PLAN_PHYSICAL_ROWS,
                PLAN_TABLE_COLUMNS,
            )
    with pl.spmd(qa_workers, name_hint="qproj_a", allow_early_resolve=True, sync_start=False) as qa_tid:
        qa = q_a_stage(normed, wq_a, qa, pl.tile.get_block_idx(), GROUPS, QA_M16_PARALLEL)
    projection_dependencies = pl.array.create(1, pl.TASK_ID)
    if not BSH_PROJECTION_ORDER:
        with pl.spmd(
            idx_project_workers, name_hint="indexer_compressor_kv_score", allow_early_resolve=True, sync_start=False
        ):
            (idx_value, idx_score) = indexer_compressor_project_stage(
                normed, idx_wkv, idx_wgate, idx_value, idx_score, pl.tile.get_block_idx(), GROUPS
            )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="qproj_a_rms_quant", allow_early_resolve=True, sync_start=False
    ):
        (qr, qr_scale, qr_matmul, qr_scale_pad) = q_r_stage(
            qa, gamma_cq, qr, qr_scale, qr_matmul, qr_scale_pad, pl.tile.get_block_idx(), VECTORS
        )
    if BSH_PROJECTION_ORDER:
        (q_proj, q_flat, kv_fp32, kv, cache_rows) = qkv_tail_tasks(
            normed,
            qr_matmul,
            qr_scale_pad,
            wq_b,
            wq_b_scale,
            rope_cos,
            rope_signed,
            rope_swap,
            q_proj,
            q_flat,
            wkv,
            gamma_ckv,
            kv_fp32,
            kv,
            native_slots,
            cache_rows,
            tokens,
            kv_workers,
        )
        (main_value, main_score, main_pooled, main_state, main_normed, main_debug, main_cache, main_project_tid) = (
            main_compressor_tasks(
                normed,
                main_wkv,
                main_wgate,
                main_value,
                main_score,
                main_state,
                main_state_table,
                main_ape,
                positions,
                main_bounds,
                main_lengths,
                main_state_slots,
                main_pooled,
                main_gamma,
                main_cos,
                main_sin,
                main_offsets,
                main_slots,
                main_normed,
                main_debug,
                main_cache,
                main_requests,
                main_project_workers,
                qa_tid,
            )
        )
        projection_dependencies[0] = main_project_tid
        with pl.spmd(
            idx_project_workers, name_hint="indexer_compressor_kv_score", allow_early_resolve=True, sync_start=False
        ):
            (idx_value, idx_score) = indexer_compressor_project_stage(
                normed, idx_wkv, idx_wgate, idx_value, idx_score, pl.tile.get_block_idx(), GROUPS
            )
    with pl.spmd(
        pl.min(requests, VECTORS), name_hint="indexer_compressor_pool", allow_early_resolve=True, sync_start=False
    ) as idx_pool_tid:
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        pl.min(requests, VECTORS),
        name_hint="indexer_state_commit",
        allow_early_resolve=True,
        sync_start=False,
        deps=[idx_pool_tid],
    ):
        idx_state = indexer_compressor_state_commit_stage(
            idx_value,
            idx_score,
            idx_ape,
            positions,
            idx_bounds,
            idx_lengths,
            idx_state_slots,
            idx_state,
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        pl.min(requests, VECTORS), name_hint="indexer_compressor_norm_rope", allow_early_resolve=True, sync_start=False
    ):
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    if BSH_PROJECTION_ORDER:
        with pl.spmd(
            idx_hadamard_workers,
            name_hint="indexer_compressor_hadamard",
            allow_early_resolve=True,
            sync_start=False,
            deps=[projection_dependencies[0]],
        ):
            compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
            idx_had = indexer_compressor_hadamard_stage(
                idx_normed, hadamard, idx_had, compact_count, pl.tile.get_block_idx(), GROUPS
            )
    else:
        compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
        compact_hadamard_tasks = (
            (compact_count + COMPACT_HAD_ROW_TILE - 1)
            // COMPACT_HAD_ROW_TILE
            * (COMPACT_HEAD_DIM // COMPACT_HAD_COL_TILE)
        )
        idx_hadamard_workers = pl.max(1, pl.min(GROUPS, compact_hadamard_tasks))
        with pl.spmd(
            idx_hadamard_workers, name_hint="indexer_compressor_hadamard", allow_early_resolve=True, sync_start=False
        ):
            idx_had = indexer_compressor_hadamard_stage(
                idx_normed, hadamard, idx_had, compact_count, pl.tile.get_block_idx(), GROUPS
            )
    with pl.spmd(GROUPS, name_hint="indexer_query_proj", allow_early_resolve=True, sync_start=False):
        idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, pl.tile.get_block_idx(), GROUPS)
    with pl.spmd(query_rope_workers, name_hint="indexer_query_rope", allow_early_resolve=True, sync_start=False):
        idx_query_rope = indexer_query_dequant_rope_stage(
            qr_scale,
            idx_wq_b_scale,
            rope_cos,
            rope_signed,
            idx_query_acc,
            idx_query_rope,
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        query_hadamard_workers, name_hint="indexer_query_hadamard", allow_early_resolve=True, sync_start=False
    ):
        idx_query_had = indexer_query_hadamard_four_tokens_stage(
            idx_query_rope_heads, hadamard, idx_query_had, tokens, pl.tile.get_block_idx(), GROUPS
        )
    with pl.spmd(pl.min(tokens, VECTORS), name_hint="indexer_query_quant", allow_early_resolve=True, sync_start=False):
        (idx_query_i8, idx_query_scale) = indexer_query_quant_stage(
            idx_query_had, idx_query_i8, idx_query_scale, tokens, pl.tile.get_block_idx(), VECTORS
        )
    with pl.spmd(weight_workers, name_hint="indexer_weights_proj", allow_early_resolve=True, sync_start=False):
        weights_partial = indexer_weights_project_stage(
            normed, weights_proj, weights_partial, pl.tile.get_block_idx(), GROUPS
        )
    with pl.spmd(
        pl.min((tokens + 15) // 16, VECTORS),
        name_hint="indexer_weights_reduce",
        allow_early_resolve=True,
        sync_start=False,
    ):
        head_weights = indexer_weights_reduce_stage(
            weights_partial, head_weights, tokens, pl.tile.get_block_idx(), VECTORS
        )
    (group_offsets, group_plan) = indexer_group_plan_stage(
        idx_bounds, idx_lengths, positions, group_offsets, group_plan, tokens, 0, QUERY_GROUP, BALANCED, ROOTS
    )
    group_offsets = indexer_scale_page_plan_stage(
        idx_bounds,
        idx_lengths,
        positions,
        idx_offsets,
        idx_slots,
        idx_cache,
        group_offsets,
        pl.tensor.dim(idx_scales, 0),
        VECTORS,
    )
    coefficient_workers = VECTORS
    if not BSH_PROJECTION_ORDER:
        coefficient_groups = pl.cast(pl.read(group_offsets, [requests]), pl.INDEX)
        coefficient_workers = pl.max(1, pl.min(VECTORS, coefficient_groups))
    with pl.spmd(
        coefficient_workers, name_hint="indexer_head_coefficients", allow_early_resolve=True, sync_start=False
    ):
        coefficients = indexer_head_coefficients_stage(
            idx_query_scale_heads,
            head_weights,
            positions,
            idx_bounds,
            idx_lengths,
            group_offsets,
            coefficients,
            pl.tile.get_block_idx(),
            VECTORS,
            QUERY_GROUP,
        )
    with pl.spmd(
        pl.min(requests, VECTORS),
        name_hint="indexer_compressor_cache_write",
        allow_early_resolve=True,
        sync_start=False,
    ):
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        VECTORS, name_hint="indexer_scale_page_commit", allow_early_resolve=True, sync_start=False
    ) as idx_cache_tid:
        idx_cache = indexer_scale_page_commit_task(idx_scales, group_offsets, idx_cache, tokens, requests)
    if not BSH_PROJECTION_ORDER:
        (main_value, main_score, main_pooled, main_state, main_normed, main_debug, main_cache, main_project_tid) = (
            main_compressor_tasks(
                normed,
                main_wkv,
                main_wgate,
                main_value,
                main_score,
                main_state,
                main_state_table,
                main_ape,
                positions,
                main_bounds,
                main_lengths,
                main_state_slots,
                main_pooled,
                main_gamma,
                main_cos,
                main_sin,
                main_offsets,
                main_slots,
                main_normed,
                main_debug,
                main_cache,
                main_requests,
                main_project_workers,
                qa_tid,
            )
        )
        (q_proj, q_flat, kv_fp32, kv, cache_rows) = qkv_tail_tasks(
            normed,
            qr_matmul,
            qr_scale_pad,
            wq_b,
            wq_b_scale,
            rope_cos,
            rope_signed,
            rope_swap,
            q_proj,
            q_flat,
            wkv,
            gamma_ckv,
            kv_fp32,
            kv,
            native_slots,
            cache_rows,
            tokens,
            kv_workers,
        )
    with pl.spmd(
        GROUPS, name_hint="indexer_score", allow_early_resolve=True, sync_start=True, deps=[idx_cache_tid]
    ) as score_tid:
        (indexer_transfer, score_arena, pair_arena) = index_score_topk_group(
            positions,
            idx_bounds,
            idx_lengths,
            idx_cache,
            indexer_table,
            idx_query_i8,
            group_offsets,
            group_plan,
            coefficients,
            idx_key0,
            idx_key64,
            indexer_transfer,
            ffts,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            SPLIT_SCORE_ROWS,
            score_arena,
            pair_arena,
            SCALE_PAGE_COALESCING,
        )
    with pl.spmd(
        pl.min(tokens, VECTORS), name_hint="indexer_topk", allow_early_resolve=True, sync_start=True, deps=[score_tid]
    ) as topk_tid:
        (topk_scores, topk_indices) = indexer_pair_publish_stage(
            idx_bounds,
            idx_lengths,
            positions,
            group_plan,
            group_offsets,
            attention_indices,
            pair_arena,
            topk_scores,
            topk_indices,
            pl.tile.get_block_idx(),
            VECTORS,
            ROOTS,
            BALANCED,
            QUERY_GROUP,
            EARLY_ATTENTION_GEOMETRY,
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS),
        name_hint="csa_attention_plan",
        allow_early_resolve=True,
        sync_start=True,
        deps=[topk_tid],
    ):
        (attention_indices, attention_bias, attention_mask) = attention_slot_plan_stage(
            topk_indices,
            positions,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            attention_mask,
            pl.tensor.dim(cache_rows, 0) // 32,
            pl.tensor.dim(main_cache, 0) // 32,
            pl.tile.get_block_idx(),
            VECTORS,
            ATTENTION_QUERY_METADATA,
            EARLY_ATTENTION_GEOMETRY,
            PLAN_PHYSICAL_ROWS,
            PLAN_TABLE_COLUMNS,
        )
    with pl.spmd(GROUPS, name_hint="sparse_attention", allow_early_resolve=True, sync_start=True) as attn_tid:
        (score_ring, pv_ring, kv_ring, probability_ring, mi_ring, li_ring, packed) = sparse_attn_group(
            q_heads,
            attention_mask,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            ffts,
            rope_cos,
            cache_rows,
            positions,
            main_cache,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            inverse_sin,
            attn_sink,
            mi_ring,
            li_ring,
            packed,
            ATTENTION_QUERY_METADATA,
            LOCAL_ATTENTION_STATS,
            PLAN_PHYSICAL_ROWS,
            PLAN_TABLE_COLUMNS,
        )
    (o_r, o_quant, o_scales, o_partials, output) = output_tasks(
        packed,
        wo_a,
        wo_b,
        wo_b_scale,
        x,
        post,
        comb,
        o_r,
        o_quant,
        o_scales,
        o_partials,
        output,
        attn_tid,
        OA_COLS,
        OB_ROWS,
    )
    return (cache_rows, main_state, main_cache, idx_state, idx_cache, topk_scores, topk_indices, output)


@pl.jit
def decode_csa_tnd(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_cq: pl.Tensor[[1024], pl.BF16],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_sin: pl.Tensor[[T, 64], pl.FP32],
    wkv: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    main_wkv: pl.Tensor[[1024, 4096], pl.BF16],
    main_wgate: pl.Tensor[[1024, 4096], pl.BF16],
    main_state: pl.InOut[pl.Tensor[[MSTATE_PAGES, MSTATE_ELEMS], pl.FP32]],
    main_state_table: pl.Tensor[[B, MTABLE_COLS], pl.INT32],
    main_ape: pl.Tensor[[4, 1024], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    main_bounds: pl.Tensor[[MQ], pl.INT32],
    main_lengths: pl.Tensor[[B], pl.INT32],
    main_state_slots: pl.Tensor[[T, 2], pl.INT32],
    main_gamma: pl.Tensor[[512], pl.BF16],
    main_cos: pl.Tensor[[MC, 64], pl.FP32],
    main_sin: pl.Tensor[[MC, 64], pl.FP32],
    main_slots: pl.Tensor[[MSLOTS, 2], pl.INT32],
    main_cache: pl.InOut[pl.Tensor[[MCACHE_ROWS, 512], pl.BF16]],
    idx_wkv: pl.Tensor[[256, 4096], pl.BF16],
    idx_wgate: pl.Tensor[[256, 4096], pl.BF16],
    idx_state: pl.InOut[pl.Tensor[[ISTATE_PAGES, ISTATE_ELEMS], pl.FP32]],
    idx_state_table: pl.Tensor[[B, ITABLE_COLS], pl.INT32],
    idx_ape: pl.Tensor[[4, 256], pl.FP32],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    idx_state_slots: pl.Tensor[[T, 2], pl.INT32],
    idx_gamma: pl.Tensor[[128], pl.BF16],
    idx_cos: pl.Tensor[[IC, 64], pl.FP32],
    idx_sin: pl.Tensor[[IC, 64], pl.FP32],
    idx_slots: pl.Tensor[[ISLOTS, 2], pl.INT32],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    idx_cache: pl.InOut[pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8]],
    original_table: pl.Tensor[[B, ORIGINAL_COLS], pl.INT32],
    compressed_table: pl.Tensor[[B, COMPRESSED_COLS], pl.INT32],
    indexer_table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8, QUANT_WEIGHT_LAYOUT],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    weights_proj: pl.Tensor[[64, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    attn_sink: pl.Tensor[[64, 1], pl.FP32],
    wo_a: pl.Tensor[[8, 4096, 1024], pl.BF16, BF16_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[8192, 4096], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[4096], pl.FP32],
    output: pl.InOut[pl.Tensor[[T, 16384], pl.BF16]],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    ROOTS: pl.constexpr,
    OA_COLS: pl.constexpr,
    OB_ROWS: pl.constexpr,
    ATTENTION_QUERY_METADATA: pl.constexpr,
    HC_CONSUMER_REDUCE: pl.constexpr,
    QA_M16_PARALLEL: pl.constexpr = False,
    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    SCALE_PAGE_COALESCING: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,
    SPLIT_SCORE_ROWS: pl.constexpr = False,
    LIMIT_EMPTY_WORKERS: pl.constexpr = False,
    BSH_PROJECTION_ORDER: pl.constexpr = False,
):
    t = pl.tensor.dim(x, 0)
    topk_indices = pl.create_tensor([t, 512], dtype=pl.INT32)
    topk_scores = pl.create_tensor([t, 512], dtype=pl.FP32)
    b = pl.tensor.dim(idx_lengths, 0)
    p8 = (t + 7) // 8 * 8
    p16 = (t + 15) // 16 * 16
    p64 = (t + 63) // 64 * 64
    p128 = (t + 127) // 128 * 128
    group = QUERY_GROUP
    mm = pl.max(p128, (t + OB_ROWS - 1) // OB_ROWS * OB_ROWS)
    x_fp32 = pl.create_tensor([t, 16384], dtype=pl.FP32)
    inv = pl.create_tensor([p16, 1], dtype=pl.FP32)
    widen_tail = pl.create_tensor([8, 16384], dtype=pl.FP32)
    partials = pl.create_tensor([p16 * 4, 32], dtype=pl.FP32)
    raw = pl.create_tensor([p16, 32], dtype=pl.FP32)
    pre = pl.create_tensor([p16, 8], dtype=pl.FP32)
    post = pl.create_tensor([t, 4], dtype=pl.FP32)
    post_tail = pl.create_tensor([8, 8], dtype=pl.FP32)
    comb = pl.create_tensor([t, 16], dtype=pl.FP32)
    comb_tail = pl.create_tensor([8, 32], dtype=pl.FP32)
    mixed = pl.create_tensor([p8, 4096], dtype=pl.BF16)
    normed = pl.create_tensor([t, 4096], dtype=pl.BF16)
    norm_tail = pl.create_tensor([8, 4096], dtype=pl.BF16)
    qa = pl.create_tensor([p16, 1024], dtype=pl.FP32)
    qr = pl.create_tensor([t, 1024], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t, 1], dtype=pl.FP32)
    qr_matmul = pl.create_tensor([p128, 1024], dtype=pl.INT8)
    qr_scale_pad = pl.create_tensor([p8, 1], dtype=pl.FP32)
    rope_signed = pl.create_tensor([t, 64], dtype=pl.FP32)
    rope_swap = pl.create_tensor([t, 64], dtype=pl.INT32)
    q_proj = pl.create_tensor([p128, 32768], dtype=pl.INT32)
    q_flat = pl.create_tensor([t, 32768], dtype=pl.BF16)
    kv_fp32 = pl.create_tensor([p16, 512], dtype=pl.FP32)
    kv = pl.create_tensor([t, 512], dtype=pl.BF16)
    main_value = pl.create_tensor([p64, 1024], dtype=pl.FP32)
    main_score = pl.create_tensor([p64, 1024], dtype=pl.FP32)
    main_pooled = pl.create_tensor([t, 512], dtype=pl.FP32)
    main_offsets = pl.create_tensor([b], dtype=pl.INT32)
    main_count = pl.create_tensor([1], dtype=pl.INT32)
    main_normed = pl.create_tensor([t, 512], dtype=pl.FP32)
    main_debug = pl.create_tensor([t, 512], dtype=pl.FP32)
    idx_value = pl.create_tensor([p16, 256], dtype=pl.FP32)
    idx_score = pl.create_tensor([p16, 256], dtype=pl.FP32)
    idx_pooled = pl.create_tensor([t, 128], dtype=pl.FP32)
    idx_offsets = pl.create_tensor([b], dtype=pl.INT32)
    idx_count = pl.create_tensor([1], dtype=pl.INT32)
    idx_normed = pl.create_tensor([p16, 128], dtype=pl.BF16)
    idx_had = pl.create_tensor([p16, 128], dtype=pl.FP32)
    idx_scales = pl.create_tensor([p16, 1], dtype=pl.FP32)
    idx_debug = pl.create_tensor([t, 128], dtype=pl.FP32)
    idx_query_acc = pl.create_tensor([p64, 8192], dtype=pl.INT32)
    idx_query_rope = pl.create_tensor([p16, 8192], dtype=pl.BF16)
    idx_query_had = pl.create_tensor([t * 64, 128], dtype=pl.FP32)
    idx_query_i8 = pl.create_tensor([t * 64, 128], dtype=pl.INT8)
    idx_query_scale = pl.create_tensor([t * 64, 1], dtype=pl.FP32)
    weights_partial = pl.create_tensor([p16 * 4, 64], dtype=pl.FP32)
    head_weights = pl.create_tensor([p16, 64], dtype=pl.FP32)
    group_offsets = pl.create_tensor(
        [((b + 1 + 16 - 1) // 16 * 16 + 4 * t + 16 - 1) // 16 * 16 + 48 + 4 * t], dtype=pl.INT32
    )
    group_plan = pl.create_tensor([8], dtype=pl.INT32)
    coefficients = pl.create_tensor([t * 16, group * 64], dtype=pl.FP16)
    indexer_transfer = pl.create_tensor([24 * 2 * group, SCORE_TILE], dtype=pl.FP32)
    score_arena = pl.create_tensor([24 * 2 * group, 8192], dtype=pl.FP32)
    pair_arena = pl.create_tensor([t * 64, 1024], dtype=pl.FP32)
    attention_indices = pl.create_tensor([p8, 512], dtype=pl.INT32)
    attention_bias = pl.create_tensor([p8, 640], dtype=pl.FP32)
    attention_mask = pl.create_tensor([p8, 16], dtype=pl.INT32)
    inverse_sin = pl.create_tensor([t, 64], dtype=pl.FP32)
    kv_ring = pl.create_tensor([24 * 3 * 128, 512], dtype=pl.BF16)
    score_ring = pl.create_tensor([24 * 3 * 64, 128], dtype=pl.FP32)
    probability_ring = pl.create_tensor([24 * 3 * 64, 128], dtype=pl.BF16)
    pv_ring = pl.create_tensor([24 * 3 * 64, 512], dtype=pl.FP32)
    mi_ring = pl.create_tensor([24 * 3 * 64, 1], dtype=pl.FP32)
    li_ring = pl.create_tensor([24 * 3 * 64, 1], dtype=pl.FP32)
    packed = pl.create_tensor([8 * p128, 4096], dtype=pl.BF16)
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    o_r = pl.create_tensor([mm, 8192], dtype=pl.FP32)
    o_quant = pl.create_tensor([mm, 8192], dtype=pl.INT8)
    o_scales = pl.create_tensor([8, p128], dtype=pl.FP32)
    o_partials = pl.create_tensor([mm, 32768], dtype=pl.INT32)
    q_heads = pl.reshape(q_flat, [t * 64, 512])
    rope_rows = pl.tensor.dim(idx_query_rope, 0)
    idx_query_rope_heads = pl.reshape(idx_query_rope, [rope_rows * 64, 128])
    idx_query_scale_heads = pl.reshape(idx_query_scale, [t, 64])
    tokens = pl.tensor.dim(x, 0)
    requests = pl.tensor.dim(idx_lengths, 0)
    main_requests = pl.tensor.dim(main_lengths, 0)
    q_heads = pl.reshape(q_flat, [tokens * 64, 512])
    idx_rope_tokens = pl.tensor.dim(idx_query_rope, 0)
    idx_query_rope_heads = pl.reshape(idx_query_rope, [idx_rope_tokens * 64, 128])
    idx_query_scale_heads = pl.reshape(idx_query_scale, [tokens, 64])
    pages = pl.tensor.dim(idx_cache, 0)
    page_bytes = pl.tensor.dim(idx_cache, 1)
    cache_flat = pl.reshape(idx_cache, [pages * page_bytes])
    key_prefix = cache_flat[0 : pages * page_bytes // 128 * 128]
    key_shifted = cache_flat[64 : 64 + (pages * page_bytes - 64) // 128 * 128]
    idx_key0 = pl.reshape(key_prefix, [pages * page_bytes // 128, 128])
    idx_key64 = pl.reshape(key_shifted, [(pages * page_bytes - 64) // 128, 128])
    qa_workers = GROUPS
    idx_project_workers = GROUPS
    idx_hadamard_workers = GROUPS
    query_rope_workers = VECTORS
    query_hadamard_workers = GROUPS
    weight_workers = GROUPS
    main_project_workers = GROUPS
    kv_workers = GROUPS
    if LIMIT_EMPTY_WORKERS:
        if QA_M16_PARALLEL:
            qa_units = QR_N_BLOCKS * ((tokens + QR_M_TILE - 1) // QR_M_TILE)
        elif tokens < QR_FIXED_SMALL_ROWS:
            qa_units = QR_N_BLOCKS * pl.min(QR_FIXED_M_GROUPS, pl.max(1, tokens // QR_FIXED_SMALL_M_TILE))
        else:
            qa_units = QR_N_BLOCKS * pl.min(QR_FIXED_M_GROUPS, pl.max(1, tokens // QR_DENSE_M_TILE))
        qa_workers = pl.min(GROUPS, qa_units)
        idx_project_workers = pl.min(
            GROUPS, INDEXER_OUT_DIM // INDEXER_N_TILE * ((tokens + INDEXER_M_TILE - 1) // INDEXER_M_TILE)
        )
        compact_hadamard_units = (
            COMPACT_HEAD_DIM
            // COMPACT_HAD_COL_TILE
            * ((pl.tensor.dim(idx_normed, 0) + COMPACT_HAD_ROW_TILE - 1) // COMPACT_HAD_ROW_TILE)
        )
        idx_hadamard_workers = pl.min(GROUPS, compact_hadamard_units)
        query_rope_workers = pl.min(
            VECTORS, IDX_N_HEADS // DQ_ROPE_H_TILE * ((tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE)
        )
        query_hadamard_workers = pl.min(GROUPS, (tokens + 3) // 4)
        weight_workers = pl.min(GROUPS, WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE))
        main_project_workers = pl.min(GROUPS, CMP_OUT_DIM // CMP_N_TILE * ((tokens + CMP_M_TILE - 1) // CMP_M_TILE))
        kv_workers = KV_HEAD_DIM // KV_N_TILE * KV_OM
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_widen_rms", allow_early_resolve=True, sync_start=True
    ):
        (x_fp32, inv) = hc_widen_rms_range(x, x_fp32, inv, widen_tail, pl.tile.get_block_idx(), VECTORS)
    with pl.spmd(
        pl.min((tokens + 15) // 16 * 4, GROUPS), name_hint="hc_linear", allow_early_resolve=True, sync_start=True
    ):
        partials = hc_project_range(x_fp32, hc_fn, partials, pl.tile.get_block_idx(), GROUPS)
    with pl.spmd(
        pl.min((tokens + 15) // 16, VECTORS), name_hint="hc_reduce", allow_early_resolve=True, sync_start=True
    ):
        raw = hc_reduce_range(partials, raw, pl.tile.get_block_idx(), VECTORS)
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_pre_gates", allow_early_resolve=True, sync_start=True
    ):
        (pre, post) = hc_gates_range(
            inv,
            raw,
            hc_scale,
            hc_base,
            pre,
            post,
            post_tail,
            pl.tile.get_block_idx(),
            VECTORS,
            partials,
            HC_CONSUMER_REDUCE,
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_mix_norm", allow_early_resolve=True, sync_start=True
    ):
        (mixed, normed) = hc_mix_norm_range(
            x_fp32, pre, norm_w, mixed, normed, norm_tail, pl.tile.get_block_idx(), VECTORS
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="hc_sinkhorn", allow_early_resolve=True, sync_start=True
    ):
        comb = hc_comb_range(
            inv,
            raw,
            hc_scale,
            hc_base_2d,
            comb,
            comb_tail,
            pl.tile.get_block_idx(),
            VECTORS,
            partials,
            HC_CONSUMER_REDUCE,
        )
    (main_offsets, main_count) = compact_offsets_and_count_stage(
        main_bounds, main_lengths, main_offsets, main_count, tokens, 0
    )
    (idx_offsets, idx_count) = compact_offsets_and_count_stage(
        idx_bounds, idx_lengths, idx_offsets, idx_count, tokens, 0
    )
    with pl.spmd(
        pl.min((tokens + 3) // 4, VECTORS), name_hint="csa_rope_sign", allow_early_resolve=True, sync_start=False
    ):
        (rope_signed, rope_swap) = q_rope_prepare_stage(
            rope_sin, rope_signed, rope_swap, pl.tile.get_block_idx(), VECTORS
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="csa_inverse_rope", allow_early_resolve=True, sync_start=False
    ):
        inverse_sin = attention_inverse_rope_sign_stage(rope_sin, inverse_sin, pl.tile.get_block_idx(), VECTORS)
    if EARLY_ATTENTION_GEOMETRY:
        with pl.spmd(
            pl.min((tokens + 7) // 8, VECTORS),
            name_hint="csa_query_geometry",
            allow_early_resolve=True,
            sync_start=False,
        ):
            attention_indices = attention_geometry_prepare_stage(
                positions,
                idx_bounds,
                idx_lengths,
                compressed_table,
                attention_indices,
                pl.tensor.dim(main_cache, 0) // 32,
                pl.tile.get_block_idx(),
                VECTORS,
                PLAN_PHYSICAL_ROWS,
                PLAN_TABLE_COLUMNS,
            )
    with pl.spmd(qa_workers, name_hint="qproj_a", allow_early_resolve=True, sync_start=False) as qa_tid:
        qa = q_a_stage(normed, wq_a, qa, pl.tile.get_block_idx(), GROUPS, QA_M16_PARALLEL)
    projection_dependencies = pl.array.create(1, pl.TASK_ID)
    if not BSH_PROJECTION_ORDER:
        with pl.spmd(
            idx_project_workers, name_hint="indexer_compressor_kv_score", allow_early_resolve=True, sync_start=False
        ):
            (idx_value, idx_score) = indexer_compressor_project_stage(
                normed, idx_wkv, idx_wgate, idx_value, idx_score, pl.tile.get_block_idx(), GROUPS
            )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS), name_hint="qproj_a_rms_quant", allow_early_resolve=True, sync_start=False
    ):
        (qr, qr_scale, qr_matmul, qr_scale_pad) = q_r_stage(
            qa, gamma_cq, qr, qr_scale, qr_matmul, qr_scale_pad, pl.tile.get_block_idx(), VECTORS
        )
    if BSH_PROJECTION_ORDER:
        (q_proj, q_flat, kv_fp32, kv, cache_rows) = qkv_tail_tasks(
            normed,
            qr_matmul,
            qr_scale_pad,
            wq_b,
            wq_b_scale,
            rope_cos,
            rope_signed,
            rope_swap,
            q_proj,
            q_flat,
            wkv,
            gamma_ckv,
            kv_fp32,
            kv,
            native_slots,
            cache_rows,
            tokens,
            kv_workers,
        )
        (main_value, main_score, main_pooled, main_state, main_normed, main_debug, main_cache, main_project_tid) = (
            main_compressor_tasks(
                normed,
                main_wkv,
                main_wgate,
                main_value,
                main_score,
                main_state,
                main_state_table,
                main_ape,
                positions,
                main_bounds,
                main_lengths,
                main_state_slots,
                main_pooled,
                main_gamma,
                main_cos,
                main_sin,
                main_offsets,
                main_slots,
                main_normed,
                main_debug,
                main_cache,
                main_requests,
                main_project_workers,
                qa_tid,
            )
        )
        projection_dependencies[0] = main_project_tid
        with pl.spmd(
            idx_project_workers, name_hint="indexer_compressor_kv_score", allow_early_resolve=True, sync_start=False
        ):
            (idx_value, idx_score) = indexer_compressor_project_stage(
                normed, idx_wkv, idx_wgate, idx_value, idx_score, pl.tile.get_block_idx(), GROUPS
            )
    with pl.spmd(
        pl.min(requests, VECTORS), name_hint="indexer_compressor_pool", allow_early_resolve=True, sync_start=False
    ) as idx_pool_tid:
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        pl.min(requests, VECTORS),
        name_hint="indexer_state_commit",
        allow_early_resolve=True,
        sync_start=False,
        deps=[idx_pool_tid],
    ):
        idx_state = indexer_compressor_state_commit_stage(
            idx_value,
            idx_score,
            idx_ape,
            positions,
            idx_bounds,
            idx_lengths,
            idx_state_slots,
            idx_state,
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        pl.min(requests, VECTORS), name_hint="indexer_compressor_norm_rope", allow_early_resolve=True, sync_start=False
    ):
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    if BSH_PROJECTION_ORDER:
        with pl.spmd(
            idx_hadamard_workers,
            name_hint="indexer_compressor_hadamard",
            allow_early_resolve=True,
            sync_start=False,
            deps=[projection_dependencies[0]],
        ):
            compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
            idx_had = indexer_compressor_hadamard_stage(
                idx_normed, hadamard, idx_had, compact_count, pl.tile.get_block_idx(), GROUPS
            )
    else:
        compact_count = pl.cast(pl.read(idx_count, [0]), pl.INDEX)
        compact_hadamard_tasks = (
            (compact_count + COMPACT_HAD_ROW_TILE - 1)
            // COMPACT_HAD_ROW_TILE
            * (COMPACT_HEAD_DIM // COMPACT_HAD_COL_TILE)
        )
        idx_hadamard_workers = pl.max(1, pl.min(GROUPS, compact_hadamard_tasks))
        with pl.spmd(
            idx_hadamard_workers, name_hint="indexer_compressor_hadamard", allow_early_resolve=True, sync_start=False
        ):
            idx_had = indexer_compressor_hadamard_stage(
                idx_normed, hadamard, idx_had, compact_count, pl.tile.get_block_idx(), GROUPS
            )
    with pl.spmd(GROUPS, name_hint="indexer_query_proj", allow_early_resolve=True, sync_start=False):
        idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, pl.tile.get_block_idx(), GROUPS)
    with pl.spmd(query_rope_workers, name_hint="indexer_query_rope", allow_early_resolve=True, sync_start=False):
        idx_query_rope = indexer_query_dequant_rope_stage(
            qr_scale,
            idx_wq_b_scale,
            rope_cos,
            rope_signed,
            idx_query_acc,
            idx_query_rope,
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        query_hadamard_workers, name_hint="indexer_query_hadamard", allow_early_resolve=True, sync_start=False
    ):
        idx_query_had = indexer_query_hadamard_four_tokens_stage(
            idx_query_rope_heads, hadamard, idx_query_had, tokens, pl.tile.get_block_idx(), GROUPS
        )
    with pl.spmd(pl.min(tokens, VECTORS), name_hint="indexer_query_quant", allow_early_resolve=True, sync_start=False):
        (idx_query_i8, idx_query_scale) = indexer_query_quant_stage(
            idx_query_had, idx_query_i8, idx_query_scale, tokens, pl.tile.get_block_idx(), VECTORS
        )
    with pl.spmd(weight_workers, name_hint="indexer_weights_proj", allow_early_resolve=True, sync_start=False):
        weights_partial = indexer_weights_project_stage(
            normed, weights_proj, weights_partial, pl.tile.get_block_idx(), GROUPS
        )
    with pl.spmd(
        pl.min((tokens + 15) // 16, VECTORS),
        name_hint="indexer_weights_reduce",
        allow_early_resolve=True,
        sync_start=False,
    ):
        head_weights = indexer_weights_reduce_stage(
            weights_partial, head_weights, tokens, pl.tile.get_block_idx(), VECTORS
        )
    (group_offsets, group_plan) = indexer_group_plan_stage(
        idx_bounds, idx_lengths, positions, group_offsets, group_plan, tokens, 0, QUERY_GROUP, BALANCED, ROOTS
    )
    group_offsets = indexer_scale_page_plan_stage(
        idx_bounds,
        idx_lengths,
        positions,
        idx_offsets,
        idx_slots,
        idx_cache,
        group_offsets,
        pl.tensor.dim(idx_scales, 0),
        VECTORS,
    )
    coefficient_workers = VECTORS
    if not BSH_PROJECTION_ORDER:
        coefficient_groups = pl.cast(pl.read(group_offsets, [requests]), pl.INDEX)
        coefficient_workers = pl.max(1, pl.min(VECTORS, coefficient_groups))
    with pl.spmd(
        coefficient_workers, name_hint="indexer_head_coefficients", allow_early_resolve=True, sync_start=False
    ):
        coefficients = indexer_head_coefficients_stage(
            idx_query_scale_heads,
            head_weights,
            positions,
            idx_bounds,
            idx_lengths,
            group_offsets,
            coefficients,
            pl.tile.get_block_idx(),
            VECTORS,
            QUERY_GROUP,
        )
    with pl.spmd(
        pl.min(requests, VECTORS),
        name_hint="indexer_compressor_cache_write",
        allow_early_resolve=True,
        sync_start=False,
    ):
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
            pl.tile.get_block_idx(),
            VECTORS,
        )
    with pl.spmd(
        VECTORS, name_hint="indexer_scale_page_commit", allow_early_resolve=True, sync_start=False
    ) as idx_cache_tid:
        idx_cache = indexer_scale_page_commit_task(idx_scales, group_offsets, idx_cache, tokens, requests)
    if not BSH_PROJECTION_ORDER:
        (main_value, main_score, main_pooled, main_state, main_normed, main_debug, main_cache, main_project_tid) = (
            main_compressor_tasks(
                normed,
                main_wkv,
                main_wgate,
                main_value,
                main_score,
                main_state,
                main_state_table,
                main_ape,
                positions,
                main_bounds,
                main_lengths,
                main_state_slots,
                main_pooled,
                main_gamma,
                main_cos,
                main_sin,
                main_offsets,
                main_slots,
                main_normed,
                main_debug,
                main_cache,
                main_requests,
                main_project_workers,
                qa_tid,
            )
        )
        (q_proj, q_flat, kv_fp32, kv, cache_rows) = qkv_tail_tasks(
            normed,
            qr_matmul,
            qr_scale_pad,
            wq_b,
            wq_b_scale,
            rope_cos,
            rope_signed,
            rope_swap,
            q_proj,
            q_flat,
            wkv,
            gamma_ckv,
            kv_fp32,
            kv,
            native_slots,
            cache_rows,
            tokens,
            kv_workers,
        )
    with pl.spmd(
        GROUPS, name_hint="indexer_score", allow_early_resolve=True, sync_start=True, deps=[idx_cache_tid]
    ) as score_tid:
        (indexer_transfer, score_arena, pair_arena) = index_score_topk_group(
            positions,
            idx_bounds,
            idx_lengths,
            idx_cache,
            indexer_table,
            idx_query_i8,
            group_offsets,
            group_plan,
            coefficients,
            idx_key0,
            idx_key64,
            indexer_transfer,
            ffts,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            SPLIT_SCORE_ROWS,
            score_arena,
            pair_arena,
            SCALE_PAGE_COALESCING,
        )
    with pl.spmd(
        pl.min(tokens, VECTORS), name_hint="indexer_topk", allow_early_resolve=True, sync_start=True, deps=[score_tid]
    ) as topk_tid:
        (topk_scores, topk_indices) = indexer_pair_publish_stage(
            idx_bounds,
            idx_lengths,
            positions,
            group_plan,
            group_offsets,
            attention_indices,
            pair_arena,
            topk_scores,
            topk_indices,
            pl.tile.get_block_idx(),
            VECTORS,
            ROOTS,
            BALANCED,
            QUERY_GROUP,
            EARLY_ATTENTION_GEOMETRY,
        )
    with pl.spmd(
        pl.min((tokens + 7) // 8, VECTORS),
        name_hint="csa_attention_plan",
        allow_early_resolve=True,
        sync_start=True,
        deps=[topk_tid],
    ):
        (attention_indices, attention_bias, attention_mask) = attention_slot_plan_stage(
            topk_indices,
            positions,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            attention_mask,
            pl.tensor.dim(cache_rows, 0) // 32,
            pl.tensor.dim(main_cache, 0) // 32,
            pl.tile.get_block_idx(),
            VECTORS,
            ATTENTION_QUERY_METADATA,
            EARLY_ATTENTION_GEOMETRY,
            PLAN_PHYSICAL_ROWS,
            PLAN_TABLE_COLUMNS,
        )
    with pl.spmd(GROUPS, name_hint="sparse_attention", allow_early_resolve=True, sync_start=True) as attn_tid:
        (score_ring, pv_ring, kv_ring, probability_ring, mi_ring, li_ring, packed) = sparse_attn_group(
            q_heads,
            attention_mask,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            ffts,
            rope_cos,
            cache_rows,
            positions,
            main_cache,
            idx_bounds,
            idx_lengths,
            original_table,
            compressed_table,
            attention_indices,
            attention_bias,
            inverse_sin,
            attn_sink,
            mi_ring,
            li_ring,
            packed,
            ATTENTION_QUERY_METADATA,
            LOCAL_ATTENTION_STATS,
            PLAN_PHYSICAL_ROWS,
            PLAN_TABLE_COLUMNS,
        )
    (o_r, o_quant, o_scales, o_partials, output) = output_tasks(
        packed,
        wo_a,
        wo_b,
        wo_b_scale,
        x,
        post,
        comb,
        o_r,
        o_quant,
        o_scales,
        o_partials,
        output,
        attn_tid,
        OA_COLS,
        OB_ROWS,
    )
    return (cache_rows, main_state, main_cache, idx_state, idx_cache, output)
