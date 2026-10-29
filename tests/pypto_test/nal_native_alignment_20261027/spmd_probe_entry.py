# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403, F405 -- Keep the diagnostic root globals identical to its production source.
"""从生产debug根仅替换28个内部分配为外部观测缓冲；算术和调度调用相同。

使用前必须证明该根与未经观测的生产debug根的全部8个外部输出逐字节一致。
"""

from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.entry import *
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.entry import _csa_core_group


@pl.jit
def spmd_stage_probe(
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
    debug_inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    debug_post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    debug_comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    debug_mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    debug_normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    debug_qa: pl.InOut[pl.Tensor[[P16, 1024], pl.FP32]],
    debug_qr: pl.InOut[pl.Tensor[[T, 1024], pl.INT8]],
    debug_qr_scale: pl.InOut[pl.Tensor[[T, 1], pl.FP32]],
    debug_kv_fp32: pl.InOut[pl.Tensor[[P16, 512], pl.FP32]],
    debug_kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    debug_q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    debug_q: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    debug_idx_pooled: pl.InOut[pl.Tensor[[T, 128], pl.FP32]],
    debug_idx_normed: pl.InOut[pl.Tensor[[P16, 128], pl.BF16]],
    debug_idx_had: pl.InOut[pl.Tensor[[P16, 128], pl.FP32]],
    debug_idx_scales: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    debug_idx_debug: pl.InOut[pl.Tensor[[T, 128], pl.FP32]],
    debug_idx_query_i8: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8]],
    debug_idx_query_scale: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 1], pl.FP32]],
    debug_idx_query_acc: pl.InOut[pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32]],
    debug_idx_query_rope: pl.InOut[pl.Tensor[[IDX_ROPE_ROWS, 8192], pl.BF16]],
    debug_idx_query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    debug_head_weights: pl.InOut[pl.Tensor[[P16, 64], pl.FP32]],
    debug_packed: pl.InOut[pl.Tensor[[PACK_ROWS, 4096], pl.BF16]],
    debug_o_r: pl.InOut[pl.Tensor[[O_MM_ROWS, 8192], pl.FP32]],
    debug_main_pooled: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    debug_main_normed: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    debug_main_debug: pl.InOut[pl.Tensor[[T, 512], pl.FP32]],
    QUERY_GROUP: pl.constexpr,
    QK_PANEL: pl.constexpr,
    SCORE_TILE: pl.constexpr,
    PREFETCH: pl.constexpr,
    TO_L0: pl.constexpr,
    BALANCED: pl.constexpr,
    ROOTS: pl.constexpr,
    OA_COLS: pl.constexpr,
    OA_ROWS: pl.constexpr,
    OB_ROWS: pl.constexpr,
    OUTPUT_GROUP_PIPELINE: pl.constexpr,
    HC_ROW_DEPENDENCIES: pl.constexpr,
    ATTENTION_QUERY_METADATA: pl.constexpr,
    HC_CONSUMER_REDUCE: pl.constexpr,
    QA_M16_PARALLEL: pl.constexpr = False,
    COOPERATIVE_PREFIX: pl.constexpr = False,
    EARLY_ATTENTION_GEOMETRY: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    SCALE_PAGE_COALESCING: pl.constexpr = False,
    LEAF_SCALE_LAYOUT: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
    SPLIT_SCORE_ROWS: pl.constexpr = False,
    OUTPUT_WORK_BALANCE: pl.constexpr = False,
    SHORT_HALF_SCORE_UB: pl.constexpr = False,
    SHORT_TOPK_PREFIX: pl.constexpr = False,
    QB_WIDE_N: pl.constexpr = False,
    PREFIX_BRANCH_GROUPS: pl.constexpr = 4,
    JOINT_BF16_PREFIX: pl.constexpr = False,
    PREFIX_PROJECTION_ROWS: pl.constexpr = 128,
    SCORE_GROUP_ORDER: pl.constexpr = False,
    FUSED_INDEXER_COEFF: pl.constexpr = False,
    PLAN_TABLE_COLUMNS: pl.constexpr = PHYSICAL_PLAN_COLUMNS,
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
    inv = debug_inv
    widen_tail = pl.create_tensor([8, 16384], dtype=pl.FP32)
    partials = pl.create_tensor([p16 * 22, 32], dtype=pl.FP32)
    hc_squares = pl.create_tensor([p16 * 22, 16], dtype=pl.FP32)
    hc_packed_input = pl.create_tensor([p16, 32768], dtype=pl.BF16)
    hc_packed_weight = pl.create_tensor([32768, 32], dtype=pl.BF16)
    hc_norm_rstd = pl.create_tensor([t, 8], dtype=pl.FP32)
    attention_copy64 = pl.create_tensor([24 * 64, 64], dtype=pl.FP32)
    raw = pl.create_tensor([p16, 32], dtype=pl.FP32)
    pre = pl.create_tensor([p16, 8], dtype=pl.FP32)
    post = debug_post
    post_tail = pl.create_tensor([8, 8], dtype=pl.FP32)
    comb = debug_comb
    comb_tail = pl.create_tensor([8, 32], dtype=pl.FP32)
    mixed = debug_mixed
    normed = debug_normed
    norm_tail = pl.create_tensor([8, 4096], dtype=pl.BF16)
    qa = debug_qa
    qr = debug_qr
    qr_scale = debug_qr_scale
    qr_matmul = pl.create_tensor([p128, 1024], dtype=pl.INT8)
    qr_scale_pad = pl.create_tensor([p8, 1], dtype=pl.FP32)
    rope_signed = pl.create_tensor([t, 64], dtype=pl.FP32)
    rope_swap = pl.create_tensor([t, 64], dtype=pl.INT32)
    q_proj = debug_q_proj
    q_flat = debug_q
    kv_fp32 = debug_kv_fp32
    kv = debug_kv
    main_value = pl.create_tensor([p64, 1024], dtype=pl.FP32)
    main_score = pl.create_tensor([p64, 1024], dtype=pl.FP32)
    main_pooled = debug_main_pooled
    main_offsets = pl.create_tensor([b], dtype=pl.INT32)
    main_count = pl.create_tensor([1], dtype=pl.INT32)
    main_normed = debug_main_normed
    main_debug = debug_main_debug
    idx_value = pl.create_tensor([p16, 256], dtype=pl.FP32)
    idx_score = pl.create_tensor([p16, 256], dtype=pl.FP32)
    idx_pooled = debug_idx_pooled
    idx_offsets = pl.create_tensor([b], dtype=pl.INT32)
    idx_count = pl.create_tensor([1], dtype=pl.INT32)
    idx_normed = debug_idx_normed
    idx_had = debug_idx_had
    idx_scales = debug_idx_scales
    idx_debug = debug_idx_debug
    idx_query_acc = debug_idx_query_acc
    idx_query_rope = debug_idx_query_rope
    idx_query_had = debug_idx_query_had
    idx_query_i8 = debug_idx_query_i8
    idx_query_scale = debug_idx_query_scale
    weights_partial = pl.create_tensor([p16 * 4, 64], dtype=pl.FP32)
    head_weights = debug_head_weights
    # 原分区及其偏移不变；仅融合域在内部offset尾追加T个INT32映射。
    group_offset_words = ((b + 1 + 16 - 1) // 16 * 16 + 4 * t + 16 - 1) // 16 * 16 + 48 + 4 * t + 2 * b
    if FUSED_INDEXER_COEFF:
        group_offset_words = group_offset_words + t
    group_offsets = pl.create_tensor([group_offset_words], dtype=pl.INT32)
    group_plan = pl.create_tensor([8], dtype=pl.INT32)
    coefficients = pl.create_tensor([t * 16, group * 64], dtype=pl.FP16)
    indexer_transfer = pl.create_tensor([24 * 2 * group, SCORE_TILE], dtype=pl.FP32)
    score_arena = pl.create_tensor([24 * 2 * group, 8192], dtype=pl.FP32)
    pair_arena = pl.create_tensor([t * 64, 1024], dtype=pl.FP32)
    attention_indices = pl.create_tensor([p8, 512], dtype=pl.INT32)
    attention_bias = pl.create_tensor([p8, 640], dtype=pl.FP32)
    attention_mask = pl.create_tensor([p8, 16], dtype=pl.INT32)
    inverse_sin = pl.create_tensor([t, 64], dtype=pl.FP32)
    kv_ring = pl.create_tensor([24 * 4 * 128, 512], dtype=pl.BF16)
    score_ring = pl.create_tensor([24 * 64, 512], dtype=pl.FP32)
    probability_ring = pl.create_tensor([24 * 64, 512], dtype=pl.BF16)
    pv_ring = pl.create_tensor([24 * 4 * 64, 512], dtype=pl.FP32)
    mi_ring = pl.create_tensor([24 * 4 * 64, 1], dtype=pl.FP32)
    li_ring = pl.create_tensor([24 * 4 * 64, 1], dtype=pl.FP32)
    packed = debug_packed
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    o_r = debug_o_r
    o_quant = pl.create_tensor([mm, 8192], dtype=pl.INT8)
    o_scales = pl.create_tensor([8, p128], dtype=pl.FP32)
    o_partials = pl.create_tensor([mm, 32768], dtype=pl.INT32)
    signal = pl.create_tensor([PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS], dtype=pl.INT32)
    q_heads = pl.reshape(q_flat, [t * 64, 512])
    rope_rows = pl.tensor.dim(idx_query_rope, 0)
    idx_query_rope_heads = pl.reshape(idx_query_rope, [rope_rows * 64, 128])
    idx_query_scale_heads = pl.reshape(idx_query_scale, [t, 64])
    pages = pl.tensor.dim(idx_cache, 0)
    page_bytes = pl.tensor.dim(idx_cache, 1)
    cache_flat = pl.reshape(idx_cache, [pages * page_bytes])
    prefix = cache_flat[0 : pages * page_bytes // 128 * 128]
    shifted = cache_flat[64 : 64 + (pages * page_bytes - 64) // 128 * 128]
    idx_key0 = pl.reshape(prefix, [pages * page_bytes // 128, 128])
    idx_key64 = pl.reshape(shifted, [(pages * page_bytes - 64) // 128, 128])
    # 所有输入档位统一在主MIX内清零并硬同步；没有独立初始化task。
    # binding准备时检查24AIC/48AIV；全设备查询保证硬屏障全占用。
    signal_init_matrix = pl.reshape(signal, [VECTORS, PHASE_COUNT * CORES * SIGNAL_SLOT_WORDS // VECTORS])
    with pl.spmd(pl.system.available_cluster_count(), sync_start=True):
        (
            x_fp32,
            inv_after,
            widen_tail,
            partials,
            raw,
            pre,
            post_after,
            post_tail,
            comb_after,
            comb_tail,
            mixed_after,
            normed_after,
            norm_tail,
            qa_after,
            qr_after,
            qr_scale_after,
            qr_matmul,
            qr_scale_pad,
            rope_signed,
            rope_swap,
            q_proj_after,
            q_flat_after,
            kv_fp32_after,
            kv_after,
            cache_rows,
            main_value,
            main_score,
            main_state,
            main_pooled,
            main_offsets,
            main_count,
            main_normed,
            main_debug,
            main_cache,
            idx_value,
            idx_score,
            idx_state,
            idx_pooled_after,
            idx_offsets,
            idx_count,
            idx_normed_after,
            idx_had_after,
            idx_scales_after,
            idx_debug_after,
            idx_cache,
            idx_query_acc_after,
            idx_query_rope_after,
            idx_query_had_after,
            idx_query_i8_after,
            idx_query_scale_after,
            weights_partial,
            head_weights_after,
            group_offsets,
            group_plan,
            coefficients,
            indexer_transfer,
            score_arena,
            pair_arena,
            topk_scores,
            topk_indices,
            attention_indices,
            attention_bias,
            attention_mask,
            inverse_sin,
            kv_ring,
            score_ring,
            probability_ring,
            pv_ring,
            mi_ring,
            li_ring,
            packed,
            o_r_after,
            o_quant,
            o_scales,
            o_partials,
            output,
            signal,
            hc_squares,
            hc_packed_input,
            hc_packed_weight,
            hc_norm_rstd,
            attention_copy64,
        ) = _csa_core_group(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            hc_base_2d,
            norm_w,
            wq_a,
            gamma_cq,
            x_fp32,
            inv,
            widen_tail,
            partials,
            raw,
            pre,
            post,
            post_tail,
            comb,
            comb_tail,
            mixed,
            normed,
            norm_tail,
            qa,
            qr,
            qr_scale,
            qr_matmul,
            qr_scale_pad,
            wq_b,
            wq_b_scale,
            rope_cos,
            rope_sin,
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
            main_count,
            main_slots,
            main_normed,
            main_debug,
            main_cache,
            idx_wkv,
            idx_wgate,
            idx_value,
            idx_score,
            idx_state,
            idx_state_table,
            idx_ape,
            idx_bounds,
            idx_lengths,
            idx_state_slots,
            idx_pooled,
            idx_gamma,
            idx_cos,
            idx_sin,
            idx_offsets,
            idx_count,
            idx_slots,
            idx_normed,
            hadamard,
            idx_had,
            idx_scales,
            idx_debug,
            idx_cache,
            q_heads,
            original_table,
            compressed_table,
            indexer_table,
            idx_wq_b,
            idx_wq_b_scale,
            idx_query_acc,
            idx_query_rope,
            idx_query_rope_heads,
            idx_query_had,
            idx_query_i8,
            idx_query_scale,
            idx_query_scale_heads,
            weights_proj,
            weights_partial,
            head_weights,
            group_offsets,
            group_plan,
            coefficients,
            idx_key0,
            idx_key64,
            indexer_transfer,
            score_arena,
            pair_arena,
            topk_scores,
            topk_indices,
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
            wo_a,
            o_r,
            o_quant,
            o_scales,
            wo_b,
            o_partials,
            wo_b_scale,
            output,
            signal,
            signal_init_matrix,
            hc_squares,
            hc_packed_input,
            hc_packed_weight,
            hc_norm_rstd,
            attention_copy64,
            QUERY_GROUP,
            QK_PANEL,
            SCORE_TILE,
            PREFETCH,
            TO_L0,
            BALANCED,
            ROOTS,
            OA_COLS,
            OA_ROWS,
            OB_ROWS,
            OUTPUT_GROUP_PIPELINE,
            HC_ROW_DEPENDENCIES,
            ATTENTION_QUERY_METADATA,
            HC_CONSUMER_REDUCE,
            QA_M16_PARALLEL,
            COOPERATIVE_PREFIX,
            EARLY_ATTENTION_GEOMETRY,
            LOCAL_ATTENTION_STATS,
            SCALE_PAGE_COALESCING,
            LEAF_SCALE_LAYOUT,
            PLAN_PHYSICAL_ROWS,
            SPLIT_SCORE_ROWS,
            OUTPUT_WORK_BALANCE,
            SHORT_HALF_SCORE_UB,
            SHORT_TOPK_PREFIX,
            QB_WIDE_N,
            PREFIX_BRANCH_GROUPS,
            JOINT_BF16_PREFIX,
            PREFIX_PROJECTION_ROWS,
            SCORE_GROUP_ORDER=SCORE_GROUP_ORDER,
            PLAN_TABLE_COLUMNS=PLAN_TABLE_COLUMNS,
            FUSED_INDEXER_COEFF=FUSED_INDEXER_COEFF,
        )
    return (cache_rows, main_state, main_cache, idx_state, idx_cache, topk_scores, topk_indices, output)
