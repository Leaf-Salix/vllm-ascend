# SPDX-License-Identifier: Apache-2.0
"""完整PTO TND根：固定HC权重加载期预整理，后缀保留冻结精度数学。"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_full_precision.hc_post import hc_post_entry
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_full_precision.heads import publish_native_heads
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_full_precision.o_proj import o_proj_entry
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.decode_csa import (
    B_DYN,
    BF16_WEIGHT_LAYOUT,
    BLOCK_SIZE,
    CMP_BLOCK_NUM_DYN,
    COMPRESS_RATIO,
    COMPRESSED_ROWS_DYN,
    COMPRESSED_TABLE_COLUMNS_DYN,
    CSA_ROPE_SIGN_T_TILE,
    CSA_ROPE_WORKERS,
    CSA_WB_TOKEN_TILE,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    IDX_NATIVE_CACHE_BLOCK_NUM_DYN,
    IDX_TOPK,
    INDEXER_PAGE_BYTES_DYN,
    INDEXER_ROWS_DYN,
    INDEXER_TABLE_COLUMNS_DYN,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM_DYN,
    INNER_STATE_PAGE_ELEMENTS_DYN,
    INNER_STATE_TABLE_COLUMNS_DYN,
    MAIN_OUT_DIM,
    MAIN_STATE_BLOCK_NUM_DYN,
    MIX_HC,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA,
    ORI_BLOCK_NUM_DYN,
    ORIGINAL_TABLE_COLUMNS_DYN,
    Q_LORA,
    QUANT_WEIGHT_LAYOUT,
    QUERY_BOUNDS_DYN,
    ROPE_HEAD_DIM,
    STATE_PAGE_ELEMENTS_DYN,
    STATE_TABLE_COLUMNS_DYN,
    T_DYN,
    T_PAD,
    TP1_CSA_WB_WORKERS,
    WO_A_WEIGHT_LAYOUT,
    D,
    H,
    _read_device_max_seq_len,
    build_compact_row_offsets,
    build_token_request,
    compressor_ratio4,
    indexer_compressor,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision.decode_sparse_attn_csa import sparse_attn_csa_tp1
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_indexer_stages import VEC_ROWS
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_stages import P16, PARTIAL_ROWS
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.norm_wkv_stages import ACC_ROWS
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_tnd_precision_spmd.rendezvous import GROUPS

from ..deepseek_v4_flash_csa_tnd_precision_spmd.indexer import indexer
from .prefix import prepacked_prefix_indexer_group
from .qkv_precision126 import qkv_proj_rope


def _full_csa_tnd_prepacked(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_packed_weight: pl.Tensor[[32768, 32], pl.BF16],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_packet: pl.Tensor[[6272], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_freqs_cos: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[COMPRESSED_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    inner_freqs_cos: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    inner_freqs_sin: pl.Tensor[[INDEXER_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    compress_state: pl.InOut[pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    state_block_table: pl.Tensor[[B_DYN, STATE_TABLE_COLUMNS_DYN], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[IDX_N_HEADS, D], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_compress_state: pl.InOut[pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
    inner_state_block_table: pl.Tensor[[B_DYN, INNER_STATE_TABLE_COLUMNS_DYN], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    idx_native_kv_cache: pl.InOut[pl.Tensor[[IDX_NATIVE_CACHE_BLOCK_NUM_DYN, INDEXER_PAGE_BYTES_DYN], pl.INT8]],
    idx_block_table: pl.Tensor[[B_DYN, INDEXER_TABLE_COLUMNS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    ori_block_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[COMPRESSED_ROWS_DYN, 2], pl.INT32],
    idx_slot_mapping: pl.Tensor[[INDEXER_ROWS_DYN, 2], pl.INT32],
    state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    inner_state_slot_mapping: pl.Tensor[[T_DYN, 2], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    cmp_query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    cmp_seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    idx_query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    query_start_loc: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    idx_topk_scores: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.FP32]],
    idx_topk: pl.Out[pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32]],
    order_packet: pl.Tensor[[2560], pl.INT32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
    spmd_square: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    spmd_square_read: pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32],
    spmd_linear: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32]],
    spmd_linear_read: pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32],
    spmd_accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    spmd_accum_read: pl.Tensor[[ACC_ROWS, 512], pl.FP32],
    spmd_idx_accum: pl.InOut[pl.Tensor[[P16, 64], pl.FP32]],
    spmd_idx_accum_read: pl.Tensor[[P16, 64], pl.FP32],
    spmd_ffts: pl.Tensor[[256], pl.INT64],
    vec_ws: pl.InOut[pl.Tensor[[VEC_ROWS, 64], pl.FP32]],
):
    """一个runtime入口，七阶段以真实Tensor读写依赖连接。"""
    x_hc.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    cmp_freqs_sin.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    inner_freqs_cos.bind_dynamic(0, INDEXER_ROWS_DYN)
    inner_freqs_sin.bind_dynamic(0, INDEXER_ROWS_DYN)
    compress_state.bind_dynamic(0, MAIN_STATE_BLOCK_NUM_DYN)
    compress_state.bind_dynamic(1, STATE_PAGE_ELEMENTS_DYN)
    state_block_table.bind_dynamic(0, B_DYN)
    state_block_table.bind_dynamic(1, STATE_TABLE_COLUMNS_DYN)
    inner_compress_state.bind_dynamic(0, INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state.bind_dynamic(1, INNER_STATE_PAGE_ELEMENTS_DYN)
    inner_state_block_table.bind_dynamic(0, B_DYN)
    inner_state_block_table.bind_dynamic(1, INNER_STATE_TABLE_COLUMNS_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    idx_native_kv_cache.bind_dynamic(0, IDX_NATIVE_CACHE_BLOCK_NUM_DYN)
    idx_native_kv_cache.bind_dynamic(1, INDEXER_PAGE_BYTES_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    ori_block_table.bind_dynamic(0, B_DYN)
    ori_block_table.bind_dynamic(1, ORIGINAL_TABLE_COLUMNS_DYN)
    cmp_slot_mapping.bind_dynamic(0, COMPRESSED_ROWS_DYN)
    idx_slot_mapping.bind_dynamic(0, INDEXER_ROWS_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, B_DYN)
    cmp_query_start_loc.bind_dynamic(0, QUERY_BOUNDS_DYN)
    cmp_seq_lens.bind_dynamic(0, B_DYN)
    idx_query_start_loc.bind_dynamic(0, QUERY_BOUNDS_DYN)
    query_start_loc.bind_dynamic(0, QUERY_BOUNDS_DYN)
    cmp_block_table.bind_dynamic(0, B_DYN)
    cmp_block_table.bind_dynamic(1, COMPRESSED_TABLE_COLUMNS_DYN)
    idx_block_table.bind_dynamic(0, B_DYN)
    idx_block_table.bind_dynamic(1, INDEXER_TABLE_COLUMNS_DYN)
    idx_topk.bind_dynamic(0, T_DYN)
    idx_topk_scores.bind_dynamic(0, T_DYN)
    vec_ws.bind_dynamic(0, VEC_ROWS)
    spmd_square.bind_dynamic(0, PARTIAL_ROWS)
    spmd_square_read.bind_dynamic(0, PARTIAL_ROWS)
    spmd_linear.bind_dynamic(0, PARTIAL_ROWS)
    spmd_linear_read.bind_dynamic(0, PARTIAL_ROWS)
    spmd_accum.bind_dynamic(0, ACC_ROWS)
    spmd_accum_read.bind_dynamic(0, ACC_ROWS)
    spmd_idx_accum.bind_dynamic(0, P16)
    spmd_idx_accum_read.bind_dynamic(0, P16)
    # Immutable load-time packets only coalesce ABI descriptors, not arithmetic.
    attn_norm_w = pl.slice(norm_packet, [4096], [0])
    gamma_cq = pl.slice(norm_packet, [1024], [4096])
    gamma_ckv = pl.slice(norm_packet, [512], [5120])
    cmp_norm_w = pl.slice(norm_packet, [512], [5632])
    inner_norm_w = pl.slice(norm_packet, [128], [6144])
    wkv_order_flat = pl.slice(order_packet, [512], [0])
    wkv_order = pl.reshape(wkv_order_flat, [32, 16])
    indexer_order_flat = pl.slice(order_packet, [1024], [512])
    indexer_order = pl.reshape(indexer_order_flat, [16, 4, 16])
    oproj_start_flat = pl.slice(order_packet, [1024], [1536])
    oproj_start = pl.reshape(oproj_start_flat, [16, 8, 8])
    tokens = pl.tensor.dim(x_hc, 0)
    mixed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    post = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([tokens, HC_MULT * HC_MULT], dtype=pl.FP32)
    normed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    projected_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    indexer_raw = pl.create_tensor([tokens, IDX_N_HEADS], dtype=pl.BF16)
    indexer_scaled = pl.create_tensor([tokens, IDX_N_HEADS], dtype=pl.BF16)
    indexer_weights = pl.create_tensor([tokens, IDX_N_HEADS], dtype=pl.FP16)
    heads = pl.create_tensor([tokens, H * HEAD_DIM], dtype=pl.BF16)
    projected_o = pl.create_tensor([tokens, O_GROUPS * O_LORA], dtype=pl.BF16)
    quantized_o = pl.create_tensor([tokens, O_GROUPS * O_LORA], dtype=pl.INT8)
    scale_o = pl.create_tensor([tokens, 1], dtype=pl.FP32)
    attention_out = pl.create_tensor([tokens, D], dtype=pl.BF16)
    # Sibling scopes release stage-private scratch without releasing published
    # root-owned values. Attention keeps its existing nested scratch scopes.
    flat_hc = pl.reshape(x_hc, [tokens, HC_DIM])
    hc_base_2d = pl.reshape(hc_attn_base, [1, MIX_HC])
    with pl.scope():
        p16 = pl.tensor.dim(spmd_idx_accum, 0)
        wide = pl.create_tensor([tokens, HC_DIM], dtype=pl.FP32)
        packed_flat = pl.create_tensor([p16, 32768], dtype=pl.BF16)
        mixed_tail = pl.create_tensor([8, D], dtype=pl.BF16)
        with pl.spmd(GROUPS, sync_start=True):
            _prefix_result = prepacked_prefix_indexer_group(
                hc_attn_scale,
                hc_attn_base,
                attn_norm_w,
                wkv,
                wkv_order,
                wide,
                spmd_square,
                spmd_square_read,
                vec_ws,
                hc_packed_weight,
                packed_flat,
                spmd_linear,
                spmd_linear_read,
                post,
                comb,
                mixed,
                mixed_tail,
                normed,
                spmd_accum,
                spmd_accum_read,
                projected_kv,
                spmd_ffts,
                flat_hc,
                hc_base_2d,
                weights_proj,
                indexer_order,
                spmd_idx_accum,
                spmd_idx_accum_read,
                indexer_raw,
                indexer_scaled,
                indexer_weights,
            )
    max_seq_len = _read_device_max_seq_len(kv_seq_lens, 0)
    # attn_out 现在是 kernel 内部张量，不再需要绑定外部动态维。
    t_dim = pl.tensor.dim(x_hc, 0)
    x_normed_t = normed
    wb_blocks = (t_dim + CSA_WB_TOKEN_TILE - 1) // CSA_WB_TOKEN_TILE

    idx_sin_signed = pl.create_tensor([t_dim, ROPE_HEAD_DIM], dtype=pl.FP32)
    cmp_row_offsets = pl.create_tensor([pl.tensor.dim(cmp_seq_lens, 0)], dtype=pl.INT32)
    idx_row_offsets = pl.create_tensor([pl.tensor.dim(kv_seq_lens, 0)], dtype=pl.INT32)
    # 逐请求 compact 偏移是跨请求前缀和，只能串行；RoPE 符号是逐块独立的，可并行。
    # 原先两件事共用一个 CORE_GROUP 任务，整段被前缀和拖成串行——泳道实测
    # csa_rope_sign count=1、Exec 13.42us 却独占一个串行窗口。拆成两个任务，
    # 符号那段走 SPMD，靠 deps 保证偏移先算好。
    # 元数据也是RMS/cache消费者的实际前置，必须允许其参与预派发资格判定。
    token_request = pl.create_tensor([t_dim], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_row_offsets", allow_early_resolve=True) as offsets_tid:
        cmp_row_offsets = build_compact_row_offsets(cmp_query_start_loc, cmp_seq_lens, cmp_row_offsets)
        idx_row_offsets = build_compact_row_offsets(idx_query_start_loc, kv_seq_lens, idx_row_offsets)
        token_request = build_token_request(query_start_loc, kv_seq_lens, token_request)

    # 向上取整分块并保留 valid_shape：t_dim = batch*6 不保证是 4 的倍数，
    # 上游 csa_rope_interleave 用的 t_dim // 4 会丢掉尾行（batch=1 时 t_dim=6 只覆盖 0~3）。
    rope_sign_blocks = (t_dim + CSA_ROPE_SIGN_T_TILE - 1) // CSA_ROPE_SIGN_T_TILE
    with pl.spmd(
        pl.min(rope_sign_blocks, CSA_ROPE_WORKERS),
        name_hint="csa_rope_sign",
        deps=[offsets_tid],
        allow_early_resolve=True,
    ) as rope_tid:
        for rope_rb in pl.range(pl.tile.get_block_idx(), rope_sign_blocks, pl.min(rope_sign_blocks, CSA_ROPE_WORKERS)):
            rope_t0 = rope_rb * CSA_ROPE_SIGN_T_TILE
            rope_rows = pl.min(CSA_ROPE_SIGN_T_TILE, t_dim - rope_t0)
            # 符号表按块重算：SPMD 下每个 worker 要有自己的一份。
            il_ones = pl.tile.full([CSA_ROPE_SIGN_T_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
            il_lane_ids = pl.cast(pl.tile.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32)
            il_col = pl.col_expand_mul(il_ones, il_lane_ids)
            il_dup_f = pl.cast(pl.cast(pl.mul(il_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
            il_lane = pl.sub(il_col, pl.mul(il_dup_f, 2.0))
            il_sign = pl.sub(pl.mul(il_lane, 2.0), 1.0)
            rope_sin_rows = pl.load(
                freqs_sin,
                [rope_t0, 0],
                [CSA_ROPE_SIGN_T_TILE, ROPE_HEAD_DIM],
                valid_shape=[rope_rows, ROPE_HEAD_DIM],
            )
            rope_sign_rows = pl.set_validshape(il_sign, rope_rows, ROPE_HEAD_DIM)
            pl.store(pl.mul(rope_sin_rows, rope_sign_rows), [rope_t0, 0], idx_sin_signed)

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    position_ids_t1 = pl.reshape(position_ids, [t_dim, 1])
    with pl.scope():
        # Projection-chain dependency marker.
        late_dep = pl.system.task_dummy(deps=[rope_tid])
        q, qa_tid = qkv_proj_rope(
            x_normed_t,
            wq_a,
            wq_b,
            wq_b_scale,
            freqs_cos,
            freqs_sin,
            gamma_cq,
            gamma_ckv,
            q,
            kv,
            qr,
            qr_scale,
            late_dep,
            projected_kv,
        )

        ori_block_num = pl.tensor.dim(kv_cache, 0)
        kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
        with pl.spmd(TP1_CSA_WB_WORKERS, name_hint="csa_cache_writeback", allow_early_resolve=True):
            wb_worker = pl.tile.get_block_idx()
            for wb_blk in pl.range(wb_worker, wb_blocks, TP1_CSA_WB_WORKERS):
                wb_t0 = wb_blk * CSA_WB_TOKEN_TILE
                for write_dt in pl.range(pl.min(CSA_WB_TOKEN_TILE, t_dim - wb_t0)):
                    write_t = wb_t0 + write_dt
                    write_page = pl.read(ori_slot_mapping, [write_t, 0])
                    write_offset = pl.read(ori_slot_mapping, [write_t, 1])
                    if write_page >= 0 and write_offset >= 0:
                        write_row = pl.cast(write_page, pl.INDEX) * BLOCK_SIZE + write_offset
                        kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

        # Keep Q_A ahead of the heavier Attention Compressor projection.
        # The lighter Indexer projection overlaps Q_A; its Hadamard gate stays below.
        compressor_dep = pl.system.task_dummy(deps=[late_dep, qa_tid])
        cmp_out = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.FP32)
        cmp_out, cmp_cache_write_tid, cmp_kv_score_tid = compressor_ratio4(
            x_normed_t,
            cmp_out,
            compress_state,
            state_block_table,
            cmp_wkv,
            cmp_wgate,
            cmp_ape,
            cmp_norm_w,
            cmp_freqs_cos,
            cmp_freqs_sin,
            cmp_row_offsets,
            token_request,
            cmp_kv,
            position_ids,
            cmp_seq_lens,
            query_start_loc,
            cmp_slot_mapping,
            state_slot_mapping,
            compressor_dep,
        )
        idx_kv_unused = pl.create_tensor([t_dim, IDX_HEAD_DIM], dtype=pl.FP32)
        idx_cache_write_tid = indexer_compressor(
            x_normed_t,
            idx_kv_unused,
            inner_compress_state,
            inner_state_block_table,
            inner_wkv,
            inner_wgate,
            inner_ape,
            inner_norm_w,
            inner_freqs_cos,
            inner_freqs_sin,
            idx_row_offsets,
            token_request,
            hadamard_idx,
            idx_native_kv_cache,
            position_ids,
            kv_seq_lens,
            query_start_loc,
            idx_slot_mapping,
            inner_state_slot_mapping,
            late_dep,
            cmp_kv_score_tid,
        )
        # Keep the root descriptor in the parent scope. Indexer writes this
        # same Out allocation; its returned SSA alias belongs to the child.
        attention_topk = idx_topk
        # Bound indexer scratch to its own runtime scope.
        with pl.scope():
            _index_scores, _index_indices = indexer(
                x_normed_t,
                qr,
                qr_scale,
                idx_wq_b,
                idx_wq_b_scale,
                indexer_weights,
                freqs_cos,
                idx_sin_signed,
                hadamard_idx,
                idx_native_kv_cache,
                idx_block_table,
                idx_topk_scores,
                idx_topk,
                position_ids,
                kv_seq_lens,
                token_request,
                query_start_loc,
                late_dep,
                idx_cache_write_tid,
                max_seq_len,
            )

        # sparse_attn_csa folds the compressed-slot masking + valid-block flags in from the
        # raw indexer topk + position.
        o_packed_heads = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
        o_packed_heads, heads_dep = sparse_attn_csa_tp1(
            q,
            kv_cache,
            ori_block_table,
            cmp_kv,
            cmp_block_table,
            attention_topk,
            position_ids_t1,
            kv_seq_lens,
            token_request,
            attn_sink,
            freqs_cos,
            freqs_sin,
            o_packed_heads,
            32,
        )
        with pl.scope():
            _published_heads = publish_native_heads(o_packed_heads, heads, t_dim, heads_dep)
    with pl.scope():
        _oproj_result = o_proj_entry(
            heads, wo_a, wo_b, wo_b_scale, oproj_start, projected_o, quantized_o, scale_o, attention_out
        )
    with pl.scope():
        _hc_post_result = hc_post_entry(attention_out, x_hc, post, comb, x_out)
    return x_out


_decode_csa_tp1_layer = _full_csa_tnd_prepacked
decode_csa_tp1_layer = pl.jit.inline(auto_scope=False)(_full_csa_tnd_prepacked)
decode_csa_tp1_layer_test = pl.jit(auto_scope=False)(_full_csa_tnd_prepacked)
