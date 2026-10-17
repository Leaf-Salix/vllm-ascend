# SPDX-License-Identifier: Apache-2.0
"""动态Attention成对MIX CPU探针，仅编译，不执行NPU。

输入按正式helper的真实动态描述符绑定：Q_ROWS=64*T，所有容量独立dynamic；
Q/L/positions/两页表/页容量及plan为同一不可变批次。plan由已验证准备阶段生成，
本探针把它作为已完成输入，不复制HC、Indexer或Attention计划数学。

阶段职责：同一24组MIX并发24AIC+48AIV。AIC执行QK(tick)、PV(tick-2)，
每组两个AIV执行采KV(tick)、softmax(tick-1)、归并/packed发布(tick-3)。
每组3槽跨query连续轮转，末尾排空；事件0..3分别KV/SCORE/PROB/PV，
不能先完成某角色再调用另一个，不能用阶段barrier替代逐块FFTS握手。
外层没有额外业务依赖，因此不加入phase信号。实际完整CSA须先等待Q/plan/cache。

cluster返回原GM别名，避免条件返回SSA使输出写回被消除；后端另外检查
环槽/packed真实TSTORE与动态ABI。CPU编译/事件模型不证明A3可见性或性能。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.attention_stream_stages import (
    ATTN_K_TILE,
    CMP_TOPK,
    COMPRESSED_COLUMNS_DYN,
    COMPRESSED_ROWS_DYN,
    FFTS_WORDS,
    HEAD_DIM,
    HEAD_RING_ROWS_DYN,
    KV_RING_ROWS_DYN,
    O_GROUP_IN,
    ORIGINAL_COLUMNS_DYN,
    ORIGINAL_ROWS_DYN,
    PACK_ROWS_DYN,
    PLAN_CAP_DYN,
    Q_ROWS_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    ROPE_CAP_DYN,
    ROPE_DIM,
    T_DYN,
    H,
    attention_qk_pv_cube_stage,
    attention_qk_pv_vector_lane_stage,
)


@pl.jit.incore
def cube(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
):
    score_transfer_gm = score_transfer
    pv_transfer_gm = pv_transfer
    worker = pl.tile.get_block_idx()
    score_transfer, pv_transfer = attention_qk_pv_cube_stage(
        q_flat,
        valid_block_mask,
        kv_transfer,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        worker,
        24,
        0,
    )
    return (
        score_transfer_gm,
        pv_transfer_gm,
    )


@pl.jit.incore
def vector(
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN, CMP_TOPK], pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]],
    score_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32],
    probability_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16]],
    pv_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
):
    pl.func_attr({"split_aiv": True})
    kv_transfer_gm = kv_transfer
    probability_transfer_gm = probability_transfer
    mi_transfer_gm = mi_transfer
    li_transfer_gm = li_transfer
    o_packed_heads_gm = o_packed_heads
    worker = pl.tile.get_block_idx()
    lane = pl.tile.get_subblock_idx()
    kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads = attention_qk_pv_vector_lane_stage(
        ori_kv_flat,
        cmp_kv_flat,
        ori_block_table,
        cmp_block_table,
        bounds,
        lengths,
        position_ids,
        cmp_sparse_indices,
        sparse_bias,
        valid_block_mask,
        attn_sink_col,
        freqs_cos,
        rope_sin_signed,
        kv_transfer,
        score_transfer,
        probability_transfer,
        pv_transfer,
        mi_transfer,
        li_transfer,
        o_packed_heads,
        ffts_workspace,
        worker,
        lane,
        24,
        0,
    )
    return (
        kv_transfer_gm,
        probability_transfer_gm,
        mi_transfer_gm,
        li_transfer_gm,
        o_packed_heads_gm,
    )


@pl.jit.inline(auto_scope=False)
def group(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16]],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN, CMP_TOPK], pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
):
    with pl.cluster():
        score_transfer, pv_transfer = cube(
            q_flat, valid_block_mask, kv_transfer, score_transfer, probability_transfer, pv_transfer, ffts_workspace
        )
        kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads = vector(
            ori_kv_flat,
            cmp_kv_flat,
            ori_block_table,
            cmp_block_table,
            bounds,
            lengths,
            position_ids,
            cmp_sparse_indices,
            sparse_bias,
            valid_block_mask,
            attn_sink_col,
            freqs_cos,
            rope_sin_signed,
            kv_transfer,
            score_transfer,
            probability_transfer,
            pv_transfer,
            mi_transfer,
            li_transfer,
            o_packed_heads,
            ffts_workspace,
        )
    return (
        score_transfer,
        pv_transfer,
        kv_transfer,
        probability_transfer,
        mi_transfer,
        li_transfer,
        o_packed_heads,
    )


@pl.jit
def entry(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16]],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN, CMP_TOPK], pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
):
    with pl.spmd(24, sync_start=True):
        score_transfer, pv_transfer, kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads = (
            group(
                q_flat,
                valid_block_mask,
                kv_transfer,
                score_transfer,
                probability_transfer,
                pv_transfer,
                ffts_workspace,
                ori_kv_flat,
                cmp_kv_flat,
                ori_block_table,
                cmp_block_table,
                bounds,
                lengths,
                position_ids,
                cmp_sparse_indices,
                sparse_bias,
                attn_sink_col,
                freqs_cos,
                rope_sin_signed,
                mi_transfer,
                li_transfer,
                o_packed_heads,
            )
        )
    return (
        score_transfer,
        pv_transfer,
        kv_transfer,
        probability_transfer,
        mi_transfer,
        li_transfer,
        o_packed_heads,
    )
