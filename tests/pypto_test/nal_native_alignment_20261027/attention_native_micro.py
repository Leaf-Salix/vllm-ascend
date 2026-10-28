# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405 -- Shared exact descriptor types for this diagnostic root.
"""Native512单SPMD独立候选；只隔离attention，不代表整层/性能验收。"""

from attention_reference import *
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.attention_native_stream_stages import *
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.attention_native_stream_stages import (
    native_attention_cube,
    native_attention_vector,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.attention_plan_stages import (
    PLAN_CAP_DYN,
    attention_inverse_rope_sign_stage,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.request_geometry import QUERY_BOUNDS_DYN


@pl.jit.incore
def native_cube_member(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
):
    out_scores, out_pv = native_attention_cube(
        q_flat,
        valid_block_mask,
        kv_transfer,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        pl.tile.get_block_idx(),
        CORE_GROUPS,
        EVENT_BASE,
    )
    return (out_scores, out_pv)


@pl.jit.incore
def native_vector_member(
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]],
    score_transfer: pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.FP32],
    probability_transfer: pl.InOut[pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.BF16]],
    pv_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    softmax_copy64: pl.InOut[pl.Tensor[[SOFTMAX_ROWS, 64], pl.FP32]],
    pre_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
    OBSERVE: pl.constexpr = False,
):
    pl.func_attr({"split_aiv": True})
    out_kv, out_probability, out_mi, out_li, out_packed = native_attention_vector(
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
        softmax_copy64,
        pre_heads,
        pl.tile.get_block_idx(),
        pl.tile.get_subblock_idx(),
        CORE_GROUPS,
        EVENT_BASE,
        OBSERVE,
    )
    return (out_kv, out_probability, out_mi, out_li, out_packed)


@pl.jit.inline(auto_scope=False)
def _native_core_group(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    cmp_sparse_indices: pl.Tensor[[PLAN_CAP_DYN, 512], pl.INT32],
    sparse_bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32],
    kv_transfer: pl.InOut[pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16]],
    score_transfer: pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.FP32],
    probability_transfer: pl.InOut[pl.Tensor[[NATIVE_HEAD_ROWS, ATTN_K_TILE], pl.BF16]],
    pv_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32],
    mi_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    li_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32]],
    o_packed_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    softmax_copy64: pl.InOut[pl.Tensor[[SOFTMAX_ROWS, 64], pl.FP32]],
    pre_heads: pl.InOut[pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16]],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
    OBSERVE: pl.constexpr = False,
):
    with pl.cluster(name_hint="native_attention_single_spmd"):
        score_after, pv_after = native_cube_member(
            q_flat,
            valid_block_mask,
            kv_transfer,
            score_transfer,
            probability_transfer,
            pv_transfer,
            ffts_workspace,
            CORE_GROUPS,
            EVENT_BASE,
        )
        kv_after, prob_after, mi_after, li_after, packed_after = native_vector_member(
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
            score_after,
            probability_transfer,
            pv_after,
            mi_transfer,
            li_transfer,
            o_packed_heads,
            ffts_workspace,
            softmax_copy64,
            pre_heads,
            CORE_GROUPS,
            EVENT_BASE,
            OBSERVE,
        )
    return (pre_heads, packed_after)


@pl.jit
def native_attention_arm(
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B_DYN, ORIGINAL_TABLE_COLUMNS_DYN], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B_DYN, COMPRESSED_TABLE_COLUMNS_DYN], pl.INT32],
    idx_topk: pl.Tensor[[T_DYN, IDX_TOPK], pl.INT32],
    position_ids: pl.Tensor[[T_DYN, 1], pl.INT64],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    o_packed_heads: pl.InOut[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16]],
    mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    bias: pl.Tensor[[PLAN_CAP_DYN, 640], pl.FP32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    pre_heads: pl.InOut[pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16]],
) -> tuple[pl.Tensor, pl.Tensor]:
    tokens = pl.tensor.dim(q, 0)
    q_flat = pl.reshape(q, [tokens * H, HEAD_DIM])
    ori_pages = pl.tensor.dim(ori_kv, 0)
    ori_flat = pl.reshape(ori_kv, [ori_pages * BLOCK_SIZE, HEAD_DIM])
    cmp_pages = pl.tensor.dim(cmp_kv, 0)
    cmp_flat = pl.reshape(cmp_kv, [cmp_pages * BLOCK_SIZE, HEAD_DIM])
    flat_positions = pl.reshape(position_ids, [tokens])
    sink_column = pl.reshape(attn_sink, [64, 1])
    signed_sin = pl.create_tensor([tokens, 64], dtype=pl.FP32)
    kv = pl.create_tensor([24 * 512, 512], dtype=pl.BF16)
    scores = pl.create_tensor([24 * 64, 512], dtype=pl.FP32)
    probability = pl.create_tensor([24 * 64, 512], dtype=pl.BF16)
    pv = pl.create_tensor([24 * 3 * 64, 512], dtype=pl.FP32)
    mi = pl.create_tensor([24 * 3 * 64, 1], dtype=pl.FP32)
    li = pl.create_tensor([24 * 3 * 64, 1], dtype=pl.FP32)
    copy64 = pl.create_tensor([24 * 64, 64], dtype=pl.FP32)
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(48, name_hint="native_micro_sin"):
        signed_sin = attention_inverse_rope_sign_stage(freqs_sin, signed_sin, pl.tile.get_block_idx(), 48)
    with pl.spmd(24, sync_start=True):
        _pre_after, _packed_after = _native_core_group(
            q_flat,
            ori_flat,
            cmp_flat,
            ori_block_table,
            cmp_block_table,
            bounds,
            seq_lens,
            flat_positions,
            idx_topk,
            bias,
            mask,
            sink_column,
            freqs_cos,
            signed_sin,
            kv,
            scores,
            probability,
            pv,
            mi,
            li,
            o_packed_heads,
            ffts,
            copy64,
            pre_heads,
            24,
            2,
            True,
        )
    return (pre_heads, o_packed_heads)
