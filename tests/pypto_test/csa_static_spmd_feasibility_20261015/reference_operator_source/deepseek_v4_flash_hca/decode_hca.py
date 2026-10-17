# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TP1 HCA 整层入口；公共投影和 mHC 复用已接入的 CSA 实现。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT
from .decode_compressor_ratio128 import (
    B_DYN,
    BOUNDS_DYN,
    CMP_PAGES_DYN,
    COMPACT_ROWS_DYN,
    STATE_COLUMNS_DYN,
    STATE_PAGE_ELEMENTS_DYN,
    STATE_PAGES_DYN,
    T_DYN,
    compressor_ratio128,
)
from .decode_sparse_attn_hca import (
    CMP_TABLE_BLOCKS_DYN,
    ORI_BLOCK_NUM_DYN,
    ORI_TABLE_COLUMNS_DYN,
    T_PAD,
    sparse_attn_hca_tp1,
)
from .hc_pre_fused import NORM_EPS, hc_pre_norm, hc_pre_norm_hbg
from .o_proj_hc_post import o_proj_hc_post
from .qkv_proj_rope import qkv_proj_rope
from .weight_warm import SINK_BF16, WARM_WORKERS, warm_kv_weights

D = 4096
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
Q_LORA = 1024
HC_MULT = 4
HC_DIM = HC_MULT * D
MIX_HC = 24
O_GROUPS = 8
O_GROUP_IN = 4096
O_LORA = 1024
WIDEN_ROWS = 8
WIDEN_ACTIVE_ROWS = 4
WIDEN_LOAD_COLS = 512
WIDEN_PIPE_STAGE = 4
RMS_COLS = 512
WIDEN_WORKERS = 48
CACHE_WORKERS = 8


def _make_decode_hca_entry(*, host_scalars):
    hc_pre_impl = hc_pre_norm_hbg if host_scalars else hc_pre_norm

    def _decode_hca_tp1_layer(
        x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
        hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
        attn_norm_w: pl.Tensor[[D], pl.BF16],
        wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
        wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
        wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
        gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
        gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
        freqs_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
        freqs_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
        cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
        cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
        cmp_ape: pl.Tensor[[128, HEAD_DIM], pl.FP32],
        cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
        state: pl.InOut[pl.Tensor[[STATE_PAGES_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32]],
        state_table: pl.Tensor[[B_DYN, STATE_COLUMNS_DYN], pl.INT32],
        state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
        positions: pl.Tensor[[T_DYN], pl.INT64],
        query_bounds: pl.Tensor[[BOUNDS_DYN], pl.INT32],
        seq_lens: pl.Tensor[[B_DYN], pl.INT32],
        cmp_cos: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
        cmp_sin: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
        cmp_slots: pl.Tensor[[COMPACT_ROWS_DYN, 2], pl.INT32],
        cmp_cache: pl.InOut[pl.Tensor[[CMP_PAGES_DYN, 32, 1, HEAD_DIM], pl.BF16]],
        cmp_table: pl.Tensor[[B_DYN, CMP_TABLE_BLOCKS_DYN], pl.INT32],
        ori_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, 32, 1, HEAD_DIM], pl.BF16]],
        ori_table: pl.Tensor[[B_DYN, ORI_TABLE_COLUMNS_DYN], pl.INT32],
        ori_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
        attn_sink: pl.Tensor[[H], pl.FP32],
        wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
        wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
        wo_b_scale: pl.Tensor[[D], pl.FP32],
        x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16]],
        host_hc_scale0: pl.Scalar[pl.FP32] = 0.0,
        host_hc_scale1: pl.Scalar[pl.FP32] = 0.0,
        host_hc_scale2: pl.Scalar[pl.FP32] = 0.0,
    ):
        x_hc.bind_dynamic(0, T_DYN)
        x_out.bind_dynamic(0, T_DYN)
        freqs_cos.bind_dynamic(0, T_DYN)
        freqs_sin.bind_dynamic(0, T_DYN)
        state.bind_dynamic(0, STATE_PAGES_DYN)
        state.bind_dynamic(1, STATE_PAGE_ELEMENTS_DYN)
        state_table.bind_dynamic(0, B_DYN)
        state_table.bind_dynamic(1, STATE_COLUMNS_DYN)
        state_slots.bind_dynamic(0, T_DYN)
        positions.bind_dynamic(0, T_DYN)
        query_bounds.bind_dynamic(0, BOUNDS_DYN)
        seq_lens.bind_dynamic(0, B_DYN)
        cmp_cos.bind_dynamic(0, COMPACT_ROWS_DYN)
        cmp_sin.bind_dynamic(0, COMPACT_ROWS_DYN)
        cmp_slots.bind_dynamic(0, COMPACT_ROWS_DYN)
        cmp_cache.bind_dynamic(0, CMP_PAGES_DYN)
        cmp_table.bind_dynamic(0, B_DYN)
        cmp_table.bind_dynamic(1, CMP_TABLE_BLOCKS_DYN)
        ori_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
        ori_table.bind_dynamic(0, B_DYN)
        ori_table.bind_dynamic(1, ORI_TABLE_COLUMNS_DYN)
        ori_slots.bind_dynamic(0, T_DYN)

        tokens = pl.tensor.dim(x_hc, 0)
        post = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
        comb = pl.create_tensor([tokens, HC_MULT * HC_MULT], dtype=pl.FP32)
        normalized = pl.create_tensor([tokens, D], dtype=pl.BF16)
        x32 = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
        x_flat = pl.reshape(x_hc, [tokens, HC_DIM])
        x32_flat = pl.reshape(x32, [tokens, HC_DIM])
        widen_blocks = (tokens + WIDEN_ACTIVE_ROWS - 1) // WIDEN_ACTIVE_ROWS
        hc_padded_rows = ((tokens + 16 - 1) // 16) * 16
        inv_rms = pl.create_tensor([hc_padded_rows, 1], dtype=pl.FP32)
        # 参考 CSA d1f170ff：加宽时按原 RMS 的 512 列次序顺手求平方和，删除 RMS 对 FP32
        # 中间缓冲的再次读取；归约次序、高精度 rsqrt 不变，结果与独立 RMS 任务逐 bit 相同。
        with pl.spmd(pl.min(widen_blocks, WIDEN_WORKERS), name_hint="hca_hc_widen_rms", allow_early_resolve=True) as widen_tid:
            for block in pl.range(pl.tile.get_block_idx(), widen_blocks, pl.min(widen_blocks, WIDEN_WORKERS)):
                row = block * WIDEN_ACTIVE_ROWS
                count = pl.min(WIDEN_ACTIVE_ROWS, tokens - row)
                sq_sum = pl.tile.full([1, WIDEN_ROWS], dtype=pl.FP32, value=0.0)
                for col_block in pl.pipeline(HC_DIM // WIDEN_LOAD_COLS, stage=WIDEN_PIPE_STAGE):
                    col = col_block * WIDEN_LOAD_COLS
                    source = pl.load(x_flat, [row, col], [WIDEN_ROWS, WIDEN_LOAD_COLS],
                                     valid_shape=[count, WIDEN_LOAD_COLS])
                    value = pl.cast(source, pl.FP32)
                    # 只写本worker有效行，不能让两个4行worker的8行物理盒互相覆盖。
                    valid_value = pl.set_validshape(value, count, WIDEN_LOAD_COLS)
                    pl.store(valid_value, [row, col], x32_flat)
                    clean = pl.fillpad(valid_value, pad_value=pl.PadValue.zero)
                    squared = pl.mul(clean, clean)
                    sum_tmp = pl.create_tile([WIDEN_ROWS, RMS_COLS], dtype=pl.FP32)
                    chunk_sum = pl.row_sum(squared, sum_tmp)
                    sq_sum = pl.add(sq_sum, pl.reshape(chunk_sum, [1, WIDEN_ROWS]))
                mean = pl.add(pl.mul(sq_sum, 1.0 / HC_DIM), NORM_EPS)
                inverse_tmp = pl.create_tile([1, WIDEN_ROWS], dtype=pl.FP32)
                inverse = pl.reshape(pl.tile.rsqrt(mean, inverse_tmp), [WIDEN_ROWS, 1])
                pl.store(pl.set_validshape(inverse, count, 1), [row, 0], inv_rms)
        # 冷 L2 下投影读权重受单核带宽限制；mHC pre 期间 Vector 核大多空闲，先预热 KV/compressor 权重。
        warm_sink = pl.create_tensor([WARM_WORKERS, SINK_BF16], dtype=pl.BF16)
        warm_kv_weights(wkv, cmp_wkv, cmp_wgate, warm_sink, widen_tid)
        with pl.scope():
            hc_pre_impl(
                x32, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, post, comb, normalized, False, inv_rms, x_hc,
                host_hc_scale0, host_hc_scale1, host_hc_scale2,
            )

        # 直接二维分配保留manual_dep；外层reshape视图会默认恢复自动依赖。
        # 四组只写各自的head，Attention通过q_ready直接等待全部生产者。
        q = pl.create_tensor([tokens, H * HEAD_DIM], dtype=pl.BF16, manual_dep=True)
        q_ready = pl.array.create(4, pl.TASK_ID)
        for group in pl.unroll(4):
            q_ready[group] = pl.system.task_invalid()
        kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        qr = pl.create_tensor([tokens, Q_LORA], dtype=pl.INT8)
        qr_scale = pl.create_tensor([tokens, 1], dtype=pl.FP32)
        with pl.scope():
            ready = pl.system.task_dummy(deps=[])
            q, qa_tid = qkv_proj_rope(
                normalized, wq_a, wq_b, wq_b_scale, wkv, freqs_cos, freqs_sin,
                gamma_cq, gamma_ckv, q, kv, qr, qr_scale, q_ready, ready,
            )
            cache_rows = pl.tensor.dim(ori_cache, 0) * 32
            cache_flat = pl.reshape(ori_cache, [cache_rows, HEAD_DIM])
            with pl.spmd(CACHE_WORKERS, name_hint="hca_raw_cache_write", allow_early_resolve=True) as raw_tid:
                for token in pl.range(pl.tile.get_block_idx(), tokens, CACHE_WORKERS):
                    page = pl.read(ori_slots, [token, 0])
                    offset = pl.read(ori_slots, [token, 1])
                    if page >= 0 and offset >= 0 and pl.read(seq_lens, [token // 6]) > 0:
                        cache_row = pl.cast(page, pl.INDEX) * 32 + offset
                        cache_flat[cache_row:cache_row + 1, :] = kv[token:token + 1, :]
            compressed_tid = compressor_ratio128(
                normalized, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, state, state_table,
                state_slots, positions, query_bounds, seq_lens, cmp_cos, cmp_sin, cmp_slots, cmp_cache, qa_tid,
            )
            packed = pl.create_tensor([O_GROUPS * T_PAD, O_GROUP_IN], dtype=pl.BF16)
            packed, heads_tid = sparse_attn_hca_tp1(
                q, ori_cache, ori_table, cmp_cache, cmp_table, positions, seq_lens, attn_sink,
                freqs_cos, freqs_sin, packed, raw_tid, compressed_tid, q_ready,
            )
            with pl.scope():
                x_out = o_proj_hc_post(packed, wo_a, wo_b, wo_b_scale, x_hc, post, comb, x_out, heads_tid)
        return x_out

    if host_scalars:
        # HBG 的三个 Host FP32 权重标量必须显式传入，不允许默认为零。
        _decode_hca_tp1_layer.__defaults__ = None
    else:
        # 保留原有 36 Tensor ABI，constexpr 默认值不进入 Torch schema。
        for name in ("host_hc_scale0", "host_hc_scale1", "host_hc_scale2"):
            _decode_hca_tp1_layer.__annotations__[name] = pl.constexpr
    return _decode_hca_tp1_layer


_decode_hca_tp1_layer = _make_decode_hca_entry(host_scalars=False)
_decode_hca_tp1_layer_hbg = _make_decode_hca_entry(host_scalars=True)
decode_hca_tp1_layer = pl.jit.inline(auto_scope=False)(_decode_hca_tp1_layer)
decode_hca_tp1_layer_test = pl.jit(auto_scope=False)(_decode_hca_tp1_layer)
decode_hca_tp1_layer_hbg = pl.jit(auto_scope=False)(_decode_hca_tp1_layer_hbg)
