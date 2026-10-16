# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考 pypto-lib 的 C128 压缩链，直接消费 Native 页、slot 与紧凑 RoPE。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import DECODE_BATCH, DECODE_SEQ, FP32_NEG_INF
from ..deepseek_v4_flash_csa.config import FLASH as M

B_DYN = pl.dynamic("HCA_REQUESTS")
T_DYN = pl.dynamic("HCA_TOKENS")
BOUNDS_DYN = pl.dynamic("HCA_QUERY_BOUNDS")
STATE_PAGES_DYN = pl.dynamic("HCA_STATE_PAGES")
STATE_PAGE_ELEMENTS_DYN = pl.dynamic("HCA_STATE_PAGE_ELEMENTS")
STATE_COLUMNS_DYN = pl.dynamic("HCA_STATE_COLUMNS")
CMP_PAGES_DYN = pl.dynamic("HCA_CMP_PAGES")
COMPACT_ROWS_DYN = pl.dynamic("HCA_COMPACT_ROWS")

D = M.hidden_size
EPS = M.rms_norm_eps
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
NOPE_DIM = HEAD_DIM - ROPE_DIM
RATIO = 128
STATE_BLOCK = 8
STATE_WIDTH = 2 * HEAD_DIM
CMP_BLOCK = 32
MM_ROWS = 64
MM_COLS = 64
MM_K = 512
PAD_TOKENS = (DECODE_BATCH * DECODE_SEQ + MM_ROWS - 1) // MM_ROWS * MM_ROWS
POOL_COLS = 128
RMS_ROWS = 8
WORKERS = 16


@pl.jit.inline(auto_scope=False)
def compressor_ratio128(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    ape: pl.Tensor[[RATIO, HEAD_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    state: pl.Tensor[[STATE_PAGES_DYN, STATE_PAGE_ELEMENTS_DYN], pl.FP32],
    state_table: pl.Tensor[[B_DYN, STATE_COLUMNS_DYN], pl.INT32],
    state_slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    query_bounds: pl.Tensor[[BOUNDS_DYN], pl.INT32],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    cos: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS_DYN, ROPE_DIM], pl.FP32],
    cmp_slots: pl.Tensor[[COMPACT_ROWS_DYN, 2], pl.INT32],
    cmp_cache: pl.Tensor[[CMP_PAGES_DYN, CMP_BLOCK, 1, HEAD_DIM], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """S6 动态 batch；先完成历史读取，再提交 Native state，避免滚动页覆盖。"""
    tokens = pl.tensor.dim(x, 0)
    requests = pl.tensor.dim(seq_lens, 0)
    kv_proj = pl.create_tensor([PAD_TOKENS, HEAD_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([PAD_TOKENS, HEAD_DIM], dtype=pl.FP32)
    pooled = pl.create_tensor([DECODE_BATCH, HEAD_DIM], dtype=pl.FP32)
    offsets = pl.create_tensor([requests], dtype=pl.INT32)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_compact_offsets"):
        prefix = pl.cast(0, pl.INDEX)
        for request in pl.range(requests):
            begin = pl.read(query_bounds, [request])
            end = pl.read(query_bounds, [request + 1])
            length = pl.read(seq_lens, [request])
            start = pl.max(length - (end - begin), 0)
            pl.write(offsets, [request], pl.cast(prefix - start // RATIO - 1, pl.INT32))
            if length > 0:
                prefix = prefix + length // RATIO - start // RATIO

    with pl.spmd(
        ((tokens + MM_ROWS - 1) // MM_ROWS) * (HEAD_DIM // MM_COLS),
        name_hint="hca_kv_score_proj", allow_early_resolve=True, deps=[late_dep],
    ) as projection_tid:
        block = pl.tile.get_block_idx()
        row = block // (HEAD_DIM // MM_COLS) * MM_ROWS
        col = block % (HEAD_DIM // MM_COLS) * MM_COLS
        valid_rows = pl.min(MM_ROWS, tokens - row)
        first_tile = pl.slice(x, [MM_ROWS, MM_K], [row, 0], valid_shape=[valid_rows, MM_K])
        kv_acc = pl.matmul(first_tile, wkv[col:col + MM_COLS, 0:MM_K], out_dtype=pl.FP32, b_trans=True)
        score_acc = pl.matmul(first_tile, wgate[col:col + MM_COLS, 0:MM_K], out_dtype=pl.FP32, b_trans=True)
        for kb in pl.pipeline(1, D // MM_K, stage=2):
            k0 = kb * MM_K
            tile = pl.slice(x, [MM_ROWS, MM_K], [row, k0], valid_shape=[valid_rows, MM_K])
            kv_acc = pl.matmul_acc(kv_acc, tile, wkv[col:col + MM_COLS, k0:k0 + MM_K], b_trans=True)
            score_acc = pl.matmul_acc(score_acc, tile, wgate[col:col + MM_COLS, k0:k0 + MM_K], b_trans=True)
        kv_proj[row:row + MM_ROWS, col:col + MM_COLS] = kv_acc
        score_proj[row:row + MM_ROWS, col:col + MM_COLS] = score_acc

    with pl.spmd(pl.min(requests, WORKERS), name_hint="hca_softmax_pool", allow_early_resolve=True, deps=[projection_tid]) as pool_tid:
        for request in pl.range(pl.tile.get_block_idx(), requests, pl.min(requests, WORKERS)):
            length = pl.read(seq_lens, [request])
            first = pl.read(positions, [request * DECODE_SEQ])
            closing = first + RATIO - 1 - first % RATIO
            if length > 0 and closing < first + DECODE_SEQ:
                # 当前步的行在 UB 中覆盖历史读取；不先写 state，不需要搬运窗口。
                window_start = closing - RATIO + 1
                for h0 in pl.range(0, HEAD_DIM, POOL_COLS):
                    values = pl.full([RATIO, POOL_COLS], dtype=pl.FP32, value=0.0)
                    scores = pl.full([RATIO, POOL_COLS], dtype=pl.FP32, value=FP32_NEG_INF)
                    for history_row in pl.range(RATIO):
                        pos = window_start + history_row
                        if pos < first:
                            page = pl.read(state_table, [request, pos // STATE_BLOCK])
                            if page >= 0:
                                column = pl.cast(pos % STATE_BLOCK, pl.INDEX) * STATE_WIDTH + h0
                                values[history_row:history_row + 1, :] = pl.slice(state, [1, POOL_COLS], [page, column])
                                scores[history_row:history_row + 1, :] = pl.slice(
                                    state, [1, POOL_COLS], [page, column + HEAD_DIM],
                                )
                        else:
                            pool_token = request * DECODE_SEQ + pos - first
                            ape_row = pl.cast(pos % RATIO, pl.INDEX)
                            values[history_row:history_row + 1, :] = pl.slice(kv_proj, [1, POOL_COLS], [pool_token, h0])
                            scores[history_row:history_row + 1, :] = pl.add(
                                pl.slice(score_proj, [1, POOL_COLS], [pool_token, h0]),
                                pl.slice(ape, [1, POOL_COLS], [ape_row, h0]),
                            )
                    maximum = pl.col_max(scores)
                    exponent = pl.col_expand_expdif(scores, maximum)
                    probability = pl.col_expand_mul(exponent, pl.recip(pl.col_sum(exponent)))
                    pooled[request:request + 1, h0:h0 + POOL_COLS] = pl.col_sum(pl.mul(values, probability))

    cache_rows = pl.tensor.dim(cmp_cache, 0) * CMP_BLOCK
    cache_flat = pl.reshape(cmp_cache, [cache_rows, HEAD_DIM])
    gamma = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.spmd(pl.min(requests, WORKERS), name_hint="hca_state_commit", allow_early_resolve=True, deps=[pool_tid]) as state_tid:
        for request in pl.range(pl.tile.get_block_idx(), requests, pl.min(requests, WORKERS)):
            if pl.read(seq_lens, [request]) > 0:
                for step in pl.range(DECODE_SEQ):
                    token = request * DECODE_SEQ + step
                    page = pl.read(state_slots, [token, 0])
                    offset = pl.read(state_slots, [token, 1])
                    if page >= 0 and offset >= 0:
                        column = pl.cast(offset, pl.INDEX) * STATE_WIDTH
                        ape_row = pl.cast(pl.read(positions, [token]) % RATIO, pl.INDEX)
                        pl.store(pl.load(kv_proj, [token, 0], [1, HEAD_DIM]), [page, column], state)
                        score = pl.add(
                            pl.load(score_proj, [token, 0], [1, HEAD_DIM]),
                            pl.load(ape, [ape_row, 0], [1, HEAD_DIM]),
                        )
                        pl.store(score, [page, column + HEAD_DIM], state)

    with pl.spmd(pl.min(requests, WORKERS), name_hint="hca_norm_rope_write", allow_early_resolve=True, deps=[pool_tid]) as cache_tid:
        for request in pl.range(pl.tile.get_block_idx(), requests, pl.min(requests, WORKERS)):
            length = pl.read(seq_lens, [request])
            first = pl.read(positions, [request * DECODE_SEQ])
            closing = first + RATIO - 1 - first % RATIO
            if length > 0 and closing < first + DECODE_SEQ:
                compact = pl.cast(pl.read(offsets, [request]), pl.INDEX) + (closing + 1) // RATIO
                # 先判断上下界再读，允许零行 compact；dummy 的陈旧 position 不可越界。
                if compact >= 0 and compact < pl.tensor.dim(cmp_slots, 0):
                    page = pl.read(cmp_slots, [compact, 0])
                    offset = pl.read(cmp_slots, [compact, 1])
                    if page >= 0 and offset >= 0:
                        # A3 的列主序归约结果按 32 字节寻址；FP32 至少需要 8 个物理行。
                        # 补齐只在 UB 内发生，实际只发布本请求的一行。
                        value = pl.full([RMS_ROWS, HEAD_DIM], dtype=pl.FP32, value=0.0)
                        value[0:1, :] = pl.slice(pooled, [1, HEAD_DIM], [request, 0])
                        variance = pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / HEAD_DIM), EPS)
                        normalized = pl.col_expand_mul(
                            pl.row_expand_mul(value, pl.recip(pl.sqrt(variance))), pl.cast(gamma, pl.FP32),
                        )
                        rope = normalized[0:1, NOPE_DIM:HEAD_DIM]
                        lane = pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), pl.FP32)
                        pair = pl.cast(pl.cast(pl.mul(lane, 0.5), pl.INT32, mode="trunc"), pl.FP32)
                        odd = pl.sub(lane, pl.mul(pair, 2.0))
                        swap = pl.cast(pl.sub(pl.add(lane, 1.0), pl.mul(odd, 2.0)), pl.INT32)
                        signed_sin = pl.mul(pl.slice(sin, [1, ROPE_DIM], [compact, 0]), pl.sub(pl.mul(odd, 2.0), 1.0))
                        rotated = pl.add(
                            pl.mul(rope, pl.slice(cos, [1, ROPE_DIM], [compact, 0])),
                            pl.mul(pl.gather(rope, dim=-1, index=swap), signed_sin),
                        )
                        output = pl.concat(normalized[0:1, 0:NOPE_DIM], rotated)
                        cache_row = pl.cast(page, pl.INDEX) * CMP_BLOCK + offset
                        cache_flat[cache_row:cache_row + 1, :] = pl.cast(output, pl.BF16, mode="rint")

    # 原先用 task_dummy 汇聚 state 与 cache 两个任务。唯一的消费者是 attention 的
    # hca_cmp_work_gather，它只读 cmp_cache、并不读压缩器 state；所以那个 dummy 既多一跳
    # AICPU 调度，又把 hca_state_commit 变成了 attention 的假依赖。这里直接交出写
    # cmp_cache 的任务 id。state 写的是 InOut 张量，整层结束前必然完成，不会被丢掉。
    return cache_tid
