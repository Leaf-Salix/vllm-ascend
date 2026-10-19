# SPDX-License-Identifier: Apache-2.0
"""动态 CSA Attention 的 AIC 直接压缩 KV 采集候选。

整体数据流与阅读顺序
    load_direct_block 按本轮物理行 plan 从 Native 主压缩 cache 直接采集到 L1；
    SWA 仍读取双 AIV 填好的 GM 环。emit_qk/emit_pv 保留原矩阵数学和写回。
    最后读 attention_direct_l1_cube_stage：三份独立 L1 Tile 跨动态 tick 携带，
    QK 领先 PV 两拍；每块只采集一次，QK 用转置视图，PV 用同一槽的列切片。

动态范围与分档
    T 来自 Query Tensor shape，query 按 q%24 归核，全部有效块来自 device mask。
    128×512 是既有数学块，三槽是流水资源，均不是某个 B/S/history 的实值化。
    外层仅在 PLAN_PHYSICAL_ROWS 的共同容量谓词成立时调用；物理索引是本次
    plan 验证后的 INT32，负数/越界由 MGATHER ZERO 填零。容量外必须回退。

核分工、同步与工作区
    压缩 KV 不再经过 AIV UB→GM→AIC L1 的中转；AIV 继续完成 softmax/归并。
    全部有效块仍保留原 READY/SCORE/PROB/PV 事件数和顺序，READY 对直接块
    维持原环槽推进约束。调用者在读主 cache 前必须获取 MAIN_OUTPUT 及 payload，
    不能只等待 plan。SWA 的数据发布仍由原 AIV/MTE3 READY 负责。
    三槽的旧内容保留到对应 PV；不把三个不同存储合成 Tile phi，因为 A3 没有
    Mat→Mat TMOV。消费者按槽留在分支中，三个输出槽本身仍显式 loop carry。
    该候选依赖本地 GM 连续子视图以及 PTO-ISA #352 的 ZERO→ND2NZ 顺序修正。
    原语测试通过不等于整层功能或性能已验证；采用状态由完整设备结果决定。
"""

import pypto.language as pl

from .attention_stream_stages import (
    ATTN_K_TILE, CMP_TOPK, COMPRESSED_ROWS_DYN, FFTS_WORDS, H, HEAD_DIM,
    HEAD_RING_ROWS_DYN, KV_RING_ROWS_DYN, PLAN_CAP_DYN, PV_N_TILE,
    Q_ROWS_DYN, QK_PRE_LAUNCH, QK_TRANSFER_SLOTS, SPARSE_BLOCKS,
    QK_KV_READY_EVENT, QK_SCORE_READY_EVENT, QK_PROB_READY_EVENT, QK_PV_READY_EVENT,
)


@pl.jit.inline(auto_scope=False)
def load_direct_block(
    main_cache: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[PLAN_CAP_DYN, CMP_TOPK], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    query: pl.Scalar[pl.INDEX],
    block: pl.Scalar[pl.INDEX],
    ring_slot: pl.Scalar[pl.INDEX],
) -> pl.Tile[[ATTN_K_TILE, HEAD_DIM], pl.BF16, pl.Mem.Mat]:
    if block == 0:
        # SWA 保留原已发布的连续 GM 中转；AIC 在调用本函数前已消费 READY。
        loaded = pl.load(kv_transfer, [ring_slot * ATTN_K_TILE, 0],
                         [ATTN_K_TILE, HEAD_DIM], target_memory=pl.Mem.Mat)
        result = pl.yield_(loaded)
    else:
        # 对原 plan 创建连续视图，不复制 indices、不扩大 allocation、不读回 Host。
        flat = pl.tensor.view(indices, shape=[pl.tensor.dim(indices, 0) * CMP_TOPK, 1])
        column = pl.tensor.slice(flat, [ATTN_K_TILE, 1],
                                 [query * CMP_TOPK + (block - 1) * ATTN_K_TILE, 0], gm_view=True)
        row = pl.tensor.view(column, shape=[1, ATTN_K_TILE])
        gathered = pl.tile.mgather(main_cache, row, coalesce="row", gather_oob="zero",
                                   target_memory=pl.Mem.Mat)
        result = pl.yield_(gathered)
    return result


@pl.jit.inline(auto_scope=False)
def emit_qk(
    query: pl.Tile[[H, HEAD_DIM], pl.BF16, pl.Mem.Mat],
    key: pl.Tile[[ATTN_K_TILE, HEAD_DIM], pl.BF16, pl.Mem.Mat],
    scores: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    row: pl.Scalar[pl.INDEX],
):
    transposed = pl.tile.transpose_view(key)
    value = pl.matmul(query, transposed, out_dtype=pl.FP32)
    pl.store(value, [row, 0], scores)
    return scores


@pl.jit.inline(auto_scope=False)
def emit_pv(
    probability: pl.Tile[[H, ATTN_K_TILE], pl.BF16, pl.Mem.Left],
    key: pl.Tile[[ATTN_K_TILE, HEAD_DIM], pl.BF16, pl.Mem.Mat],
    outputs: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    row: pl.Scalar[pl.INDEX],
):
    # 沿用原 N128/K128 和双 Acc 交替，先发下块 Cube，再写前块 FIX。
    first_right = pl.tile.extract(key, 0, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.Mem.Right)
    previous = pl.tile.matmul(probability, first_right)
    for part in pl.unroll(1, HEAD_DIM // PV_N_TILE):
        next_right = pl.tile.extract(key, 0, part * PV_N_TILE,
                                     [ATTN_K_TILE, PV_N_TILE], target_memory=pl.Mem.Right)
        current = pl.tile.matmul(probability, next_right)
        pl.store(previous, [row, (part - 1) * PV_N_TILE], outputs)
        previous = current
    pl.store(previous, [row, HEAD_DIM - PV_N_TILE], outputs)
    return outputs


@pl.jit.inline(auto_scope=False)
def attention_direct_l1_cube_stage(
    query: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    main_cache: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    indices: pl.Tensor[[PLAN_CAP_DYN, CMP_TOPK], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    scores: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probabilities: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16],
    products: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    worker: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 2,
):
    tokens = pl.tensor.dim(query, 0) // H
    blocks = ((tokens + CORE_GROUPS - 1 - worker) // CORE_GROUPS) * SPARSE_BLOCKS
    pl.system.set_ffts(ffts_workspace)
    init_a = pl.create_tile([ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    init_b = pl.create_tile([ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    init_c = pl.create_tile([ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    init_q = pl.create_tile([H, HEAD_DIM], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    # 最后两拍只做 PV 排空；首次有效 QK 先写槽，再允许对应 PV 读取。
    for tick, (slot_a, slot_b, slot_c, query_old) in pl.range(
        blocks + QK_PRE_LAUNCH, init_values=(init_a, init_b, init_c, init_q)
    ):
        if tick < blocks:
            block = tick % SPARSE_BLOCKS
            token = worker + (tick // SPARSE_BLOCKS) * CORE_GROUPS
            if block == 0:
                loaded_query = pl.load(query, [token * H, 0], [H, HEAD_DIM], target_memory=pl.Mem.Mat)
                query_now = pl.yield_(loaded_query)
            else:
                query_now = pl.yield_(query_old)
            if pl.read(mask, [token, block]) > 0:
                slot = worker * QK_TRANSFER_SLOTS + tick % QK_TRANSFER_SLOTS
                transfer_row = slot * H
                pl.system.sync_wait(EVENT_BASE + QK_KV_READY_EVENT,
                                    pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                # 保持三个存储身份；每次只替换本拍一个槽，不物化不同Mat的选择。
                if tick % QK_TRANSFER_SLOTS == 0:
                    fresh_a = load_direct_block(main_cache, indices, kv_transfer, token, block, slot)
                    written_a = emit_qk(query_now, fresh_a, scores, transfer_row)
                    fill_a, fill_b, fill_c = pl.yield_(fresh_a, slot_b, slot_c)
                else:
                    if tick % QK_TRANSFER_SLOTS == 1:
                        fresh_b = load_direct_block(main_cache, indices, kv_transfer, token, block, slot)
                        written_b = emit_qk(query_now, fresh_b, scores, transfer_row)
                        tail_a, tail_b, tail_c = pl.yield_(slot_a, fresh_b, slot_c)
                    else:
                        fresh_c = load_direct_block(main_cache, indices, kv_transfer, token, block, slot)
                        written_c = emit_qk(query_now, fresh_c, scores, transfer_row)
                        tail_a, tail_b, tail_c = pl.yield_(slot_a, slot_b, fresh_c)
                    fill_a, fill_b, fill_c = pl.yield_(tail_a, tail_b, tail_c)
                pl.system.sync_set(EVENT_BASE + QK_SCORE_READY_EVENT,
                                   pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                live_a, live_b, live_c = pl.yield_(fill_a, fill_b, fill_c)
            else:
                live_a, live_b, live_c = pl.yield_(slot_a, slot_b, slot_c)
            next_a, next_b, next_c, next_q = pl.yield_(live_a, live_b, live_c, query_now)
        else:
            next_a, next_b, next_c, next_q = pl.yield_(slot_a, slot_b, slot_c, query_old)
        if tick >= QK_PRE_LAUNCH:
            work = tick - QK_PRE_LAUNCH
            block_pv = work % SPARSE_BLOCKS
            token_pv = worker + (work // SPARSE_BLOCKS) * CORE_GROUPS
            if pl.read(mask, [token_pv, block_pv]) > 0:
                row_pv = (worker * QK_TRANSFER_SLOTS + work % QK_TRANSFER_SLOTS) * H
                pl.system.sync_wait(EVENT_BASE + QK_PROB_READY_EVENT,
                                    pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                prob_mat = pl.load(probabilities, [row_pv, 0], [H, ATTN_K_TILE], target_memory=pl.Mem.Mat)
                prob_left = pl.tile.move(prob_mat, target_memory=pl.Mem.Left)
                if work % QK_TRANSFER_SLOTS == 0:
                    out_a = emit_pv(prob_left, next_a, products, row_pv)
                else:
                    if work % QK_TRANSFER_SLOTS == 1:
                        out_b = emit_pv(prob_left, next_b, products, row_pv)
                    else:
                        out_c = emit_pv(prob_left, next_c, products, row_pv)
                pl.system.sync_set(EVENT_BASE + QK_PV_READY_EVENT,
                                   pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
        slot_a, slot_b, slot_c, query_old = pl.yield_(next_a, next_b, next_c, next_q)
    return scores, products
