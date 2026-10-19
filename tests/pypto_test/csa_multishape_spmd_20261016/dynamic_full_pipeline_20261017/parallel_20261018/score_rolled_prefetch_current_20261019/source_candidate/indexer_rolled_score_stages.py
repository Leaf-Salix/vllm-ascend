# SPDX-License-Identifier: Apache-2.0
"""动态 CSA 的 G6 Score 两面板循环：压缩展开体，保留跨组流水。

整体数据流与阅读顺序
    load_key_slot 按本轮 Native 页表向指定 L1 槽加载两页 Key；
    score_rolled_pair_leaf 复用外层已补齐的 query Left 和系数 Left，先算
    QK0，再循环七组 QK(2g+1)/QK(2g+2) 与 WS(2g)/WS(2g+1)，最后排空
    QK15/WS14/WS15。QK 为 INT8→INT32，FIX 按 2^-10 缩放并 ReLU 转
    FP16，WS 为完整 K384 的 FP16→FP32，与原舍入和归约边界一致。

动态维度与适用区间
    外层按真实 device metadata 分请求、分叶、补齐尾 query，并领取工作。
    此处 worker、请求号、cache 长度、leaf 起点和 steps 均是运行时 Scalar；
    页号、物理地址及有效范围来自本轮 device Tensor。384=6 query×64头、
    N64、1024 候选是资源策略，不能视为固定 B/S/history。仅适用于外层
    选择的 G6/N64/1024/balanced/prefetch1 组合，其余组合保留原实现。

核内流水和资源
    两个 Key L1 槽为 [2*64,128] INT8；两个 QK 结果槽为 [384,2*64] FP16。
    偶数/奇数子窗地址静态，循环只携带原位池，不携带地址不同的 Right phi。
    下一偶数 Key 在组内预取到 Right；下一奇数 Key 在上一组预取到 L1，
    跨组继续消费上一偶数 QK。与旧每组 drain 的循环不同，不在组尾计算
    最后一份 WS；与旧 L1 ring 不同，不等到下一组才从 GM 开始载入 Key。
    全部 DMA/计算依赖由原 tile 数据流和编译器维护，不能删 fence 求性能。

核间同步与写者
    本模块只有 AIC，必须与外层原双 AIV 消费者在同一 MIX 同时推进。
    transfer 每核双槽，每步发布一次 FIX READY，第 2 步起先等双方 MTE2 ACK。
    叶尾 ACK 排空由外层统一执行；不新增 task、跨核信号或 GM 工作区。
    AIV Top-K、scale、query 尾处理以及最终输出顺序全部保持外层原实现。
    编译通过不能替代回边别名审计、完整精度、图重放与同轮整层性能验证。
"""

import pypto.language as pl

from .runtime_abi import (
    B, ICACHE_PAGES, ICACHE_BYTES, INDEXER_COLS,
    KEY0_ROWS, KEY64_ROWS, INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS,
)


@pl.jit.inline(auto_scope=False)
def load_key_slot(
    pool: pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat],
    key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    native_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    cache_length: pl.Scalar[pl.INDEX],
    page_bytes: pl.Scalar[pl.INDEX],
    first: pl.Scalar[pl.INDEX],
    panel: pl.Scalar[pl.INDEX],
    SLOT: pl.constexpr,
) -> pl.Tile[[128, 128], pl.INT8, pl.Mem.Mat]:
    # 本地别名接收 DPS 返回值；不能重绑定输入形参而丢失原位池的身份。
    local_pool = pool
    for page in pl.unroll(2):
        logical_row = first + panel * 64 + page * 32
        column = pl.min(logical_row // 32, pl.max((cache_length - 1) // 32, 0))
        if column >= 0 and column < pl.tensor.dim(table, 1):
            physical = pl.cast(pl.read(table, [request, column]), pl.INDEX)
            if physical >= 0 and physical < pl.tensor.dim(native_cache, 0):
                byte_offset = physical * page_bytes
                # Native 每页含 4096B key 与 64B scale，两种二维别名覆盖
                # 128B/64B 对齐起点；保留页范围保护，不在核内重排 GM cache。
                if byte_offset % 128 == 0:
                    local_pool = pl.gather_row(
                        local_pool, key0, [SLOT * 64 + page * 32, 0],
                        [byte_offset // 128, 0], [32, 128],
                    )
                else:
                    local_pool = pl.gather_row(
                        local_pool, key64, [SLOT * 64 + page * 32, 0],
                        [byte_offset // 128, 0], [32, 128],
                    )
    return local_pool


@pl.jit.inline(auto_scope=False)
def score_rolled_pair_leaf(
    query: pl.Tile[[384, 128], pl.INT8, pl.Mem.Left],
    coefficient: pl.Tile[[16, 384], pl.FP16, pl.Mem.Left],
    key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    native_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    request: pl.Scalar[pl.INDEX],
    cache_length: pl.Scalar[pl.INDEX],
    page_bytes: pl.Scalar[pl.INDEX],
    leaf_begin: pl.Scalar[pl.INDEX],
    steps: pl.Scalar[pl.INDEX],
    READY: pl.constexpr,
    CONSUMED: pl.constexpr,
):
    for step in pl.range(steps):
        # 双槽跨核 FIFO：等待的是两 AIV 已完成 DMA 的 ACK，不是最终 Top-K。
        if step >= 2:
            pl.system.sync_wait(CONSUMED, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        first = leaf_begin + step * 1024
        transfer_row = worker * 12 + step % 2 * 6
        key_pool = pl.create_tile([128, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
        score_pool = pl.create_tile([384, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
        first_panel = pl.cast(0, pl.INDEX)
        next_panel = pl.cast(1, pl.INDEX)
        key_pool = load_key_slot(key_pool, key0, key64, native_cache, table,
                                 request, cache_length, page_bytes, first, first_panel, 0)
        prime_key = pl.tile.extract(pl.tile.transpose_view(key_pool), 0, 0,
                                    [128, 64], target_memory=pl.Mem.Right)
        key_pool = load_key_slot(key_pool, key0, key64, native_cache, table,
                                 request, cache_length, page_bytes, first, next_panel, 1)
        prime_qk = pl.tile.matmul(query, prime_key)
        score_pool = pl.tile.assemble(score_pool, prime_qk, [0, 0],
                                      pre_quant=0.0009765625, pre_relu=True)

        # 运行时循环只保留两份数学体；固定双槽避免动态子窗模运算及存储合流。
        # 进入第 g 组：score 槽0含 QK(2g)，Key 槽1含 Key(2g+1)。
        # 离开：score 槽0含 QK(2g+2)，Key 槽1已载入 Key(2g+3)。
        for pair in pl.range(7):
            odd_panel = pair * 2 + 1
            even_panel = odd_panel + 1
            following_panel = even_panel + 1
            odd_key = pl.tile.extract(pl.tile.transpose_view(key_pool), 0, 64,
                                      [128, 64], target_memory=pl.Mem.Right)
            key_pool = load_key_slot(key_pool, key0, key64, native_cache, table,
                                     request, cache_length, page_bytes, first, even_panel, 0)
            # 同组下一 Key 提前进入第二个 Right 槽，与当前 QK/WS 并存。
            even_key = pl.tile.extract(pl.tile.transpose_view(key_pool), 0, 0,
                                       [128, 64], target_memory=pl.Mem.Right)
            previous_even = pl.tile.extract(score_pool, 0, 0, [384, 64],
                                            target_memory=pl.Mem.Right)
            odd_qk = pl.tile.matmul(query, odd_key)
            previous_ws = pl.tile.matmul(coefficient, previous_even)
            score_pool = pl.tile.assemble(score_pool, odd_qk, [0, 64],
                                          pre_quant=0.0009765625, pre_relu=True)
            pl.store(pl.set_validshape(previous_ws, 6, 64),
                     [transfer_row, (odd_panel - 1) * 64], transfer)

            # 已提取本组奇数 Key 后，才复用其 L1 槽预取下一组，包含最终 Key15。
            key_pool = load_key_slot(key_pool, key0, key64, native_cache, table,
                                     request, cache_length, page_bytes, first, following_panel, 1)
            previous_odd = pl.tile.extract(score_pool, 0, 64, [384, 64],
                                           target_memory=pl.Mem.Right)
            even_qk = pl.tile.matmul(query, even_key)
            odd_ws = pl.tile.matmul(coefficient, previous_odd)
            score_pool = pl.tile.assemble(score_pool, even_qk, [0, 0],
                                          pre_quant=0.0009765625, pre_relu=True)
            pl.store(pl.set_validshape(odd_ws, 6, 64),
                     [transfer_row, odd_panel * 64], transfer)

        # 最后一组已预取 Key15；先让 QK15 与 WS14 并存，再处理 WS15。
        final_key = pl.tile.extract(pl.tile.transpose_view(key_pool), 0, 64,
                                    [128, 64], target_memory=pl.Mem.Right)
        before_last = pl.tile.extract(score_pool, 0, 0, [384, 64],
                                      target_memory=pl.Mem.Right)
        final_qk = pl.tile.matmul(query, final_key)
        before_last_ws = pl.tile.matmul(coefficient, before_last)
        score_pool = pl.tile.assemble(score_pool, final_qk, [0, 64],
                                      pre_quant=0.0009765625, pre_relu=True)
        pl.store(pl.set_validshape(before_last_ws, 6, 64), [transfer_row, 896], transfer)
        final_right = pl.tile.extract(score_pool, 0, 64, [384, 64],
                                      target_memory=pl.Mem.Right)
        final_ws = pl.tile.matmul(coefficient, final_right)
        pl.store(pl.set_validshape(final_ws, 6, 64), [transfer_row, 960], transfer)
        pl.system.sync_set(READY, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
    return transfer
