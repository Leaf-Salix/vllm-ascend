# SPDX-License-Identifier: Apache-2.0
"""动态 CSA Indexer G6 的双 M192/N128 候选：保留六query分组，只改变核内数学分块。

数据流与阅读顺序
    load_key_panel 依照真实 Native 页表把四个32行页填入 L1 双槽，保留
    key0/key64 字节对齐别名与全部地址保护。score_split_n128_leaf 先把六条
    query切成两个M192 Left，并分别读取三条有效系数行；每个N128 Key只装入
    一个Right，两个半组复用。QK仍是INT8→INT32，FIX仍按1/1024缩放、ReLU
    转FP16，然后用原FP16系数在Cube归约为FP32，保持舍入边界。

动态范围、分档与输出
    外层负责原G6的group/leaf/visible、动态query尾组补齐和工作领取；本模块
    不重新分叶或分核，不改变G6、两个AIV各三条query、Top-K树与score FIFO。
    所有页址、历史长度、leaf起点、步数和输出行来自当轮device值。128、192
    是N及三query×64头的数学tile，不是B/S/history特化。仅用于已有balanced
    G6、score_tile1024、prefetch1且Key只预取L1的资源组合；其它路径由外层回退。

流水、同步及片上资源
    两Query Left共48KiB，两系数Left共12KiB；Right为16KiB Key与48KiB
    WS输入，不能再预取第二份Key到Right。下一Key仅预取到L1另一槽。
    当前QK与上一半组WS使用不同Acc；先提交当前QK FIX，再写上一WS结果，
    允许FIX与WS计算重叠。这个次序不同于旧静态N128候选先写WS再做QK FIX。
    每1024候选仍发布一次READY，复用双槽前仍等两个AIV的ACK。叶尾排空仍
    由外层执行；本模块不增加task、阶段通知、GM工作区或Host元数据回读。
    容量是设计约束，实际分配、精确结果与性能必须以完整编译/真机验证为准。
"""

import pypto.language as pl

from .runtime_abi import (
    B, COEFF_ROWS, COEFF_COLS, ICACHE_PAGES, ICACHE_BYTES, INDEXER_COLS,
    KEY0_ROWS, KEY64_ROWS, INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS,
)


@pl.jit.inline(auto_scope=False)
def load_key_panel(
    pool: pl.Tile[[256, 128], pl.INT8, pl.Mem.Mat],
    key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    native_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    cache_length: pl.Scalar[pl.INDEX],
    page_bytes: pl.Scalar[pl.INDEX],
    first: pl.Scalar[pl.INDEX],
    panel: pl.Scalar[pl.INDEX],
) -> pl.Tile[[256, 128], pl.INT8, pl.Mem.Mat]:
    pool_local = pool
    slot = panel % 2 * 128
    for page in pl.unroll(4):
        logical_row = first + panel * 128 + page * 32
        column = pl.min(logical_row // 32, pl.max((cache_length - 1) // 32, 0))
        if column >= 0 and column < pl.tensor.dim(table, 1):
            physical = pl.cast(pl.read(table, [request, column]), pl.INDEX)
            if physical >= 0 and physical < pl.tensor.dim(native_cache, 0):
                byte_offset = physical * page_bytes
                if byte_offset % 128 == 0:
                    pool_local = pl.gather_row(pool_local, key0, [slot + page * 32, 0],
                                         [byte_offset // 128, 0], [32, 128])
                else:
                    pool_local = pl.gather_row(pool_local, key64, [slot + page * 32, 0],
                                         [byte_offset // 128, 0], [32, 128])
    return pool_local


@pl.jit.inline(auto_scope=False)
def score_split_n128_leaf(
    query_mat: pl.Tile[[384, 128], pl.INT8, pl.Mem.Mat],
    coefficients: pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16],
    key0: pl.Tensor[[KEY0_ROWS, 128], pl.INT8],
    key64: pl.Tensor[[KEY64_ROWS, 128], pl.INT8],
    native_cache: pl.Tensor[[ICACHE_PAGES, ICACHE_BYTES], pl.INT8],
    table: pl.Tensor[[B, INDEXER_COLS], pl.INT32],
    transfer: pl.InOut[pl.Tensor[[INDEX_TRANSFER_ROWS, INDEX_TRANSFER_COLS], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    group: pl.Scalar[pl.INDEX],
    request: pl.Scalar[pl.INDEX],
    cache_length: pl.Scalar[pl.INDEX],
    page_bytes: pl.Scalar[pl.INDEX],
    leaf_begin: pl.Scalar[pl.INDEX],
    steps: pl.Scalar[pl.INDEX],
    READY: pl.constexpr,
    CONSUMED: pl.constexpr,
):
    query0 = pl.tile.extract(query_mat, 0, 0, [192, 128], target_memory=pl.Mem.Left)
    query1 = pl.tile.extract(query_mat, 192, 0, [192, 128], target_memory=pl.Mem.Left)
    # 原16×384系数为query对角块。只读取本半组三条有效行/192列，其余
    # query的零系数不再参与本次K192归约；不增加或搬运一份GM系数缓存。
    coeff_mat0 = pl.load(coefficients, [group * 16, 0], [16, 192],
                         valid_shape=[3, 192], target_memory=pl.Mem.Mat)
    coeff_mat1 = pl.load(coefficients, [group * 16 + 3, 192], [16, 192],
                         valid_shape=[3, 192], target_memory=pl.Mem.Mat)
    coeff0 = pl.tile.move(coeff_mat0, target_memory=pl.Mem.Left)
    coeff1 = pl.tile.move(coeff_mat1, target_memory=pl.Mem.Left)
    for step in pl.range(steps):
        if step >= 2:
            pl.system.sync_wait(CONSUMED, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        first = leaf_begin + step * 1024
        transfer_row = worker * 12 + step % 2 * 6
        key_pool = pl.create_tile([256, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
        previous_score = pl.create_tile([192, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
        first_panel = pl.cast(0, pl.INDEX)
        key_pool = load_key_panel(key_pool, key0, key64, native_cache, table,
                                  request, cache_length, page_bytes, first, first_panel)
        for panel in pl.unroll(8):
            key = pl.tile.extract(pl.tile.transpose_view(key_pool), 0, panel % 2 * 128,
                                  [128, 128], target_memory=pl.Mem.Right)
            if panel + 1 < 8:
                next_panel = panel + 1
                key_pool = load_key_panel(key_pool, key0, key64, native_cache, table,
                                          request, cache_length, page_bytes, first, next_panel)
            qk0 = pl.tile.matmul(query0, key)
            score0_empty = pl.create_tile([192, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
            if panel > 0:
                previous_right = pl.tile.move(previous_score, target_memory=pl.Mem.Right)
                previous_ws = pl.tile.matmul(coeff1, previous_right)
                score0 = pl.tile.assemble(score0_empty, qk0, [0, 0], pre_quant=0.0009765625, pre_relu=True)
                pl.store(pl.set_validshape(previous_ws, 3, 128),
                         [transfer_row + 3, (panel - 1) * 128], transfer)
            else:
                score0 = pl.tile.assemble(score0_empty, qk0, [0, 0], pre_quant=0.0009765625, pre_relu=True)
            qk1 = pl.tile.matmul(query1, key)
            score0_right = pl.tile.move(score0, target_memory=pl.Mem.Right)
            ws0 = pl.tile.matmul(coeff0, score0_right)
            score1 = pl.create_tile([192, 128], dtype=pl.FP16, target_memory=pl.Mem.Mat)
            score1 = pl.tile.assemble(score1, qk1, [0, 0], pre_quant=0.0009765625, pre_relu=True)
            pl.store(pl.set_validshape(ws0, 3, 128), [transfer_row, panel * 128], transfer)
            previous_score = score1
        last_right = pl.tile.move(previous_score, target_memory=pl.Mem.Right)
        last_ws = pl.tile.matmul(coeff1, last_right)
        pl.store(pl.set_validshape(last_ws, 3, 128), [transfer_row + 3, 896], transfer)
        pl.system.sync_set(READY, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
    return transfer
