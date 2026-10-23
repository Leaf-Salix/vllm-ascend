# SPDX-License-Identifier: Apache-2.0
"""双query实验的设备工作表：同请求配对，保留动态Q/L与奇数尾部。

当前只提供待接入实验的metadata生产者，不改变正式HCA，也没有实现或宣称
跨核流水已可用。12组×2Cube×4AIV的通知区单独按64B缓存行分隔并初始化。
工作表只在UB中逐项更新，最后一次MTE3发布，避免每个字段进行Scalar GM短写。
"""

import pypto.language as pl

T_DYN = pl.dynamic("HCA_PAIR_TOKEN_CAPACITY")
B_DYN = pl.dynamic("HCA_PAIR_REQUEST_CAPACITY")
Q_DYN = pl.dynamic("HCA_PAIR_QUERY_BOUNDS")
MAX_TOKENS = 384
FIELDS = 16
CLUSTERS = 12
SIGNALS_PER_CLUSTER = 10
SIGNAL_ROWS = CLUSTERS * SIGNALS_PER_CLUSTER


@pl.jit.inline(auto_scope=False)
def build_pair_plan(
    query_bounds: pl.Tensor[[Q_DYN], pl.INT32],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    plan: pl.Tensor[[T_DYN, FIELDS], pl.INT32],
    count: pl.Tensor[[FIELDS], pl.INT32],
    signals: pl.Tensor[[SIGNAL_ROWS, FIELDS], pl.INT32],
):
    tokens = pl.tensor.dim(positions, 0)
    requests = pl.tensor.dim(seq_lens, 0)
    count_2d = pl.reshape(count, [1, FIELDS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_pair_plan") as ready:
        # 合法Native前缀单调且结束不超过tokens，因此pair数不超过token容量。
        # 空请求或length<=0不发出工作；单请求奇数尾部明确标记count=1。
        table = pl.tile.full([MAX_TOKENS, FIELDS], dtype=pl.INT32, value=0)
        for request, (cursor,) in pl.range(requests, init_values=(pl.cast(0, pl.INDEX),)):
            begin = pl.max(0, pl.cast(pl.read(query_bounds, [request]), pl.INDEX))
            end = pl.min(tokens, pl.cast(pl.read(query_bounds, [request + 1]), pl.INDEX))
            length = pl.read(seq_lens, [request])
            if length > 0:
                queries = pl.yield_(pl.max(0, end - begin))
            else:
                queries = pl.yield_(pl.cast(0, pl.INDEX))
            for local, (write_row,) in pl.range((queries + 1) // 2, init_values=(cursor,)):
                first = begin + local * 2
                members = pl.min(2, end - first)
                second = first + members - 1
                # 0..7：两个token、有效query数、request、两个position、长度、query起点。
                # 第二query缺席时使用安全的first地址，消费者依据members屏蔽，不能额外读下一请求。
                pl.tile.write(table, [write_row, 0], pl.cast(first, pl.INT32))
                pl.tile.write(table, [write_row, 1], pl.cast(second, pl.INT32))
                pl.tile.write(table, [write_row, 2], pl.cast(members, pl.INT32))
                pl.tile.write(table, [write_row, 3], pl.cast(request, pl.INT32))
                pl.tile.write(table, [write_row, 4], pl.cast(pl.read(positions, [first]), pl.INT32))
                pl.tile.write(table, [write_row, 5], pl.cast(pl.read(positions, [second]), pl.INT32))
                pl.tile.write(table, [write_row, 6], length)
                pl.tile.write(table, [write_row, 7], pl.cast(begin, pl.INT32))
                row_done = pl.yield_(write_row + 1)
            total = pl.yield_(row_done)
        if total > 0:
            pl.store(pl.set_validshape(table, total, FIELDS), [0, 0], plan)
        header = pl.tile.full([1, FIELDS], dtype=pl.INT32, value=0)
        pl.tile.write(header, [0, 0], pl.cast(total, pl.INT32))
        pl.tile.write(header, [0, 1], pl.cast(tokens, pl.INT32))
        pl.tile.write(header, [0, 2], pl.cast(requests, pl.INT32))
        pl.store(header, [0, 0], count_2d)
        # 每次调用重置，未来消费者必须显式依赖ready并按A3发布/读取协议使用。
        # 当前没有通知/等待实现；全零只证明初始化，不代表跨核通信已验证。
        pl.store(pl.tile.full([SIGNAL_ROWS, FIELDS], dtype=pl.INT32, value=0), [0, 0], signals)
    return ready


@pl.jit(auto_scope=False)
def pair_plan_test(
    query_bounds: pl.Tensor[[Q_DYN], pl.INT32],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    plan: pl.Out[pl.Tensor[[T_DYN, FIELDS], pl.INT32]],
    count: pl.Out[pl.Tensor[[FIELDS], pl.INT32]],
    signals: pl.Out[pl.Tensor[[SIGNAL_ROWS, FIELDS], pl.INT32]],
):
    query_bounds.bind_dynamic(0, Q_DYN)
    seq_lens.bind_dynamic(0, B_DYN)
    positions.bind_dynamic(0, T_DYN)
    plan.bind_dynamic(0, T_DYN)
    ready = build_pair_plan(query_bounds, seq_lens, positions, plan, count, signals)
    return plan, count, signals
