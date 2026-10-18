# SPDX-License-Identifier: Apache-2.0
"""动态wave的点对点依赖：根 → Top-K → M8计划 → Attention。

整体思路与阅读顺序
    wait_wave_roots按当前query的真实visible和全局(group,leaf)所有权等待根；
    wait_wave_topk_for_plan等本AIV负责的八行计划块需要的Top-K；
    wait_wave_plan_for_attention等本MIX组的query需要的计划块。
    三者仅作用于当前wave，均不创建全核屏障，也不把wave内的编号作为分核编号。

动态性与信号布局
    B/T/Q/L/position、组数、叶数与wave边界均是运行时值。G6/24组/48向量/M8
    来自现有数学及硬件策略。原60个phase后每wave独占ROOT/TOPK/PLAN三阶段，
    (phase,core)一位写者只发布一次。本次generation不因wave变化，也不在核内
    清零复用槽。容量由捕获前T的保守上界保证；首次使用必须全零。

正确性约束
    写者完成本wave全部所属业务输出后才发布。等待只轮询真正写过对应数据的
    writer：空query不等待根，空lane不伪造根，dummy仍需Top-K的空值与计划输出。
    roots的ACK仅表示Score FIFO已读取，不能用它代替根排序完成。每个helper
    使用csa_signal_wait维护信号可见性，全部必要信号满足后统一fence/cacheinvalid
    维护业务数据；A3 Scalar非cache一致性仍然遵守原phase_sync合同。
    为保持实现直接，目前同一owner可重复读取信号，未使用未经验证的64位去重
    掩码。该Scalar成本需由完整L1计时判断；本文件不是性能通过证据。
"""

import pypto.language as pl

from .indexer_group_stages import (
    OFFSET_ROWS_DYN, PLAN_WORDS, TOKENS_DYN, indexer_query_visibility,
)
from .indexer_publish_stages import indexer_visible_root_count
from .phase_sync import SIGNAL_WORDS, SIGNAL_SLOT_WORDS
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request
from .signals import csa_signal_wait


@pl.jit.inline(auto_scope=False)
def wait_wave_roots(
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    phase: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    worker: pl.Scalar[pl.INDEX],
    query_begin: pl.Scalar[pl.INDEX],
    query_end: pl.Scalar[pl.INDEX],
):
    """每个merge worker只等待自己query实际消费的叶根；平衡G6每叶一根。"""
    tokens = pl.cast(pl.tensor.dim(positions, 0), pl.INDEX)
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    first_query = query_begin + (worker - query_begin % 48 + 48) % 48
    for query in pl.range(first_query, query_end, 48):
        request, begin, end, active = query_request(bounds, lengths, query, tokens)
        if active != 0:
            visible, causal = indexer_query_visibility(bounds, lengths, positions, request, query)
            roots = indexer_visible_root_count(visible, plan, 1, True)
            group = pl.cast(pl.read(offsets, [request]), pl.INDEX) + (query - begin) // 6
            lane = ((query - begin) % 6) // 3
            for leaf in pl.range(roots):
                item = group * leaves + leaf
                if groups < 24:
                    item = leaf * groups + group
                producer = 24 + (item % 24) * 2 + lane
                offset = (phase * 72 + producer) * SIGNAL_SLOT_WORDS
                signal = csa_signal_wait(signal, offset, generation)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signal


@pl.jit.inline(auto_scope=False)
def wait_wave_topk_for_plan(
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    phase: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    worker: pl.Scalar[pl.INDEX],
    query_begin: pl.Scalar[pl.INDEX],
    query_end: pl.Scalar[pl.INDEX],
):
    """M8全局tile owner=(query//8)%48；仅等待自己将读取的Top-K行。"""
    begin_tile = query_begin // 8
    first_tile = begin_tile + (worker - begin_tile % 48 + 48) % 48
    for first in pl.range(first_tile * 8, query_end, 48 * 8):
        for query in pl.range(first, pl.min(first + 8, query_end)):
            producer = 24 + query % 48
            offset = (phase * 72 + producer) * SIGNAL_SLOT_WORDS
            signal = csa_signal_wait(signal, offset, generation)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signal


@pl.jit.inline(auto_scope=False)
def wait_wave_plan_for_attention(
    signal: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    phase: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    core_group: pl.Scalar[pl.INDEX],
    query_begin: pl.Scalar[pl.INDEX],
    query_end: pl.Scalar[pl.INDEX],
):
    """query全局owner=q%24，计划owner=q//8%48；两者不能混用。"""
    first_query = query_begin + (core_group - query_begin % 24 + 24) % 24
    for query in pl.range(first_query, query_end, 24):
        producer = 24 + (query // 8) % 48
        offset = (phase * 72 + producer) * SIGNAL_SLOT_WORDS
        signal = csa_signal_wait(signal, offset, generation)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signal
