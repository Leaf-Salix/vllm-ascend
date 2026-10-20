# SPDX-License-Identifier: Apache-2.0
"""长档G6的AIC wave交错执行器：Score完整step后推进前波Attention一拍。

整体思路与阅读顺序
    score_cube_leaf_init_g6保留原leaf的动态Query装入、片上补齐和Coeff初始化；
    interleave_cube_wave_g6持有两个阶段的显式Tile/Scalar状态，每个本地逻辑轮
    先执行一个完整Score step，再执行一个Attention tick。这里不创建task，
    不重新分核，不发布ROOT/TopK/plan phase，也不决定请求wave的边界。

动态输入与所有权
    Score输入是device group半开区间，Attention输入是前波全局query半开区间；
    两者可分别为空。Score继续按全局item%24领取叶，Attention保持q%24。
    当前请求Q/L/position/page以及所有GM容量仍为动态Tensor/Scalar；G6、
    N64、1024候选和固定片上Tile仅指定已验证数学策略，不表示B/T/历史档位。
    query_first必须由全局query区间和物理worker求出，不能用worker代替。

状态、资源与同步
    Query INT8[384,128]在Mat跨step保留，每步重新TMOV到Left；Coeff FP16
    [16,384] Mat跨step保留，每步与两半Query一起恢复Left。Attention的Q BF16[64,512]与KV BF16[256,512]
    Mat同样显式携带，GM ring仍三槽、AIC内部KV两槽，Attention每拍先PV后QK。
    create只构造尚未使用的full-valid句柄，进入有效叶/有效Attention块前不能
    读取其中数据。full-valid是Tile类型合同，不意味着未初始化数据可消费。
    原step>=2的ACK、每leaf末min(steps,2)个ACK及Attention末两拍均保留；
    ACK只表示Score FIFO已读取，绝不表示AIV排序或根写回完成。
    输入发布获取、FFTS域、对端AIV的Attention→Score顺序及wave phase由外层
    负责。源码次序不代替MTE1/M/FIX实际完成依赖；联合lowering仍须检查Mat、
    Left及回边生命周期。本文件尚未编译/上卡，不预记容量或性能通过。
"""

import pypto.language as pl

from .attention_tick_stages import (
    ATTN_K_TILE,
    FFTS_WORDS,
    HEAD_DIM as ATTN_HEAD_DIM,
    HEAD_RING_ROWS_DYN,
    KV_RING_ROWS_DYN,
    PLAN_CAP_DYN,
    Q_ROWS_DYN,
    QK_PRE_LAUNCH,
    SPARSE_BLOCKS,
    attention_cube_init,
    attention_cube_tick,
)
from .indexer_group_stages import (
    COEFF_COLS_DYN,
    COEFF_ROWS_DYN,
    OFFSET_ROWS_DYN,
    PLAN_WORDS,
    TOKENS_DYN,
)
from .indexer_score_stages import (
    CACHE_PAGES_DYN,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    KEY0_ROWS_DYN,
    KEY64_ROWS_DYN,
    NATIVE_QLI_WEIGHT_ROWS,
    PAGE_BYTES_DYN,
    QUERY_HEAD_ROWS_DYN,
    TABLE_COLUMNS_DYN,
    TRANSFER_COLS_DYN,
    TRANSFER_ROWS_DYN,
    indexer_score_cube_step_g6,
)
from .interleave_control import next_score_leaf, score_wave_step_count
from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN


@pl.jit.inline(auto_scope=False)
def score_cube_leaf_init_g6(
    qr_hadamard_i8: pl.Tensor[[QUERY_HEAD_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    coefficients: pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16],
    buf_group: pl.Scalar[pl.INDEX],
    buf_query: pl.Scalar[pl.INDEX],
    buf_live: pl.Scalar[pl.INDEX],
) -> tuple[
    pl.Tile[[384, 128], pl.INT8, pl.MemorySpace.Mat],
    pl.Tile[[16, 384], pl.FP16, pl.MemorySpace.Mat],
]:
    """仅有效叶调用；保留原live load/完整valid/尾组补齐，不读取下一请求。

    返回驻Mat Query和完整Coeff Mat；step内部恢复当前N64两半Left，
    Attention切换时不依赖旧Left内容，仍保留完整六query系数及尾query零项。
    """
    buf_query_vector = pl.load(
        qr_hadamard_i8,
        [buf_query * IDX_N_HEADS, 0],
        [6 * IDX_N_HEADS, IDX_HEAD_DIM],
        target_memory=pl.MemorySpace.Mat,
        valid_shape=[buf_live * IDX_N_HEADS, IDX_HEAD_DIM],
    )
    # A3紧凑L0C/TINSERT合同要求完整物理M；先恢复full-valid再补齐片上尾行。
    # 这里只复制本组首条有效query，对应无效query的Coeff仍由原生产者置零。
    buf_query_vector = pl.set_validshape(buf_query_vector, 6 * IDX_N_HEADS, IDX_HEAD_DIM)
    for padding_lane in pl.unroll(6):
        if padding_lane >= buf_live:
            buf_query_vector = pl.gather_row(
                buf_query_vector,
                qr_hadamard_i8,
                [padding_lane * IDX_N_HEADS, 0],
                [buf_query * IDX_N_HEADS, 0],
                [IDX_N_HEADS, IDX_HEAD_DIM],
            )
    buf_coefficients_l1 = pl.load(
        coefficients,
        [buf_group * NATIVE_QLI_WEIGHT_ROWS, 0],
        [NATIVE_QLI_WEIGHT_ROWS, 6 * IDX_N_HEADS],
        target_memory=pl.MemorySpace.Mat,
    )
    return buf_query_vector, buf_coefficients_l1


@pl.jit.inline(auto_scope=False)
def interleave_cube_wave_g6(
    qr_hadamard_i8: pl.Tensor[[QUERY_HEAD_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    coefficients: pl.Tensor[[COEFF_ROWS_DYN, COEFF_COLS_DYN], pl.FP16],
    idx_kv_cache: pl.Tensor[[KEY0_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_kv_cache_shift64: pl.Tensor[[KEY64_ROWS_DYN, IDX_HEAD_DIM], pl.INT8],
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[TOKENS_DYN], pl.INT64],
    kv_seq_lens: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    indexer_transfer: pl.InOut[pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32]],
    q_flat: pl.Tensor[[Q_ROWS_DYN, ATTN_HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, ATTN_HEAD_DIM], pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    worker: pl.Scalar[pl.INDEX],
    score_group_begin: pl.Scalar[pl.INDEX],
    score_group_end: pl.Scalar[pl.INDEX],
    attention_query_begin: pl.Scalar[pl.INDEX],
    attention_query_end: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr = 24,
    SCORE_READY_EVENT: pl.constexpr = 0,
    SCORE_CONSUMED_EVENT: pl.constexpr = 1,
    ATTN_EVENT_BASE: pl.constexpr = 2,
) -> tuple[
    pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32],
    pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32],
    pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_HEAD_DIM], pl.FP32],
]:
    """推进本物理AIC的一个wave；当前Score与前波Attention可分别为空。

    caller只在G6/balanced/roots1/N64/1024/prefetch1/toL0路径调用，且
    CORE_GROUPS必须为24，与control及原FFTS核组一致。Score输入及Attention
    前波plan/query/KV已按真实owner acquire；对端双AIV使用相同区间和事件域，
    每轮先Attention后Score。返回不表示根节点、TopK或plan已经发布。

    Query/Coeff及Attention Q/KV四份Tile显式跨轮携带；叶游标、当前step、
    step数、batch/cache/逻辑起点/lane_span共七个Scalar同样显式携带。
    计数与实际叶扫描读取同一本次不可变device计划，不能用Host shape替换。
    """
    native_page_bytes = pl.tensor.dim(idx_native_kv_cache, 1)
    score_steps = score_wave_step_count(
        bounds, kv_seq_lens, position_ids, offsets, plan,
        worker, score_group_begin, score_group_end,
    )
    t_dim, qk_core, full_block_count, attention_q, attention_kv = attention_cube_init(
        q_flat, ffts_workspace, worker, CORE_GROUPS
    )
    # 只裁当前wave的query范围。全局query仍按q%24分核，GM ring仍用qk_core；
    # init返回的完整T/原block_count不改写Tensor的维度，不能把qk_core当query_first。
    query_first = attention_query_begin + (
        (worker - attention_query_begin % CORE_GROUPS + CORE_GROUPS) % CORE_GROUPS
    )
    query_count = pl.max(
        (attention_query_end - query_first + CORE_GROUPS - 1) // CORE_GROUPS, 0
    )
    block_count = query_count * SPARSE_BLOCKS
    attention_ticks = block_count + QK_PRE_LAUNCH
    rounds = pl.max(score_steps, attention_ticks)

    # 仅建立full-valid句柄。没有有效叶时只carry、不读取这两份未初始化数据；
    # 有效叶必须先执行load/set_validshape/padding，再进入完整Score step。
    score_query = pl.create_tile([384, 128], dtype=pl.INT8, target_memory=pl.MemorySpace.Mat)
    score_coeff = pl.create_tile([16, 384], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    zero = pl.cast(0, pl.INDEX)
    for local_round, (
        query_iter, coeff_iter, attention_q_iter, attention_kv_iter,
        cursor_iter, step_iter, steps_iter, batch_iter, cache_iter, begin_iter, span_iter,
    ) in pl.range(
        rounds,
        init_values=(
            score_query, score_coeff, attention_q, attention_kv,
            worker, zero, zero, zero, zero, zero, zero,
        ),
    ):
        # AIC的有向顺序固定为Score(r)→Attention(r)。一侧耗尽后只跳过该侧；
        # 不为无工作叶制造READY/ACK，也不要求不同物理核拥有相同步数。
        if local_round < score_steps:
            if steps_iter == 0:
                (
                    next_item, found, group, leaf, batch, query, live,
                    cache_len, logical_begin, valid_count, steps, lane_span,
                ) = next_score_leaf(
                    bounds, kv_seq_lens, position_ids, offsets, plan,
                    cursor_iter, score_group_begin, score_group_end,
                )
                if found > 0:
                    query_loaded, coeff_loaded = score_cube_leaf_init_g6(
                        qr_hadamard_i8, coefficients, group, query, live
                    )
                    query_selected, coeff_selected = pl.yield_(query_loaded, coeff_loaded)
                else:
                    # 正常不可变计划下，score_steps保证这里存在下一有效叶；
                    # 仍保留found保护，绝不将create的占位Tile用于数学/业务访存。
                    query_selected, coeff_selected = pl.yield_(query_iter, coeff_iter)
                (
                    query_ready, coeff_ready, cursor_ready, step_ready, steps_ready,
                    batch_ready, cache_ready, begin_ready, span_ready,
                ) = pl.yield_(
                    query_selected, coeff_selected, next_item, zero, steps,
                    batch, cache_len, logical_begin, lane_span,
                )
            else:
                (
                    query_ready, coeff_ready, cursor_ready, step_ready, steps_ready,
                    batch_ready, cache_ready, begin_ready, span_ready,
                ) = pl.yield_(
                    query_iter, coeff_iter, cursor_iter, step_iter, steps_iter,
                    batch_iter, cache_iter, begin_iter, span_iter,
                )
            if steps_ready > 0:
                # Attention可在上一轮使用Left；每步从驻Mat Query重新TMOV恢复。
                # Coeff同样驻Mat；本步内恢复两份Left，不跨Attention保留Left。
                # GM仍是原InOut别名，只接收本次返回名；不能把固定地址的Tensor
                # 重赋值混进下方显式Tile/Scalar phi，否则SSA会追加隐式返回值。
                indexer_transfer_written = indexer_score_cube_step_g6(
                    query_ready,
                    coeff_ready,
                    idx_kv_cache,
                    idx_kv_cache_shift64,
                    idx_native_kv_cache,
                    idx_block_table,
                    indexer_transfer,
                    worker,
                    batch_ready,
                    cache_ready,
                    begin_ready,
                    span_ready,
                    step_ready,
                    native_page_bytes,
                    SCORE_READY_EVENT,
                    SCORE_CONSUMED_EVENT,
                )
                next_step = step_ready + 1
                if next_step == steps_ready:
                    # 原叶尾ACK排空不删；下一叶从slot0开始前不能残留旧ACK。
                    # ACK仅表示两AIV已读取Score FIFO，不代表root/TopK/plan就绪。
                    for drain in pl.range(pl.min(steps_ready, 2)):
                        pl.system.sync_wait(
                            SCORE_CONSUMED_EVENT, pipe=pl.PipeType.MTE2,
                            core_type=pl.KernelType.AIC,
                        )
                    step_next, steps_next = pl.yield_(zero, zero)
                else:
                    step_next, steps_next = pl.yield_(next_step, steps_ready)
                step_after, steps_after = pl.yield_(step_next, steps_next)
            else:
                step_after, steps_after = pl.yield_(zero, zero)
            (
                query_after, coeff_after, cursor_after, leaf_step_after, leaf_steps_after,
                batch_after, cache_after, begin_after, span_after,
            ) = pl.yield_(
                query_ready, coeff_ready, cursor_ready, step_after, steps_after,
                batch_ready, cache_ready, begin_ready, span_ready,
            )
        else:
            (
                query_after, coeff_after, cursor_after, leaf_step_after, leaf_steps_after,
                batch_after, cache_after, begin_after, span_after,
            ) = pl.yield_(
                query_iter, coeff_iter, cursor_iter, step_iter, steps_iter,
                batch_iter, cache_iter, begin_iter, span_iter,
            )

        if local_round < attention_ticks:
            # GM三槽与AIC Mat双槽的取模分别留在原tick helper；它先PV(r-2)，
            # 后QK(r)，显式保留末两拍。空query范围仅走两拍无访存的填排空控制。
            score_transfer_written, pv_transfer_written, attention_q_next, attention_kv_next = attention_cube_tick(
                q_flat, valid_block_mask, kv_transfer, score_transfer,
                probability_transfer, pv_transfer, local_round, block_count,
                qk_core, query_first, attention_q_iter, attention_kv_iter,
                CORE_GROUPS, ATTN_EVENT_BASE,
            )
            attention_q_after, attention_kv_after = pl.yield_(attention_q_next, attention_kv_next)
        else:
            attention_q_after, attention_kv_after = pl.yield_(attention_q_iter, attention_kv_iter)
        (
            score_query, score_coeff, attention_q, attention_kv,
            cursor, leaf_step, leaf_steps, batch_idx, cache_len, logical_begin, lane_span,
        ) = pl.yield_(
            query_after, coeff_after, attention_q_after, attention_kv_after,
            cursor_after, leaf_step_after, leaf_steps_after,
            batch_after, cache_after, begin_after, span_after,
        )
    return indexer_transfer, score_transfer, pv_transfer
