# SPDX-License-Identifier: Apache-2.0
"""动态 CSA 长档 G6 的 AIV wave 执行器：先前波 Attention，再当前波 Score。

整体输入、输出与阅读顺序
    interleave_vector_wave_g6 同时接收当前 Score 的 device group 半开区间，
    以及上一波 Attention 的全局 query 半开区间。next_score_leaf 沿原全局
    item%24 找本核叶，Score 每轮完整消费一个1024候选 step；Attention 每轮
    完整推进一拍。两条流水任何一条结束后，仅排空另一条，不增加虚假业务块。
    返回原 pair_arena、Attention 的五份可写GM别名及本lane的root_wrote Scalar。
    本函数不创建task、不决定wave边界、不执行ROOT/TopK/PLAN phase通知。

动态范围、布局与所有权
    Q/L/positions/页表/组描述符全部来自同轮device Tensor；T由原positions
    shape读取，不能按B*S推算，也不把wave长度用于EARLY_KV阈值。Score仅用于
    既有G6/balanced/单根策略，24组是硬件分工；group区间不会重编号全局item。
    Attention query_first满足q%24==worker，物理worker始终0..23；GM环%3与
    AIC的L1环%2独立。lane0/1各消费Score的3条query和Attention的32个heads。
    真实wave可使某核没有Score叶或Attention query，两种空域独立处理。

跨轮Tile与同步生命周期
    Score roots[3,1024]/segments[6,1024]及Attention的m/l[32,1]、oi左右
    [32,256]均显式作为loop carry传入/返回，swap[1,1024]只读。所有这些
    Tile类型使用固定数学shape、完整valid_shape；真实tail仍由Scalar/mask裁剪。
    Score句柄初建时不读取，只有有效叶首step才初始化NEG_INF；同叶后续step
    保留状态，叶末写根后下一叶重新初始化。空query lane仍消费FIFO并ACK。
    AIV每轮Attention→Score，与对端AIC的Score→Attention配对；保留原READY、
    ACK、有效mask与tail drain。AIV不执行AIC叶末ACK drain，不能多消费事件。
    根写回发生在每叶全部step之后，root_wrote只说明本lane至少写过一个真根；
    外层必须等本函数整wave返回，再按原发布/cache维护合同通知所有根完成。

实现与验证边界
    此处仅组织现有完整数学helper，不移动scale/sort/softmax/归并内部次序。
    额外的Score Query驻Mat重装由AIC执行器承担。本文件是联合候选源码，
    尚未编译/上卡；full-valid Tile的JIT能力不等于本组合的UB峰值、Mat布局、
    局部事件或精度已经验证，不能据Scalar循环结束推断异步引擎已经完成。

当前新增的状态与真实路径
    第五份Attention Tile为[6,32] FP32的本lane统计环，softmax写与PV消费
    之间随外层循环携带。physical_mode每wave按相同plan容量谓词计算一次。
    Score另携带next_score_leaf给出的实际valid；连续scale使用同代布局证明，
    每步只重新读取两个不可变Scalar，不重新扫描页表证明，也不把steps当有效长度。
"""

import pypto.language as pl

from .attention_plan_stages import physical_plan_supported
from .attention_tick_stages import QK_TRANSFER_SLOTS, BLOCK_SIZE

from .attention_tick_stages import (
    ATTN_K_TILE, CMP_TOPK, COMPRESSED_COLUMNS_DYN, COMPRESSED_ROWS_DYN,
    FFTS_WORDS, H, HEAD_DIM, HEAD_RING_ROWS_DYN, KV_RING_ROWS_DYN,
    O_GROUP_IN, ORIGINAL_COLUMNS_DYN, ORIGINAL_ROWS_DYN, PACK_ROWS_DYN,
    PLAN_CAP_DYN, QUERY_BOUNDS_DYN, REQUESTS_DYN, ROPE_CAP_DYN, ROPE_DIM,
    SPARSE_BLOCKS, T_DYN, attention_vector_init, attention_vector_tick,
)
from .indexer_group_stages import OFFSET_ROWS_DYN, PLAN_WORDS
from .indexer_score_stages import (
    CACHE_PAGES_DYN, PAGE_BYTES_DYN, TABLE_COLUMNS_DYN, TOPK_PAIR_WIDTH,
    TRANSFER_COLS_DYN, TRANSFER_ROWS_DYN,
)
from .indexer_topk_stages import PAIR_ROWS_DYN
from .interleave_control import next_score_leaf, score_wave_step_count
from .score_vector_steps_g6 import (
    score_vector_leaf_publish_g6, score_vector_leaf_state_g6,
    score_vector_step_g6,
)


@pl.jit.inline(auto_scope=False)
def interleave_vector_wave_g6(
    idx_native_kv_cache: pl.Tensor[[CACHE_PAGES_DYN, PAGE_BYTES_DYN], pl.INT8],
    idx_block_table: pl.Tensor[[REQUESTS_DYN, TABLE_COLUMNS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    group_offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    buf_score_transfer: pl.Tensor[[TRANSFER_ROWS_DYN, TRANSFER_COLS_DYN], pl.FP32],
    pair_arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, TOPK_PAIR_WIDTH], pl.FP32]],
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
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
    score_group_begin: pl.Scalar[pl.INDEX],
    score_group_end: pl.Scalar[pl.INDEX],
    attention_query_begin: pl.Scalar[pl.INDEX],
    attention_query_end: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    lane: pl.Scalar[pl.INDEX],
    SCORE_READY_EVENT: pl.constexpr = 0,
    SCORE_CONSUMED_EVENT: pl.constexpr = 1,
    ATTENTION_EVENT_BASE: pl.constexpr = 2,
    CACHE_QUERY_METADATA: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
    SCALE_PAGE_COALESCING: pl.constexpr = False,
):
    """完整消费本核本波，返回六份GM别名及真实root writer标记。

    参数worker是物理核组编号，lane为双AIV的0/1。metadata在整个wave期间
    不得改变；本helper不重新验证外层已获取的plan/页表发布代际。
    """
    # 保留调用级初始化的原数学；忽略原全T块数，下面按wave独立求出块数。
    # t_dim仍是真实完整T，尤其不能把短wave误送入EARLY_KV的小T分支。
    (
        t_dim, physical_core, full_block_count, lane_head, lane_kv, swap,
        initial_m, initial_l, initial_left, initial_right,
    ) = attention_vector_init(position_ids, attn_sink_col, ffts_workspace, worker, lane, 24)
    query_first = attention_query_begin + ((worker - attention_query_begin) % 24 + 24) % 24
    query_count = pl.max((attention_query_end - query_first + 23) // 24, 0)
    attention_block_count = query_count * SPARSE_BLOCKS
    # 每个owner都从relative tick0开始；末三拍是原softmax/PV归并排空。
    attention_ticks = attention_block_count + 3
    score_steps = score_wave_step_count(
        bounds, lengths, position_ids, group_offsets, plan,
        worker, score_group_begin, score_group_end,
    )
    (
        initial_next_item, initial_found, initial_group, initial_leaf, initial_batch,
        initial_query, initial_live, initial_cache_len, initial_begin, initial_valid,
        initial_steps, initial_lane_span,
    ) = next_score_leaf(
        bounds, lengths, position_ids, group_offsets, plan,
        worker, score_group_begin, score_group_end,
    )
    # full-valid句柄不是已初始化数值。只有score_steps>0的首个真实step会读
    # 根状态；它在step==0分支先填NEG_INF。空owner不会读取这两个句柄。
    initial_roots = pl.create_tile([3, 1024], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    initial_segments = pl.create_tile([6, 1024], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    # 同当前完整Attention：每个lane独占三槽统计环；首个有效softmax才写入，
    # 末尾三拍已排空后才能切下一wave。无效块始终不读取未写统计。
    initial_stats = pl.create_tile([QK_TRANSFER_SLOTS * 2, H // 2], dtype=pl.FP32)
    physical_mode = physical_plan_supported(
        pl.tensor.dim(cmp_block_table, 1), pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE,
        PLAN_PHYSICAL_ROWS,
    )
    initial_step = pl.cast(0, pl.INDEX)
    initial_root_wrote = pl.cast(0, pl.INDEX)
    for round_id, (
        m_iter, l_iter, left_iter, right_iter, stats_iter, roots_iter, segments_iter,
        next_item_iter, leaf_iter, batch_iter, query_iter, live_iter, cache_len_iter,
        begin_iter, valid_iter, steps_iter, step_iter, wrote_iter,
    ) in pl.range(
        pl.max(score_steps, attention_ticks),
        init_values=(
            initial_m, initial_l, initial_left, initial_right, initial_stats, initial_roots, initial_segments,
            initial_next_item, initial_leaf, initial_batch, initial_query, initial_live,
            initial_cache_len, initial_begin, initial_valid, initial_steps, initial_step, initial_root_wrote,
        ),
    ):
        # AIV必须先推进前波Attention；同组AIC顺序相反。此处只改变完整
        # step/tick之间的服务顺序，不拆掉原gather/softmax/PV的事件与数学。
        # 外层IterArg直接进入含unroll的helper，会触发当前UnrollLoops对外部
        # IterArg的错误深拷贝。先建立普通Scalar Var桥，再传入所有嵌套helper；
        # 这里是整数别名，不增加GM metadata读写，不改变步骤/所有权。
        score_next_item_value = next_item_iter
        score_leaf_value = leaf_iter
        score_batch_value = batch_iter
        score_query_value = query_iter
        score_live_value = live_iter
        score_cache_len_value = cache_len_iter
        score_begin_value = begin_iter
        score_valid_value = valid_iter
        score_steps_value = steps_iter
        score_step_value = step_iter
        score_wrote_value = wrote_iter
        # GM返回别名用新局部名接住；原InOut存储副作用不变，跨分支只显式
        # yield五个Attention Tile（含统计环），避免隐式Tensor phi与显式Tile phi混合。
        if round_id < attention_ticks:
            (tick_kv_alias, tick_probability_alias, tick_mi_alias, tick_li_alias, tick_output_alias, tick_m, tick_l, tick_left, tick_right, tick_stats) = attention_vector_tick(
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
                round_id,
                attention_block_count,
                physical_core,
                query_first,
                lane_head,
                lane_kv,
                t_dim,
                swap,
                m_iter,
                l_iter,
                left_iter,
                right_iter,
                stats_iter,
                physical_mode,
                24,
                ATTENTION_EVENT_BASE,
                CACHE_QUERY_METADATA,
                LOCAL_ATTENTION_STATS,
                PLAN_PHYSICAL_ROWS,
            )
            m_after, l_after, left_after, right_after, stats_after = pl.yield_(tick_m, tick_l, tick_left, tick_right, tick_stats)
        else:
            m_after, l_after, left_after, right_after, stats_after = pl.yield_(m_iter, l_iter, left_iter, right_iter, stats_iter)

        if round_id < score_steps:
            # 每叶只在首step初始化；偶数step留下的segment一直保留到下一step。
            # metadata只在换叶时扫描，不在每个step重复查找本核leaf。
            if score_step_value == 0:
                fresh_roots, fresh_segments = score_vector_leaf_state_g6()
                leaf_roots, leaf_segments = pl.yield_(fresh_roots, fresh_segments)
            else:
                leaf_roots, leaf_segments = pl.yield_(roots_iter, segments_iter)
            (step_roots, step_segments) = score_vector_step_g6(
                idx_native_kv_cache,
                idx_block_table,
                position_ids,
                lengths,
                bounds,
                group_offsets,
                buf_score_transfer,
                leaf_roots,
                leaf_segments,
                score_begin_value,
                score_step_value,
                score_steps_value,
                score_cache_len_value,
                score_batch_value,
                score_query_value,
                worker,
                lane,
                score_live_value,
                score_valid_value,
                SCORE_READY_EVENT,
                SCORE_CONSUMED_EVENT,
                SCALE_PAGE_COALESCING,
            )
            if score_step_value + 1 == score_steps_value:
                # 两AIV每叶都已消费完同数FIFO块；有真实query的lane才写根。
                # ACK早于排序/根写回，不能拿ACK或AIC叶末drain代替本次store。
                leaf_pair_alias = score_vector_leaf_publish_g6(
                    step_roots, pair_arena, lane, score_live_value, score_query_value, score_leaf_value
                )
                leaf_wrote = pl.cast(score_live_value > lane * 3, pl.INDEX)
                (
                    following_item, following_found, following_group, following_leaf,
                    following_batch, following_query, following_live, following_cache_len,
                    following_begin, following_valid, following_steps, following_lane_span,
                ) = next_score_leaf(
                    bounds, lengths, position_ids, group_offsets, plan,
                    score_next_item_value, score_group_begin, score_group_end,
                )
                (
                    next_item_after, leaf_after, batch_after, query_after, live_after,
                    cache_len_after, begin_after, valid_after, steps_after, step_after, wrote_after,
                ) = pl.yield_(
                    following_item, following_leaf, following_batch, following_query,
                    following_live, following_cache_len, following_begin, following_valid, following_steps,
                    pl.cast(0, pl.INDEX), pl.max(score_wrote_value, leaf_wrote),
                )
            else:
                (
                    next_item_after, leaf_after, batch_after, query_after, live_after,
                    cache_len_after, begin_after, valid_after, steps_after, step_after, wrote_after,
                ) = pl.yield_(
                    score_next_item_value, score_leaf_value, score_batch_value, score_query_value, score_live_value,
                    score_cache_len_value, score_begin_value, score_valid_value, score_steps_value, score_step_value + 1, score_wrote_value,
                )
            (
                roots_after, segments_after, cursor_after, current_leaf_after,
                current_batch_after, current_query_after, current_live_after,
                current_cache_after, current_begin_after, current_valid_after, current_steps_after,
                current_step_after, current_wrote_after,
            ) = pl.yield_(
                step_roots, step_segments, next_item_after, leaf_after, batch_after,
                query_after, live_after, cache_len_after, begin_after, valid_after, steps_after,
                step_after, wrote_after,
            )
        else:
            (
                roots_after, segments_after, cursor_after, current_leaf_after,
                current_batch_after, current_query_after, current_live_after,
                current_cache_after, current_begin_after, current_valid_after, current_steps_after,
                current_step_after, current_wrote_after,
            ) = pl.yield_(
                roots_iter, segments_iter, score_next_item_value, score_leaf_value, score_batch_value,
                score_query_value, score_live_value, score_cache_len_value, score_begin_value, score_valid_value, score_steps_value,
                score_step_value, score_wrote_value,
            )
        (
            final_m, final_l, final_left, final_right, final_stats, final_roots, final_segments,
            final_cursor, final_leaf, final_batch, final_query, final_live,
            final_cache, final_begin, final_valid, final_steps, final_step, root_wrote,
        ) = pl.yield_(
            m_after, l_after, left_after, right_after, stats_after, roots_after, segments_after,
            cursor_after, current_leaf_after, current_batch_after, current_query_after,
            current_live_after, current_cache_after, current_begin_after, current_valid_after, current_steps_after,
            current_step_after, current_wrote_after,
        )
    # 这里只返回writer事实；外层完成原业务release/cache维护后再发wave ROOT。
    # lane_span/valid/found由统一control计算，AIV数学用steps/cache_len/逐query
    # visibility；本版本另携带统一control返回的valid，用于连续scale尾页，不另算长度。
    return (
        pair_arena, kv_transfer, probability_transfer, mi_transfer, li_transfer,
        o_packed_heads, root_wrote,
    )
