# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""动态CSA Attention完整tick的typed-Tile接口：仅源码准备，尚未编译验证。

整体思路、阅读顺序与当前边界
    attention_cube_init / attention_vector_init保留原初始化数学与set_ffts，
    返回真实T、当前核的全局query基址/块数及本核跨tick状态；每次调用生命周期
    只初始化一次。随后调用attention_cube_tick / attention_vector_tick推进一拍。
    本模块直接抽取已冻结的两L1候选，没有实现Score交错、wave控制或新调度。
    typed-Tile参数/返回依赖尚未验证的JIT能力；源码AST相同不等于可lowering、
    设备精度/协议/资源已通过。不得作为已接入完整CSA的生产入口。

输入、输出与动态范围
    GM布局沿原动态CSA：64头/512维，原KV与压缩KV按Native device页表读取，
    Q/L/positions/mask仍是device Tensor；业务容量轴保持pl.dynamic。
    每组query公式为query_first+(work//5)*CORE_GROUPS，保持全局q%24所有权；
    query_first是当前wave的首个本核query，qk_core仅为物理核0..23。五处GM
    ring仍使用qk_core*3+relative_work%3，不能用全局query代替物理环槽。
    两init保留调用级几何/Tile初始化，wave执行器覆盖block_count并传query_first。
    tick、block_count、query_first、qk_core及两AIV lane偏移显式为Scalar；AIV的t_dim必须
    是真实全局position_ids.shape[0]，不可替换为wave长度或pad容量。
    EARLY_KV的T>=144分支因此保持原义。固定Tile尺寸来自模型/数学块，不冻结T/B。
    当前只传递full-valid数学Tile，不把valid_shape=min(T,...)的Tile跨步传入。
    full-valid仅指形状metadata；KV未填充槽的数据仍不可读取，必须沿原mask/槽时序。
    GM参数沿原Tensor/InOut注解，并显式返回原输出Tensor，保留原数据流；
    Tile参数显式返回更新句柄，不把原地缓冲语义误当成Python赋值已自动回传。

跨步状态与两个角色
    AIC持Q BF16[64,512] Mat和KV BF16[256,512] Mat，后者内部两槽；GM所有
    环仍三槽。本拍先PV(tick-2)后QK(tick)，旧KV槽实际MTE1读取完成才可覆盖。
    AIV持FP32 m/l[32,1]、left/right[32,256]四Tile；只读swap索引INT32
    [1,1024]由init一次生成。每拍gather本块、softmax上一块、merge三拍前PV，
    最后块归一化/逆RoPE/分组BF16输出并按原数学重置四Tile；swap不重置。
    本核临时Tile仍留在完整tick内，不扩成长期驻留状态；不引入新的GM工作区。

阶段/事件与尾部调用合同
    AIC调用tick范围[0,block_count+2)，AIV为[0,block_count+3)，正好保留填充
    和排空。调用方不能多推进一拍、跳过有效拍或省掉尾拍；空组仍遵守原边界。
    同组AIC与双AIV必须并发推进，FFTS基址、EVENT_BASE与原mask一致；每块
    READY/wait条件和pipe不改，普通phase通知不能代替槽协议。首query/新query
    的Q加载、无效块跳过、末块状态重置均在原位置，没有新增无条件清零。
    GM环%3与L1环%2独立；两helper源码局部顺序不是异步完成证据。后端仍须
    证明MTE1→MTE2覆盖依赖、Mat回边/布局与UB峰值；现有编译产物不能代替本模块。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FLASH as M
from .attention_plan_stages import (
    COMPRESSED_COLUMNS_DYN,
    ORIGINAL_COLUMNS_DYN,
    PLAN_CAP_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    ROPE_CAP_DYN,
    T_DYN,
    attention_query_geometry,
)

Q_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_Q_ROWS")
ORIGINAL_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_ORIGINAL_ROWS")
COMPRESSED_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_COMPRESSED_ROWS")
KV_RING_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_KV_RING_ROWS")
HEAD_RING_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_HEAD_RING_ROWS")
PACK_ROWS_DYN = pl.dynamic("CSA_ATTN_STREAM_PACK_ROWS")
FFTS_WORDS = 256  # 硬件控制区常量，当前set_ffts API要求静态Tensor。
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 448
WIN = 128
CMP_TOPK = 512
BLOCK_SIZE = 32
ATTN_K_TILE = 128
SPARSE_BLOCKS = 5
QK_PRE_LAUNCH = 2
QK_TRANSFER_SLOTS = 3  # 所有GM ring仍三槽；不得与AIC内部L1槽数合并。
QK_L1_SLOTS = 2  # 本候选仅AIC KV Mat双槽，依赖每tick先PV后QK。
PV_N_TILE = 128
SWA_RUNS = 5
FINAL_HEAD_TILE = 16
O_GROUPS = 8
HEADS_PER_GROUP = 8
O_GROUP_IN = 4096
PUBLISH_GROUPS = 2
EARLY_KV_MIN_TOKENS = 144
SOFTMAX_SCALE = M.softmax_scale
QK_KV_READY_EVENT = 0
QK_SCORE_READY_EVENT = 1
QK_PROB_READY_EVENT = 2
QK_PV_READY_EVENT = 3


@pl.jit.inline(auto_scope=False)
def attention_cube_init(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
) -> tuple[
    pl.Scalar[pl.INDEX],
    pl.Scalar[pl.INDEX],
    pl.Scalar[pl.INDEX],
    pl.Tile[[H, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat],
    pl.Tile[[QK_L1_SLOTS * ATTN_K_TILE, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat],
]:
    """初始化AIC跨拍状态；T与block_count沿原全局query公式，FFTS仅此时声明。
        返回顺序为T/core/block_count/Q Tile/KV双槽Tile；调用者保存后逐拍传回。"""
    t_dim = pl.tensor.dim(q_flat, 0) // H
    qk_core = core_id
    # 固定硬件控制区声明，不搬运业务Tensor；最终Simpler包装使用运行时FFTS基址。
    pl.system.set_ffts(ffts_workspace)
    qk_query_count = (t_dim + CORE_GROUPS - 1 - qk_core) // CORE_GROUPS
    qk_block_count = qk_query_count * SPARSE_BLOCKS
    qk_q = pl.create_tile([H, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
    # KV仍只进一次L1：QK用转置view，之后PV用同一份行切片。
    # 本候选只缩AIC内部同dtype Mat，不改GM环，不做混类型槽别名。
    # 原来三槽允许先QK后PV；双槽必须先消费旧PV，再覆盖同一槽。
    qk_l1 = pl.create_tile([QK_L1_SLOTS * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16, target_memory=pl.MemorySpace.Mat)
    # 循环界仍为真实query块数+2：首两拍填充、末两拍排空；空组不读环槽。
    # mask无效则该块两端均跳过事件/访存，不能为了双槽无条件读取旧KV。

    return t_dim, qk_core, qk_block_count, qk_q, qk_l1


@pl.jit.inline(auto_scope=False)
def attention_cube_tick(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    qk_tick: pl.Scalar[pl.INDEX],
    qk_block_count: pl.Scalar[pl.INDEX],
    qk_core: pl.Scalar[pl.INDEX],
    query_first: pl.Scalar[pl.INDEX],
    qk_q: pl.Tile[[H, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat],
    qk_l1: pl.Tile[[QK_L1_SLOTS * ATTN_K_TILE, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
) -> tuple[
    pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32],
    pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32],
    pl.Tile[[H, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat],
    pl.Tile[[QK_L1_SLOTS * ATTN_K_TILE, HEAD_DIM], pl.BF16, pl.MemorySpace.Mat],
]:
    """执行完整AIC一拍；PV旧槽读取在前，QK新槽写入在后。
        Q/双槽Tile必须来自同一次init/上一tick；返回后不能借Scalar完成推断DMA完成。
        tick从当前wave的0开始；query_first指定全局query首项，qk_core仍为物理核。
        tick边界由调用者保证，真实mask控制两端事件，不增加入口裁剪改变原语义。"""
    # Inline源码规避：实参可能是tuple返回值的TupleGetItem，不能直接重绑定
    # 形参作为赋值LHS。先建立fresh局部Tile别名，后续数学与返回只用局部名；
    # 此处不创建/复制Tile，不改数学、形状、事件或Tensor参数。联合lowering仍待验证。
    qk_q_local = qk_q
    qk_l1_local = qk_l1
    # 先消耗两拍前的L1 KV：r与r-2同属一个%2槽，必须先PV后覆盖。
    # 本段的GM概率/PV地址仍按%3，FFTS的PROB/PV事件沿用原条件。
    if qk_tick >= QK_PRE_LAUNCH:
        pv_work = qk_tick - QK_PRE_LAUNCH
        pv_sb = pv_work % SPARSE_BLOCKS
        pv_t = query_first + (pv_work // SPARSE_BLOCKS) * CORE_GROUPS
        if pl.read(valid_block_mask, [pv_t, pv_sb]) > 0:
            pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS
            pv_transfer_row = pv_slot * H
            pl.system.sync_wait(
                EVENT_BASE + QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC
            )
            pv_probability = pl.load(
                probability_transfer,
                [pv_transfer_row, 0],
                [H, ATTN_K_TILE],
                target_memory=pl.MemorySpace.Mat,
            )
            pv_l1_row = (pv_work % QK_L1_SLOTS) * ATTN_K_TILE
            # PV沿用N128/K128，保留每128候选一次softmax和query流水顺序。
            pv_probability_left = pl.tile.move(pv_probability, target_memory=pl.MemorySpace.Left)
            pv_first_right = pl.tile.extract(
                qk_l1_local, pv_l1_row, 0, [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right
            )
            pv_previous = pl.tile.matmul(pv_probability_left, pv_first_right)
            for pv_n in pl.unroll(1, HEAD_DIM // PV_N_TILE):
                pv_next_right = pl.tile.extract(
                    qk_l1_local,
                    pv_l1_row,
                    pv_n * PV_N_TILE,
                    [ATTN_K_TILE, PV_N_TILE],
                    target_memory=pl.MemorySpace.Right,
                )
                # 同时保留前后两个累加器，让前块FIX写回与后块Cube重叠。
                pv_current = pl.tile.matmul(pv_probability_left, pv_next_right)
                pl.store(pv_previous, [pv_transfer_row, (pv_n - 1) * PV_N_TILE], pv_transfer)
                pv_previous = pv_current
            pl.store(pv_previous, [pv_transfer_row, HEAD_DIM - PV_N_TILE], pv_transfer)
            pl.system.sync_set(
                EVENT_BASE + QK_PV_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
            )

    # 再写入本拍KV并做QK。这里的源级先后只建立所需数据顺序；
    # 后端仍须在最后PV MTE1读取与新KV MTE2覆盖间生成正确依赖。
    # 不能把Scalar走过PV语句当作异步搬运已经完成。
    if qk_tick < qk_block_count:
        qk_sb = qk_tick % SPARSE_BLOCKS
        qk_t = query_first + (qk_tick // SPARSE_BLOCKS) * CORE_GROUPS
        if qk_sb == 0:
            qk_q_local = pl.gather_row(qk_q_local, q_flat, [0, 0], [qk_t * H, 0], [H, HEAD_DIM])
        if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS
            qk_kv_row = qk_slot * ATTN_K_TILE
            qk_transfer_row = qk_slot * H
            pl.system.sync_wait(EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
            qk_l1_row = (qk_tick % QK_L1_SLOTS) * ATTN_K_TILE
            qk_l1_local = pl.gather_row(qk_l1_local, kv_transfer, [qk_l1_row, 0], [qk_kv_row, 0], [ATTN_K_TILE, HEAD_DIM])
            qk_l1_t = pl.tile.transpose_view(qk_l1_local)
            qk_kv_t = pl.tile.slice(qk_l1_t, [HEAD_DIM, ATTN_K_TILE], [0, qk_l1_row])
            qk_scores = pl.matmul(qk_q_local, qk_kv_t, out_dtype=pl.FP32)
            pl.store(qk_scores, [qk_transfer_row, 0], score_transfer)
            pl.system.sync_set(
                EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC
            )

    return score_transfer, pv_transfer, qk_q_local, qk_l1_local


@pl.jit.inline(auto_scope=False)
def attention_vector_init(
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
    attn_sink_col: pl.Tensor[[H, 1], pl.FP32],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    lane_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
) -> tuple[
    pl.Scalar[pl.INDEX],
    pl.Scalar[pl.INDEX],
    pl.Scalar[pl.INDEX],
    pl.Scalar[pl.INDEX],
    pl.Scalar[pl.INDEX],
    pl.Tile[[1, FINAL_HEAD_TILE * ROPE_DIM], pl.INT32, pl.MemorySpace.Vec],
    pl.Tile[[H // 2, 1], pl.FP32, pl.MemorySpace.Vec],
    pl.Tile[[H // 2, 1], pl.FP32, pl.MemorySpace.Vec],
    pl.Tile[[H // 2, HEAD_DIM // 2], pl.FP32, pl.MemorySpace.Vec],
    pl.Tile[[H // 2, HEAD_DIM // 2], pl.FP32, pl.MemorySpace.Vec],
]:
    """初始化本AIV的几何Scalar、只读swap与四份在线归约Tile。
        m从sink加载、l从零开始，sink只在原最终分母处加入；初始化数学不可替换。
        返回T/core/block_count/lane_head/lane_kv/swap/m/l/left/right。"""
    t_dim = pl.tensor.dim(position_ids, 0)
    qk_core = core_id
    qk_aiv = lane_id
    qk_query_count = (t_dim + CORE_GROUPS - 1 - qk_core) // CORE_GROUPS
    qk_block_count = qk_query_count * SPARSE_BLOCKS
    # 固定硬件控制区声明，不搬运业务Tensor；最终Simpler包装使用运行时FFTS基址。
    pl.system.set_ffts(ffts_workspace)
    qk_lane_head = qk_aiv * (H // 2)
    qk_lane_kv = qk_aiv * (ATTN_K_TILE // 2)
    m_idx = pl.tile.ci(0, [1, ROPE_DIM], dtype=pl.INT32)
    m_rem_tmp = pl.create_tile([1, ROPE_DIM], dtype=pl.INT32)
    m_lane = pl.tile.rems(m_idx, 2, m_rem_tmp)
    m_swap_row = pl.tile.adds(pl.tile.sub(m_idx, pl.tile.muls(m_lane, 2)), NOPE_DIM + 1)
    m_swap_base = pl.create_tile([FINAL_HEAD_TILE, ROPE_DIM], dtype=pl.INT32)
    m_swap_source = pl.col_expand(m_swap_base, m_swap_row)
    m_row_offsets = pl.tile.muls(pl.tile.ci(0, [1, FINAL_HEAD_TILE], dtype=pl.INT32), HEAD_DIM)
    m_swap_idx = pl.reshape(
        pl.row_expand_add(m_swap_source, pl.reshape(m_row_offsets, [FINAL_HEAD_TILE, 1])),
        [1, FINAL_HEAD_TILE * ROPE_DIM],
    )
    running_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1])
    # 上游口径：l 从 0 起算，sink 的那一项留到最终发布的分母里补。
    running_l = pl.tile.muls(running_m, 0.0)
    running_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
    running_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
    # 多走三拍排空末块归并；循环携带的m/l/oi仅属于当前query，
    # 原始t_dim决定提前KV策略，不能用pad容量代替。

    return t_dim, qk_core, qk_block_count, qk_lane_head, qk_lane_kv, m_swap_idx, running_m, running_l, running_left, running_right


@pl.jit.inline(auto_scope=False)
def attention_vector_tick(
    ori_kv_flat: pl.Tensor[[ORIGINAL_ROWS_DYN, HEAD_DIM], pl.BF16],
    cmp_kv_flat: pl.Tensor[[COMPRESSED_ROWS_DYN, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT64],
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
    qk_tick: pl.Scalar[pl.INDEX],
    qk_block_count: pl.Scalar[pl.INDEX],
    qk_core: pl.Scalar[pl.INDEX],
    query_first: pl.Scalar[pl.INDEX],
    qk_lane_head: pl.Scalar[pl.INDEX],
    qk_lane_kv: pl.Scalar[pl.INDEX],
    t_dim: pl.Scalar[pl.INDEX],
    m_swap_idx: pl.Tile[[1, FINAL_HEAD_TILE * ROPE_DIM], pl.INT32, pl.MemorySpace.Vec],
    m_iter: pl.Tile[[H // 2, 1], pl.FP32, pl.MemorySpace.Vec],
    l_iter: pl.Tile[[H // 2, 1], pl.FP32, pl.MemorySpace.Vec],
    left_iter: pl.Tile[[H // 2, HEAD_DIM // 2], pl.FP32, pl.MemorySpace.Vec],
    right_iter: pl.Tile[[H // 2, HEAD_DIM // 2], pl.FP32, pl.MemorySpace.Vec],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
    CACHE_QUERY_METADATA: pl.constexpr = False,
) -> tuple[
    pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16],
    pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32],
    pl.Tensor[[HEAD_RING_ROWS_DYN, 1], pl.FP32],
    pl.Tensor[[PACK_ROWS_DYN, O_GROUP_IN], pl.BF16],
    pl.Tile[[H // 2, 1], pl.FP32, pl.MemorySpace.Vec],
    pl.Tile[[H // 2, 1], pl.FP32, pl.MemorySpace.Vec],
    pl.Tile[[H // 2, HEAD_DIM // 2], pl.FP32, pl.MemorySpace.Vec],
    pl.Tile[[H // 2, HEAD_DIM // 2], pl.FP32, pl.MemorySpace.Vec],
]:
    """执行完整AIV一拍，并显式返回m/l/left/right四个下一拍状态。
        swap只读；原if中的pl.yield保留，只有最外loop回边改为函数返回。
        query_first只改变全局query起点，五处GM环槽仍绑定物理qk_core。
        实际全局T决定早KV分支；末块即使无效仍按原路径发布/重置，不能跳过。
        原gather/softmax/merge顺序与Scalar地址guard全部保留。"""
    qk_t = query_first + (qk_tick // SPARSE_BLOCKS) * CORE_GROUPS
    qk_sb = qk_tick % SPARSE_BLOCKS
    if qk_tick < qk_block_count:
        if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
            qk_slot = qk_core * QK_TRANSFER_SLOTS + qk_tick % QK_TRANSFER_SLOTS
            qk_kv_row = qk_slot * ATTN_K_TILE
            qk_kv_half = pl.tile.full([ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
            # mask是两角色共同事件谓词；额外地址guard只限制gather，不改变
            # 事件对数。合法调用必须使用与plan同一批Q/L/positions/页表。
            qk_b, qk_pos, q_active = attention_query_geometry(
                position_ids, bounds, lengths, valid_block_mask, qk_t, CACHE_QUERY_METADATA
            )
            if q_active != 0:
                if qk_sb == 0:
                    qk_win_len = pl.min(qk_pos + 1, WIN)
                    qk_win_start = qk_pos - qk_win_len + 1
                    qk_head = qk_win_start % BLOCK_SIZE
                    qk_rows = pl.min(pl.max(qk_win_len - qk_lane_kv, 0), ATTN_K_TILE // 2)
                    # 最多五个页run，每个lane只采自己的64候选；尾run只读
                    # qk_hi-qk_lo行，先检查页表列再检查物理page容量。
                    for qk_run in pl.unroll(SWA_RUNS):
                        qk_lo = pl.max(qk_run * BLOCK_SIZE - qk_head - qk_lane_kv, 0)
                        qk_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head - qk_lane_kv, qk_rows)
                        if qk_hi > qk_lo:
                            qk_absolute = qk_win_start + qk_lane_kv + qk_lo
                            qk_column = qk_absolute // BLOCK_SIZE
                            if qk_column >= 0 and qk_column < pl.tensor.dim(ori_block_table, 1):
                                qk_raw_page = pl.cast(pl.read(ori_block_table, [qk_b, qk_column]), pl.INDEX)
                                if (
                                    qk_raw_page >= 0
                                    and qk_raw_page < pl.tensor.dim(ori_kv_flat, 0) // BLOCK_SIZE
                                ):
                                    qk_raw_row = qk_raw_page * BLOCK_SIZE + qk_absolute % BLOCK_SIZE
                                    qk_kv_half = pl.gather_row(
                                        qk_kv_half,
                                        ori_kv_flat,
                                        [qk_lo, 0],
                                        [qk_raw_row, 0],
                                        [ATTN_K_TILE // 2, HEAD_DIM],
                                        valid_shape=[qk_hi - qk_lo, HEAD_DIM],
                                    )
                else:
                    # indices已经包含plan的causal/页mask，保留顺序并再做
                    # 地址上界保护；非法行留在本轮初始化的零KV中。
                    for qk_row in pl.range(ATTN_K_TILE // 2):
                        qk_cmp_k = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv + qk_row
                        if qk_cmp_k < CMP_TOPK:
                            qk_ridx = pl.cast(pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k]), pl.INDEX)
                            if qk_ridx >= 0:
                                qk_column = qk_ridx // BLOCK_SIZE
                                if qk_column < pl.tensor.dim(cmp_block_table, 1):
                                    qk_page = pl.cast(pl.read(cmp_block_table, [qk_b, qk_column]), pl.INDEX)
                                    if qk_page >= 0 and qk_page < pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE:
                                        qk_src = qk_page * BLOCK_SIZE + qk_ridx % BLOCK_SIZE
                                        qk_kv_half = pl.gather_row(
                                            qk_kv_half, cmp_kv_flat, [qk_row, 0], [qk_src, 0], [1, HEAD_DIM]
                                        )
            pl.store(qk_kv_half, [qk_kv_row + qk_lane_kv, 0], kv_transfer)
    # 大档先消费上一块SCORE通知，再发布已经写好的本块KV；
    # 先让Cube推进，再做上一块softmax，同时限制SCORE覆盖速度。
    if t_dim >= EARLY_KV_MIN_TOKENS:
        if qk_tick > 0 and qk_tick <= qk_block_count:
            if (
                pl.read(
                    valid_block_mask,
                    [
                        query_first + ((qk_tick - 1) // SPARSE_BLOCKS) * CORE_GROUPS,
                        (qk_tick - 1) % SPARSE_BLOCKS,
                    ],
                )
                > 0
            ):
                pl.system.sync_wait(
                    EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                )
        if qk_tick < qk_block_count:
            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                pl.system.sync_set(
                    EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                )
    if qk_tick > 0 and qk_tick <= qk_block_count:
        softmax_work = qk_tick - 1
        softmax_sb = softmax_work % SPARSE_BLOCKS
        softmax_t = query_first + (softmax_work // SPARSE_BLOCKS) * CORE_GROUPS
        if pl.read(valid_block_mask, [softmax_t, softmax_sb]) > 0:
            qk_slot = qk_core * QK_TRANSFER_SLOTS + softmax_work % QK_TRANSFER_SLOTS
            qk_transfer_row = qk_slot * H
            qk_s0 = softmax_sb * ATTN_K_TILE
            if t_dim < EARLY_KV_MIN_TOKENS:
                pl.system.sync_wait(
                    EVENT_BASE + QK_SCORE_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV
                )
            # 整条 H//2 一次做完：原先按 SOFTMAX_HEAD_TILE=8 切成四段，
            # 是 ATTN_K_TILE=512 时 UB 放不下留下的，现在 128 宽已无必要。
            # 归约临时区不跨块保存状态，限制在此阶段，让末块发布复用UB。
            qk_reduce_tmp = pl.create_tile([H // 2, ATTN_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            qk_scores_half = pl.load(score_transfer, [qk_transfer_row + qk_lane_head, 0], [H // 2, ATTN_K_TILE])
            qk_bias = pl.load(sparse_bias, [softmax_t, qk_s0], [1, ATTN_K_TILE])
            qk_masked = pl.col_expand_add(pl.mul(qk_scores_half, SOFTMAX_SCALE), qk_bias)
            # 块内取局部最大值，跨块的重标定挪到下面的归并里；
            # 这样每块 softmax 互不依赖，流水才排得开。
            qk_mi = pl.row_max(qk_masked, qk_reduce_tmp)
            qk_exp = pl.exp(pl.row_expand_sub(qk_masked, qk_mi))
            qk_li = pl.row_sum(qk_exp, qk_reduce_tmp)
            # 性能版用 rint（就近偶数），与上游一致。已封存的精度版用 round
            # 是为了复刻 Native SAS 的 CAST_ROUND——半数远离零。
            qk_probability = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
            pl.store(qk_probability, [qk_transfer_row + qk_lane_head, 0], probability_transfer)
            pl.store(qk_mi, [qk_transfer_row + qk_lane_head, 0], mi_transfer)
            pl.store(qk_li, [qk_transfer_row + qk_lane_head, 0], li_transfer)
            pl.system.sync_set(
                EVENT_BASE + QK_PROB_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
            )
    if t_dim < EARLY_KV_MIN_TOKENS:
        if qk_tick < qk_block_count:
            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                pl.system.sync_set(
                    EVENT_BASE + QK_KV_READY_EVENT, pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV
                )
    if qk_tick >= QK_PRE_LAUNCH + 1:
        pv_work = qk_tick - QK_PRE_LAUNCH - 1
        pv_sb = pv_work % SPARSE_BLOCKS
        pv_t = query_first + (pv_work // SPARSE_BLOCKS) * CORE_GROUPS
        if pl.read(valid_block_mask, [pv_t, pv_sb]) > 0:
            pv_slot = qk_core * QK_TRANSFER_SLOTS + pv_work % QK_TRANSFER_SLOTS
            pv_transfer_row = pv_slot * H
            pl.system.sync_wait(EVENT_BASE + QK_PV_READY_EVENT, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
            pv_m = pl.load(mi_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1])
            pv_l = pl.load(li_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, 1])
            next_m = pl.maximum(m_iter, pv_m)
            alpha = pl.exp(pl.sub(m_iter, next_m))
            beta = pl.exp(pl.sub(pv_m, next_m))
            next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, pv_l))
            pv_left = pl.load(pv_transfer, [pv_transfer_row + qk_lane_head, 0], [H // 2, HEAD_DIM // 2])
            next_left = pl.add(pl.row_expand_mul(left_iter, alpha), pl.row_expand_mul(pv_left, beta))
            pv_right = pl.load(
                pv_transfer, [pv_transfer_row + qk_lane_head, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2]
            )
            next_right = pl.add(pl.row_expand_mul(right_iter, alpha), pl.row_expand_mul(pv_right, beta))
            m_valid, l_valid, left_valid, right_valid = pl.yield_(next_m, next_l, next_left, next_right)
        else:
            m_valid, l_valid, left_valid, right_valid = pl.yield_(m_iter, l_iter, left_iter, right_iter)
        if pv_sb == SPARSE_BLOCKS - 1:
            # 末块直接消费本核live累加器；维持PTO数学，不把整T的FP32
            # mi/li/oi写GM再交另一个task归并。失效末块也必须发布/重置。
            packed_stride = pl.tensor.dim(o_packed_heads, 0) // O_GROUPS
            m_gather_tmp = pl.create_tile([1, FINAL_HEAD_TILE * ROPE_DIM], dtype=pl.INT32)
            # 每lane拆两段16头发布，控制UB峰值；每段对应两组4096列。
            for pub_h in pl.unroll((H // 2) // FINAL_HEAD_TILE):
                pub_local_head = pub_h * FINAL_HEAD_TILE
                m_h0 = qk_lane_head + pub_local_head
                # 先将整列reshape成行，再显式ND extract，保留后16头偏移。
                # 直接slice再列转行曾丢偏移；A3不支持ColMajor TEXTRACT。
                m_mi = pl.reshape(
                    pl.tile.extract(
                        pl.reshape(m_valid, [1, H // 2]),
                        0,
                        pub_local_head,
                        [1, FINAL_HEAD_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    ),
                    [FINAL_HEAD_TILE, 1],
                )
                m_li = pl.reshape(
                    pl.tile.extract(
                        pl.reshape(l_valid, [1, H // 2]),
                        0,
                        pub_local_head,
                        [1, FINAL_HEAD_TILE],
                        target_memory=pl.MemorySpace.Vec,
                    ),
                    [FINAL_HEAD_TILE, 1],
                )
                m_left = pl.slice(left_valid, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0])
                m_right = pl.slice(right_valid, [FINAL_HEAD_TILE, HEAD_DIM // 2], [pub_local_head, 0])
                m_oi = pl.concat(m_left, m_right)
                n_sink_bias = pl.load(attn_sink_col, [m_h0, 0], [FINAL_HEAD_TILE, 1])
                n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
                n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
                n_full = pl.row_expand_div(m_oi, n_denom)
                n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")
                m_rope = n_full[0:FINAL_HEAD_TILE, NOPE_DIM:HEAD_DIM]
                m_cos_il = pl.load(freqs_cos, [pv_t, 0], [1, ROPE_DIM])
                m_sin_signed = pl.load(rope_sin_signed, [pv_t, 0], [1, ROPE_DIM])
                # 索引已含各头的绝对行偏移，一次连续Gather避免逐行向量barrier。
                m_swapped_flat = pl.tile.gather(
                    pl.reshape(n_full, [1, FINAL_HEAD_TILE * HEAD_DIM]),
                    m_swap_idx,
                    m_gather_tmp,
                )
                m_swapped = pl.reshape(m_swapped_flat, [FINAL_HEAD_TILE, ROPE_DIM])
                m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
                n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
                n_full_bf16 = pl.concat(n_bf16[0:FINAL_HEAD_TILE, 0:NOPE_DIM], n_rope_bf16)
                n_group_bf16 = pl.reshape(n_full_bf16, [PUBLISH_GROUPS, O_GROUP_IN])
                n_pack_first = n_group_bf16[0:1, 0:O_GROUP_IN]
                n_pack_second = n_group_bf16[1:2, 0:O_GROUP_IN]
                n_pack_row = (m_h0 // HEADS_PER_GROUP) * packed_stride + pv_t
                n_pack_row_second = n_pack_row + packed_stride
                pl.store(n_pack_first, [n_pack_row, 0], o_packed_heads)
                pl.store(n_pack_second, [n_pack_row_second, 0], o_packed_heads)
            # 只重置刚完成query的在线数值状态；后续query的KV/QK/softmax
            # 仍可在飞行，不重置三槽或事件。
            reset_m = pl.load(attn_sink_col, [qk_lane_head, 0], [H // 2, 1])
            reset_l = pl.tile.muls(reset_m, 0.0)
            reset_left = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            reset_right = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            m_published, l_published, left_published, right_published = pl.yield_(
                reset_m, reset_l, reset_left, reset_right
            )
        else:
            m_published, l_published, left_published, right_published = pl.yield_(
                m_valid, l_valid, left_valid, right_valid
            )
        m_after, l_after, left_after, right_after = pl.yield_(
            m_published, l_published, left_published, right_published
        )
    else:
        m_after, l_after, left_after, right_after = pl.yield_(m_iter, l_iter, left_iter, right_iter)

    return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads, m_after, l_after, left_after, right_after


