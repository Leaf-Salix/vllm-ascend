# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Attention全局尺度独立数值候选，尚未验收或采用。

整体思路：原device metadata及Native KV采集保持；每核组GM保留五块，
AIC完成五次QK，双AIV各分两轮处理16头，以40KiB UB保存五块score。统一最大值
包含sink；以统一尺度生成BF16 P，在四个独立L0C片段里按原K块顺序
累加，最后仅交接一次完整PV。L1保留三块，两块从已有GM KV重读。
不使用A5的UB到L1直连，不新增Host tiling，不固定batch或历史长度。

正确性边界：概率BF16舍入、分母归约、分子累加位置会改变，用户已允许
独立验证，但不能自动放宽冻结门槛或宣称token一致。原采页、因果与
最终sink/逆RoPE/packed发布代码来自冻结正式源。

同步：每query五槽数据独立，SCORE必须wait后才可继续发布下个KV，
避免同一个FFTS事件重复set。PROB/PV每query各一次，含空query，
两侧一致；空query不读未初始化PV。各核组独立推进，无全核集合。
原跨query QK/PV重叠不保留，损失必须计入完整CSA L1而非只看流量。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FLASH as M
from ..deepseek_v4_flash_csa.config import FP32_NEG_INF
from .attention_plan_stages import (
    COMPRESSED_COLUMNS_DYN,
    ORIGINAL_COLUMNS_DYN,
    PLAN_CAP_DYN,
    QUERY_BOUNDS_DYN,
    REQUESTS_DYN,
    ROPE_CAP_DYN,
    T_DYN,
    attention_query_geometry,
    physical_plan_supported,
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
QK_TRANSFER_SLOTS = 3
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
def attention_qk_pv_cube_stage(
    q_flat: pl.Tensor[[Q_ROWS_DYN, HEAD_DIM], pl.BF16],
    valid_block_mask: pl.Tensor[[PLAN_CAP_DYN, 16], pl.INT32],
    kv_transfer: pl.Tensor[[KV_RING_ROWS_DYN, HEAD_DIM], pl.BF16],
    score_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.FP32]],
    probability_transfer: pl.Tensor[[HEAD_RING_ROWS_DYN, ATTN_K_TILE], pl.BF16],
    pv_transfer: pl.InOut[pl.Tensor[[HEAD_RING_ROWS_DYN, HEAD_DIM], pl.FP32]],
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
):

    """五块 QK 后，以统一尺度的 BF16 P 在四份 L0C 中跨块累加。"""
    pl.system.set_ffts(ffts_workspace)
    t_dim = pl.tensor.dim(q_flat, 0) // H
    # L1 三槽：0/1保留前两块，2在QK阶段流经后面三块；GM保留全部五块。
    # P按K128载入，16KiB即可；不申请全P的80KiB，更不假设640KiB KV能全驻L1。
    for qk_t in pl.range(core_id, t_dim, CORE_GROUPS):
        qk_l1 = pl.create_tile([3 * ATTN_K_TILE, HEAD_DIM], dtype=pl.BF16,
                               target_memory=pl.MemorySpace.Mat)
        qk_q = pl.load(q_flat, [qk_t * H, 0], [H, HEAD_DIM],
                      target_memory=pl.MemorySpace.Mat)
        last_stream_block = pl.cast(-1, pl.INDEX)
        for qk_sb in pl.range(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                qk_slot = core_id * SPARSE_BLOCKS + qk_sb
                qk_l1_slot = pl.min(qk_sb, 2)
                pl.system.sync_wait(EVENT_BASE + QK_KV_READY_EVENT,
                    pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
                qk_l1 = pl.gather_row(qk_l1, kv_transfer,
                    [qk_l1_slot * ATTN_K_TILE, 0], [qk_slot * ATTN_K_TILE, 0],
                    [ATTN_K_TILE, HEAD_DIM])
                qk_kv_t = pl.tile.slice(pl.tile.transpose_view(qk_l1),
                    [HEAD_DIM, ATTN_K_TILE], [0, qk_l1_slot * ATTN_K_TILE])
                qk_scores = pl.matmul(qk_q, qk_kv_t, out_dtype=pl.FP32)
                pl.store(qk_scores, [qk_slot * H, 0], score_transfer)
                pl.system.sync_set(EVENT_BASE + QK_SCORE_READY_EVENT,
                    pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
                if qk_sb >= 2:
                    last_stream_block = qk_sb
        # 两AIV的五块P都发布后才开始PV。空query也进行这一次交接，
        # 两侧协议一致；空query不读取未定义PV，也不访问非法KV页。
        pl.system.sync_wait(EVENT_BASE + QK_PROB_READY_EVENT,
            pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIC)
        # 四个独立64x128 Acc共128KiB；避免把CZ父Tile的行切片误作连续子Acc。
        acc0 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        acc1 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        acc2 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        acc3 = pl.create_tile([H, PV_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        started = pl.cast(0, pl.INT32)
        for pv_sb in pl.range(SPARSE_BLOCKS):
            if pl.read(valid_block_mask, [qk_t, pv_sb]) > 0:
                pv_slot = core_id * SPARSE_BLOCKS + pv_sb
                pv_l1_slot = pv_sb
                if pv_sb >= 2:
                    pv_l1_slot = pl.cast(2, pl.INDEX)
                    if pv_sb != last_stream_block:
                        # 原块0/1已消费，复用其槽重读块2/3。只从已有GM KV读，
                        # 不重复Native页表查找或AIV gather。尾部无效块按真实mask跳过。
                        pv_l1_slot = (pv_sb - 2) % 2
                        qk_l1 = pl.gather_row(qk_l1, kv_transfer,
                            [pv_l1_slot * ATTN_K_TILE, 0], [pv_slot * ATTN_K_TILE, 0],
                            [ATTN_K_TILE, HEAD_DIM])
                pv_p = pl.load(probability_transfer, [pv_slot * H, 0], [H, ATTN_K_TILE],
                               target_memory=pl.MemorySpace.Mat)
                pv_left = pl.tile.move(pv_p, target_memory=pl.MemorySpace.Left)
                # 保留原K128/N128数学tile；首个有效块初始化，随后续加。
                # 真实双缓冲/等待由生成代码审查，不预称与旧流水完全相同。
                right0 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, 0,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc0 = pl.tile.matmul_acc(acc0, pv_left, right0, init_cond=started == 0)
                right1 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, PV_N_TILE,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc1 = pl.tile.matmul_acc(acc1, pv_left, right1, init_cond=started == 0)
                right2 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, 2 * PV_N_TILE,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc2 = pl.tile.matmul_acc(acc2, pv_left, right2, init_cond=started == 0)
                right3 = pl.tile.extract(qk_l1, pv_l1_slot * ATTN_K_TILE, 3 * PV_N_TILE,
                    [ATTN_K_TILE, PV_N_TILE], target_memory=pl.MemorySpace.Right)
                acc3 = pl.tile.matmul_acc(acc3, pv_left, right3, init_cond=started == 0)
                started = pl.cast(1, pl.INT32)
        if started != 0:
            # 每组只使用自己的第0个PV槽；ABI容量暂与其它head ring保持一致。
            out_row = core_id * SPARSE_BLOCKS * H
            pl.store(acc0, [out_row, 0], pv_transfer)
            pl.store(acc1, [out_row, PV_N_TILE], pv_transfer)
            pl.store(acc2, [out_row, 2 * PV_N_TILE], pv_transfer)
            pl.store(acc3, [out_row, 3 * PV_N_TILE], pv_transfer)
        pl.system.sync_set(EVENT_BASE + QK_PV_READY_EVENT,
            pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
    return score_transfer, pv_transfer


@pl.jit.inline(auto_scope=False)
def attention_qk_pv_vector_lane_stage(
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
    ffts_workspace: pl.Tensor[[FFTS_WORDS], pl.INT64],
    core_id: pl.Scalar[pl.INDEX],
    lane_id: pl.Scalar[pl.INDEX],
    CORE_GROUPS: pl.constexpr,
    EVENT_BASE: pl.constexpr = 0,
    CACHE_QUERY_METADATA: pl.constexpr = False,
    LOCAL_ATTENTION_STATS: pl.constexpr = False,
    PLAN_PHYSICAL_ROWS: pl.constexpr = False,
):
    t_dim = pl.tensor.dim(position_ids, 0)
    physical_mode = physical_plan_supported(
        pl.tensor.dim(cmp_block_table, 1), pl.tensor.dim(cmp_kv_flat, 0) // BLOCK_SIZE,
        PLAN_PHYSICAL_ROWS,
    )
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

    # 每个AIV负责32 heads，分两轮16头，40KiB score按[5*16,128]放置；整行assemble
    # 避免A3 Vec窄列插入。真正的数学变化只在下面的全局尺度及PV累加。
    for qk_t in pl.range(qk_core, t_dim, CORE_GROUPS):
        score_bank = pl.tile.full([SPARSE_BLOCKS * FINAL_HEAD_TILE, ATTN_K_TILE],
                                 dtype=pl.FP32, value=FP32_NEG_INF)
        any_valid = pl.cast(0, pl.INT32)
        for qk_tick in pl.range(SPARSE_BLOCKS + 1):
            qk_sb = qk_tick
            if qk_tick < SPARSE_BLOCKS:
                if pl.read(valid_block_mask, [qk_t, qk_sb]) > 0:
                    qk_slot = qk_core * SPARSE_BLOCKS + qk_sb
                    qk_kv_row = qk_slot * ATTN_K_TILE
                    # mask是两角色共同事件谓词；额外地址guard只限制gather，不改变
                    # 事件对数。合法调用必须使用与plan同一批Q/L/positions/页表。
                    # 显式constexpr使关闭路径在Tile状态构建前裁掉此分支；仅靠
                    # physical_mode最后化简为0仍可能留下额外loop-carried Tile。
                    if PLAN_PHYSICAL_ROWS and physical_mode != 0 and qk_sb > 0:
                        # 本轮plan已验证唯一owner、causal位置和每个被选中页。外层已
                        # 获取其phase和业务缓存；mask来自同一plan，不是外部任意标记。
                        # 因此这里无需再次扫描Q/L或逐行查页，仍检查最终物理行容量。
                        # 无效候选为-1，保持本块初始化的零KV；顺序及64行分工不变。
                        # 索引来自已发布的本轮物理plan；一次DMA读64项到UB，避免
                        # 每行Scalar读GM。MGATHER仍逐行DMA，不能称为硬件单指令
                        # 并行采64行；zero模式保留负哨兵/容量外地址的零填充。
                        physical_column = (qk_sb - 1) * ATTN_K_TILE + qk_lane_kv
                        physical_rows = pl.load(
                            cmp_sparse_indices, [qk_t, physical_column], [1, ATTN_K_TILE // 2],
                            target_memory=pl.MemorySpace.Vec,
                        )
                        qk_kv_half = pl.tile.mgather(
                            cmp_kv_flat, physical_rows, coalesce="row", gather_oob="zero"
                        )
                    else:
                        # 只有原SWA/逻辑回退需要先清零；物理分支由MGATHER完整写出，
                        # 不让旧零Tile与新64KiB结果跨分支同时存活。
                        qk_kv_half = pl.tile.full([ATTN_K_TILE // 2, HEAD_DIM], dtype=pl.BF16, value=0.0)
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

            # 必须先取走上块SCORE事件，才能发布本块KV，避免同一事件
            # 连续set而未wait。五槽免去数据覆盖风险，但不能免去事件协议。
            if qk_tick > 0:
                softmax_sb = qk_tick - 1
                if pl.read(valid_block_mask, [qk_t, softmax_sb]) > 0:
                    pl.system.sync_wait(EVENT_BASE + QK_SCORE_READY_EVENT,
                        pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
            if qk_tick < SPARSE_BLOCKS:
                if pl.read(valid_block_mask, [qk_t, qk_tick]) > 0:
                    pl.system.sync_set(EVENT_BASE + QK_KV_READY_EVENT,
                        pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
            if qk_tick > 0:
                capture_sb = qk_tick - 1
                if pl.read(valid_block_mask, [qk_t, capture_sb]) > 0:
                    capture_row = (qk_core * SPARSE_BLOCKS + capture_sb) * H + qk_lane_head
                    captured = pl.load(score_transfer, [capture_row, 0], [FINAL_HEAD_TILE, ATTN_K_TILE])
                    bias = pl.load(sparse_bias, [qk_t, capture_sb * ATTN_K_TILE], [1, ATTN_K_TILE])
                    masked = pl.col_expand_add(pl.mul(captured, SOFTMAX_SCALE), bias)
                    score_bank = pl.tile.assemble(score_bank, masked, [capture_sb * FINAL_HEAD_TILE, 0])
                    any_valid = pl.cast(1, pl.INT32)
        # 初轮采QK时只读本lane前16头，另一半留在已完成的五槽GM中。
        # 第二轮读此前未读过的16头；与原32头一次读相比，总字节相同。
        # 保存两轮m/l只需各128B，不能为了40KiB常驻又保留另一份整score。
        m_rows = pl.tile.create([2, FINAL_HEAD_TILE], dtype=pl.FP32)
        l_rows = pl.tile.create([2, FINAL_HEAD_TILE], dtype=pl.FP32)
        for head_chunk in pl.range(2):
            if head_chunk == 1:
                for load_sb in pl.unroll(SPARSE_BLOCKS):
                    if pl.read(valid_block_mask, [qk_t, load_sb]) > 0:
                        late_row = ((qk_core * SPARSE_BLOCKS + load_sb) * H
                                    + qk_lane_head + FINAL_HEAD_TILE)
                        late_scores = pl.load(score_transfer, [late_row, 0],
                                              [FINAL_HEAD_TILE, ATTN_K_TILE])
                        late_bias = pl.load(sparse_bias, [qk_t, load_sb * ATTN_K_TILE],
                                            [1, ATTN_K_TILE])
                        late_masked = pl.col_expand_add(pl.mul(late_scores, SOFTMAX_SCALE), late_bias)
                        score_bank = pl.tile.assemble(score_bank, late_masked,
                                                     [load_sb * FINAL_HEAD_TILE, 0])
            # sink参与max。无有效块时m=sink、l=0，后续分子置零，结果严格为零。
            global_m = pl.load(attn_sink_col, [qk_lane_head + head_chunk * FINAL_HEAD_TILE, 0], [FINAL_HEAD_TILE, 1])
            for max_sb in pl.unroll(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [qk_t, max_sb]) > 0:
                    max_scores = pl.tile.extract(score_bank, max_sb * (FINAL_HEAD_TILE), 0,
                        [FINAL_HEAD_TILE, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                    max_tmp = pl.create_tile([FINAL_HEAD_TILE, ATTN_K_TILE], dtype=pl.FP32)
                    block_m = pl.row_max(max_scores, max_tmp)
                    global_m = pl.maximum(global_m, block_m)
            global_l = pl.mul(global_m, 0.0)
            for prob_sb in pl.unroll(SPARSE_BLOCKS):
                if pl.read(valid_block_mask, [qk_t, prob_sb]) > 0:
                    prob_scores = pl.tile.extract(score_bank, prob_sb * (FINAL_HEAD_TILE), 0,
                        [FINAL_HEAD_TILE, ATTN_K_TILE], target_memory=pl.MemorySpace.Vec)
                    block_exp = pl.exp(pl.row_expand_sub(prob_scores, global_m))
                    sum_tmp = pl.create_tile([FINAL_HEAD_TILE, ATTN_K_TILE], dtype=pl.FP32)
                    block_l = pl.row_sum(block_exp, sum_tmp)
                    global_l = pl.add(global_l, block_l)
                    block_p = pl.cast(block_exp, target_type=pl.BF16, mode="rint")
                    prob_row = (qk_core * SPARSE_BLOCKS + prob_sb) * H + qk_lane_head + head_chunk * FINAL_HEAD_TILE
                    pl.store(block_p, [prob_row, 0], probability_transfer)
            # 行布局保存，避免对ColMajor列向量做不支持的TEXTRACT。
            m_rows = pl.tile.assemble(m_rows, pl.reshape(global_m, [1, FINAL_HEAD_TILE]),
                                      [head_chunk, 0])
            l_rows = pl.tile.assemble(l_rows, pl.reshape(global_l, [1, FINAL_HEAD_TILE]),
                                      [head_chunk, 0])
        pl.system.sync_set(EVENT_BASE + QK_PROB_READY_EVENT,
            pipe=pl.PipeType.MTE3, ffts_mode=2, core_type=pl.KernelType.AIV)
        pl.system.sync_wait(EVENT_BASE + QK_PV_READY_EVENT,
            pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
        if any_valid != 0:
            pv_row = qk_core * SPARSE_BLOCKS * H + qk_lane_head
            left_valid = pl.load(pv_transfer, [pv_row, 0], [H // 2, HEAD_DIM // 2])
            right_valid = pl.load(pv_transfer, [pv_row, HEAD_DIM // 2], [H // 2, HEAD_DIM // 2])
        else:
            left_valid = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
            right_valid = pl.tile.full([H // 2, HEAD_DIM // 2], dtype=pl.FP32, value=0.0)
        m_valid = pl.reshape(m_rows, [H // 2, 1])
        l_valid = pl.reshape(l_rows, [H // 2, 1])
        pv_t = qk_t
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
    return kv_transfer, probability_transfer, mi_transfer, li_transfer, o_packed_heads
