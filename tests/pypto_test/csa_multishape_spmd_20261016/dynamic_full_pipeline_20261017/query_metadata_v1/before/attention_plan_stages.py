# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA Attention 的候选计划与逆 RoPE 正负号准备。

整体范围与阅读顺序
    先读 attention_slot_plan_stage：将 Indexer 的逻辑 Top-K 转为可安全消费的
    压缩索引、640 列 softmax bias 和五个候选块的有效标志；再读
    attention_inverse_rope_sign_stage：准备最终 attention 输出所需的逆旋转
    sin 符号。两段来自 canonical decode_sparse_attn_csa.py 的 plan/rope_cs，
    不包括 QK/PV、softmax、环槽事件、输出投影，也不创建 task/SPMD/核间通知。

输入布局与真正动态的请求边界
    topk 是 INT32[T,512] 逻辑压缩行号，positions 为 Native INT64[T]；Q 是
    原始 INT32[B+1] query_start_loc，L 是包含本轮 query 的 INT32[B] 长度。
    T、B、Q 实际轴独立，允许同 T 对应不同 mixed-query 分组，禁止 token//S。
    两份 INT32[B,table_columns] 页表的列容量分别动态；两种 cache 的物理页数
    由外层已有 descriptor 的 dim 以 INDEX Scalar 传入。这里不接收 cache
    payload，所以不会读 cache；后续 gather 必须复用相同 Q/L、页表和容量。
    活动 token 还须满足 position=L-(Q[r+1]-Q[r])+(token-Q[r])。无 owner、
    多 owner、dummy、非法 Q 或陈旧 position 全部生成无效计划，不读页表。
    为保留八行向量数学，Top-K 可以读取实际 T 内的 dummy 行，但不据它访问页表。

输出、所有权与容量
    compressed_indices 为 INT32[PLAN_CAP,512]，bias 为 FP32[PLAN_CAP,640]，
    valid_blocks 为 INT32[PLAN_CAP,16]；PLAN_CAP>=T 由入口捕获前校验。
    每个实际 token 每次覆盖全部 indices/bias 及 mask 前五列，其余十一列和
    T 后的工作区保持原值。mask 每行正好 64B，八行 tile 独占各行，Scalar 写
    前五列不会与其它 worker 分享同一行；底层分配须保持 64B 对齐。
    sin 为 FP32[T,64]，signed_sin 为 FP32[ROPE_CAP,64]，ROPE_CAP>=T。
    两种容量不是实际 T，不可由容量反推活动请求或让消费者读取未写 pad。
    显式 worker_id/WORKERS 把第 tile 个八行块分给 tile%WORKERS；尾块用
    min(8,T-first) 限制 load/store。所有 UB 辅助 Tile 归本核，不额外分配 GM。

数学、保护与同步责任
    压缩候选保留 FP32 clamp、INT32 trunc 上界、index/bias 和 row_max 顺序；
    新增页表列/物理页上下界条件乘入同一 mask，不能只阻止 gather 却保留零 bias。
    压缩页表先按请求以Vector验证当前长度可能使用的全部列，尾tile的有效列
    必须贯穿load、广播与row_min，未加载的UB padding不参与合法性判断；同八行tile中的
    相邻同请求复用本核结果。全部页合法时批量设置候选页mask，含空洞、列容量
    不足或超出快速检查数值边界时保留逐候选Scalar回退；不跳过有效性检查。
    SWA 保留128行窗口、最多五个32-token页run；正常页走整块bias，页空洞及
    越界页才按合法run重建本行。非法候选 index=-1、bias=-1e20、块有效位=0。
    逆 RoPE 的偶数列乘+1、奇数列乘-1，与 query 正向 RoPE 的符号相反。
    外层先等 Top-K 与 Native metadata 可见；plan 完成后再发布阶段给 QK/AIV。
    rope 可与 plan 独立运行，最终输出消费 signed_sin 前等实际 row owner。
    本模块不维护信号缓存；源码/CPU通过不能替代设备精度、同步或完整CSA验证。
"""

import pypto.language as pl

from .request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, query_request

T_DYN = pl.dynamic("CSA_ATTN_PLAN_T")
PLAN_CAP_DYN = pl.dynamic("CSA_ATTN_PLAN_CAP")
ROPE_CAP_DYN = pl.dynamic("CSA_ATTN_ROPE_CAP")
ORIGINAL_COLUMNS_DYN = pl.dynamic("CSA_ATTN_ORIGINAL_COLUMNS")
COMPRESSED_COLUMNS_DYN = pl.dynamic("CSA_ATTN_COMPRESSED_COLUMNS")
ROWS = 8
TOPK = 512
WIN = 128
BLOCK_SIZE = 32
ATTN_K_TILE = 128
SPARSE_BLOCKS = 5
BIAS_COLS = 640
MASK_COLS = 16
ROPE_DIM = 64
SWA_RUNS = 5
COMPRESS_RATIO_INV = 0.25
CSA_CMP_GE_BIAS = 1.0
NEG_INF = -1.0e20
PAGE_CHECK_TILE = 1024
# FP32可精确表达此范围内的页号和长度；超出时走原INT32/INDEX逐候选检查。
EXACT_FP32_INT = 16777216


@pl.jit.inline(auto_scope=False)
def compressed_pages_all_valid(
    table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    request: pl.Scalar[pl.INDEX],
    length: pl.Scalar[pl.INDEX],
    physical_pages: pl.Scalar[pl.INDEX],
) -> pl.Scalar[pl.INT32]:
    """保守快速判定：返回1才允许批量通过，返回0仍逐个检查候选，不丢合法页。

    一次读取1024个页号并做向量范围检查；最后只读取valid_shape内的列。
    当前请求所有query的压缩候选上界不超过length//4，故覆盖该前缀即可。
    本函数不接收候选，不改变索引/排序，不在GM保存跨调用可陈旧的判断。
    """
    needed = ((length // 4) + BLOCK_SIZE - 1) // BLOCK_SIZE
    all_valid = pl.cast(0, pl.INT32)
    if length <= EXACT_FP32_INT and physical_pages > 0 and physical_pages <= EXACT_FP32_INT:
        if needed <= pl.tensor.dim(table, 1):
            all_valid = pl.cast(1, pl.INT32)
            page_limit = pl.cast(pl.cast(physical_pages, pl.INT32), pl.FP32)
            reduce_tmp = pl.create_tile([ROWS, PAGE_CHECK_TILE], dtype=pl.FP32)
            for column in pl.range(0, needed, PAGE_CHECK_TILE):
                live = pl.min(PAGE_CHECK_TILE, needed - column)
                values = pl.cast(
                    pl.load(table, [request, column], [1, PAGE_CHECK_TILE], valid_shape=[1, live]),
                    pl.FP32,
                )
                lower_ok = pl.minimum(pl.maximum(pl.add(values, 1.0), 0.0), 1.0)
                upper_ok = pl.minimum(pl.maximum(pl.add(pl.neg(values), page_limit), 0.0), 1.0)
                # A3列向量结果要求32B物理步长，1×1 FP32归约结果不合法。
                # 同一行展开到8行后取第0个归约结果，避免不合法的4B列布局。
                # 有效列必须沿广播保留：短页表只加载live列，不能把对应1×live
                # 向量广播到全1024列后归约，否则会读取UB未初始化的padding，
                # 把全合法页误判成存在空洞。先限制左操作数，再显式约束归约输入；
                # 数学仍对[0,live)逐列检查，不把未加载列当作实际页表内容。
                ones = pl.set_validshape(
                    pl.tile.full([ROWS, PAGE_CHECK_TILE], dtype=pl.FP32, value=1.0), ROWS, live
                )
                repeated = pl.set_validshape(
                    pl.col_expand_mul(ones, pl.mul(lower_ok, upper_ok)), ROWS, live
                )
                block_ok = pl.row_min(repeated, reduce_tmp)
                if pl.read(block_ok, [0, 0]) < 1.0:
                    all_valid = pl.cast(0, pl.INT32)
    return all_valid


@pl.jit.inline(auto_scope=False)
def attention_slot_plan_stage(
    topk: pl.Tensor[[T_DYN, TOPK], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    original_table: pl.Tensor[[REQUESTS_DYN, ORIGINAL_COLUMNS_DYN], pl.INT32],
    compressed_table: pl.Tensor[[REQUESTS_DYN, COMPRESSED_COLUMNS_DYN], pl.INT32],
    indices: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, TOPK], pl.INT32]],
    bias: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, BIAS_COLS], pl.FP32]],
    valid_blocks: pl.InOut[pl.Tensor[[PLAN_CAP_DYN, MASK_COLS], pl.INT32]],
    original_pages: pl.Scalar[pl.INDEX],
    compressed_pages: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """每个八行块独占 indices/bias/mask 行；只访问合法 owner 的 Native 页表。"""
    tokens = pl.tensor.dim(positions, 0)
    # 固定8是向量tile，不是请求长度。一个tile可跨请求，逐行用Q/L判断owner。
    for first in pl.range(worker_id * ROWS, tokens, WORKERS * ROWS):
        rows = pl.min(ROWS, tokens - first)
        c_raw_tile = pl.cast(
            pl.load(topk, [first, 0], [ROWS, TOPK], valid_shape=[rows, TOPK]),
            target_type=pl.FP32,
        )
        # 从一条32B连续行reshape成列向量，让物理布局为column-major；直接
        # full([8,1])的row-major每行仅4B，不满足A3 Vec的32B物理步长。
        c_pos = pl.reshape(pl.tile.full([1, ROWS], dtype=pl.FP32, value=0.0), [ROWS, 1])
        v_len_col = pl.reshape(pl.tile.full([1, ROWS], dtype=pl.FP32, value=0.0), [ROWS, 1])
        owners = pl.reshape(pl.tile.full([1, ROWS], dtype=pl.INT32, value=-1), [ROWS, 1])
        # 先确认请求活动再读position。owner保存于本核UB，后面的页检查无需再次扫Q。
        # 尾部未用行保留owner=-1/length=0，且永远不写出这些物理tile填充行。
        for row in pl.range(rows):
            token = first + row
            request, begin, end, active = query_request(bounds, lengths, token, tokens)
            if active != 0:
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
                if position == length - (end - begin) + token - begin:
                    pl.tile.write(owners, [row, 0], pl.cast(request, pl.INT32))
                    # L为INT32且已证明position位于[0,L)，INT32中转不会损失
                    # 地址值；显式中转适配Scalar不支持INDEX直接转FP32的接口。
                    pl.tile.write(c_pos, [row, 0], pl.cast(pl.cast(position, pl.INT32), pl.FP32))
                    pl.tile.write(v_len_col, [row, 0], pl.cast(pl.cast(pl.min(position + 1, WIN), pl.INT32), pl.FP32))
        # canonical压缩上界仍先FP32再INT32 trunc；页检查读取同一个上界Tile，
        # 不另用整数除法悄悄改变浮点边界。正常模型范围内这些整数可由FP32精确表达。
        c_pos_q = pl.cast(
            pl.cast(pl.mul(pl.add(c_pos, 1.0), COMPRESS_RATIO_INV), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        c_upper_b = pl.row_expand_mul(pl.tile.full([ROWS, TOPK], dtype=pl.FP32, value=1.0), c_pos_q)
        c_ge_tile = pl.minimum(pl.maximum(pl.add(c_raw_tile, CSA_CMP_GE_BIAS), 0.0), 1.0)
        c_lt_tile = pl.minimum(pl.maximum(pl.sub(c_upper_b, c_raw_tile), 0.0), 1.0)
        c_mask_tile = pl.mul(c_ge_tile, c_lt_tile)
        # 页保护以消费者最终读取的INT32索引为准，而不是原始Top-K。在很大
        # 的整数边界上，FP32的(raw+1)-1可能舍入到相邻页；两者必须用同一地址。
        # 保存在本核Tile还避免逐候选再次Scalar读取Top-K GM。
        c_page_indices = pl.cast(pl.sub(pl.mul(c_mask_tile, pl.add(c_raw_tile, 1.0)), 1.0), target_type=pl.INT32)
        c_pages_valid = pl.tile.full([ROWS, TOPK], dtype=pl.FP32, value=0.0)
        v_columns = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
        v_col_tile = pl.col_expand_mul(pl.tile.full([ROWS, WIN], dtype=pl.FP32, value=1.0), v_columns)
        v_valid_tile = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(v_col_tile, v_len_col)), 0.0), 1.0)
        pl.store(pl.set_validshape(pl.mul(pl.sub(v_valid_tile, 1.0), -NEG_INF), rows, WIN), [first, 0], bias)
        # 页表在本次调用只读。缓存只在本核当前八行tile存活，换请求就重新验证，
        # 不把上一次decode的合法页判断沿用到本次。常见全合法请求无需512次Scalar查页。
        checked_request = pl.cast(-1, pl.INDEX)
        checked_pages_valid = pl.cast(0, pl.INT32)
        for row in pl.range(rows):
            token = first + row
            request = pl.cast(pl.read(owners, [row, 0]), pl.INDEX)
            v_block_valid = pl.cast(0, pl.INT32)
            if request >= 0:
                upper = pl.cast(pl.cast(pl.read(c_pos_q, [row, 0]), pl.INT32), pl.INDEX)
                if request != checked_request:
                    length = pl.cast(pl.read(lengths, [request]), pl.INDEX)
                    checked_pages_valid = compressed_pages_all_valid(
                        compressed_table, request, length, compressed_pages
                    )
                    checked_request = request
                if checked_pages_valid != 0:
                    # 最终INT32索引仍须满足原0<=index<upper检查；不能只验证原始Top-K，
                    # 因为原浮点(raw+1)-1可能改变极大数边界。这里只把逐元素判断向量化。
                    # row<rows已经由循环保证；显式声明该单行有效域，不能继承动态父Tile的rows。
                    final_indices = pl.cast(
                        pl.slice(c_page_indices, [1, TOPK], [row, 0], valid_shape=[1, TOPK]), pl.FP32
                    )
                    final_ge = pl.minimum(pl.maximum(pl.add(final_indices, 1.0), 0.0), 1.0)
                    upper_float = pl.cast(pl.cast(upper, pl.INT32), pl.FP32)
                    final_lt = pl.minimum(pl.maximum(pl.add(pl.neg(final_indices), upper_float), 0.0), 1.0)
                    c_pages_valid = pl.tile.assemble(c_pages_valid, pl.mul(final_ge, final_lt), [row, 0])
                else:
                    # 请求存在页空洞或容量不足时，只屏蔽非法候选；合法候选继续保留。
                    for candidate in pl.range(TOPK):
                        raw = pl.cast(pl.read(c_page_indices, [row, candidate]), pl.INDEX)
                        if raw >= 0 and raw < upper:
                            column = raw // BLOCK_SIZE
                            if column < pl.tensor.dim(compressed_table, 1):
                                page = pl.cast(pl.read(compressed_table, [request, column]), pl.INDEX)
                                if page >= 0 and page < compressed_pages:
                                    pl.tile.write(c_pages_valid, [row, candidate], pl.cast(1.0, pl.FP32))
                position = pl.cast(pl.read(positions, [token]), pl.INDEX)
                v_length = pl.min(position + 1, WIN)
                v_start = position - v_length + 1
                v_head = v_start % BLOCK_SIZE
                v_has_hole = pl.cast(0, pl.INT32)
                for v_run in pl.range(SWA_RUNS):
                    v_lo = pl.max(v_run * BLOCK_SIZE - v_head, 0)
                    v_hi = pl.min((v_run + 1) * BLOCK_SIZE - v_head, v_length)
                    if v_hi > v_lo:
                        column = (v_start + v_lo) // BLOCK_SIZE
                        page_valid = pl.cast(0, pl.INT32)
                        if column >= 0 and column < pl.tensor.dim(original_table, 1):
                            page = pl.cast(pl.read(original_table, [request, column]), pl.INDEX)
                            if page >= 0 and page < original_pages:
                                page_valid = pl.cast(1, pl.INT32)
                        if page_valid != 0:
                            v_block_valid = pl.cast(1, pl.INT32)
                        else:
                            v_has_hole = pl.cast(1, pl.INT32)
                # 仅空洞时重算本行，保留canonical常见全合法页的八行向量路径。
                if v_has_hole == 1:
                    v_columns_row = pl.cast(pl.tile.arange(0, [1, WIN], dtype=pl.INT32), pl.FP32)
                    v_valid = pl.tile.full([1, WIN], dtype=pl.FP32, value=0.0)
                    for v_hole_run in pl.range(SWA_RUNS):
                        v_hole_lo = pl.max(v_hole_run * BLOCK_SIZE - v_head, 0)
                        v_hole_hi = pl.min((v_hole_run + 1) * BLOCK_SIZE - v_head, v_length)
                        if v_hole_hi > v_hole_lo:
                            column = (v_start + v_hole_lo) // BLOCK_SIZE
                            if column >= 0 and column < pl.tensor.dim(original_table, 1):
                                page = pl.cast(pl.read(original_table, [request, column]), pl.INDEX)
                                if page >= 0 and page < original_pages:
                                    v_lo_fp32 = pl.cast(pl.cast(v_hole_lo, pl.INT32), pl.FP32)
                                    v_hi_fp32 = pl.cast(pl.cast(v_hole_hi, pl.INT32), pl.FP32)
                                    v_ge = pl.minimum(
                                        pl.maximum(pl.add(pl.sub(v_columns_row, v_lo_fp32), 1.0), 0.0), 1.0
                                    )
                                    v_lt = pl.minimum(pl.maximum(pl.add(pl.neg(v_columns_row), v_hi_fp32), 0.0), 1.0)
                                    v_valid = pl.add(v_valid, pl.mul(v_ge, v_lt))
                    pl.store(pl.mul(pl.sub(v_valid, 1.0), -NEG_INF), [token, 0], bias)
            pl.write(valid_blocks, [token, 0], v_block_valid)
        # 页合法性与数学有效性必须在写indices/bias/block flags之前合并。
        c_mask_tile = pl.mul(c_mask_tile, c_pages_valid)
        c_out_tile = pl.sub(pl.mul(c_mask_tile, pl.add(c_raw_tile, 1.0)), 1.0)
        pl.store(pl.set_validshape(pl.cast(c_out_tile, target_type=pl.INT32), rows, TOPK), [first, 0], indices)
        pl.store(pl.set_validshape(pl.mul(pl.minimum(c_out_tile, 0.0), -NEG_INF), rows, TOPK), [first, WIN], bias)
        # 每128候选形成一个块有效位；只写1..4，连同上面的SWA列0共五列。
        # 后续QK必须只读这五列，不能把同一64B行里保留的11列poison解释成任务。
        # Tile归约显式借用UB暂存；仍是原来的128列row_max，不改变归约树。
        reduce_tmp = pl.create_tile([ROWS, ATTN_K_TILE], dtype=pl.FP32)
        for block in pl.range(1, SPARSE_BLOCKS):
            c_blk_valid = pl.row_max(
                pl.slice(
                    c_mask_tile, [ROWS, ATTN_K_TILE], [0, (block - 1) * ATTN_K_TILE], valid_shape=[rows, ATTN_K_TILE]
                ),
                reduce_tmp,
            )
            for row in pl.range(rows):
                c_valid = pl.cast(pl.read(c_blk_valid, [row, 0]), pl.INT32)
                pl.write(valid_blocks, [first + row, block], c_valid)
    return indices, bias, valid_blocks


@pl.jit.inline(auto_scope=False)
def attention_inverse_rope_sign_stage(
    sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    signed_sin: pl.InOut[pl.Tensor[[ROPE_CAP_DYN, ROPE_DIM], pl.FP32]],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """逐token逆RoPE预处理；与请求Q分组无关，输出容量尾部保持未写。"""
    tokens = pl.tensor.dim(sin, 0)
    # 每个worker重建小符号Tile，不能共享另一个核的UB地址。计算顺序与
    # canonical相同，只将原先等于S的tile改为8，并保留实际尾rows。
    for first in pl.range(worker_id * ROWS, tokens, WORKERS * ROWS):
        rows = pl.min(ROWS, tokens - first)
        cs_ones = pl.tile.full([ROWS, ROPE_DIM], dtype=pl.FP32, value=1.0)
        cs_idx_f = pl.cast(pl.tile.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_idx_f)
        cs_dup_i32 = pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))
        cs_sin = pl.load(sin, [first, 0], [ROWS, ROPE_DIM], valid_shape=[rows, ROPE_DIM])
        cs_sign_rows = pl.set_validshape(cs_sign, rows, ROPE_DIM)
        pl.store(pl.mul(cs_sin, cs_sign_rows), [first, 0], signed_sin)
    return signed_sin
