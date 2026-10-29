# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""动态 CSA 的 O_A 分组投影与分组量化：外层显式分核的三个核内阶段。

输入输出与阅读顺序
    先读 o_projection_a_stage_nd/nz：AIC 消费 attention 的 packed heads，
    按八个 O group 独立做 K4096→N1024，写 FP32[P_MM,8192]。再读
    o_projection_group_quant_stage：AIV对每行先BF16舍入、跨8192列取amax，
    写INT8[P_MM,8192]和各组相同的FP32[8,P_SCALE]。
    wo_a 保持 Native BF16[8,4096,1024] 的实际 ND 或 NZ 存储；两个函数
    分开声明布局，调用方按真实权重选择，不做格式转换或导入期设备探针。

动态轴、容量与别名
    tokens 是调用方从真实业务 Tensor 动态轴得到的设备 Scalar T，不从
    packed/workspace 容量反推。packed 是 BF16[PACK_TOTAL_ROWS,4096]，
    PACK_TOTAL_ROWS=8*P_PACK，组 g 的第 t 行位于 g*P_PACK+t，P_PACK>=T。
    FP32/INT8 工作区共享 P_MM 容量合同但不共享存储；P_SCALE 独立且>=T。
    O_B 使用的 ROW_TILE 为32/96/128之一，P_MM>=ceil(T/ROW_TILE)*ROW_TILE，
    并须足以保存实际T行。packed总行数须整除8，各视图连续且不得非法别名。
    所有容量轴均 pl.dynamic；固定8组、模型维度及核内tile不代表固定B/S。
    NZ路径允许M128/N128或N256，以及容量128<T<=160的M160/N192区间；
    后者按五个N192和一个静态N64覆盖1024列。ND接口仍保持原M128路径。
    外层捕获前拒绝不一致容量/layout；不能让生产者跳写而消费者读取旧值。

分工、数学与尾行
    NZ O_A以(group,ceil(T/A_ROW_TILE)行块,ceil(1024/A_COL_TILE)列块)展平，
    logical%WORKERS为AIC owner；AIC发布和量化获取使用同一组内任务数。
    每块按所选起点循环累加16个K256；原流水只重叠本核搬运。
    仅迁移当前实际启用的 PIPELINE_OA=False，不搬未启用的显式L0流水候选。
    量化以(group,32行块)展平轮转AIV，块内以8行valid shape计算；有效行数贯穿load、
    row_max、常数、div、cast和store，绝不读取 O_A 未写的 FP32 padding。
    每组最后一个活动32行块是唯一尾owner，额外将INT8的[T,P_USED)写零，
    P_USED=ceil(T/ROW_TILE)*ROW_TILE；O_B 才能安全整块读取。T=0不产生任务。
    量化保留127/amax、amax*(1/127)及INT32 rint→FP16 round→INT8 trunc；
    反量化scale不能替换为recip(量化乘数)，也保持Native BF16往返舍入。

同步与生命周期
    groups helper 允许外层按输出组发布，而不改变原有逻辑任务的 owner。
    group_begin/end 是核内 Scalar 范围；先定位该组内满足 logical%WORKERS
    的首项，后续仍按 WORKERS 条带循环。只改变独立组之间的可见时机，不
    切断组内 K 累加、amax 归约或 padding 清零。全阶段入口调用相同数学体。
    helper不创建spmd、task或notify/wait。O_A先等所消费packed行的全部
    attention生产者，量化先等全部八组的所有O_A列块。量化的真实INT8行、
    scale和所需尾零均完成发布后，O_B才可读整ROW_TILE；外层负责业务fence、
    缓存维护和阶段信号。本核return仅返回GM别名，不能当作其它核完成。
    每个输出行列块唯一写；返回前所有本核wave已经执行，但消费者完成前
    不得复用GM。工作区标InOut保留未写区，P_USED以后的INT8和T以后的scale
    不承诺初始化。BYPASS策略如需使用，须在外层spmd直属设置。
    当前是源级阶段迁移；CPU编译不等于整层设备精度、同步或性能已通过。

Native数值资格
    M128/N128路径在17≤T≤32按group/N128选择两种半程起点；
    33≤T≤48选择四种K256起点。每族资格均由具体T实测给出，不外推整段。
    保留原64个logical任务及owner、原动态packed步长和全部完成信号；
    T240/M128/N256保留64个父任务，内部两N128按Native24核的八个K256起点累加。
    其它T或tiling的WO-A继续原路径；后续量化统一恢复BF16边界及全8192列amax。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import INT8_SCALE_MAX

O_GROUPS = 8
O_GROUP_IN = 4096
O_LORA = 1024
PROJ_A_ROW_TILE = 128
A_K_TILE = 256
QUANT_TASK_T_TILE = 32
QUANT_TOKEN_TILE = 8
PACK_TOTAL_ROWS_DYN = pl.dynamic("CSA_OA_PACK_TOTAL_ROWS")
MM_ROWS_DYN = pl.dynamic("CSA_OA_MM_ROWS")
SCALE_ROWS_DYN = pl.dynamic("CSA_OA_SCALE_ROWS")


@pl.jit.inline(auto_scope=False)
def o_projection_a_stage_nd(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    """ND O_A；WORKERS>0、worker合法、A_COL_TILE为128/256，调用前等attention。"""
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    row_blocks = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    column_blocks = O_LORA // A_COL_TILE
    # 将原group并行与行/列任务组合为本核条带；每个输出块仍完整做相同K累加。
    for logical in pl.range(worker_id, O_GROUPS * row_blocks * column_blocks, WORKERS):
        g = logical // (row_blocks * column_blocks)
        pa_unit = logical % (row_blocks * column_blocks)
        pa_rb = pa_unit // column_blocks
        nf = pa_unit % column_blocks
        pa_r0 = pa_rb * PROJ_A_ROW_TILE
        pa_rows = pl.min(PROJ_A_ROW_TILE, tokens - pa_r0)
        pa_src0 = g * pack_stride + pa_r0
        out_col_g = g * O_LORA
        n0 = nf * A_COL_TILE
        xa_first = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE])
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        # 首块matmul建立实际M行的accumulator布局；不能改成任意全128行零种子。
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE])
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        # wo_a保留group轴，acc仍带该轴；assemble沿原映射写回有效M，不reshape整个GM。
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def _rectangle_o_projection_a_tile_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    g: pl.Scalar[pl.INDEX],
    pa_src0: pl.Scalar[pl.INDEX],
    pa_r0: pl.Scalar[pl.INDEX],
    pa_rows: pl.Scalar[pl.INDEX],
    out_col_g: pl.Scalar[pl.INDEX],
    n0: pl.Scalar[pl.INDEX],
    A_ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
):
    xa_first = pl.slice(o_packed, [A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE])
    wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
    acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
    for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
        k0 = kb * A_K_TILE
        xa_k_chunk = pl.slice(o_packed, [A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE])
        wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
    o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def _rectangle_o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    A_ROW_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """NZ O_A 的动态组区间；0≤begin≤end≤8，保持全局条带 owner 与原 K 链。"""
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    row_blocks = (tokens + A_ROW_TILE - 1) // A_ROW_TILE
    column_blocks = (O_LORA + A_COL_TILE - 1) // A_COL_TILE
    group_first = group_begin * row_blocks * column_blocks
    group_limit = group_end * row_blocks * column_blocks
    first = worker_id + (group_first + WORKERS - 1 - worker_id) // WORKERS * WORKERS
    for logical in pl.range(pl.max(0, first), group_limit, WORKERS):
        g = pl.max(0, logical // (row_blocks * column_blocks))
        pa_unit = logical % (row_blocks * column_blocks)
        pa_rb = pa_unit // column_blocks
        nf = logical % ((O_LORA + A_COL_TILE - 1) // A_COL_TILE)
        pa_r0 = pa_rb * A_ROW_TILE
        pa_rows = pl.min(A_ROW_TILE, tokens - pa_r0)
        pa_src0 = g * pack_stride + pa_r0
        out_col_g = g * O_LORA
        n0 = pl.max(0, nf) * A_COL_TILE
        # NZ末维必须静态；第六块只算/写64列，不越过本组的1024列。
        if nf < 5:
            o_r_pad = _rectangle_o_projection_a_tile_nz(
                o_packed, wo_a, o_r_pad, g, pa_src0, pa_r0, pa_rows, out_col_g, n0, A_ROW_TILE, 192
            )
        else:
            o_r_pad = _rectangle_o_projection_a_tile_nz(
                o_packed, wo_a, o_r_pad, g, pa_src0, pa_r0, pa_rows, out_col_g, n0, A_ROW_TILE, 64
            )
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def _original_o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    A_ROW_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """NZ O_A 的动态组区间；0≤begin≤end≤8，保持全局条带 owner 与原 K 链。"""
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    row_blocks = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    column_blocks = O_LORA // A_COL_TILE
    group_first = group_begin * row_blocks * column_blocks
    group_limit = group_end * row_blocks * column_blocks
    first = worker_id + (group_first + WORKERS - 1 - worker_id) // WORKERS * WORKERS
    for logical in pl.range(pl.max(0, first), group_limit, WORKERS):
        g = pl.max(0, logical // (row_blocks * column_blocks))
        pa_unit = logical % (row_blocks * column_blocks)
        pa_rb = pa_unit // column_blocks
        nf = logical % (O_LORA // A_COL_TILE)
        pa_r0 = pa_rb * PROJ_A_ROW_TILE
        pa_rows = pl.min(PROJ_A_ROW_TILE, tokens - pa_r0)
        pa_src0 = g * pack_stride + pa_r0
        out_col_g = g * O_LORA
        n0 = pl.max(0, nf) * A_COL_TILE
        xa_first = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE])
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE])
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def _native_small_o_projection_a_tile_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    g: pl.Scalar[pl.INDEX],
    n0: pl.Scalar[pl.INDEX],
    FIRST_PANEL: pl.constexpr,
):
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    # 合法n0由非负logical的列编号产生，范围0..896。
    column0 = pl.max(0, n0)
    first_k = FIRST_PANEL * A_K_TILE
    first_x = pl.slice(o_packed, [128, A_K_TILE], [g * pack_stride, first_k], valid_shape=[tokens, A_K_TILE])
    first_w = wo_a[g : g + 1, first_k : first_k + A_K_TILE, column0 : column0 + 128]
    acc = pl.matmul(first_x, first_w, out_dtype=pl.FP32)
    for panel in pl.range(FIRST_PANEL + 1, O_GROUP_IN // A_K_TILE):
        k = panel * A_K_TILE
        chunk_x = pl.slice(o_packed, [128, A_K_TILE], [g * pack_stride, k], valid_shape=[tokens, A_K_TILE])
        chunk_w = wo_a[g : g + 1, k : k + A_K_TILE, column0 : column0 + 128]
        acc = pl.matmul_acc(acc, chunk_x, chunk_w)
    for wrapped_panel in pl.range(FIRST_PANEL):
        wrapped_k = wrapped_panel * A_K_TILE
        wrapped_x = pl.slice(o_packed, [128, A_K_TILE], [g * pack_stride, wrapped_k], valid_shape=[tokens, A_K_TILE])
        wrapped_w = wo_a[g : g + 1, wrapped_k : wrapped_k + A_K_TILE, column0 : column0 + 128]
        acc = pl.matmul_acc(acc, wrapped_x, wrapped_w)
    o_r_pad = pl.assemble(o_r_pad, acc, [0, g * O_LORA + n0])
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def _native_small_o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """M48范围的N128累加起点；保留原owner及真实tokens有效行。"""
    column_blocks = O_LORA // 128
    group_first = group_begin * column_blocks
    group_limit = group_end * column_blocks
    # 全局条带从非负物理worker起步，再过滤组区间。所执行的logical与原
    # 向上取整起点公式相同，但NZ证明不必穿过含减法的起点表达式。
    for logical in pl.range(worker_id, O_GROUPS * column_blocks, WORKERS):
        if logical >= group_first:
            if logical < group_limit:
                # NZ符号证明需要直接的常量除数，不能经局部Scalar别名。
                g = logical // (O_LORA // 128)
                column = logical % (O_LORA // 128)
                n0 = column * 128
                if tokens <= 32:
                    band = 0
                    if column >= 3:
                        if column < 6:
                            band = 1
                    # 静态起点确保完整内联后NZ偏移仍可证明非负。
                    if (g % 2 + band) % 2 == 1:
                        o_r_pad = _native_small_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, tokens, g, n0, 8)
                    else:
                        o_r_pad = _native_small_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, tokens, g, n0, 0)
                else:
                    # 三个M16的Native起点族；保持非负模与四种常量NZ偏移。
                    kind = (column // 3 + 8 - g) % 4
                    if kind == 0:
                        o_r_pad = _native_small_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, tokens, g, n0, 0)
                    elif kind == 1:
                        o_r_pad = _native_small_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, tokens, g, n0, 5)
                    elif kind == 2:
                        o_r_pad = _native_small_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, tokens, g, n0, 10)
                    else:
                        o_r_pad = _native_small_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, tokens, g, n0, 15)
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def _native_t240_o_projection_a_tile_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    group: pl.Scalar[pl.INDEX],
    row: pl.Scalar[pl.INDEX],
    valid: pl.Scalar[pl.INDEX],
    column: pl.Scalar[pl.INDEX],
    FIRST_PANEL: pl.constexpr,
):
    """静态K起点与既有小T helper相同，避免NZ证明依赖跨分支的Scalar定义。"""
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    n0 = pl.max(0, column) * 128
    first_k = FIRST_PANEL * A_K_TILE
    a = pl.slice(o_packed, [128, A_K_TILE], [group * pack_stride + row, first_k], valid_shape=[valid, A_K_TILE])
    b = wo_a[group : group + 1, first_k : first_k + A_K_TILE, n0 : n0 + 128]
    acc = pl.matmul(a, b, out_dtype=pl.FP32)
    for panel in pl.range(FIRST_PANEL + 1, 16):
        k = panel * A_K_TILE
        chunk_a = pl.slice(o_packed, [128, A_K_TILE], [group * pack_stride + row, k], valid_shape=[valid, A_K_TILE])
        chunk_b = wo_a[group : group + 1, k : k + A_K_TILE, n0 : n0 + 128]
        acc = pl.matmul_acc(acc, chunk_a, chunk_b)
    for wrapped in pl.range(FIRST_PANEL):
        k_wrap = wrapped * A_K_TILE
        wrap_a = pl.slice(o_packed, [128, A_K_TILE], [group * pack_stride + row, k_wrap], valid_shape=[valid, A_K_TILE])
        wrap_b = wo_a[group : group + 1, k_wrap : k_wrap + A_K_TILE, n0 : n0 + 128]
        acc = pl.matmul_acc(acc, wrap_a, wrap_b)
    o_r_pad = pl.assemble(o_r_pad, acc, [row, group * O_LORA + n0])
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def _native_t240_o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """T240：保持M128/N256父任务owner及原按组通知，只改内部N/K分块。"""
    group_first = group_begin * 8
    group_limit = group_end * 8
    # 固定64父任务再过滤组范围；避免NZ偏移证明穿过动态条带起点的除法。
    for logical in pl.range(worker_id, 64, WORKERS):
        if logical >= group_first and logical < group_limit:
            group = logical // 8
            row = logical // 4 % 2 * 128
            valid = pl.min(128, tokens - row)
            for sub in pl.unroll(2):
                column = logical % 4 * 2 + sub
                native_logical = group * 8 + column
                native_owner = native_logical // 3
                if native_logical >= 48:
                    native_owner = 16 + (native_logical - 48) // 2
                # Native M240 BMM的64个N128项：前16核各3项，后8核各2项。
                kind = native_owner % 8
                if kind == 0:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, group, row, valid, column, 0)
                elif kind == 1:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, group, row, valid, column, 2)
                elif kind == 2:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, group, row, valid, column, 4)
                elif kind == 3:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, group, row, valid, column, 6)
                elif kind == 4:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(o_packed, wo_a, o_r_pad, group, row, valid, column, 8)
                elif kind == 5:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(
                        o_packed, wo_a, o_r_pad, group, row, valid, column, 10
                    )
                elif kind == 6:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(
                        o_packed, wo_a, o_r_pad, group, row, valid, column, 12
                    )
                else:
                    o_r_pad = _native_t240_o_projection_a_tile_nz(
                        o_packed, wo_a, o_r_pad, group, row, valid, column, 14
                    )
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def o_projection_a_groups_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    A_ROW_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    if A_COL_TILE == 256 and A_ROW_TILE == 128 and tokens == 240:
        o_r_pad = _native_t240_o_projection_a_groups_stage_nz(
            o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, group_begin, group_end
        )
    elif A_COL_TILE == 128 and A_ROW_TILE == 128:
        if tokens > 16 and tokens <= 48:
            o_r_pad = _native_small_o_projection_a_groups_stage_nz(
                o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, group_begin, group_end
            )
        else:
            o_r_pad = _original_o_projection_a_groups_stage_nz(
                o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, A_ROW_TILE, group_begin, group_end
            )
    elif A_COL_TILE == 192:
        o_r_pad = _rectangle_o_projection_a_groups_stage_nz(
            o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, A_ROW_TILE, group_begin, group_end
        )
    else:
        o_r_pad = _original_o_projection_a_groups_stage_nz(
            o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, A_ROW_TILE, group_begin, group_end
        )
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def o_projection_a_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    A_ROW_TILE: pl.constexpr,
):
    """全八组兼容入口：与按组通知分支复用同一数学，不另维护计算版本。"""
    o_r_pad = o_projection_a_groups_stage_nz(
        o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, A_ROW_TILE, 0, O_GROUPS
    )
    return o_r_pad


@pl.jit.inline(auto_scope=False)
def o_projection_groups_quant_stage(
    o_r_pad: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32],
    o_r_i8_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.INT8]],
    act_scale_dq: pl.InOut[pl.Tensor[[O_GROUPS, SCALE_ROWS_DYN], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """Native整行BF16量化；原group/32行块owner和尾零合同保持。"""
    task_rows = (tokens + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE
    padded_rows = (tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE
    group_first = group_begin * task_rows
    group_limit = group_end * task_rows
    first = worker_id + ((group_first + WORKERS - 1 - worker_id) // WORKERS) * WORKERS
    for logical in pl.range(first, group_limit, WORKERS):
        group = logical // task_rows
        block = logical % task_rows
        start = block * QUANT_TASK_T_TILE
        for row in pl.range(start, pl.min(start + QUANT_TASK_T_TILE, tokens), QUANT_TOKEN_TILE):
            rows = pl.min(QUANT_TOKEN_TILE, tokens - row)
            value = pl.load(o_r_pad, [row, 0], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])
            rounded = pl.cast(pl.cast(value, pl.BF16, mode="rint"), pl.FP32)
            amax = pl.set_validshape(
                pl.row_max(pl.abs(rounded), pl.create_tile([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP32)), rows, 1
            )
            for source_group in pl.range(1, O_GROUPS):
                chunk = pl.load(
                    o_r_pad, [row, source_group * O_LORA], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA]
                )
                rounded_chunk = pl.cast(pl.cast(chunk, pl.BF16, mode="rint"), pl.FP32)
                chunk_max = pl.row_max(pl.abs(rounded_chunk), pl.create_tile([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP32))
                amax = pl.set_validshape(pl.maximum(amax, chunk_max), rows, 1)
            dequant = pl.set_validshape(pl.mul(amax, 1.0 / INT8_SCALE_MAX), rows, 1)
            denominator = pl.set_validshape(pl.add(amax, 0.0), rows, 1)
            for lane in pl.range(rows):
                if pl.read(denominator, [lane, 0]) == 0.0:
                    pl.write(denominator, [lane, 0], 1.0)
            # 从归约结果派生常量，保持其物理布局和尾行valid shape。
            numerator = pl.add(pl.mul(amax, 0.0), INT8_SCALE_MAX)
            multiplier = pl.div(numerator, denominator)
            source = pl.load(o_r_pad, [row, group * O_LORA], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])
            source_bf16 = pl.cast(pl.cast(source, pl.BF16, mode="rint"), pl.FP32)
            integer = pl.cast(pl.row_expand_mul(source_bf16, multiplier), pl.INT32, mode="rint")
            half = pl.cast(integer, pl.FP16, mode="round")
            result = pl.set_validshape(pl.cast(half, pl.INT8, mode="trunc"), rows, O_LORA)
            pl.store(result, [row, group * O_LORA], o_r_i8_pad)
            scale_row = pl.set_validshape(pl.reshape(dequant, [1, QUANT_TOKEN_TILE]), 1, rows)
            pl.store(scale_row, [group, row], act_scale_dq)
        if start + QUANT_TASK_T_TILE >= tokens:
            for row in pl.range(tokens, padded_rows, QUANT_TOKEN_TILE):
                zero = pl.cast(
                    pl.tile.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0), pl.INT8, mode="trunc"
                )
                pl.store(
                    pl.set_validshape(zero, pl.min(QUANT_TOKEN_TILE, padded_rows - row), O_LORA),
                    [row, group * O_LORA],
                    o_r_i8_pad,
                )
    return o_r_i8_pad, act_scale_dq


@pl.jit.inline(auto_scope=False)
def o_projection_group_quant_stage(
    o_r_pad: pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32],
    o_r_i8_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.INT8]],
    act_scale_dq: pl.InOut[pl.Tensor[[O_GROUPS, SCALE_ROWS_DYN], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    ROW_TILE: pl.constexpr,
):
    """全八组入口，与按组通知分支共享 amax、量化舍入和尾行清零数学。"""
    o_r_i8_pad, act_scale_dq = o_projection_groups_quant_stage(
        o_r_pad, o_r_i8_pad, act_scale_dq, tokens, worker_id, WORKERS, ROW_TILE, 0, O_GROUPS
    )
    return o_r_i8_pad, act_scale_dq
