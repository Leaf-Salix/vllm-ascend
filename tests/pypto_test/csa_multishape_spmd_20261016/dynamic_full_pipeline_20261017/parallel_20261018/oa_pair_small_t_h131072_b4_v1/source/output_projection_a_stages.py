# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""动态 CSA 的 O_A 分组投影与分组量化。

输入输出与阅读顺序
    AIC 消费 attention 的 packed heads，按八个 O group 独立做
    K4096→N1024，写 FP32[P_MM,8192]。输出组流水开启、OA_COLS=128、OB_ROWS=32 且
    设备实际 0<T<=32 时，使用 o_projection_a_pair_group_stage_nz；
    其它区间保留原独立列块 helper。AIV 再对每行每组 1024 列独立取
    amax，写 INT8[P_MM,8192] 和 FP32[8,P_SCALE]，不跨八组归约。
    wo_a 保持 Native BF16[8,4096,1024] 的实际 ND 或 NZ 存储；各入口
    声明所需布局，pair 专用于 NZ，不做格式转换或导入期设备探针。

动态轴、容量与别名
    tokens 是调用方从真实业务 Tensor 动态轴得到的设备 Scalar T，不从
    packed/workspace 容量反推。packed 是 BF16[PACK_TOTAL_ROWS,4096]，
    PACK_TOTAL_ROWS=8*P_PACK，组 g 的第 t 行位于 g*P_PACK+t，P_PACK>=T。
    FP32/INT8 工作区共享 P_MM 容量合同但不共享存储；P_SCALE 独立且>=T。
    O_B 使用的 ROW_TILE 为 32/96/128 之一，P_MM>=ceil(T/ROW_TILE)*ROW_TILE，
    并须足以保存实际 T 行。packed 总行数须整除 8，各视图连续且不得非法别名。
    所有容量轴均 pl.dynamic；固定八组、模型维度和 tile 不是 B/S 白名单。
    外层捕获前拒绝不一致容量/layout；不能让生产者跳写而消费者读取旧值。

分工、数学与尾行
    原路径以 (group,ceil(T/128) 行块,N 列块) 展平并按 logical%WORKERS
    分给 AIC，每块先 K256 matmul，再按原顺序追加 15 个 K256。
    小 T pair 路径专用于 24 AIC：组 g 的三个 pair 归 3g..3g+2，每核
    共用输入和 Left，分别累加两个 N128 输出；末两个 N128 块分别归
    (3g+3+s)%24，s 为 0/1。组内五 owner 互异，组 7 的 single 回绕到 0/1。
    pair 显式使用 Mat/Left/Right 与两个 FP32 Acc，每条链按 K128 顺序
    累加；single 保留原 K256 首块和后续流水。M128 是 tile 容量，所有
    GM 读写只覆盖实际 T 行；不把 tile 容量当作可访问的 GM 行数。
    量化以 (group,32 行块) 展平轮转 AIV，块内每 8 行流水；有效行数贯穿
    load、row_max、常数、div、cast 和 store，不读取 O_A 未写的 FP32 padding。
    每组最后一个活动 32 行块是唯一尾 owner，额外将 INT8 的 [T,P_USED) 写零，
    P_USED=ceil(T/ROW_TILE)*ROW_TILE；O_B 才能安全整块读取。T=0 不产生任务。
    量化保留 127/amax、amax*(1/127) 及 INT32 rint→FP16 round→INT8 trunc；
    反量化 scale 不能替换为 recip(量化乘数)，也不增加 BF16 往返舍入。

同步与生命周期
    原 groups helper 保留 logical%WORKERS；pair helper 使用与 cube/vector
    同步的五 owner 映射。各 helper 均不创建 spmd、task 或 notify/wait。
    O_A 先获取所消费 packed 行的全部 attention 生产者；每个 AIC 完成
    本核全部 O_A 组后才进入 O_B 等待。AIV 获取本组全部 O_A owner 后量化，
    真实 INT8、scale 与所需尾零完成后才发布。O_B 获取该组全部量化 owner
    后可读整 ROW_TILE。组间就绪次序可以变化，组内归约与尾零合同不变。
    外层负责业务 fence、缓存维护和阶段信号。函数返回 GM 别名不代表
    异步流水已完成，也不代表其它核完成；消费者读完前不得复用 GM。
    每个输出行列块唯一写；InOut 保留未写区，P_USED 以后的 INT8 和 T 以后的
    scale 不承诺初始化。BYPASS 策略如需使用，须在外层 spmd 直属设置。
    实际编译、精度与完整 L1 验证状态见对应测试档案，不由源码说明推定。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import INT8_AMAX_EPS, INT8_SCALE_MAX

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
        xa_first = pl.slice(
            o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
        )
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        # 首块matmul建立实际M行的accumulator布局；不能改成任意全128行零种子。
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
            )
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        # wo_a保留group轴，acc仍带该轴；assemble沿原映射写回有效M，不reshape整个GM。
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
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
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    """NZ O_A 的动态组区间；0≤begin≤end≤8，保持全局条带 owner 与原 K 链。"""
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    row_blocks = (tokens + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    column_blocks = O_LORA // A_COL_TILE
    group_first = group_begin * row_blocks * column_blocks
    group_limit = group_end * row_blocks * column_blocks
    # 不是每组重新从worker编号起算；保留全局logical%WORKERS才能保持负载
    # 和量化消费者的owner公式。空组或本核无任务时，首项≥上界，循环自然为空。
    first = worker_id + ((group_first + WORKERS - 1 - worker_id) // WORKERS) * WORKERS
    # NZ证明器不利用owner余数<WORKERS的区间事实；显式提供非负起点。
    # 合法输入下first本就非负，max不改变任何任务编号。
    for logical in pl.range(pl.max(0, first), group_limit, WORKERS):
        g = pl.max(0, logical // (row_blocks * column_blocks))
        pa_unit = logical % (row_blocks * column_blocks)
        pa_rb = pa_unit // column_blocks
        # row_blocks*column_blocks整除列块数，因此与pa_unit%column_blocks等价。
        # 除数直接写constexpr表达式：NZ证明器不沿Scalar别名追踪模数常量。
        nf = logical % (O_LORA // A_COL_TILE)
        pa_r0 = pa_rb * PROJ_A_ROW_TILE
        pa_rows = pl.min(PROJ_A_ROW_TILE, tokens - pa_r0)
        pa_src0 = g * pack_stride + pa_r0
        out_col_g = g * O_LORA
        # NZ访问需要编译器可证明的非负列偏移，合法worker的数学值不因此改变。
        # 只显式表达非负合同，不依赖max阻止优化；关键是上述constexpr模数。
        n0 = pl.max(0, nf) * A_COL_TILE
        xa_first = pl.slice(
            o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, 0], valid_shape=[pa_rows, A_K_TILE]
        )
        wa_first = wo_a[g : g + 1, 0:A_K_TILE, n0 : n0 + A_COL_TILE]
        acc_a = pl.matmul(xa_first, wa_first, out_dtype=pl.FP32)
        for kb in pl.pipeline(1, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [PROJ_A_ROW_TILE, A_K_TILE], [pa_src0, k0], valid_shape=[pa_rows, A_K_TILE]
            )
            wa_k_chunk = wo_a[g : g + 1, k0 : k0 + A_K_TILE, n0 : n0 + A_COL_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk)
        o_r_pad = pl.assemble(o_r_pad, acc_a, [pa_r0, out_col_g + n0])
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
):
    """原独立列块路径的全八组入口：复用 groups helper，不包含专用 pair 分工。"""
    o_r_pad = o_projection_a_groups_stage_nz(
        o_packed, wo_a, o_r_pad, tokens, worker_id, WORKERS, A_COL_TILE, 0, O_GROUPS
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
    """量化动态组区间；尾零属于该组的最后活动块，必须先于本组通知完成。"""
    task_rows = (tokens + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE
    padded_rows = (tokens + ROW_TILE - 1) // ROW_TILE * ROW_TILE
    # T=0时循环为空，不计算负尾owner；每个group的活动行及padding各只有一个写者。
    group_first = group_begin * task_rows
    group_limit = group_end * task_rows
    first = worker_id + ((group_first + WORKERS - 1 - worker_id) // WORKERS) * WORKERS
    for logical in pl.range(first, group_limit, WORKERS):
        g = logical // task_rows
        block = logical % task_rows
        col_g = g * O_LORA
        quant_start = block * QUANT_TASK_T_TILE
        # 保留原32行任务内部的8行双级流水，不跨组取amax；尾行不读FP32 pad。
        for qt in pl.pipeline(
            quant_start, pl.min(quant_start + QUANT_TASK_T_TILE, tokens), QUANT_TOKEN_TILE, stage=2
        ):
            rows = pl.min(QUANT_TOKEN_TILE, tokens - qt)
            oc_amax = pl.load(o_r_pad, [qt, col_g], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])
            reduce_tmp = pl.tile.create([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP32)
            g_row_max = pl.reshape(
                pl.set_validshape(pl.row_max(pl.abs(oc_amax), reduce_tmp), rows, 1), [1, QUANT_TOKEN_TILE]
            )
            # row_max转行后，所有常数与算术结果均保留[1,rows]，不能恢复成8个有效值。
            g_amax_floor = pl.set_validshape(
                pl.tile.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS), 1, rows
            )
            g_amax = pl.set_validshape(pl.maximum(g_amax_floor, g_row_max), 1, rows)
            g_scale_num = pl.set_validshape(
                pl.tile.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), 1, rows
            )
            g_sq_row = pl.set_validshape(pl.div(g_scale_num, g_amax), 1, rows)
            # 与canonical一致：dequant尺度为amax乘1/127，不改成recip(127/amax)。
            dequant_scale = pl.set_validshape(pl.mul(g_amax, 1.0 / INT8_SCALE_MAX), 1, rows)
            pl.store(dequant_scale, [g, qt], act_scale_dq)
            g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
            oc_q = pl.load(o_r_pad, [qt, col_g], [QUANT_TOKEN_TILE, O_LORA], valid_shape=[rows, O_LORA])
            oq_scaled = pl.set_validshape(pl.row_expand_mul(oc_q, g_sq_col), rows, O_LORA)
            oq_i32 = pl.set_validshape(pl.cast(oq_scaled, target_type=pl.INT32, mode="rint"), rows, O_LORA)
            oq_half = pl.set_validshape(pl.cast(oq_i32, target_type=pl.FP16, mode="round"), rows, O_LORA)
            oq_i8 = pl.set_validshape(pl.cast(oq_half, target_type=pl.INT8, mode="trunc"), rows, O_LORA)
            pl.store(oq_i8, [qt, col_g], o_r_i8_pad)
        # 只最后活动32行块有权填本组尾零。零填不读o_r_pad，不写scale尾部，
        # 并在同一阶段发布之前完成；O_B整RB读取不可仅依赖实际行的quant完成。
        if quant_start + QUANT_TASK_T_TILE >= tokens:
            for zt in pl.range(tokens, padded_rows, QUANT_TOKEN_TILE):
                zero_half = pl.tile.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
                zero_rows = pl.min(QUANT_TOKEN_TILE, padded_rows - zt)
                pl.store(pl.set_validshape(zero_i8, zero_rows, O_LORA), [zt, col_g], o_r_i8_pad)
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


@pl.jit.inline(auto_scope=False)
def o_projection_a_pair_group_stage_nz(
    o_packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, pl.NZ],
    o_r_pad: pl.InOut[pl.Tensor[[MM_ROWS_DYN, O_GROUPS * O_LORA], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    group_id: pl.Scalar[pl.INDEX],
):
    """仅输出组流水开启、N128、OB_ROWS=32 且 0<T<=32：每组三个 pair、两个 single。

    pair p 归 3*g+p，写列块 2p/2p+1；single s 归 (3*g+3+s)%24，写 6+s。
    本组五个 owner 互异，每核在同组只执行一种角色；外层完成本组全部
    本核写入后才发布。实际 GM 容量与有效行仍动态，不能按 M128 写满尾行。
    组间packed步长由Tensor求得，不沿用资源探针固定384；输入T可小于M128，
    valid_rows贯穿Tile的compact Acc/load/store，不能写实际GM的padding/guard。
    K1024普通range内部四K256局部Mat，只有两个FP32 Acc跨回边；每条链仍
    K128依次累加。MTE2/MTE1/M事件由后端保持，源码顺序不是异步保证。
    single保留现行Tensor matmul的首K256及后续15段pipeline，量化数学不动。
    WORKERS必须24；不含任务/通知，完整依赖由cube/vector两侧匹配公式维护。
    """
    pack_stride = pl.tensor.dim(o_packed, 0) // O_GROUPS
    g = pl.max(0, group_id)
    pair_first = g * 3
    if worker_id >= pair_first and worker_id < pair_first + 3:
        pair = pl.max(0, worker_id - pair_first)
        row_base = g * pack_stride
        n0 = pair * 256
        output_col = g * O_LORA + n0
        valid_rows = tokens
        group = g
        seed0 = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        seed1 = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        valid0 = pl.tile.set_validshape(seed0, valid_rows, 128)
        valid1 = pl.tile.set_validshape(seed1, valid_rows, 128)
        # 仅Acc跨回边，Mat不会被固定到循环入口地址再做非法Mat→Mat复制。
        for outer, (acc0, acc1) in pl.range(0, 4, init_values=(valid0, valid1)):
            kbase = outer * 1024
            # 加载K256块0；当前尚待消费的前一块仍活跃，形成两个L1窗口。
            a0 = pl.tile.load(o_packed, [row_base, kbase + 0], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w0a = pl.tile.load(wo_a, [group, kbase + 0, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w0b = pl.tile.load(wo_a, [group, kbase + 0, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            # 加载K256块1；当前尚待消费的前一块仍活跃，形成两个L1窗口。
            a1 = pl.tile.load(o_packed, [row_base, kbase + 256], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w1a = pl.tile.load(wo_a, [group, kbase + 256, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w1b = pl.tile.load(wo_a, [group, kbase + 256, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            # 消费块0的两个K128。第一次MMA后释放Right-a，第二次后释放Right-b；
            # 等待M真正完成由后端M→MTE1事件负责，不能直接按发射时间覆盖。
            w0a_flat = pl.tile.reshape(w0a, [256, 128])
            w0b_flat = pl.tile.reshape(w0b, [256, 128])
            l00 = pl.tile.extract(a0, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r00a = pl.tile.extract(w0a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            r00b = pl.tile.extract(w0b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            c0a_first = pl.tile.matmul_acc(acc0, l00, r00a, init_cond=(outer == 0))
            l01 = pl.tile.extract(a0, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r01a = pl.tile.extract(w0a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c0b_first = pl.tile.matmul_acc(acc1, l00, r00b, init_cond=(outer == 0))
            r01b = pl.tile.extract(w0b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c0a_last = pl.tile.matmul_acc(c0a_first, l01, r01a, init_cond=False)
            c0b_last = pl.tile.matmul_acc(c0b_first, l01, r01b, init_cond=False)
            # 加载K256块2；当前尚待消费的前一块仍活跃，形成两个L1窗口。
            a2 = pl.tile.load(o_packed, [row_base, kbase + 512], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w2a = pl.tile.load(wo_a, [group, kbase + 512, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w2b = pl.tile.load(wo_a, [group, kbase + 512, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            # 消费块1的两个K128。第一次MMA后释放Right-a，第二次后释放Right-b；
            # 等待M真正完成由后端M→MTE1事件负责，不能直接按发射时间覆盖。
            w1a_flat = pl.tile.reshape(w1a, [256, 128])
            w1b_flat = pl.tile.reshape(w1b, [256, 128])
            l10 = pl.tile.extract(a1, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r10a = pl.tile.extract(w1a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            r10b = pl.tile.extract(w1b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            c1a_first = pl.tile.matmul_acc(c0a_last, l10, r10a, init_cond=False)
            l11 = pl.tile.extract(a1, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r11a = pl.tile.extract(w1a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c1b_first = pl.tile.matmul_acc(c0b_last, l10, r10b, init_cond=False)
            r11b = pl.tile.extract(w1b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c1a_last = pl.tile.matmul_acc(c1a_first, l11, r11a, init_cond=False)
            c1b_last = pl.tile.matmul_acc(c1b_first, l11, r11b, init_cond=False)
            # 加载K256块3；当前尚待消费的前一块仍活跃，形成两个L1窗口。
            a3 = pl.tile.load(o_packed, [row_base, kbase + 768], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w3a = pl.tile.load(wo_a, [group, kbase + 768, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w3b = pl.tile.load(wo_a, [group, kbase + 768, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            # 消费块2的两个K128。第一次MMA后释放Right-a，第二次后释放Right-b；
            # 等待M真正完成由后端M→MTE1事件负责，不能直接按发射时间覆盖。
            w2a_flat = pl.tile.reshape(w2a, [256, 128])
            w2b_flat = pl.tile.reshape(w2b, [256, 128])
            l20 = pl.tile.extract(a2, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r20a = pl.tile.extract(w2a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            r20b = pl.tile.extract(w2b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            c2a_first = pl.tile.matmul_acc(c1a_last, l20, r20a, init_cond=False)
            l21 = pl.tile.extract(a2, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r21a = pl.tile.extract(w2a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c2b_first = pl.tile.matmul_acc(c1b_last, l20, r20b, init_cond=False)
            r21b = pl.tile.extract(w2b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c2a_last = pl.tile.matmul_acc(c2a_first, l21, r21a, init_cond=False)
            c2b_last = pl.tile.matmul_acc(c2b_first, l21, r21b, init_cond=False)
            # 消费块3的两个K128。第一次MMA后释放Right-a，第二次后释放Right-b；
            # 等待M真正完成由后端M→MTE1事件负责，不能直接按发射时间覆盖。
            w3a_flat = pl.tile.reshape(w3a, [256, 128])
            w3b_flat = pl.tile.reshape(w3b, [256, 128])
            l30 = pl.tile.extract(a3, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            r30a = pl.tile.extract(w3a_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            r30b = pl.tile.extract(w3b_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            c3a_first = pl.tile.matmul_acc(c2a_last, l30, r30a, init_cond=False)
            l31 = pl.tile.extract(a3, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            r31a = pl.tile.extract(w3a_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c3b_first = pl.tile.matmul_acc(c2b_last, l30, r30b, init_cond=False)
            r31b = pl.tile.extract(w3b_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            c3a_last = pl.tile.matmul_acc(c3a_first, l31, r31a, init_cond=False)
            c3b_last = pl.tile.matmul_acc(c3b_first, l31, r31b, init_cond=False)
            # 本轮四段全部完成后才回边；只携带两条Acc，不携带任何Mat。
            done0, done1 = pl.yield_(c3a_last, c3b_last)
        pl.tile.store(done0, [0, output_col], o_r_pad)
        pl.tile.store(done1, [0, output_col + 128], o_r_pad)

    # single交给下一组三核中的前两核，避免同组pair与single串行后才发布。
    # 该映射来自8组/24AIC结构，不依赖B/S。每核至多一个pair加一个single。
    single_first = ((g + 1) * 3) % WORKERS
    if worker_id >= single_first and worker_id < single_first + 2:
        single = pl.max(0, worker_id - single_first)
        single_n0 = 768 + single * 128
        single_src0 = g * pack_stride
        single_x_first = pl.slice(o_packed, [128, 256], [single_src0, 0], valid_shape=[tokens, 256])
        single_w_first = wo_a[g : g + 1, 0:256, single_n0 : single_n0 + 128]
        single_acc = pl.matmul(single_x_first, single_w_first, out_dtype=pl.FP32)
        for single_kb in pl.pipeline(1, 16, stage=2):
            single_k0 = single_kb * 256
            single_x = pl.slice(o_packed, [128, 256], [single_src0, single_k0], valid_shape=[tokens, 256])
            single_w = wo_a[g : g + 1, single_k0 : single_k0 + 256, single_n0 : single_n0 + 128]
            single_acc = pl.matmul_acc(single_acc, single_x, single_w)
        o_r_pad = pl.assemble(o_r_pad, single_acc, [0, g * O_LORA + single_n0])
    return o_r_pad
