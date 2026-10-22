# SPDX-License-Identifier: Apache-2.0
"""HCA 输出尾链候选：分组量化不变，O_B、反量化归约及 HC_post 共用 MIX 流水。

目的：让某个输出列块的最后一组整数结果就绪后立即收尾，不等所有列块完成。
Cube 依次计算 g=0..7 的 INT32 点积；两个 AIV 分别负责一半输出列，按原顺序
乘各组 activation scale、累加、乘 weight scale，再执行原 BF16 舍入及 HC_post。
保留原数学顺序，不合并量化尺度，不采用 FMA 或改变 FP32 归约树。

每个核组只占两个 GM 交接槽，AIC 用 FIX 发布；AIV 完成 MTE2 读取后确认回收。
硬件事件实现组内通知，不使用 Scalar 轮询或全核屏障。跨任务依赖仍由 MPMD
运行时负责。该交接在 A3 上仍经过 GM，不能称为消除了所有部分和读写流量。

候选只在冻结实验中启用。重点检查：取消分组 O_B 提前开工的代价，是否小于
归约/收尾流水的收益；较多 M 行块可能重复加载权重，必须看完整层配对实测。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.decode_o_proj import (
    D, O_GROUPS, O_GROUP_IN, O_LORA, T_PAD, PROJ_A_ROW_TILE,
    PROJ_A_MM_N_TILE, PROJ_A_LARGE_N_TILE,
    PROJ_B_SMALL_T_TILE, PROJ_B_MEDIUM_T_TILE, PROJ_B_MM_T_TILE,
    QUANT_TASK_T_TILE, QUANT_TOKEN_TILE, INT8_AMAX_EPS, INT8_SCALE_MAX, proj_a_mm,
)
from ..deepseek_v4_flash_csa.nz_mode import QUANT_WEIGHT_LAYOUT, WO_A_WEIGHT_LAYOUT

T_DYN = pl.dynamic("HCA_O_POST_TOKENS")
HC_MULT = 4
HC_DIM = HC_MULT * D
TOKEN_TILE = 8
OUTPUT_GROUPS = 24
OUTPUT_COLUMNS = 128
OUTPUT_K_TILE = 256
OUTPUT_SLOTS = 2

@pl.jit.inline
def _prepare_quantized_groups(
    o_packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    t_dim: pl.Scalar[pl.INDEX],
    heads_dep: pl.Scalar[pl.TASK_ID],
    ROW_TILE: pl.constexpr,
    A_COL_TILE: pl.constexpr,
    PIPELINE_OA: pl.constexpr,
):
    """HCA 的分组 O 投影任务组织，复用 CSA 矩阵乘，保留 HCA 收尾依赖。"""
    proj_a_rows = (t_dim + PROJ_A_ROW_TILE - 1) // PROJ_A_ROW_TILE
    proj_b_t_rows = (t_dim + ROW_TILE - 1) // ROW_TILE
    proj_b_padded_rows = proj_b_t_rows * ROW_TILE

    # 保持已有 HCA 性能版的分组量化和 INT32 部分和。
    # 与 Native 整个 token 共用一个量化标度的数值路径存在差别，不能宣称逐 bit 相同。
    o_r_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_GROUPS * O_LORA], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([O_GROUPS, T_PAD], dtype=pl.FP32)
    # 八组独立发布量化结果；下游统一等待真实生产者。
    quant_tids = pl.array.create(O_GROUPS, pl.TASK_ID)

    with pl.manual_scope():
        for g in pl.parallel(O_GROUPS):
            row_base_o = g * T_PAD
            out_col_g = g * O_LORA

            o_r_pad, pa_tid = proj_a_mm(
                o_packed, wo_a, o_r_pad, g, row_base_o, out_col_g,
                t_dim, proj_a_rows, heads_dep, A_COL_TILE, PIPELINE_OA,
            )

            col_g = g * O_LORA
            # 性能版按上游把 amax 与量化融进同一个 SPMD，并让每个 group 用自己的标度。
            # 已封存的精度版另起 oproj_token_scale 任务、跨全部 O_GROUPS 取同一个 amax，并多做
            # 两次 BF16 往返，那是为复刻 Native A3 dynamic_quant 的语义；本版本不要求
            # 与 Native 逐 bit 一致，于是省掉那遍全量扫描和往返。
            with pl.spmd(
                (t_dim + QUANT_TASK_T_TILE - 1) // QUANT_TASK_T_TILE,
                name_hint="quant",
                deps=[pa_tid],
                allow_early_resolve=True,
            ) as q_tid:
                quant_start = pl.tile.get_block_idx() * QUANT_TASK_T_TILE
                for qt in pl.pipeline(
                    quant_start, pl.min(quant_start + QUANT_TASK_T_TILE, t_dim), QUANT_TOKEN_TILE, stage=2
                ):
                    oc_amax = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    g_row_max = pl.reshape(pl.row_max(pl.abs(oc_amax)), [1, QUANT_TOKEN_TILE])
                    g_amax_floor = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                    g_amax = pl.maximum(g_amax_floor, g_row_max)
                    g_scale_num = pl.full([1, QUANT_TOKEN_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                    g_sq_row = pl.div(g_scale_num, g_amax)
                    act_scale_dq[g : g + 1, qt : qt + QUANT_TOKEN_TILE] = pl.mul(g_amax, 1.0 / INT8_SCALE_MAX)
                    g_sq_col = pl.reshape(g_sq_row, [QUANT_TOKEN_TILE, 1])
                    oc_q = o_r_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA]
                    oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
                    oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
                    oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
                    oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc")
                    o_r_i8_pad[qt : qt + QUANT_TOKEN_TILE, col_g : col_g + O_LORA] = oq_i8
                # 清零最后一个有效 O_B 行块的内部量化尾部，不扩大最终输出写区。
                if quant_start + QUANT_TASK_T_TILE >= t_dim:
                    for zt in pl.range(t_dim, proj_b_padded_rows, QUANT_TOKEN_TILE):
                        zero_half = pl.full([QUANT_TOKEN_TILE, O_LORA], dtype=pl.FP16, value=0.0)
                        zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc")
                        zero_rows = pl.min(QUANT_TOKEN_TILE, proj_b_padded_rows - zt)
                        o_r_i8_pad = pl.assemble(o_r_i8_pad, pl.set_validshape(zero_i8, zero_rows, O_LORA), [zt, col_g])

            quant_tids[g] = q_tid

    quant_ready = pl.system.task_dummy(deps=[quant_tids[i] for i in range(O_GROUPS)])
    return o_r_i8_pad, act_scale_dq, quant_ready


@pl.jit.inline(auto_scope=False)
def _output_pipeline(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
    MM_ROWS: pl.constexpr,
    MM_COLS: pl.constexpr,
):
    tokens = pl.tensor.dim(residual, 0)
    quantized, scales, ready = _prepare_quantized_groups(
        packed, wo_a, wo_b, tokens, heads_dep, MM_ROWS, MM_COLS, PIPELINE_OA=True,
    )
    residual_flat = pl.reshape(residual, [tokens, HC_DIM])
    output_flat = pl.reshape(output, [tokens, HC_DIM])
    row_blocks = (tokens + MM_ROWS - 1) // MM_ROWS
    units = row_blocks * (D // OUTPUT_COLUMNS)
    # T>=1时至少32个输出列块，始终足够24组核；使用固定正步长保留NZ地址证明。
    # 槽号只属于一个逻辑 worker；它循环处理多个列块时，离开上一块前排空确认。
    transfer = pl.create_tensor([OUTPUT_GROUPS * OUTPUT_SLOTS * MM_ROWS, OUTPUT_COLUMNS], dtype=pl.INT32)
    ffts = pl.create_tensor([256], dtype=pl.INT64)
    with pl.spmd(OUTPUT_GROUPS, name_hint="hca_output_pipeline", deps=[ready], allow_early_resolve=True):
        worker = pl.tile.get_block_idx()
        pl.system.set_ffts(ffts)
        for unit in pl.range(worker, units, OUTPUT_GROUPS):
            row0 = unit // (D // OUTPUT_COLUMNS) * MM_ROWS
            col0 = pl.max(unit % (D // OUTPUT_COLUMNS), 0) * OUTPUT_COLUMNS
            rows = pl.min(MM_ROWS, tokens - row0)
            for group in pl.range(O_GROUPS):
                if group >= OUTPUT_SLOTS:
                    # 确认两个 AIV 都读完上一代槽内容，才能再次由 FIX 写入。
                    pl.system.sync_wait(1, pipe=pl.PipeType.FIX, core_type=pl.KernelType.AIC)
                activation = pl.load(
                    quantized, [row0, group * O_LORA], [MM_ROWS, O_LORA],
                    valid_shape=[rows, O_LORA], target_memory=pl.MemorySpace.Mat,
                )
                seed_storage = pl.create_tile(
                    [MM_ROWS, OUTPUT_COLUMNS], dtype=pl.INT32,
                    target_memory=pl.MemorySpace.Acc, compact=True,
                )
                seed = pl.tile.set_validshape(seed_storage, rows, OUTPUT_COLUMNS)
                for kb, (acc,) in pl.pipeline(
                    0, O_LORA // OUTPUT_K_TILE, stage=2, init_values=(seed,),
                ):
                    k0 = kb * OUTPUT_K_TILE
                    wk0 = pl.max(group, 0) * O_LORA + k0
                    weight = pl.load(
                        wo_b, [wk0, pl.max(0, col0)], [OUTPUT_K_TILE, OUTPUT_COLUMNS],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    act = pl.tile.slice(activation, [MM_ROWS, OUTPUT_K_TILE], [0, k0])
                    next_acc = pl.tile.matmul_acc(acc, act, weight, init_cond=(kb == 0))
                    completed = pl.yield_(next_acc)
                slot_row = (worker * OUTPUT_SLOTS + group % OUTPUT_SLOTS) * MM_ROWS
                pl.store(completed, [slot_row, 0], transfer)
                pl.system.sync_set(0, pipe=pl.PipeType.FIX, ffts_mode=2, core_type=pl.KernelType.AIC)
            for drain in pl.range(OUTPUT_SLOTS):
                pl.system.sync_wait(1, pipe=pl.PipeType.FIX, core_type=pl.KernelType.AIC)

        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            pl.system.set_ffts(ffts)
            lane_col = lane * (OUTPUT_COLUMNS // 2)
            for vec_unit in pl.range(worker, units, OUTPUT_GROUPS):
                vec_row0 = vec_unit // (D // OUTPUT_COLUMNS) * MM_ROWS
                vec_col0 = vec_unit % (D // OUTPUT_COLUMNS) * OUTPUT_COLUMNS + lane_col
                vec_rows = pl.min(MM_ROWS, tokens - vec_row0)
                initial = pl.tile.full([MM_ROWS, OUTPUT_COLUMNS // 2], dtype=pl.FP32, value=0.0)
                for vec_group, (running,) in pl.range(O_GROUPS, init_values=(initial,)):
                    pl.system.sync_wait(0, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                    vec_slot = (worker * OUTPUT_SLOTS + vec_group % OUTPUT_SLOTS) * MM_ROWS
                    partial = pl.load(
                        transfer, [vec_slot, lane_col], [MM_ROWS, OUTPUT_COLUMNS // 2],
                        valid_shape=[vec_rows, OUTPUT_COLUMNS // 2],
                    )
                    # 发布读完成而不是算完；Cube 可继续下一组，Vector 保留自己的 UB 副本。
                    pl.system.sync_set(1, pipe=pl.PipeType.MTE2, ffts_mode=2, core_type=pl.KernelType.AIV)
                    scale = pl.reshape(pl.load(
                        scales, [vec_group, vec_row0], [1, MM_ROWS], valid_shape=[1, vec_rows],
                    ), [MM_ROWS, 1])
                    contribution = pl.row_expand_mul(pl.cast(partial, pl.FP32), scale)
                    next_sum = pl.add(running, contribution)
                    accumulated = pl.yield_(next_sum)

                # 同一输出列块最后一个 g 就绪即进入 HC_post；不等待其他核组的输出。
                weight_scale = pl.reshape(pl.load(
                    wo_scale, [vec_col0], [OUTPUT_COLUMNS // 2],
                ), [1, OUTPUT_COLUMNS // 2])
                for rt in pl.range(0, vec_rows, TOKEN_TILE):
                    row = vec_row0 + rt
                    valid = pl.min(TOKEN_TILE, vec_rows - rt)
                    part = pl.tile.slice(accumulated, [TOKEN_TILE, OUTPUT_COLUMNS // 2], [rt, 0])
                    attention = pl.col_expand_mul(part, weight_scale)
                    attention = pl.cast(pl.cast(attention, pl.BF16, mode="rint"), pl.FP32)
                    post_rows = pl.load(post, [row, 0], [TOKEN_TILE, 8], valid_shape=[valid, HC_MULT])
                    comb_rows = pl.load(comb, [row, 0], [TOKEN_TILE, HC_MULT * HC_MULT],
                                        valid_shape=[valid, HC_MULT * HC_MULT])
                    # 保留四路残差和门值，四个输出依旧按 0→1→2→3 做 FP32 加法。
                    residual_0 = pl.cast(pl.load(
                        residual_flat, [row, vec_col0], [TOKEN_TILE, OUTPUT_COLUMNS // 2],
                        valid_shape=[valid, OUTPUT_COLUMNS // 2],
                    ), pl.FP32)
                    residual_1 = pl.cast(pl.load(
                        residual_flat, [row, D + vec_col0], [TOKEN_TILE, OUTPUT_COLUMNS // 2],
                        valid_shape=[valid, OUTPUT_COLUMNS // 2],
                    ), pl.FP32)
                    residual_2 = pl.cast(pl.load(
                        residual_flat, [row, 2 * D + vec_col0], [TOKEN_TILE, OUTPUT_COLUMNS // 2],
                        valid_shape=[valid, OUTPUT_COLUMNS // 2],
                    ), pl.FP32)
                    residual_3 = pl.cast(pl.load(
                        residual_flat, [row, 3 * D + vec_col0], [TOKEN_TILE, OUTPUT_COLUMNS // 2],
                        valid_shape=[valid, OUTPUT_COLUMNS // 2],
                    ), pl.FP32)
                    post_columns = pl.transpose(post_rows, axis1=0, axis2=1)
                    comb_columns = pl.transpose(comb_rows, axis1=0, axis2=1)
                    for out_h in pl.unroll(HC_MULT):
                        post_weight = pl.reshape(pl.tile.slice(
                            post_columns, [1, TOKEN_TILE], [out_h, 0],
                        ), [TOKEN_TILE, 1])
                        value = pl.row_expand_mul(attention, post_weight)
                        coefficient_0 = pl.reshape(pl.tile.slice(
                            comb_columns, [1, TOKEN_TILE], [out_h, 0],
                        ), [TOKEN_TILE, 1])
                        value = pl.add(value, pl.row_expand_mul(residual_0, coefficient_0))
                        coefficient_1 = pl.reshape(pl.tile.slice(
                            comb_columns, [1, TOKEN_TILE], [HC_MULT + out_h, 0],
                        ), [TOKEN_TILE, 1])
                        value = pl.add(value, pl.row_expand_mul(residual_1, coefficient_1))
                        coefficient_2 = pl.reshape(pl.tile.slice(
                            comb_columns, [1, TOKEN_TILE], [2 * HC_MULT + out_h, 0],
                        ), [TOKEN_TILE, 1])
                        value = pl.add(value, pl.row_expand_mul(residual_2, coefficient_2))
                        coefficient_3 = pl.reshape(pl.tile.slice(
                            comb_columns, [1, TOKEN_TILE], [3 * HC_MULT + out_h, 0],
                        ), [TOKEN_TILE, 1])
                        value = pl.add(value, pl.row_expand_mul(residual_3, coefficient_3))
                        result = pl.cast(value, pl.BF16, mode="rint")
                        pl.store(pl.set_validshape(result, valid, OUTPUT_COLUMNS // 2),
                                 [row, out_h * D + vec_col0], output_flat)
    return output


@pl.jit.inline
def o_proj_hc_post(
    packed: pl.Tensor[[O_GROUPS * T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    wo_b: pl.Tensor[[O_GROUPS * O_LORA, D], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wo_scale: pl.Tensor[[D], pl.FP32],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    heads_dep: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(residual, 0)
    if tokens <= PROJ_B_SMALL_T_TILE:
        output = _output_pipeline(packed, wo_a, wo_b, wo_scale, residual, post, comb, output,
                                  heads_dep, PROJ_B_SMALL_T_TILE, PROJ_A_MM_N_TILE)
    elif tokens <= PROJ_B_MEDIUM_T_TILE:
        output = _output_pipeline(packed, wo_a, wo_b, wo_scale, residual, post, comb, output,
                                  heads_dep, PROJ_B_MEDIUM_T_TILE, PROJ_A_MM_N_TILE)
    else:
        output = _output_pipeline(packed, wo_a, wo_b, wo_scale, residual, post, comb, output,
                                  heads_dep, PROJ_B_MM_T_TILE, PROJ_A_LARGE_N_TILE)
    return output
