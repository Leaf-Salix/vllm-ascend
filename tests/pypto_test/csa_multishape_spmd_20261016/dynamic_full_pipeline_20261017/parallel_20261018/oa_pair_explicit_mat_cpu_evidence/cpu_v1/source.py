# SPDX-License-Identifier: Apache-2.0
"""O_A 双输出的独立 CPU 生命周期探针；不是完整 CSA，也不做设备测量。

输入沿用旧 oa_pair_source_v1 的真实八组 BF16 NZ 权重根视图；lhs 的组间
容量步长384、结果行步长8192与旧探针一致，valid_rows 是实际有效行的
动态 Scalar（调用合同1..128）。这里只覆盖每组前三对N128，未包括最后
两个singleton，不能把本文件当成全输出或正式动态CSA入口。

设计差异：不使用任何pl.pipeline。序言准备当前K256的三份Mat，普通range
提前加载下一组并携带它跨回边，末轮单独剥离，目标Mat峰值6×64KiB。
每轮两个K128，两条独立FP32 Acc按原K顺序更新；Left供两路N输出共用。
Right在上一消费者MMA后才产生替代值，希望只需总共两份32KiB，而不是
每路Right各自再被内外pipeline复制。实际物理槽和事件必须以CPU产物审计
为准；源码最后使用不能代替M→MTE1的真实完成依赖。

只有一个24AIC的spmd；无AIV、GM信号或FFTS协议改动。这里验证Mat回边及
L0生命周期能否被后端表达，不声称整层精度或性能收益。
"""

import pypto.language as pl


@pl.jit
def entry(
    lhs: pl.Tensor[[3072, 4096], pl.BF16],
    rhs: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorLayout.NZ],
    output: pl.Out[pl.Tensor[[384, 8192], pl.FP32]],
    valid_rows: pl.Scalar[pl.INDEX],
):
    with pl.spmd(24, name_hint="oa_pair_explicit_mat"):
        worker = pl.tile.get_block_idx()
        group = worker // 3
        pair = worker % 3
        row_base = group * 384
        n0 = pair * 256
        output_col = group * 1024 + n0
        seed0 = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        seed1 = pl.tile.create([128, 128], dtype=pl.FP32, target_memory=pl.Mem.Acc, compact=True)
        valid0 = pl.tile.set_validshape(seed0, valid_rows, 128)
        valid1 = pl.tile.set_validshape(seed1, valid_rows, 128)

        # 序言：只有第0个K256；所有load使用相同的valid_rows表达式，防止
        # 旧v4遇到的动态valid_shape/compact类型在循环回边不一致。
        a_seed = pl.tile.load(lhs, [row_base, 0], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
        w0_seed = pl.tile.load(rhs, [group, 0, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
        w1_seed = pl.tile.load(rhs, [group, 0, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)

        # 当前Mat直到最后一次extract才结束使用，下一组三份Mat提前创建：
        # 以真实活跃关系请求两个L1窗口，不由pipeline再复制所有L0操作数。
        # 两条Acc每轮均直线更新，不产生旧v5的“只更新一条Acc”的if-phi。
        for outer, (a_cur, w0_cur, w1_cur, acc0, acc1) in pl.range(
            0, 15, init_values=(a_seed, w0_seed, w1_seed, valid0, valid1)
        ):
            next_k = (outer + 1) * 256
            a_next = pl.tile.load(lhs, [row_base, next_k], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w0_next = pl.tile.load(rhs, [group, next_k, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w1_next = pl.tile.load(rhs, [group, next_k, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            # tile.extract只接受二维Mat；NZ单例group轴只reshape，不转权重。
            w0_flat = pl.tile.reshape(w0_cur, [256, 128])
            w1_flat = pl.tile.reshape(w1_cur, [256, 128])

            left0 = pl.tile.extract(a_cur, 0, 0, [128, 128], target_memory=pl.Mem.Left)
            right0a = pl.tile.extract(w0_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            right0b = pl.tile.extract(w1_flat, 0, 0, [128, 128], target_memory=pl.Mem.Right)
            acc0_first = pl.tile.matmul_acc(acc0, left0, right0a, init_cond=(outer == 0))
            # 此处right0a逻辑死，right0b仍活；left0仍需用于另一输出。
            # 下一Left/Right在MMA0之后准备，目标是与另一条MMA交错。
            left1 = pl.tile.extract(a_cur, 0, 128, [128, 128], target_memory=pl.Mem.Left)
            right1a = pl.tile.extract(w0_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            acc1_first = pl.tile.matmul_acc(acc1, left0, right0b, init_cond=(outer == 0))
            # right0b到此才死；目标重用其槽，不能提前产生第三份总Right。
            right1b = pl.tile.extract(w1_flat, 128, 0, [128, 128], target_memory=pl.Mem.Right)
            acc0_last = pl.tile.matmul_acc(acc0_first, left1, right1a, init_cond=False)
            acc1_last = pl.tile.matmul_acc(acc1_first, left1, right1b, init_cond=False)
            last_a, last_w0, last_w1, done0, done1 = pl.yield_(a_next, w0_next, w1_next, acc0_last, acc1_last)

        # 最后K256独立消费，不加载K4096以外的数据。两条Acc沿既有顺序继续，
        # 尾声不条件性选择更新哪个Acc，避免非法Acc→Acc回边复制。
        tail_w0 = pl.tile.reshape(last_w0, [256, 128])
        tail_w1 = pl.tile.reshape(last_w1, [256, 128])
        tail_l0 = pl.tile.extract(last_a, 0, 0, [128, 128], target_memory=pl.Mem.Left)
        tail_r0a = pl.tile.extract(tail_w0, 0, 0, [128, 128], target_memory=pl.Mem.Right)
        tail_r0b = pl.tile.extract(tail_w1, 0, 0, [128, 128], target_memory=pl.Mem.Right)
        tail_acc0_first = pl.tile.matmul_acc(done0, tail_l0, tail_r0a, init_cond=False)
        tail_l1 = pl.tile.extract(last_a, 0, 128, [128, 128], target_memory=pl.Mem.Left)
        tail_r1a = pl.tile.extract(tail_w0, 128, 0, [128, 128], target_memory=pl.Mem.Right)
        tail_acc1_first = pl.tile.matmul_acc(done1, tail_l0, tail_r0b, init_cond=False)
        tail_r1b = pl.tile.extract(tail_w1, 128, 0, [128, 128], target_memory=pl.Mem.Right)
        final0 = pl.tile.matmul_acc(tail_acc0_first, tail_l1, tail_r1a, init_cond=False)
        final1 = pl.tile.matmul_acc(tail_acc1_first, tail_l1, tail_r1b, init_cond=False)
        pl.tile.store(final0, [0, output_col], output)
        pl.tile.store(final1, [0, output_col + 128], output)
    return output
