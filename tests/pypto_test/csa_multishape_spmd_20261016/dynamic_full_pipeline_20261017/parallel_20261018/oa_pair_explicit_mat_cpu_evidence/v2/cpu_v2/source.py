# SPDX-License-Identifier: Apache-2.0
"""O_A双输出有限手工流水CPU探针v2；不是正式CSA、没有设备调用。

v1的Mat循环携带被生成A3不支持的Mat→Mat拷贝，本版仅两个Acc跨普通range。
每轮K1024是模型固定K维的数学分段，内部四K256；每K256两次K128，两个
N128输出各自保持0..3968的FP32累加次序。实际行数仍是动态valid_rows。

两个相邻Mat三元组(A/W0/W1)同时活跃，目标6×64KiB。计算旧组后再加载
下一组，最后一组本轮排空，没有Mat回边和外层pipeline自动复制。各K256
内部Left共用、Right在上一MMA之后替换，目标两个总Right槽，不是两路各
双缓冲。硬件异步和事件仍必须检查实际生成C++，源排序不是异步保证。

真实八组NZ根布局、lhs组容量步长384、结果行步长8192与旧探针一致；
valid_rows调用合同1..128。24pair仅覆盖每组前三对N128，singleton未实现，
因此不能称全输出正确或完整CSA。无AIV/信号/FFTS/主工作区或依赖改动。
"""

import pypto.language as pl


@pl.jit
def entry(
    lhs: pl.Tensor[[3072, 4096], pl.BF16],
    rhs: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.TensorLayout.NZ],
    output: pl.Out[pl.Tensor[[384, 8192], pl.FP32]],
    valid_rows: pl.Scalar[pl.INDEX],
):
    with pl.spmd(24, name_hint="oa_pair_four_k256"):
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
        # 仅Acc跨回边，Mat不会被固定到循环入口地址再做非法Mat→Mat复制。
        for outer, (acc0, acc1) in pl.range(0, 4, init_values=(valid0, valid1)):
            kbase = outer * 1024
            # 加载K256块0；当前尚待消费的前一块仍活跃，形成两个L1窗口。
            a0 = pl.tile.load(lhs, [row_base, kbase + 0], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w0a = pl.tile.load(rhs, [group, kbase + 0, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w0b = pl.tile.load(rhs, [group, kbase + 0, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            # 加载K256块1；当前尚待消费的前一块仍活跃，形成两个L1窗口。
            a1 = pl.tile.load(lhs, [row_base, kbase + 256], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w1a = pl.tile.load(rhs, [group, kbase + 256, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w1b = pl.tile.load(rhs, [group, kbase + 256, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
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
            a2 = pl.tile.load(lhs, [row_base, kbase + 512], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w2a = pl.tile.load(rhs, [group, kbase + 512, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w2b = pl.tile.load(rhs, [group, kbase + 512, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
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
            a3 = pl.tile.load(lhs, [row_base, kbase + 768], [128, 256], [valid_rows, 256], target_memory=pl.Mem.Mat)
            w3a = pl.tile.load(rhs, [group, kbase + 768, n0], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
            w3b = pl.tile.load(rhs, [group, kbase + 768, n0 + 128], [1, 256, 128], [1, 256, 128], target_memory=pl.Mem.Mat, cache=1)
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
        pl.tile.store(done0, [0, output_col], output)
        pl.tile.store(done1, [0, output_col + 128], output)
    return output
