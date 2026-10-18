# SPDX-License-Identifier: Apache-2.0
"""动态 CSA 大行区间 O_A：两个 M 行块共享同一份 K/N 权重。

整体思路和输入输出
    packed 是 BF16[8*P_PACK,4096]，第 g 组真实 T 行位于 g*P_PACK 开始；
    Native NZ BF16 权重为 [8,4096,1024]，输出 FP32[P_MM,8192] 布局不变。
    只适用于 128<T<=256 且原 OA_COLS=256 的区间。原实现是两份 M128、
    每份四个 N256；本实现是八个 N128，每个 N 同时维护两份独立 M 累加器。
    每组逻辑任务仍为八个，因此现有 OA→quant 的组 owner 和通知合同不变。

来自 hc_pre 的复用原则
    HC_pre 按工作量增大 M 以减少重复读权重；这里受 L0C 容量约束，改为
    两个 M128/N128 累加器共同消费每份权重。每个 K256 权重只进 L1 一次，
    同时供前128行和真实尾行使用；代价是 N256 拆为 N128，输入读取增多。
    权重逻辑请求128→64MiB，输入额外 T*256KiB；T=256时总逻辑字节不减。
    不能把逻辑字节变化当作 HBM 流量或性能收益，默认关闭，等待完整 L1 对照。
    原8个任务前4个处理满M128、后4个只处理尾M；按24条带时这两种工作量
    一直落在不同核上。新任务都含半列宽的满块和尾块，减少源级工作量差异；
    这只是分工推导，不能替代真实最慢核及整层耗时。

动态范围、分工与算术
    T、packed 容量和输出容量保留 dynamic；最后 M 的有效行数为 T-128。
    每组八个列块按 (g*8+n)%WORKERS 领取，group_begin/end来自原核内调度。
    第一版保留原N256路径的K64但MMA发射增加，完整对照只有持平/0.55%收益。
    本版每K256中使用两个K128子块，仍按K递增单链累加；硬件分块发生变化，
    不能从源码代数推断与原版逐bit相同，须重新通过原精度门槛。两个M的运算
    可以交错，不能把它们相加或改变其中一条 K 链。仅写真实 T 行。

片上流水和同步
    K256的两个输入块和一份权重各64KiB，外层显式预取下一块请求L1双缓冲；
    K128提取让两行块共享同一Right；显式交错只让两份32KiB Left/Right存活，
    避免内层pipeline同时要求四份Left。两个物理128x128 Acc合计128KiB；
    实际双缓冲、完整内存复用及事件顺序必须审计生成物。
    本helper不发布phase、不创建新SPMD。调用前外层已等所有packed写者，
    返回后仍由原OA组phase保护quant消费者。不能挪用无关阶段的空隙推算收益。
"""

import pypto.language as pl

PACK_ROWS = pl.dynamic("CSA_OA_PAIR_PACK_ROWS")
MM_ROWS = pl.dynamic("CSA_OA_PAIR_MM_ROWS")
M_ROWS = 128
N_COLS = 128
K_CHUNK = 256
K_INNER = 128
GROUPS = 8
K_FULL = 4096
N_FULL = 1024


@pl.jit.inline(auto_scope=False)
def o_projection_a_row_pair_groups_stage_nz(
    packed: pl.Tensor[[PACK_ROWS, K_FULL], pl.BF16],
    weight: pl.Tensor[[GROUPS, K_FULL, N_FULL], pl.BF16, pl.NZ],
    output: pl.InOut[pl.Tensor[[MM_ROWS, GROUPS * N_FULL], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    group_begin: pl.Scalar[pl.INDEX],
    group_end: pl.Scalar[pl.INDEX],
):
    pack_stride = pl.tensor.dim(packed, 0) // GROUPS
    tail_rows = tokens - M_ROWS
    group_first = group_begin * (N_FULL // N_COLS)
    group_limit = group_end * (N_FULL // N_COLS)
    first = worker + ((group_first + WORKERS - 1 - worker) // WORKERS) * WORKERS
    for logical in pl.range(pl.max(0, first), group_limit, WORKERS):
        group = pl.max(0, logical // (N_FULL // N_COLS))
        col = pl.max(0, logical % (N_FULL // N_COLS)) * N_COLS
        src_row = group * pack_stride
        dst_col = group * N_FULL + col
        a_first = pl.load(packed, [src_row, 0], [M_ROWS, K_CHUNK], target_memory=pl.MemorySpace.Mat)
        a_tail = pl.load(packed, [src_row + M_ROWS, 0], [M_ROWS, K_CHUNK],
                         valid_shape=[tail_rows, K_CHUNK], target_memory=pl.MemorySpace.Mat)
        b_3d = pl.load(weight, [group, 0, col], [1, K_CHUNK, N_COLS], target_memory=pl.MemorySpace.Mat)
        b_chunk = pl.tile.reshape(b_3d, [K_CHUNK, N_COLS])
        # 首K128以真实Left有效行创建Acc，避免全128行种子改变尾块紧凑L0C布局。
        # 第一份Right同时服务两个M；在第一MMA之后预取第二K128的Right。
        right_first = pl.tile.extract(b_chunk, 0, 0, [K_INNER, N_COLS], target_memory=pl.MemorySpace.Right)
        left_first = pl.tile.extract(a_first, 0, 0, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
        left_tail = pl.tile.extract(a_tail, 0, 0, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
        acc_first = pl.tile.matmul(left_first, right_first)
        right_second = pl.tile.extract(b_chunk, K_INNER, 0, [K_INNER, N_COLS], target_memory=pl.MemorySpace.Right)
        acc_tail = pl.tile.matmul(left_tail, right_first)
        left_second_first = pl.tile.extract(a_first, 0, K_INNER, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
        acc_first = pl.tile.matmul_acc(acc_first, left_second_first, right_second)
        left_second_tail = pl.tile.extract(a_tail, 0, K_INNER, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
        acc_tail = pl.tile.matmul_acc(acc_tail, left_second_tail, right_second)
        # 只预取GM→Mat，不让通用pipeline把整组Left/Right提取也提前一轮：
        # 首个自动stage2版本因此要求262144B Left，超过64KiB，未进入设备。
        # 三份Mat各保留current/next，L0仍仅按下面两M×两K128交错推进。
        a_current_first = pl.load(packed, [src_row, K_CHUNK], [M_ROWS, K_CHUNK], target_memory=pl.MemorySpace.Mat)
        a_current_tail = pl.load(packed, [src_row + M_ROWS, K_CHUNK], [M_ROWS, K_CHUNK],
                                valid_shape=[tail_rows, K_CHUNK], target_memory=pl.MemorySpace.Mat)
        b_current_3d = pl.load(weight, [group, K_CHUNK, col], [1, K_CHUNK, N_COLS], target_memory=pl.MemorySpace.Mat)
        for chunk in pl.range(1, K_FULL // K_CHUNK):
            if chunk + 1 < K_FULL // K_CHUNK:
                next_k = (chunk + 1) * K_CHUNK
                a_next_first = pl.load(packed, [src_row, next_k], [M_ROWS, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                a_next_tail = pl.load(packed, [src_row + M_ROWS, next_k], [M_ROWS, K_CHUNK],
                                     valid_shape=[tail_rows, K_CHUNK], target_memory=pl.MemorySpace.Mat)
                b_next_3d = pl.load(weight, [group, next_k, col], [1, K_CHUNK, N_COLS], target_memory=pl.MemorySpace.Mat)
            else:
                # 最后一轮不预读K4096；这些别名仅用于闭合循环状态，不再消费。
                a_next_first = a_current_first
                a_next_tail = a_current_tail
                b_next_3d = b_current_3d
            b_chunk = pl.tile.reshape(b_current_3d, [K_CHUNK, N_COLS])
            # 与首块同一交错，只把首K的初始化替换为继续累加。
            # 下一Left覆盖须等待此前MMA完成；由编译器生成真实事件，不手删wait。
            right_first = pl.tile.extract(b_chunk, 0, 0, [K_INNER, N_COLS], target_memory=pl.MemorySpace.Right)
            left_first = pl.tile.extract(a_current_first, 0, 0, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
            left_tail = pl.tile.extract(a_current_tail, 0, 0, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
            acc_first = pl.tile.matmul_acc(acc_first, left_first, right_first)
            right_second = pl.tile.extract(b_chunk, K_INNER, 0, [K_INNER, N_COLS], target_memory=pl.MemorySpace.Right)
            acc_tail = pl.tile.matmul_acc(acc_tail, left_tail, right_first)
            left_second_first = pl.tile.extract(a_current_first, 0, K_INNER, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
            acc_first = pl.tile.matmul_acc(acc_first, left_second_first, right_second)
            left_second_tail = pl.tile.extract(a_current_tail, 0, K_INNER, [M_ROWS, K_INNER], target_memory=pl.MemorySpace.Left)
            acc_tail = pl.tile.matmul_acc(acc_tail, left_second_tail, right_second)
            a_current_first = a_next_first
            a_current_tail = a_next_tail
            b_current_3d = b_next_3d
        pl.store(acc_first, [0, dst_col], output)
        pl.store(acc_tail, [M_ROWS, dst_col], output)
    return output
