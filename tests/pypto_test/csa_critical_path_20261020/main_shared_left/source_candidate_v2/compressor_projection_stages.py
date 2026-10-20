# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""动态 CSA 的两路压缩投影：主投影共享Left，保留双组Mat和双Right。

本候选总览
    主M64/N64投影仍按两组K512预取。每组两个K256输入只搬入Left
    一次，同时服务value/gate两条独立链；Right按KV0/Gate0/KV1/Gate1
    交错恢复，保持最多两个32KiB槽。每条输出的原K顺序不变。
    显式compact Acc及x_rows保留动态尾行。Indexer不改。完整生成
    地址、事件与设备精度/完整收益尚待验证，正式ops未修改。

输入输出、布局与动态容量
    x 为归一化后的 ND BF16[T,4096]，不要求 T=B*S，也不读取请求 metadata。
    每个 token 独立投影出重叠压缩窗口使用的 value 和 score：主 Compressor
    的 wkv/wgate 为 ND BF16[1024,4096]，两份输出为 FP32[PAD_MAIN,1024]；
    Indexer Compressor 对应为 ND BF16[256,4096]、FP32[PAD_INDEXER,256]。
    两张权重均按 [输出列,K] 存放，matmul 使用 b_trans=True，不引入 NZ 转换。
    T、PAD_MAIN、PAD_INDEXER 均保留 pl.dynamic；前者决定实际工作量，后两者
    只是物理容量，分别至少 align_up(T,64)、align_up(T,16)。输出只承诺前 T 行
    的数值，padding 不保证初始化。value/score 工作区与输入、权重不得非法别名。

算法、分核与阅读顺序
    先读 compressor_project_stage 的 M64/N64 主投影，再读
    indexer_compressor_project_stage 的 M16/N32 投影。两者均按行块优先、
    列块其次把二维 M/N 网格编号为 idx；worker_id+k*WORKERS 领取完整网格点，
    每个点独占 value/score 的同一输出块。每核完成全部 K4096 的八个 K512
    分片，不做跨核 split-K、atomic 累加或归约重排。主投影显式保持两K512预取；Indexer仍用stage=2。
    worker_id 是 AIC 角色内的零起点编号，要求 WORKERS>0 且
    0<=worker_id<WORKERS。T=0 或核数大于工作块数时，空核不读权重/输入。

为何两段不合并成一种 accumulator 初始化
    主投影 M64 跨四个16行分形，本候选显式设置compact=True及x_rows
    有效行，再由首个init_cond建立数值；不能用固定64行距的普通零种子。
    Indexer 的 M16 保留原 matmul_acc(init_cond=kb==0)。两段 K 顺序和尾块
    valid_shape 均来自 canonical 多 task 数学，未借迁移改变舍入或数值策略。

阶段依赖、工作区与边界
    本文件只提供 inline helper，不创建 spmd/task，不通知或等待，也不读写
    state/cache。外层先等 x_norm 可见，再运行投影；pool 消费该行所需的所有
    列块前，须等待相应 AIC 生产者。helper 返回两个 Tensor 只表达 GM 别名，
    不代表其他核已完成或数据已经发布。原 late_dep/chain_dep 的顺序责任也在
    外层保留，不能因本文件没有 TASK_ID 就默认删除。每个工作区覆盖整份 T-pad，
    本实现不循环复用槽位，因而不引入尚未建立的消费者 ACK。
    最后 M 块只加载 valid_rows；源码写回视图仍保留原完整 M 块，因此调用方
    必须提供相应 pad 容量。tile 是算法分块，不是 batch/sequence 测试点白名单。
    当前只迁移投影；请求池化、RMS/RoPE、压缩 cache 与整层同步另行组装验证。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.config import FLASH as M

# 只取模型固有维度；不导入含设备探针的 nz_mode，也不读取 DECODE_BATCH/SEQ。
D = M.hidden_size
CMP_OUT_DIM = 2 * M.head_dim
INDEXER_OUT_DIM = 2 * M.index_head_dim
K_TILE = 512
CMP_M_TILE = 64
CMP_N_TILE = 64
SHARED_L0_K = 256
INDEXER_M_TILE = 16
INDEXER_N_TILE = 32

T_DYN = pl.dynamic("CSA_COMPRESSOR_PROJECTION_T")
MAIN_PAD_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_PROJECTION_MAIN_PAD_ROWS")
INDEXER_PAD_ROWS_DYN = pl.dynamic("CSA_COMPRESSOR_PROJECTION_INDEXER_PAD_ROWS")


@pl.jit.inline(auto_scope=False)
def compressor_project_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[CMP_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[CMP_OUT_DIM, D], pl.BF16],
    kv_proj_pad: pl.Tensor[[MAIN_PAD_ROWS_DYN, CMP_OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[MAIN_PAD_ROWS_DYN, CMP_OUT_DIM], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """主投影显式K512双组预取；共享Left并保持每条K256链和动态compact行距。"""
    bs = pl.tensor.dim(x, 0)
    t_matmul = ((bs + CMP_M_TILE - 1) // CMP_M_TILE) * CMP_M_TILE
    for idx in pl.range(worker_id, t_matmul * CMP_OUT_DIM // (CMP_M_TILE * CMP_N_TILE), WORKERS):
        global_row0 = (idx // (CMP_OUT_DIM // CMP_N_TILE)) * CMP_M_TILE
        o0 = (idx % (CMP_OUT_DIM // CMP_N_TILE)) * CMP_N_TILE
        x_rows = pl.min(CMP_M_TILE, bs - global_row0)
        # 显式描述后端原来生成的compact Acc合同，不以全64行种子推断尾行距。
        # create只分配句柄，首个K256的init_cond=True建立数值；不会读取未初始化Acc。
        kv_seed = pl.tile.create([CMP_M_TILE, CMP_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
        score_seed = pl.tile.create([CMP_M_TILE, CMP_N_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, compact=True)
        kv_acc = pl.set_validshape(kv_seed, x_rows, CMP_N_TILE)
        score_acc = pl.set_validshape(score_seed, x_rows, CMP_N_TILE)
        # 普通range保留原一次预取两个K512的窗口。不能套pipeline把后续Right
        # 再提前到当前MMA之前，否则Right可能超过64KiB。固定2块只属数学K分块，
        # 不是固定B/T/历史；GM读与最终写的有效行始终是本轮x_rows。
        for kb in pl.range(0, D // K_TILE, 2):
            k0 = kb * K_TILE
            k1 = (kb + 1) * K_TILE
            x_l1_0 = pl.load(x, [global_row0, k0], [CMP_M_TILE, K_TILE], valid_shape=[x_rows, K_TILE], target_memory=pl.MemorySpace.Mat)
            kv_l1_0 = pl.load(wkv, [o0, k0], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            gate_l1_0 = pl.load(wgate, [o0, k0], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            x_l1_1 = pl.load(x, [global_row0, k1], [CMP_M_TILE, K_TILE], valid_shape=[x_rows, K_TILE], target_memory=pl.MemorySpace.Mat)
            kv_l1_1 = pl.load(wkv, [o0, k1], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)
            gate_l1_1 = pl.load(wgate, [o0, k1], [CMP_N_TILE, K_TILE], target_memory=pl.MemorySpace.Mat)

            # 第1个K512：两份32KiB Left只恢复一次，供value/gate共用。
            # 每条输出仍按K0、K256顺序累加；两个输出间交错不改变各自数学链。
            left0_0 = pl.tile.extract(x_l1_0, 0, 0, [CMP_M_TILE, SHARED_L0_K], target_memory=pl.MemorySpace.Left)
            left0_1 = pl.tile.extract(x_l1_0, 0, SHARED_L0_K, [CMP_M_TILE, SHARED_L0_K], target_memory=pl.MemorySpace.Left)
            kv_trans0 = pl.tile.transpose_view(kv_l1_0)
            gate_trans0 = pl.tile.transpose_view(gate_l1_0)
            kv_right0_0 = pl.tile.extract(kv_trans0, 0, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            gate_right0_0 = pl.tile.extract(gate_trans0, 0, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            kv_acc = pl.tile.matmul_acc(kv_acc, left0_0, kv_right0_0, init_cond=kb == 0)
            # 第一个value已发出后才恢复第二份value Right，避免三份Right同时活跃。
            # 原value槽复用必须等待其M读取结束；gate槽独立，编译器保留必要事件。
            kv_right0_1 = pl.tile.extract(kv_trans0, SHARED_L0_K, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            score_acc = pl.tile.matmul_acc(score_acc, left0_0, gate_right0_0, init_cond=kb == 0)
            gate_right0_1 = pl.tile.extract(gate_trans0, SHARED_L0_K, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            kv_acc = pl.tile.matmul_acc(kv_acc, left0_1, kv_right0_1)
            score_acc = pl.tile.matmul_acc(score_acc, left0_1, gate_right0_1)

            # 第2个K512：两份32KiB Left只恢复一次，供value/gate共用。
            # 每条输出仍按K0、K256顺序累加；两个输出间交错不改变各自数学链。
            left1_0 = pl.tile.extract(x_l1_1, 0, 0, [CMP_M_TILE, SHARED_L0_K], target_memory=pl.MemorySpace.Left)
            left1_1 = pl.tile.extract(x_l1_1, 0, SHARED_L0_K, [CMP_M_TILE, SHARED_L0_K], target_memory=pl.MemorySpace.Left)
            kv_trans1 = pl.tile.transpose_view(kv_l1_1)
            gate_trans1 = pl.tile.transpose_view(gate_l1_1)
            kv_right1_0 = pl.tile.extract(kv_trans1, 0, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            gate_right1_0 = pl.tile.extract(gate_trans1, 0, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            kv_acc = pl.tile.matmul_acc(kv_acc, left1_0, kv_right1_0, init_cond=False)
            # 第一个value已发出后才恢复第二份value Right，避免三份Right同时活跃。
            # 原value槽复用必须等待其M读取结束；gate槽独立，编译器保留必要事件。
            kv_right1_1 = pl.tile.extract(kv_trans1, SHARED_L0_K, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            score_acc = pl.tile.matmul_acc(score_acc, left1_0, gate_right1_0, init_cond=False)
            gate_right1_1 = pl.tile.extract(gate_trans1, SHARED_L0_K, 0, [SHARED_L0_K, CMP_N_TILE], target_memory=pl.MemorySpace.Right)
            kv_acc = pl.tile.matmul_acc(kv_acc, left1_1, kv_right1_1)
            score_acc = pl.tile.matmul_acc(score_acc, left1_1, gate_right1_1)
        # Tile自身保留动态valid和compact信息，仅写真实行；容量仍是原pad64。
        kv_proj_pad = pl.tile.store(kv_acc, [global_row0, o0], kv_proj_pad)
        score_proj_pad = pl.tile.store(score_acc, [global_row0, o0], score_proj_pad)
    return kv_proj_pad, score_proj_pad


@pl.jit.inline(auto_scope=False)
def indexer_compressor_project_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[INDEXER_OUT_DIM, D], pl.BF16],
    kv_proj_pad: pl.Tensor[[INDEXER_PAD_ROWS_DYN, INDEXER_OUT_DIM], pl.FP32],
    score_proj_pad: pl.Tensor[[INDEXER_PAD_ROWS_DYN, INDEXER_OUT_DIM], pl.FP32],
    worker_id: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """Indexer压缩投影；M16/N32、BF16转置权重与原init_cond累加顺序不变。"""
    bs = pl.tensor.dim(x, 0)
    t_matmul = ((bs + INDEXER_M_TILE - 1) // INDEXER_M_TILE) * INDEXER_M_TILE
    # 每个M16块有8个列块；worker条带覆盖全网格，尾部不足16行也只领取一次。
    for idx in pl.range(worker_id, t_matmul * INDEXER_OUT_DIM // (INDEXER_M_TILE * INDEXER_N_TILE), WORKERS):
        global_row0 = (idx // (INDEXER_OUT_DIM // INDEXER_N_TILE)) * INDEXER_M_TILE
        o0 = (idx % (INDEXER_OUT_DIM // INDEXER_N_TILE)) * INDEXER_N_TILE
        kv_acc = pl.create_tensor([INDEXER_M_TILE, INDEXER_N_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([INDEXER_M_TILE, INDEXER_N_TILE], dtype=pl.FP32)
        # M16只跨一个分形，保留原init_cond；不能照抄主M64首块策略来改数学。
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(INDEXER_M_TILE, bs - global_row0)
            x_tile = pl.slice(x, [INDEXER_M_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            wkv_tile = wkv[o0 : o0 + INDEXER_N_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + INDEXER_N_TILE, k0 : k0 + K_TILE]
            kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(kb == 0))
            score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(kb == 0))
        # 工作区pad16承接原M16写回视图；pad内容不作为后续池化的有效输入。
        kv_proj_pad[global_row0 : global_row0 + INDEXER_M_TILE, o0 : o0 + INDEXER_N_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + INDEXER_M_TILE, o0 : o0 + INDEXER_N_TILE] = score_acc
    return kv_proj_pad, score_proj_pad
