# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 Flash 独立 HC_pre，完整实现位于本文件。

从本仓库 CSA 版本完整搬入后，在本文件优化分块与任务划分。
原始算法来自 pypto-lib 的同名实现；运行时不导入 CSA/HCA 算子或配置。
包含 BF16 加宽、RMS 统计、投影、门控、Sinkhorn 和残差混合，
不包含 mixed 之后的 input_layernorm。T 为动态 token 数，即 B×S。

数据与输出：
    x 为 [T, 4, 4096] 的四路残差，展平后每个 token 有 K=16384 个元素。
    hc_fn 为 [24, 16384] 的投影权重；24 列结果依次用于 pre 的 4 个门值、
    post 的 4 个门值和 comb 的 4×4 矩阵。三组分别使用 hc_scale[0/1/2]。
    mixed 为 [T, 4096]，是四路输入按 pre 门值加权后的 BF16 结果；
    post 为 [T, 4] 的 FP32 门值；comb_out 为 [T, 4, 4] 的 FP32 混合矩阵。

执行顺序：
    1. 将投影的 K 轴拆段，分配给多个核组；每组两个 AIV 转换输入并统计
       平方和，AIC 做该段矩阵乘，分别把 RMS 部分和、投影部分和写入 GM。
    2. 所有核组的 AIC/AIV 完成 GM 数据交接并会合，之后才能跨 K 分段归约。
    3. 全体 AIV 重新分工：一部分做 Sinkhorn 并写 comb，另一部分做门控及
       残差混合，写 post/mixed；小 T 再切 D 轴。本阶段没有新的矩阵乘任务。

这里的 for 是 PyPTO DSL 控制结构：spmd 分配核组，split_aiv 选择组内 AIV，
range 表达设备循环，pipeline 表达核内软件流水。它们不会逐次返回 Python
发起算子调用。各组使用相同代码，但循环起点、次数和等待时间可以不同。
"""

import pypto.language as pl

# 正式 DeepSeek V4 Flash 的 HC 配置；独立入口不依赖 CSA 配置模块。
D = 4096  # 每一路残差的特征宽度。
HC_MULT = 4  # 残差路数，也是 comb 方阵的边长。
MIX_HC = (2 + HC_MULT) * HC_MULT  # 4 个 pre + 4 个 post + 16 个 comb logit。
HC_DIM = HC_MULT * D  # 投影及 RMS 归约的完整 K 轴：16384。
HC_DIM_INV = 1.0 / HC_DIM  # 将平方和转换为均方值。
HC_SINKHORN_ITER = 20  # 首轮行/列归一化加后续 19 轮迭代。
HC_EPS = 1e-6  # pre 门值及 Sinkhorn 归一化使用的小量。
NORM_EPS = 1e-6  # RMS 的均方值在开方前加的小量。

# T 保持为动态维度；下方选择编译期分块方案，不把接口限制为固定 BS/S。
TOKENS = pl.dynamic("HC_PRE_TOKENS")

# 编排读取 tensor 描述符中的 T 选择分块，六个编译分支每次只执行其中一个。
SMALL_T_MAX = 32
MEDIUM48_T_MAX = 48
MEDIUM96_T_MAX = 96
# 测试根据此表核对实际任务名，同时要求每次仅执行选中分支的 AIC/AIV 对。
HC_PRE_VARIANTS = (
    (SMALL_T_MAX, "hc_small", 32, 1536),
    (MEDIUM48_T_MAX, "hc_medium48", 48, 1024),
    (MEDIUM96_T_MAX - 1, "hc_medium96", 32, 2048),
    (MEDIUM96_T_MAX, "hc_exact96", 32, 2048),
    (144, "hc_large144", 48, 2048),
    (None, "hc_large", 64, 2048),
)

# 投影按 LINEAR_T_TILE 切 token；两个 AIV 各转换半块行，再合并交给 Cube。
# 尾段先把实际 token 均分到全部核组，再用较小的物理 tile 处理各组的连续区间。
MIX_PAD = 32  # 24 列投影补齐至 Cube N 轴和 Vector 行的 32 列。
HC_PAD = 8  # pre/post 合并为 8 个 FP32 值，恰好占一行 32B。
T_TILE = 8  # RMS/门控物理 tile 的行数；实际工作行数由每组区间及 valid_rows 决定。
STAGE1_K_TILE = 1024  # T96 内部供数块；AIV/AIC 通过两个交接槽重叠推进。
D_TILE = 512  # D轴分配的最小单位；标量路径合并本核的连续区间或处理整行D。


# 编译时内联到下面的 hc_pre；这个辅助函数本身不增加一次独立 kernel 派发。
# auto_scope=False 保留显式 spmd/split_aiv 对核内执行范围与分工的描述。
@pl.jit.inline(auto_scope=False)
def _hc_pre_variant(
    x: pl.Tensor[[TOKENS, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    mixed: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[TOKENS, HC_MULT], pl.FP32]],
    comb_out: pl.Out[pl.Tensor[[TOKENS, HC_MULT, HC_MULT], pl.FP32]],
    LINEAR_T_TILE: pl.constexpr,
    LINEAR_K_PER_SPLIT: pl.constexpr,
    LINEAR_OK: pl.constexpr,
    RMS_CHUNK: pl.constexpr,
    SLOTS: pl.constexpr,
    TAIL_MAX_ROWS: pl.constexpr,
    TASK_NAME: pl.constexpr,
    STATIC_T: pl.constexpr,
    RMS_OK: pl.constexpr,
    SCALAR_MIX: pl.constexpr,
    STREAM_K: pl.constexpr,
):
    """按编译期参数生成一个完整的混合核分支。

    LINEAR_T_TILE：每个投影任务处理的 token 行数，两个 AIV 各处理一半。
    LINEAR_K_PER_SPLIT：投影的 K 分段宽度，最后一段可含补零列。
    LINEAR_OK：K 分段数量；各分段结果保留到 GM，屏障后再统一归约。
    RMS_CHUNK：整块供数分支的平方和归约宽度，控制 UB 临时占用。
        小档/大档流式路径复用每个K512输入块，中档按各自供数宽度计算。
    RMS_OK：GM中平方和的分段数。T49～95与大档为8份；精确T96保留16份。
    STREAM_K：逻辑段内供数宽度。小档/大档K512、中档K1024，经两槽交接给AIC。
        等于逻辑K段宽度时使用整块路径；精确T96走上方独立的Tile实现。
    SLOTS：组内 AIV→AIC 数据交接通道的缓冲槽数，与下面 pipeline 的
        stage 各自作用于不同位置，均不改变核组数量。
    TAIL_MAX_ROWS：每次尾段处理的有效行数上限，独立于 8 行对齐的 RMS/门控 tile。
    TASK_NAME：生成任务的名称提示，供编译产物及 profiling 识别分档。
    STATIC_T：0 使用动态 T，96 为独立的 T=96 特化；仅精确匹配时启用。
    SCALAR_MIX：T33～96及T>144分支启用设备标量门值。
        T33～48仍按token对/D分片分配AIV，但一次加载本核负责的连续D区间；
        T49～96与T>144每核独占完整token对，直接处理整行D。两种路径都只
        循环真实token行，从行主序sigmoid tile读取四个FP32门值，使用两级流水。
        不从Host取输入值，不改变输入格式、门值公式或混合加法顺序。
        其他分档仍使用D512向量广播；是否启用由各分支的完整L1对照决定。
    """
    # shape 读取与 reshape 描述逻辑视图，不需要把 x 的数值读回 Host。
    # 后续用显式偏移访问四路输入；这些 reshape 不做整张 Tensor 的搬运。
    tokens = pl.tensor.dim(x, 0)
    if STATIC_T > 0:
        tokens = STATIC_T
    x_flat = pl.reshape(x, [tokens, HC_DIM])
    mixed_view = pl.reshape(mixed, [tokens, D])
    t_dim = tokens
    # 尾段按 8 行计数；投影只在临时空间上把 T 向 LINEAR_T_TILE 补齐。
    # 输入、输出仍保留真实 T，边界由后续 valid_shape/fillpad 处理。
    t_linear = ((tokens + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    comb = pl.reshape(comb_out, [tokens, HC_MULT * HC_MULT])
    post_flat = pl.reshape(post, [1, tokens * HC_MULT])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])
    hc_scale_2d = pl.reshape(hc_scale, [1, 3])
    # GM 中分别按投影段数 LINEAR_OK 和平方和段数 RMS_OK 保存部分结果。
    # 每个 (split, token块) 都有独立写入位置；这里不通过 atomic 累加。
    # create_tensor 只分配临时 Tensor，不表示已将内容清零；生产循环负责写入。
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    rms_partials = pl.create_tensor([RMS_OK * t_linear, 1], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    # 从运行时查询本次设备的可用 MIX 核组数。在当前 A3 上，每组是 1 AIC+2 AIV。
    # 这不是 batch size，也不是写死的 24；下面的硬件 syncall 要求全量参与，
    # 因此启动宽度随设备查询结果确定，不能仅按 linear_units 随意缩小。
    clusters = pl.system.available_cluster_count()
    linear_workers = clusters
    # 尾段以两行作为最小分配单位：post 每行 16B，两行合计 32B。
    # 各组连续输出区间从偶数行开始，避免不同核向同一 32B 块写回。
    # 商和余数在 SPMD 外只计算一次；不能按 8 行块数限制参与尾段的核组。
    tail_pairs = (tokens + 1) // 2
    # 小 T 的 token 对少于核组数，仅切 token 仍有空核；多余 AIV 分担混合的 D 轴。
    # Sinkhorn 每个 token 独立且每行输出64B，可以按单行分；post 每行16B，
    # 混合仍以两行为单位，只有负责 D 起点的 AIV 写该 token 对的 post。
    # 这里的公共商/余数在编排计算，单个 AIV 只需解出自己的 token 与 D 区间。
    sink_workers = pl.min(tokens, clusters)
    sink_rows_per_worker = tokens // sink_workers
    sink_extra_rows = tokens % sink_workers
    mix_workers = pl.min(2 * clusters - sink_workers, tail_pairs * (D // D_TILE))
    mix_workers_per_pair = pl.max(1, mix_workers // tail_pairs)
    mix_extra_pairs = mix_workers % tail_pairs
    mix_extra_prefix = mix_extra_pairs * (mix_workers_per_pair + 1)
    mix_pairs_per_worker = tail_pairs // mix_workers
    mix_extra_workers = tail_pairs % mix_workers
    # 同一 GM 缓冲的归约视图：屏障之后，尾段按列取出一个 token 块的所有 K 分段。
    parts_2d = pl.reshape(mixes_partials, [LINEAR_OK, t_linear * MIX_PAD])
    rms_2d = pl.reshape(rms_partials, [RMS_OK, t_linear])

    # 【核组级调度】一次 SPMD 派发覆盖 clusters 个逻辑 block，worker 是各组
    # 自己的编号 [0, clusters)，不是 Host 串行执行这个循环，也不是固定物理核号。
    # 循环体会提取成混合 InCore 任务；组内 AIC/AIV 各执行适合自身的指令。
    # sync_start=True 要求调度器把这批 block 作为整体启动，使它们能同时驻留并
    # 到达下方硬件屏障；它不保证各核指令逐拍对齐，也不代替中途的 syncall。
    # cross_core_slot 指定组内交接通道的环形缓冲深度：生产者提交后由消费者
    # 读取并释放槽位，避免覆盖未消费的数据；增加槽数会增加缓冲资源需求。
    for worker in pl.spmd(clusters, name_hint=TASK_NAME, sync_start=True, optimizations=[pl.cross_core_slot(slot_num=SLOTS)]):
        # 【第一阶段：跨核组分配投影/RMS】每组处理 worker、worker+clusters、
        # worker+2*clusters ...，只取小于 linear_units 的任务，不要求工作量相等。
        # T49～95采用M32/K2048双槽，T80为24份任务；精确T96也为24份任务，
        # 但显式控制L0双缓冲与同一FP32累加器，并保留原来的16份RMS部分和。
        # 大档用M48/M64增加每组token行数，减少重复读权重；小档保持原分块。
        for task in pl.range(worker, linear_units, linear_workers):
            # task 将 token 块与 K 分段展平；除法恢复 token 块，取模恢复 K 分段。
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            split = task % LINEAR_OK
            kbase = split * LINEAR_K_PER_SPLIT
            kcols = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - kbase)
            partial_row = split * t_linear + t0

            if LINEAR_K_PER_SPLIT == 1536 and STREAM_K == 512:
                # 显式维护同一L0C累加器；末K段只有1024列，只发送两个K512块。
                # RMS仍按K512顺序累加，输出保持11份部分和，完整屏障不变。
                small_acc = pl.create_tile([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                for small_init_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    small_rms_total = pl.tile.full([1, LINEAR_T_TILE // 2], dtype=pl.FP32, value=0.0)
                for small_kk in pl.pipeline((kcols + 511) // 512, stage=2):
                    small_kc = kbase + small_kk * 512
                    # 所有有效K段宽度均为512的倍数，不读取零列或半个K512块。
                    small_weight = pl.load(hc_fn, [0, small_kc], [MIX_PAD, 512], valid_shape=[MIX_HC, 512], target_memory=pl.MemorySpace.Mat)
                    small_weight_t = pl.tile.transpose_view(small_weight)
                    for small_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        small_row = t0 + small_lane * (LINEAR_T_TILE // 2)
                        small_rows = pl.max(0, pl.min(LINEAR_T_TILE // 2, t_dim - small_row))
                        small_input = pl.load(x_flat, [small_row, small_kc], [LINEAR_T_TILE // 2, 512], valid_shape=[small_rows, 512], target_memory=pl.MemorySpace.Vec)
                        small_fp32 = pl.cast(small_input, pl.FP32)
                        small_clean = pl.fillpad(pl.set_validshape(small_fp32, small_rows, 512), pad_value=pl.PadValue.zero)
                        small_full = pl.set_validshape(small_clean, LINEAR_T_TILE // 2, 512)
                        small_gathered = pl.aic_gather(small_full)
                        small_rms_tmp = pl.create_tile([LINEAR_T_TILE // 2, 512], dtype=pl.FP32)
                        small_sq = pl.tile.row_sum(pl.mul(small_full, small_full), small_rms_tmp)
                        small_rms_total = pl.add(small_rms_total, pl.reshape(small_sq, [1, LINEAR_T_TILE // 2]))
                    # L0 K256两级流水共用同一个Acc，只在第一块初始化。
                    for small_l0k in pl.pipeline(0, 512, 256, stage=2):
                        small_left = pl.tile.extract(small_gathered, 0, small_l0k, [LINEAR_T_TILE, 256], target_memory=pl.MemorySpace.Left)
                        small_right = pl.tile.extract(small_weight_t, small_l0k, 0, [256, MIX_PAD], target_memory=pl.MemorySpace.Right)
                        small_acc = pl.tile.matmul_acc(small_acc, small_left, small_right, init_cond=(small_kk * 512 + small_l0k == 0))
                pl.store(small_acc, [partial_row, 0], mixes_partials)
                for small_store_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    small_rp = partial_row + small_store_lane * (LINEAR_T_TILE // 2)
                    pl.store(pl.reshape(small_rms_total, [LINEAR_T_TILE // 2, 1]), [small_rp, 0], rms_partials)
            elif STATIC_T == 96:
                # 24 个核组各负责一个 M32/K2048 逻辑块。沿 K 分成两次供数，
                # 两个槽使 AIV 的下一块加载/转换与 AIC 的当前块矩阵乘重叠。
                # RMS 复用刚转换的 FP32 子块，各子块分别落 GM，尾段统一归约。
                # 此处使用普通 range；DB 来自外层 cross_core_slot(2) 的双槽交接，
                # AIV 可准备下一 K1024 块，AIC 同时计算当前块，不重复申请循环流水缓冲。
                stream_product = pl.create_tile([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
                for kk in pl.range(LINEAR_K_PER_SPLIT // STAGE1_K_TILE):
                    kc = kbase + kk * STAGE1_K_TILE
                    weight_chunk = pl.load(hc_fn, [0, kc], [MIX_PAD, STAGE1_K_TILE], valid_shape=[MIX_HC, STAGE1_K_TILE], target_memory=pl.MemorySpace.Mat)
                    weight_transposed = pl.tile.transpose_view(weight_chunk)
                    for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                        lane_row = t0 + lane * (LINEAR_T_TILE // 2)
                        stream_source = pl.load(x_flat, [lane_row, kc], [LINEAR_T_TILE // 2, STAGE1_K_TILE], target_memory=pl.MemorySpace.Vec)
                        stream_widened = pl.cast(stream_source, pl.FP32)
                        gathered_chunk = pl.aic_gather(stream_widened)
                        stream_reduce_tmp = pl.create_tile([LINEAR_T_TILE // 2, STAGE1_K_TILE], dtype=pl.FP32)
                        stream_sq = pl.tile.row_sum(pl.mul(stream_widened, stream_widened), stream_reduce_tmp)
                        stream_rms_row = (split * (LINEAR_K_PER_SPLIT // STAGE1_K_TILE) + kk) * t_linear + lane_row
                        pl.store(stream_sq, [stream_rms_row, 0], rms_partials)
                    # 【显式交错L0双缓冲】K1024内四个K256块保持顺序累加。
                    # 先准备0/1块；算完0后准备2，再算1；准备3后依次算2/3。
                    # 两个相邻块交错存活，使MTE1能与上一块MMA并行；每组仍只需
                    # 两份32KiB Left/Right。只有每段首块需要判断是否初始化Acc。
                    left0 = pl.tile.extract(gathered_chunk, 0, 0, [LINEAR_T_TILE, 256], target_memory=pl.MemorySpace.Left)
                    right0 = pl.tile.extract(weight_transposed, 0, 0, [256, MIX_PAD], target_memory=pl.MemorySpace.Right)
                    left1 = pl.tile.extract(gathered_chunk, 0, 256, [LINEAR_T_TILE, 256], target_memory=pl.MemorySpace.Left)
                    right1 = pl.tile.extract(weight_transposed, 256, 0, [256, MIX_PAD], target_memory=pl.MemorySpace.Right)
                    stream_product = pl.tile.matmul_acc(stream_product, left0, right0, init_cond=(kk == 0))
                    left2 = pl.tile.extract(gathered_chunk, 0, 512, [LINEAR_T_TILE, 256], target_memory=pl.MemorySpace.Left)
                    right2 = pl.tile.extract(weight_transposed, 512, 0, [256, MIX_PAD], target_memory=pl.MemorySpace.Right)
                    stream_product = pl.tile.matmul_acc(stream_product, left1, right1, init_cond=False)
                    left3 = pl.tile.extract(gathered_chunk, 0, 768, [LINEAR_T_TILE, 256], target_memory=pl.MemorySpace.Left)
                    right3 = pl.tile.extract(weight_transposed, 768, 0, [256, MIX_PAD], target_memory=pl.MemorySpace.Right)
                    stream_product = pl.tile.matmul_acc(stream_product, left2, right2, init_cond=False)
                    stream_product = pl.tile.matmul_acc(stream_product, left3, right3, init_cond=False)
                pl.store(stream_product, [partial_row, 0], mixes_partials)
            elif STREAM_K < LINEAR_K_PER_SPLIT:
                # 每个核组完成一个完整K分段，内部按STREAM_K双槽供数。
                # 两个AIV各自保留RMS累加值，段末只写一次，不扩大尾段归约的段数。
                mid_stream_product = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
                for rms_init_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    mid_stream_rms_total = pl.full([1, LINEAR_T_TILE // 2], dtype=pl.FP32, value=0.0)
                # 【大档K段内流水】四个K512块用两级pipeline，权重L1与AIV输入
                # 分别准备双缓冲；AIV→AIC仍使用独立的两槽FIFO。
                # M48/M64的FIFO分别占192/256KiB，双权重均占128KiB，
                # L1总计320/384KiB。编译器为相邻迭代安排搬运与计算的依赖；
                # 这不保证完全重叠，实际收益以完整L1 incore首尾为准。
                # RMS累加保持循环携带依赖，四次K512的FP32加法顺序不重排。
                if STREAM_K == 512:
                    for kk in pl.pipeline(LINEAR_K_PER_SPLIT // STREAM_K, stage=2):
                        kc = kbase + kk * STREAM_K
                        kvalid = pl.max(0, pl.min(STREAM_K, HC_DIM - kc))
                        mid_weight_chunk = pl.slice(hc_fn, [MIX_PAD, STREAM_K], [0, kc], valid_shape=[MIX_HC, kvalid], pad_value=pl.PadValue.zero)
                        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            lane_row = t0 + lane * (LINEAR_T_TILE // 2)
                            lane_rows = pl.max(0, pl.min(LINEAR_T_TILE // 2, t_dim - lane_row))
                            mid_stream_source = pl.slice(x_flat, [LINEAR_T_TILE // 2, STREAM_K], [lane_row, kc], valid_shape=[lane_rows, kvalid])
                            mid_stream_widened = pl.cast(mid_stream_source, pl.FP32)
                            mid_stream_clean = pl.fillpad(pl.set_validshape(mid_stream_widened, lane_rows, kvalid), pad_value=pl.PadValue.zero)
                            mid_stream_full = pl.set_validshape(mid_stream_clean, LINEAR_T_TILE // 2, STREAM_K)
                            mid_gathered_chunk = pl.aic_gather(mid_stream_full)
                            mid_stream_sq = pl.reshape(pl.row_sum(pl.mul(mid_stream_full, mid_stream_full)), [1, LINEAR_T_TILE // 2])
                            mid_stream_rms_total = pl.add(mid_stream_rms_total, mid_stream_sq)
                        if kk == 0:
                            mid_stream_product = pl.matmul(mid_gathered_chunk, mid_weight_chunk, b_trans=True)
                        else:
                            mid_stream_product = pl.matmul_acc(mid_stream_product, mid_gathered_chunk, mid_weight_chunk, b_trans=True)
                else:
                    for kk in pl.range(LINEAR_K_PER_SPLIT // STREAM_K):
                        kc = kbase + kk * STREAM_K
                        kvalid = pl.max(0, pl.min(STREAM_K, HC_DIM - kc))
                        mid_weight_chunk = pl.slice(hc_fn, [MIX_PAD, STREAM_K], [0, kc], valid_shape=[MIX_HC, kvalid], pad_value=pl.PadValue.zero)
                        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                            lane_row = t0 + lane * (LINEAR_T_TILE // 2)
                            lane_rows = pl.max(0, pl.min(LINEAR_T_TILE // 2, t_dim - lane_row))
                            mid_stream_source = pl.slice(x_flat, [LINEAR_T_TILE // 2, STREAM_K], [lane_row, kc], valid_shape=[lane_rows, kvalid])
                            mid_stream_widened = pl.cast(mid_stream_source, pl.FP32)
                            mid_stream_clean = pl.fillpad(pl.set_validshape(mid_stream_widened, lane_rows, kvalid), pad_value=pl.PadValue.zero)
                            mid_stream_full = pl.set_validshape(mid_stream_clean, LINEAR_T_TILE // 2, STREAM_K)
                            mid_gathered_chunk = pl.aic_gather(mid_stream_full)
                            mid_stream_sq = pl.reshape(pl.row_sum(pl.mul(mid_stream_full, mid_stream_full)), [1, LINEAR_T_TILE // 2])
                            mid_stream_rms_total = pl.add(mid_stream_rms_total, mid_stream_sq)
                        if kk == 0:
                            mid_stream_product = pl.matmul(mid_gathered_chunk, mid_weight_chunk, b_trans=True)
                        else:
                            mid_stream_product = pl.matmul_acc(mid_stream_product, mid_gathered_chunk, mid_weight_chunk, b_trans=True)
                mixes_partials[partial_row:partial_row + LINEAR_T_TILE, 0:MIX_PAD] = mid_stream_product
                for rms_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    rp = partial_row + rms_lane * (LINEAR_T_TILE // 2)
                    rms_partials[rp:rp + LINEAR_T_TILE // 2, 0:1] = pl.reshape(mid_stream_rms_total, [LINEAR_T_TILE // 2, 1])
            else:
                # 【组内双 AIV 数据分工】lane 是 AIV 的本地编号 0/1；两路并行执行，
                # 并非同一个 AIV 先跑 lane0 再跑 lane1。UP_DOWN 表示按行交接给 AIC。
                # 本实现显式按半块设置 shape 和 lane_row：两路各负责当前投影
                # 块的一半 token 行，并加各自的 t0；精确 T96 已由上方专用分支处理。
                for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    lane_row = t0 + lane * (LINEAR_T_TILE // 2)
                    lane_rows = pl.max(0, pl.min(LINEAR_T_TILE // 2, t_dim - lane_row))
                    source = pl.slice(x_flat, [LINEAR_T_TILE // 2, LINEAR_K_PER_SPLIT], [lane_row, kbase], valid_shape=[lane_rows, kcols])
                    fp32 = pl.cast(source, pl.FP32)
                    # 先将无效行/列填零，再把有效形状恢复为固定 tile 大小。
                    # 这样 Cube 可以按整块计算，补齐区域不会污染投影或平方和。
                    clean = pl.fillpad(pl.set_validshape(fp32, lane_rows, kcols), pad_value=pl.PadValue.zero)
                    whole = pl.set_validshape(clean, LINEAR_T_TILE // 2, LINEAR_K_PER_SPLIT)
                    # 明确标记 AIV→AIC 的数据边界：两个半块按行合成完整投影输入。
                    # 编译器生成数据通道和通知/等待；这是组内数据交接，不是 Host gather。
                    gathered = pl.aic_gather(whole)

                    # 【AIV 上的 RMS 子块归约】先交出投影输入，再读取较小的 K 子块
                    # 计算平方和，以缩短大块 FP32 数据与平方临时值同时存活的时间。
                    # AIC 收到 gathered 后可推进矩阵乘，AIV 可继续做 RMS；实际重叠
                    # 由生成代码的数据依赖、交接槽和硬件流水决定，不以 Python 行序串行化。
                    local_sum = pl.full([1, LINEAR_T_TILE // 2], dtype=pl.FP32, value=0.0)
                    # rk 只遍历本 K 分段内部：每次 RMS_CHUNK 列，不增加核组或任务。
                    # local_sum 是循环携带的 FP32 累加值，
                    # 表示每个 token 在本 K 分段的平方和。这里只用普通 range，
                    # 没有显式申请 db 循环那样的两级软件流水。
                    for rk in pl.range(LINEAR_K_PER_SPLIT // RMS_CHUNK):
                        rcol = rk * RMS_CHUNK
                        # 最后一个 K 分段可能不足整块，rvalid=0 的子块也用补零参与。
                        rvalid = pl.max(0, pl.min(RMS_CHUNK, HC_DIM - kbase - rcol))
                        xr = pl.slice(x_flat, [LINEAR_T_TILE // 2, RMS_CHUNK], [lane_row, kbase + rcol], valid_shape=[lane_rows, rvalid])
                        xr32 = pl.cast(xr, pl.FP32)
                        xr_clean = pl.fillpad(pl.set_validshape(xr32, lane_rows, rvalid), pad_value=pl.PadValue.zero)
                        xr_full = pl.set_validshape(xr_clean, LINEAR_T_TILE // 2, RMS_CHUNK)
                        # row_sum 只归约本子块的列；随后跨 rk 累加，尚未除 HC_DIM 或开方。
                        rms_sums = pl.reshape(pl.row_sum(pl.mul(xr_full, xr_full)), [1, LINEAR_T_TILE // 2])
                        local_sum = pl.add(local_sum, rms_sums)
                    # 两个 AIV 各写自己的半块 token 行，因此写入地址不重叠。
                    rp = partial_row + lane * (LINEAR_T_TILE // 2)
                    rms_partials[rp:rp + LINEAR_T_TILE // 2, 0:1] = pl.reshape(local_sum, [LINEAR_T_TILE // 2, 1])
                # 【AIC 投影】区域外的 matmul 由 Cube 执行；b_trans=True 将权重从
                # [32,K段] 按转置方式参与运算，得到 [LINEAR_T_TILE,32] 部分投影。
                # 只有前24列有效，尾8列权重补零。此处与各 AIV 的 RMS 计算无结果依赖。
                weight = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_PER_SPLIT], [0, kbase], valid_shape=[MIX_HC, kcols], pad_value=pl.PadValue.zero)
                product = pl.matmul(gathered, weight, b_trans=True)
                mixes_partials[partial_row:partial_row + LINEAR_T_TILE, 0:MIX_PAD] = product

        # 【阶段交接：本核数据完成 + 全体会合 + 消费端缓存维护】
        # 后面的每个 token 块会读取所有 K 分段，生产者可能位于任意核组。
        # 因此仅有组内 aic_gather 的通知/等待不够，必须完成跨组的 GM 交接。
        # 这四步在 split_aiv 区域外，各组 AIC/AIV 都执行；即使本组 range 为空，
        # 也必须走到同一屏障，不能移进 task 循环或 tail_lane==0/1 的条件分支。

        # 生产者侧执行全 GM 范围的缓存失效操作，配合下方 fence 处理交接可见性。
        # 无参数形式覆盖整片 GM 缓存范围，不是清零 GM，更不是把数据复制回 Host。
        pl.system.cacheinvalid()
        # 当前核的 GM 数据屏障：建立先前数据操作与后续会合的顺序，等待相关操作
        # 完成。它不等待其他核组；当前 A2/A3 产物对应 pipe_barrier(ALL)+dsb(DDR)。
        pl.system.fence()
        # MIX 表示 AIC 与 AIV 都参加；默认是硬件屏障，只负责参与核的到达同步。
        # 它本身不发布/失效业务数据缓存，也不替代 fence。全量 clusters 和外层
        # sync_start=True 共同满足此硬屏障的启动要求，避免参与核分批启动后互等。
        pl.system.syncall(core_type=pl.KernelType.MIX)
        # 消费者侧再次失效缓存，后面的 GM load 才读取其他核组刚发布的部分和，
        # 避免继续使用旧缓存。前后的两次缓存维护分别处于生产/消费边界。
        pl.system.cacheinvalid()

        # 【第二阶段：把实际工作分给全体 AIV】仍在同一 SPMD 中，不另起 kernel。
        # 每组两个 AIV 在这里独立领取角色；前 sink_workers 个执行 Sinkhorn，
        # 其余执行门控/残差混合。角色不再固定为组内 AIV0/AIV1。
        # 中大 T：两种角色各24核，分别均分 token / token对；小 T：混合再切 D。
        for tail_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            vector_id = worker * 2 + tail_lane
            if vector_id < sink_workers:
                # comb 每个 token 输出64B，独占完整行即可，不需要两行配对。
                work_begin = vector_id * sink_rows_per_worker + pl.min(vector_id, sink_extra_rows)
                work_end = work_begin + sink_rows_per_worker + pl.min(1, pl.max(0, sink_extra_rows - vector_id))
                # 只处理本组独占的连续 token 区间；它与第一阶段 K 分段分工独立。
                # 小档/中档/大档每轮最多2/4/8行，不够整块时只读写 valid_rows 行。
                # T96 每个 Sinkhorn 工作核仅一轮4行；24个核都有实际计算。
                for ob in pl.range(work_begin, work_end, TAIL_MAX_ROWS):
                    # pl.read 在设备上取一个权重标量，不做 Tensor.item() 的 Host 回读。
                    scale2 = pl.read(hc_scale, [2])
                    t0 = ob
                    valid_rows = pl.min(TAIL_MAX_ROWS, work_end - t0)
                    # 归约全部 K 分段；无效行先补零，8行物理 tile 满足 Vector 对齐。
                    # 平方和也跨 K 分段归约：rstd = 1/sqrt(sum(x*x)/16384 + eps)。
                    # 投影先分段求和，再乘每个 token 的 rstd，完成 RMS 归一化的作用。
                    c_parts_load = pl.load(parts_2d, [0, t0 * MIX_PAD], [LINEAR_OK, T_TILE * MIX_PAD], valid_shape=[LINEAR_OK, valid_rows * MIX_PAD], target_memory=pl.MemorySpace.Vec)
                    c_parts = pl.set_validshape(pl.fillpad(c_parts_load, pad_value=pl.PadValue.zero), LINEAR_OK, T_TILE * MIX_PAD)
                    c_total = pl.reshape(pl.col_sum(c_parts), [T_TILE, MIX_PAD])
                    c_sq_load = pl.load(rms_2d, [0, t0], [RMS_OK, T_TILE], valid_shape=[RMS_OK, valid_rows], target_memory=pl.MemorySpace.Vec)
                    c_sq = pl.set_validshape(pl.fillpad(c_sq_load, pad_value=pl.PadValue.zero), RMS_OK, T_TILE)
                    c_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                    c_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(pl.col_sum(c_sq), HC_DIM_INV), NORM_EPS), c_rms_tmp), [T_TILE, 1])
                    comb_inv = c_inv
                    # 投影列8～23对应每个 token 的4×4 comb logit，使用第三个 scale。
                    raw = pl.add(pl.slice(c_total, [T_TILE, 16], [0, HC_MULT * 2]), 0.0)
                    base = pl.load(hc_base_2d, [0, HC_MULT * 2], [1, 16], target_memory=pl.MemorySpace.Vec)
                    logits = pl.add(pl.mul(pl.row_expand_mul(raw, comb_inv), scale2), pl.col_expand(raw, base))
                    # 只保留本轮最多 TAIL_MAX_ROWS 个矩阵，再展平做 Sinkhorn；
                    # 这里缩小实际计算 tile，避免把8行对齐用的补零全带进20轮迭代。
                    # ri1/ri2 分别翻转矩阵内索引的低第0/1位，取同行的伙伴元素；
                    # ci1/ci2 翻转第2/3位，取同列伙伴。索引始终留在同一token的16项内。
                    # 两次 pairwise gather 即可合成四项；展平减少逐行 gather 开销。
                    if STATIC_T == 96:
                        # T96每个Sinkhorn核处理4个token。先把[token,16个分量]置换成
                        # [16个分量,token]，同行/同列伙伴的间隔随之变为4/8/16/32项。
                        # 其中后三种间隔均对齐32B，使用gatherb；最小间隔仍用gather。
                        # 输入/输出各置换一次，20轮归一化及pairwise加法树保持不变。
                        # 索引始终指向本核的4个矩阵，不发生跨核数据交换。
                        logits_flat = pl.reshape(pl.slice(logits, [TAIL_MAX_ROWS, 16], [0, 0]), [1, TAIL_MAX_ROWS * 16])
                        ids = pl.tile.arange(0, [1, TAIL_MAX_ROWS * 16], dtype=pl.INT32)
                        tmp = pl.create_tile([1, TAIL_MAX_ROWS * 16], dtype=pl.INT32)
                        component_ids = pl.tile.shrs(ids, TAIL_MAX_ROWS // 2 - TAIL_MAX_ROWS // 8)
                        token_ids = pl.sub(ids, pl.tile.shls(component_ids, TAIL_MAX_ROWS // 2 - TAIL_MAX_ROWS // 8))
                        perm_in = pl.add(pl.tile.shls(token_ids, 4), component_ids)
                        out_token = pl.tile.shrs(ids, 4)
                        out_component = pl.sub(ids, pl.tile.shls(out_token, 4))
                        perm_out = pl.add(pl.tile.shls(out_component, TAIL_MAX_ROWS // 2 - TAIL_MAX_ROWS // 8), out_token)
                        compact_logits = pl.tile.gather(logits_flat, perm_in, tmp)
                        ids16 = pl.reinterpret_view(ids, pl.INT16)
                        xor_tmp16 = pl.create_tile([1, TAIL_MAX_ROWS * 32], dtype=pl.INT16)
                        block_numbers = pl.tile.arange(0, [1, (TAIL_MAX_ROWS * 2 + 7) // 8 * 8], dtype=pl.INT32)
                        # 此分支固定4行：block_numbers为0..7，无需对8取模。
                        block_ids = pl.tile.shls(block_numbers, 5)
                        block_ids16 = pl.reinterpret_view(block_ids, pl.INT16)
                        block_tmp16 = pl.create_tile([1, (TAIL_MAX_ROWS * 2 + 7) // 8 * 16], dtype=pl.INT16)
                        # 4行紧凑布局的伙伴偏移是编译期常量，直接填充；
                        # 不再先生成1/8/16/32的tile、再在设备上逐项左移。
                        ri1_mask = pl.reinterpret_view(pl.tile.full([1, TAIL_MAX_ROWS * 16], dtype=pl.INT32, value=4), pl.INT16)
                        ri1 = pl.reinterpret_view(pl.tile.xor(ids16, ri1_mask, xor_tmp16), pl.INT32)
                        ri2_block_mask = pl.reinterpret_view(pl.tile.full([1, (TAIL_MAX_ROWS * 2 + 7) // 8 * 8], dtype=pl.INT32, value=32), pl.INT16)
                        ri2_offset = pl.reinterpret_view(pl.tile.xor(block_ids16, ri2_block_mask, block_tmp16), pl.UINT32)
                        ci1_block_mask = pl.reinterpret_view(pl.tile.full([1, (TAIL_MAX_ROWS * 2 + 7) // 8 * 8], dtype=pl.INT32, value=64), pl.INT16)
                        ci1_offset = pl.reinterpret_view(pl.tile.xor(block_ids16, ci1_block_mask, block_tmp16), pl.UINT32)
                        ci2_block_mask = pl.reinterpret_view(pl.tile.full([1, (TAIL_MAX_ROWS * 2 + 7) // 8 * 8], dtype=pl.INT32, value=128), pl.INT16)
                        ci2_offset = pl.reinterpret_view(pl.tile.xor(block_ids16, ci2_block_mask, block_tmp16), pl.UINT32)
                        sink_swap_0 = pl.tile.gather(compact_logits, ri1, tmp)
                        max2 = pl.maximum(compact_logits, sink_swap_0)
                        sink_swap_1 = pl.slice(pl.tile.gatherb(max2, ri2_offset), [1, TAIL_MAX_ROWS * 16], [0, 0])
                        max4 = pl.maximum(max2, sink_swap_1)
                        exps = pl.exp(pl.sub(compact_logits, max4))
                        sink_swap_2 = pl.tile.gather(exps, ri1, tmp)
                        sum2 = pl.add(exps, sink_swap_2)
                        sink_swap_3 = pl.slice(pl.tile.gatherb(sum2, ri2_offset), [1, TAIL_MAX_ROWS * 16], [0, 0])
                        sum4 = pl.add(sum2, sink_swap_3)
                        probs = pl.add(pl.div(exps, sum4), HC_EPS)
                        sink_swap_4 = pl.slice(pl.tile.gatherb(probs, ci1_offset), [1, TAIL_MAX_ROWS * 16], [0, 0])
                        col2 = pl.add(probs, sink_swap_4)
                        sink_swap_5 = pl.slice(pl.tile.gatherb(col2, ci2_offset), [1, TAIL_MAX_ROWS * 16], [0, 0])
                        col4 = pl.add(pl.add(col2, sink_swap_5), HC_EPS)
                        cur = pl.div(probs, col4)
                        for iteration in pl.range(HC_SINKHORN_ITER - 1):
                            sink_swap_6 = pl.tile.gather(cur, ri1, tmp)
                            row2 = pl.add(cur, sink_swap_6)
                            sink_swap_7 = pl.slice(pl.tile.gatherb(row2, ri2_offset), [1, TAIL_MAX_ROWS * 16], [0, 0])
                            row4 = pl.add(pl.add(row2, sink_swap_7), HC_EPS)
                            norm = pl.div(cur, row4)
                            sink_swap_8 = pl.slice(pl.tile.gatherb(norm, ci1_offset), [1, TAIL_MAX_ROWS * 16], [0, 0])
                            col2 = pl.add(norm, sink_swap_8)
                            sink_swap_9 = pl.slice(pl.tile.gatherb(col2, ci2_offset), [1, TAIL_MAX_ROWS * 16], [0, 0])
                            col4 = pl.add(pl.add(col2, sink_swap_9), HC_EPS)
                            cur = pl.div(norm, col4)
                        row_major = pl.tile.gather(cur, perm_out, tmp)
                        valid = pl.set_validshape(pl.reshape(row_major, [TAIL_MAX_ROWS, 16]), valid_rows, 16)
                        pl.store(valid, [t0, 0], comb)
                    else:
                        regular_logits = pl.reshape(pl.slice(logits, [TAIL_MAX_ROWS, 16], [0, 0]), [1, TAIL_MAX_ROWS * 16])
                        ids = pl.tile.arange(0, [1, TAIL_MAX_ROWS * 16], dtype=pl.INT32)
                        tmp = pl.create_tile([1, TAIL_MAX_ROWS * 16], dtype=pl.INT32)
                        idx_float = pl.cast(ids, pl.FP32)
                        ri1_low = pl.cast(pl.cast(pl.mul(idx_float, 1.0), pl.INT32, mode="floor"), pl.FP32)
                        ri1_high = pl.cast(pl.cast(pl.mul(idx_float, 0.5), pl.INT32, mode="floor"), pl.FP32)
                        ri1_bit = pl.sub(ri1_low, pl.mul(ri1_high, 2.0))
                        ri1 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ri1_bit, 2.0)), 1.0), pl.INT32)
                        ri2_low = pl.cast(pl.cast(pl.mul(idx_float, 0.5), pl.INT32, mode="floor"), pl.FP32)
                        ri2_high = pl.cast(pl.cast(pl.mul(idx_float, 0.25), pl.INT32, mode="floor"), pl.FP32)
                        ri2_bit = pl.sub(ri2_low, pl.mul(ri2_high, 2.0))
                        ri2 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ri2_bit, 4.0)), 2.0), pl.INT32)
                        ci1_low = pl.cast(pl.cast(pl.mul(idx_float, 0.25), pl.INT32, mode="floor"), pl.FP32)
                        ci1_high = pl.cast(pl.cast(pl.mul(idx_float, 0.125), pl.INT32, mode="floor"), pl.FP32)
                        ci1_bit = pl.sub(ci1_low, pl.mul(ci1_high, 2.0))
                        ci1 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ci1_bit, 8.0)), 4.0), pl.INT32)
                        ci2_low = pl.cast(pl.cast(pl.mul(idx_float, 0.125), pl.INT32, mode="floor"), pl.FP32)
                        ci2_high = pl.cast(pl.cast(pl.mul(idx_float, 0.0625), pl.INT32, mode="floor"), pl.FP32)
                        ci2_bit = pl.sub(ci2_low, pl.mul(ci2_high, 2.0))
                        ci2 = pl.cast(pl.add(pl.sub(idx_float, pl.mul(ci2_bit, 16.0)), 8.0), pl.INT32)
                        # 首轮先减行最大值再 exp，降低指数溢出风险；做行归一化并加 eps，
                        # 随后做列归一化。下面再执行19轮行/列归一化，共20轮 Sinkhorn。
                        max2 = pl.maximum(regular_logits, pl.tile.gather(regular_logits, ri1, tmp))
                        max4 = pl.maximum(max2, pl.tile.gather(max2, ri2, tmp))
                        exps = pl.exp(pl.sub(regular_logits, max4))
                        sum2 = pl.add(exps, pl.tile.gather(exps, ri1, tmp))
                        sum4 = pl.add(sum2, pl.tile.gather(sum2, ri2, tmp))
                        probs = pl.add(pl.div(exps, sum4), HC_EPS)
                        col2 = pl.add(probs, pl.tile.gather(probs, ci1, tmp))
                        col4 = pl.add(pl.add(col2, pl.tile.gather(col2, ci2, tmp)), HC_EPS)
                        cur = pl.div(probs, col4)
                        # cur 是迭代状态，当前轮依赖上一轮；每个 Sinkhorn 核独立完成自己
                        # token 块的全部迭代，这里无需每轮跨核组同步。
                        if TAIL_MAX_ROWS <= 2 and LINEAR_T_TILE == 32:
                            # 仅小档采用：每核通常只有一个4×4矩阵，标量迭代可减少短向量开销。
                            # 中档每核多行时仍用下方向量路径并行推进，避免逐矩阵串行化。
                            # 首轮仍使用原来的稳定softmax/列归一化。后19轮的4x4矩阵
                            # 保留在设备FP32标量中，避免每轮四次tile gather；只处理真实行。
                            # 每个分母仍按两对相加，eps位置不变；倒数乘法须过原精度门禁。
                            for sink_row in pl.range(valid_rows):
                                sv00 = pl.tile.read(cur, [0, sink_row * 16 + 0])
                                sv01 = pl.tile.read(cur, [0, sink_row * 16 + 1])
                                sv02 = pl.tile.read(cur, [0, sink_row * 16 + 2])
                                sv03 = pl.tile.read(cur, [0, sink_row * 16 + 3])
                                sv04 = pl.tile.read(cur, [0, sink_row * 16 + 4])
                                sv05 = pl.tile.read(cur, [0, sink_row * 16 + 5])
                                sv06 = pl.tile.read(cur, [0, sink_row * 16 + 6])
                                sv07 = pl.tile.read(cur, [0, sink_row * 16 + 7])
                                sv08 = pl.tile.read(cur, [0, sink_row * 16 + 8])
                                sv09 = pl.tile.read(cur, [0, sink_row * 16 + 9])
                                sv10 = pl.tile.read(cur, [0, sink_row * 16 + 10])
                                sv11 = pl.tile.read(cur, [0, sink_row * 16 + 11])
                                sv12 = pl.tile.read(cur, [0, sink_row * 16 + 12])
                                sv13 = pl.tile.read(cur, [0, sink_row * 16 + 13])
                                sv14 = pl.tile.read(cur, [0, sink_row * 16 + 14])
                                sv15 = pl.tile.read(cur, [0, sink_row * 16 + 15])
                                for scalar_iteration in pl.range(HC_SINKHORN_ITER - 1):
                                    sr0 = 1.0 / ((sv00 + sv01) + (sv02 + sv03) + HC_EPS)
                                    sv00 = sv00 * sr0
                                    sv01 = sv01 * sr0
                                    sv02 = sv02 * sr0
                                    sv03 = sv03 * sr0
                                    sr1 = 1.0 / ((sv04 + sv05) + (sv06 + sv07) + HC_EPS)
                                    sv04 = sv04 * sr1
                                    sv05 = sv05 * sr1
                                    sv06 = sv06 * sr1
                                    sv07 = sv07 * sr1
                                    sr2 = 1.0 / ((sv08 + sv09) + (sv10 + sv11) + HC_EPS)
                                    sv08 = sv08 * sr2
                                    sv09 = sv09 * sr2
                                    sv10 = sv10 * sr2
                                    sv11 = sv11 * sr2
                                    sr3 = 1.0 / ((sv12 + sv13) + (sv14 + sv15) + HC_EPS)
                                    sv12 = sv12 * sr3
                                    sv13 = sv13 * sr3
                                    sv14 = sv14 * sr3
                                    sv15 = sv15 * sr3
                                    sc0 = 1.0 / ((sv00 + sv04) + (sv08 + sv12) + HC_EPS)
                                    sv00 = sv00 * sc0
                                    sv04 = sv04 * sc0
                                    sv08 = sv08 * sc0
                                    sv12 = sv12 * sc0
                                    sc1 = 1.0 / ((sv01 + sv05) + (sv09 + sv13) + HC_EPS)
                                    sv01 = sv01 * sc1
                                    sv05 = sv05 * sc1
                                    sv09 = sv09 * sc1
                                    sv13 = sv13 * sc1
                                    sc2 = 1.0 / ((sv02 + sv06) + (sv10 + sv14) + HC_EPS)
                                    sv02 = sv02 * sc2
                                    sv06 = sv06 * sc2
                                    sv10 = sv10 * sc2
                                    sv14 = sv14 * sc2
                                    sc3 = 1.0 / ((sv03 + sv07) + (sv11 + sv15) + HC_EPS)
                                    sv03 = sv03 * sc3
                                    sv07 = sv07 * sc3
                                    sv11 = sv11 * sc3
                                    sv15 = sv15 * sc3
                                pl.tile.write(cur, [0, sink_row * 16 + 0], sv00)
                                pl.tile.write(cur, [0, sink_row * 16 + 1], sv01)
                                pl.tile.write(cur, [0, sink_row * 16 + 2], sv02)
                                pl.tile.write(cur, [0, sink_row * 16 + 3], sv03)
                                pl.tile.write(cur, [0, sink_row * 16 + 4], sv04)
                                pl.tile.write(cur, [0, sink_row * 16 + 5], sv05)
                                pl.tile.write(cur, [0, sink_row * 16 + 6], sv06)
                                pl.tile.write(cur, [0, sink_row * 16 + 7], sv07)
                                pl.tile.write(cur, [0, sink_row * 16 + 8], sv08)
                                pl.tile.write(cur, [0, sink_row * 16 + 9], sv09)
                                pl.tile.write(cur, [0, sink_row * 16 + 10], sv10)
                                pl.tile.write(cur, [0, sink_row * 16 + 11], sv11)
                                pl.tile.write(cur, [0, sink_row * 16 + 12], sv12)
                                pl.tile.write(cur, [0, sink_row * 16 + 13], sv13)
                                pl.tile.write(cur, [0, sink_row * 16 + 14], sv14)
                                pl.tile.write(cur, [0, sink_row * 16 + 15], sv15)
                        else:
                            for iteration in pl.range(HC_SINKHORN_ITER - 1):
                                row2 = pl.add(cur, pl.tile.gather(cur, ri1, tmp))
                                row4 = pl.add(pl.add(row2, pl.tile.gather(row2, ri2, tmp)), HC_EPS)
                                norm = pl.div(cur, row4)
                                col2 = pl.add(norm, pl.tile.gather(norm, ci1, tmp))
                                col4 = pl.add(pl.add(col2, pl.tile.gather(col2, ci2, tmp)), HC_EPS)
                                cur = pl.div(norm, col4)
                        # 只回写真实 token 行，最后不足8行的补齐部分不会写到输出外部。
                        valid = pl.set_validshape(pl.reshape(cur, [TAIL_MAX_ROWS, 16]), valid_rows, 16)
                        pl.store(valid, [t0, 0], comb)

            else:
                mix_id = vector_id - sink_workers
                # 先按 token 对均分。T96时24个混合核各4行，T168时每核6或8行。
                work_begin = (mix_id * mix_pairs_per_worker + pl.min(mix_id, mix_extra_workers)) * 2
                work_end = pl.min(tokens, work_begin + (mix_pairs_per_worker + pl.min(1, pl.max(0, mix_extra_workers - mix_id))) * 2)
                d_begin = 0
                d_end = D // D_TILE
                if mix_workers > tail_pairs:
                    # 小 T：多个工作核共同处理一个 token 对，但各写独占的 D 区间。
                    if mix_id < mix_extra_prefix:
                        pair_id = mix_id // (mix_workers_per_pair + 1)
                        d_worker = mix_id % (mix_workers_per_pair + 1)
                        d_workers = mix_workers_per_pair + 1
                    else:
                        rest_id = mix_id - mix_extra_prefix
                        pair_id = mix_extra_pairs + rest_id // mix_workers_per_pair
                        d_worker = rest_id % mix_workers_per_pair
                        d_workers = mix_workers_per_pair
                    work_begin = pl.min(tokens, pair_id * 2)
                    work_end = pl.min(tokens, work_begin + 2)
                    d_per_worker = (D // D_TILE) // d_workers
                    d_extra = (D // D_TILE) % d_workers
                    d_begin = d_worker * d_per_worker + pl.min(d_worker, d_extra)
                    d_end = d_begin + d_per_worker + pl.min(1, pl.max(0, d_extra - d_worker))
                if mix_id >= mix_workers:
                    # 仅超小 T 可能工作不足48份；不造重复计算或重复写入。
                    work_begin = 0
                    work_end = 0
                for blk in pl.range(work_begin, work_end, TAIL_MAX_ROWS):
                    t0 = blk
                    d_base = 0
                    valid_rows = pl.min(TAIL_MAX_ROWS, work_end - t0)
                    # 混合核自行归约投影和平方和，独立得到 rstd；与 Sinkhorn 核
                    # 有少量重复归约，但不等待另一角色传中间结果，可并行推进。
                    m_parts_load = pl.load(parts_2d, [0, t0 * MIX_PAD], [LINEAR_OK, T_TILE * MIX_PAD], valid_shape=[LINEAR_OK, valid_rows * MIX_PAD], target_memory=pl.MemorySpace.Vec)
                    m_parts = pl.set_validshape(pl.fillpad(m_parts_load, pad_value=pl.PadValue.zero), LINEAR_OK, T_TILE * MIX_PAD)
                    m_total = pl.reshape(pl.col_sum(m_parts), [T_TILE, MIX_PAD])
                    m_sq_load = pl.load(rms_2d, [0, t0], [RMS_OK, T_TILE], valid_shape=[RMS_OK, valid_rows], target_memory=pl.MemorySpace.Vec)
                    m_sq = pl.set_validshape(pl.fillpad(m_sq_load, pad_value=pl.PadValue.zero), RMS_OK, T_TILE)
                    m_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                    m_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(pl.col_sum(m_sq), HC_DIM_INV), NORM_EPS), m_rms_tmp), [T_TILE, 1])
                    # pre/post 共用八列归一化、偏置和 sigmoid，保留各自 scale。
                    # gate_scale_ids 为 [0,0,0,0,1,1,1,1]：前4列用 hc_scale[0]，
                    # 后4列用 hc_scale[1]；hc_scale[2] 留给 Sinkhorn 的 comb。
                    gate_base = pl.load(hc_base_2d, [0, 0], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
                    gate_scales = pl.load(hc_scale_2d, [0, 0], [1, HC_PAD], valid_shape=[1, 3], target_memory=pl.MemorySpace.Vec)
                    gate_ids = pl.cast(pl.tile.arange(0, [1, HC_PAD], dtype=pl.INT32), pl.FP32)
                    gate_scale_ids = pl.cast(pl.mul(gate_ids, 0.25), pl.INT32, mode="floor")
                    gate_tmp = pl.create_tile([1, HC_PAD], dtype=pl.INT32)
                    gate_scale_vec = pl.tile.gather(gate_scales, gate_scale_ids, gate_tmp)
                    gate_raw = pl.add(pl.slice(m_total, [T_TILE, HC_PAD], [0, 0]), 0.0)
                    gate_norm = pl.row_expand_mul(gate_raw, m_inv)
                    gate_scaled = pl.col_expand_mul(gate_norm, gate_scale_vec)
                    gate_logits = pl.add(gate_scaled, pl.col_expand(gate_scaled, gate_base))
                    gate_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_logits)), 1.0))
                    # 列广播路径使用转置后的门值；标量路径直接读 gate_sigmoid，
                    # 专门生成中档代码时，未使用的转置由编译器消去。
                    gate_transposed = pl.transpose(gate_sigmoid, axis1=0, axis2=1)
                    # 展平后一次提取每行后四列，连续写回 post，避免 16B 行转置。
                    post_ids_float = pl.cast(pl.tile.arange(0, [1, T_TILE * HC_MULT], dtype=pl.INT32), pl.FP32)
                    post_rows = pl.cast(pl.cast(pl.mul(post_ids_float, 1.0 / HC_MULT), pl.INT32, mode="floor"), pl.FP32)
                    post_ids = pl.cast(pl.add(pl.add(post_ids_float, pl.mul(post_rows, HC_MULT)), HC_MULT), pl.INT32)
                    post_tmp = pl.create_tile([1, T_TILE * HC_MULT], dtype=pl.INT32)
                    post_values = pl.tile.gather(pl.reshape(gate_sigmoid, [1, T_TILE * HC_PAD]), post_ids, post_tmp)
                    # post = 2*sigmoid(post_logits)，对应每个 token 的后4列。
                    post_result = pl.set_validshape(pl.mul(post_values, 2.0), 1, valid_rows * HC_MULT)
                    # 同一个 token 对可能由多个 AIV 分担 D；post 仅由第一个分片写。
                    if d_begin == 0:
                        pl.store(post_result, [0, t0 * HC_MULT], post_flat)
                    # 每核处理较少 token 时，沿整行D计算，四个pre门值读为设备标量。
                    # 直接从行主序 sigmoid 取门值，无需为标量路径转置。
                    # 所有 GM 读取、标量取值与向量计算仍计入同一个 incore。
                    if SCALAR_MIX and TAIL_MAX_ROWS <= 2:
                        # 【较小分档的连续D分片】保持原来按token对/D轴的核分工。
                        # d_begin/d_end以512列为单位，转成该核独占的半开区间。
                        # 物理tile预留[1,4096]，valid_shape只允许读取本核的列数，
                        # 并非每个分片核都读取整行；两级流水沿真实token行推进。
                        part_scalar_d0 = d_begin * D_TILE
                        part_scalar_cols = (d_end - d_begin) * D_TILE
                        for part_scalar_tr in pl.pipeline(valid_rows, stage=2):
                            part_scalar_row = t0 + part_scalar_tr
                            part_scalar_pre0 = pl.tile.read(gate_sigmoid, [part_scalar_tr, 0]) + HC_EPS
                            part_scalar_pre1 = pl.tile.read(gate_sigmoid, [part_scalar_tr, 1]) + HC_EPS
                            part_scalar_pre2 = pl.tile.read(gate_sigmoid, [part_scalar_tr, 2]) + HC_EPS
                            part_scalar_pre3 = pl.tile.read(gate_sigmoid, [part_scalar_tr, 3]) + HC_EPS
                            part_scalar_x0 = pl.cast(pl.load(x_flat, [part_scalar_row, part_scalar_d0 + 0], [1, D], valid_shape=[1, part_scalar_cols], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            part_scalar_y0 = pl.mul(part_scalar_x0, part_scalar_pre0)
                            part_scalar_x1 = pl.cast(pl.load(x_flat, [part_scalar_row, part_scalar_d0 + D], [1, D], valid_shape=[1, part_scalar_cols], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            part_scalar_y1 = pl.mul(part_scalar_x1, part_scalar_pre1)
                            part_scalar_sum01 = pl.add(part_scalar_y0, part_scalar_y1)
                            part_scalar_x2 = pl.cast(pl.load(x_flat, [part_scalar_row, part_scalar_d0 + 2 * D], [1, D], valid_shape=[1, part_scalar_cols], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            part_scalar_y2 = pl.mul(part_scalar_x2, part_scalar_pre2)
                            part_scalar_x3 = pl.cast(pl.load(x_flat, [part_scalar_row, part_scalar_d0 + 3 * D], [1, D], valid_shape=[1, part_scalar_cols], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            part_scalar_y3 = pl.mul(part_scalar_x3, part_scalar_pre3)
                            part_scalar_result = pl.add(part_scalar_sum01, pl.add(part_scalar_y2, part_scalar_y3))
                            part_scalar_bf16 = pl.cast(part_scalar_result, target_type=pl.BF16, mode="rint")
                            # 写回也保留相同列边界，不能覆盖同一token上其它核的分片。
                            pl.store(pl.set_validshape(part_scalar_bf16, 1, part_scalar_cols), [part_scalar_row, part_scalar_d0], mixed_view)
                    elif SCALAR_MIX:
                        for scalar_tr in pl.pipeline(valid_rows, stage=2):
                            scalar_row = t0 + scalar_tr
                            scalar_pre0 = pl.tile.read(gate_sigmoid, [scalar_tr, 0]) + HC_EPS
                            scalar_pre1 = pl.tile.read(gate_sigmoid, [scalar_tr, 1]) + HC_EPS
                            scalar_pre2 = pl.tile.read(gate_sigmoid, [scalar_tr, 2]) + HC_EPS
                            scalar_pre3 = pl.tile.read(gate_sigmoid, [scalar_tr, 3]) + HC_EPS
                            scalar_x0 = pl.cast(pl.load(x_flat, [scalar_row, 0], [1, D], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            scalar_y0 = pl.mul(scalar_x0, scalar_pre0)
                            scalar_x1 = pl.cast(pl.load(x_flat, [scalar_row, D], [1, D], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            scalar_y1 = pl.mul(scalar_x1, scalar_pre1)
                            scalar_sum01 = pl.add(scalar_y0, scalar_y1)
                            scalar_x2 = pl.cast(pl.load(x_flat, [scalar_row, 2 * D], [1, D], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            scalar_y2 = pl.mul(scalar_x2, scalar_pre2)
                            scalar_x3 = pl.cast(pl.load(x_flat, [scalar_row, 3 * D], [1, D], target_memory=pl.MemorySpace.Vec), pl.FP32)
                            scalar_y3 = pl.mul(scalar_x3, scalar_pre3)
                            scalar_result = pl.add(scalar_sum01, pl.add(scalar_y2, scalar_y3))
                            scalar_bf16 = pl.cast(scalar_result, target_type=pl.BF16, mode="rint")
                            pl.store(scalar_bf16, [scalar_row, 0], mixed_view)
                    else:
                        # pre = sigmoid(pre_logits)+eps，四个门值在该 token 的整个 D 轴复用；
                        # [T_TILE,1] 的形状用于对每行特征广播，因此放在 db 循环之外计算。
                        pre0 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [0, 0]), HC_EPS), [T_TILE, 1])
                        pre1 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [1, 0]), HC_EPS), [T_TILE, 1])
                        pre2 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [2, 0]), HC_EPS), [T_TILE, 1])
                        pre3 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [3, 0]), HC_EPS), [T_TILE, 1])

                        # 【混合工作核的软件流水】D 轴共8块，本核只执行[d_begin,d_end)。
                        # 每块读取四路 [8,512] BF16，转 FP32，加权求和后转回 BF16 写出。
                        # 各 db 写入不同特征区间，没有跨 db 的数值累加依赖。
                        # stage=2 请求两级流水缓冲，使相邻块的 GM→UB 搬入、Vector
                        # 计算和 UB→GM 写出有机会重叠；编译器负责缓冲轮转与依赖同步。
                        # 它不表示用两个 AIV，也不会再派发两份任务；该循环仅由当前
                        # 混合工作核执行；相较普通 rk 归约循环，这里显式申请软件流水。
                        # 两级缓冲会增加 UB 占用，最终重叠程度仍取决于生成代码和数据依赖。
                        for db in pl.pipeline(d_begin, d_end, stage=2):
                            d0 = d_base + db * D_TILE
                            x0_bf16 = pl.load(x_flat, [t0, 0 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                            x0 = pl.cast(x0_bf16, pl.FP32)
                            x1_bf16 = pl.load(x_flat, [t0, 1 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                            x1 = pl.cast(x1_bf16, pl.FP32)
                            x2_bf16 = pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                            x2 = pl.cast(x2_bf16, pl.FP32)
                            x3_bf16 = pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_TILE], valid_shape=[valid_rows, D_TILE], target_memory=pl.MemorySpace.Vec)
                            x3 = pl.cast(x3_bf16, pl.FP32)
                            # 每行使用自己的四个 pre 门值，对应四路输入逐元素加权。
                            y0 = pl.row_expand_mul(x0, pre0)
                            y1 = pl.row_expand_mul(x1, pre1)
                            y2 = pl.row_expand_mul(x2, pre2)
                            y3 = pl.row_expand_mul(x3, pre3)
                            # 固定为 (y0+y1)+(y2+y3) 的 FP32 加法顺序；重排可能改变舍入。
                            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
                            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                            y_valid = pl.set_validshape(y_bf16, valid_rows, D_TILE)
                            pl.store(y_valid, [t0, d0], mixed_view)
    # 三个输出均在上面直接写入传入的 GM Tensor；返回的是输出引用，不触发 Host 拷贝。
    return mixed, post, comb_out


@pl.jit(auto_scope=False)
def hc_pre(
    x: pl.Tensor[[TOKENS, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    mixed: pl.Out[pl.Tensor[[TOKENS, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[TOKENS, HC_MULT], pl.FP32]],
    comb_out: pl.Out[pl.Tensor[[TOKENS, HC_MULT, HC_MULT], pl.FP32]],
):
    """单个动态 JIT 入口；按 T=B×S 分派，每次只执行一个完整分支。

    输入 x 为 BF16，hc_fn/hc_scale/hc_base 为 FP32；mixed 为 BF16，
    post/comb_out 为 FP32。调用方负责提供满足签名的输入和输出空间。
    _hc_pre_variant 在编译时内联，各分档生成自己的分块方案；运行时根据
    实际 T 选择其一。BS/S 不单独参与计算，相同 T 使用相同分档。
    """
    # 只读取已有 shape 元数据，不回读任何 device tensor 内容。
    # 参数顺序：投影M块、K宽度、K段数、RMS子块、槽数、尾段行上限、任务名。
    # 小档保留M32/K1536和11份部分和，段内2/3次K512供数与RMS复用，
    # 用同一显式Acc与K256 L0双缓冲；尾段后19轮使用设备标量。中档每段两次K1024，
    # 大档每段四次K512并使用两级流水。
    # T49～95及T>96局部累加RMS后写8份；精确T96仍写16份RMS、8份投影部分和。
    tokens = pl.tensor.dim(x, 0)
    if tokens <= SMALL_T_MAX:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               32, 1536, 11, 512, 2, 2, "hc_small_spmd", 0, 11, False, 512)
    elif tokens <= MEDIUM48_T_MAX:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               48, 1024, 16, 512, 2, 2, "hc_medium48_spmd", 0, 16, True, 1024)
    elif tokens < MEDIUM96_T_MAX:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               32, 2048, 8, 256, 2, 4, "hc_medium96_spmd", 0, 8, True, 1024)
    elif tokens == MEDIUM96_T_MAX:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               32, 2048, 8, 256, 2, 4, "hc_exact96_spmd", 96, 16, True, 2048)
    elif tokens <= 144:
        # 大档先使投影任务落在一轮核组内：T100～144 采用3个M48块×8个K段。
        # 每个K2048逻辑段用4次K512两级流水，输入FIFO占192KiB、双权重占128KiB L1。
        # RMS复用已加宽的半块，在本段内累加成8份；不额外重读BF16输入。
        # 尾段仍由24个Sinkhorn核和24个混合核独立分工，保持20轮归一化。
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               48, 2048, 8, 512, 2, 8, "hc_large144_spmd", 0, 8, False, 512)
    else:
        # T168采用3个M64块×8个K段，避免旧M32/K1536的66份任务跨三轮。
        # 两槽交接占256KiB L1，双权重占128KiB；两级流水还生成AIV输入双缓冲。
        # 该路径也支持其他动态T，但T>192将重新出现多轮，不宣称恒定一轮。
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               64, 2048, 8, 512, 2, 8, "hc_large_spmd", 0, 8, True, 512)
    return mixed, post, comb_out
