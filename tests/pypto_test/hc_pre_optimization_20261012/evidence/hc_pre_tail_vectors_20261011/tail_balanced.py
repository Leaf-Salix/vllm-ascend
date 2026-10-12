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
    3. 按 token 块重新分工：AIV0 做 Sinkhorn 并写 comb；AIV1 做 pre/post
       门控及残差混合，写 post/mixed。此阶段没有新的矩阵乘任务。

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

# 编排读取 tensor 描述符中的 T 选择分块，完整计算仍在一个选中分支内。
SMALL_T_MAX = 32
MEDIUM48_T_MAX = 48
MEDIUM96_T_MAX = 96
# 测试根据此表核对实际任务名，同时要求每次仅执行选中分支的 AIC/AIV 对。
HC_PRE_VARIANTS = (
    (SMALL_T_MAX, "hc_small", 32, 1536),
    (MEDIUM48_T_MAX, "hc_medium48", 48, 1024),
    (MEDIUM96_T_MAX, "hc_medium96", 96, 512),
    (None, "hc_large", 32, 1536),
)

# 投影按 LINEAR_T_TILE 切 token；两个 AIV 各转换半块行，再合并交给 Cube。
# 尾段独立采用 8-token 块，避免把投影的大 M 块沿用到 Sinkhorn/门控。
MIX_PAD = 32  # 24 列投影补齐至 Cube N 轴和 Vector 行的 32 列。
HC_PAD = 8  # pre/post 合并为 8 个 FP32 值，恰好占一行 32B。
TAIL_ROWS = 4  # 候选：每个尾段任务只负责4个有效token，物理tile保留8行。
T_TILE = 8  # 尾段门控和残差混合每块的 token 数。
COMB_T_TILE = 4  # Sinkhorn 每块的 token 数；当前须与 T_TILE 一致。
D_TILE = 512  # AIV1 核内残差混合流水每次处理的特征数。
D_SPMD = 4096  # 一个尾段任务覆盖的特征宽度；当前覆盖完整 D。


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
    COMB_T_TILE: pl.constexpr,
    TASK_NAME: pl.constexpr,
):
    """按编译期参数生成一个完整的混合核分支。

    LINEAR_T_TILE：每个投影任务处理的 token 行数，两个 AIV 各处理一半。
    LINEAR_K_PER_SPLIT：投影的 K 分段宽度，最后一段可含补零列。
    LINEAR_OK：K 分段数量；各分段结果保留到 GM，屏障后再统一归约。
    RMS_CHUNK：每次平方和归约的 K 子块宽度，控制 AIV 的 UB 临时占用；
        当前各分档的 LINEAR_K_PER_SPLIT 都能被 RMS_CHUNK 整除。
    SLOTS：组内 AIV→AIC 数据交接通道的缓冲槽数，与下面 pipeline 的
        stage 各自作用于不同位置，均不改变核组数量。
    TASK_NAME：生成任务的名称提示，供编译产物及 profiling 识别分档。
    """
    # shape 读取与 reshape 描述逻辑视图，不需要把 x 的数值读回 Host。
    # 后续用显式偏移访问四路输入；这些 reshape 不做整张 Tensor 的搬运。
    tokens = pl.tensor.dim(x, 0)
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
    # GM 中按 [K分段, 补齐后的token, 投影列/平方和] 存放部分结果。
    # 每个 (split, token块) 都有独立写入位置；这里不通过 atomic 累加。
    # create_tensor 只分配临时 Tensor，不表示已将内容清零；生产循环负责写入。
    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    rms_partials = pl.create_tensor([LINEAR_OK * t_linear, 1], dtype=pl.FP32)
    linear_units = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    # 从运行时查询本次设备的可用 MIX 核组数。在当前 A3 上，每组是 1 AIC+2 AIV。
    # 这不是 batch size，也不是写死的 24；下面的硬件 syncall 要求全量参与，
    # 因此启动宽度随设备查询结果确定，不能仅按 linear_units 随意缩小。
    clusters = pl.system.available_cluster_count()
    linear_workers = clusters
    tail_round_rows = clusters * COMB_T_TILE
    token_tiles = ((tokens + tail_round_rows - 1) // tail_round_rows) * clusters
    # 同一 GM 缓冲的归约视图：屏障之后，尾段按列取出一个 token 块的所有 K 分段。
    parts_2d = pl.reshape(mixes_partials, [LINEAR_OK, t_linear * MIX_PAD])
    rms_2d = pl.reshape(rms_partials, [LINEAR_OK, t_linear])

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
        # 例如 B16/S6：T=96，LINEAR_T_TILE=96，LINEAR_OK=32，linear_units=32。
        # 若 clusters=24，组0～7各做两段 K，组8～23各做一段 K。
        for task in pl.range(worker, linear_units, linear_workers):
            # task 将 token 块与 K 分段展平；除法恢复 token 块，取模恢复 K 分段。
            t0 = (task // LINEAR_OK) * LINEAR_T_TILE
            split = task % LINEAR_OK
            kbase = split * LINEAR_K_PER_SPLIT
            kcols = pl.min(LINEAR_K_PER_SPLIT, HC_DIM - kbase)
            partial_row = split * t_linear + t0

            # 【组内双 AIV 数据分工】lane 是 AIV 的本地编号 0/1；两路并行执行，
            # 并非同一个 AIV 先跑 lane0 再跑 lane1。UP_DOWN 表示按行交接给 AIC。
            # 本实现显式按半块设置 shape 和 lane_row：T96 分支中 AIV0 负责
            # token行0～47，AIV1负责48～95；其他 token 块则加各自的 t0。
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
                # T96 分支为 512/256=2 次；local_sum 是循环携带的 FP32 累加值，
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

        # 【第二阶段：双 AIV 按任务分工】仍在同一次 SPMD 任务中，不另起 kernel。
        # NONE 不按行/列自动对半，保留本区域 tile 的完整形状；tail_lane 仍为
        # 组内 AIV 编号0/1。下面的 if 显式决定两路做不同工作：
        #   AIV0：归约 + Sinkhorn，写 comb；AIV1：归约 + 门控/残差混合，写 post/mixed。
        # 两路只共享读取第一阶段的 GM 部分和，写不同输出，无尾段结果互相依赖。
        for tail_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if tail_lane == 0:
                # 再按 worker 跨步分配 token 块，和第一阶段的 K 分段分配是两回事。
                # T96 时 token_tiles=12，若 clusters=24，仅组0～11各处理一个8行块；
                # 组12～23没有 Sinkhorn 块，但已参加上面的屏障。
                for ob in pl.range(worker, token_tiles, clusters):
                    # pl.read 在设备上取一个权重标量，不做 Tensor.item() 的 Host 回读。
                    scale2 = pl.read(hc_scale, [2])
                    round_start = (ob // clusters) * tail_round_rows
                    round_tokens = pl.min(tail_round_rows, tokens - round_start)
                    token_pairs = (round_tokens + 1) // 2
                    pairs_per_worker = token_pairs // clusters
                    extra_workers = token_pairs % clusters
                    worker_pairs = pairs_per_worker + pl.min(1, pl.max(0, extra_workers - worker))
                    t0 = round_start + (worker * pairs_per_worker + pl.min(worker, extra_workers)) * 2
                    valid_rows = pl.min(worker_pairs * 2, t_dim - t0)
                    if valid_rows > 0:
                        # 将全部 LINEAR_OK 个 K 分段按列相加，得到本8行的完整24列投影。
                        # 平方和也跨 K 分段归约：rstd = 1/sqrt(sum(x*x)/16384 + eps)。
                        # 投影先分段求和，再乘每个 token 的 rstd，完成 RMS 归一化的作用。
                        c_parts_load = pl.load(parts_2d, [0, t0 * MIX_PAD], [LINEAR_OK, T_TILE * MIX_PAD], valid_shape=[LINEAR_OK, valid_rows * MIX_PAD], target_memory=pl.MemorySpace.Vec)
                        c_parts = pl.set_validshape(pl.fillpad(c_parts_load, pad_value=pl.PadValue.zero), LINEAR_OK, T_TILE * MIX_PAD)
                        c_total = pl.reshape(pl.col_sum(c_parts), [T_TILE, MIX_PAD])
                        c_sq_load = pl.load(rms_2d, [0, t0], [LINEAR_OK, T_TILE], valid_shape=[LINEAR_OK, valid_rows], target_memory=pl.MemorySpace.Vec)
                        c_sq = pl.set_validshape(pl.fillpad(c_sq_load, pad_value=pl.PadValue.zero), LINEAR_OK, T_TILE)
                        c_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                        c_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(pl.col_sum(c_sq), HC_DIM_INV), NORM_EPS), c_rms_tmp), [T_TILE, 1])
                        comb_inv = c_inv
                        # 投影列8～23对应每个 token 的4×4 comb logit，使用第三个 scale。
                        raw = pl.add(pl.slice(c_total, [T_TILE, 16], [0, HC_MULT * 2]), 0.0)
                        base = pl.load(hc_base_2d, [0, HC_MULT * 2], [1, 16], target_memory=pl.MemorySpace.Vec)
                        logits = pl.add(pl.mul(pl.row_expand_mul(raw, comb_inv), scale2), pl.col_expand(raw, base))
                        # 展平8个4×4矩阵，用固定伙伴索引实现每行/每列的四元素归约。
                        # ri1/ri2 分别翻转矩阵内索引的低第0/1位，取同行的伙伴元素；
                        # ci1/ci2 翻转第2/3位，取同列伙伴。索引始终留在同一token的16项内。
                        # 两次 pairwise gather 即可合成四项；展平减少逐行 gather 开销。
                        logits = pl.reshape(pl.slice(logits, [COMB_T_TILE, 16], [0, 0]), [1, COMB_T_TILE * 16])
                        ids = pl.tile.arange(0, [1, COMB_T_TILE * 16], dtype=pl.INT32)
                        tmp = pl.create_tile([1, COMB_T_TILE * 16], dtype=pl.INT32)
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
                        max2 = pl.maximum(logits, pl.tile.gather(logits, ri1, tmp))
                        max4 = pl.maximum(max2, pl.tile.gather(max2, ri2, tmp))
                        exps = pl.exp(pl.sub(logits, max4))
                        sum2 = pl.add(exps, pl.tile.gather(exps, ri1, tmp))
                        sum4 = pl.add(sum2, pl.tile.gather(sum2, ri2, tmp))
                        probs = pl.add(pl.div(exps, sum4), HC_EPS)
                        col2 = pl.add(probs, pl.tile.gather(probs, ci1, tmp))
                        col4 = pl.add(pl.add(col2, pl.tile.gather(col2, ci2, tmp)), HC_EPS)
                        cur = pl.div(probs, col4)
                        # cur 是迭代状态，当前轮依赖上一轮；每个 AIV0 独立完成自己
                        # token 块的全部迭代，这里无需每轮跨核组同步。
                        for iteration in pl.range(HC_SINKHORN_ITER - 1):
                            row2 = pl.add(cur, pl.tile.gather(cur, ri1, tmp))
                            row4 = pl.add(pl.add(row2, pl.tile.gather(row2, ri2, tmp)), HC_EPS)
                            norm = pl.div(cur, row4)
                            col2 = pl.add(norm, pl.tile.gather(norm, ci1, tmp))
                            col4 = pl.add(pl.add(col2, pl.tile.gather(col2, ci2, tmp)), HC_EPS)
                            cur = pl.div(norm, col4)
                        # 只回写真实 token 行，最后不足8行的补齐部分不会写到输出外部。
                        valid = pl.set_validshape(pl.reshape(cur, [COMB_T_TILE, 16]), valid_rows, 16)
                        pl.store(valid, [t0, 0], comb)

            else:
                # 尾段任务展平为 (token块, D方向块)，然后继续按 worker 跨步分配。
                # 当前 D/D_SPMD=1，每任务覆盖8个token的完整4096维；T96、24组时，
                # 同样仅组0～11各处理一个任务，组12～23跳过这个循环体。
                for blk in pl.range(worker, token_tiles * (D // D_SPMD), clusters):
                    round_start = (blk // clusters) * tail_round_rows
                    round_tokens = pl.min(tail_round_rows, tokens - round_start)
                    token_pairs = (round_tokens + 1) // 2
                    pairs_per_worker = token_pairs // clusters
                    extra_workers = token_pairs % clusters
                    worker_pairs = pairs_per_worker + pl.min(1, pl.max(0, extra_workers - worker))
                    t0 = round_start + (worker * pairs_per_worker + pl.min(worker, extra_workers)) * 2
                    valid_rows = pl.min(worker_pairs * 2, t_dim - t0)
                    d_base = 0
                    if valid_rows > 0:
                        # AIV1 自己归约投影和平方和，独立得到 rstd；与 AIV0 有少量
                        # 重复归约，但无需等待 AIV0 再传一份中间结果，可以并行推进尾段。
                        m_parts_load = pl.load(parts_2d, [0, t0 * MIX_PAD], [LINEAR_OK, T_TILE * MIX_PAD], valid_shape=[LINEAR_OK, valid_rows * MIX_PAD], target_memory=pl.MemorySpace.Vec)
                        m_parts = pl.set_validshape(pl.fillpad(m_parts_load, pad_value=pl.PadValue.zero), LINEAR_OK, T_TILE * MIX_PAD)
                        m_total = pl.reshape(pl.col_sum(m_parts), [T_TILE, MIX_PAD])
                        m_sq_load = pl.load(rms_2d, [0, t0], [LINEAR_OK, T_TILE], valid_shape=[LINEAR_OK, valid_rows], target_memory=pl.MemorySpace.Vec)
                        m_sq = pl.set_validshape(pl.fillpad(m_sq_load, pad_value=pl.PadValue.zero), LINEAR_OK, T_TILE)
                        m_rms_tmp = pl.create_tile([1, T_TILE], dtype=pl.FP32)
                        m_inv = pl.reshape(pl.tile.rsqrt(pl.add(pl.mul(pl.col_sum(m_sq), HC_DIM_INV), NORM_EPS), m_rms_tmp), [T_TILE, 1])
                        # pre/post 共用八列归一化、偏置和 sigmoid，保留各自 scale。
                        # gate_scale_ids 为 [0,0,0,0,1,1,1,1]：前4列用 hc_scale[0]，
                        # 后4列用 hc_scale[1]；hc_scale[2] 留给 AIV0 的 comb。
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
                        gate_transposed = pl.transpose(gate_sigmoid, axis1=0, axis2=1)
                        # 展平后一次提取每行后四列，连续写回 post，避免 16B 行转置。
                        post_ids_float = pl.cast(pl.tile.arange(0, [1, T_TILE * HC_MULT], dtype=pl.INT32), pl.FP32)
                        post_rows = pl.cast(pl.cast(pl.mul(post_ids_float, 1.0 / HC_MULT), pl.INT32, mode="floor"), pl.FP32)
                        post_ids = pl.cast(pl.add(pl.add(post_ids_float, pl.mul(post_rows, HC_MULT)), HC_MULT), pl.INT32)
                        post_tmp = pl.create_tile([1, T_TILE * HC_MULT], dtype=pl.INT32)
                        post_values = pl.tile.gather(pl.reshape(gate_sigmoid, [1, T_TILE * HC_PAD]), post_ids, post_tmp)
                        # post = 2*sigmoid(post_logits)，对应每个 token 的后4列。
                        post_result = pl.set_validshape(pl.mul(post_values, 2.0), 1, valid_rows * HC_MULT)
                        pl.store(post_result, [0, t0 * HC_MULT], post_flat)
                        # pre = sigmoid(pre_logits)+eps，四个门值在该 token 的整个 D 轴复用；
                        # [T_TILE,1] 的形状用于对每行特征广播，因此放在 db 循环之外计算。
                        pre0 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [0, 0]), HC_EPS), [T_TILE, 1])
                        pre1 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [1, 0]), HC_EPS), [T_TILE, 1])
                        pre2 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [2, 0]), HC_EPS), [T_TILE, 1])
                        pre3 = pl.reshape(pl.add(pl.slice(gate_transposed, [1, T_TILE], [3, 0]), HC_EPS), [T_TILE, 1])

                        # 【AIV1 核内软件流水】沿 D 轴分成4096/512=8块，db 为0～7。
                        # 每块读取四路 [8,512] BF16，转 FP32，加权求和后转回 BF16 写出。
                        # 各 db 写入不同特征区间，没有跨 db 的数值累加依赖。
                        # stage=2 请求两级流水缓冲，使相邻块的 GM→UB 搬入、Vector
                        # 计算和 UB→GM 写出有机会重叠；编译器负责缓冲轮转与依赖同步。
                        # 它不表示用两个 AIV，也不会再派发两份任务；该循环仅由当前
                        # AIV1 执行。与上面的普通 rk 归约循环相比，这里显式申请了软件流水。
                        # 两级缓冲会增加 UB 占用，最终重叠程度仍取决于生成代码和数据依赖。
                        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
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
    # 参数顺序为：投影M块、K分段宽度、K分段数、RMS子块宽度、交接槽数、任务名。
    # 其中 11×1536=16896 大于真实 K=16384，最后一段的无效列由 valid_shape 补零；
    # 16×1024 与 32×512 则恰好覆盖 K。所有分支都执行同一套 HC_pre 数学计算。
    tokens = pl.tensor.dim(x, 0)
    if tokens <= SMALL_T_MAX:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               32, 1536, 11, 512, 1, 4, "hc_small_spmd")
    elif tokens <= MEDIUM48_T_MAX:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               48, 1024, 16, 512, 2, 4, "hc_medium48_spmd")
    elif tokens <= MEDIUM96_T_MAX:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               96, 512, 32, 256, 2, 4, "hc_medium96_spmd")
    else:
        mixed, post, comb_out = _hc_pre_variant(x, hc_fn, hc_scale, hc_base, mixed, post, comb_out,
                                               32, 1536, 11, 512, 1, 8, "hc_large_spmd")
    return mixed, post, comb_out
