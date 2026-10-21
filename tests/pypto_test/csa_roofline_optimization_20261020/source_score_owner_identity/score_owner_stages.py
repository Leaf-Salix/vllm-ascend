# SPDX-License-Identifier: Apache-2.0
"""仅作同接口身份映射对照，禁止作为生产优化。

    正常balanced输入的标志非零，本对照把唯一guard改为==0，故保留身份映射。
    与候选相比接口、容量、消费者及代码结构相同；不依赖true恰为-1或+1。

按本轮Score工作量与前段KV负载，置换Score的物理核归属。

整体思路
    不拆叶、不改变query组、tile或排序树。先统计原24个逻辑worker各自的
    1024候选步数，再把较短的逻辑worker交给前段KV较重的物理核。AIC和
    同组双AIV使用同一个映射；根消费者用逆映射等待真实物理写者。
    每个逻辑worker的工作顺序、GM交接槽和根写入位置保持原样，FFTS仍在
    原物理核组内握手，因此不是将一条流水的三种角色拆散重派。

动态数据和布局
    只在既有META单写者上计算，读取本轮group/leaf计划及各组visible。
    group_offsets尾部为布局证明保留2*B字后，追加四份24项INT32：
    Score步数、前段负载排序键、physical→logical、logical→physical。
    常量24、4、128、64、16来自既有A3核数及Q_B/KV资源划分，不是B28表。
    仅当真实group数在24..48且至少相差两个Score步时排序；其它情况写身份
    映射。Host仅选择两个M128盒的资源区间，不回读device长度或页表。

前段排序键
    精确复用KV的logical=(physical+20)%24、M64主体/M16尾和最多3行组
    分配，按每核实际计算的M行数排序；同M行数按原Q_B列块数打破平局。
    这是负载排序启发式，不把行数当作实测时间，也不声称它是最优排程。
    PREFILL/其它资源区间不启用，BRANCH_GROUPS须与前段调用保持一致。

同步和限制
    只有原META planner写这些相邻Scalar缓存行，PHASE_META负责发布；
    其它角色沿原依赖链获取，不能用普通GM标记替代DCCI/DSB。映射每轮
    重算，身份路径也覆盖全部映射，不复用上一轮结果。没有新task、信号
    阶段、atomic或业务根Tensor。代价包含META统计/排序及消费者的映射读，
    全部计入完整L1；是否采用必须由精度和真机配对决定。
"""

import pypto.language as pl

from .indexer_group_stages import (
    GROUP_DESCRIPTOR_ALIGNMENT, GROUP_DESCRIPTOR_WORDS, OFFSET_ROWS_DYN,
    MAX_LEAVES, PLAN_WORDS, SCORE_TILE, indexer_leaf_range,
)
from .indexer_scale_layout_stages import LAYOUT_WORDS_PER_REQUEST, indexer_scale_layout_base
from .kv_stages import HEAD_DIM, KV_DENSE_M_TILE, KV_FIXED_SMALL_ROWS, KV_M_TILE, KV_N_TILE, KV_OM
from .phase_sync import SIGNAL_SLOT_WORDS, SIGNAL_WORDS
from .query_projection_stages import QPROJ_N_BLOCKS, QPROJ_PIPE_M_TILE
from .runtime_abi import CORES, GROUPS, PHASE_SCORE_VECTOR, VECTORS
from .signals import csa_signal_wait

SCORE_BALANCE_BRANCH_GROUPS = 4
SCORE_BALANCE_WORDS = 4 * GROUPS
SCORE_MIN_IMBALANCE = 2  # 至少一个双槽周期，才支付24×24的Scalar选配成本。
SCORE_COST_SENTINEL = 1 << 30  # 大于当前合法工作量域；仅供有限集合最小值初始化。


@pl.jit.inline(auto_scope=False)
def score_balance_base(requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX]):
    # 即便诊断关闭布局证明，开启映射也保留这2*B项，避免两种尾区重叠。
    layout_base = indexer_scale_layout_base(requests, tokens, VECTORS)
    return layout_base + LAYOUT_WORDS_PER_REQUEST * requests


@pl.jit.inline(auto_scope=False)
def score_balance_plan_stage(
    workspace: pl.InOut[pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32]],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    requests: pl.Scalar[pl.INDEX],
    tokens: pl.Scalar[pl.INDEX],
):
    """原META单写者统计和选配；先写身份映射，所有回退均保持合法双射。"""
    base = score_balance_base(requests, tokens)
    for core in pl.range(GROUPS):
        pl.write(workspace, [base + core], pl.cast(0, pl.INT32))
        pl.write(workspace, [base + GROUPS + core], pl.cast(0, pl.INT32))
        pl.write(workspace, [base + 2 * GROUPS + core], pl.cast(core, pl.INT32))
        pl.write(workspace, [base + 3 * GROUPS + core], pl.cast(core, pl.INT32))
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    leaf_tiles = pl.cast(pl.read(plan, [3]), pl.INDEX)
    extra = pl.cast(pl.read(plan, [4]), pl.INDEX)
    if (groups >= GROUPS and groups <= 2 * GROUPS and leaves > 0 and leaves <= MAX_LEAVES
            and leaf_tiles > 0 and extra >= 0 and extra < leaves
            and tokens > KV_FIXED_SMALL_ROWS and tokens <= 2 * QPROJ_PIPE_M_TILE
            and pl.read(plan, [5]) == 6 and pl.read(plan, [6]) == 0):
        descriptor_base = ((requests + 1 + GROUP_DESCRIPTOR_ALIGNMENT - 1)
                           // GROUP_DESCRIPTOR_ALIGNMENT * GROUP_DESCRIPTOR_ALIGNMENT)
        # 原group-major工作项逐项统计。只读真实visible，不用max_history替代
        # 每组有效工作；不改任何访问地址或叶内排序边界。
        for group in pl.range(groups):
            visible = pl.cast(pl.read(workspace, [descriptor_base + GROUP_DESCRIPTOR_WORDS * group + 3]), pl.INDEX)
            for leaf in pl.range(leaves):
                _begin, _capacity, valid = indexer_leaf_range(leaf, leaf_tiles, extra, visible, True, 6)
                steps = (valid + SCORE_TILE - 1) // SCORE_TILE
                logical = (group * leaves + leaf) % GROUPS
                old = pl.read(workspace, [base + logical])
                pl.write(workspace, [base + logical], old + pl.cast(steps, pl.INT32))
        maximum = pl.cast(0, pl.INT32)
        minimum = pl.cast(SCORE_COST_SENTINEL, pl.INT32)
        for logical in pl.range(GROUPS):
            cost = pl.read(workspace, [base + logical])
            maximum = pl.max(maximum, cost)
            minimum = pl.min(minimum, cost)
        if maximum - minimum >= SCORE_MIN_IMBALANCE:
            full_rows = tokens // KV_DENSE_M_TILE * KV_DENSE_M_TILE
            padded_rows = (tokens + KV_M_TILE - 1) // KV_M_TILE * KV_M_TILE
            row_groups = pl.min(KV_OM, pl.max(1, tokens // KV_DENSE_M_TILE))
            for physical in pl.range(GROUPS):
                kv_logical = (physical + GROUPS - SCORE_BALANCE_BRANCH_GROUPS) % GROUPS
                kv_rows = pl.cast(0, pl.INDEX)
                if kv_logical < (HEAD_DIM // KV_N_TILE) * row_groups:
                    row_group = kv_logical % row_groups
                    full_stride = row_groups * KV_DENSE_M_TILE
                    tail_stride = row_groups * KV_M_TILE
                    full_blocks = pl.max((full_rows - row_group * KV_DENSE_M_TILE + full_stride - 1)
                                         // full_stride, 0)
                    tail_blocks = pl.max((padded_rows - full_rows - row_group * KV_M_TILE + tail_stride - 1)
                                         // tail_stride, 0)
                    kv_rows = full_blocks * KV_DENSE_M_TILE + tail_blocks * KV_M_TILE
                qb_columns = pl.cast(0, pl.INDEX)
                if physical >= SCORE_BALANCE_BRANCH_GROUPS:
                    qb_workers = GROUPS - SCORE_BALANCE_BRANCH_GROUPS
                    qb_worker = physical % qb_workers
                    qb_columns = (QPROJ_N_BLOCKS - qb_worker + qb_workers - 1) // qb_workers
                # 词典序排序键：KV的M行数优先，Q_B列块数只打破平局，不拟合μs。
                prefix_rank = kv_rows * (QPROJ_N_BLOCKS + 1) + qb_columns
                pl.write(workspace, [base + GROUPS + physical], pl.cast(prefix_rank, pl.INT32))
                pl.write(workspace, [base + 2 * GROUPS + physical], pl.cast(-1, pl.INT32))
                pl.write(workspace, [base + 3 * GROUPS + physical], pl.cast(-1, pl.INT32))
            # 各轮各选一个未分配physical与logical，稳定平局选择较小编号。
            # 24轮后两个表为逆置换，不需要跨核atomic或运行中的抢任务队列。
            for rank in pl.range(GROUPS):
                best_physical = pl.cast(-1, pl.INDEX)
                best_prefix = pl.cast(-1, pl.INT32)
                best_logical = pl.cast(-1, pl.INDEX)
                best_cost = pl.cast(SCORE_COST_SENTINEL, pl.INT32)
                for item in pl.range(GROUPS):
                    if pl.read(workspace, [base + 2 * GROUPS + item]) < 0:
                        prefix_cost = pl.read(workspace, [base + GROUPS + item])
                        if prefix_cost > best_prefix:
                            best_prefix = prefix_cost
                            best_physical = item
                    if pl.read(workspace, [base + 3 * GROUPS + item]) < 0:
                        score_cost = pl.read(workspace, [base + item])
                        if score_cost < best_cost:
                            best_cost = score_cost
                            best_logical = item
                # 上述有限集合的最值必然存在；保护索引以避免异常计划越界写。
                if best_physical >= 0 and best_logical >= 0:
                    pl.write(workspace, [base + 2 * GROUPS + best_physical], pl.cast(best_logical, pl.INT32))
                    pl.write(workspace, [base + 3 * GROUPS + best_logical], pl.cast(best_physical, pl.INT32))
    return workspace


@pl.jit.inline(auto_scope=False)
def score_logical_worker(
    workspace: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX],
    physical: pl.Scalar[pl.INDEX],
):
    base = score_balance_base(requests, tokens)
    return pl.cast(pl.read(workspace, [base + 2 * GROUPS + physical]), pl.INDEX)


@pl.jit.inline(auto_scope=False)
def wait_balanced_score_roots(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    workspace: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    plan: pl.Tensor[[PLAN_WORDS], pl.INT32],
    requests: pl.Scalar[pl.INDEX], tokens: pl.Scalar[pl.INDEX],
    group: pl.Scalar[pl.INDEX], local_query: pl.Scalar[pl.INDEX],
    root_count: pl.Scalar[pl.INDEX], generation: pl.Scalar[pl.INT32],
    QUERY_GROUP: pl.constexpr,
):
    """根布局不变，只将原logical owner翻译成物理发布者；完整获取协议保留。"""
    groups = pl.cast(pl.read(plan, [0]), pl.INDEX)
    leaves = pl.cast(pl.read(plan, [2]), pl.INDEX)
    base = score_balance_base(requests, tokens)
    lane = local_query // (QUERY_GROUP // 2)
    for leaf in pl.range(root_count):
        item = group * leaves + leaf
        if groups < GROUPS:
            item = leaf * groups + group
        logical = item % GROUPS
        physical = pl.cast(pl.read(workspace, [base + 3 * GROUPS + logical]), pl.INDEX)
        producer = GROUPS + 2 * physical + lane
        offset = (PHASE_SCORE_VECTOR * CORES + producer) * SIGNAL_SLOT_WORDS
        signals = csa_signal_wait(signals, offset, generation)
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signals
