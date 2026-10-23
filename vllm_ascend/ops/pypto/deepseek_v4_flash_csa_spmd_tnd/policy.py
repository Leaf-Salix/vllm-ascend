# SPDX-License-Identifier: Apache-2.0
"""动态TND CSA单SPMD的资源策略。

tokens、requests及compressed_table_columns是已分配容量；max_query_tokens和
max_seq_len是Host已知上界。
它们只选择tile、常量分支和工作区尺寸，实际Q/L、请求归属、可见历史、页表和
有效任务仍由device metadata决定；不通过D2H把动态内容变成Host调度表。

G2/G6、N64/N128、长短Score、QA/QB分工、Attention统计驻留和输出互补分工
沿已有验证区间选择。单请求可使用独立资源档位，全部数学、页保护、FIFO及
阶段完成协议仍由相应helper和调度函数保持。

PLAN_TABLE_COLUMNS在物理plan策略已开启且页表宽度为2049..2080时选择2080，
其余仍为2048。生产者/消费者共用该constexpr与device容量检查，不截短页表，
也不根据历史长度猜测已分配列数；超过容量仍保留原逻辑地址回退。

SCALE_PAGE_COALESCING启用时，连续页读取必须有本轮device证明。新增
LEAF_SCALE_LAYOUT只用于长G6、64<=T容量<=96且压缩长度上界<=6*8192：
每个Score叶在本核验证正/逆连续页，并仅归约一个有效行；META不再做整请求
证明。其它区间沿原META证明或逐页读取策略。当前仍保留原2*B尾区容量，
叶内路径不读写该证明区。它不是按B/S点枚举的固定任务表。

策略取舍与正负结果保存在DSV4_CSA_SPMD_TND_IMPLEMENTATION_LOG.md。
启用资源区间不表示所有动态输入都有同等收益，不能把代码等价当跨SDK计时。
"""

from dataclasses import dataclass

# At most six original 8192-candidate leaves before balancing.
LEAF_LAYOUT_MAX_CACHE_ROWS = 6 * 8192
DEFAULT_OA_ROWS = 128
RECTANGLE_OA_ROWS = 160
RECTANGLE_OA_COLS = 192
DEFAULT_PHYSICAL_PLAN_COLUMNS = 2048
EXTENDED_PHYSICAL_PLAN_COLUMNS = 2080

# 沿CSA已有投影行块选择窗口；不是按B/S/history测试点枚举。
EARLY_GEOMETRY_MIN_TOKENS = 64
# 与既有O_A N128输出区间一致；不是按七个测试点枚举。
LOCAL_STATS_MAX_TOKENS = 96
# 非balanced Score既有交接宽度；两AIV各消费一半。对应indexer_score_stages。
SHORT_SCORE_TILE = 768
# 既有TopK半叶直接归并的资源边界；更大范围用2560/3072/4096分支。
SHORT_PREFETCH_HALF_SORT_LIMIT = 2048
# 沿已有QA的小M16并行区间边界，两个完整M16行块起再启用Score新流水。
SCORE_SPLIT_MIN_TOKENS = 32
# At most two query-group work items per each of the 24 physical groups.
SHORT_HALF_SCORE_GROUP_LIMIT = 48


@dataclass(frozen=True)
class CSASPMDPolicy:
    query_group: int
    qk_panel: int
    score_tile: int
    prefetch: int
    balanced: bool
    roots: int
    oa_cols: int
    ob_rows: int
    oa_rows: int = DEFAULT_OA_ROWS
    output_group_pipeline: bool = True
    hc_row_dependencies: bool = True
    attention_query_metadata: bool = False
    hc_consumer_reduce: bool = False
    qa_m16_parallel: bool = False
    cooperative_prefix: bool = False
    early_attention_geometry: bool = False
    local_attention_stats: bool = False
    scale_page_coalescing: bool = False
    leaf_scale_layout: bool = False
    plan_physical_rows: bool = False
    split_score_rows: bool = False
    output_work_balance: bool = False
    short_topk_prefix: bool = False
    short_half_score_ub: bool = False
    qb_wide_n: bool = False
    prefix_branch_groups: int = 4
    joint_bf16_prefix: bool = False
    prefix_projection_rows: int = 128
    score_group_order: bool = False
    plan_table_columns: int = DEFAULT_PHYSICAL_PLAN_COLUMNS

    def constexprs(self) -> dict[str, int | bool]:
        return {
            "QUERY_GROUP": self.query_group,
            "QK_PANEL": self.qk_panel,
            "SCORE_TILE": self.score_tile,
            "PREFETCH": self.prefetch,
            "TO_L0": True,
            "BALANCED": self.balanced,
            "ROOTS": self.roots,
            "OA_COLS": self.oa_cols,
            "OA_ROWS": self.oa_rows,
            "OB_ROWS": self.ob_rows,
            "OUTPUT_GROUP_PIPELINE": self.output_group_pipeline,
            "HC_ROW_DEPENDENCIES": self.hc_row_dependencies,
            "ATTENTION_QUERY_METADATA": self.attention_query_metadata,
            "HC_CONSUMER_REDUCE": self.hc_consumer_reduce,
            "QA_M16_PARALLEL": self.qa_m16_parallel,
            "COOPERATIVE_PREFIX": self.cooperative_prefix,
            "EARLY_ATTENTION_GEOMETRY": self.early_attention_geometry,
            "LOCAL_ATTENTION_STATS": self.local_attention_stats,
            "SCALE_PAGE_COALESCING": self.scale_page_coalescing,
            "LEAF_SCALE_LAYOUT": self.leaf_scale_layout,
            "PLAN_PHYSICAL_ROWS": self.plan_physical_rows,
            "SPLIT_SCORE_ROWS": self.split_score_rows,
            "OUTPUT_WORK_BALANCE": self.output_work_balance,
            "SHORT_TOPK_PREFIX": self.short_topk_prefix,
            "SHORT_HALF_SCORE_UB": self.short_half_score_ub,
            "QB_WIDE_N": self.qb_wide_n,
            "PREFIX_BRANCH_GROUPS": self.prefix_branch_groups,
            "JOINT_BF16_PREFIX": self.joint_bf16_prefix,
            "PREFIX_PROJECTION_ROWS": self.prefix_projection_rows,
            "SCORE_GROUP_ORDER": self.score_group_order,
            "PLAN_TABLE_COLUMNS": self.plan_table_columns,
        }


def select_policy(
    *, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int, compressed_table_columns: int | None = None
) -> CSASPMDPolicy:
    """仅依据已有Host信息选策略；有效分组数、叶数及全部访问地址由AIV/AIC计算。"""
    if compressed_table_columns is not None:
        if type(compressed_table_columns) is not int or compressed_table_columns <= 0:
            raise ValueError("compressed_table_columns must be a positive allocated dimension")
    values = (tokens, requests, max_query_tokens, max_seq_len)
    if any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("tokens/requests/max_query_tokens/max_seq_len须为已有的正整数描述符")
    if tokens > requests * max_query_tokens:
        raise ValueError("token容量超过请求容量×query上界；请绑定Native同一档描述符")
    cache_rows = max_seq_len // 4
    if cache_rows < 2048:
        raise NotImplementedError("完整动态SPMD的小历史Vector Score路径尚未接入")
    if cache_rows > 262144:
        raise ValueError("当前TopK工作区合同最多支持262144个压缩候选")
    long_history = cache_rows > 8192
    group = 2
    if long_history:
        # 24个query起可用多叶填满24组；query长度不足6时不为复用Key扩张M384。
        if tokens >= 24 and max_query_tokens >= 6:
            group = 6
    elif max_query_tokens >= 6 and requests >= 24:
        # 对工作量上界比较，而不是声明T/6就是实际请求数；不等长请求仍在device
        # 独立ceil分组。首版用已有原策略的最忙核增幅上限25%作为保守启发式。
        pair_waves = ((tokens + 1) // 2 + 23) // 24
        request_waves = (requests + 23) // 24
        if 4 * 6 * request_waves <= 5 * 2 * pair_waves:
            group = 6
    # visible受请求length//4钳制，不超过这里已有的Host长度上界。用上界
    # 保守估算半叶工作量，不新增device回读，不代替核内真实范围与尾块检查。
    # 该资源区间实测支持预取；较大半叶与短G6仍保持原路径。
    short_steps_upper = (cache_rows + SHORT_SCORE_TILE - 1) // SHORT_SCORE_TILE
    short_half_span_upper = short_steps_upper * (SHORT_SCORE_TILE // 2)
    short_key_prefetch = group == 2 and short_half_span_upper <= SHORT_PREFETCH_HALF_SORT_LIMIT
    # 用户允许单请求独立分档。仅改变已有G2的历史工作分配，
    # 保留动态Q/L及原半叶边界（末叶可小于4096）；较大容量及S1维持原策略。
    single_request_leaves = requests == 1 and long_history and cache_rows <= 65536 and 5 <= max_query_tokens <= 6
    ob_rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128
    parallel_small_qa = group == 6 and 16 < tokens < 32
    leaf_scale_layout = (
        long_history
        and group == 6
        and (EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS)
        and cache_rows <= LEAF_LAYOUT_MAX_CACHE_ROWS
    )
    plan_physical_rows = tokens >= EARLY_GEOMETRY_MIN_TOKENS
    plan_table_columns = DEFAULT_PHYSICAL_PLAN_COLUMNS
    if (
        plan_physical_rows
        and compressed_table_columns is not None
        and DEFAULT_PHYSICAL_PLAN_COLUMNS < compressed_table_columns <= EXTENDED_PHYSICAL_PLAN_COLUMNS
    ):
        plan_table_columns = EXTENDED_PHYSICAL_PLAN_COLUMNS
    # One M tile removes repeated weight loads; actual rows remain device-side.
    output_rectangle = DEFAULT_OA_ROWS < tokens <= RECTANGLE_OA_ROWS
    return CSASPMDPolicy(
        query_group=group,
        qk_panel=64 if group == 6 else 128,
        score_tile=1024 if long_history else SHORT_SCORE_TILE,
        prefetch=1 if long_history or short_key_prefetch else 0,
        balanced=(long_history and group == 6) or single_request_leaves,
        roots=1 if (long_history and group == 6) or single_request_leaves else 2,
        oa_cols=RECTANGLE_OA_COLS if output_rectangle else (128 if tokens <= 96 else 256),
        oa_rows=RECTANGLE_OA_ROWS if output_rectangle else DEFAULT_OA_ROWS,
        ob_rows=ob_rows,
        # Does not infer real query lengths; only the existing Host token-capacity tier.
        # Shared-SDK three-way evidence supports the existing long-G6 [32,64) domain.
        # G2 wide-N/two-core prefix showed no benefit; keep the original four producers.
        qb_wide_n=(long_history and group == 6 and SCORE_SPLIT_MIN_TOKENS <= tokens < EARLY_GEOMETRY_MIN_TOKENS),
        prefix_branch_groups=4,
        # Keep the Indexer chain early; its four Hadamard cores skip main/KV work.
        # Restrict the joint grid to long-G6 single-M192: B16 and short-history
        # measurements regressed. Real request membership remains device-side.
        joint_bf16_prefix=long_history and group == 6 and 128 < tokens <= 192,
        prefix_projection_rows=192 if 128 < tokens <= 192 else 128,
        # Keep the measured mixed case in the existing [32, 64) Q_B resource tier.
        # Real group membership and the heavy/light proof remain device-side.
        score_group_order=(
            long_history
            and group == 6
            and tokens >= SCORE_SPLIT_MIN_TOKENS
            and tokens < EARLY_GEOMETRY_MIN_TOKENS
            and tokens < requests * group
        ),
        # Host upper bound only; real mixed-query membership remains on device.
        short_half_score_ub=(
            not long_history and group == 2 and requests * ((max_query_tokens + 1) // 2) <= SHORT_HALF_SCORE_GROUP_LIMIT
        ),
        # 沿已有G6分界选择两M16尾块的并行；G2已观察回退，不按B4/B5白名单。
        qa_m16_parallel=parallel_small_qa,
        cooperative_prefix=True,
        # 前置几何在小M16并行域及较大T区间有已测收益；中间单M32主体加尾块
        # 的域先保留后置算法，避免把metadata放到RoPE关键路径。64表示至少
        # 两个完整M32行块，仍须边界验证；不代表所有形状下都是理论最优。
        early_attention_geometry=parallel_small_qa or tokens >= EARLY_GEOMETRY_MIN_TOKENS,
        # 同一AIV生产和消费的mi/li留在UB三槽；只在长历史G6/N128域默认启用。
        # B4/B8/B15/B16重复测得收益；短历史T>96两点方向不一致，保留原GM。
        # 实际T/请求边界仍在device读取；依赖编译器删除关闭分支的死Tile状态。
        local_attention_stats=long_history and group == 6 and tokens <= LOCAL_STATS_MAX_TOKENS,
        # 连续scale读取需要先在META证明页布局；该工作并非所有负载都划算。
        # 沿已有小M16并行域和O_A N256的较大T域启用，中间域保留逐页读取。
        # 不按B/history验收点枚举；页是否连续仍由本轮device表完整证明。
        # 小M16与较大T边界已做完整同轮验证；其余验证边界见统一优化日志。
        scale_page_coalescing=(
            leaf_scale_layout
            or (long_history and group == 6 and (parallel_small_qa or tokens > LOCAL_STATS_MAX_TOKENS))
        ),
        leaf_scale_layout=leaf_scale_layout,
        # 沿已有至少两个M32行块的前置几何区间，统一在plan验证选中页，
        # 并把压缩逻辑索引转换成物理行；消费者不再重复Q/L扫描和页表查询。
        # 小T的计划生产者更少，搬移地址工作未稳定获益，保留原逻辑表示。
        # 此处只选策略；容量合法性和全部地址仍由device元数据决定，三阶段
        # 共享同一谓词，不能仅开启生产者或消费者。区间实测与限制见优化日志。
        plan_physical_rows=plan_physical_rows,
        plan_table_columns=plan_table_columns,
        # 小T保持原Score，中大T使用早FIX与单次WS写回。分界沿已有M16
        # 前段资源区间；依据完整实测取舍，不把回退唯一归因于QA或假定最优。
        # 实际有效query/长度/页表仍在device读取，不按B4/B8等测试点枚举。
        split_score_rows=long_history and group == 6 and tokens >= SCORE_SPLIT_MIN_TOKENS,
        # Full-chain gains validated below the existing 64-token resource boundary.
        # The retained prefix length and request membership remain device-side.
        short_topk_prefix=(
            long_history and group == 6 and SCORE_SPLIT_MIN_TOKENS <= tokens < EARLY_GEOMETRY_MIN_TOKENS
        ),
        # B1 reuses its existing balanced half-leaf domain. Multiple-request
        # redistribution is limited to long G6 and the existing 64..96-token
        # resource interval. Short G2/B16 regressed despite exact outputs;
        # neither equal tile counts nor source-level balance proves speedup.
        output_work_balance=(
            single_request_leaves
            or (long_history and group == 6 and EARLY_GEOMETRY_MIN_TOKENS <= tokens <= LOCAL_STATS_MAX_TOKENS)
        ),
    )
