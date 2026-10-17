# SPDX-License-Identifier: Apache-2.0
"""CSA 工作量区间入口；性能回归点只提供证据，不决定支持白名单。

这里区分原数学代码的兼容边界与尚需测量的调度策略。Host 只使用调用方
已有的形状和长度上界；核内仍读取本轮 device metadata。当前核体来自 S=6
接口，不能因为入口接受 query 参数就声称已经支持任意 S。
"""

from dataclasses import asdict, dataclass

QUERY_TOKENS = 6
MAX_TOKENS = 384
MAX_SEQ_LEN = 131200
VECTOR_LEAF_LIMIT = 2048
MEDIUM_LEAF_LIMIT = 8192
COMPRESS_RATIO = 4
CORE_GROUPS = 24


@dataclass(frozen=True)
class RangePolicy:
    batch: int
    tokens: int
    query_tokens: int
    max_seq_len: int
    history_family: str
    query_group: str
    projection_family: str
    output_projection_family: str
    schedule_policy: str
    qk_vector_indices: bool
    row_dependencies: bool

    def build_options(self):
        return dict(
            balanced_schedule=True,
            schedule_policy=self.schedule_policy,
            qk_vector_indices=self.qk_vector_indices,
            row_dependencies=self.row_dependencies,
        )

    def report(self):
        return dict(
            asdict(self),
            build_options=self.build_options(),
            cached_wait=True,
            wait_backoff_ticks=100,
            range_device_validated=False,
            range_performance_validated=False,
            note="原数学兼容边界与候选调度；七点证据不能证明整个区间",
        )


def select_policy(*, batch, tokens, max_seq_len, query_tokens=QUERY_TOKENS):
    """选取策略，不触碰 Tensor 数值，也不把历史 H 与本轮总长度 L 混用。

    max_seq_len 是包含本轮 query 的 L。测试 fixture 的历史 H 对应 L=H+S。
    长度边界来自压缩后候选数，故不能直接用测试名称 8K/128K 划界。
    """
    if any(type(value) is not int for value in (batch, tokens, max_seq_len, query_tokens)):
        raise ValueError("策略输入必须为调用方已有的 Host 整数描述信息")
    if query_tokens != QUERY_TOKENS:
        raise ValueError("当前核体仅支持每请求 S=6；其他 S 尚待实现，不能套用模板")
    if batch <= 0 or tokens != batch * query_tokens or not 1 <= tokens <= MAX_TOKENS:
        raise ValueError("当前形状要求 T=B×6，1≤B≤64；另须检查实际工作区容量")
    if not query_tokens <= max_seq_len <= MAX_SEQ_LEN:
        raise ValueError("当前总长度 L 需要在 S..131200 范围内，且已有缓存容量足够")

    compressed = max_seq_len // COMPRESS_RATIO
    if compressed < VECTOR_LEAF_LIMIT:
        history_family, query_group = "vector_leaf", "single_query"
    elif compressed <= MEDIUM_LEAF_LIMIT:
        history_family = "medium_cube"
        # 原数学分支比较两种分组的核波次数；随 B 增长并非单调切换。
        pair_waves = (tokens + 46) // 48
        request_waves = (tokens + 138) // 144
        use_request = tokens >= 144 and request_waves * CORE_GROUPS <= pair_waves * 10
        query_group = "request_s6" if use_request else "query_pair"
    else:
        history_family = "long_cube"
        query_group = "request_s6" if tokens >= 24 else "query_pair"

    projection_family = "small_rows" if tokens < 128 else "tiled_rows"
    if tokens <= 32:
        output_projection_family = "small_rows"
    elif tokens < 96:
        output_projection_family = "medium_rows"
    elif tokens == 96:
        # 这是现有 CSA 数学的分支，不是借用 HC_pre 的 T=96 配置。
        output_projection_family = "existing_csa_t96"
    else:
        output_projection_family = "large_rows"

    # 初始范围策略保持七个已测点的选择；推广到区间内其他 shape 尚需验证。
    # 这里只决定排程及已核对的索引缓存变换，不声称这些阈值已是最优边界。
    if history_family == "long_cube":
        scheduling = "earliest_ready" if tokens <= 96 else "critical_path"
    else:
        scheduling = "earliest_ready" if query_group == "request_s6" else "critical_path"
    vector_indices = history_family == "long_cube" and 32 < tokens <= 96
    # 当前成员级行依赖证明只覆盖此 CSA 输出所有权合同。扩大范围须重新证明。
    row_dependencies = history_family == "long_cube" and tokens == 96
    return RangePolicy(
        batch,
        tokens,
        query_tokens,
        max_seq_len,
        history_family,
        query_group,
        projection_family,
        output_projection_family,
        scheduling,
        vector_indices,
        row_dependencies,
    )


def policy_from_descriptors(descriptors, *, max_seq_len, query_tokens=QUERY_TOKENS):
    return select_policy(
        batch=descriptors["kv_seq_lens"]["shape"][0],
        tokens=descriptors["x_hc"]["shape"][0],
        max_seq_len=max_seq_len,
        query_tokens=query_tokens,
    )
