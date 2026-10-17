# SPDX-License-Identifier: Apache-2.0
"""动态CSA的区间策略：Host只选择数学tile，逐请求有效范围仍由device计算。

输入tokens/requests/max_query_tokens来自已有Tensor shape和Native描述符；
max_seq_len来自Native已有Host上界，禁止为了调用此函数新增device回读。
不同策略各自编译和缓存，档内T/B/Q/L/position保持动态。这里不出现七档
验收点白名单，亦不从T/B假定所有请求长度相等。

初始策略保留原CSA的Cube Score和输出投影数学：短/中历史双query或六query
分组；长历史有足够query时通过均衡历史叶增加并行度。分组6只是至多6个
query的数学tile，每请求可以拆成多组并保留尾组，不要求每请求恰好6行。
原小历史Vector Score尚待迁移，因此暂不接收max_seq_len//4<2048。这个
边界是当前实现能力，不是性能收益结论；所有策略仍须完整设备和性能验证。
输出尾段的组流水与列tile独立选择：每个输出组就绪后可先量化，O_B只等
自己读取的组。默认保留N128/N256原区间，启用按组通知；完整同进程对照
在T96长历史及T192短历史确认约2%/3%的相对收益，其他范围继续回归，
不能把该局部策略收益当作完整CSA已比原分task快30%。关闭开关仍复用数学体。
HC行依赖只按现有M16/K4与M8布局缩小等待，不改变数学、分工或动态范围。
七个固定S6回归点已通过完整精度/图/保护检查，默认开启；T96长/短历史
同进程开关对照分别降低完整CSA耗时2.13%/3.03%，不声称所有区间均有同轮消融。
关闭分支保留用于定位，完整动态场景与30%性能目标仍以统一日志为准。
阅读顺序：不可变策略→输入合同→Indexer分档→输出投影分档。
Attention查询几何复用候选使用已有mask行空余列，按次写入/发布，不缓存
上一轮请求信息。完整同进程对照已通过：128K/B16慢1.66%，8K/B32快1.33%，
收益不一致，默认保持关闭；候选和原始证据保留，不据此建立新的策略分界。
HC消费者归约候选让门值/Sinkhorn直接加载四份投影部分和，省去RAW交接。
128K/B16与8K/B32均与原分task精确一致；同进程五窗口完整L1分别慢1.06%、
快2.54%，因此默认关闭，尚不据两个点划定启用区间。见统一优化日志。
"""

from dataclasses import dataclass


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
    output_group_pipeline: bool = True
    hc_row_dependencies: bool = True
    attention_query_metadata: bool = False
    hc_consumer_reduce: bool = False

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
            "OB_ROWS": self.ob_rows,
            "OUTPUT_GROUP_PIPELINE": self.output_group_pipeline,
            "HC_ROW_DEPENDENCIES": self.hc_row_dependencies,
            "ATTENTION_QUERY_METADATA": self.attention_query_metadata,
            "HC_CONSUMER_REDUCE": self.hc_consumer_reduce,
        }


def select_policy(*, tokens: int, requests: int, max_query_tokens: int, max_seq_len: int) -> CSASPMDPolicy:
    """仅依据已有Host信息选策略；有效分组数、叶数及全部访问地址由AIV/AIC计算。"""
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
    ob_rows = 32 if tokens <= 32 else 96 if tokens <= 96 else 128
    return CSASPMDPolicy(
        query_group=group,
        qk_panel=64 if group == 6 else 128,
        score_tile=1024 if long_history else 768,
        prefetch=1 if long_history else 0,
        balanced=long_history and group == 6,
        roots=1 if long_history and group == 6 else 2,
        oa_cols=128 if tokens <= 96 else 256,
        ob_rows=ob_rows,
    )
