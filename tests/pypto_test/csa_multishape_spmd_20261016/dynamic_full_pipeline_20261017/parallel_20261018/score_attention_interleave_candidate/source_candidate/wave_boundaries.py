# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""动态CSA的group wave边界准备：仅计算device边界，不执行Score/Attention。

整体思路与接口
    next_wave_boundary读取同一轮已发布group_offsets描述符，返回下一个全局
    group编号和当前wave的query_end。输入lengths只提供真实请求数B，tokens
    来自本轮实际T（调用者从device Tensor shape取），不用B*S推导token。
    group_offsets前B+1项是请求组前缀，offsets[B]为groups；pad16(B+1)后是
    四项(request, first_query, live, max_visible)描述符，与当前正式分组源同约定。
    本helper只读first_query/live，不改变request/visible、Score owner或数学。

区间、动态变量与调用方法
    每个非末wave至少包含8个真实非空group。从first_group+8开始向后扫描，
    取第一个first_query整除8的下一group作为截点；没有截点就延至groups/T。
    8对应现有Attention plan的M8行块，不是batch白名单或冻结S。group可来自
    任意请求长度/混合长度/尾组；有效group数、位置和间隙都是运行时数据。
    调用者令first_group=0、query_begin=0，并把前次query_end作为下次begin；
    首wave包含开头dummy，最后wave的end=T包含尾dummy，组间gap落入前一wave。
    query边界只能变更消费者处理范围；不能在wave内重新编号group/item/worker，
    也不能把每wave的group数送入原全局indexer_work_item替代真实groups。

依赖、状态与容量
    这是AIC/AIV共同使用的Scalar只读helper，没有Tile、GM写、task、notify/wait
    或额外工作区。所有角色必须先完成原metadata acquire，并保证描述符在本次
    调用中不变；相同first_group必须得到相同边界，不能凭计算完边界认定根已发布。
    只对正式生成器的合法、非空、单调、不重叠、位于[0,T)内的group描述符承诺：
    wave数<=ceil(groups/8)<=ceil(T/8)。其中末wave可少于8组。groups==0时返回
    (0,T)，上层必须走原不交错路径处理dummy/gap，不能把该返回值计为一个wave。
    与同步代理约定Host只按ceil(T/8)保守预留通知槽；实际wave数和边界仍在device。

当前边界
    已接入候选调度；联合编译尚未通过，未上卡。纯Python检查只能核实边界语义，
    不能证明Scalar lowering、缓存可见性、阶段依赖或任何性能收益。
"""

import pypto.language as pl

from .indexer_group_stages import (
    GROUP_DESCRIPTOR_ALIGNMENT,
    GROUP_DESCRIPTOR_WORDS,
    OFFSET_ROWS_DYN,
)
from .request_geometry import REQUESTS_DYN

MIN_GROUPS_PER_WAVE = 8
PLAN_QUERY_ROWS = 8


@pl.jit.inline(auto_scope=False)
def next_wave_boundary(
    offsets: pl.Tensor[[OFFSET_ROWS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    tokens: pl.Scalar[pl.INDEX],
    first_group: pl.Scalar[pl.INDEX],
) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX]]:
    """给出半开group区间的末端与query_end；输入描述符须来自同次metadata。"""
    requests = pl.cast(pl.tensor.dim(lengths, 0), pl.INDEX)
    groups = pl.cast(pl.read(offsets, [requests]), pl.INDEX)
    descriptor_base = (
        (requests + 1 + GROUP_DESCRIPTOR_ALIGNMENT - 1)
        // GROUP_DESCRIPTOR_ALIGNMENT
        * GROUP_DESCRIPTOR_ALIGNMENT
    )
    initial_scan = first_group + MIN_GROUPS_PER_WAVE
    # 与next_score_leaf统一显式Scalar回边；循环条件必须读取当前scan，
    # 不能依赖自然while的隐式变量版本化。空groups/越界first保持(groups,T)。
    for scan_i, group_i, query_end_i in pl.while_(init_values=(initial_scan, groups, tokens)):
        pl.cond(first_group >= 0 and first_group < groups and scan_i < groups and group_i == groups)
        descriptor = descriptor_base + scan_i * GROUP_DESCRIPTOR_WORDS
        first_query = pl.cast(pl.read(offsets, [descriptor + 1]), pl.INDEX)
        live = pl.cast(pl.read(offsets, [descriptor + 2]), pl.INDEX)
        if live > 0 and first_query % PLAN_QUERY_ROWS == 0:
            next_group, next_end = pl.yield_(scan_i, first_query)
        else:
            next_group, next_end = pl.yield_(group_i, query_end_i)
        scan, final_group, final_end = pl.yield_(scan_i + 1, next_group, next_end)
    return final_group, final_end
