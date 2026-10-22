# SPDX-License-Identifier: Apache-2.0
"""首版TND服务的显式支持范围，只使用已有Host描述符。

query长度上界是当前DSpark服务配置，不能用它反算请求归属。历史下界保留
当前完整SPMD的Cube Score能力边界；更短历史仍待独立数值和设备验收，
在服务层回退Native，不借较大的max_model_len伪装成已支持。
"""

MAX_QUERY_TOKENS = 6
MIN_SEQUENCE_LENGTH = 8192
MAX_SEQUENCE_LENGTH = 1048576


def native_query_bound(request):
    """复用Native调度器已有真实query上界，保留旧metadata的保守回退。

    Native tiling的max_seqlen_q可能包含大于6的失活padding区间。新增字段
    只是传递common metadata的既有标量；不规约CPU数组，也不读取device Q/L。
    类型及范围仍由supported_descriptor严格检查，不把未知值强制转换为int。
    """
    actual = getattr(request, "max_seqlen_q_actual", None)
    return request.max_seqlen_q if actual is None else actual


def supported_descriptor(*, tokens, requests, actual_tokens, max_query_tokens, max_seq_len, batch_capacity):
    """输入均来自Native现成Host字段；不读device Q/L值，不要求等长请求。"""
    values = (tokens, requests, actual_tokens, max_query_tokens, max_seq_len, batch_capacity)
    if any(type(value) is not int for value in values):
        return False
    return (
        1 <= requests <= batch_capacity
        and 1 <= actual_tokens <= tokens
        and 1 <= max_query_tokens <= MAX_QUERY_TOKENS
        and actual_tokens <= requests * max_query_tokens
        and MIN_SEQUENCE_LENGTH <= max_seq_len <= MAX_SEQUENCE_LENGTH
    )
