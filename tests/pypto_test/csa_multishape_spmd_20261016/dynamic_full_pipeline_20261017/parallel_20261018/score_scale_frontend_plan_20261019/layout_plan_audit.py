# SPDX-License-Identifier: Apache-2.0
"""功能验证阶段读回 META 计划，独立整数检查快路径/回退；不进入性能测量。"""


def validate(candidate):
    buffers = candidate.buffers
    bounds = buffers["idx_bounds"].cpu().tolist()
    lengths = buffers["idx_lengths"].cpu().tolist()
    table = buffers["indexer_table"].cpu().tolist()
    tokens = buffers["x"].shape[0]
    pages_capacity = buffers["idx_cache"].shape[0]
    requests = len(lengths)
    prefix_end = (requests + 16) // 16 * 16 + 4 * tokens
    base = (prefix_end + 15) // 16 * 16 + 48 + 4 * tokens
    actual = buffers["group_offsets"][base:base + 2 * requests].cpu().tolist()
    rows = []
    for request, length in enumerate(lengths):
        begin, end = bounds[request:request + 2]
        expected = [0, 0]
        pages = (length // 4 + 31) // 32
        active = length > 0 and 0 <= begin < end <= tokens and end - begin <= length
        if active and 0 < pages <= len(table[request]) and 0 < pages_capacity <= 2**24:
            indices = table[request][:pages]
            first, last = indices[0], indices[-1]
            direction = -1 if last < first else 1
            if all(0 <= index < pages_capacity and index == first + direction * i
                   for i, index in enumerate(indices)):
                expected = [first, direction]
        got = actual[2 * request:2 * request + 2]
        rows.append(dict(request=request, active=active, pages=pages,
                         expected=expected, actual=got, passed=got == expected))
    return dict(
        passed=all(row["passed"] for row in rows),
        requests=requests, first_item=base, words=2 * requests,
        fast_requests=sum(row["actual"][1] != 0 for row in rows),
        fallback_requests=sum(row["actual"][1] == 0 for row in rows),
        rows=rows, inside_timing_window=False,
    )
