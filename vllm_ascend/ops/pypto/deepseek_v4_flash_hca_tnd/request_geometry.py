# SPDX-License-Identifier: Apache-2.0
"""把 Native 的请求边界绑定到 token；每次设备执行重新推导。

行描述符的前四列依次是请求号、query 起点、query 终点、有效 KV 长度。
其余列保留为零，整行 64 字节，避免不同核对同一缓存行做短写。
失活请求、没有分配到请求的容量尾行令长度为零，其它地址列仍取安全值。
这里不产生任务完成标志；消费者依赖的是整个 metadata task 的 TaskId。
"""

import pypto.language as pl

T_DYN = pl.dynamic("HCA_GEOMETRY_TOKENS")
B_DYN = pl.dynamic("HCA_GEOMETRY_REQUESTS")
Q_DYN = pl.dynamic("HCA_GEOMETRY_BOUNDS")
GEOMETRY_COLUMNS = 16
MAX_WORKERS = 48


@pl.jit.inline(auto_scope=False)
def build_token_geometry(
    query_bounds: pl.Tensor[[Q_DYN], pl.INT32],
    seq_lens: pl.Tensor[[B_DYN], pl.INT32],
    geometry: pl.Tensor[[T_DYN, GEOMETRY_COLUMNS], pl.INT32],
):
    """二分有序 Q；重复边界对应空请求，取包含当前 token 的最后一个起点。"""
    tokens = pl.tensor.dim(geometry, 0)
    requests = pl.tensor.dim(seq_lens, 0)
    workers = pl.min(tokens, MAX_WORKERS)
    with pl.spmd(workers, name_hint="hca_tnd_geometry", allow_early_resolve=True) as ready:
        for token in pl.range(pl.tile.get_block_idx(), tokens, workers):
            lo = pl.cast(0, pl.INDEX)
            hi = requests
            # 最多 64 个请求，七次足够收敛；无设备同步轮询。
            for iteration in pl.range(7):
                if lo < hi:
                    middle = (lo + hi) // 2
                    endpoint = pl.read(query_bounds, [middle + 1])
                    if token < endpoint:
                        hi = middle
                    else:
                        lo = middle + 1
            request = pl.min(lo, requests - 1)
            begin = pl.read(query_bounds, [request])
            end = pl.read(query_bounds, [request + 1])
            length = pl.read(seq_lens, [request])
            if token < begin or token >= end or length <= 0:
                active_length = pl.yield_(pl.cast(0, pl.INT32))
            else:
                active_length = pl.yield_(length)
            row = pl.tile.full([1, GEOMETRY_COLUMNS], dtype=pl.INT32, value=0)
            pl.tile.write(row, [0, 0], pl.cast(request, pl.INT32))
            pl.tile.write(row, [0, 1], begin)
            pl.tile.write(row, [0, 2], end)
            pl.tile.write(row, [0, 3], active_length)
            pl.store(row, [token, 0], geometry)
    return ready
