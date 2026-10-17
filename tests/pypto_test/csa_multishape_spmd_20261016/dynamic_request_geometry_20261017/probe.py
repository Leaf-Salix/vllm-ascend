# SPDX-License-Identifier: Apache-2.0
"""同源动态 helper 编译探针：仅通过文件载入生产模块，不导入 vLLM。"""

import importlib.util
import sys
from pathlib import Path

import pypto.language as pl

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "vllm_ascend/ops").is_dir())
SOURCE = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/request_geometry.py"
SPEC = importlib.util.spec_from_file_location("csa_geometry_under_test", SOURCE)
GEOMETRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEOMETRY
SPEC.loader.exec_module(GEOMETRY)

request_bounds = GEOMETRY.request_bounds
query_request = GEOMETRY.query_request
compression_geometry = GEOMETRY.compression_geometry
QUERY_BOUNDS = GEOMETRY.QUERY_BOUNDS_DYN
REQUESTS = GEOMETRY.REQUESTS_DYN
TOKENS = pl.dynamic("CSA_GEOMETRY_PROBE_TOKENS")


@pl.jit(auto_scope=False)
def entry(
    bounds: pl.Tensor[[QUERY_BOUNDS], pl.INT32],
    lengths: pl.Tensor[[REQUESTS], pl.INT32],
    request_result: pl.Out[pl.Tensor[[REQUESTS, 6], pl.INT32]],
    query_result: pl.Out[pl.Tensor[[TOKENS, 4], pl.INT32]],
):
    with pl.spmd(1, name_hint="dynamic_request_geometry"):
        # 显式核编号令整个 scope 作为同一个核体解析，三个 inline helper 不单独派发。
        worker = pl.tile.get_block_idx()
        capacity = pl.tensor.dim(query_result, 0)
        for request in pl.range(worker, pl.tensor.dim(lengths, 0), 1):
            begin, end, active = request_bounds(bounds, lengths, request, capacity)
            count, offset, compression_active = compression_geometry(bounds, lengths, request, capacity)
            pl.tensor.write(request_result, [request, 0], pl.cast(begin, pl.INT32))
            pl.tensor.write(request_result, [request, 1], pl.cast(end, pl.INT32))
            pl.tensor.write(request_result, [request, 2], active)
            pl.tensor.write(request_result, [request, 3], pl.cast(count, pl.INT32))
            pl.tensor.write(request_result, [request, 4], pl.cast(offset, pl.INT32))
            pl.tensor.write(request_result, [request, 5], compression_active)
        for query in pl.range(worker, capacity, 1):
            owner, begin, end, active = query_request(bounds, lengths, query, capacity)
            pl.tensor.write(query_result, [query, 0], pl.cast(owner, pl.INT32))
            pl.tensor.write(query_result, [query, 1], pl.cast(begin, pl.INT32))
            pl.tensor.write(query_result, [query, 2], pl.cast(end, pl.INT32))
            pl.tensor.write(query_result, [query, 3], active)
    return request_result, query_result
