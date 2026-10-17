# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""紧凑元数据/原始cache helper的动态签名编译探针；不是完整CSA。

三个入口分别检查 offsets、消费者UB取行、离散slot写回。它们不模拟整层
同步、不绑定样本形状；编译为多个测试入口不代表正式CSA要派发多个task。
生产helper均来自正式目录；输出与scratch由调用方提供，所有动态轴保留。
"""

import importlib.util
import sys
from pathlib import Path

import pypto.language as pl

REPO = Path(__file__).resolve().parents[4]
# 只在隔离命名空间绑定正式源码；必须先创建包，随后才可导入其相对依赖。
SOURCE = REPO / "vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd"
spec = importlib.util.spec_from_file_location(
    "_metadata_cache_dynamic", SOURCE / "__init__.py", submodule_search_locations=[str(SOURCE)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)

from _metadata_cache_dynamic.cache_stages import CACHE_ROWS_DYN, original_kv_writeback_stage
from _metadata_cache_dynamic.metadata_stages import (
    COMPACT_ROWS_DYN,
    compact_offsets_and_count_stage,
    compact_offsets_stage,
    load_request_compact_rope_rows,
)
from _metadata_cache_dynamic.request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN, request_bounds

T = pl.dynamic("METADATA_CACHE_PROBE_T")


@pl.jit
def count_entry(
    positions: pl.Tensor[[T], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Out[pl.Tensor[[REQUESTS_DYN], pl.INT32]],
    count: pl.Out[pl.Tensor[[1], pl.INT32]],
):
    # 同一前缀扫描的两个输出，验证动态T作为descriptor scalar而非样本常量。
    with pl.spmd(1):
        worker = pl.tile.get_block_idx()
        offsets, count = compact_offsets_and_count_stage(
            bounds, lengths, offsets, count, pl.tensor.dim(positions, 0), worker
        )
    return offsets, count


@pl.jit
def offsets_entry(
    positions: pl.Tensor[[T], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Out[pl.Tensor[[REQUESTS_DYN], pl.INT32]],
):
    with pl.spmd(1):
        worker = pl.tile.get_block_idx()
        offsets = compact_offsets_stage(bounds, lengths, offsets, pl.tensor.dim(positions, 0), worker)
    return offsets


@pl.jit
def rope_entry(
    cos: pl.Tensor[[COMPACT_ROWS_DYN, 64], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS_DYN, 64], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    out_cos: pl.Out[pl.Tensor[[T, 64], pl.FP32]],
    out_sin: pl.Out[pl.Tensor[[T, 64], pl.FP32]],
):
    with pl.spmd(8):
        worker = pl.tile.get_block_idx()
        # 探针按request分工，仅为检查helper在调用核内的返回Tile及动态尾行。
        for request in pl.range(worker, pl.tensor.dim(lengths, 0), 8):
            begin, end, active = request_bounds(bounds, lengths, request, pl.tensor.dim(positions, 0))
            if active != 0:
                for first in pl.range(begin, end, 16):
                    rows = pl.min(16, end - first)
                    c, s = load_request_compact_rope_rows(
                        cos, sin, positions, bounds, lengths, offsets, request, first, rows
                    )
                    out_cos = pl.store(pl.set_validshape(c, rows, 64), [first, 0], out_cos)
                    out_sin = pl.store(pl.set_validshape(s, rows, 64), [first, 0], out_sin)
    return out_cos, out_sin


@pl.jit
def cache_entry(
    kv: pl.Tensor[[T, 512], pl.BF16],
    slots: pl.Tensor[[T, 2], pl.INT32],
    cache: pl.InOut[pl.Tensor[[CACHE_ROWS_DYN, 512], pl.BF16]],
):
    with pl.spmd(8):
        worker = pl.tile.get_block_idx()
        cache = original_kv_writeback_stage(kv, slots, cache, worker, 8)
    return cache
