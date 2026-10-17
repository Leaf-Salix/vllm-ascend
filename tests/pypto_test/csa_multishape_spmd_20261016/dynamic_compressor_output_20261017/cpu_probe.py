# SPDX-License-Identifier: Apache-2.0
"""主compressor后处理独立AIV动态编译壳：只编译，不替代单MIX内的生产者依赖。

所有容量符号保留；不绑定B/S/T实值，不创建实际权重。normed、debug和cache
都是InOut，非关闭token、inactive/padding及未选cache行保持原值。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.compressor_output_stages import (
    CACHE_ROWS_DYN,
    COMPACT_SLOT_ROWS_DYN,
    NORMED_ROWS_DYN,
    POOLED_ROWS_DYN,
    T_DYN,
    compressor_norm_rope_cache_stage,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.metadata_stages import COMPACT_ROWS_DYN
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN


@pl.jit(auto_scope=False)
def compressor_output_probe(
    pooled: pl.Tensor[[POOLED_ROWS_DYN, 512], pl.FP32],
    gamma: pl.Tensor[[512], pl.BF16],
    cos: pl.Tensor[[COMPACT_ROWS_DYN, 64], pl.FP32],
    sin: pl.Tensor[[COMPACT_ROWS_DYN, 64], pl.FP32],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    offsets: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    slots: pl.Tensor[[COMPACT_SLOT_ROWS_DYN, 2], pl.INT32],
    normed: pl.InOut[pl.Tensor[[NORMED_ROWS_DYN, 512], pl.FP32]],
    debug: pl.InOut[pl.Tensor[[T_DYN, 512], pl.FP32]],
    cache: pl.InOut[pl.Tensor[[CACHE_ROWS_DYN, 512], pl.BF16]],
):
    for worker in pl.spmd(48):
        normed, debug, cache = compressor_norm_rope_cache_stage(
            pooled, gamma, cos, sin, bounds, lengths, positions, offsets, slots, normed, debug, cache, worker, 48
        )
    return normed, debug, cache
