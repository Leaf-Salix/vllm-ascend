# SPDX-License-Identifier: Apache-2.0
"""仅用于硬件归因的完整Indexer Score MIX，直接复用正式数学，不实现替代算法。

输入来自同一进程完整动态CSA运行后的query INT8、head系数、Native cache/页表
及本轮device计划；Key别名仍在编排内从Native字节Tensor建立。T/B/页容量保留
动态，G/N/step/prefetch沿用正式策略。24个AIC和48个AIV执行全部有效leaf的
QK、FP16/ReLU、head权重归约、scale、sort/merge与每leaf根输出。

组内保留原FFTS READY/ACK及末尾排空。这里不包含前段投影、最终query根合并、
Attention与HCpost，也没有新的GM phase协议。配对原完整CSA的pair_arena后
才能收集PMU；硬件利用率和本段耗时只作归因，不替代整层L1验收。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_score_stages import (
    indexer_score_cube_stage,
    indexer_score_vector_lane_stage,
)

T = pl.dynamic("PROFILE_SCORE_T")
B = pl.dynamic("PROFILE_SCORE_B")
Q = pl.dynamic("PROFILE_SCORE_Q")
QH = pl.dynamic("PROFILE_SCORE_QH")
CR = pl.dynamic("PROFILE_SCORE_CR")
CC = pl.dynamic("PROFILE_SCORE_CC")
CP = pl.dynamic("PROFILE_SCORE_CP")
CB = pl.dynamic("PROFILE_SCORE_CB")
BC = pl.dynamic("PROFILE_SCORE_BC")
O = pl.dynamic("PROFILE_SCORE_O")
TR = pl.dynamic("PROFILE_SCORE_TR")
TC = pl.dynamic("PROFILE_SCORE_TC")
SR = pl.dynamic("PROFILE_SCORE_SR")
PR = pl.dynamic("PROFILE_SCORE_PR")
K0 = pl.dynamic("PROFILE_SCORE_K0")
K64 = pl.dynamic("PROFILE_SCORE_K64")


@pl.jit.incore
def score_cube(
    query: pl.Tensor[[QH, 128], pl.INT8],
    coefficients: pl.Tensor[[CR, CC], pl.FP16],
    key0: pl.Tensor[[K0, 128], pl.INT8],
    key64: pl.Tensor[[K64, 128], pl.INT8],
    cache: pl.Tensor[[CP, CB], pl.INT8],
    table: pl.Tensor[[B, BC], pl.INT32],
    positions: pl.Tensor[[T], pl.INT64],
    lengths: pl.Tensor[[B], pl.INT32],
    bounds: pl.Tensor[[Q], pl.INT32],
    offsets: pl.Tensor[[O], pl.INT32],
    plan: pl.Tensor[[8], pl.INT32],
    transfer: pl.InOut[pl.Tensor[[TR, TC], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    G: pl.constexpr, N: pl.constexpr, STEP: pl.constexpr,
    PREFETCH: pl.constexpr, TO_L0: pl.constexpr, BALANCED: pl.constexpr,
):
    pl.system.set_ffts(ffts)
    transfer = indexer_score_cube_stage(
        query, coefficients, key0, key64, cache, table, positions, lengths,
        bounds, offsets, plan, transfer, pl.tile.get_block_idx(),
        G, N, STEP, PREFETCH, TO_L0, BALANCED, 0, 1,
    )
    return transfer


@pl.jit.incore
def score_vector(
    cache: pl.Tensor[[CP, CB], pl.INT8],
    table: pl.Tensor[[B, BC], pl.INT32],
    positions: pl.Tensor[[T], pl.INT64],
    lengths: pl.Tensor[[B], pl.INT32],
    bounds: pl.Tensor[[Q], pl.INT32],
    offsets: pl.Tensor[[O], pl.INT32],
    plan: pl.Tensor[[8], pl.INT32],
    transfer: pl.Tensor[[TR, TC], pl.FP32],
    scores: pl.InOut[pl.Tensor[[SR, 8192], pl.FP32]],
    pairs: pl.InOut[pl.Tensor[[PR, 1024], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    G: pl.constexpr, N: pl.constexpr, STEP: pl.constexpr,
    PREFETCH: pl.constexpr, TO_L0: pl.constexpr, BALANCED: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    pl.system.set_ffts(ffts)
    scores, pairs = indexer_score_vector_lane_stage(
        cache, table, positions, lengths, bounds, offsets, plan, transfer, scores, pairs,
        pl.tile.get_subblock_idx(), pl.tile.get_block_idx(),
        G, N, STEP, PREFETCH, TO_L0, BALANCED, 0, 1,
    )
    return scores, pairs


@pl.jit.inline(auto_scope=False)
def score_group(
    query: pl.Tensor[[QH, 128], pl.INT8],
    coefficients: pl.Tensor[[CR, CC], pl.FP16],
    key0: pl.Tensor[[K0, 128], pl.INT8],
    key64: pl.Tensor[[K64, 128], pl.INT8],
    cache: pl.Tensor[[CP, CB], pl.INT8],
    table: pl.Tensor[[B, BC], pl.INT32],
    positions: pl.Tensor[[T], pl.INT64],
    lengths: pl.Tensor[[B], pl.INT32],
    bounds: pl.Tensor[[Q], pl.INT32],
    offsets: pl.Tensor[[O], pl.INT32],
    plan: pl.Tensor[[8], pl.INT32],
    transfer: pl.InOut[pl.Tensor[[TR, TC], pl.FP32]],
    scores: pl.InOut[pl.Tensor[[SR, 8192], pl.FP32]],
    pairs: pl.InOut[pl.Tensor[[PR, 1024], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    G: pl.constexpr, N: pl.constexpr, STEP: pl.constexpr,
    PREFETCH: pl.constexpr, TO_L0: pl.constexpr, BALANCED: pl.constexpr,
):
    # 两个incore角色由cluster合成一个MIX，不能先运行完整Cube再运行Vector。
    with pl.cluster(name_hint="csa_indexer_score_profile"):
        transfer = score_cube(
            query, coefficients, key0, key64, cache, table, positions, lengths,
            bounds, offsets, plan, transfer, ffts, G, N, STEP, PREFETCH, TO_L0, BALANCED,
        )
        scores, pairs = score_vector(
            cache, table, positions, lengths, bounds, offsets, plan, transfer,
            scores, pairs, ffts, G, N, STEP, PREFETCH, TO_L0, BALANCED,
        )
    return transfer, scores, pairs


@pl.jit
def score_entry(
    query: pl.Tensor[[QH, 128], pl.INT8],
    coefficients: pl.Tensor[[CR, CC], pl.FP16],
    cache: pl.Tensor[[CP, CB], pl.INT8],
    table: pl.Tensor[[B, BC], pl.INT32],
    positions: pl.Tensor[[T], pl.INT64],
    lengths: pl.Tensor[[B], pl.INT32],
    bounds: pl.Tensor[[Q], pl.INT32],
    offsets: pl.Tensor[[O], pl.INT32],
    plan: pl.Tensor[[8], pl.INT32],
    transfer: pl.InOut[pl.Tensor[[TR, TC], pl.FP32]],
    scores: pl.InOut[pl.Tensor[[SR, 8192], pl.FP32]],
    pairs: pl.InOut[pl.Tensor[[PR, 1024], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    G: pl.constexpr, N: pl.constexpr, STEP: pl.constexpr,
    PREFETCH: pl.constexpr, TO_L0: pl.constexpr, BALANCED: pl.constexpr,
):
    # descriptor只做零拷贝别名，避免把部分重叠的原cache与Key视图重复作为根参数。
    pages = pl.tensor.dim(cache, 0)
    page_bytes = pl.tensor.dim(cache, 1)
    flat = pl.reshape(cache, [pages * page_bytes])
    prefix = flat[0 : (pages * page_bytes // 128) * 128]
    shifted = flat[64 : 64 + ((pages * page_bytes - 64) // 128) * 128]
    key0 = pl.reshape(prefix, [pages * page_bytes // 128, 128])
    key64 = pl.reshape(shifted, [(pages * page_bytes - 64) // 128, 128])
    with pl.spmd(24, sync_start=True):
        transfer, scores, pairs = score_group(
            query, coefficients, key0, key64, cache, table, positions, lengths,
            bounds, offsets, plan, transfer, scores, pairs, ffts,
            G, N, STEP, PREFETCH, TO_L0, BALANCED,
        )
    return pairs
