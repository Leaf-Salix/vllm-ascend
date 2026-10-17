# SPDX-License-Identifier: Apache-2.0
"""四个纯TopK helper的独立AIV编译壳，不执行设备数据或模拟完整CSA。

control中的行号/计数由设备Scalar读取，使后端保留真实分支与根数循环；
不对某个batch、valid_count或leaf数展开。每个壳单worker避免测试壳自行
引入所有权问题；所有输出InOut，只写指定行。实际融合由调用者分工及同步。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_topk_stages import (
    PAIR_ROWS_DYN,
    SCORE_ROWS_DYN,
    indexer_topk_half_leaf,
    indexer_topk_segment_pairs,
    merge2_top512_pairs,
    merge_top512_roots,
)

OUTPUT_ROWS_DYN = pl.dynamic("CSA_TOPK_PROBE_OUTPUT_ROWS")


@pl.jit(auto_scope=False)
def merge2_probe(
    pair: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, 1024], pl.FP32]],
    control: pl.Tensor[[3], pl.INT32],
):
    for worker in pl.spmd(1):
        left = pl.cast(pl.read(control, [0]), pl.INDEX)
        right = pl.cast(pl.read(control, [1]), pl.INDEX)
        output = pl.cast(pl.read(control, [2]), pl.INDEX)
        merge2_top512_pairs(pair, left, right, output)
    return pair


@pl.jit(auto_scope=False)
def roots_probe(
    pair: pl.Tensor[[PAIR_ROWS_DYN, 1024], pl.FP32],
    control: pl.Tensor[[2], pl.INT32],
    output: pl.InOut[pl.Tensor[[OUTPUT_ROWS_DYN, 1024], pl.FP32]],
):
    for worker in pl.spmd(1):
        base = pl.cast(pl.read(control, [0]), pl.INDEX)
        count = pl.cast(pl.read(control, [1]), pl.INDEX)
        root = merge_top512_roots(pair, base, count)
        pl.store(root, [0, 0], output)
    return output


@pl.jit(auto_scope=False)
def half_leaf_probe(
    score: pl.Tensor[[SCORE_ROWS_DYN, 8192], pl.FP32],
    pair: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, 1024], pl.FP32]],
    control: pl.Tensor[[4], pl.INT32],
):
    for worker in pl.spmd(1):
        row = pl.cast(pl.read(control, [0]), pl.INDEX)
        begin = pl.cast(pl.read(control, [1]), pl.INDEX)
        count = pl.cast(pl.read(control, [2]), pl.INDEX)
        slot = pl.cast(pl.read(control, [3]), pl.INDEX)
        indexer_topk_half_leaf(score, pair, row, begin, count, slot)
    return pair


@pl.jit(auto_scope=False)
def segment_probe(
    score: pl.Tensor[[SCORE_ROWS_DYN, 8192], pl.FP32],
    control: pl.Tensor[[3], pl.INT32],
    output: pl.InOut[pl.Tensor[[OUTPUT_ROWS_DYN, 1024], pl.FP32]],
):
    for worker in pl.spmd(1):
        row = pl.cast(pl.read(control, [0]), pl.INDEX)
        begin = pl.cast(pl.read(control, [1]), pl.INDEX)
        count = pl.cast(pl.read(control, [2]), pl.INDEX)
        root = indexer_topk_segment_pairs(score, row, begin, count, 2048)
        pl.store(root, [0, 0], output)
    return output
