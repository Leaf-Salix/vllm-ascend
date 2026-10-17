# SPDX-License-Identifier: Apache-2.0
"""动态Indexer发布CPU编译壳：声明真实Q/B/T/arena动态轴，不指定sample形状。

pair的1/2根与balanced模式显式specialize，单leaf以plan roots=0独立入口。
这里的SPMD只提供显式worker，不能证明生产者同步或整层接入已完成。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_group_stages import TOKENS_DYN
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_topk_stages import PAIR_ROWS_DYN, SCORE_ROWS_DYN
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_publish_stages import (
    indexer_pair_publish_stage,
    indexer_single_leaf_publish_stage,
)


def pair_probe(roots, balanced):
    @pl.jit(auto_scope=False)
    def entry(
        bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
        lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
        positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
        plan: pl.Tensor[[8], pl.INT32],
        arena: pl.InOut[pl.Tensor[[PAIR_ROWS_DYN, 1024], pl.FP32]],
        scores: pl.InOut[pl.Tensor[[TOKENS_DYN, 512], pl.FP32]],
        indices: pl.InOut[pl.Tensor[[TOKENS_DYN, 512], pl.INT32]],
    ):
        with pl.spmd(48, name_hint="dynamic_indexer_pair_publish"):
            worker = pl.tile.get_block_idx()
            scores, indices = indexer_pair_publish_stage(
                bounds, lengths, positions, plan, arena, scores, indices, worker, 48, roots, balanced
            )
        return scores, indices

    return entry


pair_one_balanced = pair_probe(1, True)
pair_two_balanced = pair_probe(2, True)
pair_two_sequential = pair_probe(2, False)


@pl.jit(auto_scope=False)
def single_leaf(
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    positions: pl.Tensor[[TOKENS_DYN], pl.INT64],
    plan: pl.Tensor[[8], pl.INT32],
    arena: pl.Tensor[[SCORE_ROWS_DYN, 8192], pl.FP32],
    scores: pl.InOut[pl.Tensor[[TOKENS_DYN, 512], pl.FP32]],
    indices: pl.InOut[pl.Tensor[[TOKENS_DYN, 512], pl.INT32]],
):
    with pl.spmd(48, name_hint="dynamic_indexer_single_leaf_publish"):
        worker = pl.tile.get_block_idx()
        scores, indices = indexer_single_leaf_publish_stage(
            bounds, lengths, positions, plan, arena, scores, indices, worker, 48
        )
    return scores, indices
