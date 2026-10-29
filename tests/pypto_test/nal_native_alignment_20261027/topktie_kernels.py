# SPDX-License-Identifier: Apache-2.0
"""隔离1152半叶与Native 2048分段的同分归并；不改变score数值。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.indexer_topk_stages import indexer_topk_native_short_pairs
from nal_alignment.deepseek_v4_flash_csa_tnd.indexer_topk_stages import (
    indexer_topk_half_leaf,
    merge2_top512_pairs,
)


@pl.jit
def topktie_arm(
    scores: pl.Tensor[[2, 8192], pl.FP32],
    pairs: pl.InOut[pl.Tensor[[2, 1024], pl.FP32]],
    indices: pl.InOut[pl.Tensor[[1, 512], pl.INT32]],
    values: pl.InOut[pl.Tensor[[1, 512], pl.FP32]],
    VISIBLE: pl.constexpr,
    MODE: pl.constexpr,
):
    for _worker in pl.spmd(1, name_hint="topk_tie_tree", allow_early_resolve=True):
        if MODE == 0:
            # 与实际SCORE_TILE768、三个tile的1152左右半叶一致。
            indexer_topk_half_leaf(scores, pairs, 0, 0, 1152, 0)
            indexer_topk_half_leaf(scores, pairs, 1, 1152, VISIBLE - 1152, 1)
            merge2_top512_pairs(pairs, 0, 1, 0)
        elif MODE == 1:
            # 当前控制VISIBLE在2048..2560；helper先2048再incoming尾段。
            indexer_topk_half_leaf(scores, pairs, 0, 0, VISIBLE, 0)
        if MODE == 2:
            root = indexer_topk_native_short_pairs(scores, 0, VISIBLE)
        else:
            root = pl.load(pairs, [0, 0], [1, 1024])
        output_indices = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        output_scores = pl.tile.gather_mask(root, mask_pattern=pl.tile.MaskPattern.P0101, output_dtype=pl.FP32)
        pl.store(output_indices, [0, 0], indices)
        pl.store(output_scores, [0, 0], values)
    return pairs, indices, values
