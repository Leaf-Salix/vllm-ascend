# SPDX-License-Identifier: Apache-2.0
"""Indexer投影单因素根，state写回复用未经修改的生产helper。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.compressor_pool_stages import (
    STATE_ELEMENTS_DYN,
    STATE_PAGES_DYN,
    indexer_compressor_state_commit_stage,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.compressor_projection_stages import (
    INDEXER_PAD_ROWS_DYN,
    T_DYN,
    indexer_compressor_project_stage,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.indexer_native_projection_stages import (
    indexer_native_projection_stage,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.request_geometry import QUERY_BOUNDS_DYN, REQUESTS_DYN


@pl.jit
def indexproj_arm(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    wkv: pl.Tensor[[256, 4096], pl.BF16],
    wgate: pl.Tensor[[256, 4096], pl.BF16],
    ape: pl.Tensor[[4, 256], pl.FP32],
    positions: pl.Tensor[[T_DYN], pl.INT64],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    lengths: pl.Tensor[[REQUESTS_DYN], pl.INT32],
    slots: pl.Tensor[[T_DYN, 2], pl.INT32],
    state: pl.InOut[pl.Tensor[[STATE_PAGES_DYN, STATE_ELEMENTS_DYN], pl.FP32]],
    values: pl.InOut[pl.Tensor[[INDEXER_PAD_ROWS_DYN, 256], pl.FP32]],
    scores: pl.InOut[pl.Tensor[[INDEXER_PAD_ROWS_DYN, 256], pl.FP32]],
    MODE: pl.constexpr,
):
    for worker in pl.spmd(24, name_hint="indexer_projection_order", allow_early_resolve=True):
        if MODE == 0:
            values, scores = indexer_compressor_project_stage(x, wkv, wgate, values, scores, worker, 24)
        elif MODE == 1:
            values, scores = indexer_native_projection_stage(
                x, wkv, wgate, bounds, values, scores, worker, 24, 512, False
            )
        else:
            values, scores = indexer_native_projection_stage(
                x, wkv, wgate, bounds, values, scores, worker, 24, 128, MODE == 3
            )
    for writer in pl.spmd(48, name_hint="original_indexer_state_commit", allow_early_resolve=True):
        state = indexer_compressor_state_commit_stage(
            values, scores, ape, positions, bounds, lengths, slots, state, writer, 48
        )
    return state, values, scores
