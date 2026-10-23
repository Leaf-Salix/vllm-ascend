# SPDX-License-Identifier: Apache-2.0
"""Single-SPMD HC/Norm/WKV/Indexer-weight prefix with a 32-tensor MIX ABI."""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_indexer_stages import indexer_projection_matmul
from ..deepseek_v4_flash_csa_tnd_precision_spmd.hc_prefix_stages import P16, PARTIAL_ROWS, hc_linear, hc_square
from ..deepseek_v4_flash_csa_tnd_precision_spmd.norm_wkv_stages import ACC_ROWS, T, wkv_matmul
from ..deepseek_v4_flash_csa_tnd_precision_spmd.rendezvous import handoff


@pl.jit.incore
def packed_prefix_indexer_cube(
    wide: pl.Tensor[[T, 16384], pl.FP32],
    square: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 16], pl.FP32]],
    packed_weight: pl.Tensor[[32768, 32], pl.BF16],
    packed_flat: pl.Tensor[[P16, 32768], pl.BF16],
    linear: pl.InOut[pl.Tensor[[PARTIAL_ROWS, 32], pl.FP32]],
    normalized: pl.Tensor[[T, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16],
    order: pl.Tensor[[32, 16], pl.INT32],
    accum: pl.InOut[pl.Tensor[[ACC_ROWS, 512], pl.FP32]],
    ffts: pl.Tensor[[256], pl.INT64],
    idx_weight: pl.Tensor[[64, 4096], pl.BF16],
    idx_order: pl.Tensor[[16, 4, 16], pl.INT32],
    idx_accum: pl.InOut[pl.Tensor[[P16, 64], pl.FP32]],
):
    pl.system.set_ffts(ffts)
    worker = pl.tile.get_block_idx()
    handoff()
    square = hc_square(wide, square, worker)
    handoff()
    linear = hc_linear(wide, packed_flat, packed_weight, linear, worker)
    handoff()
    handoff()
    handoff()
    handoff()
    handoff()
    accum = wkv_matmul(normalized, weight, order, accum, worker)
    idx_accum = indexer_projection_matmul(normalized, idx_weight, idx_order, idx_accum, worker)
    handoff()
    handoff()
    return (square, linear, accum, idx_accum)
