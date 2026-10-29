# SPDX-License-Identifier: Apache-2.0
"""T33同输入单因素：保持M128/N128，只改Native候选K256起点。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.output_projection_a_stages import (
    PACK_TOTAL_ROWS_DYN,
    _native_small_o_projection_a_tile_nz,
    o_projection_a_stage_nz,
)

T = pl.dynamic("WOA_TND_PROBE_T")


@pl.jit
def woa_tnd_arm(
    packed: pl.Tensor[[PACK_TOTAL_ROWS_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[8, 4096, 1024], pl.BF16, pl.NZ],
    projected: pl.InOut[pl.Tensor[[T, 8192], pl.FP32]],
    MODE: pl.constexpr,
):
    tokens = pl.tensor.dim(projected, 0)
    for worker in pl.spmd(24, name_hint="woa_tnd_same_input", allow_early_resolve=True):
        if MODE == 0:
            projected = o_projection_a_stage_nz(packed, weight, projected, tokens, worker, 24, 128, 128)
        else:
            for logical in pl.range(worker, 64, 24):
                group = pl.max(logical // 8, 0)
                column = logical % 8
                n0 = column * 128
                # 旧环境T33标定仅作假设；不把它当当前Native二进制资格。
                kind = (column // 3 + 8 - group) % 4
                if kind == 0:
                    projected = _native_small_o_projection_a_tile_nz(packed, weight, projected, tokens, group, n0, 0)
                elif kind == 1:
                    projected = _native_small_o_projection_a_tile_nz(packed, weight, projected, tokens, group, n0, 5)
                elif kind == 2:
                    projected = _native_small_o_projection_a_tile_nz(packed, weight, projected, tokens, group, n0, 10)
                else:
                    projected = _native_small_o_projection_a_tile_nz(packed, weight, projected, tokens, group, n0, 15)
    return projected
