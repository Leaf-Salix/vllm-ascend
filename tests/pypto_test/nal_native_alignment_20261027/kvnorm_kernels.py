# SPDX-License-Identifier: Apache-2.0
"""隔离同一KV accumulator的Native BF16边界，不改生产同步编排。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_native_norm_stages import kv_native_norm_rope_stage
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_stages import PAD_ROWS_DYN, T_DYN, kv_norm_rope_stage


@pl.jit
def kvnorm_arm(
    accumulator: pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32],
    gamma: pl.Tensor[[512], pl.BF16],
    cos: pl.Tensor[[T_DYN, 64], pl.FP32],
    signed_sin: pl.Tensor[[T_DYN, 64], pl.FP32],
    swap: pl.Tensor[[T_DYN, 64], pl.INT32],
    output: pl.InOut[pl.Tensor[[T_DYN, 512], pl.BF16]],
    NATIVE_BOUNDARIES: pl.constexpr,
):
    for worker in pl.spmd(48, name_hint="kv_norm_boundary", allow_early_resolve=True):
        if NATIVE_BOUNDARIES:
            output = kv_native_norm_rope_stage(accumulator, gamma, cos, signed_sin, swap, output, worker, 48)
        else:
            output = kv_norm_rope_stage(accumulator, gamma, cos, signed_sin, swap, output, worker, 48)
    return output
