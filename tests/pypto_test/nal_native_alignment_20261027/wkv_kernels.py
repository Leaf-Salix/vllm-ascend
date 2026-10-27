# SPDX-License-Identifier: Apache-2.0
"""保留原Nal生产者分派，隔离N32/K256与paired累加顺序两个因素。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_native_stages import kv_order_stage
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_precision_dispatch import kv_precision_project_stage
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.kv_stages import (
    BF16_WEIGHT_LAYOUT,
    PAD_ROWS_DYN,
    T_DYN,
    kv_project_stage,
)


@pl.jit
def wkv_arm(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    weight: pl.Tensor[[512, 4096], pl.BF16, BF16_WEIGHT_LAYOUT],
    projected: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 512], pl.FP32]],
    NARROW: pl.constexpr,
    ORDER: pl.constexpr,
    DISPATCH: pl.constexpr = False,
):
    for worker in pl.spmd(24, name_hint="wkv_same_owner", allow_early_resolve=True):
        if DISPATCH:
            projected = kv_precision_project_stage(x, weight, projected, worker, 24)
        elif NARROW:
            projected = kv_order_stage(x, weight, projected, worker, 24, ORDER)
        else:
            projected = kv_project_stage(x, weight, projected, worker, 24)
    return projected
