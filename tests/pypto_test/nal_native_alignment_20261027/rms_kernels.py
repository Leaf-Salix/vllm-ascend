# SPDX-License-Identifier: Apache-2.0
"""单因素RMS诊断；直接调用生产helper，不声称是完整SPMD性能测试。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_input_rms_stages import (
    SQUARE_ROWS,
    hc_input_rms_range,
    hc_input_square_range,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_stages import (
    HC_DIM,
    PAD_ROWS_DYN,
    T_DYN,
    hc_widen_rms_range,
)


@pl.jit
def rms_pair(
    x: pl.Tensor[[T_DYN, HC_DIM], pl.BF16],
    wide: pl.InOut[pl.Tensor[[T_DYN, HC_DIM], pl.FP32]],
    baseline: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    candidate: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    squares: pl.InOut[pl.Tensor[[SQUARE_ROWS, 16], pl.FP32]],
):
    tail = pl.create_tensor([8, HC_DIM], dtype=pl.FP32)
    for worker in pl.spmd(48, name_hint="nal_hc_widen_rms", allow_early_resolve=True):
        wide, baseline = hc_widen_rms_range(x, wide, baseline, tail, worker, 48)
    for worker in pl.spmd(24, name_hint="native_order_hc_square", allow_early_resolve=True):
        squares = hc_input_square_range(wide, squares, worker)
    for worker in pl.spmd(48, name_hint="native_order_hc_inverse", allow_early_resolve=True):
        candidate = hc_input_rms_range(squares, candidate, wide, worker)
    return wide, baseline, candidate, squares
