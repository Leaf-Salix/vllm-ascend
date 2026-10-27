# SPDX-License-Identifier: Apache-2.0
"""只改布局的逐字节资格：直接BF16输入与无损加宽后输入保持一致。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_native_projection_stages import (
    hc_pack_input,
    hc_pack_input_from_bf16,
)
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.hc_stages import PAD_ROWS_DYN, T_DYN, hc_widen_rms_range


@pl.jit
def pack_pair(
    x: pl.Tensor[[T_DYN, 16384], pl.BF16],
    wide: pl.InOut[pl.Tensor[[T_DYN, 16384], pl.FP32]],
    old_packed: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 32768], pl.BF16]],
    direct_packed: pl.InOut[pl.Tensor[[PAD_ROWS_DYN, 32768], pl.BF16]],
):
    inv = pl.create_tensor([pl.tensor.dim(old_packed, 0), 1], dtype=pl.FP32)
    tail = pl.create_tensor([8, 16384], dtype=pl.FP32)
    for worker in pl.spmd(48, name_hint="pack_reference_widen", allow_early_resolve=True):
        wide, inv = hc_widen_rms_range(x, wide, inv, tail, worker, 48)
    for worker in pl.spmd(48, name_hint="pack_pair", allow_early_resolve=True):
        old_packed = hc_pack_input(wide, old_packed, worker)
        direct_packed = hc_pack_input_from_bf16(x, direct_packed, worker)
    return wide, old_packed, direct_packed
