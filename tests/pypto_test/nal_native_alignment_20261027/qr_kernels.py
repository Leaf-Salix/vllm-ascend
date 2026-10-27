# SPDX-License-Identifier: Apache-2.0
"""隔离生产QR helper，输入可选择保留或恢复QA的BF16边界。"""

import pypto.language as pl
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.q_native_stages import q_r_native_stage
from nal_alignment.deepseek_v4_flash_csa_spmd_tnd.qkv_stages import (
    MATMUL_ROWS_DYN,
    QA_ROWS_DYN,
    SCALE_ROWS_DYN,
    T_DYN,
    q_r_stage,
)


@pl.jit
def qr_arm(
    qa: pl.Tensor[[QA_ROWS_DYN, 1024], pl.FP32],
    gamma: pl.Tensor[[1024], pl.BF16],
    qr: pl.InOut[pl.Tensor[[T_DYN, 1024], pl.INT8]],
    scale: pl.InOut[pl.Tensor[[T_DYN, 1], pl.FP32]],
    qr_pad: pl.InOut[pl.Tensor[[MATMUL_ROWS_DYN, 1024], pl.INT8]],
    scale_pad: pl.InOut[pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32]],
    NATIVE: pl.constexpr = False,
):
    for worker in pl.spmd(48, name_hint="qr_qa_boundary", allow_early_resolve=True):
        if NATIVE:
            qr, scale, qr_pad, scale_pad = q_r_native_stage(qa, gamma, qr, scale, qr_pad, scale_pad, worker, 48)
        else:
            qr, scale, qr_pad, scale_pad = q_r_stage(qa, gamma, qr, scale, qr_pad, scale_pad, worker, 48)
    return qr, scale, qr_pad, scale_pad
