# SPDX-License-Identifier: Apache-2.0
"""只调用生产QKV函数的诊断根入口；输入使用真实模型HC_pre后的Native归一化值。"""
import pypto.language as pl

from ..deepseek_v4_flash_hca.qkv_proj_rope import qkv_proj_rope, T_DYN, D, H, HEAD_DIM, Q_LORA, ROPE_DIM
from ..deepseek_v4_flash_csa.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT


@pl.jit
def qkv_probe(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    q_ready = pl.array.create(4, pl.TASK_ID)
    for group in pl.unroll(4):
        q_ready[group] = pl.system.task_invalid()
    dep = pl.system.task_dummy(deps=[])
    q, ready = qkv_proj_rope(x, wq_a, wq_b, wq_b_scale, wkv, rope_cos, rope_sin,
                            gamma_cq, gamma_ckv, q, kv, qr, qr_scale, q_ready, dep)
    return q, kv, qr, qr_scale
