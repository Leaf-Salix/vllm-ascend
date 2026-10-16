# SPDX-License-Identifier: Apache-2.0
"""HCA QKV入口：NZ的Q_B分组发布，ND沿用公共投影。"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.nz_mode import BF16_WEIGHT_LAYOUT, QUANT_WEIGHT_LAYOUT, QUANT_WEIGHT_NZ
from ..deepseek_v4_flash_csa.qkv_proj_rope import (
    HEAD_DIM,
    Q_LORA,
    QPROJ_T_PAD,
    ROPE_DIM,
    T_DYN,
    D,
    H,
    kv_proj_rope,
    q_proj_qr,
    rope_prepare,
)
from ..deepseek_v4_flash_csa.qkv_proj_rope import (
    q_proj_q as q_proj_q_separate,
)
from .q_projection_streamed import q_proj_q_streamed


@pl.jit.inline(auto_scope=False)
def q_proj_q_nd_view(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    qr_i8_matmul: pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8],
    qr_scale_pad_store: pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32],
    qproj_dep: pl.Scalar[pl.TASK_ID],
    q_ready: pl.Array[4, pl.TASK_ID],
):
    t_dim = pl.tensor.dim(x, 0)
    q_dense = pl.reshape(q, [t_dim, H, HEAD_DIM])
    q_proj_q_separate(
        x,
        wq_b,
        wq_b_scale,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        q_dense,
        qr_i8_matmul,
        qr_scale_pad_store,
        qproj_dep,
    )
    return q


q_proj_q = q_proj_q_streamed if QUANT_WEIGHT_NZ else q_proj_q_nd_view


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    q_ready: pl.Array[4, pl.TASK_ID],
):
    """Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles."""
    qr_i8_matmul = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    # The quant scale rides the qr_i8 -> qproj_matmul -> dequant chain.
    qr_scale_pad_store = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
    qa_tid = q_proj_qr(x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store)
    q_seq_dep = pl.system.task_dummy(deps=[])
    q_proj_q(
        x,
        wq_b,
        wq_b_scale,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        q,
        qr_i8_matmul,
        qr_scale_pad_store,
        q_seq_dep,
        q_ready,
    )

    return qa_tid


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H * HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    q_ready: pl.Array[4, pl.TASK_ID],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    """Fused q + kv projection: both branches share one token axis and one rope table."""
    t_dim = pl.tensor.dim(x, 0)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, q_rope_sin_signed, q_rope_swap_idx)
    qa_tid = q_proj_rope(
        x,
        wq_a,
        wq_b,
        wq_b_scale,
        gamma_cq,
        rope_cos,
        q_rope_sin_signed,
        q_rope_swap_idx,
        q,
        qr,
        qr_scale,
        q_ready,
    )
    kv_proj_rope(
        x,
        wkv,
        gamma_ckv,
        rope_cos,
        q_rope_sin_signed,
        q_rope_swap_idx,
        kv,
        late_dep,
    )
    return q, qa_tid
