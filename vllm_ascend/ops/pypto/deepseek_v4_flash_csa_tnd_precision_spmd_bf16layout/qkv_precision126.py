# SPDX-License-Identifier: Apache-2.0
"""Native-order QA routes for measured T126/T168; retain other frozen routes."""

import pypto.language as pl

from ..deepseek_v4_flash_csa_tnd_precision.qkv_proj_rope import (
    BF16_WEIGHT_LAYOUT,
    HEAD_DIM,
    PREFILL_DENSE_TILE,
    Q_LORA,
    QPROJ_MM_T_DYN,
    QPROJ_T_PAD,
    QR_DENSE_M_TILE,
    QR_K_TILE,
    QR_M_TILE,
    QR_N_TILE,
    QR_NATIVE_K_BLOCKS,
    QR_NATIVE_N_GROUP,
    QR_NATIVE_SHIFT_GROUPS,
    QR_OK,
    QR_SPLIT_K_TILE,
    QUANT_WEIGHT_LAYOUT,
    ROPE_DIM,
    STORE_ATOMIC,
    T_DYN,
    D,
    H,
    q_proj_q,
    q_proj_qa,
    q_proj_qr_normalize,
    rope_prepare,
)
from ..deepseek_v4_flash_csa_tnd_precision_spmd.qkv_proj_rope import kv_proj_rope

# Measured against the actual FULL Native QA boundary at T126 and T168.
QA_NATIVE_PACKET_ROWS = 126
QA_NATIVE_PACKET_ROWS_B28 = 168
QR_NATIVE_PACKET_BLOCKS = 2
QR_NATIVE_PACKETS = QR_NATIVE_K_BLOCKS // QR_NATIVE_PACKET_BLOCKS


@pl.jit.inline(auto_scope=False)
def q_proj_qa_126(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    qr_fp32: pl.Out[pl.Tensor[[QPROJ_MM_T_DYN, Q_LORA], pl.FP32]],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Measured Native NT schedule: K512 packets with ascending K256 halves."""
    qa_tokens = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [qa_tokens, D])
    qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
    qr_full_rows = (tile_rows // QR_DENSE_M_TILE) * QR_DENSE_M_TILE
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed"):
        for ts0 in pl.range(0, qr_t_matmul, QR_M_TILE):
            for nseed0 in pl.range(0, Q_LORA, QR_N_TILE):
                qr_seed = pl.full([QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0)
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = qr_seed

    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_SPLIT_K_TILE
        qr_native_group = q_a_col0 // QR_NATIVE_N_GROUP
        for dense_t0 in pl.range(0, qr_full_rows, QR_DENSE_M_TILE):
            dense_x0 = tile_base + dense_t0
            dense_acc = pl.create_tensor([QR_DENSE_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for dense_k in pl.pipeline(0, QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                dense_k_order = dense_k
                if qr_native_group < QR_NATIVE_SHIFT_GROUPS:
                    native_packet = dense_k // QR_NATIVE_PACKET_BLOCKS
                    if qr_native_group % 2 == 1:
                        native_packet = QR_NATIVE_PACKETS - 1 - native_packet
                    dense_k_order = (
                        qr_native_group // 2 + native_packet
                    ) % QR_NATIVE_PACKETS * QR_NATIVE_PACKET_BLOCKS + dense_k % QR_NATIVE_PACKET_BLOCKS
                # K 顺序的两条分支均在 [0, N)；max 是恒等式，供 NZ 证明跨分支的非负性。
                dense_d0 = qr_k_base + pl.max(dense_k_order, 0) * QR_K_TILE
                dense_x = x_view[dense_x0 : dense_x0 + QR_DENSE_M_TILE, dense_d0 : dense_d0 + QR_K_TILE]
                dense_w = wq_a[q_a_col0 : q_a_col0 + QR_N_TILE, dense_d0 : dense_d0 + QR_K_TILE]
                dense_acc = pl.matmul_acc(dense_acc, dense_x, dense_w, b_trans=True, init_cond=(dense_k == 0))
            qr_fp32 = pl.assemble(qr_fp32, dense_acc, [dense_t0, q_a_col0], atomic=STORE_ATOMIC)
        for t0 in pl.range(qr_full_rows, qr_t_matmul, QR_M_TILE):
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_SPLIT_K_TILE // QR_K_TILE, stage=2):
                qr_k_order = db
                if qr_native_group < QR_NATIVE_SHIFT_GROUPS:
                    native_packet = db // QR_NATIVE_PACKET_BLOCKS
                    if qr_native_group % 2 == 1:
                        native_packet = QR_NATIVE_PACKETS - 1 - native_packet
                    qr_k_order = (
                        qr_native_group // 2 + native_packet
                    ) % QR_NATIVE_PACKETS * QR_NATIVE_PACKET_BLOCKS + db % QR_NATIVE_PACKET_BLOCKS
                qr_d0 = qr_k_base + pl.max(qr_k_order, 0) * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, tile_rows - t0)
                x_t0 = tile_base + t0
                q_x_chunk_bf16 = pl.slice(
                    x_view,
                    [QR_M_TILE, QR_K_TILE],
                    [x_t0, qr_d0],
                    valid_shape=[qr_rows, QR_K_TILE],
                )
                w_chunk = wq_a[q_a_col0 : q_a_col0 + QR_N_TILE, qr_d0 : qr_d0 + QR_K_TILE]
                q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk, b_trans=True, init_cond=(db == 0))
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=STORE_ATOMIC)


@pl.jit.inline(auto_scope=False)
def q_proj_qr(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    qr_i8_matmul: pl.Out[pl.Tensor[[QPROJ_T_PAD, Q_LORA], pl.INT8]],
    qr_scale_pad_store: pl.Out[pl.Tensor[[QPROJ_T_PAD, 1], pl.FP32]],
):
    """Q LoRA, RMSNorm and quantization -- the half the indexer query chain needs.

    The two staging tensors are held by the caller so the qproj half can be issued
    separately; they are rewritten per dense tile, so a caller that runs more than one
    tile iteration must let the qproj half finish before the next one starts. Every
    current caller has t_dim <= PREFILL_DENSE_TILE, i.e. a single iteration.
    """
    t_dim = pl.tensor.dim(x, 0)
    for tile_base in pl.range(0, t_dim, PREFILL_DENSE_TILE):
        tile_rows = pl.min(PREFILL_DENSE_TILE, t_dim - tile_base)
        with pl.scope():
            qr_t_matmul = ((tile_rows + QR_M_TILE - 1) // QR_M_TILE) * QR_M_TILE
            qr_fp32 = pl.create_tensor([qr_t_matmul, Q_LORA], dtype=pl.FP32)
            if tile_rows == QA_NATIVE_PACKET_ROWS:  # noqa: SIM114 - scalar branches required by PyPTO codegen
                q_proj_qa_126(x, wq_a, qr_fp32, tile_base, tile_rows)
            elif tile_rows == QA_NATIVE_PACKET_ROWS_B28:
                q_proj_qa_126(x, wq_a, qr_fp32, tile_base, tile_rows)
            else:
                q_proj_qa(x, wq_a, qr_fp32, tile_base, tile_rows)

            q_proj_qr_normalize(qr_fp32, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store, tile_base, tile_rows)


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
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
):
    """Q LoRA, RMSNorm, quantization, and RoPE over bounded dense tiles."""
    qr_i8_matmul = pl.create_tensor([QPROJ_T_PAD, Q_LORA], dtype=pl.INT8)
    # The quant scale rides the qr_i8 -> qproj_matmul -> dequant chain.
    qr_scale_pad_store = pl.create_tensor([QPROJ_T_PAD, 1], dtype=pl.FP32, manual_dep=True)
    q_proj_qr(x, wq_a, gamma_cq, qr, qr_scale, qr_i8_matmul, qr_scale_pad_store)
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
    )


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[Q_LORA, D], pl.BF16, BF16_WEIGHT_LAYOUT],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8, QUANT_WEIGHT_LAYOUT],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
    native_kv_projected: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
):
    """Q 投影和 Native KV 后处理共用真实 TND 行数与 RoPE 表。"""
    t_dim = pl.tensor.dim(x, 0)
    q_rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    q_rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_sin, q_rope_sin_signed, q_rope_swap_idx)
    q_proj_rope(x, wq_a, wq_b, wq_b_scale, gamma_cq, rope_cos, q_rope_sin_signed, q_rope_swap_idx, q, qr, qr_scale)
    kv_proj_rope(x, gamma_ckv, rope_cos, q_rope_sin_signed, q_rope_swap_idx, kv, late_dep, native_kv_projected)
    projection_marker = pl.system.task_dummy(deps=[late_dep])
    return (q, projection_marker)
