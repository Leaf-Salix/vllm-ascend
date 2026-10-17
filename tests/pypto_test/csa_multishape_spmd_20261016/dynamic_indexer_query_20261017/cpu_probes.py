# SPDX-License-Identifier: Apache-2.0
"""六个Indexer query helper的动态签名CPU编译壳，不测试完整融合或设备。

原函数由外层SPMD给定worker；此处每个壳只建立一个独立编译scope，不代替
完整CSA内的跨阶段notify/wait。tokens只从真实上游Tensor的T轴取，不从padding
容量推导。所有函数只按符号签名编译，不绑定某个batch或执行实际权重。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.indexer_query_stages import (
    HADAMARD_ROWS_DYN,
    PROJ_ROWS_DYN,
    QUANT_ROWS_DYN,
    ROPE_HEAD_ROWS_DYN,
    ROPE_ROWS_DYN,
    SCALE_ROWS_DYN,
    T_DYN,
    WEIGHTS_PARTIAL_ROWS_DYN,
    WEIGHTS_ROWS_DYN,
    indexer_query_dequant_rope_stage,
    indexer_query_hadamard_stage,
    indexer_query_project_stage,
    indexer_query_quant_stage,
    indexer_weights_project_stage,
    indexer_weights_reduce_stage,
)


@pl.jit(auto_scope=False)
def project_probe(
    qr: pl.Tensor[[T_DYN, 1024], pl.INT8],
    w: pl.Tensor[[1024, 8192], pl.INT8],
    acc: pl.Out[pl.Tensor[[PROJ_ROWS_DYN, 8192], pl.INT32]],
):
    for worker in pl.spmd(24):
        acc = indexer_query_project_stage(qr, w, acc, worker, 24)
    return acc


@pl.jit(auto_scope=False)
def dequant_probe(
    scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    weight_scale: pl.Tensor[[8192], pl.FP32],
    cos: pl.Tensor[[T_DYN, 64], pl.FP32],
    signed_sin: pl.Tensor[[T_DYN, 64], pl.FP32],
    acc: pl.Tensor[[PROJ_ROWS_DYN, 8192], pl.INT32],
    bf16: pl.Out[pl.Tensor[[ROPE_ROWS_DYN, 8192], pl.BF16]],
):
    for worker in pl.spmd(48):
        bf16 = indexer_query_dequant_rope_stage(scale, weight_scale, cos, signed_sin, acc, bf16, worker, 48)
    return bf16


@pl.jit(auto_scope=False)
def hadamard_probe(
    scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    bf16: pl.Tensor[[ROPE_HEAD_ROWS_DYN, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    acc: pl.Out[pl.Tensor[[HADAMARD_ROWS_DYN, 128], pl.FP32]],
):
    tokens = pl.tensor.dim(scale, 0)
    for worker in pl.spmd(24):
        acc = indexer_query_hadamard_stage(bf16, hadamard, acc, tokens, worker, 24)
    return acc


@pl.jit(auto_scope=False)
def quant_probe(
    scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    acc: pl.Tensor[[HADAMARD_ROWS_DYN, 128], pl.FP32],
    result: pl.Out[pl.Tensor[[QUANT_ROWS_DYN, 128], pl.INT8]],
    dq_scale: pl.Out[pl.Tensor[[SCALE_ROWS_DYN, 1], pl.FP32]],
):
    tokens = pl.tensor.dim(scale, 0)
    for worker in pl.spmd(48):
        result, dq_scale = indexer_query_quant_stage(acc, result, dq_scale, tokens, worker, 48)
    return result, dq_scale


@pl.jit(auto_scope=False)
def weights_project_probe(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    w: pl.Tensor[[4096, 64], pl.BF16],
    partial: pl.Out[pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, 64], pl.FP32]],
):
    for worker in pl.spmd(24):
        partial = indexer_weights_project_stage(x, w, partial, worker, 24)
    return partial


@pl.jit(auto_scope=False)
def weights_reduce_probe(
    x: pl.Tensor[[T_DYN, 4096], pl.BF16],
    partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS_DYN, 64], pl.FP32],
    weights: pl.Out[pl.Tensor[[WEIGHTS_ROWS_DYN, 64], pl.FP32]],
):
    tokens = pl.tensor.dim(x, 0)
    for worker in pl.spmd(48):
        weights = indexer_weights_reduce_stage(partial, weights, tokens, worker, 48)
    return weights
