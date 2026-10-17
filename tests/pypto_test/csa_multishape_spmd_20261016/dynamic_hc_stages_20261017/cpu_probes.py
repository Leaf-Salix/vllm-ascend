# SPDX-License-Identifier: Apache-2.0
"""动态 HC 数学阶段的编译壳；不构成完整 CSA，不用于测性能。

每个壳仅为其 helper 提供一个外层 SPMD，核函数内部不创建任务。
所有动态维度保留为签名符号，不按 B16 生成源码。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.hc_stages import (
    MIX_ROWS_DYN,
    PAD_ROWS_DYN,
    PARTIAL_ROWS_DYN,
    T_DYN,
    hc_comb_range,
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
    hc_widen_rms_range,
)


@pl.jit(auto_scope=False)
def widen_probe(
    x: pl.Tensor[[T_DYN, 16384], pl.BF16],
    x_fp32: pl.Out[pl.Tensor[[T_DYN, 16384], pl.FP32]],
    inv: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32]],
    tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
):
    for worker in pl.spmd(48):
        x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, tail, worker, 48)
    return x_fp32, inv


@pl.jit(auto_scope=False)
def project_probe(
    x: pl.Tensor[[T_DYN, 16384], pl.FP32],
    weight: pl.Tensor[[24, 16384], pl.FP32],
    partials: pl.Out[pl.Tensor[[PARTIAL_ROWS_DYN, 32], pl.FP32]],
):
    for worker in pl.spmd(24):
        partials = hc_project_range(x, weight, partials, worker, 24)
    return partials


@pl.jit(auto_scope=False)
def reduce_probe(
    partials: pl.Tensor[[PARTIAL_ROWS_DYN, 32], pl.FP32],
    raw: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 32], pl.FP32]],
):
    for worker in pl.spmd(48):
        raw = hc_reduce_range(partials, raw, worker, 48)
    return raw


@pl.jit(auto_scope=False)
def gates_probe(
    inv: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    raw: pl.Tensor[[PAD_ROWS_DYN, 32], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[24], pl.FP32],
    pre: pl.Out[pl.Tensor[[PAD_ROWS_DYN, 8], pl.FP32]],
    post: pl.Out[pl.Tensor[[T_DYN, 4], pl.FP32]],
    tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
):
    for worker in pl.spmd(48):
        pre, post = hc_gates_range(inv, raw, scale, base, pre, post, tail, worker, 48)
    return pre, post


@pl.jit(auto_scope=False)
def comb_probe(
    inv: pl.Tensor[[PAD_ROWS_DYN, 1], pl.FP32],
    raw: pl.Tensor[[PAD_ROWS_DYN, 32], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[1, 24], pl.FP32],
    comb: pl.Out[pl.Tensor[[T_DYN, 16], pl.FP32]],
    tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
):
    for worker in pl.spmd(48):
        comb = hc_comb_range(inv, raw, scale, base, comb, tail, worker, 48)
    return comb


@pl.jit(auto_scope=False)
def mix_norm_probe(
    x: pl.Tensor[[T_DYN, 16384], pl.FP32],
    pre: pl.Tensor[[PAD_ROWS_DYN, 8], pl.FP32],
    weight: pl.Tensor[[4096], pl.BF16],
    mixed: pl.InOut[pl.Tensor[[MIX_ROWS_DYN, 4096], pl.BF16]],
    normed: pl.Out[pl.Tensor[[T_DYN, 4096], pl.BF16]],
    tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
):
    for worker in pl.spmd(48):
        mixed, normed = hc_mix_norm_range(x, pre, weight, mixed, normed, tail, worker, 48)
    return mixed, normed
