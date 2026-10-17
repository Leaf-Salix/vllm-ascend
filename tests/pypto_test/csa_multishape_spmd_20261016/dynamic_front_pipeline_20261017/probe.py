# SPDX-License-Identifier: Apache-2.0
"""HC 六段→Q_A→QR 的源级动态 MIX 编译探针，不是完整 CSA。

入口保留实际 T、P16、P8、四份投影容量和信号容量的动态维度；没有历史
IR、静态 shape 描述符或某档位的 worker 表。24 AIC 与 48 AIV 各进入一次
begin_invocation（包括当前 T 下没有数学工作的空核）。数学直接调用正式 helper；
本文件只负责角色和阶段依赖。

第一版保守地等待本阶段所有活跃 producer。owner 来自数学 tile 的轮转
分配，空任务核不进入等待集合；每核仍发布自己已完成的阶段。GATES 与
COMB 是同一批 AIV 的两个独立输出阶段，COMB 不依赖 NORM。这里不宣称
这一分工具有良好性能，也没有设置 BYPASS 属性或运行设备精度/同步检查。

容量合同：P16=ceil(T/16)*16、P8=ceil(T/8)*8、PARTIAL=4*P16、T>0。
hc_base 与 hc_base_2d 是调用层对同一权重的两份 GM 描述符；没有核内整张
Tensor reshape。信号区固定 24 组、9 阶段、每槽 128 个 INT32，首次清零；
所有工作区和输出由调用层分配；相同信号区只允许单流/单图顺序复用，失败后
须重置；完成信号没有 ACK，不可用来逐 tile 覆盖未消费缓冲。本探针仅用动态
签名 CPU 编译，不绑定实值。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.hc_stages import (
    hc_comb_range,
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
    hc_widen_rms_range,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.phase_sync import (
    begin_invocation,
    publish_phase,
    wait_row_owners,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.qkv_stages import q_a_stage, q_r_stage

T = pl.dynamic("FRONT_T")
P16 = pl.dynamic("FRONT_PAD16")
P8 = pl.dynamic("FRONT_PAD8")
PARTIAL = pl.dynamic("FRONT_PARTIAL_ROWS")
WORDS = pl.dynamic("FRONT_SIGNAL_WORDS")

GROUPS = 24
VECTORS = 48
CORES = 72
PHASE_INPUT = 1
PHASE_PARTIAL = 2
PHASE_RAW = 3
PHASE_GATES = 4
PHASE_COMB = 5
PHASE_NORM = 6
PHASE_QA = 7
PHASE_QR = 8


@pl.jit.incore
def cube(
    x_fp32: pl.Tensor[[T, 16384], pl.FP32],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    normed: pl.Tensor[[T, 4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16],
    qa: pl.InOut[pl.Tensor[[P16, 1024], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    """AIC：等输入扩展→HC 投影→等归一化→Q_A，无新增 incore task。"""
    worker = pl.tile.get_block_idx()
    signal, generation = begin_invocation(signal, worker)
    tokens = pl.tensor.dim(x_fp32, 0)
    signal = wait_row_owners(
        signal, PHASE_INPUT, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    partials = hc_project_range(x_fp32, hc_fn, partials, worker, GROUPS)
    signal = publish_phase(signal, PHASE_PARTIAL, worker, CORES, generation)
    signal = wait_row_owners(
        signal, PHASE_NORM, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    # cluster 的 GM 合同显式保留原参数别名；q_a_stage 的动态 M 分支只写同一缓冲。
    qa_gm = qa
    qa = q_a_stage(normed, wq_a, qa, worker, GROUPS)
    signal = publish_phase(signal, PHASE_QA, worker, CORES, generation)
    return partials, qa_gm, signal


@pl.jit.incore
def vector(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    gamma_cq: pl.Tensor[[1024], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
    partials: pl.Tensor[[PARTIAL, 32], pl.FP32],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    qa: pl.Tensor[[P16, 1024], pl.FP32],
    qr: pl.InOut[pl.Tensor[[T, 1024], pl.INT8]],
    qr_scale: pl.InOut[pl.Tensor[[T, 1], pl.FP32]],
    qr_matmul: pl.InOut[pl.Tensor[[P8, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    """AIV：按真实动态行数复用六段数学，阶段完成后才发布数据可读信号。"""
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = GROUPS + worker
    signal, generation = begin_invocation(signal, core)
    tokens = pl.tensor.dim(x, 0)
    x_fp32, inv = hc_widen_rms_range(x, x_fp32, inv, widen_tail, worker, VECTORS)
    signal = publish_phase(signal, PHASE_INPUT, core, CORES, generation)

    # 投影 task=(M16 行块*4+K4096 split)，一维 task % 24 决定真实 AIC owner。
    projection_tasks = ((tokens + 15) // 16) * 4
    signal = wait_row_owners(
        signal, PHASE_PARTIAL, CORES, generation, 0, projection_tasks, 1, GROUPS, 0, 1,
    )
    raw = hc_reduce_range(partials, raw, worker, VECTORS)
    signal = publish_phase(signal, PHASE_RAW, core, CORES, generation)
    signal = wait_row_owners(
        signal, PHASE_RAW, CORES, generation, 0, tokens, 16, VECTORS, GROUPS, 1,
    )
    signal = wait_row_owners(
        signal, PHASE_INPUT, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    pre, post = hc_gates_range(
        inv, raw, hc_scale, hc_base, pre, post, post_tail, worker, VECTORS,
    )
    signal = publish_phase(signal, PHASE_GATES, core, CORES, generation)
    comb = hc_comb_range(inv, raw, hc_scale, hc_base_2d, comb, comb_tail, worker, VECTORS)
    signal = publish_phase(signal, PHASE_COMB, core, CORES, generation)
    signal = wait_row_owners(
        signal, PHASE_GATES, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    mixed, normed = hc_mix_norm_range(x_fp32, pre, norm_w, mixed, normed, norm_tail, worker, VECTORS)
    signal = publish_phase(signal, PHASE_NORM, core, CORES, generation)

    # Q_A 每个 M 组有八个 N128 列块；不能只等每行的一个 owner。
    if tokens < 128:
        qa_tasks = 8 * pl.min(3, pl.max(1, tokens // 32))
        signal = wait_row_owners(signal, PHASE_QA, CORES, generation, 0, qa_tasks, 1, GROUPS, 0, 1)
    else:
        qa_tasks = 8 * pl.min(3, pl.max(1, tokens // 64))
        signal = wait_row_owners(signal, PHASE_QA, CORES, generation, 0, qa_tasks, 1, GROUPS, 0, 1)
    qr, qr_scale, qr_matmul, qr_scale_pad = q_r_stage(
        qa, gamma_cq, qr, qr_scale, qr_matmul, qr_scale_pad, worker, VECTORS,
    )
    signal = publish_phase(signal, PHASE_QR, core, CORES, generation)
    return x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale, qr_matmul, qr_scale_pad, signal


@pl.jit.inline
def group(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16],
    gamma_cq: pl.Tensor[[1024], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    qa: pl.InOut[pl.Tensor[[P16, 1024], pl.FP32]],
    qr: pl.InOut[pl.Tensor[[T, 1024], pl.INT8]],
    qr_scale: pl.InOut[pl.Tensor[[T, 1], pl.FP32]],
    qr_matmul: pl.InOut[pl.Tensor[[P8, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.cluster(name_hint="dynamic_hc_qa_qr_front"):
        partials, qa, signal = cube(x_fp32, hc_fn, partials, normed, wq_a, qa, signal)
        x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale, qr_matmul, qr_scale_pad, signal = vector(
            x, hc_scale, hc_base, hc_base_2d, norm_w, gamma_cq, x_fp32, inv, widen_tail, partials,
            raw, pre, post, post_tail, comb, comb_tail, mixed, normed, norm_tail, qa, qr, qr_scale,
            qr_matmul, qr_scale_pad, signal,
        )
    return partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale, qr_matmul, qr_scale_pad, signal


@pl.jit
def entry(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16],
    gamma_cq: pl.Tensor[[1024], pl.BF16],
    x_fp32: pl.InOut[pl.Tensor[[T, 16384], pl.FP32]],
    inv: pl.InOut[pl.Tensor[[P16, 1], pl.FP32]],
    widen_tail: pl.InOut[pl.Tensor[[8, 16384], pl.FP32]],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    raw: pl.InOut[pl.Tensor[[P16, 32], pl.FP32]],
    pre: pl.InOut[pl.Tensor[[P16, 8], pl.FP32]],
    post: pl.InOut[pl.Tensor[[T, 4], pl.FP32]],
    post_tail: pl.InOut[pl.Tensor[[8, 8], pl.FP32]],
    comb: pl.InOut[pl.Tensor[[T, 16], pl.FP32]],
    comb_tail: pl.InOut[pl.Tensor[[8, 32], pl.FP32]],
    mixed: pl.InOut[pl.Tensor[[P8, 4096], pl.BF16]],
    normed: pl.InOut[pl.Tensor[[T, 4096], pl.BF16]],
    norm_tail: pl.InOut[pl.Tensor[[8, 4096], pl.BF16]],
    qa: pl.InOut[pl.Tensor[[P16, 1024], pl.FP32]],
    qr: pl.InOut[pl.Tensor[[T, 1024], pl.INT8]],
    qr_scale: pl.InOut[pl.Tensor[[T, 1], pl.FP32]],
    qr_matmul: pl.InOut[pl.Tensor[[P8, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.spmd(GROUPS, sync_start=True):
        (partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
         qr_matmul, qr_scale_pad, signal) = group(
            x, hc_fn, hc_scale, hc_base, hc_base_2d, norm_w, wq_a, gamma_cq, x_fp32, inv,
            widen_tail, partials, raw, pre, post, post_tail, comb, comb_tail, mixed, normed,
            norm_tail, qa, qr, qr_scale, qr_matmul, qr_scale_pad, signal,
        )
    return partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale, qr_matmul, qr_scale_pad, signal
