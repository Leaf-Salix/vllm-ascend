# SPDX-License-Identifier: Apache-2.0
"""HC→QKV与原始KV cache写回的源级动态单MIX编译探针。

一、范围与输入格式
    本文件只组合完整query前段和原始KV分支，仍不是完整CSA：没有压缩cache、
    Indexer、Attention、输出投影或HCpost。所有数学复用正式源码helper，无历史
    IR或固定档位entry。进程NZ2：QA BF16 NZ、QB INT8 NZ，而wkv始终是Native
    ND BF16 [4096,512]；gamma_ckv仍为BF16，不做Host权重转换。

二、动态工作区与Native别名
    T是实际token数，HC/QA的P16和KV的KV_ROWS独立，均按M16容量分配；P8供
    mixed和QR scale，QR_ROWS/QPROJ_ROWS分别按M128预留。cache_rows是Native
    [页数,32,1,512]原始cache的二维连续GM别名[页数*32,512]，Native slots
    [T,2]直接给device，包含page/offset，不复制或在Host读取metadata。
    q_flat与下游[T,64,512]同数据，hc_base的两视图由调用层绑定；没有核内GM
    整体reshape。所有动态轴独立描述容量，实际数学循环始终依据真实T。

三、同一MIX内的角色和依赖
    24 AIC依次HC投影→QA→KV投影→等QR→QB。QA发布后先做KV，允许AIV同时
    执行QR；不能把KV投影放在等待KV消费者之后，否则会形成等待环。
    48 AIV先HC→QR→RoPE准备，再等KV投影/RoPE→KVnorm→原始cache写回，最后
    等QB做query反量化/RMS/RoPE。初版保守等待全部活跃owner，不宣称性能最优。
    KVnorm与query共用已经准备好的signed_sin/swap；没有额外RoPE适配任务。
    原0..11phase保持含义，12=KV_PROJECT、13=KV_NORM、14=ORIGINAL_CACHE。

四、信号、写槽和生命周期
    G24固定，共72核包括空核各begin一次；阶段writer完成全部wave后发布一次。
    signals首次清零，预留15*72*128个INT32，同一固定协议单流/单图顺序复用，
    失败后重置。完成信号没有ACK，不能逐tile复用缓冲。Native有效写slot必须
    唯一；-1 dummy或无效page/offset不访问cache，prefix只读共享不能重复写槽。

五、验证边界
    只执行CPU完整后端及动态ABI/GM别名/通知顺序检查，不启动NPU，不代表设备
    精度或缓存可见性通过，也没有性能结论。显式cluster外层SPMD不能挂BYPASS
    属性，沿用query前段记录的限制，本轮不为缓存属性改正式数学或依赖仓。
"""

import pypto.language as pl

from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.cache_stages import original_kv_writeback_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.hc_stages import (
    hc_comb_range,
    hc_gates_range,
    hc_mix_norm_range,
    hc_project_range,
    hc_reduce_range,
    hc_widen_rms_range,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.kv_stages import kv_norm_rope_stage, kv_project_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.phase_sync import (
    begin_invocation,
    publish_phase,
    wait_row_owners,
)
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.qkv_stages import q_a_stage, q_r_stage
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.query_projection_stages import (
    q_b_stage,
    q_dequant_rope_stage,
    q_rope_prepare_stage,
)

T = pl.dynamic("QKV_FRONT_T")
P16 = pl.dynamic("QKV_FRONT_PAD16")
P8 = pl.dynamic("QKV_FRONT_PAD8")
PARTIAL = pl.dynamic("QKV_FRONT_PARTIAL_ROWS")
QR_ROWS = pl.dynamic("QKV_FRONT_QR_PAD128")
QPROJ_ROWS = pl.dynamic("QKV_FRONT_PROJ_PAD128")
KV_ROWS = pl.dynamic("QKV_FRONT_KV_PAD16")
CACHE_ROWS = pl.dynamic("QKV_FRONT_CACHE_ROWS")
WORDS = pl.dynamic("QKV_FRONT_SIGNAL_WORDS")

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
PHASE_ROPE = 9
PHASE_QB = 10
PHASE_QUERY = 11
PHASE_KV_PROJECT = 12
PHASE_KV_NORM = 13
PHASE_ORIGINAL_CACHE = 14


@pl.jit.incore
def cube(
    x_fp32: pl.Tensor[[T, 16384], pl.FP32],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    partials: pl.InOut[pl.Tensor[[PARTIAL, 32], pl.FP32]],
    normed: pl.Tensor[[T, 4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ],
    qa: pl.InOut[pl.Tensor[[P16, 1024], pl.FP32]],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ],
    qr_matmul: pl.Tensor[[QR_ROWS, 1024], pl.INT8],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    """AIC：QA发布后先做KV投影，让QR可同时推进；随后等QR再做QB。"""
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
    # normed已经等待HC_NORM；KV与QA读取同一BF16输入，wkv固定ND。
    # 原GM别名显式返回cluster，实际KV写回与通知次序必须另查编译产物。
    kv_fp32_gm = kv_fp32
    kv_fp32 = kv_project_stage(normed, wkv, kv_fp32, worker, GROUPS)
    signal = publish_phase(signal, PHASE_KV_PROJECT, worker, CORES, generation)
    signal = wait_row_owners(
        signal, PHASE_QR, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    # Q_B列块n归n%24；每个owner处理所有实际M128行块。返回原GM别名给cluster。
    q_proj_gm = q_proj
    q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, GROUPS)
    signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)
    return partials, qa_gm, q_proj_gm, kv_fp32_gm, signal


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
    qr_matmul: pl.InOut[pl.Tensor[[QR_ROWS, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_sin: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    rope_swap: pl.InOut[pl.Tensor[[T, 64], pl.INT32]],
    q_proj: pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    kv_fp32: pl.Tensor[[KV_ROWS, 512], pl.FP32],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
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
    # RoPE只依赖输入sin；放在QR发布后，与AIC的QB阶段并行，不等待QB。
    rope_signed, rope_swap = q_rope_prepare_stage(rope_sin, rope_signed, rope_swap, worker, VECTORS)
    signal = publish_phase(signal, PHASE_ROPE, core, CORES, generation)
    # 每个最多512行的KV分段有4个N128列块与最多3个M组；第一段的owner
    # 集合覆盖后续各段（完整512行段为12个owner，短尾段只是它的子集）。
    first_segment = pl.min(tokens, 512)
    if first_segment < 128:
        kv_tasks = 4 * pl.min(3, pl.max(1, first_segment // 32))
        signal = wait_row_owners(signal, PHASE_KV_PROJECT, CORES, generation, 0, kv_tasks, 1, GROUPS, 0, 1)
    else:
        kv_tasks = 4 * pl.min(3, pl.max(1, first_segment // 64))
        signal = wait_row_owners(signal, PHASE_KV_PROJECT, CORES, generation, 0, kv_tasks, 1, GROUPS, 0, 1)
    signal = wait_row_owners(
        signal, PHASE_ROPE, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    kv = kv_norm_rope_stage(kv_fp32, gamma_ckv, rope_cos, rope_signed, rope_swap, kv, worker, VECTORS)
    signal = publish_phase(signal, PHASE_KV_NORM, core, CORES, generation)
    # KVnorm每32行一个owner，写cache每8行一个owner，不能误用同一分块公式。
    signal = wait_row_owners(
        signal, PHASE_KV_NORM, CORES, generation, 0, tokens, 32, VECTORS, GROUPS, 1,
    )
    cache_rows = original_kv_writeback_stage(kv, native_slots, cache_rows, worker, VECTORS)
    signal = publish_phase(signal, PHASE_ORIGINAL_CACHE, core, CORES, generation)
    # 每个head占两个NZ N256列块。首版等全部128列块的24个真实owner。
    signal = wait_row_owners(signal, PHASE_QB, CORES, generation, 0, 128, 1, GROUPS, 0, 1)
    signal = wait_row_owners(
        signal, PHASE_ROPE, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    signal = wait_row_owners(
        signal, PHASE_QR, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    q_flat = q_dequant_rope_stage(
        wq_b_scale, rope_cos, rope_signed, rope_swap, q_flat, qr_scale_pad, q_proj, worker, VECTORS,
    )
    signal = publish_phase(signal, PHASE_QUERY, core, CORES, generation)
    return (x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
            qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_flat, kv, cache_rows, signal)


@pl.jit.inline
def group(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ],
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
    qr_matmul: pl.InOut[pl.Tensor[[QR_ROWS, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_sin: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    rope_swap: pl.InOut[pl.Tensor[[T, 64], pl.INT32]],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.cluster(name_hint="dynamic_hc_qkv_front"):
        partials, qa, q_proj, kv_fp32, signal = cube(
            x_fp32, hc_fn, partials, normed, wq_a, qa, wq_b, qr_matmul, q_proj, wkv, kv_fp32, signal,
        )
        (x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
         qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_flat, kv, cache_rows, signal) = vector(
            x, hc_scale, hc_base, hc_base_2d, norm_w, gamma_cq, x_fp32, inv, widen_tail, partials,
            raw, pre, post, post_tail, comb, comb_tail, mixed, normed, norm_tail, qa, qr, qr_scale,
            qr_matmul, qr_scale_pad, wq_b_scale, rope_cos, rope_sin, rope_signed, rope_swap, q_proj, q_flat,
            gamma_ckv, kv_fp32, kv, native_slots, cache_rows, signal,
        )
    return (partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
            qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_proj, q_flat, kv_fp32, kv, cache_rows, signal)


@pl.jit
def entry(
    x: pl.Tensor[[T, 16384], pl.BF16],
    hc_fn: pl.Tensor[[24, 16384], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[24], pl.FP32],
    hc_base_2d: pl.Tensor[[1, 24], pl.FP32],
    norm_w: pl.Tensor[[4096], pl.BF16],
    wq_a: pl.Tensor[[1024, 4096], pl.BF16, pl.NZ],
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
    qr_matmul: pl.InOut[pl.Tensor[[QR_ROWS, 1024], pl.INT8]],
    qr_scale_pad: pl.InOut[pl.Tensor[[P8, 1], pl.FP32]],
    wq_b: pl.Tensor[[1024, 32768], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[32768], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_sin: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.InOut[pl.Tensor[[T, 64], pl.FP32]],
    rope_swap: pl.InOut[pl.Tensor[[T, 64], pl.INT32]],
    q_proj: pl.InOut[pl.Tensor[[QPROJ_ROWS, 32768], pl.INT32]],
    q_flat: pl.InOut[pl.Tensor[[T, 32768], pl.BF16]],
    wkv: pl.Tensor[[4096, 512], pl.BF16],
    gamma_ckv: pl.Tensor[[512], pl.BF16],
    kv_fp32: pl.InOut[pl.Tensor[[KV_ROWS, 512], pl.FP32]],
    kv: pl.InOut[pl.Tensor[[T, 512], pl.BF16]],
    native_slots: pl.Tensor[[T, 2], pl.INT32],
    cache_rows: pl.InOut[pl.Tensor[[CACHE_ROWS, 512], pl.BF16]],
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.spmd(GROUPS, sync_start=True):
        (partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
         qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_proj, q_flat, kv_fp32, kv, cache_rows, signal) = group(
            x, hc_fn, hc_scale, hc_base, hc_base_2d, norm_w, wq_a, gamma_cq, x_fp32, inv,
            widen_tail, partials, raw, pre, post, post_tail, comb, comb_tail, mixed, normed,
            norm_tail, qa, qr, qr_scale, qr_matmul, qr_scale_pad, wq_b, wq_b_scale, rope_cos, rope_sin,
            rope_signed, rope_swap, q_proj, q_flat, wkv, gamma_ckv, kv_fp32, kv, native_slots, cache_rows, signal,
        )
    return (partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
            qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_proj, q_flat, kv_fp32, kv, cache_rows, signal)
