# SPDX-License-Identifier: Apache-2.0
"""HC→QA→QR→QB→query RMS/RoPE 的源级动态单 MIX 编译探针。

一、范围与布局
    只组装 query 前段，不是完整 CSA。数学来自正式 hc_stages/qkv_stages/
    query_projection_stages，未导入冻结 IR。必须在进程 NZ2 模式下导入：
    QA 是 BF16 NZ 权重，QB 是 INT8 NZ 权重；本探针不转换权重。

二、动态 ABI 与工作区
    T 是实际 token 数，P16=ceil(T/16)*16、P8=ceil(T/8)*8。HC_PARTIAL
    为 4*P16；QR_ROWS 和 QPROJ_ROWS 分别是 INT8 QR 与 INT32 QB 暂存的
    动态容量，均至少 ceil(T/128)*128。QR scale 仍只需 P8，不能误把这两种
    容量绑定为同一个 P8。QB 只依据实际 T 计算，不扫描额外容量行。
    q_flat[T,32768] 由调用层建立 GM 二维视图，必须与下游 q[T,64,512]
    共用数据；hc_base 两个视图亦由外层绑定。BF16 norm/gamma 保持原格式。

三、角色、生产消费与同步
    单个24组MIX：AIC执行HC投影→QA→等QR→QB；48个AIV执行widen/RMS、
    HC门值/comb/mix、QR，再准备RoPE，最后等QB和RoPE完成后反量化/RMS/RoPE。
    RoPE准备与QB可并行。当前保守等待全部真实活跃owner，不引入syncall。
    阶段0存代际，1..8沿用HC至QR，9=ROPE_PREP，10=QB，11=QUERY。
    每个核包括空核均begin一次；每个writer完成全部wave后发布该phase一次。
    完成信号无ACK，不用于逐tile覆盖槽；业务fence和缓存维护由phase_sync负责。

四、生命周期与验证边界
    signals预留12*72*128个INT32，首次清零，固定G24/phase语义，单流单图
    顺序复用；失败后重置。所有Tensor视图和容量由调用层保证，不读取device
    数值生成Host描述符。仅做CPU完整后端、动态ABI和实际存取/通知次序审计；
    未验证设备精度、缓存可见性、性能或完整CSA。显式cluster的外层SPMD不能挂
    BYPASS属性，full_v1已保留该真实失败；本轮不强加缓存策略，后续另处理。
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
from vllm_ascend.ops.pypto.deepseek_v4_flash_csa_spmd.query_projection_stages import (
    q_b_stage,
    q_dequant_rope_stage,
    q_rope_prepare_stage,
)

T = pl.dynamic("QUERY_FRONT_T")
P16 = pl.dynamic("QUERY_FRONT_PAD16")
P8 = pl.dynamic("QUERY_FRONT_PAD8")
PARTIAL = pl.dynamic("QUERY_FRONT_PARTIAL_ROWS")
QR_ROWS = pl.dynamic("QUERY_FRONT_QR_PAD128")
QPROJ_ROWS = pl.dynamic("QUERY_FRONT_PROJ_PAD128")
WORDS = pl.dynamic("QUERY_FRONT_SIGNAL_WORDS")

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
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    """AIC：HC投影、QA、QB共用同一个核程序，QR写完后才读取INT8暂存。"""
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
    signal = wait_row_owners(
        signal, PHASE_QR, CORES, generation, 0, tokens, 8, VECTORS, GROUPS, 1,
    )
    # Q_B列块n归n%24；每个owner处理所有实际M128行块。返回原GM别名给cluster。
    q_proj_gm = q_proj
    q_proj = q_b_stage(wq_b, qr_matmul, q_proj, tokens, worker, GROUPS)
    signal = publish_phase(signal, PHASE_QB, worker, CORES, generation)
    return partials, qa_gm, q_proj_gm, signal


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
            qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_flat, signal)


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
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.cluster(name_hint="dynamic_hc_query_front"):
        partials, qa, q_proj, signal = cube(
            x_fp32, hc_fn, partials, normed, wq_a, qa, wq_b, qr_matmul, q_proj, signal,
        )
        (x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
         qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_flat, signal) = vector(
            x, hc_scale, hc_base, hc_base_2d, norm_w, gamma_cq, x_fp32, inv, widen_tail, partials,
            raw, pre, post, post_tail, comb, comb_tail, mixed, normed, norm_tail, qa, qr, qr_scale,
            qr_matmul, qr_scale_pad, wq_b_scale, rope_cos, rope_sin, rope_signed, rope_swap, q_proj, q_flat, signal,
        )
    return (partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
            qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_proj, q_flat, signal)


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
    signal: pl.InOut[pl.Tensor[[WORDS], pl.INT32]],
):
    with pl.spmd(GROUPS, sync_start=True):
        (partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
         qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_proj, q_flat, signal) = group(
            x, hc_fn, hc_scale, hc_base, hc_base_2d, norm_w, wq_a, gamma_cq, x_fp32, inv,
            widen_tail, partials, raw, pre, post, post_tail, comb, comb_tail, mixed, normed,
            norm_tail, qa, qr, qr_scale, qr_matmul, qr_scale_pad, wq_b, wq_b_scale, rope_cos, rope_sin,
            rope_signed, rope_swap, q_proj, q_flat, signal,
        )
    return (partials, qa, x_fp32, inv, raw, pre, post, comb, mixed, normed, qr, qr_scale,
            qr_matmul, qr_scale_pad, rope_signed, rope_swap, q_proj, q_flat, signal)
