# SPDX-License-Identifier: Apache-2.0
"""Indexer query局部MIX链。只在本task内使用冻结参考的完成信号。

所有QUERY_WORKERS个AIC及两倍AIV每次共同推进代际，即使无数据也不跳过。
信号保留最大72参与者容量，实际编号/阶段步长和全部owner按本次固定核数组合计算。
各阶段完成本核GM写回后发布；仅等待实际生产者。GM中间存储、规约、
量化和Top-K规则不变；外部依赖和生命周期仍由TMR处理。
query_signal由准备对象首次清零并顺序独占，不能并发复用或跨布局重用。


新增WEIGHTS_PROJECT阶段：AIC先发布query投影，再在AIV做RoPE时计算head weights四份partial。
AIV发布RoPE后等待真实partial owner，再按原左折叠规约。QUANT通知也覆盖本核head weights写回；
head weights的owner集合由ceil(T/16)行块确定，是min(T,48)个quant owner的子集，故系数在
等待QUANT后才能读取完整weights。所有参与者仍推进同一代际，信号工作区改为六个阶段槽。

M256 Hadamard直接复用冻结84751e34的四query helper；仍按每行完整K128计算，
真实尾行由valid_shape约束。producer单位改为ceil(T/4)，等待集合一起更新。
只影响融合query路径；小容量独立task保持原数学和分工。
"""

import pypto.language as pl

from .indexer_group_stages import indexer_head_coefficients_stage
from .indexer_query_stages import (
    DEQUANT_T_TILE,
    DQ_ROPE_H_TILE,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    MM_ROW_TILE,
    QR_MM_N_TILE,
    WEIGHTS_OK,
    indexer_query_dequant_rope_stage,
    indexer_query_project_stage,
    indexer_query_quant_stage,
    indexer_weights_project_stage,
    indexer_weights_reduce_stage,
)
from .query_phase_sync import begin_invocation, publish_phase, wait_row_owners
from .runtime_abi import (
    COEFF_COLS,
    COEFF_ROWS,
    CORES,
    GROUP_OFFSET_ROWS,
    IDX_HEAD_ROWS,
    IDX_PROJ_ROWS,
    IDX_ROPE_HEAD_ROWS,
    IDX_ROPE_ROWS,
    IDX_SCALE_T_ROWS,
    IQ,
    WEIGHTS_PARTIAL_ROWS,
    WEIGHTS_ROWS,
    B,
    T,
)

QUERY_SIGNAL_WORDS = pl.dynamic("CSA_QUERY_SIGNAL_WORDS")
QUERY_SIGNAL_COUNT = 6 * CORES * 128
(PROJECT, ROPE, HADAMARD, QUANT) = (1, 2, 3, 4)
WEIGHTS_PROJECT = 5


@pl.jit.inline(auto_scope=False)
def _query_hadamard_four_tokens(
    query_rope: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    tokens: pl.Scalar[pl.INDEX],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
):
    """M256只合并独立行；每行的K128乘加、输出舍入和head顺序保持不变。"""
    rows = tokens * 64
    weight = hadamard[0:128, 0:128]
    for block in pl.range(worker, (rows + 255) // 256, WORKERS):
        first = block * 256
        valid = pl.min(256, rows - first)
        source = pl.slice(query_rope, [256, 128], [first, 0], valid_shape=[valid, 128])
        result = pl.matmul(source, weight, out_dtype=pl.FP32)
        query_had = pl.assemble(query_had, result, [first, 0])
    return query_had


@pl.jit.incore
def query_chain_cube(
    qr: pl.Tensor[[T, 1024], pl.INT8],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
    idx_query_rope_heads: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    normed: pl.Tensor[[T, 4096], pl.BF16],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    idx_query_acc: pl.InOut[pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32]],
    idx_query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    query_signal: pl.InOut[pl.Tensor[[QUERY_SIGNAL_WORDS], pl.INT32]],
    QUERY_WORKERS: pl.constexpr,
):
    worker = pl.tile.get_block_idx()
    tokens = pl.tensor.dim(qr, 0)
    (query_signal, generation) = begin_invocation(query_signal, worker)
    idx_query_acc = indexer_query_project_stage(qr, idx_wq_b, idx_query_acc, worker, QUERY_WORKERS)
    query_signal = publish_phase(query_signal, PROJECT, worker, 3 * QUERY_WORKERS, generation)
    weights_partial = indexer_weights_project_stage(normed, weights_proj, weights_partial, worker, QUERY_WORKERS)
    query_signal = publish_phase(query_signal, WEIGHTS_PROJECT, worker, 3 * QUERY_WORKERS, generation)
    rope_units = (tokens + DEQUANT_T_TILE - 1) // DEQUANT_T_TILE * (IDX_N_HEADS // DQ_ROPE_H_TILE)
    query_signal = wait_row_owners(
        query_signal, ROPE, 3 * QUERY_WORKERS, generation, 0, rope_units, 1, 2 * QUERY_WORKERS, QUERY_WORKERS, 1
    )
    idx_query_had = _query_hadamard_four_tokens(
        idx_query_rope_heads, hadamard, idx_query_had, tokens, worker, QUERY_WORKERS
    )
    query_signal = publish_phase(query_signal, HADAMARD, worker, 3 * QUERY_WORKERS, generation)
    return (idx_query_acc, idx_query_had, weights_partial, query_signal)


@pl.jit.incore
def query_chain_vector(
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.Tensor[[T, 64], pl.FP32],
    idx_query_acc: pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32],
    idx_query_had: pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32],
    idx_query_scale_heads: pl.Tensor[[IDX_SCALE_T_ROWS, 64], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    weights_partial: pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32],
    idx_query_rope: pl.InOut[pl.Tensor[[IDX_ROPE_ROWS, 8192], pl.BF16]],
    head_weights: pl.InOut[pl.Tensor[[WEIGHTS_ROWS, 64], pl.FP32]],
    idx_query_i8: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8]],
    idx_query_scale: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 1], pl.FP32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16]],
    query_signal: pl.InOut[pl.Tensor[[QUERY_SIGNAL_WORDS], pl.INT32]],
    QUERY_GROUP: pl.constexpr,
    QUERY_WORKERS: pl.constexpr,
):
    pl.func_attr({"split_aiv": True})
    worker = pl.tile.get_block_idx() * 2 + pl.tile.get_subblock_idx()
    core = QUERY_WORKERS + worker
    tokens = pl.tensor.dim(qr_scale, 0)
    (query_signal, generation) = begin_invocation(query_signal, core)
    query_signal = wait_row_owners(
        query_signal,
        PROJECT,
        3 * QUERY_WORKERS,
        generation,
        0,
        IDX_N_HEADS * IDX_HEAD_DIM // QR_MM_N_TILE,
        1,
        QUERY_WORKERS,
        0,
        1,
    )
    idx_query_rope = indexer_query_dequant_rope_stage(
        qr_scale, idx_wq_b_scale, rope_cos, rope_signed, idx_query_acc, idx_query_rope, worker, 2 * QUERY_WORKERS
    )
    query_signal = publish_phase(query_signal, ROPE, core, 3 * QUERY_WORKERS, generation)
    weight_units = WEIGHTS_OK * ((tokens + MM_ROW_TILE - 1) // MM_ROW_TILE)
    query_signal = wait_row_owners(
        query_signal, WEIGHTS_PROJECT, 3 * QUERY_WORKERS, generation, 0, weight_units, 1, QUERY_WORKERS, 0, 1
    )
    head_weights = indexer_weights_reduce_stage(weights_partial, head_weights, tokens, worker, 2 * QUERY_WORKERS)
    query_signal = wait_row_owners(
        query_signal, HADAMARD, 3 * QUERY_WORKERS, generation, 0, (tokens + 3) // 4, 1, QUERY_WORKERS, 0, 1
    )
    (idx_query_i8, idx_query_scale) = indexer_query_quant_stage(
        idx_query_had, idx_query_i8, idx_query_scale, tokens, worker, 2 * QUERY_WORKERS
    )
    query_signal = publish_phase(query_signal, QUANT, core, 3 * QUERY_WORKERS, generation)
    query_signal = wait_row_owners(
        query_signal, QUANT, 3 * QUERY_WORKERS, generation, 0, tokens, 1, 2 * QUERY_WORKERS, QUERY_WORKERS, 1
    )
    coefficients = indexer_head_coefficients_stage(
        idx_query_scale_heads,
        head_weights,
        positions,
        idx_bounds,
        idx_lengths,
        group_offsets,
        coefficients,
        worker,
        2 * QUERY_WORKERS,
        QUERY_GROUP,
    )
    return (idx_query_rope, head_weights, idx_query_i8, idx_query_scale, coefficients, query_signal)


@pl.jit.inline(auto_scope=False)
def indexer_query_chain(
    qr: pl.Tensor[[T, 1024], pl.INT8],
    idx_wq_b: pl.Tensor[[1024, 8192], pl.INT8],
    idx_query_rope_heads: pl.Tensor[[IDX_ROPE_HEAD_ROWS, 128], pl.BF16],
    hadamard: pl.Tensor[[128, 128], pl.BF16],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    idx_wq_b_scale: pl.Tensor[[8192], pl.FP32],
    rope_cos: pl.Tensor[[T, 64], pl.FP32],
    rope_signed: pl.Tensor[[T, 64], pl.FP32],
    idx_query_scale_heads: pl.Tensor[[IDX_SCALE_T_ROWS, 64], pl.FP32],
    positions: pl.Tensor[[T], pl.INT64],
    idx_bounds: pl.Tensor[[IQ], pl.INT32],
    idx_lengths: pl.Tensor[[B], pl.INT32],
    group_offsets: pl.Tensor[[GROUP_OFFSET_ROWS], pl.INT32],
    normed: pl.Tensor[[T, 4096], pl.BF16],
    weights_proj: pl.Tensor[[4096, 64], pl.BF16],
    idx_query_acc: pl.InOut[pl.Tensor[[IDX_PROJ_ROWS, 8192], pl.INT32]],
    idx_query_rope: pl.InOut[pl.Tensor[[IDX_ROPE_ROWS, 8192], pl.BF16]],
    idx_query_had: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.FP32]],
    weights_partial: pl.InOut[pl.Tensor[[WEIGHTS_PARTIAL_ROWS, 64], pl.FP32]],
    head_weights: pl.InOut[pl.Tensor[[WEIGHTS_ROWS, 64], pl.FP32]],
    idx_query_i8: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 128], pl.INT8]],
    idx_query_scale: pl.InOut[pl.Tensor[[IDX_HEAD_ROWS, 1], pl.FP32]],
    coefficients: pl.InOut[pl.Tensor[[COEFF_ROWS, COEFF_COLS], pl.FP16]],
    query_signal: pl.InOut[pl.Tensor[[QUERY_SIGNAL_WORDS], pl.INT32]],
    QUERY_GROUP: pl.constexpr,
    QUERY_WORKERS: pl.constexpr,
):
    with pl.cluster(name_hint="indexer_query_chain"):
        (idx_query_acc, idx_query_had, weights_partial, query_signal) = query_chain_cube(
            qr,
            idx_wq_b,
            idx_query_rope_heads,
            hadamard,
            normed,
            weights_proj,
            idx_query_acc,
            idx_query_had,
            weights_partial,
            query_signal,
            QUERY_WORKERS,
        )
        (idx_query_rope, head_weights, idx_query_i8, idx_query_scale, coefficients, query_signal) = query_chain_vector(
            qr_scale,
            idx_wq_b_scale,
            rope_cos,
            rope_signed,
            idx_query_acc,
            idx_query_had,
            idx_query_scale_heads,
            positions,
            idx_bounds,
            idx_lengths,
            group_offsets,
            weights_partial,
            idx_query_rope,
            head_weights,
            idx_query_i8,
            idx_query_scale,
            coefficients,
            query_signal,
            QUERY_GROUP,
            QUERY_WORKERS,
        )
    return (
        idx_query_acc,
        idx_query_rope,
        idx_query_had,
        weights_partial,
        head_weights,
        idx_query_i8,
        idx_query_scale,
        coefficients,
        query_signal,
    )
