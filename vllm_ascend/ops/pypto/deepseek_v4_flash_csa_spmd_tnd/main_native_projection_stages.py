# SPDX-License-Identifier: Apache-2.0
"""主Compressor投影数值候选；保留原M64/N64逻辑生产者，内部拆N32。

K128与列组循环偏移参考Native Compressor及旧精度分支，不改state/池化或信号。
非joint生产路径接入K128/ROTATE；参数保留用于单因素对拍。
"""

import pypto.language as pl

from .compressor_projection_stages import CMP_M_TILE, CMP_N_TILE, CMP_OUT_DIM, MAIN_PAD_ROWS_DYN, T_DYN, D
from .request_geometry import QUERY_BOUNDS_DYN


@pl.jit.inline(auto_scope=False)
def main_native_projection_stage(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[CMP_OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[CMP_OUT_DIM, D], pl.BF16],
    bounds: pl.Tensor[[QUERY_BOUNDS_DYN], pl.INT32],
    kv_proj: pl.InOut[pl.Tensor[[MAIN_PAD_ROWS_DYN, CMP_OUT_DIM], pl.FP32]],
    score_proj: pl.InOut[pl.Tensor[[MAIN_PAD_ROWS_DYN, CMP_OUT_DIM], pl.FP32]],
    worker: pl.Scalar[pl.INDEX],
    WORKERS: pl.constexpr,
    K_STEP: pl.constexpr,
    ROTATE: pl.constexpr,
):
    tokens = pl.tensor.dim(x, 0)
    padded = (tokens + CMP_M_TILE - 1) // CMP_M_TILE * CMP_M_TILE
    column_group = pl.cast(64, pl.INDEX)
    if ROTATE:
        # Native arch32 SetBaseSize: uniform requests select 16 groups/head
        # at this capacity; ragged requests retain 8. Read live TND bounds.
        first = pl.read(bounds, [1]) - pl.read(bounds, [0])
        # Native Init trims only trailing zero-span requests before SetBaseSize.
        # Interior zero spans still participate; L=0 with a nonzero Q span does too.
        live_requests = pl.cast(0, pl.INDEX)
        for request in pl.range(pl.tensor.dim(bounds, 0) - 1):
            if pl.read(bounds, [request + 1]) > pl.read(bounds, [request]):
                live_requests = request + 1
        deviation = pl.cast(0, pl.INT32)
        for request in pl.range(1, live_requests):
            delta = pl.read(bounds, [request + 1]) - pl.read(bounds, [request]) - first
            deviation = deviation + delta * delta
        if deviation == 0 and tokens <= 384:
            column_group = pl.cast(32, pl.INDEX)
    for job in pl.range(worker, padded * CMP_OUT_DIM // (CMP_M_TILE * CMP_N_TILE), WORKERS):
        row = job // (CMP_OUT_DIM // CMP_N_TILE) * CMP_M_TILE
        parent_column = job % (CMP_OUT_DIM // CMP_N_TILE) * CMP_N_TILE
        rows = pl.min(CMP_M_TILE, tokens - row)
        for sub in pl.range(2):
            column = parent_column + sub * 32
            value = pl.create_tensor([CMP_M_TILE, 32], dtype=pl.FP32)
            score = pl.create_tensor([CMP_M_TILE, 32], dtype=pl.FP32)
            for step in pl.pipeline(0, D // K_STEP, stage=2):
                offset = step * K_STEP
                if ROTATE:
                    offset = (step * K_STEP + (column % 512 // column_group) * 256) % D
                a = pl.slice(x, [CMP_M_TILE, K_STEP], [row, offset], valid_shape=[rows, K_STEP])
                b_value = pl.slice(wkv, [32, K_STEP], [column, offset])
                b_score = pl.slice(wgate, [32, K_STEP], [column, offset])
                # Preserve the original compact-accumulator peel for dynamic M64.
                if step == 0:
                    value = pl.matmul(a, b_value, out_dtype=pl.FP32, b_trans=True)
                    score = pl.matmul(a, b_score, out_dtype=pl.FP32, b_trans=True)
                else:
                    value = pl.matmul_acc(value, a, b_value, b_trans=True)
                    score = pl.matmul_acc(score, a, b_score, b_trans=True)
            kv_proj[row : row + CMP_M_TILE, column : column + 32] = value
            score_proj[row : row + CMP_M_TILE, column : column + 32] = score
    return kv_proj, score_proj
