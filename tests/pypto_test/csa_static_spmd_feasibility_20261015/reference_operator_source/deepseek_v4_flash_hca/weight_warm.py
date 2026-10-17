# SPDX-License-Identifier: Apache-2.0
"""权重 L2 预热：由空闲 Vector 核提前按块读一遍随后要用的 ND 权重。

冷 L2 下每个 AIC 核读权重只有约 24 GB/s，投影阶段 24 个核合计远低于 HBM 带宽；
同期 Vector 核大多空闲。这里让 Vector 核并发读取即将使用的 ND 权重，把它们带进
L2，随后的矩阵乘从 L2 命中。NZ 权重只能以 Cube 操作数读入 Mat，不在此预热。
任务只读权重，并把每块首行的一小段写入本地哨兵，防止读取被当作无用代码删除；
哨兵不被任何计算读取，因此不改变任何数值。
"""

import pypto.language as pl

from ..deepseek_v4_flash_csa.nz_mode import WO_A_WEIGHT_LAYOUT

D = 4096
HEAD_DIM = 512
O_GROUPS = 8
O_GROUP_IN = 4096
O_LORA = 1024

WARM_WORKERS = 8  # 少量块，减少与 mHC pre 阶段边界同时到来的完成事件
BF16_ROWS = 64
BF16_COLS = 512
SINK_BF16 = 16

WKV_COL_TILES = HEAD_DIM // BF16_COLS
WKV_TILES = (D // BF16_ROWS) * WKV_COL_TILES
CMP_COL_TILES = D // BF16_COLS
CMP_TILES = (HEAD_DIM // BF16_ROWS) * CMP_COL_TILES
WO_A_COL_TILES = O_LORA // BF16_COLS
WO_A_TILES = (O_GROUPS * O_GROUP_IN // BF16_ROWS) * WO_A_COL_TILES


@pl.jit.inline
def warm_kv_weights(
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    sink_bf16: pl.Tensor[[WARM_WORKERS, SINK_BF16], pl.BF16],
    after_widen: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    # 层入口的 widen 读取残差流；预热在其后启动，避免与之争抢带宽。
    with pl.spmd(WARM_WORKERS, name_hint="hca_warm_kv_weights", allow_early_resolve=True, deps=[after_widen]) as warm_tid:
        worker = pl.tile.get_block_idx()
        for item in pl.pipeline(worker, WKV_TILES, WARM_WORKERS, stage=2):
            r0 = item // WKV_COL_TILES * BF16_ROWS
            c0 = item % WKV_COL_TILES * BF16_COLS
            kv_tile = pl.load(wkv, [r0, c0], [BF16_ROWS, BF16_COLS], target_memory=pl.MemorySpace.Vec)
            pl.store(pl.tile.slice(kv_tile, [1, SINK_BF16], [0, 0]), [worker, 0], sink_bf16)
        for item in pl.pipeline(worker, CMP_TILES, WARM_WORKERS, stage=2):
            r0 = item // CMP_COL_TILES * BF16_ROWS
            c0 = item % CMP_COL_TILES * BF16_COLS
            cmp_kv_tile = pl.load(cmp_wkv, [r0, c0], [BF16_ROWS, BF16_COLS], target_memory=pl.MemorySpace.Vec)
            pl.store(pl.tile.slice(cmp_kv_tile, [1, SINK_BF16], [0, 0]), [worker, 0], sink_bf16)
        for item in pl.pipeline(worker, CMP_TILES, WARM_WORKERS, stage=2):
            r0 = item // CMP_COL_TILES * BF16_ROWS
            c0 = item % CMP_COL_TILES * BF16_COLS
            cmp_gate_tile = pl.load(cmp_wgate, [r0, c0], [BF16_ROWS, BF16_COLS], target_memory=pl.MemorySpace.Vec)
            pl.store(pl.tile.slice(cmp_gate_tile, [1, SINK_BF16], [0, 0]), [worker, 0], sink_bf16)
    return warm_tid


@pl.jit.inline
def warm_wo_a(
    wo_a: pl.Tensor[[O_GROUPS, O_GROUP_IN, O_LORA], pl.BF16, WO_A_WEIGHT_LAYOUT],
    sink_bf16: pl.Tensor[[WARM_WORKERS, SINK_BF16], pl.BF16],
    after_q_a: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    # wo_a 与 Native 同为 ND；Q_A 之后 Vector 核在 Q_B 期间空闲，提前把它带进 L2。
    wo_a_rows = pl.reshape(wo_a, [O_GROUPS * O_GROUP_IN, O_LORA])
    with pl.spmd(WARM_WORKERS, name_hint="hca_warm_wo_a", allow_early_resolve=True, deps=[after_q_a]) as warm_tid:
        worker = pl.tile.get_block_idx()
        for item in pl.pipeline(worker, WO_A_TILES, WARM_WORKERS, stage=2):
            r0 = item // WO_A_COL_TILES * BF16_ROWS
            c0 = item % WO_A_COL_TILES * BF16_COLS
            oa_tile = pl.load(wo_a_rows, [r0, c0], [BF16_ROWS, BF16_COLS], target_memory=pl.MemorySpace.Vec)
            pl.store(pl.tile.slice(oa_tile, [1, SINK_BF16], [0, 0]), [worker, 0], sink_bf16)
    return warm_tid
