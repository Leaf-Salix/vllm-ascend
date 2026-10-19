# SPDX-License-Identifier: Apache-2.0
"""直接 GM→L1 行采集的三槽生命周期探针，不是完整 CSA 或性能用例。

目标是先确认现有 PyPTO/PTOAS 能表达三个独立 Mat Tile 的动态循环携带，
并用 GM INT32 索引直接采集 BF16 行。索引切片只改变设备地址，没有 Host
取值；128×512 是 Attention 的数学块，不是固定 batch 或历史长度。
每拍写当前槽，两拍后消费旧槽，覆盖与消费之间保持三个槽的原生命周期。
消费用 QK 矩阵乘，确保旧数据真正参与输出，不能仅靠循环结束推断正确。
本探针不模拟 AIV、FFTS、生产者发布或完整三槽 Attention 事件协议。
"""

import pypto.language as pl

ROWS = pl.dynamic("DIRECT_L1_TABLE_ROWS")
INDICES = pl.dynamic("DIRECT_L1_INDICES")
OUTPUT = pl.dynamic("DIRECT_L1_OUTPUT_ROWS")


@pl.jit.incore
def triple_slot_core(
    table: pl.Tensor[[ROWS, 512], pl.BF16],
    indices: pl.Tensor[[INDICES, 1], pl.INT32],
    query: pl.Tensor[[16, 512], pl.BF16],
    out: pl.Out[pl.Tensor[[OUTPUT, 128], pl.FP32]],
):
    steps = pl.tensor.dim(indices, 0) // 128
    query_mat = pl.load(query, [0, 0], [16, 512], target_memory=pl.Mem.Mat)
    initial_a = pl.create_tile([128, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    initial_b = pl.create_tile([128, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    initial_c = pl.create_tile([128, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    for tick, (slot_a, slot_b, slot_c) in pl.range(
        steps + 2, init_values=(initial_a, initial_b, initial_c)
    ):
        if tick < steps:
            # full-width 连续行 GM 别名。范围由真实 indices 容量保证，不扩大分配。
            index_column = pl.tensor.slice(indices, [128, 1], [tick * 128, 0], gm_view=True)
            # Mat行采集接口要求[1,R]，同一128元素连续视图无损reshape，不搬运。
            index_view = pl.tensor.view(index_column, shape=[1, 128])
            if tick % 3 == 0:
                loaded_a = pl.tile.mgather(
                    table, index_view, coalesce="row", gather_oob="zero", target_memory=pl.Mem.Mat
                )
                filled_a, filled_b, filled_c = pl.yield_(loaded_a, slot_b, slot_c)
            else:
                if tick % 3 == 1:
                    loaded_b = pl.tile.mgather(
                        table, index_view, coalesce="row", gather_oob="zero", target_memory=pl.Mem.Mat
                    )
                    tail_a, tail_b, tail_c = pl.yield_(slot_a, loaded_b, slot_c)
                else:
                    loaded_c = pl.tile.mgather(
                        table, index_view, coalesce="row", gather_oob="zero", target_memory=pl.Mem.Mat
                    )
                    tail_a, tail_b, tail_c = pl.yield_(slot_a, slot_b, loaded_c)
                filled_a, filled_b, filled_c = pl.yield_(tail_a, tail_b, tail_c)
            next_a, next_b, next_c = pl.yield_(filled_a, filled_b, filled_c)
        else:
            next_a, next_b, next_c = pl.yield_(slot_a, slot_b, slot_c)
        if tick >= 2:
            completed = tick - 2
            # 不把不同物理L1槽合并成一个Tile phi：当前后端会插入A3不支持的
            # Mat→Mat TMOV。消费者留在对应分支，共用原循环槽和相同矩阵数学。
            if completed % 3 == 0:
                score_a = pl.matmul(query_mat, pl.tile.transpose_view(next_a), out_dtype=pl.FP32)
                pl.store(score_a, [completed * 16, 0], out)
            else:
                if completed % 3 == 1:
                    score_b = pl.matmul(query_mat, pl.tile.transpose_view(next_b), out_dtype=pl.FP32)
                    pl.store(score_b, [completed * 16, 0], out)
                else:
                    score_c = pl.matmul(query_mat, pl.tile.transpose_view(next_c), out_dtype=pl.FP32)
                    pl.store(score_c, [completed * 16, 0], out)
        slot_a, slot_b, slot_c = pl.yield_(next_a, next_b, next_c)
    return out


@pl.jit
def triple_slot(
    table: pl.Tensor[[ROWS, 512], pl.BF16],
    indices: pl.Tensor[[INDICES, 1], pl.INT32],
    query: pl.Tensor[[16, 512], pl.BF16],
    out: pl.Out[pl.Tensor[[OUTPUT, 128], pl.FP32]],
):
    with pl.spmd(1, name_hint="direct_l1_probe"):
        out = triple_slot_core(table, indices, query, out)
    return out
