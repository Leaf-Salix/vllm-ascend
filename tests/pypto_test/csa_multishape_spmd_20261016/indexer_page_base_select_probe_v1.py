# SPDX-License-Identifier: Apache-2.0
"""CPU探针：先选择已有GM视图，再做一次Key加载；不等同于Tensor slice。

只有确认生成代码真实选择两份输入指针、且没有额外GM复制，才能推广到CSA。
本探针没有设备结果，不用于宣称精度或性能。
"""

import pypto.language as pl


@pl.jit
def entry(
    even: pl.Tensor[[96, 128], pl.INT8],
    odd: pl.Tensor[[96, 128], pl.INT8],
    query: pl.Tensor[[16, 128], pl.INT8],
    control: pl.Tensor[[2], pl.INT32],
    out: pl.Out[pl.Tensor[[16, 32], pl.INT32]],
):
    with pl.spmd(1, name_hint="page_base_select"):
        parity = pl.tensor.read(control, [0])
        row = pl.cast(pl.tensor.read(control, [1]), pl.INDEX)
        if parity == 0:
            source = pl.yield_(even)
        else:
            source = pl.yield_(odd)
        key = pl.tile.create([32, 128], dtype=pl.INT8, target_memory=pl.Mem.Mat)
        key = pl.tile.gather_row(key, source, [0, 0], [row, 0], [32, 128])
        q = pl.tile.load(query, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
        score = pl.matmul(q, pl.tile.transpose_view(key), out_dtype=pl.INT32)
        out = pl.tile.store(score, [0, 0], out)
    return out
