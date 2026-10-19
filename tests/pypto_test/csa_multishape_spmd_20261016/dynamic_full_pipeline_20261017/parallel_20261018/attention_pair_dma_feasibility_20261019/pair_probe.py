# SPDX-License-Identifier: Apache-2.0
"""两条任意物理行的2D DMA可行性，不是完整CSA或性能验证。

两个有效物理行按地址递增，以真实连续span建立零拷贝视图，再只加载
该视图每行前512元素。额外span也必须在原allocation内；绝不通过扩大
描述符来越过真实storage。下降次序在UB恢复，重复/非法/大跨度退回逐行。
R与地址都在device，2/512是这项搬运微结构，不是固定batch。
"""
import pypto.language as pl

ROWS = pl.dynamic("PAIR_SOURCE_ROWS")
WIDTH = 512
# A3 GM->UB以32位字节gap编码；合法范围外不假设硬件会合成一次DMA。
MAX_ROW_DISTANCE = ((1 << 32) - 1) // (WIDTH * 2) + 1


@pl.jit.incore
def pair_core(
    cache: pl.Tensor[[ROWS, WIDTH], pl.BF16],
    indices: pl.Tensor[[2], pl.INT32],
    output: pl.Out[pl.Tensor[[2, WIDTH], pl.BF16]],
):
    rows = pl.tensor.dim(cache, 0)
    a = pl.cast(pl.read(indices, [0]), pl.INDEX)
    b = pl.cast(pl.read(indices, [1]), pl.INDEX)
    low = pl.min(a, b)
    high = pl.max(a, b)
    delta = high - low
    span_rows = 2 * delta
    pitch = delta * WIDTH
    result = pl.tile.full([2, WIDTH], dtype=pl.BF16, value=0.0)
    if a >= 0 and b >= 0 and a < rows and b < rows and delta > 0 and delta <= MAX_ROW_DISTANCE and low + span_rows <= rows:
        # span形状与实际字节相符，view前后元素数完全相同；加载只触及两端。
        span = pl.tensor.slice(cache, [span_rows, WIDTH], [low, 0], valid_shape=[span_rows, WIDTH])
        pitched = pl.tensor.view(span, shape=[2, pitch])
        gathered = pl.load(pitched, [0, 0], [2, WIDTH])
        if a <= b:
            result = pl.tile.assemble(result, gathered, [0, 0])
        else:
            first = pl.tile.slice(gathered, [1, WIDTH], [1, 0])
            second = pl.tile.slice(gathered, [1, WIDTH], [0, 0])
            result = pl.tile.assemble(result, first, [0, 0])
            result = pl.tile.assemble(result, second, [1, 0])
    else:
        for lane in pl.unroll(2):
            row = pl.cast(pl.read(indices, [lane]), pl.INDEX)
            if row >= 0 and row < rows:
                result = pl.gather_row(result, cache, [lane, 0], [row, 0], [1, WIDTH])
    pl.store(result, [0, 0], output)
    return output


@pl.jit
def pair_probe(
    cache: pl.Tensor[[ROWS, WIDTH], pl.BF16],
    indices: pl.Tensor[[2], pl.INT32],
    output: pl.Out[pl.Tensor[[2, WIDTH], pl.BF16]],
):
    with pl.spmd(1, name_hint="pair_row_dma_probe"):
        output = pair_core(cache, indices, output)
    return output
