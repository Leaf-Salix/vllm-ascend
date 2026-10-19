# SPDX-License-Identifier: Apache-2.0
"""只比较根GM视图与GM切片后视图的编译行为，不声明两者读取相同行。"""
import pypto.language as pl

ROWS = pl.dynamic("VIEW_SOURCE_ROWS")


@pl.jit.incore
def slice_view_core(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    # 要求调用者提供至少4行。元素数4*512 == 2*1024，视图没有扩大存储。
    span = pl.tensor.slice(x, [4, 512], [0, 0], gm_view=True)
    pitched = pl.tensor.view(span, shape=[2, 1024])
    tile = pl.load(pitched, [0, 0], [2, 512])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit.incore
def root_view_core(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    # 对照仅确认直接根GM Tensor可view；读取0/1行，不是上述0/2行的数值对照。
    rows = pl.tensor.dim(x, 0)
    direct = pl.tensor.view(x, shape=[rows, 512])
    tile = pl.load(direct, [0, 0], [2, 512])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit
def slice_view(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    with pl.spmd(1, name_hint="slice_view"):
        out = slice_view_core(x, out)
    return out


@pl.jit
def root_view(x: pl.Tensor[[ROWS, 512], pl.BF16], out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
    with pl.spmd(1, name_hint="root_view"):
        out = root_view_core(x, out)
    return out
