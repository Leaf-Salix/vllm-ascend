# SPDX-License-Identifier: Apache-2.0
"""JIT 描述符维度算术的 CPU 最小复现；不创建、不运行设备任务。"""

import pypto.language as pl

PAGES = pl.dynamic("REPRO_CACHE_PAGES")


@pl.jit.incore
def copy_key(keys: pl.Tensor, out: pl.Out[pl.Tensor]):
    value = pl.load(keys, [0, 0], [1, 128])
    pl.store(value, [0, 0], out)
    return out


@pl.jit
def entry(
    src: pl.Tensor[[PAGES, 4160], pl.INT8],
    out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
):
    pages = pl.tensor.dim(src, 0)
    width = pl.tensor.dim(src, 1)
    flat = pl.reshape(src, [pages * width])
    shifted = flat[64 : 64 + ((pages * width - 64) // 128) * 128]
    keys = pl.reshape(shifted, [(pages * width - 64) // 128, 128])
    return copy_key(keys, out)


if __name__ == "__main__":
    entry.specialize()
