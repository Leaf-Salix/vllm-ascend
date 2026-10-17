# SPDX-License-Identifier: Apache-2.0
"""只验证真实A3的GM→L1非对齐行搬运；不是CSA性能候选。"""

import pypto.language as pl


@pl.jit.inline
def gather_probe(src: pl.Tensor[[768, 512], pl.BF16],
                 indices: pl.Tensor[[128], pl.INT32],
                 selector: pl.Tensor[[4], pl.INT32],
                 eye: pl.Tensor[[128, 128], pl.BF16],
                 out: pl.Out[pl.Tensor[[384, 512], pl.FP32]]):
    # 与完整QK相同的三槽L1尺寸；先用不同于源行的数据填充全部槽。
    slots = pl.tile.create([384, 512], dtype=pl.BF16, target_memory=pl.Mem.Mat)
    slots = pl.tile.gather_row(slots, src, [0, 0], [384, 0], [384, 512])
    slot = pl.cast(pl.tensor.read(selector, [0]), pl.INDEX)
    mode = pl.cast(pl.tensor.read(selector, [1]), pl.INDEX)
    if mode == 0:
        for row in pl.range(128):
            physical = pl.cast(pl.tensor.read(indices, [row]), pl.INDEX)
            slots = pl.tile.gather_row(slots, src, [slot * 128 + row, 0], [physical, 0], [1, 512])
    else:
        start = pl.cast(pl.tensor.read(selector, [2]), pl.INDEX)
        count = pl.cast(pl.tensor.read(selector, [3]), pl.INDEX)
        slots = pl.tile.gather_row(slots, src, [slot * 128 + start, 0], [0, 0],
                                   [32, 512], valid_shape=[count, 512])
    identity = pl.tile.load(eye, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
    left = pl.tile.move(identity, target_memory=pl.Mem.Left)
    # 通过单位阵读回全部三槽。逐元素核对能同时发现相邻行、相邻槽被覆盖。
    for which in pl.range(3):
        for column in pl.range(4):
            right = pl.tile.extract(slots, which * 128, column * 128, [128, 128],
                                    target_memory=pl.Mem.Right)
            result = pl.tile.matmul(left, right)
            out = pl.tile.store(result, [which * 128, column * 128], out)
    return out


@pl.jit
def entry(src: pl.Tensor[[768, 512], pl.BF16], indices: pl.Tensor[[128], pl.INT32],
          selector: pl.Tensor[[4], pl.INT32], eye: pl.Tensor[[128, 128], pl.BF16],
          out: pl.Out[pl.Tensor[[384, 512], pl.FP32]]):
    for worker in pl.spmd(1):
        out = gather_probe(src, indices, selector, eye, out)
    return out
