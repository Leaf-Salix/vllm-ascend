# SPDX-License-Identifier: Apache-2.0
"""两个JIT形状问题的纯CPU复现：不创建NPU tensor，不编译或执行设备代码。"""

import json

import pypto.language as pl

N = pl.dynamic("SHAPE_REPRO_N")


@pl.jit.incore
def write_workspace(work: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]):
    zero = pl.tile.full([16, 8], dtype=pl.FP32, value=0)
    pl.store(zero, [0, 0], work)
    pl.store(zero, [0, 0], out)
    return work, out


@pl.jit
def tuple_shape(out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]):
    work = pl.create_tensor((16, 8), dtype=pl.FP32)
    with pl.spmd(1):
        work, out = write_workspace(work, out)


@pl.jit
def list_shape(out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]):
    work = pl.create_tensor([16, 8], dtype=pl.FP32)
    with pl.spmd(1):
        work, out = write_workspace(work, out)


@pl.jit
def reshape_scalar_alias(x: pl.Tensor[[N, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]):
    n = pl.tensor.dim(x, 0)
    padded = (n + 15) // 16 * 16
    work = pl.create_tensor([padded, 16], dtype=pl.FP32)
    view = pl.reshape(work, [padded * 2, 8])
    with pl.spmd(1):
        view, out = write_workspace(view, out)


@pl.jit
def reshape_descriptor(x: pl.Tensor[[N, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]):
    n = pl.tensor.dim(x, 0)
    padded = (n + 15) // 16 * 16
    work = pl.create_tensor([padded, 16], dtype=pl.FP32)
    rows = pl.tensor.dim(work, 0)
    view = pl.reshape(work, [rows * 2, 8])
    with pl.spmd(1):
        view, out = write_workspace(view, out)


if __name__ == "__main__":
    for entry in (tuple_shape, list_shape, reshape_scalar_alias, reshape_descriptor):
        try:
            entry.specialize()
        except Exception as error:
            print(json.dumps({"entry": entry.__name__, "status": "FAIL", "error": str(error)}), flush=True)
        else:
            print(json.dumps({"entry": entry.__name__, "status": "PASS"}), flush=True)
