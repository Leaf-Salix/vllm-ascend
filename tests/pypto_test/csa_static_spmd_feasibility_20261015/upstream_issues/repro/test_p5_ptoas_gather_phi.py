# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_conditional_gather_preserves_one_mat_buffer():
    """Two in-place branch writes must not introduce an illegal Mat->Mat copy."""

    @pl.program
    class ConditionalGather:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            base = pl.tile.create([32, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
            if flag != 0:
                first = pl.tile.gather_row(base, x, [0, 0], [0, 0], [16, 64])
            else:
                first = pl.tile.gather_row(base, x, [0, 0], [16, 0], [16, 64])
            if flag != 0:
                second = pl.tile.gather_row(first, x, [16, 0], [32, 0], [16, 64])
            else:
                second = pl.tile.gather_row(first, x, [16, 0], [48, 0], [16, 64])
            transposed = pl.tile.transpose_view(second)
            rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
            left = pl.tile.move(transposed, target_memory=pl.Mem.Left)
            right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
            product = pl.tile.matmul(left, right)
            output = pl.tile.store(product, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ConditionalGather)
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert not any(line.count("loc=mat") == 2 for line in moves), mlir
    bases = {
        line.split("pto.subview ", 1)[1].split("[", 1)[0]
        for line in mlir.splitlines()
        if "pto.subview " in line
    }
    assert len(bases) == 1, mlir
