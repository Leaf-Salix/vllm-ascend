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

def test_ptoas_column_carry_move_uses_destination_view():
    """A column carry can share storage with a row-major full initializer."""

    @pl.program
    class ColumnCarry:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
        ) -> pl.Tensor[[32, 1], pl.FP32]:
            row_base = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            base = pl.tile.reshape(row_base, [32, 1])
            for i, (carry,) in pl.range(2, init_values=(base,)):
                if flag != 0:
                    row = pl.reshape(carry, [1, 32])
                    updated = pl.add(row, 1.0)
                    column = pl.reshape(updated, [32, 1])
                    value = column
                else:
                    value = carry
                result = pl.yield_(value)
            output = pl.store(result, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ColumnCarry)
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert moves, mlir
    for move in moves:
        shapes = re.findall(r"rows=(\d+), cols=(\d+)", move)
        assert len(shapes) == 2 and shapes[0] == shapes[1], move
