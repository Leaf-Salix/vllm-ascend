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

def test_ptoas_reshape_reads_phi_handle_after_branch():
    """A post-if view cannot reuse a same-typed but branch-local old handle."""

    @pl.program
    class BranchView:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[32, 1], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            base = pl.load(x, [0, 0], [32, 1])
            if flag != 0:
                row = pl.reshape(base, [1, 32])
                updated = pl.add(row, 1.0)
                column = pl.reshape(updated, [32, 1])
                result = column
            else:
                result = base
            after = pl.reshape(result, [1, 32])
            head = pl.tile.extract(after, 0, 0, [1, 16], target_memory=pl.Mem.Vec)
            output = pl.store(head, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, BranchView)
    mlir = _codegen(optimized, emit_tile_addr=False)
    extract = next(line for line in mlir.splitlines() if "pto.textract ins(" in line)
    source = extract.split("ins(", 1)[1].split(",", 1)[0]
    view = next((line.strip() for line in mlir.splitlines() if line.strip().startswith(source + " =")), "")
    assert "= pto.treshape " in view, mlir
    phi = view.split("= pto.treshape ", 1)[1].split()[0]
    assert any("pto.tmov" in line and "outs(" + phi + " :" in line for line in mlir.splitlines()), mlir
