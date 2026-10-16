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


def _run_pipeline(program: ir.Program) -> ir.Program:
    """Run SSA -> infer-memory -> expand-mixed-kernel.

    InjectGMPipeBuffer is intentionally excluded so this file's Expecteds
    pin only what ExpandMixedKernel produces.
    """
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(program)))

def test_hand_written_group_rebinds_shared_inout_results():
    """A preceding member's writeback must bind to a parameter in the next member."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            return sink, out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            sink = self.cube_side(sink)
            sink, out = self.vec_side(sink, out)
            return sink, out

    after = _run_pipeline(Program)
    cube = after.get_function("cube_side")
    vec = after.get_function("vec_side")
    group = after.get_function("mixed")
    assert cube is not None and vec is not None and group is not None
    for member, indices in ((cube, [1]), (vec, [1, 0])):
        returns = [stmt for stmt in ir.flatten_to_stmts(member.body) if isinstance(stmt, ir.ReturnStmt)]
        assert len(returns) == 1
        expected = ir.ReturnStmt([member.params[i] for i in indices], returns[0].span)
        ir.assert_structural_equal(returns[0], expected)
        assert "FREE_VAR" not in member.as_python()
    calls = [
        stmt.value
        for stmt in ir.flatten_to_stmts(group.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    ]
    assert len(calls) == 2
    assert all(list(call.args) == list(group.params) for call in calls)

def test_hand_written_group_submit_rebinds_tuple_writeback_no_dep():
    """Tuple/Submit lineage preserves both buffer identity and the no_dep position."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            second: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            return second, first

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            second: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            (second, first), _cube_tid = pl.submit(self.cube_side, first, second)
            second, _vec_tid = pl.submit(self.vec_side, pl.no_dep(second))
            return second

    after = _run_pipeline(Program)
    vec = after.get_function("vec_side")
    group = after.get_function("mixed")
    assert vec is not None and group is not None
    returns = [stmt for stmt in ir.flatten_to_stmts(vec.body) if isinstance(stmt, ir.ReturnStmt)]
    assert len(returns) == 1
    ir.assert_structural_equal(returns[0], ir.ReturnStmt([vec.params[1]], returns[0].span))
    submits = [
        stmt.value
        for stmt in ir.flatten_to_stmts(group.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Submit)
    ]
    vec_submit = next(submit for submit in submits if submit.op.name == "vec_side")
    assert list(vec_submit.args) == list(group.params)
    assert vec_submit.attrs["arg_direction_overrides"] == [1]
    assert "FREE_VAR" not in vec.as_python()
