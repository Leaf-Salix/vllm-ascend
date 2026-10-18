# SPDX-License-Identifier: Apache-2.0
"""Print the scalar lowering of x % 24; inspect whether negative x is corrected."""
from pypto import DataType, backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.op import tensor

backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)
span = ir.Span.unknown()
dtype = DataType.INT32
out_type = ir.TensorType([1], dtype)
out = ir.Var("out", out_type, span)
x = ir.Var("x", ir.ScalarType(dtype), span)
result = ir.Var("result", ir.ScalarType(dtype), span)
written = ir.Var("written", out_type, span)
body = ir.SeqStmts([
    ir.AssignStmt(result, ir.FloorMod(x, ir.ConstInt(24, dtype, span), dtype, span), span),
    ir.AssignStmt(written, tensor.write(out, [0], result), span),
    ir.ReturnStmt([written], span),
], span)
function = ir.Function("floor_mod", [out, x], [out_type], body, span, ir.FunctionType.InCore)
print(codegen.PTOCodegen().generate(ir.Program([function], "repro", span)))
