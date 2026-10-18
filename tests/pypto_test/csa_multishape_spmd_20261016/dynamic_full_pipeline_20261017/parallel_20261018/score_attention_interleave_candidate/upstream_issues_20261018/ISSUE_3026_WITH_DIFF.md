### Component

Codegen

### Description

有符号整数和 INDEX 的 `FloorMod` 直接生成 `arith.remsi`，负数输入时与 IR 及算术简化器的 floor 余数语义不同。比如 `-97 % 24` 应为 23，却得到 -1。它会让核分工表达式 `begin + (worker - begin) % 24` 落到本轮起点之前，导致生产者/消费者工作次数不一致。

### Steps to Reproduce

在已构建的 PyPTO 环境中将下面内容保存为 `floor_mod_repro.py`，执行 `python floor_mod_repro.py`。这是 CPU IR/codegen 复现，不需要 NPU。

```python
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
```

### Expected Behavior

`FloorMod(a,b)` 的非零结果与除数同号，与 Python `%` 一致。生成代码需要保留这个语义；`x=-97`、除数24的实际数值应为23。

### Actual Behavior

未修正的输出核心只有：

```mlir
%0 = arith.remsi %arg1, %c24_i32 : i32
pto.store %0, %arg0[%c0_index] : !pto.ptr<i32>, i32
```

`arith.remsi` 是截断除法余数，结果跟随被除数符号。使用官方 PTOAS 0.66/0.67 编译上述类型的代码，并在 CPU 执行原样生成的 Scalar C++，均能复现错误。10项测试涵盖 INT32/INT64/INDEX、动态/正负常量除数、整数边界以及简化后的 query owner：旧核心10项均失败，建议修正后10项均通过。

不能用 `(x % n + n) % n` 作为可靠源级规避，因为算术简化器按照正确的 floor 语义可以将它简化为 `x % n`。

### Git Commit ID

最新主线 `14736f65c04a7db06547edb656f515054950555c` 的 `PTOCodegen::VisitExpr_(FloorMod)` 仍是上述直接 remsi 实现。负对照在基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c` 的隔离核心执行；两者这一函数相同。另在最新主线建立干净本地分支、独立构建后验证了建议修正。

### NPU Kind

N/A (not hardware-specific)

### Host Platform

Linux (aarch64)

### Additional Context

建议在 `src/codegen/pto/pto_scalar_expr_codegen.cpp` 中对有符号整数及 INDEX 进行语义修正：

```text
r = remsi(a, b)
needs_fix = (r != 0) && ((r < 0) != (b < 0))
result = select(needs_fix, r + b, r)
```

常量正除数只需判断 `r < 0`，常量负除数只需判断 `r > 0`。除数非零及最小负数/-1的前置条件沿用 `ir/arith/int_operator.h`；不要改坏简化器的 floor 定义。其它 dtype、FloorDiv 和 Tensor view 的保守合法性规则可独立处理。

本地建议修复已通过当前主线的176项唯一 codegen 回归：175项首次通过，多缓冲槽位测试由“直接定义必须是rem”改为核对完整SSA依赖链后，剩余1项通过。已额外用官方PTOAS0.67实际编译该多缓冲IR，动态槽位及对应event id仍保留；不能只放开断言而不检查消费者。中英文说明和回归都已在本地保留，本次只提交问题和修改建议，不提交PR。

### 建议代码 diff（2026-10-18，仅供参考，未提交 PR）

以下补丁针对 PyPTO 主线 `14736f65c04a7db06547edb656f515054950555c` 的
`src/codegen/pto/pto_scalar_expr_codegen.cpp`，只修正 signed integer / INDEX 的
FloorMod lowering，不改变 FloorDiv、浮点余数或 unsigned 路径。

本地已独立构建；INT32、INT64、INDEX 与正负常量/动态除数的10项数值回归通过。
完整相关 codegen 验证是首次175项通过，加上修正旧多缓冲测试的SSA链断言后剩余1项通过，
合计176项唯一用例；不是一次全绿运行。下面仅列核心修改，回归应同时保留：
负数/零余数/正负除数、简化后的query owner，以及多缓冲slot仍依赖逻辑余数而非字节偏移。
除数非零、最小负数除以-1的已有前置条件仍适用。

```diff
diff --git a/src/codegen/pto/pto_scalar_expr_codegen.cpp b/src/codegen/pto/pto_scalar_expr_codegen.cpp
index 5b37addc6..15f39ba95 100644
--- a/src/codegen/pto/pto_scalar_expr_codegen.cpp
+++ b/src/codegen/pto/pto_scalar_expr_codegen.cpp
@@ -196,7 +196,39 @@ void PTOCodegen::VisitExpr_(const ir::FloorDivPtr& op) {
   VisitBinaryArithExpr(op, "arith.divsi", "arith.divf");
 }
 void PTOCodegen::VisitExpr_(const ir::FloorModPtr& op) {
-  VisitBinaryArithExpr(op, "arith.remsi", "arith.remf");
+  const auto dtype = ir::GetScalarDtype(op);
+  if (dtype != DataType::INDEX && !dtype.IsSignedInt()) {
+    // Preserve other dtype paths; this correction covers signed integer and
+    // index arithmetic, matching the scalar simplifier's floor semantics.
+    VisitBinaryArithExpr(op, "arith.remsi", "arith.remf");
+    return;
+  }
+
+  const std::string type = dtype == DataType::INDEX ? "index" : GetTypeString(dtype);
+  const std::string lhs = EmitArithOperand(op->left_, type);
+  const std::string rhs = EmitArithOperand(op->right_, type);
+  const std::string remainder = EmitArithBinaryOp("arith.remsi", lhs, rhs, type);
+
+  // FloorMod and the arithmetic simplifier use the divisor's sign. remsi
+  // instead follows the dividend; for example (1 - 48) % 24 must be 1, not
+  // -23. Correct the nonzero remainder when the signs differ. Constant
+  // divisors, as used by core ownership, need only one sign comparison.
+  const std::string zero = GetOrEmitConstant(static_cast<int64_t>(0), dtype);
+  std::string needs_correction;
+  if (auto divisor = As<ir::ConstInt>(op->right_); divisor && divisor->value_ != 0) {
+    needs_correction = EmitArithCmpi(divisor->value_ > 0 ? "slt" : "sgt", remainder, zero, type);
+  } else {
+    const std::string remainder_negative = EmitArithCmpi("slt", remainder, zero, type);
+    const std::string divisor_negative = EmitArithCmpi("slt", rhs, zero, type);
+    const std::string signs_differ =
+        EmitArithBinaryOp("arith.xori", remainder_negative, divisor_negative, "i1");
+    const std::string nonzero = EmitArithCmpi("ne", remainder, zero, type);
+    needs_correction = EmitArithBinaryOp("arith.andi", nonzero, signs_differ, "i1");
+  }
+  const std::string corrected = EmitArithBinaryOp("arith.addi", remainder, rhs, type);
+  const std::string result = NewTemp();
+  Emit(result + " = arith.select " + needs_correction + ", " + corrected + ", " + remainder + " : " + type);
+  fs_.current_expr_value = result;
 }
 
 // ========================================================================
```
