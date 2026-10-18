### 问题与影响

合法的常量 SSA 起点会以符号表达式保留到 MemRef：`row = 64; tile.slice(pool, ..., [row, 0])` 与直接写 `[64, 0]` 得到不同的内存规划结果。常量别名不应使静态子视图失去可证明的偏移。

在一个完整融合的 A3 CSA kernel 中，早期阶段使用 128 KiB Acc；后续同时需要 96 KiB QK 与 32 KiB WS pool。本地已具备子区间复用能力，但命名常量起点继承到 `matmul_acc` 的结果后被识别为动态偏移，从而触发保护规则，分配变为 160 KiB 并失败。仅把同一偏移表达式直接放进 slice/assemble 就能在 128 KiB 内编译通过。这个完整容量例子使用了本地 arena 改进，不能当作主线完整 CSA 的独立复现。

下面提供不依赖 CSA、模型或 NPU 的主线最小复现，直接检查 InitMemRef 语义。PyPTO main 在本次核对为 `14736f65c04a7db06547edb656f515054950555c`。运行环境是该主线派生的独立 worktree，含另行提交的 FloorMod / view / While 修正，**InitMemRef 文件未改**；复现失败于固定偏移仍为 Mul。修正后的独立编译通过该例及 InitMemRef / L0 跨形状复用共 65 项测试。

### CPU 复现

保存为 `tests/ut/ir/transforms/test_constant_slice_origin_repro.py`，按仓库要求构建并执行 pytest，无需设备：

```python
import pytest
import pypto.language as pl
from pypto import ir, passes

def test_nested_constant_origin_folds_without_freezing_loop_metadata():
    """Constant SSA origins and loop-carried origins need different handling."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def main(
            self,
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            pool = pl.tile.create([128, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            row: pl.Scalar[pl.INDEX] = 16
            for iteration in pl.range(2):
                base: pl.Scalar[pl.INDEX] = 32
                fixed_row = base * 2
                constant_view = pl.tile.slice(pool, [16, 16], [fixed_row, 0])
                carried_view = pl.tile.slice(pool, [16, 16], [row, 0])
                indexed_view = pl.tile.slice(pool, [16, 16], [iteration * 16, 0])
                _first = pl.tile.store(constant_view, [0, 0], output)
                _second = pl.tile.store(carried_view, [0, 0], output)
                _third = pl.tile.store(indexed_view, [0, 0], output)
                row = row + 16
            return output

    after = passes.init_mem_ref()(passes.convert_to_ssa()(Before))
    found = {}

    class Origins(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):
            for name in ("constant_view", "carried_view", "indexed_view"):
                if stmt.var.name_hint.startswith(name):
                    found[name] = stmt.var.type.memref.byte_offset_
            super().visit_assign_stmt(stmt)

    Origins().visit_stmt(next(iter(after.functions.values())).body)
    assert set(found) == {"constant_view", "carried_view", "indexed_view"}
    assert isinstance(found["constant_view"], ir.ConstInt)
    assert found["constant_view"].value == 64 * 16 * 4
    assert not isinstance(found["carried_view"], ir.ConstInt)
    assert not isinstance(found["indexed_view"], ir.ConstInt)

if __name__ == "__main__":
    pytest.main([__file__, "-q"])
```

预期：固定起点偏移为 ConstInt(4096)，循环携带的 row 和 loop index 起点仍为动态表达式。
实际：第一个断言失败，`constant_view` 的偏移是 Mul；没有获得常量地址。

### 修改建议（仅建议代码，不提 PR）

可以在 InitMemRef 的 SSA 前置条件下记录可证明的整数常量，只在 slice 起点使用时折叠，不全局替换任意 Scalar。必须同步改写 slice 的实际参数，否则后续 MemoryReuse 重建 view 时会重新得到符号偏移；同时 NZ 布局判断与偏移计算应看到同一个起点。

不要删除非 slice 动态别名的安全保护，不要假定循环携带值等于初值，也不要放宽硬件容量。下列局部补丁的基线是本地工作树所用 `cb2484471`，供主线维护者参考：

```diff
diff --git a/src/ir/transforms/init_memref.cpp b/src/ir/transforms/init_memref.cpp
index dfae5236e..83449de8b 100644
--- a/src/ir/transforms/init_memref.cpp
+++ b/src/ir/transforms/init_memref.cpp
@@ -29,6 +29,7 @@
 #include "pypto/core/dtype.h"
 #include "pypto/core/error.h"
 #include "pypto/core/logging.h"
+#include "pypto/ir/arith/analyzer.h"
 #include "pypto/ir/cast_saturation.h"
 #include "pypto/ir/core.h"
 #include "pypto/ir/expr.h"
@@ -472,6 +473,36 @@ class InitMemRefMutator : public IRMutator {
                     FunctionType func_type)
       : declared_allocs_(declared_allocs), handler_(handler), func_type_(func_type) {}
 
+  ExprPtr VisitExpr_(const CallPtr& op) override {
+    auto result = IRMutator::VisitExpr_(op);
+    auto call = As<Call>(result);
+    if ((!IsOp(call, "tile.slice") && !IsOp(call, "tensor.slice")) || call->args_.size() < 3) {
+      return result;
+    }
+    auto offsets = As<MakeTuple>(call->args_[2]);
+    if (!offsets) return result;
+
+    // Fold only proven constant slice origins. Simplify deliberately keeps
+    // named scalars inside nested regions, but that spelling must not turn a
+    // static subview into a dynamic MemRef and disable legal arena packing.
+    // Rewrite the actual slice too: MemoryReuse recomputes view offsets from
+    // its arguments, and NZ geometry must see the same origin as InitMemRef.
+    auto elements = offsets->elements_;
+    bool changed = false;
+    for (auto& element : elements) {
+      if (As<ConstInt>(element)) continue;
+      auto folded = scalar_constants_.Simplify(element);
+      if (As<ConstInt>(folded)) {
+        element = folded;
+        changed = true;
+      }
+    }
+    if (!changed) return result;
+    auto new_call = MutableCopy(call);
+    new_call->args_[2] = std::make_shared<MakeTuple>(elements, offsets->span_);
+    return new_call;
+  }
+
   /// Whether `type` is bound to one of this function's declared allocations.
   [[nodiscard]] bool HasUserBinding(const TypePtr& type) const {
     if (declared_allocs_.empty()) return false;
@@ -880,6 +911,10 @@ class InitMemRefMutator : public IRMutator {
 
     // Default case: visit the variable normally
     auto new_var = GetNewVar(op->var_);
+    if (As<ScalarType>(new_var->GetType())) {
+      auto constant = scalar_constants_.Simplify(new_value);
+      if (As<ConstInt>(constant)) scalar_constants_.Bind(new_var, constant);
+    }
     return std::make_shared<AssignStmt>(new_var, new_value, op->span_);
   }
 
@@ -1019,6 +1054,12 @@ class InitMemRefMutator : public IRMutator {
   }
 
   std::map<VarPtr, VarPtr> var_map_;
+  // InitMemRef requires SSAForm: definitions dominate their uses, and branch
+  // results / loop IterArgs have distinct identities. Bind only proven integer
+  // constants, never an IterArg's initial value or a runtime bound. Thus nested
+  // constant bindings cannot freeze loop-varying or branch-selected metadata.
+  // One traversal and constant-size bindings keep this analysis O(N log N).
+  arith::Analyzer scalar_constants_;
   const DeclaredAllocMap& declared_allocs_;
   const backend::BackendHandler* handler_ = nullptr;
   FunctionType func_type_ = FunctionType::InCore;
```

### 验证边界

- 独立工作树 C++ 构建通过。
- 65 项相关 CPU 回归通过，包含真正动态起点不得冻结的反例。
- 此问题解决的是编译和内存规划的等价性；不宣称整层性能提升。合并 WS 写回的算子候选本身已经真机证明性能退步，并未合入正式算子。
