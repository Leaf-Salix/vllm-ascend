### Component

Other (please specify in description)

### Description

MaterializeSemanticAliases / 内存元数据传播：原地结果被重新绑定到实际目标 MemRef 后，紧随其后的 transpose_view、reshape 或 slice 仍可能保留旧分配/旧窗口。元数据 view 没有数据搬运，后续消费者却据此访问另一个未写入的窗口。这是必须别名关系丢失，不是可选的激进内存复用。

### Steps to Reproduce

在已构建的 PyPTO 环境中将下面内容保存为 `view_alias_repro.py`，执行 `python view_alias_repro.py`。这是 CPU IR/codegen 复现，不需要 NPU。

```python
# SPDX-License-Identifier: Apache-2.0
import pypto.language as pl
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.pypto_core import passes

backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)

def _collect_named_tile_memrefs(program: ir.Program, names: set[str]) -> dict[str, ir.MemRef]:
    """Collect MemRefs for named tile definitions anywhere in a function body."""
    result: dict[str, ir.MemRef] = {}
    function = next(iter(program.functions.values()))

    class _MemRefCollector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):  # type: ignore[override]
            tile_type = stmt.var.type
            if stmt.var.name_hint in names and isinstance(tile_type, ir.TileType):
                assert tile_type.memref is not None
                result[stmt.var.name_hint] = tile_type.memref
            super().visit_assign_stmt(stmt)

    _MemRefCollector().visit_stmt(function.body)
    missing = names - result.keys()
    assert not missing, f"tile definitions not found: {sorted(missing)}"
    return result


@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        pool: pl.Tensor[[16, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[8, 64], pl.FP16]],
    ) -> pl.Tensor[[8, 64], pl.FP16]:
        current: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 1024)
        stale: pl.Ptr = pl.tile.alloc(pl.Mem.Mat, 1024)
        seed: pl.Tile[[16, 32], pl.FP16, pl.MemRef(current, 0, 1024), pl.Mem.Mat] = (
            pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        )
        gathered: pl.Tile[[16, 32], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.gather_row(seed, pool, [0, 0], [0, 0], [16, 32])
        )
        transposed: pl.Tile[[32, 16], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.transpose_view(gathered)
        )
        reshaped: pl.Tile[[8, 64], pl.FP16, pl.MemRef(stale, 0, 1024), pl.Mem.Mat] = (
            pl.tile.reshape(transposed, [8, 64])
        )
        return pl.tile.store(reshaped, [0, 0], out)

after = passes.materialize_semantic_aliases()(Before)
refs = _collect_named_tile_memrefs(after, {"seed", "gathered", "transposed", "reshaped"})
for name in ("gathered", "transposed", "reshaped"):
    assert ir.MemRef.same_allocation(refs[name], refs["seed"])
    assert ir.structural_equal(refs[name].byte_offset_, refs["seed"].byte_offset_)
    assert refs[name].size_ == 1024
# A second fixed pass is a no-op; this must not need a fixed-point loop.
ir.assert_structural_equal(passes.materialize_semantic_aliases()(after), after)
```

### Expected Behavior

示例中 gather、transpose_view、reshape 应沿当前 seed 的分配和零偏移窗口传播；再次运行 pass 应幂等。示例主动构造pass输入的陈旧描述符，用于验证该pass已有的重绑定修复是否完整传播，不是推荐用户手写陈旧MemRef。

### Actual Behavior

最新主线的内存 pass 未修改时，`gathered` 已跟随 seed，后续 `transposed`/`reshaped` 却仍保留 stale 分配，`MemRef.same_allocation` 断言失败。类似的同base非零slice窗口会停在1024而不是应有的2048。

在最新主线对应内存实现上，新增view链测试5例中4失败、1通过（不应别名的真实transpose对照通过）。与while遍历问题一起修正后，整个 `test_memory_reuse.py` 的154项通过；旧隔离核心也能复现上面的最小脚本。

### Git Commit ID

基线 `14736f65c04a7db06547edb656f515054950555c`。最新主线负对照使用该基线的原样 `memory_reuse_pass.cpp`；核心另带独立 FloorMod codegen 修正，但本复现只执行 IR pass，不经过 codegen。

### NPU Kind

N/A (not hardware-specific)

### Host Platform

Linux (aarch64)

### Additional Context

根因位置：`src/ir/transforms/memory_reuse_pass.cpp` 的 `NormalizeIdentityCopyBuffersMutator::ReanchorInplaceOutput`。它只修注册的in-place输出；前面的RetypeApplier已经访问过body，同一轮后续view无法获知新重绑定。

建议在同一次正向遍历中同步传播已有的must-alias关系：

1. 使用注册的 `IsBufferAliasingViewOp` 区分纯view与真实数据搬运。真实transpose/extract不能因为此修正与源强行别名。
2. 让view跟随当前输入的base，并正确合成输入byte_offset与view相对offset；保留输出窗口size及slot_count/slot_index。
3. 不能只比较allocation base：同base、不同offset/size/slot同样需要更新。
4. 未经YieldFixup决定存储的if-phi，应把unsettled状态沿view/rename/in-place链传播，避免过早固定到陈旧buffer。
5. 保留幂等、非零窗口、真实transpose、独立extract、未定phi链等回归，不放宽硬件容量或独立活跃值的复用约束。

上述最小脚本的零偏移Mat链来自动态CSA整层编译中的实际现象。非零boxed Mat布局的通用物理offset模型并非本问题已证明覆盖的范围；相关IR窗口检查不能冒充所有设备布局验证。本次提供问题与建议，本地补丁保留，不提修改PR。

### 建议代码 diff（2026-10-18，仅供参考，未提交 PR）

以下补丁针对 PyPTO 主线 `14736f65c04a7db06547edb656f515054950555c` 的
`src/ir/transforms/memory_reuse_pass.cpp`，从本地候选中单独提取view传播修改，
未混入 #3028 的While遍历补丁。

它修复已确定的must-alias关系：跟随当前输入分配、只合成一次view相对偏移、
保留窗口及pipeline slot身份，并把未决phi状态传播到后续别名。真实数据搬运transpose
不应进入这条路径。

验证边界：这组view回归在原实现上4失败/1通过；**与 #3028 补丁合用**的独立构建
已通过整个 `test_memory_reuse.py` 的154项。没有把联合验证写成该拆分diff单独跑过154项。
新增检查涵盖view链、同base非零窗口、未决phi和幂等；非零boxed Mat布局的通用物理偏移
仍需组件侧继续审查，不由这份IR回归替代设备布局验证。

```diff
--- a/src/ir/transforms/memory_reuse_pass.cpp
+++ b/src/ir/transforms/memory_reuse_pass.cpp
@@ -5153,8 +5153,8 @@
   NormalizeIdentityCopyBuffersMutator() = default;
 
   /// `unsettled` seeds the IfStmt return_vars whose buffer YieldFixup has not
-  /// chosen yet; the traversal grows it with their bare-Var renames.  See
-  /// `ReanchorInplaceOutput` for why anchoring onto one is wrong.
+  /// chosen yet; the traversal grows it with their semantic aliases. See
+  /// `ReanchorAliasingOutput` for why anchoring onto one is wrong.
   explicit NormalizeIdentityCopyBuffersMutator(std::set<const Var*> unsettled)
       : unsettled_(std::move(unsettled)) {}
 
@@ -5170,7 +5170,7 @@
       auto new_src = AsVarLike(VisitExpr(op->value_));  // follow prior substitutions
       // A rename of an unsettled value denotes that same storage, so it is just as
       // unsettled: `tmp = phi` then `t = <in-place>(tmp, ...)` would otherwise walk
-      // straight past the guard in ReanchorInplaceOutput and anchor onto the phi's
+      // straight past the guard in ReanchorAliasingOutput and anchor onto the phi's
       // stale buffer through the alias.
       if (IsUnsettled(src_var) || IsUnsettled(new_src)) {
         unsettled_.insert(op->var_.get());
@@ -5190,17 +5190,11 @@
     // Retargeting and pipeline/control-flow lowering can expose a stale output
     // type; normalize it even when the input itself was not substituted during
     // this traversal (the non-divisor K-peel matmul_acc tail is one example).
-    if (auto reanchored = ReanchorInplaceOutput(op)) return reanchored;
+    if (auto reanchored = ReanchorAliasingOutput(op)) return reanchored;
     return IRMutator::VisitStmt_(op);
   }
 
  private:
-  /// Re-anchor an in-place op's output onto its reused input's buffer, preserving
-  /// the output's tile metadata (shape/dtype/space) and swapping only the MemRef.
-  /// This also repairs a producer/output mismatch introduced by control-flow or
-  /// pipeline lowering even when the input itself was not substituted in this
-  /// traversal. Returns nullptr when the allocation already agrees.
-  ///
   /// True when `var` denotes storage whose buffer YieldFixup has not chosen yet —
   /// an IfStmt return_var seeded into `unsettled_`, or a bare-Var rename of one.
   [[nodiscard]] bool IsUnsettled(const VarPtr& var) const {
@@ -5218,24 +5212,51 @@
   /// can lower, while the post-fixup run silently re-anchors the producer back —
   /// leaving the phi buffer unwritten on the taken arm.  Two conditionals over
   /// one in-place tile (what `pl.unroll(2)` around an `if` produces) are enough.
-  StmtPtr ReanchorInplaceOutput(const AssignStmtPtr& op) {
+  /// Reconcile a registered in-place result or metadata view with its current
+  /// input. A gather may be re-anchored here after RetypeApplier has already
+  /// visited the body; its downstream transpose_view/slice must follow in this
+  /// same forward walk. These are required aliases, not new reuse decisions.
+  /// Real data-moving transpose is excluded by IsBufferAliasingViewOp.
+  StmtPtr ReanchorAliasingOutput(const AssignStmtPtr& op) {
     auto call = As<Call>(op->value_);
     if (!call || !call->op_) return nullptr;
     auto reuse_idx = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
-    if (!reuse_idx.has_value()) return nullptr;
-    auto in_var = AsVarLike(call->args_[*reuse_idx]);
+    const bool is_view = op_predicates::IsBufferAliasingViewOp(call->op_->name_);
+    if (!reuse_idx.has_value() && (!is_view || call->args_.empty())) return nullptr;
+    auto in_var = AsVarLike(call->args_[is_view ? 0 : *reuse_idx]);
     if (!in_var) return nullptr;
     auto new_in = AsVarLike(VisitExpr(in_var));  // follow prior subst_ renames
     if (!new_in) return nullptr;
-    if (IsUnsettled(in_var) || IsUnsettled(new_in)) return nullptr;
+    if (IsUnsettled(in_var) || IsUnsettled(new_in)) {
+      // The whole alias chain remains unsettled until YieldFixup chooses the
+      // phi's storage. Otherwise a view would hide that status from a later
+      // in-place writer and pin it to the phi's stale allocation.
+      unsettled_.insert(op->var_.get());
+      return nullptr;
+    }
     auto lhs_tile = GetTileTypeWithMemRef(op->var_->GetType());
     auto in_new_tile = GetTileTypeWithMemRef(new_in->GetType());
-    if (!lhs_tile || !in_new_tile ||
-        SamePhysicalWindow(GetDefinedMemRef(lhs_tile), GetDefinedMemRef(in_new_tile))) {
+    if (!lhs_tile || !in_new_tile) return nullptr;
+    const auto& input_memref = GetDefinedMemRef(in_new_tile);
+    MemRefPtr target = input_memref;
+    if (is_view) {
+      const auto& output_memref = GetDefinedMemRef(lhs_tile);
+      ExprPtr additional = ComputeViewByteOffset(call, new_in->GetType());
+      auto zero = As<ConstInt>(additional);
+      if (!zero || zero->value_ != 0 || output_memref->size_ != input_memref->size_) {
+        // Preserve the view's window and the input's pipeline-slot identity;
+        // neither a subview nor a carried slot denotes the whole allocation.
+        target = std::make_shared<MemRef>(
+            input_memref->base_, AddByteOffsets(input_memref->byte_offset_, additional),
+            output_memref->size_, output_memref->span_, input_memref->is_pinned_,
+            input_memref->slot_count_, input_memref->slot_index_);
+      }
+    }
+    if (SamePhysicalWindow(GetDefinedMemRef(lhs_tile), target) &&
+        lhs_tile->GetMemorySpace() == in_new_tile->GetMemorySpace()) {
       return nullptr;
     }
-    auto new_type = CloneTypeWithMemRef(op->var_->GetType(), GetDefinedMemRef(in_new_tile),
-                                        in_new_tile->GetMemorySpace());
+    auto new_type = CloneTypeWithMemRef(op->var_->GetType(), target, in_new_tile->GetMemorySpace());
     auto new_lhs = std::make_shared<Var>(op->var_->name_hint_, new_type, op->var_->span_);
     subst_[op->var_] = new_lhs;
     auto recursed = As<AssignStmt>(IRMutator::VisitStmt_(op));
@@ -5261,7 +5282,7 @@
 
 /// The pre-YieldFixup half of the identity-copy normalization bracket: identical
 /// to the plain run except that an in-place producer is never anchored onto a phi
-/// whose buffer YieldFixup has yet to choose (`ReanchorInplaceOutput`).
+/// whose buffer YieldFixup has yet to choose (`ReanchorAliasingOutput`).
 StmtPtr NormalizeIdentityCopyBuffersBeforeYieldFixup(const StmtPtr& body) {
   IfPhiReturnVarCollector collector;
   collector.VisitStmt(body);
```
