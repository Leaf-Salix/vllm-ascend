## 问题

`FlattenTileNdTo2D` 重建了 `tile.load → tile.read → cast` 产生的 Scalar 后，`tensor.view` 的动态 shape/stride 仍引用旧 Scalar。后续消费者替换类型时会新建一个没有对应定义的 Var，打印为 `alias_1__FREE_VAR`。完整 codegen 随后报告 `Tensor view not found for parameter`。

这个最小复现只使用现有 `tensor.view`，**不使用 #3042 的新 GM 子视图原型**。它暴露的是已有 pass 对非 Tile 操作的定义/使用及类型元数据重写不一致。

## 最小 CPU 复现

```python
import pypto.language as pl
from pypto import passes
from pypto.ir.printer import python_print

@pl.program
class Repro:
    @pl.function(type=pl.FunctionType.InCore)
    def core(self, x: pl.Tensor[[4, 512], pl.BF16],
             indices: pl.Tensor[[1, 16], pl.INT32],
             out: pl.Tensor[[2, 512], pl.BF16]) -> pl.Tensor[[2, 512], pl.BF16]:
        ids = pl.tile.load(indices, [0, 0], [1, 16])
        pitch = pl.cast(pl.tile.read(ids, [0, 0]), pl.INDEX)
        if pitch == 1024:
            alias = pl.tensor.view(x, shape=[2, pitch])
            value = pl.tile.load(alias, [0, 0], [2, 512])
            out = pl.tile.store(value, [0, 0], out)
        return out

before = python_print(Repro)
after = python_print(passes.flatten_tile_nd_to_2d()(Repro))
print("before FREE_VAR:", "__FREE_VAR" in before)
print("after FREE_VAR:", "__FREE_VAR" in after)
print(after)
```

实测输出：`before FREE_VAR: False`，`after FREE_VAR: True`。该例无需 NPU、PTOAS 或新接口；执行期的 `pitch==1024` 保证 `[4,512]→[2,1024]` 元素数相同。

失败 IR 中，`alias` 的赋值存在，load 却引用另一个对象 `alias_1__FREE_VAR`。不能仅用两者显示名称接近来认为是同一个定义。

## 环境与源码核对范围

最小复现在本地 CSA 原父工具链 `d01de8523` 的独立工作区运行，该工作区包含其他已报告的本地修复；**该 pass 的 rewrite.cpp 相对其基点无本地修改**。修复候选在另一个独立工作区构建，没有修改共享安装。

同时只读核对当前 main 的 `src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp`：仍有专门处理 `pld.tile.remote_load` 类型重映射的分支，以及通用路径中“non-tile ops keep the original type”的逻辑。未宣称已经在干净 main 上完整构建验证。

## 原因与修改建议

该 pass 已对 `pld.tile.remote_load` 做过同类处理：一起替换 RHS Call 的类型和定义 Var，再登记 ctx.var_map。建议 `tensor.view`（以及保留 GM 表示的 slice）沿用这个路径，更新类型中的 shape、stride、valid_shape Scalar 引用；不能仅更新 Call 参数后保留旧类型。

局部原型把 GM view/slice 加入这个分支。4 项相关 CPU 回归通过，包括新的定义身份回归和既有 tensor view/partial source view 用例。完整 CSA 双行 DMA 候选仍在编译验证，尚无新真机或性能结论。以下补丁是建议，不是 PR；没有删除地址界检查或把真实 Tile 重新解释成 GM。

```diff
--- a/src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp
+++ b/src/ir/transforms/flatten_tile_nd_to_2d/rewrite.cpp
@@ -905,15 +905,15 @@
 
     const auto& op_name = call->op_->name_;
 
-    // remote_load is a non-"tile." op that nevertheless produces a TileType
-    // whose valid_shape can embed variables remapped by this pass. Rebuild both
-    // the call and its defining Var together; otherwise a later Substitute on
-    // the consumer remaps only the Var's type and creates a definition-less
-    // clone (printed as __FREE_VAR).
-    if (IsOp(call, "pld.tile.remote_load")) {
+    // remote_load 的 valid_shape、GM view/slice 的 shape/stride 都可能引用
+    // 本 pass 重建的 Scalar。必须一起重写 Call 类型和定义 Var，再登记映射；
+    // 只替换参数、保留旧类型会让消费者单独克隆 Var，产生没有定义的引用。
+    // 这是描述符元数据更新，不物化 Tile，也不改变 GM 别名的物理地址。
+    if (IsOp(call, "pld.tile.remote_load") || IsOp(call, "tensor.view") ||
+        IsOp(call, "tensor.slice")) {
       auto new_value = As<Call>(Substitute(assign->value_, ctx.var_map));
       INTERNAL_CHECK_SPAN(new_value, assign->span_)
-          << "FlattenTileNdTo2D: remote_load substitution must remain a Call";
+          << "FlattenTileNdTo2D: descriptor substitution must remain a Call";
       if (new_value.get() == call.get()) {
         result.push_back(stmt);
         ctx.Insert(assign->var_, assign->var_);
--- a/tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py
+++ b/tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py
@@ -33,6 +33,30 @@
 _OP_TILE_MATMUL_ACC = ir.get_op("tile.matmul_acc").name
 _OP_TILE_RESHAPE = ir.get_op("tile.reshape").name
 _OP_TILE_STORE = ir.get_op("tile.store").name
+
+
+def test_runtime_gm_view_keeps_definition_identity():
+    """Tile Scalar 参与 GM shape 时，重写后 load 必须仍引用对应的定义。"""
+    @pl.program
+    class Repro:
+        @pl.function(type=pl.FunctionType.InCore)
+        def core(self, x: pl.Tensor[[4, 512], pl.BF16],
+                 indices: pl.Tensor[[1, 16], pl.INT32],
+                 out: pl.Tensor[[2, 512], pl.BF16]) -> pl.Tensor[[2, 512], pl.BF16]:
+            ids = pl.tile.load(indices, [0, 0], [1, 16])
+            pitch = pl.cast(pl.tile.read(ids, [0, 0]), pl.INDEX)
+            if pitch == 1024:
+                alias = pl.tensor.view(x, shape=[2, pitch])
+                value = pl.tile.load(alias, [0, 0], [2, 512])
+                out = pl.tile.store(value, [0, 0], out)
+            return out
+
+    from pypto.ir.printer import python_print
+
+    # 打印器的 FREE_VAR 标记基于 IR Var 对象定义集合，不只比较变量名。
+    assert "__FREE_VAR" not in python_print(Repro)
+    result = passes.flatten_tile_nd_to_2d()(Repro)
+    assert "__FREE_VAR" not in python_print(result)
 
 # ----------------------------------------------------------------------------
 # Helpers
```

这个修复也应遵守别名和内存元数据合同：类型表达式替换不能改变实际 base/offset/stride、不能生成没有支配定义的新 Var、不能绕过既有真实 Tile→GM view 的拒绝。建议上游补充直接根 view、动态 shape/stride、分支/循环内 view 链及下游 load 的定义一致性回归。
