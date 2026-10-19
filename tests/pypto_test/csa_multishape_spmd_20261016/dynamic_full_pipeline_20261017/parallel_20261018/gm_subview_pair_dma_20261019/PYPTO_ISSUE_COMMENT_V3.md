补充一个已在独立工作区验证的实现建议。**仅作为 issue 中的原型与参考补丁，不提交 PR，也未替换共享安装。**

选择显式 `pl.tensor.slice(..., gm_view=True)`，默认 `False` 的现有 slice 语义不变。首版限制为 rank-2、连续 ND、完整行窗口，拒绝列偏移、缩列、drop_dims、clamp、sub-byte 和真正计算得到的 Tile。

```python
# a/b、rows 都是 device Scalar；先检查整段 span 与 DMA gap。
lo = pl.min(a, b)
delta = pl.max(a, b) - lo
if a >= 0 and b >= 0 and a < rows and b < rows and delta > 0 and delta <= 4194304 and lo + 2 * delta <= rows:
    span = pl.tensor.slice(cache, [2 * delta, 512], [lo, 0], gm_view=True)
    pitched = pl.tensor.view(span, shape=[2, delta * 512])
    pair = pl.load(pitched, [0, 0], [2, 512])
    # 下降次序需要在 UB 恢复；不能改变下游候选次序。
```

实现涉及：

1. Python language/IR wrapper 的显式 bool 参数与 C++ op 属性登记。
2. 类型推导验证连续整行合同，并为新 TensorView 设置与新 shape 对应的显式 stride。仅保留 offset 而漏掉 stride，会在 codegen 阶段失败，原型已修正。
3. ConvertTensorToTileOps 在数据流分析、转换与 IncoreTileOps 验证三个位置一致识别该描述符模式；仍拒绝真正的 Tile 输入。仅修改 mutator 不足以通过完整 pass verifier，原型已补齐。
4. 复用现有 GM view codegen；通过元素单位 `pto.addptr` 推进一次基址，再建立 tensor view。后继 view 沿用这个基址，不能重复推进或丢失偏移。
5. 对齐英文/中文 pass 文档及相关单测。

验证范围：

- 在独立本地 PyPTO 工作区重建，继承本 issue 原复现所用的 CSA 编译环境及其既有本地修复；**不是干净 main 的验证结论**。
- slice/view/新 gm_view 相关 CPU 单测 **52 passed**。普通根 view、显式 GM slice→view→load 的完整 CPU 编译均成功。
- A3 真机 **26 组**搬运地址检查通过：动态 rows=17/97，正序、逆序、重复、尾界、完整 span 不成立时的回退、负索引/越界/INT32 极值；同址 graph 逐轮更新索引，BF16 用 INT16 比较全部位型，输入只读和输出保护区均通过。
- 这里只证明接口原型与最小搬运正确性；没有将其当作完整 CSA 已验证或性能加速的证据。完整 CSA 的收益另行验证。

建议上游可选择更通用的 GM subview API；这里的窄范围实现是为了让 offset/stride/有效域与默认 slice 语义可审核。动态边界的执行期检查仍由调用方保证，不能用伪造 shape 扩大 allocation。

<details>
<summary>本地原型相对继承基线的最小建议补丁（不含已有 CSA 修复）</summary>

```diff
--- a/python/pypto/ir/op/tensor_ops.py
+++ b/python/pypto/ir/op/tensor_ops.py
@@ -288,6 +288,8 @@
     pad_value: PadValue | int | float | None = None,
     clamp: bool = False,
     span: Span | None = None,
+    *,
+    gm_view: bool = False,
 ) -> Call:
     """Create a slice of a tensor with new shape and offset.
 
@@ -320,6 +322,10 @@
             instead. Use it for a fixed-width tail read whose overhang is never
             addressed.
         span: Optional source span for debugging (auto-captured if not provided)
+        gm_view: Preserve a zero-copy GM descriptor inside InCore. The initial
+            implementation accepts contiguous rank-2 ND full-row windows only;
+            it rejects clamp, rank reduction, and sub-byte dtypes. Use an explicit
+            tile.load to move the selected data. False keeps normal slice lowering.
 
     Returns:
         Call expression creating a tensor slice
@@ -350,6 +356,10 @@
         kwargs["pad_value"] = pad_value if pad_value is PadValue.null else normalize_pad_value(pad_value)
     if clamp:
         kwargs["clamp"] = True
+    if type(gm_view) is not bool:
+        raise TypeError("tensor.slice gm_view must be a compile-time bool")
+    if gm_view:
+        kwargs["gm_view"] = True
 
     return _ir_core.create_op_call("tensor.slice", args, kwargs, actual_span)
 
--- a/python/pypto/language/op/tensor_ops.py
+++ b/python/pypto/language/op/tensor_ops.py
@@ -482,6 +482,8 @@
     drop_dims: Sequence[int | Expr] | None = None,
     pad_value: PadValue | int | float | None = None,
     clamp: bool = False,
+    *,
+    gm_view: bool = False,
 ) -> _TensorT:
     """Create a slice of a tensor with new shape and optional valid shape.
 
@@ -512,6 +514,11 @@
             rejected when that provably fails; ``clamp=True`` lets the window
             overhang and cuts the valid region back to the source edge. Use it
             for a fixed-width tail read whose overhang is never addressed.
+        gm_view: Keep a zero-copy GM descriptor in InCore, without loading the
+            window into a Tile. Initially limited to contiguous rank-2 ND
+            full-row windows, without clamp/rank reduction/sub-byte elements.
+            Follow it with tensor.view and an explicit load as needed. False
+            preserves the existing implicit Tile-load behavior.
 
     Returns:
         Tensor wrapping the slice operation
@@ -550,6 +557,7 @@
         drop_dims,
         pad_value=pad_value,
         clamp=clamp,
+        gm_view=gm_view,
     )
     return tensor.__class__(expr=call_expr)
 
--- a/src/ir/op/tensor_ops/memory.cpp
+++ b/src/ir/op/tensor_ops/memory.cpp
@@ -34,6 +34,7 @@
 #include "pypto/ir/kind_traits.h"
 #include "pypto/ir/op_registry.h"
 #include "pypto/ir/scalar_expr.h"
+#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
 #include "pypto/ir/type.h"
 #include "pypto/ir/type_inference.h"
 
@@ -257,6 +258,29 @@
                                      GetKwargOr<bool>(kwargs, "clamp", false), args[0]->span_);
   ValidateDropDimsValidExtents(drop_dims, full_valid, "tensor.slice", args[0]->span_);
   const std::vector<ExprPtr> valid_shape = ApplyDropDims(full_valid, drop_dims);
+
+  if (GetKwargOr<bool>(kwargs, "gm_view", false)) {
+    // 显式GM窗口不物化Tile。首版只支持连续ND整行范围，禁止把列切片的
+    // 原始pitch误当作新shape的dense stride；动态行范围仍遵守上方有效域合同。
+    CHECK_SPAN(tensor_type->shape_.size() == 2 && full_shape.size() == 2 &&
+                   offsets.size() == 2 && drop_dims.empty(), args[0]->span_)
+        << "tensor.slice gm_view requires a rank-2 full-row window without drop_dims";
+    auto col_offset = As<ConstInt>(offsets[1]);
+    CHECK_SPAN(col_offset && col_offset->value_ == 0 &&
+                   AreExprVectorsEqual({full_shape[1]}, {tensor_type->shape_[1]}), args[0]->span_)
+        << "tensor.slice gm_view requires zero column offset and the full source width";
+    CHECK_SPAN(!GetKwargOr<bool>(kwargs, "clamp", false) && tensor_type->dtype_.GetBit() >= 8,
+               args[0]->span_)
+        << "tensor.slice gm_view does not support clamp or sub-byte elements";
+    if (tensor_type->tensor_view_) {
+      const auto& view = *tensor_type->tensor_view_;
+      CHECK_SPAN(view.layout == TensorLayout::ND &&
+                     (view.stride.empty() || AreExprVectorsEqual(
+                         view.stride, tensor_view_semantics::BuildRowMajorStrides(tensor_type->shape_))),
+                 args[0]->span_)
+          << "tensor.slice gm_view requires contiguous ND source strides";
+    }
+  }
 
   // A read view over padded bytes has to keep saying they are padded, so the
   // source pad mode carries through unless the caller overrides it. The
@@ -296,9 +320,13 @@
   const TensorLayout result_layout = trailing_plane_intact ? source_layout : TensorLayout::ND;
 
   std::optional<TensorView> result_tv;
-  if (result_layout != TensorLayout::ND || !AreExprVectorsEqual(valid_shape, new_shape) ||
+  const bool gm_view = GetKwargOr<bool>(kwargs, "gm_view", false);
+  if (gm_view || result_layout != TensorLayout::ND || !AreExprVectorsEqual(valid_shape, new_shape) ||
       result_pad != PadValue::null) {
-    result_tv = TensorView({}, result_layout, valid_shape, result_pad);
+    // 保留在InCore内的GM描述符需要显式stride；普通slice仍由原Tile路径
+    // 建立布局，不改变其类型推导。整行连续条件已经在上方证明。
+    result_tv = TensorView(gm_view ? tensor_view_semantics::BuildRowMajorStrides(new_shape)
+                                  : std::vector<ExprPtr>{}, result_layout, valid_shape, result_pad);
   }
   if (auto dt = As<DistributedTensorType>(args[0]->GetType())) {
     return std::make_shared<DistributedTensorType>(new_shape, tensor_type->dtype_, std::nullopt,
@@ -584,6 +612,7 @@
     .set_output_memory_inherit_input()
     .set_attr<PadValue>("pad_value")
     .set_attr<bool>("clamp")
+    .set_attr<bool>("gm_view")
     .f_deduce_type([](const std::vector<ExprPtr>& args,
                       const std::vector<std::pair<std::string, std::any>>& kwargs) {
       return DeduceTensorSliceType(args, kwargs);
--- a/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp
+++ b/src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp
@@ -248,6 +248,10 @@
   return IsOp(call, "tensor.dim") || IsOp(call, "tensor.view");
 }
 
+bool IsExplicitGMSubview(const CallPtr& call) {
+  return IsOp(call, "tensor.slice") && GetKwargOr<bool>(call->kwargs_, "gm_view", false);
+}
+
 /// Declared GM cache policy per source tensor, keyed by the param Var the
 /// declaration resolved to. ``CachePolicy`` stored as ``int``, the type the
 /// ``tile.load`` ``cache`` kwarg is registered with.
@@ -779,7 +783,7 @@
       if (AsVarLike(op->value_) || As<TupleGetItemExpr>(op->value_)) {
         AddFlow(op->value_, op->var_);
       } else if (auto call = As<Call>(op->value_); call && registry_.Lookup(call->op_->name_)) {
-        if (IsOp(call, "tensor.write") || IsOp(call, "tensor.assemble")) {
+        if (IsOp(call, "tensor.write") || IsOp(call, "tensor.assemble") || IsExplicitGMSubview(call)) {
           AddFlow(call->args_[0], op->var_);
         } else if (IsOp(call, "tensor.expand_clone")) {
           AddFlow(call->args_[1], op->var_);
@@ -1159,6 +1163,13 @@
       return HandlePassThroughAssign(op, new_value);
     }
 
+    if (IsExplicitGMSubview(call)) {
+      // 此模式是GM描述符，不是隐式load。计算结果若已转为Tile仍必须拒绝，
+      // 不可把UB地址重新解释为GM；普通slice继续走原转换器。
+      CHECK_SPAN(!call->args_.empty() && AsTensorTypeLike(call->args_[0]->GetType()), call->span_)
+          << "tensor.slice gm_view requires a GM Tensor input, not a computed Tile";
+      return HandlePassThroughAssign(op, new_value);
+    }
     const auto* entry = conv_registry_.Lookup(call->op_->name_);
     if (!entry) {
       if (IsOp(call, "tensor.view")) {
@@ -3427,7 +3438,10 @@
       // ``AsTensorTypeLike`` also whitelists ``DistributedTensorType``, which the
       // conversion registry above keeps as ``tensor.read`` / ``tensor.write`` so the
       // PTO codegen can lower it as a local-rank ``pto.load`` / ``pto.store``.
-      if ((IsOp(call, "tensor.read") || IsOp(call, "tensor.write")) && !call->args_.empty() &&
+      // 显式GM子视图与GM标量访存一样可以保留；普通slice仍必须转换，
+      // 且这里仍检查源为Tensor，不能用属性放行真正的Tile。
+      if ((IsOp(call, "tensor.read") || IsOp(call, "tensor.write") || IsExplicitGMSubview(call)) &&
+          !call->args_.empty() &&
           AsTensorTypeLike(call->args_[0]->GetType())) {
         return;
       }
--- a/src/backend/common/pto_ops_memory.cpp
+++ b/src/backend/common/pto_ops_memory.cpp
@@ -986,10 +986,12 @@
   // the ``RegisterTensorView`` call below. The LHS also aliases the input base
   // pointer and CommContext so later tensor or distributed ops address the
   // original storage.
-  reg("tensor.view", [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
+  auto emit_gm_view = [](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
     auto& codegen = AsPto(codegen_base);
-    INTERNAL_CHECK_SPAN(op->args_.size() >= 1 && op->args_.size() <= 3, op->span_)
-        << "tensor.view requires 1 to 3 args (input[, shape[, valid_shape]])";
+    const bool subview = ir::IsOp(op, "tensor.slice");
+    INTERNAL_CHECK_SPAN(subview ? (op->args_.size() >= 3 && op->args_.size() <= 5)
+                               : (op->args_.size() >= 1 && op->args_.size() <= 3), op->span_)
+        << "Internal error: invalid GM view argument count";
     auto input_var = AsVarLike(op->args_[0]);
     INTERNAL_CHECK_SPAN(input_var, op->span_) << "tensor.view input must be a Var/IterArg";
 
@@ -1012,13 +1014,26 @@
 
     std::string result_buf = codegen.GetCurrentResultTarget();
     INTERNAL_CHECK_SPAN(!result_buf.empty(), op->span_) << "Internal error: result buf must be set";
+    auto input_type = ir::AsTensorTypeLike(input_var->GetType());
+    INTERNAL_CHECK_SPAN(input_type, op->span_)
+        << "Internal error: GM view source must remain a Tensor";
     std::string input_base_ptr = codegen.GetTensorBasePtr(input_var);
+    if (subview) {
+      INTERNAL_CHECK_SPAN(ir::GetKwargOr<bool>(op->kwargs_, "gm_view", false), op->span_)
+          << "Internal error: only explicit GM slices may survive TensorToTile";
+      auto offsets = As<ir::MakeTuple>(op->args_[2]);
+      INTERNAL_CHECK_SPAN(offsets, op->span_) << "Internal error: GM slice offsets must be a tuple";
+      // 类型推导已证明连续ND整行切片；偏移以元素为单位。将真实base推进
+      // 一次，再共用下方make_tensor_view，后续view不能丢失或重复该偏移。
+      const std::string offset = GetFlatOffsetSSA(offsets, input_type->shape_, codegen);
+      const std::string shifted = codegen.NewTemp();
+      const std::string ptr_type = "!pto.ptr<" + codegen.GetTypeString(input_type->dtype_) + ">";
+      codegen.Emit(shifted + " = pto.addptr " + input_base_ptr + ", " + offset + " : " +
+                   ptr_type + " -> " + ptr_type);
+      input_base_ptr = shifted;
+    }
     codegen.RegisterBasePtr(lhs_var, input_base_ptr);
     codegen.RegisterCommCtxFor(lhs_var, codegen.GetCommCtxSSAFor(input_var.get()));
-
-    auto input_type = ir::AsTensorTypeLike(input_var->GetType());
-    INTERNAL_CHECK_SPAN(input_type, op->span_)
-        << "tensor.view input must be TensorType or DistributedTensorType";
 
     // The result SSA name (auto-allocated by VisitStmt_(AssignStmt) for the
     // backend-dispatched RHS Call) doubles as the tensor_view SSA name —
@@ -1077,7 +1092,9 @@
     }
     oss << "x" << codegen.GetTypeString(lhs_type->dtype_) << ">";
     return oss.str();
-  });
+  };
+  reg("tensor.view", emit_gm_view);
+  reg("tensor.slice", emit_gm_view);
 
   reg("tile.load", [](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
     return MakeTileLoadCodegenPTO(op, codegen);
--- a/tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py
+++ b/tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py
@@ -574,6 +574,37 @@
         After = passes.convert_tensor_to_tile_ops()(Before)
         ir.assert_structural_equal(After, Before)
 
+    def test_explicit_gm_view_slice_stays_descriptor_before_load(self):
+        """整行GM子视图不提前搬运大span，后续显式load保持原读取范围。"""
+
+        @pl.program
+        class Before:
+            @pl.function(type=pl.FunctionType.InCore)
+            def kernel(self, x: pl.Tensor[[16, 512], pl.BF16],
+                       out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]) -> pl.Tensor[[2, 512], pl.BF16]:
+                span = pl.tensor.slice(x, [4, 512], [3, 0], gm_view=True)
+                pitched = pl.tensor.view(span, shape=[2, 1024])
+                value = pl.tile.load(pitched, [0, 0], [2, 512])
+                result = pl.tile.store(value, [0, 0], out)
+                return result
+
+        After = passes.convert_tensor_to_tile_ops()(Before)
+        ir.assert_structural_equal(After, Before)
+
+    def test_explicit_gm_view_rejects_computed_tile_source(self):
+        """不能用gm_view把真正的计算Tile伪装成GM地址。"""
+
+        @pl.program
+        class Before:
+            @pl.function(type=pl.FunctionType.InCore)
+            def kernel(self, x: pl.Tensor[[4, 512], pl.BF16]) -> pl.Tensor[[4, 512], pl.BF16]:
+                computed = pl.add(x, x)
+                span = pl.tensor.slice(computed, [4, 512], [0, 0], gm_view=True)
+                return span
+
+        with pytest.raises((ValueError, TypeError), match="GM Tensor|TensorType"):
+            passes.convert_tensor_to_tile_ops()(Before)
+
     def test_tensor_slice_drop_dims_lowers_to_tile(self):
         """Rank reduction preserves source validity and emits a separate reshape.
 
--- a/tests/ut/ir/operators/test_tensor_ops.py
+++ b/tests/ut/ir/operators/test_tensor_ops.py
@@ -60,6 +60,32 @@
         assert isinstance(dim, ir.ConstInt)
         dims.append(dim.value)
     return dims
+
+
+def test_tensor_slice_gm_view_full_rows():
+    """显式GM窗口继承有效域，元素数与真实行span一致。"""
+    x = _tensor_var("x", [16, 512], DataType.BF16)
+    result = tensor.slice(x, [4, 512], [3, 0], gm_view=True)
+    assert _const_shape(result) == [4, 512]
+    assert isinstance(result.type, ir.TensorType)
+
+
+@pytest.mark.parametrize("shape,offset,extra,match", [
+    ([4, 256], [0, 0], {}, "full source width"),
+    ([4, 512], [0, 1], {"clamp": True}, "zero column offset|full source width"),
+    ([4, 512], [0, 0], {"clamp": True}, "does not support clamp"),
+    ([1, 512], [0, 0], {"drop_dims": [0]}, "without drop_dims"),
+])
+def test_tensor_slice_gm_view_rejects_unsupported_window(shape, offset, extra, match):
+    x = _tensor_var("x", [16, 512], DataType.BF16)
+    with pytest.raises(ValueError, match=match):
+        tensor.slice(x, shape, offset, gm_view=True, **extra)
+
+
+def test_tensor_slice_gm_view_requires_bool():
+    x = _tensor_var("x", [16, 512], DataType.BF16)
+    with pytest.raises(TypeError, match="compile-time bool"):
+        tensor.slice(x, [4, 512], [0, 0], gm_view=1)
 
 
 def test_tensor_create():
--- a/tests/ut/codegen/test_pto_codegen_ops.py
+++ b/tests/ut/codegen/test_pto_codegen_ops.py
@@ -795,6 +795,26 @@
 
 class TestB01PrecisionAndRowExpandAddCodegen:
     """Exact PTOAS forms added for the first op batch."""
+
+    def test_explicit_gm_view_subview_advances_pointer_once(self):
+        """非零行偏移必须传到后续view，不额外加载四行中间span。"""
+        @pl.program
+        class Prog:
+            @pl.function(type=pl.FunctionType.InCore)
+            def kernel(self, x: pl.Tensor[[16, 512], pl.BF16],
+                       out: pl.Out[pl.Tensor[[2, 512], pl.BF16]]):
+                span = pl.tensor.slice(x, [4, 512], [3, 0], gm_view=True)
+                pitched = pl.tensor.view(span, shape=[2, 1024])
+                value = pl.load(pitched, [0, 0], [2, 512])
+                pl.store(value, [0, 0], out)
+
+        mlir = self._generate_mlir(Prog)
+        shifted = re.findall(r"(%[\w]+) = pto.addptr", mlir)
+        assert len(shifted) == 1, mlir
+        views = [line for line in mlir.splitlines() if "pto.make_tensor_view " + shifted[0] in line]
+        assert len(views) == 2, mlir
+        assert mlir.count("pto.tload ") == 1, mlir
+        assert "arith.constant 1536 : index" in mlir, mlir
 
     def _generate_mlir(self, program_cls, backend_type=BackendType.Ascend910B) -> str:
         backend.reset_for_testing()
--- a/docs/en/dev/passes/11-convert_tensor_to_tile_ops.md
+++ b/docs/en/dev/passes/11-convert_tensor_to_tile_ops.md
@@ -79,6 +79,22 @@
 loop iterations. Returning another tensor, or returning nothing, does not
 discard the GM write. `InOut` describes argument effects; it does not request an
 unconditional whole-tensor store at function exit.
+
+### Explicit GM row subviews
+
+`pl.tensor.slice(..., gm_view=True)` keeps a GM descriptor in an InCore function;
+normal slices keep their existing implicit Tile-load behavior. This initial mode
+accepts only contiguous rank-2 ND full-row windows (zero column offset and full
+source width), without `drop_dims`, `clamp`, or sub-byte elements. Dynamic row
+origins and extents remain device Scalars and must describe an in-bounds window.
+The source validity is still intersected with the window; no data is moved until
+an explicit `pl.load`. A computed Tile cannot be reinterpreted as GM.
+
+A following `tensor.view` can change the dense shape while preserving the exact
+number of elements. Codegen advances the GM base pointer by the slice origin
+exactly once, then creates a view with the inferred shape and strides. This can
+express two pitched row reads without first loading the intervening span into UB.
+It does not infer arbitrary strides or silently densify a column subview.
 
 ### GM Store Coherence Restriction
 
--- a/docs/zh/dev/passes/11-convert_tensor_to_tile_ops.md
+++ b/docs/zh/dev/passes/11-convert_tensor_to_tile_ops.md
@@ -648,3 +648,15 @@
 | Spmd / Group（转发到转换后 InCore） | 签名镜像 InCore 新增的 `Out` 参数，内部调用转发这些参数（阶段二a） |
 | Spmd / Group（未转发到转换后 InCore） | 不变 |
 | Orchestration / Opaque | 更新调用点 —— 为每个新增 `Out` 参数插入 `tensor.create`（阶段二b） |
+
+## 显式GM整行子视图
+
+`pl.tensor.slice(..., gm_view=True)`在InCore内保留GM描述符；普通slice仍
+使用原有隐式Tile搬运。首版仅支持连续rank-2 ND整行窗口（列偏移0，宽度
+等于源宽度），不支持drop_dims、clamp或不足8位的元素。动态起始行和行数
+仍在device计算，调用者必须保证整个窗口在allocation内；有效域仍与源求交。
+只有显式pl.load才搬运数据，真实计算Tile不能借此重新解释为GM。
+
+后续tensor.view可在元素数守恒下重新解释连续窗口；codegen将真实GM基址
+按行偏移推进一次，再建立形状和步长描述符，因此无需先把间隔span搬入UB。
+该模式不推断任意stride，也不把列切片误当dense布局。
```

</details>
