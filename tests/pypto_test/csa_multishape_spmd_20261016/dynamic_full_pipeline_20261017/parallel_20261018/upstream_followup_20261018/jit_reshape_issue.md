### 问题

`pl.reshape` 的尺寸仅由输入 Tensor 的动态维度和整数算术组成，调用后把结果传给另一 JIT 函数，仍报 `missing inferred tensor metadata`。rank/dtype 与 descriptor 维度应可保留，不必把动态 shape 固化为样例值。

这不同于已关闭的 #1755：直接常量 reshape 已支持，此处缺的是 `pages * width`、`(pages * width - 64) // 128` 等 descriptor-derived 表达式。

本次核对的 PyPTO main 为 `14736f65c04a7db06547edb656f515054950555c`。CPU 复现运行于该主线派生的独立工作树（已有 FloorMod、view 与 While 的另案补丁）；本问题涉及的 JIT 文件未改。修复侧来自独立 CSA 工作树，未修改共享安装。

### CPU 最小复现（不依赖模型/NPU）

保存为 Python 文件并直接运行：

```python
# SPDX-License-Identifier: Apache-2.0
"""JIT 描述符维度算术的 CPU 最小复现；不创建、不运行设备任务。"""

import pypto.language as pl

PAGES = pl.dynamic("REPRO_CACHE_PAGES")


@pl.jit.incore
def copy_key(keys: pl.Tensor, out: pl.Out[pl.Tensor]):
    value = pl.load(keys, [0, 0], [1, 128])
    pl.store(value, [0, 0], out)
    return out


@pl.jit
def entry(
    src: pl.Tensor[[PAGES, 4160], pl.INT8],
    out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
):
    pages = pl.tensor.dim(src, 0)
    width = pl.tensor.dim(src, 1)
    flat = pl.reshape(src, [pages * width])
    shifted = flat[64 : 64 + ((pages * width - 64) // 128) * 128]
    keys = pl.reshape(shifted, [(pages * width - 64) // 128, 128])
    return copy_key(keys, out)


if __name__ == "__main__":
    entry.specialize()
```

实际：`copy_key` 的 `keys` 缺少推导 metadata；`_extract_local_tensor_metas` 在第一层 flatten reshape 就漏掉 `flat`。
预期：保留动态表达式，并把结果视图的实际描述符绑定给依赖函数。

### 原因与建议代码

`_reshape_meta` 调用严格 `_resolve_shape(shape_node)`；DynDim 的算术无法折叠成 Python int，因此整个结果被跳过。可以像既有 `create_tensor` 一样生成 descriptor 绑定的派生 DynDim，但仅接收由已知 shape 值组成的算术。真实 `tensor.read` 值、未知标量、除零等必须继续拒绝；不能用 placeholder 静态上界替代真实维度，也不能把尺寸相等证明从 reshape IR 中删除。

以下是针对本地 `cb2484471` 基线的最小建议代码；仅提 issue，不提 PR：

```diff
diff --git a/python/pypto/jit/decorator.py b/python/pypto/jit/decorator.py
index 2d89d14ef..8fcb8a5bc 100644
--- a/python/pypto/jit/decorator.py
+++ b/python/pypto/jit/decorator.py
@@ -1444,8 +1444,10 @@ def _extract_local_tensor_metas(
     A ``pl.create_tensor`` / ``pld.window`` dim that no static rule resolves —
     a runtime extent such as ``pld.world_size()``, ``pl.tensor.read(cfg, [0])``,
     or arithmetic over a ``DynDim`` — becomes a synthesized ``DynDim`` (see
-    :func:`_synthesized_dyn_dim`) rather than voiding the whole meta. Everything
-    else not statically resolvable is still skipped silently, and the clear
+    :func:`_synthesized_dyn_dim`) rather than voiding the whole meta. Reshape
+    dimensions derived solely from tensor dimensions and integer arithmetic
+    use the same descriptor-bound symbols; their expressions remain in IR.
+    Other unresolved reshape dimensions are still skipped, and the clear
     ``ValueError`` in ``Specializer._build_params`` fires for that variable.
     When ``stop_at_dep`` is provided, extraction stops
     immediately before the first source-ordered call to that dependency. This
@@ -1506,8 +1508,8 @@ def _extract_local_tensor_metas(
             rhs = _resolve_shape_elt(elt.right)
             # Arithmetic over DynDim is not statically resolvable here; we
             # only fold int·int. Anything else (DynDim+int, DynDim*DynDim)
-            # is left unresolved — the parent-dim fallback in _slice_meta /
-            # the silent skip in _create_tensor_meta is the right behaviour.
+            # is left unresolved — the caller can use a parent bound or a
+            # synthesized descriptor dimension, depending on the operation.
             if isinstance(lhs, int) and isinstance(rhs, int):
                 return _fold_int_arith(elt.op, lhs, rhs)
             return None
@@ -1596,11 +1598,18 @@ def _extract_local_tensor_metas(
         )
         if shape_node is None:
             return None
-        # Strict on purpose (no ``target``): a reshape's dims are constrained by
-        # the source's element count, which a freshly invented symbol cannot
-        # express. ``pl.slice`` below is strict for the mirror reason — it
-        # already has a better answer, the parent dim's static bound.
         shape = _resolve_shape(shape_node)
+        if (
+            shape is None
+            and isinstance(shape_node, ast.List)
+            and all(_is_dimension_arithmetic(elt) for elt in shape_node.elts)
+        ):
+            # Rank/dtype discovery need not fold a descriptor-derived extent.
+            # Keep its arithmetic in the reshape IR, and let dependency
+            # parameters bind the resulting dimension from the view's actual
+            # descriptor. This does not assert an element-count equality or
+            # replace a runtime dimension by the placeholder's static bound.
+            shape = _resolve_shape(shape_node, target)
         if shape is None:
             return None
         # A reshape re-groups the dims a layout describes, so the source layout
@@ -1611,6 +1620,18 @@ def _extract_local_tensor_metas(
             return None
         return TensorMeta(shape=shape, dtype=src_meta.dtype)
 
+    def _is_dimension_arithmetic(elt: ast.expr) -> bool:
+        """Accept dimension-derived arithmetic, not arbitrary runtime values."""
+        if _resolve_shape_elt(elt) is not None:
+            return True
+        if isinstance(elt, ast.BinOp) and isinstance(elt.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
+            if isinstance(elt.op, ast.FloorDiv) and _resolve_shape_elt(elt.right) == 0:
+                return False
+            return _is_dimension_arithmetic(elt.left) and _is_dimension_arithmetic(elt.right)
+        if isinstance(elt, ast.UnaryOp) and isinstance(elt.op, (ast.UAdd, ast.USub)):
+            return _is_dimension_arithmetic(elt.operand)
+        return False
+
     def _slice_meta(call: ast.Call, target: str | None = None) -> TensorMeta | None:
         # pl.slice(tensor, shape, offset, ...) — shape is positional index 1 or kw `shape=`.
         src = call.args[0] if call.args else None
```

### 验证

- 上述最小入口：未修改 JIT 的主线派生环境失败，建议修正侧 specialize 通过。
- 局部 metadata 回归：descriptor 算术传递通过，混入未知 `tensor.read` 标量继续不推导。
- 本次相关 JIT/别名定向检查共 9 项通过；不把 specialize 成功称作 NPU 功能或性能通过。
