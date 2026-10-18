### 问题

显式声明了 `pl.Tile[[8,64], FP32, Vec]` 的 `@pl.jit.inline` 参数在特化时被丢弃注解，随后 parser 报 `Parameter 'tile' missing type annotation`。这阻止把单 SPMD 中的片上数学状态拆成可读的小 helper；底层 `pl.function` 能表达 Tile，不代表 JIT 对应入口已保留该类型。

本次核对的 PyPTO main 为 `14736f65c04a7db06547edb656f515054950555c`。CPU 复现运行于该主线派生的独立工作树（已有 FloorMod、view 与 While 的另案补丁）；本问题涉及的 JIT 文件未改。修复侧来自独立 CSA 工作树，未修改共享安装。

### CPU 最小复现

```python
# SPDX-License-Identifier: Apache-2.0
"""使用与正式 CSA 一样的模块级 pl 绑定，保留动态 Tensor 轴。"""

import pypto.language as pl

DYNAMIC_ROWS = pl.dynamic('TILE_PARAM_DYNAMIC_ROWS')


@pl.jit.inline(auto_scope=False)
def keep_tile(tile: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec]) -> pl.Tile:
    return tile


@pl.jit.incore
def tile_kernel(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    rows = pl.min(pl.tensor.dim(source, 0), 8)
    tile = pl.load(source, [0, 0], [8, 64], valid_shape=[rows, 64])
    same_tile = keep_tile(tile)
    pl.store(same_tile, [0, 0], output)
    return output


@pl.jit
def tile_entry(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    return tile_kernel(source, output)

if __name__ == "__main__":
    tile_entry.specialize()
```

实际：`keep_tile` 的源码已有完整 Tile 注解，但生成的 preparse 函数参数变成裸名字，specialize 失败。修正后应保留数学 shape、dtype、memory space 以及调用点有效范围，不新增 Tile 整体复制。

### 修改建议

在 specializer 的非 Tensor 参数注解保留路径中，复用已有 Array 参数处理：仅对 inline 的 `_LangTile` 保留原注解 AST，并使用已有 `_fold_const_names` 展开模块/闭包常量。

```diff
diff --git a/python/pypto/jit/specializer.py b/python/pypto/jit/specializer.py
index a7aa5c2c8..69ff6e690 100644
--- a/python/pypto/jit/specializer.py
+++ b/python/pypto/jit/specializer.py
@@ -49,6 +49,7 @@ from typing import Any, cast
 from pypto._external_source import EXTERNAL_INCLUDE_DIRS_ATTR, encode_external_include_dirs
 from pypto._function_attrs import DUAL_AIV_DISPATCH_ATTR
 from pypto.language.typing.array import Array as _LangArray
+from pypto.language.typing.tile import Tile as _LangTile
 from pypto.pypto_core import DataType
 from pypto.pypto_core.ir import TensorLayout
 
@@ -2098,6 +2099,11 @@ class Specializer:
                 continue
             if isinstance(ann_obj, _LangArray) and ann_obj.extent is not None and ann_obj.dtype is not None:
                 array_param_anns[arg.arg] = f"pl.Array[{ann_obj.extent}, {_array_dtype_str(ann_obj.dtype)}]"
+            elif is_inline and isinstance(ann_obj, _LangTile):
+                # Inline Tile parameters already have a complete type. Keep its
+                # shape, dtype and memory space instead of emitting a bare name;
+                # module/closure constants use the existing annotation folder.
+                array_param_anns[arg.arg] = ast.unparse(_fold_const_names(arg.annotation, ctx.py_globals))
 
         params = self._build_params(
             all_param_names,
```

### 验证及未覆盖边界

- 上述入口本次在主线派生环境失败，局部修正环境 specialize 通过。
- 已有独立 CPU lowering/codegen/PTOAS 证据：固定 Vec Tile identity、full-valid roots/segments 的 assemble→tuple→loop carry 通过；动态 Tensor T 仍为运行时参数，identity 无新增 TMOV。
- 不宣称任意 TileView/MemRef 注解、全部内存空间和动态 valid tuple carry 均支持；动态 valid 与 helper 的固定 valid 签名不一致时应继续类型检查。
- 同一个 helper 的 constexpr 参数用于 Tile shape 仍有独立未支持边界：如下 `ROWS` 在注解求值时 NameError。本地六行修复没有解决它，建议在该入口的类型特化中一并跟踪，先把 constexpr 环境绑定再求值注解；不要把未解析参数变成固定的样例 shape。

```python
# SPDX-License-Identifier: Apache-2.0
"""延迟注解仅用于暴露同一 helper constexpr 名字的明确未支持边界。"""

from __future__ import annotations

import pypto.language as pl

DYNAMIC_ROWS = pl.dynamic('TILE_CONSTEXPR_DYNAMIC_ROWS')


@pl.jit.inline(auto_scope=False)
def keep_tile(
    tile: pl.Tile[[ROWS, 64], pl.FP32, pl.MemorySpace.Vec],
    ROWS: pl.constexpr,
) -> pl.Tile:
    return tile


@pl.jit.incore
def tile_kernel(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    tile = pl.load(source, [0, 0], [8, 64], valid_shape=[pl.min(pl.tensor.dim(source, 0), 8), 64])
    same_tile = keep_tile(tile, 8)
    pl.store(same_tile, [0, 0], output)
    return output


@pl.jit
def tile_entry(
    source: pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32],
    output: pl.Out[pl.Tensor[[DYNAMIC_ROWS, 64], pl.FP32]],
):
    return tile_kernel(source, output)

if __name__ == "__main__":
    tile_entry.specialize()
```

仅提交 issue 和建议，不创建修改 PR；没有为这份报告增加 NPU 测试。
