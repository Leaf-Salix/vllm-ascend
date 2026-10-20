补充一个同类的 **TMOV 写侧子视图** 复现：输入验证通过，`PTOResolveBufferSelect` 将单行子窗口的物理列宽恢复到母 Tile 后，触发 A3 TMOV 的同形状检查。可能需要与本 issue 的 TGATHER 读侧问题一起处理。

本轮发生在 CSA 单 SPMD 的半叶分数留 UB 优化中，PyPTO `tile.assemble(parent_1x4096, src_1x1024, [0, col])` 生成下面的合法子视图。仅做 CPU 编译，未执行未初始化 Tile；运行场景的 col 是 0/1024/2048/3072。

```mlir
module {
  func.func @assemble_column(%offset: index) {
    %c0 = arith.constant 0 : index
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %parent = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4096, v_row=1, v_col=4096, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %view = pto.subview %parent[%c0, %offset] sizes [1, 1024] : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=4096, v_row=1, v_col=4096, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tmov ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%view : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024, v_row=1, v_col=1024, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
```

```bash
ptoas --pto-arch=a3 --pto-level=level2 --mlir-print-ir-after-all column.pto -o out.cpp
```

在本地 `c04ecdf9ca29a26b3b6d026faf18c12df476d201` 上复现；该树在本 issue 已列出的路径之外有成对压缩扩展，未为本例修改 ResolveBufferSelect/TMOV。不声称已测试此刻上游最新 main。

变换前 TMOV 两侧均为 `vec, 1x1024xf32`；失败 pass 后成为：

```mlir
"pto.tmov"(%src, %view) :
  (!pto.tile_buf<vec, 1x1024xf32>,
   !pto.tile_buf<vec, 1x4096xf32, valid=1x1024>) -> ()
```

报错：`expects A2/A3 non-mat tmov to use matching src/dst shapes`。

对照：父 Tile 改为 `4x1024`、使用 `[row,0]` 的 `1x1024` 子视图后，同一工具链编译成功。算子侧暂以四行存储再 reshape 回 `1x4096`，完整 CSA 的符号编译也通过；设备精度与性能尚在验证，不把编译成功写成设备 PASS。

建议把这份写侧回归也加入单行连续 subview 的合法化测试：对单行、连续且对齐窗口，考虑保留偏移后的独立窄 Tile 形状，或在目标指令前做合法窄视图转换；不能仅放宽 TMOV verifier。多行真实行距、动态边界、别名及同步范围应继续保留。当前只是 issue 补充和修改建议，没有 PR。
