## 问题

A3 index-form `pto.tgather` 的输入是合法的单行连续indices子视图，原IR中
`cols=32, valid_cols=32`。`PTOResolveBufferSelect` 将其物理列数恢复为母Tile的
64后，触发TGATHER的 `indices valid_shape[1] to equal indices cols` 检查。

独立源码基线 `e5c1c432` 和其上增加Scalar compare gather支持的本地候选都复现；
本复现只用index-form，不依赖PyPTO或compare扩展，也不运行未初始化Tile。

## 最小复现

```mlir
module {
  func.func @gather_prefix_view() {
    %c0 = arith.constant 0 : index
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %indices = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %tmp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %short = pto.subview %indices[%c0, %c0] sizes [1, 32] : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tgather ins(%src, %short, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=64, v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
```

```bash
ptoas --pto-arch=a3 --pto-level=level2 --mlir-print-ir-after-all gather_prefix_view.pto -o out.cpp
```

输入验证通过，到 `PTOResolveBufferSelect` 后失败；失败IR的indices类型变为
`!pto.tile_buf<vec, 1x64xi32, valid=1x32>`。

## 期望与建议

对于本例的单行、连续、对齐子窗口，保留可供TGATHER消费的32列视图，或者
在合法化时自动物化32列索引；不要通过放宽TGATHER检查让硬件访问不合法布局。
多行、非连续、动态有效范围仍需维持真实行跨度，不能简单把所有子视图缩窄。

本地算子暂时尝试用整数加零物化短索引，CPU lowering通过，但会新增一次Vector
操作。包含该方式的完整探针曾另遇布尔表达式代码生成问题，见#1648。修复后，
59组Top-K用例的三次变输入图回放逐bit通过；这只验证当前物化方式，不等于
直接indices子视图已得到支持，也不作为所有形状的通用设备保证。

建议增加子视图直接作为indices的正例与非法布局反例，检查生成类型、实际地址
及同步范围，而不是只给本测试输入增加特殊规则。这里只提issue与修改建议，不提PR。
