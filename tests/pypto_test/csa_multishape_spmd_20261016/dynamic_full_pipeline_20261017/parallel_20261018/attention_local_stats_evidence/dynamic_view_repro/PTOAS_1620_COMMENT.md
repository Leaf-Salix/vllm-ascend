补充一个不含自动分核、不含exp的32行独立PTO复现：同一动态地址别名问题也遗漏 **MTE2→MTE3**。

本次只是追加问题与建议，没有提交PR或修改标签。

### 环境与观察边界

- 当前核对的PyPTO main为`14736f65c04a7db06547edb656f515054950555c`。
- CPU输入来自其派生独立树`72cadb1e5`（另有FloorMod/view/While修正）；本例涉及的`allocate_memory_addr_pass.cpp`和`pto_ops_datamove.cpp`与该main逐字相同。
- 独立PTOAS报告`ptoas 0.66`，与本issue原报告0.68区分；下面去除机器路径后的完整PTO已直接重编译，现象不依赖PyPTO运行环境。
- **本次这个主线输入仅验证生成代码，没有新增它的NPU pass/fail。** 另一个旧kernel调试分支的动态偏移丢失已用A3复现，但那是不同现象，不能拿其数值失败证明这里的同步缺失。

### 触发

前端操作只是：

```python
pool = pl.load(src, [0, 0], [6, 32])
index = pl.cast(pl.read(row, [0]), pl.INDEX)  # 调用合同0 <= index < 6
selected = pl.tile.slice(pool, [1, 32], [index, 0])
column = pl.reshape(selected, [32, 1])
pl.store(column, [0, 0], out)
```

主线已保留`index*32*4`的动态地址，不能归因为reshape读错第0行。
`TLOAD`写UB `[0,768)`，`TSTORE`读 `[index*128,(index+1)*128)`；全部合法index都有重叠。
但两个操作之间没有set/wait或barrier。核入口/退出的barrier不保护中间这条RAW边。

```bash
ptoas slice.pto -o slice.cpp --enable-insert-sync --pto-level=level3 --pto-arch=a3
ptoas extract.pto -o extract.cpp --enable-insert-sync --pto-level=level3 --pto-arch=a3
```

<details><summary>完整slice.pto，32行</summary>

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @slice_reshape_core(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<i32>, %arg2: !pto.ptr<f32>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0_i64 = arith.constant 0 : i64
  %c6_index = arith.constant 6 : index
  %c32_index = arith.constant 32 : index
  %c1_index = arith.constant 1 : index
  %c0_index = arith.constant 0 : index
  %c4_index = arith.constant 4 : index
  %src__ssa_v0_view = pto.make_tensor_view %arg0, shape = [%c6_index, %c32_index], strides = [%c32_index, %c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>
  %row__ssa_v0_view = pto.make_tensor_view %arg1, shape = [%c1_index], strides = [%c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?xi32>
  %out__ssa_v0_view = pto.make_tensor_view %arg2, shape = [%c32_index, %c1_index], strides = [%c1_index, %c32_index] {layout = #pto.layout<dn>} : !pto.tensor_view<?x?xf32>
  %pool__ssa_v0 = pto.alloc_tile addr = %c0_i64 valid_row = %c6_index valid_col = %c32_index : !pto.tile_buf<loc=vec, dtype=f32, rows=6, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %src__ssa_v0_pview = pto.partition_view %src__ssa_v0_view, offsets = [%c0_index, %c0_index], sizes = [%c6_index, %c32_index] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<6x32xf32>
  pto.tload ins(%src__ssa_v0_pview : !pto.partition_tensor_view<6x32xf32>) outs(%pool__ssa_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=6, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  %t__tile = pto.load %arg1[%c0_index] : !pto.ptr<i32> -> i32
  %0 = arith.index_cast %t__tile : i32 to index
  %1 = arith.muli %0, %c32_index : index
  %2 = arith.muli %1, %c4_index : index
  %3 = arith.addi %c0_index, %2 : index
  %4 = arith.index_cast %3 : index to i64
  %selected__ssa_v0 = pto.alloc_tile addr = %4 valid_row = %c1_index valid_col = %c32_index : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %slice_view = pto.subview %pool__ssa_v0[%0, %c0_index] sizes [1, 32] : !pto.tile_buf<loc=vec, dtype=f32, rows=6, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=1, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %5 = arith.muli %0, %c32_index : index
  %6 = arith.muli %5, %c4_index : index
  %7 = arith.addi %c0_index, %6 : index
  %8 = arith.index_cast %7 : index to i64
  %column__ssa_v0 = pto.alloc_tile addr = %8 valid_row = %c32_index valid_col = %c1_index : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=col_major, slayout=none_box, fractal=512, pad=0>
  %out__ssa_v0_pview = pto.partition_view %out__ssa_v0_view, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x1xf32>
  pto.tstore ins(%column__ssa_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%out__ssa_v0_pview : !pto.partition_tensor_view<32x1xf32>)
  return
  }
}
```
</details>

### 工作对照

只将前端slice改为显式ND `pl.tile.extract(pool,index,0,[1,32],target_memory=pl.MemorySpace.Vec)`，
其余数据与输出相同。生成代码有`MTE2→V`保护TEXTRACT，以及`V→MTE3`保护TSTORE。
这个对照存在真实的128B本地copy，不能当作已经修复了零拷贝别名。

<details><summary>完整extract.pto，24行</summary>

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @extract_reshape_core(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<i32>, %arg2: !pto.ptr<f32>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0_i64 = arith.constant 0 : i64
  %c768_i64 = arith.constant 768 : i64
  %c6_index = arith.constant 6 : index
  %c32_index = arith.constant 32 : index
  %c1_index = arith.constant 1 : index
  %c0_index = arith.constant 0 : index
  %src__ssa_v0_view = pto.make_tensor_view %arg0, shape = [%c6_index, %c32_index], strides = [%c32_index, %c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>
  %row__ssa_v0_view = pto.make_tensor_view %arg1, shape = [%c1_index], strides = [%c1_index] {layout = #pto.layout<nd>} : !pto.tensor_view<?xi32>
  %out__ssa_v0_view = pto.make_tensor_view %arg2, shape = [%c32_index, %c1_index], strides = [%c1_index, %c32_index] {layout = #pto.layout<dn>} : !pto.tensor_view<?x?xf32>
  %pool__ssa_v0 = pto.alloc_tile addr = %c0_i64 valid_row = %c6_index valid_col = %c32_index : !pto.tile_buf<loc=vec, dtype=f32, rows=6, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  %src__ssa_v0_pview = pto.partition_view %src__ssa_v0_view, offsets = [%c0_index, %c0_index], sizes = [%c6_index, %c32_index] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<6x32xf32>
  pto.tload ins(%src__ssa_v0_pview : !pto.partition_tensor_view<6x32xf32>) outs(%pool__ssa_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=6, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  %t__tile = pto.load %arg1[%c0_index] : !pto.ptr<i32> -> i32
  %0 = arith.index_cast %t__tile : i32 to index
  %selected__ssa_v0 = pto.alloc_tile addr = %c768_i64 valid_row = %c1_index valid_col = %c32_index : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  pto.textract ins(%pool__ssa_v0, %0, %c0_index : !pto.tile_buf<loc=vec, dtype=f32, rows=6, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index, index) outs(%selected__ssa_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
  %column__ssa_v0 = pto.alloc_tile addr = %c768_i64 valid_row = %c32_index valid_col = %c1_index : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=col_major, slayout=none_box, fractal=512, pad=0>
  %out__ssa_v0_pview = pto.partition_view %out__ssa_v0_view, offsets = [%c0_index, %c0_index], sizes = [%c32_index, %c1_index] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x1xf32>
  pto.tstore ins(%column__ssa_v0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=col_major, slayout=none_box, fractal=512, pad=0>) outs(%out__ssa_v0_pview : !pto.partition_tensor_view<32x1xf32>)
  return
  }
}
```
</details>

### 建议与实际影响

1. 动态地址alloc_tile与已有buffer可能重叠时，保留基址/符号偏移的别名关系；不能仅因SSA不同就漏掉RAW依赖。无法证明不相交时应保守建立依赖，或明确诊断该输入合同不支持。
2. 回归应覆盖MTE2→MTE3直接搬运，不只覆盖MTE2→V；合法row取0/1/4/5并使用不同行内容，检查真正的数据依赖，而不是固定事件ID。
3. 也可在PyPTO侧保留源view的SSA关系；但部分窗口reshape要满足PTOAS物理窗口/布局合同，不能只强行改成treshape就认为已正确。
4. 当前算子采用显式extract作为局部规避，既保留依赖又避免动态slice→reshape在旧分支中丢偏移。未关闭内存规划的安全检查，也未放宽精度阈值。

实际应用是在完整动态CSA的同一AIV内，将每块softmax的mi/li保存在768B的三槽UB环中，
以后续动态slot取出。旧调试分支因为非slice动态别名的下游地址风险而关闭整个Vec空间的细分复用，
导致255104B超限；显式extract后实际布局为163872B。完整CSA两档已精确/图通过，
但这只是算子规避版本的证据，**不是对本issue修复或主线动态别名设备正确性的验证**。
