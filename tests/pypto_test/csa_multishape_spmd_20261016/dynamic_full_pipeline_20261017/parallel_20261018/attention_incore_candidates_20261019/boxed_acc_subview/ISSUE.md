## 问题：可证明不重叠的L0C列窗口仍被同步分析当作父池整体

这是保守依赖分析造成的性能限制，不是漏同步或当前精度错误。PTOAS 0.66、
0.67均可用下面的小型IR复现，不需要PyPTO、CSA权重、跨核通信或设备执行。

完整动态CSA里，QK score与两份PV累加器已经分别分配到0、32768、65536，
每份64×128 FP32，共96KiB。它们来自同一个64×384 L0C父Tile的三个列窗口。
尽管物理范围不重叠，QK的FIX写回与PV的Cube计算之间仍生成FIX→M等待。
本报告不把模拟时间作为实际性能收益，不宣称已经完成修正版本的真机验证。

## 实际复现结果

相同load、matmul和store次序，仅改变Acc表示：

| 表示 | 0.66的FIX→M wait位点数 | 0.67 |
|---|---:|---:|
| 同一个boxed Acc池，互不重叠的列窗口 | 1 | 1 |
| 同一个boxed Acc池，相同窗口（确有重叠） | 1 | 1 |
| 两个独立alloc_tile，地址0与32768 | 0 | 0 |

期望第一行与第三行一致，同时第二行继续保留等待。输入均经CLI完整编译成功。

```bash
ptoas boxed_disjoint.pto --enable-insert-sync --pto-level=level3 -o boxed_disjoint.cpp
```

`boxed_disjoint.pto`：

```mlir
// CPU编译复现；未运行设备。所有Tile形状与地址明确，不含CSA、GM轮询或手工同步。
module {
  func.func @boxed_disjoint(%a: !pto.ptr<bf16>, %b: !pto.ptr<bf16>, %o0: !pto.ptr<f32>, %o1: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c384 = arith.constant 384 : index
    %a0 = arith.constant 0 : i64
    %a16384 = arith.constant 16384 : i64
    %a32768 = arith.constant 32768 : i64
    %tva = pto.make_tensor_view %a, shape = [%c64, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xbf16>
    %va = pto.partition_view %tva, offsets = [%c0, %c0], sizes = [%c64, %c128] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<64x128xbf16>
    %tvb = pto.make_tensor_view %b, shape = [%c128, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xbf16>
    %vb = pto.partition_view %tvb, offsets = [%c0, %c0], sizes = [%c128, %c128] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<128x128xbf16>
    %tvo0 = pto.make_tensor_view %o0, shape = [%c64, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xf32>
    %vo0 = pto.partition_view %tvo0, offsets = [%c0, %c0], sizes = [%c64, %c128] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<64x128xf32>
    %tvo1 = pto.make_tensor_view %o1, shape = [%c64, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xf32>
    %vo1 = pto.partition_view %tvo1, offsets = [%c0, %c0], sizes = [%c64, %c128] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<64x128xf32>
    %am = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=mat, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %bm = pto.alloc_tile addr = %a16384 : !pto.tile_buf<loc=mat, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %al = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=left, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %br = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=right, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %pool = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=384, v_row=64, v_col=384, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %cfirst = pto.subview %pool[%c0, %c0] sizes [64, 128] : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=384, v_row=64, v_col=384, blayout=col_major, slayout=row_major, fractal=1024, pad=0> -> !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %csecond = pto.subview %pool[%c0, %c128] sizes [64, 128] : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=384, v_row=64, v_col=384, blayout=col_major, slayout=row_major, fractal=1024, pad=0> -> !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    pto.tload ins(%va : !pto.partition_tensor_view<64x128xbf16>) outs(%am : !pto.tile_buf<loc=mat, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tload ins(%vb : !pto.partition_tensor_view<128x128xbf16>) outs(%bm : !pto.tile_buf<loc=mat, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tmov ins(%am : !pto.tile_buf<loc=mat, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%al : !pto.tile_buf<loc=left, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tmov ins(%bm : !pto.tile_buf<loc=mat, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%br : !pto.tile_buf<loc=right, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
    pto.tmatmul ins(%al, %br : !pto.tile_buf<loc=left, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%cfirst : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
    pto.tstore ins(%cfirst : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) outs(%vo0 : !pto.partition_tensor_view<64x128xf32>)
    pto.tmatmul ins(%al, %br : !pto.tile_buf<loc=left, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%csecond : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
    pto.tstore ins(%csecond : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) outs(%vo1 : !pto.partition_tensor_view<64x128xf32>)
    return
  }
}
```

对照1：把`%csecond`的列offset从`%c128`改为`%c0`，两窗口确实重叠，应保留FIX→M。

对照2：把`%pool`和两个`pto.subview`替换为两个独立`pto.alloc_tile`，其Tile类型
均与`%cfirst/%csecond`一致，地址分别为0和32768，其他计算及输出完全不变。
该对照不生成上述额外等待。

## 定位

已只读核对当前主仓的`lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp`：
`getPtoSubViewBaseAddresses`在SLayout不是NoneBox时直接返回nullopt；随后
`UpdateTileSubViewAliasBufferInfo`使用整个父buffer的保守范围。

因此这个限制不由GM计时退避、动态batch或AICPU调度引起。在固定地址的小型
IR里同样出现。适当扩大可证明的范围分析即可，不建议删除同步或设置全局开关。

## 建议修改范围

先支持可严格证明连续的窄集合：

- AddressSpace为ACC，FP32/INT32，4字节元素；
- BLayout=ColMajor、SLayout=RowMajor、fractal=1024，CompactMode=Null；
- 父shape为静态正数且按16对齐；
- rowOffset=0、rowSize覆盖父Tile所有行；
- colOffset/colSize为常量、按16对齐并通过边界检查。

此时一个完整列块窗口的物理范围是：

```cpp
byte_offset = colOffset * parentRows * elementBytes;
byte_size   = colSize   * parentRows * elementBytes;
```

同时缩小`baseAddresses`与`allocateSize`。不能仅改变起点，却继续沿用NoneBox
分段的一列长度。父范围/地址需要已知，乘法应检查溢出。无法证明的部分行、
动态offset、compact和其它boxed布局继续走原保守路径。

建议回归覆盖：相邻不相交列窗口、相同或部分重叠窗口、不同alloc但物理地址
重叠、嵌套窗口、动态offset保守回退；同时检查写后读/读后写所需同步仍存在。

正在独立本地PTOAS worktree验证上述窄范围改进；保留原共享安装。先以issue
提供复现和建议，不提交PR、不加标签。真机收益将单独验证后补充。

## 与已有报告的区别

相关#1587关注NoneBox UB的动态offset关系；本报告即使offset全部是常量也会
出现，原因是boxed L0C布局直接被保守回退。#536讨论别名来源正确性，本例
保留别名来源，只提高可证明静态物理区间的精度。

## 本地修改的CPU验证

独立worktree已编译通过。disjoint/overlap/separate_disjoint/partial_overlap/
dynamic_offset五个小例的FIX→M wait位点数分别为0/1/0/1/1，符合预期。
完整动态CSA的相同PTO IR也通过生成；QK写回后、PV计算前的额外wait消失，
保护下一次QK复用的事件仍保留。尚未据此宣称真机精度或性能通过。

下面为本地修改建议，供维护者审阅（无PR）：

```diff
diff --git a/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp b/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp
index efc60851..21aec057 100644
--- a/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp
+++ b/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp
@@ -24,7 +24,9 @@
 // [P0 新增] 引入副作用接口和 PTO 接口
 #include "mlir/Interfaces/SideEffectInterfaces.h"
 
+#include <limits>
 #include <optional>
+#include <utility>
 
 #define DEBUG_TYPE "pto-ir-translator"
 
@@ -183,12 +185,65 @@ static int64_t getTileMajorStride(pto::TileBufType type) {
   return stride;
 }
 
+// L0C按16×16块存储。覆盖父Tile全部行、按16列对齐的列窗口是连续物理
+// 区间；不能把它扩大成整个父池，否则独立QK/PV窗口也会形成FIX→M等待。
+// 这里只识别可直接证明的FP32/INT32普通布局。部分行、动态offset、compact
+// 或其它布局仍由原保守路径处理，不能用逻辑矩形冒充任意boxed布局的跨度。
+static std::optional<std::pair<uint64_t, uint64_t>>
+getContiguousAccColumnRange(pto::SubViewOp op, pto::TileBufType sourceType,
+                            int64_t elemBytes) {
+  if (getTileAddressSpace(sourceType) != pto::AddressSpace::ACC ||
+      !isStaticRank2Shape(sourceType.getShape()) || elemBytes != 4 ||
+      sourceType.getBLayoutValueI32() !=
+          static_cast<int32_t>(pto::BLayout::ColMajor) ||
+      sourceType.getSLayoutValueI32() !=
+          static_cast<int32_t>(pto::SLayout::RowMajor) ||
+      sourceType.getSFractalSizeI32() != 1024 ||
+      sourceType.getCompactModeI32() !=
+          static_cast<int32_t>(pto::CompactMode::Null) ||
+      op.getOffsets().size() != kTileRank2D) {
+    return std::nullopt;
+  }
+  auto sizes = op.getSizes();
+  if (!sizes || sizes.size() != kTileRank2D) {
+    return std::nullopt;
+  }
+  int64_t rowOffset = 0, colOffset = 0;
+  if (!getConstIndexValue(op.getOffsets()[0], rowOffset) ||
+      !getConstIndexValue(op.getOffsets()[1], colOffset)) {
+    return std::nullopt;
+  }
+  int64_t rows = sourceType.getShape()[0];
+  int64_t cols = sourceType.getShape()[1];
+  int64_t rowSize = cast<IntegerAttr>(sizes[0]).getInt();
+  int64_t colSize = cast<IntegerAttr>(sizes[1]).getInt();
+  if (rows <= 0 || cols <= 0 || rows % 16 != 0 || cols % 16 != 0 ||
+      rowOffset != 0 || rowSize != rows || colOffset < 0 ||
+      colOffset > cols || colSize <= 0 || colSize > cols - colOffset ||
+      colOffset % 16 != 0 || colSize % 16 != 0) {
+    return std::nullopt;
+  }
+  uint64_t maxBytes = std::numeric_limits<uint64_t>::max();
+  if (static_cast<uint64_t>(rows) > maxBytes / elemBytes) {
+    return std::nullopt;
+  }
+  uint64_t columnBytes = static_cast<uint64_t>(rows) * elemBytes;
+  if (static_cast<uint64_t>(cols) > maxBytes / columnBytes) {
+    return std::nullopt;
+  }
+  return std::make_pair(static_cast<uint64_t>(colOffset) * columnBytes,
+                        static_cast<uint64_t>(colSize) * columnBytes);
+}
+
 static std::optional<SmallVector<uint64_t>>
 getPtoSubViewBaseAddresses(pto::SubViewOp op, pto::TileBufType sourceType,
                            int64_t elemBytes) {
   if (!isStaticRank2Shape(sourceType.getShape())) {
     return std::nullopt;
   }
+  if (auto range = getContiguousAccColumnRange(op, sourceType, elemBytes)) {
+    return SmallVector<uint64_t>{range->first};
+  }
   if (sourceType.getSLayoutValueI32() !=
       static_cast<int32_t>(pto::SLayout::NoneBox)) {
     return std::nullopt;
@@ -1019,6 +1074,9 @@ void PTOIRTranslator::UpdateSlotSelectedAliasBufferInfo(Value result,
 static uint64_t getSubViewSegmentSize(pto::SubViewOp op,
                                       pto::TileBufType sourceType,
                                       unsigned elemBytes) {
+  if (auto range = getContiguousAccColumnRange(op, sourceType, elemBytes)) {
+    return range->second;
+  }
   auto sizesAttr = op.getSizes();
   int64_t rowSize = cast<IntegerAttr>(sizesAttr[0]).getInt();
   int64_t colSize = cast<IntegerAttr>(sizesAttr[1]).getInt();

```
