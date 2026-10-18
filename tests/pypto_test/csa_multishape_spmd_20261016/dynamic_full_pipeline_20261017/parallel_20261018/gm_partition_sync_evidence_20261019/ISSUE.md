## 问题

动态CSA Attention的PV依次把四个FP32列片段写到同一GM Tensor。
各窗口为64×128、GM行距512、列起点0/128/256/384，地址不重叠；
InsertSync仍在这些写回之间插入`pipe_barrier(PIPE_FIX)`。
这是可证明范围丢失导致的保守同步/优化缺失，不是当前已发现的数值错误。

## 复现版本与定位

- 发布版PTOAS 0.66和0.67的最小例均复现。
- 本次只读核对上游默认分支`master`：`PTOIRTranslator::dispatchAliasViewOp`
  对`make_tensor_view`、`partition_view`只执行`UpdateAliasBufferInfo`，没有保留分片范围。
- `MemoryDependentAnalyzer::isGMBufferOverlap`随后对同一未知大小的根返回重叠。
- 与 #1623 不同：#1623 是片上L0C boxed subview；本问题是GM partition_view。
  与 #1587 的动态Vec subview也不是同一种地址范围。

## Before / Expected / Actual

同一输出根的两个ND列窗口，行距256元素、列区间[0,128)、[128,256)，
窗口和行距均64B对齐。两份Acc独立，排除片上别名造成的等待。

| 输入 | 0.66 FIX barrier | 0.67 FIX barrier | 预期 |
|---|---:|---:|---|
| 独立输出根 | 0 | 0 | 保持无额外等待 |
| 同一根、不相交列 | 1 | 1 | 可以省去该barrier |
| 同一根、动态行起点、不相交列 | 1 | 1 | 可以省去该barrier |
| 同一根、列重叠 | 1 | 1 | 必须保留依赖 |
| 同一根、动态列起点 | 1 | 1 | 未证明，不应消除 |

```bash
ptoas same_root_disjoint.pto --enable-insert-sync --pto-level=level3 -o same_root_disjoint.cpp
```

完整最小IR（只做CPU编译，不需要CSA、权重或NPU）：

```mlir
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// CPU编译复现；未运行设备。所有Tile形状与地址明确，不含CSA、GM轮询或手工同步。
module {
  func.func @same_root_disjoint(%a: !pto.ptr<bf16>, %b: !pto.ptr<bf16>, %o0: !pto.ptr<f32>, %o1: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c384 = arith.constant 384 : index
    %c256 = arith.constant 256 : index
    %a0 = arith.constant 0 : i64
    %a16384 = arith.constant 16384 : i64
    %a32768 = arith.constant 32768 : i64
    %tva = pto.make_tensor_view %a, shape = [%c64, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xbf16>
    %va = pto.partition_view %tva, offsets = [%c0, %c0], sizes = [%c64, %c128] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<64x128xbf16>
    %tvb = pto.make_tensor_view %b, shape = [%c128, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xbf16>
    %vb = pto.partition_view %tvb, offsets = [%c0, %c0], sizes = [%c128, %c128] : !pto.tensor_view<?x?xbf16> -> !pto.partition_tensor_view<128x128xbf16>
    %tvo0 = pto.make_tensor_view %o0, shape = [%c64, %c256], strides = [%c256, %c1] : !pto.tensor_view<?x?xf32>
    %vo0 = pto.partition_view %tvo0, offsets = [%c0, %c0], sizes = [%c64, %c128] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<64x128xf32>
    %tvo1 = pto.make_tensor_view %o0, shape = [%c64, %c256], strides = [%c256, %c1] : !pto.tensor_view<?x?xf32>
    %vo1 = pto.partition_view %tvo1, offsets = [%c0, %c128], sizes = [%c64, %c128] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<64x128xf32>
    %am = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=mat, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %bm = pto.alloc_tile addr = %a16384 : !pto.tile_buf<loc=mat, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %al = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=left, dtype=bf16, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %br = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=right, dtype=bf16, rows=128, cols=128, v_row=128, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %cfirst = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
    %csecond = pto.alloc_tile addr = %a32768 : !pto.tile_buf<loc=acc, dtype=f32, rows=64, cols=128, v_row=64, v_col=128, blayout=col_major, slayout=row_major, fractal=1024, pad=0>
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

## 建议修正：从实际partition恢复窄范围的列不相交证明

下面本地patch仅在以下条件成立时返回不重叠，其余全部走原逻辑：

1. 两个直接二维ND partition，来自make_tensor_view；不跨嵌套view、addptr或cast推测地址。
2. 完全相同的基指针SSA和元素类型，内层stride=1，相同的正2次幂行距。
3. 列起点和宽度静态、正宽度、位于一个行距内，行距/列起点/宽度均64B对齐。
4. 列区间互斥；行起点可以动态，低位列关系仍不相交。
5. packed FP4、不规则布局、不同/动态stride、未知指针偏移及所有真实重叠保持保守。

2次幂限制使整数地址环绕仍保留列的低位关系；计算先用减法验证上界，避免
start+count溢出，并限制元素字节乘法范围。没有直接删生成C++里的等待。

```diff
diff --git a/lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp b/lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp
index 3ea40e17..8de88916 100644
--- a/lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp
+++ b/lib/PTO/Transforms/InsertSync/MemoryDependentAnalyzer.cpp
@@ -14,6 +14,7 @@
 #include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
 #include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
 #include "mlir/Dialect/Arith/IR/Arith.h"
+#include <limits>
 
 using namespace mlir;
 using namespace mlir::pto;
@@ -274,8 +275,91 @@ bool MemoryDependentAnalyzer::MemAlias(const BaseMemInfo *a,
   return crossRootOverlap;
 }
  
+namespace {
+struct GMColumnWindow {
+  Value pointer;
+  Type elementType;
+  int64_t rowStride;
+  int64_t begin;
+  int64_t end;
+};
+
+static std::optional<int64_t> getStaticIndex(Value value) {
+  auto constant = value.getDefiningOp<arith::ConstantOp>();
+  if (!constant)
+    return std::nullopt;
+  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
+  if (!integer || integer.getValue().getBitWidth() > 64)
+    return std::nullopt;
+  return integer.getValue().getSExtValue();
+}
+
+// 只证明一层二维ND partition的固定列窗口。行起点可以动态：同一pointer、
+// 同一正的2次幂行距下，列窗口在每一行内互斥，就不可能因不同行起点
+// 变成重叠；2次幂限制也保留整数地址环绕时的低位列关系。
+// 不跨addptr/cast/嵌套view猜基址，也不推断动态列、动态行距或NZ布局。
+// 64B对齐的窗口保守覆盖DMA/FP32 FIX的最小写粒度，禁止按未对齐的
+// 逻辑小切片消除可能仍共享物理写粒度的依赖。
+static std::optional<GMColumnWindow> getGMColumnWindow(Value value) {
+  auto partition = value.getDefiningOp<pto::PartitionViewOp>();
+  if (!partition || partition.getOffsets().size() != 2 ||
+      partition.getSizes().size() != 2)
+    return std::nullopt;
+  auto view = partition.getSource().getDefiningOp<pto::MakeTensorViewOp>();
+  if (!view || view.getShape().size() != 2 || view.getStrides().size() != 2)
+    return std::nullopt;
+  auto viewType = dyn_cast<pto::TensorViewType>(view.getResult().getType());
+  if (!viewType)
+    return std::nullopt;
+  auto layout = view.getLayoutAttr();
+  if (!layout)
+    layout = viewType.getLayoutAttr();
+  if (layout && layout.getLayout() != pto::Layout::ND)
+    return std::nullopt;
+  auto partitionLayout = partition.getResult().getType().getLayoutAttr();
+  if (partitionLayout && partitionLayout.getLayout() != pto::Layout::ND)
+    return std::nullopt;
+
+  Type elementType = viewType.getElementType();
+  if (!elementType.isIntOrFloat())
+    return std::nullopt;
+  unsigned bits = elementType.getIntOrFloatBitWidth();
+  if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
+    return std::nullopt;
+  auto stride = getStaticIndex(view.getStrides()[0]);
+  auto innerStride = getStaticIndex(view.getStrides()[1]);
+  auto begin = getStaticIndex(partition.getOffsets()[1]);
+  auto count = getStaticIndex(partition.getSizes()[1]);
+  if (!stride || !innerStride || !begin || !count || *stride <= 0 ||
+      *innerStride != 1 || *begin < 0 || *count <= 0 || *count > *stride ||
+      *begin > *stride - *count)
+    return std::nullopt;
+  if ((*stride & (*stride - 1)) != 0 ||
+      *stride > std::numeric_limits<int64_t>::max() / (bits / 8))
+    return std::nullopt;
+  int64_t alignmentElements = 64 / (bits / 8);
+  if (*stride % alignmentElements != 0 || *begin % alignmentElements != 0 ||
+      *count % alignmentElements != 0)
+    return std::nullopt;
+  return GMColumnWindow{view.getPtr(), elementType, *stride, *begin,
+                        *begin + *count};
+}
+
+static bool haveDisjointGMColumns(const BaseMemInfo *a, const BaseMemInfo *b) {
+  auto left = getGMColumnWindow(a->baseBuffer);
+  auto right = getGMColumnWindow(b->baseBuffer);
+  return left && right && left->pointer == right->pointer &&
+         left->elementType == right->elementType &&
+         left->rowStride == right->rowStride &&
+         (left->end <= right->begin || right->end <= left->begin);
+}
+} // namespace
+
 bool MemoryDependentAnalyzer::isGMBufferOverlap(const BaseMemInfo *a,
                                                 const BaseMemInfo *b) const {
+  // 未命中的情况全部沿用原范围/根分析；不删除循环复用、真实重叠或未知地址依赖。
+  if (haveDisjointGMColumns(a, b))
+    return false;
   if (a->baseAddresses.empty() || b->baseAddresses.empty()) {
     return true;
   }
```

## 已完成的本地验证

- 12个CPU分片/指针/stride对照以及FileCheck通过：安全分片移除barrier，
  同列、部分重叠、动态列、动态或不同stride、非2次幂行距、非单位内stride、
  未对齐窗口和未知指针偏移保留。
- 额外循环复用测试通过：同列跨迭代重复写，仍有FIX barrier保护。
- 同一份完整动态CSA PTO IR分别用修正前后编译器生成C++：
  FIX barrier静态位置81→77，MTE3 barrier112→110；set/wait数量均1675不变。
  TSTORE171、TLOAD349、TEXTRACT214、MMA79、MMA_ACC68及FFTS通知/等待7/8均不变。
  **仅滤去set_flag/wait_flag/pipe_barrier语句后，两份C++逐行完全相同**。
- 这些都是生成物/CPU证据，不是设备正确性或性能证明；正式真机验证尚未完成，
  不预先宣称CSA加速。模型流水只用于提出候选，判断收益仍看正式等待版本的完整L1。
- 编译器仅在独立worktree修改，新可执行文件另行链接；没有修改共享安装，
  没有覆盖正在排队的其它实验使用的原可执行文件。

没有提交PR；附修改建议供组件维护者评估适用边界。
