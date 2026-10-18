## 问题

在A3的L1双缓冲中，TEXTRACT读取第一个槽，随后TLOAD写另一个不重叠槽，
InsertSync仍插入MTE1→MTE2等待，阻止本可重叠的预取。

两个保守来源叠加：boxed Mat subview丢失精确物理范围；TEXTRACT的Read
效果按整个源Tile收集，没有按offset和目标shape缩小读取范围。
这是同核MTE流水依赖，与GM跨核通知/Scalar缓存一致性无关。

## 已验证版本

- 本地0.66及官方0.67二进制均复现下列生成代码症状。
- 只读核对默认master（7a32cb83ef927dab4b0c4aa928093acacea24abc），
  boxed subview仍走保守路径，Read仍直接UpdateDefUseVec；未构建该新master。
- 本地建议修正来自独立工作树，未修改共享安装。尚无本修正的真机性能结论。

## 无设备最小复现

```bash
ptoas repro.pto --enable-insert-sync --pto-level=level3 -o repro.cpp
```

```mlir
module {
  func.func @boxed_disjoint(%src: !pto.ptr<i8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %a0 = arith.constant 0 : i64
    %a16384 = arith.constant 16384 : i64
    %tv = pto.make_tensor_view %src, shape = [%c32, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xi8>
    %input = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c32, %c128] : !pto.tensor_view<?x?xi8> -> !pto.partition_tensor_view<32x128xi8>
    %pool = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %transposed = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=128, v_row=128, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %right = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=right, dtype=i8, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %write_pool = pto.alloc_tile addr = %a0 : !pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %first = pto.subview %pool[%c0, %c0] sizes [32, 128] : !pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0> -> !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=128, v_row=32, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %second = pto.subview %pool[%c32, %c0] sizes [32, 128] : !pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0> -> !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=128, v_row=32, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %later = pto.subview %write_pool[%c64, %c0] sizes [32, 128] : !pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=128, v_row=128, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0> -> !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=128, v_row=32, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    pto.tload ins(%input : !pto.partition_tensor_view<32x128xi8>) outs(%first : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=128, v_row=32, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tload ins(%input : !pto.partition_tensor_view<32x128xi8>) outs(%second : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=128, v_row=32, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.textract ins(%transposed, %c0, %c0 : !pto.tile_buf<loc=mat, dtype=i8, rows=128, cols=128, v_row=128, v_col=128, blayout=row_major, slayout=col_major, fractal=512, pad=0>, index, index) outs(%right : !pto.tile_buf<loc=right, dtype=i8, rows=128, cols=64, v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
    pto.tload ins(%input : !pto.partition_tensor_view<32x128xi8>) outs(%later : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=128, v_row=32, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    return
  }
}
```

生成物关键顺序：

```cpp
TEXTRACT(...);
set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
TLOAD(...);
```

将write_pool改为独立地址16384时，同样的TLOAD前没有这条等待。
实际动态CSA的key预取也采用该双槽结构，TEXTRACT结果供后续QK使用。

## 物理范围证明

不能只看逻辑行号。按PTO-ISA GetTileOffset枚举所有字节，Mat INT8
ColMajor/RowMajor(512B分形)与同址RowMajor/ColMajor转置视图逐元素一致。
本例读取范围为4个2048B段，写入范围为4个1024B段：

|段|TEXTRACT读取|后续TLOAD写入|
|---|---|---|
|0|[0,2048)|[2048,3072)|
|1|[4096,6144)|[6144,7168)|
|2|[8192,10240)|[10240,11264)|
|3|[12288,14336)|[14336,15360)|

均不重叠。若将写offset改为32，则有4096B真实交叠，必须保留等待。
TLOAD→TEXTRACT的数据就绪依赖也必须保持，不能整体删MTE1/MTE2同步。

## 建议修改

1. 对静态、完整分形边界对齐的Mat INT8 Nz/Zn subview，保留物理分段。
2. 为TEXTRACT创建指令专属读范围，依据源物理布局、静态offset、目标物理
   shape计算；不要改写源Tile全局alias映射。使用目标物理shape而非动态
   valid_shape，以覆盖可能读到的最大范围。
3. 仅在直接alloc_tile、已知物理基址、无compact、匹配布局/类型及合法边界
   时证明；动态offset、未支持布局/类型、派生物理stride均保守回退。
4. 连续分段合成原连续范围，避免仅因表示不同干扰同步去重。
5. 指令专属BaseMemInfo需要活到分析、同步移动、事件分配和代码生成结束。

检查时还复现了#1620同类问题：显式动态alloc_tile地址没有标记
aliasesUnknownRange，不同根可能被错判独立。建议将未知的显式本地地址
保守标记；无addr的独立逻辑分配保持原行为。下面patch含此小型正确性保护。

## 本地验证及限制

- 218组Nz/Zn窗口的闭式分段与PTO-ISA逐元素偏移枚举一致。
- 9个最小用例通过：分离/重叠槽、独立地址、动态Subview offset、FP16回退、
  动态TEXTRACT offset、显式未知物理地址、48列对齐窗口、循环真实重用。
- 循环真实重用保留MTE1→MTE2及MTE2→MTE1等待。未知地址原先新旧均失败，
  没有将其伪记PASS；增加上述保守标记后恢复必要依赖。
- 同一完整动态CSA IR除同步语句外C++逐行相同，PIPE_ALL/FIX/MTE3、
  FFTS跨核同步均不变。该IR的2394个alloc_tile基址均为常量，因此未知
  地址保护不是这份CSA性能试验的变化因素。
- 完整CSA静态set/wait位置各1706→1707；修复的重点是将等待改为每个槽
  对应的旧读取，让另一槽预取可推进，并非简单追求等待条数减少。
- 真机同轮精度/图/状态保护与完整L1对照已提交，结果待补，不预记收益。

## 建议代码（未提PR）

以下为独立树上的实现建议，包含此前#1623 Acc列范围修正的文件上下文；
移至新master需适配上下文。未混入此前#1624的GM列范围性能修改。

<details><summary>建议patch</summary>

```diff
diff --git a/include/PTO/Transforms/InsertSync/PTOIRTranslator.h b/include/PTO/Transforms/InsertSync/PTOIRTranslator.h
index cb79f829..b9cbdad0 100644
--- a/include/PTO/Transforms/InsertSync/PTOIRTranslator.h
+++ b/include/PTO/Transforms/InsertSync/PTOIRTranslator.h
@@ -65,6 +65,9 @@ private:
   Buffer2MemInfoMap &buffer2MemInfoMap_;
   MemoryDependentAnalyzer &memAnalyzer_;
   SyncAnalysisMode mode_;
+  // 单条指令的精确读范围不能改写源Tile的全局alias映射。Translator在整个
+  // InsertSync分析/移动/分配/生成期间存活，此处持有Compound.useVec引用的对象。
+  SmallVector<std::unique_ptr<BaseMemInfo>> preciseReadRegions_;
  
   // --- 递归遍历逻辑 ---
   void RecursionIR(Region *region);
@@ -112,6 +115,8 @@ private:
 
   // 根据 Values 填充 Def/Use 列表
   void UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec);
+  bool UpdateTExtractReadRegion(Operation *op, Value source,
+                               SmallVector<const BaseMemInfo *> &vec);
 
   // 调试辅助
   std::string getPipelineName(PipelineType pipe) const;
diff --git a/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp b/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp
index 21aec057..00ba557a 100644
--- a/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp
+++ b/lib/PTO/Transforms/InsertSync/PTOIRTranslator.cpp
@@ -24,6 +24,7 @@
 // [P0 新增] 引入副作用接口和 PTO 接口
 #include "mlir/Interfaces/SideEffectInterfaces.h"
 
+#include <algorithm>
 #include <limits>
 #include <optional>
 #include <utility>
@@ -107,9 +108,20 @@ appendUniqueMemInfo(SmallVectorImpl<std::unique_ptr<BaseMemInfo>> &infos,
 
 } // namespace
 
+// 仅穿透数值可证明不变的整数cast。直接剥掉trunc会把65536->i16的0
+// 错认成65536，造成地址/窗口不重叠的错误证明。不能证明时返回unknown，
+// 交给保守alias路径；此处不尝试执行会改变数值的常量表达式。
 static bool getConstIndexValue(Value value, int64_t &out) {
-  while (true) {
+  unsigned requiredSignedWidth = 64;
+  bool requireNonNegative = false;
+  for (unsigned depth = 0; ; ++depth) {
+    if (depth > 64)
+      return false;
     if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
+      if (auto target = dyn_cast<IntegerType>(castOp.getResult().getType()))
+        requiredSignedWidth = std::min(requiredSignedWidth, target.getWidth());
+      else if (!isa<IndexType>(castOp.getResult().getType()))
+        return false;
       value = castOp.getIn();
       continue;
     }
@@ -118,31 +130,27 @@ static bool getConstIndexValue(Value value, int64_t &out) {
       continue;
     }
     if (auto extOp = value.getDefiningOp<arith::ExtUIOp>()) {
+      requireNonNegative = true;
       value = extOp.getIn();
       continue;
     }
     if (auto truncOp = value.getDefiningOp<arith::TruncIOp>()) {
+      auto target = dyn_cast<IntegerType>(truncOp.getResult().getType());
+      if (!target)
+        return false;
+      requiredSignedWidth = std::min(requiredSignedWidth, target.getWidth());
       value = truncOp.getIn();
       continue;
     }
     break;
   }
-  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>()) {
-    out = constIndex.value();
-    return true;
-  }
-  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>()) {
-    out = constInt.value();
-    return true;
-  }
   auto constOp = value.getDefiningOp<arith::ConstantOp>();
-  auto intAttr =
-      constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr();
-  if (!intAttr) {
+  auto intAttr = constOp ? dyn_cast<IntegerAttr>(constOp.getValue()) : IntegerAttr();
+  if (!intAttr || !intAttr.getValue().isSignedIntN(64))
     return false;
-  }
-  out = intAttr.getInt();
-  return true;
+  llvm::APInt bits = intAttr.getValue().sextOrTrunc(64);
+  out = bits.getSExtValue();
+  return (!requireNonNegative || out >= 0) && bits.isSignedIntN(requiredSignedWidth);
 }
 
 static std::optional<uint64_t> getKnownPhysicalAddress(Value value) {
@@ -235,6 +243,95 @@ getContiguousAccColumnRange(pto::SubViewOp op, pto::TileBufType sourceType,
                         static_cast<uint64_t>(colSize) * columnBytes);
 }
 
+struct BoxedI8Region {
+  SmallVector<uint64_t> offsets;
+  uint64_t segmentBytes;
+};
+
+// 只证明Mat的普通INT8 Nz/Zn布局，且窗口沿完整512B分形边界切分。
+// 每段含同一外层块行/列中的若干完整小块，不能用逻辑连续矩形代替物理跨度。
+// 其它dtype/compact/未对齐/动态范围保守回退；段数上限仅约束编译期开销。
+static std::optional<BoxedI8Region>
+getBoxedI8Region(pto::TileBufType type, int64_t rowOffset, int64_t colOffset,
+                 int64_t rowSize, int64_t colSize) {
+  if (!type || getTileAddressSpace(type) != pto::AddressSpace::MAT ||
+      !type.getElementType().isInteger(8) ||
+      !isStaticRank2Shape(type.getShape()) || type.getSFractalSizeI32() != 512 ||
+      type.getCompactModeI32() != static_cast<int32_t>(pto::CompactMode::Null))
+    return std::nullopt;
+  bool nz = type.getBLayoutValueI32() ==
+                static_cast<int32_t>(pto::BLayout::ColMajor) &&
+            type.getSLayoutValueI32() ==
+                static_cast<int32_t>(pto::SLayout::RowMajor);
+  bool zn = type.getBLayoutValueI32() ==
+                static_cast<int32_t>(pto::BLayout::RowMajor) &&
+            type.getSLayoutValueI32() ==
+                static_cast<int32_t>(pto::SLayout::ColMajor);
+  if (!nz && !zn)
+    return std::nullopt;
+  int64_t rows = type.getShape()[0], cols = type.getShape()[1];
+  int64_t innerRows = nz ? 16 : 32, innerCols = nz ? 32 : 16;
+  if (rows <= 0 || cols <= 0 || rowOffset < 0 || colOffset < 0 ||
+      rowOffset > rows || colOffset > cols || rowSize <= 0 || colSize <= 0 ||
+      rowSize > rows - rowOffset || colSize > cols - colOffset ||
+      rows % innerRows || cols % innerCols ||
+      rowOffset % innerRows || rowSize % innerRows ||
+      colOffset % innerCols || colSize % innerCols)
+    return std::nullopt;
+  uint64_t limit = std::numeric_limits<uint64_t>::max();
+  if (static_cast<uint64_t>(rows) > limit / static_cast<uint64_t>(cols))
+    return std::nullopt;
+  int64_t count = nz ? colSize / innerCols : rowSize / innerRows;
+  constexpr int64_t maxSegments = 4096;
+  if (count > maxSegments)
+    return std::nullopt;
+  BoxedI8Region region;
+  region.segmentBytes = nz ? static_cast<uint64_t>(rowSize) * innerCols
+                           : static_cast<uint64_t>(colSize) * innerRows;
+  region.offsets.reserve(static_cast<size_t>(count));
+  for (int64_t block = 0; block < count; ++block) {
+    uint64_t offset;
+    if (nz) {
+      offset = static_cast<uint64_t>(colOffset / innerCols + block) *
+                   static_cast<uint64_t>(rows) * innerCols +
+               static_cast<uint64_t>(rowOffset) * innerCols;
+    } else {
+      offset = static_cast<uint64_t>(rowOffset / innerRows + block) *
+                   static_cast<uint64_t>(cols) * innerRows +
+               static_cast<uint64_t>(colOffset) * innerRows;
+    }
+    region.offsets.push_back(offset);
+  }
+  // 完整主维窗口的分段往往首尾相接。合成原来的连续范围，避免仅因等价
+  // 表示不同而使后续同步去重/移动丢失“相同访问”的判断。
+  bool contiguous = true;
+  for (size_t i = 1; i < region.offsets.size(); ++i)
+    contiguous &= region.offsets[i] ==
+                  region.offsets[i - 1] + region.segmentBytes;
+  if (contiguous && region.offsets.size() > 1) {
+    region.segmentBytes *= region.offsets.size();
+    region.offsets.resize(1);
+  }
+  return region;
+}
+
+static std::optional<BoxedI8Region>
+getBoxedI8SubviewRegion(pto::SubViewOp op, pto::TileBufType sourceType) {
+  // 嵌套subview可能继承更大父Tile的物理stride；仅直接alloc_tile的shape
+  // 能作为这里的物理根布局。未知/派生根仍保留原保守alias传播。
+  if (!op.getSource().getDefiningOp<pto::AllocTileOp>() ||
+      op.getOffsets().size() != kTileRank2D ||
+      !op.getSizes() || op.getSizes().size() != kTileRank2D)
+    return std::nullopt;
+  int64_t row = 0, col = 0;
+  if (!getConstIndexValue(op.getOffsets()[0], row) ||
+      !getConstIndexValue(op.getOffsets()[1], col))
+    return std::nullopt;
+  return getBoxedI8Region(sourceType, row, col,
+                         cast<IntegerAttr>(op.getSizes()[0]).getInt(),
+                         cast<IntegerAttr>(op.getSizes()[1]).getInt());
+}
+
 static std::optional<SmallVector<uint64_t>>
 getPtoSubViewBaseAddresses(pto::SubViewOp op, pto::TileBufType sourceType,
                            int64_t elemBytes) {
@@ -244,6 +341,8 @@ getPtoSubViewBaseAddresses(pto::SubViewOp op, pto::TileBufType sourceType,
   if (auto range = getContiguousAccColumnRange(op, sourceType, elemBytes)) {
     return SmallVector<uint64_t>{range->first};
   }
+  if (auto region = getBoxedI8SubviewRegion(op, sourceType))
+    return region->offsets;
   if (sourceType.getSLayoutValueI32() !=
       static_cast<int32_t>(pto::SLayout::NoneBox)) {
     return std::nullopt;
@@ -563,7 +662,9 @@ LogicalResult PTOIRTranslator::UpdateAllocTileOpMemInfo(pto::AllocTileOp op) {
   // 3. 注册 Buffer 信息
   auto newMemInfo = std::make_unique<BaseMemInfo>(
       res, res, space, SmallVector<uint64_t>{baseAddr}, sizeInBytes,
-      knownPhysicalAddress.has_value() && isLocalAddressSpace(space));
+      knownPhysicalAddress.has_value() && isLocalAddressSpace(space),
+      static_cast<bool>(op.getAddr()) && !knownPhysicalAddress.has_value() &&
+          isLocalAddressSpace(space));
 
   buffer2MemInfoMap_[res].emplace_back(newMemInfo->clone());
   return success();
@@ -715,7 +816,8 @@ void PTOIRTranslator::UpdatePTOOpInfoWithPipeline(Operation *op,
        // 只有当 Value 在我们的 BufferMap 中有记录时，才视为有效依赖
        // (过滤掉比如 Loop Iterator 或其他标量)
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
-          UpdateDefUseVec({val}, useVec);
+          if (!UpdateTExtractReadRegion(op, val, useVec))
+            UpdateDefUseVec({val}, useVec);
        } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
           UpdateDefUseVec({val}, defVec);
        }
@@ -1077,6 +1179,8 @@ static uint64_t getSubViewSegmentSize(pto::SubViewOp op,
   if (auto range = getContiguousAccColumnRange(op, sourceType, elemBytes)) {
     return range->second;
   }
+  if (auto region = getBoxedI8SubviewRegion(op, sourceType))
+    return region->segmentBytes;
   auto sizesAttr = op.getSizes();
   int64_t rowSize = cast<IntegerAttr>(sizesAttr[0]).getInt();
   int64_t colSize = cast<IntegerAttr>(sizesAttr[1]).getInt();
@@ -1130,6 +1234,17 @@ void PTOIRTranslator::UpdateTileSubViewAliasBufferInfo(pto::SubViewOp op) {
     UpdateConservativeAliasBufferInfo(result, source);
     return;
   }
+  if (getBoxedI8SubviewRegion(op, sourceType)) {
+    uint64_t end = subViewAddresses->back() + segmentSize;
+    for (auto &parent : buffer2MemInfoMap_[source]) {
+      if (parent->aliasesUnknownRange || !parent->hasKnownPhysicalAddresses ||
+          end > parent->allocateSize ||
+          parent->baseAddresses[0] > std::numeric_limits<uint64_t>::max() - end) {
+        UpdateConservativeAliasBufferInfo(result, source);
+        return;
+      }
+    }
+  }
 
   auto &resultMemInfoVec = buffer2MemInfoMap_[result];
   for (auto &parentInfo : buffer2MemInfoMap_[source]) {
@@ -1156,6 +1271,55 @@ void PTOIRTranslator::UpdateDefUseVec(ValueRange values, SmallVector<const BaseM
   }
 }
 
+bool PTOIRTranslator::UpdateTExtractReadRegion(
+    Operation *op, Value source, SmallVector<const BaseMemInfo *> &vec) {
+  auto extract = dyn_cast<pto::TExtractOp>(op);
+  if (!extract || source != extract.getSrc() ||
+      !source.getDefiningOp<pto::AllocTileOp>())
+    return false;
+  auto sourceType = dyn_cast<pto::TileBufType>(source.getType());
+  auto destType = dyn_cast<pto::TileBufType>(extract.getDst().getType());
+  if (!sourceType || !destType ||
+      !isStaticRank2Shape(destType.getShape()) ||
+      sourceType.getElementType() != destType.getElementType() ||
+      sourceType.getBLayoutValueI32() != destType.getBLayoutValueI32() ||
+      sourceType.getSLayoutValueI32() != destType.getSLayoutValueI32() ||
+      sourceType.getSFractalSizeI32() != destType.getSFractalSizeI32() ||
+      destType.getCompactModeI32() != static_cast<int32_t>(pto::CompactMode::Null))
+    return false;
+  auto destSpace = getTileAddressSpace(destType);
+  if (destSpace != pto::AddressSpace::LEFT && destSpace != pto::AddressSpace::RIGHT)
+    return false;
+  int64_t row = 0, col = 0;
+  if (!getConstIndexValue(extract.getIndexRow(), row) ||
+      !getConstIndexValue(extract.getIndexCol(), col))
+    return false;
+  // 用目标物理shape覆盖最大读取范围，不以动态valid shape缩小依赖。
+  auto region = getBoxedI8Region(sourceType, row, col,
+                                destType.getShape()[0], destType.getShape()[1]);
+  auto found = buffer2MemInfoMap_.find(source);
+  if (!region || region->offsets.empty() || found == buffer2MemInfoMap_.end() ||
+      found->second.empty() || !hasSingleBaseAllocatedParents(found->second))
+    return false;
+  uint64_t end = region->offsets.back() + region->segmentBytes;
+  for (auto &parent : found->second) {
+    if (parent->aliasesUnknownRange || !parent->hasKnownPhysicalAddresses ||
+        parent->scope != pto::AddressSpace::MAT || end > parent->allocateSize ||
+        parent->baseAddresses[0] > std::numeric_limits<uint64_t>::max() - end)
+      return false;
+  }
+  for (auto &parent : found->second) {
+    auto info = parent->clone(source);
+    info->baseAddresses.clear();
+    for (uint64_t offset : region->offsets)
+      info->baseAddresses.push_back(parent->baseAddresses[0] + offset);
+    info->allocateSize = region->segmentBytes;
+    vec.push_back(info.get());
+    preciseReadRegions_.push_back(std::move(info));
+  }
+  return true;
+}
+
 // ============================================================================
 // 9. 调试与打印支持
 // ============================================================================

```
</details>


## 后续验证

完整七档设备输出/TopK/score/cache-state、保护区、协议及图均通过，
长B16 metadata/padding回放通过。相同源级SPMD、同进程五窗口中位
对照（参考编译器→候选编译器，us）：

|场景|参考|候选|
|---|---:|---:|
|128K/B4|503.64|493.12|
|128K/B8|641.04|641.24|
|128K/B16|833.94|830.90|
|128K/B24|1082.00|1076.02|
|128K/B28|1244.46|1224.18|
|8K/B16|652.52|634.08|
|8K/B32|933.30|926.38|

六档中位降低，B8差0.20us基本持平；不把全部窗口说成方向一致。
最大两项中位降低分别为8K/B16的2.826%、128K/B4的2.089%，
B28为1.630%。这些是完整CSA耗时，包含内部等待，不是模型预测。

额外加了两个缩窄cast控制，原完整流水已会先正确折叠它们；没有将其
报告成新的已复现生成错误。建议helper仅在数值保持时穿透trunc/extui，
不能证明则保守回退，避免范围证明依赖特定折叠顺序。更新后的建议
通过11项范围/循环/cast控制与7项既有Acc回归。

此保守加强生成独立v3二进制，没有覆盖进行中的v2测试。将七档实际
PTO输入逐一用v3重生成，完整C++与实测v2逐文件一致（包括同步），
因此保留既有设备证据，不重复上卡，也不把只比对计数当作等价证明。
共享安装未改变，未提交PR或标签。
