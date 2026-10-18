补充一个独立于“已有最大块可直接细分”的容量例子：前段 Left 同时存活 48+12 KiB，后段同时存活 32+32 KiB。峰值为 64 KiB，但没有任何一个原始 root 的 size 达到 64 KiB。固定 root 容量的子区间装箱仍会留下 80 KiB，必须允许同一个编译器所有的复用区在生命周期约束下有界增长。

2026-10-18 核对 main `14736f65c` 的派生独立构建（含已另报的 FloorMod/view/While 修正）：下述原样脚本报 `Left buffer usage (81920 bytes) exceeds platform limit (65536 bytes)`；本地有界 arena 修正通过。无 NPU、无模型依赖。

```python
import pypto.language as pl
from pypto import backend, passes
from pypto.backend import BackendType
backend.reset_for_testing()
backend.set_backend_type(BackendType.Ascend910B)

first_rows = 192
total_rows = first_rows + 48 + 256

@pl.program
class PhaseArena:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(
        self,
        x: pl.Tensor[[total_rows, 128], pl.FP16],
        w: pl.Tensor[[128, 32], pl.FP16],
        out: pl.Out[pl.Tensor[[total_rows, 32], pl.FP32]],
    ) -> pl.Tensor[[total_rows, 32], pl.FP32]:
        weight = pl.tile.load(w, [0, 0], [128, 32], target_memory=pl.Mem.Mat)
        right = pl.tile.move(weight, target_memory=pl.Mem.Right)
        a_mat = pl.tile.load(x, [0, 0], [first_rows, 128], target_memory=pl.Mem.Mat)
        b_mat = pl.tile.load(x, [first_rows, 0], [48, 128], target_memory=pl.Mem.Mat)
        a = pl.tile.move(a_mat, target_memory=pl.Mem.Left)
        b = pl.tile.move(b_mat, target_memory=pl.Mem.Left)
        a_out = pl.tile.matmul(a, right)
        out_a = pl.tile.store(a_out, [0, 0], out)
        b_out = pl.tile.matmul(b, right)
        out_b = pl.tile.store(b_out, [first_rows, 0], out_a)
        pl.system.fence()
        c_mat = pl.tile.load(x, [first_rows + 48, 0], [128, 128], target_memory=pl.Mem.Mat)
        d_mat = pl.tile.load(x, [first_rows + 176, 0], [128, 128], target_memory=pl.Mem.Mat)
        c = pl.tile.move(c_mat, target_memory=pl.Mem.Left)
        d = pl.tile.move(d_mat, target_memory=pl.Mem.Left)
        c_out = pl.tile.matmul(c, right)
        out_c = pl.tile.store(c_out, [first_rows + 48, 0], out_b)
        d_out = pl.tile.matmul(d, right)
        out_d = pl.tile.store(d_out, [first_rows + 176, 0], out_c)
        return out_d

p = passes.init_mem_ref()(PhaseArena)
p = passes.materialize_semantic_aliases()(p)
p = passes.memory_reuse()(p)
passes.allocate_memory_addr()(p)
```

建议沿现有 MemoryReuse 的容量压力 fallback 做有界改进，而不是把新根 size 伪造为整卡容量：

1. 原布局能够放下时保持原分配；原布局、固定 root 子区间重装箱均失败后，最多再尝试一次允许 compiler-owned、offset=0 root 增长的 pack。
2. 每个候选的冲突仍使用实际成员 `[offset,offset+size)` 和生命周期；保留 pipeline、hazard、pinned allocation、动态非 slice 别名等门槛。增长上限是空间容量减保留区。
3. `buffer.capacity = max(buffer.capacity, chosen_offset + candidate.size)`。向下游传递最终 arena 范围，AllocateMemoryAddr 必须按同 base 成员的最大静态 `offset+size` 预留，而不是只取最大成员 size。
4. 不应把本例归因于物理硬件不足，也不应一律减小数学 tile 或降低 pipeline 深度。

下面是本地 allocator 配套建议（基于 `cb2484471`，需配合上面 MemoryReuse 的增长逻辑；不是孤立补丁可修复所有问题）：

```diff
diff --git a/src/ir/transforms/allocate_memory_addr_pass.cpp b/src/ir/transforms/allocate_memory_addr_pass.cpp
index 05e6b0861..2ee8ad90a 100644
--- a/src/ir/transforms/allocate_memory_addr_pass.cpp
+++ b/src/ir/transforms/allocate_memory_addr_pass.cpp
@@ -13,6 +13,7 @@
 #include <any>
 #include <cstddef>
 #include <cstdint>
+#include <limits>
 #include <map>
 #include <memory>
 #include <optional>
@@ -320,7 +321,7 @@ class PinnedAllocCollector : public IRVisitor {
  * @brief Allocate memory addresses using the given allocation policy
  *
  * MemRefs sharing the same ``base_`` Ptr are co-located in a single bumped
- * slot sized by the largest member.size_ in the group.  This handles view
+ * slot sized by the largest static member end in the group. This handles view
  * MemRefs (e.g. produced by ``tile.slice``) — every view physically aliases
  * its parent allocation, so they should share one address slot rather than
  * each consuming size_ bytes of fresh L1.
@@ -373,14 +374,23 @@ std::vector<std::pair<const MemRef*, MemRefPtr>> AllocateMemoryAddresses(
     for (const Var* base_key : base_order) {
       const auto& group = base_groups.at(base_key);
 
-      // Slot size = max member.size_.  The root MemRef (byte_offset == 0) is
-      // sized to the full alloc; views are sub-regions and never exceed it.
+      // A grown reuse arena may contain windows extending past the original
+      // representative. Reserve their union, not just the largest tile size.
+      // Dynamic views remain covered by their root's complete static extent.
       uint64_t slot_size = 0;
       for (const auto& ref : group) {
         INTERNAL_CHECK_SPAN(ref->size_ > 0, ref->span_)
             << "AllocateMemoryAddr encountered zero-sized MemRef '" << ref->name_hint_
             << "'. InitMemRef should reject dynamic or invalid allocation shapes before address assignment.";
-        slot_size = std::max(slot_size, static_cast<uint64_t>(ref->size_));
+        uint64_t end = ref->size_;
+        if (auto offset = As<ConstInt>(ref->byte_offset_)) {
+          INTERNAL_CHECK_SPAN(offset->value_ >= 0 && static_cast<uint64_t>(offset->value_) <=
+                                                         std::numeric_limits<uint64_t>::max() - end,
+                              ref->span_)
+              << "Internal error: allocation window has an invalid static range";
+          end += static_cast<uint64_t>(offset->value_);
+        }
+        slot_size = std::max(slot_size, end);
       }
       // A declared allocation reserves what its alloc statement says, which for a
       // multi-slot declaration exceeds any single member. Everything else is sized
```

本地回归包含另一例前段 64+12 KiB 真实同时存活，仍须保持超限失败；不能因“后段可复用”忽略这一峰值。本轮 InitMemRef/L0 相关共 65 项通过，独立脚本的原/修改后对照也已完成。此前三参数真机数学回归曾逐项 max_abs=0；那是旧局部 arena 验证，不能代替新的完整 CSA 性能结论。

只补充 issue 的复现和建议，不创建 PR。
