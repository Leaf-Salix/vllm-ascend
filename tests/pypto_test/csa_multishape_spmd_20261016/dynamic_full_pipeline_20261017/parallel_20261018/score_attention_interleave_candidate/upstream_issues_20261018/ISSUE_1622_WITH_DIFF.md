### Component

EmitC / Codegen (lib/PTO/Transforms/PTOToEmitC.cpp)

### Description

循环回边的并行赋值在 CFG → C++ 降低中被串行覆盖，静默改变合法 SSA 程序的计算结果。官方 PTOAS 0.66/0.67 均可复现。

### Reproduction (minimal)

```bash
ptoas --version
ptoas shift_prev.pto -o shift_prev.cpp
```

分别使用官方 PTOAS 0.66 与 0.67，均编译成功但生成错误的状态更新。

保存为 `shift_prev.pto`：

```mlir
module attributes {pto.target_arch = "a2a3"} {
  func.func @shift_prev(%out: !pto.ptr<i64>, %limit: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %step = arith.constant 48 : index
    %fi, %fc, %fp = scf.while (%i = %zero, %current = %zero, %previous = %zero)
        : (index, index, index) -> (index, index, index) {
      %keep = arith.cmpi slt, %i, %limit : index
      scf.condition(%keep) %i, %current, %previous : index, index, index
    } do {
    ^bb0(%ib: index, %cb: index, %pb: index):
      %ni = arith.addi %ib, %one : index
      %nc = arith.addi %cb, %step : index
      scf.yield %ni, %nc, %cb : index, index, index
    }
    %c64 = arith.index_cast %fc : index to i64
    %p64 = arith.index_cast %fp : index to i64
    pto.store %c64, %out[%zero] : !pto.ptr<i64>, i64
    pto.store %p64, %out[%one] : !pto.ptr<i64>, i64
    return
  }
}
```

### Expected behavior

同一次 `scf.yield` 的所有右侧值都来自旧状态。更新等价于 Python 的：

```python
i, current, previous = i + 1, current + 48, current
```

初态均为零。`limit=2` 时输出应为 `[96, 48]`；`limit=1` 应为 `[48, 0]`。
生成 C++ 必须先保存旧 current，或采用保持并行拷贝语义的赋值顺序。

### Actual behavior / error logs

官方 0.67 生成的回边如下，v10/v11 分别为 current/previous：

```cpp
v8 = (int64_t) ((uint64_t) v10 + (uint64_t) v5);
v9 = v7;
v10 = v8;
v11 = v10;  // 读到了已经更新的 current
goto label2;
```

`limit=2` 实际输出 `[96, 96]`，`limit=1` 为 `[48, 48]`。
不是编译器报错，而是静默生成错误代码。

已将生成函数的算术与控制流原文编译为 CPU C++ 执行：仅去掉设备前言，并将
pipe_barrier/dcci/dsb 设为空函数。shift、三变量轮换及条件交换共20例，0.66与
0.67均12例数值失败。这个最小复现不需要 PyPTO、NPU、atomic 或缓存一致性。

### Git commit

官方 v0.67 release：`426b676ef57abdd72036a6faaabe848eae89a6e8`。
测试包为 `ptoas-0.67-cp310-cp310-manylinux_2_34_aarch64.whl`；`ptoas --version` 返回 `ptoas 0.67`。
0.66 官方安装也复现相同错误。

### Host platform

Linux (aarch64)

### Target Ascend arch (if relevant)

a3

### PTOAS build level (if relevant)

N/A

### Additional context

定位到 [v0.67 的 materializeControlFlowOperands](https://github.com/hw-native-sys/PTOAS/blob/v0.67/tools/ptoas/ptoas_cpp_rewrite.cpp#L480)：
当前只为 `emitc::ExpressionOp` 分支实参保存临时值；循环 carry 作为 `BlockArgument`
传入时没有同样保护。后续 C++ 顺序写回会覆盖仍需使用的旧值。

建议在写入目标 block arguments 之前快照需要保留的源值，或实现完整的 parallel-copy
lowering。必须覆盖链式旧值依赖、交换/轮换形成的环，以及 `cf.cond_br` 两条边；
不能只依赖重新排序来解决所有情况。

本地最小候选是在该函数现有临时值物化逻辑中，将条件从只接受 ExpressionOp 扩展为
ExpressionOp 或 BlockArgument。独立源码构建现已完成，新二进制的20项Scalar数值回归
全部通过；具体diff、源码版本和验证边界见下方补充，不代表完整CSA设备验证。

实际动态 CSA 单 SPMD 中也观察到 AIC/AIV 两个 wave 回边覆盖旧 query_begin，导致
消费者等待没有对应生产者的计算。当前算子仅通过等价重排 carry 临时规避；修复
应位于 PTOAS，使合法 SSA 程序不必依赖变量排列顺序。完整 CSA 还有独立的 PyPTO
负数 FloorMod 问题，这份最小复现已经排除它。

### 建议代码 diff 与最新验证（2026-10-18，仅供参考，未提交 PR）

以下补丁基于本地独立 PTOAS worktree 的
`f5eff3ee249697f6157088f649c6434fcc9d7c5b`，文件是
`tools/ptoas/ptoas_cpp_rewrite.cpp`。新构建的本地二进制 `--version` 仍显示0.66，
不是官方0.67 wheel；官方0.66和0.67作为负对照各有12/20失败。

候选通过现有临时变量物化路径，在CFG边写入目标block arguments之前快照源BlockArgument。
这种保守实现可能增加临时变量；后续可以按真正的覆盖依赖减少快照，但必须保持parallel-copy
语义，不能依赖简单调整赋值顺序处理交换/轮换。

独立源码构建已完成，使用新二进制重新编译同一份最小IR，再原样执行生成的Scalar C++：
**20/20通过**。覆盖shift、三变量轮换、条件交换，迭代次数0/1/2/3/5，条件交换两条分支。
这是编译后Scalar数值回归；仅去掉设备前言、空实现屏障，未改生成的赋值和控制流。
它不验证NPU内存一致性、完整CSA功能或性能，也没有替换共享安装。

下面仅列错误修正，不包含本地原生CLI构建辅助改动。建议组件侧另跑官方编译回归，
尤其覆盖 `cf.br`、`cf.cond_br`、链式旧值依赖及环形并行拷贝。

```diff
diff --git a/tools/ptoas/ptoas_cpp_rewrite.cpp b/tools/ptoas/ptoas_cpp_rewrite.cpp
index 648c2c37..271d7444 100644
--- a/tools/ptoas/ptoas_cpp_rewrite.cpp
+++ b/tools/ptoas/ptoas_cpp_rewrite.cpp
@@ -477,6 +477,12 @@ static Type getEmitCVariableStorageType(Type valueType) {
 // emitter prints cf.br/cf.cond_br operands by variable name rather than by
 // recursively emitting an expression. Materialize such operands so CFG-based
 // lowering (e.g. scf.while -> cf.*) stays valid.
+// The emitter also assigns successor block arguments one by one. A branch
+// operand that is itself a block argument can alias an earlier destination:
+// (current, previous) <- (next, current) would incorrectly become
+// current = next; previous = current. Snapshot block-argument operands before
+// any edge assignment to preserve the parallel-copy semantics, including
+// swaps, cycles, and conditionals with operands on either successor edge.
 void materializeControlFlowOperands(Operation *rootOp) {
   llvm::SmallVector<Operation *, kBranchInlineCapacity> branches;
   rootOp->walk([&branches](Operation *op) {
@@ -491,7 +497,7 @@ void materializeControlFlowOperands(Operation *rootOp) {
     for (OpOperand &operand : op->getOpOperands()) {
       Value value = operand.get();
       auto expr = dyn_cast_or_null<emitc::ExpressionOp>(value.getDefiningOp());
-      if (!expr) {
+      if (!expr && !isa<BlockArgument>(value)) {
         continue;
       }
```
