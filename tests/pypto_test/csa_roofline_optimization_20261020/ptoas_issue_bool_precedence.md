## 问题与影响

在本地 LLVM 19 配套构建中，合法的嵌套 `arith.andi/ori` 被 EmitC 输出为错误的
C++ 运算分组，编译成功但改变结果。独立源码基线 `e5c1c432` 和其上仅增加
Scalar gather_compare 支持的候选均复现；不是 compare gather 修改引入。
未据此宣称官方所有 LLVM 构建或 release 包都存在同一问题。

实际 Top-K 调用中，这会绕过“保留数不超过1024”的分支条件，允许更大的输入
进入512/1024容量路径。真机探针出现UB越界；下面的纯Scalar最小复现已经
独立证明条件语义错误，修复后还须重跑真机，不能仅据它排除其他越界原因。

## 不需要 PyPTO/NPU 的最小复现

```mlir
module {
  func.func @mixed_bool(%a: i1, %b: i1, %c: i1, %d: i1, %e: i1) -> i1 {
    %ab = arith.andi %a, %b : i1
    %cd = arith.ori %c, %d : i1
    %x = arith.andi %ab, %cd : i1
    %y = arith.andi %x, %e : i1
    return %y : i1
  }
}
```

```bash
ptoas --pto-arch=a3 mixed_bool.pto -o mixed_bool.cpp
```

实际生成：

```cpp
return (v1 & v2 & v3 | v4) & v5;
```

正确语义应为：

```cpp
return (v1 & v2 & (v3 | v4)) & v5;
```

将生成函数原文取出，只去掉 `AICORE` 修饰后在CPU执行全部32组布尔输入，
**6组错误**。例如 `a=0,b=0,c=0,d=1,e=1`：期望0，实际1。
不需要设备前言、atomic、缓存同步、模型或权重。

## 定位与建议

所用 LLVM 19 的 `CppEmitter::emitOperand` 在为子表达式加括号后，将内部
precedence设为最低值，而非该子表达式自己的precedence，导致更深层子树
所需的括号丢失。

PTOAS本地已有 `FormEmitCExpressionsCompatPass` 对三元表达式的显式临时量
保护，但没有覆盖位或/位异或与逻辑组合。建议：

1. 根本修复是使用正确恢复表达式precedence的MLIR C++ emitter，或升级至
   已修复版本；须实际确认，不以版本号替代本最小复现。
2. 在需要兼容旧LLVM的PTOAS路径上，可暂时将包含低优先级运算的producer保留
   为普通SSA临时量，避免错误嵌套。它们不是volatile，后续C++编译器仍可优化。

本地候选将现有检测扩展为：

```cpp
if (isa<emitc::ConditionalOp, emitc::BitwiseOrOp, emitc::BitwiseXorOp,
        emitc::LogicalAndOp, emitc::LogicalOrOp>(op)) {
  return true;
}
```

调用方沿用 `producer.setDoNotInline(true)`。这是建议补丁，不是修改PR；
独立重建已完成；生成C++的i1真值表32项和i32位运算625项合计657项全部通过，
新增FileCheck及原三项gather相关回归也通过。完整设备探针尚待重跑结果。
建议至少覆盖i1的32组真值表、i32混合位运算、原三元表达式与CFG状态回归，
并将生成C++实际执行，不只做编译或文本检查。


## 本地修复后的设备补充（2026-10-20）

同一Top-K helper的Python AST保持不变，重建隔离PTOAS后重新生成设备代码。
59组有效长度/保留数/同分/负阈值/旧根不足的用例，在1024→2048→1024三个
随机seed下同地址ACLGraph回放全部通过：score/index逐bit一致、输入不变、
两侧保护区保持。修复前任务在首次调用发生UB越界，修复后无该异常。

这支持分支错误是该探针故障的原因；不代表所有生成式、所有LLVM版本或完整
CSA都已经完成验证。完整B16/B28对照还在进行，不预记其结果。修复及测试仅
在本地独立worktree，没有更改共享安装、提修改PR或添加标签。
