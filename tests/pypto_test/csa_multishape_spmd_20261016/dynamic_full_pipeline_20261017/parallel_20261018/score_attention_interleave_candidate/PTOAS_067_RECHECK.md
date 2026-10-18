# PTOAS 0.67 与 PyPTO PR #2962 的定点核对

核对时间：2026-10-18。用户要求确认新版是否已经修复动态 CSA 交错流水遇到的
编译问题。本次下载并隔离安装官方 0.67 wheel，实际执行已有最小复现，未修改共享
安装，也没有为版本核对新增 NPU 任务。

## 结论

| 问题 | 官方 PTOAS 0.67 结果 | 所属组件与处理 |
| --- | --- | --- |
| 循环回边的并行赋值被串行覆盖 | 20 个 CPU 数值用例中 12 个失败，与 0.66 一致 | PTOAS 的 CFG → C++ 降低；本地修复仍有必要 |
| 负数 `FloorMod` 被当作截断余数 | 旧 PyPTO + 0.67 的 10 项用例全部失败 | PyPTO 已生成 `arith.remsi`，PTOAS 无法恢复已经丢失的 floor 语义 |
| 本地 PyPTO 的 FloorMod 修正 | 修正 PyPTO + 0.67 的同一组 10 项全部通过 | 0.66 同组也已通过，不依赖升级到 0.67 |

这不是 PR #2962 的全面评审，也不是 FP4 或整层 CSA 的精度/性能验收。没有把
PTOAS 新版本号、PR 描述或 CI 通过当作本问题已修复的证据。

## 核对的精确版本

- [PyPTO PR #2962](https://github.com/hw-native-sys/pypto/pull/2962)：本次查看的
  head 为 `1c089c9ad09d83171d461a4212732f6b8afd90af`，查询时仍未合入。
- [PTOAS v0.67](https://github.com/hw-native-sys/PTOAS/releases/tag/v0.67)：
  release 指向 `426b676ef57abdd72036a6faaabe848eae89a6e8`，使用官方
  `ptoas-0.67-cp310-cp310-manylinux_2_34_aarch64.whl`。
- 隔离安装：工作区根下 `.cache/dsv4-toolchain/ptoas-0.67/`，
  `bin/ptoas --version` 实际返回 `ptoas 0.67`。
- PyPTO 旧对照：`.worktrees/pypto-csa-static-fixes-20261016`；修正对照：
  `.worktrees/pypto-csa-memory-reuse-20261018`，使用各自实际核心库。

版本标识用于准确定位源码，不进行文件 hash 校验。下载 wheel、host 可执行文件和
测试 `.so` 仅本地保留，不能提交到本仓库。

## 一、循环回边：为什么仍然错误

同一个 `scf.yield` 的所有值必须按旧状态求值。例如：

```text
(iteration, current, previous) <- (iteration + 1, current + 48, current)
```

初始值均为零，迭代两轮应得到 `current=96, previous=48`。0.67 实际生成的
C++ 仍按如下顺序覆盖变量：

```cpp
v8 = (int64_t) ((uint64_t) v10 + (uint64_t) v5);
v9 = v7;
v10 = v8;
v11 = v10;
```

其中 `v10` 是 current，`v11` 是 previous；最后一行读到已经更新的 current，
实际结果是 `(96, 96)`。三变量轮换与条件交换也受到影响。

源码核对：[v0.67 materializeControlFlowOperands](https://github.com/hw-native-sys/PTOAS/blob/v0.67/tools/ptoas/ptoas_cpp_rewrite.cpp#L480)
只为 `emitc::ExpressionOp` 物化临时值，未保护作为分支实参的 `BlockArgument`。
本地候选修复在独立 PTOAS worktree 中扩展这一保护。独立原生构建现已完成，
新二进制编译同一最小 IR 后，生成的 Scalar C++ 在 CPU 执行 **20/20 通过**，
见 `ptoas_parallel_copy_repro/fixed_host_v1/result.json`。构建源码基线为
`f5eff3ee249697f6157088f649c6434fcc9d7c5b`，本地二进制版本仍显示0.66，
不是官方0.67 wheel。此回归不证明完整CSA或NPU同步正确；共享安装未改。
动态 CSA 的已验收轮仍通过重排等价的 loop carry 避开覆盖，没有使用这个新二进制。

证据：

- `ptoas_parallel_copy_repro/parallel_copy.pto`：包含 shift、三变量轮换、条件交换。
- `ptoas_parallel_copy_repro/parallel_copy_v067.cpp`：官方 0.67 原始输出。
- `ptoas_parallel_copy_repro/v067_host_v1/result.json`：20 例逐例实际值/期望值，12 例失败。
- `ptoas_parallel_copy_repro/old_host_v1/result.json`：0.66 的相同失败对照。

纯 Scalar C++ 数值复现：在本目录执行，`PTOAS_ROOT` 设置为上述隔离安装目录。
输出目录必须是新目录。`--expect-bug` 表示确认已知失败，不表示算子正确。

```bash
"$PTOAS_ROOT/bin/ptoas" ptoas_parallel_copy_repro/parallel_copy.pto -o /tmp/csa_parallel_copy_067.cpp
python ptoas_parallel_copy_repro/execute_scalar_cpp.py /tmp/csa_parallel_copy_067.cpp \
  --output /tmp/csa_parallel_copy_067_result --expect-bug
```

harness 仅替换设备前言及屏障为空实现，保留生成函数的全部赋值与控制流原文。
这足以验证上述标量状态变换，不验证 NPU 缓存一致性、FFTS 或流水性能。

## 二、负数取模：问题发生在进入 PTOAS 之前

PR #2962 head 的
[`PTOCodegen::VisitExpr_(FloorMod)`](https://github.com/hw-native-sys/pypto/blob/1c089c9ad09d83171d461a4212732f6b8afd90af/src/codegen/pto/pto_scalar_expr_codegen.cpp#L198)
仍直接调用 `VisitBinaryArithExpr(op, "arith.remsi", "arith.remf")`。
例如 Python/PyPTO floor 语义要求 `-97 % 24 == 23`，截断余数为 `-1`。
用于全局 owner 对齐时，`begin + (worker - begin) % 24` 会选到 begin 之前的
query，导致参与者计算/等待次数不一致。

测试包含 INT32/INT64/INDEX、正负静态 divisor、动态 divisor、边界值和实际
query owner 表达式。生成的 C++ 在 CPU 原样编译执行，旧/新 PyPTO 都使用
同一个官方 PTOAS 0.67。数值结果：

- `pypto_floor_mod_repro/old_core_ptoas067_v1.log`：10 failed，1.72 s。
- `pypto_floor_mod_repro/new_core_ptoas067_v1.log`：10 passed，1.76 s。
- 对应子目录保存 provenance、JUnit、逐例 `.pto/.cpp`；host 二进制不入 Git。

本地修正仅针对 signed integer 与 INDEX 的 FloorMod：先计算截断余数，余数非零且
符号与 divisor 不同时加 divisor。常量 divisor 可直接确定所需符号条件；其它
类型保留原路径。本轮没有修复负数 FloorDiv，也不扩大到所有除法语义。

PR 中的 FP4 载体/别名/控制流描述符传播，以及 all_to_all_v 同缓存行计数发布修复，
均不能替代上面两个标量控制流修复。all_to_all_v 解决的是多写者计数发布竞态；
本次循环变量覆盖即使在单 CPU 线程也能复现。

## 三、动态 CSA 验证的关联边界

采用本地 FloorMod 修正与 loop carry 源级规避的任务
`task_20261018_103107_374461711536`（128K/B16/S6）已从设备调用正常返回。
此前两轮核间等待卡死没有在这一轮重现，但测试随后在协议检查器失败：
`group_plan[6]` 的 true 被编译成 INT32 `-1`，检查器原来只接受 `1`。

原始快照 `interleave_floor_mod_h131072_b16_v3/failure_snapshot.json` 保存两波完整
ROOT/TOPK/PLAN 发布。`check_saved_wave_snapshot.py` 离线检查实际写者全部匹配，
也验证 0/2/-2/37 等错误标记仍被拒绝。这个核对不追改失败轮为 PASS；当轮数值
比较结果未及时落盘，完整精度与性能仍须重提验证。新检查器保留实际写者、代际和
所有保护区校验，并在协议断言前先保存已经完成的数值比较结果。

后续任务 `task_20261018_103852_401858324030` 已提交；其最终结果另记，本文不预记通过。

### 后续结果与上游提交

上述 v4 后来完成 PASS：A/B/A 的输出、Top-K/score、完整 cache/state 与保护区、
信号写者/代际及基本图重放全部通过。5窗完整 L1 中位数：原多task 877.84 μs，
动态候选889.96 μs，静态参考787.30 μs。仅表示128K/B16/S6这一轮功能通过，
动态候选仍慢于原多task，没有达到性能目标。

用户授权直接将组件缺陷提交上游后，PTOAS问题已提交为
[#1622](https://github.com/hw-native-sys/PTOAS/issues/1622)。正文内的最小IR另行
编译执行，limit=1/2分别错误输出48/48与96/96；远端正文已逐行核对一致。
首次按pass模板创建因仓库缺少`ir-pass`标签失败，读取实际issues列表确认未创建，
随后使用已有通用Bug模板成功创建；保留两份模板正文，未重复创建issue。
PyPTO FloorMod修复另从最新主线14736f65c建立干净本地分支并独立构建。
用户随后明确改为只提交issue与修改建议，不直接提PR；本地代码未推送、未创建PR。

### Issue中的代码建议与验证边界

2026-10-18 已将最小修改diff直接补入以下四份issue正文，并读取远端正文核对：

- [PyPTO #3026](https://github.com/hw-native-sys/pypto/issues/3026)：signed/INDEX
  FloorMod语义修正；10项数值回归通过，相关codegen共176项唯一用例通过
  （首次175项，修正多缓冲测试的SSA依赖检查后剩余1项通过）。
- [PyPTO #3027](https://github.com/hw-native-sys/pypto/issues/3027)：原地结果重绑定后
  将真实MemRef继续传播到下游view，保留窗口、slot与未决phi语义。
- [PyPTO #3028](https://github.com/hw-native-sys/pypto/issues/3028)：两处内存别名
  递归入口补充While子树遍历，不新增无条件合并内存的规则。
- [PTOAS #1622](https://github.com/hw-native-sys/PTOAS/issues/1622)：CFG边赋值前
  快照源BlockArgument；旧官方0.66/0.67各12/20失败，本地新构建20/20通过。

PyPTO两项内存修改在最新主线的原实现上新增10例有6失败、4通过；二者合用的
独立构建通过完整 `test_memory_reuse.py` 的154项。issue中的diff已分别拆开，
没有把联合回归冒充拆分补丁分别跑过全套，也没有把旧基线158项与新主线154项混用。
非零boxed Mat的通用物理偏移及完整NPU设备行为仍不在这组IR验证的结论内。

按用户最新要求不添加标签、不创建修改PR。四份正文、补丁及远端回读快照保存在
`upstream_issues_20261018/`，结果索引为 `ISSUE_DIFFS_UPDATED.json`。
