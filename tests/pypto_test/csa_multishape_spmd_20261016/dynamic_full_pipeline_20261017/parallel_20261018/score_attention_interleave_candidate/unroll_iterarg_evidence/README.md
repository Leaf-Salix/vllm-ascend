# 外层 IterArg 在 unroll 深克隆时泄漏：最小 CPU 对照

2026-10-18。本次只隔离联合候选 v2 的一类 SSA 错误；未修改依赖、共享安装或 CSA 源码，未重编整层，也没有 PTOAS / NPU 执行。

结论：外层 `pl.range` 的 Scalar IterArg 直接传给含 `pl.unroll` 的 inline helper，会在 `UnrollLoops` 首次产生无定义的 IterArg 克隆。调用前增加两个普通 Scalar 局部别名后，同一最小程序的 SSA verifier 通过。它支持在真实 AIV executor 的 unroll helper 调用前建立 Scalar bridge；不能据此宣称联合候选的其余 while / if phi 问题已解决。

## 直接证据

`scope_kernels.py` 两例保留相同动态 Tensor 轴、由 `seed` inline tuple 返回的初值、固定 full-valid Vec Tile 的参数/返回/loop carry、两次 assemble 和相同输出。唯一差异是 `step_arg=step_iter`、`live_arg=live_iter` 两个显式 Scalar 别名；`audit_evidence.py` 删除它们并反向替换调用参数后，函数 AST 完全相同。没有 if、while 或 GM Tensor loop carry，避免混入另一类 phi 问题。

| 既有 / 最小产物 | Frontend | Inline | Unroll | CtrlFlow | SSA | SSA verifier |
|---|---:|---:|---:|---:|---:|---|
| 整 CSA 冻结 v2 的 `__FREE_VAR` 数 | 0 | 0 | 102 | 102 | 118 | 原报告228条错误 |
| 最小 direct | 0 | 0 | 4 | 4 | 4 | 8条作用域错误 |
| 最小 alias | 0 | 0 | 0 | 0 | 0 | PASS |

最小结果见 `cpu_v2/result.json`。direct 在 `02_unroll.py:18` 首次出现 `step_iter_1__FREE_VAR` / `live_iter_1__FREE_VAR`，而外层循环仍只定义原 `step_iter` / `live_iter`。8条 verifier 错误包括新造 IterArg 及其 init 值的作用域；不是8个独立根因。alias 的同一位置引用已定义的 `step_arg` / `live_arg`。

整 CSA 的第一处泄漏是 Score `single_segments` assemble offset 中的 `step_iter_2__FREE_VAR`；有限片段与完整文件路径/大小保存在 `AUDIT.json` 和 `full_v2_*_first_free.txt`。整层出现更多错误，不能把全部228条归因于这一个机制。

## 源码机制与限制

隔离依赖 `.worktrees/pypto-csa-static-fixes-20261016`：

- `src/ir/transforms/unroll_loops_pass.cpp:88–99` 对 unroll body 使用 `DeepClone(..., clone_def_vars=false)`；seed 映射只有本层 unroll induction variable。
- `src/ir/transforms/utils/deep_clone_utils.cpp:92–98` 对未映射的普通 Var 原样返回；`:101–115` 对未映射 IterArg 却创建 fresh IterArg，并克隆其 init/type。克隆的外层 IterArg 不在当前 body 中定义，因而成为无定义引用。
- `python/pypto/language/parser/ast_parser.py:2760` 是最初建立 IterArg 的位置；错误把该 span 列为 Location，不足以说明 parser 是故障发生阶段。

精确片段保存在 `DEPENDENCY_EXCERPTS.txt`。本次只确认现有二进制对应行为和源码规避，不修复 DeepClone，不验证任意 TileView / 动态 valid carry，也不推断后续资源规划、同步或数值正确性。bridge 必须放在含 unroll helper 的调用之前；放在 unroll body 内仍会克隆其 IterArg 右值。

## 执行和失败记录

```bash
bash run_cpu.sh --output <本目录>/cpu_v2
python audit_evidence.py
```

CPU 复现只运行 specialize → InlineFunctions → UnrollLoops → CtrlFlowTransform → ConvertToSSA → SSA verifier，复用既有 Tile 注解 overlay 与隔离依赖二进制；不提供样例 shape。`cpu_v1` 是 fixture 把 `set_validshape` 错写成 list 参数后的 parse 失败，两侧均未进入 pass；原源码、完整错误及日志保留。v2 仅将其改成现接口的独立 `valid_rows, valid_cols`，没有调整被研究的循环。

原整 CSA v2 的622383862字节 result与311244311字节日志原样保留，只记录路径/大小，没有复制进本小归档。原报告路径见 `AUDIT.json`。`ARCHIVE_FILES.txt` 是本目录小文本归档 allowlist。

主会话随后执行的整链 v3 已通过 SSA，继续到 AllocateMemoryAddr 后因 `Left buffer usage (77824 bytes) exceeds platform limit (65536 bytes)` 失败，见 `FULL_CHAIN_V3_LINK.json`。v3 同时包含其他代理的 while / if phi 源码修正，不能把整链 SSA 通过全归功于两个 alias。这个结果尚不是整层编译通过，Left 资源问题由另外两位代理处理。本代理没有再编译整层。

没有改 compiler 的 `DeepClone` / `UnrollLoops`；当前仅以源码 Scalar alias 规避，原始组件缺陷仍存在。

## 编译错误输出限长

联合候选私有 `check_cpu.py` 已改为调用同目录 `cpu_failure_report.py`；`prepare_runners.py` 同步该有限变换。原始异常全文只写 `failure_error_full.txt` 一次；栈单独写 `failure_traceback_stack.txt`，避免再附加全文。result/stdout只保留4096字符预览和各最多64种、每种最多512字符的 Message / Location 计数，并记录完整文件路径/字节数。`CPU_FAILURE_REPORT_CHECK.json` 的纯标准库合成检查通过；旧大日志没有删除或覆盖。这项改动不改变编译控制、源码和设备检查。

主归档目录为联合候选根 `unroll_iterarg_evidence/`；`archive_evidence.py` 只按小文本 allowlist 复制，不携带原整层 IR / 二进制 / cache。
