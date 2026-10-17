# 静态 CSA 的完整编译现状与接续方法

> 2026-10-16 更新：用户已授权独立 PyPTO worktree 修复；CORE 扩容、
> 共享 InOut 单 MIX 小例子真机执行均已通过；完整 B4 v21 默认 PYPTO 代码生成
> 已通过。完整 Torch 注册新遇到 64 入参限制，见
> [注册限制与入口方案](TORCH_SCHEMA_CAPACITY_20261016.md)。尚无完整设备核结果。
> 后续 B4 program 产物的 CCE/调度二进制编译及组装也已通过，见
> [二进制编译记录](BINARY_COMPILE_20261016.md)。
> 本文以下为修复前诊断与接续约束，
> 最新证据见 [独立 PyPTO 修复记录](PYPTO_STATIC_FIXES_20261016.md)。

本文件针对 `static/*/entry.py` 的融合原型。它与原动态 CSA 的真机 PASS、
PyPTO #3009 的最小复现分别记录，不能互相替代。没有静态 CSA 真机运行结果。

## 已准备好的接口

`prepared_arguments.PreparedStaticArguments(case_directory, native_call.args)`
直接消费原 `NativeCSACall` 的 56 份 Tensor。初始化建立一个 INT8 workspace，
在 Host 建立其各 dtype 的别名视图；Native metadata/cache/state 和四份 NZ
权重保持原存储。初始化时清零 progress、上传 records/waits，重放不重新初始化。
无每步 D2H、权重转换或 metadata 展开 kernel。2026-10-16 已在独立 Simpler
worktree 完成 256 Tensor 扩容与设备验证，见 [扩容记录](SIMPLER_CORE_EXPANSION_20261016.md)。

最终测试原型把全部具体 GM 视图作为根入口和核参数：长 history 为 132 项，
短 history 为 133 项。早期 57 项 arena 根入口不是现行接口；它依赖的
orchestration dtype reinterpret 不被当前运行时支持。保留生成器
`--workspace-binding arena` 只供复现该限制。

七档 `static/argument_binding_result.json` 使用 **meta Tensor** 核对 shape、
dtype、storage 别名、NZ 对象身份及 32 参数容量拒绝。没有加载权重数据、
没有设备格式验证，也没有精度或性能结论。实际 NPU 格式检查应在任务中执行。

同一准备对象只适合固定 Native allocation 地址和已记录的形状/编排分支；
未来测试须先用已有 Host metadata 匹配模板，再构造对象。不同流或并发调用
各用自己的 workspace。此对象不注册/调用算子；共享安装仍为旧容量，未来实际
运行必须绑定扩容 worktree 的 Python、native binding、runtime 和新编译核产物。

## Program 模式完整编译证据

以下目录均在 `../results/csa_static_spmd_feasibility_20261015/` 下。

| 记录 | 实际到达位置 | 结果 |
| --- | --- | --- |
| `static_b4_cpu_v15` | PYPTO 的 `AllocateMemoryAddr` | Left 81920 B 超过 65536 B |
| `static_b4_cpu_v16_dsa_rp` | DSA_RP 容量规划 | canonical greedy 未找到容量内布局 |
| `static_b4_cpu_v17_ptoas` | PyPTO codegen | 早期 arena 根入口的 reinterpret 不支持；Group 返回别名未正确表达 |
| `static_b4_cpu_v18_external` | PyPTO codegen | 前两处本地表达已修正，遇到 `set_validshape` 缩窄 tile view 限制 |
| `static_b4_cpu_v19_valid_view` | PTOAS 解析生成 MLIR | qk_pv AIV 中两处 SSA 值的定义不支配使用点 |
| `static_b4_cpu_v21_buffer_ir` | Buffer IR 存储合法化 | 原型的多 Tile 元组赋值尚未展开 |
| `static_b4_cpu_v22_buffer_ir` | PTOAS 检查生成 MLIR | 元组已逐项展开；剩余 Mat→Mat 及不匹配 shape 的 tmov 被拒绝 |
| `static_b4_cpu_v23_flat_phi` | 不启用 Buffer IR 的 PTOAS 路径 | 逐项赋值后仍有两处 SSA 定义不支配使用点 |

v17 的修正是外部初始化绑定 typed view，以及让 Group 返回自己的参数身份，
不把 AIC 返回的 SSA 临时名误传给 AIV。v18 的 view 有效范围在 v19 中改写为
零偏移、相同物理 shape 的 `tile.slice(..., valid_shape=...)`，不修改数学、
tile 数据或 pipeline 级数。生成器记录每个核体的此类转换数量。

v19 的 `kernel/ptoas/csa_static_group.pto` 第 13128、13131 行引用
`%stage_32_v97__ssa_v0`、`%stage_32_v123__ssa_v0`；其定义在前面的条件区域中。
这是 PTOAS 对输入 MLIR 的解析拒绝，不能据此认定是 PTOAS 算错了。应从
PyPTO 的 region/alias lowering 以及打印 IR 再入前端的方式核对，尚未用独立
最小用例确认为组件 bug。失败原始 MLIR、passes 和错误全文均保留。

默认规划器的容量审计见 `static/left_memory_audit.json`：融合 AIC 得到两组
Left 复用槽，最大尺寸分别为 49152、32768 B，总计 81920 B。前者的最大使用
来自 Indexer score/top-k，后者来自 O-proj A。原独立核最大的对应 Left 总量
为 65536 B。**这提示需要核对融合后的生命周期和复用分组，不能据此宣布 A3
硬件无法执行。** 原独立核与融合核不是同一个分配问题。

本次没有为了通过容量检查缩小计算 tile、减少流水级数或跳过容量验证。
对 NZ 核体重新走了原有 lowering；其最终分块和资源分配仍须与原核产物对照，
不能仅凭源代码算式相同认定最终指令完全相同。

进一步将两段提取为 13 参数的独立编译复现：Indexer 与 O-proj 各自代码生成
通过，融合后 Left/Right 都是 81920 B，PYPTO 与 DSA_RP 均失败。详见
[MEMORY_REUSE_PHASES_REPRO.md](MEMORY_REUSE_PHASES_REPRO.md)。

Buffer IR 的 v21 暴露了本地生成器的 Tile 元组表达问题；v22 将分支末尾赋值
逐项展开，并断言输出名与 RHS 名不相交，保持并行赋值语义。它通过存储合法化后，
生成的 `tmov` 仍包含 A3 不支持的 Mat→Mat，以及非 Mat 源/目标 shape 不一致。
这些被 PTOAS 正常拒绝，不能当作硬件运行错误或已完成编译。
v20 是本地脚本的 PassContext 参数配置错误，已修正，不算组件失败证据。

## 提交结构与 while 状态

`static/b4_ir_audit.json` 核对最新 v23 的部分产物：

- orchestration 恰好一处 `rt_submit_task`；`MixedKernels`，24 组，sync_start=true。
- pass25 中 74 个进度轮询均保留 initial、condition、yield 和 loop carry；
  无 `__FREE_VAR`。没有添加循环后的伪使用。
- 这是显式 Group 的 AIC/AIV 两半路径；不能声称自动 MIX 拆分的 #3009 已修复。
- 整体 codegen 仍失败，未生成可运行完整算子；单次提交结构成立不等于功能 PASS。

## JIT 入口仍需解决的表达问题

同源 `entry_jit.py` 已生成。检查方式是 JIT 源码的 `entry.compile()`，其产物
模式为 **program**，没有调用 `pypto.torch.init` 或启动 kernel runtime。

`static_b4_jit_cpu_v1` 的 NZ 签名 TensorView 包装已改为等价的 `pl.NZ`；
v2 的旧局部 Tensor annotation 中未绑定 dynamic 名字，已改由原 tensor op
与完整实参类型推断。最新 v3 到达 SPMD 结构检查，拒绝 scope 内两个 InCore
调用。不能把这份 JIT 源码标成已经可接 `pypto.torch.register`。

当前 program 原型用显式 `FunctionType.Group` 作为 SPMD 的单一 callee。
JIT 的 `pl.cluster` 与外层 SPMD 组合不能直接等同于这个显式 Group；
反过来在一个 cluster 中放两个 SPMD 也被 outliner 明确禁止。接续应选用组件
正式支持的单 Group JIT 调用表达，或在 #3009 修复后验证自动 MIX 的内联表达。
不加入无意义 block_idx 读取欺骗 dispatch 检查，不拆成两次运行时提交。

另以 5 参数、无 while 的小例子核实了 opaque wrapper 备选表达；它到 codegen
仍因 Opaque 不具备 AIC/AIV/Group 类型而失败，见
[JIT_SINGLE_GROUP_NOTES.md](JIT_SINGLE_GROUP_NOTES.md)。

最新 inline wrapper 的非共享 InOut 小例子已完整 CPU 编译和 Meta 注册通过。
B4 生成器采用 inline wrapper 后，v6 已通过 SPMD 和 InOut 使用检查，之后
在 Group ABI 统一中暴露共享 InOut 别名问题。5 参数最小复现的首个破坏点是
`ExpandMixedKernel`，详见 [PYPTO_GROUP_SHARED_INOUT_REPRO.md](PYPTO_GROUP_SHARED_INOUT_REPRO.md)。
不要继续把入口问题归结为必须拆两次提交，也不要将小例子结果当作完整 CSA 通过。

## 恢复顺序

1. 参数容量已完成。使用独立 Simpler 扩容版本及已有 binding 清单，不重复采集。
2. 与用户确认 PyPTO 修复边界后，使用 B4 源码处理完整编译和共享 InOut 限制；
   保留原分块的两阶段 DSA 地址分配诊断已经通过，见内存复现文档；不能用临时
   改写编译 pass 或丢弃资源硬约束作为生产修复。检查最终片上容量、分块、
   while 状态、内部 FFTS 事件、一次 MIX 提交。先不要重编七档完整算子。
3. 通过 task-submit 验证最小跨组 flag 可见性/代际，再做 B4 原动态/静态功能、
   cache/state/保护区对比及重放。CPU 随机推进模型不能替代这一步。
4. 功能稳定后扩展七档精度，最后按既有 CSA compiled 测试方法及 L1 incore
   首尾口径比较性能。DEBUG 补采不是历史最佳性能的证据。

CPU 命令沿用 README 的环境初始化，以下命令不调用设备：

```bash
python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/compile_static_entry.py \
  --source tests/pypto_test/csa_static_spmd_feasibility_20261015/static/h131072_b4/entry.py \
  --output /tmp/csa_static_after_fix --memory-planner PTOAS

python tests/pypto_test/csa_static_spmd_feasibility_20261015/audit_static_ir.py \
  --compiled /tmp/csa_static_after_fix \
  --layout tests/pypto_test/csa_static_spmd_feasibility_20261015/static/h131072_b4/layout.json \
  --output /tmp/csa_static_ir_audit.json
```

编译脚本今后在输出目录保留输入源码副本，避免后续生成器更新使失败行号失去
对应关系。已有 v15～v19 的实际 IR 与错误必须以各自 passes/MLIR 为准。
