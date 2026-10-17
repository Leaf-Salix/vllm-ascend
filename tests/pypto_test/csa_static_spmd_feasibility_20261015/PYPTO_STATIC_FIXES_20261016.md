# CSA 静态融合：独立 PyPTO 修复记录（2026-10-16）

用户已授权“另开 PyPTO worktree 修复”。本轮所有组件修改只在独立 worktree，
不修改共用 PyPTO/Simpler checkout，不替换全局安装，不改 PTOAS、PTO-ISA、pypto-lib。

## 分支与依赖

- PyPTO：`.worktrees/pypto-csa-static-fixes-20261016`，
  `feat/csa-static-spmd-fixes-20261016`，从调试分支基线 `cb2484471` 创建。
  在本 worktree 全量构建，产物为 `python/pypto/pypto_core*.so`。
- Simpler：`.worktrees/simpler-csa-core-args-256-20261016`，
  `feat/csa-core-args-256-20261016`；已验证的扩容修改本地提交为 `57e499238`，
  中文提交日志包含 ABI 变化、测试范围和未覆盖项，尚未 push。
- 独立 PyPTO 的 `runtime` 子模块与 `_kernel_abi.py` 同步到该 Simpler 提交。
  Simpler 在自己的 `.venv` 重建 native binding，保留 SDK 版本一致性检查。
- `bootstrap_isolated.py` 只重定向当前测试进程的源码和扩展，要求
  `CSA_STATIC_PYPTO_ROOT` 与 `HCA_SIMPLER_ROOT` 显式指定独立 worktree。

首次小例子设备任务 `task_20261016_095538_391201623768` 在编译前被 clean SDK
检查拒绝，**没有执行设备核**。原因是扩容版尚未提交，而 kernel descriptor 要求
精确且干净的 Simpler revision；通过本地提交和重新安装解决，没有放松检查。

## 已实现的修复及证据

### 1. Group 成员共享 InOut 的 SSA 别名

`ExpandMixedKernel` 原先只把 Group 的原始形参映射到新 adapter 形参。
`c = cube(..., c); c, y = vector(c, ..., y)` 中第二次调用收到的是前一次调用的
SSA 返回别名，克隆后变成成员函数中的 free variable。

修复复用现有 `return_lineage` 批量追踪完整参数写回；覆盖普通调用、tuple 返回、
Submit，并把 `no_dep` 的位置同步映射到统一 ABI。view/new-buffer 不能冒充完整
参数别名，明确要求在 Group 外绑定后显式传入。

验证：普通调用与 Submit/tuple/no_dep 两个回归通过；完整 JIT 编译回归生成一次
MIX 提交且保留 sync_start。完整 B4 已越过原先 `progress__ssa_v1` 的未定义参数错误。

### 2. 默认规划器跨阶段复用区不能扩展

原先固定最大成员作为 arena：前阶段 48＋12 KiB、后阶段 32＋32 KiB，虽然两个
阶段不同时存活，仍可能占用 80 KiB。现仅在旧布局及固定容量子区间重试都超限时，
增加一次有容量上限的扩展重试，只有总分配量减少才采用。

保留原生命周期、hazard、no-alias、pipeline 分离约束；pinned allocation 和
不安全动态 view 不参与。allocation 扩展后各 Tile 保留自身窗口大小，最终地址
分配按最大成员 `offset + size` 预留，防止下一 allocation 覆盖扩展部分。

- `phase_partition_PYPTO_fix_v2`：3 参数复现完整代码生成通过。
- `memory_pair_PYPTO_fix_v2`：真实 Indexer＋O-proj 的 13 参数融合完整代码生成通过。
- 两个新单测分别验证可复用场景通过、实际同时存活的 64＋12 KiB 仍报超限。
- 没有修改模型数学、源 tile 或 pipeline stage。编译器自身产生的流水降档提示
  仍照实保存在 `perf_hints.log`，不声称最终流水与动态基线性能等价。

### 3. 嵌套分支合流之后的别名仍带旧 storage

`YieldFixup` 先处理内层分支，改变 phi 的缓冲区；但纯 `alias = phi` 在外层
选择目标前没有更新，造成搬运到已废弃旧缓冲，直到整个遍历结束才修正别名。
现让同一遍历中的纯别名即时跟随源窗口，外层不再选择旧目标。

新嵌套分支单测通过。完整 B4 的 Vec 用量从 267008 降到 201472 字节，消除
65536 字节冗余，但仍超过 188416 字节上限，不能记完整编译通过。

### 4. PTOAS planner 下 reshape/transpose_view 的无操作判断

B4 PTOAS 路径仍在 `qk_pv` 产生两处不支配使用点的 SSA：分支外 reshape 错误
选中分支内旧 producer 的句柄。原无操作判断只比较类型；在没有固化物理地址时，
类型相同不代表当前 source 与旧 MemRef 代表使用同一句柄。

已限制为固定地址模式或当前源句柄确实相同才省略 view，否则从实际 source 发出
`pto.treshape`。相关代码生成回归 228 项通过；B4 的这两处 SSA 错误已消除。

### 5. PTOAS 共用物理窗口的 phi 与多类型句柄

条件 gather 两分支已经写入同一窗口时，`YieldFixup` 在所有 planner 下同步
phi 类型；不同窗口不在 PTOAS 下强行合并。后续 transpose/reshape 让同一存储
具有多种视图类型时，允许复用已经声明且类型完全相同的 handle，不修改 SSA 类型。
这消除了完整 B4 中不支持的 Mat→Mat `tmov`。正例同时检查后续 transpose 和
matmul 的消费句柄，避免只让非法拷贝消失、却让消费者读取另一块缓冲。

### 6. 分支内部原地循环的 allocation 别名生命周期

gather 循环的最终别名因 phi 后使用而延长了生命周期，但其分支内部的其他原地
别名没有加入同一 phi family，导致与互斥分支错误干涉。现按 allocation 建立
定义索引，只为本分支范围内的别名加入 family；不延长这些别名的实际使用期。
分支外仍被独立读取的旧值不加入，反例确认它继续阻止复用。

两块 64 KiB gather 缓冲已合并。完整 B4 总量未同步降到容量内，因为其他复用区
布局随之改变；不能把某两块合并等同于整个函数已满足容量。

### 7. PTOAS loop-carry 的行列视图搬运

目标底层 handle 为 `[1,32]`、逻辑 carry 为 `[32,1]` 时，原代码直接向行
handle 发出列向量 `tmov`，被 PTOAS 的形状检查正确拒绝。现先创建目标存储的
静态 `treshape` view，再使用实际源/目标类型发出搬运；不新增 allocation。
新增回归复现行初始化、列 carry、条件更新、循环写回。

完整 B4 的 PTOAS 路径已越过所有上述 MLIR 合法性错误，推进到 PTOAS 自身的
Mat 地址分配：540672 B 超过 524288 B，仍超 16 KiB；没有修改 PTOAS。

### 8. 仅容量超限时消除分支遍历顺序造成的假干涉

如果旧值只在 else 中读取，then 中的新值不一定需要另占空间。原 then→else
线性遍历将它们判成同时存活。现只在原布局超限时增加一次 else→then 生命周期
分析与有界装箱重试；任一顺序可证明不相交才放开该干涉，保留其他约束。

正例、同分支写后读取反例、循环下一轮读取反例均通过；原本可放下的布局不变。
最新 memory/codegen 定向回归为 `378 passed, 9 warnings`，日志已归档。
完整 B4 的 Vec 降到 197632 B，超过 188416 B 共 9216 B，仍未完整编译通过。

### 9. 剩余 Vec 问题已缩小到 QK/PV 单阶段

`build_memory_probe.py --core aiv` 从当前 B4 静态源码原样抽出 QK/PV AIV，
保留数学、Tile 和流水，只删除跨任务等待与发布；20 参数，禁止单独上卡。
该阶段单独编译仍为 197632 B，因此当前剩余 Vec 超限不是其他阶段同时占用造成。

对照历史动态版 `full_baseline_cpu/kernel/passes_dump/37_after_MemoryReuse.py`：
`qk_pv_aiv` 为 181248 B；当前多出一块 16384 B 复用区。仍需继续核对
核体重新进入前端后的别名/生命周期差异，不能据此认定硬件不可实现。

## 当前验证边界

**最新结论：完整 B4 默认 PYPTO 路径已经完成代码生成与 program 产物的二进制
编译、callable 组装，尚未运行完整设备核。**
static_b4_jit_cpu_v21_carry_scoped 已生成 MIX C++；CPU IR 审计确认一次
24 组 MIX 提交，74 个轮询的 initial/condition/yield/carry 完整，无 FREE_VAR，
sync_start 保留。后续 `static_b4_binary_v1` 复用该产物，AIC/AIV 的 CCE
编译链接、AICPU 调度编译和 132 参数 callable 组装均通过，详见
[二进制编译记录](BINARY_COMPILE_20261016.md)。两项结果都不是完整 CSA 真机通过。

最后一项根因是 TopDownRetargeter::RetargetAssign 要求纯 SSA 别名的旧
MemRef 预先一致，阻断从循环初始化目标递归对齐 producer。修复仅在循环
carry 传播中允许追踪 stale identity，仍检查 producer 的 liveness、no-alias
与 view-offset；其他路径保留旧门槛，尤其不放松累加器分支目标选择。

过程完整保留：新增“无需状态搬运”断言首先失败；第一次无作用域限制的候选
让 QK/PV 与完整 B4 v20 编译通过，但旧累加器 fail-closed 回归失败
（168 passed / 1 failed），因此未作为最终修复。收敛到循环 carry 后，
相关两文件 169 passed，完整 B4 v21 再次代码生成通过。
QK/PV 的 6 次 32 KiB 额外状态搬运消失，其 Vec 从 197632 恢复到 181248 B，
与历史动态版容量一致；完整融合 AIV 最终分配为 163872 B，详见 memory_usage.json。
未手改算术、Tile 或源 pipeline。

**历史 Torch 注册限制已选择直调路线。** 完整 132 参数入口的 CPU Meta 注册报
“The function schema has 132 arguments but this PyTorch build only supports 64”。
torch_schema_capacity_cpu_v1 独立确认 64 成功、65/132 失败。
Tensor[] 的 CPU/Meta 调用可容纳 132 Tensor，但当前 Torch 的可变 Tensor
列表 functionalization 失败，不能直接当作完整集成方案。
用户已要求直接使用 native binding，62 项 workspace 方案取消。
现有 JITFunction.__call__ 已通过 nanobind adapter 提交 kernel，不必经过
132 项 torch.library schema；该路线的完整 CSA 设备验证尚未完成。
见 [直调入口核对](DIRECT_NATIVE_ENTRY_20261016.md)。

- 已完成 207 项定向 CPU 用例的分批验证，包括现有相关回归和新增用例；
  没有重跑此前已完成的 Simpler 大参数真机矩阵。
- 历史默认规划 v18 的 Vec 超限 9216 B 已由 v21 修复；失败证据仍保留。
- PTOAS B4：`static_b4_jit_cpu_v17_column_view`，前端合法性错误消除，Mat 超限 16384 B。
- 当前正式 DSA_RP 路径 `static_b4_jit_cpu_v19_dsa_post_fixes` 在 AIC 容量规划失败；
  它不是固定 PYPTO Tile 的分配诊断，不能与前面的两阶段固定 Tile 通过混为一谈。
- 固定 Tile 的全量 DSA 诊断 `full_b4_fixed_tiles_dsa_v2` 无法重新解析打印 IR 中
  的 `stage_18_v25_1__FREE_VAR`，尚未进入地址分配，不能记为 DSA 无解。
- JIT 设备 v2 在初始化时发现旧 torch_npu adapter 的 Simpler revision 不匹配，
  未执行设备核。独立 worktree 已重新构建 `_torch_npu` 并同步进程加载路径。
- `task_20261016_104815_276112411679` / `jit_group_shared_inout_device_v3` 真机通过：
  24 组、双 AIV、共享 InOut、当前流、连续三次调用与输出保护区均 PASS。
  它证明单 MIX 注册与执行机制，不是完整 CSA 数值/性能验证。
- 分支别名修复后的阶段回归 `165 passed`；最终相关三文件回归 `378 passed`。
  这些计数包含重叠用例，不相加当成独立覆盖数。
- 曾有一次 pytest-xdist 子进程加载了共用旧 PyPTO，相关结果作废；已在独立
  bootstrap 进程内使用 `-n 0` 重新验证。首次两个新增测试的缺少 target_memory
  属测试脚本错误，修正后通过，不列为组件 bug。
- `task_20261016_105329_294883330178` / `phase_partition_device_v1` 已真机通过：
  192、48、128、128 四个行区间逐项核对，max_abs=0；只验证 arena 扩展的复用，
  不代表完整 CSA 或全部控制流改动已真机验证。
- QK/PV 固定 Tile 的 DSA 诊断 `qkpv_fixed_tiles_dsa_v1` 已实际进入地址分配，
  canonical greedy 未找到容量内布局；与早期完整 IR 重解析失败区分记录。

全部结果目录在 `../results/csa_static_spmd_feasibility_20261015/`。
完整 CSA 的直调入口、单次 MIX 设备执行、进度协议可见性、
状态/精度和性能均未验证通过。组件级大参数、最小 JIT 与分阶段 arena 的真机
PASS 不能替代完整核验证。

## 独立环境复现

在当前 vLLM worktree 中，先加载既有环境，再选择两个独立组件：

```bash
source "$PTO_EAGER_ROOT/env-dsv4-0251rc1.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CSA_STATIC_PYPTO_ROOT="$PTO_EAGER_ROOT/.worktrees/pypto-csa-static-fixes-20261016"
export HCA_SIMPLER_ROOT="$PTO_EAGER_ROOT/.worktrees/simpler-csa-core-args-256-20261016"
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"

python tests/pypto_test/csa_static_spmd_feasibility_20261015/bootstrap_isolated.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/compile_jit_entry.py \
  --source tests/pypto_test/csa_static_spmd_feasibility_20261015/static/h131072_b4/entry_jit.py \
  --memory-planner PYPTO --output /tmp/csa_static_isolated
```

CPU 编译不占 NPU。所有设备验证必须通过 `task-submit --device auto`。
