# Simpler CORE 参数扩容：实现与验证

2026-10-16，用户明确授权本会话直接修改 Simpler，并要求在独立 worktree 实现与验证。
原来的 32 Tensor、64 位 dump 掩码和 512 B 派发结构均属于本次允许修改的范围。

## 结果

参数容量这一项已经完成组件级实现与真机验证。独立分支
`feat/csa-core-args-256-20261016`，基于调试分支 `ef6812891`，位于兄弟 worktree
`../simpler-csa-core-args-256-20261016`。仅安装到该 worktree 的 `.venv`；原 Simpler
checkout 与 vLLM 共享环境未修改。此次没有修改 PyPTO、PTOAS、PTO-ISA 或 pypto-lib。

| 项目 | 扩容结果 |
| --- | --- |
| CORE Tensor / scalar 槽 | 256 / 16 |
| Local / Global context 槽 | 272 / 273，由容量常量推导 |
| CoreCallable 固定头 / 二进制偏移 | 1040 / 1088 B |
| DispatchPayload | 2304 B，首 64 B 控制区位置保持一致 |
| 参数 dump 掩码 | 5 × uint64，覆盖实际 272 位 |
| ArgsDumpTaskMetadata | 96 B，POD 布局 |
| HBG TaskPayload / ChipTaskStorage | 256 / 384 B |
| HBG Graph Tensor 边界 | 256 |
| TMR TaskPayload 标量区偏移 | 640 + 256 × 128 = 33408 B |
| DepGenRecord | 33664 B；Tensor 区从 896 B 开始；overflow 容量按记录大小推导 |

A2/A3、A5 的 HBG/TMR 对应布局、派发、元数据与设备侧读取同步修改。
固定容量扩大会增加参数池预留空间；例如 HBG Host mirror 的 Tensor 池每任务槽
多 28 KiB，而设备 live image 仍按实际使用量压紧。未把这些公式值当作性能测试结果。

## 有效验证

| 验证 | 结果 |
| --- | --- |
| A2/A3、A5，HBG/TMR，onboard 与 sim runtime 构建 | 全部通过 |
| C++ 参数边界、dump 池、Graph replay、内存布局及依赖重放 | 14 个目标通过 |
| Python 参数接口、callable 缓存、Host runtime ABI | 281 PASS，4 个需硬件标记的测试在此 CPU 调用中跳过 |
| A3 HBG program，133 / 256 Tensor + 16 scalar | 输出与保护区通过；三轮运行通过，另做单轮 dump |
| A3 TMR program，133 / 256 Tensor + 16 scalar | 输出、保护区及单轮 dump 通过 |
| A3 HBG kernel，133 / 256 Tensor + 16 scalar | 每种规格三次 launch 全部通过 |
| A3 TMR kernel，133 / 256 Tensor + 16 scalar | 每种规格三次 launch 全部通过 |

每个入口连续提交两项 24 核组 MIX，包含 AIC、AIV0、AIV1；第二项依赖第一项，
启用 early resolve 和 sync_start。核内读取全部 Tensor 与 16 个标量，使用不同核组、
不同 lane 的输出区验证上下文；所有未写入位置也参与比较。
选择性 dump 实际验证索引 63/64/128、最后一个 Tensor、最后一个标量：
133 参数对应 132/148，256 参数对应 255/271；四份采集均无截断/丢记录。
A5 没有在本机真机运行，仅完成构建与 CPU 回归。

真机任务：

- HBG program + dump：`task_20261016_093902_323003320750`。
- TMR program + dump：`task_20261016_094207_335223022023`。
- HBG kernel：`task_20261016_093836_321677328546`。
- TMR kernel：`task_20261016_094208_335284912316`。

## 实际发现并修正的构建问题

第一次 TMR 设备采集显示：第一项任务输出正确，第二项 early-dispatch 输出异常；
AICPU dump 中的输入和最后一个 scalar 均正确。定位到 runtime 的自定义 CCE 构建
未跟踪 `dispatch_payload.h`：已修改 scalar 偏移，但 AICore executor 仍使用此前生成
的旧 `.o`，导致 gated 路径从旧地址读取标量。补齐 A2/A3 与 A5 的
`arg_direction.h`、`intrinsic.h`、`dispatch_payload.h` 和 A5 Graph 调度头依赖，
重编译后两种参数量全部通过；生成的 Ninja 依赖也已核对包含这些头文件。
这说明不能把一次“增量构建结束”直接当作全链路 ABI 一致的证据。

另有一次设备初始化报 `rtSetDevice(9): 507033 / E39007`，发生在内核提交前，属于
设备子进程启动超时；后续独立进程复跑成功。它与上述稳定复现的旧偏移问题分开记录。
最初 C++ Graph 测试假定 Graph Tensor 池必然大于 CORE 单槽；容量对齐后该断言已
过时，改为验证新的 Tensor 预算关系，同时保留标量预算与实际 Graph 重放检查。

## 复现与证据

组件用例在独立 Simpler worktree 的 `tests/st/a2a3/core_argument_capacity/`：

- `test_core_argument_capacity.py`、`kernels/`：两种 runtime 的大参数 MIX。
- `kernel_mode.py`：调用方拥有 device Tensor 与 stream 的 kernel 模式。
- `verify_dump.py`：只读验证真实 dump 的高位参数、标量值与任务记录。
- `README.md`：运行命令、容量定义、ABI 重编译要求。

本仓库证据目录：`../results/csa_static_spmd_feasibility_20261015/simpler_core256_v1/`，
包含汇总 JSON、CPU 日志、真机任务日志和四份 dump 元数据。大二进制 dump 本地保留，
路径见该目录的 `LOCAL_ARTIFACT_PATHS.md`；未计算 hash。

## 仍然未完成的原目标

本次证明 Simpler 能承载原型所需的 132/133 个 Tensor，不代表整层 CSA 已经融合运行。
完整 CSA 融合的片上内存规划/代码生成问题，以及可注册的 JIT 单 MIX 入口仍需继续。
未重复七档全量测试，未运行性能测量，未将容量测试标记为 CSA 精度或性能 PASS。
