# CSA 单 SPMD 参数容量：来源、扩容影响与待讨论选择

本轮先推进参数容量；尚未修改或重新安装 Simpler、PyPTO，也未提交 NPU 任务。
三项原目标均保留，融合编译及 JIT 注册问题没有因本次 CPU 检查而解决。

## 32 从哪里来

本地 Simpler `src/common/task_interface/arg_direction.h:29` 直接定义
`CORE_MAX_TENSOR_ARGS = 32`，同时定义 `CORE_MAX_SCALAR_ARGS = 16`。
这是 runtime 的固定容量选择，不是编译器发现了 A3 最多只能使用 32 份 Tensor。
Tensor 参数在派发边界转换为 GM 指针，32 表示参数槽数，不是 Tensor 元素数、
寄存器数或可访问缓冲的硬件数量。

本地仓库是浅克隆；`git blame` 到浅历史边界，未找到最初选择“恰好 32”的设计说明。
不能把后面列出的实现效果写成作者当时的已证实动机。特别是 64 位 dump 掩码
只要求 Tensor 加标量不超过 64，不能单独推导出 Tensor 上限必须为 32。

## 已查明的关联约束

以下路径均相对于 Simpler 仓库：

| 位置 | 当前约束 | 扩容时需要共同处理 |
| --- | --- | --- |
| `src/common/task_interface/task_args.h` | 固定数组，增加第 33 个 Tensor 抛异常 | 存储容器及容量检查 |
| `src/common/task_interface/callable.h` | `CoreCallable` 为 32 项签名；头部 144 B、核二进制偏移 192 B | wire 布局、验证、打包、缓存兼容 |
| `src/a2a3/runtime/host_build_graph/common/intrinsic.h` | 上下文指针固定为 `args[48]`、`args[49]` | AICPU 派发与 AICore 核使用一致的参数尾部布局 |
| `src/a2a3/runtime/host_build_graph/runtime/dispatch_payload.h` | 派发结构固定 512 B，并有断言 | 参数向量大小、结构布局及相关平台变体 |
| `src/common/platform/include/aicpu/dump_arg_selection.h` | 单个 64 位掩码，断言 Tensor 加标量不超过 64 | 参数选择、dump 传输/解析/重放，不能静默漏掉高位参数 |
| `src/common/host_build_graph/shared_memory.h` | Host mirror 按 `max_tasks × MAX_TENSOR_ARGS` 预留 Tensor 池 | 容量扩展的内存代价；设备 live image 按实际使用量压紧，不能与 mirror 混淆 |
| `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/runtime_types.h` | 每任务内嵌定长 Tensor 数组 | 每个任务槽的静态空间代价 |

A5、TMR/HBG 也有对应的上下文和派发实现；公共 ABI 变化不能只更新一个头文件。
HBG 的 `GRAPH_MAX_TENSOR_ARGS = 128` 是另一层容量：若未来入口经过 graph boundary，
须核对该路径；不能用 CHIP 的 256 或 Graph 的 128 冒充 CORE 已支持大参数。

## 七档参数审计

`audit_core_parameters.py` 读取现有生成源码和绑定文件，结果在
`static/core_parameter_audit.json`。未重新采集映射或编译全量 CSA。

| 项目 | 长 history 四档 | 短 history 三档 |
| --- | ---: | ---: |
| 核共享 Tensor 参数 | 132 | 133 |
| Native 描述符 / 原始命名参数 | 60 / 56 | 60 / 56 |
| 临时描述符 / 临时 allocation | 69 / 66 | 70 / 67 |
| 协议描述符（progress、records、waits） | 3 | 3 |
| AIC 计算体引用的参数 | 54 | 54 |
| AIV 计算体引用的参数 | 115 | 116 |
| 两半计算体均不引用的参数 | 0 | 0 |

这里统计名称引用，排除函数尾部仅转交别名的 return；不证明 56 个 Native 对象的
实际地址两两不同。合并重复视图仅有 7 项描述符的候选空间，而且需先证明核内
视图的表达和寻址正确；它不能把当前接口压到 32。两半 Group 共享 payload，
不能简单各删一套参数使参数位置错位。

## CPU 模板容量/布局探针

`probe_capacity_layout.cpp` 使用实际 Simpler 的通用 C++ 模板，分别实例化容量，
不改变生产 `CoreCallable` 别名。结果在 `static/capacity_layout_probe.json`。
仅证明容器能否装下占位描述符和编译器计算出的布局，不证明 runtime 可运行。

| 容量 | 能装入 133 项 | Callable 头部 / B | 核二进制偏移 / B | 现有 dump 掩码兼容 |
| --- | --- | ---: | ---: | --- |
| 32 | 否 | 144 | 192 | 是 |
| 128 | 否 | 528 | 576 | 否 |
| 160 | 是 | 656 | 704 | 否 |
| 256 | 是 | 1040 | 1088 | 否 |

仅核算 HBG Host mirror 的 Tensor 池，按当前 128 B/runtime Tensor：
32→160 每任务槽多 16 KiB；32→256 每任务槽多 28 KiB。
例如假设配置 8192 个槽，分别多 128 MiB、224 MiB。此数是布局公式推算，
不是当前进程的实测占用，也不是每个活跃任务都要传输的字节数。

## 历史方案与用户决定

建议先解决 Simpler 的参数承载能力：保留 Native 的输入、NZ 格式、cache/state
及当前核体，评估至少可承载 133 项的接口。160 是满足本原型且为 8 倍数的一个
验证档，256 是更宽的候选；两者均需完整 ABI 修改，不能当作改常量即完成。
生产设计还应考虑为小参数任务保留紧凑路径，避免所有任务承担最大容量成本。

另一条路线是在 PyPTO/Simpler 中正式定义描述符集合和核内 GM 子视图，属于更大
的接口设计。当前没有已验证的 API 能把全部 Native/NZ/不同 dtype 的工作区
恢复为同一计算语义，因此不在 CSA 内临时伪造指针或强制搬运输入。

用户随后明确授权本会话在独立 Simpler worktree 实施并验证容量扩展，包含原先的
64 位参数 dump 掩码与 512 B 派发限制。当前分支为
`feat/csa-core-args-256-20261016`，位于兄弟 worktree
`../simpler-csa-core-args-256-20261016`，基线仍是调试分支 `ef6812891`。

已实现 256 个 CORE Tensor 槽，保留 16 个标量槽；上下文索引改由容量推导为
272/273。dump 使用覆盖 272 位的 5 字 POD 掩码；更新 CoreCallable、HBG/TMR
派发、Graph 边界、HBG dump 元数据以及依赖记录。A3 的 Host/AICPU/AICore 已完整
构建，并只安装到该 worktree 的 `.venv`。原来的 Simpler 源码与共享环境未变。
CPU 回归和 A3 两套 runtime 的 program/kernel 大参数执行、高位 dump 均已通过。
参数容量这一组件目标完成，详见 [扩容验证记录](SIMPLER_CORE_EXPANSION_20261016.md)。PyPTO、PTOAS、PTO-ISA 和 pypto-lib 未修改。
