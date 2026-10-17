# CSA 动态调度固化为单个 SPMD：完整过程总账

本账覆盖 2026-10-15 起本会话的 CSA 单 SPMD 工作，包括失败、作废、撤回及
仍未完成的尝试。后续继续追加，不能只保留成功结果。

用户指定归档位置为：
`vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_static_spmd_feasibility_20261015/`。
开发仍在独立 worktree；这里的脚本、记录、结果摘要和补丁快照同步到指定目录，
不覆盖其他会话的生产代码。详细编译 IR、二进制和大快照保留原位置，
路径、大小及保留方式见 `artifact_inventory.json`；不生成内容 hash。

## 读哪些文件

- [逐次尝试清单](ALL_ATTEMPTS.md)：按实际目录枚举，包含成功、失败和没有最终
  result 的尝试；完整错误见对应的原始 JSON/log，不用目录名猜测 PASS。
- [问题与修复登记](ISSUES_AND_PRS.md)：issue、PR、本地提交、未提交修复分别记录。
- [最新 PyPTO 修复](PYPTO_STATIC_FIXES_20261016.md)：代码位置、原因、改法、验证边界。
- [Simpler 扩容](SIMPLER_CORE_EXPANSION_20261016.md)：256 Tensor、16 scalar、dump、
  派发、图缓存等 ABI 和设备验证。
- [PR #3013 核对](PR3013_REVIEW_20261016.md)：精确 head、独立构建、原始 while 复现和验证边界。
- [JIT 直调入口](DIRECT_NATIVE_ENTRY_20261016.md)：用户取消 62 项方案后的调用路线与 TorchAir 参考。
- [编译接续说明](COMPILE_AND_RESUME.md)：修复前各路径、接口约束和恢复顺序。
- `validation_log_excerpt.md`：主验证日志第 603 节起的原文，保留时间顺序。
- `session_logs/`：本轮构建、单测、完整编译和诊断原始终端日志。
- `component_changes/`：独立组件的当前补丁、提交日志及分支状态，不包含 .so/bin。
- `evidence/`：从结果目录归档的 JSON、日志、输入源码等文本证据。

## 目标与不变约束

按较优动态调度的实测分片、逐核顺序与依赖关系，形成一次 24 组 MIX SPMD
提交，包含 24 AIC、48 AIV。保留原有计算、Tile、Cube/Vector 内部事件和
Native device Tensor 入参。公共源码顺序不能额外变成全卡屏障。

先功能、后精度、最后性能。正式权重固定为
`/data/model/DeepSeek-V4-Flash-0731-w8a8`。所有 NPU 执行通过 task-submit。
不能用无意义 block_idx/循环结果读取绕过编译器，不能跳过容量或 SDK 检查。
编译器自行发生的流水降档必须保留 perf_hints，不能声称与动态版性能等价。

## 阶段与所有尝试的归类

| 阶段 | 做了什么、遇到什么 | 证据入口与当前结论 |
| --- | --- | --- |
| 初始可行性 | 审计历史 trace，区分 task、block_idx、AIV lane、物理 worker；不能从时间条长短推断任务依赖 | `assessment*.json`、`trace_audit.json`、历史 schedule |
| 原动态基线与映射 | 七档补采完整 CSA 的逐核映射、真实实参和保护区；DEBUG 调度受扰动，不能记作最佳性能 | `mapping_*`、`bindings/`；原动态验证不等于静态验证 |
| 核体导出与独立编译 | 从原完整编译产物提取 73 个物理核体；处理打印 IR、动态维度及 JIT metadata 推断限制 | `cpu_v1…v5`、`vector_bodies_cpu_v1…v3`、`generated/` |
| 进度协议与模板 | 固化逐核顺序、等待目标，消除传递冗余；按 worker 隔离标记，加入代际防旧值满足新等待 | 七档 `static/h*/layout.json` 与 `protocol_model.json`；仅 CPU 模型通过 |
| 完整 program 原型 | v1～v14 包含本地生成器/表达调试：SSA、动态维度、NZ、slice、reinterpret 等；v15 才推进到主要容量问题 | `static_b4_cpu_v*`；不把每一次生成器错误列为组件 bug |
| 多 planner 与 Buffer IR | PYPTO Left/Right 80 KiB 超 64 KiB；DSA_RP 无布局；PTOAS 出现 view/SSA/Mat 拷贝问题；Buffer IR 暴露 tuple 表达问题 | `static_b4_cpu_v15…v23*`、内存复现文档；均未完整运行 |
| 单 MIX JIT 入口 | 直接双调用、cluster/SPMD、opaque wrapper、inline Group 等逐项试验；最终 inline Group 可表达单 MIX | `static_b4_jit_cpu_v1…v6*`、`jit_group_*` |
| 参数容量 | 原型长/短 history 为 132/133 Tensor；原 CORE 32，dump 64 位、派发 512 B 均不足 | 独立 Simpler 扩容到 256+16；关联 ABI 和四类设备入口验证通过 |
| 融合分阶段内存 | Indexer、O-proj 各自通过、合并失败；缩成 48+12 / 32+32 KiB 三参数复现 | arena 有界扩展修复后，三参数及真实 13 参数代码生成通过 |
| 独立 PyPTO | Group InOut 别名、嵌套 phi、分支 allocation 生命周期、PTOAS view/handle/搬运逐项修复 | `static_b4_jit_cpu_v7…v21*`；最终 v21 默认 PYPTO 完整代码生成 PASS |
| 设备入口验证 | clean SDK、adapter revision 两次在核执行前失败；隔离构建对齐后通过 | `jit_group_shared_inout_device_v3`，24 组、双 AIV、当前流、三次调用与保护区 PASS |
| 剩余容量归因 | B4 Vec 267008 → 201472 → 197632 B；QK/PV 多 6 次状态搬运及一块 16 KiB 缓冲 | 限定循环 carry 的纯别名修复后，单阶段恢复历史 181248 B；完整 B4 代码生成通过 |
| 完整 Torch 注册 | 完整入口 132 个具名 Tensor，当前 Torch schema 上限 64；Tensor[] 的 CPU/Meta 可调用但 functionalization 失败 | `static_b4_jit_meta_v1`、`torch_schema_capacity_cpu_v1`；62 项入口方案待讨论，尚未实现 |
| 完整二进制编译 | 保留现行 132 参数接口，复用 B4 v21 program 产物验证 CCE/调度编译与组装 | `static_b4_binary_v1`：AIC、AIV、AICPU 和 callable 组装通过；没有完整设备运行 |

## 必须保留的失败与无效结果

1. `static_b4_cpu_v20_buffer_ir` 为 PassContext 配置错误，不作为组件故障。
2. 固定 Tile 的完整 DSA 诊断曾在打印 IR 重入前端时报 FREE_VAR，尚未运行到
   DSA 分配；不能记为“DSA 已证明放不下”。
3. 两次最小 JIT 设备失败发生在 SDK/adapter 版本检查，未执行核；修正隔离
   构建后已有 v3 PASS，不能把前两次说成设备数值失败。
4. 新增测试的 shape/target_memory 写法错误与 pytest-xdist 加载共用旧扩展
   均单列为测试环境/脚本问题。作废结果不计覆盖，串行独立进程复测有效。
5. 最初无容量门槛的分支顺序优化改变原本能放下的布局，使旧约束测试失败；
   已收敛为仅超限时启用，并补同分支尾读、跨循环读取反例。
6. 调试用生命周期/配对打印已经移除，原始日志保留。
7. 缺失历史输入源码副本的早期尝试，以各自保存的 pass dump/错误为证据；
   不将当前生成器源码冒充当时源码。
8. 归档脚本第二次刷新遇到设备原日志的只读权限被复制到归档副本；第三次
   修正为只开放归档副本的 owner 写权限，原始日志不变。三次归档日志均保留。
9. 纯别名首次候选没有限定作用域：完整 B4 v20 虽代码生成通过，却使累加器
   不一致别名的反例不再拒绝，回归 168 passed / 1 failed。该版本不作为最终
   修复；收敛到循环 carry 后 169 passed，完整 v21 再次代码生成通过。
10. Tensor[] 小例子只验证 132 个 Tensor 经两个 schema 入参的 CPU/Meta 调用。
    functionalization 失败，因此没有把它标为 PyPTO 注册或整网集成方案通过。

## 2026-10-16 后续追加

- 逐 op/shape/dtype 对齐历史与当前 QK/PV，当前多 6 次 32 KiB 状态搬运；
  BF16 归一化结果因此另占一块 16 KiB。序列对齐只是定位线索，
  `static/qkpv_extra_arena_alignment.json` 不作为数学等价证明。
- 现有嵌套 phi 小例子增加“纯别名不应阻断 carry 原地更新”检查，修复前失败。
  修复限定在循环 carry 传播，保留累加器合并的旧门槛，最终两文件 169 passed。
- 完整 B4 `static_b4_jit_cpu_v21_carry_scoped` 默认 PYPTO 代码生成通过；
  一次 24 组 MIX 提交，74 个轮询状态完整，无 FREE_VAR。AIV 的 tile.alloc
  大小求和为 163872 B，最终地址分配通过；还没有 CCE 二进制或完整设备结果。
- 新发现 Torch 64 入参限制。独立复现、完整注册栈和入口方案记录在
  [TORCH_SCHEMA_CAPACITY_20261016.md](TORCH_SCHEMA_CAPACITY_20261016.md)。已按
  原目标“遇到关键点停下讨论”询问 56 个 Native Tensor + 6 个分 dtype workspace
  的具名入口方案，等待答复期间没有实现该方案或修改共享 PyTorch。
- 与入口选择无关的现有产物二进制编译已独立推进：`static_b4_binary_v1`
  在 115.418 秒内完成 AIC/AIV CCE 编译链接、AICPU 调度编译和 132 参数
  callable 组装。原产物未改，二进制本地保留；见
  [BINARY_COMPILE_20261016.md](BINARY_COMPILE_20261016.md)。

## 当前完成边界

CORE 扩容与最小单 MIX JIT 注册/执行已通过。完整 B4 的默认 PYPTO 路径已
完成代码生成及 program 产物二进制编译/组装，PTOAS/DSA_RP 的历史失败仍保留。
用户已选择现有 JIT native 直调入口；完整单 SPMD 的设备功能、进度可见性、连续状态、
数值和性能尚未通过。
原动态基线、CPU 模型、最小 JIT 设备例子三者都不能替代完整 CSA 验证。

后续每次尝试追加：目的、源码/分支版本、改动、命令或任务 ID、实际到达阶段、
结果、保留/撤回决定、下一步。已发的 issue/PR 记录真实链接；未发的只列本地
补丁，不虚构 PR。整理材料不自动创建 PR、不自动 push。

## 历史接续等待审计（用户已答复，见后续追加）

2026-10-16 第三次连续接续核对同一入口阻塞：当前 Torch schema 限制仍为
64 项，完整入口仍为 132 项，56+6 候选未获用户答复。前一轮已独立完成
现有产物的全部二进制编译，CPU 编译进程均已结束；不存在仍在排队或执行的
本会话任务。本轮没有重跑测试，只读结果、注册代码状态和已结束进程。

按原目标“有过不去的关键点停下找我讨论”，接下来须先确定入口方案。
不能将编译通过当作完整注册或设备功能通过，也不擅自修改共享 Torch 或
实施尚待讨论的 workspace 接口。满足三轮同一阻塞的审计条件，目标未完成；
实际目标状态写入 `goal_audit.json`。用户答复后从现有产物接续。

## 2026-10-16：直调选择、上游核对及必要 bug 上报

用户明确选择直接 pybind/native binding，并指定 TorchAir 的多种接入示例。
已确认 PyPTO 现有 JITFunction.__call__ → launch.invoke → nanobind prepare/enqueue
路径，不经过 132 项 torch.library schema；当前 adapter 已使用 OpCommand 入队。
62 项 workspace 候选取消，旧等待审计只留历史。

按用户新要求只提必要正确性 bug，创建 PyPTO #3014～#3017，分别为 Group InOut、
PTOAS 分支 view、条件 gather phi 和列向量 carry。四份均内嵌完整复现、代码位置、
修改建议、不可放松的约束和验证边界。随后补入精确上游 head 的对照：5 failed；
相同复现在本地候选为 5 passed。创建与更新均逐份回读核对正文。
首个回读发现 GitHub 未应用请求的 bug 标签，记录实际为空而不重建 issue；
一次 GraphQL EOF 改用 REST 回读，未重复创建。

Simpler 扩容及 PyPTO 内存规划改进按用户指示保留本地；没有新建扩容 issue/PR，
没有 push 组件分支。Simpler 构建依赖的主线修复已确认是 #2331 / #2340，避免重复上报。

PR #3013 拉取到独立 detached worktree，自行构建；七个新增定向用例通过，
本会话原 scalar、tile、tile 保留结果三种复现均完成代码生成且无 ready FREE_VAR。
两次环境准备失败及修正均见独立审阅记录。没有新 NPU 任务或整层性能数据。

## 2026-10-16：完整 native 直调验证，用户指定 128K/B16

新增 `test_static_jit_device.py` 与队列入口 `run_static_jit_device.sh`。静态入口
直接调用 JITFunction，不建立 132 项 Torch schema；动态参考复用已有 56 项合同。
正式权重仍为 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 第 4 层，冻结计算源与
调度提取保持一致。检查输出、六份 cache/state、只读 metadata、allocation 保护区、
workspace 首尾及 72 核的进度代际；不清零跨次进度。此阶段没有性能结论。

首个 B4 任务 `task_20261016_152842_14127126353` 完成 Native 参考后，
在 NativeCSACall 参数准备处报 `KeyError: host_max_seq_len`；原因是测试驱动
直接取源码签名包含了 constexpr。恢复动态参考原有 56 项 schema 后修正。
该失败未进入静态核设备调用，不属于组件编译 bug，不预记 PASS。

用户随后指定直接测 **128K/B16**，即 history=131072、16 请求、每请求 6 个 query
token、总 T=96。已提交 `task_20261016_153044_22825110597`，结果目录
`static_b16_native_direct_v1`；不重跑 B4。后续结果另行追加。

128K/B16 首轮已完成 Native、冻结动态 CSA 及保护区检查，56 个 Native 输入
成功绑定为 132 个静态入参。但旧 B16 `entry_jit.py` 仍把两半核直接放在
SPMD 的 cluster 中，前端报一个 SPMD scope 有两次 dispatch；静态设备未执行。
这不是新组件 bug，是该档旧生成物未同步此前 B4 的 inline Group 修正。
重新使用现有 generator 仅生成 B16，逐 AST 核对 AIC/AIV 计算体完全未变，
全部 entry.bindings 参数映射完全未变，仅更新 wrapper。旧源码已保留在
`prepared_b16_inline_v2` 及失败任务的 source.py。

生成时第一次从实验子目录执行，catalog 内相对路径未找到；改从 worktree 根目录
执行后成功。`static_b16_kernel_cpu_v1` 仍读取旧模板并复现 dispatch 错误；
`static_b16_kernel_cpu_v2` 已读取含 inline Group 的完整新源，正在 CPU kernel-mode
代码生成。CPU 编译不占卡；没有把动态参考或前端解析当作静态设备通过。

`static_b16_kernel_cpu_v2` 完整 kernel-mode CPU 代码生成通过。生成的 orchestration
只有一次 `rt_submit_task`，对应 MixedKernels、24 组、sync_start=true；独立审计
写入同目录 `single_mix_audit.json`。正式设备续测任务
`task_20261016_153714_4094199440`，输出 `static_b16_native_direct_v2`。
此时只证明 B16 编译及单 MIX 结构，没有提前宣称设备结果通过。

## 2026-10-16：仅继续静态 128K/B16，保留全部静态原型

用户指出最终 batch_size/seqlen 应可动态变化。已澄清当前生成器把 shape、
循环范围和逻辑 block 数按捕获档位特化，属于静态验证原型，并非最终动态实现。
用户随后明确先测试当前静态 **128K/B16**，并要求已完成的静态代码不要删除。
因此保留全部静态版本、失败快照和验证记录，只继续这一档；动态化后续处理。

任务 `task_20261016_153714_4094199440` 已失败：完整 Native/动态参考完成后，
静态入口 preflight 拒绝 view_98 与原 Indexer cache 的可写部分重叠，尚未发射静态核。
这是原缓冲与合法子视图同时作为独立外部 Tensor 的合同限制；不关闭别名检查。
参照生产动态图，在 orchestration 内从原 cache 用 reshape/slice 派生 view_98/99。
外部输入 132→130，leaf/Core 仍为 132 项，原存储和 AIC/AIV/Group 计算体均未变。
两份子视图只是描述符，不增加搬运、设备 task 或新的 cache。
旧版备份在 `prepared_b16_native_views_v3`，AST 对比记录同目录。

`static_b16_kernel_cpu_v3_native_views` 未通过元信息推导：JIT 局部 Tensor
推导只识别公开的 pl.reshape/pl.slice，生成器用了 pl.tensor.reshape/pl.tensor.slice。
本轮只改生成器中这两份根视图的写法，不改组件、不改别名校验和核体。
修改前文件另存 `prepared_b16_public_views_v4`；下一次 CPU 编译结果另记。

static_b16_kernel_cpu_v4_public_views 完整 kernel-mode 代码生成通过。
仅公开 view API 写法修正，AIC/AIV/Group 的 AST 与上版一致。
真机任务 task_20261016_155540_123832232627 已提交，仅 128K/B16；
结果目录 static_b16_native_direct_v3，未提前记设备 PASS。

设备 v3 已结束，exit=1。AIC/AIV 和编排库全部编译完成，native adapter 已提交；
Simpler 设备日志报告 S1:running-stalled，completed=0/1、running=1、orch_done=1，
随后返回 507018。不能把 result.json 中 device_execution=false 理解为未发射：
该字段只在同步成功后置 true，本轮设备侧已观察到运行中的单 MIX，但未完成。
尚无输出、状态或精度结论；待用实际进度定位卡住阶段，不将其当作参数上限失败。

用户转来另一会话关于动态 metadata 与静态模板边界的分析，要求留作后续参考。
已核对主仓三个源码位置及当前 B16 模板的 device Tensor 读取，单独记录于
DYNAMIC_SPMD_REFERENCE_20261016.md，并从 README 链接。仅保留设计参考，
不扩展当前 128K/B16 范围，不宣称当前停滞由该分析中的问题造成。

## 2026-10-16：停滞定位先检查原控制表

重新核对任务 v3 已终止（exit=1）；上一轮属于有进展：完整编译、设备启动及
S1:running-stalled 证据改变了下一步。没有重提同一完整任务或重跑七档。

新增 diagnose_static_protocol.py，从 B16 完整源的明确 fence 边界之间移除数学
段，保留原 records/waits、核号公式、代际和可见性操作。有限轮询失败后记录
核、batch、等待对象、目标、实际值及轮询次数，不伪造完成信号；后续工作跳过，
使故障信息有机会返回。只保留三个控制 Tensor，不能作为完整 CSA 或 132 参数
入口通过证据。原完整静态代码不覆盖、不删除。

初版 static_b16_protocol_v1 在 CPU 检查时发现诊断新增常量写入 INT64 Tensor
缺少显式 cast，未上卡。另核对自动类型推断：去掉矩阵运算后 InCore 会被归到
AIV，不能据此代表原 AIC。因此 v2 使用 program 模式显式 AIC/AIV 两半、单 MIX
及同一 24 组，保留公开合法接口，不修改 JIT 或组件。program 还可选择性 dump
进度；kernel 模式目前明确不支持 args dump。该模式区别单独标注，不混作完整
native 入口验证。v2 CPU 完整代码生成通过，一次 rt_submit_task。

控制表任务 task_20261016_162004_206538014143 已返回，设备调用本身完成，有限
轮询发现等待失败；完整逐核信息见 static_b16_protocol_v2/result.json。例：
core0 等待 core32 的 1025，200000 次轮询均未达到，记录值为 1024；最终取回
core32 已为 1025。72 核均发布了初始代际，问题已经可在移除数学后的同一
控制表复现。该证据提示轮询/发布可见性问题，尚不能仅据最终快照断言具体优化。

当前 tensor.read 没有 volatile 参数，生成 C++ 是普通 GM 指针读取；PTOAS 的
DCCI helper 只发 dcci。另建 static_b16_protocol_v3_volatile，仅将诊断产物的
progress 标量访问改为 volatile，保留原表、轮询预算及其他代码，比较是否改变
结果。没有修改 PyPTO/Simpler/PTOAS 源码，也没有覆盖完整静态实现。
准备脚本最初误要求三份 C++ 命中；实际参与 CCE 的 AIC/AIV 两份已正确修改，
随后逐字核对并修正报告断言，不重提任务。任务 task_20261016_162414_222408211822。
一次 llvm-objdump 输出均为 not available，仅作失败尝试保留，不当作汇编证据。

当前 16 卡均由 CI 任务占用，volatile 对照仍在原队列 pending，未重启或重提。
CPU 侧尝试 CCE -emit-llvm、-print-after=licm，均明确不支持，失败命令与输出
分别保留；没有拿到优化 IR，不能宣称已经从汇编证明 CSE/LICM。

另准备 static_b16_protocol_v4_fence，直接在 PyPTO 诊断源的消费者
cacheinvalid 后插入既有 pl.system.fence，完整 CPU 编译通过。目的是核对现有
接口能否表达所需可见性，而不是急于改组件。源文件与编译快照逐字相同；
上卡驱动调整为使用既存候选源，不在运行前重新生成覆盖诊断变化。

fence 对照已提交为 task_20261016_163312_271626813951；与 volatile 对照均仍在
原队列 pending。重新查询证实 16 卡由 CI 占用，没有本会话存活的 CPU 编译。
v2 的 72 核均记录有限等待失败，其中 17 个等待目标在最终快照中已达到；
另存 failure_summary.json，明确时序信息不足，不能据此单独认定编译器优化。
新增 B16_PROGRESS_DIAGNOSIS_20261016.md 汇总实验区别及未证实项；README
将旧限制明确标成历史，接续范围改为用户最新指定的 128K/B16。

## 2026-10-16：对照结果返回，先复核 Scalar 非一致缓存协议

原队列任务已结束：v3 读写 volatile 的 task_20261016_162414_222408211822
连续 3 个代际的控制进度符合预期；v4 消费者增加 fence 的
task_20261016_163312_271626813951 首轮 72 核有限等待失败，设备正常返回。
随后拆分读写影响：v5 仅读 volatile 的 task_20261016_164435_356209620998
连续 3 个代际符合预期；v6 仅写 volatile 的 task_20261016_164436_356260019239
首轮 72 核有限等待失败。全部使用原控制表与相同 200000 次轮询预算，原数学体
仍被移除，不能记为完整 CSA 或完整缓存协议通过。所有任务均已终止。

prepare_protocol_volatile.py 增加 all/read/write 选择；只修改各次诊断的 C++
产物，AIC/AIV 每份识别 170 个读点、590 个写点，不改完整静态核或组件源码。
CPU 另尝试外部标量 helper 表达：func.call 的最小 PTOAS lowering 通过；
emitc.call_opaque 版本因 !pto.ptr 类型不被接受而失败，材料保留于
volatile_scalar_call_probe_v1。没有据此实现 PyPTO API，也没有上卡执行该 helper。

用户指出 Scalar cache 非一致性问题后，停止推进 volatile 接口及完整设备续测。
读取指定 Simpler 分支的 ATOMIC_USAGE_GUIDE、cache_preload_usage_guide、
icache_miss_usage_guide，保留固定 Git 版本的文本快照与来源。对照结果仅保留为
诊断现象：volatile 约束编译器访问，不建立跨核 coherence，不能归因为缺少该
选项即可修复的组件 bug。

代码审查发现生成器在阶段前后使用整缓存 DCCI，完整 B16 静态调用点共 168 处，
没有完成逐 cacheline 所有权及 payload 交接论证。消费者先 fence 后整缓存 DCCI
也不能据此证明后者已在业务读取前完成。两份 row offsets 实际由一个 worker 写，
valid-QK-plan 按 64 B 行分片；不能误报成已证实的多写者覆盖。

用户进一步转来 syncall(SOFT) 与 CSA 进度方案的分析，并修正写侧职责。
已核对当前本地 syncall_soft.hpp：A3 使用 st_atomic 累加共享计数器，DCCI + DSB
发布，ld_dev 轮询。A3 存在该原子累加路径，“A3 没有 atomic”的泛化不准确。
CSA 每核进度行只有一个写者时不需要多写者 atomic RMW；应分别保证普通写发布、
读侧新值获取及业务数据同步，不能因此改成全体 barrier。A5 文档的保守 atomic-only
规则不能直接推出 A3 单写者协议必需 atomic。详见 SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md。

本轮未增加任何 PyPTO/Simpler/PTOAS 修复、未删除静态代码、未新增设备任务。
恢复上下文后 get_goal 实际返回 paused；记录实际状态，不自行恢复目标。
当前工作仅为用户要求的协议分析、纠正旧结论和过程归档。

用户随后提出 PyPTO 应提供 incore task 内点对点通知同步。已核对现有接口：
syncall 是集合 barrier；sync_set/sync_wait 是 Cube/Vector 硬件事件；
pld.system.notify/wait 已有跨 rank 信号能力，但输入为 DistributedTensor。
建议补齐普通 GM 信号的单 SPMD 核间通知/等待，明确多读者进度阈值、代际、
发布/读取与 payload 内存顺序，以及 MIX 双 AIV 的执行归属。接口建议写入
SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md；未实现、未上卡、未新提 issue。

## 2026-10-16：按用户授权提交 InCore 点对点通知/等待功能需求

用户明确要求将同一个融合 InCore task 内生产者通知必要消费者的能力提交到
hw-native-sys/pypto。已读主仓 feature_request 模板，检索 notify、point-to-point、
sync 相关 issue，核对 #2072、#2130、#1906 的范围；没有发现此次普通本地 GM
信号接口需求的直接重复。以公开主线 8325a8de8422678d6699e17f31cef953ace66445
的 system_ops 和 distributed system_ops 为现有接口依据，保存来源文本。

已创建 https://github.com/hw-native-sys/pypto/issues/3020：
[Feature] 支持单个 InCore/SPMD 内基于普通 GM Tensor 的核间点对点 notify/wait。
正文为中文，说明 CSA 融合场景、已有接口差异、建议语义、数据与信号的同步区别、
动态阈值/代际、多消费者、MIX lane 和编译器副作用、替代方案及独立小例验收范围。
明确这是功能需求，没有把现有原型停滞定性为组件 bug，也未宣称完整 CSA 通过。

REST 回读确认作者 nalinaly、状态 OPEN，标题和完整正文与本地文件一致。
创建命令请求 enhancement 标签，但回读标签为空；不以标签缺失重复创建 issue。
全部材料在 upstream_issues/incore_p2p_notify_wait_20261016/。
没有新组件实现、设备测试、commit、push 或 PR；原优化目标的暂停状态不变。


## 2026-10-16：按用户授权实现临时普通 GM notify/wait

用户明确要求“先自己凑合着做一个，等将来主线上实现了之后再替换”，已恢复原目标。
独立 PyPTO 新增 system.signal_notify / signal_wait，普通一维 GM INT32，动态 offset
与 INT32 阈值，Set / 非消费 Ge 等待。复用 PTO-ISA TNOTIFY/TWAIT，不需要通信窗口，
不增加 syncall。CSA 只通过 local_signal_adapter.py 两个 inline 函数访问，保留替换边界。
独立 Simpler、PTOAS、PTO-ISA、pypto-lib 及共享安装本轮均未修改。

首次构建因 GetOrEmitConstant 整数重载歧义失败，改显式 int64_t 后 pypto_core 构建通过。
首轮 pytest xdist 子进程误导入共享 PyPTO，5 项 Unknown op 失败；改隔离 bootstrap
单进程执行，5 passed / 262 deselected。错误检查收窄为 ValueError 后两例通过。
以上失败与修正都保留 session_logs，不记为上游 bug。

生成 static_local_signal/h131072_b16 新候选，旧 static 目录未改。
progress 由 4608 INT64 改成 9216 INT32，总字节和每 worker 512 B 间隔不变；
保留单写者及代际协议。增加实际 arena 地址对齐检查和测试前代际范围检查。
替换通知/等待，并在消费者 cacheinvalid 后补 fence；AST 核对 AIC 40 / AIV 44 个
数学计算段全部相同，没有改计算 Tile、流水分片和原事件依赖。

static_b16_local_signal_cpu_v1 完整 kernel-mode 代码生成通过，C++ 有 TNOTIFY/TWAIT，
一次24组MIX且sync_start=true。设备任务 task_20261016_174712_328046030630 已提交，
仅测 128K/B16，三次串行调用的输出、状态、进度和保护区；不测性能及其他档位。
此时设备尚无返回结果，不预记 PASS。详细合同见 LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md。


### 临时接口首轮完整 B16 返回，进度通过但数学结果失败

设备任务 task_20261016_174712_328046030630 已 completed(exit=1)，首轮 kernel 正常返回。
72 核末次进度逐项相同；全部 Native/metadata 保护检查和工作区首尾保护通过。
输出、Top-K 与六份 cache/state 均未逐元素匹配动态参考。x_out 含20480个非有限值，
state/indexer_state 也有非有限值；动态参考无相应非有限值。第一轮失败即停止，未执行
计划中的第二/三轮，没有把进度 PASS 记成完整 CSA PASS。
summary.json 和 failure_by_rows_cpu.json 保存诊断，1.8 GiB failure_states.pt 本地保留。

为区分临时信号/数据交接与完整融合数学，补 test_local_signal_payload.py 小例：
仍为24组MIX、24AIC+48AIV，覆盖跨组AIC→AIV双消费者、AIV→AIC、AIV0→AIV1，
同时使用 Scalar 写、多缓存行 payload、matmul/MTE 写、Vector/MTE 写和后续 Scalar 读。
重复三代际但不重置 signal/payload。CPU v1 的 INDEX/INT32 混算生成类型不一致；
在测试中显式以 INDEX 做代际算术、在 API 边界 cast INT32 后 v2 编译通过，
未为此新增编译器修复。设备 task_20261016_175638_384060212210 已提交。


### 数据交接定位：AIC 发信号前必须真正排空本核流水

小例v1 task_20261016_175638_384060212210 首轮正常返回但5376/18432个checked元素不符。
只补诊断后v2 task_20261016_175919_40304807000 复现：72个信号全部为1，最终mm完全正确，
Scalar payload也正确，但部分AIV消费矩阵时仍看到初始化-999，Vector计算后为-998。
说明不能只看TNOTIFY信号或函数返回推断业务数据已完成。

生成C++显示AIC的gm fence为PIPE_MTE2/MTE3/FIX分别的pipe_barrier与DSB；它不等于
Scalar发布前的PIPE_ALL完成条件。只在CSA thin notify里增加pl.system.bar_all()，
即本核PIPE_ALL，v3 task_20261016_180315_11129226453三个代际全部通过：
AIC→AIV双消费者、AIV→AIC、跨组AIV0→AIV1，所有payload及保护区检查PASS。
这是当前小例的证据；没有增加72核全体集合，没有修改PTO-ISA或编译器的fence实现。

旧adapter归档到local_signal_variants/v1，所有四个已完成设备结果目录保存对应版本。
完整B16仅复测这一项修正：task_20261016_180421_21978513378，结果目录
static_b16_local_signal_device_v2_drain。仍只有128K/B16，没有性能测量。


## 2026-10-16：完整单 SPMD 128K/B16 三次调用通过

任务 task_20261016_180421_21978513378 已 completed(exit=0)。同一份正式权重、冻结动态
PTO 与原数学体，仅在临时 notify 前增加本核 PIPE_ALL。单个24组MIX，外部130个Tensor、
Core132个Tensor，使用native JIT直调和当前非默认流；没有增加跨核全体barrier。

三个代际均通过：x_out、idx_topk、swa、compressed、state、indexer INT8/scale、
indexer_state 对冻结动态PTO逐项完全一致；72核进度、全部Native/metadata保护区和
workspace首尾保护区通过。每轮恢复fixture，signal不清零；不是连续decode状态轨迹。
关键修正是AIC数据流水完成与Scalar发信号之间的顺序，volatile和DCCI本身不能替代它。
结果：static_b16_local_signal_device_v2_drain/result.json 与 summary.json。

本轮没有测性能，所以没有耗时或加速比；没有验证ACLGraph、动态B/S、七档或整模型。
冻结动态PTO与Native本身尚有数值/Top-K差异，x_out max_abs=0.03125；本次静态化结果
完全复现该冻结动态基线，不把它写成Native逐bit相同或整模型精度验收。

当前三项原目标（参数容量、完整融合编译、可调用单MIX JIT）均已在指定B16范围有证据。
临时接口与CSA薄封装本地保留，未来按#3020替换；不新增测试、不commit、不push。


## 2026-10-16：按用户要求启动单 SPMD / 动态 PTO 性能对比

只测128K/B16/S6，不执行Native forward。新增独立benchmark入口，共享正式权重及fixture，
捕获两边PTO图，同卡AB/BA交替采20次Event及每侧3个L1窗口。先检查计时图与动态eager
结果一致，恢复缓存排除在计时外，保留信号代际。任务 task_20261016_183457_19106108116，
结果目录static_vs_dynamic_perf_b16_v1。口径见PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md。
尚在运行，未记录性能结论；本轮没有改生产算子或依赖组件。


## 2026-10-16：单 SPMD / 动态 PTO 性能比较完成

任务 task_20261016_183457_19106108116 completed(exit=0)，8号卡，128K/B16/S6，第4层正式权重，
同进程同卡AB/BA交替，未执行Native forward。两侧图重放与动态eager输出/Top-K/六份cache-state
均完全一致；静态累计30代际，72核进度、保护区通过。每步恢复fixture，不是连续decode。

L1三次：动态897.70/900.14/877.42 μs，单SPMD1519.30/1380.22/1350.06 μs；
中位数897.70→1380.22 μs，耗时增加53.75%。每侧5次预热+20次Event采样，中位数
942.62→1467.75 μs，增加55.71%。所有样本保留。L1中动态66 task/973核记录，
单SPMD 1 MIX task/72核记录，两边覆盖72物理核。退步已在incore首尾范围内，
尚未用内部阶段时间戳区分具体开销。本轮只测量，没有继续修改算子或依赖组件。

原始result.json、summary.json和六份L1泳道位于static_vs_dynamic_perf_b16_v1；
报告PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md，重命名下载汇总在该结果目录download/。
