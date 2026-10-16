以下为主验证日志第 603 节起的记录，仅调整相对链接以适应归档位置。

## 603. 2026-10-15：CSA 单 SPMD 的七档准备与独立编译限制

冻结本工作树 HC_pre 整层集成对照中的 baseline_ops 作为计算来源，正式 W8A8 权重、
第4层/S6。任务 task_20261015_204423_7444918488 与
 task_20261015_211437_18279799282 完成七档原动态 CSA 的同轮 block/subslot/worker 映射；
原图重放、cache/state 与保护区检查通过。每档66个计算调用、72个worker、98条计算依赖。
DEBUG补采仅用于结构绑定，不作为最佳性能排程或静态性能证据。

导出73个物理计算体并通过解析；七档实参/分片绑定全部通过，长/短history分别核对
292/293个Tensor实参。静态表保留逐核顺序、分片与等价依赖偏序，CPU代际模型通过。
七档完整program源解析通过；测试适配器meta检查通过，Native别名及四份NZ权重对象保持。
没有修改依赖仓库或生产CSA/HCA/HC_pre已有改动，没有运行静态CSA或重测性能。

显式Group原型的B4部分产物有且仅有一次24组MIX提交、sync_start=true；74个进度
while保留初值/条件/yield/carry。这不表示自动MIX拆分的PyPTO #3009已修复。
完整编译未通过：PYPTO规划器Left使用81920 B，上限65536 B；DSA_RP未找到布局。
PTOAS路径修正本地视图表达后，生成的qk_pv AIV MLIR仍有SSA定义不支配使用点。
另外，原型核参数132/133超过Simpler CORE上限32；真实头文件CPU容器复现确认。
同源JIT入口当前被一个SPMD内两个InCore调用检查拒绝，未注册到kernel runtime。
不能声称只修复#3009便可运行；恢复应先解决接口和编译，再功能、精度、性能。

[完整准备清单](README.md)；
[编译证据及接续步骤](COMPILE_AND_RESUME.md)；
[CORE参数容量复现](SIMPLER_CORE_ARGUMENT_CAPACITY.md)。


## 604. 2026-10-16：静态 CSA 内存规划缩为两段复现，JIT 表达与 Buffer IR 接续检查

继续过程603的单SPMD准备。抽取Indexer score/top-k AIC和O-proj A两个原阶段，
数学和pipeline不变；单段分别10/4个Tensor参数，均完成CPU代码生成与PTOAS C++生成。
顺序融合后仅13个Tensor，PYPTO报告Right 81920 > 65536 B，DSA_RP也找不到布局。
MemoryReuse后单段Left/Right为60/64、64/64 KiB，合并为80/80 KiB。
因此片上规划失败独立于32参数上限，不能归因为算子单段超出A3能力。
这些是编译诊断，未包含数据生产者和配对AIV，不能作为设备功能或精度PASS。

完整B4额外检查了Buffer IR。修正生成器分支末尾的多Tile元组表达为安全逐项赋值，
保留循环carry和所有数学；七档完整源重新解析通过。Buffer IR路径通过存储合法化后，
生成Mat→Mat和shape不匹配的tmov被PTOAS拒绝；普通PTOAS路径仍有qk_pv AIV的SSA作用域问题。
最新部分产物仍恰好一次24组MIX提交，74个进度while状态完整，整体编译仍未通过。
5参数JIT opaque Group wrapper也独立验证，编译器未将它识别为可提交的Group核。
没有修改依赖仓库、生产算子或权重；没有新增NPU任务或性能数据。

[两阶段分配复现及对照](MEMORY_REUSE_PHASES_REPRO.md)；
[JIT Group表达记录](JIT_SINGLE_GROUP_NOTES.md)；
[完整编译接续](COMPILE_AND_RESUME.md)。


## 605. 2026-10-16：独立组件复现包与受阻接续审计

完成22份纯文本材料的组件复现包，约40KB，包含两阶段内存规划、JIT Group、CORE容量
以及#3009 while状态复现。包内独立驱动三个用例仅做解析，全部通过；已去除vLLM和
本地SDK启动脚本依赖。没有重跑已有编译对照、七档采集或NPU测试，没有权重/快照/bin入包。

再次只读核对：#3009仍OPEN，本地while活跃性种子仍未包含condition，Simpler CORE上限仍32。
此前七档静态签名实际为132/133，完整融合编译与JIT入口限制仍在。相同组件前提已连续
三轮核实；准备材料保留，目标未完成，进入受阻接续审计。恢复须先具备对应组件支持，
再执行完整B4 CPU编译、设备协议/功能、七档精度，最后性能。

[组件复现包](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/component_repro_kit_20261016.zip)；
[接续总览](README.md)。


## 606. 2026-10-16：新目标第 1 步，核参数容量来源和扩容影响

恢复后的新目标是逐步解决容量、融合编译、JIT 入口；遇关键接口选择先与用户讨论。
对七档现有源码做静态参数审计：132/133 项由 60 个 Native 描述符、69/70 个临时
描述符、3 个协议描述符组成。Native 对应 56 个命名参数，AIC 实际引用 54 项，
AIV 115/116 项，两半均不引用的参数为 0；不能靠删除无用参数压到 32。

只读核对确认 32 是 Simpler 固定 ABI 容量，关联 args[48]/args[49] 上下文、
512 B 派发结构、CoreCallable wire 布局和 64 位 dump 参数掩码；不是 A3 Tensor
数量的硬件限制。本地为浅克隆，未找到最初选择恰好 32 的设计记录，不臆测动机。

新增 CPU C++ 模板探针：32/128 不能装下 133 项，160/256 可装下；后两者均与现有
dump 掩码约束不兼容。该探针只实例化通用模板，不修改生产别名，不加载设备核。
扩容方案及内存影响见 csa_static_spmd_feasibility_20261015/CORE_CAPACITY_DECISION_20261016.md。
此处就是否修改独立依赖 worktree 与用户讨论，目标保留 active；融合编译和 JIT
阻塞尚未解决。未修改依赖、未重新安装、未重跑七档 NPU 测试。


## 607. 2026-10-16：授权后在独立 Simpler worktree 完成 CORE 256 参数扩容验证

用户明确允许修改 Simpler 的早期容量约束。新分支
`feat/csa-core-args-256-20261016` 从调试基线 `ef6812891` 建立，未改共享安装环境。
CORE 扩为 256 Tensor + 16 scalar，上下文改为 272/273，dump 使用 5 字掩码，
派发结构、Graph 边界、HBG 元数据、A5 调度偏移及 dep_gen 同步更新。

A2/A3/A5 的 HBG/TMR onboard/sim 构建通过；14 个 C++ 目标通过，Python 回归
281 PASS/4 skipped。A3 两套 runtime 的 program/kernel 均完成 133/256 参数 MIX
执行验证；kernel 每档三次 launch，输出与保护区全部通过。四份单轮选择性 dump
确认索引 63/64/128、尾部 Tensor 和第 16 个标量完整保留，无截断/丢记录。

验证期间定位并修复 CCE 自定义编译缺少 ABI 头文件依赖：TMR early-dispatch 曾
使用旧 scalar 偏移，补齐构建依赖并重编译后通过。另一次 507033 是内核提交前
设备子进程启动超时，独立进程复跑恢复，未混同为容量缺陷。

详细记录：csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md。
证据：results/csa_static_spmd_feasibility_20261015/simpler_core256_v1/。
原目标的融合编译和 JIT 单 MIX 入口仍未完成；本次不记整层 CSA PASS，不跑性能。

## 608. 2026-10-16：JIT inline Group 与融合内存分配进一步归因

容量扩容验证沿用第 607 节，不重复上卡。现有 `@pl.jit.inline` wrapper 可在前端
保持一次 SPMD 调用，并由后续 pass 形成 Group。5 参数、不共享 InOut 的小例子
`jit_group_inline_v1` 完整 CPU 代码生成通过，一次 24 核组 MIX、sync_start 保留；
`jit_group_inline_meta_v1` 的 torch 注册、Meta 输出别名检查通过。

B4 JIT 生成器改用 inline wrapper，按 InOut 规则接回两次成员调用的返回值。
v6 通过 SPMD 与 InOut 检查，但在 Group ABI 统一后出现 `progress__ssa_v1` 的
未定义别名。另提取 5 参数、无 while 的 `repro_jit_group_shared_inout.py`：
pass24 参数与返回正常，pass25 ExpandMixedKernel 首次将 AIV 的返回变成
`c__ssa_v1__FREE_VAR`，之后 codegen 拒绝。定位 BuildGroupAbiAdapter 只替换
Group 参数，没有追溯前一个成员的返回别名。详细说明与建议验收已写入
csa_static_spmd_feasibility_20261015/PYPTO_GROUP_SHARED_INOUT_REPRO.md。
这与 #3009 的 while 标量 liveness 不是同一个复现，不混同记录。

内存问题新增三参数复现：阶段一 Left 48+12 KiB，阶段二 32+32 KiB；PYPTO 与
PTOAS 分配失败，DSA_RP 完整 CPU codegen 通过。真实 O-proj 的直接 DSA_RP 编译
会改变自动分块，产生同时存活的两块 64 KiB Left，故失败；固定原 PYPTO Tile IR
后只执行 DSA 分配，O-proj 及 Indexer+O-proj 两段均通过。后者实际分配上界为
Left/Right 各 64 KiB、Acc 100 KiB、Mat 256 KiB。该结果只是分配诊断，未记完整
融合编译或功能 PASS。完整命令和差异见 MEMORY_REUSE_PHASES_REPRO.md。

v4 的 inline 生成脚本导入失败后误用了旧源码，不是新表达失败证据；v5 未接回
InOut 返回值，被前端拒绝。对旧 v15 打印 IR 的 full_b4_fixed_tiles_dsa_v1 诊断
因历史 free variable 无法重新解析，没有进入地址分配。以上均单独注明范围。

非共享 InOut 的注册入口真机测试已提交 task_20261016_095538_391201623768，
当前因其他任务占满 16 卡仍排队；不预记执行通过。记录位置为
results/csa_static_spmd_feasibility_20261015/jit_group_inline_device_v1/。
测试只验证单 MIX 注册、当前 stream、双 AIV、三次执行及保护区，不跑整层性能。
用户此前禁止擅改 PyPTO，本次授权仅覆盖 Simpler；已询问是否允许独立 PyPTO
worktree 修复上述已复现的问题，确认前保持依赖源码只读，目标仍 active。


## 609. 2026-10-16：独立 PyPTO 修复及共享 InOut 单 MIX 真机通过

用户明确授权“另开 PyPTO worktree 修复”。创建
`feat/csa-static-spmd-fixes-20261016`（基线 cb2484471），所有构建在自己的
worktree 完成；共用 PyPTO/Simpler 安装、PTOAS、PTO-ISA、pypto-lib 未修改。
独立 Simpler 扩容版为本地提交 57e499238；PyPTO runtime 子模块、SDK revision
和重新构建的 torch_npu adapter 均同步到该 ABI。未放松 clean SDK 检查，未 push。

已修复并验证 Group 共享 InOut 的 SSA 参数别名；48+12 / 32+32 KiB 分阶段
arena 扩展；嵌套 phi 后纯别名的旧缓冲传播；PTOAS reshape/transpose_view
使用分支内旧句柄；共用 Mat 句柄因后续 transpose 被误判而生成的 Mat→Mat 拷贝。
原 Indexer＋O-proj 两阶段默认 PYPTO 完整代码生成通过，未修改源 tile 或 stage。
完整 B4 的 Vec 从 267008 降至 201472 B，仍超过 188416 B；后续分支生命周期
与行列向量 carry 搬运修复正在验证，完整 CSA 尚未记 PASS。

`task_20261016_104815_276112411679`：共享 InOut 的 5 参数小例子，24 组双 AIV，
当前非默认 stream，连续三次执行，输出、保护区及输入不变检查均通过；只提交
一次 MIX，保留 sync_start。结果为
`results/csa_static_spmd_feasibility_20261015/jit_group_shared_inout_device_v3/result.json`。
前两次设备任务分别在 clean SDK / 旧 adapter ABI 检查阶段停止，没有执行设备核。

CPU 定向验证分批通过 207 项基础回归、228 项代码生成相关回归；新增分支别名
修复后相关 memory/codegen 文件 165 项通过。隔离 bootstrap 仅重定向当前进程，
pytest 使用 `-n 0`，避免子进程重新导入共享环境。本轮一次 xdist 试跑因此作废，
之后使用本地扩展串行重跑，不能把该次失败当成组件回归。
分阶段 L0A 数值任务 `task_20261016_105329_294883330178` 当前排队，尚不记 PASS。
详细根因、命令与后续更新见
`csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md`。


## 610. 2026-10-16：分阶段复用真机通过，QK/PV 剩余容量及完整过程归档

独立 PyPTO 继续修复同窗口 phi、分支 allocation 别名、PTOAS 列 carry view，
以及仅在超限时使用反向分支遍历消除假干涉。保留同分支尾读、循环再次读取和
hazard/pipeline 约束。最终 memory/codegen 相关三文件 378 passed、9 warnings；
原先分批 207/228/165 计数有重叠，不相加。无效 xdist 及测试脚本错误单独登记。

B4 默认 planner v18 为 Vec 197632 > 188416 B；PTOAS v17 已越过 SSA、
Mat→Mat 与 shape 错误，进入 Mat 540672 > 524288 B；正式 DSA_RP v19
未找到 AIC 容量内布局。20 参数 QK/PV AIV 单阶段同样 197632 B，历史动态
编译产物为 181248 B，差 16 KiB，尚不能认定硬件无解。固定该单阶段 Tile 的
DSA 诊断已实际运行分配，canonical greedy 仍未找到布局。

任务 task_20261016_105329_294883330178 已结束 PASS：分阶段 L0A arena
复用核对 192/48/128/128 行，max_abs=0。之前共享 InOut 单 MIX JIT v3 PASS
仍有效。两者都不是完整静态 CSA 验证，完整核的设备功能、状态和性能尚未通过。

按用户新要求，所有现存尝试、失败、issue/PR 状态、构建/测试日志和补丁统一
整理到 csa_static_spmd_feasibility_20261015/，并同步到主仓同名 tests/pypto_test
目录。入口 PROCESS_LOG.md；ALL_ATTEMPTS.md 逐次列目录与真实结果；
ISSUES_AND_PRS.md 区分 #3009 OPEN、Simpler 本地提交与 PyPTO 未提交补丁。
没有新建 PR、没有 push；大 IR/二进制/快照保留本地路径，不做内容 hash。


## 611. 2026-10-16：完整 B4 单 MIX 代码生成通过，Torch 注册遇到 64 入参限制

将 QK/PV 与历史动态产物逐 op/shape/dtype 对齐，定位纯 SSA identity 的旧
MemRef 阻断循环 carry 目标向 producer 传播，额外产生 6 次 32 KiB 状态
搬运，并使 BF16 归一化结果另占 16 KiB。新增无冗余搬运断言首先复现失败。
第一次取消窗口门槛后，QK/PV 及完整 B4 v20 代码生成通过，但累加器不一致
别名的反例不再拒绝，定向回归 168 passed / 1 failed；该候选未作为最终修复。

最终仅在循环 carry 传播中允许追踪旧 identity 窗口，保留 producer liveness、
no-alias、view-offset 及累加器合并门槛。相关两文件 169 passed / 7 warnings。
QK/PV 单阶段 Vec 恢复到历史 181248 B；完整 B4
`static_b4_jit_cpu_v21_carry_scoped` 默认 PYPTO 完整代码生成 PASS，AIV
tile.alloc 大小求和为 163872 B。IR 审计为一次 24 组 MIX 提交、74 个轮询
状态完整、无 FREE_VAR、sync_start 保留。没有改数学、源 tile 或源 pipeline。
这是到 PTOAS 生成 C++ 的 CPU 结果，尚未完成完整 CCE 二进制或设备执行。

真实 `pypto.torch.register` 的 Meta 驱动在定义 schema 时失败：当前 Torch
build 最多 64 项入参，完整入口为 132 项。独立 CPU 小例子确认 64 成功、
65/132 失败。Tensor[] 可使 132 Tensor 经两个 schema 参数完成 CPU/Meta
调用，但可变列表 functionalization 失败，未记为 PyPTO 集成方案通过。

按原目标“遇到关键点先讨论”，已询问保留 56 个 Native Tensor 加 6 个
分 dtype workspace、合计 62 项具名根入口的方案。内部继续使用独立描述符，
该方案尚未实现或验证；等待答复期间只整理记录，不修改共享 PyTorch。
新问题、命令和原始结果见
`csa_static_spmd_feasibility_20261015/TORCH_SCHEMA_CAPACITY_20261016.md`。
没有新增设备任务或性能数据，没有新增上游 issue/PR，没有 push。


## 612. 2026-10-16：完整 B4 现行 132 参数产物二进制编译与组装通过

等待 Torch 入口方案讨论期间，继续与该选择无关的二进制编译验证。复用
第 611 节 v21 默认 PYPTO 产物，复制 C++/配置到新结果目录，不重跑前端和
历史回归，不改变 132 参数接口。`static_b4_binary_v1` 在 115.418 秒内完成
AIC/AIV 的 CCE 编译链接、AICPU 调度交叉编译及 132 参数 callable 组装。
两个可装载核的代码分别为 58548 B、146956 B，调度共享库为 861416 B。

本项是 program 模式产物的 CPU 编译验证，不创建 Worker、不执行 NPU，
不代表完整 kernel 模式注册、进度可见性、状态、精度或性能通过。当前 Torch
64 入参限制仍在；56 Native + 6 workspace 候选尚未实现，等待用户讨论。
无新的依赖源码修改、无新增 issue/PR、无 push；二进制仅在本地结果目录。
驱动、完整命令、原始结果及大产物路径同步到用户指定主仓目录，详见
`csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md`。


## 613. 2026-10-16：Torch 入口选择待讨论，接续阻塞审计

上一轮的完整 B4 二进制编译已成功结束，没有本会话存活的编译/设备任务。
本轮只读确认：现行入口依旧为 132 参数；当前 Torch 的 64 项 schema 限制
没有变化；独立 PyPTO 注册代码尚无新的接口修改，56 Native + 6 workspace
方案也没有用户答复。没有重跑已有测试，没有新增二进制或性能数据。

相同阻塞连续出现三轮。已完成与入口选择无关的 CORE 扩容、最小 JIT 设备
验证、完整 B4 代码生成和二进制编译；下一步按原目标的关键点讨论要求等待
用户选择入口方案。目标未完成，实际状态及逐轮证据见
`csa_static_spmd_feasibility_20261015/goal_audit.json`。


## 614. 2026-10-16：用户选择 native 直调，核对 #3013 并上报必要编译 bug

用户明确取消 62 项 workspace 候选，要求使用现有 native binding；已核对
JITFunction.__call__ 经 nanobind adapter / OpCommand 入队的直调路径，及
TorchAir 的 pybind、torch_library 示例与八种内核启动组合。完整 CSA 的
132 参数直调设备验证仍待做，不再以等待入口选择作为当前阻塞。

用户要求先提交必要 bug，扩容等改动继续本地保留。本轮已创建并回读确认
PyPTO #3014（Group InOut）、#3015（PTOAS 分支 view）、#3016（条件 gather phi）、
#3017（列 carry 目标 view），每份包含完整最小复现、修改建议及约束。
在未打本地补丁的 PR #3013 head 上五项回归全失败，本地修复版五项全通过；
对照已补入 issue。Simpler 扩容和 PyPTO 内存规划改进未提交上游；构建依赖
已关联现有 Simpler #2331 / #2340。没有 push 组件分支。

另在独立 worktree 自行构建 PR #3013 精确 head 72b7ffb，七个定向回归通过，
本会话原 scalar/tile/保留结果三种 while 复现均完成代码生成且无 ready FREE_VAR。
首次缺子模块头文件及旧 editable 清单缺新模块属于环境准备失败，修正后结果有效。
没有新 NPU 任务、完整 CSA 数值结果或性能数据。详见 csa_static_spmd_feasibility_20261015/
下 ISSUES_AND_PRS.md、PR3013_REVIEW_20261016.md、DIRECT_NATIVE_ENTRY_20261016.md。

## 615. 2026-10-16：保留静态原型，仅验证 128K/B16 的完整 native 入口

用户确认当前静态代码保留，不删除；本轮只测试 history=131072、B16、每请求
6 个 query token（T96），动态 batch_size/seqlen 在静态验证之后处理。生成模板
目前仍特化 shape、循环范围及逻辑 block 数，不把静态通过当作动态实现完成。

新增完整 native 直调驱动与队列脚本，沿用正式权重第 4 层、冻结动态 CSA 和
Native 缓冲。静态入口不经 Torch schema，检查完整输出、六份 cache/state、
只读 metadata、allocation/workspace 保护区及跨调用进度代际；暂不测性能。

首个 B4 尝试在测试绑定中误把 constexpr 当作 runtime 输入，未进入静态核；
修正测试驱动后按用户指定转至 B16，没有重跑 B4。B16 首轮使用旧 wrapper，
遗漏 inline Group，前端报告两次 dispatch；仅重新生成 B16 wrapper，核体与
入参绑定不变。完整 kernel-mode v2 代码生成通过，但设备 v2 在 preflight
因两个 Native cache 子视图的可写部分重叠被拒绝，静态核仍未发射。

现由根入口从 Native 原 cache 派生 view_98/99，外部 130 项、Core 132 项，
不增加复制、缓存或 device task，不关闭别名检查。v3 发现 pl.tensor.reshape/
slice 不能进入 JIT 局部元信息推导，改为公开 pl.reshape/pl.slice 后，v4 完整
kernel-mode 代码生成通过，C++ 仍一次 24 组 MIX 提交，sync_start=true。
AIC/AIV/Group 计算体逐 AST 不变；每轮旧代码、错误与结果均保留。

续测任务 task_20261016_155540_123832232627，目录
results/csa_static_spmd_feasibility_20261015/static_b16_native_direct_v3。
此刻仅提交任务，未记静态完整设备 PASS；后续结果另行追加。详见
csa_static_spmd_feasibility_20261015/PROCESS_LOG.md。

上述设备任务已结束（exit=1）：二进制编译与 native 启动完成，设备调度日志为
S1:running-stalled，completed=0/1、running=1、orch_done=1，最终返回 507018。
静态完整输出未返回，无功能、精度或性能 PASS。原 result.json 的 device_execution
只有同步成功后才置 true；本轮 false 不能解释为没有发射。另存 failure_analysis.json
区分发射与完成状态，后续先定位跨核进度或原内部事件的停滞。

用户转来的动态范围及模板边界分析，保存到
csa_static_spmd_feasibility_20261015/DYNAMIC_SPMD_REFERENCE_20261016.md。
已只读核对主仓三个位置和 B16 生成代码的长度/位置读取；本次记录不扩展测试范围，
也不把目前停滞归因于尚未验证的动态边界问题。

## 616. 2026-10-16：完整 B16 停滞的控制协议诊断

保留完整静态核，仅从原源代码抽取 records/waits、逐核顺序与进度发布/等待，
移除数学段并加入有限轮询、失败字段。诊断为 3 参数 program 显式 AIC/AIV
单 24 组 MIX，不能作为完整 132 参数 native CSA 通过证据。

初版因常量写 INT64 缺 cast 在 CPU 失败，未上卡。v2 CPU 代码生成通过，
task_20261016_162004_206538014143 设备正常返回，但 72 核均记录等待超时。
17 个失败等待的生产者在最终快照中已达到目标，提示轮询可见性或执行时序问题；
尚无优化 IR/汇编证明，不能直接认定 CSE/LICM 或具体组件责任。

准备两组同表、同轮询预算对照：v3 只把诊断生成 C++ 的 progress 标量访问
设为 volatile；v4 使用现有 pl.system.fence，在消费者失效缓存后补流水屏障。
分别为 task_20261016_162414_222408211822 和 task_20261016_163312_271626813951，
当前待 CI 释放设备后执行。没有修改完整静态源、数学算法或依赖组件。
不重提排队任务，不重跑 B4 或七档；两组结果后继续完整 128K/B16 验证。

诊断说明、失败尝试及接续约束见 csa_static_spmd_feasibility_20261015/
B16_PROGRESS_DIAGNOSIS_20261016.md 和 PROCESS_LOG.md。

## 617. 2026-10-16：Scalar 缓存协议复核，区分原子更新与发布

第 616 节的排队状态已过时：v3 读写 volatile、v5 仅读 volatile 均在相同 B16
控制表诊断中完成 3 个代际；v4 消费者 fence、v6 仅写 volatile 均首轮有限等待
失败，设备返回。任务和所有失败明细见进度诊断文档，当前没有未结束的本会话任务。
这些用例不包含数学体，不能证明完整 CSA 的同步、状态或精度正确。

用户指出 A3/A5 Scalar cache 同样非一致，要求先读指定 Simpler 指南。已保存三个
usage guide 的来源快照，并审查生成器的发布/轮询顺序和 168 个整缓存 DCCI
静态调用点。停止推进 volatile API 或完整设备续测；没有新增组件源码修改。

按用户补充核对 A3 syncall(SOFT)：st_atomic 处理多个核并发累加同一个计数器，
DCCI + DSB 处理该路径的发布与完成顺序，ld_dev 绕过本核 Scalar DCache 轮询。
CSA 每 worker 独占进度行的方案没有多写者更新竞争，可以不用 atomic RMW，
但仍须建立正确发布、读侧新值获取和 payload 同步。不能把 volatile 当成 coherence，
也不能把“没有 A5 原子接口”泛化成“A3 没有原子累加指令”。

保留逐前驱依赖，不改为无依据的 72 核全体 barrier。当前只完成源码审查和记录，
未恢复实现或测试。复核文档：
csa_static_spmd_feasibility_20261015/SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md。

## 618. 2026-10-16：提交单个 InCore 内核间点对点通知/等待功能需求

按用户明确授权创建 PyPTO #3020：
https://github.com/hw-native-sys/pypto/issues/3020。
中文正文区分 syncall、Cube/Vector 事件与跨 rank notify/wait，提出普通 GM
信号上的单 SPMD 核间依赖接口，说明动态进度、代际、单写者多消费者、信号发布
与业务数据同步合同，以及 MIX/编译器语义和最小验收场景。
已回读确认 nalinaly 发布、OPEN、标题正文一致。需求与本地缓存协议诊断分开，
没有新增实现或设备测试。正文、来源与回读证据保存在
csa_static_spmd_feasibility_20261015/upstream_issues/incore_p2p_notify_wait_20261016/。


## 619. 2026-10-16：临时普通 GM 点对点接口与完整 B16 候选

用户授权在主线 #3020 落地前先实现临时 notify/wait，原缓存复核暂停已解除。
独立 PyPTO 新增 signal_notify / signal_wait，通过普通 INT32 GM Tensor 复用
TNOTIFY(Set) / TWAIT(Ge)；只处理信号，payload fence/cache维护显式保留。
没有改共享环境、Simpler、PTOAS、PTO-ISA 或生产 CSA。

5 项隔离定向单测通过；首次构建重载歧义、首轮 xdist 导入错误均保留记录。
独立 static_local_signal/h131072_b16 候选使用同字节数的 INT32 进度区，保留旧静态版。
AIC 40 / AIV 44 个数学段 AST 相同；完整 CPU kernel-mode 编译通过，仍一次24组MIX。
设备任务 task_20261016_174712_328046030630 已提交，只测128K/B16三次串行调用，
结果未返回，尚无设备通过结论。说明见
csa_static_spmd_feasibility_20261015/LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md。


第619节设备回报：task_20261016_174712_328046030630 首轮正常返回，72核进度全对，
全部保护区通过；输出及cache/state存在错误和非有限值，完整CSA未通过，第二/三轮未执行。
现场保存于static_b16_local_signal_device_v1，未重跑其他档位。
新增24组MIX的信号+Scalar/MTE payload小例，CPU编译通过；设备
 task_20261016_175638_384060212210 正在验证重复代际及多个消费者。


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


## 620. 2026-10-16：完整单 SPMD 128K/B16 三次调用通过

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


## 621. 2026-10-16：单 SPMD / 动态 PTO 性能比较完成

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
