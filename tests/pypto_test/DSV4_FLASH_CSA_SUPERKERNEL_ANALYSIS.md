# PyPTO 独立编译 task 与动态同核交接：Native 参考及执行方案

起始日期：2026-10-09；更新：2026-10-11。状态：局部Set/Wait及单卡CSA功能已有证据，完整CSA稳定净收益尚未成立。此前原型改变核选择、ACK后派发及用户early-resolve表达，已被用户明确否定；当前按下列约束收敛，历史结果不能作为新实现验收。仅做TMR；按用户最新要求，本轮单卡对比扩展为128K×B4/B8/B16/B24及8K×B16/B24/B32，覆盖此前仅128K/B16的限制。HBG与16卡不在本轮范围。

**最新实施约束（2026-10-11，优先于后文历史方案）：**

1. early-start只作用于正常调度已经dispatch到同核的后继，不改任务队列、优先级、核选择或block分配。
2. 保持用户原有`allow_early_resolve`表达；Set/Wait和编译器内部能力不能隐式开启该开关。
3. 不增加ACK触发派发；ACK可以因真实完成约束延后，ACK之后仍走原流程。
4. 无合格后继必须执行原有全流水等待，再FIN。2026-10-11用户已接受必要固定开销，选择低开销有效候选保留，不再继续追逐入口零成本。
5. 若发现CSA某条真实交接中后继Scalar超过前驱剩余流水时间，报告task、区间和证据，不能笼统把它作为普遍回退理由。
6. 参考AscendC的局部Set/Wait和最小编译边界约束；区分空asm的编译约束与真正设备指令，不以源码没有指令证明零运行成本。
7. 本核流水前驱不必是后继的数据生产者；对已正常派发的gated pending后继，局部Wait分别保证本核流水完成和完整fanin就绪，不能用直接WAIT边附加限制其安全Scalar交接。
8. 覆盖目标是让所有已early_dispatch且存在安全Scalar前缀的task获得提前执行机会；MIX、sync_start和idle路径的缺失必须标为当前实现限制，不能当作理论禁止或隐含验收范围缩减。提前派发本身不允许读取未就绪数据，也不保证仍有可重叠的本核流水。

基础方案选择见[实现日志过程046](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#48-过程-046接受固定开销保留指针交接位方案)。
当时保留PyPTO `320beb494`、Simpler `66d6ce63e`、PTOAS `4a5ce22a`：单block核内额外均值0.067 μs，
32次图内均摊额外0.352 μs。用户接受此项代价后，零开销goal已暂停；这不等于证明额外开销必然不可消除。
冷函数方案未保留，尚未验证的独立物理双入口改动已撤掉。
过程047按用户要求跳过已有cache去重，只补同步后继的重复控制行写回和tail kind重复读取。
过程047/048的私有PyPTO为`1f8206d09`、Simpler为`d00f8f421`；3项单卡基础功能通过。
过程048进一步完成同SDK七档CSA对照：均值四档改善、三档回退，最大值五档下降、两档增加，
长短档8:2加权耗时仅减少0.1569%；128K/B16改善1.50%，128K/B24回退1.01%。
七档两版输出、状态及改变输入后的重放逐bit一致，保护区通过；整体稳定净收益尚未成立。
[完整表格与原始样本](superkernel_csa_seven_handoff_cleanup_20261011/README.md)。
过程050已移除gated交接必须存在本核直接WAIT生产者的额外限制；当前私有PyPTO为`fdac65f66`、
Simpler为`5c8418a02`。跨核数据依赖最小用例两轮均48/48发生无关本核流水与Scalar重叠，
完整数据等待、输出及保护区通过；相关10项调度单测和既有3项单卡回归通过。
过程050仅验证功能，[记录与证据](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#52-过程-050移除本核前驱必须是直接-wait-生产者的额外限制)。
过程051已在该SDK完成七档复测：均值四档改善、三档回退，最大值五档下降、两档增加；
128K平均减少1.0164%，8K减少0.2525%，按8:2加权减少0.8636%。
128K/B16均值回退0.07%，8K/B24和B32分别回退1.26%和0.31%；功能及保护区全部通过。
本轮与上一轮的普通基线也有漂移，不能把两轮收益相减当作解除限制的精确因果收益。
[最新七档min/mean/max、全部样本及复现](superkernel_csa_seven_unrelated_pending_20261011/README.md)。

截至Simpler `5c8418a02`，仍有以下覆盖限制，后续须与真正的安全条件区分：

| 条件 | 当前行为与性质 |
| --- | --- |
| 安全Scalar前缀/Wait边界 | 由编译器验证；未就绪输入读取和前驱仍占用的资源复用必须留在等待之后，这是安全条件 |
| 前驱异步尾部 | 已在原函数末尾等完流水就没有本核重叠窗口；当前由局部Set生成相应能力 |
| pending且本核前后均为同类型纯AIC/AIV | idle核仅准备运行时参数、没有执行算子Scalar前缀；MIX及其成员交接未接入，属于实现覆盖不足 |
| sync_start后继 | 当前直接排除；目标是在保留整组计算启动语义的前提下允许安全Scalar先行，不能直接删除同步 |
| 仅在前驱返回时探测一次 | 后继在探测之后才派发，即使前驱流水尚未完成也会错失交接；属于探测时机限制 |
| 泳道/PMU/参数dump启用 | 当前关闭实际交接，观测工具仍使用串行task归属；本轮性能比较必须关闭这些功能 |

带predicate的消费者原调度器本就不提前派发，不把它重复算为新增early_start限制。
以上是源码条件清单，当前没有CSA逐项拒绝计数，不能据此编造每项漏掉的任务数或性能收益。

**最新接口要求：当前可以在原task内手工插入局部同步点，全自动边界识别作为未来方向。**
用户进一步澄清：问题首先在于原型要求整体改变InCore task写法，不能以拆helper、复制计算主体、
手写`if ready`或AIC/AIV专用异步属性作为接入条件。近期方案改为参考SK/PDL的局部协议语句，
原有SPMD、数学表达和流水保持原写法，隐藏入口及完成协议由编译器/runtime处理，详见第5.8节。
第5.7节保留为远期自动分析方案，不再作为近期基础demo的前置条件。
局部语句接口已实现；不同收尾协议和调度策略下的历史结果分别见实现日志，不能混为同一性能基线。
过程023补齐ready独立后继：48 AIV/24 AIC blocks各两轮全部证明同核计算重叠，
不要求虚造数据边，也不等待无关block；有WAIT边的路径仍等待全fan-in。
按用户最新要求简化开发流程，先验证必要功能与设备时序；当前 goal 的 demo 约束见 [实现日志第 1 节](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#1-目标范围与当前进度)。
Native 最小复现、TMR AIV 动态交接及图重放、单 block AIC Scalar/Cube 重叠已有证据；人工长Scalar小图测得净收益，私有CSA已从NZ O-A扩展到既有可接入的局部边界。
用户2026-10-11的七档复测要求覆盖此前仅128K/B16的限制；当前档位为128K×B4/B8/B16/B24及8K×B16/B24/B32。
CSA功能验证也只用单卡，不提交16卡整模型测试；本goal不设置16卡验收门槛。
功能与性能分开验收：功能正确且动态同核重叠已证明，不因暂未测得稳定加速而撤回路线。
性能侧单独检查公共路径和接续开销，没有实际回退则保留机制并持续优化；功能验收不等待性能显著改善。

具体修改、失败尝试、命令和结果持续记录在 [独立实现过程文档](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md)。

目标是 **task各自独立编译，保持Simpler原来的dispatch结果，由AICore执行器提前进入已经下发的同核后继，使前驱的异步Vector/Cube工作与后继的安全Scalar重叠**。
编译每个 task 时不绑定相邻 task，不预先生成 A+B、A+C 等组合 kernel；各 task 保留独立身份、依赖及真实完成语义。
PTO 整体退出时仍须保证所有输出可见；另一物理核提前准备参数不能单独证明本目标达成。

执行顺序：**恢复基线、定义协议并补齐工具链 → 独立 task 的动态接续基线 → 异步返回及真实完成 → 动态同核交接和等待后移 → 所需能力扩展 → 小用例性能 → CSA 接入与验收**。
先在独立 PyPTO 小张量上逐步实现，经过 CPU 编译、单卡功能和机制验证后再扩大范围。

## 1. 修正后的核心判断

Native 的同核连续调用、子算子内部等待位置和完成信号值得借鉴。已复核的 AOT 路径在执行前建立任务队列，
并可以调用分别编译的子入口；它不等同于必须先合并两份源码，也没有自动抽取后继 Scalar 回调。
PyPTO 的目标则要求运行时根据就绪情况决定接续关系，不能把这种决策重新固定到编译期。

**撤回“先把 A/B 放进一个编译单元，并把它作为最终接入路线”的定位。**
固定合组只保留为机制对照；即使它正确、同核且有收益，也没有证明独立 task 的动态交接能力。
动态调度和完成协议改造是本目标的必经部分，不能再列成“固定合组不适用时才考虑”的可选扩展。

`allow_early_resolve` 提供提前派发的部分条件。它没有改变当前生成函数末尾的完整等待，
也没有把 B 的用户 Scalar 移到 gate 之前，更没有分离“指令已提交”与“输出已完成”。
只打开此开关、提前 FIN 或删除生成 C++ 的 barrier 都不满足目标。

首要待证前提是：**A 能按明确的编译/调用合同异步返回，同时由运行时在真实完成后处理它的完成通知及资源释放**。
B 的可提前 Scalar 和安全等待边界由其自身编译器信息确定，实际前驱、物理核、完成 token 和代次由运行时传入。
这些是目标合同；TMR 原型已完成基础 AIV/AIC、长链和图重放验证，详见实现日志过程012～019。
过程027～029进一步在私有完整128K/B16 CSA证明动态Cube接续及结果/状态正确；后续继续按单卡范围验收。
上述功能证据与完整CSA的性能收益分别判断，不以固定合组替代动态调度。
本次进一步采用 PDL 的协议分层：提前调度许可与数据完成分开，后继自行在安全边界等待。
PDL 不提供本项目要求的同物理核保证；AscendC 继续提供流水交接和状态处理参考，不能用其固定队列替代动态调度。
用户已明确“PTOAS 该改就改”，因此必要的编译器能力纳入实现范围，旧的 PTOAS 不修改限制撤回；PTO-ISA 边界保持不变。

固定 shape 可以用于最小动态调度测试；shape 专门化与接续关系固定是两件事。
验收须证明同一批独立编译的入口，在运行时输入/任务实例变化时可以选择 A→B 或 A→C，无需编译组合入口。
已有 Native 复现只作机制参考；PyPTO 动态协议及基础 AIC 的独立证据见实现日志，真实 CSA 收益尚未证明。

## 2. 证据与版本边界

| 对象 | 机制复核时的版本/位置 | 能证明的范围 |
| --- | --- | --- |
| 当前 CSA | vLLM Ascend `8e83adda`，`vllm_ascend/ops/pypto/deepseek_v4_flash_csa/` | 唯一维护的 CSA 实现与调用边界 |
| PyPTO | `cb2484471`，`feat/kernel-mode-integration-test` | 当前 wrapper、参数解析和 PTOAS 调用方式 |
| Simpler | `ef6812891`，`feat/kernel-mode-integration-test` | TMR/HBG 的提前派发、gate、ACK/FIN |
| 安装 CANN | `/data/pyptouser/yejia/vllm-cann92-main/env/Ascend/cann-9.2.0-beta.2` | 编译 Python、AscendC 头文件、实际链接的 `libascendsk.so` |
| SuperKernel 开源实现 | `cann/graph-autofusion`，`0dc84872fc20015953ea12c0c7622dbcd23f2c2a`，2026-10-08 | 当前上游 AOT Host 优化器及设备队列执行器；不等同于已经验证本地加载的二进制版本 |
| PTOAS | 实际探针使用 `.cache/dsv4-toolchain/ptoas-0.66/bin/ptoas`；本地源码 `f5eff3ee` | 探针证明当前二进制可生成外部调用；源码解释同步策略，未断言二者提交完全一致 |

GitCode 读取禁用代理；只读参考仓使用 `--depth=1 --single-branch` 下载到 `/tmp/csa-sk-graph-autofusion-20261009`，没有更新工作用工具链。
以下安装包相对路径均相对于上述 CANN 根目录；PyPTO、Simpler、PTOAS 相对路径均相对于各自仓库。
当前 PyPTO 另有未完成的诊断改动；它们不属于上表已验证的设备能力。历史构建/导入失败见第 7.3 节；导入已修复、定向 UT 3 passed，后续开发迁至独立工作区，见实现过程 008。

## 3. Native 的完整调用链

### 3.1 TorchAir 配置进入的是图优化器

本地 TorchAir `_acl_concrete_graph/acl_graph.py:1207` 在图捕获后调用 `NPUGraph.super_kernel_optimize()`。
安装包 `torch_npu/npu/graphs.py:405` 做参数检查，再进入 C++ 绑定；安装的 `SkInterface.h` 声明 SuperKernel 接口调用。

开源 AOT 路径的入口为 `super_kernel/src/aot/super_kernel.cpp:80` 的 `aclskOptimize()`：

1. 从捕获模型建立 `SuperKernelGraph`，读取任务、依赖及子算子二进制信息。
2. `SuperKernelOptimizer::Process()` 划分可融合范围，检查图依赖和死锁条件。
3. `SkTaskBuilder::Build()` 建立 AIC/AIV 队列、同步任务和预取任务。
4. 更新原捕获图，由 SuperKernel 入口执行该范围内的子任务。

这与 GE 路径的 Python JIT 生成整段调用代码要区分。安装包中的 `python/site-packages/superkernel/super_kernel.py` 可以生成顺序子函数调用，
而本次用户指定的 `npugraph_ex` 重点应追踪 AOT 图改写和队列执行，不能仅依据 JIT 文档推断。

来源：[AOT 图优化入口](https://gitcode.com/cann/graph-autofusion/blob/0dc84872fc20015953ea12c0c7622dbcd23f2c2a/super_kernel/src/aot/super_kernel.cpp)、
[范围调度](https://gitcode.com/cann/graph-autofusion/blob/0dc84872fc20015953ea12c0c7622dbcd23f2c2a/super_kernel/src/aot/sk_optimizer.cpp)。

### 3.2 子算子须提供可被调用的入口和能力标记

安装包 `python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py:590` 生成 sub-combine 入口：
入参为参数包和 `sk::SkSystemArgs`，并从 `SkGetTaskSyncCfg()` 设置当前子算子的 Early-Start 配置。
`SkSystemArgs` 在 `arm64-linux/ascendc/include/basic_api/impl/kernel_operator_sys_var_intf_impl.h:55` 定义，包含逻辑 block index、block 数和同步配置。

同一编译器的 `compile_sk_bind()`（`compile_op.py:1602`）生成 `SK_BIND` 元信息，其中 bit 0 表示有 Wait 接口，bit 1 表示有 Set 接口。
bit 2 表示关闭该子算子的 DCCI；它不是函数数量，不能照搬示例中的 `SK_BIND(..., 4, ...)` 后认为已声明 Set/Wait。
`super_kernel_sub_op_compile.py:141` 按配置和算子能力启用对应宏；AOT sub-combine 使用 V3 同步配置。
`kernel_info_infer.py:76` 检查预处理源码中的接口使用情况。这里是识别算子已经适配的接口，不是通用的纯 Scalar 自动提取器。

设备执行器读取已注册的子函数地址，直接调用该入口；它不需要在每个子算子之间重新向 Host 启动一个普通 kernel。
普通 `__global__` 入口和 SK 可调用入口有不同的调用合同，不能把现有 PTO custom op 加到外层 SK scope 后，假设其内部所有 task 也会自动融合。

来源：[能力位及二进制解析](https://gitcode.com/cann/graph-autofusion/blob/0dc84872fc20015953ea12c0c7622dbcd23f2c2a/super_kernel/src/aot/sk_node.h)、
[直调算子适配](https://asc.gitcode.com/guide/programming_guide/advanced_programming/super_kernel/kernel_direct_call_adaptation.html)。

### 3.3 Host 先生成正常依赖，再把符合条件的同步移入子算子

`super_kernel/src/aot/sk_task_builder.cpp:2357` 在正常同步分析之后执行 `ApplyEarlyStartSyncPass()`。
其中 `ApplyEarlyStartSyncForQueue()`（1316）分别处理 AIC 和 AIV 序列：

- 前驱具有 Set 能力时，把对应同类核同步映射为前驱内部的 Set 配置。
- 后继具有 Wait 能力时，把相应等待映射为后继内部的 Wait 配置。
- 涉及 AIC/AIV 交叉依赖时，按拓扑处理跨类型的发送/接收信号。
- 同时保留需要由外层补齐的同步项；不支持的组合保持原同步或拒绝该改写。

`funcEarlyStartConfig` 随 `TYPE_FUNC` 任务的 `extraInfo` 下传；`syncEarlyStartConfig` 用于补齐同步任务。
这说明优化需要“调度器知道依赖”和“算子知道安全等待位置”同时成立，单独加一个 Set/Wait 接口不够。

来源：[Early-Start 同步改写及队列构建](https://gitcode.com/cann/graph-autofusion/blob/0dc84872fc20015953ea12c0c7622dbcd23f2c2a/super_kernel/src/aot/sk_task_builder.cpp)。

### 3.4 同一个物理核连续调用子算子

`super_kernel/kernel/sk_entry.asc:266` 的 `spk_entry_impl()` 是设备侧核心：

```text
选择本核对应的 AIC/AIV 队列
逐项读取 TaskInfo
    PRELOAD：预取子函数代码和参数
    FUNC：构造该子算子的 SkSystemArgs，直接调用函数指针
    SYNC：执行 Host 指定的同步或补齐同步
    EVENT：处理外部通知/等待
全部任务结束后完成最终同步
```

子函数调用位于约 367 行。Scalar 按普通函数调用顺序执行；计算/搬运流水有自己的异步进度。
由此可以出现 A 的 Scalar 已返回、A 的计算/搬运仍在流水中，而同核 Scalar 已进入 B 的情况。
这不需要两个 Scalar 线程，也没有“暂停 A 的 Scalar 栈，调用 B 的准备回调，再恢复 A”这一步。

来源：[设备任务队列执行器](https://gitcode.com/cann/graph-autofusion/blob/0dc84872fc20015953ea12c0c7622dbcd23f2c2a/super_kernel/kernel/sk_entry.asc)。

### 3.5 Set/Wait 实际是流水和核间同步协议

安装包 `arm64-linux/ascendc/include/basic_api/impl/dav_c220/kernel_operator_sync_impl.h:130` 的 Set 实现按配置发出 `ffts_cross_core_sync`：
AIV 默认使用 MTE3，AIC 默认使用 FIX；240 行开始的 Wait 实现使用 `wait_flag_dev` 并处理 C/V 交叉关系。

安全顺序是：A 提交计算及输出，并把完成信号排在相应流水之后；B 先计算不依赖 A 结果的地址、维度和循环参数，
再等待该信号，之后才读依赖数据或使用不能提前复用的资源。Set 不是把 A 的结果立即标为完成。
实际可隐藏的区间取决于 A 剩余的异步流水；常见的是末尾搬运，也可能仍覆盖其上游计算。不能仅凭接口存在断言 Cube/Vector 重叠了多长。

纯同类核示意：

```text
同核 Scalar： A 发出计算/输出和完成信号 ── A 返回 ── B 独立 Scalar ── 等待 A ── B 主体
同核 V/C/MTE：       A 的异步计算和输出 ────────────────── 完成
                                                     ↑
                              目标是这段与 B Scalar 重叠
```

`kernel_tpipe_impl.h:595` 在 `__ASCENDC_ENABLE_SUPER_KERNEL__` 下省去默认析构的全流水等待，其他资源清理仍保留。
它是允许子算子交接的组成部分；本任务关注的是 CSA 内部 A→B 的交接，CSA 整体输出完成条件不变。
不能笼统删除所有 `wait_flag`：必须分清接收流水、局部 buffer/event 复用及数据依赖。

来源：[SetNextTaskStart](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_task_sync/SetNextTaskStart.html)、
[WaitPreTaskEnd](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_task_sync/WaitPreTaskEnd.html)。

### 3.6 少核子算子需要补齐同步

`sk_entry.asc:223` 的带配置 `AutoCoreSyncImpl()` 对当前子算子的活跃核直接返回；未执行该子算子的核则按配置补 Set/Wait。
Host 通过 `DispatchSyncTasks()`（`sk_task_builder.cpp:2045`）传递对应子算子的活跃核数。
因此 12→48、48→12 等核数变化也能保持同步参与者完整；不能只给“真正计算的核”加信号。

这对 CSA 特别重要：实际 task 的核数不同，还有多轮 SPMD、AIC/AIV 切换和多个前驱。
直接复制 Native 的全核 flag 编号到 Simpler 动态任务池可能等待不存在的参与者，必须先建立正确的执行组。

### 3.7 当前 SK 基线不能等同于 Early-Start 已生效

官方 `npugraph_ex` 文档把 `super_kernel_optimize` 与 `early_start` 分开，后者默认关闭。
本地 `.venv-dsv4-0251rc1/lib/python3.10/site-packages/torch_npu/npu/graphs.py:429` 的参数白名单只有 `preload_code`、`split_mode`、`stream_fusion`，
没有 `early_start`；按当前文件，直接传该选项会被 Python 检查拒绝。当前读到的 CANN 头文件则已经有 Early-Start 选项。

已检查的历史 Native 七档摘要只记录了 SK 开关，没有 Early-Start 生效证据。本轮没有更换安装包或重跑基线。
所以历史 SK 收益可能包含融合调度、预取等效果，不能单独归因于 Scalar 重叠。
本次直接链接已安装的 `libascendsk.so`，走 C++ `aclskOptimize`，绕过 Python 参数白名单。
这样可以验证实际 Native 能力，无需更新 torch_npu 或更换公共 CANN。库包含 Early-Start 配置及同步改写实现；
本次由设备回写的 `0x10/0x20` 配置及 `8→1` 捕获任务变化确认实际应用，详见第 7 节；没有依赖库字符串推断生效。

来源：[TorchAir 配置文档](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md)。

### 3.8 直接编译复现时必须落实的接口条件

本次用例位于 [superkernel_repro_20261009](superkernel_repro_20261009/README.md)。它完整执行：

```text
编译两个普通 __global__ 入口及各自 __sk__ 子入口，生成 SK_BIND 元信息
普通入口预运行，完成注册/加载
aclmdlRICaptureBegin → aclskScopeBegin → A → B → aclskScopeEnd → aclmdlRICaptureEnd
aclskOptimize(model, EARLY_START / SPLIT_MODE / PRELOAD_CODE)
aclmdlRIExecuteAsync → 校验结果、读取设备实际同步配置与物理核号
```

直接调用 AscendC 公共接口时，使用 `SetNextTaskStart<PIPE_MTE3, PIPE_FIX, true>()` 和 `WaitPreTaskEnd<true>()`。
最后的 `true` 为 `FORCE`，避免本例未走自动注入编译宏的流程时接口变成空操作。
注意公开 `WaitPreTaskEnd` 只有一个 bool 模板参数；`WaitPreTaskEndImpl<-1, true>` 是内部实现签名，不能混用。
每次进入 SK 子函数都从 `sysArgs->SkGetTaskSyncCfg()` 设置本次配置，不能硬编码为永远 Set/Wait。

本例 A 的能力位为 2，B 为 1；均保留默认 DCCI。AIV→AIV 时设备配置预期包含 A 的 `0x10` 和 B 的 `0x20`。
能力位描述“支持什么”，运行时配置描述“这次调用实际要做什么”，两者不是同一组 bit。
B 的状态初始化、UB 使用和首次输入 DMA 均放在 Wait 之后。普通 kernel 的最终等待仍保留。

官方 `task_sync` 示例只启动一个普通 kernel，没有 SK 捕获/融合及相邻子算子，不能独立证明 Early-Start 重叠。
本机 `llvm-objdump` 对设备指令返回 `<not available>`，Bisheng 拒绝设备 `-S`、`-mllvm -print-after`，
因此本次不能交付设备汇编证明。预处理文件、符号表和源码调用顺序只作为编译证据；设备运行结果单独记录。

来源：[官方 TaskSync 示例](https://gitcode.com/cann/asc-devkit/tree/master/examples/01_simd_cpp_api/03_basic_api/05_sync_control/task_sync)、
[本例源码](superkernel_repro_20261009/native_early_start.asc)。

### 3.9 Native 机制与 PyPTO 动态目标的边界

| 问题 | 本次复核的 Native AOT 路径 | 本目标要求 |
| --- | --- | --- |
| 接续关系何时确定 | 图优化/队列构造时 | Simpler 运行时选取合法后继实例 |
| 是否需要知道具体后继才能编译 | 子入口提供 Set/Wait 能力，队列安排调用 | 每个 task 只声明自身能力；不得编译 A+B 专用入口 |
| 同核由谁保证 | 已构造的本核队列及设备入口 | 动态调度与 executor 的交接选择 |
| 真实完成由谁处理 | SK 队列及同步协议 | Simpler 的 block 完成、task 依赖与资源回收协议 |
| shape 与调度关系 | 参考的是已捕获图的执行计划 | 允许独立 kernel 按 shape 专门化，接续关系仍须动态决定 |

### 3.10 NVIDIA PDL：参考依赖协议，明确同核目标的额外要求

PDL（Programmatic Dependent Launch）允许同一 CUDA stream 的后继 kernel 在前驱结束前启动，
由 `cudaLaunchAttributeProgrammaticStreamSerialization` 显式启用；也支持 CUDA Graph 的 programmatic 依赖边。
两个 kernel 保持独立，重叠只是运行时机会，不能把并发作为正确性成立的前提。
[CUDA Programming Guide：PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

必须区分以下两项；API 名称里的 LaunchCompletion 不表示 kernel 数据已经完成：

| CUDA 协议 | 确切边界 | 对 PyPTO 的设计启示 |
| --- | --- | --- |
| `cudaTriggerProgrammaticLaunchCompletion()` | 全部 CTA 都已触发或退出后，允许指定后继被调度；触发本身不提供内存可见性保证 | 单列提前调度许可，不发 FIN、不回收前驱资源 |
| `cudaGridDependencySynchronize()` | 等待所依赖的前驱 grid 完成；对应等待还要保证其内存操作对后继可见 | 后继先做安全前缀，首次依赖访问之前执行完成/可见性检查 |

依据：[CUDA Runtime API：触发与等待](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html)、
[PTX `griddepcontrol`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)。
PTX 将这两项分别表达为 `.launch_dependents` 与 `.wait`；依赖对象由 runtime 定义，指令不硬编码后继函数地址。
有正确作用域的额外 fence 可以建立特定内存顺序，但触发通知本身不能被当成结果已可读。

本项目借鉴上述分层，并增加 Ascend 执行模型约束，以下属于方案推导，不是 NVIDIA 对 Ascend 的能力保证：

- PDL 不承诺同 SM 的放置关系，也没有要求前驱先函数返回；PyPTO 本目标仍要求同一 AIV/AIC。
  在当前串行调用的 Scalar executor 中，A 必须先交出 Scalar 控制权，B 才能在该核进入；首版只让 A 在提交完本 block 后异步返回。
- CUDA 示例的独立工作可以包含缓冲初始化；本项目安全前缀先限定为不会干扰前驱的 Scalar，不能直接提前复用 UB/L1/L0 或修改 atomic/mask 等核状态。
- CUDA 的依赖以 grid/依赖边表达；PyPTO 需要自己定义 block 许可聚合、task 完成和动态实例代次。
  `allow_early_resolve` 的“所有 block 已发布”不等于都执行过触发，更不等于结果完成。
- PDL 可以服务于预先声明的 stream/graph 依赖，不是任意任务图的动态调度器；A→B/A→C 的实际选择仍由 Simpler 负责。
- 强制串行时仍须功能正确；“观察到两个 task 时间条重叠”不等于本核 Scalar 与异步流水重叠已被证明。

## 4. 当前 PyPTO/Simpler 的具体缺口

| 环节 | Native 可见实现 | 当前 PyPTO/Simpler | 所需工作 |
| --- | --- | --- | --- |
| 后继可提前出现 | 预先构建本核任务序列 | `allow_early_resolve` 提前填 running/pending slot | 保留现有预派发；它是条件之一 |
| 后继开始位置 | 安全 Scalar 在子算子内部 Wait 之前 | executor 先等 doorbell，再调用整个 `kernel_entry` | 编译器明确安全前缀与进入主体的位置 |
| 同核交接 | 同一入口连续调用 | 前驱函数返回、报告 FIN 后再取下一任务；生成函数末尾已有同步 | 为各自独立编译的入口建立异步返回及动态交接合同 |
| 完成语义 | 由队列和 Set/Wait 保序 | 普通任务 FIN 触发完成、依赖释放和回收 | 不能把“Scalar 返回”冒充“计算完成” |
| 部分核与跨类型 | Host 生成补齐同步及 C/V 关系 | 动态 block 映射、双 buffer slot、`sync_start` | 显式处理参与者和代次，不能借用固定全核等待 |
| 算术与状态 | 子算子主体保持原语义 | wrapper 在入口执行 `set_atomic_none()` 等 | 提前区段必须排除会干扰 A 的状态修改 |

当前证据位置：

- Simpler `src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp:116`：参数组装后等待 doorbell，之后才 ACK 并执行 kernel。
- `runtime/scheduler/scheduler.h:956`：只有符合条件、block 已全部发布的生产者才能传播提前派发资格，防止后继占核阻塞尚未派发的前驱。
- `runtime/scheduler/scheduler_dispatch.cpp:614`：忙核可装入 pending；这时仍没有执行 B 的用户 Scalar。
- PyPTO `python/pypto/backend/pto_backend.py` 的 `_generate_kernel_wrapper()`：入口重置 atomic、解包参数，再进入 PTOAS 生成函数。
- 本地 PTOAS `lib/PTO/Transforms/PTOToEmitC/PTOToEmitCPass.cpp:809`：当前已查看的 tail 模式为全流水等待或 MTE3→Scalar set/wait，两者都不是异步返回合同。
- HBG 有对应 executor 和完成路径，不能只改 TMR 后把 HBG 标为支持。

现有 `pld.system.defer_wait` 是专门等待外部计数条件的任务能力，并没有自动定义 Vector/Cube 流水完成事件；不能直接替代这里的交接协议。
DFX 也要注意：task 函数时间、前置 gate、硬件流水时间是不同口径，仅看到相邻 task 条重叠不够。

## 5. 主方案：独立编译入口，运行时同核交接

本节定义拟实现的合同和必须验证的条件，不代表已有可调用 API。伪代码中的入口、状态和 token 名均为说明用途。

### 5.1 编译时确定能力，运行时决定相邻关系

编译器处理一个 task 时，只分析它自身：安全 Scalar 前缀、首次不安全操作、计算/搬运资源、输出可见性和可用完成方式。
不需要知道前驱或后继是谁，也不把下一 task 的源码、参数或函数地址硬编码到当前 task。

| 编译期职责（PyPTO 与可用工具链） | 运行期职责（Simpler） |
| --- | --- |
| 独立生成 task 入口及能力声明；普通执行和交接模式的合同清楚 | 选择具体 task/block 实例、物理核及是否允许提前进入 |
| 近期识别并验证作者插入的局部协议语句，远期自动识别并插入边界 | 确认前缀需要的参数/描述符已就绪，传入本次依赖与完成上下文 |
| 保证异步返回不损坏仍在使用的片上资源、事件及状态 | 保留前驱上下文，管理事件所有权、slot 生命周期和执行代次 |
| 保留 task 内部流水同步；声明哪些输出/流水被完成事件覆盖 | 仅在真实完成后通知 block 完成，按原 task 规则聚合依赖及回收 |
| 普通入口/不支持的能力保持完整完成语义 | 无合格后继时收尾前驱，继续正常调度并保证图退出完成 |

优先验证单入口形式：B 进入后执行安全 Scalar，再调用运行时边界处理，随后执行主体。
如果工具链只能表达 `prepare`/`compute` 两个入口，可以作为独立编译合同的备选，
但须显式传递一次计算的 Scalar 上下文，评估寄存器保存、内存访问和额外调用开销；不能默认把 Scalar 算两遍。
这项备选仍由运行时选择接续关系，不变成固定 A+B kernel，也不被称为 Native 的原样实现。

### 5.2 首个动态原型的准入条件

先做 AIV 单 block、单个在途前驱、同一物理核的 A→B；A、B 分别编译、分别注册、保留各自 task 身份。
只有以下条件同时满足才允许 B 的 Scalar 提前进入：

1. A 支持经验证的异步返回/完成合同；B 支持安全前缀和相应等待边界，版本及核类型兼容。
2. B 的参数包、shape/stride、逻辑 block 信息已发布且生命周期稳定；前缀不依赖 A 尚未产出的数据或描述符。
3. B 唯一未完成的数据前驱是正在本核异步执行的 A；其他依赖已经完成。不能把“另一个前驱已经派发”当作完成。
4. A 的全部剩余工作已经提交，后续完成不需要任何被 B 占核后无法运行的任务。
5. 首版不带跨核 `sync_start`、跨核归约或不同核数的集体条件；这些能力验证前不能进入此快路径。
6. 前驱完成记录与 B 的参数槽独立，事件及 token 可安全持有，executor 可以在没有后继时独立收尾 A。

每核首版最多持有一个尚未完成的前驱。资格不满足时使用普通等待/派发，保持任意合法任务图的功能。
复杂图的优化支持要按第 6 节扩展；回退正确不等于复杂图已经获得重叠收益。
现有“生产者 block 已全部发布”的 early-resolve 规则继续保留，但它不能单独证明以上准入条件成立。

### 5.3 必须分开的状态与完成责任

以下是概念状态，不预设新增枚举或复用现有 ACK/FIN 位：

| 状态 | 已经成立 | 尚不允许 |
| --- | --- | --- |
| `DISPATCHED` | 入口、参数及本次执行身份可访问 | 假定输入可读或资源可复用 |
| `EARLY_ELIGIBLE` | 前驱已授予提前启动许可，运行时准入检查通过 | 将许可当作数据可见、A 已返回或 FIN |
| `PREFIX_RUNNING` | 后继安全 Scalar 获准执行 | 读取未完成前驱输出、改变其在途计算依赖的状态 |
| `BODY_RUNNING` | 全部输入依赖及必要的同步条件满足 | 把发出指令当作计算已完成 |
| `ISSUED` | 当前入口提交完本 block 的计算及完成事件，Scalar 已返回 | 提前 FIN、释放输出或复用仍占用的事件/槽位 |
| `BLOCK_COMPLETE` | 本 block 相关流水已完成，输出达到合同规定的可见性 | 把一个 block 完成当作整个多 block task 完成 |
| `TASK_COMPLETE` | 所有必需 block/参与者完成，正常完成协议已聚合 | 在其他消费者仍持有引用时提前回收 buffer |

运行时必须区分“本核 A block 的完成”与“A 整个 task 的完成”；只有既有依赖语义认可的完成才能使 B 的主体就绪。
每个完成通知带 task/block 身份及执行代次，并且只处理一次。事件标识、pending slot 和 tensor/ring 引用必须有明确所有者。
`EARLY_ELIGIBLE`、`ISSUED` 和真实完成是不同条件，不是同一个位的三种叫法；上表也不是一条全局串行状态链。
提前许可可以先用于准备/派发，但本核执行 B 还要等 A 交出 Scalar 控制权。
未触发许可或没有可用重叠资源时，A 正常完成后仍须使后继继续运行；不能让不发生重叠的合法执行死锁。
首版仅一个 block，后续多 block 必须按依赖边聚合资格，不能拿某个本核的局部许可代替整个 producer 的进度条件。

概念执行序列：

```text
executor 调用独立入口 A(ctx_A)
    A 按合同发布提前启动许可（不是结果完成；首版可放在提交末尾）
    A 提交计算/输出及相应完成事件
    A 返回 ISSUED；不发布虚假的 FIN

executor 持有 A 的完成记录，动态选中符合准入条件的 B
executor 调用独立入口 B(ctx_B，前驱完成上下文)
    B 执行一次安全 Scalar
    runtime_boundary:
        确认 A 相关流水真实完成
        建立依赖输出对 B 的可见性；仅收到提前许可不满足这个条件
        经运行时完成路径发布 A 的 block 完成，并推进合法的 task 完成/依赖状态
        确认 B 的全部依赖和必要同步满足
    B 首次访问依赖数据/共享片上资源，执行主体
    B 提交自己的完成事件并返回

没有合法后继 / 最后一个 task / 图结束：executor 收尾所有在途记录，再按正常协议退出
```

**不能让 B 等待 A 的完成通知，而该通知又要等 B 返回才能发布。**
完成处理必须可以在 B 的边界等待阶段推进，或者由独立的运行时完成机制推进；具体实现须在最小原型中证明。
上面的过程不授权 AICore 写 DATA_MAIN_BASE、对 SPR 地址做 MMIO 或用无合同的跨核轮询替代完成协议。
正常 FIN 的对外含义不变；其内部发布时机、记录持有者、聚合与回收实现预计需要修改。
这对应 PDL 的“允许启动”与“等待数据”分离，但不照搬 GPU 的硬件调度或 CTA 上下文机制。

### 5.4 安全 Scalar、资源与事件合同

首版可提前区只包含已就绪参数解包、不可变描述符、逻辑 block 信息及独立整数计算。
标注必须由分析/检查保证；不能仅凭函数名带 Scalar 就允许提前。
`pl.read()` 访问前驱数据、UB/L1/L0 使用或重用、event/FIFO 操作、atomic/mask/dequant 等状态修改都不能无条件提前。
当前 wrapper 的 `set_atomic_none()` 必须移到安全等待后或证明不会影响在途指令；编译器自动插入的 prologue/epilogue 也在检查范围内。
跨函数调用的寄存器、栈及辅助存储必须满足 ABI，B 的安全前缀不能破坏 A 仍在使用的资源。

完成事件须覆盖本 task 实际涉及的全部输出和相关异步流水，以及必要的 Scalar GM 写回可见性。
AIV/MTE3 和 AIC/FIX 是 Native 的参考，不能直接假定“最后一个 MTE3/FIX 信号”覆盖任意 PyPTO task 的所有工作。
事件编号须与 PTOAS 自动同步的分配隔离，定义申请、消费、释放及代次规则，禁止直接借用未保留的固定 event 0。
当前 PTOAS 遇到显式 Set/Wait IR 会跳过自动插入同步，必须检查生成产物中 task 内部依赖仍然完整。

异常退出、参数失效、没有后继和图重放结束均须有收尾路径：不能留下在途输出、悬挂完成记录或错误释放的槽位。
新模式只对经过验证的 task 能力启用，不强制所有 task 同核，不全局翻转 `sync_start`。

### 5.5 工具链门槛、PTOAS 修改范围及仓库分工

已查看的 PTOAS tail 模式仍会完成等待，尚无证据证明它能提供上述异步返回合同。
先核实上游可用能力；缺少的编译器部分按用户新授权修改 PTOAS，以正式 IR/属性、验证和 lowering 实现。
不靠实验脚本后处理生成 C++ 删除等待，不引入绕开内部同步的全局无 barrier 开关。
**PTOAS 能力缺失现在是可实施的开发任务，不再以“不允许改 PTOAS”停止推进。**
PTO-ISA 仍优先使用上游能力，不做本地额外修改；如触及 ISA/硬件本身缺口，须报告具体限制，不能假装已由编译器解决。

PTOAS 最小改造须同时覆盖以下事项，名称与 ABI 在实现阶段确定：

1. 显式表达提前调度许可、后继依赖等待和支持异步返回的入口能力，区分副作用及内存顺序，防止优化将依赖读取移动到等待前。
2. 分离 task 内部流水同步与函数边界完成：保留内部同步，交接模式在出口提交能覆盖实际工作/输出的完成记录，由后继或 executor 等待。
   普通入口仍保留完整完成语义；所有返回分支、无后继、末尾 task 必须有正确收尾。
3. 为跨 task 的完成事件定义保留/分配和生命周期，与 task 内部 event 分开；不得直接复用未保留的 event 0。
4. 让新协议操作与自动同步协同，不能因为多了 task 边界操作就跳过整个函数的自动同步。
5. 检查最终 C++ 的 prologue/epilogue、Scalar GM flush 与 wrapper 状态重置；完成通知必须覆盖输出可见性，任何隐含全等待也要计入实际可重叠窗口。
6. 能力及 ABI 版本进入编译缓存标识；不支持核型/模式或缺少运行时合同的显式交接请求应报错，运行时可按已定义策略选择普通入口。

本次只读源码定位到：

- `lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp:79`：发现显式 Set/Wait/RecordEvent 后跳过自动同步，正常分析另会插入尾部同步。
- `lib/PTO/Transforms/PTOToEmitC/PTOToEmitCPass.cpp:809` 与 `tools/ptoas/ptoas_pipeline.cpp:1147`：尾部只有全流水等待和 MTE3→Scalar 的立即 Set/Wait 两类合同。
- `tools/ptoas/ptoas_cpp_rewrite.cpp:740`：Scalar GM 写回收尾插入点包括 return 和 auto-sync tail，需要随新出口合同审查。
- `lib/PTO/Transforms/Passes/PTOInjectBarrierAllSync.cpp:71`：该同步模式还会在 return 前补 barrier，须明确交接模式与它的兼容/拒绝规则。

这些是已定位的改造入口，不代表改一处 tail helper 就能完成全部协议；也未据此更换当前使用的编译器。

| 仓库 | 本目标中的必要职责/边界 |
| --- | --- |
| PyPTO，`feat/kernel-mode-integration-test` | 独立入口能力、协议 IR 与前缀/等待边界、wrapper 状态处理、PTOAS 能力匹配及基础用例 |
| Simpler，`feat/kernel-mode-integration-test` | 分开提前调度许可与真实完成，动态选取同核后继，处理 gate、完成记录、ACK/FIN、slot/事件生命周期及 DFX |
| vllm-ascend，当前工作分支 | 方案和过程记录；基础动态能力验证后，按实际 CSA task 能力接入 |
| PTOAS | 已允许必要修改：协议表示/验证、自动同步、异步出口、事件资源、内存可见性及普通模式兼容 |
| PTO-ISA | 使用/核实上游能力；本地不额外修改 |
| CANN、TorchAir、graph-autofusion、pypto-lib | 参考实现及当前环境；本轮文档修改不更换公共安装包 |

### 5.6 固定合组用例的保留边界

旧 `local_task_chain` 将 A/B 放到同一编译单元，只有一个运行时任务，不能验证动态选择、独立 ABI 或前驱完成记录的生命周期。
保留其已取得的编译证据和失败记录，作为等待位置的可选机制对照；不以完成该用例作为进入动态路线的前置任务。
不继续扩大固定合组到 CSA，不通过减少 task 数量替代本目标，也不以固定配对收益代替动态接续收益。

### 5.7 远期方向：普通算子源码不变，编译链自动生成交接协议

本节是待实现设计。已有原型仅验证运行时机制及人工指定边界的lowering，
不能因为生成了设备代码就称为“已自动识别Scalar前缀”。
`early_start`、`fake_end`在本节表示拟实现的内部协议语义，不声称上游已有同名PyPTO API或硬件指令。

**算子作者只写原来的数学计算、SPMD与依赖。** 不新增协议属性、显式等待、前驱/后继配对，
也不为获得重叠而把某段计算搬到特制helper。编译器内部可以携带能力属性，
但它们必须来自分析结果，不能只是把现有三个手工开关默认设为True。

1. **自动找到安全Scalar边界。** 在task独立outlining、核类型确定后，PTOAS针对lowering后的指令分析
   数据依赖、副作用和流水资源。已发布的参数包、block id、shape及其纯Scalar地址/范围计算可以作为前缀。
   前驱输出读取、未知GM读写、UB/L1/L0内容访问、计算/搬运指令、atomic/mask及同步资源状态修改必须在等待后。
   不能仅凭Python表达式看起来像Scalar就允许提前执行。初版识别现有连续前缀；之后再按def-use与控制流
   支配关系提升独立Scalar，禁止把依赖tile结果的计算移到等待前。首版复杂控制流或未知调用自动回退，记录原因。
2. **自动生成early-start入口与等待。** 能力描述允许运行时提前进入该task；编译器在首个不安全操作前
   插入等待本核在途流水和本task完整数据依赖的内部操作。前缀结果须通过明确的编译依赖保留在等待之前，
   并检查后续CCEC编译没有将其全部下沉；仅给wait加内存副作用不足以约束纯Scalar指令下沉。
   等待之后才能设置atomic/mask及进入计算主体。参数解包、wrapper和EmitC隐式prologue也在此合同内。
3. **自动生成fake-end出口。** PTOAS根据实际流水使用情况和现有同步关系，确定能覆盖所有在途工作与
   输出可见性的完成事件；可证明覆盖时才把编译器插入的函数尾部全等待改为“记录完成事件并交还Scalar”。
   task内部的搬运/计算、buffer复用及用户显式同步保持正确；不能仅按AIC/AIV名称写死完成掩码。
   fake-end只表示本block已发完指令，运行时仍持有其事件、参数槽及资源引用。真实完成以后才发布FIN、
   聚合task依赖和回收资源。无合格后继或图结束时由executor完成等待；不能把fake-end直接转成FIN。
4. **编译结果统一驱动调度。** PTOAS输出每个kernel的early-start/fake-end能力、完成方式及ABI信息。
   PyPTO用同一份结果生成隐藏上下文传参、wrapper及orchestration，不再分别读取算子源码里的人工属性。
   Simpler继续在运行时选择实际同核后继；编译器既不绑定A→B，也不编译A+B组合入口。
   未知外部kernel、暂不支持的MIX或同步模式保持普通完成路径，能力缺失不能通过作者加注解绕过。

编译链落点已从私有源码核实：

| 位置 | 拟修改内容 |
| --- | --- |
| PyPTO `python/pypto/backend/pto_backend.py` | PTOAS阶段结束后才生成orchestration，可在这两阶段之间汇总逐kernel能力；wrapper改为消费相同的能力结果，并自动传隐藏上下文 |
| PyPTO `src/codegen/orchestration/orchestration_codegen.cpp` | 现有`BuildSingleKernelSubmit`读取手工属性，改为消费编译器推导的能力；保留实际task依赖与SPMD配置 |
| PTOAS `tools/ptoas/ptoas_pipeline.cpp` | 在布局/内存及隐式操作明确后、自动同步资源分配前分析前缀和出口；最后校验生成的状态初始化与等待边界一致 |
| PTOAS `lib/PTO/Transforms/InsertSync/` | 协议参与尾部同步及事件分配，不触发现有“显式Set/Wait导致跳过自动同步”的路径；保留内部同步 |
| PTOAS EmitC及PyPTO wrapper | 等待前只保留已证明安全的Scalar；生成的mask/atomic初始化后移，未知副作用回退 |
| Simpler TMR | 复用已验证的同核交接、真实完成和资源生命周期，接收自动生成的能力；无需知道相邻task源码 |

本方向实施时的顺序：普通无标记PyPTO小用例自动生成协议 → 单卡AIV/AIC独立入口及动态A→B/A→C验证
→ 必要的依赖/退出回归 → 原CSA源码的128K/B16单卡验证。
先采用现有安全前缀，不把复杂的跨分支Scalar提升作为最小demo前提。
功能以“无需算子标记、自动边界正确、真实同核重叠、数值与状态正确”为准；
性能另记min/mean/max，分析普通路径开销与交接收益，不能预先承诺所有自动接入的task都有净收益。

### 5.8 当前方案：参考SK与PDL，只在原task中插入局部协议语句

2026-10-10用户进一步说明：自动识别是未来方向，即使当前允许算子标记，也不应整体改写InCore task。
本次重新查阅安装的CANN9.2实现、`graph-autofusion`的AOT/JIT代码、AscendC接口文档及NVIDIA官方PDL/PTX文档。
GitCode文档使用禁用代理的HTTP请求读取；源码仍使用既有只读参考副本，没有更新公共安装。

**AscendC SK的插入方式。** 作者在原子算子中调用`WaitPreTaskEnd()`和`SetNextTaskStart()`。
前者通常位于第一条搬运指令之前，允许其前面的独立准备工作提前执行；后者通常位于最后一条搬运指令之后。
它们均为无返回值的局部语句，不要求将计算主体搬到新helper或包在一个`if ready`里。
安装包`kernel_info_infer.py:76`识别这些调用，`super_kernel_sub_op_compile.py:143`启用对应编译宏，
`compile_op.py:1602`将Wait/Set能力分别编码到`SK_BIND`的bit0/bit1；AOT同步pass再依据能力把等待放到内部边界。
编译框架还生成SK子函数入口及同步配置，作者无需为此重写整个task。

A3实现中Set按配置在MTE3/FIX等流水发`ffts_cross_core_sync`，Wait使用`wait_flag_dev`。
`TPipe::Destroy()`在`__ASCENDC_ENABLE_SUPER_KERNEL__`下省去默认`PIPE_ALL`，
其他清理与内部同步仍存在。因此同核Scalar可在子函数返回后进入后继，前驱流水由事件合同收尾。
Set调用本身既不是C++ `return`，也不意味着整task已经真实完成。
来源：[SetNextTaskStart](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_task_sync/SetNextTaskStart.html)、
[WaitPreTaskEnd](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/inter_task_sync/WaitPreTaskEnd.html)。

**CUDA PDL的插入方式。** 前驱在允许后继开始的位置调用`cudaTriggerProgrammaticLaunchCompletion()`，
调用后仍可继续运行自己的代码；后继先做独立工作，再于依赖读取前调用`cudaGridDependencySynchronize()`。
Host侧为后继设置programmatic stream serialization的launch属性，或建立相应graph依赖。
底层协议为`griddepcontrol.launch_dependents`和`griddepcontrol.wait`：前者允许后继调度，后者等待所依赖grid
真实结束及其内存操作可见。GPU不以“前驱函数先返回”为前提，也不保证同SM放置或一定发生并发。
来源：[CUDA PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)、
[PTX griddepcontrol](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)。

**术语修正。** 本次核对的公开接口不叫`fake_end()`；不能把GPU的LaunchCompletion等同于函数异步返回，
也不能把SK的Set或TMR的提前许可当成真实FIN。“许可提前进入”“Scalar交还执行权”“工作真实完成”要分别说明。
PyPTO目前同核Scalar交接需要前驱交还Scalar，具体出口生成由PTOAS/编译ABI承担，不能让作者因此重构计算。

当前源码接口如下。2026-10-10已在私有PyPTO/PTOAS实现并完成单卡功能验收；性能单独记录：

```python
with pl.spmd(...):  # 原有task与依赖
    # 原有独立Scalar计算
    ...
    pl.system.wait_pre_task_end()   # 新增局部等待点
    # 原有load / matmul / pipeline / store
    ...
    pl.system.set_next_task_start() # 新增局部接续许可点
```

近期实现应从这些语句派生内部能力，无需作者另加`scalar_prefix`或区分`async_aic_return`/`async_aiv_return`。
在outlining后保留语句位置，推导核类型、验证协议路径，并由后端生成隐藏上下文、等待后的模式初始化、
异步尾部及普通回退。现有`wait_predecessor`返回0用于EXIT处理；应在生成的控制流中处理退出，
不把`ready`值或包裹整个计算主体的分支暴露给作者。原有task身份、block数、数学表达、tiling和流水不变。
初版接续许可放在最后计算/写回提交之后；GPU在函数中段触发后继续运算的能力不能直接外推为
单Scalar核上任意位置的抢占/恢复。新的协议IR仍须保留内部自动同步。

当前实现状态（2026-10-10）：

- PyPTO注册两个无返回值的局部操作，outlining后从函数体推导等待/接续能力；
  隐藏dispatch参数、核类型、wrapper状态初始化、orchestration能力均由编译器生成。
- PTOAS验证局部边界、保留跨Wait存活的Scalar结果，并生成退出分支；Cancel路径不发未提交的尾部事件。
  Set后禁止继续工作，仍保留内部自动同步；MIX/双AIV/HBG不在本demo准入范围。
- 原NZ O-A接入与冻结普通版相比仅`decode_o_proj.py`新增两行，未拆helper，未改计算、tiling或pipeline。
- 一个解析round-trip、四个PTOAS正反例、AIV动态分支及AIV/AIC同核诊断三项CPU二进制编译通过；
  完整128K/B16 CSA普通/局部语句两份也完成预编译。
- 单卡任务`task_20261010_181636_150374113561`在0卡exit=0，三个基础用例通过。
  AIV两轮各48/48、AIC两轮8/24与24/24观察到前驱计算与后继Scalar在同核重叠；
  动态B/C选择、无后继、更换输入及保护区均通过。
- 完整128K/B16 CSA输出、Top-K、score、全部缓存状态和改输入图重放逐bit一致，
  两侧各31项保护区/metadata检查通过。双方均为73个kernel，原数学与流水未改。
- 30次单次CSA重放min/mean/max（μs）：普通版946.68/967.64/990.48，
  局部语句版950.50/970.21/986.74；候选均值慢0.27%，14/30对更快，尚无稳定性能收益。
  实现与验收见实施日志过程035/036，完整精简结果在
  [局部语句复现记录](superkernel_local_boundaries_20261010/README.md)。

第5.8节限定的局部语句demo已完成功能验收；未把性能微小波动当成稳定收益。
自动寻找插入点后续复用相同IR/ABI，不要求现在先完成通用Scalar提取与提升pass。

### 5.9 扩展到其余可接入的 CSA task（2026-10-10）

按用户要求，将局部语句扩展到完整CSA的独立AIC/AIV任务。私有vLLM提交`2cd33a9a`
在原O-A版本之上十文件新增104行；原源码的算术、tiling、流水表达和sync_start设置不变。
这不等于生成代码的内部同步不变：后续过程039发现异步尾部预留event7使一个AIV计算循环退化为PIPE_ALL。
PTOAS `f556407f`补齐本次普通Vector指令的校验范围，GM Scalar读只允许在Wait之后。

本次128K/B16共编译73个kernel：59个Set/Wait、2个仅Wait、12个MIX成员保持原协议。
59个是编译产物覆盖数，不能理解成每次执行都触发59次重叠；运行时分支和sync_start仍决定实际资格。
仅Wait的两项为row_offsets与valid_qk_plan，保留正常Scalar写回完成；
MIX的Indexer Score及QK/PV尚未接入本原型的局部交接协议；这是实现范围限制，不是MIX原理上不能使用Set/Wait。
正常MIX完成统计已经存在，不能写成“运行时没有MIX组完成协议”。当前缺口为：
PyPTO的MixedKernels提交没有传递局部交接能力；PTOAS异步尾部校验尚不接受跨核同步；
Simpler的新交接入口只放行独立AIC/AIV。后续应保持原MIX任务和核间流水，按成员核生成安全Scalar边界及尾部完成事件，
再接入既有组完成统计。某个成员核交出Scalar不代表整个MIX完成；也不要求等待全组完成后才允许安全Scalar前缀。
具体源码依据和接入边界见实施日志过程038。

完整CPU编译及五项PTOAS正反例通过，普通/仅O-A/扩展三版已在同一0卡完成比较。
每组三十次单次图重放，min/mean/max（μs）分别为954.34/974.90/1004.82、
955.92/970.49/1011.26、975.56/989.89/1007.46。扩展版均值比普通版慢1.54%，
26/30配对轮次较慢；功能通过（9项输出/状态逐bit一致，三组各31项保护区检查通过），但没有净性能收益。
本轮保留私有实验改动，不将全量标记视作默认性能优化；未追加无关设备测试。
详见[全量接入实验记录](superkernel_all_boundaries_20261010/README.md)
和实施日志过程037。

已有产物进一步解释了为什么尚无净收益（过程039，无新占卡）：`mix_x_rms_norm`的内部MTE2→V同步
因event7预留发生PIPE_ALL降级，每block对应8次；Scalar-prefix预派发强制寻找同类型前驱所在pending核，
使O-A(AIC)→quant(AIV)→O-B(AIC)等跨核边失去原本可用的普通提前派发资格。
此外无交接时也有事件链、上下文写回等开销，两个带sync_start的单核后继仍被排除，MIX关键计算尚未接入。
以上代码证据不能分摊14.99 μs的回退；应先消除内部流水和预派发退化，再用轻量机制trace量化真实命中与收益。

2026-10-11过程040已修复上述事件挤占、普通预派发回退及重复填参。59个异步函数内部同步均恢复与普通版一致，
单卡功能通过；同一新SDK下普通/仅O-A/扩展均值为966.99/976.13/974.89 μs，扩展仍慢0.82%，尚无净收益。
无pending后继时的协议成本仍需缩减。仅O-A没有稳定收益，不维护独立长期版本；保留本轮私有诊断快照及Git记录。
详见[新SDK三组对照](superkernel_all_pipes_20261011/README.md)，不得把跨SDK绝对耗时变化作为单项修复收益。
随后过程041尝试无pending后继时直接普通返回：功能通过，但同卡均值973.75→985.54 μs，P95也升高，
候选已撤回。该策略不属于当前保留实现；后续优先审视GM完成标记和控制行写回成本，净收益目标仍未完成。

过程042验证调度器预置完成标记以省去未交接控制行的GM写入：功能通过，但同SDK下原扩展版
均值983.40 μs，新候选986.54 μs，最大值1007.92→1103.26 μs，P95基本不变；已撤回。
源码少写一次GM不能单独证明CSA净收益，下一步须结合CSA实际交接及关键路径诊断。
仅O-A不再独立维护或测试；历史快照仅作为诊断参考。
[本轮三组完整结果](superkernel_prearmed_tail_20261011/README.md)。
[失败候选证据](superkernel_pending_tail_20261011/README.md)。

## 6. 按动态交接目标分阶段执行

本节保留动态机制的分阶段计划和验收依据；人工标记原型已完成的阶段不能替代第5.7节的自动编译验收。
近期推进顺序以第5.8节为准，先完成只插局部语句的基础demo；第5.7节自动识别及复杂工程化后移。
此前正常 A→B 基线已有单卡证据，固定合组和新诊断草稿的状态见第 7.3 节。
各阶段先汇报证据和限制再推进；近期任务详细验证，后续复杂能力按候选实际依赖展开。

通用约束：先 CPU 编译/检查，再统一 task-submit 做单卡；不做过度 hash 校验，不预先铺开七档或整模型。
排队任务须固定 kernel 源码、Python 包、加载的 C++ 扩展及工具链版本。只冻结一个测试文件不足以防止共享依赖变化。
仍直接引用工作树的排队任务存续期间，不修改其依赖。诊断导入不一致已修复；为避免影响另一优化会话，后续使用 `.worktrees/scalar-overlap-demo-20261009/` 中的私有源码、venv 和构建缓存。

### 步骤零：恢复基线，定义协议并补齐工具链合同

1. 使 PyPTO Python 源码与实际加载的 C++ 扩展一致；记录加载路径和版本，诊断操作未通过不能当作已有能力。
2. 对照 PDL 的许可/等待语义与当前 PTOAS/PTO-ISA/CANN 能力，定义提前调度、Scalar 交接、真实完成及内存可见性的独立责任。
3. 检查 ABI、函数尾同步、隐含状态恢复、片上存储/事件及 Scalar GM 写回的完成要求；列出可支持的核类型和 task 形态。
4. 形成 PyPTO/PTOAS/Simpler 的最小合同：普通模式、交接模式、能力声明、实例上下文、许可、等待、完成事件及失败/收尾处理。
5. 按第 5.5 节补齐必要的 PTOAS 能力，先定向编译/IR 检查普通路径保留、所有出口覆盖、内部同步及非法模式拒绝，再走单卡。

通过条件：工具链合同有可编译的具体表达，并清楚说明保留了哪些内部同步；必要编译器缺口已解决，未解决的硬件限制如实记录。
完整二进制编译是必要条件，硬件异步返回仍须在步骤二证明。手写 AscendC 探针只作参考，不能替代 PyPTO 路径。

### 步骤一：独立 task 与运行时选择的正常功能基线

复用小整数 FP32 的简单运算，先保留正常依赖和完整等待。A/B/C 各自独立编译、注册，不生成组合 kernel。

1. 先检查已有 A→B 基线，只补环境变更实际影响的部分，不重复无关测试。
2. 给 B/C 不同的可精确识别运算，通过同一批编译产物的运行时选择分别执行 A→B、A→C，并改变 task 实例/参数。
3. 检查中间/最终输出、更换输入及 NaN 输出；记录实际入口、task/block ID、代次及选择结果。
4. 用编译调用记录和复用的入口证明没有因相邻关系变化重编译 A+B/A+C；不靠重复全量 hash。

通过条件：PyPTO 完整路径及单卡执行正确，运行时分支确实改变接续对象，task 身份仍分离。
固定 shape 允许，固定相邻配对不算动态选择。此阶段不要求重叠或性能改善。

### 步骤二：独立 A 的异步返回与真实完成

先只用 A，分别验证普通完成模式和交接模式，不急于引入 B 的提前 Scalar。

1. 使用步骤零确认的生成方式；保留 A 内部流水同步，按合同提交完成事件后返回。
2. 同一物理核记录指令提交/函数返回、运行时等待和真实完成，确认不是尾部已经完全等待后才返回。
3. executor 在无后继时消费完成事件、发布一次正确完成，核对输出、保护区及资源生命周期。
4. 验证末尾 task、重复调用与最小 graph replay 的收尾；不得因没有 B 而遗漏 FIN 或释放仍在使用的资源。

通过条件：独立 PyPTO 入口实际具备异步交接合同，且普通模式和无后继收尾正确。
生成 C++ 中没有某个 barrier 不等于异步返回已被证明；仅有 Native 或静态合组证据不能通过本步骤。

### 步骤三：运行时同核接续，再移动后继等待

先 TMR、AIV 单 block，按第 5.2 节准入条件逐项实现：

1. runtime 动态选择 B/C，把后继交给 A 所在物理核；独立入口、task 身份和完成记录保持分离。
2. 先把等待放在后继入口前部，证明前驱完成、后继 gate 和资源所有权正确，尤其排除“B 等 A 的 FIN、FIN 等 B 返回”的死锁。
3. 再允许后继一次性执行安全 Scalar，把边界处理移到首次不安全操作前；普通路径继续可用。
4. Scalar 用运行时参数做整数运算并校验结果；证明它在设备执行、没有被删掉、移到 Host 或算两遍。
5. 分别覆盖 A→B、A→C、无合格后继、能力不兼容回退；保持正确完成和图退出。
6. 记录本核 predecessor/block 完成与逻辑 task 完成的区别，不能借全核假同步或提前 FIN 掩盖协议问题。
7. 强制关闭提前执行，检查普通串行仍正确；注入“许可已到、数据尚未完成”的场景，确认 B 可执行前缀但不会提前读取依赖数据。

通过条件：同一批独立编译的入口，在不同实际接续关系下都正确；真实物理核、Scalar 位置、边界等待及完成链路可核对。
以设备内观测或可信指令/仿真证据证明重叠窗口，区分 Vector/Cube 与末尾搬运；证据缺失则保持“重叠待证”。
诊断操作的原始 tick 不假定等于核周期或微秒，不用未验证的插桩接口制造硬件证据。

### 步骤四：按 CSA 所需能力扩展动态协议

仍使用独立 PyPTO 小用例，一次扩展一种能力，未经验证的组合保持普通路径。

| 顺序 | 要补的能力 | 必须证明 |
| --- | --- | --- |
| 1 | 多 block、尾块/padding、同类核多轮派发 | 分开许可聚合和完成聚合；一个 block 触发或返回不能代替全部参与者，结果及保护区正确 |
| 2 | 多前驱、fan-out、不同核数及 `sync_start` | 不占满核等待尚未执行的生产者；其他消费者能在 A 真正完成后按原依赖推进；同步起点语义不被偷偷改变 |
| 3 | pending/参数槽重用、事件重用、长链与 graph replay | token 代次及所有权正确，通知不丢失/重复，无旧准备结果、ABA 或悬挂资源；长链不饿死其他可运行任务 |
| 4 | 异常退出、图结束及不支持能力的回退 | 在途工作和引用有明确收尾，普通路径正确，不发生假成功 |
| 5 | AIC→AIC 小 matmul、必要的混合类型依赖 | 重新证明 Cube/FIX 等完成覆盖及状态合同；AIV 结论不外推到 AIC，跨类型通信不算同物理核重叠 |
| 暂缓 | HBG 的独立入口、gate 与完成路径 | 2026-10-10 用户要求先只做 TMR；HBG 不作为本阶段前置条件，将来恢复时独立验收 |

2026-10-10 用户明确“先做 TMR 就可以”“HBG 先不管”，当前实现和验收限定 TMR，HBG 暂缓。
TMR 小用例已通过的范围直接进入公平性能对照；未验证后端不启用新模式，不外推 TMR 的通过结论。
具体 CSA 候选依赖的新能力，必须先回本步骤验证，不能在模型内临时绕过；动态协议本身不再是可选项。

### 步骤五：独立 task 的公平重叠与性能对照

保持 task 数量、依赖图、算术、数据、工作量、核数、工具链、诊断开销和样本数一致，比较：

| 组别 | 执行方式 | 用途 |
| --- | --- | --- |
| A | 原有独立 task 动态调度、普通完整完成 | 当前功能和性能基线 |
| B | 独立 task 的新交接合同，后继等待前置 | 测新增交接、记录和完成处理开销 |
| C | 与 B 相同的动态交接，等待放在后继安全 Scalar 后 | C/B 测等待后移的净作用，C/A 测端到端净收益 |

B/C 都不得合并 task 或固定编译相邻组合；三组 Scalar 工作量及校验完全相同。
旧固定合组只作额外机制参考，不参加主验收。A→B 与 A→C 均须有实际选择证据，不只跑一条静态链。

使用设备内区间或连续重放降低 Host 空隙影响，保留全部样本的 min/mean/max；正式计时不带 profiler，必要诊断只采三个 step。
报告安全 Scalar、前驱剩余流水、交接/等待、完成处理和上下文保存开销，区分 A 的硬件完成与函数返回。
已有某些 PMU 模式会抑制 pending，不能以改变了调度的采样模式证明普通路径重叠。
功能通过和性能有效分开记录；人为加长工作只能证明机制，不能推算 CSA 收益。
开销抵消窗口时单独记录并定位，不因此撤回已通过的功能；后续功能扩展仍按必要依赖逐步推进。

2026-10-10首轮三组同卡图重放已完成，详见[过程019](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#21-过程-019聚焦-tmr关闭诊断后做公平计时)。
17-task链、每task 1024轮Scalar时，C相对A均值下降1.084%、相对B下降1.006%，两项均30/30采样轮更快，min/max也下降。
64轮Scalar的C/B均值差仅0.039%，分布重叠且max略高，不宣称稳定受益。计时不带trace/profiler，所有阶段与保护区精确。
普通组使用同一实验SDK；实际候选还需评估未修改SDK的公共路径开销，不能以这个小图替代CSA验收。

### 步骤六：接入 CSA 的实际动态 task

进入条件：候选依赖的核类型、运行时、前缀安全性、完成及同步能力均已有独立小用例证据。
只为符合条件的 task 声明能力，由 Simpler 根据运行时状态决定是否接续；不在 CSA 中写死 A+B 组合来替代目标。

可检查 `hc_pre.py` 的 `hc_pre_linear_reduce` 及后续 AIV 阶段，但先核实实际核类型、row 分工、`inv_rms` 依赖、fan-out 和核数。
`split_pre_post`/`comb_sinkhorn` 的 `hc_scale` 读取不能仅凭“是 Scalar”提前；该链不是已经确定的优化对象。
Indexer 长档 Score 的 `allow_early_resolve`/`sync_start` 涉及历史长尾，不作为未经协议证明就直接翻转的首项。
Q/O 中 AIC→AIV 交替不能替代同物理核证明。

先小输入检查算术、索引、状态及写区，再冻结 `PTO_CSA_VARIANT=pkg:<name>` 整包及其运行依赖，确认根依赖图可解析。
记录真正发生交接的 task/block、准入命中/回退原因、Scalar 窗口、其他消费者的等待和最大耗时变化。
不能仅有新开关或声明而没有实际交接，就把 CSA 接入标完成。

过程025已否定较慢的手工tile重写。过程026改用原tensor算术和已有内部cache属性，
原scope/独立普通/异步入口均值111.9365/111.9240/110.3405 μs，
两轮各40条接续、其中37/40和40/40有实际Cube重叠，输出/保护区通过。
这是同实验SDK、连续16次折算的隔离O-A结果；后续私有完整128K/B16 CSA首轮结果见步骤七。
使用内部cache参数索引只服务该demo，不把该写法当作新公共DSL合同。

### 步骤七：完整 CSA 单卡验收

2026-10-10先按单卡128K/B16推进；2026-10-11用户要求扩展至七档，最新结果见过程051。
先同卡比较原PTO与新接续路径，记录min/mean/max、版本和相同部署口径，关注最大值及长尾。
区分实验SDK内的标记开关对照与未修改SDK基线；本项没有新增Native测量。
策略的性能结论只取各档相同口径的实测，不能由脚本拼最优结果；功能错误或异常长尾不能靠均值抵消。
功能状态与性能状态分别汇报，尚无稳定加速不等于功能未完成；保留已证明的动态交接机制继续降低其开销。

过程027已完成首轮同实验SDK完整图对照：原路径944.56 / 969.94 / 997.56 μs，
接续路径937.32 / 965.70 / 985.80 μs（min/mean/max）。输出、Top-K、全部Native缓存/状态、
保护区及改输入后的图重放逐bit一致。均值少0.44%，17/30配对样本更快，尚不认定稳定收益。
过程028独立诊断已证明完整图两轮36/40、39/40边存在同核Cube重叠，输出及等待合同通过；
过程029原SDK括号对照已完成：原SDK前/后均值977.55/959.08 μs，
实验SDK普通/接续均值979.67/980.68 μs。输出跨SDK逐bit一致；同SDK接续均值高0.10%，
首轮收益未复现。原SDK自身漂移1.89%，不能把括号均值差直接归因新增公共开销。
先检查普通任务新增路径与计时Host间隔，不通过重复采样筛出有利结果。
隔离O-A在过程026中仅节省约1.60 μs，相对约970 μs的完整CSA仅约0.16%；
这是量级参考，未假定隔离收益能线性搬到复杂图，也不是机制正确性的判据。

首先与当前 PTO 检查结果/状态：整数、索引和保护区精确一致，浮点在同算术及确定性配置下检查 bit。
已有 Native/PTO 低接受率差异仍开放，简单一致请求不能代替全部精度验收。
按最新要求仅做单卡CSA验证，不做16卡token/DSpark测试；这一项不作为本goal的完成门槛。
本轮文档修订完成不等于尚待测候选的功能或性能验收完成。

## 7. 实操记录与完成边界

完成了 TorchAir→AOT 优化器→能力位→任务队列→设备调用→AscendC Set/Wait 的源码链路复核。
已区分上游源码、本地安装头文件和历史运行证据，没有将它们合称为同一个已验证版本。

### 7.1 此前的 PTOAS CPU 探针

在本次深入复核前进行过 CPU PTOAS 代码生成探针：
以历史 `hc_pre_linear_reduce.pto` 的副本插入外部 `func.call`，当前 PTOAS 0.66 的原版和插桩版都编译退出 0，
生成代码中调用保留在 `TADD` 之后。它只说明现有 lowering 可以保留外部调用位置，**没有证明 Native 式子函数异步交接、CCEC 汇编顺序、设备重叠或性能收益**。
探针没有改变生产源码；其中循环内多次调用也不是建议的生产实现。

[探针记录](results/csa_scalar_overlap_feasibility_20261009/probe_results.json)、
[探针生成代码](results/csa_scalar_overlap_feasibility_20261009/scalar_hook.cpp)。`results/` 是本地实验产物，复核要点已在本文件保留。

### 7.2 当前 Native 最小复现

已用当前 CANN 的 Bisheng 成功编译普通 kernel、SK 子入口、`SK_BIND` 元数据与 ACL Host 程序。
最小用例含四组对照，输入/输出共不到 1 MiB，不需要模型权重和 PyPTO。
单卡任务 `task_20261009_165431_362034525498` 在物理卡 2 完成，exit=0；设备已释放。
每组 5 次预热、20 次计时，再做一次计时外的“换输入、NaN 中间输出”校验。四组 16384 个元素均精确一致，Scalar 校验值也一致。
原始产物目录：`tests/pypto_test/results/native_superkernel_repro_20261009/`。
随源码保留 [完整样本与设备 trace](superkernel_repro_20261009/results.json)，避免仅链接 Git 忽略的实验目录。

| 对照 | 捕获任务数：优化前→后 | A/B 原始物理核号 | A/B 同步配置 | B Scalar / Wait（tick） |
| --- | --- | --- | --- | --- |
| 普通图 | 2→2 | 54 / 55 | 0 / 0 | 26 / 0 |
| SK，Early-Start 关 | 8→1 | 34 / 34 | 0 / 0 | 22 / 0 |
| SK，Early-Start 开，Wait 前置 | 8→1 | 62 / 62 | 0x10 / 0x20 | 22 / 85 |
| SK，Early-Start 开，Wait 后移 | 8→1 | 40 / 40 | 0x10 / 0x20 | 22 / 64 |

SK scope 捕获的 8 项包含控制任务，不代表 8 个计算算子；计算主体始终只有 A/B 两个。
物理核号是 `get_coreid()` 原始值，不是逻辑 block 序号；要求每次 A/B 同核，没有要求不同进程固定在同一核。
表中的 tick 来自各组最后一次计时重放，不是 20 次平均；未擅自假定计数器频率。

同一 Early-Start 合同下，B 入口到可读输入的区间为：

```text
Wait 前置：Wait 85 + Scalar 22 = 107 tick
Wait 后移：Scalar 22 + Wait 64 =  86 tick
```

后移后 Scalar 保持 22 tick，Wait 少 21 tick，符合其与前驱尚未完成的异步流水重叠。
A 的最后 Set 提交到 B 入口仍有 58/56 tick 的交接区间，说明连续调用本身也有队列/参数处理代价。
目前观测的是依赖 A 的最终 MTE3 完成信号；不能据此分别报出 Vector、MTE3 或 Cube 的重叠微秒数。

保留 ACL event 包围单次图重放的原始计时，单位 μs：

| 对照 | min | mean | max |
| --- | ---: | ---: | ---: |
| 普通图 | 44.280 | 62.379 | 70.940 |
| SK，Early-Start 关 | 38.760 | 63.640 | 83.540 |
| SK，Early-Start 开，Wait 前置 | 33.220 | 61.367 | 70.020 |
| SK，Early-Start 开，Wait 后移 | 39.960 | 58.965 | 72.240 |

这个区间包含 Host 提交及图调度空隙，样本波动明显；不以 61.367→58.965 μs 宣称稳定重叠收益，
更不能将它作为 CSA 或整网收益。后续性能验收应增加设备内完整区间统计，或采用连续批量重放以减小 Host 空隙影响。

启动过程另修复了一个脚本问题：task-submit 自动追加 `--device N`，不能把它当作输出目录。
早期失败发生在二进制启动前，不是 SK 设备错误；当前脚本已兼容追加参数，并在失败时报告文件及行号。

上述 Native 复现当时没有重测七档或整模型，也没有修改 PyPTO/Simpler/PTOAS/PTO-ISA 生产代码；后续诊断草稿另列如下。
目前不承诺具体微秒收益；收益上限取决于关键链上的可重叠窗口，不能将泳道图全部空白或所有核的间隙直接相加。

### 7.3 PyPTO 已有产物的重新定性

| 产物 | 实际证据 | 在修正目标中的作用 |
| --- | --- | --- |
| 独立 `two_task_chain` | 任务 `task_20261009_173158_215808217987`，TMR/物理卡 8，1 passed；两次换输入/NaN 校验精确一致 | 正常完成模式的历史功能基线；未证明运行时更换后继或异步交接 |
| 固定 `local_task_chain` | `local_compile_v2.xml` 为 3 passed，包含两个等待形式的设备二进制编译 | 固定合组机制对照，不能作为动态路线通过证据 |
| 固定合组单卡任务 | `task_20261009_173748_25502084348` 最终 exit=4；conftest 导入时报 `system.read_clock` 未注册，未进入设备计算 | 设备功能未通过，不能继续记作 pending 或证明重叠 |
| `read_clock` / `physical_core_id` 诊断 | 私有环境任务 `task_20261009_231035_382005015310`，单卡 1 passed；同一组独立 A/B/C 二进制运行时选择 B/C/B，输出与标记精确一致 | 普通调度核号为 51/49、49/51、49/49，未保证同核；这里只证明动态选择及诊断可用 |
| 独立 AIV 异步返回探针 | 过程 009，任务 `task_20261009_232737_2932955117`，单卡 1 passed；真实等待交给 executor 后，返回仍有 181/207 tick 工作待完成，换输入结果精确一致 | 手写 external-source 入口证明返回边界不强制排空；尚未接入后继 Scalar，不是完整重叠验收 |
| 独立 A/B/C 动态同核交接 | 过程010，两轮单卡通过，运行时选择B/C，输出精确一致；后继Scalar结束后前驱完整流水仍有15～88 tick待完成 | 手写入口证明同核动态交接及异步流水重叠；当时未区分Vector与末尾MTE3 |
| Vector专门诊断 | 过程011，128×128候选新断言未通过；128×256候选单卡通过，8/8同核交接且输出精确一致；后移等待时Scalar运行20～21 tick后Vector仍需73～78 tick完成 | 手写独立入口已证明真实Vector/Scalar重叠；DSL/PTOAS自动生成路径继续验证，不等于整体性能收益 |

固定合组测试文件虽然未改，其依赖的共享 PyPTO 包发生了变化，说明排队冻结须覆盖实际加载依赖。
纯DSL/PTOAS路径在2026-10-10通过：任务`task_20261010_090008_334195597`
四次运行时选择独立B/C入口，4/4同核、逐元素精确一致；Scalar执行19 tick后，
前驱Vector仍需313～331 tick完成。任务`task_20261010_090221_1213099353`
进一步验证无后继收尾、普通前驱回退，以及回退后恢复动态交接。
私有PTOAS已构建，编译过程中修复了`InitMemRef`未同步更新函数属性参数引用的问题。
原始trace、编译器合同负例及限制见[过程012](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#14-过程-012纯-pypto-dsl-的动态同核重叠通过)。

后续私有环境已恢复定向设备测试；结果按动态选择、完成语义、计算/搬运重叠分别记录。

多block扩展任务`task_20261010_091746_84074625444`单卡 **2 passed**：
4 blocks每次4组同核重叠，96 blocks在48个物理AIV上多轮派发、每次48组同核重叠。
后继读取其他block的输出，所有等待返回都在前驱全体block真实完成之后；
有效数据、115行尾块、padding和保护区精确通过。
许可仍以所有前驱block已发布为前提，计算读入仍以整项task完成为前提。
带`sync_start`的后继暂不启用前缀交接，避免未经验证地改变其启动语义。
实现和完整min/mean/max见[过程013](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#15-过程-013多-block尾块与整项-task-完成依赖)。

多前驱/fan-out扩展任务`task_20261010_092940_154350625597`单卡 **2 passed**，
包含`A、B → C、D`新图及一个既有单前驱回归。四个task均各4 blocks，
两次换输入并交换A/B工作量，每次8组同核重叠，C/D两条fan-out均有覆盖。
其中每次4组在Scalar结束后另一路前驱仍未完成；C/D全部等待边界仍晚于A/B全部block完成，
跨block数据、Scalar结果与保护区均正确。
接续资格检查真实WAIT边与当前producer的fan-out，RETAIN-only或slot地址相等本身不构成资格。
这些是基础复杂图的功能/计算重叠证据，未覆盖长链slot复用、AIC/HBG或公平整体性能。
细节见[过程014](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#16-过程-014多前驱与-fan-out-的动态接续)。
任务`task_20261010_093502_181053721032`另补 **1 passed**：4-block前驱接2-block后继，
后继sync_start开启时保持普通派发，全部Scalar在全部前驱完成后启动，数据与保护区正确。
这是不改变sync_start语义的回退证据，重叠支持仍未开启。

连续链已在任务`task_20261010_095812_342714219743`单卡通过：window=8、
运行时3/33/7个task，全部40条相邻边都证明Scalar/Vector重叠，逐元素及末尾收尾正确。
中间task同时消费前驱与发布自身异步尾部，PTOAS将合法Scalar写入的cache刷写放在首次tile工作之前，
避免隐式尾部PIPE_ALL破坏异步合同；manual_scope仍须显式提供依赖。
该轮还修正了复合队列命令缺少设备参数、raw/masked核号不一致的问题：
旧过程013的“卡2”标注撤回，正确选卡的96-block重测在卡5通过。
详见[过程015](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#17-过程-015连续交接窗口复用与设备选择更正)。
窗口复用的这份证据不覆盖NPU graph replay、长链上的其他分支公平性或task_id全范围回绕。

最小kernel模式graph replay也已补齐：任务`task_20261010_101904_49575923354`在卡7通过。
同一独立编译node和注册句柄捕获3/33-task两张图，再按3→33→3重放，36/36相邻边都存在同核
Scalar/Vector重叠；全部阶段数据、Scalar参数、保护区及紧接PTO的PyTorch copy/add均精确。
长链跨8-slot窗口，重放前换输入和毒化缓冲，图销毁及Worker收尾正常。
私有Simpler已固定本地提交`7078fc45f`并重建匹配的kernel接口，没有绕过SDK版本检查。
该证据使用既有Program IR生成KernelArtifact，不等于新增JIT前端语法、一般公平性、AIC或HBG已完成。
细节见[过程016](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#18-过程-016kernel-模式图捕获与重放通过)。


长链上的独立支线在任务`task_20261010_102351_80709110225`（卡6）通过：33-task链、window=8，
stage=8提交无依赖支线；支线完成时主链尚余28个task。主链31/32条边同核重叠，
另一条普通回退；主支线全部输出、Scalar状态及保护区精确。这只证明该基本图的推进，
一般饱和负载公平性仍未验证，见[过程017](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#19-过程-017长链中的独立分支可以继续推进)。


单block AIC机制在任务`task_20261010_104658_21401368904`（卡6）通过：两个独立Cube入口，
前驱17次FP16 matmul累加后异步返回，后继先执行64轮整数Scalar，再等待M/FIX完成并读前驱数据。
两次均在raw核19接续，Scalar为20 tick，其结束后Cube仍需370/417 tick，随后FIX还需177/39 tick。
全部输出、依赖数据、Scalar状态及保护区精确，无后继收尾通过；同任务的AIV长链回归40/40边仍重叠。
这扩展了基础计算重叠证据，但不覆盖AIC复杂图/图重放、HBG或端到端收益；
具体改动、源版本和边界见[过程018](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md#20-过程-018独立-aic-task-的-scalarcube-重叠通过)。

完整过程、命令和产物链接见 [实现过程记录](DSV4_FLASH_CSA_SUPERKERNEL_IMPLEMENTATION_LOG.md)。
