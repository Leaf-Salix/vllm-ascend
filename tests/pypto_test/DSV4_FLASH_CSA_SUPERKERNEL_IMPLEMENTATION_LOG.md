# PyPTO 同核 Scalar 重叠：实现过程记录

起始日期：2026-10-09。按用户要求独立记录本项工作的实现过程，后续持续追加。

本文件记录实际修改、尝试、命令、结果、失败原因和后续判断。
[分析与执行方案](DSV4_FLASH_CSA_SUPERKERNEL_ANALYSIS.md) 保留设计依据和阶段验收条件；
[总验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md) 保留阶段摘要及本文件入口。
失败记录只在有明确原因后定性，编译通过、设备功能通过、重叠被证明、性能有效分别记账。

## 1. 目标、范围与当前进度

2026-10-11最新决定：用户接受必要的固定开销，暂停“保证无额外开销”目标。
已恢复并保留PyPTO `320beb494`、Simpler `66d6ce63e`、PTOAS `4a5ce22a` 的指针交接位方案，
后续以此为基准审查其他协议开销，详见过程046；过程047在相同入口方案上完成两处小清理，
当时私有PyPTO `1f8206d09`、Simpler `d00f8f421`，PTOAS不变。过程048按最新要求完成七档单卡对照，
均值四档改善、三档回退，长短档8:2加权仅减少0.1569%；功能通过，整体稳定净收益仍未成立。
过程050移除本核前驱必须是直接WAIT生产者的限制，当前私有PyPTO `fdac65f66`、Simpler `5c8418a02`。
过程051完成新SDK同卡交错七档复测：长短档8:2加权减少0.8636%，均值仍四档改善、三档回退，
全部功能检查通过。过程052以Simpler `8171b78c5`、PyPTO `19f2c091a`接入idle与sync_start安全前缀，
四种组合均有48/48时序证据。新SDK的128K/B16均值减少0.86%，但max增加4.24μs。
过程053已完成MIX成员接入及单卡功能验证；128K/B16新增MIX均值回退10.64μs，
过程054已把该参数准备恢复到AICore：同SDK普通/原标记/MIX均值982.46/982.22/979.16μs，
MIX比普通减少0.34%，数值、状态与保护区通过；小幅收益尚未证明稳定。
过程055补充真实CSA插桩：两轮各973个物理成员执行，551/586个gated实例全部进入前缀回调路径，
实际同核交接210/237次。72/108条pending未交接仍缺派发时间戳，不能全部算作可挽回窗口。
单次探测时机继续处理；诊断结果不当作性能测量。
过程056已在AIC/AIV小例子验证有序DMA完成通知及轮询，输出、重放和保护区通过。
无后继时轮询版小图mean多0.69～1.28μs且波动较大；未接入CSA，下一步先证实后到窗口捕获。
过程057完成两任务验证：交替通知候选两轮分别有6/4条后到交接仍有原Cube计算等待，
最新图重放又有1条，输出和保护区正确。诊断小图AIC mean少1.57μs、AIV多6.49μs，
不能外推CSA；通用scratch、事件和通知区仍需纳入编译/运行时合同，公共路径不变。
以下历史目标和试验结论不覆盖这些最新决定与测量。

### 当前 goal 的执行约束：先完成最小 demo

2026-10-09 用户补充：“这个是做 demo 验证，你不用遵循这么严格的工作流”，并要求纳入 goal。
当前 goal 以本文件为执行依据，因此以下约束直接适用于后续实施：

- 2026-10-11 用户再次确认：本轮由局部 `wait_pre_task_end()` 标识安全 Scalar 前缀，不做自动识别；开启 chip swimlane、PMU 或参数 dump 时保留串行回退，这不是本轮待消除的限制。
- 优先用最小代码改动逐步验证独立编译 task 在 PyPTO 动态调度下的同核 Scalar 与 Vector/Cube 重叠，先小用例，后 CSA。
- demo 阶段简化开发流程，不以完整工程化、完整测试矩阵、双语 API 文档或发布流程作为每步前置条件；不做过度 hash 校验或重复测试。
- 保留能支撑结论的必要验证：实际独立入口及动态选择、真实物理核、输出正确性、实际重叠时序；同步/可见性和完成回收语义不能省略。
- PyPTO、Simpler、PTOAS 按 demo 需要做最小修改；复杂能力和正式产品化在基础机制验证后推进，每步实际结果继续记录。
- 2026-10-10 用户明确“先做 TMR 就可以”“HBG 先不管”：当前只推进 TMR，HBG 暂缓，不作为性能小用例或 TMR CSA 接入的前置验收条件。HBG 仅做过源码检查，未实施改动，也未声明支持新协议。
- 2026-10-10 用户曾限定只测128K/B16；2026-10-11最新要求“基于当前的修改，测试一下7组CSA性能的对比”覆盖此前档位限制：本轮单卡测128K×B4/B8/B16/B24及8K×B16/B24/B32，记录min/mean/max。子任务T=96对应B16×验6，其隔离结果不能代替完整CSA。
- 2026-10-10 用户明确“你测试CSA时，用单卡进行测试，别用16卡”：CSA功能与性能均只用单卡；本goal不再以16卡整模型token/DSpark作为验收门槛。已取消尚未执行的16卡排队任务，后续不再提交。
- 2026-10-10 用户明确“功能是功能，性能是性能，分开来看”，没有性能负收益就应持续沿此方向优化：功能按独立入口、动态同核重叠、真实完成/安全等待及数值状态验收，不以测得加速为前提；性能单独记录普通路径公共开销、接续路径净耗时和波动。尚无稳定加速不否定功能，也不撤回该技术路线。保留已验证机制继续精简交接成本；实际回退仍需定位，不能仅由理论推断没有开销。
- 2026-10-10 用户拒绝算子侧`async_aic_return`/`async_aiv_return`及`scalar_prefix`声明，明确目标为PyPTO编译自动插入`early_start`与`fake_end`。同时自动识别并插入后继等待，不要求手写`wait_predecessor()`或协议helper。先做无标记基础小用例，再扩展原CSA；现有人工标记实现仅保留为运行时机制证据，不能当作自动编译目标完成。
- 2026-10-10 后续澄清优先级：自动识别属于未来方向，当前允许在算子原task中插入局部协议语句；不接受为了协议整体改写InCore、拆helper或手写`if ready`包裹计算。近期按过程034及执行方案第5.8节推进，自动识别不再作为当前demo的前置条件。

这些约束优先于本文件旧阶段描述中超出 demo 当前需要的流程要求，不缩小最终动态同核重叠目标。

目标：task 各自独立编译，由运行时选择同一物理 AIV/AIC 上的接续对象，
让前驱异步计算/搬运与后继安全 Scalar 重叠。后继第一次不安全操作之前必须等待；
task 身份、真实完成、依赖释放及资源回收保持正确，不预编译 A+B/A+C 组合入口。
协议参考 NVIDIA PDL 的“提前启动许可＋后继依赖等待”，同核 Scalar/流水实现仍参考 AscendC 的底层交接方式。
执行顺序为定义协议并补齐工具链 → 独立 task 动态选择基线 → 异步返回与真实完成 → 动态同核交接 → 能力扩展 → 性能 → CSA。
当前已迁入私有开发环境；手写入口原型及纯PyPTO DSL/PTOAS自动生成入口均已单卡证明
独立task动态选择、异步返回与同核Scalar/Vector重叠；过程018进一步证明单block AIC的Scalar/Cube重叠。
无后继收尾、普通前驱回退、AIV长链及最小kernel图重放也已有各自证据。
过程023进一步证明已ready的独立后继：两轮48/48个AIV、24/24个AIC均同核计算重叠，
无需人为添加WAIT边或等待无关block；原有数据依赖路径回归通过。
过程024在真实形状NZ O-A的tile探针上证明自然Cube重叠及同表达内净收益；
过程025原tensor表达对照发现tile改写更慢，过程026恢复原tensor lowering后，
普通入口与原写法均值基本相同，异步接续相对原写法少1.43%，并保持自然Cube重叠与精确结果。
过程027已在私有完整128K/B16 CSA通过输出/缓存/图重放验证，首轮均值少0.44%，
17/30配对样本更快，尚不足以确认稳定净收益；过程028在完整图两轮各40条接续中
观测到36/39条自然Cube重叠，功能及等待边界通过。
过程029补齐原SDK括号对照；同实验SDK接续路径均值反而高0.10%，首轮0.44%收益未复现。
原SDK前后均值漂移1.89%，大于候选差距，暂不能定量归因公共开销。下表阶段编号对应执行方案。
**下表通过项均限于既有人工标记原型；自动识别安全前缀、插入等待与异步出口尚未实现。**
过程033记录远期自动分析方案；用户随后澄清，近期先实现原task内局部插入接口（过程034），
不再扩大已有helper/整体改写方式的CSA接入，也不以全自动识别作为近期demo的前置条件。

| 阶段 | 当前证据 | 状态 |
| --- | --- | --- |
| Native 机制参考 | 已安装 CANN 的 SK 四组单卡实测、设备 trace | 已完成参考复现，未证明 CSA 收益 |
| PDL 协议参考 | CUDA Programming Guide、Runtime API、PTX 官方语义复核 | 已修订许可/完成的分层及 PTOAS 修改范围；不是设备实现证据 |
| 0：可用基线与工具链合同 | 私有LLVM/MLIR、PTOAS、PyPTO与TMR构建通过；DSL生成独立设备入口；非法前缀/异步操作拒绝 | AIV及单block AIC原型工具链已可用 |
| 1：独立 task 与动态选择 | A/B/C 三份独立设备二进制；同一注册句柄按运行时 0/1/0 选择 B/C/B，单卡三次换输入及 NaN/trace 毒值检查通过 | 已通过正常同步基线；未声明同核保证或重叠 |
| 2：独立 A 异步返回与真实完成 | 过程012：DSL无后继时返回后632 tick才真实完成，其中V→S实际等待433 tick；输出精确一致 | 自动生成异步尾部及executor末尾收尾通过；过程016补齐kernel模式graph replay |
| 3：动态同核交接与等待后移 | 过程012：自动生成A/B/C，4/4同核、逐元素精确；Scalar运行19 tick后Vector仍需313～331 tick；普通前驱回退后再次交接通过 | 单AIV DSL动态交接与计算重叠已证明；未宣称总体性能收益 |
| 4：多 block、复杂依赖及 AIC 扩展 | 过程014～018：双前驱、sync_start回退、96-block、window=8的长链、图重放及单block AIC通过；过程023再覆盖48 AIV/24 AIC blocks的ready独立后继，每轮全部观测计算重叠 | 已支持有WAIT边及已ready的两类同核接续；具体候选需要的其余能力再补，HBG暂缓 |
| 5：独立 task 三组公平性能对照 | 过程019：无trace的TMR kernel图重放三组对照通过；17-task人工长Scalar小图，等待后移相对普通路径mean下降1.08%，min/max也下降 | 首轮小图端到端收益已证明；短Scalar差距小，不外推CSA |
| 6：真实 CSA task 对接入 | Indexer交接但无计算重叠；过程026的NZ O-A原tensor表达两轮37/40、40/40边有Cube重叠，精确结果通过，相对原写法均值少1.43% | 子任务隔离验证通过，现已在私有完整128K/B16 CSA试接入；尚未修改生产CSA |
| 7：完整 CSA 128K/B16 验收 | 单卡两路径输出与缓存逐bit一致，图重放/换输入/保护区通过；过程028证明完整图自然重叠；原SDK对照输出精确一致 | 功能按单卡范围通过；过程031普通/接续均值970.17/983.01 μs，继续优化交接成本。按用户最新要求，不做16卡验收 |
| 旧固定合组机制对照 | 两种等待形式完成二进制编译；单卡任务最终 exit=4，未进入设备计算 | 已撤回主路线定位，不作为动态交接的阶段通过证据 |

所有 NPU 执行通过统一 task-submit，先 CPU 编译、再单卡；只补当前阶段所需验证。
pytest设备必须显式传`--device "$TASK_DEVICE"`；带`&&`的复合命令不会由队列自动补设备参数。
不能把队列分配卡号当成程序实际选卡证据，详见过程015的错误记录与更正。
不提前铺开七档或整模型，不通过实验脚本删除生成 C++ 的 barrier 达成重叠。
用户新授权“PTOAS 该改就改”：PTOAS 必要的协议表示、同步及异步出口改造已纳入范围，旧的不修改限制撤回。
PTO-ISA 仍使用上游能力，不做额外修改；PTOAS 必要修改仅在私有 worktree 实施。

## 2. 版本与文件范围

下表为本项起始状态；后续版本变化单独追加，不以当前 HEAD 倒推旧实验。

| 仓库/组件 | 版本 | 本项角色 |
| --- | --- | --- |
| vllm-ascend | `dsv4-flash-pto-v0.25.1rc1` / `8e83adda` | 分析、Native 复现、过程记录；后续 CSA 接入 |
| PyPTO | `feat/kernel-mode-integration-test` / `cb2484471` | 基础 DSL/JIT 小用例；后续局部执行能力 |
| Simpler | `feat/kernel-mode-integration-test` / `ef6812891` | 当前使用既有 TMR；尚未修改 runtime |
| PTOAS | `.cache/dsv4-toolchain/ptoas-0.66/bin/ptoas` | 既有复现使用此二进制；新授权允许必要改造，尚未实施/替换安装 |
| CANN | `cann-9.2.0-beta.2`，公共 env 脚本 | Native 及 PyPTO 编译/运行环境 |
| graph-autofusion 参考源码 | `0dc84872fc20015953ea12c0c7622dbcd23f2c2a` | Native AOT/SK 实现参考，非工作工具链替换 |

PyPTO 新增基础用例
[test_scalar_overlap.py](../../../pypto/tests/st/runtime/scheduling/test_scalar_overlap.py)，
之后另有未完成的诊断 API、codegen、wrapper、UT 及
[test_scalar_overlap_trace.py](../../../pypto/tests/st/runtime/scheduling/test_scalar_overlap_trace.py) 草稿，见过程 005。
vllm-ascend 当前涉及分析文档、总验证日志、本文件及
[superkernel_repro_20261009/](superkernel_repro_20261009/README.md)。
其他会话的未提交文件保持原状；截至本节记录，尚未修改生产 CSA 或 Simpler 调度。

## 3. 过程 001：Native 源码复核与 PTOAS 探针

追踪 Native 的 TorchAir 捕获图 → AOT 优化 → SK 能力位 → 队列构造 → 同核子函数调用 → Set/Wait。
确认它利用连续调用及异步流水推进，不依赖“前驱暂停后调用后继 Scalar 回调再返回”的额外设计。
`allow_early_resolve` 只提供部分提前派发条件，不能单独证明用户 Scalar 已在同核执行。

此前在历史 `hc_pre_linear_reduce.pto` 的副本插入外部调用，原版和插桩版 PTOAS 0.66 均退出 0；
调用保留在 TADD 之后。这只证明 lowering 可表达外部调用，没有证明设备指令顺序或异步返回合同。
没有修改生产源码。

证据：[探针结果](results/csa_scalar_overlap_feasibility_20261009/probe_results.json)、
[生成代码](results/csa_scalar_overlap_feasibility_20261009/scalar_hook.cpp)、分析文档第 3～5 节。

## 4. 过程 002：Native 最小实操

新增 AscendC/ACL 两个 AIV 子算子，使用当前安装的 `libascendsk.so`，
执行图捕获、SK 优化及重放。当前 torch_npu Python 白名单不接受 early_start，
因此直接使用已有 C++ ACL/SK 接口，没有更新公共安装包。

四组：普通图、SK/Early-Start 关、SK/Wait 前置、SK/Wait 在 Scalar 后。
编译与运行命令、文件说明见 [复现 README](superkernel_repro_20261009/README.md)。
任务 `task_20261009_165431_362034525498` 在物理卡 2 完成，exit=0，设备已释放。
每组 5 次预热、20 次计时、一次计时外换输入与 NaN 校验，16384 个元素均精确一致。

| 项目 | Wait 前置 | Wait 后移 | 解释 |
| --- | ---: | ---: | --- |
| A/B 物理核 | 62/62 | 40/40 | 每次链内部同核，不要求不同运行固定同一核 |
| A/B 同步配置 | 0x10/0x20 | 0x10/0x20 | 确认运行时 Early-Start 实际生效 |
| B Scalar / Wait | 22/85 tick | 22/64 tick | 最后一次重放的设备 trace |
| B 入口至可读输入 | 107 tick | 86 tick | 支持等待后移覆盖独立 Scalar 的机制判断 |

没有分离 Vector 与末尾 MTE3 的占比，没有 AIC 或 CSA 证据。
ACL event 均值受 Host/图调度空隙影响，不能用其百分比推算生产收益。
完整 min/mean/max、全部样本及 trace 见 [results.json](superkernel_repro_20261009/results.json)。

遇到的问题及处理：

- 公开 WaitPreTaskEnd 只有 bool 模板参数；使用 `WaitPreTaskEnd<true>()`，不混用内部模板签名。
- 显式采用 Set/Wait 的 FORCE=true，避免本例未走自动编译宏时退化为空操作。
- 队列会追加 `--device N`；最初脚本误当成输出目录，现已兼容，失败发生于设备执行前。
- 设备反汇编返回 `<not available>`；设备 `-S`、`-mllvm -print-after` 被拒绝。
  保留这个证据缺口，不将预处理输出或符号表称为设备汇编。

## 5. 过程 003：独立 PyPTO 两 task 基线

### 实现及通过范围

新增 `@pl.jit two_task_chain`，两个独立 `pl.at(CORE_GROUP)`：
A 写 `mid=x+1`，B 读 mid 写 `out=mid+2`；32×32 FP32，单 AIV block，TMR。
采用现有 JIT、pass、PTOAS、CANN、ChipWorker 路径。

CPU 编译通过后，使用真实 DeviceTensor：每次先复制输入、把 mid/out 写成 NaN，
再执行并复制回来，核对输入未被修改、中间值和最终值精确一致。
两次输入为 `arange(1024)%31` 与其减 11；每个输出 1024 个元素。
这是正常依赖与完整等待的功能基线，不宣称同核映射或重叠。

### CPU 编译及失败修正

首次断言 `compiled.program` 内必须有两个 FunctionType.AIV，实际得到 0。
检查产物却已生成 producer/consumer 两份 AIV 源码及两次提交的 orchestration。
原因是断言使用的 IR 对象与最终生成的任务层级不一致，不是“两个 task 未生成”。
改为检查实际 `kernels/aiv` 产物，再调用 `compiled.load()` 完成设备二进制构建。

CPU 结果：1 passed，无设备执行。产物：
[baseline/](../../../pypto/build_output/scalar_overlap_20261009/baseline/)、
[baseline_compile.xml](../../../pypto/build_output/scalar_overlap_20261009/baseline_compile.xml)。
包含两个 `.bin`，没有把单纯生成 C++ 当作二进制编译完成。

### 单卡任务与结果

第一次任务 `task_20261009_173110_210626612109` exit=0，但只有锁日志、没有 pytest XML。
核查队列 COMMAND 按行保存，多行字符串只执行了第一行 source；**此任务不算设备测试通过**。
改用单行、带 `&&` 的完整命令后重新提交。

有效任务 `task_20261009_173158_215808217987` 在物理卡 8 完成，exit=0，设备已释放。
pytest **1 passed，0 skipped，0 failures**；JUnit 记录两次真实执行、输入输出精确一致。
产物：[baseline_device/](../../../pypto/build_output/scalar_overlap_20261009/baseline_device/)、
[baseline_device.xml](../../../pypto/build_output/scalar_overlap_20261009/baseline_device.xml)。
未采计时或 profiler，避免将测试耗时当作 task 性能。

### 复现命令

以下命令记录历史基线的执行方式；当时的诊断导入不一致已在过程 008 修复，后续实验使用该节私有环境。
从 PyPTO 仓执行 CPU 检查；后续增加参数化用例后，用 `-k test_two_task_baseline` 限定基线。
复跑请更换产物目录，避免覆盖本轮证据。

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/st/runtime/scheduling/test_scalar_overlap.py \
  -k test_two_task_baseline -n "$PYPTO_TEST_JOBS" --codegen-only --platform=a2a3 \
  --save-kernels --kernels-dir="$PWD/build_output/scalar_overlap_recheck/compile" \
  --junitxml="$PWD/build_output/scalar_overlap_recheck/compile.xml" -q
```

队列的引号内须保持单行，`$TASK_DEVICE` 在任务执行时展开：

```bash
/data/server-toolkits/pto-task/app/task-submit --device auto --max-time 240 'source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh >/dev/null && cd /data/pyptouser/qinchuanyu/pto-eager/pypto && source .claude/skills/testing/load-env.sh && export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" && python -m pytest tests/st/runtime/scheduling/test_scalar_overlap.py -k test_two_task_baseline -n 1 --platform=a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir="$PWD/build_output/scalar_overlap_recheck/device" --junitxml="$PWD/build_output/scalar_overlap_recheck/device.xml" -q'
```

## 6. 过程 004：固定合组与两种等待位置的编译原型

本过程是历史尝试，已撤回其主路线定位：它把两个计算放进一个运行时 task，
不能验证独立编译 task 的动态交接。现仅保留机制对照和失败证据，不继续作为 CSA 接入方案。

基础设备功能通过后，在同一 ST 文件增加 `local_task_chain`：

1. A 与 B 放在一个 `pl.at(CORE_GROUP)` 编译单元，外部仍按一个完整任务完成。
2. A 的 store 与 B 首次读取之间保留显式 `pl.system.bar_all()`。
3. 编译期参数选择等待在 Scalar 前或后；当前是人工构造的局部链，不是通用自动合组 pass。
4. Scalar 使用运行时 seed/steps 做 `(state*17+3)%65521`；等待后才把结果写到独立 INT64 输出。
5. A/B 浮点算术仍为 +1/+2；另检查 Scalar 结果，防止递推成为无用计算。
6. 保留编译器正常内部同步及最终等待；没有提前 FIN，没有手改生成 C++。

选择显式 barrier 而非手动 set/wait，是因为当前 PTOAS 遇到显式 set/wait 会跳过自动同步插入。
小用例先验证功能，不能因此推断自动插入的位置、最终机器指令顺序或实际可重叠窗口已符合目标。

首次 CPU 收集失败：误写 `pl.Constexpr[bool]`，当前 API 为 `pl.constexpr`，已修正。
同次审查发现 RunConfig 没有 `with_changes`，改用现有 `dataclasses.replace`；这一项在进入相关测试前修正。
首次失败证据：[local_compile.xml](../../../pypto/build_output/scalar_overlap_20261009/local_compile.xml)。
后续结果使用新目录记录，不覆盖这份失败记录。

修正后，基线及两个 constexpr 分支的 CPU 检查 **3 passed**，包含 `compiled.load()` 的设备二进制编译。
结果：[local_compile_v2.xml](../../../pypto/build_output/scalar_overlap_20261009/local_compile_v2.xml)。

实际生成代码核对结果：

| 位置 | Wait 前置 | Wait 后移 |
| --- | --- | --- |
| A 最后 store | TSTORE，随后 MTE3→MTE2 set_flag | 相同 |
| 独立 Scalar 前 | PIPE_ALL barrier | 无这一 boundary barrier |
| Scalar | 运行时 seed/steps 的整数循环 | 相同 |
| Scalar 后 | 写 Scalar 校验值 | PIPE_ALL barrier，然后写 Scalar 校验值 |
| B 首次读取 | MTE3→MTE2 wait_flag，然后 TLOAD | 相同 |
| 整体结束 | 最终同步、Scalar 写回的 DCCI/DSB 和正常 tail | 相同 |

对应源码：[wait_first C++](../../../pypto/build_output/scalar_overlap_20261009/local_compile_v2/local_False/ptoas/overlap_local_chain.cpp)、
[scalar_first C++](../../../pypto/build_output/scalar_overlap_20261009/local_compile_v2/local_True/ptoas/overlap_local_chain.cpp)。
每种形式只生成一个 AIV 内核，orchestration 只提交一个任务；Scalar 递推仍在设备函数内，未被提升到 Host。
这证明当前编译产物的结构与源码顺序；尚不能证明最终设备指令的重排情况或隐藏了多少流水时间。
Scalar 校验输出引入的 DCCI/DSB 是诊断开销，后续性能对照必须三组一致，不能拿当前无 Scalar 的基线直接作公平计时。

单卡功能任务 `task_20261009_173748_25502084348` 在 17:38 查询时为 pending；
当时 16 张卡均被 CI 任务 `task_20261009_172929_198441518784` 占用，没有绕过队列。
后续实际分配物理卡 1，最终 **exit=4**：conftest 导入时报 `system.read_clock` 未注册，未进入设备计算，设备已释放。
原计划结果位置为 `pypto/build_output/scalar_overlap_20261009/local_device_v2.xml`，没有取得三个功能检查通过的证据。
命令沿用第 5 节队列形式，去掉 `-k test_two_task_baseline`，产物目录改为 `local_device_v2`，
JUnit 改为 `local_device_v2.xml`；单 worker 顺序执行三个功能检查。
排队期间该测试源码未改，但共享 PyPTO Python 包被诊断草稿修改，实际加载扩展没有相应注册，导致上述失败。
只冻结测试文件不足以冻结执行依赖；后续必须一并固定 Python 包、实际加载的 C++ 扩展及工具链。
当时该 ST 的 Ruff check 通过，format --check 提示需格式化；当前仅修正文档，未继续格式化或修改源码。

原“继续完成固定合组的单卡验证”不再是主线下一步。恢复开发后先处理依赖一致性，
再核实独立入口异步返回合同，按修订后的动态路线推进。尚未进入 CSA。

## 7. 过程 005：设备诊断接口草稿与导入失败

在固定合组任务排队期间，尝试增加 `pl.system.read_clock()` 与 `physical_core_id()`，
用于记录设备计数值、实际物理核和等待区间。这是未完成的诊断工具，不是异步交接实现。

| 改动范围 | 草稿内容 | 验证状态 |
| --- | --- | --- |
| `src/ir/op/sync_ops/sync.cpp`、Python IR/language API、API 文档 | 注册并暴露两个诊断操作 | 实际加载扩展尚无新操作，Python 导入失败 |
| `src/backend/common/pto_ops_memory.cpp`、`src/codegen/pto/pto_codegen.cpp` 及其头文件 | 诊断操作 lowering、返回值生成 | C++ 构建通过；端到端 codegen 未通过 |
| `python/pypto/backend/pto_backend.py` | A2/A3 wrapper 适配 `get_sys_cnt()` / `get_coreid()`，限制不支持后端 | 未经完整编译和设备验证 |
| 两个定向 UT 文件、`test_scalar_overlap_trace.py` | API/codegen 检查及带 trace 的设备用例 | UT/ST 收集阶段失败，未取得通过证据 |

执行过 `cmake --build build`，初次构建及返回值处理修正后的增量构建成功。
日志为 `/tmp/pypto-scalar-diagnostic-build.log`、`/tmp/pypto-scalar-diagnostic-build-v2.log`。
新扩展生成在 `build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so`，
没有同步成当前 Python 环境实际加载的扩展；Python backend 在导入时查找新操作，因注册表缺项而失败：

```text
python/pypto/backend/pto_backend.py
ValueError: Operator 'system.read_clock' not found in registry
```

定向 UT 与 trace ST 的 CPU 检查均止于这个导入错误，日志为
`/tmp/pypto-device-diagnostic-ut.log`、`/tmp/pypto-scalar-trace-compile.log`；后者不能称为完整设备二进制编译通过。
新增 trace 文件另有 Ruff E501 长行待处理；不将诊断草稿视为可用设备观测 API。
前一节排队任务同样因此失败，不是设备死锁、同步错误或数值失败。

用户随后要求暂停实现；查询/取消原排队任务时它已结束。本次文档修订只记录事实，
没有继续安装扩展、回滚源码、修复测试或提交新的 NPU 任务。恢复开发时须先使源码与加载扩展一致。

## 8. 过程 006：按独立编译与动态同核交接重写执行方案

用户指出 PyPTO 在运行时选择 task，无法预先知道两个相邻 task 并编译成一个入口。
据此纠正旧方案：固定合组不能替代目标，Simpler 的动态选择及完成协议改造是必经工作。
Native 的可调用子入口与内部 Set/Wait 可作参考，但其预先构造的执行队列不能直接代表 PyPTO 动态调度。

本次只修订 [分析与执行方案](DSV4_FLASH_CSA_SUPERKERNEL_ANALYSIS.md) 并同步本记录和总验证日志：

1. 明确 PyPTO 独立编译前缀/等待边界与 Simpler 运行时决定接续关系的分工。
2. 区分 `ISSUED`、block 真正完成、整个 task 完成；禁止提前 FIN、提前回收和 B 等 A FIN 的循环等待。
3. 补齐安全前缀、wrapper 状态、事件所有权、代次、片上资源及无后继收尾约束。
4. 把工具链异步返回合同列为首要门槛；当前无证据证明 PTOAS 可表达，不以静态合组绕过。
5. 阶段验收先用同一批 A/B/C 独立入口证明运行时选择，再验证交接及公平性能，最后扩展到 CSA。
6. 更正旧任务 pending 状态，保留真实通过、编译通过、导入失败与尚未实现之间的边界。

本次文档修改没有恢复功能开发或设备测试，也没有把动态交接、重叠或 CSA 性能标为完成。
文档检查通过：`git diff --check` 无错误，两份方案/过程文档共 25 个本地链接均可解析，代码围栏配对完整。
后续过程继续追加；本节完成的是方案修订。

## 9. 过程 007：复核 NVIDIA PDL，分开启动许可与数据完成，允许改造 PTOAS

日期：2026-10-09。用户要求参照 GPU PDL 审视实现过程，并明确授权“PTOAS 该改就改”。
结论：有必要修订。独立编译和动态同核交接继续作为目标，进一步把提前启动许可从完成协议中单列，
并把缺失的 PTOAS 能力列为开发任务。本轮只做官方资料/本地源码复核及文档修改。

### 9.1 官方 PDL 事实与不能直接照搬的部分

PDL 在独立 CUDA kernel 之间允许提前启动后继，需要显式启用；支持 stream 和 CUDA Graph 依赖。
它是依赖协议，不负责把 A/B 源码合编，也不是由前驱任意创建后继的 Dynamic Parallelism。
并发是机会而非保证，正确性不得依赖一定出现并发。
依据：[CUDA Programming Guide：PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)。

| 参考点 | 官方机制 | 对当前方案的修正 |
| --- | --- | --- |
| 提前启动 | `cudaTriggerProgrammaticLaunchCompletion()` 在全部 CTA 触发或退出后满足相应调度边；自身不保证数据可见 | 增加独立的提前许可，不能复用 FIN 或数据完成位 |
| 后继等待 | `cudaGridDependencySynchronize()` 等待直接 grid 依赖完成；PTX `.wait` 还明确内存操作的可见性 | 后继安全前缀后等待真实依赖完成，再进入数据访问/资源复用区 |
| 指令语义 | PTX `.launch_dependents` / `.wait` 针对 runtime 定义的依赖对象 | 编译 task 不嵌入相邻函数地址；实例和依赖上下文由 Simpler 传递 |

依据：[CUDA Runtime API：触发与等待](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EXECUTION.html)、
[PTX `griddepcontrol`](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol)。
不要被 LaunchCompletion 的名称误导：它释放的是调度条件，不能因此读取前驱输出。

对 Ascend 的实现判断如下，属于结合本地执行模型的推导：

- PDL 没有同 SM 的放置保证；本项目仍须单独保证并验证同一物理 AIV/AIC。
- GPU 的前驱调用 trigger 后可以继续执行，PDL 本身不要求函数返回。当前 Ascend Scalar executor 串行调用入口，
  因此同核方案仍需 A 在提交完自己的工作后交出 Scalar 控制权，首版不引入暂停/恢复 A 的调用栈。
- GPU 示例中可提前的初始化不能直接搬到共用片上状态的 NPU task；UB/L1/L0、event、atomic/mask 等操作仍受等待边界约束。
- PyPTO 任意 DAG 的准入、B/C 选择、不同 block 数和实例代次由 Simpler 实现；PDL 文档不能替代这些协议证明。

因此 Native SK 不只提供效果数字，还提供 Ascend 侧流水/状态处理参考；其静态队列不作为动态调度实现。
设计分工为：PDL 参考依赖语义，AscendC 参考硬件交接，PyPTO/PTOAS/Simpler 实现独立 task 的动态协议。

### 9.2 从两项完成状态扩展为明确分开的协议条件

以下是设计术语，尚不是已提供的 API：

1. **提前许可**：前驱/运行时允许后继被考虑提前调度，不表示数据可读。
2. **Scalar 已交接**：本核前驱已经返回 `ISSUED`，相关异步工作仍可在途；此时同核后继才有机会执行安全前缀。
3. **真实完成与可见性**：相关 block/task 的输出达到约定可见性，才允许依赖读取、正常 FIN 及按引用规则回收。

`allow_early_resolve` 已发布全部 producer block 的条件属于预派发基础，不能直接等同于 PDL 的触发聚合或真实完成。
首版一个 AIV block 可简化聚合，但多 block 扩展时必须分别聚合提前许可和真实完成，不能以局部进度代替整个依赖。
没有提前许可、没有合格同核后继或没有可重叠窗口时，都要能完整串行结束。
B 的边界等待必须能推进 A 的完成处理，不能形成“B 等 FIN、FIN 又等 B 返回”的死锁。

### 9.3 PTOAS 的只读源码结论与必要改动点

本轮读取的是本地 `ptoas-src`，没有修改源码、构建新二进制或切换安装版本。
原 `ptoas-0.66` 探针和当前本地源码仍分开记账；下列观察不声称已验证所有上游版本。

| 已定位的源码 | 当前行为 | 后续开发要求 |
| --- | --- | --- |
| `PTOToEmitCPass.cpp` 的 `ptoas_auto_sync_tail()` | 全流水 barrier 或 MTE3→Scalar 的立即 set/wait，都会等待 | 提供显式交接模式：提交完成记录后返回，由调用方或后继消费完成；普通入口保留原语义 |
| `PTOInsertSync.cpp` 的显式同步检查及 `analyzer.Run()` | Set/Wait/RecordEvent 会跳过自动插入；正常分析包含末尾同步 | 将 task 协议同步与 task 内部流水同步区分，新协议不能使内部依赖失去同步 |
| `ptoas_pipeline.cpp` 的 tail hint 校验 | 仅接受当前两类 tail hint | 新能力需正式表示、验证及 ABI/编译缓存区分，不能静默忽略属性 |
| `ptoas_cpp_rewrite.cpp` 的 Scalar GM flush | return / auto-sync tail 附近另有写回处理 | 审查新出口下的写回可见性及隐含等待，不能提前发布完成 |
| `PTOInjectBarrierAllSync.cpp` | 该同步模式会在 return 前补 barrier | 明确交接模式的兼容或拒绝条件，防止最终生成代码又隐式等待 |

完整路径及仓库分工见 [执行方案第 5.5 节](DSV4_FLASH_CSA_SUPERKERNEL_ANALYSIS.md#55-工具链门槛ptoas-修改范围及仓库分工)。
PTOAS 还需确保等待边界的副作用/内存顺序、事件保留和全部返回分支；不能只增加一个“关闭尾部 barrier”开关。
PyPTO 负责表达能力和安全前缀、处理 wrapper 状态；Simpler 负责许可准入、同核选择、完成通知和资源生命周期。
用户授权已足以开展必要 PTOAS 改造，不再为同一修改范围重复请求许可；硬件/ISA 缺口仍须如实报告。

### 9.4 对后续验收的补充与本轮结果

保留先 CPU、单卡小用例、最后 CSA 的顺序，不增加无关测试矩阵。新增关注点是：

- 相同独立入口在运行时选择 A→B/A→C；强制串行仍正确，不依赖重叠才能完成。
- 故意保留“许可已到、结果尚未完成”的窗口，验证后继前缀可以执行，而依赖读取不能越过等待。
- 区分每个 block 的许可/返回/完成和整个 task 的聚合；覆盖无后继、未触发许可、回退及图末尾收尾。
- 编译验证内部同步未丢失、所有出口覆盖、非法交接模式被拒绝；设备证据验证真实同核、内存可见性及流水重叠。
- 性能仍比较 task 数量不变的普通路径、交接且前置等待、交接且前缀后等待，保留全部样本 min/mean/max。

已同步更新分析文档的 PDL 对照、状态、PTOAS 范围和阶段计划，并更新本文件当前约束。
本轮没有代码实现或设备测试；旧诊断草稿的 Python/扩展导入不一致仍待恢复，动态协议没有被标成已经可用。
文档检查：`git diff --check` 通过，两份文档的 26 个本地链接可解析，代码围栏配对完整；不重复运行与文档无关的功能/性能测试。

## 10. 过程 008：恢复诊断扩展，隔离并发开发环境

日期：2026-10-09。用户授权按最小 demo 推进，并指出另一会话正在优化 PyPTO 算子。

### 已完成与当前边界

- 在隔离指令前，重新以 editable 方式构建、安装共享 PyPTO 的同源扩展，修复过程 005 的操作注册缺失。
  `system.read_clock` 和 `system.physical_core_id` 已可导入；两个相关 UT 文件按诊断关键词筛选，**3 passed**。
  日志：`/tmp/pypto-scalar-demo-install.log`、`/tmp/pypto-scalar-demo-ut.log`。
- 添加三个独立 `pl.at` 入口 A/B/C 的基线，由运行时 `choice` 选择 A→B 或 A→C。
  同一份编译产物、同一注册句柄测试 0/1/0 三次选择；输入改变，输出及 trace 先写入毒值。
  这一步仍保留正常同步，不宣称已有重叠。
- 首次 CPU 编译发现分支标记常量是 INDEX，而 trace 为 INT64，`tensor.write` 拒绝类型不匹配。
  私有副本已增加显式 INT64 转换，并避免回读输入覆盖后续样本的基准输入。
  首次失败日志：`/tmp/pypto-dynamic-successor-compile.log`；失败发生于 DSL 解析，没有设备执行。

### 私有工作区

工作区根目录为本工作空间的 `.worktrees/scalar-overlap-demo-20261009/`：

| 子目录 | 来源 | 隔离方式 |
| --- | --- | --- |
| `pypto/` | `cb2484471` 加本项诊断/用例草稿 | 独立 worktree、独立构建目录，runtime 指向右侧私有 Simpler |
| `simpler/` | `ef6812891` | 独立 worktree、独立构建和 runtime 缓存 |
| `ptoas-src/` | `f5eff3ee` | 独立 worktree；未复制共享仓无关的 `.codex/CLAUDE.md` 修改 |
| `.venv/`、`env.sh`、`artifacts/` | 本项新建 | 私有项目安装、环境入口、测试/设备日志 |

PyPTO 的 libbacktrace/msgpack 源码也使用独立 worktree，未复用其他 checkout 的编译产物。
CANN/GCC 和第三方 Python 依赖只读复用；共享 venv 的 editable `.pth` 不在私有环境执行。
后续不再向共享 venv 安装项目，也不编辑另一会话使用的共享 PyPTO/Simpler/PTOAS 源码。
本记录继续放在 vllm-ascend 原路径，方便后续查阅；生产 CSA/HCA 不受本项修改。

私有 PyPTO/Simpler 已完成构建安装。`artifacts/environment_paths.json` 记录并确认私有 venv、
PyPTO/Simpler Python 模块、两个 C++ 扩展、runtime 源码及缓存目录均在该工作区内。
首次路径检查误把 venv 的 Python 符号链接解析到基础解释器后判为越界，已改为检查 `sys.prefix`；
这是检查方式错误，没有改动系统解释器或公共环境。
首次私有 CPU 编译 `dynamic_compile_v1` 在 CANN 后端失败：诊断 helper 使用 `+r` 约束，
报 `Unsupported other types than int64 for inline`。参考 runtime 既有 64 位寄存器读取使用的 `l` 约束，
私有 PyPTO wrapper 改为 `+l`，仍不添加任何硬件等待；相关旧 UT 的预期字符串同步修改。
再次执行 `test_dynamic_successor_baseline --codegen-only --platform=a2a3`：**1 passed**，
三个独立入口的设备二进制均完成构建。产物为私有 `artifacts/dynamic_compile_v2/` 及同名日志/XML。
仅修改私有源码；原 `dynamic_compile_v1` 失败产物保留。

首次单卡任务 `task_20261009_230648_365736914227`，物理卡 0，exit=1：调用正常返回，
但 mid 仍全为预填 NaN，因而功能不通过。检查已链接 ELF 发现 `.text=0x1000`，
两个诊断 helper 分别占前 8/12 字节，`kernel_entry=0x1014`；PyPTO 当前加载器只抽取 `.text`，
从起点调用，未检查 entry 偏移，实际执行了 helper 并直接返回。
因此将 helper 改为 C linkage 下的 `static inline + always_inline`，使它们只在实际 kernel 内展开。
本用例复用 Simpler 已有 ELF 解析器，在 CPU 编译阶段检查三个入口均在 `.text` 起点，避免再次占卡才发现空执行。
没有修改共享环境，也没有增加每次 dispatch 的验证开销。

修正后的 `dynamic_compile_v3`：**1 passed**，包含三个真实设备二进制和入口布局检查；
随后以 `artifacts/dynamic_device_v2/` 新目录重跑单卡基线：
`task_20261009_231035_382005015310` 在物理卡 0 完成，exit=0，pytest **1 passed**，卡已释放。
同一注册句柄按 choice=0/1/0 执行 B/C/B，三次输入分别加 0/-11/+7；输入未被修改，
1024 个 mid/out 元素每次均精确符合 A:+1、B:+2、C:+5 的预期，trace 标记也分别为 2/5/2。
生产者/消费者物理核依次为 **51/49、49/51、49/49**，说明普通调度不会保证同核；
本轮只验证独立入口、动态选择及诊断工具，没有将这些起始时间差当作重叠或性能收益。
完整结果见私有 `artifacts/dynamic_device_v2/dynamic_successor_results.json` 和同名 XML。
尚未修改动态交接或提前完成协议。

进一步源码核实发现，PTOAS 的 `appendScalarGMFlush()` 会为 `tensor.write` 这类 Scalar GM 写回
补 `PIPE_ALL`、DCCI 和 DSB。当前基线正常全同步，因此 trace 写回不影响该基线结论；
后续异步 producer 不能直接照搬这种末尾 trace 写法，否则观测动作本身就排空了待重叠流水。
异步 demo 的 producer 时序应由 executor 记录，后继的 trace 在真实等待后写回；计时 helper 本身不加硬件 barrier。

运行时最小候选（尚未实现/验证）：只接受单 AIV block、唯一前驱、同核已预派发且明确声明安全前缀的接续。
A 异步返回后暂不发 FIN，B 可以先执行 Scalar，但 **B 的 ACK 延迟到等待 A 真正完成之后**。
B 在等待边界发布 A 的真实 FIN，利用既有 gated pending 的完成处理推进依赖，再进入主体。
这样需要区分“B 已进入 Scalar”与现有 ACK，不能直接沿用 executor 在调用 B 前发 ACK 的位置。
没有合格同核后继时，executor 自行等待并收尾 A；不等待未来任务出现，不假设动态调度一定产生重叠。

本机可访问路径尚未找到 LLVM/MLIR 19 开发包。已在私有 `.deps/llvm-project` 以
`--depth 1 --single-branch --branch feature-vpto` 获取源码，版本 `fa2fd1f75`，
开始准备不含 Python bindings 的 CLI 构建依赖（`artifacts/llvm_configure.log`、`llvm_build.log`）。
这不会替换公共 PTOAS；后续本地编译器的切换仅在私有 `env.sh` 中完成。

## 11. 过程 009：独立 AIV 异步返回与 executor 真实收尾

继续仅在过程 008 的私有环境实现，不修改共享 PyPTO/Simpler 安装。
先用 PyPTO `external_source` 接入一个独立 AIV 二进制，隔离验证硬件调用边界；
PTOAS 自动生成异步返回的路径另行实现，不能把手写入口通过等同于编译器路径通过。

私有 Simpler 的 `GlobalContext` 增加异步完成事件标记与诊断地址，
`aicore_executor.cpp` 在入口返回后消费 MTE3→S EVENT_ID7，再允许现有 FIN 流程继续。
生产者用 MTE2→V→MTE3 的事件串联覆盖工作完成，Scalar 不在入口末尾等待；
内部 load/add/store 同步保留，不改写生成的 C++ 来删 barrier。
trace 由 executor 在真实等待后写回，不引入 producer 末尾的 `tensor.write` 全流水等待。

用例为私有 PyPTO `tests/st/runtime/scheduling/test_async_aiv_return.py`，
手写入口在同目录 `kernels/async_aiv_producer.cpp`。
单个 128×128 FP32 tile 连续加 1 共 64 次；同一注册句柄用运行时参数切换
“入口内等待”与“executor 等待”，四次都换输入并将输出预填 NaN。
首次 CPU 用例仅因错误假定 external-source 的 `.o` 会放进普通 kernels 目录而失败，
去掉这项路径假定后 `async_return_compile_v2` 为 **1 passed**。
私有运行时重编译记录为 `artifacts/async_runtime_build_v1.log`。

单卡任务 `task_20261009_232737_2932955117`，物理卡 0，exit=0，**1 passed**，卡已释放。
输入不被修改，四次全部 16,384 个输出元素精确等于输入+64。

| 入口内等待 | 输入偏移 | 核号 | 入口到返回/tick | 返回到真实完成/tick |
|---|---:|---:|---:|---:|
| 是 | 0 | 29 | 703 | 0 |
| 否 | 7 | 27 | 485 | 207 |
| 是 | -11 | 73 | 728 | 1 |
| 否 | 19 | 25 | 502 | 181 |

证据：私有 `artifacts/async_return_device_v1/async_return_results.json` 与同名 XML。
结论限于：独立函数返回允许 AIV 流水继续执行，executor 可跨调用边界消费事件并正确收尾。
当前没有执行后继 Scalar，因此不是跨 task 重叠或性能收益的验收。
下一步接入同核、唯一前驱、显式安全 Scalar 前缀的动态后继，ACK 仍须在真实等待后发布。

PTOAS 私有实现草稿增加 `async_aiv_event7` 尾部策略、限定 AIV load/add/store 原型范围，
并为相关 pipe-pair 保留 EVENT_ID7；尚未构建或验证，不宣称可用。
LLVM/MLIR 19 私有依赖构建遇到 GCC 15 下 `int64_t` 未声明，
仅给私有依赖的 `ValueBoundsOpInterfaceImpl.h` 补 `<cstdint>` 后继续构建；公共工具链未改。

## 12. 过程 010：运行时选择独立 B/C，完成同核 Scalar 交接

私有 Simpler 没有把 A+B/A+C 预先编译成一个入口。
`CoreTaskArgs.scalar_prefix` 是新增的显式原型合同：单 AIV、唯一前驱，入口先做纯 Scalar，
第一次改 vector/atomic 模式、读前驱数据或访问 UB 之前调用 executor 的等待边界。
调度器只为满足条件的后继选择其仍在运行的前驱物理核；其他情况保留普通完成/派发路径。
`allow_early_resolve` 仍负责预派发条件，本身不等于允许用户 Scalar 提前执行。

具体交接：A 返回后先不发 FIN；若已预派发的同核 B/C 合格，则进入该独立入口，但暂不发其 ACK。
后继的 Scalar 前缀结束后通过 runtime callback 消费 EVENT_ID7，再发布 A 的真实 FIN；
现有 gated-pending 完成处理释放依赖并发后继 doorbell，随后发布后继 ACK，才执行后继主体。
没有合格后继时 executor 自行等待 A，不等待未来任务出现；最后一个任务也走真实收尾。
当前 profiler/PMU/args-dump 开启时仍退回普通完成流程，避免把既有串行记录所有权误当成已支持重叠。

私有 runtime context 扩展后 DispatchPayload 为 576 字节，所有参与方在同一私有环境重建；
不能将这些原型二进制与共享环境混装。核内修改的 context 在发布 FIN 前须写回，
避免双缓冲槽复用时旧的脏缓存行覆盖下一次派发字段；这项补充已进入后续候选，另行验证。

源码：私有 PyPTO `test_dynamic_scalar_prefix.py` 与 `kernels/async_aiv_producer.cpp`、
`scalar_prefix_b.cpp`、`scalar_prefix_c.cpp`、`scalar_prefix_consumer.h`。
同一注册句柄按运行时分支切换 B/C，A 做 1024 次 tile 加1，B/C 分别再加2/5，
Scalar 做64轮整数递推。每次换输入并毒化全部输出，八次输入保护、mid/out逐元素和 Scalar 结果精确通过。
首次 CPU 编译因 orchestration 不支持将 `pl.cast` 表达式直接作为调用实参而失败；
先绑定为 Scalar 变量后 `dynamic_prefix_compile_v2` 为 **1 passed**。

首轮任务 `task_20261009_233917_74379513109`，物理卡0，exit=0，**1 passed**；
加入 `wait_flag` 紧邻的前后时间戳后，任务 `task_20261009_234139_83933231932` 再次 **1 passed**。
后者八次都发生同核交接，记录如下；tick 未换算为微秒，不擅自假设时钟频率。

| 序号 | 后继 | 等待位置 | A/B物理核 | Scalar/tick | 真实硬件wait/tick |
|---|---|---|---:|---:|---:|
| 0 | B | Scalar 前 | 65/65 | 19 | 111 |
| 1 | C | Scalar 前 | 61/61 | 28 | 97 |
| 2 | B | Scalar 后 | 61/61 | 19 | 26 |
| 3 | C | Scalar 后 | 61/61 | 19 | 29 |
| 4 | B | Scalar 前 | 55/55 | 19 | 104 |
| 5 | C | Scalar 前 | 55/55 | 33 | 43 |
| 6 | B | Scalar 后 | 51/51 | 19 | 88 |
| 7 | C | Scalar 后 | 51/51 | 33 | 15 |

后移等待的四次中，Scalar 结束后0～5 tick进入硬件wait，wait仍持续15～88 tick：
这支撑“同核后继 Scalar 执行时前驱异步流水仍未完成”。
硬件wait的 min/mean/max 为：前置43/88.75/111 tick，后置15/39.5/88 tick。
这是功能/时序原型证据；跨重放物理核和启动条件有差别，尚未做完整动态路径的公平性能验收，
不将等待缩短当成 CSA 或整模型收益。
产物：私有 `artifacts/dynamic_prefix_device_v1/`、`dynamic_prefix_device_v2/`，
各自 `dynamic_scalar_prefix_results.json`、XML 与队列日志；两次卡均已释放。

编译器路径已经开始：`async_aiv_return` 映射 PTOAS 异步尾部合同；
`system.wait_predecessor` 携带 raw dispatch 参数调用同一个 runtime 边界；
marked Scalar-prefix 函数将 atomic/vector 模式初始化推迟到等待之后，PTOAS 检查前缀中不含不安全操作。
完整 DSL 对应例为 `test_dynamic_scalar_prefix_dsl.py`，CPU 发射 IR 已通过，
尚未编译成设备二进制。DSL 暂不支持 Scalar `&`，时钟依赖改用运行时模数1，结果不变且不被常量折叠。

私有 PTOAS 当前 `PTO_ENABLE_PYTHON_BINDING=OFF` 配置仍无条件建立 Python 扩展，配置失败；
补充该既有配置的独立 CLI 入口，仍调用同一 `runPTOAS` 驱动后配置通过。
没有替换默认公共 PTOAS，没有改其优化逻辑以回避验证。LLVM/MLIR依赖与PTOAS对象继续在私有目录构建。

## 13. 过程 011：区分 Vector 计算重叠与末尾搬运重叠

2026-10-10 继续在私有环境推进。将公共环境脚本的当时内容保存为私有
`base-env.snapshot.sh`，私有 `env.sh` 不再实时 source 可被其他会话修改的公共入口；
CANN/GCC/第三方依赖仍只读复用，项目源码、安装包、运行时库和缓存保持独立。

过程010的 EVENT_ID7 MTE3→S 等待覆盖了 Vector 和末尾搬运，不能据此单独声称
Vector 计算在后继 Scalar 结束时仍未完成。补充 V→S EVENT_ID7 诊断标记，
executor 先等待 Vector 标记并读取时间，再等待原 MTE3 标记；不提前发布 FIN。
相关编译器事件保留同步更新，A trace 第7项由未使用的0改为 Vector 等待结束时间。
为钉住 Scalar 递推起点，手写后继也增加运行时模数1，将开始时钟作为递推数据依赖。

私有 runtime 重编译通过；CPU `vector_marker_compile_v1` **1 passed**。
单卡任务 `task_20261010_001003_198030811462` 的八次输出、Scalar 结果和同核交接均正确，
但新增的“Scalar结束后 Vector 仍未完成”断言失败，因此不能记为完整通过。
原128×128 tile下，等待前置时 V→S 等待为10/10/7/1 tick；
等待后移时四次均为0 tick，而后者MTE3等待仍为83/22/90/22 tick。
这说明旧证据没有区分计算与搬运；也不证明 Scalar 与 Vector 完全没有重叠，
因为 Vector 可能在 Scalar 期间完成。保留该证据缺口，不能用更强结论替代实测。
原始数据：私有 `artifacts/vector_marker_device_v1/dynamic_scalar_prefix_results.json`、XML及任务日志。

下一候选将 tile 扩到128×256，保持加法、运行时选择和64轮 Scalar 不变，
让 Vector 尾部更长，再检查真实 V→S 等待；编译器 DSL 小例子同步采用此形状。
仍不把诊断等待数据换算为完整路径或 CSA 性能收益。

该候选CPU编译 `vector_marker_compile_v2` **1 passed**；
单卡任务 `task_20261010_001224_207913213299`，物理卡0，**1 passed**，8/8同核交接，
每次32,768个mid/out元素精确一致、输入保护和Scalar整数结果通过。
输出目录为私有 `artifacts/vector_marker_device_v2/`，JUnit为同名XML。

| 序号 | 后继 | 等待位置 | A/B物理核 | Scalar/tick | V→S等待/tick | 此后MTE3等待/tick |
|---|---|---|---:|---:|---:|---:|
| 0 | B | Scalar前 | 43/43 | 19 | 89 | 182 |
| 1 | C | Scalar前 | 41/41 | 19 | 100 | 173 |
| 2 | B | Scalar后 | 39/39 | 20 | 78 | 179 |
| 3 | C | Scalar后 | 37/37 | 20 | 73 | 185 |
| 4 | B | Scalar前 | 33/33 | 19 | 81 | 174 |
| 5 | C | Scalar前 | 33/33 | 19 | 107 | 37 |
| 6 | B | Scalar后 | 31/31 | 21 | 77 | 153 |
| 7 | C | Scalar后 | 29/29 | 20 | 77 | 176 |

后移等待四次都满足：B/C在A返回后进入同一物理核，Scalar结束后才调用真实等待，
V→S仍需73～78 tick，明确支撑前驱Vector计算与后继Scalar重叠。
V→S等待的min/mean/max：前置81/94.25/107 tick，后置73/76.25/78 tick；
Scalar对应19/19/19与20/20.25/21 tick。
这同时验证了过程010补充的context写回和单AIV/无predicate限制。
当前仍是手写独立入口与真实PyPTO动态调度的联合原型；不宣称DSL/PTOAS路径或整体加速已通过。

新增Vector标记后补查末尾无后继的收尾：`test_async_aiv_return.py` 使用同一128×256生产者，
分别在入口等待或在executor等待，四次换输入全部精确一致。
CPU `vector_marker_terminal_compile` 与单卡任务 `task_20261010_001857_233220211273`
均 **1 passed**，物理卡0已释放。两次异步返回到真实完成分别373/229 tick，
其中实际V→S等待均187 tick；两次入口内等待后的executor等待均0 tick。
产物为私有 `artifacts/vector_marker_terminal_device/`、同名XML和`.task.log`。
这补充验证了新增事件不会因缺少后继而悬挂；不重复展开形状测试矩阵。

LLVM/MLIR源码构建后期改为只构建PTOAS链接实际缺失的库，复用已完成对象；
记录为 `artifacts/llvm_required_build.log`。PTOAS自己的IR/Transforms对象构建仍在继续。

## 14. 过程 012：纯 PyPTO DSL 的动态同核重叠通过

2026-10-10，私有LLVM/MLIR依赖与PTOAS CLI完整构建结束。
PTOAS基于`f5eff3ee`，构建记录`artifacts/ptoas_full_build_v1.log`，最后为`[225/225] Linking CXX executable bin/ptoas`。
私有`env.sh`现在设置`PTOAS_ROOT="$SCALAR_OVERLAP_ROOT/ptoas-build"`；公共环境和另一会话工具链未替换。

### 编译链实际修复与合同检查

新增纯DSL用例位于私有PyPTO `tests/st/runtime/scheduling/test_dynamic_scalar_prefix_dsl.py`。
A用`async_aiv_return`和显式OUT trace声明异步出口；B/C用`scalar_prefix`及
`pl.system.wait_predecessor()`分开安全Scalar前缀和数据/Vector访问。
没有`external_source`，没有合编A+B/A+C，也没有后处理删除生成C++的barrier。

首次完整编译`dsl_compile_v1`失败于`async_return_trace`不再指向当前参数。
逐pass定位为`InitMemRef`重建Tensor参数但漏更新函数属性；在私有PyPTO
`src/ir/transforms/init_memref.cpp`用现有`MapAttrExprs`同步重映射属性引用。
新增定向回归先复现`UseAfterDefCheck`，修复后与既有load/add/store测试共 **2 passed**；
重安装记录`artifacts/init_memref_attr_install_v1.log`。
这是引用身份修复，没有按参数名字猜测绑定，也没有跳过校验。

CPU `dsl_compile_v2` **1 passed**，包含三份独立的实际设备ELF入口。
检查生成代码确认：A内部MTE2/Vector/MTE3依赖仍在，末尾只提交EVENT_ID7；
B/C第一次wait前没有Vector/atomic状态重置，初始化在wait成功后执行。
另有两项CPU负例均按预期拒绝：在Scalar前缀插入GM写入、在异步A中插入未支持的Scalar GM写入。
证据为`artifacts/contract_rejections_v1/results.json`及对应`.pto`、诊断输出。

### DSL 单卡重叠证据

任务`task_20261010_090008_334195597`，物理卡0，**1 passed**，卡已释放。
同一注册句柄在运行时选择B/C/B/C；每次换输入、毒化mid/out及trace，
32,768个mid/out元素、输入保护、Scalar整数结果全部精确一致。
数据路径A为tile加1共1024轮，B/C分别加2/5；Scalar递推64轮。

| 序号 | 后继 | A/B物理核 | Scalar/tick | Scalar之后V→S等待/tick | 随后MTE3等待/tick |
|---|---|---:|---:|---:|---:|
| 0 | B | 58/58 | 19 | 331 | 181 |
| 1 | C | 54/54 | 19 | 313 | 167 |
| 2 | B | 52/52 | 19 | 331 | 182 |
| 3 | C | 48/48 | 19 | 324 | 193 |

四次都满足A返回后才进入B/C，Scalar结束后才执行实际V→S等待，且Vector尚未完成。
V→S等待min/mean/max为 **313/324.75/331 tick**；Scalar均19 tick。
这证明编译器自动生成入口的同核Scalar/Vector重叠；等待较长不等于性能提升，
没有与手写入口的不同指令序列直接作性能A/B，也不外推CSA收益。

证据：私有`artifacts/dsl_device_v1/dynamic_scalar_prefix_dsl_results.json`、
`dsl_device_v1.xml`、`dsl_device_v1.task.log`，原始`.pto`、生成C++和各自二进制同目录保存。

### 无后继、普通前驱及恢复交接

DSL用例补充普通完整同步A，并由同一图的运行时参数选择
“仅异步A → 普通A接B → 普通A接C → 异步A接B”四次调用。
没有后继时检查B输出仍为NaN、B trace仍为-1；普通A不发布异步trace。
CPU `dsl_fallback_compile_v1` **1 passed**；任务`task_20261010_090221_1213099353`
物理卡0，**1 passed**，卡已释放，所有应写输出与Scalar结果精确一致。

- 无后继A：入口返回到实际完成632 tick，V→S实际等待433 tick，之后MTE3等待194 tick。
- 普通A接B/C：wait边界各1 tick且返回ready=1，输出正确，无异步trace。
- 再次异步A接B：44/44同核，Scalar19 tick之后Vector实际等待332 tick，未受前面回退调用污染。

证据：私有`artifacts/dsl_fallback_device_v1/fallback/dynamic_scalar_prefix_dsl_results.json`
与同级根目录下的XML、任务日志。只补当前能力边界，没有展开形状或全模型矩阵。
当前通过范围仍是TMR、单AIV、唯一前驱；多block、AIC/Cube、HBG、graph replay和公平性能对照待推进。

## 15. 过程 013：多 block、尾块与整项 task 完成依赖

2026-10-10继续按基础能力扩展，未接入CSA。原单AIV协议的关键不是把单block限制直接去掉，
而是保留“所有前驱block已发布”和“整项前驱task真实完成”两条不同门禁。

### 实现范围与完成合同

私有Simpler的`scheduler_dispatch.cpp`允许多block AIV后继选取正在运行其唯一前驱的物理核。
选取的是当前可接续的前驱block，不要求前后逻辑block编号相同，也没有静态编译配对。
沿用`propagate_dispatch_fanin`已有的`published_block_count >= logical_block_num`门禁，
在接续选择/上下文准备处再检查一次，防止后继占满核心而使尚未派发的前驱block无法运行。

executor仍先等待本核前驱Vector及MTE3的EVENT_ID7，再发布这个block的FIN。
后继Scalar已可先执行，但`wait_predecessor`仍等待AICPU对整个前驱task完成聚合后的doorbell；
不将局部block FIN视为所有前驱输出可读，不提前ACK后继。
本轮仍限定唯一WAIT前驱，带predicate或`sync_start`的后继保留普通执行；这些能力分别验证后再开启。

私有PyPTO包装层将`async_return_trace`扩为`[N,8]`，每个block按运行时block_idx写一条64字节记录。
`[8]`只记录block零，超出容量的block跳过诊断写入；算术与完成不受影响。
测试提供N+1行，额外一行保持-1以验证越界保护；没有用多个block并发覆盖同一条trace。
本轮PTOAS源码无新修改。runtime重编译记录为`artifacts/multiblock_runtime_build_v1.log`。

### 单卡用例与结果

私有PyPTO用例：`tests/st/runtime/scheduling/test_dynamic_scalar_prefix_spmd.py`。
两档block数为4、96；每个block为128×256 FP32，最后一个block只有115行有效数据，
张量末尾另外分配16行保护区。A各block分别做`1024 + (block_idx % 4) × 128`轮加1，
刻意错开完成时刻；B的block i读取A的block `(i+1)%N`，不能只等本核生产者。
B先做64轮运行时整数递推，再等待整个A，随后读取数据并加2或5。
同一注册句柄执行两次、换输入并重新毒化全部输出；没有展开其他形状矩阵。

CPU `multiblock_compile_v1` **2 passed**，实际生成producer/consumer两份独立设备二进制。
orchestration产物包含正确的block_num、A的allow_early_resolve和B的scalar_prefix。
任务`task_20261010_091746_84074625444`，**2 passed，0 skipped**，队列锁已释放。
**设备标注更正（过程015）**：队列分配2，但该复合命令未给pytest传设备参数，实际沿用默认0；
原“物理卡2”结论撤回。96-block正确选卡补测见过程015，不能将旧分配日志作为实际选卡证明。

| block数 | B加数 | A使用物理AIV数 | 已证明同核重叠组数 | B Scalar min/mean/max，tick | Scalar后Vector剩余等待 min/mean/max，tick |
|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 4 | 4 | 19 / 19.00 / 19 | 294 / 317.75 / 328 |
| 4 | 5 | 4 | 4 | 19 / 19.25 / 20 | 307 / 341.50 / 354 |
| 96 | 2 | 48 | 48 | 19 / 19.21 / 20 | 303 / 338.46 / 351 |
| 96 | 5 | 48 | 48 | 19 / 19.11 / 20 | 277 / 351.52 / 366 |

96 blocks使用48个物理AIV，证明经过了多轮派发；只统计trace确证的48组计算重叠，
不把其他正常派发block算作重叠。所有A/B有效元素精确一致，输入未变，
末尾13行padding、16行保护区、B旋转读取对应的padding及两份trace保护行均保持毒值。
所有B的等待返回时间均不早于所有A block的最大真实完成时间，Scalar整数结果也全部精确。
因此覆盖了整项task完成聚合及跨block可见性，没有仅凭局部数值通过就推断同步正确。

产物：私有`artifacts/multiblock_device_v1/blocks_4/multiblock_results.json`、
`blocks_96/multiblock_results.json`，根目录旁的`multiblock_device_v1.xml`、`.task.log`。
两份JSON的A trace字段沿用过程012；B为`core, scalar_start, scalar_end, after_wait,
scalar_result, addend, ready, source_block`。原始全部block记录和重叠匹配关系保留。

此轮证明多block动态同核Scalar/Vector重叠，尚未给出普通路径/前置等待/后置等待的公平性能A/B。
下一步继续复杂依赖和核类型能力的小用例，未验证的组合保持普通执行，不提前套用到CSA。

## 16. 过程 014：多前驱与 fan-out 的动态接续

2026-10-10继续在私有环境扩展TMR AIV demo，未接入CSA，也未修改其他会话的算子或环境。

### 实现与完成边界

私有Simpler仅新增修改`scheduler_dispatch.cpp`，移除“恰好一个fanin”的资格限制。
通过现有`for_each_fanin_slot_state`遍历inline/spill边，只从带`DEP_WAIT`的真实前驱中
选择本核正在运行的task；RETAIN-only及规约删除的WAIT边不构成接续资格。
除核类型和全block已发布检查，还核对当前前驱的completion fan-out中确实包含该后继。
原因是ordering-only边在wiring后不再持有producer引用，仅比较slot地址可能误认复用后的task；
本核的running slot由当前调度线程负责，因此遍历期间这个前驱不会完成并被复用。
这是代码层面的代次保护，尚不替代后续长链/小窗口复用验证。

多前驱的早派发仍沿用`dispatch_fanin`门禁：每个WAIT前驱须已全部可启动或已完成。
后继只与其中一个本核前驱重叠Scalar，边界中先等本核Vector/MTE3并发布真实FIN，
再等原有AICPU对**全部WAIT前驱、全部block**完成聚合后发来的doorbell。
没有将本核完成偷换成全部输入可读，没有改变ACK/FIN或依赖释放协议。
带`sync_start`、predicate或运行时profiling的组合仍走普通执行。

### 单卡图与结果

私有PyPTO新增`tests/st/runtime/scheduling/test_dynamic_scalar_prefix_dependencies.py`。
图为`A、B → C、D`：A/B分别4 blocks，两个独立输入与输出；C/D也各4 blocks，
都读取A的`(i+1)%4`及B的`(i+2)%4`块，再分别加2/5。producer与consumer为独立设备入口，
同一个入口可供多个动态task使用，没有预编译A+C/A+D组合。

沿用128×256、115行有效尾块和16行保护区。C/D安全Scalar均为64轮运行时整数递推。
同一注册句柄换输入执行两次，并交换A/B的基础Vector迭代数4096/8192，每block再加`i×128`，
刻意让另一路输入更晚完成。所有输出先毒化，检查有效元素、输入未变、padding、保护区及trace。

CPU完整编译 **1 passed**；私有runtime构建成功。
任务`task_20261010_092940_154350625597`，物理卡0，**2 passed，0 skipped**，卡已释放：
新依赖图1项及既有4-block单前驱回归1项。

| A/B基础迭代数 | 确证重叠组数 | 实际接续 | Scalar min/mean/max，tick | Scalar后Vector剩余等待 min/mean/max，tick | Scalar结束时另一路尚未完成 |
| --- | ---: | --- | --- | --- | ---: |
| 4096 / 8192 | 8 | A→D 4组，B→C 4组 | 19 / 19.12 / 20 | 276 / 324.75 / 358 | 4组 |
| 8192 / 4096 | 8 | A→D 4组，B→C 4组 | 19 / 19.12 / 20 | 304 / 346.38 / 366 | 4组 |

C/D全部8个block的等待返回都晚于A/B全部8个block的最大真实完成时间；
返回时间减该完成时间的min/mean/max分别为83/99.88/117、55/77.00/95 tick。
数据与Scalar结果精确一致，所有保护检查通过。两条fan-out均取得实际重叠证据，
并验证了“本核前驱已完成，但另一路仍未完成”时不会提前读输入。
原始tick仅作时序诊断，不是微秒或核周期，也不是公平端到端性能比较。

产物位于私有环境`artifacts/`：

- `fanin_fanout_runtime_build_v1.log`、`fanin_fanout_compile_v1.log/.xml`。
- `fanin_fanout_device_v1/fanin_fanout/dependency_results.json`：所有A/B/C/D trace、重叠匹配、字段名。
- `fanin_fanout_device_v1/blocks_4/multiblock_results.json`：受影响的单前驱回归。
- `fanin_fanout_device_v1.xml`、`fanin_fanout_device_v1.task.log`：退出状态及卡释放记录。

复现沿用私有`env.sh`与`task-submit --device auto --max-time 240`，pytest选上述新文件和
`test_dynamic_scalar_prefix_spmd.py -k "not 96" -n 1 --platform=a2a3 --save-kernels`。
只覆盖这次依赖逻辑的新增能力和一个受影响项，没有重跑既有全矩阵。

### 不同 block 数与 sync_start 回退

同一测试文件追加`sync_fallback`参数：A/B仍各4 blocks，C/D各2 blocks并声明`sync_start=True`。
不改变runtime，只验证现有排除条件确实保留普通同步启动；没有宣称sync_start已支持Scalar提前执行。
首次CPU编译未通过，因为`pl.spmd(..., sync_start=闭包变量)`违反当前DSL仅接受布尔字面量的要求；
用编译期分支选择带`sync_start=True`或不带该参数的调用，未修改parser。
`sync_fallback_compile_v2` **1 passed**，生成代码确认后继block_num=2及require_sync_start=true。

任务`task_20261010_093502_181053721032`，物理卡0，**1 passed，0 skipped**，卡已释放。
两次换输入并交换A/B工作量，输出、未用输出块、尾块、保护区及未用trace行全部精确。
两次均0组提前重叠，C/D所有Scalar起点都晚于A/B所有block的最大真实完成时间：

| A/B基础迭代数 | Scalar启动距全部前驱完成 min/mean/max，tick | 普通wait边界 min/mean/max，tick |
| --- | --- | --- |
| 4096 / 8192 | 178 / 189.50 / 202 | 0 / 2.25 / 6 |
| 8192 / 4096 | 79 / 97.25 / 114 | 0 / 2.75 / 9 |

产物：私有`artifacts/sync_fallback_device_v1/sync_fallback/dependency_results.json`，
以及`sync_fallback_device_v1.xml`和`.task.log`。复现只选新文件`-k sync_fallback`，其他参数沿用上述命令。
此结果仅验证不提前启动及正确回退，不用Scalar起点差值重新定义原有sync_start的核间同步精度。

本轮实际变更是私有Simpler调度资格、PyPTO小用例和说明文档；PTOAS没有新增修改。
下一步补任务/参数槽和事件的复用、短长链与replay，再单独建立AIC/Cube与HBG证据；
公平端到端性能和CSA接入仍在后续阶段。

## 17. 过程 015：连续交接、窗口复用与设备选择更正

2026-10-10仍在私有环境完成基础AIV动态链，没有接入CSA或改公共安装包。
本轮增加“一个task既是带安全Scalar前缀的消费者，又是异步返回的生产者”的组合合同。
所有任务实例复用一个独立编译入口；AICPU按运行时length提交task，没有静态合组。

### 新增修改与发现的问题

1. PTOAS `ptoas_pipeline.cpp`允许通过前缀合同检查的计时/核号/wait调用进入异步AIV函数。
   Scalar GM写入只允许在顶层wait之后、任何tile操作之前；只可嵌套非重复的if区域，
   避免循环第二轮在流水已运行时再次写入。校验通过后为该写入设置内部标记。
2. PTOAS `ScalarPtr.cpp`在标记写入处立即做cache刷写和DSB，不再为这些写入延迟生成函数尾部
   `PIPE_ALL + dcci + dsb`。这是编译器lowering修改，没有用实验脚本删生成C++的barrier。
   普通Scalar写入维持既有策略；异步函数中不满足边界的写入继续拒绝。
3. PyPTO新增`test_dynamic_scalar_prefix_chain.py`，node同时声明两种属性。
   首版用了manual_scope却没有显式传前驱TaskId，task事实上并行执行，第二阶段读到NaN；
   已用`pl.submit(..., deps=[previous])`显式串边，不把缺失依赖归咎于handoff协议。
4. Simpler的诊断trace改用raw COREID，新增onboard/sim同名getter。
   调度内部的masked物理核号保持原用途。卡5上raw值带die位，例如32827与masked值59相差32768；
   旧比较把同一核误判为不同核。修复只统一诊断身份，不修改选核规则。

首次CPU `chain_compile_v1`因未支持组合合同失败；扩展后v2编译通过。
设备结果又揭示Scalar写入的隐式尾部全流水刷写会让Vector剩余等待归零，
因此仅允许该操作还不够，必须同时调整其合法位置与刷写时刻。
最终`chain_compile_v3` **1 passed**，代码确认运行时for循环、显式TaskId边、
scalar_prefix/allow_early_resolve双属性；最后TSTORE之后仅执行异步尾部，没有上述额外全流水等待。

四个CPU反例在最终PTOAS构建下都按预期拒绝：等待前的不安全写入、普通异步生产者中的Scalar写入、
首次tile操作之后的写入、循环里的Scalar写入。产物`chain_contract_rejections_v2/results.json`，
输入PTO保存在v1同名目录。没有为拓展组合合同取消原有安全边界。

### 设备选择错误与旧记录更正

`chain_device_v1`任务`task_20261010_094453_254329528151`，队列分配7，
但复合命令未把设备参数传给pytest，后者仍用默认0；在`halMemCtl`初始化返回EACCES=13，
未执行node。根因是队列对带`&&`的复合命令明确不自动追加`--device`，不是长链内核报错。

后续命令显式使用`--device "$TASK_DEVICE"`，两个受影响用例在设备执行前检查队列卡号等于RunConfig，
JSON也记录实际device_id。**不能把拿到队列锁视为程序已正确选卡。**
回查过程013的任务命令也缺此参数，撤回其“物理卡2”标注；其当时默认device_id为0。
过程011的128×128负例任务同样未显式传卡号，只保留失败结果，不据分配卡1推断实际运行卡。
旧Native参考脚本显式设置visible devices、旧独立two_task基线显式传`$TASK_DEVICE`，不受此问题影响。

正确选卡的v2任务`task_20261010_094817_274793512416`在卡5执行，两个用例因raw/masked核号不一致失败；
96-block的全部数据检查已通过，但不能按当时的核号比较认定重叠通过。
修正后v3任务`task_20261010_095150_298433212395`在卡5执行，**96-block通过、长链失败**：
前者补齐了正确选卡证据，后者暴露上述缺失显式依赖的问题。
所有这些任务均已退出并释放队列锁，没有设备复位或终止他人的任务。

96-block补测两次均48组同核重叠、48个物理AIV、多轮派发，数据/padding/保护区均精确：

| B加数 | Scalar min/mean/max，tick | Scalar后Vector剩余等待 min/mean/max，tick |
| ---: | --- | --- |
| 2 | 23 / 31.50 / 57 | 302 / 331.15 / 350 |
| 5 | 19 / 19.63 / 24 | 297 / 349.73 / 366 |

证据：私有`artifacts/chain_device_v3/blocks_96/multiblock_results.json`和同名前缀XML/任务日志。
原过程013数值保留为历史观测，当前多轮派发结论以这次正确选卡证据为准。

### 最终长链单卡结果

任务`task_20261010_095812_342714219743`，队列和RunConfig均为卡5，**1 passed，0 skipped**，卡已释放。
同一注册句柄依次执行3、33、7个node，length为运行时参数；task window配置为8。
每node含64轮整数Scalar递推，再对128×256 FP32做标量偏置与1024轮Vector加1。
每轮换输入、重新毒化所有输出，检查所有阶段数据、Scalar状态、未使用阶段和16行保护区。

| 链长 | 确证相邻重叠边 | raw物理核号 | Scalar min/mean/max，tick | Scalar后前驱Vector剩余等待 min/mean/max，tick | 末尾node返回至真实完成，tick |
| ---: | ---: | ---: | --- | --- | ---: |
| 3 | 2 / 2 | 32810 | 19 / 19.50 / 20 | 315 / 323.00 / 331 | 475 |
| 33 | 32 / 32 | 32810 | 19 / 19.03 / 20 | 321 / 333.38 / 340 | 474 |
| 7 | 6 / 6 | 32806 | 19 / 19.17 / 20 | 318 / 328.67 / 336 | 476 |

40条相邻边全部证明同核Scalar/Vector计算重叠；各后继wait返回都晚于前驱真实完成，
所有阶段逐元素精确。33-task链超过8-slot窗口，连续复用task槽、双缓冲参数槽和EVENT_ID7；
长链后的7-task调用没有读到旧准备结果或留下旧trace。三个末尾node无后继，Vector实际等待均437 tick，
说明收尾仍等待真实计算完成。不同调用不要求使用同一个物理核。
这不等于寄存器task_id全范围回绕、其他独立分支的公平性或NPU graph replay已经验证。

最终产物均在私有`artifacts/`：

- `chain_ptoas_build_v2.log`、`chain_runtime_build_v1.log`、`chain_compile_v3.log/.xml`。
- `chain_device_v4/chain/chain_results.json`：三次调用全部trace及重叠边；
  `chain_trace_0/1/2.json`保存每次原始观测，成功或后续断言失败都可排查。
- `chain_device_v4.xml`、`chain_device_v4.task.log`：完整状态和选卡/释放日志。

复现从私有PyPTO目录提交，命令须保持单行：

```bash
task-submit --device 5 --max-time 240 'cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto && source ../env.sh && python -m pytest tests/st/runtime/scheduling/test_dynamic_scalar_prefix_chain.py -n 1 --platform=a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir="$SCALAR_OVERLAP_ROOT/artifacts/chain_recheck" --junitxml="$SCALAR_OVERLAP_ROOT/artifacts/chain_recheck.xml" -q'
```

等待上述返回的task ID，检查XML与JSON；仅提交返回task ID不算执行完成。
原始tick仍不换算微秒，也不作为端到端收益。下一步先补graph replay/必要的并行链边界，
再独立推进AIC/Cube和HBG小用例；公平性能与CSA接入仍在之后。

## 18. 过程 016：kernel 模式图捕获与重放通过

2026-10-10复用过程015的同一PrefixChain计算与动态TaskId依赖，在私有环境补齐
真实torch_npu NPUGraph capture/replay。没有修改生产CSA、公共环境或其他会话源码。
此过程不重写JIT前端：将已验证的Program IR按KernelABI编译为KernelArtifact，
走实际进程Worker、native enqueue、caller stream通知及graph replay路径。
这证明kernel执行与重放链路；不额外声称新用户侧JIT语法已经覆盖manual_scope/pl.submit。

### 版本固定与构建

私有PyPTO原来未启用torch_npu接口，先打开`PYPTO_BUILD_TORCH_NPU=ON`构建。
首次完整编译`capture_compile_v1`在设备编译前被拒绝：kernel模式要求匹配的干净Simpler SDK，
私有Simpler当时有未提交的运行时修改，`_runtime_revision`因此不能提供版本。
没有关闭版本检查，也没有伪造旧revision；把已单卡验证的9个Simpler文件固定为本地提交
`7078fc45f19d3799c02b83dd3ecc9c0afc9da7a9`（详细中文、Signed-off-by），
更新私有PyPTO的`SIMPLER_KERNEL_REVISION`，重建私有binding、TMR二进制和torch接口。
公共Simpler/PyPTO安装及分支未被推进。后续编辑期仍优先program模式，避免每次改动都重建kernel SDK。

接口首次安装命令因当前pip不支持`-C`而退出，改用CMAKE_ARGS/SKBUILD环境选项；
构建明确限制两路，Simpler的pip构建只选`_task_interface`，TMR用受限RuntimeBuilder单独构建，
没有触发全平台并行编译。最终`capture_compile_v2.xml` **1 passed**，完整设备二进制通过。

### 实际单卡验证

新增私有PyPTO `tests/st/runtime/kernel/test_scalar_prefix_capture.py`：

- 同一KernelArtifact/registration，只在捕获前初始化并预热一次；task window=8。
- 捕获两张图：运行时length=3/seed=7和length=33/seed=11；node独立编译，AICPU仍逐task动态派发。
- 每张图在PTO调用后紧接普通PyTorch的末阶段copy及add，用于检查caller stream完成顺序。
- 按短→长→短重放三次，分别换输入、把输出重新填NaN、trace填-1；核对全部阶段和保护区。
  重放仅调用`graph.replay()`，不重新执行Python enqueue或准备内核；第三次也不能复用第一次的旧结果。
- graph reset之后drain/close进程Worker，正常退出。

任务`task_20261010_101904_49575923354`，队列与程序均为设备7，**1 passed、0 skipped、exit=0**，卡已释放。

| replay / 链长 | 同核重叠边 | raw核号 | Scalar min/mean/max，tick | Scalar后前驱Vector剩余等待 min/mean/max，tick | 末尾返回至真实完成，tick |
| --- | ---: | ---: | --- | --- | ---: |
| 0 / 3 | 2 / 2 | 32833 | 19 / 19.50 / 20 | 323 / 330.50 / 338 | 474 |
| 1 / 33 | 32 / 32 | 32841 | 19 / 19.16 / 20 | 315 / 325.50 / 334 | 478 |
| 2 / 3 | 2 / 2 | 32807 | 19 / 19.00 / 19 | 321 / 323.50 / 326 | 475 |

36条相邻边全部观测到同核Scalar/Vector重叠。各stage浮点结果、Scalar整数状态、未使用行及保护区精确，
三次框架后继结果也精确，证明这些重放中的外层完成通知没有早于最后一个node的真实输出。
每次trace时钟向前推进，长短两图的参数保持各自捕获值，未出现旧trace/旧Scalar参数复用。
这里的tick仍是原始设备计数，不换算微秒，不据此宣称端到端加速。

产物在私有根`artifacts/`：`capture_simpler_install_v1.log`、`capture_runtime_build_v1.log`、
`capture_adapter_install_v3.log`、`capture_compile_v2.log/.xml`、`capture_device_v1.task.log/.xml`，
以及`capture_device_v1/capture/capture_results.json`和`replay_trace_0/1/2.json`。
`graph_adapter_install_v1/v2.log`、`capture_compile_v1.log/.xml`保留初始构建/版本失败证据。

```bash
task-submit --device auto --max-time 240 'cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto && source ../env.sh && python -m pytest tests/st/runtime/kernel/test_scalar_prefix_capture.py -n 1 --platform=a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir="$SCALAR_OVERLAP_ROOT/artifacts/capture_recheck" --junitxml="$SCALAR_OVERLAP_ROOT/artifacts/capture_recheck.xml" -q'
```

仍未覆盖任意并行图的公平性、task_id全范围回绕、sync_start重叠、AIC/Cube和HBG。
下一步先补长链中的独立分支推进，再推进AIC→AIC小matmul；公平性能与CSA仍在后续阶段。

## 19. 过程 017：长链中的独立分支可以继续推进

新增私有PyPTO `test_scalar_prefix_branch.py`，复用过程015完全相同的node IR，
仅换调度入口：33-task依赖链在提交stage=8时增加一个无前驱、使用独立数据的node。
window=8，继续显式TaskId串接主链；支线使用空deps，不能人为挂到链尾。
签名占位函数只用于解析，编译前以已经验证的PrefixChain node IR替换，不生成空计算二进制。
本轮没有新增Simpler或PTOAS实现变化，仍是私有Simpler `7078fc45f`。

`branch_compile_v1.xml`完整编译 **1 passed**。单卡任务
`task_20261010_102351_80709110225`，队列与实际设备均为6，**1 passed、0 skipped、exit=0**，卡已释放。
主链32条相邻边中31条发生同核Scalar/Vector交接，另一条正常调度；raw核号为32/34，
独立支线运行在raw核36。主链及支线全部数据、Scalar状态、保护区精确。
支线完成时主链仍有28个task未完成，证明本例中的连续交接没有阻塞独立可运行任务直至链尾。
31条重叠边的Scalar min/mean/max为19/19.23/20 tick，Scalar后前驱Vector剩余等待
323/336.26/345 tick。没有要求每条边都强制同核，普通回退仍有效。

这是一条独立支线的推进检查，不是所有核饱和、所有图形或所有调度线程的公平性证明。
证据：私有`artifacts/branch_device_v1/branch/branch_trace.json`及`branch_results.json`，
`branch_compile_v1.log/.xml`、`branch_device_v1.task.log/.xml`。
复现沿用过程016队列命令，测试文件换成`tests/st/runtime/scheduling/test_scalar_prefix_branch.py`，
并更换产物前缀；必须继续显式传队列卡号。
下一步在基础小matmul上单独扩展AIC的Cube/FIX异步完成合同，不直接接入CSA。

## 20. 过程 018：独立 AIC task 的 Scalar/Cube 重叠通过

继续使用私有环境，先做小matmul，没有接入CSA。本轮在Simpler `7078fc45f`之上增加未提交的
AIC扩展；PyPTO/PTOAS也为私有修改。只运行program模式，未把未固定的新SDK冒充kernel重放基线。
过程016的graph证据对应当时固定的AIV SDK；本轮仅追加当前executor的AIV program回归。

### 三仓修改及完成合同

- PyPTO：新增实验属性`async_aic_return`，只允许AIC函数；对应PTOAS Cube异步尾部。
  `scalar_prefix`扩展至单AIC/AIV，生产者自动带allow_early_resolve；其他runtime继续拒绝新协议。
  wrapper按核类型设置异步标记，prefix等待后才初始化atomic，Vector mask仅在AIV上初始化。
- Simpler：GlobalContext原异步字段改为`async_tail_kind`，0为普通、1为AIV、2为AIC，布局大小仍80字节。
  AIV分别消费V→S、MTE3→S事件；AIC分别消费M→S、FIX→S事件。
  等待计时在分支选择之后紧邻真正的wait，避免把读取类型字段算进剩余计算时间。
  调度资格要求前后task为同类单AIC或单AIV，保留真实WAIT边、全前驱完成及普通回退规则。
- PTOAS：增加`async_aic_event7`，通过MTE2→MTE1→M→FIX串接所有使用的流水，
  分别发出M→S与FIX→S的EVENT_ID7而不做Scalar等待；编译器为涉及的pipe pair保留event 7。
  AIC异步函数只开放load/move/extract/matmul/store和Scalar控制，AIV仍限于原load/add/store范围；
  不开放额外显式同步或其他引擎。共享前缀边界检查及Scalar写入限制保持有效。

三仓均完成私有构建，`aic_compile_v1.xml`完整设备代码编译 **1 passed**。
PTOAS公共lowering头触发91项增量构建，全程限制两路；没有占卡等待编译。
生成代码确认两个独立设备入口、AICPU的allow_early_resolve/scalar_prefix、
生产者最后TSTORE后的异步尾部，以及后继`clock→Scalar→clock→wait→读前驱数据`顺序。
普通模式的PIPE_ALL helper仍存在，不能仅grep文件中的PIPE_ALL字样判定异步路径有尾部全流水等待。

### 单卡小 matmul

新增私有PyPTO `tests/st/runtime/scheduling/test_dynamic_scalar_prefix_aic.py`：
生产者输入FP16 `[128,128] @ [128,256]`，初次matmul后再累加16次，输出FP32并保留16行保护区。
后继为另一个独立AIC入口，先执行64轮整数Scalar递推，wait之后读取前驱首元素，再执行自己的matmul。
用整数值FP16输入使CPU golden及Cube输出可逐元素精确比较；没有放宽容差掩盖功能问题。
同一注册句柄按运行时参数执行“无后继、有后继、有后继”三次，每次换输入、毒化输出和trace。
同时定向重跑既有AIV 3/33/7-task链，检查共用executor扩展没有破坏原路径。

任务`task_20261010_104658_21401368904`，队列/实际设备均为6，**2 passed、0 skipped、exit=0**，卡已释放。

| 场景 | A/B raw核号 | B Scalar，tick | B Scalar之后前驱Cube剩余等待，tick | 随后的FIX等待，tick | A函数返回至真实完成，tick |
| --- | --- | ---: | ---: | ---: | ---: |
| 无后继 | 21 / 无 | 无 | 无；executor直接等待Cube 498 | 192 | 691 |
| 后继，seed=8 | 19 / 19 | 20 | 370 | 177 | 676 |
| 后继，seed=9 | 19 / 19 | 20 | 417 | 39 | 538 |

两次均证明后继Scalar与前驱Cube计算同核重叠，不只是FIX写回重叠。
Scalar min/mean/max为20/20.00/20 tick，Scalar后Cube剩余等待370/393.50/417 tick，样本数为2。
全部matmul输出、后继所读的前驱值（分别-51、-102）、Scalar整数状态、保护区精确；
后继wait返回均晚于前驱M/FIX真实完成。无后继时executor确实排空Cube与FIX，未遗漏收尾。
AIV回归的40/40条相邻边也保持计算重叠，全部阶段结果及保护区通过：

| AIV链长 | 重叠边 | Scalar min/mean/max，tick | Scalar后Vector剩余等待 min/mean/max，tick |
| ---: | ---: | --- | --- |
| 3 | 2 / 2 | 19 / 19.50 / 20 | 328 / 331.00 / 334 |
| 33 | 32 / 32 | 19 / 19.09 / 20 | 318 / 330.03 / 336 |
| 7 | 6 / 6 | 19 / 19.00 / 19 | 324 / 330.33 / 333 |

原始tick不换算微秒，本例的长matmul工作用于证明机制，不作为CSA收益预测。
本轮没有AIC多block、多个Cube前驱、AIC组合长链、AIC graph replay或HBG的设备证据。

### 产物与复现

私有根`artifacts/`下：`aic_ptoas_build_v1.log`、`aic_runtime_build_v1.log`、
`aic_pypto_install_v1.log`、`aic_compile_v1.log/.xml`、`aic_device_v1.task.log/.xml`；
`aic_device_v1/aic/aic_results.json`及三次`aic_trace_*.json`，
`aic_device_v1/chain/chain_results.json`及三次长链trace。

```bash
task-submit --device auto --max-time 240 'cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto && source ../env.sh && python -m pytest tests/st/runtime/scheduling/test_dynamic_scalar_prefix_aic.py tests/st/runtime/scheduling/test_dynamic_scalar_prefix_chain.py -n 1 --platform=a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir="$SCALAR_OVERLAP_ROOT/artifacts/aic_recheck" --junitxml="$SCALAR_OVERLAP_ROOT/artifacts/aic_recheck.xml" -q'
```

三个新增用例的Ruff及三仓diff空白检查通过。此处原计划先推进HBG，
现按2026-10-10用户要求暂缓，下一步直接做TMR普通执行/等待前置/等待后移的公平计时，
具体候选需要的其余AIC能力再补。
动态AIV/AIC机制已分别有基础证明，不等于全部扩展、性能或CSA阶段完成。

## 21. 过程 019：聚焦 TMR，关闭诊断后做公平计时

2026-10-10 用户明确“先做 TMR 就可以”“HBG 先不管”。HBG 仅做了源码检查，没有修改，
当前不再作为 TMR 性能或 CSA 接入的前置项。继续使用私有环境，不动另一个会话的公共安装及算子文件。

### 21.1 去除正式计时的诊断开销

PyPTO 的 `async_return_trace` 改为可选：省略时 wrapper 不读时钟、不设置 trace 地址；
Simpler executor 在 trace 地址为零时不读重叠诊断时钟、不写诊断结果。
同核交接、前驱真实完成、等待全部 fan-in、ACK/FIN 及 slot 回收保持原合同。
专门的 wrapper UT **1 passed**；三个计时入口完整 CPU 编译通过，生成代码没有 `get_sys_cnt()`。

回归任务 `task_20261010_110655_338163520891`，队列/实际卡6，**1 passed、1 failed**，卡已释放。
AIV 3/33/7-task链仍为40/40相邻边重叠，全部数据及保护区精确。
失败的是 AIC 用例的“每次都观测到重叠”断言：17次matmul的两次后继均走普通路径，
实际矩阵、依赖读、Scalar状态和保护区检查全部通过。此次不能报为AIC重叠通过，
也不能据此说算术有错；已有动态交接是机会性的，前驱返回时没有合格pending就立即排空。

为给AIC调度准备留出更长时间，只把诊断用例的matmul累加由17次改为65次，不改交接协议。
完整CPU编译1 passed；任务 `task_20261010_111021_364898732167`，队列/实际卡6，
**1 passed、0 skipped、exit=0**，卡已释放。
两次同核接续raw核分别18、15；Scalar为19、21 tick，之后Cube仍需387、382 tick。
无后继收尾、全部矩阵/Scalar/保护区通过。这是扩大机制观测窗口，不是该改动能加速真实算子的证据。

私有Simpler固定为本地提交 `242654f89`（3文件，详细中文及Signed-off-by），
PyPTO私有SDK pin同步；重新构建私有binding、TMR与torch接口，保持kernel-mode干净SDK校验。
公共分支和安装未变化。系统自带clang-format 12不支持仓库BlockIndent配置，
该格式命令未完成；diff空白检查、相关Python Ruff检查通过。

证据都在私有根目录 `artifacts/`：
`tmr_timing_wrapper_ut_v1.log`、`tmr_timing_program_compile_v1/`、
`tmr_trace_optional_device_v1/`及同名前缀XML/任务日志、
`tmr_aic_window_device_v2/`及同名前缀XML/任务日志。

### 21.2 三组正式计时口径

新用例 `pypto/tests/st/runtime/kernel/test_scalar_prefix_timing.py`，仍从Program IR生成真实KernelArtifact：

- 同一artifact的三个独立AIV入口，由运行时mode选择，均为17-task链、window=8、单block，所有stage逐元素检查。
- 每task做相同整数递推与1024轮FP32 Vector add；Scalar递推分别64和1024轮，后者只用于观察机制窗口。
- A普通完整完成；B新合同但等待前置；C新合同、Scalar后等待。B递推依赖wait返回的1，避免被编译器提前。
- 三组生成的CoreTaskArgs均启用allow_early_resolve。普通组必须用 `pl.submit(..., allow_early_resolve=True)`，函数属性中的同名项不会生成该设置；CPU检查时已改正，未用错误版本作性能对照。
- 无trace、无profiler；同卡三组轮换，各30个warmup后的NPU event区间。每个区间包含一整次NPUGraph replay及TMR调用。
- 输出min/mean/max和全部样本；前后换输入/毒化输出，检查所有17个阶段和16行保护区。
- A采用同一实验runtime的普通路径，不代表未修改Simpler的历史SDK；C/B检验等待后移效果，C/A检验本次运行时内的总体变化。
- 正式不带trace的计时不会人为排除普通回退，其开销包含在统计内；人为递推不能推算CSA收益。

### 21.3 单卡正式结果

私有binding、TMR、PyPTO torch接口构建通过；kernel模式CPU编译 **1 passed**。
任务 `task_20261010_111513_392863712979`，队列/实际设备均为5，
**1 passed、0 skipped、exit=0**，卡已释放。全部17阶段、输入和16行保护区精确。

下表单位 **μs/一次完整17-task图重放**，各组30个样本，未除以task数；不是CSA耗时：

| 每task Scalar轮数 | 路径 | min | mean | max |
| --- | --- | ---: | ---: | ---: |
| 64 | A：普通完整完成 | 9681.58 | 9703.63 | 9713.48 |
| 64 | B：交接后先等待 | 9674.24 | 9695.10 | 9705.92 |
| 64 | C：Scalar后等待 | 9669.84 | 9691.28 | 9710.18 |
| 1024 | A：普通完整完成 | 9785.56 | 9793.23 | 9802.00 |
| 1024 | B：交接后先等待 | 9772.84 | 9785.56 | 9794.54 |
| 1024 | C：Scalar后等待 | 9667.52 | 9687.11 | 9698.40 |

- 64轮：C/A均值少12.36 μs（0.127%），C/B少3.82 μs（0.039%）。按相邻采样轮配对分别26/30、21/30轮更快；两组分布重叠，C的max比B高4.26 μs。只记录本轮小差异，不宣称短Scalar已稳定受益。
- 1024轮：C/A均值少106.12 μs（1.084%），C/B少98.46 μs（1.006%）。两种比较均30/30轮更快，C/B最小单轮差83.72 μs，min/max也降低，支持该人工工作量下等待后移带来净收益。
- B/A本轮均值也更短，说明新交接同时改变派发/入口衔接，不能把C/A全部归因于Scalar隐藏；C/B才是等待后移的主要对照。
- 此用例为放大可观察窗口使用长Vector段和人工Scalar递推，不是实际CSA task。下一步核对真实候选的安全Scalar前缀、剩余流水和同期调度，不能按1.08%直接估计CSA收益。
- 普通组也运行在实验SDK上；若后续声称生产路径净收益，还要与未修改SDK对照，避免普通任务的新增公共开销被共同基线掩盖。

产物：私有 `artifacts/tmr_timing_device_v1/timing/timing_results.json`（全部180个样本及汇总）、
`timing_samples.json`、生成的三个独立入口；同名前缀XML/任务日志。
构建记录 `tmr_timing_simpler_install_v1.log`、`tmr_timing_runtime_build_v2.log`、
`tmr_timing_pypto_install_v1.log`、`tmr_timing_compile_v1.log/.xml`。

复现命令（先source私有env并完成该目录的构建）：

```bash
task-submit --device auto --max-time 240 'cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/pypto && source ../env.sh && python -m pytest tests/st/runtime/kernel/test_scalar_prefix_timing.py -n 1 --platform=a2a3 --device "$TASK_DEVICE" --save-kernels --kernels-dir="$SCALAR_OVERLAP_ROOT/artifacts/tmr_timing_recheck" --junitxml="$SCALAR_OVERLAP_ROOT/artifacts/tmr_timing_recheck.xml" -q'
```

## 22. 过程 020：冻结真实 CSA 源码，首个 gate 任务对只通过功能

### 22.1 独立算子副本

在既有私有根下新增 `vllm-ascend/` detached worktree，冻结版本
`fc6d201b253f07d2dbc2b45508dd5b50f842fe07`，整包复制为
`vllm_ascend/ops/pypto/deepseek_v4_flash_csa_scalar_overlap_demo/`。
完整 `decode_csa_tp1_layer._get_dep_graph()` 已解析通过。
公共安装、生产算子和另一会话正在优化的独立 HC_pre 未修改。
私有 env 增加该源码目录及只读 vLLM core 包；生成本地忽略的 A3 `_build_info.py`。
未调用会切回公共 editable 安装的旧 activate 函数。

### 22.2 首个真实任务对

新增私有包 `scalar_gate_probe.py` 与
`tests/pypto_test/tmr_csa_gate_scalar_overlap_demo.py`：

- 复制实际 CSA 局部 hc_pre 的 `linear_reduce → split_pre_post` 算术，T=96，
  reduce 为6个block，每块16×32、依次累加4份；split 为12个block，每块8行。
- ordinary / async+prefix 都是独立编译入口，共用同一算术体；没有加人工 Scalar 递推或额外 Vector 工作。
- 后继只提前 block、tile数及worker数的 Scalar 计算；scale/base/inv 等 GM 读取全部留在 wait 后。
- 前置 CPU 编译中显式 `level=AIV` 与 InCore 的 CHIP_DIE 级别冲突，移除该冗余参数，让编译器按运算推断AIV。
- 只构建私有入口，TMR/Simpler/PyPTO/PTOAS 沿用过程019的版本。

任务 `task_20261010_113554_109022332617`，队列/实际卡8，**exit=0，卡已释放**。
两轮均重新换输入、毒化输出与保护区，普通/交接版本 raw、pre、post 逐bit一致；
四份累加与CPU精确，sigmoid参考在 rtol=1e-5、atol=1e-6 内，尾部保护区未写坏。

**但两轮均为0次同核交接、0次可证明的计算重叠。**
前驱在函数返回后仍有Vector尾部；所有后继前缀实际开始于前驱完成之后。
当前“返回时已有合格pending才交接”的机会式协议没有赶上该窗口，
不能把功能通过称为真实CSA重叠成功，更未测得完整CSA加速。

产物（相对私有根）：
`artifacts/csa_frozen_dep_graph_v3.log`、`artifacts/csa_gate_compile_v4.log`、
`artifacts/csa_gate_compile_v4/`、`artifacts/csa_gate_device_v1/results.json`、
`artifacts/csa_gate_device_v1.task.log`。
失败的环境导入与level声明记录保留在同目录的v1/v2/v3日志。

### 22.3 后续候选边界

下一候选为真实 `qr_hadamard_quant → indexer_head_coefficients`：保留48-worker量化、
FP16系数舍入及二维对角写出，仅在私有包提取独立入口。
先补PTOAS对此Vector量化运算集合的异步尾部支持，再做单卡功能/时序；
首个探针把weights作为已准备输入，因此不代表完整CSA的双前驱调度。
该候选的后续结果见过程021；不扩大到七档或16卡。

## 23. 过程 021：支持实际 Indexer 量化指令，确认真实任务的交接机会尚未形成

### 23.1 修改与编译

仅修改私有副本：

- vLLM Ascend 冻结包新增 `scalar_indexer_probe.py`，测试入口为
  `tests/pypto_test/tmr_csa_indexer_scalar_overlap_demo.py`。
- 提取真实 `qr_hadamard_quant → indexer_head_coefficients`，T=96，64个head、head_dim=128；
  量化保持48-worker步长，每个worker两轮64×128，系数使用原有双query分组、48-worker与FP16对角矩阵写出。
- 同一注册句柄用运行时mode在普通/异步两组独立二进制之间选择，不生成合组入口。
  weights作为预加载输入，不代表完整CSA的第二个设备前驱。
- GM reshape按原实现留在调度侧；把reshape移进incore会被降成tile，显式load因此报类型错误。
  JIT元信息推断还要求先用局部量接收 `pl.tensor.dim(scales, 0)`，再用它构造reshape形状。
  这些编译入口问题在CPU阶段解决，未占卡反复调试。
- PTOAS `tools/ptoas/ptoas_pipeline.cpp` 的异步AIV允许集合增加本任务实际所需的
  `tmuls/texpands/textract/tneg/tmax/trowmax/tdiv/trecip/trowexpandmul/tcvt`。
  新增op也计入“已开始tile工作”的判定，防止放宽Scalar store的位置限制。
  各op的内部同步保留，查阅A3 PTO-ISA后未发现这些路径使用交接保留的EVENT_ID7；未修改PTO-ISA。
- 异步尾部仍只join MTE2/V/MTE3，由executor完成最终V/MTE3等待；没有删运算内部同步。
  两组生成代码的运算序列相同，区别为最终barrier-all或async event7及显式诊断。
- Simpler沿用 `242654f89`，未改其派发协议；HBG未改。

PTOAS重建通过；完整PyPTO→PTOAS→独立AIV二进制编译通过。
两份已有的非法合同输入仍被拒绝：wait前不安全操作、普通异步producer中的Scalar store，
均exit=1并匹配对应诊断。Ruff与diff空白检查通过。
编译器诊断仍报告“64字节步长、1个实例共享一条64字节cacheline”的告警；
此探针每实例8×INT64且运行时断言GM首址64字节对齐，保留该告警记录，没有修改分析器。

### 23.2 单卡结果：功能正确，0次同核交接

任务 `task_20261010_115351_19908766867`，队列/实际卡6，**exit=0，卡已释放**。
T=96，两轮更换accumulator及weights；每轮分别执行ordinary和async+prefix。

- 两组INT8量化、FP32 scale和FP16系数矩阵逐bit一致。
- 测试输入使用唯一行最大值、避开INT8舍入中点，INT8结果与CPU精确；
  FP32 scale在rtol=1e-5、atol=1e-7内，FP16系数按同一舍入顺序精确一致。
- INT8/FP32/FP16输出保护区及trace保护行未修改，后继wait之后的trace晚于整个producer完成。
- **两轮均0次同核交接、0次可证明的计算重叠**。退出0只证明本轮功能通过。
- 没有加人工Scalar递推或额外Vector计算；未做该任务对的无诊断正式性能比较。

以下都是原始device timer **tick**，每轮48个producer/consumer block；不是μs。
“Vector剩余等待”从executor开始等待到V→S事件返回，是本次观测的剩余窗口，不能当成总Vector执行时间。

| 轮次 | producer函数体 min/mean/max | 返回后Vector剩余等待 min/mean/max | consumer函数内Scalar前缀 min/mean/max | 首个consumer前缀距最后producer真实完成 |
| ---: | --- | --- | --- | ---: |
| 0 | 325 / 344.33 / 374 | 0 / 19.29 / 45 | 0 / 16.25 / 68 | 606 |
| 1 | 356 / 456.33 / 513 | 7 / 18.60 / 41 | 0 / 0.31 / 11 | 440 |

consumer前缀计时不包含wrapper参数展开；它只包含该函数实际可提前的block及循环上界计算。
两轮全部producer的真实完成都早于首个consumer前缀；当前协议没能把后继准备到前驱返回前。
因此不能将失败归为算术问题，也不能靠本轮称量化任务已经获得并发或性能收益。

### 23.3 源码核对：下一步应解决派发准备时机

TMR当前机会式路径有以下条件，且各条件承担实际的依赖/槽位保护，不能简单删除：

1. producer全部block已发布，consumer确实有WAIT依赖，避免未启动的producer被后继占核饿死。
2. consumer只占本线程所管、已有运行producer且可用的pending槽；普通发布会设置
   `pending_occupied`，ACK/FIN处理后才释放后续发布容量。
3. `try_early_dispatch` 在普通ready派发之后执行；存在ready任务时先让正常任务推进。
4. AICore异步函数返回时只查看一次 `DATA_MAIN_BASE`：必须已有匹配前驱身份的
   gated scalar-prefix consumer；否则立即排空，不在设备上等待后继到来。

本轮trace证明“未赶上交接”，但未采AICPU分阶段时钟，尚不能把606/440 tick全部归因于某一条条件。
下一步优先审查ACK处理、consumer候选传播和参数预备的时机；保留真实完成、全部fan-in、slot代次约束。
若需扩展预备pending而不提前覆盖doorbell，必须先在小图证明新协议，再回到这个真实任务对。
不以增加无用Vector轮数或丢弃同步作为“真实CSA受益”的替代。

### 23.4 产物

私有根 `artifacts/`：
`csa_indexer_ptoas_build_v1.log`、`csa_indexer_compile_v1～v5.log`、
`csa_indexer_compile_v5/build/`、`csa_indexer_contract_rejections_v1/results.json`、
`csa_indexer_device_v1/results.json`、`csa_indexer_device_v1.task.log`。
完整block trace、运行时模式和基准源码版本保留在results.json。

复现（私有env和工具链已构建）：

```bash
task-submit --device auto --max-time 180 'cd /data/pyptouser/qinchuanyu/pto-eager/.worktrees/scalar-overlap-demo-20261009/vllm-ascend && source ../env.sh && python tests/pypto_test/tmr_csa_indexer_scalar_overlap_demo.py --tokens 96 --device "$TASK_DEVICE" --output "$SCALAR_OVERLAP_ROOT/artifacts/csa_indexer_recheck"'
```

## 24. 过程 022：ACK后即时预派发使真实Indexer能够交接，入口进一步去冗余

### 24.1 定位与修改

沿用过程021的真实T=96、48-worker量化/系数探针，不改变算术、分块和测试输入规则。
源码核对发现普通发布占用pending容量，ACK处理才释放该容量；原实现要到本轮后部才尝试early dispatch。
只有“检查前驱返回时后继已经在pending槽”的机会式接续，因此这些延迟对短任务直接决定是否能交接。

私有Simpler的4个TMR文件修改（相对 `src/a2a3/runtime/tensormap_and_ringbuffer/`）：

- `runtime/scheduler/scheduler_completion.cpp`：ACK确实释放pending槽后调用即时接续尝试。
- `runtime/scheduler/scheduler_dispatch.cpp`：新增 `try_scalar_prefix_on_ack`，一次只在
  该物理核预派发一个匹配的Scalar-prefix消费者；保留真实WAIT边、同类核、全部producer block已发布、
  tagged task代次、predicate/sync_start排除，以及normal ready/dummy/sync/MIX优先级。
  非匹配候选放回队列，未成功预派发的任务仍由正常完成路径释放。
- 同文件 `build_payload` 为Scalar-prefix消费者在AICPU提前填好参数表，但保留非零src_payload和原有gate。
  这只移动描述符准备，不提前宣布数据就绪。
- `aicore/aicore_executor.cpp`：缓存已验证的后继token，进入同一代次时不再次整缓存失效；
  没有ACK之前slot不能复用。已由CPU准备的Scalar-prefix参数表也不再由AICore重建。
  前驱V/MTE3或M/FIX完成、后继全fan-in gate、FIN/ACK顺序保持原合同。
- `runtime/scheduler/scheduler_context.h` 增加内部helper声明，未改变GlobalContext/DispatchPayload布局。
- 交接已准备、后继尚未进入等待点时若收到退出信号，先排空仍归前驱所有的完成事件再确认退出。
  此竞争分支尚未做强制故障注入，不把正常close当作该竞争已覆盖。

没有改变HBG或公共安装，也未把本轮改动上线生产CSA。

### 24.2 同核交接结果与边界

原始tick，只用来诊断本核时序，**不是μs或正式性能比较**。
“入口间隔”是producer函数返回到consumer函数内前缀开始，含executor交接及wrapper展开；
consumer计时不覆盖整个wrapper。各轮仍检查普通/异步输出逐bit一致、CPU参考和保护区。

| TMR版本 | 任务/实际卡 | 两轮同核交接数 | 第一轮入口间隔 min/mean/max | 第二轮入口间隔 min/mean/max | 可证明的Vector计算重叠 |
| --- | --- | --- | --- | --- | --- |
| 原版本242654f89 | `task_20261010_115351_19908766867` / 6 | 0 / 0 | 无 | 无 | 0 |
| ACK释放pending时立即预派发 | `task_20261010_120723_253107030889` / 6 | 3 / 18 | 60 / 70.00 / 81 | 55 / 68.00 / 82 | 0 |
| 再加参数预填及去除重复DCCI | `task_20261010_121245_272761723638` / 10 | 14 / 13 | 41 / 50.07 / 72 | 35 / 42.77 / 58 | 0 |

两次新任务均 **exit=0、卡已释放**，真实探针全部功能/参考/保护区检查通过。
最终版本首个consumer前缀分别早于整个producer task真实完成284/262 tick，
但匹配的同核producer在后继wait时V→S剩余等待都为0。
因此已证明真实算子任务之间的动态同核交接，**尚未证明其函数内Scalar与前驱Vector计算重叠或端到端净收益**。
也不能据此排除wrapper的一部分Scalar曾与Vector重叠；当前时钟位置不足以证明这一点。
两候选使用不同物理卡、样本数仅2轮且带诊断，不能把入口间隔变化直接当作正式收益。

### 24.3 受影响路径回归

即时ACK候选同卡任务包含3项回归，**3 passed**：AIV长链、双前驱双后继、sync_start普通回退。
最终版本同卡任务包含4项回归，**4 passed**，并新增AIC检查：

- window=8、3/33/7-task链全部40条相邻边保持同核Scalar/Vector计算重叠，全部阶段与保护区精确。
- 双前驱/双后继两轮各8条重叠边；后继真正读输入之前均晚于全部WAIT前驱完成。
- sync_start两轮均按普通路径等待整个producer集合，未绕过原有同步约束。
- AIC的无后继收尾通过；两次后继Scalar均19 tick，之后同核Cube仍等待422/438 tick，
  矩阵结果、依赖读取、Scalar整数状态和保护区精确。
- 没有用这些人工放大窗口的小用例替代真实Indexer的负结果；本轮未重复七档、16卡或正式公平计时。

当前helper仍在私有prototype中；普通任务公共开销及未修改SDK对照尚未测量，
不能根据异步任务命中次数直接宣称可以默认生产启用。

### 24.4 产物与下一步

私有Simpler已保存为 `b39d9e7c70b66da1221f1ec044979257842e0af2`（4文件，中文详细说明及Signed-off-by）；未push。
PyPTO私有SDK pin随之更新，公共分支/安装不变。首次commit因私有worktree无作者配置失败，
使用既有提交的nalinaly身份仅在本次命令指定，未改全局Git配置。
私有Simpler/PyPTO重装最终均exit=0（`tmr_fast_entry_*_install_v2.log`）；
新pin下kernel-artifact CPU完整编译/加载检查 `test_scalar_prefix_timing.py --codegen-only`
为 **1 passed**，证据 `tmr_fast_entry_kernel_compile_v1.log/.xml`。这一步不占用NPU，
不算新增性能结果。`install_v1`对应提交失败后误沿用的旧pin，不作为新版本SDK的证据。

私有根 `artifacts/` 下：
`tmr_ack_prefix_runtime_build_v1.log`、`tmr_ack_prefix_runtime_build_v2.log`、
`tmr_ack_prefix_v1.patch`、`tmr_fast_entry_v2.patch`（两个候选的准确源码快照）；
`tmr_ack_device_v1.task.log`、`csa_indexer_ack_device_v2/results.json`、`tmr_ack_regression_v1/`及XML；
`tmr_fast_entry_device_v2.task.log`、`csa_indexer_fast_entry_device_v3/results.json`、
`tmr_fast_entry_regression_v2/`及XML。

此后不继续靠增加量化循环来扩大窗口。当前实现只允许有真实WAIT边的同核后继，
这是原型的保护性限制，不是用户目标要求只覆盖数据依赖任务。
下一步先用小图扩展“已满足所有输入依赖的独立后继”接续，再回到CSA连续Cube任务：
已ready的后继只需等本核前驱释放流水资源，有WAIT边的消费者仍须等全部fan-in。
必须继续维护真实完成、独立task身份与slot代次，不改成预编译A+B组合入口。

## 25. 过程 023：先在小图扩展已ready的独立后继

目的：普通ready队列里的独立后继已经满足自己的全部输入依赖，
也应允许其安全Scalar前缀与同核前项任务剩余Vector/Cube计算重叠。
这与过程022的真实WAIT消费者是不同的派发情形，不能给ready任务虚造数据边或等待一个不会到来的early doorbell。

修改仅限私有Simpler的TMR scheduler/executor：ready后继由普通队列派发，
只有同类AIV/AIC、无predicate/sync_start且声明checked Scalar-prefix时才允许接续；
在wait边界排空本核旧任务的计算和写回，再发布旧FIN、新ACK。
已有gated消费者仍要求真实WAIT边，并等待全fan-in；不放宽编译器的前缀访问限制。
HBG继续暂缓。

新增私有PyPTO用例 `tests/st/runtime/scheduling/test_dynamic_scalar_prefix_ready.py`：
前后输入、输出、trace独立，共享权重只读；先按AIV 48 blocks、AIC 24 blocks覆盖一个核组。
各block工作量错开，用来检查独立后继无需等整个前项任务结束。
检查FP32输出、Scalar状态、16行保护区、同核时序与剩余计算等待，重复两次更换输入；
计时为原始tick诊断，不是性能收益测量。

准备阶段的CPU编译错误已单独保留：v1的函数类型需literal AST；v2的func_attr键需字面量；
v3同一条件函数两分支都会先做类型检查；v4的闭包bool在func_attr中成为Expr且AIV tile超出UB。
用标准InCore自动定核、decorator的已求值attrs、同一FP16输入dtype及AIV 64-row tile修正，
不为用例修改parser或调高UB限制。以上均发生在CPU，没有占用NPU。
v5完整CPU编译 **2 passed**。decorator attrs目前有弃用提示；这里仅用其已求值bool入口，
trace张量仍由函数内func_attr绑定，未修改公共API。

### 25.1 实现边界

- `scheduler_dispatch.cpp::prepare_subtask_to_core`：只在正常ready的 `src_payload==0`
  且本核已有同类运行任务时给pending后继登记Scalar-prefix资格与前驱token；
  原force_gate路径仍调用 `is_scalar_prefix_wait_producer` 验证真实WAIT边。
- `aicore_executor.cpp::handoff_scalar_prefix`：后继token/前缀合同匹配后允许接续ready payload。
- `wait_scalar_prefix_predecessor`：先排空旧V/MTE3或M/FIX；只有src_payload非零才等待依赖doorbell；
  ready后继直接在真实完成后ACK。仍不提前宣告旧任务完成，不提前ACK新任务。
- 未改变队列优先级、资源分配、ABI结构尺寸、PTOAS前缀规则或公共安装。

### 25.2 同卡旧版对照与新版本验证

旧版 `b39d9e7c7`：`task_20261010_123406_402518025413` / 卡6。
两种引擎各两轮数据、Scalar状态、保护区、同核资源时序均通过，计算重叠均0；
两项测试只在最终“必须观测重叠”的断言失败，exit=1并正常释放卡。
这不是算术功能错误，而是新能力在旧版缺失的复现。

新版本：`task_20261010_123707_419287226709` / 同一卡6，**6 passed**、exit=0、卡已释放。
两轮真实同核计算重叠 AIV **48/48 + 48/48**，AIC **24/24 + 24/24**。
下表为匹配交接的原始tick，三元组为 **min / mean / max**，不是μs：

| 引擎/轮次 | 前项返回→后继前缀开始 | 后继Scalar前缀 | Scalar之后旧计算仍需等待 | 早于前项整组完成的后继数 |
| --- | ---: | ---: | ---: | ---: |
| AIV / 0 | 33 / 39.19 / 56 | 19 / 19.27 / 22 | 121 / 150.58 / 161 | 46 |
| AIV / 1 | 33 / 40.31 / 55 | 19 / 19.23 / 23 | 137 / 155.96 / 164 | 47 |
| AIC / 0 | 32 / 42.04 / 57 | 19 / 19.29 / 21 | 412 / 428.88 / 440 | 23 |
| AIC / 1 | 33 / 42.33 / 50 | 19 / 19.38 / 20 | 421 / 429.62 / 441 | 22 |

两类后继均没有等待无关block的整组结束；所有匹配的同核旧任务则在后继访问数据/流水前真正完成。
全部FP32输出与CPU计算精确一致、Scalar整数状态精确、16行输出保护区与trace保护行保持原值。
受影响的既有回归：3/33/7-task AIV链共40条边仍重叠；双前驱两轮8/8；
sync_start两轮正常回退0/0；有WAIT边的AIC两次重叠及无后继收尾通过。
仍是人工放大窗口的功能探针，不宣称CSA收益或生产SDK整体收益。

产物（私有根 `artifacts/`）：`tmr_ready_compile_v1`至`v5`及日志/XML；
`tmr_ready_baseline_device_v1.task.log`与对应结果目录；`tmr_ready_runtime_build_v1.log`；
`tmr_ready_successor_v1.patch`；`tmr_ready_device_v1.task.log/.xml`与
`tmr_ready_device_v1/{ready_aiv,ready_aic,chain,fanin_fanout,sync_fallback,aic}/`。
下一步只抽取冻结CSA中的真实Cube工作负载，保持原算术和分块，验证自然尾部窗口与净收益；
不向生产CSA一次性铺开。
私有Simpler已保存为 `2e09ec35380ac10ef33329aee24a4d0ce1c32f77`（2文件、27增14删，中文说明与Signed-off-by），未push。
私有PyPTO SDK pin随之更新，Simpler/PyPTO重装均exit=0；随后新pin下kernel-artifact CPU编译/加载
`test_scalar_prefix_timing.py --codegen-only`为 **1 passed**。
证据 `tmr_ready_{simpler,pypto}_install_v1.log`、`tmr_ready_kernel_compile_v1.log/.xml`，未新增NPU测试。

## 26. 过程 024：准备真实NZ O投影的独立Cube接续探针

候选来自冻结版本fc6d201b的 `_proj_a_mm_nz`、T=96分支：8个group，
每组8个N128 block，M容量128、有效M96、K4096按K256分16段，K循环stage=2；
与真实根路径一样使用PIPELINE_OA=False，不改算术、不增加矩阵乘或Scalar迭代。
隔离图只保留这8组O-A投影，输入在调用前已ready；不声称它保留完整CSA中与量化等任务的并发。

私有冻结包增加 `scalar_o_proj_probe.py` 与
`tests/pypto_test/tmr_csa_o_projection_scalar_overlap_demo.py`。
普通/async+prefix入口共用投影主体；Scalar前缀仅计算原有行、列、组偏移及有效行数。
async入口同时作为上一个block的后继与下一个block的前驱，继续独立编译/动态派发。
权重由CANN `npu_format_cast(...,29)`准备。最初ChipWorker方案仅复制Native物理字节，
因下述program入口NZ元数据校验问题停下；最终使用与CSA一致的TMR kernel入口，
直接借用Native NZ设备存储，完全移除跨进程权重复制。CPU编译阶段不初始化设备。

CPU准备记录：v1的cache声明不允许放在裸inline helper；v2不允许给只调用预定义kernel的SPMD空scope附cache策略；
v3的局部trace view作为返回值时被JIT按根张量形状推断，改为固定9行Out入参且不返回trace view；
v4给incore helper再套CORE_GROUP scope引入残留ScopeStmt且错误定核，编译拒绝，未绕过验证。
v5改用显式tile.load上的BYPASS，同时保留M/N/K、NZ格式、L1→L0搬运和两级K流水。
这是原矩阵算术的tile表达探针；需核对生成指令后才能宣称与原完整kernel调度一致。
v5完整CPU编译通过，生成的PTO确认NZ TLOAD保留 `l2_bypass`，K256的stage=2循环及末尾M/FIX事件仍在。
trace的ScalarWriteLineShared提示文字虽说“64 bytes apart / 1 of them share”，
实际每block独占8×INT64、每group跨度9条cache line；测试显式确认基址64字节对齐并检查每组保护行。

### 26.1 首次入口错误与切换

`task_20261010_125403_111113032259` / 卡1：编译、Native格式转换和runtime初始化通过，
但首次handle调用在Python `_validate_device_tensor` 拒绝NZ参数；尚未提交投影task，不能称为Cube执行失败。
其info仍为逻辑rank3 `(8,4096,1024)`，参数比较却先转换为NZ物理rank5，出现错误文字
“expects rank 3 (shape (8,4096,1024)); got DeviceTensor shape (8,4096,1024)”。
没有跳过校验或顺手修改通用接口；改用 `specialize()` 的pre-pass IR生成KernelArtifact，
走 `pypto.torch` 的正式Native NZ借用路径。此任务exit=1并释放卡。

### 26.2 kernel入口的真实形状投影证明自然Cube重叠

kernel CPU编译通过；`task_20261010_125718_13213544895` / 卡8，exit=0，卡已释放。
T=96，两轮换输入；每轮普通/异步各执行一次。
FP32输出与整数值BF16输入的CPU矩阵参考精确一致，普通/异步逐bit一致；
全部padding/16行额外保护区、每组trace保护行通过；权重始终格式29且data_ptr不变。

每轮64个block中的**40条动态接续边全部证明Cube计算重叠**；其余首次上核的block正常独立启动。
原始tick诊断（min / mean / max）：

| 轮次 | 同核接续/有计算重叠 | 前项返回→后继函数内首个时钟 | 两个前缀时钟之间 | 边界后旧Cube仍需等待 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 40 / 40 | 38 / 50.33 / 69 | 0 / 0.075 / 1 | 153 / 170.53 / 188 |
| 1 | 40 / 40 | 38 / 50.65 / 78 | 0 / 0.050 / 1 | 140 / 170.88 / 192 |

这里没有人为增加Scalar递推或matmul次数。两个时钟之间很短，不代表整个Scalar准备只有这些时间：
wrapper参数解包在首个时钟之前，普通偏移计算也可能被编译器提前。
可确定的是后继已经进入自己的独立入口，而同核旧Cube仍在工作；不能用此插桩结果宣称净收益。
当前图是保持真实形状、分组、算术和NZ策略的**显式tile投影表达**，
不是完整CSA，也尚未逐条确认与生产tensor表达的生成指令完全一致。

产物：`csa_o_projection_compile_v1`至`v5`及日志；`csa_o_projection_device_v1.task.log`；
`csa_o_projection_kernel_compile_v1`及日志；
`csa_o_projection_kernel_device_v2/results.json`、`program.py`、构建产物及同名前缀task日志。

### 26.3 去掉诊断的隔离图计时准备

同文件追加无trace的ordinary/async Scalar-first入口，共用同一投影主体；
`tmr_csa_o_projection_scalar_timing.py` 用两张NPUGraph，warmup后同卡交替次序各采30次完整图重放，
记录NPU event的min/mean/max；构建核验生成入口不含诊断时钟。
计时前后检查输出与padding，末尾改符号并重新毒化输出，排除只重放旧结果。
两侧都是同一实验SDK，测量范围是隔离O-A图，不冒充完整CSA或原始SDK对照。
CPU构建通过。首轮 `task_20261010_130235_152737821825` / 卡8，exit=0并释放卡：
普通133.02 / 159.43 / 286.30 μs，异步132.54 / 152.78 / 203.60 μs；
均值少4.17%，但仅18/30对更快，普通侧243/286 μs两次长尾明显，不能据此定为稳定收益。

因此只补一轮针对计时噪声的验证：一张图连续包含16次相同投影调用，
用完整图的event时间除以16，减少每次单独Host提交/等待引入的空隙；
仍交替次序各30个样本，不修改kernel、算术或权重。不要将这组均值的max称作单次调用的max。
`task_20261010_130440_163280731359` / 同一卡8，exit=0并释放卡：

| 计时方式/路径 | min μs | mean μs | max μs | 配对更快数 |
| --- | ---: | ---: | ---: | ---: |
| 逐次单独event / 普通 | 133.02 | 159.43 | 286.30 | — |
| 逐次单独event / 异步Scalar-first | 132.54 | 152.78 | 203.60 | 18/30 |
| 连续16次折算 / 普通 | 121.2763 | 125.3783 | 127.8675 | — |
| 连续16次折算 / 异步Scalar-first | 119.6175 | 122.5314 | 125.2975 | 29/30 |

连续口径平均每次少 **2.8470 μs / 2.2707%**，min/mean/max均降低；
中位数125.7069→122.4981 μs，样本标准差1.7003/1.5488 μs。
两轮均完成普通/异步、图重放前后、改符号输入和padding保护检查，且Native NZ存储地址/格式保持不变。
这支持**同一实验SDK下、隔离的真实形状tile O-A图有净收益**，
不证明完整CSA加速、原tensor表达等价性能、或相对未修改SDK的公共开销已经消失。
首轮4.17%保留为带长尾的历史观测，不作为主收益数字；连续16次口径的2.27%作为本阶段结果。

命令：`VLLM_ASCEND_ENABLE_NZ=2 python tests/pypto_test/tmr_csa_o_projection_scalar_timing.py
--device "$TASK_DEVICE" --replays-per-sample 16 --output <私有artifacts>/csa_o_projection_timing_device_v2`，
整条命令经task-submit单卡锁运行。原始60个event样本同时保存graph_us与折算us。
证据：`csa_o_projection_timing_compile_v1`及日志；
`csa_o_projection_timing_device_v1/results.json`、`csa_o_projection_timing_device_v2/results.json`及两份task日志。

下一步先把冻结的原tensor表达纳入同图对照，确认提取为tile表达没有改变基准工作量或引入额外代价；
再只在私有CSA的这个已验证位置试接入，保留未标记路径。此后才考虑其他任务与完整CSA验收。

## 27. 过程 025：原tensor表达对照发现探针改写的L0流水损失

为避免把表达改写的性能变化算成动态接续收益，将冻结的 `_proj_a_mm_nz`
非PIPELINE_OA分支加入同一隔离图。三路共用同一输入、Native格式29权重、输出和私有SDK，
同样8组、每组8块、T=96、K4096；源表达仅移除本隔离图无需等待的外部heads依赖，
保留tensor slice/matmul/assemble、cache BYPASS和K256的stage=2循环。

首次CPU编译 `csa_o_projection_source_compile_v1` 在KernelArtifact完成检查处失败。
错误同时列出权重逻辑rank3→物理rank5和输出Out→InOut；实际拒绝原因是**输出读写角色变化**，
不是NZ形状变化：既有校验已允许同dtype/角色/元素数的NZ视图转换。
将探针根与源表达helper的部分写入输出正确声明为InOut，v2编译通过；未放宽校验或修改通用编译器。

设备任务 `task_20261010_142900_16597412433`，卡1，exit=0并释放卡。
每张图连续16次投影、各30个样本、三路轮换次序、无trace/profiler；下表为每次折算μs，
其中min/max是批次均值的极值，不是单个task或单次调用的极值。

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| tile表达普通完成 | 120.5650 | 127.4395 | 131.2625 |
| tile表达异步Scalar-first | 118.7937 | 125.4575 | 128.5300 |
| 冻结原tensor表达 | 107.5900 | 113.5393 | 116.3488 |

三路输出与整数输入的精确CPU参考一致，改符号后的图重放、padding保护及权重借用检查通过。
异步相对同tile普通表达少1.56%，25/30对更快；但比原tensor表达**慢10.50%**。
因此过程024的2.27%仅为同tile表达内收益，不代表原O-A或完整CSA收益，不能直接推广该改写。

生成代码的明确差异：原tensor lowering把每个K256 L1块拆成两个K128 L0块，
使用两个L0缓冲和TEXTRACT；探针却把整个K256直接TMOV到L0。
这改变了核内分块与流水，先前“保留K256 stage=2”不足以证明原流水被保留。
下一步仅在私有探针恢复K128 L0拆分，再与原表达同卡对照；不改生产CSA或调度策略。

证据（私有根artifacts）：`csa_o_projection_source_compile_v1.log`、
`csa_o_projection_source_compile_v2/`、`csa_o_projection_source_device_v1/results.json`及task日志；
改写前源码快照 `csa_o_projection_probe_before_l0_split.py`。
用户在本阶段限定完整CSA性能只测128K/B16，已同步执行方案；不再扩展七档。

## 28. 过程 026：保留原tensor lowering后证明O-A自然重叠与局部收益

先尝试手工恢复K128 TEXTRACT：外层stage=2又复制了显式L0缓冲，CPU编译报告
Left需要131072 bytes、硬件容量65536 bytes。该候选未上卡，未调高容量限制。
随后撤回手工tile重写，保留原tensor slice/matmul/assemble。
预定义InCore函数使用已有内部 `cache_policy=[(1,1)]`（第1号只读参数wo_a、BYPASS=1），
继续由原ConvertTensorToTileOps生成NZ加载、K128 L0拆分和双缓冲。
这仅是私有demo对已有内部属性的使用，不新增公共DSL API；参数顺序变化时须同步属性索引。

该表达首次CPU编译被实验异步指令白名单拒绝：原lowering还会调用 `pto.set_validshape`，
更新局部L0C tile的有效行列。复核PTOAS现有实现后，只将该元数据操作加入Cube异步尾部允许集合；
它不提交计算或访问GM，原内部同步与event7完成合同不变。
独立Scalar-prefix校验仍禁止该操作出现在wait之前，负例已验证被正确拒绝。
修改范围：私有 `ptoas-src/tools/ptoas/ptoas_pipeline.cpp`、私有probe与timing脚本；
Simpler仍为 `2e09ec353`，本阶段没有继续改调度。

PTOAS以资源上限2线程重建通过；timing三路与trace两路完整CPU编译/KernelArtifact加载通过。
生成代码确认原scope表达与独立普通入口的engine API调用次序一致（只归一化临时变量名）；
L0均为128×128，均保留K256 L1、K128 L0、stage=2和NZ BYPASS。
这项静态检查配合下述功能/计时，不作为整个二进制完全相同的证明。

### 28.1 同卡机制与功能

任务 `task_20261010_143841_681878510` / 卡9，先trace再独立进程无诊断计时，exit=0并释放卡。
T=96对应B16×验6；权重64 MiB Native格式29直接借用。
两轮各40条同核接续，观测实际Cube重叠37/40、40/40；无人工增大Scalar或矩阵工作量。
原始tick的min/mean/max：

| 轮次 | 同核接续/计算重叠 | 前项返回→后继首时钟 | 两个前缀时钟之间 | 边界后旧Cube仍需等待 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 40 / 37 | 47 / 65.475 / 93 | 0 / 0.050 / 1 | 0 / 32.725 / 56 |
| 1 | 40 / 40 | 40 / 59.625 / 87 | 0 / 0.050 / 1 | 11 / 38.500 / 58 |

每轮普通/异步输出与整数输入的FP32 CPU参考精确一致，padding、16行输出保护区和trace保护行通过。
时钟未覆盖wrapper全部Scalar，不能把0～1 tick当成完整Scalar准备时长。

### 28.2 不带诊断的三路计时

每图连续16次调用、5次warmup、每路30个样本，三路轮换次序；表中是批次时间除以16。
min/max仍是批次折算值的极值，不能解释成单次CSA调用的极值。

| 隔离O-A路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 冻结原tensor scope | 105.1300 | 111.9365 | 115.9837 |
| 相同tensor算术、独立普通入口 | 105.6825 | 111.9240 | 116.2925 |
| 相同tensor算术、异步Scalar-first | 103.5037 | 110.3405 | 114.0938 |

普通独立入口相对原scope均值差-0.011%，此前tile改写的额外损失已消除。
异步相对原scope均值少 **1.5960 μs / 1.4258%**，22/30配对样本更快；
相对普通独立入口少1.4148%，23/30更快。min/mean/max均改善，但收益量级仍小，不能外推完整CSA。
三路初次调用、graph replay、改符号输入后的输出与保护区均精确一致；NZ权重地址/格式不变。

证据（私有根artifacts）：`csa_o_projection_l0_split_compile_v1.log`与
`csa_o_projection_probe_l0_split_compile_failed.py`；`csa_o_projection_tensor_compile_v1.log`；
`csa_o_projection_tensor_ptoas_build_v1.log`；`csa_o_projection_tensor_compile_v2/`、
`csa_o_projection_tensor_trace_compile_v1/`、`csa_o_projection_tensor_codegen_summary.json`；
`csa_o_projection_validshape_before_wait_invalid.pto/.log`；
`csa_o_projection_tensor_trace_device_v1/results.json`、`csa_o_projection_tensor_device_v1/results.json`
和 `csa_o_projection_tensor_device_v1.task.log`。

下一步只接入私有CSA的NZ O-A候选位置，保留原始路径作为同卡对照，验证完整 **128K/B16**
的输出/状态及min/mean/max；不测七档，不触碰HBG，不改共享安装。
暂未创建完整CSA候选或执行完整CSA，不能将隔离图结果填写为CSA性能。

## 29. 过程 027：私有完整128K/B16 CSA首轮功能与计时

### 29.1 接入范围与编译

在私有vLLM worktree冻结的原包 `deepseek_v4_flash_csa_scalar_overlap_demo` 之外，
创建整包副本 `deepseek_v4_flash_csa_scalar_overlap_full`。仅在NZ O-A的T=96、N128、
非PIPELINE_OA分支调用过程026已验证的 `projection_timing_overlap` 独立InCore入口；
保留原heads_dep、组/块数量、allow_early_resolve、后续量化/O-B/HC_post和其他形状回退。
不改共享生产包，不改Native cache布局，也不更改算术、归约或量化策略。
`_get_dep_graph()`解析、完整CPU编译及KernelArtifact加载均通过。

版本延续过程026：vLLM冻结fc6d201b，Simpler 2e09ec353，私有PyPTO/PTOAS补丁及CANN9.2。
两路共用同一实验SDK，不是原封不动的上游SDK与候选SDK对照。
新增脚本在私有worktree `tests/pypto_test/`：
`tmr_csa_full_128k_b16_compile.py`使用meta张量预编译；
`tmr_csa_full_128k_b16_demo.py`在队列分配单卡上加载编译产物和第2层真实权重。
权重固定 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，seed=1024，NZ mode=2，
`VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0`、deterministic_level=0，无profiler、无EPLB。
使用Native builders、页表、compact metadata、原始cache视图；不调用会切回共享SDK的activate()。

首个设备任务 `task_20261010_145239_131349323600` / 卡1 在fixture初始化报缺少
`_C_ascend.npu_sparse_attn_sharedkv_metadata`，尚未执行CSA，exit=1后释放卡。
原因是隔离Python checkout没有Native扩展，脚本也未注册其binding。
脚本补充只读加载原发布的 `.cache/csa/native-install/vllm_ascend_C.cpython-310-aarch64-linux-gnu.so`
并调用enable_custom_op；未复制或修改共享二进制，PyPTO/Simpler/PTOAS仍全部来自私有环境。

### 29.2 功能与单次完整图计时

修正fixture后的任务 `task_20261010_145725_151890513849` / **卡2**，exit=0并释放卡。
普通版和接续版的x_out、idx_topk、idx_topk_scores以及6个Native缓存/状态视图均逐bit一致。
metadata/允许写入slot之外的保护检查通过；两张NPUGraph各5次warmup后仍一致；
修改hidden符号后，输出确实改变，两路径输出和全部状态仍逐bit一致。
本次只对比PTO调度路径，不是Native算术精度或整模型token验收。

30轮交替A/B先后次序，每路30个样本；每次测一张完整单层图的一次replay，
NPU event包围整个HC_pre+norm+CSA+HC_post调用，缓存恢复放在计时间隔外。
不同于隔离O-A的16次折算，以下min/max是这30次完整CSA单次调用的极值。

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 原CSA，普通task完成 | 944.5600 | 969.9360 | 997.5600 |
| O-A采用同核动态Scalar-first接续 | 937.3200 | 965.7000 | 985.8000 |

均值少4.2360 μs（0.4367%），最大值少11.7600 μs；30对中17对接续版更快。
均值差明显小于样本波动，本次只记为**首轮小幅改善，尚未证明稳定净收益**，
不能拿隔离O-A的1.43%直接外推，也不能宣称已消除新SDK的公共路径开销。

复现命令（私有vLLM worktree，先source ../env.sh；实际设备命令由task-submit执行）：

```bash
VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 \
python tests/pypto_test/tmr_csa_full_128k_b16_compile.py \
  --output "$SCALAR_OVERLAP_ROOT/artifacts/csa_full_128k_b16_compile_v1"
VLLM_ASCEND_ENABLE_NZ=2 VLLM_ASCEND_PTO_CSA_ATOMIC_ADD=0 \
python tests/pypto_test/tmr_csa_full_128k_b16_demo.py --device "$TASK_DEVICE" \
  --compiled "$SCALAR_OVERLAP_ROOT/artifacts/csa_full_128k_b16_compile_v1" \
  --output "$SCALAR_OVERLAP_ROOT/artifacts/csa_full_128k_b16_device_v2"
```

证据均在私有根artifacts：`csa_full_128k_b16_compile_v1/`及同名log，
`csa_full_128k_b16_device_v1/results.json`及task.log记录fixture失败；
`csa_full_128k_b16_device_v2/results.json`保留全部样本、权重格式、形状与逐项校验，
同名task.log保留队列任务、设备与完整执行日志。
下一步使用独立trace包证明完整CSA中的实际同核交接；诊断不混入正式计时样本。

## 30. 过程 028：完整CSA诊断包与原SDK对照准备

诊断包 `deepseek_v4_flash_csa_scalar_overlap_trace` 从完整候选独立复制；
仅将O-A入口换为已验证的trace版本，并沿Python helper传入两个72×8的int64输出缓冲。
64行对应8组×8块，另8行作为每组保护行；参数追加在根入口末尾，不挪动原56个参数。
原heads依赖、quant/O-B/HC_post以及运行时动态选择保持不变。

首次CPU编译 `csa_full_128k_b16_trace_compile_v1` 在入口ABI验证失败，未上卡：
返回别名识别为[55,56,57]，而复杂根函数新增tuple返回没有推断出return_types。
trace缓冲本已显式声明为Out，无需再加入返回值；恢复原单一layer_output返回后，
`csa_full_128k_b16_trace_compile_v2` 的完整编译和KernelArtifact加载通过。
没有绕过ABI检查，也没有为诊断修改通用返回值推断。

设备任务 `task_20261010_150632_176227614666` 已提交自动单卡队列。
等待卡释放后在卡8执行完毕，exit=0并释放卡；完整图机制结论来自下述实际设备时序。
测试脚本使用 `--trace-compiled`，仅跑两种输入的普通/trace输出及全状态对照、保护检查、
实际核号/时钟和安全等待边界；不收集正式性能数字，不启用会关闭交接的runtime DFX。
产物目标：私有根artifacts下 `csa_full_128k_b16_trace_device_v1/`及同名task.log。

完整图任务两轮均观测到40条O-A同物理AIC接续边，其中36/40、39/40在后继安全Scalar结束后，
前驱Cube仍有超过5 tick的计算尚未完成。完整图没有固定A+B组合入口或人为Scalar循环。
第二轮翻转hidden输入；两轮相对原路径的x_out/Top-K/score及6个缓存状态视图均逐bit一致，
Native metadata和允许slot外的保护区、每组trace保护行及安全等待边界全部通过。

| 轮次 | 同核接续 / 计算重叠 | 前项返回→后继首时钟 min/mean/max tick | 两个前缀时钟间 min/mean/max tick | 边界后旧Cube等待 min/mean/max tick |
| --- | ---: | ---: | ---: | ---: |
| 0 | 40 / 36 | 48 / 96.825 / 173 | 0 / 0.050 / 1 | 0 / 62.925 / 136 |
| 1 | 40 / 39 | 40 / 102.600 / 180 | 0 / 0.100 / 1 | 0 / 62.000 / 118 |

原始时钟与核号在 `csa_full_128k_b16_trace_device_v1/results.json`，
汇总在同目录 `mechanism_summary.json`。时钟不覆盖wrapper全部Scalar，0～1 tick不能解释成
完整Scalar准备时长；诊断写回会扰动时序，这组数据只用于证明重叠与等待合同，不计性能收益。
无诊断完整图计时仍使用过程027的独立结果，不把这两次trace与其样本混算。

为补齐过程027中“两组共用新运行时”的结论边界，在私有根新增 `sdk-baseline/`：
PyPTO `cb2484471`、Simpler `ef6812891`，均从原始commit建立detached worktree；
独立venv、独立源码编译和缓存，没有复用实验工程的.so。
CANN9.2、GCC15、PTO-ISA与PTOAS采用相同工具链，待测CSA使用相同冻结包、权重和脚本。
这组将比较去掉Scalar接续补丁前后的PyPTO/Simpler公共开销，不混入PTOAS版本变化。

Simpler binding和TMR host/AICPU/AICore运行时已经构建成功；PyPTO正在以2线程构建。
新venv初次pip命令不支持短参数 `-C`，改为已有长参数 `--config-settings` 即安装成功，
没有升级公共pip或改变任何共享环境。baseline的runtime路径仅作本地symlink指向其独立Simpler，
不把该symlink当Git submodule改动提交。

对照功能已合并到同一个 `tmr_csa_full_128k_b16_demo.py`，使用与过程027相同的fixture与计时；
准备时的临时sdk_compare脚本已删除，避免维护两份重复入口。该脚本
支持只加载原路径（原SDK）或同时加载原/接续路径（新SDK），保存输出用于跨SDK逐bit比较。
后续只在同一卡上做原SDK→新SDK→原SDK的括号对照；每次30个单次CSA样本，报告min/mean/max。
进程间对照仍可能受频率/温度变化影响，不能比同进程逐step交替更强；不展开其他shape。
此项尚未CPU编译CSA或上卡，不能提前记作公共开销已通过。

## 31. 过程 029：原SDK括号对照通过功能，完整CSA小幅收益未复现

原SDK的独立源码构建和冻结CSA编译均已完成，版本仍为PyPTO `cb2484471`、
Simpler `ef6812891`；实验SDK的Simpler为`2e09ec353`。两边使用同一私有PTOAS、
CANN9.2、GCC15、PTO-ISA及第2层真实权重，没有复制另一worktree的项目二进制。
CPU检查生成的`proj_a_mm`、`qk_pv_aic`、`hc_widen_rms`三个普通InCore函数体：
仅去除整行注释和空行后源代码一致；不包括wrapper，也不等于完整二进制一致。
结果在私有artifacts下`csa_full_128k_b16_ordinary_codegen_check.json`。

新增私有脚本`tmr_csa_full_128k_b16_sdk_pair.sh`，在同一卡依次执行原SDK、实验SDK、原SDK。
每个进程使用相同fixture和30次单次完整CSA计时，实验SDK内部交替普通/接续路径；
保存输出用于跨进程逐bit比较。记录的是HC_pre+norm+CSA+HC_post，仍仅128K/B16，
无trace/profiler，缓存恢复在event计时外；当前每次event结束都等待Host，后续不混用计时口径。

首次任务`task_20261010_153156_274495231554`在卡0失败并释放卡。
原SDK环境脚本覆盖PYTHONPATH，丢失CANN的`tbe`模块，Native权重转NZ报ACL 500001；
此时尚未执行CSA，不是异步交接或算术错误。保留继承的CANN Python路径后，
CPU确认`tbe`来自同一CANN9.2，无需修改/重编算子或runtime。
失败日志保留在`csa_full_128k_b16_sdk_pair_v1/original_before.log`。

修复后的任务`task_20261010_153414_284637034`在卡9完成，exit=0并释放卡：

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 原SDK，前测 | 960.0400 | 977.5493 | 1004.6600 |
| 实验SDK，普通CSA | 951.9600 | 979.6740 | 1014.1400 |
| 实验SDK，动态Scalar接续CSA | 960.9400 | 980.6773 | 1002.7800 |
| 原SDK，后测 | 944.3800 | 959.0753 | 986.3000 |

实验SDK内部接续版均值高1.0033 μs（0.1024%），仅11/30配对样本更快；
**过程027的0.44%改善没有复现，不认定完整CSA稳定受益**。
原SDK后测相对前测均值低1.8898%，其括号均值为968.3123 μs。
实验普通/接续相对该均值分别高1.1733%/1.2770%，但原SDK自身漂移更大，
这两个比例只描述本轮样本，不能作为已证实的runtime公共开销。

原SDK前/后与实验SDK的`x_out`、Top-K和score均逐bit一致。
各进程保护区、图重放及改输入检查通过；实验普通/接续的6个Native缓存/状态视图也逐bit一致。
没有由这组单层结果推断16卡token/DSpark通过。完整样本、SDK路径、功能检查及对照输出在
`csa_full_128k_b16_sdk_pair_v2/`各子目录，根目录`summary.json`保留全部四路结果与漂移。

下一步先检查普通task的新增调度/上下文处理开销和计时中的Host间隔，
只补能区分机制开销的128K/B16验证，不通过反复取样筛出正收益；HBG继续暂缓。

## 32. 过程 030：精简普通任务的交接上下文写回

过程029之后先审查公共路径，未改变算术或计时方法。
原型的`execute_task()`在每个普通任务开始时也写零`async_tail_kind`和`scalar_overlap_trace`，
末尾无条件调用`flush_overlap_context()`，导致未发生交接的任务也写回两条控制缓存线。
私有Simpler提交`4cd49d3a7`把这两个字段与派发上下文一起由AICPU初始化；
普通AICore任务保持上下文只读，只在真实Scalar交接的同步后继退出时保留额外写回。
异步前驱的V/MTE3、M/FIX等待和FIN前写回完整保留，双缓冲代次及ACK/FIN语义不变。
这只消除源码中明确存在的普通路径工作；不能据此定量认定过程029的差距来自此处。

两处C++修改的TMR构建已完成。任务`task_20261010_154942_353355718572`在卡1执行，
6项设备检查通过：ready AIV/AIC、普通前驱和无后继回退、AIC真实WAIT依赖、
双前驱双后继、sync_start普通回退。第7项kernel图捕获测试在CPU编译检查处被拒绝，
原因是Simpler当时尚未提交，而KernelABI要求精确匹配干净SDK revision；未进入图执行。
脚本因此未开始完整CSA计时，任务exit=1并正常释放卡，不能将其记为整套通过。

补救保持版本检查：提交Simpler改动，同步私有PyPTO的revision引用并重建binding，
重新CPU生成与该revision匹配的CSA产物，然后只补图重放和128K/B16完整CSA。
已通过的六项不因版本文字变化重复执行。
第二次任务`task_20261010_160354_5460021283`在卡0仍未进入图执行：
Simpler binding和CSA产物已更新，但PyPTO的`_torch_npu`桥接模块还嵌入旧revision，
`pypto.torch.init()`因此拒绝加载。属于私有环境构建遗漏，不是设备端功能失败；
任务exit=1并释放卡，日志为`tmr_clean_context_v2.task.log`。
随后仅增量重建`_torch_npu`，在CPU调用`_load_native()`确认其revision与Python引用
均为`4cd49d3a7`，保留原版本检查。构建日志为`pypto_clean_context_adapter_build_v1.log`。
第三次任务`task_20261010_161813_727410715`复用已匹配的CPU编译产物，
仅补图重放和128K/B16 CSA，结果另存v3，不覆盖前两次失败证据。
首次脚本CPU收集检查还发现PyPTO的`tests`命名空间被私有vLLM的同名包覆盖，
回归脚本在pytest阶段去除vLLM Python路径，CSA阶段重新加载私有env；设备任务使用的即修正后的脚本。
原始任务日志为`tmr_clean_context_v1.task.log`，六项trace/结果在
`tmr_clean_context_regression_v1/`，Junit报告同名xml。构建与binding日志分别为
`tmr_overlap_clean_context_build_v1.log`、`tmr_clean_context_binding_install_v1.log`。

最终16卡功能验收仅做CPU准备，尚未提交设备任务：沿用固定混合128K bank最前16个不同请求，
TP1、DP=EP16、每请求192 token、NZ2、atomic关闭、确定性level1/HCCL开启，HCA两侧保持同一现有PTO实现。
比较普通CSA与动态接续CSA的逐token、逐请求DSpark事件及总计数，不筛选数值一致请求。
两侧仍使用私有PyPTO/Simpler/vLLM，Native扩展只读借用已有安装，OPP/static_kernel各自私有。
支持脚本导入和私有OPP准备已在CPU确认；比较器的相同报告及token/事件差异负例检查通过。
这些准备不算整模型验收，也不产生整模型性能结论；CSA性能范围始终只有128K/B16。

## 33. 过程 031：普通任务写回精简通过功能，接续路径开销仍需优化

第三次任务`task_20261010_161813_727410715` / 卡0，exit=0并释放卡。
PyPTO kernel图捕获/重放通过；完整128K/B16 CSA的原路径与接续路径在首次执行、
warmup和修改hidden输入后的图重放均逐bit一致，覆盖输出、Top-K、score及6个缓存/状态视图。
Native metadata、允许slot之外的保护区通过。连同过程030已经通过的六项基础检查，
本次上下文写回精简没有发现功能回退；没有重复那六项硬件检查。

性能仍为同一实验SDK `4cd49d3a7`、每路30次单次完整CSA、交替A/B先后次序，
CANN9.2、NZ2、atomic关闭、无profiler，缓存恢复不计入event区间：

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 普通CSA | 954.0800 | 970.1713 | 1001.7999 |
| 动态Scalar接续CSA | 964.6600 | 983.0100 | 1003.8800 |

接续路径均值高12.8387 μs（1.3233%），30对中仅6对更快。
这轮存在明确的样本性能回退，不能写成“已经无负收益”；也不能把不同卡、不同时间的
过程029与本轮直接相减，归因为写回精简的绝对收益。机制和结果正确性单独保留，
后续继续削减实际交接成本，不以功能已通过替代性能优化，也不因性能暂未改善撤回路线。
证据：私有artifacts的`tmr_clean_context_v3.task.log`、`tmr_clean_context_capture_v3.xml`、
`csa_full_128k_b16_clean_context_v3/results.json`（含全部60个样本及逐项功能检查）。

源码复核发现真实异步任务仍需在FIN前写回两条控制缓存线：当前`GlobalContext`
从DispatchPayload偏移464开始，AICore会改写的字段横跨448和512两条线。
下一候选利用已有payload尾部padding，把这些可变字段集中到同一条64B缓存线，
保持DispatchPayload总长576B、冷字段位置和完成语义。
待应用补丁放在私有artifacts的`single_control_line_candidate/candidate.patch`；
以候选头文件作CPU布局编译，确认GlobalContext=104B、payload仍576B、可变字段为
偏移512起的56B，冷字段偏移保持不变。没有应用到SDK或做设备验证，不能预报收益；
布局结果和后续所需构建项在同目录`layout_result.json`及`README.md`。

最终16卡功能对照已提交`task_20261010_162022_87419315499`，使用过程030准备的
固定混合bank和私有普通/接续CSA，输出目标`tmr_scalar_overlap_model_pair_v1/`。
随后用户明确CSA测试只用单卡。该任务尚处pending时已取消，未分配/占用卡、未产生模型结果；
不再把整模型检查列为本goal的待完成项。解除这项排队任务所需的源码冻结，继续单卡候选。

## 34. 过程 032：交接控制字段集中到单缓存线，仅做单卡验证

用户限定CSA测试只用单卡后，取消未执行的16卡任务；废弃的两个整模型准备脚本
移出私有vLLM活动测试目录，保留在artifacts的`abandoned_model_pair_preparation/`供过程追溯。
后续功能和性能均在单卡完成，完整CSA范围仍只有128K/B16。

私有Simpler提交`a1a25dc33`涉及三个文件：`common/intrinsic.h`重排实验交接字段，
`runtime/dispatch_payload.h`增加缓存线起点和长度断言，`aicore/aicore_executor.cpp`
仅刷写对应一条控制缓存线。sub_block_id、DMA workspace位置和576B payload总长不变，
前驱实际完成等待、FIN前写回和双缓冲代次均保持。
与前一revision相比+30/-21行，包含注释及布局断言；没有增添新的运行开关或改动算术。

CPU候选头文件布局检查及TMR Host/AICPU/AICore构建通过。
同步PyPTO revision，顺序重建Simpler binding及PyPTO `_torch_npu`桥接，CPU `_load_native()`
确认两侧均为`a1a25dc334c811ead7c371e010ae159cd9b400cc`，避免再次占卡后才发现版本遗漏。
四项基础用例（ready AIV/AIC、AIC WAIT、长链图重放）CPU预编译通过（4 passed），
完整CSA普通/接续两份产物也已完成CPU编译与加载准备，尚无该候选设备结论。
入口为私有artifacts的`run_single_control_line_v1.sh cpu|device`，device模式只接受单一卡号。
构建与版本检查日志为`tmr_single_control_line_build_v1.log`、
`tmr_single_control_line_binding_build_v1.log`、`pypto_single_control_line_adapter_build_v1.log`、
`tmr_single_control_line_adapter_preflight_v1.log`；CPU预编译日志`tmr_single_control_line_cpu_v1.log`。

算子接入边界也已向用户说明：当前是显式`scalar_prefix`/异步返回属性与
`pl.system.wait_predecessor()`，编译器检查安全边界，运行时决定同核后继。
原SPMD task依赖、block数和数学主体不变；尚未实现自动提取Scalar前缀和自动插入等待。
vLLM接入文件的+37/-1行不含独立InCore探针实现，统计时须分别列出，不能当成全部算子侧代码量。

## 35. 过程 033：撤回算子手工标记要求，改为编译器自动识别与插入

用户明确指出：要求前驱声明`async_aic_return`/`async_aiv_return`、后继声明`scalar_prefix`不可接受，
目标应为PyPTO编译自动插入`early_start`与`fake_end`。当前先核实编译链并记录实施思路；
本过程没有修改编译器或runtime实现，没有提交NPU任务。

源码确认现有能力是人工边界的检查与代码生成：PTOAS的`validateScalarPrefixContracts()`要求
函数已带属性且恰有一个显式`pypto_wait_predecessor`，不会自动找边界；PyPTO wrapper与
`BuildSingleKernelSubmit`也直接读取这些属性。因此现有“DSL自动生成入口”的证据仅指机器代码生成，
不能表述为算子无需标记。私有NZ O-A的+37/-1接入及独立helper不作为最终写法保留要求。

修正方案已写入[执行方案第5.7节](DSV4_FLASH_CSA_SUPERKERNEL_ANALYSIS.md#57-远期方向普通算子源码不变编译链自动生成交接协议)：

- 编译器独立分析每个task，从稳定参数/shape/block信息及纯Scalar运算识别安全前缀，自动在第一处
  数据、流水或硬件状态危险操作之前等待。初版先利用原有连续前缀，再逐步做安全Scalar提升。
- wrapper与EmitC的隐式atomic/mask初始化同样必须在等待之后；前缀结果需建立能保留到最终编译的
  顺序依赖，避免CCEC把纯Scalar计算下沉到等待后。不能把“读到三个True属性”作为自动分析。
- PTOAS保留内部同步，分析实际流水完成覆盖关系，自动生成fake-end与相应事件；Simpler仅在真实
  完成后FIN/释放依赖/回收资源。没有可交接后继时正常收尾，未知操作或模式自动回退。
- PTOAS输出统一能力描述，PyPTO用它生成隐藏ABI、wrapper和orchestration。当前后端顺序正是先PTOAS、
  后orchestration，适合增加这个反馈路径；运行时依旧动态选择同核后继，不在编译期配对task。

下一项功能验收从普通、无协议标记的PyPTO小用例开始，证明自动生成并真正同核重叠后，
再用原CSA源码做128K/B16单卡验证。性能与功能分开记录，HBG和16卡不进入本轮范围。
过程032的单缓存线候选已经完成CPU预编译，设备验证暂不继续；没有新增性能结论。

## 36. 过程 034：对照SK/PDL，收敛为原task内部的局部协议语句

用户澄清自动识别是未来方向，当前即使需要手工标记，也不应整体改写InCore task。
本轮仅复核来源并修正方案，没有修改编译器/runtime/算子实现，没有提交设备任务。

核实的来源和结果：

- 安装CANN9.2的`kernel_info_infer.py:76`直接识别`SetNextTaskStart`/`WaitPreTaskEnd`调用；
  `compile_op.py:1602`生成SK_BIND能力位，`super_kernel_sub_op_compile.py:143`控制对应编译宏。
  A3头文件将接口落为流水Set及Wait；`kernel_tpipe_impl.h:595`在SK模式省去默认尾部全流水等待。
  原计算函数可直接插入两类void语句，不要求作者重构主体；SK入口及编译合同由框架生成。
- 禁用代理读取AscendC两份接口文档，确认推荐位置分别为第一条搬运前和最后一条搬运后；
  每核执行路径必须保持接口次数匹配，不能因分支漏掉同步。参考AOT/JIT补齐同步与设备队列代码。
- NVIDIA官方PDL Guide及PTX文档确认，前驱Trigger后仍可继续执行，后继在独立工作之后调用GridDependencySynchronize；
  配合launch/graph属性启用。`launch_dependents`不是函数返回，`wait`承担真实完成与可见性等待；不保证同SM或必然并发。

现有原型之所以整体改写：`func_attr`只能放函数prologue，能力目前附着于独立InCore函数；
`wait_predecessor`返回ready值，EXIT返回0后需要作者显式跳过主体，因此私有CSA拆出helper并加`if ready`。
这是实验接口和lowering的局限，不是动态同核交接的必需写法。

近期待实现接口改为原SPMD/task内部的局部等待及接续许可语句，编译器从语句派生内部能力，
核类型、隐藏上下文、状态初始化、异常退出及异步收尾由后端处理；原计算、循环、tiling与流水保持原写法。
先在简单PyPTO case验证，仅加局部语句也可实现动态同核重叠，再扩展到原CSA 128K/B16单卡。
全自动确定插入点保留为远期方向，复用相同协议IR。详情和来源链接在
[执行方案第5.8节](DSV4_FLASH_CSA_SUPERKERNEL_ANALYSIS.md#58-当前方案参考sk与pdl只在原task中插入局部协议语句)。

## 37. 过程 035：以原task内Set/Wait替代整体函数属性，完成编译并排队单卡

按执行方案第5.8节重新实现，所有源码、构建和实验仍在私有
`.worktrees/scalar-overlap-demo-20261009/`；公共安装与另一个算子优化会话不变。
本过程当前记录到CPU预编译与排队，设备结果在取得后追加，不先预报功能或性能通过。

**实现。** PyPTO新增`system.wait_pre_task_end` / `system.set_next_task_start`注册、IR/DSL导出，
从outlining后的函数体统一收集局部边界，生成隐藏dispatch上下文参数、TMR提交能力和PTOAS合同。
作者无需填写`scalar_prefix`、`async_aic_return`、`async_aiv_return`，无需拆helper或写`if ready`。
PTOAS在真实位置生成等待及Cancel返回，保留跨Wait存活的Scalar结果，Set仍在原最后一次写回之后。
用于保持编译顺序的空asm由后端生成，仅约束运行时结果，常量不加额外约束；不要求作者给Wait补dependency参数。
早退不宣告未生成的尾部事件；没有提前交接时正常执行，有Set但无后继时由运行时排空再FIN。
内部流水同步保持，拒绝不安全前缀、重复Wait、Set后的工作、MIX/双AIV及非TMR入口。

私有Simpler继续使用`a1a25dc334c811ead7c371e010ae159cd9b400cc`，本过程未修改Simpler。
PTOAS仍基于`f5eff3ee`加私有异步尾部实现；本过程修改`ptoas_pipeline.cpp`和
`PTOToEmitC/SyncComm/FuncCall.cpp`。PTO-ISA仍为官方`327cd586`，CANN9.2，工具链与上一阶段一致。

**算子源码影响。** 整包冻结普通CSA为`deepseek_v4_flash_csa_local_boundaries`；
与`deepseek_v4_flash_csa_scalar_overlap_demo`的活动源码比较，只有`decode_o_proj.py`增加：

```python
# 原 pa_unit/pa_rows/pa_src0/n0 计算之后：
pl.system.wait_pre_task_end()
# 原 if/else 的 matmul / pipeline / store 全部保持，之后：
pl.system.set_next_task_start()
```

两行具体diff在`artifacts/csa_local_task_boundary_source.patch`。
原任务名、block数、heads依赖、NZ权重入口、K/N tiling、stage=2流水和数学表达保持。
旧`scalar_o_proj_probe`等独立helper未复制进新包，新的依赖图可解析。

**基础用例。** `test_task_boundary_statements.py`是普通`@pl.jit`+三个原位SPMD任务，
用运行时choice=0/1/-1/0检查同一二进制动态A→B、A→C、无后继及重复使用，包含更换输入与输出保护区。
`test_local_task_boundary_overlap.py`用相同源码形式分别覆盖48 AIV和24 AIC block。
仅测试harness在outlining后指定诊断缓冲；该诊断属性不参与开启交接，算子源码仍仅使用Set/Wait。
设备诊断将核对同物理核、前驱函数返回、Vector/Cube真实完成和后继Scalar区间，不能只凭泳道条相交判断。

**CPU结果。** 两侧C++构建成功；PyPTO使用本worktree产物安装到私有venv，确认实际载入的新操作注册。
第一次运行发现已构建但私有安装的扩展未更新，在CPU阶段修正；未因此占卡。
PTOAS第一次局部退出lowering缺`emitc.yield`导致验证器拒绝，补齐后完整编译通过。
诊断用例最初把Scalar写放到尾部，被既有安全检查拒绝；改为在Wait后、首次tile操作前写诊断，未扩大异步支持集合。

- 解析round-trip：1 passed（`local_boundary_parser_v2.log`）。
- PTOAS局部Wait生成、重复Wait拒绝、危险前缀拒绝、Set后仍工作拒绝：4项通过；
  结果`local_task_boundary_ptoas_checks.json`，源码在`ptoas-src/test/lit/pto/local_task_*.pto`。
- 三项基础用例完整CCEC二进制编译/加载：3 passed（`local_boundary_final_cpu_v1.log`）。
- 完整CSA普通/局部语句候选预编译通过，`csa_local_boundaries_compile_v2/{baseline,local}`。
- 相关Python Ruff检查通过。

构建日志：`pypto_local_task_boundaries_build_v1.log`、`pypto_local_task_boundaries_install_v1.log`、
`ptoas_local_task_boundaries_build_v3.log`；CSA编译日志`csa_local_boundaries_baseline_cpu_v2.log`、
`csa_local_boundaries_cpu_v2.log`。上述路径均相对私有目录的`artifacts/`。

**排队。** 完成硬件架构预检查后提交单卡任务`task_20261010_181057_117862815002`，
入口`artifacts/run_local_task_boundaries_v1.sh`。先跑3项基础设备用例，任何失败即停止；
通过后只跑CSA128K/B16的普通/局部语句A/B，逐项比较输出、cache、state、保护区、图重放及更换输入。
性能用30次交替顺序、预热后的单次完整CSA重放，外部NPU event计时，状态恢复在区间外，报告min/mean/max。
NZ2、atomic关闭、无profiler、第二层真实权重。没有16卡任务，HBG未进入范围。
排队时卡由其他任务占用。首个任务在2026-10-10 18:14:35获得0卡、18:14:36退出128，
未进入pytest或CSA，日志仅有锁记录。原因是从非Git工作目录source私有env时，
`load-env.sh`里的`git rev-parse`在`set -e`下退出。修正入口为先cd私有PyPTO再source环境，
CPU执行相同初始化并确认新操作注册可用，重提`task_20261010_181636_150374113561`。
该启动失败不是InCore或TMR失败；不消耗设备重复验证已通过的CPU编译。
两版完整CSA均为73个kernel且任务名集合相同。当前仍无新增设备结果，功能与性能分别待验收。

## 38. 过程 036：局部语句路径单卡功能通过，CSA性能单独记录

`task_20261010_181636_150374113561`于2026-10-10 18:26:51在0卡开始，18:28:20以exit=0结束并释放卡。
它先完成三个普通PyPTO局部语句设备用例，再完成预编译的完整CSA对照；未运行16卡或HBG。
首个任务的环境启动问题见过程035，未重复已经通过的CPU预编译。

**基础功能与真实同核重叠。** 同一套三个独立kernel的运行时choice依次为0/1/-1/0，
分别覆盖A→B、A→C、无后继及再次A→B；输入每次变化，所有数据和Scalar状态精确，未执行分支的输出保护区保持。
AIV/AIC独立任务用例各跑两轮，不通过固定A+B函数入口合组；真实时钟与核号结果如下。

| 诊断 | 第一轮计算重叠 | 第二轮计算重叠 | 在无关A block仍运行时完成等待的B block |
| --- | ---: | ---: | ---: |
| AIV（48 blocks） | 48/48 | 48/48 | 46、46 |
| AIC（24 blocks） | 8/24 | 24/24 | 24、23 |

计数要求同物理核、A函数已返回、B Scalar前缀位于A实际完成等待之前，且计算流水仍有可观测的剩余tick。
只使用设备计时器原始tick，未换算核周期。不同轮次的AIC重叠覆盖率不同，未把编译资格当成每次必有重叠的保证。
全部数学输出、Scalar递推、改输入及保护区均通过。诊断缓冲只由测试harness指定，作者源码没有整体能力属性。

**完整CSA功能。** CANN9.2、TMR、NZ2、atomic关闭、无profiler，第二层真实权重，128K/B16。
普通版及只增加两行局部语句的候选，各自先eager运行再图捕获/重放，然后更换hidden输入重放。
`x_out`、`idx_topk`、`idx_topk_scores`、SWA、compressed、state、indexer key/scale、indexer_state共9项逐bit一致。
每侧31项保护区/metadata检查全部通过，允许slot范围之外的改写字节均为0。
两侧任务名集合相同，均73个kernel；冻结普通包与基线Git `fc6d201b`的活动源码一致。
私有CSA提交只把已测试候选的两行改动落到原`deepseek_v4_flash_csa/decode_o_proj.py`，无helper、算术或流水重写。

**性能。** 预热后30次交替顺序的单次完整CSA图重放，外部NPU event，恢复状态在计时区间外。
该区间为单层HC_pre+norm+CSA+HC_post，与前面单独O-A探针不同；没有把初始化编译计入。

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 同SDK普通CSA | 946.68 | 967.64 | 990.48 |
| 原位局部Set/Wait | 950.50 | 970.21 | 986.74 |

均值增加2.57 μs（+0.266%），最小值增加3.82 μs，最大值减少3.74 μs；候选14/30对更快。
本轮证明功能与同核机制，**尚无稳定净性能收益**。不因最大值下降而声称整体加速，不把旧整体属性路径的计时拼入本轮。
按用户“不做过度测试”的要求，结果齐备后不重复占卡；进一步压缩交接开销属于后续性能工作。

**提交和证据。** 源码均在私有开发环境；中文提交带Signed-off-by：

- PyPTO `b32ee10ff`：局部DSL/IR、能力推导、隐藏ABI、wrapper、六项相关UT、两份当前设备用例与中英文文档。
  gitlink以160000指向Simpler `a1a25dc33`；工作区`runtime`符号链接未被提交成符号链接。
- PTOAS `1696b3e6`：无Python绑定模式的原生CLI构建入口；`75615d31`：局部协议lowering、异步尾部、边界检查与四项正反例。
- 私有vLLM `46597692`：原NZ O-A task仅+2行。公共生产源码不应用该实验协议。
- Simpler沿用`a1a25dc33`，本过程没有新的运行时改动。PTO-ISA保持官方版本。

最终相关CPU UT 6 passed（`artifacts/local_boundary_final_ut.log`），PTOAS正反例4项通过，
设备pytest 3 passed（`artifacts/local_boundary_device_v1.xml`）。
原始结果：私有`artifacts/local_boundary_final_v1/{results.json,local_aiv/ready_results.json,local_aic/ready_results.json}`
及`artifacts/csa_local_boundaries_device_v1/results.json`。
提交内精简结果、全部60个CSA计时样本及两行patch见
[局部语句复现记录](superkernel_local_boundaries_20261010/README.md)。

**第5.8节验收复核。**

| 要求 | 当前证据 |
| --- | --- |
| 原task只插局部语句，无整体属性/helper/ready分支 | 两份普通JIT/SPMD用例源码，原CSA两行diff |
| 从语句推导核类型、隐藏参数和调度能力 | PyPTO生成的AIV/AIC入口与TMR提交；源码不声明能力属性 |
| 保留Scalar位置，危险操作在Wait后 | PTOAS live-value约束、危险前缀反例、设备Scalar/计算重叠与精确结果 |
| 退出处理由后端承担 | EmitC生成的取消返回分支及FileCheck；退出不执行Set或未提交的尾部事件。未另做设备故障注入矩阵 |
| 内部流水保持，真实完成再释放 | 生成代码保留流水Set/Wait；数据依赖、保护区、无后继收尾、同核完成时间戳通过 |
| 原task身份、block数、数学、tiling、流水保持 | 同为73个kernel、相同任务名、CSA源文件仅+2行，输出和全部状态逐bit一致 |
| 先简单PyPTO，再CSA128K/B16单卡 | 同一任务按顺序3项基础用例通过后再执行CSA；任务记录仅0卡 |
| 功能/性能分别记录，过程可追溯 | 以上原始结果、min/mean/max及过程035/036；不声称性能成功 |

本goal限定的原位局部语句demo已实现并验证。全自动寻找插入点（第5.7节）、
扩大到所有CSA task/其他形状、HBG、16卡，以及稳定性能收益均未纳入本阶段完成声明。

## 39. 过程 037：扩展到 CSA 全部可接入单核 task，再做同卡三组对照

用户要求将其余能接入的 InCore task 都加上局部边界，再比较完整 CSA。继续限定私有环境、
CANN9.2、TMR、单卡128K/B16、第二层真实权重；不改变算术、tiling、任务身份或原有sync_start。

原CSA十个文件共新增104行局部语句（私有提交`2cd33a9a`），结合原先NZ O-A的两行，共52处Set/Wait和2处仅Wait。
逐文件移除这两种调用后，AST与修改前完全一致；没有抽取helper、增加作者能力属性或改写计算。
函数已有独立InCore入口时（NZ O-B、Top-K merge/publish），标记插在该原入口，避免在调用层重复插入。
候选整包冻结为私有`deepseek_v4_flash_csa_all_boundaries`，普通和旧O-A候选继续使用既有冻结包。

CPU编译后，三组完整CSA仍各73个kernel，普通版与扩展版任务名集合一致。扩展版覆盖如下：

| 当前128K/B16中的kernel | 数量 | 处理 |
| --- | ---: | --- |
| 独立AIC/AIV，支持异步尾部 | 59 | 局部Wait和Set |
| row_offsets、valid_qk_plan | 2 | 仅Wait，Scalar GM写回仍由正常尾部处理 |
| Indexer Score及QK/PV的混合核成员 | 12 | 保持原协议；对应6个MIX模块，尚未接入本次局部交接机制，正常组完成统计已存在 |

原有sync_start保持：这类后继不触发当前runtime的提前Scalar交接，但仍可作为异步收尾的前驱。
编译能力标记不代表每个实例都会发生重叠。

初版PTOAS只放行少量AIV算术；本次按CSA实际指令扩展普通Vector归约、指数、重排、排序、Vec搬运和
Scalar UB元素访问，保留InsertSync内部依赖。GM Scalar读取只允许在Wait之后；GM写回、跨核通信和
显式流水同步仍不直接放行。首轮CPU编译仅sort32漏在白名单外，补齐后第二轮完成全部二进制编译。
五个PTOAS正反例通过，其中新增“GM Scalar读不能越过Wait”反例；未重跑无关UT或基础设备用例。
PTOAS代码、文档与反例已提交为`f556407f`；本次PyPTO和Simpler没有新增修改。

设备入口`artifacts/run_csa_all_boundaries_v2.sh`使用普通/仅O-A/全部可接入三组，
每组三十次单次完整图重放，循环改变执行顺序，状态恢复在计时外；无profiler。
脚本修正原来只支持两组的循环，并逐候选记录eager和更换输入重放的精确比较。
任务`task_20261010_192303_125266920291`正常排队后在0卡完成，exit=0。
三组eager与图重放均完成，两种候选与普通版的`x_out`、Top-K/score、SWA、compressed、state、
indexer key/scale及indexer_state共9项逐bit一致；更换hidden输入后仍一致。
三组各31项保护区/metadata检查通过。

| 路径 | min（μs） | mean（μs） | max（μs） | mean相对普通版 |
| --- | ---: | ---: | ---: | ---: |
| 普通PTO | 954.34 | 974.90 | 1004.82 | 基线 |
| 仅NZ O-A加入Set/Wait | 955.92 | 970.49 | 1011.26 | -0.45% |
| 全部可接入单核task | 975.56 | 989.89 | 1007.46 | +1.54% |

扩展版均值增加14.99 μs（+1.537%），相对仅O-A增加19.40 μs（+2.00%）。
仅O-A版18/30轮更快，扩展版仅4/30轮更快；扩展版min也增加21.22 μs，
不能归因于某一个最大值样本。P95依次996.17/981.22/1006.24 μs，扩展版相对普通版约+1.01%。
本轮未出现明显孤立长尾证据。仅O-A版小幅收益与之前不同轮次正负摆动并存，不宣称稳定收益。

结论：原位局部语句可扩展至上述单核task，功能通过；全量接入未带来净性能收益，当前配置反而回退。
本次没有新增机制trace，具体哪个task的接续开销或调度变化占主因仍待定位；不能仅凭全量标记推断
实际重叠次数。代码保留在私有实验环境，公共生产路径不应用。结果齐备后不重复占卡。
[精简结果、全部90个样本及源码/编译覆盖清单](superkernel_all_boundaries_20261010/README.md)。

产物均在私有根目录`artifacts/`：
`csa_all_boundaries_source_inventory_v1.json`、`csa_all_boundaries_compiled_inventory_v2.json`、
`csa_all_boundaries_ptoas_checks.json`、`csa_all_boundaries_cpu_v{1,2}.log`、
`csa_all_boundaries_compile_v2/`和`csa_all_boundaries_device_v2/`。

## 40. 过程 038：澄清 MIX 未接入的原因和实际缺口

用户询问为什么MIX不能使用`wait_pre_task_end`/`set_next_task_start`。
修正表述：原理上没有排除MIX；本轮只实现、验证了独立AIC/AIV的交接路径，不能把原型限制说成算子限制。
本节只审查代码和修正文档，没有改动MIX实现，也没有新占卡测试。

私有代码依据：

- PyPTO `b32ee10ff`的`src/codegen/orchestration/orchestration_codegen.cpp`：
  `BuildSingleKernelSubmit`会按局部标记生成`set_scalar_prefix`和`set_allow_early_resolve`；
  `BuildMixedGroupDispatchPlan`目前仅生成普通`MixedKernels`提交，没有对应能力传递。
  双AIV入口还显式拒绝当前scalar-prefix原型。
- PTOAS `f556407f`的`tools/ptoas/ptoas_pipeline.cpp`：`validateAutoSyncTailHints`只放行已审查的
  单核运算集合，显式同步/跨核操作尚未纳入异步尾部协议。实际Score及QK/PV使用`set_ffts`、
  `sync_set`/`sync_wait`，不是增加两个Python调用就能通过现有后端。
- Simpler `a1a25dc33`的`runtime/scheduler/scheduler_dispatch.cpp`：
  `is_scalar_prefix_wait_producer`、ready-prefix分支和`try_scalar_prefix_on_ack`明确要求
  `ResourceShape::AIC`或`ResourceShape::AIV`，MIX无法进入新交接路径。
  同仓`aicore/aicore_executor.cpp`当前按本核M/FIX或V/MTE3事件收尾；
  `runtime/scheduler/scheduler_completion.cpp::complete_slot_task`已经调用`on_subtask_complete`
  汇总成员完成，只有逻辑任务完成后才释放其下游依赖，正常MIX完成统计并不缺失。

应当区分两件事：成员核的Scalar控制交接，以及整个MIX任务的真实完成。
例如AIC已发完矩阵乘指令时，AIV仍可能处理它的输出；AIC可以在满足同核资源和调度条件时先进入后继的安全Scalar前缀，
但不能因此将AIV尚未完成的输出发布为就绪。后继到Wait边界时仍须满足本核尾部完成和自身全部输入依赖；
各成员的真实FIN继续汇总到原组完成统计，跨核同步标志和共享缓冲的使用期必须保留。
这并不等于“必须等整个MIX结束，才能开始后继Scalar”。

后续MIX适配应保留原task、tiling和核间流水：编译器在拆分出的各成员函数里生成适用的局部边界与尾部事件，
调度器支持组内成员的提前Scalar交接，完成处理复用已有聚合逻辑，并检查跨核事件与资源复用。
作者仍使用局部Set/Wait语句，不要求拆helper或给整个incore函数加新的能力属性。
不能仅删除上述白名单/shape判断，就声称MIX功能已实现；过程037的性能数据也不能用于推断MIX交接的收益。

## 41. 过程 039：已有产物解释全量接入为何没有净性能收益

用户要求分析普通974.90、仅O-A 970.49、扩展989.89 μs的差异。本过程只读已有90个样本、
生成C++和私有编译器/运行时代码，不新占卡、不重新编译，也没有修改算子或运行时实现。
结构化证据见[performance_diagnosis.json](superkernel_all_boundaries_20261010/performance_diagnosis.json)。

### 41.1 回退不是一个异常最大值导致

扩展版相对普通版均值+14.99 μs，配对差值中位数+16.01 μs，26/30轮更慢；
三种轮换次序下均值分别+18.20/+11.33/+15.44 μs，方向相同。
仅O-A版本轮均值-4.41 μs，但过程036同口径为+2.57 μs，因此仍不认定稳定收益。
这些是同SDK单层数据，不是Native、整模型或七档比较。

### 41.2 确认一处内部流水退化：预留event7导致PIPE_ALL

此前“原内部流水不变”仅依据源码表达，现修正为“源码表达未改，生成同步需要单独检查”。
扫描59个异步kernel的生成函数体，排除尾部helper并归一化event编号后，
58个内部同步调用序列一致；`mix_x_rms_norm`发生变化。它是HC_pre的独立AIV task，不是MIX task。

普通版此处使用MTE2→V的event0至event7。实验PTOAS的
`SyncEventIdAllocation::GetCompilerAvailableEventIdNum`为异步尾部预留event7，
该流水对的内部可用数量从8变7；`ChangeNoEventIdSyncToPipeAll`在分配失败时删除对应Set并将Wait降级为PIPE_ALL。
生成C++与该降级一致：原event7同步对消失，计算循环中出现`pipe_barrier(PIPE_ALL)`。
精确差异见[内部同步差异](superkernel_all_boundaries_20261010/mix_x_rms_norm_internal_sync.diff)。

新增barrier在`all/build/kernels/aiv/mix_x_rms_norm.cpp:948`，位于0到16、步长2的循环体，
每个执行该路径的block新增8次，位置在两次row-expand计算之间，不是任务末尾的正常排空。
这是已确认的生成代码退化；尚无该task的本轮计时，不能将14.99 μs全部归因于它。
尾部事件策略必须避免挤占内部流水导致这种退化，同时保留事件使用期和真实完成保证。

### 41.3 Scalar-prefix标记改变了原有预派发，而非单纯增加重叠机会

Simpler `scheduler_dispatch.cpp::early_dispatch_shape`的`prefix_pending`分支要求：
消费者有WAIT前驱、无predicate、无sync_start，且仅在同类型前驱所在的pending核上预派发。
该分支会覆盖原先的idle/pending候选集合；找不到匹配前驱时，候选被放回队列，不能继续普通提前派发。

当前T=96的O投影使用MANUAL依赖：`proj_a_mm_0(AIC) → quant_0(AIV) → _proj_b_mm_nz_kernel__2(AIC)`。
这两条边的生产者和消费者类型不同，无法满足`is_scalar_prefix_wait_producer`的同类型条件。
普通版可以在依赖未就绪时预备后继、等待doorbell；扩展版给这些后继加标记后会丧失上述资格，
未预派发的部分只能等依赖完成后走普通ready路径。应保留原预派发，只有可用时才附加同核交接。
此处确认的是代码路径/资格变化；具体本轮延迟多少仍需要实际时序证据。

生成orchestration还显示3处此前未开启的`allow_early_resolve`被Set能力推导开启：
`qproj_matmul`、两个`compress_state_commit`提交点。原来的提前释放策略也发生了变化，
不能将此次A/B解释为只替换末尾等待。清单是包含条件分支的静态提交点，不代表实际task实例数。

### 41.4 固定开销存在，可隐藏的工作和命中机会有限

- 无交接仍支付协议开销：Wait读取回调并判断；Set写入`async_tail_kind`；生成尾部事件链；
  executor探测后继，失败则就地等待计算/写回事件、清零状态并DCCI写回控制行。
  这些动作发生在异步标记task的实际执行中，不只发生在成功重叠时。
- `build_payload`会为标记消费者在AICPU填参数，即使该次最终未发生同核交接。
  这是对原来可在gated AICore等待期间准备参数的策略改变，成本目前未分项计时。
- 本次61个带Wait的独立kernel中，跨Wait保留的显式Scalar值数量为0/1/2/3/4时，kernel数分别为7/37/12/1/4。
  例如`hc_pre_linear_reduce`的显式前缀只有block_idx×16；这不包括wrapper参数展开，
  也不能将活跃值数直接换成指令数或时间，但不能假设所有task都存在足够长的可隐藏Scalar。
- `kv_rms_norm_rope`和T=96分支的`proj_b_act_hc_post_0`带sync_start，当前不能作为提前Scalar后继。
  后者还处于本次CSA区间末尾；不能仅以它带Set就认为能为本次CSA带来后继重叠收益。
- Score和QK/PV的MIX仍未接入；同核交接只在函数返回时尝试一次，后继未及时到达则立即正常收尾。
  59个具备Set/Wait的编译产物并不是59次有效重叠，更不能按kernel数量推断关键路径收益。

结论：现有实现同时引入内部同步退化、预派发限制和固定开销，扩展标记没有带来净收益是可以解释的；
不能据此判定同核Scalar/计算重叠方向无价值。优先修正event分配退化与跨类型依赖的普通预派发回退，
再压缩无交接路径成本。之后只对128K/B16补轻量trace，统计成功交接、后继入口时旧计算剩余量及关键路径变化。
当前常规swimlane/PMU会禁用实验交接，不能用打开该模式后的时序证明优化运行时的命中率。
本过程没有作上述修复、没有新增性能结果；各原因对14.99 μs的定量贡献仍未测得。

## 42. 过程 040：修复内部事件挤占和预派发退化，验证仍无净收益

2026-10-11。继续仅在私有环境推进 TMR，单卡 CSA 128K/B16，不扩大到 HBG、七档或整模型。
目标为消除过程039确认的退化，再压缩未交接成本；仅O-A版作为临时诊断对照，不作为独立长期实现。

### 42.1 修改及原理

- PTOAS `4ef2f93c`：新增 AIV/AIC AllPipes 尾部模式。AIV 发出 V/MTE3/MTE2→S，
  AIC 发出 M/FIX/MTE1/MTE2→S，均为 event7。目标为 Scalar 的内部事件已在 Scalar 到达尾部前消费，
  因而不用在整个函数体预留非 Scalar 事件对。旧 joined 模式保留原预留及旧完成协议。
- Simpler `26c327024`：匹配前驱 pending 核不存在时保留普通 idle/pending 派发候选，
  不再重新入队等待同核机会；正常 gated 路径仍等待所有输入依赖。
  只在实际选中 gated 同核交接时由 AICPU 提前准备参数，其他 gated 情况由等待的 AICore 准备。
- Simpler `58f6581eb`：完成类型 3/4 等待所有上述流水，保留旧类型 1/2，GlobalContext 布局不变。
  真实 FIN、资源复用和后继依赖数据访问均在完成等待之后。
- PyPTO `3d88caef9`：局部 Set 推导新尾部模式及类型；旧显式 async 属性保留旧协议。
  SDK 描述符与 gitlink 同步到 `58f6581eb`，未将工作区的临时 runtime 符号链接提交为链接文件。
  算子写法、任务、tiling、算术、sync_start 与原依赖保持不变。

### 42.2 验证过程与构建问题

真实调度器的 CPU 回归在修复前派发数为0，之后捕获到重复填参；修复后相关7项测试通过。
PyPTO定向编译单测3项、局部边界codegen-only 3项及ruff通过；PTOAS新尾部生成与两项边界拒绝检查通过。
真实`mix_x_rms_norm`的内部同步恢复到8个事件编号、107项同步操作，新插入的PIPE_ALL消失；
完整重编译后59个异步函数内部同步序列均与普通版相同（忽略事件编号重命名）。

Simpler源码修改触发编译器的干净SDK限制，因此先提交私有修复并重建绑定、运行时，同步PyPTO SDK pin。
三组73个kernel均重新编译。首个设备任务`task_20261011_111314_64746019976`在0卡通过
AIV、AIC、动态分支3项设备验证，保留同核计算重叠时间戳和保护区检查。
随后CSA启动被旧`_torch_npu`适配库的SDK版本检查拒绝，尚未执行CSA或采样；已释放设备。
重新配置构建该适配库，CPU确认描述符、Simpler绑定、torch适配库版本三方一致。
该问题属于私有构建遗漏，不作为CSA算法失败或性能数据。

只补跑CSA的任务为`task_20261011_111649_78784228265`，0卡退出0并释放。
命令入口在私有`artifacts/run_csa_all_pipes_v2.sh`，产物在
`artifacts/csa_all_pipes_compile_v2/`和`artifacts/csa_all_pipes_device_v2/`。
以后变更Simpler SDK时须同时更新描述符、绑定、运行时与torch适配库，避免再次占卡发现版本遗漏。

### 42.3 结果及判定

CANN9.2、NZ2、atomic_add关闭，第二层真实权重，含HC_pre+norm+CSA+HC_post。
每组30次单次图重放，三组轮换顺序，reset在计时外，profiler关闭。

| 版本 | min μs | mean μs | max μs | P95 μs | mean相对普通版 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 普通版 | 940.00 | 966.99 | 982.64 | 982.47 | — |
| 仅O-A | 954.66 | 976.13 | 999.10 | 996.28 | +0.95% |
| 扩展版 | 959.22 | 974.89 | 993.98 | 988.52 | +0.82% |

两个候选各9项输出/状态与普通版逐元素精确一致（torch.equal），改变输入后图重放也一致；三组各31项保护区通过。
扩展版23/30配对轮次更慢、配对差值中位数+8.68 μs，仍没有净收益。
不能把旧974.90/970.49/989.89与本轮绝对值的差直接归因于某项修复；两轮的SDK不同。
仅O-A本轮慢9.15 μs，加上之前方向不稳定，不维护独立长期版本；若以后仅该位置有益，也在统一实现内选择位置。
已修复的事件分配和预派发退化有代码及测试证据，但无后继时仍存在Set状态写入、尾部事件和控制行写回成本。
下一步继续缩短无pending后继时的正常返回路径；不将本候选标作性能优化通过，不将目标标完成。

[完整结果、检查与原始样本](superkernel_all_pipes_20261011/README.md)。

## 43. 过程 041：无pending后继时走普通尾部，功能通过但性能回退，已撤回

2026-10-11。停止继续测试独立O-A对照，只比较普通版和扩展版。
在过程040的同一Simpler SDK上，仅修改私有PyPTO wrapper与PTOAS尾部生成：
Set读取DATA_MAIN_BASE，若仍是当前task token则保持普通完成类型，PTOAS尾部走PIPE_ALL；
否则仍发送逐流水异步完成事件，由既有运行时处理交接或排空。算子作者接口不变。
此尝试意在省掉无pending后继时的GM状态写入、事件链及控制行写回，但也引入寄存器探测和条件分支，
且尾部选择普通返回后不会再承接稍后到达的后继，不能预设一定有收益。

CPU编译曾因引用未暴露的platform header失败，无占卡；改为仅比较当前task token后，
PTOAS条件尾部检查、三项局部边界codegen-only以及完整CSA编译通过。
诊断新增普通尾部标识0，记录真实同步完成上界；只有标识7且满足严格时间关系的记录计入计算重叠。

设备任务`task_20261011_113308_139157212552`在0卡退出0并释放：三项基础设备用例通过，
其中AIV两轮普通尾部32/0个、计算重叠16/48对，AIC两轮普通尾部8/0个、计算重叠9/24对。
这些是基础demo证据，不是完整CSA的命中率。
CSA的9项输出/状态及改变输入后重放均与普通版torch.equal一致，两组各31项保护区通过。

| 版本 | min μs | mean μs | max μs | P95 μs |
| --- | ---: | ---: | ---: | ---: |
| 普通版 | 960.70 | 973.75 | 997.58 | 984.25 |
| 条件尾部候选 | 959.28 | 985.54 | 1107.72 | 1002.63 |

同一SDK、CANN9.2、NZ2、atomic_add关闭，第二层128K/B16，每组30次单次图重放、两组轮换，
reset在计时外，profiler关闭。候选均值慢11.79 μs（1.21%），24/30配对轮次较慢；
配对差值中位数+5.49 μs，P95也升高，不能只以min略低保留。
候选补丁与产物归档，已恢复过程040的私有源码及编译器，不修改公共算子，不把目标标完成。

下一步需要直接压缩真实交接协议的状态传递成本，并结合完整CSA的命中证据选择改动；
不再用“假设无pending必然更快”替代测量。优先审视GM完成标记/清零/DCCI能否在未交接时保留于运行时局部状态，
对完成等待、取消分支和旧编译产物兼容分别验证；不改变算子作者的两个局部语句。

[失败候选、统计和完整样本](superkernel_pending_tail_20261011/README.md)。

## 44. 过程 042：调度器预置完成标记，未获得相对原扩展版的收益，已撤回

2026-10-11。停止单独维护、测试仅O-A，保留其历史诊断快照；后续只在统一实现中选择启用位置。
本轮尝试减少未交接时的GM协议成本，未修改算子作者的两个局部语句，未改变数学、tiling、依赖或sync_start。

### 44.1 实现与隔离

私有Simpler `2fb78a6ee` 增加内部提交能力，由调度器按AIC/AIV预置完成类型。
运行时识别高位标记，正常收尾省去清零；只有未被前缀交接或诊断修改的控制行才省去写回，
旧协议仍保留原语义。私有PyPTO基于`3d88caef9`，自动输出提交能力，Set只保留编译屏障，
Wait取消时撤销完成类型，防止等待未发出的事件。PTOAS保持`4ef2f93c`，没有新增算子作者属性。

为避免跨SDK对比，将普通版、原扩展版和预置标记版都重编译到同一候选运行时。
测试驱动只为原扩展版恢复旧Set adapter并去掉预置提交字段。静态检查中67个PTO文件
在一一对应重命名SSA后相同，调度代码除60处提交字段与自动变量名外相同。
直接按文本比较最初被自动变量序号差异拦下，确认是重命名后再比较；未改变计算或调度来通过检查。

8项调度器CPU检查通过（首次新增测试因Arg缺模板参数编译失败，修正后重新构建通过），
3项PyPTO定向代码生成和3项基础场景编译通过。描述符、Simpler绑定与torch适配库的SDK版本
在占卡前确认一致。设备任务`task_20261011_121205_282836328294`在0卡退出0并释放。
3项基础真机用例通过，AIV诊断重叠16/48、48/48，AIC为24/24、24/24；它们不是CSA命中率。

### 44.2 同SDK三组结果

CANN9.2、TMR、NZ2、atomic_add关闭，128K/B16第二层真实权重，含HC_pre+norm+CSA+HC_post。
每组30次单次图重放，三组轮换顺序，reset在计时外，profiler关闭。

| 版本 | min μs | mean μs | max μs | P95 μs |
| --- | ---: | ---: | ---: | ---: |
| 普通版 | 976.26 | 989.97 | 1015.86 | 1005.48 |
| 原扩展版 | 959.40 | 983.40 | 1007.92 | 1002.02 |
| 预置标记版 | 970.04 | 986.54 | 1103.26 | 1002.04 |

两个候选各9项输出/状态与普通版逐元素精确一致（torch.equal），改变输入后的图重放也一致；
三组各31项保护区通过。没有做整模型token测试，也没有额外验证真实退出竞态。

候选较原扩展版均值+3.14 μs（+0.32%），16/30配对轮次更慢，配对差值中位数+0.61 μs。
三个顺序下差值+2.22/−3.99/+11.18 μs，方向不稳定；第5轮出现1103.26 μs最大值。
P95基本不变，不能称为P95整体恶化，也不对这一次最大值作未经验证的机制归因。
候选较普通版均值−0.35%，但原扩展版已经更快，故不能据此接受这项改动。

### 44.3 结论与恢复

撤回预置标记候选。Simpler `9d6757269` 的源码恢复为`58f6581eb`内容，PyPTO恢复
过程040实现，仅更新匹配的SDK pin和gitlink（PyPTO `9daa07428`）；已重建私有前端、绑定、
运行时及torch适配库，三方SDK检查通过。恢复源码与既有验证版本一致，未重复占卡测试。
没有修改公共CSA算子，没有新增独立O-A维护路径，净性能目标仍未完成。

源码删除GM写入并不能单独证明完整CSA受益。下一步先获取CSA实际交接比例及关键路径证据，
再针对真实支出优化，避免把基础demo的交接数量或相对普通版的微弱领先当作候选净收益。

[三组完整样本、功能检查、补丁与复现入口](superkernel_prearmed_tail_20261011/README.md)。

## 45. 过程 043：恢复原调度语义并将普通尾部等待交给执行器

### 45.1 用户修正与实施边界

用户明确拒绝为提高交接概率而改动核选择、ACK后派发或自动启用`allow_early_resolve`。
目标是在正常流程已dispatch的后继上提前执行安全Scalar。无后继须有原流水等待兜底，
无交接开销应基本不变；达不到时先讨论，不能自行接受为“合理代价”。
CSA若发现后继Scalar超过前驱剩余流水的情况，应逐task报告以便审视，不作为普遍回退借口。
前述历史原型的相关改动不符合此范围，后续不再保留。

### 45.2 修改与AscendC参考

- 私有Simpler `a9edc8455`删除匹配busy核覆盖idle核候选、`try_scalar_prefix_on_ack`及其ACK钩子。
  保留正常派发已经选定pending槽之后的交接资格检查、完整fanin等待和真实完成后ACK/FIN。
- PyPTO从局部Set生成内部`runtime_barrier_tail`能力，不再隐式生成early-resolve；用户的显式
  true/false/default保持原含义。没有改CSA的数学、分块、sync_start和任务依赖。
- PTOAS为单核Set入口使用`runtime_barrier_all`合同，内部同步不变，函数尾不新增完成事件。
  AICore执行器在无合格后继时执行原`PIPE_ALL`；交接成功时将其延后至后继Wait，之后才FIN。
  Set不写GM状态，普通收尾不清零/写回控制行，只有真实交接和诊断写入刷新控制行。
  此合同即使取消进入后继也能安全执行PIPE_ALL，不存在等待未发出Event7的问题。
- 复核CANN9.2 A3的`SetNextTaskStartV3Impl`/`WaitPreTaskEndV3Impl`：按配置使用流水Set与
  `wait_flag_dev`，没有逐个Scalar结果的GM保存或广泛编译屏障。PyPTO保留跨Wait所需的
  寄存器依赖，移除每个值的memory clobber；局部Wait去掉原先强制物化常量0的伪依赖。
  安装CCEC拒绝设备端`-S`和`-emit-llvm`，现有objdump显示`<not available>`，故不宣称
  已通过ISA比较证明零指令/零寄存器代价。

### 45.3 验证结果

CPU调度器9项、PyPTO定向codegen 6项、PTOAS正向2项/负向4项通过。
覆盖匹配busy前驱时仍使用idle核、完成合同槽复用，以及同一函数显式false、true、default提交的early-resolve语义。
私有绑定、TMR runtime及torch适配库已重建，SDK一致为`a9edc8455`；PyPTO `d8e8e5935`，PTOAS `d53e3af3`。

`task_20261011_131301_90184725429`在0卡通过3项基础设备检查。
两轮AIV同核Scalar/流水重叠48/48、48/48条，AIC为15/24、24/24条，不能作为CSA交接比例。
新trace记录全部本核流水完成，不能当成独立Vector/Cube完成时间。
同一任务测无后继小图：单个48-block task，每核一个block，无后续任务，交错各30次。

| Vector循环 | 路径 | min μs | mean μs | max μs |
| ---: | --- | ---: | ---: | ---: |
| 1 | 普通 | 81.42 | 89.82 | 119.40 |
| 1 | 修正版 | 83.08 | 93.93 | 109.96 |
| 64 | 普通 | 97.80 | 110.99 | 118.92 |
| 64 | 修正版 | 106.90 | 111.87 | 116.22 |

短图均值增加4.11 μs（4.57%），21/30配对较慢；较长图增加0.88 μs（0.80%），20/30配对较慢。
尚未达到无交接基本无开销，不用最大值下降替代该要求；未将整图差值错误地当作单核开销。

`task_20261011_131501_9774708873`在0卡完成128K/B16第二层、CANN9.2/TMR/NZ2/atomic关闭的同SDK对照。
HC_pre+norm+CSA+HC_post，profiler关闭，warmup后交错各30次单次重放：

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 普通 | 959.04 | 984.39 | 1011.80 |
| 修正版全部局部边界 | 956.18 | 971.36 | 1000.06 |

均值少13.03 μs（1.32%），23/30配对更快；9项输出/状态与改变输入后的重放均精确一致，
两版各31项保护区通过。两份orchestration的63处early-resolve逐项一致；除内部能力字段及
自动变量一一对应重命名外，执行代码一致。

两次任务均退出0并释放设备。无后继开销仍未达标，按用户要求停下进一步调参并讨论；
保留私有功能候选与证据，不切换公共默认，不把本轮CSA加速当成所有路径无回退。
本轮没有采集CSA逐task真实流水完成trace，没有证据将任何CSA task定性为Scalar-bound。

[完整样本、源代码补丁及复现入口](superkernel_original_dispatch_20261011/README.md)。

## 46. 过程 044：以无后继短任务定位固定开销

用户追加目标：以无后继、短任务为基础定位额外开销，目标是保证无额外开销。
本轮未新测CSA/HBG/整模型，仅使用私有环境和单卡；不更改原派发、ACK、early-resolve和数学。

### 46.1 计时分层与归因修正

原有单次Event口径含Host分别提交start/replay/end的空档，短图的+4.11 μs不能全部归为单核协议。
增加32次串行调用均摊后，首轮Wait+Set较普通多1.125 μs，Wait单项多0.992 μs；
但下一轮对照没有稳定复现，撤回“已定位约1 μs间接调用成本”的推断。
48-block核内采集还缺39/1920条记录且受并行访存影响，因此进一步使用spmd(1)。
单block三组采集分别完整取得120/120、90/90、90/90条记录。

`task_20261011_134526_223926230311`：单block各20次，普通核内均值1.309 μs，
Wait为1.623；只保留空检查、删除间接调用仍为1.623；删除控制字段读取/回调为1.374。
证据不支持直接把间接调用/寄存器保存定为主因，优先处理无交接时的控制字段访问。
简化控制只用于无后继诊断，不能作为真实交接实现。

### 46.2 私有候选与结果

Simpler `0de0c4c62`在AsyncCtx原有4字节对齐空隙保存实际交接标志，结构大小/已有字段偏移不变。
普通dispatch归零，只有真实交接时AICore置位；Wait先读LocalContext热行，置位时才调用原回调。
真实交接结束后清零并在FIN前写回热行。该写回是新增实际交接成本，尚未验证其净性能收益。
PyPTO `4a23fb1e9`接入热行判断并增加隔离诊断驱动；PTOAS仍为`d53e3af3`，没有改动。
所有安装和实验产物留在私有根，公共路径未切换。

`task_20261011_135356_26315003897`，同SDK中用旧/新Wait生成代码，单block各30次：

| 路径 | min μs | mean μs | max μs |
| --- | ---: | ---: | ---: |
| 普通 | 1.240 | 1.381 | 1.900 |
| 原控制字段标记 | 1.340 | 1.558 | 2.360 |
| 热行标志 | 1.360 | 1.490 | 2.080 |

热行候选比旧标记少0.068 μs，仍比普通多0.109 μs。该核内profiling会关闭实际交接探测，
不能据此宣称非profiling尾部路径零成本。同任务的48-block无profiler单次图均值
普通106.273、候选106.302 μs（+0.029）；32次均摊普通70.912、候选71.241（+0.329）。
单次候选最大值也从111.340升到117.380，不能挑均值接近的一项代替完整验收。

CPU调度器9项、定向codegen 3项、设备基础功能3项通过；输出和保护区正常。
基础trace两轮AIV观察到18/48、48/48条同核Scalar/全流水重叠，AIC为24/24、24/24。
这些不是CSA的交接率，本轮未确认任何CSA task为Scalar-bound。

### 46.3 未保留的尝试与剩余工作

- 去掉Wait开头重复memory clobber，在1/48-block短例生成机器码相同，故未重复上卡；实现保留原屏障。
- 内联同一个runtime完成回调，kernel文本456→856字节；
  `task_20261011_140154_300542529112`核内均值普通1.468、热行1.536、内联1.537 μs。
  无收益，不接入。诊断最初因include缺失编译失败；首次上卡加载又因include配置不一致
  触发编译失败（`task_20261011_140015_292719532146`，未执行设备算子），修正后才取得数据。

最近两轮核内仍有0.07–0.11 μs额外开销，目标未完成，不把它当作用户已接受的代价。
下一步审查内部调用约定能否直接传递executor已知状态，避免Wait访存；同时拆开尾部探测和原流水等待。
若方案需要接受新的运行代价，按用户第5条先讨论，不自行接受。
全部任务已终止并释放卡。[完整数据、失败记录、补丁与复现](superkernel_no_successor_overhead_20261011/README.md)。

## 47. 过程 045：内部传参移除 Wait 访存，仍未达到无额外开销

接续无后继短任务定位，不新测 CSA/HBG/整模型，也不改变原派发、ACK 或 early-resolve。
先用第二个物理入口参数传递 executor 已知状态（Simpler `946011175`）；
虽然删除 Wait 的 GM 控制字段读取，但执行器额外查询入口 ABI，因此未作为最终入口方案保留。
随后恢复单指针物理入口（Simpler `66d6ce63e`），仅在真实交接分支用 args 对齐低位传递状态；
wrapper 在任何解引用前清位，内部 Wait 直接使用 Scalar 状态。普通 executor 不查询 ABI，
新交接路径也不再写回 LocalContext 热行。作者仍只写局部 Wait/Set，PTOAS 负责转发隐藏参数。

两次单卡任务分别为 `task_20261011_142723_416298025094`、
`task_20261011_145252_116220028054`，均退出 0、释放 0 卡。各通过 3 项设备功能检查，
两轮都观察到 AIV/AIC 的实际同核交接；最新 CPU 调度器 9 项和定向 codegen 检查通过。
每次单block核内窗口完整取得 90/90 条记录；每版 30 次，单位 μs：

| 实验 | 普通 min/mean/max | 同 SDK 热行控制 min/mean/max | 新状态路径 min/mean/max |
| --- | --- | --- | --- |
| 第二物理参数 | 1.380/1.534/2.200 | 1.520/1.807/2.460 | 1.540/1.711/2.340 |
| 指针交接位 | 1.360/1.539/2.180 | 1.500/1.713/2.520 | 1.400/1.605/2.460 |

最新候选比同轮热行控制少 0.107 μs，但仍比普通多 0.067 μs。无 profiler 的48-block检查，
32次图内调用均摊普通69.252/71.782/74.931，候选70.434/72.134/74.483 μs，均值多0.352。
单次 Event 均值虽从102.984降到101.363 μs，但含Host提交影响，不能替代零开销验收。
同SDK普通基线也不能证明SDK共同路径没有额外成本。未定位任何CSA task为Scalar-bound。

用户指出迭代过慢后，本轮不再追加小变体。当前仍是私有候选，不切公共默认，目标未完成。
下一步优先审查编译器生成独立普通入口的可行性，避免无交接路径携带Wait分支；
这只是后续方向，代码体积、注册和完成语义尚未验证，不自行接受现有负收益。
[原始样本、功能记录、两个候选补丁与细节](superkernel_no_successor_handoff_abi_20261011/README.md)。

## 48. 过程 046：接受固定开销，保留指针交接位方案

尝试把可能调用完成回调的路径拆进独立冷函数，普通入口仍根据交接位选择路径。
CCEC的`-fstack-usage`显示普通短task栈帧0字节、原指针交接位版本64字节、冷函数普通入口16字节。
`musttail`可编译但仍为16字节，没有保留；栈帧差异不是全部时间差的归因证明。

任务`task_20261011_150410_169423030812`在0卡完成，3项功能检查通过，单block记录60/60完整。
每版交错30次：普通核内min/mean/max为1.340/1.485/2.100，冷函数1.400/1.747/2.240 μs；
核内仍多0.262 μs，文本456→752字节。48-block单次Event均值103.807→104.258，
32次图内均摊71.858→70.864 μs；不同口径不一致，不能挑后一项认定普通路径开销被消除。
实际AIV/AIC交接通过，任务退出0、设备释放，未新跑CSA或整模型。

随后编写的独立物理双入口方案尚未构建和上卡，用户明确要求停止继续追零成本，
从已有有效方案中选择低开销候选保留，并转向其他优化空间。
已撤掉冷函数与全部未验证双入口改动，恢复以下源码与已安装私有二进制：

- PyPTO `320beb494`：原单指针入口解码实际交接位，Wait使用Scalar状态。
- Simpler `66d6ce63e`：只在真实交接的原ACK分支携带状态，保持原dispatch与early-resolve。
- PTOAS `4a5ce22a`：转发隐藏状态，保留Scalar前缀和取消处理。

选择依据是有效功能候选中较低的单核额外记录、无普通Wait的GM标志读取、
无新交接热行写回，以及比冷函数更小的代码体积。保留版同轮单block均值1.539/1.605 μs，
额外0.067；32次均摊71.782/72.134，额外0.352。不同轮次不能证明绝对全局最优。
用户接受这一阶段的固定代价，原零开销goal已暂停；没有将零开销目标标成已实现，也没有证明其理论上不可能。

后续源码审查发现三个候选点，尚未实施或宣称收益：

1. Wait回调已清理并写回控制行，同步后继返回分支再次写回；先限定无trace及无后续控制写入的场景。
2. 后继资格检查前做全D-cache操作；审查最小控制行同步与真正交接时的参数可见性如何分开。
3. executor、完成等待helper重复读取volatile tail/trace状态；只考虑复用已读状态，保留所有原流水等待。

公共安装和CSA未切换，其他session改动未动。[原始样本、栈帧记录及选择细节](superkernel_no_successor_handoff_abi_20261011/README.md)。

## 49. 过程 047：先核对历史，只补两处小范围协议清理

用户要求已优化过的点跳过，未处理的尽力优化，不投入过多精力。核对Git记录：

| 候选点 | 既有工作 | 本轮处理 |
| --- | --- | --- |
| 控制行写回 | `4cd49d3a7`已跳过普通无交接路径写回，后续也已简化无交接尾部 | 只跳过局部Wait已清理控制行后，同步后继无trace返回时的第二次写回；旧接口及诊断保留 |
| 全D-cache操作 | `b39d9e7c7`已跳过成功交接后的第二次全cache操作，当前仍保留 | 不重做；失败候选的探测策略涉及可见性，暂不展开 |
| 重复状态读取 | 主循环读过tail kind，drain helper仍再读一次 | 直接传入已读kind；trace读取及其他入口不扩展 |

Simpler `d00f8f421`仅修改一个执行器文件并补充文档；PyPTO `1f8206d09`仅同步SDK pin/gitlink，
PTOAS仍为`4a5ce22a`。入口保持已选定的指针交接位方案，不改数学、流水等待、真实FIN、
原ACK流程、派发与early-resolve设置；无交接普通路径没有新增资格判断。

私有binding/TMR runtime/torch adapter重建成功，版本一致。
任务`task_20261011_153654_318500013205`，单卡0、CANN9.2/TMR，现有基础功能3项通过：
分支后继及无后继、输出/保护区和重放检查正常；AIV两轮实际重叠16/48、48/48，AIC为8/24、24/24。
任务退出0，设备释放。未重测CSA或展开性能矩阵，因此只确认去掉了冗余操作及功能未回退，
不报告本轮CSA净收益，也不将66d6版本的0.067μs旧数据标为本轮新成绩。

[测试摘要](superkernel_handoff_cleanup_20261011/summary.json)、
[功能报告](superkernel_handoff_cleanup_20261011/functional.xml)、
[修改补丁](superkernel_handoff_cleanup_20261011/simpler.patch)及原始轨迹、任务日志均已保存。

## 50. 过程 048：当前 SDK 下完成七档普通边界与 early_start 对照

用户要求基于当前修改测试七组CSA，覆盖此前只测128K/B16的限制。
本轮未再修改kernel或运行时，只给原单卡驱动增加batch/history参数，使用同一份当前SDK：
PyPTO `1f8206d09`、Simpler `d00f8f421`、PTOAS `4a5ce22a`，私有CSA源码基点 `2cd33a9a`。
两版63处early-resolve设置逐项一致；七档共用动态程序，每版仅预编译一次。

任务`task_20261011_154838_375836631548`在2卡完成，退出0并释放卡。
CANN9.2/TMR/NZ2/atomic关闭，第2层真实权重及合成历史；区间含HC_pre+norm+CSA+HC_post。
profiler关闭，warmup后两版轮换各30次单次图重放。单位μs：

| 档位 | 普通 min/mean/max | early_start min/mean/max | mean耗时减少 |
| --- | --- | --- | ---: |
| 128K/B4 | 638.20 / 653.38 / 673.54 | 636.78 / 656.31 / 683.98 | -0.45% |
| 128K/B8 | 745.80 / 760.03 / 783.62 | 748.22 / 758.13 / 772.84 | +0.25% |
| 128K/B16 | 970.28 / 999.46 / 1052.56 | 965.32 / 984.49 / 1008.92 | +1.50% |
| 128K/B24 | 1199.06 / 1248.08 / 1283.14 | 1203.12 / 1260.67 / 1294.46 | -1.01% |
| 8K/B16 | 726.90 / 746.48 / 768.92 | 715.98 / 737.04 / 758.92 | +1.26% |
| 8K/B24 | 873.24 / 923.16 / 991.34 | 884.72 / 928.95 / 962.28 | -0.63% |
| 8K/B32 | 1013.96 / 1075.10 / 1116.78 | 1018.04 / 1066.00 / 1113.32 | +0.85% |

七档均值四快三慢，最大值五降两升；128K平均减少0.0726%，8K减少0.4944%，
组内档位等权、长短档8:2加权仅减少0.1569%。B16长档25/30配对更快，
B24长档21/30配对更慢，不能把全面加标记定性为普遍加速。
七档两版9项输出/缓存/状态、改变输入后的重放逐bit一致，两版每档各31项保护区通过。

普通版不插入局部Wait/Set，可以避开此前单block“有标记无后继”约67ns的额外成本；
该数字不是每个CSA task的常数，同SDK对照也不能排除公共运行时相对旧SDK的成本。
本轮没有新测Native或整模型，没有用总耗时认定Scalar-bound，没有切换公共默认。
零开销goal仍暂停；本次仅完成最新授权的七档测量与归档，不额外增加优化轮次。
[完整报告、420次样本及复现脚本](superkernel_csa_seven_handoff_cleanup_20261011/README.md)。

## 51. 过程 049：只读分析 early_dispatch 到 early_start 的覆盖缺口

用户要求先从理论分析有多少提前派发task应产生Scalar重叠收益。本轮没有占卡或修改运行时。
明确此前七档只测了整体耗时，没有记录实际early_dispatch、handoff及有效重叠次数；
不能用63处early-resolve设置或73个kernel定义替代实际命中数。

按128K/B16、T=96实际分支展开生成orchestration，并展开8个O projection组：
共66个AICore任务组、973个单核block调用（MIX一block按1AIC+2AIV计），不含AICPU dummy。
其中64组有Wait；2组MIX未接入，144个单核调用被排除。
当前runtime还排除sync_start后继，当前分支中的kv_rms_norm_rope和HC_post共51个block，
所以入口能力层最多剩62组、778个单核调用可作为后继。以上是静态资格上界，非实际交接数。

源码确认更强的额外限制：gated pending后继要求本核前驱恰好是它同类型的直接WAIT生产者，
仅“后继已经正常dispatch到本核”仍不足。该条件比用户所要求的范围窄；
安全Scalar可先执行、在Wait先完成本核前驱再等自己的完整fanin，从原理上不必要求数据生产者
就是本核前驱。O-A→quant→O-B等跨资源链因此不能天然转化为当前gated交接收益。
实际fanin可能含额外依赖，不仅凭显式链断言整类task全部未交接。

进一步核对pending选槽：`CoreTracker::get_pending_core_offset_states`按核类型、忙闲及槽位选择，
`stage_consumer_blocks`从可用槽取`pop_first()`，均未要求本核当前task是消费者的数据生产者。
因此“排在本核X之后、数据依赖另一核A”是源码允许的情况，不能混同排队次序与数据依赖。
修正交流中“主要等待另一核就被挡”的表述：只有X不属于直接WAIT生产者集合时才被此条件拒绝；
若X与A均是直接生产者，等待A本身不触发此拒绝。没有将源码允许的情况冒充本轮实测命中。

另有idle预派发不进入算子Scalar前缀、MIX前驱不支持异步交接，以及只在前驱返回时一次探测后继的限制。
这些是覆盖与窗口问题，不能当作early_start本身收益上限，也没有证据认定任何CSA task为Scalar-bound。
理想局部节省为min(剩余前驱流水时间,安全Scalar前缀时间)减协议成本；
多核同时节省的时间不能直接相加为CSA终点收益。

下一项所缺证据是idle/pending提前派发、资格拒绝原因、真实handoff、有效重叠窗口与关键路径覆盖的计数。
本轮只记录缺口与理论判断，没有新测试，也未改用户的dispatch、ACK、sync_start或early-resolve。
[详细清单、源码位置及静态展开数据](superkernel_csa_seven_handoff_cleanup_20261011/early_dispatch_coverage.md)。

## 52. 过程 050：移除本核前驱必须是直接 WAIT 生产者的额外限制

用户指出该限制不符合已正常派发后继的交接目标，本轮将其移除。
Simpler `5c8418a02`只修改派发时的交接资格判定：以真实running槽中的同资源类型任务
确定本核流水前驱，不再遍历消费者fanin或锁定前驱fanout来证明直接数据依赖。
消费者的`src_payload`和完整fanin门保留；局部Wait仍先排空本核流水、发布本核前驱真实FIN，
随后等待消费者自己的全部数据生产者完成，再ACK和执行计算。
不改核选择、原派发队列、ACK后流程、allow_early_resolve或sync_start。

先补C++回归，在旧实现上复现AIC/AIV两种无直接依赖本核交接均被拒；修改后相关10项单测通过。
私有PyPTO `fdac65f66`同步SDK pin/gitlink并增加局部Wait/Set写法的单卡用例；
PTOAS仍为`4a5ce22a`，本轮没有编译器或CSA算术修改。
私有binding、TMR runtime及torch adapter已重建，SDK版本一致。

新用例数据图为`AIC A → AIV B`，另一组AIV X与A、B没有数据依赖。
X有48个block，B有48个block；B按原调度排在X后面，数据只等待另一核A。
任务`task_20261011_165028_24449843490`在单卡0运行，CANN9.2/TMR，退出0。
两次改变输入的执行中，均有**48/48个B安全Scalar与无关本核X尾部重叠**；
两轮全部48个前缀也均在A计算完成前结束，而所有B均在A真实完成后才越过Wait。
输出逐元素精确一致，输出及trace保护区保持不变。

首次任务`task_20261011_164849_237931830320`在单卡1完成既有三项回归：
分支/无后继、AIV独立后继、AIC独立后继均通过，后两者两轮重叠分别为27/48、48/48和9/24、24/24。
同批新用例因单block SPMD未显式读取block索引，在前端解析时失败，未进入设备执行；
补上索引后只重跑新用例，即上面的成功任务。保留首次失败日志，不冒充首轮全通过。

该结果证明先前限制确实会排除合法交接，并证明去除后完整数据等待仍有效；
48/48是构造用例的命中率，不能作为CSA的命中率。本轮未重测CSA性能或七档矩阵，
此前七档数据仍属于Simpler `d00f8f421`，不得改标为新版本成绩。所有设备任务已结束。
[摘要、原始时序、补丁及测试日志](superkernel_unrelated_pending_20261011/summary.json)。

## 53. 过程 051：解除直接生产者限制后的七档复测与剩余覆盖范围

用户要求重新测试七档，并再次明确“只要已early_dispatch，就应允许early_start”。
本轮不再修改运行时或kernel，保持过程050的PyPTO `fdac65f66`、Simpler `5c8418a02`、
PTOAS `4a5ce22a`；沿用私有vLLM-Ascend `2cd33a9a`下的两份冻结CSA包及原测量驱动。
普通版没有局部Wait/Set，候选版启用已有边界，两版63处allow_early_resolve设置一致。
这是同一新SDK下的两条PTO路径对照，不是Native对比，也不是旧SDK/新SDK交错对比。

任务`task_20261011_170750_320296621605`在单卡2完成，退出0、设备释放。
CANN9.2.0-beta.2、TMR、NZ2、atomic_add=0、deterministic_level=0，第2层真实权重及seed1024合成历史。
区间为HC_pre+norm+CSA+HC_post，含运行时调度；profiler/PMU/参数dump关闭。
每版warmup5次，轮换先后顺序各30次单次NPUGraph重放；缓存恢复不计时。
七档共享动态程序，每版只编译一次；420次原始样本全部保留，没有按结果筛选或追加轮次。

| 档位 | 普通 min/mean/max（μs） | early_start min/mean/max（μs） | mean耗时减少 |
| --- | --- | --- | ---: |
| 128K/B4 | 633.16 / 653.30 / 678.54 | 625.40 / 645.70 / 668.24 | +1.16% |
| 128K/B8 | 749.06 / 772.00 / 827.60 | 745.92 / 755.68 / 764.04 | +2.11% |
| 128K/B16 | 962.98 / 983.60 / 1001.54 | 958.50 / 984.32 / 1003.54 | -0.07% |
| 128K/B24 | 1201.76 / 1251.98 / 1290.02 | 1187.38 / 1241.17 / 1286.16 | +0.86% |
| 8K/B16 | 733.74 / 757.86 / 788.38 | 718.48 / 740.16 / 760.32 | +2.34% |
| 8K/B24 | 886.82 / 919.19 / 969.92 | 906.46 / 930.81 / 954.94 | -1.26% |
| 8K/B32 | 1027.30 / 1063.97 / 1108.66 | 1020.38 / 1067.32 / 1115.72 | -0.31% |

均值四档改善、三档回退；max五降两升。128K组内百分比等权平均减少1.0164%，
8K减少0.2525%，长短档8:2加权减少0.8636%。128K/B16均值基本持平但略慢，
8K/B24均值慢1.26%而max少14.98μs，B32均值和max均回退，不能仅报告加权正收益。
上一轮加权减少0.1569%，但两轮普通基线也有变化；不能相减声称本次解除限制净赚0.7067个百分点。

七档9项输出/缓存/状态及改变输入后的图重放逐bit一致，每档两版各31项保护区检查通过。
本轮没有新测Native、整模型token或实际early_dispatch/handoff计数；
不把源码静态数量、总耗时差或最小用例48/48命中率当作CSA的实际覆盖率，也不据此认定Scalar-bound。

复核源码后，将以下两类条件明确分开：

- 必要安全条件：提前执行部分不读取未就绪输入、不复用前驱仍占用的资源；
  在Wait处完成本核流水并等待自己的完整fanin，之后才能执行依赖计算。
  前驱若已经完全同步结束，就没有本核Vector/Cube剩余窗口，但不应据此禁止后继的安全前缀先行。
- 当前覆盖不足：仅pending同类型纯AIC/AIV可交接，idle核只预备运行时参数；
  MIX及sync_start后继未接入；只在前驱函数返回时探测一次，错过后到任务；
  泳道/PMU/参数dump开启时关闭交接。这些不是理论限制。

predicate消费者本来就不走原early_dispatch，不将其计作新增限制。
后续扩大覆盖仍须保持原核选择、派发/ACK流程、allow_early_resolve与sync_start计算同步语义。
本轮完成测量和约束说明，没有顺带扩大实现或增加优化轮次；公共默认不变。
[完整报告、420次样本、任务日志及复现脚本](superkernel_csa_seven_unrelated_pending_20261011/README.md)。

## 54. 过程 052：接入 idle 与 sync_start 前缀，优先验证 128K/B16

用户将原则设为新goal：所有已early_dispatch且有安全Scalar前缀的task均应获得提前执行机会，
MIX等场景一并考虑，性能优先测128K/B16。本阶段先完成两处覆盖缺口，不将其替代完整目标。

Simpler `8171b78c5`增加gated安全前缀能力，idle核仍自行准备原运行时参数，随后进入前缀；
局部Wait无本核前驱时仅等待自己的完整fanin/cohort doorbell。有前驱时仍先排空流水并发真实FIN，
使原pending晋升可完成sync_start整组rendezvous，再由原doorbell放行计算及ACK。
原核选择、派发、ACK后流程和用户early-resolve均不改。PyPTO `19f2c091a`同步SDK pin/gitlink，
扩展跨核依赖用例并修正中英文文档，PTOAS保持`4a5ce22a`。

idle及sync_start pending定向CPU用例在旧逻辑下失败，修改后相关10项调度检查通过。
四种组合的CPU全编译通过。任务`task_20261011_173025_10211416583`在单卡0通过7项设备测试：
idle/pending×普通/sync_start的四组，两次改变输入均48/48个前缀早于远端完成；
pending两组均48/48与本核前驱尾部重叠，计算未越过输入完成或sync_start全组本核完成。
输出和保护区通过，原AIC/AIV、分支及无后继回归通过。任务退出0，设备释放。

CSA首次任务`task_20261011_173201_16262712771`在预编译产物加载阶段被精确clean SDK校验拒绝，
没有执行CSA。随后提交并同步版本、重建私有binding/runtime/torch adapter，在CPU重编译两版后，
任务`task_20261011_173903_4357828989`于单卡2完成128K/B16并释放卡。
同SDK普通/early_start，CANN9.2/TMR/NZ2/atomic=0，profiler关闭，warmup5次后交错各30次；
第2层真实权重与合成历史，区间含HC_pre+norm+CSA+HC_post，缓存恢复不计时。

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通 | 971.10 | 987.81 | 1012.80 |
| early_start | 955.02 | 979.30 | 1017.04 |

均值少8.51μs（0.8619%），max多4.24μs；9项输出/缓存/状态及改输入重放逐bit一致，
两版各31项保护区通过。没有用不同SDK、不同时间的两轮差值定量归因idle/sync_start改动，
也没有把构造用例48/48当作CSA实际交接率。功能与性能分开，当前不声称全面性能改善。

MIX最小局部Wait/Set用例已在CPU编译入口复现统一拒绝，尚未进入设备。
下一步需要按AIC/AIV0/AIV1传递成员能力，扩展Group提交及PTOAS跨核pipe合同，
保持pipe初始化在Wait之后、每个成员的真实FIN与原整组完成计数。不能简单开启整组布尔标记。
单次探测窗口及诊断路径仍为开放项；本轮没有修改公共默认或占用16卡。
[完整样本、时序、补丁和失败记录](superkernel_prefix_coverage_20261011/README.md)。

## 55. 过程 053：MIX成员能力与局部边界接入

2026-10-11，私有Simpler `a2df6bf24`、PyPTO `e905f40e5`、
PTOAS `52b8eaf1`。用户明确本轮不做自动前缀识别，诊断模式串行回退保留。

Simpler把整组布尔能力改为AIC/AIV0/AIV1位图，解除纯核/同资源形状限制，
保持每成员FIN与既有组完成统计。PyPTO补齐Group/双AIV提交能力，并修复隐式
参数加载被提到Wait前的问题；pipe初始化也放到Wait之后。PTOAS仅在
runtime_barrier_all下接入跨核pipe、原有FFTS同步及相应tile操作。

CPU首次编排失败还受私有新二进制未安装影响，安装后3项通过；参数加载2项通过，
调度11项通过。连续MIX单卡两次均24/24 AIC、48/48 AIV提前进入Scalar，
输出/保护区正确。基础用例旧JIT调用方式和旧sync_start全图X断言分别修正，
两失败项定向复测通过。原失败缺完整时序，记录不据此虚构具体调度原因；
后续断言前保存完整时序。详见原始XML及独立证据目录。

私有CSA整包复制后，在Indexer两条Score实现和稀疏QK/PV各加入Wait/Set，
共6条语句，不改数学、tiling、核数或early-resolve。73个编译成员全部有Wait，
其中12个MIX成员全部有Set/Wait；静态覆盖不等于运行时重叠命中。

单卡1任务`task_20261011_182414_219668930373`，每条路径warmup5、轮换30次：

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通路径 | 972.12 | 989.03 | 1014.58 |
| 原 early_start | 951.24 | 981.37 | 1009.28 |
| early_start + MIX | 962.82 | 992.01 | 1014.60 |

新增MIX相对原early_start均值回退10.64μs，max增加5.32μs；9项状态和换输入
重放逐bit一致，保护区通过。功能保留，稳定净收益尚未成立。源码发现gated
交接会把原本AICore可完成的参数准备额外搬到AICPU，下一步移除这项工作再比较。
[时序、90条样本、版本、补丁和复现](superkernel_mixed_prefix_20261011/README.md)。

## 56. 过程 054：gated后继参数准备恢复到AICore

Simpler `3f86962a7` 去掉为交接额外加入的 AICPU 参数写入；gated后继一律由接收它的
AICore在进入函数前准备参数，保持ready路径、派发顺序、完整fanin和sync_start/ACK语义。
PyPTO `58ccfcc72` 同步SDK，PTOAS和CSA源码不变。11项CPU检查、9项单卡功能检查通过。
MIX两轮24/24 AIC及48/48 AIV前缀提前；AIV各48/48交接，本轮局部尾等待均不超过5tick，
不能据此把真实CSA归为Scalar bound。纯AIV、纯AIC及跨核依赖的原重叠验证也通过。

单卡0任务`task_20261011_183915_2833602685`，退出0并释放设备，128K/B16：

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通 | 960.10 | 982.46 | 1004.46 |
| 原early_start标记范围 | 959.00 | 982.22 | 1001.40 |
| early_start+MIX | 959.30 | 979.16 | 1002.42 |

各30次轮换单次图重放，配置与区间同过程053。MIX较本轮普通均值减少3.31μs（0.34%），
max减少2.04μs，18/30配对更快；较本轮原标记范围均值减少3.07μs但max增加1.02μs。
9项输出/缓存/状态及换输入重放逐bit一致，各31项保护区通过。保留代码清理与功能；
不能将跨卡跨轮992.01→979.16μs全部解释为该项因果收益，稳定净收益尚待后续证据。

同时仅在CPU检查后到任务的探测方案：CANN9.2虽声明try_wait，两参数调用在
A3 dav-c220-vec被编译器明确拒为目标不支持；未把它接入runtime或进行设备试验。
单次探测限制仍存在。自动识别与诊断模式重叠均按用户最新决定不做。
[完整时序、90条样本、补丁、失败探针和复现](superkernel_aicore_args_20261011/README.md)。

## 57. 过程 055：真实CSA前缀覆盖及完成查询探针

SDK保持过程054版本，私有驱动只在生成kernel wrapper时加入trace，没有改变任务依赖、
early-resolve、调度、算术或tiling。任务`task_20261011_185759_356175720609`在单卡0退出0并释放卡。
128K/B16两轮改变隐藏输入，9项输出/缓存/状态逐bit一致，各31项保护区通过。

两轮各973个物理成员实例；gated提前派发551/586次，均进入有资格提前执行的前缀回调路径。
实际同核交接210/237次，其中148/168次到Wait时仍有大于5tick的尾等待。
主要交接在投影链，O-A各64/64交接；Score和QK/PV的MIX成员进入gated前缀路径，
但本次CSA没有观察到它们的同核交接。机制支持不能替代实际到达时序证据。

72/108条有pending资格却没有交接，其中7/12条的本核前驱同步收尾，原本没有异步窗口。
其余65/96条缺AICPU派发时刻，无法区分探测之后计算结束之前到达，还是流水已结束但
CPU尚未处理FIN时派发。不能把它们全部计作可挽回收益；也没有充分证据认定某task为Scalar bound。
插桩增加读钟和GM写入，本轮不报告新的CSA性能，当前无插桩结果仍为过程054。

CPU探针确认get_status/get_condition_flag可编译，但未找到完整流水完成契约，未用于runtime。
A3 Cube侧“各流水尾部接到MTE3，再由L1写GM完成标志”也通过编译；缺pad_t的首次错误保留。
这是待验证候选，尚未上设备，下一步只在小用例检查可中断等待及无后继时的开销。
不改变AICPU派发，不把一次编译成功等同于设备正确性。源码枚举RuntimeBarrierAll为5，
旧说明中的6为文字错误，不是本轮协议改值。

[完整原始时序、分类、驱动、编译失败记录与复现](superkernel_csa_prefix_coverage_20261011/README.md)。

## 58. 过程 056：AIC/AIV有序DMA完成通知上板及成本探针

私有PyPTO仅新增测试驱动`17377846a`；Simpler/PTOAS及SDK不变。驱动在生成wrapper时
为小用例加入独立scratch和GM通知标志，保持原AICPU派发及ACK流程，算子侧仍为局部Wait/Set。
AIC的MTE2/MTE1/M/FIX、AIV的MTE2/V分别有序接到MTE3，最后由DMA写标志。
三路均保留诊断与末尾完整PIPE_ALL，分别比较普通等待、通知后等待、通知后轮询。

任务`task_20261011_192846_151447026363`在单卡0退出0并释放设备。
两种计算长度、两类核、三路径各两组输入输出逐bit正确，输出/trace保护区通过；
轮询均读到完成标志后退出，显式末尾等待前的输出word采样也正确，之后图重放完成。
该结果限于此小例子，不等于所有操作的完整流水合同或后到任务已经成功交接。

每路预热5次，再轮换各15次单图重放，以下三元组为min/mean/max（μs）：

| 小例子 | 普通等待 | 通知后等待 | 通知后轮询 |
| --- | --- | --- | --- |
| AIV，loops=1 | 110.00/114.48/125.28 | 101.84/113.08/121.02 | 108.46/115.72/120.68 |
| AIV，loops=64 | 114.74/119.34/124.76 | 115.56/121.56/131.68 | 114.84/120.63/124.62 |
| AIC，loops=1 | 99.08/106.44/116.32 | 102.94/109.37/117.66 | 102.66/107.13/112.74 |
| AIC，loops=64 | 119.46/136.71/148.40 | 120.58/138.29/150.60 | 123.12/137.85/150.18 |

四组轮询mean多0.69～1.28μs，但端到端波动明显，不能当精确固定开销；也不把通知版
偶然更快当优化。未直接接入CSA，下一步先在两task例子证明初次探测未到、计算未完时
后继到达能够获得新增交接，并保持已有立即交接路径。正式scratch/事件预留尚未实现。

CPU阶段先修正Acc/Vec变量重名、探针ABI参数排序及Tensor描述符取地址；错误产物未上设备。
三次无设备计算的启动失败分别为ATB脚本与set-u冲突、错误的帮助探测提交、不在Git目录
加载资源限制脚本；修正后完成上述任务。全部失败和原始证据按阶段保存，不归因于设备算子。
[完整结果、180条样本、原始trace、驱动及复现](superkernel_completion_probe_20261011/README.md)。

## 59. 过程 057：后到交接、原计算完成点与交替通知

私有PyPTO新增测试驱动`276c4b42e`，Simpler/PTOAS不变。两任务保留真实数据依赖、
局部Wait和原early-resolve；Set中的额外探测只读取既有DMB，返回后仍由原执行器验证并交接。
没有新增派发、FIN/ACK路径或修改核分配。首探位于Set内，比原执行器探测略早，
所以不能把每条后到记录等同于旧执行器必然漏掉的实例，也不能按两轮交接数相减定量归因。

首轮任务`task_20261011_194106_199331717166`在卡1功能通过，但RuntimeBarrierAll尾等待
包含新增通知，不能证明原M/V重叠。随后以现有AicAllPipes/AivAllPipes加独立计算标记，
AIC额外将通知的MTE3汇合到FIX收尾；任务`task_20261011_194824_22740514841`
在卡0通过，长人工Scalar下未得新增原计算重叠证据。任务`task_20261011_195005_23413354268`
缩短Scalar为0后，在AIC loops=4观察到1条后到交接仍有7tick原Cube计算等待。

候选进一步采用+1/−1交替通知，移除任务入口的标志清零、源重写及两行写回；
搬运与轮询放入noinline辅助函数，已有后继直接返回。任务`task_20261011_195604_258083210411`
在卡0通过，AIC一轮有6条后到交接仍有21～62tick原计算等待。它们是小用例充分证据，
不能据此把Wait处没有剩余计算的所有其他交接定为无收益，或推断真实CSA任务Scalar bound。

任务`task_20261011_200022_273830715020`在卡0完成loops=4、Scalar额外循环0的检查与计时。
每路预热5次、轮换各15次单图重放，含诊断和运行时，单位μs：

| 类型 | once min | once mean | once max | late min | late mean | late max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AIV | 103.34 | 113.55 | 136.30 | 111.10 | 120.03 | 136.76 |
| AIC | 112.50 | 116.18 | 120.86 | 109.92 | 114.61 | 120.20 |

AIC mean减少1.57μs，AIV增加6.49μs；本轮AIC又观察到4条后到原计算重叠，重放后还有1条。
改变输入、完整输出、fanin读取值、保护区及图重放后结果均通过，所有设备任务退出0并释放卡。
前后阶段的负载、诊断协议或卡号不同，不能用跨阶段时间差量化noinline或交替通知的独立收益。

候选仅在小例子中实现。下一步需预留通用L1/UB scratch与通知事件、提供每物理核通知区，
并在原运行时探测位置核实后到窗口；PTOAS现有event7尾部预留不覆盖全部新增MTE3汇合边。
不得把小例子的固定地址直接放到CSA或改变算子作者接口。资源合同具备后才做128K/B16
同SDK、无诊断的完整CSA对照。本轮不改变公共默认，不把AIC小图收益当CSA新成绩。
[五阶段原始数据、最终驱动、增量patch与复现](superkernel_late_arrival_20261011/README.md)。

## 60. 过程058：正式资源预留、后到回调与128K/B16三路对照

PyPTO `fee0196c2`、Simpler `1c258b041`、PTOAS `c63b7d28`、私有vLLM驱动`60d19a78`。
编译器在FIFO之后、tile之前预留64B Mat/Vec scratch；PTOAS保留新增MTE3汇合事件，
显式事件函数拒绝未证明的混用。Simpler新增每物理核192B HBM通知区，A3合计13.5KiB，
DispatchPayload仍576B。算子作者局部Wait/Set、原派发/early-resolve及诊断回退均保持。

runtime直接DMA先后在SO通知、显式flush及HBM通知上卡住；HBM与SO失败位置一致，
撤回“只是SO位置导致”的推断，确切指令根因未定。改为在原首次探测失败后，用参数tag进入
编译器生成的独立尾部回调；该入口在算子主体之前返回，不重复计算，无新增FIN/ACK。
AIC/AIV两种长度、两组输入、输出/fanin/保护区及图重放通过，确实记录到原首次探测后的交接。
MIX两项设备测试通过。通知尾等待包含新增DMA，不能单独当作原计算重叠证据。

任务`task_20261011_210335_103255211417`在单卡2完成128K/B16，无诊断，warmup5后各30次轮换，
同SDK、CANN9.2/TMR/NZ2/atomic0、HC_pre+norm+CSA+HC_post：

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通 | 967.66 | 989.89 | 1017.96 |
| 同MIX标记，原单次探测 | 969.76 | 993.18 | 1015.04 |
| 同MIX标记，后到回调 | 964.56 | 997.50 | 1022.46 |

后到版较单次探测mean多4.32μs、max多7.42μs，10/30配对更快；较普通mean多7.61μs。
9项输出/缓存/状态及改变输入重放逐bit一致，每路31项保护区通过。普通版也用新增通知区冷初始化，
不能从此表推断新SDK未标记路径开销为零。功能保留为私有实验，尚无CSA性能收益，公共默认不变。

失败任务、严格SDK拒绝、CPU构建修正、8个单卡任务和90条计时全部归档；设备均释放。
下一步单独减少每个Set重复写能力及控制行flush，保持同SDK对照，优先128K/B16。
[完整过程、版本、补丁、数据与复现](superkernel_runtime_late_poll_20261011/README.md)。

## 61. 过程059：编译能力随原提交传递，去掉每次Set写GM

Simpler `5085a6ce0`、PyPTO `207741212`、私有vLLM驱动`16e3d7c5`，PTOAS不变。
新协议8通过原任务元数据逐成员声明回调能力，正常Set无GM写入，收尾不再为这些写入flush；
诊断和旧协议7保留写回。没有新增派发、ACK路径、核选择或early-resolve变化。
12项调度CPU检查、8项编排检查通过；int属性读取类型错误在CPU阶段修正。

单卡任务`task_20261011_211954_169757631100`，卡0，AIC/AIV两种长度两组输入、
完整fanin读取、输出保护区与重放通过，首次探测后的后到交接仍有效；MIX两项通过。
首次CSA启动因env前未进Git目录退出128，未执行设备计算；修正cd位置后完成任务
`task_20261011_212234_18104924066`，卡0，所有设备均释放。

128K/B16同SDK四路，CANN9.2/TMR/NZ2/atomic0/诊断关闭，第2层HC_pre+norm+CSA+HC_post，
预热5次后轮换各30次单次图重放，reset在计时外：

| 路径 | min（μs） | mean（μs） | max（μs） |
| --- | ---: | ---: | ---: |
| 普通 | 971.72 | 988.50 | 1004.24 |
| 单次探测 | 966.30 | 981.03 | 998.76 |
| 后到通知，旧Set写入 | 960.60 | 985.07 | 1011.42 |
| 后到通知，新能力元数据 | 962.60 | 984.59 | 1004.02 |

新方案较旧写入mean少0.48μs、max少7.40μs，但配对差值中位数+2.99μs，仅14/30更快，
不能声称稳定加速。较普通mean少3.91μs（0.40%），较单次探测仍多3.56μs；后到通知成本仍需处理。
9项输出/缓存/状态及换输入重放逐bit一致，每路31项保护区通过。保留私有功能与清理，
不改公共默认，不将有后到交接等同于一定有净性能收益。无七档重测或16卡任务。
[完整过程、120条样本、源码增量与复现](superkernel_clean_tail_20261011/README.md)。

## 62. 后续记录要求

每次有意义的修改/实验追加一节，至少包含：目的、修改仓与文件、与上一候选的差异、
命令及版本、任务 ID/设备、实际结果、失败定性、产物路径、结论边界和下一步。
证伪结论在原处明确撤回，保留导致改判的证据；不把退出 0、编译成功或泳道条重叠单独作为功能/性能通过。
涉及性能时保留全部样本的 min/mean/max，记录计时是否含 Host/调度以及是否启用 profiler。
