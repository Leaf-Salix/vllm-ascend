# CSA 完整单 SPMD 优化总记录

首次历史归档：2026-10-17T14:37:51+08:00。本文件位于正式算子目录，集中保留本会话从静态原型到动态区间策略的完整过程。
原文、脚本、原始日志、泳道图和二进制继续保留在 `tests/pypto_test/`；此文不嵌入原始日志或二进制。
**主日志只首次生成一次。后续必须在末尾“持续追加”区手工记录，禁止运行脚本覆盖新记录。**

<a id="current-objective"></a>
## 当前目标与状态

目标：把 CSA 全部 incore task 融合为一个完整 SPMD，支持动态输入形状和按实际工作量覆盖区间的分档，
并让每个验收档位的完整 L1 incore 首尾耗时相对同轮原动态分 task PTO **降低至少 30%**：
`SPMD ≤ dynamic PTO × 0.70`。旧的 `128K/B16 < 700 μs` 单点目标已被替换；下方原文中的旧目标只是当时快照。

- 当前七档均为 S=6：历史 128K 的 B4/8/16/24/28，历史 8K 的 B16/32。2026-10-17 按用户要求移除 8K/B24、增加 128K/B28；旧数据保留。完整源级动态单SPMD的首次七档精度/图/保护检查及成对L1均已完成，但性能全部未达标。后续优化仍需七档重新验收，不能用单点或平均收益替代。
- 分档根据 CSA 自身 B、历史长度、本轮 query 数与总 T，覆盖区间并验证代表值和边界；允许不同形状编译不同二进制。
  不把七个回归点做成支持白名单，不照搬 HC_pre 的阈值，不把目前 S=6 的实现边界缩减成最终目标。
  用户进一步明确允许增加性能策略档位；七个验收点不限制内部策略数量，新增分支须有工作量依据和完整 CSA 收益证据。
- 档内由实际 device metadata 决定有效范围、地址、循环和尾块，不新增 Host 取值。
- metadata计算不强制全部位于单SPMD。允许复用框架推导或独立前置计算；新增成本须计入完整链路，不通过移出计时范围获得虚假收益。
- 后续优化直接进入正式动态源码与生成入口，停止围绕冻结的 B16/S6 或其它单点生成文件继续做变换。允许按 shape 分别 JIT；B16 只是验证点，静态历史实现及结果保留。
- 静态 shape 源码仅存于 `tests/pypto_test` 的历史目录。`ops/pypto` 中的正式算子必须保留 `pl.dynamic` 和按真实输入计算的循环/分核，不允许以多档包装替代动态实现。
- 新动态代码必须有完整中文注释；文件最开头给出算子总览与阅读顺序，解释动态输入、分档、阶段数据流、AIC/AIV 分工、同步及工作区。关键调度节点说明原因和正确性约束，按阶段拆分函数，明确未验证边界。
- 使用正式权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，同输入、同设备配置、同轮多次采样中位数比较完整 CSA。
  计时包含内部等待和阶段空隙，排除外围 AICPU/orchestrator 首尾；Native、局部阶段、CPU 模型和插桩数据不能替代验收。
- 先功能、再精度、最后性能。输出、Top-K、cache/state、动态 metadata、图重放、保护区与核间同步的既有门槛保持不变。

**需要覆盖历史表述的最新核实：**

1. 当前 A3 的 `aic_gather` 底层仍是 `AIV UB → GM FIFO → AIC L1`，不是绕过 GM 的直接 UB→L1。
   历史文档里“直传/消除 GM 往返”的设想必须按后续源码审计纠正，不能当作已获得的性能收益。
2. 历史 B16 原型插桩中 HC 首个数学开始到最后 mix/norm 发布约 60.78 μs；该档当时完整结果距 30% 目标仍差约 173.60 μs。
   只优化 HC 前段不足以单独证明完整目标可达；这只是量级比较，不是同轮消融或性能下限。
3. `hc_sinkhorn_scalar_v1` 只完成独立 AST 候选和静态审计，未进行 PyPTO parse/lowering、编译或设备验证。
   Scalar `/` 与 A3 `vdiv` 的舍入、`vcadd` 内部归约树均未证明逐 bit 一致；8 行串行还有性能风险。
4. **已撤回此前“正式动态生成链”的表述与入口。** 源码核实发现生成器会将动态维度替换为
   当前 shape，并逐阶段展开。按用户最新要求，20 个模块的完整快照已保存到
   [static_shape_archive_20261017](../../../tests/pypto_test/csa_multishape_spmd_20261016/static_shape_archive_20261017/)，
   静态生成链已从正式目录移出。正式目录现已有完整源级动态入口，但动态场景及性能验收尚未完成，
   不导出旧 `build_csa`，也不以旧模板作默认回退。此前 B17 编译、B16 性能均只是历史证据。
5. 隔离 S4 同源数学与形状实值化后的单 SPMD 已通过 CPU 后端编译，但不是目标动态入口的
   完成证明；真实 S4 设备精度、奇数 S 和 mixed-query 仍待完成。HC 紧凑双槽保留为历史候选，
   不更新当前已选性能，也不再沿冻结生成文件继续优化。
6. 正式入口为`deepseek_v4_flash_csa_spmd/entry.py::decode_csa_spmd`，捕获前绑定为
   `binding.py::PreparedDynamicCSA`。HC、QKV、两压缩/state/cache、Indexer、Attention、
   输出投影与HCpost已经接成一次24组MIX；130个根Tensor及动态形状/容量轴保留。
   完整七档S6输出、TopK/score、cache/state与原动态分task零差异，图重放及保护检查通过。
   页表helper另有mixed/dummy/空洞/短表真机probe；完整CSA的固定S6同图历史/位置/页表与紧凑metadata变化已通过；mixed-query、
   整请求padding在128K/B16、8K/B32同址图中已通过，请求生命周期仍未完成，真实服务路由未切换。
   最新HC行依赖＋页检查有效列修正已完成同源码七档：128K B4/8/16/24/28单SPMD
   560.76/719.46/914.22/1199.54/1356.86 μs；8K B16/32为716.32/1031.30 μs。
   同轮均慢于原分task PTO（3.83%～12.42%），所有30%目标均未达。
   输出按组发布采用60个phase；HC行依赖默认开启，两档同进程消融降低2.13%/3.03%。
   HC开启后128K/B16与8K/B32各六次同址metadata变化及B→B-1→1→B的
   padding重放均已通过；固定S6七档及整请求padding不代替mixed-query、空rank或生命周期。
   所有数值启用测试编译时100tick轮询退避，尚未默认接入binding；详情见文末与
   tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/HC_ROWS_PLAN_COLUMNS_SEVEN.md。
   数学与分工均来自正式动态源码，不以历史静态模板或局部通过替代完整目标。

### 后续记录入口

按用户要求，**本文件是 CSA 单 SPMD 优化过程的统一主记录**。后续实现、优化、编译失败、
同步问题、issue/PR、精度和性能结果都追加在文末；历史目录保留代码、脚本和原始证据。
不再以历史目录中新建的零散过程 Markdown 作为后续主账，也不删除已有静态版本和失败记录。

- [完整生成链迁入正式包](#production-generator-migration)
- [HC 紧凑双槽与 S4 源码探索的最新记录](#hc-compact-and-query-shape-progress)
- [原始静态版本、完整历史与逐次尝试](#static-ledger)
- [动态区间策略及既有七档对照](#dynamic-ledger)
- [持续追加区](#ongoing-log)

每次追加至少写清：**问题与依据 → 具体改动 → 验证范围及证据路径 → 性能口径 → 是否采用及原因 → 未完成项**。
没有上卡或没有采样时明确写出；失败、撤回和未采用候选同样保留。

## 阅读说明与目录

下列每篇历史文档都标明原始路径并保留全文，包含失败、初稿、过期目标和当时未解决的问题。
除代码区域外的 Markdown 相对链接重定位外，不改写原内容。历史正文的“当前”“已通过”“下一步”仅指写作当时，
不能覆盖上面的现目标或末尾新记录。源码块、原 issue 正文及重复保存的证据快照均保留。

本次盘点 122 篇 Markdown，共 1,118,103 字节；其中 117 篇本会话文档全文纳入，
原文 863,714 字节；5 篇外部只读参考仅列原文链接和本地快照。
清单与逐链接重定位记录：
[history_inventory.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_migration_20261017/history_inventory.json)。
清单中的字节偏移对应首次归档快照；本文件首页会更新、末尾会追加，当前正文应通过稳定的
`SOURCE_BEGIN`/`SOURCE_END` 标记定位，不能沿用旧字节偏移截取。
归档脚本：
[assemble_history.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_migration_20261017/assemble_history.py)。

- [当前目标与状态](#current-objective)
- [静态原型总账](#static-ledger)
- [静态逐次尝试](#static-attempts)
- [静态专项记录、组件问题与 issue 正文](#static-special)
- [动态区间入口总账与七档结果](#dynamic-ledger)
- [动态专项实现、诊断与失败尝试](#dynamic-special)
- [交接、旧目标与工作区历史](#handoff-history)
- [外部只读参考原文](#external-references)
- [历史归档结束与持续追加](#ongoing-log)

<!-- CSA_SPMD_HISTORY_ARCHIVE_BEGIN -->

<a id="static-ledger"></a>
## 静态原型总账

- [001 · csa_static_spmd_feasibility_20261015/README.md](#history-document-001)
- [002 · csa_static_spmd_feasibility_20261015/PROCESS_LOG.md](#history-document-002)
- [003 · csa_static_spmd_feasibility_20261015/validation_log_excerpt.md](#history-document-003)

<a id="history-document-001"></a>
### 历史原文 001：README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/README.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/README.md)；原文 14,066 字节。静态原型总览、过程总账或当时验证摘录，全文保留。

<!-- SOURCE_BEGIN history-document-001 -->
# CSA 单 SPMD：完整过程与接续入口

完整过程统一从 [PROCESS_LOG.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PROCESS_LOG.md) 阅读；逐次成功/失败尝试见
[ALL_ATTEMPTS.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/ALL_ATTEMPTS.md)，issue/PR 与本地修复见
[ISSUES_AND_PRS.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/ISSUES_AND_PRS.md)。材料同步归档到用户指定主仓的同名
`tests/pypto_test/csa_static_spmd_feasibility_20261015/`，归档清单见 ARCHIVE_STATE.json。

目标是把较优 CSA 动态实现的计算分片、逐核顺序及必要依赖固化到一次 MIX SPMD
提交中。保留原数学 tile、同组 Cube/Vector 握手和 Native device Tensor 输入。
目前没有生产替换。128K/B16 静态 CSA 已完成三次设备调用，对冻结动态 PTO 的
输出、Top-K、六份 cache/state 逐项完全相同，进度及保护区通过。后续同卡性能测试已完成：
L1中位数动态897.70 μs、单SPMD1380.22 μs，耗时增加53.75%；固定输入图重放比较通过。
详见[性能报告](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md)。

当前静态模板特化了 batch、history、shape 和逻辑 block 数，全部按用户要求保留。
最终需要动态 batch_size/seqlen；静态原型通过不代表动态实现完成。
设备验证范围仍仅为 **128K/B16（history=131072，16 请求，每请求 6 query token）**。
用户已明确授权实现临时普通 GM notify/wait，原缓存审查暂停已解除。
独立 PyPTO 接口和完整 B16 CPU 编译通过。首版返回但输出错误，补齐通知前的
本核 PIPE_ALL 后，独立 payload 小例与完整 B16 各三次调用均通过；见
[临时通知/等待实现](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md)。
[Scalar 缓存协议复核](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md) 保留此前审查依据。

用户转来的动态范围、模板边界与负载均衡分析，单独保存于
[动态 SPMD 后续参考](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/DYNAMIC_SPMD_REFERENCE_20261016.md)。

## 2026-10-16 独立组件修复进展

用户已授权分别修改独立 Simpler 与 PyPTO worktree。最新结果统一记录在
[PYPTO_STATIC_FIXES_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md)。以下历史限制
描述保留为修复前基线，不再代表共享 InOut、CORE 容量或 JIT 小例子的当前状态。

- Simpler CORE 扩容已验证并本地提交 `57e499238`；共享安装未切换。
- PyPTO 分支 `feat/csa-static-spmd-fixes-20261016` 已完成完整 B4 默认 PYPTO
  代码生成；最终 carry 修复相关两文件 169 项回归通过，补丁尚未提交。
- 共享 InOut 单 MIX 的 24 组、双 AIV、当前流和三次执行真机验证已通过。
- Torch schema 的 64 入参限制已经绕开：用户明确选择现有 JIT native 直调，
  62 项候选已取消。详见 [直调入口记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/DIRECT_NATIVE_ENTRY_20261016.md)。
- B16 完整 kernel-mode 代码生成已通过。设备验证首先遇到两个 Native cache
  子视图与根缓冲的部分可写重叠；改在入口内部派生这两个描述符，外部 130 项、
  Core 仍 132 项，不增加搬运或设备 task。逐次结果见 PROCESS_LOG.md。
- B16 完整 native 任务已完成二进制编译并实际启动，但设备报告
  `S1:running-stalled`，没有完整输出。抽取同一控制表的诊断也出现进度等待超时；
  五组历史控制表诊断均已结束：读写 volatile 或仅读 volatile 的三代际观察通过，
  原版、仅消费者 fence、仅写 volatile 均失败。这不证明完整同步协议正确。
  范围、任务和证据见
  [B16 进度协议诊断](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/B16_PROGRESS_DIAGNOSIS_20261016.md)。
- 后续临时普通 GM notify/wait 与本核流水等待已解决当前 B16 原型的运行阻塞：
  task_20261016_180421_21978513378 的三次调用功能及对冻结动态 PTO 的精确比较通过。
  每轮恢复 fixture，进度不清零；不是连续 decode 轨迹，也没有测性能。
- 完整 B4 的现有 program 产物已通过 CCE/调度二进制编译及 callable 组装，
  详见 [二进制编译记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md)。尚未执行完整设备核。

## 已完成的准备

- 使用冻结的 `results/hc_pre_layer_integration_20261012/baseline_ops` 计算实现，
  官方 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 权重，第 4 层，S6，补齐七档同轮
  `task/block_idx/subslot/runtime worker` 映射。各档原动态图重放与保护区检查通过。
- 每档 66 个计算调用、4 个依赖占位任务、72 个 worker；全部真实 Tensor 实参与
  shape/stride/offset 对齐。长 history 检查 292 个视图，短 history 检查 293 个。
- 完整导出 73 个物理核计算体，独立 PyPTO 解析通过；本七档选中的每档 68 个
  计算体实例已绑定到静态模板。未另写 QKV/Indexer/Attention 数学。
- 静态表精确保留逐核分片、顺序和等待目标；去除传递冗余后验证依赖偏序相等。
  合并源码批次仅是减少重复代码，不把公共源码顺序变成全卡屏障。
- 准备工作区和进度协议，检查所有视图边界、allocation 不重叠；3 个随机推进
  顺序各执行 3 轮代际模型通过。此结果不证明 NPU 可见性或内部事件无死锁。
- 七档完整 `entry.py` 均通过 PyPTO 解析，同时生成同源 `entry_jit.py` 供 kernel
  模式接入准备。B4 JIT 使用 inline Group，共享 InOut 的 ABI 别名问题已在独立
  PyPTO 修复，完整 B4 v21 默认 PYPTO 代码生成通过；下文保留修复前失败过程。
- 测试适配器 `prepared_arguments.py` 已准备；七档 meta Tensor 检查通过，
  Native 存储别名、四份 NZ 权重对象保持。它不执行静态 CSA。

| history | B | T | 原核执行成员数 | 工作区字节 | 原型核 Tensor 参数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 131072 | 4 | 24 | 712 | 280782848 | 132 |
| 131072 | 8 | 48 | 786 | 285502464 | 132 |
| 131072 | 16 | 96 | 973 | 295028736 | 132 |
| 131072 | 24 | 144 | 1057 | 321288704 | 132 |
| 8192 | 16 | 96 | 1005 | 294218752 | 133 |
| 8192 | 24 | 144 | 1057 | 321006592 | 133 |
| 8192 | 32 | 192 | 1188 | 329963520 | 133 |

工作区暂不复用原临时 allocation，避免为节省 GM 增加依赖。没有权重内容或缓存
快照入本目录。以上工作区包含同步标记和只读执行表，不是新增 metadata 数据转换。

## 修复前限制与定位过程（历史记录）

本节保留原始失败过程。共享 InOut、参数容量及默认路径融合编译的当前结果以
上方“独立组件修复进展”为准；完整 B16 的设备执行和对动态基线精确比较现已通过，
后续仍待动态模板、图重放以及性能等当前验证范围之外的工作。

1. 自动 MIX 展开中 while 条件的 liveness 问题已提交
   [PyPTO #3009](https://github.com/hw-native-sys/pypto/issues/3009)，独立复现见
   [PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md)。
   本次另准备了显式 Group 内 AIC/AIV 两半的写法；仍是一次 MIX 提交，不能把它
   与自动拆分路径的验证结果混为一谈，也没有增加循环后伪使用绕过问题。
2. 原 Simpler 的 32 Tensor 限制已在独立 worktree 扩至 256 Tensor + 16 scalar；
   dump 多字掩码、派发结构及所有关联布局同步更新，A3 的 HBG/TMR program/kernel
   大参数验证通过。详见 [扩容记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md)。共享环境
   没有切换，后续整层运行必须显式使用独立扩容版本。当前原型仍有 132/133 个
   Tensor 描述符；原 Native 为 56，早期 arena 入口不是当前实现。
3. B4 完整融合已推进到片上内存分配：默认 PYPTO 规划器报告 Left 需要 81920 字节、
   上限 65536；DSA_RP 也未找到容量内布局。PTOAS 路径修正本地视图表达后，
   最新 v23 生成的 qk_pv AIV MLIR 仍有两处 SSA 定义不支配使用点，未完成编译。
   原独立核最大 Left 用量为 65536 B，应核对融合生命周期/复用，不能认定硬件
   不可实现。本次没有为通过检查缩小 tile 或降低 pipeline stage。
   现已抽出 13 参数的两阶段复现：Indexer/O-proj 单独编译均通过，合并后
   Left/Right 都达到 80 KiB。固定原 Tile 后仅改用 DSA_RP 地址分配，两段合并
   可以放进 Left/Right 各 64 KiB；但直接切换 planner 会改变分块并触发另一个
   容量问题。见 [内存分配复现](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/MEMORY_REUSE_PHASES_REPRO.md)。
4. inline Group 已让 5 参数小例子生成单 MIX 并通过 Meta 注册；共享 InOut 的
   小例子在 ExpandMixedKernel 首次产生 free variable。完整证据及接续方法见
   [COMPILE_AND_RESUME.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md)。
   详见 [JIT Group 表达记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/JIT_SINGLE_GROUP_NOTES.md) 和
   [共享 InOut 复现](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_GROUP_SHARED_INOUT_REPRO.md)。

完整编译证据位于 `../results/csa_static_spmd_feasibility_20261015/`。中间 v1～v14
包含生成器调试失败，不能当作各组件的独立 bug；当前源代码与最终结论以
`preparation_progress.json` 和后续编译结果为准。

## 文件与命令

- `bindings/*_prepared.json`：同轮映射、原标量/视图实参、显式依赖、所选编排分支。
- `generated/core_bodies_ir.py`、`core_body_catalog.json`：完整物理核体及原事件清单。
- `static/h*/layout.json`：逐核表、只读等待表、工作区地址及原内部事件的实参。
- `static/h*/entry.py`、`entry_jit.py`：完整 program/kernel 模式静态入口原型。
- `static/h*/protocol_model.json`、`parse_result.json`：CPU 验证范围与结果。
- `static/abi_capacity.log`：使用实际 Simpler 头文件的 CPU 容量复现。
- `static/argument_binding_result.json`：七档测试适配器的 meta Tensor 绑定检查。
- `static/b4_ir_audit.json`：一次 24 组 MIX 提交及 74 个 while 状态的部分 IR 检查。
- `static/left_memory_audit.json`：原独立核与融合核的 Left 分配记录。
- `static/memory_probe_summary.json`：两个独立可编译阶段及融合后的资源对照。

在当前 worktree 根目录进行 CPU 准备：

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
source /data/pyptouser/qinchuanyu/pto-eager/pypto/.claude/skills/testing/load-env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/prepare_all_static.py

python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/compile_static_entry.py \
  --source tests/pypto_test/csa_static_spmd_feasibility_20261015/static/h131072_b4/entry.py \
  --output /tmp/csa_static_b4_new_compile
```

编译检查可显式指定 `--memory-planner PYPTO/DSA_RP/PTOAS`。不能把更换规划器当作
已经满足原性能目标；需要检查它是否保留原 pipeline 资源分离及最终片上容量。

## 协议与接续约束

以下 INT64 描述针对保留的旧 static 模板；已通过的临时通知版位于
static_local_signal/h131072_b16，采用 INT32 信号、同样的 512 B 间距，
并在通知前加入本核 PIPE_ALL。以临时接口文档的最终合同为准。

进度值为 `generation * 1024 + 原核内完成序号`；每个 worker 独占 512 字节。
初始化只清零进度区、写入只读表，重放时不重置整个进度数组。所有 worker 每次
调用均进入新代际，前一轮标记不能满足本轮等待。同一工作区只允许串行使用，
并发调用或不同流必须独立拥有工作区。需分别核对 INT64 访问的对齐与非撕裂保证、
写入发布、读侧路径以及本核流水完成与 payload 对端可见性，不能只凭几次真机
通过证明协议成立。单写者进度不要求为多个读者额外引入 atomic RMW。

静态 worker 是源 trace 核编号对应的逻辑标签，不把 `get_block_idx()` 当成裸硬件
physical core id。完整 24 组 MIX 启动允许整体置换核组；相同静态组中的 AIC/AIV0/
AIV1 配对、每组工作量及所有依赖必须保持同构。不能任意只置换其中一条 lane。

当前模板只代表已捕获的七种形状和编排分支。positions、seq_lens、slot、页表与
state 仍然在设备上读取，但不能将一个模板用于跨越原 Indexer 拓扑分支的任意
history；未来服务入口应使用已有 Host metadata 选模板，不新增热路径 D2H。

while 独立回归、完整 B4/B16 默认路径代码生成及单 MIX 结构检查已经完成，
不要因接手重复。用户限定的设备范围是静态 128K/B16；当前先完成协议讨论，
恢复设备验证后再核对
完整 native 入口的输出、cache/state、保护区和跨调用代际。本轮不重跑 B4、
七档精度或性能；后续扩展范围再按用户指示处理。
原动态七档补采已经完成，不要因接手而重跑。DEBUG 补采会扰动调度，不能把其耗时
当作历史最佳排程或新的性能结果。

此前准备阶段未修改依赖仓库；获用户授权后的组件修改仅在独立 worktree，
未修改生产 CSA/HCA/HC_pre 的已有工作树改动。


## 组件接手包（修复前历史快照）

2026-10-16 核对：本地 CORE Tensor 容量仍为32，#3009仍为OPEN，完整融合编译与
JIT入口尚未通过。七档结构准备、参数绑定及独立复现均保留；目标尚未完成。
恢复需要组件侧的参数承载与编译/入口支持，不能仅把已有CPU解析结果当作可部署证据。

[下载组件复现包，约40 KB](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/component_repro_kit_20261016.zip)。
包内22份文本材料，不含模型权重、KV快照、设备二进制或校验摘要。独立编译驱动
只依赖已安装的PyPTO；三个包内用例解析通过，无vLLM/本工作树导入依赖。
详见 `component_kit_validation.json`。不要因接手重复已完成的采集、绑定和编译对照。

<!-- SOURCE_END history-document-001 -->

---

<a id="history-document-002"></a>
### 历史原文 002：PROCESS_LOG.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/PROCESS_LOG.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PROCESS_LOG.md)；原文 33,204 字节。静态原型总览、过程总账或当时验证摘录，全文保留。

<!-- SOURCE_BEGIN history-document-002 -->
# CSA 动态调度固化为单个 SPMD：完整过程总账

本账覆盖 2026-10-15 起本会话的 CSA 单 SPMD 工作，包括失败、作废、撤回及
仍未完成的尝试。后续继续追加，不能只保留成功结果。

用户指定归档位置为：
`vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_static_spmd_feasibility_20261015/`。
开发仍在独立 worktree；这里的脚本、记录、结果摘要和补丁快照同步到指定目录，
不覆盖其他会话的生产代码。详细编译 IR、二进制和大快照保留原位置，
路径、大小及保留方式见 `artifact_inventory.json`；不生成内容 hash。

## 读哪些文件

- [逐次尝试清单](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/ALL_ATTEMPTS.md)：按实际目录枚举，包含成功、失败和没有最终
  result 的尝试；完整错误见对应的原始 JSON/log，不用目录名猜测 PASS。
- [问题与修复登记](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/ISSUES_AND_PRS.md)：issue、PR、本地提交、未提交修复分别记录。
- [最新 PyPTO 修复](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md)：代码位置、原因、改法、验证边界。
- [Simpler 扩容](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md)：256 Tensor、16 scalar、dump、
  派发、图缓存等 ABI 和设备验证。
- [PR #3013 核对](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PR3013_REVIEW_20261016.md)：精确 head、独立构建、原始 while 复现和验证边界。
- [JIT 直调入口](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/DIRECT_NATIVE_ENTRY_20261016.md)：用户取消 62 项方案后的调用路线与 TorchAir 参考。
- [编译接续说明](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md)：修复前各路径、接口约束和恢复顺序。
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
  [TORCH_SCHEMA_CAPACITY_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/TORCH_SCHEMA_CAPACITY_20261016.md)。已按
  原目标“遇到关键点停下讨论”询问 56 个 Native Tensor + 6 个分 dtype workspace
  的具名入口方案，等待答复期间没有实现该方案或修改共享 PyTorch。
- 与入口选择无关的现有产物二进制编译已独立推进：`static_b4_binary_v1`
  在 115.418 秒内完成 AIC/AIV CCE 编译链接、AICPU 调度编译和 132 参数
  callable 组装。原产物未改，二进制本地保留；见
  [BINARY_COMPILE_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md)。

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

<!-- SOURCE_END history-document-002 -->

---

<a id="history-document-003"></a>
### 历史原文 003：validation_log_excerpt.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/validation_log_excerpt.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/validation_log_excerpt.md)；原文 30,307 字节。静态原型总览、过程总账或当时验证摘录，全文保留。

<!-- SOURCE_BEGIN history-document-003 -->
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

[完整准备清单](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/README.md)；
[编译证据及接续步骤](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md)；
[CORE参数容量复现](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_ARGUMENT_CAPACITY.md)。


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

[两阶段分配复现及对照](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/MEMORY_REUSE_PHASES_REPRO.md)；
[JIT Group表达记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/JIT_SINGLE_GROUP_NOTES.md)；
[完整编译接续](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md)。


## 605. 2026-10-16：独立组件复现包与受阻接续审计

完成22份纯文本材料的组件复现包，约40KB，包含两阶段内存规划、JIT Group、CORE容量
以及#3009 while状态复现。包内独立驱动三个用例仅做解析，全部通过；已去除vLLM和
本地SDK启动脚本依赖。没有重跑已有编译对照、七档采集或NPU测试，没有权重/快照/bin入包。

再次只读核对：#3009仍OPEN，本地while活跃性种子仍未包含condition，Simpler CORE上限仍32。
此前七档静态签名实际为132/133，完整融合编译与JIT入口限制仍在。相同组件前提已连续
三轮核实；准备材料保留，目标未完成，进入受阻接续审计。恢复须先具备对应组件支持，
再执行完整B4 CPU编译、设备协议/功能、七档精度，最后性能。

[组件复现包](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/component_repro_kit_20261016.zip)；
[接续总览](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/README.md)。


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

<!-- SOURCE_END history-document-003 -->

---

<a id="static-attempts"></a>
## 静态逐次尝试

- [004 · csa_static_spmd_feasibility_20261015/ALL_ATTEMPTS.md](#history-document-004)

<a id="history-document-004"></a>
### 历史原文 004：ALL_ATTEMPTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/ALL_ATTEMPTS.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/ALL_ATTEMPTS.md)；原文 29,910 字节。静态逐次尝试总表，包含失败及未完成尝试，全文保留。

<!-- SOURCE_BEGIN history-document-004 -->
# CSA 单 SPMD：逐次尝试清单

按现存结果目录的时间顺序归档；目录时间不是精确任务开始时间。
没有最终 JSON 的目录明确标记，不能据目录存在推断任务执行或通过。
各条完整错误、源码快照和日志见 evidence；大 IR/二进制路径见 artifact_inventory.json。
CPU 单测、构建、无效试验另见 session_logs 和 PROCESS_LOG.md。

| 尝试 | 实际记录 |
| --- | --- |
| cpu_v1 | [compile_results.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/cpu_v1/compile_results.json): 结果明细见 JSON，不推断整体 PASS |
| cpu_v2 | [compile_results.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/cpu_v2/compile_results.json): 结果明细见 JSON，不推断整体 PASS |
| cpu_v3 | [compile_results.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/cpu_v3/compile_results.json): 结果明细见 JSON，不推断整体 PASS |
| cpu_v4 | [compile_results.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/cpu_v4/compile_results.json): 结果明细见 JSON，不推断整体 PASS |
| cpu_v5 | [compile_results.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/cpu_v5/compile_results.json): 结果明细见 JSON，不推断整体 PASS |
| standalone_repro | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/standalone_repro/scalar/result.json): device_execution=False; 1 function(s) failed to compile:<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/standalone_repro/tile/result.json): device_execution=False; 1 function(s) failed to compile:<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/standalone_repro/tile_keep_result/result.json): device_execution=False |
| mapping_b4_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_b4_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| vector_bodies_cpu_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/vector_bodies_cpu_v1/result.json): device_execution=False; FlattenTileNdTo2D: tile op 'tile.load' has a dynamic (non-constant) dimension 0 in its >2D tile shape, which cannot be flattened to 2D. Hardware tiles map to fixed-size on-chip buffers, so every tile dimension must be a  |
| vector_bodies_cpu_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/vector_bodies_cpu_v2/result.json): device_execution=False; @pl.jit: missing inferred tensor metadata for parameter 'kv_cache_flat' of 'cache_writeback_body'. This usually means the tensor's shape/dtype could not be statically determined, or that the call site passing it was not  |
| vector_bodies_cpu_v3 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/vector_bodies_cpu_v3/result.json): device_execution=False |
| full_baseline_cpu | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/full_baseline_cpu/result.json): device_execution=False |
| mapping_b4_v2 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_b4_v2/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| mapping_h131072_b8_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_h131072_b8_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| mapping_h131072_b16_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_h131072_b16_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| mapping_h131072_b24_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_h131072_b24_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| mapping_h8192_b16_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_h8192_b16_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| mapping_h8192_b24_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_h8192_b24_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| mapping_h8192_b32_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/mapping_h8192_b32_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS |
| static_b4_cpu_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v1/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Verification failed after 'pipeline_input' for properties {TypeChecked, NoRedundantBlocks, BreakContinueValid, InOutUseValid, ManualDepsOnSubmitOnly, AtomicAddDtypeValid}: |
| static_b4_cpu_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v2/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; IR auto-name base cannot contain reserved delimiter '__': stage_0_w_chunk_inline445__tile_t |
| static_b4_cpu_v3 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v3/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Verification failed after 'ConvertToSSA' for properties {SSAForm}: |
| static_b4_cpu_v4 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v4/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; tile.extract requires a 2D source tile, but got rank 5 |
| static_b4_cpu_v5 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v5/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; tile.slice requires the window and the source to have the same rank, but got window rank 2 [256, 128] and source rank 3 [1, 256, 128] [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tes |
| static_b4_cpu_v6 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v6/result.json): full_codegen_passed=False; parse_passed=False; device_execution=False; Unknown shape variable: QKV_QPROJ_MM_T_DYN |
| static_b4_cpu_v7 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v7/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; NZ layout supports slicing the leading axes only: shape[0] (rows) must take the whole axis at offset 0. A window inside the trailing matrix selects one run per fractal column block, which no contiguous stride describes.  |
| static_b4_cpu_v8 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v8/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; tile.reinterpret_view only supports flat tiles with slayout=none_box, except UINT8/FP8E8M0 MX-scale byte aliases [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/csa_sta |
| static_b4_cpu_v9 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v9/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; NZ layout requires the slice offset on shape[-2] to be non-negative, and this one cannot be proven to be. A negative offset is clamped to 0 at the partition view rather than rejected, so it would read the wrong fractal s |
| static_b4_cpu_v10 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v10/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; NZ layout requires the slice offset on shape[-2] to be non-negative, and this one cannot be proven to be. A negative offset is clamped to 0 at the partition view rather than rejected, so it would read the wrong fractal s |
| abi_capacity | 没有最终 result；查原目录与日志，未记 PASS |
| static_b4_cpu_v11 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v11/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Loop-carried values 'stage_0_v13__iter_v1' and 'stage_0_v17__iter_v1' share the same on-chip buffer, so one iteration cannot preserve both. Give each loop-carried tile its own initial value -- build them with separate op |
| static_b4_cpu_v12 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v12/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Internal error: MemoryReuse cannot reconcile divergent L0C accumulator buffers with tile.move; accumulator control-flow values must be coalesced before YieldFixup. [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre- |
| static_b4_cpu_v13 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v13/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Verification failed after 'InferTileMemorySpace' for properties {TileMemoryInferred, AivSplitValid, AccToGmStoreValid, FixpipeEpilogueValid, AccCompactValid}: |
| static_b4_cpu_v14 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v14/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Internal error: MemoryReuse cannot reconcile divergent L0C accumulator buffers with tile.move; accumulator control-flow values must be coalesced before YieldFixup. [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre- |
| static_b4_cpu_v15 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v15/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Left buffer usage (81920 bytes) exceeds platform limit (65536 bytes) |
| static_b4_cpu_v16_dsa_rp | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v16_dsa_rp/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; DSA-RP could not find a placement for 'csa_static_aic' within the on-chip memory capacities: canonical greedy found no capacity-fitting placement [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration- |
| static_b4_cpu_v17_ptoas | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v17_ptoas/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; 2 function(s) failed to compile: |
| static_b4_cpu_v18_external | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v18_external/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; 1 function(s) failed to compile: |
| static_b4_jit_cpu_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v1/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; @pl.jit function 'entry': parameter 'arg_wq_a' annotates a TensorView in its layout slot, which @pl.jit does not yet support — it would be dropped and the parameter compiled as ND. Use a plain layout (e.g. pl.MX_A_ZZ), o |
| static_b4_jit_cpu_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v2/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Unknown shape variable: QKV_QPROJ_MM_T_DYN |
| static_b4_cpu_v19_valid_view | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v19_valid_view/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; 1 function(s) failed to compile: |
| static_b4_jit_cpu_v3 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v3/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; `with pl.spmd(...)` dispatch body launches 2 kernels; only one kernel dispatch per SPMD scope is supported. |
| memory_probe_DSA_RP | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_probe_DSA_RP/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; DSA-RP could not find a placement for 'csa_static_aic' within the on-chip memory capacities: canonical greedy found no capacity-fitting placement [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration- |
| memory_probe_PYPTO | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_probe_PYPTO/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Right buffer usage (81920 bytes) exceeds platform limit (65536 bytes) |
| jit_group_wrapper_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/jit_group_wrapper_v1/result.json): full_codegen_passed=False; device_execution=False; 1 function(s) failed to compile: |
| memory_oproj_PYPTO | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_oproj_PYPTO/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False |
| memory_indexer_PYPTO | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_indexer_PYPTO/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False |
| memory_pair_compact_PYPTO | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_pair_compact_PYPTO/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Right buffer usage (81920 bytes) exceeds platform limit (65536 bytes) |
| memory_pair_compact_DSA_RP | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_pair_compact_DSA_RP/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; compile() was called with memory_planner while a PassContext is already active. Set the memory planner on the existing PassContext instead. |
| static_b4_cpu_v20_buffer_ir | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v20_buffer_ir/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; compile() was called with memory_planner while a PassContext is already active. Set the memory planner on the existing PassContext instead. |
| memory_pair_compact_DSA_RP_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_pair_compact_DSA_RP_v2/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; DSA-RP could not find a placement for 'csa_static_aic' within the on-chip memory capacities: canonical greedy found no capacity-fitting placement [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration- |
| static_b4_cpu_v21_buffer_ir | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v21_buffer_ir/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; IR Verification Report |
| static_b4_cpu_v22_buffer_ir | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v22_buffer_ir/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; 1 function(s) failed to compile: |
| static_b4_cpu_v23_flat_phi | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_cpu_v23_flat_phi/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; 1 function(s) failed to compile: |
| component_repro_kit_20261016 | 没有最终 result；查原目录与日志，未记 PASS |
| component_kit_parse | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_kit_parse/indexer/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_kit_parse/oproj/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_kit_parse/pair/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False |
| phase_partition_PYPTO_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/phase_partition_PYPTO_v1/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Left buffer usage (81920 bytes) exceeds platform limit (65536 bytes) |
| phase_partition_PTOAS_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/phase_partition_PTOAS_v1/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; 1 function(s) failed to compile: |
| phase_partition_DSA_RP_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/phase_partition_DSA_RP_v1/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False |
| memory_indexer_DSA_RP_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_indexer_DSA_RP_v1/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False |
| memory_oproj_DSA_RP_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_oproj_DSA_RP_v1/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; DSA-RP could not find a placement for 'csa_static_aic' within the on-chip memory capacities: canonical greedy found no capacity-fitting placement [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration- |
| simpler_core256_v1 | 没有最终 result；查原目录与日志，未记 PASS |
| jit_group_inline_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/jit_group_inline_v1/result.json): full_codegen_passed=True; device_execution=False |
| jit_group_inline_meta_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/jit_group_inline_meta_v1/result.json): passed=True; device_execution=False |
| static_b4_jit_cpu_v4_inline | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v4_inline/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; `with pl.spmd(...)` dispatch body launches 2 kernels; only one kernel dispatch per SPMD scope is supported. |
| static_b4_jit_cpu_v5_inline | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v5_inline/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Verification failed after 'pipeline_input' for properties {TypeChecked, NoRedundantBlocks, BreakContinueValid, InOutUseValid, ManualDepsOnSubmitOnly, AtomicAddDtypeValid}: |
| static_b4_jit_cpu_v6_alias | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v6_alias/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; 1 function(s) failed to compile: |
| oproj_fixed_tiles_dsa_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/oproj_fixed_tiles_dsa_v1/result.json): full_codegen_passed=False; allocation_passed=True; device_execution=False |
| pair_fixed_tiles_dsa_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/pair_fixed_tiles_dsa_v1/result.json): full_codegen_passed=False; allocation_passed=True; device_execution=False |
| jit_group_shared_inout_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/jit_group_shared_inout_v1/result.json): full_codegen_passed=False; device_execution=False; 1 function(s) failed to compile: |
| full_b4_fixed_tiles_dsa_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/full_b4_fixed_tiles_dsa_v1/result.json): full_codegen_passed=False; allocation_passed=False; device_execution=False; Undefined variable 'progress__ssa_v1__FREE_VAR' |
| jit_group_inline_device_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/jit_group_inline_device_v1/result.json): passed=False; device_execution=False; Kernel compilation requires the exact clean Simpler SDK revision in its descriptor |
| phase_partition_PYPTO_fix_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/phase_partition_PYPTO_fix_v2/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False |
| memory_pair_PYPTO_fix_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/memory_pair_PYPTO_fix_v2/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False |
| static_b4_jit_cpu_v7_pypto_fix | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v7_pypto_fix/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Vec buffer usage (267008 bytes) exceeds platform limit (188416 bytes) |
| static_b4_jit_cpu_v8_phi_fix | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v8_phi_fix/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Vec buffer usage (201472 bytes) exceeds platform limit (188416 bytes) |
| full_b4_fixed_tiles_dsa_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/full_b4_fixed_tiles_dsa_v2/result.json): full_codegen_passed=False; allocation_passed=False; device_execution=False; Undefined variable 'stage_18_v25_1__FREE_VAR' |
| static_b4_jit_cpu_v9_ptoas | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v9_ptoas/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; 1 function(s) failed to compile: |
| jit_group_shared_inout_device_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/jit_group_shared_inout_device_v2/result.json): passed=False; device_execution=False; The PyPTO torch_npu adapter uses a different Simpler revision; rebuild it |
| static_b4_jit_cpu_v10_ptoas_view | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v10_ptoas_view/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; 1 function(s) failed to compile: |
| isolated_sdk_binding_v3 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/isolated_sdk_binding_v3/result.json): passed=True; device_execution=False |
| jit_group_shared_inout_device_v3 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/jit_group_shared_inout_device_v3/result.json): passed=True; device_execution=True |
| static_b4_jit_cpu_v11_common_phi | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v11_common_phi/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; 1 function(s) failed to compile: |
| static_b4_jit_cpu_v12_shared_handle | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v12_shared_handle/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; 1 function(s) failed to compile: |
| static_b4_jit_cpu_v13_branch_lifetime | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v13_branch_lifetime/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Vec buffer usage (201472 bytes) exceeds platform limit (188416 bytes) |
| static_b4_jit_cpu_v14_lifetime_diag | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v14_lifetime_diag/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Vec buffer usage (201472 bytes) exceeds platform limit (188416 bytes) |
| static_b4_jit_cpu_v15_gate_diag | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v15_gate_diag/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Vec buffer usage (201472 bytes) exceeds platform limit (188416 bytes) |
| static_b4_jit_cpu_v17_column_view | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v17_column_view/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; 1 function(s) failed to compile: |
| static_b4_jit_cpu_v16_dual_order | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v16_dual_order/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Vec buffer usage (197632 bytes) exceeds platform limit (188416 bytes) |
| static_b4_jit_cpu_v18_capacity_gated | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v18_capacity_gated/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; Vec buffer usage (197632 bytes) exceeds platform limit (188416 bytes) |
| static_b4_jit_cpu_v19_dsa_post_fixes | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v19_dsa_post_fixes/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; DSA-RP could not find a placement for 'csa_static_aic' within the on-chip memory capacities: canonical greedy found no capacity-fitting placement [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration- |
| qkpv_memory_probe_PYPTO_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/qkpv_memory_probe_PYPTO_v1/result.json): full_codegen_passed=False; parse_passed=True; device_execution=False; Vec buffer usage (197632 bytes) exceeds platform limit (188416 bytes) |
| qkpv_fixed_tiles_dsa_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/qkpv_fixed_tiles_dsa_v1/result.json): full_codegen_passed=False; allocation_passed=False; device_execution=False; DSA-RP could not find a placement for 'csa_static_aiv' within the on-chip memory capacities: canonical greedy found no capacity-fitting placement [/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration- |
| phase_partition_device_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/phase_partition_device_v1/result.json): passed=True; device_execution=True |
| qkpv_memory_probe_PYPTO_v2_identity | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/qkpv_memory_probe_PYPTO_v2_identity/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False |
| static_b4_jit_cpu_v20_identity_retarget | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v20_identity_retarget/result.json): full_codegen_passed=True; kernel_runtime_validated=False; device_execution=False |
| static_b4_jit_meta_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_meta_v1/result.json): passed=False; device_execution=False; The function schema has 132 arguments but this PyTorch build only supports 64 |
| torch_schema_capacity_cpu_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/torch_schema_capacity_cpu_v1/result.json): device_execution=False |
| static_b4_jit_cpu_v21_carry_scoped | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_cpu_v21_carry_scoped/result.json): full_codegen_passed=True; kernel_runtime_validated=False; device_execution=False |
| static_b4_binary_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_binary_v1/result.json): binary_compile_passed=True; callable_assembled=True; device_execution=False |
| pr3013_original_scalar_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/pr3013_original_scalar_v2/result.json): device_execution=False |
| pr3013_original_tile_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/pr3013_original_tile_v2/result.json): device_execution=False |
| pr3013_original_tile_keep_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/pr3013_original_tile_keep_v2/result.json): device_execution=False |
| static_b4_native_direct_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_native_direct_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_native_direct_v1/result.json): passed=False; device_execution=False; 'host_max_seq_len' |
| static_b16_native_direct_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_native_direct_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_native_direct_v1/result.json): passed=False; device_execution=False; `with pl.spmd(...)` dispatch body launches 2 kernels; only one kernel dispatch per SPMD scope is supported. |
| prepared_b16_inline_v2 | 没有最终 result；查原目录与日志，未记 PASS |
| static_b16_kernel_cpu_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_kernel_cpu_v1/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; `with pl.spmd(...)` dispatch body launches 2 kernels; only one kernel dispatch per SPMD scope is supported. |
| static_b16_kernel_cpu_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_kernel_cpu_v2/result.json): full_codegen_passed=True; kernel_runtime_validated=False; device_execution=False |
| static_b16_native_direct_v2 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_native_direct_v2/fusion_result.json): 结果明细见 JSON，不推断整体 PASS<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_native_direct_v2/result.json): passed=False; device_execution=False; Parameter 'view_98' partially overlaps another tensor with a writable alias |
| prepared_b16_native_views_v3 | 没有最终 result；查原目录与日志，未记 PASS |
| static_b16_kernel_cpu_v3_native_views | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_kernel_cpu_v3_native_views/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; @pl.jit: missing inferred tensor metadata for parameter 'view_98' of 'csa_static_aic'. This usually means the tensor's shape/dtype could not be statically determined, or that the call site passing it was not resolved (no |
| prepared_b16_public_views_v4 | 没有最终 result；查原目录与日志，未记 PASS |
| static_b16_kernel_cpu_v4_public_views | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_kernel_cpu_v4_public_views/result.json): full_codegen_passed=True; kernel_runtime_validated=False; device_execution=False |
| static_b16_native_direct_v3 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_native_direct_v3/fusion_result.json): 结果明细见 JSON，不推断整体 PASS<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_native_direct_v3/result.json): passed=False; device_execution=False; synchronize:../torch_npu/csrc/core/npu/NPUStream.cpp:361 NPU function error: c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream()), error code is 507018 |
| static_b16_protocol_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v1/cpu/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; pl.tensor operation 'write': tensor.write requires value dtype to match tensor dtype, but got value dtype index and tensor dtype int64 |
| static_b16_protocol_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v2/cpu/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v2/result.json): passed=False; 原控制表的有限轮询诊断未通过，见逐核等待记录 |
| static_b16_protocol_v3_volatile | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v3_volatile/cpu/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v3_volatile/result.json): passed=True |
| static_b16_protocol_v4_fence | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v4_fence/cpu/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v4_fence/result.json): passed=False; 原控制表的有限轮询诊断未通过，见逐核等待记录 |
| static_b16_protocol_v6_volatile_write | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v6_volatile_write/cpu/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v6_volatile_write/result.json): passed=False; 原控制表的有限轮询诊断未通过，见逐核等待记录 |
| static_b16_protocol_v5_volatile_read | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v5_volatile_read/cpu/result.json): full_codegen_passed=True; parse_passed=True; device_execution=False<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_protocol_v5_volatile_read/result.json): passed=True |
| volatile_scalar_call_probe_v1 | 没有最终 result；查原目录与日志，未记 PASS |
| static_b16_local_signal_cpu_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_local_signal_cpu_v1/result.json): full_codegen_passed=True; kernel_runtime_validated=False; device_execution=False |
| local_signal_payload_cpu_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/local_signal_payload_cpu_v1/result.json): full_codegen_passed=False; kernel_runtime_validated=False; device_execution=False; 1 function(s) failed to compile: |
| local_signal_payload_cpu_v2 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/local_signal_payload_cpu_v2/result.json): full_codegen_passed=True; kernel_runtime_validated=False; device_execution=False |
| local_signal_payload_device_v2_diagnostic | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/local_signal_payload_device_v2_diagnostic/result.json): passed=False; Tensor-likes are not equal! |
| local_signal_payload_device_v3_drain | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/local_signal_payload_device_v3_drain/result.json): passed=True; device_execution=True |
| local_signal_payload_device_v1 | [result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/local_signal_payload_device_v1/result.json): passed=False; Tensor-likes are not equal! |
| static_b16_local_signal_device_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_local_signal_device_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_local_signal_device_v1/result.json): passed=False; device_execution=True; 完整静态 CSA 与动态基线、进度或保护区不一致 |
| static_b16_local_signal_device_v2_drain | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_local_signal_device_v2_drain/fusion_result.json): 结果明细见 JSON，不推断整体 PASS<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b16_local_signal_device_v2_drain/result.json): passed=True; device_execution=True |
| static_vs_dynamic_perf_b16_v1 | [fusion_result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/fusion_result.json): 结果明细见 JSON，不推断整体 PASS<br>[result.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/result.json): passed=True |

<!-- SOURCE_END history-document-004 -->

---

<a id="static-special"></a>
## 静态专项记录、组件问题与 issue 正文

- [005 · csa_static_spmd_feasibility_20261015/B16_PROGRESS_DIAGNOSIS_20261016.md](#history-document-005)
- [006 · csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md](#history-document-006)
- [007 · csa_static_spmd_feasibility_20261015/CORE_CAPACITY_DECISION_20261016.md](#history-document-007)
- [008 · csa_static_spmd_feasibility_20261015/DIRECT_NATIVE_ENTRY_20261016.md](#history-document-008)
- [009 · csa_static_spmd_feasibility_20261015/DYNAMIC_SPMD_REFERENCE_20261016.md](#history-document-009)
- [010 · csa_static_spmd_feasibility_20261015/ISSUES_AND_PRS.md](#history-document-010)
- [011 · csa_static_spmd_feasibility_20261015/JIT_SINGLE_GROUP_NOTES.md](#history-document-011)
- [012 · csa_static_spmd_feasibility_20261015/LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md](#history-document-012)
- [013 · csa_static_spmd_feasibility_20261015/MEMORY_REUSE_PHASES_REPRO.md](#history-document-013)
- [014 · csa_static_spmd_feasibility_20261015/PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md](#history-document-014)
- [015 · csa_static_spmd_feasibility_20261015/PR3013_REVIEW_20261016.md](#history-document-015)
- [016 · csa_static_spmd_feasibility_20261015/PYPTO_GROUP_SHARED_INOUT_REPRO.md](#history-document-016)
- [017 · csa_static_spmd_feasibility_20261015/PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md](#history-document-017)
- [018 · csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md](#history-document-018)
- [019 · csa_static_spmd_feasibility_20261015/SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md](#history-document-019)
- [020 · csa_static_spmd_feasibility_20261015/SIMPLER_CORE_ARGUMENT_CAPACITY.md](#history-document-020)
- [021 · csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md](#history-document-021)
- [022 · csa_static_spmd_feasibility_20261015/STATIC_BASELINE_COMMIT_20261016.md](#history-document-022)
- [023 · csa_static_spmd_feasibility_20261015/TORCH_SCHEMA_CAPACITY_20261016.md](#history-document-023)
- [024 · csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/JIT_SINGLE_GROUP_NOTES.md](#history-document-024)
- [025 · csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/MEMORY_REUSE_PHASES_REPRO.md](#history-document-025)
- [026 · csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/README.md](#history-document-026)
- [027 · csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/SIMPLER_CORE_ARGUMENT_CAPACITY.md](#history-document-027)
- [028 · csa_static_spmd_feasibility_20261015/evidence/simpler_core256_v1/LOCAL_ARTIFACT_PATHS.md](#history-document-028)
- [029 · csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/REPORT.md](#history-document-029)
- [030 · csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/README.md](#history-document-030)
- [031 · csa_static_spmd_feasibility_20261015/upstream_issues/incore_p2p_notify_wait_20261016/issue_body.md](#history-document-031)
- [032 · csa_static_spmd_feasibility_20261015/upstream_issues/p1_group_inout.md](#history-document-032)
- [033 · csa_static_spmd_feasibility_20261015/upstream_issues/p1_group_inout_initial.md](#history-document-033)
- [034 · csa_static_spmd_feasibility_20261015/upstream_issues/p4_ptoas_branch_view.md](#history-document-034)
- [035 · csa_static_spmd_feasibility_20261015/upstream_issues/p4_ptoas_branch_view_initial.md](#history-document-035)
- [036 · csa_static_spmd_feasibility_20261015/upstream_issues/p5_ptoas_gather_phi.md](#history-document-036)
- [037 · csa_static_spmd_feasibility_20261015/upstream_issues/p5_ptoas_gather_phi_initial.md](#history-document-037)
- [038 · csa_static_spmd_feasibility_20261015/upstream_issues/p7_ptoas_column_carry.md](#history-document-038)
- [039 · csa_static_spmd_feasibility_20261015/upstream_issues/p7_ptoas_column_carry_initial.md](#history-document-039)

<a id="history-document-005"></a>
### 历史原文 005：B16_PROGRESS_DIAGNOSIS_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/B16_PROGRESS_DIAGNOSIS_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/B16_PROGRESS_DIAGNOSIS_20261016.md)；原文 4,539 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-005 -->
# 128K/B16 单 MIX：进度协议诊断

记录日期：2026-10-16。仅针对当前静态 history=131072、B16、每请求 6 query
token（T96）。保留完整静态源码和全部失败快照，不更改数学分片，不扩展档位。

## 已确认的事实

完整 native 任务 `task_20261016_155540_123832232627` 已完成 CCE/调度库编译，
单个 24 组 MIX 已在设备运行，但未完成。Simpler 报告 `S1:running-stalled`：
completed=0/1、running=1、orch_done=1，最终返回 507018。
没有完整静态输出、精度或性能结果。

为定位停滞，`diagnose_static_protocol.py` 从完整 B16 源码抽取控制段，保留
records/waits、核号映射、逐核顺序、代际以及原 cacheinvalid/fence 操作。
计算段按已核对的边界移除，并增加 200000 次有限轮询和失败字段；失败后不发布
虚假的完成标记。只保留 3 个协议 Tensor，使用 program 模式显式 AIC/AIV 类型，
仍是单个 24 组 MIX。这是控制协议诊断，不能替代 132 参数完整 CSA 验证。

## 对照与当前状态

| 尝试 | 改动 | 结果 / 任务 |
| --- | --- | --- |
| protocol_v1 | 初始抽取诊断 | CPU 常量写入类型错误，未上卡；修正 INT64 cast，并改用显式 AIC/AIV。 |
| protocol_v2 | 原控制协议，移除数学并加入有限轮询 | `task_20261016_162004_206538014143` 已结束；设备正常返回，但 72 核均记录等待失败。 |
| protocol_v3_volatile | 仅把诊断生成 C++ 的 progress 标量访问改为 volatile；表及轮询预算不变 | `task_20261016_162414_222408211822` 已结束；3 个代际的控制进度符合预期。 |
| protocol_v4_fence | 只在消费者失效缓存后增加现有 `pl.system.fence()` | `task_20261016_163312_271626813951` 已结束；设备返回，首轮 72 核有限等待失败。 |
| protocol_v5_volatile_read | 仅 progress 读改为 volatile | `task_20261016_164435_356209620998` 已结束；3 个代际的控制进度符合预期。 |
| protocol_v6_volatile_write | 仅 progress 写改为 volatile | `task_20261016_164436_356260019239` 已结束；设备返回，首轮 72 核有限等待失败。 |

以上是去掉业务计算后的诊断观察，不能证明跨核缓存一致性、payload 交接或完整 CSA
正确。当前无本会话待执行的对照任务。用户要求先复核 Scalar 非一致缓存协议，
暂停继续增加接口或上卡；最新解释见
[Scalar 缓存协议复核](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md)。

以上目录位于工作树的 `tests/pypto_test/results/csa_static_spmd_feasibility_20261015/`，
完整目录名以 `static_b16_` 开头。归档文本在本目录 `evidence/`；大产物仅保留
本地路径与大小，不计算内容 hash。

## v2 可以说明什么

例：core0 等待 core32 的 1025，有限轮询记录值一直没有达到目标，最后为
1024；Host 取回的最终快照中 core32 已是 1025。72 核均进入当前初始代际，
其中 17 个失败等待的目标在最终快照中已达到。

这说明移除数学计算后，同一控制表仍能出现等待失败，优先排查协议实现。
但最终快照不包含精确写入时刻，不能仅凭上述差异断言编译器做了 CSE/LICM；
有限轮询预算和执行先后也要由同预算对照排除。

当前生成 C++ 的 progress 读取是普通 GM 指针读取。只读核对发现，现有
PTO-ISA 的 TWAIT/TTEST 对信号读取使用 volatile，并配合缓存与流水操作。
两者表达不同是待验证线索，尚未据此认定 PyPTO、PTOAS 或 ISA 的责任。

CCE 的 `-emit-llvm` 与 `-print-after=licm` 都不受当前工具链支持；
llvm-objdump 输出为 `not available`。失败命令和输出已保留，没有可用的
优化 IR 或汇编证据。

## 接续边界

全部对照已结束，不能沿用此前“只有 volatile 有效就推进组件修复”的推断。
读侧 volatile 改变诊断结果，但它不保证绕过 Scalar DCache；尚无优化 IR 或汇编
证据可认定具体编译器错误。写侧单写者不因存在多个读者就必须使用 atomic，
需要分别核对 DCCI + DSB 发布、读取路径和业务数据同步，保留原逐前驱依赖。
当前仅整理协议审查结果，不继续实现或测试。

协议通过后仍必须返回完整静态 128K/B16 native 验证，检查数学输出、六份
cache/state、保护区和跨调用代际。协议诊断的成功不等于完整 CSA 成功。
本轮尚未修改 PyPTO、Simpler 或 PTOAS 的组件源码。

<!-- SOURCE_END history-document-005 -->

---

<a id="history-document-006"></a>
### 历史原文 006：BINARY_COMPILE_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md)；原文 2,497 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-006 -->
# 完整 B4 单 SPMD：二进制编译与组装结果

2026-10-16。等待用户决定 Torch 入口方案期间，继续验证与该选择无关的
现有 132 参数核；保留现行接口，没有实现 56+6 workspace 候选。

## 结果

`static_b4_binary_v1` 通过。复用 `static_b4_jit_cpu_v21_carry_scoped` 的
默认 PYPTO 代码生成产物，在新的结果目录编译，不修改原始证据：

- AIC：CCE 编译、设备链接、提取可装载代码成功，58548 B。
- AIV：CCE 编译、设备链接、提取可装载代码成功，146956 B。
- AICPU 调度：交叉编译共享库成功，861416 B。
- 两个 132 参数 CORE Callable 与 132 参数 ChipCallable 组装成功。
- 总编译和组装耗时 115.418 秒；这是 Host 编译时间，不是算子延迟。

使用独立 PyPTO/Simpler worktree 及既有 CANN 9.2.0-beta.2 工具链。
原始命令包含 CCE 的 Cube/Vector 架构、优化和头文件参数，保存在
`session_logs/csa-static-b4-binary-v1.log`。没有修改 PTOAS/PTO-ISA 或共享安装。

## 验证范围

这是从 JIT 源码产生的 **program 模式产物** 的完整二进制编译和内存中
callable 组装。不创建 Worker，不加载设备，不调用 Torch 注册接口。
因此它补齐“代码生成之后能否由 CCE 编译”的证据，但不代表完整 CSA 的
单次设备执行、跨核进度协议、状态、数值或性能已经通过。

当前 Torch 64 入参限制仍在，见
[注册限制与候选方案](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/TORCH_SCHEMA_CAPACITY_20261016.md)。
目标保持 active；按原目标的关键点讨论要求，接口方案仍待用户答复。

## 复现及产物

- [编译驱动](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/compile_existing_binary.py)。
- [结构化结果](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_binary_v1/result.json)。
- 可复现命令：先按 [独立组件环境](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md#独立环境复现)
  设置环境，在本目录执行：

```bash
python bootstrap_isolated.py compile_existing_binary.py \
  --artifact ../results/csa_static_spmd_feasibility_20261015/static_b4_jit_cpu_v21_carry_scoped/kernel \
  --output /tmp/csa_static_b4_binary_repro
```

`--artifact` 应指向原开发 worktree 的现存完整产物；主仓归档不复制大产物，
其绝对路径见结果 JSON 的 `source_artifact`。输出目录必须不存在，避免覆盖
旧尝试。二进制仅留原结果目录，路径和大小进入 `artifact_inventory.json`，
不提交二进制、不额外计算内容 hash。归档不会重跑这个编译。

<!-- SOURCE_END history-document-006 -->

---

<a id="history-document-007"></a>
### 历史原文 007：CORE_CAPACITY_DECISION_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/CORE_CAPACITY_DECISION_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/CORE_CAPACITY_DECISION_20261016.md)；原文 6,264 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-007 -->
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
参数容量这一组件目标完成，详见 [扩容验证记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md)。PyPTO、PTOAS、PTO-ISA 和 pypto-lib 未修改。

<!-- SOURCE_END history-document-007 -->

---

<a id="history-document-008"></a>
### 历史原文 008：DIRECT_NATIVE_ENTRY_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/DIRECT_NATIVE_ENTRY_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/DIRECT_NATIVE_ENTRY_20261016.md)；原文 3,655 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-008 -->
# CSA 大参数入口：用户选择直调与 TorchAir 参考核对

2026-10-16。用户明确要求直接使用 pybind 路径，并提醒参考 TorchAir 的多种接入方式。
此前“56 Native + 6 workspace”的 62 项入口方案取消，不为绕过 Torch schema 改写 CSA 参数。

## 两层接口不能混为一谈

当前 Torch build 的 dispatcher 在 DispatchKeyExtractor 中使用 64 位 bitset 标记 schema
参数位置，因此 132 项具名 schema 无法定义。这是 torch.library 注册限制，
不代表 Python→C++ binding、Simpler CORE 或 A3 限制为 64 项。

本地 PyPTO 已经有直接调用路径，无需先增加新的 C++ 导出：

1. `JITFunction.__call__`（`python/pypto/jit/decorator.py:3374`）解析实参并获取 kernel artifact。
2. `python/pypto/torch/launch.py:88` 的 `invoke` 进入现有 Worker / capture 生命周期。
3. 同文件 `native.prepare` 接收 Tensor 列表、dtype、scalar、当前 stream 和 device。
4. `python/bindings/torch/torch_npu_adapter.cpp` 使用 nanobind 的 `NB_MODULE` /
   `m.def("prepare", ...)` 导出；Tensor 列表按 CHIP 容量检查，不定义 132 项 Torch schema。
5. 同 adapter 的 `OpCommand::RunOpApiV2("PyPTOKernel", ...)` 把启动放入 torch_npu task queue，
   lambda 内使用 `stream(false)`。已有 Tensor/storage 持有及 capture 处理应直接复用。

这里用户所说的“pybind 直调”对应已有的 native binding；该实现实际使用 nanobind。
`pypto.torch.register` 是另外一层可选封装，不应把其 schema 限制误写成整个 PyPTO
kernel 调用的限制。此处仅完成源码核对，尚未把完整 132 参数 CSA 的设备执行记为通过。

## TorchAir 示例实际展示了什么

用户提供的 GitCode 页面本次读取失败，因此核对的是 Ascend 官方 GitHub 仓库对应目录，
记录的 master commit 为 `e1118e8b0fc1b95190ab53b732c0872989550e20`。
未声称两个站点在本次读取时的分支头完全一致。

- [pybind 示例](https://github.com/Ascend/torchair/tree/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/pybind)：
  C++ 函数通过 PYBIND11_MODULE 导出；其示例测试还在 Python 层包了一层 torch.library / Meta。
  因此不能仅因目录名为 pybind 就说示例的 torch.compile 路径完全不经过 dispatcher。
- [torch_library 示例](https://github.com/Ascend/torchair/tree/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/torch_library)：
  在 C++ 中定义 schema、PrivateUse1 实现及 Meta 实现。
- 两组示例均演示 NPUGraph、make_graphed_callables、npugraph_ex 三种调用环境。
- [启动方式说明](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/pybind/csrc/add_custom.asc)
  列出八种内核启动组合，包括正确方式与错误对照。注册方式和 task queue/stream 启动方式是两层问题。
  当前 PyPTO adapter 已采用其中对应的 stream(false) + OpCommand 入队模式；
  不需要改成裸 stream(false) 启动。

## 接续边界

保留 132/133 个 CORE 描述符，优先调用已有 JIT native 路径验证完整 kernel。
先做单次功能、状态与保护区，再检查原始 NPUGraph 捕获/重放。
整网 torch.compile / npugraph_ex 的外层 opaque op、Meta 及 mutation 语义单独核对；
直调接口存在不等于这些组合已经全部验证通过。

用户随后要求优先提交必要 bug issue，并让扩容等改动继续留在本地；
本轮没有为了入口直调修改依赖源码，没有新设备性能数据。

<!-- SOURCE_END history-document-008 -->

---

<a id="history-document-009"></a>
### 历史原文 009：DYNAMIC_SPMD_REFERENCE_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/DYNAMIC_SPMD_REFERENCE_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/DYNAMIC_SPMD_REFERENCE_20261016.md)；原文 4,762 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-009 -->
# CSA 单 SPMD 动态范围与模板边界：后续参考

记录日期：2026-10-16。来源：用户转来的另一会话分析；本会话只读核对了下述
代码位置。本文件是后续设计参考，不是新增功能或验证通过记录。

## 当前决策

已完成的静态代码、失败快照和验证结果全部保留，不删除。当前仅验证静态
128K/B16（history=131072、B=16、每请求 6 个 query token、T=96）。动态 batch
和 seqlen 的完整支持后续处理，不因本记录而扩大本轮测试范围。

目标是固化较优执行安排，同时保留每步动态工作量。静态模板并不意味着其内部
所有数值和循环次数都成为常量；核内存在动态读取，也不等于整个入口已支持
任意 batch、shape、长度策略和工作区容量。

## 已有动态计算分布

PTO 有自己的工作划分和动态范围计算，没有直接复用 Native SAS/QLI 的调度
metadata；融合为单 SPMD 时，这些职责仍须保留。

| 职责 | 当前代码与作用 |
| --- | --- |
| 选择 Indexer 长短档策略 | `vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_csa.py` 的 `_read_device_max_seq_len`（核对时约第 135 行）：TMR 编排从 device lengths 读取最大长度，进入相应分支。文件另有 `_use_host_max_seq_len` 和 `host_metadata` 工厂选择；不能把 TMR 的 device-read 描述套用于 HBG。 |
| 分配 query 与历史分片 | 同目录 `decode_indexer.py`，`indexer_score_topk_native_pair` 附近（约第 688 行）：读取 `kv_seq_lens`、`position_ids`，计算最大叶片数、各 query 可见候选范围、尾块和循环次数。 |
| 准备 attention 有效块 | 同目录 `decode_sparse_attn_csa.py` 的 `csa_slots_build_valid_qk_plan`（约第 207 行）：由本轮 Top-K、位置及相关 metadata 生成 sparse indices、bias、valid-block mask，供 QK/PV 消费。该阶段融合后仍需实际执行。 |

以上路径以 `vllm-ascend-dsv4-pto-0251rc1` 为根。行号会随修改变化，以函数名和
变量名定位为准；静态生成使用的计算基线仍是已冻结的 baseline_ops，不能假定
主仓后续修改会自动进入现有模板。

## 后续设计区分三层

1. 初始化或编译时确定数学 tile、核数、各核阶段安排、依赖关系及容量边界。
2. 每步从现有输入 Tensor 读取长度、位置、页表、Top-K，计算实际分片、访问地址、
   有效尾块和循环次数。不能把这些读取替换为捕获那一步的数值。
3. 将原跨 task 的必要依赖转为 SPMD 内部同步，保证生产者写入完成并可见后，
   消费者才读取；不能仅删除原依赖，也不应无依据增加全卡屏障。

在同一模板覆盖的形状、策略和容量范围内，每步长度增长原则上只需更新现有
device Tensor 内容，不要求每步重新编译，也不由此推出需要新增 Host
tiling-update 接口。这是设计预期；现有静态整层尚未通过，不能声称已验证。

## 当前 B16 模板已经保留的部分

[`static/h131072_b16/entry_jit.py`](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/static/h131072_b16/entry_jit.py) 中仍有：

- 约第 1050 行读取 `arg_kv_seq_lens` 并计算 Indexer 历史范围；用户转述版本约为
  第 1047 行，本轮 wrapper 变化造成行号偏移。
- 约第 1100～1101 行读取 `arg_position_ids` 与当前请求长度，决定候选可见范围。
- 后续 attention 有效块准备仍消费本轮位置和 Top-K，未被替换成历史快照。

与此同时，生成器仍特化 Tensor shape、部分循环边界、原逻辑 block 数及捕获到
的编排分支。当前七档模板不能直接代表单个完全动态的 CSA 实现。

## 后续应核对的边界与性能问题

- 跨越 Indexer 长短策略边界时，核对是否需要切换模板，或在单 SPMD 内补齐
  动态分支及对应依赖。边界条件必须来自实际代码，不能只按历史长度标签判断。
- batch/token shape 变化及工作区容量增长是另外的边界；“更新 Tensor 内容”
  不能自动解决描述符 shape 或容量不足。
- 同档内请求长度分布变化可能使固定逐核安排出现长尾。动态长度处理正确与
  分核均衡是两件事，不能把不均衡直接归因为缺少每步 Host tiling 更新。
- 后续以边界场景核对功能，再评估分配策略和性能；本轮不据此启动全量测试。

## 与当前执行停滞的关系

静态 128K/B16 的首次设备启动已报告 running-stalled。现有日志只证明单 MIX
任务发射后没有完成，尚未定位到具体核内阶段。没有证据将其归因于本文件提到
的动态模板边界或负载分布；当前故障定位与后续动态化设计分别记录。

<!-- SOURCE_END history-document-009 -->

---

<a id="history-document-010"></a>
### 历史原文 010：ISSUES_AND_PRS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/ISSUES_AND_PRS.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/ISSUES_AND_PRS.md)；原文 8,772 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-010 -->
# CSA 单 SPMD：问题、issue、PR 与本地修复登记

2026-10-16 核对。登记只覆盖本次把 CSA 动态排程固化成单个 SPMD 的工作。
“已发 issue”“已发 PR”“仅本地修复”分别列出，不能互相替代。

2026-10-16 用户另行明确授权提交 InCore 内点对点通知/等待功能需求，已创建
[PyPTO #3020](https://github.com/hw-native-sys/pypto/issues/3020)，作者 nalinaly，
状态 OPEN。与下面四份编译正确性 bug 分开登记，不改变本地组件补丁的提交范围。
正文、主线源码依据和回读结果见 `upstream_issues/incore_p2p_notify_wait_20261016/`。

## 此前用户收窄后的 bug 提交范围

只提交确定的编译正确性 bug issue。下面四项已经创建并回读核对标题、正文和状态；
代码补丁仍在本地，没有 push PyPTO/Simpler 分支或创建修复 PR。

| 本地编号 | 已提交 issue | 必要性与范围 |
| --- | --- | --- |
| P1 | [PyPTO #3014](https://github.com/hw-native-sys/pypto/issues/3014) | Group 共享 InOut 的 SSA 别名未绑定，直接阻塞当前 JIT 路径 |
| P4 | [PyPTO #3015](https://github.com/hw-native-sys/pypto/issues/3015) | PTOAS 分支后 view 使用错误的 SSA 句柄；备用路径正确性 |
| P5 | [PyPTO #3016](https://github.com/hw-native-sys/pypto/issues/3016) | 条件 gather 同存储 phi 生成非法 Mat→Mat；备用路径正确性 |
| P7 | [PyPTO #3017](https://github.com/hw-native-sys/pypto/issues/3017) | 行缓冲/列 carry 的 TMOV 形状不符；备用路径正确性 |

P2/P3/P6/P8/P9 的内存规划和状态搬运改进先留在本地；Simpler S1/S2 的
CORE 32→256 Tensor、64 位 dump 掩码扩展、派发及相关 ABI 也按用户指示留在本地。
`runtime` 子模块和 `_kernel_abi.py` 的修改只是独立环境与扩容 ABI 配套，不是另一个 bug。

Simpler S3 已有 [issue #2331](https://github.com/hw-native-sys/simpler/issues/2331)
与已合入的 [PR #2340](https://github.com/hw-native-sys/simpler/pull/2340)。主线使用
CCE depfile 跟踪头文件；本地调试分支缺少此改动，显式依赖仅作当前环境修复，
后续应优先采用主线实现，不新建重复 issue。

四份完整复现、正文、发布记录和回读结果保存在 `upstream_issues/`。
五个独立 CPU 用例在本地修复版全部通过；没有重跑完整设备或七档性能。
相同复现在未打本地补丁的 PR #3013 精确 head 中全部失败；该对照已补入四份 issue。
创建请求指定 bug 标签，但 GitHub 回读没有应用标签；标题和完整正文均一致，
没有为此重复创建 issue。一次 GraphQL 回读 EOF 后改用 REST 回读确认。

## 已提交的上游 issue

| 项目 | 链接与状态 | 问题、处理方式、证据 |
| --- | --- | --- |
| PyPTO #3009 | [issue](https://github.com/hw-native-sys/pypto/issues/3009)，OPEN，作者 nalinaly，2026-10-15 创建 | 自动 MIX 展开删除仅用于 while condition 的 iter_arg，产生 FREE_VAR。最小复现、原/预期 IR、命令见 PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md。当前显式 Group 两半路径绕开自动拆分步骤，不代表这个 issue 已修复 |
| PyPTO #3020 | [功能需求](https://github.com/hw-native-sys/pypto/issues/3020)，OPEN，作者 nalinaly，2026-10-16 创建 | 单个 InCore/SPMD 内，基于普通 GM 信号的点对点 notify/wait；明确动态阈值、单写者多消费者、代际、信号与 payload 内存语义、MIX lane 副作用及验收范围。尚未实现 API |

创建请求、创建结果、首次回读分别保留在 `issue_payload.json`、
`issue_creation_result.json`、`issue_verified.json`。
本次 GitHub 回读保留在 `upstream_records/csa-pypto-issue-3009-current.json`。
此前登记时尚未观测到关联 PR；现已按用户指示拉取
[PR #3013](https://github.com/hw-native-sys/pypto/pull/3013)，head
`72b7ffb98ea3cee12bf7bbf533ab89b848a8980c`，在独立 detached worktree 核对。
该 PR 修复 #3009 的 while 条件活跃性，和本轮四个新 issue 区分登记。
独立构建通过，七个定向回归通过；本会话原始 scalar/tile/control 三种复现
均完成代码生成且无 ready FREE_VAR，见 [PR 核对记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PR3013_REVIEW_20261016.md)。

## PR 登记

本轮尚未提交 PyPTO 或 Simpler PR，也未 push 这两个独立修复分支。
分别按下面两个 head 查询全部状态 PR，返回均为空，原始结果在
`upstream_records/csa-pypto-pr-current.json` 与
`upstream_records/csa-simpler-pr-current.json`：

- PyPTO：`feat/csa-static-spmd-fixes-20261016`。
- Simpler：`feat/csa-core-args-256-20261016`。

后续创建 PR 时在此追加真实 URL、head/base、提交、验证范围和合并状态；
用户本次要求记录 PR，不等于已经存在 PR。

## 独立组件的本地修复

| 编号 | 问题 | 修改与状态 | 证据 |
| --- | --- | --- | --- |
| S1 | CORE 只有 32 Tensor，当前需要 132/133 | 已扩 256 Tensor + 16 scalar；上下文、派发、参数池、图缓存、依赖记录同步 | SIMPLER_CORE_EXPANSION_20261016.md；本地 commit `57e499238fd4ee3548947ad860be456aaeeb16b5` |
| S2 | 参数 dump 单 uint64，派发结构固定 512 B | 272 位分 5 字掩码；派发 2304 B，关联布局重新计算 | 同一 S1 commit；高位 dump 和四类 runtime/device 验证通过 |
| S3 | 修改 ABI 头文件后旧 CCE kernel 可能被缓存复用 | 增量编译依赖加入 ABI 头文件 | 同一 S1 commit |
| P1 | Group 内调用返回别名未映射到统一形参 | 使用 return_lineage 追踪完整 InOut 别名；同步 tuple/Submit/no_dep | 独立 PyPTO 本地修改；CPU 回归及最小共享 InOut JIT 真机 PASS |
| P2 | 跨阶段 48+12 / 32+32 KiB 无共同最大 root，错误占用 80 KiB | 超限时有界扩展复用区；最终分配按最大 offset+size 保留 | 三参数与真实 Indexer+O-proj 13 参数代码生成 PASS；三参数真机四阶段 max_abs=0 |
| P3 | 内层 phi 更新后纯别名仍保留旧缓冲 | 同遍历中即时跟随，防止外层选回旧 storage | 嵌套 phi 回归；完整 B4 Vec 减少 64 KiB |
| P4 | PTOAS view 仅按类型省略，引入不支配使用的 SSA | 要求实际源句柄一致，否则生成 reshape view | 完整 B4 两处 SSA 错误消除 |
| P5 | 同一物理窗口的条件 gather 仍引入 Mat→Mat 拷贝 | phi 保持同一窗口；多类型 identity 只复用已声明且类型完全匹配的 handle | Mat→Mat 错误消除；含 transpose+matmul 消费者的回归通过 |
| P6 | 分支原地循环的别名未加入对应 phi family | 只标记分支内同 allocation 的别名；保留外部独立读取约束 | gather 正/反例通过，两块 64 KiB 合并 |
| P7 | loop-carry 列向量向行 handle 搬运，shape 不匹配 | 先生成同存储的目标列 view，再搬运 | 列 carry 回归；完整 PTOAS 路径进入实际内存分配 |
| P8 | then→else 线性顺序造成假干涉 | 仅超限时加一次反向分支遍历；同分支/循环后续读取仍禁止复用 | 完整 Vec 197632 B，仍超 9216 B；相关三文件 378 项通过 |
| P9 | 旧 MemRef 的纯别名阻断循环 carry 目标传播，生成冗余状态搬运 | 只在循环 carry 追踪中允许越过旧 identity 窗口；累加器合并保持原门槛 | 最终相关两文件 169 项通过；QK/PV 恢复历史 181248 B；完整 B4 v21 代码生成 PASS |
| T1 | 当前 PyTorch build 的 schema 最多 64 项，完整入口为 132 项 | 用户已选择现有 JIT native 直调路线，62 项候选取消 | DIRECT_NATIVE_ENTRY_20261016.md；没有修改 PyTorch，没有为此提 issue/PR |
| P10 | 普通 GM 信号缺少单 SPMD 核间 notify/wait 接口，属于功能需求 | 独立 PyPTO 临时 Set/Ge API，复用 TNOTIFY/TWAIT；本地保留，未提交 PR | #3020；LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md；与前九项修复分开 |

PyPTO 修改尚未 commit。补丁快照与基线/分支/状态见
`component_changes/pypto/`；Simpler 提交全文见
`component_changes/simpler/`。修改没有覆盖共享安装，没有改 PTOAS、PTO-ISA 或 pypto-lib。

## 仍待解决

- 完整 B4 默认路径代码生成和 program 二进制组装已通过；完整 B16 kernel 模式
  native JIT 直调现已三次通过，对冻结动态 PTO 输出/Top-K/cache/state 精确一致。
- PTOAS 正式路径在 Mat 分配超 16 KiB；DSA_RP 正式路径在 AIC 无容量内布局。
  没有把它们认定为硬件不可实现，也没有放宽硬件容量。
- 临时通知/等待及本核流水等待已在 B16 验证进度、状态与保护区；动态模板、
  ACLGraph、连续 decode 轨迹及性能仍在本轮验证范围之外。
- #3009 自动 MIX 的 while 活跃性问题未在本分支修改，不标为修复。

<!-- SOURCE_END history-document-010 -->

---

<a id="history-document-011"></a>
### 历史原文 011：JIT_SINGLE_GROUP_NOTES.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/JIT_SINGLE_GROUP_NOTES.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/JIT_SINGLE_GROUP_NOTES.md)；原文 4,101 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-011 -->
# JIT 的单 MIX Group 表达：当前核对结果

目标仍为一次 24 组 MIX 提交，不用两个提交替代。本说明记录接口表达限制，
不是宣称所有 JIT 写法均不可实现，也不要求组件必须新增某一个具体装饰器。

## 已验证的路径

显式 program 原型包含 `FunctionType.Group`，根入口 SPMD 只调用这个 Group。
其部分 orchestration 产物已确认只有一次 MIX 提交；完整核编译问题另见
[COMPILE_AND_RESUME.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md)。

直接将它改写为 JIT 的以下结构，会在解析阶段被拒绝：

```python
with pl.spmd(24, sync_start=True):
    with pl.cluster():
        c = cube(a, b, c)
        y = vector(x, y)
```

错误为 `dispatch body launches 2 kernels; only one kernel dispatch per SPMD scope`。
源于 parser 对 dispatch body 的检查；后端 `FindFirstInnerCall` 只能取一个 callee。
不能通过新增无用 block_idx 读取欺骗该检查。

进一步检查了先用 `@pl.jit.opaque` 包裹 cluster，再让 SPMD 只调用一次 wrapper：
[repro_jit_group_wrapper.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_jit_group_wrapper.py)。它是 5 参数、无模型依赖、
无 while 的 CPU 编译小例子。两个物理计算核可生成，但 orchestration 拒绝：

```text
InferFunctionCoreType expects AIC or AIV (Spmd/Group are filtered upstream);
got Opaque on function 'group_wrapper'.
```

证据：`../results/csa_static_spmd_feasibility_20261015/jit_group_wrapper_v1/`。
这说明该 opaque wrapper 不能当作已支持的 Group kernel 入口。

## 本地源码依据与接续选择

- `python/pypto/language/parser/ast_parser.py`：SPMD 的单 callee 检查。
- `src/ir/transforms/outline_cluster_scopes_pass.cpp`：一个 cluster 只允许一个
  嵌套 SPMD，不能用两个 SPMD 分别包住 AIC/AIV 再冒充一次 MIX。
- `python/pypto/jit/decorator.py`、`specializer.py`：现有 JIT 类型没有原生
  DSL Group 装饰器；extern mixed 可以生成 Group，但需已有正确的物理核产物。
- `tests/ut/jit/test_roundtrip.py` 明确将模块级 `pl.function` 调用列为 JIT
  依赖发现不覆盖的模式；不能直接把 program 的 IR 函数塞给 JIT 当作完成接入。

后续可选组件正式支持的 Group 调用表达，或在 #3009 修复后验证自动 MIX 内联
表达。若采用 extern mixed 承接 PyPTO 生成的两份物理核，仍须验证同一参数布局、
双 AIV、一次 24 组提交及 kernel runtime/当前流，不能用 program runner 替代。

## 2026-10-16：inline wrapper 推进到共享 InOut 限制

`repro_jit_group_inline.py` 用现有 `@pl.jit.inline` 包装 cluster：SPMD 前端只见一个
wrapper 调用，内联之后 outliner 建立 Group。5 参数、不共享 InOut 的对照已完整
CPU 编译通过，只有一次 24 核组 MIX，sync_start 保留。`pypto.torch.register`
及 Meta 返回别名验证也通过，证据为 `jit_group_inline_v1`、`jit_group_inline_meta_v1`。

这个结果不能直接推广到 CSA：CSA 的 AIC/AIV 共享 InOut 缓冲。按 JIT 的要求
接回每次调用的返回值后，`ExpandMixedKernel` 的 Group ABI 统一没有把返回别名
追溯到 Group 参数，导致 free variable。5 参数最小失败例和首个失败 pass 已保存，
见 [共享 InOut 复现](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_GROUP_SHARED_INOUT_REPRO.md)。

生成器现在为 B4 生成 inline wrapper，并按同名参数接回两次成员调用的返回值。
`static_b4_jit_cpu_v6_alias` 通过入口结构及 InOut 检查，之后 codegen 报
`Tensor view not found for parameter: progress__ssa_v1`。v5 使用调用前别名，被
InOutUseDiscipline 拒绝；v4 是本地生成脚本 import 路径失误，仍编译了旧源码，
不构成新 inline 表达的失败证据。

非共享 InOut 的设备小例子正在排队：`task_20261016_095538_391201623768`，
脚本 `test_jit_group_inline_device.py` 验证注册入口、调用方 stream、双 AIV、三次
执行和完整输出保护区。结果以 `jit_group_inline_device_v1/result.json` 为准，
本段不预记 PASS。完整 `entry_jit.py` 仍不是生产入口。

<!-- SOURCE_END history-document-011 -->

---

<a id="history-document-012"></a>
### 历史原文 012：LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md)；原文 10,434 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-012 -->
# 普通 GM 核间 notify/wait：临时实现与验证

用户于 2026-10-16 明确要求先实现临时接口，未来主线
[PyPTO #3020](https://github.com/hw-native-sys/pypto/issues/3020) 落地后替换。
此前因 Scalar 非一致缓存讨论而暂停的实现工作已恢复。

本次只在独立 PyPTO worktree 增加接口，并在测试目录建立 128K/B16 候选；
没有切换共享安装，没有改 Simpler、PTOAS、PTO-ISA 或生产 CSA。
旧 `static/h131072_b16` 完整保留。

## 接口与替换边界

```python
pl.system.signal_notify(signal, [offset], value)
pl.system.signal_wait(signal, [offset], expected)
```

- signal 是普通一维 GM INT32 Tensor，不要求 DistributedTensor 或通信窗口。
- offset 支持设备标量；value / expected 必须为 INT32。Python 整数越界立即拒绝；
  动态标量必须由调用方保证范围，不能把溢出的 INT64 静默当成正确进度。
- notify 是单写者 Set；wait 是非消费式 >= 等待。多个消费者可以独立等待同一信号。
- 每条信号缓存行只允许一个写者，不与 payload、其他写者共享。
- 仅用于明确 AIC/AIV 核体；没有新增自动 MIX lane 放置规则。
- 所有必要生产者须可继续执行，不能用阻塞轮询等待尚未驻留的后续调度波次。
- 首次执行前初始化；代际单调递增。并发调用使用独立工作区，溢出前排空后重置。

CSA 通过 [local_signal_adapter.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/local_signal_adapter.py) 中两个 inline 函数调用。
主线接口可用后，优先只替换这两个函数；若主线改用不同信号类型或槽布局，
同时调整候选生成器的 progress section，不能机械替换名称。

PyPTO 改动位置（均在独立 `pypto-csa-static-fixes-20261016` worktree）：

| 文件 | 职责 |
| --- | --- |
| python/pypto/language/op/system_ops.py | pl.system 的 DSL 入口 |
| python/pypto/ir/op/system_ops.py | IR 构造、整数常量范围检查 |
| src/ir/op/sync_ops/sync.cpp | 普通 Tensor / INT32 / 一维 offset 类型验证；SyncOp 副作用 |
| src/backend/common/pto_ops_memory.cpp | 普通本地 GM 单元素视图，下沉到现有 TNOTIFY/TWAIT |
| tests/ut/language/parser/test_system_ops.py | round-trip、错误类型和越界检查 |
| tests/ut/codegen/test_pto_codegen_ops.py | 等待/通知顺序保留，无通信上下文、无新增 syncall |
| docs/en/dev/ir/05-operators.md、docs/zh/dev/ir/05-operators.md | 同步更新临时接口合同 |

## 底层复用及缓存语义

当前 A3 `pto::comm::TNOTIFY(..., Set)` 执行：
单行 DCCI → volatile GM 写 → 单行 DCCI → DSB_DDR → PIPE_ALL。
本原型只用 Set，未使用 AtomicAdd；单写者缓存行不需要并发加法。

`TWAIT(..., GE)` 使用 volatile GM 读取，并在每次检查前对信号行做 DCCI；
等待期间周期执行 PIPE_ALL，底层有 100000000 次轮询的断言上限。
它不是 ld_dev 版本，也不能从 volatile 推导出硬件 cache 一致性。
两者只接受 INT32，所以没有直接套在旧 INT64 进度 Tensor 上。
底层已有能力的复用不等于 A5 或任意协议已经验证。

**信号到达和 payload 可读分开处理。** 信号接口不承诺自动刷新所有业务数据：

1. 生产者完成该阶段计算后执行 fence → 全缓存 cacheinvalid → fence。
2. CSA notify 薄封装先执行 `bar_all()`（本核 PIPE_ALL），排空本核前序流水，
   然后执行 signal_notify。这一步是小例对照后补齐的，不能只依赖第 1 步的 gm fence。
3. 消费者等待必要前驱：signal_wait。
4. 消费者进入计算前：fence → 全缓存 cacheinvalid → fence。

第 4 步最后一个 fence 是本次对旧候选的补充，明确等待缓存维护完成。
第 2 步的 PIPE_ALL 只作用于当前物理核，不是 syncall，也不要求 72 个核集合；
它会排空当前核前序工作，后续性能优化需要在正确性成立后再评估更细粒度的等待。
全缓存 DCCI 会清理并失效，可能写回 dirty line；并不是任意多写者数据都可安全套用。
当前 B16 的数学体 Scalar 写落在 temporary_14/15/54：
前两者分别独占 64 B、由同一个 row-offsets worker 写；
temporary_54 每行 16 个 INT32（64 B），按 8 行整块分配，
没有两个 Scalar 写者共享该行。真实 arena 基址另加 512 B 对齐检查。
上述核对是当前候选的使用前提，不外推为通用缓存协议证明。

## B16 候选

[prepare_local_signal_candidate.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/prepare_local_signal_candidate.py) 生成独立目录
`static_local_signal/h131072_b16/`：

- history=131072，B16，S6，T96；正式权重沿用现有驱动。
- progress 从 4608 个 INT64 改为 9216 个 INT32；总字节数仍为 36864。
  每 worker 仍独占 512 B，工作区其他 offset 和总量不变。
- 代际仍为 `generation * 1024 + completed_sequence`。
  测试前检查计划代际未超出协议 INT32 范围。
- AIC 40 个、AIV 44 个计算段的 AST 数学体逐段相等；
  只换通知/等待并补消费者 fence，没有改 Tile、矩阵计算、原事件或依赖表。
- 外部 130 Tensor、Core 132 Tensor；仍 native JIT 直调，不走 Torch schema。
- CPU 生成代码确认单个 24 组 MIX，sync_start=true，无新增全体 barrier。

原静态版本及本候选都不是最终 dynamic batch/seqlen 实现。

## 当前验证记录

| 项目 | 结果及边界 |
| --- | --- |
| 独立 pypto_core 构建 v1 | 常量重载歧义，未通过；改为显式 int64_t |
| 独立 pypto_core 构建 v2 | 通过；没有覆盖共享安装 |
| 定向单测 v1 | xdist worker 误导入共享 PyPTO，5 项失败；不是新接口逻辑失败的证据 |
| 定向单测 v2 | bootstrap 隔离 + 单进程，5 passed；262 deselected |
| 错误类型收窄 v3 | 两项 dtype 拒绝用例使用 ValueError，2 passed；55 deselected |
| 完整 B16 CPU kernel 编译 | static_b16_local_signal_cpu_v1，通过；确认 TNOTIFY/TWAIT 已进入 C++ |
| 完整 B16 设备 | task_20261016_174712_328046030630 首轮正常返回；72 核进度及保护区通过，输出和 cache/state 失败，exit=1 |
| 独立 payload 小例 CPU v1 | INDEX/INT32 算术生成类型不一致，未上卡；测试显式区分 INDEX 计算与 INT32 通知阈值 |
| 独立 payload 小例 CPU v2 | local_signal_payload_cpu_v2 编译通过 |
| 独立 payload 小例设备 v1 | task_20261016_175638_384060212210 正常返回但数据错，5376/18432 个 checked 元素不符 |
| 小例诊断 v2 | task_20261016_175919_40304807000：72 信号正确、最终 mm 正确，但消费者读到旧矩阵值 -999；对应 Vector 输出 -998 |
| 小例修正 v3 | task_20261016_180315_11129226453：notify 前加入本核 PIPE_ALL，三代际信号、数据、保护区全部通过 |
| 完整 B16 修正复测 | task_20261016_180421_21978513378，exit=0；三次调用的输出、Top-K、六份 cache/state 对冻结动态 PTO 逐项完全相同，进度及保护区通过 |

完整设备 v1 原计划检查三次串行调用；首轮失败即保存现场，第二/三轮没有执行。
该失败版输出有 20480 个非有限值，不能当成普通末位误差，也不能仅因进度正确排除 payload 问题。
1.8 GiB 的 failure_states.pt 留在原结果目录，文本诊断见 summary.json /
failure_by_rows_cpu.json。v1 不记完整 CSA PASS。

补齐通知前 PIPE_ALL 后，v2 的完整 128K/B16 **三次调用通过**。
生成 C++ 的 86 个 TNOTIFY 静态调用点都确认前置本核 PIPE_ALL。
三轮均与冻结动态 PTO 的输出、Top-K、六份 cache/state 完全一致，无超差；
72 核代际、Native/metadata 未写区域及工作区保护区均通过。
这是完整 attention 半层的静态模板、正式权重与合成历史验证，
没有 MoE/EP16、动态模板覆盖、ACLGraph 或性能结果。
每次调用前恢复 fixture，signal 保留并进入下一代际；不是连续 decode 状态轨迹验证。
与 Native 的关系仍以冻结动态基线自身的差异记录为准，不把本轮 PASS 外推为
Native 与 PTO 全部逐 bit 相同。结果目录 summary.json 清楚保留这层区别。

[test_local_signal_payload.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/test_local_signal_payload.py) 独立验证 24 组 MIX 中
AIC→AIV（两个消费者）、AIV→AIC、跨组 AIV0→AIV1 三种交接；payload 跨多缓存行，
覆盖 Scalar 写、矩阵/MTE 写、Vector/MTE 写、MTE 读与后续 Scalar 读。
连续三个代际保持信号地址不变、不清零旧 payload，独立检查输出与首尾保护区。
该小例没有 CSA 数学，不能替代完整 B16 验证。

小例的首轮失败不是缓存行多写者冲突：Scalar payload 正确，最终矩阵输出也正确，
但部分 AIV 在接收 AIC 信号后读到了尚未更新的矩阵数据。生成的 AIC gm fence 是
MTE2/MTE3/FIX 分别的 pipe_barrier 加 DSB，缺少 Scalar 发布前的本核流水汇合。
只在 notify 前加入 PIPE_ALL 后，小例三个代际通过；这个对照支持上述定位，
不代表任意协议、A5 或完整 CSA 已经通过。

官方语义参考：[CANN asc_sync_pipe](https://asc.gitcode.com/api/SIMD-API/c_api/sync/intra_core_sync/asc_sync_pipe.html)
说明 PIPE_ALL 等待本核此前提交的各流水操作完成。不能把单流水 barrier 或 Scalar DSB
直接等同于这个完成条件。旧 adapter 保存在 local_signal_variants/v1/，
各结果目录另保存实际使用的 adapter 源码。
本轮不测性能，不重测 B4 或其他档位。

原始构建/单测日志归档于 session_logs，完整结果在 evidence 对应目录；
大文件与设备二进制保留原路径，清单见 artifact_inventory.json，不做 hash 校验。

## 复现命令

以下在运行 worktree 中执行；原输出目录已经存在，不要覆盖，复测用新的目录。

```bash
root=/data/pyptouser/qinchuanyu/pto-eager
csa_work="$root/.worktrees/hc-pre-layer-integration-20261012"
csa_case="$csa_work/tests/pypto_test/csa_static_spmd_feasibility_20261015"
csa_out="$csa_work/tests/pypto_test/results/csa_static_spmd_feasibility_20261015"
task-submit --device auto --max-time 1200 \
  "bash '$csa_case/run_static_jit_device.sh' '$csa_out/static_b16_local_signal_device_NEW' '$csa_case/static_local_signal/h131072_b16'"
```

脚本显式加载独立 PyPTO 与扩容 Simpler，校验 binding 所在路径及 ABI，
只允许由 task-submit 分配设备后执行。

<!-- SOURCE_END history-document-012 -->

---

<a id="history-document-013"></a>
### 历史原文 013：MEMORY_REUSE_PHASES_REPRO.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/MEMORY_REUSE_PHASES_REPRO.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/MEMORY_REUSE_PHASES_REPRO.md)；原文 7,935 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-013 -->
# 两个独立可编译 AIC 阶段融合后的片上内存分配复现

日期：2026-10-16。此问题独立于 while 的 #3009，也独立于 Simpler 的 32 Tensor
容量：下面的融合复现只需要 13 个 Tensor 参数，且全部是 CPU 编译。
未修改依赖仓库、未降低 tile 尺寸或 pipeline stage。

## 输入及对照结果

从当前 B4 静态原型直接抽取两个阶段：

- `indexer_score_topk_native_pair_aic`，原 AIC batch 13；独立入口 10 个 Tensor。
- `proj_a_mm`，原 AIC batch 15；独立入口 4 个 Tensor。
- 按原先后顺序放进一个 AIC 函数，两者共享 records 参数，合计 13 个 Tensor。

抽取脚本：[build_memory_probe.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/build_memory_probe.py)。保留原数学、分片读取、
FFTS 操作、pipeline 和本核完成边界；删除进度发布/等待及无关参数。
三个自包含 PyPTO 文件在 `static/memory_indexer/entry.py`、
`static/memory_oproj/entry.py`、`static/memory_pair_compact/entry.py`。
无需模型权重、Native 服务、真实 Tensor 数据或 NPU 即可编译。

| 编译输入 | PYPTO | DSA_RP |
| --- | --- | --- |
| Indexer 单段 | 代码生成及 PTOAS C++ 生成通过 | 代码生成及 PTOAS C++ 生成通过 |
| O-proj 单段 | 代码生成及 PTOAS C++ 生成通过 | canonical greedy 找不到容量内布局 |
| 两段融合，13 参数 | Right 81920 B > 65536 B | canonical greedy 找不到容量内布局 |

结果目录位于 `../results/csa_static_spmd_feasibility_20261015/`：

- `memory_indexer_PYPTO/result.json`
- `memory_oproj_PYPTO/result.json`
- `memory_pair_compact_PYPTO/result.json`
- `memory_pair_compact_DSA_RP_v2/result.json`
- `memory_indexer_DSA_RP_v1/result.json`
- `memory_oproj_DSA_RP_v1/result.json`

`memory_probe_PYPTO`、`memory_probe_DSA_RP` 是尚未裁掉无关根参数的早期复现，
也失败。`memory_pair_compact_DSA_RP` 是本地编译脚本的 PassContext 参数冲突，
**不算组件失败证据**；该调用方式修正后的结果以 `_v2` 为准。

## 分配审计

`static/memory_probe_summary.json` 汇总 MemoryReuse 后各物理复用槽的最大字节数。
这是 IR 中的分配量，不是运行时测得的瞬时峰值。

| 输入 | Left / KiB | Right / KiB | Acc / KiB | Mat / KiB |
| --- | ---: | ---: | ---: | ---: |
| Indexer 单段 | 60 | 64 | 100 | 112 |
| O-proj 单段 | 64 | 64 | 64 | 256 |
| 两段融合 | 80 | 80 | 100 | 256 |

两个原阶段各自可以放进目标资源，合并后 Left/Right 复用槽总量变成 80 KiB。
应检查不同阶段的 allocation 生命周期、复用分组、pipeline 硬约束与候选布局。
**单段通过和顺序关系提示存在可复用空间，但不是完整硬件可行性证明**；仍须
正确处理原流水和同步，不能直接删除编译器约束或把失败当作硬件必然不支持。

## 脱离 vLLM 的 CPU 编译

在组件维护者正常可用的 PyPTO/PTOAS 环境中，以 `entry.py` 为输入：

```python
from pathlib import Path
import pypto.language as pl
from pypto import ir, passes
from pypto.runtime import RunConfig

source = Path("static/memory_pair_compact/entry.py")
program = pl.parse(source.read_text(), filename=str(source.resolve()))
compiled = ir.compile(program, **RunConfig(
    platform="a2a3",
    memory_planner=passes.MemoryPlanner.PYPTO,  # 对照时改为 DSA_RP
    save_kernels=True,
    save_kernels_dir="/tmp/csa_memory_pair_repro",
    dump_passes=True,
).compile_kwargs())
```

将输入分别替换为两个单段入口即可得到对照。不要调用 compiled 对象或 `.load()`。
这两个段缺少其前序数据生产者和 MIX 的 AIV 半边，独立上卡不属于此复现的范围。

这里没有把两段小例子的 PASS 当成完整 CSA PASS；目标是让分配问题可直接接手。

## 三参数的顺序阶段分区复用小例子

[repro_phase_partition.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_phase_partition.py) 进一步去掉 NZ、pipeline、模型、
进度通信和多核。一个 AIC 函数的第一阶段同时使用 48 + 12 KiB Left，结束后
第二阶段同时使用 32 + 32 KiB Left；阶段之间有 `pl.system.fence()`。
每个 Tile 都参与 matmul 和输出写回，没有为构造生命周期添加虚假使用。

| planner | 实际结果 |
| --- | --- |
| PYPTO | Left 81920 B > 65536 B |
| DSA_RP | 完整代码生成通过；Left 地址为 a=0、b=49152、c=0、d=32768 |
| PTOAS | Left 655360 bit > 524288 bit，分配失败 |

证据分别为 `phase_partition_PYPTO_v1`、`phase_partition_DSA_RP_v1`、
`phase_partition_PTOAS_v1` 的 `result.json` 和 `kernel/passes/`。
DSA_RP 的实际 C++ 为 `phase_partition_DSA_RP_v1/kernel/ptoas/body.cpp`。

本地 `memory_reuse_pass.cpp` 的默认分组按 Tile 大小排序，复用槽按最大成员
固定容量。已有 subrange 尝试，但不能把一个 48 KiB 槽扩大成两段连续的
32 + 32 KiB；于是第二个 32 KiB Tile 另开槽，总容量变为 48 + 32 = 80 KiB。
`docs/en/dev/passes/36-memory_reuse.md` 也说明了该分组方式。
这证明小例子的失败不是 A3 物理容量不足；DSA_RP 已给出可放进 64 KiB 的布局。
**它不证明完整 CSA 已有合法布局**：真实 O-proj 单段在 DSA_RP 下仍失败，且
不同 planner 会影响 AutoTileMatmulL0 的选择，需要继续核对实际分配约束。

## 固定原分块后，真实两段的 DSA_RP 分配可以通过

[probe_fixed_tiles_dsa.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/probe_fixed_tiles_dsa.py) 读取 PYPTO 路径保存的
`36_after_MaterializeSemanticAliases.py`，只执行 `allocate_memory_addr`，
在其 PassContext 选择 DSA_RP。没有修改 IR、Tile 大小或流水阶段。
这是一项隔离分配因素的 CPU 诊断，**不是完整编译，也不是生产入口的解决方案**。

| 固定 Tile 的输入 | Left / KiB | Right / KiB | Acc / KiB | Mat / KiB | 分配 |
| --- | ---: | ---: | ---: | ---: | --- |
| O-proj | 64 | 64 | 64 | 256 | 通过 |
| Indexer + O-proj | 64 | 64 | 100 | 256 | 通过 |

真实代码的两段合并在保持原分块的条件下，已有满足容量的地址布局。
证据为 `oproj_fixed_tiles_dsa_v1/allocated.py`、`pair_fixed_tiles_dsa_v1/allocated.py`
及 `fixed_tiles_dsa_summary.json`；空间上界取实际 MemRef 的 offset + size。
复现命令：

```bash
python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/probe_fixed_tiles_dsa.py \
  --source tests/pypto_test/results/csa_static_spmd_feasibility_20261015/memory_pair_compact_PYPTO/kernel/passes_dump/36_after_MaterializeSemanticAliases.py \
  --output /tmp/csa_pair_fixed_tiles_dsa
```

直接选择 DSA_RP 的原始源码编译与上面的固定 Tile 诊断不同：
O-proj 在前者中生成两个同时存活的 128×256 BF16 Left Tile，每个 64 KiB；
`LowerPipelineLoops` 的 IR 顺序是先产生两组 Left/Right，再执行两个 matmul_acc，
所以需要的 Left 已超过 64 KiB。PYPTO 原路径则使用 128×128、每块 32 KiB。
不能要求分配器把同时存活的两块 64 KiB 强行放到一块 64 KiB 中。

建议 PyPTO 组件分开处理：

1. 默认 MemoryReuse 的固定槽大小导致阶段之间无法重新分区，3 参数例子可直接验收。
2. AutoTileMatmulL0 的策略随 memory planner 改变，应计入调用方已有 pipeline 的
   同时存活资源，或提供保留原分块策略、仅更换分配算法的正式配置。不要在应用侧
   临时篡改 PassContext 或删除硬约束绕过。
3. 先验收上面真实两段的源码完整编译，再验证完整 CSA；后者还有独立的区域 SSA
   代码生成问题。不能把本次“地址分配通过”扩写为“整层编译通过”。

对历史 `static_b4_cpu_v15` 的中间 IR 做相同诊断时，打印源码含已有的
`progress__ssa_v1__FREE_VAR`，在重新解析时失败。该记录为
`full_b4_fixed_tiles_dsa_v1`，不构成完整 CSA 的 DSA 分配成功或失败证据。

<!-- SOURCE_END history-document-013 -->

---

<a id="history-document-014"></a>
### 历史原文 014：PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md)；原文 4,706 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-014 -->
# 128K/B16：单 SPMD 与原动态 PTO 性能对比

用户于 2026-10-16 要求只比较单 SPMD 与原动态 PTO，不测 Native。

## 被测范围与口径

- history=131072、batch=16、S=6、T=96；正式 W8A8 权重，第 4 层。
- 原动态 PTO 使用 `results/hc_pre_layer_integration_20261012/baseline_ops` 冻结实现，
  与静态排程提取、上一轮精确比较同源。
- 单 SPMD 使用 `static_local_signal/h131072_b16/entry_jit.py` 与已验证的
  `local_signal_adapter.py`；notify 前包含本核 PIPE_ALL。
- 两边在同卡同进程、同一流串行执行，共用权重、输入和缓存地址，AB/BA 交替。
- 每次调用前恢复完全相同的缓存初态。恢复和输出毒化不进入计时区间；进度不清零。
- 先编译及 eager 比较，再捕获只含 PTO 调用的 ACLGraph；重放结果先对动态 eager
  精确比较。该图仅用于固定输入计时，不代表完整请求生命周期或连续 decode 验收。
- 设备 Event：每侧预热 5 次、采样 20 次。Event 在图外记录，不复用图内旧时间戳。
- L1：每侧独立 3 个窗口，一窗口只运行一侧一次图重放。
  主指标为原始 `aicore_tasks` 的 `max(end)-min(start)`，按记录中的设备时钟换算。
  不把收任务时间当成 incore 开始，不累加各核时长；保留内部等待及任务间隙。
- 两边使用同一独立 PyPTO/Simpler 调试安装、TMR runtime 和 L1 构建配置。
  Event 采样时 DFX 窗口关闭；L1 与 Event 分开报告，不混合样本。
- 不执行 Native attention forward，不使用此前 Native 性能作为本次对照。

## 首轮任务

- task ID：`task_20261016_183457_19106108116`
- 输出目录：`../results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1`
- 启动：`run_static_vs_dynamic_performance.sh <全新输出目录>`，必须通过 `task-submit --device auto`。
- 最终状态：completed (exit=0)，输出、状态、信号与保护区检查通过。

测试入口为 `benchmark_static_vs_dynamic.py`，未修改生产 CSA、数学计算体或依赖组件。

## 最终结果

主口径为 L1 incore 首尾跨度，单位 μs。动态 PTO 共 66 个计算 task、973 个物理核执行记录；
单 SPMD 为 1 个 MIX task、72 个物理核执行记录。两边均覆盖 24 AIC + 48 AIV。

| 采样 | 原动态 PTO | 单 SPMD |
| --- | ---: | ---: |
| L1 第 1 次 | 897.70 | 1519.30 |
| L1 第 2 次 | 900.14 | 1380.22 |
| L1 第 3 次 | 877.42 | 1350.06 |
| **L1 中位数** | **897.70** | **1380.22** |
| Event 中位数（各 20 次） | 942.62 | 1467.75 |

单 SPMD 的 L1 中位数增加 482.52 μs，耗时增加 **53.75%**，动态/静态比为 **0.6504×**。
设备 Event 中位数增加 525.13 μs，耗时增加 **55.71%**。当前静态化版本没有取得加速。

L1 动态范围 877.42～900.14 μs；静态范围 1350.06～1519.30 μs。
Event 动态范围 926.12～967.02 μs；静态范围 1442.78～1819.22 μs，所有样本保留，未剔除离群值。
L1 与 Event 是不同采样窗口，不能把两者中位数直接相减当作某个阶段的精确开销。

退步已经发生在 incore 首尾区间内，不能只归因于 Simpler/orchestrator 的外围首尾调度。
单 SPMD 的 L1 仅显示每核整段融合计算，没有内部阶段时间戳；本次结果尚不能分别量化
notify/wait、缓存维护、PIPE_ALL、静态排程或融合后编译结果各自的影响。未继续做优化。

## 正确性和范围

- 单 SPMD eager、两侧图捕获后重放、两侧最终 L1 重放的输出、Top-K、六份 cache/state
  均与动态 eager 参考逐元素完全一致。无 Native forward 比较。
- 单 SPMD 累计执行 30 个代际，进度区不清零；72 核末次进度和工作区保护通过。
- Native 格式缓存的未写区域、只读 metadata 保护检查通过。
- 这是固定输入、固定地址的计时图；每次恢复缓存，不代表连续 decode 轨迹、动态 B/S
  或完整整网性能。

## 数据与复现

- 原始结果：`../results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1/result.json`。
- 汇总：同目录 `summary.json`，含全部 20 次 Event 及 3 次 L1 数据。
- 泳道：同目录 `l1/dynamic_pto/step_{1,2,3}/swimlane.json` 和
  `l1/single_spmd/step_{1,2,3}/swimlane.json`，各目录同时保留原始记录与名称映射。
- `download/` 汇总六份重命名的泳道、报告与 JSON；可以直接用于 Perfetto/Chrome tracing。
- 归档副本：本目录 `evidence/static_vs_dynamic_perf_b16_v1/`。

正式权重、大张量、动态库及设备二进制不入归档；只记录本地路径，没有计算 hash。

<!-- SOURCE_END history-document-014 -->

---

<a id="history-document-015"></a>
### 历史原文 015：PR3013_REVIEW_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/PR3013_REVIEW_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PR3013_REVIEW_20261016.md)；原文 2,915 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-015 -->
# PyPTO PR #3013：独立拉取与复现核对

2026-10-16。用户指定 [PR #3013](https://github.com/hw-native-sys/pypto/pull/3013)，
对应此前本会话提交的 [issue #3009](https://github.com/hw-native-sys/pypto/issues/3009)。

## 拉取范围

独立 detached worktree：`.worktrees/pypto-pr3013-review-20261016`。
精确 head：`72b7ffb98ea3cee12bf7bbf533ab89b848a8980c`。
两个提交为 `1a61b888ed06475158c15154aa7cd9947de9d8b8` 和上述 head。
该源码未叠加本会话九项本地修复；共享调试分支、安装和 CSA 独立修复 worktree 未被切换。

PR 在 while 状态裁剪时把 condition 的 iter_arg 引用加入活跃性种子，
继续传播 yield 依赖，并检查裁剪后 condition/yield 没有悬空引用。
这与当前显式 Group 的统一参数 ABI 问题不同。

## 实际验证

| 验证 | 实际结果 |
| --- | --- |
| worktree 自行构建 pypto_core | 通过；没有复制其他 checkout 的 .so |
| PR 新增 condition-only scalar / tile / FIFO 状态回归及 codegen 四种组合 | 7 passed |
| 本会话原 scalar polling 复现 | 完成代码生成，ExpandMixedKernel 后无 ready FREE_VAR |
| 本会话原 tile polling 复现 | 完成代码生成，ExpandMixedKernel 后无 ready FREE_VAR |
| tile polling 保留结果的对照 | 完成代码生成，ExpandMixedKernel 后无 ready FREE_VAR |
| 新提交四类正确性问题的相同复现 | 5 failed，证实 #3013 不覆盖这些问题 |
| 上述五个用例在本地候选修复版 | 5 passed |

原复现只为新版本 RunConfig 把 platform 参数改为 backend_type，并显式
codegen_only=True；计算及 polling 核体不改。适配后文件为
`upstream_issues/pr3013_repro_mixed_while.py`。
全部是 CPU IR/代码生成检查，没有启动 NPU，不等于完整静态 CSA 功能通过。

结论：本会话 #3009 的原 scalar/tile 最小复现已被该 PR 修复。
在本次限定的源码审阅和定向回归范围内，没有发现新的阻塞问题；不替代上游 CI 或更广的验证。
PR 仍为 OPEN；本轮没有替维护者合并，也没有把它安装到共享环境。

## 环境失败必须区分

- 首次构建缺少 runtime 子模块头文件，补齐该 worktree 自身的 submodule 后构建成功。
- 首轮测试沿用旧 editable 安装的模块清单，找不到新版本 `pypto._target`，
  尚未运行测试。增加仅当前进程的 filesystem 导入模式，从目标 worktree 源码及
  自建扩展直接加载，第二轮验证有效。没有修改共享 site-packages。
- 这两项是测试环境准备问题，没有冒充 PyPTO 正确性 bug 上报。

日志为 `session_logs/csa-pr3013-build*.log`、
`session_logs/csa-pr3013-focused-tests-v2.log` 和三个 `csa-pr3013-original-*-v2.log`。
原复现结果目录为 `evidence/pr3013_original_*_v2/`；
独立回归的逐项摘要见 `upstream_issues/validation_summary.json`。

<!-- SOURCE_END history-document-015 -->

---

<a id="history-document-016"></a>
### 历史原文 016：PYPTO_GROUP_SHARED_INOUT_REPRO.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_GROUP_SHARED_INOUT_REPRO.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_GROUP_SHARED_INOUT_REPRO.md)；原文 4,499 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-016 -->
# JIT 单 MIX Group：共享 InOut 参数在 ABI 统一后变为 free variable

日期：2026-10-16。本例无需模型、设备、while、NZ 或大参数，只有 5 个 Tensor。
问题首次出现于 PyPTO `ExpandMixedKernel`，尚未进入 PTOAS；与 Simpler 容量无关。
本会话没有修改 PyPTO 源码。源码基线为本机调试分支 `cb2484471`，使用该目录已有
`build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so`，不是新构建干净上游。
共享目录存在其他会话的修改，本结论仅针对实际加载的本地构建。

## 对照

- [repro_jit_group_inline.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_jit_group_inline.py)：AIC 写 c，AIV 写 y，两个成员
  没有共同 InOut；完整 CPU 编译通过，生成一次 24 核组 MIX，sync_start 保留。
- [repro_jit_group_shared_inout.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_jit_group_shared_inout.py)：AIV 额外接收并
  原样返回 c；它仍只写 y，核间没有通过 c 交换计算数据。wrapper 正确接回 InOut
  返回值，前端检查通过，但 `ExpandMixedKernel` 之后产生 free variable。
- 全量 B4 同源 JIT 也受影响，错误为 `Tensor view not found for parameter:
  progress__ssa_v1`；结果目录 `static_b4_jit_cpu_v6_alias`。不把小例子通过当作全量通过。

关键结构：

```python
@pl.jit.inline
def group_wrapper(a, b, c, x, y):  # 完整带类型签名见可运行文件
    with pl.cluster(name_hint="csa_group_probe"):
        c = cube(a, b, c)
        c, y = vector(c, x, y)
    return c, y

@pl.jit
def entry(a, b, c, x, y):
    with pl.spmd(24, sync_start=True):
        c, y = group_wrapper(a, b, c, x, y)
    return c, y
```

直接传调用前的 c 会被 InOutUseDiscipline 正确拒绝。不能用旧变量、弱化 InOut
声明或新增无用 block_idx 读取绕过。这里也没有拆成两次 runtime 提交。

## CPU 复现

当前工作区命令；通用组件环境可去掉 bootstrap 与 `dsv4_csa_env.activate()`：

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python tests/pypto_test/hc_pre_layer_integration_20261012/bootstrap.py \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_jit_group_shared_inout.py \
  --output /tmp/csa_group_shared_inout
```

实际失败证据：`../results/csa_static_spmd_feasibility_20261015/jit_group_shared_inout_v1/`。
成功对照：`jit_group_inline_v1/`。Meta 注册和返回别名检查通过于
`jit_group_inline_meta_v1/`；它不等于设备执行。

## 第一处 IR 破坏

`kernel/passes_dump/24_after_LowerAutoVectorSplit.py` 中 AIV 成员参数和返回均为
`c__ssa_v0`，Group 的调用参数为前一个成员的返回别名 `c__ssa_v1`。

`25_after_ExpandMixedKernel.py` 统一 AIC/AIV 签名之后：

```python
def vector(a__ssa_v0, b__ssa_v0, c__ssa_v0, x__ssa_v0, y__ssa_v0):
    # 省略只写 y 的正常 vector 计算
    return c__ssa_v1__FREE_VAR, y_v1__ssa_v0
```

`c__ssa_v1` 既不是新函数参数，也不是其局部定义。后续 codegen 报不能物化该符号，
错误提示建议传入 Scalar，但此处实际上是 Tensor 别名，不适用这个建议。

## 建议组件修复位置与验收

本地 `src/ir/transforms/expand_mixed_kernel_pass.cpp`：

- `BuildGroupAbiAdapter`（约 2521 行）构造 Group 参数到 fresh 参数映射。
- 约 2543 行直接把 `member.inner_call.args[i]` 替换到该映射后 DeepClone。
- 当前映射只有 Group 参数，没有把成员调用返回的 Tensor 别名追溯到 Group 参数；
  所以 `c__ssa_v1` 留在新函数体内，而调用本身已被改成统一参数表。

修复应先解析 Group 内部 SSA 返回别名，沿被调函数的 return-to-param 关系追溯，
再建立 member-to-adapter 映射。可参考现有 `return_lineage_utils.h` 的接口；
不能按变量名猜测身份，也不能把所有未知值都替换成第一个参数。
无法证明为边界别名的参数应明确拒绝，保留原语义约束。

最小验收：成功对照和失败例都生成一次 MIX，24 组、双 AIV、sync_start；
pass25 无 free variable，输出别名正确；补充重排参数、tuple 返回、多级别名与
无法追溯参数的负例。之后再验证真实 132/133 参数 CSA，不预记全量 PASS。

与 #3009 的关系：同处 `ExpandMixedKernel`，但本例是 Group ABI 统一中的
Tensor 别名映射；#3009 是 while 条件的 loop-carried scalar liveness。
修复其中一个不能代替另一个的验收。

<!-- SOURCE_END history-document-016 -->

---

<a id="history-document-017"></a>
### 历史原文 017：PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md)；原文 21,059 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-017 -->
### Pass Name

`ExpandMixedKernel`，具体定位到内部的 `FinalizeSplitCoreBody → StripDeadIterArgs`。

### Description

MIX SPMD 中的 AIV 轮询循环，如果携带值 `ready` **只在 while 条件中使用**，而循环结果 `final_ready` 在循环后没有使用，`ExpandMixedKernel` 会把这个仍然控制循环终止的携带值当成死状态删除。

删除后，初值读取、循环回边读取及 yield 一并消失，但条件仍然引用 `ready`，生成 `ready__FREE_VAR`。随后 PyPTO PTO codegen 报错，尚未进入设备运行。

典型的合法语义是：

```python
initial_ready = pl.read(flags, [0])
for (ready,) in pl.while_(init_values=(initial_ready,)):
    pl.cond(ready != 1)
    pl.system.cacheinvalid()
    final_ready = pl.yield_(pl.read(flags, [0]))
# 后续计算只需要等到循环结束，不需要再次使用 final_ready。
```

`pl.cond(...)` 在解析后存入 `WhileStmt.condition_`，不是循环 body 中的一条普通语句。条件中的使用仍然必须让该 iter_arg 保持活跃。当前错误不能通过给函数额外传一个名为 `ready` 的标量来修正，那会改变循环语义。

发现背景是将现有 CSA 动态排程固化为单个 MIX SPMD，需要表达核组之间的局部通知与等待。复现已剥离 CSA/vLLM，不需要模型权重、实际输入 Tensor 或 NPU 执行。

### Git Commit ID

`cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`

- 本地分支：`feat/kernel-mode-integration-test`。
- Python：`3.10.19`；Host：Linux aarch64；编译目标：`a2a3`。
- 加载本 checkout 的 `build/python/bindings/pypto_core.cpython-310-aarch64-linux-gnu.so`。
- 工作区存在其他调试改动；本报告没有修改 PyPTO。`expand_mixed_kernel_pass.cpp` 与 `utils/loop_state_repair.cpp` 没有本地修改。
- 这是当前实际使用的本地调试构建上的复现，不冒充干净 main 的重新构建结果。只读查看 main 时，同一活跃性判断也未纳入 while 条件；main 的独立编译复测尚未做。

### Before IR (Input)

下面是 tile 读取版 `24_after_LowerAutoVectorSplit.py` 中受影响的 InCore 函数，保留 `@pl.program` 包装。完整复现脚本会生成包含 Spmd/Orchestration 的全部 pass dump。

```python
import pypto.language as pl


@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
    def mixed_while_repro_mixed_while_repro(
        a__ssa_v0: pl.Tensor[[384, 16], pl.FP16],
        b__ssa_v0: pl.Tensor[[16, 16], pl.FP16],
        cube_out__ssa_v0: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
        scratch__ssa_v0: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        flags__ssa_v0: pl.InOut[pl.Tensor[[16], pl.INT32]],
        vector_out__ssa_v0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> tuple[pl.Tensor[[384, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True})
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        ac__ssa_v0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tile.load(a__ssa_v0, [worker__ssa_v0 * 16, 0], [16, 16], [16, 16], target_memory=pl.Mem.Mat)
        bc__ssa_v0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tile.load(b__ssa_v0, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Mat)
        ac__ssa_v0_Left: pl.Tile[[16, 16], pl.FP16, pl.Mem.Left] = pl.tile.move(ac__ssa_v0, target_memory=pl.Mem.Left)
        bc__ssa_v0_Right: pl.Tile[[16, 16], pl.FP16, pl.Mem.Right] = pl.tile.move(bc__ssa_v0, target_memory=pl.Mem.Right)
        cc__ssa_v0: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(ac__ssa_v0_Left, bc__ssa_v0_Right)
        cube_out__store: pl.Tensor[[384, 16], pl.FP32] = pl.tile.store(cc__ssa_v0, [worker__ssa_v0 * 16, 0], cube_out__ssa_v0)
        for lane__ssa_v0 in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if lane__ssa_v0 == 0:
                if worker__ssa_v0 == 0:
                    av__ssa_v0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Vec] = pl.tile.load(a__ssa_v0, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec)
                    t__tmp_v0: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.cast(av__ssa_v0, target_type=pl.FP32, mode="round")
                    t__tmp_v1: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v0, 1.0)
                    scratch__store: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(t__tmp_v1, [0, 0], scratch__ssa_v0)
                    pl.system.cacheinvalid()
                    pl.system.fence()
                    pl.tensor.write(flags__ssa_v0, [0], pl.cast(1, pl.INT32))
                    pl.system.cacheinvalid()
                    pl.system.fence()
                else:
                    if worker__ssa_v0 == 1:
                        initial_tile__ssa_v0: pl.Tile[[16], pl.INT32, pl.Mem.Vec] = pl.tile.load(flags__ssa_v0, [0], [16], [16], target_memory=pl.Mem.Vec)
                        initial_ready__ssa_v0: pl.Scalar[pl.INT32] = pl.tile.read(initial_tile__ssa_v0, [0])
                        for (ready,) in pl.while_(init_values=(initial_ready__ssa_v0,)):
                            pl.cond(pl.cast(ready, pl.INDEX) != 1)
                            pl.system.cacheinvalid()
                            flag_tile__ssa_v0: pl.Tile[[16], pl.INT32, pl.Mem.Vec] = pl.tile.load(flags__ssa_v0, [0], [16], [16], target_memory=pl.Mem.Vec)
                            t__tmp_v2: pl.Scalar[pl.INT32] = pl.tile.read(flag_tile__ssa_v0, [0])
                            final_ready: pl.Scalar[pl.INT32] = pl.yield_(t__tmp_v2)
                        pl.system.cacheinvalid()
                        received__ssa_v0: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(scratch__ssa_v0, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec)
                        t__tmp_v3: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(received__ssa_v0, 2.0)
                        vector_out__store: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(t__tmp_v3, [0, 0], vector_out__ssa_v0)
        return cube_out__ssa_v0, scratch__ssa_v0, vector_out__ssa_v0
```

### Expected IR (After Transformation)

展开后应在 AIV 函数中保留 while 的初值、条件、回边和携带变量。下面仅列受影响的 AIV 函数；AIC、Group、Spmd 和 Orchestration 应照常生成。

这是根据成功对照用例整理的预期 IR，已去掉为定位问题而添加的循环后 `flags[1] = final_ready`。它是预期结果说明，不是已修复编译器的实测输出。

```python
import pypto.language as pl


@pl.program
class Expected:
    @pl.function(type=pl.FunctionType.AIV, level=pl.Level.AIV, role=pl.Role.SubWorker)
    def mixed_while_repro_mixed_while_repro_aiv(
        a__ssa_v0: pl.Tensor[[384, 16], pl.FP16],
        b__ssa_v0: pl.Tensor[[16, 16], pl.FP16],
        cube_out__ssa_v0: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
        scratch__ssa_v0: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        flags__ssa_v0: pl.InOut[pl.Tensor[[16], pl.INT32]],
        vector_out__ssa_v0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> tuple[pl.Tensor[[384, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
        pl.func_attr({"mx_tensor_views_blocked": True, "split_aiv": True, "dual_aiv_dispatch": True})
        worker__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_block_idx()
        lane__ssa_v0: pl.Scalar[pl.INDEX] = pl.tile.get_subblock_idx()
        if lane__ssa_v0 == 0:
            if worker__ssa_v0 == 0:
                av__ssa_v0: pl.Tile[[16, 16], pl.FP16, pl.Mem.Vec] = pl.tile.load(a__ssa_v0, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec)
                t__tmp_v0: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.cast(av__ssa_v0, target_type=pl.FP32, mode="round")
                t__tmp_v1: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__tmp_v0, 1.0)
                scratch__store: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(t__tmp_v1, [0, 0], scratch__ssa_v0)
                pl.system.cacheinvalid()
                pl.system.fence()
                pl.tensor.write(flags__ssa_v0, [0], pl.cast(1, pl.INT32))
                pl.system.cacheinvalid()
                pl.system.fence()
            else:
                if worker__ssa_v0 == 1:
                    initial_tile__ssa_v0: pl.Tile[[16], pl.INT32, pl.Mem.Vec] = pl.tile.load(flags__ssa_v0, [0], [16], [16], target_memory=pl.Mem.Vec)
                    initial_ready__ssa_v0: pl.Scalar[pl.INT32] = pl.tile.read(initial_tile__ssa_v0, [0])
                    for (ready,) in pl.while_(init_values=(initial_ready__ssa_v0,)):
                        pl.cond(pl.cast(ready, pl.INDEX) != 1)
                        pl.system.cacheinvalid()
                        flag_tile__ssa_v0: pl.Tile[[16], pl.INT32, pl.Mem.Vec] = pl.tile.load(flags__ssa_v0, [0], [16], [16], target_memory=pl.Mem.Vec)
                        t__tmp_v2: pl.Scalar[pl.INT32] = pl.tile.read(flag_tile__ssa_v0, [0])
                        final_ready: pl.Scalar[pl.INT32] = pl.yield_(t__tmp_v2)
                    pl.system.cacheinvalid()
                    received__ssa_v0: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(scratch__ssa_v0, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec)
                    t__tmp_v3: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.muls(received__ssa_v0, 2.0)
                    vector_out__store: pl.Tensor[[16, 16], pl.FP32] = pl.tile.store(t__tmp_v3, [0, 0], vector_out__ssa_v0)
        return cube_out__ssa_v0, scratch__ssa_v0, vector_out__ssa_v0
```

### Actual IR or Error

同一函数经过 `25_after_ExpandMixedKernel.py` 后，AIV 消费者分支变为：

```python
if worker__ssa_v0 == 1:
    while pl.cast(ready__FREE_VAR, pl.INDEX) != 1:
        pl.system.cacheinvalid()
    pl.system.cacheinvalid()
    # 后续 scratch 的读取和输出计算仍在。
```

`ready` 的定义、init_values、循环内 flags 读取及 yield 已经消失。最终异常的关键部分：

```text
PartialCodegenError: 1 function(s) failed to compile:
mixed_while_repro_mixed_while_repro
PTO codegen cannot materialize symbol 'ready' in function
'mixed_while_repro_mixed_while_repro_aiv': it is not a
physical tensor dimension, a scalar parameter, or a loop
variable of this kernel.
```

后续错误文本关于 `TensorView(valid_shape=...)` 的通用建议与本例的触发原因不符：本例 Tensor shape 全部固定，`ready` 是在上一 pass 中仍然有效的循环携带变量。

### NPU Kind (if relevant)

N/A (not hardware-specific)。本次只在 CPU 上运行编译流程，目标设置为 `a2a3`，没有分配 NPU Tensor 或执行设备算子。

### Host Platform (if relevant)

Linux (aarch64)

### Additional Context

**复现命令与实测结果**

在已安装/构建 PyPTO、已配置其匹配 PTOAS 的开发环境中，将下方完整脚本保存为 `repro_mixed_while_loop_state.py`。各条命令独立执行；当前前两条预期退出 1，第三条退出 0。

```bash
python repro_mixed_while_loop_state.py --variant scalar --output /tmp/pypto-while-scalar
python repro_mixed_while_loop_state.py --variant tile --output /tmp/pypto-while-tile
python repro_mixed_while_loop_state.py --variant tile --keep-result --output /tmp/pypto-while-tile-keep
```

| 2026-10-15 实测 | CPU 代码生成 | ExpandMixedKernel 后 |
| --- | --- | --- |
| GM 标量读取，循环后不用 final_ready | 失败，退出 1 | 出现 ready__FREE_VAR |
| Vec tile 读取，循环后不用 final_ready | 失败，退出 1 | 出现 ready__FREE_VAR |
| 相同 Vec tile 读取，只增加循环后写出 final_ready | 完成，退出 0 | 初值、条件和回边读取保留 |

第三个对照只增加 `pl.write(flags, [1], final_ready)`，用于让循环结果在循环后可见。它不是生产实现的修复方案，也不能代替对原始合法写法的支持。

已检查第三个对照的 PTOAS C++ 输出：条件分支及回边仍在，回边中保留 `TLOAD` 和 `GetValue` 更新下一次判断值。这里的“完成”只指 PyPTO/PTOAS 代码生成；没有编译设备二进制或验证跨核通知协议的真机正确性。

每次输出目录包含 `result.json`、失败时的 `exception.txt`，以及 `kernel/passes_dump/`。搜索 pass 名称即可定位边界，其他版本的编号可能变化。`result.json` 记录实际加载的 Python 模块与扩展路径。

<details>
<summary>完整独立复现脚本：repro_mixed_while_loop_state.py</summary>

```python
# SPDX-License-Identifier: Apache-2.0
"""CPU 复现 MIX 展开丢失 while 条件携带值；不分配或运行 NPU Tensor。"""

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path

import pypto
import pypto.language as pl
from pypto.runtime import RunConfig


@pl.jit
def mixed_while_repro(
    a: pl.Tensor[[384, 16], pl.FP16],
    b: pl.Tensor[[16, 16], pl.FP16],
    scratch: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
    flags: pl.InOut[pl.Tensor[[16], pl.INT32]],
    cube_out: pl.Out[pl.Tensor[[384, 16], pl.FP32]],
    vector_out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    TILE_POLLING: pl.constexpr = False,
    KEEP_RESULT: pl.constexpr = False,
):
    # 矩阵运算只用于使这一段成为 MIX 核；flags 协议只作为编译输入。
    for worker in pl.spmd(24, sync_start=True, name_hint="mixed_while_repro"):
        ac = pl.load(a, [worker * 16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        bc = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
        cc = pl.matmul(ac, bc, out_dtype=pl.FP32)
        pl.store(cc, [worker * 16, 0], cube_out)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            if lane == 0:
                if worker == 0:
                    av = pl.load(a, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                    pl.store(pl.add(pl.cast(av, pl.FP32), 1.0), [0, 0], scratch)
                    pl.system.cacheinvalid()
                    pl.system.fence()
                    pl.write(flags, [0], pl.cast(1, pl.INT32))
                    pl.system.cacheinvalid()
                    pl.system.fence()
                elif worker == 1:
                    if TILE_POLLING:
                        initial_tile = pl.load(flags, [0], [16], target_memory=pl.MemorySpace.Vec)
                        initial_ready = pl.tile.read(initial_tile, [0])
                        for (ready,) in pl.while_(init_values=(initial_ready,)):
                            pl.cond(ready != 1)
                            pl.system.cacheinvalid()
                            flag_tile = pl.load(flags, [0], [16], target_memory=pl.MemorySpace.Vec)
                            final_ready = pl.yield_(pl.tile.read(flag_tile, [0]))
                        if KEEP_RESULT:
                            # 仅供定位活跃性分析的对照；不作为生产代码绕过方案。
                            pl.write(flags, [1], final_ready)
                    else:
                        initial_ready = pl.read(flags, [0])
                        for (ready,) in pl.while_(init_values=(initial_ready,)):
                            pl.cond(ready != 1)
                            pl.system.cacheinvalid()
                            final_ready = pl.yield_(pl.read(flags, [0]))
                        if KEEP_RESULT:
                            pl.write(flags, [1], final_ready)
                    pl.system.cacheinvalid()
                    received = pl.load(scratch, [0, 0], [16, 16], target_memory=pl.MemorySpace.Vec)
                    pl.store(pl.mul(received, 2.0), [0, 0], vector_out)
    return cube_out, vector_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("scalar", "tile"), default="scalar")
    parser.add_argument("--keep-result", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "variant": args.variant,
        "keep_result": args.keep_result,
        "python": sys.version,
        "host_architecture": platform.machine(),
        "pypto_module": pypto.__file__,
        "pypto_core_module": getattr(sys.modules.get("pypto.pypto_core"), "__file__", None),
        "platform": "a2a3",
        "device_execution": False,
    }
    try:
        compiled = mixed_while_repro.compile(
            TILE_POLLING=args.variant == "tile",
            KEEP_RESULT=args.keep_result,
            config=RunConfig(
                platform="a2a3",
                save_kernels=True,
                save_kernels_dir=str(output / "kernel"),
                dump_passes=True,
            ),
        )
        result["code_generation_completed"] = True
        result["output_dir"] = str(compiled.output_dir)
    except Exception as exc:
        result["code_generation_completed"] = False
        result["error"] = str(exc)
        (output / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")

    # 记录 pass 边界，便于直接将生成文件与编译错误关联。
    boundary_files = sorted((output / "kernel/passes_dump").glob("*ExpandMixedKernel.py"))
    result["expand_mixed_kernel_dumps"] = [str(path.relative_to(output)) for path in boundary_files]
    result["free_ready_after_expand"] = any(
        "ready__FREE_VAR" in path.read_text(encoding="utf-8") for path in boundary_files
    )
    result["pypto_core_module"] = getattr(sys.modules.get("pypto.pypto_core"), "__file__", None)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "variant", "keep_result", "code_generation_completed", "free_ready_after_expand"
    )}, ensure_ascii=False))
    return 0 if result["code_generation_completed"] and not result["free_ready_after_expand"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

</details>

**源码定位与因果链**

以下是根据实际前后 IR、对照编译和源码检查得到的定位；没有在依赖仓库中打补丁验证修复。

1. `ExpandMixedFunction` 对生成的 AIV body 调用 `FinalizeSplitCoreBody`。
2. `FinalizeSplitCoreBody` 调用 `StripDeadIterArgs`；该函数同时处理 ForStmt 和 WhileStmt。
3. `StripDeadIterArgs` 用循环 body 中的引用和循环结果在后续语句中的使用来初始化 `live`，没有把当前 `WhileStmt.condition_` 的引用加入活跃性种子。
4. 本例的 `ready` 只用于条件；每次 yield 产生的是新读取的 flag，不依赖旧 ready；final_ready 又没有循环后用途。因此这一 slot 被误判为 dead，yield 闭包传播也无法把它救回来。
5. slot 和 yield 被删掉，随后 DCE 删除失去用途的读取；条件保留却指向已移除的 iter_arg，形成 FREE_VAR。

对应源码位置（以本地复现版本为准）：

- `src/ir/transforms/expand_mixed_kernel_pass.cpp`：`ExpandMixedFunction`，AIV 的 `FinalizeSplitCoreBody` 调用，约 2081 行。
- `src/ir/transforms/utils/loop_state_repair.cpp`：`StripDeadIterArgs`，约 351 行；活跃性种子约 399–413 行；清除 slot/yield 约 415–440 行。
- 同文件 `FinalizeSplitCoreBody`，约 830 行。
- [main 的 StripDeadIterArgs 所在文件](https://github.com/hw-native-sys/pypto/blob/main/src/ir/transforms/utils/loop_state_repair.cpp)：只读检查可见相同的 body/后续使用判断；行号随版本变化。
- [官方 while/SSA 写法](https://github.com/hw-native-sys/pypto/blob/main/docs/en/user/language/02-control-flow.md)：本例使用显式 init_values、pl.cond 和赋值形式的 pl.yield_。

因此问题应先由 **PyPTO IR pass/循环状态活跃性分析**处理。当前证据没有指向 Simpler 调度、PTOAS 指令选择或 PTO-ISA 硬件实现；坏 IR 已在这些环节之前出现。

**建议修复范围与验收**

- 为 WhileStmt 将 condition 中使用的 iter_args 纳入 live 种子，再做现有 yield 依赖闭包；不要通过虚构标量入口参数修补 FREE_VAR。
- 复核删除后的校验是否覆盖 while condition，而不只检查幸存的 yield。
- 补一个仅在 while condition 使用 iter_arg、yield 为独立新值、循环结果未使用的 pass 单测；覆盖当前嵌套 worker/lane 分支及嵌套条件中的 while。
- 两个原始复现均能完整代码生成，生成 IR 无悬空变量，循环体中的重新读取与条件回边仍在。保留第三个对照用于比较，不应要求业务额外写出 final_ready。
- 对照已有多 slot FIFO 的活跃性闭包用例，避免修复条件使用时破坏已有 yield 依赖传播。

Related: #2716。它涉及多个 FIFO slot 之间通过 yield 形成的活跃性传播；本例是单个状态仅被 while condition 使用，属于不同触发条件。此前 issue 的相关修正已存在于当前源码中，本例仍然失败。

<!-- SOURCE_END history-document-017 -->

---

<a id="history-document-018"></a>
### 历史原文 018：PYPTO_STATIC_FIXES_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md)；原文 13,051 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-018 -->
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
[二进制编译记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md)。两项结果都不是完整 CSA 真机通过。

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
见 [直调入口核对](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/DIRECT_NATIVE_ENTRY_20261016.md)。

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

<!-- SOURCE_END history-document-018 -->

---

<a id="history-document-019"></a>
### 历史原文 019：SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md)；原文 13,024 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-019 -->
# CSA 单 SPMD：Scalar 非一致缓存协议复核

## 当前结论和范围

2026-10-16 用户指出：A3/A5 普通 Scalar cache 的非一致性问题相同，不能因平台不同而
忽略；两代原子指令接口则需区分。停止推进 volatile 接口或提交完整设备续测，先复核协议。所有静态源码、
失败与诊断通过记录保留。本次只读代码并整理记录，没有新增组件修复或设备任务。

此前“只读改为 volatile 就通过”的结论仅描述诊断结果，不能证明缓存一致性、payload
发布、跨核交接或完整 CSA 正确。现阶段不能将问题定性为 PyPTO/PTOAS 缺少 volatile
接口就能解决的 bug。首先要证明本仓生成的跨核协议成立。

## 已读参考与适用边界

已查看指定分支同目录三个 usage guide 的结论、接口语义、验证边界及相关采集方法。
参考快照和来源见 `references/simpler_atomic_guides_20261016/source.json`，对应
nalinaly/simpler `fdwic-swimlane-deps` 的 Git 版本
`3699b7ae4ff47abcfa1e2916881d4309cf39c00f`。不计算文件内容 hash。

- [ATOMIC_USAGE_GUIDE.md](https://github.com/nalinaly/simpler/blob/3699b7ae4ff47abcfa1e2916881d4309cf39c00f/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md)：
  普通 Scalar 缓存访问没有跨核 coherence；DCCI 可能 clean/writeback，并非按地址纯
  invalidate；DSB 约束当前核的完成顺序；控制交接与 payload 可见性必须分别建立协议。
  其 A5 atomic 方案及特定路径可省 DSB 的实测不能照搬到 A3。
- [cache_preload_usage_guide.md](https://github.com/nalinaly/simpler/blob/3699b7ae4ff47abcfa1e2916881d4309cf39c00f/tests/atomic_probe/cache_preload_usage_guide.md)：
  preload 是性能 hint，不能替代写回、同步和交接；普通 store 被接受与完成 GM 发布
  是不同边界。current-PC ICache 预取还受实际代码地址与独立工作窗口约束。
- [icache_miss_usage_guide.md](https://github.com/nalinaly/simpler/blob/3699b7ae4ff47abcfa1e2916881d4309cf39c00f/tests/atomic_probe/icache_miss_usage_guide.md)：
  诊断代码即使运行时不落记录，仍可能改变代码布局、到达时序和竞争。PMU 与泳道采用
  各自构建和统计边界，不能逐 tick 相减；A5 selector/频率不能直接作为 A3 常量。

## 当前源码的实际协议

代码位置以 `generate_static_entry.py` 为准，下面行号为本轮核对时的位置。

| 环节 | 当前实现 | 尚缺的论证 |
| --- | --- | --- |
| 进度布局 | 371～379：每核 64 个 INT64，即 512 B；单核写自己的进度，多核读取 | 间隔不等于 coherence。仍须检查实际基址对齐、整个 line 的访问者及跨调用生命周期。 |
| 发布进度 | 430～436：本核 fence、整缓存 DCCI、fence、普通 Scalar 写进度、单行 DCCI、fence | 普通写与 DCCI 不能仅因诊断通过就被当成有明确契约的跨核通知。payload 与控制交接分别证明。 |
| 轮询进度 | 419～424：单行 DCCI 后普通 Scalar 读取；循环中重复 | 原版未在这些 DCCI 后显式等待完成。volatile 仅影响编译器访存，既不绕过 DCache，也不建立跨核同步。 |
| 消费 payload | 426～427：先 fence，再整缓存 DCCI，随后进入数学体 | fence 在 DCCI 之前，不能据此证明随后的失效/写回已在 payload 访问前完成。 |
| 全缓存维护 | 每个阶段进出均无参 cacheinvalid | 实际生成 C++ 为 ENTIRE_DATA_CACHE，可能涉及当前阶段之外的缓存 entry，不能按单行失效理解。 |

完整 B16 源码中的无参 cacheinvalid 静态调用点为 AIC 80 处、AIV 88 处，共 168 处；
这不是动态执行次数。该策略由本仓生成器加入，没有逐条 cacheline 所有权证明。
因此首先审计生成器协议，不应先把问题归咎于编译器。

## 初步布局核对（不是完整正确性证明）

- records/waits 是本轮执行期间只读表，和不断变化的 progress 不应按同类对象处理。
- progress 的 arena offset 为 0、每 worker 间隔 512 B；当前适配器按 512 B 规划
  offset/guard，但没有显式检查实际 allocation 基址的 cacheline 对齐。
- B16 数学体共有 4 处 Scalar GM 写入语句，落在 temporary_14/15/54。
  temporary_14/15 分别为 64 B 的两份 compact row offsets，由同一个原
  `csa_row_offsets` worker 写入；不能误报为多核争写同一行。
- temporary_54 为 [96,16] INT32，行距 64 B；原 valid-QK-plan 分片按 8 行划分。
  已核对行布局，但完整消费者、重复调用及 Scalar/MTE 可见性还没有闭合证明。
- 当前 Simpler toolchain 显式关闭 `cce-aicore-dcci-insert-for-scalar`，未显式关闭
  `cce-aicore-dcci-before-kernel-end`。后者的实际编译产物仍需核对，不能仅按源码
  推断所有 DCCI。没有在本轮改动这些编译选项。

## 保留的对照结果及其限度

所有以下任务均已终止，当前没有本会话排队或运行的设备任务。

| 同一 B16 控制表诊断 | 任务 | 观察 |
| --- | --- | --- |
| 原标量访问 | task_20261016_162004_206538014143 | 首轮 72 核有限轮询失败；设备返回。 |
| progress 读写都加 volatile | task_20261016_162414_222408211822 | 连续 3 个代际的进度值符合预期。 |
| 仅消费者加 fence | task_20261016_163312_271626813951 | 首轮 72 核有限轮询失败；设备返回。 |
| 仅 progress 读加 volatile | task_20261016_164435_356209620998 | 连续 3 个代际的进度值符合预期。 |
| 仅 progress 写加 volatile | task_20261016_164436_356260019239 | 首轮 72 核有限轮询失败；设备返回。 |

这些用例移除了数学体，只保留 3 个控制 Tensor，使用 program 显式 AIC/AIV；
没有覆盖真实 payload 的缓存状态和共享访问。三轮完成不能升级为完整协议或 CSA
通过，读写差分也不能单独证明某个 CCE 优化就是根因。

在用户纠正之前，另外只做过 CPU lowering 可表达性探针：带私有函数声明的
func.call 可以生成 C++；直接 emitc.call_opaque 携带 !pto.ptr 被当前 PTOAS 拒绝。
它们没有上卡，也没有形成 PyPTO/Simpler/PTOAS 源码补丁，现停止沿此方向直接修接口。

## 后续讨论的前提

保留原逐前驱依赖关系，优先审查单写者进度发布与绕过 Scalar DCache 的读取方式，
逐类核对 payload 写回、消费者缓存刷新、cacheline 归属及跨调用代际。单写者可以
消除并发更新冲突，不因有多个读者就必须增加 atomic；但发布和读取仍需完整成立条件。
不能用“512 B 间隔”“加 volatile”或几轮通过代替该论证，也不能无依据把原依赖图
改成阶段间全卡屏障。

本次只完成问题复核并保存证据；同步实现方向待与用户讨论，尚未恢复修改或测试。

## 用户补充分析：A3 软 barrier 与单写者进度发布的区别

已只读核对当前本地 `pto-isa-src/include/pto/common/syncall_soft.hpp` 第 35～65 行、
96～116 行，以及 `pypto/python/pypto/language/op/system_ops.py` 第 154～202 行。
这是本地源码核对，不等于独立编译路径已经采用该实现，更不等于本轮进行了设备验证。

A3 的共享计数器到达流程为：

```cpp
set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
dcci(counter, SINGLE_CACHE_LINE);
st_atomic<int32_t>(1, counter);
dcci(counter, SINGLE_CACHE_LINE);
dsb(DSB_DDR);
```

职责必须分开：

- `st_atomic` 保证多个参与核累加同一计数器时不丢更新。A3 源码实际存在这条路径，
  因此“A3 没有 atomic”的笼统表述不准确；不能与 A5 的原子接口混为一谈。
- DCCI 清理并失效对应 cacheline；该源码明确把写后的 DCCI 作为原子更新发布点。
  DSB 保证相关本核内存操作完成与顺序，不能单独建立跨核 cache coherence。
- 轮询使用 `ld_dev`，绕过本核 Scalar DCache；不等于每次都直接访问 HBM，
  也不代替计算流水完成或业务数据的跨核交接。

`syncall(SOFT)` 是参与集合的 barrier。每个参与者本轮到达后等计数达到本轮阈值；
CSA 原型则只等依赖表中的必要前驱。不能为复用前者，把后者的所有阶段强制改成
72 核集合。源码中的局部 `before = load; atomic_add(1)` 也不是返回旧值的单条 RMW；
此处只描述已有 barrier 算法，不把它当作通用 fetch-add。

对于当前每 worker 自己写的进度：

1. 确保整条进度 cacheline 只有该 worker 写入；其他核不能写同一行的邻接字，
   也不能留下会覆盖新值的旧 dirty line。静态模板还需保证串行复用和正确代际。
2. 先完成生产者计算与数据写回，再普通写进度，按所需顺序 DCCI + DSB 发布进度。
   这里没有多写者更新竞争，不需要仅为“其他核要读”引入 atomic RMW。
3. 消费者可采用相应宽度的 bypass load 轮询进度；若仍用缓存读取，必须给出
   相应的缓存维护与编译器读取保证。`volatile` 不等同于 bypass load。
4. 观察到进度后，再按业务数据的实际访问路径完成消费者缓存处理与顺序约束。
   标记到了与业务数据可读是两个必须连接起来的条件。

当前进度为 INT64，而上述 barrier 计数器为 INT32；未来实现不能只复制一行
`ld_dev` 就忽略访问宽度、对齐和非撕裂读取要求。本轮未修改类型或同步实现。

ATOMIC_USAGE_GUIDE 中“控制变量使用 atomic/硬件原语”是其 A5 工程保守规则；
它提醒审查普通 Scalar/DCCI 混用的风险，不能据此推导“A3 单写者协议必然需要
atomic”。用户此次修正已采纳：原子更新、缓存发布、读取路径和 payload 同步分别论证。

## 单个 incore task 内的点对点通知接口建议

用户确认需求：将原多 task 的依赖保留在一个融合 incore task 内，PyPTO 应提供
生产者通知、必要消费者等待的接口。这一抽象方向合理，不能用全体 barrier 替代。
当前本地源码并非只有 syncall，需区分现有接口的适用范围：

| 现有接口 | 本地定义与范围 | 与本次需求的关系 |
| --- | --- | --- |
| pl.system.syncall | language/op/system_ops.py：参与核集合 barrier | 等全部指定参与者到达，不能等价替换逐前驱依赖。 |
| pl.system.sync_set / sync_wait | 同文件：Cube/Vector 硬件事件，event_id、pipe、ffts_mode、core_type | 已有细粒度核间事件；接口没有普通 GM 信号 Tensor 与动态进度阈值，不能直接视为本次任意 worker 进度接口。 |
| pld.system.notify / wait | language/distributed/op/system_ops.py:124、160：DistributedTensor、peer rank、信号 offset | 已有跨 rank 信号抽象并 lowering 到 TNOTIFY/TWAIT；当前入参是通信窗口上的 DistributedTensor。 |

因此需求可准确表述为：为普通 GM 信号提供单个 SPMD 内的核间 notify/wait，或扩展
现有信号接口以支持该场景。底层优先核对复用已有 PTO-ISA 能力，不预设一定重写实现。
用户随后明确授权提交正式功能需求，已创建
[PyPTO #3020](https://github.com/hw-native-sys/pypto/issues/3020)。
尚未实现 API 或修改组件；正文及回读记录在
`upstream_issues/incore_p2p_notify_wait_20261016/`。

建议明确以下合同：

1. 用逻辑 worker 对应的信号 slot 和动态进度值表达依赖，支持单写者、多读者，
   消费者等待 value >= expected；通知不因一个消费者读取而被清除。
2. 支持代际和循环复用，说明计数宽度、溢出、初始化以及工作区跨流并发规则。
3. 区分 Set 与 AtomicAdd：单写者无需 RMW；多写者聚合必须选用相应原子更新。
4. 明确发布/等待的内存顺序和 payload 合同：接口是否等待生产者流水并处理相关
   数据缓存，或需要调用者显式执行哪些操作；不能只保证信号变化却暗示业务数据已可读。
5. 编译器应把这些操作作为有同步副作用的操作，保留轮询读取及相关访问顺序；
   MIX 展开须明确 AIC/AIV 和双 AIV lane 的执行归属，避免重复通知。
6. 后端封装 A3/A5 的实际访问、DCCI/DSB 等差异；说明参与核驻留与可推进前提。
   等待仍发生在执行核内，不通过 AICPU 重新派发阶段。


## 后续实现与验证（2026-10-16 18:07 更新）

上文保留暂停审查时的判断。用户随后授权临时普通GM notify/wait，现已完成独立实现。
首版信号进度正确，但AIC写回尚未完成就可能发出通知；在CSA notify封装加入本核
PIPE_ALL后，24组数据交接小例及完整128K/B16各三次调用通过。完整输出、Top-K及
六份cache/state对冻结动态PTO完全相同，所有保护区通过。此结果不代表通用协议证明、
A5验证、动态模板或Native数值验收。详见LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md。

<!-- SOURCE_END history-document-019 -->

---

<a id="history-document-020"></a>
### 历史原文 020：SIMPLER_CORE_ARGUMENT_CAPACITY.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_ARGUMENT_CAPACITY.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_ARGUMENT_CAPACITY.md)；原文 4,286 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-020 -->
# CSA 单 SPMD：核参数容量约束与 CPU 复现

这是独立于 PyPTO [#3009](https://github.com/hw-native-sys/pypto/issues/3009)
的接口限制，不是 A3 不支持这些数学运算，也不是 while 修复后自动消失的问题。
本次没有修改或重新安装 Simpler/PyPTO/PTOAS/PTO-ISA。

## 已确认的容量

本地 Simpler `src/common/task_interface/arg_direction.h` 定义：

```cpp
CORE_MAX_TENSOR_ARGS = 32;
CHIP_MAX_TENSOR_ARGS = 256;
CORE_MAX_SCALAR_ARGS = 16;
```

`TaskArgsTpl::add_tensor` 在达到容量时抛出 `TaskArgs: tensor capacity exceeded`。
`CoreCallable` 签名也使用 CORE 容量；Host/CHIP 或 Graph 的较大容量不能替代它。
手写 Group 的 AIC/AIV 成员共享同一份 Tensor-first payload，不能简单地为两半
各声明一套不一致的 32 参数布局。依据是 PyPTO
`docs/en/dev/passes/24-expand_mixed_kernel.md` 的 shared argument ABI 说明。

CPU 复现文件：[repro_core_arg_capacity.cpp](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_core_arg_capacity.cpp)。
在本 worktree 根目录执行：

```bash
c++ -std=c++17 \
  -I /data/pyptouser/qinchuanyu/pto-eager/simpler/src/common/task_interface \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_core_arg_capacity.cpp \
  -o /tmp/csa_core_capacity
/tmp/csa_core_capacity
```

本次结果保存在 `static/abi_capacity.log`：

```text
core_limit=32, next_tensor_rejected=TaskArgs: tensor capacity exceeded
chip_limit=256, root_entry_tensor_count=57
```

这里调用的是实际头文件中的容器，没有 NPU 初始化、设备任务或手写容量模拟。

## CSA 需要区分的几个数字

| 对象 | Tensor 数 | 含义 |
| --- | ---: | --- |
| 原 CSA 根入口 | 56 | Native 权重、metadata、cache/state 和输出 |
| 早期 arena 原型根入口 | 57 | 原 56 项加一个 GM 工作区，orchestration dtype reinterpret 未被支持 |
| 最终测试原型根入口 | 132 / 133 | 具体 GM 视图在初始化时建立，与核共享 ABI 一致 |
| 原型 MIX 核共享参数 | 132 / 133 | 长/短 history 模板；包含原计算调用用到的 GM 视图及三份协议视图 |

132/133 来自逐项实参去重，见各档 `static/*/entry.bindings.json` 的
`_entry_contract`。同一 allocation 的不同 shape/offset 视图仍是不同描述符。
工作区里的 67 个临时 allocation 已合并到一个物理缓冲，但在 Host 初始化时
恢复 typed view 并传入核，仍会消耗核参数槽。**早期根入口 57 不是核参数数。**

## 为什么没有强行把权重打包

共有 26 份只读权重。假设它们都能零拷贝恢复原视图，根入口可压成 32 项；
这只是算术上的候选方案，尚不是当前 API 可执行的接口。

其中 `wq_a/wq_b/wo_a/wo_b` 使用 NZ。当前 `tensor.view` 不能产生 NZ，
`tensor.reinterpret_view` 不接受 NZ，依据为 PyPTO
`src/ir/op/tensor_ops/transform.cpp`。只打包 ND 权重仍有 36 个根描述符。
本次既没有把 NZ 标成 ND，也没有修改权重 dtype、量化方式或搬运 Native state。

另一个不能混淆的点：在 `InCore` 里写 `reinterpret_view` 会参与 Tensor-to-Tile
转换，不能未经验证就当作核内任意 GM 指针重建接口。orchestration 又不支持
改变 dtype 的视图。因此最终测试原型在 Host 初始化绑定工作区别名，原 Native
Tensor 保持原存储，如实保留 132/133 的核 ABI。七档 meta 绑定检查通过，未启动 NPU。

## 可供组件维护者评估的方向

1. 为一个完整 MIX 核支持更大的或可变长的 Tensor 描述符集合。本原型需要至少
   133 项；不能只改一个常量，需共同检查 CoreCallable、payload、签名校验、
   wrapper 参数解包、HBG/动态 runtime 与图重放的 ABI 一致性。
2. 支持一个描述符集合参数，以及明确的核内 GM 子视图构造。编译器保留 dtype、
   NZ、stride、offset 和别名/读写信息，runtime 继续识别外部依赖；避免新增每步
   数据打包调用。此方向需 PyPTO 与 Simpler 共同定义，不能用未声明的地址转换替代。

上述为待评估接口方向，不是已实现的修复，也没有要求组件采用某个具体 ABI。
扩展前本地脚本应拒绝 NPU 提交；完整核尚未做功能、数值或性能验证。

<!-- SOURCE_END history-document-020 -->

---

<a id="history-document-021"></a>
### 历史原文 021：SIMPLER_CORE_EXPANSION_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md)；原文 5,325 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-021 -->
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

<!-- SOURCE_END history-document-021 -->

---

<a id="history-document-022"></a>
### 历史原文 022：STATIC_BASELINE_COMMIT_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/STATIC_BASELINE_COMMIT_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/STATIC_BASELINE_COMMIT_20261016.md)；原文 1,597 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-022 -->
# 静态单 SPMD 提交边界（2026-10-16）

本次提交目标：nalinaly/vllm-ascend 的 dsv4-flash-pto-v0.25.1rc1。
基于远端 f7ee8dda 建立独立提交 worktree，不混入其他会话的未提交代码。

- static/ 保留七档原始静态模板；static_local_signal/h131072_b16/ 保留已通过的临时notify/wait版本。
- benchmark_static_vs_dynamic.py 及性能报告保留128K/B16/S6同卡动态PTO对照，单SPMD当前更慢，未写成优化成功。
- evidence/ 保留逐次尝试的结果、日志、源码快照和六份L1泳道，包括失败尝试。
- reference_operator_source/ 是本次测试冻结参考算子的完整Python源码，不改变生产算子。
- references/symbolic_ir/ 保留两份生成器需要的符号IR；其他重复编译生成快照本地保留。
- GIT_RETENTION.json 记录提交/本地保留的路径与大小；artifact_inventory.json 保留全部大产物本地路径。
- component_changes/ 保留独立PyPTO和Simpler修改补丁；没有向这两个组件仓推送。
- 共享安装、生产CSA/HCA入口及pypto-lib均不在本次修改范围。

静态结果已经验证固定输入ACLGraph重放，不代表动态batch/seqlen已完成。
动态shape候选在独立目录开发，不覆盖这里的静态实现和历史结果。

复现环境仍使用已有bootstrap_isolated.py和env-dsv4-0251rc1.sh。
若原机器results/hc_pre_layer_integration_20261012/baseline_ops不可用，可将
HC_PRE_TEST_OPERATOR_SOURCE 指向此目录的reference_operator_source/；不得误用后续改动过的生产算子作为原静态基线。

<!-- SOURCE_END history-document-022 -->

---

<a id="history-document-023"></a>
### 历史原文 023：TORCH_SCHEMA_CAPACITY_20261016.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/TORCH_SCHEMA_CAPACITY_20261016.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/TORCH_SCHEMA_CAPACITY_20261016.md)；原文 4,351 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-023 -->
# 完整 CSA 单 SPMD：Torch 注册 64 入参限制

2026-10-16。本文记录完整 B4 已完成默认 PYPTO 代码生成后，在真实注册入口
发现的新限制。它与 Simpler CORE 的 Tensor 描述符容量是不同层的问题。

## 实际结果与归属

环境为本会话既有 `torch 2.10.0+cpu`、`torch_npu 2.10.0.post2`；没有替换
PyTorch 或共享组件安装。完整 `entry_jit.py` 的 132 个具名 Tensor 在
`pypto.torch.register` → `RegistrationSignature.define` → `torch.library.define`
时报错：

```text
The function schema has 132 arguments but this PyTorch build only supports 64
```

错误发生在 dispatcher schema 定义阶段，还没有构造 Meta 实参、编译 CCE
二进制或启动 NPU。因此这不是 A3 计算错误，也不能归因于 Simpler 扩容失败。
不推断其他 Torch 版本的上限；本轮没有为此重建或修改 PyTorch。

| 检查 | 实际结果 | 不能据此推出什么 |
| --- | --- | --- |
| 64 个具名 Tensor 的 schema | 定义成功 | 未测试完整 CSA |
| 65、132 个具名 Tensor 的 schema | 定义失败，明确上限 64 | 不是 CORE 参数装载失败 |
| 56 个只读 Tensor 列表 + 76 个可变 Tensor 列表 | CPU、Meta 成功；共 132 Tensor、2 个 schema 参数，原对象身份保持 | 只是 Torch 小例子，没有接入 PyPTO |
| 上述 Tensor[] 小例子的 functionalization | 失败，报告可变 alias schema 不受支持 | 不能把 Tensor[] 当成现成的整网解决方案 |
| 扩容后的 CORE 133/256 Tensor + 16 scalar | 之前四类 runtime 设备验证已通过 | 不改变 Torch dispatcher schema 的限制 |

## 复现与原始证据

- [独立 CPU 复现](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_torch_schema_capacity.py)：只依赖 Torch。
- [完整 CSA Meta 注册驱动](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/test_static_jit_meta.py)：复用真实生成入口和准备对象。
- [独立复现结果](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/torch_schema_capacity_cpu_v1/result.json)。
- [完整注册结果和异常栈](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_b4_jit_meta_v1/result.json)。
- `session_logs/csa-torch-schema-capacity-cpu-v1.log`。
- `session_logs/csa-static-b4-jit-meta-v1.log`。

独立 CPU 复现命令（使用本会话 Python 环境，不占 NPU）：

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python repro_torch_schema_capacity.py --output /tmp/csa_torch_schema_repro
```

完整入口复现先按 [独立组件环境](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md#独立环境复现)
设置两个 worktree，在本目录执行：

```bash
python bootstrap_isolated.py test_static_jit_meta.py \
  --case static/h131072_b4 --output /tmp/csa_static_jit_meta_repro
```

以上是供后续接手复现的命令；本轮归档没有再次执行这些测试。

## 历史候选入口：已取消，不是已完成实现

2026-10-16 后续用户明确选择直接 native binding 调用。下述 62 项方案仅保留
调查历史，不再等待确认、不再推进。已核对现有 PyPTO JIT 直调入口及 TorchAir
示例，见 [直调入口说明](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/DIRECT_NATIVE_ENTRY_20261016.md)。

建议保留原 56 个 Native 具名 Tensor，把临时存储按六种 dtype 分成六个
workspace Tensor，根入口合计 62 项。内部继续使用展开的独立描述符，
不要求 CORE 从 132 项降到 62 项。预期由固定偏移 view 建立内部绑定，
不引入逐步 metadata 搬运或新的设备适配算子。

当前 B4 临时视图的 dtype 统计为 FP32 42、BF16 10、INT32 8、INT8 4、
FP16 1、INT64 2；另外三个协议区也是 INT64。Native 的 56 个根参数在
核体内对应 60 个具体描述符，因此当前总数是 60 + 69 + 3 = 132。
根入口参数数目与内部视图描述符数目不能混用。

这个候选与之前失败的单 INT8 arena 不同：每个 workspace 已具有自己的
dtype，避免依赖 orchestration 的跨 dtype reinterpret。仍需实际验证
kernel 模式的 view 绑定、mutation/返回别名、注册和运行时行为，不能因
62 小于 64 就预记成功。并发调用仍需各自 workspace。

此前按原目标“有过不去的关键点停下讨论”询问用户。用户现已选择直调；
没有实现这套 62 项新入口，也没有修改共享 PyTorch。
本问题尚未提交上游 issue/PR，后续真实链接统一登记到
[ISSUES_AND_PRS.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/ISSUES_AND_PRS.md)。

<!-- SOURCE_END history-document-023 -->

---

<a id="history-document-024"></a>
### 历史原文 024：evidence/component_repro_kit_20261016/JIT_SINGLE_GROUP_NOTES.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/JIT_SINGLE_GROUP_NOTES.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/JIT_SINGLE_GROUP_NOTES.md)；原文 2,621 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-024 -->
# JIT 的单 MIX Group 表达：当前核对结果

目标仍为一次 24 组 MIX 提交，不用两个提交替代。本说明记录接口表达限制，
不是宣称所有 JIT 写法均不可实现，也不要求组件必须新增某一个具体装饰器。

## 已验证的两条路径

显式 program 原型包含 `FunctionType.Group`，根入口 SPMD 只调用这个 Group。
其部分 orchestration 产物已确认只有一次 MIX 提交；完整核编译问题另见
[COMPILE_AND_RESUME.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/COMPILE_AND_RESUME.md)。

直接将它改写为 JIT 的以下结构，会在解析阶段被拒绝：

```python
with pl.spmd(24, sync_start=True):
    with pl.cluster():
        c = cube(a, b, c)
        y = vector(x, y)
```

错误为 `dispatch body launches 2 kernels; only one kernel dispatch per SPMD scope`。
源于 parser 对 dispatch body 的检查；后端 `FindFirstInnerCall` 只能取一个 callee。
不能通过新增无用 block_idx 读取欺骗该检查。

进一步检查了先用 `@pl.jit.opaque` 包裹 cluster，再让 SPMD 只调用一次 wrapper：
[repro_jit_group_wrapper.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/repro_jit_group_wrapper.py)。它是 5 参数、无模型依赖、
无 while 的 CPU 编译小例子。两个物理计算核可生成，但 orchestration 拒绝：

```text
InferFunctionCoreType expects AIC or AIV (Spmd/Group are filtered upstream);
got Opaque on function 'group_wrapper'.
```

证据：`../results/csa_static_spmd_feasibility_20261015/jit_group_wrapper_v1/`。
这说明该 opaque wrapper 不能当作已支持的 Group kernel 入口。

## 本地源码依据与接续选择

- `python/pypto/language/parser/ast_parser.py`：SPMD 的单 callee 检查。
- `src/ir/transforms/outline_cluster_scopes_pass.cpp`：一个 cluster 只允许一个
  嵌套 SPMD，不能用两个 SPMD 分别包住 AIC/AIV 再冒充一次 MIX。
- `python/pypto/jit/decorator.py`、`specializer.py`：现有 JIT 类型没有原生
  DSL Group 装饰器；extern mixed 可以生成 Group，但需已有正确的物理核产物。
- `tests/ut/jit/test_roundtrip.py` 明确将模块级 `pl.function` 调用列为 JIT
  依赖发现不覆盖的模式；不能直接把 program 的 IR 函数塞给 JIT 当作完成接入。

后续可选组件正式支持的 Group 调用表达，或在 #3009 修复后验证自动 MIX 内联
表达。若采用 extern mixed 承接 PyPTO 生成的两份物理核，仍须验证同一参数布局、
双 AIV、一次 24 组提交及 kernel runtime/当前流，不能用 program runner 替代。

目前 `entry_jit.py` 是准备中的源码，不是已通过 `pypto.torch.register` 的生产入口。

<!-- SOURCE_END history-document-024 -->

---

<a id="history-document-025"></a>
### 历史原文 025：evidence/component_repro_kit_20261016/MEMORY_REUSE_PHASES_REPRO.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/MEMORY_REUSE_PHASES_REPRO.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/MEMORY_REUSE_PHASES_REPRO.md)；原文 3,648 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-025 -->
# 两个独立可编译 AIC 阶段融合后的片上内存分配复现

日期：2026-10-16。此问题独立于 while 的 #3009，也独立于 Simpler 的 32 Tensor
容量：下面的融合复现只需要 13 个 Tensor 参数，且全部是 CPU 编译。
未修改依赖仓库、未降低 tile 尺寸或 pipeline stage。

## 输入及对照结果

从当前 B4 静态原型直接抽取两个阶段：

- `indexer_score_topk_native_pair_aic`，原 AIC batch 13；独立入口 10 个 Tensor。
- `proj_a_mm`，原 AIC batch 15；独立入口 4 个 Tensor。
- 按原先后顺序放进一个 AIC 函数，两者共享 records 参数，合计 13 个 Tensor。

抽取脚本：[build_memory_probe.py](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/build_memory_probe.py)。保留原数学、分片读取、
FFTS 操作、pipeline 和本核完成边界；删除进度发布/等待及无关参数。
三个自包含 PyPTO 文件在 `static/memory_indexer/entry.py`、
`static/memory_oproj/entry.py`、`static/memory_pair_compact/entry.py`。
无需模型权重、Native 服务、真实 Tensor 数据或 NPU 即可编译。

| 编译输入 | PYPTO | DSA_RP |
| --- | --- | --- |
| Indexer 单段 | 代码生成及 PTOAS C++ 生成通过 | 未测 |
| O-proj 单段 | 代码生成及 PTOAS C++ 生成通过 | 未测 |
| 两段融合，13 参数 | Right 81920 B > 65536 B | canonical greedy 找不到容量内布局 |

结果目录位于 `../results/csa_static_spmd_feasibility_20261015/`：

- `memory_indexer_PYPTO/result.json`
- `memory_oproj_PYPTO/result.json`
- `memory_pair_compact_PYPTO/result.json`
- `memory_pair_compact_DSA_RP_v2/result.json`

`memory_probe_PYPTO`、`memory_probe_DSA_RP` 是尚未裁掉无关根参数的早期复现，
也失败。`memory_pair_compact_DSA_RP` 是本地编译脚本的 PassContext 参数冲突，
**不算组件失败证据**；该调用方式修正后的结果以 `_v2` 为准。

## 分配审计

`static/memory_probe_summary.json` 汇总 MemoryReuse 后各物理复用槽的最大字节数。
这是 IR 中的分配量，不是运行时测得的瞬时峰值。

| 输入 | Left / KiB | Right / KiB | Acc / KiB | Mat / KiB |
| --- | ---: | ---: | ---: | ---: |
| Indexer 单段 | 60 | 64 | 100 | 112 |
| O-proj 单段 | 64 | 64 | 64 | 256 |
| 两段融合 | 80 | 80 | 100 | 256 |

两个原阶段各自可以放进目标资源，合并后 Left/Right 复用槽总量变成 80 KiB。
应检查不同阶段的 allocation 生命周期、复用分组、pipeline 硬约束与候选布局。
**单段通过和顺序关系提示存在可复用空间，但不是完整硬件可行性证明**；仍须
正确处理原流水和同步，不能直接删除编译器约束或把失败当作硬件必然不支持。

## 脱离 vLLM 的 CPU 编译

在组件维护者正常可用的 PyPTO/PTOAS 环境中，以 `entry.py` 为输入：

```python
from pathlib import Path
import pypto.language as pl
from pypto import ir, passes
from pypto.runtime import RunConfig

source = Path("static/memory_pair_compact/entry.py")
program = pl.parse(source.read_text(), filename=str(source.resolve()))
compiled = ir.compile(program, **RunConfig(
    platform="a2a3",
    memory_planner=passes.MemoryPlanner.PYPTO,  # 对照时改为 DSA_RP
    save_kernels=True,
    save_kernels_dir="/tmp/csa_memory_pair_repro",
    dump_passes=True,
).compile_kwargs())
```

将输入分别替换为两个单段入口即可得到对照。不要调用 compiled 对象或 `.load()`。
这两个段缺少其前序数据生产者和 MIX 的 AIV 半边，独立上卡不属于此复现的范围。

这里没有把两段小例子的 PASS 当成完整 CSA PASS；目标是让分配问题可直接接手。

<!-- SOURCE_END history-document-025 -->

---

<a id="history-document-026"></a>
### 历史原文 026：evidence/component_repro_kit_20261016/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/README.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/README.md)；原文 2,400 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-026 -->
# CSA 单 SPMD：组件复现包

本包不包含模型权重、KV 快照、设备二进制或性能数据；只需已安装的 PyPTO/PTOAS，
容量复现另需 Simpler 源码头文件。所有 Python 命令只编译，不运行 NPU。

## 1. 两个独立可编译阶段，融合后片上规划失败

```bash
python portable_memory_compile.py --case indexer --output /tmp/csa_mem_indexer
python portable_memory_compile.py --case oproj --output /tmp/csa_mem_oproj
python portable_memory_compile.py --case pair --output /tmp/csa_mem_pair
python portable_memory_compile.py --case pair --planner DSA_RP --output /tmp/csa_mem_pair_dsa
```

本次前两项代码生成通过，后两项失败。融合复现只需13个Tensor参数，
独立于Simpler的32参数限制。详见 MEMORY_REUSE_PHASES_REPRO.md；本包已有结果在 reports/。
原说明中提到的完整工作树路径仅是历史证据位置，运行本包不需要那些路径。

## 2. JIT 单 Group 表达

```bash
python repro_jit_group_wrapper.py --output /tmp/csa_jit_group
```

这是5参数、无while的小例子。本次编译器未将opaque wrapper识别为可提交Group。
具体范围见 JIT_SINGLE_GROUP_NOTES.md，不据此断言所有JIT表达都不可行。
本包版本只删除了本机的SDK激活代码；计算、接口及编译检查保持原样。

## 3. CORE 参数容量

```bash
c++ -std=c++17 -I /path/to/simpler/src/common/task_interface \
    repro_core_arg_capacity.cpp -o /tmp/csa_core_capacity
/tmp/csa_core_capacity
```

这是实际TaskArgs容器检查。本次CORE容量32，完整CSA原型需要132/133。
示例中CHIP的57参数仅用于区分CHIP与CORE容量，不是最终融合核参数数。
详见 SIMPLER_CORE_ARGUMENT_CAPACITY.md。

## 4. 已提交的 while 活跃性问题

https://github.com/hw-native-sys/pypto/issues/3009

```bash
python repro_mixed_while_loop_state.py --variant scalar --output /tmp/csa_while_scalar
python repro_mixed_while_loop_state.py --variant tile --output /tmp/csa_while_tile
```

无需为了复现读取权重或执行设备任务。修复后应核对while条件、初值、回边和carry，
不能只凭进程退出码判断。包内的Python驱动会保留失败记录并返回非零退出码。

这些复现不是完整CSA功能测试。恢复完整CSA时仍需解决参数承载与完整融合编译，
再按功能、精度、性能的顺序验证。

<!-- SOURCE_END history-document-026 -->

---

<a id="history-document-027"></a>
### 历史原文 027：evidence/component_repro_kit_20261016/SIMPLER_CORE_ARGUMENT_CAPACITY.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/SIMPLER_CORE_ARGUMENT_CAPACITY.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/SIMPLER_CORE_ARGUMENT_CAPACITY.md)；原文 4,286 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-027 -->
# CSA 单 SPMD：核参数容量约束与 CPU 复现

这是独立于 PyPTO [#3009](https://github.com/hw-native-sys/pypto/issues/3009)
的接口限制，不是 A3 不支持这些数学运算，也不是 while 修复后自动消失的问题。
本次没有修改或重新安装 Simpler/PyPTO/PTOAS/PTO-ISA。

## 已确认的容量

本地 Simpler `src/common/task_interface/arg_direction.h` 定义：

```cpp
CORE_MAX_TENSOR_ARGS = 32;
CHIP_MAX_TENSOR_ARGS = 256;
CORE_MAX_SCALAR_ARGS = 16;
```

`TaskArgsTpl::add_tensor` 在达到容量时抛出 `TaskArgs: tensor capacity exceeded`。
`CoreCallable` 签名也使用 CORE 容量；Host/CHIP 或 Graph 的较大容量不能替代它。
手写 Group 的 AIC/AIV 成员共享同一份 Tensor-first payload，不能简单地为两半
各声明一套不一致的 32 参数布局。依据是 PyPTO
`docs/en/dev/passes/24-expand_mixed_kernel.md` 的 shared argument ABI 说明。

CPU 复现文件：[repro_core_arg_capacity.cpp](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/component_repro_kit_20261016/repro_core_arg_capacity.cpp)。
在本 worktree 根目录执行：

```bash
c++ -std=c++17 \
  -I /data/pyptouser/qinchuanyu/pto-eager/simpler/src/common/task_interface \
  tests/pypto_test/csa_static_spmd_feasibility_20261015/repro_core_arg_capacity.cpp \
  -o /tmp/csa_core_capacity
/tmp/csa_core_capacity
```

本次结果保存在 `static/abi_capacity.log`：

```text
core_limit=32, next_tensor_rejected=TaskArgs: tensor capacity exceeded
chip_limit=256, root_entry_tensor_count=57
```

这里调用的是实际头文件中的容器，没有 NPU 初始化、设备任务或手写容量模拟。

## CSA 需要区分的几个数字

| 对象 | Tensor 数 | 含义 |
| --- | ---: | --- |
| 原 CSA 根入口 | 56 | Native 权重、metadata、cache/state 和输出 |
| 早期 arena 原型根入口 | 57 | 原 56 项加一个 GM 工作区，orchestration dtype reinterpret 未被支持 |
| 最终测试原型根入口 | 132 / 133 | 具体 GM 视图在初始化时建立，与核共享 ABI 一致 |
| 原型 MIX 核共享参数 | 132 / 133 | 长/短 history 模板；包含原计算调用用到的 GM 视图及三份协议视图 |

132/133 来自逐项实参去重，见各档 `static/*/entry.bindings.json` 的
`_entry_contract`。同一 allocation 的不同 shape/offset 视图仍是不同描述符。
工作区里的 67 个临时 allocation 已合并到一个物理缓冲，但在 Host 初始化时
恢复 typed view 并传入核，仍会消耗核参数槽。**早期根入口 57 不是核参数数。**

## 为什么没有强行把权重打包

共有 26 份只读权重。假设它们都能零拷贝恢复原视图，根入口可压成 32 项；
这只是算术上的候选方案，尚不是当前 API 可执行的接口。

其中 `wq_a/wq_b/wo_a/wo_b` 使用 NZ。当前 `tensor.view` 不能产生 NZ，
`tensor.reinterpret_view` 不接受 NZ，依据为 PyPTO
`src/ir/op/tensor_ops/transform.cpp`。只打包 ND 权重仍有 36 个根描述符。
本次既没有把 NZ 标成 ND，也没有修改权重 dtype、量化方式或搬运 Native state。

另一个不能混淆的点：在 `InCore` 里写 `reinterpret_view` 会参与 Tensor-to-Tile
转换，不能未经验证就当作核内任意 GM 指针重建接口。orchestration 又不支持
改变 dtype 的视图。因此最终测试原型在 Host 初始化绑定工作区别名，原 Native
Tensor 保持原存储，如实保留 132/133 的核 ABI。七档 meta 绑定检查通过，未启动 NPU。

## 可供组件维护者评估的方向

1. 为一个完整 MIX 核支持更大的或可变长的 Tensor 描述符集合。本原型需要至少
   133 项；不能只改一个常量，需共同检查 CoreCallable、payload、签名校验、
   wrapper 参数解包、HBG/动态 runtime 与图重放的 ABI 一致性。
2. 支持一个描述符集合参数，以及明确的核内 GM 子视图构造。编译器保留 dtype、
   NZ、stride、offset 和别名/读写信息，runtime 继续识别外部依赖；避免新增每步
   数据打包调用。此方向需 PyPTO 与 Simpler 共同定义，不能用未声明的地址转换替代。

上述为待评估接口方向，不是已实现的修复，也没有要求组件采用某个具体 ABI。
扩展前本地脚本应拒绝 NPU 提交；完整核尚未做功能、数值或性能验证。

<!-- SOURCE_END history-document-027 -->

---

<a id="history-document-028"></a>
### 历史原文 028：evidence/simpler_core256_v1/LOCAL_ARTIFACT_PATHS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/simpler_core256_v1/LOCAL_ARTIFACT_PATHS.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/simpler_core256_v1/LOCAL_ARTIFACT_PATHS.md)；原文 927 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-028 -->
# 本地大参数验证产物

Simpler worktree：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016`。

- `outputs/TestCapacity_host_build_graph_133_tensors_133_scalars_16_20261016_093915/args_dump/args_dump.json` 及同目录 `args.bin`；此处只复制 JSON，二进制留在上述 worktree。
- `outputs/TestCapacity_host_build_graph_256_tensors_256_scalars_16_20261016_093916/args_dump/args_dump.json` 及同目录 `args.bin`；此处只复制 JSON，二进制留在上述 worktree。
- `outputs/TestCapacity_tensormap_and_ringbuffer_133_tensors_133_scalars_16_20261016_094220/args_dump/args_dump.json` 及同目录 `args.bin`；此处只复制 JSON，二进制留在上述 worktree。
- `outputs/TestCapacity_tensormap_and_ringbuffer_256_tensors_256_scalars_16_20261016_094223/args_dump/args_dump.json` 及同目录 `args.bin`；此处只复制 JSON，二进制留在上述 worktree。

<!-- SOURCE_END history-document-028 -->

---

<a id="history-document-029"></a>
### 历史原文 029：evidence/static_vs_dynamic_perf_b16_v1/REPORT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/REPORT.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/REPORT.md)；原文 4,706 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-029 -->
# 128K/B16：单 SPMD 与原动态 PTO 性能对比

用户于 2026-10-16 要求只比较单 SPMD 与原动态 PTO，不测 Native。

## 被测范围与口径

- history=131072、batch=16、S=6、T=96；正式 W8A8 权重，第 4 层。
- 原动态 PTO 使用 `results/hc_pre_layer_integration_20261012/baseline_ops` 冻结实现，
  与静态排程提取、上一轮精确比较同源。
- 单 SPMD 使用 `static_local_signal/h131072_b16/entry_jit.py` 与已验证的
  `local_signal_adapter.py`；notify 前包含本核 PIPE_ALL。
- 两边在同卡同进程、同一流串行执行，共用权重、输入和缓存地址，AB/BA 交替。
- 每次调用前恢复完全相同的缓存初态。恢复和输出毒化不进入计时区间；进度不清零。
- 先编译及 eager 比较，再捕获只含 PTO 调用的 ACLGraph；重放结果先对动态 eager
  精确比较。该图仅用于固定输入计时，不代表完整请求生命周期或连续 decode 验收。
- 设备 Event：每侧预热 5 次、采样 20 次。Event 在图外记录，不复用图内旧时间戳。
- L1：每侧独立 3 个窗口，一窗口只运行一侧一次图重放。
  主指标为原始 `aicore_tasks` 的 `max(end)-min(start)`，按记录中的设备时钟换算。
  不把收任务时间当成 incore 开始，不累加各核时长；保留内部等待及任务间隙。
- 两边使用同一独立 PyPTO/Simpler 调试安装、TMR runtime 和 L1 构建配置。
  Event 采样时 DFX 窗口关闭；L1 与 Event 分开报告，不混合样本。
- 不执行 Native attention forward，不使用此前 Native 性能作为本次对照。

## 首轮任务

- task ID：`task_20261016_183457_19106108116`
- 输出目录：`../results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1`
- 启动：`run_static_vs_dynamic_performance.sh <全新输出目录>`，必须通过 `task-submit --device auto`。
- 最终状态：completed (exit=0)，输出、状态、信号与保护区检查通过。

测试入口为 `benchmark_static_vs_dynamic.py`，未修改生产 CSA、数学计算体或依赖组件。

## 最终结果

主口径为 L1 incore 首尾跨度，单位 μs。动态 PTO 共 66 个计算 task、973 个物理核执行记录；
单 SPMD 为 1 个 MIX task、72 个物理核执行记录。两边均覆盖 24 AIC + 48 AIV。

| 采样 | 原动态 PTO | 单 SPMD |
| --- | ---: | ---: |
| L1 第 1 次 | 897.70 | 1519.30 |
| L1 第 2 次 | 900.14 | 1380.22 |
| L1 第 3 次 | 877.42 | 1350.06 |
| **L1 中位数** | **897.70** | **1380.22** |
| Event 中位数（各 20 次） | 942.62 | 1467.75 |

单 SPMD 的 L1 中位数增加 482.52 μs，耗时增加 **53.75%**，动态/静态比为 **0.6504×**。
设备 Event 中位数增加 525.13 μs，耗时增加 **55.71%**。当前静态化版本没有取得加速。

L1 动态范围 877.42～900.14 μs；静态范围 1350.06～1519.30 μs。
Event 动态范围 926.12～967.02 μs；静态范围 1442.78～1819.22 μs，所有样本保留，未剔除离群值。
L1 与 Event 是不同采样窗口，不能把两者中位数直接相减当作某个阶段的精确开销。

退步已经发生在 incore 首尾区间内，不能只归因于 Simpler/orchestrator 的外围首尾调度。
单 SPMD 的 L1 仅显示每核整段融合计算，没有内部阶段时间戳；本次结果尚不能分别量化
notify/wait、缓存维护、PIPE_ALL、静态排程或融合后编译结果各自的影响。未继续做优化。

## 正确性和范围

- 单 SPMD eager、两侧图捕获后重放、两侧最终 L1 重放的输出、Top-K、六份 cache/state
  均与动态 eager 参考逐元素完全一致。无 Native forward 比较。
- 单 SPMD 累计执行 30 个代际，进度区不清零；72 核末次进度和工作区保护通过。
- Native 格式缓存的未写区域、只读 metadata 保护检查通过。
- 这是固定输入、固定地址的计时图；每次恢复缓存，不代表连续 decode 轨迹、动态 B/S
  或完整整网性能。

## 数据与复现

- 原始结果：`../results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1/result.json`。
- 汇总：同目录 `summary.json`，含全部 20 次 Event 及 3 次 L1 数据。
- 泳道：同目录 `l1/dynamic_pto/step_{1,2,3}/swimlane.json` 和
  `l1/single_spmd/step_{1,2,3}/swimlane.json`，各目录同时保留原始记录与名称映射。
- `download/` 汇总六份重命名的泳道、报告与 JSON；可以直接用于 Perfetto/Chrome tracing。
- 归档副本：本目录 `evidence/static_vs_dynamic_perf_b16_v1/`。

正式权重、大张量、动态库及设备二进制不入归档；只记录本地路径，没有计算 hash。

<!-- SOURCE_END history-document-029 -->

---

<a id="history-document-030"></a>
### 历史原文 030：evidence/static_vs_dynamic_perf_b16_v1/download/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/README.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/README.md)；原文 1,416 字节。本会话静态专项记录或随证据包保存的文档快照，全文保留。

<!-- SOURCE_BEGIN history-document-030 -->
# CSA 128K/B16/S6：原动态 PTO 与单 SPMD

同卡正式权重测试，仅比较 PTO，task_20261016_183457_19106108116 已完成。
两侧输出、Top-K、六份 cache/state 精确相同，保护区和30代际进度检查通过。
每步恢复初态，这不是连续decode轨迹。

| 口径（μs，中位数） | 动态 PTO | 单 SPMD | 耗时变化 |
| --- | ---: | ---: | ---: |
| L1 incore 首尾（各3次） | 897.70 | 1380.22 | +53.75% |
| 图外 Event（各20次） | 942.62 | 1467.75 | +55.71% |

六份 JSON 是 Perfetto/Chrome tracing 泳道；只看 incore 起止，不把接收任务时间当成计算开始。
summary.json 保留全部样本和原始路径。

- [CSA_128K_B16_S6_dynamic_pto_L1_step1.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/CSA_128K_B16_S6_dynamic_pto_L1_step1.json)
- [CSA_128K_B16_S6_dynamic_pto_L1_step2.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/CSA_128K_B16_S6_dynamic_pto_L1_step2.json)
- [CSA_128K_B16_S6_dynamic_pto_L1_step3.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/CSA_128K_B16_S6_dynamic_pto_L1_step3.json)
- [CSA_128K_B16_S6_single_spmd_L1_step1.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/CSA_128K_B16_S6_single_spmd_L1_step1.json)
- [CSA_128K_B16_S6_single_spmd_L1_step2.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/CSA_128K_B16_S6_single_spmd_L1_step2.json)
- [CSA_128K_B16_S6_single_spmd_L1_step3.json](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/evidence/static_vs_dynamic_perf_b16_v1/download/CSA_128K_B16_S6_single_spmd_L1_step3.json)

[完整报告](/data/pyptouser/qinchuanyu/pto-eager/.worktrees/hc-pre-layer-integration-20261012/tests/pypto_test/results/csa_static_spmd_feasibility_20261015/static_vs_dynamic_perf_b16_v1/REPORT.md)

<!-- SOURCE_END history-document-030 -->

---

<a id="history-document-031"></a>
### 历史原文 031：upstream_issues/incore_p2p_notify_wait_20261016/issue_body.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/incore_p2p_notify_wait_20261016/issue_body.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/incore_p2p_notify_wait_20261016/issue_body.md)；原文 9,189 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-031 -->
### 需求概述

希望 PyPTO 提供用于**同一个融合后的 InCore task / MIX SPMD 内部**的核间点对点通知与等待接口：

> 生产者核完成指定阶段并发布数据后，通知依赖它的消费者核；消费者只等待自己的必要前驱，无依赖的核继续执行。

希望支持普通本地 GM 信号 Tensor、动态进度阈值和循环代际。可以扩展现有信号 API 或复用已有 PTO-ISA 能力，具体命名与实现方式由组件设计决定。

这是功能需求，不将当前原型的停滞直接认定为 PyPTO 编译器 bug。

### 使用场景与动机

我们正在将 DeepSeek-V4 Flash CSA 中原有多个 InCore task 的计算体与依赖，融合到一个 MIX SPMD 中。当前目标场景为 A3、24 个核组，即 24 AIC + 48 AIV；先验证 history=128K、B16 的静态原型，最终仍需要动态 batch/sequence metadata。

融合前，任务之间的依赖由运行时调度维护。融合后，同一 task 内需要保留原有的核间生产者/消费者关系：

- 某个生产者完成阶段 k 后，消费者 A、B 可以继续读取该阶段结果。
- 与该生产者无依赖的消费者 C 不需要等待。
- 一个消费者可能有多个前驱，必须等所有必要前驱达到各自的进度。
- 核间执行顺序和每核工作量可以不同，不能把所有阶段变成全体集合 barrier。
- 信号只表示阶段完成；相关 workspace 的覆盖/复用仍必须保留原依赖或独立确认，消费者不能因看到更大的进度值就读取已被覆盖的数据。

希望将上述依赖完全保留在已启动的 InCore task 内，阶段之间不重新通过 AICPU 派发。此需求不预设一定能取得性能收益，首先要求同步语义正确、可验证、可维护。

### 与现有接口的区别

已核对公开主线版本 [8325a8de8422678d6699e17f31cef953ace66445](https://github.com/hw-native-sys/pypto/commit/8325a8de8422678d6699e17f31cef953ace66445)，当前并非只有 syncall：

| 接口 | 已有能力 | 本次希望补齐的部分 |
| --- | --- | --- |
| [pl.system.syncall](https://github.com/hw-native-sys/pypto/blob/8325a8de8422678d6699e17f31cef953ace66445/python/pypto/language/op/system_ops.py#L150) | 参与核集合的 barrier，包含 HARD/SOFT 模式 | 按指定前驱和进度等待，保留无关核之间的独立推进。 |
| [pl.system.sync_set / sync_wait](https://github.com/hw-native-sys/pypto/blob/8325a8de8422678d6699e17f31cef953ace66445/python/pypto/language/op/system_ops.py#L90) | Cube/Vector 硬件事件，event_id、pipe、ffts_mode 等 | 普通 GM 信号上的可复用进度阈值、逻辑 worker 间依赖，以及单写者多消费者的明确合同。 |
| [pld.system.notify / wait](https://github.com/hw-native-sys/pypto/blob/8325a8de8422678d6699e17f31cef953ace66445/python/pypto/language/distributed/op/system_ops.py#L124) | DistributedTensor 通信窗口上的跨 rank 信号，lowering 到 TNOTIFY/TWAIT | 同一设备、同一 InCore task 内的普通 GM Tensor 信号，无需为此引入通信组和跨 rank 窗口。 |

相关 issue：

- #2072 已提供显式 Cube/Vector 事件接口；本需求不是重复要求增加 sync_set/sync_wait。
- #2130 涉及流水与跨核调度表达，和本需求相关，但本次范围是底层通知/等待合同。
- #1906 关注跨 rank TPUT 的完成与发布；本需求优先解决单设备本地 GM 的核间交接，不要求一并解决远端 TPUT 完成问题。

### 建议的 API 与行为

以下仅为行为示意，**不是当前可直接执行的 PyPTO API，也不限定最终名称或签名**。代码位于同一个已启动的 InCore/SPMD 内：

```python
# signals：普通 GM 信号 Tensor；每个生产者的信号独占所需缓存行。
# worker：此次启动中的逻辑 worker，包含明确的 AIC/AIV lane 映射。
# epoch_step / required_step：设备端进度值，可在循环中变化。

if worker == producer:
    produce_payload(...)
    # 按接口合同完成所需的数据流水、写回和发布顺序。
    pl.system.notify(signals, offsets=[producer, 0], value=epoch_step)

if worker == consumer_a or worker == consumer_b:
    pl.system.wait(
        signals, offsets=[producer, 0],
        expected=required_step, cmp="ge",
    )
    # 按接口合同完成消费者所需的数据缓存维护与读取顺序。
    consume_payload(...)

if worker == independent_worker:
    independent_work(...)
```

希望接口至少明确以下语义：

1. **点对点及一对多依赖。** 单写者可以有多个消费者；wait 不消费或清零信号。消费者可以逐个等待自己的前驱，也可由组件提供等价批量接口。不要求所有启动核在相同位置调用。
2. **动态阈值和复用。** 支持设备端 slot/expected value、循环中的单调进度及代际复用。说明信号类型、访问宽度/对齐、初始化、溢出处理和跨调用生命周期，防止上一轮信号误满足本轮等待。
3. **单写者与多写者分开。** 单写者 Set 不因多个读者而必须使用 atomic RMW。若提供多写者计数聚合，则明确 AtomicAdd 等操作及其合同；这不是最小单写者接口的前置条件。
4. **信号发布和读取保证。** notify 的写入应按规定发布；wait 必须重新观察进度，不能因编译器复用旧值或 Scalar DCache 命中旧值而无法前进。A3/A5 后端各自处理必要的缓存维护、内存顺序和读取路径。
5. **业务数据同步合同。** 明确 notify/wait 是只同步信号，还是同时提供对指定 payload 的 release/acquire 保证。若需要显式数据 fence/cache 操作，请提供可组合的公开接口与正确示例，明确其覆盖范围及执行次序。不能只保证标记变化，却暗示消费者一定可读业务结果。
6. **编译器与 MIX 语义。** 标记为具有同步副作用的操作，防止错误删除、提升、合并或重排；明确 AIC、AIV0、AIV1 的执行归属，避免一次逻辑通知在 MIX 展开后重复发布。同时说明与自动同步、内存复用和现有事件资源的相互作用。
7. **可推进条件。** 说明参与核驻留、启动规模和等待拓扑的前提。接口不承诺挽救循环依赖或等待未获调度核造成的死锁，也不应隐含插入全卡 barrier。

### 内存语义方面需要避免的误解

- Scalar DCache 非一致性是独立于编译器 volatile 的问题；volatile 不能建立跨核缓存一致性。
- 原子更新解决并发修改冲突，不自动等价于业务数据发布。
- DCCI 可能包含 clean/writeback，不能一律理解为纯失效；必须考虑整条 cacheline 的所有权。
- DSB 是相应本核操作的完成与顺序约束，本身不是跨核会合。
- 单写者进度可以通过正确发布与读取协议实现，不要求机械加入多写者 atomic。
- A3 的 st_atomic 等路径与 A5 的原子接口需要分别处理，不能概括成“A3 没有任何 atomic”。

这些细节应由公共接口合同和后端实现集中承载，避免每个融合算子各自手写普通 Scalar load/store、DCCI、轮询循环。

### 已考虑的替代方案

- **阶段间 syncall(SOFT)：** 改变原有依赖偏序，引入无关核之间的等待；不能作为点对点通知的等价替换。
- **继续保留独立 task：** 现有正确路径可以保留，但不能满足本次单个融合 task 内部表达依赖的目标。
- **手写 GM 进度轮询：** 当前原型采用每 worker 单写自己的进度行、消费者按前驱表轮询。必须自行承担缓存发布、编译器读取、代际及 payload 交接的完整合同，适合诊断但不适合作为各算子重复实现的基础机制。
- **手改生成 C++ 的 volatile：** 简化控制表诊断中观察到结果变化，但不构成缓存协议正确或完整 CSA 通过的证据，也不是公共 API 方案。
- **复用现有通知后端：** 欢迎基于已有 TNOTIFY/TWAIT 或其他硬件机制实现；本需求不预设要重新发明底层同步，只要求普通本地 GM 场景的公开表达和明确合同。

### 建议的验收范围

不要求以完整 CSA 模型作为组件的最小验收用例，建议用独立小例子覆盖：

- 单生产者/单消费者、单生产者/多个消费者，以及消费者有多个必要前驱。
- 同一个 MIX 中的 AIC/AIV、不同核组与双 AIV lane，另有无关 worker 独立推进。
- 多轮动态进度及重复调用，不被旧代际误唤醒；明确计数回绕/重新初始化规则。
- 实际 payload 跨多个 cacheline 的读写交接，包含 Scalar 与搬运流水的代表路径，不只检查信号最终值。
- 最小 kernel-mode/JIT 路径及声明支持的 program 路径；若支持图重放，覆盖固定地址信号的复用合同。
- 代码生成验证同步副作用与 lane 归属；设备用例有有限超时和可诊断的失败信息。优先 A3，并明确 A5 的支持与验证状态。

当前完整融合 CSA 尚未得到功能/精度通过结论；本 issue 不宣称根因已归属于某个组件，也不把几轮控制表诊断通过当成同步协议正确性证明。

<!-- SOURCE_END history-document-031 -->

---

<a id="history-document-032"></a>
### 历史原文 032：upstream_issues/p1_group_inout.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p1_group_inout.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p1_group_inout.md)；原文 8,774 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-032 -->
### Component

Other (please specify in description)

### Description

组件为 ExpandMixedKernel 的手写 Group ABI 规范化。
Group 中先调用 AIC 成员并接收共享 InOut 的新 SSA 返回值，再把它传给 AIV 成员时，
adapter 只映射原 Group 形参，没有把前一成员的完整参数写回别名映射到统一 ABI。
克隆后的 AIV 成员引用 Group 局部 SSA 值，产生未绑定的 free variable。

普通调用以及 tuple 返回 + pl.submit + pl.no_dep 都能触发。
这是合法参数写回的编译正确性问题，与 while 活跃性 #3009 / PR #3013 不同，
也不依赖大于 32 个 Tensor 的扩容。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p1_group_inout.py，运行：

```bash
python -m pytest -q -o addopts='' test_p1_group_inout.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(program: ir.Program) -> ir.Program:
    """Run SSA -> infer-memory -> expand-mixed-kernel.

    InjectGMPipeBuffer is intentionally excluded so this file's Expecteds
    pin only what ExpandMixedKernel produces.
    """
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(program)))

def test_hand_written_group_rebinds_shared_inout_results():
    """A preceding member's writeback must bind to a parameter in the next member."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            return sink, out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            sink = self.cube_side(sink)
            sink, out = self.vec_side(sink, out)
            return sink, out

    after = _run_pipeline(Program)
    cube = after.get_function("cube_side")
    vec = after.get_function("vec_side")
    group = after.get_function("mixed")
    assert cube is not None and vec is not None and group is not None
    for member, indices in ((cube, [1]), (vec, [1, 0])):
        returns = [stmt for stmt in ir.flatten_to_stmts(member.body) if isinstance(stmt, ir.ReturnStmt)]
        assert len(returns) == 1
        expected = ir.ReturnStmt([member.params[i] for i in indices], returns[0].span)
        ir.assert_structural_equal(returns[0], expected)
        assert "FREE_VAR" not in member.as_python()
    calls = [
        stmt.value
        for stmt in ir.flatten_to_stmts(group.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    ]
    assert len(calls) == 2
    assert all(list(call.args) == list(group.params) for call in calls)

def test_hand_written_group_submit_rebinds_tuple_writeback_no_dep():
    """Tuple/Submit lineage preserves both buffer identity and the no_dep position."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            second: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            return second, first

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            second: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            (second, first), _cube_tid = pl.submit(self.cube_side, first, second)
            second, _vec_tid = pl.submit(self.vec_side, pl.no_dep(second))
            return second

    after = _run_pipeline(Program)
    vec = after.get_function("vec_side")
    group = after.get_function("mixed")
    assert vec is not None and group is not None
    returns = [stmt for stmt in ir.flatten_to_stmts(vec.body) if isinstance(stmt, ir.ReturnStmt)]
    assert len(returns) == 1
    ir.assert_structural_equal(returns[0], ir.ReturnStmt([vec.params[1]], returns[0].span))
    submits = [
        stmt.value
        for stmt in ir.flatten_to_stmts(group.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Submit)
    ]
    vec_submit = next(submit for submit in submits if submit.op.name == "vec_side")
    assert list(vec_submit.args) == list(group.params)
    assert vec_submit.attrs["arg_direction_overrides"] == [1]
    assert "FREE_VAR" not in vec.as_python()
```

### Expected Behavior

两个成员都只引用自己规范化后的形参；共享 InOut 返回值映射回对应 Group 参数。tuple 的返回次序和 no_dep 的参数位置必须同时保留。不得产生 FREE_VAR。

### Actual Behavior

普通调用下 AIV 的返回值/参数别名仍指向前一个调用的 sink SSA；Submit/tuple 场景同样保留 Group 局部别名。完整 CSA 曾在 progress__ssa_v1 的未定义引用处失败。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/ir/transforms/expand_mixed_kernel_pass.cpp:
BuildGroupAbiAdapter / GroupMemberCallRewriter / NormalizeHandWrittenGroupAbis。
本地候选复用 return_lineage 的完整参数写回追踪，一次性建立 alias→Group parameter 映射，
并用于成员参数重写和 no_dep 下标映射。不能把 slice/view 或新 buffer 冒充完整参数别名。

验证边界：

附带的两个最小回归在本地候选上通过。已有 5 参数 JIT 小例子还验证了
24 组 MIX、双 AIV、当前 stream、连续三次执行和保护区；这不是完整 CSA 设备验证。
代码补丁仍保留本地，本 issue 不附带 PR，不声称候选已通过全部上游 CI。

Related: #2003（另一条返回 lineage / orchestration 路径，不是本次 Group ABI adapter 的缺失映射）、#3009、#3013。

### 2026-10-16 补充：上游原版与本地候选对照

在独立 worktree 自行构建 PR #3013 的精确 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c`，源码未打本地补丁。
本 issue 的 2 个 CPU IR/MLIR 回归全部按预期失败；相同复现文件在本地候选中全部通过。
本轮四类问题合计 5 个用例：上游原版 5 failed，本地候选 5 passed。
这确认 #3013 修复 while 条件活跃性后，本文问题仍然存在；不是沿用旧日志推断。
该对照仅证明上述 IR/代码生成边界，不代表完整 CSA 真机、精度或性能验证。

<!-- SOURCE_END history-document-032 -->

---

<a id="history-document-033"></a>
### 历史原文 033：upstream_issues/p1_group_inout_initial.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p1_group_inout_initial.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p1_group_inout_initial.md)；原文 8,172 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-033 -->
### Component

Other (please specify in description)

### Description

组件为 ExpandMixedKernel 的手写 Group ABI 规范化。
Group 中先调用 AIC 成员并接收共享 InOut 的新 SSA 返回值，再把它传给 AIV 成员时，
adapter 只映射原 Group 形参，没有把前一成员的完整参数写回别名映射到统一 ABI。
克隆后的 AIV 成员引用 Group 局部 SSA 值，产生未绑定的 free variable。

普通调用以及 tuple 返回 + pl.submit + pl.no_dep 都能触发。
这是合法参数写回的编译正确性问题，与 while 活跃性 #3009 / PR #3013 不同，
也不依赖大于 32 个 Tensor 的扩容。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p1_group_inout.py，运行：

```bash
python -m pytest -q -o addopts='' test_p1_group_inout.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(program: ir.Program) -> ir.Program:
    """Run SSA -> infer-memory -> expand-mixed-kernel.

    InjectGMPipeBuffer is intentionally excluded so this file's Expecteds
    pin only what ExpandMixedKernel produces.
    """
    return passes.expand_mixed_kernel()(passes.infer_tile_memory_space()(passes.convert_to_ssa()(program)))

def test_hand_written_group_rebinds_shared_inout_results():
    """A preceding member's writeback must bind to a parameter in the next member."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            return sink, out

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            sink = self.cube_side(sink)
            sink, out = self.vec_side(sink, out)
            return sink, out

    after = _run_pipeline(Program)
    cube = after.get_function("cube_side")
    vec = after.get_function("vec_side")
    group = after.get_function("mixed")
    assert cube is not None and vec is not None and group is not None
    for member, indices in ((cube, [1]), (vec, [1, 0])):
        returns = [stmt for stmt in ir.flatten_to_stmts(member.body) if isinstance(stmt, ir.ReturnStmt)]
        assert len(returns) == 1
        expected = ir.ReturnStmt([member.params[i] for i in indices], returns[0].span)
        ir.assert_structural_equal(returns[0], expected)
        assert "FREE_VAR" not in member.as_python()
    calls = [
        stmt.value
        for stmt in ir.flatten_to_stmts(group.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call)
    ]
    assert len(calls) == 2
    assert all(list(call.args) == list(group.params) for call in calls)

def test_hand_written_group_submit_rebinds_tuple_writeback_no_dep():
    """Tuple/Submit lineage preserves both buffer identity and the no_dep position."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_side(
            self,
            first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            second: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> tuple[pl.Tensor[[16, 16], pl.FP32], pl.Tensor[[16, 16], pl.FP32]]:
            return second, first

        @pl.function(type=pl.FunctionType.AIV)
        def vec_side(
            self,
            sink: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return sink

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            first: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            second: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            (second, first), _cube_tid = pl.submit(self.cube_side, first, second)
            second, _vec_tid = pl.submit(self.vec_side, pl.no_dep(second))
            return second

    after = _run_pipeline(Program)
    vec = after.get_function("vec_side")
    group = after.get_function("mixed")
    assert vec is not None and group is not None
    returns = [stmt for stmt in ir.flatten_to_stmts(vec.body) if isinstance(stmt, ir.ReturnStmt)]
    assert len(returns) == 1
    ir.assert_structural_equal(returns[0], ir.ReturnStmt([vec.params[1]], returns[0].span))
    submits = [
        stmt.value
        for stmt in ir.flatten_to_stmts(group.body)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Submit)
    ]
    vec_submit = next(submit for submit in submits if submit.op.name == "vec_side")
    assert list(vec_submit.args) == list(group.params)
    assert vec_submit.attrs["arg_direction_overrides"] == [1]
    assert "FREE_VAR" not in vec.as_python()
```

### Expected Behavior

两个成员都只引用自己规范化后的形参；共享 InOut 返回值映射回对应 Group 参数。tuple 的返回次序和 no_dep 的参数位置必须同时保留。不得产生 FREE_VAR。

### Actual Behavior

普通调用下 AIV 的返回值/参数别名仍指向前一个调用的 sink SSA；Submit/tuple 场景同样保留 Group 局部别名。完整 CSA 曾在 progress__ssa_v1 的未定义引用处失败。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/ir/transforms/expand_mixed_kernel_pass.cpp:
BuildGroupAbiAdapter / GroupMemberCallRewriter / NormalizeHandWrittenGroupAbis。
本地候选复用 return_lineage 的完整参数写回追踪，一次性建立 alias→Group parameter 映射，
并用于成员参数重写和 no_dep 下标映射。不能把 slice/view 或新 buffer 冒充完整参数别名。

验证边界：

附带的两个最小回归在本地候选上通过。已有 5 参数 JIT 小例子还验证了
24 组 MIX、双 AIV、当前 stream、连续三次执行和保护区；这不是完整 CSA 设备验证。
代码补丁仍保留本地，本 issue 不附带 PR，不声称候选已通过全部上游 CI。

Related: #2003（另一条返回 lineage / orchestration 路径，不是本次 Group ABI adapter 的缺失映射）、#3009、#3013。

<!-- SOURCE_END history-document-033 -->

---

<a id="history-document-034"></a>
### 历史原文 034：upstream_issues/p4_ptoas_branch_view.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p4_ptoas_branch_view.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p4_ptoas_branch_view.md)；原文 6,769 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-034 -->
### Component

Codegen

### Description

在 memory_planner=PTOAS / emit_tile_addr=False 下，分支内对列向量作
reshape→add→reshape，分支外再对 phi 结果 reshape。当前 view no-op 判断只比较
预声明目标句柄的类型与结果类型。类型相同并不说明旧 MemRef 代表句柄就是当前 phi。
分支外消费可能错误地选到只在分支内定义的 producer，导致非法 SSA dominance。
transpose_view 使用同类 no-op 判断，也需要同一约束。

这是生成 IR 的合法性问题；PTOAS 对该 IR 的拒绝是合理的，不要求 PTOAS 放宽验证。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p4_ptoas_branch_view.py，运行：

```bash
python -m pytest -q -o addopts='' test_p4_ptoas_branch_view.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_reshape_reads_phi_handle_after_branch():
    """A post-if view cannot reuse a same-typed but branch-local old handle."""

    @pl.program
    class BranchView:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[32, 1], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            base = pl.load(x, [0, 0], [32, 1])
            if flag != 0:
                row = pl.reshape(base, [1, 32])
                updated = pl.add(row, 1.0)
                column = pl.reshape(updated, [32, 1])
                result = column
            else:
                result = base
            after = pl.reshape(result, [1, 32])
            head = pl.tile.extract(after, 0, 0, [1, 16], target_memory=pl.Mem.Vec)
            output = pl.store(head, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, BranchView)
    mlir = _codegen(optimized, emit_tile_addr=False)
    extract = next(line for line in mlir.splitlines() if "pto.textract ins(" in line)
    source = extract.split("ins(", 1)[1].split(",", 1)[0]
    view = next((line.strip() for line in mlir.splitlines() if line.strip().startswith(source + " =")), "")
    assert "= pto.treshape " in view, mlir
    phi = view.split("= pto.treshape ", 1)[1].split()[0]
    assert any("pto.tmov" in line and "outs(" + phi + " :" in line for line in mlir.splitlines()), mlir
```

### Expected Behavior

分支后的 view 必须从实际 phi/源句柄派生。没有固化物理地址时，不能仅凭同类型省略 view。后续 extract 应读取由 phi 派生的 treshape，其源定义支配使用点。

### Actual Behavior

真实 CSA 的 PTOAS 路径曾产生两处分支外引用分支内旧句柄的问题。最小回归检查 extract 的源确实由 phi 的 treshape 产生，而不是只检查能够打印 MLIR。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/backend/common/pto_ops_datamove.cpp 的 tile.reshape / tile.transpose_view。
本地候选仅在固定地址模式或实际 source handle 与已绑定 target 相同时执行原 no-op，
否则从实际 source 发出 view。此处不修改内存容量、算术、Tile 或流水深度。

验证边界：

附带最小 CPU 回归在本地候选上通过，原完整 CSA 的两处 SSA 合法性错误也已越过。
本问题来自 PTOAS 备用编译路径；当前完整 CSA 默认 PYPTO 路径不依赖此项。
完整 PTOAS 路径仍存在独立的 Mat 容量问题，不能将本项修复记为整层编译或真机通过。

Related: #1642（固定地址 view 丢失依赖链，本例为无地址模式下分支后的 SSA 句柄）、#2693。

### 2026-10-16 补充：上游原版与本地候选对照

在独立 worktree 自行构建 PR #3013 的精确 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c`，源码未打本地补丁。
本 issue 的 1 个 CPU IR/MLIR 回归全部按预期失败；相同复现文件在本地候选中全部通过。
本轮四类问题合计 5 个用例：上游原版 5 failed，本地候选 5 passed。
这确认 #3013 修复 while 条件活跃性后，本文问题仍然存在；不是沿用旧日志推断。
该对照仅证明上述 IR/代码生成边界，不代表完整 CSA 真机、精度或性能验证。

<!-- SOURCE_END history-document-034 -->

---

<a id="history-document-035"></a>
### 历史原文 035：upstream_issues/p4_ptoas_branch_view_initial.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p4_ptoas_branch_view_initial.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p4_ptoas_branch_view_initial.md)；原文 6,167 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-035 -->
### Component

Codegen

### Description

在 memory_planner=PTOAS / emit_tile_addr=False 下，分支内对列向量作
reshape→add→reshape，分支外再对 phi 结果 reshape。当前 view no-op 判断只比较
预声明目标句柄的类型与结果类型。类型相同并不说明旧 MemRef 代表句柄就是当前 phi。
分支外消费可能错误地选到只在分支内定义的 producer，导致非法 SSA dominance。
transpose_view 使用同类 no-op 判断，也需要同一约束。

这是生成 IR 的合法性问题；PTOAS 对该 IR 的拒绝是合理的，不要求 PTOAS 放宽验证。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p4_ptoas_branch_view.py，运行：

```bash
python -m pytest -q -o addopts='' test_p4_ptoas_branch_view.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_reshape_reads_phi_handle_after_branch():
    """A post-if view cannot reuse a same-typed but branch-local old handle."""

    @pl.program
    class BranchView:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[32, 1], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            base = pl.load(x, [0, 0], [32, 1])
            if flag != 0:
                row = pl.reshape(base, [1, 32])
                updated = pl.add(row, 1.0)
                column = pl.reshape(updated, [32, 1])
                result = column
            else:
                result = base
            after = pl.reshape(result, [1, 32])
            head = pl.tile.extract(after, 0, 0, [1, 16], target_memory=pl.Mem.Vec)
            output = pl.store(head, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, BranchView)
    mlir = _codegen(optimized, emit_tile_addr=False)
    extract = next(line for line in mlir.splitlines() if "pto.textract ins(" in line)
    source = extract.split("ins(", 1)[1].split(",", 1)[0]
    view = next((line.strip() for line in mlir.splitlines() if line.strip().startswith(source + " =")), "")
    assert "= pto.treshape " in view, mlir
    phi = view.split("= pto.treshape ", 1)[1].split()[0]
    assert any("pto.tmov" in line and "outs(" + phi + " :" in line for line in mlir.splitlines()), mlir
```

### Expected Behavior

分支后的 view 必须从实际 phi/源句柄派生。没有固化物理地址时，不能仅凭同类型省略 view。后续 extract 应读取由 phi 派生的 treshape，其源定义支配使用点。

### Actual Behavior

真实 CSA 的 PTOAS 路径曾产生两处分支外引用分支内旧句柄的问题。最小回归检查 extract 的源确实由 phi 的 treshape 产生，而不是只检查能够打印 MLIR。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/backend/common/pto_ops_datamove.cpp 的 tile.reshape / tile.transpose_view。
本地候选仅在固定地址模式或实际 source handle 与已绑定 target 相同时执行原 no-op，
否则从实际 source 发出 view。此处不修改内存容量、算术、Tile 或流水深度。

验证边界：

附带最小 CPU 回归在本地候选上通过，原完整 CSA 的两处 SSA 合法性错误也已越过。
本问题来自 PTOAS 备用编译路径；当前完整 CSA 默认 PYPTO 路径不依赖此项。
完整 PTOAS 路径仍存在独立的 Mat 容量问题，不能将本项修复记为整层编译或真机通过。

Related: #1642（固定地址 view 丢失依赖链，本例为无地址模式下分支后的 SSA 句柄）、#2693。

<!-- SOURCE_END history-document-035 -->

---

<a id="history-document-036"></a>
### 历史原文 036：upstream_issues/p5_ptoas_gather_phi.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p5_ptoas_gather_phi.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p5_ptoas_gather_phi.md)；原文 7,213 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-036 -->
### Component

Codegen

### Description

条件两分支均对同一个 Mat tile 执行原地 gather_row，返回的 phi 理应仍指向该存储。
连续两次这种条件更新、后接 transpose_view 和 matmul 时，PTOAS 路径会给 phi 引入新的
目标句柄并产生 Mat→Mat tmov。A2/A3 不支持这种地址空间对，PTOAS 正确拒绝编译。

触发点包含两个相连的边界：YieldFixup 在 PTOAS 模式完全跳过 if-phi 同窗口归一，
以及共享存储存在多种 view 类型时，codegen 一律拒绝复用已声明、且恰好同类型的句柄。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p5_ptoas_gather_phi.py，运行：

```bash
python -m pytest -q -o addopts='' test_p5_ptoas_gather_phi.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_conditional_gather_preserves_one_mat_buffer():
    """Two in-place branch writes must not introduce an illegal Mat->Mat copy."""

    @pl.program
    class ConditionalGather:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            base = pl.tile.create([32, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
            if flag != 0:
                first = pl.tile.gather_row(base, x, [0, 0], [0, 0], [16, 64])
            else:
                first = pl.tile.gather_row(base, x, [0, 0], [16, 0], [16, 64])
            if flag != 0:
                second = pl.tile.gather_row(first, x, [16, 0], [32, 0], [16, 64])
            else:
                second = pl.tile.gather_row(first, x, [16, 0], [48, 0], [16, 64])
            transposed = pl.tile.transpose_view(second)
            rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
            left = pl.tile.move(transposed, target_memory=pl.Mem.Left)
            right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
            product = pl.tile.matmul(left, right)
            output = pl.tile.store(product, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ConditionalGather)
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert not any(line.count("loc=mat") == 2 for line in moves), mlir
    bases = {
        line.split("pto.subview ", 1)[1].split("[", 1)[0]
        for line in mlir.splitlines()
        if "pto.subview " in line
    }
    assert len(bases) == 1, mlir
```

### Expected Behavior

两分支原本就在同一物理窗口原地更新时，phi 继续使用该存储。后续 transpose 和 matmul 必须读到此窗口，不生成非法 Mat→Mat 拷贝，也不能只删除拷贝而让消费者读未写入的新 buffer。

### Actual Behavior

完整 CSA 在修复分支 view 问题后报错：
`'pto.tmov' op expects a supported tmov address-space pair for this target`。
附带最小回归保留连续条件 gather、transpose 和 matmul，检查没有 Mat→Mat move，
且两轮 gather 的 subview 共享同一个根句柄。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/ir/transforms/memory_reuse_pass.cpp 的 YieldFixupMutator，
以及 src/codegen/pto/pto_codegen.cpp / pto_control_flow_codegen.cpp 的共享句柄选择。
本地候选只对已同窗口的两个分支同步 phi；不同窗口不得强行合并。
多类型存储仅复用已经声明且类型完全匹配的 handle，不能修改一个 SSA handle 的类型。

验证边界：

附带最小 CPU 回归在本地候选上通过，完整 CSA 已越过该 Mat→Mat 错误。
本问题属于 PTOAS 备用路径的正确性 bug，不是当前 PYPTO 路径的性能优化。
尚未用完整 CSA 真机验证该 PTOAS 分支，不声明完整算子通过。

Related: #1956、#2693；#2902 是 AutoTileMatmulL0 对已有 Mat 操作数额外搬运，本例是条件 phi / 多类型共享句柄，根因不同。

### 2026-10-16 补充：上游原版与本地候选对照

在独立 worktree 自行构建 PR #3013 的精确 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c`，源码未打本地补丁。
本 issue 的 1 个 CPU IR/MLIR 回归全部按预期失败；相同复现文件在本地候选中全部通过。
本轮四类问题合计 5 个用例：上游原版 5 failed，本地候选 5 passed。
这确认 #3013 修复 while 条件活跃性后，本文问题仍然存在；不是沿用旧日志推断。
该对照仅证明上述 IR/代码生成边界，不代表完整 CSA 真机、精度或性能验证。

<!-- SOURCE_END history-document-036 -->

---

<a id="history-document-037"></a>
### 历史原文 037：upstream_issues/p5_ptoas_gather_phi_initial.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p5_ptoas_gather_phi_initial.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p5_ptoas_gather_phi_initial.md)；原文 6,611 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-037 -->
### Component

Codegen

### Description

条件两分支均对同一个 Mat tile 执行原地 gather_row，返回的 phi 理应仍指向该存储。
连续两次这种条件更新、后接 transpose_view 和 matmul 时，PTOAS 路径会给 phi 引入新的
目标句柄并产生 Mat→Mat tmov。A2/A3 不支持这种地址空间对，PTOAS 正确拒绝编译。

触发点包含两个相连的边界：YieldFixup 在 PTOAS 模式完全跳过 if-phi 同窗口归一，
以及共享存储存在多种 view 类型时，codegen 一律拒绝复用已声明、且恰好同类型的句柄。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p5_ptoas_gather_phi.py，运行：

```bash
python -m pytest -q -o addopts='' test_p5_ptoas_gather_phi.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_conditional_gather_preserves_one_mat_buffer():
    """Two in-place branch writes must not introduce an illegal Mat->Mat copy."""

    @pl.program
    class ConditionalGather:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP16],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[64, 16], pl.FP32]],
        ) -> pl.Tensor[[64, 16], pl.FP32]:
            base = pl.tile.create([32, 64], dtype=pl.FP16, target_memory=pl.Mem.Mat)
            if flag != 0:
                first = pl.tile.gather_row(base, x, [0, 0], [0, 0], [16, 64])
            else:
                first = pl.tile.gather_row(base, x, [0, 0], [16, 0], [16, 64])
            if flag != 0:
                second = pl.tile.gather_row(first, x, [16, 0], [32, 0], [16, 64])
            else:
                second = pl.tile.gather_row(first, x, [16, 0], [48, 0], [16, 64])
            transposed = pl.tile.transpose_view(second)
            rhs = pl.tile.load(x, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
            left = pl.tile.move(transposed, target_memory=pl.Mem.Left)
            right = pl.tile.move(rhs, target_memory=pl.Mem.Right)
            product = pl.tile.matmul(left, right)
            output = pl.tile.store(product, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ConditionalGather)
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert not any(line.count("loc=mat") == 2 for line in moves), mlir
    bases = {
        line.split("pto.subview ", 1)[1].split("[", 1)[0]
        for line in mlir.splitlines()
        if "pto.subview " in line
    }
    assert len(bases) == 1, mlir
```

### Expected Behavior

两分支原本就在同一物理窗口原地更新时，phi 继续使用该存储。后续 transpose 和 matmul 必须读到此窗口，不生成非法 Mat→Mat 拷贝，也不能只删除拷贝而让消费者读未写入的新 buffer。

### Actual Behavior

完整 CSA 在修复分支 view 问题后报错：
`'pto.tmov' op expects a supported tmov address-space pair for this target`。
附带最小回归保留连续条件 gather、transpose 和 matmul，检查没有 Mat→Mat move，
且两轮 gather 的 subview 共享同一个根句柄。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/ir/transforms/memory_reuse_pass.cpp 的 YieldFixupMutator，
以及 src/codegen/pto/pto_codegen.cpp / pto_control_flow_codegen.cpp 的共享句柄选择。
本地候选只对已同窗口的两个分支同步 phi；不同窗口不得强行合并。
多类型存储仅复用已经声明且类型完全匹配的 handle，不能修改一个 SSA handle 的类型。

验证边界：

附带最小 CPU 回归在本地候选上通过，完整 CSA 已越过该 Mat→Mat 错误。
本问题属于 PTOAS 备用路径的正确性 bug，不是当前 PYPTO 路径的性能优化。
尚未用完整 CSA 真机验证该 PTOAS 分支，不声明完整算子通过。

Related: #1956、#2693；#2902 是 AutoTileMatmulL0 对已有 Mat 操作数额外搬运，本例是条件 phi / 多类型共享句柄，根因不同。

<!-- SOURCE_END history-document-037 -->

---

<a id="history-document-038"></a>
### 历史原文 038：upstream_issues/p7_ptoas_column_carry.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p7_ptoas_column_carry.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p7_ptoas_column_carry.md)；原文 6,349 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-038 -->
### Component

Codegen

### Description

以 [1,32] 行向量 full 初始化，经 reshape 得到 [32,1] 列向量作为循环 carry。
循环内包含条件更新，PTOAS 路径的 carry 逻辑类型为列向量，但其底层已声明 handle 仍是行向量。
tile.move codegen 直接向底层行 handle 发出列→行 tmov，源和目标形状不一致，PTOAS 拒绝编译。

这不是要求硬件支持新搬运方式，而是 codegen 应对同一存储使用正确的目标 view。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p7_ptoas_column_carry.py，运行：

```bash
python -m pytest -q -o addopts='' test_p7_ptoas_column_carry.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_column_carry_move_uses_destination_view():
    """A column carry can share storage with a row-major full initializer."""

    @pl.program
    class ColumnCarry:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
        ) -> pl.Tensor[[32, 1], pl.FP32]:
            row_base = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            base = pl.tile.reshape(row_base, [32, 1])
            for i, (carry,) in pl.range(2, init_values=(base,)):
                if flag != 0:
                    row = pl.reshape(carry, [1, 32])
                    updated = pl.add(row, 1.0)
                    column = pl.reshape(updated, [32, 1])
                    value = column
                else:
                    value = carry
                result = pl.yield_(value)
            output = pl.store(result, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ColumnCarry)
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert moves, mlir
    for move in moves:
        shapes = re.findall(r"rows=(\d+), cols=(\d+)", move)
        assert len(shapes) == 2 and shapes[0] == shapes[1], move
```

### Expected Behavior

写回 carry 时先在现有存储上建立匹配 [32,1] 的目标 view，然后发出源/目标形状匹配的 tmov。不新增无意义 allocation，不改变循环结果。

### Actual Behavior

完整 CSA 报错：
`'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes`。
最小回归检查每条 tmov 的实际源/目标 tile_buf rows/cols，要求均匹配。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/backend/common/pto_ops_elementwise.cpp 的 tile.move PTOAS 分支。
本地候选对比逻辑 destination type 与底层 handle type，必要时建立同存储的静态
treshape view，再按实际 view 类型发出搬运；保留静态有效形状约束。

验证边界：

附带最小 CPU 回归在本地候选上通过。完整 CSA 已越过此类 MLIR 合法性错误，
随后遇到独立的 PTOAS Mat 容量限制；因此没有宣称完整 PTOAS 编译或设备通过。
当前默认 PYPTO 路径不依赖本项。

Related: #1649（sub-column spill 的 valid_shape 问题，与这里行/列 carry 的底层 handle 形状不同）、#2693。

### 2026-10-16 补充：上游原版与本地候选对照

在独立 worktree 自行构建 PR #3013 的精确 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c`，源码未打本地补丁。
本 issue 的 1 个 CPU IR/MLIR 回归全部按预期失败；相同复现文件在本地候选中全部通过。
本轮四类问题合计 5 个用例：上游原版 5 failed，本地候选 5 passed。
这确认 #3013 修复 while 条件活跃性后，本文问题仍然存在；不是沿用旧日志推断。
该对照仅证明上述 IR/代码生成边界，不代表完整 CSA 真机、精度或性能验证。

<!-- SOURCE_END history-document-038 -->

---

<a id="history-document-039"></a>
### 历史原文 039：upstream_issues/p7_ptoas_column_carry_initial.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p7_ptoas_column_carry_initial.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_issues/p7_ptoas_column_carry_initial.md)；原文 5,747 字节。本会话提交的 issue 正文或初稿；全部全文保留。

<!-- SOURCE_BEGIN history-document-039 -->
### Component

Codegen

### Description

以 [1,32] 行向量 full 初始化，经 reshape 得到 [32,1] 列向量作为循环 carry。
循环内包含条件更新，PTOAS 路径的 carry 逻辑类型为列向量，但其底层已声明 handle 仍是行向量。
tile.move codegen 直接向底层行 handle 发出列→行 tmov，源和目标形状不一致，PTOAS 拒绝编译。

这不是要求硬件支持新搬运方式，而是 codegen 应对同一存储使用正确的目标 view。

### Steps to Reproduce

在已构建的 PyPTO 开发环境中，将下面完整代码保存为 test_p7_ptoas_column_carry.py，运行：

```bash
python -m pytest -q -o addopts='' test_p7_ptoas_column_carry.py
```

CPU 编译/IR 检查即可，无需模型权重或 NPU。

```python
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re
import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pytest.fixture(autouse=True)
def ascend_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_pipeline(
    memory_planner: passes.MemoryPlanner, program: ir.Program
) -> tuple[ir.Program, list[str]]:
    """Run the Default pipeline under a PassContext with the given planner.

    Returns the optimized program and the concrete list of executed pass names.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=memory_planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    return optimized, list(pm.pass_names)

def _codegen(optimized: ir.Program, *, emit_tile_addr: bool) -> str:
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)

def test_ptoas_column_carry_move_uses_destination_view():
    """A column carry can share storage with a row-major full initializer."""

    @pl.program
    class ColumnCarry:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
        ) -> pl.Tensor[[32, 1], pl.FP32]:
            row_base = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
            base = pl.tile.reshape(row_base, [32, 1])
            for i, (carry,) in pl.range(2, init_values=(base,)):
                if flag != 0:
                    row = pl.reshape(carry, [1, 32])
                    updated = pl.add(row, 1.0)
                    column = pl.reshape(updated, [32, 1])
                    value = column
                else:
                    value = carry
                result = pl.yield_(value)
            output = pl.store(result, [0, 0], out)
            return output

    optimized, _ = _run_pipeline(passes.MemoryPlanner.PTOAS, ColumnCarry)
    mlir = _codegen(optimized, emit_tile_addr=False)
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert moves, mlir
    for move in moves:
        shapes = re.findall(r"rows=(\d+), cols=(\d+)", move)
        assert len(shapes) == 2 and shapes[0] == shapes[1], move
```

### Expected Behavior

写回 carry 时先在现有存储上建立匹配 [32,1] 的目标 view，然后发出源/目标形状匹配的 tmov。不新增无意义 allocation，不改变循环结果。

### Actual Behavior

完整 CSA 报错：
`'pto.tmov' op expects A2/A3 non-mat tmov to use matching src/dst shapes`。
最小回归检查每条 tmov 的实际源/目标 tile_buf rows/cols，要求均匹配。

### Git Commit ID

最初发现于调试分支基线 `cb24844712fa4329563a8cfbdd8d9f77ba0dc20c`。
另已核对上游 PR #3013 head `72b7ffb98ea3cee12bf7bbf533ab89b848a8980c` 中对应逻辑仍在；
该 PR 的 while-condition 修复不覆盖本 issue。上游 head 的独立构建与复现结果另行记录。

### NPU Kind

Ascend 910B

本例选择 BackendType.Ascend910B，即 A2/A3 编译后端；复现本身不执行设备。

### Host Platform

Linux (aarch64)

### Additional Context

定位及本地候选方向：

src/backend/common/pto_ops_elementwise.cpp 的 tile.move PTOAS 分支。
本地候选对比逻辑 destination type 与底层 handle type，必要时建立同存储的静态
treshape view，再按实际 view 类型发出搬运；保留静态有效形状约束。

验证边界：

附带最小 CPU 回归在本地候选上通过。完整 CSA 已越过此类 MLIR 合法性错误，
随后遇到独立的 PTOAS Mat 容量限制；因此没有宣称完整 PTOAS 编译或设备通过。
当前默认 PYPTO 路径不依赖本项。

Related: #1649（sub-column spill 的 valid_shape 问题，与这里行/列 carry 的底层 handle 形状不同）、#2693。

<!-- SOURCE_END history-document-039 -->

---

<a id="dynamic-ledger"></a>
## 动态区间入口总账与七档结果

- [040 · csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md](#history-document-040)
- [041 · csa_multishape_spmd_20261016/README.md](#history-document-041)
- [042 · csa_multishape_spmd_20261016/ATTEMPTS.md](#history-document-042)
- [043 · csa_multishape_spmd_20261016/SELECTED_SEVEN_RESULTS.md](#history-document-043)
- [044 · csa_multishape_spmd_20261016/BALANCED_SEVEN_RESULTS.md](#history-document-044)
- [045 · csa_multishape_spmd_20261016/CACHED_SEVEN_RESULTS.md](#history-document-045)
- [046 · csa_multishape_spmd_20261016/QK_SEVEN_RESULTS.md](#history-document-046)

<a id="history-document-040"></a>
### 历史原文 040：GOAL_AND_CONSTRAINTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md)；原文 3,002 字节。动态目标、过程总账或七档结果，全文保留。

<!-- SOURCE_BEGIN history-document-040 -->
# CSA 单 SPMD：当前目标与验收约束

用户于 2026-10-17 明确确认：**降低至少 30%**，以 `SPMD ≤ 动态分 task PTO × 0.70` 逐档验收。
本目标替换此前 128K/B16 <700 μs 单点门槛。原始目标文件保留在 GOAL_USER_ORIGINAL_20261017.txt。

目标：将 CSA 的所有 incore task 融合为一个完整的 SPMD 算子，支持动态输入形状和按工作量分档，并使每个性能验收档位相较原动态分 task PTO 的完整耗时降低至少 30%。

一、动态形状与分档
1. 根据 CSA 自身的 batch_size、历史 seqlen、本轮 query 数及总 token 数，选择合适的分块、分核、历史切分、流水和合并策略。
2. 分档应覆盖输入区间，不能把几个测试点写成支持白名单。HC_pre 仅作为设计思路参考，不照搬其阈值或专用分支。
3. 档内继续根据实际 device metadata 计算有效范围、地址、循环次数和尾块，不能固化测试输入。
4. 允许不同形状分别 JIT 编译、缓存不同二进制，不要求一份二进制覆盖多个 batch。
5. 明确支持范围和策略边界；精确形状特化必须有 CSA 自身的依据及实测收益。

二、性能目标
逐档满足：单 SPMD 完整耗时 ≤ 同轮原动态分 task PTO 完整耗时 × 0.70

起始回归规格均为每请求本轮 query 数 S=6：
- 历史 128K：B4、B8、B16、B24。
- 历史 8K：B16、B24、B32。

七个规格必须各自达标，不能用平均收益替代。它们只是回归点；最终各策略区间还须选择代表值和边界值验证正确性及性能，不能只优化孤立样例。

三、比较与验收口径
1. 使用正式权重 /data/model/DeepSeek-V4-Flash-0731-w8a8。
2. 基线为原动态分 task PTO，比较完整 CSA 算子；不以 Native、子阶段或整网其他开销替代。
3. 计时采用完整 L1 incore 最早开始至最晚结束，包含内部等待和阶段间空隙，排除外围 AICPU/orchestrator 首尾开销。
4. 两侧使用相同输入、权重、设备配置和测量条件，以同轮多次采样的中位数计算收益。历史数据用于参考，最终达标需复核。
5. CPU 成本模型、插桩计时和单次最小值不能作为性能达标证据。

四、实施与交付
先保证功能，再验证精度，最后优化性能。遵守既有精度要求，检查输出、Top-K、cache/state、动态 metadata、图重放和核间同步正确性，不通过跳过计算或放宽阈值获得性能收益。
保留原静态版本及既有结果，持续记录实现、尝试、失败原因、测试证据和逐档对比。完成可维护的区间策略入口，明确尚未支持或尚未验证的范围。只有动态形状要求及所有性能验收档位均完成，才将目标标记为完成。

当前实现仍有 S=6、核体支持范围及尚未验证区间的限制；这些是待处理事项，不是对用户目标的缩减。CPU 绑定、lowering、真机精度和正式性能证据分别记录。

<!-- SOURCE_END history-document-040 -->

---

<a id="history-document-041"></a>
### 历史原文 041：README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/README.md)；原文 16,673 字节。动态目标、过程总账或七档结果，全文保留。

<!-- SOURCE_BEGIN history-document-041 -->
# CSA 单 SPMD：按形状选择编译与调度策略

**当前目标（2026-10-17 更新）：七档各自降低至少 30%，详见 [目标约束](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md)。原 700 μs 目标及下文历史记录保留，不再作为当前完成标准。**

用户补充的目标约束见 [GOAL_AND_CONSTRAINTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md)：按 CSA 自身
工作量划分区间，档内动态处理输入；七个规格是回归点，不是支持白名单。HC_pre 只作为
分档思路参考，不照搬其阈值或专用分支。最新区间实现见 [RANGE_POLICY.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/RANGE_POLICY.md)。

**当前选择（2026-10-17 13:22）：** 七档见
[SELECTED_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/SELECTED_SEVEN_RESULTS.md)，128K/B16完整无插桩L1
现选首轮中位807.16μs，最新同进程现选807.50μs；七档均尚未达到新的30%门槛。
不同batch及长度策略允许分别编译、分别调优。


三处RoPE逐行gather合并的候选完成全门槛PASS，但完整L1中位815.10μs，未优于现选，
未采用、不扩档，见[RoPE完整记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/ROPE_GATHER_EXPERIMENT.md)。固定分核MILP三轮CPU搜索
没有找到更好安排，没有上卡，见[模型记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/FIXED_CORE_MILP_EXPERIMENT.md)。

全阶段noinline与现选100tick等待组合的B16诊断已完整PASS，L1中位807.02μs，
与现选持平，未采用、不扩档；该诊断不计正式700目标。全部任务已终态。
见[组合诊断记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/OUTLINED_CACHED_WAIT_EXPERIMENT.md)。当前队列以PENDING_JOBS.json为准，
以下历史段落中的“全部任务终态”仅指相应记录当时。

本轮将QK分工候选的softmax/PV/发布分块分别设为32/16/8，CPU Vec分配167552B，
完整设备精度、图重放、30代和保护区PASS。L1三窗830.46/827.18/832.04μs，中位830.46；
同轮动态911.08μs。仍未优于现选，未采用、不扩档，见[分工及分块记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_ROLE_SPLIT_EXPERIMENT.md)。
固定分核的局部交换和主Q后处理成组后移均未改善CPU模型，没有上卡，
见[局部排程记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md)。全部任务终态，700μs目标仍未达到。

本轮B16 QK提前KV通知完整精度/图/保护区PASS，L1中位823.38μs，同轮动态905.06μs；
未采用、不扩档。见[提前通知记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_EARLY_KV_EXPERIMENT.md)。
Indexer任意stride/GM基址条件选择的CPU接口检查见[页视图记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PAGE_BASE_SELECT_CPU.md)，
没有上卡。双AIV按KV搬运与attention数学分工的原型完整设备精度通过，
L1中位885.08μs，未采用、不扩档；见[分工记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_ROLE_SPLIT_EXPERIMENT.md)。
单次阶段诊断已完成，主要退步在QK/PV，AIV1数学/内部等待/维护段中位222.80μs，
历史现选158.12μs；插桩及暖机差异限制跨轮比较。当前全部任务终态，七档选择不变。

本轮修正测试驱动的提前编译问题：三个候选的首次artifact审计在编译扩展外，
实际未应用所请求的100tick等待，之前约1000μs的结果不能归因为数学核回退。
已移入正确上下文，并在执行前核对实际AIC/AIV源码。扫描74项已完成测试仅这三项受影响。
修正后WS加载顺序814.12μs、PV状态分块820.24μs、N64 L1环806.38μs，均完整PASS。
未替换现选，也未扩七档。详细证据和所有历史数据见
[编译上下文更正](../../../tests/pypto_test/csa_multishape_spmd_20261016/COMPILE_CONTEXT_CORRECTION.md)。

PV状态分块解决了196608B的UB编译超限，实际占用163872B，但本轮仍未证明性能收益。
见[PV完整记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_MERGE_STATE_EXPERIMENT.md)。
另有[两组query完整流水CPU评估](../../../tests/pypto_test/csa_multishape_spmd_20261016/QUERY_GROUP_SCHEDULE_FEASIBILITY.md)：六种模型均未优于现选，未上卡。
WS首对保留原序、后14对后移已验证；全15对后移的CPU失败证据仍保留。

新的N64两面板循环让第二块Key先搬到L1，再做第一块QK/WS。完整CPU编译、
Right/Acc/L1地址及真实提交/等待顺序审计已通过，只让score双槽跨循环。
任务`task_20261017_111540_229620229969`完成全门槛PASS，但完整L1中位813.42μs，
同轮动态898.80μs；未采用、不扩档。见[两面板预取记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_KEY_PAIRS_EXPERIMENT.md)。

本轮Indexer N32双Acc完整PASS但中位981.12μs，未采用，见
[N32记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N32_ACC_EXPERIMENT.md)。整个KV标记删除后远端等待增加490，未上卡；
保留汇总通知的纯信号候选完整PASS，首轮802.96μs，但同进程现选807.50、候选812.08μs，
未测出收益、不纳入选择。见[KV标记记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/KV_TOUCH_SIGNAL_EXPERIMENT.md)。
另有[固定L1循环更新CPU探针](../../../tests/pypto_test/csa_multishape_spmd_20261016/TILE_LOOP_INPLACE_CPU.md)：正确接收assemble返回值的版本
完整编译通过，未上卡，不能外推CSA大Tile流水。全部本轮任务已终态，七档选择不变。

上一轮N64两面板成组与2048分数交接均完整PASS，但L1中位为920.58/834.68μs，
未采用；见[分组记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PANEL_GROUP_EXPERIMENT.md)、
[交接记录与首轮错误修复](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_TRANSFER_2048_EXPERIMENT.md)。
Key L1池扩大通过CPU编译但未减少等待，未上卡，见[CPU记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_KEY_POOL_CPU_EXPERIMENT.md)。
主Q双N128列块首轮806.46μs；同进程复核现选808.92、候选806.02μs，
差0.36%且三窗区间重叠，未采用、不扩档；见[主Q完整记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/QPROJ_L1_COLUMNS_EXPERIMENT.md)。
所有本轮任务均已终态。七档选择不变，700μs目标未达到。

此前短尾候选B16/B24与B16递增轮询均完整PASS，L1分别820.12/1127.60/829.94μs，
没有明确收益，均不采用；见[短尾记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_SHORT_TAIL_EXPERIMENT.md)与
[递增轮询记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/RAMP_WAIT_EXPERIMENT.md)。N64展开循环化两版均CPU编译失败，
未上卡，见[循环化失败记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_ROLLED_PANELS_EXPERIMENT.md)。本轮所有任务已终态。

已新增完整CSA同地址动态metadata验证：128K/B16为131072→131071→130048→131072，
8K/B16为8192→8191→8186→8192；两侧输出/TopK/六份状态/保护区逐bit检查通过，
每侧固定一张图，地址不变，无新增编译单元。短历史首版7168跨策略失败记录保留，
修正版在派发前比较原编排策略，跨策略明确要求另选模板。见
[DYNAMIC_METADATA_VALIDATION.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/DYNAMIC_METADATA_VALIDATION.md)。这不是连续decode或整网验证。

本轮O-A K512与O-A八阶段共享循环均完整PASS，但L1为827.94/830.74μs，未替换现选。
同一QK数学体独立编译没有减少循环屏障；Indexer Key提交探针为11.81μs，
整体Indexer AIC约210.23μs，均为诊断证据，不参加700μs验收。
新增QK32行KV准备与HC48 AIV流式供数均完整exact/图/保护区PASS，但L1中位
分别831.32/844.40μs，未优于815.84，均未采用、不扩其他档。HC自动pipeline版本
在CPU审计中发现READY前置顺序被改变，未上卡；显式range版本通过。见
[HC_STREAM_PRE_SYNC_AUDIT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_STREAM_PRE_SYNC_AUDIT.md)。
新增Indexer每叶16384与M192/N64双Acc两项候选也均完整PASS，但L1中位分别829.22/968.92μs，
均未纳入选择。地址/提交顺序核对、完整结果和限制见
[INDEXER_PARTITION_EXPERIMENTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PARTITION_EXPERIMENTS.md)。
早期PV只缩小incoming Tile的版本因UB需196608B失败，未上卡；后续拆分循环状态后的结果见上方最新记录。
QK直接GM→L1候选完整exact/图/30代/保护区PASS，但L1中位1100.10μs，同轮动态892.66μs，
明确不采用。小搬运探针、同步模型、生成代码审计和失败记录见
[QK_DIRECT_L1_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_DIRECT_L1_EXPERIMENT.md)。
新增按query/head细化依赖：128K/B16首轮807.16μs，同进程复核旧841.94→新817.88μs，
完整精度和128K→64K→128K同图动态metadata检查通过。已纳入B16候选，其余六档不动。
见[行依赖实验](../../../tests/pypto_test/csa_multishape_spmd_20261016/ROW_DEPENDENCY_EXPERIMENT.md)。Indexer N80完整精度通过但L1为879.10μs，不采用，见
[候选记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N80_EXPERIMENT.md)。此前任务均已终态。最新HC整行驻留与主Q显式L0预取均完整通过，但L1为819.02/830.62μs，均未采用，见
[候选记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_QPROJ_EXPERIMENTS.md)。
新增HC三阶段融合、内部Tile直传、Indexer输入行依赖均完整PASS，L1中位分别
806.76/810.44/812.62μs，均未替换现选。细节见
[本轮候选和失败记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_FRONT_AND_INPUT_DEPENDENCIES.md)。现选版阶段探针完整PASS，
仅供归因，不能作为正式性能结果。原12成员HC mix的分块UB缓冲也完整PASS，
L1中位814.74μs，未采用；见[分块UB记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_CHUNK_BUFFER_EXPERIMENT.md)。
新增成员级离线排程三种策略只完成CPU评估，没有明显模型优势，未上卡；见
[成员级排程](../../../tests/pypto_test/csa_multishape_spmd_20261016/MEMBER_SCHEDULE_CPU_EXPERIMENT.md)。恢复权重缓存声明后MLIR确实改变，
但C++与现选全文相同，未重复测量；见[缓存策略核对](../../../tests/pypto_test/csa_multishape_spmd_20261016/WEIGHT_CACHE_POLICY_EXPERIMENT.md)。
QK每query地址缓存完整PASS，中位811.82μs，未采用；见
[QK地址缓存](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_QUERY_METADATA_EXPERIMENT.md)。N80 WS K128完整PASS，但中位855.10μs，未采用；见[WS分段记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N80_WS128_EXPERIMENT.md)。
QK全有效块路径完整PASS，中位817.26μs，未采用；见[分支记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_ALL_VALID_EXPERIMENT.md)。
全部尝试及失败见ATTEMPTS.md，接续见NEXT_STEP.md；
PENDING_JOBS.json是观测快照，不是实时队列。

## 目标与边界

2026-10-16 用户明确：**不要求不同 batch 共用一份二进制，目标是各档性能都好**。
参考 `vllm_ascend/ops/pypto/hc_pre.py` 的做法，入口允许动态描述维度，按档位选择
分块、分核和流水参数；JIT 可以按结构形状分别编译并缓存，也允许精确档位特化。

旧静态源码、正确性和性能结果在相邻的
`csa_static_spmd_feasibility_20261015/`，已以 `a919f119` 推送到
`nalinaly/vllm-ascend:dsv4-flash-pto-v0.25.1rc1`。本目录不覆盖旧结果。

- 第一版继续使用冻结的同一套 CSA 数学核体，参数化形状、视图、工作区和核序列。
  这样可以把调度变化与数学变化分开验证，暂不切换另一会话的算子代码。
- 各档选择自己的历史调度参考；重新计算实际分片数量与依赖。不得把 B16 的
  Tile、Indexer/O-proj 分支或完整等待表硬套到其他 batch。
- `position_ids`、`kv_seq_lens`、slot、页表、紧凑行 metadata 和 cache/state
  始终是 device Tensor。初始化只读取 shape/stride/dtype，不增加每步 `.item()`。
- 调度的分支选择允许使用调用方已有的 Host `max_seq_len`。它只选择 Indexer
  策略，实际候选范围、页地址和尾块仍在核内读取 device metadata 后计算。
  若跨越策略分界，应选新变体；不把跨分界错误执行称为“支持动态长度”。
- 本轮沿用 CSA 的 S=6 契约。此前独立 HC_pre 的 B/S 测试表不是 CSA 的七档。

## 验证次序

1. CPU 检查各规格的调用匹配、分片覆盖、MIX 配对、工作区边界和依赖偏序。
2. 用正式权重检查单卡输出、Top-K、cache/state、保护区和图重放。
3. 性能只比较原动态 PTO 与单 SPMD，沿用完整 CSA 的 L1 incore 首尾口径，
   每档 3 个窗口。另保留图重放 Event 时间；不执行 Native forward。

原七档为 128K/B4、B8、B16、B24 和 8K/B16、B24、B32。
另外选一个非历史 batch 验证参数化，避免只包装七份硬编码文件。
CPU 准备通过不代表真机正确或性能通过；新档未经测量不预记性能收益。

## 当前基线

已完成的 128K/B16 静态对照：原动态 PTO L1 中位数 897.70 μs，
单 SPMD 1380.22 μs，耗时增加 53.75%。这是保留的旧结果，不是多档版的新测量。
后续需先定位、减少融合后的控制/同步成本，再按各档独立选择策略。

## 当前目标与进展

活动目标：先完成多档形状支持，再将 **128K/B16 的无插桩完整 L1 incore
首尾耗时降到 700 μs 以下**。不以某个子阶段、核内累计忙时或插桩结果替代。

实现入口是 `shape_specialization.py:build_variant`：根据实际 Tensor 描述符
重绑定原符号编排，选择相同数学分支的参考调度，生成当前规格的 JIT 源码及
初始化工作区合同。PyPTO 自身缓存编译产物。它是实验准备入口，尚未替换生产入口。
不同规格允许编译不同二进制，运行时 metadata 保持 device Tensor。

`prepared_v1/result.json`：原七档及额外 128K/B12 的 CPU 形状、视图、MIX 配对、
分片覆盖、依赖偏序与三代进度模型均通过。
`prepared_v1/static_b16_identity.json`：B16 生成源码 AST 与已真机验证静态版一致，
不重复进行相同 B16 基线测量。

`device_compare.py` 从实际 Native 参数描述符生成变体，沿用原静态对照的
正式权重、合成历史、exact 输出/Top-K/六份 cache/state、保护区、图重放和
交替 AB/BA 计时方法。每侧 5 次预热、20 次 Event 样本、3 次 L1 窗口。

阶段探针在 `stage_clock.py`：仅当前测试进程拦截 PTOAS 返回的 C++，保留
插桩前后文件和边界检查记录。用每核进度区空闲的独占缓存行存四个 SYS_CNT
时间点；不写信号所在缓存行，不改 PyPTO/Simpler 仓库。插桩会影响耗时，
该结果标记 `eligible_for_700us_target=false`，只用于阶段归因。

全部新尝试与任务编号见 `ATTEMPTS.md`。

## 多档最新实测

重分配候选已完成原七档真机比较，全部通过固定 fixture 的精度、状态及图重放检查。
详细表格见 [BALANCED_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/BALANCED_SEVEN_RESULTS.md)。
七档矩阵中 128K/B16 无插桩结果为 **1067.36 μs**，同轮原动态为 **902.28 μs**；
700 μs 目标尚未达到，其他六档也仍慢于原动态。

`balanced_schedule.py` 在保持数学 DAG、分片覆盖和 MIX 配对的前提下重新分配。
完成通知汇聚会增加尾延迟，当前候选关闭；独立阶段片上内存规划仍在实验中。

21:22 补充：任务计数、轮询退避、独立阶段编译均未产生整体收益，具体负结果和定位证据已记入 ATTEMPTS.md。当前仍以 balanced_plain 的七档结果为基线；未切换生产算子默认路径。

21:55 补充：普通 balanced 新复测中位数 1054.70 μs，但三窗含一次 3922.40 μs 长尾，
不把中位数变化当作稳定优化。全阶段 noinline 诊断为 1044.54 μs，未切换生产入口。
外部 PMU 与实际参数裁剪记录见 ATTEMPTS.md、noinline_and_pmu_results.json。

22:44 补充：cached GM信号退避取得收益。B16在0/50/100tick下分别
1047.58/894.10/851.32μs，全部正确；100tick三窗850.94/851.32/854.06，
同轮动态907.02μs。700μs目标尚未达到，其他六档尚未测该退避候选。
此前“轮询退避未获收益”仅指已测ld_dev候选，不能概括此DCCI读侧新结果。
见cached_backoff_results.json与ATTEMPTS.md；未修改生产入口。

23:12 补充：新七档已全部完成并通过，见 [CACHED_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/CACHED_SEVEN_RESULTS.md)。
各档独立成本与分核、2μs信号退避使L1较同轮原动态降低2.89%～8.54%；
128K/B16为832.28μs，仍未到700μs。Event单独列出，部分档仍略慢。
当前仅在B16继续验证N256输出投影列分块，复用原算子已有分支，
原动态对照保持；首次完整CPU代码生成v2通过，真机结果尚待完成。


## 2026-10-16 23:55 增量

最新按档保留的七档结果见[SELECTED_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/SELECTED_SEVEN_RESULTS.md)。
QK压缩行地址批量在UB计算，只在128K/B8、B16保留新候选；其他档继续用原标量读取。
B16完整无插桩L1中位815.84μs，700μs目标尚未达到。全部新旧数据、负结果和探针限制
见[QK_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_SEVEN_RESULTS.md)及ATTEMPTS.md。
独立GM复用实验逐位通过，节省25051136B但833.28μs与原性能基本持平。
主Q投影不同worker数只做CPU准备/模型检查，未把模型值当成实测。


## 2026-10-17 09:58：N64循环流水与query完成粒度

新增[Indexer L1环实验记录](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_L1_RING_EXPERIMENT.md)。v1/v3编译拒绝，
v2虽编译成功但地址错误被审计拦截，未上卡；v4仅score跨循环，完整精度/图/状态/30代通过，
L1中位1008.86μs（同轮动态905.34），未采用、不扩档。现选七档保持。

[query完成粒度模型](../../../tests/pypto_test/csa_multishape_spmd_20261016/indexer_query_completion_model.json)覆盖1..32叶所有权：
仅按request重排而保留整阶段通知，下游仍不能提前。需连同merge/QK消费者及
O-A/O-B整批行限制评估，不把score块数当实际时间。后续边界见[NEXT_STEP](../../../tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP.md)。

<!-- SOURCE_END history-document-041 -->

---

<a id="history-document-042"></a>
### 历史原文 042：ATTEMPTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/ATTEMPTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/ATTEMPTS.md)；原文 156,957 字节。动态目标、过程总账或七档结果，全文保留。

<!-- SOURCE_BEGIN history-document-042 -->
# 多档 CSA 单 SPMD 尝试记录

## 2026-10-16：建立参数化准备入口

- 静态基线已提交并 push：`a919f119`，目标分支 `dsv4-flash-pto-v0.25.1rc1`。
- 使用独立 worktree `.worktrees/csa-spmd-dynamic-20261016`，未改共享生产入口、
  PyPTO、Simpler 或 pypto-lib 源码。NPU 测试仍使用此前已授权的隔离组件。
- `prepared_v1` 八个 CPU 规格全部通过；B16 源码 AST 与已验证版本相同。
- 测试脚本初次复制时把相对 `Path('.')` 的 parent 当作仓库父目录，读取失败；
  改为 `Path.cwd()` 后生成完成。未启动错误测试或改动原脚本。

## B12 非历史形状验证

1. `device_b12_v1` / `task_20261016_190445_377635926395`：退出 1。
   实际描述符绑定、分片/视图检查成功，尚未执行融合核。
   新增逻辑分片逐块展开使 AIV 源码段从 41 增至 69，PTOAS 触发固定 60 秒超时。
   保留生成源码、编译报错和 result.json，不记精度或性能通过。
2. `device_b12_v2` / `task_20261016_191204_200117150`：退出 1。
   一个原控制位置承接同一任务的多个分片，核内循环后合并发布进度，AIV 段回到
   41，AIC 38。CPU 检查通过。PTOAS 对外层 INDEX 循环变量直接用作内层起点
   报 `emitc.for op expected induction variable to be same type as bounds and step`，
   证据在 `build_output/entry_d8ja688g/report/codegen_errors.txt` 及 `.pto` 346、7581 行。
   没有修改 PTOAS 或放宽同步条件。
3. `device_b12_v3` / `task_20261016_191520_21364431342`：已提交。
   用固定波数循环计算 `block=参考block+wave*参考块数`，再判断范围，保留相同
   逻辑 block 集合、顺序与 MIX 配对，避免上述嵌套起点 lowering 问题。
   结果以该目录 `result.json` 和队列终态为准。

## 分档与阶段归因任务

- `b16_stage_clock_v1` / `task_20261016_191754_31988114769`：B16 阶段插桩；
  40 个 AIC 段与 44 个 AIV 段的插入边界已在旧 C++ 上核对；每核最多 20 个实际
  控制位置，80 个时间戳落在信号缓存行之后，位于每核既有 512 B 空间内。
- `device_128k_b4_v1` / `task_20261016_191807_32922422901`：长历史小档。
- `device_8k_b24_v1` / `task_20261016_191809_33070718758`：短历史 S6 Indexer 分支。

后续结果仍需查看任务终态，不能据“已提交”或编译完成推断 PASS。

## 设备结果与第一轮性能定位

- B12 v3：设备计算 PASS。eager/graph/最终输出、Top-K、六份 state/cache、保护区及
  30 代信号通过；L1 原动态 796.76 μs，单 SPMD 1130.30 μs。队列显示 exit=2：
  Python 已正常写出 complete，随后 Bash 因脚本末行在作业运行期间被修改而报 EOF。
  队列日志明确是包装脚本解析失败，不把整个 job 标为 exit=0。后续不原地修改正在
  使用的 shell 脚本；当前脚本 bash -n 通过，不为该包装问题重跑已完成的设备测量。
- 128K/B4：`task_20261016_191807_32922422901` exit=0，全部检查 PASS；
  原动态 L1 583.28 μs，单 SPMD 799.34 μs。
- 8K/B24：`task_20261016_191809_33070718758` exit=0，全部检查 PASS；
  原动态 L1 854.02 μs，单 SPMD 1286.18 μs。
- B16 阶段探针：`task_20261016_191754_31988114769` exit=0，全部检查 PASS。
  原动态 L1 892.48 μs，插桩单 SPMD 1334.38 μs；此值不用于目标验收。
  973 个成员的四点时间戳全部有效。Indexer 数学段中位数约 217 μs、QK/PV
  约 161 μs，与动态版约 222/162 μs 接近。明显延迟在 Indexer 启动前：
  原动态约 303 μs 启动，插桩融合约 599 μs。不能把等待累计时间直接当作可消除开销。
- 读侧绕过缓存候选：`b16_bypass_wait_v1` / `task_20261016_192452_64367617729`
  exit=0，全部检查及 30 代信号通过。只把 TWAIT 换成 A3 ld_dev 的 GE 轮询，
  TNOTIFY、业务数据维护与数学均未改。原动态 L1 901.88 μs，候选 1366.36 μs，
  未见明显性能改善，不声称已证明 DCCI 轮询是主要瓶颈。

## 保持数学偏序的完成通知汇聚

`completion_aggregation.py` 将每个成员分为计算与发布两个节点。只选择
“下一计算节点本就等待该任务全部生产者”的核，在其计算后、发布前汇聚。
随后消费者可等待一个完成标记。无合格核的任务保持原逐核依赖。

B16 CPU 结果：66 个任务中 37 个可汇聚，等待从 16556 降为 3641
（计算前 3134，发布前 507），发布点 711。检查双阶段图无环，并逐节点投影
核对原数学偏序完全相同；序号翻译后的偏序也相同，三代协议模型通过。
未增加全核屏障，未删除业务数据同步。`prepared_gather_v1` 保存完整准备结果。
本候选仍需真机与 L1 验证，CPU 减少的等待数量不是性能收益证明。

- 真机任务：`b16_gather_v1` / `task_20261016_193007_91656320040`。
  19:48 仍 pending；全 16 卡由前序整网任务占用，没有重复提交。

## 融合前后局部流水差异

检查同一 HC_pre widen 数学体：独立动态 PTO C++ 中 set/wait 各 34 处，
融合 C++ 中各 56 处，循环中额外出现 PIPE_ALL。数学操作种类和数量一致。
这证明生成流水不同，尚不能仅据事件数量推断性能差额或判定寄存器溢出。
`hc_widen_sync_diff.txt` 保留对比。

CPU 编译/反汇编尝试保留在 assembly_*：

1. v1 使用 -S，CCEC 拒绝 device 侧该选项。
2. v2 改为 ELF object，缺 tensor.h；v3 补 runtime Tensor include 后缺
   tensormap_and_ringbuffer/task_id.h；v4 补 common 后缺 intrinsic.h。
3. v5 使用 Simpler 正式 include 查询方法，AICore object 编译成功。
   llvm-objdump 只给出 not available，指定 dav-c220-vec 也没有指令。
   两份 ELF 的 stack-size record 都是编译选项 0x8000，不能据此证明真实栈溢出。
   暂停这个没有产出有效指令证据的分支，不据猜测下结论。

`preserve_math_pipeline.py` 是归因候选：在测试进程 C++ 生成边界，把
hc_widen_rms、mix_x_rms_norm、qproj_matmul、idx_qr_proj_matmul 数学区
替换为同轮动态 PTO 编译的 device 函数调用。核间等待、发布、业务缓存维护
与原分核次序保留；函数以 noinline 编译，仍然只有一个融合任务。
该候选不是最终 PyPTO 接口实现，明确不作为 700 μs 目标完成依据。

- `preserve_pipeline_cpu_v1`：8 个控制位置的实际 Tensor/Scalar/逻辑 block
  绑定核对完成，AIC/AIV CCEC 编译通过。
- `b16_preserve_pipeline_v1` / `task_20261016_194834_198181913198`：
  已提交设备对照，尚未得到结果；不预记正确性或性能收益。

## 汇聚实测与按规格重分配

- 完成汇聚 `task_20261016_193007_91656320040`：exit=0，输出、Top-K、
  六份状态、保护区、图重放及 30 代进度均 PASS。动态 L1 906.14 μs；
  单 SPMD 三窗为 1487.84、4639.64、4005.94 μs，中位数 4005.94 μs。
  Event 中位数 1355.20 μs，但最大 9254.46 μs，长尾明显；不采用此结果宣称优化。
- 保留局部流水 v1：`task_20261016_194834_198181913198` 首次融合执行失败，
  synchronize 报 507018 / AICPU 0x2a。编译成功不等于可执行；目前没有证据
  将故障归因于 noinline、缓存一致性或某一组件。变更前脚本保存在 test_source。
- 新的 `balanced_schedule.py` 按数学 task DAG 优先级和核空隙分配分片，
  保留所有逻辑 block、真实 metadata 计算和 MIX 三核配对。旧 trace 中 qproj 的
  20 片落在 12 个 AIC 上，最多同核 3 片；该偶然资源次序不再固定。
  CPU 代价来自参考 L1 中位数，仅用于分配，不作为实际性能结果。
- `prepared_balanced_all_v1/result.json`：原七档加 B12 的重分配、汇聚、
  视图边界及三代协议模型全部 PASS。真机覆盖不能由 CPU 结果替代。
- `b16_balanced_gather_v1` / `task_20261016_195218_216602122487`：
  全部设备检查和 30 代进度 PASS。动态 L1 897.42 μs，
  单 SPMD 1227.12 μs，Event 954.23/1334.08 μs。仍未达到 700 μs 目标。
- `b16_balanced_plain_v1` / `task_20261016_195623_23741736566`：
  只改变分配，去掉汇聚，独立测量其作用；已提交。
- `b16_preserve_inline_v2` / `task_20261016_195623_23739921091`：
  将归因 helper 改成 always_inline，仍保留原 PTOAS 局部流水，缩小上次
  失败的定位范围；已提交，不预记通过。

## 完成七档的无插桩重分配对照（20:11）

- `b16_balanced_plain_v1` / `task_20261016_195623_23741736566`：exit=0；
  输出、Top-K、六份状态、保护区和 30 代信号全部 PASS。无插桩 L1：
  动态 902.28 μs，单 SPMD 1067.36 μs。只保留重新分配，比通知汇聚更快。
- `b16_balanced_stage_v1` / `task_20261016_195951_255674726835`：
  exit=0，检查通过；插桩 L1 1166.52 μs，只用于归因。Indexer 数学段约
  214 μs；qproj 在该安排下延至 Indexer 后，另有优化资源次序的空间。
- 其余六档分三组顺序运行：
  `task_20261016_200302_26991286640`（128K/B4、B8）、
  `task_20261016_200302_26991692585`（128K/B24、8K/B16）、
  `task_20261016_200302_269932320183`（8K/B24、B32），全部 exit=0。
  七档完整数值见 `BALANCED_SEVEN_RESULTS.md`，均通过相同检查。
  每次恢复固定输入，不能据此宣称连续 decode 轨迹通过。
- 最初的三个 matrix 包装任务
  `task_20261016_200028_25818234812`、
  `task_20261016_200028_258186424881`、
  `task_20261016_200029_258202311263` 在解析队列自动附加的
  `--device` 参数时退出 2，没有执行 NPU；包装器补接收并核对 TASK_DEVICE 后提交上述任务。

## 局部流水替换及阶段内存规划

- `b16_preserve_inline_v2` / `task_20261016_195623_23739921091`：
  与 noinline 版本一样，首次融合执行报 507018 / AICPU 0x2a。
  这不能证明根因是内联。停止此 C++ 数学替换方向，保留失败现场。
- `balanced_memory_dump_v1`：CPU 全流水编译通过，保留 MemoryReuse 前后的 IR。
  `stage_memory_aiv0_v1` 的单阶段内存规划通过，但该阶段不是 widen。
- `stage_memory_widen_v1` 完成内存规划后因未设 backend 无法 codegen；
  补明确 Ascend910B 后，`stage_memory_widen_v2` CPU/PTOAS 编译通过。
  同一 widen 数学体单阶段规划时 set_flag/wait_flag 各 33 处，整层规划时各 56 处。
  这是编译流水差异的证据，不是设备性能收益证明。
- `phase_memory.py` 只在本测试进程逐阶段调用原 PyPTO MemoryReuse 与
  AllocateMemoryAddr，拒绝跨阶段共享片上值，保持原数学、核间依赖和流水排空。
  不修改共享 PyPTO/Simpler。
- `balanced_phase_memory_cpu_v1` 失败于嵌套 pass 触发外层 dump 回调，
  使外层 pass_index 越界；没有设备执行。
  v2 对嵌套 pass 隔离 instrument，同时继承完整验证级别及其他编译选项；
  CPU 编译已启动，结果待查。

## 20:38 新候选结果

- 阶段内存规划 CPU v2 遇到原打印 IR 中的 FREE_VAR，无法文本重解析。
  v3 直接组合 IR 节点、保留 Var/MemRef 身份，完整 CPU/PTOAS 编译通过。
  完整融合 C++ 的 widen set/wait 各 57 处（原各 56），不能把单阶段独立
  编译时的 33 对事件当作完整融合方案的改善；见 phase_memory_event_comparison.json。
- 编译调度表候选按等差分段生成每核字段，并转置/分组等待关系；68 个源码阶段、
  973 个成员的依赖全部精确重建。GM records/waits 不再在核内读取，业务 metadata
  保持 device Tensor。此变化仍未带来实测收益。
- 业务缓存候选保留每轮入口 DCCI、每阶段本核流水屏障、Scalar GM 写后的
  DCCI/DSB、Scalar GM 读前的 DCCI/DSB，以及全部信号维护。
  Tile DMA 路径不依赖 Scalar DCache；实际也未取得明显性能收益。
- earliest_ready 改为先安排可最早执行的已就绪任务，仍保持全部数学 DAG、
  逻辑分片及 MIX 配对。Q 投影在模型中能提前，但实测没有改善。
- b16_phase_memory_v1 / task_20261016_202018_36492382421：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 889.78 μs，单 SPMD 1089.86 μs。
- b16_compiled_control_v1 / task_20261016_202404_384514429980：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 901.92 μs，单 SPMD 1148.16 μs。
- b16_payload_cache_v1 / task_20261016_202739_40037602798：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 903.88 μs，单 SPMD 1077.06 μs。
- b16_earliest_ready_v1 / task_20261016_203314_1112088253：exit=0，全部精度/状态/保护区/图重放及 30 代信号 PASS；动态 892.32 μs，单 SPMD 1115.12 μs。

以上候选均未替代 balanced_plain 最佳版本。没有为这些退步候选重跑全七档。

## 局部编译边界继续定位

- 对旧 C++ 数学替换失败日志复核，实际是 Simpler S1:running-stalled，
  completed=0/1，约 60 秒无进展后退出；不是已定位的硬件故障。
  调度未结束的具体核内位置尚未知。
- stage_codegen.py 从当前融合 IR 提取一个完整阶段（等待、数学、发布），
  使用同一组 GM 参数和明确传入的外层代际独立编译，再作 device 函数调用。
  仍只有一个 SPMD 运行时任务。首次仅选择 hc_widen_rms，CPU 编译进行中。
  此路径是归因实验，不能作为生产完成或 700 μs 达标结果。

## 20:55 编译边界、参数与轮询诊断

- stage_codegen_cpu_v1 因使用 name_hint 当作 SSA 身份，遇到重复 t__tile 而退出；
  v2 按 IRNode.same_as 区分变量，并隔离嵌套 pass instrument，完整 CPU 编译通过。
  widen helper 的 set/wait 各 34 处，GM 参数直接来自同一融合函数，外层代际显式传入。
- stage_codegen_compact_cpu_v1：只保留 helper 实际使用的参数，CPU 编译通过。
  它只缩减内部 device 函数参数，不改变外层运行时任务数量和 ABI。
- b16_independent_widen_v1 / task_20261016_204559_77126426784：已排队。
  使用完整参数的独立 widen，仅作归因，不用于目标验收。
- b16_workspace_base_v1 / task_20261016_204727_83145615808：已排队。
  按真实 arena 基址加偏移重建临时 GM 指针，运行前检查实际 data_ptr 对应关系；
  数学、调度及外层 ABI 不变，仅作跨阶段活跃指针成本诊断。
- balanced_ptoas_cpu_v1：使用已有 PTOAS 规划路径编译失败：
  Mat 需要 4456448 bit（544 KiB），超过 4194304 bit（512 KiB）。
  没有设备执行，没有缩小 Tile 或放宽容量。
- b16_balanced_lddev_v1 / task_20261016_205309_10618254230；
  b16_backoff_25_v1 / task_20261016_205309_106179220657：已排队。
  前者为同一 balanced 分配上的 ld_dev 读侧对照，后者仅在未就绪时等待
  25 个 SYS_CNT tick（当前 50 MHz 下为 0.5 μs）再读，首次读取不延迟。
  写侧 TNOTIFY 的 DCCI/DSB、业务数据维护及代际检查保持不变。
  假设是大量空闲核轮询可能影响前置小核数计算，未预记收益。
- stage_codegen_hot_cpu_v1 正在 CPU 检查四个热点的独立编译及实际参数裁剪；
  尚无该候选设备结果。

- stage_codegen_hot_cpu_v1：四个热点的参数裁剪与独立 PTOAS 编译全部通过，内部函数参数数为 9～13 个；完整数据见 stage_codegen_hot_summary.json。尚未提交四热点设备变体，先等待单 widen 结果。

## 21:10 排队实验完成

- b16_independent_widen_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 894.2 μs，单 SPMD 1120.22 μs。
- b16_workspace_base_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 905.08 μs，单 SPMD 1065.62 μs。
- b16_balanced_lddev_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 897.08 μs，单 SPMD 1327.28 μs。
- b16_backoff_25_v1：全部精度、状态、保护区、图重放和 30 代信号检查 PASS；动态 900.1 μs，单 SPMD 1146.62 μs。

独立 widen、GM 基址重建仅作归因，不作为目标验收；GM 基址重建的 1065.62 μs 与原 1067.36 μs 接近，未证明明显收益。连续 ld_dev 与 0.5 μs 退避都未优于原等待方式，不扩大七档复测。

任务计数候选复用当前 A3 PTO-ISA TNOTIFY(AtomicAdd)：每个原逻辑 task 独立计数，消费者等待原 DAG 前驱全部完成。atomic 防止多写者丢更新，DCCI/DSB 承担发布职责；保留业务数据及本核流水屏障。先做依赖偏序等价核对，再验证设备；不预记收益。

- task_counter_prepare_v1：68 个源码阶段、973 个控制成员，任务计数等待 2149 项，原逐核等待 20435 项；全部祖先偏序完全一致，数学体和业务缓存屏障未改。每个逻辑 task 独占 512 B 行中的第二段，不与代际信号共用缓存行。
- task_counter_cpu_v1：完整 CPU/PTOAS 编译通过。运行时仍须验证 AtomicAdd、图重放及 66 个计数器代际。
- b16_earliest_stage_v1 / task_20261016_211142_18013885165：只补测已知退步调度的阶段时间，定位资源安排与实际数学耗时；插桩结果不作为目标验收。

## 21:16 任务计数与编译同步消除

- b16_task_counter_v1 / task_20261016_211233_181811726212：全部输出、状态、保护区、图重放、30 代单核标记及 66 个任务计数器检查 PASS。动态 L1 905.74 μs，单 SPMD 1235.48 μs；减少等待项没有带来性能收益，不采用此策略。
- earliest_stage 阶段探针通过。Q 投影确实从 632～700 μs 提前到 264～359 μs，但 Indexer 仍到约 659 μs，QK/PV 约 747 μs 才开始，前置任务与末尾投影仍有较长执行段。完整插桩结果为 1126.76 μs，不作为无插桩对照或目标验收。见 scheduling_stage_comparison.json。
- 只读 PTOAS 源码 lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp：hasGatherScatterLikeOps 命中任一 gather/scatter 时，整函数跳过 RemoveRedundantSync。该条件未区分 A3/A5；注释说明是 A5 保守处理。当前融合函数含 gather，独立 widen 不含，这是局部流水差异的待验证原因，不能仅凭源码断言设备退步根因。
- stage_codegen_all_cpu_v1：准备将同一融合 IR 的全部阶段分别编译。完整 outline 后，外层只有 Scalar 分发/代际初始化，检查并去掉外层原数学的成对 auto-sync 初始化/回收，避免污染内部函数独立事件生命周期。每个 helper 原数学、跨核事件与数据维护保持。只在测试进程改写生成代码，不改 PTOAS 源码。

## 21:22 全阶段编译结果与后续范围

- stage_codegen_all_cpu_v1：68 个完整阶段独立规划/编译全部通过，内部函数仅有 8～26 个参数。外层旧数学残留的 set/wait 成对核对后全部移除，内部函数自己维护完整事件生命周期。
- b16_all_stage_codegen_v1 / task_20261016_211655_201090727888：全部输出、Top-K、cache/state、保护区、图重放、30 代信号检查 PASS；动态 901.72 μs，融合 1153.34 μs。仍较 balanced_plain 退步，不采用。
- ptoas_sync_phase_evidence/summary.json：实际 PTOAS 0.66 CPU 日志，整层 AIC 的冗余同步消除有执行（3188→772 对），整层 AIV 没有减少（2584→2584 对）；独立 widen 减少 46→28 对。EventId 分配会再改变数目，不能拿这一步计数当最终 set/wait 数量，更不能直接当性能收益。
- b16_all_stage_noinline_v1 / task_20261016_212120_215646228211：保留全部阶段的设备函数边界，核实 CCEC 重新内联后的影响。外层已没有数学流水事件；与此前旧版局部替换超时实验不同。待设备结果，不预记 PASS。

- b16_pipe_profile_v1 / task_20261016_212705_262039629706：只为当前 balanced 基线追加外部 PipeUtilization，计时完成后两侧各采三步，正确性和代际检查覆盖追加重放。完整持久核 PMU 与 L1 核内计时严格区分；不绕过 Simpler kernel 模式对 enable_pmu 的限制。
- 更完整的编译证据及限制见 PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md。


## 21:55：noinline 完成，外部硬件计数和实际 ABI 裁剪诊断

- `task_20261016_212120_215646228211` / `b16_all_stage_noinline_v1` PASS。
  动态 / 单 SPMD L1 中位数 893.54 / 1044.54 μs，三次单 SPMD 为
  1037.44、1044.54、1063.24 μs。全阶段内联版 1153.34 μs，保留设备调用边界
  更快，但尚未证明相对普通 balanced 的稳定收益；这个 C++ 组合诊断不用于正式验收。
- `task_20261016_212705_262039629706` / `b16_pipe_profile_v1` PASS。
  正常计时先完成，L1 动态 / 单 SPMD 中位数 892.56 / 1054.70 μs。
  单 SPMD 三窗 1054.70、3922.40、1040.26 μs，保留第二窗长尾，不能说稳定到 1054 μs。
  后追加每侧三个 PipeUtilization step，33 代进度、输出、Top-K、状态及保护区均通过。
- 外部 PMU 完整持久核中位数：AIC MAC 189.587→189.167 μs，AIC MTE2
  357.960→464.451 μs，AIV Vector 186.749→189.997 μs，AIV Scalar
  470.434→714.641 μs。以上覆盖外围运行时，流水间可重叠，不能直接相加解释 L1 差值。
  指令缓存 miss rate：AIC 0.002→0.006，AIV 0.003→0.013；按工具原字段值记录，
  没有仅凭比率认定 ICache 为瓶颈。原始 CSV 与 `pipe_utilization_summary.json` 保留。
- 新增 `compact_abi.py` 仅诊断实际入参：130 个根 / 132 个叶子参数收敛为
  59 / 61，其中工作区视图合并为一个 arena。Native 权重、state 和缓存不搬运。
  必须核对同规格源 AST、全部工作区及控制表一致，并在设备启动前核对实际指针。
  原先 workspace_base 只是指针重建、没有减少实际传入参数；这次才验证 ABI 部分。
- CPU 准备 v1/v2 发现保存的包装 C++ 已把 AICORE 改写为 static __aicore__；
  修正严格签名匹配，不修改数学体。v3 导入成功但 mixed extern 多输出返回在 Group
  解析中报 TupleGetItemExpr/UnknownType。v4 改用公开 void/in-place extern，仍以
  InOut 标注全部写入，不改 PyPTO。CPU 编译和设备结果待实际完成后追加。


## 21:58：实际参数压缩设备对照完成，无收益

`task_20261016_215358_37103827122` / `b16_compact_abi_v1` PASS。
CPU v4 已通过 kernel artifact 编译，使用 mixed extern 的 void/in-place 形式，
全部写入通过 InOut 声明。设备启动前 71 项工作区地址关系逐项正确；实际根参数
59 项、叶子参数 61 项、72 个 workspace 视图共用一个 arena，无新增 copy。
输出、Top-K、六份状态、图重放、30 代进度、Native/工作区保护区全部通过。
L1 动态中位数 907.10 μs，单 SPMD 1108.48 μs（1108.48/1111.24/1084.90）。
Event 中位数 957.820 / 1186.610 μs。没有整体收益，不切换生产入口。

这只能排除“本次具体参数压缩实现已经有效”，不能证明所有参数/寄存器开销为零。
该实现使用原 PTOAS C++ 和公开 extern，仅为诊断，仍不计入正式 PyPTO 目标验收。

追加同规格实测阶段成本候选：用已有 stage_clock 的数学+维护+通知中位数，
排除等待时间；严格匹配所有根 shape、全部 task/slot；离线重新分配时使用。
不在运行时读取 profile，不改变数学体、依赖偏序或业务 device metadata。

工具记录：本轮两次误用 `task-submit list/status` 被解释成无设备 shell 任务，
编号 task_20261016_214559_336266930669、task_20261016_215431_372084912112，
均 exit=127；没有占 NPU。后续查询已使用正确的 --status/--list 参数。


## 22:03：实测成本重新分核完成，无整体收益

`task_20261016_215905_38576944603` / `b16_measured_schedule_v1` PASS。
CPU 四个成本源/策略组合只用于选择候选，未把估计作为性能：最后选用
同规格 earliest_stage 数学+维护+通知中位数，earliest_ready 策略。
估计 883.40 μs；真正 L1 动态 885.36 μs，单 SPMD 1109.76 μs，三窗
1096.72/1109.76/1357.80 μs。Event 中位数 945.150/1211.380 μs。
相对原 balanced 没有收益，保留负结果；未切换默认策略、未重复七档测试。
等待项 20435→17546，发布 729 次，逻辑分片、MIX 配对和依赖偏序均检查通过。
实际设备输出、Top-K、状态、图重放、信号及保护区均通过。

新增 `core_specialization.py` 归因：结合此前 compiled_control，把物理 group/lane
作为 C++ 模板常量，24 个 AIC 和 48 个 AIV 分别常量折叠各自的调度条件。
仍只有一个运行时 MIX SPMD；长度、位置、页表等业务 Tensor 不特化。
数学/信号 C++ 函数体逐字保持，只替换核编号参数来源并增加分派外壳。
目的在于检验无关阶段判断、控制标量与代码活跃范围的影响。CPU 代码生成与二进制
编译必须先过，再决定设备验证；不是已测收益，也不是最终生产 PyPTO 接口。

另只做了一次 GM 临时区复用可行性分析：按完整 task-DAG 的严格先后关系，
294434944 B 可保守分组到 168272384 B，未修改布局、未执行复用。
空访问 temporary_47 保守独占，没有基于“未找到调用”删除它。其余未完整审查的
生命周期、返回别名和缓存语义必须进一步分析，不能直接据此宣称安全或性能收益。


## 22:23：核身份全展开的编译成本与原数学核完整组合

上一轮属于实际进展：完成两组设备负结果及新的核身份特化源码。当前活动目标没有完成。
CPU `core_identity_cpu_v1` 代码生成通过；`core_identity_binary_v1` 中 AIC 编译/链接完成，
代码段 1532220 B（ELF 含调试信息 24324936 B，不能混成指令体积）。
AIV 的 ccec 进程持续运行，最终由原编译器 900 秒超时终止，exec session 25592 exit=1。
这是已确认的编译超时，不是 NPU 挂死；未占卡、未提交该候选设备任务。没有因日志不动重启。
保留源码和部分 AIC 产物，不反复重编全部数学体；后续优先共享原数学函数，只简化外层控制。

新增 `reuse_identical_core_binaries`：设备候选仅在源 C++ 逐字相同、现有编译上下文相同
时复用已完成 CPU 编译的 ELF；其余回退正常编译。复用不增加 hash，不绕过运行时注册、
编译上下文检查或二进制解析。本次超时候选没有可复用的完整产物。

全部原数学函数组合：`preserve_all_prepare_v2` 已逐项映射 68 个物理阶段的 GM 指针、
scalar、shape/stride、整除恢复维度、block_num 和 MIX lane。和早前四个局部数学替换不同：
全部数学核保留原动态 PTO 编译出的局部流水，外层仅留控制、等待/通知和 payload 维护；
严格成对清理外层原数学 auto-sync 的初始化/回收，避免事件生命周期混用。
还保留原物理包装器每次调用 `set_atomic_none()` 的约定。每个原数学 helper 用 noinline，
共用代码，不为 72 个物理核复制全部数学体。此阶段仍为 C++ 诊断，不是生产接口。

`preserve_all_prepare_v1` 初次映射不支持动态 shape 的 /32 恢复表达式，已补充仅允许
明确 shape/divisor 且精确整除的映射，没有对任意 C++ 字符串执行 eval。
`preserve_all_cpu_v1` 因命令填错历史 build 目录而失败，已保留错误记录；改为实际存在的
`_decode_csa_tp1_layer__lncpugg` 后启动 v2。该输入路径错误不归因于编译器。


## 22:34：完整原数学核设备通过；继续定位等待/计算开销

`task_20261016_222628_28521515432` / `b16_preserve_all_v1` PASS。
动态/单 SPMD L1 中位数 891.18/1086.06 μs，单 SPMD 三窗
1086.06、1098.00、1062.96 μs；Event 中位数 956.650/1188.480 μs。
输出、Top-K、六份状态、保护区、图重放和 30 代信号均通过，仍无整体收益。
本结果不能单独证明旧四阶段替换失败的唯一原因，但完整原数学核的事件生命周期
清理、调用入参和单任务组合已获本规格真机验证。

CPU preserve_all_binary_v1 用 28.54 秒完成两类核和调度二进制；AIC 48728 B、
AIV 122656 B（代码段）。预编译复用未命中，走正常编译，未使用错误二进制。
复查发现缓存 JSON 外层含 context/files，复用检查初版拿外层和 context 字段比较，
造成保守拒绝，已修正解析。源文件也逐字不相同，首批差异为 IR 内联编号注释；
仍不放宽源码一致条件，不把未复用描述成已复用或直接归因于 SDK 不兼容。

已提交 `task_20261016_223130_37234825791`，b16_preserve_all_stage_v1：
相同完整原数学核组合，加阶段 SYS_CNT，区分等待、数学/维护和通知，只用于归因。
另提交 `task_20261016_223226_39023724918`，b16_cached_backoff_50_v1：
保留 PTO-ISA DCCI+volatile GM 的读侧语义，只在信号未就绪时以 SYS_CNT 间隔 50 tick
（1 μs）重新轮询；写侧 DCCI+DSB、PIPE_ALL 和 payload 维护不变。
这是检查轮询争用的独立候选，与此前 ld_dev 绕过缓存的退避不是同一种读法。
二者结果尚待完成，不能预记收益。


## 22:39：DCCI 轮询退避首次取得明确收益

`task_20261016_223226_39023724918` / `b16_cached_backoff_50_v1` PASS。
完整 L1 动态 906.52 μs、单 SPMD 894.10 μs，单 SPMD 三窗
894.10/896.94/892.28 μs；Event 动态956.480/单SPMD999.510 μs。
L1 已接近同轮动态，但 Event 仍更慢，两个口径分别保留。输出、Top-K、
六份状态、图重放、30代信号和保护区全部通过。固定fixture，不扩称连续decode。
只在单元素 GE 信号尚未就绪时退避50个SYS_CNT tick（1 μs）；每次读前DCCI，
写侧和业务数据维护不变。该结果不能单独区分 helper 与退避贡献，故下一步
增加 cached 0 对照、100tick候选和50tick阶段探针，不重复七档负候选。

`task_20261016_223130_37234825791` / `b16_preserve_all_stage_v1` PASS。
插桩 L1 动态902.32/单SPMD1094.30 μs，不计入正式目标。原数学函数下仍有
较长前置阶段：hc_widen_rms数学/维护中位22.44 μs，Indexer约377.56 μs
开始，qproj矩阵乘被本分核安排到Indexer之后。此处先只作为定位证据；
不能把每核等待相加当作整层耗时，也不能认定只改数学编译就能解决。


## 22:44：退避对照与原动态实测成本

同一 cached helper 的0/50/100 tick版本均通过。0 tick
`task_20261016_223952_51986417600`为动态902.46/单SPMD1047.58 μs，
50 tick为906.52/894.10 μs，100 tick
`task_20261016_223952_51980931381`为907.02/851.32 μs（三窗850.94/851.32/854.06）。
这支持退避本身有显著贡献；不能仅从时延推断具体L2/互连争用机制。
100 tick的Event为951.840/951.610 μs，完整正确性、状态和保护区均通过。

50 tick阶段探针 `task_20261016_223952_5198234964` PASS，插桩中位892.60 μs，
只作归因。widen数学/维护中位13.51 μs；Indexer约233 μs开始，
主Q投影约444 μs开始，QK/PV约542 μs开始。与无退避探针相比前置
数学阶段也缩短，仍需要区分实际争用与代码生成影响，不先认定硬件瓶颈。

新增 `export_dynamic_cost.py`：从已完成原动态三窗L1提取同规格70个task/slot
成本。严格核对task ID、物理核类型、核体名称、root shape和完整覆盖；
第三列正确当作dispatch_seq，不误认kernel_id。数学成员内部等待仍包含在成本中。
离线模型：critical_path802.51/earliest_ready773.43 μs，不是设备实测；
仅选择后者配50tick重新对照，任务`task_20261016_224301_58928227892`。
业务长度/位置等仍是device metadata，各batch允许自己的分核和二进制。


## 22:49：退避达到平台，检查静态控制和等待热点

200tick task_20261016_224348_60377711059 PASS，L1动态894.12/单SPMD848.38μs，
单SPMD840.16/848.38/858.60。400tick task_20261016_224348_60385920731 PASS，
909.26/846.56μs，846.56/853.76/846.32。较100tick的851.32μs只差几微秒，
不宣称稳定改善；停止继续扩大统一退避间隔，组合候选沿用100tick（2μs）。

原动态成本+earliest_ready+50tick task_20261016_224301_58928227892 PASS，
892.68/855.36μs（859.20/855.36/854.94）。比旧分核50tick894.10更低，
据此提交同分核100tick task_20261016_224732_68404112819。
另提交原balanced+100tick+静态控制表 task_20261016_224732_68414320903，
检查轮询开销下降后，消除调度GM读取是否有用；尚未预记收益。

新增rotated_waits.py只在初始化时重排每个消费者的等待表：按core号循环旋转，
使首个轮询地址分散，仍遍历完全相同的前驱集合，信号不清除。
20435条依赖逐项多重集合一致，714个成员改变次序；kernel源码、记录表和数学体不变。
CPU原协议验证器的9组随机顺序/代际全部通过。第一次CPU执行因脚本多一个右括号
报SyntaxError，修正后通过，不归因于PyPTO。设备结果待后续记录。


## 22:59：七档退避+独立成本分核完成

见 CACHED_SEVEN_RESULTS.md / cached_seven_results.json。七档全部PASS，
L1均较同轮原动态低2.89%～8.54%，B16当前832.28μs仍未达700μs。
保留Event差异：多数档Event接近或仍略慢，不能宣称整网端到端均加速。

B16静态控制表 task_20261016_224732_68414320903 PASS874.24μs，
等待顺序旋转 task_20261016_224904_7179821170 PASS839.48μs，
未超过实际成本+earliest_ready+100tick的832.28μs，不设为默认策略。
新阶段探针 task_20261016_225258_84749429745 PASS，插桩830.74μs仅用于定位。

CPU另扫描16个就绪时间/剩余关键路径权重，未优于原earliest_ready；
结果b16_schedule_weight_search.json，所有分片、MIX配对和依赖偏序证明通过，
未把这些CPU负候选提交NPU。原动态成本的资源/关键路径下界只是估算，
不能据此承诺700μs或宣布硬件已到极限。

继续两项B16同步候选：只保留有Scalar业务读写的DCCI（已有独立审查与旧版验证），
以及保持数学偏序的完成通知汇聚，在有效退避基础上重新检查其效果。
其他六档不随之重复测试，除非B16有明确收益。汇总脚本初版的命令示例字符串
转义报SyntaxError，改成单行命令后完成；未影响已完成的设备结果。


## 23:01：退避后裁剪缓存维护和完成汇聚没有额外收益

`task_20261016_225624_9520089233` / b16_cached_dynamic_precise_v1 PASS：
动态898.72/单SPMD838.68μs，三窗845.16/838.68/826.80，Event954.870/940.560。
`task_20261016_225727_97074417001` / b16_cached_dynamic_aggregate_v1 PASS：
动态906.82/单SPMD860.30μs，三窗870.28/854.96/860.30，Event959.590/966.030。
未超过已测832.28μs；不把这两个同步变体设为默认、不重测其他六档。

CPU用有效退避下的新阶段数学+维护+通知成本再分核：原earliest_ready估计760.19μs，
加0.05关键路径权重估计754.30μs；这仅选候选，不是实际提速。后续继续用CPU
检查前置主Q投影优先级能否减少资源串行，再选择值得验证的最少设备候选。


## 23:10：新的实测成本仍未更优，进入原有列分块候选

新阶段成本+0.05权重 `task_20261016_230307_106795216098` PASS：
动态909.60/单SPMD863.66μs，三窗871.50/863.66/863.56，未超832.28μs。
CPU关键路径主Q投影偏置扫描最好估计758.78μs，也未优于已测试模型，未提交NPU。

新增oproj_retile.py：从冻结原CSA读取已有proj_a_mm_2（N256）核体，
逐AST节点核对相对N128只改列索引、列分片数及局部Tensor列维。
原有行128、K256、K归约顺序、dtype和cache策略不动；重新绑定原入参名。
B16每个O-A任务8→4逻辑分片，8个group仍覆盖全部1024列。
oproj_n256_body_check_v1和oproj_n256_prepare_v1通过；后者包含分片、MIX配对、
等待偏序、工作区与9组CPU代际模拟。使用相同输出缓冲，不改变原动态参考。

这是首次尝试改变列Tile策略，不再宣称数学核体逐字不变；exact门槛保持，
不能从实数等价或归约顺序一致直接预记浮点逐位通过。新成本尚未测量，
离线初排显式按N列数2倍估计原成员成本，标注unmeasured_duration_scales。
CPU完整代码生成正在oproj_n256_cpu_v1执行，成功之后才提交B16设备验证。


### N256 CPU v1参数重绑定错误及v2修正

oproj_n256_cpu_v1前端拒绝未定义t_dim__ssa_v0。原因是本地AST等价核对
把用于执行的同一份AST也做了去inline编号规范化，后续按原名字重绑定失效。
改为仅对深拷贝做规范化比较，执行AST保留原名；补充函数体自由变量必须
完全由原参数和局部赋值绑定的检查。不是PyPTO bug，不修改依赖仓。
oproj_n256_prepare_v2通过全部CPU协议检查，CPU完整编译v2继续执行。
原v1源码、失败日志和result均保留，不覆盖。


## 23:12：N256 CPU完整代码生成通过，提交B16设备候选

oproj_n256_cpu_v2完整PyPTO/PTOAS代码生成通过；未把CPU通过称为设备通过。
没有编译器或runtime新改动。设备候选沿用原动态成本初排、earliest_ready、
100tick退避，仅新增O-A N256列块；原exact输出与状态检查保持。


## 23:16：N256输出投影真机逐位通过，未形成明确整体收益

`task_20261016_231256_12244625650` / b16_cached_oproj_n256_v1 PASS。
原动态N128中位911.72μs、单SPMD N256为839.00μs，三窗840.92/839.00/838.60；
Event961.780/942.670μs。输出、Top-K、六份cache/state、图重放、30代信号及
全部保护区仍按原门槛通过。不同列块并没有在本规格引入可见逐位差异。
但839.00没有低于此前N128的832.28，且原动态同轮也有波动，不宣称这项列分块
有稳定的额外收益；不切换七档策略、不扩大重测。原N128和N256代码与全部证据均保留。

当前没有未完成的设备任务或CPU编译会话。七档性能汇总以CACHED_SEVEN_RESULTS.md为准，
700μs目标未达到，活动目标保持继续。下一步应依据阶段/生命周期证据继续检查，
不要重新扫描统一退避间隔或重跑本轮已经完成的七档。


## 23:27：临时GM等大小复用的CPU证明通过

新增gm_reuse.py，只用完整task-DAG证明所有旧使用者先于全部新使用者。
不增加依赖，不使用trace时间或单核偶然先后；FFTS temporary_50/62、
未知用途temporary_47和全部控制区独占。PyPTO入口拒绝部分可写重叠，
因此本轮仅允许字节范围完全相同的别名，未改变依赖库的校验。

gm_reuse_prepare_v1：arena从294996992B缩至269945856B，节省25051136B。
生成源码AST及逐核安排与当前832.28μs版本完全相同；没有数学或同步语句变化。
新增最终布局检查，旧workspace_regions_disjoint仅标注baseline，不能冒充复用后互斥。
控制区重叠、部分重叠和同大小但使用者无严格先后均被拒绝。
逻辑scratch的调用后中间值不再保持，正式输出/TopK/Native状态不参与复用。
CPU检查不替代硬件正确性，接下来仅提交128K/B16，保留原exact门槛。


## 23:33：GM复用真机PASS，性能持平

task_20261016_232922_148038020750 / b16_cached_gm_reuse_v1 PASS。
完整L1原动态908.64μs、单SPMD833.28μs，Event955.460/938.520μs。
输出、TopK、六份cache/state、图重放、30代信号及保护区通过。
相对先前832.28μs无明确性能收益；保留其节省25051136B内存的独立候选，
不改七档性能默认策略、不重跑已经通过的七档。

主Q投影原为20worker处理128个列块，在静态图中有独立资源排程，
新增qproj_partition.py允许24/32/64/128个逻辑worker候选，保持24物理AIC上限。
仅改循环次数及列块编号，K256四次INT8点积和数学体逐AST相同；
CPU枚举保证128个列块各一次。首次只准备24worker，不盲目四档全上设备。


## 23:41：主Q投影CPU候选与QK地址计算方向

qproj24_prepare_v1协议和qproj24_cpu_v1完整代码生成通过；并非设备通过。
模型扫描v1误将重分核后的schedule当作原trace anchor，报KeyError end；
v2使用冻结anchor修正调用，不修改调度器或依赖仓。
qproj_partition_model_v2估计20/24/32/64/128worker分别
773.43/782.65/774.52/778.34/777.95μs，均无改善依据，未提交NPU。
新粒度成本按20/workers粗估，不能作为实际性能结论。

QK/PV压缩KV的64行循环每行Scalar读稀疏索引、再读页表。新增
qk_index_cache.py候选：MTE2批量读取索引和当前请求页表，Vector整数
位移+gather计算物理行，Scalar改读本核UB；负索引分支、逐行KV DMA、
QK/PV与softmax数学及FFTS同步不变。只改本实验body_bundle副本。
页表仍是原device Tensor，每块都重读，没有Host取值或GM展开metadata。
页表尾部使用显式valid_shape，未把1026列当成1032列实际数据读取。


## 23:45：QK地址缓存完整CPU代码生成通过，提交单个B16

qk_index_prepare_v1：20016项CPU地址公式检查通过，包含负索引与页边界；
qk_index_cpu_v1完整代码生成PASS。原核体集合只有qk_pv_aiv被改写，
其余函数AST不变，完整调度与工作区分配不变。
提交task_20261016_234523_17348223934 / b16_cached_qk_indices_v1，
采用832.28μs版本的调度与100tick轮询，未叠加GM复用或主Q重分片，
完整exact/图重放/状态/保护区检查及3窗L1保持。尚未获得设备结果。


## 23:49：QK地址缓存B16完整PASS，L1中位815.84μs

task_20261016_234523_17348223934 / b16_cached_qk_indices_v1 PASS。
单SPMD三窗829.04/815.84/813.34，中位815.84μs；原动态中位902.80μs。
Event原动态957.650/单917.870μs。完整输出、TopK、六份状态、图重放、
30代信号和保护区逐位检查通过。相比原832.28中位少16.44μs；
这是跨轮单SPMD比较，同轮动态对照均保留，不把三次测量当大样本统计结论。

因本改动影响所有规格，提交其余六档完整检查，任务清单qk_indices_matrix_jobs.json。
同时提交B16 stage_clock定位本次收益；探针结果明确不计入700μs门槛。
没有重测未改变的旧七档基线；每份新任务内只运行原动态PTO对照和本候选。


## 23:55：七档地址缓存验证完成，按档选择而非统一启用

七个无插桩规格及B16阶段探针全部PASS，任务状态completed(exit=0)。
QK新版本128K B4/B8/B16/B24为541.84/641.56/815.84/1137.90μs；
8K B16/B24/B32为632.58/865.16/943.20μs。
对照旧单SPMD，B4持平，B8/B16改善，其他四档退步；没有删除负结果。
选择只在128K/B8/B16保留新地址缓存，其余五档沿用旧实现。
各档采用既有独立调度，不要求跨batch或策略共用二进制。
完整数据QK_SEVEN_RESULTS.md；当前所选数据SELECTED_SEVEN_RESULTS.md；
机器可读选项selected_shape_policies.json（不是生产入口自动切换）。

B16探针task_20261016_234943_18100881575完整PASS但插桩L1 830.12μs
不进入目标门槛。原/新QK/PV AIC成员数学+通知中位155.82/153.85，
AIV160.58/158.41μs；Indexer AIC209.88/212.10、AIV202.83/202.31μs。
探针变化小于无插桩跨轮差异，不能把完整16.44μs全部解释成地址缓存收益。
探针本身会改变指令、寄存器和计时，当前只作定位；没有用它宣称<700。

当前无未完成设备任务或CPU编译会话。B16最好无插桩815.84μs，
700μs目标仍未完成。原七档、当前七档都已完成，不应无改动再次全量重测。


## 2026-10-17 00:13：Indexer N128首版因L0C容量失败，未上设备

indexer_n128_prepare_v1通过绑定与CPU协议检查，分核、调用、根描述符、
等待表均与现选B16相同。AST审计首版脚本只看module.body，漏掉类内函数
而断言失败；修正为ast.walk后确认仅Indexer AIC变化，其余函数不变。
完整CPU代码生成indexer_n128_cpu_v1失败：Acc204800B超过平台131072B。
理论左/右内存容量足够不代表Acc足够；未改平台限制或依赖仓，未上设备。
原N128/M384的QK本身需要192KiB Acc，不能采用该形状。保留失败生成物。
下一版将每个N128面板分成两个M192 QK，仍让六query共用一次Key加载；
WS分别规约三个query，Right占48KiB，仅保留一个16KiB Key Right，
下一面板只预取到L1。峰值Acc理论104KiB，仍需实际编译验证。


## 00:16：Indexer N128两组三query版CPU通过并提交B16

indexer_n128_prepare_v2 / indexer_n128_cpu_v2完整代码生成通过。
新QK为M192/N128/K128 INT8，WS为M16(有效3)/N128/K192 FP16，
同一个Key Right先后供两个QK使用，下一Key仅预取L1。原分组、
通信双槽和AIV处理不变，没有新增GM缓存或Host数据读取。
任务task_20261017_001600_235577716532，输出b16_cached_indexer_n128_v2，
现选B16调度+QK地址缓存+100tick等待，只额外启用--indexer-n128。
已提交，待设备正确性与三窗L1；未预记PASS或收益。


## 00:21：设备排队期间，准备QK全有效块搬运候选

当前16卡被另一个整机任务占用，N128任务仍pending，没有绕开task-submit。
检查现选B16生成C++可见：64行循环每行仍读一次raw索引、一次物理行，
并执行列边界/负索引判断；没有逐行额外pipe barrier。新增qk_full_block.py：
当前64行raw索引先做INT32 row_min，全部非负时展开8行一组搬运，只读
物理行；含负索引时完整保留原回退。范围判定来自device，不固定历史长度。
不改变数学、FFTS、页表或GM工作区。先做CPU编译，尚未提交设备。
qk_full_prepare_v1通过进程内组合构建，未修改运行中/排队任务的导入模块。


## 00:23：N128设备PASS但性能退步，保留N64

task_20261017_001600_235577716532完成exit=0，所有exact/状态/TopK/
图重放/信号代际和保护区通过。单SPMD L1三窗881.60/883.00/897.02μs，
中位883.00μs；同轮原动态889.42/903.92/884.82μs，中位889.42μs。
对比现选815.84μs没有收益，不扩七档、不设默认。两组三query共享Key
数学正确，但N扩大不自动带来收益；QK/WS调用数不变且失去第二个L0B
预取槽，不能仅按面板宽度宣称更快。未采阶段探针，不将退步精确归因。

QK全有效块CPU首版qk_full_cpu_v1在PTOAS失败：row_min的1x1 INT32
列主序结果只有4B，未满足32B对齐。v2把同一64行索引复制为8个相同行
再规约，结果8x1满足对齐，仍只读第0行；未改编译器，完整CPU编译中。


## 00:24：QK全有效块CPU v2通过并提交B16

qk_full_prepare_v2/qk_full_cpu_v2完整代码生成PASS；使用8x64 INT32
规约输入，8x1结果满足32B对齐。全部input/工作区/等待表保持。原回退
循环在还原结果变量名后AST逐节点一致，8种block/lane组合均覆盖且仅覆盖
当前64行，列索引处于0..511。以上不是设备fallback分支验证。
提交task_20261017_002425_25613488606 / b16_cached_qk_full_v2，
仅在当前815.84μs配置上加--qk-full-blocks，不叠加N128候选。
现有七档性能策略仍保持，未无改动重测。


## 00:28：QK全有效块设备PASS但未获收益

task_20261017_002425_25613488606完成exit=0，完整exact/状态/TopK/
图重放/信号代际与保护区检查均通过。单SPMD三窗840.78/839.56/829.12μs，
中位839.56μs；同轮动态894.94/887.30/901.92μs，中位894.94μs。
未超过现选815.84μs，不扩七档，不设默认。新块级规约和展开改变流水/
代码量，不能仅用减少分支数量推断收益；当前未做精确阶段归因。

现有stage_clock总时长无法区分Indexer和QK/PV的Cube/Vector相互等待。
新增internal_wait_clock.py，仅当前测试进程包装4个对应阶段里的
wait_flag_dev，累计Scalar等待SYS_CNT/次数，并记录该阶段耗时。使用每核
原512B进度区的第2/3条缓存行，不碰信号行，末尾DCCI+DSB发布观测值；
不改变原同步、数学或共享安装。只作诊断，不计入700μs性能目标。
CPU变换审计internal_wait_probe_cpu_v1识别Indexer AIC/AIV的2/1个静态
等待点、QK AIC/AIV的2/2个等待点。其他流水可在Scalar等待时工作，
所以累计等待不是整核空闲，也不能直接当成可消除时间。


## 00:32：提交一次内部FFTS等待归因

任务task_20261017_003153_273890930040 / b16_internal_wait_clock_v1，
基于现选815.84μs的N64+QK地址缓存，不启用两项退步候选，仅增加
--internal-wait-clock。保留全部正确性门槛；插桩结果不参加达标判定。


## 00:37：HC行分工只读审阅与记录归档

已有B16阶段探针中hc_widen_rms与mix_x_rms_norm各12个Vector分片，
数学+维护+发布中位13.48/15.29μs。记录2/4行分块的可行性和约束于
HC_ROW_SPLIT_REVIEW.md：不能复用唯一tail区给多个新满块写；所有输入
和输出有效形状须相应更新，inv_rms不得跨块覆盖，数学K顺序保持。
尚未实现、未编译、未测性能；不把12到48分片当成4倍加速承诺。
本轮文本已镜像到canonical同名目录，大生成物仍在独立worktree，
LOCAL_ARTIFACTS.json只有路径和大小，不做hash。


## 00:39：内部FFTS探针PASS，定位两段的主导侧

task_20261017_003153_273890930040完成exit=0；全部exact/图重放/状态/
TopK/代际/保护区通过，eligible_for_700us_target=false。探针完整L1
837.10μs，只作旁证，不替换无插桩815.84μs。末次重放按成员中位数：

|阶段|侧|阶段耗时μs|Scalar内部等待μs|等待次数中位|等待比例中位|
|---|---|---:|---:|---:|---:|
|Indexer|AIC|214.75|16.34|22|7.70%|
|Indexer|AIV|200.52|74.37|22|37.34%|
|QK/PV|AIC|150.54|109.97|40|73.10%|
|QK/PV|AIV|156.71|4.64|40|2.95%|

证据b16_internal_wait_clock_v1/internal_wait_clock/waits.json，保留每核
原始值和C++插桩审计。比例为逐核比例的中位，不是两列中位相除。
该计时包含读SYS_CNT的开销；Scalar等待期间其他pipe可能运行，且一次
探针样本不足以构成稳定性能统计，不能把等待全部当成可消除或整核空闲。
但它支持优先检查Indexer Cube侧、QK/PV Vector侧的工作和相互供给。
下一步可将QK/PV AIV再分KV搬运、softmax、PV合并/输出几段定位；
无需再扫统一通知退避或原样复测N128、全有效块、七档。

本轮所有CPU会话和三个设备任务均结束；没有待收取任务。当前七档选择
不变，B16最好815.84μs，700μs目标未完成。


## 2026-10-17 00:50：QK AIV分段探针通过，定位编译器插入的交接等待

新增qk_phase_clock.py，在原生成C++的KV搬运、softmax、PV合并、
最终输出分支内累计SYS_CNT，不插入中间pipe屏障。CPU边界审计
qk_phase_probe_cpu_v1通过；task_20261017_004440_298994623828 /
b16_qk_phase_clock_v1完整PASS，eligible_for_700us_target=false。
末次重放48个AIV成员中位：阶段161.82μs，KV58.73/20次，softmax12.36/20次，
PV合并2.88/20次，最终输出14.92/4次。各段允许异步pipe在段外完成，
不能把差额直接等同某项开销，也不能作为未经插桩性能承诺。

qk_phase_probe_cpu_v1/uncovered_sync.txt保存未包入四段的同步指令：
每轮含两处PIPE_ALL及多处跨pipe交接，邻近有效块metadata的GM Scalar读。
新增qk_mask_cache.py，将本核query的有效块标记每次从原device Tensor
预读UB；B16为4x16 INT32共256B，只改AIV内6处读取来源，保留判定、
数学与FFTS。不直接删除任何同步，不新增Host读取或GM展开缓冲。
首个准备脚本误设读取点数量为7而断言失败；逐AST核对确为6处，改正后
准备qk_mask_prepare_v2。实际编译是否减少PIPE_ALL和性能均仍待验证。


## 01:10：mask候选无屏障收益；确认事件槽回退并提交MGATHER

qk_mask_cpu_v1完整CPU代码生成PASS，但原/候选QK tick内PIPE_ALL均4处，
FFTS wait/set均2处。qk_mask_prepare_v2/address_audit.json覆盖187200项
本核query映射；没有设备性能结论，不因CPU通过就采用，不扩大七档。

qk_sync_diagnostic_v1的module-scope打印要求禁用线程，v2尝试的参数
被PTOAS 0.66拒绝；v3去掉module-scope但debug3输出过大，主动终止
(exit143，日志本地保留)。v4用debug2完整通过。实际原最佳B16 IR在
EventId Allocation之前的依赖2708/2709/2712为V→MTE2，2714为MTE3→V，
之后回退为PIPE_ALL；另2687/2688/2689在PV入口回退后合并。
原始IR内无这四处显式屏障。证据qk_dependency_nodes.json/qk_fallbacks.json，
不能直接删这些UB复用依赖，也不能用静态屏障数直接解释所有未测耗时。

新增qk_mgather.py：已有本核UB物理行地址基础上，用MGATHER(row,zero)
替代64次PyPTO逐行控制。负稀疏索引用physical|min(raw,0)保留符号位，
在物理行数<2^31的前提下是越界哨兵；合法行地址不变。ISA仍逐行DMA。
prepare_v1仅qk_pv_aiv变化，其余72个函数不变，60项符号边界CPU检查通过。
CPU v1因保留跨SWA/压缩分支的64KiB初始零Tile，UB200960>188416B失败。
v2将零Tile作用域收至SWA分支，压缩分支由MGATHER完整写出，CPU完整通过；
生成QK区段PIPE_ALL为5处而不是原4处，不宣称减少同步。

任务task_20261017_010828_337615615012 / b16_cached_qk_mgather_v2，
使用当前最佳的earliest_ready/cached100/QK地址缓存，只加MGATHER。
额外qk_mgather_boundary.py在计时外检查64x512 BF16全有效、混合负索引、
全负索引；固定地址图重放时更新device索引并检查输出保护区。
保留整层原exact/TopK/六状态/图/代际/保护区门槛，结果待收取。
没有修改依赖仓或生产入口。


## 01:14：MGATHER完整PASS，无性能收益；准备HC行分工候选

task_20261017_010828_337615615012完成exit0。独立64x512 BF16边界检查
在同一图更新device索引，全有效/混合负索引/全负索引均exact和保护区PASS。
完整CSA输出、TopK、六份状态、图重放、30代信号、Native/工作区保护区PASS。
无插桩L1单SPMD三窗827.18/821.62/820.38μs，中位821.62；同轮原动态
902.20/911.62/911.18μs，中位911.18。没有优于现选815.84，不采用、
不扩七档；证据qk_mgather_result.json和原任务结果。

沿HC_ROW_SPLIT_REVIEW.md实现hc_row_partition.py：只改变hc_widen_rms
与mix_x_rms_norm的逻辑行分片，物理8行Tile及K顺序保留；2/4行可分档。
B16/T96的两个阶段各12→48成员；widen_rows、block_num、member_count
一起更新，再重建分核/完整等待表。显式限制inv_rms有效写行数，pre/mixed
load使用有效行，满块判断改到新行数，避免多核共写唯一tail区。
候选仅允许T%8==0，其他规格仍走原8行实现，不将padding初始化混入本实验。
Scalar循环不随行数缩短，初始调度成本仅估计为原0.5，待实测；没有承诺4倍收益。
hc_rows2_prepare_v1正在准备，未编译/未提交设备，默认hc_rows仍8。


## 01:19：HC行分片CPU通过，提交B16

hc_rows2_prepare_v1脚本把kernel_names误当单元素列表，实际为带None的
三槽位数组，导致自己的断言失败；v2按非空kernel名集合匹配。
hc_rows2_prepare_v2与hc_rows2_cpu_v1完整CPU/PTOAS生成PASS。
address_math_audit.json确认仅两个函数变化，所有for/pipeline表达式相同，
T=8..384步8、每档2/4行共96种行映射逐行无遗漏/重复。

任务task_20261017_011729_35529046475 / b16_cached_hc_rows2_v1已提交，
保留当前最佳earliest_ready/cached100/QK地址缓存，只启用--hc-rows 2。
没有叠加MGATHER等退步候选；全部原正确性门槛保留，设备结果待收取。


## 01:23：HC 2行分片完整PASS，但整层退步

task_20261017_011729_35529046475完成exit0；整层exact/TopK/六状态/
图重放/30代信号/Native及工作区保护区全部PASS。单SPMD无插桩L1
842.48/843.20/850.10μs，中位843.20；同轮原动态893.34/889.28/892.56μs，
中位892.56。未优于现选815.84，不采用、不扩七档；不能把核数增加当收益。
保留--hc-rows 2/4为实验开关，默认8；本轮没有测4行，也不冒称已通过。

新增indexer_page_views.py候选：原N64的32个Key页加载奇偶分支，分别
使用未偏移和偏移64B视图。改为从相同原缓冲按device byte offset切出
4096B再reshape[32,128]，交给原gather_row写L1。页号夹取、QK/WS/FFTS
和所有数学顺序保持；无Host读、无新GM分配。只是准备CPU编译，尚无设备结论。


## 2026-10-17 01:34：页视图CPU失败，核实并消除QK重复清零

indexer_page_views_prepare_v1的地址审计及98个数学调用顺序检查通过，但
indexer_page_views_cpu_v1完整编译失败：InCore中的tensor.slice会lower为
Tile加载，gather_row要求GM Tensor，不能把该写法当零拷贝GM子视图。
没有提交NPU，没有修改PyPTO或越过Tensor窗口边界规避检查。该开关仍默认
关闭，保留失败源码/日志供参考；不宣称算子或性能通过。

重新核对当前最佳B16的真实生成C++：QK KV分支外有一处TEXPANDS，
压缩分支内还有一处TEXPANDS，压缩块确实执行两次64KiB清零。新增
qk_scoped_zero.py，只将前一处移入SWA分支；压缩块保留自己的清零与
原逐行gather/负索引处理，所有数学/FFTS保持。不使用此前MGATHER或
全有效块候选。qk_zero_prepare_v1正在准备，尚无编译或设备结论。


## 01:37：QK初始化作用域CPU通过，提交完整B16

qk_zero_prepare_v1/ast_audit.json确认仅qk_pv_aiv改变，其余72个函数
AST不变；把初始化移回原处后，整个函数AST与原版完全相同。
qk_zero_cpu_v1完整CPU/PTOAS代码生成通过。codegen_comparison.json
确认压缩块由连续两次清零改为分支内一次；整阶段PIPE_ALL静态数未变。

task_20261017_013726_407690029470 / b16_cached_qk_zero_v1提交。
沿用earliest_ready、原动态成本、cached100、QK地址缓存；只增加
--qk-scoped-zero，不叠加其他退步候选。结果未回收前不预记通过/提速。


## 01:42：仅消除重复清零完整PASS，无性能改善

task_20261017_013726_407690029470完成exit0，完整exact/TopK/六状态/
图重放/30代信号/Native及工作区保护区PASS。单SPMD无插桩L1三窗
817.22/822.98/826.50μs，中位822.98；同轮动态897.20/906.32/893.42，
中位897.20。未超过现选815.84，不采用，不扩七档。

继续qk_valid_gather.py：先在device确认64个原始索引全部非负，全有效
块用create未初始化Tile，8行展开覆盖全部64行；否则保留原清零和原
64行条件循环。SWA仍在自己的分支清零。与旧full_block候选的具体区别
是全有效路径由两次清零降到零次，负索引路径为一次；不是原样重跑旧候选。
新增qk_valid_gather_boundary.py，从实际候选AST提取规约与两条搬运分支
复用，拟检查同一图全有效→混合负索引→全负索引→全有效往返。
正在准备CPU与编译，尚无设备或性能结论。


## 01:45：全有效块首版UB超限，改成分支共用目标缓冲

qk_valid_prepare_v1确认只有QK AIV变化，原负索引循环AST完全保留
（仅约定的结果变量改名），8×8展开覆盖64行，无重复或遗漏。
qk_valid_boundary_cpu_v1独立边界代码完整编译通过，但整层
qk_valid_cpu_v1失败：Vec 201472B > 188416B。没有提交NPU。

V2在有效性分支前create共同64KiB目标；全有效分支直接覆盖，负索引
分支先tile.expands(0)再执行原循环。expands广播标量，不读取未初始化
数值；不使用“乘零”规避初始化。SWA仍独立清零。只调整实验源码以表达
共用目标，不修改规划器或容量检查。qk_valid_prepare_v2正在准备。


## 01:48：共用目标V2的字面量类型修正

qk_valid_cpu_v2与qk_valid_boundary_cpu_v2均在类型检查失败：
tile.expands(target, 0.0)的标量0.0被推导为FP32，返回Tile与BF16
注解不符。这发生在内存规划之前，不能声称共用目标已经解决容量。
V3显式将零标量cast为BF16，再广播；保留V1/V2源码与失败日志。
只修正候选写法，不修改PyPTO规则。qk_valid_prepare_v3准备中。


## 01:51：V3仍超UB；不用缺少后端的expands，改现有原地填充

qk_valid_cpu_v3仍报Vec 201472>188416B；显式BF16只解决类型，普通
expands并不保证别名，不能据此宣称两路径共用分配。独立
qk_valid_boundary_cpu_v3另报No codegen registered for operation: tile.expands。
源码/日志全部保留，未修改依赖、未提交设备。

V4使用既有tile.fillpad_inplace(tile.set_validshape(blank, 0, 512))。
PyPTO注册明确set_output_reuses_input(0)，并将结果有效形状恢复为完整
物理形状；当前A3 ISA在srcValidRow=0、srcValidCol=512时填满底部64行，
没有读取有效业务行。这里使用原地填充的已有语义，不降低容量阈值，也
不手改生成C++。实际代码生成与负索引图重放仍须验证，不能只凭源码判PASS。


## 01:53：原地填充的独立编译通过，整层局部变量类型分离

qk_valid_boundary_cpu_v4完整编译通过。qk_valid_cpu_v4在parser失败：
同名stage_30_v255原为无pad标记Tile，原地填充产生pad=zero Tile，
被判断为不同类型的重赋值。尚未到内存规划，不能预记容量已解决。
V5只把压缩分支的初始局部值改名qk_cmp_tile_start，区别于SWA中的
旧同名SSA；保留原地填充、所有搬运和数学。负索引循环审计允许此初值
及原有结果名两处重命名，其余AST必须逐节点相同。


## 01:57：V5的快速/回退初值仍同名，V6分别命名

qk_valid_cpu_v5仍在parser失败，定位到全有效与回退分支共享
stage_30_v302，分别为pad未指定与pad=zero类型。此前只区别于SWA
还不够；V6将三分支初值全部区分，快速分支qk_valid_tile_start、
回退分支qk_fallback_tile_start，内存仍来自同一qk_valid_blank。
没有修改数学或有效性条件，当前qk_valid_prepare_v6在准备。


## 02:00：V6分支结果的pad类型冲突，V7用恒等视图统一

qk_valid_boundary_cpu_v6完整编译通过；qk_valid_cpu_v6仍在parser失败，
这次是融合生成器展开yield后的共同结果stage_30_v303，两个分支的pad
标记不同。初值改名只解决局部重赋值，不能解决共同结果类型。
V7给全有效与SWA分支结果添加64x512、offset[0,0]的恒等tile.slice，
pad_value=zero，统一为与回退相同的类型。三条路径本来都是完整有效Tile，
此处无无效区域，不需要实际填充或拷贝；负索引的真实清零仍由原地填充
完成。实际生成代码需确认未新增搬运/清零，再决定是否上卡。
独立与整层失败/通过范围继续分别记录，不把小例通过算整层通过。


## 02:04：全有效块V7完整CPU通过，提交一次B16

qk_valid_cpu_v7与qk_valid_boundary_cpu_v7完整代码生成PASS。
qk_valid_prepare_v7/codegen_comparison.json：全有效分支8个TLOAD展开点，
没有TEXPANDS/TFILLPAD；SWA保留一处TEXPANDS，负索引保留TFILLPAD。
整个QK AIV阶段静态PIPE_ALL由9增至11，不把清零减少直接说成性能收益。
候选的UB容量已通过原规划器校验，没有手改地址或改依赖。

提交task_20261017_020426_59480118626 / b16_cached_qk_valid_v7。
使用现选earliest_ready/cached100/QK地址缓存，只增--qk-valid-gather。
计时外从实际候选AST提取边界片段做同图4变体往返；整层保持原exact/TopK/
六状态/图/30代信号及保护区闸门，性能为原3个L1窗口。结果待收取。
所有V1～V7的准备和失败产物保留，前面的CPU会话均已收取终态。


## 02:09：全有效块完整PASS，839.84μs，仍不采用

task_20261017_020426_59480118626完成exit0。实际候选片段的同图
全有效→混合负索引→全负索引→全有效四变体，输出exact、只读输入和
保护区PASS。完整CSA输出、TopK、六状态、图重放、30代信号和Native/
工作区保护区全部PASS，未放宽任何准确性门槛。

单SPMD无插桩完整L1三窗839.84/838.18/841.76μs，中位839.84；
同轮原动态899.10/890.14/890.88，中位890.88。没有优于现选815.84，
不采用、不扩七档，不重复同一候选。codegen里额外两个PIPE_ALL是已见
差异，但未做阶段探针，不能断言它单独解释全部回退。

本轮新增qk_scoped_zero.py、qk_valid_gather.py、qk_valid_gather_boundary.py，
只在实验目录接入默认关闭的开关；生产入口、冻结静态、PyPTO和Simpler均未改。
全部CPU与NPU任务结束，没有待收结果。七档当前选择不变，700μs目标仍active。


## 2026-10-17 02:23：同一数学体独立QK编译未减少循环屏障

qk_isolation_cpu_v1 完整CPU编译成功，选中现选版本的同一QK AIV数学体，
局部编译后该阶段仍9处PIPE_ALL，实际QK tick仍4处；循环文本约160205→158220B，
set/wait各69→73。没有从原动态另一数学体的3处PIPE_ALL作因果推断。
证据comparison.json；不提交这个负候选NPU，不删除cached_wait/独立编译组合的安全限制。

## 02:30：Indexer AIC提交区间探针通过

indexer_phase_clock.py 只包围原32个页表读取至Key DMA提交窗口和1024分数块循环，
另计原wait_flag_dev，不给普通异步wait_flag虚构Scalar等待时间。保持原中间流水屏障。
CPU提取审计indexer_clock_prepare_v1通过：32个静态页加载点、1个分数块循环、
2处FFTS等待；每核512B进度区的第二行记录，首行信号不动。

task_20261017_022628_11712641926 / b16_cached_indexer_clock_v1 完整PASS。
24个AIC中位：阶段210.23μs、Key提交累计11.81μs/704次、分数块193.96μs/22次、
FFTS等待15.00μs/22次。普通wait是异步pipe指令，区间会重叠且插桩扰动队列，
不能简单相减成计算/搬运独占耗时。完整插桩L1为845.64μs，不用于700μs目标。
精度/TopK/六份状态/图/30代信号/保护区全部PASS。页地址分支暂不继续优先优化。

## 02:35：完整CSA同地址动态metadata A→B→C→A通过

dynamic_metadata_replay.py 与 --metadata-replay-only：固定B16/S6、页表容量、
全部根参数/工作区地址和每侧捕获的一张图；只在测试准备时改设备Tensor内容。
A=131072，B=131071并交换请求物理页，C=130048并交换页，最后恢复A。
位置、长度、页表、slot、RoPE共19个根输入变化；B/C参考输出确实不同，A恢复完全相同。
每组先运行原动态PTO eager，再重放原动态图和单SPMD图；全部输出、Top-K、六份
cache/state逐bit一致，完整allocation未写区/metadata只读检查通过。

task_20261017_023022_12848062458 / b16_metadata_replay_v1 完整PASS，
无新增编译单元；6代进度准确且变体之间未清零信号，工作区保护区PASS。
这次只做功能，没有Event/L1重复采样。证据dynamic_metadata_result.json及
b16_metadata_replay_v1/metadata_replay/replay.json。尚不代表连续decode、
异构请求长度或生产自动派发，也不要求跨batch共用二进制。

## 02:36：下一候选改为小档O-A的M/K预取分块

oproj_k512.py：T<=96时M按16对齐，保持N128、原逻辑分片、全局视图，
将外层K预取256×16改为512×8，拟保留内层K128及其顺序。较大档仍原策略。
先只准备/编译B16；需核对实际内层代码和内存容量，尚未上卡，不预记性能收益。


## 02:39：O-A K512完整CPU通过，提交性能；另补短历史动态metadata

oproj_k512_prepare_v1与oproj_k512_cpu_v1通过。实际C++左/右矩阵内层仍K128，
同一个Acc覆盖K=0,128,…,3968，外层8个K512块；M96/N128，单阶段PIPE_ALL仍1。
代码审计codegen_audit.json；解析C++常量必须限定AIC函数，不能与AIV同名局部值混合。

task_20261017_023822_152092626476 / b16_cached_oproj_k512_v1已提交，仅B16性能候选。
按既有5次预热/20次Event/3个L1窗口及完整exact检查；没有更改其他档位选择。
短历史的Indexer数学分支不同，另提交B16/8192的同地址A→B→C→A功能检查，
沿用该档原critical_path/cached100策略，不开QK地址缓存、不采性能、不套长历史二进制。


## 02:42：O-A K512完整PASS但没有更快，不扩大测试

task_20261017_023822_152092626476 / b16_cached_oproj_k512_v1 完成exit0。
输出、Top-K、六份状态、图、30代进度、全部保护区继续通过。
无插桩单SPMD L1为827.94/833.74/825.80μs，中位827.94；同轮原动态885.68μs。
没有超过保留815.84μs，因此不扩测其他档，不更改selected_shape_policies。
CPU准备和代码生成证据保留，不能用减少外层加载次数推断整体提速。
短历史动态metadata任务 task_20261017_023946_156511419169 尚在运行，未预记通过。


## 02:45：短历史首版用例跨策略失败，补编排一致性检查

task_20261017_023946_156511419169 已完成exit1。A8192、B8191通过；
C7168在图比较中失败：x_out最大绝对差0.03125、172666个元素不同，
Top-K15576项不同；六份cache/state及所有保护区仍通过。旧探针没有记录
失败的具体graph侧，因此不能据此归因给单SPMD或原动态图。

源码核对：8192+6的压缩长度2049，进入[2048,8192]策略；7168+6压缩长度1793，
已进入另一套score/topk数学分支。当前不要求跨策略共用二进制，原用例错误地
要求复用该捕获图。保留short_metadata_failure_v1.json和原失败日志，不覆盖。

新增metadata_cases用CPU编排模拟器核对原call_signature及selected_control_path，
只使用测试已知Host长度，不回读device。每个待执行变体必须同原策略；
跨策略候选在派发前拒绝并记录。8K档C改为8186（压缩长度恰为2048），
即8192→8191→8186→8192，形状/地址/信号不变；8个更短候选先被CPU拒绝。
short_metadata_strategy_prepare_v2/cases.json记录完整分支证据。
同时失败报告保存active_step和failed_checks，避免再次丢失具体比较侧。
已提交修正版b16_short_metadata_replay_v2，不预记PASS。


## 02:49：短历史策略内动态metadata完整通过

task_20261017_024535_170390925040 / b16_short_metadata_replay_v2完成exit0。
8192→8191→8186→8192四组在完整CSA上输出、Top-K、六份cache/state逐bit一致，
每侧固定一张图、地址固定、无新增编译单元；6代信号及全部保护区PASS。
8个跨策略长度在CPU准备时拒绝。没有重复任何Event/L1性能采样。
证据short_dynamic_metadata_result.json及该目录metadata_replay/replay.json。

继续CPU候选oproj_phase_loop.py：原O-A八个相邻阶段只在group、源行/目标列
基址不同，逐AST正规化后八体完全相同。按原顺序[6,5,7,4,1,3,0,2]循环，
每次仍用原record/wait表，不删等待/通知/缓存维护，不改变任何数学Tile。
原JIT文本587476→569145B，尚不能推断机器码大小或性能。CPU编译已启动，
不改PyPTO/Simpler，不把不同阶段外部事件拼接成独立device函数。


## 02:52：O-A阶段循环完整CPU通过，提交B16

oproj_phase_loop_cpu_v1 完整编译通过。AIC生成C++ 962738→753992B，
静态PIPE_ALL/TWAIT/TNOTIFY位点35/34/35→28/27/28，因为八段共享循环体；
实际动态执行次数与原顺序相同，不把静态位点减少冒充删除同步。
逐AST检查八体除group/源行/目标列外一致；原record基址按432步长保留。
提交b16_cached_oproj_loop_v1，仅在现选B16上检查完整exact/图/信号/保护区及性能。


## 2026-10-17 02:56：O-A阶段循环完整PASS但未改善整层，全部任务已收取

task_20261017_025159_192231313200 / b16_cached_oproj_loop_v1 完成exit0。
全部exact/TopK/六份状态/图/30代进度/保护区PASS。无插桩L1三窗
[835.94, 826.24, 830.74]，中位830.74μs，同轮原动态903.74μs。
AIC bin由66832降至54900B，AIV保持153408B；没有超过已选815.84μs。
因此不采用、不扩其他档，也不把源代码或二进制缩小当成性能收益。
本轮所有CPU session和NPU任务均已收取，没有待等待任务。


## 2026-10-17 03:20：QK按32行分两次准备KV候选，边界包装修正中

用户再次明确：允许batch/长度分档与精确特化，目标是各档均有较好性能；参考hc_pre的编译期分块与入口分派，不要求共用二进制。现选七档不变。

qk_chunked_gather.py保持64行GM交接布局、原通知/等待和QK/PV数学，将AIV本地BF16[64,512]改为两个[32,512]。SWA与压缩分支均分块，压缩metadata仍一次准备64行。CPU地址枚举1696个SWA场景和压缩行序检查通过，qk_chunk_cpu_v1完整代码生成通过。实际tick代码不再有64行KV Tile；PIPE_ALL静态位点4→5，不能据此声称速度提升。

qk_chunked_boundary.py直接提取候选生产片段，拟用同图覆盖SWA短窗口、页边界、负页、长位置，压缩有效/负索引、分支返回及保护区。首轮CPU边界编译qk_chunk_boundary_cpu_v1失败：打印IR中的t__tile_1为保留命名；同时提交的task_20261017_031425_239558810336/b16_cached_qk_chunk_v1因同一问题exit1，完整CSA与性能均未运行。v2对片段局部变量/形参作无碰撞重命名，CPU进一步暴露嵌套incore调用未内联（No codegen registered for operation: _gather）；测试辅助函数改为jit.inline，CPU_v3进行中。以上只修测试包装，未改PyPTO/Simpler。


## 03:23：QK边界CPU通过等待真机；准备HC流式候选

qk_chunk_boundary_cpu_v3完整代码生成通过，jit.inline使实际候选片段在边界入口内联。task_20261017_031908_248319030873/b16_cached_qk_chunk_v2已提交，03:23仍pending，未预记PASS或性能。

hc_stream_pre.py在实验生成源码中改写三个HC阶段，仅接收T96/24组/原M16-K4096四分片。48颗AIV按组内8行×K512加宽，每个K512的row_sum写入原T96闲置尾部暂存的12288字节。AIC用K512 L1供数、两个原K256 matmul依次累加，仍输出4份K4096部分和；原linear_reduce额外按0..31顺序恢复每行RMS。其余HC门控、Sinkhorn、归一化与CSA阶段不变。

原AIC等待全部12个加宽任务的GM轮询被组内FFTS READY8/9与ACK10/11替换。全部48个AIV参与，原12个进度发布记录保留。两槽限制未消费通知，AIV末尾排空两个ACK；不新增Scalar GM信号，也不删后续各阶段数据维护。CPU检查原加宽无前驱、原AIC仅依赖加宽、reduce等24个AIC、两个inv_rms消费者等全部reduce。128个随机调度×3代事件模型无未消费通知覆盖/死锁/残留；这不证明硬件pipe可见性。hc_stream_prepare_v1已生成，hc_stream_cpu_v1编译进行中。尚无真机精度/性能结论。


## 03:29：QK32行候选完整PASS但未提升；HC自动pipeline候选不上卡

task_20261017_031908_248319030873/b16_cached_qk_chunk_v2完整PASS，8项同图边界、输出/TopK/六份状态/图/30代进度/保护区全部通过。无插桩L1为831.32/828.92/838.78μs，中位831.32；同轮原动态886.70。没有优于保留815.84，未采用，不扩其他档。证据qk_chunk_result.json。

HC CPU_v1因把GM store结果重新赋给根参数导致MIX整参别名检查失败，改为原地store表达式，无依赖仓修改。v2的动态事件ID在PTOAS产生作用域错误；v3改为偶/奇静态8/9与10/11，CPU编译通过。但审计两版PTO IR均发现pl.pipeline重排：输入tload先于对应READY wait，生成C++也保留错误顺序。v2/v3都没有上卡，不以编译PASS冒充交接正确。证据hc_stream_pipeline_order_audit.json。v4把两侧手工FFTS循环改为pl.range，RMS纯数学累加仍可pipeline；CPU_v4在编译，待检查READY→输入加载→ACK及双store→READY顺序。


## 03:31：HC显式循环CPU与交接顺序通过，提交B16整层

hc_stream_cpu_v4完整代码生成PASS。逐IR与C++核对READY wait→输入TLOAD→MTE2 ACK，AIV输入/平方和两次TSTORE→MTE3 READY，且末尾等待两个ACK。首个自动审计用固定前后行数误计前一RoPE的store，断言失败；按实际C++循环花括号限定后通过，未改变候选或放松顺序要求。codegen_audit.json保留位点。task_20261017_032959_27383327818/b16_cached_hc_stream_v1已提交，沿用原exact/保护区/图/30代及3窗L1；未预记PASS。


## 2026-10-17 03:35：HC流式版完整PASS但未提速，本轮任务全部收取

task_20261017_032959_27383327818/b16_cached_hc_stream_v1完成exit0。输出、TopK、六份状态、图、30代进度和全部保护区PASS，未放松exact要求。无插桩L1为834.18/844.40/847.60μs，中位844.40，同轮原动态907.64。没有优于保留815.84，未采用、不扩其他档；不能把功能正确、48 AIV参与或开始供数更早当成提速。没有额外采阶段探针，暂不能将整层差值定量归因给ACK、RMS临时区或RoPE前置顺序。

本轮新CPU/NPU任务均已收取。QK分块和HC流式代码、边界、失败、生成代码审计及性能结果均保留。现选七档与生产入口不变；无PyPTO/Simpler修改。目标active，700μs未达到；不存在外部阻塞，不标暂停/完成。

## 2026-10-17 03:47：Indexer叶分片容量实验已提交

新增indexer_leaf_capacity.py/--indexer-leaf16384，仅T96长历史；原8192候选/叶对应6叶，
不是此前NEXT_STEP中误写的9叶。新16384候选/叶为3叶，同步更新Cube、Vector和最终合并。
每worker仍22个1024分数块，query/系数加载4→2次，pair_arena布局/容量/FFTS协议/分数数学不变。
CPU检验4133种动态长度和每种最多5个有效前缀，覆盖/无重叠/active leaf一致性通过；
indexer_leaf_prepare_v1和indexer_leaf_cpu_v1完整lowering通过。仅CPU证据，未证明TopK tie顺序或精度。
真机task_20261017_034636_30078467438，b16_cached_indexer_leaf16k_v1，待结果。
另只读核对payload_cache.py：此前精确Scalar缓存维护候选已覆盖按阶段读写删DCCI，不重复该负候选。
scalar_gm_access_audit_initial.json仅为源级访问清单，不是新的同步正确性证明。

## 03:53：16384候选/叶完整通过，未纳入当前选择

task_20261017_034636_30078467438 exit0；输出、TopK、六份state/cache、图重放、30代信号及保护区均通过。
L1三窗829.22/829.42/813.78μs，中位829.22；同轮原动态897.84μs。
未证明优于现选历史815.84μs，保持七档选择不变，也不按最小值813.78替代中位统计。
现选SPMD与候选没有同进程A/B，不能把历史差值当作确定回退幅度。indexer_leaf_result.json。
后续新候选indexer_score_pingpong.py：M384/N64分成两份M192/N64，使两份QK INT32分数同时活跃，
合计96KiB Acc，原叶数/页访问/1024分数交接/Vector TopK与FFTS保留。与已测N128的区别是
单份Acc由96KiB减到48KiB，两份在源级有重叠生命周期，且Key预取至Right。尚待实际地址/同步与设备验证。

## 03:55：Indexer双Acc的完整CPU编译与地址核对通过

indexer_pingpong_prepare_v1与indexer_pingpong_cpu_v1完成；16组两份INT32 Acc各49152B，
实际地址0与49152不重叠。C++中每组两次TMATMUL均位于第一份TINSERT/FIX之前。
codegen_audit.json记录逐组地址与C++变量；这是地址/提交顺序证明，不是硬件并发或提速证明。
首次审计按TMOV匹配量化写回导致断言失败；实际lowering为TINSERT，改审计匹配后通过，算子代码未为此变化。
未用固定MemRef，也未删同步：通过两份结果同时活跃阻止Acc别名。
task_20261017_035437_31474978052已提交，输出b16_cached_indexer_pingpong_v1，先严格exact后计时。

## 2026-10-17 04:03：双Acc完整PASS但性能回退，本轮收尾

task_20261017_035437_31474978052 exit0，30代信号、输出、TopK、六份cache/state、图重放和保护区通过。
L1三窗968.92/965.56/972.00，中位968.92；同轮原动态904.72。不采用，不扩其他档。
对实际NPU构建再次执行只读codegen地址与提交顺序检查，16对Acc仍不重叠，提交顺序保持。
生成Indexer AIC静态位点TMATMUL32→64、TINSERT16→32、TSTORE16→32、set/wait166→187，PIPE_ALL仍1。
不能把这些静态数直接换算为耗时，无新探针定量解释回退。INDEXER_PARTITION_EXPERIMENTS.md汇总两项。
只读审阅HC前端reduce/split/comb/mix原函数与现有阶段时戳；尚未实现新融合。
全部CPU/NPU任务已终态，依赖/生产/冻结静态不改；七档选择保持，goal active，700目标未达。


## 2026-10-17 04:30：PV分head合并被UB上限拒绝，直接L1供数候选开始验证

qk_merge_heads.py尝试将每AIV的32-head PV合并拆为16-head，再assemble原状态。
qk_merge_heads_prepare_v1及cpu_v1：规划仍产生196608B Vec，超过188416B上限，未上卡。
prepare_v2只生成8-head源码，没有编译或设备执行；看到全状态生命周期证据后没有继续盲扫。
prepare_v3/cpu_v3改为先完成left、再处理right，仍196608B超限。两版CPU失败原样保留，
没有改编译器或扩大硬件容量，没有精度/性能结果，七档选择不变。

新qk_direct_l1.py让Cube直接从Native KV到三槽L1。Vector批量把私有valid-plan索引临时区
改成物理行（外部TopK不改），并准备每组32行零源。一次FFTS4发布后Cube维护Scalar缓存再读。
原事件0从KV_READY改为两AIV完成PV读取的确认；Cube每三槽复用前等待，调用末尾排空。
原score/probability/PV事件1/2/3和全部点积、softmax、归约顺序保持；无新Host读取或根参数。
该候选尚未完整设备验证或计时，不能据此宣称提速。

qk_direct_l1_probe_cpu_v1完整lowering通过；生成C++子视图仍保留父Mat Rows=384，
validRow=1/动态，地址按实际行偏移，因此没有把NZ列跨度错误缩成16。
真实A3小用例首轮task_20261017_042346_360920825849缺少pypto.torch.init，exit1，
没有执行kernel；补测试初始化后task_20261017_042437_362436518114 exit0。
qk_direct_l1_probe_device_v2同一图7项全部PASS：三个槽逐行gather、非对齐起点、动态
有效行数、最后槽最后一行以及返回首槽；全部三槽数据、相邻行/槽、GM保护区/只读输入通过。
只是搬运接口证据，不代表完整CSA正确。未采性能、未运行Native。

qk_direct_l1_protocol.py的CPU交接模型通过3328个调度、322443步、2/3代复用：
枚举两query全部1024种有效块mask并随机两AIV进度，另外检查4query；确认事件按代排空、
score/probability/KV/PV三槽未提前复用，无死锁。1310种SWA长度/负页模式的区间覆盖通过。
模型不证明真实DMA或缓存语义。直接L1完整CPU正在qk_direct_l1_cpu_v1编译，随后核对实际
生成顺序；所有改动只在本实验目录，冻结静态/生产/依赖仓均未改。


## 04:31：直接L1完整CPU与提交顺序通过，上卡验证

qk_direct_l1_cpu_v1完整lowering PASS。IR与实际C++确认：地址/零源store后发布FFTS4；
Cube等待4后fence/cacheinvalid/fence，再开始Scalar地址读取；三槽复用等待0在新QK之前；
两份PV的TLOAD都在MTE2发布0之前；最后一个ACK在退出QK前排空。
qk_direct_l1_prepare_v1/codegen_audit.json保存行号与原文。保留原矩阵乘pipeline，手工交接
使用pl.range，没有出现上轮HC自动pipeline将TLOAD移至READY之前的问题。
任务task_20261017_043021_371809025343/b16_cached_qk_direct_l1_v1已提交；
只比较原动态PTO，固定正式权重fixture，原exact门槛先行，未预记PASS或性能收益。


## 2026-10-17 04:38：直接L1完整通过但明显退步，本轮全部收取

task_20261017_043021_371809025343 exit0。正式权重固定fixture的输出/TopK/六份状态、
图、30代信号与保护区全部PASS。单SPMD L1三窗1096.08/1100.10/1100.20μs，中位1100.10；
同轮动态892.66μs，增加23.24%。明确不采用、不扩其他档，也不追加该负候选的边界/metadata重放。
历史选定815.84μs仍保留；没有选定SPMD同进程A/B，不能把历史差值当成精确回退量。
qk_direct_l1_result.json汇总；实际NPU C++与已核对CPU C++全文相同（文本比较，无hash）。

只读代码归因：AIV整个QK阶段PIPE_ALL静态位点9→5，但Cube逐行搬运变多。
真实compressed128行循环内没有pipe_barrier/set_flag/wait_flag；不能说每行多了屏障。
原两AIV各64行搬运改为单Cube128行ND2NZ，Scalar地址也改从GM读取。这些结构变化
可能增加延迟，但未采内部探针，不能将全部回退定量分配给它们。
第一次按record注释界定C++阶段失败，改用实际花括号边界后生成codegen_comparison，算子未改。
另只读核对当前A3 TPush/Tpop Vec→Mat使用GM FIFO，不实施接口换名式尝试。

下一方向仅做只读初查：query_dependency_inspection_initial.json记录现选控制表中
QK Cube等待17个前驱、valid-plan47个、TopK merge71个、首批部分O-A71个；并不证明可删。
需要先映射数学读写范围，再决定是否细化，不能把当前按核独立推进误说成全卡barrier。
本轮CPU和NPU均终态，无共享依赖、生产或冻结静态改动。goal仍active，700μs未达。


## 2026-10-17 05:06：按行细化成员依赖与同图变长复核

新增row_dependencies.py，严格限定T96长历史，实际Tensor别名/行所有权与FFTS阶段复用有显式契约。
原数学、物理分工、核内顺序和发布/cache维护不变；四类完整task依赖改为成员依赖，
所以不是原task偏序不变。等待16729→12183，发布730→722，细化图约简/翻译后祖先关系一致。
1536个merge worker×leaf_count组合覆盖96query、所有1～32leaf，CPU证明不替代设备。
原生成函数体AST一致；固定成员成本最长链748.55→748.53只是离线分析，不作性能预测。

- task_20261017_044912_40225339550 / b16_cached_row_deps_v1 exit0完整PASS：
  新807.16/804.40/811.26，中位807.16；同轮动态905.08；30代/保护区通过。
- task_20261017_050033_181823706 / b16_row_deps_paired_metadata_v1 exit0完整PASS：
  同进程旧候选841.94/843.14/833.20，新候选806.84/817.88/821.56；中位缩短2.86%。
  同一任务候选图131072→131071→65536→131072，6leaf变3leaf再恢复；
  原动态eager/图与候选图逐bit一致，地址/无新编译/11代/保护区通过。
- paired_row_dependency_perf.py仅为本轮对照辅助；动态metadata增加显式额外history选项，
  先CPU检查原编排call_signature和selected_control_path，无device回读。
- 选用行依赖候选，仅B16长历史。其他档不重测、不套该表。
- 实现、范围和证据见ROW_DEPENDENCY_EXPERIMENT.md；未改生产/F/依赖仓。


## 2026-10-17 05:10：Indexer N80容量与流水检查

indexer_n80.py保持原QK M384/K128、WS M16/K384、FIX量化/ReLU、1024列分数传输、
叶计划及FFTS/TopK AIV。仅N64→N80，13个panel（末64列有效），matmul32→26。
每个80列panel按Native32行页分成32/32/16或16/32/32段，总39段，原来32段。
Native带scale页的0/64字节偏移视图保留；禁止把scale区域连续当Key读。

indexer_n80_prepare_v1与完整CPU生成indexer_n80_cpu_v1通过（session30043已收取）。
和已选行依赖组合，原输出区域/叶所有权/FFTS尾排空不变。
实际C++中13个QK Acc均0地址、122880B；前12个WS Acc在122880、末次WS在0复用，
峰值128000B<131072B。末次复用经过FIX→MTE1→M传递依赖，非无同步覆盖。
WS Right和Key Right均0地址，生成M→MTE1通知/等待保护复用；因60KiB+10KiB超64KiB，
不能沿用N64时同时保留两者的预取。完整设备性能才能判断减少panel是否足以抵消。

审计脚本首版assert失败：跨AIC/AIV合并了重名C++常量，且误要求末次WS也占122880。
限制作用域并识别末次合法复用后通过，未改算子、编译器或容量。证据
indexer_n80_codegen_audit.json/indexer_n80_codegen_fragment.cpp。
任务task_20261017_050914_1722359619/b16_row_deps_indexer_n80_v1已提交，结果待终态。


N80终态：task_20261017_050914_1722359619 exit0，完整exact/图/30代/保护区PASS。
L1879.10/887.62/878.82，中位879.10，同轮动态895.08；Event981.13对956.57。
明显慢于已选行依赖候选，不采用、不扩测。实际NPU与CPU的52个N80 Acc/Right Tile
类型/形状/地址一致；首版阶段文本提取因SSA名称被优化失败，没有把分配一致称为全文一致。
完整限制及代码生成证据见INDEXER_N80_EXPERIMENT.md。

另做两项只读/离线调度核对，没有上卡：Indexer数学父任务为cache scale提交、Hadamard量化、
head coefficients，不包含主Q矩阵。主Q在模型最长链中出现来自资源日历，不可据此直接删wait。
把qproj_matmul与qproj_dequant_rms_nope_rope优先级偏置均设为-500，分别critical_path与
带权earliest_ready，固定旧成本的估计均799.38，对照773.43，不比现选有利，不生成新二进制。
证据main_q_indexer_dependency_inspection.json、deferred_main_q_schedule_analysis.json。
CPU session89931已收取；本轮所有CPU/NPU任务已终态。


## 2026-10-17 05:44：HC整行驻留失败修正及主Q显式L0预取

HC候选见HC_MIX_QPROJ_EXPERIMENTS.md。prepare/cpu v1 runtime tile.full失败，v2 scalar cast失败，
v3/v4 CPU通过。NPU v1 task_20261017_053012_51687711509与v2 task_20261017_053800_64833816891
均exit1：x_out15613项max0.015625、TopK825项不同，六cache/state中四份有差异。未计时。
两版仅Scalar/Vector部分和累加不同，结果相同；不能归因为Scalar加法。
仔细重读冻结body发现原来是16×K256，候选误用8×K512；这是候选实现错误，不是依赖bug。
hc_mix_resident_prepare_v5/cpu_v5恢复16×256和16次原顺序Vector累加，完整CPU通过。
实际IR与V到S屏障已核对；审计脚本首版只找顶层AST导致空集合，改ast.walk后仅mix变化。
任务task_20261017_054339_75460017083/b16_row_deps_hc_mix_resident_v3已提交，未预记PASS。

主Q普通N128的prepare/cpu_v1通过，但Right仍全部地址0，双缓冲未实现，未上卡。
新的qproj_l0_prefetch.py保留N256/20worker/列分工，T96的QR一次加载L1，K128八段按序INT8累加；
先取下一Left/Right再提交当前matmul。prepare/cpu_v1通过，qproj_prefetch_codegen_audit.py
确认两处主Q的Left地址0/12288、Right0/32768交替，下一Right的TMOV先于当前matmul。
这只是实际分配和提交顺序，不是测得的硬件重叠。首版临时检查没有递归解析C++地址cast，
显示None，已改用MLIR分配及C++操作对应关系，不改算子。
任务task_20261017_054303_7422476042/b16_row_deps_qproj_l0_prefetch_v1已提交。

当前七档选择不变，B16仍807.16首轮、817.88同进程复核，700目标未达。
本轮只改N内辅助实现/测试与记录，不改生产、冻结目录或依赖。两个新NPU任务待终态。


## 2026-10-17 05:50：两候选终态，无新性能收益

HC task_20261017_054339_75460017083 exit0，16×256修正后完整exact/TopK/六状态/图/30代/保护区PASS。
L1为822.46/819.02/817.06μs，中位819.02，同轮动态911.56。此前两版15613项输出差异已消除。
主Q task_20261017_054303_7422476042 exit0，同样完整PASS；L1为825.54/831.16/830.62，
中位830.62，同轮动态892.68。实际NPU产物的两处Left/Right双缓冲与CPU审计一致。
两项都不采用、不扩测其他档。七档选择保持，128K/B16首轮807.16/独立同进程复核817.88仍是现选。
没有现选同进程A/B，不将历史差值解释成确定回退幅度。
HC_MIX_QPROJ_EXPERIMENTS.md和hc_mix_qproj_results.json记录全部正负结果与限制。

只读避免重复：Indexer M192/N128、WS K192的两组三query方案已在indexer_n128.py测试过883μs，
没有再次生成或上卡。现有HCpost本身已经整行4096列，不继续基于错误分块前提设计。
本轮所有CPU/NPU任务终态。没有修改生产、F、PyPTO或Simpler仓；动态目录未commit。


## 2026-10-17 06:00：HC reduce→split→mix 合并候选

hc_front_fusion.py保留原数学体，仅把reduce独立行16改8，与split/mix的8行一致；
同一12个AIV每个完成自己的8行三阶段。四份投影部分和仍0..3，RMS仍16×256。
原三份GM写回保留，两处内部fence，显式传递store/loop结果Tensor版本。
原reduce/split两个task移除，Sinkhorn改等融合阶段，仍独立执行；原HCpost等待也重定向。
根参数及全部临时GM分配完全相同；66→64任务，722→704发布，等待12183→12189（重排后略增），
不把阶段减少说成总等待数量必然减少。完整CPU协议/视图/三代模型通过。
HC融合成本仅用三份原动态中位成本之和估计，balanced_schedule新cost kind明确未实测融合核。
prepare/cpu v1完整通过，实际生成TSTORE→PIPE_ALL→TLOAD顺序在两个边界均成立。
task_20261017_055741_99590920590/b16_row_deps_hc_front_fusion_v1已提交，结果待终态。

后续Tile直传版本hc_front_forward_tiles.py已做prepare v1：T96保证split循环恰好一次，
将其显式展开；reduce8×32先转32×8，再按行slice0/4并转回，避免非对齐列4切片。
原GM写回保留供Sinkhorn与诊断，split的两次raw回读和mix的pre回读改为SSA Tile直传，
两处内部PIPE_ALL删除，阶段外完成发布和编译器自动事件保留。
独立CPU准备进程临时绑定helper，不编辑运行任务的导入模块；该版本CPU编译中，尚未上卡。

## 2026-10-17 06:15：HC融合收取与Indexer输入行依赖

HC三阶段融合task_20261017_055741_99590920590完整PASS；L1三窗804.84/810.80/806.76μs，中位806.76μs，对原动态同轮900.14μs。与现选807.16μs是不同运行，不据此升级选型。

HC Tile直传CPU v1因转置临时Tile形状错误失败；v2切片静态valid与临时Tile动态valid不一致；v3改标准三参transpose仍有相同valid类型不一致，均未上卡。v4将对齐行slice改为extract，保持UB数据重排而不新增算术，CPU/PTOAS通过。生成HC数学区域静态TLOAD22→19，pto.fence4→0；保留GM写、阶段外发布/等待与自动事件。task_20261017_061445_12767489743已提交，未预记PASS。

Indexer scale物理页owner候选prepare/cpu v1通过，但历史探针显示204μs完成scale之后仍需等待235μs的query系数，调度估计也不降低总时长。因此只保留CPU候选，未上卡。

新增indexer_input_dependencies.py，扩展原row_dependencies的三个显式合同：quant→系数、quant→Indexer、系数→Indexer。T96每query的64头由w=q%48独占，scale视图平铺逐元素同址；每Indexer组覆盖两个请求，量化写者48→6、系数写者16→2。1..32叶×24组768种覆盖检查通过；保持MIX三成员同一输入要求、原同核顺序及QK的216条FFTS资源边。等待12183→10674，发布722→729。CPU完整编译进行中，尚无真机或性能结论。

## 2026-10-17 06:30：Tile直传、Indexer输入依赖终态与新阶段探针

task_20261017_061445_12767489743 / b16_row_deps_hc_front_tiles_v1 exit0完整PASS，三窗819.74/810.44/810.44μs，中位810.44，同轮动态891.96。
task_20261017_061600_130106629188 / b16_row_deps_indexer_inputs_v1 exit0完整PASS，三窗812.62/815.48/810.34μs，中位812.62，同轮动态882.64。两者均不替换现选。证据HC_FRONT_AND_INPUT_DEPENDENCIES.md及hc_front_and_input_deps_results.json。

为定位后续方向，允许已有row_dependencies组合stage_clock；CPU核对实际PTOAS原始C++的两物理函数、全部阶段边界及信号空闲区，每核88个时戳槽；开始误传wrapper源码导致定位失败，改用正确ptoas/csa_static_group.cpp后通过，未修改算子。task_20261017_062004_137819816687 / b16_row_deps_stage_clock_v1 exit0完整PASS。插桩span796.24μs仅用于归因，不参加700μs目标。

该样本Indexer AIC234.42..458.16、AIV247.90..461.94，QK/PV约481.7..647.36，O-A650.26..735.32，O-B712.20..766.78，HCpost772.68..796.22μs。KV norm虽然被Indexer占核推迟到456us，但本样本已在QK plan完成前完成，不能当作主要瓶颈。按最晚显式前驱回溯的链，数学/核内同步/发布段累计724.36us、就绪至开工间隙71.88us；这些不是硬件理论下限，也不能把后者全称为软同步开销。证据row_deps_stage_summary.json、row_deps_stage_critical_chain.json。

新增hc_mix_chunk_buffer.py候选：保持12成员×8行、原两个16×K256 pipeline和全部算术，以8×4096 BF16 UB Tile存每块结果；norm从UB提取，保留mixed GM写。区别于先前48成员整行resident算法，不增加Scalar归约。prepare_v1通过，CPU编译中；尚未接driver开关或提交NPU，也未声称收益。准备过程的qk_index_cache包装仅在该CPU子进程内生效，未改共享导入或已运行任务。

## 2026-10-17 06:34：HC分块UB缓冲CPU通过，已提真机

hc_mix_chunk_buffer_cpu_v1失败：8×4096宽Tile窄列assemble降低出的pto.tmov不满足A2/A3形状约束。v2把16个8×256块按连续行存入128×256 BF16，norm读取对应8行，未修改数学。prepare/cpu_v2完整通过，65536B驻留Tile只有一份，地址98336..163872，未越188416B可用UB。hc_mix_chunk_buffer_layout_proof.json检查32768元素独占与完整覆盖。
新增正式实验开关--hc-mix-chunk-buffer；prepare_v3生成entry的完整AST与已编译v2相同，没有重复完整CPU编译。task_20261017_063343_160455814507 / b16_row_deps_hc_mix_chunk_buffer_v1已提交，仍按原输出/TopK/六状态/图/30代/保护区和三窗L1检查，不预记通过或收益。
阶段归因增加可复现脚本summarize_stage_path.py；现选探针973个成员与调度一一对应，生成row_deps_stage_analysis/，结果与此前手工汇总相同。

## 2026-10-17 06:38：HC分块UB候选终态

task_20261017_063343_160455814507 / b16_row_deps_hc_mix_chunk_buffer_v1 exit0完整PASS；单SPMD三窗[814.74, 817.52, 812.92]μs，中位814.74μs，同轮动态907.88μs。输出/TopK/六状态、图/30代/保护区均通过；只固定fixture恢复，不是连续decode轨迹。相对现选未显示明确收益，不采用、不扩七档。未做同进程现选A/B，不将跨运行差值作为精确回退幅度。
所有CPU/NPU任务已终态，stage诊断和候选失败保留。无生产/F/依赖仓修改，无新commit。


## 2026-10-17 06:58：成员级CPU排程与权重缓存声明核对

member_schedule.py三种策略完成CPU图检查，973成员/877单元、216条FFTS资源边保留。
同成本原估计771.53us；critical802.49、earliest768.79、finish858.52，最好仅-2.74us且等待更多。
不上卡，不生成新二进制，见MEMBER_SCHEDULE_CPU_EXPERIMENT.md。
O-A物理IR已有K128内层和Left/Right双缓冲地址0/32768，避免重复优化。
进一步确认实验生成器过滤func_attr，导致Tensor数学体QA/主Q/O-A丢失原BYPASS；
KV已有显式Tile cache=1，仍保留两个IR属性，不能混为全部丢失。
weight_cache_policy.py按冻结原声明和本次绑定恢复融合AIC函数属性，检查整个调用图只读合同。
prepare_v1通过，移除新增属性后完整AST与现选相同；cpu_v1编译中。尚未上卡。

## 2026-10-17 07:05：权重缓存候选CPU终态与QK每query地址准备

weight_cache_policy_cpu_v1完整通过，IR缓存属性2→44；但C++完整2784561字节与现选相同。
当前PTOAS只有显式BYPASS offset才调整地址，无offset生成普通TLOAD。原动态也一样，
不是现选突然丢失了已生效的硬件功能。不提交等价NPU任务，保留开关默认关闭。
未修改PTOAS/ISA/Simpler，未猜测硬编码设备地址偏移，详见WEIGHT_CACHE_POLICY_EXPERIMENT.md。

新增qk_query_metadata.py：每query在tick%5==0准备512个压缩KV地址，
两个INT32 Tile共4096字节跨tick携带，四个压缩块复用；尾部tick禁止读超界query。
不假定第一个块有效，保留原负索引判断、逐行DMA、数学、FFTS和三槽流水。
代价是每AIV的地址计算256→512项，收益预期是页表读取最多4→1次。
prepare_v1进行中，未编译/上卡，不声称收益。


## 2026-10-17 07:15：QK每query候选终态、N80 WS分块候选

QK prepare/cpu_v1完整通过，驻留UB两份2KiB末端139264<188416。
任务task_20261017_070649_214406126149 exit0，完整exact/TopK/六状态/图/30代/保护区PASS。
L1三窗811.82/815.08/805.18，中位811.82，同轮动态904.92；没有明确优势，不采用、不扩七档。
详见QK_QUERY_METADATA_EXPERIMENT.md，尚无现选同进程A/B。

新增indexer_n80_ws128.py：针对旧N80的60KiB WS Right无法容纳下一Key，
将WS K384拆为0/128/256三个K128，Right每块20KiB；理论双WS+双Key=60KiB。
提前提交下一Key到Right，保留QK M384/N80、13个panel、原叶/FFTS/AIV TopK。
WS指令分段确实改变，不能只凭零系数或数学等价认定exact；CPU prepare_v1进行中。


## 2026-10-17T07:19:42：N80 WS K128 CPU通过并提交真机

prepare/cpu_v1完整通过。实际13份Key/39份WS Right，WS0/20480、后续Key40960/51200，
最大末端61440；12处下一Key extract先于当前QK且两Key槽不重叠，见物理审计。
task_20261017_071758_23254658630/b16_row_deps_indexer_n80_ws128_v1已提交，未预记PASS。


## 2026-10-17T07:23:07：N80 WS K128终态与QK全有效路径

N80 WS K128任务task_20261017_071758_23254658630 exit0完整PASS，L1三窗844.52/855.10/858.08，
中位855.10，同轮动态906.78。未采用，不扩七档；没有现选/旧N80同进程配对。

新增qk_all_valid.py：每个物理成员先扫描本组query×5个valid-plan项；全部>0才进入
消除重复块判断的快循环，任意无效则保留原循环。只删证明恒真的外层块判断，
稀疏行负索引、尾部、数学、KV搬运和FFTS保持。prepare_v1进行中，未编译/上卡。


## 2026-10-17T07:29:00：QK全有效CPU通过及真机任务

prepare/cpu_v1完整通过；AIV快/回退PIPE_ALL静态点3/4，mask读取0/4，
AIC为0/0、0/2。只说明生成代码变化，不预记性能。
审计FFTS发布初版搜set_flag_dev得到零，改为实际ffts_cross_core_sync统计；未改算子。
task_20261017_072611_24632809131已提交，终态待收取，见QK_ALL_VALID_EXPERIMENT.md。


## 2026-10-17T07:30:43：QK全有效终态

task_20261017_072611_24632809131 exit0，完整exact/TopK/六状态/图/30代/保护区PASS。
单SPMD三窗804.30/817.26/820.92μs，中位817.26，同轮动态905.18。
未显示明确收益，未采用、不扩七档，没有现选同进程A/B。
本轮全部CPU/NPU任务已收取终态。七档选择不变，700目标仍未达。


## 2026-10-17 07:43：Indexer设备短尾快路径

新增indexer_short_tail.py/--indexer-short-tail，最后1024分数块只剩1..64候选时，
仅计算第一个M384/N64 QK与M16/K384 WS。device valid_count选择分支，不写死历史。
原完整路径AST、FP16 FIX缩放/ReLU、FFTS交接/排空、AIV TopK均保持。
未写score_transfer后960列属于无效尾部；核对原九条排序分支的fillpad，
CPU检查叶内1..8192有效数、184314个query/step组合，通过2554个短尾组合。
沿用原S6组内位置有序前提；不是设备证明。见indexer_short_tail_contract.json。
prepare/cpu_v1完整通过，C++短/完整路径matmul2/32、FIX写L1为1/16、GM store1/16，
两条路径内部PIPE_ALL均0。indexer_short_tail_codegen_audit.json。
工作量模型：B4/B8/B16的最大核面板数不变；B24从528到513。不能凭总计算减少推断
B16明显提速；此模型不含TopK、FFTS或调度时间。indexer_short_tail_workload.json。
任务task_20261017_074327_274989723191已提交，只做B16现选基线上一个候选，
先exact/状态/图/30代/保护区，再三窗完整L1；未预记PASS，也未改七档选择。


07:48 B16短尾终态：exit0，全部输出/TopK/六状态、图、30代和保护区PASS。
L1为820.12/820.32/819.08μs，中位820.12，同轮动态902.04，不替换现选。
B24 prepare_v1也通过；工作量模型每核都省15个N64面板，单独提交
task_20261017_074803_283067829725 / b24_indexer_short_tail_v1，不扩大其他档。

另一独立候选--ramp-wait：沿用DCCI后volatile读取，失败后的等待间隔
4/8/16/32/64/100tick，随后保持100。每个新依赖重启短等待，不增加原子或删缓存维护。
wait_ramp_cpu_v1/audit.json核对：默认fixed完整C++与现选逐字相同；递增版
只有helper改变，GE条件、每次读前DCCI、写侧发布、payload及周期PIPE_ALL保持。
不把先前均匀0..400tick扫描重复一遍，仅检验一种由短到长的策略。
task_20261017_074820_28363061302 / b16_row_deps_ramp_wait_v1已提交，未预记PASS。
两项运行时均不再编辑driver/imported helper。


07:53 两项后续任务均exit0、原exact输出/TopK/六状态、图/30代/保护区PASS：
- B24短尾task_20261017_074803_283067829725：1146.14/1127.42/1127.60，中位1127.60μs；同轮动态1175.66。
- B16递增轮询task_20261017_074820_28363061302：833.74/829.94/826.22，中位829.94μs；同轮动态910.94。
两者均无明确性能收益，均未采用，不扩七档或继续扫描退避。未做现选同进程A/B，
不将跨运行历史差值定性为精确收益/回退。全为固定fixture，不代表连续decode。
见INDEXER_SHORT_TAIL_EXPERIMENT.md、RAMP_WAIT_EXPERIMENT.md及相应result。
本轮三个NPU候选均已终态；所有CPU exec session已收取；没有等待任务。


## 2026-10-17 07:58：Indexer N64循环化的CPU尝试

indexer_rolled_panels.py把16个显式展开panel改首块剥离、15次核内循环、末WS排空；
每步仍16次QK与16次WS、原M/N/K、量化/ReLU和FFTS不变，只减少展开代码。
这尚非指令缓存瓶颈的测量证据。prepare_v1通过，cpu_v1失败：编译器插入
Mat→Mat同地址tile.move，源/目标均180224；未提交设备。初版携带Key L1池，
该池每次只供当轮新写入的Key提取，无需跨迭代携带，因此v2改为迭代内创建，
仅循环携带Key Right和上一score L1。prepare_v2通过、cpu_v2编译中。
不改PyPTO/ISA，保留所有失败；循环物理缓冲未审核前不上卡。
indexer_rolled_source_contract.json只记源码操作顺序；其中页地址同表达式求值
不是独立验证，不将其计为正确性测试通过。


08:00 循环化v2终态：cpu_v2失败，实际PTO出现Key Right→Right tmov，
PTOAS拒绝此A3地址空间组合；另有score Mat→Mat移动。session12565已收取exit1。
仅CPU失败，不提交设备、不改依赖、不放宽校验。INDEXER_ROLLED_PANELS_EXPERIMENT.md
记录v1/v2与IR片段。未来可以研究两面板成组、避免Right跨循环携带，当前未实现。
本轮最终三个设备任务全部终态并PASS但未采用；CPU失败均保留，无运行任务。


## 2026-10-17 08:08：N64两面板成组循环

indexer_panel_groups.py / --indexer-panel-group 2：组内两个N64展开，组间8次循环，
组末完成WS再继续，不跨循环携带Key Right或score。QK/WS数学、叶分配、原页地址、
每1024块FFTS及AIV TopK不变。原N64 C++中QK与WS使用不同MTE1事件，
没有把它误当错误的共同等待；证据indexer_selected_qk_ws_order.cpp.txt。
prepare/cpu_v1完整通过。实际Key Right 0/49152，WS Right0，第二Key与前WS不重叠；
QK Acc0/98304B，WS0 Acc98304/4096B也不重叠；无Right→Right move。
生成组循环4个matmul、2个FIX、2个store、0个PIPE_ALL，循环C++片段27706字节。
这些仅证明分配/提交，不是并发或I-cache瓶颈测量。indexer_group2_codegen_audit.json。
生成AST实际store列检查2/4/8组均覆盖0..960的16块；只编译/上卡group2，
没有预记group4/8已通过。task_20261017_080841_317060310914已提交，待设备结论。


08:13 group2终态：task_20261017_080841_317060310914 exit0，完整输出/TopK/六状态、
图/30代/保护区PASS。L1 920.58/924.02/918.76，中位920.58μs，同轮动态909.50。
明显无收益，不采用，不扩其他档/组长。没有测到I-cache miss，不能将代码缩短视为优化成功。
证据indexer_group2_result.json和任务目录。


## 2026-10-17 08:22：Indexer 2048分数交接候选

保留原16个N64展开流水和叶分区，每两个1024块合为一次交接，私有双槽GM
[288,1024]→[288,2048]，Native/root布局不变。奇数尾块不增加数学量；
AIV段span按本叶剩余有效数截断，原三query Top-K/root合并AST保持。
contract核对40962组有效段、全部1..8192叶长与双槽通知计数模型；不是硬件证明。
prepare_v1与cpu_v1完整代码生成通过，实际AIV最大Vec端点131072B，
load为3×2048、stride2048，AIC READY在FIX、AIV ACK在MTE2 load之后，
尾部仍排空min(steps,2)。indexer_transfer2048_codegen_audit.json。
此处full_codegen_passed只指CPU，旧runtime_abi_supported=false来自32参数旧检查，
本会话使用独立256参数runtime；尚未预记设备PASS。

任务task_20261017_082119_33910122896已提交，仅128K/B16；运行期间不改driver/helper。

08:26 首轮2048交接exit1，输出与Top-K不符，六份状态完全一致；未采性能。
本候选遗漏三处root初始化步号：旧1024轮的step<2应改为新2048轮step==0。
保留失败result/log，修复在本目录helper，未修改依赖；v2准备通过，
contract补查初始化/合并选择，source核对仅三处条件改变，其余Top-K AST保持。
任务task_20261017_082629_34925986670已提交，未预记PASS。


## 2026-10-17 08:33：2048交接终态及Key池CPU结论

v2任务task_20261017_082629_34925986670 exit0，exact输出/TopK/六状态、图/30代/保护区全PASS。
L1 836.98/834.68/819.72，中位834.68μs；同轮动态899.98。未显示现选之上的收益，
未采用、不扩档。详见INDEXER_TRANSFER_2048_EXPERIMENT.md，保留v1失败和修正原因。
另indexer_key_pool.py只做CPU：128行Key池扩大1024行，源数学/地址/顺序不变。
完整codegen通过，但完整stage16 MTE1→MTE2 wait 19→20，反向均18，32个matmul不变；
没有目标等待减少，未上卡。抽取采用done_16与C++完整括号，纠正初次v30前缀只覆盖后半段。
见INDEXER_KEY_POOL_CPU_EXPERIMENT.md。CPU session96611已收取exit0，两NPU均终态。

08:35 新CPU候选qproj_l1_columns：保持20worker/原N256列所有权，每组拆两个N128，
QR L1入口驻留，两份K1024权重各一次加载。两个INT32累加链、K256显式交错，
理论L1 360448、Left49152、Right65536、Acc98304B；以实际编译分配为准。
仅准备/CPU编译中，未预记设备或性能PASS，不改变精度门槛。

08:37 主Q双N128列块CPU代码生成通过，实际Left地址0/24576，Right0/32768，
双Acc0/49152且彼此不重叠；两份整列权重位于L1 65536/196608，QR327680，
均在硬件容量内。两次matmul初始化+六次matmul_acc，保持每链K256升序。
证据qproj_l1_columns_codegen_audit.json；CPU session99975已收取exit0。
已提交task_20261017_083715_36883533451，先完整精度再L1，不预记收益。

08:42 主Q双列v1任务exit0，完整exact输出/TopK/六状态、图/30代/保护区PASS。
L1 809.22/806.46/797.56，中位806.46μs；同轮原动态903.86μs。
与现选历史807.16接近，不能直接宣布收益，下一次只作现选与候选同进程三窗L1。
对照版必须开启现选row_dependencies=True、QK地址缓存和100tick轮询；
复用paired_row_dependency_perf，旧调用默认行为保持，不重做动态metadata或Event采样。


## 2026-10-17T08:47:35：主Q同进程终态

task_20261017_084154_377975115763 exit0，双方exact/图/六状态/保护区/7代PASS。
现选L1 805.28/808.92/810.24，中位808.92；候选806.02/804.06/810.32，中位806.02。
中位差-0.36%，均值仅-1.35μs且范围重叠；未采用、不扩档、不重复该比较。
对照生成AST逐项等于b16_cached_row_deps_v1，不是此前未加行依赖的旧版。
测试只2次预热/每侧3窗L1，没有重复Event或动态metadata。
本轮所有CPU exec及NPU任务均已收取终态。七档现选不变，700μs目标未达成。

只读排查也确认：SWA原分支已按Native物理页批量gather，不能再假设它逐行读；
row_dependencies已有成员图传递约简和逐核observed过滤，不需要重复增加相同的seen缓存。
这些没有新增设备任务。下一候选可评估N32的双QK Acc流水（尚未实现），
与已失败/变慢的M192/N64分query方案区别是保留M384/WS K384，只沿候选列拆分。


## 2026-10-17T08:55:50：N32双QK Acc候选启动

只改Indexer AIC面板为M384/N32/K128与M16/N32/K384 WS，保留原归约和1024分数交接。
32面板展开，下一QK在当前FIX前提交。prepare_v1通过；cpu_v1编译中，尚无设备结果。
此方案区别于此前M192/N64分query，容量和源码先后不能当作设备流水收益。

N32 cpu_v1完整codegen通过，31对相邻QK Acc均0/49152交替、生成顺序保持。
Left query0/系数49152；首两Key Right0/4096，后续Key24576与WS Right0分区。
完整阶段64matmul/32FIX/32store。task_20261017_085633_40447024100已提交，尚无性能结论。


## 2026-10-17T09:06:11：N32设备终态

任务task_20261017_085633_40447024100 exit0；完整exact/图/六状态/30代/保护区PASS。
L1三窗985.94/976.20/981.12μs，中位981.12；同轮动态901.76。未采用，不扩档、不重测。
见INDEXER_N32_ACC_EXPERIMENT.md及indexer_n32_acc_result_v1.json。


## 2026-10-17T09:08:10：KV原样回写标记折叠

新增kv_touch_elision.py：将唯一kv_touch的父依赖直接接到QK/PV，保留写回完成和缓存维护。
准备v1被本地合同拒绝：call实参annotation已去掉Out，真实核体形参仍pl.Out。
修正为结合核体形参与allocation检查，属于候选检查器假设错误，不是组件bug。


## 2026-10-17T09:12:13：保留KV汇总者，只删除业务空操作

完全删除kv_touch的prepare_v2/cpu_v1通过，任务66→65、成员973→972，
但远端等待12183→12673，发布数仍722。汇总等待被展开成写回成员等待，未提交设备。
新增kv_touch_signal_only.py保留原阶段、等待表、代际和TNOTIFY/WAIT，只删除1024B自拷贝
以及无业务数据阶段的4个fence/2个全Scalar缓存维护。其他源码逐AST相同。
kv_touch_signal_only_prepare_v1通过，cpu_v1编译中；没有预记性能收益。

纯信号版本cpu_v1完整通过，实际C++ load/store各1→0、PIPE_ALL5→1、全DCCI2→0；
TWAIT/TNOTIFY保留，其他源AST和等待表不变。task_20261017_091307_19138223772已提交。


## 2026-10-17T09:19:37：KV纯信号首轮通过

首轮exit0，完整exact/六状态/图/30代/保护区PASS；L1中位802.96μs（802.96/812.72/798.04）。
同轮动态894.00。与现选差距小，补一次同进程三窗复核task_20261017_091800_27634922996，
只2次预热/3窗L1，不重复Event或动态metadata，候选尚未纳入现选。


## 2026-10-17T09:24:22：KV纯信号同进程终态

任务task_20261017_091800_27634922996 exit0，双方exact/图/六状态/7代/保护区PASS。
现选805.14/808.22/807.50，中位807.50；候选812.08/803.92/812.32，中位812.08μs。
候选未测出收益，不纳入现选、不扩档、不重测；对照生成AST与现选一致。
本轮N32、KV纯信号两轮NPU均终态，所有CPU exec已收取终态。


## 2026-10-17T09:25:07：L1循环原位更新CPU探针

小例子v1因遗漏SPMD index被前端拒绝；v2忽略assemble返回值，触发Mat目标不能复用的
保留区域拷贝错误；v3使用pool=assemble(...)完整CPU/PTOAS通过。
实际循环TINSERT写同一512B地址Mat，循环后TEXTRACT与FIX→MTE1同步保留，无Mat→Mat移动。
只覆盖小FP16矩阵/无分支/无量化，没有上卡，也不证明CSA完整流水。见TILE_LOOP_INPLACE_CPU.md。


## 2026-10-17T09:28:48：N64固定L1环候选启动

上一目标轮完成N32/KV候选终态和固定L1小例子，属于取得新证据的进展；无运行遗留任务。
本轮新增indexer_l1_ring.py：固定三槽Key/双槽score Mat，DPS返回值写回同一变量，
Right留在循环内部，保留M384/N64/K128和WS K384及原1024交接/AIV/叶计划。
先CPU准备和物理代码审计，不把小例子成功当作大核通过。

N64 L1环v1准备通过，但cpu_v1失败：Key池在最后一轮不更新的分支合流处被插入
Mat→Mat同地址188416自拷贝。并非此前Right循环携带，Right已在本迭代局部使用。
v2将最后QK/WS剥离，循环1..14每轮固定预取下一Key，避免可选更新的外层if。
旧源码、passes_dump和错误均保留，不修改编译器或降低检查。


## 2026-10-17T09:41:15：N64环v2被物理地址审计拒绝

v2完整CPU编译成功，但实际Key池中间phi17位于L1地址0，其他池/循环读回位于188416。
第二页TLOAD写入0地址，后续Key提取仍使用188416，且没有有效拷贝。不能上卡。
indexer_l1_ring_codegen_audit.py按物理地址拒绝该版本，证据与rejected_audit均保存。
只读查看gather_row定义确认它是ReadWrite DPS并复用输入0，不是无副作用API。
下一版尝试在同一个分支中完成两页加载，避免两次顺序分支产生中间phi；
继续先CPU审计，不修改依赖仓库，也不绕过地址检查。


## 2026-10-17T09:44:27：L1环v3编译拒绝，v4撤去Key循环携带

v3把两页加载合并在同一分支叶子，prepare通过，但CPU/PTOAS仍因Mat→Mat搬运失败。
错误定位source.py:904:25，原始PTO/passes/report全部保留，未上卡。
v4仅保留score双槽Mat跨迭代，每轮重新创建64×128 Key池并加载本面板。
这会降低Key预取提前量，因此不能预判收益；保留QK(current)/WS(previous)跨面板交错。


v4 prepare/完整CPU/PTOAS通过，物理审计Key池163840、score65536；
QK/WS Acc0/98304、循环内Right49152/0分区，无同层搬运和PIPE_ALL。
实际循环两个matmul、一FIX写L1、一次WS写GM，首块/排空均保留。
新增driver开关--indexer-l1-ring，并在首次设备执行前重新审计实际运行artifact。
task_20261017_094749_7744056502已提交，只128K/B16，不预记精度或性能。


## 2026-10-17T09:50:17：条件gather循环缩减探针

probe_loop_gather_phi.py CPU完整编译通过，初始池、两个if phi和前后transpose视图
物理地址均为0，没有复现完整CSA v2的0/188416不一致。未上卡。
该探针只含循环携带+两次条件DPS写，缺少完整CSA的外层分支和全图L1复用上下文。
保存allocation_audit.json及完整PTO/C++文本，不能把它当作v2已修复的证据。


## 2026-10-17T09:56:58：L1环v4设备终态及下一方向的边界

task_20261017_094749_7744056502 exit0，执行前真实artifact物理审计通过；
完整输出/Top-K/六cache-state、eager/图/30代、未写区域和workspace保护均PASS。
L1 984.56/1034.82/1008.86，中位1008.86μs；同轮动态905.34。
Event候选1133.55、动态951.86μs。未采用，不扩档，不补同进程复测。
现选策略仍807.16μs历史记录，最新现选同进程807.50；这两者不是v4同轮对照。

新增indexer_query_completion_model.py/json：保持各worker叶所有权，枚举1..32叶
共8448个item，证明按request重排不漏不重。实际B16当前6叶，每叶score块6/6/6/5/5/5，
原每核22块；按request重排前8请求的数据可在11块后就绪，后8仍22块。
但原阶段通知不改时所有消费者仍等22块，所以不能只重排循环就宣称收益。
merge每成员跨2个query、QK跨4个query，O-A整批行/O-B M96也会限制下游流动。
此模型没有时间预测、编译或设备验证，不新提交无收益依据的任务。

设备终态后将循环Key/WS Right区间不重叠纳入自动审计，对已有CPU与设备artifact
重新执行纯文本检查均PASS；未重新运行NPU、未改变任何数值结果。


## 2026-10-17T10:15:22：两组query完整成员图评估

上一轮为进展：L1环v4设备终态与两组query所有权模型已保存，任务已复核exit0。
本轮新增query_group_schedule_model.py，将Indexer/merge/plan/QK/O-A/quant/O-B/HCpost按48行拆组，
保持MIX原logical→physical映射，完整检查数学边、资源区间与同worker交接区复用。
1313成员/1121单元/43953数学边。使用原动态同形成本，拆分数学按一半、每成员另留1μs，
这是理想CPU模型，不是设备测量，也未生成设备算子。
前半优先855.325、最早就绪780.470、关键路径813.585μs；同模型现选771.530。
三种排程均未改善，所以不实现/上卡粗粒度两半流水，也不做更悲观投影成本扫描。
前半优先会延迟主Q；其余策略第二半QK仍把第一半O-A挤到后面。
模型与边界保留，不能称拆组方案本身在所有排程下都不可能优化。


## 2026-10-17T10:17:59：PV合并改为8-head，先检查UB

旧16-head的两版均需196608B超过188416B，未上卡。本次将每块16行缩为8行，
保留完整32-head运行状态、相同乘加顺序、先左后右、原FFTS和GM输出。
prepare_v4被旧“row-deps不得与qk_merge_heads组合”保护条件拦截，尚未进入编译。
补入明确合同：所有GM写表达式、scalar metadata读、显式同步与其他函数AST完全不变；
PV读取小Tile的并集仍32×512，且row_dependencies原行分片合同仍执行。
在这个已核对范围内允许组合现选row-deps/QK地址缓存，重新prepare_v5/CPU_v5；
未放宽UB上限或任何数值/保护区门槛，没有改依赖。


## 2026-10-17T10:28:40：两组调度补查及PV状态分块

flexible_mixed earliest允许每半重新分配物理核组，仍严格保护logical worker的GM交接区及FFTS；模型780.470μs，与固定映射相同。
prefix_then_phase0先为未拆分主Q等前缀安排位置，再优先前半链路，模型841.340μs。二者都不优于同模型771.530μs，不生成设备候选。
上述是理想CPU模型，不是实测；完整核对数学边、MIX bijection、物理/GM资源区间。五个模型结果均保留，无NPU。
qk_merge_heads prepare_v5成功，CPU_v5仍因196608B>188416B失败。MemoryReuse转储显示4个32768B状态基址和1个65536B共享arena；缩小incoming为8行不能消除整状态双份分配。
qk_merge_heads_v5_memory_allocation.json保存基址及高端使用点；其中15个动态视图未参与静态末端求值，超限结论来自编译器。
保存旧8-head变换到source_snapshots/qk_merge_heads_8_full_state_v5.py。v6将每侧loop-carried状态本身分成两个16×256值，移除完整状态slice/assemble链，逐元素old*alpha+incoming*beta顺序保持。
显式展开初始化、循环/分支yield、reset及最终16-head输出引用；所有GM写入表达式、scalar metadata读取、显式同步、其他函数AST仍做不变检查。尚待完整编译及设备验证。

### 2026-10-17T10:32:54：v6完整编译、实际地址审计通过并提交B16

qk_merge_heads_prepare_v6/cpu_v6完整codegen PASS；AIV五个分配槽为65536/32768/32768/32768/32B，总163872B。四个16-head状态位于65536/81920/98304/114688，各16384B，初始化/合并/重置原位且互不重叠。
新增qk_merge_heads_codegen_audit.py；CPU已通过，设备driver在第一次kernel执行前也检查实际artifact。禁止仅凭编译通过宣称地址正确或性能改善。
codegen比较显示PV合并局部PIPE_ALL原/新均0，V barrier9→13，set/wait各6→10；最终输出V barrier25→24。候选不保证提速，不把UB缩小外推为耗时下降。
task_20261017_103218_149245329068提交单B16/128K/S6，现选row-deps/QK地址缓存/100tick等待，只启用qk-merge-heads；正式权重，只与原动态PTO比较。待结果；不扩七档。

## 2026-10-17T10:39:13：PV循环状态分块v6设备完成，不采用

task_20261017_103218_149245329068 exit0完整PASS：eager/graph/final输出、Top-K、六cache-state零差异，30代通知和全部保护区通过。
L1候选1033.44/1351.48/1029.72，中位1033.44；同轮动态910.98/914.34/912.20，中位912.20。Event1118.08 vs959.95。未丢慢样本，不采用、不扩档、不重测。
现选此前807.16/807.50不是同轮现选配对。全部阶段静态指令计数只有QK AIV stage30变化：PIPE_ALL9→8、set/wait各80→82、TLOAD22→24、row_expand_mul4→8。不能据此定量归因回退。
QK_MERGE_STATE_EXPERIMENT.md、qk_merge_heads_result_v6.json及full_codegen比较保存。

## 2026-10-17T10:40:56：只调整Indexer的WS Right加载顺序，CPU候选

只读原体确认：现选已经是16个N64展开的QK(current)/WS(previous)，不能将再次展开称为新增流水。
新增indexer_ws_load_order.py，仅交换15对互无值依赖的WS Right extract与当前QK matmul，其他节点AST多重集合完全相同，首块/末WS drain、其他核体不变。
v1/v2准备失败是变换器遍历正在交换的列表，第二次把已移动节点当新模式；修为只遍历原列表快照，同时检查唯一末块排空。cpu_v2因prepare未成功、source不存在而退出，未进入编译，不是编译器问题。日志保留。
prepare_v3只在当前CPU准备进程注入catalog，未改默认catalog、生产入口或驱动参数，尚未NPU。物理Right若因生命周期缩短被复用并串行化，应停止，不直接上卡。

### 2026-10-17T10:44:21：WS顺序候选v4编译通过，但Right复用，不上卡

v3准备仅在末WS名称假设检查失败；核对真实buf_last_pair后改为最后5条AST位置不变检查。prepare_v4及cpu_v4完整通过。
实际地址检查：1/15对当前QK和前块WS Right重叠，现选15对全不重叠；两侧Acc均互斥。加载移到消费后缩短了活跃区间，自动复用破坏预期重叠。
阶段静态matmul32、PIPE_ALL1、M→MTE1wait32、MTE1→Mwait37均不变；set/wait166→165不是性能依据。不上卡、不改默认，无新设备结果。详见INDEXER_WS_LOAD_ORDER_CPU.md。

## 2026-10-17T10:48:26：WS后移仅保留后14对，先核对实际流水

上一goal轮为进展：PV状态分块已完整验证回退，WS全15对后移完成CPU编译并证实仅首对Right复用。本轮复核设备任务已终态、无旧编译进程。
保存all15_v4源码快照。当前只保留buf_previous_pair...ssa_v1原顺序，后14对交换；AST多重集合、首对、末5条排空不变。新增prepare_ws_order.py为可复现CPU准备入口。prepare_v5通过，cpu_v5编译中。
除实际Right分配，还要检查C++ set/wait对应位置，区分源发射顺序与真正等待；如原版本就允许QK与WS加载重叠，则不上卡重复测。

### 2026-10-17T10:52:19：14对CPU及实际C++通过，准备单B16验证

prepare_v5/cpu_v5完整PASS，实际Right与Acc的15对均不重叠，原首对仍load→QK，其余14对QK→load。集成build/device开关后入口文本与已编译v5逐字符相同，无额外CPU编译。
新增C++ set/wait位置审计：原版本在WS TEXTRACT之后到QK之间没有等待WS加载完成的直接串行边，不能声称新版本首次实现计算/搬运重叠。
实际改变是将QK提交放到WS TEXTRACT之前的FIX→MTE1和M→MTE1等待前。例如panel2，旧行1265/1266等待、1269TEXTRACT、1279QK；新1266QK、1276/1277等待、1280TEXTRACT。
新M→MTE1 EVENT_ID3来自更早的set（panel2示例行1001），不等待当前QK。所有14对都检查了该直接边界，无PIPE_ALL介入；不能外推实际重叠时长。
静态总matmul32、PIPE_ALL1、M→MTE1wait32、MTE1→Mwait37、set/wait166均相同。保留首对避免v4的Right复用。设备首次执行前也审计实际地址与14对顺序。


## 2026-10-17T11:04:49：审计提前编译绕过等待扩展，撤回三项退步归因

WS v5任务105218完成exit0、全部精度/状态/通知/保护区PASS；L1 1033.52/1037.68/1032.64，中位1033.52，同轮动态898.64。发现与PV v6异常接近后核查编译流程。
三项新增artifact审计均位于ExitStack外，_resolve_kernel_artifact提前缓存了默认TWAIT版本。实际CCEC输入AIC/AIV均74个默认TWAIT、0个CSA_WAIT_LOCAL_GM、无退避；现选基线则74个替换调用及100tick。不是PyPTO编译bug。
扫描74项已完成cached_wait测试，仅ring v4/PV v6/WS v5三项缺少回调记录，实际默认源码均确认。原始数据保留；精度有效，撤回由此认定数学核回退/不重测的结论。
修复driver三处审计进入所有编译扩展后再解析；新增执行前实际AIC/AIV源检查。已有正确产物通过、3份错误产物均因缺回调及源码不符被拒绝。初次CPU正例因最早manifest无后来ramp_backoff字段失败，改用已有明确false的正例，不篡改旧记录。
修正WS v6任务task_20261017_110148_204947624179，仅128K B16，原精度门槛/3窗完整L1保留。详见COMPILE_CONTEXT_CORRECTION.md、compile_context_*json。
CPU main_q_then_phase0模型862.105μs，不优于同模型771.530；没有设备候选。CPU资源下界当前665.622、两组理想拆分672.955，仅给定成本模型，不是硬件下界或达标证明。

### 2026-10-17T11:08:29：WS等待策略修正后完整通过

task_20261017_110148_204947624179 exit0，实际AIC/AIV均74个替换等待、100tick门禁通过；全部精度/状态/图/保护区/30代PASS。L1 814.12/812.34/816.48，中位814.12；同轮动态902.42/896.00/899.98，中位899.98。未采用、不扩档。
提交PV v7 task_20261017_110518_211373122343及ring v5 task_20261017_110544_21223701219，均仅B16/128K修正配置。期间误提交不完整命令task_20261017_110519_211380121075，发现后cancel提示已运行，立即kill并确认终止；约25秒内取消，没有性能结果，目录保留。

### 2026-10-17T11:13:15：三项修正验证完成，准备N64两面板预取CPU候选

PV v7 exit0，全门槛PASS，L1中位820.24、同轮动态909.56；ring v5 exit0，全门槛PASS，L1中位806.38、同轮动态885.92。两项均实际等待策略门禁PASS。没有七档选择变化。
新增indexer_l1_key_pairs.py/prepare_indexer_key_pairs.py：首块0，循环(1,2)…(13,14)，末块15及排空；每对在第一块QK前提交第二块Key的GM→L1，只让score双槽跨回边，避免原Key分支phi别名。
相同M384/N64/K128、WS K384、FIX 2^-10/ReLU、原1024交接/AIV/叶计划；prepare_v1通过，CPU_v1 session96548运行。实际地址与等待位置未核对，暂不加设备开关。

### 2026-10-17T11:16:31：两面板Key预取完整CPU通过并提交B16

CPU_v1 full_codegen PASS；相同score双槽98304B，Key池分支/循环原位，L1与score不重叠；三个QK/WS并行对的Right/Acc均分离，循环内无PIPE_ALL/非法同层搬运。
实际C++第二块Key四个静态地址分支TLOAD位点269/302/356/389，在首QK415、WS427之前；MTE2完成事件391发出，484才等待，位于首QK/WS后、次QK513前。是提交/等待证据，不是已测重叠收益。
接入独立--indexer-key-pairs/build选项；集成入口文本与已编译CPU入口完全相同，不重复CPU编译。实际设备仍须先进行同样地址/顺序及100tick源码门禁。
task_20261017_111540_229620229969提交，只测128K/B16/S6，正式权重、原动态PTO对照、完整精度和3窗L1，不更改七档选择。

### 2026-10-17T11:20:58：两面板Key预取设备完整PASS，未采用

task_20261017_111540_229620229969完成exit0；设备实际Right/Acc/L1/预取顺序、100tick等待源码门禁均通过。输出、TopK、六cache/state eager/graph/final全部零差异，30代与全部保护区PASS。
完整L1 813.42/814.14/811.40，中位813.42；同轮动态907.18/898.80/892.24，中位898.80。Event926.35 vs950.66，不能替代L1验收。没有优于现选的证据，不采用、不扩档、不重复。
真实C++确有第二Key预取发射提前，但没有转化为整层已测收益，不把静态事件顺序作为速度证明。原N64单面板ring旧产物仍通过扩展后audit默认路径。


## 2026-10-17 11:50：排除重复缓存方案、页基址选择CPU限制，提前KV通知未获收益

交接提出的Scalar读写/Tile DMA分类维护，已由payload_cache.py和b16_cached_dynamic_precise_v1实现验证；旧中位838.68μs，03:47也已复核。本轮不重复实现或上卡。
Indexer任意stride参数当前JIT明确拒绝，读取接口与现有TestUnsupportedLayoutSlot确认。GM双基址先select再gather探针v1因未用block index被拒绝；v2改显式worker后因不同backing SSA的if返回被拒绝。均CPU、无NPU、无依赖修改，见INDEXER_PAGE_BASE_SELECT_CPU.md。
QK early候选只改三个阈值144→96，原AST其余部分不变。新向量索引路径与旧阈值制定时不同，因此只重新评估该T96配置；不把旧阈值当bug。prepare_v1/cpu_v1完整通过，集成入口与编译过的源码文本相同。实际顺序Score wait→KV notify→softmax→Prob notify→PV wait；2wait/2notify/9PIPE_ALL与原相同。所有artifact解析仍在编译扩展入栈后，100tick门禁保留。
任务task_20261017_114240_273482232688、b16_row_deps_qk_early_kv_v1，exit0。eager/graph/final输出、TopK、六cache/state均零差异，30代通知和保护区全PASS。L1 823.38/816.34/833.82，中位823.38；同轮动态905.06/907.42/900.04，中位905.06。没有优于现选证据，不采用、不扩档、不重复。没有本轮现选配对，不精确归因跨轮差额。见QK_EARLY_KV_EXPERIMENT.md。

## 2026-10-17 11:51：双AIV按职责分工CPU原型

qk_role_split.py：AIV0采全128行KV，AIV1负责64head softmax/PV归并/输出，Cube保持原样；两条AIV都保留原Score/PV等待及KV/Prob通知。无任务内新增GM信号、无新增workspace。64head状态128KiB拆成16个8×256 FP32循环值；softmax与最终发布按8head处理，每head运算次序保持设计，尚需设备验证。所有packed输出改由AIV1发布，明确不能沿用旧O-A按head半区选择AIV0/1的依赖。CPU准备暂保留原整父任务依赖，不能作为现选策略的性能比较。
qk_role_split_protocol.py通过3328个调度、471867动作：两query全部1024种valid mask×3个调度，加256种四query随机mask×3代。五份缓冲区槽复用与每代事件排空通过，事件credit抽象峰值3；不证明硬件容量/DMA/缓存语义。
prepare_v1通过；cpu_v1在ConvertToSSA拒绝分支外kvbuf引用，未到内存规划。保留source_v1快照和全部失败产物；v2将原始KV和压缩KV各自store留在各自分支，用独立kv_done接收循环结果，数学与事件不变。prepare_v2通过，cpu_v2正在编译；尚无设备结果或性能。

### 2026-10-17 11:57：双AIV分工完整CPU通过，输出所有权依赖修正后上卡

cpu_v2 full_codegen PASS。实际16个8×256 FP32状态共131072B，起点16384..139264，每份8192B，互不重叠；init/carried/valid地址相同，16个PV add逐个写回。QK显式Vec shape地址范围163328B，非完整allocator报告。C++8个rowmax/rowsum/最终rowdiv，两分支4wait/4notify，5PIPE_ALL，源码事件提交顺序审计PASS。
row_dependencies新增默认关闭的角色所有权分支：从真实源码检查全部packed输出只在AIV1，768行唯一写入且八组O-A消费全覆盖；O-A改等对应AIV1。旧六项源合同和行覆盖证明PASS。prepare_v3使用新证明，角色数学AST与已编译v2相同，无重复数学编译。新增--qk-role-split，生产/依赖不改。
task_20261017_115641_29670009014 / b16_row_deps_qk_roles_v1，只测128K B16 S6、正式权重、原完整精度/图/30代/保护区和三窗L1，设备编译后再检查实际状态地址/事件及100tick等待。提交时无设备结论，见QK_ROLE_SPLIT_EXPERIMENT.md。

### 2026-10-17 12:04：双AIV分工完整设备PASS，但性能无收益

task_20261017_115641_29670009014 exit0。eager/graph/final输出、TopK、六cache/state均零差异，30代进度、全部保护区PASS，实际产物的状态地址/事件顺序/100tick等待门禁PASS。L1 887.72/885.08/870.46，中位885.08；同轮动态897.74/901.30/908.12，中位901.30。未采用、不扩档，不把4个内部PIPE_ALL减少直接当作提速。
观察期40s工具等待超时不是kernel超时，原任务后续正常完成，没有取消重启。只读本地ISA集群映射说明mode2两AIV与4bit计数，未发现设备事件丢失证据。
为了定位角色分工为何没有收益，提交单次阶段时戳诊断task_20261017_120238_306765112993 / b16_row_deps_qk_roles_stage_v1，warmup0/iters1/windows1。含插桩且暖机不同，eligible_for_700us_target=false，不能替代正式结果或精确推导跨轮退步幅度；未重复全量七档。运行中不改其导入代码。

### 2026-10-17 12:08：双AIV分工阶段诊断完成，主要退步定位到QK/PV

task_20261017_120238_306765112993 exit0，eager/graph/final全exact、4代进度、保护区PASS。插桩span867.12，L1单窗878.80均不参加700目标。与历史探针相比，Indexer AIC数学/内部等待/维护中位211.03→212.99；QK AIC150.93→206.39、AIV0 157.90→207.90、AIV1 158.12→222.80μs。本轮QK AIC/AIV0/AIV1最后结束701.30/703.50/718.38，O-A最早720.94，AIV1是组内尾部。
暖机与采样不同，不精确分摊跨轮差额，也不把阶段墙钟当pipe独占忙时。小8-head Tile令softmax变为AIV1八组、发布更细，且单条AIV承担双倍head；这是待CPU核对的开销来源，没有独占阶段探针证明占比。原版不重复上卡，先考虑更大head Tile实际UB/状态别名，不能只凭理论余量扫描。详见QK_ROLE_SPLIT_EXPERIMENT.md和qk_role_split_stage_comparison_v1.json。
只读复核member_schedule.py：原成员三种启发式和query两组六种模型已验证过，不重复；未安装/使用新求解器。当前所有任务终态，生产/F/依赖仓/共享安装保持，七档选择不变，goal active，700目标未达到。


### 2026-10-17T12:22:59：角色分工的head Tile分离，CPU确认发布峰值超UB

在现有8-head分工版本上，将PV状态块扩为16×256、softmax扩为32×128，最初发布也为16head。
qk_role_tiles_h16_s32_prepare_v1准备PASS，cpu_v1在AllocateMemoryAddr拒绝Vec194048B>188416B；无设备执行。
从37_after_MemoryReuse逐项核对，AIV各allocation求和正好194048B。状态仍128KiB，PV的两份16KiB临时区和
最终32KiB FP32发布结果复用；16KiB BF16输出与两份4KiB RoPE临时量、4KiB swap索引同时存在，
因此单独缩softmax并不能消除发布峰值。诊断qk_role_tiles_ub_diagnosis_v1.json保留全部allocation。

将publication_tile从state head_tile解耦：仅发布回到8head，从对应16head状态slice连续行，
仍为32head softmax、16head PV；原运算次序、Cube、GM接口和事件保持。旧8/8及16/32同发布
版本AST回归一致，旧8/8/8编译产物扩展审计PASS。新增快照qk_role_split_source_h16_s32_p16.py.txt。
qk_role_tiles_h16_s32_p8_prepare_v1 PASS，cpu_v1编译中；当前尚无此变体设备结果，不修改七档选择。


### 2026-10-17T12:24:22：16/32/8候选完整CPU通过

qk_role_tiles_h16_s32_p8_cpu_v1 full_codegen_passed=true。MemoryReuse的全部AIV Vec allocation合计
167552B，低于188416B；与h16/s32/p16的194048B相比减少26496B，实际全局复用结果不同，
不能仅按局部Tile字节数估算。8个16×256 FP32状态起点0..114688，互不重叠且原位更新；
incoming不覆盖状态，8次旧值乘法/更新逐槽匹配。softmax2个32head rowmax/rowsum，发布8个8head rowdiv，
仍4wait/4notify/5PIPE_ALL，提交顺序PASS。全部记录qk_role_tiles_h16_s32_p8_codegen_audit_v1.json。

新增默认关闭--qk-role-wide-heads，限定与--qk-role-split组合，明确使用(状态16,softmax32,发布8)；
设备执行前检查实际编译Tile与请求一致。行依赖继续检查AIV1独占全部8组packed输出。
CPU集成准备中，不重复已经通过的数学编译；尚无设备性能，不修改现选。


2026-10-17T12:25:55：集成行依赖后全部数学体AST与CPU通过版本一致；提交task_20261017_122441_342567127888，b16_row_deps_qk_roles_wide_v1。仅B16/128K/S6，全门槛+三窗无插桩L1。任务执行中冻结driver/helper，不改现选。


### 2026-10-17T12:28:27：32head softmax/16head PV/8head发布全门槛PASS，未采用

task_20261017_122441_342567127888 exit0，b16_row_deps_qk_roles_wide_v1。
eager/graph/final的输出、TopK、六cache-state均零差异，30代进度及全部保护区PASS。
实际Tile(16,32,8)、状态/事件及100tick等待源码门禁PASS。无插桩完整L1
830.46/827.18/832.04，中位830.46；同轮动态899.68/911.08/912.48，中位911.08。
比旧8/8/8分工的历史885.08明显降低，但仍未优于现选807.16/807.50；不同轮次无现选配对，
不精确归因差额。未采用、不扩档；暂关闭仅靠分工/增大head Tile的优化方向。
见qk_role_tiles_result_v1.json。所有设备任务终态，700目标未达。

独立新增fixed_core_local_schedule.py，只在CPU从现选顺序进行固定物理分核的相邻交换，
与前轮三种从头贪心分配不同。数学边与216条FFTS组复用边固定，MIX按共同开始建模，
防止误用成员独立开始的时间；原模型771.53，共同开始模型773.41。120秒局部搜索中。
未生成二进制、未接入生产/测试driver，运行中的设备helper没有修改。


### 2026-10-17T12:35:07：固定分核局部搜索及整组后移模型均无收益

fixed_core_local_schedule_cpu_v1：120秒评估7679次、149个环被拒绝、13个等长移动；
整层共同MIX开始模型773.41→773.41，未改善。另将48条AIV上的主Q后处理整组放在Indexer后，
fixed_core_defer_q_dequant_cpu_v1完整图无环，但773.41→788.94，未采用、未生成二进制、未上卡。
Indexer最早开始207.49→191.08，最后结束仍444.24；merge和plan延后，QK最早465.47→481.00。
这些是CPU成本图，不是设备性能归因。源码、全部模型、关键链见FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md。
原始L1不含可直接使用的逻辑block索引，需要匹配派发日志，未尝试从完成顺序猜分片。
目前所有设备与CPU任务都已终态；本轮不改变七档选择，goal仍active、700目标未达。


### 2026-10-17T12:47:41：固定分核全局MILP排程模型开始求解

只读复核Indexer页视图/Key双面板/N128/双Acc旧候选，未重复它们的测试。已有Key提交探针
11.81μs不足以填补整层差额；不把异步提交时间当独占搬运忙时。
尝试在N/cache/schedule_solver_env安装ortools：默认索引Proxy403失败，官方索引也未取到版本。
全部日志保留，未修改共享环境。改用现有SciPy1.13.1/HiGHS1.2.0，无需外部安装。

fixed_core_milp_schedule.py保持物理映射和数学/FFTS依赖，MIX共同开始，每核成员区间互斥，
允许全局换序而非前轮相邻交换。877单元、973成员、2241顺序二进制、34641约束；
已把原773.41μs共同开始安排代入所有约束，最大误差5.68e-14，证明模型保留合法基线。
求解120秒，8线程。若有解还要独立检查数值、同核区间，补足真实核组就绪前驱、重建进度表
并检查偏序，不能把模型共同开始当作已实现的设备同步。目前没有新NPU任务或性能结论。


### 2026-10-17T12:53:57：MILP前两轮结束，约束闭包验证通过，开始定向可行性搜索

v1在120秒内没有找到整数可行解，solver status=1；不能据此称不可行，已有合法基线见证保留。
weighted_precedence.py对带时间差DAG约简24825→20402条边，逐对242701个可达关系的最长
时间差全部保持，最大误差0。另4457条资源顺序已由数学路径蕴含，v2模型约束34641→25762。
v2最多允许96对顺序偏离基线，包含成组换序机会，120秒找到773.41μs合法安排但没有改善。
候选独立约束/整数/同核区间检查PASS；补230条真实MIX就绪前驱后模型仍773.41；
通知/等待翻译的偏序一致，远端等待12176、发布754。
求解gap12.65%，未证明最优，不能把675.54的模型下界当设备下界，也不能当实测达标。
源码v1/v2快照、完整日志、候选图均保留，没有上卡。

v3改为零目标函数的可行性问题，限定模型安排≤750μs，继续相同96对邻域和数学约束；
这是筛选有意义改进的门槛，不是把设备700目标改为750。程序单独读取makespan变量，
不会把零目标值当运行耗时。当前120秒搜索中；不存在新设备任务。


### 2026-10-17T13:02:43：固定分核MILP定向搜索终态，保持现选

v3限定模型≤750μs、最多96个顺序变化，120.074秒内未找到整数可行解；不是不可行证明，也不是700目标达标。全部求解日志与audit保留，无设备任务。当前三个MILP方向均结束，不扩大重复搜索。全部CPU任务已终态，修正PENDING_JOBS旧会话30868。
下一步只做源码级RoPE相邻交换CPU探针，评估是否能合法避开gather触发的整函数冗余同步消除禁用；不删除PTOAS安全检查、不修改依赖、无性能结论。用户明确允许按batch/长度档位分别编译，当前已按此约束实现七档，继续保留。


### 2026-10-17T13:06:38：RoPE交换与批量gather候选

rope_pair_swap_probe_v1在PTOAS因[256,2] FP32行只有8B而失败，A3要求32B对齐。v2用[64,8]转置、4对行extract/assemble再转回，完整CPU编译通过，无gather，但新增两次转置和8行复制；仅保留可行性证据，未上卡、未称性能改善。
另新增rope_flat_gather.py，保留原输入索引，改第r行的c为r*64+c；仅替换kv_rms_norm_rope(32行)、rmsnorm_rope_cache_write(16行)、rmsnorm_rope(16行)的三个逐行gather循环。修改外AST逐项复原一致，枚举262144个合法行/列选择验证索引归属。GM、原数学/尾块/所有跨核同步不改；单行tail保持原样。完整CPU准备进行中，无设备任务。


2026-10-17T13:08:57：rope_flat_gather_prepare_v1完整准备通过；CPU v1命令误用static路径，FileNotFoundError，未进入编译。v2按实际variants目录开始编译。新增默认关闭--rope-flat-gather接入，integrated prepare_v2进行中；数学体与编译候选核对后才提交设备。详见ROPE_GATHER_EXPERIMENT.md。


### 2026-10-17T13:11:33：批量RoPE完整CPU通过并提交B16

rope_flat_gather_cpu_v2 full_codegen PASS。实际三个TGATHER分别为1024/1024/2048 FP32元素，索引均由原列号加整数((linear>>6)<<6)产生；同时活跃的源/输出/INT32索引/scratch分别核对地址区间，无重叠。完整报告rope_flat_gather_codegen_audit_cpu_v2.json。集成prepare_v2全部AST与已编译v1相同，不重复数学编译。新增执行前实际产物门禁，仍在所有编译扩展入栈后解析artifact，100tick检查保持。
task_20261017_131052_534942170 / b16_row_deps_rope_flat_v1，只测128K/B16/S6，正式权重、原动态PTO对照、完整exact/TopK/六cache-state/30代/保护区及三窗L1。提交时无设备结论，不改七档选择；执行中冻结导入driver/helper。


### 2026-10-17T13:15:08：批量RoPE完整设备PASS，未采用

task_20261017_131052_534942170 exit0；eager/graph/final输出、TopK、六cache-state均零差异，30代与保护区全部PASS。实际三处Tile/索引链/UB与100tick等待门禁PASS。完整L1 819.58/815.10/810.50，中位815.10；同轮动态892.30/923.18/920.58，中位920.58。没有优于现选证据，未采用、不扩档；没有本轮现选配对，不精确归因跨轮差额。源码和原始结果保留，不重复该候选。

新增compile_outlined_cached_wait.py，只在CPU组合已有全阶段独立/noinline代码生成与现选100tick等待；数学使用现选原版本，不叠加批量RoPE。原全阶段独立实验发生在旧默认等待时期，不能直接当现选组合的性能结论。未删除driver的组合保护；先核对全部阶段外层事件闭合、helper自身事件保持及最终产物等待替换，尚无新设备结论。该C++组合仍是诊断，不作为生产或700目标完成。


### 2026-10-17T13:19:27：全阶段独立编译加100tick CPU通过，窄范围开放组合诊断

outlined_cached_wait_cpu_v1完整CPU通过，74个noinline helper覆盖全部现选阶段，外层AIC/AIV数学事件只在成对闭合且全部阶段已分离时清理；helper自身事件不改。实际两个CCEC源码各74个CSA_WAIT_LOCAL_GM、无TWAIT残留，DCCI后volatile读取与100tick均在。独立门禁首次仅因后端将AICORE正规化成static __aicore__而拒绝，修正识别语法后完整覆盖/外层无事件/等待链检查PASS，未改候选生成代码。
只开放B16/128K、全部阶段、compact参数、noinline、现选QK索引和rowdeps/earliest_ready/cached100组合，部分阶段仍拒绝。task_20261017_131724_17808923318被另一处旧rowdeps组合保护拒绝，exit1，未执行算子；保留失败日志。该保护只对已受上述严格检查的cached组合开放，CPU直接执行参数解析/校验前缀，确认原现选和允许组合PASS，部分阶段和缺noinline仍拒绝。
重提task_20261017_131834_1981138300 / b16_row_deps_outlined_cached_v2；仍完整exact/30代/保护区及3窗L1，无Native。此C++组合归因eligible_for_700us_target=false，不把诊断数值记作正式实现达标，不改七档选择。


### 2026-10-17T13:22:21：分函数与现选等待组合完整PASS，性能持平

task_20261017_131834_1981138300 exit0。现选数学/行依赖与全部阶段独立noinline+cached100的组合完整通过：eager/graph/final输出、TopK、六状态零差异，30代通知与所有保护区PASS，实际74helper/事件闭合/100tick门禁PASS。完整L1 807.02/805.34/810.66，中位807.02；同轮动态886.36/906.28/902.84，中位902.84。与现选约807持平，不采用、不扩档、不再重复该组合。该归因诊断eligible_for_700us_target=false。不能把静态屏障数减少直接当收益，也不能据此断言同步完全没有成本：函数调用/独立规划可能抵消其变化。
全部任务已终态，七档选择不变，700目标仍未达到，goal active。


### 2026-10-17T13:26:47：下一方向的主Q生产者范围CPU核对

qproj_dequant_ownership.py仅按冻结源码公式枚举B16/T96，不改数学或依赖。现有head_tile16时每个后处理worker读32个N256列块，覆盖全部20个投影worker，因此直接删一部分父核是不成立的。head_tile4重新分配行/head后，每个后处理worker仍执行16个head×8行的数学迭代，但只需8个投影worker；全部12×64个行/head块各有唯一写者。head_tile8需16父核，head_tile1/2也需8父核，但QK关联的后处理worker扩大到48/32；tile4和当前16的QK关联均为16。
这只是读写范围与公式枚举，尚未核对完整GM别名/其他共享产物、绑定常量、全图偏序或实际额外循环开销。没有新等待表，没有CPU时间收益预测，没有设备候选；不能直接删边上卡。下一步先补完整合同并以CPU模型筛选，再决定是否值得编译。该方向与已关闭的仅改投影worker数/主Q后移/Tile预取不同。


### 2026-10-17T13:42:01+08:00：主Q后处理依赖范围模型终态及用户分档澄清

qproj_dequant_dependencies.py已核对投影INT32共享视图、后处理BF16输出与QK reshape地址一致、额外共享只读输入及唯一写者。保留原同核顺序和216条FFTS复用边；通知/等待翻译偏序检查通过。qproj_dequant_dependency_model_v1三种head_tile16/8/4均为771.53μs，未缩短CPU模型关键路径；额外后处理成本1/2/4μs分别使整层等量变长。未修改数学分块、未上卡、未采用，不将模型值作为设备性能。该方向暂结束。

用户进一步明确：参考hc_pre.py按seqlen区间和batch场景选择实现策略，七个历史性能点不能作为算子支持范围。只读核对hc_pre入口实际按T<=32、33..48、49..95、精确96、97..144、>144分支；CSA不能照搬其数值，须结合Indexer历史长度分支、B/T及工作区容量。已有CSA冻结源本身是S6契约，尚不能声称支持其他S。后续先落实范围策略与测试点的区别，128K/B16低于700μs的目标不变。

重新读取七份原始result.json并由每侧3个L1窗口重算中位数，全部与selected_shape_policies一致；未新测、未运行Native。归档selected_seven_readback_20261017.json。

## 2026-10-17T14:00:38.544110+08:00：全档30%新目标及区间入口

用户确认每档单SPMD≤同轮动态分task PTO×0.70，替换原B16<700us单点门槛。原目标与原始粘贴文件均保留。七档历史L1原始起止回读一致，无新测试，差距见target_30pct_historical_gap.json。

新增range_policy.py，按原数学区间、B/T及波次选择策略；七档不是支持白名单。原192点分类、384端点、七点已选策略一致，成本严格全root shape与H匹配。CLI显式默认调优参数被静默覆盖的问题已修，首轮失败及7反例修复证据保留。

新增源码DAG fallback，合并前后期读写效应与显式deps/dummy/manual合同，检查RAW/WAW/WAR，保持MIX配对和通知协议。B12/H65536及缺参考的B17/H65536完整生成/协议/PYPTO lowering PASS；temporary_54 ScalarWriteLineShared告警对应INT32[T,16]逐行信号，行距64B，未修改同步代码。CPU不代表设备。两点分别提交 task_20261017_140019_112364329866 / task_20261017_140020_112371518858；未提前记精度或性能PASS。

### 区间代表点真机完成

上述两任务均exit0。B12/H65536动态714.66、单SPMD690.70μs，降低3.35%；B17/H65536动态893.52、单SPMD863.08μs，降低3.41%。eager/graph/final输出、TopK及六state均通过exact，信号代际/保护区PASS；原始L1起止重新计算一致，单SPMD各窗仅1task/72成员。B17明确使用source_derived_task_dag。均未达到30%目标，不把两个代表点当全区间证明。摘要range_entry_device_results_v1.json。

### 新目标后的独立源码与性能调查

七档阶段读回及O-A双输出tile输入复用见seven_stage_opportunities_v1/README.md。T192单group资源探针v2 CPU通过，Mat327680/Left49152/Right65536/Acc98304B，但Left请求depth2实际退为1；真实跨8组NZ view/stride合同尚未证明。v1 JIT TensorView根参数不支持，v3显式内层展开Right131072B超64KB；保留失败，不上卡、不当性能收益。

query_shape_support_audit_v1给出S5 pair跨请求、双AIV奇数尾漏query、O投影容量320非128整除及mixed[2,4,6]请求定位反例。原动态与冻结IR都需处理，不只是fixture常量；当前S6范围如实标记为限制，未收缩最终目标。

为补新source-DAG的动态metadata证据，提交task_20261017_140653_1348347342，H65536→65535→32766→65536同地址同二进制图重放，C点L32772是长历史下边界。不重采性能。

### B17新源码DAG的动态metadata边界PASS

task_20261017_140653_1348347342 exit0。固定地址/容量/同一二进制，H65536→65535→32766→65536（C1的L32772为长历史策略下边界），页表交换与位置/RoPE等原址更新，输出/TopK/state exact及保护区通过。每侧仅一次图捕获，无新增编译单元，变体间未清信号。此任务不采性能，也不是连续decode轨迹。40秒wait观察超时后继续核对原task，正常完成，未重提。

<!-- SOURCE_END history-document-042 -->

---

<a id="history-document-043"></a>
### 历史原文 043：SELECTED_SEVEN_RESULTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/SELECTED_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/SELECTED_SEVEN_RESULTS.md)；原文 2,252 字节。动态目标、过程总账或七档结果，全文保留。

<!-- SOURCE_BEGIN history-document-043 -->
# 当前保留的七档候选

单位μs，完整CSA L1 incore首尾，三窗中位数。每一行使用所选实现自己的同轮动态PTO对照。

| 历史长度 | Batch | 单SPMD | 同轮动态PTO | QK地址缓存 |
|---:|---:|---:|---:|:---:|
| 131072 | 4 | 541.92 | 582.82 | 关闭 |
| 131072 | 8 | 641.56 | 679.84 | 启用 |
| 131072 | 16 | 807.16 | 905.08 | 启用 |
| 131072 | 24 | 1121.12 | 1169.46 | 关闭 |
| 8192 | 16 | 621.96 | 680.02 | 关闭 |
| 8192 | 24 | 836.74 | 861.62 | 关闭 |
| 8192 | 32 | 905.72 | 972.72 | 关闭 |

七档输出、Top-K、状态、图重放和保护区均通过。原静态及各负结果继续保留。
128K/B16现选首轮中位807.16μs，目标<700μs尚未达到。
这仍是固定fixture重放，不代表连续decode或整模型性能。

QK地址缓存仅选128K/B8、B16，其余五档保持旧实现。B4两版相差0.08μs，视为持平。
B8改进不到1%，三窗只支持保留候选，不构成长期稳定收益证明。

B16探针只显示QK/PV成员数学加通知中位约减少2μs；不能把无插桩跨轮16.44μs差值全部归因于这一段。
探针增加读时钟/写记录并改变代码布局，不能替代无插桩完整L1计时。

[全部新旧数据](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_SEVEN_RESULTS.md)；[机器可读候选配置](../../../tests/pypto_test/csa_multishape_spmd_20261016/selected_shape_policies.json)；
[阶段探针对比](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_stage_comparison.json)。

各档独立编译；实际长度、位置、slot和页表继续从device Tensor读取。
没有要求不同batch共用二进制，也没有为本实验更改生产入口或依赖仓。

2026-10-17新增：仅128K/B16启用按query/head细化的成员依赖。独立同进程复核中，
此前单SPMD为841.94μs，新候选817.88μs（各3窗，缩短2.86%）；这是另一轮测量，
不与上表同轮原动态数据混用。新候选保持同图、同地址、同二进制通过
131072→131071→65536→131072，Indexer从6个leaf变成3个再恢复。
见[依赖细化及动态验证](../../../tests/pypto_test/csa_multishape_spmd_20261016/ROW_DEPENDENCY_EXPERIMENT.md)。其余六档不重测、不换表。


2026-10-17 09:26补记：KV纯信号候选同进程对照中，现选B16三窗805.14/808.22/807.50μs，
中位807.50；候选中位812.08，无收益。没有更新七档选择，详见KV_TOUCH_SIGNAL_EXPERIMENT.md。

<!-- SOURCE_END history-document-043 -->

---

<a id="history-document-044"></a>
### 历史原文 044：BALANCED_SEVEN_RESULTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/BALANCED_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/BALANCED_SEVEN_RESULTS.md)；原文 1,164 字节。动态目标、过程总账或七档结果，全文保留。

<!-- SOURCE_BEGIN history-document-044 -->
# 按规格重分配候选：CSA 七档对照

正式 W8A8 权重，S=6。比较原动态 PTO 与单 SPMD；没有执行 Native forward。
全部七档的 eager/graph 输出、Top-K、六份 state/cache、保护区和 30 代信号检查通过。
每次调用恢复同一 fixture，不代表连续 decode 状态轨迹已验证。

计时取三次 L1 的完整 incore 首尾中位数，单位 μs；没有阶段插桩。

| 历史长度 | B | 原动态 PTO | 单 SPMD | 耗时变化 |
|---|---:|---:|---:|---:|
| 131072 | 4 | 579.26 | 791.74 | +36.68% |
| 131072 | 8 | 683.02 | 940.14 | +37.64% |
| 131072 | 16 | 902.28 | 1067.36 | +18.30% |
| 131072 | 24 | 1166.84 | 1533.28 | +31.40% |
| 8192 | 16 | 656.82 | 911.72 | +38.81% |
| 8192 | 24 | 845.28 | 1196.20 | +41.52% |
| 8192 | 32 | 1004.50 | 1278.36 | +27.26% |

本候选仍慢于原动态 PTO，128K/B16 的 700 μs 目标尚未完成。
相对旧捕获映射的 B16 约 1380.22 μs，重分配后为 1067.36 μs；旧值来自此前测量。
原始样本、每窗泳道及检查详情位于 JSON 中 case 指向的目录。
这一版只启用 balanced_schedule，不启用通知汇聚或 C++ 数学替换。

<!-- SOURCE_END history-document-044 -->

---

<a id="history-document-045"></a>
### 历史原文 045：CACHED_SEVEN_RESULTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/CACHED_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/CACHED_SEVEN_RESULTS.md)；原文 2,919 字节。动态目标、过程总账或七档结果，全文保留。

<!-- SOURCE_BEGIN history-document-045 -->
# CSA 多档单 SPMD：DCCI 轮询退避与分核实测

2026-10-16，正式模型第4层权重、合成历史，S=6。只比较原动态 PTO 与单 SPMD，未运行 Native forward。
计时为完整 CSA 的 L1 incore 最早开始到最晚结束，每侧3窗取中位数；不计入外部调度头尾。
七档各自编译，采用各自同规格既有动态 L1 的成本和调度策略；均只在信号未就绪时退避100 tick（50MHz下2μs）。

|历史|B|原动态 L1 μs|单 SPMD L1 μs|耗时变化|单 SPMD 三窗 μs|策略|
|---:|---:|---:|---:|---:|---|---|
|131072|4|582.82|541.92|-7.02%|541.92 / 541.98 / 537.00|earliest_ready|
|131072|8|684.32|647.48|-5.38%|647.48 / 649.44 / 644.14|earliest_ready|
|131072|16|892.78|832.28|-6.78%|828.36 / 832.28 / 834.48|earliest_ready|
|131072|24|1169.46|1121.12|-4.13%|1121.12 / 1114.96 / 1124.70|critical_path|
|8192|16|680.02|621.96|-8.54%|613.20 / 622.04 / 621.96|critical_path|
|8192|24|861.62|836.74|-2.89%|838.36 / 834.08 / 836.74|earliest_ready|
|8192|32|972.72|905.72|-6.89%|898.08 / 905.72 / 916.86|critical_path|

七档全部通过 eager、aclgraph 及最终输出/Top-K/六份状态的 exact 对照，Native 和工作区保护区通过，30代信号正确。
每次调用恢复同一 fixture；不把这组结果扩称为连续 decode 轨迹或整网服务验证。
128K/B16 的832.28 μs尚未达到700 μs目标。生产入口没有切换。

图重放 Event 另行报告，它包含的范围与 L1 不同，部分小档仍慢于原动态，不能将 L1 收益说成端到端全面收益。

|历史|B|原动态 Event μs|单 SPMD Event μs|
|---:|---:|---:|---:|
|131072|4|637.01|648.46|
|131072|8|728.59|746.33|
|131072|16|959.50|931.23|
|131072|24|1229.54|1230.75|
|8192|16|717.97|724.35|
|8192|24|909.47|941.56|
|8192|32|1043.86|1025.07|

## 复现与证据

各项任务号、三窗样本和相对路径在 [cached_seven_results.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/cached_seven_results.json)，CSV 同名。
每个结果目录保存 result.json、原动态/单 SPMD各3窗L1、实际入参绑定、分核表和生成的PyPTO源码。
业务长度、位置、slot和页表仍在设备读取；初始化只根据已知shape/stride和Host长度策略选择模板。
不做额外hash检查；大编译产物留独立worktree，路径清单见LOCAL_ARTIFACTS.json。

执行方式：

```bash
task-submit --device auto --max-time 1800 'bash <本目录>/run_device_compare.sh <B> <history> <新结果目录> --balanced-schedule --cached-wait --wait-backoff-ticks 100 --schedule-policy <本档策略> --schedule-cost-profile <本档成本JSON>'
```

128K/B16本次使用 b16_dynamic_l1_cost.json，其余六档使用 per_shape_costs/h<history>_b<B>.json。
读侧每次读前DCCI，写侧DCCI+DSB和业务payload维护均保留；不是依赖Scalar缓存一致性。
等待helper的本地生成边界适配位于signal_polling.py，后续主线notify/wait完备后再收敛替换。

<!-- SOURCE_END history-document-045 -->

---

<a id="history-document-046"></a>
### 历史原文 046：QK_SEVEN_RESULTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_SEVEN_RESULTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_SEVEN_RESULTS.md)；原文 1,607 字节。动态目标、过程总账或七档结果，全文保留。

<!-- SOURCE_BEGIN history-document-046 -->
# 七档QK/PV地址缓存对比

单位：μs。完整CSA的L1 incore首尾耗时，三窗中位数。
旧/新单SPMD为跨轮比较；新任务各自包含原动态PTO对照，不重跑旧单SPMD。

| 历史长度 | Batch | 原单SPMD | 新QK地址缓存 | 新同轮动态PTO | 相对原单SPMD |
|---:|---:|---:|---:|---:|---:|
| 131072 | 4 | 541.92 | 541.84 | 581.96 | -0.01% |
| 131072 | 8 | 647.48 | 641.56 | 679.84 | -0.91% |
| 131072 | 16 | 832.28 | 815.84 | 902.80 | -1.98% |
| 131072 | 24 | 1121.12 | 1137.90 | 1167.68 | +1.50% |
| 8192 | 16 | 621.96 | 632.58 | 660.12 | +1.71% |
| 8192 | 24 | 836.74 | 865.16 | 858.48 | +3.40% |
| 8192 | 32 | 905.72 | 943.20 | 979.52 | +4.14% |

七档均通过输出、Top-K、六份状态、图重放、进度代际及保护区检查。
这是固定输入恢复后的重放，不能替代连续decode轨迹验证。

新路径仅在当前压缩KV块内，把索引和页表载入本核UB，用Vector整数计算物理行。
Scalar从UB取地址，原逐行KV DMA、负索引分支、attention算术及FFTS保持。
Native metadata继续通过device Tensor读取，无Host回读、无额外GM展开缓冲。

所有任务使用各规格原有调度选择、cached等待与100tick退避。仅新增--qk-vector-indices；
没有叠加GM复用或主Q重分片。B16阶段探针另存，不进入此表或700μs门槛。

每档三次采样不足以证明小幅差值稳定；原始L1与Event数据保留在
[JSON](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_indices_seven_results.json)与[CSV](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_indices_seven_results.csv)，
两种计时范围不同，不能把L1改善直接等同于整模型收益。

<!-- SOURCE_END history-document-046 -->

---

<a id="dynamic-special"></a>
## 动态专项实现、诊断与失败尝试

- [047 · csa_multishape_spmd_20261016/COMPILE_CONTEXT_CORRECTION.md](#history-document-047)
- [048 · csa_multishape_spmd_20261016/DYNAMIC_METADATA_VALIDATION.md](#history-document-048)
- [049 · csa_multishape_spmd_20261016/FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md](#history-document-049)
- [050 · csa_multishape_spmd_20261016/FIXED_CORE_MILP_EXPERIMENT.md](#history-document-050)
- [051 · csa_multishape_spmd_20261016/HC_FRONT_AND_INPUT_DEPENDENCIES.md](#history-document-051)
- [052 · csa_multishape_spmd_20261016/HC_MIX_CHUNK_BUFFER_EXPERIMENT.md](#history-document-052)
- [053 · csa_multishape_spmd_20261016/HC_MIX_QPROJ_EXPERIMENTS.md](#history-document-053)
- [054 · csa_multishape_spmd_20261016/HC_PRE_REFERENCE_REVIEW.md](#history-document-054)
- [055 · csa_multishape_spmd_20261016/HC_ROW_SPLIT_REVIEW.md](#history-document-055)
- [056 · csa_multishape_spmd_20261016/HC_STREAM_PRE_SYNC_AUDIT.md](#history-document-056)
- [057 · csa_multishape_spmd_20261016/INDEXER_KEY_PAIRS_EXPERIMENT.md](#history-document-057)
- [058 · csa_multishape_spmd_20261016/INDEXER_KEY_POOL_CPU_EXPERIMENT.md](#history-document-058)
- [059 · csa_multishape_spmd_20261016/INDEXER_L1_RING_EXPERIMENT.md](#history-document-059)
- [060 · csa_multishape_spmd_20261016/INDEXER_N32_ACC_EXPERIMENT.md](#history-document-060)
- [061 · csa_multishape_spmd_20261016/INDEXER_N80_EXPERIMENT.md](#history-document-061)
- [062 · csa_multishape_spmd_20261016/INDEXER_N80_WS128_EXPERIMENT.md](#history-document-062)
- [063 · csa_multishape_spmd_20261016/INDEXER_PAGE_BASE_SELECT_CPU.md](#history-document-063)
- [064 · csa_multishape_spmd_20261016/INDEXER_PANEL_GROUP_EXPERIMENT.md](#history-document-064)
- [065 · csa_multishape_spmd_20261016/INDEXER_PARTITION_EXPERIMENTS.md](#history-document-065)
- [066 · csa_multishape_spmd_20261016/INDEXER_ROLLED_PANELS_EXPERIMENT.md](#history-document-066)
- [067 · csa_multishape_spmd_20261016/INDEXER_SHORT_TAIL_EXPERIMENT.md](#history-document-067)
- [068 · csa_multishape_spmd_20261016/INDEXER_TRANSFER_2048_EXPERIMENT.md](#history-document-068)
- [069 · csa_multishape_spmd_20261016/INDEXER_WS_LOAD_ORDER_CPU.md](#history-document-069)
- [070 · csa_multishape_spmd_20261016/KV_TOUCH_SIGNAL_EXPERIMENT.md](#history-document-070)
- [071 · csa_multishape_spmd_20261016/MEMBER_SCHEDULE_CPU_EXPERIMENT.md](#history-document-071)
- [072 · csa_multishape_spmd_20261016/OUTLINED_CACHED_WAIT_EXPERIMENT.md](#history-document-072)
- [073 · csa_multishape_spmd_20261016/PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md](#history-document-073)
- [074 · csa_multishape_spmd_20261016/QK_ALL_VALID_EXPERIMENT.md](#history-document-074)
- [075 · csa_multishape_spmd_20261016/QK_DIRECT_L1_EXPERIMENT.md](#history-document-075)
- [076 · csa_multishape_spmd_20261016/QK_EARLY_KV_EXPERIMENT.md](#history-document-076)
- [077 · csa_multishape_spmd_20261016/QK_MERGE_STATE_EXPERIMENT.md](#history-document-077)
- [078 · csa_multishape_spmd_20261016/QK_QUERY_METADATA_EXPERIMENT.md](#history-document-078)
- [079 · csa_multishape_spmd_20261016/QK_ROLE_SPLIT_EXPERIMENT.md](#history-document-079)
- [080 · csa_multishape_spmd_20261016/QK_SYNC_FALLBACK_ANALYSIS.md](#history-document-080)
- [081 · csa_multishape_spmd_20261016/QPROJ_L1_COLUMNS_EXPERIMENT.md](#history-document-081)
- [082 · csa_multishape_spmd_20261016/QUERY_GROUP_SCHEDULE_FEASIBILITY.md](#history-document-082)
- [083 · csa_multishape_spmd_20261016/RAMP_WAIT_EXPERIMENT.md](#history-document-083)
- [084 · csa_multishape_spmd_20261016/RANGE_POLICY.md](#history-document-084)
- [085 · csa_multishape_spmd_20261016/ROPE_GATHER_EXPERIMENT.md](#history-document-085)
- [086 · csa_multishape_spmd_20261016/ROW_DEPENDENCY_EXPERIMENT.md](#history-document-086)
- [087 · csa_multishape_spmd_20261016/TILE_LOOP_INPLACE_CPU.md](#history-document-087)
- [088 · csa_multishape_spmd_20261016/WEIGHT_CACHE_POLICY_EXPERIMENT.md](#history-document-088)
- [089 · csa_multishape_spmd_20261016/anchor_gap_design_v1/README.md](#history-document-089)
- [090 · csa_multishape_spmd_20261016/hc_front_cost_audit_20261017/README.md](#history-document-090)
- [091 · csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/README.md](#history-document-091)
- [092 · csa_multishape_spmd_20261016/hc_sinkhorn_scalar_v1/README.md](#history-document-092)
- [093 · csa_multishape_spmd_20261016/observed_member_cost_v1/README.md](#history-document-093)
- [094 · csa_multishape_spmd_20261016/production_spmd_package_audit_20261017/README.md](#history-document-094)
- [095 · csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/README.md](#history-document-095)
- [096 · csa_multishape_spmd_20261016/query_shape_support_audit_v1/README.md](#history-document-096)
- [097 · csa_multishape_spmd_20261016/range_policy_validation_v1/README.md](#history-document-097)
- [098 · csa_multishape_spmd_20261016/range_source_audit_v1/README.md](#history-document-098)
- [099 · csa_multishape_spmd_20261016/seven_stage_opportunities_v1/README.md](#history-document-099)

<a id="history-document-047"></a>
### 历史原文 047：COMPILE_CONTEXT_CORRECTION.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/COMPILE_CONTEXT_CORRECTION.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/COMPILE_CONTEXT_CORRECTION.md)；原文 4,645 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-047 -->
# 2026-10-17：提前审计JIT产物绕过等待策略，修正三项性能比较

这是本目录测试驱动的错误，不是PyPTO或Simpler的编译正确性问题。
三个候选的实际设备输出检查仍然有效，但测到的二进制没有应用所请求的
`cached_wait + 100 tick`等待策略，不能把它们相对现选实现的耗时差归因于数学核改动。
原始日志、结果和源码全部保留；本文件更正此前“不采用/不重测”的性能判断依据。

## 原因和直接证据

`device_compare.py`为三个候选增加了首次执行前的实际地址/指令审计。
审计调用`module.entry._resolve_kernel_artifact(...)`，它会实际编译并缓存artifact。
原位置在`ExitStack`之前，因此尚未进入`bypass_signal_wait(...)`编译扩展。
后面的首次执行命中该artifact，编译回调不再发生。命令行和result中的
`cached_wait=True, wait_backoff_ticks=100`只记录请求，不能证明实际应用。

PyPTO本地`jit/decorator.py`中`_resolve_kernel_artifact`调用`_resolve_compiled`，
后者使用`self._cache[memory_key]`保存产物。这正是其正常语义。

| 产物 | AIC/AIV各自默认TWAIT调用 | 替换后的等待调用 | 100 tick退避 |
|---|---:|---:|---|
| 现选`b16_cached_row_deps_v1` | 0 | 74 | 有 |
| `b16_row_deps_indexer_l1_ring_v4` | 74 | 0 | 无 |
| `b16_row_deps_qk_merge_state_v6` | 74 | 0 | 无 |
| `b16_row_deps_indexer_ws_order_v5` | 74 | 0 | 无 |

核对的是实际交给CCEC的`build_output/entry_*/kernels/{aic,aiv}/*.cpp`。
`ptoas/*.cpp`保存替换前源码，即使正确基线也仍有默认TWAIT，不能据它判断等待策略。
三个问题产物的`signal_polling/change.json`也都缺失，证明替换回调没有执行。
证据见`compile_context_audit_20261017.json`及`compile_context_scope_audit_20261017.json`。

## 修正和防止重现

1. 三处artifact审计移入同一个`ExitStack`，所有编译扩展先入栈，再解析artifact。
2. 第一次执行前调用`signal_polling.audit_wait_artifact`，同时检查编译记录、实际AIC/AIV
   源码、所有等待调用的替换、DCCI后读取以及实际退避常量。不匹配则禁止执行。
3. 使用已有正确产物作正例、三个错误产物作反例完成CPU检查；即使给错误产物搭配
   正确change.json，实际源码检查也会拒绝。没有放宽输出或状态精度门槛。
4. 使用新输出目录和独立缓存重测受影响候选，避免复用旧默认等待二进制。

最早的现选manifest没有后来新增的`ramp_backoff`字段；首次CPU正例因此失败。
随后用已有明确记录`ramp_backoff=false`的正确产物验证门禁，最早现选的实际源码单独核对。
这不是设备失败，也没有修改或补写历史manifest。

## 受影响的历史测量

全部为128K/B16/S6，完整CSA的L1 incore首尾，单位μs；数值是实际测量，配置与预期不符。

| 候选 | 历史单SPMD中位数 | 同轮动态PTO | 任务 |
|---|---:|---:|---|
| N64 L1环 | 1008.86 | 905.34 | `task_20261017_094749_7744056502` |
| PV循环状态分块 | 1033.44 | 912.20 | `task_20261017_103218_149245329068` |
| WS后14对加载后移 | 1033.52 | 898.64 | `task_20261017_105218_188961010694` |

三项均完成eager/graph/final输出、Top-K、六份cache/state、30代通知和保护区验证。
这些正确性结果针对实际默认等待二进制；修正后的二进制仍需重新通过相同门槛。
不能预先认定修正等待策略后这三项一定提速，也不能把历史差值全算作等待策略开销。

修正后的WS候选提交为`task_20261017_110148_204947624179`，输出
`b16_row_deps_indexer_ws_order_v6`；最终结果在任务结束后补记。
七档现选策略未因本次发现改变，128K/B16低于700μs的目标仍未达到。

## 修正后的设备结果

三项均完成同一精度门槛，实际AIC/AIV等待策略门禁通过；任务均exit0。

| 候选 | 三个完整L1窗口 μs | 中位数 | 同轮动态PTO中位数 | 任务 |
|---|---|---:|---:|---|
| WS后14对加载后移 v6 | 814.12 / 812.34 / 816.48 | 814.12 | 899.98 | `task_20261017_110148_204947624179` |
| PV状态分块 v7 | 820.58 / 812.90 / 820.24 | 820.24 | 909.56 | `task_20261017_110518_211373122343` |
| N64 L1环 v5 | 804.14 / 813.50 / 806.38 | 806.38 | 885.92 | `task_20261017_110544_21223701219` |

本轮不替换七档选择。N64环与此前现选807.16/807.50接近，未证明稳定收益；本轮没有同进程现选配对。
PV与WS也没有支持更换现选的证据。以上均为每次恢复fixture的图重放，不是连续decode。

<!-- SOURCE_END history-document-047 -->

---

<a id="history-document-048"></a>
### 历史原文 048：DYNAMIC_METADATA_VALIDATION.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/DYNAMIC_METADATA_VALIDATION.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/DYNAMIC_METADATA_VALIDATION.md)；原文 4,415 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-048 -->
# CSA 单 SPMD：分档与同地址动态 metadata 验证

按用户要求，不要求多个 batch 共用一份二进制。编译/初始化使用已有
Tensor shape/stride 及 Host 长度上界选择原有数学分支；每一步的实际长度、
位置、slot、页表继续读取设备 Tensor。不同档位的已测策略保留在
`selected_shape_policies.json`，不能把一个档位的优化直接设成所有档位的默认值。

## 128K/B16/S6 已通过的功能验证

任务 `task_20261017_023022_12848062458`，结果目录 `b16_metadata_replay_v1`。
使用正式 `/data/model/DeepSeek-V4-Flash-0731-w8a8` 第4层权重及合成历史，
对比完整 CSA attention 半层的单 SPMD 与冻结的原动态 PTO，不执行 Native forward。

| 重放 | 历史长度 | 请求物理页归属 | 原动态 eager / 原动态图 / 单 SPMD 图 |
|---|---:|---|---|
| A | 131072 | 原顺序 | PASS |
| B | 131071 | 交换顺序 | PASS |
| C | 130048 | 交换顺序 | PASS |
| A_return | 131072 | 恢复原顺序 | PASS |

每侧只捕获一次图，所有根参数和工作区地址保持不变，变体之间不清零进度。
B/C 的参考输出确实区别于 A，恢复 A 后参考输出完全一致。单 SPMD 累计6代
信号检查通过；重放期间没有新增编译单元。

所有变体的 `x_out`、Top-K 和六份 cache/state 均逐bit匹配原动态 eager。
完整 allocation 的非 slot 写区、只读 metadata 与工作区首尾保护区也都通过。
这是独立恢复历史后的四组用例，不代表连续 decode、同批异构请求长度或整网验证。

B/C 实际改变以下19个设备根输入；读取 CPU 内容仅发生在测试构造和结果比较中，
没有给生产算子增加 Host tiling-update：

- 长度/位置：`cmp_seq_lens`、`kv_seq_lens`、`position_ids`。
- 页表：`cmp_block_table`、`idx_block_table`、`ori_block_table`、
  `state_block_table`、`inner_state_block_table`。
- slot：`cmp_slot_mapping`、`idx_slot_mapping`、`ori_slot_mapping`、
  `state_slot_mapping`、`inner_state_slot_mapping`。
- RoPE：`freqs_cos/sin`、`cmp_freqs_cos/sin`、`inner_freqs_cos/sin`。

## 复现入口与性能口径

实现为 `dynamic_metadata_replay.py`，由 `device_compare.py --metadata-replay-only`
调用。仍须经统一 `task-submit --device auto` 分配设备。基于该档已选调度参数，
在 `run_device_compare.sh` 的正常命令末尾加此标志即可；不得覆盖已有结果目录。

该标志只进行首次编译/eager/图检查和 A→B→C→A，不重复 Event、L1 或 PMU 采样。
结果显式标记 `performance_measured=false`、`eligible_for_700us_target=false`。
真实性能仍以无插桩完整 CSA 的 L1 incore 首尾计时为准，当前选中128K/B16结果
仍为815.84μs，尚未达到700μs目标。

## 长度策略边界

原8K档按 `max_seq_len // 4` 选择 Indexer 的 `[2048,8192]` 压缩候选策略。
`max_seq_len` 含本轮6个query token；均匀fixture对应 history 下界8186。
首轮8K测试的 C=7168 跨入 `<2048` 的另一数学分支，图输出/Top-K比较失败；
A=8192与B=8191已通过，cache/state与保护区没有损坏。失败证据保留在
`short_metadata_failure_v1.json`，不把该结果当作当前档内运行失败。

新增 `metadata_cases` 在派发前用原编排模拟器比较函数调用签名和分支选择；
跨策略候选被拒绝并记录，8K同档用例因此改为8192→8191→8186→8192。
这里只是测试入口的边界检查，生产自动选择尚未接入。跨该边界需选择相应
数学模板/捕获图；本轮没有验证更短历史模板。匹配策略只是必要条件，
还须满足页表/工作区容量和S6契约，不能把少数样例通过说成整个区间已全覆盖。

详细证据：`dynamic_metadata_result.json`、
`b16_metadata_replay_v1/metadata_replay/replay.json`、
`b16_metadata_replay_v1/result.json`。8K/B16 的另一 Indexer 分支另做同类检查，
不能用长历史的通过结果代替。


## 8K/B16/S6 策略内验证结果

修正后的 task_20261017_024535_170390925040 / b16_short_metadata_replay_v2 完整PASS：
8192→8191→8186→8192。每次输出、Top-K、六份状态及保护区均通过，
图和地址固定，6代进度正确，没有新增编译单元，也没有做性能采样。
见 `short_dynamic_metadata_result.json`；首版跨策略失败记录继续保留。

<!-- SOURCE_END history-document-048 -->

---

<a id="history-document-049"></a>
### 历史原文 049：FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md)；原文 3,737 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-049 -->
# 固定分核局部重排的CPU评估（2026-10-17）

两种候选均未缩短模型中的整层关键路径，因此没有生成设备代码或提交NPU任务。
数学、物理核映射、已证明行依赖和216条Indexer→QK的FFTS复用边保持。
这组工作与此前从头分配所有成员的三种贪心排程不同，不是那些模型的重复测试。

## 模型与安全边界

`fixed_core_local_schedule.py`从现选973个成员、877个调度单元及同形动态L1成本开始。
每个MIX核组的三个成员按共同开始时刻建模，每条资源边仍用对应角色的持续时间；
这保证不会把AIC/AIV之间的内部等待误当成可以独立执行的空隙。
纯AIC/AIV任务保持原分核；同核顺序可调整，每次都重新检查完整图是否有环。
只允许数学上无依赖的反序；两个MIX不能只在一条物理核上交换。

所有成本仍为同任务、同角色的历史中位数加1μs，包含成员内等待。
共同开始是模型约束，尚未实现为设备上的新等待表。
模型不含新的发布/等待开销评估，不能把候选图直接投入设备，更不能作为硬件性能下界。

## 固定分核相邻交换

从现选顺序开始，按“整层最长路径、所有成员完成时刻之和”的字典序搜索。
第二项只用于跨越多条等长路径的平台，不作为最终性能指标。
运行120秒，评估7679个交换，拒绝149个有环候选；接受13个等长交换。
整层估计始终为773.41μs，没有缩短。

原逐成员模型为771.53μs，新增共同MIX开始约束后为773.41μs；
这两个数都是模型值，不能代替现选设备L1约807μs。
结果：`fixed_core_local_schedule_cpu_v1/{audit,plan}.json`。

## 成组延后主Q后处理

单次只交换一条核可能跨不过所有核共同决定的阶段边界，因此额外评估一个明确的整组移动：
把48条AIV上位于Indexer之前的主Q反量化/RMS/RoPE成员移到各自Indexer之后。
主Q后处理不是Indexer query的生产者；移动后完整数学/资源图仍无环。
不改变Indexer的逻辑分片或FFTS映射。

估计结果773.41→788.94μs，变差15.53μs。阶段模型显示：

| 阶段 | 局部交换后模型（μs） | 成组延后模型（μs） |
|---|---:|---:|
| Indexer最早开始 | 207.49 | 191.08 |
| Indexer最后结束 | 444.24 | 444.24 |
| merge最后结束 | 455.09 | 472.50 |
| plan最后结束 | 463.59 | 481.00 |
| QK最早开始 | 465.47 | 481.00 |
| 整层最后结束 | 773.41 | 788.94 |

这个模型中，部分Indexer提前并未让最慢核组更早结束，
延后的主Q后处理又占用了merge前的Vector时段，最终延迟QK。
这是模型中的排队解释，不是设备归因。
结果：`fixed_core_defer_q_dequant_cpu_v1/`、`fixed_core_cohort_stage_model_v1.json`。

## 复现与后续

```bash
python fixed_core_local_schedule.py --output 新目录 --seconds 120 --rounds 100
python fixed_core_local_schedule.py --output 另一个新目录 --defer-main-q-dequant --rounds 0
```

`fixed_core_baseline_critical_chains_v1.json`另存纯数学图640.92μs和固定资源图771.53μs的
一条最长路径，仅作定位，不能据此保证700μs可达。
后续若继续排程，应寻找改变最慢Indexer核组释放时刻、同时保住下游merge资源的方案；
不延长同一相邻交换搜索，也不将上述两个负结果上卡。

历史L1原始记录只有物理核、任务ID和派发序号，逻辑分片映射需要匹配的派发日志。
冻结清单的生成器明确校验两者逐项对应，不能按完成顺序猜block_idx。
目前没有为新成本窗口收集或验证这份映射，不伪造逐逻辑分片的精细成本。

<!-- SOURCE_END history-document-049 -->

---

<a id="history-document-050"></a>
### 历史原文 050：FIXED_CORE_MILP_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/FIXED_CORE_MILP_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/FIXED_CORE_MILP_EXPERIMENT.md)；原文 2,878 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-050 -->
# 固定物理分核的全局排程模型

2026-10-17。三轮均没有找到优于原方案的安排，没有生成或测量新的设备版本。
这些数值是历史成本驱动的CPU模型，不是硬件下界，也不是L1实测。

## 约束与实现

`fixed_core_milp_schedule.py`使用已有SciPy 1.13.1所带HiGHS 1.2.0，
保留973个成员、877个调度单元的物理核映射、数学依赖和216条FFTS核组复用边。
MIX的AIC/AIV0/AIV1共用开始变量，同核成员不能同时占用资源。
每对无依赖成员通过二进制变量选择先后；可同时改变多对顺序，区别于相邻交换。

各成员成本为同形动态PTO历史L1同任务/角色的中位数加1μs。
原方案的773.41μs模型安排已代入全部约束，最大残差5.68e-14；
求解超时无解不表示原问题不可行。

`weighted_precedence.py`把24825条数学边约简为20402条，
242701个可达关系的最长时间差均保持，最大误差0；4457条资源顺序已被数学路径蕴含。
约简后模型25762条约束，3119个变量，其中2241个顺序二进制变量。

若找到候选，程序独立复查约束残差、整数性和同核区间；重建实际核组前驱并检查偏序闭包。
模型里MIX共同开始不是免费的设备屏障，不能省略这一步直接上卡。

## 三轮结果

| 目录 | 限制 | 120秒内结果 |
| --- | --- | --- |
| fixed_core_milp_cpu_v1 | 全部顺序变量 | 无整数候选，超时 |
| fixed_core_milp_neighborhood_cpu_v2 | 最多96对顺序偏离原方案 | 773.41μs，无改善 |
| fixed_core_milp_feasible750_cpu_v3 | 同一邻域，只寻找≤750μs安排 | 无整数候选，超时 |

v2候选约束残差1.14e-13、整数误差0；补230条核组就绪前驱后仍为773.41μs。
实际等待表重建得到12176个远端等待、754个发布；未部署。
求解gap为12.65%，没有最优性证明。675.54的模型下界不代表设备下界。
v3零目标函数只用于可行性搜索，程序读取makespan变量，绝不把目标函数0当成耗时。
750μs是筛选模型安排的门槛，没有更改128K/B16设备L1小于700μs的目标。

## 环境与复现

曾尝试在本目录`cache/schedule_solver_env`独立安装ortools：默认源Proxy 403，
官方源重试未得到可安装版本；日志保留，没有修改共享环境。
随后直接使用工作区已有`.venv/bin/python`和SciPy。

```bash
python fixed_core_milp_schedule.py --output 新目录 --seconds 120 \
  --max-order-changes 96 --reduce-precedence
python fixed_core_milp_schedule.py --output 另一个新目录 --seconds 120 \
  --max-order-changes 96 --reduce-precedence --feasibility-target-us 750
```

原始v1/v2源码快照、模型、日志、audit和v2候选图全部保留。
不继续延长相同搜索，不把负结果转成NPU测试；后续需要新的成本或结构依据。

<!-- SOURCE_END history-document-050 -->

---

<a id="history-document-051"></a>
### 历史原文 051：HC_FRONT_AND_INPUT_DEPENDENCIES.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/HC_FRONT_AND_INPUT_DEPENDENCIES.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_FRONT_AND_INPUT_DEPENDENCIES.md)；原文 7,227 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-051 -->
# HC 三阶段融合与 Indexer 输入依赖：2026-10-17

本轮三个真机候选都通过完整用例，但未显示优于现选行依赖版本的明确收益，均未纳入
`selected_shape_policies.json`。128K/B16/S6 现选首轮为 807.16 μs，同进程复核为
817.88 μs；不同运行不能直接据相差几微秒判定新候选更快或更慢。

计时使用**完整、无内部插桩的 L1 incore 首尾**，包括所有等待和阶段间隙。每侧三个窗口，
只比较原动态 PTO 和单 SPMD，正式权重，不运行 Native forward。目标 <700 μs 尚未达到。

| 候选 | 单 SPMD 三窗 / μs | 中位 / μs | 同轮动态 PTO 中位 / μs | 任务 |
| --- | --- | --- | --- | --- |
| HC reduce→split→mix/norm 融合，保留 GM 传递 | 804.84 / 810.80 / 806.76 | 806.76 | 900.14 | `task_20261017_055741_99590920590` |
| 上述融合内改用 Tile 直传 | 819.74 / 810.44 / 810.44 | 810.44 | 891.96 | `task_20261017_061445_12767489743` |
| 原数学及分核，进一步细化 Indexer 输入依赖 | 812.62 / 815.48 / 810.34 | 812.62 | 882.64 | `task_20261017_061600_130106629188` |

三项任务均 exit=0；输出、Top-K、六份 cache/state、图重放、信号代际及保护区检查通过。
这是固定 fixture 恢复后的算子用例，不代表连续 decode 轨迹或整网验证。结构化结果见
[hc_front_and_input_deps_results.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_front_and_input_deps_results.json)。

## HC 三阶段融合

实现：[hc_front_fusion.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_front_fusion.py)，开关 `--hc-front-fusion`。

- reducer 的六个 16 行成员改为十二个 8 行成员，与后续 split/mix 的行分工一致。
- 保持四份投影的累加顺序、四路混合 `(y0+y1)+(y2+y3)`、BF16 舍入点、
  16 个 K256 RMS 部分和的原顺序及 rsqrt 路径。
- 复用三个原函数的 AST 数学体，在一个成员里依次执行。首版保留原 GM 读写，内部加两处
  本核 fence；原阶段外的发布、等待及缓存维护保留。
- Sinkhorn 独立保留，改为等待融合阶段。工作区和根参数不变。
- 数学 task 从 66 个减至 64 个；发布从 722 减至 704，但重排后的等待从 12183 变为
  12189，不能声称所有控制项都减少。离线融合成本使用原三段成本之和，仅为估计。

CPU 证据：`hc_front_fusion_prepare_v1/`、`hc_front_fusion_cpu_v1/`。
真机证据：`b16_row_deps_hc_front_fusion_v1/`。

## HC 内部 Tile 直传及编译失败记录

实现：[hc_front_forward_tiles.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_front_forward_tiles.py)，额外开关 `--hc-front-tiles`。

T96 的每个成员只有一次 8 行 split，因此展开该单次循环，将 reduce 的 8×32 Tile
先转置为 32×8，提取行 0～7 和 4～11，再转置回 pre/post 的 8×8 Tile。
这样避免非对齐的列 4 UB 视图。mix 直接使用 split 产生的 pre Tile。

保留所有 GM 写出、原数学、编译器自动事件及阶段外同步，删除内部三次 GM 读取和两处
显式 fence。PTO IR 数学区域静态 TLOAD 从 22 处降至 19 处，`pto.fence` 从 4 处降至 0 处。
这是静态指令数，不能当作执行次数、时间或最终性能收益。

CPU 编译记录全部保留：

1. `hc_front_tiles_cpu_v1`：显式 transpose scratch 的形状错误，编译失败。
2. `hc_front_tiles_cpu_v2`：修正 scratch 后，slice 的静态 valid 与 scratch 的动态 valid
   在 PTO IR 中不一致，编译失败。
3. `hc_front_tiles_cpu_v3`：改用标准三参 transpose，让编译器生成 scratch，仍遇到该
   valid 类型不一致：`!pto.tile_buf<vec,8x8xf32>` 与 `valid=?x?`。
4. `hc_front_tiles_cpu_v4`：将对齐行 slice 改为 extract，使用有实体输出的 UB 提取，
   CPU/PTOAS 通过，随后真机完整通过。没有改 PyPTO、PTOAS 或 PTO-ISA。

实际脚本、生成源码和错误分别位于对应 `prepare_v*/` 与 `cpu_v*/`。
IR 检查见两个成功 CPU 目录中的 `hc_front_codegen_audit.json` 和
`hc_front_codegen_phase.pto.txt`。真机结果在 `b16_row_deps_hc_front_tiles_v1/`。

## Indexer 输入行依赖

实现：[indexer_input_dependencies.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/indexer_input_dependencies.py)，
[row_dependencies.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/row_dependencies.py) 的可选扩展，开关 `--indexer-input-dependencies`。
保持现选版的数学体、物理分核、核内先后、工作区、通知实现和缓存维护，只放宽已证明的三条整任务边。

- quant 成员 `w` 写 query `w`、`w+48` 的全部 64 个头，每 query 的 scale 为独占 256 B。
- 系数成员 `r` 只读该请求的六个 query，写独占的 12288 B 系数块。
- T96 的长 Indexer 按 `item=leaf*16+request` 分到 24 组。覆盖 1～32 叶后，每组只读
  两个请求，需要六个 quant 成员和两个系数成员；原来等待全部 48/16 个成员。
- 同一 MIX 组的三成员保守地使用相同输入要求。原 QK 复用 FFTS 的 216 条组内资源边保留。
- scale 的 `[24576,1]` 与 `[384,64]` 视图按元素地址检查等价，不只检查 allocation 名称。

CPU 检查 32 叶数×24 组共 768 个覆盖情形；原未修改边、同核先后和进度表转换的祖先关系
全部检查通过。等待从 12183 减到 10674，但发布从 722 增到 729。等待项减少不能直接
推导性能提升，本轮实测尚无明确收益，保持可选。

证据：`indexer_input_deps_prepare_v1/`、`indexer_input_deps_cpu_v1/`、
`b16_row_deps_indexer_inputs_v1/`。当前仅验证 B16/S6 长策略，不能套用其他档位。

## 未上卡的 scale 页 owner 候选

[indexer_scale_page_owner.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/indexer_scale_page_owner.py) 的八成员版本按物理页分配写者，
同一页的 compact 行仍由同一个成员按原序提交，避免 64 B 整行回写互相覆盖。
`indexer_scale_owners_prepare_v1/` 和 `indexer_scale_owners_cpu_v1/` 已通过 CPU 检查和编译。

历史探针中 scale 约 204 μs 完成，Indexer 还要等待约 235 μs 的 query 系数；离线模型也
没有缩短完整时长。因此暂不占卡测该候选。见
[indexer_scale_owner_priority_analysis.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/indexer_scale_owner_priority_analysis.json)。

## 复现参数及下一步

沿用 `run_device_compare.sh 16 131072 新结果目录`，共同参数为：

```text
--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile <本目录>/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies
```

再加表中对应候选开关。通过 `task-submit --device auto` 提交，不裸跑占卡，不覆盖旧结果。
新候选均未扩测其他六档，因为尚未显示值得推广的收益。

现选行依赖版阶段探针 `b16_row_deps_stage_clock_v1` 已完整通过。Indexer AIC 的数学及核内等待段
中位约 211.03 μs，QK/PV AIV 约 158.02 μs；后投影还存在同核三个 O-A、两个 O-B 的串行链。
按最晚显式前驱回溯，数学/内部同步/发布段累计 724.36 μs，外部就绪至开工间隙 71.88 μs。
这不是硬件理论下限，不能把所有间隙都称为软同步成本，更不能用插桩 span 796.24 μs
替代正式 L1 成绩。证据见 `row_deps_stage_summary.json` 和 `row_deps_stage_critical_chain.json`。
此前各档结果和选择不变。

<!-- SOURCE_END history-document-051 -->

---

<a id="history-document-052"></a>
### 历史原文 052：HC_MIX_CHUNK_BUFFER_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_CHUNK_BUFFER_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_CHUNK_BUFFER_EXPERIMENT.md)；原文 3,039 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-052 -->
# HC mix 分块 UB 缓冲：2026-10-17

候选完整通过，但没有显示出优于现选策略的明确收益，未采用、未扩测其他六档。
128K/B16/S6 的完整无插桩 L1 中位为 **814.74 μs**，同轮原动态 PTO 为 907.88 μs。
完整三窗、Event 时间和校验摘要见 [hc_mix_chunk_buffer_result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_mix_chunk_buffer_result.json)。
现选行依赖版的首轮结果为 807.16 μs；没有同进程直接 A/B，不能将跨运行差值当作精确回退幅度。

实现为 [hc_mix_chunk_buffer.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_mix_chunk_buffer.py)，实验开关 `--hc-mix-chunk-buffer`。
它保持原来的十二个成员、每成员八行、两个 16×K256 pipeline，以及所有数学、BF16 舍入、
RMS 累加顺序和 norm 权重类型。只在混合阶段保存每个 BF16 块，供随后的 norm 从 UB 复用。
原 mixed GM 写出保留；删除每轮共 786432 B 的 mixed GM 回读。

与此前 `hc_mix_resident.py` 的区别：此前版本改为 48 成员逐整行处理，并用 Scalar 读取各部分和。
本版本保持原八行 Vector 数学体，没有增加 Scalar 归约。也没有叠加 HC 三阶段融合。

## 布局及编译过程

首版 `hc_mix_chunk_buffer_cpu_v1` 将 8×256 块插入 8×4096 宽 UB Tile，PTOAS 拒绝生成的
`pto.tmov`：`expects A2/A3 non-mat tmov to use matching src/dst shapes`。未上卡。

第二版按块优先存放为 128×256 BF16，每个 K256 块占连续八行：

```text
UB[k*8 + row, col] = mixed[row, k*256 + col]
norm 第 k 块读取 UB[k*8 : k*8+8, :]
```

`hc_mix_chunk_buffer_layout_proof.json` 验证 32768 个元素完整覆盖、唯一写入。
`hc_mix_chunk_buffer_prepare_v2/` 与 `hc_mix_chunk_buffer_cpu_v2/` 完整编译通过。
生成 IR 中仅一份 65536 B 驻留 Tile，地址 98336～163872，未越过 188416 B 可用 UB。
原 pipeline 的分块插入与 norm 提取均落入生成代码，证据在 CPU 目录的 `codegen_audit.json`。
没有修改 PyPTO、PTOAS、PTO-ISA 或硬件容量配置。

接入正式实验开关后，`hc_mix_chunk_buffer_prepare_v3` 的完整 entry AST 与已编译 v2 相同，
没有重复执行一次相同的 CPU 完整编译。

## 真机结果和范围

任务 `task_20261017_063343_160455814507` exit=0，原始结果在
`b16_row_deps_hc_mix_chunk_buffer_v1/result.json`。

- 正式 W8A8 权重，原动态 PTO 为参照，不执行 Native forward。
- 输出、Top-K、六份 cache/state 逐 bit 检查通过。
- 图重放、30 代信号和输入/工作区保护区通过。
- 固定 fixture 每次恢复，不代表连续 decode 状态轨迹或整网验证。
- 完整 L1 含等待与阶段间隙，三窗；无阶段探针。没有以局部数学耗时替代目标。

复现仍通过 `task-submit --device auto` 调用 `run_device_compare.sh 16 131072 新目录`，
参数为现选行依赖版的公共参数，再加 `--hc-mix-chunk-buffer`。完整任务命令可由队列记录核对，
实际驱动已保存在结果目录 `benchmark_driver.py`。不覆盖旧结果，不裸跑占卡。

<!-- SOURCE_END history-document-052 -->

---

<a id="history-document-053"></a>
### 历史原文 053：HC_MIX_QPROJ_EXPERIMENTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_QPROJ_EXPERIMENTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_MIX_QPROJ_EXPERIMENTS.md)；原文 5,000 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-053 -->
# HC 整行驻留与主 Q 预取候选：2026-10-17

本轮继续按档编译；当前两项只限 B16/S6/T96。对照是原动态 PTO，正式 Layer4 权重，
完整 CSA 的 L1 incore 首尾为性能口径。没有执行 Native，没有修改生产或依赖仓。

## HC mix/norm 整行驻留

`hc_mix_resident.py` 将 mix/norm 从 12 个 AIV、每个 8 行改为 48 个 AIV、每个 2 行。
一行的四路输入和 BF16 mixed 留在 UB，保留原 mixed GM 写出，但删除后续 GM 回读
786432 字节。门控相加顺序为 (y0+y1)+(y2+y3)，norm 权重仍是 BF16 入参。

必须纠正早期实现中的归约契约错误：冻结的 `mix_x_rms_norm` 实际是
`pl.pipeline(16)`，每次 256 列，即 **16 个 K256 部分和**依次 Vector 累加。
初版错误地使用 8 个 K512 部分和，改变了归约顺序。旧 audit 没有检查这一点，
不能作为数学等价的证据。相关源码和失败结果保留，不改写历史结果。

| 版本 | 结果 |
| --- | --- |
| prepare/cpu v1 | runtime Scalar 不能作为 tile.full 的 value，CPU 失败 |
| prepare/cpu v2 | INDEX 到 FP32 的 scalar cast 不支持，CPU 失败 |
| prepare/cpu v3 | CPU 通过；错误的 8×512 RMS，Scalar 累加 |
| prepare/cpu v4 | CPU 通过；8×512 不变，仅恢复 Vector 累加 |
| prepare v5 | 恢复 16×256 和 0..15 Vector 累加，并强化冻结源码契约检查；验证中 |

两个旧候选都在精度门槛处停止，没有采性能：

- `task_20261017_053012_51687711509` / `b16_row_deps_hc_mix_resident_v1`：exit1。
- `task_20261017_053800_64833816891` / `b16_row_deps_hc_mix_resident_v2`：exit1。
- 两者相同：输出 15613 处不同、max_abs=0.015625；Top-K 825 处不同。
  SWA、compressed、state 和 indexer_state 也有差异；Indexer INT8 cache 和 scale 相同。
- Scalar 改 Vector 后结果未变，因此不能归因于 Scalar 加法舍入。
  16×256 修正尚待设备证明，不预先宣称它已解决全部差异。

原 CPU 审计还保留一次脚本错误：最初对冻结 body 而非已选 QK 地址缓存 body 做比对，
错误地把 QK 改动也计入本候选；修正参考后仅 mix 函数变化。

## 主 Q 的普通 N128：未上卡

`qproj_n128.py` 保持 20 个 worker、K256 的四段 INT8 累加，将 N256 改为 N128。
prepare/cpu v1 完整生成通过，但实际四个 Right Tile 全部地址 0；
`qproj_n128_codegen_audit.json` 明确 `two_right_buffers_present=false`。
没有实现预期双缓冲，因此没有为此候选提交 NPU 测量。

## 主 Q 的显式 L0 预取

`qproj_l0_prefetch.py` 保持 N256、20 worker、128 列块，将 K 改为 128，
先加载下一块 Left/Right 再提交当前块 matmul，制造真实的重叠活跃期。
QR 的 96×1024 INT8 仅在 worker 开始时装入 L1。每份 Right 为 32KiB，Left 为 12KiB；
两份分别占 64KiB、24KiB，Acc 为 96KiB。INT8 1024 项点积的最大绝对和
16777216，不会 INT32 溢出；仍需完整输出逐 bit 验证。

prepare v1 已通过；cpu v1 进行中。源级预取不等于实际双缓冲或硬件重叠，
必须核对生成 C++ 的 Left/Right 地址与提交顺序后才决定是否上卡。


05:44补充：HC prepare/cpu v5完整通过，IR确认16×256，代码生成保留V到S屏障；
任务task_20261017_054339_75460017083已提交，设备结果待收取。
主Q CPU v1完整通过，两处Left地址0/12288、Right地址0/32768交替，八段提交顺序通过；
任务task_20261017_054303_7422476042已提交。无设备结果前不预记精度或性能。

## 最终结果：两个新任务均 exit0

| 候选 | 单SPMD三个完整L1窗口/μs | 中位/μs | 同轮原动态中位/μs | 采用 |
| --- | --- | --- | --- | --- |
| HC 16×256整行驻留 | 822.46 / 819.02 / 817.06 | 819.02 | 911.56 | 否 |
| 主Q显式L0双缓冲 | 825.54 / 831.16 / 830.62 | 830.62 | 892.68 | 否 |

两者正式权重固定fixture的输出、TopK、六份cache/state、图重放、30代进度和保护区全部PASS。
HC恢复原16×256归约后消除了前两轮差异；没有修改阈值，也没有把精度失败结果当性能测量。
实际NPU产物确认HC的归约指令与CPU相同；主Q两处Left/Right双缓冲地址及提交顺序再次通过，
见各任务目录中的codegen_audit文件。不使用hash。

没有显示比现选行依赖方案更有利的结果，因此不采用、不追加跨档或动态边界测试。
与现选807.16（首轮）/817.88（同进程复核候选）的比较来自不同运行，
不能量化为严格的退步幅度。七档选择不变，700μs目标未达。

## 避免重复的只读核对

重新审视Indexer M192/N128、WS K192、六query拆成两组三query的设想时，确认
indexer_n128.py已经完整实现并测过：此前883μs，不再重做。
主Q普通N128未形成双缓冲与本次显式K128双缓冲是不同候选，两者都保留证据。
当前HCpost已经按一整行4096列计算，不能误称它还在用16次N256分块。

<!-- SOURCE_END history-document-053 -->

---

<a id="history-document-054"></a>
### 历史原文 054：HC_PRE_REFERENCE_REVIEW.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/HC_PRE_REFERENCE_REVIEW.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_PRE_REFERENCE_REVIEW.md)；原文 3,413 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-054 -->
# 用户指定 HC_pre 优化与当前 CSA 单 SPMD 的对应关系

2026-10-17，用户指定参考 canonical 仓 `vllm_ascend/ops/pypto/hc_pre.py`。
本轮已读取实际文件，并保存 `hc_pre_reference_20261017.py` 文本快照；不修改原文件、不做hash。
独立worktree中的同名文件不能默认等同于用户指定文件。

## 有直接参考价值的部分

1. 第179～299行：K分段内流式供数，AIV加宽后的同一份FP32输入用于投影供数与平方和统计，
   用 aic_gather 交给AIC，减少完整FP32输入写回/读取GM。当前hc_stream_pre候选仍在GM的view_2中
   store/load，这与独立HC_pre的组内供数不是同一实现。
2. 第230～246行：显式交错L0双缓冲，先准备块0/1，算0后准备2，再算1，保留同一FP32 Acc及K顺序。
   这是把下一次搬运提前并设计实际活跃区间，不能仅以pl.pipeline(stage=2)名义推断重叠。
   可作为CSA投影/Indexer局部流水参考；不同L0C容量和现有事件要逐项核对。
3. 第140～153、370～381行：按阶段重分工，全体48 AIV分别承担Sinkhorn与mix，token少时再切D；
   投影的token×K分片与尾段token/D分片彼此独立。这是当前融合大SPMD重排阶段应借鉴的工作量设计。
4. 第503～585行小档Sinkhorn标量迭代、第667～710行设备标量门值与连续D区间/整行mix，减少小tile
   gather/broadcast/转置及过多碎片。不能把这些算法直接套到所有attention数学；要核对规模和计算顺序。
5. 第775～800行按T区间选择M/K/供数宽度/尾段行数，依据核波次和存储容量，保留真实尾块。
   CSA还要考虑B、历史长度和query分组，不能照搬HC_pre的阈值或仅按T统一分档。

## 之前的 HC 候选不等价于完整吸收

`hc_front_fusion.py`只是T96的reduce→split→mix三阶段结合，原数学及中间GM保留，Sinkhorn独立。
`hc_stream_pre.py`保留原4份K4096投影部分和、原32项RMS顺序，走GM流式交接。
因此这些候选未优于现选，不证明独立HC_pre整套策略在完整CSA中无价值，也不能声称已全部移植。

## 实际移植的边界

- HC_pre没有input_layernorm。当前CSA核体也先把混合结果round到BF16，再转FP32统计RMS并执行norm
  （冻结core_bodies_ir.py第357～381行）；串接BF16 mixed本身不会新增这处舍入。
  真正精度风险包括投影部分和、RMS分段归约、Sinkhorn加法/除法顺序的改变。
- 独立实现第364行是全参与者硬件syncall。把这一屏障嵌入完整CSA前必须确保所有参与核都到达且
  不破坏当前点对点进度图；不能把局部阶段等待随意改成整算子全局会合。
- standalone变快不能代替完整CSA L1收益。当前所有七档仍需各自达到同轮动态PTO的0.70。

## 本轮 QK 负结果与参考方向的关系

`qk_row_load_pipeline_v1/lowering_audit.json`证明stage=2没有把两行DMA提前：
TLOAD→MTE2/V wait→TMOV→第二行TLOAD→wait→TMOV，两行还共用一个UB地址。
更关键的是现选索引缓存基线原本可以直接对目标行连续提交TLOAD，没有候选引入的逐行assemble等待。
所以关闭当前候选，不上卡、不扫stage；不能把错误的基线瓶颈假设带到下一轮。
HC_pre给出的方向是先核对真实供数与消费关系，再改变实际数据复用和分工。

<!-- SOURCE_END history-document-054 -->

---

<a id="history-document-055"></a>
### 历史原文 055：HC_ROW_SPLIT_REVIEW.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/HC_ROW_SPLIT_REVIEW.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_ROW_SPLIT_REVIEW.md)；原文 2,419 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-055 -->
# 2026-10-17：HC前端行分工审阅与后续结果

## 后续验证（01:23）

已在实验目录实现`hc_row_partition.py`，2/4行可选，默认仍8行。
`hc_rows2_prepare_v2`及`hc_rows2_cpu_v1`通过；
`task_20261017_011729_35529046475`的B16 2行分片完整PASS，
无插桩完整L1中位843.20μs，慢于现选815.84μs，未采用。4行未做真机测量。
详细边界与结果见`ATTEMPTS.md`、`hc_rows2_result.json`。

## 实现之前的审阅记录

现选B16的已有阶段探针（`b16_cached_qk_indices_stage_v1`）中：

- `hc_widen_rms`：12个逻辑Vector成员，数学+维护+发布中位13.48μs。
- `mix_x_rms_norm`：12个逻辑Vector成员，中位15.29μs。
- 这是插桩阶段数据，不能直接相加当成可获得的整体收益。

两者按8行一块划分T96。可评估每块处理2行，从12分片变为48分片；
也可逐档选择4/8行。每行的K分块和累加顺序保持，不能为增加核数改变RMS
舍入。每个物理Tile可仍为8行、使用2个有效行，保留FP32列主序规约结果
32B对齐；实际收益需测，因为K方向循环次数仍不减少。

不能只把8改成2，已核对的具体风险：

1. `hc_widen_rms`原来仅最后一块走`widen_tail`。改成2行后如果仍判断
   `w_rows == 8`，所有块都会写同一份tail工作区，形成竞争。新满块需直接
   按有效2行写回；不使用共享tail。
2. 最后`w_inv`物理形状8×1，必须明确有效2×1，不能覆盖邻块的inv_rms。
   这是MTE3有效形状问题，不可以套用Scalar缓存发布替代正确写入范围。
3. `mix_x_rms_norm`的pre_val输入和第二遍mixed_input原先加载8行。
   新块起点94时会超出T96，需要显式有效2行。输出也必须只写当前2行。
4. mix_x的`x_normed_tail_store`同样是唯一尾块工作区，不能让48个满块共写。
5. 必须更新逻辑task block_num/member_count与widen_rows实参，再重建完整
   分核和等待表。原12成员表不能复用为48成员完成条件。
6. 当前T=B×6总是偶数，2行划分无残余尾行；仍需检查每个受支持T的实际
   分片覆盖、GM范围和有效形状传播，不用这个性质推断未来S/布局。

所有结论来自冻结body bundle和已完成探针的只读审阅。本文件没有声称
该候选已编译、已真机通过或能改善性能，也没有更改冻结源码/生产入口。

<!-- SOURCE_END history-document-055 -->

---

<a id="history-document-056"></a>
### 历史原文 056：HC_STREAM_PRE_SYNC_AUDIT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/HC_STREAM_PRE_SYNC_AUDIT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/HC_STREAM_PRE_SYNC_AUDIT.md)；原文 3,751 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-056 -->
# HC 前段流式供数候选：同步与数值合同

本候选在 `hc_stream_pre.py` 中，只改变实验生成的 CSA 单 SPMD 的 HC 前段。
目前仅验证 T96（B16/S6），不是各档统一策略，也没有替换生产入口。

## 数值保持方式

- 保留 M16、K4096、4 份投影部分和。每份仍依次执行 K=0、256、…、3840 的16次矩阵乘。
- AIV 每次对8行、K512的 BF16 输入做无损 FP32 转换，与原加宽的数学 Tile 相同。
- RMS 的每个 K512 `row_sum` 单独写出。归约阶段从0初始化，按原32个K块顺序逐项相加，继续使用原缩放、epsilon、rsqrt。
- 不将32项改为四个局部和，不替换 standalone HC_pre 的另一套分片/归约策略。

## 核分工与工作区

24个AIC组各处理一个原M16/K4096分片，每组两颗AIV各处理8行。
全部48颗AIV供数。仍写入原 FP32 加宽区；后续HC混合阶段会读取它，不能删除。
32份RMS小结果占12288字节，使用T96原本无尾行时闲置的 `temporary_5`。
没有新外部参数、GM分配或 Scalar 信号。

原 `hc_pre_linear_reduce` 的6个AIV成员除了归约四份投影，还计算对应16行的RMS。
检查了两个inv_rms消费者均等待所有归约成员，因此原加宽完成标记不会被误当成新RMS已完成。

## 组内交接

事件8/9为两个READY槽，10/11为对应ACK槽。每个组独立，两个AIV分别发布，AIC等待成组就绪。

1. AIV在复用某个事件前消费上一轮ACK，加载BF16并转换。
2. AIV写原FP32加宽区、写RMS分片，再在MTE3发布READY。
3. AIC先等待READY，再在MTE2加载该K512；随后在MTE2发布ACK，继续两次K256矩阵乘。
4. AIV末尾消费剩余两个ACK，不能带着未消费事件进入后面的Indexer/QK。

原AIC整块GM等待由这一组内协议替代；后续阶段GM进度、代际、DCCI/DSB及数据维护仍保留。
128个随机交错×连续3代的CPU事件模型检查无覆盖、无死锁、无残留。模型假定现有FFTS API语义成立，
不能证明真实硬件pipe或缓存可见性；完整真机exact/图/保护区检查仍为必需。

## 编译中遇到的问题

1. CPU_v1：直接将store结果重新赋给根参数，触发MIX整参别名检查。改用原地store表达式，保持原参数返回。
2. CPU_v2：pipeline内动态event_id在PTOAS生成C++时报operand不在scope。候选改为偶/奇静态ID分支。
3. CPU_v3：静态ID后编译通过，但IR和C++均显示自动pipeline把输入TLOAD移到READY等待之前。这版未上卡。
4. CPU_v4：两个有手工FFTS的循环改用pl.range；仅纯数学RMS累加保留pipeline。CPU编译与指令顺序审计通过。

证据：`hc_stream_pipeline_order_audit.json`、`hc_stream_prepare_v4/codegen_audit.json`，以及各CPU目录的日志/源快照。
这次没有修改PyPTO、PTOAS或Simpler，不把作用域错误和pipeline顺序问题混为同一原因。
当前规避方式不支持把任意手工同步循环直接改成自动pipeline。若以后需要恢复该优化，应先补最小复现并修正编译器副作用依赖。

## 真机验证

`task_20261017_032959_27383327818` / `b16_cached_hc_stream_v1` 已完成exit0，固定fixture完整exact、图重放、30代进度和全部保护区通过。
对照只执行原动态PTO；正式Layer4 W8A8权重、固定fixture、原exact阈值；完整CSA L1 incore首尾3窗。
不得用该候选的局部阶段时间代替整层性能。结果以该目录result.json为准。

三窗L1：834.18/844.40/847.60μs，中位844.40；同轮原动态907.64。
未优于保留815.84μs，未采用，不扩其他档。不依据单次整层差值断言具体哪个新增阶段造成回退。
汇总见 `hc_stream_result.json`。

<!-- SOURCE_END history-document-056 -->

---

<a id="history-document-057"></a>
### 历史原文 057：INDEXER_KEY_PAIRS_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_KEY_PAIRS_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_KEY_PAIRS_EXPERIMENT.md)；原文 3,707 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-057 -->
# N64循环版两面板Key预取（2026-10-17）

完整128K/B16/S6验证通过，L1中位813.42μs，没有优于现选约807μs的证据。
不采用、不扩其他档、不重复本配置。原七档选择不变，700μs目标仍未达到。

## 变化及边界

`indexer_l1_key_pairs.py`保留已通过的N64 score双槽循环，在每次循环中计算两个面板。
完整顺序为首块0、循环(1,2)…(13,14)、末块15、排空WS15。每一对在第一个QK前
提交第二块Key的GM→L1，第二块Key的Right提取仍在第一个QK/WS之后。

只有score双槽跨循环携带，Key/Right都局部创建和消费，避免之前Key分支phi跨回边的
别名问题。Key池物理地址复用受MTE1→MTE2完成事件保护；它不是两个Key同时保存在同一区域。

保持QK M384/N64/K128 INT8、WS M16/N64/K384 FP16，FIX 2^-10与ReLU转换不变。
保持每个request的原leaf集合、动态长度/位置/页表读取、1024分数交接、双槽FFTS、
AIV Top-K、归并和最终排空。没有新增Host读取、Native forward或依赖仓改动。

## CPU与生成代码

`prepare_indexer_key_pairs.py`、`indexer_key_pairs_prepare_v1`、`indexer_key_pairs_cpu_v1`
完整codegen通过。集成`build_variant(indexer_key_pairs=True)`后生成入口与已编译入口
文本完全相同，没有重复完整CPU编译。设备开关为`--indexer-key-pairs`，默认关闭。

`indexer_l1_ring_codegen_audit.py --key-pairs`核对实际L1、Right、Acc分配及FIX转换：

- score为98304B双槽，地址65536；Key池地址163840，全部分支/循环原位。
- 三个QK(current)/WS(previous)静态对的Right与Acc均分开，没有不支持的同层搬运。
- 静态QK/WS各4处，FIX与GM写各4处；双面板循环内4次matmul、2次FIX、2次GM写，无PIPE_ALL。
- 以循环C++行号计：第二Key静态加载位点269/302/356/389；391发出MTE2完成事件。
  第一QK415、第一WS427；484才等待第二Key，第二QK在513。搬运提交提前且完成等待推迟。

这证明源编排在C++中保留了对应提交顺序，不等于已经证明设备重叠时长或性能收益。
证据见`indexer_key_pairs_codegen_v1.json`及`.cpp.txt`/`.loop.cpp.txt`。
扩展审计后，已有单面板ring设备产物仍通过默认审计路径。

## 设备结果

任务`task_20261017_111540_229620229969`，exit0。
输出目录`b16_row_deps_indexer_key_pairs_v1`，机器摘要`indexer_key_pairs_result_v1.json`。
实际设备产物执行前通过同样地址/提交顺序检查，并确认AIC/AIV各74处自定义等待，
DCCI后读取及100tick退避真正写入CCEC输入源码。

| 实现 | 三个完整L1窗口 μs | 中位数 μs |
|---|---|---:|
| 两面板Key预取单SPMD | 813.42 / 814.14 / 811.40 | 813.42 |
| 同轮原动态PTO | 907.18 / 898.80 / 892.24 | 898.80 |

Event为926.35/950.66μs，仅作附录，不参与700μs验收。
完整eager/graph/final的输出、Top-K与六份cache/state零差异，30代通知和全部保护区通过。
每次调用恢复fixture，不代表连续decode轨迹。与此前现选没有同进程配对，
不把跨轮小差值定量归因于预取；本轮结果不支持更换现选。

命令使用`run_device_compare.sh 16 131072 <新目录>`，追加：

```text
--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile <本目录>/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies
--indexer-key-pairs
```

必须通过`task-submit --device auto`运行。此前三项等待配置错误及更正结果见
[编译上下文更正](../../../tests/pypto_test/csa_multishape_spmd_20261016/COMPILE_CONTEXT_CORRECTION.md)，不要把旧约1000μs结果用作本候选的有效比较基线。

<!-- SOURCE_END history-document-057 -->

---

<a id="history-document-058"></a>
### 历史原文 058：INDEXER_KEY_POOL_CPU_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_KEY_POOL_CPU_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_KEY_POOL_CPU_EXPERIMENT.md)；原文 1,007 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-058 -->
# Indexer Key L1行池扩大：CPU通过，未上卡

2026-10-17，`indexer_key_pool.py`只变换现选B16生成源码中的Indexer AIC循环。
Key行池从128×128 INT8扩大至1024×128，16KiB→128KiB，每个Native Key页
占独立32行；16个N64 Right提取使用0,64,...,960。页地址、原提交顺序、
QK/WS、FP16 FIX/ReLU、store和FFTS保持；没有增加预取深度。

`indexer_key_pool_prepare_v1`和`indexer_key_pool_cpu_v1`通过。
用done_16标记和完整C++块边界抽取stage16，两版均32个matmul，
MTE1→MTE2 wait由19处变20处，MTE2→MTE1 wait均18处，PIPE_ALL均1处。
实际没有减少目标等待，因此不提交设备、未接driver、不扩档。
静态指令计数不等于运行时停顿；不声称精度或性能已经测过。
证据`indexer_key_pool_codegen_audit.json`与两份完整阶段C++片段。
初次按v30前缀抽取只覆盖后半段，已改用完整块并断言32个matmul；
不使用此前15个matmul的不完整计数作结论。

<!-- SOURCE_END history-document-058 -->

---

<a id="history-document-059"></a>
### 历史原文 059：INDEXER_L1_RING_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_L1_RING_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_L1_RING_EXPERIMENT.md)；原文 6,179 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-059 -->
# Indexer N64 L1 循环流水实验（2026-10-17）

> 2026-10-17更正：本次设备产物提前JIT编译，未应用请求的cached_wait/100tick。精度结果保留，不能据此判断数学核改动的性能回退；见[编译上下文修正](../../../tests/pypto_test/csa_multishape_spmd_20261016/COMPILE_CONTEXT_CORRECTION.md)。


目标：在原单 SPMD 的 Indexer AIC 内减少长展开体，保持原 M384/N64/K128
INT8 QK、FP16 WS K384、FIX 量化 2^-10 与 ReLU，以及原叶计划、1024 分数交接、
AIV Top-K 和任务依赖。只有 `indexer_score_topk_native_pair_aic` 的面板计算体改变。
以下版本都没有改生产代码、冻结的静态目录或依赖仓库。

## 尝试和边界

| 版本 | 写法 | CPU 结果 | 是否允许上卡 |
|---|---|---|---|
| v1 | Key 三槽、score 双槽；循环最后一轮跳过预取 | 同地址 188416 的 Mat→Mat 移动无法生成 | 否 |
| v2 | 剥离最后一轮，循环 1..14 总是预取下一 Key | 完整 codegen 通过，但物理地址错误 | 否 |
| v3 | 两个 Native 页在同一个分支叶子中加载 | PTOAS 拒绝不支持的 Mat→Mat 地址空间组合 | 否 |
| v4 | Key 每轮局部创建，只让 score 双槽跨循环 | 完整编译与物理审计通过 | 完整精度通过，性能变慢，不采用 |

v1～v3 都不是设备精度失败，也没有性能数据；没有把编译通过当作可运行依据。
旧源码、准备产物、passes_dump 和错误均保留。独立的小 FP16 原位循环探针成功，
不能证明包含 Native 页分支与循环携带值的完整 CSA 也正确。

## v2 的实际错误

`indexer_l1_ring_cpu_v2/kernel/passes_dump/38_after_AllocateMemoryAddr.py`
和 `kernel/ptoas/csa_static_group.pto` 已显示不一致：

- 循环携带的 Key 池以及后续 Key 提取使用 Mat 地址 **188416**。
- 第一个 Native 页加载后，中间 `stage_16_v71__phi_v17` 被分到 Mat 地址 **0**。
- 第二个 Native 页的 `pto.subview` / `pto.tload` 实际写入这个地址 0 的池。
- 最终合流值重新指向 188416；中间没有把第二页搬到该地址的有效操作。

因此，两页中的后半页不会按源程序含义进入下一次 QK 的输入池。
这是物理数据流错误，不是性能波动。`indexer_l1_ring_v2_misaligned_pool_evidence.pto.txt`
保留了关键分配与使用点；`indexer_l1_ring_v2_rejected_audit.json` 记录拒绝结果。
原审计脚本在 `source_snapshots/indexer_l1_ring_codegen_audit_v2.py`。

只读检查当前隔离 PyPTO 的 `src/ir/op/tile_ops/paged_gather.cpp` 可见，
`tile.gather_row` 已声明 `ReadWrite` 与 `output_reuses_input(0)`。
错误在 PTOAS 前的 PyPTO 分配产物已经出现；本次尚未进一步确定到具体的
MemoryReuse / 分支合流修复代码行，不能宣称已修复编译器。

## v4 的取舍

首块生成 QK0 并将其 FIX 到 score 槽 0；循环 p=1..15 中加载本轮 Key，
执行 QK(p) 与 WS(p-1)，将 QK(p) 写入槽 p%2；循环结束后排空 WS15。
Key 和 Right 均不携带到下一迭代，只有 score 池采用 DPS 原位循环更新。

源模型检查确认 32 个 Native 页、16 个 QK/WS 面板和 1024 列输出各出现一次。
记录在 `indexer_l1_ring_slot_contract_v4.json`。这仍不是设备正确性的证明。
取消 Key 提前预取可能带来代价，必须由完整无插桩 L1 首尾决定是否采用。

完整编译命令使用冻结工具 `../csa_static_spmd_feasibility_20261015/compile_jit_entry.py`，
输入为各版 `indexer_l1_ring_prepare_v*/variants/h131072_b16/entry_jit.py`，
参数 `--memory-planner PYPTO --kernel-mode`，输出保存在对应 `indexer_l1_ring_cpu_v*`。
物理审计还要核对 Mat/Right 地址、Acc 不重叠、FIX 量化与 ReLU，以及循环指令。

当前七档策略不因这些候选而改变；128K/B16/S6 的 700 μs 目标尚未达到。

## v4 物理审计与设备任务

Key 两个局部池均分配在 Mat 163840，分支合流地址一致；score 双槽分配在65536。
循环 QK/WS Acc 分别0/98304，Key/WS Right 分别49152/0。实际循环包含两个
TMATMUL、一个带2^-10/ReLU的TINSERT、一次TSTORE，没有Mat→Mat或Right→Right，
也没有PIPE_ALL。完整审计和循环C++见 `indexer_l1_ring_codegen_audit_v4*`。

已提交 `task_20261017_094749_7744056502`，结果目录 `b16_row_deps_indexer_l1_ring_v4`。
新增设备驱动开关 `--indexer-l1-ring`；驱动在执行前对真实artifact再次执行地址审计。
终态为 exit=0；完整精度与图重放通过，性能结果如下，现选策略不变。

## 缩减探针的范围

`probe_loop_gather_phi.py` 保留循环携带、两次条件gather与前后Key读取，CPU完整编译通过。
`probe_loop_gather_phi_cpu_v1/allocation_audit.json` 显示各池与视图均为地址0，
未复现完整CSA v2的不一致。完整CSA的外层分支与全图存储复用可能是触发条件，
目前仅为待定位范围，不能据此确定根因。缩减探针没有设备调用。

## v4 设备结果（不采用）

完整输出、Top-K、六份cache/state的eager/图/最终比较逐bit一致，30代信号、
未写区与workspace保护通过。正式第4层权重；每次恢复fixture，不代表连续decode轨迹。

| 指标 | 原动态 PTO | L1 score环 v4 |
|---|---:|---:|
| 完整L1三窗 μs | 905.34 / 908.00 / 899.98 | 984.56 / 1034.82 / 1008.86 |
| 完整L1中位 μs | 905.34 | 1008.86 |
| 图重放Event中位 μs | 951.86 | 1133.55 |

候选明显退步，不采用、不扩档、不重复测；现选历史807.16μs，最新配对测量807.50μs
属于此前不同任务，不能写成这轮同进程对照。结果见 `indexer_l1_ring_result_v4.json`。
缺少候选内部阶段插桩，不能把全部退步定量归因于Key预取取消；无需为已否决版本加测。

## 修正等待配置后的v5

`task_20261017_110544_21223701219`完成exit0，全门槛PASS，实际AIC/AIV 100tick等待门禁通过。
完整L1三窗804.14/813.50/806.38μs，中位806.38；同轮动态885.92。
与此前现选807.16/807.50接近，未证明稳定收益，不替换七档选择。
机器记录indexer_l1_ring_result_v5.json。旧v4精度结果保留，性能退步归因撤回。

<!-- SOURCE_END history-document-059 -->

---

<a id="history-document-060"></a>
### 历史原文 060：INDEXER_N32_ACC_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N32_ACC_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N32_ACC_EXPERIMENT.md)；原文 1,838 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-060 -->
# Indexer N32双QK累加区：精度通过，性能未采用

2026-10-17。原M384/N64/K128点积改32个M384/N32/K128面板，WS仍K384，
FIX的FP16量化/ReLU、原1024分数交接、Native页读取、叶分片和AIV Top-K保持。
不同于旧M192/N64按query拆分，这次只沿候选列拆分。

`indexer_n32_acc.py`在本测试目录生成可选数学体。`--indexer-n32-acc`默认关闭，
先限制128K/B16/S6。没有修改生产算子、冻结静态结果或依赖仓库。

prepare_v1、cpu_v1完整代码生成通过。`indexer_n32_codegen_audit.json`核对：

- 31对相邻QK Acc均在0/49152交替，两份各48KiB，不重叠。
- 下一QK在当前FIX之前提交，源顺序确实保留到C++；不代表实际无停顿。
- Query Left0、系数Left49152，总计60KiB。首两Key Right0/4096，
  后续Key24576与WS Right0分区。没有Right到Right搬运。
- 完整Indexer AIC阶段静态64matmul/32FIX/32store，数学调用数为旧版两倍。

任务`task_20261017_085633_40447024100`终态exit0。输出、Top-K、六份cache/state、
图重放、30代信号及所有保护区PASS。完整无插桩L1三窗985.94/976.20/981.12μs，
中位981.12μs；同轮原动态PTO901.76μs，候选慢8.80%。Event中位1089.69μs，
另列、没有混进L1口径。现选B16历史807.16μs，最近一次同进程现选808.92μs；
本任务没有再采现选，也无需用小差值判断：此候选退步明显。

结论：容量和提交顺序成立不等于收益。N32增加的矩阵/FIX/存储调用及流水变化
没有带来完整CSA加速；没有单独测到各项成本，不能把差值全归因于某一个硬件事件。
不采用、不扩档、不重复同一测试。现选七档不变，700μs目标仍未达到。
每次调用恢复fixture，不代表连续decode；没有运行Native forward。

<!-- SOURCE_END history-document-060 -->

---

<a id="history-document-061"></a>
### 历史原文 061：INDEXER_N80_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N80_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N80_EXPERIMENT.md)；原文 3,365 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-061 -->
# B16/S6 Indexer N80 候选：精度通过，性能不采用

2026-10-17。文件 `indexer_n80.py`，显式开关 `--indexer-n80`。
只在实验目录支持长历史 B16，与已选的 QK 地址缓存和行依赖细化组合。
没有修改生产入口、冻结数学源文件、PyPTO、Simpler 或 PTO-ISA。

## 改动与预期

原 Indexer 每 1024 个候选分为 16 个 N64 panel。本候选采用 13 个 N80 panel，
末 panel 只存 64 列，QK/WS 矩阵调用总数 32→26。
保持 QK M384/K128、六个 query、WS M16/K384、FIX 量化/ReLU 和原归约顺序。
叶划分、1024 列 GM 分数块、双缓冲、FFTS 发布/确认及排空、Top-K AIV 全部保留。

Native Key 页每 32 行附带 scale。80 列 panel 分成 32/32/16 或 16/32/32 行加载，
不能直接跨页把 scale 字节当作 Key。每 1024 列的加载段从 32 增至 39；
尾部多计算的 16 列不写入 GM，也不改变有效候选数。

N80 的 WS Right 占 60 KiB，Key Right 占 10 KiB，不能同时驻留 64 KiB L0B。
因此仅将下一个 Key 预取到 L1，当前 QK 消费 Key 后，才覆盖 Right 放入 WS 数据。
调用次数减少与预取重叠减少同时发生，不能只凭前者断言提速。

## CPU 与实际生成代码

`indexer_n80_prepare_v1/` 完成分档准备、依赖与协议检查。
`indexer_n80_cpu_v1/` 完成 CPU 全流程代码生成；这不代表设备执行或数值通过。

实际 AIC 代码中：

- 13 个 QK Acc 均在 0 地址，占 122880 B；前 12 个 WS Acc 位于 122880，占 5120 B。
- 最后一个 WS 在 QK FIX 已消费结果后复用 Acc 0 地址，峰值 128000 B，小于 131072 B。
- WS Right 与 Key Right 同在 0 地址，M→MTE1 的通知/等待保护复用。
- CPU 与设备实际生成代码的 52 个 N80 Acc/Right Tile 类型、形状、地址一致。

证据：`indexer_n80_codegen_audit.json`、`indexer_n80_codegen_fragment.cpp`、
`indexer_n80_device_codegen_comparison.json`。没有做 hash 校验。
初版地址提取脚本误合并不同函数的重名常量、误要求末 WS 使用 122880，已修正；
全文阶段提取又遇到被优化掉的 SSA 名称，改为直接读取实际 Tile 与 TASSIGN。
这些是审计脚本错误，不是设备算子失败；没有宣称完整 C++ 阶段逐字一致。

## 设备结果与结论

任务 `task_20261017_050914_1722359619`，结果 `b16_row_deps_indexer_n80_v1/result.json`，exit=0。
正式第 4 层 W8A8 权重，两侧均为 PTO，未执行 Native forward。
输出、Top-K、六份 state/cache、eager/graph/最终结果逐 bit 一致；
30 代信号、只读区、未写区及工作区保护区通过。

| 完整无插桩 L1 incore 首尾 | 三窗（μs） | 中位数（μs） |
| --- | --- | --- |
| 原动态 PTO | 893.40 / 895.08 / 919.26 | 895.08 |
| 行依赖 + N80 单 SPMD | 879.10 / 887.62 / 878.82 | 879.10 |

图外 Event 中位数为动态 956.57 μs、候选 981.13 μs，不混入 L1 目标。
N80 虽比同轮原动态 L1 略快，但明显慢于此前已选行依赖单 SPMD 的两轮 807.16/817.88 μs。
**不采用 N80，不扩测其他档位，不因其减少矩阵调用而放宽验收。**

生成代码证明预取重叠减少和页段增加，尚未单独测出两者各贡献多少微秒；
本轮没有为负候选追加阶段探针或动态长度测试。代码和失败尝试完整保留。

<!-- SOURCE_END history-document-061 -->

---

<a id="history-document-062"></a>
### 历史原文 062：INDEXER_N80_WS128_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N80_WS128_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_N80_WS128_EXPERIMENT.md)；原文 1,906 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-062 -->
# 2026-10-17：N80 WS K128分段与Key Right预取

旧N80候选完整精度通过但中位879.10μs，未采用。它的WS Right为
384×80 FP16=60KiB，无法同时容纳10KiB Key，限制了预取。

新候选`indexer_n80_ws128.py`保留QK M384/N80、13个panel、原叶分配、
FP16量化/ReLU、FFTS、AIV TopK、根参数及GM工作区。WS按0/128/256三段累加，
每个Right块降为20KiB，提前把下一Key搬到Right。此变换增加matmul指令数量，
也改变K的指令分段，不能仅凭理论容量与数学等价断言更快或逐bit相同。

开关`--indexer-n80-ws128`默认关闭，首轮只验证128K/B16/S6，未与其他新候选叠加。
prepare/cpu_v1完整通过。实际IR有13个Key Right和39个WS Right：
WS地址为0/20480，后续Key为40960/51200，峰值末端61440。
检查了12组“下一Key extract在当前QK matmul之前”，对应Key地址区间不重叠。
这是实际分配与提交顺序证据，还不是硬件重叠测量。审计在
`indexer_n80_ws128_codegen_audit.json`。

任务`task_20261017_071758_23254658630`已提交，结果目录
`b16_row_deps_indexer_n80_ws128_v1/`。先完整exact/TopK/六状态/图/30代/保护区，
通过后测三窗无插桩L1；同轮对照仅原动态PTO，不运行Native forward。
当前尚无设备结论，未改变现选七档。

## 终态：完整通过，性能不采用

任务exit0，输出/TopK/六状态、eager/图、30代信号与保护区均通过。
单SPMD三窗844.52/855.10/858.08μs，中位855.10；同轮原动态906.78μs。
Event单独见result，不混入L1计时。未做同进程旧N80/现选配对，不能把跨运行
879.10→855.10当作精确收益。该候选仍慢于现选，未采用、不扩七档。
这证明K分段本fixture通过exact；不代表任意输入均已验证，也不是连续decode轨迹。
证据`indexer_n80_ws128_result.json`及原任务目录。

<!-- SOURCE_END history-document-062 -->

---

<a id="history-document-063"></a>
### 历史原文 063：INDEXER_PAGE_BASE_SELECT_CPU.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PAGE_BASE_SELECT_CPU.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PAGE_BASE_SELECT_CPU.md)；原文 1,980 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-063 -->
# Indexer 页视图的 CPU 可行性检查（2026-10-17）

结论：本轮两个入口方向均不满足当前编译接口要求，没有提交设备任务，没有改依赖。
不能将这些结果理解为 CSA 性能或精度失败。

Native 每页4160字节：4096字节Key加64字节scale。当前AIC按物理页地址对128的余数，
选择原始和偏移64字节的两个GM视图。希望消除逐页奇偶分支，先核对以下接口边界。

1. 自定义三维stride：当前PyPTO JIT Tensor参数不接受自定义`TensorView(stride=...)`。
   本地`tests/ut/jit/test_jit_compile_extraction.py`的`TestUnsupportedLayoutSlot`明确检查此拒绝。
   `pl.view`也不提供任意stride/offset描述。已读源码，不再为已有明确限制重做设备探针。
2. 先选择两个已有GM输入之一、再统一`gather_row`：v1使用未读取block index的
   `with pl.spmd(1)`被前端拒绝，是探针写法问题。v2改为`for worker in pl.spmd(1)`并
   在control地址中使用worker；随后明确拒绝不同backing SSA的if返回：
   `IfStmt in-place return_var 'source__rv_v0' yields different backing SSAs across branches`，
   两个backing分别为`even__ssa_v0_view`和`odd__ssa_v0_view`。

完整探针与日志：`indexer_page_base_select_probe_v1.py`、`indexer_page_base_select_probe_v2.py`，
`indexer_page_base_select_probe_cpu_v1/`和`indexer_page_base_select_probe_cpu_v2/`。
已有相同backing的Tensor条件返回修复不能直接外推为GM指针选择支持。
没有尝试将65×64页视图随意reshape为32×128的NZ Tile，因为物理布局等价性未证明。

顺带核对了交接建议中的缓存维护分类：`payload_cache.py`已按Scalar GM读写分类，
保留信号DCCI/DSB和pipeline fence，仅省去无Scalar读写阶段的整DCache维护。
旧`b16_cached_dynamic_precise_v1`已完整PASS，中位838.68μs；03:47也复核过。
本轮只是确认它不是新方案，没有重复实现或重测。

<!-- SOURCE_END history-document-063 -->

---

<a id="history-document-064"></a>
### 历史原文 064：INDEXER_PANEL_GROUP_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PANEL_GROUP_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PANEL_GROUP_EXPERIMENT.md)；原文 1,143 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-064 -->
# Indexer N64两面板循环：未采用

2026-10-17，保留M384/N64/K128 QK与M16/N64/K384 WS，16个展开面板改8组、每组2面板。
组末完成WS，避免Key Right/score跨迭代携带。运行时数学量、FP16 FIX/ReLU、叶分片、
每1024分数交接和AIV Top-K不变。代价是组边界可能打断跨面板重叠。

`indexer_group2_prepare_v1`与`indexer_group2_cpu_v1`通过。
实际分配及生成循环见`indexer_group2_codegen_audit.json`、`indexer_group2_generated_loop.cpp.txt`。
仅group2经过编译和设备测试；group4/8只有源码生成/列覆盖核对。

任务`task_20261017_080841_317060310914`终态exit0，输出、Top-K、六份cache/state、
图、30代信号、保护区均PASS。完整无插桩L1三窗920.58/924.02/918.76μs，
中位920.58μs；同轮原动态PTO中位909.50μs。证据`indexer_group2_result.json`。
没有收益，未采用、不扩档位。现选B16历史807.16μs（不是本任务同进程对照）。

未测到I-cache miss，不能把代码缩短视为已识别的性能收益。
每次调用恢复fixture，不代表连续decode轨迹；未运行Native forward。

<!-- SOURCE_END history-document-064 -->

---

<a id="history-document-065"></a>
### 历史原文 065：INDEXER_PARTITION_EXPERIMENTS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PARTITION_EXPERIMENTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_PARTITION_EXPERIMENTS.md)；原文 4,923 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-065 -->
# 2026-10-17：128K/B16 Indexer分片与Acc流水实验

两项候选均通过完整CSA固定fixture的精度和图重放检查，均未替换当前选择。
使用正式`/data/model/DeepSeek-V4-Flash-0731-w8a8`第4层权重、B16/S6和历史131072。
性能口径始终是完整CSA的L1 incore首尾，包含核内等待和阶段间空隙，不执行Native forward。

|方案|完整L1三窗/μs|中位/μs|同轮原动态PTO中位/μs|采用|
|---|---|---|---|---|
|既有选择：QK地址缓存|见原结果|815.84|902.80|保持|
|Indexer每叶16384候选|829.22 / 829.42 / 813.78|829.22|897.84|否|
|Indexer M192/N64双Acc|968.92 / 965.56 / 972.00|968.92|904.72|否|

现选SPMD与新候选没有同一进程直接A/B，历史差值不能当作精确的回退幅度。
尤其不能用第一项的单窗最小值813.78替代三窗中位数宣布提升。700μs目标尚未达到。

## 每叶容量8192→16384

代码为`indexer_leaf_capacity.py`，入口开关`--indexer-leaf16384`。
原配置每叶最多8192候选，131072历史加S6对应最大32769个压缩候选；
按3的倍数均衡后是6叶。此前NEXT_STEP中“9叶”的推测错误，现已修正。
新容量16384对应3叶；Cube、Vector与最终Top-K merge同时采用同一计划。

- 每个Cube仍执行22个1024分数块；query和系数的加载轮次4→2。
- 保留原分数数学、分页读取、1024块交接、FFTS协议和pair_arena布局。
- 最终merge的有效叶数随device长度计算，不把131072写入核内候选范围。
- Top-K的分组合并改变，数学等价不足以证明tie行为相同，因此仍做原exact检查。

`indexer_leaf_prepare_v1`的CPU模型覆盖4133种长度，每种最多5个有效前缀，
检查完整覆盖、无重叠、生产者与merge有效叶数一致、叶容量和arena根数上限。
该模型覆盖范围不等于当前图的页表和KV allocation支持范围。
`indexer_leaf_cpu_v1`完整编译通过。

任务`task_20261017_034636_30078467438`退出0，结果位于
`b16_cached_indexer_leaf16k_v1/result.json`与`indexer_leaf_result.json`。
输出、Top-K、六份cache/state、图重放、30代信号及保护区均通过；
只验证固定fixture，不代表连续decode或该候选的动态metadata图变体已经验证。

## M384/N64→两组M192/N64

代码为`indexer_score_pingpong.py`，入口开关`--indexer-score-pingpong`。
原INT32分数Tile为384×64，占96KiB；本候选分两组三query，每份192×64占48KiB。
先提交两份QK，再写回第一份，使两份结果的生命周期相交，阻止分配器将其别名。
保持原6叶计划、query可见范围、分页地址、分数缩放/Relu、GM交接及Vector Top-K。

完整CPU编译`indexer_pingpong_cpu_v1`通过。逐16组核对实际IR：
两份Acc分别位于0、49152字节处，均不重叠；生成C++的两次TMATMUL
也都在第一份TINSERT/FIX前。没有依靠固定MemRef、修改编译器或删除同步。
这些证据证明地址和提交顺序，不证明实际硬件重叠幅度。

首次审计错误地按TMOV寻找量化写回，断言失败；实际指令是TINSERT。
只修正审计匹配，算子未因此改变。复现工具为`indexer_score_codegen_audit.py`。

任务`task_20261017_035437_31474978052`退出0，结果位于
`b16_cached_indexer_pingpong_v1/result.json`与`indexer_pingpong_result.json`。
完整输出/Top-K/状态、图重放、30代信号及保护区均通过，但整层明显没有提速。

|Indexer AIC生成代码中的静态位点|现选|双Acc候选|
|---|---:|---:|
|TMATMUL|32|64|
|TINSERT/FIX|16|32|
|TSTORE|16|32|
|set_flag / wait_flag|166 / 166|187 / 187|
|PIPE_ALL|1|1|

缩小Tile使两份Acc能同时容纳，也增加了矩阵指令、写回和同步提交次数。
该权衡没有在整层上产生收益；没有阶段探针，不能把静态指令数量换算成具体耗时，
也不能声称已测得Cube/FIX重叠程度。此候选不扩到其他档位，不加入默认选择。

## 复现入口与约束

在本目录使用`run_device_compare.sh 16 131072 新输出目录`，共同参数为：

```text
--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile 本目录/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices
```

两项实验分别追加`--indexer-leaf16384`或`--indexer-score-pingpong`，不能组合。
设备任务必须通过`task-submit --device auto --max-time 1800`提交。
wrapper会保存实际driver；首次精度检查失败就停止，不继续采性能。
运行时metadata仍是device Tensor，未增加host回读。

实际编译产物的地址审计可以单独执行，不需要设备：

```bash
python indexer_score_codegen_audit.py \
  --artifact indexer_pingpong_cpu_v1/kernel/ptoas \
  --output 新审计结果.json
```

以上所有改动只在本实验目录，冻结静态版本、生产入口和PyPTO/Simpler仓保持原状。

<!-- SOURCE_END history-document-065 -->

---

<a id="history-document-066"></a>
### 历史原文 066：INDEXER_ROLLED_PANELS_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_ROLLED_PANELS_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_ROLLED_PANELS_EXPERIMENT.md)；原文 2,852 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-066 -->
# 2026-10-17：Indexer N64面板循环化的CPU失败记录

本候选未执行设备，无精度或性能结果，未纳入现选，也未给device driver增加开关。
实现只位于`indexer_rolled_panels.py`，`build_variant`提供CPU实验参数
`indexer_rolled_panels=True`。没有修改PyPTO、PTOAS、ISA或Simpler。

## 尝试的目的

现选每个1024分数块展开16个N64面板。候选保留QK M384/N64/K128、
WS M16/N64/K384、FP16 FIX缩放/ReLU、原Native页读取及AIV Top-K。
先剥离QK0，再循环panel1..15，同时做当前QK和上一面板WS，最后排空WS15。
源码矩阵乘位点从32变4，运行时仍是16次QK与16次WS。
每块FFTS交接次数及叶末尾排空不变，没有减少数学工作或放松同步。

动机是减少重复代码和标量指令流；没有测到指令缓存miss，不能宣称已证明瓶颈在指令缓存。
`indexer_rolled_source_contract.json`只记录源码顺序；其页表达式自比较不是独立正确性测试。

## v1：Key池作为循环携带值

`indexer_rolled_prepare_v1`入口准备通过；`indexer_rolled_cpu_v1`完整代码生成失败：

```text
tile.move requires distinct source and destination addresses in Mat,
but both resolve to byte offset 180224
```

原设计携带Key L1池、Key Right和上一score L1。源码没有主动调用Mat→Mat move，
循环/分支合流lowering插入了移动，而内存规划将相关Key池放到同一地址。
错误在PyPTO后端生成阶段触发，未到设备；没有删除该校验或手工改地址。

## v2：移除不必要的Key池携带

下一Key的L1内容每次都在本迭代写入再提取，不需要跨迭代保留。
v2改成局部L1临时区，只携带Key Right和score L1。
`indexer_rolled_prepare_v2`通过，`indexer_rolled_cpu_v2`仍失败，PTOAS报告：

```text
'pto.tmov' op expects a supported tmov address-space pair for this target
```

实际PTO中合流和循环回边出现Right→Right的INT8 128×64 `pto.tmov`，
另有score的Mat→Mat移动。当前A3编译路径拒绝前者。相关IR保存在
`indexer_rolled_v2_unsupported_moves.pto.txt`，完整产物仍在本地CPU目录。
这不能仅归咎于PTOAS：从源循环携带到插入移动的变换也需要纳入分析。
没有给上游提交未经最小化的新issue，没有将此问题说成现选版本的设备错误。

## 后续可以考虑的方向

可以进一步研究每轮处理两个固定面板的循环，在一轮内完成两份Key和score的消费，
避开Right Tile跨循环携带。这样会改变跨面板预取/排空关系，不能假定保留原性能；
需重新核对实际缓冲与同步后才考虑设备验证。该方向尚未实现或测试。

所有失败和原生成源码保留。当前没有本候选的运行任务，不要把CPU准备成功当作完整编译成功。

<!-- SOURCE_END history-document-066 -->

---

<a id="history-document-067"></a>
### 历史原文 067：INDEXER_SHORT_TAIL_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_SHORT_TAIL_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_SHORT_TAIL_EXPERIMENT.md)；原文 3,798 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-067 -->
# 2026-10-17：Indexer设备端短尾分支

本候选未替换现选七档。B16已完整通过但没有明确性能收益；B24也已完整通过，但没有明确性能收益。
目标仍是多档位分别编译、128K/B16完整无插桩L1低于700μs，不把局部计算量当性能。

## 改动与边界

`indexer_short_tail.py`在原N64的1024候选块循环里，读取已有设备计算的
`valid_count - score_begin`。剩余1..64项时只执行第一面板；其余情况保留
原完整路径AST。运行时长度、位置和页表仍来自device Tensor；没有新增Host取值、
GM工作区或硬编码131072。开关`--indexer-short-tail`默认关闭。

保留QK M384/N64/K128、WS M16/N64/K384、FP16 FIX缩放/ReLU及原归约。
每个1024块的FFTS READY/CONSUMED次数、两槽复用和末尾排空不变。
只改Cube；AIV仍按原逻辑读分数、逐元素乘scale、按query有效长度fillpad、再Top-K。

短路径没有写`score_transfer`的64..1023列。这些是内部无效尾部，不能被当作
新计算出的有效值；原AIV在进入排序前排除它们。沿用原S6组内位置有序的前提，
最后query提供该组最大可见范围。未扩大到不满足原输入合同的乱序position场景。

## CPU证据

- `indexer_short_tail_prepare_v1`、`indexer_short_tail_cpu_v1`完整通过。
- `indexer_short_tail_contract.py/json`核对原九条fillpad分支，并枚举叶内1..8192
  有效数、184314个query/step组合，其中2554个短尾组合。检查未写区不进入有效排序前缀。
- `indexer_short_tail_codegen_audit.json`核对实际C++：短/完整路径矩阵乘2/32，
  FIX写L1为1/16，GM store为1/16，两路径内部PIPE_ALL均0。
- B24的`indexer_short_tail_b24_prepare_v1`准备通过；设备任务自行完整编译。

`indexer_short_tail_workload.json`保留原叶分配、按物理核统计面板数量：

|128K/S6档位|最繁忙核原面板数|短尾后|解释|
|---|---:|---:|---|
|B4|96|96|仅少数核有短尾，最繁忙核不变|
|B8|176|176|同上|
|B16|352|352|16核省15面板，另8核不变|
|B24|528|513|每核都有一次短尾|

此模型不含FFTS、Top-K、缓存及调度成本，不是耗时预测。它说明不能用B16某些核
减少计算推断整层提速，也说明为何只额外验证B24，没有重跑B4/B8或七档全量。

## 设备结果

使用正式W8A8权重、S6、原完整输出/Top-K/六状态exact门槛，以及图重放、30代信号
和保护区。固定fixture每次恢复，不代表连续decode轨迹；不执行Native forward。
计时为完整无插桩L1 incore首尾，保留等待和阶段间空隙。

|档位|任务|单SPMD三窗/μs|中位/μs|同轮原动态PTO/μs|结论|
|---|---|---|---:|---:|---|
|128K B16|task_20261017_074327_274989723191|820.12 / 820.32 / 819.08|820.12|902.04|全部检查PASS，不采用|
|128K B24|task_20261017_074803_283067829725|1146.14 / 1127.42 / 1127.60|1127.60|1175.66|全部检查PASS，不采用|

B16证据为`indexer_short_tail_b16_result.json`和`b16_row_deps_indexer_short_tail_v1/`。
没有同进程现选SPMD配对，不把历史807.16与本轮820.12的差值称为精确回退。

复现沿用`run_device_compare.sh`，追加`--indexer-short-tail`；B16用现选earliest_ready、
QK地址缓存和row deps；B24用现选critical_path、自己的成本文件，不加row deps或QK地址缓存。
所有设备执行经`task-submit --device auto`，依赖仓、冻结静态版本与生产入口均未修改。

B24任务exit0，检查范围与B16相同。证据`indexer_short_tail_b24_result.json`及
`b24_indexer_short_tail_v1/`。工作量减少没有转成可确认的整层收益；没有同进程现选
配对，不能把历史1121.12与本轮1127.60之差当精确回退。保留候选，停止扩测。

<!-- SOURCE_END history-document-067 -->

---

<a id="history-document-068"></a>
### 历史原文 068：INDEXER_TRANSFER_2048_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_TRANSFER_2048_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_TRANSFER_2048_EXPERIMENT.md)；原文 2,812 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-068 -->
# Indexer 2048分数交接：精度通过，未采用

2026-10-17，仅128K/B16/S6。原动态PTO为对照，没有运行Native forward。

## 实现范围

`indexer_transfer_2048.py`保留原M384/N64/K128 QK、M16/N64/K384 WS、
16面板展开流水、FP16 FIX/ReLU、Native页地址和叶分片；每两块1024分数合成
一次2048交接。私有双槽GM由[288,1024]扩大为[288,2048]，增加1179648字节。
AIC只算实际存在的1024子块；奇数尾叶不会新增数学量。AIV每轮加载3×2048分数，
段span受本叶剩余有效数限制，排除尾槽未写区。各query排序/合并数学保持。

原1024轮每两轮才排序一次，新轮每轮排序一次，所以root初始化由step<2改为step==0。
READY仍在AIC FIX，ACK仍在AIV MTE2完成分数读取之后；叶尾排空min(steps,2)。
没有删除跨核数据发布/等待。正式根Tensor与Native cache/state布局不变。

## CPU证据

`indexer_transfer2048_prepare_v1`和`indexer_transfer2048_cpu_v1`完成代码生成；
新score load为3×2048，stride2048，AIV阶段Vec最大端点131072字节。
`indexer_transfer2048_codegen_audit.json`只描述实际分配和指令，不代替生命周期/硬件证明。
旧result中的runtime_abi_supported=false是32参数旧检查，本会话使用独立256参数runtime。

`indexer_transfer_contract_v2.json`枚举1..8192叶长、40962种有效排序段，
同时比较排序段和root初始化选择；双槽计数模型只证明计数流程，不证明DMA可见性。

## 失败与修正

首轮`task_20261017_082119_33910122896` exit1。六份cache/state相同，
Top-K有48999个不同位置，输出最大差0.1484375；在精度门槛退出，没有采性能。
原因是本候选保留旧step<2，第二个2048段错误覆盖第一个段的root。
只修三处初始化判断；反向恢复这三处后，Top-K数学AST与原逻辑逐项相同。
这是本目录变换遗漏，不归因为PyPTO、Simpler或ISA。
失败原始entry、result和日志保留，摘要`indexer_transfer2048_failure_v1.json`。

## 修正后设备结果

`task_20261017_082629_34925986670` exit0，`b16_row_deps_indexer_transfer2048_v2`：
输出/Top-K/六份状态、eager/graph、30代信号、保护区全部PASS。
完整无插桩L1三窗836.98、834.68、819.72μs，中位834.68μs；
同轮动态PTO为889.94、899.98、909.36μs，中位899.98μs。
Event中位单SPMD940.45μs、动态954.57μs，口径独立于L1。

没有显示出优于现选B16策略的收益，未采用、不扩其他档。
现选B16首轮807.16μs及同进程复核817.88μs是历史数据，
这里没有现选同进程A/B，不把差值解释为精确回退幅度。
每次调用恢复fixture，不代表连续decode轨迹。
证据：`indexer_transfer2048_result_v2.json`及完整任务目录。

<!-- SOURCE_END history-document-068 -->

---

<a id="history-document-069"></a>
### 历史原文 069：INDEXER_WS_LOAD_ORDER_CPU.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_WS_LOAD_ORDER_CPU.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/INDEXER_WS_LOAD_ORDER_CPU.md)；原文 1,810 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-069 -->
# Indexer WS加载顺序CPU检查（2026-10-17）

**没有提交NPU。** 仅把15处上一块WS的Right extract放到当前QK matmul之后。
完整编译通过，但其中1对当前QK/上一块WS Right区间发生复用，预期的并行存储条件不成立。
当前选择不变，不把源代码顺序变化称为硬件重叠。

现选本来就有QK(current)/WS(previous)和下一Key预取；本候选不重复实现整条流水。
`indexer_ws_load_order.py`逐对检查无值依赖，全部434个原顶层语句AST多重集合保持相同，
原最后5条排空语句位置/内容及其他函数不变。prepare只在该CPU进程内注入catalog，
没有改默认形状派发或新增设备driver选项。

准备v1/v2因遍历被原地交换的列表而重复匹配已交换节点失败；v3修为原列表快照后，
对末块名称的假设不正确，15处交换正确但末块计数0。最终v4核对真实buf_last_pair及
最后5条语句，准备/完整编译通过。CPU_v2在source不存在时退出，未进入编译。
这些是变换器/试验编排错误，不是PyPTO编译器故障。所有日志保留。

| 静态位点 | 现选 | 加载后移 |
|---|---:|---:|
| matmul | 32 | 32 |
| PIPE_ALL | 1 | 1 |
| M→MTE1 wait | 32 | 32 |
| MTE1→M wait | 37 | 37 |
| set/wait各 | 166 | 165 |
| 当前QK/前块WS的Right全部互斥 | 是 | 否 |
| 当前QK/前块WS的Acc全部互斥 | 是 | 是 |

这里只检查实际区域与静态位点，不能用少一个set/wait宣称性能收益。
若继续，应先保证需要重叠的Right数据拥有独立生命周期及存储；不能删掉保护真实复用的同步。

证据：`indexer_ws_order_prepare_v4/`、`indexer_ws_order_cpu_v4/`、
`indexer_ws_order_codegen_v4.json`及`.cpp.txt`、`indexer_ws_order_selected_codegen.json`。

<!-- SOURCE_END history-document-069 -->

---

<a id="history-document-070"></a>
### 历史原文 070：KV_TOUCH_SIGNAL_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/KV_TOUCH_SIGNAL_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/KV_TOUCH_SIGNAL_EXPERIMENT.md)；原文 4,109 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-070 -->
# KV原样回写标记：保留汇总通知，删除业务空操作

2026-10-17。本轮只修改测试目录中的单SPMD生成器，原动态对照、生产代码和依赖仓保持。
原`kv_touch`对Native原始KV起点执行一次BF16 `[1,512]` load/store，数值不变。
它在TMR中用来建立写回到QK/PV的WAR依赖；单SPMD已把这些依赖显式写入等待表。

## 完全删除节点：CPU通过，未上卡

`kv_touch_elision.py`先证明load/store/返回为严格同址自拷贝，核对唯一写回父任务，
将QK/PV等待直接改接`csa_cache_writeback`。删除marker后，所有保留任务的祖先集合
等于原集合去掉marker；真正的写回顺序未放宽。随后重新分核、重建等待与发布序号。

prepare_v1检查失败：调用清单中的实参类型已去掉Out装饰，而真实核体形参仍为pl.Out。
这是本候选检查器的假设错误；改用核体形参和allocation联合核对。失败记录保留。
prepare_v2、cpu_v1完整生成通过。

但任务66→65、成员973→972后，远端等待12183→12673，增加490项，发布数仍722。
原marker还承担完成状态汇总，删掉它会把等待展开到多个写回核。
因此不直接做设备测试。`kv_touch_elision_cpu_comparison.json`保存计数，
不把该计数差等同于实测微秒，也不声称完全删除一定变慢。

## 纯信号节点候选

`kv_touch_signal_only.py`和`--kv-touch-signal-only`保留原节点、物理核序列、等待表、
代际与通知，只删除marker内部1024B自拷贝及对应的4个fence、2个全Scalar cacheinvalid。
其他阶段的源码经还原后逐AST相同。生产者的业务写出/发布、QK消费者的读前维护保留。

安全边界：这是严格零业务计算的汇总者。它没有要发布给后继的新业务数据；
所有写回核已经按原协议完成流水排空和缓存发布，汇总者按原等待表确认后才通知后继。
信号自身的DCCI/DSB没有删除；没有依赖Scalar跨核缓存一致性。消费者仍执行原有
业务读取前的维护。不能把此删除规则推广到普通数学阶段。

prepare_v1与cpu_v1通过。生成C++的marker阶段：TLOAD/TSTORE各1→0，
PIPE_ALL 5→1（剩余的是等待轮询内部原有屏障），全Scalar DCCI 2→0，
TWAIT/TNOTIFY均各1不变。见`kv_touch_signal_codegen_audit.json`和两份C++阶段片段。

任务`task_20261017_091307_19138223772`，目录`b16_row_deps_kv_touch_signal_v1/`，
只测128K/B16/S6，使用现选QK地址缓存/行依赖/100tick轮询。
当前提交后等待终态，未预记PASS或性能收益。完整L1 incore首尾为验收口径；
不把Event或插桩时间混入，不运行Native forward。

## 首轮终态与一次同进程复核

首轮任务exit0，完整输出/TopK/六状态/图/30代/保护区PASS。
L1三窗802.96/812.72/798.04μs，中位802.96；同轮动态894.00μs。
与历史现选807.16差距小，不能用跨轮结果证明收益。
因此提交一次同进程复核task_20261017_091800_27634922996，
目录b16_kv_touch_signal_paired_v1；只2次预热及每侧3窗L1，不重复Event/动态metadata。
新增--paired-kv-signal-spmd-only复用已有比较器，原主Q模式保持。
最终采用结论等待终态；不再重复N32或其他七档。

## 同进程终态：未采用

`task_20261017_091800_27634922996` exit0。双方eager/图/输出/Top-K/六份状态及保护区PASS，
双方各7代完成信号PASS。对照入口AST与现选`b16_cached_row_deps_v1`相同，
见`kv_touch_paired_reference_audit.json`。

| 完整无插桩L1 | 三窗 μs | 中位 μs | 均值 μs |
| --- | --- | --- | --- |
| 现选 | 805.14 / 808.22 / 807.50 | 807.50 | 806.95 |
| KV纯信号 | 812.08 / 803.92 / 812.32 | 812.08 | 809.44 |

候选中位慢0.57%，没有测出收益。纯信号替代在本次正确性范围内通过，
但删除几条源指令不能直接推出整层加速，现阶段不纳入选择、不扩档、不重复测试。
七档选择保持，128K/B16的700μs目标仍未达到。每次调用恢复fixture，非连续decode。

<!-- SOURCE_END history-document-070 -->

---

<a id="history-document-071"></a>
### 历史原文 071：MEMBER_SCHEDULE_CPU_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/MEMBER_SCHEDULE_CPU_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/MEMBER_SCHEDULE_CPU_EXPERIMENT.md)；原文 2,114 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-071 -->
# 2026-10-17：先按成员依赖分核的 CPU 评估

本轮没有提交 NPU 任务，也没有生成候选二进制。现选七档保持不变。

`member_schedule.py` 从已证明的行依赖重新构造数学 DAG，再做成员级分配；
原方案则是先按整个 task 分核，再收窄成员等待。共 973 个物理成员，
MIX 的 AIC/AIV0/AIV1 绑为一组后得到 877 个调度单元。
保留 Indexer/QK 的同组映射、216 条 FFTS 资源复用边、明确的数学依赖；
重新建立同核顺序、传递约简及发布/等待表，并检查翻译前后祖先集合相同。

以下均是固定原阶段成本的离线估计，不是设备性能，也不是硬件下限：

| 策略 | 原安排估计 μs | 新安排估计 μs | 变化 μs | 远端等待 | 发布 |
| --- | ---: | ---: | ---: | ---: | ---: |
| critical | 771.53 | 802.49 | +30.96 | 15569 | 747 |
| earliest | 771.53 | 768.79 | -2.74 | 12696 | 720 |
| finish | 771.53 | 858.52 | +86.99 | 13643 | 739 |

最好的估计仅减少 2.74 μs，而且比现选的 12183 次远端等待更多，暂不上卡。
这不能证明成员级重排没有价值，只说明当前三种启发式没有形成值得验证的优势。
脚本是 CPU 原型，尚未接入 `shape_specialization.py`。输入使用现选行依赖审计，
将来接入时仍须重新运行源数学合同核对，不能只依赖历史 JSON。

结果位于 `member_schedule_{critical,earliest,finish}_cpu_v1/`。每份目录包含
`audit.json`、`schedule.json`、`logical_and_physical_graph.json`。
生成器和三份归档中已明确标注原分核/等待统计被成员排程替代，避免继承旧审计的
“原核顺序保持”描述。此修正只改描述性元数据，没有改已计算的调度图或结果。

复现示例：

```bash
python member_schedule.py \
  --schedule b16_cached_row_deps_v1/specialization/bindings/h131072_b16_prepared.json \
  --row-audit b16_cached_row_deps_v1/specialization/control_variants/h131072_b16/row_dependencies/audit.json \
  --cost-profile b16_dynamic_l1_cost.json --policy earliest --output 新目录
```

<!-- SOURCE_END history-document-071 -->

---

<a id="history-document-072"></a>
### 历史原文 072：OUTLINED_CACHED_WAIT_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/OUTLINED_CACHED_WAIT_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/OUTLINED_CACHED_WAIT_EXPERIMENT.md)；原文 3,259 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-072 -->
# 全阶段独立编译与现选等待策略的组合诊断

2026-10-17。复用`stage_codegen.py`和`signal_polling.py`，没有修改依赖或生产入口。
外部仍为一个SPMD任务；这里的74个函数是同任务内部device函数，不是新增74个运行时任务。

## 为什么增加这一对照

早期全部阶段独立编译/noinline的1044.54μs，发生在默认等待时期。
当前约807μs的版本已采用cached100、QK索引缓存、行依赖，不能用旧轮次直接判断该组合性能。
此次只评估现有数学和同步语义下的编译边界影响，没有叠加本轮未获收益的批量RoPE。

## CPU证据与执行前门禁

`compile_outlined_cached_wait.py`从现选`b16_cached_row_deps_v1`的源码/布局开始，
在全部编译扩展生效后解析JIT artifact。CPU结果在`outlined_cached_wait_cpu_v1/`：

- 74个阶段完整独立编译，全部noinline，内部只传必要的参数和外层计算的同一代际。
- 仅在所有阶段已分离时清理外层原数学事件，且原set/wait列表必须严格配对。
  helper自身的流水和FFTS事件保持，业务数据维护保持。
- 实际交给CCEC的AIC/AIV源码各含74处替换等待；DCCI后volatile读、100tick退避正确，
  无原TWAIT残留。外层无遗留流水事件，各阶段正好调用一次。
- 审计最初没识别后端正规化后的`static __aicore__`拼写；修正识别规则后通过，
  没有修改候选C++或放宽覆盖条件。

`outlined_cached_audit.py`在真正设备执行前再次检查上述最终源码。
driver只开放已核对的B16/128K、全阶段、compact参数、noinline、QK Vector索引、
行依赖、earliest_ready、cached100组合；部分阶段或缺少noinline仍拒绝。
CPU参数门禁记录在`outlined_cached_argument_guard_audit.json`。

## 任务记录

- `task_20261017_131724_17808923318` / `b16_row_deps_outlined_cached_v1`：
  被另一处旧行依赖组合保护拒绝，exit1，尚未执行算子；不记功能或性能结果。
- 补齐同一窄范围组合的保护后，提交`task_20261017_131834_1981138300` /
  `b16_row_deps_outlined_cached_v2`。先完整exact、六状态、图、30代与保护区，再采三窗L1。
  截至本记录等待结果。

这是C++函数边界归因实验，`eligible_for_700us_target=false`。
即使更快，也不能直接宣布正式PyPTO接口接入或700μs目标完成。


## 设备结果：完整PASS，未采用

task_20261017_131834_1981138300 exit0。现选数学/行依赖与全部阶段独立noinline+cached100的组合完整通过：eager/graph/final输出、TopK、六状态零差异，30代通知与所有保护区PASS，实际74helper/事件闭合/100tick门禁PASS。完整L1 807.02/805.34/810.66，中位807.02；同轮动态886.36/906.28/902.84，中位902.84。与现选约807持平，不采用、不扩档、不再重复该组合。该归因诊断eligible_for_700us_target=false。不能把静态屏障数减少直接当收益，也不能据此断言同步完全没有成本：函数调用/独立规划可能抵消其变化。

详见`outlined_cached_result_v2.json`及原结果目录。40秒观察等待曾超时，但任务正常继续并exit0；未取消、未把观察超时当kernel故障。

<!-- SOURCE_END history-document-072 -->

---

<a id="history-document-073"></a>
### 历史原文 073：PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md)；原文 4,188 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-073 -->
# CSA 大 SPMD 中冗余流水同步消除的范围诊断

2026-10-16。本文记录实际 CPU 编译证据和设备对照，**不是已确认的整层性能根因**。
没有修改 PTOAS、PyPTO 或 Simpler 的共享源码/安装。

## 观察到的编译行为

只读本地 PTOAS 的
`lib/PTO/Transforms/InsertSync/PTOInsertSync.cpp`：

- `hasGatherScatterLikeOps` 检查整个函数是否含 TGather、TGatherB、TScatter、
  MGather 或 MScatter。
- 命中任一项时，跳过整个函数的 `RemoveRedundantSync`，随后仍做 EventId 分配。
- 附近注释解释的是 A5 上 gather/scatter 相关同步可能被过度消除的问题；
  这个跳过条件本身没有限定为 A5。

因此，A3 大融合函数中一个阶段使用 gather，也会让其他阶段失去这一步优化。
这只能说明编译范围不同，不能据此直接删除保守分支或认定同步可以安全减少。

## 已安装 PTOAS 0.66 的直接证据

真实命令保存在 `ptoas_sync_phase_evidence/commands.json`；没有依赖修改后的编译器。
输入为同一套 CSA IR 的整层函数和独立 widen 阶段，命令使用：

```text
ptoas INPUT.pto --enable-insert-sync --pto-level=level3 \
    --pto-insert-sync-debug=1 -o OUTPUT.cpp
```

两份输入都声明 `pto.target_arch = "a2a3"`。日志、摘要及生成 C++ 位于
`ptoas_sync_phase_evidence/`。

| 编译对象 | Sync Motion 后 set/wait 对数 | RemoveRedundantSync 后 | EventId 分配后 |
| --- | ---: | ---: | ---: |
| 整层 AIC | 3188 | 772 | 938 |
| 整层 AIV | 2584 | 2584 | 1060 |
| 独立 widen AIV | 46 | 28 | 33 |

整层 AIV 的 EventId 分配还把部分事件转为 barrier：日志中的 barrier 数从 1169
变为 2854。以上是编译 IR 中的静态操作数量；包含条件与循环，不是动态执行次数，
也不是耗时。C++ 文件中额外公共 helper 的定义还会影响文本计数，不应混用口径。

## 设备对照结果

使用正式模型第 4 层权重、128K/B16/S6；两侧均为完整 CSA PTO，不运行 Native
forward。L1 指标为整层 `min(incore start)` 到 `max(incore end)` 的三窗中位数。

| 方案 | 同轮原动态 PTO / μs | 单 SPMD / μs | 结论 |
| --- | ---: | ---: | --- |
| 原 balanced 单 SPMD | 902.28 | 1067.36 | 当前基线 |
| 只独立编译 widen，内部内联 | 894.20 | 1120.22 | 没有改善 |
| 全部 68 个阶段独立编译，内部内联 | 901.72 | 1153.34 | 没有改善 |
| 全部阶段独立编译，noinline | 893.54 | 1044.54 | 相比内部内联更快，未达到目标 |

后两项是 `stage_codegen.py` 的测试进程内诊断：从同一融合 IR 提取阶段，
保留原等待、数学、写回屏障及通知；分别经过原 MemoryReuse、AllocateMemoryAddr
和 PTOAS 后再组合。外部仍是一个 SPMD 任务，完整泳道仍只有 72 条核记录。

全阶段独立版本把内部参数裁剪为每函数 8～26 项。由于外层已不执行数学操作，
核对并移除外层原数学的成对 auto-sync 初始化/回收，避免与内部函数的事件编号
生命周期混用。内部函数的流水事件、MIX 跨核事件保持原样。

两项设备实验均通过输出、Top-K、六份状态、保护区、图重放及 30 代信号检查。
它们使用固定输入恢复，不代表连续 decode 轨迹；C++ 组合诊断也不作为正式接口
完成或 700 μs 目标验收依据。

## 后续判断边界

保留设备函数调用边界的 noinline 对照已经 PASS，见上表，三窗为 1037.44、1044.54、
1063.24 μs。外部 PyTorch PipeUtilization 采样也已完成，原始表在
`b16_pipe_profile_v1/pipe_profile/`；AIC MAC 及 AIV Vector 时间接近，单 SPMD
的 Scalar、MTE2 时间及 ICache miss rate 增加，尚不能据此确认单一根因。当前 kernel
模式不接受 Simpler `enable_pmu`，不绕过该限制；外部 PMU 覆盖完整持久核，
包括外围调度，其时间不能替代 L1 incore 指标。

现有证据支持“融合改变了同步消除范围”，**不支持“修复这一点即可降到 700 μs”**。
只有确认可安全消除的同步并获得设备收益后，才适合进一步提出组件修改建议。

<!-- SOURCE_END history-document-073 -->

---

<a id="history-document-074"></a>
### 历史原文 074：QK_ALL_VALID_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_ALL_VALID_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_ALL_VALID_EXPERIMENT.md)；原文 2,094 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-074 -->
# 2026-10-17：QK全有效块快路径

候选`qk_all_valid.py`在原QK的tick循环外，从device plan扫描本组query×5个块。
仅全部标记>0时使用快循环；任意标记非正则执行原循环。三个物理成员读取同一
不可变plan，原plan前驱和FFTS依赖保持。不读取Host端Tensor内容，不增加GM缓冲。

快循环去掉已证明恒真的外层块判断，保留内部稀疏行负索引、SWA页、尾部tick、
softmax、PV及输出。展开条件的末尾yield改为普通SSA赋值；回退循环AST保持原样。
行依赖审计先检查原访问体，再加入条件专门化，不以新分支绕过访问范围核对。

prepare/cpu_v1完整通过。实际PTOAS C++两个分支的静态统计：

| 物理函数 | 快循环PIPE_ALL | 回退PIPE_ALL | 快循环mask读取点 | 回退mask读取点 |
| --- | ---: | ---: | ---: | ---: |
| AIC | 0 | 0 | 0 | 2 |
| AIV | 3 | 4 | 0 | 4 |

AIV源码六个判断经B16常量分支优化后，实际回退只剩四处mask读取。上述是静态
位点数，不能当作每次执行次数或性能收益。核内前置扫描仍有成本，代码也变大。
FFTS发布统计使用实际`ffts_cross_core_sync`，初版误搜`set_flag_dev`显示零，
已修正审计字段，未改变算子。证据`qk_all_valid_codegen_audit.json`及四份C++片段。

任务`task_20261017_072611_24632809131`已提交，目录
`b16_row_deps_qk_all_valid_v1/`。正式权重128K/B16/S6；先原exact/状态/图/保护区，
通过后比较原动态PTO与单SPMD的完整三窗L1。未运行Native forward。当前尚无终态，
不预记PASS，不改变现选七档。

## 终态：完整通过，未采用

任务exit0，输出/TopK/六状态、eager/图、30代及保护区全部通过。
单SPMD三窗804.30/817.26/820.92μs，中位817.26；同轮原动态905.18μs。
没有现选同进程A/B，未显示明确优势，未采用、不扩七档。
不能把当前完整fixture通过称为所有padding场景或连续decode轨迹已通过。
结果`qk_all_valid_result.json`，全量记录保留于原任务目录。

<!-- SOURCE_END history-document-074 -->

---

<a id="history-document-075"></a>
### 历史原文 075：QK_DIRECT_L1_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_DIRECT_L1_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_DIRECT_L1_EXPERIMENT.md)；原文 5,316 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-075 -->
# QK/PV直接L1供数实验

本候选仅在实验生成的B16/S6单SPMD内启用，未替换生产入口或七档选择。
目标是去掉Vector上的64×512 BF16 KV临时块与GM交接；是否提速以完整CSA的
无插桩L1 incore首尾为准，不能从少了一次搬运直接推断。

## 改变与保持

旧路径：Native KV → 两AIV各64行UB → GM kv_transfer → Cube三槽L1。
候选：Native KV → Cube三槽L1。AIV继续执行原32-head softmax和PV在线合并。
QK/PV矩阵乘、五块合并顺序、归一化/RoPE/输出布局都保持。

原`csa_slots_build_valid_qk_plan`生成的私有稀疏索引临时区，仅该生产者和QK使用。
两AIV分别处理每query的256列，将逻辑索引批量转为物理KV行并原地写回。
外部TopK输出不改变；下一次调用仍由原valid-plan重建逻辑索引。
负索引保留负号，Cube按原无效行语义补零。页表、位置仍为原设备Tensor，未新增Host取值。

每组lane0在原kv_transfer的`core*384`起点写32行零值，供无效行和SWA尾部读取。
该空间现在只有零源用途。没有新GM分配或外部参数，原三槽L1大小不变。
SWA按最多五个32-token页段搬运；不完整页、负页、尾部逐段补零。

## 交接协议

|事件|生产者发布点|消费者|
|---|---|---|
|4|两AIV的MTE3完成物理索引/零源写入|Cube等待两lane，执行fence/cacheinvalid/fence后读取地址|
|1|Cube FIX完成score写回|两AIV计算原softmax|
|2|两AIV MTE3完成probability/mi/li写回|Cube计算原PV|
|3|Cube FIX完成PV写回|两AIV读取并在线合并|
|0|两AIV MTE2完成左右两份PV读取|Cube在复用三槽前等待两lane|

事件0原来表示KV_READY，现在表示旧PV已消费。不能只删除旧KV_READY等待：
它还间接限制流水超前程度。候选在Cube的tick≥3时等待tick-3对应的有效块；
循环结束后再等待最后一块，确保进入下一阶段/调用前没有残留事件。
有效块标记在三方保持一致，空块不发布也不等待。

地址表由DMA写回，不能假设Scalar缓存自动一致。Cube等待事件4后明确维护缓存；
原跨阶段fence、DCCI和GM代际通知也保留。这里使用已有FFTS组内交接，未增加自制Scalar信号。

## 证据与限制

- `qk_direct_l1_probe_cpu_v1`：真实编译子视图保留父Mat的Rows=384跨度，validRow可为1或运行时值。
- `task_20261017_042346_360920825849`：测试驱动漏初始化kernel runtime，未执行kernel，保留失败。
- `task_20261017_042437_362436518114` / `qk_direct_l1_probe_device_v2`：A3同图7项搬运哨兵PASS。
  三槽逐行、非对齐起点、动态行数、末行及回到原槽，全部数据/相邻行/保护区/只读输入通过。
- `qk_direct_l1_protocol_result.json`：3328个CPU随机交错调度、322443步，2/3代复用；
  包含两query全部1024种有效块组合及4query抽样，事件排空、slot生命周期和无死锁通过。
  SWA有1310种区间覆盖检查。模型不证明真实DMA/缓存可见性。
- `qk_direct_l1_cpu_v1`：完整CSA CPU代码生成PASS。
- `qk_direct_l1_prepare_v1/codegen_audit.json`：实际IR/C++的发布、等待、地址缓存维护、
  PV加载及末尾排空顺序人工核对。手工FFTS循环采用`pl.range`，原纯数学pipeline保留。
- `task_20261017_043021_371809025343` / `b16_cached_qk_direct_l1_v1`：完成exit0。
  原exact、TopK、六份cache/state、图、30代信号和保护区全部PASS，之后完成3窗L1。
  真实NPU构建C++与已审计CPU C++全文相同；仅比较文本，没有hash。

## 完整性能与结论

|路径|三窗完整CSA L1 incore首尾（μs）|中位（μs）|
|---|---|---|
|同轮原动态PTO|892.66 / 920.94 / 888.36|892.66|
|直接L1单SPMD|1096.08 / 1100.10 / 1100.20|1100.10|

候选比同轮原动态增加23.24%，明确不采用、不扩其他档。现选历史815.84μs仍保留；
本轮没有现选单SPMD同进程A/B，不把历史差值解释为精确回退幅度。
因为候选已被性能淘汰，不追加实际候选片段的负索引/短SWA或metadata重放测试；
这些边界不能由小搬运探针、CPU区间模型或固定fixture的PASS代替。

生成代码提供的有限归因：Vector整个QK阶段的`PIPE_ALL`静态位点从9降至5，
但整层变慢。原两AIV各64行的并行搬运变为一个Cube的128行循环，伴随逐行GM Scalar
地址读取与单行ND2NZ。真实128行循环内部没有额外pipe_barrier/set_flag/wait_flag，
不能说是“每行新增屏障”导致。搬运串行化及ND2NZ粒度成本是合理排查方向，但没有新增
内部探针量化各项耗时，不将全部回退定量归到某一项。证据：
`qk_direct_l1_codegen_comparison.json`和`qk_direct_l1_compressed_loop.cpp`。
首次只读统计试图按record注释定位阶段失败；改用C++花括号边界后得到上述计数，算子没改。

另只读确认当前A3的`TPUSH/TPop` Vec→Mat路径仍经GM FIFO，不能假定换接口就能绕过GM。
源码为独立Simpler构建所用`build/pto-isa/include/pto/npu/a2a3/TPush.hpp`的
`pushVec2GMFiFo` / `popMatTileFromGMFiFo`，没有为此改依赖或运行新测试。

所有变更位于本实验目录。冻结静态结果、PyPTO、Simpler、PTOAS及生产算子没有修改。

<!-- SOURCE_END history-document-075 -->

---

<a id="history-document-076"></a>
### 历史原文 076：QK_EARLY_KV_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_EARLY_KV_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_EARLY_KV_EXPERIMENT.md)；原文 1,982 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-076 -->
# B16 向量索引路径的提前 KV 通知（2026-10-17）

完整验证通过，但未观察到优于现选的性能，不采用、不扩档、不重复测量。

生产`EARLY_KV_MIN_TOKENS=144`有旧实测依据。本次不是修正所谓错误阈值，而是核对
新增向量索引路径后，T96是否适合已有的提前通知策略。
`qk_early_kv.py`仅把三个条件中的144改为96，恢复条件后整个AST必须与原始一致。
保留数学、三槽、pre-launch=2、全部FFTS事件和GM访问；开关默认关闭。

`qk_early_kv_prepare_v1/`与`qk_early_kv_cpu_v1/`完整CPU编译PASS。
集成`--qk-early-kv`后的`qk_early_kv_integrated_v1/`入口文本与已编译源码完全一致，
没有重复CPU编译。`qk_early_kv_codegen_audit.py`在所有编译扩展启用后检查实际artifact，
同时保留100tick等待策略源码门禁。

真实AIV C++阶段内顺序：1192等待前块Score → 1203通知当前KV → 1330 row-max →
1357 row-sum → 1424通知Prob → 1464等待PV。两个wait、两个notify位点和9个PIPE_ALL
与现选一致。静态位点/顺序本身不证明运行时重叠或性能收益。

任务`task_20261017_114240_273482232688`，目录`b16_row_deps_qk_early_kv_v1/`，exit0。
正式权重、128K/B16/S6、PTO对PTO；eager/graph/final的输出、TopK和六cache/state均零差异，
30代进度通知、输入/写入范围/工作区保护区全部PASS。每次恢复fixture，不是连续decode轨迹。

| 实现 | 完整无插桩L1三窗（μs） | 中位数（μs） |
|---|---|---:|
| 本次提前通知单SPMD | 823.38 / 816.34 / 833.82 | 823.38 |
| 同轮原动态PTO | 905.06 / 907.42 / 900.04 | 905.06 |

现选B16历史807.16μs，最近另一轮同进程现选807.50μs。本次没有加入现选SPMD配对，
不能把跨轮约16μs差额精确归因，但没有理由用该候选替换现选。
只计完整incore首尾，保留等待/空隙，排除外围调度。摘要`qk_early_kv_result_v1.json`。

<!-- SOURCE_END history-document-076 -->

---

<a id="history-document-077"></a>
### 历史原文 077：QK_MERGE_STATE_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_MERGE_STATE_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_MERGE_STATE_EXPERIMENT.md)；原文 3,827 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-077 -->
# QK/PV循环状态分块（2026-10-17）

> 2026-10-17更正：本次设备产物提前JIT编译，未应用请求的cached_wait/100tick。精度结果保留，不能据此判断数学核改动的性能回退；见[编译上下文修正](../../../tests/pypto_test/csa_multishape_spmd_20261016/COMPILE_CONTEXT_CORRECTION.md)。


**未采用。** B16/128K/S6完整精度通过，但完整无插桩L1中位1033.44μs；同轮原动态PTO为912.20μs。
现选单SPMD此前约807μs，两者不是同轮配对。七档选择保持，不扩档，不重复本候选。

## 改动与编译记录

- 最早16-head及后续8-head版只缩小PV incoming Tile，仍保留32×256整状态并assemble。
  v1/v3/v5均因196608B Vec超过188416B失败；v4仅准备阶段受组合保护拦截。
  失败代码及日志保留，没有扩大硬件容量或改编译器。
- v5 MemoryReuse转储显示四个32768B状态基址和一个65536B共享arena。
  缩小incoming不会自动消除整状态双份分配。证据：`qk_merge_heads_v5_memory_allocation.json`。
- v6把每侧循环状态本身分成两个16×256 FP32值，完整展开初始化、分支/循环yield和reset；
  最终16-head输出直接读取相应状态。每元素old*alpha+incoming*beta及m/l、五块合并次序不变。
  保持所有GM写表达式、scalar metadata读、显式同步和其他函数AST。
- `qk_merge_heads_prepare_v6` / `qk_merge_heads_cpu_v6`完整codegen通过。
  AIV分配65536+32768+32768+32768+32=163872B。
  四个状态块地址65536/81920/98304/114688，每块16384B；初始化/更新/reset地址一致、互不重叠。
  `qk_merge_heads_codegen_audit.py`对CPU及设备实际artifact均通过，设备首次执行前检查。

## 设备结果

任务`task_20261017_103218_149245329068`，exit=0。
正式第4层权重，合成128K历史，固定输入图重放；两侧均PTO，没有Native forward。

| 项目 | 原动态PTO | 单SPMD状态分块 |
|---|---:|---:|
| L1窗口1（μs） | 910.98 | 1033.44 |
| L1窗口2（μs） | 914.34 | 1351.48 |
| L1窗口3（μs） | 912.20 | 1029.72 |
| L1中位（μs） | 912.20 | 1033.44 |
| 图Event中位（μs） | 959.95 | 1118.08 |

输出、Top-K及六份cache/state的eager、图重放、最终结果全部零差异；30代通知、所有未写区域、
metadata与workspace保护通过。每次恢复fixture，不代表连续decode轨迹。
三个窗口全部原样保留，没有删除较慢样本。

## 已核对的生成代码影响

与现选产物比较全部AIC/AIV阶段的静态指令位点，只有QK AIV第30阶段这些计数改变：
PIPE_ALL 9→8、set/wait各80→82、TLOAD 22→24、row-expand-mul 4→8。
其他阶段这些计数没有变化；这不是所有指令或地址完全相同的证明，也不是回退的定量归因。
PV内部局部V屏障9→13、set/wait各6→10，完整流水并未因为UB缩小就自然提速。
没有为负结果追加探针或复测。

## 文件

- 实现：`qk_merge_heads.py`；旧8-head源码：`source_snapshots/qk_merge_heads_8_full_state_v5.py`。
- 生成代码审计：`qk_merge_heads_codegen_audit_v6.json`及同名merge.cpp.txt。
- 全阶段静态比较：`qk_merge_heads_full_codegen_comparison_v6.json`。
- 完整结果/泳道：`b16_row_deps_qk_merge_state_v6/`；摘要：`qk_merge_heads_result_v6.json`。
- 大产物只保留本地，过程镜像中的LOCAL_ARTIFACTS.json记录路径，不做hash。

## 修正等待配置后的v7

`task_20261017_110518_211373122343`完成exit0，全门槛PASS，实际AIC/AIV 100tick等待门禁通过。
完整L1三窗820.58/812.90/820.24μs，中位820.24；同轮动态909.56。
没有证明优于现选807.16/807.50，未采用；没有同进程现选配对，不给出精确回退比例。
机器记录qk_merge_heads_result_v7.json。上文v6数值保留，但其性能归因已由编译上下文更正撤回。

<!-- SOURCE_END history-document-077 -->

---

<a id="history-document-078"></a>
### 历史原文 078：QK_QUERY_METADATA_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_QUERY_METADATA_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_QUERY_METADATA_EXPERIMENT.md)；原文 1,925 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-078 -->
# 2026-10-17：QK每query地址准备候选

候选通过完整设备检查，但未显示相对现选的明确收益，不采用、不扩测七档。

实现位于`qk_query_metadata.py`，开关`--qk-query-metadata`默认为关闭。
在当前query的tick%5==0时批量计算512个压缩KV物理行，四个压缩块复用。
原始索引与物理行各占一个1×512 INT32 Tile，共4096B，由循环状态跨tick携带。
页表每query最多4次加载减至1次；每AIV处理的地址总数则从256增到512，
这是明确的额外工作，不能只按加载次数推断性能。

保留原负索引判断、KV逐行gather、所有数学、三槽交接和FFTS。提前准备不依赖
第一个压缩块有效；尾部drain tick不加载越界query。原plan写满512项，把不合法
候选变为-1。CPU检查B1..40/S6共2519040个lane/行映射，所有query恰好覆盖一次，
每次使用前已更新元数据。该检查不代替设备数值验证。

prepare/cpu_v1通过；实际驻留UB地址135168与137216，末端139264，小于188416。
任务`task_20261017_070649_214406126149`终态exit0，正式第4层权重、128K/B16/S6，
输出/TopK/六cache-state、eager/图、30代信号及未写区域保护检查均通过。
只固定fixture恢复重放，不能称为连续decode状态轨迹。未运行Native forward。

| 路径 | L1三窗 μs | 中位 μs |
| --- | --- | ---: |
| 单SPMD候选 | 811.82 / 815.08 / 805.18 | 811.82 |
| 同轮原动态PTO | 904.92 / 906.44 / 893.10 | 904.92 |

计时为完整L1 incore首尾，未插桩。现选历史807.16、同进程复核817.88均来自其他运行，
本轮没有现选同进程A/B，不把几微秒历史差值作为精确收益或回退。
结果`qk_query_metadata_result.json`；完整记录`b16_row_deps_qk_query_metadata_v1/`；
地址合同与物理审计分别为`qk_query_metadata_contract.json`、`qk_query_metadata_codegen_audit.json`。

<!-- SOURCE_END history-document-078 -->

---

<a id="history-document-079"></a>
### 历史原文 079：QK_ROLE_SPLIT_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_ROLE_SPLIT_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_ROLE_SPLIT_EXPERIMENT.md)；原文 9,499 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-079 -->
# QK 双 AIV 按职责分工（2026-10-17）

T96/128K两版均完整设备精度通过：原8-head分块L1中位885.08μs；
扩大softmax/PV分块并独立限制发布分块后为830.46μs。两版均未优于现选约807μs，
不采用、不扩档，不改变七档选择或生产实现。阶段时戳诊断单独记录，不参加性能验收。

## 改变的工作分配

原来每条AIV准备64行KV、处理32个head。候选中AIV0准备全部128行KV，AIV1处理全部
64个head的softmax、PV归并、归一化、RoPE和packed输出，尝试重叠搬运与数学。
这会减少并行搬运核数，也增加单条AIV的工作量；是否有净收益必须测完整CSA，不能由分工推断。

Cube QK/PV数学、五块归约次序、三槽L1与GM、pre-launch=2、正式权重和全部外部入参保持。
原四种FFTS事件保留两条AIV参与：无相应数学工作的AIV仍消费Score/PV通知，发送KV/Prob通知。
采用先消费上一块Score再通知本块KV的顺序，没有新增任务内Scalar GM信号或workspace。

两条AIV的事件序列与原提前KV策略一致；角色分工改变实际数据生产/消费方。
`qk_role_split_protocol.py`独立检查3328个调度、471867动作，包含两query全部1024种
有效块mask、256种四query随机mask以及2/3代复用。检查GM KV、L1 KV、score、prob、PV
五份槽的复用和每代事件排空；模型事件credit峰值3。它不证明硬件credit容量或DMA/缓存可见性。

## 数学状态与内存

AIV1的64×512 FP32状态共131072字节，拆为16个8×256循环值，逐块原位更新。
softmax与最终发布每块8个head。每个head仍按128列softmax、五块PV归并、FP32归一化后
rint转BF16；设计保留运算次序，设备精度不能仅由源码分析宣称通过。

首轮`qk_role_split_prepare_v1/`通过，`qk_role_split_cpu_v1/`在ConvertToSSA拒绝
分支外`kvbuf`引用，没有进入内存规划。失败源码快照`qk_role_split_source_v1.py.txt`保留。
v2把raw/cmp两条路径的store放回各自分支，以独立`kv_done`接收压缩行循环结果。
`qk_role_split_cpu_v2/`完整编译通过，未修改编译器。

实际IR地址检查：16个状态起点16384至139264，每份8192字节、互不重叠；初始化、回边和
有效块状态地址相同，16次PV加法逐个写回原槽。QK显式Vec Tile的shape地址范围到163328，
低于188416字节；该范围不是完整allocator报告。
C++有8个row-max、8个row-sum、8个最终row-divide；两分支合计4个wait、4个notify，
5个PIPE_ALL。实际提交顺序由`qk_role_split_codegen_audit.py`检查，不以静态计数当性能结论。

## 必须同时改变的输出依赖

旧O-A按head半区分别等AIV0/AIV1；现在packed的全部8组由AIV1写出，必须全部等待对应
QK组的AIV1。否则可能在AIV0完成KV搬运后提前读未写完的输出。

`row_dependencies.py`增加默认关闭的角色所有权参数。新分支检查真实写入代码确实在AIV1，
枚举96个query的768行packed结果，要求唯一写入、八组O-A消费范围全部覆盖。
旧策略的六项源合同和行覆盖证明仍通过。其他依赖、原物理核先后和信号发布维护保持。

`qk_role_split_prepare_v3/`使用新依赖证明生成完整变体，其角色分支AST与已编译v2相同。
不重复编译相同数学体；设备编译后再次检查地址/事件顺序，以及真实100tick等待实现。

## 设备任务

`task_20261017_115641_29670009014`，结果目录`b16_row_deps_qk_roles_v1/`。
仅128K/B16/S6，原动态PTO对照、正式权重、完整exact/图/30代/保护区门槛之后才测三窗L1。
任务exit0。eager/graph/final输出、TopK和六cache/state全部零差异；30代进度与输入、
写入范围、工作区保护区PASS。实际设备artifact通过状态地址、FFTS顺序和100tick等待门禁。
完整无插桩L1：887.72 / 885.08 / 870.46μs，中位885.08；同轮动态PTO中位901.30μs。
与历史现选807.16/807.50相比没有采用理由，未做本轮现选配对，不精确分摊跨轮差值。
摘要`qk_role_split_result_v1.json`。fixture逐次恢复，不是连续decode轨迹。

观察任务时有一次40秒等待工具超时，后续原任务正常exit0；没有重启、终止或设备超时错误。
只读核对本地PTO-ISA集群映射文档说明mode2的两AIV参与及4位计数；没有将事件模型当硬件证明。

为定位为何分工没有收益，提交一次阶段诊断`task_20261017_120238_306765112993`，
目录`b16_row_deps_qk_roles_stage_v1/`。仅一次L1窗口、warmup0/iters1；其结果含插桩且
warmup与历史不同，不能替代上方正式三窗结果或精确计算跨轮回退幅度。

诊断任务exit0，全部eager/graph/final exact、4代进度、保护区PASS。时戳见
`qk_role_split_stage_comparison_v1.json`；以下是阶段的“数学、内部等待与写侧维护”，不是pipe独占忙时：

| 阶段 | 历史现选中位（μs） | 分工候选中位（μs） |
|---|---:|---:|
| Indexer AIC | 211.03 | 212.99 |
| QK/PV AIC | 150.93 | 206.39 |
| QK/PV AIV0 | 157.90 | 207.90 |
| QK/PV AIV1 | 158.12 | 222.80 |

本次QK AIC最后结束701.30μs、AIV0为703.50μs、AIV1为718.38μs，O-A最早720.94μs。
候选的AIV1成为组内尾部，且Cube自身也没有因分工缩短。现有证据把主要退步定位于QK/PV，
不能归咎于外层等待策略或Indexer。尚未测出KV、softmax、PV、输出各自的独占成本，
不能精确地把多出的时间分给某个操作。

源码层面，单条AIV现在负责双倍head，且为控制UB改用8-head小块，softmax静态位点由
每AIV1组32-head变为AIV1上8组8-head，最终发布也变细。额外指令提交和工作集中是
下一步CPU分析的候选原因，不能仅凭调用数认定其时间占比。
不重复此版设备测试；若继续该方向，先验证更大head Tile的实际UB与别名，再决定是否值得上卡。


### 2026-10-17T12:22:59：角色分工的head Tile分离，CPU确认发布峰值超UB

在现有8-head分工版本上，将PV状态块扩为16×256、softmax扩为32×128，最初发布也为16head。
qk_role_tiles_h16_s32_prepare_v1准备PASS，cpu_v1在AllocateMemoryAddr拒绝Vec194048B>188416B；无设备执行。
从37_after_MemoryReuse逐项核对，AIV各allocation求和正好194048B。状态仍128KiB，PV的两份16KiB临时区和
最终32KiB FP32发布结果复用；16KiB BF16输出与两份4KiB RoPE临时量、4KiB swap索引同时存在，
因此单独缩softmax并不能消除发布峰值。诊断qk_role_tiles_ub_diagnosis_v1.json保留全部allocation。

将publication_tile从state head_tile解耦：仅发布回到8head，从对应16head状态slice连续行，
仍为32head softmax、16head PV；原运算次序、Cube、GM接口和事件保持。旧8/8及16/32同发布
版本AST回归一致，旧8/8/8编译产物扩展审计PASS。新增快照qk_role_split_source_h16_s32_p16.py.txt。
qk_role_tiles_h16_s32_p8_prepare_v1 PASS，cpu_v1编译中；当前尚无此变体设备结果，不修改七档选择。


### 2026-10-17T12:24:22：16/32/8候选完整CPU通过

qk_role_tiles_h16_s32_p8_cpu_v1 full_codegen_passed=true。MemoryReuse的全部AIV Vec allocation合计
167552B，低于188416B；与h16/s32/p16的194048B相比减少26496B，实际全局复用结果不同，
不能仅按局部Tile字节数估算。8个16×256 FP32状态起点0..114688，互不重叠且原位更新；
incoming不覆盖状态，8次旧值乘法/更新逐槽匹配。softmax2个32head rowmax/rowsum，发布8个8head rowdiv，
仍4wait/4notify/5PIPE_ALL，提交顺序PASS。全部记录qk_role_tiles_h16_s32_p8_codegen_audit_v1.json。

新增默认关闭--qk-role-wide-heads，限定与--qk-role-split组合，明确使用(状态16,softmax32,发布8)；
设备执行前检查实际编译Tile与请求一致。行依赖继续检查AIV1独占全部8组packed输出。
CPU集成准备中，不重复已经通过的数学编译；尚无设备性能，不修改现选。


2026-10-17T12:25:55：集成行依赖后全部数学体AST与CPU通过版本一致；提交task_20261017_122441_342567127888，b16_row_deps_qk_roles_wide_v1。仅B16/128K/S6，全门槛+三窗无插桩L1。任务执行中冻结driver/helper，不改现选。


### 2026-10-17T12:28:27：32head softmax/16head PV/8head发布全门槛PASS，未采用

task_20261017_122441_342567127888 exit0，b16_row_deps_qk_roles_wide_v1。
eager/graph/final的输出、TopK、六cache-state均零差异，30代进度及全部保护区PASS。
实际Tile(16,32,8)、状态/事件及100tick等待源码门禁PASS。无插桩完整L1
830.46/827.18/832.04，中位830.46；同轮动态899.68/911.08/912.48，中位911.08。
比旧8/8/8分工的历史885.08明显降低，但仍未优于现选807.16/807.50；不同轮次无现选配对，
不精确归因差额。未采用、不扩档；暂关闭仅靠分工/增大head Tile的优化方向。
见qk_role_tiles_result_v1.json。所有设备任务终态，700目标未达。

独立新增fixed_core_local_schedule.py，只在CPU从现选顺序进行固定物理分核的相邻交换，
与前轮三种从头贪心分配不同。数学边与216条FFTS组复用边固定，MIX按共同开始建模，
防止误用成员独立开始的时间；原模型771.53，共同开始模型773.41。120秒局部搜索中。
未生成二进制、未接入生产/测试driver，运行中的设备helper没有修改。

<!-- SOURCE_END history-document-079 -->

---

<a id="history-document-080"></a>
### 历史原文 080：QK_SYNC_FALLBACK_ANALYSIS.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QK_SYNC_FALLBACK_ANALYSIS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QK_SYNC_FALLBACK_ANALYSIS.md)；原文 2,582 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-080 -->
# QK/PV AIV内PIPE_ALL来源核对（2026-10-17）

使用当前815.84μs的128K/B16/S6配置生成的原PTO IR，只运行已安装
PTOAS 0.66的同步诊断，没有修改编译器、ISA或依赖仓。

## 结论

原PyPTO IR的QK tick内没有这四个显式PIPE_ALL。实际PTOAS日志对照
RemoveRedundantSync之后、EventId Allocation之后的同一依赖编号，确认
部分普通set/wait依赖在事件分配时回退为全pipe屏障。

|依赖编号|分配前的方向|落点|分配后|
|---|---|---|---|
|2708|V→MTE2|KV段后/softmax段前|PIPE_ALL|
|2709|V→MTE2|softmax分支的Score加载|PIPE_ALL|
|2712|V→MTE2|PV合并分支入口|PIPE_ALL|
|2687、2688、2689|MTE3→V|PV合并分支入口|PIPE_ALL，邻近屏障会合并|
|2714|MTE3→V|最终输出的concat|PIPE_ALL|

编号只对应本次保存的IR，不是稳定API。前四个中的两个屏障位于此前
四段SYS_CNT计时区间外；另外两个位于softmax与最终输出内部。
不能把四段计时差额全部归因于屏障，也不能把静态屏障数当运行次数或耗时。
这些同步包含真实的片上缓冲读写依赖，不能直接删除。

将valid_block_mask预读到本核UB的候选已经通过CPU生成，但屏障仍是4处；
因此“GM标量读直接导致这四个屏障”的猜测没有成立。mask候选没有上卡测性能。

## 可追溯证据

- `qk_sync_diagnostic_v4/sync.log`：完整已安装PTOAS的诊断与pass后IR。
- `qk_sync_diagnostic_v4/qk_dependency_nodes.json`：对应生产者/消费者节点。
- `qk_sync_diagnostic_v4/qk_fallbacks.json`：同一编号的前后记录。
- `qk_mask_prepare_v2/codegen_sync_comparison.json`：mask缓存前后代码计数。
- `qk_phase_probe_cpu_v1/uncovered_sync.txt`：分段计时外的同步位置。

可复现命令（D为本实验worktree，N为本目录，输出另选空目录）：

```bash
/data/pyptouser/qinchuanyu/pto-eager/.cache/dsv4-toolchain/ptoas-0.66/bin/ptoas \
  "$N/b16_cached_qk_indices_v1/build_output/entry_57ys1c05/ptoas/csa_static_group.pto" \
  --enable-insert-sync --pto-level=level3 --pto-insert-sync-debug=2 \
  --mlir-print-ir-after=pto-insert-sync -o "$OUT/group.cpp" \
  > "$OUT/stdout.log" 2> "$OUT/sync.log"
```

v1/v2的打印参数失败、v3调试日志过大而主动终止，均记在ATTEMPTS.md；
最终结论只来自完整结束的v4。大日志保留本地并记路径，不做hash。

MGATHER和HC 2行分片随后各通过整层exact门槛，但未优于815.84μs，
默认配置不变。详见`qk_mgather_result.json`、`hc_rows2_result.json`。

<!-- SOURCE_END history-document-080 -->

---

<a id="history-document-081"></a>
### 历史原文 081：QPROJ_L1_COLUMNS_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QPROJ_L1_COLUMNS_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QPROJ_L1_COLUMNS_EXPERIMENT.md)；原文 3,105 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-081 -->
# 主Q双N128列块驻留L1

2026-10-17，128K/B16/S6，`qproj_l1_columns.py`；默认关闭。

## 与既有预取候选的区别

原主Q每个worker处理一组N256列，K1024分四次K256累加。
此前显式L0预取候选用N256/K128，完整L1为830.62μs，未采用。
本候选保留原20worker和每worker的原列归属，把每组N256拆成两个N128，
各自仍按0/256/512/768的K顺序累加。两个累加链交错，
第一列的矩阵计算提交后，准备下一K块，再提交第二列的计算。
QR每worker入口加载一次；两份1024×128权重每列组各加载一次到L1。

INT8乘积和INT32累加不溢出，绝对值上界16777216；原输出Tensor、权重NZ布局、
20worker以及所有32768列的唯一归属保持。没有Host读取设备数据，也没有额外GM布局转换。
这里只先验证T96，其他档位需要重新确定M及内存容量，不直接套用固定M96。

## CPU分配证据

`qproj_l1_columns_prepare_v1`与`qproj_l1_columns_cpu_v1`通过。
实际Left在0/24576，Right在0/32768，两个Acc在0/49152，成对不重叠。
L1两份权重在65536/196608，QR在327680，最大端点425984字节。
每个原N256列组为两次matmul初始化及六次matmul_acc。
`qproj_l1_columns_codegen_audit.json`记录实际地址和操作，
`qproj_l1_columns_generated_stage.cpp.txt`保存整个stage5。
源级存活容量与实际地址均不代替流水重叠或性能测量。

## 首轮完整CSA

任务`task_20261017_083715_36883533451` exit0；输出/Top-K/六份cache/state、
eager/graph、30代信号、保护区全PASS。正式权重，未执行Native forward。
完整无插桩L1为809.22/806.46/797.56μs，中位806.46μs；
同轮原动态PTO中位903.86μs。Event中位单SPMD912.24、动态951.82μs，分别报告。

现选历史首轮807.16μs、同进程复核817.88μs。首轮数字接近且各窗有变化，
不能拿不同运行的最小值或中位数直接宣布收益。

## 同进程对照完成：未采用

`--paired-qproj-spmd-only`复用配对测试助手，对照端明确启用现选row_dependencies=True、
QK地址缓存和100tick退避；新候选只有主Q不同。两边相同根Tensor地址、独立工作区和信号，
每侧2次预热、3窗L1，AB/BA交替；不重复Event或动态metadata验证。
任务`task_20261017_084154_377975115763`，目录`b16_qproj_l1_columns_paired_v1`。
该任务exit0，输出/Top-K/六状态及双方工作区、7代信号全PASS。
对照端生成源码AST与现选完全一致，见`qproj_paired_reference_source_audit.json`。

| 版本 | 三窗L1（μs） | 中位数（μs） |
|---|---|---:|
| 现选单SPMD | 805.28 / 808.92 / 810.24 | 808.92 |
| 主Q双列候选 | 806.02 / 804.06 / 810.32 | 806.02 |

中位差-2.90μs（-0.36%），均值差-1.35μs，范围明显重叠，
未证明足以替换现选的稳定收益。未采用、不扩七档、不重复该比较。
`selected_shape_policies.json`保持原选择。结果`qproj_l1_columns_paired_result.json`。

所有重放在调用前恢复fixture，不能据此声称连续decode轨迹已通过。

<!-- SOURCE_END history-document-081 -->

---

<a id="history-document-082"></a>
### 历史原文 082：QUERY_GROUP_SCHEDULE_FEASIBILITY.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/QUERY_GROUP_SCHEDULE_FEASIBILITY.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/QUERY_GROUP_SCHEDULE_FEASIBILITY.md)；原文 3,449 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-082 -->
# 两组 query 完整流水的 CPU 评估（2026-10-17）

当前六个排程原型都没有显示收益，暂不生成对应的设备算子，不提交 NPU。
这不是所有更细流水都不可能优化的证明。

前一轮的所有权模型证明：保持每个 worker 的叶子集合，先完成前8个请求，
可以让其数据在11个score块后就绪；但原整阶段通知仍会等满22块。
因此本轮模型同时拆分 Indexer、merge、plan、QK、O-A、quant、O-B 和 HCpost，
每组48个query。没有只改Indexer后假设下游自然提前。

`query_group_schedule_model.py` 读取现选 B16 调度、已证明的行依赖和同形动态L1成本：

- 产生1313成员、1121调度单元、43953条数学边。
- Indexer与QK保持原logical block到物理核组映射。
- 同一MIX组下一阶段要等前一阶段三成员都结束，保护FFTS复用。
- 同一logical worker的GM交接区在两半之间不得重叠使用。
- 按实际query行重新核对Indexer→merge→plan→QK及投影消费者集合。
- plan原12个有效8行成员分成每半6个；HCpost的48个2行成员分成每半24个。
- 保留现有前端和其他未拆分任务；未假设主Q投影的96行可以提前分批完成。

这里的时间只是成本模型。拆分计算按原成本的一半估算，每个新成员仍保留1μs。
O-A/O-B也先按理想的一半估计，没有把新增权重读开销作为已测值。
因此模型不表示实际性能，不保证任何数值、内存分配或设备同步正确性。

| 排法 | CPU估计总长 μs | 同模型现选 μs |
|---|---:|---:|
| 优先前半组链路 | 855.325 | 771.530 |
| 最早可启动 | 780.470 | 771.530 |
| 剩余关键路径优先 | 813.585 | 771.530 |
| 最早可启动，允许重排MIX物理组 | 780.470 | 771.530 |
| 前缀先排、前半链路优先，允许重排MIX物理组 | 841.340 | 771.530 |
| 仅保护主Q两阶段、再前半优先，允许重排MIX物理组 | 862.105 | 771.530 |

前半优先时Indexer抢先占用资源，主Q投影延后，第一半QK直到384.31才开始；
另两种排法中，第二半QK仍把第一半O-A挤到后面，未形成期望的全链路交错。
这些是对应模型的排队现象，不能替代真实硬件归因。

结果分别保存在 `query_group_schedule_ideal_phase_first.json`、
`query_group_schedule_ideal_earliest.json`、`query_group_schedule_ideal_critical.json`，
选择结论在 `query_group_schedule_assessment.json`。

```bash
python query_group_schedule_model.py --policy phase0_first --projection-fraction 0.5 --output query_group_schedule_ideal_phase_first.json
python query_group_schedule_model.py --policy earliest --projection-fraction 0.5 --output query_group_schedule_ideal_earliest.json
python query_group_schedule_model.py --policy critical --projection-fraction 0.5 --output query_group_schedule_ideal_critical.json
```

没有继续扫描更悲观的投影成本，也没有以“前半数据提前一半”宣称整层会快一半。
如果将来继续此方向，需要能同时处理主Q生产时序、MIX资源复用及输出投影权重开销的新排程依据。

补查的两种排法分别在`query_group_schedule_flexible_earliest.json`和`query_group_schedule_prefix_then_phase0.json`。允许每半不同物理组时仍检查每个MIX调用24组一一对应、每个logical worker交接区的跨半区间不重叠。结果未改善，不据此继续上卡。

<!-- SOURCE_END history-document-082 -->

---

<a id="history-document-083"></a>
### 历史原文 083：RAMP_WAIT_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/RAMP_WAIT_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/RAMP_WAIT_EXPERIMENT.md)；原文 1,882 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-083 -->
# 2026-10-17：依赖等待从短到长退避

这是单个新轮询策略，未重复以前的均匀0/50/100/200/400tick扫描。
开关`--ramp-wait`默认关闭，必须与`--cached-wait --wait-backoff-ticks 100`使用。
本候选先在128K/B16现选row deps/QK地址缓存版本独立验证，不叠加短尾等数学候选。

`signal_polling.py`每个新依赖第一次读取仍立即执行；未就绪后的间隔依次为
4、8、16、32、64、100tick，之后保持100tick。当前SYS_CNT一tick为0.02μs。
假设是对很快就绪的依赖缩短唤醒等待，对持续未就绪的依赖仍控制轮询频率。
它也可能增加读请求，收益必须以完整CSA衡量。

每次读取前仍DCCI后volatile读；GE比较、单写者单调进度、周期PIPE_ALL和spin上限不变。
TNOTIFY写侧DCCI+DSB以及调用者业务payload维护完全保持。没有假设Scalar缓存一致性，
没有用普通store替代信号发布，没有删同步或改数学。仅修改实验目录，不改ISA或依赖仓。

`wait_ramp_cpu_v1/audit.json`验证默认固定间隔生成的完整C++与现选逐字相同；
递增版在等待helper之外的C++与固定版逐字相同。该核对不是设备正确性或性能证明。

任务`task_20261017_074820_28363061302`，输出`b16_row_deps_ramp_wait_v1/`。
先完整exact/Top-K/六状态、图、30代信号和保护区，再三窗完整无插桩L1。
正式W8A8权重，不运行Native forward。任务已完成exit0，全部检查PASS，未修改现选策略。

三窗单SPMD 833.74/829.94/826.22μs，中位829.94；同轮原动态910.94μs。
没有明确收益，不采用，不继续扫描更多间隔或扩七档。未做现选同进程A/B，
不把与历史807.16μs的差额当精确回退。证据`ramp_wait_result.json`及任务目录。
固定fixture恢复、30代信号验证不等于连续decode轨迹。

<!-- SOURCE_END history-document-083 -->

---

<a id="history-document-084"></a>
### 历史原文 084：RANGE_POLICY.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/RANGE_POLICY.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/RANGE_POLICY.md)；原文 4,860 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-084 -->
# CSA 工作量区间入口（实现中）

当前目标是七个验收点各自相对同轮动态分 task PTO 至少降低 30%，见
[GOAL_AND_CONSTRAINTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md)。以下范围是生成器当前可表达的
数学合同；没有把 CPU 检查写成设备验证，也没有宣称这些阈值已经最优。

## 入口及长度定义

`range_policy.build_range_variant(descriptors, max_seq_len=..., output_root=...)`
按输入的 shape/stride/dtype 和调用方已有 Host 长度上界生成变体。
这个函数负责源码与布局；cached100 是设备编译上下文中的策略，不能把单独生成/编译的源码
当作已启用 cached100。`device_compare.py --range-policy` 会在编译前安装现有
`signal_polling.bypass_signal_wait(..., cached=True, backoff_ticks=100)` 上下文，并检查实际CCEC源码。
测试入口增加 `run_device_compare.sh B HISTORY OUTPUT --range-policy`，不再要求手写
排程、QK 索引缓存或行依赖开关。仍可使用原研究开关单独调优，但不能与范围入口混用。

H 是本轮之前的历史长度，S 是每请求本轮 query 数，L 是包含本轮 query 的长度。
当前 fixture 中 L=H+S；原编排选择使用 C=floor(L/4)，故 H=8192 不是精确分界。
当前核体仍限 S=6、T=B×6≤384、6≤L≤131200；实际页表、cache 和工作区也必须足够。
测试 fixture 另有限制 B≤40。其他 S、padding/dummy 和混合 query 数不能视为已支持。

| 本轮 L 区间 | 原 Indexer 数学分支 | query 分组 |
| --- | --- | --- |
| 6～8191 | Vector leaf | 单 query |
| 8192～32771 | 中等历史 Cube | 比较 pair 与 request 的核波次数 |
| 32772～131200 | 长历史 Cube | B<4 为 pair，否则 request |

中等历史的 request 范围并非简单 B≥24，而是 B=24、B33～48、B57～64。
B1～23、B25～32、B49～56 走 pair。公式保留在 select_policy 中，与原 IR 对照。
原 QR/KV 数学在 T=128 切换；输出投影包含 T≤32、32<T<96、T=96、T>96 四种合同。
T=96 来自原 CSA 本身，不是复制 HC_pre 的专用规则。

## 性能选择与证据边界

初始范围策略保持已有七点选择：长历史且 T≤96 用 earliest_ready，较大 T 用
critical_path；中等历史 request 用 earliest_ready、pair 用 critical_path。
长历史 32<T≤96 启用已核对的 QK Vector 索引变换。现有行依赖只在长历史 T=96
的输出所有权合同启用，扩大范围须重新证明。全区间沿用 cached100 通知等待方式。

这些是可测试的初始范围策略，不代表邻近点已经得到相同收益。七点的成本资料只在
B/H/S 和所有 root shape 完全匹配时使用；没有同规格资料的形状仍可生成，成本明确
标为参考模型，不能作为实测性能。长度、位置、slot、页表以及有效循环范围仍由核体
读取实际 device Tensor，CPU 选档不新增每步 item/cpu/tolist。

## 缺少整图参考的处理

已有同构参考的分支保留原生成路径。没有参考时，范围入口从本轮真实绑定、逻辑接口
和后期 IR 的读写效应、显式 deps、dummy、manual scope 构造任务图，再重新分核。
不套用相邻 B 的等待表，不构造整图串行兜底。生成后仍检查逻辑分片覆盖、MIX 配对、
依赖偏序和信号布局；CPU 通过后才进入 lowering 和设备精度验证。

相关代码在 `anchor_gap_design_v1/source_schedule.py`，审计和验证证据保留在同目录。
当前该路径仍属于实验生成器；生产入口接入、各策略代表点/边界的完整真机验证尚未完成。

## 新性能门槛

`performance_acceptance.py` 使用同轮至少三窗完整 L1 中位数检查 `candidate≤baseline×0.70`。
功能通过与性能达标分开记录：`passed` 是本次功能检查，`performance_target.passed`
才是本规格的性能门槛。单规格性能通过也不能代表七档及区间动态要求全部完成。
历史七档差距保存在 `target_30pct_historical_gap.json`，原始 L1 起止重新计算一致，
没有为此重跑测试；最终达标仍须正式复核。

## 当前真机代表点证据

B12/H65536 与 B17/H65536 均完整 exact/图重放/30代信号/保护区PASS。B17确实走源码DAG路径。
L1同轮三窗中位分别为690.70/714.66μs、863.08/893.52μs（单SPMD/动态），仅降低3.35%/3.41%，
均未达30%。见 `range_entry_device_results_v1.json`。这两点不证明整个区间，当前仍需边界、
动态query及后续数学策略验证。

B17另通过H65536→65535→32766→65536同地址图重放；L32772覆盖长历史数学分支的下边界，
每侧只捕获一次且没有新增编译单元。该证据在`range_b17_h65536_metadata_v1/result.json`，
包含页表交换和原址metadata更新；不代表连续decode状态轨迹。

<!-- SOURCE_END history-document-084 -->

---

<a id="history-document-085"></a>
### 历史原文 085：ROPE_GATHER_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/ROPE_GATHER_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/ROPE_GATHER_EXPERIMENT.md)；原文 3,522 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-085 -->
# RoPE相邻交换与逐行gather合并实验

2026-10-17。只在本实验目录修改源码，不改生产入口、冻结静态原型、PyPTO、
Simpler或PTOAS。允许各batch/历史档位分别编译；仍以完整CSA无插桩L1首尾为性能口径。

## 无gather相邻交换CPU探针

背景见`PTOAS_GATHER_SYNC_SCOPE_DIAGNOSIS.md`：当前PTOAS在整个函数含gather/scatter时，
保守地跳过冗余同步消除。本实验不会删除这项安全检查。

- `rope_pair_swap_probe_v1.py`把512个FP32元素reshape为256×2，再extract奇偶列并concat。
  PTOAS拒绝：8字节行宽不满足A3的32字节对齐。未执行设备代码。
- `rope_pair_swap_probe_v2.py`改为64×8，转置后交换四对对齐行，再转回。
  `rope_pair_swap_probe_cpu_v2/result.json`记录完整CPU代码生成通过。
  这需要两次转置及八行复制；不把编译通过当成更快，暂未集成或上卡。
  CPU生成代码没有TGATHER，但这不能证明完整CSA消除gather后能达到700μs。
- 现选QK页表缓存还有真正的非连续gather。只替换RoPE不能使整个AIV函数无gather；
  不会偷偷关闭已验证的QK地址缓存，也不把减少源操作数当作时间收益。

## 三处RoPE逐行gather合并

`rope_flat_gather.py`仅替换以下三个循环：

| 数学函数 | 原循环行数 | 新gather次数 |
| --- | ---: | ---: |
| kv_rms_norm_rope | 32 | 1 |
| rmsnorm_rope_cache_write | 16 | 1 |
| rmsnorm_rope | 16 | 1 |

原第r行的列选择c转换为连续Tile里的`r*64+c`。c仍读取原输入Tile，
没有将任意索引偷偷假定为相邻交换，也没有新增Host读或GM工作区。
整数位移生成行基址，浮点数学和运算顺序不变。原单行tail不改。

AST检查把三个替换块还原后，与原完整catalog完全一致；穷举262144个合法行/列/选择组合，
确认地址仍对应同一行的原选择列。GM读写、FFTS、跨核等待与发布保持原样。
这不替代设备精度、边界和图重放检查。

CPU准备`rope_flat_gather_prepare_v1`通过。首个编译命令误用了`static/`路径，
`rope_flat_gather_cpu_v1`只记录FileNotFoundError，未进入编译；正确目录是`variants/`。
修正后`rope_flat_gather_cpu_v2`在编译。集成开关`--rope-flat-gather`默认关闭，
先与现选QK Vector索引、行依赖及100tick等待组合验证128K/B16/S6。
截至此记录尚无新性能结果，七档选择保持不变。


## 完整CPU编译与设备提交

CPU v2完整PASS；实际1024/1024/2048元素的三个gather索引链、FP32/INT32 Tile与同时活跃UB区间检查通过。集成prepare_v2与已编译源码全部AST相同。详见rope_flat_gather_codegen_audit_cpu_v2.json和rope_flat_gather_integration_audit.json。
任务`task_20261017_131052_534942170`已提交，输出`b16_row_deps_rope_flat_v1`；128K/B16/S6，完整精度/图/30代/保护区后才比较三窗L1。等待设备结果，不预记PASS。


## 完整设备结果：未采用

task_20261017_131052_534942170 exit0；eager/graph/final输出、TopK、六cache-state均零差异，30代与保护区全部PASS。实际三处Tile/索引链/UB与100tick等待门禁PASS。完整L1 819.58/815.10/810.50，中位815.10；同轮动态892.30/923.18/920.58，中位920.58。没有优于现选证据，未采用、不扩档；没有本轮现选配对，不精确归因跨轮差额。源码和原始结果保留，不重复该候选。
详见`rope_flat_gather_result_v1.json`及`b16_row_deps_rope_flat_v1/`。

<!-- SOURCE_END history-document-085 -->

---

<a id="history-document-086"></a>
### 历史原文 086：ROW_DEPENDENCY_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/ROW_DEPENDENCY_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/ROW_DEPENDENCY_EXPERIMENT.md)；原文 5,696 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-086 -->
# B16/S6 单 SPMD：按 query 与 head 细化成员依赖

2026-10-17。实现位于 `row_dependencies.py`，仅允许长历史 B16/S6、独立临时区、
原数学体与原核内顺序，可组合此前已选的 QK 地址缓存。其他 batch 不直接复用此等待表。

## 改了什么

原实现把某个 task 的输出作为整体建立依赖。本候选在核对每个消费者实际读取的行后，
将四类完整 task 依赖改成对应的成员依赖。这确实放宽了原 task 的数学偏序，
不能宣称“原 task DAG 完全不变”；重新验证的是细化后依赖图、核内顺序与信号翻译的一致性。

| 消费者 | 实际需要等待的生产者 | 依据与边界 |
| --- | --- | --- |
| Top-K merge | 3 个相关 Indexer 核组的全部 9 个成员 | 长历史 T96 的 item 按 leaf-major 排列；merge worker w 处理 q=w,w+48；覆盖所有 1～32 个 leaf 的可能生产者 |
| 有效块计划 | 所处理 8 个 query 的 merge 成员 | plan worker w 处理从 w×8 起、步长 128 的行块；T96 时 12～15 号 plan 不读 Top-K |
| QK/PV 核组 | 本组 4 个 query 的 plan 成员 | q=c,c+24,c+48,c+72；其余 Q/KV/RoPE 父任务依赖保留 |
| O-A 投影 | 对应 head 组的 24 个 AIV 成员 | AIV0 写 head 组 0～3，AIV1 写 4～7；packed 行区间不重叠 |

另为每个 QK 成员保留同一物理核组前一 Indexer 的 AIC+AIV0+AIV1 完成依赖，
共 216 条，用于约束 FFTS 资源的阶段复用。这些顺序本来就包含在原图祖先关系中。
每核原有执行序列、GM 独立分配、pipeline fence、DCCI/DSB 与通知发布不改。
“只等数据所属核”不能成为删除资源复用依赖或缓存发布的理由。

## CPU 证据

`row_dependencies_prepare_v1/control_variants/h131072_b16/row_dependencies/` 保存
`audit.json` 与 `member_graph.json`。关键循环与存储下标有 AST 契约检查；
输入 Tensor 的 allocation/offset/shape/stride/dtype/layout 必须吻合，其他共享引用
只允许原只读根输入，不能隐含共享临时区。

- 1536 个 merge worker×leaf_count 组合（每个 worker 两个 query）覆盖 96 个 query、1～32 个 leaf，及全部 plan/QK/packed 行。
- 逐核远端进度等待项：16,729 → 12,183；发布点：730 → 722。
- 原成员边 40,155；细化后必需边 32,805；计划边 17,630 → 13,084。
- 传递约简及发布序号翻译后，逐节点祖先集合与细化后的依赖图相同。
- 生成 `csa_static_aic/aiv/group/entry` 的函数体 AST 与此前候选一致。
  变化集中在等待表与发布表；参数的等待表长度描述相应变化。
- 原协议模型的多代检查通过。它不代替 NPU 内存一致性或数值验证。

`row_dependency_cost_model.json` 使用旧动态 PTO 每成员 L1 成本估算最长依赖链，
748.55 → 748.53 μs。此数不是实测，也不包含新的等待开销；它提示本次改动主要减少
等待表开销，不能指望仅靠删这些依赖就消除约百微秒的计算路径。

## 首轮完整设备结果

任务 `task_20261017_044912_40225339550`，目录 `b16_cached_row_deps_v1/`，exit=0。
正式第 4 层 W8A8 权重，原动态 PTO 与单 SPMD 输出、Top-K、六份 state/cache、
eager/graph/最终结果逐 bit 一致；30 代完成信号、只读区、未写区及工作区保护区通过。
两侧均为 PTO，没有执行 Native forward。

| 完整无插桩 L1 incore 首尾 | 三次采样（μs） | 中位数（μs） |
| --- | --- | --- |
| 原动态 PTO | 888.48 / 905.08 / 909.80 | 905.08 |
| 行依赖单 SPMD | 807.16 / 804.40 / 811.26 | 807.16 |

相对同轮动态 PTO 缩短 10.82%。此前所选单 SPMD 的历史中位数为 815.84 μs，
只相差约 1%，不能把跨轮差值直接当成稳定收益。尚未达到 700 μs。

## 同进程复核与动态长度

`paired_row_dependency_perf.py` 在同进程、同权重及根输入/输出地址下，
分别编译此前所选单 SPMD 和行依赖候选；各采 3 个完整 L1 窗口，交替先后顺序。
两边的工作区和进度信号独立，分别验证输出与信号代际。该辅助检查只由
`--paired-selected-spmd --row-dependencies --metadata-replay-only` 显式启用。

同一任务随后保持候选图、地址和二进制，执行
131072 → 131071 → 65536 → 131072，并交换页表、更新位置与 RoPE。
64K 使 Indexer 从 6 个 leaf 变为 3 个，覆盖本次等待表依赖运行时 leaf 分工的风险。
CPU 编排签名确认这些长度仍使用同一数学策略；16384 跨策略，明确不放入该图。
证据为 `row_dependency_metadata_strategy.json`。

任务 `task_20261017_050033_181823706`，结果目录 `b16_row_deps_paired_metadata_v1/`。
此段先记录验证范围；终态与采用结论以下文补记为准，不预记 PASS。

### 终态与采用结论

任务 exit=0、完整 PASS。旧单 SPMD 三窗为 841.94/843.14/833.20 μs，
新候选为 806.84/817.88/821.56 μs，中位数 841.94→817.88 μs，缩短 2.86%。
新候选最慢窗口也快于旧候选最快窗口；在本轮范围内收益得到确认。
不能把首轮 807.16 和复核 817.88 当成同一轮的不同统计量。

A/B/C1/A_return 四个变体全部通过原动态 eager、原动态图与候选图的输出/Top-K/
六份状态及保护区检查。固定地址、无新增编译单元通过；候选共11代、旧候选7代
进度检查通过。是不同输入快照重放，不是连续decode状态轨迹，也没有覆盖prefix共享。

已更新 `selected_shape_policies.json`：只为128K/B16选择行依赖细化；
其余六档仍使用各自已有候选。本目录为实验实现，未改生产入口。

<!-- SOURCE_END history-document-086 -->

---

<a id="history-document-087"></a>
### 历史原文 087：TILE_LOOP_INPLACE_CPU.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/TILE_LOOP_INPLACE_CPU.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/TILE_LOOP_INPLACE_CPU.md)；原文 1,762 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-087 -->
# 固定L1 Tile循环更新：CPU接口探针

2026-10-17。只编译CPU/PTOAS，不占NPU。目的是为后续N64循环流水判断写法，
不是当前CSA优化、正确性或性能结果，也没有修改依赖仓。

- `tile_inplace_loop_cpu_v1`：小例子遗漏SPMD block index，被前端拒绝。
  按报错在单worker中显式读取block index并用于输出行偏移；不是组件缺陷。
- `tile_inplace_loop_cpu_v2`：`pl.tile.assemble(pool, product, offset)`忽略返回值，
  编译器不能将结果复用到原Mat，要求不支持的保留区域Mat→Mat拷贝，完整编译失败。
  `probe_tile_inplace_loop.py`保留该写法，供查看失败原因。
- `tile_inplace_loop_cpu_v3`：改成`pool = pl.tile.assemble(pool, product, offset)`，
  完整代码生成通过。代码位于`probe_tile_inplace_loop_assigned.py`。

实际IR：`pool`为512B地址处的16×32 FP16 Mat，循环内TINSERT原位写入两个16列窗口，
循环结束后的TEXTRACT读取同一pool；没有Mat→Mat搬运。生成代码在消费前有FIX→MTE1
同步。见`tile_inplace_loop_codegen_audit.json`和完整`.pto.txt`、`.cpp.txt`。

边界：这里只覆盖16×16 FP16矩阵、无分支循环、无pre_quant。尚未证明原Indexer的
M384/INT32到FP16量化、动态Native页分支、双缓冲/FFTS或设备数值。小例子不执行设备。
不能据此宣称之前的Right→Right循环回边问题已解决。

后续可在新候选中保持固定Key/score L1池并回写DPS返回值，让Right提取留在循环内部，
避免把Right作为循环携带值。与旧展开流水相比，下一Key的Right预取可能提前量减少，
要检查实际分配和同步后决定是否值得测128K/B16。旧失败候选和现选均保留。

<!-- SOURCE_END history-document-087 -->

---

<a id="history-document-088"></a>
### 历史原文 088：WEIGHT_CACHE_POLICY_EXPERIMENT.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/WEIGHT_CACHE_POLICY_EXPERIMENT.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/WEIGHT_CACHE_POLICY_EXPERIMENT.md)；原文 3,739 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-088 -->
# 2026-10-17：恢复融合时丢失的权重缓存声明

目标是检查融合单 SPMD 与原动态 PTO 的权重缓存策略是否一致。
不修改数学、调度、精度门槛或依赖仓，不运行 Native forward。

## 发现与边界

原动态 `10_after_OutlineIncoreScopes.py` 对 QA、主 Q、KV、O-A 的只读权重
声明 `cache_policy=[(参数编号, 1)]`；1 对应 `CachePolicy.BYPASS`。
原动态实际 `ptoas/*.pto` 的对应权重 TLOAD 有 `cache_policy` 属性。

实验生成器 `generate_static_entry.py` 合并数学体时过滤原函数的 `pl.func_attr`。
QA、主 Q、O-A 使用 Tensor 数学体重新 lowering，因此这些声明丢失。
KV 使用已经转换的 Tile 数学体，原有两个 `pl.tile.load(..., cache=1)` 仍保留；
所以不能说全部权重的策略都丢了。现选融合实际 IR 共两个缓存属性，位于 KV 读取。
这是实验生成器的行为，当前证据不指向 PyPTO/PTOAS/Simpler 缺陷。

检查 O-A 物理流水时还确认：它已有 K128 内层搬运，Left/Right 在地址 0 与 32768
之间双缓冲，无需再做同样的 L0 双缓冲改写。片段保存为
`oa_selected_physical_fragment.pto.txt`，不把“生成了双缓冲”等同于实际无停顿。

## 候选实现

`weight_cache_policy.py` 从冻结逻辑 IR 的显式声明和本次调用绑定提取四份权重，
检查整个调用图同根视图均未标为 Out/InOut，再按融合后 AIC 参数编号恢复属性。
现有显式 Tile cache 参数保持；没有给 KV cache、state 或临时 Tensor 添加 BYPASS。

BYPASS 的前提是执行期间这些字节没有写者；模型权重满足此合同。
这不是跨核业务数据的发布/等待机制，不能代替原有 fence、缓存维护和进度协议。

开关 `--restore-weight-cache-policy` 默认为关闭，准备合同记录全部原声明及新参数编号。
`weight_cache_policy_prepare_v1` 已通过准备检查；移除新增函数属性后，完整 AST
与现选 `b16_cached_row_deps_v1` 相同。源数学、等待表和 GM 布局保持。

CPU/PTOAS 编译在 `weight_cache_policy_cpu_v1`，完成后必须核对实际 TLOAD/C++
是否含目标缓存属性，再提交正式权重的完整 exact/状态/图检查和三窗无插桩 L1。
当前未有 NPU 结果，不声称性能收益。

## 终态：CPU通过，但生成C++相同，因此未上卡

`weight_cache_policy_cpu_v1`完整CPU/PTOAS通过，实际MLIR的带缓存属性TLOAD由2变44。
然而其`ptoas/csa_static_group.cpp`与现选完整2784561字节文本相同，没有使用hash。
原动态QA的生成代码同样只有普通TLOAD，不能说融合后才丢失有效的硬件BYPASS。

本地PTOAS源码`lib/PTO/Transforms/PTOToEmitC/LoadStore/TLoad.cpp:77`明确要求
`L2Bypass && op.getOffset()`才调整源地址；没有offset时仍生成普通TLOAD。
其`test/lit/pto/tload_cache_policy.pto`也检查这种行为。这与当前生成产物一致，
但不把本地源码HEAD等同于已安装0.66的源码版本。PyPTO的API文档承诺与实际
可见效果有差距；尚未确定应由哪个层补齐运行时offset提供，不将其归为ISA硬件不支持。

A2/A3 PTO-ISA另有`TLoadL2Hint::NotAlloc*`以及运行时填充的
`g_opL2CacheHintCfg.l2Cacheoffset`。当前Simpler动态可调用函数的加载路径与标准
CANN kernel注册不同，未验证该符号能被正确填充；没有硬编码设备别名偏移，也未
修改依赖仓去尝试。此方向保持CPU记录，开关默认关闭，不重复运行等价设备代码。
证据`weight_cache_policy_codegen_audit_v1.json`。初始“全部权重策略丢失”的判断已纠正：
KV显式cache仍在，且标记存在不代表硬件策略生效。

<!-- SOURCE_END history-document-088 -->

---

<a id="history-document-089"></a>
### 历史原文 089：anchor_gap_design_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/anchor_gap_design_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/anchor_gap_design_v1/README.md)；原文 8,654 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-089 -->
# 无同构历史 anchor 时，从源码构造 CSA 单 SPMD 模板

本目录只做 CPU 绑定、依赖与源码构造。没有运行 NPU，没有修改原静态归档、生产代码、
PyPTO/Simpler 依赖或共享 driver。性能目标沿用上级 `GOAL_AND_CONSTRAINTS.md`，
本目录的成本模型与逻辑检查不构成性能达标证据。

## 结论与接口

可以不依赖新增硬件 trace 构造缺失分支。七个历史档位的 task DAG 可以从原编排中的
显式 TaskRef、后期 IR 的 Tensor 读写效应、实际 Tensor view 与手动作用域合同恢复。
历史 trace 对功能构造并非必需；它仍可以提供离线分核所需的初始成本参考。

```python
from anchor_gap_design_v1.source_schedule import build_source_schedule

schedule, audit = build_source_schedule(
    bindings,       # bind_orchestration.bind 的本轮结果
    bodies,         # shape_specialization.catalog()，带归档源文件的绝对路径
    anchors,        # name -> 历史 *_prepared.json 字典，仅用于同名核成本
    policy="earliest_ready",
)
anchor_for_later_rebalance = audit["cost_seed_anchor"]
```

`schedule` 含 `tasks/call_bindings/per_core_sequences/root_arguments/temporary_allocations`
及已有 prepare/generate/validate 所需字段。它已经调用现有 `balanced_schedule.rebalance`
按本轮 block 数分核，并调用 `rebuild_waits` 生成核间进度等待。后续若需再次分核，可将
`audit["cost_seed_anchor"]` 作为 `rebalance(schedule, anchor)` 的第二参数。
不要把 schedule 本身当作历史成本 anchor：rebalance 已把成员 `end` 转为模型字段。

原七档继续使用原入口是合理的最小集成方式。只有明确允许源码生成、且 `choose_anchor`
无同构候选时，才调用本模块。不能宽泛捕获其他绑定异常后静默转入此路径。

## 为什么 bind 的参数注解本身不够

`bind_orchestration.py` 执行的是 `10_after_OutlineIncoreScopes.py`。该阶段某些写参数仍标为
普通 `pl.Tensor`，例如 `hc_widen_rms` 的 `inv_rms`。仅把普通 Tensor 当只读会漏掉依赖。
`generated/core_body_catalog.json` 从 `25_after_ExpandMixedKernel.py` 导出的参数已经记录
Out/InOut 效应；对 MIX 任务，需要将 AIC 与 AIV 的参数效应并集。

最终模块还将逻辑接口与后期效应做保守并集，并对已有 allocation 的写保守加入读依赖，
覆盖 WAW。自动作用域显式检查三种顺序：

- RAW：生产者写后，消费者读取对应重叠 view。
- WAW：已有 view 的先前写完成后才允许后续写。
- WAR：先前读者完成后，才允许后续写重叠 view。

不能把此算法称为历史 runtime ArgType 的逐项重放。每档约 292 个 Tensor 参数中，
有 12 项后期 Out、历史 trace 为 INOUT；它们不改变本轮七档的最终偏序。
正式依据是完整偏序对照及保守效应合同，而非参数标签恰好相同。

`SourceContracts` 对照实际后期 IR 与 catalog 的参数名及注解，核对每个本轮调用与
逻辑源函数的参数顺序/注解、物理核参数覆盖、scope 与 task 选项。未知调度选项、未知
角色组合、未绑定物理参数、非纯 TaskRef 的 dummy 等不作猜测。

## TaskRef、dummy、手动作用域和 manual_dep

四个 dependency-only 任务不执行数学。其显式依赖递归展开到真实 compute 前驱；空 dummy
不会凭空产生屏障。嵌套列表形式的 deps（最终 8 组输出投影的列表）也递归展开。
TaskRef 使用本轮 ordinal，不能沿用历史 runtime task_id。

Simpler A3 当前 `dep_compute.h` 中：

- `compute_task_fanin` 在 manual scope 直接退出自动查找。
- `register_task_outputs` 在 manual scope 也不注册自动输出。
- `manual_dep` 跳过 modifier overlap map，保留原接口的显式关系。

本构造器保留上述手动合同，不能把八个输出投影组的整个共享 allocation 包围区间当作
真实写集合，给它们添加全串行链。原源码的显式组内链、QK 前驱和最终汇合依赖保留。
新静态工作区在入口前分配并在整个执行期间存活，不需要把历史 allocator/retain-only
任务当作 AICore 数学阶段；如果以后引入工作区复用，必须另行证明生命周期。

## 七档对照结果

`seven_anchor_audit.json` 是最初只读依赖复原证据；最终的 RAW/WAW/WAR 模式证据见
`builder_check_v3_war/summary.json` 与各案例的 `source_audit.json`。

| 项目 | 每个原七档的结果 |
|---|---:|
| 数学任务 | 66 |
| 原图直接边 | 98 |
| 重建图直接边 | 101 |
| 原图与重建图的可达前后关系 | 均为 1377 |
| 缺失偏序 | 0 |
| 新增偏序 | 0 |
| 仅显式 TaskRef 的可达关系 | 222 |

多出的三条直接边均已被原图的传递关系覆盖：17→44、26→28、29→39。
WAR 识别出 17→41、20→21、24→25、26→27、29→30，全部已在原偏序中。
66 个任务并没有排成总序；2145 对可能的先后关系中仅有 1377 对受约束。

旧全图签名拒绝 B17～21，并非数学体缺失。例如长历史 B17 的前段使用 B16 所用的核体，
输出投影使用 B24 所用的核体，最终 HC_post 的 sync_start 选项也随原数学分支变化。
新模块从实际编排生成这些组合，不按名称模糊匹配后套旧参数或等待表。

## 缺失分支与 schema 验证

`missing_branch_audit.json` 检查原审计中 10 个缺少 anchor 的签名代表。

- 长/中历史的缺口仍有 66 个数学任务，可恢复与历史图相同的 ordinal 偏序。
- 短历史 `<8192` 的 Vector Indexer 分支有 65 个数学任务、1333 对可达关系。
  其中 `indexer_score_topk_leaf`、`indexer_topk_single_leaf_publish` 不在原七份 trace 中，
  但完整核体与接口已经存在于冻结 catalog；不能用长历史 Indexer 的核体替代。
- 长历史 B1～3 的特定 Indexer 核也已经在 catalog，尽管不在原七份 trace。

`builder_check_v2` 覆盖七个历史档位与十个缺失签名的完整 schema、分核覆盖、MIX 配对、
工作区边界、等待表、三代际随机推进模型和源码生成。v2 已加载加入 WAR 之前的模块。
`builder_check_v3_war` 使用最终 WAR 版本重建；对已生成案例逐项比较每核执行表、等待、
调用绑定和工作区，不相同就失败，不把旧源码直接当作新版本证据。另补 B17/H65536、
B21/H131072、B3/H16384，覆盖中间长度和缺口边界。最终 v2 的 17 个案例、v3 的 20 个案例全部完成，均为 CPU schema/layout/protocol/source PASS。七个参考档位新增 WAR 后偏序仍无缺失或额外关系。

这些不是 PyPTO lowering 或设备 PASS。CPU 进度模型假定既有发布/获取可见性正确；实际
Scalar DCache、DCCI/DSB、原核内 FFTS、重复重放、精度和性能仍需按现有真机流程验证。

## 成本和性能边界

每个新 task/slot 的初始成本优先从七份历史 trace 中选择同名核、最近 T 的样本中位数。
这些是跨形状调度估计；不会标成同规格测量。完全未测的核使用显式标注的初始模型权重
10.0，可通过 `model_cost_overrides` 指定其他正有限权重，仍不代表实际 μs。

现有 balanced 调度器按这些权重选择并行执行顺序；未添加阶段级全核屏障，也没有全串行
fallback。新分支最终需要测量并调优，不能将“可以生成源码”写成“该区间性能良好”。

本模块使用 bind 中的 allocation/view 身份；它不额外发现调用方不同根 Tensor 之间的
运行时别名。入口仍须遵守已有外部 buffer 不擅自别名的合同。任意新增外部别名、数学体
变更或 allocation 复用都需要重新审计。

## 本次尝试记录

1. 初始从后期效应与显式依赖恢复七档：七档偏序全部一致，证明新硬件 trace 不是功能
   DAG 构造的前置条件。
2. 核对真实 trace 参数方向时发现 Out 与 INOUT 的 12 项差异，改为逻辑接口及后期效应
   的保守并集，额外写读约束；未将参数标签等价当作已证明事实。
3. `builder_check_v1` 在首个案例的 Python 测试脚本中遗漏 `use_local_signals` 的 layout
   和 bindings 参数，TypeError；未进入编译或设备。修正调用后运行 v2。
4. Root 复核指出还需覆盖 WAR；构造器加入先前 reader 到后续 writer 的保守边，v3
   重新检查，不沿用旧版本 PASS 名义。
5. 未提交 NPU 任务，没有改 frozen、依赖或共享入口。集成及真机验证由 root 后续完成。

<!-- SOURCE_END history-document-089 -->

---

<a id="history-document-090"></a>
### 历史原文 090：hc_front_cost_audit_20261017/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/hc_front_cost_audit_20261017/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_front_cost_audit_20261017/README.md)；原文 8,852 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-090 -->
# 完整 CSA 中 HC 前段的历史成本审计

2026-10-17，仅重读指定的七档结果、21 份动态 L1 泳道和三份 B16 单 SPMD 阶段记录。
没有重跑测试，没有编译，没有读取旧 worktree。原始路径已在 `evidence.json` 中列明。
当前目标是**七档完整单 SPMD 各自至少降时 30%**，不是 HC 子阶段或 standalone 达标。

## 结论

建议优先移植 HC_pre 的**完整供数与数据复用路径**：AIV 加宽后的 FP32 输入通过
`aic_gather` 给 AIC，同时复用该数据统计平方和；配合实际存在的 L0 双缓冲，再衔接
尾段的门值/混合与 CSA input norm。优先解决当前 widen 与 linear 完全串行、完整
FP32 中间输入落 GM 的问题，而不是再做一次只加 Vector 核数或只融合 reduce/split。

但 HC 不是足以独自实现七档 30% 的证据：现选 B16 的插桩中，首个 HC 数学开始至最后
mix/norm 发布只有约 **60.78 μs**，而该档历史完整结果距 30% 门槛还差 **173.60 μs**。
即使采用极乐观的“去掉全部这段”估计，也不能直接填满此样本差距。该比较仅说明量级，
不是硬件性能下限，也不能将三份不同运行数据当作同轮消融测量。

## 单 SPMD 的直接证据：128K/B16

来源：`b16_row_deps_stage_clock_v1/stage_clock/stages.json`，现选 row-dependencies 策略。
时间相对本次插桩记录最早进入阶段的时间；数学区间包含核内等待和写侧维护，非纯计算忙时。
成员中位数不能跨核求和当成关键路径。

| 阶段 | 成员数 | 数学/维护最早开始～最晚结束 μs | 最晚发布 μs | 成员数学/维护中位 μs |
|---|---:|---:|---:|---:|
| widen/RMS | 12 | 12.76～29.36 | 30.12 | 13.26 |
| linear | 24 | 31.70～39.64 | 42.18 | 6.58 |
| linear reduce | 6 | 45.94～48.70 | 50.06 | 1.07 |
| split pre/post | 12 | 51.00～54.80 | 55.12 | 1.94 |
| mix/input norm | 12 | 56.78～73.22 | 73.54 | 14.73 |
| comb/Sinkhorn | 12 | 50.94～67.06 | 67.08 | 13.42 |

- widen 与 linear 的数学区间没有重叠：存在真实的跨阶段供数串行机会。
- comb 与 split 重叠 3.80 μs，与 mix 重叠 10.28 μs；comb 比 mix 最晚发布早 6.46 μs。
  最晚显式前驱回溯链中没有 comb。单独加速 Sinkhorn 不能按其 13.42 μs 成员时长
  计算整层收益；可能释放资源，但本记录没有给出资源竞争的独占因果证据。
- `row_deps_stage_critical_chain.json` 的 HC 链是
  widen→linear→reduce→split→mix/norm。该链的数学/维护/发布合计 **46.06 μs**，
  相邻 HC 节点的就绪至开工空隙合计 **12.30 μs**。起点另外有 15.18 μs。
  空隙包含未记录控制、等待观察和资源先后，不能全部称为 Scalar 同步开销。
- 两份更早的同 B16 探针中，HC 首个数学至 mix 最后发布为 65.96、62.62 μs；
  widen/linear 同样不重叠，comb 同样提前完成。它们配置不同，只支持此结构判断，
  不构成三轮同配置性能统计。

三份探针的完整 span 为 823.36、826.60、796.24 μs，均有读时钟/写记录和代码布局扰动，
**均不能作为七档 30% 性能验收值**。其余六档没有本次可用的对应单 SPMD 内部 HC 探针。

## 七档动态分 task 背景：不能冒充融合内瓶颈

以下来自各自已选结果的同轮三份**动态 PTO**泳道。六个阶段列为成员 kernel-duration
中位数；最后一列为 HC 首个 widen body 至最后 mix body 的阶段墙钟中位数，含阶段间隙。
不是单 SPMD 的内部阶段测量，更不是可获得的优化收益。单位 μs。

| 历史/B | widen | linear | reduce | split | mix/norm | comb | 动态 HC 前段墙钟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128K/4 | 13.60 | 9.04 | 1.56 | 2.82 | 16.84 | 14.42 | 54.12 |
| 128K/8 | 13.83 | 9.38 | 1.62 | 3.15 | 16.92 | 14.25 | 59.72 |
| 128K/16 | 14.84 | 10.99 | 1.70 | 3.60 | 17.42 | 14.72 | 69.24 |
| 128K/24 | 15.10 | 12.08 | 1.82 | 4.36 | 17.69 | 15.10 | 82.12 |
| 8K/16 | 13.19 | 10.39 | 1.67 | 4.04 | 17.58 | 14.91 | 67.90 |
| 8K/24 | 14.21 | 12.44 | 1.82 | 4.28 | 18.46 | 15.29 | 78.98 |
| 8K/32 | 13.70 | 16.65 | 1.87 | 5.34 | 17.96 | 16.57 | 85.38 |

这支持“线性投影和尾段分工需要按规模选择”的候选筛选方向，不能外推融合核中哪项
是唯一瓶颈。单 SPMD 正式 L1 只有完整融合 task 的首尾，无法从该文件拆出 HC。

## 已关闭候选：已知事实与尚未证明的原因

以下五项均仅测 B16/S6，完整输出/Top-K/六状态/图/30代/保护区 PASS，未采 HC 内部阶段
探针；不应将跨轮几微秒差额定量归因给某项指令或同步。

| 候选 | 完整 SPMD 中位 μs | 同轮动态 μs | 已证明的结构及局限 |
|---|---:|---:|---|
| `hc_stream_pre` | 844.40 | 907.64 | 48 AIV 供数，但仍经完整 FP32 GM store/load；保持原 M16/K4096 四分片，RMS 写 32 份结果；不是独立 HC_pre 的 aic_gather 路径。 |
| `hc_front_fusion` | 806.76 | 900.14 | 只把 reduce→split→mix 合并；widen/linear 原样，GM 中间传递原样，Sinkhorn 独立但等待融合阶段。 |
| 融合内 Tile 直传 | 810.44 | 891.96 | 删除三处静态 GM load、两处 fence，但保留 GM 写与外部同步；新增 Tile 转置/extract。静态指令减少不等于动态关键链缩短。 |
| mix 整行驻留 | 819.02 | 911.56 | 48 AIV 取代 12 AIV，删除 mixed 的 786432 B 回读；仍无明确完整收益。 |
| mix 分块 UB 复用 | 814.74 | 907.88 | 保持原 12 成员及数学顺序，删除同样的 mixed 回读，仍无明确完整收益。 |

来源依次为 `b16_cached_hc_stream_v1/result.json`、`b16_row_deps_hc_front_fusion_v1/result.json`、
`b16_row_deps_hc_front_tiles_v1/result.json`、`b16_row_deps_hc_mix_resident_v3/result.json`、
`b16_row_deps_hc_mix_chunk_buffer_v1/result.json`，均复核了 stage_clock=false 与三个正式 L1 窗口。
前三项相对于现选历史 807.16/817.88 μs，没有足够的同进程直接 A/B 证据证明改善。
不能把 stream 844.40 与不同运行 815.84 的差额全归因于 ACK、RMS 临时区或调度。

已知失败原因也应保留：

- stream 的自动 pipeline 曾把输入 TLOAD 移至 READY 之前，未上卡；最终改 pl.range 才通过
  顺序审计。该实现因此没有证明自动流水形成了原先期待的供数重叠。
- 三阶段融合把 66 task 降到 64，发布 722→704，却使重排后的等待项 12183→12189；
  不能说控制成本全面下降。融合三段只覆盖短小 reduce/split 加 mix，未解决首段串行。
- mix 驻留早期将 16×K256 RMS 错改 8×K512，产生数值差异；恢复原归约后通过。
  这是真实数学合同问题，不能归因于 Scalar/Vector 标签。

以上足以说明旧候选没有完整移植 HC_pre，不能证明完整移植无价值；但也不构成对
完整移植收益的预先保证。

## 建议的迁移顺序及约束

1. 把 `hc_pre.py` 的 AIV→AIC `aic_gather` 供数、FP32 加宽/RMS 数据复用和实际 L0 双缓冲
   作为一体移入 HC 前段。先核对数学分片和累加顺序，初版可以保留 CSA 的四份投影部分和
   与既有 RMS 顺序；不要为了照搬 T96 分支改变归约后宣称逐 bit 等价。
2. 真正移除完整 FP32 输入 GM 中间区的条件，是 mix 从原 BF16 残差按需加载并无损转 FP32，
   而非仍消费旧 widen 的 GM 输出。仅换供数而保留全量中间落地，只吸收了一半策略。
3. 尾段让 pre/post 与 mix 在合适的 AIV 内直连，并继续执行 CSA 的 input norm；
   standalone HC_pre 本身不含该 norm，不能省略。保留 mixed→BF16→FP32 的原舍入点。
4. comb 保持和主混合链并行。优先缩短影响 `x_normed` 就绪的链，再衡量 Sinkhorn 标量化
   是否通过释放 Vector 资源带来整体收益，不把它 13 μs 的成员时间直接算作节省量。
5. canonical `hc_pre.py` 的阶段交接有全参与者硬件 syncall。完整 CSA 中的融合必须让所有
   预期参与核到达，并保留生产/消费缓存维护，不能把这段局部代码塞到部分 worker 分支中。
   也不能为了搬代码把现有无关 RoPE/metadata 分支随意加进一个新的全局屏障。
6. 分档按当前 T/B/历史策略与真实资源容量设计；HC 输入仅取决于 T，并不证明整个 CSA
   只按 T 选档即可。最终仍用完整 CSA 的同轮七档计时判断收益。

这里只提出基于已有证据的优先级，没有启动新的性能/精度测试。

## 本次读取过程

`analyze_history.py` 初次读取动态 trace 时未去掉其 `_spmd` 展示后缀，HC 名称集合断言失败；
核对七个明确文件后修正展示名归一化，再读原数据完成。没有改测量值或放宽事件数断言。

<!-- SOURCE_END history-document-090 -->

---

<a id="history-document-091"></a>
### 历史原文 091：hc_pre_direct_stream_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/README.md)；原文 11,376 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-091 -->
# HC_pre 第一阶段紧凑流式交接设计核对（2026-10-17）

本文是未采用候选的源码设计审计，未编译、未启动 NPU。正式动态 SPMD 实现由主会话放入 `vllm_ascend/ops/pypto/`；本目录只保留本次设计与纯 CPU 协议检查。

## 结论与需要纠正的前提

**在当前 A3 软件路径上，`pl.aic_gather` 不是 AIV 的 UB 直接向 AIC 的 L1 发送数据。**

它确实能形成编译器管理的双槽流式交接，但底层仍为 `AIV UB → GM FIFO → AIC L1`。因此不能把采用 `aic_gather` 解释成消除 GM 往返；收益可能来自及时供数、紧凑地址、重用 BF16→FP32 转换、流水交错和较小驻留集合，必须由完整 CSA L1 incore 对照证明。

本次核对的依赖根目录均为独立 worktree，没有修改共享依赖：

- PyPTO：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016`
- PTO-ISA：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa`

可直接接手的原始源码证据：

| 位置（以上述相应依赖根为起点） | 实际含义 |
| --- | --- |
| `python/pypto/language/op/tile_ops.py:795` | `aic_gather` 必须处于 `split_aiv` 区域，由编译器形成 Vec→Mat 边界。 |
| `src/backend/910B/backend_910b_handler.cpp:31` | 明确说明 A2/A3 cross-core transfer goes through GM；GM→Mat 边界转换为 NZ。 |
| `include/pto/npu/a2a3/TPush.hpp:184` | `pushVec2GMFiFo` 将 Tile 用 `TSTORE_IMPL` 写入 GM 环形槽；两个 AIV 通过行偏移写不同半块。 |
| 同文件 `:147` 附近 | V2C READY 用 `ffts_cross_core_sync(PIPE_MTE3, ...)` 发布。 |
| 同文件 `:434` | `popMatTileFromGMFiFo` 将同一 GM 槽用 `TLOAD_IMPL` 读入本地 L1 槽。 |
| 同文件 `:373` 附近 | V2C FREE 在 AIC 的 MTE2 上发布，表示 GM 槽已经读入本地。 |
| `include/pto/npu/a2a3/TPop.hpp:22` | TPOP 执行 wait→load→按周期 free；Tile TFREE 本身在 A3 实现为空。 |
| `include/pto/npu/a2a3/TPush.hpp:505` | TPipe 析构按真实剩余 credit 排空 FREE，防止下一次调用误消费旧事件。 |
| `src/ir/transforms/allocate_memory_addr_pass.cpp:107` | reserve_buffer 不是普通有生命周期的 MemRef；其他 Tile 均放在保留区之后。 |

不能用 A5 的直接写 L1 能力，或源码中存在 `TINSERT` 一词，就推导 A3 的该交接路径绕过 GM。

## 与已有候选的实际差别

原数学核来自仓库内 `tests/pypto_test/csa_static_spmd_feasibility_20261015/generated/core_bodies_ir.py`：

- `hc_widen_rms`（第 66 行）：每 8 行按 K512 计算 32 项平方和，逐项 FP32 相加；同时保存完整 `[T,16384]` FP32 输入。
- `hc_pre_linear`（第 108 行）：M16，每行 4 个 K4096 部分和，每部分严格 16 次 K256 顺序累加。
- `hc_pre_linear_reduce`（第 137 行）：4 个投影部分和按原顺序相加。
- `mix_x_rms_norm`（第 327 行，实际输入读取第 346～349 行）：还会再次读取完整 FP32 输入。它不是投影的独占缓冲。

已有 `../hc_stream_pre.py` 已使用 48 AIV、M16/K512 及时供数、两个 READY/ACK 槽和 32 项 RMS 部分值，但仍向完整 `view_2` 的原行跨度写入和读取。它限定 T96；历史真机 `../hc_stream_result.json` 为输出与保护区通过、完整 L1 844.40 μs，未采用。这只能否定该具体实现，不能把紧凑交接或本次未实测设计标为有效。

最新正式 `vllm_ascend/ops/pypto/hc_pre.py` 中 K 流式路径与双槽、L0 交错值得参考；其 `aic_gather` 在 A3 同样经过 GM。该文件第二阶段有全硬件核 `syncall`，不能原样插进只处理 CSA 部分阶段的局部 DAG；本候选继续用组内 FFTS 和原阶段进度协议。

删除完整 FP32 输入前，必须先把 `mix_x_rms_norm` 的四次 FP32 load 改为读原 BF16 并无损扩为 FP32。保持 `y01 + y23`、BF16 rint、RMS 与 norm 运算顺序和写入位置。主会话正在独立验证此变换。只给投影加 FIFO 而保留完整 FP32 写入，会增加一次中转写入，不能称为缓冲消除。

## 可落地的最小候选

优先采用**普通 GM Tensor 双槽 + 现有 `tile.load/store` 与 FFTS READY/ACK**，避免在整个大 SPMD 中永久预留 L1 环形区。

保留所有原数学划分：

```text
t_linear = ceil(T / 16) * 16
linear_units = ceil(T / 16) * 4
linear_workers = min(linear_units, 24)

对原 AIC hc_pre_linear 的每个物理组：
  logical_block = 原 records 中的 mapped_block
  for task = logical_block; task < linear_units; task += linear_workers:
    row0 = (task // 4) * 16
    split = task % 4
    两 AIV 的 row0 分别加 0、8
    各自处理 8 个 K512 块，经两个紧凑 GM 槽交给本组 AIC
    AIC 每次仍按 K256 顺序做两次 MMA，累计原 K4096 的同一个 Acc
    完成本 task 后两 AIV 排空各自两槽 ACK，再进入下一 task
```

`logical_block` 必须来自 layout/records 的真实映射，不能假定它恒等于物理 `group_id`。若 layout 不是一逻辑成员对应一个物理组，或 AIC HC 阶段不在第一个 AIC 阶段，需要拒绝该候选并重新分析依赖。

### 多波次与尾行

| T | M16/K4096 逻辑任务数 | 参与 AIC 组 | 每组 task 数 |
| --- | ---: | ---: | --- |
| 24 | 8 | 8 | 每组 1 |
| 48 | 12 | 12 | 每组 1 |
| 96 | 24 | 24 | 每组 1 |
| 144 | 36 | 24 | 12 组 2、12 组 1 |
| 192 | 48 | 24 | 每组 2 |

这不是五点白名单。相同公式覆盖 T 区间；分界来自 M16 以及 24 组执行资源。组内 AIC 与两 AIV 必须使用同一个 logical_block 和步长，循环次数严格相同。

尾行需要 `lane_rows = max(0,min(8,T-row0-lane*8))`。AIV 原 BF16 load 必须携带该 valid_shape，转 FP32 后先 fillpad，再将交接 Tile 恢复为完整 `[8,512]`。即使第二 AIV 有零行，也发送全零半块和 READY；否则 Cube 聚合等待会死锁。不能让尾行越界访问原输入，也不能把两 lane 都缩短后改变 GM 槽的行偏移。

Cube 的数学有效 M 仍按原 `min(T-row0,16)` 处理；从完整槽加载后为参与 MMA 的 Left Tile 设置原 valid_shape，避免改变尾部累加器的物理解释和 store 规则。原投影结果区是 padded 行，保持原 store 行形状。该部分必须检查真实生成指令，不能仅凭填零断言等价。

### 地址与容量

单组两个 GM 槽：`2 * 16 * 512 * 4 = 65536` 字节。按物理 group_id 划分，24 组最多 1.5 MiB。每个槽的两 lane 各写连续 16 KiB，不共享写回块。

可以在确证 mix 已不再读取 `view_2` 之后，重用其前部作为 GM FIFO；但完整 allocation 的容量可能仍保留，不能宣称释放了整个分配。若重用 `[T,16384]` 的原 allocation，必须给出新 `[T*32,512]` 连续视图，访问起点为：

```text
fifo_row = physical_group * 32 + slot * 16 + lane * 8
```

必须检查 `(max_active_group+1)*65536 <= T*16384*4`；不能只按参与组数检查而忽略稀疏物理组编号。T=1～3 等小输入可能不满足复用容量，这只是候选拒绝条件：现有合法策略仍需保留，或使用明确定义的新小型工作区，不能将其变成算子不支持这些形状的理由。

注意：在 incore 中重建 GM Tensor 视图是否被当前 parser/lowering 接受，尚未编译验证。不要仅改函数参数 annotation 后把原 Tensor 当成新 stride；必须让真实视图描述符与索引一致。正式实现可在已有 Host shape/stride 绑定阶段提供共享同一 allocation 的 FIFO 视图，无需读任何 device 数值。

RMS 保持 `[32,t_linear]` 的逻辑布局，每项 K512 的平方和独立写入。可复用原 `temporary_5: [8,16384] FP32`，要求 `32*t_linear <= 8*16384`。当前 T≤384 的区间远小于容量上限，但仍需确认它不再被尾行 widen 或其他阶段使用；不要与 FIFO 复用同一位置。

归约要从 0 开始按 K512 的全局索引 0..31 顺序累加。不得先在每个 K4096 内归约成 4 项再相加；后者改变 FP32 累加顺序。

### 同步语义

沿用旧候选的静态事件：READY=8/9，ACK=10/11。FFTS 事件是组内交接，不涉及非一致 Scalar GM 信号读取；原跨组阶段仍使用已验证的发布/等待实现。

1. AIV 使用某槽前，若不是本 task 的最初两个块，先消费对应 ACK。
2. 完成 FIFO 写入和该 K512 的 RMS 部分写入后，在 MTE3 上发布 READY。
3. AIC 消费本组两 AIV 的 READY，再把完整槽加载进独立本地 L1 Tile。
4. AIC 在 MTE2 上发布 ACK，允许覆盖 GM 槽；其后仍可用已经读进本地的 Tile 做 MMA。
5. 退出每 task 前两 AIV 排空两个 ACK；多波次及下一次图重放不能继承旧事件。
6. AIC 发布原 HC 完成进度前保留原 fence。消费者等齐原 HC AIC 成员后读取投影与 RMS 部分；RMS 发布先于 READY，且 AIC 已消费 READY，才能建立这一传递链。

全阶段进度约束仍需重新审计：删除原 AIC 等全部 widen 的 wait 只可在该 wait 的前驱集合完全等于已被双槽协议替代的 widen 成员时进行；inv_rms 的消费者必须经过写入新 inv_rms 的 linear_reduce 阶段。未满足条件时拒绝变换。

### 为什么不直接引入显式 TPipe

当前 PyPTO 支持显式 `reserve_buffer`、`import_peer_buffer`、`aic/aiv_initialize_pipe`、`tpush_to_aic`、`tpop_from_aiv`。独立 AIC/AIV 函数使用这些 API 在语义上可行，源码例子为 `tests/st/runtime/cross_core/test_cross_core.py:574`、`:707`。

但 M16/K512 两槽需要保留 64 KiB L1，该保留在大函数全生命周期内占用。已有整 CSA 候选的 Mat 峰值为 475136 字节；简单相加为 540672，超过 A3 524288 字节容量。这是容量风险估算，不是本候选编译失败记录。K256 两槽保留 32 KiB，估算507904可容纳，但通知数量翻倍，RMS仍须保留K512行归约，而且实际对齐/碎片还需编译验证。

普通 `tile.load` 所得 L1 Tile 由内存规划器按阶段生命周期复用，不需要为整 CSA 永久保留一个 FIFO 区，是本轮更小的候选路径。不能手写一个固定 SRAM 地址绕过分配器，否则可能与后续 QK/PV 的 Mat 区别名。

## 本次已完成与后续证据门槛

本目录 `plan_audit.py` 只检查工作覆盖、RMS 地址互斥、复用容量和抽象 READY/ACK 协议；它不模拟硬件 pipe，也不能证明数据可见性、数值或性能。

已执行纯 CPU 检查，证据为 `plan_audit.json`：T=1..384 的恒等/逆序物理映射覆盖通过；另外确认稀疏物理映射必须按最高组编号检查容量。1..4 波、每种 128 个随机交错、连续 3 代的 READY/ACK 模型全部通过。未生成正式变换、未编译、未提交 NPU，尚无本候选数值或性能结论。

接手后顺序：

1. 独立验证 mix 读取 BF16 的无损变换，保留现有全部精度门槛。
2. 给正式路径接入可维护的 FIFO 视图与按 task 映射的供数候选，CPU lowering 检查资源、地址与静态事件；不能把本目录设计当作已经实现。
3. 少量代表值先验证所有输出/Top-K/state/guards/连续图重放及两槽排空，再看同轮完整 CSA L1 incore。
4. 有明确收益才扩展不同 T、尾行和多波次；不得以 HC 子阶段或某单核跨度替代完整 CSA 的30%目标。

<!-- SOURCE_END history-document-091 -->

---

<a id="history-document-092"></a>
### 历史原文 092：hc_sinkhorn_scalar_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/hc_sinkhorn_scalar_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_sinkhorn_scalar_v1/README.md)；原文 3,608 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-092 -->
# HC Sinkhorn 标量候选：静态审计，尚未接入

2026-10-17。此目录只提供独立候选，不修改生产 `ops/pypto`、共享生成器或依赖仓库。
没有执行 PyPTO parse/lowering、CCEC 编译或 NPU 测试，没有性能或精度通过结论。

## 改动范围

`transform.py::transform(source)` 接收冻结核体源码，返回新源码和审计字典。
目标是 `comb_sinkhorn` 中首轮稳定 softmax / 列归一化之后的 19 轮迭代。
首轮、函数参数、返回值、全部 GM load/store、完整行/尾行写回分支、其它函数的 AST 均不变。
不增加工作区、跨核信号或同步，也不改变调度器的依赖和成员分配。

原成员处理 8 个 token，候选按 `pl.range(valid_rows)` 逐 token 执行；每个 token 在
16 个设备 FP32 标量中完成 19 轮。支持原成员映射范围内任意 T 的尾块，未添加七档白名单。
只处理真实行，原尾部 scratch 中无效行的数值不承诺与旧算法相同；原 scratch 写回范围不变，
有效输出的读取仍由原 `valid_rows` 限制。实际 scratch/保护区门禁仍须在设备验证。

参考了 `vllm_ascend/ops/pypto/hc_pre.py` 的设备标量思路，但没有照搬其倒数乘法。
候选每个元素仍执行 `numerator / denominator`，先行归一化，再列归一化，eps 位置不变。
列和显式保持 `((row0 + row1) + (row2 + row3)) + eps`。

## 不能静态保证逐 bit 的原因

1. 冻结 A3 向量除法使用 `vdiv`，候选是 Scalar FP32 `/`。同为 FP32 不表示舍入结果相同；
   此项目此前 QR 已遇到这两种除法的舍入差异。保持除法顺序只能避免额外改动数学表达式，
   不能证明当前精度门槛通过。
2. 冻结行归约来自 `pl.tile.row_sum`。当前本地 PTO-ISA 的
   `include/pto/npu/a2a3/TRowSum.hpp` 第 32～36 行将 FP32 归约落到 `vcadd`；
   `TRowReduceOps.hpp` 第 293～305 行对本核 `[8,8]` tile 选择单 repeat 路径，
   第 210～235 行设置 mask 后调用该指令。包装源码没有展开 `vcadd` 的内部加法树。
   候选明确使用 `((col0 + col1) + (col2 + col3)) + eps`，不能仅凭包装代码断言
   与硬件归约树逐 bit 一致。参考 HC_pre 的标量实现使用同样相邻分组，也不能替代本候选验证。
3. 首轮仍完全使用原向量数学，未引入 Scalar exp 或改变稳定 softmax。

上述是候选的真实未证实项；禁止降低精度门槛、静默修改基线或把 CPU 审计称为精度通过。

## 性能风险与后续决策

HC_pre 小档标量路径每核通常只处理一两个矩阵；这里保留原 8 行成员。
候选最多串行执行 `8 × 19 × 32 = 4864` 次 Scalar 除法。
即使省掉短向量指令，8 行串行化也可能明显变慢；comb_sinkhorn 还不一定处于完整 CSA 的关键路径。
当前不接入口、不提交设备任务。若后续决定继续，应先看完整 CSA 的依赖和关键链证据，
再独立决定是否改变每成员行数，并重新证明覆盖、尾块存储和调度依赖。

## 本次审计

执行：

```bash
python tests/pypto_test/csa_multishape_spmd_20261016/hc_sinkhorn_scalar_v1/audit_source.py
```

结果见 `source_audit.json`；可直接审阅 `candidate_comb_sinkhorn.py.txt`。
审计检查 AST 修改边界、GM 访问不变、16 次 tile 读写、每轮 32 个直接除法，
并枚举 T=1..384 的原 8 行成员划分，确认真实 token 恰好覆盖一次。
这只是 Python/AST 与逻辑覆盖检查，不是 PyPTO 解析、编译、设备精度、调度正确性或性能证据。

<!-- SOURCE_END history-document-092 -->

---

<a id="history-document-093"></a>
### 历史原文 093：observed_member_cost_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/observed_member_cost_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/observed_member_cost_v1/README.md)；原文 7,136 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-093 -->
# 逐逻辑成员成本复核：主 Q 后处理缩依赖

2026-10-17。仅使用已有 128K/B16/S6 单次插桩数据，未提交 NPU 任务，未修改数学、等待表、生产代码或依赖仓。

结论：此前“同 task/slot 中位成本”确实抹平了主 Q 投影的工作量和启动差异，但换成逐成员实测成本后，head_tile=4 的整层模型收益仍只有 0.60～2.26 μs。加入后处理每成员 4 μs 的额外开销便完全抵消。因此本次不建议仅凭这个模型生成并测试新数学候选；关闭这条 CPU 调查，不把旧模型无收益描述为硬件上绝不可能受益，也不把新模型的小幅变化描述为已实现优化。

## 输入身份和测量边界

历史探针为 `../b16_row_deps_stage_clock_v1`，现选为 `../b16_cached_row_deps_v1`：

- 两边 `entry_jit.py`、`entry.bindings.json`、`layout.json` 文本完全相同。
- prepared 清单的 root arguments、数学调用绑定、任务、每核顺序全部相同；额外差异只有审计描述字段。row dependency 合同的旧字段全部相同，探针新增 `indexer_input_dependencies: null`。
- 973 条时戳以 `(core, sequence)` 一一匹配 layout 和现选成员；task、kernel、物理角色也逐项相同。没有按运行结束顺序猜逻辑核。
- 两份真正交给 CCEC 的 AIC/AIV 源码均有 296 个时戳写入，并保留 DCCI/volatile 读取与 cached100 等待。路径列在 `result.json` 的 identity 中。
- 每成员成本为 `math_and_fence_us + notify_us`，明确排除外层 wait 区间。后者包含轮询、读侧维护等，不能直接当成可消除的纯空闲。

这是一个最终调用留下的单次样本，不是多轮逐成员中位数。探针完整跨度 796.24 μs 本身也不作为正式性能数据；当前无插桩现选约 807 μs 仍是正式参考。

首次脚本因 row audit 新增 null 描述字段而拒绝，日志 `run.log` 保留；核实新增字段不改变合同后，逐项比较所有旧字段并记录新增字段，`run_v2.log` 通过。补充实际 CCEC 检查后 `run_v3.log` 通过；为保持单份结果在2MiB过程镜像范围内拆分逐成员时间轴，最终执行 `run_v4.log` 通过，模型值保持不变。

## 逐成员成本的确暴露了差异

主 Q 投影共 128 个 N256 块，以 worker + round × 20 分给 20 个逻辑 worker。

| 每 worker 的块数 | worker 数 | 数学与写侧维护中位数 | 再含 notify 的中位数 | 观察到的最后发布时刻 |
|---|---:|---:|---:|---:|
| 7 | 8 | 49.65 μs | 55.02 μs | 213.72 μs |
| 6 | 12 | 42.29 μs | 47.59 μs | 221.28 μs |

最后完成的是较晚启动的 6 块 worker，而非单人成本较大的 7 块 worker。完整逐 worker 明细保存在 `result.json` 的 projection_members。这个结果只证明该次 B16 的观察，不能外推其他 batch 的核间竞争。

## 固定分核模型

现选数学依赖、同核顺序和 216 条 Indexer→QK FFTS 复用边保留。候选只收窄已由 `qproj_dequant_dependencies.contract` 证明的投影→后处理、后处理→QK 两组依赖，没有移动成员或进行贪心/MILP 搜索。

MIX 模型把同逻辑组 AIC/AIV0/AIV1 的开始时刻合并，后继边仍使用对应生产者角色的耗时。另一套独立成员模型作为敏感性对照；两者均不能保证对应硬件上的准确耗时。

“观察启动偏移”只把每条物理核第一个阶段 stamp0 的相对时间作为起始释放条件；后续成员仍由候选依赖计算开始时间，没有把整条已观测时间轴冻结下来。

以下是 **CPU 模型数值，不是硬件时间**，单位 μs：

| 成本及模型 | 原依赖 | head16 缩依赖 | head8 缩依赖 | head4 缩依赖 |
|---|---:|---:|---:|---:|
| 旧 task/slot 中位数+1，独立成员 | 771.53 | 771.53 | 771.53 | 771.53 |
| 旧 task/slot 中位数+1，MIX共同开始 | 773.41 | 773.41 | 773.41 | 773.41 |
| 逐成员观察，独立成员，零启动偏移 | 727.86 | 727.86 | 727.86 | 727.26 |
| 逐成员观察，独立成员，观察启动偏移 | 742.64 | 742.64 | 742.64 | 742.04 |
| 逐成员观察，MIX共同开始，零启动偏移 | 737.24 | 737.24 | 736.54 | 734.98 |
| 逐成员观察，MIX共同开始，观察启动偏移 | 752.02 | 752.02 | 751.32 | 749.76 |

不能把 771.53→727.86 当成优化：这是成本来源改变，不是代码变快。有效比较必须在同一行内。

head4 依赖 8 个投影 worker，而原 head16 涉及全部 20 个。即便如此，完整关键路径只缩短 2.26 μs。原关键链由最后一个主 Q 投影 worker（物理 AIC4、逻辑19）接后处理，再到 Indexer MIX组；head4 改为另一条后处理→Indexer组链，随后仍接 merge、cache writeback、QK、OA/OB/HC post。没有绕开 Indexer、QK 及输出投影的主要成本。完整关键链含每个成员的开始、结束与边见 `result.json`。

在“MIX共同开始+观察启动偏移”模型中，额外循环成本的敏感性为：

| 每个后处理成员新增成本 | head8 模型 | head4 模型 | 原依赖参考 |
|---|---:|---:|---:|
| 0 μs | 751.32 | 749.76 | 752.02 |
| 1 μs | 752.32 | 750.76 | 752.02 |
| 2 μs | 753.32 | 751.76 | 752.02 |
| 4 μs | 755.32 | 753.76 | 752.02 |

head8 不到 1 μs 的额外成本就抵消收益；head4 约 2.26 μs 抵消。这里没有测量 head4 的真实新增循环成本。

## 局限与后续边界

- `math_and_fence` 内仍含 MIX 内部 FFTS/pipe 等待，不是纯计算。观测到 Indexer 同组数学入口偏斜 7.30～21.78 μs，中位14.40；QK 为0.46～4.64，中位2.24。强制共同开始再保留旧时长可能重复计入内部等待，独立模型又可能低估耦合。
- 候选仍借用原数学的每成员时长和 notify 时长。分块改动会改变外层循环、地址局部性、发布集合和竞争；不能保证成本不变。
- 未计外层轮询延迟、分支控制、真实 kernel prologue 等。模型不是设备下界，也不能用752.02宣称距700只剩52.02。
- 本合同只覆盖 B16/S6、T96、单 tile、head64、N256/20投影worker和48后处理worker。它不是分档规则，更不是所有 batch 的专用分支建议。
- 如将来有更强依据要做该候选，还缺实际 head 分块改动、相应等待表生成与缓存发布合同、完整 CPU lowering/实际 CCEC 检查、正式权重输出和状态精度验证及无插桩 L1对照。
- 泛化到其他 batch/seqlen 范围前，要重新枚举行/head覆盖、尾块、实际投影worker划分及 workspace边界，使用对应形状的成本和验证。不能套 HC_pre 阈值或用本次 B16 数据代替其他区间证据。

## 复现

在本实验根目录执行：

```bash
python observed_member_cost_v1/analyze.py
```

脚本只读历史材料并重写本子目录结果，约6秒CPU；不使用NPU、不做hash、不改变共享helper。`model_table.json` 为紧凑比较表，`result.json` 包含身份验证、全部成员观测、源合同、所有模型和关键链。`*_timelines.json` 分别保存两类模型的逐成员开始/结束数组，以 result 内的 timeline_key 对应。

<!-- SOURCE_END history-document-093 -->

---

<a id="history-document-094"></a>
### 历史原文 094：production_spmd_package_audit_20261017/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/production_spmd_package_audit_20261017/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_spmd_package_audit_20261017/README.md)；原文 13,191 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-094 -->
# CSA 动态形状单 SPMD：生产包依赖闭包审计

2026-10-17。只读代码设计审计，没有运行测试、重新编译、启动 NPU 或修改共享 helper。
本报告与同级 `hc_front_cost_audit_20261017/` 成本报告配套。工作目录为当前 canonical 仓库；
旧动态 worktree 只作为历史来源，不作写入目标。

**迁移进展补记：** root 已将 73 个数学核体迁入生产 `kernels.py`，并迁入 `policy.py`、
`signals.py`、`signal_codegen.py`；新的验证已读取这些生产实现。下文的“当前来源”表
用于交代原依赖闭包，不能误读为要求再复制这批核体。下一步重点是消除剩余逻辑/late
IR 的重复依赖，以及收敛生成与绑定入口。新增只读 AST 审计见 `logical_contract_audit.json`。

## 结论

正式实现可以放到 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`，而且可以完全不读取
七份历史 anchor、trace、binding JSON 或成本 profile。现有 `build_source_schedule` 已具有
这种算法能力：`anchors={}` 时构造任务图、MIX 分组和逐核调度，未测核使用模型权重 10.0。
这不是各档性能已经得到保留的证明；当前 selected 七档使用的是实测成本，换成常量后
分核和等待表可能变化，不能继承原七档耗时或设备 PASS。

目前 `range_policy.build_range_variant` 仍不独立：有同构 anchor 时先走旧重绑定，只有
缺少 anchor 才走源码 DAG。正式入口应把**所有规格统一走源码 DAG**，保留区间策略，
删除“找不到旧 anchor 才允许新形状”的控制结构。

最大未完成项不是 DAG，而是数学源码和运行入口的正式化。现在生成器读取冻结 IR，
运行测试还依赖测试参数准备器及全局编译回调。单纯搬 `range_policy.py` 并改 import
路径不构成独立生产算子。

已经迁入的生产数学体应继续作为唯一维护点，不能再维护一份 tests 候选核体作为线上
依赖；本报告以下给出进一步压缩剩余接口源码的可行方法。

## 当前 selected 路径的依赖闭包

下表只含当前已选路径，不把 `shape_specialization.py` 的几十个历史实验全部带入。
N 表示本目录上级，F 表示同级 `csa_static_spmd_feasibility_20261015`。

| 环节 | 当前来源 | 性质与迁移要求 |
|---|---|---|
| 区间选择 | N/range_policy.py `select_policy` | 纯 Host 描述符算法。保留 B/T/S、长度容量、分支边界；删除 selected JSON 与 CLI 参数处理 |
| 编排及 Tensor view 绑定 | F/bind_orchestration.py | 算法本身无需测量，但当前解释冻结逻辑 IR；保留 TaskRef、dummy、scope、manual_dep、别名及方向合同 |
| 数学源码和效果 | F/generated/core_bodies_ir.py、F/references/symbolic_ir 的 Outline/ExpandMixed 文件 | 算子代码来源，不是性能 trace；当前是历史文件依赖，必须转为包内受维护的源码/明确接口合同 |
| DAG | N/anchor_gap_design_v1/source_schedule.py、audit_dependencies.py 的核心函数 | 可纯源码生成，RAW/WAW/WAR + 显式依赖；移除审计 CLI、F 常量和绝对路径 |
| 分核 | N/balanced_schedule.py | 核日历、earliest_ready/critical_path、MIX 三成员配对是纯算法；成本 profile 读取不是必需项 |
| 等待偏序 | N/shape_specialization.py `rebuild_waits`，F/build_schedule_manifest.py `ancestor_sets` | 抽取纯函数；不能为了一个拓扑工具 import 整个 trace 读取模块 |
| QK 索引优化 | N/qk_index_cache.py | long 且 32<T<=96 的已选优化；纯源码变换但依赖 SSA 名和源合同，适合写成正式数学分支 |
| B16 行依赖 | N/row_dependencies.py | long、T=96、B=16 的已选优化；按函数名查任务，不靠旧 task_id；保留行映射证明和 FFTS 资源边 |
| 工作区与成员批次 | F/prepare_static_layout.py `prepare` | 纯算法；不复用临时 allocation，布局含控制表和进度区；可迁移 |
| 入口生成 | F/generate_static_entry.py `generate/jit_source` | 纯代码生成 + 数学源码依赖；含必要的 NZ、动态维度和别名处理，不能只保留调度外壳 |
| GM 信号 | N/signal_lowering.py、F/local_signal_adapter.py | 算子必需的发布/获取协议，迁为包内相对 import；保持 payload fence/cacheinvalid、bar_all 和 signal notify/wait |
| 等待退避 | N/signal_polling.py | selected 的 cached+DCCI+100 tick 来自这里；当前 monkey-patch 全局编译函数，不宜原样作为生产入口 |
| Native Tensor 绑定 | F/prepared_arguments.py | 可抽取，当前从 tests JSON 读布局；正式版直接接受生成的合同/布局对象 |
| 实际 JIT 调用 | N/device_compare.py 的少量入口逻辑 | `module.entry(*core_args, config=options)`，直接 native prepare/enqueue，不依赖 torch.library 大参数注册；测试驱动不可导入生产 |

`reuse_gm`、HC front/stream 实验、其它 Indexer/QK 实验、stage clock、协议随机推进、L1 采集、
Native 对照和 fixture 均不属于当前生产最小闭包。

## 无 anchor 生成的具体合同

当前已有可调用 API：

```python
schedule, audit = build_source_schedule(
    bindings, catalog, anchors={}, policy=policy.schedule_policy,
    model_cost_overrides=None,
)
```

它已经做一次 `rebalance`，返回含 `tasks/call_bindings/per_core_sequences/root_arguments/
temporary_allocations/wait_plan` 的完整 schedule，可交 `prepare/generate`。不应再把它当作
历史 anchor。若兼容旧二次 `rebalance`，第二参数应是 `audit['cost_seed_anchor']`，但新入口
直接删除重复分核更清楚。

当前 `rebalance` 为每个模型成员成本再加 1.0，所以无覆盖值时实际调度使用 11.0。
这属于无量纲分核权重；旧结构以 `start/end ×50` 承载它，不应输出成预测的实测微秒。
生产版建议直接传 `cost[(task, role)]`，把 seed anchor 名称及伪时间字段从接口中去掉。
可以先使用统一常量，后续按矩阵块数/历史切片数提供明确的算法权重；两者均不需要
历史 profile。权重只影响映射和顺序，不能用于删除数学依赖或宣称 30% 达标。

源码 DAG 替代七 anchor 必须保留：

1. 逻辑接口与后期物理核的读写效果并集。仅看前期普通 `pl.Tensor` 会漏写效应；例如
   `hc_widen_rms` 的 `inv_rms`。已有 allocation 写保守包含读取，并处理 RAW/WAW/WAR。
2. TaskRef 的本轮 ordinal、嵌套 deps、dummy 展开、manual scope 和 manual_dep 原语义。
   手动作用域不能按整个共享 allocation 强行补成全串行链。
3. 每个逻辑 block 覆盖一次；MIX 对应同物理组的 AIC+AIV0+AIV1。保持原数学体内部 FFTS
   信号的资源顺序，不把任务偏序证明误作全部硬件同步证明。
4. 实际 Tensor view 的 allocation、offset、dtype、shape、stride、NZ 格式及返回别名。
   当前算法不自动发现不同根参数的运行时 storage 别名；入口需验证/约束已有别名合同。
5. 未识别的参数注解、task 选项、作用域或核角色组合明确失败。不能静默套用旧模板。

已有 CPU 证据：七档均为 66 个数学任务，旧图 98 条直接边、源码图 101 条直接边，但
可达偏序均为 1377 对，缺失/新增偏序均为 0；WAR 新识别的边也已被原偏序覆盖。
20 个 CPU 案例覆盖无同构组合和短 Vector 分支。**这些历史检查使用已有的成本种子，
并不等价于本次常量成本版本已跑过设备。** B12/H65536、B17/H65536 的后续真机 PASS
是已生成候选的证据，也不能自动覆盖正式包重构后的所有调度。

## 数学源码如何收敛，不能直接搬整份历史 dump

当前三份源码分别为 7,488、14,751、17,619 行，合计 39,858 行；其中存在重叠。
`SourceContracts` 读取 late IR 主要为了核对 catalog 的参数方向；`generate` 的主体来自
core body bundle，但遇到原 NZ 输入时会改用 Outline 阶段逻辑核体，避免 rank-5 NZ load
被二次 lowering 成错误的 Tile。不能盲目只留 late body，然后声称 NZ 路径无变化。

建议最小、可审查的收敛方式：

- 包内保留各分支实际所需的数学函数，包括未在七点出现的短 Vector、B1～3、中间 B
  组合；不要依据七份 bindings 剪掉合法区间。所有 scalar/device metadata 动态计算保留。
- 保存每个逻辑任务的正式参数、返回别名、效果、物理角色及 worker 语义合同，直接从
  包内源码推导/核对，避免同时维护 late IR 全文件和重复 catalog。
- 原 NZ 核保留适合重新进入前端的逻辑数学表达；其它核保留当前已验证的 AIC/AIV 体。
  逻辑编排可先移植现有解释器，但只解释包内受维护的编排源码。
- 让 catalog 由这些包内定义生成，或使用小型源码声明；不从 tests 读取 JSON。旧 SSA
  名相关 AST 变换可先保留严格守卫，随后改为明确的函数参数/数学分支。

这要求一次明确的数学体/接口提取与对照，不是改几个路径即可完成。已有生成器中的
`RestoreBranchYields`、`NormalizeViewValidShape`、维度绑定、InOut 别名回绑、NZ 处理都
有实际前端用途；只有在正式源码自然表达了同等语义之后才能删除。

## 建议的最小生产组织

```text
deepseek_v4_flash_csa_spmd/
  __init__.py              # 稳定调用入口
  policy.py                # CSA 区间/形状策略；无七点 JSON
  math/                    # CSA 数学体与正式编排，按现有职责拆分
  contracts.py             # 形状、效果、返回别名、MIX 及受支持范围
  graph.py                 # TaskRef/dummy/manual + Tensor view DAG、拓扑/等待
  schedule.py              # 分核、算法成本；无 trace/profile 加载
  layout.py                # 工作区及控制表生成
  codegen.py               # 单 MIX SPMD 入口生成、绑定及数学特化
  signals.py               # 单写者进度 notify/wait、代际及缓存维护
  runtime.py               # 参数/工作区准备、JIT/cache、直接调用
```

名称仅表示职责，不要求机械拆成十个文件。`qk_index_cache` 与 `row_dependencies` 可以
归入 math/graph 的带守卫分支，不能将全部历史候选模块一并复制。

`runtime` 应在 warmup/capture 前生成调度和分配工作区，通过对象传递合同/布局；每个
并发调用者拥有独立 workspace/progress。进度初始化一次，图重放不清零，耗尽 INT32
代际时必须在排空后重置。Native 输入、四份 NZ 权重及 state 保留原存储，视图只做
绑定，不做 FP32 权重转换或 state 搬运。

内核 ABI 当前要求 `leaf_tensor_count <= 256` 的独立 PyPTO/Simpler 能力，入口必须按
实际能力检查，不能继续读测试环境变量或通过 tests 的隔离 bootstrap 获得能力。
应保留直接 JIT/native 调用路径，避免重新引入 torch dispatcher 的 64 项注册约束。

## 当前 selected 策略不能遗漏的两个条件

**行依赖：** B16 长历史优化按逻辑函数名重新找 task ID，故 source ordinal ID 本身
不阻塞迁移。但是它还要求唯一的长 Indexer/merge/plan/QK、8 组 O-A、明确 SSA/view
映射和 T96 行划分；不能扩展到 B17 等形状。它额外保留本物理组 Indexer 三成员在 QK
前完成的 FFTS 资源边，这是正确性合同的一部分。

**等待退避：** `local_signal_adapter` 仅有 bar_all + signal_notify/wait；selected 的
DCCI 后 volatile 读 + 100 SYS_CNT tick 退避来自 `signal_polling` 对 PyPTO 私有全局
`_compile_pto_module` 的回调替换。迁移不能悄悄去掉它并沿用旧性能数字，也不宜在可
并发编译的生产进程中原样 monkey-patch。优先形成算子局部、正式受支持的 wait lowering
扩展；若暂保留适配，须明确隔离/串行编译作用域并审查最终 AIC/AIV 源码，不能只检查
选项布尔值。DCCI/DSB 和 payload fence 不可因退避优化被删掉。

## 边界及实施顺序

当前 selector 只接受 S=6、T=B×6、B1～64、总长度容量至 131200；这些是现有能力边界，
不是对最终动态形状目标的重新定义。长/中/短策略跨界时需换模板；档内地址、有效长度、
尾块和循环继续由 device metadata 计算，不把 synthetic metadata 留作真实输入。

建议先完成无历史数据依赖的生产闭包，统一 source DAG + 常量算法成本，形成可调用
且可审核的功能版本。数学体、同步协议、Native 原存储接口先保持明确合同，再把
已选的 QK 索引/行依赖及局部 wait 退避迁进去。测试留在 tests：对照源码偏序/覆盖、
lowering、输出与状态、图重放及最终七档完整耗时分别验证。历史 selected 数字仍留作
历史参照；正式版七档逐档 ≤ 原动态 PTO ×0.70 需新版本自己的证据。

本次没有为上述迁移运行新测试，没有复制大 trace、成本 profile 或 binding 到生产，
没有修改 shared helper、生产代码或依赖仓。

<!-- SOURCE_END history-document-094 -->

---

<a id="history-document-095"></a>
### 历史原文 095：qk_row_load_pipeline_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/README.md)；原文 7,627 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-095 -->
# QK 独立行 Tile 加载流水：CPU 负结果，停止上卡

2026-10-17。本次仅完成源码变换和完整 CSA 的 CPU lowering，没有启动 NPU，没有修改生产代码、共享 helper/driver、冻结基线或依赖仓。

**结论：当前 `stage=2` 写法不成立，不建议上卡，也不扫描 stage。** 它没有把相邻两行的 MTE2 加载提前，反而在每个有效行上增加了一次 1 KiB UB→UB 拷贝及其等待。更关键的是，现选 B16 基线原本就直接把每行 TLOAD 写到目的 Tile 的不同子行，生成 C++ 的 64 行循环内部没有逐行显式 wait。原先将“逐行 gather”推断为“每行必须等上一行完成”的假设，不能成立。

这只关闭本次具体变换。它不证明所有独立行流水结构都不可行，也不把 C++ 的提交顺序当成设备吞吐或耗时。

## 实际输入合同与变换范围

真实冻结函数是 `qk_pv_aiv`，在完整入口中为 AIV 的 `stage_30`。本次准备/完整 lowering 使用 128K B16/S6、`qk_vector_indices=True`、`row_dependencies=True`，在原 `qk_index_cache` 地址计算之后接入新变换。

**短 B32 当前策略是 `qk_vector_indices=False`。** 本报告不把长 B16 的缓存组合称为短 B32 基线。通用变换接口同时支持 `raw_scalar` 原地址计算，已完成该版本 AST 合同验证，保存在 `raw_transform_only/`；没有为 raw 版本再做完整 lowering。

[变换入口](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/row_load_pipeline.py)：

```python
changed_catalog, audit = use_row_pipeline(schedule, catalog, output_dir)
```

接口只改压缩 KV 的 64 行循环：原 `gather_row(dst, src, [row,0], [physical_row,0], [1,512])` 变成独立 `[1,512]` `tile.load`，再 `tile.assemble(dst, row_tile, [row,0])`；循环从 `pl.range` 改为 `pl.pipeline(..., stage=2)`。

以下合同保持原样，并有 AST 断言：

- Top-K 的原始索引、页表/物理行地址计算、原 `raw>=0` 有效性条件。
- 无效行保留初始零值，不读取无效地址；按原行号 assemble。
- SWA 分支、数学计算、QK AIC、输出写回与所有 FFTS 调用。
- 每轮 64 行、每行 512 个 BF16、原循环 carry；不新增 GM 分配。

[准备入口](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/prepare.py) 仅在自身 Python 进程内串联现有地址变换，不编辑共享 Python 文件。

## 完整 lowering 的实际证据

[编译结果](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/cpu_v1/result.json) 的 `full_codegen_passed=true`，`device_execution=false`。包括完整单 SPMD CSA，不是另写的简化 QK kernel。没有 CCEC 设备运行、精度或性能结论。

自动审计：[audit_lowering.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/audit_lowering.py)、[结果](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/lowering_audit.json)。

| 检查项 | 结果 |
|---|---|
| 软件流水展开 | 每次处理两行 |
| 两个行 Tile 的实际 UB 地址 | 都为 `[131072,132096)` |
| 相邻行操作顺序 | `TLOAD → wait → TMOV → TLOAD → wait → TMOV` |
| 下一行是否在上一行 assemble 前提交 | 否 |
| 每个有效行增加的 UB→UB 数据量 | 1024 字节 |
| 64 行全部有效时，单 AIV 半块增加的拷贝 | 64 KiB |
| 一个 query 的 512 个压缩候选全有效时，两 AIV 合计增加的拷贝 | 512 KiB |
| GM 读取字节数 | 未减少 |

内存规划的最高已分配结束地址（字节）：Vec **163872**，Mat/L1 **475136**，Left **65536**，Right **65536**，Acc **131072**。这是完整生成程序的静态地址上界，不是运行时采样的内存峰值。没有溢出，但两个新增行 Tile 并未形成两个同时存活的加载槽；其相同地址不应被解释成成功实现了双缓冲。QK 的 AIC/L1 tile 形状没有改动。

候选完整生成 C++ 的关键位置（路径在审计 JSON 中）：

- 第 **48807** 行：第 0 行 `TLOAD`。
- 第 **48816–48817** 行：等待 MTE2→V，然后 `TMOV` 到目的子行。
- 第 **48897** 行：此时才提交第 1 行 `TLOAD`。
- 第 **48906–48907** 行：第二次等待与 `TMOV`。

摘录：[candidate_generated_loop.cpp.txt](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/candidate_generated_loop.cpp.txt)。两行之间还存在 V→MTE2 的同步，不能把源码的 `stage=2` 当作已经发生重叠。

基线摘录：[baseline_generated_loop.cpp.txt](../../../tests/pypto_test/csa_multishape_spmd_20261016/qk_row_load_pipeline_v1/baseline_generated_loop.cpp.txt)。来自 `b16_cached_row_deps_v1/build_output/entry_q0pmqift/ptoas/csa_static_group.cpp`：第 **48777** 行的 TLOAD 直接写目的子行；整个 64 行循环没有显式 `wait_flag` 或 `pipe_barrier`，块末写回前才等待 MTE2→MTE3。这里只证明生成代码没有逐行加载完成等待，实际 DMA 排队深度、发射吞吐与访存延迟仍须设备证据。

## 为什么没有形成预期流水

两行的有效性 `if` 分别把本行 `load + assemble` 包在一起。流水展开后仍是两个完整条件块，IO 重排没有跨条件块提取两次 load；每个 load 的结果立即被 assemble 消费。因生命周期不重叠，内存规划也把两个临时行 Tile 合并在同一 UB 地址。

本地 PyPTO 的 `src/ir/transforms/canonicalize_io_order_pass.cpp` 将普通 `AssignStmt` 的 load 分类为 Load，递归重排每个 `SeqStmts`；但本次 `IfStmt` 仍作为整体留下。这与产物一致。这里没有声称这是编译器正确性错误，也没有修改编译器。

因此，这个写法同时具有两点不利因素：没有消除基线已有的等待（基线循环中并无该等待），却新增了每行 assemble 所需的 V 流水数据移动及同步。

## 若以后重开，最小结构要先满足什么

只能作为后续结构建议，本轮没有实现：显式把同一对行的**有效性判断/受保护加载**放在前半段，再在后半段按原行顺序分别 assemble，并让两份行 Tile 的生命周期重叠。仍须保留每行原有效性条件，不允许借无条件加载读取负索引地址；生成代码必须同时证明两份 UB 地址分离、第二次 TLOAD 在首次消费前提交、条件块边界没有提前 drain。

即使满足这些条件，也还需解释为什么能胜过基线的“直接 TLOAD 到目标子行”：仅获得两个临时槽不足以证明收益。必须有减少有效 DMA 指令数、降低地址生成成本或隐藏原有真实停顿的新依据，不能仅用“独立 Tile/双缓冲”命名来决定上卡。已有全有效块、MGATHER 等关闭方向不因此自动重开。

## 复现与边界

`prepare.py`、`row_load_pipeline.py`、`prepared_contract.json` 保留准备过程。`cpu_v1.log` 保留本次完整 CPU 编译日志。编译使用本 session 的独立 PyPTO/Simpler worktree，没有改变共享安装。

完整编译方式（路径按当前 workspace）：

```bash
source /data/pyptouser/qinchuanyu/pto-eager/env-dsv4-0251rc1.sh
export CSA_STATIC_PYPTO_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/pypto-csa-static-fixes-20261016
export HCA_SIMPLER_ROOT=/data/pyptouser/qinchuanyu/pto-eager/.worktrees/simpler-csa-core-args-256-20261016
export PYTHONPATH="$HCA_SIMPLER_ROOT/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

随后使用冻结目录的 `bootstrap_isolated.py compile_jit_entry.py`，输入本目录的 `prepared/variants/h131072_b16/entry_jit.py`，参数 `--memory-planner PYPTO --kernel-mode`，输出到新的 CPU 目录。审计当前留存结果执行：

```bash
python audit_lowering.py
```

本次结论基于 B16 的完整产物。raw 地址版本尚未完整 lowering，更未测试其他档位。对其他规格可泛化的只有“保留的变换合同”和“当前 if 结构不会自动跨行提取 load”这一代码结构观察；不外推设备收益，不宣称达到任意一档 30% 性能目标。

<!-- SOURCE_END history-document-095 -->

---

<a id="history-document-096"></a>
### 历史原文 096：query_shape_support_audit_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/query_shape_support_audit_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_support_audit_v1/README.md)；原文 10,839 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-096 -->
# CSA 的 query 数泛化：当前限制、可复用部分与最小安全路径

结论：**当前整层 PTO CSA 的请求语义确实限定 uniform S6，限制同时存在于入口、数学核与冻结 IR，不能只解除 fixture/入口检查。**
这应作为当前实现边界，而不是把最终目标缩减为 S6。HC_pre 提供的是依据工作量选择策略的设计思路，不能照搬其具体阈值。
本文把历史/可见 KV 长度记作 L，把本轮每请求 query 数记作 S；二者不是同一个 seqlen。
S6 的各个 B/L 是否已验证仍需分别看现有记录，本审计不扩大其设备通过范围。

只读取本 worktree 的 production、冻结逻辑 IR 与测试代码；新增文件都在本目录。
没有导入 PyPTO、没有修改 S 常量、没有重新 lowering、没有执行设备、没有改运行中任务的 helper。
`audit_query_shapes.py` 可重现 AST 检索及纯整数反例；结果在 `source_inventory.json`、`counterexamples.json`，日志在 `audit.log`。

## 1. 入口与 fixture 限制

| 位置 | 当前合同 |
|---|---|
| `config.py:193`～`:198` | DECODE_BATCH=64，DSPARK_SPEC_TOKENS=5，DECODE_SEQ=6，容量 DECODE_TOKENS=384 |
| `service_config.py:8`、`:27`～`:30`、`:75` | QUERY_TOKENS=6；图重放要求 uniform_decode 且实际 T=B×6；要求 dspark 的 speculative tokens=5 |
| `service.py:33`～`:53` | 整层服务要求 T 整除6、每请求六行，metadata.max_seqlen_q=6；补位请求另按 seq_lens=0 屏蔽 |
| `native_adapter.py:250`～`:269` | T=B×6，实际 token 数也整除6；绑定的是原 Native Tensor，不为其他 S 重排 |
| `N/shape_specialization.py`、`N/range_policy.py` | 当前入口要求 S6、T≤384；尚未提供其他 query 数数学体 |
| `tests/pypto_test/dsv4_csa_single_layer.py:95`～`:99`、`:143` | fixture 直接构造 T=B×6、bounds=b×6、length=H+6、positions=H+arange(6)、max_query_len=6 |
| `N/device_compare.py:490`、`:657`～`:667` | 测试也按 T=B×6、L=H+6 绑定 |

这些限制需要随泛化同步更新，但它们不是唯一阻碍。

## 2. 数学 kernel 中真实的 uniform-S 依赖

以下路径均相对 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa/`。

| 模块/源码位置 | 依赖 S 的行为 | 对泛化的影响 |
|---|---|---|
| `decode_indexer.py:407`、`:463`、`:569`、`:691`、`:711`、`:977`、`:1199`、`:1503`～`:1514` | query//S 找请求；query_count//S 扫描请求 | 改 query 数必须同步改变所有生产者、消费者及 Top-K merge；mixed query 不能用固定除数 |
| `decode_indexer.py:589`～`:633`、`:679`～`:703` | 每组 G 个 query 共用 Key，coefficients/通信区 shape 依赖 G，循环按 T//G | 组不能跨请求；尾组必须有明确定义，不能简单用 floor 丢弃 |
| `decode_indexer.py:967`～`:1028`、`:1180` | balance_leaves 长档每个 AIV 负责 G//2 个完整 query | **奇数 G 丢最后一条 query**；S5 整请求分组会只处理4条 |
| `decode_indexer.py:417`～`:429`、`:463`～`:470`、`:1625`～`:1642` | merge 依据长短档/分组策略决定 leaf 划分和每 query 一根或两根 Top-K root | 改分组必须同时更新 merge，不能只改 score 端 G 或关闭 balance_leaves |
| `decode_sparse_attn_csa.py:266`、`:300`、`:487` | 有效块规划、bias、QK/PV 均用 token//S 找页表和 seq_lens | uniform 其他 S 需新参数/特化；mixed 需要边界映射 |
| `decode_sparse_attn_csa.py:136`、`:354`～`:368` | RoPE tile 行数等于 S；原 S6 无请求尾块 | 需重新生成 tile/尾块；它是可调整的局部性能策略，不应反过来限定整个接口 |
| `compact_metadata.py:26`～`:33` | offsets 生产者已经读取每请求 begin/end | 这一部分本来支持不同 query 长度的边界计算，可复用 |
| `compact_metadata.py:52` | compact RoPE 消费者仍用 token//DECODE_SEQ | 生产者能处理 ragged，不代表消费者能处理；必须改请求定位 |
| `decode_compressor_ratio4.py:176`～`:197`、`:322`～`:336` | pool/commit 用 s_dim=T//B、token=request*s_dim+local | 可以表达其他 **uniform** S，不能表达 mixed S |
| `decode_compressor_ratio4.py:405` | cache 写入用 token//S | 需随请求映射一起改 |
| `decode_indexer_compressor.py:183`～`:196`、`:270` | 同样由 T//B 得到 uniform 的 pool/commit 范围 | mixed S 必须改为 begin/end |
| `decode_indexer_compressor.py:112`、`:293`～`:301`、`:475`～`:489`、`:518`～`:531` | 每请求 boundary 行容量 ceil(S/4)，first_pos=request*S，local_token<S | 支持其他 S 需容量/地址/有效行一起变化，mixed 则需逐请求真实边界 |

纯 token 行的 HC_pre、HC_post、Q/QR/KV 投影、RoPE 算术、O 投影多数按 T 分块，
数学本身不需要知道每请求有几条 query。例：`hc_pre.py:82`、`qkv_proj_rope.py:397`、`:531`、`:729`。
这些部分可作为泛化的复用主体，但仍要核对尾块与 workspace 容量，不能直接宣称所有 T 已验证。

## 3. 不能直接改 S 的具体反例

反例只代入当前源码的地址/分配公式，不执行数学或设备：

1. **保留冻结 //6，输入 uniform [5,5]。**
   query index5 已属于请求1，但 //6 得到请求0，使用错误长度和页表。
2. **全局把 S 改5，却保留 pair 分组。**
   连续 pair [4,5] 横跨两个请求；源码只从该组首 query 求一次 batch_idx，
   因此共用 Key/页表不成立。T 为奇数时 T//2 还会遗漏最后一行。
3. **改成整个 S5 请求为一组，却保留长档 balanced 双 AIV 分工。**
   两个 AIV 各处理 floor(5/2)=2 行，覆盖0～3，遗漏第4行。
   G1 在同一分支会覆盖零行，所以“先退成单 query”也不能保留此分支不动。
4. **mixed [2,4,6]，使用 pool 的 s_dim=T//B=4。**
   token2、3、6、7 被分给错误请求；与是否有 device query_start_loc 无关，消费者必须真正使用它。
   同为 T6 的 [2,2,2] 与 [4,2] 也不能靠 T 单独描述请求语义。
5. **仅令 config 的 S=5、B容量仍64。**
   DECODE_TOKENS=320，O 投影现有容量按16行对齐仍为320，
   不满足 `decode_o_proj.py:168` 的 128-row 整除断言。
   S1 对应64也不满足；不能假设改全局常量后模块一定能加载。

容量、数学策略、query 语义必须分开：token 容量不应被简单绑成 B上限×固定 S，
而应根据各阶段对齐/最大工作量配置并核对共用布局。

## 4. 为什么必须从符号源码重新生成

冻结文件 `F/references/symbolic_ir/10_after_OutlineIncoreScopes.py` 已将 S6 编译进来：

- 第520/545行：attention 请求定位 `//6`。
- 第746～750行：indexer cache 写回 `request*6`、`local_token<6`。
- 第942～944行：coefficient 分组 `//6`、`*6`。
- 第1614、1673～1702行：请求扫描/组循环/请求定位固定6，Cube query tile 为 `[384,128]`。
- 第2877～2907行：双 AIV 分组已经变为每 lane3条 query。
- 第13659/13689/13758行：compressor 仍保留 T//B，但其他消费者并未同步变动态。

`F/generated/core_bodies_ir.py` 同样保留 `//6`，第6281～6297行是 `[6,64]` RoPE tile，
第6405行是 QK/PV 的 `qk_t//6`。
AST 机械检索记录了逻辑 IR 80 处、核体86处以 literal6 为直接操作数的算术。
这只是定位清单，**不是80/86个独立bug**；某些6是分片倍数或其他调度参数，不能批量替换。

新的 query 策略会改变类型 shape、constexpr 展开、AIV分工、通信区域、Top-K root布局及事件路径。
必须在独立源码/配置上下文从数学源码重新 symbolic lowering，重新提取 logical IR 与 core catalog，
再生成 SPMD 的依赖和调度。仅改 `build_variant` 的 S6 guard、描述符或生成文件中的数字均不成立。
现有 S6 静态归档与设备证据应完整保留；新目录独立生成，不能让其他会话的正在运行任务加载变化中的数学模块。

## 5. 可复用的 Native 边界 metadata

Native builder 本来保留完整 TND 请求边界：
`vllm_ascend/attention/dsa_v1.py:950` 获取 query_start_loc，`:974`～`:980`
按相邻边界计算每请求 query 数、max_seqlen_q 和 start_pos，`:1156`～`:1163`
把边界装入 decode metadata。
本 PTO adapter 已在 `native_adapter.py:318`～`:320` 传入 compressed/indexer 的原 device query_start_loc，
但目前主要用于 compact offsets，没有覆盖 dense query→request 的所有消费者。

因此 mixed query 的必要信息不必重新在 Host 造一套。可以在确认两组边界都对应同一原始 token 顺序后，
复用其中一份已有 device bounds，传到实际需要的 Indexer、attention 和 compressor 消费者。
这说明数据来源可复用；不等于当前 PTO 核或整条 Native 路径已经由本次验证支持 mixed query。

## 6. 最小安全泛化路径

1. **把接口语义与性能策略分开。** 显式描述 token 总数 T、请求数 B、uniform S 或 device query bounds、
   KV长度/容量；调度策略按工作量选择。七个 S6 点保留为回归点，不作为支持白名单。
2. **先建立 uniform 的多 S 数学模板。** 使用明确的 query constexpr/工厂配置传播到相关模块，
   不修改共享全局 `DECODE_SEQ`；对 S5 等奇数同时解决组尾、AIV不等宽query分配及输出root合并合同。
   分组 G 必须保证每组属于同一请求，未整除部分有有效mask或专门尾组。
   并行 lane 可以不等宽；关键是每条有效 query 恰好处理一次、数据同步双方协议一致。
3. **独立配置容量。** 核对 O投影128行对齐、Indexer coefficient组容量、通信缓冲、boundary压缩行容量；
   不能让 S变化隐式打破固定容量断言或 stage 间共享layout。
4. **重新 symbolic lowering 后重建单SPMD。** 复用 tokenwise 数学、缓存可见性协议、源码DAG构建和区间入口的框架；
   新数学tile/成员所有权与等待细化要重证，S6/B16 的精细依赖或实测成本不得冒用。
5. **mixed query 是另一项实际语义工作。** 把所有 `token//S`、`request*S`、`T//B` 改为bounds派生请求区间；
   Indexer按请求内部划组，compressor按真实begin/end遍历，padding/空请求也使用明确边界。
   可让请求级worker天然知道请求号，或在融合SPMD内部准备映射；不预设必须再增加外部适配kernel。
6. **验证跟随语义。** 先用最小CPU所有权/尾组反例、重新lowering，再做必要的单卡正确性；
   uniform奇数 S、不同请求长度分布、压缩边界与padding是不同检查维度。
   性能再按新 T/query 工作量选点；不能在尚无新数学体时把入口放宽后标“动态shape完成”。

本轮只交付上述边界审计和可复现反例，不实施新 S，不扩大性能或正确性结论。

<!-- SOURCE_END history-document-096 -->

---

<a id="history-document-097"></a>
### 历史原文 097：range_policy_validation_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/range_policy_validation_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/range_policy_validation_v1/README.md)；原文 2,947 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-097 -->
# 区间入口独立复核

本目录仅保存 CPU 策略与生成验证；未调用 NPU、未运行 lowering、未测性能。
检查复用了 `../range_source_audit_v1/binding_enum.json`，没有重新执行 192 次 binder。

## 策略及成本匹配

首次记录为 `policy_checks.json` / `policy_checks.log`：

- 192 个已记录源码绑定点，与区间入口的长度族、query 分组、投影族全部一致。
- 6 个长度端点 × B1～64，共 384 次分类检查通过。
- 七个回归点的调度、QK 索引缓存、行依赖及等待参数与 `selected_shape_policies.json` 一致。
- 9 个无效类型、batch、token 数、query S 和长度输入均拒绝。
- 七个成本文件准确命中；逐个修改全部 56 个 root 的形状，共 392 个变异均不命中。
  另外 14 个增删 root、7 个历史错配均不命中。没有把邻近档位样本称为当前 shape 实测。
- 80 个非默认手工实验策略均拒绝。

## 发现并修复的 CLI 漏洞

首次检查发现：仅比较 `args` 与 parser 默认值无法识别显式默认参数。
例如 `--range-policy --schedule-policy critical_path` 被接受后会静默改为
`earliest_ready`，显式 `--wait-backoff-ticks 0` 会静默改为 100。
原始 `policy_checks.json` 因此保留 `passed=false`，这是当时真实结果。

Root 已在 `range_policy.configure_device_arguments` 中增加 `argv` 参数，
通过 parser 识别显式 action；不传时使用 `sys.argv[1:]`。
修复后只复测相关反例，见 `cli_fix_recheck.json` / `.log`：
四个原反例、两种 `--option=value` 写法，以及实际默认 `sys.argv` 路径均正确拒绝；
普通区间入口仍通过。没有重复完整成本与分类检查。
`validate_policy.py` 随后适配了新 argv 接口供未来复测，历史首次结果未覆盖。

## 非七点 B12/H65536 完整生成

`prepare_b12.py` 调用真实 `build_range_variant`，约 15.49 秒完成：

- B12、S6、T72、历史 H65536、本轮长度 L65542。
- 策略为 long_cube / request_s6，earliest_ready；QK 索引缓存启用，行依赖细化关闭。
- 借用同构参考 h131072_b8，但实际 block、参数、视图及依赖重新生成。
- 没有完全相同 shape/历史的成本样本，`cost_profile=null`，未冒用邻近实测。
- 292 个 Tensor 视图有界，工作区互不重叠，等待表重建通过。
- 3 个种子 × 3 个代际的进度逻辑模拟全部通过；依赖偏序保持，新增全局屏障为 0。

摘要为 `b12_generation_result.json`，完整源码及参数在
`b12_h65536/variants/h65536_b12/`；调度绑定在对应 `bindings/`。

该输入是 CPU 描述符 fixture，保留 B16 长历史的物理页分配，调整批维与 compact 容量。
它不是已经构造并验证的真实设备 batch，也没有验证连续 decode、Scalar 缓存一致性、
硬件同步、精度或性能。整个区间的设备验证状态仍明确为 false。

<!-- SOURCE_END history-document-097 -->

---

<a id="history-document-098"></a>
### 历史原文 098：range_source_audit_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/range_source_audit_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/range_source_audit_v1/README.md)；原文 6,881 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-098 -->
# CSA 分档原数学边界与参考覆盖审计

本审计只读原编排、原核体和现有形状绑定器，未修改生产或依赖仓，未使用 NPU。
`audit_ranges.py` 用已有 CPU 编排模拟器，枚举 S6、B1～64 和三个长度策略代表值，
核对逻辑调用签名及现有参考与新描述符的全部 Tensor 包围区间重叠关系。
共 192 点、约 99 秒；结果为 `binding_enum.json`，原始日志为 `run.log`。

**下面是当前数学实现的兼容边界，不是未来最优性能分档。**
分档应覆盖区间并选择适合工作量的策略；七档只是回归点。
HC_pre 的经验只提供这一设计目的，不能把它的 T96 或其他阈值照搬给 CSA。
本审计中出现的 T96 是 CSA 冻结源码已有输出投影分支，保留它是为了标注当前行为。

## 长度定义及原分支

这里 `L=max_seq_len=max(kv_seq_lens)` 是本轮总可见长度上界，
`H` 是测试 fixture 的历史长度；当前 S6 用例中 `L=H+6`。
例如历史 8192 的用例实际传入 L8198，不能把 H8192 当作分支精确边界。
`C=floor(L/4)` 是压缩后候选数量上界。

| L 区间（当前入口范围内） | C 条件 | 原 Indexer 策略 |
|---|---|---|
| 6～8191 | C<2048 | Vector leaf + 单叶发布 |
| 8192～32771 | 2048≤C≤8192 | 短历史 Cube；pair/request 按波次选择 |
| 32772～131200 | C>8192 | 长历史 Cube；B<4 用 pair，B≥4 用 S6 request |

来源：冻结 `references/symbolic_ir/10_after_OutlineIncoreScopes.py`
第 13954～13958 行、第 14078～14091 行、第 14240～14249 行。
生产源码对应 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_indexer.py`
第 1380～1446 行。入口允许的 L 上限 131200 来自 `shape_specialization.py:458`，
它不是性能最优边界，也不是由本次证明的设备缓存容量。

中间长度的选择公式：

```python
T = B * 6
pair_waves = (T + 46) // 48
request_waves = (T + 138) // 144
use_request = T >= 144 and request_waves * 24 <= pair_waves * 10
```

因此 B1～64 的顺序是：pair 1～23，request 24，pair 25～32，
request 33～48，pair 49～56，request 57～64。
不能简化为“B≥24 一律 request”。它表达 24 个 worker 的工作波次数及额外工作量约束。

## 当前调用图的连续区间与 anchor

原 QR 与 Q 主投影还在 T<128 / T≥128 切换核体，S6 对应 B≤21 / B≥22，
见冻结 IR 第 13438、13572 行。
输出投影分别在 T≤32、T==96、其余 T≤96 和 T>96 选择核体，
见第 14350、14446、14542 行。
将这些与 Indexer 条件求交，得到以下最小连续调用签名区间。
“有参考”只表示存在相同调用图并通过本次 CPU alias 检查，不代表该区间已在设备验证。

| L 区间 | B 区间 | 现有参考（anchor） | 状态 |
|---|---|---|---|
| 6～8191 | 1～5 | 无 | 缺参考 |
| 6～8191 | 6～15 | 无 | 缺参考 |
| 6～8191 | 16 | 无 | 缺参考 |
| 6～8191 | 17～21 | 无 | 缺参考 |
| 6～8191 | 22～64 | 无 | 缺参考 |
| 8192～32771 | 1～5 | 无 | 缺参考 |
| 8192～32771 | 6～15 | 无 | 缺参考 |
| 8192～32771 | 16 | h8192_b16 | 有参考 |
| 8192～32771 | 17～21 | 无 | 缺参考 |
| 8192～32771 | 22～23 | h8192_b32 | 有参考 |
| 8192～32771 | 24 | h8192_b24 | 有参考 |
| 8192～32771 | 25～32 | h8192_b32 | 有参考 |
| 8192～32771 | 33～48 | h8192_b24 | 有参考 |
| 8192～32771 | 49～56 | h8192_b32 | 有参考 |
| 8192～32771 | 57～64 | h8192_b24 | 有参考 |
| 32772～131200 | 1～3 | 无 | 缺参考 |
| 32772～131200 | 4～5 | h131072_b4 | 有参考 |
| 32772～131200 | 6～15 | h131072_b8 | 有参考 |
| 32772～131200 | 16 | h131072_b16 | 有参考 |
| 32772～131200 | 17～21 | 无 | 缺参考 |
| 32772～131200 | 22～64 | h131072_b24 | 有参考 |

总计 21 个连续矩形、17 种调用签名；后者中 7 种已有参考，10 种缺参考。
192 个代表点中 100 点有参考且 alias 检查通过，92 点缺参考，0 点有参考而 alias 失败。
长度区间来自源码的全部相关标量条件，而不是逐个枚举约十三万个长度所得。
同一区间中的页表、缓存容量、位置及设备数据合法性仍由实际调用约束决定。

## anchor 之外仍需保留的动态计算

`shape_specialization.py:65` 的 `choose_anchor` 按调用签名寻找参考并选最近 token 数，
不是七个点白名单。第 80 行的 `check_alias_pattern` 对参数视图包围区间做成对重叠检查。
第 175 行的 `rebind_schedule` 重新设置 block 数、逻辑分片步长与等待表；
它检查每个逻辑分片恰好一次以及 MIX 配对，再验证通知/等待保持依赖偏序。
本次仅调用绑定器和 alias 检查，**没有为全部 192 点运行 rebind、lowering 或设备测试**。

调用图相同也不意味着所有核内 tile 或循环相同。例如：

- 冻结 `generated/core_bodies_ir.py:681` 的 Q dequant head_tile 在 B16/32/48 为 16，
  B64 为 64，其余 B 为 4；尾块仍由实际 T 决定。
- 生产 `decode_indexer.py:364` 的 `indexer_long_leaf_plan` 对 query_groups=4/8/16
  调整叶数倍数，使长历史工作分配尽量填满 24 个 worker；它还读本轮长度计算分片。
- 同一二进制中实际读取原 device 的长度、位置、页表，计算访问范围和尾块；
  超出编排策略或容量则需要选择相应模板，不能把跨策略输入强行塞进旧图。

这些是区间策略内部的形状/数据计算，可保留在内核；无须为了每个叶数重新添加一个 Host 档位。

## 给区间入口的建议及证据界限

1. 入口分别输出“数学策略区间”“当前可绑定参考”“设备验证状态”和“调度性能选择”。
   不把“存在 anchor”直接标成设备支持，也不把未测点自动回退成不支持整个区间。
2. 先按 L 区间及请求工作波次定位数学策略，按实际 shape JIT 重绑参数/工作量；
   保留已有 descriptor/alias/MIX/依赖偏序检查。
3. 缺参考的 10 类应明确展示为待补。选取一个新点建立某类参考后，
   仍应核对整类区间及边界，不能仅把该点加进白名单。
4. 原有七档沿用已有设备与性能证据；新点先编译与必要边界验证，不重跑全七档。
5. 本次描述符 fixture 保留 B16 的物理 cache 分配，仅改变批维、token 维与 compact 容量。
   这适合纯绑定审计，不能用于声称 B64/长历史的实际物理缓存已充足；
   真机必须使用真实分配、slot 和页表。
6. 当前 `build_variant` 第 455～456 行仍要求 S6、T≤384。
   此处长度区间是历史/可见 KV 长度，不能据此声称本轮 query seqlen=S5 或其他 S 已支持。

所有新增文件都在本目录内。未改 F、生产代码、依赖仓或共享安装，未测设备性能。

<!-- SOURCE_END history-document-098 -->

---

<a id="history-document-099"></a>
### 历史原文 099：seven_stage_opportunities_v1/README.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/seven_stage_opportunities_v1/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/seven_stage_opportunities_v1/README.md)；原文 10,938 字节。本会话动态专项实现、失败尝试或静态审计记录，全文保留。

<!-- SOURCE_BEGIN history-document-099 -->
# 七档阶段成本与新的数学／访存候选

2026-10-17。目标按用户最新澄清：**每档完整单 SPMD L1 ≤ 同轮原动态 PTO L1 × 0.70**。本次仅整理历史数据、枚举输出所有权与逻辑流量，并编译一个独立 O-A 最小探针；没有新设备测试，没有改变任何现选或共享 driver。

## 现有差距

每侧均重新读取同轮三个无插桩 L1 窗口，取中位数；完整首尾包含等待和空隙，不含外围调度。单位 μs：

| 历史长度 | B/S | 原动态 | 单 SPMD | 动态×0.70 | 尚需缩短 |
|---|---|---:|---:|---:|---:|
| 128K | 4/6 | 582.82 | 541.92 | 407.97 | 133.95 |
| 128K | 8/6 | 679.84 | 641.56 | 475.89 | 165.67 |
| 128K | 16/6 | 905.08 | 807.16 | 633.56 | 173.60 |
| 128K | 24/6 | 1169.46 | 1121.12 | 818.62 | 302.50 |
| 8K | 16/6 | 680.02 | 621.96 | 476.01 | 145.95 |
| 8K | 24/6 | 861.62 | 836.74 | 603.13 | 233.61 |
| 8K | 32/6 | 972.72 | 905.72 | 680.90 | 224.82 |

当前没有一档达标。后续测试必须重新取同轮参考，不能把本表固定成未来测试的分母。

## 能从历史看到什么

只对四个重点档读取原动态 task 泳道。下表是各类成员的 kernel-duration 中位数，**不是该类对整层关键路径的独占贡献**；MIX 时长包含其内部等待。单 SPMD 四档的 L1 只有融合首尾，不能从该文件直接拆解内部各阶段。

| 场景 | Indexer AIV | QK/PV AIV | O-A成员 | O-B反量化+HCpost成员 |
|---|---:|---:|---:|---:|
| 128K B4 | 75.86 | 51.05 | 34.39 | 24.61 |
| 128K B8 | 121.49 | 94.50 | 30.45 | 25.46 |
| 8K B16 | 35.29 | 130.19 | 30.92 | 26.28 |
| 8K B32 | 61.57 | 240.02 | 39.12 | 36.16 |

动态 O-A 八组的完整时间区间（含组间排队）分别约105～106、94、96～100、123～125 μs，不能将单个成员中位数误当成整段只有30～40 μs。Indexer 在长B4/B8值得重视，短B16/B32则明显缩短，而QK/PV随本轮query数量增长。

`evidence.json` 保存全部七档窗口、四档每个kernel的样本数/中位数/范围、各类阶段在每个窗口中的开始和结束，以及原始 trace 路径。没有按完成顺序推测逻辑worker；四档未使用逐逻辑成员成本。B16/128K已有的阶段和内部等待探针只能作为设计提示，不能代替这些档位的融合内测量。

因此，这一轮不能把融合后与动态的差额归因为某一项编译器开销。两侧首尾只能证明整层改善；原动态阶段成本只能帮助选候选。

## 候选一：压缩 KV 的独立行 Tile 软件流水

优先针对短B32，其次短B16；同样可以覆盖长档。修改位置为冻结参考：

- `../csa_static_spmd_feasibility_20261015/reference_operator_source/deepseek_v4_flash_csa/decode_sparse_attn_csa.py:515`，压缩cache分支。
- 同源生产文件 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa/decode_sparse_attn_csa.py` 的对应段。

现在每个AIV每块依次执行64次 `qk_kv_half = gather_row(qk_kv_half, ..., [row,0], ..., [1,512])`。输出Tile沿循环传递，所有随机行都以同一目的Tile为更新对象。方案是保留原页表/Top-K索引和输出顺序，把随机行读取先放到有限个独立 `[1,512] BF16` Tile中，在组内明确组织加载、消费和回收，最后按原行号assemble回64行Tile。只在这个局部加载区域建立软件流水，不跨越现有KV/Score/Prob/PV的FFTS边界。

该方案与以下已失败方向不同：

- 不是把512维全部标量元素交给 `MGATHER`；那一版已验证无收益。
- 不是把64行拆成两次32行串行准备，也不是重复全有效块快路径。
- 不改变AIV0/AIV1职责，不修改数学、有效块策略或页表布局。
- 不删除PIPE_ALL或缓存维护。必须让编译器能证明独立目的Tile的生命周期。

收益依据：短B32 QK/PV成员约240 μs；长B16旧分段探针显示KV准备区间是较大的一段（中位58.73 μs/20块），但其中含提交和异步等待，不能按比例承诺缩短多少。候选要验证的假设是“循环携带的同目的Tile更新限制了MTE2提交并行”，目前尚未证明全部64行真的在硬件上串行。

精度风险低于重写softmax：复制同一BF16行，不改顺序及后续算术。主要风险是负索引清零、尾块、页表交换时的动态地址、独立Tile复用过早、UB容量和额外assemble开销。实际代码可能仍被PTOAS保守同步串行化。因此先做最小CPU代码与事件审计；没有提交/消费重叠证据便不值得上卡。

## 候选二：O-A双输出Tile复用，按工作区选择配对轴

修改位置：冻结参考 `decode_o_proj.py:179` 的 `_proj_a_mm_nz`，以及生成核 `generated/core_bodies_ir.py` 的 `proj_a_mm*`。本次已生成独立T192最小探针，下面说明真实能力和限制。

思路是一个worker保留两个独立FP32累加器，共用输入的一侧：

- 小／中T：同一行块处理相邻两个N128，复用L1输入A；两个N128数学形状不合成N256。
- 大T：同一N128处理两个行块，复用同一L1权重B。选择两行块大小时受L0C和Left容量约束。例如T192用M96+M96。
- 配对轴和块大小由T、原分支和硬件容量决定，历史长度仅影响其前驱依赖；不借用HC_pre的T96阈值。

**关键更正：原大T/N256分支的实际内层是K64，小T/N128才是K128。** 每个候选必须分别保持原分支的K方向归约步长和顺序。不能为方便统一写成K128。本次T192探针为K64，0、64、…、4032依次累加。

与旧实验的区别：

| 已有实验 | 本候选区别 |
|---|---|
| `oproj_n256` | 旧版扩大单Acc的N，减少逻辑列分片。本版是两条独立N128累加链共享L1操作数。大T原本已N256，改为两个M块共享B，不再次扫描N256参数。 |
| `oproj_k512` | 旧版增加单输出的K预取块。本版仍K256搬运，只减少跨输出重复读同一A或B。 |
| `oproj_phase_loop` | 旧版把八个group的代码段整理为循环，数学和数据读量未减。本版改变tile复用和逻辑读量。 |
| `qproj_l1_columns` | 那是INT8主Q、20worker、K1024。本版是BF16 O-A、K4096、分组输出；不能沿用其精度证明或性能结果。 |

输出所有权和有效行逻辑GM请求量已用Python枚举。每个T×1024输出元素恰写一次；以下包含八个group、仅统计输入A与权重B，不统计输出，也没有把逻辑访问直接当成HBM访问：

| T | 配对轴 | 原A / B（MiB） | 候选A / B（MiB） | 合计变化 |
|---|---|---|---|---|
| 24 | N | 12 / 64 | 6 / 64 | 76→70 |
| 48 | N | 24 / 64 | 12 / 64 | 88→76 |
| 96 | N | 48 / 64 | 24 / 64 | 112→88 |
| 144 | M | 36 / 128 | 72 / 64 | 164→136 |
| 192 | M | 48 / 128 | 96 / 64 | 176→160 |

大T的A访问反而增加；它换取B的重复加载减少。L2命中、实际DMApadding、核并行度和双Acc流水可能抵消收益。没有用这些MiB推算加速比，更不能据此承诺整层30%。

### 独立CPU探针结果

`oa_pair_m_t192_v2.py` 只实现一个O-A group的T192、M96+M96、N128：8个worker，每次K256权重load供两行块使用，每行块一条独立K64链。没有融合到CSA，也没有生成生产入口。

- v1直接在JIT根参数写物理TensorView NZ，前端明确拒绝；保留日志。v2使用逻辑 `[4096,1024]` 和普通NZ布局，由既有lowering转换，未改编译器。
- v2完整CPU/PTOAS代码生成通过。这里只证明探针自己的单group NZ GlobalTensor保持，**没有证明与真实O-A跨8组的根TensorView/stride/绑定等价**；两个Acc区间为0..49152与49152..98304，无重叠；两个最终输出store覆盖0..95和96..191行。
- 实际内存峰值：Mat **327680 B**、Left **49152 B**、Right **65536 B**、Acc **98304 B**。均在该独立核硬件容量内，不能外推整层融合后的规划。
- 权重load在外层pipeline二倍展开后为两处，每个K256仍只load一次、供两M块使用。两条链的64个K64偏移保持升序。
- **尚存流水风险**：MemoryReuse明确报告两个Left pipeline group请求深度2但只能保留1份，会串行化部分阶段。编译通过不等于实现预期重叠。
- v3尝试把内层四次K64显式展开，保留两条链与外层流水，反而导致Right需求131072 B超过65536 B，在CPU被拒绝；没有上卡。该失败不隐藏，不继续无依据扩展尝试。

因此当前建议是保留v2作为可编译的形状/资源可行性原型，先解决或量化Left流水退化，再决定是否集成和测卡。没有足够依据直接上线或认定它优于现选。

后续实际移植必须在原 `proj_a_mm_2` body上保留全部入参及TensorView绑定：`wo_a`为跨8组NZ根视图，继续用原group参数及物理NZ偏移，不能用本探针的单group根签名替换。保留 `row_base_o`、`out_col_g` 和原 `o_r_pad` stride，只修改两个行块和N128索引、加载与双Acc循环；先逐个验证group0..7与两条输出矩形的地址，再重建逻辑worker到quant消费者的依赖。还需覆盖M尾块和未写保护区。大T改变M/N分块有浮点实现差异风险，即使K64顺序保持也必须正式权重检查输出；Top-K和cache/state应不变。

## 取舍

两个候选都针对实际数据搬运／局部流水，而不是再扫全局排程参数。QK候选更直接对应短大batch的主要阶段；O-A候选跨历史区间，但现有CPU结果已暴露资源代价。

本轮也核对了HCpost的20个gate Scalar读取和每行8个scale读取，但未把它作为第三个主要优化：目前整段只有约25～39 μs，即使减少其中一部分也远不足以单独填补134～303 μs缺口。相关源码读点保存在 `hc_post_source_reads.json`，没有将读点数等同等待耗时。

已有小模型收益、上述逻辑流量和动态task时间都不能证明30%目标可达；也不能凭未找到候选声称目标不可达。本报告仅给下一步可复现且与已关闭方向不同的具体实现路径。

## 文件与复现

- `analyze.py` / `evidence.json`：历史七档及四档阶段读回。
- `paired_output_probe.py` / `paired_output_contract.json`：所有权与逻辑流量；生成v1原型。
- `oa_pair_m_t192_v2.py`：已CPU通过的独立原型；v1/v3及各自失败产物保留。
- `audit_cpu_probe.py` / `oa_pair_m_cpu_audit.json`：实际内存、NZ、两Acc和归约链检查。
- `oa_pair_m_cpu_v2/kernel/`：CPU生成物；没有设备二进制性能或精度结论。

历史数据读回和CPU审计：

```bash
python seven_stage_opportunities_v1/analyze.py
python seven_stage_opportunities_v1/paired_output_probe.py
python seven_stage_opportunities_v1/audit_cpu_probe.py
```

CPU编译使用本会话已有独立PyPTO/Simpler环境与 `bootstrap_isolated.py` / `compile_jit_entry.py`，参数为 `--memory-planner PYPTO --kernel-mode`。过程未修改共享安装、生产、冻结参考或其他实验helper。

<!-- SOURCE_END history-document-099 -->

---

<a id="handoff-history"></a>
## 交接、旧目标与工作区历史

- [100 · csa_multishape_spmd_20261016/GOAL_BEFORE_30_PERCENT_20261017.md](#history-document-100)
- [101 · csa_multishape_spmd_20261016/LOCAL_WORKTREE.md](#history-document-101)
- [102 · csa_multishape_spmd_20261016/NEXT_STEP.md](#history-document-102)
- [103 · csa_multishape_spmd_20261016/NEXT_STEP_BEFORE_RANGE_ENTRY_20261017.md](#history-document-103)
- [104 · csa_multishape_spmd_20261016/NEXT_STEP_before_20261017_123632.md](#history-document-104)
- [105 · csa_multishape_spmd_20261016/NEXT_STEP_before_20261017_132806.md](#history-document-105)
- [106 · csa_multishape_spmd_20261016/WORKSPACE_MIGRATION_20261017.md](#history-document-106)
- [107 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0544.md](#history-document-107)
- [108 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0550.md](#history-document-108)
- [109 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0638.md](#history-document-109)
- [110 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0732.md](#history-document-110)
- [111 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0800.md](#history-document-111)
- [112 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0849.md](#history-document-112)
- [113 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0926.md](#history-document-113)
- [114 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0958.md](#history-document-114)
- [115 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_1045.md](#history-document-115)
- [116 · csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_1125.md](#history-document-116)
- [117 · csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md](#history-document-117)

<a id="history-document-100"></a>
### 历史原文 100：GOAL_BEFORE_30_PERCENT_20261017.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/GOAL_BEFORE_30_PERCENT_20261017.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_BEFORE_30_PERCENT_20261017.md)；原文 2,971 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-100 -->
# CSA 单 SPMD：目标与分档原则

## 原目标保持不变

先把 SPMD 的 CSA 改成动态 shape（允许分多个档位支持不同的 batch_size 与 seqlen），
然后优化性能，使 128K / B16 的完整 CSA L1 incore 首尾耗时低于 700 μs。
计时包含内部等待和阶段间空隙，排除外围 AICPU / orchestrator 首尾开销。
当前目标仍在进行，不能以局部阶段计时、CPU 模型或插桩结果宣称达标。

## 用户于 2026-10-17 明确的分档原则

分档的目的，是覆盖动态输入范围，并让不同规模的工作负载采用合适的计算和调度策略，
避免一套配置在部分场景下低效。HC_pre 是设计思路的参考，其阈值和 T=96 专用分支
都不是 CSA 必须遵守的规则。

1. 按 CSA 自身的工作量特征划分区间。batch_size 影响请求间并行度；历史 seqlen
   影响扫描量和历史分片数量；本轮 query 数和总 token 数影响投影与 attention 分块。
   必须明确 seqlen 指历史长度、包含本轮 token 的总长度，还是本轮 query 长度。
2. 每档选择适合的 tile、分核、历史切分、流水深度及合并方式。边界由 CSA 的计算、
   访存、资源容量及实测收益决定，不能照搬 HC_pre，也不能仅按七个样例点写分支。
3. 档内仍动态处理实际输入。长度、位置、slot、页表和有效范围继续来自当前 device
   Tensor；据此计算分片、地址、循环次数和尾块。不得把测试输入数值固化进实现。
4. 同一档复用实现策略，不要求同一份二进制覆盖不同 batch。允许 JIT 按具体 shape
   分别编译并缓存；实际形状变化与策略变化应分别处理。
5. 七个现有规格是性能回归点，不是最终支持白名单。区间策略还要检查代表点和边界，
   不能以七点通过宣称整个区间正确或性能良好。
6. 只有 CSA 自身有明确依据和收益时才保留精确形状特化，不能因 HC_pre 有专用分支
   就机械复制。优先改善可复用的策略，避免只优化孤立测试点。
7. 超出数学策略或工作区容量时应选择合适的变体。选择可使用调用方已有的 Host
   描述信息，不为选档新增每步 device 数值回读或同步。

动态 shape 支持和性能目标都必须完成，不能把任一项缩减为更容易通过的替代目标。
现有源码的 S=6 限制、尚未覆盖的分支和验证范围应明确记录；不能把“可选择策略”
等同于“已通过设备验证”，也不能把已测固定 fixture 当作连续 decode 或整网验证。

## 接续约定

本文件是原目标的补充约束，后续入口设计、调优、验证和交接均以此为准。
当前会话的 goal 接口只能更新状态，不能修改已有 objective 文本；因此原 goal 保持
active，本文件及 NEXT_STEP.md 共同保存用户补充要求，不创建替代目标，不假报完成。

<!-- SOURCE_END history-document-100 -->

---

<a id="history-document-101"></a>
### 历史原文 101：LOCAL_WORKTREE.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/LOCAL_WORKTREE.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/LOCAL_WORKTREE.md)；原文 306 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-101 -->
# 当前工作目录

实际源码和完整过程产物：`/data/pyptouser/qinchuanyu/pto-eager/.worktrees/csa-spmd-dynamic-20261016/tests/pypto_test/csa_multishape_spmd_20261016`。

本目录同步脚本、说明及小报告；新二进制和编译临时目录仅留在独立工作目录，不入 Git。

<!-- SOURCE_END history-document-101 -->

---

<a id="history-document-102"></a>
### 历史原文 102：NEXT_STEP.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP.md)；原文 6,690 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-102 -->
# CSA 单 SPMD：当前交接

## 当前工作区

已按用户要求迁至 `vllm-ascend-dsv4-pto-0251rc1`，本目录现在是实际开发目录，不再只是文本镜像。
原 `.worktrees/csa-spmd-dynamic-20261016` 保留作历史留存，不再写入新实验。
GitHub 最新 `a919f119` 已合入本地，合并提交 `643d8039`；最新生产 HC_pre 已包含 `f7ee8dda` 的优化。
完整迁移及轻量 CPU 入口检查见 `WORKSPACE_MIGRATION_20261017.md`。没有为迁移重跑 NPU。

## 生效目标

用户已确认所有七档相对同轮动态分 task PTO **耗时降低至少 30%**，即
`SPMD≤dynamic×0.70`。原 128K/B16<700μs 单点目标被替换。
完整约束见 [GOAL_AND_CONSTRAINTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md)。目标保持 active，尚未达到。
分档须覆盖工作量区间；七点是回归点，不是支持白名单。原始粘贴目标和旧约束均保留。

## 当前功能实现

- `range_policy.py` + `device_compare.py --range-policy` 是新增区间入口，使用 B/T/L 选择数学兼容区间和初始排程，保留七个已测点的策略。L含本轮query，H是历史，当前fixture L=H+6。
- 核体目前仍限 S6、T≤384、L≤131200；fixture B≤40。范围选择不等于区间内所有点已通过设备验证。
- `exact_cost_profile` 只用 B/H/S 和所有root shape一致的成本；其他形状仍可生成，参考成本标注为估计。
- 原有同构参考的路径不变。缺参考时由 `anchor_gap_design_v1/source_schedule.py` 从本轮绑定及IR读写效应构造DAG，处理RAW/WAW/WAR、显式TaskRef/dummy/manual合同。无整图串行兜底，无新增全核屏障。
- CLI 禁止与手工实验策略混用，包括显式传默认值；发现、修复和复查均有证据。

## 已完成验证

- 区间分类192点与原编排一致，384端点检查通过；七档选择保持一致。严格成本与CLI审查见 `range_policy_validation_v1/`。
- 源码DAG 20个代表组合通过schema、分片/MIX配对、视图/工作区、通知协议和源码生成检查。七档恢复原1377对偏序，缺失/新增均0；WAR审计也未增加原偏序以外串行关系。见 `anchor_gap_design_v1/builder_check_v3_war/summary.json`。
- B12/H65536、B17/H65536 通过完整生成和PYPTO lowering。后者确实走source_derived_task_dag，未套用B16等待表。
- B12产物：`range_policy_validation_v1/b12_h65536/variants/h65536_b12/`，lowering在`range_b12_h65536_lowering_v1/`。
- B17产物：`range_b17_h65536_cpu_v1/variants/h65536_b17/`，lowering在`range_b17_h65536_lowering_v1/`。

## 当前设备任务（以实时队列为准）

- `task_20261017_140019_112364329866` → `range_b12_h65536_device_v1`。
- `task_20261017_140020_112371518858` → `range_b17_h65536_device_v1`。

两任务均已exit0，完整exact/图重放/信号代际/保护区PASS。B12 L1=690.70 vs714.66μs，降低3.35%；B17 L1=863.08 vs893.52μs，降低3.41%。两点均未达到30%。原始泳道回读及精度摘要见range_entry_device_results_v1.json。当前无本会话设备任务。

## 性能现状及下一步

七档现选完整L1仍为：128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32=621.96/836.74/905.72μs。历史收益约2.9%～10.8%，均未达到30%。
本轮回读原始L1起止并计算新门槛，见`target_30pct_historical_gap.json`；不是新测。
`performance_acceptance.py`新增独立的单规格30%门槛，功能PASS与性能达标分开写入结果。

两个代表点真机验证已完成，不重复。下一性能方向在`seven_stage_opportunities_v1/`：
按实际O-A分支保留各自K归约顺序，评估双输出tile输入复用；另调查压缩KV随机行加载的软件流水。
这些是新候选，不能拿逻辑GM请求量或CPU模型当实测收益。

主Q后处理head拆分的新逐成员成本模型调查已结束：已证实成本不均，但模型收益仅0.6～2.26μs，
少量额外开销即抵消，暂不生成设备候选。见`observed_member_cost_v1/`；不要重复旧排程扫描。
更多已关闭方向与历史失败见`ATTEMPTS.md`及`NEXT_STEP_BEFORE_RANGE_ENTRY_20261017.md`。

## 持续约束

正式权重固定 `/data/model/DeepSeek-V4-Flash-0731-w8a8`；PTO对PTO，不跑Native forward。
NPU只经`task-submit --device auto`。计时用完整L1 incore首尾，包含内部等待/空隙。
原静态归档保留，不改依赖仓或共享安装。当前单SPMD仍在本实验目录开发，尚未接入生产算子；过程记录直接在当前工作区保存。
不做hash；大产物/二进制保留本地并列路径。静态已push，动态实验尚未commit；本轮未要求commit。

## query 维度后续限制

只读审计确认S6不只是fixture条件：原Indexer按两query配对，S5会跨请求；request分组的
双AIV使用query_group_size//2，奇数S会漏尾query。冻结IR已固化//6，因此不能仅改入口常量。
后续详见`query_shape_support_audit_v1/`；原核体及当前通过的S6路径保留。

## 新补动态metadata任务

`task_20261017_140653_1348347342` → `range_b17_h65536_metadata_v1`，range-policy + metadata-replay-only。同地址/同图H65536→65535→32766→65536，C点L32772为长历史策略下边界，不采性能。已exit0完整PASS：固定地址、每侧一次图捕获、无新编译单元、信号不在变体间重置；四个变体输出/TopK/state及保护区通过。40秒观察超时后原任务正常完成，未重提。当前设备任务均终态。

## 性能候选补充

O-A独立资源探针存在Left流水退化与真实跨8组NZ绑定未证明问题，暂不直接上卡。QK压缩KV行加载独立Tile流水的CPU审查已完成，预期重叠没有实现，当前写法已停止。短B32当前不启用qk_vector_indices，长B8/B16启用，比较时不能把新增缓存组合冒充短档基线。

该CPU候选的最终报告在 `qk_row_load_pipeline_v1/lowering_audit.json`，不再等待子任务或重复扫描同一 stage 参数。

## 用户最新提示：参考 canonical HC_pre

用户询问独立优化hc_pre.py的参考价值，已读指定文件并保存快照和`HC_PRE_REFERENCE_REVIEW.md`。
当前HC流式/三阶段融合候选不是完整移植；特别是旧流式仍用GM交接，而独立HC_pre用aic_gather供数并复用加宽数据做RMS。
后续优先核对这套数据流/分核设计如何融入现有完整SPMD与norm合同，不以旧候选无收益关闭整个方向。
QK独立行Tile stage2候选已CPU证伪预期重叠并停止：基线实际已有连续TLOAD；候选额外加入逐行等待和TMOV。
详见`qk_row_load_pipeline_v1/lowering_audit.json`。不重测或扫该参数。

<!-- SOURCE_END history-document-102 -->

---

<a id="history-document-103"></a>
### 历史原文 103：NEXT_STEP_BEFORE_RANGE_ENTRY_20261017.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP_BEFORE_RANGE_ENTRY_20261017.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP_BEFORE_RANGE_ENTRY_20261017.md)；原文 8,053 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-103 -->
# CSA单SPMD动态档位：当前交接

## 新目标已生效：全七档至少降低 30%

先读 [GOAL_AND_CONSTRAINTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md)。用户已确认 `SPMD ≤ 同轮动态 PTO × 0.70`；原 700 μs 单点门槛已替换。以下旧记录保留历史语境，不再代表当前目标。

当前继续实现按工作量的区间入口、补齐缺少参考的调用分支；并调查七档共同的数学/访存瓶颈。原七档收益约 2.9%～10.8%，没有一档满足新门槛。


**先读 [GOAL_AND_CONSTRAINTS.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/GOAL_AND_CONSTRAINTS.md)。** 用户已明确：分档是按 CSA
工作量划分区间并选择计算/调度策略；七个测试点不是支持白名单，不照搬 HC_pre 的阈值
或 T=96 专用分支。原动态 shape 与 128K/B16 <700μs 目标保持不变。

当前goal active、无预算上限：按batch/历史策略分别编译的动态metadata执行，继续优化128K/B16完整无插桩L1至700μs以下。用户允许像hc_pre按档选择不同分块/分核，不要求单二进制覆盖全部batch。目标未达到，不标complete/blocked/paused。本轮有CPU代码和完整设备验证进展，没有需要用户批准的动作。

## 当前状态

全部CPU和设备任务均已终态，PENDING_JOBS.json已更新，没有待poll任务。
七档仍为S6：128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32=621.96/836.74/905.72μs；另一同进程现选B16为807.50μs。
selected_shape_policies.json没有改变，各档用自己的排程和数学策略；长B8/B16用Vector索引，只有长B16用rowdeps，均cached100。
静态a919f119已push保留；动态实验仍未commit。本轮仅修改N实验目录并镜像过程文本，没有改生产/F/依赖仓或共享安装。

## 本轮结果：不重复

1. 固定物理分核MILP：v1全局模型120秒无整数候选；v2约简约束+最多96对顺序变化得到773.41模型值，无改善；v3限定模型≤750后120秒无候选。不是不可行证明或设备下界，没有上卡。
   weighted_precedence约简24825→20402数学边，242701可达对的最长时间差全部保持，误差0。见FIXED_CORE_MILP_EXPERIMENT.md、源码v1/v2快照和三个结果目录。ortools独立安装失败日志保留，使用已有SciPy/HiGHS，没有改共享环境。
2. RoPE去gather探针：v1把FP32 reshape成256×2，PTOAS因8B行不满足32B对齐拒绝；v2先转置、交换行对、再转回完整CPU通过，但新增搬运且没有设备性能。不要据此改掉PTOAS安全分支。现选QK还有任意页表gather，不能宣称只改RoPE就让整个函数无gather。
3. rope_flat_gather.py合并三个逐行gather，原列索引保持，改绝对位置r*64+c。仅3个循环修改，其他AST完全保持；262144地址组合证明；实际1024/1024/2048 Tile与索引链、同时活跃UB不重叠通过。CPU v1只是命令路径static/误用，未进入编译；v2按variants/路径完整PASS，集成源码AST完全相同。
   task_20261017_131052_534942170 / b16_row_deps_rope_flat_v1 exit0：所有eager/graph/final输出/TopK/六state零差异，30代/保护区PASS；完整L1 819.58/815.10/810.50，中位815.10，同轮动态920.58。未优于现选、未采用、不扩档。--rope-flat-gather默认关闭。见ROPE_GATHER_EXPERIMENT.md、rope_flat_gather_result_v1.json。
4. 全阶段noinline+当前cached100组合：早期默认等待时代1044.54不能直接外推现选，因此补做该组合。CPU独立编译74helper，外层事件只有全部分离且set/wait成对时清理，helper自身事件、FFTS及数据维护不变。实际两个CCEC源码各74处cached100，完整覆盖/外层无遗留事件PASS。审计适配backend的static __aicore__拼写，没有修改编译产物。
   v1 task_20261017_131724_17808923318因另一处旧rowdeps参数保护拒绝，exit1，未执行算子。窄范围开放B16/128K/all stages/compact/noinline/现选rowdeps+QK索引/cached100；CPU解析验证原现选与此组合可用，partial或缺noinline仍拒绝。
   v2 task_20261017_131834_1981138300 / b16_row_deps_outlined_cached_v2 exit0，完整精度/图/30代/保护区和实际源码门禁PASS。L1 807.02/805.34/810.66，中位807.02，同轮动态902.84，约与现选持平，不采用、不扩档。eligible_for_700us_target=false，C++诊断不是正式PyPTO接口。40秒观察超时不等于任务故障，后续正常完成。见OUTLINED_CACHED_WAIT_EXPERIMENT.md及outlined_cached_result_v2.json。

## 新方向，仅完成源码分片枚举

qproj_dequant_ownership.py和qproj_dequant_ownership_cpu_v1.json：
B16/T96原head_tile16时，每个后处理worker读32个N256列块，覆盖全部20个投影worker；不能直接减少父核等待。
若仅将该档后处理head_tile改4，每个worker仍执行16个head×8行数学迭代，输入只涉及8个投影worker，12×64行/head块各有唯一写者。tile8需要16父核。tile1/2也需要8父核，但QK依赖的后处理worker增至48/32；tile4和原16的QK读范围都涉及16个后处理worker。

下一步可先补真正的同步合同，再CPU模型筛选：
- 核对本轮绑定tile_rows=96、tile_base=0、q_proj_i32生产/消费视图完全一致，其他共享临时区和根输入的读写性质；不能只凭allocation名称相同。
- 精确核对后处理完整行/head写回与QK的读范围、完整图偏序、原216条FFTS复用边、各核既有顺序及缓存维护。
- 原数学体head_tile16没有直接删投影依赖的空间；如比较4，要把外层循环次数变化和索引准备开销当真实风险，不能按相同数学迭代数保证同耗时。
- 可以先在CPU估计缩父核后的关键路径，再决定是否值得生成/上卡。当前没有新等待表、没有图模型收益、没有设备候选；不得预记PASS或直接删边。
该方向不同于已测的改投影worker数、主Q整体后移、N128/L0预取；避免重复旧方向。

## 保留的重要限制

- 正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8，PTO对PTO，不跑Native forward；只经task-submit --device auto使用NPU。
- 不修改运行中设备任务导入的driver/helper；上下文入栈后才_resolve_kernel_artifact，实际等待门禁必须检查kernels/，ptoas/是替换前源码。
- 完整exact输出/TopK/六state、图、30代、保护区后才采3窗完整无插桩L1，含等待与空隙，排除外围AICPU。不能用短插桩或Event时间替代700验收。
- 七档同地址动态metadata已验证，不重复。固定fixture恢复不是连续decode或整网；生产自动派发尚未接入。
- 现选QK按head分工8/8/8与16/32/8已完整PASS但885.08/830.46，均未采用；见上次NEXT_STEP快照和QK_ROLE_SPLIT_EXPERIMENT.md。
- 固定局部交换120秒773.41→773.41、主Q后移→788.94、成员三启发式和query两组六模型都无收益；不延长相同排程搜索。历史L1没有已验证logical block映射，不能靠结束顺序猜成本。
- 所有负结果见ATTEMPTS.md，不能因为目标困难或本轮无收益自行暂停/标blocked。新候选必须有新依据，避免重复七档测试。
- 原静态和所有失败产物保留。镜像只增量复制过程文本，超过2MiB和二进制留本地列路径，不做hash，不删除目标文件。


## 最新续接：2026-10-17T13:42:01+08:00

上述“主Q生产者范围CPU核对”的待办已完成：qproj_dequant_dependencies.py和qproj_dequant_dependency_model_v1/result.json证明绑定/写者/偏序；tile16/8/4模型都为771.53，没有收益，未改核体/未上卡，不重复。当前没有本会话新设备任务。

用户现在要求采用hc_pre式范围分档：seqlen区间×batch场景（精确特化须有CSA自身依据），七档仅作为性能回归点。已展示逐档同轮分task vs单SPMD表，来源重新检查保存selected_seven_readback_20261017.json。不要把七个测试点当成最终支持范围，也不要把S6冻结源码误说成已支持S5。下一步按GOAL_AND_CONSTRAINTS.md实现范围策略，并继续700目标。

<!-- SOURCE_END history-document-103 -->

---

<a id="history-document-104"></a>
### 历史原文 104：NEXT_STEP_before_20261017_123632.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP_before_20261017_123632.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP_before_20261017_123632.md)；原文 8,331 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-104 -->
# 接续：2026-10-17 12:08

Goal active：按batch/历史长度策略分别编译、各档调优，128K/B16/S6完整无插桩L1<700μs。
未达到，不暂停、不标blocked。用户允许hc_pre式分档，不要求一份二进制覆盖多batch。
本goal轮有实际进展：完成页基址CPU接口检查；提前KV通知完整设备验证；新双AIV分工从
CPU模型、编译、输出所有权证明到完整设备验证和一次阶段归因。两个性能候选都未采用。
全部任务终态，没有待等待的CPU/NPU任务。旧交接完整归档handoff_snapshots/NEXT_STEP_20261017_1125.md。

## 本轮已完成，不得重复

1. 交接提出的缓存分类已由payload_cache.py实现，旧838.68μs。03:47也复查过，不是新方案。
2. Indexer任意stride参数由JIT显式拒绝；读取已有TestUnsupportedLayoutSlot确认，没有重跑。
   GM双基址先选再gather探针v1未用block index被前端拒绝，是探针写法问题；v2显式worker后
   因不同backing SSA的if返回被拒绝。均CPU、无设备。见INDEXER_PAGE_BASE_SELECT_CPU.md。
3. qk_early_kv.py仅将三个条件144→96，在当前向量化索引路径重新评估旧提前通知策略。
   旧阈值有实测依据，不能称bug。CPU_v1完整PASS，集成入口文本相同；C++2wait/2notify/9PIPE_ALL，
   Score wait→KV notify→softmax→Prob notify→PV wait。设备task_20261017_114240_273482232688
   b16_row_deps_qk_early_kv_v1 exit0，全部exact/图/30代/保护区PASS；L1
   823.38/816.34/833.82，中位823.38，同轮动态905.06。未采用、不扩档、不重复。
   见QK_EARLY_KV_EXPERIMENT.md、qk_early_kv_result_v1.json。

## 新双AIV分工：正确，但当前Tile版本慢

qk_role_split.py/prepare_qk_role_split.py：AIV0准备128行KV，AIV1做64head数学与packed输出。
Cube原样、三槽/prelaunch2原样。两条AIV均消费Score/PV、发送KV/Prob（无数学那条做dummy参与），
先消费前Score再发本KV。无新增GM workspace/Host读取/任务内Scalar信号。
64×512 FP32状态131072B拆16个8×256，softmax和最终发布按8head。math逐head顺序保留设计。
qk_role_split_protocol.py：3328个调度、471867动作、全部1024种两query valid mask×3，加256种
四query随机mask×3代；五种槽复用/事件排空PASS，credit峰值3。模型不证明硬件DMA/缓存语义。

prepare_v1 PASS；cpu_v1拒绝分支外kvbuf变量（ConvertToSSA），未到内存规划。保留source_v1快照。
v2将raw/cmp各自store放回分支，用独立kv_done接循环返回，cpu_v2完整编译PASS。
真实状态地址16384..139264、每份8192B，init/carried/valid相同，16个PV add写回各自槽；
QK显式Vec shape范围到163328（不是完整allocator报告）。C++8rowmax/8rowsum/8最终rowdiv，
两角色4wait/4notify，5PIPE_ALL（比原9少4个，不代表性能收益）。
qk_role_split_codegen_audit.py在设备执行前检查状态地址和两角色事件顺序。

row_dependencies.py新增默认关闭qk_roles参数：源码检查全部八组packed写入只在AIV1，
768行唯一且O-A消费全覆盖；所有O-A等对应AIV1，不沿用旧head半区映射。
旧六项源合同/行证明仍PASS。prepare_v3启用新证明，角色分支AST与编译v2相同，无重复数学编译。
shape_specialization/build与device_compare新增默认关闭--qk-role-split；要求T96、balanced、
qk-vector-indices、row-dependencies；新的128行向量索引已在角色体内，不重复套旧64行变换。

设备task_20261017_115641_29670009014，b16_row_deps_qk_roles_v1 exit0：
eager/graph/final输出、TopK、六cache-state全零差异，30代与保护区全PASS，真实地址/事件及100tick门禁PASS。
L1 887.72/885.08/870.46，中位885.08，同轮动态901.30。未采用、不扩档，不重复此版本。
一次40s观察等待超时后原任务正常exit0，没有kernel超时、没有kill/restart。

诊断task_20261017_120238_306765112993，b16_row_deps_qk_roles_stage_v1 exit0；
warmup0/iters1/windows1，exact/图/4代/保护区PASS。含插桩，不用于700目标。
qk_role_split_stage_comparison_v1.json对历史探针：数学+内部等待+写侧维护中位
Indexer AIC211.03→212.99；QK AIC150.93→206.39、AIV0 157.90→207.90、AIV1 158.12→222.80。
本轮QK AIC/AIV0/AIV1最后结束701.30/703.50/718.38，O-A最早720.94。主要退步在QK/PV。
两轮warmup不同，不精确分摊跨轮差值；不是pipe独占忙时。未证明KV/softmax/PV/pub各自开销。
详细QK_ROLE_SPLIT_EXPERIMENT.md、qk_role_split_result_v1.json；source_v2快照保留当前实现。

## 下一步建议（尚未执行）

可以先CPU分析角色AIV1小Tile带来的提交成本，并评估放大head Tile是否能仍放进UB：
当前softmax从原每AIV1组32head变为AIV1上8组8head，PV状态16块，发布也更细；单核工作双倍。
当前真实状态128KiB、显式shape范围163328B，不能简单把理论剩余空间当作更大Tile一定可行。
若做候选，明确区分softmax、PV状态、发布三个tile；先核对实际UB/别名/循环回边与事件顺序，
只有代码与资源证据支持才上B16，不盲扫全部档。不得改已验证原型或旧结果来掩盖失败。
旧qk_merge_heads（原双AIV各32head）v7中位820.24，不能把已验证的16head状态方案当新方案重跑；
新分工是单AIV64head，内存与工作量不同。若大Tile无合理规划，则关闭此方向，另找结构性收益。

另一个未执行方向是更强的CPU排程求解，但先读MEMBER_SCHEDULE_CPU_EXPERIMENT.md和
QUERY_GROUP_SCHEDULE_FEASIBILITY.md：973成员/877单元的三种启发式、两组query六种模型均已测过；
原模型771.53、最好768.79且等待增加，无设备候选。不能重复这些模型；ortools当前未安装，
本轮未安装新求解器或修改共享环境。成本模型不是硬件下界/性能证明。

## 现选和重要旧记录

SELECTED_SEVEN_RESULTS.md/selected_shape_policies.json，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；8K B16/B24/B32=621.96/836.74/905.72μs。
最新另一轮同进程现选B16 807.50μs。长B8/B16向量索引；仅长B16 row deps；100tick cached wait。
动态metadata同图同地址已通过，未接生产自动派发，不重复。正式权重DeepSeek-V4-Flash-0731-w8a8。

前轮修正编译上下文：任何_resolve_kernel_artifact必须在所有编译扩展入ExitStack后调用，
否则缓存默认TWAIT。COMPILE_CONTEXT_CORRECTION.md：只三项旧结果受影响，已修正重测，
WS814.12/PV820.24/N64单面板ring806.38，全PASS但不采用。两面板Key预取813.42全PASS不采用。
真实CCEC输入kernels/aic/aiv/*.cpp需74个替换wait及100tick；ptoas/*.cpp是替换前来源不能判断。
旧版所有失败/负结果见ATTEMPTS和11:25交接，避免重复。

## 工作范围与命令

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；C=R/vllm-ascend-dsv4-pto-0251rc1。
F为相邻csa_static_spmd_feasibility_20261015，冻结只读。P/S为R/.worktrees中的
pypto-csa-static-fixes-20261016、simpler-csa-core-args-256-20261016。
只改N及canonical过程镜像；不改生产/F/依赖/共享安装。动态N未commit，静态a919f119已push保留。
无子agent、无hash、NPU仅task-submit --device auto；运行中不改其导入的driver/helper。
正式验收完整无插桩L1首尾，保留等待/空隙；exact/输出/TopK/六状态/图/30代/保护区门槛不变。
fixture重置不是连续decode。Native forward不运行。
git用--git-dir=R/.git-repositories/vllm-ascend.git/worktrees/csa-spmd-dynamic-20261016 --work-tree=D。
CPU沿用env+F/bootstrap_isolated.py F/compile_jit_entry.py --memory-planner PYPTO --kernel-mode。
设备run_device_compare.sh 16 131072 新目录 --balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100
--qk-vector-indices --row-dependencies 加候选开关。只有诊断使用--stage-clock与短迭代。
镜像python N/mirror_process_files.py --destination C/tests/pypto_test/csa_multishape_spmd_20261016，
mtime/size增量、≤2MiB白名单文本，大产物仅LOCAL_ARTIFACTS记录，不hash不删。

<!-- SOURCE_END history-document-104 -->

---

<a id="history-document-105"></a>
### 历史原文 105：NEXT_STEP_before_20261017_132806.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP_before_20261017_132806.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/NEXT_STEP_before_20261017_132806.md)；原文 8,089 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-105 -->
# CSA单SPMD动态档位：当前交接

当前goal active、无预算上限：先支持按batch/历史档位分别编译的动态metadata执行，再优化128K/B16至完整无插桩L1小于700μs。
用户明确允许类似hc_pre按档选择不同策略，不要求单二进制覆盖全部batch。目标未达到，不能mark complete。
本轮有代码、CPU编译和完整设备验证进展，不是阻塞回合；没有需要用户批准的操作。

## 当前任务与选择

所有设备与CPU任务均已终态，PENDING_JOBS.json已更新；没有待poll任务。
现选七档仍为S6：128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32=621.96/836.74/905.72μs。另一轮同进程现选B16为807.50μs。
长B8/B16用Vector索引，只有长B16用row dependencies；均100tick cached wait。
同图同地址动态metadata验证已通过，不重复；尚无生产自动派发或连续decode轨迹结论。
静态a919f119已push保留；本动态实验目录仍未commit，不改生产、冻结原型或依赖安装。

## 本轮QK分工扩大Tile：完整PASS，未采用

旧角色版qk_role_split把AIV0用于128行KV、AIV1用于64head数学，8/8/8分块，
完整L1中位885.08，阶段诊断主要慢在QK/PV。该旧版不重复上卡。
本轮把PV状态、softmax、发布三个分块拆开：

1. 状态16/softmax32/发布16：prepare PASS，CPU Vec194048>188416失败，未上卡。
   分配报告显示发布的FP32、BF16及RoPE临时量与128KiB状态同时存在；缩softmax不能消除这个峰值。
   qk_role_tiles_ub_diagnosis_v1.json保留分配详情，qk_role_split_source_h16_s32_p16.py.txt保留源码。
2. 状态16/softmax32/发布8：qk_role_tiles_h16_s32_p8_cpu_v1完整CPU PASS。
   全部AIV Vec分配167552B；8个16×256 FP32状态地址0..114688互斥、原位更新；
   incoming不覆盖状态，旧值alpha乘法与add输出逐槽匹配。
   两个rowmax/rowsum、八个发布rowdiv、4wait/4notify/5PIPE_ALL，顺序检查PASS。
   发布只从状态slice连续8行，Cube、GM接口、逐head运算顺序及原FFTS保持。
3. --qk-role-wide-heads默认关闭，必须与--qk-role-split组合；actual Tile必须是(16,32,8)。
   prepare_qk_role_split新增三个分块参数；行依赖入口只接受已编译的(8,8,8)/(16,32,8)。
   shape_specialization与device_compare接入此开关，全部artifact解析仍位于编译扩展入ExitStack之后。
   CPU integrated版本全部math AST与已编译候选相同，不重复数学编译。

设备task_20261017_122441_342567127888，b16_row_deps_qk_roles_wide_v1，exit0：
eager/graph/final输出、TopK、六cache/state全部零差异，30代进度和全部保护区PASS。
实际状态/事件/Tile与100tick源码门禁PASS。完整无插桩L1
830.46/827.18/832.04，中位830.46；同轮动态899.68/911.08/912.48，中位911.08。
没有优于现选的证据，未采用、不扩档；暂关闭仅靠角色分工/放大head Tile这个方向。
不是连续decode；没有本轮现选配对，不精确归因与旧轮次的差值。
证据qk_role_tiles_result_v1.json、QK_ROLE_SPLIT_EXPERIMENT.md及全部原目录。
旧默认8/8生成AST不变，扩展后的审计也已在旧编译产物通过，不再重复验证。

## 本轮CPU排程：均无收益、没有设备版本

fixed_core_local_schedule.py从现选队列做固定分核相邻交换，与此前member_schedule三种从头贪心不同。
973成员/877单元，保留数学边、216条FFTS复用边、原核组映射；MIX三成员按共同开始建模，
每条资源边仍用对应角色成本。此共同开始尚未转为设备等待，不能直接使用模型计划上卡。
120秒7679次交换评估，149个环被拒绝，13次等长交换；最长路径773.41→773.41，无改善。
原逐成员模型771.53、共同开始模型773.41，均不是设备测量。

额外--defer-main-q-dequant --rounds 0：将48条AIV上的主Q后处理成组移到Indexer后，
图无环但773.41→788.94。Indexer最早开始207.49→191.08，最后结束仍444.24；
merge/plan延后，QK最早465.47→481.00。因此不生成设备代码、不上卡，不继续同一局部搜索。
fixed_core_baseline_critical_chains_v1.json还记录纯数学模型640.92及固定资源模型771.53的一条关键链；
不是硬件下界或700可达证明。详细FIXED_CORE_LOCAL_SCHEDULE_EXPERIMENT.md及两份模型目录。

历史L1 chip_swimlane_records的tuple只有物理核/task/派发序号/时间等，没有已验证的逻辑block映射。
冻结build_schedule_manifest需要严格匹配派发日志；不能用结束顺序猜block_idx。
本轮没有为新成本窗口收集这份日志，也没有构造逐block精细成本。

## 后续可做的方向，尚未执行

继续优化必须提出影响最慢Indexer核组完成时刻或降低关键数学成本的新依据。
当前同核排队的数学图640.92与资源图771.53差额不能全部归因为可消除调度开销。
如继续排程，应同时保护Indexer最慢组的释放、MIX耦合及下游merge资源；
不要再做上述相邻交换、整组主Q后移、三种贪心、两组query六种启发式的重复验证。
更强的约束求解尚未实现/安装；不改变共享环境。若需逐逻辑block成本，先取得真实对应证据。
不得仅凭CPU模型改善宣称性能收益；只有值得验证的候选才做B16完整门槛和三窗L1。
700目标未达到，仍需推进，不能因为本轮负结果自作主张暂停或标blocked。

## 重要旧结论，避免重复

编译上下文问题已修复：旧WS/PV/ring首次audit过早编译默认wait，旧约1000μs并非数学回退。
修正后三项814.12/820.24/806.38，全PASS未采用。设备必须核对kernels/aic、kernels/aiv的真实CCEC源码，
不能用ptoas中的替换前C++判断100tick。COMPILE_CONTEXT_CORRECTION.md保留完整证据。
QK提前KV通知823.38全PASS未采用；两面板Key预取813.42全PASS未采用。
早期所有候选/失败都在ATTEMPTS及旧交接，任何拟议方向先查，不重复主Q分核粗模型、payload cache、
KV标记、QK索引/聚合/零填充等已经验证的方向。

## 工作范围、验证约束和命令

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；C=R/vllm-ascend-dsv4-pto-0251rc1。
F为相邻csa_static_spmd_feasibility_20261015，冻结只读。
P/S分别为R/.worktrees/pypto-csa-static-fixes-20261016与simpler-csa-core-args-256-20261016。
只改N并把过程文本镜像到C/tests/pypto_test同目录，不改生产、F、依赖仓或共享安装。
正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8。只比较动态PTO；不执行Native forward。
无子agent、无hash；NPU只走task-submit --device auto，运行时不修改其导入driver/helper。
完整无插桩L1最早incore到最晚incore，计入等待/间隙，排除外部AICPU/orchestrator；
保留exact、TopK、六状态、图、30代、保护区。fixture恢复不等于连续decode。

CANN9.2beta2、torch2.10cpu+torch_npu2.10.post2；PTOAS0.66、ISA来自S/build/pto-isa/include。
CPU使用R/env-dsv4-0251rc1.sh + CSA_STATIC_PYPTO_ROOT=P、HCA_SIMPLER_ROOT=S及原PYTHONPATH，
F/bootstrap_isolated.py F/compile_jit_entry.py --memory-planner PYPTO --kernel-mode。
设备run_device_compare.sh 16 131072 新目录 --balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100
--qk-vector-indices --row-dependencies，加明确候选开关。插桩/短迭代只作诊断，不用于700验收。
本轮大Tile候选开关--qk-role-split --qk-role-wide-heads已测完，不要重复。

git需--git-dir=R/.git-repositories/vllm-ascend.git/worktrees/csa-spmd-dynamic-20261016 --work-tree=D；
普通worktree .git路径不对。镜像使用mirror_process_files.py --destination C/tests/pypto_test/csa_multishape_spmd_20261016，
仅mtime/size、文本白名单、≤2MiB；大产物留本地列LOCAL_ARTIFACTS，不hash、不删。

<!-- SOURCE_END history-document-105 -->

---

<a id="history-document-106"></a>
### 历史原文 106：WORKSPACE_MIGRATION_20261017.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/WORKSPACE_MIGRATION_20261017.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/WORKSPACE_MIGRATION_20261017.md)；原文 2,815 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-106 -->
# 2026-10-17：统一 vllm-ascend 工作区并同步远端

按用户要求，本会话后续统一在以下工作区继续：

`/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1`

当前分支：`dsv4-flash-pto-v0.25.1rc1`。

## 代码位置

- 本轮完整 CSA 单 SPMD 的区间入口、生成器、调度改写和实验核体：本目录 `csa_multishape_spmd_20261016/`。主要入口为 `range_policy.py`、`shape_specialization.py`、`prepare_range_case.py`、`device_compare.py` 和 `run_device_compare.sh`。
- 静态原型与冻结的原动态 PTO 参考：同级 `csa_static_spmd_feasibility_20261015/`，原始实现与既有结果保留。
- 生产算子目录仍为工作区内的 `vllm_ascend/ops/pypto/`。本轮单 SPMD 实验尚未接入生产目录。
- PyPTO 与 Simpler 的独立依赖 worktree 保留原位置，不更改共享安装。

## 远端同步

实际获取 `nalinaly/vllm-ascend` 的 `dsv4-flash-pto-v0.25.1rc1` 分支，远端为 `a919f119`。
此前本地与远端分别有 20 个、2 个独有提交，无同名已提交文件冲突，也不触及原有未提交修改。
以合并提交 `643d8039` 保留双方历史，合入：

- `f7ee8dda`：HC_pre 的 RMS 输入复用及 Sinkhorn 简化。
- `a919f119`：CSA 静态单 SPMD 原型、同步修复和性能对照。

补齐该分支的 fetch refspec 和 upstream，避免后续默认 fetch 仅更新旧的 `dsv4-flash-pto` 分支。
本次仅更新本地，未 push，也未提交本轮动态实验。原有三个已跟踪文件的未提交补丁逐内容保持一致。

## 迁移与验证

完整复制原动态实验目录，包含本地编译产物、泳道图及报告；没有删除目标目录的独有文件。
迁移前的静态目录及目标目录少数被更新的过程文件已在外部备份，具体位置见 JSON 记录。
既有日志中的绝对路径表示当时实际运行位置，不重写历史。旧动态工作树只作历史留存，后续不在其中开发。
本目录不再向自身执行 `mirror_process_files.py`；过程记录直接写在这里。

只在新位置运行一次 B17/H65536 的 CPU 模板生成，确认使用当前目录的 IR、绑定与源代码 DAG，
布局及同步协议检查通过；结果在 `workspace_migration_cpu_check_20261017/`。
这不是新的 lowering、设备精度或性能结果。本次未重跑 NPU，未计算文件 hash。
文件数、大小、链接目标、路径与版本的记录见 `WORKSPACE_MIGRATION_20261017.json`。

`hc_pre_reference_20261017.py` 是同步远端之前的参考快照；后续研究应读取现生产目录的最新 `hc_pre.py`。
冻结动态基线未跟随这次远端 HC_pre 更新改变，避免把历史对照的来源混淆。
所有档位降低至少 30% 的目标仍然有效，当前尚未达到。

<!-- SOURCE_END history-document-106 -->

---

<a id="history-document-107"></a>
### 历史原文 107：handoff_snapshots/NEXT_STEP_20261017_0544.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0544.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0544.md)；原文 15,610 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-107 -->
# 接续：2026-10-17 05:44，HC修正与主Q预取两任务进行中

Goal active：按batch/长度策略生成CSA单SPMD并逐档优化；128K/B16/S6完整无插桩L1<700μs。
当前现选行依赖首轮807.16μs；同进程复核817.88对旧841.94μs，不标完成/阻塞/暂停。
用户明确参考hc_pre按档分派，不要求不同batch共用一份二进制；运行时metadata仍device Tensor。
静态a919f119已push；动态目录未commit，暂无新提交指令。不能修改冻结F、依赖或生产入口。

## 当前优先收取，禁止重复提交

- task_20261017_054303_7422476042 / b16_row_deps_qproj_l0_prefetch_v1：主Q显式双缓冲，运行中。
- task_20261017_054339_75460017083 / b16_row_deps_hc_mix_resident_v3：HC恢复16×K256，运行中。
- 两个CPU均已通过并收取；实际地址/顺序审计已通过。详见HC_MIX_QPROJ_EXPERIMENTS.md。
- HC设备v1/v2均精度失败且未计时；不是Scalar累加导致，冻结原来16×K256而候选错写8×K512。
- 普通主Q N128 CPU通过却没形成Right双缓冲，未上卡，不再提交该版本。
- 运行中不要修改device_compare/shape_specialization及其导入helper。七档现选不变。

以下05:15及更早段落是历史交接；其中“所有任务终态”只对应当时，不覆盖上述新任务。

## 最新进展

- row_dependencies.py细化Indexer→TopK、TopK→plan、plan→QK、QK→O-A四类依赖。
  T96/B16长历史限定；物理映射/每核顺序/GM独立分配不动，额外保留同物理组Indexer
  全三成员→QK成员的FFTS复用顺序。明确放宽原task偏序，不称原DAG不变。
  16729→12183等待项，730→722发布点；全部1～32leaf行归属证明和约简后祖先关系通过。
- task_20261017_044912_40225339550/b16_cached_row_deps_v1 exit0完整PASS，
  L1807.16/804.40/811.26中位807.16，同轮动态905.08，30代/保护区通过。
- task_20261017_050033_181823706/b16_row_deps_paired_metadata_v1 exit0完整PASS。
  同进程旧841.94/843.14/833.20，新806.84/817.88/821.56，中位缩短2.86%。
  固定候选图和地址，131072→131071→65536→131072，6leaf变3leaf，输出/TopK/状态
  与原动态eager/图逐bit一致，11代/保护区/无新编译通过。不要重提这些验证。
- 已更新selected_shape_policies.json/SELECTED_SEVEN_RESULTS.md，仅B16长历史row_dependencies=true。
  该表仍是实验策略清单，尚未接生产自动派发；其他六档没有重测或改策略。
- 新增indexer_n80.py/--indexer-n80，M384/N80保留WS K384、原分叶、GM与FFTS：
  每1024列13个panel，尾64；26次matmul替代32；39个Native页段替代32。
  WS Right60KiB不能和Key Right10KiB并存，下个Key只预取L1；可能抵消panel减少的收益。
  理论Acc125KiB，实际规划/生成顺序已核对，与已选row_dependencies组合准备/编译通过。
  indexer_n80_prepare_v1与CPU session30043/indexer_n80_cpu_v1均完成通过。
  实际Acc前12WS在122880、末WS在0复用，总128000B；Right复用保留M→MTE1等待。
  审计见indexer_n80_codegen_audit.json（脚本首版的作用域/末WS假设错误已记）。
  task_20261017_050914_1722359619/b16_row_deps_indexer_n80_v1 exit0完整PASS，
  L1879.10/887.62/878.82中位879.10，同轮动态895.08。不采用、不扩测。
  INDEXER_N80_EXPERIMENT.md记录实际52个Tile地址与CPU一致、审计脚本两次修正及限制。
- 两项只读/离线核对：Indexer实际数学父任务不含主Q，主Q排队来自资源日历。
  同时推迟主Q矩阵/反量化的两种放置估计均799.38，对照773.43，不作新NPU候选。
  见deferred_main_q_schedule_analysis.json；不要将估计当设备时延。
- 本轮没有在运行任务。CPU sessions30043、89931已收取，PENDING_JOBS是终态观测快照。
  本轮未改生产/F/依赖；文本镜像在结束前同步到canonical同名目录。

## 最新结果和下一步

- qk_merge_heads.py / --qk-merge-heads：16-head PV合并后assemble完整状态。
  prepare_v1/cpu_v1与先left后right的prepare_v3/cpu_v3均需196608B Vec，超过188416B。
  prepare_v2只是生成8-head源码，未编译。三份源和两份失败保留，未上卡；不要重复扫chunk大小。
- qk_direct_l1.py / --qk-direct-l1：Cube从Native KV直接读三槽L1；两AIV先批量生成物理地址，
  使用原私有索引临时区，外部TopK不改；原kv_transfer每组32行零源。FFTS4发布地址，
  Cube等待后fence/cacheinvalid/fence；FFTS0改为两AIV的MTE2 PV消费确认，三槽复用前等待并排空。
  原QK/PV数学、FFTS1/2/3不变。无新GM/根参数/Host读取。完整CPU与实际NPU C++顺序一致。
  task_20261017_043021_371809025343 / b16_cached_qk_direct_l1_v1 exit0：完整exact、TopK、
  六状态、图、30代与保护区PASS。L1 1096.08/1100.10/1100.20，中位1100.10，
  同轮动态892.66，增加23.24%。不采用，不做额外边界或多档扩测。
- 小L1 gather探针cpu_v1通过；device_v1因缺kernel runtime init没执行kernel，失败保留。
  task_20261017_042437_362436518114 / qk_direct_l1_probe_device_v2同图7项哨兵PASS。
  protocol模型3328调度、322443步、2/3代，1024有效mask及1310SWA区间PASS，仅CPU协议证据。
- QK_DIRECT_L1_EXPERIMENT.md / qk_direct_l1_result.json / codegen_comparison保存完整证据。
  Vector阶段PIPE_ALL静态位点9→5，仍退步。实际Cube128行循环里没有额外逐行屏障；
  两Vector并行搬运改为单Cube逐行ND2NZ，不能只按GM字节数判断收益，也不能定量归因未测成本。
  A3 TPush/Tpop Vec→Mat当前仍经GM FIFO，不要把接口替换当成去GM路径。

以下是已完成行依赖实验之前的初查，最新结果见上节：
只读初查保存在query_dependency_inspection_initial.json：QK Cube每核等待17个前驱进度，
valid-plan每核47个，TopK merge每核71个，部分首批O-A每核71个。数字只是当前控制表的
前驱数，不证明这些依赖可删，也不是耗时。要证明具体读写行、原task内部FFTS、控制顺序与
临时区生命周期后才能缩小依赖。跨阶段实际core顺序本来不是全体barrier，不得误称全卡同步。
若需要拆原task为query微阶段，必须明确这会改变分工并重新证明协议，不能只删wait。

本轮没有新依赖/生产修改，也没有待运行CPU或NPU任务。原七档选择不变。

## 上轮Indexer结果，不能重复

- indexer_leaf_capacity.py / --indexer-leaf16384：实际原配置8192候选/叶，B16长历史为6叶，
  此前NEXT_STEP“9叶”是错误推测，已纠正。新16384容量为3叶，AIC/AIV/merge同改。
  每worker仍22个1024分数块，query/系数加载4→2，原分数数学/FFTS/arena不变。
  indexer_leaf_prepare_v1 CPU模型4133种长度×最多5前缀覆盖及merge匹配PASS；
  indexer_leaf_cpu_v1完整CPU编译PASS。该模型范围不代表当前KV分配支持范围。
  task_20261017_034636_30078467438 / b16_cached_indexer_leaf16k_v1 exit0完整PASS，
  输出/TopK/六份状态/图/30代/保护区通过；L1三窗829.22/829.42/813.78，中位829.22，
  同轮原动态897.84。没有现选SPMD同进程A/B，不能把历史差值量化为确定回退；不采用、不扩档。
- indexer_score_pingpong.py / --indexer-score-pingpong：原M384/N64分成两组M192/N64。
  两份INT32 Acc各48KiB同时存活；先提交两份QK，再FIX第一份。下一Key提前取到Right。
  保持原6叶、页地址、1024交接、TopK AIV和FFTS。与已测N128（单份Acc96KiB）不同。
  indexer_pingpong_prepare_v1 / indexer_pingpong_cpu_v1完整CPU通过；实际16对Acc地址0/49152，
  不重叠；C++两次QK在第一份TINSERT前。indexer_score_codegen_audit.py可复现，
  已核对CPU与实际NPU构建两份产物。初次审计误找TMOV失败，只修匹配为TINSERT，未改数学。
  task_20261017_035437_31474978052 / b16_cached_indexer_pingpong_v1 exit0完整PASS；
  L1三窗968.92/965.56/972.00，中位968.92，同轮动态904.72，明确不采用。
  指令静态位点：TMATMUL32→64、TINSERT16→32、TSTORE16→32、set/wait166→187，PIPE_ALL仍1。
  不把静态数量或源级交错当成测得的硬件重叠，也没有阶段探针可定量归因回退。
  汇总INDEXER_PARTITION_EXPERIMENTS.md / indexer_leaf_result.json / indexer_pingpong_result.json。
- scalar_gm_access_audit_initial.json只是只读源级清单；payload_cache.py已经做过逐阶段Scalar
  读写删DCCI且838.68μs，无新实现，不重复它。没有删除DCCI/DSB或使用假设一致的Scalar轮询。

两项新CPU session均已收取，两项NPU任务均完成。PENDING_JOBS.json是观测快照。
shape_specialization.py/device_compare.py新增两个互斥开关，均默认关闭；七档选择不变。
所有文本镜像到canonical同名N，大编译目录留本地，LOCAL_ARTIFACTS记录路径和大小，不hash。

## 后续方向，仅分析过，尚未实施

继续按有证据的结构瓶颈推进，避免重复改变Tile大小或轮询参数。
现选阶段探针（只作归因）中HC前端：widen 12 AIV，数学+维护中位13.13μs；
linear 24 AIC 5.54；reduce 6 AIV 1.06；split 12 AIV 1.84；Sinkhorn 12 AIV 13.42；
mix/norm 12 AIV 14.85。整个HC前端约5.22..67.84μs，有多次GM交接和任务通知。
可评估按同一8行分片融合reduce→split→mix，保持原4份投影部分和与RMS累加顺序；
Sinkhorn当前与mix并行，不能盲目串入主链。此方向需要重新证明DAG/分片及别名，未写实现。
简单把standalone的M32/K2048/8部分和搬进来会改变旧4部分和顺序，不预设exact通过。

## 上轮已完成，保留，不重复

- QK32行：task_20261017_031908_248319030873，b16_cached_qk_chunk_v2完整PASS，
  包含8项同图边界。L1中位831.32，动态886.70，未采用。首轮边界编译失败保留。
- HC stream：task_20261017_032959_27383327818，b16_cached_hc_stream_v1完整PASS，
  L1中位844.40，动态907.64，未采用。48 AIV分块供数，原M16/K4096/4部分和。
  自动pipeline CPU v2/v3的TLOAD先于READY，未上卡；v4手写FFTS交接循环用pl.range后通过。
  HC_STREAM_PRE_SYNC_AUDIT.md / hc_stream_pipeline_order_audit.json保留全部证据。

## 七档已选，不无改动重测

SELECTED_SEVEN_RESULTS.md / selected_shape_policies.json：
128K B4/B8/B16/B24 = 541.92/641.56/815.84/1121.12μs；
8K B16/B24/B32 = 621.96/836.74/905.72μs。
仅长B8/B16用QK地址缓存。cached轮询100tick。各自调度/cost_profile见选择表。
使用正式Layer4 W8A8权重；固定fixture，不是连续decode或生产自动派发。

## 本轮已经完成，不能重提

- qk_isolation_cpu_v1 exit0：相同QK AIV独立PTOAS后phase/tick PIPE_ALL仍9/4，
  不上NPU，不解除cached_wait+independent_stages的安全限制。
- task_20261017_022628_11712641926 / b16_cached_indexer_clock_v1 PASS：
  Indexer AIC中位210.23μs；Key提交11.81/704次；score_step193.96/22次；
  FFTS15/22次。是Scalar提交区间，不是异步pipe独占忙时；不能简单相减。
  完整插桩L1845.64不参与目标。页读取分支不继续优先优化。
- task_20261017_023022_12848062458 / b16_metadata_replay_v1 PASS：
  固定B16/S6/容量/地址/图，131072→131071→130048→131072，19个根输入变化。
  每步原动态eager、原动态图、单SPMD图输出/TopK/六份状态逐bit一致；
  全保护区、6代信号、无新增编译单元。纯功能，无性能重测。
- task_20261017_023946_156511419169 / b16_short_metadata_replay_v1 exit1：
  A8192/B8191通过；C7168跨compressed候选2048阈值，误要求复用另一数学策略的图。
  x_out max0.03125/172666项，TopK15576项不同；cache/state与guard通过。
  旧日志没有具体graph侧，不能归因单SPMD或原动态。失败保留。
- dynamic_metadata_replay.metadata_cases 用CPU编排call_signature+selected_control_path
  检查待执行变体同策略；不用device回读。8K的C选8186，8个跨策略候选先被拒绝。
  task_20261017_024535_170390925040 / b16_short_metadata_replay_v2完整PASS，
  8192→8191→8186→8192，无新编译、6代/地址/图/保护区通过，无性能采样。
  该检查目前是测试入口，不代表生产已接分档；匹配策略还须满足页表/工作区容量。
- oproj_k512.py / --oproj-k512：T<=96时M按16对齐，K预取512×8替代256×16，
  N128/原逻辑分片/同一K128累加顺序。oproj_k512_prepare_v1与CPU_v1通过；
  task_20261017_023822_152092626476 / b16_cached_oproj_k512_v1完整PASS，
  827.94/833.74/825.80，中位827.94；原动态885.68。未采用，不扩其他档。
- oproj_phase_loop.py / --oproj-phase-loop：把连续8个原proj_a_mm_0阶段共享循环体，
  原group顺序[6,5,7,4,1,3,0,2]，record步长432。逐AST确认只替换group/两个基址。
  不改数学、等待/通知次数、DCCI/DSB、记录表和顺序，不拼接独立C++helper事件。
  CPU_v1通过，AIC C++962738→753992B；实际AIC bin66832→54900B，AIV不变。
  task_20261017_025159_192231313200 / b16_cached_oproj_loop_v1完整PASS，
  三窗[835.94, 826.24, 830.74]，中位830.74；原动态903.74。未采用，不扩其他档。

以上CPU session均已收取，最新NPU任务都终态；PENDING_JOBS.json已更新。
没有新PyPTO/Simpler/生产/F修改；只在N下增加候选、探针、功能验证和记录。

## 更早负候选保留，不重复

MGATHER821.62、HC两行843.20、QK分支清零822.98、全有效无清零839.84、
Indexer N128883、全块搬运839、O-A N256839、GM复用833、完成汇聚860.30、
精确payload维护838.68，均未优于现选。全部CPU失败/边界检查见ATTEMPTS。
QK fallback确实有UB依赖/事件槽回退，不能删除；见QK_SYNC_FALLBACK_ANALYSIS.md。
任何探针/编译诊断不能用于700目标，也不能把未测区间全算屏障开销。

## 下一步参考

当前阶段探针主要时段：HC前端约0..68μs；Indexer约242..470；QK约511..672；
后端投影/HCpost约677..823（探针会扰动，不作为验收）。细小搬运/调度候选已多次无收益，
后续应针对有证据的数学/供数/任务结构变化；不要继续无变化扫描轮询或重复清零、K/N块。
单纯O-A代码缩小未提速，不能据此假定合并O-B重复代码就有收益。
只读核对standalone hc_pre.py：它是M32/K2048/8份投影部分和，冻结CSA仍M16/K4096/4份。
两者归约顺序不同，不能直接替换并宣称原exact保持。没有实施或验证此方向。
已有动态分档入口仍S6，不能拿独立HC_pre的S5表替代CSA七档。

## 路径和运行

R=/data/pyptouser/qinchuanyu/pto-eager
D=R/.worktrees/csa-spmd-dynamic-20261016，N=D/tests/pypto_test/csa_multishape_spmd_20261016。
F=相邻csa_static_spmd_feasibility_20261015冻结目录；P=R/.worktrees/pypto-csa-static-fixes-20261016；
S=R/.worktrees/simpler-csa-core-args-256-20261016。正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8。
NPU全部task-submit --device auto，查询task-submit --status ID；不跑Native forward。
单B16现选参数：run_device_compare.sh 16 131072 新目录 --balanced-schedule
--schedule-policy earliest_ready --schedule-cost-profile N/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies。
只做动态metadata时追加--metadata-replay-only，不采Event/L1。
文本同步canonical同名N，大文件/二进制/缓存留本地；LOCAL_ARTIFACTS列路径和大小，不hash。

<!-- SOURCE_END history-document-107 -->

---

<a id="history-document-108"></a>
### 历史原文 108：handoff_snapshots/NEXT_STEP_20261017_0550.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0550.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0550.md)；原文 6,907 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-108 -->
# 接续：2026-10-17 05:50，所有任务终态

Goal active：按batch/长度策略生成CSA单SPMD并逐档优化，128K/B16/S6完整无插桩L1<700μs。
未完成，不暂停、不标blocked。用户允许hc_pre式多档多二进制；长度/位置/页表仍device Tensor。
现选行依赖：首轮807.16μs，同进程复核817.88对旧841.94μs。

## 本轮收取，不重复

- HC整行驻留hc_mix_resident.py：只改mix/norm；12 AIV×8行改48 AIV×2行，整行保留UB，
  保留原mixed GM写，删除786432B回读。原冻结RMS实际是16×K256；初版错用8×K512。
- prepare/cpu v1 tile.full runtime Scalar失败，v2 INDEX→FP32 cast失败；v3/v4 CPU通过。
  NPU v1 task_20261017_053012_51687711509，v2 task_20261017_053800_64833816891均精度失败：
  x_out15613项max0.015625，TopK825项，四份cache/state不同；未计时。
  v4仅Scalar加法改Vector，结果相同，不能归因Scalar舍入。
- helper当前是16×256、0..15 Vector累加，prepare/cpu v5通过；
  task_20261017_054339_75460017083 / b16_row_deps_hc_mix_resident_v3 exit0完整PASS，
  包含输出/TopK/六状态/图/30代/保护区；L1822.46/819.02/817.06，中位819.02，动态911.56。
  初版精度差异由恢复原归约顺序消除，但无性能收益，不采用、不扩测。
- qproj_n128.py：普通N128/20worker/K256 prepare/cpu_v1通过，Right实际只有地址0，
  没有双缓冲，未上卡；不要因CPU编译通过就提交该普通版本。
- qproj_l0_prefetch.py：T96、M96/N256/K128八段，QR在L1常驻，显式先取下一Left/Right。
  prepare/cpu_v1通过。两处Left0/12288、Right0/32768交替，下一Right TMOV先于当前matmul，
  qproj_prefetch_codegen_audit.py核对CPU和实际NPU产物均通过。
  task_20261017_054303_7422476042 / b16_row_deps_qproj_l0_prefetch_v1 exit0完整PASS；
  L1825.54/831.16/830.62，中位830.62，同轮动态892.68；不采用、不扩测。
- 证据HC_MIX_QPROJ_EXPERIMENTS.md、hc_mix_qproj_results.json、两任务codegen_audit。
  没有与现选同进程A/B，不把不同运行差值量化为确定回退。
- 所有CPU会话均收取，无运行NPU任务；PENDING_JOBS.json是终态观测快照。

## 当前已选七档与动态验证

selected_shape_policies.json / SELECTED_SEVEN_RESULTS.md：S6不变，
128K B4/B8/B16/B24 = 541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32 = 621.96/836.74/905.72μs。
只有长B8/B16启用QK地址缓存，只有长B16启用row_dependencies；cached等待100tick。
其他档各自的schedule_policy和cost_profile见策略表，不能把B16参数套到所有档。
实验build_variant入口可按描述符分别生成，不是已接生产自动派发。

row_dependencies.py等待16729→12183、发布730→722，按query/head依赖替代四类完整task依赖。
数学/物理分配/各核顺序不变，但原task偏序被有意放宽，不宣称DAG不变。
保留同组Indexer AIC+2AIV→QK全部边保证FFTS复用。1..32leaf覆盖证明和约简祖先关系通过。
- task_20261017_044912_40225339550 / b16_cached_row_deps_v1 PASS，三窗807.16/804.40/811.26。
- task_20261017_050033_181823706 / b16_row_deps_paired_metadata_v1 PASS：
  同进程旧841.94、新817.88，固定图/地址131072→131071→65536→131072、6leaf→3→6，
  与原动态eager/图逐bit一致；11代/保护区/无新编译通过。不要重复这些验证。
- 另已测短历史8192→8191→8186→8192同图；首版7168跨数学策略失败保留。
  metadata_cases先CPU检查call_signature/selected_control_path。没有host读device数值。

## 已否定方向，防止重复

- Indexer M192/N128、WS K192、六query分两组三query方案已经indexer_n128.py测883μs。
  本轮重新审视后确认是旧方案，未重复实施。不要误认为先前N128是M384直接超Acc。
- Indexer每叶16384：829.22；M192/N64双Acc：968.92；M384/N80：879.10，全部完整PASS但不选。
  N80实际Acc128000B、WS Right60KiB挤掉Key预取；详见INDEXER_N80_EXPERIMENT.md。
- QK直接GM→L1：1100.10 vs动态892.66，完整PASS且明显退步；协议小探针已完成，不重复。
  QK16head合并PV两版都UB196608>188416，CPU失败，未上卡。
- QK32行gather831.32，HC48AIV流式widen844.40，HC两行843.20；均不选。
  HC自动pipeline曾把TLOAD移到FFTS READY前，显式range版才正确，不再用有问题版本。
- O-A K512827.94，O-A共享八阶段循环830.74，普通N256839；不选。
- 精确payload缓存维护838.68、GM复用833、完成汇聚860.30、MGATHER821.62；不选。
- 推迟主Q两数学任务的两个离线调度估计均799.38，对照773.43，未上卡。
  qproj worker24/32/64/128离线模型也无优势；不盲扫。
- HCpost当前已经按一整行4096列计算；不要误认为它仍是16次N256分块。

## 后续需要真正的新结构收益

当前历史阶段探针（有扰动，只用于归因）：HC前端约0..68μs；主Q矩阵中位42μs；
Indexer约242..470、AIC中位211.77μs；QK约511..672；后投影/HCpost约677..823。
普通N/K分块和小搬运大多已失败。不要继续无变化扫轮询，不能把全部未测时间算成屏障。
可分析HC的reduce→split→mix融合，推迟非关键路径Sinkhorn，但必须保持原4份投影和16×256 RMS顺序，
重新证明写行、别名和同步；尚未实施。standalone hc_pre是M32/K2048/8份部分和，
冻结CSA是M16/K4096/4份，不能直接替换并假定逐bit相同。
若考虑query级流水，需要真正拆分Indexer/QK阶段并发布细粒度进度，不是直接删task依赖；
GM代际、DCCI+DSB和原FFTS资源复用必须成立。此项仅方向，未实施/验证。

## 路径、约束与运行

R=/data/pyptouser/qinchuanyu/pto-eager
D=R/.worktrees/csa-spmd-dynamic-20261016；N=D/tests/pypto_test/csa_multishape_spmd_20261016。
F=相邻csa_static_spmd_feasibility_20261015（冻结，只读）；
P=R/.worktrees/pypto-csa-static-fixes-20261016；S=R/.worktrees/simpler-csa-core-args-256-20261016。
正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8；不执行Native forward。
只编辑N，文本镜像canonical同名目录；不改生产/F/依赖共享安装，不使用hash，不启动子agent。
静态a919f119已push；动态N未commit，暂无新提交指令。
NPU仅task-submit --device auto，任务运行时不要编辑driver及其导入helper。
现选参数：run_device_compare.sh 16 131072 新目录 --balanced-schedule
--schedule-policy earliest_ready --schedule-cost-profile N/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies。
性能口径完整无插桩L1 incore首尾，3窗；Event另列，stage_clock不能参加700μs验收。
大文件/二进制/缓存仅本地；LOCAL_ARTIFACTS列路径和大小。
本轮以前的完整交接见handoff_snapshots/NEXT_STEP_20261017_0544.md和ATTEMPTS.md。

<!-- SOURCE_END history-document-108 -->

---

<a id="history-document-109"></a>
### 历史原文 109：handoff_snapshots/NEXT_STEP_20261017_0638.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0638.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0638.md)；原文 6,021 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-109 -->
# 接续：2026-10-17 06:38

Goal active：多batch/长度档位可独立编译和调优，128K/B16/S6完整无插桩L1<700μs。
未达标，不暂停、不标blocked。现选七档不变；长B16首轮807.16μs，同进程复核817.88μs。
仅实验入口，未接生产自动派发。全部元数据仍device Tensor，不新增host回读。

## 本轮完成，勿重复

- hc_front_fusion.py：reduce→split→mix/norm同一8行成员，12成员；四份投影和16×K256 RMS顺序不变。
  删除两个task，Sinkhorn保留；GM读写和内部2fence保留。prepare/cpu_v1 PASS。
  task_20261017_055741_99590920590 / b16_row_deps_hc_front_fusion_v1 PASS，三窗804.84/810.80/806.76，动态900.14。
- hc_front_forward_tiles.py：在上述融合内展开split单次循环，8×32转置→对齐行提取→转置，pre直接Tile传mix。
  删除3GM回读/2局部fence，保留GM写和阶段外发布缓存。CPU v1 scratch形状错；v2/v3 slice静态valid与
  transpose scratch动态valid不一致；v4用extract而非slice完整PASS，没有改依赖。
  task_20261017_061445_12767489743 / b16_row_deps_hc_front_tiles_v1 PASS，L1 819.74/810.44/810.44，动态891.96。
- indexer_input_dependencies.py + row_dependencies.py可选indexer_inputs：quant→系数、quant→Indexer、系数→Indexer
  三条边按真实读写行细化；每Indexer组2请求、6quant/2系数写者，768个动态叶×worker覆盖检查。
  等待12183→10674，发布722→729，FFTS 216条资源边和同核顺序保持。prepare/cpu_v1 PASS。
  task_20261017_061600_130106629188 / b16_row_deps_indexer_inputs_v1 PASS，L1 812.62/815.48/810.34，动态882.64。
  三个候选均未显示相对现选明确收益，不采用/不扩七档。没有同进程现选A/B，不将跨运行几微秒差值作定论。
- indexer_scale_page_owner.py八AIV按物理page%8归属，保留同页compact顺序，CPU v1 PASS。
  历史scale204us已完成，query系数235us才完成，模型也不改善；只保留CPU候选，不上卡。
- 文档HC_FRONT_AND_INPUT_DEPENDENCIES.md、hc_front_and_input_deps_results.json和ATTEMPTS已记录。

## 最新阶段证据与后续

现选row_dependencies允许stage_clock组合，原信号空闲区时戳探针未变。
row_dependencies_stage_clock_cpu_audit.json通过；task_20261017_062004_137819816687 /
b16_row_deps_stage_clock_v1 exit0完整PASS。插桩span796.24us不能参加700us验收。
row_deps_stage_summary.json与row_deps_stage_critical_chain.json：
Indexer AIC234.42..458.16、AIV247.90..461.94，QK/PV481.7..647.36，
OA650.26..735.32、OB712.20..766.78、HCpost772.68..796.22。
KV norm456..465虽被占核推迟，但在QK plan完成前结束，不能当作当前主瓶颈。
最晚显式前驱链显示core4连做kv_score、idx_qr、kv_score、weights、主Q；随后core43的主Q反量化拖到249.54，
其Indexer AIV约253.96才开工。尾部core11有3个OA和2个OB串行。
链上数学/核内同步/发布累计724.36，外部就绪至开工71.88；不是硬件下限，不能把所有间隙都当同步。
原先两版主Q推迟离线调度已失败799.38vs773.43，不要直接重做。
可能需改变真正关键的分核或流水，单纯减少等待计数的新候选尚未带来收益。

HC分块UB复用候选hc_mix_chunk_buffer.py已准备并编译：保持12成员×8行、原16×K256数学与pipeline，
额外65536B BF16 UB供norm复用，保留mixed GM写。v1窄列assemble到8×4096因TMOV约束失败；
v2改128×256布局（16个8×256块连续存），prepare/cpu_v2完整PASS。
驻留Tile只有一份，地址98336..163872（<188416）。layout_proof检查32768元素唯一覆盖。
正式flag --hc-mix-chunk-buffer，prepare_v3 entry完整AST与CPU_v2相同。
task_20261017_063343_160455814507 / b16_row_deps_hc_mix_chunk_buffer_v1 已exit0完整PASS；
单SPMD中位814.74μs，同轮动态907.88μs；30代/图/保护区均通过。不采用、不扩七档，无同进程现选A/B。
证据hc_mix_chunk_buffer_result.json及HC_MIX_CHUNK_BUFFER_EXPERIMENT.md。
阶段汇总可复现：summarize_stage_path.py --case b16_row_deps_stage_clock_v1 --output 新分析目录。

## 已选和历史负候选

selected_shape_policies.json / SELECTED_SEVEN_RESULTS.md：S6，
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32=621.96/836.74/905.72μs。只长B8/B16启用QK地址缓存，只长B16启用row deps。
长B16固定图131072→131071→65536→131072已逐bit/状态/保护区/无新编译PASS；短动态也已PASS。
不要重跑。详细上一版交接见handoff_snapshots/NEXT_STEP_20261017_0550.md。
HC48行resident恢复16×K256后PASS819.02，主Q显式L0预取PASS830.62，未采用。
Indexer N128/双Acc/N80/16K叶、QK直接L1/32行gather、HC流式48AIV、OA K512/N256/共享循环、
全局等待backoff扫描、compiled control、completion aggregation等均已有负结果。不要重新试同一个方向。

## 路径/约束

R=/data/pyptouser/qinchuanyu/pto-eager，D=R/.worktrees/csa-spmd-dynamic-20261016。
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻csa_static_spmd_feasibility_20261015（冻结只读）。
P=R/.worktrees/pypto-csa-static-fixes-20261016；S=R/.worktrees/simpler-csa-core-args-256-20261016。
正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8；不运行Native forward；只改N并镜像文本到canonical。
不改生产/F/依赖共享安装，不使用hash，不开子agent。静态a919f119已push；动态目录未commit，无新提交指令。
NPU仅task-submit --device auto，运行期间不编辑driver/imported helper。所有CPU/NPU均终态并已收取，当前无本会话运行任务。
现选参数：run_device_compare.sh 16 131072 新目录 --balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies。
大文件/缓存/二进制仅本地；文本镜像manifest LOCAL_ARTIFACTS.json，mtime/size比较不用hash。

<!-- SOURCE_END history-document-109 -->

---

<a id="history-document-110"></a>
### 历史原文 110：handoff_snapshots/NEXT_STEP_20261017_0732.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0732.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0732.md)；原文 5,840 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-110 -->
# 接续：2026-10-17 07:32

Goal仍active：按batch/长度策略分别编译，128K/B16/S6完整无插桩L1<700μs。
未达到目标，不暂停、不标blocked。现选七档不变，B16首轮807.16μs，同进程复核817.88μs。
只做实验入口，尚未接生产自动派发；实际长度/位置/页表始终device Tensor。

## 本轮完成，勿重测

1. member_schedule.py：成员级数学DAG先于物理分配，973成员/877单元，216条FFTS资源边保留。
   三种CPU策略：同成本原771.53μs，critical802.49、earliest768.79、finish858.52。
   最好仅-2.74且等待更多，不上卡、不生成二进制。源row审计的原核顺序/等待描述已标为被替代。
   见MEMBER_SCHEDULE_CPU_EXPERIMENT.md及member_schedule_*_cpu_v1。未接driver。
2. O-A实际已有K128内层、Left/Right 0/32768双缓冲，勿重复做。片段oa_selected_physical_fragment.pto.txt。
3. weight_cache_policy.py：实验融合生成器过滤func_attr，QA/主Q/O-A Tensor数学体丢声明；
   KV Tile load的cache=1已有保留。恢复属性prepare/cpu_v1完整PASS，IR cache属性2→44，
   但实际C++完整2784561字节与现选全文相同，未上卡。PTOAS只有L2Bypass且显式offset才调整地址。
   原动态同样无有效BYPASS C++；不是融合独有性能原因，也不能归因为ISA不支持。
   开关--restore-weight-cache-policy默认关。WEIGHT_CACHE_POLICY_EXPERIMENT.md及codegen_audit_v1.json。
4. qk_query_metadata.py：每query一次准备512地址，两个2KiB UB Tile跨tick，四压缩块复用。
   页表最多4→1次，但每AIV地址计算256→512项；负索引、KV DMA、数学、FFTS保持。
   prepare/cpu_v1 PASS，驻留135168/137216..139264。B1..40/S6 CPU地址扫描2519040项通过。
   task_20261017_070649_214406126149 / b16_row_deps_qk_query_metadata_v1 exit0完整PASS。
   L1 811.82/815.08/805.18中位811.82，同轮动态904.92；不采用、不扩七档。
5. indexer_n80_ws128.py：旧N80 WS Right60KiB挤掉下一Key；WS改0/128/256三段K128，
   两WS Right20KiB+两Key10KiB，实际末端61440；WS0/20480、后续Key40960/51200。
   保持QK M384/N80、13panel、叶/FFTS/AIV TopK。K指令边界改变，不能只称数学不变。
   prepare/cpu_v1 PASS，12处下一Key extract先于当前QK；非硬件重叠证明。
   task_20261017_071758_23254658630 / b16_row_deps_indexer_n80_ws128_v1 exit0完整PASS。
   L1 844.52/855.10/858.08中位855.10，同轮动态906.78；不采用、不扩七档。
6. qk_all_valid.py：三成员从同一plan扫本组query×5块；全>0用消除外层块判断的循环，
   否则保留原循环AST。负索引/尾部/数学/FFTS未删，新增Host回读和GM缓冲均0。
   prepare/cpu_v1完整PASS；AIV快/回退PIPE_ALL静态3/4、mask读取0/4，AIC0/0、0/2。
   FFTS发布统计初版搜错set_flag_dev，已改为ffts_cross_core_sync；未改算子。
   task_20261017_072611_24632809131 / b16_row_deps_qk_all_valid_v1 exit0完整PASS。
   L1 804.30/817.26/820.92中位817.26，同轮动态905.18；不采用、不扩七档。

三项新设备实验均原exact输出/TopK/六状态、eager/图、30代/保护区PASS；固定fixture恢复，
不是连续decode轨迹；未做现选同进程A/B，不把历史几微秒差值当精确收益/回退。
相应*_EXPERIMENT.md、*_result.json、ATTEMPTS.md包含细节。所有CPU/NPU会话已收取终态。
当前无本会话运行任务，PENDING_JOBS.json全部terminal。不要重复提交。

## 当前方向证据

现选阶段探针（上轮已完成，勿重测）row_deps_stage_analysis：Indexer约234..462，
QK/PV482..647，O-A650..735，O-B712..767，HCpost773..796μs。插桩不参加700验收。
最晚前驱链数学/内部同步/发布724.36，就绪到开工71.88；不是硬件下限，也不等于全部软同步。
仅减少等待计数、局部metadata处理或一个屏障位点，至今没有明确改进。

本轮只读核对预取，未实现候选：ISA TPREFETCH是GM→临时UB搬运，不是异步SDMA；
PyPTO没有对应tile API。TPREFETCH_ASYNC/pl.prefetch使用SDMA，当前独立Simpler
python/simpler/worker.py:6651仍明确拒绝kernel+enable_sdma，
device_runner_base.cpp:700也说明kernel没有该通道。不要绕过检查或硬编码L2地址偏移。
没有修改任何依赖仓或共享安装。更早负候选和已验证metadata见下方交接归档。

## 已选档位与约束

selected_shape_policies.json / SELECTED_SEVEN_RESULTS.md，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；
8K B16/B24/B32=621.96/836.74/905.72μs。
只长B8/B16启用QK地址缓存，只长B16启用row deps。
长B16固定图131072→131071→65536→131072、短metadata验证早已通过，不要重跑。
前一轮详细交接handoff_snapshots/NEXT_STEP_20261017_0638.md和0550版，保留所有失败。

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻静态feasibility目录，冻结只读。
P=R/.worktrees/pypto-csa-static-fixes-20261016，S=R/.worktrees/simpler-csa-core-args-256-20261016。
只改N并镜像到canonical同名目录；不改生产/F/依赖共享安装，不用hash，不开子agent。
正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8，不运行Native forward，NPU只task-submit --device auto。
运行任务时不要编辑driver/imported helper。新开关默认关，所有新候选均未纳入七档选择。
静态a919f119早已push；本动态目录未commit，无新提交指令。
现选B16参数：--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100
--qk-vector-indices --row-dependencies。大文件/缓存本地，LOCAL_ARTIFACTS.json列路径；不用hash。

<!-- SOURCE_END history-document-110 -->

---

<a id="history-document-111"></a>
### 历史原文 111：handoff_snapshots/NEXT_STEP_20261017_0800.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0800.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0800.md)；原文 5,206 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-111 -->
# 接续：2026-10-17 08:00

Goal仍active：按batch/长度策略分别编译，128K/B16/S6完整无插桩L1<700μs。
未达到目标，不暂停、不标blocked。现选七档不变，B16首轮807.16μs，同进程复核817.88μs。
最新用户强调可以hc_pre式分档，不需一份二进制支持所有batch；已遵循。

## 本轮新完成，勿重测

1. indexer_short_tail.py / --indexer-short-tail：device valid_count剩1..64项时只算
   一个N64面板，原完整路径AST、QK/WS数学、FP16 FIX、FFTS/AIV TopK保留。
   后960列内部scratch不写，排序前由原各query有效长度fillpad排除。
   contract.py/json核对9条fillpad分支、184314个query/step区间，其中2554短尾；
   prepare/cpu_v1通过，实际C++短/原matmul2/32、FIX1/16、store1/16。
   工作量模型B4/B8/B16最繁忙核不变；B24各核528→513面板，非时间预测。
   - task_20261017_074327_274989723191 / b16_row_deps_indexer_short_tail_v1 exit0 PASS；
     L1 820.12/820.32/819.08，中位820.12；同轮动态902.04，不采用。
   - B24 prepare_v1通过。task_20261017_074803_283067829725 / b24_indexer_short_tail_v1
     exit0 PASS；L1 1146.14/1127.42/1127.60，中位1127.60；动态1175.66，不采用。
   未扩B4/B8/七档；详见INDEXER_SHORT_TAIL_EXPERIMENT.md。
2. signal_polling.py新增ramp_backoff，driver --ramp-wait默认关：每个新依赖
   失败后等4/8/16/32/64/100tick再读，此后100；每次DCCI+volatile读、GE条件、
   原TNOTIFY写侧、payload、周期PIPE_ALL和spin限额保持。不是重扫固定间隔。
   wait_ramp_cpu_v1/audit.json：默认C++与现选全文相同，递增版只改变helper。
   task_20261017_074820_28363061302 / b16_row_deps_ramp_wait_v1 exit0 PASS；
   L1 833.74/829.94/826.22，中位829.94；动态910.94，不采用。
   见RAMP_WAIT_EXPERIMENT.md。三个NPU均原exact输出/TopK/6状态、图/30代/保护区PASS。
   均固定fixture恢复，不是连续decode；无现选同进程A/B，不把历史差值当精确回退。
3. indexer_rolled_panels.py（仅CPU入口indexer_rolled_panels=True，未接device driver）：
   16个N64展开面板改首QK+15次循环+末WS，数学和FFTS不变，运行时仍32matmul。
   - prepare_v1成功，cpu_v1失败：循环携带Key L1池导致同地址Mat move，180224。
   - v2把Key L1池改迭代内临时区，只carry Key Right/上一score L1；prepare_v2通过，
     cpu_v2失败：实际PTO生成Right→Right tmov，A3 PTOAS不支持。另有Mat→Mat移动。
   所有CPU exec均收取终态，包括session18915/12565 exit1。没有提交设备或修改依赖。
   INDEXER_ROLLED_PANELS_EXPERIMENT.md和indexer_rolled_v2_unsupported_moves.pto.txt有证据。
   indexer_rolled_source_contract.json的页公式自比较不是独立测试，不计为正确性证明。

## 下一步可推进

可以研究把两个固定面板放在一次loop内，消费完Key/score后再迭代，以避免Right跨循环carry。
这个方向尚未实现，可能损失跨pair的QK/WS重叠；须先检查实际内存与同步，不能预记更快。
不要强行删除编译器校验、手改硬件容量或绕过缓存同步。也不能把代码变短认定为已测到I-cache瓶颈。

上轮完整证据保存在handoff_snapshots/NEXT_STEP_20261017_0732.md，包含成员排程CPU负结果、
恢复cache policy生成C++相同（勿上卡）、QK每query metadata811.82、N80 WS128 855.10、
QK全有效817.26。更早0638/0550交接和ATTEMPTS保留所有失败，不重复跑。
现选阶段探针row_deps_stage_analysis：Indexer约234..462、QK/PV482..647、
O-A650..735、O-B712..767、HCpost773..796μs；仅诊断，不参加700验收。
当前无运行的CPU/NPU任务，PENDING_JOBS全部terminal。

## 已选与固定约束

selected_shape_policies.json/SELECTED_SEVEN_RESULTS.md，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；8K B16/B24/B32=621.96/836.74/905.72μs。
只长B8/B16用QK地址缓存，只长B16用row deps；新候选默认全关。动态metadata验证早已通过勿重跑。
仍为实验入口，未接生产自动派发；长度/位置/页表等仍device Tensor。

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻static feasibility，冻结只读。
P=R/.worktrees/pypto-csa-static-fixes-20261016，S=R/.worktrees/simpler-csa-core-args-256-20261016。
只改N并镜像canonical同名目录；不改生产/F/依赖/共享安装，不用hash，不开子agent。
正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8，不运行Native forward，NPU只task-submit --device auto。
任务运行期间禁止编辑driver/imported helper。静态a919f119早已push；动态N未commit，无新提交指令。
当前Simpler kernel模式没有SDMA，勿重提直接async_prefetch；旧O-A已有K128 Left/Right双缓冲。
B16现选参数：--balanced-schedule --schedule-policy earliest_ready --schedule-cost-profile N/b16_dynamic_l1_cost.json
--cached-wait --wait-backoff-ticks 100 --qk-vector-indices --row-dependencies。
大文件/缓存本地保留，canonical LOCAL_ARTIFACTS.json列路径，不做hash。

<!-- SOURCE_END history-document-111 -->

---

<a id="history-document-112"></a>
### 历史原文 112：handoff_snapshots/NEXT_STEP_20261017_0849.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0849.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0849.md)；原文 6,208 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-112 -->
# 接续：2026-10-17 08:49

Goal active：按batch/长度策略分别编译，128K/B16/S6完整无插桩L1<700μs。
未达到，不暂停、不标blocked。现选七档不变；B16历史807.16μs，最新同进程对照808.92μs。
用户允许hc_pre式分档，不要求单份二进制覆盖所有batch。

## 本轮完成，勿重复

1. N64分组循环group2：task_20261017_080841_317060310914 exit0，完整PASS，
   L1中位920.58μs；动态909.50。未采用、不扩group4/8或其他档。
   `INDEXER_PANEL_GROUP_EXPERIMENT.md`、indexer_group2_result.json。
   group4/8只有生成AST列覆盖检查，没有编译/设备PASS。
2. `indexer_transfer_2048.py`：保留原16个N64展开与叶分片，两个1024块一次交接。
   私有GM[288,1024]→[288,2048]，2槽；AIC只算有效子块，AIV段span限定叶尾。
   - prepare_v1/cpu_v1完整代码生成通过，Vec端点131072、score load3×2048。
   - task_20261017_082119_33910122896 / b16_row_deps_indexer_transfer2048_v1 exit1，
     输出/TopK不符但六状态一致，无性能。原因是本候选遗漏root步号转换：
     原1024轮step<2必须改为2048轮step==0（三query共三处）。不是依赖组件bug。
   - prepare_v2通过，contract_v2补查首段初始化。task_20261017_082629_34925986670
     / b16_row_deps_indexer_transfer2048_v2 exit0，完整PASS，L1中位834.68μs
     （836.98/834.68/819.72），同轮动态899.98。未采用、不扩档。
   - 见INDEXER_TRANSFER_2048_EXPERIMENT.md；失败和所有原始源码均保留。
3. `indexer_key_pool.py`只做CPU：原128行L1 Key池改1024行，页各占独立32行，
   原数学/页地址/提交顺序/FFTS不变，也没有增加预取深度。
   prepare_v1/cpu_v1通过。完整stage16仍32matmul，MTE1→MTE2 wait19→20，
   反向18→18，没有目标等待下降，未接driver、未上卡。
   完整C++块用done_16+括号匹配；早先v30前缀只取后半段的计数已纠正，勿使用。
   INDEXER_KEY_POOL_CPU_EXPERIMENT.md / indexer_key_pool_codegen_audit.json。
4. `qproj_l1_columns.py` / --qproj-l1-columns：原20worker/N256列归属保持，每组拆双N128，
   QR L1驻留，两个1024×128权重L1一次加载，四段K256交错双INT32累加链。
   - prepare_v1/cpu_v1通过。实际Left0/24576、Right0/32768、Acc0/49152均不重叠；
     L1权重65536/196608、QR327680，端点425984。2matmul+6matmul_acc。
   - task_20261017_083715_36883533451 / b16_row_deps_qproj_l1_columns_v1 exit0，
     完整PASS；L1 809.22/806.46/797.56，中位806.46；动态903.86。
   - 因与历史现选接近，做一次同进程对照（不重复Event/metadata）：
     task_20261017_084154_377975115763 / b16_qproj_l1_columns_paired_v1 exit0，
     双方exact/图/六状态/保护区/7代PASS；现选805.28/808.92/810.24，中位808.92，
     候选806.02/804.06/810.32，中位806.02。差-0.36%，均值差-1.35μs，范围重叠。
     未采用、不扩档、不重复比较。对照生成AST与现选逐项相同。
   - paired_row_dependency_perf.compare新增可选reference_row_dependencies/candidate_label/folder，
     原默认行为保持；driver新增--paired-qproj-spmd-only，仅测三窗L1及其正确性保护。
     QPROJ_L1_COLUMNS_EXPERIMENT.md / qproj_l1_columns_paired_result.json。

所有NPU与CPU exec已收取终态，无运行任务。CPU session96611/99975已exit0，
wait session88304已exit0。PENDING_JOBS.json全部terminal。候选默认全关，现选不变。
正式权重、固定fixture恢复，不代表连续decode，未运行Native forward。

## 下一方向（尚未实现，不预记收益）

可先CPU研究Indexer N32、M384双QK Acc：每份48KiB、两份96KiB；
保留原M384/WS K384，沿候选列拆N，而非已测变慢的M192/N64分query。
尝试QK(next)先提交，再FIX(current)，然后WS(current)，显式展开32面板以避免
Right/Mat跨循环carry。两Key Right各4KiB，两WS Right各24KiB，WS Acc各2KiB，
理论容量可容纳，但要看实际分配/同步。数学调用翻倍及代码体积增加可能抵消收益；
CPU不成立就不提交设备。原N64仍基线，不叠加本轮未采用的其他数学候选。

只读已排除两件事：SWA原来就按物理页批量gather，不是逐行搬运；
row_dependencies已有传递约简及逐核observed过滤，不要再做同样seen缓存。
原N64 QK/WS用独立MTE1事件，不是错误共用等待。

## 已有选择和约束

SELECTED_SEVEN_RESULTS.md/selected_shape_policies.json，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；8K B16/B24/B32=621.96/836.74/905.72μs。
只长B8/B16用QK地址缓存，只长B16用row deps；默认100tick cached wait。
B16基线参数--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100
--qk-vector-indices --row-dependencies。动态metadata早已通过，勿重复。
仍实验入口，未接生产自动派发；长度、位置、页表仍device Tensor。

前轮负结果见handoff_snapshots/NEXT_STEP_20261017_0800.md、0732/0638/0550和ATTEMPTS.md。
短尾B16/B24、递增退避、N64循环化失败、N80、各种QK/OA/HC试验勿重复。
权重cache属性恢复生成C++全文相同，勿上卡；当前Simpler kernel不支持SDMA；
原O-A已有K128 Left/Right双缓冲。完整阶段诊断row_deps_stage_analysis仅用于归因，
不能计入700us；Indexer约234..462、QK/PV482..647、O-A650..735、HCpost773..796μs。

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻static feasibility，冻结只读。
P=R/.worktrees/pypto-csa-static-fixes-20261016，S=R/.worktrees/simpler-csa-core-args-256-20261016。
只改N并镜像canonical同名目录；不改生产/F/依赖/共享安装，不做hash，不开子agent。
正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8，NPU只task-submit --device auto。
任务运行期间不改driver/imported helpers。静态a919f119早已push；动态N未commit，无新提交指令。
大文件/缓存留本地，canonical LOCAL_ARTIFACTS.json列路径，无hash。

<!-- SOURCE_END history-document-112 -->

---

<a id="history-document-113"></a>
### 历史原文 113：handoff_snapshots/NEXT_STEP_20261017_0926.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0926.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0926.md)；原文 6,332 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-113 -->
# 接续：2026-10-17 09:26

Goal active：按batch/长度策略分别编译，128K/B16/S6完整无插桩L1<700μs。
未达到，不暂停、不标blocked。现选七档不变；B16历史807.16μs，最新同进程现选807.50μs。
用户明确允许hc_pre式分档，不要求一份二进制支持全部batch。

## 本轮完成，勿重复

1. Indexer N32双QK Acc：`indexer_n32_acc.py` / --indexer-n32-acc。
   保留M384、WS K384，只拆候选列；32面板展开，原1024交接/AIV/叶计划不动。
   prepare_v1/cpu_v1完整通过；31对相邻Acc0/49152交替，下一QK先于当前FIX，
   Left query0/系数49152；实际64matmul/32FIX/32store。CPU审计已保存。
   task_20261017_085633_40447024100 / b16_row_deps_indexer_n32_acc_v1 exit0，
   全部exact/图/6状态/30代/保护区PASS。L1 985.94/976.20/981.12，中位981.12μs，
   同轮动态901.76。未采用、不扩档、不重复。见INDEXER_N32_ACC_EXPERIMENT.md。
2. 删除整个kv_touch：`kv_touch_elision.py` / build drop_kv_touch=True。
   核对严格1行BF16自拷贝，把QK父依赖改为真正csa_cache_writeback，保留任务祖先集合。
   prepare_v1因检查call实参的Out装饰失败（实参类型已去装饰、真实核体仍Out），
   修正用核体形参+allocation核对。prepare_v2/cpu_v1完整通过。
   任务66→65/成员973→972，但远端等待12183→12673增加490，发布仍722。
   原marker还是写回完成汇总者，故这个版本未上卡，不能宣称实际变慢。
3. KV纯信号：`kv_touch_signal_only.py` / --kv-touch-signal-only。
   保留节点/等待表/原通知，只删除1024B自拷贝及前后4fence/2cacheinvalid。
   其他源AST完全相同，信号DCCI/DSB、生产者发布、QK读取前维护保持。
   prepare_v1/cpu_v1通过，实际阶段TLOAD/TSTORE各1→0，PIPE_ALL5→1、全DCCI2→0，
   TWAIT/TNOTIFY各1保持。task_20261017_091307_19138223772 exit0，完整PASS；
   L1 802.96/812.72/798.04，中位802.96，同轮动态894.00。
   一次同进程复核task_20261017_091800_27634922996 / b16_kv_touch_signal_paired_v1 exit0：
   - 现选805.14/808.22/807.50，中位807.50；候选812.08/803.92/812.32，中位812.08。
   - 双方exact/图/6状态/7代/保护区PASS，reference AST等于现选。
   - 只有2次预热/每侧3窗L1，无重复Event/动态metadata。未采用、不扩档、不重测。
   driver新增--paired-kv-signal-spmd-only，复用已有compare，原主Q模式保持。
   见KV_TOUCH_SIGNAL_EXPERIMENT.md、kv_touch_signal_paired_result_v1.json。
4. CPU L1原位循环小例子：probe_tile_inplace_loop.py及_assigned.py。
   - v1少SPMD index被前端拒绝；加worker用于输出行偏移。
   - v2忽略assemble返回值，无法原位复用Mat，因保留区域Mat→Mat复制而失败。
   - v3 `pool=pl.tile.assemble(pool, product, offset)`完整CPU/PTOAS通过。
     实际固定Mat512B，循环内TINSERT原位写，循环后TEXTRACT；无Mat→Mat移动，
     有FIX→MTE1同步。见TILE_LOOP_INPLACE_CPU.md及两个完整codegen文本。
   仅小FP16/无分支/无pre_quant，没有设备执行，不能当完整CSA正确性或性能证据。

全部NPU和CPU exec已收取终态，无运行任务。PENDING_JOBS为终态快照。
本轮未改生产/F/依赖/共享安装。所有新增候选默认关闭，selected_shape_policies.json不动。
原动态PTO对照与正式第4层权重，恢复fixture，不代表连续decode；未跑Native forward。

## 下一方向，尚未实现

CPU小例子说明“固定Mat池+接收DPS返回值”可避免某类循环回边移动。
可另写N64固定L1环形池候选，保留原M384/N64/K128和WS K384：
Key页进固定三槽L1；score通过assemble回写固定两槽L1；循环内部提取当前Key Right
和上一score Right，不将Right作为显式循环携带值。先剥离QK0、循环QK(current)/WS(previous)、
最后排空WS15，原1024交接/页索引/叶计划/AIV不动。仍须CPU审计才考虑B16设备。
这个写法可能减少原先Right提前预取深度；不能假定比原展开快。
小例子未覆盖Native页if分支、INT32→FP16 FIX pre_quant和真实大Tile，不能外推。
旧rolled v1/v2和group2负结果均保留，不能重复旧写法或预记PASS。

## 既有结果和不要重测的方向

SELECTED_SEVEN_RESULTS.md/selected_shape_policies.json，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；8K B16/B24/B32=621.96/836.74/905.72μs。
只长B8/B16选QK地址缓存，只长B16选row deps。100tick cached wait。
B16参数--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100
--qk-vector-indices --row-dependencies。动态metadata已通过，不重复。
仍为实验入口，未接生产自动派发；长度/位置/页表仍device Tensor。

前轮完整交接见handoff_snapshots/NEXT_STEP_20261017_0849.md、0800/0732/0638/0550。
N64group2 920.58、2048交接834.68、主Q双列806.46但同进程0.36%无收益，均未采用。
Key大L1池CPU未减少等待，未上卡。短尾/退避/N80/各QK/OA/HC负结果勿重复。
权重cache属性恢复C++全文相同，勿再上卡；当前Simpler kernel无SDMA；
原O-A已有K128 Left/Right双缓冲，SWA已按物理页批量gather，rowdeps已有传递约简/observed过滤。
阶段探针row_deps_stage_analysis仅供归因，不是700us验收或下限：
Indexer约234..462，QK/PV482..647，O-A650..735，HCpost773..796μs。

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻static feasibility，冻结只读。
P=R/.worktrees/pypto-csa-static-fixes-20261016，S=R/.worktrees/simpler-csa-core-args-256-20261016。
只改N并镜像canonical同名目录；不改生产/F/依赖/共享安装，不做hash，不开子agent。
NPU仅task-submit --device auto；任务运行时不改driver/已导入helper。
静态a919f119已push；动态N未commit，无新提交指令。
镜像可用新脚本mirror_process_files.py --destination C/tests/pypto_test/csa_multishape_spmd_20261016。
按mtime_ns/size增量复制≤2MiB白名单文本，不做hash、不删文件，排除缓存/大产物；
canonical LOCAL_ARTIFACTS.json列出本地保留路径。

<!-- SOURCE_END history-document-113 -->

---

<a id="history-document-114"></a>
### 历史原文 114：handoff_snapshots/NEXT_STEP_20261017_0958.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0958.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_0958.md)；原文 6,561 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-114 -->
# 接续：2026-10-17 09:58

Goal active：按batch/长度策略分别编译，128K/B16/S6完整无插桩L1<700μs。
未达到，不暂停、不标blocked。用户允许hc_pre式分档，不需要一份二进制覆盖所有batch。
现选七档保持；B16历史807.16μs，最新同进程现选807.50μs（之前KV配对任务）。

## 本轮完成，勿重复

1. 新增indexer_l1_ring.py/build --indexer_l1_ring/设备--indexer-l1-ring。
   保留M384/N64/K128 INT8 QK、M16/N64/K384 WS、FIX 2^-10/ReLU、原1024交接、叶计划与AIV。
   仅替换Indexer AIC叶计算体，原prefix/drain及其他函数AST核对保持。
2. v1：Key三槽/score双槽，循环末尾if不预取，CPU失败于Mat同地址188416自拷贝。
   v2剥离最后QK/WS，循环1..14固定预取下一Key。完整codegen通过，但实际地址审计失败：
   第一个页加载后的phi17分到Mat0，第二页写0，而下一Key提取/loop carry仍188416。
   无拷贝修正，不允许上卡。证据indexer_l1_ring_v2_misaligned_pool_evidence.pto.txt/rejected_audit。
   v3把两页加载放在同一分支叶子，CPU仍因Mat→Mat地址空间组合被PTOAS拒绝。
   三版均无NPU结果；所有source/passes/report保留，source_snapshots有v2/v3与旧审计。
   不改依赖、不能把编译通过当数据流正确。
3. v4：撤去Key循环携带，每轮局部64×128 Key池；只让score384×128双槽跨循环。
   QK0首块，loop1..15执行QK(current)/WS(previous)，最后WS15；源slot模型覆盖32页/1024列一次。
   prepare_v4/cpu_v4完整通过；实际Key Mat163840、score65536，循环QK/WS Acc0/98304，
   Right49152/0。无Mat→Mat/Right→Right、循环无PIPE_ALL。实际阶段4matmul/2FIX/2store静态位点。
   indexer_l1_ring_codegen_audit_v4.json及stage/loop C++保存；driver首次执行前审计实际artifact。
   终态后补充Right区间自动检查，CPU/设备artifact均通过，未重复执行NPU。
4. task_20261017_094749_7744056502 / b16_row_deps_indexer_l1_ring_v4 exit0：
   完整输出/Top-K/六cache-state eager/graph/final逐bit一致，30代信号、所有未写区/workspace PASS。
   L1候选984.56/1034.82/1008.86，中位1008.86μs；同轮动态905.34。
   Event候选1133.55 vs动态951.86。未采用、不扩档、不重复、不加插桩归因。
   没有同轮现选单SPMD对照；807.16/807.50是此前记录，不能写成同轮配对。
   见INDEXER_L1_RING_EXPERIMENT.md和indexer_l1_ring_result_v4.json。
5. probe_loop_gather_phi.py缩减CPU例完整通过，但池/phi/transpose全是Mat0，没有复现完整CSA错误。
   只覆盖循环携带+两次条件gather，缺少外层分支与全图存储复用。
   不宣称编译器bug已修复、不把小例当完整图证明；无设备调用。
   probe_loop_gather_phi_cpu_v1/allocation_audit.json和完整PTO/C++文本保存。
6. 新CPU分析indexer_query_completion_model.py/json：保留原叶所有权，1..32叶共8448item不漏不重。
   现B16长度131078/4=32769，33score块，6叶块数6/6/6/5/5/5；每物理worker22块。
   原叶优先所有请求都22块才ready；每worker按request重排可让前8请求在11块ready，后8仍22。
   **原阶段通知不改仍全部等22块**，不能只重排循环就宣称提前或收益。
   这不是时间模型，不是实际性能，不是数值/编译证明。

全部CPU exec/NPU已收取终态，无运行任务，PENDING_JOBS为终态快照。
本轮只改N及canonical过程镜像，没改生产/F/依赖/共享安装；没有新commit/push指令。

## 下一方向：更细的query完成粒度，先评估完整链路

当前row_dependencies已按真实行收窄，不要重复同样工作。
它仍只能等待原task成员结束，Indexer每成员处理多个request；merge成员跨2个query、
QK成员跨4个query，所以仅重排Indexer叶循环不能让下游提前。

可评估把B16 Indexer分成request0..7/8..15两个阶段，在原owner上分别处理对应叶；
AIC/AIV的相同owner/叶次序必须一致，FFTS每叶排空、score-transfer复用和所有发布缓存语义保持。
动态叶数仍从device kv_seq_lens读取，不把样本的6叶固化成唯一合法值。
需要连同merge/QK消费范围和依赖拆分，才可能发布有用的中间完成点。

另一个限制：proj_a_mm_0的row tile128/实际96行覆盖整批；_proj_b_mm_nz_kernel__2为M96。
O-A/O-B不会自动按前半query推进。更细的行tile会增加权重读取/任务开销，
不能把“前8请求ready提前一半”外推成整层时间缩短一半，更不能宣称已到700。
先用源读写范围、既有阶段证据和模型决定是否值得实现；避免只重排无收益依据的设备测试。

## 现选与前轮信息

SELECTED_SEVEN_RESULTS.md/selected_shape_policies.json，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；8K B16/B24/B32=621.96/836.74/905.72μs。
长B8/B16选QK地址缓存，只长B16选row deps；100tick cached wait。
B16：--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100
--qk-vector-indices --row-dependencies。动态metadata已通过，不重复。
实验入口尚未生产自动派发；正式权重、仅PTO对PTO；恢复fixture不代表连续decode。

前轮完整交接：handoff_snapshots/NEXT_STEP_20261017_0926.md及0849/0800/0732/0638/0550。
N32 981.12、group2 920.58、2048交接834.68、主Q双列同进程0.36%无收益、KV纯信号
同进程812.08 vs现选807.50均不采用、不重测。其他已有负结果查ATTEMPTS，勿重复。
row_deps_stage_analysis只归因：Indexer约234..462，QK482..647，O-A650..735、HCpost773..796。
无插桩完整L1首尾是验收；插桩/CPU模型不是700验收或下限。

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻static feasibility，冻结只读。
P=R/.worktrees/pypto-csa-static-fixes-20261016，S=R/.worktrees/simpler-csa-core-args-256-20261016。
只改N并镜像canonical同名目录；不改生产/F/依赖/共享安装，不做hash，不开子agent。
NPU仅task-submit --device auto；任务运行时不改driver/已导入helper。
静态a919f119已push；动态N未commit，无新提交指令。
mirror_process_files.py --destination C/tests/pypto_test/csa_multishape_spmd_20261016，
按mtime/size增量复制≤2MiB白名单文本，不hash不删；canonical LOCAL_ARTIFACTS.json保留大产物路径。

<!-- SOURCE_END history-document-114 -->

---

<a id="history-document-115"></a>
### 历史原文 115：handoff_snapshots/NEXT_STEP_20261017_1045.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_1045.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_1045.md)；原文 6,869 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-115 -->
# 接续：2026-10-17 10:45

Goal active：按batch/长度策略分别编译，128K/B16/S6完整无插桩L1<700μs。
未达到，不暂停、不标blocked。用户允许hc_pre式分档，不要求一份二进制覆盖多batch。
现选七档保持；B16首轮807.16μs，最近同进程现选807.50μs是此前KV配对任务。

## 本轮完成，勿重复

### 1. 两组query完整流水的CPU模型

`query_group_schedule_model.py`将Indexer/merge/plan/QK/O-A/quant/O-B/HCpost按48query拆两组。
现选行依赖证明、原动态L1成本；1313成员/1121单元/43953数学边。
保持FFTS组切换三成员完成及logical worker GM交接区不重叠；有逐query消费者核对。
理想缩行成本0.5、每新成员1μs。模型不是实测，不代表数值或设备同步正确。
同模型现选771.530μs；以下均未更好，未生成对应设备算子，无NPU：

- 固定MIX物理组：前半优先855.325、earliest780.470、critical813.585。
- 允许每半重新分配MIX物理组但保留GM区间约束：earliest780.470。
- 同样允许MIX重分配，先安排未拆分主Q等前缀、再前半优先：841.340。

见QUERY_GROUP_SCHEDULE_FEASIBILITY.md、query_group_schedule_*json。
不要重复这五种排法，不据此证明所有更细流水不可能收益。

### 2. PV循环状态分块：完整通过，但回退，不采用

`qk_merge_heads.py` v5仍保留完整32-head running state，只缩小incoming到8head。
prepare_v5成功，cpu_v5因Vec196608>188416失败。MemoryReuse实际为4×32768状态+65536共享arena。
`qk_merge_heads_v5_memory_allocation.json`有证据，15个动态view未纳入静态末端枚举。
旧代码保留在source_snapshots/qk_merge_heads_8_full_state_v5.py及此前16-head快照。

v6把每侧循环状态拆成两个16×256独立Tile，不再整状态assemble；初始化、所有分支/
循环yield、reset及最终16-head输出引用一起展开。逐元素乘加次序、m/l及五块次序不变。
所有GM写表达式、scalar metadata读、显式同步、其他函数AST保持。
prepare_v6/cpu_v6完整通过，AIV总UB163872B。
四个16KiB状态地址65536/81920/98304/114688，初值/合并/reset同址且互不重叠。
`qk_merge_heads_codegen_audit.py`在CPU和设备首次执行前均通过。

任务task_20261017_103218_149245329068 / b16_row_deps_qk_merge_state_v6 exit0：
输出/TopK/六cache-state eager/graph/final零差异，30代通知及全部保护区PASS。
L1候选1033.44/1351.48/1029.72，中位1033.44；同轮动态910.98/914.34/912.20，中位912.20。
Event候选1118.08 vs动态959.95。保留全部窗口，未删慢样本。
**不采用、不扩档、不重复、不追加诊断NPU。** 807.16/807.50不是这轮现选配对。
完整报告QK_MERGE_STATE_EXPERIMENT.md、qk_merge_heads_result_v6.json。

只读核对真实C++各阶段静态计数，只有QK AIV stage30变化：PIPE_ALL9→8、set/wait各80→82、
TLOAD22→24、row_expand_mul4→8。其余阶段这些计数相同，不代表所有指令/地址相同，
也不能定量归因回退。证据qk_merge_heads_full_codegen_comparison_v6.json。

### 3. Indexer WS Right加载顺序：CPU完成，1/15对Right复用，未上卡

先核对原体：它已经有16个N64展开的QK(current)/WS(previous)及下一Key预取。
不能重新实现同样流水再称优化。新增`indexer_ws_load_order.py`仅将15处WS Right
extract移到当前QK matmul之后，保持全部434顶层语句AST多重集合、其他核体及末5条排空。

prepare_v1/v2是变换器遍历被交换列表导致重复匹配失败；cpu_v2因source不存在退出，未编译。
prepare_v3修遍历后，对drain名称假设错误而失败。v4核对真实buf_last_pair和末5条AST后成功。
这些是试验变换器错误，不是PyPTO问题；日志都保留。
只在CPU准备进程注入catalog，没有新增build_variant/device driver开关，默认路径不变。

indexer_ws_order_prepare_v4 / cpu_v4完整通过。审计脚本`indexer_ws_load_order_codegen.py`：
当前QK panel1 Right地址8192、前块WS panel0 Right地址0/49152B，**只有这一对**重叠；
其余14对仍不重叠。原版15对全部不重叠；两版Acc均不重叠。
静态matmul32、PIPE_ALL1、M→MTE1wait32、MTE1→Mwait37相同，set/wait166→165。
少一组set/wait不是性能依据。尚无NPU，见INDEXER_WS_LOAD_ORDER_CPU.md及审计JSON/C++。

## 下一步可推进的明确候选

上一段只有首对Right被复用：可保留panel1/WS0原加载顺序，仅交换其余14对。
先检查实际Right区间、C++中M/MTE1提交与等待落点是否真的改善，不能只数静态指令。
若仍复用或同步序列没有实质改变，不上卡。不删保护实际复用的等待，不扩大容量，不改依赖。
这份14对方案尚未实现。若有足够生成代码依据，再加独立选项并只测B16/128K。
不要重复整圈N64 L1环、panelgroup或N32等已完成负结果。

## 现选、边界与操作

SELECTED_SEVEN_RESULTS.md/selected_shape_policies.json，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；8K B16/B24/B32=621.96/836.74/905.72μs。
长B8/B16选QK地址缓存，仅长B16选row deps；100tick cached wait。
B16：--balanced-schedule --schedule-policy earliest_ready
--schedule-cost-profile N/b16_dynamic_l1_cost.json --cached-wait --wait-backoff-ticks 100
--qk-vector-indices --row-dependencies。动态metadata已验证，不重复。
实验入口尚未生产自动派发；正式权重、仅PTO对PTO，恢复fixture不代表连续decode。
700验收只用完整无插桩L1首尾，不能用Event/CPU模型/局部时钟代替。

本轮所有CPU exec与唯一NPU任务已收取终态；PENDING_JOBS是终态快照，恢复时复核。
只改N及canonical过程镜像，没有改生产/F/依赖/共享安装。动态N仍未commit，无新提交指令。
上轮交接handoff_snapshots/NEXT_STEP_20261017_0958.md，再早0926/0849/0800等；所有失败查ATTEMPTS。

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻static feasibility，冻结只读。
P=R/.worktrees/pypto-csa-static-fixes-20261016；S=R/.worktrees/simpler-csa-core-args-256-20261016。
只改N，镜像canonical同名目录；不做hash，不开子agent；NPU仅task-submit --device auto。
运行设备任务时不能改driver/它导入的helper。静态a919f119已push且保留。
CPU编译用F/bootstrap_isolated.py F/compile_jit_entry.py --memory-planner PYPTO --kernel-mode；
先source R/env-dsv4-0251rc1.sh及已有CSA_STATIC_PYPTO_ROOT/HCA_SIMPLER_ROOT/PYTHONPATH变量。
镜像：python mirror_process_files.py --destination C/tests/pypto_test/csa_multishape_spmd_20261016。
按mtime/size增量，≤2MiB白名单文本，大产物路径记LOCAL_ARTIFACTS.json，不hash不删。

<!-- SOURCE_END history-document-115 -->

---

<a id="history-document-116"></a>
### 历史原文 116：handoff_snapshots/NEXT_STEP_20261017_1125.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_1125.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/handoff_snapshots/NEXT_STEP_20261017_1125.md)；原文 7,351 字节。交接、旧目标或工作区快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-116 -->
# 接续：2026-10-17 11:25

Goal active：按batch/长度策略分别编译、各档调优，128K/B16/S6完整无插桩L1<700μs。
未达到，不暂停、不标blocked。用户允许hc_pre式分档，不要求一份二进制覆盖多batch。
本goal轮有进展：修正测试驱动编译顺序并重做三项受影响配置；实现/编译/验证新的Key预取候选。
七档选择保持；B16现选首轮807.16μs，此前KV同进程配对现选807.50μs。

## 本轮重要更正

旧ring v4、PV状态v6、WS加载顺序v5的artifact审计在ExitStack外提前调用
_resolve_kernel_artifact，缓存了默认TWAIT二进制，未应用请求的cached_wait/100tick。
约1000μs的测量是真的，但不能由此认定数学核改动回退。精度针对旧二进制仍有效。
修正driver：所有编译扩展进入ExitStack后，再执行三个artifact审计。
新增signal_polling.audit_wait_artifact，执行前检查change.json和真正CCEC输入
kernels/aic/csa_static_aic.cpp、kernels/aiv/csa_static_aiv.cpp，要求74处替换等待、
DCCI后读取、100tick常量。ptoas/*.cpp保存的是替换前代码，不能用它判断实际等待策略。
扫描74项已完成cached_wait测试，只有上述三项缺回调且实际源码未替换。
证据compile_context_audit_20261017.json、compile_context_scope_audit_20261017.json，
完整说明COMPILE_CONTEXT_CORRECTION.md。原始result和旧driver快照保留。
最早manifest没有后来ramp_backoff字段，初次CPU正例失败；换用已有明确false的正确产物
验证正例，三个错误产物均被拒绝。即使给错误源码搭配正确manifest也会拒绝。

## 修正配置后的设备验证已完成，不得重复

全部128K/B16/S6、正式权重、PTO对PTO、exit0；完整eager/graph/final输出/TopK/六cache-state
零差异，30代通知和保护区PASS，实际地址及等待策略门禁通过。恢复fixture不是连续decode。

| 候选 | L1三窗μs | 中位数 | 同轮动态PTO | 目录/任务 |
|---|---|---:|---:|---|
| WS后14对加载后移 | 814.12/812.34/816.48 | 814.12 | 899.98 | b16_row_deps_indexer_ws_order_v6 / task_20261017_110148_204947624179 |
| PV循环状态分块 | 820.58/812.90/820.24 | 820.24 | 909.56 | b16_row_deps_qk_merge_state_v7 / task_20261017_110518_211373122343 |
| N64单面板L1环 | 804.14/813.50/806.38 | 806.38 | 885.92 | b16_row_deps_indexer_l1_ring_v5 / task_20261017_110544_21223701219 |

没有替换七档选择。单面板环接近现选，不足以证明稳定收益。没有本轮现选配对，不按跨轮小差值归因。
摘要qk_merge_heads_result_v7.json、indexer_l1_ring_result_v5.json、indexer_ws_order_result_v6.json。
PV源码仍为四个独立16-head循环状态，实际UB163872B；前版完整状态196608B失败保留。
WS只保留首对load→QK，后14对QK→load，15对Right/Acc均分离。
它只是把QK提交提前到WS加载的阻塞等待前；原实现已经允许计算/搬运重叠，不能声称首次实现。

## 新的N64两面板Key预取也已完整通过，无收益

新增indexer_l1_key_pairs.py、prepare_indexer_key_pairs.py；--indexer-key-pairs默认关闭。
首块0、循环(1,2)…(13,14)、末块15及排空。每对先提取第一Key Right，再提交第二Key GM→L1，
然后第一QK/前块WS，最后提取第二Key Right并执行第二QK/第一块WS。
只让score双槽跨循环；Key/Right局部。保留M384/N64/K128、WS K384、FIX2^-10/ReLU、
原叶计划/1024交接/AIV，不改生产或依赖。prepare_v1/cpu_v1完整通过；集成入口文本与已编译源码一致。
扩展indexer_l1_ring_codegen_audit.py的--key-pairs模式，原单面板实际产物仍通过默认模式。
真实L1 score98304B@65536，Key@163840；三对QK/WS Right/Acc分离，无同层搬运、循环PIPE_ALL。
第二Key TLOAD静态位点269/302/356/389，391发完成事件；首QK415/WS427，484才等第二Key，次QK513。
这是C++提交顺序，不代表已测重叠时长。

task_20261017_111540_229620229969 / b16_row_deps_indexer_key_pairs_v1 exit0，全门槛PASS。
L1 813.42/814.14/811.40，中位813.42；同轮动态898.80。未采用、不扩档、不重复。
见INDEXER_KEY_PAIRS_EXPERIMENT.md、indexer_key_pairs_result_v1.json。

## CPU模型补充和下一步

query_group_schedule_model.py新增main_q_then_phase0：只保护主Q两阶段、再前半优先，
允许MIX重分配。模型862.105μs，不优于同模型771.530；六种模型均没有收益，无设备候选。
query_group_schedule_optimistic_bounds.json：给定成本表下AIC工作量/24界，原665.622、
两组理想拆分672.955μs。不是硬件下界或700可达证明。selected_member_model_spans.json是模型跨度。

不要继续重复面板展开/分组扫参。可检查阶段边界的缓存维护，但先阅读旧precise_payload_cache实验，
避免重复旧方案。一个待评估方向：按实际Scalar GM读写和Tile DMA读取分类，判断是否有可证明冗余的
整DCache维护。尚未实现；不能据此删fence、DSB、信号DCCI或跨核发布。必须追踪参数别名、workspace、
Scalar metadata跨调用更新和同步语义，无法证明则放弃。Scalar非一致性约束和精度门槛不变。

本轮只读核对还发现：QK EARLY_KV_MIN_TOKENS=144有历史实测依据，小规格曾无收益；
HCpost已是Tile直传，没有out_bf16 GM往返。不要将这两项当新发现重复上卡。
Indexer零拷贝页视图旧CPU失败见ATTEMPTS 01:34，没有新lowering方案不要重复tensor.slice。

## 现选、状态和操作约束

SELECTED_SEVEN_RESULTS.md/selected_shape_policies.json，S6：
128K B4/B8/B16/B24=541.92/641.56/807.16/1121.12μs；8K B16/B24/B32=621.96/836.74/905.72μs。
长B8/B16选QK地址缓存，仅长B16选row deps；100tick cached wait。
B16使用balanced-schedule、earliest_ready、b16_dynamic_l1_cost.json、cached-wait/100tick、
qk-vector-indices、row-dependencies。同图动态metadata已验证，不重复。
实验入口未接生产自动派发；正式权重/data/model/DeepSeek-V4-Flash-0731-w8a8。
验收只用完整无插桩L1首尾；Event、模型、阶段探针不能替代。

本轮所有CPU/NPU任务终态，PENDING_JOBS为快照。误提交task_20261017_110519_211380121075参数不完整，
立即kill，exit130，无性能结果；目录保留。不得改运行中任务导入的driver/helper。
只改N及canonical过程镜像，生产/F/依赖/共享安装未改。动态N未commit，无新提交指令。
静态a919f119已push且保留；前一交接归档handoff_snapshots/NEXT_STEP_20261017_1045.md。

R=/data/pyptouser/qinchuanyu/pto-eager；D=R/.worktrees/csa-spmd-dynamic-20261016；
N=D/tests/pypto_test/csa_multishape_spmd_20261016；F为相邻static feasibility，冻结只读。
P=R/.worktrees/pypto-csa-static-fixes-20261016；S=R/.worktrees/simpler-csa-core-args-256-20261016。
无hash、无子agent；NPU仅task-submit --device auto。
git用--git-dir=R/.git-repositories/vllm-ascend.git/worktrees/csa-spmd-dynamic-20261016 --work-tree=D。
CPU用F/bootstrap_isolated.py F/compile_jit_entry.py --memory-planner PYPTO --kernel-mode。
镜像python N/mirror_process_files.py --destination C/tests/pypto_test/csa_multishape_spmd_20261016，
mtime/size增量，≤2MiB白名单文本，大产物仅记LOCAL_ARTIFACTS，不hash不删。

<!-- SOURCE_END history-document-116 -->

---

<a id="history-document-117"></a>
### 历史原文 117：COMPILE_AND_RESUME.md

> **当时快照，全文保留。** 来源：[tests/pypto_test/csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/COMPILE_AND_RESUME.md)；原文 9,923 字节。静态编译交接快照，全文保留，不作为当前状态。

<!-- SOURCE_BEGIN history-document-117 -->
# 静态 CSA 的完整编译现状与接续方法

> 2026-10-16 更新：用户已授权独立 PyPTO worktree 修复；CORE 扩容、
> 共享 InOut 单 MIX 小例子真机执行均已通过；完整 B4 v21 默认 PYPTO 代码生成
> 已通过。完整 Torch 注册新遇到 64 入参限制，见
> [注册限制与入口方案](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/TORCH_SCHEMA_CAPACITY_20261016.md)。尚无完整设备核结果。
> 后续 B4 program 产物的 CCE/调度二进制编译及组装也已通过，见
> [二进制编译记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/BINARY_COMPILE_20261016.md)。
> 本文以下为修复前诊断与接续约束，
> 最新证据见 [独立 PyPTO 修复记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_STATIC_FIXES_20261016.md)。

本文件针对 `static/*/entry.py` 的融合原型。它与原动态 CSA 的真机 PASS、
PyPTO #3009 的最小复现分别记录，不能互相替代。没有静态 CSA 真机运行结果。

## 已准备好的接口

`prepared_arguments.PreparedStaticArguments(case_directory, native_call.args)`
直接消费原 `NativeCSACall` 的 56 份 Tensor。初始化建立一个 INT8 workspace，
在 Host 建立其各 dtype 的别名视图；Native metadata/cache/state 和四份 NZ
权重保持原存储。初始化时清零 progress、上传 records/waits，重放不重新初始化。
无每步 D2H、权重转换或 metadata 展开 kernel。2026-10-16 已在独立 Simpler
worktree 完成 256 Tensor 扩容与设备验证，见 [扩容记录](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/SIMPLER_CORE_EXPANSION_20261016.md)。

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
[MEMORY_REUSE_PHASES_REPRO.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/MEMORY_REUSE_PHASES_REPRO.md)。

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
[JIT_SINGLE_GROUP_NOTES.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/JIT_SINGLE_GROUP_NOTES.md)。

最新 inline wrapper 的非共享 InOut 小例子已完整 CPU 编译和 Meta 注册通过。
B4 生成器采用 inline wrapper 后，v6 已通过 SPMD 和 InOut 使用检查，之后
在 Group ABI 统一中暴露共享 InOut 别名问题。5 参数最小复现的首个破坏点是
`ExpandMixedKernel`，详见 [PYPTO_GROUP_SHARED_INOUT_REPRO.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/PYPTO_GROUP_SHARED_INOUT_REPRO.md)。
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

<!-- SOURCE_END history-document-117 -->

---

<a id="external-references"></a>
## 外部只读参考原文

这些文件是外部资料快照，不属于本会话编写的尝试记录，因此只列链接；原快照不删除。
引用链接保留当时来源，不表示本次重新核验了上游最新状态。所有 `upstream_issues/**` 的本会话 issue 正文已在前文全文纳入。

- [tests/pypto_test/csa_static_spmd_feasibility_20261015/references/simpler_atomic_guides_20261016/ATOMIC_USAGE_GUIDE.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/references/simpler_atomic_guides_20261016/ATOMIC_USAGE_GUIDE.md)（45,537 字节）；[上游原文](https://github.com/nalinaly/simpler/blob/fdwic-swimlane-deps/tests/atomic_probe/ATOMIC_USAGE_GUIDE.md)。外部只读参考资料；保留原文链接和本地快照，不重复全文。
- [tests/pypto_test/csa_static_spmd_feasibility_20261015/references/simpler_atomic_guides_20261016/cache_preload_usage_guide.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/references/simpler_atomic_guides_20261016/cache_preload_usage_guide.md)（46,490 字节）；[上游原文](https://github.com/nalinaly/simpler/blob/fdwic-swimlane-deps/tests/atomic_probe/cache_preload_usage_guide.md)。外部只读参考资料；保留原文链接和本地快照，不重复全文。
- [tests/pypto_test/csa_static_spmd_feasibility_20261015/references/simpler_atomic_guides_20261016/icache_miss_usage_guide.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/references/simpler_atomic_guides_20261016/icache_miss_usage_guide.md)（145,440 字节）；[上游原文](https://github.com/nalinaly/simpler/blob/fdwic-swimlane-deps/tests/atomic_probe/icache_miss_usage_guide.md)。外部只读参考资料；保留原文链接和本地快照，不重复全文。
- [tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_records/csa-torchair-library-readme.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_records/csa-torchair-library-readme.md)（8,467 字节）；[上游原文](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/torch_library/README.md)。外部只读参考资料；保留原文链接和本地快照，不重复全文。
- [tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_records/csa-torchair-pybind-readme.md](../../../tests/pypto_test/csa_static_spmd_feasibility_20261015/upstream_records/csa-torchair-pybind-readme.md)（8,455 字节）；[上游原文](https://github.com/Ascend/torchair/blob/e1118e8b0fc1b95190ab53b732c0872989550e20/examples/_kernel_extension_aclgraph/pybind/README.md)。外部只读参考资料；保留原文链接和本地快照，不重复全文。

<!-- CSA_SPMD_HISTORY_ARCHIVE_END -->

**历史全文归档到此结束。下面的内容为后续持续记录，禁止再次生成本文件覆盖本区。**

<a id="ongoing-log"></a>
## 持续追加

### 2026-10-17：统一工作区、建立总记录及首批生产组件抽取

按用户要求，vllm-ascend 工作统一到 `vllm-ascend-dsv4-pto-0251rc1`，分支为 `dsv4-flash-pto-v0.25.1rc1`。动态实验及历史产物已迁入，原静态基线与失败尝试保留。远端 `a919f119` 已以本地合并提交 `643d8039` 合入；这条迁移记录本身不表示已 push。原日志中的旧绝对路径仍表示当时实际执行位置，不伪造历史运行位置。

用户随后要求把正式动态 SPMD 实现放在 `vllm_ascend/ops/pypto/`。首批已抽取到 `deepseek_v4_flash_csa_spmd/` 的组件为 `__init__.py`、`kernels.py`、`policy.py`、`signals.py`、`signal_codegen.py`；测试侧策略和信号 lowering 改为薄包装，生成器改读正式数学核体。首批源码等价审计记录了 73 个数学函数 AST、信号 AST、576 个策略输入及 5 个拒绝用例，并检查生产文件没有测试目录导入。证据见 [source_equivalence.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_migration_20261017/source_equivalence.json)。

**首批组件迁入，完整生成器和后端退避仍待独立化；尚非完整生产接入。** B17/H65536 新入口的完整生成源码 AST 与迁移前一致，仅 signals 模块导入路径变化；证据见 [entry_equivalence.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_migration_20261017/entry_equivalence.json)。新入口 CPU lowering 的 `full_codegen_passed=true`，正式 signals 包导入通过；证据见 [cpu_lowering/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_migration_20261017/cpu_lowering/result.json)。没有运行 NPU；源码/CPU 证据不代替真实 runtime ABI、设备精度、性能或整网验证。七档至少降低 30% 的完整目标仍未达到。

### 2026-10-17：生产依赖闭包审计最终补充

下文是归档完成后收尾的最终审计稿，补齐 10 个 NZ 原体、67 个返回别名及参数顺序处理。
前文对应文档保留第一次汇入时的快照；此处追加最终全文，不覆盖历史。
这是静态源码审计，尚未实现完整生成器迁移，也不是新的运行或性能结果。

来源：[production_spmd_package_audit_20261017/README.md](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_spmd_package_audit_20261017/README.md)。

<!-- CSA_SPMD_PACKAGE_AUDIT_FINAL_BEGIN -->
# CSA 动态形状单 SPMD：生产包依赖闭包审计

2026-10-17。只读代码设计审计，没有运行测试、重新编译、启动 NPU 或修改共享 helper。
本报告与同级 `hc_front_cost_audit_20261017/` 成本报告配套。工作目录为当前 canonical 仓库；
旧动态 worktree 只作为历史来源，不作写入目标。

**迁移进展补记：** root 已将 73 个数学核体迁入生产 `kernels.py`，并迁入 `policy.py`、
`signals.py`、`signal_codegen.py`；新的验证已读取这些生产实现。下文的“当前来源”表
用于交代原依赖闭包，不能误读为要求再复制这批核体。下一步重点是消除剩余逻辑/late
IR 的重复依赖，以及收敛生成与绑定入口。新增只读 AST 审计见 `logical_contract_audit.json`。

## 结论

正式实现可以放到 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd/`，而且可以完全不读取
七份历史 anchor、trace、binding JSON 或成本 profile。现有 `build_source_schedule` 已具有
这种算法能力：`anchors={}` 时构造任务图、MIX 分组和逐核调度，未测核使用模型权重 10.0。
这不是各档性能已经得到保留的证明；当前 selected 七档使用的是实测成本，换成常量后
分核和等待表可能变化，不能继承原七档耗时或设备 PASS。

目前 `range_policy.build_range_variant` 仍不独立：有同构 anchor 时先走旧重绑定，只有
缺少 anchor 才走源码 DAG。正式入口应把**所有规格统一走源码 DAG**，保留区间策略，
删除“找不到旧 anchor 才允许新形状”的控制结构。

最大未完成项不是 DAG，而是数学源码和运行入口的正式化。现在生成器读取冻结 IR，
运行测试还依赖测试参数准备器及全局编译回调。单纯搬 `range_policy.py` 并改 import
路径不构成独立生产算子。

已经迁入的生产数学体应继续作为唯一维护点，不能再维护一份 tests 候选核体作为线上
依赖；本报告以下给出进一步压缩剩余接口源码的可行方法。

## 当前 selected 路径的依赖闭包

下表只含当前已选路径，不把 `shape_specialization.py` 的几十个历史实验全部带入。
N 表示本目录上级，F 表示同级 `csa_static_spmd_feasibility_20261015`。

| 环节 | 当前来源 | 性质与迁移要求 |
|---|---|---|
| 区间选择 | N/range_policy.py `select_policy` | 纯 Host 描述符算法。保留 B/T/S、长度容量、分支边界；删除 selected JSON 与 CLI 参数处理 |
| 编排及 Tensor view 绑定 | F/bind_orchestration.py | 算法本身无需测量，但当前解释冻结逻辑 IR；保留 TaskRef、dummy、scope、manual_dep、别名及方向合同 |
| 数学源码和效果 | F/generated/core_bodies_ir.py、F/references/symbolic_ir 的 Outline/ExpandMixed 文件 | 算子代码来源，不是性能 trace；当前是历史文件依赖，必须转为包内受维护的源码/明确接口合同 |
| DAG | N/anchor_gap_design_v1/source_schedule.py、audit_dependencies.py 的核心函数 | 可纯源码生成，RAW/WAW/WAR + 显式依赖；移除审计 CLI、F 常量和绝对路径 |
| 分核 | N/balanced_schedule.py | 核日历、earliest_ready/critical_path、MIX 三成员配对是纯算法；成本 profile 读取不是必需项 |
| 等待偏序 | N/shape_specialization.py `rebuild_waits`，F/build_schedule_manifest.py `ancestor_sets` | 抽取纯函数；不能为了一个拓扑工具 import 整个 trace 读取模块 |
| QK 索引优化 | N/qk_index_cache.py | long 且 32<T<=96 的已选优化；纯源码变换但依赖 SSA 名和源合同，适合写成正式数学分支 |
| B16 行依赖 | N/row_dependencies.py | long、T=96、B=16 的已选优化；按函数名查任务，不靠旧 task_id；保留行映射证明和 FFTS 资源边 |
| 工作区与成员批次 | F/prepare_static_layout.py `prepare` | 纯算法；不复用临时 allocation，布局含控制表和进度区；可迁移 |
| 入口生成 | F/generate_static_entry.py `generate/jit_source` | 纯代码生成 + 数学源码依赖；含必要的 NZ、动态维度和别名处理，不能只保留调度外壳 |
| GM 信号 | N/signal_lowering.py、F/local_signal_adapter.py | 算子必需的发布/获取协议，迁为包内相对 import；保持 payload fence/cacheinvalid、bar_all 和 signal notify/wait |
| 等待退避 | N/signal_polling.py | selected 的 cached+DCCI+100 tick 来自这里；当前 monkey-patch 全局编译函数，不宜原样作为生产入口 |
| Native Tensor 绑定 | F/prepared_arguments.py | 可抽取，当前从 tests JSON 读布局；正式版直接接受生成的合同/布局对象 |
| 实际 JIT 调用 | N/device_compare.py 的少量入口逻辑 | `module.entry(*core_args, config=options)`，直接 native prepare/enqueue，不依赖 torch.library 大参数注册；测试驱动不可导入生产 |

`reuse_gm`、HC front/stream 实验、其它 Indexer/QK 实验、stage clock、协议随机推进、L1 采集、
Native 对照和 fixture 均不属于当前生产最小闭包。

## 无 anchor 生成的具体合同

当前已有可调用 API：

```python
schedule, audit = build_source_schedule(
    bindings, catalog, anchors={}, policy=policy.schedule_policy,
    model_cost_overrides=None,
)
```

它已经做一次 `rebalance`，返回含 `tasks/call_bindings/per_core_sequences/root_arguments/
temporary_allocations/wait_plan` 的完整 schedule，可交 `prepare/generate`。不应再把它当作
历史 anchor。若兼容旧二次 `rebalance`，第二参数应是 `audit['cost_seed_anchor']`，但新入口
直接删除重复分核更清楚。

当前 `rebalance` 为每个模型成员成本再加 1.0，所以无覆盖值时实际调度使用 11.0。
这属于无量纲分核权重；旧结构以 `start/end ×50` 承载它，不应输出成预测的实测微秒。
生产版建议直接传 `cost[(task, role)]`，把 seed anchor 名称及伪时间字段从接口中去掉。
可以先使用统一常量，后续按矩阵块数/历史切片数提供明确的算法权重；两者均不需要
历史 profile。权重只影响映射和顺序，不能用于删除数学依赖或宣称 30% 达标。

源码 DAG 替代七 anchor 必须保留：

1. 逻辑接口与后期物理核的读写效果并集。仅看前期普通 `pl.Tensor` 会漏写效应；例如
   `hc_widen_rms` 的 `inv_rms`。已有 allocation 写保守包含读取，并处理 RAW/WAW/WAR。
2. TaskRef 的本轮 ordinal、嵌套 deps、dummy 展开、manual scope 和 manual_dep 原语义。
   手动作用域不能按整个共享 allocation 强行补成全串行链。
3. 每个逻辑 block 覆盖一次；MIX 对应同物理组的 AIC+AIV0+AIV1。保持原数学体内部 FFTS
   信号的资源顺序，不把任务偏序证明误作全部硬件同步证明。
4. 实际 Tensor view 的 allocation、offset、dtype、shape、stride、NZ 格式及返回别名。
   当前算法不自动发现不同根参数的运行时 storage 别名；入口需验证/约束已有别名合同。
5. 未识别的参数注解、task 选项、作用域或核角色组合明确失败。不能静默套用旧模板。

已有 CPU 证据：七档均为 66 个数学任务，旧图 98 条直接边、源码图 101 条直接边，但
可达偏序均为 1377 对，缺失/新增偏序均为 0；WAR 新识别的边也已被原偏序覆盖。
20 个 CPU 案例覆盖无同构组合和短 Vector 分支。**这些历史检查使用已有的成本种子，
并不等价于本次常量成本版本已跑过设备。** B12/H65536、B17/H65536 的后续真机 PASS
是已生成候选的证据，也不能自动覆盖正式包重构后的所有调度。

## 数学源码如何收敛，不能直接搬整份历史 dump

当前三份源码分别为 7,488、14,751、17,619 行，合计 39,858 行；其中存在重叠。
`SourceContracts` 读取 late IR 主要为了核对 catalog 的参数方向；`generate` 的主体来自
core body bundle，但遇到原 NZ 输入时会改用 Outline 阶段逻辑核体，避免 rank-5 NZ load
被二次 lowering 成错误的 Tile。不能盲目只留 late body，然后声称 NZ 路径无变化。

建议最小、可审查的收敛方式：

- 包内保留各分支实际所需的数学函数，包括未在七点出现的短 Vector、B1～3、中间 B
  组合；不要依据七份 bindings 剪掉合法区间。所有 scalar/device metadata 动态计算保留。
- 保存每个逻辑任务的正式参数、返回别名、效果、物理角色及 worker 语义合同，直接从
  包内源码推导/核对，避免同时维护 late IR 全文件和重复 catalog。
- 原 NZ 核保留适合重新进入前端的逻辑数学表达；其它核保留当前已验证的 AIC/AIV 体。
  逻辑编排可先移植现有解释器，但只解释包内受维护的编排源码。
- 让 catalog 由这些包内定义生成，或使用小型源码声明；不从 tests 读取 JSON。旧 SSA
  名相关 AST 变换可先保留严格守卫，随后改为明确的函数参数/数学分支。

这要求一次明确的数学体/接口提取与对照，不是改几个路径即可完成。已有生成器中的
`RestoreBranchYields`、`NormalizeViewValidShape`、维度绑定、InOut 别名回绑、NZ 处理都
有实际前端用途；只有在正式源码自然表达了同等语义之后才能删除。

### 已核实的逻辑 stub 与 late IR 去重方案

本次仅解析源 AST，并执行原 `Recorder.resolve_returns` 的纯 AST 别名分析函数，未执行
编排、数学体、PyPTO lowering 或设备。完整结构结果与 67 份别名映射保存于
[`logical_contract_audit.json`](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_spmd_package_audit_20261017/logical_contract_audit.json)，映射同时列在本文末尾。

**late IR 可去掉：** 生产 `kernels.py` 的全部 73 个物理函数，剔除
`original_block_idx/original_block_num/original_subblock_idx/original_subblock_num` 后，
参数名到完整注解的映射与原 late IR 全部相同，没有缺失或方向/形状差异。
但是 54 个函数的参数列表次序发生了 scalar-first 重排，例如 `hc_widen_rms`、
`hc_pre_linear`、`split_pre_post`。`SourceContracts` 应按参数名比较注解映射；实际生成
调用继续使用生产函数自己的 `inline_parameter_order`。不能只去掉 worker 参数后继续
比较列表，也不能把生产排列当成逻辑函数原调用顺序。

必须区分三种顺序：原逻辑签名顺序用于 `Recorder.call` 绑定参数；物理函数的完整注解
用于效果与接口核对；生产函数含 worker 的顺序用于最终传参。由包内函数推导这些信息
即可，17,619 行 late IR 不需要作为线上依赖保留。剔除的名字应是精确的四项白名单，
并校验其 INDEX 类型及确实来自 worker 改写，不能泛化为删除所有 `original_*` 参数。

**67 个逻辑函数均可解析返回别名：** `Recorder.call` 对非 root 函数只使用签名/参数
注解和 `resolve_returns`，不执行核体。全部 67 个函数的返回都唯一归于已有参数，
没有失败。因此保留原签名及返回注解，将函数体变为直接 `return param`、
`return param1, param2` 或 `return None`，足以服务 CPU 形状编排。必须保留空返回与
单/多返回的差别，不能按 Out 参数数量猜返回值；别名分析结论比参数方向更强。
完整 `_decode_csa_tp1_layer` 编排仍须保留，共 1,513 行，其中包含所有区间分支、
workspace 创建、Tensor view、TaskRef、manual scope 和动态分支选择。

**10 个 NZ 函数不能同时删数学体：** 当前 `generate` 遇到任一 NZ 实参会从 logical
源码读取原逻辑数学体，以避免二次 rank-5 lowering。下列 10 个函数应保留原体，或者
移到单独的生产 NZ 数学模块，并修改生成器明确从那里取体：

1. `_proj_b_mm_nz_kernel`
2. `_proj_b_mm_nz_kernel__2`
3. `_proj_b_mm_nz_kernel__3`
4. `proj_a_mm`
5. `proj_a_mm_0`
6. `proj_a_mm_1`
7. `proj_a_mm_2`
8. `qproj_matmul`
9. `qr_proj_matmul`
10. `qr_proj_matmul_0`

这 10 项按全部逻辑函数签名中的 NZ 参数得到，覆盖各形状分支，不仅是七档调用记录。
最小改动方案保留 root + 10 个 NZ 原体 + 57 个 stub；按源行数与两行 stub 粗估约
2,053 行（不含文件声明/注释格式），代替 14,751 行 Outline dump。后续也可拆为
`orchestration.py` + `logical_contracts.py` + `kernels_nz.py`，但不需要为拆分复制整份 IR。

风险：stub 的生成不能掩盖数学体之后的返回语义变化；需要在源码变更时更新/验证
别名合同。非 NZ 数学体仍以已迁 `kernels.py` 为准，效果以物理注解与逻辑合同的保守
并集为准。stub 仅是 Host 绑定合同，不得被误选作真实生成的计算体。完整 stub 重构
后的绑定/代码生成/精度尚未执行，本次结构可行性不能写成这些验证 PASS。

## 建议的最小生产组织

```text
deepseek_v4_flash_csa_spmd/
  __init__.py              # 稳定调用入口
  policy.py                # CSA 区间/形状策略；无七点 JSON
  math/                    # CSA 数学体与正式编排，按现有职责拆分
  contracts.py             # 形状、效果、返回别名、MIX 及受支持范围
  graph.py                 # TaskRef/dummy/manual + Tensor view DAG、拓扑/等待
  schedule.py              # 分核、算法成本；无 trace/profile 加载
  layout.py                # 工作区及控制表生成
  codegen.py               # 单 MIX SPMD 入口生成、绑定及数学特化
  signals.py               # 单写者进度 notify/wait、代际及缓存维护
  runtime.py               # 参数/工作区准备、JIT/cache、直接调用
```

名称仅表示职责，不要求机械拆成十个文件。`qk_index_cache` 与 `row_dependencies` 可以
归入 math/graph 的带守卫分支，不能将全部历史候选模块一并复制。

`runtime` 应在 warmup/capture 前生成调度和分配工作区，通过对象传递合同/布局；每个
并发调用者拥有独立 workspace/progress。进度初始化一次，图重放不清零，耗尽 INT32
代际时必须在排空后重置。Native 输入、四份 NZ 权重及 state 保留原存储，视图只做
绑定，不做 FP32 权重转换或 state 搬运。

内核 ABI 当前要求 `leaf_tensor_count <= 256` 的独立 PyPTO/Simpler 能力，入口必须按
实际能力检查，不能继续读测试环境变量或通过 tests 的隔离 bootstrap 获得能力。
应保留直接 JIT/native 调用路径，避免重新引入 torch dispatcher 的 64 项注册约束。

## 当前 selected 策略不能遗漏的两个条件

**行依赖：** B16 长历史优化按逻辑函数名重新找 task ID，故 source ordinal ID 本身
不阻塞迁移。但是它还要求唯一的长 Indexer/merge/plan/QK、8 组 O-A、明确 SSA/view
映射和 T96 行划分；不能扩展到 B17 等形状。它额外保留本物理组 Indexer 三成员在 QK
前完成的 FFTS 资源边，这是正确性合同的一部分。

**等待退避：** `local_signal_adapter` 仅有 bar_all + signal_notify/wait；selected 的
DCCI 后 volatile 读 + 100 SYS_CNT tick 退避来自 `signal_polling` 对 PyPTO 私有全局
`_compile_pto_module` 的回调替换。迁移不能悄悄去掉它并沿用旧性能数字，也不宜在可
并发编译的生产进程中原样 monkey-patch。优先形成算子局部、正式受支持的 wait lowering
扩展；若暂保留适配，须明确隔离/串行编译作用域并审查最终 AIC/AIV 源码，不能只检查
选项布尔值。DCCI/DSB 和 payload fence 不可因退避优化被删掉。

## 边界及实施顺序

当前 selector 只接受 S=6、T=B×6、B1～64、总长度容量至 131200；这些是现有能力边界，
不是对最终动态形状目标的重新定义。长/中/短策略跨界时需换模板；档内地址、有效长度、
尾块和循环继续由 device metadata 计算，不把 synthetic metadata 留作真实输入。

建议先完成无历史数据依赖的生产闭包，统一 source DAG + 常量算法成本，形成可调用
且可审核的功能版本。数学体、同步协议、Native 原存储接口先保持明确合同，再把
已选的 QK 索引/行依赖及局部 wait 退避迁进去。测试留在 tests：对照源码偏序/覆盖、
lowering、输出与状态、图重放及最终七档完整耗时分别验证。历史 selected 数字仍留作
历史参照；正式版七档逐档 ≤ 原动态 PTO ×0.70 需新版本自己的证据。

本次没有为上述迁移运行新测试，没有复制大 trace、成本 profile 或 binding 到生产，
没有修改 shared helper、生产代码或依赖仓。

## 附录：67 个逻辑函数的完整返回别名

顺序与逻辑函数的返回顺序一致；`无返回` 表示返回别名列表为空。此表是 CPU 绑定合同，不能取代真实数学体。

| 逻辑函数 | 返回的原参数别名（按顺序） |
|---|---|
| `_proj_b_mm_nz_kernel` | `partials__ssa_v0` |
| `_proj_b_mm_nz_kernel__2` | `partials__ssa_v0` |
| `_proj_b_mm_nz_kernel__3` | `partials__ssa_v0` |
| `comb_sinkhorn` | `comb_t__ssa_v0`、`comb_tail_store_inline449__ssa_v0` |
| `compress_state_commit` | 无返回 |
| `compress_state_commit_0` | 无返回 |
| `csa_cache_writeback` | 无返回 |
| `csa_rope_sign` | `idx_sin_signed__ssa_v0` |
| `csa_row_offsets` | 无返回 |
| `csa_slots_build_valid_qk_plan` | `cmp_sparse_indices_inline4329__ssa_v0`、`sparse_bias_inline4301__ssa_v0` |
| `hc_pre_linear` | `mixes_partials_inline487__ssa_v0` |
| `hc_pre_linear_reduce` | `mixes_raw_inline448__ssa_v0` |
| `hc_widen_rms` | `inv_rms__ssa_v0` |
| `idx_kv_scale_commit` | `idx_native_kv_cache__rv_v2` |
| `idx_qr_dequant_rope` | 无返回 |
| `idx_qr_proj_matmul` | `qr_acc_pad_inline2007_inline4067__ssa_v0` |
| `indexer_boundary_init` | `normed_kv_inline574__ssa_v0` |
| `indexer_head_coefficients` | `coefficients_inline1281_inline8756__ssa_v0` |
| `indexer_head_coefficients_0` | `coefficients_inline2879_inline9040__ssa_v0` |
| `indexer_head_coefficients_1` | `coefficients_inline1735_inline9271__ssa_v0` |
| `indexer_head_coefficients_2` | `coefficients_inline3035_inline9660__ssa_v0` |
| `indexer_score_topk_leaf` | `score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair` | `buf_score_transfer_inline8845__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair_0` | `buf_score_transfer_inline9167__ssa_v0`、`score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair_1` | `buf_score_transfer_inline9382__ssa_v0`、`score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_score_topk_native_pair_2` | `buf_score_transfer_inline9767__ssa_v0`、`score_arena_inline3255_inline4168__ssa_v0`、`pair_arena_inline3226_inline4202__ssa_v0` |
| `indexer_topk_query_merge` | 无返回 |
| `indexer_topk_query_merge__2` | 无返回 |
| `indexer_topk_single_leaf_publish` | 无返回 |
| `kv_and_cache_write` | `idx_kv_scale_values_inline3994__ssa_v0`、`idx_native_kv_cache__ssa_v0` |
| `kv_hadamard` | `kv_final_inline3998__ssa_v0` |
| `kv_proj_matmul` | `kv_fp32_inline3665__ssa_v0` |
| `kv_proj_matmul_0` | `kv_fp32_inline3665__ssa_v0` |
| `kv_rms_norm_rope` | `kv_view_inline3736__ssa_v0` |
| `kv_score_proj` | 无返回 |
| `kv_score_proj_0` | `kv_proj_pad_inline3911__ssa_v0`、`score_proj_pad_inline3938__ssa_v0` |
| `kv_touch` | `ori_kv_flat_inline4326__ssa_v0` |
| `mix_x_rms_norm` | `x_normed_t__ssa_v0` |
| `proj_a_mm` | `o_r_pad_inline4514__iter_v1` |
| `proj_a_mm_0` | `o_r_pad_inline4642__iter_v1` |
| `proj_a_mm_1` | `o_r_pad_inline4770__iter_v1` |
| `proj_a_mm_2` | `o_r_pad_inline4871__iter_v1` |
| `proj_b_act_hc_post` | 无返回 |
| `proj_b_act_hc_post_0` | 无返回 |
| `proj_b_act_hc_post_1` | 无返回 |
| `proj_b_act_hc_post_2` | 无返回 |
| `q_rope_prepare` | 无返回 |
| `qk_pv` | `kv_transfer_inline4342__ssa_v0`、`score_transfer_inline4398__ssa_v0`、`probability_transfer_inline4399__ssa_v0`、`pv_transfer_inline4320__ssa_v0`、`mi_transfer_inline4318__ssa_v0`、`li_transfer_inline4403__ssa_v0`、`o_packed_heads__ssa_v0` |
| `qproj_dequant_rms_nope_rope` | `q_flat_inline8548__ssa_v0` |
| `qproj_matmul` | `q_proj_i32_inline7_inline3585__ssa_v0` |
| `qr_hadamard_matmul` | `qh_acc_gm_inline2065_inline4066__ssa_v0` |
| `qr_hadamard_quant` | `qr_hadamard_i8_inline583__ssa_v0`、`qr_hadamard_scale_dq_inline582__ssa_v0` |
| `qr_proj_matmul` | `qr_fp32_inline1_inline3598__ssa_v0` |
| `qr_proj_matmul_0` | `qr_fp32_inline1_inline3598__ssa_v0` |
| `qr_rms_norm_quant` | `qr_scale_pad_store_inline3589__ssa_v0`、`qr_scale_view_inline8490__ssa_v0`、`qr_i8_matmul_inline3588__ssa_v0`、`qr_view_inline8492__ssa_v0` |
| `quant` | `act_scale_dq_inline4531__iter_v1`、`o_r_i8_pad_inline4534__iter_v1` |
| `quant_0` | `act_scale_dq_inline4655__iter_v1`、`o_r_i8_pad_inline4645__iter_v1` |
| `quant_1` | `act_scale_dq_inline4765__iter_v1`、`o_r_i8_pad_inline4738__iter_v1` |
| `quant_2` | `act_scale_dq_inline4892__iter_v1`、`o_r_i8_pad_inline4874__iter_v1` |
| `rmsnorm_rope` | `normed_kv_inline574__ssa_v1` |
| `rmsnorm_rope_cache_write` | 无返回 |
| `rope_cs` | `rope_sin_signed_inline4321__ssa_v0` |
| `scatter_softmax_pool` | `pooled_kv_inline567__ssa_v0`、`pooled_kv_inline567__ssa_v0` |
| `scatter_softmax_pool_0` | `pooled_kv_inline3908__ssa_v0` |
| `split_pre_post` | `post_t__ssa_v0`、`pre_val_store__ssa_v0` |
| `weights_proj` | `weights_partial_inline3185_inline4185__ssa_v0` |
| `weights_proj_reduce` | `weights_inline3183_inline4184__ssa_v0` |

<!-- CSA_SPMD_PACKAGE_AUDIT_FINAL_END -->

### 2026-10-17：本轮收尾与后续记录约定

主代理已独立复查所有源文档：117 篇本会话记录均可从总记录恢复完整内容，5 篇外部参考原件仍在，
122 篇源文档全部有对应条目。闭包报告在首轮归档后收尾，因此追加最终版本全文，保留首轮快照。
未计算内容 hash。复查结果见
[history_verification.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_migration_20261017/history_verification.json)。

后续正式实现统一写入 [deepseek_v4_flash_csa_spmd](deepseek_v4_flash_csa_spmd/)，
本文件是优化过程的主记录；历史目录继续存放验证脚本、未采用候选、日志和原始证据。
新的尝试按“改动及依据、功能/精度证据、同轮完整 L1 性能、是否采用及原因”追加，失败和撤回也保留。
不重新生成或覆盖上面的历史全文区。

当前没有本会话正在运行的设备任务，未新增真机性能结论。
下一步先迁入精简的完整编排/生成入口及缓存轮询实现，去掉运行时对测试目录和历史 anchor 的依赖；
继续保持按实际形状生成、device metadata 与全部精度/同步门槛。
随后结合 HC 前段已核对的紧凑 GM 双槽设计继续优化，同时处理更大的关键链开销；
不能只凭 HC 局部收益声称七档 30% 已完成。

<a id="production-generator-migration"></a>
### 2026-10-17：完整源码生成链迁入正式包，保留旧版数学与等待语义

本轮延续总目标：完整 CSA 单 SPMD，按实际工作量覆盖区间，七个验收规格完整 L1 incore
首尾耗时逐档降低至少 30%。上一轮的总日志归档已经完成；本轮推进正式实现，未启动 NPU，
没有新增性能数据，也没有将迁移成功写成目标完成。

**正式实现与测试入口。** 正式包位于
[deepseek_v4_flash_csa_spmd](deepseek_v4_flash_csa_spmd/)。新增 `compiler.build_csa`，从本包的
编排、数学体和合同生成完整入口；`GeneratedCSA` 保存源码、绑定合同和布局，
`prepare_arguments`/`PreparedCSAArguments` 绑定 Native Tensor 与调用者独占的 GM 工作区。
生成器不再读取 tests、历史 anchor、trace、绑定 JSON 或 profile。Host 只使用已有
shape/stride/dtype 和 `max_seq_len` 上界；旧验证描述符携带的 `metadata_values` 不参与正式编排。
`binding.read` 只允许原 `kv_seq_lens` 最大长度分支使用显式 Host 上界，其它 Tensor 读一律拒绝；
实际有效长度、位置、页表和有效循环仍在设备数学体中读取。

生产数学仍由 `kernels.py` 的 73 个物理核维护。重复的 late IR 已从运行依赖中移除，
`orchestration.py` 保留完整 root、10 个 NZ 数学原体、57 个原签名/返回别名合同。
逻辑调用顺序、物理核参数按名称的方向合同、scalar-first 最终传参顺序分别核对，不能混为一套。
`catalog.py` 从本包源码与小型角色声明生成参数和 system 操作合同。

`graph.py`、`schedule.py`、`wait_plan.py` 统一从当前源码 DAG 构造所有形状的映射，保留
RAW/WAW/WAR、dummy/manual scope、MIX 配对与成员偏序。成本按 `(call_ordinal, slot)`、
`(kernel_name, slot)`、`kernel_name` 的优先级显式传入，缺省 10.0 是未测模型权重，
不再藏入旧调度器的 `+1`。因此缺省模型生成结果不能沿用旧七档性能。
测试侧如使用同规格历史测量，先核对 shape、调用与核角色，再将旧 task ID 转为调用序号，
显式保留旧 `median_us + 1`；生产包本身不读取这些历史文件。

`qk_indices.py`、`row_dependencies.py`、`worker_loops.py` 仅迁入已选索引缓存、T96 行依赖及
逻辑分片多波次循环，不搬入几十种未采用实验开关。T96 行依赖继续保留本组 Indexer 的三个
MIX 成员完成后才能复用 FFTS 的资源顺序。`wait_codegen.py` 仅提供纯源码变换，保留
DCCI→volatile 读、未就绪 100 tick 退避，写侧发布和业务数据维护不变。

验证入口 `device_compare.py --range-policy` 已改为调用正式包；Native 参数绑定也使用正式
`PreparedCSAArguments`。原实验开关路径仍保留在 tests，供历史候选复现。
测试侧 `signal_polling.py` 的已选 cached/100tick 分支调用正式源码变换，回调安装仍限定在隔离
验证进程中，未将全局 monkey-patch 原样搬成可并发共享的生产 JIT。

**实现过程中遇到的问题与修正。**

1. 首次完整构建在 `layout.prepare` 发现新 catalog 缺少 `system_operations` 字段；已从正式
   数学 AST 提取完整操作、实参和关键字，保留同步协议，随后完整编译通过。
2. 独立审查发现旧测试适配器没有检查不同 root 的可写存储重叠，但源码 DAG 将它们视为不同
   allocation。正式绑定现用 Host storage 地址、偏移与跨度拒绝未建模的写侧别名，保留同一 root
   的派生视图以及纯只读共享。该检查不读取设备元素。
3. 正式绑定补上双向物理格式检查：NZ 必须 29，ND 接受已有 Native 适配合同中的 0/2；不转换权重。
4. `pipeline_check_v1` 把迁移后的源码行号也纳入逐字段相等判断而失败。v2 仅忽略 `source_line`
   与明确更改的 ABI/说明字段，仍比较实际同步操作、全部布局内容与完整数学 AST。失败输出保留。

**本轮实际证据。**

- [绑定与源码合同差分](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/binding_migration.json)：
  root+10 NZ 原体 AST、57 个签名/返回别名、73 个物理参数/worker 顺序及 system 操作合同一致。
  原七档已执行描述符/编排差分；记录明确首次规范化的比较范围，后续 B4 保留 task.scope 严格比较。
- [DAG 与分核差分](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/graph_migration.json)：
  七档各 66 数学任务、1377 个可达依赖对，与原参考闭包一致；B17/H65536 显式同成本下逐核分片、
  顺序、等待和发布一致。短 Vector B1 及非法成本、环/未知前驱、WAR/manual 等边界另有 CPU 检查。
  JSON 区分先前执行与本轮脚本整理，不冒充重新跑过完整七档。
- [完整生产流水差分](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/pipeline_check_v2/result.json)：
  B17/H65536 同成本下完整入口 AST 与迁移前完全相同；B16/H131072 的索引变换、行依赖默认路径
  与旧实现一致。两例 CPU 布局/代际模型通过，各检查 292 个 Tensor 视图。Meta 绑定检查 130 个
  root 输入、58 个 Native 存储视图与 4 份保持原对象的 NZ 权重；跨 root 写侧重叠拒绝、只读共享保留。
- [正式验证入口 B16 对照](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/device_bridge_cpu/variants/h131072_b16/legacy_selected_comparison.json)：
  使用同规格成本重映射后，正式 `--range-policy` 生成源码与历史已选 B16 完整 AST 一致
  （仅规范化信号 import 路径）。仍为 34 个 AIC 阶段、40 个 AIV 阶段、12183 项远端等待。
- [B17 完整 CPU 编译](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/cpu_lowering/result.json)：
  正式包缺省模型调度，PyPTO parse/lowering/PTOAS 的 kernel artifact 生成通过；未启动 runtime/NPU，
  不能据此宣布实际部署 ABI 或精度已验证。编译器对 Scalar 行共享的提示另按布局核对：
  `temporary_14/15` 是唯一 `csa_row_offsets` 成员写入；`temporary_54` 为 INT32[T,16]，每 query
  独占 64B 行。该源码核对不代替设备同步验证。
- [真实生成 CPP 的等待变换对照](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/wait_cpp_and_dependency_check.json)：
  AIC/AIV 源码各 73 处等待替换，与旧 cached/100tick 变换逐字节相同；未再次执行 CCEC/NPU。
  正式 20 个 Python 模块无 tests 导入或历史动态目录硬编码。控制模块 Ruff 通过；抽取数学源码未纳入风格改写。

复现脚本、失败输出与汇总集中于
[production_generator_20261017](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/)，
[summary.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/production_generator_20261017/summary.json)
明确列出证据边界。不做 hash；不删除原静态版本、历史数学、旧生成器或测量结果；本轮没有 commit/push。

**尚未完成及下一步。** 当前正式数学仍为每请求 S=6，不能把支持 B 区间当成任意 query shape
已完成。生产 JIT 的等待回调/缓存接入、不同 S 与混合边界、范围代表点/边界点、七档完整 30%
性能目标均待完成。迁移后的真机验证沿用既有输出/Top-K/cache/state/保护区/图重放门槛。
接下来在已迁正式入口基础上实现 HC 紧凑 GM 双槽候选，并继续处理更大的关键链开销；该候选当前
只是开始实现，尚未采纳或测量。HC 局部收益不能替代完整 CSA 的逐档验收。

<a id="hc-compact-and-query-shape-progress"></a>
### 2026-10-17：统一后续主账，记录 HC 紧凑双槽和 S4 源码探索

**归档范围与后续约定。** 用户明确要求在 `ops/pypto` 下保存全部 SPMD 优化过程，并继续记录。
本文件已经全文收纳两份历史实验树的 117 篇会话文档，包含静态原型、动态分档、全部逐次尝试、
失败分析、9 份 issue 正文/初稿和 PR 跟踪；5 篇外部参考保留来源与本地快照链接。
本次再次盘点两树 Markdown，未发现首次清单以外的新过程文档。生产生成链迁移的完整经过已在
上一节追加；本节补上归档后正在推进的候选。原始文件、脚本、日志、泳道图和编译产物均留在
`tests/pypto_test`，不删除、不移动二进制到正式算子包，也不重复执行首次汇编脚本覆盖新记录。
首页状态和跳转已更新；历史正文中的旧目标、旧路径及当时结论继续作为历史保存。

**HC 紧凑 GM 双槽候选：正在实现，尚未采用。**

源码位于
[hc_pre_direct_stream_v1/transform.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/transform.py)，
接收生成后的源码、布局和绑定合同，返回候选源码、更新后的绑定合同及变换审计。
它不是修改共享 PyPTO/Simpler，也没有替换正式包内的已选数学体。

原 `hc_widen_rms` 先将 `[T,16384]` 输入完整加宽到 FP32，投影等待全部加宽完成。
候选将同一存储中的 `view_2` 描述符改为紧凑 `[T*32,512]` FIFO，按实际物理核组划分两个槽；
每槽为 `16×512` FP32，两个 AIV 各负责 8 行。AIV 按 K512 片段加宽，AIC 分两次 K256 MMA
消费，保留原 K4096 分段及段内累加顺序。`mix_x_rms_norm` 改为从原 BF16 输入读取后无损 cast，
避免继续依赖全尺寸加宽内容。RMS 仍须按原来 32 个 K512 片段的顺序累加，不能随分核改归约树。
第一版复用原 allocation，**只改变实际访问方式，不声称 GM 分配容量已经减少**。

同步改动必须和流水一起完成：原 AIC 等待“整个 widen 任务完成”的边需要被逐槽 READY/ACK
协议替代；若保留旧等待，而 AIV 又等待 AIC 的 ACK 才复用槽，将形成死锁。
READY 使用同组 FFTS 事件 8/9，ACK 使用 10/11；AIV 写回后发布 READY，AIC 将 GM 数据读入
本地 L1 后发布 ACK，之后 MMA 使用本地 L1。每个逻辑任务末尾须排空 ACK，覆盖多波次与反复调用。
不足 8 行的尾部按有效行处理；0 有效行的 AIV 必须显式生成零 Tile，不能依赖越界地址配
`valid_rows=0` 就一定安全。原平方和、reduce、mix 的数据依赖仍须逐项保留。

已有设计模型覆盖不能替代实际变换后代码的证明。初版 B16 源码变换已通过 Python AST 生成；
当前还在补零行分支和真实阶段 DAG 的 READY/ACK 模型，**尚未完成本候选的 PyPTO 后端编译或
真机验证，不记录精度 PASS，不更新性能表**。

测试入口
[device_compare.py](../../../tests/pypto_test/csa_multishape_spmd_20261016/device_compare.py)
新增默认关闭的 `--hc-compact-stream`，只允许与 `--range-policy` 配合。变换发生在正式生成后、
实际 Tensor 绑定前；运行目录保留变换前源码、变换前绑定合同、变换实现和审计结果。
原布局进度模型在报告中明确改名为 `baseline_cpu_protocol`，避免误称其已经证明新 FIFO 协议。
正式权重、原动态 PTO 对照、输出/Top-K/cache/state/保护区/图重放门槛及完整 L1 计时保持原口径。
当前只完成测试入口语法检查，没有为了此次文档迁移新提交 NPU 任务。

**S4 源码工厂：逻辑 lowering 通过，动态 query 支持尚未完成。**

候选与证据位于
[query_shape_source_v1](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/)，
实际 CPU 结果见
[cpu_evidence.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/s4_t384_v1/cpu_evidence.json)。
工厂从原 CSA 数学源码构造隔离模块命名空间，显式区分每请求 `query_tokens=4` 与独立
`token_capacity=384`；只在副本 config 中调整具名配置，其余相对 import 数学闭包保持原文。
这条路径从真实源码重新生成逻辑 IR，未对冻结 S6 IR 全局替换数字。

已完成 17 个模块的隔离导入，8 个 `DECODE_SEQ` 消费者全部读取 4；执行 specialize 及
`InlineFunctions` 到 `OutlineIncoreScopes` 的 10 个 pass，生成 68 个逻辑函数。
生成结果中可见 `query//4`、`request*4`、RoPE `[4,64]`、每个 AIV 2 行等对应表达。
整数范围模型覆盖 uniform S4 的 B1..96、4 种压缩相位，共 384 个相位案例；该范围仅指源码公式
和容量的 CPU 核对，**不扩大正式入口目前已声明的支持范围**。

尚未完成新 query 规格的 tile/backend lowering、融合单 SPMD、Native 参数绑定与 NPU 精度/性能
对比。S4 偶数分组通过也不能证明奇数 S、mixed-query、真实请求生命周期或连续 decode 通过。
本候选目前只保留探索证据，不替换正式 S6 实现。

**性能与下一步。** 本节没有新增设备性能。七档历史完整 L1 结果仍在原记录中，全部尚未满足
逐档降低 30% 的目标。接下来先完成 HC 候选的真实同步/编译审查，再决定是否值得做单规格设备
对照；同时继续检查更大的 Indexer、QK/PV 和输出投影关键链。任何模型估计、源码访存量减少或
局部阶段数字都不得替代完整 CSA 的同轮性能证据。后续结论继续直接追加本文件。

### 2026-10-17：本轮 CPU 候选收口补记

HC 候选已补零有效行分支，B16 定版源码与绑定合同保存在
[cpu_transform_v2/production_h131072_b16](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/cpu_transform_v2/production_h131072_b16/)。
[audit.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/cpu_transform_v2/production_h131072_b16/audit.json)
明确核对：旧 AIC 等 widen 完成的等待已从执行源码移除、非 HC 数学保持、mix 其余 AST 保持、
RMS 消费者改等 reduce、原 FFTS 事件与新 READY/ACK 编号不冲突、未增加永久 L1 预留。
该文件的 `cpu_lowering_passed/device_validated/performance_measured` 均为 false，当前允许推进
下一步编译，不等于编译或设备通过。

S4 生成 IR 的独立语义审计现已完成，见
[generated_ir_audit.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/s4_t384_v1/generated_ir_audit.json)。
审计确认 8 类函数中 12 处具名请求定位为 `//4`，整请求/pair query tile 分别为
`[256,128]`/`[128,128]`，双 AIV 每 lane root 为 `[2,1024]`，RoPE 为 `[4,64]`，
长/短通信区为 `[192,1024]`/`[192,768]`。此次 NZ 模式为 1，即 INT8 NZ、BF16 ND。
复现脚本为
[reproduce_cpu.sh](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/reproduce_cpu.sh)，
使用新输出子目录，避免覆盖已有证据。

开发审计器时发生 3 次误匹配，已保留在
[audit_development_failures.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/audit_development_failures.json)：
第一版把 coefficient pair 的正确 `//2` 误认为请求定位；第二版把 `qk_block_count` 前缀误认为
`qk_b`；第三版无依据地要求至少 15 处命中。最终改为精确赋值目标及 8 类函数覆盖检查，实际
记录 12 处请求定位。三次失败均来自审计脚本，原数学源码 import/specialize/outline 首次即通过，
没有为通过审计改数学表达。该结果仍不证明完整后端、融合、设备精度或性能通过。

文档整理后按稳定标记检查：117 份初始正文均存在，唯一晚于初始快照修改的生产包审计报告，其
最终全文已在前面的“生产依赖闭包审计最终补充”中另行保存；历史归档起止标记各一处。
新增本地证据链接可访问；未计算 hash，未为文档整理执行设备测试。

**HC 真实阶段模型的已完成部分。**
[cpu_transform_v2/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/cpu_transform_v2/result.json)
保存 T24/48/96/102/144/192 六组实际生成源码的 AST/绑定检查和模型结果：各 8 种随机核推进顺序、
各 3 轮代际，保留原 72 核外围阶段等待/发布，仅将 HC 边改成 READY/ACK；均完成，FIFO 每轮排空，
RMS 在 reduce 前就绪。每组另保留原 widen 等待作为负例，模型能检出死锁。此模型以 MTE3 发布、
MTE2 ACK 和组内双 AIV 通知的硬件语义成立为前提，未模拟其它阶段全部内部 FFTS，更不能证明
硬件缓存可见性。

首次整组脚本在第七组 T6 读取旧目录的 `layout.json` 时发生 `FileNotFoundError`，该组未执行
候选变换。前六组已打印的完成事实恢复进报告，未补造模型推进步数；报告明确
`run_completed=false`、`completed_cases_passed=true`。脚本随后改为逐组落盘。
T6 的补充生成/检查尚在进行，不将其提前记为通过，也不重复前六组验证。

**输出投影的下一候选分析：尚未实现。**
[critical_path_next_20261017/analysis.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/critical_path_next_20261017/analysis.json)
从既有 L1 和阶段插桩核对瓶颈，保留 Indexer/QK 等已关闭方向。B16 插桩中 O-A 连续三个成员的
关键链约 86.50 μs；这是诊断量，不是本轮完整性能采样。

候选为 T96 时每组 8 个 N128 输出块拆成“3 对相邻块 + 2 单块”：8 组共 24 pair + 16 singleton，
24 个 AIC 各分一对、其中 16 个再分一个单块，最大仍为 3 个输出块；直接把全部块配对会产生
32 pair，导致 8 核承担 4 块长尾。保留 M128 valid96、N128/K128 归约，pair 共享 A 读取；
有效逻辑 A 读取估算由 48 降至 30 MiB，不推导为设备耗时同比下降。

双 Acc 将恰好使用 128 KiB，Right 必须只占两槽；嵌套 pipeline 若生成四槽会溢出，需先核对
K/N 微步和生命周期。NZ 实际数学来自 `orchestration.py`，不能只修改 `kernels.py`；后续还需
更新 pair/single 的来源合同、quant 对两个结果的依赖及按 group 的行依赖证明。
历史已有 T192 双 M 资源探针，本项是既有共享输入方向的 T96 具体方案，不当作已经实现或验证
的新优化。当前只保存分析，未编译、未上卡、未更改已选生产实现。

### 2026-10-17：HC 紧凑双槽开始完整编译，保留首轮失败

继续原目标，上一轮完成过程迁移属于已落盘进展。本轮先编译固定 B16/H131072 候选，
并行推进 S4 的完整导出和 O-A pair 的真实资源探针；未修改七档目标或精度门槛。

首轮
[hc_compact_stream_cpu_v1/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_stream_cpu_v1/result.json)
在 PyPTO 前端失败：`hc_compact_fp32` 初始零 Tile 无 `TileView`，有效行分支的
`fillpad→set_validshape` 带 `pad=zero`，重赋时类型不同。失败源码和日志完整保留，未启动 NPU。
修正仅让零 Tile 分支也显式 `fillpad→set_validshape`，两端统一成相同属性；保留尾部清零与
`rows>0` 才读 BF16 的保护。旧 transform 保存在候选目录，v3 与 v2 的 AST 差分只含这处初始化，
绑定、布局、数学顺序及事件不变，因此不重复已有协议模型。
证据：
[cpu_transform_v3/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/cpu_transform_v3/result.json)。
第二轮完整编译输出为 `tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_stream_cpu_v2`，
使用独立 PyPTO/Simpler、PYPTO planner、kernel mode；此处记录提交事实，尚不预记通过。

此前 T6 缺历史布局的问题已通过正式包生成该形状源码解决；独立
[cpu_tail_t6_v1/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/cpu_tail_t6_v1/result.json)
完成零行 lane、8 种推进顺序×3 代和保留旧等待的死锁负例。只补缺失规格，没有重复前六组。
`temporary_4` 在 T6 合同中为 `[16,1]`，尾部 8 行写入处于原 padding 容量内。

收益假设进一步限定：紧凑 FIFO 减少的是活跃地址范围并允许 widen/linear 重叠，AIV FP32 写和
AIC FP32 读的累计字节量不会因此消失。RMS 部分值的 GM 存取是新增开销；mix 从 BF16 读取才
实际减少该处输入字节。最终是否采用仍以完整 CSA 同轮 L1 与原有精度/同步门槛为准。

A3 FFTS 只读源码对照见
[ffts_source_audit.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_direct_stream_v1/ffts_source_audit.json)：
现有 TPush 的 Vector→GM 以 MTE3 发 READY，Consumer 在 GM→L1 取数后以 MTE2 发 ACK。
候选使用相同发布管线与 mode2；仍需核对本次生成产物和真机行为，源码类比不替代验证。

### 2026-10-17：HC 候选完整 CPU 通过，设备入口隔离修正

[hc_compact_stream_cpu_v2/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_stream_cpu_v2/result.json)
的完整 kernel artifact 生成通过。实际产物检查见
[compiled_protocol_audit.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_stream_cpu_v2/compiled_protocol_audit.json)：
两份 GM store 在 MTE3 READY 前，FIFO TLOAD 在 MTE2 ACK 前，事件为常数 8/9/10/11，
每个逻辑任务尾部排空两槽 ACK。最大 Mat 475136 B、Vec 163872 B，Left/Right 各 65536 B、
Acc 131072 B，均在相应预算内；局部 L1/UB 复用具有编译器事件保护。
完整编译的复现入口为同目录 `reproduce.sh`，使用已保存的固定 `source.py`。

首次设备任务 `task_20261017_152016_31798452638` 已 exit=1，失败不是算子数值或设备同步：
在导入生成入口时找不到正式 `deepseek_v4_flash_csa_spmd.signals`。测试驱动为冻结动态基线，
把 `vllm_ascend.ops.pypto.__path__` 设为单一历史快照，该目录没有刚迁入的正式 SPMD 包。
失败输出保存在
[hc_compact_device_b16_v1/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_device_b16_v1/result.json)，
尚未执行候选，也没有采集性能。

修正仅作用于测试驱动：在切换基线搜索目录前，显式加载当前正式信号模块并检查真实路径；
随后动态基线仍只搜索原冻结目录，未添加可回退到当前数学的搜索路径。CPU 检查确认信号来自
当前正式包、基线 `variant.py` 仍来自冻结目录，见
[import_context_check.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_stream_cpu_v2/import_context_check.json)。
未修改数学、生成源码或依赖仓，因此不重跑完整 CPU 编译。

修正后提交 `task_20261017_152415_32613777240`，输出目录 `hc_compact_device_b16_v2`；
仍为 B16/H131072/S6、正式权重、原动态 PTO 对照，保持全部精度/状态/图重放门槛和三窗 L1。
此段仅记录提交与入口修正，最终设备结果待后续追加。

### 2026-10-17：对照独立 HC_pre 的具体借鉴及尚未吸收部分

用户询问对 [hc_pre.py](hc_pre.py) 的参考。重新逐段核对当前文件后，明确以下对应关系：

1. 按工作量分档，且投影与尾段可重新分核。投影将 token×K 分片铺到核组；尾段按实际 token
   重新分给 Sinkhorn 和混合，小档再切 D。CSA 范围策略借鉴这一原则，不能照搬 HC 的 T 阈值。
2. 双槽供数与 AIC 消费重叠。当前紧凑 FIFO 候选实际采用这一点，加宽的数据同时用于 RMS；
   A3 路径仍经过 GM，不能称为 UB→L1 直连，也不能把理论重叠当成实测收益。
3. 减少重复读取与转换。HC 复用已加宽输入；输出投影的相邻 N 输出共享 A 是该原则在 CSA
   另一阶段的候选应用。不同数学与布局仍需各自验证。
4. 区分组内交接槽、L1 流水和 L0 双缓冲。独立 HC 的 T96 路径显式交错 K256 加载与 MMA，
   控制相邻块生存期。O-A pair 探针实际遇到 Right 双重流水膨胀，需要同样按生成代码核对。
5. 小计算避免多余 Vector 搬排。设备标量 Sinkhorn、标量门值乘连续 D 区间值得参考；现 CSA
   尚未验证并采纳这些路径。Scalar 与 Vector 舍入、归约树和加法顺序不能忽略。

目前主要落实的是前段供数流水，**并没有完整吸收独立 HC 的尾段 48 AIV 重新分工**。
下一步先据实际成员分布、每核行数及输出写回粒度判断是否存在空核，再设计有数值顺序依据的
变换，不能只按“48 核”这一数字重写。独立 HC 的 K 段归约需要全体会合；完整 CSA 仍保留
按真实依赖通知/等待，不把每个阶段都改成全卡屏障。

### 2026-10-17：HC 双槽 B16 真机完整通过，性能未形成明确新收益

`task_20261017_152415_32613777240` 已 exit=0。实际生成候选与先前完整编译的固定源码 AST
一致，见
[cpu_source_comparison.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_device_b16_v2/cpu_source_comparison.json)。
eager、两侧 graph replay 和计时后 final 的 `x_out`、`idx_topk` 及六份 cache/state 全部
逐 bit 相同，所有比较 mismatches=0；30 代进度、Native 保护区和工作区保护区通过。
结果及判定见
[result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_device_b16_v2/result.json)、
[review_summary.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_device_b16_v2/review_summary.json)。

| B16 / H131072 / S6 | L1 窗口 1（μs） | 窗口 2 | 窗口 3 | 中位数 |
| --- | ---: | ---: | ---: | ---: |
| 原动态分 task PTO | 917.50 | 914.06 | 896.38 | 914.06 |
| 单 SPMD + HC 紧凑双槽 | 804.58 | 807.36 | 805.02 | 805.02 |

同轮完整 L1 降低 11.93%，30% 门槛为 639.842 μs，仍差 165.178 μs，未达标。
当前结果接近历史已选单 SPMD 约 807 μs；本轮没有现选单 SPMD 的同进程配对，因此不能将
跨轮约 2 μs 差额归因于双槽，也不能认定其具有稳定净收益。决定保留源码与完整证据，
**不替换现选、不扩测七档**。这次只验证固定 fixture 重放，不是连续 decode，也没有为新候选
补动态 metadata 边界或其它 query 规格；不得沿用原现选的边界结果替新候选背书。

用户进一步强调应借鉴 HC_pre 从多 task 到单 SPMD 的整个过程。本轮前段双槽只是其中一项；
后续重点审查原任务边界、中间结果的就地消费、融合后重新分核和尾段角色划分，不能满足于
把旧 task 的分块及全部等待原样装入一个 kernel。已安排对照历史成功/失败记录与当前成员分布，
结论继续在本主账追加。

### 2026-10-17：O-A 共享 A 的资源探针收口，尚未接入完整 CSA

完整材料位于
[oa_pair_source_v1/audit.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/oa_pair_source_v1/audit.json)。
v6 `pair_sequential_k_v6.py` 完整 CPU kernel artifact 通过：真实八组 NZ 根布局，
独立两条 N128/K128 累加链，每个 K 的 Left 供两个输出使用；Mat 393216 B、Left/Right 各
65536 B、Acc 131072 B。有效行作为参数覆盖 1..128，T96 只是当前资源探针点。
核对了 NZ `[8,64,256,16,16]` 及 stride `[4194304,65536,256,16,1]`，未改权重格式。

但编译器 PH-MR-001 明确提示 Right 请求的 depth=2 被降为 1，因此只证明资源可放下，
不能证明预期流水或性能有效。v3/v7 恢复内层 stage2 均实际需要 131072 B Right，超过
65536 B；v5 合并 K/N 微步在 Acc 的 if-phi coalesce 处失败。另有入口适配和加载错误 Simpler
`_task_interface` 的失败，后者只在候选 runner 前置独立 venv 后解决。8 次尝试、源码、
准确命令与日志均在目录中保留，没有隐去失败。

审计给出 24 pair + 16 singleton 的输出覆盖、quant 对 pair 两输出的依赖及 group 行依赖合同，
但 singleton/quant 与完整 CSA 尚未接入。该有界资源探针到此收口，不追加盲目参数扫描，
不运行设备，不改生产默认，也不把“共享输入字节减少”写成性能收益。

### 2026-10-17：明确 807.16 与 805.02 两版代码的实际差别

应用户追问，逐项核对历史 `b16_cached_row_deps_v1` 与本轮 `hc_compact_device_b16_v2`。
两轮都是卡 0、B16/H131072/S6、完整 CSA 单 SPMD，权重加载记录相同；均启用 earliest_ready
均衡调度、Q-proj 20 worker、O-A N128、QK 索引缓存、细化行依赖和 cached/100tick 等待。
正式包迁移前后的差异不是这两个性能数字的解释：只规范化信号 import 路径后，历史完整源码
与本轮应用 HC 变换之前的完整源码 AST 相同，见
[selected_before_hc_comparison.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_compact_device_b16_v2/selected_before_hc_comparison.json)。

807.16 版 HC 仍先由 12 个 AIV 完成全尺寸 FP32 加宽与本地 RMS 归约，再由 24 个 AIC
执行投影；投影 K256×16 为 stage2 pipeline。805.02 版仅替换 HC 前段及相关输入读取：
48 个 AIV 按匹配 AIC 的行/K 分片供数，GM 双槽 READY/ACK 替代等完整 widen；
RMS 的 32 个部分值先写 GM 后保持原顺序归约；mix 从原 BF16 读取后加宽。
新投影为 K512×8、每块两次 K256 MMA，当前初版没有独立 HC_pre 的显式交错 L0 双缓冲。
这些变化可能互相抵消，尚无本候选的阶段插桩能证明具体原因。
HC 尾段 comb/split/mix 的重新分工未实施，Indexer/QK/PV/O-proj 等其它数学未更换；
S4 探索和 O-A pair 资源探针均未包含在本次 805.02 的代码内。

历史三窗 807.16/804.40/811.26，中位 807.16；本轮 804.58/807.36/805.02，中位 805.02。
绝对差 2.14 μs，约 0.27%，采样区间重叠；两轮各自动态对照为 905.08 与 914.06 μs，
分别降低 10.82% 与 11.93%。这是不同轮测量，不能把后者整体收益当作本次 HC 改动收益。
旧版继续为现选，新版保留为通过功能/精度的候选，未替换默认。

### 2026-10-17：uniform S4 完整融合 CPU 编译完成

证据总入口：
[query_shape_source_v1/full_fusion_summary.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/full_fusion_summary.json)。
从真实 S4 数学源码完成动态 PTO 全编译（PYPTO planner），抽取并独立 parse 73 个物理核和
67 个逻辑计算体；B16/S4/H131072 使用新合同生成 66 compute + 4 dummy 的 DAG 及唯一
24 组完整 SPMD，130 root/132 leaf Tensor，工作区 288292864 B。292 个视图边界和跨代际
CPU 模型通过，未复用 S6 冻结数学或 S6/T96 专用行依赖。

首次融合重入编译时，ND O-proj-a 的 late IR 在 `ConvertTensorToTileOps` 被误改操作数：
BF16 load 换成 FP32 accumulator，之后 `InferTileMemorySpace` 报 lhs/rhs 类型不匹配；
逐 pass 证据见候选目录 `nd_projection_operand_trace.json`。不能把此失败归因到后面的
`AutoTileMatmulL0`。实验生成器对 8 次 ND O-proj-a 调用改用同次导出的早期逻辑原体，
保持数学和 schedule/layout，修复源选择；没有修改共享编译器。

其后 PTOAS 内存规划报告 Mat 544 KiB 超过 512 KiB。相同完整源码改用既有 PYPTO planner 后，
全部 PyPTO passes 与 PTOAS 后端 kernel artifact 生成通过，见
[b16_s4_h131072_fused_lowering_v3/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/b16_s4_h131072_fused_lowering_v3/result.json)。
失败产物、编译命令、布局/代际模型及三个 ScalarWriteLineShared 缓冲的所有权核对均保留。

范围仍有限：NZ=1；描述符保留旧 B16 的 cache/metadata 容量，仅令 T96→T64，不是真实服务
capture。尚未验证 runtime ABI、NPU 同步、输出/状态精度或性能；不代表 odd/mixed query
已支持。正式 S6 入口未解锁新 query 规格，下一步须将这条真实源码生成路径接入可维护入口，
并用真实 S4 输入完成设备验证；不能只据 CPU 编译通过改变支持声明。

### 2026-10-17：目标七档调整，静态生成链撤回历史目录，转向源级动态实现

用户连续明确三点：8K 移除 B24、128K 增加 B28；停止围绕 B16/S6 特化生成文件继续优化；
静态 shape 版本只能作为历史参考，正式 `ops/pypto` 只保留目标动态源码。目标引用的附件、
`GOAL_AND_CONSTRAINTS.md` 与本页当前目标均已同步；机器可读验收点记录于
[current_acceptance_cases.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/current_acceptance_cases.json)。
七档仍逐档要求同轮完整 L1 耗时降低至少 30%，不以历史静态结果充当动态验收。

核实发现：旧 `codegen.py` 将 Tensor shape 输出成具体整数；`specialize` 再用实际 extent
替换动态变量，并在 Python 侧逐阶段内联核体。核内实际长度、位置与页表读取仍存在，
但不能据此称其为目标动态 shape 实现。旧 `row_dependencies` 还明确限制 B16/T96。
此前把范围入口和多形状生成等同于目标动态实现的表述需要纠正，不能仅改名称。

已将当时正式包的 20 个模块完整保留在
[static_shape_archive_20261017](../../../tests/pypto_test/csa_multishape_spmd_20261016/static_shape_archive_20261017/)，
复制时逐文件内容一致，未生成 hash。清单、路径及字节数见该目录 `archive_manifest.json`。
正式目录移除冻结数学 IR、形状实值化编译器、逐阶段展开器与专用变换；不再导出 `build_csa`，
不设静态默认回退。通用 notify/wait 与等待码适配仍可复用。原多 task CSA 实现未改。
历史脚本改用明确的 `historical_sources` 加载器，新结果字段注明
`historical_shape_specialized_spmd`、`target_dynamic_shape=false`；历史源文件和已采结果不改写。
仅完成 CPU 语法与导入隔离检查，73 份历史核体合同仍可加载，正式加载器拒绝静态编译入口；
没有重跑 NPU 或性能测试。证据为归档目录 `migration_check.json`。

HC 紧凑双槽 805.02 μs 与现选 807.16 μs 的差额未证明新收益；停止该冻结候选的扩测。
尾段候选尚未写入静态变换，已停止这条路径。保留的源级设计材料包括
[HC 尾段历史分布审计](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_tail_distribution_audit_v1.json)、
[HC_pre 历史优化依据](../../../tests/pypto_test/csa_multishape_spmd_20261016/hc_pre_transition_lessons_v1.json) 和
[动态单 SPMD 源级设计](../../../tests/pypto_test/csa_multishape_spmd_20261016/csa_dynamic_source_incore_design_v1.json)。
后者是设计、未编译执行：入口保留动态 B/T，核内按实际工作量循环，各阶段用输出所有权公式
计算前驱通知/等待，不继续重放冻结 records/waits 表；Indexer 分支及组内握手仍须保持一致。

下一步是维护高层动态数学源码及其参数化入口，再实现完整的源级 MIX SPMD。
源码准备工具只能证明动态符号/同源数学保留，不能冒充已经完成融合。完整动态 CSA 当前尚未
完成执行、精度或性能验收，旧静态的 B16/S4 CPU 编译亦不能替代这些证据。

用户另要求新动态代码完整中文注释：每个文件开头必须提供整体思路总览和阅读顺序，覆盖
输入输出、布局、动态维度、分档、各阶段数据流、AIC/AIV 分工、同步和工作区。关键循环、流水、
分核、通知/等待、缓存发布与尾块处理解释原因及正确性约束；按职责拆函数，未验证范围明示。
此要求已写入当前目标，后续源码按此维护。

### 2026-10-17：动态请求边界与 mixed-query 入参合同审计

只读源码审计保存于
[dynamic_input_contract_audit_20261017.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_input_contract_audit_20261017.json)。
原 CSA 根入口的 `cmp_query_start_loc` / `idx_query_start_loc` 实际来自 Native 原始 TND query
前缀和，不是压缩输出前缀和；因此动态请求归属可以利用现有 device 边界，无需先增加展开映射
或 Host 回读。需要把该边界继续传给消费者，用原始边界而非 `query//S` 决定请求，B 与 T
不再因 uniform S 假设绑定。Indexer 的请求内分组还须处理尾组，压缩计数与输出偏移按实际
请求长度计算。Indexer compressor 的 projected pool 目前连 `seq_lens` 也未传入，dummy
应先屏蔽再访问位置和状态，不能仅替换一个除法就宣称 mixed-query 已支持。
这些是源码合同与后续改动清单，尚未修改数学或完成 mixed-query 验证。

### 2026-10-17：高层动态源码准备接口与中文总览落地

新增正式模块 [query_source.py](deepseek_v4_flash_csa_spmd/query_source.py)，文件顶部按用户要求
给出输入输出、动态变量/容量、准备流程、AIC/AIV 与工作区职责、完整 CSA 关系及当前边界。
提供 `QuerySourceConfig` 与独立命名空间的源码准备接口；仅调整原数学四项具名配置，保持
原高层函数正文、B_DYN/T_DYN 与真实 metadata 读取。没有旧静态生成器依赖，没有接入历史
冻结核体，不修改共享安装、原数学 config 或 sys.path。

CPU 证据为
[production_source_cpu_v2/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/query_shape_source_v1/production_source_cpu_v2/result.json)。
同进程 S4/容量384、S6/容量384、S4/容量128 的 17 模块隔离导入、8 个 query 配置消费者、
16 个非 config 文件不变检查及高层符号解析通过；输入输出保留 T_DYN，请求长度保留 B_DYN。
S4/容量384 额外执行至第 10 个 pass `OutlineIncoreScopes`，动态两轴保留。非法配置拒绝、
仅导入准备工具不加载 torch/PyPTO 的检查通过。v1 曾因审计脚本对 self 空 annotation 调用
ast.unparse 失败，修复审计后 v2 通过，失败日志保留；不是数学编译失败。

这仍是原多 task 数学的动态源码准备，**尚未融合成完整动态单 SPMD**。统一偶数 S 的结构
候选不等于 mixed-query 或其它 S 设备支持，容量配置也不是本轮实际 B/T。未做新的后端编译、
NPU 精度或性能测试；不会把这个工具的通过当作动态 CSA 完成或 30% 性能目标达成。

### 2026-10-17：源级动态请求几何与单写者阶段通知

新增 [request_geometry.py](deepseek_v4_flash_csa_spmd/request_geometry.py)，直接消费 Native
的 device query 前缀和与 KV 长度。B/Q/T 保持独立动态维度，不用 `query//S` 反推请求；
请求驱动消费者可直接取得边界，只有缺少 request 时才进行 O(B) 反查。C4 压缩计数按真实
关闭组数计算，允许活动请求本轮零关闭组，也不固定每请求最多两行。紧凑偏移可能为负，
只有加上合法关闭位置后才构成输出下标；dummy 与非法范围先屏蔽再访问业务数据。

证据：[dynamic_request_geometry_20261017/summary.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_request_geometry_20261017/summary.json)。
同一高层函数的 CPU AST 逻辑与独立区间 oracle 比较，51 个场景、1487 项通过，涵盖
S1/2/5/6/7/9、mixed 2-2-2/4-2/3-3、dummy、奇数和不合法范围。初次探针未读取 block_idx，
SPMD 被解析成三个 dispatch；只修探针的 worker 使用后，v2 完整 CPU 后端编译通过。
最终 ABI 的 Tensor 维度仍为 -1，C++ 从运行时描述符取维度；未绑定样本 shape。
这些结果不证明整个 CSA 已支持 mixed-query，其他消费者仍须改造与设备验证。

新增 [phase_sync.py](deepseek_v4_flash_csa_spmd/phase_sync.py)：每核、每阶段独占 512 字节
信号槽，阶段 0 保存调用代际，业务阶段使用单写者完成通知。按消费行 tile 的实际所有权
计算前驱，不导入历史 waits/records，不生成固定七档的等待表。业务写回完成与信号发布、
信号达到与业务缓存可读分别处理；不是每阶段全卡 syncall。工作区首次清零，所有参与核
每轮各 begin 一次，含空核；不能在同一 workspace 改 G、核映射或阶段含义，也不能失败后
继续使用部分推进的代际。完成通知没有 ACK，不能用同代际反复发布逐 tile 进度或提前复用
业务缓冲。运行时布局身份绑定仍须在完整入口实现，当前通过明确合同约束调用者。

证据：[dynamic_phase_sync_20261017/cpu_v1/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_phase_sync_20261017/cpu_v1/result.json)
及同目录上级的 `review.json`、`protocol_model.json`。720 组所有权集合比较通过，动态
matmul→Vector 小探针编译为一次 MIX 提交；实际 C++ 含 4 处 TNOTIFY、2 处 TWAIT。
固定布局的 CPU 顺序事件模型执行 1792 次、162304 个事件，无提前读或死锁。负例证实旧
workspace 按 G=6→3→6 复用会失效：256 种顺序中 219 次提前读、37 次死锁；另保留漏空核
推进、同代际重复 tile 的失败反例。这只是协议模型，不模拟 A3 DCache 或硬件流水。
当前生成 TWAIT 使用 DCCI+volatile，没有启用历史 100 tick 退避变换，更不是 ld_dev；
不能沿用历史等待性能。全套新动态同步尚未上卡验证。

### 2026-10-17：HC、Q_A/QR 源级动态阶段及一次 MIX 前段编译

[hc_stages.py](deepseek_v4_flash_csa_spmd/hc_stages.py) 按六段拆分：加宽/RMS、四份投影、
左折叠归约、pre/post 门值、comb Sinkhorn、混合归一化。函数接收显式 worker/worker 数，
按实际 T 计算核内波次，不嵌套 spmd/task。原 FP32 输入投影、K256 累加顺序、除法 Sinkhorn、
BF16 舍入后 RMS 和 Native BF16 norm 权重保持。二维 GM 视图由外层描述符绑定；不能把原
task 外的整块 reshape 机械搬到 incore 内，后者会被转成 Tile。

证据目录：[dynamic_hc_stages_20261017](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_hc_stages_20261017/)。
`source_check.json` 记录六段数学 AST 一致及 6912 组所有权比较；六段前端 lowering 通过，
widen/project 的完整 CPU 后端编译通过。PAD16/PAD8 是工作区容量，真实输入 T 之外不读；
四份尾块 scratch 各只有全局最后短块可写，改分工时必须重新核对所有权。

[qkv_stages.py](deepseek_v4_flash_csa_spmd/qkv_stages.py) 实现 Q_A、QR。Q_A 按实际 T<128
选 M32，否则 M64，尾块保留 M16，完整 K 顺序不变；QR 保留八行、256 列平方和/amax、
原 rsqrt/div/recip 与 INT32→FP16→INT8 舍入链。helper 内不藏 cache policy 或创建任务。
没有把 Native BF16 gamma 预先改成 FP32。QR 本段只需 pad8，但对接 Q_B 后 INT8 暂存须
按后者 pad128 要求分配，不能照局部最小容量分配整链。

证据目录：[dynamic_qkv_stages_20261017](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_qkv_stages_20261017/)。
QA dense/tail K 流水与 QR 数学 AST 比较通过，14 项常量一致；T0..384 及 8/16/24 AIC
分工覆盖恰好一次。QA 在 ND、NZ2 均通过完整 CPU 编译，QR 在 ND 探针通过。
失败尝试保留：inline 内 cache policy 不是合法 scope 属性；整张 GM reshape 被转成
Tile；M64/M16 使用同名 accumulator 导致类型冲突。分别移到调用合同、直接 GM+小块 UB
reshape、拆开 dense/tail 局部变量解决。独立 layout 声明避免仅为读取常量导入原模块时
触发不相关的三维权重设备探针，没有修改依赖或转换权重。

随后在测试目录构建 HC→Q_A→QR 前段的高层动态 MIX，正式数学仍来自上述 helper。
证据：[dynamic_front_pipeline_20261017/full_v3/compiled_audit.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_front_pipeline_20261017/full_v3/compiled_audit.json)。
固定 24 AIC+48 AIV，所有核每次 begin；初版保守等待本阶段全部实际活跃生产者，所有权
按实际 T 计算，不等无任务核，也不宣称是最终高性能分工。390 个 T（含 T168 及多波次）
的 phase DAG/所有权检查通过。完整 CPU 后端只生成一次 MIX 提交，AIC/AIV 从真实描述符
提取 T/PAD16/PAD8/PARTIAL/SIGNAL 五个动态维度，没有替换为样本字面量。

前段编译失败及处理均保留：v1 将跨 member 只读输入误标 InOut，改为真实只读合同；
v2 中 QA 动态分支返回的 SSA 无法被 cluster 识别为 whole-parameter writeback alias，
测试壳显式保留原 GM 参数别名返回，数学仍写同一缓冲。v3 后端已成功，末尾的 ABI 正则
漏右括号误报；仅复核已有产物，未为修报告重编译。复核确认 QA 四个 dense/tail TSTORE
没有被 DCE，均位于 NORM TWAIT 后、QA TNOTIFY 前；Vector 从同一共享参数读取并先等 QA。
原失败结果不覆盖，独立审计记录纠正。当前没有权重 BYPASS 属性、没有设备数值/同步结果、
没有新性能数据；**这个前段不是完整 CSA**，也不能用于七档 30% 目标验收。

### 2026-10-17：动态源码中文总览与调度注释完善

按用户要求，新增动态源码均在许可证后首先给出中文整体说明：输入输出及布局、动态维度、
容量边界、分块与分核、阶段数据流、同步与缓存职责、工作区用途、阅读顺序及验证范围。
目录 `__init__.py` 增加源码阅读地图；数学按 HC 六段、QA/QR、query 投影、几何和阶段同步
分文件。注释说明 wave 条带归属、pipeline 不改变累加依赖、尾行有效 store、四份 K owner、
Sinkhorn 循环外首轮加循环内 19 轮，以及先 BF16 舍入再计算 mixed RMS 的数值约束。

同步注释明确单写者不需要 atomic 不等于无需缓存发布；完成通知没有 ACK；可选后端替换
仅适用于单元素 INT32、GE TWAIT，默认动态探针没有启用。源码准备工具的“无设备操作”
表述收窄：物化只做 AST/文件操作，但导入原 nz_mode 在 NZ2 且 NPU 已初始化时可能执行
权重格式探针。没有为了补注释重复已完成的设备或性能验证。

### 2026-10-17：Q_B、query 反量化/RoPE 与准备阶段的动态源码

新增 [query_projection_stages.py](deepseek_v4_flash_csa_spmd/query_projection_stages.py)，提供
`q_b_stage`、`q_rope_prepare_stage`、`q_dequant_rope_stage`。数学来自原高层
`q_projection.py/qkv_proj_rope.py`，没有冻结 IR。T、QR 暂存行、投影工作区均保留动态；
由外层传入核编号与 worker 数，不创建独立 task。Q_B 沿用 ND M64/N512/K128 与 NZ
M128/N256/K256 路径，按实际 token 数计算 pad128 行。query 输出通过外层的二维 GM
别名连接 `[T,64,512]`，核内不重排整张 Tensor。

源码明确相关同步：Q_B 等 QR 的 INT8；反量化同时等对应 Q_B 列块、QR scale 与 RoPE
准备完成。RoPE 准备只依赖本轮 sin，可与 Q_B 并行。反量化保持先 activation_scale 再
weight_scale、512 列归约以及原 NOPE/RoPE 运算；完整八行 rsqrt、尾行 recip(sqrt) 的
差别不抹平。尾分支固定四 head 的依据是 16/64 head 策略要求 T 为 96/384 的倍数，均无
八行尾块；以后修改分组阈值必须一起重审该条件。

证据：[dynamic_query_projection_20261017/nz_v3/result.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_query_projection_20261017/nz_v3/result.json)，
同目录上级 `stage_contract.json` 和 `source_equivalence.json`。四个源级函数的抽取后
控制/数学 AST、完整/尾部 head 数学循环一致；T0..384 每行 64 head 覆盖恰好一次。
三段 NZ2 完整 CPU 后端编译全部通过，没有导入原三维权重探针模块。ND 本轮只做源码
审计，未再编译。v1 复现 shell 在 CANN set_env 的未定义 ZSH_VERSION 上退出，尚未开始
源码编译；修复环境加载期间 nounset 使用后，v2 的文档缩进触发源码审计误报，仅排除
docstring 后 v3 通过。失败日志均保留，没有修改依赖库。

九个正式文件的中文注释复核记录为
[dynamic_source_readability_audit_20261017.json](../../../tests/pypto_test/csa_multishape_spmd_20261016/dynamic_source_readability_audit_20261017.json)。
本轮发现的七项具体缺口全部修正；九文件纯语法编译通过，未导入模块或启动设备。
前段编译快照与当前数学（排除注释/docstring）AST 相同，无需因注释变化重编译。
当前仍没有完整动态 CSA 设备运行、数值结果或性能数据；所有七档验收状态保持待验证。


### 2026-10-17：动态 KV、原始 cache、紧凑 metadata 与前段组合

新增正式 [kv_stages.py](deepseek_v4_flash_csa_spmd/kv_stages.py)：KV 投影保留原 ND BF16
权重、M32/M64 区间、M16 尾块、N128/K512 顺序；512 行段内按实际 T 计算分工。
归一化/RoPE 保留八个 64 列平方和的原加法顺序、sqrt/div 与 BF16 舍入。输入 gamma
仍为 Native BF16，不在初始化升级为 FP32。projection pad16 是容量，不能推断有效 T。
证据目录 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_kv_stages_20261017`：
两段各 1600 个 T/worker 几何模型、数学 AST 比较和完整 CPU 后端编译通过，未上卡。

新增 [cache_stages.py](deepseek_v4_flash_csa_spmd/cache_stages.py)，直接写 Native 原始 KV
二维连续别名；按真实 T 的八行块分核，先检查 page/offset 再读取 KV，不建立搬运窗口。
新增 [metadata_stages.py](deepseek_v4_flash_csa_spmd/metadata_stages.py)：worker 0 依实际
Q/L 生成每请求紧凑偏移，消费者在单请求内直接 gather Native 紧凑 RoPE 行，只返回 UB
Tile。请求长度可以混合，关闭组数由真实位置计算；C=0 或无关闭组不能读取负下标。

证据目录 `dynamic_metadata_cache_20261017`：同源 AST CPU oracle 的 116 项 metadata、
70 个 cache 场景通过；CPU v1 的 offsets 编译通过，rope 失败于返回值被识别为 Tensor。
将局部单位旋转的 `pl.full` 改为明确的 `pl.tile.full` 后，v2 的 rope/cache 完整编译通过，
不重复 offsets。产物审计最初只查 AIV 的维度提取注释，遗漏 offsets 的 T 由 AICPU 从
Tensor descriptor 读取并以 Scalar 传入；复查真实产物后通过，未为修报告重新编译。
这仍是动态 shape 元数据，不是回读 Tensor 内容或形状字面量替换。

独立审查 `review.json` 未发现合法合同内实质错误，并补充必须由完整入口兑现的约束：
Native startPos=L−Qlen，actualNumReqs 排除的尾请求在 PTO 的 L 中为零；单位旋转只提供
越界保护，不能代替活动/causal/slot 判定；有效写 slot 唯一；cache/offsets 的发布与消费
仍需外层同步。116 项不是所有防御分支的覆盖证明；真实服务 dummy/prefix 生命周期待验。

Q_A/QR 的 INT8 matmul 暂存与 scale 暂存拆成两个动态容量轴：前者接 Q_B 时需要 pad128，
后者只需 pad8。`dynamic_qkv_stages_20261017/distinct_capacity_v1` 的 QR CPU 编译通过；
数学不变，未重复 Q_A 检查。

`dynamic_query_front_pipeline_20261017/full_v2` 将 HC→QA→QR→Q_B/RoPE/query 接成一个
24 组 MIX；其 v1 失败是测试壳将 cache-policy 放在不接受该属性的显式 cluster SPMD 上，
移除测试壳属性后通过，不修改依赖。`dynamic_qkv_front_pipeline_20261017/full_v1` 再加入
KV 与原始 cache，一次完整 CPU 后端编译通过。QA/QB 使用 NZ2，KV 权重保持 ND；九个
动态维度与 41 个 Tensor 入参保留，390 个 T 的所有权/阶段 DAG 模型通过。AIC 在 QA 后
先做 KV 投影再等 QR，AIV 在 QR 后消费 KV，避免双方相互等待尚未执行的生产任务。
以上更大组合没有设备精度或性能证据，不能作为完整 CSA 结果。

### 2026-10-17：动态 HC→QA→QR 前段首次设备数值与代际验证

设备任务 `task_20261017_162439_53401511955` 经 `task-submit --device auto` 在卡 8 完成，
exit 0。证据：`dynamic_front_device_20261017/device_v1/result.json`。T168、T169 两种实际
动态输入分别执行固定地址的 A→B→A，六次候选调用的有效输出对原动态分 task PTO 数学
逐 bit 一致，包含 HC 中间量、QA、QR 和量化 scale；512 字节前后保护区、声明未写区、
输入不变性、72 个参与核的阶段信号与重复 A 均通过。没有运行 Native forward。

使用正式 checkpoint 的 layer 4 六组权重与合成 BF16 hidden，ND 布局。初次 CPU prepare
误以为 checkpoint norm 张量本身是 BF16；实际存档是 FP32，Native make_layer 创建
BF16 参数后 copy。测试按同一加载语义转为 BF16；该层两份 norm 转回 FP32 完全相同。
这不是把 Native BF16 权重预先升为 FP32。失败 `cpu_prepare_v1` 保留，v2 的权重/schema/
原分 task 参考完整编译通过。参考保留当前 canonical 数学；两处临时分配仅外置以供比较，
逆变换 AST 一致性已经检查，不使用历史冻结生成体作参考。

实际运行时尚无执行前高层 `source_snapshot`，因此没有事后补造。任务结束后只归档真实
编译产物，`device_v1/evidence_manifest.json` 明确时间与来源，原 `result.json` 未修改。
`compiled_evidence` 保存 39 份编译文本约 3.4 MB；13 份二进制仅列本地路径与大小，不做
hash、不入源码。产物确认候选只有一次 MIX 提交，24 组，五个动态维度从描述符取得；
使用默认 TWAIT，Cube 两处、Vector 六处，没有历史 100 tick backoff 或 ld_dev 变换。

这是前段功能、数值、信号代际的有界证据，不含完整 CSA、Top-K/state、aclgraph、历史
长度或七档性能。没有采样性能，也不沿用历史静态约 805/807 μs 作为新动态实现结果。
下一步继续接 Indexer 与请求驱动压缩阶段；完整功能通过后再做完整精度与同轮 L1 对比。

### 2026-10-17：压缩迁移合同与当前 canonical 数学优先

`dynamic_compressor_source_plan_20261017.json` 将主/Indexer 压缩拆成八项明确迁移：
真实 T 动态投影、Q/L 请求池化、直接 Native state commit、紧凑 norm/RoPE、Hadamard、
量化/key 写入及 scale 单写者 RMW。原 Indexer pool 未传 Q/L，固定每请求 ceil(S/4) 行
的布局必须整体改造；不能只换一个 token//S 除法。主 M64 投影首块必须保持 pl.matmul，
不能机械替换成 matmul_acc(init)，两者有效 accumulator 行距不同。

原 cache 写的 safe_row=min(row,C−1) 在 C=0/负行时先读后判，迁移时必须改成先检查
compact 行，再读取 slot，最后检查页与页内偏移。Indexer 每 32 个 FP16 scale 共享 64B
RMW 区域，slot 唯一不足以允许跨核并写；首版保留单写者。state 在所有 pool 历史读取
完成后才 commit，prefix 共享的跨请求读写依赖不能无证据删除。

迁移 Indexer query 时最初依据旧精度记录提出“先合并两尺度”，随后核对当前正式
`decode_indexer.py`，现行数学是 `(acc * activation_scale) * weight_scale`。已立即以
当前 canonical 为准保留两次乘法，完整块/尾行一致；历史修正记录不能覆盖当前源码。
此次只迁移调度与动态容量，不夹带反量化数学变化。实现与 CPU 证据另按完成情况追加。


### 2026-10-17：中文可读性第二轮审阅与动态 Indexer query 六阶段

按用户要求，本轮继续将总览写在源码文件开头，并以阶段函数组织代码。KV、原始 cache、
metadata 和包级阅读地图完成独立中文审阅，补充实际输入布局、区间 M 选择、所有权、
动态容量、流水与跨核发布边界、Native 紧凑偏移算例。该次四文件修改去掉 docstring 后
AST 严格相同，证据 `dynamic_source_readability_review_v2_20261017.json`；未因注释重编译
或上卡。随后 metadata 新增实际关闭计数属于单独功能改动，另有下述新证据，不混为注释。
包级地图继续按实际新增模块更新，明确缺少的完整入口与后段，不以目录存在代表集成完成。

新增 [indexer_query_stages.py](deepseek_v4_flash_csa_spmd/indexer_query_stages.py)：query
INT8 投影、两次 scale 乘法/RoPE、Hadamard、量化以及独立 weights 投影/归约共六阶段。
与主 query Q_B 的 NZ2 不同，这里的 INT8 权重保持 ND[1024,8192]；BF16 输出通过外层
同 GM 别名供 Hadamard 消费，不在 incore 内 reshape 整张 GM。按 T 的 8/16/64 行块
动态计算波次，T<8 保留逐行尾数学，没有固定 S。

证据目录 `dynamic_indexer_query_20261017`：六段 canonical 数学 AST、240 组 T/worker
所有权/尾行覆盖通过。full_v1 的测试壳漏标输出 pl.Out，ABI 检查报 In→Out；只修测试
壳后 full_v2 六段完整 CPU 后端编译通过。`compiled_audit.json` 确认 Hadamard、quant、
weights_reduce 的实际 T 通过 Scalar ABI 进入循环，而非从 padding 容量推算。该六段
尚未融合到前段 MIX、未上卡；数学顺序未替换成历史“先合并两尺度”的版本。

### 2026-10-17：真实请求驱动的两路压缩投影、在线池化与 state 提交

新增 [compressor_projection_stages.py](deepseek_v4_flash_csa_spmd/compressor_projection_stages.py)，
保持主 M64/N64 首块 pl.matmul 与后续 matmul_acc，Indexer M16/N32 保留 init_cond；
两者 K512 二级流水顺序未改。主/Indexer pad 分开动态声明，循环只按 x 的真实 T。
`dynamic_compressor_projection_20261017/cpu_v1` 首次完整编译通过，一个测试 SPMD
同时调用两个 helper；两路各 1600 组所有权与 4800 组容量模型通过，完整 idx 循环对
canonical AST 一致。三个动态维度从真实描述符取得，未导入 NZ 设备格式探针。

新增 [compressor_pool_stages.py](deepseek_v4_flash_csa_spmd/compressor_pool_stages.py)：
主512列/Indexer128列两套 pool 与各自 state commit，共四个显式 worker helper。
pool 按 Native Q/L 分配请求，H=L−Qlen；历史读 Native 页表，本轮位置直接读投影加 APE。
每请求真实 token 区间循环，不使用 T/B 或 request*S。仍以当前位置后半区起步，再按
原七位置顺序更新 mi/li/oi。负历史位置与负历史页保留原来不同的 score 初始化语义。

活动请求的非关闭行置零；inactive/dummy、Q 未覆盖区与容量尾部保持未写，后续新 norm
必须只读活动请求，不能接旧版无条件全 T 消费者。state 提交独立执行，causal/page/offset
检查先于副作用，全部 pool 历史读取完成后才允许覆盖。两种投影和 pooled 容量均须>=T。
合法 Native Q 无重叠、有效写槽唯一与跨组 metadata 一致仍是完整入口合同。

`dynamic_compressor_pool_20261017/cpu_v1` 失败于测试壳漏导入 COMPRESS_RATIO，尚未
开始编译；修正导入后 cpu_v2 四段完整 CPU 后端通过。独立 `oracle.json` 共132例，
含 mixed/odd/短请求、dummy、零行、陈旧位置、历史页和非法slot边界；同源 AST FP32
在线数学对独立八项 FP64 稳定 softmax 最大绝对差 2.384185791015625e-7，检查阈值
atol/rtol=3e-6。这只是 CPU oracle 容差，不是放宽正式 NPU 精度门槛。state 精确写回、
poison/guards/只读输入通过；六条在线更新、初始化、七次循环与末尾除法对 canonical
AST 一致。共享历史错误交错反例产生主0.8574、Indexer0.5601差异，证明不能把 commit
随意并进每核 pool 尾部。尚未上卡，不构成完整请求生命周期或 prefix 共享验证。

### 2026-10-17：紧凑行数保留在 device，主压缩 norm/cache 改为请求子块

metadata 新增 `compact_offsets_and_count_stage`，与只需偏移的接口共用唯一
`_scan_compact_offsets`。worker0 同次扫描写 offsets[B] 和 INT32[1] 实际关闭数；完整
消费者二选一调用，不重复扫描。不回读Host、不把cos.shape容量当有效C。count来自Q/L，
陈旧position不能缩减count破坏后续请求偏移；norm初始化候选行，再按causal覆盖，cache
另作逐行有效性检查。工作区容量须覆盖count，完整入口须用已有shape信息保障容量。

`dynamic_metadata_cache_20261017/count_oracle.json` 的19个新增计数场景通过，含空B、
dummy、mixed/odd及仅worker0写；`count_v1/result.json` 完整CPU编译通过，没有重跑旧
RoPE/cache检查。原offset-only API 保持，后续完整 Indexer Hadamard 等待此生产者后在
设备 read/cast 计数作为 Scalar 循环界，不能将它取回Host作为JIT形状常量。

新增 [compressor_output_stages.py](deepseek_v4_flash_csa_spmd/compressor_output_stages.py)：
主512列的RMS/RoPE、调试KV与Native BF16压缩cache写回。每worker领取请求，再切成
<=16行子块，尾块读取/写回只到本请求end；保持原256→128→64平方和折叠、recip(sqrt)、
Native BF16 gamma与RoPE乘加顺序。先检查active/causal/关闭边界与compact行，再读slot，
最后检查page/offset；没有旧safe_row先读负地址的问题。

证据 `dynamic_compressor_output_20261017`：17个原数学表达式、逐行gather索引等价和
126组边界模型通过；前三轮frontend分别遇到显式Tile row_sum缺tmp、Tensor gather接口
用于Tile、索引混合Tensor/Tile。仅增加UB scratch并采用已验证的Tile绝对索引接口，
未修改归约或依赖。full_v4完整CPU后端通过，产物核对normed两段TSTORE的有效rows、
同GM读回及MTE3→MTE2 EVENT4，实际debug/cache写均受前置边界保护。未上卡、未测性能。


### 2026-10-17：Indexer 真实紧凑 norm/Hadamard 与直接 Native key/scale 写回

新增 [indexer_compressor_output_stages.py](deepseek_v4_flash_csa_spmd/indexer_compressor_output_stages.py)。
每请求按 Q/L/offsets 得到真实 compact prefix/count，先在本核清零自己拥有的紧凑候选行，
再对 causal 关闭行写 BF16 norm/RoPE；不同请求区间互不覆盖，无新增全局初始化任务。
pad 行保持未写。Hadamard 接收由 device count 读出的 Scalar C_valid，以 M16 valid_shape
仅加载真实行，C_valid=0 不读权重或数据；N64/K128 数学保持。输出容量须覆盖pad16(C)。

`dynamic_indexer_compressor_output_20261017`：54组同源CPU数学/地址oracle与51组
Hadamard几何检查通过。cpu_v1的Hadamard完整编译通过，norm的动态父Tile单行subview
在PTOAS校验失败；显式补充已由inner<rows保证的valid_shape=[1,64]后，cpu_v2只重编
norm并通过，未改依赖。产物中norm六个动态轴、Hadamard的真实device计数保留。初始化
零值不是写cache许可，后段仍独立核对active/causal/slot。尚无设备数值或性能结果。

新增 [indexer_cache_stages.py](deepseek_v4_flash_csa_spmd/indexer_cache_stages.py)。量化
按请求的实际关闭行分块，不反算固定S；FP32→BF16(rint)→FP32后再缩放，保留原amax/
div/recip和INT32→FP16→INT8顺序。FP32 scale按真实紧凑行保存，合法key直接写Native
页；scale由唯一worker0在等齐量化后串行64B读改写，保持同页相邻历史槽。

`dynamic_indexer_cache_20261017`：cpu_v1 的scale提交完整通过；quant首先失败于
常数[1,16]与归约结果[1,rows]的尾有效形状不一致，显式设置两常数的valid_shape后，
cpu_v2遇到Tile动态Python下标不支持。改为pl.slice(...,valid_shape=[1,128])后cpu_v3
quant完整后端通过；未重复已通过的scale编译。全程没有改变数学或编译器。
独立`oracle.json`共57例（19场景×worker1/3/48）通过，覆盖mixed/odd、非16对齐compact
起点、C=0、非法地址、陈旧causal、同页不同slot；11条量化表达式对canonical归一化
AST一致，key/scale/debug、保护区、只读输入、历史邻槽及串行RMW顺序检查通过。
这仅是CPU模型，A3 div/recip/cast的逐bit设备结论仍未建立。

一个明确的组装约束：quant会检查values、scale_values、slots容量，scale_commit没有
values入参；若非法values容量导致quant跳过，不能让commit读取上一轮旧scale。完整
入口必须统一按已有Tensor shape验证全部容量；当前合法输入合同要求覆盖C_valid。
不能把容量不足的输入当作支持场景，也不能靠注释替代最终入口校验。

上述新增源码均有中文整体流程、参数/布局/动态范围、AIC/AIV分工、同步与工作区说明，
关键range、尾块和RMW注明原因。两路压缩正在进行单MIX组合编译，独立helper通过不等于
已经完成组装、设备同步或完整CSA；七档动态性能验收仍全部未完成。


### 2026-10-17：纯 Top-K 动态 helper 与剩余链路迁移边界

新增 [indexer_topk_stages.py](deepseek_v4_flash_csa_spmd/indexer_topk_stages.py)，包含原实现的
`indexer_topk_half_leaf`、`indexer_topk_segment_pairs`、`merge2_top512_pairs` 与
`merge_top512_roots`。score/pair arena 的行容量分别为动态轴；有效候选数、逻辑起点、
根数由运行时 Scalar 传入。保留 sort32 的 score/index 位编码、各分支归并输入顺序及
同分优先关系；GM pair 写为 InOut，其余行保持。零候选由调用者处理，不能无条件读取根0。

`dynamic_indexer_topk_20261017/source_audit.json`：四个完整数学体与当前 canonical AST
一致，21 个 half-leaf、24 个 segment/offset、10 个根数模型通过。
`full_v1/result.json` 四个独立 helper 首轮完整 CPU 编译通过；`compiled_audit.json`
确认动态 Scalar/shape、各排序分支和真实 GM 写回。审计初次漏计两条 mrgsort，按原数学
六分支重数为20，只修正审计常数，没有因此改源码或重新编译。上述不是设备同分排序或
最终 Top-K 正确性的证明，也尚未接入 score 生产、请求分组及最终索引发布。

后续只读迁移方案保存为 `dynamic_indexer_score_source_plan_20261017.json` 和
`dynamic_attention_output_source_plan_20261017.json`。关键约束：请求内 G2/G6 分组和
有效 query 尾数必须由 Q/L 算出，producer/consumer/final merge 共用同一策略；短路径
INT32→FP32 与 Cube FP16 路径的数学不能混为一套。Attention 三处 token//S 需改用
原始边界；QK/PV 三槽的 READY/ACK 与尾部排空须保留，不能用 phase 完成通知替代。
O 投影的 packed stride、矩阵乘容量和 scale 容量必须分开，量化尾部只读真实 T 行，
同时初始化后继整 M 读取所需的 INT8 padding；原尾八行没有 valid_rows 不能直接照搬。
最终保持八组 FP32 反量化求和、weight scale、BF16 舍入及 HCpost 四残差加法顺序。
这些 JSON 是源码设计依据，未执行相关设备验证，也不表示八个后段切片已经实现。

### 2026-10-17：两路压缩组成一个动态 MIX 的 CPU 证据

`dynamic_compressor_pipeline_20261017/cpu_v2/result.json` 将两路投影、pool、Native
state 提交、主 norm/cache 与 Indexer norm/Hadamard/quant/key/scale 接入一个 MIX。
24 AIC 与 48 AIV，48 个 Tensor 入参、27 个独立动态维度、14 个信号阶段（含代际槽0）；
未提供样本形状，保留两路独立 T/Q/L/页表/容量。v1 在解析时失败，因为 vector 的
pl.func_attr(split_aiv=True) 前放了别名赋值；仅调整测试壳语句顺序后 v2 完整编译通过。

72 核控制流模型覆盖80组随机推进、每组三代调用（共240代），共1153536事件；检查全部核恰好
begin 一次、metadata 单写者、全部 pool 历史读完才 commit、全部 quant 完成才 scale
提交，以及等待 metadata 后才读取 device count。此模型不证明实际缓存可见性。
`cpu_v2/compiled_audit.json` 核对一次真实 MIX 提交、所有业务写回未被 DCE，以及
metadata 四次 Scalar 写均位于 AIV worker0 条件内。AIC 对 phase4/producer24 的 TWAIT
在 cube.cpp:1617，等待后的 DCCI/DSB 完成后才在1670读取 idx_count；AIV对应等待7075、
读取8385。编译器 ScalarWriteLineShared 为保守提示，核对实际分支后没有为此改编译器。

本轮未运行压缩组合 NPU 测试、精度或性能测试；单 MIX 编译通过不能替代 Native state
共享、缓存维护、设备 count 可见性及完整动态 CSA 的验证。

证据入口为该目录 `summary.json`，复现入口为 `reproduce_cpu.sh`、`audit_compiled.py`。
原始 result.json 的 random_schedules=240 是80组×3代的计数，summary 已澄清，不再误读为
240组×3代；没有为修报告重跑模型。CPP 的77/78个形参由48个Tensor、27个descriptor动态
维度和2/3个worker索引组成，不能等同77/78个runtime dispatch槽；本轮没有验证设备派发。


### 2026-10-17：动态源码中文总览与关键调度注释收口

将中文可读性作为目标中的固定交付要求：每个新的动态算子文件开头先说明整体算法、
输入输出/布局、实际有效边界与容量、分档依据、AIC/AIV分工、数据流、同步和工作区，
并按函数给出阅读顺序。关键 range/pipeline/spmd/split_aiv/notify/wait 说明为何这样
分配、等谁、何时可复用存储；不以逐行翻译语法替代原因与约束。

当前19个正式Python文件的顶部与关键节点已人工复核，记录于
`dynamic_source_readability_final_20261017.json`；最后补充了Top-K四helper仅能在AIV
执行、AIC不能消费其UB返回值的说明。包 `__init__.py` 增加完整目标数据流、Q/L/H/C
符号、实际行与pad容量、phase完成与ACK区别，以及共享历史读完再提交state的约束。
数学阶段按文件和职责拆分，完整入口尚未接齐的部分明确标为待接入。

19文件仅做语法解析和中文顶部存在检查，没有为注释收口导入NPU、重编译或重跑性能。
此前各阶段编译/数学/设备证据保持各自原有范围；完整动态CSA和七档性能仍未验收。
后续新增正式动态源码及组合探针沿用同样的中文总览与调度注释要求。


### 2026-10-17：Indexer请求内分组与输出链源级迁移

正式新增 `indexer_group_stages.py`：Native Q/L 驱动请求内G2/G6分组，小前缀
`offsets[B+1]`与8项device计划统一producer/consumer/final merge的组号、叶区间、
平衡策略和根数。任意正query长度可分多组，尾组补零、不跨请求读取；实际position须
满足causal关系。系数保留先分别FP16转换再相乘的原数学和对角块布局。所有容量动态，
不回读Host，不在GM展开T行映射。G1/roots0仅供后续短Vector几何，当前未专门验证该路径。

`dynamic_indexer_group_20261017`：23组输入，G2/G6、worker1/3/48的138个系数所有权
场景与1744个组几何检查通过；叶计划完整数学AST与FP16转换乘法表达式保持。CPU v1
末端offsets[B]的裸tensor.dim下标生成`ptr[]`，PTOAS报缺SSA；显式cast为INDEX后v2
的geometry/coeff2/coeff6完整后端通过，无编译器修改。产物保留动态B/Q/T/独立容量、
每组唯一TSTORE及2/6个FP16乘法。初次审计误在CPP查pto.store字样，改查真实指针赋值
后只读通过，未重编。summary/contract/oracle/compiled_audit均保留，未NPU。

O_A/分组量化与O_B/dequant/HCpost正式源码已分别完成源级迁移和必要CPU编译，证据在
`dynamic_output_projection_a_20261017`、`dynamic_output_projection_b_20261017`。
正在 `dynamic_output_pipeline_20261017` 将它们组合为一个24AIC+48AIV MIX；v1测试壳
把Cube只读quant/scales/output误标InOut，触发返回值使用纪律校验，已按角色区分
只读/写入再编译。当前不声称该组合或完整CSA已通过设备验证。

本轮用户询问七档动态性能：再次核对`current_acceptance_cases.json`，七档仍全部未
验收。`selected_seven_readback_20261017.json`中的数据是历史静态单SPMD与当轮原动态
分task的比较，旧8K/B24已移出当前验收，新增128K/B28无对应历史成组结果；不能把这些
旧数字或局部CPU通过当作目标动态版本性能。当前完整入口及score/attention并发流水
还在接入，暂不具备新动态整层七档计时条件。

### 2026-10-17：TMR/HBG 的 Host 标量差异与动态 SPMD 处理边界

核对当前 canonical `decode_csa.py`、`host_metadata.py`、`native_adapter.py` 和
`service.py`，TMR→HBG 增加的运行时 Host 控制入参只有 `host_max_seq_len`。
它是包含本轮 query 的未压缩 KV 最大总长度：TMR 在设备编排中扫描
`kv_seq_lens`；HBG 直接使用 Native 已有 CPU `decode.max_seq_lens`，不执行
device Tensor 的 `.item()` 回读。原始每请求长度、Q 前缀、positions、slot、
页表仍是原 Native device Tensor。B/T 的 shape/numel 与入口判定使用的
max_seqlen_q/请求计数本来就是 Host 可见描述信息，不是本次新增的数据搬运。

旧 Host 最大值用于 Indexer 的 Vector/Cube、短/长 score 与 Top-K 编排选择。
HBG 被外层 NPUGraph 捕获后，更新 Python 标量不能改变已捕获的 Host 分支；
现有 `CSAHostMetadata.graph_key/validate_replay` 因而检查拓扑区间。

目标单 SPMD 将实际工作量放回核内：已实现的
`indexer_group_stages.py:indexer_group_plan_stage` 由一个 AIV 扫描实际 Q/L，
生成请求组前缀、最大压缩历史、叶数/叶尺寸；`indexer_query_visibility` 再按
每个 query 的 L/position 限制真实可见候选。入口仍须在计划写回发布后让
AIC/双 AIV/最终 Top-K 消费同一计划，并保证策略常量和工作区合同一致。
这部分 helper 已有 CPU 证据，完整入口的相应同步和图重放尚未验证。

Host 已有最大长度可以用于选择覆盖区间的 tile/分核策略和匹配的图，但不能
替代 kernel 每步的 device 长度、循环或尾块。跨策略或容量区间时要切换匹配
入口/图；区间内 metadata 变化由核内处理。最终 Host 策略绑定尚未接齐，不能
把“无需回读”表述为完整 ABI/图兼容已经验收。核内 O(B) 计划扫描及通知成本
必须计入完整 L1 耗时，后续只能通过共享扫描等真实优化减少，不能隐藏计时。

### 2026-10-17：规划位置讨论——优先评估少量 AIV 与前段计算重叠

用户指出 AICPU Scalar 优势与约5μs基础调度成本之间的取舍；该数值作为本次
讨论给定的估计，不是本会话新测结果。当前 Native SAS/QLI 都是独立AICPU算子，
由 metadata builder 提交，结果写入device INT32[1024]，再由对应attention/
indexer读取。本轮计划生产与消费存在前后依赖；Host异步提交不能据此宣称两者
并行。step内同类层可复用计划，并非每层都重新启动一遍这两个规划算子。

对目标单SPMD的建议优先级调整为：公共Q/L几何由一个、必要时少量AIV的Scalar
计算，并与不依赖计划的HC/QKV前段重叠；首次真正消费计划时才等待，避免把
所有72核放在入口屏障后。当前HC起始八行分块，七档T<=192，数学工作最多
分配给24个AIV，存在可用于规划的初始空闲worker。应按实际tile数选角色，
不能把这七点或某核永久空闲写死；更大输入需另核对角色占用与关键路径。

单写者方案减少多核协同，不消除Scalar缓存维护：plan完整写回发布后再通知，
消费者等待后维护其计划缓存；plan和信号分离，生命周期内不覆写。多规划核
仅按独占缓存行的请求分段写局部结果，再由单核合并，暂不引入共享计数器。
依赖本层Top-K的Attention有效块仍在本层生成，不能冒充step公共计划。
当前group_plan_stage固定worker0的实现仅是基础功能版，尚未实现上述初始
空闲AIV选择/前段重叠，也没有性能结论。是否外置AICPU或做跨层共享，需要
按包含准备、发布、等待和核占用的实际成本决定，不能只比较孤立Scalar速度。

### 2026-10-17：公共 metadata 前移与完整源级动态单 SPMD 接线

按本轮讨论新增正式 `runtime_abi.py / cube_schedule.py / vector_schedule.py / entry.py`
及捕获前 `policy.py / binding.py`。唯一 JIT 入口为 `decode_csa_spmd`，一次24组SPMD，
24 AIC和48 AIV构成一个MIX；整条HC→QKV→两压缩→Indexer→Attention→输出投影/HCpost
现已在源码层接通。数学继续调用本目录源级动态helper，没有导入静态IR或冻结排程。
132个Tensor入参、58个动态容量/请求/token轴；44个phase槽含代际槽0。生产源码顶部
新增中文整体总览、布局/别名/容量、阶段分工、等待环约束和未验收边界。

公共计划只包含两套Q/L的compact前缀/实际count和Indexer分组/叶计划。AIV先完成并
发布HC widen，再由 `min(ceil(T/8),47)` 号AIV扫描：前段有空核时选首个空核，满48核
时选47号先完成其widen，不永久保留核。其余角色不在入口等META，AIC可开始HC投影；
首次读取计划的消费者才等待该真实producer。helper里的worker=0表示外层选定唯一
写者，物理核不是固定0号。计划一次写入、后续只读；phase发布仍执行业务fence/
cache维护与单写者信号发布，wait后做读侧维护。TopK依赖的Attention有效块计划仍
留在TopK后。Indexer scale的64B RMW仍由AIV0单写，不能和公共planner混为一谈。

初始统一编排先保留保守阶段依赖：pool全部历史读取结束后才提交Native state；
Score/Attention的AIC与双AIV必须同时推进，不能先等待整段AIC完成。Indexer事件0/1，
Attention事件2..5；两段各自保留环槽READY/ACK及末尾排空。输出packed→OA→量化→OB
→HCpost接通，HC post/comb与原hidden跨阶段保留。暂未宣称这些保守等待已有性能收益。

`dynamic_full_pipeline_20261017/assemble_source.py`只是本次迁移记录：从已通过组合的
动态源签名整理重复ABI，编排体人工编写，不接收B/T/history，不读取IR/DAG，不在生产
运行时使用。正式运行只导入ops/pypto。原静态试验和全部历史结果保留。

CPU证据（没有给任何样例shape、没有执行NPU）：
- `cpu_v1/result.json`：program完整后端PASS，最终一个`rt_submit_task`/MIX，58轴保留；
- `kernel_v1/result.json`：补查实际直调使用的kernel执行模式，完整后端PASS，同样一个MIX；
- `schedule_model.json`：直接解释正式两份schedule AST，144组不同T/Q分布/G2/G6/随机
  推进顺序，477792个事件完成；覆盖T0、尾行、T376/377/384/385的planner满核边界、
  dummy/gap和不等长请求。验证的是phase偏序及角色配对无等待环，不是硬件cache/FFTS
  或数值模拟。此前通过的独立helper数学/CPU检查没有重跑。

编译器对内联的metadata Scalar写发出多实例共享cache行告警。实际统一编排外层明确
`if worker == planner`，本次只有一位写者；不能因为告警消失/存在判断硬件一致性。
后续仍须核对最终CPP的条件及真机计划值；没有关闭检查或修改依赖仓库。

捕获前绑定保持Native BF16 norm和NZ权重原对象，Q/L/position/slot均为device Tensor，
新增Host操作仅是已有shape/stride描述与独立工作区分配。策略覆盖工作量区间而非七点
白名单；当前先支持Native NZ2和Cube Score（Host已有max_seq_len//4>=2048），小历史
Vector Score尚待接入。允许不同策略二进制，但有效分组/叶数/地址仍在设备重新求值。

已提交首轮正式权重128K/B16/S6完整设备对照：
`task_20261017_182726_328850326556`，输出目录`dynamic_full_pipeline_20261017/device_b16_v1`。
只对比当前原动态分task PTO，不运行Native attention性能测试；计划A→B→A检查输出、
TopK、五份cache/state、保护区和代际，第一轮失败即保留诊断停止。提交不是PASS；
当前仍没有新的完整动态CSA七档性能数据，更没有30%目标达标结论。

首轮设备提交记录修正：v1任务在参数解析即退出，队列自动追加`--device 8`，新脚本原先
未接收该参数，未执行模型或候选。增加与`TASK_DEVICE`一致性检查后提交v2：
`task_20261017_182902_33224055334`。v2正式权重、Native fixture、130余项绑定准备完成，
原动态分task PTO的A/B参考及保护区通过；候选首轮在PyPTO入参预检查被拒，未发射NPU：
`Parameter 'idx_key0' partially overlaps another tensor with a writable alias`。

原因是Torch interop允许同字节范围别名与只读重叠，但拒绝可写Native cache加部分
重叠只读key根入参。Native Indexer页16417×4160字节，0/64字节起始的128列视图
都只有完整key行范围，与cache本体不等长。不是Scalar同步失败，也不是设备精度失败。
修正限定在本仓库：只传Native cache本体，JIT编排内部按其shape建立flat/0/64字节
零拷贝GM视图，再传给MIX角色；不复制cache、不改数学、不放宽Torch检查、不改依赖。
根入口从132减到130个Tensor，内部MIX仍含key视图。`kernel_v2`补验该ABI/内部view变更；
原phase模型的数学与等待顺序未变，因此不重复144组推进模型。

### 2026-10-17：动态 cache 视图的 JIT 类型推导缺口

`kernel_v2/v3` 在符号特化阶段失败，尚未发射设备：`indexer_score_cube_stage`
的 `idx_kv_cache` 缺少推导类型。原JIT元数据提取器支持直接动态维度别名，却不支持
`pages * page_bytes`、`(pages * page_bytes - 64) // 128` 这样的维度运算；把
create_tensor写在reshape前面不能解决，后续赋值会正确清除旧类型，不能靠虚假分配
或静态容量蒙混通过。正式入口改为先命名slice、再reshape，只产生Native cache的视图。

因此上一段“不改依赖”的预期没有实现。本轮在既有独立PyPTO worktree新增一项局部
修正：只对能追溯到Tensor维度及整数运算的reshape形状产生descriptor动态维度；
IR仍保留原始运算与真实view，没有把动态容量变成Host常量，也没有放宽Torch别名检查。
任意device标量或未知局部变量仍保留原有拒绝行为。共享安装和Simpler没有修改。

最小复现及验证保存于 `dynamic_full_pipeline_20261017`：
- `reshape_before.log`：独立worktree中两项复现失败，一项未知标量拒绝检查通过；
- `reshape_after.log`：修正后22项相关元数据/reshape回归通过，含最小完整后端编译；
- `pypto_reshape_metadata.patch`：局部修正、三个回归及中英文说明；
- 独立worktree增量构建、ruff检查通过。pytest使用单进程，避免xdist新进程遗漏
  本会话进程内依赖隔离后误加载共享安装；先前那次无效并行结果不作为修复证据。

`kernel_v4`正在补验完整CSA的130根入参及内部零拷贝视图。它仍不能替代设备精度、
信号缓存语义或性能检查；未重跑已通过的phase偏序模型和独立数学helper。

`kernel_v4`完整codegen已通过，一个MIX、sync_start保留，130根Tensor入参没有
新增GM分配。原检查误将两个内部Key视图维度也要求保留独立Extract：它们实际上在
内联后变为原cache尺寸的乘除表达式。保留原`result.json`失败记录，改为分别检查
56个根动态轴及两个零拷贝派生视图；直接检查已有产物而未重编译，证据为
`kernel_v4/codegen_audit.json`。最终AIV CPP仍有唯一planner条件分支，未变成48核
重复写公共plan。相关编译器保守告警不关闭。

重新提交128K/B16/S6 A→B→A：`task_20261017_184705_395742613755`，
输出`device_b16_v3`。同时对G2/768候选块/双根策略进行一次完整CPU组合编译，
对应`kernel_medium_v1`；这是另一策略的接线检查，不重复长历史组合，也不占NPU。

### 2026-10-17：正式动态完整CSA首次七档设备与L1对照

`device_b16_v3`完成PASS：128K/B16/S6，A→B→A三代，9项输出/TopK/score与
六份cache/state视图全部完全一致，31项Native保护检查及70项暂存保护检查通过；
信号隔离区未改写、所有72核代际正确。公共META只由预期core36发布。G2完整组合的
`kernel_medium_v1`也完成CPU PASS。这里才获得完整源级动态版本的设备证据。

新增`l1_compare.py`沿用历史六列L1记录的第4/5列incore起止定义；两侧同进程、
同卡、同输入和权重，先验证eager与图，再AB/BA交替采样，每侧5次取中位数。
恢复cache/state不进入测量窗口。最终再次比较完整结果、保护区和信号代际。
没有运行Native attention或把外部事件时间替代完整L1。所有原始记录及trace保留。

七档均完成精度、图和保护检查，但性能全部退步，不能宣称目标达到：

| 历史 | B | 原分task μs | 动态单SPMD μs | 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K | 4 | 513.16 | 1103.72 | -115.08% |
| 128K | 8 | 640.32 | 1387.26 | -116.65% |
| 128K | 16 | 868.34 | 1521.52 | -75.22% |
| 128K | 24 | 1170.24 | 2030.22 | -73.49% |
| 128K | 28 | 1292.28 | 2286.56 | -76.94% |
| 8K | 16 | 654.76 | 1285.32 | -96.30% |
| 8K | 32 | 988.30 | 1628.42 | -64.77% |

任务、policy、实际metadata写者及逐次样本见
`dynamic_full_pipeline_20261017/seven_shapes_v1.json`与`SEVEN_SHAPES_V1.md`。
七点均只观察到一个正确META写者，没有48核重复写plan；尚未覆盖mixed-query、
padding、生命周期或同图长度/页表变化。上述测试也不改变Native精度历史结论。

下一步定向归因：比较默认TWAIT和已有100tick未就绪退避（保留每读DCCI、写侧
TNOTIFY、payload维护），另给每核每phase的隔离槽第2条cache行写时间戳和累计
等待耗时。插桩绝不进入正式性能判定。第一个诊断任务
`task_20261017_190012_2613381911`因注入器错把原始`AICORE void`匹配成后处理后的
`static __aicore__ void`而主动终止；CPU源检查修正后另提
`task_20261017_190101_28537610312`，不覆盖首轮目录。
无插桩退避对照为`task_20261017_190014_26217529674`。

### 2026-10-17：等待退避收益与动态 Attention 页表检查定位

128K/B16的`perf_h131072_b16_backoff_v1`完成PASS：原动态分task中位数
870.86 μs，单SPMD为1149.80 μs；相对默认等待的1521.52 μs明显改善，但仍比
同轮基线慢32.03%，不满足30%降低目标。仅改变尚未就绪时的信号轮询间隔，
每次读前DCCI、TNOTIFY写侧、业务缓存维护与全部数学保持不变。当前由测试
编译回调显式应用`wait_codegen.py`，尚未成为正式binding的默认行为。

`diag_h131072_b16_clock_v2`完成PASS，得到1706个阶段端点；插桩含额外Scalar
写回及DCCI，不作为性能验收时间。默认等待下公共META阶段自身约2.72 μs，
结束于HC前段；Attention plan最忙核的非TWAIT区间约168.20 μs，主要是新增
页保护每query最多512次Scalar候选查页。阶段区间也包含循环、缓存维护和
插桩开销，不能将其全部称作数学时间。现有70项工作区和31项Native保护检查
继续通过。证据位于`phase_clock/phase_clock.json`及对应trace。

正式`attention_plan_stages.py`新增保守快速路径：按请求用Vector检查当前长度
可能使用的压缩页表前缀，全部物理页合法才批量生成候选合法mask；有空洞、
短表或超出FP32精确检查范围时保留原逐候选INT32/INDEX路径。同八行tile中
相邻query只复用本次调用的本核判断，不跨decode缓存。实际最终索引仍检查
0<=index<upper，SWA页检查、原bias/归约顺序与dummy条件均保留。

CPU迭代证据均保存，没有覆盖失败记录：
- `kernel_plan_fast_v1`：A3的1×1归约结果不满足32B列布局，失败；
- `kernel_plan_fast_v2`：动态父Tile子切片的有效行数推导不符，失败；
- `kernel_plan_fast_v3`：改为8行物理归约布局、显式单行valid_shape后，完整
  kernel后端PASS，一个动态MIX，未提供样例shape；
- `plan_fast_oracle_v1.json`：32个CPU边界用例PASS，覆盖空洞、越界、短表、
  dummy、mixed-query、尾行和大整数边界，20个原数学表达式AST核对一致。

设备验证任务`task_20261017_191833_66770729044`已提交，目录
`perf_h131072_b16_plan_fast_v1`，包括三个专用页表边界probe、完整B16精度/图/
保护区，以及启用同一100tick等待退避的五窗L1成对比较。提交不是PASS，
CPU结果也不能替代设备页表语义或七档性能。

`perf_h131072_b16_plan_fast_v1`已完成PASS：完整输出/TopK/score/cache/state、
图重放和保护区通过；三个独立NPU页表probe覆盖T13尾行、Q=[0,6,9,13]、dummy、
空洞/物理页越界及短表，CPU参考逐元素一致，mask保留列和容量尾部未写。
成对L1中位数：原分task875.84 μs，单SPMD1006.56 μs，仍慢14.93%；相比前一版
1149.80 μs减少143.24 μs，但跨轮差值只供优化观察，不是相对基线达标证据。

此轮新增probe的kernel_config误混入基线名称表，func_id=0的显示名称被覆盖；
只影响trace函数名，不影响原始incore起止、任务/核数量或上述中位数。
`l1_compare.py`改为每侧只接受自己唯一入口的配置，`correct_trace_names.py`
基于已有原始数据另存`corrected_names/`，不重测、不覆盖原记录；修正清单
`trace_name_correction.json`逐窗断言首尾跨度完全相同。

进一步插桩任务`task_20261017_192108_71720427030`（`diag_h131072_b16_fast_backoff_v1`）
PASS：公共META自身约2.20 μs；Attention plan最忙核非TWAIT区间约16.08 μs，
此前168.20 μs的主要长尾已消除。此时Indexer compact norm完成约408.90 μs，
query Hadamard已在326.22 μs完成，主压缩链仍挡在Indexer pool/norm前。

因此新增独立分支重排候选，记录脚本`reorder_independent_compressors.py`及
`schedule_overlap_v1/before/`保留原源文件：AIC先Indexer压缩投影/query Hadamard，
再主压缩投影；AIV先Indexer pool→所有读者完成集合→state commit→norm，再
query量化/系数与主压缩链。公共META在首次使用offset前等待，双分支仍各自
遵守Native共享prefix state读完才能覆盖的合同。没有删数学、改K累加或引入
shape实值化。`schedule_overlap_v1/schedule_model.json`144组477792事件PASS，
只证明phase/角色无等待环；`kernel_schedule_overlap_v1`补验完整编译中，
收益和硬件同步尚未验证。

重排的完整CPU编译PASS。脚本旧版只接受相对目录名，本次误传绝对路径导致
产物前缀重复；已修正脚本并把既有产物移至`kernel_schedule_overlap_v1`，
`path_relocation.json`保留原路径，不为整理路径重新编译。
设备任务`task_20261017_192558_80156815231`（`perf_h131072_b16_schedule_overlap_v1`）
PASS，输出/TopK/cache/state、图和保护区通过；原分task867.38 μs，单SPMD
987.10 μs，仍慢13.80%。相对前轮1006.56 μs只是小幅改善，未达目标。

继续对照canonical源码发现五份流式权重的BYPASS在完整编排迁移时遗漏：
`qkv_proj_rope.py`中的wq_a/wkv，`q_projection.py`中的wq_b，以及
`decode_o_proj.py`中的wo_a/wo_b。数学helper注释已有该合同，但外层没有实际
声明，不能把注释当作生效证据。`kernel_weight_bypass_v1`在预定义MIX的SPMD
派发scope放set_cache_policy，被PyPTO明确拒绝（此scope没有直接GM读取）；
不是设备精度失败。失败源保存在该目录`entry_scope_attempt.py`。

改为在实际`cube_schedule`预定义incore上用PyPTO现有`func_attr(cache_policy)`
声明。该属性当前按参数序号表达，CPU脚本校验1/8/10/50/53分别对应
wq_a/wq_b/wkv/wo_a/wo_b，避免签名变动后静默设置错权重；测试组装脚本从
参数名计算序号。`kernel_weight_bypass_v2`正在检查完整lowering及最终TLOAD
提示，未修改编译器。只有最终生成及真机结果成立后，才判断恢复此策略的收益。

`kernel_weight_bypass_v2`完整编译PASS，但生成物核对得到更严格的限制：PTO IR
有25处`l2_bypass`，PTOAS输出没有`TLoadL2Hint`；去除整行`//`注释之后，与
`kernel_schedule_overlap_v1`的C++文本完全相同（不删除标识符、表达式或
指令）。原分task生成物也没有该提示。因此源码缺失虽已补齐，但本工具链下
没有实际机器侧变化，不能据此解释退步，也不为同一C++单独重跑NPU性能。
证据`kernel_weight_bypass_v2/cache_policy_audit.json`；后续可据此做PTOAS
独立最小复现，本轮没有改其源码或切换共享工具链。

HC前段发现另一项不必要的串行：混合/输入归一化只读pre与hidden，不依赖
comb Sinkhorn；原单SPMD却先算comb再发布NORM。`advance_hc_norm.py`将
NORM提前发布，让AIC的QA与AIV的Sinkhorn重叠。gates和mix_norm严格采用相同
M8/worker条带，pre由本核生产且GATES发布已排空，所以删除等待全部gate
owner的冗余跨核集合；comb仍在进入后续阶段前完成，最终HCpost数学不变。
旧源码保留`hc_norm_early_v1/before/`，144组phase模型PASS，任务
`task_20261017_193328_93136027979`（`perf_h131072_b16_hc_norm_early_v1`）
进行完整精度、图与L1对照中。没有复测未改变的页表probe。

`perf_h131072_b16_hc_norm_early_v1`完成PASS，原分task859.14 μs，单SPMD
953.16 μs，仍慢10.94%。完整输出/TopK/score/cache/state、图和保护区通过。

下一轮源码级`prioritize_indexer.py`候选将Indexer压缩/query输入提前，在AIC
主QB期间处理AIV Indexer阶段；KV安排在等待Indexer RoPE之前。把QR/RoPE
的获取同步移到第一个消费者，主Query稍后读取仍不缺少可见性条件。所有
数学helper、K链与owner未变，未使用历史DAG或实值shape，旧源码位于
`indexer_priority_v1/before/`。144组phase模型PASS。

当前候选三点完整精度/图/保护检查均PASS，但均未达性能目标：

| 历史/B | 原分task μs | 当前单SPMD μs | 降低 | 任务 |
| --- | ---: | ---: | ---: | --- |
| 128K/B16 | 867.98 | 951.82 | -9.66% | task_20261017_194019_104644110295 |
| 128K/B4 | 494.50 | 587.28 | -18.76% | task_20261017_194305_10967838023 |
| 8K/B16 | 668.48 | 877.44 | -31.26% | task_20261017_194305_109677916112 |

目录分别为`perf_h131072_b16_indexer_priority_v1`、
`perf_h131072_b4_indexer_priority_v1`、`perf_h8192_b16_indexer_priority_v1`。
B16相对前一版953.16 μs基本持平，不能称为明确收益；另外两点只说明当前
动态区间组合仍正确，不构成重排本身的同轮消融。其余四档没有此版本新数据。

诊断`task_20261017_194305_109684217391`（`diag_h131072_b16_indexer_priority_v1`）
PASS：Indexer scale发布约342.84 μs、Score Cube结束580.22 μs、Attention
Cube结束803.84 μs，后者非TWAIT区间最忙约171.62 μs。插桩不能用于正式计时，
也没有再凭1～2 μs跨轮差值扩大调度收益结论。下一步关注数学阶段内部地址准备
成本，历史MGATHER/独立KV Tile流水失败方向保持关闭；只借鉴已选过的本核
UB地址缓存思路，重新编写保留动态页表/物理容量保护的源码实现。


### 2026-10-17 19:58：动态Attention页地址UB缓存候选与probe封装修正

正式源码新增`attention_address_stages.py::compressed_half_addresses`：读取本轮
64个候选与真实C列压缩页表，在本核UB用INT32位移/gather求页号、物理行；
`attention_stream_stages.py`仅在256<C<=2048且物理cache行数<=2**31时启用。
动态请求/页表/位置/尾块检查保留，负索引、逻辑列越界、负页及物理页越界仍
屏蔽；夹紧只保证UB gather合法，不能把无效KV变有效。没有MGATHER、新GM
工作区或数学顺序变化。其他工作量/容量沿用Scalar路径。

`kernel_qk_address_cache_v1`完整符号编译PASS，无样例shape；旧正式stream
保存在`qk_address_cache_v1/before/`。首轮设备任务
`task_20261017_195315_12814974025`的整链输出/TopK/score/cache/state及保护区
检查通过，但独立地址probe把auto_scope=False的helper直接放在SPMD编排层，
触发Tile缺少InCore归属错误，整任务FAIL，尚无L1性能数据。此失败属于测试
封装，未放宽断言或跳过边界验证。

probe改用显式`@pl.jit.incore`封装；`address_probe_cpu_v2`独立CPU编译PASS。
真机复提`perf_h131072_b16_qk_address_cache_v2`，先验证C=1/255/1025/2048、
INT32边界、页表空洞/越界与保护区，再做同轮五窗口完整L1对照。

`task_20261017_195930_139088123457`完成PASS：四种页宽的独立地址边界检查、
完整CSA精度/状态/图重放/保护区均通过。五窗口原分task中位874.08 μs，
候选962.66 μs，慢10.13%；相对上版951.82 μs也无收益。因此恢复
`qk_address_cache_v1/before/attention_stream_stages.py`，不把无收益候选继续
叠在正式实现里。完整候选源码保存在本轮结果`source/`，地址helper另存
`qk_address_cache_v1/attention_address_candidate.py`，独立probe转为导入此
历史helper；正式目录移除该未选模块。本轮没有改动PTOAS/Simpler或共享安装。

接下来做group_descriptor_v1：早期唯一AIV扫描除组前缀外，生成每组
(request,first,live,max_visible)四项设备描述符，替代Score每个(group,leaf)
在AIC/两AIV重复O(B)定位与组内位置扫描。仍由本轮Q/L/positions实时计算，
只扩展原group_offsets工作区，不新增Host数值、task或全核入口集合。
系数也按组号条带读同一描述符；每query的causal保护和FP16运算仍保留。
旧源码保存在`group_descriptor_v1/before/`；尚无本候选编译或性能结论。

组描述符CPU oracle通过1744项几何、138组系数owner检查，含同T不同Q、
混合query、陈旧position、dummy及G2/G6。叶划分与FP16 cast/multiply数学
AST仍与canonical相同。更新活跃probe/oracle接口，旧证据快照保留不改。
`kernel_group_descriptor_v1`完整符号编译PASS，仍为单MIX、动态轴和原130根
Tensor ABI；144组phase依赖模型PASS（不代替设备一致性检查）。
新`group_device_probe.py`从7号AIV发布描述符，其余AIV读取不同组，输出行独占
64B；G2/G6分别在同址图中A→B→A→全dummy→A重放，专门检查跨核发布与陈旧
描述符不能被误用。`group_probe_cpu_v1`编译PASS，随完整B16作业验证。

`task_20261017_200745_15374263310`（`perf_h131072_b16_group_descriptor_v1`）
完成PASS：完整CSA输出/TopK/score/cache/state、图重放及保护区通过；独立
描述符probe的G2/G6共10次同址图内metadata变化通过，包含陈旧位置、失活请求、
全dummy后恢复。原动态分task中位872.10 μs，单SPMD951.50 μs，慢9.10%。
与前版951.82 μs基本持平，不能宣称性能提升。暂保留该单写者前置描述符方案：
它符合本轮只推导一次、跨核复用的metadata分工，且没有引入新的task/根入参；
收益结论需其他工作量区间验证，不把CPU复杂度下降当作完整耗时下降。

组描述符新增区间验证：128K/B28 `task_20261017_201211_16164271196`，完整精度/
图/保护PASS，原分task1301.04 μs、单SPMD1421.70 μs（慢9.27%）；8K/B32
`task_20261017_201212_161664323588`同样PASS，983.94/1213.54 μs（慢23.33%）。
目录为`perf_h131072_b28_group_descriptor_v1`和`perf_h8192_b32_group_descriptor_v1`。
本版尚有四档未更新，不以之前七档代替当前验收。

下一候选scale_page_owners_v1针对实测B16约13.9 μs的串行FP16 scale提交：
物理page%48决定唯一AIV writer，同页全部更新由同一个核完成，避免Native
64B scale区多写者RMW竞争。早期唯一planner生成每owner链表，consumer只
处理自己链中的合法compact行；所有Q/L/positions/slots仍为device输入。
暂扩展已有group_offsets工作区，不新增外部Tensor/Host数值或独立meta任务。
将PHASE_IDX_SCALE改为每个AIV完成发布，Cube/Vector在消费cache前等待48个
writer；Prepared工作区必须新建，不复用旧阶段合同的signal。旧源码保存在
`scale_page_owners_v1/before/`，尚未编译/验证，不预记收益。

`kernel_scale_page_owners_v1`完整符号编译PASS，仍是一组MIX、130根Tensor。
原生device while链表已通过PyPTO/PTOAS lowering，未新增编译器修改。
108组CPU地址/页writer互斥/全部cache字节/保护区检查通过，含1/8/48个worker；
144组更新后的阶段模型PASS，包含48个scale完成发布。CPU不能证明DCache一致。
`scale_page_probe_cpu_v1`编译PASS；新设备probe在同一图内切换同页、异页且
owner碰撞、陈旧位置、非法slot、dummy后恢复，逐字节检查完整cache及保护页。
完整B16配对任务随此probe提交，未再重跑已通过且不变的group/address probe。


### 2026-10-17 20:30：scale页并行首轮超时，定位自然while回边丢失

`task_20261017_202205_17915867050`（`perf_h131072_b16_scale_page_owners_v1`）
在首次完整candidate执行超时，未通过功能、未产生性能。设备日志为Simpler
S1 running-stalled；该日志不代表AICPU是根因。独立probe的生成PTO IR已有
直接证据：自然`while node >= 0`变为`scf.while : () -> ()`，条件永远读取
循环前的head；末尾虽然读取next，scf.yield却没有携带它，C++回跳不更新node。
任一非空链因此不能退出。IR路径`scale_page_probe_cpu_v1/kernel/ptoas/
scale_page_probe_core.pto:308`，完整失败结果保存原源码与生成物。此前“编译
通过”仅说明lowering没有报错，不证明循环语义正确。

先在本仓库改为显式`pl.while_(init_values=(initial_node,))`、pl.cond及
pl.yield_传回next，检查生成回边后再运行。未修改PyPTO/Simpler，本轮不把
错误归因于DCache可见性；编译器自然while问题留作独立复现和后续定位。

显式while独立编译`scale_page_probe_cpu_v2`恢复`(index)->(index)`回边，
next成为scf.yield操作数，CPU独立预期的108组字节/owner检查再次通过。
设备脚本将该新增probe前移到完整CSA前，先排除局部循环/发布错误，避免
先运行整链才暴露同类问题；原断言和性能口径不变。

`task_20261017_203116_195316828364`（`perf_h131072_b16_scale_page_owners_v2`）
完整PASS：显式while的六次独立同址图probe、完整CSA输出/TopK/score/cache/state、
图重放及全部保护区均通过。五窗口原分task中位872.52 μs，单SPMD932.54 μs，
仍慢6.88%；相对上一候选951.50 μs减少18.96 μs仅是跨轮参考，不能当作同轮
消融。当前保留已验证的显式while方案，继续跨区间检查。

等待期间曾怀疑大融合while前的自动事件维护，尝试增加每owner链长并用
pl.range遍历，快照保存在`scale_page_owners_v3/source/`，CPU地址检查通过、
`kernel_scale_page_owners_v3`编译单独记录。随后v2完整任务正常完成，所以
该怀疑没有成立证据；没有第二次超时，也没有对for备选发起NPU测试。正式
源码恢复v2，不因等待首次CCEC/执行的中间状态而叠加未经必要性证明的修改。


### 2026-10-17 20:43：页owner方案跨区间结果与完整metadata图重放补验

显式while版本scale_page_owners_v2的另外两点完整精度/图/保护区PASS：
128K/B28 `task_20261017_203540_20420884427`为1308.74/1416.62 μs（原分task/
单SPMD，慢8.24%）；8K/B32 `task_20261017_203540_20424403390`为995.90/1183.92 μs
（慢18.88%）。此前同两点为1421.70/1213.54 μs；跨轮差值不能当作同轮消融，
当前仍无任何逐档30%达标结论。三点之外四档未更新。

自然while最小CPU复现`natural_while_carry_repro.py` / `natural_while_carry_v1`
稳定复现：natural无回边，explicit有index回边。无需运行可能不退出的设备
最小核；原CSA失败任务已有真机证据。没有因此改动依赖仓或重复提交旧#3009。
for链长备选`kernel_scale_page_owners_v3`因count在inline上下文的INDEX/INT32
类型冲突编译失败，正式代码已恢复已通过v2，未为了该备选扩大修复范围。

新增`metadata_replay.py`在完整算子同一张图、全部地址固定的条件下，切换
历史131072/131071/131070和正反物理页映射，并改变C4合法关闭数量。三组用
同一hidden及初始cache数据，只有本轮metadata变化；各自先跑原分task PTO
形成参照，再执行A→B→C→A→B→A，同步检查五份完整allocation、输出/TopK/
score、输入metadata、工作区保护与72核代际。计划暂存每步填毒，禁止借用
上一轮内容。固定S6，不能据此宣称mixed-query、padding或请求生命周期已过。
任务`task_20261017_204256_222892518376`，目录`metadata_h131072_b16_v1`，
尚未有设备结论，本轮不附加性能采样。

`metadata_h131072_b16_v1` / `task_20261017_204256_222892518376`完整PASS，
同一图六次A→B→C→A→B→A，全部根/工作区地址不变，72核代际逐步+1，
输出、TopK、score及五份完整allocation逐bit相同，输入metadata及保护检查
通过。固定S6下的本轮历史/位置/页表/关闭数变化已验证；尚不覆盖mixed-Q、
padding、prefix共享和连续状态演进。本测试恢复相同初始cache，不是连续decode。

下一候选release_sync_v1仅精简阶段发布：先显式PIPE_ALL排空本核所有业务
流水，再全DCache清理/失效，再用gm fence/DSB保证发布完成，最后TNOTIFY。
移除原先前后gm fence叠加notify内部前置PIPE_ALL的重复维护；新ready helper
明确要求调用者已完成业务release。普通notify仍自带PIPE_ALL，begin_invocation
不变；等待端、单写者槽、代际、phase合同与数学均不改。AIC gm fence实际
生成分流水barrier，所以不能只删除原notify前PIPE_ALL；本候选把该完整排空
移到DCCI之前，绝不把分流水fence当作FIX完成。旧源码保存release_sync_v1/before。
尚无本候选功能或性能结论。

release_sync_v1完整符号编译`kernel_release_sync_v1`通过，生成代码核实
AIC/AIV均在DCCI前有PIPE_ALL，DCCI后有gm fence及DSB，之后才TNOTIFY；
未删除信号本身的DCCI/DSB或等待侧维护。提交
`task_20261017_204900_23377666999`（`perf_h131072_b16_release_sync_v1`）。
测试脚本在L1前记录实际代际，再按本次图调用次数检查增量，兼容前面已经
做过metadata回放，不再假定只有最初一次eager。阶段提示区分首次编译/调用
与返回后的readback比较，避免把首次CCEC耗时误当作设备停滞。

### 2026-10-17：release_sync_v1 完成，允许进一步按工作量细分策略

`task_20261017_204900_23377666999` 完整 PASS，包括固定 S6 下六次同址 metadata 图重放。
B16/128K 五窗原分 task PTO 中位 871.80 μs，单 SPMD 949.32 μs（慢 8.89%）。
此前页并行版本为 872.52→932.54 μs；不同轮数据不能精确归因，但此候选未显示收益，
不采用、不扩档。已将两个同步文件恢复为 release_sync_v1/before，候选保存在 not_selected，
不改变数学、等待协议或精度门槛。

用户进一步明确可以增加档位，只要对性能综合有利。七个规格仍是验收点，
不限制内部策略数量；可以按历史长度、query 数、总 token、分核波次数等工作量细分区间，
每个新增分支须有完整 CSA 收益证据，档内仍处理 device 动态范围，不写测试点白名单。
用户随后要求先比较此前静态 shape 七档性能；本轮先回读历史，不新测。

### 2026-10-17：按要求回读历史静态七档，未重测

逐项核对 selected_shape_policies 的七份原始 result.json，用完整 L1 样本重新计算中位数，
静态结果为 128K B4/8/16/24：541.92/641.56/807.16/1121.12 μs；
8K B16/24/32：621.96/836.74/905.72 μs，同轮耗时降低 2.89%～10.82%，均未达 30%。
原表没有后来新增的 128K/B28。正式动态 scale_page_owners_v2 的已测三档为
128K/B16 932.54、128K/B28 1416.62、8K/B32 1183.92 μs。
B16 与 8K/B32 相比历史静态数值分别慢 15.53% 与 30.72%；跨轮比较不作因果消融。
当前同源码版本的其余四档尚无新测结果，禁止用旧动态首版结果混作最新版本。
静态 Q-B 20 核、O-A N128、成员排程/依赖细化，与当前 24 核、分区间 N128/N256、
若干整阶段完成依赖并不相同；动态 shape 不要求这些变化，后续按区间重新验证。
完整表、策略对照与原始路径见 tests/pypto_test/csa_multishape_spmd_20261016/
STATIC_VS_DYNAMIC_SHAPE_HISTORY_20261017.md 和同名小写 JSON。

### 2026-10-17：输出按组就绪流水 output_group_pipeline_v1 开始

前一目标轮产生了有效进展：回读静态七档并识别不同分核/tile/依赖策略，
同步精简候选实测无收益后撤回。继续正式动态入口，不修改静态历史源码。
本候选保留全局logical%worker的每核任务归属和组内K累加/量化顺序，
仅将O_A和量化完成通知细分为8个模型输出组；AIV可在其它组O_A仍运行时
处理已就绪组，本核已完成O_A的AIC可消费对应量化组做O_B。
每组量化包含其INT8尾零。没有业务缓冲复用，没有移除缓存发布/获取。
新增16个组phase使信号区44→60阶段，须新建清零信号区；Tensor根参数仍130个。
OUTPUT_GROUP_PIPELINE是区间策略的constexpr旋钮，保留原整阶段分支用于消融。
测试脚本支持显式oa_cols和流水开关并记录实际策略，未新增环境变量。

正式编排AST模型576组、2011200事件PASS：包含空工作量、混合query、dummy、
T边界、两种Indexer分组、N128/N256及开关两种路径；核对等待无环和尾段
输出唯一覆盖及先生产后消费。该模型不代替设备cache/FFTS/数值验证。
数学循环AST核对四个helper不变，仅改变分组区间与owner首项。
CPU v1在O_A的NZ偏移非负证明失败；显式非负首项后v2越过O_A，
在O_B NZ列偏移证明失败。继续用合法下标本已满足的显式非负约束表达，
不改PyPTO证明器、不将NZ换ND、不放宽正确性合同。完整编译及设备尚无结论。
候选前源码与测试脚本保存dynamic_full_pipeline_20261017/output_group_pipeline_v1/before。

output_group_pipeline CPU v3仍在O_B NZ证明失败。检查Simplify后的真实IR发现
`(worker+W-group_first%W)%W`被化简为`(worker-group_first%W)%W`；
后端有符号余数可能留下负值，不能依赖Python的非负模语义。v1～v3均未上卡。
正式源码改为`worker+((group_first+W-1-worker)//W)*W`，合法worker∈[0,W)时
除法分子始终非负。O_B从非负全局wave起算，只跳过首wave中落在前组的work，
保留原数学循环与owner，不改依赖仓。CPU v4完整PASS（58符号轴、1 MIX、130根Tensor）。
最新helper AST解释78336组任务区间PASS，OA/量化/OB数学体与候选前相同；
最新编排56个代表组合197080事件PASS。原576组合模型也保留，标注早期首项表达。

提交完整候选：128K/B16 task_20261017_211444_281633115739，带固定S6 metadata
六次同址图重放；8K/B32 task_20261017_211444_281660532728。两者通过完整精度
后才各自做五窗成对L1，默认原OA列tile128/256不变，先只观察分组就绪流水。
此时还在首次编译执行，无设备或性能结论；未提交N128大档的混合候选。

output_group_pipeline首轮两档完整PASS，组phase设备快照16组owner检查均PASS。
128K/B16原分task885.90→组流水945.00 μs，慢6.67%；
8K/B32原分task997.20→组流水1156.78 μs，慢16.00%。
B16六次同址metadata图也通过。两者均未达性能目标；与上一轮932.54/1183.92
跨轮相比，B16没有明确改善，B32候选下降约27 μs，尚不当作同轮消融证据。
下一步使用同进程输出策略矩阵：B16仅N128流水开/关；B32为N128/N256×开/关。
共享正式权重和Native根地址，各候选独立工作区/代际，逐候选精确验证和图检查后
再轮换顺序采完整五窗L1；每候选保留准确kernel_config，禁止多编译入口名称混淆。
矩阵属定位实验，不因结果最快就把单点写成正式支持白名单。

同进程策略矩阵任务已提交：B16 task_20261017_212357_298893032124，
B32 task_20261017_212358_29897173879；各使用5窗完整L1，尚无矩阵结论。

等待编译期间补齐首项算术的独立CPU复现：owner_modulo_codegen_v1，
Simplify删除+24后仍lower为arith.remsi，宿主执行原生成Scalar函数体在192项中
出现56项差异，非负整除公式0差异。没有执行坏版本NPU，也没有修改依赖仓。
详细问题、复现步骤、原始IR/C++和修复建议见PYPTO_OWNER_MODULO_SIMPLIFY_REPRO.md。
这是本次遇到的编译语义问题记录，不把宿主算术核对当作设备验证。

同进程矩阵两任务均完整PASS（各候选eager/graph/最终输出、cache/state、保护、代际）。
B16原分task868.22；N128整阶段946.76，N128组流水928.00 μs，组流水相对
当前整阶段分支降低1.98%，但仍比原分task慢6.89%。该消融的整阶段分支共用
新helper数学，不是调用历史932.54 μs轮次的二进制，不能把跨轮结果混为消融。
B32原分task992.86；N256整阶段1184.40，N256组流水1150.10，
N128整阶段1203.74，N128组流水1155.58 μs。组流水相对同N256关闭时降低2.90%。
N128没有优于N256的证据，保留原OA列tile区间，默认采用组通知。完整CSA仍
比分task慢15.84%，目标未达。当前正式输出策略为N128(T≤96)/N256(T>96)，
按组就绪流水；新内部分档旋钮可继续按工作量试验，不固定七点白名单。
下一步只补本版本尚未覆盖的5个验收点：128K B4/B8/B24/B28与8K B16。
已完成同版本B16/B32矩阵不重复。小档空闲owner、大档尾行及全七档性能仍须核实。


### 2026-10-17：输出按组流水完成同源码七档验收

剩余五档任务全部completed(exit=0)，复用已完成B16/B32矩阵的正式选中策略，未重复这两档。
七档源文件去除注释/docstring后的AST均与当前正式包一致；从原始L1记录重算每窗完整incore首尾，
每个单SPMD窗口均为72核、1个task，五窗中位数与报告一致；不计算hash。
七份eager信号快照的16个新组phase owner逐项核对通过；未发布旧全阶段40/41槽。
所有固定S6的输出、TopK/score、cache/state、图、保护与代际均通过。

| 历史 / B | 原分task PTO μs | 动态单SPMD μs | 相对基线 | 历史静态SPMD μs（跨轮） |
|---|---:|---:|---:|---:|
| 128K / B4 | 498.24 | 580.92 | 慢16.59% | 541.92 |
| 128K / B8 | 634.80 | 757.16 | 慢19.28% | 641.56 |
| 128K / B16 | 868.22 | 928.00 | 慢6.89% | 807.16 |
| 128K / B24 | 1171.78 | 1233.66 | 慢5.28% | 1121.12 |
| 128K / B28 | 1303.38 | 1367.82 | 慢4.94% | 未测 |
| 8K / B16 | 653.50 | 836.68 | 慢28.03% | 621.96 |
| 8K / B32 | 992.86 | 1150.10 | 慢15.84% | 905.72 |

主结果：dynamic_full_pipeline_20261017/OUTPUT_GROUPS_SEVEN.md及output_groups_seven.json。
汇总脚本summarize_output_groups.py，原始任务列表output_group_jobs.json。
原静态七档包含8K/B24而没有128K/B28；旧数据不补造。历史静态三窗与本轮动态五窗不是
同轮消融，不能将全部差异归因于dynamic shape。静态所有点也未达30%。

采用组通知流水：B16/B32同进程对照证明相对同tile的整阶段分支有约2%/3%收益，
并完成其余五点正确性与完整L1。保留OA N128(T≤96)/N256(T>96)区间；B32矩阵
未显示N128优于N256。全部性能仍使用测试100tick失败轮询退避hook，正式绑定尚未启用。
当前仍需mixed-query/padding/生命周期/prefix/连续状态轨迹和正式服务接入；不把固定S6
全通过写成完整功能完成。目标保持active，后续继续核查造成静态/动态性能差距的依赖、
分核与流水，不能通过恢复静态shape或放宽精度门槛完成目标。


### 2026-10-17：metadata放置位置是可选优化策略

用户明确：metadata计算并不是必须全部放进单SPMD，不要求为此单列性能报告。
后续可比较复用框架已有metadata、独立前置推导与核内计算，按依赖、复用及完整链路
收益选择；移出时新增执行、通知/等待、Host同步或调度开销仍应进入公平比较，不能
只缩短单kernel统计范围。当前两份已提交的阶段诊断保留为内部定位证据，不扩测。
当前实现仍由一名AIV Scalar作前置公共计划，TopK后另有并行Attention计划；没有
为了追求单kernel形式认定此放置必须保留。下一候选只精简HC前段的行依赖等待。


### 2026-10-17：HC行依赖候选及Attention页检查有效列问题

HC行依赖候选hc_row_dependencies_v1按现有M16/K4/M8分工缩小等待：AIC只等实际
消费的M8输入；AIV归约等对应四个K片；gate/comb等一至两个RAW owner，inv仍是
同核widen产物。未改变数学体、任务归属、发布/获取或代际布局。实际编排AST独立
枚举HC数学消费者，84组合/302852事件PASS；CPU符号编译kernel_hc_row_dependencies_v1
完整PASS，未提供实值shape。正式源码有constexpr开关，但默认false；尚未上卡，
不能预记功能/性能。候选前源码及脚本完整保存在hc_row_dependencies_v1/before。

两份已提交阶段诊断task_20261017_215603_36675094484（128K/B16）及
task_20261017_215603_366766218999（8K/B32）完整精度PASS；原始阶段时戳留在
diag_h*_output_groups_v1。时戳含插桩与发布开销，不能作为性能验收。内部定位显示
短历史Attention plan的other约127～129μs，长历史约12～14μs；前置META分别
约7.04/18.16μs且只有一个AIV参与。按用户要求不另制作metadata性能分表。

检查短历史生成C++有直接边界证据：compressed_pages_all_valid的load/比较只含
live列，但左广播输入是固定[8,1024]，TCOLEXPANDMUL输出及TROWMIN都用1024
有效列。未加载的UB尾列参与归约，可能把全合法页误判为需要逐512候选回退；
不能直接断言全部时差都由此造成。先修左广播输入与归约结果的valid_shape为
[8,live]，保留所有真实页检查和空洞/短表回退。此次暂不启用HC候选，避免混测。
新page_validity_probe直接检查1/2/32/33/65/66/1024/1025列、空洞及短表的0/1
判定，既有完整plan oracle仍检查最终indices/bias/mask。首次编译和设备结果待续。

页检查有效列修正CPU完整符号编译kernel_plan_valid_columns_v1 PASS；生成C++已确认
TCOLEXPANDMUL输入/输出及TROWMIN源Tile均带live有效列，未将1024物理容量当逻辑范围。
该编译同时覆盖HC行依赖开启分支；随后三份设备任务明确关闭HC开关，只检验页检查修正。
提交8K/B16 task_20261017_220556_397382631984（同任务补验新增快速判定probe及已有plan oracle），
8K/B32 task_20261017_220556_3973512880，128K/B16 task_20261017_220556_397349110944
（含六次同址metadata图回放）。全输出通过后各采五窗成对完整L1；尚未记设备PASS。

页检查修正首轮：8K/B32 task_20261017_220556_3973512880完整PASS，五窗原分task
1013.52→单SPMD1037.02μs，慢2.32%；此前组流水版1150.10μs仅作跨轮参考，
不把113.08μs差值写成严格同轮消融。128K/B16 task_20261017_220556_397349110944
完整PASS且六次同址metadata回放通过，878.74→920.36μs，慢4.73%，仍未达目标。

8K/B16 task_20261017_220556_397382631984的完整CSA首次精度/cache/state/保护
已经通过，但新增page_validity_probe在编译时报Tile留在Spmd编排体，无片上内存；
这是测试包装问题，任务整体失败、未采L1，不写成完整PASS。用显式jit.incore
包装原helper及Scalar循环，调用端仍pl.spmd(1)，未改算子数学或依赖仓。
probe_cpu_v2独立符号编译PASS。

补8K/B16时合并HC行依赖的同进程开关对照，避免先完成一轮成对测量又为同档
单独重跑。另对128K/B16做相同单轴对照，所有候选共用页有效列修正、OA策略、
输入权重地址，工作区/信号独立。两个候选都逐项通过eager/图/保护后才采L1。
矩阵不混入OA tile变化，原分task仍为第三条完整基线；HC默认仍关闭等待实测。


### 2026-10-17：HC行依赖同进程对照通过，补齐其余五档

修正后的page_validity_probe直接分类通过：1050列容量下9个合法前缀均返回1，
两种空洞/越界页返回0；64列容量下前5项合法、其余容量不足项返回0，输出填毒
区保持原值。既有异构Q/dummy/尾行/空洞/短页表plan oracle也通过。当前页修正
没有以跳过验证或“所有页一律合法”的办法获得性能。

HC同进程开/关矩阵均完整PASS：
- 128K/B16 task_20261017_221157_415833825443：原分task870.02，关闭934.14，
  开启914.22μs，开关同轮降低2.13%，仍慢于原分task5.08%。
- 8K/B16 task_20261017_221156_41582138021：原分task672.42，关闭738.72，
  开启716.32μs，开关同轮降低3.03%，仍慢于原分task6.53%。

两侧同一页有效列修正、数学、输入权重、OA策略、设备及进程，候选独立信号/工作区。
每侧5窗轮换顺序，完整incore首尾；全部候选输出/TopK/score/cache/state/图/保护/
72核代际均通过。矩阵原始结果在matrix_h*_b16_hc_row_dependencies_v1，默认
HC开关暂保留false，待剩余五档补齐后决定全区间采用。下一轮显式开启HC行依赖，
覆盖128K B4/B8/B24/B28和8K B32；不重复已经完成两档的矩阵。


### 2026-10-17 22:24：HC行依赖＋页有效列修正完成七档，默认采用

其余五个任务均completed(exit=0)，与两档已完成矩阵选中HC开启分支合并。
所有任务的正式源码AST一致（注释及HC已测开关默认值单独记录）；逐窗原始L1重新计算
与报告一致，单SPMD每窗72核且仅1个task。输出组phase owner回读每档16项全部通过。
所有固定S6输出、TopK/score、完整cache/state、图、保护与代际通过。

| 历史 / B | 同轮分task PTO μs | 动态单SPMD μs | 相对基线 | 历史静态 μs（跨轮） |
|---|---:|---:|---:|---:|
| 128K / B4 | 498.82 | 560.76 | 慢12.42% | 541.92 |
| 128K / B8 | 645.20 | 719.46 | 慢11.51% | 641.56 |
| 128K / B16 | 870.02 | 914.22 | 慢5.08% | 807.16 |
| 128K / B24 | 1145.66 | 1199.54 | 慢4.70% | 1121.12 |
| 128K / B28 | 1306.84 | 1356.86 | 慢3.83% | 未测 |
| 8K / B16 | 672.42 | 716.32 | 慢6.53% | 621.96 |
| 8K / B32 | 971.36 | 1031.30 | 慢6.17% | 905.72 |

证据：dynamic_full_pipeline_20261017/HC_ROWS_PLAN_COLUMNS_SEVEN.md、
hc_rows_plan_columns_seven.json及hc_rows_plan_columns_jobs.json。全部原结果、生成物和旧版本保留。
页检查修正与HC行依赖保留；默认HC_ROW_DEPENDENCIES从false改为true，此后默认
七点policy字典与实测开启配置逐项一致。只是选择已测分支，没有修改数学或新增设备调用。
输出OA tile区间、输出按组流水、FFTS和60阶段单写者信号合同均未改变。

历史静态仍更快，不能把跨轮差值全部归因于dynamic轴；新版本仍慢于各自原分task
基线，未有任何30%达标。旧8K/B24已移出当前七点、128K/B28无历史静态实测。
性能继续使用100tick测试编译hook，不能宣称正式binding默认具备这一退避。
固定S6的metadata变化在页修正版HC关闭时通过，HC开启后尚未补最新metadata变化；
mixed-query/padding/生命周期/prefix/连续状态及服务路由仍待验证。后续继续动态
目标，不恢复静态形状版本、不放宽精度、不扩大已通过范围。

### 2026-10-17 22:36：补最新默认策略的同址 padding 图验证

新增 dynamic_full_pipeline_20261017/padding_replay.py，保持完整桶形状与同一组
输入地址，Native builder 更新有效请求数 B→B-1→1→B：L=0、原始/state
slot=-1、页表=0，positions/RoPE 尾部保留旧值。Native compact producer 在
图外沿用满档捕获描述符生成，另按实际请求数生成有效行/写区 oracle；只捕获
完整单 SPMD，不声称验证了 Native producer 图捕获，不采性能或新增正式适配。
逐轮对照原分 task PTO 的有效输出/TopK/score、满档有效前缀、五份完整
cache/state allocation、全部非 slot 写区、metadata、工作区保护和72核代际。
dummy 输出不规定数值，空 rank 与 mixed-Q 仍不在本用例范围。

提交128K/B16 task_20261017_223602_7069429923，目录
replay_h131072_b16_hc_rows_padding_v1。合并已有的六次历史/位置/页表变化重放，
补 HC 行依赖默认开启后的动态 metadata 检查；未重复七档性能。仅测试脚本
语法检查已通过，设备结果待续。本轮没有修改正式算子或依赖仓。

128K/B16 task_20261017_223602_7069429923 完整PASS（exit=0）。最新默认
HC行依赖开启、页检查修正、输出组流水配置下，A→B→C→A→B→A六次历史/
位置/页表图重放，以及16→15→1→16四次padding重放，逐轮输出、TopK/score、
五份完整allocation、非slot写区、metadata、scratch保护和72核代际均通过。
padding基线有效前缀与满档原分task结果精确一致，Native compact有效行的独立
oracle一致。信号保留代际，其它工作区每轮填毒，仍使用同一图/地址/形状和策略。
不扩大为mixed-Q、空rank、连续状态或服务接入验收。随后补8K/B32，覆盖
medium Indexer策略与T192大桶，仅补新动态场景，不重新采性能。

8K/B32 task_20261017_224033_8295157738 完整PASS（exit=0），目录
replay_h8192_b32_hc_rows_padding_v1。G2、双根、N256/OB128策略下，六次
历史/位置/页表同址变化和32→31→1→32四次padding重放全部通过。与长历史
G6、单根、N128/OB96用例共同补齐两种策略代表的本轮新场景；没有重测七档L1。

逐项审计两任务原始result、metadata_replay与padding_replay：检查项全PASS，
同图同址、72核代际、完整allocation及scratch保护均成立；两份实际执行源码
AST均与当前正式包完全一致，不使用hash。汇总机器可读证据保存在
dynamic_full_pipeline_20261017/dynamic_scenarios_v1.json。新脚本、全部原始
结果和执行源码保留；正式算子/依赖无新改动，上一轮性能表仍为最新测量。

本轮属于功能覆盖的实质推进，不是新的性能优化或30%达标证据。固定S6整请求
padding在上述两桶通过，不能扩大为mixed-query、空rank、生命周期/prefix共享、
连续cache/state轨迹或正式服务接入。目标继续active；七档性能均未达标。

### 2026-10-17：复用Attention计划中的请求几何，消除逐候选块Q/L重扫

上一轮两种策略的动态场景验证构成功能覆盖进展。本轮继续性能，候选
query_metadata_v1只复用已经在attention_slot_plan_stage中确认的request与
causal position：写入现有INT32[PLAN_CAP,16] mask的第5/6列，与前五个
候选块标志同一64B行、同一owner、同次phase发布。dummy显式写(-1,0)，
第7..15列及T后的容量保持未写；不新增Tensor、allocation、phase或同步。
新attention_query_geometry在开启分支只读这两项，替代两个AIV在五个块中
反复O(B)扫描；关闭分支仍做原Q/L与causal检查。实际gather的页表列/物理
页保护及完整数学体AST不变。位置从INT64原值无损转INT32，合法性由正
INT32长度与causal关系保证，不从FP32 c_pos回取地址。

正式入口新增ATTENTION_QUERY_METADATA策略项，暂默认false。先做同卡
同进程单轴开关对照，不与HC/OA/tile变化混合。关闭分支的函数封装也随同
验证，不能把跨轮旧性能当作本候选消融；全部性能仍明确使用测试100tick hook。
旧正式源码与测试入口保存在dynamic_full_pipeline_20261017/query_metadata_v1/before。

kernel_query_metadata_v1完整符号编译PASS，无样例shape，58动态轴、130根
Tensor和单MIX保持。CPU源码解释8种几何PASS：同T不同Q、mixed/dummy/tail、
stale position、重叠Q、全dummy、FP32及INT32位置边界；plan数学相同、每元素
唯一写者、候选消费者只读两项plan。首轮测试解释器曾因Python对象id复用把
旧mask标签赋给新数组，清理每组对象标签后修正；未改算子或放宽条件。
phase/owner模型36组合129812事件PASS；以上均不替代设备cache/数值证据。

已提交128K/B16与8K/B32完整开关矩阵，每项逐候选功能/精度/图通过后才采
五窗口完整L1。B16同时补新几何字段的异构Q/dummy/stale/页空洞设备probe，
以及现有历史/页表和padding重放。设备结果待续，不预记收益或开启默认。

两任务均以exit=0结束：128K/B16 task_20261017_225635_12278994621，
8K/B32 task_20261017_225636_122852430692。候选开/关与原动态分task在同卡
同进程、相同根Tensor地址上轮换测量，各五个完整L1窗口；所有输出、TopK/
score、五份cache/state allocation、图、保护区和72核代际通过。B16另通过
四组plan设备probe、六次metadata变化、16→15→1→16的padding重放。

| 历史/B | 原分task μs | 关闭复用 μs | 开启复用 μs | 开启相对关闭 |
|---|---:|---:|---:|---:|
| 128K/B16 | 873.20 | 896.20 | 911.06 | 增加1.66% |
| 8K/B32 | 978.82 | 1036.34 | 1022.52 | 降低1.33% |

结论：减少重复Q/L扫描没有产生一致的完整链路收益；两档变化方向相反，且
开启后仍比原分task慢4.34%/4.46%。默认保持ATTENTION_QUERY_METADATA=false，
不扩七档、不凭两个点设新的阈值；保留动态候选、before/candidate源码和原始
结果。不能仅从完整首尾推断回退由写plan、寄存器压力或同步中哪项造成。
此轮没有修改依赖仓或共享安装。测试仍用100tick编译hook，不作为服务性能。
汇总：dynamic_full_pipeline_20261017/query_metadata_v1/result_summary.json。

### 2026-10-17 晚：阶段提交与S5/S6分档约束

用户要求阶段性详细中文commit，保存中间优化过程。当前补齐一次累计阶段快照：
正式源级动态SPMD、历史静态档案及过程脚本/日志/报告一并记录，采用项与未采用
项明确区分。大快照、完整编译/泳道图产物、权重及二进制仍在本地，提交路径与
大小清单，不做hash校验；不以归档为由重跑既有真机验证。后续每个有结论的
实现/性能阶段及时提交，不等七档30%全部完成。

用户进一步明确区分两类seqlen：KV cache历史长度和batch必须按工作量分档；
本轮query长度偏静态，Dspark=5优先考虑S6，也需要覆盖置信度变化下的S5。
S5/S6可以分别特化或动态共用，按测量决定；T/B、历史长度及实际Q/L/slot/
page table仍动态，不能用固定S反推请求。七档当前成绩只覆盖S6，S5及同轮
不同有效query长度尚未完成整链验证。下一阶段先把固定S6测试驱动参数化，
用S5代表长/短历史策略检查完整功能精度，再讨论是否需要新增query策略。

用户再次指定两个优化参考：既有静态SPMD与独立ops/pypto/hc_pre.py。后续按
相同数学阶段对照分核、依赖、流水和数据复用，记录动态版本新增的成本；
HC_pre重点参考融合、双AIV分工和按工作量选择实现。迁移保留实际Q/L、页表、
尾块及动态循环，不回填冻结调度表，也不直接照搬HC_pre的阈值。跨轮静态
成绩与独立HC_pre成绩只提供方向，正式结论仍由完整CSA同轮对照给出。

### 2026-10-17 23:26：补S5 Native参照，保留S6性能验收口径

上一阶段属于实质进展：query_metadata_v1两档对照完成、默认保持关闭，并
完成详细中文阶段commit 2417166f。此次不重跑七档，先落实用户新增S5范围。
核对发现原分task本身仍有DECODE_SEQ=6：NativeCSACall两处六行闸门，
compact_metadata及两条compressor的token//S，Indexer的pair/整组分工及
Attention请求定位均有S假设。旧query_source工厂还明确拒绝奇数query。
因此不能只把fixture改成S5，再用原分task结果当作正确性oracle。

本轮只参数化测试fixture的query_tokens（默认仍6），Dspark配置上限保留5，
不伪装成num_speculative_tokens=4。默认6代回后，除新增合同检查、docstring
和返回记录字段外，make_fixture算法AST与阶段commit完全相同；证据见
dynamic_full_pipeline_20261017/s5_native_validation_v1/default_s6_source_compatibility.json。
正式动态SPMD无需修改，另建native_query_probe机械绑定真实Native输入，
用真实Native attention半层作功能/精度参照；不测Native性能、不改变原
七档S6的原动态PTO性能基线。保留零容差诊断，功能/图稳定与Native数值
是否逐元素相同分开报告，不把有限值通过自动当作精度验收。

首版两任务因bootstrap自动追加--device而被parser拒绝（exit=2）；补上
设备参数并校验与TASK_DEVICE一致。第二版B16在取Hadamard时误用decode
子metadata（exit=1），改为Native父metadata；均未进入SPMD计算，不是
算子精度失败。失败任务号与原因完整保存到s5_native_validation_v1/jobs.json。
第三版128K/B16 task_20261017_232512_171491921674已取得真实Native参照，
正在编译执行；8K/B32 task_20261017_232631_174021021834并行提交。输出
保存在native_h131072_b16_s5_v3、native_h8192_b32_s5_v3，结果待续。


### 2026-10-17 23:41：S5尾组QK紧凑L0C与TINSERT物理行距不匹配

S5第三版长/短历史均完成eager及两轮同址graph：保护区、72核代际和自身
逐bit重放通过，但不是Native精度PASS。128K/B16输出max_abs=0.1875、
RMSE=0.013906；8K/B32为0.1953125、0.011746。每请求第五行Top-K平均
替换454.19/331.56项，前四行只有约3～6项，说明存在独立的奇数尾组问题。
逐行统计保存在对应结果目录per_query_position.json。

用score_tail_probe直接复用正式indexer_score_cube_stage及Vector helper，
构造同一query/KV、同址live=2→1→2的最小对照。任务
 task_20261017_233532_189926624217，score_tail_v1：768/1024两策略，完整
组2048个score均与CPU精确INT32点积、显式FP16舍入参考逐bit相同；仅把
有效query数变为1，每种策略1792个score错误，Top512根也大幅不同。每个
128列panel只有前16列正确，不是Native前置HC/QR误差或Top-K并列规则。

生成C++中QK的Acc为physical M=128、valid M=64、CompactMode::Normal。
mad按有效M紧凑存放，N方向每16列分形的源步长为64行；实际依赖头文件
.worktrees/simpler-csa-core-args-256-20261016/build/pto-isa/include/pto/npu/a2a3/TInsert.hpp
的TInsertAccToMat仍令srcStride=SrcTileData::Rows，并将固定Rows作为搬运
M传给pto_copy_matrix_cc_to_cbuf，因而第二个分形起读错。TStore已有
CompactMode::Normal分支，TInsert没有对应分支；本轮没有修改依赖头文件。
这定位为当前组合的紧凑Acc→Mat布局不匹配，不宣称所有上游版本都存在。

正式动态score helper内修正：GM仅load live条有效query；在本核L1把缺少
的query槽用本组第一条query填满，再以完整物理M计算QK。系数生产者原已
将无效槽置零，head归约使用完整物理K；补齐整数点积经1/1024和ReLU仍有限，
零系数隔离所有补齐项。Vector仍只写live条query，实际Q/L、组数、页表、
循环和输出均不改，不读下一请求或越界GM。额外搬运/计算只发生尾组，
完整组保留算术顺序。相比经GM存回重载，该办法不增加workspace或跨核阶段。

修正后task_20261017_233838_195151931247，score_tail_v2：两策略共六轮，
score、Top-K根、CPU参考全零差异；同址2→1→2通过。已发长/短S5整链v4、
两档S6对原分task精确回归及G6尾组helper检查，结果待续。未重跑七档性能。
Native零容差是诊断，不是新制定的精度要求；历史性能版本来就与Native
不逐bit一致（VALIDATION_LOG约4870行）。剩余数值差异仍如实保留，不能
因为尾组隔离PASS就宣称整层Native精度或生成token验收完成。


### 2026-10-17 23:49：尾组修正验证收尾，剩余S5精度单独保留

G6 helper task_20261017_234124_2018905507：live=6→1→2→3→4→5→6的
第一query score/Top512根及CPU参考全部零差异；未据此宣称G6整链混合请求
或balanced尾组全覆盖。S6两档task_20261017_233941_197286026429和
 task_20261017_233942_197336924422完整A→B→A对原分task PASS，包含输出、
Top-K、score、cache/state、保护区和代际；无性能采样。

S5 v4两档结构/两轮同址graph/保护区/代际继续PASS；Native输出max_abs从
0.1875/0.1953125降至两档均0.03125。第五行Top-K每行平均集合替换从
454.1875/331.5625降至6.625/3.5625。前四行PTO输出/Top-K及Native参考均
逐bit未变，明确隔离并修复的是尾组布局错误。Native整层仍不精确相同，
冻结输出门槛0.01+0.01*abs(reference)仍有5113/1606个超差元素；不能把
这些差异自动归为允许误差或以helper通过代替token验收。没有放宽阈值。
原性能版已有Native非逐bit差异，但S5剩余差异仍需单独确认，不擅自关闭。

本阶段中文说明、最小复现、失败驱动、before/after源码、任务日志及原始
结果归档于dynamic_full_pipeline_20261017/s5_native_validation_v1及相邻
结果目录。阶段commit只纳入本次源码/文本证据，不提交.pt/编译二进制/权重，
不动其它会话工作区。S6七档最新完整性能仍沿用HC_ROWS_PLAN_COLUMNS_SEVEN，
此轮没有新性能数字、没有任何30%达标；目标保持active。


### 2026-10-17 23:55：以真实S6因果前缀补充S5原PTO数学参照

上一目标轮是实质进展：修复S5尾组QK布局，G2/G6 helper与两档S6完整
精确回归通过，详细中文commit为2d68a00d。S5对Native仍有冻结容差超差，
因此本轮先补比较依据，不直接宣称精度通过或启动新七档性能测试。

新增causal_extension.py：原分task仍真实执行S6，将前五条query的输入、
位置、物理页映射及初始KV与单SPMD的真实S5对齐，第六条分别取X与-X。
先观测S6前五行和共同缓存是否不受第六行影响，再精确比较S5输出、Top-K、
score及所有公共分配字节。只排除S6独有合法写槽，不排除任何S5可写字节；
两边保护区仍各自完整检查，不改DECODE_SEQ、不伪造S6 metadata。

该参照属于因果性质诊断，不自动替代Native或整模型token验收；两边Indexer
策略也可能不同，若出现差异需区分策略排序/舍入与动态实现错误。所有比较
保留零容差和原始差异，不根据预期降低门槛。任务
 task_20261017_235222_220641110995（128K/B16）与
 task_20261017_235222_22062647905（8K/B32）已进入该对照，结果待续。


两档S5因果前缀参照均PASS：第六行X/-X改变不影响S6前五行或共同缓存；
真实S5单SPMD与该前缀的输出、Top-K、score及五份allocation公共字节均
零差异，S5所有合法写入字节完整纳入比较。Native与两侧自身graph检查继续
保留。证明在这两组输入上，剩余Native偏差也由既有原PTO数学产生，不能
归为动态改造额外引入的误差；仍不代表整模型token或所有mixed-query完成。
两任务exit=1来自未改动的Native零容差报告，causal_extension.passed=true。

### 2026-10-18 00:00：HC消费者就地归约候选开始

借鉴独立hc_pre的消费者内归约，新增HC_CONSUMER_REDUCE编译策略，默认
关闭。门控和Sinkhorn直接读四份HC投影partial的所需列，按原左折叠次序
((p0+p1)+p2)+p3相加；删除该路径独立RAW阶段的写回、通知及消费者重读。
不改变投影K分片、RMS、门值或Sinkhorn数学，不增加全卡barrier。每个M8
消费者只等其M16行块四个AIC owner，后续wave的owner周期由48/24核映射
推出；inv仍由同核widen提供。RAW容量暂留作同进程开关消融，true路径
必须保持其填毒值且PHASE_RAW不发布。精度未过不测性能，收益未证实不默认开启。
候选前源码及驱动完整保留在hc_consumer_reduce_v1/before；CPU源依赖模型
和全符号编译已启动，不沿用历史静态HCstream/HCcompact结果代替本次验证。

## 2026-10-18：补充真实 S=1 精度，HC 消费者归约候选继续隔离

用户要求补测 S=1。提交正式 W8A8 权重的 H128K/B4、H128K/B16、H8K/B32，分别覆盖 T=4 尾块、长历史及短历史；Dspark 上限仍为 5，本轮每请求实际 query=1。测试使用提交 `2d68a00d` 的正式动态源码快照，不混入正在做 CPU lowering 的 HC 候选。Native 只作精度参照；同时用原分 task 真实 S6 的第一行作因果前缀参照，并改变后五行检查独立性。检查输出、Top-K、score、全部共同 cache/state、eager/graph、信号代际及保护区。任务及源码清单见 `tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/s1_precision_20261018/`，尚未预记 PASS。

HC 消费者归约 CPU 第一版因 Tile/Tensor 混合算术失败；改为显式 load 后第二版因 Tile 使用 Tensor 下标赋值失败，第三版统一使用 pl.store。均为本仓候选写法问题，未改 PyPTO/Simpler；默认策略仍关闭，尚未设备执行或获得性能收益。

### 2026-10-18：HC 消费者归约两档消融完成，保持默认关闭

完整符号 CPU v3 通过，调度依赖模型 52 个用例通过。真机任务 `task_20261018_000855_250681529778`（128K/B16）与 `task_20261018_000857_250737420046`（8K/B32），开启/关闭候选对原分 task 的输出、Top-K、score、cache/state 逐 bit 一致；eager/graph、代际和所有保护区通过。开启候选后 RAW 保持填毒 37、PHASE_RAW 无发布，证明交接确实删除。

同进程五次完整 L1 中位数：128K/B16 原分 task 881.26 μs、旧 RAW 路径 901.20 μs、消费者归约 910.74 μs（相对旧路径慢 1.06%）；8K/B32 分别 995.18/1039.60/1013.20 μs（相对旧路径快 2.54%）。收益方向相反，默认仍关闭；没有据两个点建立白名单或将局部改善称为 30% 达标。候选、失败 CPU 尝试、原始精度及泳道证据见 `dynamic_full_pipeline_20261017/hc_consumer_reduce_v1/README.md`。长历史退步的阶段原因尚待隔离，不用通知次数推算代替完整计时。

S1 前三例的 eager/graph、保护区通过；原分 task 因果参照首版错误要求整个 state 页表等形。S6 有合法额外尾列（state 的逻辑 block 很小），共同前缀映射仍一致。只修测试参照为逐项核对 S1 全部可访问列，保留 S6 额外列；重新提交三例。额外 H131075/B16/S1 验证本轮恰好完成 ratio4 压缩，原分 task 因果前缀全部精确通过。Native 输出仍有冻结门槛超差，不能据原PTO等价直接宣称 Native 精度或整网 token 通过。

### 2026-10-18：S1 四例完成，精度结论严格区分参照

最终四例 H131072/B4、H131072/B16、H8192/B32、H131075/B16 均对原分 task 真实 S6 因果前缀逐 bit 一致：输出、Top-K、score、全部共同 cache/state 字节，额外五行 X/-X 均通过；两次同址 graph、保护区和信号代际全部通过。测试使用正式动态提交 `2d68a00d` 快照，未混入 HC 消费者归约候选。

对 Native 输出 max_abs 分别 0.0234375、0.03125、0.03125、0.03125，冻结 atol=0.01/rtol=0.01 仍有 134、1150、235、945 个超差元素。没有非有限值；这些差异也由既有原 PTO 数学产生，但 Native 精度/整网 token 仍未验收。没有放宽阈值、没有把局部参照通过改写成整体 PASS。完整任务、首轮页表宽度检查失败、后续修正、原始报告、中文复现及本地大产物清单见 `dynamic_full_pipeline_20261017/s1_precision_20261018/README.md`。

### 2026-10-18：用户切换 Ultra 并明确授权多 agent 并行优化

前一阶段提交 `593c5d25` 保存 S1/S5 因果前缀精度及 HC 消费者归约两档正反结果，本轮基于该提交建立三个独立稀疏 worktree。初次新 worktree 继承 bare 仓库配置，sparse-checkout 无法直接运行；只在新建 worktree 设置 core.bare=false 并 read-tree，未改主仓或依赖安装。

三个并行候选分别为：Attention 入口按阶段偏序折叠重复等待；G2/768 Indexer 在同 AIV 内保留半叶 score，避免 GM 写回再读排序；Attention 同 lane 的 mi/li 使用本地三槽，保留 GM 写出及原数学/FFTS/drain。各 agent 仅改自己 worktree，主 agent 负责统一测试入口、证据审核及后续合入。

测试驱动新增 --spmd-source、--policy-json 和 --source-matrix-reference；旧 SPMD 与候选通过独立 Python 包别名在同一进程导入，绑定相同 Native 根 Tensor，各自独占工作区/信号，不切共享安装。冻结基线为593c5d25，原分 task 仍是目标验收分母；三方轮换五个完整 L1 窗口，先精确核对输出/Top-K/score/cache/state和图/保护区/代际，再看性能。CPU可指定 --controls-json，但仍不传入任何shape样例，不将符号编译成功当作设备精度或性能通过。未新增生产默认开关或预记收益。索引见 dynamic_full_pipeline_20261017/parallel_20261018/manifest.json。

### 2026-10-18：三路并行候选首轮收口，保留正反结果

Attention入口传递依赖折叠：144种偏序场景、长/中历史无样例shape符号编译、
两档A→B→A完整精确对照、三方graph、保护区与72核代际均通过。五窗口完整
L1：128K/B16原分task882.04、旧SPMD912.02、候选915.70μs；8K/B32分别
986.30/1014.36/1011.18μs。相对旧SPMD分别慢0.40%、快0.31%，收益方向
不一致且幅度小，不合入正式源码、不扩七档。删除数千次信号槽读取不等于
已经获得完整延迟收益；本轮没有用插桩把抵消原因硬归到某个硬件因素。

Attention mi/li同lane三槽：保留全部数学、GM写回及FFTS，33825种mask模型
通过；两个[3,32]环与一个[6,32]交错环都符号解析成功，但完整Vec规划分别
255232/255104 B，超过188416 B。本轮有界停止，未上卡。逻辑768 B的环
MemRef没有被重复分配，不能说它自身扩大到255KB；是加入长期存活对象后
整体规划超限。两次失败源码/日志与只读HC广播融合建议均留档，无依赖修改。

Indexer G2/768候选把半叶score从同AIV的GM scratch往返改为33792 B本地
缓冲；原六类Top-K排序/并列次序不变，每query visibility移至半叶排序前。
CPU前四版依次修正Tile形参注解、重名valid_shape、跨分支SSA和A3 UB窄列
assemble，第五版完整通过；第六版用策略常量表达式适配未启用分支的前端
类型检查，启用区间两角色C++去注释/空白后完全相同，没有重测同档。
150轮同址helper图重放覆盖六个半叶分支、随机/全同分及live2→1→2，pair
arena和Cube transfer逐字节一致，cache只读、候选GM scratch不写均通过。

Indexer两档整链/图/保护/代际均PASS；五窗口完整L1：8K/B16原分task675.50、
旧SPMD712.38、候选691.98μs（相对旧改善2.86%）；8K/B32为991.12/1020.64/
1025.04μs（慢0.43%）。不将B16改善推广到整个策略区间，也不建立孤立B16
或T96白名单。补充G6/768和G2/1024 CPU虽通过，但无用缓存声明改变了活跃
Tile地址；G6/768的UB末端163872→166912 B，G2/1024末端仍为163872 B，
不能宣称未启用路径生成物不变。候选
暂不合入正式代码，保留失败、局部改善和下一步定位依据，不做无收益重测。

主工作区新增隔离源码加载器及同进程三方比较入口，四个设备矩阵已验证
该入口；旧SPMD与候选绑定相同Native根Tensor，各有工作区/信号。全部新
结果仍以原动态分task为30%目标分母，无一达标；Native精度、整网token、
完整mixedQ/空rank/连续状态演进没有由这些对照自动验收。最新完整七档
仍沿用HC_ROWS_PLAN_COLUMNS_SEVEN。本轮证据集中在
`dynamic_full_pipeline_20261017/parallel_20261018/README.md`，源码、失败日志、
原始窗口和结果入Git，大编译产物及二进制本地保留路径/大小清单，不做hash。

后续并行工作：一条只读构造两wave请求交接的依赖/核资源模型，须计入同核
串行执行与原FFTS排空，不能假定有免费闲核；另一条在新独立worktree实现
有限容量Vector物理行plan，替代整页预扫描并发布私有物理indices，保留
外部TopK、bias、双AIV搬运与三槽事件。不把Scalar512循环简单挪给更少
plan owner，也不把历史直接L1/UB缓存失败误当成本方案实测。后续结果另记。

两wave只读模型已收口：在显式切开query范围、wave发布和同组FFTS排空后，
301节点/882边及72核资源检查无环，可实现跨组重叠。但当前B16固定owner
均衡案例每组均承担22个Score块+20个Attention块；零交接成本下新旧链长
同为22s+20a，没有据此启动设备测试。若不切开helper范围，则q48引出明确
的Attention0→Plan1→TopK1→Indexer1→Attention0环。模型只否决这一固定
owner、三角色整段排空方案，不泛化为任何细粒度交错/重新分核都无收益；
源码、假设、反例与结果归档于本轮parallel_20261018/wave_audit。

### 2026-10-18：Attention物理行plan完成，同进程改善0.81%，未默认采用

独立worktree的候选在既有plan内用Vector把512个逻辑候选映射为私有INT32
物理行。生产和消费共用同一容量谓词：压缩页表列1..2048、完整物理页数
1..2^24；其余回退原逻辑。安全夹紧只用于访问，未夹紧值仍控制合法性；
最终地址用INT32位移和加法，不经FP32。外部TopK、bias/mask、原SWA/geometry、
双AIV分工、三槽FFTS及全部归约数学不变，无新Tensor或phase。

完整无样例shape的符号编译通过，58动态轴、130根Tensor；132轮CPU解释器
oracle与15轮真实A3地址probe通过，覆盖页表容量回退、整型边界、非法页、
odd/mixed/dummy/stale和A→B→A。另一agent独立源码审阅未发现阻塞问题；
审阅不能代替真实设备的写回/获取语义验证。

任务task_20261018_010403_360472313010完成exit0。128K/B16/S6整层对原PTO
输出/TopK/score/五份cache-state逐bit一致，图、页表/历史重放、保护区及72核
协议均通过。同进程三方五窗口完整L1中位：原分task880.32、旧SPMD904.78、
候选897.44μs；相对旧降低0.811%，相对原分task仍慢1.945%。没有达到30%，
不将局部指令减少称为整层明显收益；本轮保持隔离候选默认关闭，不扩七档。
早先task_20261018_005742_348545224475在编译期主动取消以补设备probe，
无数值/性能结论，不计为算子错误。源码、失败/取消记录、完整证据和路径大小
归档至parallel_20261018/physical_plan_evidence与physical_plan_h131072_b16_v2。
本轮未执行整层S1/S5/mixedQ，不把plan局部probe提升为这些场景的通过结论。

三路首轮已以详细中文提交7dc60748保存，正式ops源码未变。新的有界诊断
将补当前593基线的阶段时钟与Indexer B32候选对照；旧阶段记录早于HC行依赖，
且TWAIT计时未包含FFTS，不能把other直接认作纯计算或据旧差值挑新候选。

### 2026-10-18：补齐当前阶段诊断，区分Score供数与Attention推进瓶颈

任务task_20261018_011813_39503015186完成当前593基线128K/B16/S6的5次
完整数值/图/保护/代际检查；task_20261018_011739_393773027095完成8K/B32
旧版和半叶score候选共10次检查。两项只诊断阶段时钟，不重复未插桩性能测试。
原16..19列继续记录notify及GM TWAIT，20/21列在既有wait_flag_dev前后计时，
没有新增fence。插桩影响调度，不能用这些区间替代完整L1或直接相减预测收益。

长档三轮Score AIC的FFTS区间中位16.05/16.60/16.15μs，AIV为107.90/
106.69/107.63μs；Attention AIC为119.71/108.31/112.51μs，AIV为18.35/
14.51/14.99μs。二者等待方向相反：优先核对Score的AIC供数链与Attention
的AIV推进链，而非笼统归为Vector排序或Cube矩阵乘。剩余other仍含DMA、
Scalar和未单独计量的同步，不能称纯计算。B32半叶缓存少做的局部工作主要
转成等待Score READY，整体Score窗口没有稳定大幅缩短，保留此前整层无收益结论。

独立核对CANN官方及本地dav_c220路径：A3跨核wait会阻塞后续流水，不能套用
A5指定pipe语义称原位时钟只记录发射。最终机器码次序仍未知：原始.text已
提取，两个官方objdump连同已有A3控制对象都无法反汇编，保存失败证据，
不记ISA通过。详细记录在parallel_20261018/CURRENT_PHASE_DIAGNOSTIC.md，
原始时钟、十次对照和工具局限分别归档current_clock与indexer_phase_evidence。

同次生成文件只读审计确认三处融合降槽：HC Right、Indexer压缩投影Right、
主QB Left由原两地址变一地址，并有对应M→MTE1覆盖依赖。OA/OB/QA/KV及
主压缩仍有真实两槽；Score的48KiB WS和两份8KiB Key也保留。性能提示不是
最终地址证据，不据提示数量扫参。完整定位、旧尝试排除与源码提取见
parallel_20261018/pipeline_depth_audit；仅提出compact K512 pair保留真实
交错活跃期的有界修复，尚未记设备或性能结论。

本会话另做compact生产优先候选：AIC在主QB后先完成compact Hadamard再做
独立主压缩；AIV在系数后先提交Indexer量化/scale，再做主Query/主压缩。
两块原等待/数学/发布完整移动，Score的双槽及双AIV ACK未改。顶层AST语句
多重集合、144组调度模型及58动态轴/130 roots完整CPU编译通过；独立审阅
确认所有AIC先发布QB/MAIN_PROJECT再等Score ACK，不形成新增等待环。
指定前段与Score的Left/Right地址未因重排改变；CPU输出投影策略不同，
不声称整文件分配等价。设备任务task_20261018_013723_19521519782已提交，
在其完成前不预记PASS或收益。各候选仍隔离，正式默认源码未改变。

compact优先长B16任务已完成exit0，完整输出/TopK/score/cache-state与原PTO
一致，A→B→A、同址图/动态历史页表、保护区与72核代际均通过。同进程五窗
完整L1中位：原task861.48、旧SPMD913.06、候选888.38μs；相对旧降低2.703%
（24.68μs），相对原task仍慢3.123%，30%目标未达。已保存全五窗，不择最小值，
不与物理plan的跨任务数字拼接收益。8K/B32另一负载代表任务
task_20261018_014358_34138622015已提交，尚不把单档收益推广到整个区间。
compact_priority_evidence保存源码、生成器、模型、CPU/独立审阅、完整日志、
结果与本地大产物路径大小；正式默认及完整七档表尚未更新。

### 2026-10-18：Indexer compact投影显式L0双槽候选

根据同次原task/融合生成物的实际降槽，另一隔离worktree仅修改Indexer
压缩投影helper。每两个K512先装入两份KV Right，保留其交错活跃期，再按
KV0/gate0/KV1/gate1原顺序做M16/N32/K512 MMA；K0..7累加、pair0初始化、
动态有效M及原pad写回不变。只改源码内搬运时刻，不改依赖仓/物理地址/同步。

首轮58动态轴完整CPU通过；156例实际helper AST分别比较load、MMA、store
三类列表，每类内部操作数、shape、valid、init和顺序一致，不比较三类混合
时序，亦不模拟浮点。实际Right为0/32768/0/0，KV1搬运先于首MMA，原gate0
到KV1搬运的覆盖等待消失；必要gate1复用依赖及PIPE_M仍在。全函数分配
多重集仅一份Right地址0→32768，其他helper地址/形状/数量完全一致，各
空间地址上界未增加。CPU证明不当作性能收益。

源码已冻结；独立长B16同进程三方五窗task_20261018_014826_4729059881
已提交，暂不记设备或精度通过，不叠加compact优先排序。证据/补丁/源快照/
CPU日志/156例/生成片段和本地路径大小在parallel_20261018/compact_l0_evidence。

并行只读路线也已收口。Key由双AIV整理的方案在15组/1793状态有限模型中，
私有scale双槽无覆写/死锁，共享scale在单leaf三步和跨两个单步leaf均有
快lane覆盖慢lane待读数据的反例。GM方案为长B16额外增加约132MiB Key
交接字节，若scale也放私有GM则另增4.125MiB交通；不是直接声称这些字节
全部到HBM或等于固定耗时。当前A3的aic_gather/TPUSH/TPOP仍经GM FIFO，
没有找到消除此往返的可用片上直传；A5和GridPipe mock不能混用。自动FIFO
还涉及全函数L1保留底座、工作区及手工FFTS事件避让，不直接推进GM实现。
旧静态Key提交11.81μs/704次探针仅作历史限制，不当作当前动态纯计算分解。
设计、反例、资源模型和分层源码依据见key_prefetch_design/key_transport_audit。

Attention审计仅使用现有current_clock同作业产物：106项Tile操作序列
同序，额外请求扫描/guard由已有query_metadata和physical plan试验覆盖。
融合三处新增PIPE_ALL在阶段初始化；原task逐tick末PIPE_ALL在融合对应
六条MTE2→V局部wait，不能按静态数量断言更慢。整层AIV bin151012B与
独立Attention10548B不是同范围对照；Attention PTO去loc仅增10.25%，
没有独立机器码边界或icache证据。本轮无充分依据的新方向，不再编译/上卡。
attention_readonly_audit保存七份文本摘录、提取脚本、结果与明确限制。

### 2026-10-18：第二轮回收，阶段前移进入正式源，显式L0不采用

上述第二轮证据已详细中文提交ce7c055d，未push。随后两项待测均终态exit0。
compact前移8K/B32 task_20261018_014358_34138622015全部PASS，同进程五窗
完整L1原task998.48/旧SPMD1024.52/候选1001.56μs，相对旧降低2.241%
（22.96μs），相对原task仍慢0.308%。结合已完成长B16改善2.703%，主会话
只迁回cube_schedule.py/vector_schedule.py两份动态编排；正式33文件与
验证快照逐字相同，没有静态shape、单点白名单或数学变更。只补本版本尚未
覆盖的128K B4/B8/B24/B28及8K B16，两份已有矩阵不重测；未预记其余点通过。

独立L0双槽task_20261018_014826_4729059881同样完整数值/图/保护/代际
通过，但原task881.60/旧SPMD910.02/候选919.58μs，慢1.051%（9.56μs）。
五窗均慢于旧版；不合入正式、不扩档、不与另一次阶段前移相加。提前装载、
pair循环与剩余事件均可能有成本，本轮没有分项证据，不能强行归因为一条
指令。保留真实两槽成立却整层回退的完整源码/编译/设备反例。

### 2026-10-18：compact生产前移七档收齐，正式采用，两项收益范围分开记录

其余五档任务全部completed(exit=0)，未重提已完成的128K/B16与8K/B32。
当前正式33份Python源与已验证冻结候选逐字一致；生产代码仅移动两份编排
中的完整阶段，全部数学helper、动态轴、device metadata、核数及信号/FFTS
合同不变，没有把七个验证点固化成白名单。

| 规格 S6 | 原动态分task μs | 593旧SPMD μs | 当前正式SPMD μs | 相对旧SPMD降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 513.88 | 559.82 | 531.92 | 4.984% |
| 128K/B8 | 645.16 | 712.98 | 684.44 | 4.003% |
| 128K/B16 | 861.48 | 913.06 | 888.38 | 2.703% |
| 128K/B24 | 1155.24 | 1193.86 | 1163.08 | 2.578% |
| 128K/B28 | 1281.36 | 1357.70 | 1332.30 | 1.871% |
| 8K/B16 | 668.74 | 712.50 | 700.14 | 1.735% |
| 8K/B32 | 998.48 | 1024.52 | 1001.56 | 2.241% |

每行三方同进程交替五窗，完整L1包含内部等待，均使用正式第4层W8A8权重；
两份SPMD使用相同100tick失败轮询退避测试hook。七档相对原PTO的输出/
TopK/score/cache-state、三代A→B→A、同址图、动态历史及页表、未写区域、
72核代际和末窗检查均通过。七档仍全慢于各自同轮原分task；相对原task
降低30%的门槛为0/7。不能把上述1.73%～4.98%改写成目标达标，亦不将
其它候选的跨轮数字相加。Native attention、token/整模型、本版本S1/S5及
全部动态边界不在此次验收范围。

完整105窗引用、原始样本、七个唯一任务、每档策略与边界说明归档于
`parallel_20261018/compact_priority_evidence/SEVEN_CASES.md/json`；五个补测
的提交/收取脚本和日志在同目录remaining_five。大编译产物与binary本地保留，
LOCAL_ARTIFACTS_REMAINING.json仅登记路径/大小，没有hash或复制大文件。

本轮并行只读审计也考察“只有真实compact Had owner在QB前做Had”。
13例、26种阶段偏序无环，但七档只有2～4名Had owner，它们同时承担较重
QB，B28还承受最多Score步数。旧时钟显示QB原起点距IDX_NORM就绪仍差
13～20μs；后续仍有COEFF、全scale与双槽背压，不能把Had提前等同于
整层提前。没有新收益依据，不生成、不编译、不执行这个变体。模型、反例、
源级owner枚举与时钟局限保存在`parallel_20261018/compact_owner_priority_analysis/`。

### 2026-10-18：有效版本已提交，继续分开审查分片平衡与空发布

上一节生产变更及七档证据已以详细中文提交`d56c9eb6`，本轮未push。
后续同进程消融参照冻结于`parallel_20261018/adopted_source_d56c9eb6/`，
33份动态源直接复制，不与旧593版本混算新增收益。

O_B逐组通知→八组反量化的只读审计已完成。不同于现有OA→quant→OB组流水，
行外层仅第一行能提前前七组归约；T≥48时获取1152→3072次，另增64次
AIC组发布。组外层需要每AIV16～64KiB常驻FP32累加器，或额外5.25～42MiB
GM逻辑载荷。保留g=0..7左折叠、投影BF16边界与单token门值是硬约束。
旧长B16三轮OB末端到OUT约25～26μs，包含不能提前的HCpost，不能全部计为
流式收益；同轮原task末段成员约21μs也不是可直接消除的TMR调度开销。
没有充分新收益依据，本轮不编译或上卡。257例有限所有权/组次序检查、
现有UB生成文本、时钟与限制在`parallel_20261018/ob_hcpost_stream_audit/`。

正在隔离准备的最小候选只对AIV RAW/GATES/NORM/COMB四处完成发布加有效
消费者条件。INPUT与planner合同、72核begin、原wait/math及真实producer
的fence/cache发布全部保留；不把“无跨核写竞争”误作无需维护缓存。RAW
数学按P16容量执行，若容量大于最小pad16(T)，仍可能计算未被真实T使用的
尾行，因此严格称“无有效消费者的通知”，不声称一律无数学工作。跨代
不清零未发布槽，必须由当前代际及消费者owner集合保证不误用旧值。当前
没有此候选的设备精度或性能结论，不因减少通知次数预记收益。

### 2026-10-18：HC有效消费者通知首轮两档通过，另两项并行候选分开验证

四处guard的完整动态CPU首轮通过：58动态轴、130根Tensor、一个MIX；
不提供sample shapes。生成C++中RAW/GATES/NORM/COMB的notify及其原有
PIPE_ALL、DSB和DCache维护均处于有效owner条件内；原数学和所有wait不变。
仅过滤无有效消费者的通知，RAW仍按既有P16容量做数学，不截断业务计算。

以下为各自同进程交替五窗完整L1中位数，单位μs；d56已包含compact前移，
不再与未含该项优化的593版本混算：

| 场景 | 原分task | d56 SPMD | HC通知候选 | 对d56降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 490.16 | 549.46 | 543.32 | 1.117% |
| 128K/B16 | 863.36 | 884.44 | 876.40 | 0.909% |

任务分别为`task_20261018_023555_148822030471`、
`task_20261018_023556_14883229443`。两例原完整数值/TopK/score/cache-state、
A→B→A、图、历史/页表重放、保护区、72核代际通过，各自五个配对均更快。
额外从原始signal快照核对四阶段发布集合及未发布槽不清零，分别少181/150次
通知，也全部通过；不是另外一轮设备执行。没有Native/token验收结论，
相对原分task仍更慢，30%目标未达。短历史B32已提交
`task_20261018_024231_16810916793`，用于另一策略和更大T；尚未收齐，正式源
此时仍保持d56。可复现脚本和原始五窗见
`parallel_20261018/hc_active_notifications_evidence/DEVICE_RESULTS.md/json`。

Indexer G4/G6与叶分配的有限审查已归档到`indexer_query_group_audit/`：
六叶方案增加根/排空成本并在混合长度案例回退；保留五叶只改owner的方案
最重核41→40，但部分核叶数6→8，根/drain更集中。该数字仅是整数工作量，
没有微秒或性能收益结论；本轮不实施、不上卡，不重复历史G4/G6试验。

两条新实现线保持隔离：一条只恢复主Q_B的Left双槽，首版完整CPU通过但
使原Mat搬运窗口退化并增加前置等待，因此不上卡；只允许一次保留原192KiB
Mat生命周期的有限修正。另一条只删无人等待的AIC SCORE_CUBE/ATTN_CUBE
通知，保留Score每叶ACK排空与后继plan/query、ATTN_VECTOR数据获取；
GATES/COMB/两COMMIT的本核或跨阶段业务可见性不随之删除。两线均独立从
d56出发，不叠加HC guard，CPU和设备结论随后单独记录。

### 2026-10-18：HC通知裁剪三个代表点完成并迁入正式动态源

短历史B32任务`task_20261018_024231_16810916793`完整PASS，五窗中位数为
原task987.00、d56为1000.44、候选996.90μs，相对d56降低0.354%。五个配对
差值均为正；另两例的1.117%/0.909%仍以各自同进程数据为准。三例合计
15个配对方向一致，但没有据此宣称统计显著性或整个策略区间性能验收。
短B32四阶段实际发布集合亦通过，每次减少108条无消费者通知；原结果不改。

已将验证候选`vector_schedule.py`迁入正式动态包，仅四处发布条件变化，
33份源与验证副本直接内容比对一致，无hash；72核代际、INPUT/META、原
数学/等待和有效生产者缓存维护都保持。原始档位是回归代表，不构造白名单。
尚未补齐本项其余四个验收点，完整七档最新数据仍是前节d56版；S1/S5、
Native/token及整层mixed-Q的既有未验收范围也未因本项通过而解除。

并行QB修正版cpu_v2通过：中间两份QR Mat与两份weight Mat保持原192KiB
活跃窗口，Left0/32768、Right单0；MTE1→M就绪事件位于Right0之后、Left1
之前，MMA0不等待下一块Left完成。排除QB后的915组allocation完全相同。
v1的Mat退化与前置等待完整保留，不上卡；v2设备任务为
`task_20261018_024836_184487321209`，尚不预记收益。

独立AIC通知候选完整CPU也通过：2365条物理allocation按顺序一致，FFTS
和ACK排空、本核事件、后继获取均保持，AIV C++去注释后全文相同。
任务`task_20261018_024740_181758629231`只验证删除SCORE_CUBE/ATTN_CUBE
两处发布，未叠加HC guard；设备精度/性能以最终结果为准。

### 2026-10-18：独立QB与AIC通知结果收齐，只组合已获正收益的通知改动

QB Left双槽v2任务`task_20261018_024836_184487321209`完整PASS，但同轮
原task858.64、d56为870.98、候选876.70μs；五个配对全部更慢，候选比d56
慢0.657%。不采用、不扩档、不与其它候选组合。v1的L1退化和v2恢复原Mat
窗口/双Left/事件位置均保留，后者只证明代码生成符合预期，不证明性能收益。
也不能只凭完整L1把回退归因于某一片上缓冲或事件。详情见
`parallel_20261018/qb_left_doublebuffer_evidence/DEVICE_RESULTS.md/json`。

两处AIC orphan任务`task_20261018_024740_181758629231`完整PASS，同轮
原task867.98、d56为885.90、候选875.34μs，相对d56降低1.192%；五个
配对均更快。额外只读首次清零后A→B→A三次signal，两个被删除阶段的
全部槽仍为0，72核代际正常。尚不能凭单B16结果采用整个策略区间。
详情见`parallel_20261018/aic_orphan_notifications_evidence/DEVICE_RESULTS.md/json`。

随后构造通知组合源：保留已通过的HC四处guard，只加入上述两处AIC删除，
不包括QB或其它数学改动。比较参照改为已采用HC的33文件冻结副本，故新的
增量只衡量两个AIC通知，不把独立作业收益相加。没有另做一遍独立CPU，
组合的完整动态编译与设备验证直接在三个队列任务中执行：长B4
`task_20261018_025544_19883426214`、长B16
`task_20261018_025545_198838227253`、短B32
`task_20261018_025545_198842116479`。原三方五窗/完整检查门槛不变，
此时尚未有组合性能结论。命令与源在`parallel_20261018/notification_combo_evidence/`。

### 2026-10-18：通知组合三档完成，保持仅HC guard的正式版本

三个组合任务全部exit0、原完整检查通过。参照是已采用HC四处guard的
冻结源码，因此下表是两个AIC通知在该版本上的增量，不是对d56的总收益：

| 场景 | 原task μs | HC guard μs | HC+AIC通知组合 μs | 相对HC降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 517.00 | 557.90 | 528.62 | 5.248% |
| 128K/B16 | 863.08 | 868.32 | 868.24 | 0.009% |
| 8K/B32 | 988.26 | 990.44 | 996.62 | -0.624% |

长B4的4/5配对改善；长B16方向不一，只有0.08μs中位数差；短B32的4/5
配对回退。不能把独立AIC候选的1.192%与HC收益相加，也不能用B4的改善
代替其它区间。正式源不采用组合、不按三个点建立特化白名单，仅保留三档
独立HC guard均改善的4处条件。QB v2也维持否决；对应测试源都完整保留。

额外从原始设备signal读回核对HC四阶段owner集合与旧槽保持、两个AIC
删除阶段全0；phase0仍为所有72核当前代际。未增加设备轮次或放宽检查，
未验证Native/token。原始45窗、配对值、三任务和源码在
`parallel_20261018/notification_combo_evidence/DEVICE_RESULTS.json`及README。
本阶段正式变更仍只改`vector_schedule.py`，33文件与HC验证副本一致。
本项未补其余四档，完整七档最近一轮仍为d56，30%目标继续未达。

### 2026-10-18：短G2预取与O_A量化原值复用的并行消融均不采用

以正式d253d627为冻结参照，三个独立worktree分别审查Indexer、Attention、
投影阶段；正式33源和共享依赖没有变动。两个新候选各做一次完整符号CPU，
随后各跑两个同进程三方设备矩阵。四任务均完整PASS：相对原分task的输出、
Top-K、score、cache/state精确对照，A→B→A、图、保护区及72核代际保持。
仍使用正式W8A8和原L1口径，SPMD双方测试hook均为100 tick失败轮询退避；
这些检查不扩张为Native/token、S1/S5或整层mixed-Q全部验收。

短G2候选只把现有G2/N128/768的PREFETCH由0改1，TO_L0=true，其它数学、
半叶地址、根顺序不变；短G6及长档均不变。真实生成物核对两Key Right各
16KiB、WS Right32KiB恰好64KiB，六panel地址与复用事件通过，完整片上
上界未增加。CPU脚本medium默认硬设0，因此用显式controls=1编译；设备
矩阵从候选和参照各自加载policy，结果亦核对只有PREFETCH不同。不能把
过去仅改长档、短档作为控制组的试验当成本候选已经测过。

量化候选只令o_projection_groups_quant_stage复用同qt第一次载入的带符号
FP32 Tile，避免为量化再读一次。abs/row_max必须独立存活，实际代码已核对
没有覆盖原值。流水双展开加尾共三处TLOAD由6降3，数学、有效形状和store
计数不变；整个AIV UB末端从163872降到155872 B，其它AIC片上上界保持。
这些代码生成改进不等于完整耗时下降。

| 候选 / 场景 | 原task μs | d253参照 μs | 候选 μs | 相对参照降低 | 更快配对窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G2预取 / 8K B16 | 680.90 | 682.86 | 693.48 | -1.555% | 0/5 |
| G2预取 / 8K B32 | 991.36 | 1002.38 | 999.38 | 0.299% | 4/5 |
| O_A量化复用 / 128K B16 | 876.46 | 877.40 | 877.86 | -0.052% | 2/5 |
| O_A量化复用 / 8K B32 | 982.24 | 989.88 | 997.74 | -0.794% | 0/5 |

两个候选均不合入、不组合、不按单点建立特化，也不追加七档或S1/S5边界
测试。预取的B16五窗全退步，B32只是小幅改善；量化长档基本持平且短档
五窗全退步，尚不能从完整L1把原因强归到某条事件或内存分配。两者离原
分task降低30%的目标均有距离。正式版本继续保持d253，完整七档最新表
仍是d56，不能用本轮不同作业的参照值拼成一次七档新结果。

任务：预取B16 `task_20261018_031515_236275422318`、B32
`task_20261018_031515_236279311548`；量化长B16
`task_20261018_031725_241975130621`、短B32
`task_20261018_031725_241981318600`。完整60窗、实际policy、逐窗配对和
复算脚本分别在parallel_20261018下的short_g2_prefetch_evidence、
oa_quant_reuse_evidence及report_prefetch_quant_round.py。CPU大快照本地
保留，清单只记录路径与大小，不做hash。

只读Attention复核没有找到尚未试验过且有充分收益依据的大额GM往返；
QK/PV已共用同一L1 KV，mi/li本核环及物理地址plan各有既有结果。下一步
仅补当前版本的逐tick gather、softmax、PV归并与原FFTS等待观测，不把
插桩结果用于性能验收，也不改正式数学/同步。只读排重见attention_aiv_d253_audit
和dense_projection_d253_audit，避免后续会话重复已否决方向。

### 2026-10-18：当前版本Attention逐tick诊断完成，聚焦压缩gather

唯一任务`task_20261018_032708_262386210870`完整PASS。冻结源仍为d253数学
和调度（e581只增加上一阶段记录）；128K/B16/S6，五次原PTO精确/保护区/
代际检查和同址图重放通过，A→B→A三轮共4896条逐tick记录。没有Native
forward或token验收，也没有完整L1性能窗口。

测试hook只向各物理核独占signal槽[32:64]写32个word，每核60条上限，
[64]为溢出标记；原通知[0]和既有phase_clock[16:20]互不重叠。独立审查
确认4320条容量槽唯一、真实AIC与AIV0/1索引隔离，插桩剥离后原C++逐字
恢复且原同步序列保持。原阶段drain/DCCI/DSB/发布仍负责记录写回。审计中
raw_cpp_path遗漏/ptoas仅补充路径映射，没有改已运行入口或为此重测。

| 逐核累计中位数 μs | A0 | B | A2 |
| --- | ---: | ---: | ---: |
| AIV gather/ring发射区间 | 68.30 | 66.26 | 67.39 |
| AIV softmax到prob发布发射 | 25.49 | 23.05 | 23.77 |
| AIV PV gate/merge（含PV wait） | 24.42 | 24.39 | 24.58 |
| AIC KV FFTS等待 | 114.12 | 108.97 | 116.65 |

压缩块占逐核gather累计的比例中位数约94.94%–94.97%。单半块中位数：
SWA约0.66–0.68μs，压缩第1块约6.12–6.20，第2块3.82–3.90，第3块
2.64–2.68，第4块2.92–2.98。首压缩块较慢在每核四个query中反复出现，
不只是整个helper的第一次；但没有cache事件，不能直接归因于某级cache。
两lane没有固定慢侧，最晚发布核也不恒等于gather累计最大的核。

同组两lane的gather区间并集与AIC KV等待重合约47.06–51.48μs（逐组累计
中位数），只说明时间关系，不是可直接节省量。gather同时包含清零、几何、
Scalar索引/页表、KV加载与ring写发射；局部wait_flag marker近零只是可能
的Scalar发射开销，不能认定MTE/UB等待近零。tick_end不包含自身记录写入、
下一tick初始化及最终drain；publish_end是最后TEXPANDS发射后，非完成点。
新tick hook没有额外硬件fence，但组合的旧phase_clock仍有其DCCI/DSB。
各区间有嵌套，不能相加冒充纯计算或完整L1，也不用于30%性能验收。

基于该范围，只推进另一份独立候选：证明同代plan已过滤非法列/页后，
删除consumer重复的两层地址guard，保留负索引、原page读取、零KV、SWA、
行序及事件。完整CPU已通过，两档完整矩阵为
`task_20261018_033720_281285324271`（128K/B16）和
`task_20261018_033720_281290823741`（8K/B32），此时未预记性能收益。
“全有效块跳清零”已有旧候选整链通过但未改善的记录；本轮不重开该方向。

报告及完整原始三轮数据：parallel_20261018/attention_tick_clock_evidence/
ANALYSIS.md/json、independent_review，以及attention_tick_h131072_b16_v1。
首次使用的测试脚本、符号CPU、槽审查、计时限制及路径更正均已保存。

## 2026-10-18：Attention地址guard候选结束，无完整L1收益

上节两任务均完成，所有原PTO精确对照、A→B→A图重放、cache/state、保护区
与72核代际检查通过。测量为同进程同输入三方交替五窗中位数：

| 场景 | 原动态分task μs | d253单SPMD μs | guard候选 μs | 候选相对d253降低 |
| --- | ---: | ---: | ---: | ---: |
| 128K / B16 / S6 | 862.28 | 874.12 | 874.28 | -0.018% |
| 8K / B32 / S6 | 983.60 | 995.36 | 996.02 | -0.066% |

两档均3/5配对窗口略快，但中位数无改善，不采用、不扩七档，也不与其他
失败候选叠加。CPU实际少3个比较、1个andi和1个整数除法，片上分配不变；
数学/传输/同步调用没有变化。独立审阅确认在本入口同代plan契约下省略两层
guard有依据，但源码或指令数量减少并未形成可用的整层收益。正式33份源
保持d253；精度PASS不代表Native/token及S1/S5完整动态范围已全部验收。

另只读审计同次作业融合2个和分task73个kernel：现有raw bin/manifest/
compiled_meta/perf_hints均没有寄存器数、stack/spill或local scratch统计，
不据较大二进制推断Scalar溢出或icache瓶颈；未知字段保留null。未重复已有
官方反汇编工具的A3失败路径，没有为该审计新编译或上卡。

证据位于parallel_20261018/attention_guard_elision_evidence/DEVICE_RESULTS.md
及json、INDEPENDENT_REVIEW.md、源码/CPU差分、两档原始三方五窗；资源审计
见scalar_resource_audit/。逐tick诊断的三份约2.4MB原始JSON属于文本证据，
随本阶段保存；大编译产物仅登记路径和大小，不做hash校验。

## 2026-10-18：三路有限复审收口，PV分块仅进入CPU检查

- Attention同轮实际wrapper的64行压缩循环无逐行wait/copy，核心搬运同形；
  首压缩块发生在前一query最终发布前一拍，不能用该发布解释其固定偏慢。
  原task的score在KV UB区外、融合在区内，但没有独立硬件完成计时支持瓶颈
  归因。更正旧文字：原task压缩分支有两次TEXPANDS，融合一次，并非互斥。
- HC三轮现成插桩均COMB完成先于QA完成；平方和/逆RMS临时量已为UB Tile。
  约27μs的OB→OUT包络未分离为可隐藏时间，不重开旧streaming方案。
- Indexer当前五份策略/几何源码与已有审计一致；长低B G6及短G2/G6分配的
  主要变体都有既有证据，未重做旧模型或参数扫测。

上述证据分别位于parallel_20261018/attention_gather_compare、
hc_tail_clock_audit、indexer_interval_reaudit，无新设备测试。
下一独立候选仅将Attention AIC PV的N128四块改为N256两块，K128/64heads/
三槽/FFTS/数学顺序保持。先在独立33源副本完成一次完整符号CPU，检查实际
Right/Acc槽和事件；不能用理论128KiB Acc恰好容纳当作编译或性能通过。

PV首档随后完成：128K/B16原分task870.82μs，d253参考881.40μs，N256候选
873.04μs；相对d253降低0.948%，五个配对窗口全部更快。完整PTO精确对照、
图/状态/保护区及代际通过；相对原分task仍慢约0.255%，不是30%目标达成。
据此只补长B4与短B32两代表，任务分别为task_20261018_040259_32962431633、
task_20261018_040259_329628229360，源冻结不变，正式代码尚未迁移。

PV三档最终均完整PASS：128K/B16的d253→N256为881.40→873.04μs（降低0.948%，
5/5更快）；128K/B4为557.58→534.14μs（降低4.204%，5/5更快）；8K/B32为
998.02→1017.26μs（慢1.928%，0/5更快）。各自原分task分别870.82、505.30、
978.36μs，三个点仍均未达到相对原分task降低30%的目标。
不全局启用，不以B4/B16点名形成白名单；正式源码保持d253。
完整五窗/策略/协议/任务见pv_n256_evidence/DEVICE_RESULTS.md/json。
长/短历史、T、Score策略与输出tile同时变化，不能仅据这三点就把早KV或
任何单一差异定为原因；后续只读实际产物审计另行记录。

PV后续仅补两个可区分区间因素的短历史点，源保持同一冻结N256候选：
8K/B16/T96（七档之一，G2/OA128/OB96/晚KV），
8K/B20/T120（区间内探针，G2/OA256/OB128/晚KV）；既有8K/B32/T192为
G2/OA256/OB128/早KV。二者不用于制造测试点白名单；只有区间实测证据
一致才考虑正式策略。任务与完整命令见pv_n256_evidence/interval_registry.json。

PV实际资源复审未找到短档独有变化：两档Attention外allocation分别2178/2020条
全同，AIV全同，各空间地址上界全同；QK Acc原本就与PV首Acc同址0，跨tick
FIX→M两版均存在。两档都只是PV Acc 2×32→2×64KiB、Right单槽32→64KiB，
以及相同的4→2分块事件缩减。不能据此把T=144直接定成启用边界。
证据见pv_n256_allocation_audit/；最小constexpr接线仅作设计保存在
pv_n256_evidence/POLICY_WIRING_PLAN.md，未改正式实现。

WS移到AIV的额外只读复审见ws_aiv_readonly/README.md：有旧分task的
Vector→Cube WS历史，但没有当前动态G6 exact反向消融；直接迁移让单step
交接24→768KiB，半组384KiB超过单AIV UB，还要与TopK竞争并核对累加顺序。
当前不推进GM直迁，无新增编译/NPU。

两个短历史区间点均完整PASS但性能回退：B16/T96参考688.66→候选693.46μs
（慢0.697%，1/5更快）；B20/T120为808.90→821.04μs（慢1.501%，0/5更快）。
同T96的长/短表现相反，短T120走晚KV也回退，不能按T144或输出tile边界
启用N256。实际编译审计两侧均46个wait站点、100 tick未就绪退避，未发现
把reference漏打hook的证据。报告见pv_n256_evidence/INTERVAL_RESULTS.md/json。

当前只补剩余长历史B8/B24/B28，完整任务清单在long_remaining_registry.json。
另由独立agent在私有源准备PV constexpr接线，默认128，仅将“既有长G6
平衡单根域用256”作为待测策略候选；没有修改正式源码，也不宣称已找到
性能分化的根因。新增参数不得增加runtime根或静态化实际输入形状。


## 2026-10-18：PV N256七档收口，接线保留为历史候选

剩余三档全部完整PASS，按statistics_us角色键读取五窗中位数：
128K/B8 d253 677.70→N256 673.82μs（降低0.573%，3/5更快）；
128K/B24 1186.54→1183.62μs（降低0.246%，3/5）；
128K/B28 1352.18→1355.54μs（慢0.248%，1/5）。
初读grep显示顺序时把三方角色读反，已立即更正；最终汇总脚本只按明确角色键取数，
不按JSON键插入顺序推断候选。原始结果未改动。

完整七档另加短B20均通过原PTO精确对照、图/状态/保护区及72核协议；
长G6域也有回退，短域三点全部回退，故不统一启用、不造B4/B16白名单、
不做长历史边界扩测。没有一个N256目标点达到相对同轮原动态分task降低30%。
详细表和可只读重算脚本分别为pv_n256_evidence/SEVEN_AND_INTERVAL_RESULTS.md/json、
parallel_20261018/report_pv_n256.py。正式33份Python经直接文本逐份核对仍与d253一致。

四文件PV constexpr接线在独立副本完成，N128/N256各一次完整CPU：130根、
58动态轴、单MIX与sync_start保持；同控的完整分配序列和compiled_meta相同，
去掉pto SSA调试注释后整份生成C++逐字相同。它只证明接线无额外运行时代码，
不构成性能采用依据，源码/patch/审计/日志保留历史，主包不迁移。
大编译产物本地保留路径和大小，无hash。资源复审仍不能解释长短分化根因。

下一独立候选检查Query反量化的head分组：在原4/16/64基础上补8/32，
按实际T与48AIV均匀波次减少重复scale/RoPE准备；单独agent交叉核对生产者等待、
尾行数学、head owner与UB，不改矩阵乘、不叠加PV或此前失败优化。


## 2026-10-18：Query反量化按完整波次扩大head分组，三档待实测

独立候选仅改变query_projection_stages.py的分组控制：在完整八行时按
64/32/16/8降序选择能形成48AIV整波的最大head组，其他情况仍4；新加8/32。
等价选择条件为T>=6*h且T%(6*h)==0，不是batch测试点白名单。
T48/144/192分别由4/4/16变8/8/32，每核head数不增，重复scale/cos/sin/
swap准备任务数减半。T96保持16，有任何八行尾块时仍固定4，原尾块
recip(sqrt)与完整块rsqrt数学分别保持，不改QB矩阵乘、输入或工作区。

私有源owner模型枚举T1..512：每token/head唯一写、逐核head及token-head
负载均不增加，数学主体和其它函数AST一致。独立复审确认入口已等全部
24个QB owner、QUERY发布/消费覆盖全部48AIV，QR/ROPE owner也完整获取；
因此本候选不改schedule与notify/wait。T49等尾部不得误用8头分组。

一次完整符号CPU通过，130根/58动态轴/单MIX与sync_start不变；DQ完整
分配地址/形状与旧版相同，DQ UB末端131072B，全AIV UB163872B，AIC
Mat/Left/Right/Acc为475136/65536/65536/131072B；全部硬件调用和局部
同步计数不变。CPU不代表设备精度或完整L1收益。

三档冻结源同进程三方五窗任务：
128K/B8 task_20261018_043459_392309626395；
128K/B24 task_20261018_043459_392319716700；
8K/B32 task_20261018_043459_392330520909。
其余四个目标点规则未变，当前不重测。registry与独立review位于
parallel_20261018/query_head_group_evidence；此时未预记PASS或采用。


Q_B另一路只读复审已结束，暂不做M64/N512/K256的CPU候选：列块128→64
并未减少列分工最忙核的6×256=3×512个有效列；T96逻辑输入读取44→70MiB，
其中权重32→64MiB、QR12→6MiB。M/N联合分工在部分大T可降低峰值，但权重
读取仍增加，不能据列块轮数减半宣称收益。K256×N512的Right需128KiB，超过
64KiB必须继续切分；高层Mat可由AutoTile拆分，因此不称其必然编译失败。
报告与可复算模型见parallel_20261018/qb_tile_readonly，无新增CPU或NPU。

另一私有候选检查T<=48的Q_A统一M16块分工：原T24/48仅8个AIC参与QA，
扩核需改global block ordinal与QA消费者等待；只改DENSE_M或ceil无法把T24
第二个尾块分到另一组核。新方案最多24核，T>48保留原路径，正在一次完整CPU。
增加QA参与核可能延迟它们原本可提前执行的Indexer投影，不预估整层净收益。


## 2026-10-18：采用Query完整波次分组，三档完整结果保存

上一节三个任务均完整PASS。按明确角色键读取五窗中位数：
128K/B8 原分task636.22、d253参考677.02、候选674.40μs，降低0.387%，3/5更快；
128K/B24为1171.34、1185.34、1172.80μs，降低1.058%，5/5更快；
8K/B32为981.00、997.08、995.62μs，降低0.146%，3/5更快。
B24收益较一致，另两档幅度接近测量波动，不宣称大幅提速。三档相对原分task
仍未达到降低30%。其它四个目标点规则不变，没有重测或改标旧证据。

已把冻结候选的query_projection_stages.py一份文件迁回正式动态包，其他32份
保持d253；不叠加PV、预取或量化失败候选。源码直接按实际T做均匀波次分组，
没有新增Host值、Tensor/Scalar接口、工作区或同步。证据见query_head_group_evidence
和report_query_head_group.py。完整PTO精确PASS不等于Native/token验收。

最终TopK根的128前缀认证仅做只读评审，不推进：当前mrgsort没有输出限512
接口，exhausted表示任一输入耗尽；不同有效输入长度可用，阈值可从UB直接
取Scalar，但有本地流水等待、R+1读值及失败fallback。R4的128前缀要求各根
恰贡献128且边界无同分；当前R5第五根可仅1项。R6也只是两次原最终merge，
不能把32768候选全排序的成本当作可节约量。未隔离纯merge耗时，收益上界
未知，不为该方向另做CPU/NPU。见final_root_prefix_readonly/README.md。


## 2026-10-18：小T QA M16分工通过CPU，独立两档已启动

私有33份源以d253为参照，仅qkv_stages.py/vector_schedule.py变化：
T<=48将ceil(T/16)个M16块按全局block ordinal分给最多24AIC，N128/K256
保持；尾行读取valid_shape，写入原pad16工作区。QR前的QA owner等待数
同步改为8*ceil(T/16)，不遗漏T24的core8..15；T>48完全保留原路径。
只改DENSE_M或floor→ceil仍会把T24尾块留在原8核，因此没有采用该错误简化。

一次完整CPU通过；源模型枚举T0..96及127/128/129/192/384；独立审阅另核对
T1/15/16/17/24/31/32/33/47/48/49，实际writer与waited集合相同、有效元素
唯一写，原M16尾数学仅局部变量更名，全NORM等待和全24QA发布保持。
已有2365条alloc的地址/shape/有效维度归一表达式不变，新增分支15条。
QA前同步序列相同；QA后4716条同步经局部ID归一相同，紧接投影的三对
M→FIX局部事件ID顺延，不把这描述为额外同步。详细audit另随本候选归档。

正式当前已提交Query head分组9d5327e5，本QA参照/候选刻意保持先前冻结
d253，以做独立消融，不重新编译或把DQ收益混进QA。两档任务：
128K/B4 task_20261018_044522_413498115985；
128K/B8 task_20261018_044522_413502612248。
命令及独立审阅在qa_small_m16_evidence，此时未预记完整验证/性能PASS。
新增QA owners原可早做Indexer投影，且T32权重逻辑读取8→16MiB、T33..48
16→24MiB；更充分用核不能替代完整CSA实测。若采用，再独立验证与DQ组合。


## 2026-10-18：QA M16 第一版两档完成，保留区间改进方向与三项只读结论

`qa_small_m16_evidence/DEVICE_RESULTS.md/json` 保存完整结果。五窗完整L1中位数：
128K/B4 原分task502.44、d253参考543.60、QA v1为523.90 μs，降低3.624%，4/5窗更快；
128K/B8为648.00、679.66、683.58 μs，回退0.577%，仅1/5更快。
两档均通过原PTO精确对照、A→B→A图重放、状态/保护区及72核代际，但都未达到
相对原分task降低30%。参照和候选均不含后来采用的DQ分组，不能与另轮DQ收益相加。

T<=48统一M16不采用。B4原路径的两块M16只改变owner，不增加总权重读取；
B8原M32+M16改成三个M16使权重逻辑读取16→24 MiB，这是结构性区别而非已证明
的唯一硬件归因。v1虽保留大T源码，完整编译仍使原M32/M64局部事件排列改变：
各有8/4组相邻同步序列内部顺序变化，数学语句与同步边界保持，不能声称机器行为不变。
因此另建v2，依据原纯尾块区间16<T<32启用QA M16并行，并使用constexpr编译分支。
区间外必须只生成旧路径，区间内仍以真实T计算块数和尾部，不写B白名单。
v2正在独立CPU核对，不用本节v1结果预记v2性能。

只读复审另归档三项：
- `wqa_cache_policy_readonly`：当前权重cache_policy声明可改变IR，但已有两轮候选的
  实际TLOAD C++不变；当前QA也没有显式L2 hint。因此不能把IR标签当作真实旁路缓存，
  不为单改声明重复编译或上卡；未修改依赖仓。
- `qa_dq_composition_readonly`：QA与DQ修改文件不重叠，但存在QA→QR→QB→DQ依赖及
  Indexer竞争，收益不能相加。v1组合没有生成或上卡；下一候选以正式9d作为冻结参考。
- `dq_rope_flat_index_readonly`：DQ预存八行展平索引在DQ侧等价，可去掉部分索引向量指令；
  但同一rope_swap还被KV完整32行路径按0..63列索引消费，直接改布局会越界。
  原两函数方案否决，不以T24测试会掩盖的问题作为通用实现。尚未做编译或上卡。

正式动态算子保持9d5327e5，未合入上述QA、缓存声明或RoPE候选。大编译产物只记录路径
和大小，源/审计/原始日志/精度与性能报告入Git；不做hash校验。


## 2026-10-18：Query连续分工与QA区间constexpr两路独立上卡

Query候选以正式9d为参考，把完整八行的(row_tile, head)线性工作量均分为
48个floor区间，跨64头行边界再切chunk；T8/T16和含尾行时保留原路径。
这样T144准备144→60次、T168准备336→66次，最大每核head计算量不增加。
只改变query_projection_stages.py，不改变每个head的数学、等待或工作区。
T1..512 owner/数学AST通过；唯一CPU中2370条Tile分配地址/shape均相同，
完整同步序列归一相同，AIC C++相同。长度1的pipeline实际C++可正确跳过
双头主循环、排空预置事件并执行单头尾部；仍需设备边界验证。
报告/source/CPU摘要见query_linear_evidence。任务：
128K/B24 task_20261018_050316_27283926583；
128K/B28 task_20261018_050316_27292720517。

QA v2只在16<T<32开启M16块并行，constexpr False/True各完整CPU一次。
False省略新参数仍兼容，2370条分配和四份实际生成代码去行注释后与9d一致；
True保留动态T的块数/尾行，130根/58动态轴/单MIX不变、QA外分配不变。
实际Host区间T18/T24/T30自动开启，T16/T32/T48关闭；不需要测试强制override。
为排除T18同时进入原G2策略，本轮先测同G6的B4/T24和B5/T30：
128K/B4 task_20261018_050316_2729667380；
128K/B5 task_20261018_050316_27300510000。

两路分别冻结正式9d参考和各自候选，未互相叠加。所有任务采用相同正式权重、
同进程三方五窗、完整L1首尾及100tick失败轮询退避。此处只记录提交，不预记PASS，
不将CPU工作量或编译通过当作性能结论。正式包尚未修改。


## 2026-10-18：连续head分工不合入，QA v2两档正向并补区间下端

两路四个任务全部完成，原PTO精确输出/TopK/分数、图状态/保护区及72核协议PASS。
只按同进程三方角色键汇总五窗完整L1中位数（单位μs）：

| 候选 / H / B | 原动态分task | 9d单SPMD | 候选 | 相对9d降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 连续DQ / 131072 / 24 | 1178.12 | 1174.38 | 1181.24 | -0.584% | 1/5 |
| 连续DQ / 131072 / 28 | 1312.22 | 1337.28 | 1328.84 | 0.631% | 3/5 |
| QA v2 / 131072 / 4 | 515.94 | 548.26 | 540.98 | 1.328% | 4/5 |
| QA v2 / 131072 / 5 | 582.92 | 615.46 | 604.14 | 1.839% | 5/5 |

连续DQ虽减少准备次数，完整性能长B24回退、B28仅小幅正向，因此不合入正式包，
不根据两点人为制造B28白名单。真实T40的H8192/B8/S5长度1流水尾部已单独提交
`task_20261018_050845_41225224225`，仅做功能和因果前缀验证，不做Native性能，
结果保存用于动态候选的边界记录；它不改变本轮不采用的性能判断。

QA v2初始两点较一致，补原策略会选择G2的T18/H131072/B3：
`task_20261018_051020_44336828121`，仍以同轮9d为参考、原分task为目标分母。
此前T24/T30均为G6，不能代表跨Indexer策略后的整层净收益。关闭路径已有同控实际
C++逐字一致证据，不为关闭分支重复全量七档。正式源码仍未迁移QA。

结果可通过`report_query_qa_interval.py`只读重算；QA登记新增B3后须等该任务结束
才重新生成完整报告，此刻已保存的报告只含前两点。四个任务都未满足30%目标。

独立DQ本地索引方案见`dq_local_swap_readonly`：完整块内用arange(512) XOR 1，
不改变共享rope_swap和KV，不新增GM/root；A3 XOR会展开六个向量阶段，并非单指令。
现有mask gather不支持相邻交换，reshape+[1,0]也不会逐行广播；当前TINTERLEAVE
架构guard不含A3。仅据本地API说明，不断言硬件不存在该能力。下一候选基于正式9d，
独立核对源数学和一次CPU的真实资源/同步，不叠加已否决的连续DQ，不提前记收益。


## 2026-10-18：S5单head边界结果与大额省工方向复审

连续DQ的H8192/B8/S5（T40）已结束：真实单head尾流水、两次图重放、
地址/保护区/72核代际通过，functional_passed=True。原PTO S6前五行的
因果对照以额外第六行X和−X各一次，输出/TopK/分数以及所有可比cache/state
字节逐bit一致；只排除额外第六行的合法写入，覆盖全部实际S5写入。
Native逐bit不一致仍保留：输出max_abs=0.0234375、RMSE=0.0031557713，
TopK/cache/state也有差异，原顶层passed=False没有改写。没有性能测量，
也不改变连续DQ不采用的决定。见query_linear_evidence/S5_BOUNDARY_RESULT.md/json。

两项高价值方向只读收口：
- indexer_work_reduction_readonly：有效QK dot没有可直接删除的计算；将稀疏WS
  搬到AIV会把原聚合输出交接扩大32倍，逐bit归约顺序也未证明。既有短尾裁剪、
  WS搬运和TopK前缀方案均已排重。最终根只1–2次merge，exhausted不能保证限512项，
  未找到足以支持新一轮编译的省工候选；不把算术上限当设备收益。
- qb_dq_handoff_readonly：已有长B16三轮逐核诊断中，DQ可进入时全部QB已就绪
  12.72/13.26/14.64μs；QB再拆行/head通知不能提前当前消费者。DQ内24次QB检查
  累计约3.1–3.4μs是已就绪信号读取成本，不是QB生产未结束。QUERY由后继
  SCORE_VECTOR→plan获取链覆盖，之前合并等待已有完整负例，不重复上卡。
  逐列交接不减少GM字节，还新增通知；报告限于现有源码和已观察规格。

另一路局部索引首次CPU明确失败于A3 TXORS仅支持i8/i16，而非PyPTO接口表面
可接收的INT32，未生成有效C++。保留失败记录后只允许一个INT16修正版：
0..511索引范围不会溢出，XOR后转回gather要求的INT32；不修改依赖仓，
额外转换、真实UB与事件仍须以后端输出为准。此时没有设备或性能结果。


## 2026-10-18：最终采用G6两M16尾块区间，保留其它区间旧路径

QA下端G2/B3任务完整PASS，但五窗中位数原分task548.36、9d572.84、候选586.20μs，
回退2.332%，仅1/5窗更快。最终Host条件收窄为 `group==6 and 16<tokens<32`。
根据现有policy，这等价于长历史、query上界至少6、24<=T<32；沿用原G6分界，
不建B4/B5白名单。G6/T24和T30已分别改善1.328%/1.839%，但T/G/负载同时变化，
不宣称G6是唯一硬件原因或所有历史/混合Q均加速。后续范围验证仍需保留。

正式迁移policy/entry/cube_schedule/vector_schedule/qkv_stages五文件。
除Host条件与注释外，实际kernel AST与已测v2相同；关闭分支已有同控CPU实际
C++逐字相同证据，未追加编译或上卡。996个Host描述符组合核对除新开关外原策略
字段不变，开启的56种组合均符合新区间。全部33源与source_selected_g6逐文本一致。
原v1/v2和三档正负结果均保留，详细决策、审计和迁移文本见qa_small_m16_v2_evidence。
正式仍使用9d的DQ整波策略，连续DQ和局部XOR没有合入。

LD_DEV只读复审找到历史完整CSA的0/25tick负例；未找到当前动态+100tick的精确
同轮消融。ABI兼容并不等于值得继续扫描，保留报告，不为该方向新增设备试验。
不把异轮1327.28/1146.62与旧1067.36μs直接计算为同轮因果比例。

本轮采用项只改善一个目标点及一个区间代表点，七档没有任何一档已达到相对
原动态分task降低30%的目标。目标保持进行中，不把过程归档或阶段commit当完成。


## 2026-10-18：DQ本地XOR索引两次CPU收口，不采用

记录见parallel_20261018/dq_local_swap_cpu_evidence。v1使用INT32 TXORS，
虽符号解析保留58轴，A3 PTOAS明确拒绝该dtype，未生成有效C++；失败源与日志保留。
v2只做一个修正：TCI生成0..511的INT16、TXORS后显式转FP32再INT32，全部数值精确。
A3没有直接INT16→INT32 cast实现，不能把前端接受dtype当后端可执行。

v2完整CPU通过130根/58轴/单MIX；TXORS输入67328、临时65536、输出68352，
各1024B互不重叠，最终INT32索引仍为36864/2048B。全部空间地址上界与9d相同，
全AIV UB末端163872B，DQ以外Tile分配序列相同。但同步不是完全不变：局部event
重新分配，DQ发布后一个MTE3→V等待被PIPE_ALL替换，后续也有等待前移。

PTO层少1次TLOAD、3次TCVT、2次TMULS和1次TADD，增加TXORS；不能止于这层
就宣称更快。A3 TCI_b16使用1792B scratch，原b32为768B；固定512的模板展开
模型中Vector intrinsic调用由21到34，另多2个内部PIPE_V。TXORS有5个内部PIPE_V，
外层C++仅少4个PIPE_V。它以新增准备/屏障换少2KiB读取，收益依据不足，停止CPU，
不继续占设备。调用数是源码展开模型，不是机器指令计数或运行时性能。

三套33源、两次日志、CPU结果、必要生成片段与事件差分全部归档；大生成物保留
本地路径/大小，无hash。未修改正式源码/依赖，没有设备精度或性能结果。


## 2026-10-18：跨核搬运与权重复用三路并行复审

正式基线为66882634。当前三路分别审计Q_B跨M行块复用权重、输出投影A的显式
缓冲生命周期，以及A3 Cube/Vector交接；各自在独立worktree工作，未改共享依赖。

已完成的只读结论与证据位于parallel_20261018：
- acc_to_dq_transport_readonly：当前A3 Acc→AIV仍经GM，T96有效INT32交接单程
  12MiB、双向24MiB；小FIFO仅减少驻留容量，不自动省掉一写一读。当前AIV在DQ前
  等待QB之后的compact Hadamard，直接加入有限FIFO消费ACK会形成背压环。
  必须区分前端可表达、后端可执行与硬件直接传输，未提出无证据的直接替换。
- attention_kv_multiconsumer_readonly：同一query唯一归属一组，AIC计算全部64heads，
  双AIV各搬64候选，已有三槽KV复用；并不存在每head/Indexer分片重复搬同query KV。
  跨query物理row重合尚未测量，不能从共享prefix推定节省量，也不能省略READY/ACK。
- oa_pair_explicit_lifetime_readonly：以前OA pair的外层pipeline使两条Right流各申请
  双槽，导致128KiB超过64KiB，或被退为单槽。新设计借鉴HC_pre的手工交错生命周期，
  显式K256 Mat双缓冲共384KiB，任意时刻只保留两Right和两Acc。其loop-carried
  coalesce、事件和真实性能仍待验证，未把容量算术当成功实现。

当前只授权Q_B单项完整CPU lowering及OA独立最小CPU探针，尚无本轮设备或性能结果。
Q_B首选仅B常驻，保留M128与四段K256 INT32累加；QR+B同时常驻的M64备选虽然
进一步减少逻辑QR读取，却增加50% MMA/FIX和Right载荷，因此暂不叠加。该原理在
旧分task中已有先例，本轮核对的是当前动态SPMD双M区间，不能把它宣传为新发现。


## 2026-10-18：O_A显式双缓冲v1失败、v2通过CPU

完整最小探针归档于parallel_20261018/oa_pair_explicit_mat_cpu_evidence。
v1真实分配已达到Mat384KiB、Left/Right各64KiB、Acc128KiB，但MemoryReuse
在Mat循环回边插入三次64KiB Mat→Mat，当前A3 PTOAS拒绝，不存在可执行C++。
没有修改依赖，也没有忽略后端错误；源、命令、失败日志与IR均保留。

v2只改变这个明确的表达边界：外层普通K1024循环，内部四个固定K256数学块
手工交错供数，仅两条Acc跨回边；每条输出仍按原K128 FP32顺序累加。CPU完整
通过，实际Mat仍384KiB、Left/Right各64KiB、Acc128KiB，没有Mat→Mat。C++中
下一组三次TLOAD已位于当前组提取前，当前组仅等待自己的MTE2事件，不提前
等下一组，故生成的依赖保留供数窗口；这不是设备重叠或性能收益的证据。

该探针使用真实八组NZ根布局与动态valid_rows，但仅覆盖每组三对N128，余下
两个singleton没有包含，不能当成完整CSA或正式动态算子。下一步正在独立补齐
完整输出、唯一写者与组通知/等待，尚未修改正式包。

Q_B另一分支的一次完整CPU在Right128KiB>64KiB处失败；还没有本轮NPU任务。
已定位到pipeline区域CanonicalizeIOOrder提前Mat→Right extract，导致两个
K256 Right同时活跃。原路径真实K本来也为256，不是数学K突然变大。失败源与
修正候选仍在独立worktree，后续结果另记，不将当前过程提交误报为性能改善。


## 2026-10-18：两版Q_B通过CPU，独立提交受影响区间代表

B-only v1失败已定位：CanonicalizeIOOrder只在pipeline祖先内将Mat→Right
extract视为load前移，使K256/K512两份64KiB Right在首MMA前同时活跃；
不是first Right存活，也不是原K128改K256。原正式Q_B实际也采用K256。
修正v2去掉该pipeline，把固定四K256数学块写成手工QR双槽供数，显式
Right提取在上一MMA之后；不改依赖。完整CPU通过，B为256KiB、两QR槽
合64KiB，Right单64KiB，动态尾行与原四次INT32累加顺序保持。

独立QR+B版进一步在N列循环外加载QR容量256×1024，valid为实际device T。
QR位于Mat0，B位于262144，两者恰512KiB；没有固定64KiB Mat保留底座，
Left/Right各只用一个槽，Acc保持动态compact。CPU通过130根/58动态轴，
没有新增GM根、phase或依赖安装。相邻B复用的M→MTE1→MTE2链仍保留，
不能宣称整K载入没有启动成本或已与前一MMA重叠。

两版均仅由Host已有shape描述符选择128<T≤256的自然双M区间，实际T、
有效行和M循环在核内读取。关闭支经常量折叠后的源AST与正式相同；这不是
重新编译的关闭二进制逐字等价证明。参考均为正式66882634，不能把失败的
B-only v1当运行基线，也不能把两个候选的不同设备绝对时间直接判优。

已提交三方同进程五窗完整L1及完整精度/图状态验证，尚未收取结论：
- B-only：128K/B24 task_20261018_055057_11658413779；
  8K/B32 task_20261018_055057_116588019925。
- QR+B：128K/B24 task_20261018_055122_11762576461；
  8K/B32 task_20261018_055122_117629623586。

对应冻结源/CPU审阅/命令分别在qb_b_resident_evidence与
qb_qr_b_resident_evidence，报表由report_resident_round.py只读重算。
正式算子仍不变；只有设备验证完成后才讨论采用，不把逻辑字节节省当时间收益。


## 2026-10-18：O_A完整动态接线通过CPU并提交代表档位

独立候选保存在oa_pair_manual_evidence：沿原OA_COLS128/T≤96区间，每组
三对N128与两个singleton；pair归3g+p，single归下一组前两核(3g+3+s)%24。
本组五owner互异，全卡24对+16单块，每核最多三个N128输出，覆盖全部八组。
两端使用同一动态区间谓词，新增wait只等待该五owner并统一一次业务cache维护。
原信号槽/代际/工作区和GM根均不增加。所有本核O_A仍先于O_B的量化等待，
没有把未完成生产放在可能阻塞的消费之后。

完整CPU只使用longG6、OA128/OB96、QA=false，130根/58轴/单MIX保持。
pair内六份Mat总384KiB，在完整地址空间从64KiB起，末端448KiB；全AIC
Mat末端464KiB。Left/Right各64KiB、Acc128KiB，不能把局部384KiB误报为
全函数地址上界。T由device shape传给valid/Acc/store，pair中间无Mat copy。
当前块只等待自己的MTE2事件，下块TLOAD已发出；Right覆写事件继续保留。

代表设备任务各自重新JIT实际policy，不能用上述CPU替代其实际编译和正确性：
128K/B4 task_20261018_055534_12872405208（含已有QA小T策略）；
8K/B16 task_20261018_055534_128729815924（G2）。两任务尚待完成。
候选独立于两版Q_B，没有先合并多个变更再解释收益。


## 2026-10-18：两版Q_B四任务完整PASS，均无整层收益，不采用

正式权重、三方同进程五窗完整L1，单位μs；“参考”为正式668单SPMD。

| 候选 / 历史 / B | 原分task | 参考668 | 候选 | 相对参考降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-only / 131072 / 24 |1170.54|1164.80|1168.00|-0.275%|2/5|
| B-only / 8192 / 32 |979.06|997.16|1012.34|-1.522%|0/5|
| QR+B / 131072 / 24 |1170.90|1185.06|1185.38|-0.027%|2/5|
| QR+B / 8192 / 32 |977.96|993.88|997.86|-0.400%|3/5|

四任务均完成输出/TopK/分数、A→B→A图重放、cache/state/保护区以及72核
代际检查。逻辑B读取64→32MiB，进一步合并QR读取也没有证明整层净收益；
不能由理论带宽节省直接推算时延，也不能把两作业不同设备/时刻的绝对值
当候选间对照。两版都不迁移正式源、不扩测七档、不另建测试点白名单。

完整样本/策略/命令/原始trace与日志见各自evidence，ADOPTION.json明确
不采用，DEVICE_ARTIFACTS.json只为大编译产物与数值快照登记路径及大小。
report_resident_round.py、collect_resident_round.py可只读重算/收取，不启动
新设备任务。QR+B相对同控B-only v2，535项非QB AIC分配中262项Mat换址，
shape/dtype/容量不变；1812项AIV相同。这不是与正式668所有机器码相同的
证据，也不能据换址直接判定退步原因。完整性能已作为采用决策依据。

O_A双输出候选仍在独立设备验证，未把Q_B负结果扩大为所有优化方向均不可行。


## 2026-10-18：O_A双输出完整四档收口，区间整体暂不采用

四档均通过原分task PTO的精确输出/TopK/分数、A→B→A图、cache/state、
保护区与72核代际；性能为三方同进程五窗完整L1，单位μs。

| 历史 / B / S | 原分task | 正式668 | O_A pair | 相对668降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K / 4 / 6 |505.98|555.80|527.14|5.157%|5/5|
| 128K / 8 / 6 |640.68|694.32|692.80|0.219%|3/5|
| 128K / 16 / 6 |870.96|884.84|889.28|-0.502%|1/5|
| 8K / 16 / 6 |676.50|710.28|695.36|2.101%|5/5|

前两代表的收益没有推广到整个T1..96区间：B8很小，长历史B16出现回退。
正式源继续保持66882634，不直接迁移四文件，不按测试点白名单绕开回退。
下一步只能基于真实工作量/分核机制建立更合理区间，或改变实现；不能用
四档平均值或异轮绝对值代替逐档结论。四档相对原分task都仍未达30%。

S1/H8K/B16/T16与S5/H8K/B7/T35均完成：功能、两次同输入graph与原PTO
S6因果前缀精确比较通过；Native零容差失败，顶层passed保留false。S1/S5
没有采性能，不代表连续decode、mixed Q或A→B→A metadata覆盖。Native
差异的明细及同规格历史对照由独立边界报告记录，不能自动归为既有差异。

正式迁移前的关闭分支审计另做一次OA256/OB128完整CPU：130根/58动态轴
保持，归一化完整PTO、553项AIC分配、AIC1700/AIV3088局部事件与同控参考
一致；不宣称wait辅助头和机器码全文一致。关闭组流水仅源AST等价。

结果、任务日志、原始trace、冻结33文件双侧源码与CPU审计在
parallel_20261018/oa_pair_manual_evidence；DEVICE_ARTIFACTS列文本及本地
大产物路径/大小，无hash。与两版Q_B负结果一起保留，避免重复试错。


独立S1/S5验收补充：详见oa_pair_manual_evidence/boundary_review/README.md。
S1输出对Native的max_abs=0.03125、RMSE=0.003030962，TopK16/16行集合不同，
50个替换项；S5 max_abs=0.0234375、RMSE=0.003181939，TopK35/35行集合不同，
124个替换项。检索现有890份result类JSON后，未找到同H/B/Q正式源旧结果，
因此相对旧正式源的Native差异仍未归因；不把它自动豁免为旧差异。
两档因果前缀精确对照均通过，故本轮支持的是与原PTO前缀数学一致的结论。
队列completed(exit=1)对应保留的Native精确失败，不是编译或设备崩溃。
内部70个allocator外侧红区与31项guards通过；外部输出未设独立红区，
协议检查为phase0代际+1及所有槽保留字不变，不扩张为全部业务phase owner审计。


## 2026-10-18：小T输出资源域候选提交，尚待设备结果

旧T1..96候选已随8ea77811完整保留，正式33源仍668。四档L1后续只读分析
见oa_pair_manual_evidence/l1_followup_readonly：单SPMD没有内部phase clock，
只能确认B4/短B16的AIC结束更早、长B16更晚，不能把核端点叫作OA纯耗时。

新的oa_pair_small_t_evidence仅把生产/等待两个谓词收为现有OB_ROWS==32
且0<T≤32的连续资源域，原OA_COLS128、组流水条件仍保留；数学/尾行/同步
和Host policy完全不变。现有policy本就T≤32选OB32，每组量化只有一个32行
任务。这是可检验的机制假说，不是B4/B5支持白名单，也不是区间收益保证。

两端predicate以外执行AST不变，OB96/OB128及关闭组流水等12个函数折叠
对照回到668原实现；这只是源级证明，不冒充新生成物或机器码等价。
文档总览纠正旧logical%24和显式L0说明，7处docstring差异单列；剥除
docstring后审计全部执行AST，不声称完整含字符串AST相等。

未重复数学体完整CPU；无新Tile/API/容量，直接让以下正式任务各自JIT并
执行完整原PTO精确/graph/保护及三方同进程五窗L1：
- H128K/B4/S6/T24：task_20261018_061655_17204335454。
- H128K/B5/S6/T30：task_20261018_061655_17204758133。

两任务尚待完成，不预记PASS或采用。仅在tests冻结候选，正式ops尚未迁移。


## 2026-10-18：OB32小T候选两档完整PASS但收益不足，不采用

H128K/B4/T24原分task502.66、正式668 546.38、候选543.34μs，相对参考
仅降低0.556%，3/5窗；H128K/B5/T30为576.36、608.02、608.24μs，回退
0.036%，2/5窗。两任务均完成原PTO精确输出/TopK/分数、图/状态/保护
及代际检查。各自五窗与原始trace保存，异常高参考样本没有删除。

既有OB32/单量化块是合理可检验的资源边界，但本次实测不支持启用。
不迁移四文件，不继续补测其它点，不根据本轮挑选B4白名单；正式33源
保持66882634。前一版T1..96在B4观察到5.157%不等于新版或跨轮都复现，
不能把初次观测写成已采用的稳定收益。生成物只读对照另存，不将其作为
性能归因或替代同轮整层结果。详见oa_pair_small_t_evidence。


小T负结果的B4生成物复审：前96版与新32版完整PTO只差两处cmp上界96→32，
C++两个计算函数各仅一处阈值变化；AIC585/AIV1812项分配、1730/3090条
局部事件及PIPE_ALL20/203均相同。raw AIC45508B、AIV147016B大小不变，
各直接比较仅一字节不同。没有新的布局/数学/局部事件变化证据，也不能
据该立即数字节归因异轮时延。全部五窗保留，含参考626.46μs高值，不删
异常值或挑低值。详见oa_pair_small_t_evidence/b4_codegen_comparison。


## 2026-10-18：Indexer RoPE→Hadamard精确owner等待，CPU通过并提交代表

独立于未采用的OA候选，从正式668只修改cube_schedule一处等待：活跃AIC
不再等全部RoPE AIV，只等16*floor(worker/8)..+15这16核；worker≥T无
Hadamard任务，仅跳过无关获取，原空发布保留。Had每个M64块是一token的
64heads，token=worker+24*w；Rope unit=floor(token/8)*16+hg，24token
步长使unit恰加48，因此所有wave所需producer集合不变。T≥17已有48
生产者；T≤16仅16/32个，但活跃消费者选段完整存在。数学与后续QB所依赖
的QR acquire不动，全部wave后发布、业务fence/cache维护和HAD发布均保留。

标准库对T0..4096、98328个AIC/T情况枚举，精确集合无漏无多；另一个
agent独立验证24个边界×24 AIC与连续rope/head别名、无隐含QB依赖。
固定24AIC/48AIV、64head/8token/4head布局是公式合同，不扩张到任意核数。

唯一完整CPU（longG6/OA128/OB96/QAfalse）PASS，130根/58轴/singleMIX。
生成C++确认worker<T、16次wait、获取维护在HAD TLOAD前，HAD TSTORE
后FIX完成及原发布保持。对照668同controls：AIC558/AIV1812项分配、
AIC1711/AIV3088局部事件相同；AIC仅剔除变化wait段后完整PTO相同，AIV
完整PTO相同。CPU默认wait与设备退避辅助不同，不声称机器码全文相等。

正式源未改。长短历史完整设备/原PTO精确/图/保护与三方五窗任务：
- 128K/B16：task_20261018_063046_197324517994。
- 8K/B32：task_20261018_063046_197328827670。

源码、owner模型、独立复审、生成摘录与命令见idx_had_owner_evidence。
任务刚提交，尚无设备PASS或性能结论，不把少等待次数直接等同于时延收益。


## 2026-10-18：RoPE→HAD 16-owner两代表完整PASS，性能混合，暂不采用

正式668/候选完整L1：128K/B16为875.28→884.52μs，回退1.056%、0/5窗；
8K/B32为998.86→991.82μs，降低0.705%、4/5窗。同轮原分task分别875.78、
977.82μs，仍未达30%目标。两任务完整精确/TopK/分数、图/状态/保护及
代际PASS。依赖集合推导正确不保证完整流水更快；暂不迁移、不按两个点
选启用白名单，不扩跑七档。原始样本与trace完整保留，见DEVICE_RESULTS。

下一独立方向只修改Vector侧HAD→query量化获取，仍以668为参考，不组合
本条Cube候选。量化AIV v处理t=v+48*w，对应Had producer=t%24=v%24，
只在v<T时获取一核；原发布、后继全QUERY_QUANT及其它业务获取均保留。
T0..4096、196656个AIV/T与8390656个token块模型PASS，独立复审确认
到后继全QUERY_QUANT前仅新增提前读本核Had数据，全QUANT前驱并集恢复
全部活跃Had前驱。T0仅模型，正式policy仍拒绝空shape。

唯一完整CPU long/OA128/OB96/QAfalse已PASS，尚待完整生成审阅与设备验证；
源及准备证据在idx_query_quant_owner_evidence，不预记精度或性能。

## 2026-10-18：Query HAD→Quant单owner候选CPU通过并提交

仅收窄活跃AIV的HAD获取为worker%24，后续全QUANT等待保持。完整CPU及生成代码复审通过，AIC全文和AIV除唯一等待块外PTO相同，分配与事件相同。独立提交128K/B16、8K/B32五窗三方完整L1对照，当前待结果，不预记收益。

- 131072/B16：`task_20261018_064439_225528924817`
- 8192/B32：`task_20261018_064439_22553316286`

## 2026-10-18：Query量化单owner两档收口，继续保留正式等待

两任务均exit0，原PTO输出、Top-K、score、cache/state、A→B→A同址图、保护区与72核代际全部通过。五窗三方完整L1：

| 历史/B | 原分task μs | 正式668 μs | 单owner μs | 相对正式降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 870.92 | 871.90 | 885.00 | -1.502% | 0/5 |
| 8K/B32 | 979.04 | 1008.62 | 998.42 | 1.011% | 5/5 |

该候选同样不采用、不扩七档；减少信号等待集合不能代替完整L1收益证据。未改变精度门槛或正式33源，仍未达到逐档30%目标。完整源码、CPU、两次任务日志和全部五窗见parallel_20261018/idx_query_quant_owner_evidence；二进制和数值快照仅保留路径/大小。

新的独立候选attention_scalar_geometry_evidence只在单AIV当前gather query内部缓存四个Scalar，首个有效block才调用原geometry，后续有效block复用；每次kernel入口重置，不跨调用保留Q/L。用tag而非sb==0处理首块无效，active=0也缓存，PV晚三拍reset不触碰gather缓存。原guard、数学、FFTS及发布不改。源/同源模型7987调用、132331拍通过，唯一完整CPU正在运行，尚无设备结论。

几何缓存完整CPU及生成审阅已通过：130根/58动态轴、单MIX，AIC整函数归一相同，AIV分配/局部事件相同。4个Scalar的初始化、真实tick/mask/tag三层guard、无效/drain透传与PV reset不干扰均已核对。任务已提交：
- 131072/B16：`task_20261018_065503_256120011764`
- 8192/B32：`task_20261018_065503_256123927920`
两档仍在运行，暂不预记精度/性能通过。与前面的两条Indexer等待候选完全独立，不将跨轮收益相加。

## 2026-10-18：本核几何缓存首轮设备结果与交叉代表

两个首轮任务均完整PASS。128K/B16正式877.84→候选882.72us，回退0.556%，0/5更快；8K/B32正式1005.18→982.90us，改善2.217%，5/5更快。同轮原分task分别878.74/969.80us，仍未满足30%目标。数学、状态、同址图与协议均通过，正式源暂不改。

仅补8K/B16与128K/B24两个交叉代表，分别保持B/T或长历史策略的一侧对照，核实收益是否与请求扫描规模或既有策略有关；这些点不能单独证明性能因果。没有重测已完成两档，没有开启七档扫描。原33源和单次CPU不变。首轮结果单独保留FIRST_TWO_RESULTS.md/json，待补测完成后再决定区间，不预定启用白名单。
- 8192/B16：`task_20261018_070212_271203632049`
- 131072/B24：`task_20261018_070212_27121133702`

并行设计dq_score_interleave_readonly保留“原owner内4-head段穿插Score”的具体接口，但本轮不实现/不编译/不上卡。旧同compact次序时钟显示DQ大部分已与MAIN_PROJECT重叠，QUERY仅约2–4us尾部落在其全体就绪后；这不是精确可省上界。分段会重复RoPE/scale准备并增加live UB，当前不足以支持进一步调测。已有草稿任务在写源码前收住。


## 2026-10-18：已有Top-K交集只读分析的边界

只读取历史正式W8A8第4层、随机hidden/KV fixture的两份Top-K（共589824B），未读取权重或KV payload、未编译/上卡。128K/B16与8K/B32同请求相邻query平均逻辑交集分别5.339%与29.915%；同编号128候选块平均仅1.8094与10.7453行。物理页映射未保存在states，依据当时保留fixture源码推导，明确不冒充实测metadata；原report metadata只读检查全PASS。按当前q%24分核，q→q+24属于不同请求，在该独占页fixture中推导物理交集均0。

这些不是整网真实请求激活，不能推断真实服务共享价值。当前数据不足以据此实施跨query KV共享；保存报告、原始统计、最小只读复现及来源清单于attention_query_overlap_readonly，未启动新性能任务。


## 2026-10-18：本核几何缓存四代表收口，暂不采用

补充的8K/B16与128K/B24两任务均完整PASS，五窗三方完整L1结果如下（正号表示相对正式668降低）：

| 历史/B | 原分task μs | 正式668 μs | 候选 μs | 相对正式降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 878.74 | 877.84 | 882.72 | -0.556% | 0/5 |
| 8K/B32 | 969.80 | 1005.18 | 982.90 | 2.217% | 5/5 |
| 8K/B16 | 667.50 | 698.60 | 710.04 | -1.638% | 0/5 |
| 128K/B24 | 1169.62 | 1163.42 | 1182.38 | -1.630% | 1/5 |

四档均通过原动态PTO精确输出、Top-K、分数、cache/state、A→B→A同址图、保护区与72核代际，未改变精度要求；不代表Native/token或完整动态支持验收。四档仅一档改善，不足以归纳可靠区间，正式33源继续保持668。即使短B16/B32都处于同一G2策略，也未表现一致收益；不能将“少扫描Q/L”直接等同完整流水更快。暂不采用，不写测试点白名单，不继续扩七档。所有任务、源、CPU与五窗trace保留在attention_scalar_geometry_evidence。

下一独立候选qb_global_n128_evidence仅把主Q_B的NZ列块从256缩至128，全局256任务按24核重新分配；ND512保持不变，M128及动态T尾块保持，Vector获取计数使用QPROJ_N_BLOCKS。旧20核固定T96的globalN128已CPU通过但Right仍别名、未上卡，必须保留为先例，不能把本次宣称为新出现的双缓冲能力。当前只准备源码与一次完整CPU，双槽与性能均待生成物和真机确认。


## 2026-10-18：主Q_B全局N128候选CPU通过并提交两代表

唯一完整CPU通过：130根/58动态轴、单MIX、sync_start，未提供sample shape。NZ全局256个N128任务、M128动态尾行及四段K256顺序保持；ND512不变。独立生成审阅确认4个Right[256,128]仍同addr0，Left也同addr0，stage2没有形成双Right。Mat有两组QR/B供数槽，QB Acc减至64KiB；全局资源上界不变，仍保留24核QB发布/获取。AIV归一PTO全文、1812项分配与3088条局部事件相同。

需保留的消融局限：8条非QB Acc分配迁至98304；AIC局部事件1711→1718，其中QB供数链净+4、compact Hadamard多1个MTE1 barrier、主compressor多2个。故后续完整L1差异不能纯归因于列块均衡，更不能声称来自双Right。完整审阅位于qb_global_n128_evidence/generated_review（独立报告待归档），编译产物仅保存路径/大小。

正式源未改；提交同进程三方、完整精确/图/状态/保护及五窗完整L1任务：
- 131072/B16：`task_20261018_071700_29885368792`。
- 8192/B32：`task_20261018_071700_29885801413`。
两任务待结果，不预记精度或收益，不组合几何缓存与其他未采用候选。


## 2026-10-18：主Q_B全局N128两代表完整通过，无收益不采用

两任务均exit0，原动态PTO精确输出、Top-K、score、cache/state、A→B→A同址图、保护区与72核代际全部PASS。三方同进程、五窗轮换完整L1中位数：

| 历史/B | 原分task μs | 正式668 μs | 全局N128 μs | 相对正式降低 | 更快窗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 873.60 | 881.40 | 882.74 | -0.152% | 2/5 |
| 8K/B32 | 985.64 | 1004.20 | 1005.74 | -0.153% | 1/5 |

本轮没有实测改善；单核最大逻辑列量下降不能替代完整L1收益。正式源保持668，不合入、不扩七档、不追加手工双Right变体；生成物Right仍单槽，且非QB分配/局部barrier变化作为消融边界记录。源码、动态模型、唯一CPU、独立生成审阅、任务日志与全部五窗结果保存于qb_global_n128_evidence。设备结果仅覆盖两代表，未进行Native/token或完整动态区间验收，30%逐档目标未完成。

下一步仅为Score服务区间诊断：复用正式668、128K/B16、三轮A/B/A和原完整精确/图/保护检查，无新优化候选或性能窗。两个agent分别实现C++计时hook与runner，第三agent独立审阅；只采固定j8细分及全step/ACK累计，在原release前一次写本核独占padding。不将插桩累计、局部wait或单panel样本当端到端性能，也不乘16外推；正式源和依赖均不改。


## 2026-10-18：正式668 Score服务诊断准备完成并提交唯一任务

本次不是优化候选，正式33源逐字保持668。两个agent分别实现测试C++ hook和runner，第三agent独立审阅。复用已有formal668原始C++做离线检查，没有新启动PyPTO CPU编译。固定128K/B16/S6、long G6/N64/1024，仅本次诊断限制，不改变正式动态支持范围。

hook记录整个Score body/step与ACK，并只在固定第九个QK面板附近采样页/Key、QK+FIX、前一面板WS和原local wait。20个独立起始变量避免嵌套覆盖，独立review逐项核对同一C++大括号作用域。原同步语句序列、Vector活动函数不变，剥离标记逐字恢复原C++。每核只在phase34自己的32..63列一次写入，768唯一word，schema/epoch与计数严格检查；17个反例、0/1/2/3/7/33动态step离线模型通过。仅保留原phase-clock16..19，不混入Attention tick或FFTS20/21。

原phase release承担可见性；本hook没有新增硬件fence，组合的旧phase-clock仍有其既有DCCI/DSB。Scalar区间包括发射、原依赖和插桩扰动，邻接区间含计时开销，不能称纯异步引擎耗时、相加为完整延迟或将j8样本乘16。当前工具无法读取最终A3机器码，未核实CCEC时钟指令次序和spill。

runner复用正式权重、原动态PTO精确oracle、状态/保护区及图代际检查，三轮A→B→A只改变hidden正负，不声称本次切换了history/page metadata。无DFX/L1性能采样窗，eligible_for_performance=false。主提交包装初次字段名不匹配在task-submit前报错，修正为实际ready键后才执行唯一任务，未重复占卡。
唯一任务：`task_20261018_073047_323892912778`；当前待设备结果，尚不记PASS。源码/runner/设计/离线证据位于score_service_clock_evidence。


## 2026-10-18：Score服务诊断完整通过，保留发射与依赖解释边界

唯一任务task_20261018_073047_323892912778完成exit0。eager、graph预热及三轮A/B/A共5次完整检查全部PASS，原PTO精确输出/Top-K/score/cache/state、保护区及72核代际正确，最终generation=5；正式668的33源未变。本次A/B仅hidden正负，未切换history/page，不冒充额外动态metadata验收。

下表是每核本次累计量再取24核中位数（us），不是单panel或完整L1性能。每轮每核均4个叶、22个step、22次ACK、22个固定j8样本和220处local wait。

| 轮次 | body | step | ACK | j8页/Key | j8 QK+FIX | 对应WS | 嵌套local wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 232.28 | 214.24 | 30.91 | 3.83 | 0.20 | 7.44 | 6.42 |
| 1 | 239.39 | 219.23 | 32.27 | 3.84 | 0.20 | 7.43 | 6.28 |
| 2 | 233.21 | 215.76 | 29.71 | 3.87 | 0.18 | 7.50 | 6.33 |

生成事件映射确认，采样实际为预取Key9、QK8、WS7；10处local wait分为页3处、QK2处、WS5处。最后WS的M→FIX事件位于先发QK8、再发WS7之后，保障此前M队列前缀完成，不能把约7.5us当WS独占计算。6.3us是10站合计，不能归给某个wait，也不与WS直接相减出纯compute。QK+FIX接近0仅说明这段Scalar发射区间短，不代表QK硬件耗时为0。不能将j8乘16、相加嵌套区间或将探针端点当完整性能收益。

同次阶段端点与独立统计见DEVICE_DIAGNOSTIC_CHECKS.json和analysis/；实际事件/生产消费来源见wait_mapping/。三个核组body与ACK呈相同分组趋势，而固定j8页/WS无相同单调趋势，最多支持进一步核对消费相位/背压，不能据此归因硬件慢核。历史全局G3/N128已测并后被G6替代，当前观察不足以重试；另一个G6内部双M192/N128负例独立列出，不混淆两者。

完整探针、runner、源快照、初稿修正、离线/独立审阅、真机结果/日志均在score_service_clock_evidence。无额外L1窗口、无七档重测、无新增正式优化。当前仅继续只读评估不同请求波次的Score/Attention块交错，与旧whole-stage两wave方案排重，核对有向事件顺序和同时live资源；未开始实现或编译。

## 2026-10-18：三路并行审查 Score/Attention 跨请求交错，完成两L1槽源码准备

本轮继续七档逐档相对原动态分task PTO完整L1降低至少30%的目标，正式动态33源仍为66882634。Score服务插桩证据已单独提交4fe299d1；插桩数据不计入性能验收。三路并行审查分别负责同步拓扑、Mat/UB/Left容量、可暂停step的源码状态；归档见parallel_20261018/score_attention_interleave_design。

候选机制为同组AIC先Score step再前波Attention tick、双AIV先前波Attention tick再消费当前Score。前波的叶根、TopK和plan必须先完整发布获取，各组有独立游标，不能把ACK当根写回完成。旧整helper排空切换的负例不直接否定此机制。保留原owner时B16连续四group的24叶只覆盖12个AIC，因此不能以总任务数等于24宣称均衡；混合query和dummy行的覆盖、M8 plan切点也须显式处理。

原两helper直接同驻留不成立：Score Left60KiB与Attention临时16KiB相冲突，原Attention Mat448KiB加Score临时112KiB也超限。三L1槽借已释放槽的想法受boxed Tile reinterpret限制，未用低层IR绕过。优先准备仅AIC内部KV两槽、每tick先PV后QK的源码，GM三槽和AIV函数完全保留；Query48KiB驻Mat并每次切回重TMOV、系数12KiB留Left。保守容量496KiB，额外计入leaf初始化系数暂态508KiB，只有容量依据，不代表联合planner或性能已通过。

两槽源码准备在parallel_20261018/score_attention_two_l1_source，完整diff与源级审查显示：AIV装饰器至末逐字节不变；QK/PV子树除内部L1模数外保持；GM地址五项相同；动态T、mask、首尾tick保留。尚未PyPTO编译或NPU验证，不作为独立性能候选提交任务。需验证真实MTE1→MTE2槽复用、48KiB重装和PV先行损失的原流水重叠。

状态拆分进一步发现当前@pl.jit.inline特化层未保留Tile参数注解；底层pl.function支持Tile不等于JIT入口能发现外部Inline函数。下一步先在新的独立Python源码overlay复现必要接口缺口，复用只读既有二进制，不改共享安装、旧隔离依赖或正式算子；只考虑最小注解保留修复，不扩展其它功能。另两路继续动态wave边界源码与按波隔离phase容量设计。当前没有新增完整L1性能、没有新精度PASS、没有全七档重测，目标仍未达成。

### 本阶段收口：动态helper源码与窄接口CPU验证完成

三路源码准备均已保存：`score_attention_cube_step_source`提取完整AIC G6 Score step；`score_attention_vector_step_source`提取AIV叶状态初始化、完整step及根发布；`score_attention_tick_source`提取Attention两角色的初始化/tick。固定维度仅是原G6/N64/1024、head与数学tile；B/T、Q/L、位置、页表与有效候选继续动态。源级AST还原相等，原数学/排序/事件保留；联合状态机仍未接入，不能记完整CSA编译或精度PASS。

`score_attention_wave_boundaries`的纯整数检查共1235场景通过，覆盖S1/S5/S6、G1/G2/G6、mixed/odd、dummy/gaps及T至4096；只证明分波边界和容量关系，不是动态CSA功能验收。`score_attention_interleave_design/wave_progress`明确保留原60阶段并为各wave追加ROOT/TOPK/PLAN，原根ABI不变，runner固定60的回读/保护检查须配套修改。Attention还必须把query_first与物理qk_core分开：6处query地址用前者，5处GM ring仍用后者；每wave相对tick从0起并排空，不可用global tick跳过首部，当前冻结helper尚未接此参数。

`jit_tile_parameter_evidence`归档独立overlay最小补丁与56项小文本。实际最小失败为inline参数缺注解；specializer只增加Tile注解保留和已有模块常量折叠。普通Tile原样返回以及实际full-valid状态的assemble/tuple/loop carry均完成CPU lowering/PTOAS，动态T仍为运行时参数。full-state生成Vec roots/segments在0/12288B，incoming在36864B；仅原assemble的两次TMOV，无helper整体复制。旧隔离依赖文件保持原文，共享安装未动。

边界如实保留：首次夹具把pl放局部引发注解命名空间失败，修正为与正式源一致的模块导入后再次参考/候选对照；动态valid_shape的loop carry仍因valid类型不一致失败；同helper constexpr维度注解仍不支持，InOut仅specialize通过。Mat/Left的新helper及联合CSA尚未编译，不扩大Vec小例结论。所有失败、小生成PTO/C++和本地大文件路径大小清单均保存；没有新增NPU任务或性能采样。下一阶段先组装动态wave控制、查询基址和全部状态，再做一次有理由的代表性联合lowering。


## 2026-10-18：动态wave联合接入完成，编译推进至Left容量约束

继续三路并行，分别负责AIC step/tick、双AIV状态与所有权、隔离runner/JIT边界，主会话接入动态wave控制、根/TopK/plan依赖及完整schedule。候选和所有过程在 `parallel_20261018/score_attention_interleave_candidate`；正式33源和独立reference经直接逐字节比对仍为66882634，没有hash。

本轮首次真正把此前拆出的数学helper接进完整动态CSA。只在已有G6/N64/Score1024/prefetch1/toL0/balanced/roots1策略且device描述符可形成多个M8对齐wave时启用；其余保留原路径。Score保持global item%24，TopK q%48，plan M8条带，Attention q%24，query_first和物理ring owner分离。首wave生成Score/TopK/plan，中间wave在同组中AIC先Score后前波Attention、AIV反序，末wave独立排空。根ACK与排序写回明确分离；每波ROOT/TOPK/PLAN只由真实writer整波完成后发布，原60阶段与每核generation保留，容量按Host已有T扩为60+3*ceil(T/8)，没有新的根或Host数值metadata。

源码核对通过：range仅裁范围，原数学body AST相同；AIC初始化/末叶ACK、Attention六query五ring地址分别审计；18个代表整数场景覆盖S1/S5/S6、mixed、dummy/gap、陈旧位置、zero-visible和T0，44wave/1503有效叶的分工、步数及根等待均有对应producer。33824个同组事件DAG案例未发现新增等待环；这些都不是硬件同步、精度或性能验证。runner保留原检查并加扩展padding；新的wave word0完整writer/代际递推验收仍需接入，已有独立设计，不能省略。

实际联合CPU尝试逐项保存：

- v1：58动态轴解析通过，Inline因Tile形参原地重绑定遇实际tuple元素而导致AssignStmt左侧非Var。仅在两个helper给被重绑Tile加局部别名，去别名逆rename后整模块AST相同；没有改依赖。独立Mat/Left full-valid helper、tuple与loop carry已完成CPU/PTOAS，不能把v1泛化为全部Tile接口不可用。
- v2：Inline通过，ConvertToSSA共228条诊断。自然while更新被提前版本化造成cursor/total/wave退出值越作用域；显式Tile返回与隐式GM重绑混用造成phi数量不符。改成显式pl.while_/cond/yield，固定GM仅接局部返回别名。另102个FREE_VAR首次在UnrollLoops出现：底层DeepClone克隆了外部IterArg。最小对照直接IterArg失败、helper调用前局部Scalar别名SSA通过；在AIV保留数学不变地增加桥接，没有修改编译器。
- v3：58动态轴保留，无sample shape；通过Inline/Unroll/SSA，直到实际AllocateMemoryAddr因Left77824B>65536B失败。实际pass37中Score Query48KiB与Attention临时16KiB已经复用同一64KiB池；64KiB上界来自更早主压缩器的两份32KiB槽。超出的12KiB来自单列的持久Coeff池，不能归因为Query与Attention未复用。Mat同时存在48+48+256+384=736KiB四个池，AllocateMemoryAddr逐base计入容量；Left先失败掩盖了Mat超限，简单把Coeff搬至Mat也不能解决。进一步核对pass35→36发现gather结果被MaterializeSemanticAliases换到新MemRef，但紧随的transpose_view仍保留旧MemRef，无实际copy。这是必须优先修复的别名正确性缺口，不能靠合池隐藏。

v2诊断把整函数对每条错误重复打印，产生622MB result等大文件；原件本地保留，仓库仅保存错误类型/位置摘要及路径大小。后续私有runner将完整异常单独保存，stdout/result限长而不删失败信息。一次从工作区根启动wrapper exit128且无编译产物，切仓库cwd后正常；新wrapper在激活环境前明确cd仓库，记为环境启动问题。

本阶段没有NPU任务、没有新的精度PASS或完整L1性能样本，没有七档重测。正式实现不受这些未验证候选影响；七档逐档相较原动态分task PTO至少降低30%的目标仍未达成。下一步先解决实际Left活跃区间与容量，再做代表档位联合验证。

### 用户授权内存复用改进，建立新的依赖对照工作区

用户明确允许合理改进PyPTO或PTOAS的内存复用。新建`.worktrees/pypto-csa-memory-reuse-20261018`，分支`csa-spmd-memory-reuse-20261018`，基于旧静态SPMD依赖同一`cb2484471`；继承旧工作区全部31个常规跟踪文件源码修改，加此前已验证的Tile参数注解六行修复。旧`.worktrees/pypto-csa-static-fixes-20261016`保持原状，不是本轮新工作区。runtime仍要求`57e499238`，构建必须在新树独立进行，不复制或软链旧core二进制。初始继承用直接字节比较确认，补丁及来源存于候选的`compiler_memory_reuse_worktree/`。

首先修复MaterializeSemanticAliases未传播view别名的问题，再依据真实活跃区间处理其它内存池；不放宽硬件容量，不改变数学或异步完成约束。当前候选仍未通过完整编译，更无新增精度或性能结果。

## 2026-10-18：将完整静态参考算子独立归档到 ops，并完成搬迁后真机复现

用户明确要求静态 SPMD 也在 `ops/pypto` 单开目录保留，且必须具备独立复现所需的完整
PyPTO 算子代码，tests 只保留测试。这是对原“静态仅放 tests”约束的明确调整；正式动态
目录和七档性能目标不变，不把静态形状包装成动态实现。

新增 `deepseek_v4_flash_csa_static_reference/`，保留历史已通过的 128K/B4、B8、B16、B24
和 8K/B16、B24、B32，均 S6。各档包含完整 `entry_jit.py`、参数合同、工作区与逐核
records/waits 调度表；共享本包的 A3 signals、cached100 等待生成处理和独立参数准备。
公开入口 `reference.prepare_static_csa` 在首次调用时完成 JIT 编译，核对真正交给 CCEC
的 AIC/AIV 等待源码，后续使用同一工作区与进度代际。Native 输入与原 NZ 权重保持原对象，
没有 Host 读回 metadata。所有运行必需文件均来自该包，不再导入 tests 中的历史算子或
读取历史生成目录；manifest 中旧路径仅是来源记录。七份数学 AST 与所选历史版本一致。

独立测试放在 `tests/pypto_test/csa_static_reference/`，仅复用既有通用 fixture、权重加载
和原多 task PTO 作为比较基线。唯一搬迁验证任务 `task_20261018_095114_228736420852`
完成 exit=0；128K/B16/S6，正式 W8A8 第4层权重，固定 seed 合成历史。A→B→A 的输出、
Top-K/score、完整 cache/state 逐 bit 一致，Native/arena 保护区、72 核代际、只读调度表
与图重放全部通过。三窗完整 L1：原多 task 864.90/854.08/880.98 μs，静态
782.66/778.06/787.36 μs，中位数 864.90→782.66 μs。每个静态窗口均为覆盖72核的
一个 MIX task。其余六档本次未重测；8K/B24保留历史价值，不恢复为当前目标档位。

本轮实际两方测量顺序均为多 task 后静态，结果原样保存；随后修正测试驱动中先轮转再
反转相互抵消的 AB/BA 顺序，并将统计字段计数对象改为 static_spmd。该修正不改数学，
不追改已测结果，也不为计数字段重复占卡。复现说明、完整结果、三窗原始L1与泳道图、
数学搬迁核对及显式提交清单均归档；编译二进制/缓存仅保存本地路径和大小，无hash。

128K/B28尚无可执行验收通过的静态参考，其旧driver导入失败按用户要求保留、不修复。
本次静态归档与测试完成，不代表动态交错候选通过，也不计入动态单SPMD逐档降低30%的
目标完成度。主优化工作仍为正式动态源码及其候选。

## 2026-10-18：修正内存别名后进入设备验证，定位并复现两项标量编译错误

本节证据均在 `tests/pypto_test/csa_multishape_spmd_20261016/`
`dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/`。
正式动态33源仍保持66882634；未将失败候选合入正式入口，也未放宽精度阈值。

独立 PyPTO worktree 完成两组最小修复：MaterializeSemanticAliases 重新绑定父
Tile 后，view 别名随父分配更新且保留 byte offset/window；内存复用的 For/If-phi
遍历补入 While body，避免漏掉内层 Mat carry/累加器。相关 `test_memory_reuse.py`
158项唯一用例通过：初轮153通过，5个新fixture违反Mat/Left使用约束，修正测试
构造后5项通过；旧核心负对照准确复现两项缺陷。证据与完整补丁在
`compiler_memory_reuse_worktree/VALIDATION_RESULTS.json` 等文件，新核心独立构建。

整层CPU v4已满足Left64KiB、Mat496KiB及Vec188416B容量，但代码生成遇到内联
参数位置上的tile核心ID表达式；源内先绑定Scalar再传参后，v5完成58动态轴、
无sample shape实值化的完整lowering/PTOAS。没有通过扩大硬件容量来掩盖溢出。

设备首轮 `task_20261018_092846_145569930612` 与诊断轮
`task_20261018_095309_236293230276` 均卡在候选首次调用，返回507018。原多task
输入对照已完成；候选没有精度或性能结果。第二轮失败后用ACL同步回读信号、
group_plan/offsets，保留实际生成PTO/C++与本任务日志路径。phase34/38只有core0
前进，第二波ROOT只有AIV24/25写入，定位到实际控制流而不是笼统归因AICPU。

第一项缺陷在 PTOAS：循环 `scf.yield` 的多变量更新应同时取旧值，C++ 回边却
先覆盖 current，再把已覆盖的 current 赋给 previous。实际AIC/AIV各一处外层
wave回边存在该错误；其它普通前向边的重复赋值不能一律判错。独立最小程序包含
shift、三变量轮换和条件交换，原0.66生成C++在CPU执行20例有12例失败。例如
两轮应得到 `(current, previous)=(96,48)`，实际为 `(96,96)`。

PTOAS候选修复仅扩展 `materializeControlFlowOperands`，在生成边赋值前物化
BlockArgument旧值；独立工作区 `.worktrees/ptoas-csa-parallel-copy-20261018`，
基线f5eff3ee。管理脚本遇到源仓已跟踪的异机workspace元数据而回滚，保留该文件
后手动完成隔离；pip代理及CMake原生入口问题分别记录，当前复用已有LLVM依赖、
独立构建PTOAS本体，未复制旧对象或替换共享二进制。截至本节记录仍在编译，
不宣称修复后的二进制已通过。完整CSA先通过等价重排carry顺序避开该覆盖问题。

第二项缺陷在 PyPTO：FloorMod直接发射`arith.remsi`，与IR/简化器采用的floor
语义不一致。`(worker-begin)%24`在负数输入时得到负余数，部分AIV会选到wave
开始之前的query，形成与AIC不一致的工作/等待次数。新工作区修正signed/INDEX
的余数符号，浮点及其它类型原路径保持；未顺带宣称修复负数FloorDiv。10项实际
生成C++数值测试涵盖INT32/INT64/INDEX、正负与动态除数及query owner：旧核心
10失败，新核心10通过。初始fixture误用InCore标量返回且未锁定PTOAS路径，
明确记为无效测试；改用Tensor输出和显式PTOAS_ROOT后的v2才是有效证据。

用户要求核对[PyPTO PR #2962](https://github.com/hw-native-sys/pypto/pull/2962)
所要求的PTOAS0.67。本轮隔离安装官方ARM64/Python3.10 wheel并实测：回边20例
仍12失败；旧PyPTO+0.67的FloorMod10例仍全部失败，修正PyPTO+0.67全部通过。
PR当前head也仍直接发射remsi。FP4打包/别名及all_to_all_v计数发布修复与上述
缺陷不是同一问题，不能靠版本升级替代。源码定位、版本、命令、结果及边界见
候选目录 `PTOAS_067_RECHECK.md`；共享安装未改。

采用两处修正/规避的128K/B16/S6设备轮
`task_20261018_103107_374461711536` 已正常返回，随后测试检查器失败：实际
plan中的true经i1符号扩展表示为-1，原检查器只接受1。保存控制区的两波
ROOT/TOPK/PLAN实际写者离线核对全部匹配。检查器现只接受1/-1两种true表示，
仍拒绝0/2/-2/37并保留完整writer、generation与padding验证；没有跳过协议。
当轮数值比较未在该异常前落盘，不能补记精度PASS。runner另在协议检查前落盘
独立数值/保护区结果，后续异常不会再掩盖已完成的比较。

重提 `task_20261018_103852_401858324030`：同一正式权重、128K/B16/S6，A→B→A、
完整cache/state/Top-K/score、保护区与图重放通过后，才执行5窗三方完整L1。
截至本节记录仍运行，候选尚无已验收性能结果。metadata/padding专用重放的新增
wave写者递推检查尚未全部接入，仍需后续处理；不把单个规格扩大成七档完成。

另保存一次同进程对照 `task_20261018_092130_124935114251`，原多task、正式动态
单SPMD、历史静态单SPMD的5窗中位数分别869.78、889.40、792.78μs，A/B/A与图
重放等检查通过。动态对静态96.62μs差距在同轮确实存在，不能用跨轮环境差异
解释。用户要求的Native128K/B28优化对照也已完成，独立目录
`tests/pypto_test/native_h131072_b28_optimized_20261018/`保存实际静态编译、
superkernel与图配置证据；30窗外部kernel跨度中位1701.75μs，20次非profiling
Event中位1653.08μs。Native外部口径与PTO内部L1不同，不计算混口径收益。

### 同轮动态交错候选完成：功能通过，性能未改善

`task_20261018_103852_401858324030` 后续完成PASS，详见
`interleave_floor_mod_h131072_b16_v4/result.json`。A→B→A 的输出、Top-K/score、
完整cache/state逐bit一致，Native与所有工作区保护区、实际wave写者/代际和基本
图重放全部通过。完整L1五窗如下，均为同进程、同根Tensor与正式权重：

| 实现 | 五窗耗时（μs） | 中位数（μs） |
| --- | --- | --- |
| 原动态分task PTO | 877.84 / 861.94 / 874.54 / 879.18 / 882.02 | 877.84 |
| 动态单SPMD交错候选 | 888.00 / 888.54 / 894.44 / 896.40 / 889.96 | 889.96 |
| 历史静态单SPMD | 784.06 / 787.30 / 789.80 / 783.72 / 799.00 | 787.30 |

候选相较原分task慢1.38%，静态仍快102.66μs；这轮不采纳候选为性能优化版本。
没有用修复编译错误来宣称性能达标，也不以这一点的功能通过替代S1/S5、混合长度、
策略边界及其余六档验证。后续优化继续针对动态版本的真实开销。

用户明确授权PyPTO缺陷直接提交上游PR、PTOAS缺陷直接提交issue。PTOAS已提交
[#1622](https://github.com/hw-native-sys/PTOAS/issues/1622)，有独立CPU最小复现和
0.66/0.67负对照，并说明本地补丁尚在编译。首次pass模板引用的ir-pass标签不存在，
确认未创建issue后换用官方通用Bug模板，未创建重复issue。PyPTO从最新主线
14736f65c建立`.worktrees/pypto-floor-mod-pr-20261018`，只迁移FloorMod修复、
测试及中英文文档，独立构建；内存别名/while遍历修复核对主线后另行提交，
不把31项历史SPMD兼容修改混入上游PR。

### 2026-10-18：按最新要求改为四份issue，附最小补丁并保留本地验证

用户随后明确要求PyPTO修改以issue和修改建议提供，不直接提修改PR，并要求不再
添加bug标签。按此执行：此前准备的PyPTO本地分支和提交保留，未推送、未创建PR；
新问题只提交issue。四份issue均已补充最小代码diff、对应源码版本、验证范围，
编辑后读取远端正文核对一致，标签未修改：

| Issue | 具体缺陷 | 建议修正 |
| --- | --- | --- |
| [PyPTO #3026](https://github.com/hw-native-sys/pypto/issues/3026) | signed/INDEX FloorMod直接变为remsi，负数余数语义错误 | remsi后按非零余数与除数符号修正，常量除数简化条件 |
| [PyPTO #3027](https://github.com/hw-native-sys/pypto/issues/3027) | 原地输出重绑定后下游view仍引用陈旧MemRef | 同一正向遍历传播must-alias关系，保留窗口/slot与未决phi |
| [PyPTO #3028](https://github.com/hw-native-sys/pypto/issues/3028) | For/Acc别名规则的遍历遗漏While子树 | 两个递归入口补While body，继续使用已有合法性约束 |
| [PTOAS #1622](https://github.com/hw-native-sys/PTOAS/issues/1622) | CFG回边顺序写回覆盖并行赋值的旧源值 | 边赋值前物化源BlockArgument临时值，保护链式依赖及交换/轮换 |

PyPTO当前主线14736f65c的干净本地构建：FloorMod相关codegen首次175项通过，
1项旧多缓冲测试错误地要求slot直接定义为rem；改为检查完整SSA依赖链、逻辑余数
及非字节缩放后该项通过，合计176项唯一用例。另实际用官方PTOAS0.67编译多缓冲
IR，动态slot与event id仍保留，不仅放宽断言。10项负数取模数值回归已通过。

两项内存缺陷另在最新主线原样memory_reuse_pass上复现：新增10例6失败/4通过。
本地联合补丁重新构建后，完整test_memory_reuse.py的154项通过；此前158项属于
旧CSA兼容基线，不能混作同一套结果。两份issue中的diff按缺陷分别拆分，明确
154项是联合构建的验证，未声称拆分补丁分别独立跑过全套。

PTOAS独立原生构建完成527/527，源码基线f5eff3ee。本地新二进制版本仍显示0.66，
实际重新编译parallel_copy.pto，再保留函数原文执行Scalar C++，shift/三变量轮换/
条件交换20项全部通过；官方0.66/0.67同组均12失败。结果在候选目录
`ptoas_parallel_copy_repro/fixed_host_v1/result.json`，编译命令与版本在
`fixed_compile_v1.json`。仅验证Scalar状态变换，不验证NPU缓存一致性、完整CSA或
性能；共享安装未替换，已完成的CSA v4仍使用原PTOAS加源级carry顺序规避。

全部issue正文、源码diff、最小复现和远端编辑前后快照在候选目录
`upstream_issues_20261018/`。`ISSUE_DIFFS_UPDATED.json`记录四份远端正文核对结果。
这些编译正确性修正不构成性能收益，动态CSA七档30%目标仍未完成。

### 2026-10-18：回到动态性能，投影分支分核与早就绪Indexer压缩候选

上一阶段已完成四项组件问题的复现、issue与代码建议，属于实际进展；目标仍为
七档动态单SPMD完整L1各比原分task降低至少30%，不能将修复编译错误当性能收益。
本轮没有重跑已被否决的Score/Attention交错候选，也没有改历史静态算子。

最新同轮889.96/787.30微秒差距促使核对真实前段顺序。历史静态阶段记录显示，
主Q_B由20名AIC执行，其他核在其间推进Hadamard/KV；当前动态版的独立分支主要
按统一顺序串行推进。历史时钟与当前非同轮，不能相减预估收益，但源码与真实
worker记录足以支持一个新的分核候选。

正式动态目录新增projection_branches.py及PROJECTION_BRANCH_GROUPS选项，默认0。
候选先做原24核主压缩，随后前4核做head权重、query/compact Hadamard和KV，
后20核做Q_B；结束后回到原24组Indexer。query Hadamard采用四token的M256，
每行K128与FP32输出保持，尾块依实际T裁剪。AIV提前处理更早就绪的Indexer
pool/commit/norm，再做query RoPE及head权重。全部Native设备入参、原130根Tensor
ABI、phase布局与缓存维护保留；不按B16/S6冻结输入。8辅助核仅作为可选资源策略，
未预设启用区间。对外默认0，未宣称已采用为性能改进。

测试记录位于parallel_20261018/projection_branch_evidence。CPU v1误用Tile store，
v2的分支局部请求数逃逸，v3的减法worker映射无法证明NZ非负，均保留失败快照。
分别改为Tensor assemble、外提共同设备shape读取、后部连续核编号的取模双射。
v4完整符号编译通过：单MIX/sync_start、58动态轴、130运行时Tensor，不传样例shape。
实际生成PTO的M256 load/matmul/store都使用动态有效行；未关闭容量/合法性检查。

前段控制检查直接执行正式源码的Scalar表达式，数学仅记为串行节点，不估算时长。
450个T/S/compact/核组组合的发布唯一、依赖无环、消费者覆盖及Q_B列块覆盖通过；
这不是设备数学或非一致性cache验证。本轮使用既有独立PyPTO内存修复及Simpler
扩容环境，没有新增依赖修改，也未切到新构建PTOAS。

设备任务task_20261018_114949_268532326529已执行128K/B16/S6的A→B→A，输出、
Top-K/score、完整cache/state、全部保护区与阶段协议逐轮通过。随后在同进程编译
历史静态参考，再做图重放及五窗三方完整L1；截至本条尚无性能结论，不预记PASS。

### 2026-10-18：投影4辅助核候选退步，并核对905与880微秒基线来源

上述task_20261018_114949_268532326529最终exit0。A→B→A、完整输出/Top-K/score、
六份cache/state、信号代际、Native及暂存保护区和基本graph replay通过。
同进程五窗完整L1：原分task878.50/890.20/880.06/872.78/893.20，
中位880.06μs；候选946.46/946.96/947.16/958.24/950.28，中位947.16μs；
历史静态781.94/785.94/778.68/778.52/785.96，中位781.94μs。
候选比同轮分task慢7.62%，不采用，正式策略默认仍为0。没有把功能通过等同于
性能收益，也没有以静态列的跨轮变化作为优化结论。

同根输入、同进程两版动态SPMD的阶段时钟任务task_20261018_115704_29584807981
已进入运行，A→B→A三轮，用于定位该分核候选的退步；非完整L1性能验收。
记录在parallel_20261018/projection_branch_evidence，冻结两版源码和命令已保存。

用户追问128K/B16原分task905μs为何成为880μs。只读核对得到：905.08来自
task_20261017_044912_40225339550的三窗888.48/905.08/909.80；本轮五窗中位
880.06，相差25.02μs/2.76%。两次均为设备0、正式第4层W8A8、H131072/B16/S6，
完整incore首尾、50MHz记录时钟、66task/973核记录/72核，统计范围未缩减。
冻结包与当前CSA子包23个Python文件逐字相同，权重加载记录相同；编排C++仅
去除路径和自动inline编号差异后相同。顶层独立HC_pre的变化不在该调用链内。

但两次PyPTO来自不同独立worktree，67份IR和73份核C++中分别有29/35份在上述
有限归一化后仍不同；旧流程先5次预热及20次Event再采3窗，新流程三角色轮换、
3次图预热后采5窗。未受控隔离编译结果、cache/频率等变量，不能把25.02μs
归因优化或普通波动。历史905.08保留，新值仅作为当轮基线，仍按同轮配对验收。
此前表格换用当前值却未充分解释轮次差异，现补齐可复核来源；无新增NPU测试。
脚本及证据：projection_branch_evidence/audit_multitask_baseline.py与
MULTITASK_905_VS_880_AUDIT.json，不计算文件散列。

### 2026-10-18：阶段诊断完成，归档失败候选并更新七档现有数据

阶段任务task_20261018_115704_29584807981 exit0，原动态SPMD与4辅助核候选
三轮A→B→A、数值/完整state/cache、保护区及图重放通过。以各版最早BEGIN为零，
三轮阶段最晚notify端点中位数：Indexer norm195.30→148.46μs、主压缩投影
287.50→182.82μs，但KV投影164.02→288.10μs、Indexer query量化
245.16→325.84μs、key/scale275.18→350.72μs，Score Vector结束
542.98→621.14μs。主Q_B245.38→245.54μs基本不变。

候选把KV排在4辅助核的weights/query Hadamard/compact Hadamard后面，AIV
仍先等待KV后处理才做Indexer量化，因此较晚KV阻断关键消费者。主压缩producer
提前不等于其consumer能提前；约75～80μs延迟沿后续链传递。这些是插桩就绪时刻，
不是阶段纯数学耗时，最终输出插桩902.38→973.54也不用于L1性能验收。
完整原始时钟、复现脚本及PHASE_CLOCK_RESULT_V1.json保留在候选过程目录。
下一版应同时安排两侧producer/consumer，不能只扫4/8辅助核。

候选在正式动态目录开发并完成验证后，34份源码完整保存于source_candidate，
原33份源码位于source_reference。核对正式4个修改文件与被测快照相同、原版与
HEAD相同后，仅恢复本轮4文件并移除已归档helper。正式目录恢复12aed06a的代码，
不保留失败策略接口；CPU和控制图历史复现入口改为指向候选快照，不重跑已有验证。

用户随后要求七档Native/分task/静态/动态对照，B28缺项才补测、静态只用现有数据。
生成dynamic_full_pipeline_20261017/LATEST_SEVEN_CASE_PERFORMANCE_20261018.md
及同名JSON，逐项从原始result读取并复核中位数。本次没有启动新NPU测试。
正式B16最近同轮分task869.78/动态889.40；静态最新781.94为独立轮次，同轮
分task880.06。最新失败候选947.16单列，避免替换正式版本或混用基线分母。
B28已有优化Native1701.75、分task1312.22、动态1337.28；静态无成功记录，
按要求留空。动态B28仍是9d5327e5的记录，明确当前新增QA小档不作用于T168、
最新依赖环境未重测；其余Native沿用10月12日历史环境。完整L1七档30%目标仍0/7。

### 2026-10-18：七档新增包含AICPU的多task外部跨度，仅补测B28

用户追加外部可见的多task完整耗时。读取Simpler原始schema确认L1 aicore_tasks
第六列为receive_to_start增量，不是AICPU完成时间；现有L1缺少scheduler/lifecycle
边界，不能从这些列伪造完整AICPU跨度。采用已有校准口径：PyTorch profiler所有
设备流首个CSA kernel开始至最后一个结束，剔除图外reset/poison，保留AICPU和
AICore整体。它不是incore时长加所有AICPU工作量，重叠时间只计一次。

六档沿用10月12日a00afa9f原分task外部记录，30次中位：128K/B4 624.50、B8
730.00、B16 952.75、B24 1223.75；8K/B16 706.75、B32 1038.375μs。原始CSV
确认每步一个simpler_aicpu_kernel_exec和一个aicore_kernel_mode_0。
这些是历史版本和设备环境，不与当前L1列相减宣称纯调度开销。

缺失B28单独复用整层服务npugraph_ex图测试，task_20261018_122158_10688028513
队列device8、exit0。当前正式分task源码、独立PyPTO memory-reuse及Simpler
参数扩容环境；5次预热、20次Event、独立30次profiling。外部跨度中位
1357.75μs，范围1312.50～1438.75；Event中位1366.26μs单列。
固定图输出、Top-K、六份cache/state及保护区检查通过，PTO服务派发计数3，
无Native fallback。每步两类kernel各一条，完整跨度统计无reset混入。

所有复现脚本、实际环境、源文件快照、CSV和原始report位于
dynamic_full_pipeline_20261017/multitask_outer_profile_20261018。
七档MD/JSON新增外层列并保留旧六档日期/B28本轮设备；其余Native/L1/静态没有
重新测试，B28静态继续留空。目标仍是动态单SPMD对同轮分task完整L1降低30%，
新增外层指标不更改验收分母，目标未完成。

### 2026-10-18：停止投影分核微调，转向大阶段硬件证据

依据上一轮KV阻塞证据，在正式动态源码做第二版：head权重/KV恢复24核，4辅助核
只做Hadamard、20核做Q_B，各核随后做原owner主压缩；AIV先完成Indexer量化/
系数/key/scale再处理KV。动态shape和全部数学保留，450项控制检查及完整符号
编译通过。task_20261018_123227_5932334047 exit0，A→B→A、输出/Top-K/score、
完整cache/state、保护区、代际和图通过。

同轮五窗完整L1：原多task878.08、原正式动态SPMD878.64、候选882.20μs。
没有收益，不继续扫4/8核或扩档；正式源码已恢复，完整34份候选及33份原版、
所有结果留在projection_hadamard_overlap_evidence。没有用上一轮947.16的
跨轮差值冒充本版收益。

用户明确要求抓大放小。转向Score约230μs和Attention约160μs的大阶段。
先试CPU保留真实CCEC对象核对资源：动态AIC .text50200B，entry2048B、主体
48152B；固定stack record32768等于编译预算，不是实际spill证据。CCEC拒绝
print-after/###诊断选项，save-temps未给出汇编，停止继续猜选项。
命令、对象本地路径及失败在compiler_resource_capture_20261018。

新增indexer_score_hardware_evidence：完整CSA正式权重运行后，复用真实query/
系数/cache/metadata与完全相同Score数学helper，独立24组MIX测3步PipeUtilization。
测量前后须对完整pair/score allocation逐bit匹配完整CSA，保留两槽协议；只补
供数链硬件证据，不做理想GEMM、不以局部吞吐替代完整L1。首次有效CPU符号编译
通过，13根Tensor、单MIX/sync_start；task_20261018_125149_165487929633已在运行，
尚不预记设备PASS或硬件瓶颈结论。正式动态七档30%目标仍未完成。

### 2026-10-18：Score硬件归因通过，验证WS搬运方式

任务task_20261018_125149_165487929633 exit0；完整CSA A→B→A及独立Score
eager/graph/profiling后三次完整pair/score allocation逐bit一致。保留全部三步
原始CSV。AIC MTE1活跃中位154.814μs、矩阵103.951μs、FIX133.343μs、
MTE2 77.640μs，AIV Vector99.168μs。独立图外Event308.340μs包含外调度，
不作为整层L1；pipeline时间可重叠，不相加/相减换算收益。

用户要求核内double buffer/cache预取。当前Score已存在Key两槽及提前一个
面板到L0B，旧WS后移/Key成对预取也已验证，不重复当新方案。新证据指向
进一步检查WS的48KiB FP16 NZ→L0B：底层当前使用load3d，尝试等价load2d
分形转置搬运。只改变16处搬运实现，原数学、地址、事件及双缓冲保留，
去helper和还原调用后实际C++逐字符相同。两次CPU准备错误已记录，实际CCEC
cpu_v2通过。任务task_20261018_130455_238129612172待设备结果，先验证实际
中间Tensor的完整Score逐bit结果，再看硬件流水；未改正式算法和共享依赖。
过程目录parallel_20261018/score_load2d_evidence；七档30%目标仍未完成。

### 2026-10-18：WS load2d无整体收益，Q_B实现真实双L0槽

load2d任务task_20261018_130455_238129612172 exit0。两种搬运的Score三次
完整pair/score逐bit一致；独立图Event原329.790/新328.700μs，所有慢样本
保留；PMU kernel Duration287.120/286.780μs，MTE1活跃154.691/152.870μs。
不是完整CSA L1或交错配对证据，没有足够收益，不合入、不扩测七档。

按用户double buffer要求，继续检查主Q_B供数：原N256/K256 Right已占满
64KiB，stage2并没有真实双Right。新候选保留N256与动态M128尾行，K改为
八个128整数块，先预取下一片再消费当前片；不改变T/B动态性或列owner。
CPU v1遇独立if之间Tile定义作用域问题，v2显式排空末K后完整符号编译通过。
实际生成Left0/16384、Right0/32768；八次MMA按槽交替，只等当前槽MTE1
事件，Acc的CompactMode::Normal及动态有效行保留。L1仍复用，未把地址
检查当实测收益。原32模块不变，完整源在qb_k128_pipeline_evidence。
任务task_20261018_131417_2840949760待设备：128K/B16正式权重，原分task/
正式SPMD/候选五窗完整L1；先经过全部精确、图、保护与代际检查再计时。

### 2026-10-18：Score转置片上布局完成符号编译，提交完整CSA对照

load2d无整体收益后，改评估更大的数据流变化：QK与WS同时转置，Query48KiB
和系数12KiB常驻Right，WS FP16中间值48KiB改走Left非转置读取，两Key
8KiB双槽也在Left。原INT8点积、FIX/FP16/ReLU及WS K顺序保留；GM双槽
改为[1024,8]，AIV片上转回query-major后继续原scale/TopK。交接区增加
393216B；没有新根、Host metadata或全局barrier。

覆盖G6/N64/1024/prefetch1/toL0/balanced策略区间，B/T和device数据完全
动态；其余保留原路径。AIV布局小探针通过；完整v1因Left create/extract的
TileView不同被类型检查拒绝，v2取消不可能分支并用真实首Key初始化后，
完整CPU通过130根/58动态轴/一个MIX。实际Right0/49152、Left WS0与
Key49152/57344在64KiB内，首Key临时0仅在WS加载前使用；保留原事件保护。

原16卡整机任务结束后，Q_B已开始运行。Score布局任务
task_20261018_132551_345883327524也已提交：只测128K/B16一个代表，
先精确功能再同轮三方五窗完整L1，不叠加Q_B候选。结果尚待收齐，不预记
设备PASS和性能收益。全部过程在parallel_20261018/score_transposed_layout_evidence。

### 2026-10-18：Q_B双L0候选完整通过但回退4.71%

任务task_20261018_131417_2840949760 exit0；原精确数值、Top-K、score、
六份cache/state、A→B→A、图、保护区及代际检查全部通过。同轮三方五窗
完整L1中位为原分task885.24、原正式SPMD854.66、K128双L0候选894.94μs，
相对正式回退4.712985%，五窗均更慢。没有剔除样本，不采用、不扩测七档。
真实双槽和current-slot等待已确认，但同时存在K块/MMA数量增加、L1仍复用
等变化；当前完整L1不能把40.28μs全部归给某一项。源/CPU/设备结果留存。

Score Key-major候选已经过完整A→B→A精确/图检查，正在编译同轮正式参考
用于完整L1采样。精度通过不提前等同性能收益，正式目录仍为原已验证算法。

### 2026-10-18：Key-major首版数值通过但回退，先定位大段成本

任务task_20261018_132551_345883327524 exit0，完整精确输出/Top-K/score、
六cache/state、A→B→A、图、保护区、代际全部通过。同轮五窗原分task
874.56、原正式SPMD880.26、Key-major候选1533.96μs，回退74.262%。
未合入、不扩七档，不把数学等价/双槽容量通过当性能收益。

[1024,8] AIV TTRANS按真实PTO-ISA源码满足向量路径的stride对齐条件，
不能无证据称其进入Scalar尾路；AIC M/N方向、WS的Left搬运、GM小列写回
和AIV布局恢复都仍可能影响结果。下一步只采相同Score helper的PMU，
不靠整层数字猜具体根因。原算法及共享依赖未改，七档30%目标仍未完成。

### 2026-10-18：Key-major回退定位到FIX/交接方向，准备整行写回

同helper任务task_20261018_133558_38898924592 exit0，三次完整pair/score
逐bit一致。三步PMU中位AIC MTE1 63.471、矩阵112.801、FIX876.487、
MTE2 549.857μs；AIV Vector108.386、MTE2 707.563μs。原路径独立
PMU约MTE1155/矩阵104/FIX133/Vector99，非同轮端到端配对，不能据此
精确拆分654μs回退，但大幅变化明确在FIX/交接方向，而非AIV向量工作量。

只做一项修正消融：WS本来已算64×16，原候选只写64×6且GM行stride8，
改为完整64×16、64B整行写回，GM stride16。AIV仍load前8列，32KiB
及原transpose/scale/sort/事件保持；不扩UB、不增MMA、不改前六列数学。
原v1源码在首次设备source及提交57ced237保留，当前两文件改动和
aligned_store_v2.diff可核对。aligned_cpu_v1完整符号编译正在执行，
尚无该修正的设备或性能结论，不提前将窄列假设称为根因。

### 2026-10-18：64B整行修正版通过，但仍有约24%回退

任务task_20261018_134422_286630303 exit0，全部原精确/状态/图/保护/代际
通过，同轮五窗原分task880.58、正式SPMD863.32、整行候选1072.02μs，
相对正式回退24.174%。前轮1533.96不得作为验收基线；当前
仍不采用、不扩七档。已确认AIV生成C++去注释与前版相同。

下一步只检查一个新的FIX减量接口：把8份64×16 FP32 WS结果汇集于
32KiB Acc后一次写回，QK96KiB加它恰128KiB。矩阵/FP32累加次序不改，
不增加GM暂存；需要确认Tile slice/assemble能否保持Acc别名及真实地址，
不能从容量计算直接上卡。小探针在score_transposed_layout_evidence/
ws_output_pool_cpu；尚无完整CSA或性能结论。

### 2026-10-18：Acc输出集合小探针通过，完整内存规划待改进

独立小探针8个WS子片实际共享32KiB Acc，8次MMA/1次TSTORE且无
TINSERT。完整动态pool候选却分配163840B而超过Acc131072B上限。
MemoryReuse后QK的视图98304B位于公共mem_acc_382，但该公共slot
受早期Q_B大tile影响保留131072B；WS的mem_acc_502另占32768B。
当前仅发现全局slot大小与本阶段真实视图不同，仍需最小复现核对生命周期
与别名，不能放宽平台容量、删同步或直接硬改地址。没有为此上卡。
所有NPU任务已终态，正式动态源和依赖均未改。整行已测版独立保存于
source_aligned_v2，source_candidate保留未编译成功的pool源码供后续接续。

### 2026-10-18：WS Acc pool 真机失败方向收口，常量偏移修正完成

命名 Scalar 的 slice 起点不能折叠，导致非 slice 原地结果继承动态 MemRef，
从而禁用现有安全子区间复用。算子先把等价表达式内联到 slice/assemble，
完整符号 CPU 编译通过：QK 96 KiB + WS pool 32 KiB，两个 FP32 pool 写回，
130 根、58 个动态轴、一个 MIX/sync_start，AIV 生成 C++ 去注释不变。

设备 task_20261018_141647_127665331649 exit0：A→B→A、精确输出/TopK/score、
完整 cache/state、图重放、保护及代际通过。同轮五窗中位原分task883.38、
正式SPMD873.50、pool候选1082.32 μs；相对正式回退23.91%。不采用、不扩七档。
该轮仍使用编译器修正之前的 inline-offset 算子变体。结果与源码完整保留，
不能把容量通过或 TSTORE 16→2 当作性能收益。

之后在独立 PyPTO 的 InitMemRef 中仅折叠已证明整数常量的 slice 起点，并同步
改写调用参数；不冻结 loop IterArg/真实 metadata，不去掉动态别名安全门槛。
主线派生环境的新最小回归失败，修正后 InitMemRef/L0 共65项通过；原先失败的
命名 Scalar 完整 CSA 源码也通过无样例 shape 的完整 CPU 编译。修正本地提交
`d46a813ca`，上游 issue #3032 附复现和建议补丁，没有提交 PR或修改共享安装。
证据位于 parallel_20261018/score_transposed_layout_evidence/acc_pool_constant_offset。

### 2026-10-18：组件问题及时上游登记与后续优化取向

用户要求组件问题不要积累本地；只提 issue、可附建议代码，不提 PR、不加 bug
label。随后补充：Simpler 非 bug 能力增强继续本地保留。据此新增 PyPTO
#3032/#3033/#3034/#3035，补充已有 #1908、#3027 及 PyTorch #162870；全文
及远端回读在 parallel_20261018/upstream_followup_20261018。原 #3014～3017、
#3020、#3026～3028、PTOAS #1622 继续对应已有问题，避免重复。Simpler参数/ABI
扩容不新提 issue；构建依赖已有#2331/#2340。

本次回看同轮正式动态889.40/静态792.78/多task869.78 μs，动态差96.62 μs。
静态不是仅把dynamic轴替成常量：保留每核任务依赖排程与20核Q_B等不同分工；
动态主要统一阶段，并有运行时范围/尾块控制。真实分配还发现融合相关双槽
退化，但现有证据没有把96.62 μs逐项分解，不以Scalar或寄存器spill猜测替代。
4辅助核候选主压缩虽提前约105 μs，KV/Indexer消费者反而迟到，量化约晚81 μs；
说明仅生产者前移不够。WS与双L0等局部改动多次回退，后续按用户“抓大放小”
要求聚焦完整链的分工、消费者顺序和实质流水，不再围绕局部写回次数扩测。
七档30%目标仍未完成，正式33模块算法未替换，没有新增七档性能结果。

### 2026-10-18：整层区间编译候选，七档同轮对照已启动

回读确认4/20核第二版已无收益，未重做分核微调。本轮在正式动态十个模块中
接入仅由已有shape决定的编译事实：QA/KV原M32/M64区间、T的8行对齐、原DQ
48-AIV整波head分组、Attention原144行事件次序。默认开关关闭，不改变数学
选择、phase依赖、Native dtype/缓冲、130根ABI；T/B及device Q/L/页表仍动态。
T>512时KV继续原分段选择，未把大T的最后短段误当成M64完整段。绑定层校验
事实与实际T，未新增D2H、meta kernel或Host数据值读取。

原DQ源码Scalar前缀与新事实在T=1..4096一致，包含128/144/512边界；这不是
性能模型。长历史M32/head16与短历史M64/head32完整CPU编译通过，均58动态轴/
一个MIX/sync_start。原版也用相同独立PyPTO重编对照。长档PTO AIC 4358→3970行、
AIV10139→9613行，主要为裁掉未选分支；不等于运行数学或性能减少相同比例。
HC Right仍单槽，不能宣称本轮已修复此前三处双缓冲退化。

七档各自提交task_20261018_150204_*，完整ID/命令在
parallel_20261018/shape_region_codegen_evidence/SEVEN_SUBMISSIONS_V1.json。
全部使用冻结源码、正式权重、同进程三方、五窗完整L1；目前任务运行中，
尚无新设备PASS或收益结论。默认正式策略未启用新开关，不替换旧七档表。

### 2026-10-18：区间编译七档通过但没有缩小长B16主要差距

七任务均exit0。所有原PTO精确输出/TopK/score、cache/state、A→B→A、图、
保护区和代际检查通过。各档同进程五窗中位，依次为多task/原正式/候选（μs）：
128K B4 508.42/555.00/543.04；B8 650.12/673.14/673.30；
B16 874.02/886.36/887.94；B24 1164.68/1174.46/1190.60；
B28 1301.04/1337.88/1332.90；8K B16 677.46/698.58/682.32；
8K B32 1021.64/1008.02/989.64。

三档相对正式改善1.82%～2.33%，其余接近或退步，目标仍0/7。尤其长B16
没有变化，不能把静态约97μs优势归给此次裁掉的分支，也不能把源代码规模
下降当性能收益。候选未设为默认，核对当前十文件仍等于冻结候选后恢复正式
源；候选/原版/全部原始五窗和检查均在shape_region_codegen_evidence保存。
不继续扫区间开关或给B4/B16制造白名单，未重复Native/静态或追加S1/S5真机。

下一阶段改为直接对齐静态/动态完成端点。已核对历史静态B16布局：72核独占
128个INT32，最大dispatch_seq=22，全部已发布端点可在每核第16..107字保留
四字段时钟而不碰word0。真实静态成员名称与seq从layout读取，不按源码排列
猜依赖；参考用例本身仍读取device records/waits。新诊断须保持数值和其它
保护区检查，插桩结果仅归因，不计入性能验收。当前尚未提交该诊断任务。

### 2026-10-18：静态/动态配对端点诊断通过，差距在Score与Attention段扩大

任务task_20261018_152604_395917310445 exit0。128K/B16/S6，正式第4层权重、
合成历史、同进程同输入地址；动态正式33模块与历史静态b16_cached_row_deps
在A→B→A三轮均通过精确输出/Top-K/score、cache/state、图重放、保护及代际。
本轮只给两侧原TNOTIFY增加相同四字时钟，不改数学/等待/FFTS/业务fence。
静态每核128字槽内最大seq22，时钟末字107；只豁免实际时钟字，其余保护保留。

动态/静态完成端点三轮中位（μs）：HC归一化61.80/72.32，QA81.20/95.56，
Indexer query RoPE159.82/182.52，Score AIV550.46/520.22，Top-K569.92/531.24，
Attention AIV780.76/706.30，OB894.14/818.62。前段动态反而提前，到Score与
Attention差距扩大；应优先核对这条完整依赖链，不能再泛称metadata或动态
shape是主要原因。动态1619/静态794条时钟记录，插桩总扰动并不相等；端点
含等待/搬运/缓存维护/探针，不能精确拆分此前未插桩96.62μs差距或报新性能。

源码确认动态AIV先做主Query反量化、主压缩pool/state/cache后才进入Score，
这些主Attention输入不是Score直接前驱。Score两槽ACK使迟到消费者也会限制
AIC推进；静态按每核成员依赖前进，不能视为动态源码的常量替换。Attention
目前只有完整AIV完成端点，仍需区分入场等待与核内流水，未声称指令根因已定。

证据与复算脚本在parallel_20261018/static_dynamic_phase_alignment_20261018。
只使用静态各核末成员确实发布的端点；Attention AIC/最终HC_post缺完整时钟，
不伪造。正式算法未改，未新测七档性能；目标保持0/7，后续优先处理整层排程
和Attention流水，不重做已经无收益的分支裁剪或局部写回微调。

### 2026-10-18：Attention索引缓存补齐没有收益，转入完整前段协同重排

静态B16实际代码已用UB缓存索引/页表，而动态仍逐行GM Scalar读取。本轮
补齐8KiB页表预算的Vector地址候选，保留负索引、列界、物理页和INT32边界；
超过预算走原路径。1008项实际源码地址控制检查、完整58动态轴/130根/单MIX
符号编译通过。任务task_20261018_154822_5768851355 exit0，128K/B16/S6
精确/状态/图/保护/代际全通过，同轮五窗多task867.40、正式866.12、候选
880.56μs，回退1.67%。不采用、不扩七档；候选和原始数据在
parallel_20261018/attention_index_cache_evidence，正式代码已移除该开关。

随后在正式动态目录实施COOPERATIVE_PREFIX：24核主压缩先生产，4核compact/
query Hadamard与20核Q_B并行，KV仍由24逻辑worker轮转分担。AIV同时将
主压缩pool/state/cache前移到Q_B期间，原KV后处理移到本核Score之后；
无相关请求/行块的角色不等无关阶段。两个新增helper模块完整中文说明，
不删共享prefix读完依赖或A3缓存维护，不新增Host metadata/工作区/task。

完整CPU编译通过；450组实际源码控制检查覆盖发布、等待来源、依赖无环、
Q_B与KV分工，不能替代设备数学/FFTS。已提交128K/B16、128K/B28、8K/B32
代表对照，任务task_20261018_155959_100159021669、100164212022、10017467588。
源码与命令保存在parallel_20261018/cooperative_prefix_evidence；默认关闭，
尚无该协同候选的设备/性能结论，不提前记PASS或改变0/7目标状态。

### 2026-10-18：协同前段七档获得一致改善，完成S1/S5及同图回归后默认启用

七档完整精确/状态/图/保护/代际均通过，各自同进程五窗中位（多task/旧动态/
协同动态，μs）：128K B4 517.56/525.36/515.00；B8 641.60/704.60/658.86；
B16 880.26/872.00/847.98；B24 1160.50/1170.00/1142.04；
B28 1313.04/1336.98/1290.26；8K B16 673.78/693.66/673.56；
B32 989.16/995.64/971.74。相对旧动态降低1.97%～6.49%，相对原多task的
30%目标仍0/7；B8仍慢于同轮多task，短B16接近持平，不夸大这轮收益。
七档原始样本、历史静态来源与完整结果见cooperative_prefix_evidence/SEVEN_RESULTS_V1。

真实B1/S1任务task_20261018_161630_168463818084、B5/S5任务
task_20261018_161630_168461724394均exit0。未补成S6；相对旧正式动态SPMD
的输出、Top-K/score、完整allocation/保护和A→B→A图重放精确通过，含S5
四token Hadamard尾块。不以此替代Native/整网token精度。两个前置测试脚本
失败（漏接--device、缺失纯绑定helper模块）在算子执行前终止，完整保留，
不称为组件问题。task_20261018_161516_163215116226的同图metadata六次与
padding四次重放也通过；empty-rank、mixed-query和连续轨迹未在此轮扩测。

据此select_policy默认启用COOPERATIVE_PREFIX，关闭路径保留供消融。新核函数
AST与被测快照一致，七档和S1/S5默认constexpr逐项相同；采纳时只改Host默认
选择、完善中文总览并删无用import。没有启用失败的Attention索引缓存，也未
更改共享依赖。证据/采纳审查/失败记录均在cooperative_prefix_evidence。

### 2026-10-18：协同前段新端点完成；NZ写回负例收口

任务task_20261018_163026_21993375548 exit0，新默认动态与历史静态完整数值、
状态、图和保护检查通过。Score AIV完成端点514.36/514.62μs，Top-K
534.60/526.60μs，Attention729.58/694.16μs，OB846.98/812.62μs。
动态1619/静态794条时钟，扰动不等量，不能把端点差精确拆成完整L1收益。
证据在cooperative_prefix_evidence/h131072_b16_phase_alignment_v1。

Score转置+NZ写回task_20261018_165718_32204744106 exit0，完整原PTO精确、
Top-K/score、状态、A→B→A、图和保护检查通过。同进程五窗完整L1：原多task
884.40，正式动态855.58，候选1412.62μs，回退65.11%，不采用、不扩七档。
候选同时改变转置数据流与AIV布局读取，不能将整层回退全归因于NZ store。
测试侧hook只换16个匹配的Score Acc写回，不改共享编译器。API只读限制
另提PyPTO #3036（无标签、无PR），已向issue补充负收益终态，不宣称必要bug
或加速。全部尝试、最小复现、源/结果/回读在score_nz_writeback_evidence。

### 2026-10-18：继续抓大放小，恢复静态已有的动态行依赖

Score端点已接近静态，优先处理Top-K→plan→Attention仍扩大的区间。
新候选从正式ops动态源修改：Top-K按实际query根获取Score writer，八行plan
只等对应Top-K行，Attention只等本组q=group+k*24的plan owner；不固化
sample，不改数学/页保护/FFTS，不省业务缓存获取，原共享cache依赖保留。
原静态row_dependencies同样收窄这些边，本次把映射改为实际device group
前缀、有效根数和动态T；B16/G6每query根只有3个不同AIV写者，旧版等48个。

107457项根依赖、4992项plan依赖、600项mixed映射离线检查通过；数学AST
和未涉及模块逐字核对通过。完整CPU保留58动态轴、130根、一个MIX，通过。
已提128K/B16、128K/B28、8K/B32三个代表任务，暂未记设备或性能PASS；
记录在attention_dependency_evidence。目标仍0/7，不重跑已完成的旧版验证。

### 2026-10-18：动态行依赖七档通过，正式采用

完整PTO精确/Top-K/score、cache/state、A→B→A、图、保护区与代际均通过。
同进程五窗L1（多task/旧动态/新动态，μs）：128K B4 509.52/514.56/496.64，
B8 642.06/684.98/641.24，B16 884.16/852.94/844.84，
B24 1160.66/1154.08/1121.36，B28 1314.46/1302.44/1282.28；
8K B16 675.98/673.28/643.28，B32 967.62/964.04/951.80。
相对旧动态降低0.95%～6.39%；相对原多task降低30%仍0/7。

B1/S1、B5/S5对旧动态的精确及图/状态保护通过，六次metadata、四次padding
同图回放通过。没有扩称mixed-query、空rank、连续状态或Native/token通过。
十个验证任务均exit0，全部原始窗口、冻结36源、脚本/诊断/任务登记保存在
attention_dependency_evidence。五个正式改动按暂存区逐字核对实际被测源。

配对端点task_20261018_174436_80108314392也exit0：Score AIV518.58/522.02，
Top-K534.78/528.78，Attention722.46/733.26，OB840.30/846.70。
本轮Attention已不再稳定落后静态，不沿用旧35μs结论。probe次数不同且
静态端点波动，不能把这些差值当未插桩性能或精确拆账。另证明静态plan的
16成员中12个真实数据写者均发布，四个未发布成员的block12～15循环为空；
全成员退出仍不可观测，payload端点动态/静态548.70/536.36μs。

继续前置几何候选：借既有attention_indices待写前缀保存同轮请求/因果/页
判断，由原ROPE phase发布，Top-K读取后plan再覆盖。CPU首版GM Scalar与
MTE3混写被既有检查拒绝，改为UB写并统一MTE3发布后完整CPU通过；这是算子
表达修正，未绕过检查或改编译器。320案例6919query离线几何/保护通过，
两个代表NPU任务已提交，尚未记PASS。过程在attention_preplan_evidence。

### 2026-10-18：前置几何七档完成，保留六档收益并转向区间选择

完整PTO精确、Top-K/score、cache/state、A→B→A、图、保护及代际通过。
同进程五窗L1（原多task/已采用动态行依赖/前置几何，μs）：128K B4
508.40/499.20/491.32，B8 645.24/646.70/653.38，B16 880.18/844.08/821.52，
B24 1179.58/1125.94/1108.30，B28 1294.58/1281.74/1265.24；8K B16
659.18/667.76/660.26，B32 982.96/974.32/947.82。六档改善1.12%～2.72%，
B8回退1.03%，4/5配对窗口更慢，不能忽略或只取最小值。全部原始五窗见
attention_preplan_evidence/SEVEN_RESULTS_V1.json。S1/S5对旧动态精确/图回归、
六次metadata与四次padding同图回放通过；不扩称Native、整网或连续轨迹。

用户明确性能优先，允许适当增加档位，也要求避免为少数case堆砌难维护代码。
因此不整体启用前置几何，也不丢掉六档收益：下一候选在policy集中选择执行
时点，数学、页保护及同步协议共用。计划沿已有G6小M16并行区间或T>=64启用，
中间范围保留后置；T64只是至少两个完整M32行块的初始分界，须验证T30/T66等
边界，不称为理论最优。七档是回归点，不做测试点白名单，device Q/L仍动态。

### 2026-10-18：几何后移与独立query WS分拆终态，均暂不采用

attention_preplan_overlap_evidence把同一几何从ROPE发布前移到主压缩消费后、
Indexer Hadamard等待前，复用原全体IDX_SCALE获取。三个任务均exit0，完整
精确/图/状态保护通过。同进程五窗（原多task/旧动态/候选，μs）：128K B8
653.26/642.34/646.86，B16 888.44/848.22/857.46，8K B32
975.60/943.68/953.62，分别回退0.70%/1.09%/1.05%。不采用该位置、不扩七档。
这说明合法依赖及理论重叠不能替代实际完整L1证据。

score_ws_query_split_evidence保持QK M384/N64/K128及两份8KiB Key预取，只将
独立的六条query拆成两个M3/K192 WS。两份Right各24KiB与Key合计64KiB，
每条query的64head仍一次MMA，保留FP16边界及AIV排序。CPU前两次分别因跨
if的Tile定义域和valid_shape类型不一致失败，改算子表达后通过，未改编译器。
实际C++的两半MTE1事件独立；记录不等于已证明硬件搬运重叠收益。

B16任务task_20261018_181940_220242617174 exit0，完整精确/图/保护通过。
同进程五窗原多task883.40、前置几何参考832.42、WS分拆825.00μs，中位改善
0.891%；4/5窗口更快，但3/5配对差值不超过1.3μs。新增两套WS搬运/写回的
复杂度暂不值得凭这一点扩档，保留候选，正式Score恢复原体。两个实验的
TERMINAL_RESULTS.json包含原始窗口，均未称为30%目标达标，目标仍0/7。

### 2026-10-18：按工作量区间选择几何执行时点，保留六档收益

正式增加一个early_attention_geometry策略：已有G6小M16并行区间或T>=64
使用前置，其余保留原Top-K/plan推导。T64代表至少两个M32投影行块的初始
分界，不是最优阈值；选择集中在policy.py，entry只传constexpr，两条路径
共用页检查、候选数学和同步协议，无新Host数值、无新根或额外task。

新五点任务均exit0，完整精确/图/状态保护通过。同进程五窗完整L1（原多task/
已采用动态/区间候选，μs）：128K B5/T30 584.78/571.26/544.50，
B11/T66 765.72/750.90/731.56，B16 882.78/842.88/832.62；8K B32
968.78/968.12/951.06。新增边界及代表分别改善4.68%/2.58%/1.22%/1.76%。
128K B8为650.52/642.32/646.38，仍记录0.632%慢且4/5窗口慢，不删数据。
核对该关闭路径实际AIC/AIV可执行C++与原版完全一致，仅注释不同，因此
不把这4.06μs解释为新增几何计算成本，也不追加孤立shape分支。

14项逐行C++比较覆盖所选策略：小T30与旧T24、中T66/T96与旧T96、短B32，
以及补CPU的大T长历史和中T短历史均与已测前置数学体一致。只忽略整行注释，
不做hash；CPU未注入等待时与同一hook前原始C++比较。四次完整符号编译保留
58动态轴、130根、一个MIX。复用既有前置七档/S1/S5/metadata/padding与原
后置验证，不重跑未变化的七档，也不称本区间候选已重新测完所有动态场景。
正式36模块执行AST与冻结候选一致，仅完善文档后采纳；全过程在
attention_preplan_policy_evidence，30%目标仍0/7。

顺带修复旧可选plan探针漏接行依赖接口：显式incore、已准备状态的信号Tensor、
generation与constexpr全部绑定。独立v3四组页/异构Q/dummy/尾部保护及两种
页前缀列范围检查通过。v1漏初始化runtime、v2漏显式constexpr的测试适配失败
完整保存；未改依赖库，不将这些脚本错误误报组件bug，未把探针当性能证据。

### 2026-10-18：Attention同lane统计本地化突破旧编译阻塞，七档完整正确

重评此前只到CPU的768B mi/li本核三槽：其写读都属于同一AIV，取消GM
写回与回读，保留全部softmax、在线m/l/oi、SWA/压缩页访问和FFTS/drain。
旧33825种mask时序模型仍适用，接口暂保留两个未用GM形参。

v1新独立内存复用树仍报Vec255104B>188416B。定位到动态slice→reshape
非静态地址使本地安全门槛禁用Vec细分复用，不是环容量自膨胀。v2在UB
显式extract两行各128B再reshape，完整CPU通过，实际Vec上界163872B，
环768B，58动态轴/130根/一个MIX保持；没有修改依赖或绕过正确性门槛。

组件界限：当前kernel调试树的最小A3探针中，row=1/4/5/1误读row0，extract
五次全对。主线14736f65的派生树CPU对照已保留动态偏移，不能重报为main
仍丢地址；其PTOAS0.66产物另有动态地址别名的MTE2→MTE3同步缺失。
32行独立PTO与24行extract控制直接重编译确认，已补现有PTOAS #1620
comment6249288578，含完整复现与建议，无PR/标签。未给这个主线输入做新
NPU数值对照，不混用旧分支硬件失败。所有过程在attention_local_stats_evidence。

七档完整PTO精确/Top-K/score/状态/图/保护/代际通过。同进程五窗完整L1
（原多task/原动态/本地统计，μs）：128K B4 507.46/511.12/488.42，
B8 662.34/650.82/643.22，B16 884.38/831.10/823.26，
B24 1174.22/1119.14/1120.28，B28 1317.86/1254.60/1271.88；
8K B16 675.44/648.52/669.78，B32 983.80/943.38/934.70。
四档改善，三档回退，不全域采用。S1/B1、S5/B5对旧动态的精确/图回归、
六次metadata及四次padding同图回放通过，不代表Native或完整生命周期。

按用户允许增加策略区间的要求，继续准备long G6且T<=96、short G2且T>96
的局部统计选择，其余保留GM；沿现有O_A tile分界，不做七点白名单。
开启路径CPU代码已与被测版本一致；关闭路径首版仍携带无效loop Tile，
未声称与原版代码一致，正在显式断开未启用缓存phi后复核。不预记采纳。

另只读复核现成的协同前段端点：MAIN_OUTPUT最晚约244μs，IDX_SCALE最晚
约277μs，QUERY最晚约300μs；三轮QUERY前同核区间中位20.87/20.97/21.03μs。
旧DQ穿插Score设计因DQ已与MAIN等待重叠而被降级，但现在MAIN已前移完成，
该抵消因素改变。新依据与8-head偶数Score步设计保存在dq_score_current_readiness_20261018；
仅只读证据与后续设计，未写候选，不把插桩区间当可省性能。

### 2026-10-18：统计区间开关的死循环状态未消除，及时上报PyPTO #3037

首版区间五种完整CPU通过；开启代码与已测本地统计一致，关闭代码却仍保留
768B无用loop Tile并改变UB布局。第二版断开关闭路径的内层phi，三种CPU
仍通过，但纯alias传递仍保留，开/关C++均不能称与各自原实现相同。
尚未给区间版提交NPU，未用重复采样解释掉该代码差异。

主线14736f65派生树72cadb1e5最小CPU复现保留ghost[6,32] FP32、地址128、
768B；其引用只组成if/loop传递闭环，无语义消费者。相关Simplify/DCE未改。
已提交https://github.com/hw-native-sys/pypto/issues/3037，明确是死状态/资源
优化问题，包含完整复现、live-root可达性建议以及while条件、TaskId、attrs/
deps、调用副作用和真实循环状态反例要求。无PR、无标签，正文回读一致。

完整两版与八份CPU结果保存在attention_local_stats_policy_evidence。正式
四个临时文件在与冻结v2逐字核对后恢复25de3f32，未触碰其它会话改动。
四档收益和三档负结果全部保存，待修复死状态消除后再做区间边界验证；
这是尚未完成的工作，不是放弃区间优化，也没有缩小30%逐档目标。

### 2026-10-18：死Tile传递闭环修复，收口长历史本地统计区间

在已有独立PyPTO内存复用树修复EliminateDeadYieldSlots：纯别名与loop/if
通道构成依赖图，从真实运算、条件、描述符、返回和调度用途求可达性，
同时删除无消费者通道、别名与原始未初始化tile.create。不推断其它调用
纯度，保留TaskId、Submit、零次迭代初值及while条件。新增7项回归与既有
Simplify/依赖/phase fence合计179项通过，完整最小复现ghost分配已消失。
初次用例编写错误和子进程导入共享树的两项失败均保存，未改断言掩盖失败。
完整建议patch已补PyPTO #3037 comment6249851423；无PR/标签。未动共享安装。

五类完整符号编译与三类关闭路径对照通过：58动态轴、130根、一个MIX。
关闭路径原C++相同，仅忽略整行注释；新编译器会删其它阶段死分配，不能
宣称跨编译器逐字相同。五类T操作token计数只有TASSIGN减少，实际Vec声明
地址末端仍163872B；这些不是生命周期峰值或设备性能结论。

七档加T90/T102两个边界全部完整PTO精确/TopK/score/状态/A→B→A/图/
保护/代际通过。五窗同进程L1，多task/原动态/区间候选μs：
128K B4 519.14/520.90/489.60；B8 653.16/654.90/637.70；
B15 889.30/850.50/839.40；B16 877.84/837.38/827.06；
B24 1174.00/1109.34/1112.14；B28 1308.62/1261.98/1260.42；
8K B16 675.20/652.12/644.30；B17 741.54/738.78/736.50；
B32 983.76/943.34/945.58。原始窗口全部保留，不挑最小值。

最终仅long G6且T<=96默认启用本地统计，长B4/B8/B15/B16相对原动态
改善6.01%/2.63%/1.31%/1.23%；短历史T>96两点+0.31%/-0.24%未稳定，
保留原GM。额外一次CPU核对证明short大T关闭代码等于此次真机已测原动态
C++。B24/B28/短B16本就关闭，实例之间的中位差不能归因本地统计。
正式35个数学/编排模块AST与被测候选相同，Host策略只收窄默认区间，不
复制计算体或把B值写成白名单。S1/S5及混合/连续轨迹本轮未追加测试，
不能扩大S6结论；目标仍0/7。全部证据在dead_state_dce_fix_evidence。

用户再次要求抓大放小。此微调在本条收口，不追加小收益采样；下一步优先
处理静态成员调度仍能覆盖、动态整阶段串行留下的关键链，参考已有端点
及完整结果，不再沿用已经过时的97μs归因。静态跨轮数值只能定位优先级。

### 2026-10-18：主Query穿插Score已完成两类整链验证，收益不足不扩档

直接在ops动态源码提取共享DQ八行/head数学，原顺序入口与新分段入口共用。
原head分组、worker+k*48所有权、512列归约、完整行rsqrt和尾行recip(sqrt)
不变。balanced偶数步及非balanced每步在ACK/scale写后至多推进8head，
余量在本核Score后补齐再发布QUERY；所有输入提前获取，不新增前驱等待。
Score READY/ACK/drain未改，没有以SCORE_CUBE整段完成作为Vector前驱。

两类完整符号编译通过，58动态轴、130根、单MIX，Vec声明末端163872B。
真实Scalar控制抽取核对T1..512/48owner/六种可用窗口，共147456组合有序
覆盖及owner一致；这不代替设备数学或同步。两个真机任务均完整精确/状态/
图/保护/代际通过。五窗同进程L1，多task/正式/候选μs：
128K B16 870.08/827.12/829.22，回退0.254%；
8K B32 1003.26/954.82/942.34，改善1.307%。
长B16参考最后一窗1297.10原样保留，未剔除；其余四窗候选三窗更慢。
短B32五对窗口都更快，但整层收益仍小；不据此引入较大Score耦合，不扩
七档、不扫head块大小。AIV bin增大12228B是事实，不当作I-cache根因证明。

正式五文件在与冻结候选执行AST核对后恢复f222a917，所有共享依赖未变。
全部源码、CPU、147456控制检查、设备窗口、负例和恢复凭据保存于
dq_score_interleave_evidence。未新测静态，当前30%逐档目标仍0/7。
继续检查完整阶段的就绪与重叠关系，不把21μs插桩区间等同于可省时长；
已完成的QB全K常驻及直接L1负例不能重新命名重跑。

### 2026-10-18：复核HC_pre/静态借鉴方向，转向O_A跨M权重复用

用户要求性能表改用（基准−动态）/动态作为展示提升率，已更新完整七档表；
验收仍为原目标的同轮完整L1降低30%，不因分母变更而重定义目标。近期展示
交互没有新增算子性能成果，不能把表格重排计作优化收益。

WS独立归约探针确认普通Vector row_sum不能逐bit替换Cube单K384。正值、
有符号、宽指数三例分别122/226/216处末位不同；54种有限CPU归约次序也
无共同exact方案。未放宽门槛、未做完整CSA性能，正式Score保持原Cube。

连续物理页ND→NZ能力在实际PTO-ISA pinned include存在，但PyPTO当前
不支持显式stride GM view。CPU按canonical视图通过，设备入口则正确
拒绝非连续输入，未执行搬运。已提交PyPTO#3038功能需求和建议，不提PR、
不加标签；不是静默误算bug。测试脚本漏收队列device参数的首轮失败保留。

重新对照hc_pre的M分档减少权重读，与当前大T的O_A形成新候选：128<T<=256
两M128×四N256改为八N128各计算两个M，权重K256共享、各自保留K64链。
每组8个owner与原quant前驱集合不变；动态T和尾行、所有phase不变。原
N-pair共享A、QB resident、HC mix驻留等旧尝试不等同本方案。

正式源码新增默认关闭OA_PAIR_ROWS及中文独立helper；完整符号CPU通过，
130根/58轴/一个MIX。实际Acc0/65536各64KiB、尾CompactNormal和动态store
保留。审计初次混读合并C++的AIC/AIV同名v常量已纠正，未据错误审计上卡。
长B24与短B32任务已提交，功能/性能待终态，不预记通过。全部证据在
parallel_20261018/oa_row_pair_weight_reuse_evidence；详细取舍见HC_STATIC_REVIEW.md。

### 2026-10-18：O_A共享权重完成K128与区间边界验证，保留候选不默认采用

首版K64两档精确/图/保护检查全部通过，长B24完整L1慢0.084%，短B32快
0.545%。随后沿hc_pre显式L0交错思路改成K128：自动外层stage2导致Left
要求256KiB，资源检查正确拒绝；手工Mat current/next轮转则产生不支持
的Mat→Mat TMOV。最小独立CPU例同样复现，已提交PyPTO#3039，附完整
代码、实际不同地址和双活跃buffer要求，以及地址轮转/双bank修改建议。
仅在隔离本地树验证，不声称干净main已复现；无PR/标签，未改共享安装。
本轮依赖库没有新增源码修改。原native页stride需求为PyPTO#3038。

v4保留外层普通range，K128在两个独立M累加器间显式交错，单链K仍递增。
完整符号编译通过：130根、58动态轴、单MIX；两Left与两Right各32KiB，
两Acc各64KiB。外层不称为双槽L1预取。真实S1/S5/S6完整数值、Top-K/score、
cache/state、保护、A→B→A图和代际全部通过；数学分块改变不是仅据代数
等价便推定exact。下面为同进程五窗完整L1，参考f222/候选μs：

- 128K/B24/S6：1109.80/1119.60，慢0.883%，五窗全慢。
- 8K/B32/S6：948.26/934.26，快1.476%，五窗全快；同轮原多task973.84。
- 8K/B22/S6（T132）：822.86/816.56，快0.766%，四窗更快。
- 8K/B129/S1（T129）：986.38/980.92，快0.554%，五窗全快。
- 8K/B32/S5（T160）：938.46/936.54，快0.205%，三窗更快，收益很弱。
- 8K/B256/S1（T256）：1558.84/1569.10，慢0.658%，两窗更快。
- H32760/B32/S6（T192）：1132.84/1139.96，慢0.629%，五窗全慢。

H32760的max_seq_len=32766、压缩候选8191，仍在既有short G2区间。该边界
否定了拟议的“short G2且128<T<=192”整体启用范围。本轮不为0.2%～1.5%
的小收益继续扫描更细历史阈值，不按H8192/B32建立白名单。候选全部留在
oa_row_pair_weight_reuse_evidence/source_candidate_v1..v4。正式四文件
与新增helper先与v4逐字核对，再只恢复本轮自有改动至f222a917，未触碰
其它会话文件。关闭分支CPU C++仅去整行注释后与同编译器旧路径相同。

S1/S5驱动由既有双SPMD图重放用例扩展，只增加完整L1导出。三方S6脚本
不接受query-tokens，首次两任务在argparse退出；S1大batch又遇到Dspark
token预算与公共fixture BlockTable固定40行两次初始化失败。均保留日志。
最终只在新驱动构造fixture时包装原BlockTable扩大容量，真实S仍为1；
未改生产代码、公共fixture或Native数学。S1/S5只有旧/新SPMD性能，不能
把它写成原多task验收。完整窗口、失败脚本、任务终态和恢复凭据均归档。

复核hc_pre与静态版后，已吸收的分块、HC行依赖、4/20协同、几何前置、
O_A组通知和本地mi/li不再重复算作新方向。仍值得深入的是Score/Attention
实际供数与内存规划：核对MTE依赖和物理槽活跃期，解决真正的串行点；
跨阶段重叠须先证明改善当前关键链，不能重复已有波交错/DQ穿插负例。
本轮未增加正式性能收益，七档30%目标仍0/7，继续有效。

### 2026-10-18～19：Score提前两块未获实质收益，转向完整72核incore流水

Score候选把GM→L1提前两块、L1→Right保持提前一块，两Right与16KiB L1池
不扩容，AIV计算和环槽协议不变。完整CPU符号编译及生成指令顺序核对通过。
128K/B16/S6任务task_20261018_232714_11062101501完整精确、Top-K/score、
cache/state、图、保护及代际通过；五窗中位数原多task/正式/候选为
886.80/833.22/832.74 μs，只减0.48 μs且两窗更快，不采用、不扩测。
正式源码核对后恢复，候选和全部证据在score_l1_two_panel_ahead_evidence。

用户要求使用pypto-lib的incore-profiling，并明确当前MPMD分工必须采全核。
本轮开始完整CSA的一次24-block MIX模拟回放，目标覆盖24 AIC+48 AIV，保留
各worker的分工、真实device metadata、跨核信号与等待。不是单核Score探针。
采用原版skill和已有配套CANN9.2 worker，不改pypto-lib或共享依赖安装。

捕获任务task_20261018_235240_21310317727完成、原完整精度对照通过；
124份物理storage保留NZ/别名/保护区。独立入口从实际wrapper恢复132指针
及56动态标量，AIC/AIV数学体token与已验证wrapper一致。当前正在构建完整
模拟回放，须待72核覆盖与输出核对后再分析。模拟器周期不充当真机性能。
过程与失败记录在incore_pipeline_profiling_20261018；七档目标仍0/7。

独立包装首轮遇到设备专用wait helper的Host类型解析失败，仅给生成harness
添加设备编译保护后通过。第二轮CPU模拟task_20261019_001135_28286531625
确认启动24组1C2V。另用相同kernel.so链接真实runtime，任务
task_20261019_002239_326921630131在设备0完成：77份可写完整storage、
756895848字节与原调用after逐字节一致。该检查只验证独立ABI/数学/内存，
未测性能；模拟轨迹仍待终态。不得把真机包装验证写成模拟器采集完成。

### 2026-10-19：全核模拟卡住定位为SYS_CNT恒零，保留通信协议构建诊断副本

首轮完整采集触发本地25分钟队列上限，未导出完整轨迹。延长时限并开启dump
后，72核的instr_log全部显示SYS_CNT=0，模型指令周期增长而退避差值始终
为0，卡在100tick内层循环，无法返回下一次GM信号检查。不能把模型tick
增长当作CSA阶段在推进。定位后停止原binary的两个无效模型任务。
正式binary已在真机完成77份完整storage逐字节核对；当前证据不支持把此
停滞归因于生产信号遗漏、A3硬件通信能力或PyPTO数学代码。

提pypto-lib#1486记录msprof分钟参数/脚本秒级时限不一致；另提#1487记录
SYS_CNT模型兼容性和skill预检查建议。均为issue及建议，无PR/标签；不声称
后者是PyPTO编译bug或缩减probe已经执行。

只在tests的独立模拟副本给退避加128次读取上限。外层真实GM条件、DCCI/
volatile、发布DSB、payload fence、FFTS和全部72核分工保持；正式ops及
原二进制不动。诊断副本改变轮询时间语义，不可当作正式等待/完整耗时证据。
有界版任务task_20261019_004740_6124323569已启动，待完整输出及轨迹验收；
记录位于incore_pipeline_profiling_20261018。七档性能目标仍0/7。

有界诊断版模型已完整结束，总tick2493765，CPU运行888440ms；77份完整
可写storage共756895848字节与原真机after逐字节相同。最终72核trace仍在
msopprof解析导出，尚不能记全部采集完成。SYS_CNT只用于轮询退避，非同步
正确性必需；但旧版B16 cached 0/100tick曾为1047.58/851.32μs，不能据模型
兼容性问题直接删除有收益依据的退避。当前无新增真机性能结论，正式代码未改。

### 2026-10-19：完整72核导出完成，开始从实际流水定位Attention写回

同一有界诊断任务最终completed、exit=0，原版skill manifest为exported，
72条核trace和72份指令CSV全部存在。后处理确认24 AIC各有MMAD、48 AIV
各有非同步Vector计算，原始图23707837个事件；77份完整可写storage的
逐字节PASS沿用既有结果，不重复测试。原始大图、逐核图、概览和报告集中到
downloads/CSA_dynamic_128K_B16_S6_all72_incore_model_bounded_20261019。

已将长FIX指令对应回正式attention_stream_stages.py的QK score与四段PV
写出。24个AIC均执行20次score写出、4×20次PV写出；Score示例MMAD每核
22次，与实际计划一致。不能先把该档差距归因为工作数量不均。下一步结合
L0C生命周期、生成事件边与双AIV供数检查Cube/FIX重叠。当前仅为模型中的
定位线索，没有修改正式算子或新增真机性能结论；详见PIPELINE_FINDINGS.md。

### 2026-10-19：解决全核图过大、无法直接阅读的问题

用户指出原始19.34GB、清理1.31GB不适合人直接查看。只整理已有数据，
新增10.40MB离线交互总览，支持核组/角色筛选、逐pipe展开与共同时间轴缩放。
详细图按24个1C2V核组拆分，每份6.82～9.43MB；同时保留72份单核图，
每份2.18～3.46MB。1392602条清理事件全部保留，时间/时长/flow id原样复制，
拆分后逐项读回一致；仅把重复源码栈移入PC索引并省略寄存器参数。

19.54MB下载包包含总览、全部24份核组图和PC源码索引，不重复打包单核版。
完整原始图及旧清理图保留。没有重采或修改算子；模拟副本的时间限制仍适用。
脚本、页面模板、轻量结果清单记录在incore_pipeline_profiling_20261018，
生成HTML/JSON/ZIP只放下载目录，不把大型轨迹装入Git。

用户进一步询问有界退避是否影响模拟分析：明确影响不限于等待区间。
128次读取未按真机100tick校准，轮询频率改变可影响DCCI/GM流量、共享
访存竞争、阶段起点及计算/搬运重叠。77份storage一致只证明本用例结果，
不证明时序等价；数学源码相同也不能保证机器指令/寄存器分配完全相同。
现有模型图用于结构检查与提出候选，不能据其FIXP占用或空隙直接认定
正式版瓶颈，后续必须用正式退避版本的真机证据核实。当前不新增测试。

### 2026-10-19：用现有全核流水提出候选，按当前真机结果重验历史方案

用户明确不为模拟器修改真机等待，使用现有流水提出候选，是否保留以真机
精度和完整L1决定；旧负例不永久封禁，有理论依据可以在新瓶颈下重验。
PV N256复测128K/B16为当前838.32→847.26μs，8K/B32为942.14→936.16μs，
短档趋势已与旧记录相反。两档全部冻结门槛通过，暂不整体采用。

QK/PV显式96KiB Acc池暴露PTOAS保守别名：非NoneBox子视图回退整个父池，
已知不相交列窗口仍插FIX→M。0.66/0.67最小IR均复现，上游issue #1623已提，
附复现与建议patch，无PR/标签。独立PTOAS增加窄范围静态连续Acc列分析，
7个CPU回归及FileCheck通过；完整生成物移除该虚假依赖，参考同步顺序不变。
共享安装未改，独立worktree继承原有并行赋值修复。

8个真机任务均完整精确/Top-K/score/cache/state/保护/代际/图PASS。96KiB池
长B16为816.92→830.14，短B16为667.42→654.04，短B32为954.48→929.30，
32760/B32为1116.36→1123.40μs。128KiB三PV池长B16为824.36→850.84，
短B32为944.78→934.20μs。均为各自同轮三方五窗中位数，不跨轮拼收益。
短档存在收益，但尚无足够区间证据；全部作为实验保留，继续L1三槽方向。
详见attention_incore_candidates_20261019/ALL_RESULTS.json及完整源码。

### 2026-10-19：将动态三槽表示改成独立L1对象，验证整段供数流水

不叠加Acc候选，恢复参考后将Attention的384×512 Mat父池改为三份128×512。
只展开固定三相位，外层实际T/query/mask/地址/尾排空仍动态，AIV完全不改。
无样例shape CPU通过，58动态轴，Mat共464KiB、没有Mat→Mat TMOV；6288组
整数控制检查通过。两档真机已提交，未预记精度/性能，过程位于
attention_l1_triplet_20261019。正式目录当前是待验证候选，原参考完整保留。

L1三槽两任务最终均exit0，精确/图/状态保护/协议全部通过。五窗完整L1：
128K/B16多task875.40、参考824.52、候选827.40；8K/B32多task971.68、
参考943.06、候选945.50μs。两档小幅回退，不采用、不扩七档。完整记录保留。
随后恢复参考基底，另做PV按32head完整512列写回：L0B以K64控制在64KiB，
每个Right供两半head复用，两Acc共128KiB，两个连续写回替代四个跨行写回。
只改变固定模型维度分块；AIV、协议及动态输入不变。K切分须设备精确验收。
CPU首轮参数拼错于编译前退出，已更正重启；未预记PASS，见
attention_pv_contiguous_rows_20261019。当前正式目录是这一待验证候选。

PV整行候选CPU完成，58动态轴、Acc128KiB/Right64KiB，两个64KiB整行
写回；两档真机已排队，16卡暂被占满，尚未执行。完整候选保存在独立快照，
正式ops已恢复已验证参考，避免把待验证候选当作当前默认。

### 2026-10-19：由PV的FIX barrier定位GM列分片范围丢失

PTOAS的partition_view别名传播直接复制整个GM根，即使静态不相交列也
插入FIX barrier。0.66/0.67最小IR复现，上游master只读源码仍相同；
已提PTOAS #1624，附建议patch，无PR/标签。与#1623的片上Acc范围不同。
独立MemoryDependentAnalyzer修复只证明二维ND、同指针/类型、静态2次幂
行距、单位内stride、64B对齐且行内互斥的列；行起点可动态，其它保守回退。
12项正反例和循环复用共13项FileCheck通过，跨迭代真实重用仍有同步。

同一完整CSA IR的生成物FIX barrier81→77、MTE3 112→110，set/wait仍
各1675；滤去同步语句后所有C++逐行相同。数学、搬运、FFTS计数不变。
这不是设备收益证明。单独链接新bin，原build/bin不覆盖；测试再冻结两份
工具链，仅候选调用切换新编译器，原多task/参考保持旧版，三个角色同进程。
两个代表档已经排队，尚无真机结果。证据、issue与复现路径在
parallel_20261018/gm_partition_sync_evidence_20261019。

补充读取现有72核Scalar指令计数（SCALAR_INSTRUCTION_COUNTS），没有重采。
AIC合计DIV1838/REM2396，AIV DIV3785/REM5605；大部分Scalar记录含模拟
退避，不能把计数当作时间或据此断言64位算术为瓶颈，本轮没有据此改算法。
当前继续以真机验证候选，不因旧负例永久禁用方案；性能目标仍0/7。

PV行块两档随后完成，全部冻结精度/Top-K/score/cache/state/图/保护/代际
通过。五窗L1：长B16参考827.66→候选827.62μs持平；短B32 955.02→933.30μs，
约降低2.27%。原多task分别871.14/960.02μs。暂不全局采用，完整候选保留。
GM编译器隔离驱动首轮两任务在归档tuple引用旧run_device.sh时退出，尚无
CSA执行；修正实际启动文件名，核对归档输入后重提v2，不报告成组件bug。

GM隔离驱动v2另缺source_interface.json，同样在CSA调用前退出。补齐清单，
并扫描整个driver的所有HERE字面资源后重提v3；当前原多task已有67次真实
PTOAS调用日志，均指向冻结旧编译器。后续候选和参考同样核对实际路径，
不能只依赖环境配置声明。独立PTOAS改动本地提交a5e0a766，7+13项回归，
无PR/push；共享安装不改。目标保持active，等待设备结果继续评估。

GM编译器v3两个真机任务终态exit0，完整精确/Top-K/score/cache/state/保护/
代际/图通过。实际调用日志验证只有候选完整CSA用新bin，原多task与参考
保持冻结旧版。五窗完整L1：长B16多task871.18、参考840.80、修正821.58μs，
降低2.286%；短B32多task979.76、参考946.58、修正960.54μs，回退1.475%。
不全局采用。先区分四处FIX与两处MTE3的效果，再复评Acc组合，不能从
静态barrier减少推断统一收益。正式源码保持已验证版；ALL_RESULTS保存全部
窗口和编译路径，RETEST_CONDITIONS记录旧方案重验的具体条件。目标仍0/7。

为区分GM修复的两部分，在独立PTOAS a5e0a766上仅给本地候选加FP32限制，
新链接toolchain_fp32，不覆盖另外两份bin。完整CSA IR确认只删四处FIX，
MTE3仍112，数学/搬运及全部非同步C++逐行不变。两档受限候选已提交，
尚待设备结果；不把类型过滤推广为所有FP32都走FIX，也不改上游通用建议。
