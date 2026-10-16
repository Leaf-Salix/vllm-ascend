# CSA 单 SPMD：完整过程与接续入口

完整过程统一从 [PROCESS_LOG.md](PROCESS_LOG.md) 阅读；逐次成功/失败尝试见
[ALL_ATTEMPTS.md](ALL_ATTEMPTS.md)，issue/PR 与本地修复见
[ISSUES_AND_PRS.md](ISSUES_AND_PRS.md)。材料同步归档到用户指定主仓的同名
`tests/pypto_test/csa_static_spmd_feasibility_20261015/`，归档清单见 ARCHIVE_STATE.json。

目标是把较优 CSA 动态实现的计算分片、逐核顺序及必要依赖固化到一次 MIX SPMD
提交中。保留原数学 tile、同组 Cube/Vector 握手和 Native device Tensor 输入。
目前没有生产替换。128K/B16 静态 CSA 已完成三次设备调用，对冻结动态 PTO 的
输出、Top-K、六份 cache/state 逐项完全相同，进度及保护区通过。后续同卡性能测试已完成：
L1中位数动态897.70 μs、单SPMD1380.22 μs，耗时增加53.75%；固定输入图重放比较通过。
详见[性能报告](PERFORMANCE_STATIC_VS_DYNAMIC_20261016.md)。

当前静态模板特化了 batch、history、shape 和逻辑 block 数，全部按用户要求保留。
最终需要动态 batch_size/seqlen；静态原型通过不代表动态实现完成。
设备验证范围仍仅为 **128K/B16（history=131072，16 请求，每请求 6 query token）**。
用户已明确授权实现临时普通 GM notify/wait，原缓存审查暂停已解除。
独立 PyPTO 接口和完整 B16 CPU 编译通过。首版返回但输出错误，补齐通知前的
本核 PIPE_ALL 后，独立 payload 小例与完整 B16 各三次调用均通过；见
[临时通知/等待实现](LOCAL_SIGNAL_NOTIFY_WAIT_20261016.md)。
[Scalar 缓存协议复核](SCALAR_CACHE_PROTOCOL_REVIEW_20261016.md) 保留此前审查依据。

用户转来的动态范围、模板边界与负载均衡分析，单独保存于
[动态 SPMD 后续参考](DYNAMIC_SPMD_REFERENCE_20261016.md)。

## 2026-10-16 独立组件修复进展

用户已授权分别修改独立 Simpler 与 PyPTO worktree。最新结果统一记录在
[PYPTO_STATIC_FIXES_20261016.md](PYPTO_STATIC_FIXES_20261016.md)。以下历史限制
描述保留为修复前基线，不再代表共享 InOut、CORE 容量或 JIT 小例子的当前状态。

- Simpler CORE 扩容已验证并本地提交 `57e499238`；共享安装未切换。
- PyPTO 分支 `feat/csa-static-spmd-fixes-20261016` 已完成完整 B4 默认 PYPTO
  代码生成；最终 carry 修复相关两文件 169 项回归通过，补丁尚未提交。
- 共享 InOut 单 MIX 的 24 组、双 AIV、当前流和三次执行真机验证已通过。
- Torch schema 的 64 入参限制已经绕开：用户明确选择现有 JIT native 直调，
  62 项候选已取消。详见 [直调入口记录](DIRECT_NATIVE_ENTRY_20261016.md)。
- B16 完整 kernel-mode 代码生成已通过。设备验证首先遇到两个 Native cache
  子视图与根缓冲的部分可写重叠；改在入口内部派生这两个描述符，外部 130 项、
  Core 仍 132 项，不增加搬运或设备 task。逐次结果见 PROCESS_LOG.md。
- B16 完整 native 任务已完成二进制编译并实际启动，但设备报告
  `S1:running-stalled`，没有完整输出。抽取同一控制表的诊断也出现进度等待超时；
  五组历史控制表诊断均已结束：读写 volatile 或仅读 volatile 的三代际观察通过，
  原版、仅消费者 fence、仅写 volatile 均失败。这不证明完整同步协议正确。
  范围、任务和证据见
  [B16 进度协议诊断](B16_PROGRESS_DIAGNOSIS_20261016.md)。
- 后续临时普通 GM notify/wait 与本核流水等待已解决当前 B16 原型的运行阻塞：
  task_20261016_180421_21978513378 的三次调用功能及对冻结动态 PTO 的精确比较通过。
  每轮恢复 fixture，进度不清零；不是连续 decode 轨迹，也没有测性能。
- 完整 B4 的现有 program 产物已通过 CCE/调度二进制编译及 callable 组装，
  详见 [二进制编译记录](BINARY_COMPILE_20261016.md)。尚未执行完整设备核。

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
   [PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md](PYPTO_MIXED_WHILE_LOOP_STATE_REPRO.md)。
   本次另准备了显式 Group 内 AIC/AIV 两半的写法；仍是一次 MIX 提交，不能把它
   与自动拆分路径的验证结果混为一谈，也没有增加循环后伪使用绕过问题。
2. 原 Simpler 的 32 Tensor 限制已在独立 worktree 扩至 256 Tensor + 16 scalar；
   dump 多字掩码、派发结构及所有关联布局同步更新，A3 的 HBG/TMR program/kernel
   大参数验证通过。详见 [扩容记录](SIMPLER_CORE_EXPANSION_20261016.md)。共享环境
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
   容量问题。见 [内存分配复现](MEMORY_REUSE_PHASES_REPRO.md)。
4. inline Group 已让 5 参数小例子生成单 MIX 并通过 Meta 注册；共享 InOut 的
   小例子在 ExpandMixedKernel 首次产生 free variable。完整证据及接续方法见
   [COMPILE_AND_RESUME.md](COMPILE_AND_RESUME.md)。
   详见 [JIT Group 表达记录](JIT_SINGLE_GROUP_NOTES.md) 和
   [共享 InOut 复现](PYPTO_GROUP_SHARED_INOUT_REPRO.md)。

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
