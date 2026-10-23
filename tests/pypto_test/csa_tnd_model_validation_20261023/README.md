# CSA TND 真实模型功能验证续作

本轮接续 `csa-repro-20261021-clean/kv-repro` 中断的真实模型验证，
先完成 128K/B16/S6 的 Native、TND-SPMD、TND-MPMD 三侧对照，再验证 B24。
不改变已完成的 PyTorch 单卡复现手册，也不重复生成已审计的历史 KV。

## 固定口径

- 真实权重 `/data/model/DeepSeek-V4-Flash-0731-w8a8`，TP1、DP=EP16，DSpark 出5验6。
- CANN 9.2.0-beta.2；PyPTO/PTOAS/Simpler 从公共清单绑定，框架和算子按提交冻结。
- 只切换 CSA；20 层 HCA 始终 Native。NZ mode=2，EPLB 关闭。
- 保留 `npugraph_ex`、`FULL_DECODE_ONLY`，关闭静态编译，不加入 SuperKernel。
- 确定性 level=1、HCCL_DETERMINISTIC=true、AIV 展开模式、event_work_mode=1 三侧一致。
- 同一批混合问题、每请求192输出token；固定历史实际长度131104..131346，来自128K公共前缀加问题后缀。
- 首轮 B16 显存预算三侧均0.92。真实S6满档、无抢占是功能门槛，不能缩小实际并发来绕过OOM。
- 功能检查与精度对照分开：功能检查要求实际21层CSA路由、完整token和DSpark事件；
  精度对照报告逐请求token、首处分歧、DSpark接受长度分布和简单问题答案，不宣称浮点bit一致。

## 历史中断点

旧 `decode-native_clean2` 已生成256请求×192token，平均接受3.80650。
旧 `decode-spmd_clean2` 在首个S6 decode申请210 MiB时OOM，MPMD随后未执行。
另一次8K MPMD尝试也在相同申请量处OOM。两次都使用0.95显存预算及静态编译。
这些是诊断线索，不能当作当前版本的结果或与本轮直接计算性能收益。

历史 KV：`/data/pyptouser/qinchuanyu/pto-eager/csa-repro-20261021-clean/kv-repro/mixed128k_fixed`。
只读使用前16/24个不同问题；完整40问题及生成、审计证据仍在原目录。

## 执行

`prepare.py <实验名>` 冻结 HEAD、当前测试入口和公共SDK清单，
不会带入其他会话未提交的算子修改；不复制SDK、权重或KV。
用 `preflight.sh <冻结源码> <结果JSON>` 先排单卡核对Native依赖和两种TND依赖图。
然后通过 `task-submit` 为 `run.sh <冻结源码> <输出目录> <native|spmd_tnd|mpmd_tnd> <16|24> <预算>` 分配16卡。
每次使用新输出目录。B16三侧及输出对照通过后，才能开始B24。

`check.py` 检查真实路由和功能完成，不代表Native/PTO输出一致。
各次实验的任务号、结果及对照结论写回主 `DSV4_FLASH_CSA_VALIDATION_LOG.md`。

## 旧外部工作区实现的历史证据

- `native_b16_summary.json`：Native 16卡B16功能通过，无抢占，平均接受3.80650；
  32个简单答案通过，跨DP及旧Native输出token差异均为0。
- `workspace_validation.json`：两种TND的临时工作区复用单卡验证通过。
  128K/B16的S1/S6及两层交替，eager和graph的输出、Top-K、cache/state逐bit一致，保护区通过。
- `workspace_pool_estimate.json`：CPU meta容量估算，不是模型设备峰值。
- `redispatch_failures.json`：v05两种PTO在收尾阶段触发的Native query/Q前缀容量不一致。
- `redispatch_validation.json`：修复后SWA/HCA/CSA的单卡Native容量图参考对照通过。
- `jobs.json`：B16三侧旧外部工作区实现已完成，B24 Native/MPMD完成、SPMD收尾OOM；
  以及后续内部工作区原型的编译、设备任务和失败归因。旧实现结果不能代替新实现验收。

有工作区修复的冻结包使用`prepare.py <实验名> --revision 99268437c --workspace-fix`；
图回退修复再加`--redispatch-fix`。
启动`run.sh`时第六个参数提供单卡`result.json`。该结果必须包含两种实现的PASS，
否则启动器直接退出。该门槛不替代后续整模型正确性检查。

## 当前运行时的容量依据

`runtime_capacity_estimate.json` 是对当前已构建 Simpler 的 CPU 合同查询，
不分配 NPU 显存。默认 device scratch 声明容量为3.076 GiB；
`heap_mb=[256,128,256,32]、task_window=4096`时为1.191 GiB。
当前256参数的TaskPayload使默认任务窗口更大，不能沿用旧版1.343 GiB估算。
此CPU结果只用于选配置；当前TND每ring16MiB的设备验证证据另见`b24_workspace_validation.json`。

单卡入口支持`workspace_probe.sh <源码> <输出> --batch 24
--ring-heap-mb 256 128 256 32 --ring-task-window 4096`。
对应整模型入口在单卡结果路径之后加
`--ring-heap-mb 256,128,256,32 --ring-task-window 4096`，
会核对单卡结果的batch与ring配置。B24设备测试仍须等B16输出对照通过后再提交。

B16完整对照见`b16_comparison.json`及`b16_comparison.texts.md`。
两种PTO全部输出token和接受事件一致；对Native简单问题全部一致，复杂问题仍有差异。
B24已通过每ring16MiB、window4096的单卡验证；
整模型当前三侧统一`--hccl-buffer-mb 768`、预算0.935。旧v11的父进程512被旧启动器覆盖成1800，不能用于验收。

当前B24 Native任务为`task_20261023_220617_238843122886`，冻结v12。
Native、MPMD均已通过每卡33个有效24×S6 step与零抢占；SPMD在收尾OOM，尚未通过。
旧v11三项任务已退出或取消，原因见`b24_v11_configuration_failure.json`。
新的检查同时要求Worker实际HCCL配置匹配，并记录已有Host metadata中的
有效请求数、有效token数；图padding到144行本身不能作为24×S6通过证据。

## 当前内部工作区版本

用户明确：MPMD与SPMD的纯临时工作区都由Simpler在执行中申请和管理；
Python不分配scratch、普通Top-K或信号，也不使用模型级外部共享池。
当前实现位于已冻结的`experiments/v25_internal_model/source`，正式算子目录尚未迁移。
两种普通入口只接收55个外部Tensor；诊断入口额外接收Top-K输出供校验。
MPMD保留AICPU多task调度，不新增软同步。SPMD信号每调用由设备任务清零，
其初始化和依赖包含在调用和图重放中。

`internal_service_validation.json`是实际PreparedDynamicCSA单卡证据：
128K/B16/S1/layer2、B16/S6/layer2、B24/S6/layer4两种TND全部通过；
诊断输出、Top-K索引与分数、cache/state与各自旧实现逐bit一致，普通入口两次
图重放与保护区通过，外部诊断输出哨兵保持不变。任务
`task_20261024_031432_271551524352`已exit=0。有效容量配置为
`heap_mb=[16,256,16,16] MiB`、`task_window=4096`。
此前把大heap配到ring0的判断已撤回：AICPU executor有外层scope，实际内部
Tensor位于ring1；首次失败日志与全部过程保留在主验证日志，不作为通过证据。

公共工具链`csa-shared-20261024-group-dims`已完成相关验证并标记validated。
PyPTO为`d44925c1`；Simpler、PTOAS、CANN仍用公共清单与环境的固定版本。
SDK验证通过不代表模型验证已通过。当前16卡B16任务：

- MPMD：`task_20261024_034554_36644472287`，结果`runs/v25_mpmd_b16`。
- SPMD：`task_20261024_034603_366976914195`，结果`runs/v25_spmd_b16`。

实际状态以队列和`jobs.json`为准。两侧共用预算0.92、HCCL1800、相同源码/SDK/heap。
B16完整功能与输出对照完成后才提交16卡B24。新模型检查要求21层CSA实际接入、
真实B×S6、192token完整输出、无抢占，并要求所有层外部scratch字节数为0。

Native正确性参考沿用`v02_native_b16`和`v12_native_b24`。新增
`compare.py --reuse-native-reference`明确披露只作用于PTO的SDK/heap差异；
权重、KV、问题、推理参数和实际Worker配置仍严格检查，不用于跨版本性能比较。
简单问题须逐token一致；复杂问题差异完整呈现。`export_qa.py --comparisons`
可分别导出Native、MPMD、SPMD全文及原始token，不把一侧回答替代另一侧。
旧外部工作区问答位于`downloads/CSA_TND_128K_B16_B24_QA_20261023/`，
它是历史结果，不能当作当前内部工作区版的整模型验收。

编译问题与复现：

- [#3033](https://github.com/hw-native-sys/pypto/issues/3033#issuecomment-6369484280)：动态reshape metadata缺失；
  使用描述符读取绕过，代码`jit_shape_repro_20261024.py`。
- [#3137](https://github.com/hw-native-sys/pypto/issues/3137)：自动SPMD包装不能反推复合尺寸；
  MPMD使用显式incore边界，SPMD避免局部尺寸别名；`compound_extent_repro_20261024.py`。
- [#3138](https://github.com/hw-native-sys/pypto/issues/3138)：Group成员动态类型重绑定遗漏，
  公共d44925c1已修复；`mixed_param_extent_repro_20261024.py`和`trace_internal_scale_symbol.py`。

详细中文过程持续追加主`DSV4_FLASH_CSA_VALIDATION_LOG.md`。
