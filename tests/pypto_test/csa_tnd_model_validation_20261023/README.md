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

## 当前证据

- `native_b16_summary.json`：Native 16卡B16功能通过，无抢占，平均接受3.80650；
  32个简单答案通过，跨DP及旧Native输出token差异均为0。
- `workspace_validation.json`：两种TND的临时工作区复用单卡验证通过。
  128K/B16的S1/S6及两层交替，eager和graph的输出、Top-K、cache/state逐bit一致，保护区通过。
- `workspace_pool_estimate.json`：CPU meta容量估算，不是模型设备峰值。
- `redispatch_failures.json`：v05两种PTO在收尾阶段触发的Native query/Q前缀容量不一致。
- `redispatch_validation.json`：修复后SWA/HCA/CSA的单卡Native容量图参考对照通过。
- `jobs.json`：当前v09两种PTO B16整模型任务及其前置验证路径。B16三侧整模型及输出对照已完成，B24单卡已通过，修正整模型启动器的HCCL覆盖后正在重跑。

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

## 最新约束与进行中的改造

用户明确沿用既有精度口径，完整披露复杂问题差异。完整问答已导出到
`/data/pyptouser/qinchuanyu/pto-eager/downloads/CSA_TND_128K_B16_B24_QA_20261023/`，
含左右对照HTML、Markdown、原始输出token、24个包含128K背景的完整输入。
回答为本次实际生成的全部192token，不是无限长度的完整解题结果。

用户同时明确：MPMD与SPMD的纯临时工作区都必须在执行中由Simpler内部申请
和管理，不能在Python中外部分配，也不以外部共享池作为最终方案。
现有B16和B24 MPMD证据属于旧外部工作区实现；生产路径尚未完成该项迁移。

私有原型推进到v17，普通入口内部创建MPMD71个、SPMD72个临时Tensor（含Top-K）。
诊断入口只额外接受测试需要观察的Top-K输出；旧外部实现仅作数值参考。
SPMD内部信号有设备清零任务，初始化依赖及全部开销属于这次调用；不能隐藏为
捕获前一次性外部分配。数学任务主体沿用冻结源，MPMD不引入软同步。
v14在JIT动态reshape metadata推导失败，未运行内部版本。纯CPU最小复现
`jit_shape_repro_20261024.py`区分tuple形状支持与动态局部尺寸变量两类问题。
前者main已有[#2880](https://github.com/hw-native-sys/pypto/pull/2880)修复；
后者已在[#3033追加中文复现](https://github.com/hw-native-sys/pypto/issues/3033#issuecomment-6369484280)。
v17四个入口×两组策略共8项完整CPU特化已通过，证据为
`internal_workspace_cpu_validation.json`。下一步才是设备验证；不把CPU特化
成功写成NPU编译、内部内存生命周期或功能正确性通过。
