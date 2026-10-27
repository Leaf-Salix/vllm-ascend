# 2026-10-25 CSA八档六列性能整理

按[SPEC.md](SPEC.md)执行：八档固定场景，每格至少10个有效单次窗口，取全部合格样本
中位数；最后一列分母为TND-SPMD。本轮已完成，最终表见[PERFORMANCE.md](PERFORMANCE.md)，
原始样本、来源和排除理由见[PERFORMANCE.json](PERFORMANCE.json)。没有恢复优化Goal，不做16卡测试，不push。

## 选定版本与历史复用

- Native复用已有八档CANN9.2、npugraph_ex、dynamic=False、static kernel、SuperKernel、
  inplace pass、多stream、NZ2的实际设备记录。每档30窗；重新读取原始Profiler CSV并
  核对图外复位剔除规则、真实SK/NZ kernel和正确性，不拿开关配置代替生效证明。
  这是当前可获取实现的实测值，不能代表客户侧最优性能。
- TND-SPMD固定正式f72a1f752，即init_coeff_single_spmd_v2。初始化、硬屏障和全部业务
  位于唯一MIX task，L1覆盖24AIC+48AIV。八档正式验收各10窗，加上相同源码/SDK/输入/
  正常采样的后续控制组，保留全部270窗；不挑最快一轮或最快窗口。
  插桩、L2/GM扰动及改变预热节奏的五组记录排除，未采用候选不填本列。
- BSH两列沿用已验收的算子版本，包括150tick和短档临时Tile声明兼容修正；原源码在
  61af469e到当前版本之间无生产修改。旧样本使用26e94e056 SDK，不能改标为本轮d44925c1，
  因此补采七个S6场景。混长T56的两个BSH格按规范保留N/A。
- TND-MPMD使用当前10511222c中的实现，算术/调度沿用已采用多task策略，当前入口
  使用Simpler内部工作区。旧v22外部工作区/旧SDK数据不能替代本列，补采八档。
  现有纯PyTorch复现结果未给出与原Native fixture逐项等价证据，也不拼入本表。

Native与PTO均保持原规范的正式第4层W8A8/NZ2、BF16激活、seed1024、固定query列表、
物理页映射、非均匀Indexer scale和状态起点；C02是真256K，C08是T56而非B×6。
PTO统一公共csa-shared-20261024-group-dims：PyPTO d44925c1、PTOAS914863ed、Simpler57e49923。
主工作树已快进到已发布10511222c；没有新造依赖分支或重建共享SDK。

## 只补缺项

[prepare.py](prepare.py)从既有runner及冻结源码生成私有包，变动限于当前内部工作区
绑定接口、测量侧选择、环境记录和同版ring配置，不修改算子数学或调度。
TND两种实现仍执行eager/graph正确性对照；性能只采缺失的MPMD侧10个L1窗口，再从
同一图单独采30次完整device跨度，保留AICPU/AICore记录并做采样后检查。
BSH两个真实入口分别采10窗，不将TND binding冒充BSH接口。

TND采用已验收heap[16,256,16,16]MiB、window4096，BSH保持其既有默认runtime配置；
这些实际分配策略记录于manifest及结果，不调整到某个最快候选。
所有任务通过task-submit，task ID/命令/输出见[jobs.json](jobs.json)。
C04代表档通过后，剩余13个单卡补测也全部exit=0。七档BSH各侧10窗；八档TND-MPMD
各10个L1窗及30个独立Profiler窗。共复用510窗、新采460窗，没有失败/重试窗口。
输出、TopK/score、cache/state、图重放和保护区检查全部通过。源码未在排队/运行中改写。

L1保留task间等待与空隙，不能解释成完全扣除调度的纯计算时间；AICPU列为全设备
活动首尾，不是AICPU自身耗时。L1与外层Profiler分开采集，不将两列相减冒充精确调度成本。
Profiler绝对时间戳约为1.8e15微秒，汇总使用Decimal相减后转为时长，避免普通float
大数相减损失亚微秒精度；过滤、归组及计时边界不变，旧float样本另存作对照，不重测。
完整链路是HC_pre → input RMSNorm → CSA → HC_post；输入复位、warmup与D2H在区间外。
沿用既有输出/TopK/score/cache/state/保护区验收，Native原有非bit精度差异单列；
单层性能不证明整模型token、完整答案或DSpark统计一致。

## 复算

```bash
python tests/pypto_test/csa_performance_20261025/collect.py
```

只读取JSON/L1/Profiler原始记录，不占卡、不做源码hash扫描。大型trace保留原路径，
不覆盖旧结果。旧壳的sources.mpmd字段在BSH家族仍沿用candidate参数；汇总器明确
记录真实BSHEntry/NativeCSACall来源，不以该旧字段将BSH误标成TND。

## 提交检查

本轮只做测试适配、原始记录复算和证据归档，未修改生产算子或SDK。
Python局部Ruff/格式、Shell语法、JSON/窗口边界/采样数量/表格百分比检查通过。
按仓库要求尝试format.sh ci，当前环境缺pre-commit，不能标为全仓格式检查通过。
只作详细中文Signed-off-by本地提交；没有push。

26个历史SPMD任务日志均确认CANN9.2及公共d44925c1工具链，摘录见TASK_LOG_EVIDENCE.json。
历史SPMD未单独导出框架版本，引用同公共启动器与同环境Native的版本记录，不冒充逐任务新指纹。
汇总验证收据见VALIDATION.json。
