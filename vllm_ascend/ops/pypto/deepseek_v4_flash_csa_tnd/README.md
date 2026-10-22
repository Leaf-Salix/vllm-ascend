# TND CSA：AICPU 调度的多 task 实现

本目录按用户2026-10-22的架构约束重构，以`deepseek_v4_flash_csa`的独立计算任务
及动态依赖图为骨架，复用TND SPMD已验证的输入合同和核内数学。禁止GM完成标志、
代际、轮询等算法层软同步；跨task依赖由AICPU的Tensor/切片和显式TaskId管理。
Score、Attention自身的Cube/Vector硬件流水保留。

当前八档性能规范的单卡功能、图、状态及完整计时已通过；保持62/63组实际任务。
前段投影/量化/cache和八组输出链按BSH策略取消额外整组起跑等待，保留Tensor/TaskId
依赖及原数学；HC、Score、TopK、Attention和HC_post仍保留既有起跑设置。
本轮完整L1中位数较v9下降10.15%～14.71%，八档最大值也下降；仍慢于TND-SPMD，
**尚未达到原10%目标**。同址metadata十场景及四步连续状态均通过。
长档保留AICPU与AICore的BOOL整数0/1编码修复。旧融合实现的结果不代替本实现验收。
旧源码和八档性能保留在Git `67134f28b`与原证据目录中。

## 任务与数据

- HC的widen/RMS、linear、reduce、gates、mix/norm、Sinkhorn独立。
- Q_A/QR、Indexer Compressor、Indexer query/weights、主Compressor、Q_B、KV各自成链。
- Score、Top-K、Attention页计划、Attention独立；O_A→量化→O_B按八个输出组独立调度，
  HC_post通过TaskId等待全部输出组。
- 输入沿用Native TND hidden、设备Q/L、页表、slots、compact RoPE和cache/state。
  活动query由Q/L决定，不用`T/B`或`token//6`猜测；padding不参与业务计算。
- 请求/组/叶和压缩偏移目前在AICPU编排计算，仍保留旧消费者的数据描述表作为迁移接口。
  后续须缩减冗余表、通过任务参数/切片表达工作范围，不能把此状态当作最终任务粒度达标。
- scale页写入保持同页唯一写者，避免64B读改写竞争。连续页读取证明由实际Score消费者
  对本叶页表做Vector校验，结果只留本核Scalar，没有独立metadata任务或GM证明表。
  每次图重放重新读取真实输入；规划和校验成本属于CSA，不移到Host准备或测试复位。

`binding.PreparedDynamicCSA`只按描述符准备工作区并绑定Native存储，不增加Host
`.item()`/D2H。原信号工作区、初始化和根参数已移除。业务临时量仍由单流/单图独占。
服务入口名称保持`PTO_CSA_VARIANT=mpmd_tnd`；本轮不运行16卡或整模型。

## 验证与记录

重构实验在`tests/pypto_test/csa_mpmd_aicpu_20261022/`，通过`prepare.py`冻结算子和
测试入口；SDK统一采用公共manifest。CPU检查不占卡，NPU执行通过`task-submit`。
最新八档表为该目录的`scheduling_v16_PERFORMANCE.md`，对应完整样本与实际路径见
`scheduling_v16_qualified.json`，同址状态见`scheduling_v16_state_validation.json`。
Native、BSH和SPMD复用同条件合格记录；本轮只补变更的TND-MPMD两列。

原10%优化目标的固定混长/padding场景与参考保持原义；最新八档六列性能另按
`CSA_PERFORMANCE_TEST_SPEC.md`汇总，不能混算场景、权重和分母。主比较为完整CSA
L1首尾跨度，同时记录含AICPU的完整device跨度，不把规划前移误报成成本消失。

全过程追加到
[DSV4_FLASH_CSA_VALIDATION_LOG.md](../../../../tests/pypto_test/DSV4_FLASH_CSA_VALIDATION_LOG.md)
第735.58节起。架构、功能、性能分别给出证据，不用task数量或局部收益代替完整验收。
