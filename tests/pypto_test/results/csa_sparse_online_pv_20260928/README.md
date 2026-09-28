# Sparse 累计 softmax：移除新 PV 的冗余重标定

来源为最新 ops-transformer b5b33e14 的
`sparse_flash_mla_csa_block_vector.h::SoftmaxFlashV2Compute/DealBmm2ResBaseBlock`。
Native 在概率转 BF16 前纳入累计最大值，PV 合并只缩放之前的结果，然后加上当前 PV。

当前生产 d1f170ff 仍按 N128 局部最大值量化，PV 需要同时缩放旧结果和新结果。
历史累计最大值候选（验证日志 §258–259）虽然改变了概率量化边界，
却保留两侧重标定，完整 CSA 慢 2.10%，已经停止扩测。
本项的新变量是利用累计最大值单调性消除冗余 Vector 计算，不原样重测旧候选。

独立工作树 `.cache/csa-sparse-online-pv-a66255ea`，以 a66255ea 冻结，
仅修改性能版 Sparse；不包含 QR、Top-K UB、PV L0B 或主工作树的 WO_A 修改。
原生 cache、N128 分块、三槽预发、跨 query 流水、任务数及调度标志均不变。

## 算术合同与生成代码

- 每个 query 的 softmax 累计最大值从 sink 初始化，首块重置，空块保留。
  独立于较晚执行的 PV 状态，下一 query 不等待前一 query 排空。
- BF16 概率采用累计最大值及 Native 的 `round`；仍与 Native N512 分块有差异，
  也不同于 pypto-lib 局部最大值及 `rint`，不能宣称完整复刻或数值中性。
- PV 读取当前块的累计最大值 `pv_m`，必有 `pv_m >= m_iter`，因此 `next_m = pv_m`、
  `beta = exp(pv_m-next_m) = 1`。只对旧结果计算 alpha 和乘法。
  这一推导依赖有效块按 query 顺序处理与首块重置；设备验证仍须覆盖这些条件。

完整 CPU lowering/PTOAS/CCE/AICPU 链接通过：[compile.py](compile.py)、[compile.log](compile.log)。
生成的 `qk_pv_aiv.cpp` 每个循环体静态 `TEXP` 从 3 处减到 2 处，
`TROWEXPANDMUL` 从 4 处减到 2 处；累计最大值比较移到 softmax 阶段。
不据指令减少推算核内或完整 CSA 加速。

## 验证安排与状态

当前尚未提交设备任务，未合入生产。先等待已经排队的 QR 和 Top-K UB 对照，避免堆积测试。
后续先复用固定 Native Q/cache/Top-K 的单卡诊断及 B3/H255 解析尾块；
比较原性能版、旧累计最大值与本候选的逐元素误差，确认只缩放旧 PV 的等价性。
必要时再测长短 B16 的核内、完整 CSA/P95、保护区和图重放。
涉及算术变化，不能复用要求所有浮点状态与生产版零容差相同的搬运候选收集器；
应独立记录误差，整数/cache/保护区仍严格一致，最终保留仍需真实 EP16 token/DSpark。

[sparse_case.py](sparse_case.py)只选择冻结源码并调用现有诊断。
诊断入口新增 `--device`，按 task-submit 分配设备运行，默认 0 保持旧命令可复现；不改生产设备逻辑。
