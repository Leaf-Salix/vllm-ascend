# SDMA 新版吸纳与 L2 预取验证

用户于 2026-10-20 要求吸纳 Simpler #2583、PyPTO #3001 的新修改，再考虑
CSA 的 L2 prefetch。Attention 优化方向继续停止；最终仍须完整动态单 SPMD。
本目录记录本轮来源、构建、功能与后续性能证据。所有尚未完成的验证均不预记通过。

## 来源与隔离

- Simpler #2583 最新 head：`97343062d08ad2ce4fac73cc17e42180972da603`。
  相对此前 `0b34c256` 增量迁入新的 `simpler-csa-sdma-refresh-20261020`
  worktree，本地提交 `685bec17d`；保留 256 Tensor + 16 scalar 派发扩容。
- PyPTO #3001 自身仍停留在 `ed612fbdb`；实际新修复来自该 PR 的
  [最新评论](https://github.com/hw-native-sys/pypto/pull/3001#issuecomment-6288341715)
  指定的 `TaoZQY/pypto:0093d6171dda55886846766b2483a80425b5bb28`。
  配套 PR #1 已关闭且未合并，API 中保留的旧 head `24b263c2` 不是最新方案。
  迁入新的 `pypto-csa-sdma-refresh-20261020`，本地提交 `9c36a3450`；
  保留既有 CSA 编译修复、GM 子视图和 NZ 整体展平。
- PyPTO runtime gitlink / ABI 指向本地合成 Simpler `685bec17d`。
  PTO-ISA 仍使用原 pin `327cd586`，不应用已被替代的 ISA Wait 改动。
- 原有两个 SDMA worktree 不改动；Simpler 未提交的旧通道草稿另存补丁，
  新版采用上游驻留编号实现。共享安装不改动，依赖仓库不 push。

## 与旧版的实质区别

1. 通道由 Simpler 驻留 AIV 身份分配，TMR/HBG 两份派发缓冲均设置；
   不再用会在同组两条 AIV 上重号的原生 block index。
2. 每个 prefetch context 的 256B UB scratch 纳入两套内存规划，
   整个函数生命周期保留，避开业务 Tile 和用户保留空间。
3. PyPTO 的预取等待适配使用公开 `AsyncEvent::Test()`，未完成探测间隔约
   1 μs，每事件 20 ms 期限；超时仍返回 false，不取消传输或允许提前复用。
4. 新版可以正确验证完整 48 AIV 并发，但这些是功能修复，不能直接推出性能收益。

## 本机验证边界

先重建 Simpler binding、TMR/HBG runtime、PyPTO core 与 torch adapter。
复用上游完整 24 AIC + 48 AIV MIX 用例，覆盖活跃 Tile、变输入图重放和
65 次提交的 flag-slot 复用。测试必须通过任务队列；按实际 torch/CANN 版本记录，
不能把上游 CANN 9.0 / torch_npu 2.6 的结果替代本机验证。

再检查真实生成代码中的通道、scratch 和等待适配是否生效。
只在本机功能通过后，进行完整 CSA 的配对性能对照。

## L2 预取选择原则

旧版 Q_B 三种策略均无完整 CSA 净收益，不能原样重跑或把局部 Q_B 缩短写成收益。
优先寻找长 Indexer 阶段可覆盖的后继只读权重读取，限制预取范围与发起者数量，
避免在 HC 输入、归一化、QA 关键前段争用访存。预取仅改善 GM→L2 供数，
不能替代 MTE 到 UB/L1 的搬运，也不保证数据一直驻留 L2。

本轮 MemoryAccess 已完成，但额外 `l2_cache.csv` 仅含测试夹具的 TensorMove，
没有对应 CSA 的记录，不能拿这些行计算 CSA 命中率。补采 `AiCMetrics.L2Cache`
仅用于缺失的实际 CSA 命中/缺失计数；PMU 外层时间不替代完整 L1 首尾耗时。

候选仍先 B16/B28：至少一档相对同轮已验证单 SPMD 净省 20 μs，另一档无
明确退步，再扩档；最终五个 128K 档相对多 task 完整 L1 降低 30% 的目标不变。


## 本机功能验证结果

- Simpler binding 与 TMR/HBG runtime 已完成独立重建，20 项编译路径 UT 通过。
- PyPTO core/torch adapter 已重建，15 项预取代码生成、两套内存规划与等待语义 UT 通过。
- 完整 24 AIC / 48 AIV：TMR/HBG 的 eager（每 AIV 提交1次）和 ACLGraph replay
  （每 AIV 连续提交65次）四场景均 PASS，各100轮变输入。业务 Tile 跨预取存活，
  Cube/Vector 输出与完成保护区均精确通过。不是全16项上游矩阵的本机复测。
- 实际环境：torch 2.10.0+cpu、torch_npu 2.10.0.post2，CANN 9.2.0-beta.2。
  任务和完整结果见 MIX_TASKS.json / STATE.json，各用例没有插入诊断 C++ 或换 ISA。
- 完整 CSA 候选的动态 CPU 编译通过：一个 MIX、sync_start 与动态根轴保留。
  生成代码明确使用驻留通道、256B scratch、1us/20ms等待适配和8MiB物理区间。
  1896个Vec分配根地址均从256字节以后开始；25个动态子视图未做常量地址求值，
  不能把这些视图误说成静态分配。具体范围见 OB_CODEGEN_AUDIT.json。
- B16/B28 完整精度、图与五窗 L1 对照已提交，尚无性能采用结论。

构建附带发现 libbacktrace 的空并行参数生成裸 make -j，已向主仓提交
[PyPTO #3083](https://github.com/hw-native-sys/pypto/issues/3083)，附复现和修改建议，
未提PR、未加标签。复现脚本显式给定预算；它与SDMA的设备修复无关。
