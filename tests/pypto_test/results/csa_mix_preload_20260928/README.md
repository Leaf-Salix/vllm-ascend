# Simpler #2389 在当前 CSA kernel-mode 的评估

本轮在 HC 输入加宽/RMS 融合候选收尾后进行设备对照；准备期间不修改其冻结环境。
目前只有源码分析和隔离构建，**尚无当前 CSA 的 PR 性能结论**。

## 改动及适用范围

[上游 PR](https://github.com/hw-native-sys/simpler/pull/2389) 的固定版本为
`ad9bb31e7854bfde199f4f0ecfc944cb9cac5f08`，包含 `23d66c71`、`ad9bb31e` 两个提交，目标为 main。
本地以已验证的 kernel-mode 分支 `a54c05095` 为基底移植，候选为 `52c4e019e`。
18 个文件只加入原 PR 的调度与配置传递；3 个上下文冲突文件按当前代码补入相同字段/赋值。
配套 PyPTO 只将 runtime 子模块与 SDK 标记同步到候选，独立构建绑定及 torch_npu 扩展。

普通 ready 队列中的 MIX 任务如果进入忙碌 cluster 的 pending 槽，就不能再转移到其他更早空闲的 cluster。
PR 使用每个 kernel 的耗时 EWMA 减去已执行时间，只有估计剩余时间小于门限才允许预加载，
否则留在共享 ready 队列。`SIMPLER_MIX_PRELOAD_MAX_REMAINING_US=50` 为上游默认，`0` 关闭门限。
首次无历史样本的 kernel 不拒绝预加载；跨重放保留估计。

| 当前任务 | 实际路径 | PR 直接影响 |
| --- | --- | --- |
| 128K Score | MIX，sync_start=True，allow_early_resolve=False | sync_start 放置路径不变 |
| 8K Score | MIX，allow_early_resolve=True | 只有回到普通 ready 的 pending 放置才受门限影响 |
| Sparse qk_pv | MIX，allow_early_resolve=True | 同上；已提前派发的任务不重新选址 |
| HC、QR、KV、Indexer query、输出投影等纯 AIC/AIV 任务 | 纯核任务 | pending 策略不变，但增加执行时长采样开销 |

不能把上游 HCA B16/S8 的收益直接用于当前 Native 权重接口、S6、原生分页 cache 的 CSA。
上游报告的约 5.7% 是其历史环境的 HCA 结果，其独立 CSA attention 对照接近中性。

## 已有泳道的检查

读取 `csa_qr_k512_20260928` 中 e58ddc94 基线的长短 B16 各 2 个窗口，未运行新设备测试。
Score/Sparse 的已匹配 Scheduler dispatch → Worker receive 间隔最大约 7.32 μs，未观察到 >50 μs。
部分 AIV Worker 记录缺少匹配的 Scheduler 记录，且这 4 个窗口不覆盖无 profiler 的偶发异常。
因此只能说这些窗口没有复现 PR 所针对的长等待，不能据此证明它没有收益或长尾已解决。
Worker local_setup 和 kernel-duration 分开，不能把这里的等待或 setup 全部解释为调度成本。
[逐窗口等待样本及未匹配数量](existing_waits.json)。

## kernel-mode 需要单独确认的限制

- 当前 `kernel_launch_owner.cpp` 复用 `arena_banks_[0]`；执行器每次使用当前 callable 的函数表，
  `runtime_reset_for_reuse()` 保留 SchedulerState，`destroy()` 也不清空新增估计表。
  PR 的表只按局部 func_id 索引，没有 callable 身份。不同 callable 共用 arena 时可能沿用其他 kernel 的估计。
  这影响调度预测，不能直接定性为输出正确性缺陷；整模型验收必须观察实际多 callable 场景。
- 上游估计表是跨调度线程读写的普通 uint32_t，存在 C++ 数据竞争问题。
  首轮性能评估保留原 PR 行为；若保留进入正式分支，需要处理共享采样的并发语义。
- 已核实 `decide_slot_transition()` 会在 pending ACK/FIN 时把 `running_done` 置为 true，
  而 PR 仅以 `running_done` 为采样条件，会把后续任务的时间计入前一个 kernel。
  [上游评审也指出这一点](https://github.com/hw-native-sys/simpler/pull/2389#discussion_r4056116140)，
  此处结论来自当前移植源码核对，不能把该估计称为精确的 kernel 执行时长。
- `0` 仅关闭门限，新增计时和 EWMA 采样仍执行；所以仍需原 a54 运行时对照，不能把 `0` 当成完全原始二进制。

## 最小设备对照

固定本轮最终选定的 CSA 算子、PyPTO 算术、PTOAS 0.66、PTO-ISA 327cd586、CANN 9.0，
layer4 正式权重、合成历史、mode2/atomic0/det1、EPLB 关闭。
8K/B16、128K/B16 各比较：原运行时、PR 门限 0、PR 门限 50。
每组先预热，再保存全部无 profiler 样本、均值/中位数/P95/最大值；独立 DFX 检查任务排队与启动分散。
随测试比较 8 类状态及保护区，不重新生成 bank，不先扩大七档或 EP16。
只有长短档收益成立且 P95 不恶化，才继续评估多 callable 和真实 EP16 forward。

构建目录：`.cache/simpler-pr2389-a54c05095`、`.cache/pypto-pr2389-3e87a843`。
原生产安装与源码未切换，未将本实验提交推送到上游。
[执行脚本](run_layer.sh)、[隔离环境入口](isolated_case.py)、[结果汇总脚本](summarize.py)。
