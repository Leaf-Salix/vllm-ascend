# WO_A 跟随 Native NZ：CANN 9.2 集成与核内对照

状态：3b27c7fd完整编译/PTOAS/CCE/链接/load通过；task_20260928_231959_32180753279
已由auto分配card11执行长短B16，设备结果待完成，不提前宣称收益。

## 原因与边界

七档 e33d842a 采集期间，另一会话提交了 3b27c7fd：CANN 9.2 的 Native 三维
BF16 wo_a 实际是 NZ29，因此根布局跟随 Native 存法，避免转回 ND 的每层 64 MiB 副本。
这项修改已有权重/显存探针；探针不覆盖完整 CSA 的计算结果与核内性能。
旧七档仍明确标 e33d842a，不将该新提交默认为已被旧结果覆盖。

本轮直接复用提交，不额外改 tile、规约、cache 或调度；精度版不在设备测试范围。
基线 `.cache/csa-cann92-baseline-e33d842a`，候选 `.cache/csa-wo-a-native-nz-3b27c7fd`，
生产差异只有 nz_mode 和 Native adapter，详见 [candidate.patch](candidate.patch)。

最新 ops-nn 19614968 的 A3 TransposeBatchMatMul 为 ND/NZ 分开实现权重加载，
`pp_matmul_ein_sum_kernel.h:CopyTileB` 在 FormatB=NZ 时直接 GM NZ→L1 NZ；
本轮先验证现有 PTO `_proj_a_mm_nz` 同样直接消费 Native 格式29的效果。
pypto-lib 提供 O projection 分块/流水参考，但这里保留 Native 三维权重方向及分组，
不按上游私有布局重新打包；原地址借用与 kernel 声明必须同时成立。

## 最小受影响范围

- 128K/B16 与 8K/B16，同一卡，正式 layer4 权重和合成历史，物理页反序及变化 scale。
- 公共 CANN9.2、mode2、atomic0、det0、S6、EPLB 关；与七档相同工具链。
- 基线/候选八类状态精确比较、metadata/保护区、A→B→A 图重放；不改变算术容差。
- 两侧各5次预热、20次无 profiler 完整图计时；另各4个 DFX 窗口。
- 单列 O_A 核时/最慢核、必要多波工作量、包络，以及完整 CSA/P95；不把布局收益归给调度。
- 核实 Native/PTO wo_a 都是29且借用同一地址。跨实现 Native 数值和模型验收不在本轮声称通过。

复现：[CPU编译](compile.sh)、[auto队列设备入口](run_layer.sh)、[主机汇总](summarize.sh)。
编译先于正式设备计时；不在运行中的正式16卡计时旁主动启动重编译。
