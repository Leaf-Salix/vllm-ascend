# CSA 八档六列：AICPU 多 task 版本

按 [当前规范快照](SPEC.md) 汇总，结果见 [PERFORMANCE.md](PERFORMANCE.md)，
完整样本、来源和检查见 [PERFORMANCE.json](PERFORMANCE.json)。单位 μs，均为
单次完整链路跨度的中位数；百分比的分母为 TND-SPMD。

本轮 15 个补测任务均完成并退出 0。八行共 46 个有效数据格、2 个按规范不适用格，
保留 790 个原始样本，每格至少 10 次。原始 L1 与全流 CSV 重算、实际执行路径、
数值及保护区复核均通过，收据见 [VALIDATION.json](VALIDATION.json)。
PTO 两侧对照的已检查张量逐 bit 一致；这不代表与 Native 逐 bit 一致。

## 本轮固定版本

- TND-MPMD：`d263bc3f2`，独立计算 task，由 AICPU 管理跨任务依赖。
  不含已淘汰的阶段软同步，也不采用未保留的 v11/v12 实验。
- TND-SPMD：另一会话的 `score_group_order_v4` 冻结快照。混长分组重排的完整
  数值、同址 metadata/连续状态及服务验证均已完成；七个关闭分档与 combined_v1
  的生成代码相同。具体证据和终态见 [manifest.json](manifest.json)。
- BSH 两列：沿用上一版已验证快照，包括正式 B1 编译参数、150tick 等待和
  短档临时 Tile 静态类型兼容修正。本轮没有修改算法。
- PTO 统一使用公共已验证 SDK `csa-shared-20261022-logical-codegen`：
  PyPTO `26e94e056`、PTOAS `914863ed3`、Simpler `57e499238`。
- Native：复用 CANN 9.2 beta2 的实际 npugraph_ex、静态编译、SuperKernel、
  多流及 NZ2 路径记录，不依赖本次更新的 PTO SDK。

场景为规范的固定八档，正式第 4 层 W8A8/NZ2 权重、BF16、seed=1024；覆盖
`HC_pre → input RMSNorm → CSA → HC_post`。混长实际 T=56，无业务 padding；
BSH 两格按规范填 N/A。256K 使用真实 H=262144 的 cache、页表和容量。

## 复用与补测

[inventory.json](inventory.json) 是补测前的逐格清单，
[history_audit.json](history_audit.json) 说明排除原因。

- Native 八格各复用 30 次原始 Profiler 窗口。
- C05 的 TND-MPMD-L1 复用同一 v9 在两次任务中的全部 20 次有效窗口，包括
  后一次作为控制组的窗口，不使用该任务中的 v11 候选结果。
- C08 的 TND-SPMD 复用组重排 v2 的全部 10 次窗口。已有完整生成代码等价证明
  及后续 v3/v4 功能门禁，支持作为选定 v4 开启分档的记录。
- 其余 PTO 旧表使用前一个 SDK，保留原始标签，不能冒充当前公共 SDK 的结果。
  仅补这些格子；已有合格 L1 的侧跳过计时，仍参与必要数值及图检查。

每个新 L1 格预热 3 次，采集 10 次单 replay；全部有效窗口纳入中位数，不挑快值。
TND-MPMD 的含 AICPU 列另采 30 次全设备 Profiler，使用同进程、同图、同源码和
同输入，保留所有相关流；DFX L1 与外层 Profiler 分开，避免导出互相干扰。
编译、warmup、测试专用复位和读回不计时，算子内部 metadata/搬运/等待均计入。

每次设备任务检查 eager、图重放、采样后输出、TopK/score、cache/state 和保护区。
Native 原有数值差异单列保留，不把本单层验证解释为整模型 bit/token 一致。
Native 外层 device 与 TND-SPMD L1 的百分比仅按规范展示，不等于端到端吞吐收益。

## 证据位置

任务与命令记录在 [jobs.json](jobs.json)。本机 `sources/` 为冻结算子和测量入口，
`runs/` 保存各次结果、全部 L1 原始记录及外层 PyTorch profiling。
大型产物不进入 Git，JSON 保留可访问的绝对路径及任务 ID。
同一 attempt 禁止覆盖，公共 SDK 若变化，启动器会拒绝混用。
原场景清单中的 `reference_commit` 是输入构造历史标签，实际算子版本以本轮
manifest、冻结源码和每次结果的 `sources` 为准，不能将该旧字段当成运行版本。

```bash
# 只重汇总已有原始结果，不上卡。
python tests/pypto_test/csa_performance_20261022_aicpu/collect.py
# 只读复核原始计时边界、样本、路径及数值证据。
python tests/pypto_test/csa_performance_20261022_aicpu/validate.py
```

本轮只采集单卡数据，不新增优化、不运行 16 卡、不自动 push。

## 八档单次 L1 下载

已从所有有效样本中为每档选择最接近中位数的一次真实执行，导出到：
`/data/pyptouser/qinchuanyu/pto-eager/downloads/CSA_TND_MPMD_L1_8cases_20261022_AICPU/`。
八个 JSON 按档位编号，图内使用实际计算名称及输出组编号；每个物理核上的全部
incore 记录和任务间隙都保留，直接使用 incore begin/end，不把派发/本地准备混入条形。
命名依据、原始记录和每档所选时间见 [download_manifest.json](download_manifest.json)。
