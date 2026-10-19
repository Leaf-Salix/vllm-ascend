# TND 精度版 SPMD：真实整模型精度和性能记录（2026-10-20）

## 结论和实现身份

当前分支为 `dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
验收实现固定为 `e9784b9e5c4e4de8daf9917d88c8571a0eabc626`，
冻结精度控制实现为 `7d5202300a1b3828ec9ecd01278f45654a51b7c5`。

本轮两档真实整模型的所有已保存 token 与接受事件均与既有 Native 金值一致，且零抢占。
8K 通过后才运行 128K。未重新运行整模型 Native2。
SPMD 版覆盖完整 HC-pre→HC-post，在一个 PTO 根中融合 HC-pre、输入 Norm、WKV 和
Indexer 权重投影为一个 SPMD 前缀；后续仍是原 PTO 子调用。
这不是整条 CSA 链路单一 SPMD kernel。原精度实现的 43 个运行文件未修改。

## 精度与性能同表

单位 ms/forward。Native 与冻结精度版是历史实测，并非同场交错 A/B。
提交/活跃并发指每个 DP rank；128K 的另外 3 条请求等待入场。

| 配置 | 提交/实测活跃 | 校验 token / 差异 | 接受事件 / 差异 | Native forward | 冻结精度 forward | SPMD forward | SPMD 比 Native 慢 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8K/B40 | 40/40 | 491,520 / 0 | 105,152 / 0 | 89.482 | 108.405 | 107.231 | 19.84% |
| 128K/B24 | 24/21 | 294,912 / 0 | 61,568 / 0 | 79.730 | 91.434 | 91.153 | 14.33% |

SPMD 相比冻结精度控制的 forward 原始均值分别减少约 1.08%、0.31%；
这是跨任务、跨时段比较，不足以认定稳定优化收益。
早期 panel 实现约 651 ms 的旧结果不是本轮 SPMD 的直接父基准，不作为收益分母。

| 配置 | Native cycle | 冻结精度 cycle | SPMD cycle | SPMD forward 标准差 |
| --- | --- | --- | --- | --- |
| 8K/B40 | 105.575 | 124.058 | 122.860 | 0.620 |
| 128K/B24 | 92.293 | 103.952 | 103.513 | 0.719 |

## 测试方法与范围

- 真实模型 `DeepSeek-V4-Flash-0731-w8a8`；TP=1，DP=EP=16。
- DSpark 出5验6；不强制接收步长等于3.8。
- 21 个 CSA 层使用本 SPMD 精度实现，20 个 HCA 层保留 Native。
- FULL graph，NZ2，`super_kernel_optimize=False`，EPLB 关闭，Native 确定性 level1、HCCL 确定性开启。
- 每 rank 暖身生成192 token，再执行3轮各192 token。总校验数包含暖身，不能只按3轮计算。
- 性能窗口每轮先跳过8步，再收集20个 forward 和19个 cycle 间隔。
  每步取16 rank 最大值后汇总全部轮次；不删除离群样本，不用完整任务时长冒充算子延迟。
- 8K/B40：40条全部活跃，FULL buckets `[6,240]`，实测240行。
- 128K/B24：提交24条、实测21条活跃，FULL buckets `[6,144]`；126真实行、144全局graph容量。
  这次整模型测试本身不能证明每请求不补S6；该能力另有短query及混合1～6行单层FULL证据。
- 题目、顺序与聊天后缀沿用既有 nalinaly 对齐方案；Native 前缀/cache bank 是本项目生成的金值，
  未证明与 nalinaly 原 bank 的全部字节一致。

8K 的三个候选窗口均逐字段匹配历史 Native 三个窗口。
128K 固定 C1→N1、C2/C3→N2；16 rank 合计960条同阶段观测记录，
除调用计数 `call` 外全部 CPU 字段严格一致，包括请求key、query bounds、positions、input_ids。
不归一化0/-1哨兵，不跨阶段择优挑选。CPU 观测相同不等于所有实际设备 Tensor 已逐bit导出比较。
所有输出token、接受事件一致不代表任意输入下所有内部中间量逐bit一致。

## 环境与复现

CANN9.2 beta2；Torch2.10.0+cpu，Torch-NPU2.10.0.post2；vLLM0.25.1，vLLM-Ascend0.25.1rc1。
沿用 PyPTO `3e87a843`、Simpler `a54c0509`；不更换 SDK 或 PTOAS。
运行时设置 `PYTHONNOUSERSITE=1`，并显式传入已记录的 `ASCEND_HOME_PATH`、`PTOAS_ROOT`。
编译使用相同 checkout ccache、`CMAKE_BUILD_PARALLEL_LEVEL=4`。

证据目录：`evidence_tnd_precision/spmd_real_model_20261020/`。
两份 `*-raw.tar.gz` 各包含202个文本文件：候选全部16 rank报告、Native 金值token与事件、
历史 Native 和直接冻结精度控制报告、完整helpers、3379文件源清单、57个实际运行源码及审计。
`manifest.json` 固定归档哈希；`summary.json` 保留全部计时样本、逐轮均值、标准差。
完整源码清单 SHA256 为
`c86c7b9cc90e710ce5a87006ef2ba69809aa95839e1d9a64692f25d78604e06c`。

从归档恢复 `tools/`，按其中helper manifest校验文件，再使用 `configuration.json` 记录的 bank、
Native gold、static metadata 和源快照启动。模型权重、完整 bank 和二进制缓存不包含在小型文本证据包中，
需要预先准备；不能仅解压这些文本就重新完成整模型生成。
`tools/` 内原启动器保留了完整参数；复现时应将原实验根映射到自己的独立实验目录，保持源与金值身份。
每项任务调度上限780秒、内部timeout750秒；8K先通过再运行128K，首差立即停止。

离线复算无需 vLLM/NPU：

```bash
python -m pytest tests/pypto_test/test_precision_spmd_real_model_evidence_cpu.py -q
```

检查全部归档SHA、57源码、每rank每轮token/事件、实测入场轨迹和原始性能样本。
设备结果为 `task_20261020_013440_19865192674`（8K）与
`task_20261020_021808_9597610351`（128K），均已完成、exit0，实际执行未超过15分钟。

## 使用限制与后续候选

工作区和graph replay必须按设备执行顺序串行，不支持并发stream借用同一工作区。
静态包资格、原生metadata producer与cache生命周期检查保留；不满足条件拒绝替换。
当前仍明显慢于历史Native，尚不能宣称达到nalinaly性能版速度。

固定HC权重预打包是另一组新增文件中的候选，只完成独立前缀对拍时不能视作本报告的整模型通过版本。
本报告的所有结果固定绑定上述e978提交，不随候选后续修改而重标。
