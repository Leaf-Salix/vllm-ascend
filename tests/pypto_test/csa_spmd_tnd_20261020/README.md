# CSA 原生 TND 单 SPMD：实现与验证记录

实现包为 `vllm_ascend/ops/pypto/deepseek_v4_flash_csa_spmd_tnd`，完整链路包含
HC_pre、input RMSNorm、CSA、HC_post。过程记录位于
[实现日志](../../../vllm_ascend/ops/pypto/DSV4_CSA_SPMD_TND_IMPLEMENTATION_LOG.md)。

## 当前结论

- 原生 TND、真实服务路由、S1～S6、混长、图重放及连续状态已有设备证据。
  原 Native 数值差异保持原阈值单列，验收要求是无新增回归。
- 采用的新增输出分工只覆盖既有 B1 半叶域及长 G6、T 容量 64～96。
- 最新一致八档为 `cohort_output_balance_v4_result.json`，等权
  `GM(TND/BSH)=0.991928345`，耗时约降低 0.81%，**10% 目标尚未达到**。
- QA N64 移植 v83b 只完成开启/关闭 CPU 编译和关闭路径代码等价检查。
  WS/scale 移植 v84 已完成 v87 开启/关闭编译及 L0C 资源审计，
  独立页布局探针 v90 的 25 项 CPU 检查已通过，设备矩阵尚未执行。
  Score 核分工移植 v88 已完成开启/关闭编译、唯一写者及逆映射等待检查。
  其独立大 T 回放探针 v89 另有 20 项 CPU 前检通过，设备用例尚未执行。
  两项 QA 任务已排队，均尚无设备结果；WS 与 Score 核分工未提交设备任务，以上候选均未采用。
- 前 104 个设备任务均已终态，另有两项 QA 任务在排队。
  失败、无收益及不合法夹具的原始记录保留。

## 阅读顺序

| 文件 | 内容 |
| --- | --- |
| `GOAL.txt`、`GOAL_UPDATES.md` | 原始目标与用户后续调整，包括八档、10% 和阶段性提交 |
| `RESULTS.md`、`validation_summary.json` | 按终态及实际结果汇总功能和性能，不跨候选拼最快值 |
| `jobs.json` | 完整任务 ID、命令、输出目录及终态 |
| `latest_complete_cohort.json` | 显式选择最新完整八档报告，避免将单项试验充作全档结论 |
| `cohort_output_balance_v4*.json` | 已采用输出分工的八档、原始五窗和代码复用来源 |
| `cross_session_review_v83.json` | 两个会话的最新优化来源、迁移优先级和排除理由 |
| `recovery_state.json` | 当前交接状态，不替代原始测量结果 |
| `LOCAL_ARTIFACTS.json` | 本机原始报告、源码快照及大产物位置 |

## 复现入口

CPU 合同检查不分配 NPU：

```bash
bash tests/pypto_test/csa_spmd_tnd_20261020/run_cpu.sh -q \
  tests/ut/ops/test_dsv4_csa_spmd_tnd.py \
  tests/pypto_test/test_csa_variant.py \
  tests/ut/compilation/a2/test_acl_graph.py
```

设备任务统一经 `submit_probe.py` 调用 task-submit。完整历史命令在 `jobs.json`；
重新执行时必须使用新的输出目录，并明确指定候选源码。`run_probe.sh` 固定
launcher、保存实际源码快照和工具链指纹，拒绝覆盖已有输出。

原生等待需要 [依赖补丁](dependency_patches/README.md) 对应的隔离构建。
正式 BSH 基线是 `a241fa61` 加两处未启用 B1 分支的静态形状兼容修正，
来源与原始失败在 `bsh_baseline_a241fa61_short_typefix_manifest.json`。
冻结的旧基线及候选目录继续本地保留，不用当前源码替换历史测量输入。

性能取全 72 核最早 incore 开始至最晚结束，包含等待和阶段空隙，使用
五个成对窗口的中位数。B1 的 S1～S6 矩阵在固定 S6 采样之后执行。
`evaluate_s6_cohort.py` 要求恰好八档、同一冻结候选和基线、完整窗口与
工具链审计；复用旧测量还需严格的生成代码及 Host 编排等价证书。

## 归档与提交边界

Git 保存正式实现、测试入口、候选生成器和补丁、轻量结果、完整窗口及
本地证据索引。`runs/` 下的原始报告、编译目录、生成 C++、泳道、权重和
二进制继续保存在本机；清单中的路径不是已上传到 Git 的文件。

候选生成器保留其当时的源码前提，前提不符时会拒绝执行；不能将旧实验
直接套到更新后的正式包或跨会话整目录覆盖。后续每个实现、验收、采用或
否决阶段及时提交，不等待总体性能目标完成再统一提交。用户已指定推送目标为 `origin/dsv4-flash-pto-v0.25.1rc1`；
当前独立 worktree 跟踪此上游，前六笔累计提交已推送至 `aa2e2a41`。

本地探针回归 `test_score_owner_harness.py` 与 `test_ws_layout_harness.py`
依赖 `runs/` 中冻结的候选和探针；缺少这些本机产物时显式跳过。当前本机
分别完成 20、25 项检查，并确认两模块共收集 45 项。跳过不作为验收通过。
