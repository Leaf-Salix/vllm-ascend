# 连续 Indexer cache 与 CSA 本体优化（2026-09-27）

最新优先级：参考 pypto-lib 优化 CSA 本体，入口拆分和出口写回暂缓。
所有当前计时均为单卡正式 layer 4、合成历史、TP1/S6、mode=2、atomic=1、确定性级别0、无 EPLB；
5次预热后20次无 profiler 图重放。三段独立图不能相加冒充同一次完整区间。

- [已有计时汇总](timing_summary.csv)：完整路径与本体分列，全部样本均值及中位数均保留。
- `baseline_cd1fdaa1`：冻结旧工作树的 8K/B16 对照。
- `v0_split`：仅按物理页分离 key/scale，Score 仍分页读取。
- `v1_contiguous`：Torch 按请求页序排列 key/scale，Score 连续读取。
  保护页数量后来发现不足以覆盖所有尾块，新候选已按完整读取 tile 扩大；v1 仅作为阶段结果。
- `v2_weight_bypass_fixed`：Q/KV/O-A 权重读取策略候选，8K/B16 本体835.24 μs，
  相比v1 847.06 μs改善1.4%；完整路径仍1131.98 μs，尚未达到端到端目标。
- `upstream_direct_cap16_h8192_b16`：最新上游 direct-score 的合成输入参考，均值810.20 μs。
  编译容量16、FP32残差/scale，与接入侧容量64、正式层权重存在差别，不能当严格A/B。
  本地PyPTO缺少FIXPIPE接口时，脚本在隔离副本删去B≥64才进入的分支；实际B16分支保持原样。
- `v3_oproj`：上游 O projection 分块，8K/B16 本体817.61 μs，128K/B16本体1863.97 μs；
  后者中位数1611.49 μs，长尾未解决。完整路径仍慢于Native。
- `v4_buffered_score`：N768/FIXPIPE FP16 双缓冲，128K/B16 本体均值1576.93 μs、p50 1580.39 μs、
  p95 1817.54 μs；8K/B16 保留direct分支，本体828.49 μs。仍未达到Native或整模型验收目标。
  代码在 `5523ff0d`，前置桥接/投影为 `f35c9fc4` / `be42f262`。
- `fixpipe`：PyPTO main #2838 的本地移植，提交2a4e09ff；35个CPU用例、一个A3用例通过。
  v3起使用新工具链，移植前后的数字分段记录。

已有泳道可直接下载：

- [v1 128K/B16](v1_contiguous/h131072_b16/swimlane/dfx/merged_swimlane.json)
- [v4 128K/B16 正常窗口](v4_buffered_score/h131072_b16/swimlane/dfx/merged_swimlane.json)
- [v4 128K/B16 长尾窗口](v4_buffered_score/h131072_b16/swimlane/dfx/window_1/merged_swimlane.json)
- [上游 8K/B16 参考](upstream_direct_cap16_h8192_b16/swimlane/dfx/merged_swimlane.json)
- [物理拆分 v0 8K/B16](v0_split/h8192_b16/swimlane/dfx/merged_swimlane.json)

其余窗口在相应 `dfx/window_N`。带DFX的时间只用于解释任务差异；性能主数字取各 timing/report.json。
已有的六档 token/DSpark结果不覆盖本轮新候选；当前没有新的16卡验收结论。
