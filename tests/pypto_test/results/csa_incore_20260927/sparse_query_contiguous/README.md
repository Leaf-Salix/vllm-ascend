# Sparse Attention连续query区间先导

2026-09-27，基底 `cd910e2c`。仅将每核query遍历由间隔24，改为
`[core*T//24, (core+1)*T//24)` 的连续区间，参考Native按batch/gS1起止区间处理工作。
任务数仍为24 AIC＋48 AIV，跨任务依赖和attention算术不变。
Native还会沿S2切分，并不宣称此分配与Native metadata完全相同。

CPU ABI/PTOAS/AICPU编译通过；任务 `task_20260927_163543_207106025514` 退出0。
固定Native输入Sparse输出逐bit一致；B3/S6短历史尾块解析检查通过。
随后以8K/B40、正式layer4权重＋合成历史、mode2/atomic1/确定性0、无EPLB测试，
5次预热/20次无profiler采样，另外4个DFX窗口。

| 指标 | 当前基底 | 连续遍历 |
| --- | ---: | ---: |
| CSA本体均值 μs | 1361.40 | 1349.62 |
| 本体p50 / p95 μs | 1363.31 / 1390.32 | 1345.46 / 1389.94 |
| qk_pv AIC窗口均值范围 μs | 301.01–318.44 | 301.89–309.44 |
| merge_norm AIV窗口均值范围 μs | 39.66–40.70 | 38.08–39.08 |
| 同轮Native μs | 1401.18 | 1445.39 |
| 完整PTO μs | 1708.50 | 1685.56 |

本体变化−0.87%，qk_pv范围重叠，轮次间Native也有变化；不能认定稳定的核内及本体收益。
没有扩大测试，候选已撤回，正式实现仍为 `cd910e2c`。
保护区和索引结构通过、非有限值0；完整CSA输出max_abs=0.03125、RMSE=0.003292603，
Top-K集合替换901。零容差对Native仍FAIL，未做整模型验收。

- [补丁](candidate.patch)、[复现命令](run.sh)、[计时/误差/原始泳道路径](report.json)。
- 原始结果：`../../csa_split_optimization_20260927/sparse_query_contiguous/h8192_b40/`。

下一项依据更直接：Native `PreloadPipeline` 的gloop跨query延续，只在分配范围末尾追加drain；
当前PTO每个query都重新运行5＋2/3拍的流水并清空。
应先在核内合并相邻query的流水、保持各query的量化和归约顺序，
核对三槽复用和最后一个query的输出发布，再做独立代表档验证。
