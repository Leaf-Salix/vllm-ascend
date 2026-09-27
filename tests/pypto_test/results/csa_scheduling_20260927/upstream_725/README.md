# 调度固定参照：727.98 μs历史泳道＋最新pypto-lib源码

2026-09-27按用户要求，在调度阶段持续对照“725 μs”上游泳道；源码只参考官方最新main。
本轮 `git fetch --depth=1 upstream main` 确认main为
`216456332c2a74d89cca23b7824dab264ce34bff`，本地checkout与它一致，无需更新源码或安装。
这不证明旧泳道来自该提交。后续调度候选须分别记录源码参考版本、历史图边界及自身实测。

## 输入与计时范围

- 历史上游图：原 `shangyou-merged_swimlane_20260924_005402.json`，从Git `30795c69^` 恢复。
  只有Worker记录和 `tensormap_and_ringbuffer` 标记，缺源码版本、权重、NZ mode和工具链配置。
  历史对照按B16/S6/H8192使用，但原图本身没有完整输入证明；不能把以下差异当同输入严格A/B。
- 当前源码：`c7a52af5`，性能版、8K/B16/S6/TP1，A3 device0，正式第4层权重及合成输入/历史，
  mode2/atomic1/deterministic0、EPLB关闭、第二CSA层复用metadata。
- 当前任务 `task_20260927_175341_273207222150` 退出0；5次预热、20次无profiler计时，另4个DFX窗口。
  Worker首尾仅覆盖CSA本体，不包括拆分/写回，也不等于无profilerNPU Event时间。
- 每侧首Worker receive各归零。kernel-duration含核内搬运与同步；setup包括接收后的准备/依赖门控。
  同类任务启动分散包含多波执行与竞争，不是纯调度器软件开销；任务窗口不求和。

## 当前结论

| 不重叠分段 μs | 历史上游 | 当前4窗口 |
| --- | ---: | ---: |
| 首Worker→norm结束 | 66.62 | 86.82–91.48 |
| norm结束→Sparse首receive | 317.14 | 321.56–358.06 |
| Sparse首receive→merge结束 | 184.90 | 168.10–179.84 |
| merge结束→末Worker | 159.32 | 182.10–185.88 |
| 总Worker窗口 | 727.98 | 774.98–809.56 |

1. **前段约多20–25 μs。** 当前需要BF16→FP32的hc_widen（12 Worker，约10.64–11.18 μs窗口），
   原图没有；hc_pre_linear核内已更短，却晚约20 μs开始。额外转换及前置依赖必须单列，不能归咎矩阵乘慢。
2. **Q/Indexer仍有任务竞争和启动分散。** Indexer Q投影核内18.89→13.31–15.88 μs，
   启动分散1.40→31.46–84.42 μs；Q_B核内36.44→34.50–40.05 μs，启动分散11.66→15.04–52.42 μs。
   Q_A提前没有自动解决后续分支的竞争。当前Score改为双query/片上Cube规约，不能按同名旧算子直接比较。
3. **Sparse阶段不是当前对这份旧图的主要退步来源。** 完整分段已缩短5–17 μs，
   但merge_norm核内16.57→24.47–26.64 μs仍更长。这不撤销与Native之间的核内差距。
4. **尾段约多23–27 μs。** B16 O_A均为64 Worker，核内20.19→27.73–29.97 μs，
   启动分散44.14→63.56–66.84 μs；O_B核内更短但未补回O_A阶段损失。
   不能把尾段差值全叫调度开销。旧图无法确认权重/NZ模式，不能从图反推当前NZ必然慢于ND。
5. **额外工作更新。** Worker总数983→1131：hc_widen +12、row_offsets +1、boundary_init +16、
   Q_A split-K +48、KV投影 +24、head_coefficients +48、移除rope_swap −1。
   当前没有旧接入侧的48 Worker key_repack；Score/publish与当前Score/merge虽然数量相同，算法不同。

无profiler当前CSA本体均值 **832.26 μs**，p50 **799.37**，p95 **820.80**；
20次中一次 **1424.66 μs** 长尾保留，不删除样本、不拿p50代替均值。
Native均值929.81 μs；PTO含拆分写回完整均值1100.77 μs，仍慢于Native。
Native输出max_abs=0.03125、RMSE=0.003320147，Top-K替换366；保护区/结构/非有限值检查通过，零容差仍FAIL。
当前单卡结果不能替代16卡token/DSpark验收。

## 最新源码对照与下一项

最新pypto-lib `models/deepseek_v4_flash_dspark/decode_o_proj.py` 的O_A沿行块×N块并行；
每组quant依赖整组O_A，O_B依赖该组quant，各组按O_A→quant→O_B登记。
当前ND相同，NZ因为早期偏移非负校验而只按N分块、行块放核内循环。
B16仅一行块，任务数相同；B24/32/40超过128行，差异才会改变任务分配。
此前只试“全部O_A先登记”未获明确收益，已撤回；下一项复用上游二维任务网格，保留Native NZ物理权重，
列偏移用 `max(remainder,0)` 明确非负（合法block下值不变），不改K规约或每组量化依赖。
该项属于任务粒度/负载分配，不再追加新的核内算术优化。

## 关键路径及证据

使用pypto-lib [critical-path技能](../../../../../../pypto-lib/.claude/skills/critical-path/SKILL.md)和当前Simpler官方解析器，
复用本轮现有4个level4窗口，未额外跑设备。每窗AICore/Scheduler/解析后各1131行，逻辑block数量吻合；
见[记录完整性](artifacts.json)。分析器把4个独立窗口显示为4个“rank”，本报告不将其解释为多卡。

[关键路径完整表及等待归因](critical_path_summary.md)固定选择最后一个window_3，仅此单次调用；
dispatch→finish779.64 μs、AICore首start→末end774.76 μs，Static CPM572.54 μs。
该窗口是预先固定序号，未选最快rank；工具模板的single/fastest措辞不适用于跨窗口筛选。
工具将逻辑任务跨度中的非重叠部分称compute，它仍包含SPMD启动跨度，不能等同核内指令时间。
dummy生产者缺物理时戳的KV/Compressor行标记unverifiable，相关ready→dispatch数值不作完整依赖归因。
其余观察中，Indexer Q反量化前有14.96 μs end→FIN、9.06 μs FIN→dispatch；
未证明全引擎descriptor饱和，不点名同时运行的任务为确定阻塞者。
历史上游无Scheduler View，不能生成同口径CPM或计算纯调度软件开销差。

- [四窗口完整逐任务对照](comparison.md)、[机器可读数据](report.json)、[分析脚本](compare.py)、[设备命令](run.sh)。
- [历史727.98 μs原图](../../csa_baseline_20260926/upstream_gap/upstream_worker_trace.json)。
- [当前第0窗口](../../csa_split_optimization_20260927/schedule_c7a52af5/h8192_b16/swimlane/dfx/merged_swimlane.json)。
- [当前无profiler报告](../../csa_split_optimization_20260927/schedule_c7a52af5/h8192_b16/timing/report.json)。
