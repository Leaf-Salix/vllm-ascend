# Native成对KV搬运：生成代码先导

2026-09-27，基底 `cd910e2c`，复用8K/B40固定Native输入。
两项仅修改诊断程序生成的C++；生产算子、PyPTO、Simpler、PTOAS和PTO-ISA均未改。
结论是独立诊断的小幅变化，尚无完整CSA收益证据，暂不扩展PyPTO接口或接入生产。

## Native差异与表达能力

Native `sparse_attn_sharedkv_scfa_block_vector.h::CopyInKv` 计算两个物理地址，
跨度合法时按地址升序发出blockCount=2的DMA；无效索引、重复地址、跨度或边界不满足时回退。
`ProcessVec0L` 每16行左右写出一次，并轮换两个UB缓冲。
PTO则逐行DMA，收齐每AIV的64行后写出工作区。

本次浅拉取检查PyPTO main `b046b15c`：`gather_row` 仍只有原来的连续行窗口与
`valid_shape` 参数，没有动态源行距。现有PTO-ISA的DMA本身支持该能力。
没有为了接口先改工具链，而是用生成代码证明收益大小。

两项候选：

1. 成对DMA：保留64行UB与原有同步边界，将两条合法行按物理地址升序一次读入；
   不合法的组合仍逐行读取，原来的UB清零保留。
2. 成对DMA＋16行写出：继续使用64行UB，拆成四个互不重叠的16行区域，
   每区读完便写GM，使MTE2/MTE3有机会交叠；保留结束处原有同步。
   这并非Native两块16行UB的完整复制，不宣称降低了UB占用。

固定输入共有61,440对，全部可合并，其中30,561对交换读取顺序。
compressed KV DMA指令次数从122,880变为61,440；逻辑读取量仍为120MiB，**流量没有减半**。
顺序变化只发生在两个有效候选之间；本探针只覆盖该固定真实权重合成历史输入，未做通用尾块验收。

## 单次独立PMU结果

program模式、事件组2，各变体各一次；24个AIC/48个AIV记录完整。只报告核心cycles，
不使用系统计数器频率将它们换算为μs，也不将它们当作完整CSA稳态性能。

| 实现 | qk_pv AIC平均cycles | 对基底 | qk_pv AIV平均cycles |
| --- | ---: | ---: | ---: |
| 当前保留实现 | 513120.83 | — | 517973.79 |
| 成对DMA | 502897.08 | −1.99% | 507986.48 |
| 成对DMA＋16行写出 | 498027.46 | −2.94% | 503067.31 |

单次2%～3%差异不足以确认稳定的完整CSA收益，因此没有扩测七档或增加公开API。
它也不能证明成对DMA没有价值；后续若成为关键瓶颈，可用同一固定输入继续隔离测量。

两项对当前PTO均只有10/7,864,320个元素不同，max_abs=0.000244140625、RMSE=1.4683662e-7；
对Native max_abs均0.0009765625、RMSE=6.3484288e-5，非有限值0。
变化来自候选对内重排影响归约舍入，不能称为逐bit等价，也没有完成整模型验收。

## 证据与复现

- [生成代码修改与运行脚本](run.py)，准备步骤从Git固定读取 `cd910e2c` 的Sparse源码。
- [成对DMA补丁](generated.patch)、[分批写出补丁](chunked/generated.patch)。
- [完整计数器汇总和误差](summary.json)、[离线汇总脚本](summarize.py)、[输入配对统计](input_pair_stats.json)。
- 原始PMU：[成对DMA](build/dfx_outputs/pmu.csv)、[分批写出](chunked/build/dfx_outputs/pmu.csv)。
- [单卡执行](run.sh)、[分批写出执行](run_chunked.sh)。

先source固定环境，在CPU执行 `python <本目录>/run.py --prepare`；
第二项加 `--chunked`。均须见 `PAIR_DMA_CPU_COMPILE_OK`，再用task-submit执行对应shell脚本。
设备任务 `task_20260927_162723_20380203069`、`task_20260927_163121_205364024952` 均退出0。
准备过程只修改本目录生成物，不修改生产算子；大输入和生成目录不整体加入Git。
