# 初始化分离Indexer cache：对照结果与模型验证

基底2a740c1f，正式层4权重、mode2、S6、第二CSA metadata复用；两侧不改变算术、规约或任务调度。
各候选在独立工作树，主工作树仍保留原Native布局。Native模型及PTO precision初始化不改。
实验修改只命中PTO performance的A3 C4 Indexer，raw总字节及所有权不变，无逐步拆分/写回复制。

## 1. 仅分离物理key/scale池：没有足够收益

`.cache/csa-source-split-2a740c1f`，同一allocation前段全部key、后段全部scale；仍逐物理页读取。
页表保留原fixture的非连续/反序页。任务task_20260927_233915_1485984376退出0。
第一次task_20260927_233823_14812137747在source环境时因set -u与ATB脚本的ZSH_VERSION兼容问题退出，未开始case；修正shell后提交上项。

| 档位/状态 | 原布局CSA μs | 分离CSA μs | 变化 | 原/新P95 μs |
| --- | ---: | ---: | ---: | ---: |
| 8K/B16 常规 | 806.23 | 794.30 | −1.48% | 828.40/806.16 |
| 8K/B16 压力 | 835.95 | 827.85 | −0.97% | 866.86/843.32 |
| 128K/B16 常规 | 1244.74 | 1250.62 | +0.47% | 1268.18/1272.30 |
| 128K/B16 压力 | 1303.78 | 1313.67 | +0.76% | 1318.12/1334.68 |

仅这一修改不足以采用新布局，且没有消除PTO对缓存访问状态的敏感。
[全部样本与Native控制](summary.json)、[代码补丁](candidate.patch)、[设备命令](run.sh)。

## 2. 分离后合并连续四页：长档取得超过5%的单卡收益

`.cache/csa-source-coalesced-2a740c1f`，基于分离方案，确认四个物理页连续且全部有效后，将四次32行读取合并为一次N128读取。
非连续页和尾部仍走分页回退，不假设生产页表天然连续。
本轮两侧fixture都在初始化时给予每个请求连续Indexer页，其他cache不变；用来判断连续性有无价值，
**没有修改生产请求分配策略，不将这一诊断输入称作真实模型的页分布**。
任务task_20260927_234828_15573393332退出0。

| 档位/状态 | 原布局CSA μs | 合并读取CSA μs | 变化 | 原/新P95 μs |
| --- | ---: | ---: | ---: | ---: |
| 8K/B16 常规 | 797.68 | 782.29 | −1.93% | 817.02/800.52 |
| 8K/B16 压力 | 832.41 | 809.25 | −2.78% | 859.58/843.88 |
| 128K/B16 常规 | 1250.40 | 1179.34 | −5.68% | 1261.90/1192.46 |
| 128K/B16 压力 | 1303.64 | 1230.52 | −5.61% | 1333.14/1252.12 |

本轮与第一轮fixture页顺序不同，不能跨表相减声称某一行代码的收益。
[全部样本与Native控制](coalesced/summary.json)、[完整补丁](coalesced/candidate.patch)、
[连续页诊断输入](contiguous_case.py)、[设备命令](coalesced/run.sh)。
CPU初稿用条件pl.load引入不支持的Mat→Mat移动；改为向既有L1 tile执行gather_row后完整PTOAS/AICPU编译通过，没有改工具链。

## 检查范围与口径

两轮均先单卡B4的8K/128K固定规约检查，再进行B16性能测试。
使用251种随物理行变化的scale，改造前后Native控制及PTO的8类逻辑输出/cache/state逐元素零差异；
候选同地址A→B→A图重放通过，metadata/保护区通过。
这证明所测cache变更数值中性，不代表PTO与Native算术全部逐bit一致。

CPU直接调用真实runner初始化函数，验证Native模型和PTO precision仍为交错布局、PTO performance选择分离布局；
共享所有权、偏移、保护区及实验ABI拒收旧布局检查通过。
[CPU检查](check_initializer.json)、[脚本](check_initializer.py)。

性能两侧每组5预热20次，全部保留；先常规，再于计时开始前写入384MiB独立缓冲制造压力。
API查询本卡L2为192MiB，压力写入不计时；这不保证完全冷缓存，也不模拟全部EP16。
每轮在8K先原版后候选，128K反向；不是同一进程交错A/B，Native控制波动和全部样本保留。
[计时驱动](pressure_case.py)、[仅测试用计时钩子](baseline_harness.patch)、[汇总](summarize.py)。

## 3. 真实16卡模型：当前候选没有取得forward收益

task_20260927_235803_165209211208退出0，代表档全部采集完成。两侧mode2、atomic1、det0、HCCL确定性关、EPLB关，
容量40、捕获全部六档；warmup后连续10步纯decode forward，16rank全部满档，无丢弃慢样本。

| 档位 | Native forward ms | PTO forward ms | 变化 | Native/PTO P95 ms | 验收 |
| --- | ---: | ---: | ---: | ---: | --- |
| 128K/B16 | 73.497 | 76.283 | +3.79% | 74.293/77.287 | token一致；15/16 rank DSpark统计不同 |
| 8K/B40 | 102.743 | 103.311 | +0.55% | 104.725/104.229 | token/DSpark通过；性能未胜出 |

两档共229376个token零差异。长档rank0接受数4800→4794，草稿数1005均相同；原因仍待隔离，不能称精度通过。
每步最慢rank均值分别慢3.67%/0.56%。长档P95/P50为1.010/1.011，短档1.021/1.009；
本轮反差不能主要归因于偶发P95拖尾。[逐rank结果](coalesced/model/RESULTS.md)、
[全部样本和配置](coalesced/model/forward.json)、[模型命令](coalesced/run_model.sh)。

独立rank0 Level0的CSA均值却为：长档Native1291.26/PTO1167.38μs，短档1430.92/1328.47μs；
对应43层主图区间76.091/73.389ms和104.862/103.824ms。这些是另一轮请求恢复后的3步profile，
不与正式10步forward混算。[解析脚本](coalesced/analyze_model_profile.py)、[63个CSA区间/档](coalesced/model/model_gap_rank0.json)。

CPU调用真实BlockPool与CompressAttentionManager，已复现同一组页三次恢复/释放时
“递增→递减→递增”。当前N128合并只识别四页递增，说明不同轮次可能命中不同读取分支，
但这个单请求CPU复现不是模型真实页分布。[复现](baseline_model/check_page_reuse.py)、[结果](baseline_model/check_page_reuse.json)。
task_20260928_001924_199312121354已退出0。原布局2a740c1f两档PTO，复用本轮Native控制：
128K/B16为75.979ms（+3.38%），8K/B40为102.907ms（+0.16%）；229376 token及32组rank的DSpark均通过。
[完整原布局模型结果](baseline_model/RESULTS.md)、[命令](baseline_model/run.sh)、[诊断补丁](baseline_model/diagnostic.patch)。
这排除了“仅scale提前读取就必然造成该DSpark差异”的说法，但不能由一次通过证明新布局一定存在功能错误；
原布局与分离候选的运行顺序及atomic规约波动仍需考虑。当前分离候选不合入。

bank恢复处的实际CPU页序记录确认，长档正式计时轮递增四页0/4096，递减3898/4096；
profile轮递增3754/4096。短档正式轮递增0/640，profile轮203/640。
这些是原布局实机的allocator记录；分离版本也未改allocator，但不能冒充对旧候选同一次调用的分支硬件计数。
[页序汇总](baseline_model/page_order.json)、[解析器](baseline_model/page_order.py)。

## 4. 正在验证：完整批次入场＋仅PTO新页排序

路由诊断同时发现旧测试虽为满档，Native/PTO step的请求position组成仍可能不同。
测试修正8dd737f4通过公开level0调度暂停、整批enqueue、DP CPU barrier和只恢复调度来对齐入场，
不卸载模型、不清空cache；计时轮记录CPU位置用于核对。两项CPU检查通过，新Native长档已取得16rank完整窗口，
每请求起始位置一致。新旧入场结果不混作同一轮。

task_20260928_004008_231719416500使用`.cache/csa-source-ordered-2a740c1f`，
双方采用新测试入口，仅PTO performance对Indexer新分配后缀排序，保留已存在历史、共享前缀、refcount及非连续分页回退。
四项CPU分配检查通过；kernel与已有coalesced候选相同，没有额外变更算术。
[候选完整补丁](ordered/candidate.patch)、[模型命令](ordered/run_model.sh)、[位置配对收集器](ordered/collect_model.py)。
模型任务仍在运行；若最终token/DSpark、同step位置或性能不通过，就不作为达标方案合入。

8K/B40的模型内CSA快而forward持平，另用固定f76b3ad4采集真实EP16专家路由；
[路由诊断](../csa_model_forward_f76b3ad4_20260927/routing_b40/run_retry.sh)只作归因，不把复制/同步开销混进性能成绩。

更接近上游的384/512行连续key/scale读取探针也已编写，CPU编译通过；它要求请求历史连续，
目前没有生产分配生命周期保证或设备结果，不能部署或作为已验证优化。先完成当前候选模型验证，再按收益决定扩大分配改造。
