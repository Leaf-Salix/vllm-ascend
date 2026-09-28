# 去除dummy并保留真实任务准入

基线fa53246c，生产算子仍为4ffccb7b；私有整包`pkg:dsv4_csa_direct_chain_fa53246c`。
本次是调度候选，不叠加未采用的分数驻留UB，也不改核内算术、cache、worker数或early/sync标志。

## 与第一次去dummy的区别

§372的全量替换仅取得约0.1%的加权CSA变化，未采用。
它不仅删除AICPU中转，也把两个空dummy改为无前置哨兵，可能改变消费者预派发资格；
长档weights提前29.745μs，Score反而延后13.975μs，不能将节点减少等同于整体收益。
`allow_early_resolve`属于生产者，消费者所有真实前置均允许时才具备预派发资格。

本次仍将四个dummy降到零，但用已有真实任务维持或增加准入约束：

| 位置 | 当前生产 | 本次候选 |
| --- | --- | --- |
| RoPE后投影分支 | RoPE→dummy→消费者 | 直接传RoPE TaskId |
| Attention Compressor投影 | RoPE/Q_A→汇合dummy→投影 | 显式deps=[rope_tid, qa_tid] |
| Q_B空前置 | 空dummy，禁止消费者预派发 | 真实RoPE前置，其early标志仍为false |
| weights空前置 | 空dummy，时间上可与Q_A重叠 | 直接依赖Attention Compressor投影，投影early标志仍为false |

Q_B的自动量化张量依赖仍保留；新增RoPE显式边不替代数据依赖。
weights的新增前置是一项明确的顺序策略，用来试验晚一点运行小投影是否有利于关键链；
不预先断言weights造成全引擎饱和，也不预先把收益归给AICPU开销下降。

## 现有证据与可检验假设

复用上一轮基线4ffccb7b的固定window_3，未挑选最快窗口，未新占卡采样。
128K/B16相对首个kernel：Q_A为91.340–108.080μs，weights为105.740–114.900μs，
weights reduce在125.460μs结束；Attention投影128.800–186.460μs，
系数最终数据就绪305.480μs，Score于321.620μs启动。
weights存在提前完成余量，可尝试移到Attention投影之后；这些重叠时刻不证明资源阻塞。
Score前置end→FIN为4.460μs、FIN→dispatch为8.280μs、dispatch→start为3.400μs，分别记录。
[两档既有窗口与依赖证据](baseline_schedule.json)。

预计变化是Q_B仍不预派发、weights不再与Q_A同一时段执行，而Score能否提前需真机判断。
同时观察weights是否延后过度、Q_B和两个Compressor是否变化，以及完整CSA/P95/max；
不以单个任务提前或dummy数变少作为采用依据。

## 编译和设备对照

两套入口_get_dep_graph通过；候选完整CPU编译/load通过。
编排生成码已确认dummy提交为零、RoPE/Q_A及weights真实前置存在，
没有修改PyPTO/Simpler/PTOAS/PTO-ISA。

task_20260929_063915_20346931582已通过task-submit自动分配单卡执行。
128K/B16、8K/B24，两侧CANN9.2/mode2/atomic0/det0，layer4正式权重与独立合成历史，EPLB关闭。
ring=[256,128,256,32]MiB/task_window4096。每侧5预热/20次无profiler真实编译图计时，
另采四窗level-4；长基线→候选、短候选→基线，按8:2判定完整CSA，P95/max单列。
复用八类跨版本状态零容差、图/eager和保护区检查，不扩七档/16卡。
收集器逐窗确认dummy4→0、六条真实显式边、两个准入前置不允许预派发及实际行为。

所有设备源码、依赖、runner在提交前复制并完成编译，排队后不编辑。
任务已完成exit=0，结果判定见下文。生产没有移入该组合。

[候选补丁](candidate.patch)、[冻结来源](source.txt)、[编译结果](compile_candidate.json)、
[生成编排](lowering.json)、[设备入口](run.sh)、[结果收集](collect.py)。

## 结果：不采用该组合

| 档位 | CSA基线→候选μs | 变化 | P95μs | maxμs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 1034.253→1042.744 | +0.821% | 1046.740→1058.540 | 1047.020→1065.180 |
| 8K/B24 | 985.651→965.088 | −2.086% | 994.980→984.940 | 1007.160→989.040 |

长短8:2为+0.240%，未满足保留判据。两档八类完整状态零容差、图/eager、Top-K结构、
metadata和保护区通过；16个DFX窗口经官方时钟域及原始/joined/block行数核对。
每个候选窗口都确认dummy为零、六条显式真实前置存在，Q_B/weights实际没有预派发。
因此这是策略未取得整体收益，并非修改未生效。

四窗口Worker时刻以首个receive归零：长档Q_A结束112.050→109.300μs，
但Q_B首次start161.015→169.985μs，weights启动分散2.830→25.295μs，
weights reduce结束134.215→250.010μs。
长档Score AIC/AIV核时253.307/259.062→258.100/263.810μs，
merge结束625.855→629.960μs，关键链没有取得预期提前。

单独按首个kernel为原点统计，Score首start长334.395→335.890μs、短369.040→376.080μs。
长档平均FIN→dispatch4.150→3.245μs，但前置数据就绪319.630→322.325μs，
这一局部派发间隙缩短没有转化为提前启动；不将重叠区间相加当软件开销。
两侧长档各有一窗Score实际提前派发，短档也存在全/部分提前派发，不能说Score始终等FIN。
核函数算术未改，系数和Sparse核时的下降包含执行环境与等待变化，不登记为新incore算法收益。

独立DFX Worker跨度长1014.280→1008.515μs、短939.415→937.185μs，
它们不替代无profiler计时，也不能解释未同时捕获的正式样本。
短档均值改善保留为观测，但该组合没有整体收益；不据此扩测或先合入短档分支。
没有证明具名资源阻塞或修复历史P95/EP16长尾。

**生产保持4ffccb7b。** 两轮去dummy试验均保留补丁与证据，不原样重复。
下一步补已保留系数优化的双query缺口，再按同源码完成阶段出口七档。
[结果](RESULTS.md)、[任务分项](TASKS.md)、[完整状态与依赖精简证据](summary.json)、
[采用判定](decision.json)。固定window_3前后泳道：

| 档位 | 基线 | 候选 |
| --- | --- | --- |
| 128K/B16 | [泳道JSON](h131072_b16/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [泳道JSON](h131072_b16/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
| 8K/B24 | [泳道JSON](h8192_b24/swimlane/baseline/dfx/window_3/merged_swimlane.json) | [泳道JSON](h8192_b24/swimlane/candidate/dfx/window_3/merged_swimlane.json) |
