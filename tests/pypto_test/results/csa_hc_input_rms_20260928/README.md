# HC输入加宽与RMS统计合并候选

基底为e58ddc94算子（9edb8dfe只增加记录），候选冻结在`.cache/csa-hc-input-rms-9edb8dfe`。
仅性能版把`hc_widen`与`hc_pre_rms`合并为`hc_widen_rms`：BF16读取后转FP32，
一份值写入原FP32中间缓冲，另一份按原512列、32段顺序平方归约，最后用原高精度rsqrt。
省去独立RMS任务对整份FP32缓冲的GM读取，B16/S6逻辑读取量约6 MiB；这不是PMU实测流量或时间收益。

Cube仍读取FP32、保持纯AIC；门控、20次Sinkhorn、混合及BF16/RMSNorm边界共用原实现。
共用模块只抽出“已有RMS统计后的门控”与“已有门控后的混合/Norm”，旧精度入口仍自行运行原RMS任务。
尾行在cast后显式恢复valid_shape并填零，避免历史T60无效行参与归约问题。

最新AscendC直接参考：ops-transformer b5b33e14的
`mhc/mhc_pre_sinkhorn/op_kernel/mhc_pre_sinkhorn_m_split_core.h::MhcPreSinkhornStage1::Process`。
其AIV将BF16输入加宽，一份写给Cube，原UB值直接算平方和；本候选沿用这一读取复用方式，
保留PTO原512列分段顺序和高精度rsqrt，没有移植该源码的分块/同步及Sqrt+Div算术。
同仓另一条`mhc_pre_sinkhorn_cube_compute.h::ComputeDecode/MmadA2/MmadAB`则在L1/L0A复用输入，
用Cube同时计算平方和与投影；两条实现必须区分，当前PTO没有采用Cube A2路线。
本仓Native的`hc_pre_m_k_split_core.h`及`hc_pre_cube_compute.h`用于核对BF16输入转换和Cube消费边界。
本候选仍用现有Vector归约，不把它表述为原样复制Native Cube算法。
pypto-lib 73078d0的HC入口是FP32，无本接入的BF16桥接；因此仍保留必要FP32缓冲，消除的是本地重复读取。

完整CPU lowering/PTOAS/CCE/链接通过，生成代码有AIV `hc_widen_rms`和纯AIC `hc_pre_linear`，
独立`hc_pre_rms`消失。融合可能延后Cube启动，需同时看核内累计工作、HC关键链及完整CSA。

原任务`task_20260928_103131_11716825293`已在pending时取消，未执行：用户要求先核实短档相邻CSA波动。
短档Score/Sparse开关及四路Top-K对照已结束，现以任务`task_20260928_131803_262469420227`恢复。
该任务已完成、退出0，device1；保留两者独立的冻结源码以便区分收益。
长短B16各20次无profiler图计时及2个DFX窗口，
layer4真实权重、合成历史、mode2/atomic0/det1。两侧计时与两窗口DFX均在本轮同卡采集，
不跳过没有结果链接的基线DFX。两侧均基于e58ddc94，不叠加四路Top-K；
归并候选的结果不能当作HC的单变量基线。

两档八类PTO及Native固定配置的跨版本状态精确一致，保护区、自重放及候选A→B→A通过。
输入加宽＋RMS累计核内工作，128K/B16从212.46降至155.37μs（−26.87%），
8K/B16从212.33降至148.96μs（−29.85%）；由两组各12个worker变为一组12个。
这是相同工作范围的累计核时间，不能当作约60μs的整层缩短。

| 档位 | 完整CSA均值，基线→候选 μs | P95 μs | max μs | Native控制均值 μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 1116.186→1111.012（−0.46%） | 1130.40→1121.16 | 1133.96→1124.20 | 1315.276→1302.437 |
| 8K/B16 | 782.734→782.060（−0.09%） | 795.70→799.98 | 798.60→804.16 | 930.811→950.485 |

HC首Worker→norm结束反而84.66→88.79、83.99→86.53μs，Cube首次启动平均推迟3.97/4.38μs。
融合让Cube等待输入变长，是下一步需要解决的取舍；两档完整层都不足以证明稳定加速，短档P95略升。
必要尾块任务task_20260928_132736_2807752475完成、退出0：B10/S6/T60的性能版和精度版均通过，
两侧八类状态精确一致，Native控制、保护区、自重放和A→B→A通过；本轮没有重复性能矩阵。
按核内保留规则接入性能版输入/RMS融合，共用函数抽取保留精度版原RMS/门控/混合算术。
同时清理HC注释中“本项目NZ关闭”及仅凭507057就断言BYPASS必崩的过时表述，缓存策略没有改变。
后续用与四路Top-K组合后的最终源码补必要EP16/token/DSpark；当前尚未完成新增策略的整网验收。

[完整结果](RESULTS.md)、[状态/样本/核内窗口](report.json)、[汇总脚本](summarize.py)、
[尾块命令](run_tail.sh)、[性能版尾块](tail_b10/performance/comparison.json)、
[精度版尾块](tail_b10/precision/comparison.json)。

[执行脚本](run_layer.sh)、[源码差异](candidate.patch)、[编译日志](compile.log)。

[Simpler #2389评估](../csa_mix_preload_20260928/README.md)已结束，当前不采用；
按用户要求不追加该PR的修复或设备对照。本轮冻结运行时保持不变。
