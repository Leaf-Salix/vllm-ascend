# Top-K UB与QR输入驻留组合验收

冻结生产源码2d2f9ca0，工作树`.cache/csa-ub-combined-2d2f9ca0`。
相对已完成真实EP16的d1f170ff，新增多leaf四路Top-K累计根留在UB、QR输入/gamma驻留及尾行直接UB发布。
这两项已分别完成单卡状态、图重放及必要归并/尾块检查；本轮核对组合，不能将独立收益相加。
不包含其他会话未提交的WO_A或测试Runner修改；Native流程、原生cache与工具链保持既定环境。
累计softmax/PV单侧缩放和PV L0B候选均不在本组合。

先CPU完整lowering/PTOAS/CCE/AICPU链接，再单卡128K/B16及8K/B16。
每侧5预热+20次无profiler图计时、两个独立DFX，交换长短执行顺序；两侧各保留Native控制。
采用layer4既定真实权重、反向物理页与随物理行变化的Indexer scale，mode2/atomic0/det0。
八类状态零容差、metadata/保护区、自重放、图计时状态及候选A→B→A必须通过。
本组合不改算术，不能使用Sparse算术候选的MEASURED_OUTPUT选项。

长短耗时变化率按7:3计算，核内、完整CSA/P95分别记录；以两窗口作组合观测，
不替代单变量四窗口因果证据。已完成的独立边界测试不无条件重复。
单卡通过后安排冻结同一源码的真实EP16 decode forward、token/DSpark与PyTorch profile；
阶段结束再更新同一最终源码完整七档及对应泳道。

[CPU入口](compile.py)、[单卡命令](run_layer.sh)。CPU完整编译/链接通过。
单卡任务`task_20260928_160301_64273522658`完成、退出0，两档八类状态/保护区/图重放通过。

队列开始时独立Key预取候选的CPU lowering仍在执行，
两侧128K计时报告在该进程挂起前完成；[时间与限制记录](cpu_compile_overlap.json)保留。
这两份原始计时仍记录，但不据小幅差额判断组合净收益；状态/图重放证据继续收集。
当时已挂起同一CPU进程，单卡结束后恢复；模型正式窗口前完成全部CPU编译。
不为此额外重复已有单层状态，后续既定EP16的forward/P95决定模型效果。

## 单卡组合结果

两档八类PTO输出/状态逐元素零容差，metadata/保护区、自重放、图计时状态和A→B→A通过。
这验证2d2f9ca0的组合实现，不能用旧QR首版结果替代；已有T60和Top-K边界不重复。

| 档位 | d1f170ff / 2d2f9ca0 完整CSA μs | 变化 | P95 μs | max μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 1125.885 / 1107.323 | −1.65% | 1139.62 / 1121.50 | 1141.94 / 1138.80 |
| 8K/B16 | 799.225 / 784.552 | −1.84% | 823.46 / 794.02 | 826.94 / 802.84 |

原始7:3综合变化−1.705%，**长档存在上述CPU编译重叠，不作为净收益验收**。
Native控制长档1310.652→1337.379μs，短档956.030→928.436μs，全部样本保留，不做归一化。
两窗口长档Top-K核内11.882→8.407μs，QR 6.550→6.543μs；
短档Top-K 8.553→9.089μs，QR 6.741→6.397μs。未改的Sparse/merge_norm也有波动，
不把组合观测归给某项单变量，不将两个窗口当成稳定因果结论。
八个DFX窗口的Sparse均24个AIC各一份，不能据此宣布旧尾部已修复。

[128K原始样本/窗口/状态](h131072_b16/evidence.json)、[8K证据](h8192_b16/evidence.json)、
[CPU收集命令](summarize.sh)。完整summary保留所有任务，精简证据保留相关核内任务与调度计数。

真实EP16任务`task_20260928_162727_150252329060`完成、退出0，两档结果如下；
[设备入口](run_model.sh)、[forward/token/DSpark收集](collect_model.py)、
[主机/GC/入场诊断](analyze.py)。两侧同一冻结源码/权重，TP1、DP=EP16、出5验6、mode2/atomic0/det0、EPLB关；
96 token预热、跳过8步后连续10步纯decode forward，独立3步PyTorch profile，不包含加载/编译/草稿/采样。
正式窗口不新增Event或同步，保留所有rank及逐步最慢rank；结果按7:3记录，P95与token/DSpark单列。

设备任务成功结束后执行[离线profile导出](export_profiles.sh)；入口先核对该任务已完成，
使用2d2f9ca0冻结Runner解析，避免混入其他会话的Runner修改。
[区间与相邻层报告](profile_report.py)复用已有分析，分别输出完整CSA的7:3变化率、
各档P95、原第12/14层明细和四份真实EP16 PyTorch JSON下载入口。
正式forward单独输出7:3变化率；两种指标都以同轮Native为基线，不作旧PTO到新PTO的因果归因。

## 真实EP16结果

以下均对应冻结2d2f9ca0，不含正在单卡验证的Key预取，也不含其他会话未提交的WO_A修改。
正式forward用每rank预热8步后的连续10步；CSA用独立rank0三步profile的63个完整区间，含首次metadata。
两种采样分别报告，不能把profile分项直接分摊到正式forward。

| 档位 | Native/PTO完整CSA μs | CSA变化 | Native/PTO forward ms | forward变化 | Native/PTO forward P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 1284.26/1103.12 | −14.10% | 72.784/69.626 | −4.34% | 73.559/72.454 |
| 8K/B16 | 971.59/794.54 | −18.22% | 65.553/61.147 | −6.72% | 67.256/62.969 |

以同轮Native为基线，128K/8K变化率7:3加权：完整CSA **−15.340%**，正式forward **−5.054%**。
两档P95、max及逐步最慢rank均值均较低，按正式位置对齐后各10/10步最慢rank更快。
131072个输出token零差异、32组rank的DSpark统计一致；位置、运行配置及预热计时事件检查通过。
尚未完成2d2f组合的七档模型验收，旧版七档不拼入此表。

**长尾未关闭。** 长档PTO step14的rank13相对通常入场晚3.408ms，自身forward68.971ms，
其余15个rank平均增加2.795ms。该rank的attention_metadata阶段7.740ms、同rank十步中位4.137ms，
当步线程CPU7.569ms；正式窗口无GC。主机forward入场相对中位晚3.498ms，支持metadata准备延迟被EP等待放大的解释。
现有标记只覆盖`_build_attention_metadata`，尚未定位具体子调用，不能仅凭此归为OS抢占或CSA Score。
该样本全部保留；两档P95低于Native也不等于已消除罕见波动。

短档相邻层波动也单列：本轮最大向上跳变为step2的18→20层，771.14→803.64μs；
原第12/14层在第三步为790.40→805.10μs，Worker778.74→796.56μs。
不同层输入/权重不同，且未采对应incore DFX，不能从这次差额变小宣称调度修复。

[正式forward及全部rank样本](model/RESULTS.md)、[模型CSA/FFN分项](model/MODEL_GAP.md)、
[相对入场](ARRIVAL.md)、[主机/GC](HOST.md)、[metadata区间明细](PHASES.md)、
[相邻CSA](model/ADJACENT_CSA.md)、[四份PyTorch JSON集中下载](download/README.md)。
提交[尾部精简证据](metadata_tail.json)：异常rank的全部十步主机/metadata标记及所有rank的正式窗口GC计数；
全部原始host/phases和较大trace保留在本地结果目录，下载清单记录路径，不将大型重复trace写入Git。
离线导出在Key单卡任务仍pending时完成；复用CPU进程组保护以防新设备计时开始，
本轮未触发挂起。[导出命令结果](profile_export.log)、[UTC监控记录](profile_export_guard.jsonl)。
