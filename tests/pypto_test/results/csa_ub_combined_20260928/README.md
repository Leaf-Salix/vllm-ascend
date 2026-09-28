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

真实EP16任务`task_20260928_162727_150252329060`已提交，尚无模型结果；
[设备入口](run_model.sh)、[forward/token/DSpark收集](collect_model.py)、
[主机/GC/入场诊断](analyze.py)。两侧同一冻结源码/权重，TP1、DP=EP16、出5验6、mode2/atomic0/det0、EPLB关；
96 token预热、跳过8步后连续10步纯decode forward，独立3步PyTorch profile，不包含加载/编译/草稿/采样。
正式窗口不新增Event或同步，保留所有rank及逐步最慢rank；结果按7:3记录，P95与token/DSpark单列。
