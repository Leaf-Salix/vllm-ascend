# S6原cache版本：CSA收益与FFN增量

算子9a01a276，测试入场修复8dd737f4，任务task_20260928_034149_296992718490。
正式16rank各10步无profiler forward：128K/B16快0.81%，8K/B40快0.54%，token/DSpark全部通过。
均值、P95、最大值、慢卡及所有样本见[正式成绩](RESULTS.md)。幅度仍小，不称为稳定优势。

下表是同一作业另一次请求轮的rank0三步Level0 profile均值，单位ms。
两轮请求重新分配cache，EP到达时刻也不同；**不把此表分解套到正式10步差额**。
处理脚本复用[既有分析器](../../csa_source_split_ab_20260927/ordered/analyze_profile.py)，
原始分解见[model_gap_rank0.json](model_gap_rank0.json)。

| 区间 | 128K/B16 Native | PTO | PTO−Native | 8K/B40 Native | PTO | PTO−Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 主图 | 74.695 | 71.848 | −2.848 | 104.943 | 104.129 | −0.814 |
| 21层完整CSA | 27.200 | 23.605 | −3.595 | 29.621 | 27.868 | −1.754 |
| 其他attention半层 | 14.758 | 14.027 | −0.731 | 18.845 | 18.127 | −0.718 |
| 43层FFN | 32.606 | 33.927 | +1.321 | 56.333 | 57.823 | +1.490 |
| 半层区间之外 | 0.131 | 0.289 | +0.158 | 0.144 | 0.311 | +0.168 |

这里的完整CSA含HC_pre、norm和HC_post，长档平均每层1295.239→1124.029μs（−13.22%），
短档1410.541→1327.030μs（−5.92%）。S6长档的核内收益确实传入模型CSA，
并非全被PTO调度抵消；随后FFN增量吃掉部分收益。

## FFN：专家GMM增加，通信还受EP到达影响

| 任务或区间 | 128K/B16 Native→PTO ms | 8K/B40 Native→PTO ms |
| --- | ---: | ---: |
| 两类专家GMM任务duration之和 | 6.059→7.763 | 10.485→11.474 |
| Dispatch任务duration之和 | 5.583→5.758 | 9.678→9.699 |
| Combine任务duration之和 | 11.157→10.884 | 22.824→23.317 |
| 首层FFN区间 | 2.065→1.952 | 1.416→1.972 |
| 排除首层后的42层FFN区间之和 | 30.541→31.975 | 54.917→55.851 |

任务duration可能重叠，不相加冒充FFN critical span。GMM增量分别1.704/0.989ms，
且排除首层后FFN仍增加，不能只归因于入场第一层通信等待。
以前对实际相同请求的[路由诊断](../../csa_model_forward_f76b3ad4_20260927/routing_b40/README.md)
确认PTO改变后续专家选择、增加活跃专家；本轮未重复路由采集，不能把历史计数冒充S6计数。
本轮profile仍支持关注这项整模型成本，但不足以证明哪一个舍入或atomic导致它。
Native控制保持原流程，没有为了制造胜负而修改其确定性、EPLB或cache策略。

## 后续取舍

1. 保留已有明确核内收益，长短档的query分组候选经单卡通过后，以同一套源码再验真实EP16。
2. 继续分开报告模型CSA与FFN，不用单卡CSA百分比推算整网增益，也不删除偏慢样本。
3. QKV BF16、累计softmax、WO-B量化及source-split先前候选均未带来合格的模型增益，不盲目叠加。
4. 当前正式P95并无异常上升，不再把全部问题归因于长尾；七档和明确稳定优势仍未完成。

两侧PyTorch JSON位于本目录`h131072/b16/{native,pto}/trace/rank0/*/ASCEND_PROFILER_OUTPUT/trace_view.json`
和`h8192/b40/{native,pto}/trace/rank0/*/ASCEND_PROFILER_OUTPUT/trace_view.json`。
只离线导出已有profile，没有新增设备采集、hash或计时区间同步。
