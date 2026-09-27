# WO-B按整token量化：评估对后续MoE的影响

基底2a740c1f，原Native cache布局、性能版；[候选](candidate.patch)只修改输出投影算术。
独立工作树`.cache/csa-oproj-token-2a740c1f`；当前没有合入生产。

现有性能版沿pypto-lib：WO-A的FP32结果按8个组分别量化，每组自己的amax/scale；
WO-B每组INT32结果先反量化，再合并FP32。Native则先发布WO-A BF16，拼接8192列，
同一token只使用一个量化scale，WO-B整数结果合并后再反量化。
这不是cache布局问题，也不能断言哪种理论精度更高；但算术差异可能改变下游专家工作量。

候选从已有精度版迁入上述Native量化边界，保留性能版当前的NZ权重读取、M/N分块及按形状选tile：

- 所有WO-A组完成后按token块取全8192列BF16的amax。
- 统一scale量化各组，INT32部分和先合并，再按channel→token顺序反量化并转BF16。
- 增加跨组标度依赖，会牺牲一部分组间流水；能否换来整模型收益必须实测。

设备执行前将标度工作块从32 token细化为8 token：B16/S6对应12个工作块，B40对应30个，
量化任务仍保留32 token。每个token仍按原组顺序取同一amax，算术没有变化。
原先3个B16标度任务的版本未运行；排队任务task_20260928_012807_25358029066已取消，不计入测试结果。

先做单卡正式layer4/B16/H8192完整CSA计时，以及B3固定规约A→B→A图/尾块/保护区，
不会因更像Native就直接安排七档或声称通过。
[命令](run_layer.sh)；task_20260928_014832_382436620956退出0。

## 单卡结果：没有局部收益，保留一次整网干预的验证范围

| B16/H8192 | 原性能版 | 整token量化候选 |
| --- | ---: | ---: |
| PTO完整CSA均值 μs | 786.753 | 811.353 |
| PTO P95 μs | 806.660 | 832.820 |
| Native控制均值 μs | 925.906 | 943.864 |
| 整层输出对Native RMSE | 0.0033201673 | 0.0033237122 |

基线复用`csa_softmax_cumulative_20260928/layer_b16/baseline/report.json`，与本轮相隔约40分钟，
PTO均值增加3.13%，Native控制也增加1.94%；不把全部24.60μs差额当作候选净成本。
整层误差基本没变，没有局部精度或性能收益证据。功能保护区、metadata、有限值和Top-K结构通过；
B3的固定形状A→B→A图重放通过，覆盖18 token尾块，未据此声称不同batch padding图切换通过。
[精简数值和原始样本](single_layer.json)，原始报告在`timing_b16/report.json`、`graph_b3/report.json`。

仍安排一次两档模型干预：本候选把进入MoE前的量化方式改为Native合同，可能改变专家路由/工作量；
先前已观察到真实GMM的增量，但未证明本候选能消除它。正式forward必须补回CSA额外成本，
否则不保留、不扩大七档；不能凭“更像Native”或局部RMSE宣称成功。

## 后续受控模型对照

task_20260928_015726_18185222732已完成、退出0：[两档模型命令](run_model.sh)及[收集入口](collect_model.py)。
进程成功不等于验收通过，最终结论见下方。
使用独立工作树`.cache/csa-oproj-token-ordered-2a740c1f`，
组合既有`ordered/candidate.patch`与本目录的输出投影补丁。
PTO两次候选之间固定分离/新页排序、atomic1、mode2、det0/HCCL=false及EPLB关闭。
Native复用`csa_source_split_ab_20260927/ordered/model/h*/b*/native`；不把复用结果写成新跑的控制。

统一入场和10步CPU position相同能排除既有的请求入场错位，不能单独证明完整设备输入相同：
异步spec decode在设备上更新采样及草稿token，`input_ids.cpu`不一定是实际输入的镜像。
因此不能为补齐记录而把该CPU缓冲当作设备token，也不往正式计时流里添加复制或同步。
正式forward仍衡量实际请求的部署表现；严格的路由因果分析需使用已有独立路由诊断的实际设备输入。
输出token和DSpark统计仍分别验收，长档旧候选DSpark未通过的事实不变。

只读既有原始记录进一步量化该差异（覆盖本档预热、正式轮及profile轮，不只10步forward）：
128K/B16的Native接受76800/76800个草稿token；ordered/atomic1为76782/76800，
少18个（第4/5个草稿位置分别少4/14）；atomic0为76788/76800，仅最后草稿位置少12个。
8K/B40两侧均为192000/192000。少量接受差异依然未满足精确统计合同，
但不能将其比例当作forward约3%差距的耗时归因，或把全轮次计数写成正式10步内发生的拒绝。

## 最终两档EP16结果：不合入、不扩大测试

| 档位 | Native forward ms | 整token量化PTO ms | 变化 | PTO P95 ms | token差异 | DSpark差异rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 72.008891 | 75.402653 | +4.71% | 76.654678 | 0/65536 | 15/16 |
| 8K/B40 | 104.357671 | 104.846230 | +0.47% | 106.401840 | 0/163840 | 0/16 |

按既定10步纯forward，P95/P50分别1.017/1.012，没有本轮异常长尾证据。
与相同ordered布局、atomic1的前轮PTO74.261609/104.120537 ms相比，均未改善。
Native控制是复用结果，有时间漂移边界；此处用于否定当前候选已经胜出，不把小差额当作精确净成本。
全部16rank的10步CPU位置相同，设备草稿token证据边界仍按上文保留。
[正式结果](model/RESULTS.md)、[逐rank原始样本](model/forward.json)。

独立rank0三步profile（不拼接为正式forward的分摊）：

| 档位 | Native/PTO CSA body合计 ms | Native/PTO FFN合计 ms | 原ordered/PTO两类专家GMM合计 ms |
| --- | ---: | ---: | ---: |
| 128K/B16 | 26.888/25.386 | 33.729/36.461 | 10.065/10.091 |
| 8K/B40 | 30.069/28.507 | 56.127/58.143 | 13.455/13.571 |

两类GMM累积task duration均没有减少；CSA body相对原ordered增加0.429/0.755 ms。
FFN task可能重叠，且首层dispatch包含跨rank到达等待，不能把FFN总增量全算到数值差异上。
[profile边界](model/model_gap_rank0.json)、[长档FFN](model/h131072/ffn_breakdown_rank0.json)、
[短档FFN](model/h8192/ffn_breakdown_rank0.json)。原trace保存在各PTO目录的profile输出下。

该量化候选没有补回自己的成本，也没有解决长档DSpark差异，停止扩测并保留生产2a740c1f。
依赖同一整token scale的WO-B两份整数中间结果候选仅保留CPU编译证据，本阶段不继续占卡。
后续针对更上游的QKV BF16数值边界做独立单卡筛查，不叠加本候选。
