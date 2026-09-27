# QKV的Native BF16数值边界：独立筛查

基于生产2a740c1f，独立工作树`.cache/csa-qkv-bf16-2a740c1f`，仅修改性能版QKV。
没有叠加整token WO-B量化或source-split cache，Native和精度版不变。

Native路径在`attention/dsa_v1.py`的decode prolog：QA投影后进入fused RMS/quant，
QB反量化后进入Q RMS，再进入RoPE；KV投影后进入RMS，再进入RoPE。
这些独立算子的输入/输出使用BF16。性能版沿上游融合策略，中间直接使用FP32值，省略了部分舍入。
这种差别不是功能错误，也不意味着更多FP32一定更差；当前实验只检查是否会扩大实际模型的专家工作量。

候选在片内补齐以下BF16→FP32 round-trip：

- QA输出：平方和/amax扫描和量化扫描都转换，保证使用同一数值。
- QB反量化输出，以及Q RMS之后进入RoPE的分量。
- KV投影输出的所有扫描，以及KV RMS之后进入RoPE的分量。

完整行和尾块均覆盖，不增加GM张量、任务或跨任务依赖。仍保留性能版NZ分块、split-K/atomic、
Q动态head分块、平方和/rsqrt/amax及量化规则，不能宣称已完全复刻Native算术。
[最小补丁](candidate.patch)。

CPU完整QKV lowering/PTOAS/CCE/链接通过，[入口](compile.py)、[日志](compile.log)。
这不是设备正确性或性能证据。编译命令：

```bash
source ../env-dsv4-0251rc1.sh
python tests/pypto_test/results/csa_qkv_bf16_20260928/compile.py \
  --source ../.cache/csa-qkv-bf16-2a740c1f \
  --output tests/pypto_test/results/csa_qkv_bf16_20260928/compiled
```

单卡task_20260928_023535_1798714557：[命令](run_layer.sh)。
同轮原版/候选B16/H8192、正式layer4权重、mode2/atomic1/det0，5次预热20次图计时；
另测B3/atomic0/det1固定形状A→B→A和尾块。主要观察完整CSA成本和完整HC输出误差，
不能凭某个中间量更接近Native就认定整网受益。当前未合入；完成下述单卡筛查后才提交16卡代表档。

## 单卡结果与进入EP16的依据

task_20260928_023535_1798714557退出0，原版/候选正式layer4、B16/H8192、mode2/atomic1/det0：

| 指标 | 原性能版 | QKV边界候选 |
| --- | ---: | ---: |
| PTO完整CSA均值 μs | 781.998 | 789.491 |
| PTO P95 μs | 798.820 | 807.520 |
| Native控制均值 μs | 919.414 | 934.725 |
| 完整HC输出对Native RMSE | 0.0033201341 | 0.0029213311 |
| 输出零容差差异元素 /1572864 | 652014 | 605887 |
| Top-K集合替换索引总数 /49152 | 366 | 272 |

完整误差降低12.01%，集合替换数降低25.68%。PTO均值增加0.96%、Native控制同时增加1.67%，
不能把7.49μs当作候选的精确净成本；没有宣称CSA已加速，也没有将零容差FAIL改为PASS。
各次metadata/保护区、有限值/Top-K结构通过。B3的atomic0/det1固定形状A→B→A图通过，
覆盖18 token尾块；不代表跨batch padding图验收。
[精简对照与原始样本](single_layer.json)，原始报告位于baseline/candidate/graph_b3的report.json。

本轮第一次在完整层输出上观察到明确的误差下降，且局部成本不大，值得检验是否减少下游专家工作。
提交task_20260928_024844_45752012651：128K/B16与8K/B40，正式16卡EP16、mode2/atomic1/det0、
HCCL=false、EPLB关闭，严格整批入场后各取warmup后10步pure decode forward。
[模型命令](run_model.sh)、[只读收集器](collect_model.py)。

**本轮Native控制重新采集**，不复用两小时前的控制来判定小幅胜负。Native生产流程未修改。
独立模型工作树`.cache/csa-qkv-bf16-model-2a740c1f`＝2a740c1f＋测试入场修复8dd737f4＋本目录候选。
保留原Native cache布局；当前没有ordered/source-split布局补丁，不能与前轮分离布局的成绩直接作单因素相减。
只有真实forward、P95和token/DSpark达到要求才考虑保留；局部误差下降本身不是整模型优势。

## EP16结果：本候选不合入

task_20260928_024844_45752012651退出0；两档都使用同轮重新采集的Native控制。
正式warmup后10步，无profiler，各16rank等权均值：

| 档位 | Native/PTO forward ms | PTO变化 | Native/PTO P95 ms | Native/PTO最慢rank均值 ms |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 72.071/76.510 | +6.16% | 73.023/77.301 | 72.421/76.988 |
| 8K/B40 | 104.243/103.593 | −0.62% | 107.595/105.494 | 104.332/103.679 |

229376输出token零差异，32组rank的DSpark统计全部一致；两侧10步CPU位置数组一致，
不将其外推为所有设备草稿输入相同。PTO P95/P50为1.009/1.019，没有观察到异常尾部。
长档明显落后、短档小收益不足以宣布阶段完成；不合入、不扩大七档。
[完整结果及口径](model/RESULTS.md)、[逐rank原始计时样本和配置](model/forward.json)。

独立rank0三步Level0 profile已离线解析，未新增设备测试：

| 指标 ms/step | 128K Native/PTO | 8K Native/PTO |
| --- | ---: | ---: |
| 21层CSA body区间之和 | 26.845/27.547 | 29.819/28.145 |
| 43层FFN区间之和 | 34.701/32.358 | 57.089/56.439 |
| 两类专家GMM任务duration之和 | 6.211/7.497 | 10.368/11.068 |
| 第一层FFN区间 | 2.567/1.056 | 1.920/1.301 |
| 主图区间 | 76.318/74.401 | 105.917/103.025 |

长档模型内CSA均值1278.310/1313.516μs（PTO慢2.75%），短档1419.946/1342.344μs（快5.47%）。
专家GMM仍增加1.286/0.700ms，但FFN区间反而缩短；通信到达等待与可重叠任务不能混算。
本轮长档profile主图与正式10步forward的胜负反转，首层EP等待明显不同；
因此这份独立profile不能闭合正式计时的4.439ms差距，不能拿profile结果覆盖正式结论。
相较此前ordered的GMM增量有所缩小，但cache布局和轮次不同，尚不能作为QKV的单因素因果结论。

[profile分解](model/model_gap_rank0.json)、逐层任务明细
[128K](model/h131072/ffn_breakdown_rank0.json)/[8K](model/h8192/ffn_breakdown_rank0.json)。
原始PyTorch JSON在各`model/h*/b*/{native,pto}/trace/rank0/*/ASCEND_PROFILER_OUTPUT/trace_view.json`。
这些路径属于本次独立profile轮；不冒充无profiler主计时或单卡DFX泳道。
