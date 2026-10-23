# TND完整精度版：性能优化前验收快照（2026-10-23）

## 固定的实现

分支为`dev/pypto-dsv4-csa-nalinaly-tnd-20261008-codex`。
优化前算子源码提交为`6eabea0ec89313d6ab020ef7f94eb2b85a689fc8`；
本次只固定近期测试证据，不更改默认实现、数值处理或已验收环境。
完整替换范围为HC-pre至HC-post，单个PTO顶层入口、纯DSL、真实TND请求边界。

nalinaly对照固定为其公开`67134f28bdbd9881b14c4d91f0e27092d88f6501`
归档的TND-SPMD `combined_v1`，不是其全部实现的结论。
测试用既有共同工具链，删除不受支持的`backoff_ticks=150`参数并使用SDK默认等待，
不修改生成C++。因此不是其截图原SDK的绝对性能复现。

## 三方单层共同测试

真实第4层权重、合成hidden/history，相同输入、Native metadata和cache初态，
各实现独立可写cache。Native为确定性level1、HCCL确定性、SK0、13份实际重选静态包。
全部覆盖HC-pre至HC-post；12个交错窗口，每窗100次raw FULL graph replay，保留全部样本。

| 配置 | Native完整图中位数μs | nalinaly完整图μs | 我方完整图μs | 我方对Native | nalinaly输出对Native relative L2 |
| --- | ---: | ---: | ---: | --- | ---: |
| 8K/B32/S6 | 1236.05 | 1239.96 | 1857.68 | A/B八边界逐位一致 | 0.333% |
| 128K/B16/S6 | 1247.97 | 1222.68 | 1538.85 | A/B八边界逐位一致 | 0.417% |

另行DFX采样的完整incore L1中位数分别为：

| 配置 | 我方L1 μs | nalinaly L1 μs |
| --- | ---: | ---: |
| 8K/B32/S6 | 1791.17 | 1118.25 |
| 128K/B16/S6 | 1557.18 | 1108.40 |

L1含72核间等待和空隙，DFX与event来自不同任务，不作中位数相减或端到端吞吐推断。
这些数字是本轮共同测试，不是跨任务稳定收益保证。

## 真实整模型短输出

模型为DeepSeek-V4-Flash-0731-w8a8；8K固定offline P/D bank前16条prompt，
每rank B16，TP1、DP=EP16、DSpark出5验6、greedy、EPLB关闭，暖64加计时64 token。
16条prompt在16个rank重复；不声称256条不同prompt，也不替代历史192-token验收。

| 实现 | token结论 | 其他检查 |
| --- | --- | --- |
| Native | 同任务自重复16384对一致 | 接受事件和输入轨迹一致 |
| 我方完整精度TND | 对Native的32768对全部一致 | 8032组接受事件、完整生成输入轨迹一致，21层CSA完整替换，20层HCA保持Native，零抢占 |
| nalinaly兼容TND | 同首步NONE/S1输入已观察到首token反例 | 后续设备错误使完整64未完成；不授予完整输出或公平FULL性能结论 |

首token反例：`h8192_q06`为Native 9854、nalinaly 3648；`q15`为Native 46543、nalinaly 9854，
均在全部16rank出现。证据来自实际device输入实参的下一步首token，未推断失败forward的输出。
前缀重建规则已用完成的Native/我方共65280个可观测gold位置和各7904组接受事件推进核验。
这不是完成64token的nalinaly API结果。

我方计时轮FULL/T96每步中位数为66.22ms，Native为59.06ms；
首个NONE/T16步分别为5468.46ms与258.78ms，需单列普通派发开销，
不能将整轮active时间比当作CSA图回放速度比。该测试带trace，不能当服务吞吐。

## 分阶段同输入误差定位

8K/B32/T192，真实L4权重。新增Native eager观测先通过与Native FULL八边界逐位门禁。
各条件对拍固定模块输入，排除上游传播；所有张量只比较有效T。

| 模块 | 固定输入的结果 | 可得结论 |
| --- | --- | --- |
| HC-pre | mixed 1953处不同；post 576；comb 3060 | 最早可观测首差在HC-pre出口，未细分内部投影/RMS份额 |
| InputNorm | 同mixed仍9处不同 | 独立新增差异 |
| QA/WKV | 同normed、BF16输出分别43/26处不同 | 投影路径独立不同，不能只归因缺少BF16 publication |
| QR | QA先BF16后的Native参考，INT8 10382处不同，均差1 | 包含输入BF16、RMS及量化的组合差异 |
| Indexer系数 | 同normed，10904处不同 | 投影/缩放边界不同 |
| Indexer Hadamard及量化 | 同RoPE query，INT8 149786处不同 | 该组合独立新增差异 |
| 最终TopK索引 | 同query/key/FP16系数及scale，98304索引逐位一致 | 仅本例索引，不等于完整score张量一致 |
| Attention及逆RoPE | 同Q/cache/TopK，heads relative L2为0.265967% | 该组合独立新增差异，未细分softmax/PV/inverse份额 |
| O-proj | 同heads，attended 621696处不同 | 输出投影组合独立新增差异 |
| HC-post | 重建nalinaly attended后，Native HC-post逐位复现3145728个输出元素 | 本例可排除HC-post |

各模块误差不可相减或相加分配总误差贡献；本轮没有穷尽所有误差来源，
也未把整模型具体首token差异唯一归因于某个单层模块。

## 后续优化边界

用户允许直接复用nalinaly代码；保持来源、许可和具体数学差异记录。
优先借鉴实际producer等待、数据复用/预取、片上驻留、heads直送和HC-post融合。
保留Native BF16发布、归约树、完整8192列量化尺度、INT32累加及最终反量化顺序。

每个候选先做不带hook的单因素完整链路计时；有稳定收益再做A/B逐位、真实TND短query/混长，
最后8K/B40、128K/B24真实模型token回归。任务逐项提交，每项不超过15分钟；首差停止扩大矩阵。
未完成这些检查的候选不替换本快照对应的默认精度基准。

## 复现资产

机器指标、三方重算结果及分阶段实际观测脚本见
[证据目录](evidence_tnd_precision/native_alignment_checkpoint_20261023/README.md)。
原始张量、checkpoint和编译二进制不提交Git；它们的SHA、固定输入、工具链、脚本身份保留。
