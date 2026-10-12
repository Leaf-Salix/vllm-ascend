# 5μs固定链原始结果汇总

单位μs；17级×48 AIV，work5，Scalar256，8次调用/图，30个样本，不删最大值。
plain无Set/Wait，early有Set/Wait；每轮前→候选→后同卡。所有逐级输出、换输入及保护区检查通过。

| 轮次/版本 | 模式 | min | mean | max | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| retry1 / compact | plain | 286.54 | 293.79 | 297.21 | 296.81 |
| retry1 / compact | early | 291.62 | 295.39 | 298.33 | 298.11 |
| retry1 / normal_after | plain | 289.46 | 292.79 | 295.96 | 295.21 |
| retry1 / normal_after | early | 295.53 | 300.86 | 304.11 | 303.69 |
| retry1 / normal_before | plain | 287.54 | 292.46 | 296.80 | 295.64 |
| retry1 / normal_before | early | 297.09 | 301.20 | 307.86 | 305.17 |
