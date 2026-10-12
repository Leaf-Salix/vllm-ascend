# 5μs固定链原始结果汇总

单位μs；17级×48 AIV，work5，Scalar256，8次调用/图，30个样本，不删最大值。
plain无Set/Wait，early有Set/Wait；每轮前→候选→后同卡。所有逐级输出、换输入及保护区检查通过。

| 轮次/版本 | 模式 | min | mean | max | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| retry1 / current_sdk | plain | 289.24 | 293.75 | 303.23 | 300.29 |
| retry1 / current_sdk | early | 291.36 | 295.52 | 299.94 | 298.23 |
| retry1 / old_after | plain | 283.76 | 288.34 | 294.93 | 292.84 |
| retry1 / old_before | plain | 286.42 | 289.53 | 294.07 | 292.52 |
