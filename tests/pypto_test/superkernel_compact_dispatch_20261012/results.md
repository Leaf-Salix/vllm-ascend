# 5μs固定链原始结果汇总

单位μs；17级×48 AIV，work5，Scalar256，8次调用/图，30个样本，不删最大值。
plain无Set/Wait，early有Set/Wait；每轮前→候选→后同卡。所有逐级输出、换输入及保护区检查通过。

| 轮次/版本 | 模式 | min | mean | max | P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| retry1 / compact | plain | 285.35 | 291.98 | 296.08 | 295.63 |
| retry1 / compact | early | 291.69 | 295.07 | 299.40 | 298.22 |
| retry1 / normal_after | plain | 290.25 | 294.41 | 299.55 | 298.93 |
| retry1 / normal_after | early | 298.67 | 301.88 | 306.10 | 305.08 |
| retry1 / normal_before | plain | 284.45 | 290.35 | 294.85 | 294.54 |
| retry1 / normal_before | early | 293.55 | 297.88 | 301.78 | 300.81 |
| retry2 / compact | plain | 288.78 | 291.88 | 295.77 | 295.12 |
| retry2 / compact | early | 290.65 | 296.85 | 342.42 | 297.99 |
| retry2 / normal_after | plain | 288.17 | 292.89 | 300.35 | 296.73 |
| retry2 / normal_after | early | 294.94 | 298.86 | 303.80 | 303.42 |
| retry2 / normal_before | plain | 289.03 | 293.55 | 304.76 | 296.62 |
| retry2 / normal_before | early | 296.07 | 303.34 | 331.85 | 310.58 |
