# 5μs固定链对照

单位μs；每轮30样本，work5、Scalar256、17级×48 AIV、8次图内调用，逐级数值、换输入与保护区全部通过。

| 轮次/版本 | 模式 | min | mean | max |
| --- | --- | ---: | ---: | ---: |
| retry1 / normal_after | plain | 289.15 | 295.92 | 304.11 |
| retry1 / normal_after | early | 290.66 | 299.42 | 311.72 |
| retry1 / normal_before | plain | 290.72 | 298.51 | 305.21 |
| retry1 / normal_before | early | 293.19 | 299.23 | 304.54 |
| retry1 / probe | plain | 288.70 | 296.46 | 302.55 |
| retry1 / probe | early | 290.37 | 296.73 | 302.92 |
| retry2 / normal_after | plain | 290.45 | 293.96 | 298.20 |
| retry2 / normal_after | early | 292.82 | 295.41 | 298.16 |
| retry2 / normal_before | plain | 287.80 | 293.56 | 298.89 |
| retry2 / normal_before | early | 292.87 | 296.81 | 300.78 |
| retry2 / probe | plain | 289.05 | 294.71 | 299.48 |
| retry2 / probe | early | 293.46 | 298.90 | 307.83 |
