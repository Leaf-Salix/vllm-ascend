# 首轮5μs固定链对照

单位μs；原17级×48 AIV、work5、Scalar256、30样本、8次图内调用；全部精确检查通过。

| 版本 | 模式 | min | mean | max |
| --- | --- | ---: | ---: | ---: |
| normal_after | plain | 290.03 | 295.25 | 301.16 |
| normal_after | early | 294.49 | 298.42 | 302.22 |
| normal_before | plain | 288.27 | 292.91 | 300.10 |
| normal_before | early | 291.28 | 295.71 | 301.02 |
| probe | plain | 286.13 | 291.51 | 295.45 |
| probe | early | 290.57 | 295.17 | 298.72 |
