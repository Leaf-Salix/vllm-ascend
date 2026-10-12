# 首轮5μs固定链对照

单位μs；原17级×48 AIV、work5、Scalar256、30样本、8次图内调用；全部精确检查通过。

| 版本 | 模式 | min | mean | max |
| --- | --- | ---: | ---: | ---: |
| normal_after | plain | 291.77 | 297.24 | 302.37 |
| normal_after | early | 291.39 | 298.85 | 303.15 |
| normal_before | plain | 292.67 | 298.51 | 302.12 |
| normal_before | early | 292.93 | 299.05 | 304.66 |
| probe | plain | 294.25 | 299.74 | 303.84 |
| probe | early | 294.21 | 301.36 | 305.73 |
