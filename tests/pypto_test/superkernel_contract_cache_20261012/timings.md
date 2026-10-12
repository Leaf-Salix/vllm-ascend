# 完整 min/mean/max

单位μs。min/max 是30个图内8次调用均摊样本的范围，不是单次调用尾延迟。

| 路径 | 目标单task | 级数 | 模式 | min | mean | max |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| old_before | 5 | 1 | plain | 74.27 | 78.58 | 83.80 |
| old_before | 5 | 17 | plain | 285.15 | 289.20 | 292.66 |
| old_before | 50 | 1 | plain | 120.92 | 123.08 | 126.57 |
| old_before | 50 | 17 | plain | 1012.73 | 1015.87 | 1020.02 |
| candidate | 5 | 1 | plain | 79.83 | 83.20 | 87.29 |
| candidate | 5 | 1 | early | 83.49 | 86.21 | 89.21 |
| candidate | 5 | 17 | plain | 287.54 | 291.40 | 298.37 |
| candidate | 5 | 17 | early | 304.65 | 308.40 | 312.98 |
| candidate | 50 | 1 | plain | 124.22 | 127.07 | 131.01 |
| candidate | 50 | 1 | early | 127.16 | 129.89 | 134.12 |
| candidate | 50 | 17 | plain | 1017.69 | 1020.94 | 1027.67 |
| candidate | 50 | 17 | early | 998.78 | 1002.52 | 1005.41 |
| old_after | 5 | 1 | plain | 74.85 | 77.87 | 82.87 |
| old_after | 5 | 17 | plain | 284.84 | 289.27 | 295.53 |
| old_after | 50 | 1 | plain | 118.53 | 122.68 | 126.09 |
| old_after | 50 | 17 | plain | 1013.00 | 1015.78 | 1019.25 |
