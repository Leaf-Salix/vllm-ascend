# 完整 min/mean/max

单位μs。min/max 是30个图内8次调用均摊样本的范围，不是单次调用尾延迟。

| 路径 | 目标单task | 级数 | 模式 | min | mean | max |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| old_before | 5 | 1 | plain | 76.23 | 79.67 | 82.37 |
| old_before | 5 | 17 | plain | 285.91 | 290.62 | 294.85 |
| old_before | 50 | 1 | plain | 121.44 | 124.53 | 127.12 |
| old_before | 50 | 17 | plain | 1014.67 | 1019.10 | 1032.73 |
| cached_before | 5 | 1 | plain | 79.10 | 83.60 | 88.15 |
| cached_before | 5 | 1 | early | 82.41 | 85.97 | 90.33 |
| cached_before | 5 | 17 | plain | 288.14 | 292.63 | 296.75 |
| cached_before | 5 | 17 | early | 304.04 | 311.27 | 341.00 |
| cached_before | 50 | 1 | plain | 125.53 | 128.24 | 133.18 |
| cached_before | 50 | 1 | early | 127.89 | 130.91 | 134.92 |
| cached_before | 50 | 17 | plain | 1018.72 | 1022.15 | 1028.05 |
| cached_before | 50 | 17 | early | 1001.72 | 1005.59 | 1011.41 |
| inline | 5 | 1 | plain | 80.02 | 83.42 | 92.28 |
| inline | 5 | 1 | early | 82.97 | 86.65 | 91.76 |
| inline | 5 | 17 | plain | 287.53 | 292.47 | 300.28 |
| inline | 5 | 17 | early | 307.22 | 310.81 | 314.29 |
| inline | 50 | 1 | plain | 124.92 | 129.03 | 135.16 |
| inline | 50 | 1 | early | 127.94 | 131.87 | 137.13 |
| inline | 50 | 17 | plain | 1018.74 | 1022.21 | 1038.84 |
| inline | 50 | 17 | early | 1003.30 | 1006.85 | 1010.91 |
| cached_after | 5 | 1 | plain | 79.83 | 84.63 | 88.73 |
| cached_after | 5 | 1 | early | 83.15 | 87.60 | 91.83 |
| cached_after | 5 | 17 | plain | 288.16 | 294.06 | 298.84 |
| cached_after | 5 | 17 | early | 307.57 | 311.34 | 317.32 |
| cached_after | 50 | 1 | plain | 124.66 | 128.08 | 131.88 |
| cached_after | 50 | 1 | early | 127.50 | 131.11 | 137.32 |
| cached_after | 50 | 17 | plain | 1017.23 | 1021.34 | 1026.49 |
| cached_after | 50 | 17 | early | 1003.05 | 1006.71 | 1009.92 |
| old_after | 5 | 1 | plain | 74.45 | 77.59 | 81.47 |
| old_after | 5 | 17 | plain | 283.66 | 288.19 | 292.31 |
| old_after | 50 | 1 | plain | 119.09 | 122.86 | 127.25 |
| old_after | 50 | 17 | plain | 1012.14 | 1016.52 | 1020.49 |
