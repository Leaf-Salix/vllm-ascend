# 全部计时

单位μs。min/max为30个图内8次调用均摊样本的范围。

| 路径 | 单task目标 | 级数 | 模式 | min | mean | max |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| old_before | 5 | 1 | plain | 74.87 | 79.10 | 84.48 |
| old_before | 5 | 17 | plain | 286.53 | 289.63 | 293.88 |
| old_before | 50 | 1 | plain | 121.00 | 123.96 | 128.23 |
| old_before | 50 | 17 | plain | 1013.86 | 1016.40 | 1022.58 |
| cached_before | 5 | 1 | plain | 80.64 | 82.87 | 86.33 |
| cached_before | 5 | 1 | early | 82.87 | 85.16 | 88.88 |
| cached_before | 5 | 17 | plain | 290.03 | 293.30 | 298.13 |
| cached_before | 5 | 17 | early | 305.97 | 309.86 | 313.12 |
| cached_before | 50 | 1 | plain | 125.40 | 128.62 | 132.40 |
| cached_before | 50 | 1 | early | 126.77 | 131.44 | 137.05 |
| cached_before | 50 | 17 | plain | 1020.62 | 1023.30 | 1027.49 |
| cached_before | 50 | 17 | early | 1000.32 | 1004.50 | 1007.06 |
| no_late_poll | 5 | 1 | plain | 80.79 | 83.79 | 87.67 |
| no_late_poll | 5 | 1 | early | 82.25 | 84.85 | 87.85 |
| no_late_poll | 5 | 17 | plain | 288.51 | 294.28 | 299.01 |
| no_late_poll | 5 | 17 | early | 298.46 | 303.47 | 307.18 |
| no_late_poll | 50 | 1 | plain | 126.46 | 129.11 | 131.90 |
| no_late_poll | 50 | 1 | early | 126.91 | 130.35 | 134.81 |
| no_late_poll | 50 | 17 | plain | 1019.38 | 1023.30 | 1028.49 |
| no_late_poll | 50 | 17 | early | 998.72 | 1002.09 | 1005.10 |
| cached_after | 5 | 1 | plain | 80.69 | 83.79 | 86.21 |
| cached_after | 5 | 1 | early | 83.65 | 86.04 | 89.29 |
| cached_after | 5 | 17 | plain | 288.44 | 294.27 | 300.24 |
| cached_after | 5 | 17 | early | 303.40 | 309.15 | 312.90 |
| cached_after | 50 | 1 | plain | 125.82 | 128.48 | 131.50 |
| cached_after | 50 | 1 | early | 128.95 | 131.66 | 134.52 |
| cached_after | 50 | 17 | plain | 1017.34 | 1022.36 | 1027.26 |
| cached_after | 50 | 17 | early | 1000.86 | 1004.94 | 1026.27 |
| old_after | 5 | 1 | plain | 74.54 | 78.09 | 81.06 |
| old_after | 5 | 17 | plain | 283.55 | 288.31 | 292.66 |
| old_after | 50 | 1 | plain | 119.12 | 121.35 | 126.87 |
| old_after | 50 | 17 | plain | 1010.40 | 1015.45 | 1021.89 |
