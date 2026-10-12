# 全部计时

单位μs。min/max为30个图内8次调用均摊样本的范围。

| 路径 | 单task目标 | 级数 | 模式 | min | mean | max |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| old_before | 5 | 1 | plain | 76.37 | 79.75 | 82.93 |
| old_before | 5 | 17 | plain | 289.91 | 293.91 | 301.34 |
| old_before | 50 | 1 | plain | 119.91 | 123.70 | 128.75 |
| old_before | 50 | 17 | plain | 1013.92 | 1016.88 | 1021.57 |
| cached_before | 5 | 1 | plain | 80.37 | 82.52 | 84.55 |
| cached_before | 5 | 1 | early | 83.19 | 85.67 | 89.24 |
| cached_before | 5 | 17 | plain | 287.23 | 292.90 | 296.73 |
| cached_before | 5 | 17 | early | 305.03 | 308.58 | 316.38 |
| cached_before | 50 | 1 | plain | 124.50 | 127.31 | 130.97 |
| cached_before | 50 | 1 | early | 126.68 | 130.54 | 133.67 |
| cached_before | 50 | 17 | plain | 1018.45 | 1021.94 | 1025.51 |
| cached_before | 50 | 17 | early | 1000.45 | 1003.99 | 1008.13 |
| core_init | 5 | 1 | plain | 79.68 | 83.08 | 93.88 |
| core_init | 5 | 1 | early | 82.73 | 86.75 | 93.41 |
| core_init | 5 | 17 | plain | 289.80 | 293.25 | 298.94 |
| core_init | 5 | 17 | early | 296.31 | 300.36 | 306.74 |
| core_init | 50 | 1 | plain | 124.09 | 127.34 | 132.84 |
| core_init | 50 | 1 | early | 127.70 | 130.42 | 136.59 |
| core_init | 50 | 17 | plain | 1016.02 | 1021.35 | 1024.59 |
| core_init | 50 | 17 | early | 999.16 | 1004.59 | 1053.56 |
| cached_after | 5 | 1 | plain | 78.34 | 81.64 | 86.21 |
| cached_after | 5 | 1 | early | 80.77 | 84.52 | 89.70 |
| cached_after | 5 | 17 | plain | 285.47 | 290.82 | 295.30 |
| cached_after | 5 | 17 | early | 302.81 | 305.93 | 309.86 |
| cached_after | 50 | 1 | plain | 123.01 | 126.07 | 131.14 |
| cached_after | 50 | 1 | early | 126.40 | 129.37 | 131.92 |
| cached_after | 50 | 17 | plain | 1017.32 | 1020.52 | 1023.95 |
| cached_after | 50 | 17 | early | 999.43 | 1002.28 | 1005.65 |
| old_after | 5 | 1 | plain | 74.58 | 77.41 | 81.53 |
| old_after | 5 | 17 | plain | 284.04 | 288.38 | 294.74 |
| old_after | 50 | 1 | plain | 117.57 | 121.93 | 128.36 |
| old_after | 50 | 17 | plain | 1011.44 | 1015.02 | 1019.90 |
