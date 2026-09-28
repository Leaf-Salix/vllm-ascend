# 当前性能版整模型forward（554b3bca; Key L1 + UB/QR; mode2/atomic0/det0; EPLB off）

每rank前8步后连续10步纯decode _model_forward；16rank等权均值；无profiler。

P95由16rank×10步的160个相关样本计算，最慢rank分布另列；不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。CPU position相同只验证请求位置对齐，不证明设备上的草稿token完全相同。

单位ms，负变化表示PTO更快。

| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B4 | 45.605 | 44.308 | -2.85% | 47.415/47.628 | 47.720/47.862 | MEASURED_TOKEN_PASS |
| 128K/B8 | 55.394 | 55.439 | +0.08% | 56.552/56.231 | 56.725/56.359 | MEASURED_TOKEN_PASS |
| 128K/B16 | 72.894 | 69.162 | -5.12% | 73.867/70.198 | 73.998/70.628 | MEASURED_TOKEN_PASS |

每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。

| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 | PTO较快步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 45.925 | 44.698 | -2.67% | 1.045/1.094 | 7/10 |
| 128K/B8 | 55.691 | 55.833 | +0.26% | 1.023/1.013 | 4/10 |
| 128K/B16 | 73.247 | 69.595 | -4.99% | 1.012/1.016 | 10/10 |

P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。
每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。

已取得3/7档。逐rank样本、每步最慢rank分布、实际配置和错误详情见[forward.json](forward.json)。
旧版本全模型和当前单层数据不混入本表。
