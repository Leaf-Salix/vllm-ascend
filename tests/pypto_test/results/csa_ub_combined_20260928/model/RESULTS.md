# 当前性能版整模型forward（2d2f9ca0: Top-K UB + fixed QR input/gamma residency over d1f170ff; prewarmed events; atomic0/mode2/det0）

每rank前8步后连续10步纯decode _model_forward；16rank等权均值；无profiler。

P95由16rank×10步的160个相关样本计算，最慢rank分布另列；不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。CPU position相同只验证请求位置对齐，不证明设备上的草稿token完全相同。

单位ms，负变化表示PTO更快。

| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B16 | 72.784 | 69.626 | -4.34% | 73.559/72.454 | 73.768/72.591 | MEASURED_TOKEN_PASS |
| 8K/B16 | 65.553 | 61.147 | -6.72% | 67.256/62.969 | 67.418/63.118 | MEASURED_TOKEN_PASS |

每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。

| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 | PTO较快步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 73.268 | 69.941 | -4.54% | 1.008/1.045 | 10/10 |
| 8K/B16 | 65.886 | 61.377 | -6.84% | 1.025/1.033 | 10/10 |

P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。
每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。

已取得2/2档。逐rank样本、每步最慢rank分布、实际配置和错误详情见[forward.json](forward.json)。
旧版本全模型和当前单层数据不混入本表。

128K/8K的正式forward均值变化率按7:3加权为-5.054%。
同轮Native为基线，不代表对旧PTO的单因素收益。各档P95、逐步最慢rank及token/DSpark单独列示，不能由均值权重抵消异常。
