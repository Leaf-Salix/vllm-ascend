# 当前性能版整模型forward（2a740c1f (original cache layout)）

每rank前8步后连续10步纯decode _model_forward；Native复用前一任务235803控制，PTO为原布局2a740c1f；入场仍未门控。

P95由16rank×10步的160个相关样本计算，最慢rank分布另列；不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。

单位ms，负变化表示PTO更快。

| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B16 | 73.497 | 75.979 | +3.38% | 74.293/77.580 | 74.581/77.974 | MEASURED_TOKEN_PASS |
| 8K/B40 | 102.743 | 102.907 | +0.16% | 104.725/103.793 | 104.877/103.861 | MEASURED_TOKEN_PASS |

每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。

| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 73.917 | 76.279 | +3.20% | 1.010/1.024 |
| 8K/B40 | 102.816 | 102.964 | +0.14% | 1.021/1.007 |

P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。
每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。

已取得2/2档。逐rank样本、每步最慢rank分布、实际配置和错误详情见[forward.json](forward.json)。
旧版本全模型和当前单层数据不混入本表。
