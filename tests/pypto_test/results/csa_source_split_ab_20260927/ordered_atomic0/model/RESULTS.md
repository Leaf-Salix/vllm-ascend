# 当前性能版整模型forward（2a740c1f + ordered/candidate.patch; atomic_add=0）

每rank前8步后连续10步纯decode _model_forward；16rank等权均值；无profiler。Native复用ordered对照；唯一新增干预为PTO atomic_add=0，Native不使用此开关。

P95由16rank×10步的160个相关样本计算，最慢rank分布另列；不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。

单位ms，负变化表示PTO更快。

| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B16 | 72.009 | 73.701 | +2.35% | 73.052/75.243 | 73.159/75.869 | FAIL |
| 8K/B40 | 104.358 | 104.072 | -0.27% | 107.715/110.292 | 107.850/110.417 | MEASURED_TOKEN_PASS |

每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。

| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 72.371 | 74.188 | +2.51% | 1.015/1.022 |
| 8K/B40 | 104.445 | 104.199 | -0.23% | 1.033/1.065 |

P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。
每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。

已取得2/2档。逐rank样本、每步最慢rank分布、实际配置和错误详情见[forward.json](forward.json)。
旧版本全模型和当前单层数据不混入本表。
