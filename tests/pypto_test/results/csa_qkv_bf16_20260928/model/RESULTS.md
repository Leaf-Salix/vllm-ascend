# 当前性能版整模型forward（2a740c1f + csa_qkv_bf16_20260928/candidate.patch; test admission=8dd737f4）

每rank前8步后连续10步纯decode _model_forward；16rank等权均值；无profiler。

P95由16rank×10步的160个相关样本计算，最慢rank分布另列；不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。CPU position相同只验证请求位置对齐，不证明设备上的草稿token完全相同。

单位ms，负变化表示PTO更快。

| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B16 | 72.071 | 76.510 | +6.16% | 73.023/77.301 | 73.345/77.520 | MEASURED_TOKEN_PASS |
| 8K/B40 | 104.243 | 103.593 | -0.62% | 107.595/105.494 | 107.722/105.607 | MEASURED_TOKEN_PASS |

每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。

| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B16 | 72.421 | 76.988 | +6.31% | 1.013/1.009 |
| 8K/B40 | 104.332 | 103.679 | -0.63% | 1.032/1.019 |

P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。
每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。

已取得2/2档。逐rank样本、每步最慢rank分布、实际配置和错误详情见[forward.json](forward.json)。
旧版本全模型和当前单层数据不混入本表。
