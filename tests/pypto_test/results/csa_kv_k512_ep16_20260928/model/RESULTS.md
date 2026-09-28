# 当前性能版整模型forward（30f2b228 + KV K512; prewarmed timing events and CPU phase diagnostics; atomic0/mode2/det0）

每rank前8步后连续10步纯decode _model_forward；16rank等权均值；无profiler。

P95由16rank×10步的160个相关样本计算，最慢rank分布另列；不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。CPU position相同只验证请求位置对齐，不证明设备上的草稿token完全相同。

单位ms，负变化表示PTO更快。

| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B4 | 45.708 | 44.694 | -2.22% | 47.830/45.486 | 48.124/45.576 | MEASURED_TOKEN_PASS |
| 128K/B8 | 56.780 | 55.631 | -2.02% | 57.839/56.196 | 58.173/56.381 | MEASURED_TOKEN_PASS |
| 128K/B16 | 73.327 | 70.947 | -3.25% | 74.634/72.039 | 74.749/72.385 | MEASURED_TOKEN_PASS |
| 8K/B40 | 104.243 | 101.996 | -2.16% | 107.740/103.586 | 107.846/103.679 | MEASURED_TOKEN_PASS |

每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。

| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 | PTO较快步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 46.030 | 45.024 | -2.18% | 1.050/1.017 | 10/10 |
| 128K/B8 | 57.285 | 56.004 | -2.23% | 1.017/1.008 | 10/10 |
| 128K/B16 | 73.868 | 71.393 | -3.35% | 1.015/1.015 | 10/10 |
| 8K/B40 | 104.326 | 102.095 | -2.14% | 1.035/1.017 | 10/10 |

P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。
每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。

已取得4/4档。逐rank样本、每步最慢rank分布、实际配置和错误详情见[forward.json](forward.json)。
旧版本全模型和当前单层数据不混入本表。
