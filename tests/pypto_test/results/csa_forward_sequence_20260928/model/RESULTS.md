# 当前性能版整模型forward（30f2b228 unchanged operator; optional CPU/GC diagnostics; atomic0/mode2/det0）

每rank前8步后连续10步纯decode _model_forward；16rank等权均值；无profiler。

P95由16rank×10步的160个相关样本计算，最慢rank分布另列；不是CSA时间、完整decode周期或初始化耗时。10步不能证明罕见长尾已消失。CPU position相同只验证请求位置对齐，不证明设备上的草稿token完全相同。

单位ms，负变化表示PTO更快。

| 档位 | Native均值 | PTO均值 | PTO变化 | Native/PTO P95 | Native/PTO最大值 | token/DSpark |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B4 | 46.577 | 43.301 | -7.03% | 47.144/43.753 | 47.426/43.877 | MEASURED_TOKEN_PASS |
| 128K/B8 | 56.583 | 54.760 | -3.22% | 57.241/55.491 | 57.502/55.872 | MEASURED_TOKEN_PASS |
| 128K/B16 | 72.675 | 70.526 | -2.96% | 79.681/72.817 | 80.071/73.260 | MEASURED_TOKEN_PASS |

每步取16rank中最大的forward耗时，再对10步求均值；用于观察EP16最慢rank的影响。

| 档位 | Native最慢rank均值 | PTO最慢rank均值 | PTO变化 | Native/PTO P95÷P50 | PTO较快步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 46.973 | 43.635 | -7.11% | 1.010/1.011 | 10/10 |
| 128K/B8 | 56.908 | 55.147 | -3.09% | 1.011/1.012 | 10/10 |
| 128K/B16 | 72.995 | 70.977 | -2.76% | 1.104/1.035 | 10/10 |

P95÷P50使用全部rank样本；每步最慢rank序列共10个样本，其P95等于最大值。
每步按相同稳态步编号对齐，最慢rank耗时不包含各rank起始时间偏差或步间等待。

已取得3/3档。逐rank样本、每步最慢rank分布、实际配置和错误详情见[forward.json](forward.json)。
旧版本全模型和当前单层数据不混入本表。
