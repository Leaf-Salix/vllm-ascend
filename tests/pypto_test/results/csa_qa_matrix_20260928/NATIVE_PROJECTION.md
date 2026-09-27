# 当前固定规约 Q/KV 与 Native

Native为真实模型rank0三步63个CSA层；PTO为layer4合成输入单窗口DFX，非同输入A/B。

Native整kernel设备时间与PTO单block核内均值范围不同，不直接计算加速比。

单位μs。Native第4层列仅3步均值，完整63层样本另保存在JSON。PTO未把seed及调度等待加到matmul列。

| 档位 | Native第4层QA | PTO QA block均值/数量 | Native第4层KV | PTO KV block均值/数量 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 19.79 | 17.70/8 | 9.09 | 34.19/4 |
| 128K/B8 | 14.97 | 20.30/8 | 10.09 | 37.38/4 |
| 128K/B16 | 19.15 | 16.33/24 | 15.33 | 27.76/12 |
| 8K/B16 | 18.88 | 15.87/24 | 19.20 | 27.86/12 |
| 8K/B24 | 18.96 | 19.09/16 | 18.81 | 36.07/8 |
| 8K/B32 | 19.85 | 17.42/24 | 14.43 | 24.06/12 |
| 8K/B40 | 20.77 | 27.05/24 | 16.97 | 45.57/12 |

QA是RmsNormDynamicQuant的同stream最近前驱FP matmul；另外两个FP matmul必须在同一辅助stream，按_mla_prolog_multistream→cv_indexer_select_qli的源码顺序依次为KV与head权重投影。每层要求3个FP matmul、1个RmsNormDynamicQuant；小档trace未单列KV Norm，不用它强行匹配。

当前优先检查固定K条件下的M分工。M分组减少每核行数，但重复读权重会增加总搬运，必须同时看总核工作量、本体和EP16。
