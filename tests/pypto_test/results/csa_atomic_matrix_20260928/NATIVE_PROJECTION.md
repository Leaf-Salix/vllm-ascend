# 当前固定规约 Q/KV 与 Native

Native为真实模型rank0三步63个CSA层；PTO为layer4合成输入单窗口DFX，非同输入A/B。

Native整kernel设备时间与PTO单block核内均值范围不同，不直接计算加速比。

单位μs。Native第4层列仅3步均值，完整63层样本另保存在JSON。PTO未把seed及调度等待加到matmul列。

| 档位 | Native第4层QA | PTO QA block均值/数量 | Native第4层KV | PTO KV block均值/数量 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 18.47 | 17.44/8 | 9.10 | 33.05/4 |
| 128K/B8 | 15.35 | 27.15/8 | 10.55 | 44.34/4 |
| 128K/B16 | 18.49 | 30.42/8 | 13.89 | 45.67/4 |
| 8K/B16 | 18.83 | 31.90/8 | 22.01 | 54.77/4 |
| 8K/B24 | 18.76 | 33.45/8 | 16.78 | 58.02/4 |
| 8K/B32 | 19.17 | 38.50/8 | 14.61 | 61.02/4 |
| 8K/B40 | 19.81 | 61.06/8 | 20.43 | 79.63/4 |

QA是RmsNormDynamicQuant的同stream最近前驱FP matmul；另外两个FP matmul必须在同一辅助stream，按_mla_prolog_multistream→cv_indexer_select_qli的源码顺序依次为KV与head权重投影。每层要求3个FP matmul、1个RmsNormDynamicQuant；小档trace未单列KV Norm，不用它强行匹配。

当前优先检查固定K条件下的M分工。M分组减少每核行数，但重复读权重会增加总搬运，必须同时看总核工作量、本体和EP16。
