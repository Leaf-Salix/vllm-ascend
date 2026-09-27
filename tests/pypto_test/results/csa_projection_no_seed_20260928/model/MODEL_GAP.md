# 免seed两档模型区间

rank0独立3步Level0，不分摊正式10步forward；两轮重新分配请求cache。
Native/PTO实际event模式0/1；FFN task busy可能重叠，不是critical span。首层通信包含跨rank到达等待。

CSA单位μs，FFN与GMM单位ms。独立profile不精确分账正式forward。

| 档位 | Native/PTO CSA均值 | Native/PTO CSA P50 | Native/PTO FFN总区间 | Native/PTO专家GMM任务总时间 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B8 | 1004.15/876.74 | 1001.46/877.50 | 25.264/25.338 | 4.782/4.873 |
| 8K/B16 | 969.55/797.36 | 968.20/797.58 | 32.700/31.020 | 6.651/6.443 |

当前8K/B16 rank0 CSA中位797.58μs，750μs目标仍未达到；只导出rank0不证明全rank层分布。
两侧原始全rank profiler数据仍保留，各case的rank0 trace已导出。
