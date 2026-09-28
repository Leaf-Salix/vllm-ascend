# 持续的主机入场偏移

本机跨进程monotonic_ns；每步相对16rank中位数，不减各rank十步中位数。

主机入口不是设备开始或通信到达；不从正式forward扣除差额，不单独证明CPU抢占。

设备起始事件分析会消去稳定的跨设备时钟偏移，也会消去持续的rank迟到；本表补充后者。

| 档位 | 侧 | 十步平均execute入口跨度ms | forward入口跨度ms | forward入口最大跨度ms |
| --- | --- | ---: | ---: | ---: |
| 128K/B4 | native | 1.667 | 1.114 | 3.830 |
| 128K/B4 | pto | 1.693 | 1.112 | 5.432 |
| 128K/B8 | native | 1.573 | 0.894 | 2.146 |
| 128K/B8 | pto | 8.245 | 2.581 | 3.125 |

以下列出十步中位偏移≥1ms的rank，仅为诊断筛选；全部rank保留在JSON。

| 档位 | 侧/rank | 主机forward中位偏移ms | 设备forward中位ms | metadata墙钟/线程CPU中位ms |
| --- | --- | ---: | ---: | ---: |
| 128K/B8 | pto/5 | 2.231 | 53.257 | 6.165/4.064 |
| 128K/B8 | pto/9 | 1.852 | 53.501 | 5.771/4.095 |

父子builder区间嵌套，不相加。墙钟与线程CPU差距包含等待及调度，尚未区分具体阻塞调用。
[逐rank十步偏移、设备forward及分项](host_spread.json)、[瞬时偏移](ARRIVAL.md)、[正式forward](model/RESULTS.md)。
