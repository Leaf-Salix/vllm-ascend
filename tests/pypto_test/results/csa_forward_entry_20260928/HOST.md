# forward入场主机诊断

同一次正式10步；主机monotonic_ns可在本机跨进程比较，thread_cpu_ns只在同一线程内作差。

主机墙钟减线程CPU时间含阻塞、等待和调度，不能只称CPU抢占；GC重合是观测，不自动证明唯一根因。

正式性能见[RESULTS.md](model/RESULTS.md)，设备相对进入异常见[ARRIVAL.md](ARRIVAL.md)。

| 档位 | 侧 | Worker冻结对象数范围 | 正式窗口GC次数/最大ms | 最慢准备墙钟/线程CPU ms | 最慢forward提交墙钟ms |
| --- | --- | --- | ---: | ---: | ---: |
| 128K/B16 | native | 0–0 | 0/0.000 | 60.791/56.453 | 0.744 |
| 128K/B16 | pto | 0–0 | 0/0.000 | 62.224/53.086 | 1.114 |

相对设备入场异常超过2ms的步骤（诊断筛选，不是验收阈值）：

| 档位 | 侧 | step | rank | 设备异常ms | 主机execute相对中位ms | 主机forward相对中位ms | 准备/通常ms | 同execute内GC ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

本次未出现超过2ms的设备入场异常；不能由未复现宣称先前长尾已修复。

逐rank、逐step的原始主机时间、CPU差值、GC区间及设备forward见[host.json](host.json)。
表中GC只统计首个正式execute_entry到最后正式execute_return之间（含步间）的观测，其他GC仍保留在JSON，不用预热期GC解释正式窗口长尾。
没有修改GC策略，也没有把准备耗时从正式结果中扣除。
