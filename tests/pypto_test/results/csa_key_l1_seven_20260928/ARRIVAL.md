# 正式forward尾部：相对进入时刻

复用同一次正式forward的设备起始事件，不重跑，不剔除或修正性能样本。

每rank起始时间减rank0，再减该差值的10步中位数，消去稳定的跨设备时钟偏移；每步再减16rank中位数，得到相对通常启动关系的瞬时偏移。

这是短窗口内的相对启动异常，不是绝对同步时间；不校正长期时钟漂移。只能定位到forward开始事件之前，不能区分CPU调度、GC、前序草稿或设备队列等待。

以下只列相对进入异常超过2ms的诊断点；2ms不是验收阈值，JSON保留全部10步。

| 档位 | 侧 | 正式step | 相对晚进入rank | 偏移ms | 该rank forward/平时中位ms | 其余rank增加ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | native | 8 | 4 | 2.002 | 44.913/45.336 | 1.649 |
| 128K/B4 | native | 16 | 0 | 2.560 | 44.764/45.203 | 1.998 |
| 128K/B4 | pto | 11 | 14 | 2.144 | 43.111/43.435 | 1.696 |
| 128K/B4 | pto | 15 | 14 | 3.786 | 43.289/43.435 | 3.491 |
| 128K/B4 | pto | 16 | 3 | 4.534 | 43.037/43.410 | 4.093 |

晚进入rank自身forward正常、其余rank耗时同量级增加时，与EP等待放大相符；需逐样本核对，不能只因进入较晚就认定根因或给CSA加sync_start。
正式均值/P95/max仍原样保留在[RESULTS.md](model/RESULTS.md)。

单位依据：[Ascend API](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1780.html)、[torch_npu实现](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/core/npu/NPUEvent.cpp#L173)。
