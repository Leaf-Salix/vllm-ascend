# CANN 9.0 / 9.2 Native 单卡结果

同一卡、同源码、自定义算子包和 ATB；HC_pre→norm→CSA→HC_post 图重放。
每侧预热 5 次、正式 20 次；下表均值/P95不含 profiler。

| 档位 | 9.0 均值 μs | 9.2 均值 μs | 变化 | 9.0 P95 μs | 9.2 P95 μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 1326.612 | 1323.392 | -0.243% | 1332.14 | 1328.34 |
| 8K/B16 | 939.637 | 938.628 | -0.107% | 944.92 | 943.26 |

两档兼容性通过，均值变化不足 0.3%，没有证据表明仅切换运行库带来明显加速。
不是全七档或模型验收，也没有把 release QLI/Sparse 换成 9.2 内置实现。

## PyTorch profiling JSON

- [128K/B16 cann90](h131072_b16/cann90/profile/native/liteserver-hps-365a-00001_839566_20260928215152427_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [128K/B16 cann92](h131072_b16/cann92/profile/native/liteserver-hps-365a-00001_819262_20260928215057796_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B16 cann90](h8192_b16/cann90/profile/native/liteserver-hps-365a-00001_867181_20260928215324802_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)
- [8K/B16 cann92](h8192_b16/cann92/profile/native/liteserver-hps-365a-00001_859729_20260928215241573_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json)

9.2 最初自动导出因安装目录属主检查失败；用本用户目录的同版本 profiler 离线重导出通过。
未重跑设备采样，未更改他人安装。分算子数据来自独立一次 profile，不与20次正式均值相加或混算。
