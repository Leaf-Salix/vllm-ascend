# 最新标准Native七档基线

Native七档已完成；不等待PTO配套采集。CANN9.2、mode2、det0、EPLB关闭。
显式torch.compile backend=npugraph_ex、dynamic=False/fullgraph=True/inplace_pass=True；
static_kernel_compile和SuperKernel开启，多流由后端图捕获。
固定shape/地址，使用后端创建的唯一NPUGraph直接replay，无主机更新节点。
正式第二个CSA层真实权重、独立合成历史；完整HC_pre+norm+CSA+HC_post，单位μs。
5次预热后20次设备事件，profile独立采集；不含编译/初始化，不是模型forward。

| 档位 | Native均值 | P50 | P95 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.298 | 748.570 | 751.100 | 751.840 |
| 128K/B8 | 859.563 | 859.610 | 861.620 | 864.720 |
| 128K/B16 | 1130.853 | 1130.420 | 1135.580 | 1139.520 |
| 128K/B24 | 1281.888 | 1281.150 | 1289.880 | 1290.440 |
| 8K/B16 | 757.482 | 758.100 | 760.120 | 761.040 |
| 8K/B24 | 915.371 | 915.180 | 920.340 | 920.900 |
| 8K/B32 | 1062.079 | 1062.140 | 1066.220 | 1070.680 |

静态编译安装、实际SuperKernel/多流、八类同图状态及保护区均检查通过。
这是Native自身图重放检查，不是Native/PTO跨实现精度或逐token/DSpark验收。
PTO使用现有已验证路径和数据，不要求复刻Native编译配置；新HC_post收益不推算进旧版本读数。
[原样本、实际配置与profile](native_summary.json)。
