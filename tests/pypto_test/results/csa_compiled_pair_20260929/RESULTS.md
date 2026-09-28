# 同配置编译后的 Native / PTO attention 半层

d8627207，CANN9.2、mode2、det0；同一卡，正式 layer4 权重 / 合成历史。
5次预热后的20次设备事件计时，profile独立采集。单位μs。

| 档位 | Native | PTO | PTO变化 | Native/PTO P95 | Native/PTO max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B16 | 1217.529 | 1090.391 | -10.442% | 1223.500/1391.560 | 1226.040/1455.720 |
| 8K/B24 | 1026.713 | 988.208 | -3.750% | 1030.200/1003.020 | 1033.800/1011.380 |

两侧开启同一套编译配置；Native追踪并静态编译内部算子，PTO保留生产自定义算子边界，
内部由PyPTO编译；实际CANN描述符、静态包和PTO调用次数分别记录。
PTO复用第二层已有的compact metadata，Native保留自己的metadata调用。
检查PTO自身eager/图重放、Top-K合法性和保护区；未做两侧逐元素精度或模型token/DSpark验收。
范围含HC_pre + norm + CSA + HC_post，不含MoE或EP16，不等于整模型forward。

**长档P95尚未通过**：PTO有3次1249.74–1455.72μs的拖尾，P95高于Native；
均值优势不能抵消此问题，异常样本全部保留，单独追加长档profile定位。

[完整证据与profile路径](evidence.json)
