# Native后端生成图的直接重放计时

CANN9.2 / mode2 / det0，显式torch.compile backend=npugraph_ex，static和superkernel开启。
每档5次预热、20次无profiler图外设备事件；profile另采，单位μs。

| 档位 | 均值 | P50 | P95 | 最大值 | 独立profile设备跨度 | 同图状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 128K/B16 | 1122.652 | 1122.490 | 1128.400 | 1131.380 | 1151.000 | 8项零容差通过 |
| 8K/B24 | 940.762 | 940.520 | 945.280 | 945.740 | 997.500 | 8项零容差通过 |

测量图由npugraph_ex生成并优化，保留两条计算stream；未捕获第二张外图。
固定地址/shape、唯一后端图且无主机更新节点；八类状态与同初态的compiled callable零容差一致。
报告中的eager_comparison沿用旧字段名，本轮参考实际为同一图通过编译包装调用的结果。
这是CSA完整设备区间的计时校准，不是改变输入地址的生产接口或整模型验收。
不再测superkernel关闭组；不同轮的调用计时与独立profile不能相减作精确开销归因。
此前七档Native仍属旧编译/捕获入口，不能替换其中两行后宣称得到新版七档。

[精简样本与检查](summary.json)、[完整配置与profile](evidence.json)。
