# 单根Indexer采用后的PTO七档与现有Native

Native复用任务task_20260929_095651_194851727802的最新标准基线；PTO两档复用task_20260929_113003_337101716086，五档补测task_20260929_120323_390116317823。
PTO为同一冻结源码；各侧设备/来源在summary/evidence中逐项保留。
这是当前已有结果对照，不是同次Native/PTO A/B，不用于归因单项优化收益。
单根方案的严格同卡局部A/B见../csa_score_single_root_20260929/RESULTS.md。
七档PTO各自图重放的八类状态通过；旧七档未保存完整state，不能声称七档跨版本状态一致。
跨版本状态依据仅限已完成的两档A/B和尾段/padding；不是Native/PTO或模型token/DSpark验收。

CANN9.2/mode2/det0；PTO atomic0；每档5预热20次设备事件，单位μs。
Native显式torch.compile backend=npugraph_ex、dynamic=False/fullgraph=True/inplace_pass=True，自管图/static/superkernel；PTO生产自定义算子图。

| 档位 | Native | PTO | 耗时变化 | Native/PTO P95 | Native/PTO max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.298 | 650.226 | -13.106% | 751.100/662.620 | 751.840/664.760 |
| 128K/B8 | 859.563 | 746.698 | -13.131% | 861.620/760.540 | 864.720/763.660 |
| 128K/B16 | 1130.853 | 976.950 | -13.609% | 1135.580/998.520 | 1139.520/1007.640 |
| 128K/B24 | 1281.888 | 1249.672 | -2.513% | 1289.880/1271.480 | 1290.440/1277.760 |
| 8K/B16 | 757.482 | 781.510 | +3.172% | 760.120/799.360 | 761.040/801.300 |
| 8K/B24 | 915.371 | 956.560 | +4.500% | 920.340/981.440 | 920.900/1002.480 |
| 8K/B32 | 1062.079 | 1065.568 | +0.329% | 1066.220/1087.720 | 1070.680/1090.280 |

各上下文内batch等权，长短8:2：-7.938%。

核内数据来自独立profile/DFX；以下完整融合范围不能拆成单个Native QLI或Sparse核时，
也不能与各PTO均值直接相减计算纯算术收益。完整原始kernel名称和范围见evidence。

| 档位 | Native可见端点及完整范围AIC/AIV | PTO系数 | PTO Score AIC/AIV | PTO Indexer merge | PTO融合Sparse AIC/AIV |
| --- | --- | ---: | ---: | ---: | ---: |
| 128K/B4 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 273.608/276.361 | 1.584 | 65.798/72.661 | 6.857 | 47.086/52.072 |
| 128K/B8 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 317.510/320.053 | 2.114 | 115.626/119.761 | 7.644 | 82.371/86.115 |
| 128K/B16 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 517.282/519.891 | 2.210 | 226.253/229.177 | 9.696 | 149.872/153.718 |
| 128K/B24 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 582.740/585.160 | 2.607 | 328.753/336.322 | 9.008 | 228.048/231.983 |
| 8K/B16 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 136.321/138.705 | 3.650 | 44.373/48.846 | 10.045 | 126.263/130.129 |
| 8K/B24 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 192.451/194.807 | 2.757 | 30.781/44.526 | 11.130 | 178.707/182.547 |
| 8K/B32 | VllmQuantLightningIndexer→SparseAttnSharedkv（SuperKernel） 292.632/295.272 | 4.901 | 81.500/85.822 | 12.319 | 237.856/241.644 |

PTO已融合最终归一化、逆RoPE和发布，没有独立merge_norm；不是把缺失样本填0。

核内细节另采Native static compile开启、SuperKernel关闭的profile；不把其区间填入主性能表。

| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数/merge | Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 66.363/64.369 | 65.798/72.661 | 1.584/6.857 | 53.214/57.090 | 47.086/52.072 |
| 128K/B8 | 126.316/126.021 | 115.626/119.761 | 2.114/7.644 | 95.667/99.998 | 82.371/86.115 |
| 128K/B16 | 242.293/241.730 | 226.253/229.177 | 2.210/9.696 | 167.255/169.662 | 149.872/153.718 |
| 128K/B24 | 376.467/376.063 | 328.753/336.322 | 2.607/9.008 | 221.309/223.867 | 228.048/231.983 |
| 8K/B16 | 39.630/39.050 | 44.373/48.846 | 3.650/10.045 | 95.677/98.084 | 126.263/130.129 |
| 8K/B24 | 56.051/55.666 | 30.781/44.526 | 2.757/11.130 | 166.763/169.371 | 178.707/182.547 |
| 8K/B32 | 68.901/68.365 | 81.500/85.822 | 4.901/12.319 | 210.209/212.812 | 237.856/241.644 |

Native Sparse不含独立逆RoPE，而PTO融合Sparse包含它；以上仍是不同边界的参考，不能机械相减。

| 档位 | PTO P95/P50 | PTO max/P50 | 超过P50的105% |
| --- | ---: | ---: | ---: |
| 128K/B4 | 1.0158 | 1.0191 | 0/20 |
| 128K/B8 | 1.0218 | 1.0260 | 0/20 |
| 128K/B16 | 1.0232 | 1.0325 | 0/20 |
| 128K/B24 | 1.0143 | 1.0193 | 0/20 |
| 8K/B16 | 1.0251 | 1.0276 | 0/20 |
| 8K/B24 | 1.0245 | 1.0465 | 0/20 |
| 8K/B32 | 1.0208 | 1.0232 | 0/20 |

不删拖尾样本；20次正常采样不能关闭历史间歇拖尾或EP16稳定性问题。
Native重放对同图compiled callable、PTO对自身eager；不冒称Native/PTO跨实现精度通过。
全部配置、样本、28个官方join核对窗口、两侧正式路径的14个PyTorch JSON及7个Native核内诊断JSON见[evidence.json](evidence.json)。
