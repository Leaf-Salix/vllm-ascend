# CSA：最新七档核内差距与下一步

更新：2026-09-29。本页完整七档对应 **2ed8ae2e / CANN9.2**，包含HC_post残差常驻与长S6单根Indexer。
此后已保留9a868d26的O-B优化，以及通过两档与边界的收尾融合，未外推更新七档表。
七档统一冻结源码：128K B4/B8/B16/B24、8K B16/B24/B32；复用本轮长B16/短B24，补测其余五档。
Native沿用已完成的最新标准七档，各侧都是auto设备0，但采样分属不同任务；
这是当前已有结果对照，不是同次A/B，不能据此归因单项优化收益。

Native整体：CANN9.2、显式torch.compile backend=npugraph_ex、dynamic=False/fullgraph=True、
inplace_pass=True、static compile和SuperKernel开启，由后端图直接replay。
Native核内诊断只关闭SuperKernel，其他配置保持；诊断Duration不混入正式计时。
PTO保留custom-op图边界及自己的PyPTO实现；以后直接使用已有验证数据，不强制套用Native编译选项。

真实第二个CSA层权重、独立合成历史，mode2/det0、PTO atomic0、EPLB关闭。
完整HC_pre+norm+CSA+HC_post，5预热20次设备事件；各侧独立PyTorch profile、每档四窗DFX。
Native cache不改，PTO直接分页读写，WO_A借用Native NZ29原地址。
Native仍为release custom二进制，最新ops-transformer28f40354/ops-nn19614968/ops-math361722c0只是源码参考。
[完整结果](results/csa_single_root_seven_20260929/RESULTS.md)、[任务与pipeline明细](results/csa_single_root_seven_20260929/TASKS.md)、[本轮七份PTO泳道下载](results/csa_single_root_seven_20260929/download_pto_swimlanes/README.md)。

## 完整CSA

| 档位 | Native均值μs | PTO均值μs | PTO变化 | Native/PTO P95μs |
| --- | --- | --- | --- | --- |
| 128K/B4 | 748.298 | 650.226 | -13.106% | 751.100/662.620 |
| 128K/B8 | 859.563 | 746.698 | -13.131% | 861.620/760.540 |
| 128K/B16 | 1130.853 | 976.950 | -13.609% | 1135.580/998.520 |
| 128K/B24 | 1281.888 | 1249.672 | -2.513% | 1289.880/1271.480 |
| 8K/B16 | 757.482 | 781.510 | +3.172% | 760.120/799.360 |
| 8K/B24 | 915.371 | 956.560 | +4.500% | 920.340/981.440 |
| 8K/B32 | 1062.079 | 1065.568 | +0.329% | 1066.220/1087.720 |

各上下文内batch等权，128K -10.590%、8K +2.667%，8:2 -7.938%。
PTO P95/P50为1.0143–1.0251，max/P50最高1.0465；累计0/140超过各自P50的105%。
不删样本，也不以本轮正常样本关闭历史间歇尾部或EP16问题。七档各自图重放及保护区通过，
跨版本状态依据限于两代表档与尾段/padding；不能替代Native/PTO精度或模型token/DSpark验收。

## Indexer核内

Native独立QLI包含系数、Score和归并；PTO系数/Score/merge独立，PMU与worker计时边界不同。
单位μs，PTO为四窗均值；差值仅用于定位候选，不是严格纯算术差或可回收时长。

| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数 | PTO merge |
| --- | --- | --- | --- | --- |
| 128K/B4 | 66.363/64.369 | 65.798/72.661 | 1.584 | 6.857 |
| 128K/B8 | 126.316/126.021 | 115.626/119.761 | 2.114 | 7.644 |
| 128K/B16 | 242.293/241.730 | 226.253/229.177 | 2.210 | 9.696 |
| 128K/B24 | 376.467/376.063 | 328.753/336.322 | 2.607 | 9.008 |
| 8K/B16 | 39.630/39.050 | 44.373/48.846 | 3.650 | 10.045 |
| 8K/B24 | 56.051/55.666 | 30.781/44.526 | 2.757 | 11.130 |
| 8K/B32 | 68.901/68.365 | 81.500/85.822 | 4.901 | 12.319 |

长S6已改为完整query与单根发布；系数、scale更新和最终merge仍是独立任务。
短B16/B32继续双query，短B24继续S6，不能从单个Score核时推断完整Indexer胜负。
主性能Native含QLI→Sparse SuperKernel，不拆解为独立QLI/Sparse核时。
[算术、任务与输入差异](DSV4_FLASH_CSA_INDEXER_NATIVE_GAP.md)。

## Sparse attention

PTO已融合最终归一化、逆RoPE和发布，没有独立merge_norm；Native此处Sparse不含独立逆RoPE。

| 档位 | Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |
| --- | --- | --- |
| 128K/B4 | 53.214/57.090 | 47.086/52.072 |
| 128K/B8 | 95.667/99.998 | 82.371/86.115 |
| 128K/B16 | 167.255/169.662 | 149.872/153.718 |
| 128K/B24 | 221.309/223.867 | 228.048/231.983 |
| 8K/B16 | 95.677/98.084 | 126.263/130.129 |
| 8K/B24 | 166.763/169.371 | 178.707/182.547 |
| 8K/B32 | 210.209/212.812 | 237.856/241.644 |

短档Sparse仍是核内研究入口，且当前PTO包含额外逆RoPE；差值不能机械当成可回收时间。
Native各pipeline计数可重叠，不能把MAC/MTE/FIX/Vector计数相加作总时长。

## HC与Q/O任务组织

下面为128K/B16、B24的四窗均值。启动分散包括必要多波，不能全算调度器软件开销。

| Task | B16 worker/核时μs/启动分散μs | B24 worker/核时μs/启动分散μs |
| --- | --- | --- |
| hc_widen_rms | 12/12.967/1.160 | 18/12.630/5.075 |
| hc_pre_linear | 24/7.290/7.480 | 24/10.897/4.770 |
| hc_pre_linear_reduce | 6/1.596/0.270 | 9/2.147/0.235 |
| split_pre_post | 12/4.100/0.505 | 16/4.482/0.370 |
| comb_sinkhorn | 12/15.521/0.310 | 18/15.671/4.315 |
| mix_x_rms_norm | 12/18.387/0.310 | 18/16.866/3.435 |
| qproj_matmul | 24/38.646/58.685 | 24/47.792/64.070 |
| qproj_dequant_rms_nope_rope | 48/22.146/15.980 | 48/31.172/37.940 |
| proj_a_mm | 64/26.969/61.670 | 64/26.844/66.055 |
| quant | 24/7.916/54.060 | 40/6.631/60.350 |
| proj_b_mm | 64/10.670/51.020 | 64/18.699/64.460 |
| hc_post | 24/16.898/0.650 | 36/17.654/1.670 |

Native HC_post按不同token分工，而PTO每worker最多4个token，不能只比单worker均值。
已采用的残差常驻优化使两代表档HC_post核内−17.423%/−24.785%，四窗范围不重叠；
八类状态及B3/H127/padding通过，共享精度/性能实现。CSA长−1.090%、短+0.859%，8:2−0.700%；
短档CSA/P95回退单列，[证据](results/csa_hc_post_resident_20260929/README.md)。
O_A/O_B各64份工作由24个AIC多波处理，Q_B/Indexer投影也有资源交叠，不能把所有启动分散都删成收益。

## 下一步调度与未关闭项

1. 当前核内已保留Sparse末块发布、HC_post残差常驻、长S6完整query单根；本页七档均已覆盖。
   单根方案同卡局部A/B的长B16 Score AIC/AIV为−8.555%/−9.484%、CSA−4.390%；
   短B24 CSA+2.941%，8:2−2.924%。短档未改代码且核内范围重叠，不归因单根优化。
   [局部因果对照及边界](results/csa_score_single_root_20260929/README.md)。
2. 下一轮按这份任务图研究Indexer归并、AIV数据交接与CSA关键链，同时保留长小batch Score和短Sparse入口。
   128K优先、长短8:2，有真实核内收益即保留；明显场景分化在同一套算子内分支。
   后续已保留[O-B激活L1复用/K128双缓冲](results/csa_ob_activation_l1_k128_20260929/README.md)：
   长/短B16核内−5.720%/+5.852%，8:2−3.406%，完整CSA8:2−1.296%，两档及两条尾行路径状态通过。
   这组同卡A/B为auto设备9，与上方七档设备0的不同轮次数据分开；不将其增益外推更新七档表。
3. HC门控优先已否定：CSA 8:2回退1.380%，独立DFX前段提前不能代替正式CSA收益；
   不原样重试HC widen开放预派发、短Score/Sparse整组准入、quant关early及纯去dummy组合。
4. 对照[上游725μs泳道](results/csa_scheduling_20260927/upstream_725/README.md)，其输入FP32、无BF16 widen，
   缺完整版本及Scheduler View，只作组织参考。分开producer end→FIN、FIN→dispatch、dispatch→start与必要多波；
   无物理时戳的dummy不作完整ready归因，不把启动分散全部当成软件调度开销。
   [收尾预派发复核](results/csa_single_root_seven_20260929/TAIL_SCHEDULE_REVIEW.md)表明，
   固定window_3的长B16/B24、短B24中proj_b_act/hc_post在全部前置FIN后0.58–0.84μs即启动；
   其较大的setup包含必要输入等待，当前不据此盲目关闭early。
   [整行标量收尾融合](results/csa_ob_hc_scalar_fused_20260929/RESULTS.md)保留BF16边界及组/HC相加顺序：
   长B16/短B24总核内工作量−10.970%/−8.642%，跨度−24.234%/−20.993%，
   CSA−1.092%/−0.384%、P95均下降；长短8:2核内−10.504%、CSA−0.951%。
   八类跨版本状态、16窗官方覆盖、尾行/padding与共享HC入口通过，已移入生产；两档设备1，不混入上方设备0七档。
   融合任务的最后前置FIN→首start仍不足1μs。此项改变核内分工、移除GM交接和一级任务，
   不表述为关闭early或消除长ready等待的收益；首版T8/N512多行融合已因核内+144.494%否定。
5. 当前精度版迁移、CANN9.2/新B24真实EP16 token/DSpark与10步forward未完成且后置。
   单根改变同分分段规则，已测状态零差异不能覆盖任意输入；异常P95不能用均值收益抵消。

[有效策略与失败记录](DSV4_FLASH_CSA_VALIDATION_LOG.md)、[最新AscendC参考](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)。
