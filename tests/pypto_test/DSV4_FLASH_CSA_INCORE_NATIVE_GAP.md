# CSA：最新七档核内差距与下一步

更新：2026-09-29。Native最新标准与PTO性能版55b89ee2的完整七档任务已退出0；
128K B4/B8/B16/B24、8K B16/B24/B32，同一auto设备0。旧入口数字已移出本页，历史见Git及验证日志。
生产之后已保留d93bba14的HC_post残差常驻；下面七档没有该项，不推算其收益。

Native整体：CANN9.2、显式torch.compile backend=npugraph_ex、dynamic=False/fullgraph=True、
inplace_pass=True、static compile和SuperKernel开启，由后端图直接replay。
Native核内诊断只关闭SuperKernel，其他配置保持；诊断Duration不混入正式计时。
PTO保留custom-op图边界及自己的PyPTO实现；以后直接使用已有验证数据，不强制套用Native编译选项。

真实第二个CSA层权重、独立合成历史，mode2/det0、PTO atomic0、EPLB关闭。
完整HC_pre+norm+CSA+HC_post，5预热20次设备事件；各侧独立PyTorch profile、每档四窗DFX。
Native cache不改，PTO直接分页读写，WO_A借用Native NZ29原地址。
Native仍为release custom二进制，最新ops-transformer28f40354/ops-nn19614968/ops-math361722c0只是源码参考。
[完整结果](results/csa_native_inplace_seven_20260929/RESULTS.md)、[任务与pipeline明细](results/csa_native_inplace_seven_20260929/TASKS.md)、[28份JSON下载](results/csa_native_inplace_seven_20260929/download/README.md)。

## 完整CSA

| 档位 | Native均值μs | PTO均值μs | PTO变化 | Native/PTO P95μs |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 748.298 | 647.687 | -13.445% | 751.100/661.300 |
| 128K/B8 | 859.563 | 769.604 | -10.466% | 861.620/780.920 |
| 128K/B16 | 1130.853 | 1028.167 | -9.080% | 1135.580/1041.720 |
| 128K/B24 | 1281.888 | 1267.099 | -1.154% | 1289.880/1289.120 |
| 8K/B16 | 757.482 | 765.047 | +0.999% | 760.120/791.700 |
| 8K/B24 | 915.371 | 942.503 | +2.964% | 920.340/976.820 |
| 8K/B32 | 1062.079 | 1072.650 | +0.995% | 1066.220/1092.280 |

各上下文内batch等权，128K -8.536%、8K +1.653%，8:2 -6.498%。
128K四档均领先，B24只领先1.15%；8K三档均值和P95均高于Native。
PTO P95/P50为1.0137–1.0400、max/P50最高1.0468；各档0/20超过P50的105%。
不删样本，也不以20次正常样本关闭历史间歇尾部或EP16问题。两侧自身图重放及保护区通过，
不能替代两侧逐元素、逐token或DSpark验收；本页不是整模型forward。

## Indexer核内

Native独立QLI包含系数、Score和归并；PTO系数/Score/merge独立，PMU与worker计时边界不同。
单位μs，PTO为四窗均值；差值仅用于定位候选，不是严格纯算术差或可回收时长。

| 档位 | Native QLI AIC/AIV | PTO Score AIC/AIV | PTO系数 | PTO merge |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 66.363/64.369 | 70.539/80.560 | 1.854 | 5.621 |
| 128K/B8 | 126.316/126.021 | 139.297/146.302 | 2.008 | 7.021 |
| 128K/B16 | 242.293/241.730 | 250.640/256.646 | 2.581 | 9.677 |
| 128K/B24 | 376.467/376.063 | 349.266/365.861 | 3.935 | 10.844 |
| 8K/B16 | 39.630/39.050 | 41.959/46.492 | 4.522 | 10.264 |
| 8K/B24 | 56.051/55.666 | 29.955/43.154 | 2.970 | 11.774 |
| 8K/B32 | 68.901/68.365 | 79.107/83.591 | 5.351 | 12.312 |

- 长B4/B8的Score AIV仍有明显参考差距（约16/20μs），不是所有长档都已追平；四路Top-K、UB根等已保留。
- 长B16的Score AIV比Native参考高约15μs；长B24已低于Native参考，但系数、merge和数据交接仍单独存在。
- 短B16/B32双query的Score高于Native参考；短B24的Score较短，不能由此认定完整Indexer已更快。
- 新Native主性能中QLI→Sparse融合为SuperKernel，其273–585μs长档AIC范围不能拆成某个单算子核时。
[算术、任务与输入差异](DSV4_FLASH_CSA_INDEXER_NATIVE_GAP.md)。

## Sparse attention

PTO已融合最终归一化、逆RoPE和发布，没有独立merge_norm；Native此处Sparse不含独立逆RoPE。

| 档位 | Native Sparse AIC/AIV | PTO融合Sparse AIC/AIV |
| --- | ---: | ---: |
| 128K/B4 | 53.214/57.090 | 43.906/48.862 |
| 128K/B8 | 95.667/99.998 | 79.808/83.645 |
| 128K/B16 | 167.255/169.662 | 154.008/157.802 |
| 128K/B24 | 221.309/223.867 | 228.936/232.852 |
| 8K/B16 | 95.677/98.084 | 129.069/132.855 |
| 8K/B24 | 166.763/169.371 | 181.019/184.865 |
| 8K/B32 | 210.209/212.812 | 237.366/241.178 |

长B4/B8/B16的PTO融合Sparse低于Native参考；B24略高约9μs且包含额外逆RoPE，不能机械认定纯计算退化。
短档AIV高于Native约15–35μs，尤其B16，仍是明显差异。核时含数据搬运及核内流水等待，
Native各pipeline计数可重叠，不能把MAC/MTE/FIX/Vector计数相加作总时长。

## HC与Q/O任务组织

下面为128K/B16、B24的四窗均值。启动分散包括必要多波，不能全算调度器软件开销。

| Task | B16 worker/核时μs/启动分散μs | B24 worker/核时μs/启动分散μs |
| --- | ---: | ---: |
| hc_widen_rms | 12/12.854/0.995 | 18/13.072/4.395 |
| hc_pre_linear | 24/7.157/6.710 | 24/9.876/6.580 |
| hc_pre_linear_reduce | 6/1.695/0.305 | 9/2.052/0.240 |
| split_pre_post | 12/3.935/1.020 | 16/4.580/0.335 |
| comb_sinkhorn | 12/15.493/0.280 | 18/15.284/3.540 |
| mix_x_rms_norm | 12/17.569/0.400 | 18/17.741/2.600 |
| qproj_matmul | 24/37.556/58.055 | 24/47.896/62.815 |
| qproj_dequant_rms_nope_rope | 48/19.927/21.525 | 48/30.807/40.400 |
| proj_a_mm | 64/27.010/62.220 | 64/27.846/71.205 |
| quant | 24/7.741/51.310 | 40/7.223/62.700 |
| proj_b_mm | 64/11.029/50.155 | 64/18.832/64.945 |
| hc_post | 24/19.783/0.590 | 36/24.118/0.635 |

Native HC_post按不同token分工，而PTO每worker最多4个token，不能只比单worker均值。
已采用的残差常驻优化使两代表档HC_post核内−17.423%/−24.785%，四窗范围不重叠；
八类状态及B3/H127/padding通过，共享精度/性能实现。CSA长−1.090%、短+0.859%，8:2−0.700%；
短档CSA/P95回退单列，[证据](results/csa_hc_post_resident_20260929/README.md)。
O_A/O_B各64份工作由24个AIC多波处理，Q_B/Indexer投影也有资源交叠，不能把所有启动分散都删成收益。

## 下一步调度与未关闭项

1. 本轮核内已保留Sparse末块发布、HC_post残差常驻，首PV特化及多项Score候选无收益的证据保留，停止原样重试。
   当前开始按这份七档的新任务图做调度候选，保留后续长B4/B8 Score和短Sparse的核内研究入口。
2. HC关键门控优先假设已经完成对照并否定：给comb增加pre/post前置后，
   长B16完整CSA+1.891%、短B24−0.665%，8:2回退1.380%，长P95增加24.440μs。
   独立DFX的mix完成虽提前长4.360/短3.890μs，但不能解释为正式CSA收益；八类状态/保护区通过，
   实际图确认唯一新增依赖。生产HC_pre不变，不扩大测试，[结果](results/csa_hc_pre_priority_20260929/README.md)。
   下一项[完整query单根策略](results/csa_score_single_root_20260929/README.md)已完成两档：
   连续S2、UB内2048分段排序与单根consumer一起调整，短档保持；不重复仅重排query而保留两个half根的失败方案。
   长B16 Score AIC/AIV−8.555%/−9.484%，四窗范围不重叠；CSA长−4.390%、短+2.941%、8:2−2.924%。
   八类状态精确通过，正在唯一B4/H65535尾段/padding边界验证。短档核内范围重叠且算法未改，不归因新优化；
   同分顺序的潜在变化仍按独立误差/token合同处理。
3. 对照[上游725μs泳道](results/csa_scheduling_20260927/upstream_725/README.md)：其输入FP32、无BF16 widen，
   缺完整版本及Scheduler View，只作组织参考。上游pre/post亦可能晚于comb，不能声称新约束是照抄上游标志。
   Native在一个HcPre核内完成门控；本接入拆分后必须验证哪条任务交接影响attention关键链。
4. 继续分开producer end→FIN、FIN→dispatch、dispatch→start与必要多波；无物理时戳的dummy不作完整ready归因。
   不重试无新依据的HC widen开放预派发、短Score/Sparse整组准入、quant关early及纯去dummy组合。
5. 最新精度版数值中性优化迁移、CANN9.2/新B24真实EP16 token/DSpark与10步forward仍后置且未完成。
   长短8:2决定取舍，明显场景分化在同一算子内分支；功能和异常P95不能用均值抵消。

[有效策略与失败记录](DSV4_FLASH_CSA_VALIDATION_LOG.md)、[最新AscendC参考](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)。
