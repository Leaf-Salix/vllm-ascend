# CSA：当前七档核内差距与优化顺序

更新：2026-09-29。当前完整实测为 **4ffccb7b / CANN9.2**，同一auto单卡任务已退出0。
正式七档为128K B4/B8/B16/B24、8K B16/B24/B32；B40退役。
已包含系数空worker删除、按组准备和UB一次发布；未合入否定的分数UB或无dummy准入组合。
Native本轮经vLLM Ascend编译包装进入npugraph_ex，static kernel开启、superkernel关闭，
force_eager后由外层捕获。用户新要求的显式torch.compile/backend=npugraph_ex和后端自行捕获另做对照，
两代表档已完成superkernel选型和直接设备replay校准：长B16 1122.652μs、短B24 940.762μs。
该校准轮inplace_pass关闭；当前要求固定dynamic=False/inplace_pass=True，
[新完整七档](results/csa_native_inplace_seven_20260929/README.md)正在运行，不能沿用旧配置数字作新基线。
后续Native开启superkernel；下表仍属旧入口，不修改配置或拼接两行成新版七档。
整体比较开SuperKernel；细化核内任务时另采关SuperKernel、static compile仍开的profile。
生产性能版现已保留Sparse末块发布，两代表档及边界通过；下表尚未包含该优化，阶段出口统一重取七档。
短档PTO领先旧Native的结论不能外推到新基线，阶段出口统一更新。
[新Native口径](results/csa_native_graph_replay_20260929/RESULTS.md)。旧表与过程证据保留在Git和验证日志。

## 范围和读数

正式layer4（第二个CSA）权重、独立合成历史、S6/mode2/det0，PTO atomic0，EPLB关闭。
Native cache布局不改，PTO内分页读写，无入口拆分或外部写回；WO_A借用Native NZ29原地址。
各5预热/20次无profiler图事件；完整区间含HC_pre、norm、CSA、HC_post。
两侧PyTorch profile独立采集，PTO每档另有四个level-4窗口。Native PMU与PTO Worker核时
测量边界不同，且融合范围不同，不能相减作为可回收时长或严格纯算术加速比。
Native仍为release custom二进制；ops-transformer28f40354、ops-nn19614968、ops-math361722c0
是最新本地优化源码参考，未冒称重建了这些仓的最新kernel。
[完整实测](results/csa_coefficients_seven_20260929/RESULTS.md)、
[21份原始JSON](results/csa_coefficients_seven_20260929/download/README.md)、
[Native配置边界](DSV4_FLASH_CSA_NATIVE_BASELINE.md)。

## 完整CSA

| 档位 | Native均值μs | PTO均值μs | PTO变化 | PTO P95μs | PTO最大值μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 794.933 | 674.644 | -15.132% | 683.700 | 688.700 |
| 128K/B8 | 945.401 | 815.204 | -13.772% | 828.580 | 831.320 |
| 128K/B16 | 1225.289 | 1052.786 | -14.079% | 1070.600 | 1076.980 |
| 128K/B24 | 1399.230 | 1318.600 | -5.762% | 1338.240 | 1347.920 |
| 8K/B16 | 845.610 | 788.918 | -6.704% | 813.860 | 814.260 |
| 8K/B24 | 1018.715 | 978.402 | -3.957% | 995.500 | 1009.640 |
| 8K/B32 | 1190.600 | 1120.371 | -5.899% | 1148.740 | 1167.060 |

长档变化-12.186%，短档-5.520%，各上下文内batch等权后8:2为 **-10.853%**。
本轮PTO P95/P50为1.0143–1.0292，max/P50最高1.0456，各档0/20样本超过各自P50的105%。
这20次采样不关闭历史约1.4ms间歇长尾，也不替代EP16、逐token或DSpark验收。

## Indexer

Native QLI含系数、Score和归并；PTO系数及merge独立。单位μs。

| 档位 | Native QLI Duration | Native AIC/AIV参考 | PTO Score AIC/AIV | PTO系数 | PTO merge |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 241.040 | 70.917/72.050 | 82.623/92.695 | 1.862 | 7.763 |
| 128K/B8 | 242.340 | 128.450/128.875 | 150.599/157.776 | 2.182 | 8.993 |
| 128K/B16 | 359.440 | 243.242/242.668 | 248.311/254.206 | 3.076 | 14.242 |
| 128K/B24 | 381.560 | 371.840/371.446 | 351.067/367.871 | 2.900 | 15.365 |
| 8K/B16 | 51.800 | 36.758/36.161 | 45.406/49.648 | 3.731 | 8.410 |
| 8K/B24 | 53.960 | 50.341/50.476 | 34.371/50.415 | 2.147 | 9.797 |
| 8K/B32 | 91.840 | 64.248/63.654 | 83.520/88.042 | 3.963 | 11.255 |

- 长B4/B8仍有明显Score AIV差距，PTO为92.695/157.776μs，Native参考72.050/128.875μs；继续看AIV分工、数据交接与排序。
- 长B16为248.311/254.206μs，接近Native参考243.242/242.668μs，但另有14.242μs独立merge。
- 长B24 Score低于Native融合QLI参考，独立系数、scale提交及15.365μs的merge仍在，不能只看Score。
- 短B16/B32走双query，AIV为49.648/88.042μs，Native参考36.161/63.654μs；短档只占20%，明显回退时应在算子内分策略。

[最新AscendC及pypto-lib实现差异](DSV4_FLASH_CSA_INDEXER_NATIVE_GAP.md)。

## Sparse attention

Native含内部规约；PTO另有merge_norm，两列不简单相加减归因。单位μs。

| 档位 | Native Duration | Native AIC/AIV参考 | PTO QK/PV AIC/AIV | PTO merge_norm |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 56.500 | 47.970/50.592 | 47.881/49.646 | 12.656 |
| 128K/B8 | 105.440 | 96.839/100.883 | 79.072/81.120 | 15.434 |
| 128K/B16 | 179.200 | 172.146/174.655 | 164.168/166.094 | 24.772 |
| 128K/B24 | 233.100 | 220.047/222.646 | 215.591/217.437 | 33.084 |
| 8K/B16 | 117.400 | 108.997/111.577 | 124.163/126.106 | 23.603 |
| 8K/B24 | 169.540 | 159.496/161.902 | 188.631/190.362 | 29.470 |
| 8K/B32 | 234.900 | 222.747/225.306 | 230.050/231.848 | 38.845 |

长B16/B24 QK/PV低于Native参考，但仍有24.772/33.084μs独立merge_norm。
短B16/B24 QK/PV高于Native参考；已有PV N128和跨query流水保留，后续以长档为主评估，
不原样重试已否定的联合softmax。

## 其余任务与O projection

128K/B16四窗口均值，单位μs。启动分散包含必要多波执行，不能全当调度空隙。

| Task | block数 | 核内均值 | 最慢核 | 包络 | 启动分散 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hc_pre_linear | 24 | 7.624 | 9.620 | 12.220 | 5.755 |
| mix_x_rms_norm | 12 | 16.990 | 17.700 | 18.440 | 0.340 |
| qr_proj_matmul | 24 | 14.007 | 15.290 | 35.790 | 0.425 |
| kv_proj_matmul | 12 | 24.619 | 27.155 | 30.115 | 7.145 |
| qproj_matmul | 24 | 43.031 | 52.540 | 102.005 | 67.025 |
| idx_qr_proj_matmul | 24 | 17.560 | 34.515 | 71.685 | 58.125 |
| indexer_head_coefficients | 16 | 3.076 | 4.290 | 12.705 | 0.360 |
| idx_kv_scale_commit | 1 | 8.255 | 8.255 | 8.785 | 0.000 |
| proj_a_mm | 64 | 27.358 | 32.015 | 86.420 | 61.985 |
| quant | 24 | 8.454 | 11.645 | 96.730 | 56.995 |
| proj_b_mm | 64 | 10.951 | 15.440 | 57.650 | 46.425 |
| hc_post | 24 | 20.265 | 22.290 | 38.080 | 0.625 |

O_A/O_B各64份工作由24个AIC执行，必须多波；WO_A已经直接借用NZ29，旧ND转换不再是现状。

## 下一步

1. Native显式npugraph_ex的superkernel和设备replay口径均已完成校准，后续固定开启。
   长B16/短B24分别1122.652/940.762μs；阶段出口统一更新七档，不再试开关，也不替换旧表中的两行拼成新表。
   新profile的QLI→Sparse融合SuperKernel为516.580/203.380μs，AIC参考510.308/193.084μs、AIV参考512.672/195.446μs。
   这些是整段融合范围，不能作为单个QLI或Sparse PMU，也不能与各旧核均值直接相减归因。
2. Sparse最后PV块直接归一化、逆RoPE及发布已保留：两代表档CSA 8:2−2.667%、
   AIV总工作量−11.551%，完整状态与B3/H127/padding精确通过；两档P95下降。
   首版后16-head统计量切片偏移丢失的失败证据仍保留为失败，显式ND抽取修复后才采用。
   [已保留结果](results/csa_sparse_final_publish_fixed_pair_20260929/README.md)、
   [已否定的AIV分工](results/csa_score_query_split_20260929/RESULTS.md)。
3. 按28个level-4窗口区分producer end→FIN、FIN→dispatch和dispatch→start；
   固定window_3，dummy无物理时戳时不作完整ready归因。Score提前派发不等于资源已就绪。
   [长B16](results/csa_coefficients_seven_20260929/h131072_b16/schedule/README.md)、
   [短B24](results/csa_coefficients_seven_20260929/h8192_b24/schedule/README.md)。
4. Sparse融合的核内与CSA/P95结果分别判定；Indexer独立merge及scale依赖保留后续，
   scale写回涉及64字节读改写，没有页面所有权证明不删依赖。
   AscendC式首PV特化已完成并否定：零容差状态通过，但Sparse AIV长/短+5.075%/+8.726%，
   CSA与P95也回退；[结果](results/csa_sparse_first_pv_20260929/RESULTS.md)，不原样重试。
   当前独立验证[HC_post残差常驻UB](results/csa_hc_post_resident_20260929/README.md)，保持乘加顺序及任务分工。
5. 精度版数值中性优化迁移、CANN9.2/新B24真实EP16 token/DSpark与稳态10步forward仍未完成。

## 保留策略和已否定方向

- 已保留长B≥4的S6 Key复用、B<4双query、独立Key L1预取、均衡leaf、2560/3072尾排序、
  2048分段排序和UB中间根、四路Top-K，以及HC/QR/KV/Sparse已有核内优化。
  [S6实测](results/csa_small_long_s6_20260929/RESULTS.md)、[UB根](results/csa_stream_root_ub_20260929/RESULTS.md)。
- [矩阵scale广播](results/csa_score_scale_matrix_20260929/RESULTS.md)和
  [Query/系数跨leaf驻留](results/csa_query_resident_20260929/RESULTS.md)没有取得长档核内收益，未合入。
- [去除4个dummy](results/csa_direct_deps_20260929/RESULTS.md)仅8:2约−0.1%，未合入；
  短Score/Sparse整组准入、query整组准入等旧失败候选无新依据不重复测试。
- [系数直接融合Score](results/csa_coefficient_fused_20260929/README.md)状态精确，但两档CSA均回退，
  8:2为+2.373%；长Score提前19.220μs启动，AIC/AIV核时却各增加约36μs，未合入。
  后续独立系数任务的批量加载/乘法已保留；该直接融合版本不因此转为有效优化。

[最新AscendC入口](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)、
[当前PTO源码](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)。
