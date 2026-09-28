# CSA：当前七档核内差距与优化顺序

更新：2026-09-29。当前完整基线为 **c93ec723 / CANN9.2**，同一个auto单卡任务完成
128K B4/B8/B16/B24、8K B24/B32/B40。Native/PTO均走真实编译半层；
Native启用npugraph_ex、static kernel与norm/quant融合，详见[配置及边界](DSV4_FLASH_CSA_NATIVE_BASELINE.md)。
旧e33d842a手工NPUGraph矩阵及局部A/B移至[验证日志](DSV4_FLASH_CSA_VALIDATION_LOG.md)，不再作为当前表格。

## 范围和读数

正式layer4（第二个CSA）权重、独立合成历史/输入、S6/mode2/det0，PTO atomic0，EPLB关闭。
Native原cache布局不改，PTO直接分页读写，无入口拆分和外部写回。WO_A借用Native NZ29原地址。
每侧5次预热、20次无profiler图事件；完整区间含HC_pre、norm、CSA、HC_post。
两侧PyTorch profile独立采集；PTO每档另采四个level-4窗口，不以DFX时长替代正式CSA。
Native QLI/Sparse仍使用release custom二进制；最新ops-transformer28f40354、ops-nn19614968、
ops-math361722c0是优化源码参考，不冒称已重建并测量其最新kernel。

Native PMU按block/波次折算；PTO kernel-duration含DMA与内部等待，且两侧融合范围不同。
下表用于定位差距，不能直接相减计算可回收时长，或据此宣称严格的纯算术加速比。
[本轮完整证据](results/csa_compiled_seven_20260929/RESULTS.md)、
[21份原始JSON下载目录](results/csa_compiled_seven_20260929/download/README.md)。

## 完整CSA

| 档位 | Native均值μs | PTO均值μs | PTO变化 | PTO P95μs | PTO最大值μs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K/B4 | 787.465 | 701.483 | -10.919% | 715.980 | 716.900 |
| 128K/B8 | 938.911 | 802.377 | -14.542% | 814.040 | 817.440 |
| 128K/B16 | 1232.308 | 1043.478 | -15.323% | 1061.640 | 1062.200 |
| 128K/B24 | 1403.680 | 1309.122 | -6.736% | 1328.000 | 1341.000 |
| 8K/B24 | 1052.183 | 985.264 | -6.360% | 1012.320 | 1012.900 |
| 8K/B32 | 1180.726 | 1137.006 | -3.703% | 1168.860 | 1181.840 |
| 8K/B40 | 1324.654 | 1297.427 | -2.055% | 1325.660 | 1339.720 |

长档均值变化-11.880%，短档-4.039%，
各上下文内batch等权、再按8:2加权为 **-10.312%**。
七档P95/P50为1.0127–1.0341，本轮没有大拖尾；8K/B40的PTO最大值1339.720μs仍高于Native1330.740μs。
这些20次采样不能关闭历史约1.4ms的间歇长尾，也不替代EP16模型稳定性或token/DSpark验收。

## Indexer

Native QLI融合系数生成、Score、本地Top-K及最终归并；PTO仍独立生成系数和执行merge。单位μs。

| 档位 | Native QLI Duration | Native AIC/AIV参考 | PTO Score AIC/AIV | PTO merge |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 238.440 | 69.927/69.511 | 76.619/86.833 | 6.897 |
| 128K/B8 | 237.340 | 125.406/125.138 | 135.495/142.600 | 9.198 |
| 128K/B16 | 364.160 | 248.899/248.300 | 254.344/260.119 | 13.511 |
| 128K/B24 | 380.240 | 370.853/370.460 | 346.587/363.192 | 13.815 |
| 8K/B24 | 55.780 | 52.249/51.931 | 27.350/41.749 | 11.574 |
| 8K/B32 | 95.280 | 68.515/68.027 | 80.102/84.484 | 15.994 |
| 8K/B40 | 92.040 | 76.668/76.068 | 49.400/64.105 | 15.537 |

- 长B4/B8采用S6复用后，AIC已接近Native PMU参考；AIV仍分别为86.833/142.600μs，另有独立归并。
- 长B16为254.344/260.119μs，Native参考248.899/248.300μs；继续看AIV数据交接、排序和分片根。
- 长B24 Score读数低于Native融合QLI参考，但仍有系数、scale提交、merge及它们的依赖，不能只看Score。
- 短B32仍走双query，Score高于Native参考；以20%权重约束回退，必要时在同一算子内按场景分支。

[最新AscendC和pypto-lib的逐项实现差异](DSV4_FLASH_CSA_INDEXER_NATIVE_GAP.md)。

## Sparse attention

Native包含内部规约；PTO另有merge_norm，不将两列简单相加减求精确加速比。单位μs。

| 档位 | Native Duration | Native AIC/AIV参考 | PTO QK/PV AIC/AIV | PTO merge_norm |
| --- | ---: | ---: | ---: | ---: |
| 128K/B4 | 57.660 | 50.654/53.187 | 48.443/50.328 | 15.637 |
| 128K/B8 | 94.980 | 88.729/92.545 | 80.978/82.904 | 20.515 |
| 128K/B16 | 181.580 | 172.805/176.115 | 152.931/154.882 | 25.238 |
| 128K/B24 | 238.680 | 228.556/231.094 | 212.034/213.900 | 31.362 |
| 8K/B24 | 173.360 | 161.609/164.300 | 178.482/180.312 | 34.163 |
| 8K/B32 | 233.880 | 219.917/222.550 | 227.092/229.101 | 40.304 |
| 8K/B40 | 289.380 | 278.969/281.398 | 277.485/279.458 | 45.903 |

长B16/B24的QK/PV核时低于Native参考，但独立merge_norm仍需25.238/31.362μs。
短B24/B32的QK/PV偏高，且另有34.163/40.304μs归并，是短档优势收窄的待研究项。
已有PV N128、交替累加缓冲与跨query流水保留；新策略仍优先证明128K收益，不原样重试已否定联合softmax。

## 其余任务与O projection

128K/B16四窗口均值，单位μs；所有七档完整任务及物理核分配保留在本轮evidence.json。

| Task | block数 | 核内均值 | 最慢核 | 包络 | 启动分散 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hc_pre_linear | 24 | 7.585 | 9.385 | 12.050 | 4.865 |
| mix_x_rms_norm | 12 | 17.290 | 18.190 | 19.030 | 0.870 |
| qr_proj_matmul | 24 | 16.336 | 17.610 | 38.100 | 0.445 |
| kv_proj_matmul | 12 | 22.192 | 24.605 | 29.305 | 7.005 |
| qproj_matmul | 24 | 43.870 | 54.110 | 92.745 | 59.385 |
| idx_qr_proj_matmul | 24 | 16.599 | 33.610 | 63.830 | 44.635 |
| indexer_head_coefficients | 48 | 3.722 | 6.760 | 19.680 | 11.500 |
| idx_kv_scale_commit | 1 | 8.615 | 8.615 | 9.160 | 0.000 |
| proj_a_mm | 64 | 27.230 | 31.790 | 86.940 | 61.345 |
| quant | 24 | 8.444 | 11.010 | 97.595 | 57.495 |
| proj_b_mm | 64 | 11.250 | 16.090 | 58.840 | 47.895 |
| hc_post | 24 | 21.338 | 22.895 | 41.600 | 0.635 |

O_A/O_B各64份工作由24个AIC执行，需要多波。O_A的61.345μs启动分散含必要执行，不能全称调度空隙。
同轮Native TransposeBatchMatMul Duration88.880μs、AIC参考72.118μs；PTO包络86.940μs。
当前WO_A已是NZ29原地址，旧ND转换造成的差距不再作为现状。

## 当前调度证据与下一步

七档28个level-4窗口已核对原始/合并行数及每任务block数。固定window_3展示Observed路径，
Static CPM只作交叉检查；dummy缺少物理时戳时不作完整ready归因。
[长B16路径](results/csa_compiled_seven_20260929/h131072_b16/schedule/README.md)、
[短B24路径](results/csa_compiled_seven_20260929/h8192_b24/schedule/README.md)。

1. 仅删除系数空worker的私有候选已完整CPU编译/load：上限min(48,组数)，内部stride48不变。
   长B16/短B24先检验实际提交数48→16/24、Score首次启动、CSA均值及P95，再决定受影响范围覆盖。
   [候选与对照任务](results/csa_coefficient_active_workers_20260929/README.md)。
2. Score多数窗口已提前派发。长B16有些窗口最后等待系数，有些等待idx_kv_scale_commit；
   分开记录生产者end→FIN、FIN→派发和派发→开始，不将时序相关性称为资源阻塞因果。
   原cache的scale写回涉及64字节读改写，没有页面所有权证明前不得直接并行或删除依赖。
3. 长档8:2优先继续降低归并/数据交接和关键链等待；保留有证据的核内收益，CSA与P95分别判断。
   若长短明显相反，采用同一算子内的形状策略，不让调用者切历史版本。
4. 精度版保持Native舍入/规约规则，近期数值中性优化尚待迁移；CANN9.2/新B24的
   EP16逐token、DSpark和稳态10步forward验收仍未完成，不能由本轮单卡状态通过代替。

## 保留策略和已否定方向

- 已保留长B≥4的S6 Key复用、B<4双query、独立Key L1预取、均衡leaf、2560/3072尾排序、
  2048分段排序和UB中间根、四路Top-K，以及HC/QR/KV/Sparse已有核内优化。
  [S6实测](results/csa_small_long_s6_20260929/RESULTS.md)、[UB根](results/csa_stream_root_ub_20260929/RESULTS.md)。
- [矩阵scale广播](results/csa_score_scale_matrix_20260929/RESULTS.md)和
  [Query/系数跨leaf驻留](results/csa_query_resident_20260929/RESULTS.md)没有取得长档核内收益，未合入。
- [去除4个dummy](results/csa_direct_deps_20260929/RESULTS.md)仅8:2约−0.1%，未合入；
  短Score/Sparse整组准入、query整组准入等旧失败候选无新依据不重复测试。

[最新AscendC入口](DSV4_FLASH_CSA_ASCENDC_REFERENCES.md)、
[当前PTO源码](../../vllm_ascend/ops/pypto/deepseek_v4_flash_dspark_perf/decode_indexer.py)。
