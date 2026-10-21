# 本轮设备证据汇总

由 `summarize_results.py` 从原始结果生成。只有queue exit 0、stage complete、
passed true同时满足才记整项通过；原始FAIL、异常及局部通过证据保留。
基线列区分旧raw TWAIT回归与另一会话已采用的BSH150。候选试验按原始目录
分别保留；不能把不同源码或轮次的最快值拼成八档。只有同一冻结候选完整
八档经 `evaluate_s6_cohort.py` 核验，才能判定是否满足几何平均<=0.90。
当前正式参考更新为a241fa61；旧基线数据保留版本标记。
8K未启用B1分支的静态形状修复单独标记；原始源码和编译FAIL保留。
B1 G6/24叶方案的v31随机输入收益已被v35同分TopK/输出回归否决，不能采用。
等待前四次快轮询属于测试源码改写诊断，不能用于最终八档性能验收。

## 最新完整八档

a241fa61 BSH静态形状修复+150tick；候选为输出分工v4+短前缀TopK+短G2半叶UB。两个改变档位新测，六个完整C++/Host相同档位保留原成对五窗口；无跨轮择优。已采用，整体10%目标仍未达到。
完整证据：[cohort_borrowed_combined_v106_result.json](cohort_borrowed_combined_v106_result.json)。
等权GM(TND/BSH)=0.988089028，门槛0.90，
性能验收为FAIL（未达标）。后续单项试验不替换本轮任一档。

| 场景 | BSH µs | TND µs | 耗时变化 |
| --- | ---: | ---: | ---: |
| 128K/B1/S6 | 371.62 | 351.96 | -5.29% |
| 128K/B4/S6 | 492.26 | 483.16 | -1.85% |
| 128K/B8/S6 | 622.28 | 617.16 | -0.82% |
| 128K/B16/S6 | 791.40 | 780.18 | -1.42% |
| 128K/B24/S6 | 1008.46 | 1019.12 | +1.06% |
| 128K/B28/S6 | 1162.84 | 1156.10 | -0.58% |
| 8K/B16/S6 | 641.12 | 640.38 | -0.12% |
| 8K/B32/S6 | 920.58 | 917.04 | -0.38% |

## 完整L1性能

五个轮换窗口的中位数，单位µs。百分比为TND相对原单SPMD的耗时变化；
正值更慢。计全72核首尾，包含等待和阶段间隙。多task只参与其支持的S6。
标注 incremental comparison 的行以已采用 TND 为参照，只判断候选增量，不能用于 BSH 八档验收。

| 场景 | 基线 | 原SPMD | TND | 变化 | 多task | 本任务功能/采样 | 原始结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 128K/B4/S6 | raw TWAIT | 664.80 | 677.80 | +1.96% | 508.90 | PASS | [结果](runs/l1_s6_h131072_b4_raw/result.json) |
| 128K/B8/S6 | raw TWAIT | 846.02 | 822.16 | -2.82% | 639.78 | PASS | [结果](runs/l1_s6_h131072_b8_raw/result.json) |
| 128K/B16/S6 | raw TWAIT | 1028.76 | 1012.30 | -1.60% | 831.50 | PASS | [结果](runs/l1_s6_h131072_b16_raw/result.json) |
| 128K/B24/S6 | raw TWAIT | 1228.24 | 1224.40 | -0.31% | 1070.78 | PASS | [结果](runs/l1_s6_h131072_b24_raw/result.json) |
| 128K/B28/S6 | raw TWAIT | 1316.44 | 1348.48 | +2.43% | 1213.20 | PASS | [结果](runs/l1_s6_h131072_b28_raw/result.json) |
| 8K/B16/S6 | raw TWAIT | 839.46 | 838.90 | -0.07% | 679.44 | PASS | [结果](runs/l1_s6_h8192_b16_raw/result.json) |
| 8K/B32/S6 | raw TWAIT | 1074.62 | 1058.32 | -1.52% | 946.02 | PASS | [结果](runs/l1_s6_h8192_b32_raw/result.json) |
| mixed [2, 6, 1] / 8K 独立工作区 | raw TWAIT | 564.94 | 634.48 | +12.31% | 不适用 | PASS | [结果](runs/mixed_261_h8192_l1_raw_v14/result.json) |
| 128K/B1/S6 | raw TWAIT | 690.58 | 738.44 | +6.93% | 426.24 | PASS | [结果](runs/l1_s6_h131072_b1_raw/result.json) |
| mixed [2, 6, 1] / 8K 共享工作区归因 | raw TWAIT | 592.60 | 579.36 | -2.23% | 不适用 | PASS | [结果](runs/mixed_261_h8192_l1_shared_raw_v18/result.json) |
| 128K/B1/S6 G6试验（未采用） | raw TWAIT | 696.26 | 759.62 | +9.10% | 421.06 | PASS | [结果](runs/l1_s6_h131072_b1_g6_candidate_raw/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@683a701f | 400.04 | 404.72 | +1.17% | 不适用 | PASS | [结果](runs/cube_tile_publish_b1_h131072_bsh150_v21/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@683a701f | 778.62 | 786.24 | +0.98% | 不适用 | PASS | [结果](runs/cube_tile_publish_b16_h131072_bsh150_v21/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@683a701f | 401.12 | 390.44 | -2.66% | 不适用 | PASS | [结果](runs/idle_phase_publish_b1_h131072_bsh150_v22/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@683a701f | 787.02 | 796.76 | +1.24% | 不适用 | PASS | [结果](runs/idle_phase_publish_b16_h131072_bsh150_v22/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@683a701f | 397.90 | 379.10 | -4.72% | 不适用 | PASS | [结果](runs/active_prefix_wait_b1_h131072_bsh150_v23/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@683a701f | 780.02 | 781.86 | +0.24% | 不适用 | PASS | [结果](runs/active_prefix_wait_b16_h131072_bsh150_v23/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@683a701f | 785.02 | 791.92 | +0.88% | 不适用 | PASS | [结果](runs/native_wait_b16_h131072_bsh150_v28/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@683a701f | 780.42 | 777.20 | -0.41% | 不适用 | PASS | [结果](runs/static_capacity_native_b16_h131072_bsh150_v28/result.json) |
| 128K/B28/S6 独立工作区 | BSH150@683a701f | 1159.08 | 1147.42 | -1.01% | 不适用 | PASS | [结果](runs/static_capacity_native_b28_h131072_bsh150_v28/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@683a701f | 777.94 | 790.14 | +1.57% | 不适用 | PASS | [结果](runs/native_wait_b16_uninterrupted_sampling_v30/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@683a701f | 391.76 | 359.74 | -8.17% | 不适用 | PASS | [结果](runs/b1_balanced24_native_h131072_v31/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@683a701f | 780.52 | 789.94 | +1.21% | 不适用 | PASS | [结果](runs/batched_acquire_native_b16_v33/result.json) |
| 128K/B28/S6 独立工作区 | BSH150@683a701f | 1144.86 | 1155.74 | +0.95% | 不适用 | PASS | [结果](runs/batched_acquire_native_b28_v33/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61 | 386.78 | 393.96 | +1.86% | 不适用 | PASS | [结果](runs/post_columns_b1_a241_v34/result.json) |
| 128K/B4/S6 独立工作区 | BSH150@a241fa61 | 483.90 | 493.82 | +2.05% | 不适用 | PASS | [结果](runs/post_columns_b4_a241_v34/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61 | 365.82 | 388.50 | +6.20% | 不适用 | PASS | [结果](runs/initial_fast_polls_b1_a241_v36/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61 | 782.72 | 796.76 | +1.79% | 不适用 | PASS | [结果](runs/initial_fast_polls_b16_a241_v36/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61 | 367.74 | 346.84 | -5.68% | 不适用 | PASS | [结果](runs/active_prefix_b1_a241_v38/result.json) |
| 128K/B4/S6 独立工作区 | BSH150@a241fa61 | 487.90 | 491.84 | +0.81% | 不适用 | PASS | [结果](runs/active_prefix_b4_a241_v38/result.json) |
| 128K/B8/S6 独立工作区 | BSH150@a241fa61 | 627.36 | 611.48 | -2.53% | 不适用 | PASS | [结果](runs/active_prefix_b8_a241_v38/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61 | 788.26 | 845.08 | +7.21% | 不适用 | PASS | [结果](runs/score_wide_k128_b16_a241_v39/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61 | 363.28 | 357.18 | -1.68% | 不适用 | PASS | [结果](runs/cohort_prefix_a241_h131072_b1_s6_v41/result.json) |
| 128K/B4/S6 独立工作区 | BSH150@a241fa61 | 506.12 | 493.52 | -2.49% | 不适用 | PASS | [结果](runs/cohort_prefix_a241_h131072_b4_s6_v41/result.json) |
| 128K/B8/S6 独立工作区 | BSH150@a241fa61 | 617.16 | 620.40 | +0.52% | 不适用 | PASS | [结果](runs/cohort_prefix_a241_h131072_b8_s6_v41/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61 | 789.94 | 777.44 | -1.58% | 不适用 | PASS | [结果](runs/cohort_prefix_a241_h131072_b16_s6_v41/result.json) |
| 128K/B24/S6 独立工作区 | BSH150@a241fa61 | 1017.90 | 1013.46 | -0.44% | 不适用 | PASS | [结果](runs/cohort_prefix_a241_h131072_b24_s6_v41/result.json) |
| 128K/B28/S6 独立工作区 | BSH150@a241fa61 | 1153.56 | 1156.34 | +0.24% | 不适用 | PASS | [结果](runs/cohort_prefix_a241_h131072_b28_s6_v41/result.json) |
| 8K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 635.20 | 634.16 | -0.16% | 不适用 | PASS | [结果](runs/cohort_prefix_typefix_h8192_b16_s6_v47/result.json) |
| 8K/B32/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 915.52 | 918.32 | +0.31% | 不适用 | PASS | [结果](runs/cohort_prefix_typefix_h8192_b32_s6_v47/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 790.56 | 784.44 | -0.77% | 不适用 | PASS | [结果](runs/transfer_stride_pad64_h131072_b16_s6_v49/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 790.56 | 807.12 | +2.09% | 不适用 | PASS | [结果](runs/scale_owner_wait_h131072_b16_s6_v51/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 786.30 | 781.54 | -0.61% | 不适用 | PASS | [结果](runs/scale_owner_list_h131072_b16_s6_v53/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 360.38 | 358.26 | -0.59% | 不适用 | PASS | [结果](runs/scale_owner_list_b1_s6_and_s1_s6_v55/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 377.48 | 379.08 | +0.42% | 不适用 | PASS | [结果](runs/b1_oa_supply_s6_and_s1_s6_v59/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 787.32 | 814.48 | +3.45% | 不适用 | PASS | [结果](runs/score_threeway_h131072_b16_s6_v61/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 786.94 | 857.00 | +8.90% | 不适用 | PASS | [结果](runs/score_ws_pipeline_h131072_b16_s6_v63/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 782.78 | 795.86 | +1.67% | 不适用 | PASS | [结果](runs/score_visibility_h131072_b16_s6_v65/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 371.62 | 351.96 | -5.29% | 不适用 | PASS | [结果](runs/output_work_balance_b1_s6_and_s1_s6_v67/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 791.40 | 780.18 | -1.42% | 不适用 | PASS | [结果](runs/output_work_balance_h131072_b16_s6_v67/result.json) |
| 128K/B4/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 479.46 | 480.22 | +0.16% | 不适用 | PASS | [结果](runs/output_work_balance_h131072_b4_s6_v68/result.json) |
| 128K/B8/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 611.42 | 619.80 | +1.37% | 不适用 | PASS | [结果](runs/output_work_balance_h131072_b8_s6_v68/result.json) |
| 8K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 633.32 | 645.96 | +2.00% | 不适用 | PASS | [结果](runs/output_work_balance_guard_h8192_b16_s6_v70/result.json) |
| 128K/B11/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 704.08 | 695.04 | -1.28% | 不适用 | PASS | [结果](runs/output_balance_long_t66_boundary_v72/result.json) |
| 128K/B4/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 492.26 | 483.16 | -1.85% | 不适用 | PASS | [结果](runs/cohort_output_balance_h131072_b4_s6_v74/result.json) |
| 128K/B8/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 630.82 | 625.78 | -0.80% | 不适用 | PASS | [结果](runs/cohort_output_balance_h131072_b8_s6_v74/result.json) |
| 128K/B24/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 1008.46 | 1019.12 | +1.06% | 不适用 | PASS | [结果](runs/cohort_output_balance_h131072_b24_s6_v75/result.json) |
| 128K/B28/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 1162.84 | 1156.10 | -0.58% | 不适用 | PASS | [结果](runs/cohort_output_balance_h131072_b28_s6_v75/result.json) |
| 8K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 634.46 | 653.54 | +3.01% | 不适用 | PASS | [结果](runs/cohort_output_balance_h8192_b16_s6_v76/result.json) |
| 8K/B32/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 920.58 | 917.04 | -0.38% | 不适用 | PASS | [结果](runs/cohort_output_balance_h8192_b32_s6_v76/result.json) |
| 128K/B28/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 1147.02 | 1145.54 | -0.13% | 不适用 | PASS | [结果](runs/large_output_balance_h131072_b28_s6_v79/result.json) |
| 8K/B32/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 922.46 | 928.70 | +0.68% | 不适用 | PASS | [结果](runs/short_hc_consumer_h8192_b32_s6_v79/result.json) |
| 8K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 631.92 | 634.28 | +0.37% | 不适用 | PASS | [结果](runs/short_hc_consumer_h8192_b16_s6_v81/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 369.48 | 352.12 | -4.70% | 不适用 | PASS | [结果](runs/b1_output_prefix_s6_and_s1_s6_v81/result.json) |
| 128K/B24/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 1016.18 | 1015.68 | -0.05% | 不适用 | PASS | [结果](runs/large_output_balance_h131072_b24_s6_v82/result.json) |
| 128K/B1/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 367.30 | 361.62 | -1.55% | 不适用 | PASS | [结果](runs/single_m16_qa_b1_s6_and_s1_s6_v86/result.json) |
| mixed [6, 6, 4] / 128K 独立工作区 | BSH150@a241fa61+静态形状修复 | 499.24 | 498.02 | -0.24% | 不适用 | PASS | [结果](runs/single_m16_qa_t16_mixed664_v86/result.json) |
| 128K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 784.24 | 785.46 | +0.16% | 不适用 | PASS | [结果](runs/borrowed_score_stream_h131072_b16_s6_v90/result.json) |
| 128K/B28/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 1142.46 | 1160.02 | +1.54% | 不适用 | PASS | [结果](runs/borrowed_score_owner_h131072_b28_s6_v90/result.json) |
| 128K/B1/S6 独立工作区 | TND output_balance v4 (incremental comparison only) | 358.08 | 366.26 | +2.28% | 不适用 | PASS | [结果](runs/qa_incremental_b1_s6_v93b/result.json) |
| 128K/B16/S6 独立工作区 | TND output_balance v4 (incremental comparison only) | 785.02 | 788.22 | +0.41% | 不适用 | PASS | [结果](runs/ws_incremental_b16_s6_v93b/result.json) |
| 8K/B16/S6 独立工作区 | TND output_balance v4 (incremental comparison only) | 647.54 | 643.64 | -0.60% | 不适用 | PASS | [结果](runs/short_half_ub_h8192_b16_s6_v95/result.json) |
| 128K/B28/S6 独立工作区 | TND output_balance v4 (incremental comparison only) | 1151.84 | 1152.62 | +0.07% | 不适用 | PASS | [结果](runs/topk_prefix_h131072_b28_s6_v99/result.json) |
| 128K/B8/S6 独立工作区 | TND output_balance v4 (incremental comparison only) | 642.68 | 622.60 | -3.12% | 不适用 | PASS | [结果](runs/topk_prefix_h131072_b8_s6_v101/result.json) |
| 128K/B8/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 622.28 | 617.16 | -0.82% | 不适用 | PASS | [结果](runs/cohort_borrowed_combined_h131072_b8_s6_v106/result.json) |
| 8K/B16/S6 独立工作区 | BSH150@a241fa61+静态形状修复 | 641.12 | 640.38 | -0.12% | 不适用 | PASS | [结果](runs/cohort_borrowed_combined_h8192_b16_s6_v106/result.json) |

## 任务与原始证据

| 任务 | 用例 | 队列终态 | 结果阶段 | 整项通过 |
| --- | --- | --- | --- | --- |
| `task_20261020_234434_41806820769` | [mixed_261_h8192_v1](runs/mixed_261_h8192_v1/result.json) | completed (exit=1) | initializing | 否 |
| `task_20261020_234611_48699728663` | [mixed_261_h8192_v2](runs/mixed_261_h8192_v2/result.json) | completed (exit=1) | initializing | 否 |
| `task_20261020_234958_67318423765` | [mixed_261_h8192_v3](runs/mixed_261_h8192_v3/result.json) | completed (exit=1) | complete | 否 |
| `task_20261021_002804_315127125659` | [mixed_261_h8192_attribution_v4](runs/mixed_261_h8192_attribution_v4/result.json) | completed (exit=1) | complete | 否 |
| `task_20261021_003559_34885217752` | [mixed_261_h8192_service_stages_v5](runs/mixed_261_h8192_service_stages_v5/result.json) | completed (exit=1) | real_service_route | 否 |
| `task_20261021_004111_372169825135` | [mixed_261_h8192_service_v6](runs/mixed_261_h8192_service_v6/result.json) | completed (exit=1) | complete | 否 |
| `task_20261021_004754_404474221201` | [mixed_123456_capacity41_service_v7](runs/mixed_123456_capacity41_service_v7/result.json) | completed (exit=2) | complete | 否 |
| `task_20261021_005142_170195321` | [tnd_matrix_h8192_capacity41_v8](runs/tnd_matrix_h8192_capacity41_v8/result.json) | completed (exit=134) | tnd_metadata_and_continuous_state | 否 |
| `task_20261021_010447_7230338877` | [tnd_continuous_unpadded_native_v9](runs/tnd_continuous_unpadded_native_v9/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_010640_83025121361` | [mixed_history_1_to_64k_raw_service_v10](runs/mixed_history_1_to_64k_raw_service_v10/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_011523_125530925293` | [boundary_l32768_framework_raw_v11](runs/boundary_l32768_framework_raw_v11/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_011532_126446428080` | [l1_s6_h131072_b4_raw](runs/l1_s6_h131072_b4_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_012955_201076229165` | [b1_h131072_service_raw_v12](runs/b1_h131072_service_raw_v12/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013058_206820119712` | [l1_s6_h131072_b8_raw](runs/l1_s6_h131072_b8_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013058_206841531056` | [l1_s6_h131072_b16_raw](runs/l1_s6_h131072_b16_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013058_206856528351` | [l1_s6_h131072_b24_raw](runs/l1_s6_h131072_b24_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013058_206876031121` | [l1_s6_h131072_b28_raw](runs/l1_s6_h131072_b28_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013058_20689079835` | [l1_s6_h8192_b16_raw](runs/l1_s6_h8192_b16_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013059_20691856387` | [l1_s6_h8192_b32_raw](runs/l1_s6_h8192_b32_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013537_233869216938` | [boundary_l32772_framework_metadata_raw_v13](runs/boundary_l32772_framework_metadata_raw_v13/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013537_233874530894` | [mixed_261_h8192_l1_raw_v14](runs/mixed_261_h8192_l1_raw_v14/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013537_23389993243` | [boundary_l1048576_b1_service_raw_v15](runs/boundary_l1048576_b1_service_raw_v15/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013538_233941623733` | [continuous_h131072_raw_v16](runs/continuous_h131072_raw_v16/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013722_24382574878` | [l1_s6_h131072_b1_raw](runs/l1_s6_h131072_b1_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_013853_252063030343` | [boundary_l8192_b1_s1_raw_v17](runs/boundary_l8192_b1_s1_raw_v17/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_014459_281421627809` | [mixed_261_h8192_l1_shared_raw_v18](runs/mixed_261_h8192_l1_shared_raw_v18/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_014721_29211738823` | [l1_s6_h131072_b1_g6_candidate_raw](runs/l1_s6_h131072_b1_g6_candidate_raw/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_014806_295304419774` | [service_cost_context_matched_raw_v19](runs/service_cost_context_matched_raw_v19/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_022330_182138327918` | [b1_s1_to_s6_capacity6_bsh150_v20](runs/b1_s1_to_s6_capacity6_bsh150_v20/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_022555_195423132504` | [cube_tile_publish_b1_h131072_bsh150_v21](runs/cube_tile_publish_b1_h131072_bsh150_v21/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_022555_19542743018` | [cube_tile_publish_b16_h131072_bsh150_v21](runs/cube_tile_publish_b16_h131072_bsh150_v21/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_023804_33906131467` | [idle_phase_publish_b1_h131072_bsh150_v22](runs/idle_phase_publish_b1_h131072_bsh150_v22/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_023804_33959626400` | [idle_phase_publish_b16_h131072_bsh150_v22](runs/idle_phase_publish_b16_h131072_bsh150_v22/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_025144_402125724285` | [active_prefix_wait_b1_h131072_bsh150_v23](runs/active_prefix_wait_b1_h131072_bsh150_v23/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_025144_402140122430` | [active_prefix_wait_b16_h131072_bsh150_v23](runs/active_prefix_wait_b16_h131072_bsh150_v23/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_025918_113511129133` | [current_phase_b16_h131072_bsh150_v24](runs/current_phase_b16_h131072_bsh150_v24/result.json) | 未知 | 缺少结果 | 否 |
| `task_20261021_025918_11364392302` | [current_phase_b28_h131072_bsh150_v24](runs/current_phase_b28_h131072_bsh150_v24/result.json) | 未知 | 缺少结果 | 否 |
| `task_20261021_030030_161940329808` | [current_phase_b16_h131072_bsh150_v25](runs/current_phase_b16_h131072_bsh150_v25/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_030030_16197517222` | [current_phase_b28_h131072_bsh150_v25](runs/current_phase_b28_h131072_bsh150_v25/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_030409_311114030165` | [static_capacity_b16_h131072_bsh150_v26](runs/static_capacity_b16_h131072_bsh150_v26/result.json) | completed (exit=1) | native_reference_ready | 否 |
| `task_20261021_030409_311185832136` | [static_capacity_b28_h131072_bsh150_v26](runs/static_capacity_b28_h131072_bsh150_v26/result.json) | completed (exit=1) | native_reference_ready | 否 |
| `task_20261021_030948_12078975981` | [capacity_descriptors_b16_h131072_v27](runs/capacity_descriptors_b16_h131072_v27/result.json) | completed (exit=0) | descriptors_only | 否 |
| `task_20261021_033407_263280125800` | [native_wait_b16_h131072_bsh150_v28](runs/native_wait_b16_h131072_bsh150_v28/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_033407_263357822636` | [static_capacity_native_b16_h131072_bsh150_v28](runs/static_capacity_native_b16_h131072_bsh150_v28/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_033407_263433817616` | [static_capacity_native_b28_h131072_bsh150_v28](runs/static_capacity_native_b28_h131072_bsh150_v28/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_033407_26353694698` | [native_wait_mixed_123456_framework_h32766_v28](runs/native_wait_mixed_123456_framework_h32766_v28/result.json) | completed (exit=1) | real_service_route | 否 |
| `task_20261021_034309_210378126331` | [native_wait_mixed_123456_framework_h32766_v29](runs/native_wait_mixed_123456_framework_h32766_v29/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_034511_287684532064` | [native_wait_b16_uninterrupted_sampling_v30](runs/native_wait_b16_uninterrupted_sampling_v30/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_035651_98804820155` | [b1_balanced24_native_h131072_v31](runs/b1_balanced24_native_h131072_v31/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_041703_209414811817` | [b1_balanced24_native_s1_s6_v32](runs/b1_balanced24_native_s1_s6_v32/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_041749_213676022003` | [batched_acquire_native_b16_v33](runs/batched_acquire_native_b16_v33/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_041750_213690210682` | [batched_acquire_native_b28_v33](runs/batched_acquire_native_b28_v33/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_043018_286871914579` | [post_columns_b1_a241_v34](runs/post_columns_b1_a241_v34/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_043018_28688479929` | [post_columns_b4_a241_v34](runs/post_columns_b4_a241_v34/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_043018_286907020261` | [b1_balanced24_ties_h139264_a241_v35](runs/b1_balanced24_ties_h139264_a241_v35/result.json) | completed (exit=1) | complete | 否 |
| `task_20261021_043407_309493922351` | [initial_fast_polls_b1_a241_v36](runs/initial_fast_polls_b1_a241_v36/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_043407_309517524123` | [initial_fast_polls_b16_a241_v36](runs/initial_fast_polls_b16_a241_v36/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_044257_365360219269` | [current_bsh_native_b1_s1_s6_v37](runs/current_bsh_native_b1_s1_s6_v37/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_044257_365379017711` | [current_bsh_native_b1_service_graph_v37](runs/current_bsh_native_b1_service_graph_v37/result.json) | completed (exit=1) | real_service_route | 否 |
| `task_20261021_044707_38941343957` | [active_prefix_b1_a241_v38](runs/active_prefix_b1_a241_v38/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_044707_389431625197` | [active_prefix_b4_a241_v38](runs/active_prefix_b4_a241_v38/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_044707_38944097777` | [active_prefix_b8_a241_v38](runs/active_prefix_b8_a241_v38/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_050120_68460921096` | [score_wide_k128_b16_a241_v39](runs/score_wide_k128_b16_a241_v39/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_050121_6848358492` | [current_bsh_native_b1_service_graph_v40](runs/current_bsh_native_b1_service_graph_v40/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051025_12005359972` | [cohort_prefix_a241_h131072_b1_s6_v41](runs/cohort_prefix_a241_h131072_b1_s6_v41/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051026_120078423799` | [cohort_prefix_a241_h131072_b4_s6_v41](runs/cohort_prefix_a241_h131072_b4_s6_v41/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051026_120089123362` | [cohort_prefix_a241_h131072_b8_s6_v41](runs/cohort_prefix_a241_h131072_b8_s6_v41/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051026_120106812227` | [cohort_prefix_a241_h131072_b16_s6_v41](runs/cohort_prefix_a241_h131072_b16_s6_v41/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051026_120127617911` | [cohort_prefix_a241_h131072_b24_s6_v41](runs/cohort_prefix_a241_h131072_b24_s6_v41/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051026_12013206239` | [cohort_prefix_a241_h131072_b28_s6_v41](runs/cohort_prefix_a241_h131072_b28_s6_v41/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051026_12014858751` | [cohort_prefix_a241_h8192_b16_s6_v41](runs/cohort_prefix_a241_h8192_b16_s6_v41/result.json) | completed (exit=1) | native_reference_ready | 否 |
| `task_20261021_051026_120171029485` | [cohort_prefix_a241_h8192_b32_s6_v41](runs/cohort_prefix_a241_h8192_b32_s6_v41/result.json) | completed (exit=1) | native_reference_ready | 否 |
| `task_20261021_051507_150340519380` | [prefix_policy_b1_s1_s6_a241_v43](runs/prefix_policy_b1_s1_s6_a241_v43/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_051507_15036488295` | [prefix_policy_mixed_matrix_a241_v43](runs/prefix_policy_mixed_matrix_a241_v43/result.json) | completed (exit=1) | tnd_metadata_and_continuous_state | 否 |
| `task_20261021_053322_257585813797` | [prefix_policy_mixed_matrix_a241_v46](runs/prefix_policy_mixed_matrix_a241_v46/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_054041_30074296661` | [cohort_prefix_typefix_h8192_b16_s6_v47](runs/cohort_prefix_typefix_h8192_b16_s6_v47/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_054041_3007522242` | [cohort_prefix_typefix_h8192_b32_s6_v47](runs/cohort_prefix_typefix_h8192_b32_s6_v47/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_063940_212922026426` | [transfer_stride_pad64_h131072_b16_s6_v49](runs/transfer_stride_pad64_h131072_b16_s6_v49/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_065141_287902612619` | [scale_owner_wait_h131072_b16_s6_v51](runs/scale_owner_wait_h131072_b16_s6_v51/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_070255_352921911564` | [scale_owner_list_h131072_b16_s6_v53](runs/scale_owner_list_h131072_b16_s6_v53/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_071520_61535459` | [scale_owner_list_b1_s6_and_s1_s6_v55](runs/scale_owner_list_b1_s6_and_s1_s6_v55/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_073455_113981416736` | [b1_oa_supply_s6_and_s1_s6_v59](runs/b1_oa_supply_s6_and_s1_s6_v59/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_075510_226360516093` | [score_threeway_h131072_b16_s6_v61](runs/score_threeway_h131072_b16_s6_v61/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_082907_411978514951` | [score_ws_pipeline_h131072_b16_s6_v63](runs/score_ws_pipeline_h131072_b16_s6_v63/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_085454_157214514444` | [score_visibility_h131072_b16_s6_v65](runs/score_visibility_h131072_b16_s6_v65/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_091603_291753515374` | [output_work_balance_b1_s6_and_s1_s6_v67](runs/output_work_balance_b1_s6_and_s1_s6_v67/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_091605_291986121743` | [output_work_balance_h131072_b16_s6_v67](runs/output_work_balance_h131072_b16_s6_v67/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_092602_362071332328` | [output_work_balance_h131072_b4_s6_v68](runs/output_work_balance_h131072_b4_s6_v68/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_092603_362254932218` | [output_work_balance_h131072_b8_s6_v68](runs/output_work_balance_h131072_b8_s6_v68/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_094033_42904922785` | [output_work_balance_guard_h8192_b16_s6_v70](runs/output_work_balance_guard_h8192_b16_s6_v70/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_094035_4306327707` | [output_work_balance_guard_long_mixed_t65_v70](runs/output_work_balance_guard_long_mixed_t65_v70/result.json) | completed (exit=1) | tnd_metadata_and_continuous_state | 否 |
| `task_20261021_095605_14307563635` | [output_balance_long_mixed_t65_bcap16_v72](runs/output_balance_long_mixed_t65_bcap16_v72/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_095607_143256629477` | [output_balance_long_t66_boundary_v72](runs/output_balance_long_t66_boundary_v72/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_101618_280762024924` | [cohort_output_balance_h131072_b4_s6_v74](runs/cohort_output_balance_h131072_b4_s6_v74/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_101619_280897124735` | [cohort_output_balance_h131072_b8_s6_v74](runs/cohort_output_balance_h131072_b8_s6_v74/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_102626_32708358896` | [cohort_output_balance_h131072_b24_s6_v75](runs/cohort_output_balance_h131072_b24_s6_v75/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_102627_32720263427` | [cohort_output_balance_h131072_b28_s6_v75](runs/cohort_output_balance_h131072_b28_s6_v75/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_103655_374384014461` | [cohort_output_balance_h8192_b16_s6_v76](runs/cohort_output_balance_h8192_b16_s6_v76/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_103656_37456895465` | [cohort_output_balance_h8192_b32_s6_v76](runs/cohort_output_balance_h8192_b32_s6_v76/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_111201_12821755840` | [large_output_balance_h131072_b28_s6_v79](runs/large_output_balance_h131072_b28_s6_v79/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_111202_128298718985` | [short_hc_consumer_h8192_b32_s6_v79](runs/short_hc_consumer_h8192_b32_s6_v79/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_112147_186635723381` | [short_hc_consumer_h8192_b16_s6_v81](runs/short_hc_consumer_h8192_b16_s6_v81/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_112239_19267186046` | [b1_output_prefix_s6_and_s1_s6_v81](runs/b1_output_prefix_s6_and_s1_s6_v81/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_112912_227563513845` | [large_output_balance_h131072_b24_s6_v82](runs/large_output_balance_h131072_b24_s6_v82/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_121214_66656511685` | [single_m16_qa_b1_s6_and_s1_s6_v86](runs/single_m16_qa_b1_s6_and_s1_s6_v86/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_121215_667954606` | [single_m16_qa_t16_mixed664_v86](runs/single_m16_qa_t16_mixed664_v86/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_140935_275270122118` | [borrowed_score_stream_h131072_b16_s6_v90](runs/borrowed_score_stream_h131072_b16_s6_v90/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_140935_275288020871` | [borrowed_score_owner_h131072_b28_s6_v90](runs/borrowed_score_owner_h131072_b28_s6_v90/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_141858_318904115435` | [qa_incremental_b1_s6_v93](runs/qa_incremental_b1_s6_v93/result.json) | 未知 | 缺少结果 | 否 |
| `task_20261021_141858_318908415657` | [ws_incremental_b16_s6_v93](runs/ws_incremental_b16_s6_v93/result.json) | 未知 | 缺少结果 | 否 |
| `task_20261021_142045_325426029053` | [qa_incremental_b1_s6_v93b](runs/qa_incremental_b1_s6_v93b/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_142045_32543028090` | [ws_incremental_b16_s6_v93b](runs/ws_incremental_b16_s6_v93b/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_144229_86824910253` | [short_half_ub_h8192_b16_s6_v95](runs/short_half_ub_h8192_b16_s6_v95/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_150402_219144119122` | [short_half_ub_t65_r16_matrix_v98](runs/short_half_ub_t65_r16_matrix_v98/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_151458_2930389393` | [topk_prefix_h131072_b28_s6_v99](runs/topk_prefix_h131072_b28_s6_v99/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_154139_748519735` | [topk_prefix_h131072_b8_s6_v101](runs/topk_prefix_h131072_b8_s6_v101/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_155419_187365031349` | [short_half_ub_ties_device_v100b](runs/short_half_ub_ties_device_v100b/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_160738_306213119230` | [topk_prefix_t41_r12_matrix_v103](runs/topk_prefix_t41_r12_matrix_v103/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_162057_10868510136` | [topk_prefix_ties_device_v104](runs/topk_prefix_ties_device_v104/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_162824_11408309460` | [cohort_borrowed_combined_h131072_b8_s6_v106](runs/cohort_borrowed_combined_h131072_b8_s6_v106/result.json) | completed (exit=0) | complete | PASS |
| `task_20261021_162824_11409259262` | [cohort_borrowed_combined_h8192_b16_s6_v106](runs/cohort_borrowed_combined_h8192_b16_s6_v106/result.json) | completed (exit=0) | complete | PASS |

各任务完整命令和固定launcher SHA256见 `jobs.json`。
Native超阈值元素、Top-K替换、cache/state及保护区明细仍在原始result.json；
本表的PASS不表示修复了原单SPMD对Native的已有数值差异。
