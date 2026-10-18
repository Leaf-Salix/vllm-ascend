# CSA七档最新已有性能对照（2026-10-18）

正式第4层W8A8、S=6、完整CSA含HC_pre/HC_post，单位μs，均为中位数。本次未启动新测试。

动态列选当前正式采用策略的最近实测，不从实验候选中挑最小值。每行多task与动态SPMD同轮；静态列取已有独立轮次。

Native是外部profiler完整设备kernel跨度，包含调度；PTO是完整L1 incore首尾，排除外围首尾。两种口径不能直接算加速比。

| 历史 / Batch | Native外部profiling | PTO多task L1 | 静态SPMD L1（历史） | 动态单SPMD L1 | 动态相对多task耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K / B4 | 781.88 | 502.66 | 541.92 | 546.38 | +8.70% |
| 128K / B8 | 900.12 | 640.68 | 641.56 | 694.32 | +8.37% |
| 128K / B16 | 1169.25 | 869.78 | 781.94 | 889.40 | +2.26% |
| 128K / B24 | 1354.75 | 1169.62 | 1121.12 | 1163.42 | -0.53% |
| 128K / B28 | 1701.75 | 1312.22 | — | 1337.28 | +1.91% |
| 8K / B16 | 790.25 | 667.50 | 621.96 | 698.60 | +4.66% |
| 8K / B32 | 1101.50 | 985.64 | 905.72 | 1004.20 | +1.88% |

耗时变化正值为更慢，负值为更快。七档均未达到至少降低30%的目标。

## 轮次与版本说明

- B16正式版本最近一轮为多task869.78、动态889.40；静态最新已有值781.94来自另一轮，其同轮多task是880.06。不能把781.94当成与869.78同轮。
- B16最新4辅助核实验候选为947.16，同轮多task880.06，退步7.62%，未采用；候选源码与全部验证保存在projection_branch_evidence。
- 历史B16多task905.08及静态807.16保留。905.08→880.06涉及编译环境和采样流程差异，未隔离具体成因；核对见projection_branch_evidence/MULTITASK_905_VS_880_AUDIT.json。
- B28 Native为10月18日优化测量，已开静态编译、SuperKernel及NZ29投影权重，30次外部profiling中位1701.75；不带profiler的Event20次中位1653.08，未混入表格。
- B28多task/动态列已有5次同轮采样，动态版本为9d5327e5；66882634新增的QA小档分支不作用于T168。最新依赖环境没有重测B28。
- B28静态既有任务在导入阶段失败，无性能数据；按只用已有静态结果的要求留空，不补跑静态。其他三项齐全，无需补测。
- 其余Native来自10月12日已有前后两轮各30次合并中位数，不能当作10月18日同轮环境。其他静态列来自各档已验收历史记录。

## 逐档PTO配对来源

- 128K/B4: `task_20261018_061655_17204335454`；[oa_pair_small_t_h131072_b4_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_small_t_h131072_b4_v1/result.json)。
- 128K/B8: `task_20261018_060343_14576331299`；[oa_pair_manual_h131072_b8_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/oa_pair_manual_h131072_b8_v1/result.json)。
- 128K/B16: `task_20261018_092130_124935114251`；[static_same_process_h131072_b16_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/score_attention_interleave_candidate/static_same_process_h131072_b16_v1/result.json)。
- 128K/B24: `task_20261018_070212_27121133702`；[attention_scalar_geometry_h131072_b24_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_scalar_geometry_h131072_b24_v1/result.json)。
- 128K/B28: `task_20261018_050316_27292720517`；[query_linear_h131072_b28_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/query_linear_h131072_b28_v1/result.json)。
- 8K/B16: `task_20261018_070212_271203632049`；[attention_scalar_geometry_h8192_b16_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/attention_scalar_geometry_h8192_b16_v1/result.json)。
- 8K/B32: `task_20261018_071700_29885801413`；[qb_global_n128_h8192_b32_v1](/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/csa_multishape_spmd_20261016/dynamic_full_pipeline_20261017/parallel_20261018/qb_global_n128_h8192_b32_v1/result.json)。

同名JSON保存全部样本、环境、源码版本、静态同轮基线和原始结果路径；本文件不把跨轮绝对差解释为优化收益。
