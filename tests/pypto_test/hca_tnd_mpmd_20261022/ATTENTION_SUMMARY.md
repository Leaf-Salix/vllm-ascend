# HCA TND attention：2026-10-23 核内流水候选

以同任务双侧完整层L1首尾作比较，单位μs；性能提升=`control/candidate−1`。
正式入口按区间选择K128/K512；大块候选改变舍入，局部/整层检查不代表整网token验收。
NZ且Native已有Host max_seq_lens≥131072、T≥33时用K512；其他仍用K128。
T129～192再叠加已验证的Q_B/O_B投影档。所有边界/实际长度继续读取device Q/L。
Native严格storage既有差异保留。没有ACLGraph回放，没有新增Native性能测量。

| 试验 | 控制 | 候选 | 性能提升 | 更快配对 | Native输出 | 控制storage精确 |
|---|---:|---:|---:|---:|---|---|
| attention_pv_dual_b16_v2 | 499.04 | 506.34 | -1.44% | 1/7 | PASS | True |
| attention_pv_dual_b28_v2 | 766.96 | 758.38 | +1.13% | 6/7 | PASS | True |
| attention_q_l0_b16_v1 | 514.90 | 520.02 | -0.98% | 1/7 | PASS | True |
| attention_q_l0_b28_v1 | 765.52 | 768.08 | -0.33% | 3/7 | PASS | True |
| attention_macro512_full_b16_v4 | 517.60 | 497.80 | +3.98% | 7/7 | PASS | True |
| attention_macro512_full_b28_v4 | 769.46 | 748.48 | +2.80% | 7/7 | PASS | True |
| attention_macro512_full_h131072_b1_v4 | 216.82 | 210.30 | +3.10% | 3/5 | PASS | True |
| attention_macro512_full_h131072_b4_v4 | 278.22 | 279.80 | -0.56% | 2/5 | PASS | True |
| attention_macro512_full_h131072_b8_v4 | 378.82 | 354.18 | +6.96% | 5/5 | PASS | True |
| attention_macro512_full_h131072_b24_v4 | 703.80 | 696.50 | +1.05% | 3/5 | PASS | True |
| attention_macro512_full_h8192_b16_v4 | 386.64 | 384.66 | +0.51% | 2/5 | PASS | True |
| attention_macro512_full_h8192_b32_v4 | 598.60 | 628.40 | -4.74% | 0/5 | PASS | True |
| attention_macro512_qk2_b16_v1 | 535.06 | 545.94 | -1.99% | 0/7 | PASS | True |
| attention_macro512_qk2_b28_v1 | 757.44 | 759.36 | -0.25% | 1/7 | PASS | True |
| attention_policy_small_b4_v4 | 273.22 | 266.30 | +2.60% | 5/5 | PASS | True |
| attention_policy_short_b32_v4 | 603.94 | 598.44 | +0.92% | 3/5 | PASS | True |
| attention_policy_final_b8_v4 | 351.50 | 347.58 | +1.13% | 5/5 | PASS | True |
| attention_policy_final_b16_v4 | 537.84 | 518.42 | +3.75% | 7/7 | PASS | True |
| attention_policy_final_b24_v4 | 687.20 | 676.86 | +1.53% | 4/5 | PASS | True |
| attention_policy_final_b28_v4 | 771.66 | 765.88 | +0.75% | 6/7 | PASS | True |
| attention_policy_refactor_b28_v4 | 762.66 | 748.12 | +1.94% | 7/7 | PASS | True |

## 未列性能的任务

- `attention_macro512_probe_v3`：completed (exit=2)，无结果文件。
- `attention_macro512_probe_v4`：completed (exit=0)，DIAGNOSTIC_COMPLETE。
- `attention_macro512_s1_b16_v4`：completed (exit=0)，completed。
- `attention_macro512_s5_b28_v4`：completed (exit=0)，completed。
- `attention_macro512_mixed_edge_v4`：completed (exit=1)，failed。
- `attention_macro512_mixed_edge_v5`：completed (exit=1)，failed。
- `attention_policy_mixed_fp64_v4`：completed (exit=0)，DIAGNOSTIC_COMPLETE。
- `attention_policy_s5_service_v4`：completed (exit=0)，service_validated。
- `attention_policy_s1_service_v4`：completed (exit=0)，service_validated。
- `attention_policy_t33_service_v4`：completed (exit=0)，service_validated。
- `attention_policy_256k_v4`：completed (exit=0)，completed。

完整样本、attention AIC/AIV各核耗时、功能结果、失败CPU编译和任务身份见同名JSON。
源码来源、设计差异与不采用项见`ATTENTION_NATIVE_PIPELINE_REVIEW.md`。
