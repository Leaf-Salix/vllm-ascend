> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **740.460 µs**
- Selected AICore makespan coverage: **736.340 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 740.460 | 736.340 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_coefficients_seven_20260929/h131072_b8/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **740.460 µs**
- AICore makespan: **736.340 µs**
- Static CPM cross-check: **553.920 µs**
- Primary table: **Observed critical path**, 23 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 12.720 | 12.720 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 10.100 | 10.100 | 6.860 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 1.600 | 1.600 | 5.680 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 4.080 | 4.080 | 4.020 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 16.620 | 16.620 | 4.200 | data-wait | 🐌 |
| 5 | kv_score_proj_0 | `4294967318` | 20.900 | 20.900 | 10.780 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | kv_score_proj | `4294967314` | 30.260 | 30.260 | 1.980 | core-wait | 🐌 ⚠ early-dispatch unverifiable |
| 7 | qproj_matmul | `8589934596` | 60.240 | 60.240 | 5.300 | core-wait | 🐌 ⚠ early-dispatch unverifiable |
| 8 | qproj_dequant_rms_nope_rope | `8589934597` | 26.840 | 26.840 | 7.540 | data-wait | 🐌 |
| 9 | qr_hadamard_quant | `8589934606` | 16.860 | 15.420 | 0.000 | — | ⭐ partial rows 18/48 |
| 10 | indexer_head_coefficients | `8589934611` | 2.900 | 2.900 | 3.980 | data-wait | 🐌 ⭐ |
| 11 | indexer_score_topk_native_pair_aic | `8589934612` | 165.080 | 165.080 | 11.180 | data-wait | 🐌 |
| 12 | indexer_topk_query_merge__2 | `12884901888` | 13.020 | 13.020 | 7.220 | data-wait | 🐌 |
| 13 | csa_slots_build_valid_qk_plan | `4294967330` | 8.720 | 8.720 | 6.980 | data-wait | 🐌 ⭐ |
| 14 | qk_pv_aic | `4294967334` | 82.720 | 82.720 | 5.720 | data-wait | 🐌 ⭐ |
| 15 | merge_norm | `4294967336` | 17.560 | 17.560 | 5.840 | data-wait | 🐌 ⭐ |
| 16 | proj_a_mm_0 | `12884901910` | 28.820 | 28.820 | 7.320 | data-wait | 🐌 |
| 17 | proj_a_mm_0 | `12884901895` | 27.380 | 25.640 | 0.000 | — |  |
| 18 | proj_a_mm_0 | `12884901889` | 25.060 | 23.000 | 0.000 | — |  |
| 19 | quant_0 | `12884901890` | 6.040 | 6.040 | 10.820 | data-wait | 🐌 ⭐ |
| 20 | _proj_b_mm_nz_kernel__2 | `12884901891` | 17.580 | 17.580 | 2.880 | data-wait | 🐌 ⭐ partial rows 1/8 |
| 21 | proj_b_act_0 | `8589934614` | 14.460 | 14.460 | 1.580 | data-wait | 🐌 ⭐ |
| 22 | hc_post | `8589934615` | 19.760 | 19.760 | 2.380 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 32.720 | 35.720 | 38.580 | 39.580 | producer end→FIN 3.000 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.860 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 35.720–38.580 µs; dispatch→start 1.000 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 49.680 | 54.780 | 40.160 | 55.360 | producer end→FIN 5.100 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.200 µs (post-dispatch) |
| `4294967301` split_pre_post | 56.960 | 57.660 | 60.100 | 60.980 | producer end→FIN 0.700 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.440 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 57.660–60.080 µs; dispatch→start 0.880 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 65.060 | 66.180 | 68.540 | 69.260 | producer end→FIN 1.120 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.360 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 66.180–68.540 µs; dispatch→start 0.720 µs (post-dispatch) |
| `4294967318` kv_score_proj_0 | — | — | 96.000 | 96.660 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `4294967314` kv_score_proj | — | — | 118.760 | 119.540 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934596` qproj_matmul | — | — | 148.060 | 155.100 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934597` qproj_dequant_rms_nope_rope | 215.340 | 217.100 | 222.060 | 222.880 | producer end→FIN 1.760 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.960 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 217.100–217.580 µs; dispatch→start 0.820 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients | 265.140 | 268.220 | 260.000 | 269.120 | producer end→FIN 3.080 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 9.120 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_aic | 272.020 | 274.640 | 280.800 | 283.200 | producer end→FIN 2.620 µs; post-FIN ready→dispatch scheduler delay 6.160 µs; dispatch resource blocker unproven: MIX/heterogeneous launch requires cluster-aware manual inspection; dispatch→start 2.400 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge__2 | 448.280 | 450.120 | 454.560 | 455.500 | producer end→FIN 1.840 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.440 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 450.120–454.560 µs; dispatch→start 0.940 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 468.520 | 474.820 | 462.020 | 475.500 | producer end→FIN 6.300 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.480 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 484.220 | 489.160 | 476.620 | 489.940 | producer end→FIN 4.940 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.320 µs (post-dispatch) |
| `4294967336` merge_norm | 572.660 | 577.700 | 493.540 | 578.500 | producer end→FIN 5.040 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 84.960 µs (post-dispatch) |
| `12884901910` proj_a_mm_0 | 596.060 | 599.880 | 602.480 | 603.380 | producer end→FIN 3.820 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.600 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 599.880–602.480 µs; dispatch→start 0.900 µs (post-dispatch) |
| `12884901890` quant_0 | 680.840 | 690.740 | 650.900 | 691.660 | producer end→FIN 9.900 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 40.760 µs (post-dispatch) |
| `12884901891` _proj_b_mm_nz_kernel__2 | 697.700 | 698.900 | 698.320 | 700.580 | producer end→FIN 1.200 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 2.260 µs (post-dispatch) |
| `8589934614` proj_b_act_0 | 718.160 | 719.020 | 706.500 | 719.740 | producer end→FIN 0.860 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.240 µs (post-dispatch) |
| `8589934615` hc_post | 734.200 | 735.940 | 720.880 | 736.580 | producer end→FIN 1.740 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.700 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901888` | `8589934612` | indexer_score_topk_native_pair_aic | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `12884901910` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967298` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967301` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967304` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `8589934597` | `8589934596` | qproj_matmul | early-dispatch policy | direct producer lacks `early_dispatch=true` |

> The post-FIN ready→dispatch rows above prove scheduler delay, but no named dispatch blocker was proven for those rows. Level-4 scheduler phase records do not carry ordinary task IDs; a causal task is named only when every compatible core's running and pending descriptor slots are saturated throughout the interval.

## Interpretation limits

- Dispatch→finish is the closest per-rank elapsed available in the swimlane; it is not the DFX-off benchmark.
- The critical-path makespan covers first AICore start through last AICore end, excluding host/orchestrator front time and AICPU/host tail time.
- `post-dispatch same-core` is restricted to core row(s) that realize the task-global earliest start and occupancy continuing through the row-local effective-ready→start window. It proves start contention there, not that all cores were unable to accept dispatch.
- MIX, SPMD, and sync-start capacity may require cluster-aware manual inspection; the report leaves a scheduler delay unattributed unless full compatible-engine descriptor saturation is proven.
- Level-4 collection has observer cost. Keep the unprofiled benchmark as the production performance headline.
