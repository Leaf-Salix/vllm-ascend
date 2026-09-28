> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **1080.900 µs**
- Selected AICore makespan coverage: **1076.760 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 1080.900 | 1076.760 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_coefficients_seven_20260929/h8192_b32/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **1080.900 µs**
- AICore makespan: **1076.760 µs**
- Static CPM cross-check: **876.720 µs**
- Primary table: **Observed critical path**, 23 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 17.400 | 17.400 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 18.080 | 18.080 | 13.820 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 2.580 | 2.580 | 4.040 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 7.200 | 7.200 | 9.260 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 21.260 | 21.260 | 9.980 | data-wait | 🐌 |
| 5 | qr_proj_matmul_0 | `8589934593` | 16.820 | 16.820 | 3.720 | data-wait | 🐌 ⭐ |
| 6 | weights_proj | `8589934608` | 78.100 | 77.780 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | qproj_matmul | `8589934596` | 124.860 | 119.740 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | qproj_dequant_rms_nope_rope | `8589934597` | 74.360 | 74.360 | 8.760 | data-wait | 🐌 |
| 9 | kv_and_cache_write | `4294967324` | 4.160 | 0.000 | 0.000 | — |  |
| 10 | idx_kv_scale_commit | `4294967325` | 13.280 | 2.120 | 0.000 | — |  |
| 11 | indexer_score_topk_native_pair_2_aic | `8589934612` | 104.640 | 104.640 | 1.340 | data-wait | 🐌 ⭐ |
| 12 | indexer_topk_query_merge | `12884901888` | 14.140 | 14.140 | 2.800 | data-wait | 🐌 ⭐ |
| 13 | csa_slots_build_valid_qk_plan | `4294967330` | 13.460 | 13.460 | 3.780 | data-wait | 🐌 ⭐ |
| 14 | qk_pv_aic | `4294967334` | 235.860 | 235.860 | 3.860 | data-wait | 🐌 ⭐ |
| 15 | merge_norm | `4294967336` | 40.720 | 40.720 | 4.980 | data-wait | 🐌 ⭐ |
| 16 | proj_a_mm_1 | `12884901907` | 41.900 | 41.900 | 7.300 | data-wait | 🐌 |
| 17 | proj_a_mm_1 | `12884901901` | 45.720 | 44.100 | 0.000 | — |  |
| 18 | _proj_b_mm_nz_kernel__3 | `12884901912` | 58.060 | 55.060 | 0.000 | — | ⭐ partial rows 5/8 |
| 19 | _proj_b_mm_nz_kernel__3 | `12884901900` | 34.160 | 5.620 | 0.000 | — | ⭐ |
| 20 | _proj_b_mm_nz_kernel__3 | `12884901894` | 33.000 | 33.000 | 1.200 | core-wait | 🐌 ⭐ partial rows 5/8 |
| 21 | proj_b_act_1 | `8589934614` | 21.860 | 21.860 | 0.980 | data-wait | ⭐ |
| 22 | hc_post | `8589934615` | 28.920 | 28.920 | 4.320 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 45.280 | 52.780 | 57.900 | 59.100 | producer end→FIN 7.500 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.120 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 52.780–57.900 µs; dispatch→start 1.200 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 77.180 | 80.320 | 64.780 | 81.220 | producer end→FIN 3.140 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 16.440 µs (post-dispatch) |
| `4294967301` split_pre_post | 83.800 | 87.000 | 92.440 | 93.060 | producer end→FIN 3.200 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.440 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 87.000–91.540 µs; dispatch→start 0.620 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 100.260 | 102.820 | 109.480 | 110.240 | producer end→FIN 2.560 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 6.660 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 102.820–106.760 µs; dispatch→start 0.760 µs (post-dispatch) |
| `8589934593` qr_proj_matmul_0 | 131.500 | 134.160 | 112.680 | 135.220 | producer end→FIN 2.660 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 22.540 µs (post-dispatch) |
| `8589934597` qproj_dequant_rms_nope_rope | 349.560 | 353.060 | 357.580 | 358.320 | producer end→FIN 3.500 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.520 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 353.060–353.680 µs; dispatch→start 0.740 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_2_aic | 434.800 | 435.520 | 421.960 | 436.140 | producer end→FIN 0.720 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.180 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge | 540.780 | 542.760 | 440.020 | 543.580 | producer end→FIN 1.980 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 103.560 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 557.720 | 560.960 | 546.620 | 561.500 | producer end→FIN 3.240 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.880 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 574.960 | 578.160 | 562.860 | 578.820 | producer end→FIN 3.200 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.960 µs (post-dispatch) |
| `4294967336` merge_norm | 814.680 | 818.940 | 583.320 | 819.660 | producer end→FIN 4.260 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 236.340 µs (post-dispatch) |
| `12884901907` proj_a_mm_1 | 860.380 | 863.660 | 866.760 | 867.680 | producer end→FIN 3.280 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.100 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 863.660–866.260 µs; dispatch→start 0.920 µs (post-dispatch) |
| `12884901894` _proj_b_mm_nz_kernel__3 | 1004.380 | 1010.980 | 999.340 | 1015.560 | producer end→FIN 6.600 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 16.220 µs (post-dispatch) |
| `8589934615` hc_post | 1071.400 | 1075.040 | 1051.980 | 1075.720 | producer end→FIN 3.640 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 23.740 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901907` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
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
