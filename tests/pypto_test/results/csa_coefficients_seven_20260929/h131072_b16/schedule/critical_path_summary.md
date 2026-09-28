> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **1019.260 µs**
- Selected AICore makespan coverage: **1014.500 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 1019.260 | 1014.500 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_coefficients_seven_20260929/h131072_b16/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **1019.260 µs**
- AICore makespan: **1014.500 µs**
- Static CPM cross-check: **803.060 µs**
- Primary table: **Observed critical path**, 25 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 14.360 | 14.360 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 13.220 | 13.220 | 13.600 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 1.940 | 1.940 | 9.200 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 6.240 | 6.240 | 7.180 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 18.180 | 18.180 | 8.800 | data-wait | 🐌 |
| 5 | qr_proj_matmul | `8589934593` | 16.000 | 16.000 | 5.580 | data-wait | 🐌 ⭐ |
| 6 | kv_score_proj_0 | `4294967318` | 37.420 | 36.540 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | kv_score_proj | `4294967314` | 68.760 | 50.760 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | qproj_matmul | `8589934596` | 104.940 | 62.580 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 9 | qproj_dequant_rms_nope_rope | `8589934597` | 32.020 | 32.020 | 11.860 | data-wait | 🐌 |
| 10 | qr_hadamard_quant | `8589934606` | 18.420 | 7.080 | 0.000 | — |  |
| 11 | indexer_head_coefficients | `8589934611` | 4.000 | 4.000 | 1.920 | data-wait | 🐌 ⭐ |
| 12 | indexer_score_topk_native_pair_aic | `8589934612` | 262.820 | 262.820 | 14.480 | data-wait | 🐌 |
| 13 | indexer_topk_query_merge__2 | `12884901888` | 16.840 | 16.840 | 6.300 | data-wait | 🐌 |
| 14 | csa_slots_build_valid_qk_plan | `4294967330` | 9.820 | 9.820 | 4.740 | data-wait | 🐌 ⭐ |
| 15 | qk_pv_aic | `4294967334` | 169.520 | 169.520 | 7.980 | data-wait | 🐌 ⭐ |
| 16 | merge_norm | `4294967336` | 26.800 | 26.800 | 6.620 | data-wait | 🐌 ⭐ |
| 17 | proj_a_mm_0 | `12884901901` | 29.780 | 29.780 | 7.180 | data-wait | 🐌 |
| 18 | proj_a_mm_0 | `12884901895` | 31.780 | 30.780 | 0.000 | — |  |
| 19 | _proj_b_mm_nz_kernel__2 | `12884901903` | 18.160 | 17.000 | 0.000 | — | ⭐ |
| 20 | _proj_b_mm_nz_kernel__2 | `12884901909` | 23.820 | 23.400 | 0.000 | — | ⭐ partial rows 3/8 |
| 21 | _proj_b_mm_nz_kernel__2 | `12884901897` | 16.400 | 4.920 | 0.000 | — |  |
| 22 | _proj_b_mm_nz_kernel__2 | `12884901894` | 9.740 | 9.740 | 0.580 | core-wait | ⭐ |
| 23 | proj_b_act_0 | `8589934614` | 16.060 | 16.060 | 2.000 | data-wait | 🐌 ⭐ |
| 24 | hc_post | `8589934615` | 22.440 | 22.440 | 3.640 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 39.140 | 47.500 | 51.900 | 52.740 | producer end→FIN 8.360 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.400 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 47.500–51.900 µs; dispatch→start 0.840 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 65.960 | 74.380 | 60.320 | 75.160 | producer end→FIN 8.420 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.840 µs (post-dispatch) |
| `4294967301` split_pre_post | 77.100 | 79.460 | 83.560 | 84.280 | producer end→FIN 2.360 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.100 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 79.460–83.080 µs; dispatch→start 0.720 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 90.520 | 93.260 | 98.100 | 99.320 | producer end→FIN 2.740 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.840 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 93.260–98.100 µs; dispatch→start 1.220 µs (post-dispatch) |
| `8589934593` qr_proj_matmul | 117.500 | 121.920 | 99.940 | 123.080 | producer end→FIN 4.420 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 23.140 µs (post-dispatch) |
| `8589934597` qproj_dequant_rms_nope_rope | 288.960 | 295.040 | 299.900 | 300.820 | producer end→FIN 6.080 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.860 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 295.040–295.800 µs; dispatch→start 0.920 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients | 339.920 | 341.220 | 332.800 | 341.840 | producer end→FIN 1.300 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 9.040 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_aic | 345.840 | 349.000 | 356.900 | 360.320 | producer end→FIN 3.160 µs; post-FIN ready→dispatch scheduler delay 7.900 µs; dispatch resource blocker unproven: MIX/heterogeneous launch requires cluster-aware manual inspection; dispatch→start 3.420 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge__2 | 623.140 | 623.980 | 628.260 | 629.440 | producer end→FIN 0.840 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.280 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 623.980–628.260 µs; dispatch→start 1.180 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 646.280 | 650.280 | 632.840 | 651.020 | producer end→FIN 4.000 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.180 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 660.840 | 668.060 | 652.840 | 668.820 | producer end→FIN 7.220 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.980 µs (post-dispatch) |
| `4294967336` merge_norm | 838.340 | 844.180 | 676.140 | 844.960 | producer end→FIN 5.840 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 168.820 µs (post-dispatch) |
| `12884901901` proj_a_mm_0 | 871.760 | 874.520 | 877.960 | 878.940 | producer end→FIN 2.760 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.440 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 874.520–877.340 µs; dispatch→start 0.980 µs (post-dispatch) |
| `8589934614` proj_b_act_0 | 995.140 | 996.300 | 986.240 | 997.140 | producer end→FIN 1.160 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 10.900 µs (post-dispatch) |
| `8589934615` hc_post | 1013.200 | 1016.060 | 998.540 | 1016.840 | producer end→FIN 2.860 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.300 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901888` | `8589934612` | indexer_score_topk_native_pair_aic | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `12884901901` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
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
