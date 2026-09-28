> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **1272.820 µs**
- Selected AICore makespan coverage: **1268.080 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 1272.820 | 1268.080 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_coefficients_seven_20260929/h131072_b24/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **1272.820 µs**
- AICore makespan: **1268.080 µs**
- Static CPM cross-check: **1071.480 µs**
- Primary table: **Observed critical path**, 25 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 18.700 | 18.700 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 17.660 | 17.660 | 15.400 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 2.000 | 2.000 | 2.840 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 7.940 | 7.940 | 8.960 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 21.520 | 21.520 | 7.800 | data-wait | 🐌 |
| 5 | weights_proj | `8589934608` | 12.260 | 12.260 | 10.020 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | kv_score_proj_0 | `4294967318` | 26.600 | 24.780 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | kv_proj_matmul_0 | `8589934599` | 42.300 | 36.400 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | idx_qr_proj_matmul | `8589934602` | 28.500 | 19.900 | 0.000 | — |  |
| 9 | idx_qr_dequant_rope | `8589934603` | 34.280 | 34.280 | 15.640 | data-wait | 🐌 |
| 10 | qr_hadamard_matmul | `8589934605` | 17.040 | 17.040 | 9.640 | data-wait | 🐌 ⭐ partial rows 13/24 |
| 11 | qr_hadamard_quant | `8589934606` | 23.420 | 23.420 | 5.840 | data-wait | 🐌 ⭐ partial rows 29/48 |
| 12 | qproj_dequant_rms_nope_rope | `8589934597` | 464.680 | 464.680 | 3.580 | core-wait | 🐌 |
| 13 | indexer_score_topk_native_pair_aic | `8589934612` | 379.360 | 0.000 | 0.000 | — |  |
| 14 | indexer_topk_query_merge__2 | `12884901888` | 25.260 | 8.420 | 0.000 | — |  |
| 15 | csa_slots_build_valid_qk_plan | `4294967330` | 13.620 | 13.620 | 1.540 | data-wait | 🐌 ⭐ |
| 16 | qk_pv_aic | `4294967334` | 213.600 | 213.600 | 5.320 | data-wait | 🐌 ⭐ |
| 17 | merge_norm | `4294967336` | 35.040 | 35.040 | 5.680 | data-wait | 🐌 ⭐ |
| 18 | proj_a_mm_1 | `12884901910` | 30.540 | 30.540 | 7.120 | data-wait | 🐌 |
| 19 | proj_a_mm_1 | `12884901898` | 30.100 | 27.920 | 0.000 | — |  |
| 20 | _proj_b_mm_nz_kernel__3 | `12884901912` | 24.800 | 23.600 | 0.000 | — | ⭐ |
| 21 | _proj_b_mm_nz_kernel__3 | `12884901900` | 37.360 | 34.320 | 0.000 | — | ⭐ partial rows 5/8 |
| 22 | _proj_b_mm_nz_kernel__3 | `12884901897` | 24.720 | 24.720 | 1.320 | core-wait | 🐌 ⭐ |
| 23 | proj_b_act_1 | `8589934614` | 19.000 | 19.000 | 1.480 | data-wait | 🐌 ⭐ |
| 24 | hc_post | `8589934615` | 29.740 | 29.740 | 4.800 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 41.080 | 52.000 | 55.420 | 56.480 | producer end→FIN 10.920 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.420 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 52.000–55.420 µs; dispatch→start 1.060 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 74.140 | 76.160 | 62.360 | 76.980 | producer end→FIN 2.020 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.620 µs (post-dispatch) |
| `4294967301` split_pre_post | 78.980 | 82.380 | 87.140 | 87.940 | producer end→FIN 3.400 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.760 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 82.380–86.880 µs; dispatch→start 0.800 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 95.880 | 99.080 | 102.720 | 103.680 | producer end→FIN 3.200 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.640 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 99.080–102.720 µs; dispatch→start 0.960 µs (post-dispatch) |
| `8589934608` weights_proj | — | — | 134.600 | 135.220 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934603` idx_qr_dequant_rope | 228.560 | 237.960 | 243.340 | 244.200 | producer end→FIN 9.400 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.380 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 237.960–243.340 µs; dispatch→start 0.860 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 278.480 | 287.280 | 262.500 | 288.120 | producer end→FIN 8.800 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 25.620 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 305.160 | 310.360 | 303.560 | 311.000 | producer end→FIN 5.200 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 7.440 µs (post-dispatch) |
| `8589934597` qproj_dequant_rms_nope_rope | 330.500 | 333.900 | 337.120 | 338.000 | producer end→FIN 3.400 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.220 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 333.900–334.340 µs; dispatch→start 0.880 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 811.100 | 811.980 | 792.900 | 812.640 | producer end→FIN 0.880 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 19.740 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 826.260 | 830.920 | 814.220 | 831.580 | producer end→FIN 4.660 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 17.360 µs (post-dispatch) |
| `4294967336` merge_norm | 1045.180 | 1050.180 | 836.360 | 1050.860 | producer end→FIN 5.000 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 214.500 µs (post-dispatch) |
| `12884901910` proj_a_mm_1 | 1085.900 | 1089.680 | 1091.960 | 1093.020 | producer end→FIN 3.780 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.280 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 1089.680–1091.960 µs; dispatch→start 1.060 µs (post-dispatch) |
| `12884901897` _proj_b_mm_nz_kernel__3 | 1200.080 | 1203.000 | 1195.580 | 1210.720 | producer end→FIN 2.920 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.140 µs (post-dispatch) |
| `8589934614` proj_b_act_1 | 1235.440 | 1236.240 | 1204.840 | 1236.920 | producer end→FIN 0.800 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 32.080 µs (post-dispatch) |
| `8589934615` hc_post | 1255.920 | 1260.020 | 1239.020 | 1260.720 | producer end→FIN 4.100 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 21.700 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901910` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967298` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967301` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967304` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `8589934597` | `8589934596` | qproj_matmul | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `8589934603` | `4` | csa_rope_sign | early-dispatch policy | direct producer lacks `early_dispatch=true` |

> The post-FIN ready→dispatch rows above prove scheduler delay, but no named dispatch blocker was proven for those rows. Level-4 scheduler phase records do not carry ordinary task IDs; a causal task is named only when every compatible core's running and pending descriptor slots are saturated throughout the interval.

## Interpretation limits

- Dispatch→finish is the closest per-rank elapsed available in the swimlane; it is not the DFX-off benchmark.
- The critical-path makespan covers first AICore start through last AICore end, excluding host/orchestrator front time and AICPU/host tail time.
- `post-dispatch same-core` is restricted to core row(s) that realize the task-global earliest start and occupancy continuing through the row-local effective-ready→start window. It proves start contention there, not that all cores were unable to accept dispatch.
- MIX, SPMD, and sync-start capacity may require cluster-aware manual inspection; the report leaves a scheduler delay unattributed unless full compatible-engine descriptor saturation is proven.
- Level-4 collection has observer cost. Keep the unprofiled benchmark as the production performance headline.
