> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **1016.060 µs**
- Selected AICore makespan coverage: **1012.660 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 1016.060 | 1012.660 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929/h131072_b16/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **1016.060 µs**
- AICore makespan: **1012.660 µs**
- Static CPM cross-check: **786.200 µs**
- Primary table: **Observed critical path**, 25 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 14.260 | 14.260 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 10.260 | 10.260 | 7.520 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 2.180 | 2.180 | 6.840 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 4.660 | 4.660 | 7.540 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 18.340 | 18.340 | 8.620 | data-wait | 🐌 |
| 5 | qr_proj_matmul | `8589934593` | 17.620 | 17.620 | 4.420 | data-wait | 🐌 ⭐ |
| 6 | kv_score_proj_0 | `4294967318` | 34.060 | 34.060 | 0.080 | core-wait | ⚠ early-dispatch unverifiable |
| 7 | qproj_matmul | `8589934596` | 88.440 | 87.540 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | qproj_dequant_rms_nope_rope | `8589934597` | 29.220 | 29.220 | 7.040 | data-wait | 🐌 |
| 9 | idx_qr_dequant_rope | `8589934603` | 20.580 | 14.560 | 0.000 | — |  |
| 10 | qr_hadamard_matmul | `8589934605` | 5.860 | 5.860 | 3.820 | data-wait | 🐌 ⭐ |
| 11 | qr_hadamard_quant | `8589934606` | 17.640 | 17.640 | 7.920 | data-wait | 🐌 ⭐ partial rows 47/48 |
| 12 | indexer_head_coefficients | `8589934611` | 9.200 | 9.200 | 4.380 | data-wait | 🐌 ⭐ |
| 13 | indexer_score_topk_native_pair_aic | `8589934612` | 266.540 | 266.540 | 17.420 | data-wait | 🐌 |
| 14 | indexer_topk_query_merge__2 | `12884901888` | 16.060 | 16.060 | 7.500 | data-wait | 🐌 |
| 15 | csa_slots_build_valid_qk_plan | `4294967330` | 9.740 | 9.740 | 4.840 | data-wait | 🐌 ⭐ |
| 16 | qk_pv_aic | `4294967334` | 157.100 | 157.100 | 6.740 | data-wait | 🐌 ⭐ partial rows 69/72 |
| 17 | merge_norm | `4294967336` | 27.940 | 27.940 | 3.660 | data-wait | 🐌 ⭐ |
| 18 | proj_a_mm_0 | `12884901910` | 31.380 | 31.380 | 6.900 | data-wait | 🐌 |
| 19 | proj_a_mm_0 | `12884901901` | 27.840 | 26.640 | 0.000 | — |  |
| 20 | proj_a_mm_0 | `12884901889` | 29.740 | 29.320 | 0.000 | — |  |
| 21 | quant_0 | `12884901890` | 7.240 | 7.240 | 8.220 | data-wait | 🐌 ⭐ |
| 22 | _proj_b_mm_nz_kernel__2 | `12884901891` | 9.740 | 9.740 | 4.160 | data-wait | 🐌 ⭐ |
| 23 | proj_b_act_0 | `8589934614` | 18.420 | 18.420 | 1.780 | data-wait | 🐌 ⭐ |
| 24 | hc_post | `8589934615` | 23.320 | 23.320 | 4.420 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 34.500 | 38.500 | 41.100 | 42.020 | producer end→FIN 4.000 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.600 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 38.500–41.100 µs; dispatch→start 0.920 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 52.280 | 58.340 | 44.760 | 59.120 | producer end→FIN 6.060 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.360 µs (post-dispatch) |
| `4294967301` split_pre_post | 61.300 | 63.400 | 68.160 | 68.840 | producer end→FIN 2.100 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.760 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 63.400–66.920 µs; dispatch→start 0.680 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 73.500 | 77.260 | 81.340 | 82.120 | producer end→FIN 3.760 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.080 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 77.260–81.340 µs; dispatch→start 0.780 µs (post-dispatch) |
| `8589934593` qr_proj_matmul | 100.460 | 103.800 | 82.400 | 104.880 | producer end→FIN 3.340 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 22.480 µs (post-dispatch) |
| `8589934597` qproj_dequant_rms_nope_rope | 244.180 | 245.340 | 250.400 | 251.220 | producer end→FIN 1.160 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.060 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 245.340–245.700 µs; dispatch→start 0.820 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 295.000 | 298.240 | 287.540 | 298.820 | producer end→FIN 3.240 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.280 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 304.680 | 311.820 | 305.160 | 312.600 | producer end→FIN 7.140 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 7.440 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients | 330.240 | 334.020 | 316.040 | 334.620 | producer end→FIN 3.780 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.580 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_aic | 343.820 | 350.300 | 357.880 | 361.240 | producer end→FIN 6.480 µs; post-FIN ready→dispatch scheduler delay 7.580 µs; dispatch resource blocker unproven: MIX/heterogeneous launch requires cluster-aware manual inspection; dispatch→start 3.360 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge__2 | 627.780 | 629.580 | 633.980 | 635.280 | producer end→FIN 1.800 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.400 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 629.580–633.980 µs; dispatch→start 1.300 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 651.340 | 655.440 | 638.740 | 656.180 | producer end→FIN 4.100 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 17.440 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 665.920 | 672.020 | 657.260 | 672.660 | producer end→FIN 6.100 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.400 µs (post-dispatch) |
| `4294967336` merge_norm | 829.760 | 832.540 | 682.700 | 833.420 | producer end→FIN 2.780 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 150.720 µs (post-dispatch) |
| `12884901910` proj_a_mm_0 | 861.360 | 865.020 | 867.280 | 868.260 | producer end→FIN 3.660 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.260 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 865.020–867.280 µs; dispatch→start 0.980 µs (post-dispatch) |
| `12884901890` quant_0 | 955.600 | 963.160 | 917.460 | 963.820 | producer end→FIN 7.560 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 46.360 µs (post-dispatch) |
| `12884901891` _proj_b_mm_nz_kernel__2 | 971.060 | 974.660 | 965.460 | 975.220 | producer end→FIN 3.600 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 9.760 µs (post-dispatch) |
| `8589934614` proj_b_act_0 | 984.960 | 986.060 | 976.480 | 986.740 | producer end→FIN 1.100 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 10.260 µs (post-dispatch) |
| `8589934615` hc_post | 1005.160 | 1008.740 | 988.240 | 1009.580 | producer end→FIN 3.580 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 21.340 µs (post-dispatch) |

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
