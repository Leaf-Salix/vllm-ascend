> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **922.580 µs**
- Selected AICore makespan coverage: **919.140 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 922.580 | 919.140 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_coefficients_seven_20260929/h8192_b24/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **922.580 µs**
- AICore makespan: **919.140 µs**
- Static CPM cross-check: **737.480 µs**
- Primary table: **Observed critical path**, 24 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 16.100 | 16.100 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 14.680 | 14.680 | 14.220 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 2.220 | 2.220 | 3.640 | data-wait | 🐌 ⭐ |
| 3 | comb_sinkhorn | `4294967302` | 19.380 | 19.380 | 6.520 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 21.220 | 21.220 | 2.480 | core-wait | 🐌 |
| 5 | weights_proj | `8589934608` | 15.260 | 15.260 | 9.920 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | kv_score_proj_0 | `4294967318` | 24.360 | 21.760 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | kv_score_proj | `4294967314` | 70.080 | 65.780 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | idx_qr_proj_matmul | `8589934602` | 69.240 | 33.400 | 0.000 | — | ⭐ partial rows 8/24 |
| 9 | idx_qr_dequant_rope | `8589934603` | 34.460 | 34.460 | 12.240 | data-wait | 🐌 |
| 10 | qr_hadamard_matmul | `8589934605` | 7.960 | 7.960 | 1.340 | data-wait | 🐌 ⭐ |
| 11 | qr_hadamard_quant | `8589934606` | 31.280 | 31.280 | 9.400 | data-wait | 🐌 ⭐ partial rows 32/48 |
| 12 | indexer_head_coefficients_1 | `8589934611` | 13.680 | 13.680 | 2.020 | data-wait | 🐌 ⭐ |
| 13 | indexer_score_topk_native_pair_1_aic | `8589934612` | 65.620 | 65.620 | 11.040 | data-wait | 🐌 ⭐ partial rows 66/72 |
| 14 | indexer_topk_query_merge | `12884901888` | 12.080 | 12.080 | 0.740 | data-wait | ⭐ |
| 15 | csa_slots_build_valid_qk_plan | `4294967330` | 13.960 | 13.960 | 4.100 | data-wait | 🐌 ⭐ |
| 16 | qk_pv_aic | `4294967334` | 191.780 | 191.780 | 2.940 | data-wait | 🐌 ⭐ |
| 17 | merge_norm | `4294967336` | 31.940 | 31.940 | 5.340 | data-wait | 🐌 ⭐ |
| 18 | proj_a_mm_1 | `12884901889` | 32.860 | 32.860 | 9.180 | data-wait | 🐌 |
| 19 | proj_a_mm_1 | `12884901901` | 84.700 | 84.700 | 0.860 | core-wait |  |
| 20 | quant_1 | `12884901902` | 6.940 | 6.940 | 2.620 | data-wait | 🐌 ⭐ |
| 21 | _proj_b_mm_nz_kernel__3 | `12884901903` | 32.980 | 32.980 | 2.220 | data-wait | 🐌 ⭐ |
| 22 | proj_b_act_1 | `8589934614` | 18.940 | 18.940 | 1.400 | data-wait | 🐌 ⭐ |
| 23 | hc_post | `8589934615` | 23.880 | 23.880 | 4.060 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 36.300 | 45.680 | 49.580 | 50.520 | producer end→FIN 9.380 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.900 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 45.680–49.580 µs; dispatch→start 0.940 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 65.200 | 67.960 | 57.180 | 68.840 | producer end→FIN 2.760 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.660 µs (post-dispatch) |
| `4294967302` comb_sinkhorn | 71.060 | 74.600 | 76.440 | 77.580 | producer end→FIN 3.540 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 1.840 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 74.600–76.440 µs; dispatch→start 1.140 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 91.640 | 96.560 | 98.440 | 99.440 | producer end→FIN 4.920 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 1.880 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 96.560–96.940 µs; dispatch→start 1.000 µs (post-dispatch) |
| `8589934608` weights_proj | — | — | 129.640 | 130.580 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934603` idx_qr_dequant_rope | 266.780 | 273.660 | 278.160 | 279.020 | producer end→FIN 6.880 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.500 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 273.660–275.460 µs; dispatch→start 0.860 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 313.480 | 314.280 | 295.740 | 314.820 | producer end→FIN 0.800 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 19.080 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 322.780 | 331.480 | 320.140 | 332.180 | producer end→FIN 8.700 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 12.040 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients_1 | 363.460 | 364.720 | 354.100 | 365.480 | producer end→FIN 1.260 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.380 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_1_aic | 379.160 | 389.600 | 371.720 | 390.200 | producer end→FIN 10.440 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.480 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 468.640 | 472.160 | 459.360 | 472.740 | producer end→FIN 3.520 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.380 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 486.700 | 489.120 | 473.700 | 489.640 | producer end→FIN 2.420 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.940 µs (post-dispatch) |
| `4294967336` merge_norm | 681.420 | 686.000 | 494.100 | 686.760 | producer end→FIN 4.580 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 192.660 µs (post-dispatch) |
| `12884901889` proj_a_mm_1 | 718.700 | 722.120 | 726.600 | 727.880 | producer end→FIN 3.420 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.480 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 722.120–725.120 µs; dispatch→start 1.280 µs (post-dispatch) |
| `12884901902` quant_1 | 846.300 | 848.280 | 810.220 | 848.920 | producer end→FIN 1.980 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 38.700 µs (post-dispatch) |
| `12884901903` _proj_b_mm_nz_kernel__3 | 855.860 | 857.460 | 850.140 | 858.080 | producer end→FIN 1.600 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 7.940 µs (post-dispatch) |
| `8589934614` proj_b_act_1 | 891.060 | 891.780 | 859.140 | 892.460 | producer end→FIN 0.720 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 33.320 µs (post-dispatch) |
| `8589934615` hc_post | 911.400 | 914.800 | 894.440 | 915.460 | producer end→FIN 3.400 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 21.020 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901889` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967298` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967302` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967304` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `8589934603` | `4` | csa_rope_sign | early-dispatch policy | direct producer lacks `early_dispatch=true` |

> The post-FIN ready→dispatch rows above prove scheduler delay, but no named dispatch blocker was proven for those rows. Level-4 scheduler phase records do not carry ordinary task IDs; a causal task is named only when every compatible core's running and pending descriptor slots are saturated throughout the interval.

## Interpretation limits

- Dispatch→finish is the closest per-rank elapsed available in the swimlane; it is not the DFX-off benchmark.
- The critical-path makespan covers first AICore start through last AICore end, excluding host/orchestrator front time and AICPU/host tail time.
- `post-dispatch same-core` is restricted to core row(s) that realize the task-global earliest start and occupancy continuing through the row-local effective-ready→start window. It proves start contention there, not that all cores were unable to accept dispatch.
- MIX, SPMD, and sync-start capacity may require cluster-aware manual inspection; the report leaves a scheduler delay unattributed unless full compatible-engine descriptor saturation is proven.
- Level-4 collection has observer cost. Keep the unprofiled benchmark as the production performance headline.
