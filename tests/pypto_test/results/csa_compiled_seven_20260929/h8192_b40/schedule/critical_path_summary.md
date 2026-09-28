> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **1256.680 µs**
- Selected AICore makespan coverage: **1254.660 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 1256.680 | 1254.660 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929/h8192_b40/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **1256.680 µs**
- AICore makespan: **1254.660 µs**
- Static CPM cross-check: **1021.880 µs**
- Primary table: **Observed critical path**, 25 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 16.960 | 16.960 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 19.740 | 19.740 | 18.360 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 3.620 | 3.620 | 3.800 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 7.780 | 7.780 | 10.040 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 35.960 | 35.960 | 5.100 | data-wait | 🐌 |
| 5 | qr_proj_matmul_0 | `8589934593` | 34.200 | 34.200 | 2.760 | data-wait | 🐌 ⭐ |
| 6 | qr_rms_norm_quant | `8589934594` | 11.560 | 11.560 | 5.800 | data-wait | 🐌 ⭐ |
| 7 | idx_qr_proj_matmul | `8589934602` | 152.800 | 152.800 | 6.360 | data-wait | 🐌 ⭐ partial rows 5/24 |
| 8 | idx_qr_dequant_rope | `8589934603` | 41.440 | 41.440 | 5.940 | data-wait | 🐌 |
| 9 | qproj_dequant_rms_nope_rope | `8589934597` | 60.260 | 50.340 | 0.000 | — |  |
| 10 | qr_hadamard_quant | `8589934606` | 42.420 | 36.320 | 0.000 | — | ⭐ partial rows 8/48 |
| 11 | indexer_head_coefficients_1 | `8589934611` | 24.740 | 24.740 | 3.840 | data-wait | 🐌 ⭐ partial rows 32/48 |
| 12 | indexer_score_topk_native_pair_1_aic | `8589934612` | 82.700 | 82.700 | 8.420 | data-wait | 🐌 ⭐ partial rows 63/72 |
| 13 | indexer_topk_query_merge | `12884901888` | 17.380 | 17.380 | 1.740 | data-wait | 🐌 ⭐ |
| 14 | csa_slots_build_valid_qk_plan | `4294967330` | 13.780 | 13.780 | 4.180 | data-wait | 🐌 ⭐ |
| 15 | qk_pv_aic | `4294967334` | 284.060 | 284.060 | 5.560 | data-wait | 🐌 ⭐ |
| 16 | merge_norm | `4294967336` | 48.920 | 48.920 | 4.900 | data-wait | 🐌 ⭐ |
| 17 | proj_a_mm_1 | `12884901910` | 48.900 | 48.900 | 7.340 | data-wait | 🐌 |
| 18 | proj_a_mm_1 | `12884901907` | 44.740 | 41.860 | 0.000 | — |  |
| 19 | _proj_b_mm_nz_kernel__3 | `12884901903` | 23.960 | 23.960 | 0.640 | core-wait | ⭐ |
| 20 | _proj_b_mm_nz_kernel__3 | `12884901906` | 27.580 | 25.980 | 0.000 | — | ⭐ |
| 21 | _proj_b_mm_nz_kernel__3 | `12884901909` | 36.920 | 31.760 | 0.000 | — |  |
| 22 | _proj_b_mm_nz_kernel__3 | `12884901897` | 29.240 | 14.900 | 0.000 | — | ⭐ partial rows 6/8 |
| 23 | proj_b_act_1 | `8589934614` | 33.840 | 33.840 | 1.900 | data-wait | 🐌 ⭐ partial rows 48/64 |
| 24 | hc_post | `8589934615` | 51.980 | 51.980 | 2.500 | data-wait | 🐌 ⭐ partial rows 48/60 |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 42.040 | 56.180 | 59.280 | 60.400 | producer end→FIN 14.140 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.100 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 56.180–59.280 µs; dispatch→start 1.120 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 80.140 | 83.160 | 65.420 | 83.940 | producer end→FIN 3.020 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.520 µs (post-dispatch) |
| `4294967301` split_pre_post | 87.560 | 91.720 | 96.860 | 97.600 | producer end→FIN 4.160 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.140 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 91.720–96.200 µs; dispatch→start 0.740 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 105.380 | 108.180 | 109.680 | 110.480 | producer end→FIN 2.800 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 1.500 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 108.180–109.680 µs; dispatch→start 0.800 µs (post-dispatch) |
| `8589934593` qr_proj_matmul_0 | 146.440 | 148.060 | 131.040 | 149.200 | producer end→FIN 1.620 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.160 µs (post-dispatch) |
| `8589934594` qr_rms_norm_quant | 183.400 | 188.440 | 154.360 | 189.200 | producer end→FIN 5.040 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 34.840 µs (post-dispatch) |
| `8589934602` idx_qr_proj_matmul | 200.760 | 206.400 | 193.280 | 207.120 | producer end→FIN 5.640 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.840 µs (post-dispatch) |
| `8589934603` idx_qr_dequant_rope | 359.920 | 360.100 | 365.060 | 365.860 | producer end→FIN 0.180 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.960 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 360.100–365.060 µs; dispatch→start 0.800 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients_1 | 493.960 | 496.900 | 480.460 | 497.800 | producer end→FIN 2.940 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 17.340 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_1_aic | 522.540 | 530.200 | 519.500 | 530.960 | producer end→FIN 7.660 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.460 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge | 613.660 | 614.800 | 541.440 | 615.400 | producer end→FIN 1.140 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 73.960 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 632.780 | 636.240 | 618.320 | 636.960 | producer end→FIN 3.460 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.640 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 650.740 | 655.420 | 637.840 | 656.300 | producer end→FIN 4.680 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 18.460 µs (post-dispatch) |
| `4294967336` merge_norm | 940.360 | 944.460 | 658.040 | 945.260 | producer end→FIN 4.100 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 287.220 µs (post-dispatch) |
| `12884901910` proj_a_mm_1 | 994.180 | 997.640 | 1000.480 | 1001.520 | producer end→FIN 3.460 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.840 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 997.640–1000.480 µs; dispatch→start 1.040 µs (post-dispatch) |
| `8589934614` proj_b_act_1 | 1189.520 | 1190.800 | 1153.060 | 1191.420 | producer end→FIN 1.280 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 38.360 µs (post-dispatch) |
| `8589934615` hc_post | 1225.260 | 1227.100 | 1198.540 | 1227.760 | producer end→FIN 1.840 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 29.220 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901910` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967298` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967301` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967304` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `8589934603` | `4` | csa_rope_sign | early-dispatch policy | direct producer lacks `early_dispatch=true` |

> The post-FIN ready→dispatch rows above prove scheduler delay, but no named dispatch blocker was proven for those rows. Level-4 scheduler phase records do not carry ordinary task IDs; a causal task is named only when every compatible core's running and pending descriptor slots are saturated throughout the interval.

## Interpretation limits

- Dispatch→finish is the closest per-rank elapsed available in the swimlane; it is not the DFX-off benchmark.
- The critical-path makespan covers first AICore start through last AICore end, excluding host/orchestrator front time and AICPU/host tail time.
- `post-dispatch same-core` is restricted to core row(s) that realize the task-global earliest start and occupancy continuing through the row-local effective-ready→start window. It proves start contention there, not that all cores were unable to accept dispatch.
- MIX, SPMD, and sync-start capacity may require cluster-aware manual inspection; the report leaves a scheduler delay unattributed unless full compatible-engine descriptor saturation is proven.
- Level-4 collection has observer cost. Keep the unprofiled benchmark as the production performance headline.
