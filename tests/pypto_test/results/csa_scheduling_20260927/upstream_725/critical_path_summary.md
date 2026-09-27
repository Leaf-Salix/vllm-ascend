# Operator critical-path report

- Selected rank/device: **`single`** (minimum summed dispatch→finish elapsed)
- Selected operator elapsed: **779.640 µs**
- Selected AICore makespan coverage: **774.760 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 779.640 | 774.760 | ✓ |

## Dispatch 0: `CSA`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_split_optimization_20260927/schedule_c7a52af5/h8192_b16/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **779.640 µs**
- AICore makespan: **774.760 µs**
- Static CPM cross-check: **572.540 µs**
- Primary table: **Observed critical path**, 25 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen | `1` | 9.660 | 9.660 | — | — |  |
| 1 | hc_pre_linear | `4294967301` | 21.180 | 21.180 | 10.780 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967303` | 2.280 | 2.280 | 5.460 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967304` | 4.700 | 4.700 | 4.360 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967307` | 17.820 | 17.820 | 9.960 | data-wait | 🐌 |
| 5 | kv_proj_matmul | `8589934601` | 18.400 | 18.400 | 10.380 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | qr_proj_matmul | `8589934594` | 40.200 | 34.220 | 0.000 | — |  |
| 7 | kv_score_proj_0 | `4294967321` | 18.640 | 18.640 | 6.440 | core-wait | 🐌 ⚠ early-dispatch unverifiable |
| 8 | idx_qr_proj_matmul | `8589934604` | 48.380 | 45.020 | 0.000 | — | ⭐ partial rows 8/24 |
| 9 | idx_qr_dequant_rope | `8589934605` | 32.740 | 32.740 | 24.760 | data-wait | 🐌 |
| 10 | qproj_dequant_rms_nope_rope | `8589934598` | 52.840 | 49.560 | 0.000 | — |  |
| 11 | qr_hadamard_quant | `8589934608` | 24.700 | 6.980 | 0.000 | — | ⭐ partial rows 14/48 |
| 12 | indexer_head_coefficients | `8589934613` | 10.460 | 10.460 | 3.760 | data-wait | 🐌 ⭐ |
| 13 | indexer_score_topk_native_pair_aic | `8589934614` | 42.580 | 42.580 | 6.480 | data-wait | 🐌 ⭐ partial rows 30/72 |
| 14 | indexer_topk_query_merge | `12884901888` | 10.500 | 10.500 | 2.400 | data-wait | 🐌 ⭐ |
| 15 | csa_slots_build_valid_qk_plan | `4294967333` | 8.140 | 8.140 | 3.660 | data-wait | 🐌 ⭐ |
| 16 | qk_pv_aic | `4294967337` | 127.880 | 127.880 | 4.120 | data-wait | 🐌 ⭐ partial rows 54/72 |
| 17 | merge_norm | `4294967339` | 29.300 | 29.300 | 6.260 | data-wait | 🐌 ⭐ |
| 18 | proj_a_mm_0 | `12884901910` | 31.880 | 31.880 | 6.320 | data-wait | 🐌 |
| 19 | proj_a_mm_0 | `12884901907` | 32.880 | 30.320 | 0.000 | — |  |
| 20 | proj_a_mm_0 | `12884901895` | 39.120 | 35.360 | 0.000 | — |  |
| 21 | quant_0 | `12884901896` | 7.760 | 7.760 | 6.740 | data-wait | 🐌 ⭐ |
| 22 | _proj_b_mm_nz_kernel__2 | `12884901897` | 17.260 | 17.260 | 3.800 | data-wait | 🐌 ⭐ |
| 23 | proj_b_act_0 | `8589934616` | 17.900 | 17.900 | 1.320 | data-wait | 🐌 ⭐ |
| 24 | hc_post | `8589934617` | 23.240 | 23.240 | 3.980 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967301` hc_pre_linear | 32.980 | 39.500 | 42.660 | 43.760 | producer end→FIN 6.520 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.160 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 39.500–42.660 µs; dispatch→start 1.100 µs (post-dispatch) |
| `4294967303` hc_pre_linear_reduce | 64.940 | 69.600 | 58.920 | 70.400 | producer end→FIN 4.660 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.480 µs (post-dispatch) |
| `4294967304` split_pre_post | 72.680 | 74.060 | 74.100 | 77.040 | producer end→FIN 1.380 µs; dispatch→start 2.940 µs (post-dispatch) |
| `4294967307` mix_x_rms_norm | 81.740 | 84.260 | 90.920 | 91.700 | producer end→FIN 2.520 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 6.660 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 84.260–90.600 µs; dispatch→start 0.780 µs (post-dispatch) |
| `8589934601` kv_proj_matmul | 109.520 | 112.540 | 119.220 | 119.900 | producer end→FIN 3.020 µs; early-dispatch proof is unverifiable because a direct producer lacks timing; post-FIN ready→dispatch scheduler delay 6.680 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 112.540–115.600 µs; dispatch→start 0.680 µs (post-dispatch) |
| `4294967321` kv_score_proj_0 | 109.520 | 112.540 | 178.240 | 178.960 | producer end→FIN 3.020 µs; early-dispatch proof is unverifiable because a direct producer lacks timing; post-FIN ready→dispatch scheduler delay 65.700 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 112.540–115.600 µs; dispatch→start 0.720 µs (post-dispatch) |
| `8589934605` idx_qr_dequant_rope | 242.620 | 257.580 | 266.640 | 267.380 | producer end→FIN 14.960 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 9.060 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 257.580–257.820 µs; dispatch→start 0.740 µs (post-dispatch) |
| `8589934613` indexer_head_coefficients | 356.660 | 359.840 | 347.360 | 360.420 | producer end→FIN 3.180 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.060 µs (post-dispatch) |
| `8589934614` indexer_score_topk_native_pair_aic | 370.880 | 376.620 | 369.120 | 377.360 | producer end→FIN 5.740 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 8.240 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge | 419.940 | 421.780 | 389.700 | 422.340 | producer end→FIN 1.840 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 32.640 µs (post-dispatch) |
| `4294967333` csa_slots_build_valid_qk_plan | 432.840 | 435.960 | 424.860 | 436.500 | producer end→FIN 3.120 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.640 µs (post-dispatch) |
| `4294967337` qk_pv_aic | 444.640 | 448.020 | 438.140 | 448.760 | producer end→FIN 3.380 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 10.620 µs (post-dispatch) |
| `4294967339` merge_norm | 576.640 | 582.140 | 456.260 | 582.900 | producer end→FIN 5.500 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 126.640 µs (post-dispatch) |
| `12884901910` proj_a_mm_0 | 612.200 | 614.440 | 617.620 | 618.520 | producer end→FIN 2.240 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.180 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 614.440–617.620 µs; dispatch→start 0.900 µs (post-dispatch) |
| `12884901896` quant_0 | 716.080 | 722.280 | 666.140 | 722.820 | producer end→FIN 6.200 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 56.680 µs (post-dispatch) |
| `12884901897` _proj_b_mm_nz_kernel__2 | 730.580 | 733.740 | 727.980 | 734.380 | producer end→FIN 3.160 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 6.400 µs (post-dispatch) |
| `8589934616` proj_b_act_0 | 751.640 | 752.340 | 736.180 | 752.960 | producer end→FIN 0.700 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 16.780 µs (post-dispatch) |
| `8589934617` hc_post | 770.860 | 774.100 | 754.480 | 774.840 | producer end→FIN 3.240 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 20.360 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901910` | `4294967339` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967301` | `1` | hc_widen | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967307` | `1` | hc_widen | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `8589934605` | `3` | csa_rope_sign | early-dispatch policy | direct producer lacks `early_dispatch=true` |

> The post-FIN ready→dispatch rows above prove scheduler delay, but no named dispatch blocker was proven for those rows. Level-4 scheduler phase records do not carry ordinary task IDs; a causal task is named only when every compatible core's running and pending descriptor slots are saturated throughout the interval.

## Interpretation limits

- Dispatch→finish is the closest per-rank elapsed available in the swimlane; it is not the DFX-off benchmark.
- The critical-path makespan covers first AICore start through last AICore end, excluding host/orchestrator front time and AICPU/host tail time.
- `post-dispatch same-core` is restricted to core row(s) that realize the task-global earliest start and occupancy continuing through the row-local effective-ready→start window. It proves start contention there, not that all cores were unable to accept dispatch.
- MIX, SPMD, and sync-start capacity may require cluster-aware manual inspection; the report leaves a scheduler delay unattributed unless full compatible-engine descriptor saturation is proven.
- Level-4 collection has observer cost. Keep the unprofiled benchmark as the production performance headline.
