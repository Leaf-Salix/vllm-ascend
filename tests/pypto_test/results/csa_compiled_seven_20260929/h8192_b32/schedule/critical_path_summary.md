> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **1096.060 µs**
- Selected AICore makespan coverage: **1090.800 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 1096.060 | 1090.800 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929/h8192_b32/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **1096.060 µs**
- AICore makespan: **1090.800 µs**
- Static CPM cross-check: **891.680 µs**
- Primary table: **Observed critical path**, 25 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 15.680 | 15.680 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 18.080 | 18.080 | 13.400 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 3.140 | 3.140 | 4.440 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 6.340 | 6.340 | 14.700 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 21.460 | 21.460 | 8.340 | data-wait | 🐌 |
| 5 | qr_proj_matmul_0 | `8589934593` | 22.500 | 22.500 | 5.960 | data-wait | 🐌 ⭐ |
| 6 | kv_score_proj_0 | `4294967318` | 47.080 | 46.620 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | scatter_softmax_pool_0 | `4294967319` | 14.680 | 14.680 | 14.580 | data-wait | 🐌 |
| 8 | indexer_boundary_init | `4294967321` | 14.380 | 14.380 | 17.160 | data-wait | 🐌 |
| 9 | idx_qr_dequant_rope | `8589934603` | 43.820 | 43.820 | 7.860 | core-wait | 🐌 |
| 10 | qr_hadamard_matmul | `8589934605` | 28.960 | 28.960 | 8.300 | data-wait | 🐌 ⭐ |
| 11 | qr_hadamard_quant | `8589934606` | 70.280 | 70.280 | 10.060 | data-wait | 🐌 ⭐ partial rows 32/48 |
| 12 | indexer_head_coefficients_2 | `8589934611` | 9.380 | 9.380 | 1.200 | data-wait | 🐌 ⭐ |
| 13 | indexer_score_topk_native_pair_2_aic | `8589934612` | 90.300 | 90.300 | 8.020 | data-wait | 🐌 ⭐ partial rows 63/72 |
| 14 | indexer_topk_query_merge | `12884901888` | 16.720 | 16.720 | 1.820 | data-wait | 🐌 ⭐ |
| 15 | csa_slots_build_valid_qk_plan | `4294967330` | 13.560 | 13.560 | 5.620 | data-wait | 🐌 ⭐ |
| 16 | qk_pv_aic | `4294967334` | 233.200 | 233.200 | 5.260 | data-wait | 🐌 ⭐ |
| 17 | merge_norm | `4294967336` | 43.260 | 43.260 | 6.440 | data-wait | 🐌 ⭐ |
| 18 | proj_a_mm_1 | `12884901904` | 39.220 | 39.220 | 8.000 | data-wait | 🐌 |
| 19 | proj_a_mm_1 | `12884901907` | 44.840 | 42.920 | 0.000 | — |  |
| 20 | proj_a_mm_1 | `12884901892` | 41.900 | 40.440 | 0.000 | — |  |
| 21 | quant_1 | `12884901893` | 10.000 | 10.000 | 8.640 | data-wait | 🐌 ⭐ |
| 22 | _proj_b_mm_nz_kernel__3 | `12884901894` | 32.980 | 32.980 | 1.980 | data-wait | 🐌 ⭐ |
| 23 | proj_b_act_1 | `8589934614` | 21.100 | 21.100 | 1.300 | data-wait | 🐌 ⭐ |
| 24 | hc_post | `8589934615` | 34.680 | 34.680 | 4.020 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 44.020 | 52.500 | 56.360 | 57.420 | producer end→FIN 8.480 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.860 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 52.500–56.360 µs; dispatch→start 1.060 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 75.500 | 79.100 | 65.500 | 79.940 | producer end→FIN 3.600 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.440 µs (post-dispatch) |
| `4294967301` split_pre_post | 83.080 | 85.740 | 97.000 | 97.780 | producer end→FIN 2.660 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 11.260 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 85.740–89.020 µs; dispatch→start 0.780 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 104.120 | 108.300 | 111.680 | 112.460 | producer end→FIN 4.180 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.380 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 108.300–108.500 µs; dispatch→start 0.780 µs (post-dispatch) |
| `8589934593` qr_proj_matmul_0 | 133.920 | 138.760 | 116.100 | 139.880 | producer end→FIN 4.840 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 23.780 µs (post-dispatch) |
| `4294967319` scatter_softmax_pool_0 | 209.000 | 216.220 | 222.420 | 223.580 | producer end→FIN 7.220 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 6.200 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 216.220–222.420 µs; dispatch→start 1.160 µs (post-dispatch) |
| `4294967321` indexer_boundary_init | 238.260 | 248.060 | 254.360 | 255.420 | producer end→FIN 9.800 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 6.300 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 248.060–254.360 µs; dispatch→start 1.060 µs (post-dispatch) |
| `8589934603` idx_qr_dequant_rope | 260.060 | 267.480 | 276.540 | 277.660 | producer end→FIN 7.420 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 9.060 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 267.480–267.880 µs; dispatch→start 1.120 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 321.480 | 328.660 | 307.980 | 329.780 | producer end→FIN 7.180 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 21.800 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 358.740 | 368.140 | 334.600 | 368.800 | producer end→FIN 9.400 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 34.200 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients_2 | 439.080 | 439.620 | 399.740 | 440.280 | producer end→FIN 0.540 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 40.540 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_2_aic | 449.660 | 457.020 | 450.140 | 457.680 | producer end→FIN 7.360 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 7.540 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge | 547.980 | 549.220 | 470.500 | 549.800 | producer end→FIN 1.240 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 79.300 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 566.520 | 571.220 | 552.920 | 572.140 | producer end→FIN 4.700 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 19.220 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 585.700 | 590.160 | 573.540 | 590.960 | producer end→FIN 4.460 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 17.420 µs (post-dispatch) |
| `4294967336` merge_norm | 824.160 | 829.680 | 593.160 | 830.600 | producer end→FIN 5.520 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 237.440 µs (post-dispatch) |
| `12884901904` proj_a_mm_1 | 873.860 | 877.140 | 880.820 | 881.860 | producer end→FIN 3.280 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.680 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 877.140–879.600 µs; dispatch→start 1.040 µs (post-dispatch) |
| `12884901893` quant_1 | 1004.440 | 1012.440 | 935.640 | 1013.080 | producer end→FIN 8.000 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 77.440 µs (post-dispatch) |
| `12884901894` _proj_b_mm_nz_kernel__3 | 1023.080 | 1024.360 | 1018.200 | 1025.060 | producer end→FIN 1.280 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 6.860 µs (post-dispatch) |
| `8589934614` proj_b_act_1 | 1058.040 | 1058.620 | 1026.220 | 1059.340 | producer end→FIN 0.580 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 33.120 µs (post-dispatch) |
| `8589934615` hc_post | 1080.440 | 1083.640 | 1061.760 | 1084.460 | producer end→FIN 3.200 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 22.700 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901904` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967298` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967301` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967304` | `2` | hc_widen_rms | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967319` | `4294967318` | kv_score_proj_0 | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `4294967321` | `4294967319` | scatter_softmax_pool_0 | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `8589934603` | `4` | csa_rope_sign | early-dispatch policy | direct producer lacks `early_dispatch=true` |

> The post-FIN ready→dispatch rows above prove scheduler delay, but no named dispatch blocker was proven for those rows. Level-4 scheduler phase records do not carry ordinary task IDs; a causal task is named only when every compatible core's running and pending descriptor slots are saturated throughout the interval.

## Interpretation limits

- Dispatch→finish is the closest per-rank elapsed available in the swimlane; it is not the DFX-off benchmark.
- The critical-path makespan covers first AICore start through last AICore end, excluding host/orchestrator front time and AICPU/host tail time.
- `post-dispatch same-core` is restricted to core row(s) that realize the task-global earliest start and occupancy continuing through the row-local effective-ready→start window. It proves start contention there, not that all cores were unable to accept dispatch.
- MIX, SPMD, and sync-start capacity may require cluster-aware manual inspection; the report leaves a scheduler delay unattributed unless full compatible-engine descriptor saturation is proven.
- Level-4 collection has observer cost. Keep the unprofiled benchmark as the production performance headline.
