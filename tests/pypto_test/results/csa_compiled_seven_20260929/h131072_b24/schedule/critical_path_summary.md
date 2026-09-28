> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **1295.980 µs**
- Selected AICore makespan coverage: **1291.880 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 1295.980 | 1291.880 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929/h131072_b24/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **1295.980 µs**
- AICore makespan: **1291.880 µs**
- Static CPM cross-check: **1064.920 µs**
- Primary table: **Observed critical path**, 26 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 17.920 | 17.920 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 14.860 | 14.860 | 12.940 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 2.480 | 2.480 | 16.360 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 5.140 | 5.140 | 12.600 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 21.280 | 21.280 | 5.900 | data-wait | 🐌 |
| 5 | weights_proj | `8589934608` | 12.880 | 12.880 | 10.200 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | kv_score_proj_0 | `4294967318` | 26.900 | 26.900 | 0.960 | core-wait | ⚠ early-dispatch unverifiable |
| 7 | kv_score_proj | `4294967314` | 71.000 | 66.000 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | idx_qr_proj_matmul | `8589934602` | 81.680 | 50.440 | 0.000 | — | ⭐ partial rows 15/24 |
| 9 | idx_qr_dequant_rope | `8589934603` | 31.000 | 31.000 | 6.800 | data-wait | 🐌 |
| 10 | qr_hadamard_matmul | `8589934605` | 8.060 | 8.060 | 1.260 | data-wait | 🐌 ⭐ |
| 11 | qr_hadamard_quant | `8589934606` | 35.500 | 35.500 | 10.740 | data-wait | 🐌 ⭐ partial rows 16/48 |
| 12 | indexer_head_coefficients | `8589934611` | 15.920 | 15.920 | 8.400 | data-wait | 🐌 ⭐ |
| 13 | indexer_score_topk_native_pair_aic | `8589934612` | 372.580 | 372.580 | 14.140 | data-wait | 🐌 |
| 14 | indexer_topk_query_merge__2 | `12884901888` | 16.760 | 16.760 | 6.120 | data-wait | 🐌 |
| 15 | csa_slots_build_valid_qk_plan | `4294967330` | 12.800 | 12.800 | 4.100 | data-wait | 🐌 ⭐ |
| 16 | qk_pv_aic | `4294967334` | 219.780 | 219.780 | 8.240 | data-wait | 🐌 ⭐ |
| 17 | merge_norm | `4294967336` | 34.700 | 34.700 | 5.320 | data-wait | 🐌 ⭐ |
| 18 | proj_a_mm_1 | `12884901907` | 28.780 | 28.780 | 7.100 | data-wait | 🐌 |
| 19 | proj_a_mm_1 | `12884901898` | 30.620 | 27.940 | 0.000 | — |  |
| 20 | _proj_b_mm_nz_kernel__3 | `12884901912` | 49.600 | 49.600 | 0.080 | core-wait | ⭐ partial rows 1/8 |
| 21 | _proj_b_mm_nz_kernel__3 | `12884901900` | 28.180 | 0.260 | 0.000 | — | ⭐ |
| 22 | _proj_b_mm_nz_kernel__3 | `12884901903` | 30.360 | 21.300 | 0.000 | — |  |
| 23 | _proj_b_mm_nz_kernel__3 | `12884901891` | 25.900 | 17.200 | 0.000 | — |  |
| 24 | proj_b_act_1 | `8589934614` | 20.060 | 20.060 | 1.000 | data-wait | ⭐ |
| 25 | hc_post | `8589934615` | 25.600 | 25.600 | 3.880 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 37.900 | 45.460 | 49.540 | 50.840 | producer end→FIN 7.560 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.080 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 45.460–49.540 µs; dispatch→start 1.300 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 65.700 | 81.500 | 59.200 | 82.060 | producer end→FIN 15.800 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 22.860 µs (post-dispatch) |
| `4294967301` split_pre_post | 84.540 | 86.400 | 96.380 | 97.140 | producer end→FIN 1.860 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 9.980 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 86.400–88.580 µs; dispatch→start 0.760 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 102.280 | 105.440 | 107.460 | 108.180 | producer end→FIN 3.160 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.020 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 105.440–107.280 µs; dispatch→start 0.720 µs (post-dispatch) |
| `8589934608` weights_proj | — | — | 139.060 | 139.660 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934603` idx_qr_dequant_rope | 296.840 | 298.460 | 302.840 | 303.640 | producer end→FIN 1.620 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.380 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 298.460–298.480 µs; dispatch→start 0.800 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 334.640 | 335.320 | 319.940 | 335.900 | producer end→FIN 0.680 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.960 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 343.960 | 354.040 | 339.360 | 354.700 | producer end→FIN 10.080 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.340 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients | 390.200 | 397.920 | 371.500 | 398.600 | producer end→FIN 7.720 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 27.100 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_aic | 414.520 | 420.040 | 426.160 | 428.660 | producer end→FIN 5.520 µs; post-FIN ready→dispatch scheduler delay 6.120 µs; dispatch resource blocker unproven: MIX/heterogeneous launch requires cluster-aware manual inspection; dispatch→start 2.500 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge__2 | 801.240 | 802.320 | 806.380 | 807.360 | producer end→FIN 1.080 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.060 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 802.320–806.380 µs; dispatch→start 0.980 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 824.120 | 827.480 | 813.600 | 828.220 | producer end→FIN 3.360 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.620 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 841.020 | 848.620 | 829.440 | 849.260 | producer end→FIN 7.600 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 19.820 µs (post-dispatch) |
| `4294967336` merge_norm | 1069.040 | 1073.700 | 851.060 | 1074.360 | producer end→FIN 4.660 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 223.300 µs (post-dispatch) |
| `12884901907` proj_a_mm_1 | 1109.060 | 1112.360 | 1115.160 | 1116.160 | producer end→FIN 3.300 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.800 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 1112.360–1114.600 µs; dispatch→start 1.000 µs (post-dispatch) |
| `8589934615` hc_post | 1282.380 | 1285.560 | 1264.140 | 1286.260 | producer end→FIN 3.180 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 22.120 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901888` | `8589934612` | indexer_score_topk_native_pair_aic | early-dispatch policy | direct producer lacks `early_dispatch=true` |
| `12884901907` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
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
