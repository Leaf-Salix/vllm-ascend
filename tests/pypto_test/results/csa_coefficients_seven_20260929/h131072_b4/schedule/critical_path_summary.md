> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **588.860 µs**
- Selected AICore makespan coverage: **586.620 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 588.860 | 586.620 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_coefficients_seven_20260929/h131072_b4/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **588.860 µs**
- AICore makespan: **586.620 µs**
- Static CPM cross-check: **422.620 µs**
- Primary table: **Observed critical path**, 26 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 12.780 | 12.780 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 7.460 | 7.460 | 5.920 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 1.500 | 1.500 | 4.680 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 3.500 | 3.500 | 3.800 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 16.480 | 16.480 | 2.860 | data-wait | 🐌 |
| 5 | kv_score_proj_0 | `4294967318` | 18.620 | 18.620 | 5.300 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | weights_proj | `8589934608` | 9.100 | 3.020 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | kv_score_proj | `4294967314` | 20.700 | 20.700 | 6.280 | core-wait | 🐌 ⚠ early-dispatch unverifiable |
| 8 | qproj_matmul | `8589934596` | 45.740 | 45.740 | 1.460 | core-wait | 🐌 ⚠ early-dispatch unverifiable |
| 9 | qproj_dequant_rms_nope_rope | `8589934597` | 17.800 | 17.800 | 6.500 | data-wait | 🐌 |
| 10 | idx_qr_dequant_rope | `8589934603` | 19.900 | 15.680 | 0.000 | — |  |
| 11 | qr_hadamard_matmul | `8589934605` | 4.800 | 4.800 | 7.540 | data-wait | 🐌 ⭐ |
| 12 | qr_hadamard_quant | `8589934606` | 10.560 | 10.560 | 8.060 | data-wait | 🐌 ⭐ partial rows 47/48 |
| 13 | indexer_head_coefficients | `8589934611` | 2.460 | 2.460 | 3.720 | data-wait | 🐌 ⭐ |
| 14 | indexer_score_topk_native_pair_aic | `8589934612` | 98.140 | 98.140 | 10.160 | data-wait | 🐌 ⭐ |
| 15 | indexer_topk_query_merge__2 | `12884901888` | 11.760 | 11.760 | 9.440 | data-wait | 🐌 |
| 16 | csa_slots_build_valid_qk_plan | `4294967330` | 8.100 | 8.100 | 4.800 | data-wait | 🐌 ⭐ |
| 17 | qk_pv_aic | `4294967334` | 52.020 | 52.020 | 8.040 | data-wait | 🐌 ⭐ |
| 18 | merge_norm | `4294967336` | 15.220 | 15.220 | 6.340 | data-wait | 🐌 ⭐ |
| 19 | proj_a_mm | `12884901910` | 21.080 | 21.080 | 6.020 | data-wait | 🐌 |
| 20 | proj_a_mm | `12884901907` | 22.420 | 20.940 | 0.000 | — |  |
| 21 | proj_a_mm | `12884901889` | 17.340 | 15.160 | 0.000 | — |  |
| 22 | quant | `12884901890` | 5.160 | 5.160 | 8.740 | data-wait | 🐌 ⭐ |
| 23 | _proj_b_mm_nz_kernel | `12884901891` | 10.200 | 10.200 | 2.300 | data-wait | 🐌 ⭐ partial rows 2/8 |
| 24 | proj_b_act | `8589934614` | 10.660 | 10.660 | 3.140 | data-wait | 🐌 ⭐ |
| 25 | hc_post | `8589934615` | 19.700 | 19.700 | 2.280 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 35.600 | 37.100 | 40.540 | 41.520 | producer end→FIN 1.500 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.440 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 37.100–40.540 µs; dispatch→start 0.980 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 48.980 | 52.980 | 41.560 | 53.660 | producer end→FIN 4.000 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 12.100 µs (post-dispatch) |
| `4294967301` split_pre_post | 55.160 | 56.360 | 58.140 | 58.960 | producer end→FIN 1.200 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 1.780 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 56.360–57.920 µs; dispatch→start 0.820 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 62.460 | 63.200 | 64.640 | 65.320 | producer end→FIN 0.740 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 1.440 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 63.200–64.640 µs; dispatch→start 0.680 µs (post-dispatch) |
| `4294967318` kv_score_proj_0 | — | — | 86.060 | 87.100 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `4294967314` kv_score_proj | — | — | 114.200 | 115.020 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934596` qproj_matmul | — | — | 136.260 | 137.180 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934597` qproj_dequant_rms_nope_rope | 182.920 | 184.000 | 188.520 | 189.420 | producer end→FIN 1.080 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.520 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 184.000–188.520 µs; dispatch→start 0.900 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 222.900 | 229.700 | 220.160 | 230.440 | producer end→FIN 6.800 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 10.280 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 235.240 | 242.580 | 235.120 | 243.300 | producer end→FIN 7.340 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 8.180 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients | 253.860 | 256.860 | 246.560 | 257.580 | producer end→FIN 3.000 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.020 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_aic | 260.040 | 269.300 | 265.000 | 270.200 | producer end→FIN 9.260 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 5.200 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge__2 | 368.340 | 371.580 | 376.580 | 377.780 | producer end→FIN 3.240 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.000 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 371.580–376.580 µs; dispatch→start 1.200 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 389.540 | 393.640 | 380.980 | 394.340 | producer end→FIN 4.100 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.360 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 402.440 | 409.820 | 395.460 | 410.480 | producer end→FIN 7.380 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.020 µs (post-dispatch) |
| `4294967336` merge_norm | 462.500 | 468.120 | 415.000 | 468.840 | producer end→FIN 5.620 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 53.840 µs (post-dispatch) |
| `12884901910` proj_a_mm | 484.060 | 486.440 | 489.200 | 490.080 | producer end→FIN 2.380 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.760 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 486.440–489.200 µs; dispatch→start 0.880 µs (post-dispatch) |
| `12884901890` quant | 547.260 | 555.340 | 522.500 | 556.000 | producer end→FIN 8.080 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 33.500 µs (post-dispatch) |
| `12884901891` _proj_b_mm_nz_kernel | 561.160 | 562.880 | 560.580 | 563.460 | producer end→FIN 1.720 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 2.880 µs (post-dispatch) |
| `8589934614` proj_b_act | 573.660 | 576.080 | 566.980 | 576.800 | producer end→FIN 2.420 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 9.820 µs (post-dispatch) |
| `8589934615` hc_post | 587.460 | 589.000 | 577.220 | 589.740 | producer end→FIN 1.540 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 12.520 µs (post-dispatch) |

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
