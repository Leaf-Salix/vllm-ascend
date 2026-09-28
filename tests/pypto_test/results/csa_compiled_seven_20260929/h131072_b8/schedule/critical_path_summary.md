> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **766.840 µs**
- Selected AICore makespan coverage: **763.320 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 766.840 | 763.320 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929/h131072_b8/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **766.840 µs**
- AICore makespan: **763.320 µs**
- Static CPM cross-check: **588.640 µs**
- Primary table: **Observed critical path**, 24 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 13.240 | 13.240 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 9.400 | 9.400 | 7.060 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 1.940 | 1.940 | 4.960 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 4.040 | 4.040 | 6.320 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 17.020 | 17.020 | 5.200 | data-wait | 🐌 |
| 5 | kv_score_proj_0 | `4294967318` | 23.280 | 23.280 | 8.440 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | kv_score_proj | `4294967314` | 29.240 | 29.240 | 0.840 | core-wait | ⚠ early-dispatch unverifiable |
| 7 | idx_qr_proj_matmul | `8589934602` | 75.420 | 60.220 | 0.000 | — |  |
| 8 | idx_qr_dequant_rope | `8589934603` | 20.320 | 20.320 | 6.380 | data-wait | 🐌 |
| 9 | qr_hadamard_matmul | `8589934605` | 8.560 | 8.560 | 16.280 | data-wait | 🐌 |
| 10 | qr_hadamard_quant | `8589934606` | 16.080 | 16.080 | 12.800 | data-wait | 🐌 ⭐ partial rows 45/48 |
| 11 | indexer_head_coefficients | `8589934611` | 8.540 | 8.540 | 5.340 | data-wait | 🐌 ⭐ |
| 12 | indexer_score_topk_native_pair_aic | `8589934612` | 147.920 | 147.920 | 14.880 | data-wait | 🐌 |
| 13 | indexer_topk_query_merge__2 | `12884901888` | 12.420 | 12.420 | 7.840 | data-wait | 🐌 |
| 14 | csa_slots_build_valid_qk_plan | `4294967330` | 8.400 | 8.400 | 4.120 | data-wait | 🐌 ⭐ |
| 15 | qk_pv_aic | `4294967334` | 86.140 | 86.140 | 8.440 | data-wait | 🐌 ⭐ |
| 16 | merge_norm | `4294967336` | 25.280 | 25.280 | 6.980 | data-wait | 🐌 ⭐ |
| 17 | proj_a_mm_0 | `12884901910` | 26.560 | 26.560 | 6.380 | data-wait | 🐌 |
| 18 | proj_a_mm_0 | `12884901907` | 27.400 | 25.060 | 0.000 | — |  |
| 19 | proj_a_mm_0 | `12884901892` | 28.000 | 25.620 | 0.000 | — |  |
| 20 | quant_0 | `12884901893` | 7.040 | 7.040 | 7.680 | data-wait | 🐌 ⭐ |
| 21 | _proj_b_mm_nz_kernel__2 | `12884901894` | 15.080 | 15.080 | 3.640 | data-wait | 🐌 ⭐ |
| 22 | proj_b_act_0 | `8589934614` | 14.680 | 14.680 | 0.840 | data-wait | ⭐ |
| 23 | hc_post | `8589934615` | 20.540 | 20.540 | 2.280 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 34.120 | 36.780 | 40.180 | 41.180 | producer end→FIN 2.660 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.400 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 36.780–40.180 µs; dispatch→start 1.000 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 50.580 | 54.580 | 42.400 | 55.540 | producer end→FIN 4.000 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.140 µs (post-dispatch) |
| `4294967301` split_pre_post | 57.480 | 59.140 | 63.100 | 63.800 | producer end→FIN 1.660 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.960 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 59.140–63.100 µs; dispatch→start 0.700 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 67.840 | 70.280 | 72.320 | 73.040 | producer end→FIN 2.440 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.040 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 70.280–72.320 µs; dispatch→start 0.720 µs (post-dispatch) |
| `4294967318` kv_score_proj_0 | — | — | 97.740 | 98.500 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934603` idx_qr_dequant_rope | 212.080 | 212.260 | 217.640 | 218.460 | producer end→FIN 0.180 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.380 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 212.260–212.280 µs; dispatch→start 0.820 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 238.780 | 245.480 | 254.100 | 255.060 | producer end→FIN 6.700 µs; post-FIN ready→dispatch scheduler delay 8.620 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 245.480–254.100 µs; dispatch→start 0.960 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 263.620 | 275.520 | 260.720 | 276.420 | producer end→FIN 11.900 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.700 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients | 292.500 | 297.200 | 282.560 | 297.840 | producer end→FIN 4.700 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.280 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_aic | 306.380 | 311.960 | 318.700 | 321.260 | producer end→FIN 5.580 µs; post-FIN ready→dispatch scheduler delay 6.740 µs; dispatch resource blocker unproven: MIX/heterogeneous launch requires cluster-aware manual inspection; dispatch→start 2.560 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge__2 | 469.180 | 471.000 | 475.840 | 477.020 | producer end→FIN 1.820 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.840 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 471.000–475.840 µs; dispatch→start 1.180 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 489.440 | 492.880 | 483.480 | 493.560 | producer end→FIN 3.440 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 10.080 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 501.960 | 509.560 | 494.760 | 510.400 | producer end→FIN 7.600 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.640 µs (post-dispatch) |
| `4294967336` merge_norm | 596.540 | 602.740 | 514.380 | 603.520 | producer end→FIN 6.200 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 89.140 µs (post-dispatch) |
| `12884901910` proj_a_mm_0 | 628.800 | 631.360 | 634.060 | 635.180 | producer end→FIN 2.560 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.700 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 631.360–634.060 µs; dispatch→start 1.120 µs (post-dispatch) |
| `12884901893` quant_0 | 712.420 | 719.200 | 676.460 | 720.100 | producer end→FIN 6.780 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 43.640 µs (post-dispatch) |
| `12884901894` _proj_b_mm_nz_kernel__2 | 727.140 | 730.080 | 725.140 | 730.780 | producer end→FIN 2.940 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 5.640 µs (post-dispatch) |
| `8589934615` hc_post | 761.380 | 763.020 | 748.000 | 763.660 | producer end→FIN 1.640 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.660 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901888` | `8589934612` | indexer_score_topk_native_pair_aic | early-dispatch policy | direct producer lacks `early_dispatch=true` |
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
