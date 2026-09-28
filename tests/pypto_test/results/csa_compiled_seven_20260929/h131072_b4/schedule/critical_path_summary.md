> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **605.400 µs**
- Selected AICore makespan coverage: **602.540 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 605.400 | 602.540 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929/h131072_b4/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **605.400 µs**
- AICore makespan: **602.540 µs**
- Static CPM cross-check: **438.440 µs**
- Primary table: **Observed critical path**, 26 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 12.860 | 12.860 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 8.820 | 8.820 | 7.020 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 1.760 | 1.760 | 4.940 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 3.580 | 3.580 | 4.060 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 17.240 | 17.240 | 4.300 | data-wait | 🐌 |
| 5 | kv_score_proj_0 | `4294967318` | 24.200 | 24.200 | 4.560 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | weights_proj | `8589934608` | 26.760 | 14.720 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | kv_score_proj | `4294967314` | 22.520 | 11.260 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | idx_qr_proj_matmul | `8589934602` | 43.040 | 29.840 | 0.000 | — | ⭐ partial rows 3/24 |
| 9 | idx_qr_dequant_rope | `8589934603` | 16.120 | 16.120 | 12.540 | data-wait | 🐌 |
| 10 | qr_hadamard_matmul | `8589934605` | 16.060 | 16.060 | 10.140 | data-wait | 🐌 ⭐ partial rows 16/24 |
| 11 | qr_hadamard_quant | `8589934606` | 12.940 | 12.940 | 10.980 | data-wait | 🐌 ⭐ partial rows 41/48 |
| 12 | indexer_head_coefficients | `8589934611` | 12.320 | 12.320 | 7.240 | data-wait | 🐌 ⭐ partial rows 47/48 |
| 13 | indexer_score_topk_native_pair_aic | `8589934612` | 95.820 | 95.820 | 14.420 | data-wait | 🐌 ⭐ |
| 14 | indexer_topk_query_merge__2 | `12884901888` | 11.060 | 11.060 | 8.160 | data-wait | 🐌 |
| 15 | csa_slots_build_valid_qk_plan | `4294967330` | 7.740 | 7.740 | 4.640 | data-wait | 🐌 ⭐ |
| 16 | qk_pv_aic | `4294967334` | 52.380 | 52.380 | 7.600 | data-wait | 🐌 ⭐ |
| 17 | merge_norm | `4294967336` | 19.940 | 19.940 | 6.140 | data-wait | 🐌 ⭐ |
| 18 | proj_a_mm | `12884901910` | 18.120 | 18.120 | 6.840 | data-wait | 🐌 |
| 19 | proj_a_mm | `12884901907` | 20.920 | 19.520 | 0.000 | — |  |
| 20 | _proj_b_mm_nz_kernel | `12884901903` | 13.100 | 12.020 | 0.000 | — | ⭐ |
| 21 | _proj_b_mm_nz_kernel | `12884901912` | 15.520 | 15.520 | 0.420 | core-wait | ⭐ partial rows 6/8 |
| 22 | _proj_b_mm_nz_kernel | `12884901900` | 13.420 | 5.680 | 0.000 | — | ⭐ partial rows 4/8 |
| 23 | _proj_b_mm_nz_kernel | `12884901891` | 9.920 | 9.920 | 6.180 | core-wait | 🐌 |
| 24 | proj_b_act | `8589934614` | 10.440 | 10.440 | 1.260 | data-wait | 🐌 ⭐ |
| 25 | hc_post | `8589934615` | 18.880 | 18.880 | 2.340 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 36.120 | 38.460 | 42.000 | 43.140 | producer end→FIN 2.340 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.540 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 38.460–42.000 µs; dispatch→start 1.140 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 51.960 | 55.880 | 43.060 | 56.900 | producer end→FIN 3.920 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.840 µs (post-dispatch) |
| `4294967301` split_pre_post | 58.660 | 59.180 | 61.520 | 62.720 | producer end→FIN 0.520 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 2.340 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 59.180–61.520 µs; dispatch→start 1.200 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 66.300 | 68.120 | 69.780 | 70.600 | producer end→FIN 1.820 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 1.660 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 68.120–69.780 µs; dispatch→start 0.820 µs (post-dispatch) |
| `4294967318` kv_score_proj_0 | — | — | 91.540 | 92.400 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934603` idx_qr_dequant_rope | 172.420 | 179.500 | 183.900 | 184.960 | producer end→FIN 7.080 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.400 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 179.500–179.580 µs; dispatch→start 1.060 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 201.080 | 210.360 | 195.620 | 211.220 | producer end→FIN 9.280 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.600 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 227.280 | 237.560 | 222.680 | 238.260 | producer end→FIN 10.280 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.580 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients | 251.200 | 257.320 | 247.940 | 258.440 | producer end→FIN 6.120 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 10.500 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_aic | 270.760 | 284.240 | 281.260 | 285.180 | producer end→FIN 13.480 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 3.920 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge__2 | 381.000 | 383.520 | 388.180 | 389.160 | producer end→FIN 2.520 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.660 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 383.520–388.180 µs; dispatch→start 0.980 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 400.220 | 404.340 | 392.720 | 404.860 | producer end→FIN 4.120 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 12.140 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 412.600 | 419.420 | 406.120 | 420.200 | producer end→FIN 6.820 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.080 µs (post-dispatch) |
| `4294967336` merge_norm | 472.580 | 477.940 | 424.640 | 478.720 | producer end→FIN 5.360 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 54.080 µs (post-dispatch) |
| `12884901910` proj_a_mm | 498.660 | 501.100 | 504.560 | 505.500 | producer end→FIN 2.440 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.460 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 501.100–504.560 µs; dispatch→start 0.940 µs (post-dispatch) |
| `12884901891` _proj_b_mm_nz_kernel | 575.420 | 576.000 | 581.360 | 582.960 | producer end→FIN 0.580 µs; post-FIN ready→dispatch scheduler delay 5.360 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 576.000–576.640 µs; dispatch→start 1.600 µs (post-dispatch) |
| `8589934614` proj_b_act | 592.880 | 593.460 | 586.260 | 594.140 | producer end→FIN 0.580 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 7.880 µs (post-dispatch) |
| `8589934615` hc_post | 604.580 | 606.140 | 594.920 | 606.920 | producer end→FIN 1.560 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 12.000 µs (post-dispatch) |

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
