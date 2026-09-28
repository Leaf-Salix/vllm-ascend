> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **792.360 µs**
- Selected AICore makespan coverage: **787.840 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 792.360 | 787.840 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_coefficients_seven_20260929/h8192_b16/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **792.360 µs**
- AICore makespan: **787.840 µs**
- Static CPM cross-check: **578.880 µs**
- Primary table: **Observed critical path**, 25 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 14.880 | 14.880 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 18.840 | 18.840 | 12.020 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 1.860 | 1.860 | 5.160 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 5.360 | 5.360 | 7.100 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 17.380 | 17.380 | 11.400 | data-wait | 🐌 |
| 5 | qr_proj_matmul | `8589934593` | 17.800 | 17.800 | 12.340 | data-wait | 🐌 ⭐ |
| 6 | kv_score_proj_0 | `4294967318` | 36.920 | 36.320 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | idx_qr_proj_matmul | `8589934602` | 72.120 | 72.120 | 0.240 | core-wait |  |
| 8 | idx_qr_dequant_rope | `8589934603` | 22.800 | 22.800 | 7.400 | data-wait | 🐌 |
| 9 | qr_hadamard_matmul | `8589934605` | 13.120 | 13.120 | 10.460 | data-wait | 🐌 ⭐ partial rows 18/24 |
| 10 | qr_hadamard_quant | `8589934606` | 23.240 | 23.240 | 9.960 | data-wait | 🐌 ⭐ partial rows 12/48 |
| 11 | indexer_head_coefficients_2 | `8589934611` | 22.260 | 22.260 | 4.380 | data-wait | 🐌 ⭐ partial rows 7/48 |
| 12 | indexer_score_topk_native_pair_2_aic | `8589934612` | 58.040 | 58.040 | 10.820 | data-wait | 🐌 ⭐ |
| 13 | indexer_topk_query_merge | `12884901888` | 11.640 | 11.640 | 5.260 | data-wait | 🐌 ⭐ |
| 14 | csa_slots_build_valid_qk_plan | `4294967330` | 9.140 | 9.140 | 3.620 | data-wait | 🐌 ⭐ |
| 15 | qk_pv_aic | `4294967334` | 128.620 | 128.620 | 6.220 | data-wait | 🐌 ⭐ |
| 16 | merge_norm | `4294967336` | 26.300 | 26.300 | 5.160 | data-wait | 🐌 ⭐ |
| 17 | proj_a_mm_0 | `12884901910` | 32.540 | 32.540 | 7.360 | data-wait | 🐌 |
| 18 | proj_a_mm_0 | `12884901907` | 30.780 | 29.780 | 0.000 | — |  |
| 19 | _proj_b_mm_nz_kernel__2 | `12884901906` | 20.760 | 18.620 | 0.000 | — | ⭐ |
| 20 | _proj_b_mm_nz_kernel__2 | `12884901903` | 26.040 | 23.500 | 0.000 | — |  |
| 21 | _proj_b_mm_nz_kernel__2 | `12884901897` | 19.440 | 6.520 | 0.000 | — | ⭐ partial rows 1/8 |
| 22 | _proj_b_mm_nz_kernel__2 | `12884901894` | 13.460 | 13.460 | 0.400 | core-wait | ⭐ |
| 23 | proj_b_act_0 | `8589934614` | 16.740 | 16.740 | 1.160 | data-wait | 🐌 ⭐ |
| 24 | hc_post | `8589934615` | 22.600 | 22.600 | 3.900 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 42.040 | 48.500 | 53.120 | 54.060 | producer end→FIN 6.460 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.620 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 48.500–53.120 µs; dispatch→start 0.940 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 72.900 | 77.420 | 66.660 | 78.060 | producer end→FIN 4.520 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.400 µs (post-dispatch) |
| `4294967301` split_pre_post | 79.920 | 81.980 | 86.140 | 87.020 | producer end→FIN 2.060 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.160 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 81.980–85.480 µs; dispatch→start 0.880 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 92.380 | 96.740 | 102.760 | 103.780 | producer end→FIN 4.360 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 6.020 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 96.740–100.540 µs; dispatch→start 1.020 µs (post-dispatch) |
| `8589934593` qr_proj_matmul | 121.160 | 131.640 | 104.000 | 133.500 | producer end→FIN 10.480 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 29.500 µs (post-dispatch) |
| `8589934603` idx_qr_dequant_rope | 259.980 | 261.420 | 266.480 | 267.380 | producer end→FIN 1.440 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.060 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 261.420–262.160 µs; dispatch→start 0.900 µs (post-dispatch) |
| `8589934605` qr_hadamard_matmul | 290.180 | 299.980 | 275.440 | 300.640 | producer end→FIN 9.800 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 25.200 µs (post-dispatch) |
| `8589934606` qr_hadamard_quant | 313.760 | 322.960 | 317.920 | 323.720 | producer end→FIN 9.200 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 5.800 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients_2 | 346.960 | 350.740 | 340.400 | 351.340 | producer end→FIN 3.780 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 10.940 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_2_aic | 373.600 | 383.820 | 370.680 | 384.420 | producer end→FIN 10.220 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 13.740 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge | 442.460 | 446.920 | 388.360 | 447.720 | producer end→FIN 4.460 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 59.360 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 459.360 | 462.440 | 450.000 | 462.980 | producer end→FIN 3.080 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 12.980 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 472.120 | 477.680 | 464.180 | 478.340 | producer end→FIN 5.560 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 14.160 µs (post-dispatch) |
| `4294967336` merge_norm | 606.960 | 611.360 | 482.820 | 612.120 | producer end→FIN 4.400 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 129.300 µs (post-dispatch) |
| `12884901910` proj_a_mm_0 | 638.420 | 641.660 | 644.820 | 645.780 | producer end→FIN 3.240 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.160 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 641.660–644.480 µs; dispatch→start 0.960 µs (post-dispatch) |
| `8589934614` proj_b_act_0 | 770.600 | 771.180 | 758.820 | 771.760 | producer end→FIN 0.580 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 12.940 µs (post-dispatch) |
| `8589934615` hc_post | 788.500 | 791.560 | 772.660 | 792.400 | producer end→FIN 3.060 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 19.740 µs (post-dispatch) |

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
