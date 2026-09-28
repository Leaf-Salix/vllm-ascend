> 本页只分析预先固定的window_3，未比较窗口快慢；single为同一张卡的一次重放。
> dummy前置缺少物理时戳的行已屏蔽完整ready/FIN归因；Static CPM只覆盖工具保留的有时戳依赖边。

# Operator critical-path report

- Selected rank/device: **`single`** (preselected window_3; no fastest-window selection)
- Selected operator elapsed: **967.220 µs**
- Selected AICore makespan coverage: **962.780 µs**
- Gap marker: **🐌 when gap > 1.000 µs**
- Early marker: **⭐ only when structurally eligible and dispatch precedes the last predecessor FIN**

## Rank comparison

| rank/device | dispatches | dispatch→finish elapsed µs | AICore makespan µs | selected |
|---|---:|---:|---:|---|
| `single` | 1 | 967.220 | 962.780 | ✓ |

## Dispatch 0: `decode_csa_tp1_layer`

- Artifact directory: `/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/csa_compiled_seven_20260929/h8192_b24/swimlane/dfx/window_3`
- Program identity: unambiguous direct dispatch (no dispatch_program.json)
- Dispatch→finish elapsed: **967.220 µs**
- AICore makespan: **962.780 µs**
- Static CPM cross-check: **734.060 µs**
- Primary table: **Observed critical path**, 24 tasks
- `task wall span` can overlap earlier path work; only the Observed compute contribution plus canonical stall contributions tile the AICore makespan.

### Observed path

| # | operator/task | task id | task wall span µs | Observed compute contribution µs | gap from previous µs | gap kind | markers |
|---:|---|---|---:|---:|---:|---|---|
| 0 | hc_widen_rms | `2` | 18.180 | 18.180 | — | — |  |
| 1 | hc_pre_linear | `4294967298` | 16.160 | 16.160 | 15.960 | data-wait | 🐌 |
| 2 | hc_pre_linear_reduce | `4294967300` | 2.760 | 2.760 | 3.180 | data-wait | 🐌 ⭐ |
| 3 | split_pre_post | `4294967301` | 6.220 | 6.220 | 13.700 | data-wait | 🐌 |
| 4 | mix_x_rms_norm | `4294967304` | 21.980 | 21.980 | 6.520 | data-wait | 🐌 |
| 5 | weights_proj | `8589934608` | 11.880 | 11.880 | 9.060 | data-wait | 🐌 ⚠ early-dispatch unverifiable |
| 6 | kv_score_proj_0 | `4294967318` | 28.080 | 26.660 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 7 | kv_score_proj | `4294967314` | 76.600 | 69.580 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 8 | qproj_matmul | `8589934596` | 105.840 | 90.080 | 0.000 | — | ⚠ early-dispatch unverifiable |
| 9 | qproj_dequant_rms_nope_rope | `8589934597` | 47.760 | 47.760 | 6.140 | data-wait | 🐌 |
| 10 | qr_hadamard_quant | `8589934606` | 22.040 | 10.020 | 0.000 | — | ⭐ partial rows 32/48 |
| 11 | indexer_head_coefficients_1 | `8589934611` | 10.340 | 10.340 | 7.320 | data-wait | 🐌 ⭐ |
| 12 | indexer_score_topk_native_pair_1_aic | `8589934612` | 56.900 | 56.900 | 7.340 | data-wait | 🐌 ⭐ partial rows 57/72 |
| 13 | indexer_topk_query_merge | `12884901888` | 13.980 | 13.980 | 2.060 | data-wait | 🐌 ⭐ |
| 14 | csa_slots_build_valid_qk_plan | `4294967330` | 13.420 | 13.420 | 4.580 | data-wait | 🐌 ⭐ |
| 15 | qk_pv_aic | `4294967334` | 185.120 | 185.120 | 4.400 | data-wait | 🐌 ⭐ |
| 16 | merge_norm | `4294967336` | 38.140 | 38.140 | 7.740 | data-wait | 🐌 ⭐ |
| 17 | proj_a_mm_1 | `12884901904` | 37.300 | 37.300 | 6.100 | data-wait | 🐌 |
| 18 | proj_a_mm_1 | `12884901889` | 38.680 | 36.840 | 0.000 | — |  |
| 19 | _proj_b_mm_nz_kernel__3 | `12884901912` | 31.160 | 30.200 | 0.000 | — |  |
| 20 | _proj_b_mm_nz_kernel__3 | `12884901909` | 43.480 | 41.260 | 0.000 | — | ⭐ partial rows 7/8 |
| 21 | _proj_b_mm_nz_kernel__3 | `12884901894` | 28.480 | 28.480 | 1.680 | core-wait | 🐌 |
| 22 | proj_b_act_1 | `8589934614` | 20.680 | 20.680 | 1.320 | data-wait | 🐌 ⭐ |
| 23 | hc_post | `8589934615` | 27.940 | 27.940 | 3.800 | data-wait | 🐌 ⭐ |

### Dispatch investigation for 🐌 tasks

| 🐌 task | data ready µs | last predecessor FIN µs | dispatch µs | start µs | attribution |
|---|---:|---:|---:|---:|---|
| `4294967298` hc_pre_linear | 43.120 | 52.560 | 57.700 | 59.080 | producer end→FIN 9.440 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 5.140 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 52.560–57.700 µs; dispatch→start 1.380 µs (post-dispatch) |
| `4294967300` hc_pre_linear_reduce | 75.240 | 77.520 | 66.440 | 78.420 | producer end→FIN 2.280 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 11.980 µs (post-dispatch) |
| `4294967301` split_pre_post | 81.180 | 84.380 | 94.160 | 94.880 | producer end→FIN 3.200 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 9.780 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 84.380–86.480 µs; dispatch→start 0.720 µs (post-dispatch) |
| `4294967304` mix_x_rms_norm | 101.100 | 105.080 | 106.760 | 107.620 | producer end→FIN 3.980 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 1.680 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 105.080–105.280 µs; dispatch→start 0.860 µs (post-dispatch) |
| `8589934608` weights_proj | — | — | 138.040 | 138.660 | 直接前置任务缺少物理时间戳，无法完整归因；仅保留dispatch/start观测。 |
| `8589934597` qproj_dequant_rms_nope_rope | 336.860 | 337.500 | 342.240 | 343.000 | producer end→FIN 0.640 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 4.740 µs; dispatch resource blocker unproven: full-engine saturation not proven: aiv core 24 had a free descriptor slot during 337.500–337.540 µs; dispatch→start 0.760 µs (post-dispatch) |
| `8589934611` indexer_head_coefficients_1 | 400.780 | 407.460 | 385.980 | 408.100 | producer end→FIN 6.680 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 22.120 µs (post-dispatch) |
| `8589934612` indexer_score_topk_native_pair_1_aic | 418.440 | 425.240 | 419.180 | 425.780 | producer end→FIN 6.800 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 6.600 µs (post-dispatch) |
| `12884901888` indexer_topk_query_merge | 482.680 | 483.980 | 438.400 | 484.740 | producer end→FIN 1.300 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 46.340 µs (post-dispatch) |
| `4294967330` csa_slots_build_valid_qk_plan | 498.720 | 502.480 | 487.760 | 503.300 | producer end→FIN 3.760 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 15.540 µs (post-dispatch) |
| `4294967334` qk_pv_aic | 516.720 | 520.440 | 504.300 | 521.120 | producer end→FIN 3.720 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 16.820 µs (post-dispatch) |
| `4294967336` merge_norm | 706.240 | 713.200 | 525.440 | 713.980 | producer end→FIN 6.960 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 188.540 µs (post-dispatch) |
| `12884901904` proj_a_mm_1 | 752.120 | 753.900 | 757.000 | 758.220 | producer end→FIN 1.780 µs; not early-eligible: unflagged direct producer(s); post-FIN ready→dispatch scheduler delay 3.100 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 0 had a free descriptor slot during 753.900–756.620 µs; dispatch→start 1.220 µs (post-dispatch) |
| `12884901894` _proj_b_mm_nz_kernel__3 | 880.900 | 883.100 | 885.640 | 905.500 | producer end→FIN 2.200 µs; post-FIN ready→dispatch scheduler delay 2.540 µs; dispatch resource blocker unproven: full-engine saturation not proven: aic core 1 had a free descriptor slot during 883.100–885.560 µs; dispatch→start 19.860 µs (post-dispatch) |
| `8589934614` proj_b_act_1 | 933.980 | 934.620 | 893.800 | 935.300 | producer end→FIN 0.640 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 41.500 µs (post-dispatch) |
| `8589934615` hc_post | 955.980 | 959.140 | 937.240 | 959.780 | producer end→FIN 3.160 µs; actually early-dispatched; no predecessor blocked dispatch; dispatch→start 22.540 µs (post-dispatch) |

### Named blocker evidence

| 🐌 task | blocker task | blocker operator | evidence type | evidence |
|---|---|---|---|---|
| `12884901904` | `4294967336` | merge_norm | early-dispatch policy | direct producer lacks `early_dispatch=true` |
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
