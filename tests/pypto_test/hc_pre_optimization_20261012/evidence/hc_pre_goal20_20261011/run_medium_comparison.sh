#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" baseline_merged 16 6 compare_baseline_t96
bash "$candidate_root/run_case.sh" mixed_scalar_medium_v1 16 6 compare_scalar_t96
bash "$candidate_root/run_case.sh" local_scratch_views_v2 16 6 compare_views_t96
