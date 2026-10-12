#!/usr/bin/env bash
set -euo pipefail
candidate_root=/data/pyptouser/qinchuanyu/pto-eager/vllm-ascend-dsv4-pto-0251rc1/tests/pypto_test/results/hc_pre_goal20_20261011
bash "$candidate_root/run_case.sh" baseline_merged 16 5 compare_baseline_t80
bash "$candidate_root/run_case.sh" mixed_scalar_medium_v1 16 5 compare_scalar_t80
